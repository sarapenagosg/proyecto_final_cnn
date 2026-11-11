import copy  # para clonar la red del promedio móvil
import json  # para guardar la configuración y las métricas
import math  # coseno del scheduler
import time  # para medir cuánto tarda cada época
from pathlib import Path  # rutas que sirven en cualquier sistema operativo

import numpy as np  # arreglos numéricos
import pandas as pd  # tablas
import torch  # redes neuronales
from sklearn.linear_model import LogisticRegression  # modelo de la línea base de la tinta
from torch import nn  # capas y pérdidas

from . import data as D  # datos del proyecto
from .metrics import classification_summary, roughness, vote_by_surface  # métricas del proyecto
from .model import GenusCNN, count_parameters  # la red


def set_seed(seed):  # fija las semillas para que la corrida se pueda repetir
    np.random.seed(seed)  # semilla de numpy
    torch.manual_seed(seed)  # semilla de torch


def lr_at(step, total_steps, cfg, steps_per_epoch):  # tasa de aprendizaje que toca en un paso
    if cfg["schedule"] == "constant":  # tasa fija, como en la red original
        return cfg["lr"]  # siempre la misma
    warmup = cfg["warmup_epochs"] * steps_per_epoch  # pasos de calentamiento
    if step < warmup:  # durante el calentamiento
        return cfg["lr"] * (step + 1) / warmup  # sube en línea recta hasta la tasa máxima
    progress = (step - warmup) / max(total_steps - warmup, 1)  # avance de 0 a 1 después del calentamiento
    return cfg["lr"] * 0.5 * (1 + math.cos(math.pi * progress))  # baja en forma de coseno hasta cero


class WeightAverage:  # promedio móvil exponencial de los pesos (ema)

    def __init__(self, model, decay):  # recibe la red y qué tanto recuerda el promedio
        self.model = copy.deepcopy(model)  # copia de la red que guarda el promedio
        self.decay = decay  # 0.99 significa que recuerda más o menos los últimos 100 pasos
        self.steps = 0  # pasos acumulados
        for p in self.model.parameters():  # recorre los pesos de la copia
            p.requires_grad_(False)  # la copia no se entrena

    @torch.no_grad()  # actualizar el promedio no necesita gradientes
    def update(self, model):  # mezcla los pesos nuevos en el promedio
        self.steps += 1  # cuenta el paso
        decay = min(self.decay, (1 + self.steps) / (10 + self.steps))  # al principio recuerda poco para arrancar rápido
        for avg, new in zip(self.model.parameters(), model.parameters()):  # recorre los pesos
            avg.mul_(decay).add_(new.detach(), alpha=1 - decay)  # promedio = decay * promedio + (1 - decay) * nuevo
        for avg, new in zip(self.model.buffers(), model.buffers()):  # recorre las estadísticas de batchnorm
            if avg.dtype.is_floating_point:  # medias y varianzas
                avg.mul_(decay).add_(new, alpha=1 - decay)  # también se promedian, para que correspondan a los pesos promediados
            else:  # contadores enteros
                avg.copy_(new)  # esos se copian tal cual


@torch.no_grad()  # evaluar no necesita gradientes
def predict(model, images, device, batch_size=256):  # salidas de la red para un tensor de imágenes
    model.eval()  # modo evaluación: apaga dropout y usa las estadísticas fijas de batchnorm
    outputs = []  # salidas por batch
    for start in range(0, len(images), batch_size):  # recorre las imágenes por bloques
        outputs.append(model(images[start:start + batch_size].to(device)).cpu())  # salidas del bloque
    return torch.cat(outputs)  # junta todo


def evaluate(model, images, labels, device):  # pérdida, accuracy y salidas sobre imágenes fijas
    logits = predict(model, images, device)  # salidas de la red
    loss = nn.functional.cross_entropy(logits, labels).item()  # entropía cruzada promedio
    accuracy = (logits.argmax(1) == labels).float().mean().item()  # proporción de aciertos
    return loss, accuracy, logits  # devuelve las tres cosas


def ink_baseline(train_images, train_labels, eval_images):  # línea base que solo mira la tinta
    model = LogisticRegression(max_iter=1000)  # regresión logística
    model.fit(D.ink(train_images)[:, None], train_labels.numpy())  # la tinta es el único rasgo
    return model.predict(D.ink(eval_images)[:, None])  # predicciones para las imágenes pedidas


def prepare_data(cfg, data_dir):  # arma todo lo que el entrenamiento necesita de los datos
    manifest = D.build_manifest(data_dir)  # tabla de superficies
    manifest = D.split_by_family(manifest, cfg["n_train_families"], cfg["n_val_families"], cfg["split_seed"])  # split por familia
    clouds = D.load_clouds(manifest)  # nubes en memoria
    frames = {s: manifest[manifest["split"] == s] for s in ("train", "val", "test")}  # filas de cada split
    size = cfg["image_size"]  # lado de la imagen
    val = D.fixed_views(frames["val"], clouds, cfg["val_views"], cfg["split_seed"], size)  # vistas fijas de validation
    test = D.fixed_views(frames["test"], clouds, cfg["test_views"], cfg["split_seed"] + 1, size)  # vistas fijas de test
    return {"manifest": manifest, "clouds": clouds, "frames": frames, "val": val, "test": test}  # paquete de datos


def train(cfg, data_dir, out_dir, log=print):  # entrena una red y guarda todo en out_dir
    out_dir = Path(out_dir)  # carpeta de salida
    out_dir.mkdir(parents=True, exist_ok=True)  # la crea si falta
    set_seed(cfg["seed"])  # fija las semillas
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # usa la gpu si hay
    data = prepare_data(cfg, data_dir)  # datos listos
    val_images, val_labels, _ = data["val"]  # imágenes fijas de validation
    train_set = D.TrainDataset(data["frames"]["train"], data["clouds"], cfg["seed"], cfg["image_size"])  # train con rotaciones nuevas
    loader = torch.utils.data.DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True)  # baraja train en cada época
    model = GenusCNN(D.N_CLASSES, cfg["norm"], cfg["dropout"], cfg["bn_momentum"]).to(device)  # crea la red
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])  # optimizador
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])  # pérdida, con suavizado de etiquetas opcional
    average = WeightAverage(model, cfg["ema_decay"]) if cfg["ema_decay"] > 0 else None  # promedio móvil opcional
    total_steps = cfg["epochs"] * len(loader)  # pasos totales del entrenamiento
    step = 0  # paso actual
    history = []  # historial por época
    best_val_loss = float("inf")  # mejor pérdida de validation hasta ahora
    for epoch in range(1, cfg["epochs"] + 1):  # recorre las épocas
        start = time.time()  # hora de inicio
        model.train()  # modo entrenamiento
        train_loss, train_correct, seen = 0.0, 0, 0  # acumuladores de train
        for x, y in loader:  # recorre los batches
            x, y = x.to(device), y.to(device)  # pasa al dispositivo
            lr = lr_at(step, total_steps, cfg, len(loader))  # tasa de aprendizaje de este paso
            for group in optimizer.param_groups:  # recorre los grupos del optimizador
                group["lr"] = lr  # les pone la tasa de este paso
            optimizer.zero_grad()  # borra los gradientes anteriores
            logits = model(x)  # paso hacia adelante
            loss = loss_fn(logits, y)  # pérdida
            if cfg["backprop"]:  # entrenamiento normal
                loss.backward()  # backpropagation: calcula los gradientes
                if cfg["grad_clip"] > 0:  # si se pidió recortar el gradiente
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])  # limita su norma
                optimizer.step()  # actualiza los pesos (sin backpropagation no hay gradientes y este paso no existe)
            if average is not None:  # si hay promedio móvil
                average.update(model)  # lo actualiza con los pesos nuevos
            step += 1  # avanza un paso
            train_loss += loss.item() * len(y)  # suma la pérdida del batch
            train_correct += (logits.argmax(1) == y).sum().item()  # suma los aciertos
            seen += len(y)  # suma las imágenes vistas
        judged = average.model if average is not None else model  # red que se evalúa: el promedio si existe
        val_loss, val_acc, _ = evaluate(judged, val_images, val_labels, device)  # evalúa en validation
        is_best = val_loss < best_val_loss  # ¿es la mejor época?
        if is_best:  # si lo es
            best_val_loss = val_loss  # actualiza la mejor pérdida
            torch.save(judged.state_dict(), out_dir / "best.pt")  # guarda los pesos
        history.append(dict(  # fila del historial
            epoch=epoch,  # número de época
            lr=lr,  # tasa de aprendizaje al final de la época
            train_loss=train_loss / seen,  # pérdida media de train
            train_acc=train_correct / seen,  # accuracy de train
            val_loss=val_loss,  # pérdida de validation
            val_acc=val_acc,  # accuracy de validation
            best=int(is_best),  # marca la mejor época
        ))  # fin de la fila
        log(f"época {epoch:3d}  train_acc {train_correct / seen:.3f}  val_acc {val_acc:.3f}  "  # progreso
            f"val_loss {val_loss:.3f}  {'*' if is_best else ' '}  {time.time() - start:.0f} s")  # fin del mensaje
    history = pd.DataFrame(history)  # historial como tabla
    history.to_csv(out_dir / "history.csv", index=False)  # lo guarda
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))  # guarda la configuración usada
    model.load_state_dict(torch.load(out_dir / "best.pt", weights_only=True))  # carga la mejor época
    metrics = final_metrics(model, cfg, data, history, device)  # métricas finales
    metrics["parameters"] = count_parameters(model)  # número de parámetros
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))  # guarda las métricas
    return history, metrics  # devuelve historial y métricas


def final_metrics(model, cfg, data, history, device):  # métricas de la mejor época en validation y test
    val_images, val_labels, _ = data["val"]  # validation
    test_images, test_labels, test_owners = data["test"]  # test
    train_images, train_labels, _ = D.fixed_views(data["frames"]["train"], data["clouds"], 1, cfg["seed"], cfg["image_size"])  # train fijo para la tinta
    _, _, val_logits = evaluate(model, val_images, val_labels, device)  # salidas en validation
    _, _, test_logits = evaluate(model, test_images, test_labels, device)  # salidas en test
    vote_pred, vote_true = vote_by_surface(test_logits.numpy(), test_labels.numpy(), test_owners.numpy())  # voto de las vistas de cada superficie
    best = history.loc[history["val_loss"].idxmin()]  # fila de la mejor época
    return {  # diccionario de métricas
        "best_epoch": int(best.epoch),  # mejor época
        "train_acc_at_best": float(best.train_acc),  # accuracy de train en esa época
        "roughness_val_acc": roughness(history["val_acc"]),  # qué tan puntuda es la accuracy de validation
        "roughness_val_loss": roughness(history["val_loss"]),  # qué tan puntuda es la pérdida de validation
        "worst_val_acc_after_10": float(history["val_acc"][10:].min()),  # peor caída de validation después de la época 10
        "val": classification_summary(val_labels.numpy(), val_logits.argmax(1).numpy()),  # validation
        "test": classification_summary(test_labels.numpy(), test_logits.argmax(1).numpy()),  # test por imagen
        "test_vote": classification_summary(vote_true, vote_pred),  # test por superficie, juntando sus vistas
        "ink_val": classification_summary(val_labels.numpy(), ink_baseline(train_images, train_labels, val_images)),  # tinta en validation
        "ink_test": classification_summary(test_labels.numpy(), ink_baseline(train_images, train_labels, test_images)),  # tinta en test
        "chance": 1 / D.N_CLASSES,  # azar
    }  # fin de las métricas


def evaluate_checkpoint(checkpoint, cfg, data_dir):  # evalúa una red ya entrenada sobre el test de otro dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # usa la gpu si hay
    data = prepare_data(cfg, data_dir)  # datos del dataset pedido, con el mismo split por familia
    model = GenusCNN(D.N_CLASSES, cfg["norm"], cfg["dropout"], cfg["bn_momentum"]).to(device)  # misma arquitectura
    model.load_state_dict(torch.load(checkpoint, weights_only=True))  # carga los pesos
    test_images, test_labels, test_owners = data["test"]  # test
    _, _, logits = evaluate(model, test_images, test_labels, device)  # salidas en test
    vote_pred, vote_true = vote_by_surface(logits.numpy(), test_labels.numpy(), test_owners.numpy())  # voto por superficie
    return {  # métricas
        "test": classification_summary(test_labels.numpy(), logits.argmax(1).numpy()),  # por imagen
        "test_vote": classification_summary(vote_true, vote_pred),  # por superficie
    }  # fin de las métricas
