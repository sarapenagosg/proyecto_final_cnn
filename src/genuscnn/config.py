from pathlib import Path  # rutas que sirven en cualquier sistema operativo

import yaml  # lector de archivos yaml

DEFAULTS = {  # valores por defecto: son los de la red original de los cuadernos
    "name": "base",  # nombre de la corrida
    "variant": "cadena",  # dataset: cadena (método original) o conexa (género real)
    "image_size": 64,  # lado de la imagen en píxeles
    "split_seed": 2026,  # semilla del split por familia
    "seed": 11,  # semilla del entrenamiento
    "n_train_families": 25,  # familias de cada género que van a train
    "n_val_families": 5,  # familias de cada género que van a validation
    "val_views": 1,  # vistas fijas por superficie en validation
    "test_views": 3,  # vistas fijas por superficie en test
    "batch_size": 64,  # tamaño del batch
    "epochs": 50,  # número de épocas
    "lr": 1e-3,  # tasa de aprendizaje máxima
    "weight_decay": 0.01,  # decaimiento de pesos de adamw
    "schedule": "constant",  # constant (tasa fija) o cosine (calentamiento y bajada en coseno)
    "warmup_epochs": 3,  # épocas de calentamiento cuando schedule es cosine
    "ema_decay": 0.0,  # promedio móvil de los pesos; 0 lo apaga
    "label_smoothing": 0.0,  # suavizado de etiquetas; 0 lo apaga
    "grad_clip": 0.0,  # tope de la norma del gradiente; 0 lo apaga
    "norm": "batch",  # batch o group
    "bn_momentum": 0.1,  # qué tan rápido batchnorm olvida sus estadísticas; más bajo = más memoria
    "dropout": 0.3,  # proporción de neuronas apagadas en la cabeza
    "backprop": True,  # false apaga backpropagation: sirve solo como control, la red no aprende
}  # fin de los valores por defecto


def load_config(path=None, **overrides):  # lee un yaml y lo completa con los valores por defecto
    cfg = dict(DEFAULTS)  # empieza con los valores por defecto
    if path is not None:  # si se pasó un archivo
        cfg.update(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})  # pisa los valores con los del yaml
    cfg.update({k: v for k, v in overrides.items() if v is not None})  # pisa con los argumentos dados a mano
    unknown = set(cfg) - set(DEFAULTS)  # claves que no existen
    if unknown:  # si hay alguna
        raise KeyError(f"claves desconocidas en la configuración: {sorted(unknown)}")  # avisa para no ignorar un error de tipeo
    cfg["lr"] = float(cfg["lr"])  # yaml puede leer 1e-3 como texto, así que se convierte
    return cfg  # devuelve la configuración
