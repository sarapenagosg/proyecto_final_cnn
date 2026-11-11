import json  # para leer las métricas guardadas

import numpy as np  # arreglos numéricos
import pytest  # herramienta de tests
import torch  # tensores

from genuscnn.config import DEFAULTS, load_config  # configuración
from genuscnn.metrics import classification_summary, roughness, vote_by_surface  # métricas
from genuscnn.model import GenusCNN  # la red
from genuscnn.training import WeightAverage, evaluate_checkpoint, lr_at, train  # entrenamiento


def test_constant_schedule_never_changes():  # con tasa fija el valor no cambia
    cfg = load_config(None, schedule="constant", lr=1e-3)  # configuración
    assert {lr_at(s, 100, cfg, 10) for s in range(100)} == {1e-3}  # un solo valor en todos los pasos


def test_cosine_schedule_warms_up_and_decays():  # el coseno sube, llega al máximo y baja
    cfg = load_config(None, schedule="cosine", lr=1e-3, warmup_epochs=2)  # 2 épocas de calentamiento
    values = [lr_at(s, 200, cfg, 10) for s in range(200)]  # tasa en cada paso
    assert values[0] < values[10] < values[19]  # sube durante el calentamiento
    assert max(values) == pytest.approx(1e-3)  # el máximo es la tasa pedida
    assert all(a >= b for a, b in zip(values[20:], values[21:]))  # después nunca vuelve a subir
    assert values[-1] < 1e-5  # termina casi en cero


def test_weight_average_moves_slowly():  # el promedio móvil queda entre los pesos viejos y los nuevos
    model = GenusCNN()  # red
    average = WeightAverage(model, decay=0.9)  # promedio
    old = next(average.model.parameters()).clone()  # pesos antes
    with torch.no_grad():  # sin gradientes
        for p in model.parameters():  # recorre los pesos
            p.add_(1.0)  # mueve todos los pesos una unidad
    average.update(model)  # actualiza el promedio
    moved = (next(average.model.parameters()) - old).mean().item()  # cuánto se movió el promedio
    assert 0 < moved < 1  # se movió, pero menos que los pesos


def test_roughness():  # la rugosidad mide los saltos entre épocas
    assert roughness([0.5] * 20) == 0.0  # una curva plana no tiene saltos
    assert roughness([0, 1] * 10, skip=0) == pytest.approx(1.0)  # una curva que salta de 0 a 1 tiene rugosidad 1
    assert roughness(np.linspace(0, 1, 11), skip=0) == pytest.approx(0.1)  # una recta tiene saltos constantes


def test_classification_summary():  # el resumen cuenta bien aciertos y errores
    summary = classification_summary([0, 1, 2, 3], [0, 1, 3, 5])  # dos aciertos y dos errores
    assert summary["accuracy"] == 0.5  # mitad de aciertos
    assert summary["errors"] == 2  # dos errores
    assert summary["errors_distance_1"] == 1  # solo uno es por un género
    assert summary["mean_distance"] == pytest.approx(0.75)  # (0 + 0 + 1 + 2) / 4
    assert summary["bias"] == pytest.approx(0.75)  # los dos errores predicen géneros de más


def test_vote_by_surface():  # el voto promedia las vistas de cada superficie
    logits = np.array([[2.0, 0.0], [0.0, 1.0], [0.0, 3.0], [0.0, 1.0]])  # 4 vistas y 2 clases
    preds, labels = vote_by_surface(logits, [0, 0, 1, 1], [0, 0, 1, 1])  # 2 superficies con 2 vistas
    assert preds.tolist() == [0, 1]  # la primera gana por la vista más segura
    assert labels.tolist() == [0, 1]  # etiquetas por superficie


def test_config_rejects_unknown_keys():  # una clave mal escrita debe fallar
    with pytest.raises(KeyError):  # espera el error
        load_config(None, epocas=3)  # la clave correcta es epochs


def test_yaml_configs_load():  # los yaml del repositorio son válidos
    from pathlib import Path  # rutas
    for path in sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml")):  # recorre los yaml
        assert set(load_config(path)) == set(DEFAULTS)  # mismas claves que los valores por defecto


def test_training_end_to_end(tiny_datasets, tmp_path):  # una corrida corta completa
    cfg = load_config(None, epochs=2, n_train_families=3, n_val_families=1, batch_size=16, schedule="cosine", warmup_epochs=1, ema_decay=0.9)  # corrida pequeña
    history, metrics = train(cfg, tiny_datasets["cadena"], tmp_path / "run", log=lambda *_: None)  # entrena
    assert len(history) == 2  # dos épocas
    assert np.isfinite(history["val_loss"]).all()  # pérdidas finitas
    assert (tmp_path / "run" / "best.pt").exists()  # guardó el mejor modelo
    saved = json.loads((tmp_path / "run" / "metrics.json").read_text())  # métricas guardadas
    assert 0 <= saved["test"]["accuracy"] <= 1  # accuracy válida
    crossed = evaluate_checkpoint(tmp_path / "run" / "best.pt", cfg, tiny_datasets["conexa"])  # evalúa en el otro dataset
    assert np.array(crossed["test"]["confusion"]).sum() == 6 * 2 * 3  # 6 géneros x 2 muestras x 3 vistas


def test_without_backprop_the_weights_never_change(tiny_datasets, tmp_path):  # sin backpropagation la red no aprende
    cfg = load_config(None, epochs=1, n_train_families=3, n_val_families=1, batch_size=16, backprop=False)  # corrida de control
    torch.manual_seed(cfg["seed"])  # misma semilla que usa el entrenamiento
    initial = GenusCNN().state_dict()  # pesos iniciales que tendrá la red
    train(cfg, tiny_datasets["cadena"], tmp_path / "control", log=lambda *_: None)  # corre sin backpropagation
    final = torch.load(tmp_path / "control" / "best.pt", weights_only=True)  # pesos al terminar
    assert all(torch.equal(initial[k], final[k]) for k in initial if k.endswith("weight") and initial[k].dim() > 1)  # convoluciones y capas lineales intactas
