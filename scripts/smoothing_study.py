import argparse  # lee los argumentos de la línea de comandos
import json  # para leer las métricas de cada corrida
import sys  # para agregar src al camino de búsqueda
from pathlib import Path  # rutas que sirven en cualquier sistema operativo

import pandas as pd  # tablas

ROOT = Path(__file__).resolve().parents[1]  # carpeta raíz del repositorio
sys.path.insert(0, str(ROOT / "src"))  # permite importar genuscnn sin instalarlo

from genuscnn.config import load_config  # lector de configuraciones
from genuscnn.training import train  # entrenamiento

EXPERIMENTS = {  # cada experimento cambia una sola cosa frente a la red base, salvo los combinados
    "base": {},  # la red del avance, sin cambios
    "sin_backprop": {"backprop": False},  # control: sin backpropagation los pesos nunca cambian
    "dropout_05": {"dropout": 0.5},  # más dropout
    "lr_baja": {"lr": 3e-4},  # tasa de aprendizaje fija más pequeña
    "val_3_vistas": {"val_views": 3},  # validation con 3 vistas por superficie
    "coseno": {"schedule": "cosine"},  # calentamiento y bajada de la tasa en coseno
    "ema": {"ema_decay": 0.99},  # promedio móvil de los pesos
    "groupnorm": {"norm": "group"},  # groupnorm en lugar de batchnorm
    "bn_lenta": {"bn_momentum": 0.01},  # batchnorm con más memoria en sus estadísticas
    "coseno_ema": {"schedule": "cosine", "ema_decay": 0.99},  # scheduler y promedio móvil juntos
    "suave": {"schedule": "cosine", "ema_decay": 0.99, "val_views": 3, "label_smoothing": 0.05, "grad_clip": 1.0},  # receta completa
    "suave_100": {"schedule": "cosine", "ema_decay": 0.99, "val_views": 3, "label_smoothing": 0.05, "grad_clip": 1.0, "epochs": 100},  # receta completa con más épocas
}  # fin de los experimentos

parser = argparse.ArgumentParser(description="estudio para suavizar las curvas de validation")  # descripción del script
parser.add_argument("--only", nargs="*", default=None)  # corre solo los experimentos nombrados
parser.add_argument("--variant", default="cadena")  # dataset sobre el que se hace el estudio
args = parser.parse_args()  # lee los argumentos

names = args.only if args.only else list(EXPERIMENTS)  # experimentos a correr
for name in names:  # recorre los experimentos
    out_dir = ROOT / "results" / "smoothing" / name  # carpeta de resultados del experimento
    if (out_dir / "metrics.json").exists():  # si ya se corrió
        continue  # no lo repite
    cfg = load_config(None, name=name, variant=args.variant, **EXPERIMENTS[name])  # red base más el cambio del experimento
    print(f"=== {name}")  # título del experimento
    train(cfg, ROOT / "data" / cfg["variant"] / "topology_dataset", out_dir)  # entrena

rows = []  # filas de la tabla resumen
for name in EXPERIMENTS:  # recorre todos los experimentos
    path = ROOT / "results" / "smoothing" / name / "metrics.json"  # métricas del experimento
    if not path.exists():  # si todavía no se corrió
        continue  # lo salta
    m = json.loads(path.read_text())  # lee las métricas
    rows.append(dict(  # fila del experimento
        experimento=name,  # nombre
        rugosidad_val_acc=round(m["roughness_val_acc"], 3),  # salto medio de la accuracy de validation entre épocas
        rugosidad_val_loss=round(m["roughness_val_loss"], 3),  # salto medio de la pérdida de validation entre épocas
        peor_val_acc=round(m["worst_val_acc_after_10"], 3),  # peor caída después de la época 10
        mejor_epoca=m["best_epoch"],  # época del mejor checkpoint
        val_acc=round(m["val"]["accuracy"], 3),  # accuracy en validation
        test_acc=round(m["test"]["accuracy"], 3),  # accuracy en test por imagen
        test_acc_voto=round(m["test_vote"]["accuracy"], 3),  # accuracy en test por superficie
    ))  # fin de la fila
table = pd.DataFrame(rows)  # tabla resumen
table.to_csv(ROOT / "results" / "smoothing" / "summary.csv", index=False)  # la guarda
print(table.to_string(index=False))  # la muestra
