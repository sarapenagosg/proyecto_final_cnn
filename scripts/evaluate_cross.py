import argparse  # lee los argumentos de la línea de comandos
import json  # para leer y guardar métricas
import sys  # para agregar src al camino de búsqueda
from pathlib import Path  # rutas que sirven en cualquier sistema operativo

import pandas as pd  # tablas

ROOT = Path(__file__).resolve().parents[1]  # carpeta raíz del repositorio
sys.path.insert(0, str(ROOT / "src"))  # permite importar genuscnn sin instalarlo

from genuscnn.training import evaluate_checkpoint  # evalúa una red guardada en otro dataset

parser = argparse.ArgumentParser(description="evalúa cada red en el test de los dos datasets")  # descripción del script
DEFAULT_RUNS = [f"suave_{v}{s}" for s in ("", "_s12", "_s13") for v in ("cadena", "conexa")]  # las seis redes finales
parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)  # corridas a evaluar
args = parser.parse_args()  # lee los argumentos

rows, details = [], {}  # filas de la tabla y métricas completas
for run in args.runs:  # recorre las corridas
    run_dir = ROOT / "results" / "runs" / run  # carpeta de la corrida
    cfg = json.loads((run_dir / "config.json").read_text())  # configuración con la que se entrenó
    for variant in ("cadena", "conexa"):  # la evalúa en los dos datasets
        metrics = evaluate_checkpoint(run_dir / "best.pt", cfg, ROOT / "data" / variant / "topology_dataset")  # métricas en ese test
        details[f"{run}->{variant}"] = metrics  # guarda las métricas completas
        rows.append(dict(  # fila de la tabla
            corrida=run,  # nombre de la corrida
            entrenada_en=cfg["variant"],  # dataset de entrenamiento
            evaluada_en=variant,  # dataset de evaluación
            test_acc=round(metrics["test"]["accuracy"], 3),  # accuracy por imagen
            test_acc_voto=round(metrics["test_vote"]["accuracy"], 3),  # accuracy por superficie
            distancia_media=round(metrics["test"]["mean_distance"], 3),  # qué tan lejos quedan los errores
            sesgo=round(metrics["test"]["bias"], 3),  # positivo si predice géneros de más, negativo si predice de menos
        ))  # fin de la fila
out_dir = ROOT / "results" / "cross"  # carpeta de salida
out_dir.mkdir(parents=True, exist_ok=True)  # la crea si falta
table = pd.DataFrame(rows)  # tabla resumen
table.to_csv(out_dir / "summary.csv", index=False)  # la guarda
(out_dir / "details.json").write_text(json.dumps(details, indent=2))  # guarda las métricas completas
print(table.to_string(index=False))  # la muestra
