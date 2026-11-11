import argparse  # lee los argumentos de la línea de comandos
import sys  # para agregar src al camino de búsqueda
from pathlib import Path  # rutas que sirven en cualquier sistema operativo

ROOT = Path(__file__).resolve().parents[1]  # carpeta raíz del repositorio
sys.path.insert(0, str(ROOT / "src"))  # permite importar genuscnn sin instalarlo

from genuscnn.config import load_config  # lector de configuraciones
from genuscnn.training import train  # entrenamiento

RUNS = {  # corridas finales: nombre -> (configuración, dataset, semilla)
    "suave_cadena": ("suave.yaml", "cadena", 11),  # receta final en el dataset original
    "suave_conexa": ("suave.yaml", "conexa", 11),  # receta final en el dataset corregido
    "base_conexa": ("base.yaml", "conexa", 11),  # red del avance en el dataset corregido, para comparar
    "suave_cadena_s12": ("suave.yaml", "cadena", 12),  # segunda semilla en cadena
    "suave_conexa_s12": ("suave.yaml", "conexa", 12),  # segunda semilla en conexa
    "suave_cadena_s13": ("suave.yaml", "cadena", 13),  # tercera semilla en cadena
    "suave_conexa_s13": ("suave.yaml", "conexa", 13),  # tercera semilla en conexa
}  # fin de las corridas

parser = argparse.ArgumentParser(description="corridas finales del proyecto")  # descripción del script
parser.add_argument("--only", nargs="*", default=None)  # corre solo las corridas nombradas
args = parser.parse_args()  # lee los argumentos

for name in (args.only if args.only else list(RUNS)):  # recorre las corridas pedidas
    config_file, variant, seed = RUNS[name]  # datos de la corrida
    out_dir = ROOT / "results" / "runs" / name  # carpeta de resultados
    if (out_dir / "metrics.json").exists():  # si ya se corrió
        continue  # no la repite
    cfg = load_config(ROOT / "configs" / config_file, name=name, variant=variant, seed=seed)  # configuración de la corrida
    print(f"=== {name}")  # título de la corrida
    _, metrics = train(cfg, ROOT / "data" / variant / "topology_dataset", out_dir)  # entrena
    print(f"test por imagen {metrics['test']['accuracy']:.3f}, por superficie {metrics['test_vote']['accuracy']:.3f}")  # resumen
