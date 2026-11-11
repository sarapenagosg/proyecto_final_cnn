import argparse  # lee los argumentos de la línea de comandos
import sys  # para agregar src al camino de búsqueda
from pathlib import Path  # rutas que sirven en cualquier sistema operativo

ROOT = Path(__file__).resolve().parents[1]  # carpeta raíz del repositorio
sys.path.insert(0, str(ROOT / "src"))  # permite importar genuscnn sin instalarlo

from genuscnn.generate import VARIANTS, generate_dataset  # generador del proyecto

parser = argparse.ArgumentParser(description="genera los datasets de superficies")  # descripción del script
parser.add_argument("--variant", choices=VARIANTS + ("todas",), default="todas")  # cuál dataset generar
parser.add_argument("--families", type=int, default=35)  # familias por género
parser.add_argument("--samples", type=int, default=10)  # muestras por familia
parser.add_argument("--points", type=int, default=3000)  # puntos por nube
parser.add_argument("--original-zip", default=str(ROOT / "data" / "topology_dataset.zip"))  # zip original, si se tiene
args = parser.parse_args()  # lee los argumentos

variants = VARIANTS if args.variant == "todas" else (args.variant,)  # lista de variantes a generar
for variant in variants:  # recorre las variantes
    root = ROOT / "data" / variant / "topology_dataset"  # carpeta de salida de la variante
    created = generate_dataset(root, variant, args.families, args.samples, args.points, args.original_zip)  # genera
    print(f"{variant}: {created} superficies nuevas en {root}")  # resumen
