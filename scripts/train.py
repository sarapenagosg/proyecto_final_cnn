import argparse  # lee los argumentos de la línea de comandos
import sys  # para agregar src al camino de búsqueda
from pathlib import Path  # rutas que sirven en cualquier sistema operativo

ROOT = Path(__file__).resolve().parents[1]  # carpeta raíz del repositorio
sys.path.insert(0, str(ROOT / "src"))  # permite importar genuscnn sin instalarlo

from genuscnn.config import load_config  # lector de configuraciones
from genuscnn.training import train  # entrenamiento

parser = argparse.ArgumentParser(description="entrena la cnn que clasifica el género")  # descripción del script
parser.add_argument("--config", default=str(ROOT / "configs" / "suave.yaml"))  # archivo de configuración
parser.add_argument("--variant", default=None)  # cambia el dataset sin editar el yaml
parser.add_argument("--epochs", type=int, default=None)  # cambia las épocas sin editar el yaml
parser.add_argument("--seed", type=int, default=None)  # cambia la semilla sin editar el yaml
parser.add_argument("--name", default=None)  # nombre de la carpeta de resultados
args = parser.parse_args()  # lee los argumentos

cfg = load_config(args.config, variant=args.variant, epochs=args.epochs, seed=args.seed, name=args.name)  # configuración final
data_dir = ROOT / "data" / cfg["variant"] / "topology_dataset"  # carpeta del dataset
out_dir = ROOT / "results" / "runs" / cfg["name"]  # carpeta de resultados de la corrida
history, metrics = train(cfg, data_dir, out_dir)  # entrena
print(f"mejor época {metrics['best_epoch']}")  # mejor época
print(f"validation accuracy {metrics['val']['accuracy']:.3f}")  # accuracy en validation
print(f"test accuracy por imagen {metrics['test']['accuracy']:.3f}")  # accuracy en test por imagen
print(f"test accuracy por superficie {metrics['test_vote']['accuracy']:.3f}")  # accuracy en test juntando vistas
print(f"línea base de la tinta en test {metrics['ink_test']['accuracy']:.3f}")  # tinta
print(f"rugosidad de la accuracy de validation {metrics['roughness_val_acc']:.3f}")  # qué tan puntuda quedó la curva
