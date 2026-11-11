import sys  # para agregar src al camino de búsqueda
from pathlib import Path  # rutas que sirven en cualquier sistema operativo

import pandas as pd  # tablas

ROOT = Path(__file__).resolve().parents[1]  # carpeta raíz del repositorio
sys.path.insert(0, str(ROOT / "src"))  # permite importar genuscnn sin instalarlo

from genuscnn import data as D  # datos del proyecto
from genuscnn.baselines import fit_predict, shape_features  # líneas base de forma
from genuscnn.config import load_config  # lector de configuraciones
from genuscnn.metrics import classification_summary  # métricas
from genuscnn.training import ink_baseline, prepare_data  # línea base de la tinta, splits y vistas fijas

cfg = load_config(ROOT / "configs" / "suave.yaml")  # misma configuración de datos que la red final
rows = []  # filas de la tabla
for variant in ("cadena", "conexa"):  # recorre los dos datasets
    data = prepare_data(cfg, ROOT / "data" / variant / "topology_dataset")  # splits y vistas fijas
    train_images, train_labels, _ = D.fixed_views(data["frames"]["train"], data["clouds"], 1, cfg["seed"], cfg["image_size"])  # una vista fija de train, como en la línea base de la tinta
    test_images, test_labels, _ = data["test"]  # las mismas imágenes de test que usa la red
    train_x, test_x = shape_features(train_images.numpy()), shape_features(test_images.numpy())  # rasgos de la silueta
    ink_only = classification_summary(test_labels.numpy(), ink_baseline(train_images, train_labels, test_images))  # solo la tinta, igual que en el entrenamiento
    shape = classification_summary(test_labels.numpy(), fit_predict(train_x, train_labels.numpy(), test_x))  # tinta, largo, ancho y alargamiento
    rows.append(dict(dataset=variant, azar=round(1 / 6, 3), tinta=round(ink_only["accuracy"], 3), silueta=round(shape["accuracy"], 3)))  # fila del dataset
out_dir = ROOT / "results" / "baselines"  # carpeta de salida
out_dir.mkdir(parents=True, exist_ok=True)  # la crea si falta
table = pd.DataFrame(rows)  # tabla resumen
table.to_csv(out_dir / "summary.csv", index=False)  # la guarda
print(table.to_string(index=False))  # la muestra
