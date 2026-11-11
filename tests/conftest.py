import sys  # para agregar src al camino de búsqueda
from pathlib import Path  # rutas que sirven en cualquier sistema operativo

import pytest  # herramienta de tests

ROOT = Path(__file__).resolve().parents[1]  # carpeta raíz del repositorio
sys.path.insert(0, str(ROOT / "src"))  # permite importar genuscnn sin instalarlo

from genuscnn.generate import generate_dataset  # generador del proyecto


@pytest.fixture(scope="session")  # se crea una sola vez para todos los tests
def tiny_datasets(tmp_path_factory):  # datasets pequeños para probar rápido
    base = tmp_path_factory.mktemp("data")  # carpeta temporal
    roots = {}  # carpeta de cada variante
    for variant in ("cadena", "conexa"):  # las dos variantes
        roots[variant] = base / variant / "topology_dataset"  # carpeta de la variante
        generate_dataset(roots[variant], variant, families=5, samples=2, points=300, log=lambda *_: None)  # 5 familias de 2 muestras
    return roots  # devuelve las carpetas
