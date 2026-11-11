import json  # para leer los metadatos

import numpy as np  # arreglos numéricos
import pytest  # herramienta de tests

from genuscnn.generate import generate_dataset, sample_name  # funciones que se prueban


def test_sample_name_matches_original_format():  # el nombre sigue el formato del dataset original
    assert sample_name(3, 7, 2) == "g3_family07_sample002"  # género, familia y muestra


@pytest.mark.parametrize("variant", ["cadena", "conexa"])  # repite el test para las dos variantes
def test_dataset_structure(tiny_datasets, variant):  # carpetas, cantidades y formas
    root = tiny_datasets[variant]  # carpeta de la variante
    for g in range(6):  # recorre los géneros
        clouds = sorted((root / f"genus_{g}" / "pointclouds").glob("*.npy"))  # nubes del género
        metas = sorted((root / f"genus_{g}" / "metadata").glob("*.json"))  # metadatos del género
        assert len(clouds) == len(metas) == 10  # 5 familias x 2 muestras
        X = np.load(clouds[0])  # una nube
        assert X.shape == (300, 3)  # 300 puntos 3d
        assert np.isfinite(X).all()  # sin nan ni infinitos


def test_metadata_matches_the_measured_topology(tiny_datasets):  # el metadata dice la topología real
    for g in range(6):  # recorre los géneros
        name = sample_name(g, 0, 0)  # primera muestra del género
        connected = json.loads((tiny_datasets["conexa"] / f"genus_{g}" / "metadata" / f"{name}.json").read_text())  # metadata conexa
        chain = json.loads((tiny_datasets["cadena"] / f"genus_{g}" / "metadata" / f"{name}.json").read_text())  # metadata en cadena
        assert connected["euler_characteristic"] == 2 - 2 * g  # conexa: 2 - 2g
        assert connected["n_components"] == 1  # conexa: una pieza
        assert chain["n_components"] == max(g, 1)  # cadena: g piezas


def test_generation_is_reproducible(tiny_datasets, tmp_path):  # generar dos veces da exactamente lo mismo
    root = tmp_path / "again" / "topology_dataset"  # carpeta nueva
    generate_dataset(root, "conexa", families=1, samples=1, points=300, log=lambda *_: None)  # una sola familia
    name = sample_name(4, 0, 0)  # una muestra de género 4
    first = np.load(tiny_datasets["conexa"] / "genus_4" / "pointclouds" / f"{name}.npy")  # la del fixture
    second = np.load(root / "genus_4" / "pointclouds" / f"{name}.npy")  # la nueva
    assert np.array_equal(first, second)  # idénticas


def test_generation_skips_existing_families(tiny_datasets):  # volver a correr no duplica nada
    created = generate_dataset(tiny_datasets["cadena"], "cadena", families=5, samples=2, points=300, log=lambda *_: None)  # corre otra vez
    assert created == 0  # no crea nada nuevo


def test_unknown_variant_raises(tmp_path):  # una variante mal escrita debe fallar
    with pytest.raises(ValueError):  # espera el error
        generate_dataset(tmp_path, "otra")  # variante que no existe
