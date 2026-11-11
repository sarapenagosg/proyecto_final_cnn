import numpy as np  # arreglos numéricos
import torch  # tensores

from genuscnn import data as D  # módulo que se prueba
from genuscnn.geometry import random_rotation  # rotación al azar


def small_split(root):  # manifest pequeño con su split
    manifest = D.build_manifest(root)  # tabla de superficies
    return D.split_by_family(manifest, n_train=3, n_val=1, seed=2026)  # 3 familias a train, 1 a validation, 1 a test


def test_manifest_has_one_row_per_surface(tiny_datasets):  # una fila por superficie
    manifest = D.build_manifest(tiny_datasets["cadena"])  # tabla
    assert len(manifest) == 6 * 5 * 2  # 6 géneros x 5 familias x 2 muestras
    assert sorted(manifest["genus"].unique()) == list(range(6))  # géneros 0 a 5


def test_split_has_no_family_leakage(tiny_datasets):  # ninguna familia aparece en dos splits
    manifest = small_split(tiny_datasets["conexa"])  # tabla con split
    assert manifest.groupby("group")["split"].nunique().max() == 1  # cada familia en un solo split
    families = manifest.groupby(["genus", "split"])["group"].nunique().unstack()  # familias por género y split
    assert (families["train"] == 3).all()  # 3 de train por género
    assert (families["val"] == 1).all()  # 1 de validation por género
    assert (families["test"] == 1).all()  # 1 de test por género


def test_split_is_reproducible(tiny_datasets):  # misma semilla, mismo split
    first = small_split(tiny_datasets["conexa"])  # primer split
    second = small_split(tiny_datasets["conexa"])  # segundo split
    assert (first["split"] == second["split"]).all()  # iguales


def test_render_shape_and_range():  # la imagen tiene el tamaño y los valores esperados
    rng = np.random.default_rng(0)  # generador
    points = rng.normal(size=(500, 3)).astype(np.float32)  # nube de prueba
    image = D.render(points, random_rotation(rng), size=64)  # imagen
    assert image.shape == (64, 64)  # 64 x 64
    assert image.min() >= 0 and image.max() <= 1  # valores entre 0 y 1
    assert (image > 0).sum() > 0  # algo quedó pintado


def test_render_ignores_size_and_position():  # escalar o mover la nube no cambia la imagen
    rng = np.random.default_rng(1)  # generador
    points = rng.normal(size=(500, 3))  # nube de prueba
    rotation = random_rotation(rng)  # una rotación fija
    assert np.allclose(D.render(points, rotation), D.render(points * 7.5 + 3.0, rotation))  # misma imagen


def test_fixed_views_are_fixed(tiny_datasets):  # las vistas fijas no cambian entre llamadas
    manifest = small_split(tiny_datasets["cadena"])  # tabla con split
    clouds = D.load_clouds(manifest)  # nubes
    frame = manifest[manifest["split"] == "val"]  # validation
    a, labels, owners = D.fixed_views(frame, clouds, views=2, seed=7)  # primera llamada
    b, _, _ = D.fixed_views(frame, clouds, views=2, seed=7)  # segunda llamada
    assert torch.equal(a, b)  # mismas imágenes
    assert a.shape == (len(frame) * 2, 1, 64, 64)  # 2 vistas por superficie
    assert owners.max().item() == len(frame) - 1  # cada vista sabe de qué superficie es
    assert labels.tolist() == np.repeat(frame["genus"].to_numpy(), 2).tolist()  # etiquetas repetidas por vista


def test_train_dataset_gives_new_views(tiny_datasets):  # train entrega una vista distinta cada vez
    manifest = small_split(tiny_datasets["cadena"])  # tabla con split
    frame = manifest[manifest["split"] == "train"]  # train
    dataset = D.TrainDataset(frame, D.load_clouds(manifest), seed=3)  # dataset de train
    first, label = dataset[0]  # primera vista
    second, _ = dataset[0]  # segunda vista de la misma superficie
    assert first.shape == (1, 64, 64)  # un canal de 64 x 64
    assert label == int(frame["genus"].iloc[0])  # etiqueta correcta
    assert not torch.equal(first, second)  # las vistas son distintas


def test_ink_counts_painted_pixels():  # la tinta es la fracción de píxeles pintados
    images = torch.zeros(2, 1, 4, 4)  # dos imágenes vacías
    images[1, 0, :2] = 1.0  # la segunda tiene la mitad pintada
    assert D.ink(images).tolist() == [0.0, 0.5]  # 0 y 0.5
