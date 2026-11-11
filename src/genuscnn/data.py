import json  # para leer los metadatos
from pathlib import Path  # rutas que sirven en cualquier sistema operativo

import numpy as np  # arreglos numéricos
import pandas as pd  # tablas
import torch  # tensores

from .geometry import random_rotation  # rotación 3d uniforme al azar

N_CLASSES = 6  # géneros 0 a 5
ORIGINAL_FAMILIES = 5  # las familias 00 a 04 son las del zip original


def build_manifest(data_dir):  # arma la tabla con una fila por superficie
    records = []  # filas de la tabla
    for genus_dir in sorted(Path(data_dir).glob("genus_*")):  # recorre las carpetas de género
        for meta_path in sorted((genus_dir / "metadata").glob("*.json")):  # recorre sus metadatos
            meta = json.loads(meta_path.read_text())  # lee el metadata
            records.append(dict(  # agrega la fila
                sample_id=meta_path.stem,  # nombre de la superficie
                genus=meta["genus"],  # género (la etiqueta)
                family=meta["family"],  # número de familia
                group=f"g{meta['genus']}_family{meta['family']:02d}",  # identificador de la familia
                sample=meta["sample"],  # número de muestra dentro de la familia
                origin=meta.get("origin", "original"),  # las del zip original no traen este campo
                file=str(genus_dir / "pointclouds" / f"{meta_path.stem}.npy"),  # ruta de la nube
            ))  # fin de la fila
    if not records:  # si no encontró nada
        raise FileNotFoundError(f"no hay superficies en {data_dir}; corre scripts/generate_dataset.py")  # avisa qué hacer
    return pd.DataFrame(records)  # tabla con todas las superficies


def split_by_family(manifest, n_train=25, n_val=5, seed=2026):  # reparte familias completas entre train, val y test
    rng = np.random.default_rng(seed)  # generador con la semilla del split
    group_split = {}  # a qué split va cada familia
    for genus in sorted(manifest["genus"].unique()):  # recorre los géneros
        groups = sorted(manifest.loc[manifest["genus"] == genus, "group"].unique())  # familias del género, ordenadas
        groups = list(rng.permutation(groups))  # las baraja con la semilla
        for group in groups[:n_train]:  # las primeras van a train
            group_split[group] = "train"  # asigna train
        for group in groups[n_train:n_train + n_val]:  # las siguientes van a validation
            group_split[group] = "val"  # asigna validation
        for group in groups[n_train + n_val:]:  # las demás van a test
            group_split[group] = "test"  # asigna test
    manifest = manifest.copy()  # copia para no modificar la tabla original
    manifest["split"] = manifest["group"].map(group_split)  # agrega la columna split
    assert manifest.groupby("group")["split"].nunique().max() == 1  # ninguna familia queda en dos splits
    return manifest  # tabla con el split


def load_clouds(manifest):  # carga todas las nubes en memoria
    return {row.sample_id: np.load(row.file).astype(np.float32) for row in manifest.itertuples()}  # diccionario nombre -> puntos


def render(points, rotation, size=64):  # convierte una nube 3d en una imagen 2d
    centered = points - points.mean(axis=0)  # centra la nube en el origen
    scaled = centered / np.linalg.norm(centered, axis=1).max()  # el punto más lejano queda a distancia 1
    rotated = scaled @ rotation.T  # aplica la rotación
    pixels = ((rotated[:, :2] + 1) / 2 * (size - 1)).round().astype(int)  # pasa x, y de [-1, 1] a índices de píxel
    depth = (rotated[:, 2] + 1) / 2  # profundidad de 0 (lejos) a 1 (cerca)
    image = np.zeros((size, size), dtype=np.float32)  # imagen vacía
    np.maximum.at(image, (size - 1 - pixels[:, 1], pixels[:, 0]), 0.25 + 0.75 * depth)  # cada píxel se queda con el punto más cercano
    return image  # devuelve la imagen


class TrainDataset(torch.utils.data.Dataset):  # dataset que genera una rotación nueva cada vez

    def __init__(self, frame, clouds, seed, size=64):  # recibe las filas del split, las nubes y una semilla
        self.ids = list(frame["sample_id"])  # nombres de las superficies
        self.labels = list(frame["genus"])  # sus etiquetas
        self.clouds = clouds  # diccionario con las nubes
        self.rng = np.random.default_rng(seed)  # generador propio para las rotaciones
        self.size = size  # lado de la imagen

    def __len__(self):  # tamaño del dataset
        return len(self.ids)  # número de superficies

    def __getitem__(self, index):  # devuelve una imagen y su etiqueta
        image = render(self.clouds[self.ids[index]], random_rotation(self.rng), self.size)  # imagen con una rotación al azar
        return torch.from_numpy(image)[None], int(self.labels[index])  # tensor con un canal: forma (1, 64, 64)


def fixed_views(frame, clouds, views, seed, size=64):  # genera las imágenes fijas de un split
    rng = np.random.default_rng(seed)  # generador con semilla fija
    images, labels, owners = [], [], []  # imágenes, etiquetas y a qué superficie pertenece cada vista
    for position, row in enumerate(frame.itertuples()):  # recorre las superficies
        for _ in range(views):  # una o varias vistas por superficie
            images.append(render(clouds[row.sample_id], random_rotation(rng), size))  # imagen con una rotación al azar
            labels.append(row.genus)  # su etiqueta
            owners.append(position)  # número de la superficie dentro del split
    return torch.from_numpy(np.stack(images))[:, None], torch.tensor(labels), torch.tensor(owners)  # tensores listos para la red


def ink(images):  # proporción de píxeles pintados de cada imagen
    return (images > 0).float().mean(dim=(1, 2, 3)).numpy()  # fracción de píxeles distintos de cero
