import json  # para escribir los metadatos
import zipfile  # para leer el zip original si existe
from pathlib import Path  # rutas que sirven en cualquier sistema operativo

import numpy as np  # arreglos numéricos

from . import geometry as G  # funciones de geometría del proyecto

SEED_GLOBAL = 20261105  # semilla base, la misma del cuaderno de ampliación
ORIGINAL_FAMILIES = 5  # familias que trae el zip original (00 a 04)
VARIANTS = ("cadena", "conexa")  # las dos formas de construir una superficie de género g


def sample_name(g, family, sample):  # nombre base de una muestra
    return f"g{g}_family{family:02d}_sample{sample:03d}"  # mismo formato que el original


def build_family(variant, g, family, max_tries=20):  # construye la malla base de una familia
    if variant == "cadena":  # método original: g toros sueltos
        params = G.chain_params(g, SEED_GLOBAL + g * 1000 + family)  # misma semilla que el cuaderno de ampliación
        V, F = G.chain_mesh(g, params)  # malla en cadena
        return V, F, G.mesh_topology(V, F)  # malla y su topología medida
    for attempt in range(max_tries):  # variante conexa: reintenta hasta que la topología sea la correcta
        params = G.connected_params(g, SEED_GLOBAL + 7_000_000 + g * 1000 + family + attempt * 100_000)  # semilla del intento
        V, F = G.connected_mesh(g, params)  # malla conexa
        topology = G.mesh_topology(V, F)  # topología medida
        if topology["components"] == 1 and topology["closed"] and topology["holes"] == g:  # una pieza cerrada de género g
            return V, F, topology  # la acepta
    raise RuntimeError(f"no se logró una superficie conexa de género {g} para la familia {family}")  # no debería pasar


def extract_original(original_zip, root):  # extrae las 300 superficies originales
    with zipfile.ZipFile(original_zip, "r") as zf:  # abre el zip
        members = [  # archivos útiles del zip
            n for n in zf.namelist()  # recorre los nombres
            if n.startswith("topology_dataset/") and "__MACOSX" not in n and ".DS_Store" not in n  # sin basura de macos
        ]  # fin de la lista
        zf.extractall(root.parent, members)  # extrae en la carpeta que contiene a topology_dataset


def generate_dataset(root, variant, families=35, samples=10, points=3000, original_zip=None, log=print):  # genera un dataset completo
    if variant not in VARIANTS:  # revisa que la variante exista
        raise ValueError(f"variante desconocida: {variant}")  # avisa el error
    root = Path(root)  # carpeta topology_dataset de esta variante
    first_family = 0  # por defecto se generan todas las familias
    if variant == "cadena" and original_zip is not None and Path(original_zip).exists():  # si está el zip original
        if not (root / "genus_0").exists():  # y todavía no se extrajo
            extract_original(original_zip, root)  # extrae las 300 originales sin tocarlas
        first_family = ORIGINAL_FAMILIES  # y solo genera de la familia 05 en adelante
    created = 0  # cuenta las muestras nuevas
    for g in range(6):  # recorre los géneros 0 a 5
        folder = root / f"genus_{g}"  # carpeta del género
        for sub in ("pointclouds", "metadata"):  # las dos subcarpetas que usa la red
            (folder / sub).mkdir(parents=True, exist_ok=True)  # las crea si faltan
        for family in range(first_family, families):  # recorre las familias a generar
            names = [sample_name(g, family, s) for s in range(samples)]  # nombres de las muestras de la familia
            if all((folder / "metadata" / f"{n}.json").exists() for n in names):  # si la familia ya está completa
                continue  # no la repite
            V0, F, topology = build_family(variant, g, family)  # malla base y topología de la familia
            for sample, name in enumerate(names):  # recorre las muestras
                seed = SEED_GLOBAL + g * 100000 + family * 100 + sample  # semilla de la muestra, como en el cuaderno
                if variant == "conexa":  # la variante conexa usa otras semillas
                    seed += 7_000_000  # para que no repita las deformaciones de la cadena
                rng = np.random.default_rng(seed)  # generador de la muestra
                V = G.deform(V0, rng)  # deforma la familia
                X = G.sample_surface(V, F, points, rng)  # muestrea la nube de puntos
                np.save(folder / "pointclouds" / f"{name}.npy", X)  # guarda la nube
                meta = {  # metadata de la muestra
                    "genus": g,  # etiqueta
                    "euler_characteristic": topology["euler"],  # característica de euler medida en la malla
                    "n_components": topology["components"],  # piezas separadas medidas en la malla
                    "family": family,  # número de familia
                    "sample": sample,  # número de muestra
                    "surface_type": G.SURFACE_TYPE[g],  # tipo de superficie
                    "n_points": int(len(X)),  # puntos de la nube
                    "mesh_vertices": int(len(V)),  # vértices de la malla
                    "mesh_faces": int(len(F)),  # caras de la malla
                    "deformations": {"rotation": True, "anisotropic_scaling": True, "smooth_nonlinear": True, "noise": False},  # deformaciones
                    "origin": "generada",  # marca las muestras hechas por este código
                    "variant": variant,  # cadena o conexa
                    "seed": seed,  # semilla de la muestra
                }  # fin del metadata
                (folder / "metadata" / f"{name}.json").write_text(json.dumps(meta, indent=4))  # guarda el metadata
                created += 1  # suma una muestra nueva
        log(f"género {g} listo ({variant})")  # progreso
    return created  # cuántas muestras se crearon
