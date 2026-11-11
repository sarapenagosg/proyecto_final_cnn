import matplotlib.pyplot as plt  # gráficos
import numpy as np  # arreglos numéricos

from .data import render  # convierte una nube en imagen
from .geometry import random_rotation  # rotación al azar

BLUE, ORANGE, GRAY = "#1f77b4", "#e6550d", "#7f7f7f"  # colores de train, validation y referencias


def plot_history(history, title=None, chance=1 / 6):  # curvas de pérdida y accuracy de una corrida
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))  # dos paneles
    axes[0].plot(history["epoch"], history["train_loss"], color=BLUE, label="train")  # pérdida de train
    axes[0].plot(history["epoch"], history["val_loss"], color=ORANGE, label="validation")  # pérdida de validation
    axes[0].set_title("pérdida")  # título del panel
    axes[1].plot(history["epoch"], history["train_acc"], color=BLUE, label="train")  # accuracy de train
    axes[1].plot(history["epoch"], history["val_acc"], color=ORANGE, label="validation")  # accuracy de validation
    axes[1].axhline(chance, color=GRAY, ls="--", label="azar")  # línea del azar
    axes[1].set_title("accuracy")  # título del panel
    axes[1].set_ylim(0, 1)  # la accuracy va de 0 a 1
    for ax in axes:  # recorre los paneles
        ax.set_xlabel("época")  # nombre del eje x
        ax.legend(frameon=False)  # leyenda sin marco
        ax.spines[["top", "right"]].set_visible(False)  # quita los bordes de arriba y de la derecha
    if title:  # si se pasó un título
        fig.suptitle(title)  # lo pone arriba
    fig.tight_layout()  # ajusta los espacios
    return fig  # devuelve la figura


def plot_before_after(before, after, labels=("antes", "después")):  # compara validation de dos corridas
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex="col", sharey="row")  # fila de pérdida y fila de accuracy
    for column, (history, label) in enumerate(zip((before, after), labels)):  # una columna por corrida
        axes[0, column].plot(history["epoch"], history["train_loss"], color=BLUE, label="train")  # pérdida de train
        axes[0, column].plot(history["epoch"], history["val_loss"], color=ORANGE, label="validation")  # pérdida de validation
        axes[1, column].plot(history["epoch"], history["train_acc"], color=BLUE, label="train")  # accuracy de train
        axes[1, column].plot(history["epoch"], history["val_acc"], color=ORANGE, label="validation")  # accuracy de validation
        axes[1, column].axhline(1 / 6, color=GRAY, ls="--", label="azar")  # línea del azar
        axes[0, column].set_title(label)  # título de la columna
        axes[1, column].set_xlabel("época")  # nombre del eje x
    axes[0, 0].set_ylabel("pérdida")  # nombre de la fila de arriba
    axes[1, 0].set_ylabel("accuracy")  # nombre de la fila de abajo
    axes[1, 0].set_ylim(0, 1)  # la accuracy va de 0 a 1
    axes[0, 1].legend(frameon=False)  # leyenda de la pérdida
    axes[1, 1].legend(frameon=False, loc="center right")  # leyenda de la accuracy
    for ax in axes.ravel():  # recorre los paneles
        ax.spines[["top", "right"]].set_visible(False)  # quita los bordes de arriba y de la derecha
    fig.tight_layout()  # ajusta los espacios
    return fig  # devuelve la figura


def plot_study(histories, table):  # una curva de validation por experimento, con su rugosidad
    names = list(histories)  # nombres de los experimentos
    columns = 5  # paneles por fila
    rows = int(np.ceil(len(names) / columns))  # filas necesarias
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 2.6 * rows), sharey=True, squeeze=False)  # rejilla de paneles
    for ax, name in zip(axes.ravel(), names):  # un panel por experimento
        history = histories[name]  # historial del experimento
        ax.plot(history["epoch"], history["train_acc"], color=BLUE, lw=1)  # accuracy de train
        ax.plot(history["epoch"], history["val_acc"], color=ORANGE, lw=1.3)  # accuracy de validation
        value = table.loc[table["experimento"] == name, "rugosidad_val_acc"].iloc[0]  # rugosidad del experimento
        ax.set_title(f"{name}\nrugosidad {value:.3f}", fontsize=9)  # nombre y rugosidad
        ax.set_ylim(0, 1)  # la accuracy va de 0 a 1
        ax.spines[["top", "right"]].set_visible(False)  # quita los bordes de arriba y de la derecha
    for ax in axes.ravel()[len(names):]:  # paneles que sobran
        ax.axis("off")  # los esconde
    for ax in axes.ravel()[:len(names)]:  # paneles con curva
        ax.set_xlabel("época", fontsize=8)  # nombre del eje x
    for ax in axes[:, 0]:  # primera columna
        ax.set_ylabel("accuracy")  # nombre del eje y
    fig.tight_layout()  # ajusta los espacios
    return fig  # devuelve la figura


def plot_confusion(confusion, title=None, ax=None):  # matriz de confusión normalizada por fila
    cm = np.asarray(confusion, dtype=float)  # matriz como arreglo
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)  # proporción de cada género real
    if ax is None:  # si no se pasó un panel
        _, ax = plt.subplots(figsize=(4.6, 4.2))  # crea uno
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)  # mapa de calor
    ax.set_xticks(range(len(cm)))  # marcas del eje x
    ax.set_yticks(range(len(cm)))  # marcas del eje y
    ax.set_xlabel("género predicho")  # nombre del eje x
    ax.set_ylabel("género real")  # nombre del eje y
    for i in range(len(cm)):  # recorre las filas
        for j in range(len(cm)):  # recorre las columnas
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if cm[i, j] > 0.5 else "black")  # valor de la casilla
    if title:  # si se pasó un título
        ax.set_title(title, fontsize=10)  # lo pone
    return ax  # devuelve el panel


def plot_gallery(manifests, clouds, per_variant=5, seed=0):  # ejemplos de cada dataset, una fila por género
    rng = np.random.default_rng(seed)  # generador para las rotaciones
    variants = list(manifests)  # nombres de los datasets
    fig, axes = plt.subplots(6, per_variant * len(variants), figsize=(1.5 * per_variant * len(variants), 9))  # rejilla
    for block, variant in enumerate(variants):  # un bloque de columnas por dataset
        manifest = manifests[variant]  # tabla del dataset
        for genus in range(6):  # una fila por género
            subset = manifest[(manifest["genus"] == genus) & (manifest["sample"] == 0)].head(per_variant)  # una muestra por familia
            for column, row in enumerate(subset.itertuples()):  # recorre las columnas del bloque
                ax = axes[genus, block * per_variant + column]  # panel
                ax.imshow(render(clouds[variant][row.sample_id], random_rotation(rng)), cmap="gray_r", vmin=0, vmax=1)  # imagen con rotación al azar
                ax.set_xticks([])  # sin marcas en x
                ax.set_yticks([])  # sin marcas en y
                for side in ax.spines.values():  # recorre los bordes
                    side.set_visible(False)  # los esconde
            axes[genus, block * per_variant].set_ylabel(f"g = {genus}", rotation=0, labelpad=18, fontsize=10)  # género de la fila
        axes[0, block * per_variant + per_variant // 2].set_title(variant, fontsize=12)  # nombre del dataset
    fig.tight_layout()  # ajusta los espacios
    return fig  # devuelve la figura
