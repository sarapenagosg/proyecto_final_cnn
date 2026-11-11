import json  # para leer las métricas
import sys  # para agregar src al camino de búsqueda
from pathlib import Path  # rutas que sirven en cualquier sistema operativo

import matplotlib  # gráficos

matplotlib.use("Agg")  # dibuja sin abrir ventanas
import matplotlib.pyplot as plt  # interfaz de gráficos
import pandas as pd  # tablas

ROOT = Path(__file__).resolve().parents[1]  # carpeta raíz del repositorio
sys.path.insert(0, str(ROOT / "src"))  # permite importar genuscnn sin instalarlo

from genuscnn.plots import plot_before_after, plot_confusion, plot_study  # figuras del proyecto

FIGURES = ROOT / "results" / "figures"  # carpeta de las figuras
FIGURES.mkdir(parents=True, exist_ok=True)  # la crea si falta

before = pd.read_csv(ROOT / "results" / "smoothing" / "base" / "history.csv")  # historial de la red del avance
after = pd.read_csv(ROOT / "results" / "runs" / "suave_cadena" / "history.csv")  # historial de la red final
fig = plot_before_after(before, after, ("antes: red del avance", "después: receta suave"))  # compara las curvas
fig.savefig(FIGURES / "curvas_antes_despues.png", dpi=150, bbox_inches="tight")  # guarda la figura
plt.close(fig)  # libera la memoria

table = pd.read_csv(ROOT / "results" / "smoothing" / "summary.csv")  # tabla del estudio de suavizado
histories = {n: pd.read_csv(ROOT / "results" / "smoothing" / n / "history.csv") for n in table["experimento"]}  # historial de cada experimento
fig = plot_study(histories, table)  # una curva por experimento
fig.savefig(FIGURES / "estudio_suavizado.png", dpi=150, bbox_inches="tight")  # guarda la figura
plt.close(fig)  # libera la memoria

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))  # una matriz de confusión por dataset
for ax, variant in zip(axes, ("cadena", "conexa")):  # recorre los dos datasets
    metrics = json.loads((ROOT / "results" / "runs" / f"suave_{variant}" / "metrics.json").read_text())  # métricas de la red final
    plot_confusion(metrics["test"]["confusion"], f"{variant}: test, accuracy {metrics['test']['accuracy']:.3f}", ax)  # matriz de test
fig.tight_layout()  # ajusta los espacios
fig.savefig(FIGURES / "confusion_test.png", dpi=150, bbox_inches="tight")  # guarda la figura
plt.close(fig)  # libera la memoria

cross = pd.read_csv(ROOT / "results" / "cross" / "summary.csv")  # tabla del cruce entre datasets
cross = cross[cross["corrida"].str.startswith("suave")]  # se queda con las redes finales
grid = cross.groupby(["entrenada_en", "evaluada_en"])["test_acc"].mean().unstack()  # media entre semillas: filas = entrenamiento, columnas = evaluación
fig, ax = plt.subplots(figsize=(4.4, 3.6))  # figura del cruce
ax.imshow(grid.values, cmap="Blues", vmin=0, vmax=1)  # mapa de calor
ax.set_xticks(range(2), grid.columns)  # nombres de las columnas
ax.set_yticks(range(2), grid.index)  # nombres de las filas
ax.set_xlabel("evaluada en")  # nombre del eje x
ax.set_ylabel("entrenada en")  # nombre del eje y
for i in range(2):  # recorre las filas
    for j in range(2):  # recorre las columnas
        ax.text(j, i, f"{grid.values[i, j]:.3f}", ha="center", va="center", color="white" if grid.values[i, j] > 0.5 else "black")  # valor de la casilla
ax.set_title("accuracy en test, media de 3 semillas (azar = 0.167)", fontsize=9)  # título
fig.tight_layout()  # ajusta los espacios
fig.savefig(FIGURES / "cruce_datasets.png", dpi=150, bbox_inches="tight")  # guarda la figura
plt.close(fig)  # libera la memoria
print("figuras guardadas en", FIGURES)  # confirma
