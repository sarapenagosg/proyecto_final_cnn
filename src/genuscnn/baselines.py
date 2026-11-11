import numpy as np  # arreglos numéricos
from sklearn.linear_model import LogisticRegression  # clasificador simple
from sklearn.pipeline import make_pipeline  # encadena el escalado y el clasificador
from sklearn.preprocessing import StandardScaler  # pone los rasgos en la misma escala


def shape_features(images):  # rasgos de la silueta que no necesitan ver ningún hueco
    features = []  # una fila de rasgos por imagen
    for image in np.asarray(images)[:, 0]:  # recorre las imágenes (quita el canal)
        rows, cols = np.nonzero(image > 0)  # posiciones de los píxeles pintados
        coords = np.stack([cols, rows], axis=1).astype(float)  # coordenadas x, y de la silueta
        spread = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(coords.T)), 0))  # ancho y largo de la silueta (desviaciones principales)
        ink = len(rows) / image.size  # proporción de píxeles pintados
        features.append([ink, spread[1], spread[0], spread[0] / max(spread[1], 1e-9)])  # tinta, largo, ancho y qué tan alargada es
    return np.array(features)  # matriz (n_imágenes, 4)


def fit_predict(train_x, train_y, eval_x):  # entrena una regresión logística y predice
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))  # escala los rasgos y clasifica
    model.fit(train_x, train_y)  # ajusta con train
    return model.predict(eval_x)  # predicciones para las imágenes pedidas
