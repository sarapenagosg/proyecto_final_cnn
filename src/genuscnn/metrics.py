import numpy as np  # arreglos numéricos
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # métricas de clasificación


def roughness(values, skip=5):  # qué tan puntuda es una curva: salto medio entre épocas seguidas
    values = np.asarray(values, dtype=float)[skip:]  # ignora las primeras épocas, donde todo cambia rápido
    if len(values) < 2:  # con menos de dos puntos no hay saltos
        return 0.0  # rugosidad cero
    return float(np.abs(np.diff(values)).mean())  # promedio del valor absoluto de los saltos


def classification_summary(labels, preds, n_classes=6):  # resumen de métricas de un conjunto
    labels, preds = np.asarray(labels), np.asarray(preds)  # asegura arreglos
    cm = confusion_matrix(labels, preds, labels=range(n_classes))  # filas = real, columnas = predicho
    distance = np.abs(preds - labels)  # qué tan lejos queda el género predicho del real
    return {  # diccionario con todo
        "accuracy": float(accuracy_score(labels, preds)),  # proporción de aciertos
        "macro_f1": float(f1_score(labels, preds, average="macro")),  # f1 promedio entre géneros
        "recall": (np.diag(cm) / np.maximum(cm.sum(axis=1), 1)).round(4).tolist(),  # recall por género
        "mean_distance": float(distance.mean()),  # distancia media entre predicho y real
        "bias": float((preds - labels).mean()),  # sesgo: positivo si predice géneros de más, negativo si predice de menos
        "errors": int((distance > 0).sum()),  # número de errores
        "errors_distance_1": int((distance == 1).sum()),  # errores por un solo género
        "confusion": cm.tolist(),  # matriz de confusión
    }  # fin del resumen


def vote_by_surface(logits, labels, owners):  # junta las vistas de cada superficie promediando sus salidas
    logits, labels, owners = np.asarray(logits), np.asarray(labels), np.asarray(owners)  # asegura arreglos
    surfaces = np.unique(owners)  # superficies distintas
    mean_logits = np.stack([logits[owners == s].mean(axis=0) for s in surfaces])  # promedio de las salidas de sus vistas
    surface_labels = np.array([labels[owners == s][0] for s in surfaces])  # etiqueta de cada superficie
    return mean_logits.argmax(axis=1), surface_labels  # predicción por superficie y su etiqueta
