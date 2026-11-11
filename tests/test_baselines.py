import numpy as np  # arreglos numéricos

from genuscnn.baselines import fit_predict, shape_features  # lo que se prueba


def test_shape_features_measure_the_silhouette():  # los rasgos describen la silueta
    images = np.zeros((2, 1, 16, 16), dtype=np.float32)  # dos imágenes vacías
    images[0, 0, 4:12, 4:12] = 1.0  # un cuadrado de 8 x 8
    images[1, 0, 7:9, 0:16] = 1.0  # una barra horizontal de 2 x 16
    features = shape_features(images)  # tinta, largo, ancho y alargamiento
    assert features.shape == (2, 4)  # 4 rasgos por imagen
    assert features[0, 0] == 64 / 256  # tinta del cuadrado
    assert np.isclose(features[0, 3], 1.0)  # el cuadrado no es alargado
    assert features[1, 3] < 0.2  # la barra es muy alargada: ancho mucho menor que largo


def test_fit_predict_learns_a_simple_rule():  # la regresión logística aprende una regla fácil
    x = np.linspace(0, 1, 40)[:, None]  # un solo rasgo
    y = (x[:, 0] > 0.5).astype(int)  # la clase depende de pasar de 0.5
    assert (fit_predict(x, y, x) == y).mean() > 0.9  # acierta casi todo
