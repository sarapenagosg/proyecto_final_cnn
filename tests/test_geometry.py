import numpy as np  # arreglos numéricos
import pytest  # herramienta de tests

from genuscnn import geometry as G  # módulo que se prueba


@pytest.mark.parametrize("g", range(6))  # repite el test para cada género
def test_connected_mesh_has_real_genus(g):  # la variante conexa debe tener género real g
    V, F = G.connected_mesh(g, G.connected_params(g, seed=100 + g))  # malla conexa
    topology = G.mesh_topology(V, F)  # topología medida
    assert topology["components"] == 1  # una sola pieza
    assert topology["closed"]  # superficie cerrada, sin bordes
    assert topology["euler"] == 2 - 2 * g  # característica de euler de una superficie de género g
    assert topology["holes"] == g  # g huecos


@pytest.mark.parametrize("g", range(6))  # repite el test para cada género
def test_chain_mesh_is_g_separate_tori(g):  # la variante en cadena son g toros sueltos
    V, F = G.chain_mesh(g, G.chain_params(g, seed=200 + g))  # malla en cadena
    topology = G.mesh_topology(V, F)  # topología medida
    assert topology["components"] == max(g, 1)  # g piezas (una si es la esfera)
    assert topology["closed"]  # cada pieza es cerrada
    assert topology["euler"] == (2 if g == 0 else 0)  # la esfera da 2 y cada toro suma 0
    assert topology["holes"] == g  # un hueco por toro
    assert len(V) == G.UNIT * max(g, 1)  # 3150 vértices por unidad, como el original


def test_weld_removes_duplicates():  # soldar quita vértices repetidos y caras aplastadas
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)  # el último vértice repite al primero
    F = np.array([[0, 1, 2], [3, 1, 2], [0, 3, 1]])  # la última cara queda aplastada al soldar
    V2, F2 = G.weld(V, F)  # suelda
    assert len(V2) == 3  # quedan 3 vértices únicos
    assert len(F2) == 2  # la cara aplastada desaparece


def test_random_rotation_is_a_rotation():  # la matriz al azar debe ser una rotación
    R = G.random_rotation(np.random.default_rng(0))  # matriz al azar
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)  # es ortogonal
    assert np.isclose(np.linalg.det(R), 1.0)  # determinante 1: no refleja


def test_sample_surface_lies_on_the_torus():  # los puntos muestreados caen sobre la superficie
    R, r = 1.0, 0.3  # radios del toro
    V, F = G.torus_unit(R, r), G.UNIT_FACES["torus"]  # malla de un toro
    X = G.sample_surface(V, F, 2000, np.random.default_rng(1))  # nube de 2000 puntos
    distance = np.sqrt((np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2) - R) ** 2 + X[:, 2] ** 2)  # distancia al círculo central
    assert X.shape == (2000, 3)  # forma correcta
    assert np.abs(distance - r).max() < 0.02  # todos a distancia r del círculo, salvo el error de la malla


def test_deform_is_reproducible_and_centered():  # deformar es reproducible y deja la nube centrada
    V = G.torus_unit(1.0, 0.3)  # un toro
    A = G.deform(V, np.random.default_rng(5))  # primera deformación
    B = G.deform(V, np.random.default_rng(5))  # misma semilla
    assert np.array_equal(A, B)  # mismo resultado
    assert np.allclose(A.mean(axis=0), 0, atol=1e-9)  # centrada en el origen
    assert not np.allclose(A, V)  # y sí cambió la forma


def test_smooth_min_is_below_both_fields():  # el mínimo suave nunca queda por encima del mínimo normal
    a, b = np.linspace(-1, 1, 50), np.linspace(1, -1, 50)  # dos campos de prueba
    assert (G.smooth_min(a, b, 0.3) <= np.minimum(a, b) + 1e-12).all()  # siempre menor o igual
