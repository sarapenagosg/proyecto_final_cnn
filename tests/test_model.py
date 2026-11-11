import pytest  # herramienta de tests
import torch  # tensores

from genuscnn.model import GenusCNN, count_parameters  # lo que se prueba


@pytest.mark.parametrize("norm", ["batch", "group"])  # repite el test para las dos normalizaciones
def test_output_shape(norm):  # la red devuelve una salida por género
    model = GenusCNN(n_classes=6, norm=norm)  # crea la red
    out = model(torch.zeros(4, 1, 64, 64))  # 4 imágenes vacías
    assert out.shape == (4, 6)  # 4 filas y 6 géneros


def test_parameter_count_matches_the_original_network():  # misma red que la del avance
    assert count_parameters(GenusCNN()) == 258_246  # número de parámetros del cuaderno original


def test_gradients_reach_every_weight():  # backpropagation llega a todos los pesos
    model = GenusCNN()  # crea la red
    loss = model(torch.randn(8, 1, 64, 64)).logsumexp(1).sum()  # una pérdida cualquiera
    loss.backward()  # calcula los gradientes
    assert all(p.grad is not None for p in model.parameters())  # todos los pesos tienen gradiente


def test_dropout_is_off_in_eval():  # en evaluación la red es determinista
    model = GenusCNN(dropout=0.5).eval()  # modo evaluación
    x = torch.randn(2, 1, 64, 64)  # dos imágenes
    assert torch.equal(model(x), model(x))  # misma salida las dos veces


def test_unknown_norm_raises():  # una normalización mal escrita debe fallar
    with pytest.raises(ValueError):  # espera el error
        GenusCNN(norm="otra")  # normalización que no existe
