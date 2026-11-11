import torch  # tensores
from torch import nn  # capas de la red


def make_norm(kind, channels, bn_momentum=0.1):  # elige la capa de normalización
    if kind == "batch":  # batchnorm: normaliza con las estadísticas del batch
        return nn.BatchNorm2d(channels, momentum=bn_momentum)  # es la que tenía la red original (momentum 0.1)
    if kind == "group":  # groupnorm: normaliza dentro de cada imagen
        return nn.GroupNorm(8, channels)  # 8 grupos de canales; no depende del batch
    raise ValueError(f"normalización desconocida: {kind}")  # avisa el error


class GenusCNN(nn.Module):  # cnn para clasificar el género

    def __init__(self, n_classes=6, norm="batch", dropout=0.3, bn_momentum=0.1):  # recibe clases, normalización, dropout y memoria de batchnorm
        super().__init__()  # inicializa nn.Module
        channels = [1, 32, 64, 128, 128]  # canales de entrada y salida de cada bloque
        layers = []  # capas de la parte convolucional
        for c_in, c_out in zip(channels[:-1], channels[1:]):  # un bloque por cada par de canales
            layers += [  # el bloque completo
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),  # convolución 3x3; padding 1 conserva el tamaño
                make_norm(norm, c_out, bn_momentum),  # normaliza las activaciones
                nn.ReLU(),  # no linealidad
                nn.MaxPool2d(2),  # reduce el tamaño a la mitad
            ]  # fin del bloque
        self.features = nn.Sequential(*layers)  # junta los bloques
        self.pool = nn.AdaptiveAvgPool2d(1)  # promedio global: un número por canal
        self.head = nn.Sequential(  # cabeza clasificadora
            nn.Flatten(),  # aplana (batch, 128, 1, 1) a (batch, 128)
            nn.Linear(channels[-1], 128),  # capa oculta
            nn.ReLU(),  # no linealidad
            nn.Dropout(dropout),  # apaga neuronas al azar en entrenamiento
            nn.Linear(128, n_classes),  # una salida por género
        )  # fin de la cabeza

    def forward(self, x):  # paso hacia adelante
        return self.head(self.pool(self.features(x)))  # convoluciones, pooling y cabeza


def count_parameters(model):  # cuenta los parámetros entrenables
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # suma los tamaños de todos los pesos
