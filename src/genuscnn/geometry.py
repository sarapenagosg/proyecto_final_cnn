import numpy as np  # arreglos numéricos
from scipy.sparse import coo_matrix  # matriz dispersa para el grafo de la malla
from scipy.sparse.csgraph import connected_components  # cuenta las piezas separadas de un grafo
from skimage.measure import marching_cubes  # saca una malla de una superficie implícita

N_MAJOR, N_MINOR = 70, 45  # cuadrícula de 70 x 45 = 3150 vértices por unidad, igual que el dataset original
UNIT = N_MAJOR * N_MINOR  # vértices de una unidad (un toro o una esfera)
SIZE_FACTOR = np.array([1.0, 1.0, 0.78, 0.70, 0.60, 0.59])  # los toros se achican al crecer el género, medido en el original
SURFACE_TYPE = {  # nombre de cada tipo de superficie, como en el metadata original
    0: "sphere_like",  # género 0
    1: "torus",  # género 1
    2: "double_torus_like",  # género 2
    3: "triple_torus_like",  # género 3
    4: "quadruple_torus_like",  # género 4
    5: "quintuple_torus_like",  # género 5
}  # fin del diccionario de nombres


def grid_faces(n_u, n_v, wrap_u):  # triángulos de una cuadrícula n_u x n_v
    faces = []  # lista de triángulos
    for i in range(n_u if wrap_u else n_u - 1):  # recorre las filas
        i2 = (i + 1) % n_u  # fila siguiente (da la vuelta si wrap_u)
        for j in range(n_v):  # recorre las columnas
            j2 = (j + 1) % n_v  # columna siguiente (siempre da la vuelta)
            a, b = i * n_v + j, i2 * n_v + j  # dos esquinas del cuadrado
            c, d = i2 * n_v + j2, i * n_v + j2  # las otras dos esquinas
            faces += [[a, b, c], [a, c, d]]  # el cuadrado se parte en 2 triángulos
    return np.array(faces)  # arreglo (n_caras, 3)


def torus_unit(R, r):  # vértices de un toro de radio mayor R y radio de tubo r
    u = np.linspace(0, 2 * np.pi, N_MAJOR, endpoint=False)[:, None]  # ángulo alrededor del eje del toro
    v = np.linspace(0, 2 * np.pi, N_MINOR, endpoint=False)[None, :]  # ángulo alrededor del tubo
    x = (R + r * np.cos(v)) * np.cos(u)  # coordenada x
    y = (R + r * np.cos(v)) * np.sin(u)  # coordenada y
    z = r * np.sin(v) * np.ones_like(u)  # coordenada z
    return np.stack([x, y, z], -1).reshape(-1, 3)  # lista de 3150 vértices


def sphere_unit(a, b, c):  # vértices de un elipsoide de semiejes a, b, c
    u = np.linspace(0, np.pi, N_MAJOR)[:, None]  # de un polo al otro
    v = np.linspace(0, 2 * np.pi, N_MINOR, endpoint=False)[None, :]  # alrededor del eje
    x = a * np.sin(u) * np.cos(v)  # coordenada x
    y = b * np.sin(u) * np.sin(v)  # coordenada y
    z = c * np.cos(u) * np.ones_like(v)  # coordenada z
    return np.stack([x, y, z], -1).reshape(-1, 3)  # lista de 3150 vértices


UNIT_FACES = {  # caras de cada tipo de unidad
    "torus": grid_faces(N_MAJOR, N_MINOR, True),  # el toro se cierra en los dos sentidos
    "sphere": grid_faces(N_MAJOR, N_MINOR, False),  # la esfera no se cierra de polo a polo
}  # fin del diccionario de caras


def chain_faces(g):  # caras de la superficie en cadena de género g
    if g == 0:  # la esfera es una sola unidad
        return UNIT_FACES["sphere"]  # sus caras
    return np.vstack([UNIT_FACES["torus"] + i * UNIT for i in range(g)])  # g toros con los índices desplazados


def rot_xyz(ax, ay, az):  # matriz de rotación por ángulos en x, y, z
    cx, sx = np.cos(ax), np.sin(ax)  # coseno y seno del giro en x
    cy, sy = np.cos(ay), np.sin(ay)  # coseno y seno del giro en y
    cz, sz = np.cos(az), np.sin(az)  # coseno y seno del giro en z
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])  # giro alrededor de x
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])  # giro alrededor de y
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])  # giro alrededor de z
    return Rz @ Ry @ Rx  # las junta en una sola matriz


def random_rotation(rng):  # rotación 3d uniforme al azar
    q = rng.normal(size=4)  # cuaternión al azar
    q /= np.linalg.norm(q)  # lo normaliza para que sea una rotación
    w, x, y, z = q  # sus cuatro componentes
    return np.array([  # arma la matriz 3x3
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],  # primera fila
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],  # segunda fila
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],  # tercera fila
    ])  # fin de la matriz


def chain_params(g, seed):  # parámetros base de una familia en cadena (método original)
    rng = np.random.default_rng(seed)  # generador propio de la familia
    p = {  # parámetros comunes
        "R": rng.uniform(0.9, 1.3) * SIZE_FACTOR[g],  # radio mayor del toro
        "r": rng.uniform(0.22, 0.38) * SIZE_FACTOR[g],  # radio del tubo
        "spacing": rng.uniform(2.3, 3.4),  # separación entre toros, en múltiplos de R
    }  # fin de los parámetros comunes
    if g == 0:  # esfera
        p["abc"] = rng.uniform(0.8, 1.25, size=3)  # semiejes del elipsoide
    else:  # toros
        p["tilts"] = rng.uniform(-0.3, 0.3, size=(g, 2))  # inclinación de cada toro en radianes
        p["jitter"] = rng.uniform(0.9, 1.1, size=(g, 2))  # variación de R y r entre toros
    return p  # devuelve los parámetros


def chain_mesh(g, p):  # malla en cadena: g toros sueltos (o una esfera si g = 0)
    if g == 0:  # esfera
        return sphere_unit(*p["abc"]), chain_faces(0)  # un elipsoide y sus caras
    parts = []  # un bloque de vértices por toro
    for i in range(g):  # recorre los g toros
        V = torus_unit(p["R"] * p["jitter"][i, 0], p["r"] * p["jitter"][i, 1])  # toro con su tamaño
        V = V @ rot_xyz(p["tilts"][i, 0], p["tilts"][i, 1], 0).T  # lo inclina
        V[:, 0] += (i - (g - 1) / 2) * p["spacing"] * p["R"]  # lo coloca en la cadena a lo largo de x
        parts.append(V)  # guarda el toro
    return np.vstack(parts), chain_faces(g)  # junta todos los toros y sus caras


def connected_params(g, seed):  # parámetros base de una familia conexa (género real)
    rng = np.random.default_rng(seed)  # generador propio de la familia
    p = {  # parámetros comunes, con los mismos rangos que la cadena
        "R": rng.uniform(0.9, 1.3) * SIZE_FACTOR[g],  # radio mayor del toro
        "r": rng.uniform(0.22, 0.38) * SIZE_FACTOR[g],  # radio del tubo
    }  # fin de los parámetros comunes
    if g == 0:  # esfera
        p["abc"] = rng.uniform(0.8, 1.25, size=3)  # semiejes del elipsoide
    else:  # toros pegados
        p["twist"] = rng.uniform(-0.6, 0.6, size=g)  # giro de cada toro alrededor del eje de la cadena
        p["tilt"] = rng.uniform(-0.08, 0.08, size=g)  # inclinación pequeña fuera del eje
        p["jitter"] = rng.uniform(0.9, 1.1, size=(g, 2))  # variación de R y r entre toros
        p["gap"] = rng.uniform(0.2, 1.0, size=max(g - 1, 1))  # distancia entre tubos vecinos, en múltiplos de r
        p["blend"] = rng.uniform(0.3, 0.8)  # qué tan redondeada queda la unión, en múltiplos de r
    return p  # devuelve los parámetros


def torus_field(P, R, r, center_x, twist, tilt):  # distancia con signo de los puntos P a un toro
    Q = P - np.array([center_x, 0.0, 0.0])  # mueve el centro del toro al origen
    Q = Q @ rot_xyz(twist, tilt, 0)  # deshace el giro del toro (multiplicar a la derecha aplica la inversa)
    ring = np.sqrt(Q[:, 0] ** 2 + Q[:, 1] ** 2) - R  # distancia al círculo central en el plano
    return np.sqrt(ring ** 2 + Q[:, 2] ** 2) - r  # distancia al tubo: negativa adentro, positiva afuera


def smooth_min(a, b, k):  # mínimo suave de dos campos (une dos sólidos redondeando la junta)
    h = np.maximum(k - np.abs(a - b), 0.0) / k  # cuánto se mezclan cerca de la junta
    return np.minimum(a, b) - h * h * k / 4.0  # mínimo normal menos un redondeo


def connected_mesh(g, p, voxels_per_tube=3.5):  # malla conexa de género g a partir de un campo implícito
    if g == 0:  # esfera
        a, b, c = p["abc"]  # semiejes
        half = np.array([a, b, c]) + 0.15  # media caja que la contiene con margen
        h = 0.06  # tamaño del vóxel
        centers = np.zeros(1)  # un solo centro en el origen
    else:  # cadena de toros pegados
        Rs = p["R"] * p["jitter"][:, 0]  # radio mayor de cada toro
        rs = p["r"] * p["jitter"][:, 1]  # radio de tubo de cada toro
        centers = np.zeros(g)  # posición en x de cada centro
        for i in range(1, g):  # coloca cada toro junto al anterior
            centers[i] = centers[i - 1] + Rs[i - 1] + Rs[i] + p["gap"][i - 1] * min(rs[i - 1], rs[i])  # los tubos quedan solapados
        centers -= centers.mean()  # centra la cadena en el origen
        reach = (Rs + rs).max() + 0.15  # alcance de un toro con margen
        half = np.array([np.abs(centers).max() + reach, reach, reach])  # media caja que contiene la cadena
        h = rs.min() / voxels_per_tube  # vóxel pequeño frente al tubo más fino
    axes = [np.arange(-half[k], half[k] + h, h) for k in range(3)]  # coordenadas de la rejilla en cada eje
    X, Y, Z = np.meshgrid(*axes, indexing="ij")  # rejilla 3d
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # puntos de la rejilla como lista
    if g == 0:  # campo del elipsoide
        field = np.sqrt((P[:, 0] / a) ** 2 + (P[:, 1] / b) ** 2 + (P[:, 2] / c) ** 2) - 1.0  # negativo adentro
    else:  # campo de la unión de toros
        field = torus_field(P, Rs[0], rs[0], centers[0], p["twist"][0], p["tilt"][0])  # primer toro
        for i in range(1, g):  # agrega los demás toros
            other = torus_field(P, Rs[i], rs[i], centers[i], p["twist"][i], p["tilt"][i])  # campo del toro i
            field = smooth_min(field, other, p["blend"] * p["r"])  # lo une con lo que ya había
    volume = field.reshape(X.shape)  # campo con forma de rejilla
    V, F, _, _ = marching_cubes(volume, level=0.0, spacing=(h, h, h))  # malla de la superficie de nivel cero
    V = V - half  # devuelve los vértices a las coordenadas originales
    return V, F.astype(np.int64)  # vértices y caras


def weld(V, F, decimals=6):  # suelda los vértices repetidos de una malla
    _, first, inverse = np.unique(np.round(V, decimals), axis=0, return_index=True, return_inverse=True)  # vértices únicos
    F2 = inverse.reshape(-1)[F]  # caras con los índices nuevos
    keep = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])  # caras que no quedaron aplastadas
    return V[first], F2[keep]  # vértices únicos y caras válidas


def mesh_topology(V, F):  # mide la topología real de una malla
    V, F = weld(V, F)  # primero suelda vértices repetidos (los polos de la esfera)
    edges = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1)  # todas las aristas con extremos ordenados
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)  # aristas sin repetir y cuántas caras las usan
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(V), len(V)))  # grafo de la malla
    components = connected_components(graph, directed=False)[0]  # piezas separadas
    chi = len(V) - len(unique_edges) + len(F)  # característica de euler: v - e + f
    closed = bool((counts == 2).all())  # cerrada si cada arista toca exactamente dos caras
    holes = (2 * components - chi) // 2  # suma de los géneros de todas las piezas
    return {"components": int(components), "euler": int(chi), "closed": closed, "holes": int(holes)}  # resumen


def deform(V, rng):  # aplica las tres deformaciones del dataset original a una muestra
    out = V.copy()  # copia para no modificar la base
    for _ in range(3):  # tres ondas
        w = rng.normal(size=3)  # dirección de la onda
        w *= rng.uniform(0.6, 1.6) / np.linalg.norm(w)  # frecuencia entre 0.6 y 1.6
        d = rng.normal(size=3)  # dirección del desplazamiento
        d /= np.linalg.norm(d)  # la vuelve unitaria
        out += rng.uniform(0.04, 0.16) * np.sin(V @ w + rng.uniform(0, 2 * np.pi))[:, None] * d  # desplaza cada punto según la onda
    S = np.exp(rng.uniform(np.log(0.6), np.log(1.7), size=3))  # escalas anisotrópicas entre 0.6 y 1.7
    out = (out * S) @ random_rotation(rng).T  # escala cada eje y rota al azar
    return out - out.mean(0)  # centra en el origen


def sample_surface(V, F, n, rng):  # n puntos al azar sobre la superficie de la malla
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]  # los 3 vértices de cada triángulo
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)  # área de cada triángulo
    idx = rng.choice(len(F), size=n, p=area / area.sum())  # elige triángulos con probabilidad proporcional al área
    r1, r2 = rng.random(n), rng.random(n)  # dos números al azar por punto
    s = np.sqrt(r1)  # para repartir uniforme dentro del triángulo
    return (1 - s)[:, None] * A[idx] + (s * (1 - r2))[:, None] * B[idx] + (s * r2)[:, None] * C[idx]  # punto con coordenadas baricéntricas
