import io  # para leer el código como flujo de texto
import sys  # para terminar con error si algo falla
import tokenize  # separa el código en piezas y encuentra los comentarios
from pathlib import Path  # rutas que sirven en cualquier sistema operativo

ROOT = Path(__file__).resolve().parents[1]  # carpeta raíz del repositorio


def uncommented_lines(source):  # líneas de código sin comentario válido a la derecha
    commented, used = {}, set()  # comentario de cada línea y líneas que tienen código
    for token in tokenize.generate_tokens(io.StringIO(source).readline):  # recorre las piezas del código
        if token.type == tokenize.COMMENT:  # si es un comentario
            commented[token.start[0]] = token.string.lstrip("#").strip()  # guarda su texto
        elif token.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):  # si es código
            used.update(range(token.start[0], token.end[0] + 1))  # marca las líneas que ocupa
    bad = []  # líneas con problema
    for line in sorted(used):  # recorre las líneas con código
        text = commented.get(line, "")  # comentario de la línea
        if not text or not (text[0].islower() or not text[0].isalpha()):  # falta o empieza en mayúscula
            bad.append(line)  # la anota
    return bad  # devuelve las líneas con problema


failures = 0  # cuenta los problemas
for folder in ("src", "scripts", "tests"):  # carpetas con código
    for path in sorted((ROOT / folder).rglob("*.py")):  # recorre los archivos de python
        for line in uncommented_lines(path.read_text(encoding="utf-8")):  # recorre las líneas con problema
            print(f"{path.relative_to(ROOT)}:{line} sin comentario en minúscula")  # avisa cuál es
            failures += 1  # suma el problema
print("líneas sin comentar:", failures)  # resumen
sys.exit(1 if failures else 0)  # termina con error si hubo problemas
