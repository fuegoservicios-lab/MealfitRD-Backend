"""[P1-PLAN-LOTE-811 · surrogates · 2026-09-29] Ningún literal de cadena del backend puede llevar surrogates sueltos.

`routers/plans.py` tenía `logger.info(f"\\ud83d\\udd04 [API SHIFT] …")`: en Python 3 esas dos secuencias NO forman el
emoji 🔄, son dos code points sueltos (U+D83D, U+DD04) que no se pueden codificar en UTF-8. Estaba en main desde hace
meses sin que nadie lo ejecutara bajo pytest; los tests HTTP del lote 811 recorren ese camino de `/shift-plan`, el log
capturado llevó los surrogates al informe del worker de xdist y `execnet` abortó con «strings must be utf-8 encodable»
(INTERNALERROR). La fase A del gate salió NO CONCLUYENTE dos veces y cayó a la serie completa (29-sep). En producción
el mismo log revienta el handler que escriba en UTF-8. Escaneo por AST: cualquier constante de cadena con un code point
en U+D800–U+DFFF falla aquí, no en el gate.
"""
import ast
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_EXCLUIR = {"node_modules", ".venv", "venv", "__pycache__", "tests"}


def _constantes_con_surrogates():
    malos = []
    for p in _BACKEND.rglob("*.py"):
        if _EXCLUIR.intersection(p.relative_to(_BACKEND).parts):
            continue
        try:
            arbol = ast.parse(p.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str):
                if any(0xD800 <= ord(c) <= 0xDFFF for c in nodo.value):
                    malos.append(f"{p.relative_to(_BACKEND)}:{nodo.lineno} {nodo.value[:40]!r}")
    return malos


def test_ningun_literal_lleva_surrogates_sueltos():
    malos = _constantes_con_surrogates()
    assert not malos, ("literales con surrogates sueltos (escribe el emoji tal cual o \\U0001xxxx, nunca el par "
                       f"\\ud83d\\udd04): {malos}")


def test_el_log_del_shift_se_codifica_en_utf8():
    src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    linea = next(ln for ln in src.splitlines() if "[API SHIFT] Shifting" in ln)
    valor = ast.literal_eval(linea.split("logger.info(f", 1)[1].split(" [API SHIFT]", 1)[0] + '"')
    valor.encode("utf-8")
