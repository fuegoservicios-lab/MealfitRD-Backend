"""[P1-FETCH-EXPLICIT · 2026-08-14] Guard blanket: toda `execute_sql_query()` cuya
query PRODUCE filas debe declarar `fetch_all=True` (o `fetch_one=True`).

Por qué existe. `execute_sql_query` tiene un default seguro: si el caller no pide
filas y la query las devolvió igual, las retorna Y emite un WARNING pidiendo el flag
explícito. Ese WARNING es correcto — pero medido en producción (journal del VPS,
7 días): **6.016 avisos**, de los cuales 1.833 salían de UNA sola query, el desglose
de `quality_tier` del endpoint de chunk-status, que el frontend POLLEA mientras el
plan está `partial`. Un aviso por tick.

El daño no es el rendimiento: es que 6.016 líneas de una clase conocida entierran las
señales reales del mismo canal. En la sesión que abrió este fix hubo que filtrarlas a
mano para poder leer los warnings del guard de coherencia y del pipeline. *Un canal de
avisos que grita por lo que ya sabemos acaba ignorado* — la misma lección que
P1-TEST-ALERT-POLLUTION dejó escrita sobre `system_alerts`.

El fix es el flag explícito en los 48 call sites, NO silenciar el WARNING: con los
sitios estáticos callados, cada WARNING que quede vuelve a ser información (una query
dinámica nueva, o un call site recién escrito).

Semánticamente es un no-op: `fetch_all=True` devuelve `cursor.fetchall()`, que es
exactamente lo que el default seguro ya retornaba.

Alcance del guard: queries LITERALES que se puede probar que producen filas (empiezan
por SELECT/WITH, o traen RETURNING). Las queries construidas en runtime quedan fuera
—no se puede probar estáticamente qué son— y para ellas el WARNING de `db_core` sigue
siendo la red viva.
"""
import ast
import io
import os

import pytest

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SKIP_DIRS = {"tests", "__pycache__", ".git", "venv", "node_modules", "scripts", "migrations"}

# Piso anti-vacuidad: si alguien renombra el helper, el escáner encontraría 0 llamadas
# y este guard pasaría sin mirar nada — «un guard que ya no puede fallar es peor que no
# tenerlo». El piso es holgado (había 394 call sites al escribirlo).
_MIN_CALLSITES = 200


def _query_text(node):
    """Texto de la query si es literal (str, f-string de literales o concatenación)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            v.value for v in node.values
            if isinstance(v, ast.Constant) and isinstance(v.value, str)
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        izq, der = _query_text(node.left), _query_text(node.right)
        if izq is None and der is None:
            return None
        return (izq or "") + (der or "")
    return None


def _produce_filas(q):
    """True si la query DEMUESTRA que devuelve un result set. None si no es literal."""
    if q is None:
        return None
    primera = ""
    for linea in q.splitlines():
        s = linea.strip()
        if not s or s.startswith("--"):
            continue
        primera = s
        break
    if primera.upper().startswith(("SELECT", "WITH")):
        return True
    return "RETURNING" in q.upper()


def _escanear():
    """(violaciones, total_callsites, dinamicas) sobre el código de producción."""
    violaciones, dinamicas = [], []
    total = 0
    for dirpath, dirnames, filenames in os.walk(_BACKEND):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                arbol = ast.parse(io.open(path, encoding="utf-8").read())
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(arbol):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                nombre = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
                if nombre != "execute_sql_query":
                    continue
                total += 1
                kwargs = {k.arg for k in node.keywords if k.arg}
                if "fetch_one" in kwargs or "fetch_all" in kwargs:
                    continue
                q = _query_text(node.args[0]) if node.args else None
                veredicto = _produce_filas(q)
                rel = os.path.relpath(path, _BACKEND).replace("\\", "/")
                if veredicto is True:
                    violaciones.append(f"{rel}:{node.lineno}  {(q or '').strip().splitlines()[0][:70]}")
                elif veredicto is None:
                    dinamicas.append(f"{rel}:{node.lineno}")
    return violaciones, total, dinamicas


def test_el_helper_sigue_llamandose_asi():
    """Ancla: si `execute_sql_query` se renombra, el escáner mide el vacío."""
    fuente = io.open(os.path.join(_BACKEND, "db_core.py"), encoding="utf-8").read()
    assert "def execute_sql_query(" in fuente, (
        "El helper cambió de nombre: este guard quedaría vacío. Actualiza el escáner."
    )
    # Fragmento CONTIGUO: el mensaje vive partido en dos literales del f-string,
    # así que buscar la frase completa daría un falso rojo.
    assert "fetch_one/fetch_all pero la query devolvió" in fuente, (
        "Desapareció el WARNING del default seguro. Es la red viva para las queries "
        "dinámicas — si se elimina, este guard pasa a ser la ÚNICA defensa y debe "
        "endurecerse (cubrir también las dinámicas) antes de borrarlo."
    )


def test_el_escaner_ve_el_codigo_de_produccion():
    """Sanity: sin este piso, un escáner roto (o un skip de directorios de más)
    reportaría cero violaciones por no estar mirando nada."""
    _, total, _ = _escanear()
    assert total >= _MIN_CALLSITES, (
        f"Solo {total} call sites de execute_sql_query — se esperaban ≥{_MIN_CALLSITES}. "
        f"El escáner no está viendo el código de producción."
    )


def test_toda_query_que_devuelve_filas_declara_su_fetch():
    violaciones, _, _ = _escanear()
    if violaciones:
        pytest.fail(
            f"{len(violaciones)} call site(s) de execute_sql_query con query que "
            f"devuelve filas y sin fetch_all/fetch_one explícito. Cada uno emite un "
            f"WARNING por ejecución en producción (los del polling, uno por tick) y "
            f"entierra las señales reales del canal.\n\n"
            f"Fix: añade `fetch_all=True` — es un no-op semántico (el default seguro "
            f"ya devuelve `cursor.fetchall()`).\n\n  " + "\n  ".join(violaciones)
        )
