"""[P1-DASH-GENERATING-HONESTY · 2026-08-16] El Dashboard prometía minutos para
algo programado para dentro de días.

El día vacío mostraba «Estamos generando este bloque del plan. Se llenará solo en
unos minutos» con `in_flight_count > 0`. Ese contador incluye los chunks `pending`
y `stale` DORMIDOS, cuyo `execute_after` puede estar a una semana vista. El usuario
lo reportó como «parecía que se congeló»: no se congelaba nada, la pantalla mentía.

Es exactamente la mentira que el Historial cerró en mayo (`P3-HIST-CHUNK-SCHEDULED`)
partiendo `in_flight_count` por el reloj. El desglose existía SOLO en `/history-list`;
`/chunk-status`, que es el que sondea el Dashboard, nunca lo heredó.

Las cuatro trampas que rodean esta réplica, cada una con su test abajo:
  1. NO copiar el `WHERE user_id = %s` de `/history-list` — aquí el ownership ya se
     resolvió (P0-HIST-IDOR-2) y añadirlo rompe el binding de `(plan_id,)` en CADA
     tick del polling.
  2. Los dos contadores NO son una partición de `in_flight_count`: un chunk
     `processing` con `execute_after` futuro cae fuera de ambos. `in_flight_count`
     sigue en el payload y el frontend lo conserva como respaldo.
  3. Las claves van en el dict INCONDICIONAL, nunca dentro de `**_upcoming_payload`
     (gateado por `MEALFIT_UPCOMING_DAYS_UI`, un knob que no las gobierna).
  4. Sin prefijo `chunk_`, para que los asserts de `test_p3_hist_chunk_scheduled.py`
     sigan hablando solo de SU endpoint.
"""
from __future__ import annotations

import pytest

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
@pytest.fixture(scope="module", autouse=True)
def _load_frontend_sibling_sources(frontend_repo_path):
    # La fixture compartida salta el módulo antes de cualquier I/O si falta el hermano.
    _ = frontend_repo_path
    global _DASH
    _DASH = (
        _BACKEND.parent / "frontend" / "src" / "pages" / "Dashboard.jsx"
    ).read_text(encoding="utf-8")



def _cuerpo_de_chunk_status() -> str:
    ini = _PLANS.index("def api_chunk_status(")
    fin = _PLANS.find("\ndef ", ini + 10)
    return _PLANS[ini: fin if fin != -1 else len(_PLANS)]


_CHUNK_STATUS = _cuerpo_de_chunk_status()


def _sin_comentarios_py(src: str) -> str:
    return "\n".join(
        l for l in src.splitlines() if not l.strip().startswith("#")
    )


def _sin_comentarios_js(src: str) -> str:
    """Ojo con el `split`: este repo guarda en CRLF y en JS `.` no casa `\\r`.

    Aquí es Python, así que `$` en modo MULTILINE sí llega — pero se normalizan
    los finales de línea igual, para que el helper no dependa de esa asimetría.
    """
    sin_bloque = re.sub(r"/\*.*?\*/", "", src.replace("\r\n", "\n"), flags=re.S)
    return "\n".join(
        re.sub(r"(^|\s)//.*$", r"\1", l) for l in sin_bloque.split("\n")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Backend
# ─────────────────────────────────────────────────────────────────────────────

def _filtro_de(alias: str) -> str:
    """El texto del `COUNT(*) FILTER (...)` que produce `alias`.

    Se localiza por el alias y se lee hacia ATRÁS, en vez de casar el anidamiento
    de paréntesis de delante hacia atrás: `(execute_after IS NULL OR ...)` mete un
    nivel extra y un patrón que cuenta paréntesis se rompe al reformatear el SQL
    sin que cambie ni una condición. Lo que se vigila es la REGLA, no la grafía.
    """
    sql = _sin_comentarios_py(_CHUNK_STATUS)
    fin = sql.find(f"AS {alias}")
    assert fin != -1, f"El alias `{alias}` desapareció de /chunk-status."
    ini = sql.rfind("COUNT(*) FILTER", 0, fin)
    assert ini != -1, f"`{alias}` ya no lo produce un COUNT(*) FILTER."
    return sql[ini:fin]


def test_chunk_status_cuenta_los_dormidos_por_separado():
    f = _filtro_de("scheduled_count")
    for pieza in ("'pending'", "'stale'", "execute_after"):
        assert pieza in f, f"falta {pieza} en el filtro de scheduled_count"
    assert re.search(r"execute_after\s*>\s*NOW\(\)", f), (
        "`scheduled_count` ya no compara contra el reloj: sin `execute_after > NOW()` "
        "no distingue el chunk DORMIDO, y el Dashboard vuelve a prometer «unos "
        "minutos» para un bloque programado para dentro de días."
    )
    assert "'processing'" not in f, (
        "`scheduled_count` incluyó `processing`: un chunk que YA corre no está "
        "programado para más tarde."
    )


def test_chunk_status_cuenta_los_que_corren_ahora():
    f = _filtro_de("running_now_count")
    for pieza in ("'pending'", "'processing'", "'stale'"):
        assert pieza in f, f"falta {pieza} en el filtro de running_now_count"
    assert re.search(r"execute_after\s*<=\s*NOW\(\)", f), (
        "`running_now_count` ya no compara contra el reloj: contaría también los "
        "dormidos y el desglose sería inútil."
    )


def test_la_query_de_counters_no_filtra_por_user_id():
    """Trampa 1: el ownership ya está resuelto arriba; copiar el filtro de
    `/history-list` rompería el binding de `(plan_id,)` en cada tick del polling."""
    sql = _sin_comentarios_py(_CHUNK_STATUS)
    ini = sql.index("AS in_flight_count")
    bloque = sql[ini: sql.index("fetch_one=True", ini)]
    assert "user_id" not in bloque, (
        "Apareció un filtro por `user_id` en la query de counters de /chunk-status. "
        "La tupla de parámetros es `(plan_id,)`: esto es un error de binding de "
        "psycopg en CADA tick del polling del Dashboard, no un fallo silencioso."
    )


def test_los_contadores_no_dependen_del_knob_de_upcoming():
    """Trampa 3: dentro de `_upcoming_payload` quedarían gateados por
    `MEALFIT_UPCOMING_DAYS_UI`, un knob que no los gobierna.

    Se comprueba la PERTENENCIA al dict gateado, no la posición respecto al `**`.
    La primera versión de este test comparaba índices —`scheduled_count` antes de
    `**_upcoming_payload`— y la mutación que mete las claves DENTRO del bloque
    gateado pasaba en verde: el bloque se construye antes en el fuente, así que el
    orden seguía cumpliéndose. Lo descubrió la mutación, no la lectura.
    """
    codigo = _sin_comentarios_py(_CHUNK_STATUS)
    # El bloque se delimita CONTANDO LLAVES, no por indentación. Hay dos
    # `_upcoming_payload = {` en el handler (uno inicializa el dict VACÍO cuando el
    # knob está apagado) y cualquier `}` intermedio a la indentación adivinada
    # cortaba el bloque antes de tiempo: el test daba verde con las claves dentro
    # del dict gateado. Lo descubrió la mutación, no la lectura.
    m = re.search(r"_upcoming_payload\s*=\s*\{(?!\})", codigo)
    assert m, "Cambió el armado del payload gateado; revisa este guard."
    i = m.end() - 1
    profundidad, fin = 0, None
    for j in range(i, len(codigo)):
        if codigo[j] == "{":
            profundidad += 1
        elif codigo[j] == "}":
            profundidad -= 1
            if profundidad == 0:
                fin = j + 1
                break
    assert fin is not None, "No pude delimitar el bloque `_upcoming_payload`."
    bloque_gateado = codigo[i:fin]

    for clave in ('"scheduled_count":', '"running_now_count":'):
        assert clave not in bloque_gateado, (
            f"{clave} quedó DENTRO de `_upcoming_payload`, que solo existe con "
            "MEALFIT_UPCOMING_DAYS_UI encendido. Apagar ese knob dejaría al "
            "Dashboard sin desglose y volvería a la promesa de «unos minutos» "
            "sobre chunks dormidos — sin que nada se pusiera rojo."
        )
        assert clave in codigo[fin:], (
            f"{clave} no aparece en el dict de retorno incondicional."
        )


def test_in_flight_count_sigue_en_el_payload():
    """Trampa 2: los dos contadores NO son una partición. Un chunk `processing`
    con `execute_after` futuro cae fuera de ambos."""
    assert '"in_flight_count":' in _sin_comentarios_py(_CHUNK_STATUS), (
        "Desapareció `in_flight_count` del payload de /chunk-status. Es el respaldo "
        "del frontend para el chunk que no cae en ninguno de los dos buckets."
    )


def test_las_claves_no_llevan_prefijo_chunk():
    """Trampa 4: con prefijo, los asserts de `test_p3_hist_chunk_scheduled.py`
    sobre `"chunk_scheduled_count":` los satisfaría ESTE endpoint."""
    codigo = _sin_comentarios_py(_CHUNK_STATUS)
    assert '"chunk_scheduled_count":' not in codigo
    assert '"chunk_running_now_count":' not in codigo


# ─────────────────────────────────────────────────────────────────────────────
# Frontend
# ─────────────────────────────────────────────────────────────────────────────

def test_el_dashboard_distingue_corriendo_de_programado():
    js = _sin_comentarios_js(_DASH)
    assert "running_now_count" in js, (
        "El Dashboard dejó de leer `running_now_count`: vuelve a decir «se está "
        "generando» sobre chunks dormidos, que es el bug reportado."
    )
    assert "scheduled_count" in js, (
        "El Dashboard dejó de leer `scheduled_count`: pierde la rama honesta "
        "«aún no toca prepararlos»."
    )


def test_corriendo_ahora_se_evalua_antes_que_programado():
    """Con los dos a la vez, la verdad útil es que HAY trabajo en curso."""
    js = _sin_comentarios_js(_DASH)
    pos_corr = js.find("running_now_count")
    pos_prog = js.find("scheduled_count")
    assert pos_corr < pos_prog, (
        "La rama de «programado» se evalúa antes que la de «corriendo ahora»: un "
        "plan con ambas cosas diría que no toca prepararlo mientras se prepara."
    )


def test_el_respaldo_de_in_flight_no_promete_minutos():
    """Un backend anterior al desglose no permite saber si corre o duerme.
    El copy de respaldo NO debe prometer un plazo que no puede conocer."""
    js = _sin_comentarios_js(_DASH)
    ini = js.find("_emptyDayInFlight) {", js.find("running_now_count"))
    assert ini != -1, "Desapareció el respaldo para backends sin el desglose."
    bloque = js[ini: ini + 700]
    assert "unos minutos" not in bloque, (
        "El copy de respaldo volvió a prometer «unos minutos» sin saber si el "
        "chunk corre o duerme. Es exactamente la mentira original."
    )


def test_el_icono_solo_gira_cuando_hay_trabajo_real():
    """`live` anima el icono. Un giro permanente sobre algo dormido es la misma
    mentira de antes, ahora animada."""
    js = _sin_comentarios_js(_DASH)
    pos_live = js.find("live\n", js.find("running_now_count"))
    if pos_live == -1:
        pos_live = js.find("live", js.find("_corriendoAhora"))
    pos_corr = js.find("_corriendoAhora")
    pos_prog = js.find("_programados) {")
    assert pos_corr != -1 and pos_live != -1, "No encontré la rama animada."
    assert pos_corr < pos_live < pos_prog, (
        "El `live` del EmptyState salió de la rama de «corriendo ahora». Si anima "
        "la rama de «programado», el icono gira sobre un bloque dormido."
    )
