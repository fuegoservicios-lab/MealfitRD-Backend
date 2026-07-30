"""[P1-DIARY-TZ-DEFAULT-RD · 2026-07-30] Sin `date_str`/`tz_offset_mins`,
`get_consumed_meals_today` degrada a la fecha local RD (UTC-4), no a UTC.

Hermano de P1-CHAT-DAY-DEFAULT-TODAY, medido el mismo día. Aquel era un hueco
de contrato del prompt; este es de datos, y estaba VIVO: el chat no-stream
(`agent.py` ~:4521, endpoint real `POST /api/chat` en `routers/chat.py` ~:945)
llama `get_consumed_meals_today(user_id)` sin fecha ni huso.

Medido contra Neon el 2026-07-30 13:37 RD, user 55c537f9:
    ventana LOCAL RD [30-jul 04:00Z .. 31-jul 04:00Z) -> 0 filas   (correcto)
    ventana UTC      [30-jul 00:00Z .. 30-jul 23:59Z] -> 1 fila    (MENTIRA)
La fila extra es la cena de las **22:47 RD del 29** (`2026-07-30T02:47:20Z`):
el path no-stream se la presentaba al coach como "registrado HOY" mientras el
dashboard y el path stream decían 0.

La ventana UTC no está desplazada un poco: cubre [20:00 RD de ayer .. 19:59 RD
de hoy]. O sea mete 4 horas de ANOCHE y, simétricamente, **se le caen la cena y
todo lo comido después de las 20:00 de HOY** — que es justo el fallo que
`proactive_agent` ya había sufrido y parcheado en su propio callsite
(tooltip-anchor `P1-PROACTIVE-TZ`: "un usuario cumplidor caía a `consumed==[]`
y recibía el nudge indulgente").

**Por qué el fix va en el DEFAULT y no en el callsite.** De los 4 callsites de
producción, 3 ya pasaban `date_str`+`tz_offset_mins` explícitos —
`proactive_agent` y `routers/diary.py` lo hacen *precisamente porque el default
es incorrecto*. Un default que todo el mundo tiene que rodear a mano no es un
default: es una trampa con 3 supervivientes y una víctima. Y el resto de la
familia ya degrada bien a UTC-4 sin que nadie la rodee
(`tools._local_date_str_for_user`, `agent._build_plan_today_context`,
`chat_history_context.rd_today`), así que ESTA función era la única disidente.

Tooltip-anchor: P1-DIARY-TZ-DEFAULT-RD
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


def _rd_today_iso() -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=4)).date().isoformat()


@pytest.fixture
def captured_window(monkeypatch):
    """Captura los parámetros SQL en vez de tocar la DB.

    Se asserta sobre la VENTANA, no sobre las filas devueltas: la ventana es
    el contrato: unas filas de fixture podrían pasar por casualidad según la
    hora a la que corra la suite, y entonces el test sería verde de noche y
    rojo de día (o al revés) sin que nadie cambiara nada.
    """
    import db_facts

    calls: list = []

    def _fake_execute(query, params=None, fetch_all=False):
        calls.append({"query": query, "params": params})
        return []

    monkeypatch.setattr(db_facts, "execute_sql_query", _fake_execute)
    # `get_consumed_meals_today` cortocircuita a [] si el pool es falsy —
    # sin esto el fake jamás se invoca y todos los asserts pasan en vacío.
    monkeypatch.setattr(db_facts, "connection_pool", object())
    return calls


def _window(calls: list) -> tuple:
    assert calls, (
        "`execute_sql_query` no se invocó — el vehículo del test no llegó a la "
        "query (¿pool falsy? ¿excepción tragada?). Sin esto todo pasa en vacío."
    )
    params = calls[-1]["params"]
    assert params and len(params) == 3, f"params inesperados: {params!r}"
    return params[1], params[2]


# ===========================================================================
# 1. El default: fecha local RD, no UTC.
# ===========================================================================

def test_no_args_window_starts_at_rd_midnight_not_utc_midnight(captured_window):
    """El bug exacto: sin argumentos la ventana arrancaba en `00:00Z`, que son
    las 20:00 RD de AYER."""
    from db_facts import get_consumed_meals_today

    get_consumed_meals_today("u-1")
    start, end = _window(captured_window)

    d = _rd_today_iso()
    assert start == f"{d}T04:00:00Z", (
        f"P1-DIARY-TZ-DEFAULT-RD: la ventana sin argumentos debe abrir a las "
        f"00:00 RD (= {d}T04:00:00Z), no a las 00:00Z. Recibido: {start!r}. "
        f"Con `{d}T00:00:00Z` entran las comidas de las 20:00-23:59 RD de AYER."
    )
    nxt = (datetime.fromisoformat(d) + timedelta(days=1)).date().isoformat()
    assert end.startswith(f"{nxt}T03:59"), (
        f"la ventana debe cerrar al final del día RD (~{nxt}T03:59:59Z), no a "
        f"las 23:59Z de hoy — si no, se pierde todo lo comido después de las "
        f"20:00 RD de HOY (el falso-positivo nocturno de P1-PROACTIVE-TZ). "
        f"Recibido: {end!r}"
    )


def test_no_args_window_excludes_last_nights_late_meal(captured_window):
    """La fila real del incidente (`2026-07-30T02:47:20Z` = 22:47 RD del 29)
    debe quedar FUERA de la ventana de hoy — comprobado por aritmética de la
    ventana, no por fixture."""
    from db_facts import get_consumed_meals_today

    get_consumed_meals_today("u-1")
    start, _ = _window(captured_window)

    # Reproducción de la aritmética con la fecha del incidente.
    incident_utc = datetime(2026, 7, 30, 2, 47, 20, tzinfo=timezone.utc)
    rd_local = incident_utc - timedelta(hours=4)
    assert rd_local.date().isoformat() == "2026-07-29"
    window_start_for_the_30th = datetime(2026, 7, 30, 4, 0, 0, tzinfo=timezone.utc)
    assert incident_utc < window_start_for_the_30th, (
        "la cena de las 22:47 RD del 29 cae ANTES del arranque de la ventana "
        "del 30 — si este assert falla, la convención UTC-4 se rompió."
    )
    # Y la implementación viva usa esa misma convención (offset 04:00).
    assert start.endswith("T04:00:00Z")


# ===========================================================================
# 2. No pisar la intención del caller.
# ===========================================================================

def test_explicit_date_is_preserved_when_tz_is_missing(captured_window):
    """`GET /api/diary/consumed/{uid}?date=YYYY-MM-DD` sin `tzOffset` es un
    caller real (`routers/diary.py`, ambos query params opcionales). El
    default del huso NO puede pisarle la fecha que sí mandó."""
    from db_facts import get_consumed_meals_today

    get_consumed_meals_today("u-1", date_str="2026-07-29")
    start, end = _window(captured_window)
    assert start == "2026-07-29T04:00:00Z", (
        f"la fecha explícita del caller fue ignorada — recibido {start!r}. "
        f"Defaultear el huso NUNCA debe reescribir el día pedido."
    )
    assert end.startswith("2026-07-30T03:59")


def test_explicit_date_and_tz_still_win(captured_window):
    """Los 3 callsites que hoy pasan ambos valores (proactive_agent, diary,
    chat stream) no pueden cambiar de comportamiento."""
    from db_facts import get_consumed_meals_today

    get_consumed_meals_today("u-1", date_str="2026-07-29", tz_offset_mins=0)
    start, end = _window(captured_window)
    assert start == "2026-07-29T00:00:00Z"
    assert end.startswith("2026-07-29T23:59")


def test_tz_zero_is_honoured_not_treated_as_missing(captured_window):
    """`0` es UTC: un huso legítimo y falsy. Resolver por truthiness lo
    convertiría en 240 en silencio."""
    from db_facts import get_consumed_meals_today

    get_consumed_meals_today("u-1", date_str="2026-07-29", tz_offset_mins=0)
    start, _ = _window(captured_window)
    assert start == "2026-07-29T00:00:00Z", (
        "tz_offset_mins=0 (UTC) se está tratando como ausente y sustituyendo "
        "por 240 — resolución por truthiness en vez de `is not None`."
    )


# ===========================================================================
# 3. El invariante de CLASE: la familia deriva el mismo día.
#    Esto es lo que impide que la próxima función nazca disidente.
# ===========================================================================

def test_whole_family_agrees_on_which_day_is_today(captured_window):
    """`get_consumed_meals_today` (sin args), `tools._local_date_str_for_user`
    y `chat_history_context.rd_today` deben coincidir. Eran tres derivaciones
    del mismo concepto y una iba por UTC."""
    from chat_history_context import rd_today
    from db_facts import get_consumed_meals_today
    from tools import _local_date_str_for_user

    get_consumed_meals_today("u-1")
    start, _ = _window(captured_window)
    day_from_window = start[:10]

    assert day_from_window == _local_date_str_for_user(), (
        f"`get_consumed_meals_today` sin args dice que hoy es "
        f"{day_from_window}, y `tools._local_date_str_for_user` dice "
        f"{_local_date_str_for_user()}. Dos bloques del MISMO system prompt "
        f"afirmando días distintos es el modo de fallo de P1-CHAT-PAST-DAYS."
    )
    assert day_from_window == rd_today().isoformat()


def test_source_no_longer_derives_today_from_utc_date(captured_window):
    """Ancla de source: la función no puede volver a construir el día desde
    `datetime.now(timezone.utc).strftime('%Y-%m-%d')` sin restar las 4 horas.
    Acotado al cuerpo de ESTA función — sobre el archivo entero el `not in`
    chocaría con otros helpers que sí usan UTC con razón."""
    src = (_BACKEND_ROOT / "db_facts.py").read_text(encoding="utf-8")
    start = src.find("def get_consumed_meals_today")
    assert start != -1
    end = src.find("\ndef ", start + 1)
    body = src[start:end if end != -1 else len(src)]
    # Sanity del vehículo: si el slice no fuera el cuerpo real, el negativo
    # de abajo pasaría en vacío.
    assert "consumed_at >= %s" in body and len(body) > 500

    assert "P1-DIARY-TZ-DEFAULT-RD" in body, (
        "falta el marker de trazabilidad en el cuerpo de la función."
    )
    naive_utc_day = 'datetime.now(timezone.utc).strftime("%Y-%m-%d")'
    assert naive_utc_day not in body, (
        f"el cuerpo volvió a derivar el día de hoy desde la fecha UTC "
        f"({naive_utc_day}) — es exactamente el bug: entre las 20:00 y las "
        f"23:59 RD eso es MAÑANA."
    )
