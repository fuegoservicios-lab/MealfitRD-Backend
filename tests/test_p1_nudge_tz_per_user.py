"""[P1-NUDGE-TZ-PER-USER + P1-AVG-MEAL-HOUR-SIGN · 2026-08-21] El cron de nudges corría en reloj
dominicano hardcodeado para TODOS los usuarios, y la función que decide la hora estaba desviada por
2×offset.

Van juntos porque **se componen**: uno decide a qué hora se evalúa el disparo y el otro a qué hora
cree que come el usuario. Arreglar uno solo mueve el error, no lo cierra.

DEFECTO 1 — el reloj global. `run_proactive_checks` computa
`now_ast = datetime.now(timezone(timedelta(hours=-4)))` —constante literal— dentro de la función
que itera TODAS las sesiones activas. Fase 1 T5 parametrizó `get_daily_nudge_count` por usuario y
dejó el reloj que decide el DISPARO en hora dominicana. Con los horarios por defecto:

    España (UTC+2)     «¿desayunaste?» 15:00 · «¿cenaste?» 01:30 · Resumen del día 05:00
    US-Pacífico        desayuno 05:00-06:00 · resumen 19:00
    México CDMX        resumen 21:00, antes de que haya cenado — «no registraste nada hoy» falso

El Resumen dispara una **notificación push web real**, así que las 05:00 de España no son un
detalle de log. Y el knob global que existía para mitigarlo no puede: su validador es
`0 <= v <= 720`, que **rechaza estructuralmente cualquier offset negativo**. Ni como knob se podía
expresar España.

DEFECTO 2 — el signo. El comentario de `get_avg_meal_hour` dice «+8h» y lo preserva a propósito
desde F1-T5. Pero no es una constante: el doble `AT TIME ZONE` sobre una columna `timestamptz`
SUMA el offset en vez de restarlo, así que el error es `2 × offset`. Mientras el fallback era 240
para todo el mundo eso parecía un sesgo fijo; con offsets reales por usuario pasa a ser **una
función del país**, y para España **cambia de signo**:

    offset  240 (RD)   cena 14:51Z →  buggy 18.0  ·  correcta 10.0   (+8 h)
    offset -120 (ES)   cena 14:51Z →  buggy 12.0  ·  correcta 16.0   (−4 h)
    offset  360 (MX)   cena 14:51Z →  buggy 20.0  ·  correcta  8.0   (+12 h)

Sus TRES funciones hermanas de F1-T5 (`tools.py` ×2, `proactive_agent.py`) ya usan el signo
correcto: ésta era la única desalineada.

Cubre:
  A. El signo: la hora local calculada es la real, para offsets positivos y negativos.
  B. Byte-identidad RD: un usuario dominicano no ve moverse su nudge.
  C. El reloj por usuario: dos usuarios en el mismo tick evalúan horas locales distintas.
  D. El clamp del knob acepta offsets negativos (sin esto Europa es inexpresable).
  E. Parser-based: el hardcode -4 no vuelve, y el helper por usuario se usa.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_PROACTIVE = _BACKEND_ROOT / "proactive_agent.py"
_DB_FACTS = _BACKEND_ROOT / "db_facts.py"


# ── A. El signo de get_avg_meal_hour ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("offset_min,hora_utc,hora_local_esperada", [
    (240, 14, 10.0),    # RD: 14:00Z son las 10:00 AST
    (-120, 14, 16.0),   # ES verano: 14:00Z son las 16:00
    (-60, 14, 15.0),    # ES invierno
    (360, 14, 8.0),     # MX CDMX
    (300, 14, 9.0),     # CO / US-Este
])
def test_la_hora_promedio_es_la_hora_local_real(monkeypatch, offset_min, hora_utc, hora_local_esperada):
    """RED pre-fix: con offset 240 devolvía 18.0 en vez de 10.0, y con -120 devolvía 12.0 en vez
    de 16.0 — el error cambia de SIGNO al cruzar UTC, que es justo lo que el knob de países
    destapó al dar offsets reales por usuario.

    Se ejercita la query REAL: se captura el SQL y sus parámetros y se evalúa la aritmética que
    Postgres haría, en vez de mockear el resultado — un mock del resultado no puede ver un signo
    equivocado dentro del SQL."""
    import db_facts

    capturado = {}

    def _fake_execute(query, params, fetch_all=False):
        capturado["query"] = query
        capturado["params"] = params
        # Reproduce lo que haría Postgres con el intervalo QUE EL CÓDIGO PIDIÓ — leyendo el signo
        # del SQL emitido, no fijándolo aquí. Un fake que hardcodea el signo sólo puede confirmar
        # la aritmética que yo ya supuse; leyéndolo, el test mide de verdad lo que la función
        # manda a la base, que es donde vivía el defecto.
        assert "make_interval" in query
        signo = -1 if "consumed_at - make_interval" in query else +1
        mins = params[0]
        total_min = hora_utc * 60 + signo * mins
        return [{"hr": (total_min // 60) % 24, "mn": total_min % 60}]

    monkeypatch.setattr(db_facts, "execute_sql_query", _fake_execute)
    monkeypatch.setattr(db_facts, "user_tz_offset_min", lambda _uid: offset_min)
    monkeypatch.setattr(db_facts, "connection_pool", object())

    got = db_facts.get_avg_meal_hour("u1", "Cena")
    assert got == pytest.approx(hora_local_esperada, abs=0.01), (
        f"offset={offset_min}: la función cree que el usuario cena a las {got}, "
        f"cuando su hora local real es {hora_local_esperada}"
    )


def test_el_intervalo_se_resta_no_se_suma():
    """El defecto era literalmente un `+` donde sus tres hermanas de F1-T5 usan `-`. Anclarlo por
    texto evita que vuelva por un refactor que 'simplifique' la expresión."""
    src = _DB_FACTS.read_text(encoding="utf-8", errors="replace")
    i = src.find("def get_avg_meal_hour")
    assert i > 0
    _fin = src.find("\ndef ", i + 1)
    cuerpo = src[i:_fin if _fin > 0 else len(src)]
    assert "consumed_at - make_interval" in cuerpo, (
        "get_avg_meal_hour volvió a SUMAR el offset: sobre timestamptz eso desvía 2×offset"
    )
    assert "consumed_at + make_interval" not in cuerpo


# ── B/C. El reloj por usuario ───────────────────────────────────────────────────────────────────

def test_dos_usuarios_en_el_mismo_tick_evaluan_horas_locales_distintas():
    """El corazón del defecto: el reloj vivía FUERA del bucle por usuario. Se comprueba sobre el
    helper que lo calcula, con el mismo instante UTC para los dos."""
    from datetime import datetime, timezone
    import proactive_agent as pa

    ahora = datetime(2026, 8, 21, 3, 0, 0, tzinfo=timezone.utc)  # 03:00Z
    en_rd = pa._local_hour_float_for_offset(ahora, 240)     # 23:00 del día anterior en RD
    en_es = pa._local_hour_float_for_offset(ahora, -120)    # 05:00 en España
    assert en_rd == pytest.approx(23.0, abs=0.01)
    assert en_es == pytest.approx(5.0, abs=0.01)
    assert en_rd != en_es, "los dos usuarios siguen compartiendo el reloj dominicano"


def test_el_dominicano_no_ve_moverse_su_nudge():
    """Byte-identidad: con offset 240 el helper devuelve exactamente lo que devolvía el
    `datetime.now(timezone(timedelta(hours=-4)))` de antes."""
    from datetime import datetime, timezone, timedelta
    import proactive_agent as pa

    ahora = datetime(2026, 8, 21, 15, 30, 0, tzinfo=timezone.utc)
    viejo = ahora.astimezone(timezone(timedelta(hours=-4)))
    esperado = viejo.hour + viejo.minute / 60.0
    assert pa._local_hour_float_for_offset(ahora, 240) == pytest.approx(esperado, abs=0.001)


def test_el_resumen_de_las_23_se_evalua_en_hora_local(monkeypatch):
    """El Resumen del día dispara un push REAL. Antes se evaluaba `now_ast.hour == 23`: para un
    español eso son las 05:00 de la madrugada."""
    from datetime import datetime, timezone
    import proactive_agent as pa

    # 03:00Z = 23:00 en RD, 05:00 en España.
    ahora = datetime(2026, 8, 21, 3, 0, 0, tzinfo=timezone.utc)
    assert int(pa._local_hour_float_for_offset(ahora, 240)) == 23, "en RD sí es la hora del resumen"
    assert int(pa._local_hour_float_for_offset(ahora, -120)) != 23, (
        "un español recibiría el push del resumen a las 5 de la mañana"
    )


# ── D. El clamp del knob ────────────────────────────────────────────────────────────────────────

def test_el_knob_de_huso_acepta_offsets_negativos(monkeypatch):
    """El validador era `0 <= v <= 720`: rechazaba TODO offset negativo, así que Europa era
    inexpresable incluso como override manual. Un knob que no puede representar el caso que
    debería mitigar no es una mitigación."""
    import proactive_agent as pa
    monkeypatch.setenv("MEALFIT_PROACTIVE_TZ_OFFSET_MIN", "-120")
    assert pa._proactive_tz_offset_min() == -120


def test_el_knob_sigue_rechazando_un_disparate(monkeypatch):
    """Control: ampliar el rango no es quitarlo. ±14 h es el máximo real de husos IANA."""
    import proactive_agent as pa
    monkeypatch.setenv("MEALFIT_PROACTIVE_TZ_OFFSET_MIN", "99999")
    assert pa._proactive_tz_offset_min() == 240, "un valor imposible debe caer al default"


# ── E. Parser-based ─────────────────────────────────────────────────────────────────────────────

def test_el_reloj_dominicano_hardcodeado_no_vuelve():
    """`datetime.now(timezone(timedelta(hours=-4)))` dentro del cron que itera TODAS las sesiones
    era el defecto entero en una línea.

    Se escanea con AST y no por texto: la primera versión de este guard quitaba los comentarios
    `#` y se chocó igual con la PROSA de este mismo P-fix, que cita el hardcode dentro de un
    DOCSTRING para explicar qué se arregló. Un filtro de texto siempre deja una forma de prosa
    fuera; el AST no ve prosa en absoluto. (9ª vez que un comentario derrota a un guard aquí.)"""
    import ast

    arbol = ast.parse(_PROACTIVE.read_text(encoding="utf-8", errors="replace"))
    culpables = []
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Call):
            continue
        if getattr(nodo.func, "id", None) != "timedelta":
            continue
        for kw in nodo.keywords:
            if kw.arg == "hours" and isinstance(kw.value, ast.UnaryOp) \
                    and isinstance(kw.value.op, ast.USub) \
                    and getattr(kw.value.operand, "value", None) == 4:
                culpables.append(nodo.lineno)
    assert not culpables, (
        f"volvió el reloj dominicano hardcodeado en el cron de nudges (líneas {culpables})"
    )
    assert "P1-NUDGE-TZ-PER-USER" in _PROACTIVE.read_text(encoding="utf-8", errors="replace")


def test_el_cron_resuelve_el_huso_dentro_del_bucle_por_usuario():
    """El helper `user_tz_offset_min` ya estaba importado en este mismo archivo y se usaba 100
    líneas más arriba: la maquinaria existía y este call site no la llamaba."""
    src = _PROACTIVE.read_text(encoding="utf-8", errors="replace")
    i = src.find("def run_proactive_checks")
    assert i > 0
    _fin = src.find("\ndef ", i + 1)
    cuerpo = src[i:_fin if _fin > 0 else len(src)]
    assert "_local_hour_float_for_offset" in cuerpo, (
        "el cron no calcula la hora local por usuario"
    )
