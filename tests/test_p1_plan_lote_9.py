"""[P1-PLAN-LOTE-9 · 2026-09-11] Noveno lote del plan de pendientes: el hermano de G52.

G52 (lote 8) hizo que el bloque temporal del prompt leyera el huso del usuario. Pero el huso se resolvía A MANO en
otros cinco sitios con la misma forma — `form_data.get("tzOffset") or form_data.get("tz_offset_minutes") or 0` —
y esa forma tiene dos defectos que el SSOT `constants.tz_offset_min_for_form_data` ya había cerrado:

  1. Sin dato asume UTC (0). El SSOT (P3-TZ-FALLBACK-SSOT) dice RD (240) porque el 100 % de la población medida está
     en RD. Un dominicano que genera a las 21:00 recibía el plan estampado desde MAÑANA («Sábado» un viernes).
  2. `or 0` funde «ausente» con «UTC explícito». En el detector de drift de la nevera, un snapshot sin huso frente a
     un perfil vivo en 240 daba «drift mayor» = un viaje que no existió, con escalada agresiva.

Los ocho: el estampado principal de `date`/`day_name` (finalize), `_stamp_missing_day_dates` (días sintéticos),
`constants.chunk_execute_after_ceiling` (techo del chunk), `_resolve_chunk_start_anchor` (ancla), `_snapshot_tz` del
refresh de nevera, `_tz_offset_snapshot` + su re-lectura bajo el lock en `_check_chunk_learning_ready` (resync + push
de «cambio de zona horaria»), y `_persist_fresh_pantry_to_chunks` (+ el default `0` dormido en la firma de
`_get_user_tz_live`). La primera pasada contó cinco; el ratchet por regex de este archivo encontró los otros tres.

Medido antes de tocar (read-only): 0 filas vivas sin huso (2/2 perfiles en 240; 28 canceladas y 2 completadas sin
huso, del camino viejo). Es un cierre de consistencia, no de incidente.

Cada test expresa el comportamiento ESPERADO. Ninguno codifica el defecto como especificación.
"""
from __future__ import annotations

import inspect
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def _body(src: str, header: str) -> str:
    i = src.find(header)
    assert i > 0, header
    j = src.find("\ndef ", i + 10)
    k = src.find("\nasync def ", i + 10)
    ends = [x for x in (j, k) if x > 0]
    return src[i:min(ends)] if ends else src[i:]


# ─────────────────────────── días sintéticos: la misma aritmética que el prompt ───────────────────────────

def test_g52b_dias_sinteticos_sin_huso_caen_al_ssot_no_a_utc():
    import graph_orchestrator as go
    from constants import DEFAULT_TZ_OFFSET_MIN
    assert DEFAULT_TZ_OFFSET_MIN == 240, "el test asume la población medida (RD)"
    # sábado 12-sep 01:00Z = viernes 21:00 en Santo Domingo
    plan = {"days": [{"day": 1}]}
    assert go._stamp_missing_day_dates(plan, {"_plan_start_date": "2026-09-12T01:00:00+00:00"}) == [1]
    assert plan["days"][0]["date"] == "2026-09-11" and plan["days"][0]["day_name"] == "Viernes", (
        "sin huso, el día 1 es el del usuario dominicano (SSOT), no el de UTC")


def test_g52b_utc_explicito_sigue_siendo_utc_en_los_dias_sinteticos():
    import graph_orchestrator as go
    plan = {"days": [{"day": 1}]}
    go._stamp_missing_day_dates(plan, {"_plan_start_date": "2026-09-12T01:00:00+00:00", "tzOffset": 0})
    assert plan["days"][0]["date"] == "2026-09-12" and plan["days"][0]["day_name"] == "Sábado"
    plan = {"days": [{"day": 1}]}
    go._stamp_missing_day_dates(plan, {"_plan_start_date": "2026-09-12T01:00:00+00:00", "tz_offset_minutes": "240"})
    assert plan["days"][0]["date"] == "2026-09-11", "el perfil (string) también se respeta, como en el SSOT"


def test_g52b_el_estampado_principal_usa_el_ssot():
    src = _src("graph_orchestrator.py")
    i = src.find("# Renumerar días y asignar day_name obligatoriamente")
    j = src.find("[DAY NAMES] Inyectados", i)
    assert 0 < i < j
    bloque = src[i:j]
    assert "tz_offset_minutes = _tz_for_fd(form_data)" in bloque
    assert "from constants import tz_offset_min_for_form_data as _tz_for_fd" in bloque
    assert "start_dt = start_dt - timedelta(minutes=tz_offset_minutes)" in bloque, "sin `if tz:` — restar 0 es UTC"
    assert 'form_data.get("tzOffset")' not in bloque and "or 0" not in bloque


# ─────────────────────────── techo del chunk ───────────────────────────

def test_g52b_el_techo_del_chunk_sin_huso_es_el_de_rd_no_el_de_utc():
    from constants import chunk_execute_after_ceiling
    sin = {"form_data": {"_plan_start_date": "2026-08-16T10:00:00"}}
    rd = {"form_data": {"_plan_start_date": "2026-08-16T10:00:00", "tzOffset": 240}}
    utc = {"form_data": {"_plan_start_date": "2026-08-16T10:00:00", "tzOffset": 0}}
    assert chunk_execute_after_ceiling(sin, 0) == chunk_execute_after_ceiling(rd, 0) == datetime(2026, 8, 16, 4, 30, tzinfo=timezone.utc)
    assert chunk_execute_after_ceiling(utc, 0) == datetime(2026, 8, 16, 0, 30, tzinfo=timezone.utc), "UTC explícito es un dato"
    assert chunk_execute_after_ceiling({"form_data": {"_plan_start_date": "2026-08-16T10:00:00", "tzOffset": "x"}}, 0) == \
        chunk_execute_after_ceiling(rd, 0), "basura ⇒ SSOT, no UTC ni excepción"


# ─────────────────────────── refresh de nevera: ausente ≠ UTC ───────────────────────────

def test_g52b_el_drift_del_snapshot_no_confunde_ausencia_con_utc():
    src = _src("cron_tasks.py")
    cuerpo = _body(src, "def _refresh_chunk_pantry_inner(")
    assert "_snapshot_tz = tz_offset_min_for_form_data(" in cuerpo
    assert 'snapshot_form_data.get("tzOffset")' not in cuerpo.replace("snapshot_form_data.get(k)", "")
    assert "or 0" not in cuerpo.split("_snapshot_tz = tz_offset_min_for_form_data(")[0][-600:], "la cadena `or 0` desapareció"
    assert "_live_tz = _get_user_tz_live(user_id, _snapshot_tz)" in cuerpo
    # Identidad por PARSER, no por `is`: otro test de la misma sesión puede recargar `constants` y la identidad de
    # objeto cambia sin que nada esté mal (el flake de `test_detector_terms_are_ssot_shared`, misma familia).
    import cron_tasks
    assert getattr(cron_tasks.tz_offset_min_for_form_data, "__module__", "") == "constants"
    i_imp = src.find("from constants import (")
    assert 0 < i_imp < 5000 and "    tz_offset_min_for_form_data,  # [P1-PLAN-LOTE-9" in src[i_imp:i_imp + 400], (
        "importado a nivel de módulo: un nombre, no cinco imports locales")
    assert "from constants import tz_offset_min_for_form_data as _tz_for_fd" not in src


def test_g52b_el_ancla_y_el_learning_gate_del_chunk_usan_el_ssot():
    src = _src("cron_tasks.py")
    ancla = _body(src, "def _resolve_chunk_start_anchor(")
    assert "snapshot_tz = tz_offset_min_for_form_data(form_data)" in ancla
    assert "snapshot_tz = 0" not in ancla, "el «no sé» explícito es la fuente 4 (`forced_8am_utc`), no un 0 sembrado"
    assert "forced_8am_utc" in ancla, "la fuente 4 sigue existiendo: sin ancla ni huso, se dice"
    gate = _body(src, "def _check_chunk_learning_ready(")
    assert "_tz_offset_snapshot = tz_offset_min_for_form_data(form_data)" in gate
    assert "_p05_fresh_tz = tz_offset_min_for_form_data(_p05_fresh_form)" in gate
    assert "_tz_offset_live = _tz_offset_snapshot" in gate, "sin perfil vivo, el snapshot manda (y ya no es 0)"


def test_g52b_el_persist_del_snapshot_no_fabrica_utc():
    import cron_tasks
    from constants import DEFAULT_TZ_OFFSET_MIN
    cuerpo = _body(_src("cron_tasks.py"), "def _persist_fresh_pantry_to_chunks(")
    assert "fallback_minutes=0" not in cuerpo, "un perfil sin huso persistía 0 en el snapshot: UTC fabricado como dato"
    sig = inspect.signature(cron_tasks._get_user_tz_live)
    assert sig.parameters["fallback_minutes"].default is None, "el default `0` era la 4.ª respuesta dormida en la firma"
    assert cron_tasks._get_user_tz_live("guest") == DEFAULT_TZ_OFFSET_MIN
    assert cron_tasks._get_user_tz_live("", 300) == 300, "un fallback explícito sigue mandando"


# ─────────────────────────── ratchet: la forma `... or 0` no vuelve ───────────────────────────

_CADENA_OR_0 = re.compile(
    r"""\.get\(\s*["'](?:tzOffset|tz_offset_minutes)["']\s*\)\s*or\s*[\w.]+\.get\(\s*["'](?:tzOffset|tz_offset_minutes)["']\s*\)"""
    r"""(?:\s*or\s*[\w.]+\.get\([^)]*\))*\s*or\s*0\b""",
    re.S,
)


@pytest.mark.parametrize("rel", ["graph_orchestrator.py", "constants.py", "cron_tasks.py", "services.py", "tools.py",
                                 "routers/plans.py", "routers/user_data.py", "routers/diary.py", "proactive_agent.py"])
def test_g52b_ningun_productivo_resuelve_el_huso_con_or_0(rel):
    p = _BACKEND / rel
    if not p.exists():
        pytest.skip(rel)
    src = p.read_text(encoding="utf-8")
    m = _CADENA_OR_0.search(src)
    assert m is None, f"{rel}: vuelve la cadena `... or 0` para el huso — usa constants.tz_offset_min_for_form_data:\n{m.group(0)}"


def test_marker_bumpeado():
    import app
    assert "[P1-PLAN-LOTE-9 · 2026-09-11]" in _src("app.py")
    assert app._LAST_KNOWN_PFIX.startswith("P1-PLAN-") and "2026-09-11" in app._LAST_KNOWN_PFIX
