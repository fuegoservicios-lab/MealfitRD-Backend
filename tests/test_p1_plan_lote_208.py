# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-208 · 2026-09-24] «Modo automático» cumple lo que promete: no pausamos por falta de registros.

Configuración promete «No pausaremos tu plan aunque dejes de registrar comidas»; el gate sólo lo honraba con ≥ N
descuentos de la Nevera por consumo. Sin registros ni descuentos —el plan de 30 días del dueño— el bloque siguiente se
pausaba 6 h con «Loguea tus comidas para continuar», y el nudge diario amenazaba con la pausa. Ver `modo_automatico.py`.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import modo_automatico as ma  # noqa: E402


def test_activo():
    assert ma.activo("u", preferencia="auto_proxy") is True
    assert ma.activo("u", preferencia="manual") is False
    assert ma.activo("u", MagicMock(return_value={"logging_preference": "auto_proxy"})) is True
    assert ma.activo("u", MagicMock(return_value={"logging_preference": "manual"})) is False
    assert ma.activo("u", MagicMock(return_value=None)) is False
    assert ma.activo("u", MagicMock(side_effect=RuntimeError("db"))) is False, "fail-closed: conducta de siempre"


def test_knob_apagado(monkeypatch):
    monkeypatch.setenv("MEALFIT_AUTO_PROXY_HONORED", "false")
    assert ma.activo("u", preferencia="auto_proxy") is False
    assert ma.filtro_nudge() == ""


def _gate_sin_registros(preferencia: str):
    """Bloque previo de 3 días ya vivido, CERO registros y CERO descuentos de la Nevera (el caso del dueño)."""
    import cron_tasks
    now = datetime.now(timezone.utc).replace(hour=4, minute=31, second=0, microsecond=0)
    hoy = now.date()
    ancla = datetime.combine(hoy - timedelta(days=3), datetime.min.time(), tzinfo=timezone.utc)
    pd = {"days": [{"day": k + 1, "date": (hoy - timedelta(days=3 - k)).isoformat(),
                    "meals": [{"meal": t, "name": f"{t} {k}"} for t in ("Desayuno", "Almuerzo", "Cena")]}
                   for k in range(3)]}
    snapshot = {"form_data": {"_plan_start_date": ancla.isoformat(), "tz_offset_minutes": 0}, "totalDays": 30}

    def _sql(q, *a, **k):
        if "days_offset, days_count" in q:
            return {"days_offset": 0, "days_count": 3}
        if "SELECT days_count FROM plan_chunk_queue" in q:
            return {"days_count": 3}
        if "logging_preference FROM user_profiles" in q:
            return {"health_profile": {"tz_offset_minutes": 0}, "logging_preference": preferencia}
        return None

    with patch("cron_tasks._dt_p0b_now", return_value=now), \
            patch("cron_tasks.execute_sql_query", side_effect=_sql), \
            patch("cron_tasks.execute_sql_write", MagicMock()), \
            patch("cron_tasks._record_chunk_deferral", MagicMock()), \
            patch("cron_tasks._dispatch_push_notification", MagicMock()), \
            patch("cron_tasks.get_consumed_meals_since", return_value=[]), \
            patch("cron_tasks.get_inventory_activity_since", return_value={"consumption_mutations_count": 0}), \
            patch("db_facts.get_plan_meal_deviations_since", return_value=[]):
        return cron_tasks._check_chunk_learning_ready(user_id="u", meal_plan_id="p", week_number=2, days_offset=3,
                                                      plan_data=pd, snapshot=snapshot)


def test_modo_automatico_no_pausa_sin_registros():
    r = _gate_sin_registros("auto_proxy")
    assert r.get("zero_log_proxy") is True, r
    assert r.get("ready") is True and r.get("auto_proxy_used") is True, r
    assert r.get("learning_signal_strength") == "weak", r


def test_modo_manual_sigue_como_siempre():
    r = _gate_sin_registros("manual")
    assert r.get("ready") is False and r.get("zero_log_proxy") is True and not r.get("auto_proxy_used"), r


def test_el_nudge_no_amenaza_a_quien_eligio_no_loguear():
    import cron_tasks
    capturado = {}

    def _q(sql, params=None, **k):
        capturado["sql"], capturado["params"] = sql, params
        return []

    with patch("cron_tasks.execute_sql_query", side_effect=_q), patch("cron_tasks.execute_sql_write"), \
            patch("cron_tasks._dispatch_push_notification"):
        cron_tasks._nudge_chronic_zero_log_users()
    assert "logging_preference, 'manual') <> 'auto_proxy'" in capturado["sql"], capturado["sql"]
    assert len(capturado["params"]) == 3, "los parámetros no cambian"


def test_anclas_worker_y_recovery():
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    i = src.index('if learning_ready.get("auto_proxy_used"):')
    assert 'form_data["_force_variety"] = True' in src[i:i + 200]
    assert '_zl_auto = __import__("modo_automatico").activo(user_id_str, execute_sql_query)' in src
    assert "if _zl_mutations >= CHUNK_LEARNING_INVENTORY_PROXY_MIN_MUTATIONS or _zl_auto:" in src


def test_la_promesa_de_la_interfaz_sigue_escrita():
    """Si alguien cambia el texto de Configuración, este lote deja de describir lo que promete la app."""
    ui = (_BACKEND.parent / "frontend" / "src" / "pages" / "Settings.jsx")
    if ui.exists():
        assert "No pausaremos tu plan aunque dejes de registrar comidas." in ui.read_text(encoding="utf-8")


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 208 and m.group(2) >= "2026-09-24"
