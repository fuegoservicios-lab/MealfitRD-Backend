# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-207 · 2026-09-24] Esperar al bloque previo sin quemar intentos ni pedir registros.

Producción: un bloque recogido antes de que terminara el anterior salía del worker en `processing`; lo devolvía el
rescate de zombies a los 10 min sumando un intento (bloques 5, 6 y 8 de 3957a669 terminaron con attempts = 5, el tope
que los marca `failed`), re-evaluado cada ~16 min. Y sin la marca proactiva caía en el aplazamiento de aprendizaje: +12 h
y «Tu próximo bloque espera más feedback… loguea tus comidas» (4 de 4 medidos eran `temporal_gate`). Ver
`fin_bloque_previo.py`.
"""
from __future__ import annotations

import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import fin_bloque_previo as fbp  # noqa: E402


def test_frontera_es_la_medianoche_local_del_dia_siguiente():
    assert fbp.frontera_utc(date(2026, 9, 25), 240) == datetime(2026, 9, 26, 4, 30, tzinfo=timezone.utc), "RD"
    assert fbp.frontera_utc(date(2026, 9, 25), -120) == datetime(2026, 9, 25, 22, 30, tzinfo=timezone.utc), "España"
    assert fbp.frontera_utc(date(2026, 9, 25), 240, 1) == datetime(2026, 9, 25, 4, 30, tzinfo=timezone.utc), "margen"
    assert fbp.frontera_utc(None, 240) is None


def test_knob_apagado(monkeypatch):
    monkeypatch.setenv("MEALFIT_TEMPORAL_GATE_WAIT_BOUNDARY", "false")
    assert fbp.frontera_utc(date(2026, 9, 25), 240) is None
    assert not fbp.es_espera_de_calendario({"reason": "prev_chunk_day_not_yet_elapsed"})
    escribir = MagicMock()
    fbp.soltar_a_pendiente("t", escribir)
    assert not escribir.called


def test_espera_de_calendario_y_soltar():
    assert fbp.es_espera_de_calendario({"reason": "prev_chunk_day_not_yet_elapsed"})
    assert not fbp.es_espera_de_calendario({"reason": "learning_proxy_exhausted"})
    assert not fbp.es_espera_de_calendario({"ready": False, "zero_log_proxy": True})
    escribir = MagicMock()
    fbp.soltar_a_pendiente("chunk-1", escribir)
    sql, args = escribir.call_args[0]
    assert "status = 'pending'" in sql and "status = 'processing'" in sql and args == ("chunk-1",)


def test_el_gate_programa_el_bloque_en_la_frontera():
    """El caso de dea00a2f (23-sep): recogido a las 11:28 con el bloque previo terminando dos días después."""
    import cron_tasks
    now = datetime.now(timezone.utc).replace(hour=15, minute=28, second=0, microsecond=0)
    hoy = now.date()
    ancla = datetime.combine(hoy, datetime.min.time(), tzinfo=timezone.utc)
    pd = {"days": [{"day": k + 1, "date": (hoy + timedelta(days=k)).isoformat(), "meals": []} for k in range(3)]}
    snapshot = {"form_data": {"_plan_start_date": ancla.isoformat(), "tz_offset_minutes": 0}, "totalDays": 15}

    def _sql(q, *a, **k):
        return {"days_offset": 0, "days_count": 3} if "days_offset, days_count" in q else None

    escribir = MagicMock()
    with patch("cron_tasks._dt_p0b_now", return_value=now), \
            patch("cron_tasks.execute_sql_query", side_effect=_sql), \
            patch("cron_tasks.execute_sql_write", escribir), \
            patch("cron_tasks._record_chunk_deferral", MagicMock()), \
            patch("cron_tasks._dispatch_push_notification", MagicMock()):
        r = cron_tasks._check_chunk_learning_ready(user_id="u", meal_plan_id="p", week_number=2, days_offset=3,
                                                   plan_data=pd, snapshot=snapshot)
    assert r.get("reason") == "prev_chunk_day_not_yet_elapsed" and r.get("days_until_prev_end") == 2, r
    upd = [c for c in escribir.call_args_list if "_temporal_gate_retries" in c.args[0] and "GREATEST" in c.args[0]]
    assert upd, "el UPDATE del gate acota execute_after por abajo con la frontera"
    frontera = upd[0].args[1][2]
    assert frontera == datetime.combine(hoy + timedelta(days=3), datetime.min.time(), tzinfo=timezone.utc) + \
        timedelta(minutes=30), frontera


def test_anclas_en_el_worker():
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    generico = src.index("if learning_ready_deferrals < CHUNK_LEARNING_READY_MAX_DEFERRALS and not __import__(")
    assert '"fin_bloque_previo").es_espera_de_calendario(learning_ready)' in src[generico:generico + 200]
    backoff = src.index("execute_after ya bumpeado por el gate — no pausamos aún.")
    assert '__import__("fin_bloque_previo").soltar_a_pendiente(task_id, execute_sql_write)' in \
        src[backoff:backoff + 300]


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 207 and m.group(2) >= "2026-09-24"
