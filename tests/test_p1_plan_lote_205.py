# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-205 · 2026-09-24] Un bloque ya pausado no graba una fila de aplazamiento por minuto.

Producción: 2.707 filas `temporal_gate` en 4 días para un solo usuario (~44/h): `_recover_pantry_paused_chunks`
re-evalúa el gate de cada bloque pausado en cada tick, y cada evaluación insertaba en `chunk_deferrals` (y la alerta
`chronic_deferrals` contaba esas filas). La pausa ya queda registrada y ya envió su push.
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


def _evaluar(snapshot_extra: dict):
    """Bloque previo que termina PASADO MAÑANA ⇒ el gate difiere (con 1 día saltaría además el aviso de diversidad de
    la Nevera, que es otro push y se envía una sola vez); devuelve (resultado, mock de la fila, mock del push)."""
    import cron_tasks
    now = datetime.now(timezone.utc).replace(hour=0, minute=30, second=0, microsecond=0)
    hoy = now.replace(hour=0, minute=0)
    manana = (now + timedelta(days=2)).date().isoformat()
    snapshot = {"form_data": {"_plan_start_date": hoy.isoformat(), "tz_offset_minutes": 0}, "totalDays": 4,
                "_temporal_gate_retries": 4}          # el siguiente sería el del push (PUSH_AT_RETRY = 5)
    snapshot.update(snapshot_extra)

    def _sql(q, *a, **k):
        return {"days_offset": 0, "days_count": 3} if "days_offset, days_count" in q else None

    fila, push = MagicMock(), MagicMock()
    with patch("cron_tasks._dt_p0b_now", return_value=now), \
            patch("cron_tasks.execute_sql_query", side_effect=_sql), \
            patch("cron_tasks.execute_sql_write", MagicMock()), \
            patch("cron_tasks._record_chunk_deferral", fila), \
            patch("cron_tasks._dispatch_push_notification", push), \
            patch("cron_tasks.CHUNK_TEMPORAL_GATE_PUSH_AT_RETRY", 5):
        r = cron_tasks._check_chunk_learning_ready(user_id="u", meal_plan_id="p", week_number=8, days_offset=1,
                                                   plan_data={"days": [{"day": 1, "date": manana, "meals": []}]},
                                                   snapshot=snapshot)
    return r, fila, push


def test_bloque_pausado_no_graba_fila_ni_push():
    r, fila, push = _evaluar({"_pantry_pause_reason": "prev_chunk_not_concluded"})
    assert r.get("reason") == "prev_chunk_day_not_yet_elapsed", r
    assert not fila.called and not push.called


def test_bloque_sin_pausa_sigue_grabando():
    r, fila, push = _evaluar({})
    assert r.get("reason") == "prev_chunk_day_not_yet_elapsed", r
    assert fila.called, "el aplazamiento normal (antes de pausar) sigue quedando registrado"
    assert push.called, "y el push del N-ésimo aplazamiento también"


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 205 and m.group(2) >= "2026-09-24"
