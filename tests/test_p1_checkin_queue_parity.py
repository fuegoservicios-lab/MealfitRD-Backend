"""[P1-CHECKIN-QUEUE-PARITY · 2026-09-02] La cola calibra con los check-ins igual que el SSE.

Medido: el bloque P1-ADAPTIVE-RENEWAL + P1-CHECKIN-SIGNALS-GATE vivía SOLO dentro de
`api_analyze_stream` (SSE). El preparador común de la cola (`build_initial_pipeline_inputs`)
solo llamaba al JIT de `inject_learning_signals_from_profile`, que lee `weight_log` (el
registro del diario), no `health_profile.weight_history` (donde escribe el check-in) ni
`_renewal_checkins`. Desde el flip global, los check-ins no movían una caloría.

Tooltip-anchor: P1-CHECKIN-QUEUE-PARITY | inject_adaptive_renewal_from_profile
"""
import json
from pathlib import Path
from unittest.mock import patch

import generation_inputs as gi

SRC = Path(gi.__file__).read_text(encoding="utf-8")


def _hp(wh, ck=None):
    hp = {"weight": 135, "weight_history": wh}
    if ck is not None:
        hp["_renewal_checkins"] = ck
    return {"health_profile": json.dumps(hp)}


def test_injects_history_and_last_checkin_when_engine_can_activate(monkeypatch):
    monkeypatch.delenv("MEALFIT_ADAPTIVE_RENEWAL_INJECT", raising=False)
    wh = [{"date": "2026-08-01", "weight": 140, "unit": "lb"}, {"date": "2026-09-02", "weight": 135, "unit": "lb"}]
    ck = [{"date": "2026-08-01", "hunger": 4}, {"date": "2026-09-02", "hunger": 2, "energy": 4, "adherence_pct": 80}]
    pd = {}
    with patch("db_core.execute_sql_query", return_value=_hp(wh, ck)):
        gi.inject_adaptive_renewal_from_profile("u1", pd)
    assert pd["weight_history"] == wh
    assert pd["_renewal_signals"] == {"hunger": 2, "energy": 4, "adherence_pct": 80}


def test_single_weight_injects_nothing():
    pd = {}
    with patch("db_core.execute_sql_query", return_value=_hp([{"date": "2026-09-02", "weight": 135}], [{"hunger": 5}])):
        gi.inject_adaptive_renewal_from_profile("u1", pd)
    assert pd == {}, "sin 2+ registros el motor no activa: no inyectar ruido ni señales"


def test_knob_off_and_guest_are_noops(monkeypatch):
    monkeypatch.setenv("MEALFIT_ADAPTIVE_RENEWAL_INJECT", "false")
    pd = {}
    with patch("db_core.execute_sql_query", side_effect=AssertionError("no debe consultar")):
        gi.inject_adaptive_renewal_from_profile("u1", pd)
        gi.inject_adaptive_renewal_from_profile(None, pd)
    assert pd == {}


def test_fail_open_on_db_error():
    pd = {"x": 1}
    with patch("db_core.execute_sql_query", side_effect=RuntimeError("pool")):
        assert gi.inject_adaptive_renewal_from_profile("u1", pd) == {"x": 1}


def test_queue_builder_calls_it_before_the_jit():
    i = SRC.index("inject_adaptive_renewal_from_profile(actual_user_id, pipeline_data)")
    j = SRC.index("inject_learning_signals_from_profile(actual_user_id, pipeline_data)")
    assert 0 < i < j, "el historial del perfil (check-ins) debe mandar sobre el JIT de weight_log"
    assert "MEALFIT_ADAPTIVE_RENEWAL_INJECT" in SRC, "mismo knob de rollback que el SSE"
