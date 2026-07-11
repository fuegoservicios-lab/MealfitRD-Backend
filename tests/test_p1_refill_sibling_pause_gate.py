"""[P1-REFILL-SIBLING-PAUSE-GATE · 2026-07-10] Un refill/catchup NO genera (cero LLM) si el
plan ya tiene un chunk en `pending_user_action` — pausa pre-pipeline sin push duplicado.

Caso vivo (plan 8ec367f8, 2026-07-11 01:04): chunk 2 generó con 3 intentos LLM (~$0.05) y
LUEGO pausó en reservas 0/16; el cron rolling encolaba la semana siguiente cada día →
genera-para-descartar diario en cuentas con nevera desatendida. El restock del usuario
resume los pausados (P0-4 recovery) y el gate deja de disparar. initial_plan exento
(P1-CHUNK-AUTONOMY).

tooltip-anchor: P1-REFILL-SIBLING-PAUSE-GATE
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))
_CRON = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Gate: posición, scope de kinds, knob, cero-LLM (return pre-pipeline)
# ---------------------------------------------------------------------------

def test_gate_wired_early_in_worker():
    i = _CRON.find("refill/catchup: si el plan YA tiene un")
    assert i > 0, "el gate desapareció del worker"
    blk = _CRON[i: i + 2600]
    assert 'chunk_kind in ("rolling_refill", "catchup")' in blk, "solo refill/catchup — initial exento"
    assert '_env_bool("MEALFIT_REFILL_SIBLING_PAUSE_GATE", True)' in blk, "knob default ON"
    assert "status = 'pending_user_action' AND id != %s" in blk, "cuenta siblings pausados (excluye el propio)"
    assert "notify=False" in blk, "sin push duplicado (el sibling ya notificó)"
    assert "return" in blk, "pausa PRE-pipeline: el worker retorna sin generar"
    # el gate corre ANTES del deepcopy del form_data (trabajo pesado del worker)
    i_form = _CRON.find('form_data = copy.deepcopy(snap.get("form_data", {}))', i)
    assert 0 < i < i_form, "el gate debe evaluarse antes de preparar el pipeline"


def test_gate_runs_before_pipeline_invocation():
    i = _CRON.find("refill/catchup: si el plan YA tiene un")
    i_pipe = _CRON.find("Generando chunk", i)
    assert i_pipe > i > 0, "el gate vive aguas arriba de la generación del chunk"


# ---------------------------------------------------------------------------
# 2. notify param del pause helper (funcional con stubs)
# ---------------------------------------------------------------------------

def _call_pause(notify):
    import cron_tasks as ct
    pushes = []
    with patch.object(ct, "execute_sql_query", return_value={"pipeline_snapshot": {}}), \
         patch.object(ct, "_cas_pause_chunk_to_pending_user_action", return_value=True), \
         patch.object(ct, "_dispatch_push_notification", side_effect=lambda **kw: pushes.append(kw)):
        ct._pause_chunk_for_pantry_refresh("t1", "u1", 2, [], reason="empty_pantry", notify=notify)
    return pushes


def test_pause_helper_notify_false_suppresses_push():
    assert _call_pause(notify=False) == [], "notify=False no debe despachar push"


def test_pause_helper_default_still_notifies():
    pushes = _call_pause(notify=True)
    assert len(pushes) == 1 and pushes[0].get("user_id") == "u1", (
        "el default (pausa primaria) sigue notificando exactamente una vez"
    )


# ---------------------------------------------------------------------------
# 3. Marker anclado en source
# ---------------------------------------------------------------------------

def test_marker_anchored_in_source():
    assert _CRON.count("P1-REFILL-SIBLING-PAUSE-GATE") >= 3
    app_src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert "P1-REFILL-SIBLING-PAUSE-GATE" in app_src
