"""[P1-MARKER-REGEN-BUDGET-GATE · 2026-07-09] Un plan YA APROBADO por el reviewer no debe
arriesgar un timeout TOTAL por el surgical regen post-aprobación cuando queda poco budget.

Fallo vivo corr=08f60b00 (2026-07-09): reviewer aprobó el plan a los ~590s, pero el
surgical regen del Día 2 (`_critique_unresolved`) con PRO fallback corrió otros ~5min →
cruzó el wall-timeout (900s) → cancelación DURA → el KV se marcó failed → el usuario NO
recibió NADA (peor que entregar el plan aprobado con una repetición menor).

Fix: si el plan está aprobado y `remaining < MARKER_REGEN_MIN_BUDGET_S`, entregar el
aprobado (marcado _quality_degraded minor) en vez de rutear al surgical regen.

Función pura (should_retry, sin DB — el alert se mockea). tooltip-anchor: P1-MARKER-REGEN-BUDGET-GATE
"""
import time
import graph_orchestrator as go


def _approved_state_with_marker(pipeline_start):
    return {
        "review_passed": True,
        "_marker_regen_attempted": False,
        "pipeline_start": pipeline_start,
        "attempt": 1,
        "plan_result": {
            "days": [
                {"day": 1, "meals": []},
                {"day": 2, "meals": [],
                 "_critique_unresolved": {"slot_coherence": "almuerzo↔cena comparten carbo"}},
            ]
        },
    }


def test_low_budget_delivers_approved_not_marker_regen(monkeypatch):
    monkeypatch.setattr(go, "_emit_plan_quality_degraded_alert", lambda *a, **k: None)  # sin DB
    monkeypatch.setattr(go, "GLOBAL_PIPELINE_TIMEOUT_S", 720)
    monkeypatch.setattr(go, "MARKER_REGEN_MIN_BUDGET_S", 240, raising=False)
    # arrancó hace 700s → remaining = 720-700 = 20s < 240 → NO arriesgar el regen
    st = _approved_state_with_marker(time.time() - 700)
    route = go.should_retry(st)
    assert route == "end"                                    # entrega el aprobado
    assert st["plan_result"]["_quality_degraded"] is True    # honestidad: banner
    assert st["plan_result"]["_quality_degraded_severity"] == "minor"


def test_high_budget_still_routes_to_marker_regen(monkeypatch):
    monkeypatch.setattr(go, "GLOBAL_PIPELINE_TIMEOUT_S", 720)
    monkeypatch.setattr(go, "MARKER_REGEN_MIN_BUDGET_S", 240, raising=False)
    # recién arrancó → remaining ~710 >> 240 → comportamiento existente: surgical regen
    st = _approved_state_with_marker(time.time() - 10)
    assert go.should_retry(st) == "marker_regen"


def test_knob_zero_disables_gate(monkeypatch):
    monkeypatch.setattr(go, "GLOBAL_PIPELINE_TIMEOUT_S", 720)
    monkeypatch.setattr(go, "MARKER_REGEN_MIN_BUDGET_S", 0, raising=False)  # gate OFF
    st = _approved_state_with_marker(time.time() - 700)                     # budget bajo pero gate OFF
    assert go.should_retry(st) == "marker_regen"


def test_no_pipeline_start_no_gate(monkeypatch):
    monkeypatch.setattr(go, "GLOBAL_PIPELINE_TIMEOUT_S", 720)
    monkeypatch.setattr(go, "MARKER_REGEN_MIN_BUDGET_S", 240, raising=False)
    st = _approved_state_with_marker(None)                                  # sin pipeline_start → sin gate
    assert go.should_retry(st) == "marker_regen"


def test_observed_prod_scenario_default_catches_it(monkeypatch):
    """Reproduce el fallo vivo corr=08f60b00 con la config REAL de prod
    (GLOBAL_PIPELINE_TIMEOUT_S=900) y el DEFAULT del umbral (400s, NO monkeypatched):
    aprobado a ~590s → remaining 310s < 400 → el gate DEBE entregar el aprobado."""
    monkeypatch.setattr(go, "_emit_plan_quality_degraded_alert", lambda *a, **k: None)
    monkeypatch.setattr(go, "GLOBAL_PIPELINE_TIMEOUT_S", 900)   # valor REAL de prod
    # NO tocar MARKER_REGEN_MIN_BUDGET_S → usa el default del módulo (400)
    assert go.MARKER_REGEN_MIN_BUDGET_S == 400                  # ancla el default
    st = _approved_state_with_marker(time.time() - 590)         # aprobado a 590s (como el fallo real)
    assert go.should_retry(st) == "end"                         # entrega el aprobado, no timeout total
    assert st["plan_result"]["_quality_degraded"] is True
