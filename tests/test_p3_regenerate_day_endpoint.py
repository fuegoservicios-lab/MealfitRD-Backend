"""[P3-PANTRY-SUFFICIENCY · 2026-06-23] Contrato del endpoint `regenerate-day`.

El botón "actualizar el día completo" debe: cocinar desde la Nevera (loop de swaps
pantry-strict), bloquear con soft-fail si la Nevera no cubre el día (sin consumir
cuota), persistir atómicamente `days[i].meals[*]` (I6/I7), reservar inventario entre
platos (D7), y cobrar 1 crédito SOLO al final. Test parser-based sobre el handler
(Depends/LLM/DB difíciles de invocar en unit) — ancla el ORDEN y las invariantes."""
import re
from pathlib import Path

import pytest

_PLANS = (Path(__file__).resolve().parent.parent / "routers" / "plans.py").read_text(encoding="utf-8")


def _extract(fn: str) -> str:
    m = re.search(rf"^def {re.escape(fn)}\(", _PLANS, re.MULTILINE)
    assert m, f"{fn} no encontrada"
    start = m.start()
    nxt = re.search(r"^(def |@router\.)", _PLANS[start + 1:], re.MULTILINE)
    return _PLANS[start: start + 1 + nxt.start()] if nxt else _PLANS[start:]


@pytest.fixture(scope="module")
def body() -> str:
    return _extract("api_regenerate_day")


def test_route_registered():
    assert '@router.post("/{plan_id}/regenerate-day")' in _PLANS, "ruta regenerate-day ausente"


def test_ownership_filter(body: str):
    assert re.search(r"FROM meal_plans WHERE id = %s AND user_id = %s", body), \
        "el handler debe filtrar ownership (id + user_id) — I2"
    assert "Crea tu cuenta" in body, "debe bloquear a invitados"


def test_sufficiency_gate_day_scope_before_quota(body: str):
    assert "evaluate_pantry_sufficiency" in body and 'scope="day"' in body, \
        "debe correr el gate de suficiencia con scope=day"
    assert "_pantry_sufficiency_gate_on()" in body, "el gate debe estar detrás del knob master"
    gate_idx = body.find("evaluate_pantry_sufficiency")
    quota_idx = body.find('log_api_usage(user_id, "llm_regenerate_day")')
    assert gate_idx > 0 and quota_idx > 0
    assert gate_idx < quota_idx, "el gate debe preceder al cobro de cuota (no descontar si insuficiente)"


def test_softfail_no_quota_when_insufficient(body: str):
    # El soft-fail (regen_failed) y el return cuando nada se regenera preceden a log_api_usage.
    assert '"regen_failed": True' in body, "soft-fail debe usar regen_failed (HTTP 200, no 4xx)"
    assert "pantry_insufficient_for_goal" in body
    first_regen_failed = body.find('"regen_failed": True')
    quota_idx = body.find('log_api_usage(user_id, "llm_regenerate_day")')
    assert first_regen_failed < quota_idx, "el soft-fail no debe llegar a cobrar cuota"


def test_loop_swaps_strict_pantry_with_reservation(body: str):
    # [P1-SWAP-SAMEDAY-PROTEIN-GATE · 2026-07-10] la iteración usa la variante con
    # exclusiones (_form_v) y el fallback relajado (_form_relaxed); meal_form es la base.
    assert "swap_meal(_form_v)" in body, "debe iterar swap_meal por plato (variante con variedad)"
    assert "swap_meal(_form_relaxed)" in body, "fallback de factibilidad sin gates"
    assert "current_pantry_ingredients" in body, "cada swap debe recibir la Nevera como restricción"
    assert "_decrement_ledger_by_meal(ledger" in body, "debe reservar inventario entre platos (D7)"
    assert "_inventory_grams_ledger(" in body


def test_atomic_persist_and_strip_lists(body: str):
    assert "update_plan_data_atomic(plan_id, _day_mutator, user_id=verified_user_id)" in body, \
        "persistencia debe ser atómica con user_id (I6/I7)"
    assert '_day["meals"] = new_meals' in body, "el mutator debe reemplazar los meals del día"
    for k in ("aggregated_shopping_list", "aggregated_shopping_list_weekly"):
        assert k in body, f"debe strippear {k} para forzar recalc"
    assert "_plan_modified_at" in body, "debe bumpear _plan_modified_at"


def test_quota_one_credit_after_success(body: str):
    assert 'log_api_usage(user_id, "llm_regenerate_day")' in body, "1 crédito por día (D3)"
    # El cobro va DESPUÉS del persist atómico (post-éxito).
    persist_idx = body.find("update_plan_data_atomic(plan_id, _day_mutator")
    quota_idx = body.find('log_api_usage(user_id, "llm_regenerate_day")')
    assert persist_idx < quota_idx, "la cuota se cobra tras persistir (no antes)"
