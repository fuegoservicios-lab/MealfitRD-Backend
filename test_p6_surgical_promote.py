"""[P6-SURGICAL-PROMOTE] Tests para la promoción del plan post-surgical
a `_best_attempt_plan` snapshot.

Bug observable (corrida 2026-05-05 13:11-13:12):
  Self-critique timeouteó Días 2 y 3 → markers `_critique_unresolved`.
  Reviewer #1 APROBÓ (lente médico no detecta slot violations).
  P5-MARKER-APPROVED-1 disparó surgical_marker_regen → fixed Días 2,3
  exitosamente (provider recovered, 14s + 33s).
  Re-assemble + Reviewer #2 → RECHAZADO con minor en Día 1 (issue ortogonal:
  recipe menciona 'cerdo' pero ingredients tras auto-patch no lo tenía).
  P0-PIPE-1 comparó:
    - best_snapshot (pre-surgical): approved con markers (rank=0)
    - current (post-surgical): minor sin markers (rank=1)
  Y RESTAURÓ el snapshot pre-surgical → usuario recibió plan con
  violaciones de slot que P5 había resuelto. P5 fue invalidado en
  silencio.

Causa raíz:
  `_attempt_quality_rank` solo considera review_passed + severity.
  No considera `_critique_unresolved` markers como señal de calidad.
  Aplicar fix ahí cambia la semántica de comparación globalmente
  (riesgo). Más cirujano: cuando surgical regen fixea markers, promover
  el nuevo plan al snapshot inmediatamente, antes de que re-review
  decida algo.

Fix:
  En `surgical_marker_regen_node`, si `fixed_count > 0`, sobrescribir
  `_best_attempt_*` con el plan post-surgical y status approved. Si
  re-review subsecuente demote, P0-PIPE-1 ya tendrá el snapshot mejor.

Cobertura:
  - fixed_count > 0 → snapshot promovido
  - fixed_count == 0 → snapshot NO promovido (defensa: si nada se fixeó
    no tenemos garantía de mejora, dejamos snapshot intacto)
  - Sin markers (fast-path) → snapshot intacto, solo flag set
  - El plan promovido es deepcopy (mutaciones downstream no contaminan)
  - _best_attempt_number sale del state.attempt (no hardcoded)
"""
import asyncio
import copy
from unittest.mock import patch

import pytest


def _state_with_markers(marker_day_nums=(2, 3)):
    """State minimal con markers en los días indicados."""
    days = [{"day": n, "meals": []} for n in (1, 2, 3)]
    for n in marker_day_nums:
        for d in days:
            if d["day"] == n:
                d["_critique_unresolved"] = {
                    "reason": "timeout",
                    "issue": "slot violation",
                    "attempt": 1,
                }
    return {
        "attempt": 1,
        "plan_result": {
            "days": days,
            "_skeleton": {"days": []},
        },
        "form_data": {},
        "_marker_regen_attempted": False,
    }


# ---------------------------------------------------------------------------
# 1. Promotion happens when fixed_count > 0
# ---------------------------------------------------------------------------
def test_snapshot_promoted_when_at_least_one_fix_succeeds():
    """[P6-SURGICAL-PROMOTE] Caso del fix: regen tuvo éxito en al menos
    1 día → _best_attempt_plan se sobrescribe con el plan post-surgical."""
    import graph_orchestrator as go

    state = _state_with_markers((2, 3))

    # Mock el corrector para devolver corrección exitosa instant
    async def fake_invoke(_llm, _prompt, timeout=None):
        from schemas import SingleDayPlanModel
        # Crear un día corregido mínimo válido
        return SingleDayPlanModel.model_construct(
            day=0,
            day_name="X",
            meals=[],
            macros={"protein": "0g", "carbs": "0g", "fats": "0g"},
            calories=2000,
        )

    with patch.object(go, "_safe_ainvoke", side_effect=fake_invoke), \
         patch.object(go, "_route_model", return_value="gemini-3-flash-preview"), \
         patch.object(go, "ChatGoogleGenerativeAI"), \
         patch.object(go, "_build_shared_context", return_value={
             "nutrition_context_minimal": "...",
         }):
        result = asyncio.run(go.surgical_marker_regen_node(state))

    assert result.get("_marker_regen_attempted") is True
    assert "_best_attempt_plan" in result, (
        "Cuando fixed_count > 0, _best_attempt_plan debe ser promovido"
    )
    assert result.get("_best_attempt_severity") == "approved"
    assert result.get("_best_attempt_review_passed") is True
    assert result.get("_best_attempt_reasons") == []
    assert result.get("_best_attempt_number") == 1


# ---------------------------------------------------------------------------
# 2. NO promotion when nothing was fixed
# ---------------------------------------------------------------------------
def test_snapshot_NOT_promoted_when_zero_fixes():
    """Si TODAS las re-correcciones fallan (timeout otra vez, CB open,
    etc.), no podemos garantizar que el nuevo plan sea mejor — dejamos
    el snapshot pre-surgical intacto."""
    import graph_orchestrator as go

    state = _state_with_markers((2, 3))

    # Mock corrector para retornar None siempre (LLM falló)
    async def fake_invoke_none(_llm, _prompt, timeout=None):
        return None

    with patch.object(go, "_safe_ainvoke", side_effect=fake_invoke_none), \
         patch.object(go, "_route_model", return_value="gemini-3-flash-preview"), \
         patch.object(go, "ChatGoogleGenerativeAI"), \
         patch.object(go, "_build_shared_context", return_value={
             "nutrition_context_minimal": "...",
         }):
        result = asyncio.run(go.surgical_marker_regen_node(state))

    assert result.get("_marker_regen_attempted") is True
    assert "_best_attempt_plan" not in result, (
        "Sin fixes, NO debemos sobrescribir el snapshot — el pre-surgical "
        "puede ser mejor que un plan que falló en regen"
    )


# ---------------------------------------------------------------------------
# 3. Fast-path (no markers) — no promotion
# ---------------------------------------------------------------------------
def test_no_promotion_in_fast_path_no_markers():
    """Si no hay markers (defensa contra bug en should_retry), no hay
    nada que regenerar y no debemos tocar el snapshot."""
    import graph_orchestrator as go

    state = {
        "plan_result": {"days": [{"day": 1}, {"day": 2}]},  # sin markers
        "form_data": {},
    }
    result = asyncio.run(go.surgical_marker_regen_node(state))

    assert result == {"_marker_regen_attempted": True}
    assert "_best_attempt_plan" not in result


# ---------------------------------------------------------------------------
# 4. Promoted plan is a deepcopy
# ---------------------------------------------------------------------------
def test_promoted_plan_is_deepcopy():
    """El snapshot promovido debe ser deepcopy del plan_result post-surgical
    para que mutaciones downstream (re-assemble, re-review auto-patch)
    NO contaminen el snapshot."""
    import graph_orchestrator as go

    state = _state_with_markers((2,))

    async def fake_invoke(_llm, _prompt, timeout=None):
        from schemas import SingleDayPlanModel
        return SingleDayPlanModel.model_construct(
            day=0,
            day_name="X",
            meals=[],
            macros={"protein": "0g", "carbs": "0g", "fats": "0g"},
            calories=2000,
        )

    with patch.object(go, "_safe_ainvoke", side_effect=fake_invoke), \
         patch.object(go, "_route_model", return_value="gemini-3-flash-preview"), \
         patch.object(go, "ChatGoogleGenerativeAI"), \
         patch.object(go, "_build_shared_context", return_value={
             "nutrition_context_minimal": "...",
         }):
        result = asyncio.run(go.surgical_marker_regen_node(state))

    snapshot = result.get("_best_attempt_plan")
    plan_result = result.get("plan_result")
    assert snapshot is not None
    assert snapshot is not plan_result, "Snapshot debe ser deepcopy, no alias"
    # Mutar plan_result no debe afectar snapshot
    plan_result["days"].append({"day": 99})
    assert len(snapshot["days"]) == 3, (
        "Snapshot fue contaminado por mutación a plan_result"
    )


# ---------------------------------------------------------------------------
# 5. Sanity: el código del nodo referencia el promote
# ---------------------------------------------------------------------------
def test_surgical_marker_regen_references_promote_logic():
    """Sanity guard: si alguien remueve la lógica de promotion, el test
    debe fallar para alertar regresión silenciosa."""
    import inspect
    import graph_orchestrator as go
    src = inspect.getsource(go.surgical_marker_regen_node)
    assert "P6-SURGICAL-PROMOTE" in src, (
        "La promoción de snapshot debe estar marcada con P6-SURGICAL-PROMOTE"
    )
    assert "_best_attempt_plan" in src, (
        "El nodo debe escribir _best_attempt_plan cuando surgical regen succeeds"
    )
    assert "fixed_count > 0" in src, (
        "La promoción debe gatearse por fixed_count > 0 (no promote sin fixes)"
    )


# ---------------------------------------------------------------------------
# 6. Repro corrida 2026-05-05 13:11-13:12 — flujo end-to-end del fix
# ---------------------------------------------------------------------------
def test_repro_corrida_13_11_promote_protects_against_rollback():
    """Reproduce el escenario donde re-review demote el plan post-surgical:
    si _best_attempt_plan fue promovido, P0-PIPE-1 NO restaura el plan
    pre-surgical (con markers) en lugar del plan post-surgical (limpio)."""
    import graph_orchestrator as go

    # Simulamos el state DESPUÉS del surgical regen exitoso + promote.
    # Replicamos lo que el nodo escribió:
    surgical_fixed_plan = {
        "days": [
            {"day": 1, "meals": []},  # no marker
            {"day": 2, "meals": [], "regenerated": True},
            {"day": 3, "meals": [], "regenerated": True},
        ],
    }
    state_after_surgical_and_rereview_rejection = {
        "review_passed": False,
        "_rejection_severity": "minor",
        "rejection_reasons": ["Día 1: ingredient mismatch ortogonal"],
        # Pre-surgical snapshot SOBRESCRITO por P6-SURGICAL-PROMOTE:
        "_best_attempt_plan": surgical_fixed_plan,
        "_best_attempt_severity": "approved",
        "_best_attempt_reasons": [],
        "_best_attempt_review_passed": True,
        "_best_attempt_number": 1,
        # Plan actual (que fue demoted):
        "plan_result": {
            "days": [
                {"day": 1, "meals": [], "rejection_minor": True},
                {"day": 2, "meals": [], "regenerated": True},
                {"day": 3, "meals": [], "regenerated": True},
            ],
        },
    }

    # `_swap_to_best_attempt_if_better` muta el state in-place.
    # Debe restaurar el snapshot promovido (rank 0 < rank 1 del minor),
    # preservando los días regenerados.
    state = copy.deepcopy(state_after_surgical_and_rereview_rejection)
    swapped = go._swap_to_best_attempt_if_better(state)
    assert swapped is True, "Swap debe haber ocurrido (best=approved < minor)"

    restored_days = state.get("plan_result", {}).get("days", [])
    # Días 2 y 3 deben mantener `regenerated=True` (versión surgical-fixed,
    # no la pre-surgical con markers).
    day2 = next(d for d in restored_days if d.get("day") == 2)
    day3 = next(d for d in restored_days if d.get("day") == 3)
    assert day2.get("regenerated") is True, (
        "Día 2 debe ser la versión surgical-fixed, no la pre-surgical"
    )
    assert day3.get("regenerated") is True, (
        "Día 3 debe ser la versión surgical-fixed, no la pre-surgical"
    )
    assert "_critique_unresolved" not in day2
    assert "_critique_unresolved" not in day3
