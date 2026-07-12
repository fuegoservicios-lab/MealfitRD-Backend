"""[P1-SWAP-REGEN-RESUME · 2026-07-11] El spinner del swap INDIVIDUAL sobrevive el refresh
(espejo de P1-DAY-REGEN-RESUME del día completo).

Pedido del owner: "cuando actualizo un plato individual y actualizo la página, la animación
de carga desaparece — quiero que esté igual que cuando actualizo los platos del día".

Arquitectura (3 piezas backend + 3 frontend):
- B1: flag server-side `plan_data._meal_regen_inflight` al arrancar el swap (jsonb_set,
  AND user_id — I2); retirado al terminar (éxito, soft-fail o excepción).
- B2: PERSIST server-side del plato generado dentro del MISMO request — sin esto, el
  refresh mata los dos requests del cliente (swap + persist) y el plato se pierde: el
  overlay "resumido" esperaría un cambio que jamás llega. Reusa `api_swap_meal_persist`
  in-process (mismas defensas I6/I7). El cliente vivo conserva su persist propio
  (idempotente, fallback).
- B3: solo autenticados con plan_id/day_index/meal_index en el body; guests intactos.
- F1: marker `mealfit_meal_regen_inflight` + estado `mealRegenInFlight` en el contexto.
- F2: effect de resume que pollea plans-data/latest (flag fresco → sigue; ausente +
  _plan_modified_at≠baseline o nombre cambiado → aplicar + toast; sin cambios → apagar).
- F3: Dashboard re-enciende el spinner del card (regeneratingId) escopado al día VISIBLE.

tooltip-anchor: P1-SWAP-REGEN-RESUME
"""
from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_ROOT = _BACKEND.parent
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_CTX = (_ROOT / "frontend" / "src" / "context" / "AssessmentContext.jsx").read_text(encoding="utf-8")
_DASH = (_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")


def test_b1_flag_set_and_clear_filter_user_id():
    i = _PLANS.find("def _swap_meal_regen_flag_set")
    assert i != -1, "helper del flag desapareció"
    win = _PLANS[i:i + 2500]
    assert "jsonb_set(plan_data, '{_meal_regen_inflight}'" in win
    assert "WHERE id = %s AND user_id = %s" in win, "I2: toda mutación filtra AND user_id"
    j = _PLANS.find("def _swap_meal_regen_flag_clear")
    assert j != -1
    win_c = _PLANS[j:j + 1200]
    assert "plan_data - '_meal_regen_inflight'" in win_c
    assert "WHERE id = %s AND user_id = %s" in win_c


def test_b2_server_persist_reuses_guarded_endpoint():
    i = _PLANS.find("def _persist_swap_server_side")
    assert i != -1
    win = _PLANS[i:i + 4000]
    assert "api_swap_meal_persist(" in win, (
        "el persist server-side debe reusar el endpoint protegido (FOR UPDATE + user_id + "
        "clinical guard) IN-PROCESS, no duplicar la lógica"
    )
    assert '"isExpanded"' in win and "ingredients_raw" in win, (
        "espejo del merge del cliente + raw fresco del resultado"
    )


def test_b3_wiring_order_in_swap_endpoint():
    i = _PLANS.find("def api_swap_meal(")
    assert i != -1
    body = _PLANS[i:i + 30000]
    i_set = body.find("_swap_meal_regen_flag_set(data, verified_user_id)")
    i_swap = body.find("result = swap_meal(data)")
    i_persist = body.find("_persist_swap_server_side(_mri_ctx, result, verified_user_id)")
    assert -1 not in (i_set, i_swap, i_persist)
    assert i_set < i_swap < i_persist, (
        "orden: flag ANTES de la IA (el poller lo ve al refrescar) → swap → persist"
    )
    assert 'if (user_id and user_id != "guest") else None' in body, "guests: flujo previo intacto"
    assert body.count("_swap_meal_regen_flag_clear(_mri_ctx, verified_user_id)") >= 3, (
        "el flag se retira en éxito, soft-fail y excepción (sin retiro queda stale 6 min)"
    )


def test_f1_marker_and_body_ctx():
    assert "mealfit_meal_regen_inflight" in _CTX
    i = _CTX.find("const regenerateSingleMeal")
    win = _CTX[i:i + 6000]
    assert "plan_id: _swapResumable ? _swapPlanId : undefined" in win, (
        "el body del swap lleva el contexto de persistencia server-side (solo authed)"
    )
    assert "safeLocalStorageSet('mealfit_meal_regen_inflight'" in win
    fin = _CTX.find("safeLocalStorageRemove('mealfit_meal_regen_inflight')", i)
    assert fin != -1, "finally de la ruta sin-refresh limpia el marker"


def test_f2_resume_effect_polls_latest():
    i = _CTX.find("[P1-SWAP-REGEN-RESUME · 2026-07-11] Resume cross-refresh del swap INDIVIDUAL")
    assert i != -1, "effect de resume desapareció"
    win = _CTX[i:i + 8000]
    assert "_meal_regen_inflight" in win and "/api/plans-data/latest" in win
    assert "baseModifiedAt" in win, (
        "completion server-vs-server (lección P2-REGEN-RESUME-NO-CLOCKS: jamás reloj cliente)"
    )
    assert "setMealRegenInFlight" in win


def test_f3_dashboard_scoped_to_visible_day():
    i = _DASH.find("_prevMealRegenRef")
    assert i != -1, "espejo del spinner en Dashboard desapareció"
    win = _DASH[max(0, i - 800):i + 1200]
    assert "mealRegenInFlight.dayIndex === activeDayIndex" in win, (
        "regeneratingId es un índice del día VISIBLE — sin el guard el spinner aparece "
        "en el card equivocado de otro día"
    )
    assert "mealRegenInFlight" in _DASH[:30000] or "mealRegenInFlight," in _DASH, "destructure del contexto"


def test_marker_anchored():
    assert _PLANS.count("P1-SWAP-REGEN-RESUME") >= 3
    assert _CTX.count("P1-SWAP-REGEN-RESUME") >= 3
    assert _DASH.count("P1-SWAP-REGEN-RESUME") >= 1
