"""[P1-ANALYZE-NO-CHARGE-ON-FALLBACK · 2026-06-26] El cobro de `/analyze/stream`
NO debe quemar un crédito del cap mensual cuando el usuario NO recibió un plan
utilizable.

Pre-fix: `log_api_usage(user_id, "llm_analyze_stream")` vivía en el `finally` del
`event_generator` con el único gate `if user_id and user_id != "guest" and
user_id != session_id`, sin consultar el resultado. Los `break` del FALLBACK-GUARD
(rechazo crítico / IA caída / spend-cap / persist-failed) y el error duro de
pipeline caían al MISMO `finally` que el path de éxito → cobraban un crédito sin
entregar plan. Asimetría con S2 (`if not _ai_unavailable`, llm_regenerate_day) y
S3 (charge-on-success, llm_swap_meal).

Post-fix: un flag `_plan_delivery_failed` se setea True en cada exit de no-entrega;
el `finally` solo cobra si `not _plan_delivery_failed or _charge_on_fallback` (knob
MEALFIT_CHARGE_ON_FALLBACK, default false → no cobra en fallback). `cancel` y
`success` siguen cobrando (en cancel el pipeline persiste en background y el user
recupera el plan vía /pending-status).

Test parser-based: el SSE generator depende de Depends/LLM/DB difíciles de invocar
en unit. Anclamos la ESTRUCTURA (init + sets + gate) con tooltip-anchors para que un
renombre/reversión falle el test antes de cambiar producción.
"""
import re
from pathlib import Path

import pytest

_PLANS = (Path(__file__).resolve().parent.parent / "routers" / "plans.py").read_text(encoding="utf-8")


def _extract(fn_signature_re: str) -> str:
    """Extrae el cuerpo de una función top-level (incluye sus funciones anidadas)."""
    m = re.search(fn_signature_re, _PLANS, re.MULTILINE)
    assert m, f"firma {fn_signature_re!r} no encontrada en routers/plans.py"
    start = m.start()
    rest = _PLANS[start + 1:]
    nxt = re.search(r"^(async def |def |@router\.)", rest, re.MULTILINE)
    return _PLANS[start: start + 1 + nxt.start()] if nxt else _PLANS[start:]


@pytest.fixture(scope="module")
def body() -> str:
    return _extract(r"^async def api_analyze_stream\(")


# ---------------------------------------------------------------------------
# Estructura del fix
# ---------------------------------------------------------------------------

def test_flag_initialized_before_loop(body: str):
    """El flag arranca en False antes del `try`/`while True` (default = cobra)."""
    assert "_plan_delivery_failed = False" in body, (
        "[P1-ANALYZE-NO-CHARGE-ON-FALLBACK] falta el init `_plan_delivery_failed = False`."
    )
    init_idx = body.find("_plan_delivery_failed = False")
    loop_idx = body.find("while True:")
    assert init_idx > 0 and loop_idx > 0
    assert init_idx < loop_idx, "el flag debe inicializarse ANTES del while (vive en el finally)."


def test_charge_is_gated_in_finally(body: str):
    """El cobro de cuota debe estar condicionado al flag + knob, no incondicional."""
    assert 'log_api_usage(user_id, "llm_analyze_stream")' in body, (
        "el cobro (P2-LIVE-7) debe seguir presente — solo condicionado, no eliminado."
    )
    assert "if not _plan_delivery_failed or _charge_on_fallback:" in body, (
        "el cobro debe estar detrás de `if not _plan_delivery_failed or _charge_on_fallback:`."
    )
    gate_idx = body.find("if not _plan_delivery_failed or _charge_on_fallback:")
    charge_idx = body.rfind('log_api_usage(user_id, "llm_analyze_stream")')
    assert gate_idx > 0 and charge_idx > 0
    assert gate_idx < charge_idx, "el gate debe preceder al `log_api_usage` (envolverlo)."
    # Distancia acotada: el cobro vive inmediatamente dentro del gate (no en otra rama lejana).
    assert 0 < (charge_idx - gate_idx) < 400, (
        "el `log_api_usage` debe vivir dentro del bloque del gate (ventana corta)."
    )


def test_rollback_knob_default_false(body: str):
    """Knob MEALFIT_CHARGE_ON_FALLBACK existe y revierte al cobro incondicional; default false."""
    assert "MEALFIT_CHARGE_ON_FALLBACK" in body, "falta el knob de rollback."
    knob_block = body[body.find("_charge_on_fallback = os.environ.get"):][:200]
    assert '"MEALFIT_CHARGE_ON_FALLBACK"' in knob_block
    assert '"false"' in knob_block, "el default del knob debe ser 'false' (no cobrar en fallback)."


# ---------------------------------------------------------------------------
# Cada exit de "no se entregó plan" marca el flag
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("error_code", [
    "critical_restriction",   # rechazo crítico (alérgeno/condición declarada)
    "llm_unavailable",        # IA caída / spending-cap
    "plan_persist_failed",    # persistencia falló
])
def test_fallback_break_sets_flag(body: str, error_code: str):
    """Cada rama de fallback que emite `error` + `break` debe setear el flag ANTES,
    para que el finally no cobre ese exit."""
    code_idx = body.find(f"'code': '{error_code}'")
    assert code_idx > 0, f"no se encontró la rama del error_code {error_code!r}."
    # Una asignación del flag debe aparecer en la ventana inmediatamente anterior al yield.
    window = body[max(0, code_idx - 400): code_idx]
    assert "_plan_delivery_failed = True" in window, (
        f"[P1-ANALYZE-NO-CHARGE-ON-FALLBACK] la rama '{error_code}' no setea "
        f"`_plan_delivery_failed = True` antes de emitir el error → cobraría sin entregar plan."
    )


def test_pipeline_hard_error_sets_flag(body: str):
    """El error duro del pipeline (`pipeline_result["error"]`) también suprime el cobro."""
    err_idx = body.find('if pipeline_result["error"]:')
    assert err_idx > 0, "no se encontró la rama `if pipeline_result[\"error\"]:`."
    window = body[err_idx: err_idx + 300]
    assert "_plan_delivery_failed = True" in window, (
        "la rama de error duro del pipeline debe setear el flag (no entrega plan)."
    )


def test_at_least_four_no_delivery_exits_marked(body: str):
    """Defensa de cobertura: ≥4 sitios setean el flag (3 breaks + error de pipeline)."""
    n = body.count("_plan_delivery_failed = True")
    assert n >= 4, (
        f"esperaba ≥4 sets de `_plan_delivery_failed = True` (3 breaks de fallback + "
        f"error de pipeline); encontré {n}. ¿Se removió un exit de no-entrega?"
    )


# ---------------------------------------------------------------------------
# Paridad con S2/S3 (no-cobro cuando la IA no entregó) — sentinel cruzado
# ---------------------------------------------------------------------------

def test_parity_with_s2_no_charge_pattern():
    """S2 ya no cobra si la IA cayó (`if not _ai_unavailable: log_api_usage`). Este
    sentinel ancla la paridad: si alguien borra el patrón de S2, revisamos ambos."""
    assert re.search(
        r"if\s+not\s+_ai_unavailable\s*:\s*\n\s*log_api_usage\(user_id,\s*\"llm_regenerate_day\"\)",
        _PLANS,
    ), "S2 (regenerate-day) debe seguir sin cobrar cuando _ai_unavailable (paridad)."
