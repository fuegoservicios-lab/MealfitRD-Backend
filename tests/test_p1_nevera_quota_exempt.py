"""[P1-NEVERA-QUOTA-EXEMPT · 2026-06-24] `/restock` ("Ya compré la lista" → user_inventory) y
`/inventory/consume` (vaciar consumidos, sub-paso de renovar plan) son operaciones de inventario SIN
costo LLM. Estaban bajo `verify_api_quota` + `log_api_usage` → al llegar al cap mensual:
  (1) congelaban la Nevera Inteligente (no se podía meter la compra ni renovar — el 402 de consume
      abortaba la renovación con "Error al sincronizar despensa física"), y
  (2) drenaban crédito de planes: `get_monthly_api_usage` (db_profiles.py) cuenta TODA fila de
      `api_usage` sin filtrar endpoint → cada restock/consume restaba 1 del cap que luego los bloqueaba.

Decisión (espejo de P3-SHIFT-PLAN-QUOTA-EXEMPT + convención `Historial-quota-exemption`): el paywall
mensual NO debe gatear operaciones de inventario; el anti-hammering correcto es un RateLimiter
per-user/IP (`_RESTOCK_LIMITER` / `_CONSUME_LIMITER`). Si alguien re-añade `verify_api_quota` o
`log_api_usage("restock_inventory"|"consume_inventory")`, este test falla apuntando a la convención.

Cross-link (P2-HIST-AUDIT-14): slug `p1_nevera_quota_exempt` ↔ este archivo.
Tooltip-anchor: P1-NEVERA-QUOTA-EXEMPT.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_PLANS_PY = _BACKEND_ROOT / "routers" / "plans.py"


@pytest.fixture(scope="module")
def plans_src() -> str:
    return _PLANS_PY.read_text(encoding="utf-8")


def _endpoint_signature(src: str, route: str) -> str:
    m = re.search(
        r'@router\.post\(\s*["\']' + re.escape(route) + r'["\']\s*\).*?def\s+\w+\s*\((.*?)\)\s*:',
        src, re.DOTALL,
    )
    assert m, f"No se encontró el endpoint POST {route} en routers/plans.py."
    return m.group(1)


def test_restock_uses_ratelimiter_not_paywall(plans_src: str):
    sig = _endpoint_signature(plans_src, "/restock")
    assert "_RESTOCK_LIMITER" in sig, (
        "P1-NEVERA-QUOTA-EXEMPT: /restock debe usar `Depends(_RESTOCK_LIMITER)`, no el paywall mensual."
    )
    assert "verify_api_quota" not in sig, (
        "P1-NEVERA-QUOTA-EXEMPT regresión: /restock volvió a `verify_api_quota` → 402 al cap congela "
        "la Nevera + quema crédito. Usar _RESTOCK_LIMITER."
    )


def test_consume_uses_ratelimiter_not_paywall(plans_src: str):
    sig = _endpoint_signature(plans_src, "/inventory/consume")
    assert "_CONSUME_LIMITER" in sig, (
        "P1-NEVERA-QUOTA-EXEMPT: /inventory/consume debe usar `Depends(_CONSUME_LIMITER)`."
    )
    assert "verify_api_quota" not in sig, (
        "P1-NEVERA-QUOTA-EXEMPT regresión: /inventory/consume volvió a `verify_api_quota` → el 402 al "
        "cap aborta la renovación de plan. Usar _CONSUME_LIMITER."
    )


def test_no_log_api_usage_for_inventory_ops(plans_src: str):
    """Ni restock ni consume deben contar contra el cap (log_api_usage los sumaba a api_usage)."""
    for token in ("restock_inventory", "consume_inventory"):
        call_re = re.compile(r'log_api_usage\(\s*\w+\s*,\s*["\']' + token + r'["\']')
        assert not call_re.search(plans_src), (
            f"P1-NEVERA-QUOTA-EXEMPT regresión: `log_api_usage(..., \"{token}\")` reapareció → "
            "la operación de inventario vuelve a quemar crédito de planes. Quitarlo."
        )


def test_both_limiters_defined(plans_src: str):
    assert re.search(r"_RESTOCK_LIMITER\s*=\s*RateLimiter\(", plans_src), "falta `_RESTOCK_LIMITER = RateLimiter(...)`."
    assert re.search(r"_CONSUME_LIMITER\s*=\s*RateLimiter\(", plans_src), "falta `_CONSUME_LIMITER = RateLimiter(...)`."


def test_tooltip_anchor_present(plans_src: str):
    assert "P1-NEVERA-QUOTA-EXEMPT" in plans_src, "tooltip-anchor P1-NEVERA-QUOTA-EXEMPT ausente de routers/plans.py."


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
