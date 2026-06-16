"""[P3-SHIFT-PLAN-QUOTA-EXEMPT · 2026-06-15] `/shift-plan` avanza la ventana
rolling de un plan YA generado (mantenimiento, no un plan nuevo). Antes estaba
bajo `verify_api_quota` (paywall mensual) + `log_api_usage("shift_plan")`
(P2-LIVE-7) → al llegar al cap (15/15) el dashboard recibía HTTP 402 y la ventana
se congelaba, además de cobrar un crédito extra por USAR un plan ya pagado.

Decisión (alineada con la convención `Historial-quota-exemption`): el paywall
mensual NO debe gatear el mantenimiento de un plan ya generado; el anti-hammering
correcto es un RateLimiter per-user/IP (`_SHIFT_LIMITER`), no el paywall. shift es
idempotente (no-op si está al día), así que NO cuenta contra el cap.

Este test ancla la decisión a nivel código (parser): si alguien "arregla"
shift-plan volviéndolo a `verify_api_quota` o re-añade `log_api_usage("shift_plan")`,
el test falla apuntando a esta convención.

Cross-link (P2-HIST-AUDIT-14): slug `p3_shift_plan_quota_exempt` ↔ este archivo.
Tooltip-anchor: P3-SHIFT-PLAN-QUOTA-EXEMPT.
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


def _shift_plan_signature(src: str) -> str:
    """Decorador `@router.post("/shift-plan")` + def + signature hasta el `):`."""
    m = re.search(
        r'@router\.post\(\s*["\']/shift-plan["\']\s*\).*?def\s+\w+\s*\((.*?)\)\s*:',
        src, re.DOTALL,
    )
    assert m, "No se encontró el endpoint POST /shift-plan en routers/plans.py."
    return m.group(1)


def test_shift_plan_uses_ratelimiter_not_paywall(plans_src: str):
    sig = _shift_plan_signature(plans_src)
    assert "_SHIFT_LIMITER" in sig, (
        "P3-SHIFT-PLAN-QUOTA-EXEMPT: /shift-plan debe usar `Depends(_SHIFT_LIMITER)` "
        "(RateLimiter), no el paywall mensual. Ver CLAUDE.md 'Historial-quota-exemption'."
    )
    assert "verify_api_quota" not in sig, (
        "P3-SHIFT-PLAN-QUOTA-EXEMPT regresión: /shift-plan volvió a `verify_api_quota` "
        "→ 402 + crédito extra al cap, congelando un plan ya generado. Usar _SHIFT_LIMITER."
    )


def test_shift_plan_does_not_log_api_usage(plans_src: str):
    """No debe re-aparecer `log_api_usage(<id>, "shift_plan")` — contaba el shift
    contra el cap mensual. (Los comentarios que mencionan el nombre no matchean
    el patrón de llamada.)"""
    call_re = re.compile(r'log_api_usage\(\s*\w+\s*,\s*["\']shift_plan["\']')
    assert not call_re.search(plans_src), (
        "P3-SHIFT-PLAN-QUOTA-EXEMPT regresión: `log_api_usage(..., \"shift_plan\")` "
        "reapareció → el shift vuelve a contar contra el cap. Quitarlo (el RateLimiter "
        "ya cubre el anti-hammering de P2-LIVE-7)."
    )


def test_shift_limiter_defined(plans_src: str):
    assert re.search(r"_SHIFT_LIMITER\s*=\s*RateLimiter\(", plans_src), (
        "P3-SHIFT-PLAN-QUOTA-EXEMPT: falta la definición de `_SHIFT_LIMITER = RateLimiter(...)`."
    )


def test_tooltip_anchor_present(plans_src: str):
    assert "P3-SHIFT-PLAN-QUOTA-EXEMPT" in plans_src, (
        "tooltip-anchor P3-SHIFT-PLAN-QUOTA-EXEMPT ausente de routers/plans.py."
    )
