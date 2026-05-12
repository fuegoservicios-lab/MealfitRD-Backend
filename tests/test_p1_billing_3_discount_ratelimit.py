"""[P1-BILLING-3 · 2026-05-12] `/api/discount/validate` debe requerir auth
y rate-limit.

Bug observado (audit 2026-05-11):
    Pre-fix, el endpoint era público (sin `Depends(get_verified_user_id)`)
    y sin rate-limit. Atacante anónimo podía brute-force la tabla
    `discount_codes` enumerando códigos alfabéticamente sin coste — cada
    código válido encontrado era $X gratis para usar luego en `/verify`.

Fix:
    1. `Depends(_DISCOUNT_VALIDATE_LIMITER)` — RateLimiter singleton de
       20 calls/min/user (o IP fallback). El limiter inyecta
       `verified_user_id` y bloquea bursts.
    2. `if not verified_user_id: raise 401` — auth obligatoria. Sin auth
       no se valida (descuentos solo para usuarios registrados).

Lo que este test enforza:
    A) Anchor `P1-BILLING-3` presente en `routers/billing.py`.
    B) `_DISCOUNT_VALIDATE_LIMITER = RateLimiter(` declarado a nivel
       módulo con `max_calls`/`period_seconds` kwargs.
    C) Handler `api_validate_discount` usa `Depends(_DISCOUNT_VALIDATE_LIMITER)`.
    D) Handler levanta 401 si `verified_user_id` es None.
    E) `from rate_limiter import RateLimiter` aparece en imports.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_BILLING_PY = _BACKEND_ROOT / "routers" / "billing.py"


@pytest.fixture(scope="module")
def billing_src() -> str:
    return _BILLING_PY.read_text(encoding="utf-8")


def _isolate_handler(src: str) -> str:
    m = re.search(
        r'@discount_router\.post\("/validate"\)\s*\n'
        r'(?:async\s+)?def\s+\w+\([^)]*\)[^:]*:(.*?)(?=@(?:discount_router|router|webhooks_router)\.|\Z)',
        src,
        re.DOTALL,
    )
    assert m is not None, "Handler /api/discount/validate no aislable."
    return m.group(1)


def test_a_anchor_present(billing_src: str):
    assert "P1-BILLING-3" in billing_src, (
        "P1-BILLING-3: anchor desapareció. Restaurar en el bloque "
        "doc del rate-limiter o del handler."
    )


def test_b_limiter_declared(billing_src: str):
    """`_DISCOUNT_VALIDATE_LIMITER` declarado a nivel módulo."""
    assert "_DISCOUNT_VALIDATE_LIMITER = RateLimiter(" in billing_src, (
        "P1-BILLING-3: limiter singleton no declarado. Mismo patrón que "
        "`_CHAT_TTS_LIMITER` en chat.py."
    )
    # Sanity: kwargs presentes (no posicional críptico).
    decl_match = re.search(
        r"_DISCOUNT_VALIDATE_LIMITER\s*=\s*RateLimiter\(([^)]+)\)",
        billing_src,
    )
    assert decl_match is not None
    args = decl_match.group(1)
    assert "max_calls=" in args and "period_seconds=" in args, (
        "P1-BILLING-3: declaración del limiter sin kwargs explícitos. "
        "Usar `RateLimiter(max_calls=N, period_seconds=M)` para que un "
        "lector sepa qué número es cuál sin abrir rate_limiter.py."
    )


def test_c_ratelimiter_import(billing_src: str):
    assert re.search(
        r"from\s+rate_limiter\s+import\s+RateLimiter",
        billing_src,
    ), (
        "P1-BILLING-3: import `from rate_limiter import RateLimiter` ausente."
    )


def test_d_handler_uses_limiter_dep(billing_src: str):
    handler = _isolate_handler(billing_src)
    assert "Depends(_DISCOUNT_VALIDATE_LIMITER)" in handler, (
        "P1-BILLING-3: el handler `/api/discount/validate` no usa "
        "`Depends(_DISCOUNT_VALIDATE_LIMITER)`. Atacante puede seguir "
        "brute-force-eando códigos sin throttle."
    )


def test_e_handler_rejects_unauthenticated(billing_src: str):
    """El handler debe levantar 401 si `verified_user_id` es None.
    RateLimiter cae a IP-bucket sin auth pero NO rechaza la request."""
    handler = _isolate_handler(billing_src)
    assert re.search(
        r"if\s+not\s+verified_user_id\s*:",
        handler,
    ), (
        "P1-BILLING-3: el handler no rechaza requests sin auth. "
        "RateLimiter cae a IP-bucket pero NO bloquea — un atacante "
        "anónimo aún puede consumir su cuota IP enumerando."
    )
    # El raise debe ser HTTPException 401 cercano al check.
    raise_pattern = re.search(
        r"if\s+not\s+verified_user_id\s*:[\s\S]{0,300}HTTPException\(\s*status_code\s*=\s*401",
        handler,
    )
    assert raise_pattern is not None, (
        "P1-BILLING-3: el check `if not verified_user_id` no levanta "
        "HTTPException(401). Verificar el flujo."
    )
