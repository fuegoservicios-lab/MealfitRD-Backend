"""[P2-GUEST-LLM-RATELIMIT · 2026-05-30] `/swap-meal` y `/recipe/expand` deben
throttlear el burst de LLM (cost-amplification de guests anónimos).

Bug (audit prod-readiness 2026-05-30):
    Ambos endpoints invocan Gemini pero solo tenían `Depends(verify_api_quota)`.
    Para un GUEST no autenticado el paywall mensual NO aplica, así que un
    atacante podía martillar cualquiera de los dos sin tope → amplificación de
    costo LLM contra nuestra cuota Gemini.

Fix: añadir un `RateLimiter` (bucket por verified_user_id / ip:<host>) a cada
endpoint, ADEMÁS de `verify_api_quota` (mismo patrón que `/analyze` con
`_PLAN_GEN_LIMITER`).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_ROUTER = Path(__file__).resolve().parent.parent / "routers" / "plans.py"


def _read() -> str:
    return _ROUTER.read_text(encoding="utf-8")


def _endpoint_signature(src: str, path: str) -> str:
    """Devuelve la línea `def ...` del handler decorado por `@router.post("<path>")`."""
    m = re.search(
        rf'@router\.post\(\s*["\']{re.escape(path)}["\']\s*\)\s*\n\s*(?:async\s+)?def\s+\w+\([^\n]*\)',
        src,
    )
    assert m, f"No se encontró el handler @router.post({path!r})."
    return m.group(0)


def test_limiters_defined():
    src = _read()
    assert re.search(r"_SWAP_LIMITER\s*=\s*RateLimiter\(", src), "Falta `_SWAP_LIMITER`."
    assert re.search(r"_EXPAND_LIMITER\s*=\s*RateLimiter\(", src), "Falta `_EXPAND_LIMITER`."


def test_swap_meal_has_ratelimiter():
    sig = _endpoint_signature(_read(), "/swap-meal")
    assert "Depends(_SWAP_LIMITER)" in sig, (
        "`api_swap_meal` debe tener `Depends(_SWAP_LIMITER)` para throttlear el "
        "burst de LLM de guests anónimos (P2-GUEST-LLM-RATELIMIT)."
    )
    # Conserva el paywall.
    assert "verify_api_quota" in sig, (
        "`api_swap_meal` debe CONSERVAR `verify_api_quota` (paywall) además del limiter."
    )


def test_expand_recipe_has_ratelimiter():
    sig = _endpoint_signature(_read(), "/recipe/expand")
    assert "Depends(_EXPAND_LIMITER)" in sig, (
        "`api_expand_recipe` debe tener `Depends(_EXPAND_LIMITER)` para throttlear "
        "el burst de LLM de guests anónimos (P2-GUEST-LLM-RATELIMIT)."
    )
    assert "verify_api_quota" in sig, (
        "`api_expand_recipe` debe CONSERVAR `verify_api_quota` además del limiter."
    )


def test_anchor_present():
    assert "P2-GUEST-LLM-RATELIMIT" in _read()
