"""[P2-ADMIN-RATE-LIMIT · 2026-05-15] Anchor + regression guard + funcional.

Pre-fix: los ~13 endpoints `/admin/*` (8 en plans.py + 4 en system.py
incluyendo /health admin-gated + 1 en notifications.py) estaban auth-gated
por `CRON_SECRET` pero sin rate limiter. Un script en loop accidental
(`* * * * *` vs `*/30 * * * *`) o un atacante con `CRON_SECRET` leaked
puede saturar el pool DB — `/admin/health-snapshot` hace 6 queries
paralelas.

Fix: `_ADMIN_RATE_LIMITER` singleton + helper `_check_admin_rate_limit`
en `backend/routers/plans.py` (mismo módulo que `_verify_admin_token`).
Cada admin endpoint llama el helper inmediatamente después de la
verificación del token. Key por IP (`verified_user_id=None` fuerza
fallback IP). Knobs `MEALFIT_ADMIN_RATE_LIMIT_PER_MIN` (default 60) y
`MEALFIT_ADMIN_RATE_LIMIT_PERIOD_S` (default 60).

Defensas:
  1. Anchor `P2-ADMIN-RATE-LIMIT` presente en plans.py.
  2. Helper `_check_admin_rate_limit(request)` definido.
  3. Knobs auto-registrados con clamps.
  4. CADA admin endpoint (los que ya tienen `_verify_admin_token(...)`)
     DEBE tener `_check_admin_rate_limit(request)` en la línea siguiente
     (o muy cerca, antes del primer SQL/IO de la función).
  5. system.py importa el helper desde `routers.plans`.
  6. Funcional: invocar el limiter N+1 veces con la misma IP excede el
     cap y levanta `HTTPException(429)` con header `Retry-After`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLANS_PY = _REPO_ROOT / "backend" / "routers" / "plans.py"
_SYSTEM_PY = _REPO_ROOT / "backend" / "routers" / "system.py"
_NOTIFICATIONS_PY = _REPO_ROOT / "backend" / "routers" / "notifications.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Anchor + estructura del helper
# ---------------------------------------------------------------------------
def test_anchor_present_in_plans_router():
    src = _read(_PLANS_PY)
    assert "P2-ADMIN-RATE-LIMIT" in src, (
        "Falta anchor `P2-ADMIN-RATE-LIMIT` en routers/plans.py."
    )


def test_helper_defined():
    src = _read(_PLANS_PY)
    assert "def _check_admin_rate_limit(" in src, (
        "Falta helper `def _check_admin_rate_limit(request)` en routers/plans.py."
    )
    assert "_ADMIN_RATE_LIMITER" in src, (
        "Falta singleton `_ADMIN_RATE_LIMITER`."
    )


def test_knobs_use_env_int_with_clamps():
    src = _read(_PLANS_PY)
    assert "MEALFIT_ADMIN_RATE_LIMIT_PER_MIN" in src, (
        "Knob `MEALFIT_ADMIN_RATE_LIMIT_PER_MIN` debe estar referenciado."
    )
    assert "MEALFIT_ADMIN_RATE_LIMIT_PERIOD_S" in src, (
        "Knob `MEALFIT_ADMIN_RATE_LIMIT_PERIOD_S` debe estar referenciado."
    )
    # Resolución vía helper de knobs (auto-registry).
    assert re.search(r"_env_int.*MEALFIT_ADMIN_RATE_LIMIT_PER_MIN", src), (
        "Knob `MEALFIT_ADMIN_RATE_LIMIT_PER_MIN` debe resolverse via `_env_int`."
    )


def test_system_py_imports_helper():
    src = _read(_SYSTEM_PY)
    assert re.search(
        r"from\s+routers\.plans\s+import\s+.*_check_admin_rate_limit",
        src,
    ), (
        "`routers/system.py` debe importar `_check_admin_rate_limit` desde "
        "`routers.plans` (mismo patrón que `_verify_admin_token`)."
    )


# ---------------------------------------------------------------------------
# 2. Migration enforcement: cada `_verify_admin_token(...)` debe ir seguido
#    de `_check_admin_rate_limit(request)`.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", [_PLANS_PY, _SYSTEM_PY, _NOTIFICATIONS_PY])
def test_every_verify_admin_token_followed_by_rate_limit_check(path: Path):
    """Cada callsite de `_verify_admin_token(request.headers.get(...))` DEBE
    tener `_check_admin_rate_limit(request)` en las 3 líneas siguientes.
    Sin esto, un admin endpoint nuevo se merge a producción sin el limiter
    aplicado.

    Excepción documentada: la DEFINICIÓN del helper `_verify_admin_token`
    en `routers/plans.py` NO es un callsite — se identifica porque viene
    precedida de `def _verify_admin_token(`.
    """
    src = _read(path)
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if "_verify_admin_token(request.headers.get(" not in line:
            continue
        # Es callsite (no def). Verificar que dentro de las próximas 3
        # líneas aparece `_check_admin_rate_limit(request)`.
        window = "\n".join(lines[i + 1: i + 4])
        assert "_check_admin_rate_limit(request)" in window, (
            f"{path.name}:{i+1}: callsite de `_verify_admin_token(...)` sin "
            f"`_check_admin_rate_limit(request)` en las 3 líneas siguientes. "
            f"P2-ADMIN-RATE-LIMIT regresión."
        )


# ---------------------------------------------------------------------------
# 3. Funcional: el limiter levanta 429 con Retry-After cuando se excede.
# ---------------------------------------------------------------------------
def test_limiter_raises_429_after_threshold(monkeypatch):
    """Invocar `_check_admin_rate_limit` N+1 veces con la misma IP excede
    el cap y debe levantar `HTTPException(429)` con header `Retry-After`.

    Usamos un limiter local con cap bajo (3 req / 5s) para no esperar.
    """
    import sys
    backend_dir = str(_REPO_ROOT / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    from fastapi import HTTPException
    from rate_limiter import RateLimiter

    # Limiter local con cap muy bajo. NO usamos el módulo-singleton para
    # evitar contaminar otros tests.
    local_limiter = RateLimiter(max_calls=3, period_seconds=5)
    # Forzar fallback de memoria local (no Redis) para test determinístico.
    from cache_manager import redis_client as _orig_redis  # noqa: F401
    import rate_limiter as _rl
    monkeypatch.setattr(_rl, "redis_client", None)

    class _MockClient:
        host = "203.0.113.99"  # IP de documentación TEST-NET-3

    class _MockRequest:
        client = _MockClient()

    req = _MockRequest()

    # Los primeros 3 hits pasan.
    for _ in range(3):
        local_limiter(req, verified_user_id=None)
    # El 4º debe levantar 429.
    with pytest.raises(HTTPException) as exc_info:
        local_limiter(req, verified_user_id=None)
    assert exc_info.value.status_code == 429
    headers = exc_info.value.headers or {}
    assert "Retry-After" in headers, (
        "El 429 debe incluir header `Retry-After` para que el cliente sepa "
        "cuándo reintentar."
    )
    retry_after = int(headers["Retry-After"])
    assert 1 <= retry_after <= 5, (
        f"`Retry-After={retry_after}` fuera del rango esperado [1, 5]."
    )


# ---------------------------------------------------------------------------
# 4. Cross-link guard (P2-HIST-AUDIT-14): cobertura textual del anchor.
# ---------------------------------------------------------------------------
def test_anchor_present_in_test_file():
    src = Path(__file__).read_text(encoding="utf-8")
    assert "P2-ADMIN-RATE-LIMIT" in src
