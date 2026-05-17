"""[P3-SUPABASE-TRANSIENT-RETRY · 2026-05-16] El endpoint /water-intake se
dispara con frecuencia desde el listener `visibilitychange` del WaterTracker.
Cuando el tab estuvo en background, las conexiones HTTPS idle backend→Supabase
pueden estar muertas; la primer query lo descubre y httpx levanta
`RemoteProtocolError` / `ReadError` / `ConnectError`.

Pre-fix: el catch devolvia 500 al cliente → console.error en DevTools + toast
"No pudimos guardar". Post-fix: el helper `_water_supabase_with_retry`
reintenta 1 vez con backoff 350ms; si ambos fallan, devolvemos 503 (transient)
en lugar de 500 (sugiere bug del endpoint). El frontend (WaterTracker.jsx)
tambien reintenta 1 vez ante 5xx/network error antes de mostrar el toast.

Incidente observado 2026-05-16 ~20:30: GET /api/plans/water-intake → 500 +
`net::ERR_CONNECTION_CLOSED` en supabase-js (user_profiles) simultaneos, ambos
disparados por `onVisibility` al volver al tab. El fix de Supabase-js
(retry universal) queda fuera de scope; este P-fix cubre solo el path
backend de /water-intake.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_PLANS = (_BACKEND_ROOT / "routers" / "plans.py").read_text(encoding="utf-8")
_FRONTEND_ROOT = _BACKEND_ROOT.parent / "frontend"
_WATER_TRACKER = (
    _FRONTEND_ROOT / "src" / "components" / "dashboard" / "WaterTracker.jsx"
).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Backend: helper declarado + usado en ambos endpoints
# ---------------------------------------------------------------------------


def test_helper_declared_with_marker():
    """`_water_supabase_with_retry(builder_factory, op_label)` debe existir
    con el marker P3-SUPABASE-TRANSIENT-RETRY cerca, para que un refactor
    cosmetico no borre el por que del reintento."""
    assert "def _water_supabase_with_retry(" in _PLANS, (
        "Helper `_water_supabase_with_retry` no declarado en routers/plans.py."
    )
    assert "P3-SUPABASE-TRANSIENT-RETRY" in _PLANS, (
        "Marker `P3-SUPABASE-TRANSIENT-RETRY` ausente en routers/plans.py."
    )


def test_helper_uses_factory_pattern():
    """El helper DEBE invocar `builder_factory()` cada vez (los builders
    de supabase-py son stateful y no se pueden reusar tras `.execute()`
    parcial)."""
    pat = re.compile(
        r"def _water_supabase_with_retry\([^)]*\):.*?return last_exc \w*",
        re.DOTALL,
    )
    # Buscar referencias a builder_factory() (CON paréntesis):
    body_start = _PLANS.find("def _water_supabase_with_retry(")
    assert body_start > 0
    body_end = _PLANS.find("\n\n\n", body_start)
    body = _PLANS[body_start:body_end if body_end > 0 else body_start + 2000]
    assert "builder_factory()" in body, (
        "Helper debe invocar `builder_factory()` (con paréntesis) — sin ellos "
        "estarias pasando el callable a .execute() en lugar del builder."
    )


def test_helper_retries_at_least_twice():
    """Sanity: el helper debe intentar al menos 2 veces (1 try + 1 retry)."""
    m = re.search(r"_WATER_RETRY_ATTEMPTS\s*=\s*(\d+)", _PLANS)
    assert m, "Constante `_WATER_RETRY_ATTEMPTS` no declarada."
    n = int(m.group(1))
    assert 2 <= n <= 4, (
        f"_WATER_RETRY_ATTEMPTS={n} fuera de [2, 4]. 1 = sin reintento, "
        ">4 = bloquea el worker thread demasiado tiempo."
    )


def test_backoff_reasonable():
    """Backoff [0.1, 2.0] segundos. <100ms no le da tiempo a la conexion
    nueva; >2s bloquea el worker thread."""
    m = re.search(r"_WATER_RETRY_BACKOFF_S\s*=\s*([\d.]+)", _PLANS)
    assert m, "Constante `_WATER_RETRY_BACKOFF_S` no declarada."
    s = float(m.group(1))
    assert 0.1 <= s <= 2.0, f"_WATER_RETRY_BACKOFF_S={s} fuera de [0.1, 2.0]."


@pytest.mark.parametrize("endpoint_marker", [
    '@router.get("/water-intake")',
    '@router.post("/water-intake")',
])
def test_both_endpoints_use_helper(endpoint_marker):
    """GET y POST /water-intake DEBEN invocar `_water_supabase_with_retry`
    en lugar de `.execute()` directo."""
    idx = _PLANS.find(endpoint_marker)
    assert idx > 0, f"Endpoint {endpoint_marker} no encontrado."
    # Slice del endpoint hasta el proximo @router o final del bloque (~3000 chars)
    next_router = _PLANS.find("@router.", idx + len(endpoint_marker))
    end = next_router if next_router > 0 else idx + 3000
    body = _PLANS[idx:end]
    assert "_water_supabase_with_retry(" in body, (
        f"{endpoint_marker} no usa `_water_supabase_with_retry` — revierte el fix."
    )


@pytest.mark.parametrize("endpoint_marker", [
    '@router.get("/water-intake")',
    '@router.post("/water-intake")',
])
def test_both_endpoints_escalate_to_503(endpoint_marker):
    """Cuando ambos intentos fallan, el endpoint DEBE devolver 503 (transient,
    reintenta) en lugar de 500 (sugiere bug del endpoint)."""
    idx = _PLANS.find(endpoint_marker)
    assert idx > 0
    next_router = _PLANS.find("@router.", idx + len(endpoint_marker))
    end = next_router if next_router > 0 else idx + 3000
    body = _PLANS[idx:end]
    # Buscar el except final con HTTPException(503):
    assert "status_code=503" in body, (
        f"{endpoint_marker} no escala a 503 tras fallo del retry — sigue "
        "devolviendo 500."
    )
    # Y debe loguear con exc_info (para que el traceback llegue a Sentry/stdout):
    assert "exc_info=" in body, (
        f"{endpoint_marker} no pasa `exc_info=...` al logger — pierde el "
        "traceback completo del exception transient."
    )


# ---------------------------------------------------------------------------
# Frontend: WaterTracker.jsx reintenta en loadIntake + persistGlasses
# ---------------------------------------------------------------------------


def test_frontend_marker_present():
    assert "P3-SUPABASE-TRANSIENT-RETRY" in _WATER_TRACKER, (
        "Marker ausente en WaterTracker.jsx — un refactor cosmetico podria "
        "borrar el por que del reintento."
    )


@pytest.mark.parametrize("fn_signature", [
    "const loadIntake = useCallback(",
    "const persistGlasses = useCallback(",
])
def test_frontend_functions_have_retry_pattern(fn_signature):
    """Ambas funciones DEBEN seguir el patron `attemptOnce()` + check
    transient (5xx o networkError) + `setTimeout(500)` + segundo attempt."""
    idx = _WATER_TRACKER.find(fn_signature)
    assert idx > 0, f"Funcion {fn_signature!r} no encontrada."
    # Slice de ~2500 chars del callback (suficiente para incluir el patron)
    end = _WATER_TRACKER.find("}, [", idx)
    # Si no encuentra el `}, [deps]` cierre, tomar 2500 chars
    body = _WATER_TRACKER[idx: end + 100 if end > 0 else idx + 2500]

    assert "attemptOnce" in body, (
        f"{fn_signature} no define `attemptOnce` — patron de reintento perdido."
    )
    assert "res.status >= 500" in body, (
        f"{fn_signature} no chequea `res.status >= 500` para clasificar "
        "transient. 4xx no debe reintentar (determinístico)."
    )
    assert "setTimeout" in body, (
        f"{fn_signature} no tiene `setTimeout` para el backoff entre reintentos."
    )
