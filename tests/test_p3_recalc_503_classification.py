"""[P3-RECALC-503-CLASSIFICATION · 2026-05-16] El endpoint
`POST /api/plans/recalculate-shopping-list` clasifica la excepcion del bloque
except final: transient (pool exhaustion, network blip a Supabase) → 503;
determinístico (KeyError sobre plan_data, ValueError, etc.) → 500.

Pre-fix: cualquier excepcion → 500. El error específico
`couldn't get a connection after 8.00 sec` del psycopg pool del free tier
disparaba 500 al cliente → CORS error secundario porque Starlette no agrega
headers CORS en algunas rutas de error → console ruidoso + toast dismiss.

Post-fix: backend clasifica via `_is_transient_db_error` (helper compartido
en db_facts.py, extendido con `"couldn't get a connection"` y
`"remoteprotocolerror"`); frontend (Dashboard.jsx, AssessmentContext.jsx,
Pantry.jsx) reintenta 1x tras 500ms en 5xx/network — el blip transient se
resuelve sin toast de error al usuario.

Sintoma original observado 2026-05-16: usuario cambio duracion a 15 dias →
console mostro `POST /recalculate-shopping-list 500 + CORS error` aunque el
PDF se genero correctamente (el state local ya tenia la duracion nueva, el
recalc del servidor fallo pero el cliente no se entero del fallo critico).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_FRONTEND_ROOT = _BACKEND_ROOT.parent / "frontend"

_PLANS = (_BACKEND_ROOT / "routers" / "plans.py").read_text(encoding="utf-8")
_DB_FACTS = (_BACKEND_ROOT / "db_facts.py").read_text(encoding="utf-8")
_DASHBOARD = (_FRONTEND_ROOT / "src" / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")
_ASSESSMENT = (_FRONTEND_ROOT / "src" / "context" / "AssessmentContext.jsx").read_text(encoding="utf-8")
_PANTRY = (_FRONTEND_ROOT / "src" / "pages" / "Pantry.jsx").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Backend: helper extendido + clasificación en el endpoint
# ---------------------------------------------------------------------------


def test_pool_exhaustion_fragment_in_transient_helper():
    """El helper `_is_transient_db_error` debe reconocer `couldn't get a
    connection` (mensaje del psycopg pool exhaustion del free tier
    pgBouncer). Sin esto, el error se clasifica como bug determinístico
    y nunca recibe 503 → cliente nunca reintenta."""
    # El fragment debe estar en la tupla, en lowercase:
    assert "couldn't get a connection" in _DB_FACTS, (
        "Fragment `couldn't get a connection` ausente en "
        "_TRANSIENT_DB_ERROR_FRAGMENTS. Sin esto la pool exhaustion del free "
        "tier nunca se clasifica como transient → siempre 500 → cliente "
        "nunca reintenta."
    )
    assert "remoteprotocolerror" in _DB_FACTS, (
        "Fragment `remoteprotocolerror` ausente — httpx levanta este error "
        "cuando una conexion idle supabase muere mid-request."
    )


def test_recalc_endpoint_imports_helper_in_except():
    """El bloque `except Exception` del endpoint DEBE importar y usar
    `_is_transient_db_error` para clasificar antes de elegir 500 vs 503."""
    # Slice del endpoint: localizar el marker del decorator y leer hasta
    # el próximo @router (~5000 chars cubre el handler completo)
    idx = _PLANS.find('@router.post("/recalculate-shopping-list")')
    assert idx > 0, "Endpoint no encontrado."
    next_router = _PLANS.find("@router.", idx + 50)
    end = next_router if next_router > 0 else idx + 8000
    body = _PLANS[idx:end]

    assert "_is_transient_db_error" in body, (
        "El except del endpoint NO importa/usa `_is_transient_db_error` — "
        "revierte la clasificacion P3-RECALC-503-CLASSIFICATION."
    )
    assert "P3-RECALC-503-CLASSIFICATION" in body, (
        "Marker ausente cerca del except — un refactor cosmetico podria "
        "borrar la razon del classification."
    )


def test_recalc_endpoint_returns_503_when_transient():
    """El branch transient debe levantar HTTPException(503), no 500."""
    idx = _PLANS.find('@router.post("/recalculate-shopping-list")')
    end = _PLANS.find("@router.", idx + 50)
    body = _PLANS[idx:end if end > 0 else idx + 8000]

    # Buscar el branch que llama _is_transient_db_error → 503
    pat = re.compile(
        r"_is_transient_db_error\(\s*e\s*\).*?status_code=503",
        re.DOTALL,
    )
    assert pat.search(body), (
        "El branch transient no escala a 503 — sigue cayendo al 500 generico."
    )


def test_recalc_endpoint_returns_500_when_not_transient():
    """Bugs determinísticos (KeyError, ValueError) deben seguir devolviendo
    500 — son señal de bug real, no de pool transient."""
    idx = _PLANS.find('@router.post("/recalculate-shopping-list")')
    end = _PLANS.find("@router.", idx + 50)
    body = _PLANS[idx:end if end > 0 else idx + 8000]

    # Después del 503 branch, debe seguir habiendo un raise 500:
    assert "status_code=500" in body, (
        "El endpoint ya no devuelve 500 en ningun caso — los bugs "
        "deterministicos quedarian disfrazados de 503 'transient'."
    )


# ---------------------------------------------------------------------------
# Frontend: 3 callers tienen retry pattern
# ---------------------------------------------------------------------------


_FRONTEND_SOURCES = {
    "Dashboard.jsx": _DASHBOARD,
    "AssessmentContext.jsx": _ASSESSMENT,
    "Pantry.jsx": _PANTRY,
}


@pytest.mark.parametrize("label", list(_FRONTEND_SOURCES.keys()))
def test_frontend_caller_has_retry_pattern(label):
    """Cada caller debe tener: (a) helper `attemptRecalc`, (b) check de
    `res.status >= 500`, (c) `setTimeout(500)` para el backoff, (d) marker
    P3-RECALC-503-CLASSIFICATION en el archivo (no por callsite — basta
    con que el archivo declare la convención una vez)."""
    source = _FRONTEND_SOURCES[label]
    # Localizar la llamada REAL al endpoint (no la primera mencion textual,
    # que puede estar en un comentario). El callsite real usa
    # `fetchWithAuth(...recalculate-shopping-list...`.
    pat = re.compile(r"fetchWithAuth\([^)]*recalculate-shopping-list", re.DOTALL)
    m = pat.search(source)
    assert m, (
        f"{label} no contiene call real `fetchWithAuth(...recalculate-shopping-list)`."
    )
    idx = m.start()
    # Slice generoso (~4000 chars) alrededor del callsite real
    start = max(0, idx - 1500)
    end = min(len(source), idx + 3000)
    block = source[start:end]

    assert "attemptRecalc" in block, (
        f"{label}: helper `attemptRecalc` no detectado cerca del callsite "
        f"(line ~{source[:idx].count(chr(10)) + 1}). Patron retry perdido."
    )
    assert "status >= 500" in block, (
        f"{label}: no chequea `res.status >= 500` para clasificar transient. "
        "4xx (401/400) NO deben reintentar — son determinísticos."
    )
    assert "setTimeout" in block, (
        f"{label}: falta `setTimeout` para backoff entre reintentos."
    )
    # El marker basta con que esté en el archivo (no necesariamente en el slice).
    assert "P3-RECALC-503-CLASSIFICATION" in source, (
        f"{label}: marker `P3-RECALC-503-CLASSIFICATION` ausente del archivo — "
        "un refactor cosmético podría borrar el motivo del retry."
    )
