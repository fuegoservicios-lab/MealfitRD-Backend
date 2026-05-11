"""[P3-A-NO-STORE-HELPER · 2026-05-10] El endpoint `/history-list` y
los demás endpoints derivados del Historial DEBEN usar el helper SSOT
`_apply_no_store(response)` para los headers Cache-Control + Pragma,
no escribirlos inline.

Bug original (audit 2026-05-10, gap residual):
    P1-HIST-AUDIT-4-FOLLOWUP introdujo `Cache-Control: no-store, max-age=0`
    + `Pragma: no-cache` inline en `/history-list` (`routers/plans.py:6246-6247`)
    porque era el primer endpoint que necesitaba freshness garantizada
    tras restore/delete/rename. Después, P2-HIST-AUDIT-A (2026-05-09)
    extrajo el patrón a un helper SSOT `_apply_no_store(response)`
    (`routers/plans.py:3706-3710`) y migró 7 endpoints derivados del
    Historial al helper.

    El sitio original de `/history-list` quedó SIN migrar — drift
    cosmético: si alguien cambia el helper (e.g. añade `Vary: Cookie`),
    el cambio NO se propaga al sitio inline. Inconsistencia silenciosa.

    NOTA: el sitio en `/api/analyze/stream` (línea ~2845) NO está
    sujeto a esta regla — usa `Cache-Control: no-cache` (semántica
    distinta) en el constructor `StreamingResponse(headers=...)` para
    SSE. El helper `_apply_no_store` es para responses REST estándar.

Fix (P3-A · 2026-05-10):
    Reemplazar las 2 líneas inline de `/history-list` por
    `_apply_no_store(response)`. El comentario referencia el helper
    SSOT y la razón histórica.

Cobertura de este test (parser-based):
    1. El helper `_apply_no_store` existe y aplica los 2 headers
       canónicos (`Cache-Control: no-store, max-age=0` + `Pragma: no-cache`).
    2. Ningún endpoint REST del Historial setea `Cache-Control: no-store`
       inline — todos pasan por el helper.
    3. El endpoint `/history-list` específicamente invoca el helper
       (regression guard contra revertir el fix).

Out of scope:
    - Validación de `_apply_no_store` en endpoints fuera del Historial:
      el contrato es "endpoints del Historial" (P2-HIST-AUDIT-A); otros
      endpoints pueden tener semánticas distintas (e.g. SSE en
      `/api/analyze/stream` usa `no-cache` no `no-store`).
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLANS_PATH = _REPO_ROOT / "backend" / "routers" / "plans.py"


def _read(path: Path) -> str:
    assert path.exists(), f"Archivo requerido no encontrado: {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Helper SSOT existe y aplica los headers canónicos.
# ---------------------------------------------------------------------------
def test_apply_no_store_helper_exists_and_applies_canonical_headers():
    """Sanity check del SSOT: si alguien renombra el helper o cambia
    los headers canónicos, este test guía hacia los call sites."""
    from routers.plans import _apply_no_store

    src = inspect.getsource(_apply_no_store)
    assert 'Cache-Control' in src, (
        "El helper `_apply_no_store` debe setear el header `Cache-Control`."
    )
    assert 'no-store' in src and 'max-age=0' in src, (
        "El helper debe usar `Cache-Control: no-store, max-age=0` "
        "(contrato P2-HIST-AUDIT-A)."
    )
    assert 'Pragma' in src and 'no-cache' in src, (
        "El helper debe setear `Pragma: no-cache` además de Cache-Control "
        "(legacy compat: HTTP/1.0 proxies que ignoran Cache-Control)."
    )


# ---------------------------------------------------------------------------
# 2. /history-list invoca el helper (regression guard del fix P3-A).
# ---------------------------------------------------------------------------
def test_history_list_endpoint_uses_helper():
    """`/history-list` (línea 6140 aprox) DEBE llamar `_apply_no_store(response)`.
    El sitio inline original (P1-HIST-AUDIT-4-FOLLOWUP) fue migrado al
    helper en P3-A — si alguien lo revierte sin actualizar este test,
    falla loud."""
    src = _read(_PLANS_PATH)

    # Aislamos el bloque del endpoint history-list. Captura desde el
    # decorador hasta el siguiente decorador (o EOF).
    m = re.search(
        r'@router\.get\(\s*[\'"]/history-list[\'"]\s*\).*?'
        r'(?=@router\.|$)',
        src,
        re.DOTALL,
    )
    assert m, "No se encontró el endpoint @router.get('/history-list')."
    block = m.group(0)

    assert "_apply_no_store(response)" in block, (
        "El endpoint `/history-list` DEBE invocar `_apply_no_store(response)` "
        "(SSOT P2-HIST-AUDIT-A). Si añadiste headers Cache-Control inline, "
        "drift contra los demás endpoints del Historial — usa el helper."
    )


# ---------------------------------------------------------------------------
# 3. Drift detection: nadie escribe Cache-Control inline en endpoints del
#    Historial (más allá del helper).
# ---------------------------------------------------------------------------
# Endpoints REST del Historial que DEBEN pasar por el helper. Mantener
# este set en sync con `_apply_no_store` callsites; añadir aquí cualquier
# nuevo endpoint del Historial que se agregue al módulo.
_HISTORIAL_ENDPOINTS = (
    "/history-list",
    "/lessons-counts",
    "/history-status-summary",
    "/{plan_id}/lessons",
    "/{plan_id}/coherence-history",
    "/{plan_id}/chunk-metrics",
    "/{plan_id}/lifetime-lessons",
    "/{plan_id}/blocked_reasons",
)


def _extract_endpoint_block(src: str, route: str) -> str | None:
    """Captura el cuerpo del endpoint cuyo decorador es `@router.get(<route>)`
    (los endpoints del Historial son todos GET). None si no existe."""
    pattern = (
        rf'@router\.get\(\s*[\'"]{re.escape(route)}[\'"]\s*\).*?'
        rf'(?=@router\.|$)'
    )
    m = re.search(pattern, src, re.DOTALL)
    return m.group(0) if m else None


def test_no_inline_cache_control_in_historial_endpoints():
    """Drift detection: ningún endpoint del Historial setea
    `response.headers["Cache-Control"] = ...` inline. Todos pasan por
    `_apply_no_store(response)`. Si añadiste un header nuevo (e.g.
    `Vary: Cookie`), añádelo al helper, no inline.

    Excepción documentada: SSE streams (`StreamingResponse(headers=...)`)
    usan semántica distinta y no están en este set.
    """
    src = _read(_PLANS_PATH)
    offenders = []
    for route in _HISTORIAL_ENDPOINTS:
        block = _extract_endpoint_block(src, route)
        if block is None:
            # Endpoint no encontrado — el test del set debería detectarlo
            # como drift (ver test_known_endpoints_exist abajo).
            continue
        # Pattern: response.headers["Cache-Control"] = "..."
        # con cualquier whitespace/quote-style.
        if re.search(
            r'response\.headers\s*\[\s*[\'"]Cache-Control[\'"]\s*\]\s*=',
            block,
        ):
            offenders.append(route)

    assert not offenders, (
        f"Endpoints del Historial con Cache-Control inline (deben usar "
        f"`_apply_no_store(response)` en su lugar): {offenders}. "
        f"SSOT: helper en `routers/plans.py:_apply_no_store` (P2-HIST-AUDIT-A). "
        f"Si necesitas un header NUEVO en estos endpoints, añádelo al "
        f"helper para que se propague a todos en lugar de duplicar inline."
    )


# ---------------------------------------------------------------------------
# 4. Sanity del set: cada endpoint listado existe en el módulo.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("route", _HISTORIAL_ENDPOINTS)
def test_known_endpoints_exist(route):
    """Si renombras o eliminas un endpoint del Historial, este test
    falla loud — y el de drift detection arriba quedaría inerte. Mantener
    `_HISTORIAL_ENDPOINTS` en sync con la realidad del módulo."""
    src = _read(_PLANS_PATH)
    block = _extract_endpoint_block(src, route)
    assert block is not None, (
        f"Endpoint `@router.get({route!r})` no encontrado en plans.py. "
        f"Si lo renombraste, actualiza `_HISTORIAL_ENDPOINTS` en este test. "
        f"Si lo eliminaste, también remuévelo del set."
    )
