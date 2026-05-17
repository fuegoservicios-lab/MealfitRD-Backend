"""[P3-HEALTH-AGGREGATES-DISCLOSURE-DEFERRED · 2026-05-15] Decision anchor.

Pattern análogo a `P3-I18N-DEFERRED · 2026-05-13`: lo que un auditor flag
como deuda es una decisión de producto. Los 5 endpoints públicos siguientes
en `backend/routers/system.py` exponen agregados operacionales sin auth
(UUIDs ya hasheados via `P2-HEALTH-UID-STRIP`):

  - `/api/system/atomic-pool-health`
  - `/api/system/chunk-queue-health`
  - `/api/system/pantry-tolerance-health`
  - `/api/system/tz-fallback-health`
  - `/api/system/health/plan-graph`

Decisión documentada (2026-05-15): mantenerlos públicos. Razones detalladas
en [`project_p3_health_aggregates_disclosure_deferred_2026_05_15.md`](
~/.claude/projects/.../memory/) — esencialmente: utilidad operacional para
Grafana/k8s/UptimeRobot externos > el costo de business-intel leak agregado
sin PII.

Defensas que este test enforza:
  1. Anchor `P3-HEALTH-AGGREGATES-DISCLOSURE-DEFERRED` presente en
     `routers/system.py` (sirve como pointer a la decisión para futuros
     readers que vean los endpoints sin auth y quieran "arreglarlos").
  2. Los 5 endpoints siguen siendo públicos (NO llaman `_verify_admin_token`
     ni `_check_admin_rate_limit` dentro de sus bodies). Si alguien gateara
     uno, el test falla con un mensaje que apunta a la memoria — forzando
     reactualizar la decisión EXPLÍCITAMENTE (no de paso).
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_SYSTEM_PY = _BACKEND_ROOT / "routers" / "system.py"


_DEFERRED_PUBLIC_ENDPOINTS = [
    "/atomic-pool-health",
    "/chunk-queue-health",
    "/pantry-tolerance-health",
    "/tz-fallback-health",
    "/health/plan-graph",
]


def _read() -> str:
    return _SYSTEM_PY.read_text(encoding="utf-8")


def test_anchor_present_in_system_py():
    src = _read()
    assert "P3-HEALTH-AGGREGATES-DISCLOSURE-DEFERRED" in src, (
        "Falta anchor `P3-HEALTH-AGGREGATES-DISCLOSURE-DEFERRED` en "
        "`routers/system.py`. Sin anchor, un reader que ve los 5 endpoints "
        "sin auth no sabrá que es decisión consciente."
    )


def _extract_endpoint_body(src: str, route: str) -> str:
    """Extrae el cuerpo del handler de `@router.get('<route>')`.

    Devuelve desde la firma `def ...(...)` hasta antes del próximo
    `@router.` o EOF. Suficiente para escanear si llama
    `_verify_admin_token`/`_check_admin_rate_limit`.
    """
    pat = re.compile(
        r"@router\.get\(\s*[\"']"
        + re.escape(route)
        + r"[\"']\s*\)\s*\n(?:.*?\n)*?(?=@router\.|$)",
        re.DOTALL,
    )
    m = pat.search(src)
    if m is None:
        return ""
    return m.group(0)


def test_5_endpoints_remain_public():
    """Los 5 endpoints flagged en la decisión NO deben llamar al admin
    gate ni al rate limiter. Si alguien gatea uno, esto fuerza
    reactualización explícita de la decisión."""
    src = _read()
    violators = []
    for route in _DEFERRED_PUBLIC_ENDPOINTS:
        body = _extract_endpoint_body(src, route)
        if not body:
            violators.append(f"{route}: handler NO encontrado — ¿se renombró o removió?")
            continue
        if "_verify_admin_token(" in body:
            violators.append(f"{route}: gatea con `_verify_admin_token` — viola la decisión.")
        if "_check_admin_rate_limit(" in body:
            violators.append(f"{route}: aplica rate limit admin — viola la decisión.")
    assert not violators, (
        "P3-HEALTH-AGGREGATES-DISCLOSURE-DEFERRED violación:\n  "
        + "\n  ".join(violators)
        + "\n\nLa decisión documentada en "
        "`memory/project_p3_health_aggregates_disclosure_deferred_2026_05_15.md` "
        "establece que estos 5 endpoints quedan PÚBLICOS. Si querés revertir "
        "esa decisión, actualizá la memoria + este test + el anchor inline "
        "primero, NO al revés."
    )


def test_anchor_present_in_test_file():
    src = Path(__file__).read_text(encoding="utf-8")
    assert "P3-HEALTH-AGGREGATES-DISCLOSURE-DEFERRED" in src
