"""[P2-AUDIT-4 · 2026-05-12] Todo endpoint con prefijo `/admin/` en routers
DEBE invocar `_verify_admin_token` en su cuerpo (antes de tocar DB).

Contexto:
    El audit production-readiness 2026-05-12 reveló 8 endpoints `/admin/*`
    en `routers/plans.py` + 1 en `routers/system.py`. Inspección manual
    confirmó que los 9 ya invocan `_verify_admin_token`. Este test ancla
    la convención para que un futuro endpoint admin no se mergee sin auth.

    Mismo patrón documentado en `test_p1_audit_new_1_debug_scaling_removed`
    (que cubre `/debug/*` rutas) — esta es la simétrica para `/admin/*`.

    Sin el test, un dev podría añadir `@router.get("/admin/new-endpoint")`
    sin auth y nadie nota hasta que un probe lo encuentra. Los endpoints
    admin típicamente exponen datos sensibles (métricas de flota,
    administración de chunks de cualquier user, etc.) — bypass de auth
    aquí es IDOR universal con privilegios elevados.

Tooltip-anchor: P2-AUDIT-4-ADMIN-TOKEN.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_ROUTERS_DIR = _BACKEND_ROOT / "routers"

# Regex que detecta `@router.<method>("/admin/...")` decorators.
_ADMIN_ROUTE_PATTERN = re.compile(
    r'@router\.(?P<method>get|post|put|delete|patch)\(\s*[\"\'](?P<route>\/admin\/[^\"\']*)[\"\']',
    re.IGNORECASE,
)


def _iter_admin_endpoints():
    """Yields (router_filename, line_no, method, route, body_text) por cada
    endpoint con prefijo `/admin/`."""
    for fp in sorted(_ROUTERS_DIR.glob("*.py")):
        if fp.name == "__init__.py":
            continue
        src = fp.read_text(encoding="utf-8")
        for m in _ADMIN_ROUTE_PATTERN.finditer(src):
            line_no = src[: m.start()].count("\n") + 1
            # Cuerpo del handler: desde el decorador hasta el próximo
            # `@router.` o EOF. Cap a 3000 chars para evitar matches lejanos.
            next_decorator = src.find("\n@router.", m.end())
            end = next_decorator if next_decorator != -1 else m.end() + 3000
            body = src[m.start() : min(end, m.end() + 3000)]
            yield (fp.name, line_no, m.group("method"), m.group("route"), body)


def test_at_least_one_admin_endpoint_exists():
    """Sanity: hay al menos 1 endpoint /admin/* en los routers. Si baja
    a 0, el test pasa vacío — síntoma de un refactor que hay que revisar.
    """
    count = sum(1 for _ in _iter_admin_endpoints())
    assert count >= 5, (
        f"P2-AUDIT-4: solo {count} endpoint(s) /admin/* encontrados. El "
        f"audit 2026-05-12 identificó 9. Si bajaron a <5, hubo un refactor "
        f"masivo — revisar antes de aceptar."
    )


def test_every_admin_endpoint_calls_verify_admin_token():
    """Cada endpoint con prefijo `/admin/` DEBE invocar
    `_verify_admin_token` en su cuerpo. Sin este check, el endpoint expone
    funcionalidad admin a cualquier requester con la URL.

    Acepta `_verify_admin_token` en cualquier formato:
    - `_verify_admin_token(request.headers.get("authorization"))`
    - `_verify_admin_token(authorization)`
    - `await _verify_admin_token(...)`
    """
    violations = []
    for filename, line_no, method, route, body in _iter_admin_endpoints():
        if "_verify_admin_token" not in body[:2500]:
            violations.append(f"{filename}:{line_no} {method.upper()} {route}")
    assert not violations, (
        "P2-AUDIT-4 violation: endpoint(s) con prefijo /admin/ NO invocan "
        "`_verify_admin_token`. Bypass de auth para funcionalidad admin "
        "(exposición de métricas de flota, gestión de chunks cross-user, "
        "etc.).\n\n"
        "Violations:\n  " + "\n  ".join(violations) + "\n\n"
        "Fix: añadir al inicio del handler:\n"
        "  _verify_admin_token(request.headers.get(\"authorization\"))\n"
        "Asegurarse de incluir `request: Request` en la signature y "
        "lazy-import `_verify_admin_token` si causa ciclo de imports."
    )


def test_admin_token_call_appears_before_db_access():
    """Defense-in-depth: el `_verify_admin_token(...)` debe aparecer ANTES
    de cualquier llamada a DB (`execute_sql_*`, `supabase.table(`, etc.).
    Si va después, la query se ejecuta y luego se rechaza — leak de carga
    + posibles side-effects.
    """
    violations = []
    db_call_pattern = re.compile(
        r"(execute_sql_query|execute_sql_write|supabase\.(?:table|rpc|from_)\b)",
    )
    for filename, line_no, method, route, body in _iter_admin_endpoints():
        token_match = re.search(r"_verify_admin_token\s*\(", body)
        db_match = db_call_pattern.search(body[:3000])
        if not token_match:
            continue  # cubierto por test arriba
        if db_match and db_match.start() < token_match.start():
            violations.append(
                f"{filename}:{line_no} {method.upper()} {route} — "
                f"DB call ({db_match.group(0)}) aparece ANTES del "
                f"_verify_admin_token. Mover el token check al inicio."
            )
    assert not violations, (
        "P2-AUDIT-4 violation: DB access antes de `_verify_admin_token`:\n"
        "  " + "\n  ".join(violations)
    )
