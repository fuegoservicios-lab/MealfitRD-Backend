"""[P1-SYSTEM-HEALTH-ADMIN-GATE · 2026-05-12] Anchor + regression guard.

`GET /api/system/health` (`backend/routers/system.py:get_system_health`)
agrega métricas de negocio sobre toda la flota:
  - Nudge response rate global.
  - Distribución de razones de abandono (churn drivers).
  - Distribución emocional de respuestas (sentiment del usuario).
  - Quality score promedio percibido.

Pre-fix era público. No expone PII (filtra `health_profile->quality_history`
agregando solo el avg), pero SÍ expone **business-intel** que un competidor
puede polletar cada minuto para reconstruir un dashboard de la salud
comercial del producto: % de usuarios que responden a nudges, qué fricciones
causan abandono más frecuente, sentiment promedio, calidad percibida.

Defensa: gate `_verify_admin_token(...)` igual que el resto de endpoints
operacionales (`/admin/chunks/stuck`, `/admin/deploy-lag/check`, etc.). El
mismo CRON_SECRET que ya protege los demás admin endpoints.

Defensas que el test enforza:
  1. Anchor `P1-SYSTEM-HEALTH-ADMIN-GATE` presente en el handler.
  2. La función `get_system_health` acepta `request: Request` parameter.
  3. `_verify_admin_token(request.headers.get("authorization"))` se invoca
     ANTES del primer `execute_sql_query(...)` — sin esto, una excepción
     dentro del SELECT podría leakear estructura via stack trace antes de
     llegar al guard.
  4. El módulo importa `_verify_admin_token` desde `routers.plans`.

Test parser-based — no levanta el server, solo escanea source.
"""

from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SYSTEM_PY = _REPO_ROOT / "backend" / "routers" / "system.py"


def _read() -> str:
    return _SYSTEM_PY.read_text(encoding="utf-8")


def _extract_handler_body(src: str) -> str:
    """Aísla el cuerpo de `get_system_health(...)`: desde la línea de
    decorador `@router.get("/health")` hasta el siguiente `@router.` o
    EOF, lo que llegue primero."""
    m = re.search(
        r'@router\.get\(\s*[\"\']/health[\"\']\s*\)\s*\n'
        r'def\s+get_system_health\s*\([^)]*\)\s*:\s*\n'
        r'(.*?)'
        r'(?=\n@router\.|\Z)',
        src,
        re.DOTALL,
    )
    assert m is not None, (
        "No se encontró handler `get_system_health` decorado con `@router.get('/health')`."
    )
    return m.group(1)


def test_anchor_present_in_handler():
    src = _read()
    body = _extract_handler_body(src)
    assert "P1-SYSTEM-HEALTH-ADMIN-GATE" in body, (
        "Falta anchor `P1-SYSTEM-HEALTH-ADMIN-GATE` en el docstring/cuerpo "
        "de `get_system_health`. Sin anchor, un futuro reader que vea "
        "`_verify_admin_token(...)` puede pensar que es boilerplate y "
        "removerlo para 'simplificar' un health probe."
    )


def test_handler_accepts_request_parameter():
    """Para invocar `_verify_admin_token(request.headers.get(...))`, el
    handler debe declarar `request: Request` parameter. FastAPI lo inyecta
    automáticamente."""
    src = _read()
    pat = re.compile(
        r'def\s+get_system_health\s*\(\s*request\s*:\s*Request\s*\)\s*:',
    )
    assert pat.search(src), (
        "`get_system_health` debe declarar `request: Request` parameter "
        "para que FastAPI inyecte el objeto Request (necesario para leer "
        "el header Authorization)."
    )


def test_verify_admin_token_invoked_before_sql():
    """El gate debe correr ANTES del primer `execute_sql_query(...)`. Si
    se invoca después, una excepción dentro del SELECT podría leakear
    estructura del schema via stack trace antes de llegar al guard."""
    src = _read()
    body = _extract_handler_body(src)

    # Posición del verify_admin_token call
    m_verify = re.search(
        r'_verify_admin_token\(\s*request\.headers\.get\(\s*[\"\']authorization[\"\']\s*\)\s*\)',
        body,
    )
    assert m_verify is not None, (
        "Falta `_verify_admin_token(request.headers.get(\"authorization\"))` "
        "en el handler. Sin gate, el endpoint sigue público y expone "
        "business-intel agregada."
    )

    # Posición del primer execute_sql_query
    m_sql = re.search(r"execute_sql_query\(", body)
    assert m_sql is not None, (
        "El handler ya no tiene `execute_sql_query(...)` — verificar si "
        "el endpoint fue refactorizado. Si fue removido y reemplazado con "
        "otro mecanismo de query, actualizar este test al nuevo guard."
    )

    assert m_verify.start() < m_sql.start(), (
        f"`_verify_admin_token` se invoca DESPUÉS del primer "
        f"`execute_sql_query` (posiciones {m_verify.start()} vs "
        f"{m_sql.start()}). El gate debe correr PRIMERO."
    )


def test_module_imports_verify_admin_token():
    """El módulo debe importar `_verify_admin_token` desde routers.plans
    (SSOT del helper). Importarlo localmente abriría drift de implementación."""
    src = _read()
    pat = re.compile(
        r"from\s+routers\.plans\s+import\s+_verify_admin_token",
    )
    assert pat.search(src), (
        "`routers/system.py` debe importar `_verify_admin_token` desde "
        "`routers.plans` (SSOT). Sin esto, drift de implementación entre "
        "los handlers admin."
    )


def test_anchor_present_in_test_file():
    """Cross-link guard P2-HIST-AUDIT-14: el slug del marker
    `P1-SYSTEM-HEALTH-ADMIN-GATE` → `p1_system_health_admin_gate` debe
    matchear este archivo `tests/test_<slug>*.py`."""
    src = Path(__file__).read_text(encoding="utf-8")
    assert "P1-SYSTEM-HEALTH-ADMIN-GATE" in src
