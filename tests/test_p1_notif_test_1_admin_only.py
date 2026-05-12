"""[P1-NOTIF-TEST-1 · 2026-05-11] `/api/notifications/test` admin-only.

Bug original (audit 2026-05-11):
    El endpoint declaraba `test_push_route(user_id: str)` SIN auth. Cualquiera
    con la URL del backend + un `user_id` válido (UUIDs leak fácilmente
    via endpoints GET o enumeración) podía spamear push notifications a
    los devices del usuario. Sin frontend caller (verificado vía grep
    cross-codebase) — el endpoint era pure debug montado en producción.

Cierre:
    `_verify_admin_token(authorization)` — mismo gate que
    `/api/system/admin/plan-graph/invalidate` y `/api/plans/admin/...`.
    Requiere `Authorization: Bearer <CRON_SECRET>`. Sin CRON_SECRET en
    el ambiente, responde 503 (fail-secure: admin endpoints no se exponen
    sin secreto).

Tests parser-based — verifica que el gate admin esté in place sin
levantar la app FastAPI completa (que requeriría DB + connection_pool).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_NOTIF_FP = _REPO_ROOT / "backend" / "routers" / "notifications.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _NOTIF_FP.read_text(encoding="utf-8")


def _extract_test_endpoint_block(src: str) -> str:
    """Extrae el block del endpoint `/test` (desde `@router.get(\"/test\")`
    hasta el siguiente `@router.` o EOF)."""
    start_marker = '@router.get("/test")'
    start = src.find(start_marker)
    assert start > 0, "endpoint /test no encontrado en notifications.py"
    rest = src[start + len(start_marker):]
    next_decorator = re.search(r"\n(@router\.|@app\.)", rest)
    end_offset = next_decorator.start() if next_decorator else len(rest)
    return src[start: start + len(start_marker) + end_offset]


def test_endpoint_signature_includes_request(src: str):
    """`async def test_push_route(user_id: str, request: Request)` —
    request param requerido para extraer Authorization header."""
    block = _extract_test_endpoint_block(src)
    assert "request: Request" in block, (
        "P1-NOTIF-TEST-1 regresión: el handler `test_push_route` ya no "
        "acepta `request: Request` en su signature. Sin él, no podemos "
        "leer `request.headers.get('authorization')` para el verify_admin "
        "gate. Restaurar el parámetro."
    )


def test_endpoint_calls_verify_admin_token(src: str):
    """`_verify_admin_token(request.headers.get(\"authorization\"))` debe
    aparecer ANTES de cualquier acceso a `connection_pool` o lectura SQL."""
    block = _extract_test_endpoint_block(src)
    assert "_verify_admin_token(request.headers.get(\"authorization\"))" in block, (
        "P1-NOTIF-TEST-1 regresión CRÍTICA: el gate admin "
        "`_verify_admin_token(request.headers.get('authorization'))` "
        "desapareció del endpoint `/test`. Sin él, el endpoint vuelve a ser "
        "público y permite spam de push notifications a cualquier user_id "
        "conocido. Restaurar el gate como PRIMERA línea ejecutable del "
        "handler (antes del `if not connection_pool`)."
    )


def test_admin_gate_runs_before_db_access(src: str):
    """El gate debe correr ANTES del check `if not connection_pool` y de
    cualquier acceso SQL — sin esto, el atacante consume DB pool resources
    incluso si después rechazamos por auth."""
    block = _extract_test_endpoint_block(src)
    gate_idx = block.find("_verify_admin_token(request.headers.get(\"authorization\"))")
    pool_idx = block.find("if not connection_pool:")
    assert gate_idx > 0, "gate no encontrado (ya cubierto por test anterior)"
    assert pool_idx > 0, "check connection_pool no encontrado"
    assert gate_idx < pool_idx, (
        f"P1-NOTIF-TEST-1 regresión: el gate admin se ejecuta DESPUÉS del "
        f"check de connection_pool (gate_idx={gate_idx}, pool_idx={pool_idx}). "
        f"Esto permite que un atacante anónimo consuma resources del DB pool "
        f"antes del rechazo. Mover el gate como PRIMERA línea ejecutable del "
        f"body del handler."
    )


def test_anchor_for_audit_present(src: str):
    """Tooltip-anchor `P1-NOTIF-TEST-1-AUTH` para grep cross-codebase."""
    block = _extract_test_endpoint_block(src)
    assert "P1-NOTIF-TEST-1-AUTH" in block, (
        "P1-NOTIF-TEST-1 regresión: tooltip-anchor "
        "`P1-NOTIF-TEST-1-AUTH` desapareció. Sin él, un grep desde otro "
        "test/code o desde el plan de un audit futuro no encuentra esta "
        "función como ejemplo del patrón admin-gated."
    )


def test_lazy_import_of_admin_helper(src: str):
    """`_verify_admin_token` debe importarse lazy (dentro del handler) para
    evitar crear ciclo de imports `notifications → plans → ...`. Patrón
    espejo del que usa `system.py:19`."""
    block = _extract_test_endpoint_block(src)
    assert "from routers.plans import _verify_admin_token" in block, (
        "P1-NOTIF-TEST-1: import lazy de `_verify_admin_token` ya no "
        "está dentro del handler. Si moviste el import al top-level, "
        "verificar que no cree ciclo notifications↔plans (system.py lo "
        "evita con import top-level porque el ciclo aquí es benign — pero "
        "el lazy import es defensa-en-profundidad). Mantenerlo lazy o "
        "documentar la decisión."
    )


def test_no_get_verified_user_id_path_left(src: str):
    """Doble check: no debe quedar un path alternativo donde el endpoint
    valide solo `Depends(get_verified_user_id)` (eso permitiría que cualquier
    user autenticado spamee a cualquier otro user_id). Solo admin gate."""
    block = _extract_test_endpoint_block(src)
    assert "Depends(get_verified_user_id)" not in block, (
        "P1-NOTIF-TEST-1 regresión: el endpoint `/test` está usando "
        "`Depends(get_verified_user_id)` en lugar (o además) del admin gate. "
        "Eso permite que CUALQUIER user autenticado (no solo admin) spamee "
        "push a CUALQUIER user_id. El endpoint debe ser admin-only — sin "
        "frontend caller, no hay UX legítima de user-self-test desde la app "
        "(verificado vía grep en frontend/src). Si en el futuro agregas un "
        "botón \"test push\" en Settings, crear un endpoint NUEVO "
        "`/api/notifications/self-test` con `Depends(get_verified_user_id)` "
        "y forzar `user_id == verified_user_id`."
    )
