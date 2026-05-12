"""[P0-WEBHOOK-1 · 2026-05-12] `/api/webhooks/process-pending-facts` debe
fail-secure cuando falta `WEBHOOK_SECRET` en producción + comparar con
`hmac.compare_digest` (constant-time).

Bug pre-fix:
    if webhook_secret:
        # check token vs webhook_secret ...
    # ← sin else: si WEBHOOK_SECRET no estaba seteado, el check entero
    #   se saltaba y CUALQUIER llamada llegaba a process_pending_queue_sync(
    #   user_id_ajeno). Atacante con UUID enumerado podía forzar
    #   procesamiento sobre cuenta de víctima.

    Adicionalmente: `token != webhook_secret` es comparación normal
    (no constant-time) → timing oracle teórico ante secret largo.

Fix:
    1. `WEBHOOK_SECRET` ausente + `ENVIRONMENT=production` → 503.
    2. `hmac.compare_digest(token, webhook_secret)` en ambos slots
       (Authorization header y X-Webhook-Secret).
    3. Anchor `P0-WEBHOOK-1-FAIL-SECURE` en docstring.

Lo que este test enforza:
    A) `import hmac` aparece en `app.py`.
    B) `hmac.compare_digest(` se invoca dentro del handler.
    C) Patrón legacy `token != webhook_secret and custom_header_secret
       != webhook_secret` está ELIMINADO (sin else, ese path saltaba auth).
    D) `status_code=503` aparece asociado a `WEBHOOK_SECRET` ausente.
    E) Anchor `P0-WEBHOOK-1` permanece en source.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND_ROOT / "app.py"


@pytest.fixture(scope="module")
def app_src() -> str:
    return _APP_PY.read_text(encoding="utf-8")


def _isolate_handler(src: str) -> str:
    """Aísla `api_webhook_process_pending_facts` desde el decorator hasta
    el próximo `@app.` o `def api_` no parámetro.
    """
    pattern = (
        r'@app\.post\("/api/webhooks/process-pending-facts"\)'
        r'(.*?)(?=@app\.(?:get|post|put|delete|patch)|\Z)'
    )
    m = re.search(pattern, src, re.DOTALL)
    assert m is not None, "Handler webhook process-pending-facts no encontrado."
    return m.group(1)


def test_a_hmac_imported_or_local_use(app_src: str):
    """`hmac` debe estar disponible — top-level o lazy dentro del handler."""
    handler_body = _isolate_handler(app_src)
    assert "import hmac" in app_src or "import hmac" in handler_body, (
        "P0-WEBHOOK-1: `import hmac` ausente. Restaurar — necesario para "
        "constant-time compare del webhook secret."
    )


def test_b_constant_time_compare_invoked(app_src: str):
    """El handler invoca `hmac.compare_digest(` para comparar secret."""
    handler_body = _isolate_handler(app_src)
    assert "hmac.compare_digest(" in handler_body, (
        "P0-WEBHOOK-1: `hmac.compare_digest(` ausente en el handler. "
        "Pre-fix usaba `!=` directo — timing-oracle teórico."
    )


def test_c_legacy_unsafe_compare_gone(app_src: str):
    """El patrón `token != webhook_secret` (normal compare) NO debe
    aparecer ya en el handler — fue reemplazado por compare_digest.
    """
    handler_body = _isolate_handler(app_src)
    assert "token != webhook_secret" not in handler_body, (
        "P0-WEBHOOK-1 regresión: comparación normal `token != webhook_secret` "
        "reapareció. Usar `hmac.compare_digest(token, webhook_secret)`."
    )


def test_d_503_when_secret_missing_in_production(app_src: str):
    """El handler debe levantar 503 cuando `WEBHOOK_SECRET` es None y
    `ENVIRONMENT=production`. Pre-fix, el path entero del check se saltaba.
    """
    handler_body = _isolate_handler(app_src)
    assert "status_code=503" in handler_body, (
        "P0-WEBHOOK-1: el handler no levanta 503 en ningún path. "
        "Restaurar el gate fail-secure para secret ausente en producción."
    )
    # Asegurar que el 503 sí gatea por WEBHOOK_SECRET, no por otro estado.
    near = re.search(
        r"WEBHOOK_SECRET[\s\S]{0,400}status_code=503"
        r"|status_code=503[\s\S]{0,400}WEBHOOK_SECRET",
        handler_body,
    )
    assert near, (
        "P0-WEBHOOK-1: el 503 no parece ligado al check de WEBHOOK_SECRET. "
        "Verificar que el gate fail-secure efectivamente rechaza cuando "
        "el secret falta en producción."
    )


def test_e_environment_production_check(app_src: str):
    """El handler distingue producción vs dev — el 503 NO debe activarse
    fuera de producción (sino rompería dev local sin secret).
    """
    handler_body = _isolate_handler(app_src)
    assert 'ENVIRONMENT' in handler_body or 'is_production' in handler_body, (
        "P0-WEBHOOK-1: el handler no consulta `ENVIRONMENT`. El gate "
        "fail-secure debe ser solo prod, no dev."
    )


def test_f_anchor_present(app_src: str):
    handler_body = _isolate_handler(app_src)
    assert "P0-WEBHOOK-1" in handler_body, (
        "P0-WEBHOOK-1: anchor `P0-WEBHOOK-1` desapareció del handler. "
        "Restaurar en el docstring."
    )
