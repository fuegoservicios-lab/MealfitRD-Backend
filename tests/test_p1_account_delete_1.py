"""[P1-ACCOUNT-DELETE-1 · 2026-06-22] Eliminación de cuenta SELF-SERVICE.

Contrato (parser-based, sin DB) del nuevo endpoint `POST /api/account/delete` y
sus dependencias. Cierra los gaps del audit de producción:
  - NO existía endpoint user-facing (solo el admin CRON_SECRET-gated).
  - El borrado NO cancelaba PayPal → cobro fantasma post-eliminación.
  - 3 tablas Dreaming faltaban de la lista de purga.
  - La sesión first-party no se invalidaba en el server.

Diseño: reusa el motor `delete_account_data` (P1-PROD-AUDIT-2); el uid sale SOLO
del token verificado (nunca del body — I2 / P0-AGENT-1); cancel-then-delete
fail-loud; confirm gate; throttle estricto.
"""
from pathlib import Path

_BE = Path(__file__).resolve().parent.parent
_APP = (_BE / "app.py").read_text(encoding="utf-8")
_BILLING = (_BE / "routers" / "billing.py").read_text(encoding="utf-8")
_DBP = (_BE / "db_profiles.py").read_text(encoding="utf-8")


def _endpoint_body() -> str:
    """Cuerpo de la función `api_delete_my_account` hasta el próximo `@app.`."""
    i = _APP.index("async def api_delete_my_account")
    nxt = _APP.index("\n@app.", i)
    return _APP[i:nxt]


def test_endpoint_route_exists():
    assert '@app.post("/api/account/delete")' in _APP


def test_uid_from_verified_token_not_body():
    body = _endpoint_body()
    # El uid sale del Depends del limiter (que llama get_verified_user_id),
    # NO de data.get("user_id") (eso sería IDOR como el endpoint admin).
    assert "Depends(_ACCOUNT_DELETE_LIMITER)" in body
    assert 'data.get("user_id")' not in body and "data.get('user_id')" not in body
    # Reject guest + 401 sin auth.
    assert 'verified_user_id == "guest"' in body
    assert "status_code=401" in body


def test_confirm_gate():
    body = _endpoint_body()
    assert 'confirm' in body and 'ELIMINAR' in body
    assert "status_code=400" in body  # confirm inválido


def test_cancel_then_delete_fail_loud():
    body = _endpoint_body()
    # El CALL real de cancel vive ANTES del CALL real de delete (no comparamos
    # menciones del docstring, que nombra delete_account_data primero).
    assert "await cancel_paypal_subscription_for_user(" in body
    assert "asyncio.to_thread(delete_account_data" in body
    assert body.index("await cancel_paypal_subscription_for_user(") < body.index("asyncio.to_thread(delete_account_data")
    # Lee la sub del perfil para decidir si cancelar.
    assert "paypal_subscription_id" in body


def test_session_cookie_invalidated():
    body = _endpoint_body()
    assert "clear_session_cookie(response)" in body


def test_strict_rate_limiter_declared():
    assert "_ACCOUNT_DELETE_LIMITER = RateLimiter(" in _APP
    # destructivo → throttle bajo (no copiar el quota/paywall).
    assert "verify_api_quota" not in _endpoint_body()


def test_billing_cancel_helper_is_fail_loud():
    i = _BILLING.index("async def cancel_paypal_subscription_for_user")
    nxt = _BILLING.index('\n@router.post("/cancel")', i)
    helper = _BILLING[i:nxt]
    # Reusa los primitivos existentes + fail-loud (raise + alerta).
    assert "_is_paypal_cancel_idempotent_success" in helper
    assert "_persist_billing_alert" in helper
    assert "raise HTTPException(" in helper
    assert "status_code=502" in helper  # PayPal rechazó / OAuth falló
    assert "status_code=503" in helper  # fail-secure: env PayPal ausente en prod
    assert "/v1/billing/subscriptions/" in helper and "/cancel" in helper


def test_dreaming_tables_in_purge_list():
    i = _DBP.index("_USER_SCOPED_TABLES_USERID = (")
    nxt = _DBP.index("\n)", i)  # cierre del tuple en su propia línea (no un ")" de comentario)
    tuple_src = _DBP[i:nxt]
    for t in ("user_memory_profile", "dream_work_queue", "dream_consolidation_log"):
        assert f'"{t}"' in tuple_src, f"falta {t} en la lista de purga"
    # meal_plans debe seguir AL FINAL (sus children FK-cascadean a él).
    assert tuple_src.rstrip().endswith('"meal_plans",')


def test_marker_bumped():
    assert '_LAST_KNOWN_PFIX = "P1-ACCOUNT-DELETE-1' in _APP


def test_frontend_calls_delete_endpoint():
    comp = _BE.parent / "frontend" / "src" / "components" / "account" / "DeleteAccountSection.jsx"
    if not comp.exists():
        return  # frontend repo puede no estar presente en algunos checkouts CI
    src = comp.read_text(encoding="utf-8")
    assert "/api/account/delete" in src
    assert "resetApp" in src           # logout completo post-borrado
    assert "ELIMINAR" in src           # confirm gate
    # La MISMA sección se monta en los DOS settings (landing + dashboard).
    for page in ("AccountSettings.jsx", "Settings.jsx"):
        p = _BE.parent / "frontend" / "src" / "pages" / page
        if p.exists():
            assert "DeleteAccountSection" in p.read_text(encoding="utf-8"), f"{page} no monta DeleteAccountSection"
