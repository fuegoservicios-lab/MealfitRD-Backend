"""[P2-SENTRY-PII-SCRUBBING-FRONTEND · 2026-05-15] Anchor + regression guard.

Pre-fix `frontend/src/main.jsx::Sentry.init` solo tenía `replayIntegration({
maskAllText: true, blockAllMedia: true })` — masking de replays (vídeo del
DOM) OK, pero error events normales (`Sentry.captureException(err, { extra:
{ body } })`) llegaban con request body/headers/extras sin redacción.
Verificado: `grep beforeSend` → 0 matches. GDPR-relevant + risk de leak de
tokens si Sentry se ve comprometido.

Defensas que este test enforza:
  1. Anchor `P2-SENTRY-PII-SCRUBBING-FRONTEND` presente en `frontend/src/main.jsx`.
  2. Helpers `_sentryBeforeSend` y `_sentryBeforeBreadcrumb` declarados.
  3. Tuple `SENTRY_SENSITIVE_KEY_SUBSTRINGS` incluye keys críticas
     (mínimo: password, token, authorization, cookie, email, health_profile,
     plan_data).
  4. `Sentry.init({...})` referencia `beforeSend: _sentryBeforeSend` Y
     `beforeBreadcrumb: _sentryBeforeBreadcrumb` (ambos obligatorios).
  5. El bloque del filtro es fail-open (`try { ... } catch { /* fail-open */ }`)
     — un crash interno del filtro NO debe dropear el event.

Test parser-based — no levanta browser, solo escanea source con regex.
"""
from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND_MAIN = _REPO_ROOT / "frontend" / "src" / "main.jsx"


def _read_main() -> str:
    return _FRONTEND_MAIN.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Anchor + estructura
# ---------------------------------------------------------------------------
def test_anchor_present_in_frontend_main():
    src = _read_main()
    assert "P2-SENTRY-PII-SCRUBBING-FRONTEND" in src, (
        "Falta anchor `P2-SENTRY-PII-SCRUBBING-FRONTEND` en frontend/src/main.jsx. "
        "Sin anchor, un futuro reader no sabrá el modo de fallo que cierra."
    )


def test_helpers_defined():
    src = _read_main()
    assert "_sentryBeforeSend" in src, (
        "Falta función `_sentryBeforeSend` en main.jsx."
    )
    assert "_sentryBeforeBreadcrumb" in src, (
        "Falta función `_sentryBeforeBreadcrumb` en main.jsx."
    )


def test_sensitive_key_list_includes_critical_substrings():
    """`SENTRY_SENSITIVE_KEY_SUBSTRINGS` debe incluir como mínimo las keys
    que sabemos que viajan en el código del cliente (tokens, PayPal,
    health profile)."""
    src = _read_main()
    m = re.search(
        r"SENTRY_SENSITIVE_KEY_SUBSTRINGS\s*=\s*\[(.+?)\]",
        src,
        re.DOTALL,
    )
    assert m is not None, "No se encontró array `SENTRY_SENSITIVE_KEY_SUBSTRINGS`."
    body = m.group(1).lower()
    must_have = ("password", "token", "authorization", "health_profile", "plan_data", "cookie", "email")
    missing = [k for k in must_have if k not in body]
    assert not missing, (
        f"`SENTRY_SENSITIVE_KEY_SUBSTRINGS` no incluye: {missing}. "
        f"Esos substrings son críticos por modo de fallo conocido en el código del cliente."
    )


def test_sentry_init_uses_before_send_and_before_breadcrumb():
    """`Sentry.init({...})` DEBE pasar `beforeSend` y `beforeBreadcrumb`."""
    src = _read_main()
    m = re.search(r"Sentry\.init\(\{\s*(.*?)\n\}\)", src, re.DOTALL)
    assert m is not None, "No se encontró bloque `Sentry.init({...})` en main.jsx"
    block = m.group(1)
    assert "beforeSend: _sentryBeforeSend" in block, (
        "Sentry.init debe pasar `beforeSend: _sentryBeforeSend`."
    )
    assert "beforeBreadcrumb: _sentryBeforeBreadcrumb" in block, (
        "Sentry.init debe pasar `beforeBreadcrumb: _sentryBeforeBreadcrumb`."
    )


def test_helpers_are_fail_open():
    """Tanto `_sentryBeforeSend` como `_sentryBeforeBreadcrumb` deben
    envolver su lógica en `try { ... } catch { ... }` para no dropear
    events por error interno del filtro."""
    src = _read_main()
    # Buscar el cuerpo de cada helper hasta el cierre `};`
    for fn_name in ("_sentryBeforeSend", "_sentryBeforeBreadcrumb"):
        m = re.search(
            rf"const\s+{re.escape(fn_name)}\s*=\s*\([^)]*\)\s*=>\s*\{{(.+?)^\}};",
            src,
            re.DOTALL | re.MULTILINE,
        )
        assert m is not None, f"No se pudo extraer cuerpo de `{fn_name}`."
        body = m.group(1)
        assert "try {" in body and "catch" in body, (
            f"`{fn_name}` debe tener `try/catch` fail-open. Sin esto, "
            f"un crash del filtro dropea el event a Sentry."
        )


# ---------------------------------------------------------------------------
# 2. Cross-link guard (P2-HIST-AUDIT-14): el slug del marker actual matchea
#    este archivo (auto-satisfecho por nombre del file).
# ---------------------------------------------------------------------------
def test_anchor_present_in_test_file():
    src = Path(__file__).read_text(encoding="utf-8")
    assert "P2-SENTRY-PII-SCRUBBING-FRONTEND" in src
