"""[P1-SENTRY-PII-SCRUBBING-BACKEND · 2026-05-15] Anchor + regression guard.

Pre-fix `sentry_sdk.init(...)` corría sin `before_send` ni `before_breadcrumb`.
Cualquier excepción levantada dentro de un endpoint capturaba request body,
headers (incluyendo `Authorization`), cookies, query string y locals de la
frame automáticamente. Eso incluye `health_profile` (peso/altura/condiciones),
tokens PayPal en `/api/subscription/verify`, body completo del chat,
JWTs en headers `Authorization`. GDPR/HIPAA-relevant.

Defensas que este test enforza:
  1. Anchor `P1-SENTRY-PII-SCRUBBING-BACKEND` presente en `backend/app.py`.
  2. Helpers `_sentry_redact_pii` y `_sentry_redact_breadcrumb` definidos.
  3. `sentry_sdk.init(...)` referencia `before_send=_sentry_redact_pii` Y
     `before_breadcrumb=_sentry_redact_breadcrumb` (ambos obligatorios).
  4. `_SENSITIVE_KEY_SUBSTRINGS` incluye keys críticas (mínimo: password,
     token, authorization, health_profile, plan_data).
  5. Funcional: el helper realmente redacta keys conocidas, incluso a profundidad.
  6. Funcional: NUNCA dropea event por exception del filtro (fail-open
     deliberado — mejor PII filtrada incorrectamente que perder errores).
"""
from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_APP = _REPO_ROOT / "backend" / "app.py"


def _read_app() -> str:
    return _BACKEND_APP.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Anchors y estructura del fix
# ---------------------------------------------------------------------------
def test_anchor_present_in_backend_app():
    src = _read_app()
    assert "P1-SENTRY-PII-SCRUBBING-BACKEND" in src, (
        "Falta anchor `P1-SENTRY-PII-SCRUBBING-BACKEND` en backend/app.py. "
        "Sin anchor, un futuro reader que vea `before_send=` no sabrá el "
        "modo de fallo que cierra (PII leak a Sentry)."
    )


def test_helpers_defined():
    src = _read_app()
    assert "def _sentry_redact_pii(" in src, (
        "Falta helper `_sentry_redact_pii(event, hint)` en backend/app.py."
    )
    assert "def _sentry_redact_breadcrumb(" in src, (
        "Falta helper `_sentry_redact_breadcrumb(crumb, hint)` en backend/app.py."
    )


def test_sensitive_key_list_includes_critical_substrings():
    """El tuple `_SENSITIVE_KEY_SUBSTRINGS` debe incluir como mínimo las
    keys que sabemos que viajan en el código (auth tokens, PayPal, health)."""
    src = _read_app()
    m = re.search(
        r"_SENSITIVE_KEY_SUBSTRINGS\s*=\s*\((.+?)\)",
        src,
        re.DOTALL,
    )
    assert m is not None, "No se encontró tuple `_SENSITIVE_KEY_SUBSTRINGS`."
    body = m.group(1).lower()
    must_have = ("password", "token", "authorization", "health_profile", "plan_data", "cookie")
    missing = [k for k in must_have if k not in body]
    assert not missing, (
        f"`_SENSITIVE_KEY_SUBSTRINGS` no incluye: {missing}. "
        f"Esos substrings son críticos por modo de fallo conocido en el código."
    )


def test_sentry_init_uses_before_send_and_before_breadcrumb():
    """`sentry_sdk.init(...)` DEBE pasar tanto `before_send` como
    `before_breadcrumb` con los helpers canónicos. Sin breadcrumb, URLs
    con `?token=...` van limpias a Sentry."""
    src = _read_app()
    m = re.search(r"sentry_sdk\.init\(\s*(.*?)\n\)", src, re.DOTALL)
    assert m is not None, "No se encontró bloque `sentry_sdk.init(...)` en app.py"
    block = m.group(1)
    assert "before_send=_sentry_redact_pii" in block, (
        "sentry_sdk.init debe pasar `before_send=_sentry_redact_pii`."
    )
    assert "before_breadcrumb=_sentry_redact_breadcrumb" in block, (
        "sentry_sdk.init debe pasar `before_breadcrumb=_sentry_redact_breadcrumb`."
    )


# ---------------------------------------------------------------------------
# 2. Funcional: el filtro realmente redacta
# ---------------------------------------------------------------------------
def _load_helpers():
    """Carga los helpers sin ejecutar el module-init completo de app.py
    (que dispara DB pool, sentry_sdk.init, schedulers). Usa `compile + exec`
    sobre solo el bloque relevante para asilarlo de los side effects.

    Estrategia: extraer desde `_SENSITIVE_KEY_SUBSTRINGS = (` hasta la
    línea anterior a `sentry_sdk.init(`. Los helpers son contiguos en el
    source entre esos dos anchors.
    """
    src = _read_app()
    start_match = re.search(r"^_SENSITIVE_KEY_SUBSTRINGS\s*=", src, re.MULTILINE)
    assert start_match is not None, "No se encontró `_SENSITIVE_KEY_SUBSTRINGS = (` en app.py."
    end_match = re.search(r"\nsentry_sdk\.init\(", src[start_match.start():])
    assert end_match is not None, "No se encontró `sentry_sdk.init(` tras los helpers."
    code = src[start_match.start(): start_match.start() + end_match.start()]
    ns: dict = {}
    exec(compile(code, "<helpers>", "exec"), ns)
    return ns


def test_redact_pii_filters_known_keys():
    helpers = _load_helpers()
    redact = helpers["_sentry_redact_pii"]
    event = {
        "request": {
            "data": {
                "email": "user@example.com",
                "password": "hunter2",
                "regular_field": "ok",
                "health_profile": {"weight": 80, "height": 175},
            },
            "headers": {
                "Authorization": "Bearer abc.def.ghi",
                "X-Forwarded-For": "1.2.3.4",
            },
        },
        "user": {"email": "user@example.com", "id": "uuid-here"},
        "extra": {"plan_data": {"days": []}, "session_id": "sess-123"},
    }
    out = redact(event, None)
    assert out is event, "before_send debe retornar el event (no None)."
    assert out["request"]["data"]["email"] == "[Filtered]"
    assert out["request"]["data"]["password"] == "[Filtered]"
    assert out["request"]["data"]["regular_field"] == "ok", (
        "Keys no-sensitive NO deben mutarse."
    )
    assert out["request"]["data"]["health_profile"] == "[Filtered]"
    assert out["request"]["headers"]["Authorization"] == "[Filtered]"
    assert out["request"]["headers"]["X-Forwarded-For"] == "1.2.3.4"
    assert out["user"]["email"] == "[Filtered]"
    assert out["user"]["id"] == "uuid-here", "user.id NO es PII en sí — no redactar."
    assert out["extra"]["plan_data"] == "[Filtered]"
    assert out["extra"]["session_id"] == "sess-123"


def test_redact_breadcrumb_filters_url_query_string():
    helpers = _load_helpers()
    redact = helpers["_sentry_redact_breadcrumb"]
    crumb = {
        "category": "http",
        "message": "GET https://api.example.com/x?token=secret123&other=val",
    }
    out = redact(crumb, None)
    assert "[Filtered]" in out["message"]
    assert "secret123" not in out["message"]

    # URL sin keys sensitivas: no muta.
    crumb2 = {"category": "http", "message": "GET https://api.example.com/x?ok=1"}
    out2 = redact(crumb2, None)
    assert out2["message"] == "GET https://api.example.com/x?ok=1"


def test_redact_does_not_drop_event_on_internal_error():
    """Fail-open: si algo del filtro lanza, el event sigue. Mejor PII
    enviada que un error genuino perdido."""
    helpers = _load_helpers()
    redact = helpers["_sentry_redact_pii"]
    # Input patológico — request es un objeto que no es dict.
    weird_event = {"request": "not-a-dict", "extra": None}
    out = redact(weird_event, None)
    assert out is weird_event, "Event NO debe dropearse aunque request sea raro."


# ---------------------------------------------------------------------------
# 3. Cross-link guard (P2-HIST-AUDIT-14)
# ---------------------------------------------------------------------------
def test_anchor_present_in_test_file():
    """El slug del marker `p1_sentry_pii_scrubbing_backend` debe matchear
    este archivo. Auto-satisfecho por convención (nombre del file)."""
    src = Path(__file__).read_text(encoding="utf-8")
    assert "P1-SENTRY-PII-SCRUBBING-BACKEND" in src
