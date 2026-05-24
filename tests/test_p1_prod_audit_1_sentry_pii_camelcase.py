"""[P1-PROD-AUDIT-1 · 2026-05-23] Sentry PII scrubbing debe atrapar
keys camelCase, no solo snake_case.

Gap original (audit production-readiness 2026-05-23, B-P1-3):
    `_is_sensitive_key(key)` en `backend/app.py` solo matcheaba substring
    contra `_SENSITIVE_KEY_SUBSTRINGS` post-`.lower()`. Eso atrapa
    `health_profile`, `Health_Profile`, `HEALTH_PROFILE` (snake_case en
    cualquier capitalización) PERO NO:
      - `healthProfile` (camelCase)
      - `creditCard` (camelCase)
      - `health-profile` (kebab-case)

    Modo de fallo: si el frontend migra a TypeScript o usa
    `JSON.stringify({healthProfile})` y esa key llega a Sentry vía request
    body, queda SIN redactar → GDPR/HIPAA leak silencioso.

Fix:
    Normalizar ambos lados (key del payload + substring del list) removiendo
    separadores `_` y `-` antes del match. Backward compatible: el raw match
    sigue funcionando.

Cobertura:
    A) `_normalize_key` helper exportado en app.py.
    B) `_is_sensitive_key('healthProfile')` retorna True (camelCase).
    C) `_is_sensitive_key('creditCard')` retorna True (camelCase).
    D) `_is_sensitive_key('health-profile')` retorna True (kebab-case).
    E) `_is_sensitive_key('regular_field')` retorna False (no false positive).
    F) Anchor `P1-PROD-AUDIT-1-SENTRY-CAMEL` presente en source.

Tooltip-anchor: P1-PROD-AUDIT-1-SENTRY-CAMEL | audit 2026-05-23.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND_ROOT / "app.py"


def _load_helpers():
    """Carga `_is_sensitive_key` + `_normalize_key` sin disparar el module
    init de app.py (que abre DB pool, Sentry, schedulers). Estrategia
    idéntica a `test_p1_sentry_pii_scrubbing_backend.py`: extraer el bloque
    de helpers via regex + exec en namespace aislado.
    """
    src = _APP_PY.read_text(encoding="utf-8")
    start_match = re.search(
        r"^_SENSITIVE_KEY_SUBSTRINGS\s*=", src, re.MULTILINE
    )
    assert start_match is not None, "No se encontró `_SENSITIVE_KEY_SUBSTRINGS = (` en app.py"
    end_match = re.search(r"\nsentry_sdk\.init\(", src[start_match.start():])
    assert end_match is not None, "No se encontró `sentry_sdk.init(` tras helpers"
    code = src[start_match.start(): start_match.start() + end_match.start()]
    ns: dict = {}
    exec(compile(code, "<helpers>", "exec"), ns)
    return ns


def test_anchor_present_in_app():
    src = _APP_PY.read_text(encoding="utf-8")
    assert "P1-PROD-AUDIT-1-SENTRY-CAMEL" in src, (
        "Anchor `P1-PROD-AUDIT-1-SENTRY-CAMEL` ausente en app.py. Sin "
        "anchor, un futuro mantenedor que toque _is_sensitive_key pierde "
        "el contexto del fix camelCase (audit 2026-05-23)."
    )


def test_normalize_key_helper_exists():
    ns = _load_helpers()
    assert "_normalize_key" in ns, (
        "Helper `_normalize_key` ausente — la normalización camelCase NO "
        "puede funcionar sin él. Restaurar la función."
    )
    norm = ns["_normalize_key"]
    assert norm("health_profile") == "healthprofile"
    assert norm("health-profile") == "healthprofile"
    assert norm("healthprofile") == "healthprofile"


def test_is_sensitive_key_catches_camelcase():
    ns = _load_helpers()
    is_sensitive = ns["_is_sensitive_key"]

    # camelCase variants — esto es el core del fix B-P1-3.
    camelcase_sensitive = [
        ("healthProfile", "health_profile substring"),
        ("creditCard", "credit_card substring"),
        ("cardNumber", "card_number substring"),
        ("refreshToken", "refresh_token substring"),
        ("apiKey", "api_key substring"),
        ("accessKey", "access_key substring"),
        ("planData", "plan_data substring"),
    ]
    for key, reason in camelcase_sensitive:
        assert is_sensitive(key), (
            f"_is_sensitive_key('{key}') retornó False — debería ser True "
            f"(matchea {reason} en _SENSITIVE_KEY_SUBSTRINGS). El bug "
            f"camelCase NO está cerrado."
        )


def test_is_sensitive_key_catches_kebabcase():
    ns = _load_helpers()
    is_sensitive = ns["_is_sensitive_key"]

    kebab_sensitive = [
        "health-profile",
        "credit-card",
        "refresh-token",
        "api-key",
    ]
    for key in kebab_sensitive:
        assert is_sensitive(key), (
            f"_is_sensitive_key('{key}') retornó False — kebab-case con "
            f"substring sensible NO redactado. Headers HTTP usan kebab "
            f"case (`X-Auth-Token`, etc.)."
        )


def test_is_sensitive_key_preserves_snake_case_baseline():
    """Pre-fix detectaba snake_case. Backward compat: TODO el snake_case
    que ya detectaba debe seguir detectado."""
    ns = _load_helpers()
    is_sensitive = ns["_is_sensitive_key"]

    snake_sensitive = [
        "password",
        "secret",
        "token",
        "authorization",
        "Authorization",  # HTTP header capitalization
        "AUTHORIZATION",  # YELL capitalization
        "cookie",
        "email",
        "phone",
        "health_profile",
        "plan_data",
        "access_key",
        "api_key",
        "refresh_token",
        "credit_card",
        "card_number",
    ]
    for key in snake_sensitive:
        assert is_sensitive(key), (
            f"_is_sensitive_key('{key}') retornó False — REGRESIÓN. "
            f"Pre-fix detectaba esta key, post-fix camelCase NO debe "
            f"romper la detección snake_case."
        )


def test_is_sensitive_key_no_false_positives():
    """Keys no-sensitive NO deben matchear. Si normalizar el separador
    rompe esto, hay risk de redactar campos legítimos.
    """
    ns = _load_helpers()
    is_sensitive = ns["_is_sensitive_key"]

    not_sensitive = [
        "username",      # NO contiene substring sensitive
        "id",
        "name",
        "regular_field",
        "regularField",  # camelCase no-sensitive
        "weight",        # campo legitimo del form
        "height",
        "age",
        "status",
        "created_at",
        "createdAt",
    ]
    for key in not_sensitive:
        assert not is_sensitive(key), (
            f"_is_sensitive_key('{key}') retornó True — FALSO POSITIVO. "
            f"La normalización está atrapando keys que NO son sensitive. "
            f"Esto reduce señal en Sentry (campos legítimos redactados)."
        )


def test_normalized_substrings_tuple_present():
    """Pre-computed `_NORMALIZED_SUBSTRINGS` debe existir para performance —
    sin el cache, cada call a _is_sensitive_key recomputaría el normalize
    de cada substring (lista de 14 strings × N keys redactadas por request).
    """
    ns = _load_helpers()
    assert "_NORMALIZED_SUBSTRINGS" in ns, (
        "`_NORMALIZED_SUBSTRINGS` cache ausente. Sin él, _is_sensitive_key "
        "re-normaliza la lista entera por cada key del payload — overhead "
        "innecesario en hot path (Sentry filter corre por cada event)."
    )
    norm_tuple = ns["_NORMALIZED_SUBSTRINGS"]
    assert isinstance(norm_tuple, tuple), "_NORMALIZED_SUBSTRINGS debe ser tuple (immutable, hashable)"
    assert "healthprofile" in norm_tuple
    assert "creditcard" in norm_tuple
