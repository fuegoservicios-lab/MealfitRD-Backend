"""[P2-RATELIMIT-COVERAGE · 2026-05-12] Blanket parser-based: cada
endpoint mutador (POST/PUT/DELETE/PATCH) en `backend/app.py` y
`backend/routers/*.py` debe tener AL MENOS UNA capa de protección:

  (a) `Depends(get_verified_user_id)` — auth básica.
  (b) `Depends(verify_api_quota)` — auth + paywall mensual.
  (c) `Depends(_verify_admin_token)` — admin token (CRON_SECRET).
  (d) `Depends(<RateLimiter>)` — rate-limit explícito.
  (e) Marker `# [RATELIMIT-EXEMPT: <razón>]` en signature/comment previo
      cuando ninguna otra capa aplica (caso raro, requiere justificación).

Por qué este blanket existe:
    Audit production-readiness 2026-05-12 detectó que `RateLimiter` solo
    estaba aplicado a `/api/discount/validate` (P1-BILLING-3). Los 3
    webhooks/auth-paths que NO van por el paywall y procesan input
    arbitrario externo carecían de throttle:
      - `/api/webhooks/paypal` (verifica firma pero ANTES gasta 2
        round-trips httpx a api-m.paypal.com).
      - `/api/webhooks/process-pending-facts` (HMAC compare_digest pero
        flood pre-check consume CPU).
      - `/api/auth/migrate` (UUID enumeration vector).
    Fixes en mismo cierre añadieron `RateLimiter` a los 3.

    Este blanket previene REGRESIÓN: si alguien añade un nuevo endpoint
    sin auth ni rate-limit, falla loud con copy explicativo.

Trade-offs conscientes:
    - El blanket NO requiere rate-limit para endpoints que YA tienen
      `Depends(verify_api_quota)` — el paywall mensual los limita
      efectivamente (15-200 req/mes según tier).
    - Endpoints autenticados sin paywall (visualización, ops) son
      considerados "protegidos" por el JWT verify (atacante no puede
      forjar un token válido). Defensa-en-profundidad opcional, no
      requerida por este blanket.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND / "app.py"
_ROUTERS_DIR = _BACKEND / "routers"

# Patrones que satisfacen "endpoint protegido" (al menos uno debe matchear).
_PROTECTION_PATTERNS = [
    r"Depends\s*\(\s*get_verified_user_id\s*\)",
    r"Depends\s*\(\s*verify_api_quota\s*\)",
    r"Depends\s*\(\s*_verify_admin_token\s*\)",
    # RateLimiter directo o limiters nombrados con sufijo _LIMITER.
    r"Depends\s*\(\s*RateLimiter\s*\(",
    r"Depends\s*\(\s*_[A-Z_]+_LIMITER\s*\)",
    # [stale-parser fix · 2026-06-16] Patrón canónico de la familia /admin/*
    # (8 en plans.py + 3 en system.py + notifications.py): el gate de auth y
    # el rate-limit NO van como `Depends(...)` sino como llamadas in-body
    # `_verify_admin_token(request.headers.get("authorization"))` +
    # `_check_admin_rate_limit(request)` (P2-ADMIN-RATE-LIMIT · 2026-05-15;
    # el helper vive en routers.plans para no duplicar el CRON_SECRET check
    # en 5 lugares — por eso NO es una dependencia FastAPI). Ambos son
    # protección real: `_verify_admin_token` rechaza 401/403/503 con
    # `hmac.compare_digest` contra CRON_SECRET; `_check_admin_rate_limit`
    # levanta 429. El parser original solo veía el `Depends(_verify_admin_token)`
    # y marcaba estos endpoints (ya gateados) como falsos-positivos.
    r"_verify_admin_token\s*\(",
    r"_check_admin_rate_limit\s*\(",
]

_PROTECTION_RE = re.compile("|".join(_PROTECTION_PATTERNS))
_EXEMPT_MARKER_RE = re.compile(r"#\s*\[RATELIMIT-EXEMPT:\s*[^\]]+\]", re.IGNORECASE)
# [stale-parser fix · 2026-06-16] Exención documentada equivalente al marker
# `# [RATELIMIT-EXEMPT: ...]`. El endpoint `POST /api/plans/cancel` es
# intencionalmente sin auth ni rate-limit por decisión de producto aceptada
# (P2-CANCEL-NO-AUTH-ACCEPTED / P2-PROD-AUDIT-FOLLOWUP · 2026-05-28): session_id
# es un UUID no-enumerable client-side, el peor caso es DoS de un único plan
# en-vuelo (la víctima re-dispara), sin lectura/mutación de datos, y lo invocan
# guests sin JWT. La justificación completa vive en el docstring del handler
# (análoga al patrón "Decisiones de producto" / "Advisors aceptados" de
# CLAUDE.md). El anchor in-prod cumple el mismo rol que el marker.
_ACCEPTED_NO_AUTH_ANCHOR_RE = re.compile(r"P2-CANCEL-NO-AUTH-ACCEPTED")

# Decoradores que indican endpoint mutador HTTP (POST/PUT/DELETE/PATCH).
# GET no se cubre (read-only, JWT verify es suficiente; whitelist explícita
# en CLAUDE.md sección "Historial-quota-exemption" + auditoría P1-AUDIT-3).
_MUTATING_DECORATOR_RE = re.compile(
    r"@(?:[a-zA-Z_][a-zA-Z0-9_]*)\.(?:post|put|delete|patch)\s*\(",
)


def _iter_backend_python_files():
    yield _APP_PY
    for fp in _ROUTERS_DIR.glob("*.py"):
        if fp.name == "__init__.py":
            continue
        yield fp


def _extract_handler_block(text: str, decorator_idx: int) -> str:
    """Desde el índice del decorator, capturar hasta el siguiente decorator
    HTTP o EOF. Patrón simple: avanzar línea-por-línea hasta encontrar
    otro `@<x>.{post|get|...}` al inicio (col 0) o EOF."""
    lines = text[decorator_idx:].split("\n")
    block_lines = [lines[0]]
    # Inicia en línea 1 (después del decorator). Termina en el siguiente
    # decorator HTTP a column 0 o cuando vemos otro `@app.X` / `@router.X`.
    NEXT_DECORATOR = re.compile(
        r"^@(?:[a-zA-Z_][a-zA-Z0-9_]*)\.(?:get|post|put|delete|patch|head|options)\s*\("
    )
    for ln in lines[1:]:
        if NEXT_DECORATOR.match(ln):
            break
        block_lines.append(ln)
        # Cap defensivo: si pasamos de 200 líneas sin encontrar boundary,
        # el handler es atípico — devolver lo capturado.
        if len(block_lines) > 200:
            break
    return "\n".join(block_lines)


def _check_block_protected(block: str) -> tuple[bool, str | None]:
    """Retorna (is_protected, reason). is_protected=True si AL MENOS UN
    pattern de protección matchea. reason describe qué se encontró
    (debug-friendly)."""
    if _EXEMPT_MARKER_RE.search(block):
        m = _EXEMPT_MARKER_RE.search(block)
        return True, f"exempt-marker: {m.group(0).strip()}"
    if _ACCEPTED_NO_AUTH_ANCHOR_RE.search(block):
        m = _ACCEPTED_NO_AUTH_ANCHOR_RE.search(block)
        return True, f"accepted-no-auth-decision: {m.group(0).strip()}"
    m = _PROTECTION_RE.search(block)
    if m:
        return True, f"protection: {m.group(0).strip()}"
    return False, None


# ---------------------------------------------------------------------------
# 1. Anchor presente
# ---------------------------------------------------------------------------
def test_anchor_present_in_app_py():
    """Anchor `P2-RATELIMIT-COVERAGE` debe estar en app.py donde se
    definen los limiters compartidos."""
    text = _APP_PY.read_text(encoding="utf-8")
    assert "P2-RATELIMIT-COVERAGE" in text, (
        "Anchor `P2-RATELIMIT-COVERAGE` removido de app.py — el comment "
        "explicativo de los limiters defensivos es load-bearing."
    )


def test_anchor_present_in_billing_paypal_webhook():
    """Anchor secundario en billing.py para el limiter del paypal webhook."""
    text = (_ROUTERS_DIR / "billing.py").read_text(encoding="utf-8")
    assert "P2-RATELIMIT-COVERAGE-PAYPAL-WEBHOOK" in text, (
        "Anchor del limiter PayPal webhook removido de billing.py — el "
        "rationale del 30/min/IP debe permanecer trazable."
    )


# ---------------------------------------------------------------------------
# 2. Los 3 fixes específicos del cierre P2-RATELIMIT-COVERAGE
# ---------------------------------------------------------------------------
def test_paypal_webhook_has_rate_limiter():
    text = (_ROUTERS_DIR / "billing.py").read_text(encoding="utf-8")
    # Localizar el handler del paypal webhook.
    m = re.search(
        r'@webhooks_router\.post\("/paypal"\).*?def\s+\w+\s*\((.*?)\):',
        text,
        re.DOTALL,
    )
    assert m, "Handler `@webhooks_router.post('/paypal')` no encontrado."
    signature = m.group(1)
    assert "_PAYPAL_WEBHOOK_LIMITER" in signature, (
        "Handler `/api/webhooks/paypal` perdió su `Depends(_PAYPAL_WEBHOOK_LIMITER)`. "
        "Este endpoint dispara 2 round-trips httpx a api-m.paypal.com ANTES "
        "de validar la firma — sin throttle, atacante anónimo consume "
        "compute + tokens PayPal por cada flood."
    )


def test_webhook_process_pending_facts_has_rate_limiter():
    text = _APP_PY.read_text(encoding="utf-8")
    m = re.search(
        r'@app\.post\("/api/webhooks/process-pending-facts"\).*?def\s+\w+\s*\((.*?)\):',
        text,
        re.DOTALL,
    )
    assert m, "Handler `/api/webhooks/process-pending-facts` no encontrado."
    signature = m.group(1)
    assert "_WEBHOOK_FACTS_LIMITER" in signature, (
        "Handler perdió su `Depends(_WEBHOOK_FACTS_LIMITER)`. Sin throttle "
        "el HMAC `compare_digest` consume CPU por cada flood pre-check; "
        "P0-WEBHOOK-1 cierra el bypass de auth pero NO el flood."
    )


def test_auth_migrate_has_rate_limiter():
    text = _APP_PY.read_text(encoding="utf-8")
    m = re.search(
        r'@app\.post\("/api/auth/migrate"\).*?def\s+\w+\s*\((.*?)\):',
        text,
        re.DOTALL,
    )
    assert m, "Handler `/api/auth/migrate` no encontrado."
    signature = m.group(1)
    assert "_AUTH_MIGRATE_LIMITER" in signature, (
        "Handler `/api/auth/migrate` perdió su `Depends(_AUTH_MIGRATE_LIMITER)`. "
        "Aunque tiene `get_verified_user_id`, el migrate guest→user es "
        "vector de UUID enumeration brute-force con race-conditions."
    )


# ---------------------------------------------------------------------------
# 3. Knobs de los limiters: defaults conservadores + nombres canónicos
# ---------------------------------------------------------------------------
def test_limiter_definitions_have_documented_caps():
    """Los 3 limiters definidos en este cierre deben tener defaults
    explícitos (NO depender de env vars en su definición)."""
    text = _APP_PY.read_text(encoding="utf-8")
    assert "_WEBHOOK_FACTS_LIMITER = RateLimiter(max_calls=10, period_seconds=60)" in text, (
        "_WEBHOOK_FACTS_LIMITER cap cambiado — actualizar documentación + "
        "test si fue intencional. Default 10/min/IP es conservador y "
        "documentado en el comment del cierre."
    )
    assert "_AUTH_MIGRATE_LIMITER = RateLimiter(max_calls=5, period_seconds=300)" in text, (
        "_AUTH_MIGRATE_LIMITER cap cambiado — default 5/5min/IP cubre "
        "re-tries legítimos sin permitir brute-force con UUIDs enumerados."
    )
    text_billing = (_ROUTERS_DIR / "billing.py").read_text(encoding="utf-8")
    assert "_PAYPAL_WEBHOOK_LIMITER = RateLimiter(max_calls=30, period_seconds=60)" in text_billing, (
        "_PAYPAL_WEBHOOK_LIMITER cap cambiado — default 30/min/IP es "
        "generoso para PayPal real (~2-3 eventos/min en peak)."
    )


# ---------------------------------------------------------------------------
# 4. Blanket: TODOS los endpoints mutadores deben tener al menos una capa
# ---------------------------------------------------------------------------
def test_all_mutating_endpoints_protected():
    """Itera por `app.py` + `routers/*.py`. Para cada `@<x>.post|put|delete|
    patch(...)`, captura el handler block hasta el siguiente decorator HTTP
    y verifica que contenga AL MENOS UN pattern de protección.

    Si tu endpoint es legítimamente exempt (raro), añade comment
    `# [RATELIMIT-EXEMPT: <razón concreta>]` en el signature o las 5 líneas
    previas al `def`.
    """
    violations = []
    for fp in _iter_backend_python_files():
        text = fp.read_text(encoding="utf-8")
        for m in _MUTATING_DECORATOR_RE.finditer(text):
            decorator_idx = m.start()
            block = _extract_handler_block(text, decorator_idx)
            is_protected, reason = _check_block_protected(block)
            if not is_protected:
                # Capturar el path del endpoint para el mensaje (best-effort).
                path_m = re.search(
                    r'@[a-zA-Z_][a-zA-Z0-9_]*\.(?:post|put|delete|patch)\s*\(\s*[\'"]([^\'"]+)[\'"]',
                    block,
                )
                path_str = path_m.group(1) if path_m else "<unknown>"
                # Línea aproximada del decorator.
                line_no = text[:decorator_idx].count("\n") + 1
                rel = fp.relative_to(_BACKEND).as_posix()
                violations.append(
                    f"{rel}:{line_no} -> {path_str}"
                )

    assert not violations, (
        "P2-RATELIMIT-COVERAGE violation: endpoints mutadores SIN "
        "protección detectados. Cada POST/PUT/DELETE/PATCH debe tener "
        "AL MENOS UNA de:\n"
        "  - Depends(get_verified_user_id)         (auth básica)\n"
        "  - Depends(verify_api_quota)             (auth + paywall)\n"
        "  - Depends(_verify_admin_token)          (admin token)\n"
        "  - Depends(<RateLimiter o _*_LIMITER>)   (throttle)\n"
        "  - Comment `# [RATELIMIT-EXEMPT: <razón concreta>]` en\n"
        "    signature o 5 líneas previas (caso raro, justificar).\n\n"
        "Endpoints sin protección:\n  " + "\n  ".join(violations)
    )


# ---------------------------------------------------------------------------
# 5. Sanity del parser: matchea protección correctamente
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("snippet,expected_protected", [
    ("def x(verified_user_id: str = Depends(get_verified_user_id)):", True),
    ("def x(uid: str = Depends(verify_api_quota)):", True),
    ("def x(_t: str = Depends(_verify_admin_token)):", True),
    ("def x(_rl = Depends(RateLimiter(max_calls=5, period_seconds=60))):", True),
    ("def x(_rl = Depends(_PAYPAL_WEBHOOK_LIMITER)):", True),
    ("def x(_rl = Depends(_DISCOUNT_VALIDATE_LIMITER)):", True),
    ("# [RATELIMIT-EXEMPT: heartbeat consumido solo por monitor interno]\ndef x():", True),
    ("def x(data: dict = Body(...)):", False),  # sin protección
    ("def x():", False),  # vacío
])
def test_parser_sanity(snippet, expected_protected):
    is_protected, _ = _check_block_protected(snippet)
    assert is_protected == expected_protected, (
        f"Parser falló sobre snippet: {snippet!r}. "
        f"Esperaba {expected_protected}, obtuvo {is_protected}."
    )
