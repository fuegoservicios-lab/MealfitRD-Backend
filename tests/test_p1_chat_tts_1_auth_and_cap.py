"""[P1-CHAT-TTS-1 · 2026-05-11] `/api/chat/tts` requiere auth + cap + log_api_usage.

Bug original (audit 2026-05-11):
    El endpoint declaraba `data: dict = Body(...)` SIN `Depends(...)`.
    Cualquiera con la URL podía POSTear texto arbitrario; el backend
    proxieaba a ElevenLabs con la API key server-side, pagando con la
    API key del owner. Sin cap de longitud, sin log_api_usage, sin
    rate limiting per-user.

Cierre:
    1. `Depends(_CHAT_TTS_LIMITER)` (RateLimiter 60/60s) — RateLimiter
       internamente requiere `get_verified_user_id` y rechazamos 401
       si la cadena no resolvió user.
    2. Cap `len(text) <= _CHAT_TTS_MAX_TEXT_CHARS` (1500). 413 si excede.
    3. `log_api_usage(verified_user_id, "elevenlabs_tts")` post-success
       (best-effort, no aborta response).

Tests parser-based (estructurales) — NO levantamos httpx contra
ElevenLabs ni Supabase real. Las pruebas funcionales del paywall y
del rate limit ya viven en sus respectivas suites.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHAT_FP = _REPO_ROOT / "backend" / "routers" / "chat.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _CHAT_FP.read_text(encoding="utf-8")


def _extract_endpoint_block(src: str) -> str:
    """Extrae desde `@router.post("/tts")` hasta el siguiente `@router.` o
    `@app.` decorator (boundary del endpoint)."""
    start_marker = '@router.post("/tts")'
    start = src.find(start_marker)
    assert start > 0, "endpoint /tts no encontrado en chat.py"
    rest = src[start + len(start_marker):]
    next_decorator = re.search(r"\n(@router\.|@app\.)", rest)
    end_offset = next_decorator.start() if next_decorator else len(rest)
    return src[start: start + len(start_marker) + end_offset]


def test_rate_limiter_singleton_present(src: str):
    """`_CHAT_TTS_LIMITER = RateLimiter(...)` declarado a nivel módulo."""
    assert "_CHAT_TTS_LIMITER = RateLimiter(" in src, (
        "P1-CHAT-TTS-1 regresión: el singleton `_CHAT_TTS_LIMITER` "
        "desapareció. Sin él, el endpoint pierde su rate limit y un user "
        "autenticado puede spamear ElevenLabs (60/min cap se evapora). "
        "Restaurar el patrón módulo-level (mismo que `_PLAN_GEN_LIMITER` "
        "en routers/plans.py)."
    )
    # Verificar params canónicos: 60 calls/60 seg para voice mode chunks.
    m = re.search(
        r"_CHAT_TTS_LIMITER = RateLimiter\(\s*max_calls\s*=\s*(\d+)\s*,\s*period_seconds\s*=\s*(\d+)\s*\)",
        src,
    )
    assert m, (
        "P1-CHAT-TTS-1 regresión: la firma de `_CHAT_TTS_LIMITER` cambió. "
        "Esperaba `RateLimiter(max_calls=N, period_seconds=M)` con kwargs "
        "explícitos para que el grep de futuros audits funcione."
    )
    max_calls = int(m.group(1))
    period_s = int(m.group(2))
    assert max_calls <= 120, (
        f"P1-CHAT-TTS-1 regresión: `max_calls={max_calls}` excede 120 — el "
        f"rate limit se relajó a punto de no proteger contra spam. Si voice "
        f"mode necesita >120 calls/min, considerar streaming chunked TTS en "
        f"lugar de elevar el cap."
    )
    assert period_s == 60, (
        f"P1-CHAT-TTS-1: `period_seconds={period_s}` no es 60. El test asume "
        f"ventana minutaria; ajustar test si la convención cambia."
    )


def test_max_text_chars_constant_present(src: str):
    """Cap declarado como constante módulo-level para grep + ajuste rápido."""
    assert "_CHAT_TTS_MAX_TEXT_CHARS" in src, (
        "P1-CHAT-TTS-1 regresión: la constante `_CHAT_TTS_MAX_TEXT_CHARS` "
        "desapareció. Sin ella, el cap queda hardcoded inline y un revisor "
        "futuro no encuentra dónde ajustarlo si voice mode necesita más."
    )
    m = re.search(r"_CHAT_TTS_MAX_TEXT_CHARS\s*=\s*(\d+)", src)
    assert m, "no parsea el valor numérico de _CHAT_TTS_MAX_TEXT_CHARS"
    cap = int(m.group(1))
    assert 100 <= cap <= 5000, (
        f"P1-CHAT-TTS-1 regresión: cap={cap} fuera del rango razonable "
        f"[100, 5000]. <100 rompe respuestas normales del agente; >5000 "
        f"abre vector de cost-burn (ElevenLabs factura por carácter)."
    )


def test_endpoint_uses_rate_limiter_dep(src: str):
    """El handler `api_chat_tts` DEBE usar `Depends(_CHAT_TTS_LIMITER)`."""
    block = _extract_endpoint_block(src)
    assert "Depends(_CHAT_TTS_LIMITER)" in block, (
        "P1-CHAT-TTS-1 regresión: `api_chat_tts` ya no inyecta "
        "`Depends(_CHAT_TTS_LIMITER)`. Sin el limiter, un user "
        "autenticado puede spamear el TTS hasta agotar el cup ElevenLabs "
        "del owner. Restaurar `verified_user_id: Optional[str] = "
        "Depends(_CHAT_TTS_LIMITER)` en la signature."
    )


def test_endpoint_rejects_unauth_explicitly(src: str):
    """RateLimiter resuelve bucket por IP cuando no hay auth pero NO rechaza.
    El endpoint DEBE rechazar 401 explícitamente si `verified_user_id` es
    falsy (cost-bearing endpoint requiere auth obligatoria)."""
    block = _extract_endpoint_block(src)
    assert "if not verified_user_id:" in block, (
        "P1-CHAT-TTS-1 regresión: el guard `if not verified_user_id:` "
        "desapareció. RateLimiter cae a bucket IP cuando no hay auth — "
        "sin el guard explícito, un anon con IP nueva puede burnear hasta "
        "60 calls antes de que el rate limit se active. TTS es cost-bearing "
        "→ requiere auth obligatoria."
    )
    # 401 status code presente
    assert "401" in block, (
        "P1-CHAT-TTS-1: el código 401 no aparece en el block del endpoint. "
        "El rechazo de unauth debe ser status_code=401."
    )


def test_endpoint_caps_text_length(src: str):
    """Validación `len(text) > _CHAT_TTS_MAX_TEXT_CHARS` con 413."""
    block = _extract_endpoint_block(src)
    assert "len(text) > _CHAT_TTS_MAX_TEXT_CHARS" in block, (
        "P1-CHAT-TTS-1 regresión: el cap `len(text) > _CHAT_TTS_MAX_TEXT_CHARS` "
        "desapareció. Sin él, ElevenLabs cobra por carácter sin límite "
        "server-side."
    )
    assert "413" in block, (
        "P1-CHAT-TTS-1: el código HTTP 413 (Payload Too Large) no aparece "
        "en el endpoint. Es el código semánticamente correcto para text "
        "que excede el cap."
    )


def test_endpoint_logs_api_usage(src: str):
    """`log_api_usage(verified_user_id, "elevenlabs_tts")` debe estar."""
    block = _extract_endpoint_block(src)
    assert 'log_api_usage(verified_user_id, "elevenlabs_tts")' in block, (
        "P1-CHAT-TTS-1 regresión: `log_api_usage(verified_user_id, "
        "\"elevenlabs_tts\")` desapareció. Sin él perdemos accounting "
        "per-user de costo TTS — SRE no puede detectar bursts anómalos. "
        "Restaurar la llamada post-success bajo try/except (best-effort)."
    )
    # endpoint canónico para grep cross-codebase
    assert '"elevenlabs_tts"' in block, (
        "P1-CHAT-TTS-1: el endpoint name `\"elevenlabs_tts\"` cambió. Si "
        "renombras, actualizar tests E2E + dashboards SRE que filtran por "
        "este string."
    )


def test_log_api_usage_inside_try_except(src: str):
    """El log_api_usage debe estar wrapped en try/except para que un fallo
    de DB no rompa la response TTS al usuario (best-effort accounting).

    [P2-PROD-AUDIT-3 · 2026-05-30] La llamada ahora va offloaded por
    `await asyncio.to_thread(log_api_usage, verified_user_id, "elevenlabs_tts")`
    (no bloquear el event loop en el handler async); el wrapping try/except del
    finally SE PRESERVA. El test acepta ambas formas."""
    block = _extract_endpoint_block(src)
    # `rfind` para saltarnos la mención en el docstring y agarrar la llamada
    # real (al final del bloque). Acepta forma directa o offloaded (to_thread).
    log_idx = block.rfind('log_api_usage, verified_user_id, "elevenlabs_tts"')
    if log_idx < 0:
        log_idx = block.rfind('log_api_usage(verified_user_id, "elevenlabs_tts")')
    assert log_idx > 0, "log_api_usage no encontrado en el block (¿cambió el call site?)"
    # Buscar `try:` antes de la llamada y `except` después
    pre_block = block[:log_idx]
    post_block = block[log_idx:]
    last_try = pre_block.rfind("try:")
    next_except = post_block.find("except")
    assert last_try > 0 and next_except > 0, (
        "P1-CHAT-TTS-1 regresión: `log_api_usage` ya NO está wrapped en "
        "try/except. Si el DB pool falla (e.g., connection_pool drained), "
        "la response TTS se rompe — el usuario pierde voz aunque "
        "ElevenLabs ya respondió OK. Restaurar el wrapping best-effort."
    )


def test_text_must_be_str_and_nonempty(src: str):
    """`isinstance(text, str)` + `text.strip()` validación temprana."""
    block = _extract_endpoint_block(src)
    assert "isinstance(text, str)" in block, (
        "P1-CHAT-TTS-1 regresión: la validación `isinstance(text, str)` "
        "desapareció. Un dict/list maliciosos pasarían como text y "
        "podrían crashear httpx con TypeError."
    )
    assert "text.strip()" in block, (
        "P1-CHAT-TTS-1: validación de texto no-whitespace desaparecida. "
        "Si text es solo \"   \", el cap pasa pero ElevenLabs cobra por un "
        "request inútil."
    )
