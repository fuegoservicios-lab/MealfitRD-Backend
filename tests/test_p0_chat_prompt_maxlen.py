"""[P0-CHAT-PROMPT-MAXLEN · 2026-05-19] Cap de longitud para mensajes del
chat (campo `prompt` en `/api/chat/stream` y `/api/chat`, campo `content`
en `/api/chat/message`).

Bug original (audit production-readiness 2026-05-19 del Agente):
    Ninguno de los tres endpoints user-facing del chat validaba longitud
    del texto. Vectores:
      (a) DoS económico — un autenticado envía 100KB → Gemini consume
          tokens del owner desproporcionados; cuota mensual del provider
          se agota bajo abuso sostenido.
      (b) Context window saturation — el endpoint cuelga procesando el
          blob hasta el timeout total-graph (60s, P0-CHAT-LLM-TIMEOUT).
      (c) Storage abuse — `/api/chat/message` permite INSERT arbitrario
          en `agent_messages.content` (text, sin cap DB nativo).

Cierre:
    En `backend/routers/chat.py` se introdujo:
      - Knob `MEALFIT_CHAT_PROMPT_MAX_CHARS` (default 8192, clamp [256,
        65536]) auto-registrado vía `_env_int`.
      - Helper `_chat_prompt_max_chars()` con clamp defensivo.
      - Helper `_enforce_chat_prompt_cap(value, field_name)` que levanta
        `HTTPException(413)` cuando `len(value) > cap`.
      - Invocación de `_enforce_chat_prompt_cap(...)` en los 3 endpoints
        (después del IDOR check, ANTES de `save_message`/`chat_with_agent`).

Este test enforza (parser-based contra el source):
    1. El knob `MEALFIT_CHAT_PROMPT_MAX_CHARS` se lee vía `_env_int` →
       auto-registrado en `_KNOBS_REGISTRY` (visible en /health/version).
    2. Las constantes de clamp existen con valores razonables.
    3. El helper `_enforce_chat_prompt_cap` levanta `HTTPException(413)`.
    4. Los 3 endpoints invocan el helper:
       - `/message` con `field_name="content"`.
       - `/stream` con `field_name="prompt"`.
       - `/` (root POST) con `field_name="prompt"`.
    5. La invocación aparece ANTES del primer `save_message(...)` o
       `chat_with_agent(...)` callsite (defensa-en-profundidad: no
       guardar/invocar nada si el cap se excede).
    6. Tooltip-anchor `P0-CHAT-PROMPT-MAXLEN` presente — si alguien
       refactorea sin preservar el marker, falla.

Si alguien añade un cuarto endpoint user-facing que acepta texto libre
del cliente y lo enruta al LLM o a `save_message`, debe añadir su gate
+ una assertion aquí (o el slug del marker se rompe y `test_p2_hist_audit_14`
falla por cross-link missing).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHAT_ROUTER_FP = _REPO_ROOT / "backend" / "routers" / "chat.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _CHAT_ROUTER_FP.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Knob + helpers definidos
# ---------------------------------------------------------------------------

def test_knob_env_var_read_via_env_int(src: str) -> None:
    """[P0-CHAT-PROMPT-MAXLEN] el knob se lee vía `_env_int` (auto-registro
    en `_KNOBS_REGISTRY`). Si alguien lo cambia a `os.environ.get(...)`
    directo, el knob desaparece de `/health/version`."""
    assert "from knobs import _env_int" in src, (
        "[P0-CHAT-PROMPT-MAXLEN] falta `from knobs import _env_int` — "
        "el knob debe pasar por el registry para ser visible en "
        "/health/version. NO uses os.environ.get(...) directo."
    )
    pattern = re.compile(
        r'_env_int\(\s*["\']MEALFIT_CHAT_PROMPT_MAX_CHARS["\']\s*,\s*_CHAT_PROMPT_MAX_CHARS_DEFAULT'
    )
    assert pattern.search(src), (
        "[P0-CHAT-PROMPT-MAXLEN] el callsite del knob debe ser exactamente "
        "`_env_int('MEALFIT_CHAT_PROMPT_MAX_CHARS', _CHAT_PROMPT_MAX_CHARS_DEFAULT)`. "
        "Renombrar la constante rompe la simetría test↔producción."
    )


def test_knob_clamp_constants_defined(src: str) -> None:
    """[P0-CHAT-PROMPT-MAXLEN] las 3 constantes existen con valores
    canónicos. Default 8192 cubre 99.9% del chat conversacional; min 256
    deja saludo+pregunta corta; max 65536 bloquea caps absurdos."""
    assert re.search(r"_CHAT_PROMPT_MAX_CHARS_DEFAULT\s*=\s*8192", src), (
        "[P0-CHAT-PROMPT-MAXLEN] default debe ser 8192 (8KB ~ 99.9% del chat)."
    )
    assert re.search(r"_CHAT_PROMPT_MAX_CHARS_CLAMP_MIN\s*=\s*256", src), (
        "[P0-CHAT-PROMPT-MAXLEN] clamp inferior 256 evita que env var "
        "patológica rompa chat (necesita >= saludo + pregunta corta)."
    )
    assert re.search(r"_CHAT_PROMPT_MAX_CHARS_CLAMP_MAX\s*=\s*65536", src), (
        "[P0-CHAT-PROMPT-MAXLEN] clamp superior 65536 — >64KB ya no es "
        "chat sino archivo. Si necesitas más, sube esta constante (no el knob)."
    )


def test_helper_chat_prompt_max_chars_exists(src: str) -> None:
    """[P0-CHAT-PROMPT-MAXLEN] el helper que aplica el clamp existe y
    devuelve int — los endpoints lo invocan vía `_enforce_chat_prompt_cap`."""
    assert "def _chat_prompt_max_chars() -> int:" in src, (
        "[P0-CHAT-PROMPT-MAXLEN] el helper `_chat_prompt_max_chars() -> int` "
        "debe estar definido para que el clamp [256, 65536] sea aplicado."
    )


def test_helper_enforce_signature_and_413(src: str) -> None:
    """[P0-CHAT-PROMPT-MAXLEN] el helper de enforcement tiene la signature
    esperada, levanta HTTPException(413), y loguea WARNING."""
    assert 'def _enforce_chat_prompt_cap(value, field_name: str = "prompt") -> None:' in src, (
        "[P0-CHAT-PROMPT-MAXLEN] signature canónica del helper: "
        "`_enforce_chat_prompt_cap(value, field_name='prompt') -> None`. "
        "Cambiarla rompe los 3 callsites del router."
    )
    # 413 PAYLOAD_TOO_LARGE es el código semánticamente correcto.
    raise_pattern = re.compile(
        r"raise\s+HTTPException\(\s*\n?\s*status_code\s*=\s*413",
        re.MULTILINE,
    )
    assert raise_pattern.search(src), (
        "[P0-CHAT-PROMPT-MAXLEN] el helper debe levantar HTTPException(413) "
        "(PAYLOAD_TOO_LARGE). Cualquier otro código (400, 422) pierde "
        "semántica para el cliente y rompe instrumentación SRE."
    )
    # WARNING log para detectar abuso sostenido en producción.
    assert "[P0-CHAT-PROMPT-MAXLEN] rechazado" in src, (
        "[P0-CHAT-PROMPT-MAXLEN] el helper debe loguear el rechazo con el "
        "tag `[P0-CHAT-PROMPT-MAXLEN] rechazado` para grep en logs."
    )


# ---------------------------------------------------------------------------
# 2. Los 3 endpoints invocan el helper, ANTES de save/LLM
# ---------------------------------------------------------------------------

def _endpoint_body(src: str, route_prefix: str, fn_name: str) -> str:
    """Extrae el cuerpo de una función decorada con un `@router.post(<route>...)`.

    `route_prefix` es el prefijo del decorator hasta (sin incluir) el `)`
    de cierre o la coma siguiente — así toleramos tanto el form mínimo
    `@router.post("/stream")` como el form con kwargs adicionales
    `@router.post("/stream", dependencies=[Depends(...)])` que P1-CHAT-STREAM-RL
    introdujo el 2026-05-19. Tooltip-anchor: P0-CHAT-PROMPT-MAXLEN.
    """
    deco_idx = src.find(route_prefix)
    assert deco_idx >= 0, f"decorator prefix {route_prefix!r} no encontrado"
    fn_idx = src.find(f"def {fn_name}(", deco_idx)
    assert fn_idx >= 0, f"def {fn_name}() no encontrado tras {route_prefix!r}"
    # Recorta hasta el próximo `@router.` o EOF.
    next_router = src.find("@router.", fn_idx + 10)
    body_end = next_router if next_router > 0 else len(src)
    return src[fn_idx:body_end]


def test_message_endpoint_enforces_cap(src: str) -> None:
    """[P0-CHAT-PROMPT-MAXLEN] `/api/chat/message` valida `content`."""
    body = _endpoint_body(src, '@router.post("/message"', "api_save_chat_message")
    assert '_enforce_chat_prompt_cap(content, field_name="content")' in body, (
        "[P0-CHAT-PROMPT-MAXLEN] `/api/chat/message` debe invocar "
        "`_enforce_chat_prompt_cap(content, field_name='content')` para "
        "prevenir storage abuse (INSERT arbitrario en agent_messages.content)."
    )
    # Antes del save_message. Regex multi-línea tolera tanto el form
    # mínimo `save_message(session_id, role, content)` como el form
    # `save_message(\n    session_id, role, content,\n    user_id=...)`
    # introducido por DB-P1-CHAT-USER-ID-RLS.
    enforce_pos = body.find("_enforce_chat_prompt_cap(content")
    save_match = re.search(
        r"save_message\s*\(\s*\n?\s*session_id,\s*role,\s*content",
        body,
    )
    save_pos = save_match.start() if save_match else -1
    assert 0 <= enforce_pos < save_pos, (
        "[P0-CHAT-PROMPT-MAXLEN] el cap debe aplicarse ANTES de "
        "`save_message(session_id, role, content, ...)`. Defensa-en-profundidad: "
        "no persistir nada si el cap se excede."
    )


def test_stream_endpoint_enforces_cap(src: str) -> None:
    """[P0-CHAT-PROMPT-MAXLEN] `/api/chat/stream` valida `prompt` antes
    de save + invocación del LLM."""
    body = _endpoint_body(src, '@router.post("/stream"', "api_chat_stream")
    assert '_enforce_chat_prompt_cap(prompt, field_name="prompt")' in body, (
        "[P0-CHAT-PROMPT-MAXLEN] `/api/chat/stream` debe invocar "
        "`_enforce_chat_prompt_cap(prompt, field_name='prompt')` antes de "
        "`save_message` y antes del LLM (DoS económico)."
    )
    # Prefijo sin paren cerrado para tolerar `user_id=...` añadido post-DB-P1.
    enforce_pos = body.find("_enforce_chat_prompt_cap(prompt")
    save_pos = body.find('save_message(session_id, "user", prompt')
    assert 0 <= enforce_pos < save_pos, (
        "[P0-CHAT-PROMPT-MAXLEN] el cap debe aplicarse ANTES de "
        "`save_message(session_id, 'user', prompt, ...)`. Si se aplica "
        "después, un prompt gigante se persiste igual aunque rechacemos el LLM."
    )


def test_root_chat_endpoint_enforces_cap(src: str) -> None:
    """[P0-CHAT-PROMPT-MAXLEN] `/api/chat` (root POST, non-stream) también
    valida `prompt`. Vector idéntico al de `/stream`."""
    body = _endpoint_body(src, '@router.post(""', "api_chat")
    assert '_enforce_chat_prompt_cap(prompt, field_name="prompt")' in body, (
        "[P0-CHAT-PROMPT-MAXLEN] `/api/chat` (root POST) debe invocar "
        "`_enforce_chat_prompt_cap(prompt, field_name='prompt')` antes de "
        "`save_message`/`chat_with_agent` (mismo vector que /stream)."
    )
    enforce_pos = body.find("_enforce_chat_prompt_cap(prompt")
    save_pos = body.find('save_message(session_id, "user", prompt')
    assert 0 <= enforce_pos < save_pos, (
        "[P0-CHAT-PROMPT-MAXLEN] el cap debe aplicarse ANTES de "
        "`save_message(session_id, 'user', prompt, ...)` también en /api/chat root."
    )


# ---------------------------------------------------------------------------
# 3. Tests funcionales del helper
# ---------------------------------------------------------------------------

@pytest.fixture
def helpers():
    """Importa los helpers reales del router para validarlos en runtime.

    Si fastapi/supabase no están instalados (venv mínimo de dev local sin
    deps completas), saltamos elegantemente — los tests parser-based de
    arriba son la red de seguridad CI obligatoria; los funcionales sólo
    confirman runtime cuando el entorno está completo."""
    pytest.importorskip("fastapi")
    pytest.importorskip("supabase")
    import sys
    backend_dir = str(_REPO_ROOT / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from routers import chat as chat_mod  # type: ignore
    return chat_mod


def test_clamp_lower_bound(helpers, monkeypatch) -> None:
    """[P0-CHAT-PROMPT-MAXLEN] env var debajo del floor → clamp a 256."""
    monkeypatch.setenv("MEALFIT_CHAT_PROMPT_MAX_CHARS", "10")
    assert helpers._chat_prompt_max_chars() == 256


def test_clamp_upper_bound(helpers, monkeypatch) -> None:
    """[P0-CHAT-PROMPT-MAXLEN] env var sobre el ceiling → clamp a 65536."""
    monkeypatch.setenv("MEALFIT_CHAT_PROMPT_MAX_CHARS", "10000000")
    assert helpers._chat_prompt_max_chars() == 65536


def test_default_when_env_unset(helpers, monkeypatch) -> None:
    """[P0-CHAT-PROMPT-MAXLEN] env var ausente → default 8192."""
    monkeypatch.delenv("MEALFIT_CHAT_PROMPT_MAX_CHARS", raising=False)
    assert helpers._chat_prompt_max_chars() == 8192


def test_enforce_passes_under_cap(helpers, monkeypatch) -> None:
    """[P0-CHAT-PROMPT-MAXLEN] string <= cap pasa sin levantar."""
    monkeypatch.setenv("MEALFIT_CHAT_PROMPT_MAX_CHARS", "100")
    helpers._enforce_chat_prompt_cap("hola mundo")  # 10 chars, OK


def test_enforce_raises_413_over_cap(helpers, monkeypatch) -> None:
    """[P0-CHAT-PROMPT-MAXLEN] string > cap levanta HTTP 413 con
    `detail` que contiene tamaño + cap (debug-friendly para el cliente)."""
    from fastapi import HTTPException
    monkeypatch.setenv("MEALFIT_CHAT_PROMPT_MAX_CHARS", "10")
    # El cap clampa a 256 (no 10), así que el string debe superar 256 para
    # que el helper levante 413. Usamos 300 chars (> 256 clamped cap).
    with pytest.raises(HTTPException) as exc_info:
        helpers._enforce_chat_prompt_cap("x" * 300, field_name="prompt")
    assert exc_info.value.status_code == 413
    assert "300" in exc_info.value.detail  # tamaño rechazado
    assert "256" in exc_info.value.detail  # cap aplicado (clamp a 256, no 10)


def test_enforce_noop_on_none_or_empty(helpers) -> None:
    """[P0-CHAT-PROMPT-MAXLEN] None / "" no levantan — la validación
    de "missing" pertenece al caller (cada endpoint la maneja distinto)."""
    helpers._enforce_chat_prompt_cap(None)
    helpers._enforce_chat_prompt_cap("")
    helpers._enforce_chat_prompt_cap(123)  # non-string: no-op defensivo


# ---------------------------------------------------------------------------
# 4. Tooltip-anchor preservado
# ---------------------------------------------------------------------------

def test_tooltip_anchor_present(src: str) -> None:
    """[P0-CHAT-PROMPT-MAXLEN] el marker textual debe aparecer en el código
    para que (a) un grep produzca el contexto, (b) el cross-link de
    `test_p2_hist_audit_14_marker_test_link` matchee el slug del test."""
    count = src.count("P0-CHAT-PROMPT-MAXLEN")
    assert count >= 5, (
        f"[P0-CHAT-PROMPT-MAXLEN] esperaba >= 5 menciones del marker en "
        f"chat.py (1 import block + 1 helper docstring + 1 enforce docstring + "
        f"3 callsites). Encontradas: {count}. Si refactoreaste, preserva el "
        f"tooltip-anchor o el test falla aquí."
    )
