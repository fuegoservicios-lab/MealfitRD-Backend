"""[P1-CHAT-SUPABASE-RETRY · 2026-05-20] Test anti-regresión del retry
tenacity sobre `get_recent_messages` para cubrir el modo de fallo
transient `httpx.RemoteProtocolError: Server disconnected`.

Bug observado:
    Kong/PostgREST de Supabase cierra conexiones HTTP/2 idle keep-alive
    agresivamente (~30-60s). La primera request post-idle de
    `supabase-py` (que NO maneja reconnect) falla con `httpx.RemoteProtocolError:
    Server disconnected`. La segunda request abre socket nuevo y funciona.

    Stack del crash original:
        agent.py:1544 → build_memory_context → get_recent_messages →
        supabase.table("agent_messages").select(...).execute() → RemoteProtocolError

    Síntoma visible: banner "El asistente tuvo un problema procesando tu
    mensaje. Puedes reintentar." sin razón aparente para el user. Después
    de reintentar manualmente, funciona.

Fix:
    - Decorator `@retry(stop_after_attempt(2), wait_exponential)` sobre
      `get_recent_messages` cubre el caso de socket muerto. Tenacity hace
      1 reintento con backoff 0.3-1.5s.
    - Helper `_is_transient_supabase_http_error` (no usado en esta versión
      del retry pero queda como anchor para extensiones futuras a otros
      callsites — el nombre indica intención).

Tests parser-based — anchor literal en código para que un refactor
accidental que remueva el retry falle el CI.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DB_CHAT_PY = _BACKEND_ROOT / "db_chat.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_get_recent_messages_has_retry_decorator():
    """[P1-CHAT-SUPABASE-RETRY] El decorator `@retry(...)` de tenacity
    debe envolver `get_recent_messages` con stop_after_attempt(2) para
    cubrir el caso `Server disconnected` transient."""
    src = _read(_DB_CHAT_PY)
    # Buscar el decorator @retry en las líneas inmediatamente previas
    # al `def get_recent_messages`.
    fn_match = re.search(
        r"(@retry\([^)]*(?:\([^)]*\)[^)]*)*\)\s*\n)+def get_recent_messages\(",
        src,
        re.DOTALL,
    )
    assert fn_match, (
        "Decorator `@retry(...)` ausente antes de `def get_recent_messages`. "
        "Sin retry, el primer `httpx.RemoteProtocolError: Server disconnected` "
        "del socket idle de Supabase falla el chat completo. Ver "
        "P1-CHAT-SUPABASE-RETRY · 2026-05-20."
    )
    decorator_block = fn_match.group(0)
    # Sanity: stop debe ser 2 intentos (1 retry tras el primer fail).
    assert re.search(r"stop_after_attempt\(\s*2\s*\)", decorator_block), (
        "`stop_after_attempt(2)` ausente del decorator — el retry no limita "
        "intentos o usa otro límite. Ver P1-CHAT-SUPABASE-RETRY."
    )
    # Sanity: backoff exponencial presente.
    assert "wait_exponential" in decorator_block, (
        "`wait_exponential(...)` ausente del decorator — sin backoff el "
        "retry vuela al instante y no da tiempo al socket de re-establecer."
    )
    # Sanity: reraise=True para propagar tras agotar intentos.
    assert "reraise=True" in decorator_block, (
        "`reraise=True` ausente — sin esto, tenacity envuelve la exception "
        "en `RetryError` y los callers (build_memory_context) no la manejan."
    )


def test_is_transient_supabase_http_error_helper_exists():
    """[P1-CHAT-SUPABASE-RETRY] Helper `_is_transient_supabase_http_error`
    debe existir como anchor para extensiones futuras del retry a otros
    callsites (si recurre el bug en otros endpoints de supabase-py)."""
    src = _read(_DB_CHAT_PY)
    assert "def _is_transient_supabase_http_error(" in src, (
        "Helper `_is_transient_supabase_http_error` ausente. Sin él, "
        "extender el retry a otros callsites (e.g. `get_session_messages`, "
        "`get_user_chat_sessions`) requiere duplicar la detection logic. "
        "Ver P1-CHAT-SUPABASE-RETRY · 2026-05-20."
    )
    # Sanity: debe detectar RemoteProtocolError + Server disconnected.
    helper_match = re.search(
        r"def _is_transient_supabase_http_error\(.*?\n(.*?)(?=\ndef |\Z)",
        src,
        re.DOTALL,
    )
    assert helper_match
    body = helper_match.group(1)
    assert "RemoteProtocolError" in body, (
        "Detector NO matchea `RemoteProtocolError` — el modo de fallo "
        "principal queda descubierto."
    )
    assert "server disconnected" in body.lower(), (
        "Detector NO matchea mensaje 'server disconnected' — fallback de "
        "string match ausente."
    )
