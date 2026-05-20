"""[P1-AGENT-CACHE-MESSAGES · 2026-05-20] Anti-regresión del cache local
de messages para arranque instantáneo del AgentPage.

Bug observado (continuación del P1-AGENT-PERSIST-SESSION):
    Tras el fix #9 (preservar currentSessionId), el componente arrancaba
    con `messages = [welcome]` y refetcheaba el history en background.
    Durante los ~200-500ms del fetch, el user veía un "flash" del welcome
    screen → transición al chat real. Reportado 2026-05-20: "sigue igual,
    no es al instante".

Fix:
    Persistir los `messages` por session en localStorage. Al mount del
    componente, leer el cache para `currentSessionId`; si match y fresh
    (<24h), usar como initial state → arranque instantáneo sin flash.
    Refresh en background corre normal — si cambió server-side, setMessages
    los reemplaza imperceptiblemente (mismos contenidos en >99% de casos).
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENT_PAGE_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "AgentPage.jsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_messages_initializer_reads_cache():
    """[P1-AGENT-CACHE-MESSAGES] El initializer del useState `messages` DEBE
    intentar leer el cache de localStorage ANTES del fallback al welcome
    screen. Anti-pattern: `useState([welcome])` literal sin try-cache."""
    src = _read(_AGENT_PAGE_JSX)
    # Buscar el bloque del useState de messages: `useState(() => { ... })`.
    match = re.search(
        r"const\s*\[\s*messages\s*,\s*setMessages\s*\]\s*=\s*useState\(\s*\(\s*\)\s*=>\s*\{(.+?)\}\s*\);",
        src,
        re.DOTALL,
    )
    assert match, (
        "useState initializer de `messages` no encontrado como arrow function. "
        "Anti-pattern: `useState([welcome])` literal sin lazy initializer "
        "impide leer cache. Ver P1-AGENT-CACHE-MESSAGES · 2026-05-20."
    )
    body = match.group(1)
    # Anchor: el initializer debe usar safeLocalStorageGet con la cache key.
    assert "safeLocalStorageGet" in body, (
        "Initializer NO usa `safeLocalStorageGet` — no puede leer cache."
    )
    assert "_CHAT_CACHE_KEY" in body or "mealfit_chat_messages_cache" in body, (
        "Initializer NO referencia la cache key — refactor inesperado."
    )
    # Anchor: validación TTL (no usar cache stale).
    assert re.search(r"cachedAt|TTL_MS", body), (
        "Initializer NO valida TTL del cache — riesgo de servir mensajes "
        "muy antiguos (semanas) al usuario. Ver P1-AGENT-CACHE-MESSAGES."
    )
    # Anchor: validación sessionId match (no usar cache de OTRA sesión).
    assert "sessionId" in body, (
        "Initializer NO valida que el cache corresponda al currentSessionId. "
        "Sin esto, podría servir mensajes de otra session por error."
    )


def test_messages_persisted_on_change():
    """[P1-AGENT-CACHE-MESSAGES] Debe existir un useEffect que persista
    `messages` al localStorage en cada change para que el próximo mount
    encuentre el cache fresco."""
    src = _read(_AGENT_PAGE_JSX)
    # Buscar useEffect con dep array [messages, currentSessionId] o similar
    # que escriba a safeLocalStorageSet con la cache key.
    persist_match = re.search(
        r"useEffect\(\s*\(\s*\)\s*=>\s*\{(.+?)\}\s*,\s*\[\s*messages\s*,\s*currentSessionId\s*\]\s*\)",
        src,
        re.DOTALL,
    )
    assert persist_match, (
        "useEffect con `[messages, currentSessionId]` deps no encontrado. "
        "Sin esto, los messages no se persisten al cambio y el cache queda "
        "stale. Ver P1-AGENT-CACHE-MESSAGES · 2026-05-20."
    )
    body = persist_match.group(1)
    assert "safeLocalStorageSet" in body, (
        "useEffect no llama `safeLocalStorageSet` — no persiste."
    )
    assert "_CHAT_CACHE_KEY" in body or "mealfit_chat_messages_cache" in body, (
        "useEffect no usa la cache key correcta."
    )


def test_welcome_message_not_persisted():
    """[P1-AGENT-CACHE-MESSAGES] El cache NO debe persistir cuando los
    messages son solo el welcome screen (isWelcome=true). Razón: el
    welcome se regenera con datos frescos del profile en cada mount;
    cachearlo bloquea la regeneración tras cambios del profile."""
    src = _read(_AGENT_PAGE_JSX)
    persist_match = re.search(
        r"useEffect\(\s*\(\s*\)\s*=>\s*\{(.+?)\}\s*,\s*\[\s*messages\s*,\s*currentSessionId\s*\]\s*\)",
        src,
        re.DOTALL,
    )
    assert persist_match
    body = persist_match.group(1)
    # Anchor: early-return guard `if (messages.length === 1 && messages[0]?.isWelcome) return;`
    has_skip_welcome = bool(re.search(
        r"messages\.length\s*===?\s*1\s*&&\s*messages\[\s*0\s*\]\??\.isWelcome",
        body,
    ))
    assert has_skip_welcome, (
        "useEffect persiste el welcome screen como cache. Cachearlo bloquea "
        "la regeneración con datos frescos del profile en próximos mounts. "
        "Anchor esperado: `messages.length === 1 && messages[0]?.isWelcome`. "
        "Ver P1-AGENT-CACHE-MESSAGES · 2026-05-20."
    )
