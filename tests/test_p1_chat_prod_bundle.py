"""[P1-CHAT-PROD-BUNDLE · 2026-05-19] Bundle de 4 P1 production-readiness
del Agente cerrados en sesión 2026-05-19 tras audit:

  1. **P1-CHAT-STREAM-RL** — rate limiter per-minute en `/api/chat/stream`
     y `/api/chat` root. Pre-fix: solo el paywall `verify_api_quota` filtraba
     (mensual gratis=15, basic=50, plus=200, ultra=999999). A escala mensual
     no protege contra bursts: user plus puede gastar 200 prompts en 30
     segundos saturando upstream Gemini + abriendo LLMCircuitBreaker.
     Knob `MEALFIT_CHAT_STREAM_LIMITER_PER_MIN` (default 30, clamp [1, 600]).

  2. **P1-CHAT-CANCEL-ASYNC + P1-CHAT-STREAM-FINALLY-CLOSE** — defensive
     cleanup en stream path. (a) `event_generator` en routers/chat.py
     extiende el `except GeneratorExit` a `except (GeneratorExit,
     asyncio.CancelledError)` — Starlette puede cancelar el wrapper
     async cuando cliente desconecta y `CancelledError` hereda de
     `BaseException` (no `Exception`) en Python 3.8+. (b) `finally:
     stream_iter.close()` defensivo en `chat_with_agent_stream` cubre
     normal exit + exception path, no solo GeneratorExit — evita slow
     burn leak de FDs del checkpointer Postgres bajo concurrencia alta.

  3. **P1-CHAT-LOG-CTX** — `LoggerAdapter` con `session_id` + `user_id_hash`
     (SHA-256[:12] mismo patrón P2-HEALTH-UID-STRIP). Pre-fix: cada log
     del chat-flow usaba el logger crudo del módulo sin contexto. Un
     incidente reportado por user requería correlación visual + suerte
     en Sentry/CloudWatch. Con adapter, cada record carga `extra={...}`
     que Sentry filtra/agrega nativamente.

  4. **P1-SEARCH-DEEP-MEMORY-CACHE** — TTL cache in-process para
     `search_deep_memory`. Multi-turn conversations donde el LLM invoca
     la tool repetidamente generan N queries casi idénticas. Sin cache:
     N × (~150-400ms latencia + embedding cost + pgvector scan).
     Knob `MEALFIT_SEARCH_DEEP_MEMORY_CACHE_TTL_S` (default 300s = 5min,
     clamp [0, 3600], 0 desactiva). Maxsize 1024 entries con eviction
     ~10% más viejas en overflow.

Cierra los 4 gaps P1 del audit production-readiness Agente 2026-05-19.
Pendientes restantes (NO en este bundle):
  - P2: token telemetry (`response.usage.total_tokens`)
  - P2: scroll-to-bottom race en AgentPage
  - P2: `save_message` sin tenacity.retry
  - P3: virtualización mensajes >200, `role="log" aria-live`, bundle CI
  - DB-P1/P2: agent_messages sin user_id directo + RLS policies ausentes
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHAT_ROUTER_FP = _REPO_ROOT / "backend" / "routers" / "chat.py"
_AGENT_FP = _REPO_ROOT / "backend" / "agent.py"
_TOOLS_FP = _REPO_ROOT / "backend" / "tools.py"


@pytest.fixture(scope="module")
def chat_src() -> str:
    return _CHAT_ROUTER_FP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def agent_src() -> str:
    return _AGENT_FP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tools_src() -> str:
    return _TOOLS_FP.read_text(encoding="utf-8")


# ===========================================================================
# P1.1 — Rate limiter en /chat/stream + /api/chat root
# ===========================================================================

def test_p1_1_limiter_singleton_defined(chat_src: str) -> None:
    """[P1-CHAT-STREAM-RL] el singleton `_CHAT_STREAM_LIMITER` existe y
    se construye con un knob, no hardcoded."""
    assert "_CHAT_STREAM_LIMITER = RateLimiter(" in chat_src, (
        "[P1-CHAT-STREAM-RL] falta singleton `_CHAT_STREAM_LIMITER = "
        "RateLimiter(...)`. Si renombraste, actualiza este test + las "
        "tres invocaciones `Depends(_CHAT_STREAM_LIMITER)` en los "
        "decorators de /stream y /."
    )
    # Construido con la lectura del knob, NO con un literal.
    assert "max_calls=_chat_stream_limiter_per_min()" in chat_src, (
        "[P1-CHAT-STREAM-RL] `RateLimiter(max_calls=...)` debe leer el knob "
        "via `_chat_stream_limiter_per_min()`, no un literal. Esto preserva "
        "la capacidad de override sin redeploy."
    )


def test_p1_1_knob_clamp_constants(chat_src: str) -> None:
    """[P1-CHAT-STREAM-RL] constantes del knob con clamp defensivo
    [1, 600] (default 30)."""
    assert "_CHAT_STREAM_LIMITER_PER_MIN_DEFAULT = 30" in chat_src
    assert "_CHAT_STREAM_LIMITER_PER_MIN_CLAMP_MIN = 1" in chat_src
    assert "_CHAT_STREAM_LIMITER_PER_MIN_CLAMP_MAX = 600" in chat_src


def test_p1_1_knob_env_var_name(chat_src: str) -> None:
    """[P1-CHAT-STREAM-RL] el env var es `MEALFIT_CHAT_STREAM_LIMITER_PER_MIN`
    leído via `_env_int` (auto-registro en _KNOBS_REGISTRY → visible en
    /health/version)."""
    pattern = re.compile(
        r'_env_int\(\s*\n?\s*["\']MEALFIT_CHAT_STREAM_LIMITER_PER_MIN["\']'
    )
    assert pattern.search(chat_src), (
        "[P1-CHAT-STREAM-RL] el knob debe leerse con `_env_int('MEALFIT_"
        "CHAT_STREAM_LIMITER_PER_MIN', ...)`. NO uses os.environ.get(...) "
        "directo — bypassea el registry y rompe /health/version."
    )


def test_p1_1_stream_endpoint_has_dependency(chat_src: str) -> None:
    """[P1-CHAT-STREAM-RL] el decorator `/stream` invoca el limiter como
    `dependencies=[Depends(_CHAT_STREAM_LIMITER)]`."""
    pattern = re.compile(
        r'@router\.post\(\s*"/stream"\s*,\s*dependencies=\[\s*Depends\(_CHAT_STREAM_LIMITER\)\s*\]\s*\)'
    )
    assert pattern.search(chat_src), (
        "[P1-CHAT-STREAM-RL] `@router.post('/stream', dependencies=[Depends("
        "_CHAT_STREAM_LIMITER)])` debe estar presente. Si lo removiste, el "
        "endpoint queda solo bajo el paywall mensual — no protege contra burst."
    )


def test_p1_1_root_chat_endpoint_has_dependency(chat_src: str) -> None:
    """[P1-CHAT-STREAM-RL] el decorator `/` (root, non-stream) también
    invoca el limiter — mismo vector que /stream (LLM-bearing)."""
    pattern = re.compile(
        r'@router\.post\(\s*""\s*,\s*dependencies=\[\s*Depends\(_CHAT_STREAM_LIMITER\)\s*\]\s*\)'
    )
    assert pattern.search(chat_src), (
        "[P1-CHAT-STREAM-RL] `@router.post('', dependencies=[Depends("
        "_CHAT_STREAM_LIMITER)])` debe estar presente — mismo limiter que "
        "/stream porque ambos endpoints invocan el LLM."
    )


# ===========================================================================
# P1.2 — CancelledError handler + finally cleanup
# ===========================================================================

def test_p1_2_event_generator_handles_cancelled_error(chat_src: str) -> None:
    """[P1-CHAT-CANCEL-ASYNC] `event_generator` debe atrapar AMBOS
    `GeneratorExit` Y `asyncio.CancelledError`. CancelledError hereda
    de BaseException → `except Exception` no la atrapa."""
    pattern = re.compile(
        r"except\s+\(\s*GeneratorExit\s*,\s*asyncio\.CancelledError\s*\)"
    )
    assert pattern.search(chat_src), (
        "[P1-CHAT-CANCEL-ASYNC] falta `except (GeneratorExit, "
        "asyncio.CancelledError)` en event_generator. Sin esto, un "
        "cancel async se loguea como exception (stack confuso) en vez "
        "de aborto legítimo."
    )


def test_p1_2_stream_iter_finally_close(agent_src: str) -> None:
    """[P1-CHAT-STREAM-FINALLY-CLOSE] `chat_with_agent_stream` debe
    cerrar `stream_iter` en `finally` — cleanup defensivo en TODOS los
    exits (normal, exception, GeneratorExit), no solo cancel."""
    # Localiza la función.
    fn_idx = agent_src.find("def chat_with_agent_stream(")
    assert fn_idx >= 0, "función chat_with_agent_stream no encontrada"
    # Recorta hasta la siguiente def top-level (líneas que empiezan con `def `).
    next_def = re.search(r"\n(?:def\s)", agent_src[fn_idx + 10:])
    end = (fn_idx + 10 + next_def.start()) if next_def else len(agent_src)
    body = agent_src[fn_idx:end]
    # Busca el patrón `finally:` seguido (dentro de ~20 líneas) de
    # `stream_iter.close()`.
    finally_close = re.compile(
        r"finally:\s*\n(?:[^\n]*\n){0,25}?\s*stream_iter\.close\(\)",
        re.MULTILINE,
    )
    assert finally_close.search(body), (
        "[P1-CHAT-STREAM-FINALLY-CLOSE] falta `finally: ... stream_iter.close()` "
        "en chat_with_agent_stream. Sin esto, exits normales (sin "
        "GeneratorExit) dependen de GC para liberar threads/FDs del "
        "iterator de LangGraph — slow burn leak bajo concurrencia."
    )


# ===========================================================================
# P1.3 — LoggerAdapter (session_id + user_id_hash)
# ===========================================================================

def test_p1_3_hash_user_id_helper_exists(chat_src: str) -> None:
    """[P1-CHAT-LOG-CTX] helper de hash existe + usa SHA-256[:12] (mismo
    patrón P2-HEALTH-UID-STRIP de routers/system.py)."""
    assert "def _hash_user_id_for_log(" in chat_src
    assert "hashlib.sha256(" in chat_src
    # Slicing [:12] explicito o vía hexdigest[:12]
    assert ".hexdigest()[:12]" in chat_src, (
        "[P1-CHAT-LOG-CTX] hash debe ser `sha256(...).hexdigest()[:12]` — "
        "mismo length que `_hash_uuid_for_public` (P2-HEALTH-UID-STRIP)."
    )


def test_p1_3_chat_logger_helper_returns_adapter(chat_src: str) -> None:
    """[P1-CHAT-LOG-CTX] `_chat_logger(session_id, user_id)` retorna
    `logging.LoggerAdapter` con `extra={'session_id', 'user_id_hash'}`."""
    assert "def _chat_logger(" in chat_src
    pattern = re.compile(
        r"logging\.LoggerAdapter\(\s*logger\s*,\s*extra\s*=\s*\{",
        re.MULTILINE,
    )
    assert pattern.search(chat_src), (
        "[P1-CHAT-LOG-CTX] `_chat_logger` debe retornar `logging."
        "LoggerAdapter(logger, extra={...})` para que Sentry/sinks "
        "estructurados puedan filtrar por session_id / user_id_hash."
    )
    # Las dos keys del extra deben estar en el código.
    assert '"session_id"' in chat_src
    assert '"user_id_hash"' in chat_src


def test_p1_3_stream_endpoint_uses_clog(chat_src: str) -> None:
    """[P1-CHAT-LOG-CTX] `/stream` instancia `clog = _chat_logger(...)`
    y lo usa en al menos un log line del flow."""
    # El endpoint /stream debe tener `clog = _chat_logger(...)`.
    stream_idx = chat_src.find('@router.post("/stream"')
    assert stream_idx >= 0
    next_router = chat_src.find("@router.", stream_idx + 10)
    body = chat_src[stream_idx:next_router if next_router > 0 else len(chat_src)]
    assert "clog = _chat_logger(" in body, (
        "[P1-CHAT-LOG-CTX] /stream debe instanciar el adapter con "
        "`clog = _chat_logger(session_id, user_id)` después del cap "
        "P0-CHAT-PROMPT-MAXLEN."
    )
    # Y debe usarlo al menos una vez (clog.info/clog.warning/clog.exception).
    assert re.search(r"\bclog\.(info|warning|exception|error)\(", body), (
        "[P1-CHAT-LOG-CTX] /stream debe usar `clog.<level>(...)` al menos "
        "una vez — instanciar el adapter sin usarlo no aporta correlación."
    )


# ===========================================================================
# P1.4 — TTL cache en search_deep_memory
# ===========================================================================

def test_p1_4_cache_singleton_defined(tools_src: str) -> None:
    """[P1-SEARCH-DEEP-MEMORY-CACHE] cache + constantes existen."""
    assert "_SEARCH_DEEP_MEMORY_CACHE: dict = {}" in tools_src
    assert "_SEARCH_DEEP_MEMORY_CACHE_MAX_ENTRIES = 1024" in tools_src
    assert "_SEARCH_DEEP_MEMORY_CACHE_TTL_S_DEFAULT = 300" in tools_src


def test_p1_4_helpers_defined(tools_src: str) -> None:
    """[P1-SEARCH-DEEP-MEMORY-CACHE] los 4 helpers existen
    (`_env_int_safe_tools`, `_search_deep_memory_cache_ttl_s`, `_get`, `_set`)."""
    assert "def _env_int_safe_tools(" in tools_src
    assert "def _search_deep_memory_cache_ttl_s()" in tools_src
    assert "def _search_deep_memory_cache_get(" in tools_src
    assert "def _search_deep_memory_cache_set(" in tools_src


def test_p1_4_knob_env_var_name(tools_src: str) -> None:
    """[P1-SEARCH-DEEP-MEMORY-CACHE] env var canónica."""
    assert "MEALFIT_SEARCH_DEEP_MEMORY_CACHE_TTL_S" in tools_src


def test_p1_4_search_uses_cache(tools_src: str) -> None:
    """[P1-SEARCH-DEEP-MEMORY-CACHE] `search_deep_memory` invoca el
    cache get ANTES del DB query, y store DESPUÉS — además cachea el
    resultado vacío para evitar re-scan."""
    fn_idx = tools_src.find("def search_deep_memory(")
    assert fn_idx >= 0
    # Cuerpo hasta el próximo `# ===` o `@tool` o EOF.
    end_marker = tools_src.find("@tool", fn_idx + 10)
    body = tools_src[fn_idx:end_marker if end_marker > 0 else len(tools_src)]
    # get + dos sets (resultado vacío + resultado con datos).
    assert "_search_deep_memory_cache_get(user_id" in body, (
        "[P1-SEARCH-DEEP-MEMORY-CACHE] `search_deep_memory` debe invocar "
        "`_search_deep_memory_cache_get` ANTES del db query."
    )
    # Conteo de invocaciones de set: 2 (empty + non-empty).
    set_count = body.count("_search_deep_memory_cache_set(user_id")
    assert set_count >= 2, (
        f"[P1-SEARCH-DEEP-MEMORY-CACHE] `_search_deep_memory_cache_set` "
        f"debe invocarse al menos 2 veces (cacheo de resultado vacío + "
        f"cacheo de resultado con datos). Encontradas: {set_count}."
    )
    # El get debe aparecer ANTES del primer set (orden de ejecución).
    get_pos = body.find("_search_deep_memory_cache_get(user_id")
    set_pos = body.find("_search_deep_memory_cache_set(user_id")
    assert 0 <= get_pos < set_pos, (
        "[P1-SEARCH-DEEP-MEMORY-CACHE] el cache GET debe aparecer ANTES "
        "del primer SET. Caso contrario el cache nunca se consulta antes "
        "de tirar al DB."
    )


def test_p1_4_db_query_skipped_on_cache_hit(tools_src: str) -> None:
    """[P1-SEARCH-DEEP-MEMORY-CACHE] el path `if _hit:` debe `return`
    antes de `db_search_deep_memory(...)` — caso contrario el cache no
    ahorra latencia."""
    fn_idx = tools_src.find("def search_deep_memory(")
    end_marker = tools_src.find("@tool", fn_idx + 10)
    body = tools_src[fn_idx:end_marker if end_marker > 0 else len(tools_src)]
    # `if _hit:` seguido (dentro de 5 líneas) por `return _cached`.
    hit_block = re.compile(r"if\s+_hit:\s*\n(?:[^\n]*\n){0,5}?\s*return\s+_cached")
    assert hit_block.search(body), (
        "[P1-SEARCH-DEEP-MEMORY-CACHE] el branch `if _hit: ... return "
        "_cached` debe estar presente y NO debe haber db_search_deep_memory "
        "ANTES de él."
    )
    # Posición relativa: `if _hit:` debe estar ANTES de `db_search_deep_memory(`.
    hit_pos = body.find("if _hit:")
    db_pos = body.find("db_search_deep_memory(")
    assert 0 <= hit_pos < db_pos, (
        "[P1-SEARCH-DEEP-MEMORY-CACHE] `if _hit:` debe aparecer ANTES de "
        "`db_search_deep_memory(...)` — si está después, el DB query "
        "siempre se ejecuta y el cache no ahorra latencia."
    )


# ===========================================================================
# Tooltip-anchors preservados
# ===========================================================================

def test_tooltip_anchors_present(chat_src: str, agent_src: str, tools_src: str) -> None:
    """Cada P1 tiene un marker textual que un refactor accidental
    rompería primero — sirve como early-warning para code review."""
    assert chat_src.count("P1-CHAT-STREAM-RL") >= 3, (
        "P1-CHAT-STREAM-RL debe aparecer al menos 3 veces en chat.py "
        "(comment intro + docstring helper + tooltip-anchor)."
    )
    assert chat_src.count("P1-CHAT-CANCEL-ASYNC") >= 1
    assert agent_src.count("P1-CHAT-STREAM-FINALLY-CLOSE") >= 1
    assert chat_src.count("P1-CHAT-LOG-CTX") >= 3
    assert tools_src.count("P1-SEARCH-DEEP-MEMORY-CACHE") >= 3


# ===========================================================================
# Tests funcionales del cache (skipped si tools.py no importable)
# ===========================================================================

@pytest.fixture
def tools_module():
    """Importa tools.py si las deps están disponibles."""
    pytest.importorskip("langchain_core")
    pytest.importorskip("langchain_google_genai")
    pytest.importorskip("tenacity")
    import sys
    backend_dir = str(_REPO_ROOT / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    import tools as tools_mod  # type: ignore
    # Reset cache entre tests para aislamiento.
    tools_mod._SEARCH_DEEP_MEMORY_CACHE.clear()
    return tools_mod


def test_cache_ttl_clamp_negative_becomes_zero(tools_module, monkeypatch) -> None:
    """Env negativo → clamp a 0 (cache desactivado)."""
    monkeypatch.setenv("MEALFIT_SEARCH_DEEP_MEMORY_CACHE_TTL_S", "-50")
    assert tools_module._search_deep_memory_cache_ttl_s() == 0


def test_cache_ttl_clamp_huge_capped_to_3600(tools_module, monkeypatch) -> None:
    """Env demasiado grande → clamp a 3600."""
    monkeypatch.setenv("MEALFIT_SEARCH_DEEP_MEMORY_CACHE_TTL_S", "999999")
    assert tools_module._search_deep_memory_cache_ttl_s() == 3600


def test_cache_default_when_env_unset(tools_module, monkeypatch) -> None:
    """Sin env → default 300."""
    monkeypatch.delenv("MEALFIT_SEARCH_DEEP_MEMORY_CACHE_TTL_S", raising=False)
    assert tools_module._search_deep_memory_cache_ttl_s() == 300


def test_cache_hit_returns_stored_value(tools_module, monkeypatch) -> None:
    """Set + get inmediato → hit con el mismo value."""
    monkeypatch.setenv("MEALFIT_SEARCH_DEEP_MEMORY_CACHE_TTL_S", "300")
    tools_module._search_deep_memory_cache_set("u1", "q1", "result-A")
    hit, val = tools_module._search_deep_memory_cache_get("u1", "q1")
    assert hit is True
    assert val == "result-A"


def test_cache_miss_on_different_user(tools_module, monkeypatch) -> None:
    """Cache key incluye user_id — un user distinto no debe ver
    resultados ajenos (IDOR defense)."""
    monkeypatch.setenv("MEALFIT_SEARCH_DEEP_MEMORY_CACHE_TTL_S", "300")
    tools_module._search_deep_memory_cache_set("u1", "q1", "data-u1")
    hit, val = tools_module._search_deep_memory_cache_get("u2", "q1")
    assert hit is False
    assert val is None


def test_cache_disabled_when_ttl_zero(tools_module, monkeypatch) -> None:
    """TTL=0 → set es no-op, get retorna miss siempre."""
    monkeypatch.setenv("MEALFIT_SEARCH_DEEP_MEMORY_CACHE_TTL_S", "0")
    tools_module._search_deep_memory_cache_set("u1", "q1", "should-not-store")
    hit, val = tools_module._search_deep_memory_cache_get("u1", "q1")
    assert hit is False
    assert val is None


def test_cache_eviction_on_overflow(tools_module, monkeypatch) -> None:
    """Cuando el cache excede MAX_ENTRIES, ~10% más viejas se descartan."""
    monkeypatch.setenv("MEALFIT_SEARCH_DEEP_MEMORY_CACHE_TTL_S", "300")
    max_n = tools_module._SEARCH_DEEP_MEMORY_CACHE_MAX_ENTRIES
    # Llenar hasta max + 1 para forzar eviction.
    for i in range(max_n):
        tools_module._search_deep_memory_cache_set(f"u{i}", "q", f"v{i}")
    assert len(tools_module._SEARCH_DEEP_MEMORY_CACHE) == max_n
    # El próximo set debe gatillar eviction (~10% = max_n // 10).
    tools_module._search_deep_memory_cache_set("new_user", "q", "new_val")
    assert len(tools_module._SEARCH_DEEP_MEMORY_CACHE) < max_n, (
        "Tras overflow + set adicional, el cache debe haber evictado "
        "entries — no creciendo sin límite."
    )


# ===========================================================================
# Tests funcionales del logger adapter (skipped si fastapi no importable)
# ===========================================================================

@pytest.fixture
def chat_router_module():
    """Importa routers/chat.py si las deps están disponibles."""
    pytest.importorskip("fastapi")
    pytest.importorskip("supabase")
    import sys
    backend_dir = str(_REPO_ROOT / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from routers import chat as chat_mod  # type: ignore
    return chat_mod


def test_hash_user_id_for_log_guest_passthrough(chat_router_module) -> None:
    """Guests/None NO se hashean — quedan como "guest"."""
    assert chat_router_module._hash_user_id_for_log(None) == "guest"
    assert chat_router_module._hash_user_id_for_log("guest") == "guest"
    assert chat_router_module._hash_user_id_for_log("") == "guest"


def test_hash_user_id_for_log_deterministic_length(chat_router_module) -> None:
    """User real → SHA-256[:12] determinístico, 12 chars hex."""
    uid = "abc-123-def-456"
    h1 = chat_router_module._hash_user_id_for_log(uid)
    h2 = chat_router_module._hash_user_id_for_log(uid)
    assert h1 == h2  # determinístico
    assert len(h1) == 12
    assert re.fullmatch(r"[0-9a-f]{12}", h1)


def test_hash_user_id_for_log_no_collision_short(chat_router_module) -> None:
    """Dos UUIDs distintos producen hashes distintos (sanity check, no
    exhaustive collision test)."""
    h1 = chat_router_module._hash_user_id_for_log("uuid-1")
    h2 = chat_router_module._hash_user_id_for_log("uuid-2")
    assert h1 != h2


def test_chat_logger_returns_adapter_with_extra(chat_router_module) -> None:
    """`_chat_logger` retorna un LoggerAdapter con extra correcto."""
    import logging
    adapter = chat_router_module._chat_logger("sess-1", "user-1")
    assert isinstance(adapter, logging.LoggerAdapter)
    assert adapter.extra.get("session_id") == "sess-1"
    assert "user_id_hash" in adapter.extra
    # NO debe leakear el user_id raw.
    assert adapter.extra["user_id_hash"] != "user-1"


def test_chat_stream_limiter_clamp_lower(chat_router_module, monkeypatch) -> None:
    """Knob debajo del floor → clamp a 1 (nunca 0)."""
    monkeypatch.setenv("MEALFIT_CHAT_STREAM_LIMITER_PER_MIN", "-5")
    assert chat_router_module._chat_stream_limiter_per_min() == 1


def test_chat_stream_limiter_clamp_upper(chat_router_module, monkeypatch) -> None:
    """Knob sobre ceiling → clamp a 600."""
    monkeypatch.setenv("MEALFIT_CHAT_STREAM_LIMITER_PER_MIN", "100000")
    assert chat_router_module._chat_stream_limiter_per_min() == 600


def test_chat_stream_limiter_default(chat_router_module, monkeypatch) -> None:
    """Sin env → default 30."""
    monkeypatch.delenv("MEALFIT_CHAT_STREAM_LIMITER_PER_MIN", raising=False)
    assert chat_router_module._chat_stream_limiter_per_min() == 30
