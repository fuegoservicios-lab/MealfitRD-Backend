"""[P2-CHAT-PROD-BUNDLE · 2026-05-19] Bundle de 3 P2 production-readiness
del Agente cerrados en sesión 2026-05-19 tras audit (post P0+P1):

  1. **P2-CHAT-TOKEN-TELEMETRY** — `call_model` ahora invoca
     `_emit_llm_usage_event_best_effort(llm=chat_llm, result=response,
     duration_s=...)` post-invoke exitoso. Pre-fix: el `chat_llm` se
     construye directo (`ChatGoogleGenerativeAI(...)`) sin pasar por el
     override `ainvoke/astream` de `graph_orchestrator.py` que dispara
     auto-instrumentación. Resultado: SRE veía costos de plan-gen
     (P1-COST-INSTRUMENTATION) pero 0 visibilidad de costos del agente
     conversacional → cron de alerting NO podía detectar anomalías.

  2. **P2-CHAT-SCROLL-RACE** — `scrollToBottom` respeta el intent del
     usuario. Pre-fix: `useEffect(() => scrollToBottom(), [messages])`
     saltaba al fondo en CADA chunk SSE del LLM streaming → user que
     scrolleaba arriba para releer historial mid-stream era arrojado al
     fondo cada ~100ms → imposible leer. Fix: `userScrolledUpRef` (NO state,
     evita re-render por scroll tick) + `handleMessagesScroll` con umbral
     120px desde bottom. `scrollToBottom(force=true)` ignora el ref
     (caso: usuario acaba de enviar mensaje → reset del ref + auto-scroll).

  3. **P2-CHAT-SAVE-MSG-RETRY** — `save_message` (db_chat.py) ahora envuelve
     el INSERT a `agent_messages` en `tenacity.retry(stop_after_attempt(3),
     wait_exponential(multiplier=0.5, min=0.5, max=4.0))`. Pre-fix: un
     blip transient de Supabase mid-stream → respuesta del agente
     renderizada en UI pero NO persistida en DB → mensaje fantasma → LLM
     pierde contexto del turn al refrescar. El side-effect
     `handle_nudge_response` queda FUERA del retry para evitar
     re-procesamiento múltiple del nudge.

Cierra los 3 gaps P2 del audit production-readiness Agente 2026-05-19.
Pendientes restantes (NO en este bundle):
  - P3: virtualización mensajes >200 (P1-CHAT-VIRTUALIZE ya cerró
    >100 vía VirtualizedMessageList), `role="log" aria-live`, bundle-analysis CI
  - DB-P1: `agent_messages` / `conversation_summaries` sin columna `user_id` directa
  - DB-P2: RLS enabled+forced sin policies SELECT/INSERT/UPDATE
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_AGENT_FP = _REPO_ROOT / "backend" / "agent.py"
_DB_CHAT_FP = _REPO_ROOT / "backend" / "db_chat.py"
_AGENT_PAGE_FP = _REPO_ROOT / "frontend" / "src" / "pages" / "AgentPage.jsx"


@pytest.fixture(scope="module")
def agent_src() -> str:
    return _AGENT_FP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def db_chat_src() -> str:
    return _DB_CHAT_FP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def agent_page_src() -> str:
    return _AGENT_PAGE_FP.read_text(encoding="utf-8")


# ===========================================================================
# P2.1 — Token telemetry en call_model
# ===========================================================================

def test_p2_1_call_model_imports_emit_helper(agent_src: str) -> None:
    """[P2-CHAT-TOKEN-TELEMETRY] `call_model` debe importar
    `_emit_llm_usage_event_best_effort` desde graph_orchestrator
    (lazy import dentro del cuerpo para evitar ciclo module-init)."""
    fn_idx = agent_src.find("def call_model(")
    assert fn_idx >= 0, "call_model no encontrado en agent.py"
    # Recorta hasta la próxima `def ` top-level.
    next_def = re.search(r"\n(?:def\s)", agent_src[fn_idx + 10:])
    end = (fn_idx + 10 + next_def.start()) if next_def else len(agent_src)
    body = agent_src[fn_idx:end]
    assert "from graph_orchestrator import _emit_llm_usage_event_best_effort" in body, (
        "[P2-CHAT-TOKEN-TELEMETRY] `call_model` debe hacer lazy import de "
        "`_emit_llm_usage_event_best_effort` desde graph_orchestrator. "
        "Top-level import generaría ciclo (graph_orchestrator → agent → "
        "graph_orchestrator)."
    )


def test_p2_1_call_model_measures_invoke_duration(agent_src: str) -> None:
    """[P2-CHAT-TOKEN-TELEMETRY] `_chat_invoke_start = time.time()` debe
    aparecer ANTES del `llm_with_tools.invoke(...)` para que la duración
    sea precisa (no incluye CB checks ni record_success)."""
    fn_idx = agent_src.find("def call_model(")
    next_def = re.search(r"\n(?:def\s)", agent_src[fn_idx + 10:])
    end = (fn_idx + 10 + next_def.start()) if next_def else len(agent_src)
    body = agent_src[fn_idx:end]
    start_pos = body.find("_chat_invoke_start =")
    invoke_pos = body.find("llm_with_tools.invoke(")
    assert 0 <= start_pos < invoke_pos, (
        "[P2-CHAT-TOKEN-TELEMETRY] `_chat_invoke_start = time.time()` debe "
        "estar definido ANTES del `llm_with_tools.invoke(...)`. Si lo defines "
        "después, mides 0s siempre."
    )


def test_p2_1_emit_after_record_success(agent_src: str) -> None:
    """[P2-CHAT-TOKEN-TELEMETRY] el emit debe ocurrir DESPUÉS de
    `_cb.record_success()` y DENTRO de un try/except. Failure path NO
    emite (timeout no consumió tokens completos)."""
    fn_idx = agent_src.find("def call_model(")
    next_def = re.search(r"\n(?:def\s)", agent_src[fn_idx + 10:])
    end = (fn_idx + 10 + next_def.start()) if next_def else len(agent_src)
    body = agent_src[fn_idx:end]
    success_pos = body.find("_cb.record_success()")
    emit_pos = body.find("_emit_llm_usage_event_best_effort(")
    assert 0 <= success_pos < emit_pos, (
        "[P2-CHAT-TOKEN-TELEMETRY] el emit del usage event debe ocurrir "
        "DESPUÉS de `_cb.record_success()`. Failure path SOLO marca "
        "`record_failure()` y re-raises — el emit no debe alcanzarse en "
        "ese path."
    )
    # Best-effort: el emit DEBE estar envuelto en try/except.
    assert re.search(
        r"try:\s*\n\s*from graph_orchestrator import _emit_llm_usage_event_best_effort\s*\n\s*_emit_llm_usage_event_best_effort\(",
        body,
    ), (
        "[P2-CHAT-TOKEN-TELEMETRY] el emit debe estar dentro de un "
        "try/except — un fallo de import/DB NO debe romper el response."
    )


def test_p2_1_emit_passes_correct_args(agent_src: str) -> None:
    """[P2-CHAT-TOKEN-TELEMETRY] el emit recibe `llm=chat_llm` (NO
    llm_with_tools — necesitamos el model_name del CGGA original) y
    `result=response`."""
    pattern = re.compile(
        r"_emit_llm_usage_event_best_effort\(\s*\n?\s*llm=chat_llm\s*,\s*\n?\s*result=response",
        re.MULTILINE,
    )
    assert pattern.search(agent_src), (
        "[P2-CHAT-TOKEN-TELEMETRY] firma esperada: "
        "`_emit_llm_usage_event_best_effort(llm=chat_llm, result=response, "
        "duration_s=...)`. `llm=chat_llm` (NO llm_with_tools) porque "
        "el helper busca `getattr(llm, 'model')` que solo existe en el "
        "CGGA antes del bind_tools."
    )


# ===========================================================================
# P2.2 — Scroll-to-bottom race en AgentPage
# ===========================================================================

def test_p2_2_refs_defined(agent_page_src: str) -> None:
    """[P2-CHAT-SCROLL-RACE] `messagesContainerRef` y `userScrolledUpRef`
    deben estar declarados como `useRef`."""
    assert "const messagesContainerRef = useRef(null);" in agent_page_src
    assert "const userScrolledUpRef = useRef(false);" in agent_page_src


def test_p2_2_scroll_to_bottom_force_flag(agent_page_src: str) -> None:
    """[P2-CHAT-SCROLL-RACE] `scrollToBottom` acepta `force=false` y
    hace no-op silencioso si `userScrolledUpRef.current && !force`."""
    pattern = re.compile(
        r"const scrollToBottom = \(force = false\) =>\s*\{[^}]*?if \(userScrolledUpRef\.current && !force\) return;",
        re.DOTALL,
    )
    assert pattern.search(agent_page_src), (
        "[P2-CHAT-SCROLL-RACE] `scrollToBottom` debe tener signature "
        "`(force = false) => { if (userScrolledUpRef.current && !force) "
        "return; ... }`. Si lo cambias, el guard se rompe."
    )


def test_p2_2_handle_messages_scroll_threshold(agent_page_src: str) -> None:
    """[P2-CHAT-SCROLL-RACE] `handleMessagesScroll` usa umbral 120px y
    actualiza el ref (NO state — evita re-render por scroll tick)."""
    assert "const handleMessagesScroll = useCallback(" in agent_page_src
    # El umbral 120 debe estar presente.
    assert "distanceFromBottom > 120" in agent_page_src, (
        "[P2-CHAT-SCROLL-RACE] el umbral del scroll guard debe ser 120 "
        "(px desde el bottom). Cubre overshoot por scroll momentum mobile."
    )
    # Actualiza el ref, NO un setState.
    assert "userScrolledUpRef.current = distanceFromBottom > 120;" in agent_page_src


def test_p2_2_container_has_ref_and_onscroll(agent_page_src: str) -> None:
    """[P2-CHAT-SCROLL-RACE] el `messages-container` tiene ambos
    `ref={messagesContainerRef}` y `onScroll={handleMessagesScroll}`."""
    assert "ref={messagesContainerRef}" in agent_page_src, (
        "[P2-CHAT-SCROLL-RACE] el `<div className=\"messages-container\">` "
        "debe tener `ref={messagesContainerRef}` montado."
    )
    assert "onScroll={handleMessagesScroll}" in agent_page_src, (
        "[P2-CHAT-SCROLL-RACE] el `<div className=\"messages-container\">` "
        "debe tener `onScroll={handleMessagesScroll}` montado. Sin esto "
        "el ref nunca se actualiza."
    )


def test_p2_2_send_resets_scroll_ref(agent_page_src: str) -> None:
    """[P2-CHAT-SCROLL-RACE] el send-handler resetea
    `userScrolledUpRef.current = false` para que el auto-scroll vuelva
    cuando el user envía un nuevo mensaje (acción afirmativa)."""
    assert "userScrolledUpRef.current = false;" in agent_page_src, (
        "[P2-CHAT-SCROLL-RACE] tras `setInput('')` el send-handler debe "
        "hacer `userScrolledUpRef.current = false;`. Sin esto, si el "
        "user scrolleó arriba y luego mandó un nuevo prompt, la respuesta "
        "no se auto-scrollearía."
    )


# ===========================================================================
# P2.3 — save_message con tenacity retry
# ===========================================================================

def test_p2_3_tenacity_imports(db_chat_src: str) -> None:
    """[P2-CHAT-SAVE-MSG-RETRY] imports de tenacity al top del módulo."""
    assert "from tenacity import (" in db_chat_src
    for symbol in ("retry", "stop_after_attempt", "wait_exponential",
                   "retry_if_exception_type", "before_sleep_log"):
        assert symbol in db_chat_src, (
            f"[P2-CHAT-SAVE-MSG-RETRY] tenacity symbol {symbol!r} "
            f"no importado en db_chat.py"
        )


def test_p2_3_retry_helper_defined(db_chat_src: str) -> None:
    """[P2-CHAT-SAVE-MSG-RETRY] helper `_save_message_insert_with_retry`
    existe y está decorado con `@retry(...)` con `stop_after_attempt(3)`,
    `wait_exponential`, y `reraise=True`."""
    assert "def _save_message_insert_with_retry(" in db_chat_src, (
        "[P2-CHAT-SAVE-MSG-RETRY] falta helper "
        "`_save_message_insert_with_retry(session_id, role, content)`."
    )
    # El decorator @retry(...) debe preceder inmediatamente a la def.
    # Aislamos el bloque del decorator (de `@retry(` hasta el `def
    # _save_message_insert_with_retry(`) y verificamos componentes
    # individuales — más robusto a reordenar de kwargs.
    deco_pattern = re.compile(
        r"@retry\((?P<args>.*?)\)\s*\ndef _save_message_insert_with_retry\(",
        re.DOTALL,
    )
    m = deco_pattern.search(db_chat_src)
    assert m, (
        "[P2-CHAT-SAVE-MSG-RETRY] el decorator `@retry(...)` debe preceder "
        "inmediatamente a `def _save_message_insert_with_retry(`."
    )
    args = m.group("args")
    assert "stop_after_attempt(3)" in args, (
        "[P2-CHAT-SAVE-MSG-RETRY] el decorator debe usar `stop_after_attempt(3)`. "
        "Cap fijo en 3 — más intentos quema cuota mensual sin valor."
    )
    assert "wait_exponential(" in args, (
        "[P2-CHAT-SAVE-MSG-RETRY] el decorator debe usar `wait_exponential(...)`. "
        "Backoff lineal o sin backoff agrava un incidente Supabase."
    )
    assert "reraise=True" in args, (
        "[P2-CHAT-SAVE-MSG-RETRY] `reraise=True` obligatorio — sin él "
        "tenacity oculta el final exception bajo `RetryError` y el caller "
        "no sabe qué pasó."
    )


def _find_save_message_def(db_chat_src: str) -> int:
    """[P2-CHAT-SAVE-MSG-RETRY] Localiza `def save_message(`. Tolera ambas
    signatures: la mínima `def save_message(session_id: str, ...)` Y la
    multi-línea introducida por DB-P1-CHAT-USER-ID-RLS (signature con
    `user_id: Optional[str] = None` en líneas separadas)."""
    # Match: "def save_message(" followed eventually by "session_id"
    # within the signature (across newlines).
    m = re.search(r"def save_message\s*\(\s*\n?\s*session_id", db_chat_src)
    return m.start() if m else -1


def test_p2_3_save_message_uses_retry_helper(db_chat_src: str) -> None:
    """[P2-CHAT-SAVE-MSG-RETRY] `save_message` invoca el helper retry-able
    en lugar de hacer `supabase.table(...).insert(...).execute()` inline.
    El INSERT a `agent_messages` debe vivir SOLO dentro del helper."""
    fn_idx = _find_save_message_def(db_chat_src)
    assert fn_idx >= 0, "función save_message(session_id, ...) no encontrada"
    # Cuerpo hasta la próxima `def `.
    next_def = re.search(r"\ndef\s", db_chat_src[fn_idx + 10:])
    end = (fn_idx + 10 + next_def.start()) if next_def else len(db_chat_src)
    body = db_chat_src[fn_idx:end]
    # Prefijo sin paren cerrado para tolerar el user_id arg post-DB-P1.
    assert "_save_message_insert_with_retry(session_id, role, content" in body, (
        "[P2-CHAT-SAVE-MSG-RETRY] `save_message` debe invocar el helper "
        "`_save_message_insert_with_retry(session_id, role, content, ...)` "
        "en lugar del INSERT inline."
    )
    # `save_message` no debe contener el INSERT directo — ni en la forma
    # PostgREST legacy ni en la forma SQL directa (P1-NEON-DB-MIGRATION).
    assert 'supabase.table("agent_messages").insert(' not in body, (
        "[P2-CHAT-SAVE-MSG-RETRY] `save_message` NO debe contener el "
        "INSERT inline a `agent_messages` — debe delegar al helper "
        "retry-able."
    )
    assert "execute_sql_write(" not in body, (
        "[P2-CHAT-SAVE-MSG-RETRY] `save_message` NO debe invocar "
        "`execute_sql_write(...)` inline — el INSERT a `agent_messages` "
        "debe vivir SOLO dentro del helper retry-able."
    )


def test_p2_3_nudge_outside_retry(db_chat_src: str) -> None:
    """[P2-CHAT-SAVE-MSG-RETRY] `handle_nudge_response` debe quedar FUERA
    del retry — si el INSERT falla 2 veces, el nudge NO debe procesarse
    2 veces más. El side-effect debe correr UNA vez antes del INSERT."""
    fn_idx = _find_save_message_def(db_chat_src)
    assert fn_idx >= 0, "función save_message(session_id, ...) no encontrada"
    next_def = re.search(r"\ndef\s", db_chat_src[fn_idx + 10:])
    end = (fn_idx + 10 + next_def.start()) if next_def else len(db_chat_src)
    body = db_chat_src[fn_idx:end]
    nudge_pos = body.find("handle_nudge_response(")
    retry_pos = body.find("_save_message_insert_with_retry(")
    assert 0 <= nudge_pos < retry_pos, (
        "[P2-CHAT-SAVE-MSG-RETRY] `handle_nudge_response(...)` debe "
        "ejecutarse ANTES del `_save_message_insert_with_retry(...)`. "
        "Si está dentro del helper retry-able, fallos transient lo "
        "ejecutarán múltiples veces."
    )
    # `handle_nudge_response` callsite real (NO el del docstring) NO debe
    # aparecer en el cuerpo del helper. Buscamos el patrón
    # `handle_nudge_response(<arg>)` que es la invocación; en el docstring
    # es solo `handle_nudge_response` sin paréntesis.
    helper_idx = db_chat_src.find("def _save_message_insert_with_retry(")
    helper_end = db_chat_src.find("\ndef ", helper_idx + 10)
    helper_body = db_chat_src[helper_idx:helper_end if helper_end > 0 else len(db_chat_src)]
    assert not re.search(r"handle_nudge_response\(\w", helper_body), (
        "[P2-CHAT-SAVE-MSG-RETRY] el callsite `handle_nudge_response(...)` "
        "NO debe estar dentro del cuerpo de `_save_message_insert_with_retry` "
        "— retries multiplicarían el side-effect."
    )


# ===========================================================================
# Tooltip-anchors preservados
# ===========================================================================

def test_tooltip_anchors_present(agent_src: str, db_chat_src: str, agent_page_src: str) -> None:
    """Cada P2 tiene un marker textual preservado."""
    assert agent_src.count("P2-CHAT-TOKEN-TELEMETRY") >= 2, (
        "P2-CHAT-TOKEN-TELEMETRY debe aparecer ≥2 veces en agent.py "
        "(comment intro + tooltip-anchor + emit-block)."
    )
    assert db_chat_src.count("P2-CHAT-SAVE-MSG-RETRY") >= 3, (
        "P2-CHAT-SAVE-MSG-RETRY debe aparecer ≥3 veces en db_chat.py "
        "(import + decorator block + 2 invocation sites)."
    )
    assert agent_page_src.count("P2-CHAT-SCROLL-RACE") >= 3, (
        "P2-CHAT-SCROLL-RACE debe aparecer ≥3 veces en AgentPage.jsx "
        "(ref block + scrollToBottom + handler + container + send-reset)."
    )


# ===========================================================================
# Tests funcionales (skipped si supabase / fastapi no disponibles)
# ===========================================================================

@pytest.fixture
def db_chat_module():
    """Importa db_chat.py si las deps están disponibles."""
    pytest.importorskip("supabase")
    pytest.importorskip("tenacity")
    import sys
    backend_dir = str(_REPO_ROOT / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    import db_chat as db_chat_mod  # type: ignore
    return db_chat_mod


def test_retry_decorator_attributes(db_chat_module) -> None:
    """[P2-CHAT-SAVE-MSG-RETRY] tenacity-decorated callable expone
    `.retry` con la config esperada."""
    fn = db_chat_module._save_message_insert_with_retry
    # tenacity decorators expose `.retry` (the Retrying object).
    assert hasattr(fn, "retry") or hasattr(fn, "__wrapped__"), (
        "el helper debe estar decorado con @retry (tenacity)."
    )


def test_retry_triggers_3_attempts(db_chat_module, monkeypatch) -> None:
    """[P2-CHAT-SAVE-MSG-RETRY] forzar `execute_sql_write(...)` (INSERT SQL
    directo post P1-NEON-DB-MIGRATION) a levantar → el decorator reintenta
    hasta 3 veces, luego reraise."""
    call_count = {"n": 0}

    def fake_execute_sql_write(_sql, _params=None, **_kwargs):
        call_count["n"] += 1
        raise ConnectionError("db blip simulado")

    monkeypatch.setattr(db_chat_module, "execute_sql_write", fake_execute_sql_write)
    # tenacity por defecto duerme entre attempts — para test rápido,
    # patcheamos `wait` a no-op vía `.retry.wait`:
    try:
        db_chat_module._save_message_insert_with_retry.retry.wait = (
            __import__("tenacity").wait_none()
        )
    except Exception:
        # Si no podemos patchear, el test corre lento pero pasa.
        pass

    with pytest.raises((ConnectionError, Exception)):
        db_chat_module._save_message_insert_with_retry(
            "sess-x", "user", "blip-test", None  # user_id=None → guest legítimo
        )
    assert call_count["n"] == 3, (
        f"[P2-CHAT-SAVE-MSG-RETRY] esperaba 3 attempts (1 inicial + 2 "
        f"retries), recibió {call_count['n']}."
    )


def test_retry_success_first_try(db_chat_module, monkeypatch) -> None:
    """[P2-CHAT-SAVE-MSG-RETRY] cuando el INSERT pasa al primer intento,
    NO reintenta — overhead trivial. Verifica además que el INSERT viaja a
    `public.agent_messages` con los 4 named params (P1-CHAT-DB-USER-ID-RLS:
    `user_id` puede ser None para guests)."""
    call_count = {"n": 0}
    captured: dict = {}

    def fake_execute_sql_write(sql, params=None, **kwargs):
        call_count["n"] += 1
        captured["sql"] = sql
        captured["params"] = params
        return None  # éxito

    monkeypatch.setattr(db_chat_module, "execute_sql_write", fake_execute_sql_write)
    db_chat_module._save_message_insert_with_retry(
        "sess-x", "model", "hola", None
    )
    assert call_count["n"] == 1, (
        f"[P2-CHAT-SAVE-MSG-RETRY] éxito al primer intento debe NO "
        f"reintentar. call_count={call_count['n']}."
    )
    assert "INSERT INTO public.agent_messages" in str(captured["sql"]), (
        f"El helper debe INSERTar en public.agent_messages. SQL: {captured['sql']!r}"
    )
    assert captured["params"] == {
        "session_id": "sess-x",
        "role": "model",
        "content": "hola",
        "user_id": None,
    }, (
        f"Named params inesperados: {captured['params']!r} — el INSERT debe "
        "llevar session_id/role/content/user_id (user_id None = guest)."
    )
