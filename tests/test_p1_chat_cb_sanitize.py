"""[P1-CHAT-CB-SANITIZE · 2026-05-19] Bundle de tests parser-based para los
dos P1 derivados del audit production-readiness del Agente:

  1. **P1-CHAT-CB**: el nodo LangGraph `call_model` (backend/agent.py) DEBE
     envolver el `chat_llm.invoke(...)` con el `LLMCircuitBreaker` per-modelo
     del `graph_orchestrator`. Pre-fix: chat era el único path productivo de
     Gemini SIN breaker (resto del repo — pipeline plan-gen, swap — sí lo
     usa). Bajo provider degradado, cada request pagaba latencia + tokens en
     vano y amplificaba la condición.

  2. **P1-MARKDOWN-SANITIZE**: `LazyMarkdown.jsx` (frontend) DEBE pasar
     `rehype-sanitize` como rehype-plugin obligatorio. Pre-fix: react-markdown
     renderizaba HTML inline sin filtro — vector XSS si vision_agent procesa
     una imagen con texto adversario que termina en chat context. Defensa-
     en-profundidad simétrica a P0-AGENT-1 (override `user_id`) pero del
     lado output.

Cross-link convention (P2-HIST-AUDIT-14): el slug `p1_chat_cb_sanitize` matchea
este archivo `test_p1_chat_cb_sanitize.py`.

Tooltip-anchor: P1-CHAT-CB-SANITIZE-START | audit prod-readiness 2026-05-19
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
_AGENT_PY = _BACKEND_ROOT / "agent.py"
_ROUTER_CHAT_PY = _BACKEND_ROOT / "routers" / "chat.py"
_LAZY_MARKDOWN_JSX = (
    _REPO_ROOT / "frontend" / "src" / "components" / "common" / "LazyMarkdown.jsx"
)
_FRONTEND_PACKAGE_JSON = _REPO_ROOT / "frontend" / "package.json"


@pytest.fixture(scope="module")
def agent_src() -> str:
    return _AGENT_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def router_src() -> str:
    return _ROUTER_CHAT_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def lazy_markdown_src() -> str:
    return _LAZY_MARKDOWN_JSX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def package_json_src() -> str:
    return _FRONTEND_PACKAGE_JSON.read_text(encoding="utf-8")


# ===========================================================================
# Sección 1 — P1-CHAT-CB: chat usa LLMCircuitBreaker per-modelo
# ===========================================================================

def test_cb_import_present(agent_src: str):
    """agent.py importa `_get_circuit_breaker` del graph_orchestrator.
    SSOT del breaker — NO duplicar la implementación."""
    assert (
        "from graph_orchestrator import _get_circuit_breaker" in agent_src
        or "from graph_orchestrator import" in agent_src
        and "_get_circuit_breaker" in agent_src
    ), (
        "P1-CHAT-CB regresión: `_get_circuit_breaker` no importado desde "
        "graph_orchestrator. Sin ese helper, el chat instanciaría su propio "
        "LLMCircuitBreaker (duplicación + storage layer divergente) o no "
        "usaría breaker en absoluto."
    )


def test_circuit_breaker_open_exception_defined(agent_src: str):
    """Excepción dedicada `LLMCircuitBreakerOpen(RuntimeError)` definida en
    agent.py. Permite al router catchear específicamente y mapear a HTTP 503
    (semántica correcta para 'reintenta más tarde')."""
    class_re = re.compile(
        r"class\s+LLMCircuitBreakerOpen\s*\(\s*RuntimeError\s*\)\s*:",
    )
    assert class_re.search(agent_src), (
        "P1-CHAT-CB regresión: clase `LLMCircuitBreakerOpen(RuntimeError)` "
        "no definida en agent.py. Sin ella, el router no puede distinguir "
        "'breaker abierto' (503) de 'timeout' (504) o 'error genérico' (500)."
    )


def test_call_model_wraps_invoke_with_breaker(agent_src: str):
    """Dentro de `def call_model(state: ChatState):`, el `chat_llm.invoke(...)`
    (vía `llm_with_tools.invoke(...)`) debe estar precedido por
    `_get_circuit_breaker(...).can_proceed()` y rodeado por try/except con
    `record_failure()` + `record_success()`. Sin este wrap, un Gemini
    degradado pega cada request individualmente."""
    fn_re = re.compile(r"def\s+call_model\s*\(\s*state\s*:")
    m = fn_re.search(agent_src)
    assert m is not None, (
        "P1-CHAT-CB regresión: nodo `def call_model(state: ChatState)` no "
        "encontrado en agent.py."
    )
    body_start = m.end()
    next_def = re.search(r"\ndef\s|\nclass\s", agent_src[body_start:])
    body_end = body_start + (
        next_def.start() if next_def else min(4000, len(agent_src) - body_start)
    )
    body = agent_src[body_start:body_end]

    # Gate antes del invoke.
    assert "_get_circuit_breaker(" in body, (
        "P1-CHAT-CB regresión: `call_model` no llama a `_get_circuit_breaker(...)`."
    )
    assert ".can_proceed()" in body, (
        "P1-CHAT-CB regresión: `call_model` no verifica `.can_proceed()` "
        "antes del invoke. Sin el gate, el breaker registra failures pero "
        "no impide nuevas llamadas mientras está abierto."
    )
    # Raise de la excepción dedicada cuando can_proceed=False.
    assert "raise LLMCircuitBreakerOpen" in body, (
        "P1-CHAT-CB regresión: `call_model` no raise `LLMCircuitBreakerOpen` "
        "cuando el breaker está cerrado. El router no podrá mapear a 503."
    )
    # record_success + record_failure presentes.
    assert "record_success()" in body, (
        "P1-CHAT-CB regresión: `call_model` no llama `record_success()` tras "
        "un invoke exitoso. Sin esa señal, el breaker no se cierra tras "
        "recuperación del provider."
    )
    assert "record_failure()" in body, (
        "P1-CHAT-CB regresión: `call_model` no llama `record_failure()` en el "
        "except. Sin esa señal, el breaker nunca se abre — efectivamente "
        "no-op."
    )
    # Orden: el gate `can_proceed` debe estar ANTES del `invoke`.
    can_proceed_pos = body.find(".can_proceed()")
    invoke_pos = body.find(".invoke(")
    assert (
        can_proceed_pos >= 0
        and invoke_pos >= 0
        and can_proceed_pos < invoke_pos
    ), (
        "P1-CHAT-CB regresión: `.can_proceed()` aparece DESPUÉS de `.invoke(`. "
        "El gate debe ir ANTES — sino paga latencia + tokens antes de "
        "fail-fast."
    )


def test_router_catches_cb_open_503(router_src: str):
    """El handler `api_chat` en routers/chat.py importa
    `LLMCircuitBreakerOpen` y la catchea ANTES del catch de `Exception`,
    mapeando a `HTTPException(503, ...)`."""
    assert "LLMCircuitBreakerOpen" in router_src, (
        "P1-CHAT-CB regresión: routers/chat.py no importa "
        "`LLMCircuitBreakerOpen`. El catch específico no existe → CB-abierto "
        "cae al genérico 500."
    )
    # Buscar el except + raise 503 en cualquier handler.
    cb_block_re = re.compile(
        r"except\s+LLMCircuitBreakerOpen[^:]*:[\s\S]*?raise\s+HTTPException\s*\(\s*\n?\s*status_code\s*=\s*503"
    )
    assert cb_block_re.search(router_src), (
        "P1-CHAT-CB regresión: routers/chat.py no mapea "
        "`LLMCircuitBreakerOpen → HTTPException(503)`. Verificar que el "
        "status_code sea 503 (Service Unavailable — semánticamente 'reintenta')."
    )


def test_anchor_p1_chat_cb_present(agent_src: str, router_src: str):
    assert "P1-CHAT-CB" in agent_src, (
        "P1-CHAT-CB regresión: anchor textual perdido en agent.py."
    )
    assert "P1-CHAT-CB" in router_src, (
        "P1-CHAT-CB regresión: anchor textual perdido en routers/chat.py."
    )


# ===========================================================================
# Sección 2 — P1-MARKDOWN-SANITIZE: LazyMarkdown sanitiza HTML inline
# ===========================================================================

def test_rehype_sanitize_in_package_json(package_json_src: str):
    """`rehype-sanitize` en `dependencies` de frontend/package.json. Sin
    él, el import falla en runtime y AgentPage no renderiza markdown."""
    assert '"rehype-sanitize"' in package_json_src, (
        "P1-MARKDOWN-SANITIZE regresión: `rehype-sanitize` no en "
        "frontend/package.json. Correr `npm install rehype-sanitize` para "
        "añadirlo o restaurar la línea manualmente."
    )


def test_lazy_markdown_imports_sanitize(lazy_markdown_src: str):
    """`LazyMarkdown.jsx` importa `rehype-sanitize` (dynamic import dentro
    del `lazy(async () => ...)` para que entre al mismo chunk async que
    react-markdown)."""
    assert "rehype-sanitize" in lazy_markdown_src, (
        "P1-MARKDOWN-SANITIZE regresión: `LazyMarkdown.jsx` no importa "
        "`rehype-sanitize`. Sin la import, los `rehypePlugins` no incluyen "
        "el sanitizer y HTML inline malicioso se renderiza."
    )
    # Import via dynamic `import('rehype-sanitize')` o static (ambos válidos).
    dynamic_re = re.compile(r"import\s*\(\s*['\"]rehype-sanitize['\"]")
    static_re = re.compile(r"from\s+['\"]rehype-sanitize['\"]")
    assert dynamic_re.search(lazy_markdown_src) or static_re.search(
        lazy_markdown_src
    ), (
        "P1-MARKDOWN-SANITIZE regresión: `rehype-sanitize` mencionado en "
        "LazyMarkdown.jsx pero no via `import(...)` ni `from '...'`. "
        "Comentarios no cuentan — el plugin debe estar realmente importado."
    )


def test_rehype_plugins_includes_sanitize(lazy_markdown_src: str):
    """El JSX `<ReactMarkdown rehypePlugins={[..., rehypeSanitize, ...]}>` o
    equivalente debe propagar el sanitizer. NO basta con importar."""
    # Match flexible: `rehypePlugins={[rehypeSanitize` o `[rehypeSanitize,`
    # o variable que arme la lista (caso del Wrapped component).
    has_inline = re.search(
        r"rehypePlugins\s*=\s*\{[\s\[]*rehypeSanitize", lazy_markdown_src
    )
    has_via_var = (
        "rehypeSanitize" in lazy_markdown_src
        and re.search(
            r"plugins\s*=\s*\[\s*rehypeSanitize", lazy_markdown_src
        )
        is not None
    )
    assert has_inline or has_via_var, (
        "P1-MARKDOWN-SANITIZE regresión: `rehypeSanitize` no propagado a "
        "`rehypePlugins` en LazyMarkdown.jsx. La import sola no protege — "
        "ReactMarkdown debe recibir el plugin en su lista de rehypePlugins."
    )


def test_anchor_p1_markdown_sanitize_present(lazy_markdown_src: str):
    assert "P1-MARKDOWN-SANITIZE" in lazy_markdown_src, (
        "P1-MARKDOWN-SANITIZE regresión: anchor textual perdido en "
        "LazyMarkdown.jsx."
    )
