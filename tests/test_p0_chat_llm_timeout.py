"""[P0-CHAT-LLM-TIMEOUT · 2026-05-19] Test parser-based + funcional: cada uno
de los 5 callsites de `ChatGoogleGenerativeAI(...)` en `backend/agent.py` pasa
`timeout=` al constructor, y `chat_with_agent` (path non-streaming) envuelve
`chat_graph_app.invoke(...)` con `concurrent.futures.ThreadPoolExecutor` para
imponer un total-graph timeout. El router `/api/chat` mapea `TimeoutError` a
`HTTP 504`.

Por qué este test:
    Pre-fix las 5 callsites se construían sin `timeout=`. Si Gemini se colgaba
    (sobrecarga, red, quota del provider), `*.invoke(...)` bloqueaba el worker
    thread del threadpool de FastAPI indefinidamente → bajo concurrencia,
    thread pool starvation. El audit de production-readiness 2026-05-19 lo
    flageó como gap CRÍTICO (resto del repo defiende este modo de fallo con
    `LLMCircuitBreaker` y `MEALFIT_CB_*`, pero el chat-agent no lo invocaba).

Capas defendidas (los 5 timeouts + 1 graph-total):
    1. `llm` (módulo-level, swap/chat default)            → MEALFIT_CHAT_AGENT_LLM_TIMEOUT_S
    2. `swap_llm` dentro de `swap_meal`                   → MEALFIT_CHAT_SWAP_LLM_TIMEOUT_S
    3. `chat_llm` dentro de `call_model` (LangGraph node) → MEALFIT_CHAT_AGENT_LLM_TIMEOUT_S
    4. `title_llm` dentro de `generate_session_title`     → MEALFIT_CHAT_TITLE_LLM_TIMEOUT_S
    5. `router_llm` dentro de `rag_query_router`          → MEALFIT_CHAT_ROUTER_LLM_TIMEOUT_S
    + total-graph wrap                                    → MEALFIT_CHAT_GRAPH_TOTAL_TIMEOUT_S

Cross-link convention (P2-HIST-AUDIT-14): el slug `p0_chat_llm_timeout` matchea
este archivo `test_p0_chat_llm_timeout.py`.

Tooltip-anchor: P0-CHAT-LLM-TIMEOUT-START | audit prod-readiness 2026-05-19
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_AGENT_PY = _BACKEND_ROOT / "agent.py"
_ROUTER_CHAT_PY = _BACKEND_ROOT / "routers" / "chat.py"


@pytest.fixture(scope="module")
def agent_src() -> str:
    return _AGENT_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def router_src() -> str:
    return _ROUTER_CHAT_PY.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Los 5 helpers de timeout están definidos via `_env_float`
# ---------------------------------------------------------------------------
_EXPECTED_TIMEOUT_HELPERS = [
    ("_chat_agent_llm_timeout_s", "MEALFIT_CHAT_AGENT_LLM_TIMEOUT_S"),
    ("_chat_swap_llm_timeout_s", "MEALFIT_CHAT_SWAP_LLM_TIMEOUT_S"),
    ("_chat_title_llm_timeout_s", "MEALFIT_CHAT_TITLE_LLM_TIMEOUT_S"),
    ("_chat_router_llm_timeout_s", "MEALFIT_CHAT_ROUTER_LLM_TIMEOUT_S"),
    ("_chat_graph_total_timeout_s", "MEALFIT_CHAT_GRAPH_TOTAL_TIMEOUT_S"),
]


@pytest.mark.parametrize("helper_name, knob_name", _EXPECTED_TIMEOUT_HELPERS)
def test_timeout_helper_defined_with_env_float(
    agent_src: str, helper_name: str, knob_name: str
):
    """Cada helper `def _chat_*_timeout_s() -> float:` lee `_env_float(...)`
    sobre el knob `MEALFIT_*_TIMEOUT_S`, con validator de rango. Auto-registry
    en `_KNOBS_REGISTRY` (P3-NEW-D) — el SRE puede verificar overrides via
    `/api/system/admin/knobs` sin redeploy."""
    def_re = re.compile(
        rf"def\s+{re.escape(helper_name)}\s*\(\s*\)\s*->\s*float\s*:",
    )
    m = def_re.search(agent_src)
    assert m is not None, (
        f"P0-CHAT-LLM-TIMEOUT regresión: helper "
        f"`def {helper_name}() -> float:` no definido en agent.py. Sin él, "
        f"el callsite no puede leer el knob `{knob_name}` y queda sin defensa."
    )
    body_start = m.end()
    next_def = re.search(r"\n(?:def\s|llm\s*=)", agent_src[body_start:])
    body_end = body_start + (
        next_def.start() if next_def else min(600, len(agent_src) - body_start)
    )
    body = agent_src[body_start:body_end]
    assert knob_name in body, (
        f"P0-CHAT-LLM-TIMEOUT regresión: helper `{helper_name}` no lee "
        f"`{knob_name}`. Debe llamar `_env_float(\"{knob_name}\", <default>, "
        f"validator=...)`."
    )
    assert "_env_float" in body, (
        f"P0-CHAT-LLM-TIMEOUT regresión: helper `{helper_name}` no usa "
        f"`_env_float`. Sin `_env_float` no se auto-registra en _KNOBS_REGISTRY "
        f"(P3-NEW-D)."
    )
    assert "validator" in body, (
        f"P0-CHAT-LLM-TIMEOUT regresión: helper `{helper_name}` no tiene "
        f"`validator=` — env var corrupta (negativo, 0, mil) crearía timeout "
        f"patológico. Validator debe clamp a rango razonable (e.g. 0 < v <= 120)."
    )


# ---------------------------------------------------------------------------
# 2. Cada uno de los 5 callsites `ChatGoogleGenerativeAI(...)` pasa `timeout=`
# ---------------------------------------------------------------------------
def test_all_5_callsites_pass_timeout(agent_src: str):
    """Cada callsite `ChatGoogleGenerativeAI(...)` debe tener `timeout=`
    en su lista de args. Pre-fix: cero callsites lo tenían."""
    no_comments = re.sub(r"#[^\n]*", "", agent_src)
    callsite_re = re.compile(r"ChatGoogleGenerativeAI\s*\(")
    callsites = list(callsite_re.finditer(no_comments))

    assert len(callsites) == 5, (
        f"P0-CHAT-LLM-TIMEOUT tripwire: detectados {len(callsites)} callsites "
        f"de `ChatGoogleGenerativeAI(...)` en agent.py (esperados 5). Si "
        f"añadiste un callsite nuevo, asegúrate de pasarle `timeout=` y "
        f"actualizar este conteo."
    )

    timeout_kwarg_re = re.compile(r"timeout\s*=\s*_chat_\w+_timeout_s\s*\(\s*\)")
    offenders = []
    for m in callsites:
        # Buscar la posición del paréntesis de cierre matching el de apertura.
        # Implementación simple: window de 600 chars (suficiente para args
        # multi-linea legibles).
        window = no_comments[m.end():m.end() + 600]
        if not timeout_kwarg_re.search(window):
            line_no = no_comments.count("\n", 0, m.start()) + 1
            offenders.append(f"line {line_no}")

    assert not offenders, (
        f"P0-CHAT-LLM-TIMEOUT regresión: {len(offenders)} callsites de "
        f"`ChatGoogleGenerativeAI(...)` SIN `timeout=_chat_*_timeout_s()`: "
        f"{offenders}. Sin timeout per-LLM, una llamada colgada a Gemini "
        f"bloquea el worker thread indefinidamente → thread pool starvation."
    )


# ---------------------------------------------------------------------------
# 3. `chat_with_agent` (non-streaming) envuelve invoke con concurrent.futures
# ---------------------------------------------------------------------------
def test_graph_invoke_wrapped_with_total_timeout(agent_src: str):
    """`chat_with_agent` wrappea `chat_graph_app.invoke(...)` con
    `ThreadPoolExecutor` + `.result(timeout=...)` para imponer un budget
    total al pipeline LangGraph (defensa-en-profundidad sobre per-LLM)."""
    # Buscar referencias a `chat_graph_app.invoke` — tanto callsite directo
    # `chat_graph_app.invoke(...)` como pasado por referencia a un submit
    # (`_ex.submit(chat_graph_app.invoke, ...)`), que es el patrón canónico
    # post-P0-CHAT-LLM-TIMEOUT.
    invoke_re = re.compile(r"chat_graph_app\.invoke[\s,(]")
    matches = list(invoke_re.finditer(agent_src))
    assert len(matches) >= 1, (
        "P0-CHAT-LLM-TIMEOUT regresión: no se encontró `chat_graph_app.invoke` "
        "en agent.py. Es el callsite síncrono del path non-streaming."
    )

    # Para AL MENOS UNO de los callsites, debe haber `ThreadPoolExecutor`
    # + `.result(timeout=` en las ~50 líneas previas.
    found_wrap = False
    for m in matches:
        window_start = max(0, m.start() - 2000)
        window = agent_src[window_start:m.end() + 500]
        if (
            "ThreadPoolExecutor" in window
            and ".result(timeout=" in window
            and "_chat_graph_total_timeout_s" in window
        ):
            found_wrap = True
            break

    assert found_wrap, (
        "P0-CHAT-LLM-TIMEOUT regresión: ningún `chat_graph_app.invoke(...)` está "
        "envuelto en `ThreadPoolExecutor` + `.result(timeout=_chat_graph_total_timeout_s())`. "
        "Sin este wrap, el pipeline LangGraph completo puede bloquear el worker "
        "thread incluso si los per-LLM timeouts no disparan (e.g. tool interno "
        "que cuelga, recursión inesperada)."
    )


# ---------------------------------------------------------------------------
# 4. Router `routers/chat.py` mapea TimeoutError → HTTP 504
# ---------------------------------------------------------------------------
def test_router_maps_timeout_to_504(router_src: str):
    """El endpoint `POST /api/chat` (api_chat handler) DEBE catchear
    `TimeoutError` antes del catch genérico Exception, y propagar
    `HTTPException(status_code=504, ...)`. 504 GATEWAY TIMEOUT comunica
    al frontend que upstream (Gemini) no respondió a tiempo."""
    # Buscamos en el bloque del api_chat handler.
    api_chat_re = re.compile(r"def\s+api_chat\s*\(")
    m = api_chat_re.search(router_src)
    assert m is not None, (
        "P0-CHAT-LLM-TIMEOUT regresión: handler `def api_chat(...)` no "
        "encontrado en routers/chat.py. Si lo renombraste, actualizar este test."
    )
    # Body desde aquí hasta la próxima def top-level (o EOF).
    body_start = m.end()
    next_def = re.search(r"\n@router\.|\ndef\s|\nasync\s+def\s", router_src[body_start:])
    body_end = body_start + (
        next_def.start() if next_def else len(router_src) - body_start
    )
    body = router_src[body_start:body_end]

    assert "except TimeoutError" in body, (
        "P0-CHAT-LLM-TIMEOUT regresión: `def api_chat(...)` NO catchea "
        "`TimeoutError` explícitamente. Sin este catch, el TimeoutError cae al "
        "`except Exception` genérico → HTTP 500 (incorrecto semánticamente; "
        "504 GATEWAY TIMEOUT comunica que upstream no respondió)."
    )
    # El catch de TimeoutError debe raise HTTPException(504).
    timeout_block_re = re.compile(
        r"except\s+TimeoutError[^:]*:[\s\S]*?raise\s+HTTPException\s*\(\s*\n?\s*status_code\s*=\s*504"
    )
    assert timeout_block_re.search(body), (
        "P0-CHAT-LLM-TIMEOUT regresión: el catch de TimeoutError en api_chat "
        "no propaga `HTTPException(status_code=504, ...)`. Verificar que el "
        "código del status sea 504 (GATEWAY TIMEOUT)."
    )


# ---------------------------------------------------------------------------
# 5. Anchor textual P0-CHAT-LLM-TIMEOUT presente
# ---------------------------------------------------------------------------
def test_anchor_present_agent(agent_src: str):
    """Comment inline `[P0-CHAT-LLM-TIMEOUT · ...]` cerca de los knobs y los
    callsites para `grep -r P0-CHAT-LLM-TIMEOUT` localizar el fix sin abrir
    los archivos."""
    assert "P0-CHAT-LLM-TIMEOUT" in agent_src, (
        "P0-CHAT-LLM-TIMEOUT regresión: anchor textual perdido en agent.py."
    )


def test_anchor_present_router(router_src: str):
    assert "P0-CHAT-LLM-TIMEOUT" in router_src, (
        "P0-CHAT-LLM-TIMEOUT regresión: anchor textual perdido en routers/chat.py."
    )


# ---------------------------------------------------------------------------
# 6. Defaults razonables (sanity)
# ---------------------------------------------------------------------------
def test_defaults_are_reasonable():
    """Sanity check de defaults en runtime. Importamos los helpers via su
    módulo directo y verificamos que retornan valores en rango razonable
    bajo env vars vacías."""
    import importlib
    import os

    # Limpiar env vars potenciales del entorno de test.
    for kn in [
        "MEALFIT_CHAT_AGENT_LLM_TIMEOUT_S",
        "MEALFIT_CHAT_SWAP_LLM_TIMEOUT_S",
        "MEALFIT_CHAT_TITLE_LLM_TIMEOUT_S",
        "MEALFIT_CHAT_ROUTER_LLM_TIMEOUT_S",
        "MEALFIT_CHAT_GRAPH_TOTAL_TIMEOUT_S",
    ]:
        os.environ.pop(kn, None)

    try:
        agent_mod = importlib.import_module("agent")
    except Exception as e:
        pytest.skip(f"agent module import falló (deps externas): {e}")

    # Defaults DOCUMENTADOS:
    expectations = [
        ("_chat_agent_llm_timeout_s", 15.0),
        ("_chat_swap_llm_timeout_s", 30.0),
        ("_chat_title_llm_timeout_s", 10.0),
        # [P1-CHAT-EMPTY-RESPONSE · 2026-05-20] default router bumpeado 8.0 → 12.0
        # (Gemini API rechaza deadlines <10s con HTTP 400). Este test quedó stale
        # tras ese bump; alineado al default documentado del código (2026-06-01).
        ("_chat_router_llm_timeout_s", 12.0),
        ("_chat_graph_total_timeout_s", 60.0),
    ]
    for fn_name, expected_default in expectations:
        fn = getattr(agent_mod, fn_name, None)
        assert fn is not None, (
            f"P0-CHAT-LLM-TIMEOUT: helper `{fn_name}` no importable desde agent."
        )
        v = fn()
        assert v == expected_default, (
            f"P0-CHAT-LLM-TIMEOUT: `{fn_name}()` retornó {v}, esperado "
            f"{expected_default} (default documentado). Si cambias el default, "
            f"actualiza este test Y la sección operacional del CLAUDE.md."
        )
