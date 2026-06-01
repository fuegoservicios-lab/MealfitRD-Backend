"""[P2-LLM-TIMEOUT-SWEEP · 2026-05-30] Todo constructor `ChatGoogleGenerativeAI`
en los módulos auxiliares debe pasar `timeout=`.

Contexto (audit prod-readiness 2026-05-30, workflow multi-agente):
    `agent.py` ya bloqueaba sus invokes con `timeout=` (P0-CHAT-LLM-TIMEOUT),
    pero 12 constructores en 5 módulos auxiliares quedaron SIN timeout:
        - vision_agent.py (1)   → `await ainvoke` en handler async = cuelga el
          event loop del único worker uvicorn para TODAS las requests.
        - ai_helpers.py (4)     → title/recipe en threadpool de FastAPI;
          retrospectiva/flavor en thread del chunk-worker cron.
        - fact_extractor.py (3) → PRO sync mantiene tomado el fact-lock.
        - memory_manager.py (2) → cron `summarize_and_prune`, max_instances=1.
        - proactive_agent.py (2)→ cron `run_proactive_checks`, max_instances=1:
          un socket colgado dejaba el slot del job tomado para SIEMPRE (no
          dispara MISSED/ERROR — el subsistema de nudges muere sin alerta).

    El SDK de `langchain_google_genai` usa `timeout=None` por default
    (espera infinita en sockets colgados) + `max_retries` que NO avanza sobre
    un socket que no responde. El `timeout=` del constructor propaga al deadline
    del request gRPC → DeadlineExceeded, capturado por los `except` existentes
    que degradan a fallback.

Este test ancla el contrato a CI: si alguien añade un nuevo
`ChatGoogleGenerativeAI(...)` en estos módulos SIN `timeout=`, falla.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent

_MODULES = [
    "vision_agent.py",
    "ai_helpers.py",
    "fact_extractor.py",
    "memory_manager.py",
    "proactive_agent.py",
    # [P2-LLM-TIMEOUT-SWEEP · 2026-05-30] Cohorte omitida (audit prod-readiness
    # 2026-05-30 round-2): el probe LLM de auto-recovery de `_chunk_worker`
    # (GAP-6) era el único constructor LLM raw de cron_tasks.py y quedó sin
    # `timeout=` — halt silencioso del chunking en degradación Gemini.
    "cron_tasks.py",
    # [P2-LLM-TIMEOUT-SWEEP · 2026-05-30] Hot-path de chat: classify_sentiment
    # corre INLINE antes del wrapper _graph_timeout → socket colgado bloquea
    # el worker uvicorn.
    "sentiment_classifier.py",
    # [P2-PROD-AUDIT-3 · 2026-05-30] La tool médica corre en un ThreadPoolExecutor
    # cuyo thread no se puede matar; sin timeout un socket colgado nunca libera el
    # slot del pool → todo fact-check médico bloquea. Ver tools_medical.py.
    "tools_medical.py",
]

# Expectativa de cobertura mínima (defiende contra "alguien borró un módulo del
# scan"). El scan real verifica cada constructor; este conteo evita falsos verdes
# si el regex deja de matchear por un refactor de formato.
_EXPECTED_MIN_CONSTRUCTORS = {
    "vision_agent.py": 1,
    "ai_helpers.py": 4,
    "fact_extractor.py": 3,
    "memory_manager.py": 2,
    "proactive_agent.py": 2,
    "cron_tasks.py": 1,
    "sentiment_classifier.py": 1,
    "tools_medical.py": 1,
}

# [P2-LLM-TIMEOUT-SWEEP · 2026-05-30] `GoogleGenerativeAIEmbeddings` es una clase
# DISTINTA de `ChatGoogleGenerativeAI` y NO acepta `timeout=`; solo `client_args`
# (→ httpx.Client(timeout=)) acota el deadline. El sweep original solo cubrió
# `ChatGoogleGenerativeAI`, dejando los 3 constructores de embeddings sin acotar.
_EMBEDDING_MODULES = {
    "fact_extractor.py": 1,
    "vision_agent.py": 1,
    "constants.py": 1,
    # [P2-PROD-AUDIT-3 · 2026-05-30] Cliente embeddings de shopping_calculator
    # (semantic cache init + runtime embed_query) quedó sin deadline.
    "shopping_calculator.py": 1,
}


def _read(name: str) -> str:
    return (_BACKEND_ROOT / name).read_text(encoding="utf-8")


def _balanced_blocks(src: str, symbol: str):
    """Devuelve (line_no, inner_args_str) por cada `<symbol>(...)`.

    Balancea paréntesis para capturar el bloque completo del constructor
    (multi-línea), sin depender del formato exacto.
    """
    blocks = []
    for m in re.finditer(re.escape(symbol) + r"\(", src):
        start = m.end()  # justo después del '('
        depth = 1
        i = start
        while i < len(src) and depth > 0:
            c = src[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            i += 1
        inner = src[start : i - 1]
        line_no = src[: m.start()].count("\n") + 1
        blocks.append((line_no, inner))
    return blocks


def _constructor_blocks(src: str):
    return _balanced_blocks(src, "ChatGoogleGenerativeAI")


@pytest.mark.parametrize("module", _MODULES)
def test_every_constructor_has_timeout(module):
    src = _read(module)
    blocks = _constructor_blocks(src)
    assert blocks, f"No se encontró ningún ChatGoogleGenerativeAI( en {module}."
    expected = _EXPECTED_MIN_CONSTRUCTORS[module]
    assert len(blocks) >= expected, (
        f"{module}: se esperaban >= {expected} constructores ChatGoogleGenerativeAI, "
        f"se hallaron {len(blocks)}. ¿Cambió el formato o se removió un callsite? "
        "Revisa antes de asumir que el scan sigue cubriendo todo."
    )
    missing = [ln for (ln, inner) in blocks if "timeout=" not in inner]
    assert not missing, (
        f"{module}: constructores ChatGoogleGenerativeAI SIN `timeout=` en líneas "
        f"{missing}. Un Gemini colgado bloquearía el thread/event-loop indefinidamente "
        "(P2-LLM-TIMEOUT-SWEEP). Añade `timeout=<helper>()` al constructor. "
        "Lee la memoria antes de remover un timeout existente."
    )


def test_anchor_present_in_all_modules():
    for module in _MODULES:
        src = _read(module)
        assert "P2-LLM-TIMEOUT-SWEEP" in src, (
            f"Falta anchor `P2-LLM-TIMEOUT-SWEEP` en {module}."
        )


@pytest.mark.parametrize("module", sorted(_EMBEDDING_MODULES))
def test_every_embeddings_constructor_has_client_args_timeout(module):
    """[P2-LLM-TIMEOUT-SWEEP · 2026-05-30] Cada `GoogleGenerativeAIEmbeddings(...)`
    debe pasar `client_args={"timeout": ...}` — el ÚNICO arg que acota el deadline
    httpx en esta versión del SDK (NO acepta `timeout=` ni cablea `request_options`)."""
    src = _read(module)
    blocks = _balanced_blocks(src, "GoogleGenerativeAIEmbeddings")
    expected = _EMBEDDING_MODULES[module]
    assert len(blocks) >= expected, (
        f"{module}: se esperaban >= {expected} constructores GoogleGenerativeAIEmbeddings, "
        f"se hallaron {len(blocks)}. ¿Se removió/renombró un callsite?"
    )
    missing = [
        ln for (ln, inner) in blocks
        if "client_args" not in inner or "timeout" not in inner
    ]
    assert not missing, (
        f"{module}: GoogleGenerativeAIEmbeddings SIN `client_args={{'timeout': ...}}` en "
        f"líneas {missing}. Un socket de embedding colgado bloquearía el thread "
        "indefinidamente (P2-LLM-TIMEOUT-SWEEP). En esta versión del SDK `timeout=` "
        "NO existe para embeddings — usa `client_args`."
    )


def test_embeddings_timeout_helper_uses_env_float_with_validator():
    """El helper `_embeddings_llm_timeout_s` debe existir en los 3 módulos de
    embeddings y resolver vía `_env_float` con `validator=` (clamp)."""
    for module in _EMBEDDING_MODULES:
        src = _read(module)
        m = re.search(
            r"def\s+_embeddings_llm_timeout_s\s*\(\s*\)\s*->\s*float\s*:(.*?)(?:\n\ndef |\n\nclass |\Z)",
            src,
            re.DOTALL,
        )
        assert m, f"Falta el helper `_embeddings_llm_timeout_s` en {module}."
        body = m.group(1)
        assert "_env_float(" in body and "validator=" in body, (
            f"`_embeddings_llm_timeout_s` en {module} debe usar `_env_float(..., validator=)`."
        )


def test_timeout_helpers_use_env_float_with_validator():
    """Los helpers de timeout deben usar `_env_float` con `validator=` (clamp),
    para que un env var corrupto caiga al default en vez de un timeout patológico."""
    checks = {
        "vision_agent.py": "_vision_llm_timeout_s",
        "ai_helpers.py": "_ai_helpers_llm_timeout_s",
        "fact_extractor.py": "_fact_extractor_llm_timeout_s",
        "memory_manager.py": "_memory_summary_llm_timeout_s",
        "proactive_agent.py": "_proactive_llm_timeout_s",
    }
    for module, helper in checks.items():
        src = _read(module)
        m = re.search(
            rf"def\s+{re.escape(helper)}\s*\(\s*\)\s*->\s*float\s*:(.*?)(?:\n\ndef |\n\nclass |\Z)",
            src,
            re.DOTALL,
        )
        assert m, f"No se encontró el helper `{helper}` en {module}."
        body = m.group(1)
        assert "_env_float(" in body, (
            f"{helper} en {module} debe resolver el timeout vía `_env_float` "
            "(auto-registro en _KNOBS_REGISTRY)."
        )
        assert "validator=" in body, (
            f"{helper} en {module} debe pasar `validator=` a `_env_float` para "
            "clampear el rango (un env var corrupto no debe producir timeout=0 o gigante)."
        )
