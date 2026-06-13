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

# [P0-DEEPSEEK-MIGRATION · 2026-06-12] Los 4 constructores de embeddings
# per-módulo fueron consolidados en la capa pluggable `embeddings_provider.py`
# (constants/fact_extractor/vision_agent/shopping_calculator delegan a
# `get_text_embedding`/`get_embeddings_client`). El deadline vive en UN solo
# punto: `_build_client` pasa `timeout=` (kwarg nativo de OpenAIEmbeddings)
# resuelto por `_embeddings_timeout_s` — misma lección P2-LLM-TIMEOUT-SWEEP.
_EMBEDDINGS_PROVIDER = "embeddings_provider.py"


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
    # [P0-DEEPSEEK-MIGRATION · 2026-06-12] constructor renombrado.
    return _balanced_blocks(src, "ChatDeepSeek")


@pytest.mark.parametrize("module", _MODULES)
def test_every_constructor_has_timeout(module):
    src = _read(module)
    blocks = _constructor_blocks(src)
    assert blocks, f"No se encontró ningún ChatDeepSeek( en {module}."
    expected = _EXPECTED_MIN_CONSTRUCTORS[module]
    assert len(blocks) >= expected, (
        f"{module}: se esperaban >= {expected} constructores ChatDeepSeek, "
        f"se hallaron {len(blocks)}. ¿Cambió el formato o se removió un callsite? "
        "Revisa antes de asumir que el scan sigue cubriendo todo."
    )
    missing = [ln for (ln, inner) in blocks if "timeout=" not in inner]
    assert not missing, (
        f"{module}: constructores ChatDeepSeek SIN `timeout=` en líneas "
        f"{missing}. Un provider colgado bloquearía el thread/event-loop indefinidamente "
        "(P2-LLM-TIMEOUT-SWEEP). Añade `timeout=<helper>()` al constructor. "
        "Lee la memoria antes de remover un timeout existente."
    )


def test_anchor_present_in_all_modules():
    for module in _MODULES:
        src = _read(module)
        assert "P2-LLM-TIMEOUT-SWEEP" in src, (
            f"Falta anchor `P2-LLM-TIMEOUT-SWEEP` en {module}."
        )


def test_every_embeddings_constructor_has_client_args_timeout():
    """[P2-LLM-TIMEOUT-SWEEP · P0-DEEPSEEK-MIGRATION] El constructor del
    cliente de embeddings (consolidado en `embeddings_provider._build_client`)
    DEBE pasar `timeout=` — sin deadline, un socket de embedding colgado
    bloquea el thread del bg-pool / fact-lock para siempre."""
    src = _read(_EMBEDDINGS_PROVIDER)
    blocks = _balanced_blocks(src, "OpenAIEmbeddings")
    # 1 constructor real (en _build_client). El import lazy no matchea
    # `OpenAIEmbeddings(` porque no abre paréntesis.
    constructor_blocks = [b for b in blocks if "model" in b[1] or "kwargs" in b[1]]
    assert constructor_blocks or '"timeout"' in src, (
        f"{_EMBEDDINGS_PROVIDER}: no se encontró el constructor OpenAIEmbeddings."
    )
    assert '"timeout": _embeddings_timeout_s()' in src, (
        f"{_EMBEDDINGS_PROVIDER}: el constructor de embeddings debe acotar el "
        "deadline via `\"timeout\": _embeddings_timeout_s()` en sus kwargs "
        "(P2-LLM-TIMEOUT-SWEEP). Sin esto, un socket colgado bloquea el thread."
    )


def test_embeddings_timeout_helper_uses_env_float_with_validator():
    """El helper `_embeddings_timeout_s` (SSOT en embeddings_provider) debe
    resolver vía `_env_float` con `validator=` (clamp) — mismo knob legacy
    `MEALFIT_EMBEDDING_LLM_TIMEOUT_S`."""
    src = _read(_EMBEDDINGS_PROVIDER)
    m = re.search(
        r"def\s+_embeddings_timeout_s\s*\(\s*\)\s*->\s*float\s*:(.*?)(?:\n\ndef |\n\nclass |\Z)",
        src,
        re.DOTALL,
    )
    assert m, f"Falta el helper `_embeddings_timeout_s` en {_EMBEDDINGS_PROVIDER}."
    body = m.group(1)
    assert "_env_float(" in body and "validator=" in body, (
        "`_embeddings_timeout_s` debe usar `_env_float(..., validator=)`."
    )
    assert "MEALFIT_EMBEDDING_LLM_TIMEOUT_S" in body


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
