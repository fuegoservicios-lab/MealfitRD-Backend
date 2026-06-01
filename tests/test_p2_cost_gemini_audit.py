"""[P2-COST-GEMINI-AUDIT · 2026-06-01] Regresión del bundle de optimización de
COSTE de API Gemini (3 subsistemas: generación de planes, chunks/aprendizaje,
agente IA). Cierra los gaps confirmados por el audit find→verify:

  1. thinking_budget cap (helper compartido `knobs.thinking_budget_kwargs`)
     aplicado a 3 nodos thinking-capable de relleno-de-schema FUERA del pipeline
     de planes — swap_llm (agent.py), modify_llm (tools.py), fact-extractor PRO
     (fact_extractor.py). El cap P1-COST-THINKING-CAP original solo cubría
     day-gen + correctores.
  2. max_output_tokens en el generador de título de chat (output descartado).
  3. Reorder del system prompt del chat (estáticos al frente) para maximizar el
     cache implícito de Gemini (knob MEALFIT_CHAT_PROMPT_STATIC_PREFIX).
  4. Reviewer médico: split del REVIEWER_SYSTEM_PROMPT a SystemMessage cacheable
     bajo el knob existente PROMPT_CACHE_SYSTEM_MESSAGE.
  5. Fact-checker: dedup EXACTO de all_ingredients solo para el prompt
     (preserva la lista cruda para el validador determinista).

Tests parser-based: incluyen tooltip-anchors en el código de prod (un renombre
falla el test ANTES de cambiar producción). El test de `thinking_budget_kwargs`
es behavioral (knobs.py es dependency-free → importable sin tocar Gemini/DB).
"""
from __future__ import annotations

import re
import importlib
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent


def _read(name: str) -> str:
    return (_BACKEND / name).read_text(encoding="utf-8")


def _strip_comments(text: str) -> str:
    """Quita líneas de comentario completas e inline `#...`. Suficiente para
    las regiones de ensamblaje que verificamos (no contienen `#` dentro de
    strings). Espeja el patrón `_code_only` usado en otros tests del repo."""
    out = []
    for line in text.splitlines():
        # quitar inline comment (heurística: primer '#' fuera de string —
        # nuestras líneas objetivo no tienen '#' en literales)
        if "#" in line:
            line = line[: line.index("#")]
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 1. thinking_budget_kwargs — behavioral (helper compartido en knobs.py)
# ---------------------------------------------------------------------------
def test_thinking_budget_kwargs_caps_flash_with_default(monkeypatch):
    knobs = importlib.import_module("knobs")
    monkeypatch.delenv("TEST_TB_VAR_XYZ", raising=False)
    assert knobs.thinking_budget_kwargs("gemini-3.5-flash", "TEST_TB_VAR_XYZ", 2048) == {
        "thinking_budget": 2048
    }


def test_thinking_budget_kwargs_skips_lite(monkeypatch):
    knobs = importlib.import_module("knobs")
    monkeypatch.delenv("TEST_TB_VAR_XYZ", raising=False)
    # flash-lite NO soporta thinking_config → dict vacío (no rompe constructor).
    assert knobs.thinking_budget_kwargs("gemini-3.1-flash-lite", "TEST_TB_VAR_XYZ", 2048) == {}


def test_thinking_budget_kwargs_sentinel_negative_disables(monkeypatch):
    knobs = importlib.import_module("knobs")
    monkeypatch.setenv("TEST_TB_VAR_XYZ", "-1")
    # -1 = sin cap (rollback sin redeploy) → dict vacío.
    assert knobs.thinking_budget_kwargs("gemini-3.5-flash", "TEST_TB_VAR_XYZ", 2048) == {}


def test_thinking_budget_kwargs_env_override(monkeypatch):
    knobs = importlib.import_module("knobs")
    monkeypatch.setenv("TEST_TB_VAR_XYZ", "512")
    assert knobs.thinking_budget_kwargs("gemini-3.5-flash", "TEST_TB_VAR_XYZ", 2048) == {
        "thinking_budget": 512
    }


def test_thinking_budget_kwargs_empty_model(monkeypatch):
    knobs = importlib.import_module("knobs")
    monkeypatch.delenv("TEST_TB_VAR_XYZ", raising=False)
    assert knobs.thinking_budget_kwargs("", "TEST_TB_VAR_XYZ", 2048) == {}


# ---------------------------------------------------------------------------
# 2. agent.py — swap cap + title cap + chat prompt reorder
# ---------------------------------------------------------------------------
def test_agent_imports_thinking_helper():
    src = _read("agent.py")
    assert "thinking_budget_kwargs" in src, "agent.py debe importar thinking_budget_kwargs."
    assert "_env_int" in src and "_env_bool" in src


def test_swap_llm_has_thinking_cap():
    src = _strip_comments(_read("agent.py"))
    # El constructor de swap_llm debe spread el helper con su knob dedicado.
    assert 'thinking_budget_kwargs(_chat_agent_swap_model_name(), "MEALFIT_SWAP_THINKING_BUDGET"' in src, (
        "swap_llm (agent.py) debe capar reasoning vía "
        "thinking_budget_kwargs(..., 'MEALFIT_SWAP_THINKING_BUDGET', ...). "
        "tooltip-anchor: P2-COST-THINKING-CAP-EXT"
    )


def test_chat_title_has_output_cap():
    src = _strip_comments(_read("agent.py"))
    assert "def _chat_title_max_output_tokens(" in src
    assert "MEALFIT_CHAT_TITLE_MAX_OUTPUT_TOKENS" in src
    assert "max_output_tokens=_chat_title_max_output_tokens()" in src, (
        "title_llm debe pasar max_output_tokens=_chat_title_max_output_tokens(). "
        "tooltip-anchor: P3-COST-TITLE-OUTPUT-CAP"
    )


def test_chat_prompt_static_prefix_helper_and_knob():
    src = _read("agent.py")
    assert "def _chat_prompt_static_prefix(" in src
    assert "MEALFIT_CHAT_PROMPT_STATIC_PREFIX" in src


def test_chat_prompt_static_prefix_ordering_both_paths():
    """En la rama static-prefix, los bloques ESTÁTICOS (tools instructions) deben
    preceder al VOLÁTIL build_temporal_context() en AMBOS paths (stream +
    non-stream). Garantiza que el timestamp con minuto no rompa el prefijo
    cacheable."""
    src = _strip_comments(_read("agent.py"))
    # Localizar las 2 ramas `if _chat_prompt_static_prefix():` (una por path).
    branches = [m.start() for m in re.finditer(r"if _chat_prompt_static_prefix\(\):", src)]
    assert len(branches) >= 2, (
        f"Esperaba >=2 ramas `if _chat_prompt_static_prefix():` (stream + non-stream), "
        f"encontré {len(branches)}."
    )
    for start in branches:
        # Slice hasta el `else:` de esa rama (o 1200 chars de margen).
        region = src[start : start + 1200]
        else_idx = region.find("\n    else:")
        if else_idx != -1:
            region = region[:else_idx]
        # build_tools_instructions(_stream)(user_id) debe aparecer ANTES de
        # build_temporal_context() dentro de la rama static-prefix.
        m_tools = re.search(r"build_tools_instructions(?:_stream)?\(user_id\)", region)
        m_temporal = region.find("build_temporal_context()")
        assert m_tools is not None, f"build_tools_instructions no hallado en rama @ {start}"
        assert m_temporal != -1, f"build_temporal_context no hallado en rama @ {start}"
        assert m_tools.start() < m_temporal, (
            "En la rama static-prefix, build_tools_instructions (estático) debe ir "
            "ANTES de build_temporal_context() (volátil con minuto). "
            "tooltip-anchor: P2-CHAT-PROMPT-STATIC-PREFIX"
        )


# ---------------------------------------------------------------------------
# 3. tools.py — modify_llm thinking cap
# ---------------------------------------------------------------------------
def test_modify_llm_has_thinking_cap():
    src = _strip_comments(_read("tools.py"))
    assert "thinking_budget_kwargs" in _read("tools.py"), "tools.py debe importar thinking_budget_kwargs."
    assert 'thinking_budget_kwargs(_modify_model, "MEALFIT_MODIFY_THINKING_BUDGET"' in src, (
        "modify_llm (tools.py) debe capar reasoning vía "
        "thinking_budget_kwargs(_modify_model, 'MEALFIT_MODIFY_THINKING_BUDGET', ...). "
        "tooltip-anchor: P2-COST-THINKING-CAP-EXT"
    )


# ---------------------------------------------------------------------------
# 4. fact_extractor.py — PRO thinking cap
# ---------------------------------------------------------------------------
def test_fact_extractor_pro_has_thinking_cap():
    raw = _read("fact_extractor.py")
    assert "thinking_budget_kwargs" in raw, "fact_extractor.py debe importar thinking_budget_kwargs."
    src = _strip_comments(raw)
    assert 'thinking_budget_kwargs(pro_model, "MEALFIT_FACT_EXTRACTOR_THINKING_BUDGET"' in src, (
        "El PRO de _invoke_with_shadow (fact_extractor.py) debe capar reasoning vía "
        "thinking_budget_kwargs(pro_model, 'MEALFIT_FACT_EXTRACTOR_THINKING_BUDGET', ...). "
        "tooltip-anchor: P2-COST-THINKING-CAP-EXT"
    )


# ---------------------------------------------------------------------------
# 5. graph_orchestrator.py — reviewer split + factchecker dedup
# ---------------------------------------------------------------------------
def test_reviewer_systemmessage_split():
    src = _strip_comments(_read("graph_orchestrator.py"))
    # Rama cacheable: SystemMessage(REVIEWER_SYSTEM_PROMPT) bajo el knob.
    assert "if PROMPT_CACHE_SYSTEM_MESSAGE:" in src
    assert re.search(
        r"review_prompt\s*=\s*\[\s*SystemMessage\(content=REVIEWER_SYSTEM_PROMPT\)",
        src,
    ), (
        "El reviewer debe construir review_prompt como [SystemMessage(REVIEWER_SYSTEM_PROMPT), "
        "HumanMessage(...)] cuando PROMPT_CACHE_SYSTEM_MESSAGE. tooltip-anchor: P3-COST-REVIEWER-CACHE"
    )
    # Rama legacy plana preservada (rollback).
    assert "review_prompt = f\"{REVIEWER_SYSTEM_PROMPT}" in src or \
           'review_prompt = f"{REVIEWER_SYSTEM_PROMPT}' in src


def test_factchecker_ingredients_deduped_for_prompt():
    src = _strip_comments(_read("graph_orchestrator.py"))
    assert "_fc_ingredients = sorted(set(all_ingredients))" in src, (
        "El fact-checker debe dedupear all_ingredients (EXACTO) solo para el prompt. "
        "tooltip-anchor: P3-COST-FACTCHECK-INGREDIENTS-DEDUP"
    )
    assert "Ingredientes a evaluar: {_fc_ingredients}" in src, (
        "La HumanMessage del fact-checker debe usar la vista dedupeada _fc_ingredients."
    )
    # all_ingredients crudo se preserva para el validador determinista.
    assert "all_ingredients.extend(ingredients)" in src


# ---------------------------------------------------------------------------
# 6. Marker bumpeado al bundle
# ---------------------------------------------------------------------------
def test_marker_bumped_to_cost_audit():
    """El marker se bumpeó al cerrar el bundle de coste. Pasadas POSTERIORES del
    mismo árbol (p.ej. P1-EVALUATOR-THINKING-CAP · 2026-06-01) supersedan el string
    exacto (last-writer-wins del único `_LAST_KNOWN_PFIX`), así que verificamos
    floor-de-fecha 2026-06-01+ en vez del literal — espejo del patrón ya aplicado a
    test_p2_cron_opt_2/4."""
    src = _read("app.py")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"[^"]*·\s*(\d{4}-\d{2}-\d{2})"', src)
    assert m, "No se pudo parsear _LAST_KNOWN_PFIX en app.py."
    assert m.group(1) >= "2026-06-01", (
        f"El marker debe estar fechado 2026-06-01+ (cierre del bundle de coste o una "
        f"pasada posterior). Hallado: {m.group(1)!r}."
    )
