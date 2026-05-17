"""[P1-PROMPT-CACHE-SYSTEMMSG · 2026-05-15] Regression guards para el switch
a `SystemMessage` + `HumanMessage` en `plan_skeleton_node` y `generate_single_day`.

Pre-fix: `PLANNER_SYSTEM_PROMPT` (~960 tok) y `DAY_GENERATOR_SYSTEM_PROMPT`
(~4900 tok) + schema JSON estaban concatenados AL FINAL de un único
`HumanMessage`. Gemini no podía identificar la región estática del prompt
→ implicit caching hit rate ≈ 0% → se pagaba full input price siempre.

Fix: cuando `MEALFIT_PROMPT_CACHE_SYSTEM_MESSAGE=True` (default) los system
prompts viajan en un `SystemMessage` separado que Gemini procesa como
`system_instruction` (target canónico del implicit cache, ~25% del input
price normal en hits).

Estos tests son parser-based — verifican la estructura del código sin
ejecutar el pipeline LLM.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GRAPH_PATH = _BACKEND_ROOT / "graph_orchestrator.py"


def _read_graph() -> str:
    return _GRAPH_PATH.read_text(encoding="utf-8")


# ----- Knob registrado --------------------------------------------------

def test_knob_registered_in_module():
    text = _read_graph()
    assert "MEALFIT_PROMPT_CACHE_SYSTEM_MESSAGE" in text, (
        "P1-PROMPT-CACHE-SYSTEMMSG: knob `MEALFIT_PROMPT_CACHE_SYSTEM_MESSAGE` "
        "debe estar declarado en graph_orchestrator.py (kill switch sin redeploy)."
    )
    # Default True para que el caching beneficie producción desde el deploy.
    m = re.search(
        r'PROMPT_CACHE_SYSTEM_MESSAGE\s*=\s*_env_bool\(\s*[\"\']MEALFIT_PROMPT_CACHE_SYSTEM_MESSAGE[\"\']\s*,\s*(True|False)\s*\)',
        text,
    )
    assert m, "Knob debe usar `_env_bool` con default explícito True/False."
    assert m.group(1) == "True", (
        "Default debe ser `True` — el caching debe estar habilitado desde el "
        "deploy. Flip a False solo si plan_quality_degraded alerts suben."
    )


# ----- planner_node (plan_skeleton) -------------------------------------

def test_planner_callsite_uses_systemmessage_under_knob():
    """En `invoke_planner` (dentro de `plan_skeleton_node`), el callsite a
    `_safe_ainvoke` debe construir `[SystemMessage(content=PLANNER_SYSTEM_PROMPT),
    HumanMessage(content=...)]` cuando `PROMPT_CACHE_SYSTEM_MESSAGE` está True."""
    text = _read_graph()
    # Localizar la región de `invoke_planner` por anchor del log + el callsite
    # `_safe_ainvoke(planner_llm, ...)`.
    region_match = re.search(
        r"async def invoke_planner.*?await _safe_ainvoke\(planner_llm,.*?\)",
        text,
        re.DOTALL,
    )
    assert region_match, "No encontré `invoke_planner` + callsite a `_safe_ainvoke`."
    region = region_match.group(0)
    assert "PROMPT_CACHE_SYSTEM_MESSAGE" in region, (
        "El callsite del planner debe gatear por `PROMPT_CACHE_SYSTEM_MESSAGE`."
    )
    assert "SystemMessage(content=PLANNER_SYSTEM_PROMPT)" in region, (
        "Cuando knob=True el SystemMessage debe contener `PLANNER_SYSTEM_PROMPT`."
    )
    assert "HumanMessage(content=prompt_text)" in region, (
        "Cuando knob=True el HumanMessage debe contener el `prompt_text` dinámico."
    )


def test_planner_dynamic_text_excludes_static_when_knob_true():
    """El `prompt_text` que va al HumanMessage NO debe contener
    `PLANNER_SYSTEM_PROMPT` cuando el knob está habilitado.
    Si lo contiene, la región estática se duplica (SystemMessage + tail del
    HumanMessage) y rompe el cache hit (Gemini ve prefix idéntico pero el
    HumanMessage diverge en el tail estático)."""
    text = _read_graph()
    # Buscar la construcción de `dynamic_prompt_text` y verificar que el
    # gate `if PROMPT_CACHE_SYSTEM_MESSAGE:` solo añade `PLANNER_SYSTEM_PROMPT`
    # en la rama legacy.
    # El bloque `dynamic_prompt_text = (...)` tiene paréntesis anidados de
    # f-strings — buscamos solo el branch knob (la firma única del fix).
    assert "dynamic_prompt_text = (" in text, (
        "Falta la asignación de `dynamic_prompt_text` en el planner."
    )
    m = re.search(
        r"if PROMPT_CACHE_SYSTEM_MESSAGE:\s*\n\s*prompt_text\s*=\s*dynamic_prompt_text\s*\n\s*else:\s*\n\s*prompt_text\s*=\s*dynamic_prompt_text\s*\+\s*PLANNER_SYSTEM_PROMPT",
        text,
    )
    assert m, (
        "P1-PROMPT-CACHE-SYSTEMMSG: la composición del prompt del planner "
        "debe usar el patrón `dynamic_prompt_text` + branch `if knob: prompt_text "
        "= dynamic; else: prompt_text = dynamic + PLANNER_SYSTEM_PROMPT`. "
        "Cuando el knob=True el system prompt NO debe vivir dentro del "
        "HumanMessage — duplicarlo invalida el cache."
    )


# ----- day_generator (generate_single_day) ------------------------------

def test_day_generator_constructs_system_instruction_under_knob():
    """`generate_single_day` debe construir `day_system_instruction` que
    contenga `DAY_GENERATOR_SYSTEM_PROMPT` + el schema JSON cuando el knob
    está habilitado, y dejarlo en `None` cuando knob=False (legacy: schema
    inline en streaming_prompt)."""
    text = _read_graph()
    # Anclar por la línea `schema_dict = SingleDayPlanModel.model_json_schema()`
    # que es única.
    region_match = re.search(
        r"schema_dict = SingleDayPlanModel\.model_json_schema\(\).*?(?=\n\s{8}@retry)",
        text,
        re.DOTALL,
    )
    assert region_match, "No encontré la región del schema injection en day_generator."
    region = region_match.group(0)
    assert "PROMPT_CACHE_SYSTEM_MESSAGE" in region, (
        "La construcción del schema del day_generator debe gatear por el knob."
    )
    assert "day_system_instruction = DAY_GENERATOR_SYSTEM_PROMPT" in region, (
        "Cuando knob=True, `day_system_instruction` debe incluir "
        "`DAY_GENERATOR_SYSTEM_PROMPT` (5K tokens cacheables)."
    )
    assert "day_system_instruction = None" in region, (
        "Cuando knob=False, `day_system_instruction = None` señala al messages "
        "builder que use el patrón legacy de single-HumanMessage."
    )


def test_day_generator_messages_builder_branches_on_system_instruction():
    """El builder de `messages` dentro de `invoke_day` debe ramificar:
      - si `day_system_instruction is not None` → `[SystemMessage, HumanMessage]`
      - si es None → `[HumanMessage(streaming_prompt)]` legacy."""
    text = _read_graph()
    # Buscar la construcción del messages list cerca del agent loop.
    m = re.search(
        r"if day_system_instruction is not None:\s*\n\s*messages\s*=\s*\[\s*\n\s*SystemMessage\(content=day_system_instruction\),\s*\n\s*HumanMessage\(content=streaming_prompt\),?\s*\n\s*\]\s*\n\s*else:\s*\n\s*messages\s*=\s*\[HumanMessage\(content=streaming_prompt\)\]",
        text,
    )
    assert m, (
        "P1-PROMPT-CACHE-SYSTEMMSG: el builder de `messages` dentro de "
        "`invoke_day` debe ramificar entre `[SystemMessage, HumanMessage]` "
        "(knob=True) y `[HumanMessage]` (legacy)."
    )


def test_day_generator_dynamic_text_excludes_static_when_knob_true():
    """Análogo al planner: el HumanMessage del day_generator NO debe duplicar
    `DAY_GENERATOR_SYSTEM_PROMPT` cuando el knob está habilitado."""
    text = _read_graph()
    assert "dynamic_day_prompt = (" in text, (
        "Falta la asignación de `dynamic_day_prompt` en day_generator."
    )
    m = re.search(
        r"if PROMPT_CACHE_SYSTEM_MESSAGE:\s*\n\s*prompt_text\s*=\s*dynamic_day_prompt\s*\n\s*else:\s*\n\s*prompt_text\s*=\s*dynamic_day_prompt\s*\+\s*DAY_GENERATOR_SYSTEM_PROMPT",
        text,
    )
    assert m, (
        "P1-PROMPT-CACHE-SYSTEMMSG: el day_generator debe usar el patrón "
        "`dynamic_day_prompt` + branch knob para evitar duplicar el system prompt "
        "en el HumanMessage."
    )


# ----- ROI cross-link a la instrumentación P1-COST-INSTRUMENTATION ------

def test_cost_instrumentation_captures_cached_tokens():
    """El gap #1 (P1-COST-INSTRUMENTATION) debe capturar `cached_tokens` —
    sin esta columna no se puede medir el ROI de este P-fix (cache hit rate
    = cached_tokens / input_tokens). El test ancla que la integración existe;
    P1-COST-INSTRUMENTATION es el dueño del schema, este test solo verifica
    que el helper `_emit_llm_usage_event_best_effort` extrae cached_tokens."""
    text = _read_graph()
    m = re.search(
        r"def _emit_llm_usage_event_best_effort.*?^def\s",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert m, "No encontré `_emit_llm_usage_event_best_effort`."
    body = m.group(0)
    assert "cached_tokens" in body and (
        "cache_read" in body or "cached_content_token_count" in body
    ), (
        "Helper debe extraer cached_tokens del usage_metadata "
        "(LangChain `cache_read` o Gemini `cached_content_token_count`). "
        "Sin esto, no se puede medir el ROI de P1-PROMPT-CACHE-SYSTEMMSG."
    )
