"""[P3-COST-CUT-V2 · 2026-05-21] Bundle 3-en-1 de cost reduction sin perder
calidad. Hallazgos del audit del codebase post-corrección de pricing dict
(P3-PRICING-DICT-REFRESH); ahorro neto estimado ~$0.05/plan = ~$50/mes a
1000 plans/mes.

Fixes incluidos:

  #1 Cache fix del self-critique evaluator
     Pre-fix: el evaluator concatenaba role + criterios + datos-del-plan en
     un single string → `_safe_ainvoke(evaluator_llm, prompt_string, ...)` →
     cache miss garantizado aunque `PROMPT_CACHE_SYSTEM_MESSAGE=True`.
     Audit confirmó que planner + day_gen estaban OK; solo este nodo bug.
     Fix: split en {SystemMessage(static), HumanMessage(per-plan)}.
     Ahorro: ~$0.003/plan input cost + 10-15s latencia.

  #2 Inyección de tabla de nutrición al SystemMessage del day_gen
     Pre-fix: cada day_worker llamaba `consultar_nutricion` 3-4× = 9-14
     LLM roundtrips/plan. Audit reveló que `tools_nutrition.MOCK_NUTRITION_DB`
     es un dict de SOLO 15 ingredientes y ~50% de queries del LLM retornan
     "No se encontró" (gandules, yuca, guineo, yogurt griego no están).
     Fix: inyectar la tabla pre-computada en `_DAY_SYSTEM_INSTRUCTION_CACHED`
     + reglas de estimación para ingredientes no listados. Tool sigue
     disponible como fallback.
     Ahorro: ~$0.12-0.15/plan (eliminar la mayoría de tool roundtrips).

  #3 Raise default `MEALFIT_HEDGE_AFTER_BASE_S` 90 → 120
     Pre-fix: hedge fired a 90s; cuando primary gana, hedge task sigue
     corriendo + genera tokens output completos antes de que la cancel
     scheduled tome efecto → ~$0.014 desperdiciado por hedge.
     Hoy: 2/3 hedges fired con primary ganador → ~$0.028/plan desperdiciado.
     Fix: bump default a 120s — hedge fires solo cuando primary genuinamente
     atorado (>120s).
     Ahorro: ~$0.007-0.014/plan (depende de hedge fire rate).

Tests parser-based:
  - Cada fix tiene un test que valida su estructura.
  - Tests de regresión negativa para que un revert futuro falle.
  - Sanity tests para verificar que los path legacy (knob OFF) siguen funcionando.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).parent.parent
_GO_PY = _BACKEND / "graph_orchestrator.py"


# ---------------------------------------------------------------------------
# Sección 1 — #1 Self-critique evaluator cache fix
# ---------------------------------------------------------------------------


def test_evaluator_system_instruction_constant_exists():
    """`_CRITIQUE_EVALUATOR_SYSTEM_INSTRUCTION` debe estar definido a nivel
    módulo. Sin la constante, no hay parte cacheable del prompt."""
    src = _GO_PY.read_text(encoding="utf-8")
    assert "_CRITIQUE_EVALUATOR_SYSTEM_INSTRUCTION" in src, (
        "Constante `_CRITIQUE_EVALUATOR_SYSTEM_INSTRUCTION` no encontrada. "
        "Sin ella, el evaluator no separa SystemMessage cacheable del HumanMessage."
    )


def test_evaluator_uses_payload_list_when_cache_on():
    """El callsite del evaluator debe construir `evaluator_payload` como
    `[SystemMessage(...), HumanMessage(...)]` cuando `PROMPT_CACHE_SYSTEM_MESSAGE=True`.
    Patrón espejo del planner + day_gen."""
    src = _GO_PY.read_text(encoding="utf-8")
    m = re.search(
        r"if PROMPT_CACHE_SYSTEM_MESSAGE:\s*\n"
        r"\s+evaluator_payload\s*=\s*\[\s*\n"
        r"\s+SystemMessage\(content=_CRITIQUE_EVALUATOR_SYSTEM_INSTRUCTION\),\s*\n"
        r"\s+HumanMessage\(content=human_content\),\s*\n",
        src,
    )
    assert m is not None, (
        "Callsite del evaluator no usa el patrón `[SystemMessage, HumanMessage]` "
        "cuando cache ON. Verificar la rama `if PROMPT_CACHE_SYSTEM_MESSAGE:`."
    )


def test_evaluator_invoke_uses_payload_not_string():
    """`_safe_ainvoke(evaluator_llm, ...)` debe recibir `evaluator_payload`
    (que es list o str según knob), NO el `prompt` string legacy."""
    src = _GO_PY.read_text(encoding="utf-8")
    # Buscar la línea `_safe_ainvoke(evaluator_llm, ..., timeout=_eval_timeout)`
    m = re.search(
        r"_safe_ainvoke\(\s*\n?\s*evaluator_llm,\s*(\w+),\s*timeout=_eval_timeout",
        src,
    )
    assert m is not None, "Callsite `_safe_ainvoke(evaluator_llm, ...)` no encontrado."
    arg = m.group(1)
    assert arg == "evaluator_payload", (
        f"`_safe_ainvoke` del evaluator recibe `{arg}` — debe ser `evaluator_payload` "
        f"(la variable que conmuta entre lista cacheable y string legacy según knob)."
    )


# ---------------------------------------------------------------------------
# Sección 2 — #2 Nutrition lookup table injection
# ---------------------------------------------------------------------------


def test_nutrition_lookup_helper_exists():
    """`_build_nutrition_lookup_instruction()` debe existir + ser invocado al
    import para byte-equivalence del SystemMessage."""
    src = _GO_PY.read_text(encoding="utf-8")
    assert "def _build_nutrition_lookup_instruction" in src, (
        "Helper `_build_nutrition_lookup_instruction()` no encontrado."
    )
    assert "_NUTRITION_LOOKUP_INSTRUCTION = _build_nutrition_lookup_instruction()" in src, (
        "Constante `_NUTRITION_LOOKUP_INSTRUCTION` no se inicializa al import."
    )


def test_day_system_instruction_includes_nutrition_table():
    """`_DAY_SYSTEM_INSTRUCTION_CACHED` debe concatenar el bloque de nutrición
    PRE-COMPUTADA después del schema. Sin esto, el LLM no ve la tabla y sigue
    invocando tool roundtrips innecesarios."""
    src = _GO_PY.read_text(encoding="utf-8")
    m = re.search(
        r"_DAY_SYSTEM_INSTRUCTION_CACHED\s*=\s*\(\s*\n"
        r"\s+DAY_GENERATOR_SYSTEM_PROMPT\s*\n"
        r"\s+\+\s+_DAY_SCHEMA_INSTRUCTION\s*\n"
        r"\s+\+\s+_NUTRITION_LOOKUP_INSTRUCTION\s*\n"
        r"\s*\)",
        src,
    )
    assert m is not None, (
        "`_DAY_SYSTEM_INSTRUCTION_CACHED` debe ser concatenación de "
        "(DAY_GENERATOR_SYSTEM_PROMPT + _DAY_SCHEMA_INSTRUCTION + _NUTRITION_LOOKUP_INSTRUCTION)."
    )


def test_nutrition_helper_imports_mock_db():
    """El helper debe importar `MOCK_NUTRITION_DB` desde `tools_nutrition`.
    Si el dict se renombra, el helper falla al inicializar el módulo —
    fail-fast preferible a SystemMessage con tabla vacía silenciosa."""
    src = _GO_PY.read_text(encoding="utf-8")
    assert "from tools_nutrition import MOCK_NUTRITION_DB" in src, (
        "Helper no importa `MOCK_NUTRITION_DB` desde tools_nutrition.py. "
        "Si el dict se renombra/borra, el SystemMessage de nutrición quedaría vacío."
    )


# ---------------------------------------------------------------------------
# Sección 3 — #3 Hedge threshold default raise 90 → 120
# ---------------------------------------------------------------------------


def test_hedge_default_is_120():
    """Default de `MEALFIT_HEDGE_AFTER_BASE_S` debe ser 120s. Pre-fix-V2 era
    90s (P1-HEDGE-THRESHOLD-RAISE, mismo día). Subir reduce hedges
    desperdiciados (primary gana) sin necesidad de redeploy: el operador
    puede setear `MEALFIT_HEDGE_AFTER_BASE_S=90` o `=45` para rollback."""
    src = _GO_PY.read_text(encoding="utf-8")
    assert (
        '_env_float("MEALFIT_HEDGE_AFTER_BASE_S",          120.0)' in src
        or '_env_float("MEALFIT_HEDGE_AFTER_BASE_S", 120.0)' in src
    ), (
        "Default de `MEALFIT_HEDGE_AFTER_BASE_S` debe ser 120.0. Rollback "
        "via env var sin redeploy."
    )


def test_hedge_default_not_stale_90():
    """Verificación negativa: el default 90s (P1-HEDGE-THRESHOLD-RAISE
    pre-fix-V2) no debe permanecer en el código. Si alguien revertió V2,
    este test cae."""
    src = _GO_PY.read_text(encoding="utf-8")
    stale = re.search(
        r'_env_float\(\s*"MEALFIT_HEDGE_AFTER_BASE_S"\s*,\s*90\.0\s*\)',
        src,
    )
    assert stale is None, (
        "Default `MEALFIT_HEDGE_AFTER_BASE_S=90.0` detectado. Eso es revert "
        "de P3-COST-CUT-V2; el nuevo default debe ser 120.0."
    )


# ---------------------------------------------------------------------------
# Sección 4 — Marker presente
# ---------------------------------------------------------------------------


def test_marker_present_in_graph_orchestrator():
    """Marker `P3-COST-CUT-V2` debe estar como tooltip-anchor en
    graph_orchestrator.py. Cubre los 3 fixes del bundle."""
    src = _GO_PY.read_text(encoding="utf-8")
    occurrences = src.count("P3-COST-CUT-V2")
    assert occurrences >= 3, (
        f"Marker `P3-COST-CUT-V2` aparece {occurrences} veces. Esperado ≥3 "
        f"(uno por fix: nutrition table + evaluator cache + hedge raise)."
    )


# ---------------------------------------------------------------------------
# Sección 5 — Sanity: legacy path (cache OFF) sigue funcionando
# ---------------------------------------------------------------------------


def test_evaluator_legacy_path_concatenates_string():
    """Cuando `PROMPT_CACHE_SYSTEM_MESSAGE=False`, el evaluator debe caer al
    path legacy concatenando string. Esto preserva la opción de operador de
    deshabilitar caching sin romper el evaluator."""
    src = _GO_PY.read_text(encoding="utf-8")
    m = re.search(
        r"else:\s*\n"
        r"\s+# Legacy path[^\n]*\n"
        r"\s+evaluator_payload\s*=\s*\(\s*\n"
        r"\s+_CRITIQUE_EVALUATOR_SYSTEM_INSTRUCTION\s*\+\s*",
        src,
    )
    assert m is not None, (
        "Path legacy (else branch tras `if PROMPT_CACHE_SYSTEM_MESSAGE:`) no "
        "encontrado. Esto rompe rollback a `PROMPT_CACHE_SYSTEM_MESSAGE=False`."
    )


def test_nutrition_helper_includes_estimation_rules_for_dominican_ingredients():
    """El helper debe incluir reglas de estimación para ingredientes
    dominicanos NO en MOCK_NUTRITION_DB (gandules, yuca, guineo, yogurt
    griego, mango). Sin estas reglas, el LLM seguirá invocando el tool
    para esos casos."""
    src = _GO_PY.read_text(encoding="utf-8")
    # Buscar el body del helper
    m = re.search(
        r"def _build_nutrition_lookup_instruction\(\)[\s\S]+?(?=\n_NUTRITION_LOOKUP_INSTRUCTION\s*=)",
        src,
    )
    assert m is not None, "Cuerpo del helper no encontrado."
    body = m.group(0)
    # Verificar que menciona al menos 3 de los 5 ingredientes dominicanos clave
    dominican_keywords = ["gandules", "yuca", "guineo", "yogurt", "mango"]
    found = [kw for kw in dominican_keywords if kw in body.lower()]
    assert len(found) >= 3, (
        f"Helper menciona solo {found} de {dominican_keywords}. Esperado ≥3 "
        f"para reducir tool roundtrips de ingredientes dominicanos no-en-dict."
    )
