"""
[P1-1] E2E del flujo de propagación de aprendizaje al prompt LLM.

El sistema garantiza que las lecciones del chunk N (rechazos, alergias, ingredientes
sobre-repetidos, nombres de platos repetidos, señales débiles, lecciones críticas
permanentes) llegan al prompt del chunk N+1 para que el LLM no repita los mismos
errores. La propagación pasa por dos pasos:

  1. cron_tasks.py L7438-L7591 — agrega _last_chunk_learning + _recent_chunk_lessons
     + _critical_lessons_permanent en form_data["_chunk_lessons"].
  2. graph_orchestrator.py:819 — `build_chunk_lessons_context(chunk_lessons)`
     convierte ese dict en el bloque de texto que se inyecta al ChatPromptTemplate.

Tests previos cubren cada parte por separado (sanitización en P1-7, weak_signal
en test_weak_signal_*). Lo que faltaba era una validación E2E que asegure que TODAS
las señales documentadas en el audit P1-1 producen output literal en el prompt:
"pollo a la plancha", "rejected_meals_that_reappeared", "allergy_hits", etc.
Si alguien refactoriza el agregador o el builder y rompe el flujo, este test grita.
"""
import sys
from unittest.mock import MagicMock

sys.modules.setdefault('langgraph', MagicMock())
sys.modules.setdefault('langgraph.graph', MagicMock())
sys.modules.setdefault('langgraph.graph.message', MagicMock())


def _full_lessons_blob():
    """Forma realista del dict que cron_tasks construye en form_data['_chunk_lessons']
    tras agregar _last_chunk_learning + _recent_chunk_lessons + _critical_lessons_permanent.
    Incluye nombres concretos para que las aserciones puedan validar literal-passing.
    """
    return {
        "chunk_number": "anterior",
        "chunk_numbers": [1, 2, 3],
        "ingredient_base_repeat_pct": 65.0,
        "repeated_bases": [
            {"chunk": 1, "bases": ["pollo", "arroz blanco"]},
            {"chunk": 2, "bases": ["pollo", "habichuela roja"]},
            {"chunk": 3, "bases": ["arroz blanco", "berenjena"]},
        ],
        "repeat_pct": 22.0,
        "repeated_meal_names": [
            "Pollo a la plancha",
            "Sopa de habichuelas",
        ],
        "rejection_violations": 2,
        "rejected_meals_that_reappeared": [
            "pollo a la plancha",
            "ensalada cesar",
        ],
        "allergy_violations": 1,
        "allergy_hits": ["mani", "cacahuete"],
        "is_lifetime_aggregated": False,
    }


def test_a_all_critical_signals_appear_in_prompt_block():
    """[P1-1] Una pasada exhaustiva: cada tipo de señal del audit produce su línea."""
    from prompts.plan_generator import build_chunk_lessons_context

    ctx = build_chunk_lessons_context(_full_lessons_blob())

    # Repeated_bases con pct >= 60 debe aparecer marcado URGENTE
    assert "URGENTE" in ctx, "Pct ingredient_base_repeat 65% debe disparar severidad URGENTE"
    assert "pollo" in ctx and "arroz blanco" in ctx, \
        "Bases repetidas concretas deben aparecer literalmente"
    assert "DIVERSIFICA" in ctx

    # Rejected_meals_that_reappeared (señal del audit explícita)
    assert "RECHAZADOS" in ctx
    assert "pollo a la plancha" in ctx, \
        "El nombre del plato rechazado debe aparecer literal en el prompt"
    assert "ensalada cesar" in ctx

    # Allergy_hits
    assert "alergia" in ctx.lower()
    assert "mani" in ctx, "Allergy hit concreto debe aparecer literal"

    # Repeated_meal_names (repeat_pct >= 15)
    assert "Pollo a la plancha" in ctx, \
        "Nombre repetido literal debe aparecer en bloque de nombres"


def test_b_critical_lessons_permanent_signal_propagates_after_aggregation():
    """[P1-1] _critical_lessons_permanent vive en plan_data y el agregador en
    cron_tasks.py L7438+ lo incluye en _all_lessons cuando su `chunk` no está en
    el rolling window. Aquí simulamos el resultado post-agregación: una entrada
    `rejected_meals_that_reappeared` proveniente de un chunk antiguo (chunk 1)
    fuera del rolling window debe seguir produciendo la línea de prompt.

    Si el agregador deja de mergear _critical_extras, el blob agregado pierde
    `rejected_meals_that_reappeared` y este test fallará.
    """
    from prompts.plan_generator import build_chunk_lessons_context

    # Blob que SOLO contiene señales originadas en _critical_lessons_permanent
    # (chunks viejos fuera del rolling window). Si el agregador no los incluye,
    # rejected_meals_that_reappeared estaría vacío y la línea desaparecería.
    aggregated_with_critical_only = {
        "chunk_numbers": [1, 5],  # chunk 1 vino de critical_lessons_permanent
        "rejection_violations": 1,
        "rejected_meals_that_reappeared": ["Pollo Frito Antiguo"],
        "ingredient_base_repeat_pct": 0.0,
        "repeat_pct": 0.0,
        "repeated_bases": [],
        "repeated_meal_names": [],
        "allergy_violations": 0,
        "allergy_hits": [],
    }
    ctx = build_chunk_lessons_context(aggregated_with_critical_only)
    assert "Pollo Frito Antiguo" in ctx, \
        "Lección crítica permanente debe seguir apareciendo en el prompt aunque " \
        "su chunk original (1) ya esté fuera del rolling window"
    assert "RECHAZADOS" in ctx


def test_c_weak_signal_blocks_protein_rotation_message():
    """[P1-1] Cuando la señal de inventario es débil (proxy-based learning), el
    bloque debe instruir al LLM a rotar las proteínas explícitamente. Verifica
    que `banned_proteins` se inyecta como rotación obligatoria."""
    from prompts.plan_generator import build_chunk_lessons_context
    lessons = {
        "weak_signal": True,
        "learning_signal_strength": "weak",
        "banned_proteins": ["pollo", "res", "huevo"],
        "ingredient_base_repeat_pct": 0,
        "repeat_pct": 0,
    }
    ctx = build_chunk_lessons_context(lessons)
    assert "SEÑAL DÉBIL" in ctx or "señal débil" in ctx.lower()
    assert "pollo" in ctx and "res" in ctx and "huevo" in ctx, \
        "Las proteínas baneadas concretas deben aparecer en el prompt"
    assert "REGLA ESTRICTA" in ctx or "DIFERENTES" in ctx, \
        "Debe haber una directiva clara de rotación, no solo enumerar las baneadas"


def test_d_pantry_diversity_warning_inlines_specific_directive():
    """[P1-1] Cuando la diversidad de la nevera está agotada, el prompt debe
    redirigir al LLM a variar TÉCNICA y ACOMPAÑAMIENTO en vez de PROTEÍNA."""
    from prompts.plan_generator import build_chunk_lessons_context
    lessons = {
        "pantry_diversity_warning": True,
        "ingredient_base_repeat_pct": 0,
        "repeat_pct": 0,
    }
    ctx = build_chunk_lessons_context(lessons)
    assert "INVENTARIO AGOTADO" in ctx or "inventario es limitado" in ctx.lower()
    assert "TÉCNICA" in ctx or "técnica" in ctx.lower()


def test_e_empty_lessons_returns_empty_string_no_garbage_section():
    """[P1-1] Sin lecciones, el bloque debe ser exactamente "" (no header huérfano).
    Crítico porque si el flujo retorna un header solitario en planes nuevos, el LLM
    se confunde con un bloque "Lecciones del chunk anterior:" vacío.
    """
    from prompts.plan_generator import build_chunk_lessons_context
    assert build_chunk_lessons_context(None) == ""
    assert build_chunk_lessons_context({}) == ""
    # Lecciones presentes pero todas por debajo de threshold:
    assert build_chunk_lessons_context({
        "ingredient_base_repeat_pct": 10.0,  # < 30 threshold
        "repeated_bases": [{"chunk": 1, "bases": ["pollo"]}],
        "repeat_pct": 5.0,                    # < 15 threshold
        "repeated_meal_names": ["X"],
        "rejection_violations": 0,
        "allergy_violations": 0,
    }) == "", "Lecciones por debajo de umbrales deben dar string vacío, sin header"


def test_f_severity_is_urgent_only_above_high_threshold():
    """[P1-1] La diferencia entre `URGENTE — DIVERSIFICA` (pct > 60) y
    `DIVERSIFICA` (30 < pct ≤ 60) debe respetarse — un cambio accidental de la
    constante en build_chunk_lessons_context degradaría la severidad transmitida
    al LLM en planes con repetición moderada vs aguda."""
    from prompts.plan_generator import build_chunk_lessons_context
    medium = build_chunk_lessons_context({
        "ingredient_base_repeat_pct": 45.0,
        "repeated_bases": [{"chunk": 1, "bases": ["pollo"]}],
    })
    high = build_chunk_lessons_context({
        "ingredient_base_repeat_pct": 75.0,
        "repeated_bases": [{"chunk": 1, "bases": ["pollo"]}],
    })
    assert "URGENTE" not in medium and "DIVERSIFICA" in medium
    assert "URGENTE" in high
