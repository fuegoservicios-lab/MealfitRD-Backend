"""[P1-5] Lecciones lifetime se inyectan al prompt aunque el rolling window esté limpio.

Antes del fix:
    `build_chunk_lessons_context` (prompts/plan_generator.py) gateaba los bullets
    en métricas RECIENTES:
      - `rejected_meals_that_reappeared` solo si `rej_viol > 0`.
      - `repeated_meal_names` solo si `repeat_pct > 15.0`.
      - `permanent_meal_blocklist` no se leía nunca.

    Si chunk N+1 corría con rolling window limpio (chunk N sin violations) pero
    `_lifetime_lessons_summary` acumulaba "pollo rechazado hace 5 chunks" o
    `permanent_meal_blocklist` con meals repetidos en >=2 chunks pasados, el
    LLM NO recibía ninguna de esas señales — y podía regenerar comidas
    sistemáticamente bloqueadas.

Después del fix:
    1. `form_data["_chunk_lessons"]` expone campos lifetime explícitos:
       `lifetime_top_rejection_hits`, `lifetime_top_repeated_meal_names`,
       `lifetime_top_repeated_bases`.
    2. `build_chunk_lessons_context` emite 3 bullets nuevos:
       - "BLOQUEO PERMANENTE (ACUMULADO HISTÓRICO)" — siempre que
         `permanent_meal_blocklist` tenga items.
       - "RECHAZOS HISTÓRICOS DEL USUARIO (acumulado)" — cuando
         `lifetime_top_rejection_hits` no vacío Y `rej_viol == 0` (recent limpio).
       - "PLATOS HISTÓRICOS YA GENERADOS (acumulado)" — cuando
         `lifetime_top_repeated_meal_names` no vacío Y `repeat_pct == 0` Y NO
         hay `permanent_meal_blocklist` (sino se duplica con el bullet anterior).
"""
import pytest

from prompts.plan_generator import build_chunk_lessons_context


def test_permanent_blocklist_always_emitted():
    """`permanent_meal_blocklist` viene de lifetime_summary y representa meals
    repetidos en >=2 chunks distintos. Es señal sistémica — siempre debe
    aparecer en el prompt, independiente de métricas recientes."""
    chunk_lessons = {
        "rejection_violations": 0,
        "repeat_pct": 0,
        "ingredient_base_repeat_pct": 0,
        "permanent_meal_blocklist": ["Pollo Asado al Limón", "Salmón Plancha"],
        "lifetime_top_rejection_hits": [],
        "lifetime_top_repeated_meal_names": [],
        "rejected_meals_that_reappeared": [],
        "repeated_meal_names": [],
    }
    ctx = build_chunk_lessons_context(chunk_lessons)
    assert "BLOQUEO PERMANENTE" in ctx, (
        "Bullet de permanent_meal_blocklist debe emitirse siempre que el field "
        "tenga items, sin gate de rej_viol/repeat_pct."
    )
    assert "Pollo Asado al Limón" in ctx
    assert "Salmón Plancha" in ctx


def test_lifetime_rejection_hits_emitted_when_recent_clean():
    """Si chunk anterior no tuvo violations (rej_viol == 0) pero lifetime
    acumuló rejections, el LLM debe verlos. Antes el bullet entero quedaba
    suprimido por el gate `rej_viol > 0`."""
    chunk_lessons = {
        "rejection_violations": 0,  # recent clean
        "repeat_pct": 0,
        "ingredient_base_repeat_pct": 0,
        "permanent_meal_blocklist": [],
        "lifetime_top_rejection_hits": ["camarones", "calabaza"],
        "lifetime_top_repeated_meal_names": [],
        "rejected_meals_that_reappeared": [],
        "repeated_meal_names": [],
    }
    ctx = build_chunk_lessons_context(chunk_lessons)
    assert "RECHAZOS HISTÓRICOS" in ctx, (
        "Cuando recent está limpio pero lifetime tiene rejection_hits, el "
        "LLM debe ver el bullet histórico — sin esto, regenera meals "
        "rechazados hace meses."
    )
    assert "camarones" in ctx
    assert "calabaza" in ctx


def test_lifetime_repeated_meals_emitted_when_recent_clean():
    """Si recent repeat_pct==0 pero lifetime tiene meals históricos generados
    en chunks pasados, el LLM debe verlos para inventar nombres distintos."""
    chunk_lessons = {
        "rejection_violations": 0,
        "repeat_pct": 0,  # recent clean
        "ingredient_base_repeat_pct": 0,
        "permanent_meal_blocklist": [],  # vacío para no duplicar
        "lifetime_top_rejection_hits": [],
        "lifetime_top_repeated_meal_names": ["Wraps de Pollo", "Bowl de Quinoa"],
        "rejected_meals_that_reappeared": [],
        "repeated_meal_names": [],
    }
    ctx = build_chunk_lessons_context(chunk_lessons)
    assert "PLATOS HISTÓRICOS" in ctx
    assert "Wraps de Pollo" in ctx
    assert "Bowl de Quinoa" in ctx


def test_lifetime_repeated_meals_NOT_duplicated_when_permanent_blocklist_present():
    """Si `permanent_meal_blocklist` tiene items, el bullet "PLATOS HISTÓRICOS"
    se suprime para no duplicar señales (los meals con >=2 chunks ya están
    cubiertos por el bullet de bloqueo permanente)."""
    chunk_lessons = {
        "rejection_violations": 0,
        "repeat_pct": 0,
        "ingredient_base_repeat_pct": 0,
        "permanent_meal_blocklist": ["Pollo Asado"],
        "lifetime_top_rejection_hits": [],
        "lifetime_top_repeated_meal_names": ["Pollo Asado", "Salmón"],  # overlap intencional
        "rejected_meals_that_reappeared": [],
        "repeated_meal_names": [],
    }
    ctx = build_chunk_lessons_context(chunk_lessons)
    # BLOQUEO PERMANENTE sí.
    assert "BLOQUEO PERMANENTE" in ctx
    assert "Pollo Asado" in ctx
    # PLATOS HISTÓRICOS suprimido para evitar duplicación.
    assert "PLATOS HISTÓRICOS" not in ctx


def test_recent_violations_still_use_original_bullet():
    """Cuando recent SÍ tiene violations, el bullet original
    (`rejected_meals_that_reappeared` con wording "RECHAZADOS reaparecieron")
    sigue funcionando — no rompemos el path normal."""
    chunk_lessons = {
        "rejection_violations": 2,  # recent dirty
        "repeat_pct": 0,
        "ingredient_base_repeat_pct": 0,
        "permanent_meal_blocklist": [],
        "lifetime_top_rejection_hits": [],
        "lifetime_top_repeated_meal_names": [],
        "rejected_meals_that_reappeared": ["camarones rebozados"],
        "repeated_meal_names": [],
    }
    ctx = build_chunk_lessons_context(chunk_lessons)
    assert "RECHAZADOS reaparecieron" in ctx
    assert "camarones rebozados" in ctx
    assert "RECHAZOS HISTÓRICOS" not in ctx, (
        "Cuando recent SÍ tiene rej_viol, no emitimos el bullet histórico — "
        "sería redundante (rejected_meals_that_reappeared ya une recent + lifetime)."
    )


def test_no_bullets_when_everything_empty():
    """Sin lecciones recientes ni lifetime, el contexto es vacío.
    Caso típico: chunk 1 de plan nuevo sin historial heredado."""
    chunk_lessons = {
        "rejection_violations": 0,
        "repeat_pct": 0,
        "ingredient_base_repeat_pct": 0,
        "permanent_meal_blocklist": [],
        "lifetime_top_rejection_hits": [],
        "lifetime_top_repeated_meal_names": [],
        "rejected_meals_that_reappeared": [],
        "repeated_meal_names": [],
    }
    ctx = build_chunk_lessons_context(chunk_lessons)
    assert ctx == "", "Sin lecciones, no debe emitirse contexto vacío con header."


def test_form_data_has_lifetime_fields_explicit():
    """Regression guard del DATA LAYER: `_chunk_lessons` en form_data debe
    contener los 3 campos lifetime separados (no solo en `repeated_meal_names`
    union). Si por refactor desaparecieran, el prompt builder no podría leerlos.
    """
    import inspect, cron_tasks
    src = inspect.getsource(cron_tasks)
    # Los 3 campos lifetime explícitos deben aparecer en la construcción del dict.
    assert '"lifetime_top_rejection_hits"' in src
    assert '"lifetime_top_repeated_meal_names"' in src
    assert '"lifetime_top_repeated_bases"' in src


def test_long_blocklist_truncated_in_prompt():
    """`permanent_meal_blocklist` puede tener hasta 50 items en form_data, pero
    el prompt cap es 8 (suficiente para la directiva sin saturar el LLM con
    listas masivas que diluyen prioridad)."""
    blocklist = [f"Plato-{i}" for i in range(20)]
    chunk_lessons = {
        "rejection_violations": 0,
        "repeat_pct": 0,
        "ingredient_base_repeat_pct": 0,
        "permanent_meal_blocklist": blocklist,
        "lifetime_top_rejection_hits": [],
        "lifetime_top_repeated_meal_names": [],
        "rejected_meals_that_reappeared": [],
        "repeated_meal_names": [],
    }
    ctx = build_chunk_lessons_context(chunk_lessons)
    assert "Plato-0" in ctx
    assert "Plato-7" in ctx
    # 8º item es índice 7; del 8 en adelante NO debe aparecer.
    assert "Plato-8" not in ctx, (
        "Prompt cap de permanent_meal_blocklist debe ser 8 para no inflar "
        "el contexto. Si necesitas más, ajusta el slice [:8] en plan_generator.py."
    )


def test_lifetime_header_marks_aggregation_origin():
    """Cuando `is_lifetime_aggregated` es True, el header del bloque cambia
    para indicar al LLM que las métricas son acumuladas (no del último chunk)."""
    chunk_lessons = {
        "rejection_violations": 1,
        "repeat_pct": 0,
        "ingredient_base_repeat_pct": 0,
        "permanent_meal_blocklist": [],
        "lifetime_top_rejection_hits": [],
        "lifetime_top_repeated_meal_names": [],
        "rejected_meals_that_reappeared": ["pollo"],
        "repeated_meal_names": [],
        "is_lifetime_aggregated": True,
    }
    ctx = build_chunk_lessons_context(chunk_lessons)
    assert "PATRONES CRÍTICOS Y LECCIONES ACUMULADAS" in ctx, (
        "is_lifetime_aggregated=True debe disparar el header 'TODA LA VIDA "
        "DEL PLAN' para que el LLM contextualice las señales."
    )
