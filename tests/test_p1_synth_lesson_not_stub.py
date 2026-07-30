"""[P1-SYNTH-LESSON-NOT-STUB · 2026-07-30] Una lección sintetizada no es un stub.

Incidente en producción (30 jul 2026): dos usuarios reales con la semana 2 de su plan
pausada en `pending_user_action` por el guard P0-B, con `synth=26/total=4 ratio=650%`.

La cadena, medida en Neon:

  1. La semana 1 nace de la petición inicial, NO de la cola: los 17 planes de prod
     tienen `min(week_number)=2` en `plan_chunk_queue`. El chunk 2 de CUALQUIER plan
     busca las lecciones de `prev_week=1` en la cola, no las encuentra nunca, y cae
     al último recurso `_synthesize_last_chunk_learning_from_plan_days`.
  2. Esa síntesis estampa `metrics_unavailable=True` — que significa "los contadores
     numéricos no son reales, no los leas como '0 violaciones'", una pista de
     RENDERIZADO para el prompt. `_is_lesson_stub` la leía como declaración de
     AUSENCIA y devolvía True. O sea que la lección recién persistida volvía a
     leerse como stub en el intento siguiente → re-síntesis → otra fila de
     telemetría. 843 eventos en la semana 2 de 6 planes contra 18 en TODAS las
     demás semanas juntas.
  3. El guard que consume esa telemetría dividía EVENTOS entre FILAS de chunk, y
     además su propia pausa (`pending_user_action`) sacaba al chunk del denominador.

Los tests de round-trip (productor → consumidor) son los que importan: el enlace roto
estaba justo entre esas dos funciones, y un dict forjado a mano no lo habría cubierto.
"""
import re
from pathlib import Path

import pytest

CRON = Path(__file__).resolve().parent.parent / "cron_tasks.py"


def _plan_data_con_dias(target_week: int = 1) -> dict:
    """`plan_data` mínimo pero realista: días tagueados con su semana, comidas con
    nombre e ingredientes. Es lo que la síntesis lee de `meal_plans.plan_data`."""
    return {
        "days": [
            {
                "week_number": target_week,
                "day_number": i,
                "meals": [
                    {
                        "name": f"Pollo guisado con arroz día {i}",
                        "ingredients": ["Pollo", "Arroz", "Cebolla"],
                    },
                    {
                        "name": f"Ensalada de atún día {i}",
                        "ingredients": ["Atún", "Lechuga", "Tomate"],
                    },
                ],
            }
            for i in range(1, 8)
        ]
    }


# ---------------------------------------------------------------- round-trip

def test_leccion_sintetizada_no_es_stub():
    """El defecto exacto: lo que la síntesis produce, el detector lo rechazaba.

    No forjamos el dict — lo pedimos al productor real. Si mañana la síntesis cambia
    de representación (otra clave en vez de `metrics_unavailable`), este test sigue
    midiendo el contrato de verdad en lugar de un valor interno inventado.
    """
    from cron_tasks import (
        _is_lesson_stub,
        _synthesize_last_chunk_learning_from_plan_days,
    )

    leccion = _synthesize_last_chunk_learning_from_plan_days(
        "plan-de-prueba", 1, _plan_data_con_dias(1), user_id=None
    )
    assert leccion is not None, "la síntesis debería producir lección con estos días"
    assert leccion["metrics_unavailable"] is True, (
        "la síntesis SIGUE marcando los contadores como no-reales — es una pista de "
        "renderizado para el prompt y debe conservarse"
    )
    assert leccion["repeated_bases"], "debería traer bases extraídas de los días"

    assert _is_lesson_stub(leccion) is False, (
        "una lección con muestras reales NO es un stub. Si vuelve a serlo, el chunk "
        "re-sintetiza en cada intento: bucle sin salida + una fila de telemetría por "
        "intento que dispara el guard P0-B."
    )


def test_sintesis_persistida_no_vuelve_a_pedir_rebuild():
    """El bucle completo: la lección sintetizada, releída como si viniera de
    `plan_data._last_chunk_learning`, ya no debe pedir rebuild para la MISMA semana.

    Reproduce las dos condiciones del gate `_p03_needs_rebuild` en el worker:
    `_is_lesson_stub(existente)` y `existente["chunk"] != week_number - 1`.
    """
    from cron_tasks import (
        _is_lesson_stub,
        _synthesize_last_chunk_learning_from_plan_days,
    )

    week_number = 2                      # el chunk que se estaba pausando en prod
    target_week = week_number - 1        # = 1, la semana que nunca está en la cola

    persistida = _synthesize_last_chunk_learning_from_plan_days(
        "plan-de-prueba", target_week, _plan_data_con_dias(target_week), user_id=None
    )
    assert persistida is not None

    needs_rebuild = (
        _is_lesson_stub(persistida) or persistida.get("chunk") != target_week
    )
    assert needs_rebuild is False, (
        f"tras sintetizar y persistir, el intento siguiente del chunk {week_number} "
        f"volvía a pedir rebuild → re-síntesis infinita"
    )


# ------------------------------------------- el detector sigue detectando stubs

def test_marca_pelada_sin_muestras_sigue_siendo_stub():
    """El discriminante real no se debilita: sin señal numérica NI muestras, stub.

    Es el caso que `test_p0_3_is_lesson_stub_detects_metrics_unavailable_flag` ancla
    desde P0-3, y sigue vivo — lo que cambió es que la marca por sí sola ya no basta.
    """
    from cron_tasks import _is_lesson_stub

    assert _is_lesson_stub({"chunk": 2, "metrics_unavailable": True}) is True
    assert _is_lesson_stub({"chunk": 2}) is True
    assert _is_lesson_stub({}) is True
    assert _is_lesson_stub(None) is True


def test_leccion_sintetizada_vacia_sigue_siendo_stub():
    """Días sin comidas aprovechables ⇒ sin muestras ⇒ stub legítimo."""
    from cron_tasks import _is_lesson_stub

    assert _is_lesson_stub({
        "chunk": 1,
        "metrics_unavailable": True,
        "repeat_pct": 0,
        "ingredient_base_repeat_pct": 0,
        "rejection_violations": 0,
        "allergy_violations": 0,
        "fatigued_violations": 0,
        "repeated_bases": [],
        "repeated_meal_names": [],
        "rejected_meals_that_reappeared": [],
        "allergy_hits": [],
    }) is True


# --------------------------------------------- paridad numerador / denominador

def _cuerpo_desde_anchor(anchor: str, hasta: str = "fetch_one=True") -> str:
    """Recorta desde el tooltip-anchor hasta el cierre de la query.

    Anclado al ORDEN RELATIVO (anchor → cierre), no a una ventana de bytes fija:
    las ventanas fijas caducan solas en cuanto alguien añade una línea encima.
    """
    src = CRON.read_text(encoding="utf-8")
    i = src.find(anchor)
    assert i != -1, f"desapareció el tooltip-anchor {anchor!r} de cron_tasks.py"
    j = src.find(hasta, i)
    assert j != -1, f"no se encontró el cierre {hasta!r} tras {anchor!r}"
    return src[i:j]


@pytest.mark.parametrize("anchor", [
    "# tooltip-anchor: [P1-SYNTH-LESSON-NOT-STUB] paridad numerador/denominador",
    "# tooltip-anchor: [P1-SYNTH-LESSON-NOT-STUB] paridad flota",
])
def test_numerador_cuenta_chunks_distintos_no_eventos(anchor):
    """`chunk_lesson_telemetry` es append-only por INTENTO.

    Contar filas mete los reintentos en el numerador mientras el denominador cuenta
    chunks: en prod dio synth=26 sobre 4 chunks distintos ⇒ 650%. Un ratio >100% bajo
    la semántica "porcentaje de chunks" es imposible, así que era prueba estructural
    de que las dos mitades no hablaban del mismo conjunto.
    """
    cuerpo = _cuerpo_desde_anchor(anchor)
    assert "chunk_lesson_telemetry" in cuerpo, "el recorte no llegó al numerador"
    assert re.search(
        r"COUNT\(DISTINCT\s*\(\s*meal_plan_id\s*,\s*week_number\s*\)\)", cuerpo
    ), (
        "el numerador debe contar chunks DISTINTOS (meal_plan_id, week_number), no "
        "filas de telemetría — cada reintento del mismo chunk añade una fila"
    )


@pytest.mark.parametrize("anchor", [
    "# tooltip-anchor: [P1-SYNTH-LESSON-NOT-STUB] paridad numerador/denominador",
    "# tooltip-anchor: [P1-SYNTH-LESSON-NOT-STUB] paridad flota",
])
def test_denominador_incluye_los_chunks_que_el_guard_pausa(anchor):
    """Sin `pending_user_action`, pausar un chunk lo sacaba del denominador y
    empeoraba el ratio de la evaluación siguiente: el guard se realimentaba a sí mismo."""
    cuerpo = _cuerpo_desde_anchor(anchor)
    assert "plan_chunk_queue" in cuerpo, "el recorte no llegó al denominador"
    m = re.search(r"status\s+IN\s*\(([^)]*)\)", cuerpo)
    assert m, "no se encontró el filtro de status del denominador"
    estados = m.group(1)
    assert "pending_user_action" in estados, (
        "el denominador debe incluir `pending_user_action`: son chunks YA procesados "
        f"que este mismo guard pausó. Encontrado: {estados.strip()}"
    )
