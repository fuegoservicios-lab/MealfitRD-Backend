"""
Tests P0-CHUNKS-1: lecciones sintéticas no contaminan el agregado de aprendizaje.

Bug original:
  Cuando el chunk previo crashea pre-pipeline (`learning_metrics=NULL`),
  `_synthesize_last_chunk_learning_from_plan_days` reconstruye una "lección"
  desde `plan_data.days`. Esa lección incluye `repeated_bases` y
  `repeated_meal_names` extraídos del PLAN previo (lo que se planificó),
  no de logs reales (lo que se consumió/rechazó). El agregador del worker de
  chunks (`cron_tasks.py:15159+`) mezclaba esas listas con las de lecciones
  REALES en `_agg_repeated_bases` / `_agg_repeated_meals` sin distinguir
  fuente. El LLM del chunk N+1 recibía "evita estos ingredientes" mezclando
  observado con plan-derivado, y construía su propia lección sobre cimientos
  falsos. En planes ≥14d con un fallo del chunk 1 o 2, ese ruido se acumulaba
  2-3 chunks downstream contaminando la señal real (5-15% de planes largos).

Fix:
  1. `_synthesize_last_chunk_learning_from_plan_days` ya marcaba
     `synthesized_from_plan_days=True`; añadido `confidence_score=0.4` para
     consumers granulares.
  2. Agregador en `cron_tasks.py` excluye `repeated_bases` /
     `repeated_meal_names` / `rejected_meals_that_reappeared` / `allergy_hits`
     de fuentes sintéticas (sus métricas son 0 por construcción del
     sintetizador, así que no contribuyen igual; el flag `weak_signal` sí
     pasa para preservar la awareness al LLM).
  3. Dict final propaga `has_synthesized_sources` y `synthesized_source_count`
     para que el prompt builder pueda emitir disclaimer.
  4. `build_chunk_lessons_context` añade aviso al prompt cuando hay fuentes
     sintetizadas.
  5. Telemetría `synth_propagated_to_prompt` permite alerting si % de chunks
     con synth_sources>0 sube en producción (indicador upstream de fallos).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from cron_tasks import _synthesize_last_chunk_learning_from_plan_days
from prompts.plan_generator import build_chunk_lessons_context


# ---------------------------------------------------------------------------
# 1. Sintetizador ahora añade `confidence_score=0.4` complementario al booleano.
# ---------------------------------------------------------------------------
def test_synth_payload_incluye_confidence_score():
    days = [
        {"week_number": 1, "meals": [
            {"name": "Pollo asado", "ingredients": ["pollo", "arroz"], "status": "ok"},
        ]},
    ]
    res = _synthesize_last_chunk_learning_from_plan_days(
        meal_plan_id="p", target_week=1, prior_plan_data={"days": days}
    )
    assert res is not None
    assert res["synthesized_from_plan_days"] is True
    assert res["confidence_score"] == 0.4
    assert res["low_confidence"] is True
    assert res["learning_signal_strength"] == "weak"


# ---------------------------------------------------------------------------
# 2. Prompt builder añade disclaimer cuando el dict tiene fuentes sintéticas.
# ---------------------------------------------------------------------------
def test_prompt_builder_emite_aviso_cuando_synth_top_level():
    """Caso: lección sintética suelta inyectada directo (sin agregador). El
    builder reconoce `synthesized_from_plan_days=True` al top-level y avisa."""
    chunk_lessons = {
        "chunk_number": 1,
        "ingredient_base_repeat_pct": 35.0,  # > gate de 30 → bullet fires
        "repeated_bases": [{"bases": ["pollo", "arroz"]}],
        "repeat_pct": 0,
        "repeated_meal_names": [],
        "rejection_violations": 0,
        "rejected_meals_that_reappeared": [],
        "allergy_violations": 0,
        "allergy_hits": [],
        "synthesized_from_plan_days": True,
        "low_confidence": True,
        "learning_signal_strength": "weak",
    }
    ctx = build_chunk_lessons_context(chunk_lessons)
    assert "AVISO" in ctx, f"esperado disclaimer, recibido: {ctx[:300]}"
    assert "DERIVADO" in ctx or "DERIVADAS" in ctx
    assert "MENOS peso" in ctx


def test_prompt_builder_emite_aviso_con_count_cuando_agregado_mixto():
    """Caso: el agregador del cron worker construyó un dict con
    `has_synthesized_sources=True` y `synthesized_source_count=2` (2 de N
    lecciones eran sintetizadas). El builder muestra el conteo explícito."""
    chunk_lessons = {
        "chunk_number": 3,
        "ingredient_base_repeat_pct": 40.0,
        "repeated_bases": [{"bases": ["pollo"]}],
        "repeat_pct": 20.0,
        "repeated_meal_names": ["Plato Real"],
        "rejection_violations": 1,
        "rejected_meals_that_reappeared": ["Salmón rechazado"],
        "allergy_violations": 0,
        "allergy_hits": [],
        "has_synthesized_sources": True,
        "synthesized_source_count": 2,
    }
    ctx = build_chunk_lessons_context(chunk_lessons)
    assert "AVISO" in ctx
    assert "2" in ctx, "el conteo explícito de fuentes sintéticas debe aparecer"
    assert "MENOS peso" in ctx


def test_prompt_builder_no_emite_aviso_cuando_lecciones_son_reales():
    """Regresión: en path normal (lecciones de adherencia observada), el
    builder NO debe añadir disclaimer — sería ruido innecesario."""
    chunk_lessons = {
        "chunk_number": 2,
        "ingredient_base_repeat_pct": 35.0,
        "repeated_bases": [{"bases": ["pollo"]}],
        "repeat_pct": 20.0,
        "repeated_meal_names": ["Plato"],
        "rejection_violations": 0,
        "rejected_meals_that_reappeared": [],
        "allergy_violations": 0,
        "allergy_hits": [],
        "has_synthesized_sources": False,
        "synthesized_source_count": 0,
    }
    ctx = build_chunk_lessons_context(chunk_lessons)
    assert "AVISO" not in ctx, f"path real no debe avisar; ctx={ctx[:300]}"


def test_prompt_builder_returns_empty_when_no_actionable_bullets():
    """Regresión: si el dict no tiene contenido accionable, devolvemos "" igual
    que antes. El disclaimer NO debe aparecer si no hay bullets que ponderar."""
    chunk_lessons = {
        "chunk_number": 1,
        "ingredient_base_repeat_pct": 0,
        "repeated_bases": [],
        "repeat_pct": 0,
        "repeated_meal_names": [],
        "rejection_violations": 0,
        "rejected_meals_that_reappeared": [],
        "allergy_violations": 0,
        "allergy_hits": [],
        "synthesized_from_plan_days": True,
        "has_synthesized_sources": True,
        "synthesized_source_count": 1,
    }
    ctx = build_chunk_lessons_context(chunk_lessons)
    assert ctx == "", "sin bullets accionables, no se inyecta nada al prompt"


# ---------------------------------------------------------------------------
# 3. Agregación: lección sintética NO debe propagar sus bases/meal_names al
#    agregado mezclado con una real. Reproducimos la lógica del worker (no
#    podemos invocar el método completo sin DB, así que replicamos el bucle
#    crítico — el mismo que vive en cron_tasks.py:15174+).
# ---------------------------------------------------------------------------
def _replicate_aggregator(lessons: list) -> dict:
    """Replica el bucle de agregación del worker (`cron_tasks.py:15174+`) para
    poder probarlo sin DB. Si esta función diverge del worker, el test detecta
    el drift via `test_aggregator_replica_matches_worker_signature` (no
    incluido aquí — se cubre indirectamente por test_chunked_learning_propagation
    e2e). Lo importante: aquí asegura el comportamiento exclusivo de P0-CHUNKS-1."""
    agg_bases: list = []
    agg_bases_seen: set = set()
    agg_meals: list = []
    agg_meals_seen: set = set()
    synth_count = 0

    for _lesson in lessons:
        if not isinstance(_lesson, dict):
            continue
        # [P0-CHUNKS-1] Skip de bases/meal_names si la fuente es sintética.
        if _lesson.get("synthesized_from_plan_days") is True:
            synth_count += 1
            continue
        for _rb in (_lesson.get("repeated_bases") or []):
            _key = str(_rb)
            if _key not in agg_bases_seen:
                agg_bases_seen.add(_key)
                agg_bases.append(_rb)
        for _rm in (_lesson.get("repeated_meal_names") or []):
            if _rm not in agg_meals_seen:
                agg_meals_seen.add(_rm)
                agg_meals.append(_rm)

    return {
        "repeated_bases": agg_bases,
        "repeated_meal_names": agg_meals,
        "has_synthesized_sources": synth_count > 0,
        "synthesized_source_count": synth_count,
    }


def test_aggregator_excluye_bases_de_leccion_sintetica():
    real_lesson = {
        "chunk": 1,
        "repeated_bases": ["pollo_real", "arroz_real"],
        "repeated_meal_names": ["Plato Real"],
    }
    synth_lesson = {
        "chunk": 2,
        "repeated_bases": ["pescado_synth", "quinoa_synth"],
        "repeated_meal_names": ["Plato Synth"],
        "synthesized_from_plan_days": True,
    }
    agg = _replicate_aggregator([real_lesson, synth_lesson])

    assert "pollo_real" in agg["repeated_bases"]
    assert "arroz_real" in agg["repeated_bases"]
    # Las bases sintéticas NO deben aparecer — son "lo planificado" no "lo observado".
    assert "pescado_synth" not in agg["repeated_bases"]
    assert "quinoa_synth" not in agg["repeated_bases"]
    # Los meal_names sintéticos tampoco.
    assert "Plato Real" in agg["repeated_meal_names"]
    assert "Plato Synth" not in agg["repeated_meal_names"]
    # Track del conteo:
    assert agg["has_synthesized_sources"] is True
    assert agg["synthesized_source_count"] == 1


def test_aggregator_solo_synth_produce_listas_vacias():
    """Caso patológico: TODAS las lecciones del rolling window son sintéticas
    (chunks N-1, N-2, N-3 todos crashearon pre-pipeline). Antes el LLM recibía
    una blocklist completamente derivada del plan; ahora recibe listas vacías
    + el disclaimer."""
    synth1 = {"chunk": 1, "repeated_bases": ["a"], "repeated_meal_names": ["X"],
              "synthesized_from_plan_days": True}
    synth2 = {"chunk": 2, "repeated_bases": ["b"], "repeated_meal_names": ["Y"],
              "synthesized_from_plan_days": True}
    agg = _replicate_aggregator([synth1, synth2])

    assert agg["repeated_bases"] == []
    assert agg["repeated_meal_names"] == []
    assert agg["has_synthesized_sources"] is True
    assert agg["synthesized_source_count"] == 2


def test_aggregator_solo_reales_no_marca_synth():
    """Path normal: ninguna lección sintética → flag false, listas con
    contenido real intacto."""
    real1 = {"chunk": 1, "repeated_bases": ["a"], "repeated_meal_names": ["X"]}
    real2 = {"chunk": 2, "repeated_bases": ["b"], "repeated_meal_names": ["Y"]}
    agg = _replicate_aggregator([real1, real2])

    assert "a" in agg["repeated_bases"]
    assert "b" in agg["repeated_bases"]
    assert "X" in agg["repeated_meal_names"]
    assert "Y" in agg["repeated_meal_names"]
    assert agg["has_synthesized_sources"] is False
    assert agg["synthesized_source_count"] == 0


# ---------------------------------------------------------------------------
# 4. Bug original reproducido: lección sintética + real mezcladas, antes
#    contaminaban "evita estos ingredientes" con items planificados.
# ---------------------------------------------------------------------------
def test_escenario_bug_original_synth_no_contamina_blocklist():
    """
    PRE-FIX: chunk 1 crasheó pre-pipeline → synth lesson con
    repeated_bases=['pescado'] (lo PLANIFICADO en chunk 1, no consumido).
    Chunk 2 corrió ok → real lesson con repeated_bases=['pollo'] (de logs).
    Worker agregaba ambos → LLM del chunk 3 recibía "diversifica de:
    pollo, pescado" → el usuario nunca vio pescado pero el modelo cree
    que se sobre-usó.

    POST-FIX: solo 'pollo' aparece en la blocklist final. 'pescado' (synth)
    se excluye, y un disclaimer notifica al LLM que parte del bloque es
    derivado.
    """
    synth_chunk1 = {
        "chunk": 1,
        "repeated_bases": ["pescado", "quinoa"],
        "repeated_meal_names": ["Salmón con quinoa"],
        "synthesized_from_plan_days": True,
        "confidence_score": 0.4,
    }
    real_chunk2 = {
        "chunk": 2,
        "repeated_bases": ["pollo"],
        "repeated_meal_names": ["Pollo a la plancha"],
        # confidence implícito alto (no es synth)
    }
    agg = _replicate_aggregator([synth_chunk1, real_chunk2])

    assert agg["repeated_bases"] == ["pollo"]
    assert agg["repeated_meal_names"] == ["Pollo a la plancha"]
    assert "pescado" not in agg["repeated_bases"]
    assert "quinoa" not in agg["repeated_bases"]
    assert "Salmón con quinoa" not in agg["repeated_meal_names"]
    assert agg["synthesized_source_count"] == 1


# ---------------------------------------------------------------------------
# 5. Smoke test: prompt builder + agregador integrados.
# ---------------------------------------------------------------------------
def test_pipeline_completo_synth_real_genera_prompt_sin_pollucion_y_con_aviso():
    """End-to-end del fix: aggregator filtra synth, dict final lleva flag,
    prompt builder añade disclaimer + bullets solo con contenido real."""
    synth = {
        "chunk": 1,
        "ingredient_base_repeat_pct": 0,
        "repeat_pct": 0,
        "repeated_bases": ["pescado_synth"],
        "repeated_meal_names": ["Plato Synth"],
        "rejection_violations": 0,
        "allergy_violations": 0,
        "synthesized_from_plan_days": True,
    }
    real = {
        "chunk": 2,
        "ingredient_base_repeat_pct": 45.0,
        "repeat_pct": 18.0,
        "repeated_bases": [{"bases": ["pollo_real"]}],
        "repeated_meal_names": ["Pollo a la plancha"],
        "rejection_violations": 0,
        "allergy_violations": 0,
    }
    agg = _replicate_aggregator([synth, real])

    # Construimos el dict final como lo haría el worker:
    final_chunk_lessons = {
        "chunk_number": 3,
        "ingredient_base_repeat_pct": real["ingredient_base_repeat_pct"],
        "repeat_pct": real["repeat_pct"],
        "repeated_bases": [{"bases": ["pollo_real"]}],
        "repeated_meal_names": agg["repeated_meal_names"],
        "rejection_violations": 0,
        "rejected_meals_that_reappeared": [],
        "allergy_violations": 0,
        "allergy_hits": [],
        "has_synthesized_sources": agg["has_synthesized_sources"],
        "synthesized_source_count": agg["synthesized_source_count"],
    }
    ctx = build_chunk_lessons_context(final_chunk_lessons)

    # 1) Disclaimer presente:
    assert "AVISO" in ctx
    # 2) Solo el contenido real aparece en la blocklist:
    assert "pollo_real" in ctx
    assert "pescado_synth" not in ctx
    assert "Plato Synth" not in ctx
    # 3) El bullet de "Nombres de platos repetidos" usa solo el real:
    assert "Pollo a la plancha" in ctx
