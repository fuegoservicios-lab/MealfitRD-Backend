"""
Tests P1-2: Regeneración de _recent_chunk_lessons desde plan_data.days.

Cuando ambas fuentes primarias fallaron:
  - plan_data._recent_chunk_lessons truncado/vacío.
  - plan_chunk_queue.learning_metrics no provee suficientes filas.

El helper `_regenerate_recent_chunk_lessons_from_plan_days` rellena la ventana
rolling combinando seed_lessons (vienen de la cola) con lecciones sintetizadas
desde plan_data.days vía `_synthesize_last_chunk_learning_from_plan_days`.

Cubre:
  1. Sin plan_data.days → retorna seed_lessons sin cambios.
  2. Sin seed + days con chunks tagged → sintetiza todas las lecciones faltantes.
  3. Mezcla seed + sintetizadas — seeds tienen prioridad (no se duplican).
  4. Sintetizadas marcadas con synthesized_from_plan_days=True y low_confidence=True.
  5. Resultado truncado al rolling window cap del total_days_requested.
  6. Robustez ante plan_data malformado (None, sin days, days no-list).
  7. target_week=1 → no hay chunks previos a regenerar, retorna [].
"""
import pytest
from unittest.mock import patch

from cron_tasks import _regenerate_recent_chunk_lessons_from_plan_days


def _day(day_num, week_number, meals):
    """Helper: día con tag de week_number/chunk para filtrar por chunk."""
    return {
        "day": day_num,
        "week_number": week_number,
        "meals": [
            {"name": n, "type": "Almuerzo", "ingredients": ["pollo"]}
            for n in meals
        ],
    }


def _seed_lesson(chunk_idx, repeat_pct=10):
    """Lesson reconstruida desde la cola (no synthetic)."""
    return {
        "chunk": chunk_idx,
        "repeat_pct": repeat_pct,
        "ingredient_base_repeat_pct": 5,
        "rejection_violations": 0,
        "allergy_violations": 0,
        "fatigued_violations": 0,
        "repeated_bases": ["pollo"],
        "repeated_meal_names": [f"Pollo c{chunk_idx}"],
        "rejected_meals_that_reappeared": [],
        "allergy_hits": [],
        "metrics_unavailable": False,
        "low_confidence": False,
        "learning_signal_strength": "strong",
        "rebuilt_from_queue": True,
    }


def test_regen_returns_seed_unchanged_when_no_plan_days():
    """Sin plan_data.days, no podemos sintetizar — retorna seed sin tocar."""
    seed = [_seed_lesson(1), _seed_lesson(2)]
    result = _regenerate_recent_chunk_lessons_from_plan_days(
        meal_plan_id="plan-x",
        plan_data={},  # sin "days"
        target_week=4,
        total_days_requested=30,
        seed_lessons=seed,
    )
    # Sin days, retorna lo que sea que tenga seed (cap al window).
    assert result == seed


def test_regen_synthesizes_all_when_no_seed():
    """Sin seed pero con days tagged: sintetiza una lesson por chunk previo."""
    plan_data = {
        "days": [
            _day(1, 1, ["Pollo asado"]),
            _day(2, 1, ["Salmón"]),
            _day(3, 1, ["Res guisada"]),
            _day(4, 2, ["Camarones"]),
            _day(5, 2, ["Atún"]),
            _day(6, 2, ["Pavo"]),
        ]
    }
    result = _regenerate_recent_chunk_lessons_from_plan_days(
        meal_plan_id="plan-x",
        plan_data=plan_data,
        target_week=3,  # regenerar contexto para chunk 3 → chunks 1 y 2
        total_days_requested=30,
        seed_lessons=[],
    )
    # Esperado: 2 lecciones sintetizadas (chunks 1 y 2).
    assert len(result) == 2
    assert all(l.get("synthesized_from_plan_days") for l in result)
    assert all(l.get("low_confidence") for l in result)
    # Chunks ordenados: 1, 2.
    chunks = [l.get("chunk") for l in result]
    assert chunks == [1, 2]


def test_regen_seed_takes_priority_over_synth():
    """Si seed tiene chunk_idx=1, ese se usa; el chunk_idx=2 (faltante) se sintetiza."""
    plan_data = {
        "days": [
            _day(1, 1, ["Pollo"]),
            _day(2, 1, ["Salmón"]),
            _day(3, 1, ["Res"]),
            _day(4, 2, ["Camarones"]),
            _day(5, 2, ["Atún"]),
            _day(6, 2, ["Pavo"]),
        ]
    }
    seed = [_seed_lesson(1, repeat_pct=99)]  # solo chunk 1 desde queue
    result = _regenerate_recent_chunk_lessons_from_plan_days(
        meal_plan_id="plan-x",
        plan_data=plan_data,
        target_week=3,
        total_days_requested=30,
        seed_lessons=seed,
    )
    assert len(result) == 2
    # Chunk 1 viene de seed: rebuilt_from_queue=True, repeat_pct=99.
    assert result[0]["chunk"] == 1
    assert result[0].get("rebuilt_from_queue") is True
    assert result[0].get("repeat_pct") == 99
    assert not result[0].get("synthesized_from_plan_days")
    # Chunk 2 viene de síntesis: synthesized_from_plan_days=True.
    assert result[1]["chunk"] == 2
    assert result[1].get("synthesized_from_plan_days") is True
    assert result[1].get("low_confidence") is True


def test_regen_truncates_to_rolling_window_cap():
    """Con muchos chunks previos, el resultado se trunca al window cap."""
    # Plan 30d con 9 chunks (días 1-27 distribuidos en chunks de 3) → window cap es 8.
    plan_data = {"days": []}
    for chunk in range(1, 10):  # chunks 1..9
        for d_in_chunk in range(3):
            day_num = (chunk - 1) * 3 + d_in_chunk + 1
            plan_data["days"].append(_day(day_num, chunk, [f"Plato c{chunk}d{d_in_chunk}"]))

    result = _regenerate_recent_chunk_lessons_from_plan_days(
        meal_plan_id="plan-x",
        plan_data=plan_data,
        target_week=10,  # generar contexto para chunk 10 → 9 chunks previos
        total_days_requested=30,
        seed_lessons=[],
    )
    # rolling window cap para 30d es 8 (ver _rolling_lessons_window_cap).
    from cron_tasks import _rolling_lessons_window_cap
    expected_cap = _rolling_lessons_window_cap(30)
    assert len(result) == expected_cap, (
        f"Esperaba {expected_cap} lecciones (window cap), hubo {len(result)}"
    )
    # Las últimas (más recientes) deben sobrevivir.
    chunks_in_result = [l["chunk"] for l in result]
    assert chunks_in_result == list(range(10 - expected_cap, 10))


def test_regen_handles_target_week_1():
    """target_week=1 no tiene chunks previos a regenerar."""
    plan_data = {"days": [_day(1, 1, ["Pollo"])]}
    result = _regenerate_recent_chunk_lessons_from_plan_days(
        meal_plan_id="plan-x",
        plan_data=plan_data,
        target_week=1,
        total_days_requested=30,
        seed_lessons=[],
    )
    assert result == []


def test_regen_handles_malformed_plan_data():
    """plan_data None / sin days no debe crashear."""
    # plan_data None
    r1 = _regenerate_recent_chunk_lessons_from_plan_days(
        meal_plan_id="plan-x",
        plan_data=None,
        target_week=3,
        total_days_requested=30,
        seed_lessons=[_seed_lesson(1)],
    )
    assert r1 == [_seed_lesson(1)]

    # days no es list
    r2 = _regenerate_recent_chunk_lessons_from_plan_days(
        meal_plan_id="plan-x",
        plan_data={"days": "not a list"},
        target_week=3,
        total_days_requested=30,
        seed_lessons=[],
    )
    assert r2 == []


def test_regen_synthesized_lessons_have_correct_metadata():
    """Las lecciones sintetizadas deben llevar la metadata correcta para que el LLM
    las pondere apropiadamente como señal débil."""
    plan_data = {
        "days": [
            _day(1, 1, ["Pollo asado", "Avena"]),
            _day(2, 1, ["Salmón", "Yogur"]),
            _day(3, 1, ["Res guisada", "Tostada"]),
        ]
    }
    result = _regenerate_recent_chunk_lessons_from_plan_days(
        meal_plan_id="plan-x",
        plan_data=plan_data,
        target_week=2,
        total_days_requested=15,
        seed_lessons=[],
    )
    assert len(result) == 1
    lesson = result[0]
    # Los marcadores críticos:
    assert lesson["synthesized_from_plan_days"] is True
    assert lesson["low_confidence"] is True
    assert lesson["learning_signal_strength"] == "weak"
    # Debe tener nombres y bases poblados (no listas vacías).
    assert len(lesson["repeated_meal_names"]) > 0
    assert len(lesson["repeated_bases"]) > 0


def test_regen_seed_with_invalid_chunk_field_skipped():
    """Una seed lesson con chunk inválido (None, string raro) no rompe el bucle."""
    plan_data = {
        "days": [
            _day(1, 1, ["Pollo"]),
            _day(2, 1, ["Salmón"]),
            _day(3, 1, ["Res"]),
        ]
    }
    seed = [
        {"chunk": None, "repeat_pct": 0},  # inválida
        _seed_lesson(1),  # válida
    ]
    result = _regenerate_recent_chunk_lessons_from_plan_days(
        meal_plan_id="plan-x",
        plan_data=plan_data,
        target_week=2,
        total_days_requested=30,
        seed_lessons=seed,
    )
    # La inválida se ignora; la válida (chunk=1) se respeta.
    assert len(result) == 1
    assert result[0]["chunk"] == 1
    assert result[0].get("rebuilt_from_queue") is True
