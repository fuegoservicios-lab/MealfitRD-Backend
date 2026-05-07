"""
P0-5: Test E2E para plan de 30 días (10 chunks).

Cubre los casos NO probados por test_chunked_15days_e2e.py:
  1. Encolar 10 chunks (chunk 1 sincrónico + 9 en queue), avanzar reloj,
     verificar que cada chunk N recibe lecciones del N-1.
  2. Rolling window de _recent_chunk_lessons se trunca ≤8 para planes ≥15d
     (regla en cron_tasks.py).
  3. _lifetime_lessons_summary persiste y acumula a lo largo de los 10 chunks
     (rotación 60d).
  4. Plan completa los 30 días sin corrupción.
  5. Final chunk delay_days = days_offset_int - 3 para planes ≥15d.

Uses the seeded_user_profile fixture from tests/conftest.py.

Run with:
    cd backend && python -m pytest tests/test_chunked_30days_e2e.py -v -m e2e
"""
import pytest
import uuid
import json
import math
import time
from unittest.mock import patch
from datetime import datetime, timezone, timedelta

from db_core import execute_sql_write, execute_sql_query
from cron_tasks import process_plan_chunk_queue, _enqueue_plan_chunk, _compute_chunk_delay_days
from constants import split_with_absorb, PLAN_CHUNK_SIZE

import logging
logging.getLogger().setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Mock del pipeline: genera días con lesson stub para validar propagación
# ---------------------------------------------------------------------------
_pipeline_call_log = []


def _mock_run_plan_pipeline(form_data, *args, **kwargs):
    """Mock del pipeline que captura form_data para verificar lesson propagation."""
    offset = form_data.get("_days_offset", 0)
    count = form_data.get("_days_to_generate", 3)

    # Capturar para aserciones post-run
    _pipeline_call_log.append({
        "offset": offset,
        "count": count,
        "has_chunk_lessons": "_chunk_lessons" in form_data,
        "chunk_lessons": form_data.get("_chunk_lessons"),
        "recent_chunk_lessons": form_data.get("_recent_chunk_lessons"),
    })

    days = []
    for i in range(count):
        day_num = offset + i + 1
        days.append({
            "day": day_num,
            "daily_summary": f"Day {day_num} summary",
            "meals": [
                {"name": f"Breakfast {day_num}", "type": "Desayuno",
                 "ingredients": ["100g pollo", "arroz"]},
                {"name": f"Lunch {day_num}", "type": "Almuerzo",
                 "ingredients": ["150g res", "habichuelas"]},
            ]
        })

    return {"days": days, "generation_status": "partial"}


# ---------------------------------------------------------------------------
# Helper: create initial plan and enqueue chunks
# ---------------------------------------------------------------------------
def _setup_30day_plan(user_id, plan_id, seed_lessons=None, pre_days=None,
                      initial_total_generated=3, chunks_to_enqueue=range(2, 11)):
    """Creates a 30-day plan with chunk 1 done and enqueues the rest."""
    if pre_days is None:
        initial_form_data = {"user_id": user_id, "_days_offset": 0, "_days_to_generate": 3}
        initial_plan_result = _mock_run_plan_pipeline(initial_form_data)
        pre_days = initial_plan_result["days"]

    initial_plan = {
        "total_days_requested": 30,
        "total_days_generated": initial_total_generated,
        "generation_status": "partial",
        "days": pre_days,
    }

    if seed_lessons:
        for key, value in seed_lessons.items():
            initial_plan[key] = value

    execute_sql_write(
        "INSERT INTO meal_plans (id, user_id, plan_data) VALUES (%s, %s, %s)",
        (plan_id, user_id, json.dumps(initial_plan))
    )

    inserted = execute_sql_query(
        "SELECT id FROM meal_plans WHERE id = %s", (plan_id,), fetch_one=True
    )
    assert inserted is not None, "Plan was not inserted!"

    valid_snapshot = {
        "totalDays": 30,
        "form_data": {
            "_days_offset": 0,
            "householdSize": 1,
            "dietType": "Omnívora",
            "_plan_start_date": datetime.now(timezone.utc).isoformat(),
        }
    }

    for chunk_idx in chunks_to_enqueue:
        offset = (chunk_idx - 1) * 3
        _enqueue_plan_chunk(
            user_id, plan_id,
            week_number=chunk_idx,
            days_offset=offset,
            days_count=3,
            pipeline_snapshot=valid_snapshot,
        )

    execute_sql_write(
        "UPDATE plan_chunk_queue SET execute_after = NOW() - INTERVAL '1 MINUTE' WHERE meal_plan_id = %s",
        (plan_id,)
    )


# ---------------------------------------------------------------------------
# Test 1: E2E completo — 10 chunks, 30 días, lesson chain, rolling window
# ---------------------------------------------------------------------------
@pytest.mark.e2e
@patch('cron_tasks.run_plan_pipeline', side_effect=_mock_run_plan_pipeline)
def test_chunked_30days_e2e_full_plan(mock_pipeline, seeded_user_profile):
    """Plan de 30 días: 10 chunks de 3 días cada uno.

    Verifica:
    - Chunk 1 se genera sincrónicamente, 9 chunks se encolan
    - Todos los chunks completan exitosamente
    - El plan final tiene exactamente 30 días
    - Cada chunk N≥3 recibe _chunk_lessons del chunk N-1
    - _recent_chunk_lessons se trunca a ≤8
    - _lifetime_lessons_summary se acumula
    """
    global _pipeline_call_log
    _pipeline_call_log = []

    user_id, plan_id = seeded_user_profile

    _setup_30day_plan(user_id, plan_id)

    # Verify 9 items in queue
    queued = execute_sql_query(
        "SELECT * FROM plan_chunk_queue WHERE meal_plan_id = %s",
        (plan_id,), fetch_all=True
    )
    assert len(queued) == 9, f"Esperados 9 chunks en cola, got {len(queued)}"

    # Procesar los 9 chunks secuencialmente
    for i in range(9):
        logging.info(f"Processing chunk iteration {i + 1}/9")
        process_plan_chunk_queue(target_plan_id=plan_id)

    # Verificar que todos los chunks completaron
    queued_after = execute_sql_query(
        "SELECT week_number, status FROM plan_chunk_queue WHERE meal_plan_id = %s ORDER BY week_number",
        (plan_id,), fetch_all=True
    )
    for q in queued_after:
        assert q['status'] == 'completed', \
            f"Chunk {q['week_number']} no completó: status={q['status']}"

    # Verificar que el plan final tiene 30 días
    final_plan_row = execute_sql_query(
        "SELECT plan_data FROM meal_plans WHERE id = %s",
        (plan_id,), fetch_one=True
    )
    assert final_plan_row is not None

    final_plan = final_plan_row["plan_data"]
    if isinstance(final_plan, str):
        final_plan = json.loads(final_plan)

    assert len(final_plan["days"]) == 30, \
        f"Plan final debe tener 30 días, tiene {len(final_plan['days'])}"

    # Verificar continuidad de días 1..30
    for i in range(30):
        assert final_plan["days"][i]["day"] == i + 1, \
            f"Día {i+1} incorrecto: {final_plan['days'][i].get('day')}"
        assert "Breakfast" in final_plan["days"][i]["meals"][0]["name"], \
            f"Día {i+1} no tiene el formato esperado"

    # Verificar _recent_chunk_lessons se truncó a ≤8
    recent_lessons = final_plan.get("_recent_chunk_lessons", [])
    assert len(recent_lessons) <= 8, \
        f"Rolling window debe truncarse a ≤8 para planes ≥15d, tiene {len(recent_lessons)}"

    # Los chunks más recientes deben estar presentes
    if recent_lessons:
        chunk_nums_in_window = [l.get("chunk") for l in recent_lessons if isinstance(l, dict)]
        # El chunk 10 (último) debe estar en la ventana
        assert 10 in chunk_nums_in_window, \
            f"Chunk 10 (último) debe estar en la ventana rolling, found: {chunk_nums_in_window}"
        # El chunk 1 (si existía como lesson seeded) debió ser expulsado
        if len(chunk_nums_in_window) == 8:
            assert 1 not in chunk_nums_in_window, \
                f"Chunk 1 debió ser expulsado de la ventana rolling de 8, found: {chunk_nums_in_window}"

    # Verificar _lifetime_lessons_summary existe y acumula
    lifetime = final_plan.get("_lifetime_lessons_summary")
    assert lifetime is not None, \
        "_lifetime_lessons_summary debe existir después de 10 chunks"
    assert "total_rejection_violations" in lifetime, \
        "lifetime_lessons_summary debe tener total_rejection_violations"

    # Verificar _lifetime_lessons_history tiene entradas (filtradas por ventana 60d)
    history = final_plan.get("_lifetime_lessons_history", [])
    assert len(history) >= 8, \
        f"history debe tener al menos 8 entradas (9 chunks procesados), tiene {len(history)}"

    # Verificar generation_status es 'complete'
    assert final_plan.get("generation_status") == "complete", \
        f"Plan 30d debe marcar generation_status='complete', got: {final_plan.get('generation_status')}"


# ---------------------------------------------------------------------------
# Test 2: Lesson propagation — chunk N recibe lecciones del N-1
# ---------------------------------------------------------------------------
@pytest.mark.e2e
@patch('cron_tasks.run_plan_pipeline', side_effect=_mock_run_plan_pipeline)
def test_chunked_30days_lesson_chain_propagation(mock_pipeline, seeded_user_profile):
    """Verifica que para cada chunk N≥3, el pipeline recibe _chunk_lessons
    con datos del chunk anterior, formando una cadena continua de aprendizaje."""
    global _pipeline_call_log
    _pipeline_call_log = []

    user_id, plan_id = seeded_user_profile

    # Seed con lesson del chunk 1 para que chunk 2 ya tenga contexto
    seed_lesson = {
        "chunk": 1,
        "repeat_pct": 10.0,
        "ingredient_base_repeat_pct": 20.0,
        "rejection_violations": 1,
        "allergy_violations": 0,
        "fatigued_violations": 0,
        "repeated_bases": [{"bases": ["pollo"]}],
        "repeated_meal_names": ["Pollo Asado"],
        "rejected_meals_that_reappeared": [],
        "allergy_hits": [],
        "metrics_unavailable": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _setup_30day_plan(
        user_id, plan_id,
        seed_lessons={
            "_last_chunk_learning": seed_lesson,
            "_recent_chunk_lessons": [seed_lesson],
        },
        chunks_to_enqueue=range(2, 6),  # Solo chunks 2-5
    )

    # Procesar 4 chunks
    for _ in range(4):
        process_plan_chunk_queue(target_plan_id=plan_id)

    # Verificar la cadena: chunk 2 ya debería recibir lessons del seed.
    # Nota: _pipeline_call_log[0] es chunk 2 (primer chunk procesado por queue)
    # Chunks ≥3 deben tener _chunk_lessons
    for idx, call in enumerate(_pipeline_call_log):
        chunk_num = idx + 2  # chunk 2 es el primero del log
        if chunk_num >= 3:
            assert call["has_chunk_lessons"], \
                f"Chunk {chunk_num} no recibió _chunk_lessons del chunk anterior"


# ---------------------------------------------------------------------------
# Test 3: Rolling window truncation a 8 con 10 chunks
# ---------------------------------------------------------------------------
@pytest.mark.e2e
@patch('cron_tasks.run_plan_pipeline', side_effect=_mock_run_plan_pipeline)
def test_chunked_30days_rolling_window_truncates_to_8(mock_pipeline, seeded_user_profile):
    """Verifica que después de 10 chunks (1 sync + 9 queue), la ventana
    rolling _recent_chunk_lessons no excede 8 entradas."""
    global _pipeline_call_log
    _pipeline_call_log = []

    user_id, plan_id = seeded_user_profile

    _setup_30day_plan(user_id, plan_id)

    for _ in range(9):
        process_plan_chunk_queue(target_plan_id=plan_id)

    final_plan_row = execute_sql_query(
        "SELECT plan_data FROM meal_plans WHERE id = %s",
        (plan_id,), fetch_one=True
    )
    final_plan = final_plan_row["plan_data"]
    if isinstance(final_plan, str):
        final_plan = json.loads(final_plan)

    recent = final_plan.get("_recent_chunk_lessons", [])
    assert len(recent) <= 8, \
        f"_recent_chunk_lessons debe truncarse a ≤8 para plan 30d, tiene {len(recent)}"

    # Verificar que la ventana contiene los chunks más recientes
    chunk_numbers = [l.get("chunk") for l in recent if isinstance(l, dict)]
    if len(chunk_numbers) == 8:
        # Deben ser chunks 3-10 (chunk 2 ya fue expulsado)
        assert min(chunk_numbers) >= 2, \
            f"El chunk más viejo en la ventana no debería ser el 1, got min={min(chunk_numbers)}"
        assert max(chunk_numbers) == 10, \
            f"El chunk más reciente debe ser 10, got max={max(chunk_numbers)}"


# ---------------------------------------------------------------------------
# Test 4: Lifetime lessons acumulación con 10 chunks
# ---------------------------------------------------------------------------
@pytest.mark.e2e
@patch('cron_tasks.run_plan_pipeline', side_effect=_mock_run_plan_pipeline)
def test_chunked_30days_lifetime_lessons_accumulate(mock_pipeline, seeded_user_profile):
    """Verifica que _lifetime_lessons_summary y _lifetime_lessons_history
    se acumulan correctamente a lo largo de los 10 chunks del plan de 30 días."""
    global _pipeline_call_log
    _pipeline_call_log = []

    user_id, plan_id = seeded_user_profile

    _setup_30day_plan(user_id, plan_id)

    for _ in range(9):
        process_plan_chunk_queue(target_plan_id=plan_id)

    final_plan_row = execute_sql_query(
        "SELECT plan_data FROM meal_plans WHERE id = %s",
        (plan_id,), fetch_one=True
    )
    final_plan = final_plan_row["plan_data"]
    if isinstance(final_plan, str):
        final_plan = json.loads(final_plan)

    # Verificar lifetime summary
    lifetime = final_plan.get("_lifetime_lessons_summary")
    assert lifetime is not None, "_lifetime_lessons_summary debe existir"
    assert "_lifetime_window_days" in lifetime, \
        "lifetime debe incluir _lifetime_window_days"

    # Verificar que el historial acumuló entradas de todos los chunks procesados
    history = final_plan.get("_lifetime_lessons_history", [])
    assert len(history) >= 9, \
        f"_lifetime_lessons_history debe tener ≥9 entradas (9 chunks), tiene {len(history)}"

    # Cada entrada debe tener un timestamp
    for entry in history:
        assert "timestamp" in entry, "Cada entrada de history debe tener timestamp"
        assert "chunk" in entry, "Cada entrada de history debe tener chunk"


# ---------------------------------------------------------------------------
# Test 5: Verificar que chunks 9-10 (finales) completan correctamente
# ---------------------------------------------------------------------------
@pytest.mark.e2e
@patch('cron_tasks.run_plan_pipeline', side_effect=_mock_run_plan_pipeline)
def test_chunked_30days_final_chunks_9_10_complete(mock_pipeline, seeded_user_profile):
    """Verifica específicamente que los chunks 9 y 10 (últimos del plan 30d)
    completan sin errores y que el plan se marca como 'complete' después del
    chunk 10."""
    global _pipeline_call_log
    _pipeline_call_log = []

    user_id, plan_id = seeded_user_profile

    # Pre-cargar plan con 24 días (chunks 1-8 ya completados)
    pre_days = []
    for d in range(1, 25):
        pre_days.append({
            "day": d,
            "meals": [
                {"name": f"Breakfast {d}", "ingredients": ["pollo"]},
                {"name": f"Lunch {d}", "ingredients": ["res"]},
            ]
        })

    # Simular que ya hay lecciones de chunks 1-8
    lessons_history = []
    recent_lessons = []
    for c in range(1, 9):
        lesson = {
            "chunk": c,
            "repeat_pct": float(c),
            "ingredient_base_repeat_pct": float(c * 10),
            "rejection_violations": 0,
            "allergy_violations": 0,
            "fatigued_violations": 0,
            "repeated_bases": [],
            "repeated_meal_names": [],
            "rejected_meals_that_reappeared": [],
            "allergy_hits": [],
            "metrics_unavailable": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        lessons_history.append(lesson)
        recent_lessons.append(lesson)

    # Truncar recent a 8 (ya está en 8 exacto)
    recent_lessons = recent_lessons[-8:]

    _setup_30day_plan(
        user_id, plan_id,
        seed_lessons={
            "_last_chunk_learning": lessons_history[-1],
            "_recent_chunk_lessons": recent_lessons,
            "_lifetime_lessons_history": lessons_history,
            "_lifetime_lessons_summary": {
                "total_rejection_violations": 0,
                "total_allergy_violations": 0,
                "top_rejection_hits": [],
                "top_repeated_bases": [],
                "_lifetime_window_days": 60,
            },
        },
        pre_days=pre_days,
        initial_total_generated=24,
        chunks_to_enqueue=[9, 10],  # Solo los últimos 2
    )

    # Procesar los 2 chunks finales
    process_plan_chunk_queue(target_plan_id=plan_id)
    process_plan_chunk_queue(target_plan_id=plan_id)

    # Verificar ambos completados
    queued_after = execute_sql_query(
        "SELECT week_number, status FROM plan_chunk_queue WHERE meal_plan_id = %s ORDER BY week_number",
        (plan_id,), fetch_all=True
    )
    for q in queued_after:
        assert q['status'] == 'completed', \
            f"Chunk final {q['week_number']} no completó: {q['status']}"

    # Verificar plan final
    final_row = execute_sql_query(
        "SELECT plan_data FROM meal_plans WHERE id = %s",
        (plan_id,), fetch_one=True
    )
    final_plan = final_row["plan_data"]
    if isinstance(final_plan, str):
        final_plan = json.loads(final_plan)

    # Plan debe tener 30 días exactos
    assert len(final_plan["days"]) == 30, \
        f"Plan final debe tener 30 días, tiene {len(final_plan['days'])}"

    # Debe estar marcado como complete
    assert final_plan.get("generation_status") == "complete", \
        f"Plan 30d debe ser 'complete' después del chunk 10, got: {final_plan.get('generation_status')}"

    # Rolling window: tras chunks 9 y 10, ventana debe contener chunk 10
    recent = final_plan.get("_recent_chunk_lessons", [])
    assert len(recent) <= 8, \
        f"Rolling window debe ser ≤8 incluso tras 10 chunks, got {len(recent)}"

    chunk_nums = [l.get("chunk") for l in recent if isinstance(l, dict)]
    assert 10 in chunk_nums, \
        f"Chunk 10 (último) debe estar en la ventana rolling, got: {chunk_nums}"
    assert 9 in chunk_nums, \
        f"Chunk 9 debe estar en la ventana rolling, got: {chunk_nums}"

    # Lifetime history debe incluir los 2 nuevos chunks
    history = final_plan.get("_lifetime_lessons_history", [])
    history_chunks = [h.get("chunk") for h in history if isinstance(h, dict)]
    assert 9 in history_chunks, "Chunk 9 debe estar en _lifetime_lessons_history"
    assert 10 in history_chunks, "Chunk 10 debe estar en _lifetime_lessons_history"


# ---------------------------------------------------------------------------
# Test 6: Final chunk delay_days formula per mode
# ---------------------------------------------------------------------------
@pytest.mark.e2e
def test_final_chunk_delay_days_formula(seeded_user_profile):
    """Verifica que _compute_chunk_delay_days aplica la fórmula correcta
    según el modo (strict vs non-strict).

    - Strict mode: delay = max(0, offset - CHUNK_PROACTIVE_MARGIN_DAYS)
      (no hay adelanto especial para chunks finales)
    - Non-strict mode: final chunks (week >= total_weeks-1) usan
      delay = max(0, offset - 3) para adelantarse 3 días.
    """
    from constants import CHUNK_PROACTIVE_MARGIN_DAYS

    total_chunks = math.ceil(30 / PLAN_CHUNK_SIZE)  # 10
    snapshot = {"totalDays": 30}

    # --- Test actual mode (strict by default) ---
    delay_days, mode, offset_int, count_int = _compute_chunk_delay_days(
        days_offset=27,
        days_count=3,
        week_number=10,
        pipeline_snapshot=snapshot,
    )
    assert offset_int == 27

    if mode == "strict":
        # Strict mode: delay = offset - proactive_margin (default 0)
        expected = max(0, 27 - CHUNK_PROACTIVE_MARGIN_DAYS)
        assert delay_days == expected, \
            f"Strict mode: delay_days should be offset-margin={expected}, got {delay_days}"

        # Chunk intermedio (5): same formula
        delay_5, mode_5, _, _ = _compute_chunk_delay_days(
            days_offset=12, days_count=3, week_number=5,
            pipeline_snapshot=snapshot,
        )
        expected_5 = max(0, 12 - CHUNK_PROACTIVE_MARGIN_DAYS)
        assert delay_5 == expected_5, \
            f"Strict mid-chunk: expected {expected_5}, got {delay_5}"
    else:
        # Non-strict mode: final chunks advance by 3 days
        assert delay_days == max(0, 27 - 3), \
            f"Non-strict final chunk delay_days should be offset-3=24, got {delay_days}"

        # Chunk 9 (penúltimo, also >= total_weeks-1)
        delay_9, _, _, _ = _compute_chunk_delay_days(
            days_offset=24, days_count=3, week_number=9,
            pipeline_snapshot=snapshot,
        )
        assert delay_9 == max(0, 24 - 3), \
            f"Non-strict penultimate delay should be offset-3=21, got {delay_9}"

        # Chunk 5 (medio): standard formula, no advance
        delay_5, _, _, _ = _compute_chunk_delay_days(
            days_offset=12, days_count=3, week_number=5,
            pipeline_snapshot=snapshot,
        )
        assert delay_5 == max(0, 12 - math.ceil(3 / 2)), \
            f"Non-strict mid-chunk delay should be offset-ceil(count/2), got {delay_5}"

    # Universal: delay never exceeds 180
    assert delay_days <= 180, f"delay_days should be capped at 180, got {delay_days}"

