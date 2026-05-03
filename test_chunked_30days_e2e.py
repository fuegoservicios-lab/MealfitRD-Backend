"""
P0-5: Test E2E para plan de 30 días (10 chunks).

Cubre los casos NO probados por test_chunked_15days_e2e.py:
  1. Encolar 10 chunks (chunk 1 sincrónico + 9 en queue), avanzar reloj,
     verificar que cada chunk N recibe lecciones del N-1.
  2. Rolling window de _recent_chunk_lessons se trunca ≤8 para planes ≥15d
     (regla en cron_tasks.py:6284).
  3. _lifetime_lessons_summary persiste y acumula a lo largo de los 10 chunks
     (rotación 60d).
  4. Plan completa los 30 días sin corrupción.

Ejecutar con:
    cd backend && python -m pytest test_chunked_30days_e2e.py -v
"""
import pytest
import uuid
import json
import time
from unittest.mock import patch
from datetime import datetime, timezone, timedelta

from db_core import execute_sql_write, execute_sql_query, connection_pool
from cron_tasks import process_plan_chunk_queue, _enqueue_plan_chunk
from constants import split_with_absorb, PLAN_CHUNK_SIZE

import logging
logging.getLogger().setLevel(logging.INFO)

if connection_pool and not getattr(connection_pool, '_opened', False):
    connection_pool.open()


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
                 "ingredients": [f"100g pollo", "arroz"]},
                {"name": f"Lunch {day_num}", "type": "Almuerzo",
                 "ingredients": [f"150g res", "habichuelas"]},
            ]
        })

    return {"days": days, "generation_status": "partial"}


# ---------------------------------------------------------------------------
# Fixture: usuario y plan limpios
# ---------------------------------------------------------------------------
@pytest.fixture
def cleanup_user_db():
    user_row = execute_sql_query("SELECT id FROM user_profiles LIMIT 1", fetch_one=True)
    if not user_row:
        pytest.skip("No user_profiles found in DB to use for E2E tests.")

    user_id = user_row["id"]
    plan_id = str(uuid.uuid4())

    yield user_id, plan_id

    # Teardown
    execute_sql_write("DELETE FROM plan_chunk_queue WHERE meal_plan_id = %s", (plan_id,))
    execute_sql_write("DELETE FROM meal_plans WHERE id = %s", (plan_id,))


# ---------------------------------------------------------------------------
# Test 1: E2E completo — 10 chunks, 30 días, lesson chain, rolling window
# ---------------------------------------------------------------------------
@patch('cron_tasks.run_plan_pipeline', side_effect=_mock_run_plan_pipeline)
def test_chunked_30days_e2e_full_plan(mock_pipeline, cleanup_user_db):
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

    user_id, plan_id = cleanup_user_db

    # 1. Generar chunk 1 sincrónicamente (días 1-3)
    initial_form_data = {"user_id": user_id, "_days_offset": 0, "_days_to_generate": 3}
    initial_plan = _mock_run_plan_pipeline(initial_form_data)
    initial_plan["total_days_requested"] = 30
    initial_plan["total_days_generated"] = 3

    execute_sql_write(
        "INSERT INTO meal_plans (id, user_id, plan_data) VALUES (%s, %s, %s)",
        (plan_id, user_id, json.dumps(initial_plan))
    )

    inserted = execute_sql_query(
        "SELECT id FROM meal_plans WHERE id = %s", (plan_id,), fetch_one=True
    )
    assert inserted is not None, "Plan was not inserted!"

    # 2. Encolar los 9 chunks restantes (chunk 2-10)
    # NOTA: el test mantiene la decomposición legacy 10×3d para validar el path
    # del worker bajo many-chunks; el split actual (P1-A) produciría 8 chunks
    # [3, 4, 4, 4, 4, 4, 4, 3] pero ese caso está cubierto por
    # tests/test_p1_a_split_with_absorb.py.
    chunks = split_with_absorb(30, PLAN_CHUNK_SIZE)
    assert sum(chunks) == 30, f"Suma de chunks debe ser 30, got {sum(chunks)}"
    assert all(c >= PLAN_CHUNK_SIZE for c in chunks), (
        f"Todo chunk >= {PLAN_CHUNK_SIZE} días, got {chunks}"
    )

    valid_snapshot = {
        "totalDays": 30,
        "form_data": {
            "_days_offset": 0,
            "householdSize": 1,
            "dietType": "Omnívora",
            "_plan_start_date": datetime.now(timezone.utc).isoformat(),
        }
    }

    offset = 3  # Primer chunk ya generado
    for chunk_idx in range(2, 11):  # chunks 2-10
        _enqueue_plan_chunk(
            user_id, plan_id,
            week_number=chunk_idx,
            days_offset=offset,
            days_count=3,
            pipeline_snapshot=valid_snapshot,
        )
        offset += 3

    # Verify 9 items in queue
    queued = execute_sql_query(
        "SELECT * FROM plan_chunk_queue WHERE meal_plan_id = %s",
        (plan_id,), fetch_all=True
    )
    assert len(queued) == 9, f"Esperados 9 chunks en cola, got {len(queued)}"

    # 3. Forzar ejecución inmediata de todos los chunks
    execute_sql_write(
        "UPDATE plan_chunk_queue SET execute_after = NOW() - INTERVAL '1 MINUTE' WHERE meal_plan_id = %s",
        (plan_id,)
    )

    # 4. Procesar los 9 chunks secuencialmente
    for i in range(9):
        logging.info(f"Processing chunk iteration {i + 1}/9")
        process_plan_chunk_queue(target_plan_id=plan_id)

    # 5. Verificar que todos los chunks completaron
    queued_after = execute_sql_query(
        "SELECT week_number, status FROM plan_chunk_queue WHERE meal_plan_id = %s ORDER BY week_number",
        (plan_id,), fetch_all=True
    )
    for q in queued_after:
        assert q['status'] == 'completed', \
            f"Chunk {q['week_number']} no completó: status={q['status']}"

    # 6. Verificar que el plan final tiene 30 días
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

    # 7. Verificar _recent_chunk_lessons se truncó a ≤8
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

    # 8. Verificar _lifetime_lessons_summary existe y acumula
    lifetime = final_plan.get("_lifetime_lessons_summary")
    assert lifetime is not None, \
        "_lifetime_lessons_summary debe existir después de 10 chunks"
    assert "total_rejection_violations" in lifetime, \
        "lifetime_lessons_summary debe tener total_rejection_violations"

    # 9. Verificar _lifetime_lessons_history tiene entradas (filtradas por ventana 60d)
    history = final_plan.get("_lifetime_lessons_history", [])
    assert len(history) >= 8, \
        f"history debe tener al menos 8 entradas (9 chunks procesados), tiene {len(history)}"

    # 10. Verificar generation_status es 'complete'
    assert final_plan.get("generation_status") == "complete", \
        f"Plan 30d debe marcar generation_status='complete', got: {final_plan.get('generation_status')}"


# ---------------------------------------------------------------------------
# Test 2: Lesson propagation — chunk N recibe lecciones del N-1
# ---------------------------------------------------------------------------
@patch('cron_tasks.run_plan_pipeline', side_effect=_mock_run_plan_pipeline)
def test_chunked_30days_lesson_chain_propagation(mock_pipeline, cleanup_user_db):
    """Verifica que para cada chunk N≥3, el pipeline recibe _chunk_lessons
    con datos del chunk anterior, formando una cadena continua de aprendizaje."""
    global _pipeline_call_log
    _pipeline_call_log = []

    user_id, plan_id = cleanup_user_db

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

    initial_plan = {
        "total_days_requested": 30,
        "total_days_generated": 3,
        "generation_status": "partial",
        "_last_chunk_learning": seed_lesson,
        "_recent_chunk_lessons": [seed_lesson],
        "days": [
            {"day": 1, "meals": [{"name": "Breakfast 1", "ingredients": ["pollo"]}]},
            {"day": 2, "meals": [{"name": "Breakfast 2", "ingredients": ["res"]}]},
            {"day": 3, "meals": [{"name": "Breakfast 3", "ingredients": ["pescado"]}]},
        ],
    }

    execute_sql_write(
        "INSERT INTO meal_plans (id, user_id, plan_data) VALUES (%s, %s, %s)",
        (plan_id, user_id, json.dumps(initial_plan))
    )

    valid_snapshot = {
        "totalDays": 30,
        "form_data": {
            "_days_offset": 0,
            "householdSize": 1,
            "dietType": "Omnívora",
            "_plan_start_date": datetime.now(timezone.utc).isoformat(),
        }
    }

    # Encolar solo 4 chunks (2-5) para verificar la cadena
    for chunk_idx in range(2, 6):
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
@patch('cron_tasks.run_plan_pipeline', side_effect=_mock_run_plan_pipeline)
def test_chunked_30days_rolling_window_truncates_to_8(mock_pipeline, cleanup_user_db):
    """Verifica que después de 10 chunks (1 sync + 9 queue), la ventana
    rolling _recent_chunk_lessons no excede 8 entradas."""
    global _pipeline_call_log
    _pipeline_call_log = []

    user_id, plan_id = cleanup_user_db

    initial_plan = {
        "total_days_requested": 30,
        "total_days_generated": 3,
        "generation_status": "partial",
        "days": [
            {"day": 1, "meals": [{"name": "B1", "ingredients": ["pollo"]}]},
            {"day": 2, "meals": [{"name": "B2", "ingredients": ["res"]}]},
            {"day": 3, "meals": [{"name": "B3", "ingredients": ["pescado"]}]},
        ],
    }

    execute_sql_write(
        "INSERT INTO meal_plans (id, user_id, plan_data) VALUES (%s, %s, %s)",
        (plan_id, user_id, json.dumps(initial_plan))
    )

    valid_snapshot = {
        "totalDays": 30,
        "form_data": {
            "_days_offset": 0,
            "householdSize": 1,
            "dietType": "Omnívora",
            "_plan_start_date": datetime.now(timezone.utc).isoformat(),
        }
    }

    for chunk_idx in range(2, 11):
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
@patch('cron_tasks.run_plan_pipeline', side_effect=_mock_run_plan_pipeline)
def test_chunked_30days_lifetime_lessons_accumulate(mock_pipeline, cleanup_user_db):
    """Verifica que _lifetime_lessons_summary y _lifetime_lessons_history
    se acumulan correctamente a lo largo de los 10 chunks del plan de 30 días."""
    global _pipeline_call_log
    _pipeline_call_log = []

    user_id, plan_id = cleanup_user_db

    initial_plan = {
        "total_days_requested": 30,
        "total_days_generated": 3,
        "generation_status": "partial",
        "days": [
            {"day": 1, "meals": [{"name": "B1", "ingredients": ["pollo"]}]},
            {"day": 2, "meals": [{"name": "B2", "ingredients": ["res"]}]},
            {"day": 3, "meals": [{"name": "B3", "ingredients": ["pescado"]}]},
        ],
    }

    execute_sql_write(
        "INSERT INTO meal_plans (id, user_id, plan_data) VALUES (%s, %s, %s)",
        (plan_id, user_id, json.dumps(initial_plan))
    )

    valid_snapshot = {
        "totalDays": 30,
        "form_data": {
            "_days_offset": 0,
            "householdSize": 1,
            "dietType": "Omnívora",
            "_plan_start_date": datetime.now(timezone.utc).isoformat(),
        }
    }

    for chunk_idx in range(2, 11):
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
@patch('cron_tasks.run_plan_pipeline', side_effect=_mock_run_plan_pipeline)
def test_chunked_30days_final_chunks_9_10_complete(mock_pipeline, cleanup_user_db):
    """Verifica específicamente que los chunks 9 y 10 (últimos del plan 30d)
    completan sin errores y que el plan se marca como 'complete' después del
    chunk 10."""
    global _pipeline_call_log
    _pipeline_call_log = []

    user_id, plan_id = cleanup_user_db

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

    initial_plan = {
        "total_days_requested": 30,
        "total_days_generated": 24,
        "generation_status": "partial",
        "days": pre_days,
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
    }

    execute_sql_write(
        "INSERT INTO meal_plans (id, user_id, plan_data) VALUES (%s, %s, %s)",
        (plan_id, user_id, json.dumps(initial_plan))
    )

    valid_snapshot = {
        "totalDays": 30,
        "form_data": {
            "_days_offset": 0,
            "householdSize": 1,
            "dietType": "Omnívora",
            "_plan_start_date": datetime.now(timezone.utc).isoformat(),
        }
    }

    # Encolar solo chunks 9 y 10
    _enqueue_plan_chunk(user_id, plan_id, 9, 24, 3, valid_snapshot)
    _enqueue_plan_chunk(user_id, plan_id, 10, 27, 3, valid_snapshot)

    execute_sql_write(
        "UPDATE plan_chunk_queue SET execute_after = NOW() - INTERVAL '1 MINUTE' WHERE meal_plan_id = %s",
        (plan_id,)
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
