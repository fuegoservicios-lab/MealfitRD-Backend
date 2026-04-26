"""
P0-3: Tests que verifican la propagación real del aprendizaje continuo entre chunks.

Cubre los 4 caminos críticos que NO existían en el suite anterior:
  1. Escritura: _last_chunk_learning se persiste en plan_data después de que un chunk completa
  2. Lectura:   chunk N+1 recibe _chunk_lessons en form_data con datos del chunk N
  3. Ventana:   _recent_chunk_lessons se trunca correctamente (4 para 7d, 8 para 15d+)
  4. Lifetime:  _lifetime_lessons_summary acumula violaciones entre chunks

Ejecutar con:
    cd backend && python -m pytest tests/test_chunked_learning_propagation.py -v
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
from cron_tasks import process_plan_chunk_queue


# ---------------------------------------------------------------------------
# Helpers compartidos
# ---------------------------------------------------------------------------

def _make_tasks(week_number=2, days_offset=3, days_count=3, plan_id="plan_learning", extra_snapshot=None):
    snapshot = {"form_data": {"_plan_start_date": "2026-04-21T00:00:00+00:00"}}
    if extra_snapshot:
        snapshot.update(extra_snapshot)
    return [{
        "id": 1,
        "user_id": "user_learning",
        "meal_plan_id": plan_id,
        "week_number": week_number,
        "days_offset": days_offset,
        "days_count": days_count,
        "pipeline_snapshot": snapshot,
    }]


def _write_factory(tasks):
    def side_effect(query, *_args, **_kwargs):
        if "RETURNING" in query:
            return tasks
        return None
    return side_effect


def _query_factory(plan_data, tasks=None, user_profile=None):
    def side_effect(query, _params=None, _fetch_all=False, fetch_one=False, **_kwargs):
        res = None
        if "SELECT * FROM plan_chunk_queue" in query:
            res = tasks or []
        elif "generation_status" in query:
            res = {"id": "plan_learning", "status": "active", "plan_data": {"generation_status": "active"}}
        elif "SELECT plan_data FROM meal_plans" in query:
            res = {"plan_data": plan_data}
        elif "emergency_backup_plan" in query:
            res = {"backup": []}
        elif "SELECT health_profile FROM user_profiles" in query:
            res = {"health_profile": user_profile or {}}
        if res is not None:
            if fetch_one:
                return res[0] if isinstance(res, list) and res else res
            return res if isinstance(res, list) else [res]
        return None
    return side_effect


def _make_smart_cursor(prior_plan):
    """Cursor mock que devuelve plan_data o chunk-status según la última query ejecutada."""
    mock_cursor = MagicMock()
    last_query: list[str] = [""]

    def _track(query, *_a, **_kw):
        last_query[0] = query

    def _fetchone():
        if "plan_chunk_queue" in last_query[0] and "SELECT status" in last_query[0]:
            # TOCTOU check: el código espera {"status": "processing", "attempts": N}
            return {"status": "processing", "attempts": 0}
        return {"plan_data": prior_plan}

    mock_cursor.execute.side_effect = _track
    mock_cursor.fetchone.side_effect = _fetchone
    return mock_cursor


def _run_process(tasks, prior_plan, mock_pipeline_return, user_profile=None, extra_patches=None, inventory=None):
    """Ejecuta process_plan_chunk_queue con el mínimo de mocks necesarios y retorna mock_cursor."""
    mock_conn = MagicMock()
    mock_cursor = _make_smart_cursor(prior_plan)
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    patches = dict(
        mock_pool="db_core.connection_pool",
        mock_shop="shopping_calculator.get_shopping_list_delta",
        mock_pipeline="cron_tasks.run_plan_pipeline",
        mock_inventory="cron_tasks.get_user_inventory",
        mock_build_memory="cron_tasks.build_memory_context",
        mock_cron_likes="cron_tasks.get_user_likes",
        mock_db_likes="db.get_user_likes",
        mock_cron_rejs="cron_tasks.get_active_rejections",
        mock_db_rejs="db.get_active_rejections",
        mock_analyze="cron_tasks.analyze_preferences_agent",
        mock_build_facts="cron_tasks._build_facts_memory_context",
        mock_cron_facts="cron_tasks.get_all_user_facts",
        mock_db_facts="db_facts.get_all_user_facts",
        mock_cron_consumed="cron_tasks.get_consumed_meals_since",
        mock_db_consumed="db_facts.get_consumed_meals_since",
        mock_recent_plans="cron_tasks.get_recent_plans",
        mock_write="cron_tasks.execute_sql_write",
        mock_query="cron_tasks.execute_sql_query",
        mock_db_metadata="db_facts.get_user_facts_by_metadata",
        mock_learning_ready="cron_tasks._check_chunk_learning_ready",
    )
    if extra_patches:
        patches.update(extra_patches)

    active_patches = {k: patch(v) for k, v in patches.items()}
    mocks = {k: ctx.__enter__() for k, ctx in active_patches.items()}

    try:
        mocks["mock_pool"].connection.return_value.__enter__.return_value = mock_conn
        mocks["mock_shop"].return_value = {"categories": []}
        mocks["mock_pipeline"].return_value = mock_pipeline_return
        # Si se incluye mock_llm, forzar que el probe falle para mantener is_degraded=True
        if "mock_llm" in mocks:
            mocks["mock_llm"].return_value.invoke.side_effect = Exception("LLM offline - test")
        mocks["mock_inventory"].return_value = inventory if inventory is not None else ["pollo", "arroz", "avena"]
        mocks["mock_build_memory"].return_value = {"recent_messages": [], "full_context_str": "ctx"}
        mocks["mock_learning_ready"].return_value = {
            "ready": True, "ratio": 1.0, "matched_meals": 3, "planned_meals": 3
        }
        mocks["mock_cron_likes"].return_value = []
        mocks["mock_db_likes"].return_value = []
        mocks["mock_cron_rejs"].return_value = []
        mocks["mock_db_rejs"].return_value = []
        mocks["mock_analyze"].return_value = {}
        mocks["mock_cron_facts"].return_value = []
        mocks["mock_db_facts"].return_value = []
        mocks["mock_cron_consumed"].return_value = []
        mocks["mock_db_consumed"].return_value = []
        mocks["mock_recent_plans"].return_value = []
        mocks["mock_db_metadata"].return_value = []
        mocks["mock_write"].side_effect = _write_factory(tasks)
        mocks["mock_query"].side_effect = _query_factory(prior_plan, tasks=tasks, user_profile=user_profile)

        with patch("concurrent.futures.ThreadPoolExecutor") as mock_exec:
            def _sync_map(f, t):
                return [f(item) for item in t]
            def _sync_submit(fn, *args, **kwargs):
                fut = MagicMock()
                fut.result.return_value = fn(*args, **kwargs)
                return fut
            mock_exec.return_value.__enter__.return_value.map.side_effect = _sync_map
            mock_exec.return_value.__enter__.return_value.submit.side_effect = _sync_submit
            mock_exec.return_value.submit.side_effect = _sync_submit
            process_plan_chunk_queue()
    finally:
        for ctx in active_patches.values():
            ctx.__exit__(None, None, None)

    return mock_cursor, mocks


def _extract_merged_plan(mock_cursor):
    """Extrae el plan_data del UPDATE a meal_plans hecho vía cursor."""
    update_calls = [
        c for c in mock_cursor.execute.call_args_list
        if "UPDATE meal_plans SET plan_data =" in c[0][0]
    ]
    assert update_calls, "No se encontró UPDATE meal_plans SET plan_data ="
    raw = update_calls[0][0][1][0]
    return json.loads(raw) if isinstance(raw, str) else raw


# ---------------------------------------------------------------------------
# Test 1: escritura — _last_chunk_learning se persiste en plan_data
# ---------------------------------------------------------------------------

@patch("cron_tasks._calculate_learning_metrics")
def test_last_chunk_learning_is_persisted_to_plan_data_after_chunk_completes(mock_metrics):
    """Chunk 2 completa → plan_data debe contener _last_chunk_learning con los campos del chunk."""
    mock_metrics.return_value = {
        "learning_repeat_pct": 0.0,
        "ingredient_base_repeat_pct": 66.7,
        "rejection_violations": 1,
        "allergy_violations": 0,
        "fatigued_violations": 0,
        "prior_meals_count": 3,
        "prior_meal_bases_count": 3,
        "sample_repeated_bases": [{"bases": ["pollo"]}],
        "sample_repeats": [],
        "sample_rejection_hits": ["Pollo Asado"],
        "sample_allergy_hits": [],
    }

    tasks = _make_tasks(week_number=2, days_offset=3, days_count=3)
    prior_plan = {
        "total_days_requested": 7,
        "days": [
            {"day": 1, "meals": [{"name": "Pescado al horno", "ingredients": ["pescado", "limon"]}]},
            {"day": 2, "meals": [{"name": "Res guisada",      "ingredients": ["res", "tomate"]}]},
            {"day": 3, "meals": [{"name": "Quinoa bowl",      "ingredients": ["quinoa", "espinaca"]}]},
        ],
    }
    pipeline_return = {
        "days": [
            {"day": 4, "meals": [{"name": "Pollo al horno",    "ingredients": ["pollo", "ajo"]}]},
            {"day": 5, "meals": [{"name": "Pollo a la plancha","ingredients": ["pollo", "oregano"]}]},
            {"day": 6, "meals": [{"name": "Arroz con pollo",   "ingredients": ["pollo", "arroz"]}]},
        ]
    }

    cursor, _ = _run_process(tasks, prior_plan, pipeline_return)
    merged = _extract_merged_plan(cursor)

    lesson = merged.get("_last_chunk_learning")
    assert lesson is not None, "_last_chunk_learning no fue persistido en plan_data"
    assert lesson["chunk"] == 2, f"chunk esperado=2, obtenido={lesson['chunk']}"
    assert lesson["ingredient_base_repeat_pct"] == 66.7
    assert lesson["rejection_violations"] == 1
    assert lesson.get("metrics_unavailable") is False
    assert any("pollo" in str(rb) for rb in lesson.get("repeated_bases", [])), \
        "repeated_bases debería contener 'pollo'"


# ---------------------------------------------------------------------------
# Test 2: lectura — chunk N+1 recibe _chunk_lessons con datos del chunk anterior
# ---------------------------------------------------------------------------

def test_chunk_n_plus_1_receives_chunk_lessons_from_prior_plan():
    """Si plan_data tiene _last_chunk_learning con repeated_bases, el form_data del
    siguiente chunk debe incluir _chunk_lessons con esos datos para que el LLM los use."""
    tasks = _make_tasks(week_number=3, days_offset=6, days_count=3)
    prior_plan = {
        "total_days_requested": 9,
        "_last_chunk_learning": {
            "chunk": 2,
            "repeated_bases": [{"bases": ["pollo"]}],
            "ingredient_base_repeat_pct": 80.0,
            "rejection_violations": 2,
            "allergy_violations": 0,
            "repeat_pct": 50.0,
            "repeated_meal_names": ["Pollo Asado"],
            "rejected_meals_that_reappeared": ["Pollo Frito"],
            "allergy_hits": [],
            "metrics_unavailable": False,
        },
        "days": [
            {"day": 1, "meals": [{"name": "A"}]},
            {"day": 2, "meals": [{"name": "B"}]},
            {"day": 3, "meals": [{"name": "C"}]},
            {"day": 4, "meals": [{"name": "D"}]},
            {"day": 5, "meals": [{"name": "E"}]},
            {"day": 6, "meals": [{"name": "F"}]},
        ],
    }
    pipeline_return = {
        "days": [
            {"day": 7, "meals": [{"name": "G"}]},
            {"day": 8, "meals": [{"name": "H"}]},
            {"day": 9, "meals": [{"name": "I"}]},
        ]
    }

    _, mocks = _run_process(tasks, prior_plan, pipeline_return)

    mocks["mock_pipeline"].assert_called_once()
    form_data = mocks["mock_pipeline"].call_args[0][0]

    lessons = form_data.get("_chunk_lessons")
    assert lessons is not None, "_chunk_lessons no fue inyectado en form_data del chunk N+1"
    assert lessons["rejection_violations"] == 2, \
        f"rejection_violations esperado=2, obtenido={lessons['rejection_violations']}"
    repeated_bases_flat = " ".join(str(rb) for rb in lessons.get("repeated_bases", []))
    assert "pollo" in repeated_bases_flat, \
        f"'pollo' debería estar en repeated_bases, obtenido: {lessons.get('repeated_bases')}"
    assert "Pollo Frito" in lessons.get("rejected_meals_that_reappeared", []), \
        "rejected_meals_that_reappeared debería incluir 'Pollo Frito'"


# ---------------------------------------------------------------------------
# Test 3: ventana rolling — se trunca correctamente según total_days_requested
# ---------------------------------------------------------------------------

@patch("cron_tasks._calculate_learning_metrics")
def test_recent_chunk_lessons_rolling_window_caps_at_4_for_7d_plan(mock_metrics):
    """Para plan de 7 días, _recent_chunk_lessons no debe superar 4 entradas."""
    mock_metrics.return_value = {
        "learning_repeat_pct": 0.0,
        "ingredient_base_repeat_pct": 0.0,
        "rejection_violations": 0,
        "allergy_violations": 0,
        "fatigued_violations": 0,
        "prior_meals_count": 3,
        "prior_meal_bases_count": 3,
        "sample_repeated_bases": [],
        "sample_repeats": [],
        "sample_rejection_hits": [],
        "sample_allergy_hits": [],
    }

    existing_lessons = [
        {"chunk": i, "repeat_pct": 0.0, "ingredient_base_repeat_pct": 0.0,
         "rejection_violations": 0, "allergy_violations": 0, "metrics_unavailable": False}
        for i in range(1, 6)  # 5 lecciones ya presentes
    ]
    tasks = _make_tasks(week_number=6, days_offset=5, days_count=2)
    prior_plan = {
        "total_days_requested": 7,
        "_recent_chunk_lessons": existing_lessons,
        "days": [{"day": i, "meals": [{"name": f"Plato {i}"}]} for i in range(1, 6)],
    }
    pipeline_return = {
        "days": [
            {"day": 6, "meals": [{"name": "Nuevo A"}]},
            {"day": 7, "meals": [{"name": "Nuevo B"}]},
        ]
    }

    cursor, _ = _run_process(tasks, prior_plan, pipeline_return)
    merged = _extract_merged_plan(cursor)

    recent = merged.get("_recent_chunk_lessons", [])
    assert len(recent) <= 4, \
        f"Plan 7d: ventana rolling debe ser ≤4, tiene {len(recent)} entradas"
    # El chunk recién procesado debe ser el último de la ventana
    assert recent[-1]["chunk"] == 6, \
        f"El chunk 6 debería ser el último en la ventana, obtenido chunk={recent[-1]['chunk']}"


@patch("cron_tasks._calculate_learning_metrics")
def test_recent_chunk_lessons_rolling_window_caps_at_8_for_15d_plan(mock_metrics):
    """Para plan de 15 días, _recent_chunk_lessons no debe superar 8 entradas."""
    mock_metrics.return_value = {
        "learning_repeat_pct": 0.0,
        "ingredient_base_repeat_pct": 0.0,
        "rejection_violations": 0,
        "allergy_violations": 0,
        "fatigued_violations": 0,
        "prior_meals_count": 3,
        "prior_meal_bases_count": 3,
        "sample_repeated_bases": [],
        "sample_repeats": [],
        "sample_rejection_hits": [],
        "sample_allergy_hits": [],
    }

    existing_lessons = [
        {"chunk": i, "repeat_pct": 0.0, "ingredient_base_repeat_pct": 0.0,
         "rejection_violations": 0, "allergy_violations": 0, "metrics_unavailable": False}
        for i in range(1, 10)  # 9 lecciones ya presentes
    ]
    tasks = _make_tasks(week_number=10, days_offset=9, days_count=3, plan_id="plan_15d")
    prior_plan = {
        "total_days_requested": 15,
        "_recent_chunk_lessons": existing_lessons,
        "days": [{"day": i, "meals": [{"name": f"Plato {i}"}]} for i in range(1, 10)],
    }
    pipeline_return = {
        "days": [{"day": i, "meals": [{"name": f"Nuevo {i}"}]} for i in range(10, 13)]
    }

    cursor, _ = _run_process(tasks, prior_plan, pipeline_return)
    merged = _extract_merged_plan(cursor)

    recent = merged.get("_recent_chunk_lessons", [])
    assert len(recent) <= 8, \
        f"Plan 15d: ventana rolling debe ser ≤8, tiene {len(recent)} entradas"
    assert recent[-1]["chunk"] == 10, \
        f"El chunk 10 debería ser el último en la ventana, obtenido chunk={recent[-1]['chunk']}"


# ---------------------------------------------------------------------------
# Test 4: lifetime summary — acumula violaciones entre chunks
# ---------------------------------------------------------------------------

@patch("cron_tasks._calculate_learning_metrics")
def test_lifetime_summary_accumulates_violation_counts_across_chunks(mock_metrics):
    """_lifetime_lessons_summary.total_rejection_violations debe acumular (no resetear)
    después de cada chunk. Si ya había 3 y el nuevo chunk tiene 2, debe quedar en 5."""
    mock_metrics.return_value = {
        "learning_repeat_pct": 0.0,
        "ingredient_base_repeat_pct": 0.0,
        "rejection_violations": 2,
        "allergy_violations": 1,
        "fatigued_violations": 0,
        "sample_repeated_bases": [],
        "sample_repeats": [],
        "sample_rejection_hits": ["Pollo Frito"],
        "sample_allergy_hits": ["maní"],
    }

    tasks = _make_tasks(week_number=3, days_offset=6, days_count=3)
    prior_plan = {
        "total_days_requested": 9,
        "_lifetime_lessons_summary": {
            "total_rejection_violations": 3,
            "total_allergy_violations": 2,
            "top_rejection_hits": ["Res al Horno"],
            "top_repeated_bases": ["res"],
        },
        "days": [{"day": i, "meals": [{"name": f"Plato {i}"}]} for i in range(1, 7)],
    }
    pipeline_return = {
        "days": [{"day": i, "meals": [{"name": f"Nuevo {i}"}]} for i in range(7, 10)]
    }

    cursor, _ = _run_process(tasks, prior_plan, pipeline_return)
    merged = _extract_merged_plan(cursor)

    lifetime = merged.get("_lifetime_lessons_summary")
    assert lifetime is not None, "_lifetime_lessons_summary no fue persistido"
    assert lifetime["total_rejection_violations"] == 5, \
        f"Esperado 3+2=5 rejection_violations, obtenido {lifetime['total_rejection_violations']}"
    assert lifetime["total_allergy_violations"] == 3, \
        f"Esperado 2+1=3 allergy_violations, obtenido {lifetime['total_allergy_violations']}"
    # Los hits nuevos se deben acumular junto con los existentes
    assert "Pollo Frito" in lifetime["top_rejection_hits"], \
        "top_rejection_hits debe incluir la nueva violación 'Pollo Frito'"
    assert "Res al Horno" in lifetime["top_rejection_hits"], \
        "top_rejection_hits debe preservar la violación preexistente 'Res al Horno'"


# ---------------------------------------------------------------------------
# Test P0-1: Smart Shuffle respeta las bases aprendidas
# ---------------------------------------------------------------------------

def test_smart_shuffle_excludes_high_fatigue_days_using_learned_bases():
    """En modo degraded, el Smart Shuffle debe excluir días cuyos ingredientes
    coincidan con bases marcadas como repetitivas en _last_chunk_learning.

    Setup:
      - prior_plan tiene _last_chunk_learning con repeated_bases=["pollo"]
      - El pool tiene 2 días con pollo y 2 días sin pollo
      - Con days_count=2, el shuffle debería elegir SOLO los días sin pollo
    """
    # Días con pollo en nombre / ingredientes
    day_pollo_1 = {
        "day": 1,
        "meals": [{"name": "Pollo Guisado", "ingredients": ["200g pollo", "tomate"]}],
    }
    day_pollo_2 = {
        "day": 2,
        "meals": [{"name": "Pollo a la Plancha", "ingredients": ["pollo", "oregano"]}],
    }
    # Días sin pollo
    day_res = {
        "day": 3,
        "meals": [{"name": "Res Guisada", "ingredients": ["200g carne de res", "papa"]}],
    }
    day_pescado = {
        "day": 4,
        "meals": [{"name": "Pescado al horno", "ingredients": ["salmon", "limon"]}],
    }

    tasks = _make_tasks(
        week_number=2,
        days_offset=4,
        days_count=2,
        extra_snapshot={"_degraded": True},
    )
    prior_plan = {
        "total_days_requested": 6,
        "_last_chunk_learning": {
            "chunk": 1,
            "repeated_bases": [{"bases": ["pollo"]}],
            "ingredient_base_repeat_pct": 80.0,
            "rejection_violations": 0,
            "allergy_violations": 0,
            "repeat_pct": 0.0,
            "metrics_unavailable": False,
        },
        "days": [day_pollo_1, day_pollo_2, day_res, day_pescado],
    }

    # El probe LLM debe fallar para que is_degraded se mantenga True
    extra = {"mock_llm": "langchain_google_genai.ChatGoogleGenerativeAI"}
    cursor, mocks = _run_process(tasks, prior_plan, mock_pipeline_return={}, extra_patches=extra)

    # El pipeline nunca se llama en modo degraded
    mocks["mock_pipeline"].assert_not_called()

    # Los 2 días generados deben ser los días sin pollo
    merged = _extract_merged_plan(cursor)
    new_day_names = [
        meal["name"]
        for d in merged["days"][4:]         # días 5 y 6 (offset=4, days_count=2)
        for meal in d.get("meals", [])
    ]
    for name in new_day_names:
        assert "pollo" not in name.lower(), (
            f"El Smart Shuffle incluyó un día de pollo ({name!r}) "
            f"pese a que 'pollo' está marcado como base fatigada."
        )


# ---------------------------------------------------------------------------
# Test 7 (P1-1): hybrid quantity mode — reintentar y anotar, nunca fallar
# ---------------------------------------------------------------------------

def test_hybrid_mode_retries_on_quantity_violation_then_annotates():
    """
    Hybrid: LLM solicita 250g pollo con solo 100g en despensa (>130% límite).
    Debe reintentar _PANTRY_MAX_RETRIES veces (pipeline llamado 3x) y luego
    anotar la violación en _pantry_quantity_violations en lugar de fallar el chunk.
    """
    tasks = _make_tasks(week_number=2, days_offset=3, days_count=3)
    prior_plan = {
        "total_days_requested": 7,
        "days": [
            {"day": 1, "meals": [{"name": "Arroz con vegetales", "ingredients": ["arroz", "tomate"]}]},
            {"day": 2, "meals": [{"name": "Ensalada de quinoa",  "ingredients": ["quinoa", "lechuga"]}]},
            {"day": 3, "meals": [{"name": "Sopa de lentejas",    "ingredients": ["lentejas", "zanahoria"]}]},
        ],
    }
    # Pantry: 100g pollo. Pipeline siempre pide 250g pollo → excede 1.30x (130g) en cada intento.
    pantry = ["100g pollo"]
    pipeline_return = {
        "days": [
            {"day": 4, "meals": [{"name": "Pollo al horno",  "ingredients": ["250g pollo", "ajo"]}]},
            {"day": 5, "meals": [{"name": "Pollo guisado",   "ingredients": ["250g pollo", "cebolla"]}]},
            {"day": 6, "meals": [{"name": "Arroz con pollo", "ingredients": ["250g pollo", "arroz"]}]},
        ]
    }

    with patch("cron_tasks.CHUNK_PANTRY_QUANTITY_MODE", "hybrid"):
        cursor, mocks = _run_process(tasks, prior_plan, pipeline_return, inventory=pantry)

    # hybrid reintenta hasta agotar _PANTRY_MAX_RETRIES=2 → 3 llamadas totales al pipeline
    assert mocks["mock_pipeline"].call_count == 3, (
        f"Hybrid mode debe llamar el pipeline 3 veces (1 inicial + 2 reintentos), "
        f"obtenido: {mocks['mock_pipeline'].call_count}"
    )

    # Tras agotar reintentos, hybrid acepta con anotación en lugar de fallar
    merged = _extract_merged_plan(cursor)
    assert "_pantry_quantity_violations" in merged, (
        "Hybrid mode debe anotar _pantry_quantity_violations cuando se agotan los reintentos"
    )
    assert len(merged.get("days", [])) == 6, "El plan debe continuar con los 6 días generados"


def test_advisory_mode_does_not_retry_on_quantity_violation():
    """
    Advisory: misma violación de cantidades que en hybrid, pero el pipeline se llama
    una sola vez — anota inmediatamente sin reintentar.
    """
    tasks = _make_tasks(week_number=2, days_offset=3, days_count=3)
    prior_plan = {
        "total_days_requested": 7,
        "days": [
            {"day": 1, "meals": [{"name": "Arroz con vegetales", "ingredients": ["arroz", "tomate"]}]},
            {"day": 2, "meals": [{"name": "Ensalada de quinoa",  "ingredients": ["quinoa", "lechuga"]}]},
            {"day": 3, "meals": [{"name": "Sopa de lentejas",    "ingredients": ["lentejas", "zanahoria"]}]},
        ],
    }
    pantry = ["100g pollo"]
    pipeline_return = {
        "days": [
            {"day": 4, "meals": [{"name": "Pollo al horno",  "ingredients": ["250g pollo", "ajo"]}]},
            {"day": 5, "meals": [{"name": "Pollo guisado",   "ingredients": ["250g pollo", "cebolla"]}]},
            {"day": 6, "meals": [{"name": "Arroz con pollo", "ingredients": ["250g pollo", "arroz"]}]},
        ]
    }

    with patch("cron_tasks.CHUNK_PANTRY_QUANTITY_MODE", "advisory"):
        cursor, mocks = _run_process(tasks, prior_plan, pipeline_return, inventory=pantry)

    # advisory nunca reintenta → solo 1 llamada al pipeline
    assert mocks["mock_pipeline"].call_count == 1, (
        f"Advisory mode no debe reintentar, esperado 1 llamada, obtenido: {mocks['mock_pipeline'].call_count}"
    )

    # También anota la violación
    merged = _extract_merged_plan(cursor)
    assert "_pantry_quantity_violations" in merged
