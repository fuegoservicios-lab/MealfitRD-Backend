"""[P0-A] Tests para telemetría de lecciones sintetizadas (low-confidence).

Cubre:
  1. `_record_chunk_lesson_telemetry` ejecuta el INSERT esperado.
  2. Wiring: cuando `_synthesize_last_chunk_learning_from_plan_days` devuelve dict
     no-None, el call site emite evento 'lesson_synthesized_low_confidence'.
  3. Wiring: cuando `_regenerate_recent_chunk_lessons_from_plan_days` devuelve una
     lista mezclada (queue + sintetizadas), el call site emite evento
     'recent_lessons_partial_synthesis' con counts correctos. Si no hay entradas
     sintetizadas, NO se emite (caso queue-only puro).
  4. `_alert_high_synthesized_lesson_ratio`:
       - No alerta si total_chunks < CHUNK_LESSON_SYNTH_MIN_SAMPLES (silencio).
       - No alerta si ratio < threshold.
       - Alerta si ratio >= threshold (con cooldown previo no-existente).
       - No alerta si existe row reciente en system_alerts (cooldown activo).

Ejecutar:
    cd backend && python -m pytest tests/test_p0_a_synthesized_telemetry.py -v
"""
import sys
import os
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_stub(module_name, **attrs):
    if module_name in sys.modules:
        return sys.modules[module_name]
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module
    return module


# Stubs mínimos para que `import cron_tasks` no rompa cuando se corre aislado.
if "supabase" not in sys.modules:
    _install_stub("supabase", Client=object, create_client=lambda *_a, **_kw: None)
if "dotenv" not in sys.modules:
    _install_stub("dotenv", load_dotenv=lambda *_a, **_kw: None)
if "langchain_google_genai" not in sys.modules:
    _install_stub(
        "langchain_google_genai",
        GoogleGenerativeAIEmbeddings=object,
        ChatGoogleGenerativeAI=object,
    )
_install_stub(
    "db_core",
    execute_sql_query=lambda *_a, **_kw: None,
    execute_sql_write=lambda *_a, **_kw: None,
    connection_pool=None,
)
_install_stub(
    "db_inventory",
    deduct_consumed_meal_from_inventory=lambda *_a, **_kw: None,
    # [test fix] Production `get_inventory_activity_since` returns Dict (see
    # db_inventory.py:810). Returning `[]` here caused AttributeError at
    # cron_tasks.py:11408 (`activity.get(...)`) for any subsequent test that
    # let _check_chunk_learning_ready proceed past the early-exit branches.
    get_inventory_activity_since=lambda *_a, **_kw: {
        "mutations_count": 0,
        "last_mutation_at": None,
        "low_stock_items": 0,
        "consumption_mutations_count": 0,
        "manual_mutations_count": 0,
    },
    get_raw_user_inventory=lambda *_a, **_kw: [],
    get_user_inventory_net=lambda *_a, **_kw: [],
    release_chunk_reservations=lambda *_a, **_kw: None,
    # [test fix] db_inventory.reserve_plan_ingredients returns int (count of items reserved).
    # Stubbing with a dict caused TypeError: '>=' not supported between dict and int in
    # cron_tasks.py:5239/16519. Mirror production: count ingredients with len>=3 in days[2].
    reserve_plan_ingredients=lambda *_a, **_kw: sum(
        1 for d in (_a[2] if len(_a) >= 3 else (_kw.get("days") or []))
        for m in ((d or {}).get("meals") or [])
        for i in (m.get("ingredients") or [])
        if i and len(str(i).strip()) >= 3
    ),
)
_install_stub(
    "db",
    get_latest_meal_plan_with_id=lambda *_a, **_kw: None,
    get_user_likes=lambda *_a, **_kw: [],
    get_active_rejections=lambda *_a, **_kw: [],
    get_recent_plans=lambda *_a, **_kw: [],
)
_install_stub(
    "db_facts",
    get_all_user_facts=lambda *_a, **_kw: [],
    get_consumed_meals_since=lambda *_a, **_kw: [],
    get_user_facts_by_metadata=lambda *_a, **_kw: [],
)
_install_stub("pydantic", BaseModel=object, Field=lambda default=None, **_kw: default)
_install_stub("schemas", HealthProfileSchema=object, ExpandedRecipeModel=object)
_install_stub("graph_orchestrator", run_plan_pipeline=lambda *_a, **_kw: {})
_install_stub("memory_manager", build_memory_context=lambda *_a, **_kw: "")
_install_stub("services", _save_plan_and_track_background=lambda *_a, **_kw: None)
_install_stub("agent", analyze_preferences_agent=lambda *_a, **_kw: {})


def _stub_parse_quantity(text, *_a, **_kw):
    return (1.0, "ud", str(text or ""))


# Sólo stubear shopping_calculator si NO se puede cargar el real. El stub rompe
# `validate_ingredients_against_pantry` (lo deja contando todo como 1ud) y eso
# poluciona tests posteriores como `test_p0_4_pantry_post_merge` que dependen
# de cantidades reales.
try:
    import shopping_calculator  # noqa: F401
except ImportError:
    _install_stub(
        "shopping_calculator",
        get_shopping_list_delta=lambda *_a, **_kw: [],
        _parse_quantity=_stub_parse_quantity,
    )
apscheduler_pkg = _install_stub("apscheduler")
apscheduler_triggers_pkg = _install_stub("apscheduler.triggers")
apscheduler_cron_pkg = _install_stub("apscheduler.triggers.cron", CronTrigger=object)
apscheduler_pkg.triggers = apscheduler_triggers_pkg
apscheduler_triggers_pkg.cron = apscheduler_cron_pkg


from unittest.mock import patch
import cron_tasks


# ---------------------------------------------------------------------------
# 1. Helper escribe el INSERT esperado
# ---------------------------------------------------------------------------
def test_record_helper_emits_insert_with_expected_columns():
    captured = {}

    def fake_write(query, params=None):
        captured["query"] = query
        captured["params"] = params
        return None

    with patch("cron_tasks.execute_sql_write", side_effect=fake_write):
        ok = cron_tasks._record_chunk_lesson_telemetry(
            user_id="11111111-1111-1111-1111-111111111111",
            meal_plan_id="22222222-2222-2222-2222-222222222222",
            week_number=3,
            event="lesson_synthesized_low_confidence",
            synthesized_count=1,
            queue_count=0,
            metadata={"prev_week": 2},
        )

    assert ok is True
    assert "INSERT INTO chunk_lesson_telemetry" in captured["query"]
    assert "synthesized_count" in captured["query"]
    assert "queue_count" in captured["query"]
    params = captured["params"]
    assert params[0] == "11111111-1111-1111-1111-111111111111"
    assert params[1] == "22222222-2222-2222-2222-222222222222"
    assert params[2] == 3
    assert params[3] == "lesson_synthesized_low_confidence"
    assert params[4] == 1
    assert params[5] == 0
    assert "prev_week" in params[6]


def test_record_helper_returns_false_on_db_failure():
    def boom(*_a, **_kw):
        raise RuntimeError("table missing")

    # [G8/STALE-FIX · 2026-05-29] user_id/meal_plan_id deben ser UUIDs válidos: el gate
    # P2-CHUNK-10 (cron_tasks.py:17472, añadido 2026-05-28) rechaza no-UUIDs ANTES del
    # execute_sql_write, así que con "u"/"p" la función retornaba False por el gate y nunca
    # ejercitaba el path de fallo de DB que este test pretende cubrir (el contador quedaba en 0).
    with patch("cron_tasks.execute_sql_write", side_effect=boom):
        ok = cron_tasks._record_chunk_lesson_telemetry(
            user_id="11111111-1111-1111-1111-111111111111",
            meal_plan_id="22222222-2222-2222-2222-222222222222",
            week_number=1,
            event="lesson_synthesized_low_confidence",
        )
    assert ok is False
    # El contador in-memory debe haber subido para detectar fallos sistémicos.
    assert cron_tasks._chunk_lesson_telemetry_failures["count"] >= 1
    # Reset para no contaminar otros tests.
    cron_tasks._chunk_lesson_telemetry_failures["count"] = 0
    cron_tasks._chunk_lesson_telemetry_failures["last_error"] = None


# ---------------------------------------------------------------------------
# 2. _regenerate_recent_chunk_lessons_from_plan_days: detección de mezcla
# ---------------------------------------------------------------------------
def _days_for_synthesis():
    """Días con tag de chunk para que la síntesis pueda extraer entradas por chunk."""
    return [
        {"week_number": 1, "meals": [
            {"name": "Pollo arroz", "ingredients": ["pollo", "arroz"], "status": "ok"},
        ]},
        {"week_number": 2, "meals": [
            {"name": "Tortilla", "ingredients": ["huevos"], "status": "ok"},
        ]},
    ]


def test_regenerate_returns_synthesized_when_no_seed():
    """Sin seed_lessons (queue NULL), todas las lecciones son sintetizadas."""
    res = cron_tasks._regenerate_recent_chunk_lessons_from_plan_days(
        meal_plan_id="p",
        plan_data={"days": _days_for_synthesis()},
        target_week=3,
        total_days_requested=15,
        seed_lessons=None,
    )
    # Debe haber al menos una lección sintetizada para week 1 y 2.
    synth_count = sum(
        1 for l in res
        if isinstance(l, dict) and l.get("synthesized_from_plan_days")
    )
    assert synth_count >= 2
    # Y todas son low_confidence cuando vienen de síntesis.
    for l in res:
        if l.get("synthesized_from_plan_days"):
            assert l.get("low_confidence") is True


def test_regenerate_preserves_seed_over_synthesis():
    """Si seed_lessons cubre un chunk, NO lo re-sintetizamos."""
    seed = [{
        "chunk": 1,
        "repeat_pct": 25,
        "ingredient_base_repeat_pct": 30,
        "rejection_violations": 0,
        "allergy_violations": 0,
        "fatigued_violations": 0,
        "repeated_bases": ["pollo"],
        "repeated_meal_names": ["Pollo arroz"],
        "rejected_meals_that_reappeared": [],
        "allergy_hits": [],
        "timestamp": "2026-01-01T00:00:00+00:00",
    }]
    res = cron_tasks._regenerate_recent_chunk_lessons_from_plan_days(
        meal_plan_id="p",
        plan_data={"days": _days_for_synthesis()},
        target_week=3,
        total_days_requested=15,
        seed_lessons=seed,
    )
    week1 = next((l for l in res if l.get("chunk") == 1), None)
    assert week1 is not None
    # Si proviene del seed (queue), NO debe estar marcado como sintetizado.
    assert not week1.get("synthesized_from_plan_days"), (
        "el seed (queue-based) debió tomar prioridad sobre la síntesis"
    )


# ---------------------------------------------------------------------------
# 3. _alert_high_synthesized_lesson_ratio: no fires below threshold
# ---------------------------------------------------------------------------
def _make_query_responder(stats_row=None, existing_alert=None):
    """Devuelve un fake `execute_sql_query` que responde según el query."""
    def fake_query(query, params=None, fetch_one=False, fetch_all=False):
        q = (query or "").strip()
        if "FROM chunk_lesson_telemetry" in q:
            return stats_row
        if "FROM system_alerts" in q:
            return existing_alert
        return None
    return fake_query


def test_alert_does_not_fire_below_min_samples():
    """total < MIN_SAMPLES → no se evalúa, no se inserta."""
    writes = []
    fake_query = _make_query_responder(
        stats_row={"synthesized_events": 5, "total_chunks": 3},  # < 10 default
    )

    def fake_write(query, params=None):
        writes.append((query, params))
        return None

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write):
        cron_tasks._alert_high_synthesized_lesson_ratio()

    inserts = [w for w in writes if "INSERT INTO system_alerts" in (w[0] or "")]
    assert inserts == [], "no debe disparar alerta con muestra insuficiente"


def test_alert_does_not_fire_below_threshold():
    """ratio 5/100 = 5% < 20% threshold → no inserta."""
    writes = []
    fake_query = _make_query_responder(
        stats_row={"synthesized_events": 5, "total_chunks": 100},
    )

    def fake_write(query, params=None):
        writes.append((query, params))
        return None

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write):
        cron_tasks._alert_high_synthesized_lesson_ratio()

    inserts = [w for w in writes if "INSERT INTO system_alerts" in (w[0] or "")]
    assert inserts == [], "ratio bajo no debe disparar alerta"


def test_alert_fires_above_threshold():
    """ratio 30/100 = 30% > 20% → inserta alert con metadata correcta."""
    writes = []
    fake_query = _make_query_responder(
        stats_row={"synthesized_events": 30, "total_chunks": 100},
        existing_alert=None,  # no hay row previa, cooldown libre
    )

    def fake_write(query, params=None):
        writes.append((query, params))
        return None

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write):
        cron_tasks._alert_high_synthesized_lesson_ratio()

    inserts = [w for w in writes if "INSERT INTO system_alerts" in (w[0] or "")]
    assert len(inserts) == 1, f"esperaba 1 INSERT a system_alerts, hubo {len(inserts)}"
    _query, params = inserts[0]
    # Posicional: (alert_key, title, message, metadata_json, affected_user_ids_json).
    # `alert_type` y `severity` están inlined como literales en el query.
    assert params[0] == "chunk_lesson_synth_ratio_high"
    metadata_json = params[3]
    assert "0.3" in metadata_json or "30" in metadata_json


def test_alert_respects_cooldown():
    """Si hay row reciente en system_alerts, no inserta nueva alerta."""
    writes = []
    fake_query = _make_query_responder(
        stats_row={"synthesized_events": 50, "total_chunks": 100},
        existing_alert={"triggered_at": "2026-05-01T10:00:00+00:00"},
    )

    def fake_write(query, params=None):
        writes.append((query, params))
        return None

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write):
        cron_tasks._alert_high_synthesized_lesson_ratio()

    inserts = [w for w in writes if "INSERT INTO system_alerts" in (w[0] or "")]
    assert inserts == [], "cooldown activo debe suprimir alerta"
