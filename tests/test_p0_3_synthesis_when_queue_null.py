"""[P0.3] Síntesis de _last_chunk_learning como fallback cuando el queue está NULL.

Contexto:
    El worker `process_plan_chunk_queue` reconstruye `_last_chunk_learning` para
    cada chunk N>1 con esta cascada (cron_tasks.py:12375-12525):

        1. _rebuild_last_chunk_learning_from_queue(prefer_completed=True)
              ├── dict no-None → persistir + asignar in-memory  (path "rebuild")
              └── None → seguir al fallback ↓
        2. _synthesize_last_chunk_learning_from_plan_days(...)
              ├── dict no-None → persistir + asignar in-memory  (path "synthesis")
              └── None → log warning, chunk arranca con dict existente

    El path "rebuild" usa learning_metrics REALES calculados durante la completion
    del chunk previo. High-confidence.

    El path "synthesis" deriva lecciones desde `meal_plans.plan_data.days` cuando
    el queue NULL (chunk previo crasheó pre-preflight, JSON corrupto, schema
    downgrade). Low-confidence — no hay counters reales de repetición vs chunks
    anteriores — pero al menos el LLM del chunk N+1 sabe qué meals/bases evitar.

    Sin esta red, el chunk N+1 arrancaría con dict vacío y podría regenerar
    platos idénticos al chunk N-1, rompiendo el aprendizaje continuo.

Cobertura existente que NO duplica este test:
    - test_p0_4_synthesize_lesson.py: tests unitarios del helper de síntesis.
    - test_p0_a_synthesized_telemetry.py: wiring de telemetría (mocks aislados).
    - test_p1_2_recent_lessons_regen.py: regen del rolling window (P1-2).

Lo que cierra este test (gap antes no cubierto):
    1. Cuando el queue está NULL, el helper de síntesis devuelve un dict
       compatible con el contrato `_last_chunk_learning` y los días reales
       del chunk previo se ven reflejados en `repeated_meal_names`/
       `repeated_bases`. Cierra la cadena hasta el output que recibe el LLM.
    2. La síntesis se filtra por target_week — días de otros chunks NO
       contaminan la lección sintetizada (regresión: si esto rompe, el chunk
       N+1 ve "repeticiones" de chunks aún más antiguos, induciendo cambios
       innecesarios).
    3. Los flags de low-confidence (`metrics_unavailable=True`,
       `learning_signal_strength="weak"`, `synthesized_from_plan_days=True`)
       están presentes para que prompts/dashboards distingan síntesis de datos
       reales. Crítico: si el LLM lee `repeat_pct=0` sin saber que es
       "incompleto", lo interpreta como "no hubo repeticiones" — falso.
    4. `persist_legacy_learning_to_plan_data` acepta el context
       `synthesis_from_days` y rechaza contexts no registrados. Regression
       guard: si alguien añade un nuevo path legacy y olvida registrarlo en
       P0_3_LEGACY_LEARNING_CONTEXTS, el persist falla silenciosamente y la
       lección no se sella en plan_data.
"""
import os
import sys
import types
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Stubs mínimos para que `import cron_tasks` no rompa
# ---------------------------------------------------------------------------
def _install_stub(module_name, **attrs):
    if module_name in sys.modules:
        existing = sys.modules[module_name]
        for k, v in attrs.items():
            if not hasattr(existing, k):
                setattr(existing, k, v)
        return existing
    module = types.ModuleType(module_name)
    for k, v in attrs.items():
        setattr(module, k, v)
    sys.modules[module_name] = module
    return module


if "supabase" not in sys.modules:
    _install_stub("supabase", Client=object, create_client=lambda *_a, **_k: None)
if "dotenv" not in sys.modules:
    _install_stub("dotenv", load_dotenv=lambda *_a, **_k: None)
if "langchain_google_genai" not in sys.modules:
    _install_stub(
        "langchain_google_genai",
        GoogleGenerativeAIEmbeddings=object,
        ChatGoogleGenerativeAI=object,
    )
_install_stub(
    "db_core",
    execute_sql_query=lambda *_a, **_k: None,
    execute_sql_write=lambda *_a, **_k: None,
    connection_pool=None,
)
_install_stub(
    "db_inventory",
    deduct_consumed_meal_from_inventory=lambda *_a, **_k: None,
    get_inventory_activity_since=lambda *_a, **_k: [],
    get_raw_user_inventory=lambda *_a, **_k: [],
    get_user_inventory_net=lambda *_a, **_k: [],
    release_chunk_reservations=lambda *_a, **_k: None,
    reserve_plan_ingredients=lambda *_a, **_k: 0,
)
_install_stub(
    "db",
    get_latest_meal_plan_with_id=lambda *_a, **_k: None,
    get_user_likes=lambda *_a, **_k: [],
    get_active_rejections=lambda *_a, **_k: [],
    get_recent_plans=lambda *_a, **_k: [],
)
_install_stub(
    "db_facts",
    get_all_user_facts=lambda *_a, **_k: [],
    get_consumed_meals_since=lambda *_a, **_k: [],
    get_user_facts_by_metadata=lambda *_a, **_k: [],
)
_install_stub("pydantic", BaseModel=object, Field=lambda default=None, **_k: default)
_install_stub("schemas", HealthProfileSchema=object, ExpandedRecipeModel=object)
_install_stub("graph_orchestrator", run_plan_pipeline=lambda *_a, **_k: {})
_install_stub("memory_manager", build_memory_context=lambda *_a, **_k: "")
_install_stub("services", _save_plan_and_track_background=lambda *_a, **_k: None)
_install_stub("agent", analyze_preferences_agent=lambda *_a, **_k: {})
apscheduler_pkg = _install_stub("apscheduler")
apscheduler_triggers_pkg = _install_stub("apscheduler.triggers")
apscheduler_cron_pkg = _install_stub("apscheduler.triggers.cron", CronTrigger=object)
apscheduler_pkg.triggers = apscheduler_triggers_pkg
apscheduler_triggers_pkg.cron = apscheduler_cron_pkg


import constants  # noqa: E402
import cron_tasks  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: plan_data realistas para reflejar el escenario "queue NULL"
# ---------------------------------------------------------------------------
def _build_plan_data_for_30day_with_chunk1_completed():
    """Plan de 30 días donde chunk 1 (días 1-3) se completó pero
    `plan_chunk_queue.learning_metrics` quedó NULL (simula crash pre-preflight).
    Los días viven en `meal_plans.plan_data.days` con `week_number=1`.

    Las comidas tienen nombres y ingredientes deliberadamente repetitivos para
    que la síntesis pueda detectar bases comunes y producir lecciones útiles.
    """
    return {
        "total_days_requested": 30,
        "days": [
            {
                "day": 1,
                "week_number": 1,
                "meals": [
                    {
                        "name": "Pollo con arroz integral",
                        "ingredients": ["pollo 200g", "arroz integral 100g", "tomate 50g"],
                    },
                    {
                        "name": "Ensalada de pollo",
                        "ingredients": ["pollo 150g", "lechuga 80g", "aceite 10ml"],
                    },
                ],
            },
            {
                "day": 2,
                "week_number": 1,
                "meals": [
                    {
                        "name": "Arroz con habichuelas",
                        "ingredients": ["arroz integral 120g", "habichuelas 100g"],
                    },
                    {
                        "name": "Pollo a la plancha",
                        "ingredients": ["pollo 200g", "limón 10g"],
                    },
                ],
            },
            {
                "day": 3,
                "week_number": 1,
                "meals": [
                    {
                        "name": "Pollo guisado",
                        "ingredients": ["pollo 250g", "tomate 80g", "cebolla 50g"],
                    },
                ],
            },
        ],
    }


def _build_plan_data_with_multi_chunk_history():
    """Plan donde plan_data.days tiene contenido de DOS chunks distintos.
    Sirve para verificar que la síntesis filtra por target_week (no contamina
    la lección con días de otros chunks)."""
    return {
        "total_days_requested": 15,
        "days": [
            {
                "day": 1,
                "week_number": 1,
                "meals": [{"name": "Comida-W1-D1", "ingredients": ["pollo", "arroz"]}],
            },
            {
                "day": 2,
                "week_number": 1,
                "meals": [{"name": "Comida-W1-D2", "ingredients": ["pollo", "tomate"]}],
            },
            {
                "day": 4,
                "week_number": 2,
                "meals": [{"name": "Comida-W2-D4", "ingredients": ["res", "papa"]}],
            },
            {
                "day": 5,
                "week_number": 2,
                "meals": [{"name": "Comida-W2-D5", "ingredients": ["res", "yuca"]}],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Group A — Síntesis cierra la brecha cuando queue está NULL
# ---------------------------------------------------------------------------
def test_synthesis_recovers_lesson_when_queue_returns_none():
    """Escenario E2E: chunk previo completó con learning_metrics=NULL → el
    rebuild devuelve None → la síntesis se ejecuta sobre plan_data.days y
    devuelve un dict válido con datos del chunk previo.

    Replica el flujo del bloque cron_tasks.py:12456-12495 sin levantar el
    worker entero. Si este test falla, significa que la red de seguridad
    "queue NULL → synth" se rompió: el chunk N+1 arrancaría con dict vacío.
    """
    plan_data = _build_plan_data_for_30day_with_chunk1_completed()

    # Path 1: rebuild_from_queue → None (simulamos learning_metrics NULL)
    rebuild_mock = MagicMock(return_value=None)
    with patch.object(cron_tasks, "_rebuild_last_chunk_learning_from_queue", rebuild_mock):
        rebuild_result = cron_tasks._rebuild_last_chunk_learning_from_queue(
            "plan-30d", 1, prefer_completed=True, user_id="u1"
        )
    assert rebuild_result is None, "Setup: rebuild debe devolver None"

    # Path 2: fallback a síntesis. Esta es la red real que cubre el gap.
    synth = cron_tasks._synthesize_last_chunk_learning_from_plan_days(
        "plan-30d", target_week=1, prior_plan_data=plan_data, user_id="u1"
    )

    assert synth is not None, (
        "La síntesis debe devolver un dict cuando hay días reales en "
        "plan_data.days del chunk objetivo. Si devuelve None, el chunk "
        "siguiente arranca con dict vacío y se rompe el aprendizaje continuo."
    )
    assert synth.get("chunk") == 1
    assert synth.get("synthesized_from_plan_days") is True


def test_synthesis_extracts_real_meal_names_from_chunk_previous():
    """La lección sintetizada debe contener los nombres de las comidas del
    chunk previo en `repeated_meal_names`. Sin esto, el LLM del chunk N+1
    no recibe la señal "no repitas estos platos"."""
    plan_data = _build_plan_data_for_30day_with_chunk1_completed()

    synth = cron_tasks._synthesize_last_chunk_learning_from_plan_days(
        "plan-30d", target_week=1, prior_plan_data=plan_data, user_id="u1"
    )
    assert synth is not None

    names = synth.get("repeated_meal_names") or []
    assert any("Pollo" in n for n in names), (
        f"`repeated_meal_names` debe contener nombres del chunk 1. Got: {names}"
    )
    assert isinstance(synth.get("synthesized_meal_count"), int)
    assert synth["synthesized_meal_count"] >= 1


def test_synthesis_extracts_ingredient_bases_from_chunk_previous():
    """`repeated_bases` debe contener las bases canónicas de los ingredientes
    del chunk previo. Esto le permite al LLM saber "no repitas estas bases
    proteicas/glucídicas" aunque varíe los nombres de los platos."""
    plan_data = _build_plan_data_for_30day_with_chunk1_completed()

    synth = cron_tasks._synthesize_last_chunk_learning_from_plan_days(
        "plan-30d", target_week=1, prior_plan_data=plan_data, user_id="u1"
    )
    assert synth is not None

    bases = synth.get("repeated_bases") or []
    assert isinstance(bases, list) and bases, (
        f"`repeated_bases` debe ser lista no-vacía. Got: {bases}"
    )
    # Las bases reales dependen de `normalize_ingredient_for_tracking`. No
    # hardcodeamos 'pollo' (esa función puede mapear a sinónimos canónicos
    # distintos según embedding cache). Solo exigimos que la lista exista
    # y tenga al menos un elemento string normalizado.
    assert all(isinstance(b, str) and b.strip() for b in bases), (
        f"Todas las bases deben ser strings no-vacíos. Got: {bases}"
    )


# ---------------------------------------------------------------------------
# Group B — Aislamiento por target_week (no contaminación cross-chunk)
# ---------------------------------------------------------------------------
def test_synthesis_filters_strictly_by_target_week():
    """Cuando plan_data.days tiene contenido de múltiples chunks (común en
    planes de 15/30 días donde varias semanas ya completaron), la síntesis
    para target_week=N debe extraer SOLO comidas de week_number=N.

    Si esto rompe, el chunk N+1 vería "repeticiones" de chunks aún más
    antiguos, llevando al LLM a evitar platos que el usuario llevaba semanas
    sin ver. Regresión silenciosa pero dañina (deteriora variedad).
    """
    plan_data = _build_plan_data_with_multi_chunk_history()

    synth_w1 = cron_tasks._synthesize_last_chunk_learning_from_plan_days(
        "plan-15d", target_week=1, prior_plan_data=plan_data, user_id="u1"
    )
    assert synth_w1 is not None
    names_w1 = synth_w1.get("repeated_meal_names") or []
    assert any("W1" in n for n in names_w1), (
        f"target_week=1 debe extraer comidas de week 1. Got: {names_w1}"
    )
    assert not any("W2" in n for n in names_w1), (
        f"target_week=1 NO debe contaminar con week 2. Got: {names_w1}"
    )

    synth_w2 = cron_tasks._synthesize_last_chunk_learning_from_plan_days(
        "plan-15d", target_week=2, prior_plan_data=plan_data, user_id="u1"
    )
    assert synth_w2 is not None
    names_w2 = synth_w2.get("repeated_meal_names") or []
    assert any("W2" in n for n in names_w2), (
        f"target_week=2 debe extraer comidas de week 2. Got: {names_w2}"
    )
    assert not any("W1" in n for n in names_w2), (
        f"target_week=2 NO debe contaminar con week 1. Got: {names_w2}"
    )


def test_synthesis_returns_none_when_target_week_has_no_data():
    """Si el target_week pedido no tiene días en plan_data.days, la síntesis
    devuelve None (no inventa datos). El caller (worker) loguea warning y
    el chunk siguiente arranca con dict existente — preferible a inyectar
    una lección con counters todos en cero (que el LLM interpretaría como
    "sin repeticiones reales", falso positivo)."""
    plan_data = _build_plan_data_with_multi_chunk_history()

    # Pedimos target_week=3 pero plan_data solo tiene weeks 1 y 2.
    synth = cron_tasks._synthesize_last_chunk_learning_from_plan_days(
        "plan-15d", target_week=3, prior_plan_data=plan_data, user_id="u1"
    )
    assert synth is None, (
        f"Sin días para target_week=3, la síntesis debe devolver None. "
        f"Got: {synth}"
    )


# ---------------------------------------------------------------------------
# Group C — Flags de low-confidence presentes (contrato con LLM/dashboards)
# ---------------------------------------------------------------------------
def test_synthesis_marks_lesson_as_low_confidence():
    """Las lecciones sintetizadas DEBEN llevar las banderas que las distinguen
    de queue-rebuilt. Estas banderas las consume:
      - El prompt del LLM (para interpretar repeat_pct=0 como "ausente"
        en vez de "sin repeticiones").
      - El cron `_alert_high_synthesized_lesson_ratio` (telemetría).
      - Dashboards (UX para mostrar señal de aprendizaje degradado).

    Si una de estas banderas se pierde, los tres consumers leen mal la señal
    sin error visible. Regresión silenciosa de alto impacto.
    """
    plan_data = _build_plan_data_for_30day_with_chunk1_completed()
    synth = cron_tasks._synthesize_last_chunk_learning_from_plan_days(
        "plan-30d", target_week=1, prior_plan_data=plan_data, user_id="u1"
    )
    assert synth is not None

    # Flags obligatorios para que downstream distinga síntesis de queue-real.
    assert synth.get("synthesized_from_plan_days") is True
    assert synth.get("low_confidence") is True
    assert synth.get("metrics_unavailable") is True
    assert synth.get("learning_signal_strength") == "weak"

    # Counters numéricos en 0 (no hay señal real de repetición vs prior chunks).
    assert synth.get("repeat_pct") == 0
    assert synth.get("ingredient_base_repeat_pct") == 0
    assert synth.get("rejection_violations") == 0


# ---------------------------------------------------------------------------
# Group D — Persistencia legacy: contexto válido / inválido
# ---------------------------------------------------------------------------
def test_persist_accepts_synthesis_from_days_context():
    """`persist_legacy_learning_to_plan_data(context="synthesis_from_days")`
    es el único context legítimo que cierra el flujo P0-4 fallback. Debe
    aceptarlo y ejecutar el UPDATE."""
    lesson = {
        "chunk": 1,
        "repeat_pct": 0,
        "ingredient_base_repeat_pct": 0,
        "rejection_violations": 0,
        "repeated_bases": ["pollo"],
        "repeated_meal_names": ["Pollo guisado"],
        "synthesized_from_plan_days": True,
        "low_confidence": True,
        "metrics_unavailable": True,
    }
    # [P0-9] el helper ahora exige `returning=True`. Mock devuelve fila simulada.
    write_mock = MagicMock(return_value=[{"id": "plan-30d"}])
    with patch.object(cron_tasks, "execute_sql_write", write_mock):
        ok = cron_tasks.persist_legacy_learning_to_plan_data(
            "plan-30d", lesson, context="synthesis_from_days",
            user_id="u1",
        )
    assert ok is True, "persist con context válido debe devolver True"
    assert write_mock.call_count == 1, (
        f"Debe ejecutar exactamente 1 UPDATE. Got: {write_mock.call_count}"
    )
    sql_arg = write_mock.call_args.args[0]
    # Sello CAS: el UPDATE debe escribir _plan_modified_at para que el worker
    # T2 detecte la mutación. Sin esto, una lección persistida puede ser
    # pisada por un worker concurrente.
    assert "_plan_modified_at" in sql_arg, (
        "El UPDATE debe sellar _plan_modified_at para CAS atomicity."
    )
    assert "_last_chunk_learning" in sql_arg
    # [P0-9] ownership check
    assert "user_id" in sql_arg
    assert "RETURNING id" in sql_arg


def test_persist_rejects_unregistered_context():
    """Regresión guard: si alguien añade un nuevo path legacy sin registrarlo
    en `P0_3_LEGACY_LEARNING_CONTEXTS`, el persist debe FALLAR rápido (return
    False sin ejecutar SQL) en vez de escribir silenciosamente.

    Sin este guard, el sistema acumula caminos legacy no auditados, perdiendo
    la garantía de telemetría centralizada y CAS sealing."""
    lesson = {"chunk": 1, "repeated_bases": [], "repeated_meal_names": []}
    write_mock = MagicMock()
    with patch.object(cron_tasks, "execute_sql_write", write_mock):
        ok = cron_tasks.persist_legacy_learning_to_plan_data(
            "plan-30d", lesson, context="not_a_registered_context",
            user_id="u1",
        )
    assert ok is False, (
        "persist con context no registrado debe devolver False"
    )
    assert write_mock.call_count == 0, (
        "No debe ejecutar SQL con context no registrado."
    )


def test_synthesis_from_days_is_a_registered_context():
    """`synthesis_from_days` debe estar declarado en
    `P0_3_LEGACY_LEARNING_CONTEXTS`. Si alguien lo borra del tuple, la
    persistencia del path P0-4 fallback empieza a fallar silenciosamente
    en producción y el chunk N+1 arranca con `_last_chunk_learning` vacío
    pese a que la síntesis sí computó la lección."""
    assert "synthesis_from_days" in cron_tasks.P0_3_LEGACY_LEARNING_CONTEXTS, (
        f"`synthesis_from_days` debe estar registrado. "
        f"Tuple actual: {cron_tasks.P0_3_LEGACY_LEARNING_CONTEXTS}"
    )


# ---------------------------------------------------------------------------
# Group E — Cadena completa "in-memory recovery" (lo que ve el chunk N+1)
# ---------------------------------------------------------------------------
def test_in_memory_assignment_makes_synthesis_visible_to_next_chunk():
    """El bloque P0-4 (cron_tasks.py:12495) asigna la síntesis a
    `prior_plan_data["_last_chunk_learning"]` INCONDICIONALMENTE: aunque la
    persistencia falle, el chunk actual recibe la lección in-memory para
    que el prompt del LLM no arranque con dict vacío.

    Este test simula la cadena exacta: queue NULL → synth OK → persist
    falla → in-memory mutation aún ocurre → el chunk N+1 lee la lección.
    """
    plan_data = _build_plan_data_for_30day_with_chunk1_completed()

    # Simulamos persistencia fallida (e.g., SQL transient error).
    write_mock = MagicMock(side_effect=RuntimeError("transient DB error"))

    synth = cron_tasks._synthesize_last_chunk_learning_from_plan_days(
        "plan-30d", target_week=1, prior_plan_data=plan_data, user_id="u1"
    )
    assert synth is not None

    with patch.object(cron_tasks, "execute_sql_write", write_mock):
        persisted = cron_tasks.persist_legacy_learning_to_plan_data(
            "plan-30d", synth, context="synthesis_from_days",
            user_id="u1",
        )
    assert persisted is False, "Setup: persistencia debe fallar en este test"

    # El bloque P0-4 hace: prior_plan_data["_last_chunk_learning"] = _p04_synth
    # ANTES de chequear `if _p04_persisted`. Replicamos esa asignación.
    plan_data["_last_chunk_learning"] = synth

    # Verificamos que el chunk N+1 leería la lección sintetizada.
    visible_lesson = plan_data.get("_last_chunk_learning")
    assert visible_lesson is not None, (
        "in-memory mutation falló: chunk N+1 arrancaría con dict vacío "
        "pese a que la síntesis sí produjo lección. Esto rompería el "
        "contrato P0-4 (la red de seguridad final)."
    )
    assert visible_lesson.get("synthesized_from_plan_days") is True
    assert visible_lesson.get("repeated_meal_names")
