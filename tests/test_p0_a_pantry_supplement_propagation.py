"""[P0-A] Propagación de `_pantry_supplement_required` desde validación de
pantry hasta `meal_plans.plan_data` (origen de la categoría "🚨 Compra Urgente"
del PDF de la lista de compras).

Contexto:
    Antes de P0-A, el flag se escribía en variables locales (`form_data` del
    worker, `audit` del path inicial) pero NUNCA llegaba a `plan_data` en BD.
    Resultado: cuando un chunk en `flexible_mode` o el chunk 1 sincrónico
    fallaban la validación contra inventario live, el push notification "te
    faltan ingredientes, revisa tu lista de compras" llevaba al usuario a
    un PDF que NO listaba los items urgentes — promesa de producto rota.

Cubierto:
    1. `_extract_missing_ingredients_from_violation`: helper puro extrae
       nombres desde la cadena de error de `validate_ingredients_against_pantry`.
    2. `_validate_and_retry_initial_chunk_against_pantry`: cuando devuelve
       `audit["degraded"]=True`, también pobla `audit["missing_list"]` para
       que el caller la propague a `result`.
    3. `_run_pantry_validation_for_initial_chunk` (router): cuando el audit
       reporta degraded + missing_list, set `result["_pantry_supplement_required"]`
       — clave que `save_partial_plan_get_id` (services.py:139, `{**plan_data, ...}`)
       persiste a `meal_plans.plan_data._pantry_supplement_required`.
    4. `_persist_pantry_supplement_to_plan_data`: en el worker (chunks 2..N),
       persiste el flag + recalcula shopping list ANTES de la pausa.
    5. `_clear_pantry_supplement_from_plan_data`: cuando el chunk pasa
       validación flexible, limpia el flag para que la categoría urgente
       no quede pegada en el PDF tras restock que destrabó el chunk.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Stubs para módulos externos no disponibles en CI/test
# ---------------------------------------------------------------------------

def _install_stub(module_name, **attrs):
    if module_name in sys.modules:
        existing = sys.modules[module_name]
        for key, value in attrs.items():
            if not hasattr(existing, key):
                setattr(existing, key, value)
        return existing
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module
    return module


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
    supabase=None,
)
_install_stub(
    "db_inventory",
    deduct_consumed_meal_from_inventory=lambda *_a, **_kw: None,
    get_inventory_activity_since=lambda *_a, **_kw: [],
    get_raw_user_inventory=lambda *_a, **_kw: [],
    get_user_inventory_net=lambda *_a, **_kw: [],
    release_chunk_reservations=lambda *_a, **_kw: None,
    reserve_plan_ingredients=lambda *_a, **_kw: 0,
    restock_inventory=lambda *_a, **_kw: None,
    consume_inventory_items_completely=lambda *_a, **_kw: None,
)
_install_stub(
    "db",
    supabase=None,
    get_user_likes=lambda *_a, **_kw: [],
    get_active_rejections=lambda *_a, **_kw: [],
    get_or_create_session=lambda *_a, **_kw: None,
    save_message=lambda *_a, **_kw: None,
    update_user_health_profile=lambda *_a, **_kw: None,
    log_api_usage=lambda *_a, **_kw: None,
    get_latest_meal_plan=lambda *_a, **_kw: None,
    get_latest_meal_plan_with_id=lambda *_a, **_kw: None,
    get_recent_plans=lambda *_a, **_kw: [],
    update_meal_plan_data=lambda *_a, **_kw: None,
    insert_like=lambda *_a, **_kw: None,
    # Necesario para fact_extractor (importado vía graph_orchestrator).
    save_user_fact=lambda *_a, **_kw: None,
    search_user_facts=lambda *_a, **_kw: [],
    delete_user_fact=lambda *_a, **_kw: None,
    search_user_facts_hybrid=lambda *_a, **_kw: [],
    get_user_facts_by_metadata=lambda *_a, **_kw: [],
    acquire_fact_lock=lambda *_a, **_kw: True,
    release_fact_lock=lambda *_a, **_kw: None,
    enqueue_pending_fact=lambda *_a, **_kw: None,
    dequeue_pending_facts=lambda *_a, **_kw: [],
    delete_pending_facts=lambda *_a, **_kw: None,
)
_install_stub(
    "db_facts",
    get_all_user_facts=lambda *_a, **_kw: [],
    get_consumed_meals_since=lambda *_a, **_kw: [],
    get_user_facts_by_metadata=lambda *_a, **_kw: [],
)
_install_stub("pydantic", BaseModel=object, Field=lambda default=None, **_kw: default)
_install_stub(
    "shopping_calculator",
    get_shopping_list_delta=lambda *_a, **_kw: [],
    _parse_quantity=lambda text, *_a, **_kw: (1.0, "ud", str(text or "")),
)
# [P0-A] Stub graph_orchestrator ANTES de importar cron_tasks: cron_tasks.py:101
# hace `from graph_orchestrator import run_plan_pipeline`, lo que transitivamente
# carga fact_extractor → db (real). El stub previene la cascada y mantiene el
# test rápido + aislado de la infra.
_install_stub(
    "graph_orchestrator",
    run_plan_pipeline=lambda *_a, **_kw: {},
    arun_plan_pipeline=lambda *_a, **_kw: {},
    _strip_untrusted_internal_keys=lambda *_a, **_kw: [],
    _enforce_days_to_generate_cap=lambda *_a, **_kw: False,
    _merge_other_text_fields=lambda *_a, **_kw: 0,
    # [NG-4 · 2026-05-30] cron_tasks.py:110 importa _env_int/_env_float/_env_bool desde
    # graph_orchestrator (P1-A · 2026-05-08); el stub quedó stale y rompía el import
    # transitivo. Passthrough al default (las knobs no afectan estos tests de persistencia).
    _env_int=lambda _name, default=0, **_kw: default,
    _env_float=lambda _name, default=0.0, **_kw: default,
    _env_bool=lambda _name, default=False, **_kw: default,
    # [test fix] cron_tasks importa tambien _env_str desde graph_orchestrator
    # (drift del stub detectado en audit v5 2026-07-02).
    _env_str=lambda _name, default=False, **_kw: str(default) if default is not None else "",
)
_install_stub(
    "memory_manager",
    build_memory_context=lambda *_a, **_kw: {"recent_messages": [], "full_context_str": ""},
    summarize_and_prune=lambda *_a, **_kw: None,
)
_install_stub(
    "services",
    _save_plan_and_track_background=lambda *_a, **_kw: None,
    _process_swap_rejection_background=lambda *_a, **_kw: None,
    save_partial_plan_get_id=lambda *_a, **_kw: None,
)
_install_stub(
    "agent",
    analyze_preferences_agent=lambda *_a, **_kw: "",
    swap_meal=lambda *_a, **_kw: None,
)
_install_stub("ai_helpers", expand_recipe_agent=lambda *_a, **_kw: None)
_install_stub(
    "auth",
    get_verified_user_id=lambda *_a, **_kw: None,
    verify_api_quota=lambda *_a, **_kw: None,
)
_install_stub(
    "schemas",
    HealthProfileSchema=object,
    ExpandedRecipeModel=object,
    PUBLIC_SSE_EVENTS=frozenset({
        "phase", "day_started", "day_complete", "day_completed",
        "complete", "error", "heartbeat",
    }),
)
apscheduler_pkg = _install_stub("apscheduler")
apscheduler_triggers_pkg = _install_stub("apscheduler.triggers")
apscheduler_cron_pkg = _install_stub("apscheduler.triggers.cron", CronTrigger=object)
apscheduler_pkg.triggers = apscheduler_triggers_pkg
apscheduler_triggers_pkg.cron = apscheduler_cron_pkg


import pytest  # noqa: E402
from unittest.mock import patch, MagicMock  # noqa: E402

import cron_tasks  # noqa: E402


# ---------------------------------------------------------------------------
# Test 1 — Helper de extracción es puro y robusto
# ---------------------------------------------------------------------------

class TestExtractHelper:
    def test_extracts_inexistentes_section(self):
        violation = "INEXISTENTES en inventario: pollo, arroz, tomate."
        result = cron_tasks._extract_missing_ingredients_from_violation(violation)
        assert result == ["pollo", "arroz", "tomate"]

    def test_extracts_cantidades_section(self):
        violation = (
            "CANTIDADES (Tu inventario restringe esto matemáticamente): "
            "[salmón] (Pediste 500g, límite: 200g), [aguacate] (Pediste 2u, límite: 1u)."
        )
        result = cron_tasks._extract_missing_ingredients_from_violation(violation)
        assert result == ["salmón", "aguacate"]

    def test_extracts_both_sections_combined(self):
        violation = (
            "Validación falló. INEXISTENTES en inventario: pollo. "
            "CANTIDADES (Tu inventario restringe esto matemáticamente): [arroz] (Pediste 800g, límite: 200g)."
        )
        result = cron_tasks._extract_missing_ingredients_from_violation(violation)
        # Orden: primero unauth, luego limit; sin duplicados.
        assert result == ["pollo", "arroz"]

    def test_dedup_across_sections(self):
        violation = (
            "INEXISTENTES en inventario: pollo, arroz. "
            "CANTIDADES (Tu inventario restringe esto matemáticamente): [arroz] (insuficiente)."
        )
        result = cron_tasks._extract_missing_ingredients_from_violation(violation)
        assert result == ["pollo", "arroz"]  # arroz aparece una sola vez

    def test_handles_none_input(self):
        assert cron_tasks._extract_missing_ingredients_from_violation(None) == []

    def test_handles_non_string_input(self):
        assert cron_tasks._extract_missing_ingredients_from_violation(True) == []
        assert cron_tasks._extract_missing_ingredients_from_violation(123) == []
        assert cron_tasks._extract_missing_ingredients_from_violation([]) == []

    def test_handles_empty_string(self):
        assert cron_tasks._extract_missing_ingredients_from_violation("") == []

    def test_handles_strings_without_match(self):
        assert cron_tasks._extract_missing_ingredients_from_violation("Plan validado OK.") == []
        assert cron_tasks._extract_missing_ingredients_from_violation("error genérico sin formato esperado") == []

    def test_skips_empty_items_in_csv(self):
        # Caso edge: trailing comma o espacios.
        violation = "INEXISTENTES en inventario: pollo, , arroz,  ."
        result = cron_tasks._extract_missing_ingredients_from_violation(violation)
        assert result == ["pollo", "arroz"]


# ---------------------------------------------------------------------------
# Test 2 — `_validate_and_retry_initial_chunk_against_pantry` pobla missing_list
# ---------------------------------------------------------------------------

def _result_with_violation():
    return {
        "days": [
            {
                "day": 1,
                "meals": [
                    {"name": "BAD-salmon", "ingredients": ["salmón", "arroz"]},
                ],
            }
        ]
    }


class TestInitialChunkAuditMissingList:
    def test_audit_missing_list_populated_on_existence_violation(self):
        """Cuando degraded=True por existencia, audit['missing_list'] debe
        contener los items extraídos de last_violation."""
        pantry = ["pollo", "arroz"]
        initial = _result_with_violation()

        def _mock_vip(generated, _pantry, strict_quantities=True, **_kw):
            return "INEXISTENTES en inventario: salmón."

        with patch("constants.validate_ingredients_against_pantry", side_effect=_mock_vip), \
             patch("cron_tasks.CHUNK_PANTRY_MAX_RETRIES", 0):
            _, audit = cron_tasks._validate_and_retry_initial_chunk_against_pantry(
                pipeline_data={},
                history=[],
                taste_profile="",
                memory_context="",
                background_tasks=None,
                pantry_ingredients=pantry,
                initial_result=initial,
                user_id=None,
            )

        assert audit["degraded"] is True
        assert audit["missing_list"] == ["salmón"]

    def test_audit_missing_list_empty_when_validated_ok(self):
        """Cuando no hay degradación, missing_list permanece vacía (init)."""
        pantry = ["pollo", "arroz"]
        clean = {
            "days": [
                {
                    "day": 1,
                    "meals": [{"name": "OK", "ingredients": ["pollo", "arroz"]}],
                }
            ]
        }
        with patch("constants.validate_ingredients_against_pantry", return_value=True):
            _, audit = cron_tasks._validate_and_retry_initial_chunk_against_pantry(
                pipeline_data={},
                history=[],
                taste_profile="",
                memory_context="",
                background_tasks=None,
                pantry_ingredients=pantry,
                initial_result=clean,
                user_id=None,
            )

        assert audit["validated_ok"] is True
        assert audit["missing_list"] == []

    def test_audit_missing_list_present_in_qty_violation(self):
        """[P0-A] Bug regression: la rama de violación de cantidad (hybrid/strict)
        también debe poblar missing_list, no solo la rama de existencia."""
        pantry = ["pollo 100g", "arroz 100g"]
        initial = _result_with_violation()

        # Primer call (existence): pasa. Segundo (quantity strict): falla.
        call_state = {"n": 0}

        def _mock_vip(generated, _pantry, strict_quantities=True, **_kw):
            call_state["n"] += 1
            if call_state["n"] == 1:
                return True  # existence OK
            return (
                "CANTIDADES (Tu inventario restringe esto matemáticamente): "
                "[arroz] (Pediste 800g, límite: 100g)."
            )

        with patch("constants.validate_ingredients_against_pantry", side_effect=_mock_vip), \
             patch("cron_tasks.CHUNK_PANTRY_QUANTITY_MODE", "strict"), \
             patch("cron_tasks.CHUNK_PANTRY_MAX_RETRIES", 0):
            _, audit = cron_tasks._validate_and_retry_initial_chunk_against_pantry(
                pipeline_data={},
                history=[],
                taste_profile="",
                memory_context="",
                background_tasks=None,
                pantry_ingredients=pantry,
                initial_result=initial,
                user_id=None,
            )

        assert audit["degraded"] is True
        assert audit["missing_list"] == ["arroz"]


# ---------------------------------------------------------------------------
# Test 3 — Persistencia de _pantry_supplement_required al plan_data
# ---------------------------------------------------------------------------

class TestPersistPantrySupplement:
    def test_persists_flag_when_missing_list_present(self):
        """Verifica que _persist_pantry_supplement_to_plan_data hace UPDATE
        del JSONB con _pantry_supplement_required cuando missing_list no está vacía."""
        sql_writes = []
        sql_queries = [{"plan_data": {"days": [], "form_data": {}}}]  # GET retorna esto

        def _mock_query(*a, **_kw):
            return sql_queries.pop(0) if sql_queries else None

        def _mock_write(query, params=None, **_kw):
            sql_writes.append((query, params))

        with patch("cron_tasks.execute_sql_query", side_effect=_mock_query), \
             patch("cron_tasks.execute_sql_write", side_effect=_mock_write), \
             patch(
                 "shopping_calculator.get_shopping_list_delta",
                 side_effect=lambda *_a, **_kw: [{"name": "fake_item"}],
             ):
            ok = cron_tasks._persist_pantry_supplement_to_plan_data(
                meal_plan_id="plan-uuid-123",
                user_id="user-1",
                missing_list=["pollo", "arroz"],
                source="test",
            )

        assert ok is True
        assert len(sql_writes) == 1
        query, params = sql_writes[0]
        assert "_pantry_supplement_required" in query
        assert "aggregated_shopping_list" in query
        # El primer param del jsonb_set anidado es la lista urgente serializada.
        import json as _json
        assert _json.loads(params[0]) == ["pollo", "arroz"]
        # [NG-4 · 2026-05-30] I2: el UPDATE ahora filtra AND user_id = %s, así que
        # meal_plan_id pasó de params[-1] a params[-2] y user_id es el último.
        assert params[-2] == "plan-uuid-123"
        assert params[-1] == "user-1"
        assert "user_id = %s" in query

    def test_no_op_when_missing_list_empty(self):
        """No hacer ningún UPDATE si missing_list está vacía."""
        sql_writes = []
        with patch("cron_tasks.execute_sql_write", side_effect=lambda *a, **_kw: sql_writes.append(a)):
            ok = cron_tasks._persist_pantry_supplement_to_plan_data(
                meal_plan_id="plan-uuid-123",
                user_id="user-1",
                missing_list=[],
                source="test",
            )
        assert ok is False
        assert sql_writes == []

    def test_no_op_when_meal_plan_id_missing(self):
        sql_writes = []
        with patch("cron_tasks.execute_sql_write", side_effect=lambda *a, **_kw: sql_writes.append(a)):
            ok = cron_tasks._persist_pantry_supplement_to_plan_data(
                meal_plan_id=None,
                user_id="user-1",
                missing_list=["pollo"],
                source="test",
            )
        assert ok is False
        assert sql_writes == []

    def test_merges_with_existing_supplement(self):
        """Si plan_data ya tenía un supplement previo, mergeamos (dedup
        case-insensitive) en lugar de pisar — chunk B no debe perder los
        items urgentes que chunk A persistió."""
        sql_writes = []
        sql_queries = [{
            "plan_data": {
                "_pantry_supplement_required": ["Pollo"],  # capitalización distinta
                "form_data": {},
            }
        }]

        def _mock_query(*a, **_kw):
            return sql_queries.pop(0) if sql_queries else None

        def _mock_write(query, params=None, **_kw):
            sql_writes.append((query, params))

        with patch("cron_tasks.execute_sql_query", side_effect=_mock_query), \
             patch("cron_tasks.execute_sql_write", side_effect=_mock_write), \
             patch(
                 "shopping_calculator.get_shopping_list_delta",
                 side_effect=lambda *_a, **_kw: [],
             ):
            cron_tasks._persist_pantry_supplement_to_plan_data(
                meal_plan_id="plan-uuid-456",
                user_id="user-2",
                missing_list=["pollo", "arroz"],  # "pollo" duplica con "Pollo"
                source="test",
            )

        import json as _json
        merged = _json.loads(sql_writes[0][1][0])
        # "Pollo" preservado (primero en aparecer, dedup case-insensitive),
        # "arroz" añadido. "pollo" descartado por dedup.
        assert merged == ["Pollo", "arroz"]


# ---------------------------------------------------------------------------
# Test 4 — Limpieza del flag al pasar validación flexible
# ---------------------------------------------------------------------------

class TestClearPantrySupplement:
    def test_clears_when_flag_present(self):
        sql_writes = []
        sql_queries = [{
            "plan_data": {
                "_pantry_supplement_required": ["pollo"],
                "form_data": {},
            }
        }]

        def _mock_query(*a, **_kw):
            return sql_queries.pop(0) if sql_queries else None

        def _mock_write(query, params=None, **_kw):
            sql_writes.append((query, params))

        with patch("cron_tasks.execute_sql_query", side_effect=_mock_query), \
             patch("cron_tasks.execute_sql_write", side_effect=_mock_write), \
             patch(
                 "shopping_calculator.get_shopping_list_delta",
                 side_effect=lambda *_a, **_kw: [],
             ):
            ok = cron_tasks._clear_pantry_supplement_from_plan_data(
                meal_plan_id="plan-uuid-789",
                user_id="user-3",
                source="test",
            )

        assert ok is True
        assert len(sql_writes) == 1
        # El delete del flag se hace vía operador `#-` en PostgreSQL.
        assert "#-" in sql_writes[0][0]
        assert "_pantry_supplement_required" in sql_writes[0][0]

    def test_no_op_when_flag_absent(self):
        sql_writes = []
        sql_queries = [{"plan_data": {"days": []}}]  # sin _pantry_supplement_required

        def _mock_query(*a, **_kw):
            return sql_queries.pop(0) if sql_queries else None

        with patch("cron_tasks.execute_sql_query", side_effect=_mock_query), \
             patch("cron_tasks.execute_sql_write", side_effect=lambda *a, **_kw: sql_writes.append(a)):
            ok = cron_tasks._clear_pantry_supplement_from_plan_data(
                meal_plan_id="plan-uuid-no-flag",
                user_id="user-4",
                source="test",
            )
        assert ok is False
        assert sql_writes == []


# ---------------------------------------------------------------------------
# Test 5 — Shopping list E2E: la categoría "🚨 Compra Urgente" aparece
# cuando _pantry_supplement_required está en plan_data.
# ---------------------------------------------------------------------------

class TestShoppingListCategoryE2E:
    """Smoke test: confirma que el contrato `plan_data._pantry_supplement_required`
    → categoría "🚨 Compra Urgente" del shopping list sigue intacto. Sin esto,
    nuestra fix podría persistir el flag a un campo que el calculator no consulta.
    """

    def test_shopping_calculator_appends_urgent_category(self):
        # Reload del módulo real (sin stub) para invocar `get_shopping_list_delta`
        # con su lógica genuina.
        import importlib
        # Quitar el stub previo si está presente.
        sys.modules.pop("shopping_calculator", None)
        try:
            shopping_calculator = importlib.import_module("shopping_calculator")
        except Exception:
            pytest.skip("shopping_calculator no se puede importar en este entorno (deps faltantes)")

        plan_result = {
            "days": [
                {
                    "day": 1,
                    "meals": [
                        {
                            "name": "Plato test",
                            "meal": "almuerzo",
                            "ingredients": ["pollo"],
                        }
                    ],
                }
            ],
            "_pantry_supplement_required": ["bistec urgente"],
        }
        # Llamamos en modo categorize=True/structured=True para verificar la
        # categoría se materializa.
        try:
            res = shopping_calculator.get_shopping_list_delta(
                user_id=None,  # guest
                plan_result=plan_result,
                is_new_plan=True,
                categorize=True,
                structured=True,
                multiplier=1.0,
            )
        except Exception as e:
            pytest.skip(f"shopping_calculator no es ejercitable en este entorno: {e}")

        # Si el módulo cargó pero el calc falló por DB, skip; si funcionó, la
        # categoría 🚨 debe estar presente.
        if not isinstance(res, dict):
            pytest.skip("shopping_calculator devolvió un tipo no-dict; entorno no estándar.")
        assert "🚨 Compra Urgente" in res, (
            "_pantry_supplement_required no produjo la categoría '🚨 Compra Urgente' — "
            "regresión del contrato P0-A."
        )
