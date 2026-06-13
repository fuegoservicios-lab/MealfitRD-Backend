"""[P2-4 · 2026-05-08] Tests del guard de drift householdComposition en
`db_inventory._compute_dynamic_consumption_rates`.

La lista weekly se escala con el multiplier del momento de generar el plan
y se persiste en `plan_data.calc_household_multiplier` (M_cached). Si el
usuario actualiza su `householdComposition` entre chunks, el multiplier
actual (M_now) puede divergir; los rates derivados de la lista weekly
quedan sesgados.

El guard compara `abs(M_now - M_cached) / M_cached` contra
`MEALFIT_DYNAMIC_RATE_HOUSEHOLD_DRIFT_THRESHOLD` (default 0.20). Si excede,
retorna `{}` para que el caller caiga al fallback hardcoded por categoría.

Ejecutar:
    cd backend && python -m pytest tests/test_p2_4_dynamic_rate_household_drift_guard.py -v
"""
import os
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_sql_mock(plan_data: dict | None, health_profile: dict | None = None):
    """[P1-NEON-DB-MIGRATION · 2026-06-12] Mock de `db_inventory.execute_sql_query`
    (el transporte PostgREST `supabase.table(...)` fue reemplazado por SQL
    directo via pool psycopg). Enruta por la tabla referenciada en el SQL:

      - `FROM meal_plans`     → `{"plan_data": plan_data}` (fetch_one) o None.
      - `FROM user_profiles`  → `{"health_profile": health_profile}` o None.

    Además verifica la forma del SQL productivo (scoping por user): el SELECT
    de meal_plans DEBE filtrar `WHERE user_id = %s` y el de user_profiles
    `WHERE id = %s` — si el source cambia la query, el router falla loud en
    vez de devolver data del mock a una query distinta.

    Paridad de tipos post-migración: psycopg decodifica jsonb a dict nativo,
    así que `plan_data`/`health_profile` se devuelven como dict (no string).
    """
    def _router(sql, params=None, fetch_one=False, fetch_all=False):
        if "FROM meal_plans" in sql:
            assert "WHERE user_id = %s" in sql, (
                f"SELECT de meal_plans sin filtro `WHERE user_id = %s` — "
                f"scoping perdido o query renombrada. SQL: {sql}"
            )
            return {"plan_data": plan_data} if plan_data is not None else None
        if "FROM user_profiles" in sql:
            assert "WHERE id = %s" in sql, (
                f"SELECT de user_profiles sin filtro `WHERE id = %s`. SQL: {sql}"
            )
            return {"health_profile": health_profile} if health_profile is not None else None
        raise AssertionError(f"Query SQL inesperada en _compute_dynamic_consumption_rates: {sql}")

    return MagicMock(side_effect=_router)


def _queried_tables(sql_mock) -> list:
    """Tablas tocadas por el mock (en orden), derivadas del primer arg SQL."""
    tables = []
    for call in sql_mock.call_args_list:
        sql = call.args[0]
        if "FROM meal_plans" in sql:
            tables.append("meal_plans")
        elif "FROM user_profiles" in sql:
            tables.append("user_profiles")
    return tables


@contextmanager
def _transport(sql_mock):
    """Inyecta el transporte mockeado: `_db_available()` lee
    `db_core.connection_pool` lazy (truthy ⇒ DB ok) y la función bajo test
    usa el símbolo módulo-level `db_inventory.execute_sql_query`."""
    with patch("db_core.connection_pool", object()), \
         patch("db_inventory.execute_sql_query", sql_mock):
        yield


def _weekly_with_chicken(qty_g: float = 1400.0):
    """Lista weekly con un solo item — pollo a `qty_g` gramos (rate = qty_g/7)."""
    return [
        {
            "name": "Pechuga de pollo",
            "market_qty": qty_g,
            "market_unit": "g",
        }
    ]


# ---------------------------------------------------------------------------
# 1. Sin drift: rates devueltos
# ---------------------------------------------------------------------------
def test_no_drift_returns_rates_when_multipliers_match():
    plan_data = {
        "calc_household_multiplier": 4.0,
        "aggregated_shopping_list_weekly": _weekly_with_chicken(1400.0),
    }
    sql_mock = _make_sql_mock(plan_data)
    with _transport(sql_mock):
        from db_inventory import _compute_dynamic_consumption_rates
        rates = _compute_dynamic_consumption_rates("user-1", current_household_multiplier=4.0)
    assert rates, "Sin drift, los rates dinámicos deben devolverse"
    # 1400g / 7d = 200g/día
    assert any(abs(v - 200.0) < 0.01 for v in rates.values())


# ---------------------------------------------------------------------------
# 2. Drift dentro del threshold (default 20%): rates devueltos
# ---------------------------------------------------------------------------
def test_drift_within_threshold_returns_rates():
    plan_data = {
        "calc_household_multiplier": 4.0,
        "aggregated_shopping_list_weekly": _weekly_with_chicken(),
    }
    sql_mock = _make_sql_mock(plan_data)
    # 4.0 → 4.5 = drift 12.5%, dentro del 20% default
    with _transport(sql_mock):
        from db_inventory import _compute_dynamic_consumption_rates
        rates = _compute_dynamic_consumption_rates("user-1", current_household_multiplier=4.5)
    assert rates, "Drift 12.5% < 20% threshold → rates devueltos"


# ---------------------------------------------------------------------------
# 3. Drift excede threshold: {} + WARNING
# ---------------------------------------------------------------------------
def test_drift_exceeds_threshold_returns_empty():
    plan_data = {
        "calc_household_multiplier": 4.5,
        "aggregated_shopping_list_weekly": _weekly_with_chicken(),
    }
    sql_mock = _make_sql_mock(plan_data)
    # 4.5 → 3.0 = drift 33.3%, excede 20%
    with _transport(sql_mock):
        from db_inventory import _compute_dynamic_consumption_rates
        rates = _compute_dynamic_consumption_rates("user-1", current_household_multiplier=3.0)
    assert rates == {}, "Drift 33% > 20% threshold → caller debe caer al fallback hardcoded"


# ---------------------------------------------------------------------------
# 4. Plan sin calc_household_multiplier: {} (conservador)
# ---------------------------------------------------------------------------
def test_missing_calc_household_multiplier_returns_empty():
    plan_data = {
        # Sin `calc_household_multiplier` → no podemos validar drift
        "aggregated_shopping_list_weekly": _weekly_with_chicken(),
    }
    sql_mock = _make_sql_mock(plan_data)
    with _transport(sql_mock):
        from db_inventory import _compute_dynamic_consumption_rates
        rates = _compute_dynamic_consumption_rates("user-1", current_household_multiplier=4.0)
    assert rates == {}, "Plan sin metadata → conservador, fallback hardcoded"


# ---------------------------------------------------------------------------
# 5. Knob threshold bajado: drift antes aceptado ahora bloquea
# ---------------------------------------------------------------------------
def test_threshold_knob_override_tightens_guard():
    plan_data = {
        "calc_household_multiplier": 4.0,
        "aggregated_shopping_list_weekly": _weekly_with_chicken(),
    }
    sql_mock = _make_sql_mock(plan_data)
    # 4.0 → 4.5 = drift 12.5%, threshold ajustado a 5% → debe bloquear
    with patch.dict(os.environ, {"MEALFIT_DYNAMIC_RATE_HOUSEHOLD_DRIFT_THRESHOLD": "0.05"}):
        with _transport(sql_mock):
            # Forzar re-lectura del knob: el helper hace import lazy cada llamada
            from db_inventory import _compute_dynamic_consumption_rates
            rates = _compute_dynamic_consumption_rates("user-1", current_household_multiplier=4.5)
    assert rates == {}, "Threshold 5% < drift 12.5% → debe bloquear"


# ---------------------------------------------------------------------------
# 6. Knob threshold relajado: drift antes bloqueado ahora pasa
# ---------------------------------------------------------------------------
def test_threshold_knob_override_relaxes_guard():
    plan_data = {
        "calc_household_multiplier": 4.5,
        "aggregated_shopping_list_weekly": _weekly_with_chicken(),
    }
    sql_mock = _make_sql_mock(plan_data)
    # drift 33%, threshold 50% → pasa
    with patch.dict(os.environ, {"MEALFIT_DYNAMIC_RATE_HOUSEHOLD_DRIFT_THRESHOLD": "0.50"}):
        with _transport(sql_mock):
            from db_inventory import _compute_dynamic_consumption_rates
            rates = _compute_dynamic_consumption_rates("user-1", current_household_multiplier=3.0)
    assert rates, "Threshold 50% > drift 33% → debe pasar"


# ---------------------------------------------------------------------------
# 7. Caller pasa current_multiplier: NO se hace query a user_profiles
# ---------------------------------------------------------------------------
def test_caller_passes_multiplier_skips_user_profiles_query():
    plan_data = {
        "calc_household_multiplier": 4.0,
        "aggregated_shopping_list_weekly": _weekly_with_chicken(),
    }
    sql_mock = _make_sql_mock(plan_data)  # health_profile=None → si se consultara, retornaría None
    with _transport(sql_mock):
        from db_inventory import _compute_dynamic_consumption_rates
        rates = _compute_dynamic_consumption_rates("user-1", current_household_multiplier=4.0)
    # Verificar que solo se consultó meal_plans (no user_profiles)
    table_calls = _queried_tables(sql_mock)
    assert "user_profiles" not in table_calls, (
        "Si el caller pasa current_household_multiplier, el guard NO debe consultar user_profiles"
    )
    assert rates


# ---------------------------------------------------------------------------
# 8. Caller pasa None: SÍ se consulta user_profiles para resolver M_now
# ---------------------------------------------------------------------------
def test_caller_passes_none_queries_user_profiles():
    plan_data = {
        "calc_household_multiplier": 4.0,
        "aggregated_shopping_list_weekly": _weekly_with_chicken(),
    }
    health_profile = {"householdComposition": {"adults": 4, "children": 0}}
    sql_mock = _make_sql_mock(plan_data, health_profile=health_profile)
    with _transport(sql_mock):
        from db_inventory import _compute_dynamic_consumption_rates
        rates = _compute_dynamic_consumption_rates("user-1", current_household_multiplier=None)
    table_calls = _queried_tables(sql_mock)
    assert "user_profiles" in table_calls, (
        "Sin current_household_multiplier, el guard debe resolver M_now consultando user_profiles"
    )
    # 4 adultos = mult 4.0, M_cached 4.0, drift 0 → rates devueltos
    assert rates


# ---------------------------------------------------------------------------
# 9. No hay plan activo: {} (path independiente del guard)
# ---------------------------------------------------------------------------
def test_no_plan_returns_empty_without_invoking_guard():
    sql_mock = _make_sql_mock(plan_data=None)
    with _transport(sql_mock):
        from db_inventory import _compute_dynamic_consumption_rates
        rates = _compute_dynamic_consumption_rates("user-1", current_household_multiplier=4.0)
    assert rates == {}, "Sin plan activo → {} (no se evalúa el guard)"


# ---------------------------------------------------------------------------
# 10. Weekly vacía: {} aunque pase el guard
# ---------------------------------------------------------------------------
def test_empty_weekly_returns_empty_even_with_no_drift():
    plan_data = {
        "calc_household_multiplier": 4.0,
        "aggregated_shopping_list_weekly": [],
    }
    sql_mock = _make_sql_mock(plan_data)
    with _transport(sql_mock):
        from db_inventory import _compute_dynamic_consumption_rates
        rates = _compute_dynamic_consumption_rates("user-1", current_household_multiplier=4.0)
    assert rates == {}, "Weekly vacía → no hay rates qué derivar"


# ---------------------------------------------------------------------------
# 11. Drift exactamente igual al threshold: pasa (NO bloquea)
# ---------------------------------------------------------------------------
def test_drift_at_exact_threshold_does_not_block():
    plan_data = {
        "calc_household_multiplier": 5.0,
        "aggregated_shopping_list_weekly": _weekly_with_chicken(),
    }
    sql_mock = _make_sql_mock(plan_data)
    # 5.0 → 4.0 = drift exactamente 0.20 (igual al default)
    with _transport(sql_mock):
        from db_inventory import _compute_dynamic_consumption_rates
        rates = _compute_dynamic_consumption_rates("user-1", current_household_multiplier=4.0)
    assert rates, "Drift == threshold (no excede) → debe pasar"


# ---------------------------------------------------------------------------
# 12. Knob registrado en _KNOBS_REGISTRY del orchestrator
# ---------------------------------------------------------------------------
def test_knob_registered_in_global_registry():
    plan_data = {
        "calc_household_multiplier": 4.0,
        "aggregated_shopping_list_weekly": _weekly_with_chicken(),
    }
    sql_mock = _make_sql_mock(plan_data)
    with _transport(sql_mock):
        from db_inventory import _compute_dynamic_consumption_rates
        _compute_dynamic_consumption_rates("user-1", current_household_multiplier=4.0)
    from graph_orchestrator import get_knobs_registry_snapshot
    snap = get_knobs_registry_snapshot()
    assert "MEALFIT_DYNAMIC_RATE_HOUSEHOLD_DRIFT_THRESHOLD" in snap, (
        "El knob debe quedar registrado vía _env_float para diagnóstico"
    )
    info = snap["MEALFIT_DYNAMIC_RATE_HOUSEHOLD_DRIFT_THRESHOLD"]
    assert info["default"] == 0.20
    assert info["type"] == "float"
