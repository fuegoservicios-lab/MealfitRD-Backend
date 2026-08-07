"""[P1-EAT-PLAN-MEAL · 2026-08-07] "Me comí este plato del plan".

El camino de consumo mas preciso del sistema: el plato del plan YA trae su
lista de ingredientes con cantidades (la misma que lee `reserve_plan_ingredients`),
asi que descontar la Nevera es aritmetica sobre datos que el backend escribio.
Cero LLM, cero vision, cero `_infer_typical_portion`.

Los dos invariantes que este archivo protege:

  1. **El cliente manda coordenadas, NUNCA contenido.** Nombre, slot, macros e
     ingredientes salen de `plan_data` leido server-side con `AND user_id = %s`.
     Un cliente que pudiera declarar `ingredients` descontaria de la Nevera lo
     que quisiera, en la cantidad que quisiera. Misma doctrina que I-Billing-1
     (`tier` derivado del plan_id de PayPal, no de `data.get("tier")`).

  2. **Quota-exempt.** Registrar lo que comes de un plan que YA pagaste no
     tiene costo LLM, y `get_monthly_api_usage` cuenta toda fila de `api_usage`
     sin filtrar endpoint — con `verify_api_quota` el usuario al cap no podria
     registrar comidas Y ademas quemaria credito de planes.
"""
import re
from pathlib import Path
from unittest.mock import patch

import pytest


_BACKEND = Path(__file__).resolve().parent.parent
_DIARY = _BACKEND / "routers" / "diary.py"


@pytest.fixture(scope="module")
def diary_src() -> str:
    return _DIARY.read_text(encoding="utf-8")


def _handler_src(src: str) -> str:
    start = src.index("def api_log_consumed_meal_from_plan")
    end = src.index("\n@router.", start)
    return src[start:end]


# ---------------------------------------------------------------------------
# 1. Contrato de entrada: coordenadas, no contenido
# ---------------------------------------------------------------------------

def test_request_model_accepts_only_coordinates(diary_src: str):
    """Si el modelo gana un campo de contenido, el cliente puede mentirle a la
    Nevera. Los unicos campos legitimos son coordenadas + backdate."""
    from routers.diary import ConsumedFromPlanRequest

    fields = set(ConsumedFromPlanRequest.model_fields)
    assert fields == {"plan_id", "day_index", "meal_index", "days_ago"}, (
        f"campos inesperados en ConsumedFromPlanRequest: {fields}. "
        f"Nombre/macros/ingredientes se LEEN del plan server-side; aceptarlos "
        f"del cliente abre descuento arbitrario de la Nevera."
    )
    for prohibido in ("ingredients", "calories", "protein", "meal_name", "meal_type"):
        assert prohibido not in fields


def test_request_model_ignores_extra_fields():
    """`extra: ignore` — un cliente que mande `ingredients` no debe provocar
    500 ni, peor, colarlos."""
    from routers.diary import ConsumedFromPlanRequest

    req = ConsumedFromPlanRequest(
        plan_id="p1", day_index=0, meal_index=0,
        ingredients=["500 lb caviar"], calories=99999,
    )
    assert not hasattr(req, "ingredients")
    assert not hasattr(req, "calories")


@pytest.mark.parametrize("bad", [
    {"plan_id": "p1", "day_index": -1, "meal_index": 0},
    {"plan_id": "p1", "day_index": 0, "meal_index": -1},
    {"plan_id": "p1", "day_index": 0, "meal_index": 0, "days_ago": 8},
    {"plan_id": "p1", "day_index": 0, "meal_index": 0, "days_ago": -1},
    {"plan_id": "", "day_index": 0, "meal_index": 0},
])
def test_request_model_rejects_structural_garbage(bad):
    from pydantic import ValidationError
    from routers.diary import ConsumedFromPlanRequest
    with pytest.raises(ValidationError):
        ConsumedFromPlanRequest(**bad)


# ---------------------------------------------------------------------------
# 2. Anclajes de seguridad sobre el handler
# ---------------------------------------------------------------------------

def test_read_filters_by_user_id(diary_src: str):
    """Invariante I2. Sin el filtro, cualquiera con un plan_id ajeno registra
    comidas en su propio diario leyendo el plan de otro."""
    body = _handler_src(diary_src)
    assert re.search(
        r"FROM meal_plans WHERE id = %s AND user_id = %s", body
    ), (
        "P1-EAT-PLAN-MEAL: el SELECT de `plan_data` perdio el filtro "
        "`AND user_id = %s` (invariante I2)."
    )


def test_handler_never_reads_payload_content_fields(diary_src: str):
    """El handler solo puede leer coordenadas del payload."""
    body = _handler_src(diary_src)
    leidos = set(re.findall(r"payload\.(\w+)", body))
    assert leidos <= {"plan_id", "day_index", "meal_index", "days_ago"}, (
        f"el handler lee del payload campos que no son coordenadas: "
        f"{leidos - {'plan_id', 'day_index', 'meal_index', 'days_ago'}}"
    )


def test_macros_and_ingredients_come_from_the_plan(diary_src: str):
    body = _handler_src(diary_src)
    for key in ('meal.get("name")', 'meal.get("meal")', 'meal.get("ingredients")'):
        assert key in body, f"{key} deberia leerse del plato del plan, no del cliente"
    # El esquema del plato usa `cals`/`fats`; el diario, `calories`/`healthy_fats`.
    assert '_macro("cals")' in body and '_macro("fats")' in body, (
        "las macros deben mapearse desde las claves del PLATO (`cals`/`fats`), "
        "no desde las del diario."
    )


def test_endpoint_is_quota_exempt(diary_src: str):
    """Doctrina P1-NEVERA-QUOTA-EXEMPT: registrar consumo no tiene costo LLM."""
    body = _handler_src(diary_src)
    assert "verify_api_quota" not in body, (
        "P1-EAT-PLAN-MEAL: al cap mensual el usuario no podria registrar lo que "
        "come, y cada registro quemaria credito de PLANES "
        "(`get_monthly_api_usage` no filtra por endpoint)."
    )
    assert "_PLAN_MEAL_LIMITER" in body, (
        "el anti-spam correcto es un RateLimiter per-bucket, no el paywall."
    )


def test_dedup_sentinel_blocks_double_deduction(diary_src: str):
    """P2-CONSUMED-DEDUP-INVENTORY: `"deduped"` significa que NO hubo INSERT.
    Descontar igual bajaria la Nevera al DOBLE del consumo real."""
    body = _handler_src(diary_src)
    assert 'logged == "deduped"' in body
    assert re.search(r"if ingredients and not already_logged", body), (
        "la deduccion debe estar gateada por el sentinel de dedup."
    )


def test_fail_loud_when_log_fails(diary_src: str):
    """Doctrina P1-PROD-AUDIT-3: sin esto un blip de DB devolvia success:true
    sobre data fantasma."""
    body = _handler_src(diary_src)
    assert "if not logged:" in body and "status_code=500" in body


def test_tooltip_anchor_alive(diary_src: str):
    assert "P1-EAT-PLAN-MEAL-ENDPOINT" in diary_src


# ---------------------------------------------------------------------------
# 3. Funcional
# ---------------------------------------------------------------------------

_UID = "11111111-1111-4111-8111-111111111111"
_PLAN = "22222222-2222-4222-8222-222222222222"

_MEAL = {
    "meal": "Desayuno",
    "name": "Mangú con Los Tres Golpes",
    "cals": 750, "protein": 35, "carbs": 80, "fats": 30,
    "ingredients": ["2 huevos", "1 platano verde", "2 lascas de queso frito"],
}
_PLAN_DATA = {"days": [{"meals": [_MEAL]}]}


_DEFAULT_ROW = object()  # sentinel: distingue "no pases plan_row" de "el SELECT no devolvio fila"


def _call(payload_kwargs=None, plan_row=_DEFAULT_ROW, deduct_summary=None):
    """Invoca el handler con la DB mockeada. Devuelve (resultado, mocks).

    `plan_row=None` significa que el SELECT no devolvio fila (plan ajeno o
    inexistente) — por eso el default es un sentinel y no `None`: con `None`
    como default, un test que quisiera probar el caso "sin fila" acababa
    probando el plan valido en silencio.
    """
    import routers.diary as diary
    from routers.diary import ConsumedFromPlanRequest

    payload = ConsumedFromPlanRequest(**{
        "plan_id": _PLAN, "day_index": 0, "meal_index": 0, **(payload_kwargs or {})
    })
    row = {"plan_data": _PLAN_DATA} if plan_row is _DEFAULT_ROW else plan_row

    import db_inventory
    with patch.object(diary, "execute_sql_query", return_value=row), \
         patch.object(diary, "log_consumed_meal", return_value="row-id-1") as log_mock, \
         patch.object(db_inventory, "deduct_consumed_meal_from_inventory",
                      return_value=(deduct_summary if deduct_summary is not None else {
                          "succeeded": ["2 huevos"], "inferred": [],
                          "failed_to_deduct": [], "not_in_pantry": ["2 lascas de queso frito"],
                      })) as deduct_mock, \
         patch.object(diary, "trigger_incremental_learning"):
        out = diary.api_log_consumed_meal_from_plan(payload, verified_user_id=_UID)
    return out, log_mock, deduct_mock


def test_logs_the_plan_meal_verbatim():
    out, log_mock, _ = _call()
    assert out["success"] is True
    assert out["meal_name"] == "Mangú con Los Tres Golpes"
    # El slot viaja normalizado para que el matcher por `meal_type` lo vea.
    assert out["meal_type"] == "desayuno"
    args, kwargs = log_mock.call_args
    assert args[0] == _UID
    assert args[1] == "Mangú con Los Tres Golpes"
    assert (args[2], args[3], args[4], args[5]) == (750, 35, 80, 30)
    assert kwargs["ingredients"] == _MEAL["ingredients"]
    # Descontamos acto seguido → el cron de reconciliacion no debe repetirlo.
    assert kwargs["mark_inventory_synced"] is True


def test_deducts_the_recipe_ingredients():
    _, _, deduct_mock = _call()
    assert deduct_mock.call_args[0][0] == _UID
    assert deduct_mock.call_args[0][1] == _MEAL["ingredients"]


def test_surfaces_what_did_not_come_off_the_fridge():
    """[P1-PANTRY-NAME-RESOLUTION] Si la UI no puede distinguir lo descontado
    de lo ausente, el usuario asume que todo bajó — la misma mentira que aquel
    P-fix elimino del lado del chat."""
    out, _, _ = _call()
    assert out["deducted"] == ["2 huevos"]
    assert out["not_in_pantry"] == ["2 lascas de queso frito"]


def test_double_tap_does_not_deduct_twice():
    import routers.diary as diary
    from routers.diary import ConsumedFromPlanRequest
    import db_inventory

    payload = ConsumedFromPlanRequest(plan_id=_PLAN, day_index=0, meal_index=0)
    with patch.object(diary, "execute_sql_query", return_value={"plan_data": _PLAN_DATA}), \
         patch.object(diary, "log_consumed_meal", return_value="deduped"), \
         patch.object(db_inventory, "deduct_consumed_meal_from_inventory") as deduct_mock, \
         patch.object(diary, "trigger_incremental_learning"):
        out = diary.api_log_consumed_meal_from_plan(payload, verified_user_id=_UID)

    assert out["already_logged"] is True
    deduct_mock.assert_not_called(), (
        "re-tap dentro de la ventana de dedup: no hubo INSERT nuevo, descontar "
        "otra vez bajaria la Nevera al DOBLE del consumo real."
    )


@pytest.mark.parametrize("kwargs", [
    {"day_index": 5},    # dia fuera de rango
    {"meal_index": 9},   # plato fuera de rango
])
def test_out_of_range_coordinates_are_404(kwargs):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        _call(payload_kwargs=kwargs)
    assert exc.value.status_code == 404


def test_foreign_plan_is_404_and_does_not_leak_existence():
    """El SELECT filtra `AND user_id = %s`, asi que un plan ajeno devuelve 0
    filas — indistinguible de uno inexistente, que es justo lo que queremos:
    el mensaje no debe permitir enumerar plan_ids de otros usuarios."""
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        _call(plan_row=None)
    assert exc.value.status_code == 404
    assert "no encontrado" in str(exc.value.detail).lower()


def test_meal_without_name_is_422_not_a_junk_row():
    from fastapi import HTTPException
    bad = {"plan_data": {"days": [{"meals": [{"meal": "Cena", "cals": 400}]}]}}
    with pytest.raises(HTTPException) as exc:
        _call(plan_row=bad)
    assert exc.value.status_code == 422


def test_meal_without_ingredients_logs_but_deducts_nothing():
    """Un plato viejo sin `ingredients` sigue contando calorias; simplemente no
    hay nada que restar. Y NO debe marcarse `inventory_synced`: el cron de
    reconciliacion tiene que poder recogerlo si el plato se completa luego."""
    sin_ing = {"plan_data": {"days": [{"meals": [
        {"meal": "Cena", "name": "Sopa", "cals": 300, "protein": 20}
    ]}]}}
    out, log_mock, deduct_mock = _call(plan_row=sin_ing)
    assert out["success"] is True and out["calories"] == 300
    assert out["deducted"] == [] and out["not_in_pantry"] == []
    deduct_mock.assert_not_called()
    assert log_mock.call_args[1]["mark_inventory_synced"] is False


def test_non_finite_or_negative_macros_do_not_contaminate_the_diary():
    """`plan_data` es JSONB: un plato corrupto puede traer NaN/negativos y
    contaminarian los agregados del dia."""
    raro = {"plan_data": {"days": [{"meals": [{
        "meal": "Cena", "name": "Raro",
        "cals": float("nan"), "protein": -50, "carbs": float("inf"), "fats": "x",
    }]}]}}
    out, log_mock, _ = _call(plan_row=raro)
    assert out["calories"] == 0
    assert (log_mock.call_args[0][2], log_mock.call_args[0][3]) == (0, 0)
    assert (log_mock.call_args[0][4], log_mock.call_args[0][5]) == (0, 0)


def test_backdating_is_clamped_and_applied():
    import routers.diary as diary
    from routers.diary import ConsumedFromPlanRequest
    import db_inventory

    payload = ConsumedFromPlanRequest(plan_id=_PLAN, day_index=0, meal_index=0, days_ago=2)
    with patch.object(diary, "execute_sql_query", return_value={"plan_data": _PLAN_DATA}), \
         patch.object(diary, "log_consumed_meal", return_value="id") as log_mock, \
         patch.object(db_inventory, "deduct_consumed_meal_from_inventory", return_value={}), \
         patch.object(diary, "trigger_incremental_learning"):
        diary.api_log_consumed_meal_from_plan(payload, verified_user_id=_UID)

    override = log_mock.call_args[1]["consumed_at_override"]
    assert override, "days_ago > 0 debe sellar la fecha real de consumo"


def test_unauthenticated_is_rejected():
    from fastapi import HTTPException
    import routers.diary as diary
    from routers.diary import ConsumedFromPlanRequest
    payload = ConsumedFromPlanRequest(plan_id=_PLAN, day_index=0, meal_index=0)
    with pytest.raises(HTTPException) as exc:
        diary.api_log_consumed_meal_from_plan(payload, verified_user_id=None)
    assert exc.value.status_code == 403
