"""[P1-PHOTO-DEDUCTS + P1-VISION-PLATO-ITEMS · 2026-08-07] La foto ya descuenta.

Escanear comida era la unica de las tres superficies de consumo que NO tocaba la
Nevera, y no por falta de datos: al modelo YA se le pedia el inventario completo
de componentes del plato — pero en 'description', texto libre. Dos lineas lo
tiraban:

  vision_agent._MEAL_VISION_PROMPT   "SI ES 'plato' (deja items vacio)"
  vision_agent._coerce_meal_scan     "items": []   <- HARDCODED en modo plato

y aunque hubieran sobrevivido, el endpoint no los habria aceptado:

  routers.diary.ConsumedMealRequest  sin campo `ingredients`
                                     + model_config extra="ignore"
                                     -> un cliente que los mandara los perdia
                                        EN SILENCIO

Este archivo cubre las tres capas: que el modelo los pida, que la sanitizacion
los deje pasar, y que el endpoint los acepte y descuente.

Nota sobre por que aqui SI puede el cliente mandar ingredientes (y en
`consumed-from-plan` no): alli describen un plato del PLAN, que el backend puede
releer y verificar — aceptarlos del cliente seria dejarle declarar el contenido
de un dato que el servidor ya posee. Aqui describen lo que el usuario declara
haber comido fuera del plan, igual que si lo escribiera en el chat; no hay
fuente server-side contra la cual verificarlos. La confirmacion humana en el
modal es la autorizacion.
"""
import re
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_VISION = _BACKEND / "vision_agent.py"
_DIARY = _BACKEND / "routers" / "diary.py"


# ---------------------------------------------------------------------------
# 1. Vision: el plato ahora trae sus componentes
# ---------------------------------------------------------------------------

def _plato(**over):
    base = {
        "photo_kind": "plato", "is_food": True,
        "meal_name": "Mangú con los tres golpes",
        "description": "mangu, huevo frito, salami, queso frito",
        "calories": 750, "protein": 35, "carbs": 80, "healthy_fats": 30,
        "items": [
            {"name": "huevo", "quantity": 2, "unit": "unidad"},
            {"name": "queso frito", "quantity": 2, "unit": "lasca"},
        ],
    }
    base.update(over)
    return base


def test_plato_carries_its_components():
    """El corazon del fix: antes esto devolvia [] pasara lo que pasara."""
    from vision_agent import _coerce_meal_scan
    out = _coerce_meal_scan(_plato())
    assert out["photo_kind"] == "plato"
    assert [i["name"] for i in out["items"]] == ["huevo", "queso frito"]
    assert out["items"][0]["quantity"] == 2
    assert out["items"][1]["unit"] == "lasca"
    # Las macros del contrato v3 siguen intactas.
    assert out["calories"] == 750 and out["protein"] == 35


def test_plato_items_are_sanitized_like_the_items_mode():
    """Misma sanitizacion que el modo 'items' — no una segunda ruta paralela
    que pueda driftear (la casa ya tiene esa familia de bugs)."""
    from vision_agent import _coerce_meal_scan
    out = _coerce_meal_scan(_plato(items=[
        {"name": "", "quantity": 1, "unit": "unidad"},              # sin nombre → fuera
        {"name": "  huevo  ", "quantity": "2", "unit": "  UNIDAD "},  # trim + lower
        {"name": "x" * 200, "quantity": 1, "unit": "u"},             # name capado
    ]))
    nombres = [i["name"] for i in out["items"]]
    assert "" not in nombres
    assert "huevo" in nombres
    assert all(len(n) <= 60 for n in nombres)
    assert out["items"][0]["unit"] == "unidad"


def test_plato_with_no_items_is_still_a_plato():
    """A diferencia del modo 'items' (que degrada a 'otro' si no identifica
    nada), un plato sin desglose sigue siendo un registro de diario valido:
    tiene macros. Degradarlo perderia el registro calorico entero."""
    from vision_agent import _coerce_meal_scan
    out = _coerce_meal_scan(_plato(items=[]))
    assert out["photo_kind"] == "plato"
    assert out["is_food"] is True
    assert out["items"] == []
    assert out["calories"] == 750


def test_items_mode_and_otro_mode_unchanged():
    """Anti-regresion: el fix toca SOLO la rama 'plato'."""
    from vision_agent import _coerce_meal_scan
    compra = _coerce_meal_scan({
        "photo_kind": "items", "is_food": True, "meal_name": "", "description": "compra",
        "calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0,
        "items": [{"name": "arroz blanco", "quantity": 1, "unit": "paquete"}],
    })
    assert compra["photo_kind"] == "items" and compra["calories"] == 0
    otro = _coerce_meal_scan({"photo_kind": "otro", "is_food": False, "items": [],
                              "description": "", "meal_name": "",
                              "calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0})
    assert otro["photo_kind"] == "otro" and otro["items"] == []


def test_prompt_asks_for_components_on_plato():
    """Parser-based: si alguien reescribe el prompt y vuelve a decirle al
    modelo que deje `items` vacio en 'plato', la foto deja de descontar en
    silencio — y solo se notaria en produccion."""
    # Se inspecciona el STRING EN RUNTIME, no el fuente: el prompt se arma
    # concatenando literales con comentarios intercalados, y los comentarios
    # (que citan la instruccion vieja para explicar el fix) darian falsos
    # positivos contra el fuente crudo. Lo que se le manda al modelo es esto.
    from vision_agent import _MEAL_VISION_PROMPT as P
    assert "P1-VISION-PLATO-ITEMS" in _VISION.read_text(encoding="utf-8"), "marker ausente"
    assert "deja items vacio" not in P, (
        "el prompt volvio a pedirle al modelo que deje `items` vacio para "
        "'plato' — eso desconecta la foto de la Nevera."
    )
    assert re.search(r"SI ES 'plato'[\s\S]{0,400}llena 'items'", P), (
        "el prompt ya no pide los componentes estructurados para 'plato'."
    )
    # La cantidad pedida es la SERVIDA, no la del empaque: pedir la del
    # empaque descontaria un carton entero por un huevo frito.
    assert "SERVIDA EN EL PLATO" in P


def test_coerce_no_longer_hardcodes_empty_items_for_plato():
    """El `\"items\": []` literal en la rama plato era el bug."""
    src = _VISION.read_text(encoding="utf-8")
    plato_branch = src[src.index("# ---- Modo PLATO"):]
    plato_branch = plato_branch[:plato_branch.index("return result")] if "return result" in plato_branch else plato_branch
    assert '"items": []' not in plato_branch, (
        "la rama 'plato' volvio a hardcodear items vacios."
    )
    assert "plato_items" in plato_branch


# ---------------------------------------------------------------------------
# 2. El endpoint acepta ingredientes
# ---------------------------------------------------------------------------

def test_request_model_accepts_ingredients():
    from routers.diary import ConsumedMealRequest
    assert "ingredients" in ConsumedMealRequest.model_fields, (
        "sin este campo, `extra: ignore` descarta los ingredientes EN SILENCIO "
        "y la foto vuelve a no tocar la Nevera."
    )
    r = ConsumedMealRequest(meal_name="Mangú", ingredients=["2 huevos", "2 lascas de queso"])
    assert r.ingredients == ["2 huevos", "2 lascas de queso"]


def test_ingredients_are_cleaned_not_trusted_verbatim():
    from routers.diary import ConsumedMealRequest
    r = ConsumedMealRequest(meal_name="X", ingredients=["2 huevos", "  ", "ab", "  1 pan  ", "y" * 300])
    assert "  " not in r.ingredients and "ab" not in r.ingredients
    assert "1 pan" in r.ingredients          # trimmed
    assert all(len(i) <= 120 for i in r.ingredients)


def test_ingredients_all_junk_becomes_none():
    """`None` y `[]` deben ser indistinguibles aguas abajo: ambos significan
    'no hay nada que descontar', y `[]` disparaba `mark_inventory_synced` en
    algunas versiones del helper."""
    from routers.diary import ConsumedMealRequest
    assert ConsumedMealRequest(meal_name="X", ingredients=["a", ""]).ingredients is None
    assert ConsumedMealRequest(meal_name="X").ingredients is None


def test_ingredients_list_is_capped():
    from pydantic import ValidationError
    from routers.diary import ConsumedMealRequest
    with pytest.raises(ValidationError):
        ConsumedMealRequest(meal_name="X", ingredients=[f"{i} huevos" for i in range(41)])


# ---------------------------------------------------------------------------
# 3. Funcional: registrar descuenta, y dice lo que no bajo
# ---------------------------------------------------------------------------

_UID = "11111111-1111-4111-8111-111111111111"


def _post(ingredients=None, logged="row-1", summary=None):
    import routers.diary as diary
    from routers.diary import ConsumedMealRequest
    import db_inventory

    payload = ConsumedMealRequest(
        user_id=_UID, meal_name="Mangú con los tres golpes", meal_type="desayuno",
        calories=750, protein=35, carbs=80, healthy_fats=30,
        ingredients=ingredients,
    )
    default = {"succeeded": ["2 huevos"], "inferred": [],
               "failed_to_deduct": [], "not_in_pantry": ["2 lascas de queso frito"]}
    with patch.object(diary, "log_consumed_meal", return_value=logged) as log_mock, \
         patch.object(db_inventory, "deduct_consumed_meal_from_inventory",
                      return_value=(summary if summary is not None else default)) as ded_mock, \
         patch.object(diary, "trigger_incremental_learning"):
        out = diary.api_log_consumed_meal(payload, verified_user_id=_UID)
    return out, log_mock, ded_mock


def test_scanned_meal_deducts_the_confirmed_ingredients():
    out, log_mock, ded_mock = _post(ingredients=["2 huevos", "2 lascas de queso frito"])
    assert out["success"] is True
    ded_mock.assert_called_once()
    assert ded_mock.call_args[0][0] == _UID
    assert ded_mock.call_args[0][1] == ["2 huevos", "2 lascas de queso frito"]
    # Descontamos acto seguido → el cron de reconciliacion no debe repetirlo.
    assert log_mock.call_args[1]["mark_inventory_synced"] is True
    assert log_mock.call_args[1]["ingredients"] == ["2 huevos", "2 lascas de queso frito"]


def test_surfaces_what_did_not_come_off_the_fridge():
    out, _, _ = _post(ingredients=["2 huevos", "2 lascas de queso frito"])
    assert out["deducted"] == ["2 huevos"]
    assert out["not_in_pantry"] == ["2 lascas de queso frito"]


def test_without_ingredients_nothing_is_deducted():
    """Anti-regresion del comportamiento historico: registrar solo macros
    (usuario que no confirmo ningun ingrediente) sigue funcionando igual."""
    out, log_mock, ded_mock = _post(ingredients=None)
    assert out["success"] is True
    ded_mock.assert_not_called()
    assert log_mock.call_args[1]["mark_inventory_synced"] is False
    assert out["deducted"] == [] and out["not_in_pantry"] == []


def test_double_tap_does_not_deduct_twice():
    """P2-CONSUMED-DEDUP-INVENTORY: `"deduped"` = no hubo INSERT nuevo."""
    out, _, ded_mock = _post(ingredients=["2 huevos"], logged="deduped")
    assert out["already_logged"] is True
    ded_mock.assert_not_called(), (
        "re-tap dentro de la ventana: descontar otra vez bajaria la Nevera al "
        "DOBLE del consumo real."
    )


def test_failed_log_still_fails_loud():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        _post(ingredients=["2 huevos"], logged=None)
    assert exc.value.status_code == 500


# ---------------------------------------------------------------------------
# 4. Anclajes
# ---------------------------------------------------------------------------

def test_endpoint_gates_deduction_on_dedup_sentinel():
    # [P1-MANUAL-FOOD-LOG · 2026-08-11] El sentinel se mudó con el cuerpo a
    # `_persist_consumed_meal`, el camino común de la foto y el componedor manual. La
    # propiedad es la misma de siempre: un doble-tap dentro de la ventana NO descuenta
    # dos veces. La condición ganó el término `deduct` (el interruptor de Nevera del
    # componedor); si pierde el `not _already_logged`, el doble-tap vuelve a bajar la
    # Nevera al doble del consumo real.
    src = _DIARY.read_text(encoding="utf-8")
    body = src[src.index("def _persist_consumed_meal("):]
    body = body[:body.index("\n@router.")]
    assert '_logged_ok == "deduped"' in body
    assert "if _ingredients and deduct and not _already_logged" in body


def test_markers_alive():
    assert "P1-PHOTO-DEDUCTS" in _DIARY.read_text(encoding="utf-8")
    assert "P1-VISION-PLATO-ITEMS" in _VISION.read_text(encoding="utf-8")
