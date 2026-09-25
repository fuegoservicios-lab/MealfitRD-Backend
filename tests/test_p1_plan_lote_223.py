# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-223 · 2026-09-24] El escáner de comida, reconstruido.

Un tester de Android, con captura de «Revisa y registra»: «no me deja quitar el 0 para agregar otro número». El dueño,
encima: «también debería poder mandarse platos múltiples como en el agente IA chat… hazlo lo mejor y más cómodo
posible para el usuario». Lo que toca al backend:

  1. Lo que aporta cada ingrediente. Desmarcar las albóndigas no movía las calorías porque el análisis traía el total
     de la foto y una lista de componentes sin sus cifras. El prompt YA pedía estimar cada componente por separado y
     sumar; ahora cada entrada de `items` trae sus cuatro cifras y `_coerce_meal_scan` reparte los totales YA
     corregidos en esas proporciones: la suma de los componentes es exactamente lo que el modal precarga.
  2. La media porción de un plato servido (½ taza de habichuelas) deja de redondearse a 1, y quince lascas dejan de
     colapsar a UNA (la regla «más de 12 es un peso impreso mal leído» es de la compra, no de un plato).
  3. `POST /api/diary/consumed` acepta `deduct_pantry`: la Nevera es un interruptor propio y la casilla del ingrediente
     pasa a decir «lo comí». Solo un `False` explícito apaga la resta; ausente = la conducta de siempre.

Las pruebas funcionales de la pantalla viven en `frontend/src/__tests__/lote223.test.jsx`; aquí, además, las anclas
que el frontend tiene que conservar (el test las salta mientras el frontend de al lado no traiga el lote 223).

Tooltip-anchor: P1-PLAN-LOTE-223
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"
_UID = "11111111-1111-4111-8111-111111111111"


# El CI del backend clona el `main` del frontend (su checkout no fija `ref:`). Mientras el lote 223 no llegue allí, ese
# árbol trae el escáner ANTERIOR: los archivos existen, pero no son estos. La prueba del lote en el frontend dice qué
# árbol es; con ella presente, un archivo que falte es deriva y el test falla en vez de saltarse.
_LOTE_EN_EL_FRONTEND = "src/__tests__/lote223.test.jsx"


def _front(rel: str) -> str:
    if not (_FRONT / _LOTE_EN_EL_FRONTEND).exists():
        pytest.skip("el frontend de este checkout aún no trae el lote 223 (el CI del backend clona su `main`)")
    return (_FRONT / rel).read_text(encoding="utf-8").replace("\r\n", "\n")


def _plato(**over):
    base = {
        "photo_kind": "plato", "is_food": True,
        "meal_name": "Espaguetis en salsa con albondigas",
        "description": "espaguetis en salsa, albondigas, queso rallado",
        "calories": 900, "protein": 45, "carbs": 100, "healthy_fats": 35,
        "items": [
            {"name": "espaguetis en salsa", "quantity": 2, "unit": "taza",
             "calories": 440, "protein": 14, "carbs": 86, "healthy_fats": 6},
            {"name": "albondigas", "quantity": 4, "unit": "unidad",
             "calories": 380, "protein": 26, "carbs": 12, "healthy_fats": 25},
            {"name": "queso rallado", "quantity": 1, "unit": "cucharada",
             "calories": 30, "protein": 2, "carbs": 0, "healthy_fats": 2},
        ],
    }
    base.update(over)
    return base


def _sumas(items):
    return {k: round(sum(i["macros"][k] for i in items), 1) for k in ("calories", "protein", "carbs", "healthy_fats")}


# ───────────────────────── 1. Lo que aporta cada ingrediente ─────────────────────────

def test_cada_componente_lleva_su_parte_y_la_suma_es_el_total_que_se_precarga():
    from vision_agent import _coerce_meal_scan
    out = _coerce_meal_scan(_plato())
    assert [set(i) for i in out["items"]] == [{"name", "quantity", "unit", "macros"}] * 3
    totales = {k: out[k] for k in ("calories", "protein", "carbs", "healthy_fats")}
    for k, v in _sumas(out["items"]).items():
        assert abs(v - totales[k]) <= 0.2, (k, v, totales[k])
    # la proporción es la que dio el modelo: los espaguetis siguen pesando más que las albóndigas
    kcal = [i["macros"]["calories"] for i in out["items"]]
    assert kcal[0] > kcal[1] > kcal[2]


def test_se_reparten_los_totales_YA_corregidos_no_las_cifras_crudas():
    """Con kcal incoherentes con sus macros manda 4P+4C+9G (lote 53): los componentes se llevan ESE total."""
    from vision_agent import _coerce_meal_scan
    out = _coerce_meal_scan(_plato(calories=2000))   # 4·45 + 4·100 + 9·35 = 895
    assert out["calories"] == 895
    assert abs(_sumas(out["items"])["calories"] - 895) <= 0.2


def test_con_el_tope_de_un_plato_los_componentes_bajan_en_la_misma_proporcion():
    from vision_agent import _coerce_meal_scan, _MEAL_KCAL_PLAUSIBLE_MAX
    out = _coerce_meal_scan(_plato(calories=3400, protein=150, carbs=400, healthy_fats=110))
    assert out["calories"] == _MEAL_KCAL_PLAUSIBLE_MAX and out.get("low_confidence") is True
    assert abs(_sumas(out["items"])["calories"] - _MEAL_KCAL_PLAUSIBLE_MAX) <= 0.2


def test_un_macro_sin_desglosar_se_reparte_por_las_calorias():
    from vision_agent import _coerce_meal_scan
    items = [dict(i, healthy_fats=0) for i in _plato()["items"]]
    out = _coerce_meal_scan(_plato(items=items))
    assert abs(_sumas(out["items"])["healthy_fats"] - out["healthy_fats"]) <= 0.2
    assert out["items"][0]["macros"]["healthy_fats"] > 0


@pytest.mark.parametrize("items", [
    [{"name": "huevo", "quantity": 2, "unit": "unidad"}, {"name": "queso frito", "quantity": 2, "unit": "lasca"}],
    [{"name": "huevo", "quantity": 2, "unit": "unidad", "calories": 0}, {"name": "pan", "quantity": 1, "unit": "unidad"}],
])
def test_sin_desglose_usable_no_se_inventa(items):
    """Sin cifras por componente (un modelo que no las dé): los componentes van SIN `macros` y el modal registra el
    total de la foto por la porción, como siempre."""
    from vision_agent import _coerce_meal_scan
    out = _coerce_meal_scan(_plato(items=items))
    assert out["items"] and all("macros" not in i for i in out["items"])
    assert out["calories"] == 895 or out["calories"] == 900


def test_un_plato_en_cero_kcal_no_lleva_desglose():
    from vision_agent import _repartir_macros_del_plato
    crudas = [{"calories": 100, "protein": 5, "carbs": 10, "healthy_fats": 3}]
    assert _repartir_macros_del_plato(crudas, {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}) is None
    assert _repartir_macros_del_plato([], {"calories": 500}) is None


def test_la_compra_no_cambia_ni_una_coma():
    """Modo 'items': sin `macros`, con las reglas de siempre (el peso impreso leído como piezas colapsa a 1)."""
    from vision_agent import _coerce_meal_scan
    out = _coerce_meal_scan({"photo_kind": "items", "is_food": True, "description": "Compra.",
                             "items": [{"name": "arroz", "quantity": 500, "unit": "paquete", "calories": 900},
                                       {"name": "pechuga de pollo", "quantity": 2, "unit": "lb"}]})
    assert out["items"] == [{"name": "arroz", "quantity": 1.0, "unit": "paquete"},
                            {"name": "pechuga de pollo", "quantity": 2.0, "unit": "lb"}]
    assert out["calories"] == 0


# ───────────────────────── 2. Cantidades de un plato servido ─────────────────────────

@pytest.mark.parametrize("qty,unit,esperada", [
    (0.5, "taza", 0.5),        # antes 1: media taza de habichuelas
    (1.5, "unidad", 1.5),      # antes 2
    (15, "lasca", 15.0),       # antes 1: la regla de la compra («>12 es un peso impreso»)
    (0.2, "cucharada", 0.5),   # piso de media
    (99, "unidad", 30.0),      # techo de siempre
    (150, "g", 150.0),         # gramos y libras, como siempre
    (float("nan"), "unidad", 1.0),
])
def test_la_media_porcion_de_un_plato_servido_existe(qty, unit, esperada):
    from vision_agent import _sane_item_qty
    assert _sane_item_qty(qty, unit, plato=True) == esperada


def test_la_compra_conserva_su_saneado():
    from vision_agent import _sane_item_qty
    assert _sane_item_qty(15, "lasca") == 1.0
    assert _sane_item_qty(0.5, "taza") == 1.0
    assert _sane_item_qty(500, "paquete") == 1.0


def test_el_plato_usa_el_mismo_helper_en_su_modo():
    src = (_BACKEND / "vision_agent.py").read_text(encoding="utf-8")
    assert 'qty = _sane_item_qty((it or {}).get("quantity"), unit, plato=True)' in src
    assert "_repartir_macros_del_plato(crudas, result)" in src


# ───────────────────────── 3. El prompt y el esquema ─────────────────────────

def test_el_prompt_pide_las_cifras_de_cada_componente_y_sigue_siendo_ascii():
    from vision_agent import _MEAL_VISION_PROMPT as P
    assert P.isascii()
    assert "'calories', 'protein', 'carbs' y" in P and "de ESE componente en esa cantidad" in P
    assert "0.5 taza de habichuelas" in P
    # lo que ya anclaban otros lotes sigue ahí
    assert re.search(r"SI ES 'plato'[\s\S]{0,400}llena 'items'", P)


def test_el_esquema_las_admite_sin_volverlas_obligatorias():
    from vision_agent import _MEAL_VISION_SCHEMA, _MealVisionItem
    item = _MEAL_VISION_SCHEMA["properties"]["items"]["items"]
    assert set(item["required"]) == {"name", "quantity", "unit"}
    assert {"calories", "protein", "carbs", "healthy_fats"} <= set(item["properties"])
    vacio = _MealVisionItem(name="huevo")
    assert (vacio.calories, vacio.protein, vacio.carbs, vacio.healthy_fats) == (0, 0, 0, 0)


# ───────────────────────── 4. `deduct_pantry` en /consumed ─────────────────────────

def _post(deduct_pantry="__ausente__"):
    import routers.diary as diary
    from routers.diary import ConsumedMealRequest
    import db_inventory

    kw = dict(user_id=_UID, meal_name="Espaguetis", meal_type="cena", calories=900, protein=45, carbs=100,
              healthy_fats=35, ingredients=["2 taza de espaguetis en salsa", "4 unidad de albondigas"])
    if deduct_pantry != "__ausente__":
        kw["deduct_pantry"] = deduct_pantry
    payload = ConsumedMealRequest(**kw)
    with patch.object(diary, "log_consumed_meal", return_value="row-1") as log_mock, \
         patch.object(db_inventory, "deduct_consumed_meal_from_inventory",
                      return_value={"succeeded": ["4 unidad de albondigas"], "inferred": [],
                                    "failed_to_deduct": [], "not_in_pantry": []}) as ded_mock, \
         patch.object(diary, "nevera_activa", return_value=True), \
         patch.object(diary, "trigger_incremental_learning"):
        out = diary.api_log_consumed_meal(payload, verified_user_id=_UID)
    return out, log_mock, ded_mock


def test_sin_el_campo_se_descuenta_como_siempre():
    out, log_mock, ded_mock = _post()
    ded_mock.assert_called_once()
    assert out["deducted"] == ["4 unidad de albondigas"]


def test_true_descuenta():
    _, _, ded_mock = _post(True)
    ded_mock.assert_called_once()


def test_false_GUARDA_los_ingredientes_pero_no_toca_la_Nevera():
    out, log_mock, ded_mock = _post(False)
    ded_mock.assert_not_called()
    assert out["success"] is True and out["deducted"] == []
    # se guardan (son el detalle de la comida) y quedan sincronizados: el cierre del chunk no los descontará después
    assert log_mock.call_args[1]["ingredients"] == ["2 taza de espaguetis en salsa", "4 unidad de albondigas"]
    assert log_mock.call_args[1]["mark_inventory_synced"] is True


def test_el_modelo_acepta_el_campo_y_lo_deja_en_None_por_defecto():
    from routers.diary import ConsumedMealRequest
    assert ConsumedMealRequest(meal_name="X").deduct_pantry is None
    assert ConsumedMealRequest(meal_name="X", deduct_pantry=False).deduct_pantry is False
    src = (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")
    assert "deduct=payload.deduct_pantry is not False," in src


# ───────────────────────── 5. Anclas del frontend ─────────────────────────

def test_el_escaner_manda_lo_que_el_servidor_espera():
    sm = _front("src/components/dashboard/ScanMealModal.jsx")
    assert "deduct_pantry: descontar," in sm
    assert "ingredients: ingredientesParaGuardar(p)," in sm
    assert "days_ago: daysAgo," in sm
    dishes = _front("src/components/dashboard/scanMealDishes.js")
    # el formato que `_parse_quantity` entiende, con punto decimal
    assert "`${cantidadParaServidor(c.qty)} ${c.unit} de ${c.name}`" in dishes
    assert "export const MAX_PLATOS = 4;" in dishes
    # las cifras por componente que manda el servidor son las que usa el modal
    assert "it.macros && typeof it.macros === 'object'" in dishes


def test_la_cantidad_se_puede_borrar_y_escribir():
    qs = _front("src/components/common/QuantityStepper.jsx")
    assert 'type="text"' in qs and 'inputMode="decimal"' in qs
    assert "if (n !== null) onChange(Math.min(n, tope));" in qs
    sm = _front("src/components/dashboard/ScanMealModal.jsx")
    assert "import QuantityStepper from '../common/QuantityStepper';" in sm
    assert "chooseNativeGalleryImages(Math.max(1, MAX_PLATOS - platosRef.current.length))" in sm


@pytest.mark.parametrize("loc", ["en-US", "pt-BR", "fr-FR", "it-IT"])
def test_los_textos_nuevos_estan_traducidos(loc):
    cat = json.loads(_front(f"src/i18n/locales/{loc}.json"))
    for k in ("¿Comiste algo más?", "Añadir otro plato", "Registrar {n} platos", "Menos {nombre}", "Más {nombre}",
              "Desmarca lo que no comiste o ajusta la cantidad: las calorías se recalculan solas."):
        assert cat.get(k), f"{loc}: falta «{k}»"


def test_el_marcador_esta_al_dia():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'^_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · ', src, re.M)
    assert m and int(m.group(1)) >= 223
    assert "P1-PLAN-LOTE-223" in (_BACKEND / "docs" / "diario_registrar_comida.md").read_text(encoding="utf-8")
