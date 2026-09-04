"""[P1-4] Tests para garantizar que `humanize_plan_ingredients` preserva la
lista pre-humanización en `meal["ingredients_raw"]` y que el aggregator
de la lista de compras la prefiere sobre la versión humanizada al
re-procesar planes persistidos.

Bug original (audit P1-4):
  `humanize_plan_ingredients` muta `meal["ingredients"]` reemplazando los
  strings métricos del LLM ("200g pechuga de pollo") por strings legibles
  para la UI ("1 pechuga de pollo (porción)"). PERO cuando el plan
  persistido se re-aggrega después (regenerar via `/shift-plan`,
  recalcular shopping al cambiar `householdSize`, recolectar para el
  delta de un nuevo ciclo), `get_shopping_list_delta` lee `meal.ingredients`
  y los pasa a `_parse_quantity`. La versión humanizada pierde la unidad
  métrica:
     "1 pechuga de pollo (porción)" → match falla unidad métrica →
     unit='unidad', qty=1 → aggregator consolida "1 unidad pollo".
  Para listas escaladas ×4 mensual, la cantidad derivada del
  `density_g_per_unit` del master puede divergir significativamente del
  peso semántico real (4 × 200g = 800g vs 4 × density_per_unit_master
  = 4 × ??? que puede ser 320g o 1200g según el master), produciendo
  shopping lists descuadrados.

Fix:
  1. `humanize_plan_ingredients` clona la lista original a
     `meal["ingredients_raw"]` ANTES de mutar (idempotente: si ya existe
     no se sobrescribe).
  2. `get_shopping_list_delta` prefiere `meal.get("ingredients_raw")` sobre
     `meal.get("ingredients")` al recolectar para el aggregator.
  3. Planes legacy (pre-P1-4 sin `ingredients_raw`) caen al fallback
     `meal["ingredients"]` (humanizado, con el bug histórico ya conocido).

Cobertura:
  - test_humanize_preserves_raw_ingredients_in_meal
  - test_humanize_is_idempotent_does_not_re_overwrite_raw
  - test_humanize_only_creates_raw_when_ingredients_present
  - test_get_shopping_list_delta_prefers_ingredients_raw
  - test_get_shopping_list_delta_falls_back_to_ingredients_for_legacy
  - test_shopping_calculator_source_uses_ingredients_raw_first
"""
import inspect

import pytest

import shopping_calculator
import humanize_ingredients
from shopping_calculator import get_shopping_list_delta
from humanize_ingredients import humanize_plan_ingredients

# [P2-CI-BACKEND-SIBLINGS · 2026-09-04] Este módulo necesita el catálogo/la base de datos o el
# .env local (pasa en el checkout del dueño; en el CI sin NEON_DATABASE_URL se salta con motivo).
pytestmark = pytest.mark.needs_local_data


# ---------------------------------------------------------------------------
# 1. `humanize_plan_ingredients` setea `ingredients_raw` correctamente.
# ---------------------------------------------------------------------------
def test_humanize_preserves_raw_ingredients_in_meal():
    """Tras humanizar, `meal['ingredients_raw']` debe contener la lista
    original sin mutar."""
    plan = {
        "days": [{
            "meals": [{
                "name": "Almuerzo",
                "ingredients": ["200g pechuga de pollo", "1 taza arroz"],
            }],
        }],
    }
    humanized = humanize_plan_ingredients(plan)
    meal = humanized["days"][0]["meals"][0]
    assert "ingredients_raw" in meal
    assert meal["ingredients_raw"] == ["200g pechuga de pollo", "1 taza arroz"]
    # `ingredients` puede haberse mutado; `ingredients_raw` NO.
    assert isinstance(meal["ingredients"], list)


def test_humanize_is_idempotent_does_not_overwrite_raw():
    """Si `humanize_plan_ingredients` se llama dos veces, la segunda llamada
    NO debe sobrescribir `ingredients_raw` con la versión humanizada (eso
    perdería el original para siempre)."""
    plan = {
        "days": [{
            "meals": [{
                "ingredients": ["200g pechuga de pollo"],
            }],
        }],
    }
    # Primera llamada.
    humanize_plan_ingredients(plan)
    raw_after_first = list(plan["days"][0]["meals"][0]["ingredients_raw"])
    # Segunda llamada (la humanizada queda como entrada).
    humanize_plan_ingredients(plan)
    raw_after_second = plan["days"][0]["meals"][0]["ingredients_raw"]
    assert raw_after_second == raw_after_first, (
        "P1-4: ingredients_raw debe ser idempotente, NO sobrescribir con humanizado"
    )


def test_humanize_only_creates_raw_when_ingredients_present():
    """Meals sin `ingredients` no reciben `ingredients_raw` (no hay nada
    que preservar)."""
    plan = {
        "days": [{
            "meals": [{
                "name": "Suplemento",
                # sin "ingredients"
            }],
        }],
    }
    humanize_plan_ingredients(plan)
    assert "ingredients_raw" not in plan["days"][0]["meals"][0]


def test_humanize_clones_list_does_not_share_reference():
    """`ingredients_raw` debe ser una COPIA, no la misma referencia que
    `ingredients`. Si compartiera referencia, mutaciones futuras en
    `ingredients` (humanizado) corromperían `ingredients_raw`."""
    original_list = ["200g pechuga de pollo"]
    plan = {
        "days": [{
            "meals": [{"ingredients": original_list}],
        }],
    }
    humanize_plan_ingredients(plan)
    raw = plan["days"][0]["meals"][0]["ingredients_raw"]
    humanized = plan["days"][0]["meals"][0]["ingredients"]
    assert raw is not humanized, "P1-4: ingredients_raw debe ser una COPIA"


# ---------------------------------------------------------------------------
# 2. `get_shopping_list_delta` prefiere `ingredients_raw`.
# ---------------------------------------------------------------------------
def test_get_shopping_list_delta_prefers_ingredients_raw():
    """Si meal tiene tanto `ingredients_raw` (métrico) como `ingredients`
    (humanizado), el aggregator debe leer `ingredients_raw`."""
    plan = {
        "days": [{
            "meals": [{
                "ingredients": ["1 pechuga de pollo (porción)"],  # humanizada (sin métrica)
                "ingredients_raw": ["200g pechuga de pollo"],     # original métrico
            }],
        }],
    }
    # Llamamos sin user_id (guest path) para evitar query a inventory.
    result = get_shopping_list_delta(
        user_id=None, plan_result=plan, is_new_plan=True, structured=True,
    )
    # [REESCRITO · 2026-07-26] Antes esto afirmaba `market_unit in ("lb","lbs")`. Esa aserción
    # se volvió **CIEGA**, no solo caducada: desde P1-SUPERMARKET-DB el pollo se presenta en
    # envases del súper, y el redondeo al paquete de 454 g da el MISMO display por los dos
    # caminos —
    #
    #     leyendo ingredients_raw  (200 g/día)  ->  "3 paquetes (1 lb c/u)"   base_qty=1400 g
    #     leyendo ingredients      (1 pechuga)  ->  "3 paquetes (1 lb c/u)"   base_qty=1190 g
    #
    # — así que el test habría pasado o fallado igual leyera la lista que leyera. Justo lo que
    # NO puede hacer un test de regresión.
    #
    # `base_qty` (P1-COHERENCE-BASE-QTY · 2026-07-26) sí distingue: es la cantidad física antes
    # de la presentación de mercado. Se afirma sobre ella.
    pollo_items = [r for r in result if isinstance(r, dict) and "pollo" in r.get("name", "").lower()]
    assert pollo_items, "P1-4: debe agregar pollo desde ingredients_raw"
    item = pollo_items[0]
    base_qty = item.get("base_qty")
    assert item.get("base_unit") == "g", f"se esperaba base en gramos: {item}"
    # 200 g/día × 7 (el agregador proyecta a semana: base_duration_scale = 7/num_days).
    assert base_qty == pytest.approx(1400.0, rel=0.02), (
        f"P1-4: base_qty debe reflejar los 200 g de `ingredients_raw` proyectados a 7 días "
        f"(1400 g), got {base_qty}. ~1190 g significa que leyó la línea HUMANIZADA "
        f"('1 pechuga' ≈ 170 g)."
    )


def test_la_asercion_anterior_no_discriminaba():
    """Guarda contra volver a afirmar sobre la unidad de mercado.

    Se ejecutan los DOS caminos y se comprueba que su presentación coincide: mientras eso sea
    cierto, `market_unit`/`market_qty_numeric` no pueden distinguir si el agregador leyó la
    lista métrica o la humanizada. Si algún día divergen, este test falla y avisa de que la
    aserción vieja volvería a tener sentido — pero la de `base_qty` sigue siendo la correcta."""
    def _pollo(plan):
        for it in get_shopping_list_delta(user_id=None, plan_result=plan, is_new_plan=True,
                                          structured=True):
            if isinstance(it, dict) and "pollo" in str(it.get("name", "")).lower():
                return it
        return {}

    con_raw = _pollo({"days": [{"meals": [{
        "ingredients": ["1 pechuga de pollo (porción)"],
        "ingredients_raw": ["200g pechuga de pollo"]}]}]})
    sin_raw = _pollo({"days": [{"meals": [{
        "ingredients": ["1 pechuga de pollo (porción)"]}]}]})

    assert (con_raw.get("market_unit"), con_raw.get("market_qty_numeric")) == \
           (sin_raw.get("market_unit"), sin_raw.get("market_qty_numeric")), (
        "la presentación de mercado ya distingue los dos caminos — revisa si la aserción "
        "histórica sobre market_unit vuelve a ser útil"
    )
    assert con_raw.get("base_qty") != sin_raw.get("base_qty"), (
        "base_qty DEBE distinguirlos: es lo único que hace verificable esta invariante"
    )


def test_get_shopping_list_delta_falls_back_to_ingredients_for_legacy():
    """Plan legacy SIN `ingredients_raw` debe seguir funcionando vía
    `ingredients` (compatibilidad con planes pre-P1-4 ya persistidos)."""
    plan = {
        "days": [{
            "meals": [{
                "ingredients": ["200g pechuga de pollo"],
                # NO ingredients_raw (legacy)
            }],
        }],
    }
    result = get_shopping_list_delta(
        user_id=None, plan_result=plan, is_new_plan=True, structured=True,
    )
    # Debe seguir funcionando: el aggregator lee `ingredients` como antes.
    pollo_items = [r for r in result if isinstance(r, dict) and "pollo" in r.get("name", "").lower()]
    assert pollo_items, "P1-4: legacy fallback debe mantener compatibilidad"


# ---------------------------------------------------------------------------
# 3. Defensa textual contra reintroducción del patrón roto.
# ---------------------------------------------------------------------------
def test_humanize_source_creates_ingredients_raw():
    """Defensa: el source de `humanize_plan_ingredients` debe contener la
    asignación de `ingredients_raw`."""
    src = inspect.getsource(humanize_plan_ingredients)
    assert "ingredients_raw" in src, (
        "P1-4 regression: humanize_plan_ingredients debe preservar `ingredients_raw`"
    )


def test_shopping_calculator_source_uses_ingredients_raw_first():
    """Defensa: `get_shopping_list_delta` debe leer `ingredients_raw` ANTES
    de fallback a `ingredients`."""
    src = inspect.getsource(shopping_calculator)
    # Buscamos el patrón canónico: `meal.get("ingredients_raw") or meal.get("ingredients", ...)`.
    import re as _re
    pattern = _re.compile(
        r'meal\.get\(\s*["\']ingredients_raw["\']\s*\)\s*or\s*meal\.get\(\s*["\']ingredients["\']'
    )
    assert pattern.search(src), (
        "P1-4: get_shopping_list_delta debe usar el patrón "
        "`meal.get('ingredients_raw') or meal.get('ingredients', ...)`"
    )


def test_documentation_p1_4_present():
    """Comentario `[P1-4]` documenta el rationale."""
    src_humanize = inspect.getsource(humanize_ingredients)
    src_shopping = inspect.getsource(shopping_calculator)
    assert "[P1-4]" in src_humanize, "falta [P1-4] en humanize_ingredients"
    assert "[P1-4]" in src_shopping, "falta [P1-4] en shopping_calculator"
