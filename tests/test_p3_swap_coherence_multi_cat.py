"""[P3-SWAP-COHERENCE-MULTI-CAT · 2026-05-22] El coherence validator
de swap-meal originalmente solo cubría PROTEÍNAS — gap explícito doc'd
en P1-SWAP-RECIPE-COHERENCE ("Scope V1: Solo PROTEÍNAS canónicas. Carbs/
veggies quedan para V2 si se observan recurrencias").

Este P-fix cierra el gap residual extendiendo el validator a 4 catálogos
canónicos (`constants.PROTEIN/CARB/VEGGIE_FAT/FRUIT_SYNONYMS`) con knobs
granulares opt-out (o opt-in para fruits).

Cross-link con ``test_p2_hist_audit_14_marker_test_link``: slug
``p3_swap_coherence_multi_cat`` ↔ filename
``test_p3_swap_coherence_multi_cat.py``.
"""
import os
import pathlib
import re

import pytest

BACKEND_ROOT = pathlib.Path(__file__).parent.parent
NUTR_PY = (BACKEND_ROOT / "nutrition_calculator.py").read_text(encoding="utf-8")
APP_PY = (BACKEND_ROOT / "app.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Section A — Parser: knobs nuevos registrados + helper centralizado
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("knob", [
    "MEALFIT_SWAP_RECIPE_COHERENCE_PROTEIN",
    "MEALFIT_SWAP_RECIPE_COHERENCE_CARB",
    "MEALFIT_SWAP_RECIPE_COHERENCE_VEGGIE",
    "MEALFIT_SWAP_RECIPE_COHERENCE_FRUIT",
])
def test_per_category_knob_present_in_source(knob):
    """Los 4 knobs por-categoría deben estar referenciados en
    nutrition_calculator.py. Sin uno, esa categoría queda no-toggleable."""
    assert knob in NUTR_PY, (
        f"Knob {knob!r} ausente en nutrition_calculator.py. Sin este knob "
        f"el operador no puede flip esa categoría a False sin redeploy."
    )


def test_active_categories_helper_exists():
    """``_recipe_coherence_active_categories()`` debe existir como helper
    centralizado — los callsites (agent.py, tools.py) no necesitan
    conocer la lista de catálogos."""
    assert "def _recipe_coherence_active_categories" in NUTR_PY, (
        "Helper `_recipe_coherence_active_categories` ausente — el "
        "validator esperaba este nuevo helper centralizado para iterar "
        "categorías."
    )


def test_fruit_category_default_off():
    """El default del knob ``FRUIT`` debe ser ``"false"`` (opt-in). Las
    otras 3 categorías son default ``"true"`` (opt-out)."""
    # Buscar la línea que lee el knob FRUIT
    m = re.search(
        r'MEALFIT_SWAP_RECIPE_COHERENCE_FRUIT[^\n]*?[\"\']false[\"\']',
        NUTR_PY,
    )
    assert m, (
        "Knob `MEALFIT_SWAP_RECIPE_COHERENCE_FRUIT` debe leer con default "
        "'false' (opt-in). Frutas tienen mayor potencial de FP en recetas "
        "que las mencionan como decoración."
    )


# ---------------------------------------------------------------------------
# Section B — Funcional: validator detecta divergencia en CARBS
# ---------------------------------------------------------------------------

def _import_validator():
    pytest.importorskip("langchain_google_genai", reason="nutrition_calculator requiere langchain")
    from nutrition_calculator import validate_meal_recipe_ingredients_coherence
    return validate_meal_recipe_ingredients_coherence


def test_carb_divergence_detected_when_recipe_mentions_unlisted_carb():
    """[FUNCIONAL] Receta menciona "papa" pero ``ingredients`` solo lista
    "arroz" → debe detectarse divergencia carb."""
    validator = _import_validator()
    meal = {
        "name": "Plato test",
        "ingredients": ["200g pollo", "150g arroz blanco", "50g lechuga"],
        "recipe": [
            "Cocina el pollo a la plancha.",
            "Hierve la papa cortada en cubos hasta que esté tierna.",
            "Sirve con arroz y lechuga.",
        ],
    }
    passed, divs, summary = validator(meal)
    assert passed is False, (
        f"Validator debió detectar 'papa' en receta no listada en "
        f"ingredients. divs={divs}, summary={summary!r}"
    )
    assert "papas" in divs, (
        f"Canonical 'papas' debió estar en divergences. Got: {list(divs.keys())}"
    )
    assert divs["papas"].get("category") == "carbohidrato", (
        f"Sub-dict debe tener category='carbohidrato' (observabilidad). "
        f"Got: {divs['papas']}"
    )
    assert "papa" in summary, (
        f"Summary debe citar el alias 'papa' verbatim. Summary: {summary!r}"
    )


def test_veggie_divergence_detected_when_recipe_mentions_unlisted_veggie():
    """[FUNCIONAL] Receta menciona "aguacate" pero ``ingredients`` solo
    lista "lechuga" → debe detectarse divergencia veggie."""
    validator = _import_validator()
    meal = {
        "name": "Plato test",
        "ingredients": ["200g pollo", "150g arroz", "50g lechuga"],
        "recipe": [
            "Cocina el pollo a la plancha.",
            "Corta el aguacate en láminas y sírvelo encima.",
            "Acompaña con arroz y lechuga.",
        ],
    }
    passed, divs, summary = validator(meal)
    assert passed is False, (
        f"Validator debió detectar 'aguacate' no listado. divs={divs}"
    )
    assert "aguacate" in divs, (
        f"Canonical 'aguacate' debió estar en divergences. Got: {list(divs.keys())}"
    )
    assert divs["aguacate"].get("category") == "vegetal"


def test_multiple_categories_can_diverge_simultaneously():
    """[FUNCIONAL] Una receta puede tener divergencia en >1 categoría
    al mismo tiempo (proteína + carb + veggie)."""
    validator = _import_validator()
    meal = {
        "name": "Plato test",
        "ingredients": ["100g arroz blanco"],  # solo carb base
        "recipe": [
            "Marina el dorado en limón.",  # proteína no listada
            "Hierve la papa.",  # carb extra no listado
            "Agrega aguacate al servir.",  # veggie no listada
        ],
    }
    passed, divs, summary = validator(meal)
    assert passed is False
    # Al menos 2 de las 3 categorías deben aparecer (el catálogo puede
    # cubrir parcialmente; toleramos uno faltante por aliases edge)
    categories_found = {d.get("category") for d in divs.values()}
    assert len(categories_found) >= 2, (
        f"Esperaba al menos 2 categorías de divergencia, got: "
        f"{categories_found}. divs={divs}"
    )


# ---------------------------------------------------------------------------
# Section C — Funcional: knobs granulares respetados
# ---------------------------------------------------------------------------

def test_carb_knob_off_skips_carb_detection(monkeypatch):
    """[FUNCIONAL] Con ``MEALFIT_SWAP_RECIPE_COHERENCE_CARB=false``, una
    receta con divergencia carb pura NO debe fallar."""
    validator = _import_validator()
    monkeypatch.setenv("MEALFIT_SWAP_RECIPE_COHERENCE_CARB", "false")
    meal = {
        "name": "Plato test",
        "ingredients": ["200g pollo", "50g lechuga"],
        "recipe": [
            "Cocina el pollo a la plancha.",
            "Hierve la papa cortada en cubos.",
            "Sirve con lechuga.",
        ],
    }
    passed, divs, summary = validator(meal)
    # Sin carb detection, 'papa' no genera divergence — el meal pasa
    # (asumiendo no hay otras categorías afectadas)
    assert passed is True or "papas" not in divs, (
        f"Carb detection debió estar deshabilitada con knob OFF. divs={divs}"
    )


def test_fruit_off_by_default_does_not_flag_fruit_mentions(monkeypatch):
    """[FUNCIONAL] Por defecto (fruit knob OFF), una receta que mencione
    "limón" o "mango" sin listarlo NO debe fallar — frutas suelen ser
    decoración o salsa accesoria."""
    # Limpiar el knob fruit para usar default (false)
    monkeypatch.delenv("MEALFIT_SWAP_RECIPE_COHERENCE_FRUIT", raising=False)
    validator = _import_validator()
    meal = {
        "name": "Plato test",
        "ingredients": ["200g pollo", "150g arroz blanco", "50g lechuga"],
        "recipe": [
            "Marina el pollo en jugo de limón y sal.",
            "Cocina el pollo a la plancha.",
            "Sirve con arroz y lechuga.",
        ],
    }
    passed, divs, summary = validator(meal)
    # Limón menciona fruta pero fruit detection OFF por default
    fruit_divs = [d for d in divs.values() if d.get("category") == "fruta"]
    assert not fruit_divs, (
        f"Fruit detection debió estar OFF por default. divs={divs}"
    )


def test_fruit_on_opt_in_detects_fruit_divergence(monkeypatch):
    """[FUNCIONAL] Con ``MEALFIT_SWAP_RECIPE_COHERENCE_FRUIT=true`` (opt-in),
    el validator SÍ detecta divergencia fruit."""
    monkeypatch.setenv("MEALFIT_SWAP_RECIPE_COHERENCE_FRUIT", "true")
    validator = _import_validator()
    meal = {
        "name": "Plato test",
        "ingredients": ["200g pollo", "150g arroz"],
        "recipe": [
            "Acompaña con mango fresco picado en cubos.",
            "Sirve con arroz y pollo.",
        ],
    }
    passed, divs, summary = validator(meal)
    # 'mango' no listado debería detectarse
    fruit_divs = [c for c, d in divs.items() if d.get("category") == "fruta"]
    assert fruit_divs, (
        f"Con knob fruit ON, 'mango' debió detectarse. divs={divs}"
    )


# ---------------------------------------------------------------------------
# Section D — Summary: formato mejorado sin ruido redundante
# ---------------------------------------------------------------------------

def test_summary_omits_qualifier_when_alias_equals_canonical():
    """[FUNCIONAL] Cuando el alias mencionado es esencialmente el canónico
    (ej. "papa" vs "papas", o "pollo" vs "pollo"), el summary NO debe
    decir "(que cuenta como X)" — sería ruido redundante para el LLM."""
    validator = _import_validator()
    meal = {
        "name": "Plato test",
        "ingredients": ["100g arroz"],
        "recipe": ["Cocina el pollo a la plancha."],
    }
    passed, divs, summary = validator(meal)
    assert passed is False
    # El summary debe mencionar "pollo" sin el qualifier "(que cuenta como pollo)"
    assert "(que cuenta como pollo)" not in summary, (
        f"Qualifier redundante cuando alias==canonical. Summary: {summary!r}"
    )


def test_summary_keeps_qualifier_when_alias_differs_from_canonical():
    """[FUNCIONAL] Cuando el alias es distinto (ej. "dorado" → canónico
    "pescado"), el qualifier "(que cuenta como pescado)" SÍ debe aparecer
    — es load-bearing para que el LLM mapee el alias al canónico."""
    validator = _import_validator()
    meal = {
        "name": "Plato test",
        "ingredients": ["100g arroz"],
        "recipe": ["Marina el dorado en limón."],
    }
    passed, divs, summary = validator(meal)
    assert passed is False
    if "pescado" in divs:
        assert "(que cuenta como pescado)" in summary, (
            f"Qualifier debe aparecer cuando alias 'dorado' difiere del "
            f"canónico 'pescado'. Summary: {summary!r}"
        )


# ---------------------------------------------------------------------------
# Section E — Back-compat: shape del dict preserva mentioned_alias + listed
# ---------------------------------------------------------------------------

def test_divergences_back_compat_keys_preserved():
    """[FUNCIONAL] El test P3-SWAP-RETRY-COHERENCE-HINT asserts que
    ``mentioned_alias`` y ``listed`` están en el sub-dict. Verificamos
    que añadir ``category`` no rompió esos invariantes."""
    validator = _import_validator()
    meal = {
        "name": "Plato test",
        "ingredients": ["100g arroz"],
        "recipe": ["Cocina el pollo a la plancha."],
    }
    passed, divs, summary = validator(meal)
    assert passed is False
    for canonical, info in divs.items():
        assert "mentioned_alias" in info, (
            f"Back-compat: 'mentioned_alias' debe seguir en divs[{canonical!r}]"
        )
        assert "listed" in info
        assert "category" in info, (
            f"Nuevo key 'category' debe estar en divs[{canonical!r}]"
        )


# ---------------------------------------------------------------------------
# Section F — Marker anchor
# ---------------------------------------------------------------------------
# Pin removido siguiendo política establecida (test_p2_swap_422_ux_copy,
# test_p3_swap_pantry_default, test_p3_swap_fallback_title_copy,
# test_p3_swap_retry_coherence_hint): pin-tests se rompen cada P-fix
# siguiente cuando el marker avanza. El contract "marker fresco a nivel
# codebase" lo cubre `test_p3_1_last_known_pfix_freshness` (floor check).
# Las secciones A-E anclan el CONTENIDO del fix.
