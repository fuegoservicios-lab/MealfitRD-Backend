"""[P1-RECIPE-POLISH-5 · 2026-07-12] Batch de pulido de recetas (12 screenshots del owner,
plan vivo 1bfda745), en su orden de prioridad:

(a) Víver cook-required en paso-complemento: "0.5 pedazo de ñame" + "incorpora también...
    durante la preparación" SIN método de cocción — el ñame crudo no se come (la yuca cruda
    es tóxica). El complemento para víveres/leguminosas de _MUST_COOK_VIVER_LEGUME_RE emite
    paso explícito de hervido (descarta el agua).
(b) Display "NN g de queso" GENÉRICO (3/12 platos) cuyo raw nombra el queso específico
    ("47.98g de queso cottage") → display hereda el nombre del raw (la lista ya compraba lo
    correcto; defecto de display puro; solo renombra, jamás gramos/raw).
(c) Piso PROTAGONISTA para CARBOS: "cama de yautía asada" con 15g horneados 30 min → piso
    60g (knob MEALFIT_PROTAGONIST_CARB_MIN_G) headroom-aware, espejo del de proteína.
(d) "1 tazas de yogurt" / "1 dientes de ajo" → palabra-unidad singular (humanize regla c).
(e) Techo de pan ≤3 rebanadas/comida ("5 rebanadas" + queso "en 4 lonjas, 2 por tostada" —
    aritmética interna imposible; el techo la re-encuadra).

tooltip-anchor: P1-RECIPE-POLISH-5
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))


def test_a_viver_complement_gets_boil_step():
    from graph_orchestrator import _ensure_ingredients_used_in_recipe
    m = {"name": "Bowl de Camarones con Auyama",
         "ingredients": ["0.5 pedazo de ñame", "110 g de camarones"],
         "recipe": ["MISE EN PLACE: pela la auyama.", "Saltea los camarones.", "MONTAJE: sirve."]}
    assert _ensure_ingredients_used_in_recipe(m) >= 1
    blob = " ".join(m["recipe"]).lower()
    assert "hierve" in blob and "agua de cocc" in blob, (
        "víver cook-required sin paso → hervido explícito (el ñame crudo no se come)"
    )
    assert "incorpora también pedazo de ñame durante la preparación" not in blob


def test_a_non_viver_keeps_generic_complement():
    from graph_orchestrator import _ensure_ingredients_used_in_recipe
    m = {"name": "Ensalada Fresca",
         "ingredients": ["10 g de semillas de girasol", "1 taza de lechuga"],
         "recipe": ["MISE EN PLACE: lava la lechuga.", "MONTAJE: sirve la lechuga."]}
    n = _ensure_ingredients_used_in_recipe(m)
    blob = " ".join(m["recipe"]).lower()
    assert n >= 1 and "hierve" not in blob, "no-víveres conservan el complemento genérico"


def test_b_generic_cheese_display_enriched_from_raw():
    from graph_orchestrator import _enrich_generic_cheese_display_from_raw
    m = {"name": "Pepino con Ricotta",
         "ingredients": ["0.25 taza de queso ricotta", "20 g de queso"],
         "ingredients_raw": ["0.34 taza de queso ricotta", "47.98g de queso cottage"]}
    assert _enrich_generic_cheese_display_from_raw(m) == 1
    assert m["ingredients"][1] == "20 g de queso cottage"
    assert m["ingredients_raw"][1] == "47.98g de queso cottage", "el raw JAMÁS se toca"


def test_b_no_touch_when_display_already_specific():
    from graph_orchestrator import _enrich_generic_cheese_display_from_raw
    m = {"name": "Tostadas",
         "ingredients": ["25 g de queso blanco fresco"],
         "ingredients_raw": ["25 g de queso blanco fresco en lonjas"]}
    assert _enrich_generic_cheese_display_from_raw(m) == 0


def test_b_wired_in_both_finalizers():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count("_enrich_generic_cheese_display_from_raw(") >= 3, (
        "helper + finalize plan-level + finalizador single-meal (updates/swaps)"
    )


def test_c_carb_protagonist_floor():
    from graph_orchestrator import _floor_subservible_portions
    days = [{"day": 1, "meals": [{
        "name": "Pechuga Guisada con Cama de Yautía Asada",
        "ingredients": ["120 g de pechuga de pollo", "15 g de yautía"],
        "ingredients_raw": ["120 g de pechuga de pollo", "15 g de yautía"],
        "recipe": ["Hornea la yautía."],
        "protein": 30, "carbs": 10, "fats": 8, "cals": 280}]}]
    _floor_subservible_portions(days, day_kcal_target=None, db=None)
    y = next(i for i in days[0]["meals"][0]["ingredients"] if "yaut" in str(i).lower())
    assert y.startswith("60 g"), (
        "carbo protagonista (nombre∩línea) bajo el piso → bump a PROTAGONIST_CARB_MIN_G "
        "(vivo: cama de yautía asada con 15g)"
    )


def test_c_carb_not_in_name_untouched():
    from graph_orchestrator import _floor_subservible_portions
    days = [{"day": 1, "meals": [{
        "name": "Pechuga a la Plancha con Ensalada",
        "ingredients": ["120 g de pechuga de pollo", "30 g de arroz blanco"],
        "ingredients_raw": ["120 g de pechuga de pollo", "30 g de arroz blanco"],
        "recipe": ["Cocina."], "protein": 30, "carbs": 10, "fats": 8, "cals": 280}]}]
    _floor_subservible_portions(days, day_kcal_target=None, db=None)
    assert any(str(i).startswith("30 g de arroz") for i in days[0]["meals"][0]["ingredients"]), (
        "un carbo NO mencionado en el nombre no es protagonista — el guarnición de 30g queda"
    )


def test_c_knob_defined():
    import graph_orchestrator as go
    assert go.PROTAGONIST_CARB_MIN_G == 60.0
    assert "yautia" in go._PROTAGONIST_CARB_TOKENS and "name" in go._PROTAGONIST_CARB_TOKENS


def test_d_unit_word_singularized():
    from humanize_ingredients import humanize_ingredient
    assert humanize_ingredient("1 tazas de yogurt griego natural sin azúcar") \
        == "1 taza de yogurt griego natural sin azúcar"
    assert humanize_ingredient("1 dientes de ajo") == "1 diente de ajo"
    # [2026-07-12 v2] 'rodajas' se escapó de la lista en vivo ("1 rodajas (50g) de Lechosa").
    assert humanize_ingredient("1 rodajas (50g) de Lechosa fresca") == "1 rodaja (50g) de Lechosa fresca"
    # cantidad ≠ 1 intacta
    assert humanize_ingredient("2 dientes de ajo") == "2 dientes de ajo"


def test_e_bread_slices_capped_at_3():
    from graph_orchestrator import _cap_unrealistic_portions
    days = [{"day": 1, "meals": [{
        "name": "Tostadas con Queso Fresco",
        "ingredients": ["5 rebanadas de pan integral familiar", "25 g de queso blanco fresco"],
        "ingredients_raw": ["5 rebanadas de pan integral familiar", "25 g de queso blanco fresco"],
        "recipe": ["Tuesta el pan."]}]}]
    _cap_unrealistic_portions(days)
    assert days[0]["meals"][0]["ingredients"][0].startswith("3 rebanadas"), (
        "techo realista: 5 rebanadas de pan para 1 persona → 3 (vivo: la aritmética "
        "pan↔lonjas del plato no cuadraba)"
    )
    assert days[0]["meals"][0]["ingredients_raw"][0].startswith("3 rebanadas"), "lockstep raw"


def test_e_v3_bread_gram_form_capped():
    """[v3] El techo por CONTEO no ve líneas gram-lead que humanize convierte a
    'rebanadas' DESPUÉS de los caps (vivo: '4 rebanadas' ≈ 180g lo evadió).
    Rama en gramos: >135g de pan (~3 rebanadas de 45g) → cap; word-boundary
    protege 'panqueques'."""
    from graph_orchestrator import _cap_unrealistic_portions
    days = [{"day": 1, "meals": [{
        "name": "Tostada Integral", "ingredients": ["180 g de pan integral familiar"],
        "ingredients_raw": ["180 g de pan integral familiar"], "recipe": ["Tuesta."]}]}]
    _cap_unrealistic_portions(days)
    assert days[0]["meals"][0]["ingredients"][0].startswith("135 g")
    days2 = [{"day": 1, "meals": [{
        "name": "Panqueques", "ingredients": ["200 g de panqueques de avena"],
        "ingredients_raw": ["200 g de panqueques de avena"], "recipe": ["Cocina."]}]}]
    _cap_unrealistic_portions(days2)
    assert days2[0]["meals"][0]["ingredients"][0].startswith("200 g"), "word-boundary: panqueques exento"


def test_d_v3_tortas_singularized():
    from humanize_ingredients import humanize_ingredient
    assert humanize_ingredient("1 tortas pequeño de casabe").startswith("1 torta ")


def test_e_three_or_fewer_untouched():
    from graph_orchestrator import _cap_unrealistic_portions
    days = [{"day": 1, "meals": [{
        "name": "Tostadas", "ingredients": ["2 rebanadas de pan integral"],
        "ingredients_raw": ["2 rebanadas de pan integral"], "recipe": ["Tuesta."]}]}]
    _cap_unrealistic_portions(days)
    assert days[0]["meals"][0]["ingredients"][0].startswith("2 rebanadas")


def test_marker_anchored():
    go_src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    hu_src = (_BACKEND / "humanize_ingredients.py").read_text(encoding="utf-8")
    assert go_src.count("P1-RECIPE-POLISH-5") >= 5
    assert hu_src.count("P1-RECIPE-POLISH-5") >= 1
