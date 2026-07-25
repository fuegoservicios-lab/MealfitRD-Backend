"""[P1-DISPLAY-RAW-QTY-RECONCILE · 2026-07-24] Los macros y la lista salían de listas distintas.

    meal["ingredients"]      → lo que el usuario lee Y de donde salen los macros
                               (`_truth_up_meal_macros_from_strings` lee ESTA lista)
    meal["ingredients_raw"]  → lo que COMPRA la lista de compras
                               (`meal.get("ingredients_raw") or meal.get("ingredients")`)

Medido sobre los 6 planes más recientes, pareando por alimento resuelto (no por índice — las
listas están reordenadas, y comparar por posición enfrenta "½ tomate" contra "0.5 ají cubanela"
y no mide nada): de 214 alimentos presentes en AMBAS listas, **52 (24%) discrepan más de un 5%**.

    Bollos de Harina    ING "15 g de queso blanco"        RAW "80.19 g de queso blanco"   0.19×
    Bowl Tibio          ING "1½ tazas de yogurt"          RAW "75g de yogurt"             4.90×
    Filete de pescado   ING "½ taza de espinacas frescas" RAW "2.49 tazas de espinacas"   0.20×

Y 27 alimentos aparecen SOLO en el display: no se compran nunca — la misma clase de defecto que
la guanábana (P1-PHANTOM-INGREDIENT), entrando por otra puerta.

La divergencia va en ambas direcciones, así que no es "una lista está más fresca": son pases
distintos actualizando listas distintas. Este pase cierra el síntoma en la frontera (justo antes
de construir la lista) y deja telemetría con la magnitud para poder perseguir al culpable con
datos en vez de a ojo.
"""
import pytest

import graph_orchestrator as go
import shopping_calculator as sc


FAKE_CATALOG = [
    {"name": "Queso blanco", "density_g_per_unit": None, "density_g_per_cup": None},
    {"name": "Espinacas", "density_g_per_unit": None, "density_g_per_cup": 30},
    {"name": "Yogurt", "density_g_per_unit": None, "density_g_per_cup": 245},
    {"name": "Aguacate", "density_g_per_unit": 250, "density_g_per_cup": 150},
    {"name": "Mero", "density_g_per_unit": None, "density_g_per_cup": None},
]


@pytest.fixture(autouse=True)
def _catalog(monkeypatch):
    monkeypatch.setattr(sc, "get_master_ingredients", lambda *a, **k: FAKE_CATALOG)
    yield


# ───────────── 1. los casos medidos en planes reales ─────────────

def test_queso_blanco_0_19x():
    meal = {"name": "Bollos de Harina",
            "ingredients": ["15 g de queso blanco"],
            "ingredients_raw": ["80.19 g de queso blanco"]}
    out = go._reconcile_display_raw_lines([{"day": 1, "meals": [meal]}])
    assert len(out) == 1 and out[0]["kind"] == "qty_divergence"
    assert meal["ingredients_raw"] == ["15 g de queso blanco"], (
        "la lista compra lo que el usuario lee, no 5× más"
    )


def test_yogurt_4_9x_en_la_direccion_contraria():
    """La divergencia va en ambos sentidos: aquí el display pide MÁS que raw."""
    meal = {"name": "Bowl Tibio",
            "ingredients": ["1½ tazas de yogurt"],
            "ingredients_raw": ["75g de yogurt"]}
    go._reconcile_display_raw_lines([{"day": 1, "meals": [meal]}])
    assert meal["ingredients_raw"] == ["1½ tazas de yogurt"]


def test_alimento_solo_en_display_se_apendea_a_raw():
    """27 alimentos en 6 planes estaban así: visibles y jamás comprados."""
    meal = {"name": "Bowl", "ingredients": ["270 g de mero", "½ aguacate"],
            "ingredients_raw": ["270 g de mero"]}
    out = go._reconcile_display_raw_lines([{"day": 1, "meals": [meal]}])
    assert [o["kind"] for o in out] == ["missing_in_raw"]
    assert any("aguacate" in str(i).lower() for i in meal["ingredients_raw"])


def test_idempotente():
    meal = {"name": "Bollos", "ingredients": ["15 g de queso blanco"],
            "ingredients_raw": ["80.19 g de queso blanco"]}
    days = [{"day": 1, "meals": [meal]}]
    assert go._reconcile_display_raw_lines(days)
    assert go._reconcile_display_raw_lines(days) == [], "tras alinear, el ratio es 1.0"


# ───────────── 2. lo que NO debe tocar ─────────────

def test_respeta_la_precision_de_raw_bajo_la_tolerancia():
    """`42.17 g` en raw contra `42 g` en display es el DISEÑO (raw = pre-humanización),
    no un bug. Sólo se toca lo que supera la tolerancia."""
    meal = {"name": "Plato", "ingredients": ["42 g de queso blanco"],
            "ingredients_raw": ["42.17 g de queso blanco"]}
    assert go._reconcile_display_raw_lines([{"day": 1, "meals": [meal]}]) == []
    assert meal["ingredients_raw"] == ["42.17 g de queso blanco"]


def test_no_toca_lo_que_esta_solo_en_raw():
    """Ese caso es de `_restore_display_from_raw_orphans` (P1-DISPLAY-RESTORE-FROM-RAW), que
    lo repone al display prettificado y con su propio cap. Duplicar esa responsabilidad aquí
    haría que dos pases se pisaran."""
    meal = {"name": "Plato", "ingredients": ["270 g de mero"],
            "ingredients_raw": ["270 g de mero", "15 g de queso blanco"]}
    assert go._reconcile_display_raw_lines([{"day": 1, "meals": [meal]}]) == []
    assert len(meal["ingredients_raw"]) == 2


def test_sin_gramos_confiables_no_se_toca():
    """Si alguna línea del alimento no convierte a gramos, la comparación no es fiable:
    antes que alinear a ciegas, se deja como está."""
    meal = {"name": "Plato", "ingredients": ["2 cdas de queso blanco"],
            "ingredients_raw": ["80.19 g de queso blanco"]}
    assert go._reconcile_display_raw_lines([{"day": 1, "meals": [meal]}]) == []


def test_meal_sin_raw_no_falla():
    meal = {"name": "Plato", "ingredients": ["15 g de queso blanco"]}
    assert go._reconcile_display_raw_lines([{"day": 1, "meals": [meal]}]) == []


# ───────────── 3. cableado y contrato ─────────────

def test_los_macros_salen_del_display_que_es_la_autoridad():
    """La razón por la que el display manda: `_truth_up_meal_macros_from_strings` calcula los
    macros desde `meal["ingredients"]`. Si esto cambia, la autoridad de este pase cambia."""
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("def _truth_up_meal_macros_from_strings")
    assert 'ings = meal.get("ingredients")' in src[i:i + 1500], (
        "los macros salen del display; si pasaran a salir de raw, invertir la autoridad"
    )


def test_corre_al_final_y_antes_de_la_lista():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    i_dm = src.index('_merge_duplicate_food_lines(result.get("days")')
    i_rc = src.index('_reconcile_display_raw_lines(result.get("days")')
    i_list = src.index("# Calcular shopping lists")
    assert i_dm < i_rc < i_list, (
        "último pase antes de la lista: los anteriores dejan el display en su forma definitiva"
    )


def test_knobs():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'RECONCILE_DISPLAY_RAW = _env_bool("MEALFIT_RECONCILE_DISPLAY_RAW", True)' in src
    assert '_env_float("MEALFIT_RECONCILE_DISPLAY_RAW_TOL", 0.10' in src
    assert 0.02 <= go.RECONCILE_DISPLAY_RAW_TOL <= 0.50
