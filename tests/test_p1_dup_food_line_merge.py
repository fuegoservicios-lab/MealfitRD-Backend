"""[P1-DUP-FOOD-LINE-MERGE · 2026-07-24] El mismo alimento en dos líneas de la misma comida.

Plan vivo `732588f8`, D1 "Bowl Poke Tropical de Mero":

    ingredients[2]  "¼ taza de aguacate fresco"     0.25 × 150 g/taza  =  37.5 g
    ingredients[3]  "½ aguacate"                    0.5  × 250 g/unid  = 125.0 g

El usuario ve el aguacate dos veces, con dos unidades distintas. En `ingredients_raw` (lo que
lee la lista de compras) pasa lo mismo con casabe y huevo en otras comidas del plan.

**Lo que este pase NO arregla** — importante, porque yo mismo lo asumí de más antes de mirar:
el agregador ya fusiona por nombre canónico y normaliza a gramos ANTES de asignar empaques
(shopping_calculator ~L7423), así que las dos líneas NO producían doble redondeo de empaque.
Se sumaban correctamente. El defecto es de presentación y de portación: dos líneas que
describen el mismo alimento suman 162 g de aguacate en un bowl sin que ningún cap lo vea,
porque cada línea por separado está bajo el techo.
"""
import pytest

import graph_orchestrator as go
import shopping_calculator as sc


FAKE_CATALOG = [
    {"name": "Aguacate", "density_g_per_unit": 250, "density_g_per_cup": 150},
    {"name": "Casabe", "density_g_per_unit": 30, "density_g_per_cup": None},
    {"name": "Huevo", "density_g_per_unit": 50, "density_g_per_cup": 243},
    {"name": "Mero", "density_g_per_unit": None, "density_g_per_cup": None},
    {"name": "Cebolla", "density_g_per_unit": 150, "density_g_per_cup": 160},
]


@pytest.fixture(autouse=True)
def _catalog(monkeypatch):
    monkeypatch.setattr(sc, "get_master_ingredients", lambda *a, **k: FAKE_CATALOG)
    # [P1-MISALIGN-DEEP-TRACE · 2026-07-24] Los índices y el memo línea→alimento son globales de
    # módulo (se construyen una vez por proceso, que es lo correcto en producción: el catálogo no
    # cambia a mitad de generación). En tests hay que arrancar en frío o un caso hereda lo que
    # resolvió el anterior — que es justo lo que rompió `test_sin_catalogo_no_hace_nada`.
    for attr in ("_CATALOG_DENSITY_INDEX_CACHE", "_PHANTOM_CATALOG_INDEX_CACHE"):
        monkeypatch.setattr(go, attr, None, raising=False)
    go._LINE_FOOD_GRAMS_CACHE.clear()
    yield
    go._LINE_FOOD_GRAMS_CACHE.clear()
    go._CATALOG_DENSITY_INDEX_CACHE = None
    go._PHANTOM_CATALOG_INDEX_CACHE = None


def _bowl_poke():
    return {
        "name": "Bowl Poke Tropical de Mero",
        "ingredients": ["270 g de mero", "¼ taza de aguacate fresco", "½ aguacate",
                        "½ cebolla", "Sal al gusto"],
        "ingredients_raw": ["270 g de mero", "0.25 taza de aguacate fresco", "0.5 aguacate",
                            "Sal al gusto"],
    }


# ───────────── 1. el caso vivo ─────────────

def test_aguacate_duplicado_se_funde_en_una_linea():
    meal = _bowl_poke()
    out = go._merge_duplicate_food_lines([{"day": 1, "meals": [meal]}])
    assert out, "el aguacate aparecía dos veces en la misma comida"
    aguacates = [i for i in meal["ingredients"] if "aguacate" in str(i).lower()]
    assert len(aguacates) == 1, f"una sola línea de aguacate, no {aguacates}"


def test_conserva_el_total_en_gramos():
    """Fusionar no puede cambiar cuánta comida hay: 37.5 g + 125 g = 162.5 g ≈ ⅔ de aguacate."""
    meal = _bowl_poke()
    go._merge_duplicate_food_lines([{"day": 1, "meals": [meal]}])
    linea = next(i for i in meal["ingredients"] if "aguacate" in str(i).lower())
    qty, unit, canon = sc._parse_quantity(linea, apply_yield_multiplier=False)
    gramos = go._dup_merge_line_to_grams(qty, unit, canon)
    assert abs(gramos - 162.5) <= 12, f"{linea!r} → {gramos} g (esperado ≈162.5)"


def test_escribe_en_la_unidad_dominante_no_en_gramos_crudos():
    """La línea que más aporta manda: '½ aguacate' (125 g) sobre '¼ taza' (37.5 g).
    Escribir '163 g de Aguacate' en el display sería un retroceso de legibilidad."""
    meal = _bowl_poke()
    go._merge_duplicate_food_lines([{"day": 1, "meals": [meal]}])
    linea = next(i for i in meal["ingredients"] if "aguacate" in str(i).lower())
    assert "taza" not in linea.lower() and " g " not in linea, linea
    assert linea == "⅔ Aguacate", linea


def test_prefiere_gramos_antes_que_inventar_comida():
    """Los cuartos solos redondeaban 0.65 → ¾ = 187.5 g: +15% de aguacate inventado por un
    detalle de formato. Con tercios el caso vivo cae en ⅔ (+2.6%); y si ninguna fracción queda
    dentro del 5%, se escribe en gramos — menos bonito, nunca falso."""
    # 0.6 unidades de casabe (sin fracción cercana: ⅔=0.667 desvía 11%) → gramos.
    linea = go._dup_merge_format(18.0, "unidad", "Casabe")
    assert linea == "18 g de Casabe", linea
    # Caso vivo: ⅔ desvía 2.6% → se permite la fracción.
    assert go._dup_merge_format(162.5, "unidad", "Aguacate") == "⅔ Aguacate"


def test_las_dos_listas_se_procesan_por_separado():
    """P1-PHANTOM-RAW-PARITY: no están alineadas por índice y la lista de compras lee raw."""
    meal = _bowl_poke()
    assert len(meal["ingredients"]) != len(meal["ingredients_raw"])
    go._merge_duplicate_food_lines([{"day": 1, "meals": [meal]}])
    assert len([i for i in meal["ingredients_raw"] if "aguacate" in str(i).lower()]) == 1


def test_idempotente():
    meal = _bowl_poke()
    days = [{"day": 1, "meals": [meal]}]
    assert go._merge_duplicate_food_lines(days)
    assert go._merge_duplicate_food_lines(days) == [], "ya no hay duplicados que fundir"


# ───────────── 2. lo que NO debe tocar ─────────────

def test_alimentos_distintos_no_se_tocan():
    meal = {"name": "Plato", "ingredients": ["270 g de mero", "½ cebolla", "½ aguacate"]}
    assert go._merge_duplicate_food_lines([{"day": 1, "meals": [meal]}]) == []
    assert len(meal["ingredients"]) == 3


def test_unidad_no_convertible_aborta_el_grupo():
    """'2 cdas de aguacate' no tiene densidad por cucharada de confianza: antes que inventar
    una conversión, se deja el duplicado. El fallo caro sería alterar la cantidad."""
    meal = {"name": "Plato", "ingredients": ["2 cdas de aguacate", "½ aguacate"]}
    assert go._merge_duplicate_food_lines([{"day": 1, "meals": [meal]}]) == []
    assert len(meal["ingredients"]) == 2


def test_al_gusto_no_estorba():
    meal = {"name": "Plato", "ingredients": ["Sal al gusto", "Pimienta negra al gusto",
                                             "½ aguacate", "¼ taza de aguacate fresco"]}
    go._merge_duplicate_food_lines([{"day": 1, "meals": [meal]}])
    assert "Sal al gusto" in meal["ingredients"] and "Pimienta negra al gusto" in meal["ingredients"]


def test_sin_densidades_no_hace_nada(monkeypatch):
    """Fail-open: sin densidades no se fusiona (nunca se adivina una cantidad).

    Se vacía el ÍNDICE, no el catálogo: `_catalog_density_index` también cae a `UNIT_WEIGHTS`/
    `VOLUMETRIC_DENSITIES` de constants, así que "sin catálogo" no implica "sin densidades"
    — con `get_master_ingredients` vacío el aguacate sigue resolviendo a 250 g/unidad. Afirmar
    sobre el catálogo probaba una premisa falsa (y sólo pasaba por el orden de los tests)."""
    monkeypatch.setattr(go, "_catalog_density_index", lambda: {}, raising=False)
    meal = _bowl_poke()
    assert go._merge_duplicate_food_lines([{"day": 1, "meals": [meal]}]) == []
    assert len(meal["ingredients"]) == 5, "las líneas quedan como estaban"


# ───────────── 3. cableado ─────────────

def test_corre_despues_del_fantasma_y_antes_de_la_lista():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    i_ph = src.index('_repair_declared_but_unlisted_ingredients(result.get("days")')
    i_dm = src.index('_merge_duplicate_food_lines(result.get("days")')
    i_list = src.index("# Calcular shopping lists")
    assert i_ph < i_dm < i_list, (
        "después del fantasma (si la línea reinsertada coincide con una existente, aquí se funden) "
        "y antes de la lista"
    )


def test_knob_de_rollback():
    from pathlib import Path
    src = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'MERGE_DUPLICATE_FOOD_LINES = _env_bool("MEALFIT_MERGE_DUPLICATE_FOOD_LINES", True)' in src
