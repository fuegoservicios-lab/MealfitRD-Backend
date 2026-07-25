"""[P1-RECONCILE-LAST-WORD · 2026-07-25] El reconciliador display↔raw corría demasiado pronto.

Tercera vez que aparece el MISMO error de orden en una sesión. Las dos anteriores ya están
cerradas: `P1-CAPS-LAST-WORD` (los techos de porción) y `P1-FINALIZE-TAIL-PARITY` (cocido→seco
tras el refill). Este es el tercer pase que se ejecutaba antes de los pases que lo invalidan.

`_reconcile_display_raw_lines` tenía UN solo callsite, en `assemble_plan_node`, justo antes de
construir la lista de compras. Parecía el sitio correcto — es la frontera. Pero después de esa
frontera siguen corriendo pases que cambian cantidades, y la lista se **re-agrega** detrás:

    reconciliador             ← alinea display↔raw
    DISPLAY-RESTORE-FROM-RAW
    FINAL-BAND-CLOSER         ← cambia cantidades
    GAINMUSCLE-KCAL-FLOOR     ← cambia cantidades
    listas RE-AGREGADAS       ← compra desde el raw que volvió a divergir

Evidencia del plan vivo `1d3c6643` (12 comidas, banda 1.00, entregado al owner):

    receta: 75g de costilla de cerdo      lista:  55g      D1   -27%
    receta: 75 g de costilla de cerdo     lista:  45 g     D2   -40%
    receta: 85 g de queso cottage         lista: 125g      D1   +47%
    receta: 55 g de queso cottage         lista:  85g      D2   +55%
    receta: 15g de queso de hoja cocido   lista:  10g      D1   -33%

Las divergencias se concentran en proteínas y lácteos: exactamente lo que tocan el cerrador de
proteína y el refill de gain-muscle, que corren DESPUÉS del único callsite. La comida del
pescado de ese plan lo confirma en sus propios flags — `_protein_closed`,
`_gainmuscle_kcal_floor` y `_portion_realism_capped`, los tres presentes.

⚠️ Nota sobre el falso positivo que casi me cuesta un "fix": en ese mismo plan la receta decía
`1½ filetes de pescado` y la lista `209.63 g de filete de pescado blanco`. Parece divergencia y
NO lo es (un filete ≈ 140 g). Cualquier auditoría de esta clase tiene que resolver el alimento a
gramos antes de comparar; comparar el texto de la cantidad produce ruido.
"""
import re
from pathlib import Path

import pytest

import graph_orchestrator as go
import shopping_calculator as sc


SRC = (Path(go.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")


FAKE_CATALOG = [
    {"name": "Costilla de cerdo", "density_g_per_unit": None, "density_g_per_cup": None},
    {"name": "Queso cottage", "density_g_per_unit": None, "density_g_per_cup": 226},
    {"name": "Queso de hoja", "density_g_per_unit": None, "density_g_per_cup": None},
]


@pytest.fixture(autouse=True)
def _catalog(monkeypatch):
    monkeypatch.setattr(sc, "get_master_ingredients", lambda *a, **k: FAKE_CATALOG)
    for attr in ("_CATALOG_DENSITY_INDEX_CACHE", "_PHANTOM_CATALOG_INDEX_CACHE"):
        monkeypatch.setattr(go, attr, None, raising=False)
    go._LINE_FOOD_GRAMS_CACHE.clear()
    yield
    go._LINE_FOOD_GRAMS_CACHE.clear()
    go._CATALOG_DENSITY_INDEX_CACHE = None
    go._PHANTOM_CATALOG_INDEX_CACHE = None


def _finalize_body() -> str:
    """Cuerpo de `finalize_plan_data_coherence` (hasta el siguiente `def` de nivel de módulo).

    Acotar es obligatorio: el fichero tiene ~30k líneas y varios pases con nombres parecidos.
    Sin acotar, un `in SRC` mide la existencia del texto en cualquier parte y no prueba nada
    sobre el orden dentro de esta función.
    """
    i = SRC.index("def finalize_plan_data_coherence(")
    j = SRC.index("\ndef finalize_single_meal_recipe_coherence(", i)
    return SRC[i:j]


# ───────────── 1. el cableado: el pase existe y va AL FINAL ─────────────

def test_finalize_llama_al_reconciliador():
    """Sin esto el reconciliador sigue teniendo un solo callsite, aguas arriba de los cerradores."""
    assert "_reconcile_display_raw_lines(days)" in _finalize_body()


def test_va_despues_de_los_caps():
    """Los caps sólo BAJAN el display. Reconciliar antes propagaría a la lista una cantidad que
    el cap está a punto de recortar → se compraría de más, y encima con la firma de estar
    'reconciliado'."""
    body = _finalize_body()
    assert body.index("_cap_unrealistic_portions(days, db=db)") < body.index(
        "_reconcile_display_raw_lines(days)")


def test_es_lo_ultimo_antes_del_return():
    """Cualquier pase nuevo que se cuele detrás vuelve a abrir exactamente este bug."""
    body = _finalize_body()
    cola = body[body.index("_reconcile_display_raw_lines(days)"):]
    # Sólo debe quedar el logging del propio bloque y el return de la función.
    assert re.search(r"return \(total, \", \"\.join\(parts\)\)", cola), "el return debe venir detrás"
    entre = cola[:cola.index("return (total")]
    assert "_cap_unrealistic_portions" not in entre
    assert "_normalize_cooked_grain_lines" not in entre
    assert "_merge_duplicate_food_lines" not in entre


def test_conserva_el_callsite_de_assemble():
    """Este pase NO reemplaza al de assemble: le pone una segunda oportunidad detrás de los
    pases aditivos. Assemble sigue alineando antes de la PRIMERA construcción de la lista."""
    assert SRC.count("_reconcile_display_raw_lines(") >= 3  # def + assemble + finalize


def test_respeta_el_knob():
    body = _finalize_body()
    blk = body[body.index("P1-RECONCILE-LAST-WORD"):]
    assert "if RECONCILE_DISPLAY_RAW:" in blk[:blk.index("_reconcile_display_raw_lines(days)")]


def test_fail_open():
    """Un fallo del reconciliador NUNCA puede tumbar la entrega de un plan ya generado."""
    body = _finalize_body()
    blk = body[body.index("P1-RECONCILE-LAST-WORD"):]
    blk = blk[:blk.index("return (total")]
    assert "except Exception" in blk and "logger.warning" in blk


# ───────────── 2. los casos medidos en el plan vivo 1d3c6643 ─────────────

@pytest.mark.parametrize("display,raw,esperado_g", [
    ("75g de costilla de cerdo (4-5 huesos pequeños)", "55g de costilla de cerdo (4-5 huesos pequeños)", 75),
    ("75 g de costilla de cerdo", "45 g de costilla de cerdo", 75),
    ("85 g de queso cottage", "125g de queso cottage", 85),
    ("55 g de queso cottage", "85g de queso cottage", 55),
    ("15g de queso de hoja cocido", "10g de queso de hoja cocido", 15),
])
def test_alinea_los_casos_reales(display, raw, esperado_g):
    """Autoridad = display: es lo que el usuario lee y lo que respaldan los macros declarados."""
    meal = {"name": "Costilla de Cerdo", "ingredients": [display], "ingredients_raw": [raw]}
    out = go._reconcile_display_raw_lines([{"day": 1, "meals": [meal]}])
    assert len(out) == 1 and out[0]["kind"] == "qty_divergence"
    _, g = go._resolve_line_food_grams(meal["ingredients_raw"][0])
    assert g == pytest.approx(esperado_g, rel=0.02), meal["ingredients_raw"]


def test_no_toca_lo_que_solo_difiere_en_FORMATO():
    """`¼` contra `0.25` es el pulido de display, no una divergencia. Si este pase los 'arregla'
    convierte 60 líneas por plan en ruido y entierra las 8 que importan."""
    meal = {"name": "Puré", "ingredients": ["¼ taza de queso cottage"],
            "ingredients_raw": ["0.25 taza de queso cottage"]}
    assert go._reconcile_display_raw_lines([{"day": 1, "meals": [meal]}]) == []


def test_idempotente():
    meal = {"name": "Cerdo", "ingredients": ["75 g de costilla de cerdo"],
            "ingredients_raw": ["45 g de costilla de cerdo"]}
    days = [{"day": 1, "meals": [meal]}]
    assert go._reconcile_display_raw_lines(days)
    assert go._reconcile_display_raw_lines(days) == []
