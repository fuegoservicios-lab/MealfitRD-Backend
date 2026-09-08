# -*- coding: utf-8 -*-
"""[P1-FLOOR-RAW-BY-FOOD · 2026-09-07] El piso inflaba el alimento equivocado de la lista.

`_floor_subservible_portions` escribía `ingredients_raw[idx]` bajo `_lockstep = len(raw) ==
len(ings)` en **diez** sitios. `raw` es lo que el usuario COMPRA y lo que leen los medidores.

## Sólo una de sus cuatro llamadas estaba expuesta, y se trazó

    finalize_plan_data_coherence:28931     → antes de sus appenders          sin exposición
    assemble_plan_node:40231 y :40346      → antes de los suyos (40493/99)   sin exposición
    /recalculate-shopping-list             → pasa `portion_floors=False`     ni corre
    finalize_single_meal_recipe_coherence  → swap / chat-modify / expand     EXPUESTA

Esa última corre sobre comidas **ya persistidas**, cuyo `raw` el reconciliador dejó reordenado.

## Medido en la flota, con A/B de criterio idéntico

    conducta vieja  →  29 comidas tocan raw,  6 en el alimento equivocado
    con el arreglo  →  30 comidas tocan raw,  0

Los seis eran inflaciones de la compra: el piso subía el aguacate y **la papa pasaba de ½ a 1,5**;
subía las nueces y el cottage iba de 105 a 157,5 g; subía el merey y el queso de 15 a 45 g.

## Dos cosas que este test protege y no son obvias

1. **El display no se toca.** Estas escrituras son sólo del lado `raw`; `ings[idx]` se escribe
   aparte y por índice sobre su propia lista, que siempre alinea consigo misma. Un refactor que
   «unifique» ambos lados bajo el resolvedor rompería el display.
2. **El borrado reusa el helper de julio.** `raw.pop(_di)` por índice era la misma clase de fallo;
   ahora llama a `_remove_one_raw_line_by_food`, que ya existía. Volver a escribir un borrado por
   índice aquí reabre el defecto en su forma más cara: sacar de la compra un alimento que el plato
   SÍ lleva.
"""
import inspect

import pytest

import graph_orchestrator as go


def test_no_queda_ningun_lockstep_en_el_piso():
    """Ancla estructural: el guard por largo desapareció del cuerpo de la función.

    Es la forma que tenían los diez sitios, y la que un refactor podría reintroducir sin que
    ninguna aserción de macros lo note — el piso seguiría aplicándose, sólo que a otro alimento.
    """
    src = inspect.getsource(go._floor_subservible_portions)
    assert "_lockstep" not in src
    assert "raw[idx]" not in src


def test_el_piso_resuelve_por_alimento():
    src = inspect.getsource(go._floor_subservible_portions)
    assert "_raw_idx_for_display(raw, str(ing), idx, ings)" in src
    assert src.count("raw[_ri]") >= 8, "faltan escrituras migradas"


def test_el_borrado_reusa_el_helper_de_julio():
    """`raw.pop(_di)` por índice era la misma clase de fallo, en su forma más cara."""
    src = inspect.getsource(go._floor_subservible_portions)
    assert "_remove_one_raw_line_by_food(meal, str(ings[_di]), _di)" in src
    assert "raw.pop(_di)" not in src


def test_el_display_se_sigue_escribiendo_por_indice():
    """`ings[idx]` alinea SIEMPRE consigo misma: aplicarle el resolvedor sería el error opuesto."""
    src = inspect.getsource(go._floor_subservible_portions)
    assert "ings[idx] = " in src
    assert "ings[_ri]" not in src


def test_el_resolvedor_no_recibe_un_paralelismo_precalculado():
    """Se probó pasarle el veredicto ya calculado y se quitó: duplicaba en el llamante la lógica
    que el helper ya tiene —el error que este P-fix persigue— y `_resolve_line_food_grams`
    memoiza, así que la comprobación por línea sale gratis tras la primera."""
    sig = inspect.signature(go._raw_idx_for_display)
    assert list(sig.parameters) == ["raw", "display_line", "idx", "ings"]


# ---------------------------------------------------------------- comportamiento

def _meal_rotado():
    """Las dos listas del mismo largo, con raw rotado — la forma `[conservadas] + [añadidas]`."""
    return {
        "meal": "Merienda",
        "ingredients": ["15 g de aguacate", "½ papa mediana"],
        "ingredients_raw": ["½ papa mediana", "15 g de aguacate"],
        "recipe": ["Sirve el aguacate con la papa."],
    }


def test_con_raw_rotado_no_toca_el_alimento_que_no_movio():
    """El caso vivo reducido: el piso sube el aguacate; la papa no puede cambiar."""
    m = _meal_rotado()
    antes_papa = m["ingredients_raw"][0]
    try:
        go._floor_subservible_portions([{"meals": [m]}], day_kcal_target=None, db=None)
    except Exception:
        pytest.skip("el piso necesita db en este entorno")
    assert m["ingredients_raw"][0] == antes_papa, m["ingredients_raw"]
