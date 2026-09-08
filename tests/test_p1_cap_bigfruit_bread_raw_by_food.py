# -*- coding: utf-8 -*-
"""[P1-CAP-BIGFRUIT-BREAD-RAW-BY-FOOD · 2026-09-07] Dos recortes que escribían en la línea de otro.

`_cap_unrealistic_portions` reescribía `ingredients_raw[idx]` bajo `_lockstep = len(raw) ==
len(ings)` en dos ramas: la ración de fruta grande y el techo de rebanadas de pan. **`raw` es lo
que COMPRA la lista.**

## Por qué «mismo largo» no bastaba aquí, trazado

Dentro de `finalize_plan_data_coherence` el orden es:

    28680  _cap_unrealistic_portions
    28931  _floor_subservible_portions
    28946  _reconcile_display_missing_in_raw   ← APPENDEA líneas a raw (lo dice su docstring)
    28952  _reconcile_raw_missing_in_display   ← APPENDEA líneas a display
    29324  _cap_unrealistic_portions OTRA VEZ  ← corre DESPUÉS de los dos appenders
    29360  _reconcile_display_raw_lines

Tras los appenders el largo vuelve a coincidir y el orden no: la forma `[conservadas] + [añadidas]`
que midió `P2-RAW-PAIR-BY-FOOD` («el 93,5 % tiene largos iguales; sólo el 48,1 % de ESAS son
paralelas»). La segunda invocación evalúa `_lockstep` → True y recorta la línea equivocada.

Es el caso vivo que `P1-CAP-RAW-BY-FOOD` ya citaba en ESTA MISMA función (943c604b): el display
bajó a 120 g de queso cottage, raw se quedó en 496,8 g y el recorte cayó sobre «Sal al gusto».
Aquel P-fix migró una rama y dejó estas dos con el guard viejo.

## Alcance

Sólo estas dos, que son las **confirmadas por el orden de arriba**. Las 6 de
`_floor_subservible_portions` quedan diagnosticadas y sin tocar: su llamada va ANTES de los
appenders y no están demostradas. *El alcance de un arreglo es lo que se ha demostrado, no lo que
se ha sospechado.*
"""
import inspect

import pytest

import graph_orchestrator as go


# ---------------------------------------------------------------- el resolvedor

def test_con_listas_paralelas_devuelve_el_mismo_indice():
    """El caso barato y mayoritario: si alinean de verdad, el índice sirve."""
    ings = ["1 lechosa", "2 cdas de aceite de oliva"]
    raw = ["1 lechosa", "30 g de aceite de oliva"]
    assert go._raw_idx_for_display(raw, ings[0], 0, ings) == 0


def test_con_raw_ROTADO_no_devuelve_el_indice_ciego():
    """La forma `[conservadas] + [añadidas]`: el índice apunta a otro alimento."""
    ings = ["1 lechosa", "Sal al gusto"]
    raw = ["Sal al gusto", "1 lechosa"]
    assert go._raw_idx_for_display(raw, "1 lechosa", 0, ings) != 0


def test_ante_ambiguedad_devuelve_None_y_el_llamante_no_toca_raw():
    """Dos líneas de raw con el mismo alimento: no se adivina cuál recortar."""
    ings = ["1 lechosa"]
    raw = ["100 g de lechosa", "200 g de lechosa"]
    assert go._raw_idx_for_display(raw, "1 lechosa", 0, ings) is None


@pytest.mark.parametrize("basura", [None, "", 123, []])
def test_fail_safe_devuelve_None(basura):
    """Si no puede resolver, devuelve None: no recortar es la mitad segura."""
    assert go._raw_idx_for_display(basura, "1 lechosa", 0, ["1 lechosa"]) is None


# ---------------------------------------------------------------- los dos sitios

def test_las_dos_ramas_ya_no_escriben_por_indice_ciego():
    """Ancla estructural sobre el cuerpo de la función.

    Un refactor que devuelva `raw[idx] = …` bajo `_lockstep` en estas dos ramas reabre el defecto
    sin que ningún test de comportamiento lo note: el recorte seguiría ocurriendo, sólo que en la
    línea de otro alimento, y eso no rompe ninguna aserción de macros.
    """
    src = inspect.getsource(go._cap_unrealistic_portions)
    assert "_raw_idx_for_display(raw, s, idx, ings)" in src
    assert "raw[_ri_bf]" in src and "raw[_ri_br]" in src
    # Las dos ramas migradas ya no pueden escribir por índice bajo el guard débil.
    assert "raw[idx] = _bigfruit_bare_count_serving" not in src
    assert "raw[idx] = _resc(str(raw[idx]), 3.0 / _nbr)" not in src


def test_el_resolvedor_usa_el_contrato_de_la_casa():
    """No se escribe un comprobador nuevo: es la lección de `P1-DIET-CANON-SSOT`."""
    src = inspect.getsource(go._raw_idx_for_display)
    assert "_raw_display_parallel_by_food" in src
    assert "_resolve_line_food_grams" in src


def test_el_hermano_migrado_en_julio_sigue_intacto():
    """`P1-CAP-RAW-BY-FOOD` migró una tercera rama de esta función en julio.

    Si desaparece, es que alguien deshizo aquel arreglo mientras tocaba éste.
    """
    src = inspect.getsource(go._cap_unrealistic_portions)
    assert "_rescale_raw_by_food(raw, [s], [factor])" in src
