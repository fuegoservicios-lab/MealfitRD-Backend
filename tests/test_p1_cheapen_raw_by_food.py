# -*- coding: utf-8 -*-
"""[P1-CHEAPEN-RAW-BY-FOOD · 2026-09-07] La receta abarataba y la lista seguía comprando caro.

`_apply_budget_cheapen_pass` sustituye una proteína cara por una barata en el plato de un usuario
con presupuesto bajo. Escribía `ingredients_raw[idx]` por índice:

    ings[idx] = sub(rx → candidato, ings[idx])
    if isinstance(raw, list) and idx < len(raw) and isinstance(raw[idx], str):
        raw[idx] = sub(rx → candidato, raw[idx])

## Por qué este NO daba «alimento equivocado» y aun así era el peor

La escritura pasa por `_re.sub(rf"\\b(?:{rx})\\b", ...)`. Si `raw[idx]` es otro alimento, el patrón
no casa y la línea **no cambia**: seguro por construcción, y por eso las sondas que buscaban
«escribe en la línea de otro» daban 0.

El daño está en el modo **silencioso**: la línea REAL de la proteína cara nunca se abarata. El
display dice yogurt natural, la lista compra yogurt griego — **RD$320/lb contra RD$164** — y el
usuario paga la diferencia sin que nada lo señale. Es exactamente la incoherencia receta↔lista
contra la que existe el coherence guard, entrando por una puerta que el guard no mira.

*Una sonda que busca escrituras equivocadas no ve las escrituras que faltan.*

## Medido

Sobre 1.172 comidas vivas, forzando presupuesto bajo: el pase abarata el display en 402 y en
**171 la lista se quedaba comprando el premium**. Con el arreglo: **2**. (Ese 171 es cota superior:
en producción el pase sólo corre para presupuesto bajo.)

Exposición trazada: `assemble_plan_node` lo llama dos veces —en 40045, ANTES de los appenders, y en
**41038, DESPUÉS**— más la vía del chat sobre planes ya persistidos.

## El orden es load-bearing

El índice se resuelve **antes** de `ings[idx] = new_line` y **desde la línea vieja**: lo que hay que
localizar en raw es la proteína CARA, que es la que todavía está ahí. Resolverlo después, o desde
la línea nueva, busca un alimento que en raw aún no existe.
"""
import inspect

import graph_orchestrator as go


def test_resuelve_por_alimento_y_no_por_indice():
    src = inspect.getsource(go._apply_budget_cheapen_pass)
    assert "_raw_idx_for_display(raw, ing, idx, ings)" in src
    assert "raw[idx] = _re.sub" not in src


def test_resuelve_ANTES_de_mutar_el_display():
    """Si se resuelve después, `ings[idx]` ya es la línea barata y se buscaría en raw un alimento
    que allí todavía no existe. El orden no es estilo: es la condición para que funcione."""
    src = inspect.getsource(go._apply_budget_cheapen_pass)
    i_res = src.index("_ri_ch = _raw_idx_for_display")
    i_mut = src.index("ings[idx] = new_line")
    assert i_res < i_mut, "el índice se resuelve DESPUÉS de mutar el display"


def test_resuelve_desde_la_linea_VIEJA():
    """`ing` es la línea antes de la sustitución — la que aún nombra la proteína cara."""
    src = inspect.getsource(go._apply_budget_cheapen_pass)
    assert "_raw_idx_for_display(raw, ing, idx, ings)" in src
    assert "_raw_idx_for_display(raw, new_line" not in src


def test_conserva_la_sustitucion_por_patron():
    """El `sub` sigue siendo la segunda guarda: acertar la línea no autoriza a reescribirla entera.

    Si la línea de raw no nombra la proteína cara, el patrón no casa y no se toca — que es la
    conducta correcta y la razón de que este fallo nunca escribiera un alimento equivocado.
    """
    src = inspect.getsource(go._apply_budget_cheapen_pass)
    assert 'raw[_ri_ch] = _re.sub(rf"\\b(?:{rx})\\b"' in src


def test_el_pase_sigue_inventariado():
    """El ratchet de `P1-RAW-INDEX-INVENTORY` debe seguir contando esta escritura."""
    from tests.test_p1_raw_index_inventory import _INVENTARIO

    n, veredicto = _INVENTARIO["graph_orchestrator.py"]["_apply_budget_cheapen_pass"]
    assert n == 1
    assert veredicto.startswith("resuelto"), f"veredicto sin actualizar: {veredicto!r}"
