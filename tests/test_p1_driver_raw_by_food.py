# -*- coding: utf-8 -*-
"""[P1-DRIVER-RAW-BY-FOOD · 2026-09-08] El gemelo del cheapen, y por qué su «0 medido» era falso.

`_apply_budget_driver_aware_pass` es el hermano de `_apply_budget_cheapen_pass`: su propio docstring
dice «guards idénticos». También escribía `ingredients_raw[idx]` por índice, bajo el mismo
`_re.sub(rf"\\b(?:{rx})\\b", ...)`.

## Por qué el inventario lo daba por limpio

`P1-RAW-INDEX-INVENTORY` le puso el veredicto «0 medido 07-sep». Ese 0 salió de una sonda que
preguntaba **«¿escribe sobre la línea de OTRO alimento?»** — y para una escritura bajo `sub` la
respuesta es siempre no: si el índice cae en otra línea, el patrón no casa y no pasa nada.

`P1-CHEAPEN-RAW-BY-FOOD` (unas horas antes, 07-sep) demostró que esa orientación es ciega justo para
esta forma de escritura. El veredicto de este gemelo venía de la MISMA sonda ciega, así que había
que rehacerlo, no heredarlo. *Un veredicto vale lo que valga el instrumento que lo produjo.*

## Medido con la sonda al derecho

Sobre las 1.172 comidas vivas, con la `weekly_list` costeada de cada plan: el pase abarata el
display en 165 comidas (14,1 %) y en **38 (3,2 %) la lista se quedaba comprando el driver caro** —
granada en vez de guineo, espárragos en vez de vainitas. Con el arreglo: **0**.

## Las otras dos bajo patrón, medidas a la vez

`_cap_cheese_dumps_final` y `_baking_powder_cap_pass` comparten la forma (escriben bajo `sub` /
`rescale`) pero **no se activan ni una vez** en las 1.172 comidas. Su veredicto correcto no es
«escritura segura» sino «no dispara»: son cosas distintas y el inventario ahora lo dice, porque un
detector que no dispara nunca sale perfecto.
"""
import inspect

import graph_orchestrator as go


def test_resuelve_por_alimento_y_no_por_indice():
    src = inspect.getsource(go._apply_budget_driver_aware_pass)
    assert "_raw_idx_for_display(raw, ing, idx, ings)" in src
    assert "raw[idx] = _dedup_unit_noun_collision" not in src


def test_resuelve_ANTES_de_mutar_el_display():
    """Resuelto después, `ings[idx]` ya es la línea barata y se buscaría en raw un alimento que allí
    todavía no existe. El orden no es estilo: es la condición para que el arreglo funcione."""
    src = inspect.getsource(go._apply_budget_driver_aware_pass)
    assert src.index("_ri_dr = _raw_idx_for_display") < src.index("ings[idx] = _dedup_unit_noun_collision")


def test_resuelve_desde_la_linea_VIEJA():
    """`ing` es la línea antes de la sustitución — la que aún nombra el driver caro."""
    src = inspect.getsource(go._apply_budget_driver_aware_pass)
    assert "_raw_idx_for_display(raw, ing, idx, ings)" in src
    assert "_raw_idx_for_display(raw, new_line" not in src


def test_conserva_la_sustitucion_por_patron():
    """El `sub` sigue siendo la segunda guarda: acertar la línea no autoriza a reescribirla entera."""
    src = inspect.getsource(go._apply_budget_driver_aware_pass)
    assert 'rf"\\b(?:{rx})\\b", candidate, raw[_ri_dr]' in src


def test_los_dos_gemelos_usan_EL_MISMO_resolvedor():
    """Los dos pases de presupuesto fallaban igual y se arreglaron con distintas horas de por medio.

    Si mañana alguien cambia el resolvedor de uno y no del otro, vuelve la asimetría que hizo que
    este gemelo se quedara fuera de `P1-CHEAPEN-RAW-BY-FOOD`.
    """
    for fn in (go._apply_budget_cheapen_pass, go._apply_budget_driver_aware_pass):
        assert "_raw_idx_for_display(raw, ing, idx, ings)" in inspect.getsource(fn), fn.__name__


def test_el_pase_sigue_inventariado_y_con_el_veredicto_al_dia():
    from tests.test_p1_raw_index_inventory import _INVENTARIO

    inv = _INVENTARIO["graph_orchestrator.py"]
    n, veredicto = inv["_apply_budget_driver_aware_pass"]
    assert n == 1
    assert veredicto.startswith("resuelto"), f"veredicto sin actualizar: {veredicto!r}"
    # Y las dos que comparten forma deben decir «no dispara», no «0 medido» a secas.
    for fn in ("_cap_cheese_dumps_final", "_baking_powder_cap_pass"):
        assert "no dispara" in inv[fn][1], f"{fn}: veredicto heredado de la sonda ciega"
