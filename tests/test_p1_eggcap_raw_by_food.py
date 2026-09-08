# -*- coding: utf-8 -*-
"""[P1-EGGCAP-RAW-BY-FOOD · 2026-09-08] «3 claras de huevo» escrito sobre la línea «Sal al gusto».

`_cap_daily_whole_eggs` pasa a claras los huevos enteros que exceden el tope diario. Su docstring
prometía «lockstep `ingredients_raw`», pero el lockstep era por ÍNDICE, en dos sitios:

    raw[idx] = new_lines[0]      # la línea capeada
    raw[_j]  = _new_ln           # el merge con una línea de claras existente

En un plan vivo escribió «3 claras de huevo» **encima de «Sal al gusto»**, y —la otra mitad del
mismo fallo— la línea de huevo real nunca se arregló: la lista siguió comprando 3 huevos enteros
mientras la receta pedía claras.

## Por qué su veredicto decía «0» y era falso

`P1-RAW-INDEX-INVENTORY` la clasificó «0 medido 07-sep · sobreescribe entera (sonda ruidosa
aplica)». El razonamiento era correcto —sobreescribir la línea entera SÓLO puede fallar de forma
ruidosa, y para eso la sonda del 07-sep servía— pero **esa sonda nunca se verificó contra un caso
conocido**. Es el tercer veredicto «0» que cae hoy por el instrumento, después del gemelo del
cheapen y de las dos «0 activaciones».

Re-medido con el instrumento validado en `test_p2_coherence_eje_ciego` (conjunto de tokens de la
clave canónica, 7 casos de discriminación): **1 comida de 1.194**. Pequeño, y aun así es la misma
clase que `P1-PHANTOM-DAIRY-RAW-BY-FOOD` cerró esta madrugada — mismo alimento, misma escritura,
mismo día. Tras el arreglo: **0**, y sigue tocando raw en 2 comidas, así que no quedó inerte.

*Un veredicto vale lo que valga el instrumento que lo produjo — aunque el razonamiento sea bueno.*
"""
import inspect

import graph_orchestrator as go


def test_los_DOS_sitios_resuelven_por_alimento():
    src = inspect.getsource(go._cap_daily_whole_eggs)
    assert "_raw_idx_for_display(raw, old, idx, ings)" in src, "el sitio del capeo sigue por índice"
    assert "_raw_idx_for_display(raw, _ln, _j, ings)" in src, "el sitio del merge sigue por índice"
    assert "raw[idx] = new_lines[0]" not in src
    assert "raw[_j] = _new_ln" not in src


def test_ambos_resuelven_ANTES_de_mutar_su_linea_de_display():
    """Resolver después busca en raw el alimento NUEVO, que allí todavía no existe."""
    src = inspect.getsource(go._cap_daily_whole_eggs)
    assert src.index("_ri_eg = _raw_idx_for_display") < src.index("ings[idx] = new_lines[0]")
    assert src.index("_ri_mg = _raw_idx_for_display") < src.index("ings[_j] = _new_ln")


def test_el_docstring_ya_no_promete_un_lockstep_por_indice():
    """El docstring decía «lockstep `ingredients_raw`» y era exactamente lo que fallaba.

    Una promesa falsa en el docstring es peor que ninguna: invita a confiar citándola.
    """
    doc = go._cap_daily_whole_eggs.__doc__ or ""
    assert "P1-EGGCAP-RAW-BY-FOOD" in doc
    assert "lockstep `ingredients_raw`" not in doc


def test_el_veredicto_del_inventario_esta_al_dia():
    from tests.test_p1_raw_index_inventory import _INVENTARIO

    n, veredicto = _INVENTARIO["graph_orchestrator.py"]["_cap_daily_whole_eggs"]
    assert n == 2
    assert veredicto.startswith("resuelto"), f"veredicto sin actualizar: {veredicto!r}"
