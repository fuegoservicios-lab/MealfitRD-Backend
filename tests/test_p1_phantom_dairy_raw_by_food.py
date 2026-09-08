# -*- coding: utf-8 -*-
"""[P1-PHANTOM-DAIRY-RAW-BY-FOOD · 2026-09-07] «30 huevos» en la lista de la compra.

`_repair_name_phantom_dairy` sube a 30 g el lácteo que el NOMBRE del plato promete. Escribía la
línea de `ingredients_raw` por índice con la guarda más floja que hay —`_i_pd < len(raw)`, sin
comparar largos siquiera— y **no escala: sustituye el número inicial por 30**. Con las listas
desalineadas eso aterrizaba en otro alimento:

    display: 25 g de queso blanco   →  30 g de queso blanco     ✓
    raw    : 2 huevos               →  30 huevos
    raw    : 2 dientes de ajo       →  30 dientes de ajo
    raw    : 1.5 limones            →  30 limones

Medido en la flota: **18 comidas reales de 1.098** (una decimonovena era falso positivo de la
sonda — «queso fresco» contra «queso fresco bajo en sodio», genérico↔específico).

## Dos condiciones, no una

Acertar la línea no basta: el display venía en gramos y raw puede venir en tazas o en unidades.
Por eso se exige además que **la línea de raw empiece por gramos** y esté por debajo del objetivo.
Sin esa segunda condición, «½ taza de queso» se convertiría en «30 taza de queso».

Efecto medido: toca 46 comidas → 2, y los 18 daños → 0. La caída de 46 a 2 es la segunda
condición haciendo su trabajo, no el resolvedor fallando.

## Cómo se encontró

No estaba en mi auditoría: yo había buscado el *guard* (`_lockstep = …`) en vez de la *escritura*,
así que los sitios con guarda distinta quedaron fuera. *Buscar el síntoma que conoces encuentra
sólo los casos que ya conocías.*
"""
import inspect

import graph_orchestrator as go


def test_resuelve_la_linea_por_alimento():
    src = inspect.getsource(go._repair_name_phantom_dairy)
    assert "_raw_idx_for_display(raw, _s_pd, _i_pd, ings)" in src
    assert "raw[_i_pd]" not in src, "queda una escritura por índice ciego"


def test_exige_que_raw_venga_en_GRAMOS():
    """La segunda condición: acertar la línea no basta si viene en otra unidad."""
    src = inspect.getsource(go._repair_name_phantom_dairy)
    assert "g|gr|gramos" in src.split("_ri_pd")[1][:400], \
        "no se comprueba que la línea de raw esté en gramos antes de reescribirla"


def test_el_display_se_sigue_escribiendo_por_indice():
    """`ings[_i_pd]` alinea siempre consigo misma; aplicarle el resolvedor sería el error opuesto."""
    src = inspect.getsource(go._repair_name_phantom_dairy)
    assert "ings[_i_pd] = _new_pd" in src


# ------------------------------------------------- el helper compartido, ya en su tercera casa

def test_el_micro_closer_reusa_el_resolvedor():
    """`_sync_one_raw_line` tenía la TERCERA copia del mismo lookup.

    Tenerlo repetido es el mecanismo por el que unos sitios se arreglan y otros no — exactamente
    lo que pasó entre julio y hoy con esta familia.
    """
    src = inspect.getsource(go._sync_one_raw_line)
    assert "_raw_idx_for_display(_raw, display_old, idx" in src
    assert "_hits" not in src, "quedó la copia local de la búsqueda por alimento"


def test_el_knob_del_micro_closer_sigue_apagando_solo_el_by_food():
    """`MICRO_CLOSER_RAW_BY_FOOD` documenta que revierte el MAPEO por alimento, no el camino por
    índice. Al consolidar era fácil dejarlo inerte — un rollback que no revierte es peor que no
    tenerlo."""
    src = inspect.getsource(go._sync_one_raw_line)
    assert "if _i != idx and not MICRO_CLOSER_RAW_BY_FOOD:" in src


def test_el_contador_de_telemetria_solo_cuenta_los_by_food():
    """`_closer_raw_by_food` mide cuántas veces hizo falta el mapeo, no cuántas escrituras hubo."""
    src = inspect.getsource(go._sync_one_raw_line)
    i = src.index("_closer_raw_by_food")
    assert "if _i != idx:" in src[max(0, i - 120):i]
