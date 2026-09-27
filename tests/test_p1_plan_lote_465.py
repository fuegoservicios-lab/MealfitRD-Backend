# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-465 · 2026-09-27] La proteína sustituta de la compra única, por proteína: pescado en lata antes que
garbanzos para un omnívoro.

Los garbanzos cocidos tienen ~9 g de proteína por 100 g y la pechuga que sustituyen ~23 (cruda). En la rueda de tres
(atún, sardinas, garbanzos) un tercio de las sustituciones de un omnívoro caía en garbanzos y el día perdía proteína
que ningún pase podía devolver: replay del chain completo forzando la compra única, días con garbanzos a 0,57-0,71 del
objetivo. Ahora la rueda de un omnívoro es de pescado en lata y los garbanzos quedan de reserva: sólo si los dos
pescados ya están en el día o no son seguros. El vegetariano sigue con legumbres.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import compra_unica as cu  # noqa: E402


def test_el_omnivoro_recibe_pescado_en_lata():
    subs = {cu.sustituto_seguro(cu.PROTEINA_TABLA, s, False) for s in range(8)}
    assert subs == {"atun en agua", "sardinas en lata"}, subs


def test_los_garbanzos_son_la_reserva():
    # [P1-PLAN-LOTE-495] antes van las claras pasteurizadas; los garbanzos, si no se pueden cocinar o no son seguras
    assert cu.sustituto_seguro(cu.PROTEINA_TABLA, 0, False, evitar={"atun en agua", "sardinas en lata"}) \
        == "claras de huevo"
    assert cu.sustituto_seguro(cu.PROTEINA_TABLA, 3, False, alergias=["Pescado"]) == "claras de huevo"
    assert cu.sustituto_seguro(cu.PROTEINA_TABLA, 3, False, alergias=["Pescado"], excluir=("claras de huevo",)) \
        == "garbanzos cocidos"


def test_el_vegetariano_sigue_con_legumbres():
    subs = {cu.sustituto_seguro(cu.PROTEINA_TABLA, s, True) for s in range(4)}
    assert subs == {"garbanzos cocidos", "lentejas cocidas"}, subs


def test_la_proyeccion_del_mes_de_un_omnivoro_no_compra_garbanzos_por_la_rueda():
    import shopping_calculator as sc
    single = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "none",
                           "batch_cooking": "never"}, "diet": {"type": "balanced", "allergies": []}}
    p = {"total_days_requested": 30, "_plan_policy": {"effective": single},
         "days": [{"day": n, "meals": [{"meal": "Almuerzo", "name": "Pollo",
                                        "ingredients": ["200 g de pechuga de pollo"],
                                        "ingredients_raw": ["200 g de pechuga de pollo"]}]} for n in (1, 2, 3)]}
    cu._MEMO.clear()
    try:
        resto = [x for d in sc.shopping_source_days(p)[7:] for m in d["meals"] for x in m["ingredients_raw"]]
    finally:
        cu._MEMO.clear()
    assert resto and not any("garbanzo" in cu._sa(x) for x in resto), resto[:6]
    assert any("atun" in cu._sa(x) for x in resto) and any("sardina" in cu._sa(x) for x in resto)
