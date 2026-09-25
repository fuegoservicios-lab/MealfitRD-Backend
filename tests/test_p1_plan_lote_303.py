# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-303 · 2026-09-25] «1 ½ cdtas» es UN número mixto.

Batería del 25-sep (mes sin pescado): «mide 25 g de casabe, 1 2 cdtas de mantequilla de maní natural». El paso decía
«1 ½ cdtas» (con espacio); la mención empezaba en el «½» y la reescritura a la lista («2 cdtas») dejaba el «1» delante."""
from __future__ import annotations

import graph_orchestrator as go


def test_uno_y_medio_con_espacio_no_se_vuelve_uno_dos():
    m = {"ingredients": ["1¼ tortas pequeñas de casabe", "2 cdtas de mantequilla de maní", "1 ciruela fresca"],
         "recipe": ["Mise en place: corta 1 ciruela en trozos; mide 25 g de casabe, 1 ½ cdtas de mantequilla de maní "
                    "natural y 1 pizca de canela."]}
    go._sync_recipe_step_quantities(m)
    s = m["recipe"][0]
    assert "1 2 cdtas" not in s, s
    assert "25 g de casabe, 2 cdtas de mantequilla de maní natural" in s, s


def test_el_numero_mixto_es_una_sola_cantidad():
    mm = go._STEP_QTY_MENTION_RE.search("añade 1 ½ tazas de leche")
    assert mm and mm.group("qty") == "1 ½" and mm.group("unit") == "tazas"
    assert go._qtysync_qty_to_float(mm.group("qty")) == 1.5
    # lo de antes sigue igual: el pegado «2½» y la fracción ASCII intacta
    assert go._STEP_QTY_MENTION_RE.search("unta 2½ cdas de mantequilla").group("qty") == "2½"
    m = {"ingredients": ["1¼ cda de aceite de oliva (18ml)"], "recipe": ["mezcla con 1/2 cda de aceite de oliva."]}
    go._sync_recipe_step_quantities(m)
    assert "1/2 cda de aceite" in m["recipe"][0]
