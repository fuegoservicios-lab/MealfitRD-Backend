# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-581 · 2026-09-27] «Nada» de tiempo + cocina por tandas a menudo.

Batería real (turno nocturno, «Nada», tandas, 192 g de proteína): el prompt decía a la vez «nada de guisos» y «cocina por
tandas… un guiso», el cerrador sólo ofrecía proteínas «listas» y un día quedó al 84 % de su proteína.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import horizon  # noqa: E402
import proteina_lista as pl  # noqa: E402

_TANDAS = {"cookingTime": "none", "batchCooking": "often"}


def test_el_prompt_dice_que_lo_de_la_tanda_llega_cocido():
    regla = horizon.cooking_time_rule(_TANDAS)
    assert "COCINA POR TANDAS" in regla and "YA COCIDO" in regla, regla
    gen = horizon.explain_form_codes_for_prompt(_TANDAS)["cookingTime"]
    assert gen.endswith(regla), "el generador y los correctores leen el MISMO texto"


def test_sin_tandas_el_texto_de_siempre():
    for fd in ({"cookingTime": "none"}, {"cookingTime": "none", "batchCooking": "never"},
               {"cookingTime": "30min", "batchCooking": "often"}):
        assert "COCINA POR TANDAS" not in horizon.cooking_time_rule(fd), fd


def test_el_cerrador_no_se_limita_a_lo_listo():
    cands = [(31.0, "Pechuga de pollo", None), (19.0, "Atún", None), (13.0, "Huevos", None)]
    assert pl.filtrar_listas(cands, _TANDAS, listos={"atun"}) == cands
    assert [c[1] for c in pl.filtrar_listas(cands, {"cookingTime": "none"}, listos={"atun"})] == ["Atún", "Huevos"]
