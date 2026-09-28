# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-663 · 2026-09-28] El «fresco» del queso no deja crudo el brócoli de al lado.

Batería real del 28-sep sobre el 636 (estudiante, día 3): «Canoas crujientes de plátano verde rellenas de queso fresco y
brócoli al ajo» — 250 g de brócoli que ningún paso cocina, servido «con el queso blanco fresco y el brócoli al ajo como
relleno»; el 540 leía ese «fresco» como brócoli servido crudo. Replay de 5.054 comidas: 1 cambio, éste.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import verdura_sin_coccion as vsc  # noqa: E402

_CANOAS = {
    "name": "Canoas crujientes de plátano verde rellenas de queso fresco y brócoli al ajo con edamame",
    "ingredients": ["30 g de queso blanco", "½ plátano verde mediano", "250 g de brócoli", "½ diente de ajo",
                    "1 cdta de aceite de oliva", "300 g de edamame cocido"],
    "recipe": [
        "Mise en place: pela ½ plátano verde y córtalo a lo largo; separa 250 g de brócoli en floretes pequeños y pica "
        "½ diente de ajo.",
        "El Toque de Fuego: pincela las mitades de plátano verde con el aceite y cocínalas en la airfryer a 190 °C "
        "durante 15-18 min, hasta que estén tiernas y doradas.",
        "Montaje: sirve las canoas de plátano verde recién hechas, con el queso blanco fresco y el brócoli al ajo como "
        "relleno. Acompaña con edamame.",
    ],
}


def test_el_fresco_del_queso_no_es_el_del_brocoli():
    m = copy.deepcopy(_CANOAS)
    assert vsc.cocer(m) == 1
    assert m["recipe"][1].startswith("💡 Cocción previa: hierve el brócoli")


@pytest.mark.parametrize("montaje", [
    "Montaje: sirve las canoas con el brócoli crudo en floretes finos como relleno.",
    "Montaje: sirve las canoas con una ensalada de brócoli y queso.",
    "Montaje: sirve las canoas con el brócoli fresco rallado encima.",
])
def test_el_brocoli_servido_crudo_se_queda_crudo(montaje):
    m = copy.deepcopy(_CANOAS)
    m["recipe"][2] = montaje
    antes = list(m["recipe"])
    assert vsc.cocer(m) == 0 and m["recipe"] == antes
