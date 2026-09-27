# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-587 · 2026-09-27] La avena se cocina con la leche que la espesa; el resto se sirve en un vaso.

Batería real (estudiante): «Avena cremosa de remolacha…» con 30 g de avena y 575 ml de leche, «cocina 7-9 minutos
removiendo hasta que espese».
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import avena_con_su_leche as acl  # noqa: E402


def _meal(avena="30 g de avena", leche="575 ml de leche descremada",
          toque="El Toque de Fuego: en una olla pequeña lleva la leche descremada a fuego medio con la canela; cuando rompa "
                "el hervor añade la avena y cocina 7-9 minutos removiendo hasta que espese.",
          nombre="Avena cremosa de remolacha y canela con queso blanco fresco"):
    return {"name": nombre, "ingredients": [avena, leche, "½ cdta de canela en polvo"],
            "ingredients_raw": [avena, leche, "½ cdta de canela en polvo"],
            "recipe": [f"Mise en place: mide {avena} y {leche}.", toque,
                       "Montaje: sirve la avena caliente en un tazón hondo."]}


def test_la_bateria_cocina_con_240_y_sirve_335_en_un_vaso():
    m = _meal()
    lista = list(m["ingredients"])
    assert acl.separar(m) == 335
    assert "lleva 240 ml de la leche descremada a fuego medio" in m["recipe"][1]
    assert m["recipe"][2].endswith("Sirve los 335 ml de leche restantes en un vaso aparte.")
    assert m["ingredients"] == lista, "la leche de la lista (compra y macros) no se toca"


def test_cocina_la_avena_con_la_leche():
    m = _meal(avena="15 g de avena", leche="680 ml de leche descremada",
              toque="El Toque de Fuego: cocina la avena con la leche descremada a fuego medio 6-8 min, removiendo hasta que "
                    "quede cremosa.")
    assert acl.separar(m) == 530
    assert "cocina la avena con 150 ml de la leche descremada" in m["recipe"][1]


def test_una_avena_cremosa_se_queda_como_esta():
    m = _meal(leche="240 ml de leche descremada")                  # 8 ml/g
    assert acl.separar(m) == 0 and "240 ml de la leche" not in m["recipe"][1]


def test_justo_sobre_el_umbral():
    assert acl.separar(_meal(leche="430 ml de leche descremada")) == 190     # 14,3 ml/g
    assert acl.separar(_meal(leche="420 ml de leche descremada")) == 0       # 14,0 ml/g


@pytest.mark.parametrize("kw", [
    {"toque": "El Toque de Fuego: mezcla la avena con la leche y deja remojar toda la noche en la nevera."},
    {"nombre": "Batido de avena y guineo", "toque": "El Toque de Fuego: licúa la avena con la leche 1 minuto."},
    {"leche": "575 ml de leche evaporada"},
])
def test_lo_que_no_es_una_gacha_con_leche_no_se_toca(kw):
    m = _meal(**kw)
    antes = [list(m["ingredients"]), list(m["recipe"])]
    assert acl.separar(m) == 0 and [m["ingredients"], m["recipe"]] == antes


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("avena_con_su_leche").separar(meal)  # [P1-PLAN-LOTE-587]' in src
