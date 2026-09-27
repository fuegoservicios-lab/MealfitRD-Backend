# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-496 · 2026-09-27] Un plato con dos proteínas frescas recibe UN duradero.

Replay forzado de los días 21+ (322 planes): 8 de 1.240 comidas con proteína sustituida tenían dos líneas de proteína
(«1 filete de pescado» + «75 g de camarones cocidos», «1½ filetes de pescado» + «½ filete de pescado») y la rueda —que
aparta lo que el día ya lleva— les daba dos duraderos distintos: «Atún con zanahoria y batata… Calienta atún y sírvela
como proteína del plato… Acompaña con atún y sardinas en lata». La segunda recibe el de la primera y suma su cantidad.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import compra_unica as cu  # noqa: E402
import graph_orchestrator as go  # noqa: E402
import sustitucion_fresca as sf  # noqa: E402

SINGLE = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "none",
                       "batch_cooking": "never"}, "diet": {"type": "balanced", "allergies": []}}


class _NoopDB:
    def macros_from_ingredient_string(self, s):
        return None


def test_la_segunda_proteina_recibe_el_mismo_duradero():
    r = cu.sustituir_linea("75 g de camarones cocidos", 20, {"need_days": 21, "allow_frozen": False}, semilla=1,
                           evitar={"atun en agua"}, forzar="atun en agua", gramos_de=lambda _t: None, listo=False)
    assert r and r[1] == "atun en agua" and r[0] == "75 g de atún en agua", r


def test_en_el_plato_se_suman(monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = [{"day": i + 1, "meals": [{"meal": "Cena", "name": "x", "ingredients": ["1 taza de arroz"],
                                      "ingredients_raw": ["1 taza de arroz"]}]} for i in range(22)]
    days[20]["meals"] = [{"meal": "Almuerzo", "name": "Pescado con zanahoria y batata",
                          "ingredients": ["150 g de filete de pescado", "140 g de batata", "75 g de camarones cocidos"],
                          "ingredients_raw": ["150 g de filete de pescado", "140 g de batata", "75 g de camarones cocidos"],
                          "recipe": ["Montaje: sirve el pescado con la batata. Acompaña con camarones."]}]
    go._single_trip_fresh_substitute(days, db=_NoopDB(), effective=SINGLE, diet="balanced")
    m = days[20]["meals"][0]
    prot = [x for x in m["ingredients"] if "atún" in x or "sardina" in x]
    assert len(prot) == 1, m["ingredients"]
    assert prot[0] in ("225 g de atún en agua", "225 g de sardinas en lata"), prot
    assert "140 g de batata" in m["ingredients"]
    assert [x for x in m["ingredients_raw"] if "atún" in x or "sardina" in x] == prot, m["ingredients_raw"]


def test_la_suma_solo_con_la_misma_medida():
    assert sf._suma_lineas("150 g de atún en agua", "75 g de atún en agua") == "225 g de atún en agua"
    assert sf._suma_lineas("4 claras de huevo", "3 claras de huevo") == "7 claras de huevo"
    assert sf._suma_lineas("150 g de atún en agua", "1 lata de atún en agua") is None
    assert sf._suma_lineas("150 g de atún en agua", "75 g de sardinas en lata") is None
