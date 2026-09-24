# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-203 · 2026-09-24] Lo que el usuario lee ya no trae gramos con decimales.

Planes vivos del 24-sep: «67.56 g de edamame cocido» en el relleno de un usuario real (lo escribió el último pase de
proteína del merge, que corre DESPUÉS del pulido de la cola del escudo) y «1.21 g de Ajo», «1.07 g de Ajo» en el plan
canario del dueño (entre 1 y 2,5 g, la zona que el cuantizador deja sin tocar).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pulido_lineas as pl  # noqa: E402


@pytest.mark.parametrize("antes,despues", [
    ("1.21 g de Ajo", "½ diente de ajo"),
    ("1.07 g de ajo picado", "½ diente de ajo picado"),
    ("1.23 g de semillas de linaza", "½ cdta de semillas de linaza"),
    ("1.12 g de semillas de girasol sin sal", "½ cdta de semillas de girasol sin sal"),
    ("1.5 g de canela en polvo", "½ cdta de canela en polvo"),
    ("1.2 g de ajo en polvo", "½ cdta de ajo en polvo"),
    ("2.4 g de sal", "¼ cdta de sal"),
])
def test_entre_1_y_2_5_g_con_decimales_se_mide_en_cocina(antes, despues):
    assert pl.pulir_linea(antes) == despues
    assert pl.pulir_linea(despues) == despues, "idempotente"


@pytest.mark.parametrize("igual", [
    "2 g de sal",                    # ya es una medida
    "1.5 g de aguacate",             # ni ajo ni especia ni semilla: no se inventa una medida
    "3.2 g de semillas de chía",     # ≥ 2,5 g: dominio del cuantizador
    "½ diente de ajo",
    "1.44 g de maní tostado sin sal",  # «sin sal» no la hace sal (corrida guardada real)
])
def test_lo_demas_queda_igual(igual):
    assert pl.pulir_linea(igual) == igual


def test_el_ultimo_pase_de_proteina_vuelve_a_pulir(monkeypatch):
    import graph_orchestrator as go
    import protein_floor_last_word as pflw
    comida = {"meal": "Almuerzo", "name": "Pasta integral con costilla y edamame",
              "ingredients": ["25 g de edamame cocido"], "ingredients_raw": ["25 g de edamame cocido"]}
    plan = {"days": [{"day": 1, "meals": [comida]}]}
    mediciones = iter([
        {"cortos": [{"dia": 1, "proteina_g": 80.0}], "cumple": False, "dias_medidos": 1, "piso_g": 94.5},
        {"cortos": [], "cumple": True, "dias_medidos": 1, "piso_g": 94.5},
    ])
    monkeypatch.setattr(pflw, "medir", lambda *a, **k: next(mediciones))

    def _bump(pd):  # lo que hizo el re-encuadre en el plan real
        pd["days"][0]["meals"][0]["ingredients"] = ["67.56 g de edamame cocido"]
        pd["days"][0]["meals"][0]["ingredients_raw"] = ["67.56 g de edamame cocido"]
        return True

    monkeypatch.setattr(go, "reconcile_protein_band_post_finalize", _bump)
    monkeypatch.setattr(go, "_cap_unrealistic_portions", lambda *a, **k: None)
    monkeypatch.setattr(go, "refresh_delivered_macros", lambda *a, **k: None)
    pflw.reencuadra_y_mide(plan, surface="chunk-T1 semana 8")
    assert comida["ingredients"] == ["70 g de edamame cocido"], comida["ingredients"]


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 203 and m.group(2) >= "2026-09-24"
