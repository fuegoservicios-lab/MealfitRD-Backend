# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-195 · 2026-09-24] DM2: el casabe que queda lleva la nota de cómo comerlo, donde el revisor la lee.

rd20 (DM2 + insulina): el 2.º intento cayó CRÍTICO por «¾ de torta de casabe tiene una carga glucémica alta… no es
adecuada para el control de la diabetes»; en otras corridas el mismo casabe pasó como aviso. El lote 178 deja uno por
bloque a propósito (mesa dominicana): lo que faltaba es que el plan diga cómo se come."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_DM2 = {"medicalConditions": ["Diabetes tipo 2"], "medications": ["Insulina"], "dietType": "balanced"}


def _plan():
    casabe = {"meal": "Merienda AM", "name": "Casabe con queso blanco y mango",
              "ingredients": ["1 torta pequeña de casabe", "30 g de queso blanco", "½ mango"],
              "recipe": ["Sirve el casabe con el queso y el mango."]}
    otra = {"meal": "Almuerzo", "name": "Pollo guisado con arroz", "ingredients": ["150 g de pechuga de pollo"],
            "recipe": ["Guisa el pollo."]}
    return {"days": [{"day": 3, "meals": [casabe, otra]}]}, casabe, otra


def test_dm2_el_casabe_lleva_la_nota_que_lee_el_revisor():
    import etiquetas_clinicas as ec
    import graph_orchestrator as go
    plan, casabe, otra = _plan()
    ec.etiquetar(plan, _DM2)
    assert any("sube la glucosa" in p for p in casabe["recipe"]), casabe["recipe"]
    assert "sube la glucosa" in go._meal_safety_notes_for_summary(casabe)
    assert not any("sube la glucosa" in p for p in otra["recipe"]), "sólo el plato con casabe"
    ec.etiquetar(plan, _DM2)
    assert sum("sube la glucosa" in p for p in casabe["recipe"]) == 1, "idempotente"


def test_sin_diabetes_no_hay_nota():
    import etiquetas_clinicas as ec
    plan, casabe, _ = _plan()
    ec.etiquetar(plan, {"medicalConditions": ["Ninguna"]})
    assert not any("sube la glucosa" in p for p in casabe["recipe"])


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 195 and m.group(2) >= "2026-09-24"
