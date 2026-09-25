# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-246 · 2026-09-25] IMAO + tiramina con guard determinista (revisor, superficies de cambio, escudo).

La regla `maoi` de `medication_rules` llama a la interacción «potencialmente LETAL» y solo existía como prompt.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import medication_rules as mr  # noqa: E402

IMAO = {"medications": ["Fenelzina"]}
PLAN = {"days": [{"day": 1, "meals": [
    {"meal": "Almuerzo", "name": "Pasta integral con pollo",
     "ingredients": ["100 g de pasta integral", "150 g de pechuga de pollo", "20 g de queso parmesano",
                     "1 cda de salsa de soya baja en sodio", "30 g de queso blanco fresco", "100 g de ricotta"]},
]}]}


def test_detecta_lo_anejado_y_lo_fermentado_solo_con_imao():
    v = mr.tyramine_violations(PLAN, IMAO)
    txt = " ".join(v)
    assert "parmesano" in txt and "salsa de soya" in txt, v
    assert "queso blanco fresco" not in txt and "ricotta" not in txt, "los quesos frescos no"
    assert mr.tyramine_violations(PLAN, {"medications": ["Metformina"]}) == []
    assert mr.tyramine_violations(PLAN, {}) == []


def test_el_backstop_de_las_superficies_de_cambio_lo_ve():
    import graph_orchestrator as go
    meal = copy.deepcopy(PLAN["days"][0]["meals"][0])
    out = go.clinical_backstop_for_meal(meal, allergies=[], diet_type="balanced", form_data=IMAO)
    assert any("tiramina" in o for o in out), out


def test_la_ultima_palabra_retira_lo_anadido():
    import restricciones_finales as rf
    p = copy.deepcopy(PLAN)
    rf.retirar_prohibidos(p, dict(IMAO, allergies=[], dislikes=[]))
    ings = p["days"][0]["meals"][0]["ingredients"]
    assert not any("parmesano" in i or "soya" in i for i in ings), ings
    assert "150 g de pechuga de pollo" in ings


def test_el_revisor_lo_marca_critico():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index('_tyr246 = __import__("medication_rules").tyramine_violations(plan, form_data)')
    assert 'severity = _severity_max(severity, "critical")' in src[i:i + 600]
    import graph_orchestrator as go
    assert not go._critical_is_non_acute(["TIRAMINA CON IMAO (interacción medicamentosa peligrosa): …"])
    assert "tooltip-anchor: P1-PLAN-LOTE-246-TIRAMINA" in (_BACKEND / "medication_rules.py").read_text(encoding="utf-8")


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 246
