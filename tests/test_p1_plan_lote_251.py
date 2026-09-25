# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-251 · 2026-09-25] Toronja/pomelo con estatina, calcioantagonista o anticoagulante: guard determinista.

Las reglas `statin`, `calcium_channel_blocker` y `anticoagulant` de `medication_rules` piden «EVITA la toronja/pomelo y
su jugo», pero solo en el prompt. «Toronja» es fila del catálogo y salió en 2 comidas de 600 planes reales.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import medication_rules as mr  # noqa: E402

PLAN = {"days": [{"day": 1, "meals": [
    {"meal": "Desayuno", "name": "Mangú ligero con huevo y cebollita",
     "ingredients": ["½ plátano verde mediano", "2 huevos", "1 cebolla", "½ toronja mediana"],
     "recipe": ["Hierve el plátano 15 min.", "Sirve con la toronja en gajos."]},
    {"meal": "Merienda", "name": "Bowl de yogur, arándanos y toronja",
     "ingredients": ["¾ taza de yogurt natural", "15 g de arándanos", "½ toronja"],
     "recipe": ["Mezcla todo."]},
]}]}


@pytest.mark.parametrize("fd", [{"medications": ["Atorvastatina"]}, {"otherMedications": "amlodipino 5 mg"},
                                {"medications": ["Warfarina"]}, {"otherMedications": "simvastatina y nifedipina"}])
def test_detecta_la_toronja_con_esos_farmacos(fd):
    v = mr.grapefruit_violations(PLAN, fd)
    assert len(v) == 2 and all("toronja" in x for x in v), v


@pytest.mark.parametrize("fd", [{"medications": ["Metformina"]}, {"medications": ["Fenelzina"]}, {}, None])
def test_sin_esos_farmacos_no_aplica(fd):
    assert mr.grapefruit_violations(PLAN, fd) == []
    assert mr.grapefruit_review_issues(PLAN, fd) == []


def test_pomelo_y_plural():
    plan = {"days": [{"meals": [{"name": "x", "ingredients": ["1 taza de jugo de pomelo", "2 toronjas"]}]}]}
    assert len(mr.grapefruit_violations(plan, {"medications": ["Atorvastatina"]})) == 2


def test_el_revisor_pide_reintento_no_emergencia():
    import graph_orchestrator as go
    issues = mr.grapefruit_review_issues(PLAN, {"medications": ["Atorvastatina"]})
    assert len(issues) == 1
    # Ni una marca de peligro agudo: si otro crítico no agudo convive con éste, sigue siendo reintento (G18).
    assert go._critical_is_non_acute(issues) is True
    assert "desayuno del día 1" in issues[0] and "merienda del día 1" in issues[0], issues
    # Un plato «vegano» o una línea «sin cocer» no pueden colarse en el texto (son marcas de peligro agudo).
    plan = {"days": [{"day": 2, "meals": [{"meal": "Cena", "name": "Bowl vegano con toronja",
                                           "ingredients": ["1 toronja sin cocer"]}]}]}
    assert go._critical_is_non_acute(mr.grapefruit_review_issues(plan, {"medications": ["Warfarina"]})) is True
    src =(_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index('grapefruit_review_issues(plan, form_data)')
    assert '_severity_max(severity, "high")' in src[i:i + 300]


def test_el_backstop_de_las_superficies_de_cambio_lo_ve():
    import graph_orchestrator as go
    meal = copy.deepcopy(PLAN["days"][0]["meals"][0])
    out = go.clinical_backstop_for_meal(meal, allergies=[], diet_type="balanced",
                                        form_data={"medications": ["Atorvastatina"]})
    assert any("toronja" in o for o in out), out
    assert not go.clinical_backstop_for_meal(meal, allergies=[], diet_type="balanced",
                                             form_data={"medications": ["Metformina"]})


def test_la_ultima_palabra_la_retira_y_avisa():
    import restricciones_finales as rf
    p = copy.deepcopy(PLAN)
    out = rf.retirar_prohibidos(p, {"medications": ["Atorvastatina"], "allergies": [], "dislikes": []})
    m0, m1 = p["days"][0]["meals"]
    assert "½ toronja mediana" not in m0["ingredients"] and "2 huevos" in m0["ingredients"]
    assert any(s.startswith("⚠️ Interacción con tu medicamento") and "toronja" in s for s in m0["recipe"]), m0["recipe"]
    # El plato que SE LLAMA «…y toronja» no se toca (el nombre mentiría): lo rechaza el revisor y queda registrado.
    assert "½ toronja" in m1["ingredients"]
    assert any(r["kind"] == "farmaco" and r["term"] == "toronja" for r in out["en_el_plato"])


def test_ancla():
    assert "tooltip-anchor: P1-PLAN-LOTE-251-TORONJA" in (_BACKEND / "medication_rules.py").read_text(encoding="utf-8")


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 251
