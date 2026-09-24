# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-188 · 2026-09-23] Batería rd16 (alergia a lácteos y mariscos): plan de EMERGENCIA otra vez.

Intento 1: «¼ taza de yogurt griego» en un batido (la guarda lo cazó). Intento 2: rechazo CRÍTICO «No se puede confirmar
que el pan integral… estén libres de lácteos. Verifique sus etiquetas» — «verifique» y «no se puede confirmar» no casaban
con los patrones de verificación («verificar», «confirme si»). Dos críticos ⇒ emergencia.

  · Lácteos: la decisión de P0-ALLERGEN-SUBS era no sustituirlos porque el catálogo no tenía un target libre del
    alérgeno («palanca de DATOS»). Ya lo tiene: yogur de coco, leches vegetales, tofu firme. Se sustituye con el primer
    candidato que no choque con OTRA alergia declarada.
  · Los patrones de verificación, por raíz verbal.
  · Embarazo: «cebolla y ají morrón crudos sin indicar lavarlos» (crítico) — la cláusula de lavado no los tenía."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import condition_rules as cr  # noqa: E402


def _repls(alergias):
    return {s["label"]: s["replacement"] for s in cr.collect_allergen_substitutions({"allergies": alergias})}


def test_lacteos_se_sustituyen_por_lo_que_hay_en_el_catalogo():
    r = _repls(["Lacteos"])
    assert r["lácteo (yogur)"] == "Yogur de coco" and r["lácteo (leche)"] == "Leche de avena"
    assert r["lácteo (queso)"] == "Tofu firme" and r["lácteo (mantequilla)"] == "Aceite de oliva"


def test_el_sustituto_no_choca_con_otra_alergia():
    assert _repls(["Lacteos", "Gluten"])["lácteo (leche)"] == "Leche de almendras", "la avena choca con el gluten"
    assert _repls(["Lacteos", "Gluten", "Frutos secos"])["lácteo (leche)"] == "Leche de coco"
    assert "lácteo (queso)" not in _repls(["Lacteos", "Soya"]), "sin candidato limpio, la fila no se aplica"


@pytest.mark.parametrize("linea", ["1 taza de leche de coco", "½ taza de yogur de coco", "1 cda de mantequilla de maní",
                                   "30 g de queso vegano", "200 ml de leche de almendras"])
def test_lo_que_ya_es_vegetal_no_se_toca(linea):
    negs = cr._ALLERGEN_DAIRY_NEGATIVES
    from constants import strip_accents
    assert any(n in strip_accents(linea.lower()) for n in negs), linea


def test_el_plan_real_del_batido():
    import graph_orchestrator as go
    plan = {"days": [{"day": 2, "meals": [{"meal": "Desayuno", "name": "Batido fresco de manzana y aguacate",
                                           "ingredients": ["1 manzana", "¼ taza de yogurt griego sin azúcar", "30 g de aguacate"],
                                           "ingredients_raw": ["1 manzana", "¼ taza de yogurt griego sin azúcar", "30 g de aguacate"],
                                           "recipe": ["Licúa la manzana con el yogurt griego y el aguacate."],
                                           "protein": 6, "carbs": 30, "fats": 7, "cals": 200}]}]}
    assert go._apply_allergen_substitutions(plan, {"allergies": ["Lacteos"]}) == 1
    m = plan["days"][0]["meals"][0]
    assert not any("yogurt griego" in x.lower() for x in m["ingredients"]), m["ingredients"]
    assert any("yogur de coco" in x.lower() for x in m["ingredients"]), m["ingredients"]
    assert go._scan_allergen_violations(plan, ["Lacteos"]) == [], "la guarda ya no ve alérgeno"


@pytest.mark.parametrize("issue", [
    "No se puede confirmar que el pan integral familiar, la harina de Negrito y la mantequilla de maní estén libres de "
    "lácteos. Verifique sus etiquetas y use versiones sin leche, suero, caseína ni otros derivados lácteos antes de servir.",
    "Confirme que la tortilla integral no contenga leche.",
])
def test_verificacion_por_raiz_verbal_es_aviso(issue):
    import graph_orchestrator as go
    aprobado, reales, _s, avisos = go._downgrade_reviewer_verification_demands(False, [issue], "critical")
    assert aprobado and reales == [] and avisos == [issue]


def test_embarazo_cebolla_y_aji_se_lavan():
    import graph_orchestrator as go
    comida = {"meal": "Cena", "name": "Ensalada de lentejas con cebolla y ají morrón",
              "ingredients": ["½ taza de lentejas cocidas", "¼ cebolla morada", "½ ají morrón"], "recipe": ["Mezcla y sirve."]}
    go._apply_pregnancy_food_safety_annotations({"days": [{"day": 3, "meals": [comida]}]},
                                                {"medicalConditions": ["Embarazo"], "gender": "female"})
    assert any("lava y desinfecta" in s for s in comida["recipe"]), comida["recipe"]


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 188 and m.group(2) >= "2026-09-23"
