# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-183 · 2026-09-23] El prompt no pide lo que la puerta rechaza.

Batería real rd11 (adulto mayor con HTA): un intento entero se perdió por «Crepas saladas de harina con ricotta y berro»
en la cena — la puerta de horario (`SLOT_INAPPROPRIATE_FOODS`, «comida de desayuno en la cena») la rechaza, y la regla
2.5 del propio prompt decía «harina → panqueques… Aplica ESPECIALMENTE a MERIENDA y CENA». Además, la línea dura de
alergias del 182 cortaba los términos por orden alfabético: «almeja, arequipe, calamar…» y fuera leche, queso, yogur."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def test_la_regla_de_los_staples_ya_no_pide_panqueques_en_la_cena():
    src = (_BACKEND / "prompts" / "day_generator.py").read_text(encoding="utf-8")
    assert src.count("pero en la CENA nunca panqueques, crepas, waffles, cereal ni avena") == 3
    assert "ESPECIALMENTE a MERIENDA y CENA (no solo al desayuno)." not in src


def test_lo_que_el_prompt_prohibe_en_la_cena_es_lo_que_la_puerta_rechaza():
    from constants import SLOT_INAPPROPRIATE_FOODS
    reglas = SLOT_INAPPROPRIATE_FOODS["cena"]
    toks = next(r["tokens"] for r in reglas if "desayuno en la cena" in r["label"])
    for t in ("panqueque", "crepa", "waffle", "cereal", "avena"):
        assert t in toks, t


def test_la_linea_dura_nombra_primero_lo_comun():
    from prompts.day_generator import allergy_hard_line
    linea = allergy_hard_line(["Lacteos", "Mariscos"])
    incl = re.search(r"\(incluye: ([^)]*)\)", linea).group(1).split(", ")
    for t in ("leche", "queso", "yogur", "camaron"):
        assert t in incl, (t, incl)
    assert allergy_hard_line(["Ninguna"]) == ""


def test_sin_lacteos_la_linea_dura_da_alternativas_filtradas():
    from prompts.day_generator import allergy_hard_line
    linea = allergy_hard_line(["Lacteos"])
    assert "Sin lácteos: los batidos van con agua, hielo o leche de coco" in linea and "fruta con maní" in linea
    con_mani = allergy_hard_line(["Lacteos", "Maní"])
    assert "fruta con maní" not in con_mani and "casabe con aguacate" in con_mani
    assert "Sin lácteos" not in allergy_hard_line(["Mariscos"])


def test_embarazo_la_nota_cubre_carnes_y_aves_y_el_revisor_la_lee():
    import graph_orchestrator as go
    plan = {"days": [{"day": 2, "meals": [{"meal": "Cena", "name": "Pollo guisado con arroz integral",
                                           "ingredients": ["1 pechuga de pollo (≈150 g)", "½ taza de arroz integral",
                                                           "½ tomate", "1 huevo"],
                                           "recipe": ["Guisa el pollo con sofrito 25 minutos.", "Sirve con el arroz."]}]}]}
    go._apply_pregnancy_food_safety_annotations(plan, {"medicalConditions": ["Embarazo"], "gender": "female"})
    comida = plan["days"][0]["meals"][0]
    nota = next(s for s in comida["recipe"] if s.startswith("🤰"))
    assert "74 °C por dentro, sin partes rosadas" in nota
    assert "74 °C por dentro" in go._meal_safety_notes_for_summary(comida)


def test_embarazo_sin_carne_no_hay_clausula_de_carne():
    import graph_orchestrator as go
    plan = {"days": [{"day": 1, "meals": [{"name": "Tilapia al horno", "ingredients": ["120 g de tilapia", "½ tomate"],
                                           "recipe": ["Hornea la tilapia."]}]}]}
    go._apply_pregnancy_food_safety_annotations(plan, {"medicalConditions": ["Embarazo"], "gender": "female"})
    assert not any("sin partes rosadas" in s for s in plan["days"][0]["meals"][0]["recipe"])


def test_hta_frutos_secos_sin_sal():
    import etiquetas_clinicas as etq
    for antes, despues in (("15 g de maní fileteado", "15 g de maní fileteado sin sal"),
                           ("1¼ cdas de mantequilla de maní natural", "1¼ cdas de mantequilla de maní natural sin sal"),
                           ("10 g de almendras fileteadas", "10 g de almendras fileteadas sin sal"),
                           ("10 g de maní tostado sin sal", "10 g de maní tostado sin sal"),
                           ("¼ cdta de nuez moscada", "¼ cdta de nuez moscada")):
        assert etq._linea_hta(antes) == despues, antes


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 183 and m.group(2) >= "2026-09-23"
