# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-193 · 2026-09-24] Semillas y frutos secos: 40 g por línea, como máximo.

De 270 líneas de semillas/frutos secos en las corridas guardadas, 8 pasaban de 40 g y 6 eran del PLAN DE EMERGENCIA:
«14,5 cdas de semillas de chía» (≈145 g, desayuno de 1.231 kcal) y «¾ taza de semillas de chía» (≈105 g). El plan de
emergencia pone a cada comida sus macros de plantilla y el cerrador de banda escala lo único denso que encuentra."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import topes_por_linea as ts  # noqa: E402


class _DB:
    _POR = {"cda": 10.0, "cdas": 10.0, "taza": 140.0, "tazas": 140.0, "g": 1.0, "ml": 1.0}

    def grams_from_ingredient_string(self, s):
        m = re.match(r"^\s*([\d.,]+|¾|½|¼)\s*(cdas?|tazas?|g|ml)\s+de\s+", str(s))
        if not m:
            return None
        q = {"¾": 0.75, "½": 0.5, "¼": 0.25}.get(m.group(1)) or float(m.group(1).replace(",", "."))
        return q * self._POR[m.group(2)]

    def macros_from_ingredient_string(self, s):
        g = self.grams_from_ingredient_string(s)
        return None if g is None else {"grams": g, "kcal": g * 5, "protein": g * 0.2, "carbs": g * 0.3, "fats": g * 0.3}


def _dia(*lineas):
    ings = list(lineas) + ["1⅓ tazas de avena"]
    return [{"day": 2, "meals": [{"meal": "Desayuno", "name": "Avena con Frutas y Semillas", "ingredients": list(ings),
                                  "ingredients_raw": list(ings), "recipe": ["Mezcla y sirve."]}]}]


def test_la_chia_del_plan_de_emergencia_baja_a_40_g():
    dias = _dia("14.5 cdas de semillas de chía")
    assert ts.cap(dias, db=_DB()) == 1
    m = dias[0]["meals"][0]
    assert m["ingredients"][0] == "4 cdas de semillas de chía", m["ingredients"]
    assert _DB().grams_from_ingredient_string(m["ingredients_raw"][0]) <= 40.5, m["ingredients_raw"]


@pytest.mark.parametrize("linea", ["250 ml de leche de almendras", "30 g de nueces", "60 g de harina de almendras",
                                   "1 cda de aceite de girasol", "Semillas de chía al gusto"])
def test_lo_que_no_es_fruto_seco_o_no_se_pasa_no_se_toca(linea):
    dias = _dia(linea)
    assert ts.cap(dias, db=_DB()) == 0
    assert dias[0]["meals"][0]["ingredients"][0] == linea


@pytest.mark.parametrize("linea, esperado_max", [("300 g de atún en agua", 170.5), ("295 g de atún claro en agua", 170.5),
                                                 ("280 g de sardinas en lata", 170.5)])
def test_el_pescado_en_lata_baja_a_170_g(linea, esperado_max):
    """rd20 (DM2): «la porción de atún de 331 g en una sola comida es excesiva… mercurio y sodio»."""
    dias = _dia(linea)
    assert ts.cap(dias, db=_DB()) == 1
    assert _DB().grams_from_ingredient_string(dias[0]["meals"][0]["ingredients"][0]) <= esperado_max


@pytest.mark.parametrize("linea", ["150 g de atún en agua", "250 g de filete de atún fresco", "200 g de salmón"])
def test_el_pescado_fresco_o_la_racion_normal_no_se_tocan(linea):
    dias = _dia(linea)
    assert ts.cap(dias, db=_DB()) == 0
    assert dias[0]["meals"][0]["ingredients"][0] == linea


def test_knob_cero_apaga(monkeypatch):
    monkeypatch.setenv("MEALFIT_SEED_NUT_LINE_CAP_G", "0")
    assert ts.cap(_dia("14.5 cdas de semillas de chía"), db=_DB()) == 0


def test_corre_en_el_bucle_de_topes_del_escudo():
    src = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    i = src.index("_pass_n += _cdwe(")
    j = src.index('__import__("topes_por_linea").cap(')
    assert 0 < j - i < 400, "va en el mismo bucle de punto fijo que los demás topes"


def _comida_queso(linea):
    return {"meal": "Merienda", "name": "Casabe con queso", "ingredients": [linea, "1 torta de casabe"],
            "ingredients_raw": [linea, "1 torta de casabe"], "recipe": ["Sirve el casabe con el queso."]}


@pytest.mark.parametrize("linea", ["30 g de queso blanco fresco", "¼ taza de queso cottage", "40 g de ricotta",
                                   "30 g de queso de hoja"])
def test_embarazo_el_queso_blando_se_calienta_hasta_que_humee(linea):
    import embarazo_seguro as es
    import graph_orchestrator as go
    comida = _comida_queso(linea)
    es.etiquetar({"days": [{"day": 1, "meals": [comida]}]}, {"medicalConditions": ["Embarazo"], "gender": "female"})
    assert any("humee" in p for p in comida["recipe"]), comida["recipe"]
    assert "humee" in go._meal_safety_notes_for_summary(comida), "la nota va donde el revisor la lee"


@pytest.mark.parametrize("form, linea", [
    ({"medicalConditions": ["Lactancia"], "gender": "female"}, "30 g de queso blanco fresco"),
    ({"medicalConditions": ["Embarazo"], "gender": "female"}, "20 g de queso parmesano"),
    ({"medicalConditions": ["Embarazo"], "gender": "female"}, "1 cda de queso crema"),
])
def test_lactancia_y_quesos_duros_no_llevan_la_nota(form, linea):
    import embarazo_seguro as es
    comida = _comida_queso(linea)
    es.etiquetar({"days": [{"day": 1, "meals": [comida]}]}, form)
    assert not any("humee" in p for p in comida["recipe"]), comida["recipe"]


@pytest.mark.parametrize("slot", ["almuerzo", "cena"])
def test_la_avena_del_empanizado_no_es_desayuno(slot):
    from constants import slot_violations_for_meal_name, SLOT_INAPPROPRIATE_FOODS
    nombre = "Pollo crujiente de avena al airfryer con ensalada de repollo"
    assert slot_violations_for_meal_name(nombre, slot, rules_table=SLOT_INAPPROPRIATE_FOODS) == []
    # …pero la avena de desayuno en la cena sigue marcada
    assert slot_violations_for_meal_name("Crujiente de avena con guineo", "cena", rules_table=SLOT_INAPPROPRIATE_FOODS)


def test_el_tope_de_pescado_tambien_al_entrar_al_revisor(monkeypatch):
    """rd19: «240 g de camarones y 285 g de tilapia (525 g)» llegó al revisor — los cerradores escalan el pescado DESPUÉS
    de la sustitución por condición. La etiqueta clínica de entrada al revisor vuelve a aplicar el tope."""
    import nutrition_db
    import etiquetas_clinicas as ec
    import embarazo_pescado as ep
    from tests.test_p1_plan_lote_187 import _DB as _DB187, _meal as _m187
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _DB187)
    plan = {"days": [
        {"day": 1, "meals": [_m187("Almuerzo", "Pollo guisado con arroz", ["150 g de pechuga de pollo"])]},
        {"day": 2, "meals": [_m187("Almuerzo", "Camarones al ajillo con arroz", ["240 g de camarones", "½ taza de arroz"])]},
        {"day": 3, "meals": [_m187("Almuerzo", "Tilapia al horno con batata", ["285 g de filete de tilapia", "1 batata"])]},
    ]}
    ec.etiquetar(plan, {"medicalConditions": ["Embarazo"], "gender": "female", "dietType": "balanced"})
    db = _DB187()
    total = sum(db.grams_from_ingredient_string(x) for d in plan["days"] for m in d["meals"]
                for x in m["ingredients"] if ep._PESCADO.search(x))
    assert total <= 340, [m["ingredients"] for d in plan["days"] for m in d["meals"]]


def test_y_como_ultima_palabra_en_el_bucle_de_topes():
    src = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    i = src.index('__import__("topes_por_linea").cap(')
    j = src.index('__import__("embarazo_pescado").limitar_pescado(_pd,')
    assert 0 < j - i < 400


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 193 and m.group(2) >= "2026-09-24"
