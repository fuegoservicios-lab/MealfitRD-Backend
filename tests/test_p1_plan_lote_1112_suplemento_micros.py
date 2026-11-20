"""[P1-PLAN-LOTE-1112 · 2026-11-19] La toma del Multi+ Phyto del dueño quedó en el diario con `nutrition = null`
aunque su pote tenía la tabla de internet: sus vitaminas no llegaban a «Micros de hoy». Ahora la toma de un pote con
tabla legible por porción guarda la etiqueta × porciones, con las reglas estrictas del contador."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

from food_label_nutrients import micros_from_snapshot, snapshot  # noqa: E402
from diary_micros import micros_plausibles  # noqa: E402
from suplemento_micros import etiqueta_del_pote  # noqa: E402


def _fila(name, amount, unit, basis="declared", form=""):
    return {"name": name, "amount": amount, "unit": unit, "basis": basis, "form": form,
            "label_text": f"{name} {amount}{unit}"}


# filas tal como las guardó la búsqueda web en el pote real (id 51951), más los casos que deben quedarse fuera
POTE = {"kcal": 0, "protein_g": 0, "carbs_g": 0, "fats_g": 0, "gramos_porcion": 0, "supplement_facts": {
    "serving_text": "4 Capsules", "units_per_serving": 4, "serving_unit": "capsule", "amount_basis": "per_serving",
    "table_readable": True, "nutrients": [
        _fila("Vitamin A", 1500, "mcg RAE"), _fila("Vitamin C", 400, "mg"), _fila("Vitamin D", 50, "mcg"),
        _fila("Vitamin K", 450, "mcg"), _fila("Vitamin B-6", 6.7, "mg"), _fila("Vitamin B-12", 200, "mcg"),
        _fila("Folate", 680, "mcg"),                                   # sin «DFE»: no se convierte
        _fila("Calcium", 50, "mg"), _fila("Magnesium", 150, "mg"), _fila("Zinc", 15, "mg"),
        _fila("Selenium", 100, "mcg"), _fila("Niacin", None, "mg NE"),
        _fila("Vitamin E Isomers", 25, "mg", form="delta and gamma tocotrienols"),  # no es alfa-tocoferol
        _fila("Iron", 18, "mg", basis="compound"),                     # masa de la sal, no hierro elemental
        _fila("Vitamin D3", 1000, "IU"),                               # UI: no se convierte
    ]}}


def test_la_tabla_del_pote_pasa_al_contador_con_sus_unidades():
    e = etiqueta_del_pote(POTE)
    got = {n["key"]: (n["amount"], n["unit"]) for n in e["nutrients"]}
    assert got == {"vit_a_mcg": (1500, "mcg RAE"), "vit_c_mg": (400, "mg"), "vit_d_mcg": (50, "mcg"),
                   "vit_k_mcg": (450, "mcg"), "vit_b6_mg": (6.7, "mg"), "b12_mcg": (200, "mcg"),
                   "calcium_mg": (50, "mg"), "magnesium_mg": (150, "mg"), "zinc_mg": (15, "mg"),
                   "selenium_mcg": (100, "mcg")}
    assert e["serving_text"] == "4 Capsules"


def test_las_porciones_multiplican_y_el_dia_las_suma():
    m = micros_from_snapshot(snapshot(etiqueta_del_pote(POTE), 0.5))        # 2 de 4 cápsulas
    assert m["values"]["vit_c_mg"] == 200 and m["values"]["vit_d_mcg"] == 25
    assert m["values"]["iron_mg"] is None and m["values"]["folate_mcg"] is None, "sin dato, no 0"
    assert micros_plausibles(m, 0) is m, "0 kcal no descarta las vitaminas"


def test_sin_tabla_legible_o_sin_base_por_porcion_no_hay_etiqueta():
    import copy
    for campo, valor in (("table_readable", False), ("amount_basis", "unknown")):
        p = copy.deepcopy(POTE)
        p["supplement_facts"][campo] = valor
        assert etiqueta_del_pote(p) is None
    assert etiqueta_del_pote({"kcal": 0, "unidades_por_toma": 6}) is None          # creatina sin tabla
    assert etiqueta_del_pote(None) is None and etiqueta_del_pote("no json") is None
    import json
    assert etiqueta_del_pote(json.dumps(POTE))["nutrients"]                         # jsonb leído como texto


def test_cableado_en_log_consumed_meal():
    src = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    i = src.index('food_facts = __import__("suplemento_micros").etiqueta_del_pote(_pote.get("serving_label"))')
    j = src.index("_nutrition = snapshot(food_facts, label_servings) if food_facts is not None else nutrition", i)
    assert j - i < 500 and "label_servings = (porciones or 1) if food_facts else label_servings" in src[i:j]
    assert src.index("def log_consumed_meal(") < i < src.index("def correct_consumed_meal(")
