# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-177 · 2026-09-23] Sexta vuelta de la batería REAL del generador (RD), con el 175 y el 176 dentro.

1. **El reequilibrio de macros del día dejaba en migajas justo el alimento que da nombre al plato.** Familia de 4:
   «Pastelitos ligeros de mapuey y queso blanco…» salió con 5 g de queso (tenía 31) y el ½ aguacate que el nombre no
   menciona, entero; el día siguió 24 % sobre su grasa. `_rebalance_day_macros_to_target` escala con el MISMO factor todas
   las fuentes y en tres pasadas: las líneas en unidades se resisten al redondeo y las líneas en gramos se lo comen
   entero. Los recortes de grasa y carbohidratos ya respetaban la identidad (lote 46); el reequilibrio no. Ahora puede
   bajarla, pero no por debajo de su piso (el de `identidad_plato`, lote 174).
2. **«…con tomate y cebolla, aguacate fresca»**: tras una coma también empieza un núcleo (la concordancia sólo miraba
   «y»/«con»/«e»).
3. **El pase de honestidad de descripciones elegía el sustituto en el orden de un `set`** (no determinista entre
   procesos): con huevo y atún presentes, «pollo» podía volverse «huevo». Ahora es determinista y prefiere lo que no es
   huevo cuando lo que falta no es huevo."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


_ROWS = [
    {"name": "Queso blanco", "aliases": ["queso blanco", "queso"], "kcal_per_100g": 298,
     "protein_g_per_100g": 18.1, "carbs_g_per_100g": 3, "fats_g_per_100g": 24},
    {"name": "Aceite de oliva", "aliases": ["aceite de oliva", "aceite"], "kcal_per_100g": 884,
     "protein_g_per_100g": 0, "carbs_g_per_100g": 0, "fats_g_per_100g": 100},
]


def _db():
    from nutrition_db import IngredientNutritionDB
    return IngredientNutritionDB(rows=_ROWS)


def _gramos(linea):
    m = re.match(r"\s*(\d+(?:[.,]\d+)?)\s*g\b", str(linea))
    return float(m.group(1).replace(",", ".")) if m else None


def test_el_reequilibrio_no_deja_el_queso_del_nombre_en_migajas():
    import graph_orchestrator as go
    meal = {"meal": "Cena", "name": "Pastelitos de mapuey y queso blanco",
            "ingredients": ["30 g de queso blanco", "10 g de aceite de oliva"],
            "protein": 5, "carbs": 1, "fats": 17, "cals": 177}
    assert go._rebalance_day_macros_to_target([meal], 1, 5, _db())
    queso = next(i for i in meal["ingredients"] if "queso" in i)
    aceite = next(i for i in meal["ingredients"] if "aceite" in i)
    assert _gramos(queso) is not None and _gramos(queso) >= 18, f"el queso que nombra el plato bajó a {queso}"
    assert _gramos(aceite) is not None and _gramos(aceite) < 10, "la grasa baja por las demás fuentes"


def test_sin_nombre_el_reequilibrio_sigue_igual():
    import graph_orchestrator as go
    meal = {"meal": "Cena", "name": "Pastelitos de mapuey",
            "ingredients": ["30 g de queso blanco", "10 g de aceite de oliva"],
            "protein": 5, "carbs": 1, "fats": 17, "cals": 177}
    go._rebalance_day_macros_to_target([meal], 1, 5, _db())
    queso = next(i for i in meal["ingredients"] if "queso" in i)
    assert _gramos(queso) is not None and _gramos(queso) < 18, "sin identidad, el queso baja como antes"


def test_no_bajo_del_piso():
    import identidad_plato as ip
    db = _db()
    linea, k = ip.no_bajo_del_piso("30 g de queso blanco", "5 g de queso blanco", 5 / 30, db)
    assert _gramos(linea) == 20 and abs(k - 20 / 30) < 0.05
    assert ip.no_bajo_del_piso("15 g de queso blanco", "5 g de queso blanco", 1 / 3, db) == ("15 g de queso blanco", 1.0)
    assert ip.no_bajo_del_piso("30 g de queso blanco", "24 g de queso blanco", 0.8, db) == ("24 g de queso blanco", 0.8)


def test_el_reequilibrio_esta_cableado():
    src = _src("graph_orchestrator.py")
    assert 'if factor < 1 and _identidad_protege(m, orig):' in src
    assert '__import__("identidad_plato").no_bajo_del_piso(orig, quant, _k, db)' in src


def test_concordancia_tras_coma():
    import graph_orchestrator as go
    assert go._fix_name_gender_agreement("Revoltillo criollo con tomate y cebolla, aguacate fresca") == \
        "Revoltillo criollo con tomate y cebolla, aguacate fresco"
    assert go._fix_name_gender_agreement("Yogur con fresas, lechosa fresco y maní") == "Yogur con fresas, lechosa fresca y maní"


def test_el_sustituto_de_la_descripcion_es_determinista_y_prefiere_no_huevo():
    import graph_orchestrator as go
    meal = {"desc": "Bowl de pollo con vegetales.", "ingredients": ["2 huevos", "80 g de atún en agua"]}
    go._desc_food_honesty_pass([{"meals": [meal]}])
    assert meal["desc"] == "Bowl de atún con vegetales.", meal["desc"]


def test_las_demandas_de_cadena_de_suministro_son_advisory():
    """Batería real (lactancia): dos rechazos CRÍTICOS por cosas que un menú no puede afirmar."""
    import graph_orchestrator as go
    casabe = ("El casabe no indica que sea de procesamiento verificado. Durante la lactancia, debe especificarse casabe "
              "de marca regulada o procesamiento seguro por el riesgo de glucósidos cianogénicos.")
    queso = ("El queso blanco fresco pasteurizado de los días 2 y 3 no especifica que sea de envasado industrial y se "
             "mantenga refrigerado.")
    ok, reales, sev, adv = go._downgrade_reviewer_verification_demands(False, [casabe, queso], "critical")
    assert ok is True and reales == [] and len(adv) == 2
    # «pasteurizado» sigue FUERA a propósito: es la regla de seguridad del embarazo, no una demanda de marca
    ok2, reales2, _s, _a = go._downgrade_reviewer_verification_demands(
        False, ["El queso blanco fresco no está identificado como pasteurizado."], "critical")
    assert ok2 is False and len(reales2) == 1


def test_lactancia_yautia_bien_cocida():
    import embarazo_seguro as e
    plan = {"days": [{"meals": [{"name": "Yautía majada", "ingredients": ["150 g de yautía"],
                                 "recipe": ["Hierve la yautía."]}]}]}
    assert e.etiquetar(plan, {"medicalConditions": ["Lactancia"]}) == 1
    assert any("completamente blanda" in p for p in plan["days"][0]["meals"][0]["recipe"])
    assert e.etiquetar(plan, {"medicalConditions": ["Lactancia"]}) == 0, "idempotente"
    harina = {"days": [{"meals": [{"name": "Arepitas", "ingredients": ["30 g de harina de yautía"], "recipe": ["Mezcla."]}]}]}
    assert e.etiquetar(harina, {"medicalConditions": ["Lactancia"]}) == 0


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src("app.py"))
    assert m and int(m.group(1)) >= 177 and m.group(2) >= "2026-09-23"
