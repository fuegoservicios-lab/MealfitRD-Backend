# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-178 · 2026-09-23] Métrica de calidad sobre los 16 planes REALES de rd6-rd8 (sin gasto de IA): lo que
quedaba a la vista después del 177.

1. **22 ingredientes que dan nombre al plato seguían en migajas** («5 g de aguacate» en un «Revoltillo… con aguacate»,
   «20 g de mandarina», «½ cdta de mantequilla de maní»): la cola del guardado sólo los subía si el día quedaba ≤ 105 %.
   Subir el techo NO es el arreglo (medido: con 1,07/1,10 las migajas bajaban de 20 a 10 y 8 días salían de ±5 % de
   kcal). Los techos siguen en 1,05 (ahora knob); si no cabe el piso entero, sube lo que quepa si saca el plato de las
   migajas; y lo que siga en migajas se PAGA dentro del día con lo que ningún nombre menciona y es del mismo macro
   (grasa con grasa: el aceite), sin mover el total. Con el 177 dentro, rd8 ya sólo tenía 3 en 5 planes.
2. **DM2: el casabe en las CUATRO corridas reales** y cada plan con el aviso ámbar al usuario. Un casabe por bloque; los
   demás, pan integral (salvo alergia/rechazo). Y «harina de trigo» refinada → avena.
3. **Title Case con inserciones en minúscula**: «…Chinola y Yogurt natural Entero», «Bowl Fresco de Arroz integral…».
4. Pisos sin sentido: «jugo de limón» con piso de fruta (60 g); dátiles y granada van en puñado, no en ración."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import culinary_coherence as cc  # noqa: E402
import identidad_plato as idp  # noqa: E402

_CAT = [{"name": n, "aliases": a, "category": c, "prep_methods": ["ninguno"]} for n, a, c in (
    ("Aguacate", [], "Frutas"), ("Tomate", ["tomates"], "Vegetales"), ("Huevo", ["huevos"], "Proteínas"),
    ("Casabe", [], "Granos"), ("Queso blanco", ["queso"], "Lácteos"), ("Mandarina", [], "Frutas"))]
_IDX = cc.build_culinary_index(_CAT)


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


class _DB:
    _POR_G = (("aguacate", (1.60, 0.15)), ("huevo", (1.55, 0.11)), ("casabe", (3.5, 0.01)), ("pan integral", (2.5, 0.03)),
              ("mandarina", (0.53, 0.0)), ("aceite", (8.84, 1.0)), ("pollo", (1.65, 0.036)))
    _CATS = {"aguacate": "Frutas", "mandarina": "Frutas", "tomate": "Vegetales", "huevo": "Proteinas"}

    def macros_from_ingredient_string(self, s):
        m = re.match(r"\s*([\d.]+)\s*g\s+de\s+(.+)$", str(s).strip(), re.IGNORECASE)
        if not m:
            return None
        g, nombre = float(m.group(1)), idp._sa(m.group(2)).strip()
        kpg, fpg = next((v for k, v in self._POR_G if k in nombre), (1.0, 0.0))
        return {"name": nombre, "grams": g, "kcal": kpg * g, "protein": 0.1 * g, "carbs": 0.1 * g, "fats": fpg * g}

    def grams_from_ingredient_string(self, s):
        mac = self.macros_from_ingredient_string(s)
        return mac["grams"] if mac else None

    def category_of(self, s):
        return next((c for k, c in self._CATS.items() if k in idp._sa(s)), "")

    def lookup(self, s):
        return None

    def __getattr__(self, _n):
        return lambda *a, **k: None


def _revoltillo():
    return {"meal": "Desayuno", "name": "Revoltillo criollo con tomate y aguacate", "cals": 300, "fats": "20g",
            "ingredients": ["2 huevos", "½ tomate", "5 g de aguacate"],
            "ingredients_raw": ["2 huevos", "½ tomate", "5 g de aguacate"]}   # el tomate abre el nombre: él es el protagonista


# ─────────────── 1. la cola del guardado: techos y subida parcial ───────────────
def test_los_techos_de_la_cola_siguen_en_105_y_son_knob(monkeypatch):
    """Medido: con 1,07/1,10 las migajas bajaban de 20 a 10 pero 8 días salían de ±5 % de kcal. El techo no es el arreglo."""
    comidas = [{"cals": 2000, "fats": "60g"}]
    margen = idp._margen_del_dia(comidas, {"kcal": 2000.0, "grasa": 60.0})
    assert round(margen["kcal"]) == 100 and round(margen["grasa"], 1) == 3.0
    monkeypatch.setenv("MEALFIT_IDENTITY_TAIL_KCAL_CEIL", "1.08")
    assert round(idp._margen_del_dia(comidas, {"kcal": 2000.0, "grasa": 60.0})["kcal"]) == 160


def test_si_no_cabe_el_piso_sube_lo_que_quepa_si_sale_de_las_migajas():
    m = _revoltillo()
    margen = {"kcal": 100.0, "grasa": 2.5}                    # el piso entero (30 g) pediría 3,75 g de grasa
    assert idp.restaurar_meal(m, _IDX, db=_DB(), margen=margen) == ["↑5→21 g de Aguacate"]
    assert "21 g de Aguacate" in m["ingredients"] and "21 g de Aguacate" in m["ingredients_raw"]
    corto = _revoltillo()
    assert idp.restaurar_meal(corto, _IDX, db=_DB(), margen={"kcal": 100.0, "grasa": 0.5}) == [], \
        "si lo que cabe no llega a la mitad del piso, no se maquilla la migaja"
    assert "5 g de aguacate" in corto["ingredients"]


def test_sin_margen_la_migaja_se_paga_dentro_del_dia_sin_mover_el_total():
    rev = _revoltillo()
    almuerzo = {"meal": "Almuerzo", "name": "Pollo guisado con arroz", "cals": 1000, "protein": 60, "carbs": 90,
                "fats": 30, "ingredients": ["200 g de pollo", "10 g de aceite de oliva"],
                "ingredients_raw": ["200 g de pollo", "10 g de aceite de oliva"]}
    dia = [{"day": 1, "meals": [rev, almuerzo]}]
    objetivos = {"kcal": 1300 / 1.05, "grasa": 50 / 1.05}      # el día ya está en su techo: sin margen
    antes = sum(float(m["cals"]) for m in dia[0]["meals"])
    assert idp.restaurar_identidad(dia, db=_DB(), index=_IDX, objetivos=objetivos) >= 1
    g_aguacate = next(float(re.match(r"(\d+)", x).group(1)) for x in rev["ingredients"] if "guacate" in x)
    assert g_aguacate >= 15, rev["ingredients"]
    assert not any(x == "10 g de aceite de oliva" for x in almuerzo["ingredients"]), "lo paga el aceite sin nombre"
    despues = sum(float(m["cals"]) for m in dia[0]["meals"])
    assert despues <= antes + 1, (antes, despues)


def test_pisos_con_sentido():
    db = _DB()
    assert idp._piso_de("jugo de limón", db) == 0, "el jugo de limón aliña, no es ración de fruta"
    assert idp._piso_de("dátiles", db) == 20
    assert idp._piso_de("granada", db) == 30
    assert idp._piso_de("aguacate", db) == 30, "de acompañante, en láminas"


# ─────────────── 2. DM2: un casabe por bloque ───────────────
def _plan_dm2():
    return {"days": [
        {"meals": [{"meal": "Merienda", "name": "Casabe con queso blanco y mandarina",
                    "ingredients": ["½ casabe (15 g)", "20 g de queso blanco"],
                    "ingredients_raw": ["15 g de Casabe", "20 g de queso blanco"], "recipe": ["Tuesta el casabe."]}]},
        {"meals": [{"meal": "Merienda", "name": "Casabe crujiente con queso blanco y mandarina",
                    "desc": "Casabe crujiente con queso blanco fresco.",
                    "ingredients": ["1 torta pequeña de casabe", "20 g de queso blanco"],
                    "ingredients_raw": ["20 g de Casabe", "20 g de queso blanco"], "recipe": ["Tuesta el casabe."]}]}]}


def test_dm2_deja_el_primer_casabe_y_cambia_los_demas():
    import dm2_seguro as dm
    plan = _plan_dm2()
    assert dm.limitar_casabe(plan, {"medicalConditions": ["Diabetes tipo 2"]}, db=_DB()) == 1
    primero, segundo = plan["days"][0]["meals"][0], plan["days"][1]["meals"][0]
    assert "½ casabe (15 g)" in primero["ingredients"], "el primero se queda"
    assert "1 rebanada de pan integral" in segundo["ingredients"]
    assert "1 rebanada de pan integral" in segundo["ingredients_raw"]
    assert not any("casabe" in x.lower() for x in segundo["ingredients"] + segundo["ingredients_raw"])
    assert segundo["name"].startswith("Pan integral"), segundo["name"]
    assert "casabe" not in segundo["desc"].lower()
    assert dm.limitar_casabe(plan, {"medicalConditions": ["Diabetes tipo 2"]}, db=_DB()) == 0, "idempotente"


def test_dm2_con_gluten_declarado_o_sin_diabetes_el_casabe_se_queda():
    import dm2_seguro as dm
    celiaco = {"medicalConditions": ["Diabetes tipo 2"], "allergies": ["Gluten"]}
    assert dm.limitar_casabe(_plan_dm2(), celiaco, db=_DB()) == 0
    assert dm.limitar_casabe(_plan_dm2(), {"medicalConditions": ["Ninguna"]}, db=_DB()) == 0


def test_el_tope_de_casabe_esta_cableado_en_la_sustitucion_clinica():
    src = _src("graph_orchestrator.py")
    i = src.index("def _apply_condition_substitutions(")
    assert '__import__("dm2_seguro").limitar_casabe(plan, form_data)' in src[i:i + 2600]


def test_dm2_harina_de_trigo_pasa_a_avena():
    import graph_orchestrator as go
    plan = {"days": [{"meals": [{"meal": "Desayuno", "name": "Panqueques de avena y canela",
                                 "ingredients": ["30 g de avena", "15 g de harina de trigo"],
                                 "ingredients_raw": ["30 g de avena", "15 g de harina de trigo"]}]}]}
    assert go._apply_condition_substitutions(plan, {"medicalConditions": ["Diabetes tipo 2"]}) >= 1
    assert not any("harina de trigo" in i for i in plan["days"][0]["meals"][0]["ingredients"])


# ─────────────── 3. Title Case ───────────────
def test_en_title_case_lo_que_entro_en_minuscula_sube():
    import graph_orchestrator as go
    fix = go._fix_name_gender_agreement
    assert fix("Casabe con Queso Blanco Fresco, Chinola y Yogurt natural Entero") == \
        "Casabe con Queso Blanco Fresco, Chinola y Yogurt Natural Entero"
    assert fix("Bowl Fresco de Arroz integral y Habichuelas con Pollo a la Plancha, Alcachofa y Aguacate") == \
        "Bowl Fresco de Arroz Integral y Habichuelas con Pollo a la Plancha, Alcachofa y Aguacate"
    assert fix("Pechuga guisada criolla con puré suave de harina de Negrito") is None, "frase normal: no se toca"


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src("app.py"))
    assert m and int(m.group(1)) >= 178 and m.group(2) >= "2026-09-23"
