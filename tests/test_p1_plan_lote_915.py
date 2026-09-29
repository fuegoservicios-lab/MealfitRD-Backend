# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-915 · 2026-09-29] Lo que casi no tiene grasa sube a su piso aunque el día ya esté en su techo de grasa.

Leídas enteras las baterías reales del 28/29-sep: «Nabo crujiente…» con 30 g de nabo, «Yuca guisada…» con 60 g de yuca.
Los pisos existen (lotes 174-179) y el día tenía 72-98 kcal de sitio; no subían porque el día estaba una décima sobre su
techo de GRASA (margen −0,15 a −1,15 g) y `_subir_linea` no sube nada sin margen de grasa, ni 70 g de nabo que traen
0,07 g. El lote 584 ya lo había resuelto para lo que FALTA en la lista («un día ya pasado de grasa no le cierra la puerta
a una yautía con 0,1 g: lo inapreciable no cuenta»); faltaba en lo que está PRESENTE por debajo de su piso.

La primera versión de este lote (pagar la subida con donantes del día) se midió sobre 566 planes y se descartó: los
donantes eran verduras. Esta no mueve nada más que la línea que sube.
"""
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
    ("Avena", [], "Granos"), ("Nabo", [], "Vegetales"), ("Yuca", [], "Víveres"), ("Pollo", ["pechuga de pollo"], "Proteínas"),
    ("Queso blanco", ["queso"], "Lácteos"), ("Guineo", ["guineos"], "Frutas"), ("Lechosa", [], "Frutas"),
    ("Atún en agua", ["atun"], "Proteínas"), ("Mango", [], "Frutas"))]
_IDX = cc.build_culinary_index(_CAT)


class _DB:
    """(kcal, grasa, proteína) por gramo."""
    _POR_G = (("avena", (3.8, 0.069, 0.13)), ("nabo", (0.28, 0.001, 0.009)), ("yuca", (1.6, 0.003, 0.014)),
              ("pollo", (1.65, 0.036, 0.31)), ("queso", (2.6, 0.2, 0.18)), ("guineo", (0.9, 0.003, 0.011)),
              ("lechosa", (0.43, 0.003, 0.005)), ("atun", (0.85, 0.009, 0.19)), ("mango", (0.6, 0.004, 0.008)))
    _CATS = {"yuca": "Viveres", "nabo": "Vegetales", "guineo": "Frutas", "pollo": "Proteinas", "queso": "Lacteos",
             "avena": "Despensa", "lechosa": "Frutas", "atun": "Proteinas", "mango": "Frutas"}

    def macros_from_ingredient_string(self, s):
        m = re.match(r"\s*([\d.]+)\s*g\s+de\s+(.+)$", str(s).strip(), re.IGNORECASE)
        if not m:
            return None
        g, nombre = float(m.group(1)), idp._sa(m.group(2)).strip()
        k, f, p = next((v for n, v in self._POR_G if n in nombre), (1.0, 0.0, 0.0))
        return {"name": nombre, "grams": g, "kcal": k * g, "protein": p * g, "carbs": 0.1 * g, "fats": f * g}

    def grams_from_ingredient_string(self, s):
        mac = self.macros_from_ingredient_string(s)
        return mac["grams"] if mac else None

    def category_of(self, s):
        return next((c for k, c in self._CATS.items() if k in idp._sa(s)), "")

    def lookup(self, s):
        return None

    def __getattr__(self, _n):
        return lambda *a, **k: None


def _plato(nombre, linea):
    return {"meal": "Cena", "name": nombre, "cals": 300, "protein": 15, "carbs": 30, "fats": 12,
            "ingredients": [linea, "40 g de queso blanco"], "ingredients_raw": [linea, "40 g de queso blanco"]}


def _margen(kcal=90.0, grasa=-0.6, proteina=float("inf"), condiciones=()):
    return {"kcal": kcal, "grasa": grasa, "proteina": proteina, "condiciones": list(condiciones)}


def _subir(meal, margen):
    return idp._subir_identidad_del_modelo(meal, _IDX, db=_DB(), allergies=[], margen=margen)


def test_el_nabo_sube_a_su_racion_con_el_dia_una_decima_sobre_su_grasa():
    m = _plato("Nabo crujiente al horno con queso blanco", "30 g de nabo")
    margen = _margen()
    assert _subir(m, margen) == ["↑30→100 g de Nabo"], m["ingredients"]
    assert m["ingredients"][0] == "100 g de Nabo" and m["ingredients_raw"][0] == "100 g de Nabo"
    assert 69 < margen["kcal"] < 71 and -0.7 < margen["grasa"] < -0.6, "el margen se gasta igual"


def test_lo_que_sube_por_lo_inapreciable_suma_su_delta_y_no_re_mide_el_plato():
    """El nivelado de kcal ajusta los NÚMEROS del plato sin tocar sus líneas: re-medirlo desde ellas mueve el día (lote
    178). Medido sobre 92 planes: con la re-medición, 3 de 43 días tocados acababan sobre el 105 % de sus kcal."""
    m = _plato("Nabo crujiente al horno con queso blanco", "30 g de nabo")      # 300 kcal guardadas; sus líneas dan 112
    assert _subir(m, _margen()) == ["↑30→100 g de Nabo"]
    assert (m["cals"], m["fats"], m["protein"]) == (320, 12, 16), (m["cals"], m["fats"], m["protein"])
    assert m["macros"] == ["P:16g", "C:37g", "G:12g"]
    m = _plato("Avena cremosa con queso blanco", "15 g de avena")               # con sitio de grasa: la subida de siempre
    assert _subir(m, _margen(kcal=90.0, grasa=5.0)) == ["↑15→30 g de Avena"]
    assert m["cals"] != 300 + 57, "re-medido desde sus líneas, como antes"


def test_una_subida_parcial_que_no_se_nota_no_se_hace():
    m = _plato("Yuca guisada con queso blanco", "85 g de yuca")
    assert _subir(m, _margen(kcal=10.0)) == [] and m["ingredients"][0] == "85 g de yuca", "6 g de 15 que faltan"
    m = _plato("Yuca guisada con queso blanco", "85 g de yuca")
    assert _subir(m, _margen(kcal=30.0)) == ["↑85→100 g de Yuca"], "cabe entera"


def test_la_yuca_tambien():
    m = _plato("Yuca guisada con queso blanco", "60 g de yuca")
    assert _subir(m, _margen()) == ["↑60→100 g de Yuca"], m["ingredients"]


def test_lo_que_trae_grasa_de_verdad_sigue_esperando_su_sitio():
    """15 g más de avena son 1 g de grasa: con el día sobre su techo, no sube (el techo de grasa se midió en el lote 46)."""
    m = _plato("Avena cremosa con queso blanco", "15 g de avena")
    assert _subir(m, _margen()) == [] and m["ingredients"][0] == "15 g de avena"
    m = _plato("Queso blanco a la plancha con nabo", "10 g de queso blanco")
    m["ingredients"] = ["10 g de queso blanco", "100 g de nabo"]
    m["ingredients_raw"] = list(m["ingredients"])
    assert _subir(m, _margen()) == [] and m["ingredients"][0] == "10 g de queso blanco"


def test_sin_kcal_no_sube_aunque_no_traiga_grasa():
    """Ni una subida de nada: «60→66 g de yuca» no le cambia el plato a nadie (la subida parcial es para migajas)."""
    m = _plato("Yuca guisada con queso blanco", "60 g de yuca")
    assert _subir(m, _margen(kcal=10.0)) == [] and m["ingredients"][0] == "60 g de yuca"
    m = _plato("Yuca guisada con queso blanco", "60 g de yuca")
    assert _subir(m, _margen(kcal=0.0)) == [] and m["ingredients"][0] == "60 g de yuca"


def test_con_sitio_de_kcal_a_medias_sube_lo_que_cabe():
    """La subida parcial del 178 sigue igual: lo que quepa de kcal, si con eso sale de las migajas."""
    m = _plato("Yuca guisada con queso blanco", "20 g de yuca")
    assert _subir(m, _margen(kcal=80.0)) == ["↑20→70 g de Yuca"], m["ingredients"]


def test_con_grasa_de_sobra_todo_como_antes():
    m = _plato("Avena cremosa con queso blanco", "15 g de avena")
    assert _subir(m, _margen(kcal=90.0, grasa=5.0)) == ["↑15→30 g de Avena"]


def test_el_techo_renal_de_proteina_tampoco_bloquea_lo_inapreciable():
    m = _plato("Nabo crujiente al horno con queso blanco", "30 g de nabo")
    assert _subir(m, _margen(proteina=0.2)) == ["↑30→52 g de Nabo"], "70 g traen 0,63 g de proteína: sube lo que cabe"
    m = _plato("Nabo crujiente al horno con queso blanco", "60 g de nabo")
    assert _subir(m, _margen(proteina=0.2)) == ["↑60→100 g de Nabo"], "40 g traen 0,36 g"


# ─────────────── la subida no deshace un tope clínico ───────────────
# Corpus del VPS (566 planes): la cola de identidad llevaba «↑50→100 g de Guineo» a un plan de cirugía bariátrica, cuyo
# tope de fruta de alto índice glucémico es de 50 g y corre ANTES que ella.
_BARIATRICA = ("Cirugía Bariátrica", "SOP (PCOS)")


def test_en_bariatrica_la_fruta_no_pasa_de_su_tope():
    m = _plato("Rodajas de guineo con queso blanco", "50 g de guineo")
    assert _subir(m, _margen(grasa=5.0, condiciones=_BARIATRICA)) == [], "alto índice glucémico: 50 g"
    m = _plato("Lechosa fresca con queso blanco", "64 g de lechosa")
    assert _subir(m, _margen(grasa=5.0, condiciones=_BARIATRICA)) == ["↑64→80 g de Lechosa"], "fruta: 80 g, no 100"
    m = _plato("Lechosa fresca con queso blanco", "64 g de lechosa")
    assert _subir(m, _margen(grasa=5.0)) == ["↑64→100 g de Lechosa"], "sin condición, su ración"


def test_en_diabetes_el_vivere_y_la_fruta_dulce_tienen_su_tope(monkeypatch):
    m = _plato("Yuca guisada con queso blanco", "60 g de yuca")
    assert _subir(m, _margen(condiciones=["Diabetes T2"])) == ["↑60→100 g de Yuca"], "el tope (100) es su ración"
    monkeypatch.setenv("MEALFIT_DM2_HIGH_GI_CAP_G", "80")
    import importlib
    import tope_clinico as tc
    assert tc.tope_de_linea("60 g de yuca", ["Diabetes T2"]) in (80.0, 100.0)
    assert tc.tope_de_linea("60 g de mango", ["Diabetes T2"]) == 120.0
    assert tc.tope_de_linea("60 g de guineo verde", ["Diabetes T2"]) is None, "el guineo verde no es fruta dulce"
    assert tc.tope_de_linea("60 g de nabo", ["Diabetes T2"]) is None


def test_en_embarazo_y_lactancia_el_pescado_no_sube():
    for cond in (["Embarazo"], ["Lactancia"]):
        m = _plato("Tostadas con queso blanco y atún en agua", "40 g de atún en agua")
        m["ingredients"] = ["40 g de queso blanco", "40 g de atún en agua"]
        m["ingredients_raw"] = list(m["ingredients"])
        assert _subir(m, _margen(grasa=5.0, condiciones=cond)) == [], cond
    m = _plato("Nabo crujiente al horno con queso blanco", "30 g de nabo")
    assert _subir(m, _margen(condiciones=["Embarazo"])) == ["↑30→100 g de Nabo"], "la verdura sí"


def test_con_enfermedad_renal_o_sin_saber_las_condiciones_lo_inapreciable_no_abre_nada():
    m = _plato("Nabo crujiente al horno con queso blanco", "30 g de nabo")
    assert _subir(m, _margen(condiciones=["Enfermedad Renal"])) == []
    m = _plato("Nabo crujiente al horno con queso blanco", "30 g de nabo")
    margen = _margen()
    margen["condiciones"] = None
    assert _subir(m, margen) == []
    m = _plato("Nabo crujiente al horno con queso blanco", "30 g de nabo")
    assert _subir(m, {"kcal": 90.0, "grasa": -0.6}) == [], "el margen de antes, sin la clave"


def test_las_condiciones_salen_de_la_politica_del_plan():
    plan = {"calories": 1800, "macros": {"fats": "60g"},
            "_plan_policy": {"effective": {"clinical": {"conditions": ["Cirugía Bariátrica"]}}}}
    assert idp.objetivos_de(plan)["condiciones"] == ["Cirugía Bariátrica"]
    plan["_plan_policy"]["effective"]["clinical"]["conditions"] = []
    assert idp.objetivos_de(plan)["condiciones"] == []
    assert "condiciones" not in idp.objetivos_de({"calories": 1800, "macros": {"fats": "60g"}}), "no se sabe"
    comidas = [{"cals": 1700, "fats": 60, "protein": 100}]
    assert idp._margen_del_dia(comidas, idp.objetivos_de(plan))["condiciones"] == []


def test_los_otros_tres_caminos_de_la_cola_tampoco_pasan_el_tope():
    # el «0 g de …» que vuelve
    m = _plato("Rodajas de guineo con queso blanco", "0 g de guineo")
    assert _subir(m, _margen(grasa=5.0, condiciones=_BARIATRICA)) == ["↑0→50 g de guineo"], m["ingredients"]
    m = _plato("Rodajas de guineo con queso blanco", "0 g de guineo")
    assert _subir(m, _margen(grasa=5.0)) == ["↑0→60 g de guineo"]
    # lo que el nombre promete, los pasos usan y la lista no trae
    falta = {"meal": "Merienda", "name": "Queso blanco con guineo", "cals": 200, "protein": 10, "carbs": 10, "fats": 10,
             "ingredients": ["40 g de queso blanco"], "ingredients_raw": ["40 g de queso blanco"],
             "recipe": ["Mise en place: pela el guineo y córtalo en rodajas.", "Montaje: sirve el queso con el guineo."]}
    import copy
    m = copy.deepcopy(falta)
    assert _subir(m, _margen(grasa=5.0, condiciones=_BARIATRICA)) == ["+50 g de Guineo"], m["ingredients"]
    m = copy.deepcopy(falta)
    assert _subir(m, _margen(grasa=5.0)) == ["+60 g de Guineo"]
    # la migaja que se paga dentro del día
    def _dia():
        return [{"meal": "Merienda", "name": "Rodajas de guineo con queso blanco", "cals": 120, "protein": 8, "carbs": 5,
                 "fats": 8, "ingredients": ["10 g de guineo", "40 g de queso blanco"],
                 "ingredients_raw": ["10 g de guineo", "40 g de queso blanco"]},
                {"meal": "Almuerzo", "name": "Pollo guisado", "cals": 900, "protein": 60, "carbs": 80, "fats": 30,
                 "ingredients": ["200 g de pollo", "300 g de yuca"], "ingredients_raw": ["200 g de pollo", "300 g de yuca"]}]
    obj = {"kcal": 1000.0, "grasa": 38.0, "condiciones": list(_BARIATRICA)}
    dia = _dia()
    assert idp.compensar_dia(dia, _IDX, _DB(), [], objetivos=obj) == 1
    assert dia[0]["ingredients"][0] == "50 g de Guineo", dia[0]["ingredients"]
    dia = _dia()
    assert idp.compensar_dia(dia, _IDX, _DB(), [], objetivos={"kcal": 1000.0, "grasa": 38.0, "condiciones": []}) == 1
    assert dia[0]["ingredients"][0] == "60 g de Guineo", dia[0]["ingredients"]


def test_con_el_knob_del_tope_apagado_la_subida_de_siempre(monkeypatch):
    monkeypatch.setenv("MEALFIT_IDENTITY_RAISE_CLINICAL_CAP", "false")
    m = _plato("Lechosa fresca con queso blanco", "64 g de lechosa")
    assert _subir(m, _margen(grasa=5.0, condiciones=_BARIATRICA)) == ["↑64→100 g de Lechosa"]


def test_con_el_knob_apagado_todo_como_antes(monkeypatch):
    monkeypatch.setenv("MEALFIT_IDENTITY_RAISE_TRACE_FAT", "false")
    m = _plato("Nabo crujiente al horno con queso blanco", "30 g de nabo")
    assert _subir(m, _margen()) == [] and m["ingredients"][0] == "30 g de nabo"


def test_ancla():
    src = (_BACKEND / "identidad_plato.py").read_text(encoding="utf-8")
    cuerpo = src[src.index("def _subir_linea("):src.index("# ─────────────── [P1-PLAN-LOTE-174")]
    assert "tooltip-anchor: P1-PLAN-LOTE-915" in cuerpo and "MEALFIT_IDENTITY_RAISE_TRACE_FAT" in src
    b = (_BACKEND / "identidad_plato.py").read_bytes()
    assert b"\x08" not in b and b"\r" not in b
