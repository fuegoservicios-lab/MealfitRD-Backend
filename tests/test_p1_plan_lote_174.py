# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-174 · 2026-09-23] Tercera vuelta de la batería real (RD): los días que salían CORTOS.

1. **Migajas en platos del modelo** (DM2 + insulina, día 3): «Casabe crujiente con queso blanco fresco, maní y huevo» con
   0,01 g de maní, «Mandarina con almendras…» con 0 g de almendras, «…sardinas en lata» con 1,97 g: el día entregó
   1.450 kcal y 110 g de proteína (meta 1.750/153). La identidad de los platos de BIBLIOTECA ya subía «lo presente pero
   pobre» al piso si el día tenía sitio (lote 49); los del MODELO no tienen plantilla y quedaban fuera. Ahora el piso sale
   del tipo de alimento, la identidad del nombre, y la proteína va PRIMERO (réplica del día 3: 1.679 kcal / 130 g).
2. **«mango maduro» con los macros del plátano**: el alias «maduro» de «Plátano maduro» ganaba por largo a «mango»
   (137 vs 67 kcal por 100 g); también «guineo maduro»."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


class _Info:
    def __init__(self, name, kcal, protein, carbs, fats):
        self.name, self.kcal, self.protein, self.carbs, self.fats = name, kcal, protein, carbs, fats


_CAT = {"mani": ("Maní", 567, 26, 16, 49, "Despensa"), "almendras": ("Almendras", 579, 21, 22, 50, "Despensa"),
        "sardinas": ("Sardinas en lata", 208, 25, 0, 11, "Proteínas"), "mango": ("Mango", 60, 0.8, 15, 0.4, "Frutas"),
        "cilantro": ("Cilantro", 23, 2, 4, 0.5, "Vegetales"), "casabe": ("Casabe", 330, 1, 80, 0.3, "Víveres")}


class _DB:
    def _row(self, s):
        n = re.sub(r"[^a-z ]", "", s.lower().replace("í", "i"))
        return next((v for k, v in _CAT.items() if k in n), None)

    def lookup(self, s):
        r = self._row(s)
        return _Info(*r[:5]) if r else None

    def category_of(self, s):
        r = self._row(s)
        return r[5] if r else None

    def grams_from_ingredient_string(self, s):
        m = re.match(r"\s*(\d+(?:\.\d+)?)\s*g", s)
        return float(m.group(1)) if m else None

    def macros_from_ingredient_string(self, s):
        g, r = self.grams_from_ingredient_string(s), self._row(s)
        if not (g is not None and r):
            return None
        f = g / 100.0
        return {"kcal": r[1] * f, "protein": r[2] * f, "carbs": r[3] * f, "fats": r[4] * f}


def _index():
    return {"__db__": True}


def test_piso_por_tipo_y_hierbas_sin_piso():
    import identidad_plato as ip
    db = _DB()
    assert ip._piso_de("Maní", db) == 10
    assert ip._piso_de("Sardinas en lata", db) == 60
    assert ip._piso_de("Cilantro", db) == 0, "una hierba nombra el plato («al cilantro») pero no pide ración"
    assert ip._piso_de("Mango", db) == 60


def test_la_linea_en_cero_nombrada_vuelve_al_piso_si_cabe():
    import identidad_plato as ip
    db = _DB()
    meal = {"name": "Mandarina con almendras", "ingredients": ["1 mandarina", "0 g de almendras tostadas sin sal"],
            "ingredients_raw": ["1 mandarina", "0 g de almendras tostadas sin sal"]}
    margen = {"kcal": 400.0, "grasa": 20.0}
    out = ip._rescatar_cero(meal, "almendras tostadas sin sal", db, margen, [])
    assert out and "10 g de almendras" in meal["ingredients"][1] and "10 g de almendras" in meal["ingredients_raw"][1]
    assert margen["kcal"] < 400.0
    sin_sitio = {"kcal": 10.0, "grasa": 1.0}
    meal2 = {"name": "Mandarina con almendras", "ingredients": ["0 g de almendras"], "ingredients_raw": ["0 g de almendras"]}
    assert ip._rescatar_cero(meal2, "almendras", db, sin_sitio, []) is None, "sin sitio en el día no se sube"
    assert ip._rescatar_cero(meal2, "almendras", db, {"kcal": 400.0, "grasa": 20.0}, ["Frutos Secos", "almendras"]) is None


def test_el_plato_del_modelo_ya_no_queda_fuera_de_la_restauracion():
    src = _src("identidad_plato.py")
    i = src.index("def restaurar_meal")
    cuerpo = src[i:src.index("\ndef ", i + 10)]
    assert "_subir_identidad_del_modelo(" in cuerpo
    assert '("proteina", "resto")' in src, "la proteína va primero cuando el margen del día es corto"


def test_mango_maduro_no_es_platano():
    from nutrition_db import IngredientNutritionDB
    rows = [{"name": "Plátano maduro", "aliases": ["plátanos maduros", "maduro"], "kcal_per_100g": 137},
            {"name": "Mango", "aliases": [], "kcal_per_100g": 60},
            {"name": "Guineo", "aliases": ["banano", "banana"], "kcal_per_100g": 89}]
    db = IngredientNutritionDB(rows=rows)
    assert db._match_row("mango maduro")["name"] == "Mango"
    assert db._match_row("150 g de mango maduro en cubos")["name"] == "Mango"
    assert db._match_row("guineo maduro")["name"] == "Guineo"
    assert db._match_row("maduro")["name"] == "Plátano maduro", "el descriptor solo sigue valiendo"
    assert db._match_row("plátano maduro")["name"] == "Plátano maduro"


def test_marker_174():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src("app.py"))
    assert m and int(m.group(1)) >= 174 and m.group(2) >= "2026-09-23"
