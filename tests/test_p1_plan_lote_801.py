# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-801 · 2026-09-28] El huevo se cuenta, no se pesa: la identidad del plato lo sube en unidades enteras.

Corpus de la cola 744 (426 planes de la batería RD): 140 líneas «60 g de huevo» / «60 g de clara de huevo» / «90 g de
huevo», 137 de la restauración de identidad: «Avena cremosa con aguacate y huevo duro» con «60 g de huevo» mientras el paso
hierve «el huevo»; «Manzana con crema de maní, pistachos y huevo» con «60 g de clara de huevo» y el paso «Cocina 3 huevos y
2 claras de huevo»; «Huevo Duro con Mandarina y Nueces» con «90 g de huevo» (1,8 huevos), fuera del tope diario de enteros.
"""
from __future__ import annotations

import json
import pathlib
import re
import unicodedata
from types import SimpleNamespace

import culinary_coherence as cc
import huevo_en_unidades as hu
import identidad_plato as ip

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_IDX = cc.build_culinary_index(json.loads((_BACKEND / "scripts" / "data" / "culinary_corpus_2026_09_12.json")
                                          .read_text(encoding="utf-8"))["catalogo_filas"])


def _sa(s) -> str:
    return unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()


class _DB:
    """Macros por gramo (kcal, proteína, carbohidrato, grasa) y el peso de una unidad, como el catálogo (50 / 33 / 17 g)."""
    _POR_G = (("clara de huevo", (0.52, 0.109, 0.007, 0.002)), ("yema de huevo", (3.22, 0.159, 0.036, 0.265)),
              ("huevo", (1.43, 0.126, 0.007, 0.095)), ("avena", (3.89, 0.169, 0.663, 0.069)),
              ("aguacate", (1.60, 0.02, 0.085, 0.147)), ("mandarina", (0.53, 0.008, 0.134, 0.003)),
              ("nueces", (6.54, 0.152, 0.137, 0.652)), ("casabe", (3.30, 0.01, 0.80, 0.003)))
    _UNIDAD = (("clara", 33.0), ("yema", 17.0), ("huevo", 50.0), ("mandarina", 88.0))
    _CATEGORIA = (("huevo", "Proteínas"), ("clara", "Proteínas"), ("yema", "Proteínas"), ("avena", "Granos"),
                  ("aguacate", "Frutas"), ("mandarina", "Frutas"), ("nueces", "Frutos secos"), ("casabe", "Víveres"))

    def _gramos(self, s):
        n = _sa(s).strip()
        m = re.match(r"([\d.]+)\s*g\s+de\s+(.+)$", n)
        if m:
            return float(m.group(1)), m.group(2)
        m = re.match(r"(\d+)\s+(.+)$", n)
        if m:
            w = next((g for k, g in self._UNIDAD if k in m.group(2)), None)
            if w:
                return int(m.group(1)) * w, m.group(2)
        return None, None

    def grams_from_ingredient_string(self, s):
        return self._gramos(s)[0]

    def macros_from_ingredient_string(self, s):
        g, nombre = self._gramos(s)
        per = next((v for k, v in self._POR_G if nombre and k in nombre), None)
        if g is None or per is None:
            return None
        return {"grams": g, "kcal": per[0] * g, "protein": per[1] * g, "carbs": per[2] * g, "fats": per[3] * g}

    def category_of(self, s):
        return next((c for k, c in self._CATEGORIA if k in _sa(s)), None)

    def lookup(self, s):
        per = next((v for k, v in self._POR_G if k in _sa(s)), None)
        return SimpleNamespace(protein=per[1] * 100) if per else None


def _con_raw(meal):
    meal["ingredients_raw"] = list(meal["ingredients"])
    return meal


def _en_gramos(lineas):
    return [x for x in lineas if re.match(r"\s*[\d.]+\s*g\s+de\s+(huevo|clara|yema)", _sa(x))]


def test_un_huevo_ya_es_la_racion_que_da_nombre():
    meal = _con_raw({"name": "Avena cremosa con aguacate y huevo duro",
                     "ingredients": ["30 g de avena", "30 g de aguacate", "1 huevo"]})
    ip._subir_identidad_del_modelo(meal, _IDX, db=_DB(), margen={"kcal": 600.0, "grasa": 40.0, "proteina": float("inf")})
    assert "1 huevo" in meal["ingredients"] and not _en_gramos(meal["ingredients"]), meal["ingredients"]
    assert "1 huevo" in meal["ingredients_raw"], meal["ingredients_raw"]


def test_la_clara_sube_en_claras_enteras():
    meal = _con_raw({"name": "Huevos y Avena", "ingredients": ["3 huevos", "40 g de avena", "1 clara de huevo"]})
    ip._subir_identidad_del_modelo(meal, _IDX, db=_DB(), margen={"kcal": 600.0, "grasa": 40.0, "proteina": float("inf")})
    assert "2 claras de huevo" in meal["ingredients"], meal["ingredients"]
    assert "2 claras de huevo" in meal["ingredients_raw"] and not _en_gramos(meal["ingredients_raw"])


def _dia(*meals):
    return {"day": 1, "meals": list(meals)}


def _huevo_duro():
    return _con_raw({"name": "Huevo duro con mandarina y nueces", "cals": 220, "fats": 14, "protein": 11,
                     "ingredients": ["75 g de huevo", "1 mandarina", "10 g de nueces"]})


def test_el_huevo_entero_no_pasa_el_tope_del_dia():
    revoltillo = _con_raw({"name": "Revoltillo de huevos", "cals": 215, "fats": 14, "protein": 19,
                           "ingredients": ["3 huevos"]})
    duro = _huevo_duro()
    ip.restaurar_identidad([_dia(revoltillo, duro)], db=_DB(), index=_IDX, objetivos={"kcal": 2000, "grasa": 70})
    assert "75 g de huevo" in duro["ingredients"], "el día ya lleva 3 enteros (+1,5): no se suma otro"
    assert not any("90 g" in x for x in duro["ingredients"]), duro["ingredients"]


def test_con_sitio_el_protagonista_sube_a_huevos_enteros():
    duro = _huevo_duro()
    ip.restaurar_identidad([_dia(duro)], db=_DB(), index=_IDX, objetivos={"kcal": 2000, "grasa": 70})
    assert "2 huevos" in duro["ingredients"] and not _en_gramos(duro["ingredients"]), duro["ingredients"]


def test_el_huevo_en_cero_vuelve_como_un_huevo():
    meal = _con_raw({"name": "Casabe con huevo", "ingredients": ["20 g de casabe", "0 g de huevo"]})
    out = ip._rescatar_cero(meal, "huevo", _DB(), {"kcal": 400.0, "grasa": 20.0}, [])
    assert out and "1 huevo" in meal["ingredients"] and "1 huevo" in meal["ingredients_raw"], meal["ingredients"]


def test_solo_el_alimento_crudo():
    assert hu.tipo("Huevo") == "entero" and hu.tipo("Clara de huevo") == "clara" and hu.tipo("Yema de huevo") == "yema"
    assert hu.tipo("Huevos rellenos") is None and hu.tipo("Pollo") is None
    assert hu.linea("Clara de huevo", 2) == "2 claras de huevo" and hu.linea("Huevo", 1) == "1 huevo"
