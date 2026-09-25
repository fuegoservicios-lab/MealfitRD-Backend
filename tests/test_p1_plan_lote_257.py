# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-257 · 2026-09-25] El techo renal de proteína se cumple en lo que se entrega.

Batería rd252 (renal + gota, techo KDIGO 60 g/día): el plan de EMERGENCIA salió con 96/75/75 g. Réplica del escudo:
el recorte solo escalaba carne y pescado (4 huevos intactos, la pechuga a 35 g) y `identidad_plato` la devolvía a 90 g
mirando kcal y grasa. Con el arreglo, la réplica da 60/60/60 con 2 huevos y 65-75 g de pechuga/pescado.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
import identidad_plato as ip  # noqa: E402
import recorte_renal as rr  # noqa: E402

# Por 100 g / por unidad: (proteína, carbohidratos, grasa)
_TABLA = {"huevo": (6.3, 0.4, 4.8), "pechuga": (23.0, 0.0, 2.0), "arroz": (2.7, 28.0, 0.3), "queso": (18.0, 3.0, 20.0)}


class _DB:
    def macros_from_ingredient_string(self, s):
        m = re.match(r"\s*([\d.]+)\s*(?:g de )?(.*)", str(s).lower())
        if not m:
            return None
        n, nombre = float(m.group(1)), m.group(2)
        for clave, (p, c, f) in _TABLA.items():
            if clave in nombre:
                k = n if clave == "huevo" else n / 100.0
                return {"protein": p * k, "carbs": c * k, "fats": f * k, "kcal": 4 * p * k + 4 * c * k + 9 * f * k}
        return None


def test_recortable_incluye_huevo_y_lacteo_solo_bajo_techo_renal():
    db = _DB()
    assert not go._ingredient_is_protein_dominant("4 huevos", db)        # su grasa pesa más en kcal
    assert not rr.recortable("4 huevos", db)
    assert rr.recortable("4 huevos", db, incluye_huevo_lacteo=True)
    assert rr.recortable("30 g de queso blanco", db, incluye_huevo_lacteo=True)
    assert not rr.recortable("1 taza de leche de coco", db, incluye_huevo_lacteo=True)
    assert rr.recortable("90 g de pechuga de pollo", db)


def test_el_recorte_renal_baja_tambien_los_huevos():
    db = _DB()
    meals = [{"name": "Huevos", "protein": 25, "carbs": 2, "fats": 19, "ingredients": ["4 huevos"]},
             {"name": "Pollo y Arroz", "protein": 25, "carbs": 42, "fats": 3,
              "ingredients": ["90 g de pechuga de pollo", "150 g de arroz"]}]
    assert go._trim_day_protein_to_ceiling(meals, 30, db, ceiling_pct=1.0, incluye_huevo_lacteo=True)
    assert meals[0]["ingredients"][0] != "4 huevos", meals[0]["ingredients"]
    # la pechuga no cae a migajas: el recorte se reparte con el huevo
    g_pollo = float(re.match(r"([\d.]+)", meals[1]["ingredients"][0]).group(1))
    assert g_pollo > 40, meals[1]["ingredients"]


def test_sin_techo_renal_el_recorte_no_toca_el_huevo():
    db = _DB()
    meals = [{"name": "Huevos", "protein": 25, "carbs": 2, "fats": 19, "ingredients": ["4 huevos"]},
             {"name": "Pollo", "protein": 25, "carbs": 0, "fats": 2, "ingredients": ["110 g de pechuga de pollo"]}]
    go._trim_day_protein_to_ceiling(meals, 30, db, ceiling_pct=1.0)
    assert meals[0]["ingredients"] == ["4 huevos"]


def test_la_identidad_ve_el_margen_de_proteina():
    plan = {"calories": 1750, "macros": {"fats": "58g", "protein": "60g"},
            "renal_protein_cap": {"applied": True, "protein_g": 60}}
    obj = ip.objetivos_de(plan)
    assert obj["proteina_techo"] == 60
    meals = [{"cals": 400, "fats": 10, "protein": 30}, {"cals": 400, "fats": 10, "protein": 29}]
    m = ip._margen_del_dia(meals, obj)
    assert 0.5 < m["proteina"] < 1.5
    sin = ip._margen_del_dia(meals, ip.objetivos_de({"calories": 1750, "macros": {"fats": "58g"}}))
    assert sin["proteina"] == float("inf")


def test_anclas():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count("incluye_huevo_lacteo=True") == 2
    assert "incluye_huevo_lacteo=_renal257" in src
    ids = (_BACKEND / "identidad_plato.py").read_text(encoding="utf-8")
    assert 'margen.get("proteina", float("inf"))' in ids and "P1-PLAN-LOTE-257-IDENTIDAD-RENAL" in ids
    assert "P1-PLAN-LOTE-257-TECHO-RENAL" in (_BACKEND / "recorte_renal.py").read_text(encoding="utf-8")


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 257
