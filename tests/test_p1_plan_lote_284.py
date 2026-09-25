# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-284 · 2026-09-25] «¾ taza de arroz blanco cocido» son ~150 kcal, no 491.

El resolvedor de macros contaba una línea de grano/legumbre COCIDO contra la fila SECA del catálogo (×3). El arreglo de
julio reescribía solo «N g de X cocido» anclado al final, en assemble (después del solver): tazas, «… y refrigerado»,
«(86 g)» y latas quedaban fuera — 75 de 581 líneas de granos/legumbres en las baterías del 25-sep; en el plan del dueño
un almuerzo decía 832 kcal y aportaba ~490."""
from __future__ import annotations

import pathlib

import cocido_en_catalogo as cc
import graph_orchestrator as go
from nutrition_db import IngredientNutritionDB

_BACKEND = pathlib.Path(__file__).resolve().parents[1]

# Filas reales del catálogo (Neon, 25-sep) — solo las columnas que el resolvedor lee.
_FILAS = [
    {"name": "Arroz blanco", "kcal_per_100g": 358.6, "protein_g_per_100g": 7.1, "carbs_g_per_100g": 79.5,
     "fats_g_per_100g": 0.7, "density_g_per_cup": 185, "category": "Despensa"},
    {"name": "Habichuelas negras", "kcal_per_100g": 348.8, "protein_g_per_100g": 21.6, "carbs_g_per_100g": 62.4,
     "fats_g_per_100g": 1.4, "density_g_per_cup": 180, "container_weight_g": 425, "category": "Despensa"},
    {"name": "Garbanzos", "kcal_per_100g": 388.4, "protein_g_per_100g": 20.5, "carbs_g_per_100g": 63.0,
     "fats_g_per_100g": 6.0, "density_g_per_cup": 200, "container_weight_g": 425, "category": "Despensa"},
    {"name": "Bulgur", "kcal_per_100g": 342.0, "protein_g_per_100g": 12.3, "carbs_g_per_100g": 75.9,
     "fats_g_per_100g": 1.3, "density_g_per_cup": 140, "category": "Despensa"},
    {"name": "Soya texturizada", "kcal_per_100g": 327.0, "protein_g_per_100g": 51.5, "carbs_g_per_100g": 30.0,
     "fats_g_per_100g": 1.2, "density_g_per_cup": 75, "category": "Proteínas"},
    {"name": "Pechuga de pollo", "kcal_per_100g": 120.0, "protein_g_per_100g": 22.5, "carbs_g_per_100g": 0.0,
     "fats_g_per_100g": 2.6, "density_g_per_unit": 200, "category": "Proteínas"},
    {"name": "Frijoles horneados", "kcal_per_100g": 94.0, "protein_g_per_100g": 4.8, "carbs_g_per_100g": 21.0,
     "fats_g_per_100g": 0.4, "density_g_per_cup": 254, "container_weight_g": 454, "category": "Despensa"},
]


def _db():
    return IngredientNutritionDB(rows=[dict(r) for r in _FILAS])


def _kcal(db, linea):
    return round((db.macros_from_ingredient_string(linea) or {}).get("kcal") or 0)


def test_la_taza_de_arroz_cocido_del_plan_del_dueno():
    db = _db()
    # 0,74 taza cocida = 117 g cocidos = 42 g crudos ≈ 150 kcal (antes 491: la taza SECA de 185 g)
    assert 140 <= _kcal(db, "0.74 taza de arroz blanco cocido y refrigerado") <= 165
    assert 60 <= _kcal(db, "⅓ taza de arroz blanco cocido") <= 75                      # antes 219
    assert _kcal(db, "60 g de arroz blanco crudo") == 215                                  # lo crudo, intacto


def test_gramos_cocidos_y_el_hint_del_modelo():
    db = _db()
    assert 140 <= _kcal(db, "110 g de habichuelas negras cocidas") <= 142                 # 110 × 1,27; antes 384
    assert 108 <= _kcal(db, "½ taza de habichuelas negras cocidas (86 g)") <= 110         # manda el hint cocido
    assert 106 <= _kcal(db, "85 g de garbanzos cocidos y escurridos") <= 110
    assert _kcal(db, "40 g de habichuelas negras secas") == 140                            # «secas»: la base ya
    assert _kcal(db, "¼ taza de garbanzos secos (60 g) cocidos") == 233                     # dice seco: literal
    assert _kcal(db, "15 g de habichuelas negras cocidas (5 g en seco)") == 17              # el paréntesis ya es seco


def test_legumbres_de_lata():
    db = _db()
    k = _kcal(db, "½ taza de habichuelas negras de lata, escurridas")                      # antes 314
    assert 105 <= k <= 112, k
    lata = _kcal(db, "1 lata de garbanzos")                                                 # antes 1.651
    assert 320 <= lata <= 360, lata                                                         # ~89 g secos
    assert _kcal(db, "1 lata de frijoles horneados") == round(454 * 0.94)                  # fila en cocido: tal cual


def test_bulgur_y_soya_texturizada():
    db = _db()
    assert 145 <= _kcal(db, "1 taza de bulgur cocido") <= 156                                # USDA: 151; antes 479
    assert 108 <= _kcal(db, "95 g de soya texturizada hidratada y cocida") <= 110            # el 0,35× de la lista
    assert _kcal(db, "½ taza de soya texturizada cocida") == round(37.5 * 3.27)              # sin densidad cocida: tal cual


def test_lo_que_no_es_grano_ni_legumbre_no_cambia():
    db = _db()
    assert _kcal(db, "150 g de pechuga de pollo cocida") == 180


def test_el_solver_ve_el_mismo_numero():
    db = _db()
    m = db.macros_for_line(0.74, "taza", "arroz blanco cocido y refrigerado")
    assert m and 140 <= m["kcal"] <= 165, m


def test_paridad_con_el_normalizador_de_julio():
    """Las kcal cocidas de referencia son las MISMAS: una línea reescrita a «crudo» da el mismo número."""
    ref = {toks[0]: k for toks, k in go._COOKED_GRAIN_REF_KCAL}
    fam = {toks[0]: k for toks, k, _t, _l in cc.FAMILIAS}
    assert fam["arroz"] == ref["arroz"] and fam["pasta"] == ref["pasta"] and fam["lenteja"] == ref["habichuela"]
    db = _db()
    antes = _kcal(db, "150 g de habichuelas negras cocidas")
    seco = round(150 * 127.0 / 348.8)
    assert abs(antes - _kcal(db, f"{seco} g de habichuelas negras secas")) <= 2


def test_el_gancho_vive_en_el_resolvedor():
    src = (_BACKEND / "nutrition_db.py").read_text(encoding="utf-8")
    assert 'return __import__("cocido_en_catalogo").en_base_de_la_fila(s, self._grams_literales(s), self)' in src
    assert 'grams = __import__("cocido_en_catalogo").en_base_de_la_fila(f"{qty} {unit} de {raw_name}", grams, self)' in src
    assert "tooltip-anchor: P1-PLAN-LOTE-284-COCIDO-EN-BASE" in (_BACKEND / "cocido_en_catalogo.py").read_text(encoding="utf-8")
