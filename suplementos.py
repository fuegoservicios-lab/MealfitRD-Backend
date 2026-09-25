# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-290 · 2026-09-25] SSOT de los suplementos en la Alacena (`user_inventory.kind = 'supplement'`).

La etiqueta por porción la decide el ENVASE (foto o marca); sin ella, un estimado genérico marcado como tal. Las
macros de una toma las calcula este módulo (etiqueta × porciones), nunca el modelo. Todo lector que trata las filas de
la Nevera como INGREDIENTES filtra `kind = 'food'` (guard por AST en tests/test_p1_plan_lote_290.py).
Spec: docs/superpowers/specs/2026-09-25-suplementos-alacena-design.md. tooltip-anchor: P1-PLAN-LOTE-290
"""
from __future__ import annotations

UNIDADES = ("scoop", "capsula", "porcion", "g")
FUENTES = ("foto", "marca", "estimado")
_CAMPOS = ("kcal", "protein_g", "carbs_g", "fats_g")

# Estimados genéricos POR PORCIÓN (unidad típica del producto). Claves = constants.SUPPLEMENT_NAMES.
ESTIMADOS = {
    "whey_protein":  {"unidad": "scoop",   "serving_g": 30, "kcal": 120, "protein_g": 24, "carbs_g": 3, "fats_g": 1.5},
    "vegan_protein": {"unidad": "scoop",   "serving_g": 33, "kcal": 130, "protein_g": 21, "carbs_g": 6, "fats_g": 2.5},
    "creatine":      {"unidad": "g",       "serving_g": 5,  "kcal": 0,   "protein_g": 0,  "carbs_g": 0, "fats_g": 0},
    "bcaa":          {"unidad": "scoop",   "serving_g": 10, "kcal": 10,  "protein_g": 0,  "carbs_g": 2, "fats_g": 0},
    "pre_workout":   {"unidad": "scoop",   "serving_g": 10, "kcal": 5,   "protein_g": 0,  "carbs_g": 1, "fats_g": 0},
    "fat_burner":    {"unidad": "capsula", "serving_g": 1,  "kcal": 0,   "protein_g": 0,  "carbs_g": 0, "fats_g": 0},
    "collagen":      {"unidad": "scoop",   "serving_g": 10, "kcal": 36,  "protein_g": 9,  "carbs_g": 0, "fats_g": 0},
    "multivitamin":  {"unidad": "capsula", "serving_g": 1,  "kcal": 0,   "protein_g": 0,  "carbs_g": 0, "fats_g": 0},
    "omega3":        {"unidad": "capsula", "serving_g": 1,  "kcal": 10,  "protein_g": 0,  "carbs_g": 0, "fats_g": 1},
    "magnesium":     {"unidad": "capsula", "serving_g": 1,  "kcal": 0,   "protein_g": 0,  "carbs_g": 0, "fats_g": 0},
    "probiotics":    {"unidad": "capsula", "serving_g": 1,  "kcal": 0,   "protein_g": 0,  "carbs_g": 0, "fats_g": 0},
    "electrolytes":  {"unidad": "porcion", "serving_g": 7,  "kcal": 10,  "protein_g": 0,  "carbs_g": 2, "fats_g": 0},
}


def _num(v):
    try:
        f = float(v)
        return f if f == f and f >= 0 else None
    except (TypeError, ValueError):
        return None


def etiqueta_valida(etiqueta) -> dict | None:
    """La etiqueta normalizada, o None si falta o es inverosímil (kcal > 9·g + 20, o macros > gramos + 1)."""
    if not isinstance(etiqueta, dict):
        return None
    g = _num(etiqueta.get("serving_g"))
    out = {"serving_g": g if g is not None else 0}
    for k in _CAMPOS:
        v = _num(etiqueta.get(k, 0))
        if v is None:
            return None
        out[k] = v
    if g:
        if out["kcal"] > 9 * g + 20:
            return None
        if out["protein_g"] + out["carbs_g"] + out["fats_g"] > g + 1:
            return None
    return out


def macros_de_porciones(etiqueta, porciones) -> dict | None:
    """Etiqueta × porciones, redondeado a 1 decimal. None sin etiqueta válida o sin porciones."""
    e = etiqueta_valida(etiqueta)
    n = _num(porciones)
    if not e or not n:
        return None
    return {k: round(e[k] * n, 1) for k in _CAMPOS}
