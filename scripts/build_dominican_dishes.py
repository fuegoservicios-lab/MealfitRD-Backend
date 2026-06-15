#!/usr/bin/env python
"""[G17-RECIPE-DECOMP · 2026-06-15 · FASE 1 piloto] Reemplaza "lo criollo puro" SIN dataset propietario:
reconstruye cada plato dominicano desde sus INGREDIENTES constituyentes (todos en el catálogo / USDA FDC,
CC0) y computa sus macros per-100g. Es como FNDDS/las tablas profesionales calculan platos mixtos — gratis,
comercial-OK, TRAZABLE (cada macro cita su ingrediente) y culturalmente correcto (la receta ES el plato DR).

Modelo: receta = lista de (ingrediente del catálogo, gramos CRUDOS) + peso COCIDO final del plato. Los macros
crudos se suman y se dividen por el peso cocido → per-100g del plato terminado (el agua absorbida en
arroz/guisos diluye; el aceite absorbido en fritos concentra). Cada plato se CROSS-CHECKEA contra su análogo
FNDDS (data/fndds_dish_reference.json, ground-truth externo CC0): si el perfil de macros coincide → confiable;
si diverge → recalibrar el peso final / la receta (auto-corrección contra fuente independiente).

Imprime una tabla computado-vs-FNDDS a stderr y escribe data/dominican_dishes.json a stdout. One-shot, NO runtime.
Uso: PYTHONPATH=backend python backend/scripts/build_dominican_dishes.py > backend/data/dominican_dishes.json

[P2-LOGGER-EXEMPT: script CLI one-shot, salida JSON a stdout intencional]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_RECIPES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "dominican_dish_recipes.json")


def _load_recipes():
    """[FASE 2] Recetas curadas (input) desde data/dominican_dish_recipes.json: dish -> {label, finished_g,
    method, fndds_key, ingredients:[[nombre_catalogo, g_crudo], ...]}. Separar el DATO curado de la lógica
    de cómputo permite expandir el set sin tocar el script."""
    with open(_RECIPES_PATH, encoding="utf-8") as f:
        return (json.load(f) or {}).get("recipes") or {}


def _fractions(p, c, f):
    kp, kc, kf = 4.0 * p, 4.0 * c, 9.0 * f
    tot = kp + kc + kf
    return (kp / tot, kc / tot, kf / tot) if tot > 0 else None


def main():
    from nutrition_db import IngredientNutritionDB
    import db_core
    if getattr(db_core, "connection_pool", None):
        db_core.connection_pool.open()
    db = IngredientNutritionDB()

    # Referencia FNDDS para el cross-check (puede faltar).
    fpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "fndds_dish_reference.json")
    try:
        fndds = (json.load(open(fpath, encoding="utf-8")) or {}).get("dishes") or {}
    except Exception:
        fndds = {}

    recipes = _load_recipes()
    out = {}
    print(f"{'plato':26} {'kcal/100g':>9} {'P':>5} {'C':>6} {'F':>5}  {'cob':>4}  FNDDS-cross", file=sys.stderr)
    for key, rec in recipes.items():
        tot = {"kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fats": 0.0}
        constituents, resolved = [], 0
        for ing in rec["ingredients"]:
            name, g = (ing["name"], ing["g"]) if isinstance(ing, dict) else (ing[0], ing[1])
            m = db.macros_from_ingredient_string(f"{g}g {name}") or {}
            ok = bool(m)
            if ok:
                resolved += 1
                for k in tot:
                    tot[k] += (m.get(k) or 0.0)
            constituents.append({"name": name, "g": g, "resolved": ok,
                                 "macros": {k: round(m.get(k) or 0.0, 1) for k in tot} if ok else None})
        fg = rec["finished_g"]
        per100 = {k: round(tot[k] / fg * 100, 1) for k in tot}
        cov = round(resolved / len(rec["ingredients"]), 2)

        # Cross-check vs FNDDS (perfil de macros + kcal/100g).
        cross = None
        fk = rec.get("fndds_key")
        if fk and fk in fndds:
            fp = fndds[fk]["per_100g"]
            af = _fractions(per100["protein"], per100["carbs"], per100["fats"])
            ff = _fractions(fp["protein"], fp["carbs"], fp["fats"])
            prof_dev = round(100 * sum(abs(a - b) for a, b in zip(af, ff)) / 3, 1) if (af and ff) else None
            cross = {"fndds_dish": fndds[fk]["fnds_description"], "fndds_per_100g": fp,
                     "profile_dev_pts": prof_dev, "kcal_delta_pct": round(100 * (per100["kcal"] - fp["kcal"]) / fp["kcal"], 1) if fp["kcal"] else None}
        out[key] = {
            "label": rec["label"], "method": rec["method"], "finished_g": fg,
            "per_100g": per100, "resolution_coverage": cov,
            "constituents": constituents, "fndds_cross_check": cross,
            "provenance": "computed from catalog/USDA constituents (CC0)",
        }
        _x = (f"dev {cross['profile_dev_pts']}pts, kcalΔ {cross['kcal_delta_pct']}%" if cross else "—")
        print(f"{key:26} {per100['kcal']:9} {per100['protein']:5} {per100['carbs']:6} {per100['fats']:5}  {cov:4}  {_x}", file=sys.stderr)

    print(json.dumps({
        "_note": ("[G17-RECIPE-DECOMP · FASE 1] Platos dominicanos reconstruidos desde ingredientes "
                  "constituyentes (CC0). Macros per-100g del plato cocido = suma de macros crudos / peso "
                  "final. Trazable (constituents) + cross-check vs FNDDS externo. Reemplaza la necesidad de "
                  "una tabla criolla propietaria. Generado por scripts/build_dominican_dishes.py."),
        "dishes": out,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
