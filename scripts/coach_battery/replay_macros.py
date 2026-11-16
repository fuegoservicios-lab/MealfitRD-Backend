# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1039 · A-41 · 2026-11-16] Replay de `macros_por_gramos.reconciliar` sobre las comidas GUARDADAS.

Gratis y de solo lectura: no llama a ningún modelo ni escribe en la base. Dice cuántas comidas anotadas por el coach
(chat/foto) cambiarían y cuánto, y con `--ver N` enseña N ejemplos para leerlos uno a uno (nombres de alimentos y
cifras; ni usuarios ni fechas). Uso, en el servidor:

    python scripts/coach_battery/replay_macros.py --dias 90 --ver 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def leer_comidas(dias: int) -> list:
    import psycopg
    from psycopg.rows import dict_row
    with psycopg.connect(os.environ["NEON_DATABASE_URL"], row_factory=dict_row) as c:
        return c.execute(
            "SELECT calories, protein, carbs, healthy_fats, ingredients FROM public.consumed_meals "
            "WHERE source = ANY(%s) AND created_at > now() - make_interval(days => %s) "
            "AND jsonb_typeof(to_jsonb(ingredients)) = 'array' ORDER BY created_at DESC",
            (["chat", "photo"], int(dias))).fetchall()


def medir(comidas: list, db, reconciliar, resolver) -> dict:
    out = {"comidas": len(comidas), "con_ingredientes": 0, "completas": 0, "con_simples": 0,
           "cambian": {"suelo_de_proteina": 0},
           "delta_proteina": [], "delta_kcal_pct": [], "ejemplos": []}
    for m in comidas:
        ings = m.get("ingredients")
        if isinstance(ings, str):
            try:
                ings = json.loads(ings)
            except Exception:
                ings = None
        if not isinstance(ings, list) or not ings:
            continue
        out["con_ingredientes"] += 1
        r = resolver(ings, db)
        if r and r["simples"]:
            out["con_simples"] += 1
            out["completas"] += int(r["completo"])
        c = reconciliar({**m, "ingredients": ings}, db)
        if not c:
            continue
        out["cambian"][c["motivo"]] += 1
        a = c["antes"]
        out["delta_proteina"].append(round(c["protein"] - a["protein"], 1))
        if a["calories"]:
            out["delta_kcal_pct"].append(round(100 * (c["calories"] - a["calories"]) / a["calories"], 1))
        out["ejemplos"].append({
            "motivo": c["motivo"],
            "antes": {k: round(v) for k, v in a.items()},
            "despues": {k: c[k] for k in ("calories", "protein", "carbs", "healthy_fats")},
            "piezas": [(f["nombre"][:50], f["gramos"], f["fila"], f["macros"] and round(f["macros"]["protein"], 1),
                        f["macros"] and round(f["macros"]["kcal"])) for f in (r or {}).get("componentes", [])],
        })
    return out


def _percentiles(xs: list) -> dict:
    if not xs:
        return {}
    s = sorted(xs)
    q = lambda p: s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))]  # noqa: E731
    return {"n": len(s), "min": s[0], "p10": q(.1), "p50": q(.5), "p90": q(.9), "max": s[-1]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dias", type=int, default=90)
    ap.add_argument("--ver", type=int, default=0)
    args = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(BACKEND / ".env")
    import db_core
    if db_core.connection_pool:
        db_core.connection_pool.open(wait=True)
    from nutrition_db import IngredientNutritionDB
    from macros_por_gramos import reconciliar, resolver
    r = medir(leer_comidas(args.dias), IngredientNutritionDB(), reconciliar, resolver)
    ejemplos = r.pop("ejemplos")
    r["delta_proteina"] = _percentiles(r["delta_proteina"])
    r["delta_kcal_pct"] = _percentiles(r["delta_kcal_pct"])
    print(json.dumps(r, ensure_ascii=False))
    for e in ejemplos[:args.ver]:
        print(json.dumps(e, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
