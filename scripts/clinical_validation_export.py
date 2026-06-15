"""[P0-CLINICAL-VALIDATION · 2026-06-14] Export para validación clínica externa/humana.

Cierra (la parte implementable en código de) el gap P0 del audit clínico: "precisión autoafirmada,
sin validación externa ni humana". Hace DOS cosas sobre planes REALES persistidos en Neon:

  1. CHECK AUTOMÁTICO DE INTEGRIDAD (validación "externa" determinista): para cada comida recomputa
     los macros DESDE LOS INGREDIENTES vía el catálogo `master_ingredients` (ground-truth independiente
     del LLM) y los compara con los macros que el LLM AFIRMÓ en el header de la comida. Una divergencia
     grande = el LLM "mintió" sobre los macros (el plato no aporta lo que dice). Esto es lo más cercano
     a una validación externa que se puede automatizar sin un dataset público.

  2. EXPORT PARA NUTRICIONISTA (habilita la validación humana): CSV por día-fila con target vs
     entregado(claim) vs recomputado + clinical_band_score + columnas en blanco para que un
     profesional marque aprobado/notas.

NO persiste nada (read-only). Uso:
    python scripts/clinical_validation_export.py [--n 15] [--days 30] [--out /tmp/clinical_review.csv]
"""
import argparse
import asyncio
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _num(x) -> float:
    try:
        s = str(x).lower().replace("kcal", "").replace("g", "").strip()
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else 0.0
    except Exception:
        return 0.0


async def _open_pools():
    """Abre los pools de Neon igual que el lifespan de FastAPI (app.py)."""
    import db_core
    if getattr(db_core, "connection_pool", None):
        db_core.connection_pool.open()
    if getattr(db_core, "async_connection_pool", None):
        await db_core.async_connection_pool.open()
    await asyncio.sleep(1.5)


def _recompute_day_from_ingredients(day, db):
    """Suma macros recomputados desde los ingredientes (catálogo) — ground-truth determinista.
    Retorna (macros_dict, resolved_count, total_count)."""
    agg = {"protein": 0.0, "carbs": 0.0, "fats": 0.0, "kcal": 0.0}
    resolved = total = 0
    for meal in (day.get("meals") or []):
        for ing in (meal.get("ingredients") or []):
            total += 1
            m = db.macros_from_ingredient_string(str(ing)) or {}
            if m:
                resolved += 1
                for k in agg:
                    agg[k] += (m.get(k) or 0.0)
    return agg, resolved, total


def _delivered_day_claimed(day):
    """Suma los macros que el LLM AFIRMÓ en el header de cada comida del día."""
    agg = {"protein": 0.0, "carbs": 0.0, "fats": 0.0, "kcal": 0.0}
    for meal in (day.get("meals") or []):
        agg["protein"] += _num(meal.get("protein"))
        agg["carbs"] += _num(meal.get("carbs"))
        agg["fats"] += _num(meal.get("fats"))
        agg["kcal"] += _num(meal.get("cals") if meal.get("cals") is not None else meal.get("calories"))
    return agg


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="nº de planes a muestrear")
    ap.add_argument("--days", type=int, default=45, help="ventana lookback (días)")
    ap.add_argument("--out", default="/tmp/clinical_review.csv")
    args = ap.parse_args()

    await _open_pools()
    from db_core import execute_sql_query
    from nutrition_db import IngredientNutritionDB
    db = IngredientNutritionDB()

    # Los planes chunked usan status partial/complete_partial/generating_next (no 'complete');
    # excluimos solo 'failed' (sin macros) y exigimos days>0 + macros presentes.
    rows = execute_sql_query(
        "SELECT id::text AS id, user_id::text AS user_id, plan_data, created_at "
        "FROM meal_plans "
        "WHERE created_at > NOW() - (%s || ' days')::interval "
        "  AND jsonb_array_length(plan_data->'days') > 0 "
        "  AND plan_data ? 'macros' "
        "  AND COALESCE(plan_data->>'generation_status','') <> 'failed' "
        "ORDER BY created_at DESC LIMIT %s",
        (str(args.days), args.n), fetch_all=True) or []

    print(f"[P0-CLINICAL-VALIDATION] {len(rows)} planes muestreados (últimos {args.days}d).", flush=True)

    out_rows = []
    # Acumuladores de integridad (claim vs recomputado) sobre celdas día×macro resueltas.
    integ_cells = 0
    integ_band = 0           # |claim-recomputado|/recomputado <= 0.15
    integ_all_cells = 0      # [P2-VALIDATION-NO-FILTER · 2026-06-15] (gap-audit G16) idem SIN el filtro res_pct>=60
    integ_all_band = 0
    days_low_res = 0         # días con res_pct < 60 (excluidos del agregado filtrado → tapan el 0-silencioso)
    band_cells = 0
    band_in = 0              # entregado(claim) dentro de [0.90,1.12]×target
    for r in rows:
        pd = r["plan_data"]
        if isinstance(pd, str):
            pd = json.loads(pd)
        is_fb = bool(pd.get("_is_fallback"))
        cbs = (pd.get("clinical_band_score") or {}).get("score")
        target = {"protein": _num((pd.get("macros") or {}).get("protein")),
                  "carbs": _num((pd.get("macros") or {}).get("carbs")),
                  "fats": _num((pd.get("macros") or {}).get("fats")),
                  "kcal": _num(pd.get("calories"))}
        for di, day in enumerate(pd.get("days") or [], 1):
            claimed = _delivered_day_claimed(day)
            recomp, res, tot = _recompute_day_from_ingredients(day, db)
            res_pct = round(100 * res / tot) if tot else 0
            row = {
                "plan_id": r["id"][:8], "user": r["user_id"][:8], "fallback": is_fb,
                "band_score": cbs, "dia": di, "res_pct": res_pct,
            }
            for mac in ("protein", "carbs", "fats", "kcal"):
                t, cl, rc = target[mac], claimed[mac], recomp[mac]
                row[f"tgt_{mac}"] = round(t)
                row[f"claim_{mac}"] = round(cl)
                row[f"recomp_{mac}"] = round(rc)
                row[f"claim_vs_tgt_%"] = None
                # banda target (solo sobre claim)
                if t > 0 and mac != "kcal":
                    band_cells += 1
                    if 0.90 <= cl / t <= 1.12:
                        band_in += 1
                # integridad claim vs recomputado (solo donde el día resolvió >=60%)
                if res_pct >= 60 and rc > 0:
                    integ_cells += 1
                    if abs(cl - rc) / rc <= 0.15:
                        integ_band += 1
                # [P2-VALIDATION-NO-FILTER · 2026-06-15] (G16) integridad SIN el filtro res_pct: incluye los
                # días de baja resolución que el agregado filtrado esconde (los peores casos del 0-silencioso).
                # Límite inferior aún más conservador (los ingredientes no-resueltos aportan 0 al recomputado).
                if rc > 0:
                    integ_all_cells += 1
                    if abs(cl - rc) / rc <= 0.15:
                        integ_all_band += 1
            if res_pct < 60:
                days_low_res += 1
            row["nutricionista_aprobado"] = ""
            row["notas"] = ""
            out_rows.append(row)

    # Escribe CSV.
    if out_rows:
        cols = ["plan_id", "user", "fallback", "band_score", "dia", "res_pct",
                "tgt_protein", "claim_protein", "recomp_protein",
                "tgt_carbs", "claim_carbs", "recomp_carbs",
                "tgt_fats", "claim_fats", "recomp_fats",
                "tgt_kcal", "claim_kcal", "recomp_kcal",
                "nutricionista_aprobado", "notas"]
        with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(out_rows)
        print(f"[P0-CLINICAL-VALIDATION] CSV escrito: {args.out} ({len(out_rows)} filas día)", flush=True)

    # Resumen.
    print("\n===== RESUMEN DE VALIDACIÓN =====", flush=True)
    print(f"  Días analizados: {len(out_rows)}", flush=True)
    if band_cells:
        print(f"  Precisión vs TARGET (claim en [0.90,1.12]×target, P/C/F): "
              f"{band_in}/{band_cells} = {round(100*band_in/band_cells)}%", flush=True)
    if integ_cells:
        print(f"  INTEGRIDAD (claim del LLM vs recomputado desde ingredientes, ±15%): "
              f"{integ_band}/{integ_cells} = {round(100*integ_band/integ_cells)}% "
              f"(el resto: el LLM afirma macros que sus ingredientes NO aportan)", flush=True)
    else:
        print("  INTEGRIDAD: sin días con resolución >=60% para comparar (catálogo no resolvió).", flush=True)
    # [P2-VALIDATION-NO-FILTER · 2026-06-15] (gap-audit G16) 2ª línea SIN el filtro res_pct: no esconde los
    # días de baja resolución (el 0-silencioso). La diferencia entre esta cifra y la filtrada = cuánto está
    # enmascarando el filtro. El benchmark EXTERNO (NutriBench/INCAP) sigue diferido por adquisición de dataset.
    if integ_all_cells:
        print(f"  INTEGRIDAD (SIN filtro res_pct — incluye días de baja resolución, G16): "
              f"{integ_all_band}/{integ_all_cells} = {round(100*integ_all_band/integ_all_cells)}% "
              f"(límite inferior; expone el 0-silencioso que el agregado filtrado oculta)", flush=True)
    if out_rows:
        print(f"  Días de BAJA RESOLUCIÓN (res_pct<60, excluidos del agregado filtrado): "
              f"{days_low_res}/{len(out_rows)} = {round(100*days_low_res/len(out_rows))}%", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
