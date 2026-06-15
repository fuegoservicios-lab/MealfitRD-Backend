"""[M2-MACRO-BENCHMARK · 2026-06-14] Benchmark REPRODUCIBLE de precisión de macros.

Mide qué tan cerca quedan los macros ENTREGADOS (suma por-día de las comidas) del TARGET que el
sistema computó, end-to-end (skeleton → generación LLM → solver → assemble → review). Convierte la
banda "90-112%" AUTOAFIRMADA en un número MEDIDO sobre un set diverso y held-out de perfiles.

QUÉ MIDE (y qué NO):
  • SÍ: precisión de ENTREGA — ¿el plan entregado cumple su propio target de kcal+P+C+F dentro de ±X%?
    Esto es lo que el solver determinista garantiza y es la afirmación de "precisión de macros".
  • NO: que el TARGET sea clínicamente óptimo (eso depende de TDEE/condición, fuera de este benchmark).
  • NO: comparación corriendo herramientas de competidores (se citan sus errores PUBLICADOS, no se re-corren).

Honestidad: el set de perfiles es FIJO y diverso (no cherry-picked); los planes de fallback matemático
se reportan por separado (son deterministas → sesgarían el número del LLM hacia arriba si se mezclan).

Uso:  python scripts/benchmark_macro_compliance.py [N]   # N = limitar nº de perfiles (default: todos)
Salida: resumen humano a stdout + JSON completo a /tmp/macro_benchmark_<ts>.json (pasar ts por env BENCH_TS).
"""
import argparse
import asyncio
import json
import os
import re
import sys

from graph_orchestrator import arun_plan_pipeline


def _num(x) -> float:
    try:
        s = str(x).lower().replace("kcal", "").replace("g", "").strip()
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else 0.0
    except Exception:
        return 0.0


# Set FIJO y diverso de perfiles held-out (no cherry-picked): cubre género, edad, peso, objetivo,
# actividad y un subconjunto con condiciones. Determinista → reproducible.
def _profile(idx, gender, age, weight, height, goal, activity, conditions):
    return {
        "_id": idx, "age": age, "weight": weight, "height": height, "gender": gender,
        "weightUnit": "kg", "mainGoal": goal, "activityLevel": activity,
        "householdSize": 1, "groceryDuration": "weekly",
        "motivation": "Mejorar mi salud de forma sostenible.",
        "allergies": ["Ninguna"], "medicalConditions": conditions,
        "scheduleType": "standard", "cookingTime": "30min", "budget": "medium",
        "sleepHours": "7-8 horas", "stressLevel": "Moderado",
        "dislikes": ["Ninguno"], "struggles": ["Ninguno"], "user_id": "guest",
    }


PROFILES = [
    _profile(1,  "male",   22, 60,  172, "gain_muscle",  "active",    ["Ninguna"]),
    _profile(2,  "male",   28, 75,  178, "gain_muscle",  "moderate",  ["Ninguna"]),
    _profile(3,  "male",   35, 90,  180, "lose_fat",     "moderate",  ["Ninguna"]),
    _profile(4,  "male",   45, 105, 175, "lose_fat",     "sedentary", ["Ninguna"]),
    _profile(5,  "male",   55, 82,  170, "maintenance",  "moderate",  ["Ninguna"]),
    _profile(6,  "male",   65, 78,  168, "maintenance",  "sedentary", ["Ninguna"]),
    _profile(7,  "female", 24, 52,  160, "gain_muscle",  "active",    ["Ninguna"]),
    _profile(8,  "female", 30, 58,  163, "lose_fat",     "moderate",  ["Ninguna"]),
    _profile(9,  "female", 38, 70,  165, "lose_fat",     "sedentary", ["Ninguna"]),
    _profile(10, "female", 48, 85,  168, "lose_fat",     "moderate",  ["Ninguna"]),
    _profile(11, "female", 58, 65,  158, "maintenance",  "moderate",  ["Ninguna"]),
    _profile(12, "female", 33, 62,  167, "maintenance",  "active",    ["Ninguna"]),
    # Perfiles con condición (prueban que el cap/guard no rompen la precisión del target ajustado)
    _profile(13, "male",   50, 88,  174, "lose_fat",     "sedentary", ["Diabetes tipo 2"]),
    _profile(14, "female", 44, 80,  162, "lose_fat",     "moderate",  ["Diabetes tipo 2"]),
    _profile(15, "male",   60, 84,  172, "maintenance",  "sedentary", ["Hipertensión"]),
    _profile(16, "female", 52, 75,  160, "lose_fat",     "moderate",  ["Hipertensión"]),
    _profile(17, "male",   58, 80,  170, "maintenance",  "sedentary", ["Enfermedad renal crónica"]),
    _profile(18, "female", 40, 68,  164, "maintenance",  "moderate",  ["Diabetes tipo 2", "Hipertensión"]),
    _profile(19, "male",   26, 70,  176, "performance",  "active",    ["Ninguna"]),
    _profile(20, "female", 36, 60,  166, "gain_muscle",  "moderate",  ["Ninguna"]),
]

BANDS = (0.10, 0.15, 0.20)
MACROS = ("kcal", "protein", "carbs", "fats")


async def _bench_one(profile, sem):
    async with sem:
        pid = profile["_id"]
        fd = {k: v for k, v in profile.items() if k != "_id"}
        try:
            r = await arun_plan_pipeline(dict(fd))
        except Exception as e:
            return {"id": pid, "error": f"{type(e).__name__}: {e}"}
        macros = r.get("macros") or {}
        target = {
            "kcal": _num(r.get("calories")),
            "protein": _num(macros.get("protein")),
            "carbs": _num(macros.get("carbs")),
            "fats": _num(macros.get("fats")),
        }
        is_fallback = bool(r.get("_is_fallback"))
        days_dev = []
        for d in (r.get("days") or []):
            delivered = {"kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fats": 0.0}
            for m in (d.get("meals") or []):
                delivered["kcal"] += _num(m.get("cals") if m.get("cals") is not None else m.get("calories"))
                delivered["protein"] += _num(m.get("protein"))
                delivered["carbs"] += _num(m.get("carbs"))
                delivered["fats"] += _num(m.get("fats"))
            dev = {}
            for mac in MACROS:
                t = target[mac]
                dev[mac] = ((delivered[mac] - t) / t) if t > 0 else None
            days_dev.append({"delivered": {k: round(v) for k, v in delivered.items()}, "dev": dev})
        return {"id": pid, "gender": profile["gender"], "goal": profile["mainGoal"],
                "conditions": profile["medicalConditions"], "target": {k: round(v) for k, v in target.items()},
                "is_fallback": is_fallback, "num_days": len(days_dev), "days_dev": days_dev}


def _aggregate(results):
    real, fb, errs = [], [], []
    for res in results:
        if res.get("error"):
            errs.append(res)
        elif res.get("is_fallback"):
            fb.append(res)
        else:
            real.append(res)

    def _stats(rows):
        # MAPE por macro + % de días en cada banda + % de días con TODOS los macros en ±10%
        per = {mac: {"abs_devs": [], "in": {b: 0 for b in BANDS}, "n": 0} for mac in MACROS}
        all10_days = all10_ok = 0
        for res in rows:
            for day in res["days_dev"]:
                all_ok = True
                counted = False
                for mac in MACROS:
                    dv = day["dev"].get(mac)
                    if dv is None:
                        continue
                    counted = True
                    per[mac]["abs_devs"].append(abs(dv))
                    per[mac]["n"] += 1
                    for b in BANDS:
                        if abs(dv) <= b:
                            per[mac]["in"][b] += 1
                    if abs(dv) > 0.10:
                        all_ok = False
                if counted:
                    all10_days += 1
                    if all_ok:
                        all10_ok += 1
        out = {"n_plans": len(rows),
               "all4_within_10pct_days_pct": round(100 * all10_ok / all10_days, 1) if all10_days else None,
               "per_macro": {}}
        for mac in MACROS:
            n = per[mac]["n"]
            mape = round(100 * sum(per[mac]["abs_devs"]) / n, 1) if n else None
            out["per_macro"][mac] = {
                "mape_pct": mape,
                "within_10pct": round(100 * per[mac]["in"][0.10] / n, 1) if n else None,
                "within_15pct": round(100 * per[mac]["in"][0.15] / n, 1) if n else None,
                "within_20pct": round(100 * per[mac]["in"][0.20] / n, 1) if n else None,
            }
        return out

    return {
        "n_total": len(results), "n_real": len(real), "n_fallback": len(fb), "n_errors": len(errs),
        "fallback_rate_pct": round(100 * len(fb) / max(1, len(results)), 1),
        "REAL_PLANS": _stats(real),
        "FALLBACK_PLANS": _stats(fb),
        "errors": [{"id": e["id"], "error": e["error"]} for e in errs],
    }


def _assert_no_regression(agg, baseline_path, max_mape_rise, max_band_drop):
    """[gap-audit G13 · 2026-06-15] Gate de no-regresión: compara el agregado REAL_PLANS contra un baseline
    JSON commiteado. ok=False si all-4-en-banda cae > max_band_drop pts O si la MAPE de algún macro sube
    > max_mape_rise pts. Lo usa el job nightly (.github/workflows/macro-benchmark-nightly.yml) para fallar
    el build ante una regresión de precisión (cambio de modelo/solver/catálogo) que hoy es invisible hasta
    un audit manual. Retorna (ok, lines)."""
    lines, ok = [], True
    try:
        with open(baseline_path, encoding="utf-8") as f:
            base = json.load(f)
    except Exception as e:
        return False, [f"  ❌ no pude leer baseline {baseline_path}: {e}"]
    cur = (agg or {}).get("REAL_PLANS") or {}
    bse = (base or {}).get("REAL_PLANS") or {}
    cb, bb = cur.get("all4_within_10pct_days_pct"), bse.get("all4_within_10pct_days_pct")
    if cb is not None and bb is not None:
        drop = bb - cb
        if drop > max_band_drop:
            ok = False
        lines.append(f"  {'❌' if drop > max_band_drop else '✓'} all-4-en-banda: baseline {bb} → actual {cb} "
                     f"(caída {drop:.1f}pts, máx {max_band_drop})")
    for mac in ("kcal", "protein", "carbs", "fats"):
        cm = ((cur.get("per_macro") or {}).get(mac) or {}).get("mape_pct")
        bm = ((bse.get("per_macro") or {}).get(mac) or {}).get("mape_pct")
        if cm is not None and bm is not None:
            rise = cm - bm
            if rise > max_mape_rise:
                ok = False
            lines.append(f"  {'❌' if rise > max_mape_rise else '✓'} {mac} MAPE: baseline {bm} → actual {cm} "
                         f"(subió {rise:.1f}pts, máx {max_mape_rise})")
    return ok, lines


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=len(PROFILES))
    ap.add_argument("--concurrency", type=int, default=3)
    # [gap-audit G13 · 2026-06-15] Gate de no-regresión vs baseline commiteado (para el job nightly).
    ap.add_argument("--baseline", default=None, help="JSON baseline; activa el gate de no-regresión")
    ap.add_argument("--max-mape-rise", type=float, default=5.0, help="subida máx permitida de MAPE (pts)")
    ap.add_argument("--max-band-drop", type=float, default=10.0, help="caída máx permitida de all-4-en-banda (pts)")
    args = ap.parse_args()
    profiles = PROFILES[: args.n]
    sem = asyncio.Semaphore(args.concurrency)

    # [P0-CLINICAL-VALIDATION · 2026-06-14] Abre los pools de Neon (como el lifespan de FastAPI en
    # app.py): los pools se crean con open=False; sin abrirlos, arun_plan_pipeline no puede leer
    # master_ingredients → PoolClosed → TODO el benchmark cae a fallback y mide el plan matemático,
    # no el path REAL del LLM+solver. Bug latente del harness standalone — sin esto los números M2
    # estarían sesgados al fallback determinista. Idempotente.
    import db_core
    if getattr(db_core, "connection_pool", None):
        db_core.connection_pool.open()
    if getattr(db_core, "async_connection_pool", None):
        await db_core.async_connection_pool.open()
    await asyncio.sleep(1.5)

    print(f"[M2-MACRO-BENCHMARK] Corriendo {len(profiles)} perfiles (concurrencia {args.concurrency})...")
    results = await asyncio.gather(*[_bench_one(p, sem) for p in profiles])
    agg = _aggregate(results)
    ts = os.environ.get("BENCH_TS", "manual")
    out_path = f"/tmp/macro_benchmark_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "raw": results}, f, ensure_ascii=False, indent=2)
    print("\n===== RESUMEN (banda = |entregado-target|/target) =====")
    print(json.dumps(agg, ensure_ascii=False, indent=2, default=str))
    print(f"\n[M2-MACRO-BENCHMARK] JSON completo: {out_path}")

    # [gap-audit G13 · 2026-06-15] Gate de no-regresión vs baseline commiteado (job nightly).
    if args.baseline:
        ok, lines = _assert_no_regression(agg, args.baseline, args.max_mape_rise, args.max_band_drop)
        print("\n===== NO-REGRESIÓN vs BASELINE (gap-audit G13) =====")
        for ln in lines:
            print(ln)
        if not ok:
            print("[M2-MACRO-BENCHMARK] ❌ REGRESIÓN de precisión detectada vs baseline — fallando el build.")
            sys.exit(1)
        print("[M2-MACRO-BENCHMARK] ✅ Sin regresión de precisión vs baseline.")


if __name__ == "__main__":
    asyncio.run(main())
