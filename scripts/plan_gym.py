# [P2-LOGGER-EXEMPT: CLI de gym — salida humana a stdout por diseño]
"""[P1-NEXT-LEVEL-BATCH · 2026-07-02] Gym CLI — evaluación multi-eje E2E multi-perfil.

Genera N planes reales (LLM + pipeline completo) con los 20 perfiles held-out del
benchmark de macros y los puntúa con `plan_gym.score_plan` en los 8 ejes del producto.
Convierte cada mejora del motor en un número comparable ANTES de tocar prod, y produce
la "serie de datos" que los flips de knobs esperan (SODIUM_EXCESS / RECIPE_CONTRACT /
MICRO_CLOSER_PERDAY) en horas en vez de semanas.

Uso (desde backend/, con .env cargable y DEEPSEEK_API_KEY):
    python scripts/plan_gym.py            # los 20 perfiles
    python scripts/plan_gym.py 3          # solo los primeros 3 (smoke)
    python scripts/plan_gym.py 3 --conc 2 # concurrencia (default 2)

Salida: tabla humana a stdout + JSON completo a GYM_OUT (default gym_report_<pid>.json
en el cwd) para diffear entre corridas (A/B de knobs).
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except Exception:
    pass

from benchmark_macro_compliance import PROFILES  # mismo set held-out (20 perfiles fijos)
from plan_gym import score_plan, aggregate_scores


async def _run_one(profile, sem):
    from graph_orchestrator import arun_plan_pipeline
    async with sem:
        pid = profile["_id"]
        fd = {k: v for k, v in profile.items() if k != "_id"}
        t0 = time.time()
        try:
            plan = await arun_plan_pipeline(dict(fd))
        except Exception as e:
            return {"id": pid, "error": f"{type(e).__name__}: {e}"}
        dur = round(time.time() - t0, 1)
        try:
            score = score_plan(plan, fd)
        except Exception as e:
            return {"id": pid, "error": f"score: {type(e).__name__}: {e}", "duration_s": dur}
        return {"id": pid, "goal": profile.get("mainGoal"), "conditions": profile.get("medicalConditions"),
                "duration_s": dur, "score": score}


async def _main(n: int, conc: int) -> dict:
    # Los pools nacen con open=False (se abren en el lifespan de la app); un script
    # standalone debe abrirlos explícitamente o shopping/costo degradan a PoolClosed.
    try:
        from db_core import connection_pool
        if connection_pool is not None:
            connection_pool.open()
    except Exception as _pe:
        print(f"(aviso) pool sync no disponible: {_pe} — ejes de costo/lista degradarán")
    try:
        from db_core import async_connection_pool
        if async_connection_pool is not None:
            await async_connection_pool.open()
    except Exception:
        pass
    sem = asyncio.Semaphore(conc)
    profiles = PROFILES[:n] if n else PROFILES
    results = await asyncio.gather(*[_run_one(p, sem) for p in profiles])
    return {"results": list(results), "aggregate": aggregate_scores(list(results))}


def main():
    # Consolas Windows default a cp1252 → los glifos del resumen crasheaban el print
    # DESPUÉS de gastar la corrida LLM completa (incidente smoke 2026-07-02).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=0, help="límite de perfiles (0 = todos)")
    ap.add_argument("--conc", type=int, default=2)
    args = ap.parse_args()

    out = asyncio.run(_main(args.n, max(1, args.conc)))
    agg = out["aggregate"]

    # El JSON se escribe ANTES del resumen humano: si un print falla, la corrida
    # (que cuesta minutos de LLM) ya quedó persistida.
    out_path = os.environ.get("GYM_OUT") or f"gym_report_{os.getpid()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)

    print("\n========== PLAN GYM - resumen ==========")
    print(f"perfiles: {agg.get('n', 0)} | global medio: {agg.get('global_mean')}")
    for ax in ("banda", "micros", "slots", "creatividad", "coherencia", "presupuesto", "entrega"):
        s = agg.get(ax)
        if s:
            print(f"  {ax:<12} mean={s['mean']:>5}  min={s['min']:>5}  (n={s['n']})")
    errs = [r for r in out["results"] if r.get("error")]
    if errs:
        print(f"  errores: {len(errs)} -> {[(e['id'], e['error'][:60]) for e in errs]}")
    for w in agg.get("worst_profiles", []):
        print(f"  peor: perfil {w['id']} global={w['global']} ejes flojos={w['axes']}")
    print(f"\nJSON completo: {out_path}")


if __name__ == "__main__":
    main()
