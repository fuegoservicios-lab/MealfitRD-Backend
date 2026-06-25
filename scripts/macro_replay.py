"""[MACRO-REPLAY · 2026-06-19] Harness de GRABAR/REPRODUCIR para validar la precisión de macros del motor
determinista SIN gastar tokens del LLM ni sufrir su ruido.

IDEA: las respuestas del LLM son la ÚNICA fuente de costo (tokens DeepSeek) y de no-determinismo (varianza
run-to-run que hizo N=8 inmedible). Las grabamos UNA vez (cassette) y las reproducimos: el resto del pipeline
(skeleton det, solver, closer, reconcile, quantize, capa clínica, validadores deterministas) corre IDÉNTICO
sobre las MISMAS respuestas → cualquier cambio determinista (knob/algoritmo) se mide como delta PURO, gratis y
reproducible. La precisión de macros ES determinista dado el plan del LLM → esto es exactamente lo que mide.

USO:
  # 1 sola vez (cuesta tokens) — graba el corpus:
  MEALFIT_MACRO_SOLVER_ENABLED=true MEALFIT_MACRO_AWARE_RECONCILE=true MEALFIT_MACRO_POSTQUANT_RECONCILE=true \
    PYTHONPATH=. python scripts/macro_replay.py record --n 8 --out /tmp/macro_corpus.json

  # cuantas veces quieras (GRATIS, sin ruido) — reproduce + mide; toggle knobs entre corridas para A/B:
  PYTHONPATH=. python scripts/macro_replay.py replay --corpus /tmp/macro_corpus.json
  MEALFIT_PROTEIN_FLOOR_FILL_PCT=1.0 PYTHONPATH=. python scripts/macro_replay.py replay --corpus /tmp/macro_corpus.json

CAVEAT honesto: el cassette es key-by-prompt. Si tu cambio NO altera los prompts (caso típico de sizing:
la generación ocurre ANTES del sizing), todo es cache-HIT → exacto. Si un cambio hace que un día necesite una
re-corrección LLM con un prompt NUEVO (flujo divergente), es cache-MISS → se cuenta y se loguea (el pipeline
maneja el None como fallo de esa llamada). Un % de miss alto = el cambio alteró el flujo, no solo el sizing.
"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ (graph_orchestrator, schemas)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # scripts/ (benchmark_macro_compliance)

import graph_orchestrator as go
from benchmark_macro_compliance import PROFILES, MACROS, _num, _aggregate  # reusa el set fijo + el agregador

# ── Registro de tipos para reconstruir las respuestas estructuradas del LLM ──
_REGISTRY = {}
try:
    from schemas import (SingleDayPlanModel, PlanSkeletonModel, MacrosModel, MealModel,
                         DailyPlanModel, PlanModel)
    for _c in (SingleDayPlanModel, PlanSkeletonModel, MacrosModel, MealModel, DailyPlanModel, PlanModel):
        _REGISTRY[_c.__name__] = _c
except Exception as _e:
    print(f"[macro-replay] aviso: no pude importar schemas ({_e})")
for _nm in ("AdversarialJudgeResult", "CritiqueEvaluation"):
    _c = getattr(go, _nm, None)
    if _c is not None:
        _REGISTRY[_nm] = _c


# ── (de)serialización del resultado de _safe_ainvoke ──
def _ser(result):
    if result is None:
        return {"__t__": "none"}
    if hasattr(result, "model_dump"):
        try:
            return {"__t__": type(result).__name__, "__d__": result.model_dump()}
        except Exception:
            pass
    if hasattr(result, "content"):  # AIMessage / similar
        return {"__t__": "AIMessage", "content": getattr(result, "content", "")}
    if isinstance(result, (str, int, float, bool, dict, list)):
        return {"__t__": "raw", "v": result}
    return {"__t__": "str", "v": str(result)}


def _deser(d):
    t = d.get("__t__")
    if t == "none":
        return None
    if t == "raw":
        return d["v"]
    if t in ("str",):
        return d["v"]
    if t == "AIMessage":
        try:
            from langchain_core.messages import AIMessage
            return AIMessage(content=d.get("content", ""))
        except Exception:
            return d.get("content", "")
    cls = _REGISTRY.get(t)
    if cls is not None:
        try:
            return cls.model_validate(d["__d__"])
        except Exception as e:
            print(f"[macro-replay] reconstrucción {t} falló: {e}")
            return None
    return d.get("__d__")


def _payload_str(payload):
    if isinstance(payload, str):
        return payload
    if isinstance(payload, (list, tuple)):
        return "\n".join((getattr(m, "content", None) or str(m)) for m in payload)
    return getattr(payload, "content", None) or str(payload)


def _key(payload):
    return hashlib.sha256(_payload_str(payload).encode("utf-8", "ignore")).hexdigest()


# ── medición de un perfil (copia la lógica de days_dev del benchmark) ──
async def _measure(profile):
    pid = profile["_id"]
    # [MACRO-REPLAY] Siembra DETERMINISTA del módulo `random` por-perfil: el prompt del planner/generador
    # incluye `random.randint` (semilla) + `random.choices` (técnicas) → sin sembrar, el prompt cambia cada
    # llamada y el cassette (key-by-prompt) nunca matchea. Sembrar fija la aleatoriedad → prompt idéntico entre
    # record y replay → cache HIT. EXIGE concurrencia=1 (el `random` global es compartido; conc>1 intercala las
    # llamadas de perfiles distintos y rompe el determinismo).
    random.seed(pid)
    fd = {k: v for k, v in profile.items() if k != "_id"}
    try:
        r = await go.arun_plan_pipeline(dict(fd))
    except Exception as e:
        return {"id": pid, "error": f"{type(e).__name__}: {e}"}
    macros = r.get("macros") or {}
    target = {"kcal": _num(r.get("calories")), "protein": _num(macros.get("protein")),
              "carbs": _num(macros.get("carbs")), "fats": _num(macros.get("fats"))}
    days_dev = []
    for d in (r.get("days") or []):
        deliv = {"kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fats": 0.0}
        for m in (d.get("meals") or []):
            deliv["kcal"] += _num(m.get("cals") if m.get("cals") is not None else m.get("calories"))
            deliv["protein"] += _num(m.get("protein"))
            deliv["carbs"] += _num(m.get("carbs"))
            deliv["fats"] += _num(m.get("fats"))
        dev = {mac: (((deliv[mac] - target[mac]) / target[mac]) if target[mac] > 0 else None) for mac in MACROS}
        days_dev.append({"delivered": {k: round(v) for k, v in deliv.items()}, "dev": dev})
    return {"id": pid, "gender": profile["gender"], "goal": profile["mainGoal"],
            "conditions": profile["medicalConditions"], "target": {k: round(v) for k, v in target.items()},
            "is_fallback": bool(r.get("_is_fallback")), "num_days": len(days_dev), "days_dev": days_dev}


async def _open_pools():
    import db_core
    if getattr(db_core, "connection_pool", None):
        db_core.connection_pool.open()
    if getattr(db_core, "async_connection_pool", None):
        await db_core.async_connection_pool.open()
    await asyncio.sleep(1.5)


def _print_aggregate(agg):
    r = agg.get("REAL_PLANS", {}) or {}
    pm = r.get("per_macro", {}) or {}
    print(f"  n_real={agg.get('n_real')} fallback={agg.get('fallback_rate_pct')}%  "
          f"all4-within10={r.get('all4_within_10pct_days_pct')}")
    for mac in MACROS:
        v = pm.get(mac, {}) or {}
        print(f"    {mac:8} MAPE {v.get('mape_pct')}  within10 {v.get('within_10pct')}  within15 {v.get('within_15pct')}")


async def _run(profiles, conc):
    sem = asyncio.Semaphore(conc)

    async def _one(p):
        async with sem:
            return await _measure(p)
    return await asyncio.gather(*[_one(p) for p in profiles])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "replay"])
    ap.add_argument("--n", type=int, default=len(PROFILES))
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--out", default="/tmp/macro_corpus.json")
    ap.add_argument("--corpus", default="/tmp/macro_corpus.json")
    args = ap.parse_args()
    if args.concurrency != 1:
        print("[macro-replay] forzando concurrency=1 (requerido para el determinismo del cassette)")
        args.concurrency = 1
    profiles = PROFILES[: args.n]
    await _open_pools()

    cassette = {}
    hits = {"h": 0, "m": 0}
    _orig = go._safe_ainvoke

    if args.mode == "record":
        previews = []
        async def _recording(llm, payload, *, timeout):
            res = await _orig(llm, payload, timeout=timeout)
            try:
                k = _key(payload)
                cassette[k] = _ser(res)
                previews.append({"k": k[:12], "p": _payload_str(payload)[:400]})
            except Exception as e:
                print(f"[macro-replay] no pude serializar una respuesta: {e}")
            return res
        go._safe_ainvoke = _recording
        print(f"[macro-replay] RECORD {len(profiles)} perfiles (conc {args.concurrency}) — esto SÍ cuesta tokens...")
        results = await _run(profiles, args.concurrency)
        agg = _aggregate(results)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"cassette": cassette, "baseline_aggregate": agg, "baseline_results": results,
                       "previews": previews, "n": args.n}, f, ensure_ascii=False)
        print(f"[macro-replay] cassette guardado: {args.out}  ({len(cassette)} respuestas LLM)")
        print("[macro-replay] AGREGADO BASELINE (lo que el LLM produjo, sized con la config de grabación):")
        _print_aggregate(agg)
    else:  # replay
        with open(args.corpus, encoding="utf-8") as f:
            data = json.load(f)
        saved = data["cassette"]

        _miss_logged = {"n": 0}
        async def _replaying(llm, payload, *, timeout):
            k = _key(payload)
            if k in saved:
                hits["h"] += 1
                return _deser(saved[k])
            hits["m"] += 1
            if _miss_logged["n"] < 2:
                _miss_logged["n"] += 1
                print(f"[macro-replay][MISS#{_miss_logged['n']}] key={k[:12]} payload[:400]=\n  {_payload_str(payload)[:400]!r}")
            return None  # cache-miss: el pipeline trata el None como fallo de esa llamada LLM
        go._safe_ainvoke = _replaying
        print(f"[macro-replay] REPLAY {len(profiles)} perfiles desde {args.corpus} — GRATIS (0 tokens LLM)")
        results = await _run(profiles, args.concurrency)
        agg = _aggregate(results)
        print(f"[macro-replay] cache: {hits['h']} HITS / {hits['m']} MISSES "
              f"({round(100*hits['m']/max(1,hits['h']+hits['m']))}% miss)")
        print("[macro-replay] AGREGADO (config actual de knobs sobre las MISMAS respuestas del LLM):")
        _print_aggregate(agg)
        base = data.get("baseline_aggregate")
        if base:
            print("[macro-replay] (referencia: agregado de grabación)")
            _print_aggregate(base)
    go._safe_ainvoke = _orig


if __name__ == "__main__":
    asyncio.run(main())
