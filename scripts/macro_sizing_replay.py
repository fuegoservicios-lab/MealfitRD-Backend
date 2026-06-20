"""[MACRO-SIZING-REPLAY · 2026-06-19] Validación de la PRECISIÓN del motor de macros SIN tokens y SIN ruido.

A diferencia del cassette del pipeline completo (descartado: el paralelismo intra-perfil + los prompts del review
dependientes del plan lo hacen no-fiel), esto reproduce SOLO el motor determinista `_apply_macro_engine`
(solver→closer→reconcile→quantize→capa clínica) sobre los PLANES CRUDOS que el LLM generó — capturados una vez
por el hook MEALFIT_MACRO_CAPTURE en assemble. Sin LLM, sin review, sin reintentos, sin paralelismo → FIEL,
GRATIS, REPRODUCIBLE. Cualquier cambio de knob/algoritmo del sizing se mide como delta PURO.

USO:
  # 1 sola vez (cuesta tokens) — captura los planes crudos corriendo el benchmark normal con el hook:
  rm -f /tmp/raw_corpus.jsonl
  MEALFIT_MACRO_CAPTURE=/tmp/raw_corpus.jsonl MEALFIT_MACRO_SOLVER_ENABLED=true MEALFIT_MACRO_AWARE_RECONCILE=true \
    MEALFIT_MACRO_POSTQUANT_RECONCILE=true PYTHONPATH=. python scripts/benchmark_macro_compliance.py 8 --concurrency 2

  # cuantas veces quieras (GRATIS) — reproduce el sizing + mide; toggle knobs entre corridas para A/B:
  MEALFIT_MACRO_SOLVER_ENABLED=true MEALFIT_MACRO_AWARE_RECONCILE=true MEALFIT_MACRO_POSTQUANT_RECONCILE=true \
    PYTHONPATH=. python scripts/macro_sizing_replay.py /tmp/raw_corpus.jsonl
  MEALFIT_PROTEIN_FLOOR_FILL_PCT=1.0 (...mismos knobs...) python scripts/macro_sizing_replay.py /tmp/raw_corpus.jsonl
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # backend/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # scripts/

import db_core
import graph_orchestrator as go
from benchmark_macro_compliance import MACROS, _num, _aggregate


def _measure(result, daily_cals, pg, cg, fg):
    target = {"kcal": daily_cals, "protein": pg, "carbs": cg, "fats": fg}
    days_dev = []
    for d in (result.get("days") or []):
        deliv = {k: 0.0 for k in MACROS}
        for m in (d.get("meals") or []):
            deliv["kcal"] += _num(m.get("cals") if m.get("cals") is not None else m.get("calories"))
            deliv["protein"] += _num(m.get("protein"))
            deliv["carbs"] += _num(m.get("carbs"))
            deliv["fats"] += _num(m.get("fats"))
        dev = {mac: (((deliv[mac] - target[mac]) / target[mac]) if target[mac] > 0 else None) for mac in MACROS}
        days_dev.append({"delivered": {k: round(v) for k, v in deliv.items()}, "dev": dev})
    return days_dev, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    args = ap.parse_args()
    with open(args.corpus, encoding="utf-8") as f:
        states = [json.loads(ln) for ln in f if ln.strip()]
    if getattr(db_core, "connection_pool", None):
        db_core.connection_pool.open()   # el sizing lee master_ingredients (sync); NO usa LLM

    results = []
    for i, st in enumerate(states):
        result = copy.deepcopy(st["result"])
        days = result.get("days", [])
        try:
            go._apply_macro_engine(result, days, st.get("skeleton"), st["daily_cals"],
                                   st["pg"], st["cg"], st["fg"], st["form_data"], st["nutrition"])
        except Exception as e:
            results.append({"id": i, "error": f"{type(e).__name__}: {e}"})
            continue
        days_dev, target = _measure(result, st["daily_cals"], st["pg"], st["cg"], st["fg"])
        fd = st.get("form_data", {})
        results.append({"id": i, "gender": fd.get("gender", "?"), "goal": fd.get("mainGoal", "?"),
                        "conditions": fd.get("medicalConditions", []),
                        "target": {k: round(v) for k, v in target.items()},
                        "is_fallback": False, "num_days": len(days_dev), "days_dev": days_dev})

    agg = _aggregate(results)
    r = agg.get("REAL_PLANS", {}) or {}
    pm = r.get("per_macro", {}) or {}
    _kn = {k: os.environ.get(k) for k in ("MEALFIT_PROTEIN_FLOOR_FILL_PCT", "MEALFIT_PROTEIN_TARGET_CLOSE",
                                          "MEALFIT_CARB_TARGET_TRIM") if os.environ.get(k)}
    print(f"[macro-sizing-replay] {len(states)} planes crudos · knobs override: {_kn or '(defaults)'}")
    print(f"  n_real={agg.get('n_real')} errors={agg.get('n_errors')}  all4-within10={r.get('all4_within_10pct_days_pct')}")
    for mac in MACROS:
        v = pm.get(mac, {}) or {}
        print(f"    {mac:8} MAPE {v.get('mape_pct')}  within10 {v.get('within_10pct')}  within15 {v.get('within_15pct')}")


if __name__ == "__main__":
    main()
