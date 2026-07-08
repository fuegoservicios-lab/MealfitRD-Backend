"""[CLINICAL-ADHERENCE-REPLAY · 2026-07-07] Mide ADHERENCIA CLÍNICA (no solo precisión de macros) del
motor determinista sobre un corpus RAW capturado — SIN tokens, SIN ruido LLM, reproducible.

Complementa `macro_sizing_replay.py` (que mide |entregado−target|/target de kcal/P/C/F). Este mide si el
plan ENTREGADO respeta las restricciones clínicas de su condición, agregando micros por-día desde los
strings de ingredientes (mismo catálogo que el motor, vía `micros_from_ingredient_string`):

  • ERC (Enfermedad renal crónica): proteína ≤ 0.8 g/kg/día (KDIGO), sodio ≤ 2000mg, K ≤ 3000mg, P ≤ 1000mg.
  • DM2 (Diabetes tipo 2): fibra ≥ 14 g/1000kcal (ADA), azúcares ≤ 10% de kcal.
  • HTA (Hipertensión): sodio ≤ 2000mg/día (DASH).

Reporta, POR CONDICIÓN, el % de días que cumplen cada restricción + el valor medio entregado. Es la métrica
"negativa" que faltaba: el benchmark de macros NO detecta un plan renal que entrega 1.5 g/kg de proteína
(su target de proteína puede estar en banda, pero la RESTRICCIÓN clínica se viola).

USO (mismo corpus que macro_sizing_replay; toggle knobs entre corridas para A/B):
  MEALFIT_MACRO_SOLVER_ENABLED=true MEALFIT_MACRO_AWARE_RECONCILE=true MEALFIT_MACRO_POSTQUANT_RECONCILE=true \
    MEALFIT_MACRO_REBALANCE=true MEALFIT_NUTRITION_UNIFIED_RESOLVER=false PYTHONPATH=. \
    python scripts/clinical_adherence_replay.py /c/tmp/raw_corpus_clinical.jsonl

# [P2-LOGGER-EXEMPT: script CLI one-shot — salida humana a stdout]
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
from nutrition_db import IngredientNutritionDB


# ── umbrales clínicos (estándares; alinear con las constantes del motor si difieren) ──────────────
ERC_PROTEIN_G_PER_KG = float(os.environ.get("CLIN_ERC_PROTEIN_GKG", "0.8"))   # KDIGO no-diálisis
ERC_SODIUM_MG = float(os.environ.get("CLIN_ERC_SODIUM_MG", "2000"))
ERC_POTASSIUM_MG = float(os.environ.get("CLIN_ERC_K_MG", "3000"))
ERC_PHOSPHORUS_MG = float(os.environ.get("CLIN_ERC_P_MG", "1000"))
DM2_FIBER_PER_1000KCAL = float(os.environ.get("CLIN_DM2_FIBER_1000", "14"))   # ADA
DM2_SUGARS_PCT_KCAL = float(os.environ.get("CLIN_DM2_SUGAR_PCT", "0.10"))     # ≤10% kcal de azúcares
HTA_SODIUM_MG = float(os.environ.get("CLIN_HTA_SODIUM_MG", "2000"))           # DASH

_COND_TOKENS = {
    "erc": ("renal", "erc", "kidney", "nefro"),
    "dm2": ("diabet", "dm2", "glucem", "glicem"),
    "hta": ("hipertens", "hypertens", "hta", "presi"),
}


def _has_cond(conditions, key):
    joined = " ".join(str(c).lower() for c in (conditions or []))
    return any(tok in joined for tok in _COND_TOKENS[key])


def _num(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def _day_micros(day, db):
    """Agrega macros + micros de un día sumando micros_from_ingredient_string sobre las comidas."""
    agg = {"kcal": 0.0, "protein": 0.0, "fiber": 0.0, "sodium_mg": 0.0, "sugars_g": 0.0,
           "potassium_mg": 0.0, "phosphorus_mg": 0.0}
    for m in (day.get("meals") or []):
        agg["kcal"] += _num(m.get("cals") if m.get("cals") is not None else m.get("calories"))
        agg["protein"] += _num(m.get("protein"))
        for ing in (m.get("ingredients") or []):
            mic = None
            try:
                mic = db.micros_from_ingredient_string(str(ing))
            except Exception:
                mic = None
            if not mic:
                continue
            for k in ("fiber", "sodium_mg", "sugars_g", "potassium_mg", "phosphorus_mg"):
                v = mic.get(k)
                if v is not None:
                    agg[k] += _num(v)
    return agg


def _checks_for_day(agg, weight_kg, conditions):
    """Devuelve {check_name: (passed_bool, value)} para las condiciones presentes."""
    out = {}
    kcal = max(1.0, agg["kcal"])
    if _has_cond(conditions, "erc"):
        gkg = agg["protein"] / weight_kg if weight_kg else None
        if gkg is not None:
            out["erc_protein<=0.8gkg"] = (gkg <= ERC_PROTEIN_G_PER_KG + 0.05, round(gkg, 2))
        out["erc_sodium<=2000"] = (agg["sodium_mg"] <= ERC_SODIUM_MG, round(agg["sodium_mg"]))
        out["erc_potassium<=3000"] = (agg["potassium_mg"] <= ERC_POTASSIUM_MG, round(agg["potassium_mg"]))
        out["erc_phosphorus<=1000"] = (agg["phosphorus_mg"] <= ERC_PHOSPHORUS_MG, round(agg["phosphorus_mg"]))
    if _has_cond(conditions, "dm2"):
        fiber_1000 = agg["fiber"] / (kcal / 1000.0)
        out["dm2_fiber>=14/1000kcal"] = (fiber_1000 >= DM2_FIBER_PER_1000KCAL, round(fiber_1000, 1))
        sugar_pct = (agg["sugars_g"] * 4.0) / kcal
        out["dm2_sugars<=10%kcal"] = (sugar_pct <= DM2_SUGARS_PCT_KCAL, f"{sugar_pct*100:.1f}%")
    if _has_cond(conditions, "hta"):
        out["hta_sodium<=2000"] = (agg["sodium_mg"] <= HTA_SODIUM_MG, round(agg["sodium_mg"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    with open(args.corpus, encoding="utf-8") as f:
        states = [json.loads(ln) for ln in f if ln.strip()]
    if getattr(db_core, "connection_pool", None):
        db_core.connection_pool.open()
    db = IngredientNutritionDB()

    # acumuladores por-check: [n_pass, n_total, [valores]]
    tally = {}
    n_clinical_days = 0
    n_plans_clinical = 0
    for i, st in enumerate(states):
        result = copy.deepcopy(st["result"])
        days = result.get("days", [])
        fd = st.get("form_data", {}) or {}
        conditions = fd.get("medicalConditions") or fd.get("medical_conditions") or []
        weight = _num(fd.get("weight"))
        if not any(_has_cond(conditions, k) for k in _COND_TOKENS):
            continue  # solo perfiles clínicos
        try:
            go._apply_macro_engine(result, days, st.get("skeleton"), st["daily_cals"],
                                   st["pg"], st["cg"], st["fg"], fd, st["nutrition"])
        except Exception as e:
            print(f"  [{i}] ERROR {type(e).__name__}: {e}")
            continue
        # [A/B] el _day_sodium_autofix corre en assemble (pre/post-motor), NO en _apply_macro_engine → el
        # replay lo omite. Con REPLAY_SODIUM_AUTOFIX=1 lo corremos post-motor para medir su efecto (incl. el
        # swap lácteo P1-SODIUM-DAIRY-SWAP) sobre la adherencia de sodio.
        if os.environ.get("REPLAY_SODIUM_AUTOFIX"):
            try:
                go._day_sodium_autofix(result.get("days") or [], fd, db)
            except Exception:
                pass
        n_plans_clinical += 1
        for d in (result.get("days") or []):
            agg = _day_micros(d, db)
            checks = _checks_for_day(agg, weight, conditions)
            if not checks:
                continue
            n_clinical_days += 1
            for name, (passed, val) in checks.items():
                t = tally.setdefault(name, [0, 0, []])
                t[0] += 1 if passed else 0
                t[1] += 1
                t[2].append(val)
        if args.verbose:
            print(f"  [{i}] {conditions} w={weight}kg days={len(days)}")

    print(f"\n[clinical-adherence-replay] {n_plans_clinical} planes clínicos · {n_clinical_days} días-condición")
    print(f"{'check':32} {'%cumple':>8} {'n':>5}   valores (muestra)")
    for name in sorted(tally):
        npass, ntot, vals = tally[name]
        pct = round(100 * npass / ntot, 1) if ntot else None
        sample = ", ".join(str(v) for v in vals[:8])
        print(f"{name:32} {pct:>7}% {ntot:>5}   [{sample}]")


if __name__ == "__main__":
    main()
