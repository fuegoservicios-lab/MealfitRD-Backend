"""[P1-NEXT-LEVEL-BATCH · 2026-07-02] Gym de evaluación — scoring multi-eje de un plan generado.

El nightly benchmark (scripts/benchmark_macro_compliance.py) mide UN eje (banda macro).
Este módulo puntúa un plan entregado en los 8 ejes del objetivo del producto y produce un
score global 0-100 — convierte cada mejora del motor en MEDIBLE y da la "serie de datos"
que los flips de knobs esperan (SODIUM_EXCESS / RECIPE_CONTRACT / MICRO_CLOSER_PERDAY) en
horas de gym en vez de semanas de prod.

Diseño:
  - `score_plan(plan, form_data)` es PURO sobre el dict del plan (no LLM, no DB obligatoria):
    cada eje se computa best-effort con los MISMOS SSOT del motor (slot SSOT de constants,
    detectores de dish-quality/raw-staple del orquestador, panel de micros persistido,
    reconciliación de presupuesto persistida). Un eje sin datos devuelve None y queda FUERA
    del promedio ponderado (no castiga ni infla).
  - CLI multi-perfil: scripts/plan_gym.py (reusa los 20 perfiles held-out del benchmark).

Ejes y pesos (suman 1.0 entre los disponibles, re-normalizados):
  banda .25 · micros .15 · slots .15 · creatividad .15 · coherencia .10 · presupuesto .10 · entrega .10

tooltip-anchor: P1-NEXT-LEVEL-GYM. Test: test_p1_next_level_batch.py.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_AXIS_WEIGHTS = {
    "banda": 0.25,
    "micros": 0.15,
    "slots": 0.15,
    "creatividad": 0.15,
    "coherencia": 0.10,
    "presupuesto": 0.10,
    "entrega": 0.10,
}

_BAND_LO, _BAND_HI = 0.90, 1.12
_KCAL_LO, _KCAL_HI = 0.95, 1.05


def _num(x) -> float:
    try:
        s = str(x).lower().replace("kcal", "").replace("g", "").strip()
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else 0.0
    except Exception:
        return 0.0


def _clamp(v: float) -> float:
    return max(0.0, min(100.0, v))


def _score_banda(plan: dict) -> dict | None:
    """% de celdas (día × macro) dentro de banda + % de días all-4."""
    days = plan.get("days") or []
    macros = plan.get("macros") or {}
    targets = {
        "kcal": _num(plan.get("calories")),
        "protein": _num(macros.get("protein")),
        "carbs": _num(macros.get("carbs")),
        "fats": _num(macros.get("fats")),
    }
    if not days or not all(targets.get(k, 0) > 0 for k in targets):
        return None
    cells_in = cells_total = 0
    all4_days = 0
    for d in days:
        meals = (d.get("meals") or []) if isinstance(d, dict) else []
        if not meals:
            continue
        delivered = {
            "kcal": sum(_num(m.get("cals") if m.get("cals") is not None else m.get("calories")) for m in meals),
            "protein": sum(_num(m.get("protein")) for m in meals),
            "carbs": sum(_num(m.get("carbs")) for m in meals),
            "fats": sum(_num(m.get("fats")) for m in meals),
        }
        day_ok = True
        for mac, tgt in targets.items():
            ratio = delivered[mac] / tgt
            lo, hi = (_KCAL_LO, _KCAL_HI) if mac == "kcal" else (_BAND_LO, _BAND_HI)
            cells_total += 1
            if lo <= ratio <= hi:
                cells_in += 1
            else:
                day_ok = False
        if day_ok:
            all4_days += 1
    if not cells_total:
        return None
    pct_cells = cells_in / cells_total
    pct_all4 = all4_days / max(1, len(days))
    return {"score": _clamp(pct_cells * 100.0), "cells_in_band_pct": round(pct_cells, 3),
            "all4_days_pct": round(pct_all4, 3)}


def _score_micros(plan: dict) -> dict | None:
    mn = plan.get("micronutrient_report")
    if not isinstance(mn, dict):
        return None
    gaps_bajo = sum(1 for g in (mn.get("gaps") or []) if g.get("status") == "bajo")
    gaps_alto = sum(1 for g in (mn.get("gaps") or []) if g.get("status") == "alto")
    pdf = mn.get("per_day_floors") or {}
    pdc = mn.get("per_day_ceilings") or {}
    score = 100.0 - 10.0 * gaps_bajo - 12.0 * gaps_alto
    if isinstance(pdf, dict) and pdf.get("flagged"):
        score -= 15.0
    if isinstance(pdc, dict) and pdc.get("flagged"):
        score -= 15.0
    return {"score": _clamp(score), "gaps_bajo": gaps_bajo, "gaps_alto": gaps_alto,
            "worst_day_floor_flag": bool(pdf.get("flagged")), "worst_day_ceiling_flag": bool(pdc.get("flagged")),
            "coverage": mn.get("coverage")}


def _score_slots(plan: dict) -> dict | None:
    try:
        from constants import canonical_slot_key, slot_violations_for_meal_name
    except Exception:
        return None
    hard = soft = evaluated = 0
    for d in plan.get("days") or []:
        for m in (d.get("meals") or []) if isinstance(d, dict) else []:
            if not isinstance(m, dict):
                continue
            slot = canonical_slot_key(m.get("meal") or m.get("time") or "")
            if not slot:
                continue
            evaluated += 1
            try:
                for v in slot_violations_for_meal_name(m.get("name", ""), slot) or []:
                    if isinstance(v, dict) and v.get("hard"):
                        hard += 1
                    else:
                        soft += 1
            except Exception:
                continue
    if not evaluated:
        return None
    return {"score": _clamp(100.0 - 30.0 * hard - 12.0 * soft),
            "hard_violations": hard, "soft_violations": soft, "meals_evaluated": evaluated}


def _score_creatividad(plan: dict) -> dict | None:
    days = plan.get("days") or []
    meals_flat = [m for d in days if isinstance(d, dict) for m in (d.get("meals") or []) if isinstance(m, dict)]
    if not meals_flat:
        return None
    raw_staple = low_quality = 0
    try:
        from graph_orchestrator import _meal_raw_staple_issue, _meal_dish_quality_issue
        for m in meals_flat:
            try:
                if _meal_raw_staple_issue(m)[0]:
                    raw_staple += 1
            except Exception:
                pass
            try:
                if _meal_dish_quality_issue(m)[0]:
                    low_quality += 1
            except Exception:
                pass
    except Exception:
        return None
    n = len(meals_flat)
    raw_ratio = raw_staple / n
    lowq_ratio = low_quality / n
    # variedad cross-día: nombres únicos / total (repetir el mismo plato en varios días baja el score)
    names = [str(m.get("name", "")).strip().lower() for m in meals_flat if str(m.get("name", "")).strip()]
    uniq_ratio = (len(set(names)) / len(names)) if names else 1.0
    score = 100.0 - 45.0 * raw_ratio - 35.0 * lowq_ratio - 25.0 * (1.0 - uniq_ratio)
    return {"score": _clamp(score), "raw_staple_ratio": round(raw_ratio, 3),
            "low_quality_ratio": round(lowq_ratio, 3), "unique_dish_ratio": round(uniq_ratio, 3)}


def _score_coherencia(plan: dict) -> dict | None:
    days = plan.get("days") or []
    meals_flat = [m for d in days if isinstance(d, dict) for m in (d.get("meals") or []) if isinstance(m, dict)]
    if not meals_flat:
        return None
    block = bool(plan.get("_shopping_coherence_block"))
    contract_viol = 0
    evaluated = 0
    try:
        from graph_orchestrator import _recipe_step_contract_issues
        for m in meals_flat:
            try:
                issues = _recipe_step_contract_issues(m)
                evaluated += 1
                if issues:
                    contract_viol += 1
            except Exception:
                continue
    except Exception:
        pass
    contract_ratio = (contract_viol / evaluated) if evaluated else 0.0
    empty_recipes = sum(1 for m in meals_flat if not (m.get("recipe") or []))
    score = 100.0 - (50.0 if block else 0.0) - 35.0 * contract_ratio - 8.0 * empty_recipes
    return {"score": _clamp(score), "shopping_coherence_block": block,
            "recipe_contract_violation_ratio": round(contract_ratio, 3), "empty_recipes": empty_recipes}


def _score_presupuesto(plan: dict) -> dict | None:
    br = plan.get("budget_reconciliation")
    if not isinstance(br, dict) or not br.get("status"):
        return None
    status = br.get("status")
    base = {"dentro": 100.0, "cerca": 80.0, "excedido": 50.0, "sin_limite": None}.get(status)
    if base is None:
        return None
    if br.get("partial_pricing"):
        base -= 10.0
    return {"score": _clamp(base), "status": status, "ratio": br.get("ratio"),
            "partial_pricing": bool(br.get("partial_pricing"))}


def _score_entrega(plan: dict) -> dict:
    """Honestidad de la entrega: fallback/degradación/review fallida."""
    if plan.get("_is_fallback"):
        return {"score": 0.0, "state": "fallback"}
    if plan.get("_review_failed_but_delivered"):
        return {"score": 40.0, "state": "review_failed_delivered"}
    if plan.get("_quality_degraded"):
        sev = str(plan.get("_quality_degraded_severity") or "minor")
        return {"score": 55.0 if sev != "minor" else 70.0,
                "state": f"degraded_{sev}", "reason": plan.get("_quality_degraded_reason")}
    return {"score": 100.0, "state": "clean"}


def score_plan(plan: dict, form_data: dict | None = None) -> dict:
    """Puntúa un plan entregado en los 8 ejes. Devuelve axes (None = sin datos, excluido),
    global 0-100 (promedio ponderado re-normalizado) y detail por eje."""
    if not isinstance(plan, dict):
        return {"global": 0.0, "axes": {}, "detail": {}, "error": "plan no es dict"}
    detail = {}
    axes = {}
    for name, fn in (
        ("banda", _score_banda), ("micros", _score_micros), ("slots", _score_slots),
        ("creatividad", _score_creatividad), ("coherencia", _score_coherencia),
        ("presupuesto", _score_presupuesto),
    ):
        try:
            r = fn(plan)
        except Exception as _e:
            logger.debug(f"[P1-NEXT-LEVEL-GYM] eje {name} no-op: {type(_e).__name__}: {_e}")
            r = None
        detail[name] = r
        axes[name] = r["score"] if isinstance(r, dict) else None
    r_ent = _score_entrega(plan)
    detail["entrega"] = r_ent
    axes["entrega"] = r_ent["score"]

    avail = {k: v for k, v in axes.items() if v is not None}
    if not avail:
        return {"global": 0.0, "axes": axes, "detail": detail}
    wsum = sum(_AXIS_WEIGHTS[k] for k in avail)
    global_score = sum(_AXIS_WEIGHTS[k] * v for k, v in avail.items()) / wsum if wsum else 0.0
    # Un plan de plantilla matemática puede clavar banda/slots (es determinista) — sin este
    # cap, un fallback puntuaba ~83 y enmascaraba el fallo del LLM. La entrega manda.
    if r_ent.get("state") == "fallback":
        global_score = min(global_score, 40.0)
    return {"global": round(global_score, 1), "axes": axes, "detail": detail}


def aggregate_scores(results: list) -> dict:
    """Agrega los scores de N corridas del gym: media/min por eje + peores perfiles."""
    scored = [r for r in results if isinstance(r, dict) and isinstance(r.get("score"), dict)]
    if not scored:
        return {"n": 0}
    axis_names = list(_AXIS_WEIGHTS.keys())
    agg = {"n": len(scored), "global_mean": round(
        sum(r["score"]["global"] for r in scored) / len(scored), 1)}
    for ax in axis_names:
        vals = [r["score"]["axes"].get(ax) for r in scored if r["score"]["axes"].get(ax) is not None]
        if vals:
            agg[ax] = {"mean": round(sum(vals) / len(vals), 1), "min": round(min(vals), 1), "n": len(vals)}
    worst = sorted(scored, key=lambda r: r["score"]["global"])[:3]
    agg["worst_profiles"] = [
        {"id": r.get("id"), "global": r["score"]["global"],
         "axes": {k: v for k, v in r["score"]["axes"].items() if v is not None and v < 80}}
        for r in worst
    ]
    return agg
