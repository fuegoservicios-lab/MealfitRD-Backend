# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-35 · 2026-09-13] CUL-P2-02: el presupuesto de reparación — cuántas capas reescriben cada comida.

Una comida del plan no sale del modelo tal cual: la toca el piso de porciones, el tope de realismo, los cerradores de
proteína y micros, el tope de huevos, las sustituciones por dieta o condición, el contrato final de la receta… Cada capa
deja su marca en la comida (`_portion_floor_adjusted`, `_protein_closed`, `_recipe_contract_final`…). Nadie las contaba
JUNTAS: el backlog pide que «no crezcan sin límite las reparaciones» y no había cifra de cuántas son hoy.

Este script cuenta, por comida, las capas que la REESCRIBIERON (`REPARADORES`, cada una con lo que hace) — no los
diagnósticos (`_solver_not_converged`, `_misalign_trace`, `_dish_quality_degraded`…), que observan sin tocar — y aparte
las reescrituras del contrato final de la receta. El solver de porciones no cuenta como reparación: es composición (el
plato se arma así) y se reporta aparte. Da la distribución (p50/p90/p95/máx), el % de comidas con ≥ 3 capas, las capas
más frecuentes, y — con el artefacto del bench real — lo que costó cada plan válido.

Con esa distribución se PROPONE un presupuesto (`MEALFIT_RECIPE_REPAIR_BUDGET`: el p95 más uno) que avise cuando una
comida necesita más capas que el 95 % de las que ya se entregan. **No se implementa aquí**: un presupuesto que corta
reparaciones cambia qué se entrega y es decisión del dueño; lo que sí queda es la cifra de la que saldría.

Solo lectura, sin base y sin LLM: lee el corpus fijo y los artefactos del bench.

    python scripts/measure_presupuesto_reparacion.py --out scripts/data/presupuesto_reparacion_<fecha>.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]

#: marca por comida → la capa que la reescribió
REPARADORES = {
    "_portion_floor_adjusted": "piso de porciones", "_portion_realism_capped": "tope de realismo de porciones",
    "_protein_closed": "cerrador de proteína", "_closer_raw_by_food": "cerrador de macros/micros",
    "_final_protein_close": "cierre final de proteína", "_refine_raw_by_food": "refinado del día",
    "_egg_day_capped": "tope diario de huevos", "_food_safety_fixed": "seguridad alimentaria",
    "_sodium_autofix_applied": "autofix de sodio", "_protein_autofix_applied": "autofix de proteína repetida",
    "_slot_autofix_applied": "autofix de franja", "_diet_subs_fixed": "sustitución por dieta",
    "_condition_subs_fixed": "sustitución por condición", "_dm2_sugar_fixed": "azúcar (DM2)",
    "_allergen_subs_fixed": "sustitución por alergia", "_micro_seed_applied": "semilla de micronutrientes",
    "_budget_substitutions": "sustitución por presupuesto", "_name_phantom_dairy_added": "lácteo que el nombre prometía",
    "_nocook_tdf_stripped": "plato sin cocción", "_recipe_timetemp_clamped": "tiempos y temperaturas",
    "_recipe_contract_repaired": "contrato de receta (capa 1)", "_gainmuscle_kcal_floor": "piso de kcal (ganancia muscular)",
    "_fat_topup": "relleno de grasa", "_day_kcal_floor": "piso de kcal del día",
    "_crossday_dish_diversified": "diversificador entre días", "_cured_ghost_rewritten": "curado fantasma",
    "_fresh_substituted": "fresco sustituido", "_mise_cook_split": "mise en place", "_seafood_blanch_injected": "blanqueo de mariscos",
    "_recipe_water_scaled": "agua de la receta", "_recipe_timetemp_injected": "tiempos inyectados",
    "_ingredients_backfilled": "ingredientes rellenados",
}
COMPOSICION = ("_solver_raw_by_food",)
CONTRATO = ("reescritas", "lista_reescrita", "estructura", "sin_lista", "repeticiones", "concordancia")


def _pct(xs: list, p: float):
    if not xs:
        return None
    s = sorted(xs)
    k = max(0, min(len(s) - 1, math.ceil(p / 100.0 * len(s)) - 1))
    return s[k]


def capas(meal: dict) -> list:
    out = [k for k in REPARADORES if meal.get(k)]
    cf = meal.get("_recipe_contract_final")
    if isinstance(cf, dict) and sum(int(cf.get(k) or 0) for k in CONTRATO) > 0:
        out.append("_recipe_contract_final")
    return out


def medir(planes: list) -> dict:
    por_comida, cuenta, contrato, solver, n = [], Counter(), 0, 0, 0
    for p in planes:
        for d in (p.get("days") or []):
            for m in (d.get("meals") or []):
                if not isinstance(m, dict):
                    continue
                n += 1
                c = capas(m)
                por_comida.append(len(c))
                cuenta.update(c)
                cf = m.get("_recipe_contract_final")
                if isinstance(cf, dict):
                    contrato += sum(int(cf.get(k) or 0) for k in CONTRATO)
                solver += 1 if any(m.get(k) for k in COMPOSICION) else 0
    return {
        "comidas": n,
        "capas_por_comida": {"p50": _pct(por_comida, 50), "p90": _pct(por_comida, 90), "p95": _pct(por_comida, 95),
                             "max": max(por_comida) if por_comida else None,
                             "media": round(statistics.mean(por_comida), 2) if por_comida else None},
        "pct_con_3_o_mas": round(100.0 * sum(1 for x in por_comida if x >= 3) / n, 1) if n else None,
        "capas_mas_frecuentes": [{"marca": k, "capa": REPARADORES.get(k, "contrato final de la receta"), "comidas": v}
                                 for k, v in cuenta.most_common(12)],
        "reescrituras_contrato_final": contrato,
        "comidas_con_solver": solver,
    }


def _ultimo(patron: str) -> Path:
    fs = sorted(glob.glob(str(_BACKEND / "scripts" / "data" / patron)))
    return Path(fs[-1]) if fs else None


def coste_por_plan_valido(real: dict, replay: dict | None) -> dict:
    total = float(real.get("coste_usd_est") or 0)
    ids = [g.get("plan_id") for g in real.get("generaciones") or []]
    validos = None
    if replay:
        ins = ((replay.get("superficies") or {}).get("insert") or {}).get("planes") or []
        validos = sum(1 for p in ins if not sum(((p.get("etapas") or {}).get("salida") or {}).values()))
    return {"planes": len(ids), "coste_usd_total": round(total, 4),
            "planes_validos_tras_insert": validos,
            "coste_por_plan_valido_usd": round(total / validos, 4) if validos else None,
            "nota": "válido = sin hallazgos de capa 1 a la salida de la cadena del INSERT (pareado real del bench)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    corpus = json.loads(_ultimo("culinary_corpus_*.json").read_text(encoding="utf-8"))
    real_p = _ultimo("bench_superficies_real_*.json")
    real = json.loads(real_p.read_text(encoding="utf-8")) if real_p else {}
    replay_p = _ultimo("bench_superficies_replay_*_l31.json") or _ultimo("bench_superficies_replay_*.json")
    replay = json.loads(replay_p.read_text(encoding="utf-8")) if replay_p else None
    res = {
        "corpus": medir([p["plan_data"] for p in corpus["planes"]]),
        "real": medir([g["plan_data"] for g in real.get("planes_generados") or []]),
    }
    todas = res["corpus"]["capas_por_comida"]["p95"], res["real"]["capas_por_comida"]["p95"]
    p95 = max(x for x in todas if x is not None)
    art = {"schema": "2026-09-13.presupuesto", "fecha": date.today().isoformat(),
           "reparadores": REPARADORES, "composicion_no_cuenta": list(COMPOSICION), "contrato_final_cuenta": list(CONTRATO),
           "fuentes": res, "coste": coste_por_plan_valido(real, replay) if real else None,
           "propuesta": {"knob": "MEALFIT_RECIPE_REPAIR_BUDGET", "valor": p95 + 1,
                         "regla": "p95 de capas por comida (corpus y planes reales) más uno",
                         "estado": "PROPUESTA — no implementada: cortar reparaciones cambia qué se entrega (decisión del dueño)"}}
    for k, r in res.items():
        cp = r["capas_por_comida"]
        print(f"{k:7s} {r['comidas']:3d} comidas · capas p50 {cp['p50']} p90 {cp['p90']} p95 {cp['p95']} máx {cp['max']} "
              f"· ≥3 capas {r['pct_con_3_o_mas']} % · contrato final {r['reescrituras_contrato_final']} reescrituras")
    if art["coste"]:
        print(f"coste: {art['coste']}")
    print(f"propuesta: {art['propuesta']['knob']} = {art['propuesta']['valor']} ({art['propuesta']['estado']})")
    if a.out:
        Path(a.out).write_text(json.dumps(art, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
