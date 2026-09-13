# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-36 · 2026-09-13] B6: ¿bajan las ventanas de repetición rotas en los planes reales?

La memoria entre días del día determinista (`P1-PLAN-LOTE-2`, 2026-09-11) nació de una medición: en los 30 días del plan
del dueño, el tope de repetición exacta de la política (`balanced`: una plantilla como mucho 2 veces por 7 días) se rompía
en 9 ventanas con las puertas de variedad apagadas. El plan decía «medir sobre el canario si bajan». Este script lo mide
sobre los planes PERSISTIDOS, con el mismo conteo que `deterministic_day._conteo_ventana`: por `_template_id` (o
`_recipe_template_id`) cuando la comida viene de la biblioteca, por nombre exacto normalizado cuando viene del modelo.

Por plan: los días entregados en orden (`_archived_days` + `days`), cada ventana deslizante de 7 días y cuántas sirven una
misma comida por encima del tope de SU política (`_plan_policy.effective.recurrence.global_mode`, `balanced` por
defecto, tabla `horizon.repetition_limits_for`). Cohortes: planes creados hasta el corte (antes de la memoria entre días) y
después; y aparte los que llevan comidas del día determinista. Un plan de menos de 7 días no tiene ventana: se cuenta como
tal, no como «sin roturas».

Solo lectura (`read_only`). Sin planes nuevos con día determinista el veredicto es «sin muestra», con la fecha.

    python scripts/measure_variety_windows.py --dias 60 --out scripts/data/variety_windows_<fecha>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

CORTE = "2026-09-11"   # P1-PLAN-LOTE-2: la memoria entre días entra en el día determinista


def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return " ".join(s.split())


def clave(meal: dict):
    if not isinstance(meal, dict):
        return None
    tid = meal.get("_template_id") or meal.get("_recipe_template_id")
    if tid:
        return f"plantilla:{tid}"
    n = _norm(meal.get("name"))
    return f"nombre:{n}" if n else None


def dias_entregados(plan_data: dict) -> list:
    dias = [d for d in (plan_data.get("_archived_days") or []) + (plan_data.get("days") or []) if isinstance(d, dict)]
    if dias and all(d.get("date") for d in dias):
        dias = sorted(dias, key=lambda d: str(d.get("date")))
    return dias


def tope_7d(plan_data: dict) -> tuple:
    import horizon
    eff = (plan_data.get("_plan_policy") or {}).get("effective") or {}
    modo = (eff.get("recurrence") or {}).get("global_mode") or "balanced"
    return modo, int(horizon.repetition_limits_for(modo, 7)["max_exact_repeat"])


def ventanas(dias: list, tope: int) -> dict:
    n = len(dias)
    if n < 7:
        return {"dias": n, "ventanas": 0, "rotas": 0, "sobre_tope": []}
    rotas, sobre_total = 0, Counter()
    for ini in range(0, n - 6):
        c = Counter(k for d in dias[ini:ini + 7] for m in (d.get("meals") or []) if (k := clave(m)))
        sobre = {k: v for k, v in c.items() if v > tope}
        if sobre:
            rotas += 1
            sobre_total.update(sobre.keys())
    return {"dias": n, "ventanas": n - 6, "rotas": rotas, "sobre_tope": [k for k, _ in sobre_total.most_common(8)]}


def medir(filas: list, corte: str = CORTE) -> dict:
    cohortes = {}
    planes = []
    for pid, creado, pd in filas:
        pd = pd or {}
        dias = dias_entregados(pd)
        modo, tope = tope_7d(pd)
        det = any((m or {}).get("_template_id") for d in dias for m in (d.get("meals") or []) if isinstance(m, dict))
        v = ventanas(dias, tope)
        coh = ("antes" if str(creado)[:10] <= corte else "despues") + ("_determinista" if det else "_modelo")
        c = cohortes.setdefault(coh, {"planes": 0, "con_ventana": 0, "ventanas": 0, "rotas": 0, "planes_con_rotura": 0})
        c["planes"] += 1
        c["con_ventana"] += 1 if v["ventanas"] else 0
        c["ventanas"] += v["ventanas"]
        c["rotas"] += v["rotas"]
        c["planes_con_rotura"] += 1 if v["rotas"] else 0
        planes.append({"plan": str(pid)[:8], "creado": str(creado)[:10], "modo": modo, "tope_7d": tope,
                       "determinista": det, **v})
    for c in cohortes.values():
        c["tasa_rotas"] = round(c["rotas"] / c["ventanas"], 3) if c["ventanas"] else None
    return {"cohortes": cohortes, "planes": planes}


def _leer(dias: int) -> list:
    from dotenv import dotenv_values
    import psycopg
    url = os.environ.get("NEON_DATABASE_URL") or dotenv_values(_BACKEND / ".env").get("NEON_DATABASE_URL")
    if not url:
        raise SystemExit("falta NEON_DATABASE_URL")
    with psycopg.connect(url) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute("""SELECT id, created_at, jsonb_build_object('days', plan_data->'days',
                                  '_archived_days', plan_data->'_archived_days', '_plan_policy', plan_data->'_plan_policy')
                             FROM meal_plans
                            WHERE created_at > now() - make_interval(days => %s)
                            ORDER BY created_at""", (int(dias),))
            return cur.fetchall()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dias", type=int, default=60)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    res = medir(_leer(a.dias))
    art = {"schema": "2026-09-13.variety_windows", "fecha": date.today().isoformat(), "ventana_dias": a.dias,
           "corte": CORTE, "linea_base": "09-10: 9 ventanas rotas en el plan de 30 días del dueño (puertas apagadas)", **res}
    for k, c in sorted(res["cohortes"].items()):
        print(f"{k:22s} planes {c['planes']:3d} · con ventana de 7 d {c['con_ventana']:3d} · ventanas {c['ventanas']:4d} · "
              f"rotas {c['rotas']:3d} ({c['tasa_rotas']}) · planes con rotura {c['planes_con_rotura']}")
    det_despues = res["cohortes"].get("despues_determinista", {})
    if not det_despues.get("con_ventana"):
        print(f"veredicto: SIN MUESTRA — ningún plan posterior al {CORTE} con día determinista y ≥ 7 días entregados")
    if a.out:
        Path(a.out).write_text(json.dumps(art, ensure_ascii=False, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
