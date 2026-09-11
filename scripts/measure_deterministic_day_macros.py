"""[P1-PLAN-LOTE-10 · 2026-09-11 · B7] Medición REPRODUCIBLE del sesgo de macros del día determinista.

Arma N días sin LLM (`deterministic_day.build_day_for_skeleton`, con memoria entre días como en producción) para
un objetivo de macros y responde tres preguntas, por día y por franja:

  1. ¿Cuánto se desvía lo servido del objetivo (proteína, carbohidratos, grasa, kcal)?
  2. ¿Dónde vive el desvío? (por franja, en gramos contra el objetivo de ESA franja)
  3. ¿Podría el scorer elegir otra cosa? Entre los candidatos ESCALABLES de cada franja, cuántos quedan por debajo
     del objetivo de carbohidratos y cuál es el suelo alcanzable. Si el suelo ya supera el objetivo, el sesgo es
     de COMPOSICIÓN de la biblioteca (qué platos hay), no de selección — y ningún cambio del scorer lo mueve.

Sólo LEE: catálogo (`master_ingredients`) y snapshots del registry; no escribe en la base ni llama al modelo.
Los knobs se fuerzan en este proceso (`MEALFIT_RECIPE_LIBRARY_SELECT`, `MEALFIT_DETERMINISTIC_DAY`), sin tocar
el entorno de producción.

Uso (desde `backend/`, entorno conda `mealfit`):
    python scripts/measure_deterministic_day_macros.py --days 14 --kcal 2000 --protein 150 --carbs 200 --fats 60

Medido el 2026-09-11 (DO, 2000 kcal · 150/200/60): proteína +2,4 % (14/14 en banda), carbohidratos +18,0 %
(5/14), grasa −17,7 % (4/14). El exceso vive en almuerzo (+17 g de 70) y desayuno (+12 g de 40): en almuerzo sólo
1,8 de 18,2 candidatos escalables quedan ≤ 70 g de carbohidratos. Siete variantes del scorer (asimetría de
proteína/carbohidrato, puerta de proteína en el empate) movieron los carbohidratos entre +16,7 % y +21,6 %: nada.
"""
from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
from collections import Counter, defaultdict

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _say(msg: str = "") -> None:
    # [P2-LOGGER-EXEMPT: CLI de medición read-only; la salida ES el informe (stdout), no un log de servicio]
    print(msg)


def _g(s) -> float:
    try:
        return float(str(s).replace("g", "").replace("kcal", "").strip() or 0)
    except ValueError:
        return 0.0


def medir(days: int, kcal: float, protein: float, carbs: float, fats: float, country: str,
          conditions: list[str], list_low_carb: int) -> dict:
    os.environ.setdefault("MEALFIT_RECIPE_LIBRARY_SELECT", "1")
    os.environ.setdefault("MEALFIT_DETERMINISTIC_DAY", "1")
    if _BACKEND not in sys.path:
        sys.path.insert(0, _BACKEND)
    os.chdir(_BACKEND)
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_BACKEND, ".env"))
    logging.basicConfig(level=logging.ERROR)
    import db_core
    db_core.connection_pool.open()
    import deterministic_day as dd

    nut = {"target_calories": kcal, "macros": {"protein": f"{protein:g}g", "carbs": f"{carbs:g}g", "fats": f"{fats:g}g"}}
    skel = {"meal_types": ["Desayuno", "Almuerzo", "Merienda", "Cena"], "protein_pool": []}
    fd = {"country": country, "user_id": "measure-deterministic-day", "health_profile": {}}
    if conditions:
        fd["medicalConditions"] = list(conditions)
    T = {"P": protein, "C": carbs, "F": fats, "K": kcal}

    # Espía sobre `elegir_plantillas`: qué había para elegir en cada franja (sólo lectura, misma decisión).
    orig = dd.elegir_plantillas
    cand = defaultdict(list)
    low_carb_names = defaultdict(Counter)

    def espia(tids, objetivo, catalogo, por_id, slot="", **kw):
        lo, hi = dd._banda(slot)
        ok = float(objetivo.get("kcal") or 0)
        cs, ps, fs = [], [], []
        for tid in (tids or []):
            t = por_id.get(tid)
            if not t:
                continue
            base = dd._macros(dd._de_plantilla(t), catalogo)
            if not base:
                continue
            f = ok / base["kcal"]
            if not (lo <= f <= hi):
                continue
            c = base["carbs_g"] * f
            cs.append(c); ps.append(base["protein_g"] * f); fs.append(base["fats_g"] * f)
            if c <= objetivo["carbs_g"]:
                low_carb_names[slot][str(t.get("name"))] += 1
        if cs:
            cand[slot].append(dict(n=len(cs), c_min=min(cs), c_med=statistics.median(cs),
                                   c_ok=sum(1 for c in cs if c <= objetivo["carbs_g"]),
                                   f_med=statistics.median(fs), p_max=max(ps),
                                   obj_c=objetivo["carbs_g"], obj_f=objetivo["fats_g"]))
        return orig(tids, objetivo, catalogo, por_id, slot, **kw)

    dd.elegir_plantillas = espia
    try:
        memoria, dias = [], []
        for d in range(1, days + 1):
            dias.append(dd.build_day_for_skeleton(nut, fd, skel, d, memoria=memoria))
    finally:
        dd.elegir_plantillas = orig
    built = [x for x in dias if x]

    fr = dict(zip(skel["meal_types"], dd._fracciones_por_franja(skel["meal_types"])))
    dP, dC, dF, dK = [], [], [], []
    slot = defaultdict(lambda: {"P": [], "C": [], "F": [], "platos": []})
    for x in built:
        P = sum(_g(m["protein"]) for m in x["meals"]); C = sum(_g(m["carbs"]) for m in x["meals"])
        F = sum(_g(m["fats"]) for m in x["meals"]); K = sum(float(m["calories"]) for m in x["meals"])
        dP.append((P - T["P"]) / T["P"]); dC.append((C - T["C"]) / T["C"])
        dF.append((F - T["F"]) / T["F"]); dK.append((K - T["K"]) / T["K"])
        for m in x["meals"]:
            r = fr.get(m["meal"], 0.25)
            s = slot[m["meal"]]
            s["P"].append(_g(m["protein"]) - T["P"] * r); s["C"].append(_g(m["carbs"]) - T["C"] * r)
            s["F"].append(_g(m["fats"]) - T["F"] * r); s["platos"].append(m["name"])

    def pct(v):
        return 100 * statistics.mean(v) if v else float("nan")

    def banda(v):
        return sum(1 for x in v if abs(x) <= 0.15)

    _say(f"# Día determinista · {country} · {days} días · objetivo {kcal:g} kcal · P {protein:g} g · C {carbs:g} g · F {fats:g} g"
         + (f" · condiciones {conditions}" if conditions else ""))
    _say(f"días construidos sin LLM: {len(built)}/{days}")
    if not built:
        return {"built": 0}
    _say(f"\n## Desvío medio del día: P {pct(dP):+.1f} % · C {pct(dC):+.1f} % · F {pct(dF):+.1f} % · kcal {pct(dK):+.2f} %"
         f"   (en banda ±15 %: P {banda(dP)} · C {banda(dC)} · F {banda(dF)} de {len(built)})")
    _say("\n## Por franja · desvío medio en gramos contra el objetivo de la franja")
    for k, s in slot.items():
        r = fr.get(k, 0.25)
        _say(f"  {k:10s} obj P {T['P']*r:5.1f} C {T['C']*r:5.1f} F {T['F']*r:4.1f} g │ ΔP {statistics.mean(s['P']):+6.1f} "
             f"ΔC {statistics.mean(s['C']):+6.1f} ΔF {statistics.mean(s['F']):+6.1f} g │ platos distintos {len(set(s['platos']))}")
    _say("\n## Candidatos escalables por franja · ¿existe plato por debajo del objetivo de carbohidratos?")
    for k, filas in cand.items():
        n = statistics.mean(f["n"] for f in filas)
        _say(f"  {k:10s} escalables {n:4.1f} · carbos mín {statistics.mean(f['c_min'] for f in filas):5.1f} g · mediana "
             f"{statistics.mean(f['c_med'] for f in filas):5.1f} g vs obj {filas[0]['obj_c']:5.1f} g · ≤ objetivo: "
             f"{statistics.mean(f['c_ok'] for f in filas):4.1f} de {n:4.1f} · grasa mediana {statistics.mean(f['f_med'] for f in filas):4.1f} "
             f"vs obj {filas[0]['obj_f']:4.1f} g")
    if list_low_carb:
        _say(f"\n## Los platos que SÍ quedan por debajo del objetivo de carbohidratos (top {list_low_carb} por franja)")
        for k, c in low_carb_names.items():
            _say(f"  {k}: " + " · ".join(f"{n} ({v})" for n, v in c.most_common(list_low_carb)))
    _say("\n## Platos servidos por franja")
    for k, s in slot.items():
        _say(f"  {k}: {dict(Counter(s['platos']).most_common())}")
    return {"built": len(built), "p_pct": pct(dP), "c_pct": pct(dC), "f_pct": pct(dF)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--kcal", type=float, default=2000)
    ap.add_argument("--protein", type=float, default=150)
    ap.add_argument("--carbs", type=float, default=200)
    ap.add_argument("--fats", type=float, default=60)
    ap.add_argument("--country", default="DO")
    ap.add_argument("--condition", action="append", default=[], help="id de condición (dm2, hta, renal…); repetible")
    ap.add_argument("--list-low-carb", type=int, default=6, help="cuántos platos bajos en carbohidrato listar por franja (0 = ninguno)")
    a = ap.parse_args(argv)
    medir(a.days, a.kcal, a.protein, a.carbs, a.fats, a.country, a.condition, a.list_low_carb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
