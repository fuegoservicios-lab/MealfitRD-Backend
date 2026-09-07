# -*- coding: utf-8 -*-
"""[P1-ARQ30-F5-NO-ADOPT · 2026-09-06] El residual del solver: ¿cuesta algo la infactibilidad?

`ARQ30-P1-02` propone sustituir el cribado actual por factibilidad CONJUNTA (LP/QP/MILP o CP-SAT).
Su propio texto pone la condición de entrada: **«primero cerrar ARQ27-P1-08 y medir el residual de
fallos»**, y la de salida: **«adoptar factibilidad conjunta solo con mejora demostrada»**. También
avisa de que `{}` en `_feasibility_report` **no es un testigo**.

Esta sonda mide ese residual. No cambia nada; cuenta.

    python scripts/solver_residual_probe.py
    python scripts/solver_residual_probe.py --dias 60 --json

## Qué compara, y por qué así

Un porcentaje de comidas infactibles no dice nada por sí solo: el cerrador y los clamps existen
justamente para absorberlas. Lo que decide si merece la pena un solver nuevo es si esas corridas
**cuestan más o entregan peor**. Por eso se emparejan por `session_id` las corridas CON al menos una
comida infactible contra las que no tienen ninguna, y se comparan cuatro cosas:

- `attempts` medio — el criterio de cierre dice «no gastar LLM regenerando con el mismo conjunto
  inviable». Si la infactibilidad provocara reintentos, se vería aquí.
- `review_passed` — si el revisor rechaza más.
- desviación calórica p50/p90 — si el plato entregado sale peor.
- tasa de fallback — si acaba en el plan matemático.

## Medición del 2026-09-06 (30 días)

    comidas dimensionadas .......... 3.217
      no convergieron .............. 1.224   38,0 %
      declaradas infactibles ......... 326   10,1 %
    corridas con >=1 infactible .... 156 de 264   59,1 %

                        CON infactibles (6.706)   sin (4.018)
    attempts medio ............ 1,55              1,54
    review_passed ............. 92,4 %            93,0 %
    desviacion p50 / p90 ...... 0,020 / 0,036     0,019 / 0,035
    fallback .................. 2,5 %             3,7 %

**La infactibilidad no cuesta ni un reintento ni un punto de calidad**, y las corridas que la sufren
caen MENOS al fallback. No hay mejora que obtener de un solver conjunto, así que no se adopta — el
encargo lo prevé: «si el prototipo no mejora, conserva el actual y registra el resultado
experimental; no cambies de librería solo para usar el nombre 3.0».

Vuelve a correr esto antes de reabrir la discusión. La decisión es de los números, no de la opinión.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def medir(dias: int = 30) -> dict:
    from dotenv import load_dotenv
    import psycopg
    from psycopg.rows import dict_row

    load_dotenv(_BACKEND / ".env")
    with psycopg.connect(os.environ["NEON_DATABASE_URL"], row_factory=dict_row) as c:
        crudo = c.execute(
            "SELECT metadata FROM pipeline_metrics WHERE node='solver_convergence' "
            "AND created_at > NOW() - make_interval(days => %s)", (dias,)).fetchall()
        pares = c.execute(
            """SELECT (s.metadata->>'infeasible_meals')::int infeas,
                      (h.metadata->>'attempts')::int attempts,
                      (h.metadata->>'review_passed') rev,
                      (h.metadata->>'avg_cal_deviation')::float dev,
                      (h.metadata->>'delivered_was_fallback') fb
               FROM pipeline_metrics s
               JOIN pipeline_metrics h ON h.session_id = s.session_id
                                      AND h.node = 'pipeline_holistic'
              WHERE s.node = 'solver_convergence'
                AND s.created_at > NOW() - make_interval(days => %s)""", (dias,)).fetchall()

    tot = {"total_meals": 0, "not_converged": 0, "infeasible_meals": 0,
           "greedy_fallback": 0, "abstained_coverage": 0}
    corridas_infeas = 0
    for r in crudo:
        m = r["metadata"] or {}
        for k in tot:
            tot[k] += int(m.get(k) or 0)
        if int(m.get("infeasible_meals") or 0) > 0:
            corridas_infeas += 1

    def cohorte(sel):
        xs = [p for p in pares if sel(p)]
        att = [p["attempts"] for p in xs if p["attempts"] is not None]
        rev = [str(p["rev"]).lower() == "true" for p in xs if p["rev"] is not None]
        dev = sorted(p["dev"] for p in xs if p["dev"] is not None)
        fb = [str(p["fb"]).lower() == "true" for p in xs if p["fb"] is not None]
        return {
            "n": len(xs),
            "attempts_medio": round(sum(att) / len(att), 3) if att else None,
            "review_passed_pct": round(100.0 * sum(rev) / len(rev), 1) if rev else None,
            "dev_p50": round(dev[len(dev) // 2], 4) if dev else None,
            "dev_p90": round(dev[int(len(dev) * 0.9)], 4) if dev else None,
            "fallback_pct": round(100.0 * sum(fb) / len(fb), 1) if fb else None,
        }

    return {
        "dias": dias, "corridas_solver": len(crudo), "corridas_emparejadas": len(pares),
        "agregado": tot,
        "corridas_con_infeasible": corridas_infeas,
        "con_infeasible": cohorte(lambda p: (p["infeas"] or 0) > 0),
        "sin_infeasible": cohorte(lambda p: not (p["infeas"] or 0)),
    }


def render(r: dict) -> str:
    a, con, sin = r["agregado"], r["con_infeasible"], r["sin_infeasible"]
    t = a["total_meals"]

    def pct(n):
        return "{:.1f} %".format(100.0 * n / t) if t else "—"

    o = [f"ventana: {r['dias']} días · corridas con telemetría de solver: {r['corridas_solver']}",
         "",
         f"  comidas dimensionadas ........ {t}",
         f"    no convergieron ............ {a['not_converged']:6d}  {pct(a['not_converged'])}",
         f"    declaradas INFACTIBLES ..... {a['infeasible_meals']:6d}  {pct(a['infeasible_meals'])}",
         f"    cayeron a greedy ........... {a['greedy_fallback']:6d}  {pct(a['greedy_fallback'])}",
         f"  corridas con >=1 infactible .. {r['corridas_con_infeasible']} de {r['corridas_solver']}",
         "",
         "  ¿CUESTA ALGO? (emparejado por session_id)",
         "  {:26s} {:>18s} {:>14s}".format("", "CON infactibles", "sin")]
    for etiq, k in (("n", "n"), ("attempts medio", "attempts_medio"),
                    ("review_passed", "review_passed_pct"),
                    ("desviacion p50", "dev_p50"), ("desviacion p90", "dev_p90"),
                    ("fallback", "fallback_pct")):
        o.append("  {:26s} {:>18s} {:>14s}".format(etiq, str(con[k]), str(sin[k])))
    o += ["",
          "  Si las dos columnas se parecen, la infactibilidad no cuesta nada y un solver conjunto",
          "  no tiene mejora que obtener. Ese era el veredicto el 2026-09-06."]
    return "\n".join(o)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dias", type=int, default=30)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = medir(a.dias)
    print(json.dumps(r, ensure_ascii=False, indent=2) if a.json else render(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
