"""[P1-PLAN-LOTE-6 · 2026-09-11 · D7] Medición READ-ONLY del volumen real de `magnitude_undersupply`.

Es el SELECT que `shopping_calculator._get_guard_undersupply_severe_knob` pide antes de encender
`MEALFIT_GUARD_UNDERSUPPLY_SEVERE` («medir en producción el volumen real: cuántos planes, cuántos alimentos
por plan, y cuántos vienen de T2»). Lee la serie diaria del cron (`_shopping_coherence_alert_job_tick` en
`pipeline_metrics`) y la history viva en `meal_plans`. Sólo SELECTs; abre la conexión en `read_only`.

Uso (desde `backend/`):  python scripts/measure_undersupply_volume.py [--days 45]

Criterio de encendido (del docstring del knob): «estable y bajo (<5% de las entries diarias), sin ráfagas
concentradas en un solo plan». Medido el 2026-09-11: 3 en 30 días, los tres en la semana de ráfaga del canario.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))


# [P2-LOGGER-EXEMPT: CLI de medición; el informe va a stdout a propósito]
def _say(*partes) -> None:
    print(*partes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=45)
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(_BACKEND / ".env")
    import psycopg

    url = os.environ.get("NEON_DATABASE_URL")
    if not url:
        _say("NEON_DATABASE_URL ausente: nada que medir")
        return 2
    with psycopg.connect(url, connect_timeout=20) as conn:
        conn.read_only = True
        cur = conn.cursor()
        cur.execute(
            """
            SELECT date_trunc('day', created_at)::date AS d,
                   count(*) AS ticks,
                   sum(coalesce((metadata->>'n_plans')::int, 0)) AS n_plans,
                   sum(coalesce((metadata->>'plans_with_div')::int, 0)) AS plans_with_div,
                   sum(coalesce((metadata->>'cap_count')::int, 0)) AS cap_count,
                   sum(coalesce((metadata->>'undersupply_count')::int, 0)) AS undersupply
            FROM pipeline_metrics
            WHERE node = '_shopping_coherence_alert_job_tick' AND created_at > now() - make_interval(days => %s)
            GROUP BY 1 ORDER BY 1
            """,
            (args.days,),
        )
        rows = cur.fetchall()
        _say(f"— serie diaria del cron ({args.days} d, {len(rows)} días con tick) —")
        _say("día | ticks | n_plans | plans_with_div | cap | undersupply")
        for r in rows:
            _say(" | ".join(str(x) for x in r))
        tot_us = sum(r[5] for r in rows)
        tot_div = sum(r[3] for r in rows)
        dias_con_us = sum(1 for r in rows if r[5])
        _say(f"TOTAL undersupply={tot_us} plans_with_div={tot_div} días_con_undersupply={dias_con_us}/{len(rows)}")

        cur.execute(
            """
            SELECT count(*) AS plans_con_history,
                   coalesce(sum(jsonb_array_length(plan_data->'_shopping_coherence_block_history')), 0) AS entries,
                   count(*) FILTER (WHERE (plan_data->'_shopping_coherence_block_history')::text LIKE '%%magnitude_undersupply%%') AS plans_con_undersupply
            FROM meal_plans
            WHERE jsonb_typeof(plan_data->'_shopping_coherence_block_history') = 'array'
              AND created_at > now() - make_interval(days => %s)
            """,
            (args.days,),
        )
        r = cur.fetchone()
        _say(f"— history viva en meal_plans ({args.days} d): planes={r[0]} entries={r[1]} planes_con_undersupply={r[2]}")

        cur.execute(
            """
            SELECT id, e->>'surface', e->>'action_taken'
            FROM meal_plans, jsonb_array_elements(plan_data->'_shopping_coherence_block_history') e
            WHERE jsonb_typeof(plan_data->'_shopping_coherence_block_history') = 'array'
              AND created_at > now() - make_interval(days => %s)
              AND e::text LIKE '%%magnitude_undersupply%%'
            LIMIT 200
            """,
            (args.days,),
        )
        det = cur.fetchall()
        _say(f"entries con undersupply: {len(det)} · por plan: {dict(Counter(str(d[0])[:8] for d in det))} "
             f"· por surface/action: {dict(Counter((d[1], d[2]) for d in det))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
