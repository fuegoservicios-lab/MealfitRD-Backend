# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-998] Medición del caché y del costo del coach (solo lectura sobre `llm_usage_events`).

Uso: python scripts/medir_cache_coach.py [horas=24]
Lee `chat_call_model` por día y por modelo: llamadas, tokens de entrada medios, % cacheado, tokens de salida,
duración media y costo. Es la vara para encender (o no) las palancas del lote 998, una a la vez, con su knob:
  MEALFIT_CHAT_SPLIT_VOLATILE_SYS_PROMPT · MEALFIT_GEMINI_CROSS_USER_CACHE_ANCHOR · MEALFIT_CHAT_PROMPT_COMPACT_ENABLED
Regla: una palanca por vez, ≥24 h de tráfico cada una, y el manual compacto solo si test_p1_plan_lote_998 lo permite.
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main(horas: int = 24) -> int:
    import psycopg
    with psycopg.connect(os.environ["NEON_DATABASE_URL"]) as c:
        filas = c.execute(
            """SELECT date_trunc('day', created_at AT TIME ZONE 'America/Santo_Domingo')::date AS dia, model,
                      count(*) AS n, round(avg(input_tokens)) AS in_medio,
                      round(100.0 * sum(coalesce(cached_tokens,0)) / nullif(sum(input_tokens),0), 1) AS pct_cacheado,
                      round(avg(output_tokens)) AS out_medio,
                      round(avg((metadata->>'duration_s')::float)::numeric, 2) AS seg_medios,
                      round(sum(cost_usd_micros) / 1e6, 4) AS usd
                 FROM llm_usage_events
                WHERE node = 'chat_call_model' AND created_at > now() - make_interval(hours => %s)
                GROUP BY 1, 2 ORDER BY 1 DESC, 3 DESC""", (horas,)).fetchall()
    print("dia | modelo | llamadas | in medio | %cacheado | out medio | s medios | USD")
    for f in filas:
        print(" | ".join(str(x) for x in f))
    if not filas:
        print("(sin llamadas en la ventana)")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 24))
