"""[P1-PLAN-LOTE-15 · 2026-09-12 · E4] Canario de ENTREGA: fallos, reintentos, latencia y coste por plan (READ-ONLY).

Cierra la parte de ARQ27-P1-06 que `delivery_battery.py` declaraba abierta: «la batería NO mide latencia ni coste por
plan entregado — eso es el canary y necesita generaciones reales». Las generaciones reales YA existen: son los planes
de producción. Este script las lee, plan a plan, y publica tasas CON SU DENOMINADOR por cohorte (país · dieta ·
clínica), nunca un promedio global donde se esconde la cohorte pequeña.

Por plan:
  · entrega     `generation_status` y días (vivos + archivados) — un plan sin días no se entregó.
  · validez     entregado Y sin `_quality_degraded` Y sin `_review_failed_but_delivered` Y sin chunk muerto.
  · latencia    bloque 1 = `plan_chunk_queue` (`chunk_kind='initial'`, created → completed); lo que espera el usuario.
  · reintentos  chunks completados con `attempts ≥ 1` (re-pickup: fallo, backoff o zombie rescue) y muertos.
  · coste LLM   `llm_usage_events` por `plan_id` — sólo existe desde que el worker atribuye (este mismo lote):
                los planes anteriores salen como «sin atribuir», no como US$ 0. Medido el 09-12 ANTES del cambio:
                0 de 116 filas de `day_generator` traían plan_id.
  · swaps       llamadas `swap_meal` atribuidas al plan (misma condición).
  · alertas     `system_alerts` con `metadata.plan_id` (calidad degradada, persist fallido, partial varado, zombie).
  · proyección  estado de la proyección de compras para la revisión actual (`classify_projection_jobs`, puro).

Sólo SELECTs; abre la conexión en `read_only`. Uso (desde `backend/`):

    python scripts/canary_plan_delivery.py [--days 30] [--json]

Cero hallazgos en N planes no demuestra una garantía universal: por eso cada tasa lleva su N.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.append(str(_BACKEND))  # al FINAL: en cabeza, scripts/plan_gym.py sombrea a plan_gym (P1-PLAN-LOTE-13)

ESTADOS_ENTREGADOS = ("complete", "complete_partial", "partial")


# [P2-LOGGER-EXEMPT: CLI de medición; el informe va a stdout a propósito]
def _say(*partes: Any) -> None:
    print(*partes)


def _f(v: Any, nd: int = 1) -> Optional[float]:
    return None if v is None else round(float(v), nd)


def percentil(valores: list, p: float, nd: int = 1) -> Optional[float]:
    """Percentil por interpolación lineal (como `percentile_cont`). None sin datos."""
    vals = sorted(float(v) for v in valores if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return round(vals[0], nd)
    k = (len(vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return round(vals[lo] + (vals[hi] - vals[lo]) * (k - lo), nd)


def cohorte_de(fila: dict) -> str:
    """`<país>·<dieta>[·clinica]` — la misma separación que la batería (país, dieta, condición)."""
    pais = (fila.get("country") or "??").upper()
    dieta = (fila.get("diet") or "sin_dieta").lower()
    clinica = "·clinica" if fila.get("conditions") else ""
    return f"{pais}·{dieta}{clinica}"


def clasificar(fila: dict) -> dict:
    """Puro: entregado / válido / motivo, a partir de una fila ya leída."""
    dias = int(fila.get("days_vivos") or 0) + int(fila.get("days_archivados") or 0)
    gs = (fila.get("gs") or "").lower()
    entregado = gs in ESTADOS_ENTREGADOS and dias > 0
    motivos = []
    if not entregado:
        motivos.append(f"gs={gs or 'null'} dias={dias}")
    if fila.get("quality_degraded"):
        motivos.append("quality_degraded")
    if fila.get("review_failed_delivered"):
        motivos.append("review_failed_but_delivered")
    if int(fila.get("dead") or 0) > 0:
        motivos.append(f"dead={fila.get('dead')}")
    valido = entregado and not fila.get("quality_degraded") and not fila.get("review_failed_delivered") \
        and int(fila.get("dead") or 0) == 0
    return {"entregado": entregado, "valido": valido, "dias": dias, "motivos": motivos}


def resumen(filas: list) -> dict:
    """Puro: tasas con denominador, por cohorte y global. `filas` son dicts con las claves que produce `leer`."""
    grupos: dict[str, list] = defaultdict(list)
    for f in filas:
        grupos[cohorte_de(f)].append(f)
    grupos["TODAS"] = list(filas)

    out: dict[str, dict] = {}
    for nombre, fs in sorted(grupos.items(), key=lambda kv: (kv[0] == "TODAS", kv[0])):
        cls = [clasificar(f) for f in fs]
        entregados = [f for f, c in zip(fs, cls) if c["entregado"]]
        validos = [f for f, c in zip(fs, cls) if c["valido"]]
        atribuidos = [f for f in entregados if f.get("llm_llamadas")]
        out[nombre] = {
            "n": len(fs),
            "entregados": len(entregados),
            "validos": len(validos),
            "no_entregados": [{"plan": str(f.get("id"))[:8], "motivos": c["motivos"]} for f, c in zip(fs, cls) if not c["entregado"]],
            "bloque1_p50_s": percentil([f.get("bloque1_s") for f in entregados], 0.5),
            "bloque1_p95_s": percentil([f.get("bloque1_s") for f in entregados], 0.95),
            "con_reintento": sum(1 for f in fs if int(f.get("done_con_reintento") or 0) > 0),
            "dead": sum(int(f.get("dead") or 0) for f in fs),
            "coste_atribuidos": len(atribuidos),
            "coste_sin_atribuir": len(entregados) - len(atribuidos),
            "usd_p50": percentil([f.get("usd") for f in atribuidos], 0.5, 3),
            "usd_p95": percentil([f.get("usd") for f in atribuidos], 0.95, 3),
            "usd_total": _f(sum(float(f.get("usd") or 0) for f in atribuidos), 3),
            "llm_s_p95": percentil([f.get("llm_s") for f in atribuidos], 0.95),
            "swaps": sum(int(f.get("swaps") or 0) for f in fs),
            "alertas": sum(int(f.get("alertas") or 0) for f in fs),
            "proyeccion": dict(sorted(_conteo(f.get("proyeccion") for f in fs).items())),
        }
    return out


def _conteo(items) -> dict:
    c: dict[str, int] = defaultdict(int)
    for it in items:
        c[str(it)] += 1
    return dict(c)


_SQL_PLANES = """
WITH planes AS (
  SELECT mp.id, mp.user_id, mp.created_at, mp.revision,
         mp.plan_data->>'generation_status' AS gs,
         jsonb_array_length(coalesce(mp.plan_data->'days', '[]'::jsonb)) AS days_vivos,
         jsonb_array_length(coalesce(mp.plan_data->'_archived_days', '[]'::jsonb)) AS days_archivados,
         (mp.plan_data->>'total_days_requested')::int AS dias_pedidos,
         mp.plan_data->>'_country' AS country,
         mp.plan_data->'_plan_policy'->'effective'->'diet'->>'type' AS diet,
         jsonb_array_length(coalesce(mp.plan_data->'_plan_policy'->'effective'->'clinical'->'conditions', '[]'::jsonb)) > 0 AS conditions,
         coalesce(mp.plan_data->>'_quality_degraded', '') = 'true' AS quality_degraded,
         coalesce(mp.plan_data->>'_review_failed_but_delivered', '') = 'true' AS review_failed_delivered
  FROM meal_plans mp
  WHERE mp.created_at > now() - make_interval(days => %s)
), chunks AS (
  SELECT meal_plan_id,
         count(*) AS chunks,
         count(*) FILTER (WHERE status = 'completed') AS done,
         count(*) FILTER (WHERE status = 'cancelled') AS cancelled,
         count(*) FILTER (WHERE status NOT IN ('completed', 'cancelled') AND dead_lettered_at IS NULL) AS abiertos,
         count(*) FILTER (WHERE dead_lettered_at IS NOT NULL) AS dead,
         count(*) FILTER (WHERE status = 'completed' AND coalesce(attempts, 0) >= 1) AS done_con_reintento,
         max(EXTRACT(EPOCH FROM (updated_at - created_at))) FILTER (WHERE chunk_kind = 'initial' AND status = 'completed') AS bloque1_s,
         max(updated_at) FILTER (WHERE status = 'completed') AS ultimo_done
  FROM plan_chunk_queue GROUP BY 1
), coste AS (
  SELECT plan_id,
         count(*) AS llm_llamadas,
         sum(cost_usd_micros) / 1e6 AS usd,
         sum(NULLIF(metadata->>'duration_s', '')::numeric) AS llm_s,
         count(*) FILTER (WHERE node = 'swap_meal') AS swaps
  FROM llm_usage_events WHERE plan_id IS NOT NULL GROUP BY 1
), alertas AS (
  SELECT (metadata->>'plan_id')::uuid AS plan_id, count(*) AS alertas
  FROM system_alerts
  WHERE alert_type IN ('plan_quality', 'plan_persist_failed', 'plan_stranded_partial', 'plan_chunk_zombie')
    AND metadata->>'plan_id' ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
  GROUP BY 1
)
SELECT p.id::text AS id, p.user_id::text AS user_id, p.created_at, p.revision, p.gs, p.days_vivos, p.days_archivados,
       p.dias_pedidos, p.country, p.diet, p.conditions, p.quality_degraded, p.review_failed_delivered,
       c.chunks, c.done, c.cancelled, c.abiertos, coalesce(c.dead, 0) AS dead, coalesce(c.done_con_reintento, 0) AS done_con_reintento,
       c.bloque1_s, c.ultimo_done,
       k.llm_llamadas, k.usd, k.llm_s, coalesce(k.swaps, 0) AS swaps,
       coalesce(a.alertas, 0) AS alertas
FROM planes p
LEFT JOIN chunks c ON c.meal_plan_id = p.id
LEFT JOIN coste k ON k.plan_id = p.id
LEFT JOIN alertas a ON a.plan_id = p.id
ORDER BY p.created_at
"""

_SQL_JOBS = """
SELECT id::text AS id, status, plan_revision, attempts, error_code, payload, processed_at, created_at
FROM plan_jobs WHERE plan_id = %s AND job_type = 'shopping_projection' ORDER BY created_at DESC LIMIT 20
"""


def leer(cur, days: int) -> list:
    """Ejecuta las consultas (todas SELECT) y devuelve una fila por plan, ya con la proyección clasificada."""
    from shopping.projection.status import classify_projection_jobs  # puro, sin DB

    cur.execute(_SQL_PLANES, (days,))
    cols = [d[0] for d in cur.description]
    filas = [dict(zip(cols, r)) for r in cur.fetchall()]
    for f in filas:
        for k in ("usd", "llm_s", "bloque1_s"):
            f[k] = _f(f.get(k), 3 if k == "usd" else 1)
        cur.execute(_SQL_JOBS, (f["id"],))
        jcols = [d[0] for d in cur.description]
        jobs = [dict(zip(jcols, r)) for r in cur.fetchall()]
        f["proyeccion"] = classify_projection_jobs(f.get("revision"), jobs).get("status")
        f["created_at"] = f["created_at"].isoformat() if f.get("created_at") else None
        f["ultimo_done"] = f["ultimo_done"].isoformat() if f.get("ultimo_done") else None
    return filas


def _imprimir(filas: list, res: dict, days: int) -> None:
    _say(f"— canario de entrega · planes creados en los últimos {days} días: {len(filas)} —")
    _say("plan | cohorte | gs | días (vivos+arch/pedidos) | bloque1 s | chunks done/canc/abiertos/dead | reintentos | "
         "LLM llamadas · US$ · s | swaps | alertas | proyección | válido")
    for f in filas:
        c = clasificar(f)
        coste = (f"{f['llm_llamadas']} · {f['usd']} · {f['llm_s']}" if f.get("llm_llamadas") else "sin atribuir")
        _say(f"  {f['id'][:8]} | {cohorte_de(f)} | {f['gs']} | {f['days_vivos']}+{f['days_archivados']}/{f['dias_pedidos']} | "
             f"{f['bloque1_s']} | {f['done']}/{f['cancelled']}/{f['abiertos']}/{f['dead']} | {f['done_con_reintento']} | "
             f"{coste} | {f['swaps']} | {f['alertas']} | {f['proyeccion']} | {'sí' if c['valido'] else 'NO: ' + '; '.join(c['motivos'])}")
    _say("\ncohorte | n | entregados | válidos | bloque1 p50/p95 s | con reintento | dead | coste atribuidos/sin | "
         "US$ p50/p95/total | LLM s p95 | swaps | alertas | proyección")
    for nombre, r in res.items():
        _say(f"  {nombre} | {r['n']} | {r['entregados']} | {r['validos']} | {r['bloque1_p50_s']}/{r['bloque1_p95_s']} | "
             f"{r['con_reintento']} | {r['dead']} | {r['coste_atribuidos']}/{r['coste_sin_atribuir']} | "
             f"{r['usd_p50']}/{r['usd_p95']}/{r['usd_total']} | {r['llm_s_p95']} | {r['swaps']} | {r['alertas']} | {r['proyeccion']}")
        for ne in r["no_entregados"]:
            _say(f"      no entregado: {ne['plan']} ({'; '.join(ne['motivos'])})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(_BACKEND / ".env")
    import psycopg

    url = os.environ.get("NEON_DATABASE_URL")
    if not url:
        _say("NEON_DATABASE_URL ausente: nada que medir")
        return 2
    days = max(1, int(args.days))
    with psycopg.connect(url, connect_timeout=20) as conn:
        conn.read_only = True
        filas = leer(conn.cursor(), days)
    res = resumen(filas)
    if args.json:
        _say(json.dumps({"days": days, "planes": filas, "resumen": res}, ensure_ascii=False, indent=1, default=str))
    else:
        _imprimir(filas, res, days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
