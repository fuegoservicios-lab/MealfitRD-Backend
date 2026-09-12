"""[P1-PLAN-LOTE-14 · 2026-09-12 · D8] Medición READ-ONLY del lag de `plan_jobs` (Fase 5 del roadmap 2.5).

El gate de la Fase 5 (`docs/plan_jobs_f5.md` → Observabilidad) pide «lag p95 < 2 min en canary y cero `dead` sin
alerta». Aquí se miden DOS lags, porque miden cosas distintas:

  · `created → processed` (todos los jobs terminados): lo que espera el USUARIO desde que el disparador encola
    hasta que el consumidor termina. Incluye reintentos con backoff, jobs muertos y revividos, y las cadenas
    `revision_changed` (el reconcile re-encola cada `MEALFIT_I18N_RECONCILE_INTERVAL_MIN`).
  · el «camino limpio» (`done` al primer intento CON trabajo real): se descompone en RECOGIDA
    (`heartbeat_at − max(execute_after, created_at)`: el tick del worker, acotado por
    `MEALFIT_PLAN_JOBS_WORKER_INTERVAL_S`) y CONSUMO (`processed_at − heartbeat_at`: el LLM o la proyección).
    El gate juzga la RECOGIDA, que es lo que el worker controla; el consumo se informa. Los jobs que terminan en
    el mismo segundo en que nacen son no-ops (nada que traducir) y se cuentan aparte: meterlos en el p95 lo
    esconde. Sin jobs limpios el veredicto es «no concluyente», nunca «pasa»: una medición sin datos no puede
    colapsar a ningún lado.

Sólo SELECTs; abre la conexión en `read_only`. Uso (desde `backend/`):

    python scripts/measure_plan_jobs_lag.py [--days 30] [--json]

Medido el 2026-09-12 (ver `docs/plan_jobs_f5.md` → «Medición del 2026-09-12»).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.append(str(_BACKEND))  # al FINAL: en cabeza, scripts/plan_gym.py sombrea a plan_gym (P1-PLAN-LOTE-13)

#: Cota del gate de la Fase 5 para la RECOGIDA del worker (p95 de `heartbeat_at − max(execute_after, created_at)`
#: en jobs `done` al primer intento con trabajo real), en segundos.
GATE_P95_S = 120.0


# [P2-LOGGER-EXEMPT: CLI de medición; el informe va a stdout a propósito]
def _say(*partes: Any) -> None:
    print(*partes)


def _f(v: Any) -> Optional[float]:
    return None if v is None else round(float(v), 1)


def veredicto(stats: dict[str, dict], dead_sin_alerta: int, cota_s: float = GATE_P95_S) -> dict:
    """Veredicto PURO del gate de la Fase 5 a partir de las estadísticas por `job_type`.

    `stats[t]["p95_recogida_s"]` es el p95 de la recogida del worker en jobs limpios con trabajo real (None si no
    hubo ninguno). Devuelve por tipo `recogida_ok` ∈ {True, False, None} y un `gate_ok` global que es None («no
    concluyente») cuando algún tipo no tiene datos, False si alguna recogida supera la cota o hay `dead` sin alerta,
    True si todo pasa. `p95_limpio_s` (total que ve el usuario) y `p95_consumo_s` viajan como información.
    """
    por_tipo: dict[str, dict] = {}
    for t, st in sorted(stats.items()):
        p95 = st.get("p95_recogida_s")
        por_tipo[t] = {
            "p95_recogida_s": p95,
            "recogida_ok": None if p95 is None else bool(float(p95) <= cota_s),
            "p95_consumo_s": st.get("p95_consumo_s"),
            "p95_limpio_s": st.get("p95_limpio_s"),
            "n_limpio": int(st.get("n_limpio") or 0),
            "n_noop": int(st.get("n_noop") or 0),
        }
    oks = [v["recogida_ok"] for v in por_tipo.values()]
    if not oks or any(o is None for o in oks):
        gate_ok: Optional[bool] = None
    else:
        gate_ok = all(oks) and int(dead_sin_alerta) == 0
    return {"gate_ok": gate_ok, "cota_p95_s": cota_s, "dead_sin_alerta": int(dead_sin_alerta), "por_tipo": por_tipo}


def medir(cur, days: int) -> dict:
    """Ejecuta las consultas (todas SELECT) y devuelve el informe como dict serializable."""
    out: dict[str, Any] = {"days": days}

    cur.execute(
        """
        SELECT job_type, status, count(*)
        FROM plan_jobs
        WHERE created_at > now() - make_interval(days => %s)
        GROUP BY 1, 2 ORDER BY 1, 2
        """,
        (days,),
    )
    out["por_estado"] = [{"job_type": r[0], "status": r[1], "n": int(r[2])} for r in cur.fetchall()]

    cur.execute(
        """
        SELECT job_type,
               count(*) AS n,
               percentile_cont(0.5)  WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (processed_at - created_at))) AS p50_s,
               percentile_cont(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (processed_at - created_at))) AS p95_s,
               max(EXTRACT(EPOCH FROM (processed_at - created_at))) AS max_s,
               percentile_cont(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (processed_at - created_at)))
                   FILTER (WHERE status = 'done' AND coalesce(attempts, 0) <= 1
                           AND processed_at > created_at + INTERVAL '1 second') AS p95_limpio_s,
               count(*) FILTER (WHERE status = 'done' AND coalesce(attempts, 0) <= 1
                                AND processed_at > created_at + INTERVAL '1 second') AS n_limpio,
               count(*) FILTER (WHERE status = 'done' AND coalesce(attempts, 0) <= 1
                                AND processed_at <= created_at + INTERVAL '1 second') AS n_noop,
               percentile_cont(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (heartbeat_at - GREATEST(execute_after, created_at))))
                   FILTER (WHERE status = 'done' AND coalesce(attempts, 0) <= 1 AND heartbeat_at IS NOT NULL
                           AND processed_at > created_at + INTERVAL '1 second') AS p95_recogida_s,
               percentile_cont(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (processed_at - heartbeat_at)))
                   FILTER (WHERE status = 'done' AND coalesce(attempts, 0) <= 1 AND heartbeat_at IS NOT NULL
                           AND processed_at > created_at + INTERVAL '1 second') AS p95_consumo_s,
               percentile_cont(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (processed_at - execute_after))) AS p95_desde_execute_after_s,
               avg(coalesce(attempts, 0)) AS avg_attempts
        FROM plan_jobs
        WHERE processed_at IS NOT NULL AND created_at > now() - make_interval(days => %s)
        GROUP BY 1 ORDER BY 1
        """,
        (days,),
    )
    stats: dict[str, dict] = {}
    for r in cur.fetchall():
        stats[r[0]] = {
            "n": int(r[1]), "p50_s": _f(r[2]), "p95_s": _f(r[3]), "max_s": _f(r[4]),
            "p95_limpio_s": _f(r[5]), "n_limpio": int(r[6] or 0), "n_noop": int(r[7] or 0),
            "p95_recogida_s": _f(r[8]), "p95_consumo_s": _f(r[9]),
            "p95_desde_execute_after_s": _f(r[10]), "avg_attempts": _f(r[11]),
        }
    out["lag"] = stats

    cur.execute(
        """
        SELECT metadata->>'job_type', metadata->>'status', metadata->>'error_code', count(*)
        FROM pipeline_metrics
        WHERE node = 'plan_jobs' AND created_at > now() - make_interval(days => %s)
          AND metadata->>'status' IN ('failed', 'dead', 'stale', 'fencing_rejected')
        GROUP BY 1, 2, 3 ORDER BY 4 DESC, 1, 2
        """,
        (days,),
    )
    out["motivos_no_done"] = [{"job_type": r[0], "status": r[1], "error_code": r[2], "n": int(r[3])} for r in cur.fetchall()]

    cur.execute(
        """
        SELECT id::text, job_type, plan_id::text, attempts, error_code, dead_lettered_at
        FROM plan_jobs WHERE status = 'dead' ORDER BY dead_lettered_at DESC NULLS LAST LIMIT 50
        """
    )
    out["dead"] = [
        {"id": r[0], "job_type": r[1], "plan_id": r[2], "attempts": r[3], "error_code": r[4],
         "dead_lettered_at": r[5].isoformat() if r[5] else None} for r in cur.fetchall()
    ]

    cur.execute(
        """
        SELECT count(*)
        FROM plan_jobs j
        WHERE j.status = 'dead'
          AND NOT EXISTS (SELECT 1 FROM system_alerts a
                          WHERE a.alert_type = 'plan_jobs_dead' AND a.metadata->>'job_id' = j.id::text)
        """
    )
    out["dead_sin_alerta"] = int(cur.fetchone()[0] or 0)

    cur.execute(
        """
        SELECT job_type, status, count(*), max(EXTRACT(EPOCH FROM (now() - created_at))) / 3600.0
        FROM plan_jobs
        WHERE processed_at IS NULL AND status <> 'dead'
        GROUP BY 1, 2 ORDER BY 1, 2
        """
    )
    out["backlog"] = [{"job_type": r[0], "status": r[1], "n": int(r[2]), "oldest_h": _f(r[3])} for r in cur.fetchall()]

    out["veredicto"] = veredicto(stats, out["dead_sin_alerta"])
    return out


def _imprimir(out: dict) -> None:
    _say(f"— plan_jobs · últimos {out['days']} días —")
    _say("job_type | status | n")
    for r in out["por_estado"]:
        _say(f"  {r['job_type']} | {r['status']} | {r['n']}")
    _say("\njob_type | n | p50 | p95 | max | avg attempts   [created→processed, segundos]")
    for t, s in out["lag"].items():
        _say(f"  {t} | {s['n']} | {s['p50_s']} | {s['p95_s']} | {s['max_s']} | {s['avg_attempts']}")
    _say("\ncamino limpio (done al 1.er intento con trabajo real): job_type | n limpio | n no-op | p95 recogida | "
         "p95 consumo | p95 total | p95 desde execute_after   [segundos]")
    for t, s in out["lag"].items():
        _say(f"  {t} | {s['n_limpio']} | {s['n_noop']} | {s['p95_recogida_s']} | {s['p95_consumo_s']} | "
             f"{s['p95_limpio_s']} | {s['p95_desde_execute_after_s']}")
    _say("\nmotivos de los no-done (pipeline_metrics.plan_jobs):")
    for r in out["motivos_no_done"] or [{"job_type": "—", "status": "—", "error_code": "—", "n": 0}]:
        _say(f"  {r['job_type']} | {r['status']} | {r['error_code']} | {r['n']}")
    _say(f"\ndead: {len(out['dead'])} (sin alerta: {out['dead_sin_alerta']})")
    for d in out["dead"]:
        _say(f"  {d['id'][:8]} {d['job_type']} plan={str(d['plan_id'])[:8]} attempts={d['attempts']} "
             f"error={d['error_code']} {d['dead_lettered_at']}")
    _say(f"backlog vivo: {out['backlog'] or 'vacío'}")
    v = out["veredicto"]
    estado = {True: "PASA", False: "NO PASA", None: "NO CONCLUYENTE (sin jobs limpios en algún tipo)"}[v["gate_ok"]]
    _say(f"\nVEREDICTO gate Fase 5 (p95 de RECOGIDA ≤ {v['cota_p95_s']:.0f} s y 0 dead sin alerta): {estado}")
    for t, r in v["por_tipo"].items():
        _say(f"  {t}: recogida p95 = {r['p95_recogida_s']} s (consumo {r['p95_consumo_s']} s, total {r['p95_limpio_s']} s) "
             f"sobre {r['n_limpio']} jobs con trabajo (+{r['n_noop']} no-op) → {r['recogida_ok']}")


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
    with psycopg.connect(url, connect_timeout=20) as conn:
        conn.read_only = True
        out = medir(conn.cursor(), max(1, int(args.days)))
    if args.json:
        _say(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    else:
        _imprimir(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
