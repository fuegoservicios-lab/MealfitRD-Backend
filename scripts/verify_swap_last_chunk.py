"""[P1-PLAN-LOTE-15 · 2026-09-12 · E4] Recetas ↔ lista de compras tras swaps y último chunk, VERIFICADAS sobre planes
reales (READ-ONLY, sin escribir una fila).

Criterio de cierre de ARQ27-P1-06: «recetas/compras tras swap y último chunk verificadas, no solo parser de call
sites». Un test que mira si `_recompute_aggregates_after_swap` está cableado no dice si la lista del plan que el
usuario tiene HOY casa con sus recetas. Esto sí lo mide, con el MISMO guard de producción (`run_shopping_coherence_guard`,
no una reimplementación), sobre el `plan_data` persistido:

  · coherencia   divergencias Σrecetas ↔ lista (hipótesis, severas = `_has_severe_divergence`), sobre el plan tal
                 cual está — es decir, DESPUÉS de todos los swaps y del último chunk que lo tocaron.
  · recetas      cada comida trae ingredientes y receta (un swap que persista un plato sin receta se vería aquí).
  · horizonte    días pedidos vs generados (vivos + archivados), estado de la cola (done/cancelados/abiertos/muertos)
                 y si el último chunk cerró.
  · proyección   estado de la proyección de compras para la revisión actual (`classify_projection_jobs`, puro).
  · swaps        llamadas `swap_meal` atribuidas al plan en `llm_usage_events` (existe desde este lote; antes 0).

Cómo evita escribir: el guard emite una métrica a `pipeline_metrics` al terminar; aquí se sustituye por un no-op en
el proceso ANTES de llamarlo. El pool se abre porque la canonicalización del guard lee `master_ingredients`; todo lo
que corre son SELECTs. Los knobs son los de producción (`prod_profile.perfil_aplicado`): un guard medido con la suite
apagada mediría otro producto.

Uso (desde `backend/`):

    python scripts/verify_swap_last_chunk.py [--days 30] [--plan <id o prefijo>] [--json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.append(str(_BACKEND))  # al FINAL: en cabeza, scripts/plan_gym.py sombrea a plan_gym (P1-PLAN-LOTE-13)


# [P2-LOGGER-EXEMPT: CLI de medición; el informe va a stdout a propósito]
def _say(*partes: Any) -> None:
    print(*partes)


def resumir_divergencias(divs: list, severa_fn=None) -> dict:
    """Puro: cuenta divergencias por hipótesis; `severas` usa el criterio del guard si se le pasa (`_has_severe_divergence`),
    y si no, la regla escrita en el doc: `cap_swallowed_modifier` o magnitud > 50 % (excepto `pantry_overdeduct`)."""
    divs = list(divs or [])
    por_hip = Counter(str(d.get("hypothesis") or "unknown") for d in divs)
    if severa_fn is not None:
        try:
            severa = bool(severa_fn(divs))
        except Exception:
            severa = None
    else:
        severa = any(
            d.get("hypothesis") == "cap_swallowed_modifier"
            or (isinstance(d.get("delta_pct"), (int, float)) and abs(float(d["delta_pct"])) > 50
                and d.get("hypothesis") != "pantry_overdeduct")
            for d in divs
        )
    faltan = [str(d.get("food")) for d in divs if d.get("hypothesis") == "cap_swallowed_modifier"]
    return {"total": len(divs), "por_hipotesis": dict(sorted(por_hip.items())), "severa": severa, "en_receta_no_en_lista": faltan[:12]}


def recetas_incompletas(plan_data: dict) -> dict:
    """Puro: comidas sin ingredientes o sin receta en los días vivos (los suplementos no cuentan, como en el aggregator)."""
    total, sin_ing, sin_rec = 0, [], []
    for di, day in enumerate(plan_data.get("days") or []):
        for mi, meal in enumerate((day or {}).get("meals") or []):
            if not isinstance(meal, dict):
                continue
            nombre = str(meal.get("name") or "")
            if "suplemento" in nombre.lower():
                continue
            total += 1
            ings = meal.get("ingredients_raw") or meal.get("ingredients")
            if not ings and isinstance(meal.get("recipe"), dict):
                ings = meal["recipe"].get("ingredients")
            if not ings:
                sin_ing.append(f"d{di}m{mi}:{nombre[:40]}")
            rec = meal.get("recipe")
            if rec is None or (isinstance(rec, (str, list, dict)) and len(rec) == 0):
                sin_rec.append(f"d{di}m{mi}:{nombre[:40]}")
    return {"comidas": total, "sin_ingredientes": sin_ing, "sin_receta": sin_rec}


def horizonte(plan_data: dict, chunks: dict) -> dict:
    """Puro: días pedidos vs generados y si la cola cerró (ningún chunk abierto y al menos uno completado)."""
    pedidos = int(plan_data.get("total_days_requested") or 0)
    vivos = len(plan_data.get("days") or [])
    arch = len(plan_data.get("_archived_days") or [])
    abiertos = int(chunks.get("abiertos") or 0)
    done = int(chunks.get("done") or 0)
    return {
        "pedidos": pedidos, "generados": vivos + arch, "vivos": vivos, "archivados": arch,
        "gs": plan_data.get("generation_status"), "chunks": chunks,
        "ultimo_chunk_cerrado": abiertos == 0 and done > 0,
        "completo": pedidos > 0 and (vivos + arch) >= pedidos,
    }


def veredicto(coherencia: dict, recetas: dict) -> dict:
    """Puro: lo que este script puede afirmar del plan tal cual está persistido.

    Sin comidas vivas (ventana ya archivada) no hay recetas que juzgar: `recetas_ok`/`ok` son None («no concluyente»),
    nunca True — una medición sin datos no puede colapsar a ningún lado."""
    coh_ok = coherencia.get("severa") is False
    if int(recetas.get("comidas") or 0) == 0:
        rec_ok: Optional[bool] = None
    else:
        rec_ok = not recetas.get("sin_ingredientes") and not recetas.get("sin_receta")
    ok: Optional[bool] = None if rec_ok is None else bool(coh_ok and rec_ok)
    return {"coherencia_ok": coh_ok, "recetas_ok": rec_ok, "ok": ok}


_SQL_PLANES = """
SELECT id::text AS id, revision, created_at, plan_data
FROM meal_plans
WHERE created_at > now() - make_interval(days => %s) AND (%s::text IS NULL OR id::text LIKE %s::text)
ORDER BY created_at
"""
_SQL_CHUNKS = """
SELECT count(*) FILTER (WHERE status = 'completed') AS done,
       count(*) FILTER (WHERE status = 'cancelled') AS cancelled,
       count(*) FILTER (WHERE status NOT IN ('completed', 'cancelled') AND dead_lettered_at IS NULL) AS abiertos,
       count(*) FILTER (WHERE dead_lettered_at IS NOT NULL) AS dead,
       max(updated_at) FILTER (WHERE status = 'completed') AS ultimo_done
FROM plan_chunk_queue WHERE meal_plan_id = %s
"""
_SQL_JOBS = """
SELECT id::text AS id, status, plan_revision, attempts, error_code, payload, processed_at, created_at
FROM plan_jobs WHERE plan_id = %s AND job_type = 'shopping_projection' ORDER BY created_at DESC LIMIT 20
"""
_SQL_SWAPS = "SELECT count(*) AS n, max(created_at) AS last FROM llm_usage_events WHERE plan_id = %s AND node = 'swap_meal'"


def verificar(days: int, plan: Optional[str]) -> list:
    """Corre la verificación sobre los planes de la ventana. Abre el pool (catálogo) y anula el emit del guard."""
    from dotenv import load_dotenv
    load_dotenv(_BACKEND / ".env")
    import db_core
    db_core.connection_pool.open()
    from db import execute_sql_query as q
    import shopping_calculator as sc
    import prod_profile
    from shopping.projection.status import classify_projection_jobs

    sc._emit_coherence_guard_metric = lambda **kw: None  # el guard no escribe nada en este proceso
    severa_fn = getattr(sc, "_has_severe_divergence", None)

    like = f"{plan}%" if plan else None
    informes = []
    with prod_profile.perfil_aplicado():
        for row in q(_SQL_PLANES, (days, like, like), fetch_all=True) or []:
            pd = row["plan_data"] if isinstance(row["plan_data"], dict) else json.loads(row["plan_data"])
            chunks = dict(q(_SQL_CHUNKS, (row["id"],), fetch_one=True) or {})
            chunks = {k: (v.isoformat() if hasattr(v, "isoformat") else (int(v) if v is not None else 0)) for k, v in chunks.items()}
            jobs = [dict(j) for j in (q(_SQL_JOBS, (row["id"],), fetch_all=True) or [])]
            swaps = dict(q(_SQL_SWAPS, (row["id"],), fetch_one=True) or {})
            try:
                divs = sc.run_shopping_coherence_guard(pd, mode_override="warn")
                coherencia = resumir_divergencias(divs, severa_fn)
                coherencia["no_evaluable"] = bool(pd.get("_shopping_coherence_guard_unevaluable"))
            except Exception as e:  # el guard es de producción: si revienta aquí, es dato, no ruido
                coherencia = {"total": None, "por_hipotesis": {}, "severa": None, "en_receta_no_en_lista": [], "error": f"{type(e).__name__}: {e}"}
            recetas = recetas_incompletas(pd)
            hor = horizonte(pd, chunks)
            proj = classify_projection_jobs(row.get("revision"), jobs)
            informes.append({
                "plan": row["id"], "revision": row.get("revision"), "created_at": row["created_at"].isoformat(),
                "coherencia": coherencia, "recetas": recetas, "horizonte": hor,
                "proyeccion": {"status": proj.get("status"), "projection_revision": proj.get("projection_revision")},
                "swaps_atribuidos": int(swaps.get("n") or 0),
                "veredicto": veredicto(coherencia, recetas),
            })
    return informes


def _imprimir(informes: list, days: int) -> None:
    _say(f"— recetas ↔ lista tras swaps y último chunk · planes de los últimos {days} días: {len(informes)} —")
    for i in informes:
        c, r, h, v = i["coherencia"], i["recetas"], i["horizonte"], i["veredicto"]
        _say(f"\n{i['plan'][:8]} · rev {i['revision']} · {i['created_at'][:16]} · gs={h['gs']} · días {h['vivos']}+{h['archivados']}/{h['pedidos']}"
             f" · cola done/canc/abiertos/dead = {h['chunks'].get('done')}/{h['chunks'].get('cancelled')}/{h['chunks'].get('abiertos')}/{h['chunks'].get('dead')}"
             f" · último chunk cerrado: {h['ultimo_chunk_cerrado']}")
        _say(f"   coherencia: {c.get('total')} divergencias {c.get('por_hipotesis')} · severa={c.get('severa')}"
             + (f" · ERROR {c['error']}" if c.get("error") else "")
             + (f" · en receta y no en lista: {c['en_receta_no_en_lista']}" if c.get("en_receta_no_en_lista") else ""))
        _say(f"   recetas: {r['comidas']} comidas · sin ingredientes {len(r['sin_ingredientes'])} {r['sin_ingredientes'][:4]} · sin receta {len(r['sin_receta'])} {r['sin_receta'][:4]}")
        _say(f"   proyección: {i['proyeccion']['status']} (rev proyectada {i['proyeccion']['projection_revision']}) · swaps atribuidos: {i['swaps_atribuidos']}")
        estado = {True: "OK", False: "NO OK", None: "NO CONCLUYENTE (sin comidas vivas)"}[v["ok"]]
        _say(f"   VEREDICTO: {estado} (coherencia {v['coherencia_ok']}, recetas {v['recetas_ok']})")
    oks = sum(1 for i in informes if i["veredicto"]["ok"] is True)
    nc = sum(1 for i in informes if i["veredicto"]["ok"] is None)
    _say(f"\nTOTAL: {oks}/{len(informes)} planes con recetas↔lista coherentes y recetas completas tal cual están persistidos"
         + (f" ({nc} no concluyentes: sin comidas vivas)." if nc else "."))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--plan", default=None, help="id (o prefijo) del plan; por defecto todos los de la ventana")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if not os.environ.get("NEON_DATABASE_URL"):
        from dotenv import load_dotenv
        load_dotenv(_BACKEND / ".env")
    if not os.environ.get("NEON_DATABASE_URL"):
        _say("NEON_DATABASE_URL ausente: nada que verificar")
        return 2
    informes = verificar(max(1, int(args.days)), args.plan)
    if args.json:
        _say(json.dumps(informes, ensure_ascii=False, indent=1, default=str))
    else:
        _imprimir(informes, max(1, int(args.days)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
