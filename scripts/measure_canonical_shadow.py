# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-19 · 2026-09-12 · E5-A] Lectura READ-ONLY de la sombra de la lista canónica (ARQ30-P1-01).

La sombra (`canonical_shopping_shadow.py`) corre donde corre el guard de coherencia y persiste en `pipeline_metrics`
(node `canonical_shopping_shadow`) la comparación entre la lista CANÓNICA (gramos de `IngredientLine`) y la lista que
hoy se entrega. Este guion la lee y da el veredicto del gate de la fase A:

    ≥ 30 planes distintos · parse_fail < 1 % · alimentos divergentes (> 10 %) < 5 %  ⇒  fase B posible (cohorte: dueño)

Dos modos, los dos sólo SELECTs (conexión `read_only`):

    python scripts/measure_canonical_shadow.py [--days 30] [--json]     # lo que la sombra ya dejó en pipeline_metrics
    python scripts/measure_canonical_shadow.py --offline [--json]        # calcula la sombra AHORA sobre los planes vivos,
                                                                         # sin escribir métrica: la primera cifra antes de
                                                                         # que la sombra acumule tráfico

En `--offline` el multiplicador es el espejo del que aplica el guard (`effective_multiplier_like_guard`); en producción
el hook del guard le pasa el suyo. Los planes se cuentan por huella de CONTENIDO (`plan_fp`), no por id: el guard no
siempre sabe el id del plan que evalúa. Medido el 2026-09-12 en `--offline`: ver `docs/arq30_e5_e7_diseno_canario.md`.
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


def _pct(n: float, den: float) -> Optional[float]:
    return round(100.0 * n / den, 2) if den else None


def resumir(sombras: list[dict]) -> dict:
    """Agrega una sombra por plan (la más reciente por `plan_fp`) en las dos cifras del gate y el detalle que las explica."""
    from canonical_shopping_shadow import gate_verdict

    por_plan: dict[str, dict] = {}
    for s in sombras:
        fp = s.get("plan_fp") or s.get("plan_id") or f"anon-{len(por_plan)}"
        por_plan[fp] = s   # las sombras vienen ordenadas de más vieja a más nueva: gana la última
    lines = sum(int(s.get("lines") or 0) for s in por_plan.values())
    pf = sum(int(s.get("parse_fail") or 0) for s in por_plan.values())
    comp = sum(int(s.get("comparables") or 0) for s in por_plan.values())
    div = sum(int(s.get("divergentes") or 0) for s in por_plan.values())
    fuentes, superficies, ejemplos = Counter(), Counter(), []
    for s in por_plan.values():
        fuentes.update(s.get("grams_source") or {})
        superficies[str(s.get("surface"))] += 1
        ejemplos.extend(s.get("ejemplos") or [])
    ejemplos.sort(key=lambda x: -(x.get("delta_pct") or 10**9))
    res = {
        "planes": len(por_plan), "sombras": len(sombras), "lines": lines, "parse_fail": pf,
        "parse_fail_pct": _pct(pf, lines), "comparables": comp, "divergentes": div,
        "divergentes_pct": _pct(div, comp),
        "canonical_only": sum(int(s.get("canonical_only") or 0) for s in por_plan.values()),
        "aggregated_only": sum(int(s.get("aggregated_only") or 0) for s in por_plan.values()),
        "no_comparables": sum(int(s.get("no_comparables") or 0) for s in por_plan.values()),
        "sin_gramos": sum(int(s.get("sin_gramos") or 0) for s in por_plan.values()),
        "grams_source": dict(fuentes), "superficies": dict(superficies),
        "planes_tras_shift": sum(1 for s in por_plan.values() if int(s.get("dias_archivados") or 0) > 0),
        "ejemplos": ejemplos[:10],
        "por_plan": [{k: s.get(k) for k in ("plan_fp", "plan_id", "surface", "lines", "parse_fail", "comparables",
                                             "divergentes", "divergentes_pct", "canonical_only", "aggregated_only",
                                             "no_comparables", "dias_archivados", "lista_usada", "multiplier")}
                     for s in por_plan.values()],
    }
    res["veredicto"] = gate_verdict(res["planes"], res["parse_fail_pct"], res["divergentes_pct"])
    return res


def leer_metricas(cur, days: int) -> list[dict]:
    from canonical_shopping_shadow import NODE
    cur.execute(
        "SELECT metadata FROM pipeline_metrics WHERE node = %s AND created_at > now() - make_interval(days => %s) "
        "ORDER BY created_at ASC",
        (NODE, int(days)),
    )
    out = []
    for (meta,) in cur.fetchall():
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                continue
        if isinstance(meta, dict):
            out.append(meta)
    return out


def sombras_offline(cur) -> list[dict]:
    """La sombra calculada AHORA sobre cada plan vivo con lista, sin persistir nada."""
    from canonical_shopping_shadow import compute_shadow, effective_multiplier_like_guard, plan_fingerprint
    cur.execute(
        "SELECT id::text, user_id::text, plan_data FROM meal_plans "
        "WHERE plan_data->'days' IS NOT NULL AND jsonb_array_length(plan_data->'days') > 0 ORDER BY created_at"
    )
    out = []
    for pid, _uid, pd in cur.fetchall():
        pd = pd or {}
        if not (pd.get("aggregated_shopping_list_weekly") or pd.get("aggregated_shopping_list")):
            out.append({"plan_id": pid, "plan_fp": plan_fingerprint(pd), "surface": "offline", "lines": 0,
                        "comparables": 0, "divergentes": 0, "sin_lista": True})
            continue
        r = compute_shadow(pd, multiplier=effective_multiplier_like_guard(pd))
        r.update({"plan_id": pid, "plan_fp": plan_fingerprint(pd), "surface": "offline"})
        out.append(r)
    return out


def render(res: dict) -> str:
    o = [f"planes distintos {res['planes']} · sombras {res['sombras']} · líneas {res['lines']} · "
         f"parse_fail {res['parse_fail']} ({res['parse_fail_pct']} %) · sin gramos {res['sin_gramos']}",
         f"alimentos comparables {res['comparables']} · divergentes >10 % {res['divergentes']} ({res['divergentes_pct']} %) · "
         f"sólo canónica {res['canonical_only']} · sólo lista {res['aggregated_only']} · no comparables {res['no_comparables']}",
         f"gramos por autoridad {res['grams_source']} · superficies {res['superficies']} · planes tras shift {res['planes_tras_shift']}"]
    for p in res["por_plan"]:
        o.append(f"  {str(p.get('plan_id') or p.get('plan_fp'))[:8]} [{p.get('surface')}] líneas {p.get('lines')} · "
                 f"parse_fail {p.get('parse_fail')} · comparables {p.get('comparables')} · divergentes {p.get('divergentes')} "
                 f"({p.get('divergentes_pct')} %) · sólo canónica {p.get('canonical_only')} · sólo lista {p.get('aggregated_only')} · "
                 f"archivados {p.get('dias_archivados')} · lista {p.get('lista_usada')} · ×{p.get('multiplier')}")
    if res["ejemplos"]:
        o.append("  peores divergencias:")
        for e in res["ejemplos"]:
            o.append(f"    {e.get('food')}: canónica {e.get('canonical_g')} g · lista {e.get('aggregated_g')} g · Δ {e.get('delta_pct')} %")
    o.append(res["veredicto"])
    return "\n".join(o)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--offline", action="store_true", help="calcula la sombra ahora sobre los planes vivos, sin escribir")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    from dotenv import load_dotenv  # noqa: E402 — el import vive aquí para que importar el módulo sea puro

    load_dotenv(_BACKEND / ".env")
    url = os.environ.get("NEON_DATABASE_URL")
    if not url:
        _say("falta NEON_DATABASE_URL (backend/.env)")
        return 2
    import psycopg  # noqa: E402

    if args.offline:
        import db_core  # noqa: E402 — el catálogo sale del pool (fuera de FastAPI hay que abrirlo)
        try:
            db_core.connection_pool.open()
        except Exception:
            pass
    with psycopg.connect(url, connect_timeout=20) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            sombras = sombras_offline(cur) if args.offline else leer_metricas(cur, args.days)
    res = resumir(sombras)
    _say(json.dumps(res, ensure_ascii=False, indent=1, default=str) if args.json else render(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
