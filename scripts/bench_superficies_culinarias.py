# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-28 · 2026-09-12] C5 · CUL-P1-07: benchmark culinario de TODAS las superficies que tocan una receta, con
dobles offline y un modo real explícito con presupuesto.

    python scripts/bench_superficies_culinarias.py                       # corpus fijo · dobles offline · informe por superficie
    python scripts/bench_superficies_culinarias.py --out X.json          # guarda el artefacto (por defecto scripts/data/bench_superficies_<fecha>.json)
    python scripts/bench_superficies_culinarias.py --informe A.json      # reconstruye el informe SOLO desde el artefacto
    python scripts/bench_superficies_culinarias.py --comparar A.json B.json   # informe PAREADO: mismo caso, dos versiones
    python scripts/bench_superficies_culinarias.py --exportar-catalogo   # (base, sólo lectura) snapshot del catálogo con nutrición
    python scripts/bench_superficies_culinarias.py --real --perfil f.json --presupuesto-usd 0.50   # genera con LLM: se NIEGA sin presupuesto

## Qué mide

Cada superficie es una CADENA determinista que muta el plan (`cadena(plan_data)`), y se mide con `repair_stage_diff.medir_cadena`:
foto culinaria (capa 1) a la entrada y a la salida, hallazgos NUEVOS, resueltos, ms, y si el contrato final de la receta
(`_recipe_contract_final`) quedó estampado en las comidas. Las superficies y su adaptador, con lo que replican:

  insert         `db_plans._finalize_plan_data_for_insert({"plan_data": p})`        el INSERT del plan (SSOT del orden)
  quality        `db_plans.apply_plan_quality_finalize_chain(p)`                   cola de assemble / T2 / recovery
  chunk_t2       …idem con `freeze_past_days=True`                                 merge del chunk con días congelados
  swap           `finalize_single_meal_recipe_coherence` × comida → `apply_update_band_parity` → `apply_final_contract`
                                                                                    lo que hace `/swap-meal/persist` sin la base
  modify         la misma cadena que swap (el callback de `tools.modify_single_meal` corre exactamente esos tres pasos)
  closers        `reconcile_protein_band_post_finalize` → `_cap_unrealistic_portions`  cerradores de banda y caps de realismo
  degradado      `culinary_contract_scan` → `cron_tasks._degrade_offending_steps` por día   el postfix del día sin LLM (Smart Shuffle)
  expand         identidad — `/recipe/expand` persiste `expanded_recipe` (LLM) sin cadena determinista propia: se dice, no se finge

## Dobles offline y modo real

Offline, el catálogo sale del corpus fijo (nombres, alias, categoría: lo que el escáner necesita) o, mejor, de un SNAPSHOT con
nutrición (`--exportar-catalogo`, una vez, con base) para que los cerradores tengan gramos y kcal reales. El artefacto
declara `modo_catalogo` («nutricion» | «nombres»): sin nutrición los cerradores de banda no tienen con qué cerrar y sus cifras
NO son comparables con producción — el informe lo dice en la cabecera. Nada de aquí escribe en la base (los planes son copias;
las conexiones, si las hay, son de sólo lectura). `--real` genera planes con el LLM y exige `--presupuesto-usd`: sin cifra
se niega, y registra en el artefacto intentos, coste estimado y estado de cada generación.

Solo lectura. tooltip-anchor: P1-PLAN-LOTE-28-BENCH-SUPERFICIES
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

_BACKEND = Path(os.environ.get("MEALFIT_BACKEND_DIR") or Path(__file__).resolve().parents[1])   # override para correrlo desde fuera
if str(_BACKEND) not in sys.path:
    sys.path.append(str(_BACKEND))

SUPERFICIES = ("insert", "quality", "chunk_t2", "swap", "modify", "closers", "degradado", "expand")


# ─────────────────────────────────────────────────────────────────────────────────────────────
# catálogo (offline: corpus o snapshot con nutrición)
# ─────────────────────────────────────────────────────────────────────────────────────────────

def _ultimo(patron: str):
    fs = sorted(glob.glob(str(_BACKEND / "scripts" / "data" / patron)))
    return fs[-1] if fs else None


def cargar_catalogo(path: str | None) -> tuple[list, str, str | None]:
    """`(filas, modo, ruta)`: `nutricion` si el snapshot trae kcal; `nombres` si sólo hay el catálogo del corpus."""
    p = path or _ultimo("catalogo_nutricion_*.json")
    if p and os.path.exists(p):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        filas = d.get("filas") if isinstance(d, dict) else d
        if filas and any("kcal_per_100g" in (f or {}) for f in filas[:20]):
            return list(filas), "nutricion", p
    c = _ultimo("culinary_corpus_*.json")
    if c:
        d = json.loads(Path(c).read_text(encoding="utf-8"))
        return list(d.get("catalogo_filas") or []), "nombres", c
    return [], "sin_catalogo", None


def exportar_catalogo(out: str | None) -> str:
    """Snapshot del catálogo CON nutrición para los dobles offline. Necesita la base (sólo lectura)."""
    try:
        from db_core import connection_pool           # fuera de FastAPI el pool nace cerrado (runbook SQL forense)
        connection_pool.open()
    except Exception:
        pass
    from shopping_calculator import get_master_ingredients
    filas = get_master_ingredients() or []
    if not filas:
        raise SystemExit("catálogo vacío: ¿pool abierto? (db_core.connection_pool.open())")
    p = Path(out) if out else _BACKEND / "scripts" / "data" / f"catalogo_nutricion_{date.today():%Y_%m_%d}.json"
    p.write_text(json.dumps({"exportado_el": date.today().isoformat(), "n": len(filas), "filas": filas},
                            ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return str(p)


def _instalar_dobles(filas: list) -> None:
    """Los adaptadores construyen `IngredientNutritionDB()` por su cuenta y leen `get_master_ingredients()`: se les da el
    catálogo del benchmark sin tocar la base."""
    import shopping_calculator as sc
    sc._master_cache = list(filas)
    sc._master_cache_ts = time.time() + 10 ** 9      # que no caduque durante la corrida
    try:
        import nutrition_db as ndb
        ndb.IngredientNutritionDB._BENCH_ROWS = list(filas)   # marca, por si alguien la mira
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────────────────────
# adaptadores: una cadena determinista por superficie
# ─────────────────────────────────────────────────────────────────────────────────────────────

def _db(filas):
    from nutrition_db import IngredientNutritionDB
    return IngredientNutritionDB(rows=list(filas)) if filas else IngredientNutritionDB()


def _ad_insert(p, filas):
    import db_plans
    db_plans._finalize_plan_data_for_insert({"plan_data": p}, surface="bench-insert")


def _ad_quality(p, filas):
    import db_plans
    db_plans.apply_plan_quality_finalize_chain(p, surface="bench-quality")


def _ad_chunk_t2(p, filas):
    import db_plans
    db_plans.apply_plan_quality_finalize_chain(p, surface="bench-chunk-t2", freeze_past_days=True)


def _ad_swap(p, filas):
    import graph_orchestrator as go
    from recipe_contract import apply_final_contract
    db = _db(filas)
    for d in p.get("days") or []:
        for m in d.get("meals") or []:
            go.finalize_single_meal_recipe_coherence(m, db)
    go.apply_update_band_parity(p, surface="swap_persist")
    apply_final_contract(p.get("days") or [], db)


def _ad_closers(p, filas):
    import graph_orchestrator as go
    go.reconcile_protein_band_post_finalize(p)
    go._cap_unrealistic_portions(p.get("days") or [], db=_db(filas))


def _ad_degradado(p, filas):
    import cron_tasks
    from culinary_coherence import culinary_contract_scan, build_culinary_index
    index = build_culinary_index(filas)
    for d in p.get("days") or []:
        # como el cron: sólo V1/V2 degradan (P1-CULINARY-CONTRACT/degradado); la primera versión de este adaptador
        # pasaba TODOS los checks y «medía» 21 hallazgos nuevos que el cron jamás habría producido
        viol = [v for v in culinary_contract_scan({"days": [d]}, filas) if v.get("check") in ("V1", "V2")]
        if viol:
            cron_tasks._degrade_offending_steps(d, viol, index)


def _ad_expand(p, filas):
    return None      # `/recipe/expand` persiste lo que el LLM devuelve: no hay cadena determinista que medir


ADAPTADORES = {"insert": _ad_insert, "quality": _ad_quality, "chunk_t2": _ad_chunk_t2, "swap": _ad_swap,
               "modify": _ad_swap, "closers": _ad_closers, "degradado": _ad_degradado, "expand": _ad_expand}


def _contrato_estampado(p) -> int:
    return sum(1 for d in (p.get("days") or []) for m in (d.get("meals") or []) if isinstance(m, dict) and m.get("_recipe_contract_final"))


def correr(planes: list, filas: list, superficies=SUPERFICIES, *, modo: str = "offline") -> dict:
    """El artefacto: por superficie y por plan, el informe de `repair_stage_diff` más lo que el adaptador dejó estampado."""
    import repair_stage_diff as rsd
    from culinary_coherence import rules_fingerprint
    _instalar_dobles(filas)
    out = {"schema_version": 1, "modo": modo, "fecha": date.today().isoformat(), "git_sha": _git_sha(),
           "reglas_huella": rules_fingerprint(), "n_planes": len(planes), "superficies": {}}
    for s in superficies:
        ad = ADAPTADORES[s]
        filas_s = []
        for pl in planes:
            p = copy.deepcopy(pl.get("plan_data") or pl)
            t0 = time.monotonic()
            if s == "expand":
                inf = {"estado": "sin_cadena_determinista", "nuevos": [], "n_nuevos": 0, "resueltos": 0, "etapas": {}}
            else:
                inf = rsd.medir_cadena(p, lambda q, _ad=ad: _ad(q, filas), surface=f"bench-{s}", catalog=filas) or {"estado": "no_medido"}
            filas_s.append({"plan_id": pl.get("plan_id"), "estado": inf.get("estado"), "n_nuevos": inf.get("n_nuevos", 0),
                            "resueltos": inf.get("resueltos", 0), "etapas": inf.get("etapas", {}),
                            "nuevos": (inf.get("nuevos") or [])[:20], "contrato_estampado": _contrato_estampado(p),
                            "ms": int((time.monotonic() - t0) * 1000)})
        out["superficies"][s] = {"planes": filas_s, "n_nuevos": sum(f["n_nuevos"] for f in filas_s),
                                 "resueltos": sum(f["resueltos"] for f in filas_s),
                                 "contrato_estampado": sum(f["contrato_estampado"] for f in filas_s),
                                 "ms": sum(f["ms"] for f in filas_s)}
    return out


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(_BACKEND), text=True).strip()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────────────────────
# informes
# ─────────────────────────────────────────────────────────────────────────────────────────────

def render(art: dict) -> str:
    cab = (f"bench superficies · {art.get('fecha')} · git {art.get('git_sha')} · reglas {art.get('reglas_huella')} · "
           f"modo {art.get('modo')} · catálogo {art.get('modo_catalogo', '?')} · planes {art.get('n_planes')}")
    if art.get("modo_catalogo") == "nombres":
        cab += "\n⚠ catálogo SIN nutrición: los cerradores de banda no tienen con qué cerrar; sus cifras no son comparables con producción"
    out = [cab, f"{'superficie':<11} {'nuevos':>7} {'resueltos':>10} {'contrato':>9} {'ms':>7}  estado"]
    for s, r in (art.get("superficies") or {}).items():
        estados = sorted({str(p.get("estado")) for p in r.get("planes") or []})
        out.append(f"{s:<11} {r.get('n_nuevos', 0):>7} {r.get('resueltos', 0):>10} {r.get('contrato_estampado', 0):>9} {r.get('ms', 0):>7}  {','.join(estados)}")
    return "\n".join(out)


def comparar(a: dict, b: dict) -> str:
    out = [f"pareado · A {a.get('git_sha')}/{a.get('reglas_huella')} → B {b.get('git_sha')}/{b.get('reglas_huella')}"]
    if a.get("modo_catalogo") != b.get("modo_catalogo") or a.get("n_planes") != b.get("n_planes"):
        out.append("⚠ los dos artefactos no miden el MISMO caso (catálogo o nº de planes distintos): la comparación no es pareada")
    out.append(f"{'superficie':<11} {'nuevos A→B':>12} {'resueltos A→B':>15} {'contrato A→B':>14}")
    for s in sorted(set(a.get("superficies") or {}) | set(b.get("superficies") or {})):
        ra, rb = (a.get("superficies") or {}).get(s, {}), (b.get("superficies") or {}).get(s, {})
        out.append(f"{s:<11} {ra.get('n_nuevos', '-'):>5} → {rb.get('n_nuevos', '-'):<4} {ra.get('resueltos', '-'):>6} → {rb.get('resueltos', '-'):<6} "
                   f"{ra.get('contrato_estampado', '-'):>6} → {rb.get('contrato_estampado', '-'):<5}")
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────────────────────
# modo real (explícito, con presupuesto)
# ─────────────────────────────────────────────────────────────────────────────────────────────

def generar_real(perfiles: list, presupuesto_usd: float) -> list:
    """Genera planes con el LLM, uno por perfil, y para en cuanto el coste estimado supere el presupuesto. Devuelve la lista de
    `{"plan_id", "plan_data", "estado", "s", "coste_usd_est"}`; el coste se estima con `llm_usage_events` si la base está."""
    import asyncio
    from graph_orchestrator import arun_plan_pipeline
    out, gastado = [], 0.0
    for i, fd in enumerate(perfiles):
        if gastado >= presupuesto_usd:
            out.append({"plan_id": f"real-{i}", "estado": "no_generado:presupuesto_agotado", "plan_data": None})
            continue
        t0 = time.time()
        try:
            plan = asyncio.run(arun_plan_pipeline(dict(fd)))
            coste = _coste_estimado(t0)
            gastado += coste
            out.append({"plan_id": f"real-{i}", "plan_data": plan, "estado": "generado", "s": round(time.time() - t0, 1), "coste_usd_est": coste})
        except Exception as e:                                                 # noqa: BLE001
            out.append({"plan_id": f"real-{i}", "estado": f"error:{type(e).__name__}", "plan_data": None, "s": round(time.time() - t0, 1)})
    return out


def _coste_estimado(desde_ts: float) -> float:
    try:
        from db import execute_sql_query
        row = execute_sql_query("SELECT COALESCE(SUM(cost_usd_micros), 0) AS micros FROM llm_usage_events WHERE created_at >= to_timestamp(%s)",
                                (desde_ts,), fetch_one=True)
        return float((row or {}).get("micros") or 0) / 1e6
    except Exception:
        return 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogo", help="snapshot del catálogo con nutrición (por defecto el más reciente en scripts/data)")
    ap.add_argument("--corpus", help="corpus culinario fijo (por defecto el más reciente)")
    ap.add_argument("--superficies", default=",".join(SUPERFICIES))
    ap.add_argument("--out", help="artefacto JSON (por defecto scripts/data/bench_superficies_<fecha>.json)")
    ap.add_argument("--sin-guardar", action="store_true")
    ap.add_argument("--informe", help="reconstruye el informe desde un artefacto y sale")
    ap.add_argument("--comparar", nargs=2, metavar=("A", "B"), help="informe pareado entre dos artefactos")
    ap.add_argument("--exportar-catalogo", action="store_true", help="(base, sólo lectura) snapshot del catálogo con nutrición")
    ap.add_argument("--real", action="store_true", help="genera planes con el LLM (exige --perfil y --presupuesto-usd)")
    ap.add_argument("--perfil", action="append", help="JSON de form_data para --real (repetible)")
    ap.add_argument("--presupuesto-usd", type=float, help="tope de gasto estimado para --real")
    a = ap.parse_args(argv)

    if a.exportar_catalogo:
        print("catálogo exportado:", exportar_catalogo(a.out))
        return 0
    if a.informe:
        print(render(json.loads(Path(a.informe).read_text(encoding="utf-8"))))
        return 0
    if a.comparar:
        A, B = (json.loads(Path(x).read_text(encoding="utf-8")) for x in a.comparar)
        print(comparar(A, B))
        return 0

    filas, modo_cat, ruta_cat = cargar_catalogo(a.catalogo)
    if not filas:
        print("sin catálogo (ni snapshot ni corpus): nada que medir")
        return 2
    superficies = tuple(s.strip() for s in a.superficies.split(",") if s.strip() in SUPERFICIES)

    if a.real:
        if not a.perfil or a.presupuesto_usd is None or a.presupuesto_usd <= 0:
            print("--real exige --perfil <json> (repetible) y --presupuesto-usd > 0: sin presupuesto no se gasta")
            return 2
        perfiles = [json.loads(Path(x).read_text(encoding="utf-8")) for x in a.perfil]
        generados = generar_real(perfiles, float(a.presupuesto_usd))
        planes = [g for g in generados if g.get("plan_data")]
        art = correr(planes, filas, superficies, modo="real")
        art["generaciones"] = [{k: v for k, v in g.items() if k != "plan_data"} for g in generados]
        art["presupuesto_usd"] = a.presupuesto_usd
    else:
        c = a.corpus or _ultimo("culinary_corpus_*.json")
        if not c:
            print("sin corpus culinario fijo en scripts/data")
            return 2
        from culinary_corpus import cargar, filas_para_medir
        planes = filas_para_medir(cargar(c))
        art = correr(planes, filas, superficies, modo="offline")
        art["corpus"] = os.path.basename(c)
    art["modo_catalogo"], art["catalogo"] = modo_cat, (os.path.basename(ruta_cat) if ruta_cat else None)
    print(render(art))
    if not a.sin_guardar:
        p = Path(a.out) if a.out else _BACKEND / "scripts" / "data" / f"bench_superficies_{date.today():%Y_%m_%d}.json"
        p.write_text(json.dumps(art, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
        print("artefacto:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
