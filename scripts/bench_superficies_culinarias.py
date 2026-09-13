# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-28 · 2026-09-12] C5 · CUL-P1-07: benchmark culinario de TODAS las superficies que tocan una receta, con
dobles offline y un modo real explícito con presupuesto.

    python scripts/bench_superficies_culinarias.py                       # corpus fijo · dobles offline · informe por superficie
    python scripts/bench_superficies_culinarias.py --out X.json          # guarda el artefacto (por defecto scripts/data/bench_superficies_<fecha>.json)
    python scripts/bench_superficies_culinarias.py --informe A.json      # reconstruye el informe SOLO desde el artefacto
    python scripts/bench_superficies_culinarias.py --comparar A.json B.json   # informe PAREADO: mismo caso, dos versiones
    python scripts/bench_superficies_culinarias.py --planes-de real.json      # re-mide SIN LLM los planes de un artefacto real (pareado real)
    python scripts/bench_superficies_culinarias.py --exportar-catalogo   # (base, sólo lectura) snapshot del catálogo con nutrición
    python scripts/bench_superficies_culinarias.py --real --perfil f.json --presupuesto-usd 0.50   # genera con LLM como /analyze: se NIEGA sin presupuesto y no escribe en la base
    python scripts/bench_superficies_culinarias.py --real ... --telemetria-prod                     # (opcional) deja que la telemetría del pipeline caiga en producción

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
se niega.

[P1-PLAN-LOTE-30 · 2026-09-12] El modo real imita a `/analyze` (`_plan_start_date`, `_days_to_generate = PLAN_CHUNK_SIZE` — el
bloque síncrono son 3 días, el resto lo hace la cola — y la rebanada del blueprint), abre los pools como el arranque de la app y,
por defecto, SUSTITUYE las funciones de escritura de `db_core` por dobles que cuentan y no ejecutan: el pipeline LEE la base
real (catálogo, registro, política) y su telemetría (`pipeline_metrics`, `app_kv_store`, `system_alerts`, `llm_usage_events`)
no cae en producción. El coste se suma EN PROCESO con la tarifa del propio emisor (`compute_llm_cost_micros`), así que el
presupuesto funciona sin leer la base; `--telemetria-prod` deja pasar esas escrituras. El artefacto registra por generación
perfil, estado, días, comidas, segundos y coste, y guarda los planes generados: son la evidencia. Lo que NO imita: el
`taste_profile` del router (LLM sobre el historial) — un usuario nuevo no tiene historial.

Solo lectura. tooltip-anchor: P1-PLAN-LOTE-28-BENCH-SUPERFICIES
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
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
    for g in art.get("generaciones") or []:
        out.append(f"  {g.get('plan_id')} · {g.get('perfil', '')} · {g.get('estado')} · {g.get('s', 0)} s · {g.get('dias', 0)} d / "
                   f"{g.get('comidas', 0)} comidas · {g.get('llamadas_llm', 0)} llamadas · ${float(g.get('coste_usd_est') or 0):.4f}")
    if art.get("modo") == "real":
        sup = art.get("escrituras_suprimidas") or {}
        out.append(f"  coste estimado ${float(art.get('coste_usd_est') or 0):.4f} de ${float(art.get('presupuesto_usd') or 0):.2f} · "
                   f"telemetría a producción: {'sí' if art.get('telemetria_prod') else 'no'} · escrituras suprimidas: "
                   + (", ".join(f"{k} {v}" for k, v in sup.items()) or "0"))
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
# modo real (explícito, con presupuesto): imita a /analyze y no escribe en la base del dueño
# [P1-PLAN-LOTE-30 · 2026-09-12] tooltip-anchor: P1-PLAN-LOTE-30-BENCH-REAL
# ─────────────────────────────────────────────────────────────────────────────────────────────

_ESCRITURAS: dict = {}      # tabla → escrituras suprimidas en la corrida (modo real sin --telemetria-prod)
_TABLA_RE = re.compile(r"(?is)\b(?:insert\s+into|update|delete\s+from)\s+([a-z_][a-z0-9_.]*)")


def _tabla(sql) -> str:
    m = _TABLA_RE.search(str(sql or ""))
    return m.group(1).lower() if m else "?"


def bloquear_escrituras(mod=None, contador: dict | None = None) -> dict:
    """Sustituye las funciones de ESCRITURA (`execute_sql_write`, `aexecute_sql_write`, `execute_sql_transaction`) por dobles
    que cuentan por tabla y no ejecutan; las lecturas siguen yendo a la base real. Llamar ANTES de importar el orquestador:
    los módulos enlazan el nombre al importarlo. Parchea también la fachada `db` si ya está cargada."""
    if mod is None:
        import db_core as mod
    contador = _ESCRITURAS if contador is None else contador

    def _w(query, params=None, returning=False, **kw):
        t = _tabla(query)
        contador[t] = contador.get(t, 0) + 1
        return [] if returning else True

    async def _aw(query, params=None, returning=False, **kw):
        return _w(query, params, returning)

    def _tx(queries_with_params):
        for q, _p in (queries_with_params or []):
            _w(q)
        return True

    mod.execute_sql_write, mod.aexecute_sql_write, mod.execute_sql_transaction = _w, _aw, _tx
    if getattr(mod, "__name__", "") == "db_core":
        fachada = sys.modules.get("db")
        if fachada is not None:
            fachada.execute_sql_write, fachada.aexecute_sql_write, fachada.execute_sql_transaction = _w, _aw, _tx
    return contador


def contar_coste(*, telemetria_prod: bool = False, original=None) -> tuple:
    """Envuelve `db_profiles.log_llm_usage_event` (el orquestador lo importa en cada llamada) y suma EN PROCESO el coste que
    ese emisor calcularía — misma tarifa, `compute_llm_cost_micros` — para que el presupuesto funcione sin leer la base. Con
    `telemetria_prod` además deja pasar la fila. Devuelve `(envoltura, estado)`; `estado` = {micros, llamadas, por_nodo, modelos}."""
    import db_profiles as dp
    original = original or dp.log_llm_usage_event
    estado = {"micros": 0, "llamadas": 0, "por_nodo": {}, "modelos": []}

    def _log(*, model, node=None, input_tokens=None, output_tokens=None, cached_tokens=None, **kw):
        try:
            estado["micros"] += int(dp.compute_llm_cost_micros(model, input_tokens, output_tokens, cached_tokens or 0) or 0)
        except Exception:
            pass
        estado["llamadas"] += 1
        estado["por_nodo"][node or "?"] = estado["por_nodo"].get(node or "?", 0) + 1
        if model and model not in estado["modelos"]:
            estado["modelos"].append(model)
        if telemetria_prod:
            return original(model=model, node=node, input_tokens=input_tokens, output_tokens=output_tokens,
                            cached_tokens=cached_tokens, **kw)
        return None

    dp.log_llm_usage_event = _log
    return _log, estado


def form_para_pipeline(perfil: dict, hoy: str | None = None) -> dict:
    """Lo que `/analyze` hace con el payload del wizard antes de llamar al pipeline: quita los `_` del cliente, fija
    `_plan_start_date` y `_days_to_generate = PLAN_CHUNK_SIZE` (el bloque síncrono son 3 días; el resto lo hace la cola) e
    inyecta la rebanada del blueprint (`horizon`, fail-open)."""
    from constants import PLAN_CHUNK_SIZE
    data = {k: v for k, v in dict(perfil or {}).items() if not str(k).startswith("_")}
    pd = dict(data)
    pd["_plan_start_date"] = hoy or date.today().isoformat()
    pd["_days_to_generate"] = PLAN_CHUNK_SIZE
    try:
        from horizon import inject_policy_into_pipeline_data
        total = int(data.get("totalDays", PLAN_CHUNK_SIZE) or PLAN_CHUNK_SIZE)
        inject_policy_into_pipeline_data(pd, form_data=data, total_days=total, days_offset=0, days_count=PLAN_CHUNK_SIZE,
                                         user_id=data.get("user_id"))
    except Exception:
        pass
    return pd


async def _ciclo(perfiles: list, presupuesto_usd: float, generar_uno) -> list:
    """Una generación por perfil, en serie; para en cuanto lo gastado alcanza el presupuesto. `generar_uno(i, perfil)` es una
    corrutina que devuelve `{"plan_data", "coste_usd_est", ...}`; una excepción se anota como estado, no tumba el bench."""
    out, gastado = [], 0.0
    for i, fd in enumerate(perfiles):
        if gastado >= presupuesto_usd:
            out.append({"plan_id": f"real-{i}", "estado": "no_generado:presupuesto_agotado", "plan_data": None})
            continue
        t0 = time.time()
        try:
            r = dict(await generar_uno(i, fd) or {})
            gastado += float(r.get("coste_usd_est") or 0.0)
            plan = r.pop("plan_data", None)
            ok = isinstance(plan, dict) and bool(plan.get("days"))
            estado = ("generado:emergencia" if plan.get("_p1_5_emergency_return") else "generado") if ok else "sin_dias"
            out.append({"plan_id": f"real-{i}", "plan_data": plan if ok else None, "estado": estado,
                        "s": round(time.time() - t0, 1), **r})
        except Exception as e:                                                 # noqa: BLE001
            out.append({"plan_id": f"real-{i}", "estado": f"error:{type(e).__name__}", "plan_data": None,
                        "s": round(time.time() - t0, 1), "detalle": str(e)[:200]})
        g = out[-1]
        print(f"  {g['plan_id']} · {g.get('perfil', '')} · {g['estado']} · {g.get('s', 0)} s · {g.get('dias', 0)} d / "
              f"{g.get('comidas', 0)} comidas · ${float(g.get('coste_usd_est') or 0):.4f}", flush=True)
    return out


def _abrir_pool_sync() -> None:
    try:
        from db_core import connection_pool           # fuera de FastAPI el pool nace cerrado (runbook SQL forense)
        if connection_pool:
            connection_pool.open()
    except Exception:
        pass


def generar_real(perfiles: list, presupuesto_usd: float, *, telemetria_prod: bool = False, etiquetas: list | None = None) -> list:
    """Genera planes con el LLM como lo haría `/analyze`, uno por perfil, y para en cuanto el coste alcanza el presupuesto. Sin
    `telemetria_prod` las escrituras del pipeline se suprimen y cuentan (`_ESCRITURAS`) y el coste se suma en proceso. Devuelve
    `[{"plan_id", "plan_data", "estado", "s", "coste_usd_est", "perfil", "corr", "dias", "comidas", "llamadas_llm", ...}]`."""
    import asyncio
    if not telemetria_prod:
        bloquear_escrituras()                            # ANTES del orquestador: los módulos enlazan el nombre al importar
    _abrir_pool_sync()
    from graph_orchestrator import arun_plan_pipeline
    from correlation import reset_correlation_id, set_correlation_id
    _log, coste = contar_coste(telemetria_prod=telemetria_prod)
    etiquetas = list(etiquetas or [])

    async def _uno(i, fd):
        antes, antes_n, corr = coste["micros"], coste["llamadas"], f"bench-real-{i}"
        tok = set_correlation_id(corr)
        t0 = time.time()
        try:
            plan = await arun_plan_pipeline(form_para_pipeline(fd))
        finally:
            reset_correlation_id(tok)
        plan = plan if isinstance(plan, dict) else {}
        dias = [d for d in (plan.get("days") or []) if isinstance(d, dict)]
        return {"plan_data": plan, "perfil": etiquetas[i] if i < len(etiquetas) else f"perfil-{i}", "corr": corr,
                "coste_usd_est": round((coste["micros"] - antes) / 1e6, 5),
                "coste_usd_db": (_coste_estimado(t0) if telemetria_prod else None),
                "dias": len(dias), "comidas": sum(len(d.get("meals") or []) for d in dias),
                "llamadas_llm": coste["llamadas"] - antes_n, "review_passed": plan.get("review_passed")}

    async def _todo():
        pool = None
        try:
            from db_core import async_connection_pool as pool
            if pool:
                await pool.open()                        # en ESTE loop: un pool async abierto en otro loop no sirve
        except Exception:
            pass
        try:
            return await _ciclo(perfiles, presupuesto_usd, _uno)
        finally:
            try:
                if pool:
                    await pool.close()
            except Exception:
                pass

    if sys.platform == "win32":                          # psycopg async no corre sobre el Proactor de Windows
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass
    return asyncio.run(_todo())


def _coste_estimado(desde_ts: float) -> float:
    """Con `--telemetria-prod`: lo que `llm_usage_events` registró desde `desde_ts` (contraste del contador en proceso)."""
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
    ap.add_argument("--planes-de", help="[P1-PLAN-LOTE-30] re-mide (sin LLM) los `planes_generados` de un artefacto real: pareado real, dos versiones")
    ap.add_argument("--superficies", default=",".join(SUPERFICIES))
    ap.add_argument("--out", help="artefacto JSON (por defecto scripts/data/bench_superficies_<fecha>.json)")
    ap.add_argument("--sin-guardar", action="store_true")
    ap.add_argument("--informe", help="reconstruye el informe desde un artefacto y sale")
    ap.add_argument("--comparar", nargs=2, metavar=("A", "B"), help="informe pareado entre dos artefactos")
    ap.add_argument("--exportar-catalogo", action="store_true", help="(base, sólo lectura) snapshot del catálogo con nutrición")
    ap.add_argument("--real", action="store_true", help="genera planes con el LLM (exige --perfil y --presupuesto-usd)")
    ap.add_argument("--perfil", action="append", help="JSON de form_data para --real (repetible)")
    ap.add_argument("--presupuesto-usd", type=float, help="tope de gasto estimado para --real")
    ap.add_argument("--telemetria-prod", action="store_true", help="--real: dejar que la telemetría del pipeline se escriba en la base (por defecto se suprime y se cuenta)")
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
        etiquetas = [Path(x).stem for x in a.perfil]
        print(f"modo real · {len(perfiles)} perfil(es) · presupuesto ${a.presupuesto_usd:.2f} · telemetría a producción: "
              f"{'sí' if a.telemetria_prod else 'NO (suprimida y contada)'}", flush=True)
        generados = generar_real(perfiles, float(a.presupuesto_usd), telemetria_prod=a.telemetria_prod, etiquetas=etiquetas)
        planes = [g for g in generados if g.get("plan_data")]
        art = correr(planes, filas, superficies, modo="real")
        art["generaciones"] = [{k: v for k, v in g.items() if k != "plan_data"} for g in generados]
        art["planes_generados"] = [{"plan_id": g["plan_id"], "perfil": g.get("perfil"), "plan_data": g["plan_data"]} for g in planes]
        art["presupuesto_usd"] = a.presupuesto_usd
        art["coste_usd_est"] = round(sum(float(g.get("coste_usd_est") or 0) for g in generados), 5)
        art["telemetria_prod"] = bool(a.telemetria_prod)
        art["escrituras_suprimidas"] = dict(sorted(_ESCRITURAS.items()))
    elif a.planes_de:
        fuente = json.loads(Path(a.planes_de).read_text(encoding="utf-8"))
        planes = [{"plan_id": p.get("plan_id"), "plan_data": p.get("plan_data")} for p in (fuente.get("planes_generados") or []) if p.get("plan_data")]
        if not planes:
            print("el artefacto no trae `planes_generados`: nada que re-medir")
            return 2
        art = correr(planes, filas, superficies, modo="real-replay")
        art["planes_de"], art["generaciones"] = os.path.basename(a.planes_de), fuente.get("generaciones")
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
        sufijo = "real_" if a.real else ("replay_" if a.planes_de else "")
        p = Path(a.out) if a.out else _BACKEND / "scripts" / "data" / f"bench_superficies_{sufijo}{date.today():%Y_%m_%d}.json"
        p.write_text(json.dumps(art, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
        print("artefacto:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
