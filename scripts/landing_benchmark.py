# [P2-LOGGER-EXEMPT: CLI de benchmark — salida humana a stdout por diseño]
"""[P1-LANDING-BENCH-1 · 2026-08-07] Runner del benchmark del landing.

Cinco modos (componibles con el mismo schema de salida — ver
`landing_benchmarks.LANDING_REPORT_SECTIONS` y docs/landing_benchmarks.md):

  structural  Hechos contables (reglas clínicas, micros DRI, catálogo) — sin LLM;
              DB opcional (best-effort para conteos de catálogo).
  live        Genera N planes REALES con la matriz fiel-al-formulario y los puntúa
              (seguridad clínica + gym 7-ejes + latencia). Requiere claves LLM
              y URLs de Neon. --changes ejercita además swap individual y el bucle
              de día (las superficies de /swap-meal y /regenerate-day).
              --provider openai fuerza la corrida COMPLETA a la familia gpt-5.6
              (cero GLM) vía los knobs per-feature sancionados — requiere
              OPENAI_API_KEY.
  remote      La corrida "cuenta de invitado": genera contra un API DESPLEGADO
              (--api-base) como user_id=guest — CERO claves locales; el routing
              de modelos lo decide el servidor (guest ⇒ day-gen/swap/reviewer en
              Luna = OpenAI desde P1-DAYGEN-TIER-MODEL / P1-SWAP-LUNA /
              P1-REVIEWER-TIER-MODELS). Puntúa localmente igual que live.
              --changes ejercita swap (regenerate-day requiere plan persistido
              con auth → fuera del alcance guest, documentado).
  telemetry   Agrega las series de PRODUCCIÓN (pipeline_metrics: change_swap /
              change_regen_day / clinical_band_final; meal_plans._quality_index;
              llm_usage_events) — solo DB, sin LLM.
  score       Re-puntúa planes crudos guardados por una corrida `live/remote
              --save-plans` (mide un cambio de scorer sin pagar LLM).

Uso (desde backend/, con .env cargable):
    python scripts/landing_benchmark.py structural
    python scripts/landing_benchmark.py live 5 --conc 2 --changes --save-plans
    python scripts/landing_benchmark.py live 20 --provider openai --conc 2
    python scripts/landing_benchmark.py remote --api-base https://app.bioboros.com --changes
    python scripts/landing_benchmark.py telemetry --days 30
    python scripts/landing_benchmark.py score --plans landing_plans_1234.json

Salida: resumen humano a stdout + JSON completo a --out (default
landing_benchmark_<modo>_<pid>.json en cwd; override env LANDING_BENCH_OUT).
El JSON se escribe ANTES del resumen (lección del gym 2026-07-02: un print que
crashea no puede perder una corrida que costó minutos de LLM).
tooltip-anchor: P1-LANDING-BENCH-1-RUNNER
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except Exception:
    pass

from landing_benchmarks import (
    aggregate_safety,
    build_landing_profiles,
    build_report,
    score_plan_safety,
    strip_benchmark_meta,
    structural_facts,
)


# [P1-LANDING-BENCH-1-OPENAI] Forzado "todo OpenAI, cero GLM" para el modo live.
# SOLO knobs per-feature sancionados (P3-PREVIEW-MODEL-KNOB) — el override global de
# modelo fue ELIMINADO adrede (P1-SINGLE-PROVIDER-RESTORE: colapsaba también el reviewer
# clínico risk-tier a un provider de test) y NO se reintroduce aquí. Estos 4 knobs
# mueven el pipeline (flash nodes + router por tier + red post-fallo); el reviewer
# (Luna/Terra/Sol), day-gen (Luna por tier) y swap (Luna fijo) YA son OpenAI por
# default y conservan su routing fail-secure propio.
_OPENAI_FORCE_KNOBS = {
    "MEALFIT_FLASH_MODEL": "gpt-5.6-luna",
    "MEALFIT_MODEL_FREE_TIER": "gpt-5.6-luna",
    "MEALFIT_MODEL_PAID_TIER": "gpt-5.6-luna",
    "MEALFIT_PRO_MODEL": "gpt-5.6-luna",
}


def _force_openai_provider():
    """Aplica el forzado ANTES de los imports lazy de graph_orchestrator (la red
    post-fallo se resuelve al boot del módulo). Fail-loud sin OPENAI_API_KEY: sin
    key, graph_orchestrator degradaría la red a GLM en silencio y la corrida
    dejaría de ser lo que dice ser."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "--provider openai requiere OPENAI_API_KEY en el entorno. Sin ella la "
            "red post-fallo cae a GLM (fail-safe P1-NET-LUNA) y la corrida no "
            "sería 'todo OpenAI'. Exporta la key y reintenta."
        )
    for k, v in _OPENAI_FORCE_KNOBS.items():
        os.environ[k] = v
    print("provider=openai — knobs forzados (solo este proceso):")
    for k, v in _OPENAI_FORCE_KNOBS.items():
        print(f"  {k}={v}")
    print("  (reviewer/day-gen/swap ya rutean a gpt-5.6 por defecto de tier)")


def _open_pools():
    """Los pools nacen con open=False (se abren en el lifespan de la app); un script
    standalone debe abrirlos o los ejes de costo/lista degradan a PoolClosed."""
    try:
        from db_core import connection_pool
        if connection_pool is not None:
            connection_pool.open()
    except Exception as _pe:
        print(f"(aviso) pool sync no disponible: {_pe} — conteos DB degradarán a None")


def _fetch_scalar(query, params=()):
    try:
        from db_core import execute_sql_query
        rows = execute_sql_query(query, params, fetch_all=True)
        if rows:
            return list(rows[0].values())[0]
    except Exception:
        return None
    return None


def _fetch_rows(query, params=()):
    try:
        from db_core import execute_sql_query
        return {"rows": execute_sql_query(query, params, fetch_all=True)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _structural_section():
    facts = structural_facts()
    # Conteos que solo la DB conoce — best-effort, None si no hay conexión.
    facts["alimentos_catalogo"] = _fetch_scalar("SELECT COUNT(*) FROM master_ingredients")
    facts["productos_supermercado"] = _fetch_scalar("SELECT COUNT(*) FROM supermarket_products")
    return facts


# ─────────────────────────────── live ───────────────────────────────

def _swap_payload(profile, meal, meal_type):
    return {
        "rejected_meal": (meal or {}).get("name", ""),
        "meal_type": meal_type or "Almuerzo",
        "target_calories": (meal or {}).get("calories") or 0,
        "target_protein": (meal or {}).get("protein") or 0,
        "target_carbs": (meal or {}).get("carbs") or 0,
        "target_fats": (meal or {}).get("fats") or 0,
        "diet_type": profile.get("dietType") or "balanced",
        "goal": profile.get("mainGoal") or "maintenance",
        "allergies": profile.get("allergies") or [],
        "medicalConditions": profile.get("medicalConditions") or [],
        "medications": profile.get("medications") or [],
        "swap_reason": "variety",
        "user_id": "guest",
    }


def _validate_swapped(meal, profile):
    from graph_orchestrator import clinical_backstop_for_meal
    allergies = [a for a in (profile.get("allergies") or []) if a and a != "Ninguna"]
    return clinical_backstop_for_meal(
        meal, allergies=allergies, diet_type=profile.get("dietType") or "balanced",
        form_data=profile)


def _exercise_changes(plan, profile):
    """Ejercita las superficies de cambio con el MISMO módulo de producción (`agent.swap_meal`):
    un swap individual (surface="individual") y el bucle EN SERIE de un día (surface="day",
    espejo de /regenerate-day). Reporta éxito, latencia y re-validación clínica del plato nuevo."""
    from agent import swap_meal
    out = {"swap": None, "regen_day": None}
    days = (plan or {}).get("days") or []
    if not days or not (days[0] or {}).get("meals"):
        return out

    meals0 = days[0]["meals"]
    target = next((m for m in meals0 if "almuerzo" in str(m.get("meal_type", m.get("type", ""))).lower()),
                  meals0[min(1, len(meals0) - 1)])
    mtype = str(target.get("meal_type") or target.get("type") or "Almuerzo")

    t0 = time.time()
    try:
        res = swap_meal(_swap_payload(profile, target, mtype), surface="individual")
        ok = isinstance(res, dict) and not res.get("swap_failed")
        out["swap"] = {
            "ok": ok,
            "duration_s": round(time.time() - t0, 1),
            "band_low": bool(isinstance(res, dict) and res.get("_macro_band_low")),
            "violaciones_post": _validate_swapped(res, profile) if ok else None,
        }
    except Exception as e:
        out["swap"] = {"ok": False, "duration_s": round(time.time() - t0, 1),
                       "error": f"{type(e).__name__}: {e}"}

    # Día completo: bucle serial (así es /regenerate-day en prod: 4-5 llamadas EN SERIE).
    per_meal, ok_count = [], 0
    t_day = time.time()
    for m in meals0[:5]:
        mt = str(m.get("meal_type") or m.get("type") or "Comida")
        t1 = time.time()
        try:
            r = swap_meal(_swap_payload(profile, m, mt), surface="day")
            good = isinstance(r, dict) and not r.get("swap_failed")
            ok_count += 1 if good else 0
            per_meal.append({"meal_type": mt, "ok": good, "duration_s": round(time.time() - t1, 1),
                             "violaciones_post": _validate_swapped(r, profile) if good else None})
        except Exception as e:
            per_meal.append({"meal_type": mt, "ok": False, "duration_s": round(time.time() - t1, 1),
                             "error": f"{type(e).__name__}: {e}"})
    out["regen_day"] = {
        "meals": len(per_meal), "ok": ok_count,
        "duration_s": round(time.time() - t_day, 1), "per_meal": per_meal,
    }
    return out


async def _run_one_live(profile, sem, do_changes):
    from graph_orchestrator import arun_plan_pipeline
    from plan_gym import score_plan
    async with sem:
        fd = strip_benchmark_meta(profile)
        t0 = time.time()
        try:
            plan = await arun_plan_pipeline(dict(fd))
        except Exception as e:
            return {"id": profile["_id"], "label": profile["_label"],
                    "error": f"{type(e).__name__}: {e}"}
        dur = round(time.time() - t0, 1)
        row = {"id": profile["_id"], "label": profile["_label"],
               "goal": profile.get("mainGoal"), "conditions": profile.get("medicalConditions"),
               "medications": profile.get("medications"), "diet": profile.get("dietType"),
               "duration_s": dur, "_plan": plan}
        try:
            row["safety"] = score_plan_safety(plan, profile)
        except Exception as e:
            row["safety_error"] = f"{type(e).__name__}: {e}"
        try:
            row["gym"] = score_plan(plan, fd)
        except Exception as e:
            row["gym_error"] = f"{type(e).__name__}: {e}"
        if do_changes:
            try:
                row["changes"] = await asyncio.to_thread(_exercise_changes, plan, profile)
            except Exception as e:
                row["changes_error"] = f"{type(e).__name__}: {e}"
        return row


def _percentiles(values, ps=(0.5, 0.95)):
    vals = sorted(v for v in values if isinstance(v, (int, float)))
    if not vals:
        return {}
    out = {}
    for p in ps:
        idx = min(len(vals) - 1, max(0, round(p * (len(vals) - 1))))
        out[f"p{int(p * 100)}"] = vals[idx]
    return out


async def _live_sections(n, conc, do_changes, save_plans_path):
    _open_pools()
    try:
        from db_core import async_connection_pool
        if async_connection_pool is not None:
            await async_connection_pool.open()
    except Exception:
        pass
    from plan_gym import aggregate_scores

    profiles = build_landing_profiles()
    if n:
        profiles = profiles[:n]
    sem = asyncio.Semaphore(conc)
    rows = await asyncio.gather(*[_run_one_live(p, sem, do_changes) for p in profiles])
    rows = list(rows)

    if save_plans_path:
        with open(save_plans_path, "w", encoding="utf-8") as f:
            json.dump({"plans": [{"id": r["id"], "label": r.get("label"), "plan": r.get("_plan")}
                                 for r in rows if r.get("_plan")]}, f,
                      ensure_ascii=False, default=str)

    for r in rows:
        r.pop("_plan", None)  # el plan crudo no viaja en el reporte (pesa cientos de KB)

    gym_rows = [{"id": r["id"], "score": r["gym"]} for r in rows if r.get("gym")]
    changes = None
    if do_changes:
        swaps = [r["changes"]["swap"] for r in rows if (r.get("changes") or {}).get("swap")]
        days = [r["changes"]["regen_day"] for r in rows if (r.get("changes") or {}).get("regen_day")]
        changes = {
            "swap": {
                "n": len(swaps),
                "ok_pct": round(100.0 * sum(1 for s in swaps if s.get("ok")) / len(swaps), 1) if swaps else None,
                "latency_s": _percentiles([s.get("duration_s") for s in swaps]),
                "con_violaciones_post": sum(1 for s in swaps if s.get("violaciones_post")),
            },
            "regen_day": {
                "n": len(days),
                "meals_ok": sum(d.get("ok", 0) for d in days),
                "meals_total": sum(d.get("meals", 0) for d in days),
                "latency_day_s": _percentiles([d.get("duration_s") for d in days]),
            },
            "per_profile": [{"id": r["id"], **r["changes"]} for r in rows if r.get("changes")],
        }

    return {
        "structural": _structural_section(),
        "safety": {
            "aggregate": aggregate_safety([r.get("safety") for r in rows if r.get("safety")]),
            "per_profile": [r.get("safety") or {"id": r["id"], "error": r.get("error") or r.get("safety_error")}
                            for r in rows],
        },
        "gym": {"aggregate": aggregate_scores(gym_rows), "per_profile": rows},
        "latency": {"generation_s": _percentiles([r.get("duration_s") for r in rows])},
        "changes": changes,
    }


# ─────────────────────────────── remote (cuenta de invitado) ───────────────────────────────

def _remote_post(api_base, path, payload, timeout_s, max_429_retries=4):
    """POST con backoff ante 429 (el /analyze tiene RateLimiter 3/60s per user|ip;
    /swap-meal 20/60s). httpx respeta HTTPS_PROXY/SSL_CERT_FILE del entorno."""
    import httpx
    url = f"{api_base.rstrip('/')}{path}"
    for attempt in range(max_429_retries + 1):
        r = httpx.post(url, json=payload, timeout=timeout_s)
        if r.status_code == 429:
            wait = 30 * (attempt + 1)
            print(f"  429 en {path} — backoff {wait}s (intento {attempt + 1}/{max_429_retries})")
            time.sleep(wait)
            continue
        if r.status_code >= 400:
            # El body lleva el `detail` de FastAPI — sin él un 4xx/5xx es indiagnosticable
            # desde fuera (lección smoke 2026-08-07: tres 500 mudos). El header
            # X-Bioboros-Review-Diag (P1-LANDING-BENCH-2) trae las razones del rechazo
            # crítico que el detail-string no puede llevar.
            _diag = r.headers.get("x-bioboros-review-diag")
            _diag_sfx = f" | diag: {_diag[:1200]}" if _diag else ""
            raise RuntimeError(f"HTTP {r.status_code} en {path}: {r.text[:400]}{_diag_sfx}")
        return r.json()
    raise RuntimeError(f"rate-limit persistente en {path} tras {max_429_retries} reintentos")


def _remote_generate_stream(api_base, payload, timeout_s, max_429_retries=4):
    """[P1-LANDING-BENCH-3 · 2026-08-07] Genera vía /analyze/stream (SSE) — el MISMO transporte
    del frontend. El endpoint síncrono corta generaciones largas (proxy_read_timeout de nginx:
    5/8 disconnects en la verificación 2026-08-07, agravado por el retry informado de
    P1-DAYGEN-DIET-CONVERGE); el stream emite heartbeats → el timeout que importa es ENTRE
    eventos (read=300s), no el total. El diagnóstico de un rechazo crítico viaja DENTRO del
    evento error (`review_issues`, paridad con el header del síncrono)."""
    import httpx
    import json as _j
    url = f"{api_base.rstrip('/')}/api/plans/analyze/stream"
    timeout = httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=60.0)
    for attempt in range(max_429_retries + 1):
        with httpx.stream("POST", url, json=payload, timeout=timeout) as r:
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  429 en /analyze/stream — backoff {wait}s "
                      f"(intento {attempt + 1}/{max_429_retries})")
                time.sleep(wait)
                continue
            ctype = r.headers.get("content-type", "")
            if r.status_code != 200 or "text/event-stream" not in ctype:
                r.read()
                raise RuntimeError(
                    f"stream no disponible (HTTP {r.status_code}, {ctype[:40]}): {r.text[:200]}")
            deadline = time.time() + timeout_s
            for line in r.iter_lines():
                if time.time() > deadline:
                    raise RuntimeError(f"stream excedió el presupuesto total de {timeout_s}s")
                if not line or not line.startswith("data: "):
                    continue
                try:
                    evt = _j.loads(line[6:])
                except Exception:
                    continue
                kind = evt.get("event")
                if kind == "complete":
                    return evt.get("data")
                if kind == "error":
                    d = evt.get("data") or {}
                    _diag_sfx = ""
                    if d.get("review_issues"):
                        _diag_sfx = " | diag: " + _j.dumps(
                            {"fallback_reason": d.get("fallback_reason"),
                             "review_issues": d.get("review_issues")}, ensure_ascii=True)[:1200]
                    raise RuntimeError(
                        f"SSE error code={d.get('code')}: {str(d.get('message'))[:200]}{_diag_sfx}")
            raise RuntimeError("stream terminó sin evento 'complete' ni 'error'")
    raise RuntimeError(f"rate-limit persistente en /analyze/stream tras {max_429_retries} reintentos")


def _run_one_remote(api_base, profile, do_changes, timeout_s, transport="sse"):
    import uuid
    from plan_gym import score_plan
    fd = strip_benchmark_meta(profile)
    payload = {
        **fd,
        # Mismo shape que Plan.jsx::dataToSend para un GUEST REAL: `user_id: null`
        # (NO el literal "guest" — eso es convención del harness in-process; contra
        # el API revienta un cast ::uuid server-side → 500, smoke 2026-08-07) y
        # `session_id: crypto.randomUUID()`. totalDays por groceryDuration
        # (weekly=7), tzOffset RD (UTC-4 → 240 min), y las claves acompañantes que
        # el cliente SIEMPRE envía aunque vacías.
        "user_id": None,
        "session_id": str(uuid.uuid4()),
        "totalDays": 7,
        "tzOffset": 240,
        "previous_meals": [],
        "current_pantry_ingredients": [],
        "durable_pantry_ingredients": [],
        "update_reason": None,
        "renewal_pantry_aware": False,
        "is_plan_expired": False,
    }
    t0 = time.time()
    try:
        if transport == "sse":
            try:
                plan = _remote_generate_stream(api_base, payload, timeout_s)
            except RuntimeError as _sse_e:
                if "stream no disponible" not in str(_sse_e):
                    raise
                # Deploy sin SSE utilizable → mismo fallback que el frontend: endpoint síncrono.
                print(f"  (aviso) SSE no disponible, cayendo al síncrono: {str(_sse_e)[:120]}")
                plan = _remote_post(api_base, "/api/plans/analyze", payload, timeout_s)
        else:
            plan = _remote_post(api_base, "/api/plans/analyze", payload, timeout_s)
    except Exception as e:
        return {"id": profile["_id"], "label": profile["_label"],
                "error": f"{type(e).__name__}: {e}"}
    dur = round(time.time() - t0, 1)
    row = {"id": profile["_id"], "label": profile["_label"],
           "goal": profile.get("mainGoal"), "conditions": profile.get("medicalConditions"),
           "medications": profile.get("medications"), "diet": profile.get("dietType"),
           "duration_s": dur, "_plan": plan}
    try:
        row["safety"] = score_plan_safety(plan, profile)
    except Exception as e:
        row["safety_error"] = f"{type(e).__name__}: {e}"
    try:
        row["gym"] = score_plan(plan, fd)
    except Exception as e:
        row["gym_error"] = f"{type(e).__name__}: {e}"

    if do_changes:
        days = (plan or {}).get("days") or []
        meals0 = (days[0] or {}).get("meals") if days else None
        if meals0:
            target = meals0[min(1, len(meals0) - 1)]
            mtype = str(target.get("meal_type") or target.get("type") or "Almuerzo")
            sp = _swap_payload(profile, target, mtype)
            sp["user_id"] = None  # guest real (ver nota del payload de /analyze)
            sp["session_id"] = payload["session_id"]
            t1 = time.time()
            try:
                res = _remote_post(api_base, "/api/plans/swap-meal", sp, timeout_s=120)
                ok = isinstance(res, dict) and not res.get("swap_failed")
                row["changes"] = {"swap": {
                    "ok": ok, "duration_s": round(time.time() - t1, 1),
                    "band_low": bool(isinstance(res, dict) and res.get("_macro_band_low")),
                    "violaciones_post": _validate_swapped(res, profile) if ok else None,
                }, "regen_day": None}  # requiere plan persistido + auth → no-guest
            except Exception as e:
                row["changes"] = {"swap": {"ok": False, "duration_s": round(time.time() - t1, 1),
                                           "error": f"{type(e).__name__}: {e}"},
                                  "regen_day": None}
    return row


def _remote_sections(api_base, n, conc, do_changes, save_plans_path, timeout_s, ids=None,
                     transport="sse"):
    from concurrent.futures import ThreadPoolExecutor
    from plan_gym import aggregate_scores

    profiles = build_landing_profiles()
    if ids:
        profiles = [p for p in profiles if p["_id"] in ids]
    elif n:
        profiles = profiles[:n]
    # conc default 1: el /analyze de un guest comparte RateLimiter por IP (3/60s);
    # con generaciones de minutos, 1-2 en vuelo no lo rozan pero >2 sí al arrancar.
    with ThreadPoolExecutor(max_workers=max(1, conc)) as ex:
        rows = list(ex.map(
            lambda p: _run_one_remote(api_base, p, do_changes, timeout_s, transport=transport),
            profiles))

    if save_plans_path:
        with open(save_plans_path, "w", encoding="utf-8") as f:
            json.dump({"plans": [{"id": r["id"], "label": r.get("label"), "plan": r.get("_plan")}
                                 for r in rows if r.get("_plan")]}, f,
                      ensure_ascii=False, default=str)
    for r in rows:
        r.pop("_plan", None)

    gym_rows = [{"id": r["id"], "score": r["gym"]} for r in rows if r.get("gym")]
    changes = None
    if do_changes:
        swaps = [r["changes"]["swap"] for r in rows if (r.get("changes") or {}).get("swap")]
        changes = {
            "swap": {
                "n": len(swaps),
                "ok_pct": round(100.0 * sum(1 for s in swaps if s.get("ok")) / len(swaps), 1) if swaps else None,
                "latency_s": _percentiles([s.get("duration_s") for s in swaps]),
                "con_violaciones_post": sum(1 for s in swaps if s.get("violaciones_post")),
            },
            "regen_day": {"skipped": "requiere plan persistido con auth — fuera del alcance guest"},
            "per_profile": [{"id": r["id"], **r["changes"]} for r in rows if r.get("changes")],
        }
    return {
        "meta": {
            "api_base": api_base, "guest": True,
            "routing": ("modelos decididos por el SERVIDOR: guest ⇒ day-gen/swap/reviewer "
                        "en gpt-5.6 (Luna) por P1-DAYGEN-TIER-MODEL/P1-SWAP-LUNA/"
                        "P1-REVIEWER-TIER-MODELS; nodos auxiliares según knobs del deploy"),
        },
        "safety": {
            "aggregate": aggregate_safety([r.get("safety") for r in rows if r.get("safety")]),
            "per_profile": [r.get("safety") or {"id": r["id"], "error": r.get("error") or r.get("safety_error")}
                            for r in rows],
        },
        "gym": {"aggregate": aggregate_scores(gym_rows), "per_profile": rows},
        "latency": {"generation_s": _percentiles([r.get("duration_s") for r in rows])},
        "changes": changes,
    }


# ─────────────────────────────── telemetry ───────────────────────────────

def _telemetry_section(days):
    _open_pools()
    d = int(days)
    return {
        "window_days": d,
        # [P1-CHANGE-OUTCOME-TELEMETRY] la pregunta que el landing quiere responder:
        # ¿los cambios de plato salen a la primera?
        "changes": _fetch_rows(
            """SELECT node, metadata->>'outcome' AS outcome, COUNT(*) AS n,
                      ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms)) AS p50_ms,
                      ROUND(percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms)) AS p95_ms
               FROM pipeline_metrics
               WHERE node IN ('change_swap','change_regen_day')
                 AND created_at >= NOW() - make_interval(days => %s)
               GROUP BY 1, 2 ORDER BY 1, 2""", (d,)),
        "banda_entregada": _fetch_rows(
            """SELECT COUNT(*) AS n, ROUND(AVG(confidence)::numeric, 3) AS media,
                      ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY confidence)::numeric, 3) AS p50
               FROM pipeline_metrics
               WHERE node = 'clinical_band_final'
                 AND created_at >= NOW() - make_interval(days => %s)""", (d,)),
        "fallback_rate": _fetch_rows(
            """SELECT COUNT(*) AS n,
                      ROUND((COUNT(*) FILTER (WHERE metadata->>'delivered_was_fallback' = 'true'))::numeric
                            / NULLIF(COUNT(*), 0), 3) AS rate
               FROM pipeline_metrics
               WHERE node = 'clinical_band'
                 AND created_at >= NOW() - make_interval(days => %s)""", (d,)),
        "generacion_latencia": _fetch_rows(
            """SELECT COUNT(*) AS n,
                      ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms) / 1000.0) AS p50_s,
                      ROUND(percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) / 1000.0) AS p95_s
               FROM pipeline_metrics
               WHERE node = 'clinical_band'
                 AND created_at >= NOW() - make_interval(days => %s)""", (d,)),
        "quality_index": _fetch_rows(
            """SELECT COUNT(*) AS n,
                      ROUND(AVG((plan_data->'_quality_index'->>'score')::float)::numeric, 1) AS media
               FROM meal_plans
               WHERE plan_data ? '_quality_index'
                 AND created_at >= NOW() - make_interval(days => %s)""", (d,)),
        "costo_por_nodo": _fetch_rows(
            """SELECT node, model, COUNT(*) AS calls,
                      ROUND((SUM(cost_usd_micros) / 1e6)::numeric, 4) AS usd
               FROM llm_usage_events
               WHERE created_at >= NOW() - make_interval(days => %s)
               GROUP BY 1, 2 ORDER BY usd DESC NULLS LAST LIMIT 12""", (d,)),
    }


# ─────────────────────────────── score (replay) ───────────────────────────────

def _score_sections(plans_path):
    with open(plans_path, encoding="utf-8") as f:
        data = json.load(f)
    profiles = {p["_id"]: p for p in build_landing_profiles()}
    results = []
    for item in data.get("plans", []):
        prof = profiles.get(item.get("id"))
        if not prof or not item.get("plan"):
            continue
        results.append(score_plan_safety(item["plan"], prof))
    return {"safety": {"aggregate": aggregate_safety(results), "per_profile": results}}


# ─────────────────────────────── main ───────────────────────────────

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("structural", "live", "remote", "telemetry", "score"))
    ap.add_argument("n", nargs="?", type=int, default=0, help="live/remote: límite de perfiles (0 = todos)")
    ap.add_argument("--conc", type=int, default=None,
                    help="concurrencia (default: live=2, remote=1 por el rate-limit per-IP)")
    ap.add_argument("--changes", action="store_true", help="live/remote: ejercitar cambios")
    ap.add_argument("--save-plans", action="store_true",
                    help="live/remote: guardar planes crudos para `score`")
    ap.add_argument("--provider", choices=("default", "openai"), default="default",
                    help="live: openai fuerza toda la corrida a gpt-5.6 (cero GLM)")
    ap.add_argument("--api-base", help="remote: URL base del API desplegado (p.ej. https://app.bioboros.com)")
    ap.add_argument("--ids", help="remote: perfiles específicos por id, p.ej. '3,9,10,13,15' "
                                  "(los clínicos; gana sobre el N posicional)")
    ap.add_argument("--transport", choices=("sse", "sync"), default="sse",
                    help="remote: sse (default; /analyze/stream con heartbeats, inmune al "
                         "proxy_read_timeout) o sync (endpoint de un solo response)")
    ap.add_argument("--timeout", type=int, default=1200, help="remote: timeout por plan en segundos")
    ap.add_argument("--days", type=int, default=30, help="telemetry: ventana en días")
    ap.add_argument("--plans", help="score: JSON de una corrida live/remote --save-plans")
    ap.add_argument("--out", help="ruta del JSON de salida")
    args = ap.parse_args()

    if args.mode == "structural":
        _open_pools()
        sections = {"structural": _structural_section()}
    elif args.mode == "live":
        if args.provider == "openai":
            _force_openai_provider()
        save_path = f"landing_plans_{os.getpid()}.json" if args.save_plans else None
        sections = asyncio.run(_live_sections(args.n, max(1, args.conc or 2), args.changes, save_path))
        if save_path:
            print(f"planes crudos: {save_path}")
    elif args.mode == "remote":
        if not args.api_base:
            ap.error("--api-base es obligatorio en modo remote")
        save_path = f"landing_plans_{os.getpid()}.json" if args.save_plans else None
        _ids = {int(x) for x in args.ids.split(",") if x.strip()} if args.ids else None
        sections = _remote_sections(args.api_base, args.n, max(1, args.conc or 1),
                                    args.changes, save_path, args.timeout, ids=_ids,
                                    transport=args.transport)
        if save_path:
            print(f"planes crudos: {save_path}")
    elif args.mode == "telemetry":
        sections = {"telemetry": _telemetry_section(args.days)}
    else:
        if not args.plans:
            ap.error("--plans es obligatorio en modo score")
        sections = _score_sections(args.plans)

    report = build_report(args.mode, **sections)
    out_path = args.out or os.environ.get("LANDING_BENCH_OUT") \
        or f"landing_benchmark_{args.mode}_{os.getpid()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)

    print("\n========== LANDING BENCHMARK - resumen ==========")
    print(f"modo: {args.mode} | schema v{report['schema_version']}")
    if "meta" in report:
        print(f"  remote: {report['meta'].get('api_base')} (guest) — {report['meta'].get('routing')}")
    if "structural" in report:
        s = report["structural"]
        print(f"  micros DRI: {s['micronutrientes_dri']} | reglas condición: "
              f"{s['reglas_condicion_backend']} (chips form: {s['condiciones_chips_formulario']}) "
              f"| reglas medicación: {s['reglas_medicacion_backend']} "
              f"(chips form: {s['medicamentos_chips_formulario']})")
        print(f"  solo-backend (el formulario YA NO puede expresarlas): "
              f"condiciones={s['condiciones_solo_backend']} medicaciones={s['medicaciones_solo_backend']}")
        print(f"  catálogo: {s['alimentos_catalogo']} alimentos | "
              f"{s['productos_supermercado']} productos supermercado")
    if "safety" in report:
        agg = report["safety"]["aggregate"]
        print(f"  seguridad: n={agg.get('n')} planes sin violaciones="
              f"{agg.get('plans_sin_violaciones_pct')}% violaciones={agg.get('violaciones_totales')} "
              f"por categoría={agg.get('violaciones_por_categoria')}")
        print(f"  min-comidas (insulina/bariátrica): {agg.get('min_meals_compliance_pct')}% | "
              f"FS9 presente: {agg.get('fs9_flag_presente_pct')}%")
    if "gym" in report and report["gym"].get("aggregate"):
        g = report["gym"]["aggregate"]
        print(f"  gym: n={g.get('n')} global={g.get('global_mean')}")
    if "latency" in report:
        print(f"  latencia generación: {report['latency'].get('generation_s')}")
    if report.get("changes"):
        c = report["changes"]
        rd = c.get("regen_day") or {}
        dia = rd.get("skipped") or (f"{rd.get('meals_ok')}/{rd.get('meals_total')} "
                                    f"lat={rd.get('latency_day_s')}")
        print(f"  swap: ok={c['swap'].get('ok_pct')}% lat={c['swap'].get('latency_s')} | día: {dia}")
    if "telemetry" in report:
        t = report["telemetry"]
        print(f"  telemetría ({t['window_days']}d): cambios={t['changes']} | "
              f"banda={t['banda_entregada']} | fallback={t['fallback_rate']} | PQI={t['quality_index']}")
    print(f"\nJSON completo: {out_path}")


if __name__ == "__main__":
    main()
