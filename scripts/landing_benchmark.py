# [P2-LOGGER-EXEMPT: CLI de benchmark — salida humana a stdout por diseño]
"""[P1-LANDING-BENCH-1 · 2026-08-07] Runner del benchmark del landing.

Cinco modos (componibles con el mismo schema de salida — ver
`landing_benchmarks.LANDING_REPORT_SECTIONS` y docs/landing_benchmarks.md):

  structural  Hechos contables (reglas clínicas, micros DRI, catálogo) — sin LLM;
              DB opcional (best-effort para conteos de catálogo).
  live        Genera los planes REALES de la matriz fiel-al-formulario (por defecto
              LA MATRIZ ENTERA: 25 perfiles) y los puntúa (seguridad clínica +
              nutrición + gym 7-ejes + latencia + entrega). Requiere claves LLM y
              URLs de Neon. --changes ejercita además swap individual y el bucle
              de día (las superficies de /swap-meal y /regenerate-day).
              --provider openai fuerza la corrida COMPLETA a la familia OpenAI
              (`_OPENAI_FORCE_KNOBS`: GPT-6 Luna) vía los knobs per-feature
              sancionados — requiere OPENAI_API_KEY.
  remote      La corrida "cuenta de invitado": genera contra un API DESPLEGADO
              (--api-base) como user_id=guest — CERO claves locales; el routing
              de modelos lo decide el SERVIDOR con los knobs de SU deploy
              (docs/llm_tier_routing.md) — el reporte no afirma un modelo.
              Puntúa localmente igual que live. --changes ejercita swap
              (regenerate-day requiere plan persistido con auth → fuera del
              alcance guest, documentado).
  telemetry   Agrega las series de PRODUCCIÓN, solo lo ENTREGADO (pipeline_metrics
              clinical_band_final de las superficies de entrega; latencia de las
              corridas sin fallback; fallback de los planes persistidos) — solo DB.
  score       Re-puntúa planes crudos guardados SIN pagar LLM: los de una corrida
              `live/remote --save-plans` (--plans) o un corpus de planes reales
              (--plans-glob + --forms-root, formato de las baterías `rdNN/`).

[P1-PLAN-LOTE-749 · 2026-09-28] Reporte schema v2: bloque `run` (commit de origen, sucio o
no, protocolo, cohorte) + secciones `nutrition` (MAPE por macro, peor macro, días 4-en-banda
con la banda del motor) y `reliability` (entrega y latencia con los fallos en el
denominador) — el formato que importa el landing. tooltip-anchor: P1-PLAN-LOTE-749

Uso (desde backend/, con .env cargable):
    python scripts/landing_benchmark.py structural
    python scripts/landing_benchmark.py live --conc 2 --changes --save-plans
    python scripts/landing_benchmark.py live --provider openai --conc 2
    python scripts/landing_benchmark.py remote --api-base https://app.bioboros.com --conc 2 --changes
    python scripts/landing_benchmark.py telemetry --days 30
    python scripts/landing_benchmark.py score --plans landing_plans_1234.json
    python scripts/landing_benchmark.py score --plans-glob "/tmp/cola/*.json" --forms-root /tmp

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
    LANDING_BENCHMARK_PROTOCOL_VERSION,
    aggregate_nutrition,
    aggregate_reliability,
    aggregate_safety,
    build_landing_profiles,
    build_report,
    build_run_meta,
    classify_remote_error,
    corpus_profile,
    engine_band_definition,
    engine_identity,
    latency_percentiles,
    plan_delivery_state,
    score_plan_nutrition,
    score_plan_safety,
    server_change_during_run,
    strip_benchmark_meta,
    structural_facts,
    telemetry_queries,
)


# [P1-LANDING-BENCH-1-OPENAI] Forzado a OpenAI de los 4 knobs del pipeline para el modo live.
# SOLO knobs per-feature sancionados (P3-PREVIEW-MODEL-KNOB) — el override global de
# modelo fue ELIMINADO adrede (P1-SINGLE-PROVIDER-RESTORE: colapsaba también el reviewer
# clínico risk-tier a un provider de test) y NO se reintroduce aquí. Estos 4 knobs
# mueven el pipeline (flash nodes + router por tier + red post-fallo); el reviewer
# (Luna/Terra/Sol), day-gen (Luna por tier) y swap (Luna fijo) son OpenAI por default
# y conservan su routing fail-secure propio — que un knob per-feature del ENTORNO puede
# cambiar, así que la corrida no afirma «cero GLM» [P1-PLAN-LOTE-749 · ronda 1].
# [P1-PLAN-LOTE-171 · 2026-09-23] Luna = GPT-6 Luna (la misma que ahora usan por defecto el reviewer free, el
# day-gen, los swaps y la red post-fallo).
_OPENAI_FORCE_KNOBS = {
    "MEALFIT_FLASH_MODEL": "gpt-6-luna",
    "MEALFIT_MODEL_FREE_TIER": "gpt-6-luna",
    "MEALFIT_MODEL_PAID_TIER": "gpt-6-luna",
    "MEALFIT_PRO_MODEL": "gpt-6-luna",
}


def _force_openai_provider():
    """Aplica el forzado ANTES de los imports lazy de graph_orchestrator (la red
    post-fallo se resuelve al boot del módulo). Fail-loud sin OPENAI_API_KEY: sin
    key, graph_orchestrator degradaría la red a GLM en silencio y la corrida
    dejaría de ser lo que dice ser."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "--provider openai requiere OPENAI_API_KEY en el entorno. Sin ella la "
            "red post-fallo cae a GLM (fail-safe P1-NET-LUNA) y los nodos forzados "
            "no serían OpenAI. Exporta la key y reintenta."
        )
    for k, v in _OPENAI_FORCE_KNOBS.items():
        os.environ[k] = v
    print("provider=openai — knobs forzados (solo este proceso):")
    for k, v in _OPENAI_FORCE_KNOBS.items():
        print(f"  {k}={v}")
    # [P1-PLAN-LOTE-749 · 2026-09-28] Antes nombraba un modelo retirado de la familia 5.x. El
    # routing propio de reviewer/day-gen/swap depende de los knobs del entorno y no se afirma aquí.
    print("  (reviewer/day-gen/swap conservan su routing propio por tier — ver docs/llm_tier_routing.md)")


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


def _select_profiles(n=0, ids=None):
    """[P1-PLAN-LOTE-749 · 2026-09-28] Por defecto, LA MATRIZ ENTERA (25). Antes `live`/`remote`
    cortaban con `profiles[:n]` y los workflows mandaban n=20: los perfiles 21-25 (renal, anemia,
    gota, hígado graso, IMAO) no entraban nunca en una corrida por defecto. `ids` gana sobre `n`;
    `n` > 0 sigue sirviendo para un smoke barato (y el reporte lo marca `cohort_status=partial`)."""
    profiles = build_landing_profiles()
    if ids:
        return [p for p in profiles if p["_id"] in ids]
    if n and n > 0:
        return profiles[:n]
    return profiles


def _country_scope(profiles):
    """Países de la cohorte, en el orden canónico del selector (DO primero)."""
    try:
        from constants import COUNTRY_PROFILES, canonicalize_country
        orden = list(COUNTRY_PROFILES)
    except Exception:
        canonicalize_country = (lambda c: (c or "DO"))
        orden = ["DO", "ES", "US", "MX", "PR", "CO"]
    vistos = {canonicalize_country(p.get("country") or "DO") for p in profiles}
    return [c for c in orden if c in vistos] + sorted(vistos - set(orden))


def _score_delivered(row, plan, profile, band):
    """Puntúa un plan que SÍ llegó al usuario: seguridad + nutrición + gym.

    [P1-PLAN-LOTE-749 · ronda 1] El perfil pasa por la MISMA unión de texto libre que hace el
    motor (`profile_with_free_text`: «Otra…» de alergias/condiciones a sus listas, centinela
    «Ninguna» incluido): el scorer de seguridad solo lee `allergies`. Idempotente sobre un perfil
    ya unido (`corpus_profile`) y un no-op en la matriz, que hoy no trae texto libre."""
    from plan_gym import score_plan
    try:
        from graph_orchestrator import profile_with_free_text
        profile = profile_with_free_text(profile)
    except Exception as e:
        row["free_text_merge_error"] = f"{type(e).__name__}: {e}"
    fd = strip_benchmark_meta(profile)
    try:
        row["safety"] = score_plan_safety(plan, profile)
    except Exception as e:
        row["safety_error"] = f"{type(e).__name__}: {e}"
    try:
        row["nutrition"] = dict(score_plan_nutrition(plan, band=band),
                                profile_id=profile.get("_id"), label=profile.get("_label"))
    except Exception as e:
        row["nutrition_error"] = f"{type(e).__name__}: {e}"
    try:
        row["gym"] = score_plan(plan, fd)
    except Exception as e:
        row["gym_error"] = f"{type(e).__name__}: {e}"
    return row


async def _run_one_live(profile, sem, do_changes, band):
    from graph_orchestrator import arun_plan_pipeline
    async with sem:
        fd = strip_benchmark_meta(profile)
        t0 = time.time()
        try:
            plan = await arun_plan_pipeline(dict(fd))
        except Exception as e:
            # La duración de un fallo terminal TAMBIÉN es latencia (contrato B-04).
            return {"id": profile["_id"], "label": profile["_label"], "delivery": "error",
                    "duration_s": round(time.time() - t0, 1),
                    "error": f"{type(e).__name__}: {e}"}
        dur = round(time.time() - t0, 1)
        row = {"id": profile["_id"], "label": profile["_label"],
               "goal": profile.get("mainGoal"), "conditions": profile.get("medicalConditions"),
               "medications": profile.get("medications"), "diet": profile.get("dietType"),
               "duration_s": dur, "_plan": plan, "delivery": plan_delivery_state(plan)}
        if row["delivery"] == "discarded_fallback":
            # [P1-PLAN-LOTE-749] El FALLBACK-GUARD del router descarta este plan (422/503): el
            # usuario no lo recibe. Antes se puntuaba como entregado y un fallback de plantilla
            # (macros ~objetivo por construcción) inflaba la banda y la seguridad.
            row["error"] = f"fallback descartado ({(plan or {}).get('_fallback_reason') or 'sin razón'})"
            return row
        _score_delivered(row, plan, profile, band)
        if do_changes:
            try:
                row["changes"] = await asyncio.to_thread(_exercise_changes, plan, profile)
            except Exception as e:
                row["changes_error"] = f"{type(e).__name__}: {e}"
        return row


def _percentiles(values, ps=(0.5, 0.95)):
    """Delegado a `landing_benchmarks.latency_percentiles` (con `n`, que el importador usa como
    muestra de la latencia). `ps` se conserva por compatibilidad de firma."""
    return latency_percentiles(values)


def _save_plans(path, rows, attempted_ids):
    """Guarda los planes crudos + el denominador (`attempted_ids`) + el estado de entrega, para
    que `score` reproduzca la corrida sin pagar LLM (incluidos los perfiles que fallaron)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"attempted_ids": list(attempted_ids),
                   "plans": [{"id": r["id"], "label": r.get("label"), "plan": r.get("_plan"),
                              "delivery": r.get("delivery")}
                             for r in rows if r.get("_plan")]}, f,
                  ensure_ascii=False, default=str)


def _profile_sections(rows, n_attempted, band):
    """Secciones comunes de live/remote/score a partir de las filas por perfil. Solo lo ENTREGADO
    entra en safety/nutrition/gym; `reliability` y `latency` cuentan TODOS los intentos."""
    from plan_gym import aggregate_scores
    delivered = [r for r in rows if r.get("delivery") in ("delivered", "delivered_fallback")]
    gym_rows = [{"id": r["id"], "score": r["gym"]} for r in delivered if r.get("gym")]
    return {
        "safety": {
            "aggregate": aggregate_safety([r.get("safety") for r in delivered if r.get("safety")]),
            "per_profile": [r.get("safety") or {"id": r["id"], "delivery": r.get("delivery"),
                                                "error": r.get("error") or r.get("safety_error")}
                            for r in rows],
        },
        "nutrition": {
            "aggregate": aggregate_nutrition([r.get("nutrition") for r in delivered
                                              if r.get("nutrition")],
                                             n_attempted=n_attempted, band=band),
            "per_profile": [r.get("nutrition") or {"profile_id": r["id"], "scored": False,
                                                   "reason": r.get("error") or r.get("nutrition_error")
                                                   or r.get("delivery")}
                            for r in rows],
        },
        "gym": {"aggregate": aggregate_scores(gym_rows),
                "per_profile": [{k: v for k, v in r.items() if k not in ("safety", "nutrition")}
                                for r in rows]},
        "latency": {"generation_s": _percentiles([r.get("duration_s") for r in rows]),
                    "delivered_s": _percentiles([r.get("duration_s") for r in delivered])},
        "reliability": aggregate_reliability(rows, n_attempted=n_attempted),
    }


async def _live_sections(n, conc, do_changes, save_plans_path, ids=None):
    _open_pools()
    try:
        from db_core import async_connection_pool
        if async_connection_pool is not None:
            await async_connection_pool.open()
    except Exception:
        pass

    band = engine_band_definition()
    profiles = _select_profiles(n, ids)
    sem = asyncio.Semaphore(conc)
    rows = await asyncio.gather(*[_run_one_live(p, sem, do_changes, band) for p in profiles])
    rows = list(rows)

    if save_plans_path:
        _save_plans(save_plans_path, rows, [p["_id"] for p in profiles])

    for r in rows:
        r.pop("_plan", None)  # el plan crudo no viaja en el reporte (pesa cientos de KB)

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

    sections = {"structural": _structural_section(), "changes": changes,
                **_profile_sections(rows, len(profiles), band)}
    ctx = {"profile_ids": [p["_id"] for p in profiles], "country_scope": _country_scope(profiles),
           "cohort": "matrix"}
    return sections, ctx


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


# Estados con los que el deploy dice que NO sirve `/analyze/stream` (ruta ausente o método no
# permitido): solo entonces el runner cae al síncrono. tooltip-anchor: P1-PLAN-LOTE-749-SSE
_STREAM_ABSENT_STATUS = (404, 405, 501)


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
            # [P1-PLAN-LOTE-749 · ronda 2] Solo es «stream no disponible» (→ el caller cae al
            # síncrono) si el deploy NO sirve la ruta (404/405/501) o responde 200 sin SSE. Antes
            # CUALQUIER no-200 caía al síncrono: el 503 `server_busy_generating` (el tope de
            # concurrencia, que solo existe en el stream) se lo saltaba y la corrida contaba una
            # segunda generación; un 422 de validación se repetía. Ahora es un error de ESTA
            # petición, clasificado por su `detail` como el del síncrono.
            if r.status_code in _STREAM_ABSENT_STATUS or (
                    r.status_code == 200 and "text/event-stream" not in ctype):
                r.read()
                raise RuntimeError(
                    f"stream no disponible (HTTP {r.status_code}, {ctype[:40]}): {r.text[:200]}")
            if r.status_code != 200:
                r.read()
                raise RuntimeError(f"HTTP {r.status_code} en /api/plans/analyze/stream: {r.text[:400]}")
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


def _run_one_remote(api_base, profile, do_changes, timeout_s, transport="sse", band=None):
    import uuid
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
                # Deploy sin SSE utilizable (ruta ausente o 200 sin event-stream) → endpoint
                # síncrono. Un rechazo del stream (503 del tope, 422, 5xx) NO llega aquí (ronda 2).
                print(f"  (aviso) SSE no disponible, cayendo al síncrono: {str(_sse_e)[:120]}")
                plan = _remote_post(api_base, "/api/plans/analyze", payload, timeout_s)
        else:
            plan = _remote_post(api_base, "/api/plans/analyze", payload, timeout_s)
    except Exception as e:
        # [P1-PLAN-LOTE-749] Un 422/503 del FALLBACK-GUARD es un fallback DESCARTADO, no un fallo
        # de red: B-04 lo cuenta en `fallback_rate`. Y la duración del fallo también es latencia.
        _msg = f"{type(e).__name__}: {e}"
        return {"id": profile["_id"], "label": profile["_label"],
                "delivery": classify_remote_error(_msg),
                "duration_s": round(time.time() - t0, 1), "error": _msg}
    dur = round(time.time() - t0, 1)
    row = {"id": profile["_id"], "label": profile["_label"],
           "goal": profile.get("mainGoal"), "conditions": profile.get("medicalConditions"),
           "medications": profile.get("medications"), "diet": profile.get("dietType"),
           "duration_s": dur, "_plan": plan, "delivery": plan_delivery_state(plan)}
    if row["delivery"] == "discarded_fallback":
        row["error"] = f"fallback descartado ({(plan or {}).get('_fallback_reason') or 'sin razón'})"
        return row
    _score_delivered(row, plan, profile, band)

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


# Claves de `/health/version` que identifican el BINARIO medido. `git_sha` es el commit del motor
# (lo inyecta el deploy por env `GIT_SHA`); sin él, `run.source_commit` en remote es el commit de
# los scorers, no el del motor (gate G-04 del landing). [P1-PLAN-LOTE-749 · ronda 1]
_SERVER_VERSION_KEYS = ("git_sha", "git_short_sha", "deploy_timestamp", "process_started_at",
                        "last_known_pfix", "expected_marker", "drift")


def _server_version(api_base):
    """`/health/version` del deploy (público, sin LLM): qué binario corría cuando se midió. Sin esto
    un reporte remote no dice contra QUÉ motor se midió. Best-effort."""
    try:
        import httpx
        r = httpx.get(f"{api_base.rstrip('/')}/health/version", timeout=10)
        if r.status_code == 200:
            d = r.json()
            return {k: d.get(k) for k in _SERVER_VERSION_KEYS if k in d}
        return {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _remote_sections(api_base, n, conc, do_changes, save_plans_path, timeout_s, ids=None,
                     transport="sse"):
    from concurrent.futures import ThreadPoolExecutor

    band = engine_band_definition()
    profiles = _select_profiles(n, ids)
    # [P1-PLAN-LOTE-749 · ronda 1] El binario se identifica ANTES de generar y otra vez al final:
    # un deploy a mitad de corrida mezcla dos motores y el reporte tiene que decirlo.
    server_start = _server_version(api_base)
    # conc: el /analyze de un guest comparte RateLimiter por IP (3/60s); con generaciones de
    # minutos, 1-2 en vuelo no lo rozan pero >2 sí al arrancar. El workflow corre con 2.
    with ThreadPoolExecutor(max_workers=max(1, conc)) as ex:
        rows = list(ex.map(
            lambda p: _run_one_remote(api_base, p, do_changes, timeout_s, transport=transport,
                                      band=band),
            profiles))
    server_end = _server_version(api_base)
    engine = engine_identity(server_start, server_end)
    server_change = server_change_during_run(server_start, server_end)

    if save_plans_path:
        _save_plans(save_plans_path, rows, [p["_id"] for p in profiles])
    for r in rows:
        r.pop("_plan", None)

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
    sections = {
        "meta": {
            "api_base": api_base, "guest": True,
            # [P1-PLAN-LOTE-749 · 2026-09-28] Antes afirmaba un modelo 5.x retirado: el routing lo
            # decide el deploy con SUS knobs y cambió varias veces desde entonces. El reporte deja
            # constancia de CONTRA QUÉ binario midió en vez de afirmar un modelo.
            "routing": ("modelos decididos por el SERVIDOR (knobs del deploy: proveedor por "
                        "defecto + router por tier + overrides per-feature, ver "
                        "docs/llm_tier_routing.md); este reporte no afirma un modelo concreto"),
            # [ronda 2] El cambio se decide con TODAS las claves que el servidor publica (sha,
            # deploy, P-fix, arranque del proceso), no solo con `git_sha` — que hoy vale
            # "unknown" y daba `false` ante un redeploy. `None` = no verificable.
            "server_version": {"inicio": server_start, "fin": server_end,
                               "cambio_durante_la_corrida": server_change["cambio"],
                               "claves_que_cambiaron": server_change["claves"]},
            # `run.source_commit` es el del runner y sus scorers; el del motor medido es
            # `run.engine_commit` (None mientras el servidor no publique su `git_sha`).
            "source_commit_es": "commit de los scorers/runner, NO del motor medido",
        },
        "changes": changes,
        **_profile_sections(rows, len(profiles), band),
    }
    ctx = {"profile_ids": [p["_id"] for p in profiles], "country_scope": _country_scope(profiles),
           "cohort": "matrix", "source_commit_role": "scorers", "engine": engine}
    return sections, ctx


# ─────────────────────────────── telemetry ───────────────────────────────

def _telemetry_section(days):
    """[P1-PLAN-LOTE-749 · 2026-09-28] SQL en `landing_benchmarks.telemetry_queries` (testeable):
    solo lo ENTREGADO (filas de banda emparejadas con su corrida); lo que no se empareja va
    aparte en `banda_excluida_sin_corrida` y `corridas_por_entrega`."""
    _open_pools()
    d = int(days)
    out = {"window_days": d}
    for name, (sql, params) in telemetry_queries(d).items():
        out[name] = _fetch_rows(sql, params)
    return out


# ─────────────────────────────── score (replay sin LLM) ───────────────────────────────

def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _corpus_form(plan_file, forms_root):
    """Formulario de un plan del corpus: el propio fichero (`form`) o, con la convención de
    `cola_corpus` (`<dir>__<fichero>.json`), el crudo `<forms_root>/<dir>/<fichero>.json`."""
    try:
        d = _load_json(plan_file)
    except Exception:
        return None, None
    plan = d.get("final_plan") or d.get("plan") or (d if isinstance(d.get("days"), list) else None)
    form = d.get("form") if isinstance(d.get("form"), dict) else None
    if form is None and forms_root:
        base = os.path.basename(plan_file)
        head, sep, tail = base.partition("__")
        if sep:
            raw = os.path.join(forms_root, head, tail)
            try:
                form = (_load_json(raw) or {}).get("form")
            except Exception:
                form = None
    return plan, form


def _score_sections(plans_path=None, plans_glob=None, forms_root=None):
    """[P1-PLAN-LOTE-749 · 2026-09-28] Re-puntúa planes guardados — seguridad + nutrición + gym —
    SIN una sola llamada a LLM. Dos entradas:
      · `plans_path`: un `--save-plans` de live/remote (perfiles de la matriz por id; el
        denominador es `attempted_ids` si viene, así los perfiles que fallaron siguen contando).
      · `plans_glob` (+ `forms_root`): un corpus de planes reales; el perfil sale del formulario
        guardado junto al plan y las expectativas clínicas de `derive_expectations`.
    tooltip-anchor: P1-PLAN-LOTE-749-SCORE"""
    import glob as _glob
    band = engine_band_definition()
    rows, profiles_used, attempted = [], [], []
    if plans_path:
        data = _load_json(plans_path)
        matrix = {p["_id"]: p for p in build_landing_profiles()}
        items = data.get("plans", []) or []
        attempted = list(data.get("attempted_ids") or [it.get("id") for it in items])
        for item in items:
            prof = matrix.get(item.get("id"))
            if not prof:
                continue
            plan = item.get("plan")
            state = item.get("delivery") or plan_delivery_state(plan)
            row = {"id": prof["_id"], "label": prof["_label"], "delivery": state}
            if state in ("delivered", "delivered_fallback"):
                _score_delivered(row, plan, prof, band)
            rows.append(row)
            profiles_used.append(prof)
        cohort = "matrix"
    else:
        files = sorted(_glob.glob(plans_glob or ""))
        for f in files:
            pid = os.path.splitext(os.path.basename(f))[0]
            attempted.append(pid)
            plan, form = _corpus_form(f, forms_root)
            if not isinstance(plan, dict):
                rows.append({"id": pid, "label": pid, "delivery": "error",
                             "error": "fichero sin plan legible"})
                continue
            form = dict(form or {})
            # [P1-PLAN-LOTE-749 · ronda 1] El perfil con la unión de texto libre de producción
            # (antes el formulario crudo: «frutos del mar» escrito a mano no llegaba al scorer).
            prof = corpus_profile(form, pid)
            if not form:
                prof["_sin_formulario"] = True
            state = plan_delivery_state(plan)
            row = {"id": pid, "label": pid, "delivery": state,
                   "goal": form.get("mainGoal"), "country": form.get("country"),
                   "conditions": prof.get("medicalConditions"), "diet": form.get("dietType")}
            if prof.get("_free_text_discarded"):
                row["free_text_discarded"] = {f: form.get(f) for f in prof["_free_text_discarded"]}
            if state in ("delivered", "delivered_fallback"):
                _score_delivered(row, plan, prof, band)
            rows.append(row)
            profiles_used.append(prof)
        cohort = "corpus"

    sections = _profile_sections(rows, len(attempted), band)
    # Un replay no mide latencia ni entrega real: esas cifras son de la corrida que generó.
    sections.pop("latency", None)
    sections.pop("reliability", None)
    ctx = {"profile_ids": attempted, "country_scope": _country_scope(profiles_used) or ["DO"],
           "cohort": cohort,
           "sin_formulario": sum(1 for p in profiles_used if p.get("_sin_formulario")),
           # Texto libre que producción descarta por el centinela «Ninguna» (P0-FORM-1): el scorer
           # sigue la regla, pero la corrida lo nombra en vez de callarlo.
           "texto_libre_descartado": [p["_id"] for p in profiles_used
                                      if p.get("_free_text_discarded")]}
    return sections, ctx


# ─────────────────────────────── run (trazabilidad) ───────────────────────────────

def _utc_now_iso():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_source_info(root=None):
    """(commit, dirty) del backend que corre el benchmark. dirty=None = no verificable (sin git):
    el importador del landing lo rechaza igual que un árbol sucio, a propósito. El commit puede
    venir de env (`LANDING_BENCH_SOURCE_COMMIT`/`GITHUB_SHA`) si no hay git, pero la limpieza NO:
    esa no se declara, se comprueba. Se mide al ARRANCAR, antes de escribir ningún fichero.

    [P1-PLAN-LOTE-749 · ronda 1] Las salidas del propio benchmark (`landing_benchmark_*.json`,
    `landing_plans_*.json`, `run_stdout.txt`) están en `.gitignore`, y los workflows mandan el
    stdout a `$RUNNER_TEMP`: el `| tee run_stdout.txt` en la raíz del checkout ensuciaba TODAS
    las corridas de Actions y el importador las rechazaba."""
    import subprocess
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit = dirty = None
    try:
        commit = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=15, check=True).stdout.strip() or None
        st = subprocess.run(["git", "-C", root, "status", "--porcelain"], capture_output=True,
                            text=True, timeout=30, check=True).stdout
        dirty = bool(st.strip())
    except Exception:
        commit = commit or os.environ.get("LANDING_BENCH_SOURCE_COMMIT") or os.environ.get("GITHUB_SHA")
        dirty = None
    return commit, dirty


# ─────────────────────────────── main ───────────────────────────────

# `run.architecture` del contrato del landing (`protocol.required_architectures`) + el valor por
# defecto, que se DECLARA (el importador lo rechaza) en vez de adivinarse.
ARCHITECTURES = ("v1", "v2.2", "unspecified")

# Parámetros que cada modo USA de verdad — `run.parameters` lleva solo esos (antes arrastraba
# `transport`/`timeout`/`days`/`provider` en score y telemetry). [P1-PLAN-LOTE-749 · ronda 1]
_MODE_PARAMETERS = {
    "structural": (),
    "live": ("n", "conc", "changes", "save_plans", "provider", "ids"),
    "remote": ("n", "conc", "changes", "save_plans", "api_base", "ids", "transport", "timeout"),
    "telemetry": ("days",),
    "score": ("plans", "plans_glob", "forms_root"),
}
_DEFAULT_CONC = {"live": 2, "remote": 1}


def _run_parameters(args) -> dict:
    """`run.parameters`: solo los del modo, con la concurrencia EFECTIVA (no el `None` del CLI)."""
    out = {}
    for k in _MODE_PARAMETERS.get(args.mode, ()):
        v = getattr(args, k, None)
        if k == "conc":
            v = max(1, v or _DEFAULT_CONC.get(args.mode, 1))
        if v is None or v is False:
            continue
        out[k] = v
    return out


def _build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("structural", "live", "remote", "telemetry", "score"))
    ap.add_argument("n", nargs="?", type=int, default=0,
                    help="live/remote: límite de perfiles (0 = la matriz entera: 25)")
    ap.add_argument("--conc", type=int, default=None,
                    help="concurrencia (default: live=2, remote=1 por el rate-limit per-IP)")
    ap.add_argument("--changes", action="store_true", help="live/remote: ejercitar cambios")
    ap.add_argument("--save-plans", action="store_true",
                    help="live/remote: guardar planes crudos para `score`")
    ap.add_argument("--provider", choices=("default", "openai"), default="default",
                    help="live: openai fuerza toda la corrida a la familia OpenAI (_OPENAI_FORCE_KNOBS)")
    ap.add_argument("--api-base", help="remote: URL base del API desplegado (p.ej. https://app.bioboros.com)")
    ap.add_argument("--ids", help="live/remote: perfiles específicos por id, p.ej. '3,9,10,13,15' "
                                  "(los clínicos; gana sobre el N posicional)")
    ap.add_argument("--transport", choices=("sse", "sync"), default="sse",
                    help="remote: sse (default; /analyze/stream con heartbeats, inmune al "
                         "proxy_read_timeout) o sync (endpoint de un solo response)")
    ap.add_argument("--timeout", type=int, default=1200, help="remote: timeout por plan en segundos")
    ap.add_argument("--days", type=int, default=30, help="telemetry: ventana en días")
    ap.add_argument("--plans", help="score: JSON de una corrida live/remote --save-plans")
    ap.add_argument("--plans-glob", help="score: glob de planes guardados de un corpus real "
                                         "({'final_plan'|'plan', 'form'?} por fichero)")
    ap.add_argument("--forms-root", help="score: raíz de los crudos para recuperar el formulario "
                                         "por la convención <dir>__<fichero>.json de cola_corpus")
    ap.add_argument("--architecture", choices=ARCHITECTURES,
                    default=os.environ.get("LANDING_BENCH_ARCHITECTURE") or "unspecified",
                    help="run.architecture (v1 | v2.2 en el contrato del landing); se DECLARA, no se adivina")
    ap.add_argument("--protocol-version", default=LANDING_BENCHMARK_PROTOCOL_VERSION,
                    help="run.protocol_version (debe coincidir con protocol.version del landing)")
    ap.add_argument("--out", help="ruta del JSON de salida")
    return ap


def _parse_args(argv=None):
    """Parsea y valida. `choices` no revisa un default que viene de env: se valida aquí también."""
    ap = _build_parser()
    args = ap.parse_args(argv)
    if args.architecture not in ARCHITECTURES:
        ap.error(f"--architecture/LANDING_BENCH_ARCHITECTURE debe ser uno de {ARCHITECTURES}")
    if args.mode == "remote" and not args.api_base:
        ap.error("--api-base es obligatorio en modo remote")
    if args.mode == "score" and not (args.plans or args.plans_glob):
        ap.error("score necesita --plans o --plans-glob")
    return args


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = _parse_args()

    started_at = _utc_now_iso()
    source_commit, source_dirty = _git_source_info()
    _ids = {int(x) for x in args.ids.split(",") if x.strip()} if args.ids else None
    params = _run_parameters(args)
    ctx = {"profile_ids": [], "country_scope": [], "cohort": "matrix"}

    if args.mode == "structural":
        _open_pools()
        sections = {"structural": _structural_section()}
        ctx["country_scope"] = list((sections["structural"].get("por_pais") or {}).keys())
    elif args.mode == "live":
        if args.provider == "openai":
            _force_openai_provider()
        save_path = f"landing_plans_{os.getpid()}.json" if args.save_plans else None
        sections, ctx = asyncio.run(_live_sections(args.n, params["conc"], args.changes,
                                                   save_path, ids=_ids))
        if save_path:
            print(f"planes crudos: {save_path}")
    elif args.mode == "remote":
        save_path = f"landing_plans_{os.getpid()}.json" if args.save_plans else None
        sections, ctx = _remote_sections(args.api_base, args.n, params["conc"],
                                         args.changes, save_path, args.timeout, ids=_ids,
                                         transport=args.transport)
        if save_path:
            print(f"planes crudos: {save_path}")
    elif args.mode == "telemetry":
        sections = {"telemetry": _telemetry_section(args.days)}
    else:
        # Los scorers leen el catálogo (master_ingredients) como en live: sin pool abierto
        # degradarían en silencio a catálogo vacío. Solo lectura.
        _open_pools()
        if args.plans:
            with open(args.plans, "rb") as _pf:
                import hashlib
                params["plans_sha256"] = hashlib.sha256(_pf.read()).hexdigest()
        sections, ctx = _score_sections(plans_path=args.plans, plans_glob=args.plans_glob,
                                        forms_root=args.forms_root)
        if ctx.get("sin_formulario"):
            params["planes_sin_formulario"] = ctx["sin_formulario"]
        if ctx.get("texto_libre_descartado"):
            params["texto_libre_descartado"] = ctx["texto_libre_descartado"]

    sections["run"] = build_run_meta(
        mode=args.mode, started_at=started_at, finished_at=_utc_now_iso(),
        source_commit=source_commit, source_dirty=source_dirty,
        architecture=args.architecture, protocol_version=args.protocol_version,
        country_scope=ctx.get("country_scope") or [], profile_ids=ctx.get("profile_ids") or [],
        full_profile_ids=[p["_id"] for p in build_landing_profiles()], parameters=params,
        cohort=ctx.get("cohort") or "matrix", source_commit_role=ctx.get("source_commit_role"),
        engine=ctx.get("engine"))
    report = build_report(args.mode, **sections)
    out_path = args.out or os.environ.get("LANDING_BENCH_OUT") \
        or f"landing_benchmark_{args.mode}_{os.getpid()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)

    print("\n========== LANDING BENCHMARK - resumen ==========")
    run = report["run"]
    print(f"modo: {args.mode} | schema v{report['schema_version']} | run {run['id']} | "
          f"cohorte {run['cohort_status']} ({run['profile_count']}/{run['full_profile_count']}) | "
          f"arquitectura {run['architecture']} | commit {run['source_commit']} "
          f"({run['source_commit_role']}, sucio={run['source_dirty']}) | motor "
          f"{run['engine_commit']} ({run['engine_commit_status']})")
    if "meta" in report:
        print(f"  remote: {report['meta'].get('api_base')} (guest) — servidor "
              f"{report['meta'].get('server_version')}")
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
    if "nutrition" in report:
        n = report["nutrition"]["aggregate"]
        print(f"  nutrición: n={n.get('n_scored')}/{n.get('n_attempted')} días={n.get('days_evaluated')} "
              f"MAPE={n.get('per_macro_mape_pct')} media={n.get('macro_mape_pct')} "
              f"peor={n.get('worst_macro')}:{n.get('worst_macro_mape_pct')} | "
              f"4-en-banda={n.get('four_macros_in_band_pct')}% (banda {n.get('band')})")
    if "gym" in report and report["gym"].get("aggregate"):
        g = report["gym"]["aggregate"]
        print(f"  gym: n={g.get('n')} global={g.get('global_mean')}")
    if "latency" in report:
        print(f"  latencia generación (todo intento): {report['latency'].get('generation_s')}")
    if "reliability" in report:
        r = report["reliability"]
        print(f"  entrega: {r.get('n_delivered')}/{r.get('n_attempted')} ({r.get('delivery_rate_pct')}%) "
              f"| fallback {r.get('fallback_rate_pct')}% | petición rechazada "
              f"{r.get('n_rejected_request')} | errores {r.get('n_errors')}")
    if report.get("changes"):
        c = report["changes"]
        rd = c.get("regen_day") or {}
        dia = rd.get("skipped") or (f"{rd.get('meals_ok')}/{rd.get('meals_total')} "
                                    f"lat={rd.get('latency_day_s')}")
        print(f"  swap: ok={c['swap'].get('ok_pct')}% lat={c['swap'].get('latency_s')} | día: {dia}")
    if "telemetry" in report:
        t = report["telemetry"]
        print(f"  telemetría ({t['window_days']}d, solo entregado): cambios={t['changes']} | "
              f"banda={t['banda_entregada']} | fallback={t['fallback_rate']} | "
              f"latencia={t['generacion_latencia']} | PQI={t['quality_index']}")
        print(f"  aparte: banda sin corrida detrás={t['banda_excluida_sin_corrida']} | "
              f"corridas por entrega={t['corridas_por_entrega']}")
    print(f"\nJSON completo: {out_path}")


if __name__ == "__main__":
    main()
