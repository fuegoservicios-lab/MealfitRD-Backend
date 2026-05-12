"""[P2-LIVE-7 + P2-LIVE-8 + P2-LIVE-9 · 2026-05-11] Tres P2 del audit live:

P2-LIVE-7 — api_usage write path para endpoints quota-gated
    `verify_api_quota` (auth.py:49) SOLO lee `get_monthly_api_usage` para
    comparar contra el tier limit — no incrementa el contador. Endpoints
    deben llamar explícitamente `log_api_usage(user_id, "<tag>")` tras
    operación exitosa. Audit live detectó 5 endpoints con `Depends(verify_api_quota)`
    SIN llamada a `log_api_usage` en su body:
      - /shift-plan, /analyze/stream, /retry-chunk, /regenerate-simplified,
        /regen-degraded.
    Resultado: un user podía retry-chunkear / shift-pleear / regen-erar
    ilimitadamente sin tocar su cap mensual (DOS amplificable, especialmente
    `/retry-chunk` que fuerza ejecución del worker LLM).
    Fix: añadir log_api_usage defensivo (try/except, no abortar request) en
    el success path de cada endpoint.

P2-LIVE-8 — hard-cap edad absoluta para scheduler_missed_<job>
    Sweep #1 standard cierra al TTL=24h (knob MEALFIT_SCHEDULER_ALERT_TTL_H).
    Audit live detectó que jobs recurring que NO recuperan (job stuck) dejan
    alerts visibles 24h pese a que el listener P1-NEW-2 (EVENT_JOB_EXECUTED)
    podría haberlos cerrado si el job fuera capaz de ejecutar. 12h es señal
    suficiente: si el job no firmó en 12h, mejor cerrar el alert y dejar que
    P0-2 detector re-emita en su próximo tick — sin alert fatigue.
    Fix: sweep #5 en `_resolve_stale_scheduler_alerts` con knob
    `MEALFIT_SCHEDULER_MISSED_HARD_CAP_HOURS` (default 12, clamp [1, 168]).
    Espejo P2-NEW-E (cascade hard-cap). Excluye scheduler_error_* (errores
    merecen TTL más largo para investigación) Y scheduler_cascade_missed
    (su propio hard-cap en sweep #4).

P2-LIVE-9 — tick observable a `_shopping_coherence_alert_job`
    Cron diario 04:00 UTC re-evalúa coherencia recetas↔lista. Pre-fix: log
    INFO/ERROR del summary, sin emit a pipeline_metrics. Sin tick:
      - audit live no podía confirmar si el job corrió ese día durante la
        cascada del scheduler (no había señal observable).
      - los counters internos (n_plans, plans_with_div, persisted_count,
        eval_errors, persist_errors, skip_reason) no quedaban analizables
        post-hoc.
    Fix: wrap body en try/finally + emit `_shopping_coherence_alert_job_tick`
    con 9 flags. Patrón espejo P3-LIVE-1 / P1-LIVE-4.

Drift detection:
    - log_api_usage removido de los 5 endpoints quota-gated → falla.
    - Knob `MEALFIT_SCHEDULER_MISSED_HARD_CAP_HOURS` renombrado/borrado → falla.
    - sweep #5 elimina el filtro `<> 'scheduler_cascade_missed'` (sobre-cierre
      del parent que tiene su propio hard-cap) → falla.
    - Tick `_shopping_coherence_alert_job_tick` desaparece o sale del finally → falla.

Tooltip-anchor: P2-LIVE-7-START | P2-LIVE-8-START | P2-LIVE-9-START | gap audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent
_PLANS = _BACKEND / "routers" / "plans.py"
_CRON = _BACKEND / "cron_tasks.py"
_APP = _BACKEND / "app.py"


@pytest.fixture(scope="module")
def plans_source() -> str:
    return _PLANS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def cron_source() -> str:
    return _CRON.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app_source() -> str:
    return _APP.read_text(encoding="utf-8")


def _read_function_body(source: str, fn_name: str) -> str:
    pattern = re.compile(
        rf"^(?:async\s+)?def\s+{re.escape(fn_name)}\s*\(",
        re.MULTILINE,
    )
    m = pattern.search(source)
    if not m:
        return ""
    next_def_pattern = re.compile(r"^(async\s+def |def |class |@)", re.MULTILINE)
    next_def = next_def_pattern.search(source, pos=m.end())
    if next_def:
        return source[m.start():next_def.start()]
    return source[m.start():]


# ---------------------------------------------------------------------------
# P2-LIVE-7: log_api_usage en endpoints quota-gated
# ---------------------------------------------------------------------------

# (endpoint_function_name, expected_log_tag_substring)
QUOTA_GATED_ENDPOINTS = [
    ("api_shift_plan", "shift_plan"),
    ("api_analyze_stream", "analyze_stream"),
    ("api_retry_chunk", "retry_chunk"),
    ("api_regenerate_dead_lettered_simplified", "regenerate_simplified"),
    ("api_regen_degraded_chunks", "regen_degraded"),
]


@pytest.mark.parametrize("fn_name, expected_tag", QUOTA_GATED_ENDPOINTS)
def test_p2_live_7_quota_gated_endpoint_calls_log_api_usage(
    plans_source: str, fn_name: str, expected_tag: str
):
    """Cada endpoint con `Depends(verify_api_quota)` debe llamar
    `log_api_usage(<user_id>, "<endpoint_tag>")` en su body. Sin esto,
    `verify_api_quota` cobra cap pero no incrementa el contador → bypass."""
    body = _read_function_body(plans_source, fn_name)
    assert body, (
        f"[P2-LIVE-7] Function `{fn_name}` no encontrada en routers/plans.py. "
        f"Si renombraste el endpoint, actualiza QUOTA_GATED_ENDPOINTS en este test."
    )
    assert "log_api_usage" in body, (
        f"[P2-LIVE-7] `{fn_name}` usa `verify_api_quota` pero NO llama "
        f"`log_api_usage`. El paywall solo lee — sin esta llamada el cap "
        f"mensual no se incrementa cuando este endpoint dispara."
    )
    # Verifica que el tag descriptive incluye `expected_tag` para post-mortem.
    log_pattern = re.compile(
        rf'log_api_usage\([^)]*["\'][^"\']*{re.escape(expected_tag)}[^"\']*["\']'
    )
    assert log_pattern.search(body), (
        f"[P2-LIVE-7] `{fn_name}` llama log_api_usage pero el tag no incluye "
        f"`{expected_tag}`. Tags claros facilitan analytics: "
        f"`SELECT endpoint, COUNT(*) FROM api_usage GROUP BY endpoint`."
    )


def test_p2_live_7_log_api_usage_imported(plans_source: str):
    """`log_api_usage` debe estar importado en plans.py."""
    assert re.search(r"from\s+db\s+import\s+[^)]*\blog_api_usage\b", plans_source) or \
           re.search(r"import\s+log_api_usage", plans_source), (
        "[P2-LIVE-7] `log_api_usage` no importado en routers/plans.py."
    )


# ---------------------------------------------------------------------------
# P2-LIVE-8: hard-cap edad absoluta para scheduler_missed_<job>
# ---------------------------------------------------------------------------

def test_p2_live_8_missed_hard_cap_sweep_present(cron_source: str):
    """`_resolve_stale_scheduler_alerts` debe incluir sweep #5 con knob
    `MEALFIT_SCHEDULER_MISSED_HARD_CAP_HOURS`."""
    body = _read_function_body(cron_source, "_resolve_stale_scheduler_alerts")
    assert body, "[P2-LIVE-8] `_resolve_stale_scheduler_alerts` no encontrado."
    assert "MEALFIT_SCHEDULER_MISSED_HARD_CAP_HOURS" in body, (
        "[P2-LIVE-8] Knob `MEALFIT_SCHEDULER_MISSED_HARD_CAP_HOURS` ausente. "
        "Sin él no hay forma de ajustar el threshold sin redeploy."
    )
    assert "missed_hard_cap_swept" in body, (
        "[P2-LIVE-8] Variable de counter `missed_hard_cap_swept` desaparecida. "
        "El tick observable la usa para post-mortem."
    )


def test_p2_live_8_sweep_excludes_cascade_parent(cron_source: str):
    """El sweep #5 debe EXCLUIR `scheduler_cascade_missed` (tiene su propio
    hard-cap en sweep #4) Y `scheduler_error_*` (errores merecen TTL más
    largo)."""
    body = _read_function_body(cron_source, "_resolve_stale_scheduler_alerts")
    # El sweep #5 debe filtrar `<> 'scheduler_cascade_missed'`.
    assert "alert_key <> 'scheduler_cascade_missed'" in body, (
        "[P2-LIVE-8] Sweep #5 no excluye `scheduler_cascade_missed`. Cerrarlo "
        "aquí sobre-escribe el contrato de P2-NEW-E (hard-cap específico)."
    )
    # El sweep #5 debe operar SOLO sobre `scheduler_missed_%`, no `scheduler_error_%`.
    sweep_5_region = body.split("P2-LIVE-8")[1] if "P2-LIVE-8" in body else ""
    assert "scheduler_missed_%%" in sweep_5_region, (
        "[P2-LIVE-8] Sweep #5 debe filtrar `alert_key LIKE 'scheduler_missed_%%'` "
        "(no scheduler_error_*)."
    )


def test_p2_live_8_tick_metadata_extended(cron_source: str):
    """El tick `_scheduler_alerts_sweep_tick` debe exponer `swept_missed_hard_cap`
    y `missed_hard_cap_h` para correlación post-mortem."""
    body = _read_function_body(cron_source, "_resolve_stale_scheduler_alerts")
    assert "swept_missed_hard_cap" in body, (
        "[P2-LIVE-8] Metadata del tick no expone `swept_missed_hard_cap`. "
        "Sin ello no podemos contar cuántas alerts cerró el sweep #5 vs los otros."
    )
    assert "missed_hard_cap_h" in body, (
        "[P2-LIVE-8] Metadata del tick no expone `missed_hard_cap_h`. "
        "Sin el valor del knob, post-mortem no puede reproducir el run."
    )


def test_p2_live_8_knob_clamped(cron_source: str):
    """Knob debe clamparse a [1, 168] (1h–7d) para evitar valores absurdos."""
    body = _read_function_body(cron_source, "_resolve_stale_scheduler_alerts")
    # Patrón: missed_hard_cap_h = max(...) o if missed_hard_cap_h > 168
    assert re.search(r"missed_hard_cap_h\s*[<>=]\s*168", body), (
        "[P2-LIVE-8] El knob `MEALFIT_SCHEDULER_MISSED_HARD_CAP_HOURS` no está "
        "clamado a max 168h. Sin clamp, un valor mal seteado (ej. 10000) "
        "desactiva el sweep efectivamente."
    )


# ---------------------------------------------------------------------------
# P2-LIVE-9: tick observable en _shopping_coherence_alert_job
# ---------------------------------------------------------------------------

def test_p2_live_9_tick_emitted_in_finally(cron_source: str):
    """El tick `_shopping_coherence_alert_job_tick` debe emitirse en un
    bloque `finally:` para garantizar emisión SIEMPRE (incluye early-returns
    de skip_reason)."""
    body = _read_function_body(cron_source, "_shopping_coherence_alert_job")
    assert body, "[P2-LIVE-9] `_shopping_coherence_alert_job` no encontrado."
    assert "_shopping_coherence_alert_job_tick" in body, (
        "[P2-LIVE-9] Tick observable ausente. Sin él, el cron diario 04:00 UTC "
        "es invisible cuando no hay alerta — no podemos confirmar que corrió."
    )
    finally_idx = body.find("finally:")
    tick_idx = body.rfind("_shopping_coherence_alert_job_tick")
    assert finally_idx != -1, (
        "[P2-LIVE-9] La función debe usar try/finally para emitir tick siempre. "
        "Pattern P3-LIVE-1: la mayoría de las salidas son early return, "
        "necesitamos `finally:` para cubrir las 6 paths de skip + el path full."
    )
    assert finally_idx < tick_idx, (
        "[P2-LIVE-9] El tick debe estar DENTRO del `finally:` block."
    )


@pytest.mark.parametrize("flag", [
    "n_plans",
    "plans_with_div",
    "cap_count",
    "eval_errors",
    "persisted_count",
    "persist_errors",
    "alert_emitted",
    "persist_history_enabled",
    "skip_reason",
])
def test_p2_live_9_tick_has_required_flag(cron_source: str, flag: str):
    """El tick debe incluir 9 flags para reconstruir el estado del cron sin
    re-correrlo."""
    body = _read_function_body(cron_source, "_shopping_coherence_alert_job")
    # Buscar en el bloque finally específicamente
    finally_idx = body.find("finally:")
    tick_block = body[finally_idx:] if finally_idx != -1 else ""
    assert flag in tick_block, (
        f"[P2-LIVE-9] Tick observable no incluye flag `{flag}`. "
        f"Reconstruir el estado del cron sin él requiere re-correrlo, "
        f"caro (24h lookback × 500 plans)."
    )


def test_p2_live_9_skip_reasons_distinguishable(cron_source: str):
    """`skip_reason` debe poder tomar valores específicos para distinguir
    los 5 early-return paths (no solo bool True/False)."""
    body = _read_function_body(cron_source, "_shopping_coherence_alert_job")
    expected_reasons = [
        "db_core_import_failed",
        "supabase_not_initialized",
        "fetch_plans_failed",
        "below_min_plans",
        "guard_persist_import_failed",
    ]
    for reason in expected_reasons:
        assert reason in body, (
            f"[P2-LIVE-9] skip_reason `{reason}` no encontrado. Post-mortem "
            f"debe poder distinguir todas las paths de salida del cron."
        )


# ---------------------------------------------------------------------------
# Marker bump
# ---------------------------------------------------------------------------

def test_marker_bumped_to_p2_live(app_source: str):
    """`_LAST_KNOWN_PFIX` debe ser >= la fecha de cierre de P2-LIVE-9.

    Diseño original (escrito al cerrar P2-LIVE-9 el 2026-05-11): chequear
    presencia literal de `P2-LIVE` en el marker, para forzar el bump al
    merge. Limitación: cualquier bump futuro con prefix distinto
    (P3-NEW-E, P0-PROD-1, etc.) hacía fallar el test sin razón
    funcional — el bump SÍ ocurrió, solo no contenía el substring.

    Relajado [P0-PROD-1 · 2026-05-12]: comparar contra date floor en
    lugar de substring. Preserva intent original (forzar bump tras
    P2-LIVE-9) sin colisionar con bumps subsecuentes. Mismo patrón que
    `test_p3_1_last_known_pfix_freshness::test_marker_date_meets_floor`.
    """
    from datetime import date, datetime
    m = re.search(
        r'_LAST_KNOWN_PFIX\s*=\s*["\']([^"\']+)["\']',
        app_source,
    )
    assert m, "[P2-LIVE-9] `_LAST_KNOWN_PFIX` no encontrado."
    marker = m.group(1)
    # Date floor: P2-LIVE-9 cerró el 2026-05-11. Cualquier marker con
    # fecha >= ese floor implica que el bump del cierre P2-LIVE ocurrió
    # (aunque el slug exacto fue superseded por P-fixes posteriores).
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert date_match, (
        f"[P2-LIVE-9] `_LAST_KNOWN_PFIX={marker}` sin fecha ISO. "
        f"El marker debe seguir formato `Pn-X · YYYY-MM-DD`."
    )
    marker_date = datetime.strptime(date_match.group(1), "%Y-%m-%d").date()
    floor = date(2026, 5, 11)  # P2-LIVE-9 closure date
    assert marker_date >= floor, (
        f"[P2-LIVE-9] `_LAST_KNOWN_PFIX={marker}` (date={marker_date}) "
        f"anterior al floor {floor} de cierre P2-LIVE-9. "
        f"Sube el marker."
    )
