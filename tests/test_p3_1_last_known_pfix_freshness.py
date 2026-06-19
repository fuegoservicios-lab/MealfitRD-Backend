"""[P3-1 · 2026-05-08] Regression guard: `_LAST_KNOWN_PFIX` debe estar fresco.

Bug observado en el audit 2026-05-08 (post-P2-2):
  `_LAST_KNOWN_PFIX` en `app.py:24` quedó stale en `"P3-B · 2026-05-08"` durante
  varias rondas de P-fixes (P3-A audit late, P1-A knobs runtime, P1-B ChatWidget,
  P2-A safeJSONParse, P3-A constants docstring) sin bump. `/health/version`
  reportaba un marker desactualizado → diagnóstico de deploy menos preciso
  ("¿el último P-fix está vivo en prod?" responde mal).

Causa: marker mantenido humanamente sin enforcement. Cada P-fix DEBE bumpearlo
pero la convención no estaba documentada ni testeada.

Fix:
  1. Convención añadida a `CLAUDE.md` ("Convenciones del repo").
  2. Comentario inline en `app.py:24` apunta a la convención + este test.
  3. Este test bloquea regresiones:
     - Formato `Pn-X · YYYY-MM-DD` o `Pn-NEW-X · YYYY-MM-DD` (suffix multi-segmento OK).
     - Date parses como ISO date.
     - Date >= floor (último audit cerrado, bumpeado junto con el marker).
     - Prefix válido (`P0`-`P9`).

Cuando un P-fix se mergea, el operador bumpea AMBOS:
  - `_LAST_KNOWN_PFIX` en `app.py:24`.
  - `_PFIX_DATE_FLOOR` aquí abajo.
Ambos en el mismo commit. Si solo uno cambia, el test falla en CI.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY_PATH = _BACKEND_ROOT / "app.py"

# Floor: el P-fix más reciente cerrado en HEAD. Bumpear AL MISMO TIEMPO que
# `_LAST_KNOWN_PFIX` en `app.py:24`. Marca la fecha mínima aceptable.
#
# Si has cerrado un P-fix posterior y olvidaste subir este floor, el test
# fallará intencionalmente — es la red de seguridad que cierra P3-1.
_PFIX_DATE_FLOOR = date(2026, 6, 19)  # [P1-RENAL-SODIUM-SUBS · 2026-06-19] bump junto con _LAST_KNOWN_PFIX (audit fresco P1: embarazo-intake/ahorradores-K/renal-truthup-recheck/renal-sodio-enforced). Histórico — floor previo P1-MEDICATION-RULES 2026-06-18 (medicamentos/warfarina-vitK/reconcile-default). Histórico — floor previo P0-DEAD-LETTER-USER-NOTIFY 2026-05-27: cierra el bug donde dead-letters del worker (GAP3 + cualquier exception tras CHUNK_MAX_FAILURE_ATTEMPTS) bypassean `_escalate_unrecoverable_chunk` en las dos UPDATEs del outer-catch (`is_critical=True/False` en cron_tasks.py:27600+/27621+). Pre-fix: `dead_letter_reason = str(e)[:240]` escribia literal raw del Exception (ej. `"[GAP3] Chunk 2 numeracion invalida..."`) que NO esta en `ESCALATION_REASONS` (constants.py:2482) → `/blocked_reasons` caia al `_unknown` fallback (plans.py:3733) y `_escalate_unrecoverable_chunk` (P2-NEW-3) rechazaria cualquier callsite posterior. La rama `is_degraded=True` fatal (L27649+) push-notificaba inline con copy mojibakeado (`"âš ï¸¸ Error extendiendo tu plan"` bytes double-encoded en source UTF-8) Y NO seteaba `_recovery_exhausted_chunks` ni `_user_action_required` en `meal_plans.plan_data` → banner del Dashboard nunca aparecia y el per-chunk alert `dead_lettered_chunk:<plan>:<week>` nunca se emitia (solo la agregada `dead_lettered_chunks_recent` que solo SRE ve). La rama `is_degraded=False` (downgrade-to-shuffle, L27692+) rescata chunks `pending`+`failed` a `pending` con `_degraded=true` pero pre-fix NO limpiaba `dead_lettered_at`/`dead_letter_reason` → cron P1-2 `_alert_new_dead_lettered_chunks` contaba falsos positivos (chunks vivos en Smart Shuffle marcados como dead-lettered). Evidencia viva audit 2026-05-27: chunk `2603e618` plan `1cb1d027` user `8e40b0fd` dead-lettered con reason raw GAP3 hace 4.2h, week 2 dias 4-7 faltantes de un plan de 15 dias, usuario en limbo sin saber nada. Fix 3-en-1: (1) ambas UPDATEs ahora escriben `dead_letter_reason = "recovery_exhausted"` canonical (forensic full sigue en `pipeline_metrics.error_message` via `_record_chunk_metric`), (2) rama `is_degraded=True` invoca `_escalate_unrecoverable_chunk(escalation_reason="recovery_exhausted")` en lugar del push mojibake — helper hace los 4 pasos idempotentemente (UPDATE chunk COALESCE-safe, UPDATE meal_plans._recovery_exhausted_chunks + _user_action_required, INSERT system_alerts per-chunk, push con copy correcto + deeplink `/dashboard?action_required=recovery_exhausted`), (3) rama `is_degraded=False` rescue UPDATE limpia `dead_lettered_at = NULL, dead_letter_reason = NULL` para revertir TODO el estado de dead-letter. Test [`test_p0_dead_letter_user_notify.py`](backend/tests/test_p0_dead_letter_user_notify.py) parser-based (7 assertions: ESCALATION_REASONS canonical, ambas UPDATEs usan `"recovery_exhausted"`, anti-regresion `str(e)[:240]`, rama fatal call signature, mojibake gone live code, rescue clears, marker anchor) + funcional (ESCALATION_REASONS sanity con stub apscheduler). Detalle: project_p0_dead_letter_user_notify_2026_05_27.md. 4 cambios runtime + 2 scripts CLI local. (P2-1 RACE-FIX-ORPHAN) `_cleanup_orphan_chunks` migrado a CTE atómico `WITH orphans AS (SELECT ... FOR UPDATE SKIP LOCKED) UPDATE ... FROM orphans RETURNING` — single-statement Postgres garantiza atomicidad contra concurrent INSERT de meal_plans entre el SELECT y el UPDATE; release_chunk_reservations ahora DESPUÉS del UPDATE (best-effort; si crash, cron de orphan-reservations limpia). (P2-2 BATCH-ATOMICITY) `_recover_failed_chunks_for_long_plans` revierte el counter `recovery_attempts` si `_enqueue_plan_chunk` falla; persiste `last_recovery_rollback_at` + reason. Cierra pérdida silenciosa donde counter bumpeado sin re-encolado real escalaba a CHUNK_MAX_RECOVERY_ATTEMPTS sin nunca haber re-intentado. (P2-6 NIGHTLY-OBS) `_nightly_refresh_all_pending_snapshots` gana per-user `_call_with_timeout` + total budget (paridad con P1-4) + `_track_cron_consecutive_failure` con alert_key `nightly_refresh_all_pending_snapshots_failures_burst`. Knobs `MEALFIT_NIGHTLY_REFRESH_PER_USER_TIMEOUT_S` default 30 clamp [5,300], `MEALFIT_NIGHTLY_REFRESH_TOTAL_BUDGET_S` default 600 clamp [60,1800]. (P2-3 UNPUSHED-AGE) script CLI standalone `backend/scripts/check_unpushed_age.py` — detecta commits unpushed + dirty files con age > threshold (default 24h); zero deps backend runtime, pensado para git hooks. (P2-4 CB-FOSSIL) script CLI `backend/scripts/cleanup_stale_cb_rows.py` — preview/apply manual de DELETE de rows fósiles `llm_circuit_breaker:<modelo_deprecado>` en zero canonical; `--apply` requiere `--models` explícito (defense). P2-5 (N+1 en drain_pending_facts_queue) YA cerrado por P1-2 con `MEALFIT_FACTS_DRAIN_USERS_PER_TICK`. Test [`test_p2_ops_bundle.py`](backend/tests/test_p2_ops_bundle.py) 21 tests parser-based + funcionales (smoke `--help` con `encoding=utf-8,errors=replace` para Windows). Detalle: project_p2_ops_bundle_2026_05_26.md. (P1-1 KV-SWEEP) nuevo cron `_sweep_stale_app_kv_store_prefixes` con catálogo declarativo `_KV_SWEEP_PREFIXES` (4 entries: title_gen_inflight:, pending_pipeline:, rag_, reflection_) — cierra leak observado de keys huérfanas hasta 449h. (P1-2 CRON-CONSECUTIVE-FAIL) helper SSOT `_track_cron_consecutive_failure` aplicado en 3 crons silenciosos (drain_pending_facts_queue, resolve_stale_scheduler_alerts, recover_failed_chunks_for_long_plans) + el nuevo P1-1 — patrón espejo del gold-standard P1-COH-CRON-CONSECUTIVE-FAIL. (P1-3 KNOB-CLAMPS) 3 knobs migrados a `_env_int(..., validator=...)`: MEALFIT_COH_ALERT_MIN_PLANS [1,10_000], MEALFIT_FAILED_DEDUCTIONS_ALERT_THRESHOLD [1,100_000], CHUNK_RECOVERY_BATCH_LIMIT [1,1000] — pre-fix tenían floor manual o cero validación. (P1-4 CRON-TIMEOUT) helper `_call_with_timeout` daemon-thread + per-user/total-budget caps en proactive refresh — un user lento ya no bloquea el resto del cron. (P1-5 ROLLING-ABANDONED) extensión del sweep stranded con segunda branch para `partial+days>0+chunks_all_pending_user_action+age>168h` — cierra gap simétrico observado en plan 1cb1d027. Test parser-based + funcional [`test_p1_cron_bundle.py`](backend/tests/test_p1_cron_bundle.py) con 25 tests. Detalle: project_p1_cron_bundle_2026_05_26.md.

# Formato de marker permitido: `P<n>(-<seg>)+ · YYYY-MM-DD`. Suffix
# multi-segmento permitido para `P2-NEW-A`, `P3-CANDIDATE-B`, etc.
_MARKER_PATTERN = re.compile(
    r"^(?P<prefix>P\d+(?:-[A-Z0-9]+)+)\s+·\s+(?P<date>\d{4}-\d{2}-\d{2})$"
)


def _read_marker_from_app_py() -> str:
    """Extrae el valor literal de `_LAST_KNOWN_PFIX` desde `app.py` sin
    importar el módulo (que dispara cron schedulers + DB init).

    Estrategia: regex sobre el source. Más rápida y aislada que `import app`.
    """
    text = _APP_PY_PATH.read_text(encoding="utf-8")
    m = re.search(
        r'^_LAST_KNOWN_PFIX\s*=\s*["\'](?P<val>[^"\']+)["\']',
        text,
        re.MULTILINE,
    )
    assert m is not None, (
        "No se encontró asignación literal `_LAST_KNOWN_PFIX = '...'` en "
        f"{_APP_PY_PATH}. ¿Fue movido a otro módulo o computado dinámicamente? "
        "Si es intencional, actualizar este test."
    )
    return m.group("val")


def test_marker_present_and_format_valid():
    """`_LAST_KNOWN_PFIX` existe y sigue `Pn-(seg-)+ · YYYY-MM-DD`."""
    marker = _read_marker_from_app_py()
    m = _MARKER_PATTERN.match(marker)
    assert m is not None, (
        f"`_LAST_KNOWN_PFIX={marker!r}` no sigue el formato "
        f"`Pn-X · YYYY-MM-DD` o `Pn-NEW-X · YYYY-MM-DD`. "
        f"Convención en CLAUDE.md → 'Convenciones del repo'."
    )


def test_marker_date_parses_as_iso():
    """La fecha en el marker debe ser ISO válida (YYYY-MM-DD)."""
    marker = _read_marker_from_app_py()
    m = _MARKER_PATTERN.match(marker)
    assert m is not None, f"Marker mal formado: {marker!r}"
    date_str = m.group("date")
    try:
        datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as e:
        pytest.fail(
            f"Fecha en `_LAST_KNOWN_PFIX={marker!r}` no es ISO válida "
            f"({date_str!r}): {e}. Usar YYYY-MM-DD."
        )


def test_marker_date_meets_floor():
    """La fecha del marker debe ser >= `_PFIX_DATE_FLOOR`. Si subes el floor
    sin bumpear `_LAST_KNOWN_PFIX` (o viceversa), este test falla.

    Este es el corazón del enforcement: humanos olvidan bumpear; la fecha
    del marker debe siempre estar al día con el último P-fix cerrado.
    """
    marker = _read_marker_from_app_py()
    m = _MARKER_PATTERN.match(marker)
    assert m is not None, f"Marker mal formado: {marker!r}"
    marker_date = datetime.strptime(m.group("date"), "%Y-%m-%d").date()
    assert marker_date >= _PFIX_DATE_FLOOR, (
        f"`_LAST_KNOWN_PFIX={marker!r}` tiene fecha {marker_date} < "
        f"floor {_PFIX_DATE_FLOOR}. Bumpear el marker en `app.py:24` "
        f"con el último P-fix cerrado, o ajustar `_PFIX_DATE_FLOOR` en "
        f"este test si el floor está desfasado."
    )


def test_marker_prefix_uses_known_pfix_category():
    """El prefix `P<n>` debe estar en {P0..P9}. P10+ no son patrones existentes
    en el repo; un valor fuera de rango sugiere un typo (`PFIX-1`, `Q3-1`, etc.).
    """
    marker = _read_marker_from_app_py()
    m = _MARKER_PATTERN.match(marker)
    assert m is not None, f"Marker mal formado: {marker!r}"
    prefix = m.group("prefix")
    pfix_num_match = re.match(r"^P(\d+)", prefix)
    assert pfix_num_match is not None, f"Prefix sin número: {prefix!r}"
    pfix_num = int(pfix_num_match.group(1))
    assert 0 <= pfix_num <= 9, (
        f"Prefix `{prefix}` con número {pfix_num} fuera del rango P0-P9. "
        f"Si es intencional (creaste P10+), actualizar este test y CLAUDE.md."
    )


def test_inline_comment_references_convention():
    """El comentario sobre `_LAST_KNOWN_PFIX` en `app.py` debe referenciar
    la convención (CLAUDE.md) o este test, para que un futuro mantenedor
    sepa POR QUÉ debe bumpearse.

    Sin este anchor, el comentario podría borrarse en un refactor cosmético
    y el siguiente operador no entendería el contexto.
    """
    text = _APP_PY_PATH.read_text(encoding="utf-8")
    # Buscar bloque de comentarios inmediatamente antes de la asignación.
    block_match = re.search(
        r"((?:^#[^\n]*\n)+)_LAST_KNOWN_PFIX\s*=", text, re.MULTILINE
    )
    assert block_match is not None, (
        "Comentario sobre `_LAST_KNOWN_PFIX` desapareció. Restaurar el bloque "
        "que apunta a la convención (CLAUDE.md) y a este test."
    )
    block = block_match.group(1)
    # Debe mencionar al menos uno: CLAUDE.md, P3-1, o el nombre del test.
    anchors = ("CLAUDE.md", "P3-1", "test_p3_1_last_known_pfix_freshness")
    assert any(a in block for a in anchors), (
        f"Comentario sobre `_LAST_KNOWN_PFIX` no menciona ninguno de "
        f"{anchors}. Sin anchor, un refactor podría borrar la convención."
    )
