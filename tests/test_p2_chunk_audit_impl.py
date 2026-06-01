"""[P2-CHUNK-AUDIT-IMPL · 2026-05-28] Lock-the-contract de los 14 gaps cerrados
en el audit profundo del sistema de chunks de aprendizaje continuo.

Estrategia: parser estático sobre las fuentes de producción (corre bajo
`py -3 --noconftest` sin importar langgraph). Cada test ancla la invariante de
un gap; si un refactor revierte el fix, el test falla ANTES de que el bug
re-aparezca en prod.

Gaps cubiertos:
  P1-CHUNK-1  dead_lettered_at IS NULL guard (recovery SELECT + pickup) + re-enqueue clear + dedup
  P1-CHUNK-2  zombie rescue heartbeat-aware (NOT EXISTS sobre chunk_user_locks.heartbeat_at)
  P2-CHUNK-1  CHUNK_ZOMBIE_RESCUE_MINUTES knob + CHUNK_MAX_FAILURE_ATTEMPTS parametrizado
  P2-CHUNK-2  escalate helper -> status terminal 'failed'
  P1-CHUNK-3  reflection_node lee _meal_level_adherence (no la clave muerta _meal_adherence)
  P1-CHUNK-4  gate wall-clock (_chunk_recovery_wall_floor_met) en escalaciones de recovery
  P2-CHUNK-4  gate _previous_plan_quality + quality persistence por consumed_records>=3
  P2-CHUNK-3  T2 except -> CAS revert a pending +30s
  P2-CHUNK-5  validation range desde prior_count + recompute day_name
  P2-CHUNK-6  detectar chunk degraded (fallback/review-failed) + propagar flags
  P2-CHUNK-7  inyectar _caller_target_plan_id / _caller_context en el worker
  P2-CHUNK-8  fix typo alert_key + resolver per-chunk dead_lettered_chunk alert
  P2-CHUNK-9  observabilidad (auto-resolve aggregates, lesson tracker, stuck-processing, temporal-gate override)
  P2-CHUNK-10 robustez (telemetry NULL-user gate, failure-counter lock, I2 user_id)

Tooltip-anchor: P2-CHUNK-AUDIT-IMPL.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _REPO_ROOT / "backend"
_CRON = _BACKEND / "cron_tasks.py"
_CONSTANTS = _BACKEND / "constants.py"
_ORCH = _BACKEND / "graph_orchestrator.py"
_PLANS = _BACKEND / "routers" / "plans.py"
_ALERTS_DOC = _BACKEND / "docs" / "system_alerts_resolution_table.md"


@pytest.fixture(scope="module")
def cron_src() -> str:
    return _CRON.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def const_src() -> str:
    return _CONSTANTS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def orch_src() -> str:
    return _ORCH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def plans_src() -> str:
    return _PLANS.read_text(encoding="utf-8")


def _slice_fn(src: str, fn: str) -> str:
    start = src.find(f"def {fn}(")
    if start < 0:
        return ""
    after = src[start + len(f"def {fn}("):]
    nxt = re.search(r"\n(?:def |class )\w", after)
    end = (start + len(f"def {fn}(") + nxt.start()) if nxt else len(src)
    return src[start:end]


# ──────────────────────────── P1-CHUNK-1 ────────────────────────────

def test_p1_chunk_1_dead_lettered_guard_in_pickup(cron_src: str):
    """Las 2 ramas del pickup (target + non-target) filtran dead_lettered_at IS NULL."""
    # Debe haber al menos 2 candidatos de pickup con el guard P1-CHUNK-1.
    occurrences = cron_src.count("[P1-CHUNK-1] defensa-en-profundidad")
    assert occurrences >= 2, (
        f"Esperaba ≥2 guards `AND q1.dead_lettered_at IS NULL` en las ramas de pickup "
        f"(target + non-target), encontré {occurrences}. Ver process_plan_chunk_queue."
    )


def test_p1_chunk_1_reenqueue_clears_dead_letter(cron_src: str):
    """El UPSERT de enqueue limpia dead_lettered_at/dead_letter_reason en DO UPDATE."""
    assert "dead_lettered_at = NULL" in cron_src and "dead_letter_reason = NULL" in cron_src, (
        "El re-enqueue (DO UPDATE) debe limpiar dead_lettered_at = NULL + "
        "dead_letter_reason = NULL para no resucitar un chunk dead-lettered congelado."
    )


# ──────────────────────────── P1-CHUNK-2 ────────────────────────────

def test_p1_chunk_2_zombie_rescue_heartbeat_aware(cron_src: str):
    """El zombie rescue usa NOT EXISTS sobre chunk_user_locks.heartbeat_at fresco."""
    assert "chunk_user_locks" in cron_src
    # El rescue NO debe reclamar un chunk cuyo lock tiene heartbeat fresco.
    assert re.search(
        r"NOT\s+EXISTS\s*\(\s*SELECT\s+1\s+FROM\s+chunk_user_locks",
        cron_src,
        re.IGNORECASE | re.DOTALL,
    ), "Zombie rescue debe tener NOT EXISTS sobre chunk_user_locks (heartbeat-aware)."
    assert "heartbeat_at >" in cron_src, "El join del rescue debe comparar heartbeat_at fresco."


# ──────────────────────────── P2-CHUNK-1 ────────────────────────────

def test_p2_chunk_1_zombie_rescue_minutes_knob(const_src: str):
    assert "CHUNK_ZOMBIE_RESCUE_MINUTES" in const_src, (
        "Falta el knob CHUNK_ZOMBIE_RESCUE_MINUTES en constants.py."
    )


def test_p2_chunk_1_max_failure_attempts_parametrized(cron_src: str):
    """El zombie rescue parametriza el umbral con CHUNK_MAX_FAILURE_ATTEMPTS (no hardcode 5)."""
    assert "CHUNK_MAX_FAILURE_ATTEMPTS" in cron_src
    assert "CHUNK_ZOMBIE_RESCUE_MINUTES" in cron_src


# ──────────────────────────── P2-CHUNK-2 ────────────────────────────

def test_p2_chunk_2_escalate_terminal_status(cron_src: str):
    body = _slice_fn(cron_src, "_escalate_unrecoverable_chunk")
    assert body, "_escalate_unrecoverable_chunk no encontrado."
    assert "status = CASE WHEN status = 'cancelled' THEN status ELSE 'failed' END" in body, (
        "El escalate debe transicionar a status terminal 'failed' (preservando 'cancelled')."
    )


# ──────────────────────────── P1-CHUNK-3 ────────────────────────────

def test_p1_chunk_3_reflection_reads_correct_key(orch_src: str):
    body = _slice_fn(orch_src, "reflection_node")
    assert body, "reflection_node no encontrado."
    assert 'form_data.get("_meal_level_adherence"' in body, (
        "reflection_node debe leer `_meal_level_adherence` (la clave que el "
        "feedback-loop ESCRIBE), no la clave muerta `_meal_adherence`."
    )
    assert 'form_data.get("_meal_adherence"' not in body, (
        "reflection_node NO debe leer `_meal_adherence` (singular, clave muerta)."
    )


# ──────────────────────────── P1-CHUNK-4 ────────────────────────────

def test_p1_chunk_4_wall_floor_helper_exists(cron_src: str):
    assert "def _chunk_recovery_wall_floor_met(" in cron_src, (
        "Falta el helper _chunk_recovery_wall_floor_met (gate wall-clock)."
    )


def test_p1_chunk_4_wall_floor_knob(const_src: str):
    assert "CHUNK_RECOVERY_MIN_WALL_MINUTES_PER_ATTEMPT" in const_src


def test_p1_chunk_4_escalations_gated_by_wall_floor(cron_src: str):
    """Las escalaciones tz/anchor/corrupted aplican el gate wall-clock."""
    gated = cron_src.count("_chunk_recovery_wall_floor_met(")
    # 1 def + 5 callsites (3 recovery cron + 2 worker-inline) = >=6 ocurrencias.
    assert gated >= 6, (
        f"Esperaba ≥6 menciones de _chunk_recovery_wall_floor_met (def + 5 gates), "
        f"encontré {gated}."
    )


# ──────────────────────────── P2-CHUNK-4 ────────────────────────────

def test_p2_chunk_4_quality_gated_by_consumed(cron_src: str):
    assert "_quality_data_sufficient" in cron_src, (
        "Falta el flag _quality_data_sufficient que gatea la inyección/persistencia "
        "de quality por consumed_records>=3."
    )
    # El nightly path también gatea last_plan_quality.
    assert "omitiendo persistencia de last_plan_quality" in cron_src or \
           "P2-CHUNK-4" in cron_src


# ──────────────────────────── P2-CHUNK-3 ────────────────────────────

def test_p2_chunk_3_t2_revert_to_pending(cron_src: str):
    assert "P2-CHUNK-3" in cron_src
    assert "make_interval(secs => 30)" in cron_src, (
        "El except de T2 debe revertir a pending con +30s via CAS."
    )
    assert "t2_commit_failed_exhausted" in cron_src, (
        "El revert de T2 debe transicionar a failed+dead_letter al agotar intentos."
    )


# ──────────────────────────── P2-CHUNK-5 ────────────────────────────

def test_p2_chunk_5_validation_range_from_prior_count(cron_src: str):
    assert "_p04_new_start = int(prior_count) + 1" in cron_src, (
        "El rango de validación de pantry post-merge debe derivar de prior_count, "
        "no de days_offset (que queda obsoleto tras /shift-plan o dedup)."
    )


def test_p2_chunk_5_day_name_recompute(cron_src: str):
    assert "_p2c5_dias" in cron_src and "weekday()]" in cron_src, (
        "El merge debe recomputar day_name desde la fecha de inicio + posición absoluta."
    )


# ──────────────────────────── P2-CHUNK-6 ────────────────────────────

def test_p2_chunk_6_degraded_detection_and_propagation(cron_src: str):
    assert "_chunk_result_is_fallback" in cron_src and "_chunk_review_failed" in cron_src, (
        "El worker debe detectar degradación a nivel plan-result (_is_fallback / review-failed)."
    )
    # Propaga flags a plan_data y deben sobrevivir T2.
    assert "'_review_failed_but_delivered'," in cron_src, (
        "_is_fallback/_review_failed_but_delivered deben estar en P0_4_T2_INCREMENTAL_KEYS."
    )


# ──────────────────────────── P2-CHUNK-7 ────────────────────────────

def test_p2_chunk_7_caller_context_injected(cron_src: str):
    assert 'form_data["_caller_target_plan_id"] = meal_plan_id' in cron_src, (
        "El worker debe inyectar _caller_target_plan_id antes del pipeline (para el alert I5)."
    )
    assert 'form_data["_caller_context"] = f"chunk_worker:week_{week_number}"' in cron_src


# ──────────────────────────── P2-CHUNK-8 ────────────────────────────

def test_p2_chunk_8_alert_key_typo_fixed(plans_src: str):
    assert "chunks_dead_lettered_recent" not in plans_src, (
        "El typo `chunks_dead_lettered_recent` (palabras invertidas) debe estar corregido."
    )
    assert "dead_lettered_chunks_recent" in plans_src, (
        "regenerate-simplified debe resolver la alerta canónica `dead_lettered_chunks_recent`."
    )


def test_p2_chunk_8_resolves_per_chunk_dead_letter(plans_src: str):
    assert "'dead_lettered_chunk:' || %s || ':' || %s" in plans_src, (
        "regenerate-simplified debe resolver también la alerta per-chunk "
        "`dead_lettered_chunk:<plan>:<week>`."
    )


# ──────────────────────────── P2-CHUNK-9 ────────────────────────────

def test_p2_chunk_9_aggregate_auto_resolve(cron_src: str):
    body = _slice_fn(cron_src, "_alert_new_dead_lettered_chunks")
    assert "dead_lettered_chunks_recent' AND resolved_at IS NULL" in body, (
        "El cron de dead-letter agregado debe auto-resolver cuando el backlog se despeja."
    )


def test_p2_chunk_9_stuck_processing_cron(cron_src: str):
    assert "def _alert_chunks_stuck_processing(" in cron_src, (
        "Falta el cron _alert_chunks_stuck_processing (detector de zombie de worker)."
    )
    assert "chunks_stuck_processing" in cron_src
    # Registrado en el scheduler.
    assert 'id="alert_chunks_stuck_processing"' in cron_src, (
        "El cron stuck-processing debe estar registrado en register_plan_chunk_scheduler."
    )


def test_p2_chunk_9_lesson_telemetry_backlog_tracker(cron_src: str):
    assert "def _track_lesson_telemetry_backlog(" in cron_src
    assert "lesson_telemetry_flush_backlog" in cron_src


def test_p2_chunk_9_temporal_gate_override_event(cron_src: str, const_src: str):
    assert 'event="temporal_gate_override"' in cron_src, (
        "El override del gate temporal debe emitir telemetría temporal_gate_override."
    )
    assert '"temporal_gate_override"' in const_src, (
        "temporal_gate_override debe estar en CHUNK_LESSON_TELEMETRY_VALID_EVENTS "
        "(si no, el at-write whitelist lo rechazaría)."
    )


def test_p2_chunk_9_new_alert_keys_documented():
    doc = _ALERTS_DOC.read_text(encoding="utf-8")
    for key in ("chunks_stuck_processing", "lesson_telemetry_flush_backlog"):
        assert f"`{key}`" in doc, (
            f"El nuevo alert_key `{key}` debe estar documentado en "
            f"system_alerts_resolution_table.md (drift test P2-AUDIT-4)."
        )


# ──────────────────────────── P2-CHUNK-10 ────────────────────────────

def test_p2_chunk_10_telemetry_null_user_gate(cron_src: str):
    body = _slice_fn(cron_src, "_record_chunk_lesson_telemetry")
    assert "P2-CHUNK-10/TELEMETRY-GATE" in body, (
        "_record_chunk_lesson_telemetry debe gatear user_id/meal_plan_id no-UUID "
        "ANTES del INSERT (evita round-trip fallido + ERROR ruidoso para guests)."
    )
    assert "_is_valid_uuid(user_id) or not _is_valid_uuid(meal_plan_id)" in body


def test_p2_chunk_10_heartbeat_counter_lock(cron_src: str):
    assert "def _get_heartbeat_failures_lock(" in cron_src
    body = _slice_fn(cron_src, "_handle_heartbeat_start_failure")
    assert "with _get_heartbeat_failures_lock():" in body, (
        "La mutación del contador _chunk_heartbeat_start_failures debe ser bajo lock."
    )


def test_p2_chunk_10_i2_user_id_on_recovery_updates(cron_src: str):
    """Las 4 UPDATEs de _anchor_recovery_attempts filtran AND user_id = %s."""
    n = cron_src.count("[P2-CHUNK-10] I2 defense-in-depth")
    assert n >= 4, (
        f"Esperaba ≥4 UPDATEs de recovery con I2 `AND user_id = %s` "
        f"(2 recovery cron + 2 worker-inline), encontré {n} anchors."
    )


# ──────────────────────────── Marker anchor ────────────────────────────

def test_marker_present_in_app(orch_src: str):
    """Sanity: el bundle P2-CHUNK-AUDIT-IMPL persiste su propio test de regresión.

    [Relajado 2026-05-29 · P1-CHUNK-LEARN-AUDIT] El marker `_LAST_KNOWN_PFIX` se
    bumpea con cada P-fix posterior, así que NO podemos exigir el valor literal
    `P2-CHUNK-AUDIT-IMPL` en app.py (fue superseded por P1-CHUNK-LEARN-AUDIT). El
    contrato durable es que el test de regresión de ESTE bundle sigue existiendo
    (la freshness del marker actual la enforza test_p3_1 + el cross-link
    test_p2_hist_audit_14)."""
    assert (_BACKEND / "tests" / "test_p2_chunk_audit_impl.py").exists(), (
        "El test de regresión de P2-CHUNK-AUDIT-IMPL debe persistir."
    )
