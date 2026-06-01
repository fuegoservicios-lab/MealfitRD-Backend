"""[P1-CHUNK-LEARN-AUDIT · 2026-05-29] Lock-the-contract de las 19 gaps cerradas
en el audit profundo del sistema de chunks de aprendizaje continuo (sigue a
P2-CHUNK-AUDIT-IMPL · 2026-05-28).

Tema sistémico dominante: **drift productor↔consumidor de claves** (la misma clase
de bug que P1-CHUNK-3 cerró UNA vez). El audit encontró ≥4 instancias más, todas
invisibles a los 991 tests porque los fixtures usaban la clave del consumer,
enmascarando el drift del producer. Estos tests anclan cada fix con un parser
estático sobre la fuente de prod (corre bajo `py -3 --noconftest` sin langgraph).

Gaps cubiertas (cluster · id):
  A1 stripped-nudge-tone-learning-signals        (graph_orchestrator._TRUSTED_INTERNAL_FORM_KEYS)
  A2 pause-reason-key-mismatch-anchor            (cron_tasks consumer lee ambas claves + producer dual-write)
  A3 pause-reason-key-mismatch-collateral        (3 producers dual-write _pantry_pause_reason)
  A4 quality-history-dead-key-gate-fallback      (graph_orchestrator lee quality_history_chunks)
  A5 quality-history-dead-key-learning-hint-ui   (routers/plans lee quality_history_chunks)
  A6 quality-history-dead-key-admin-avg-metric   (routers/system SQL quality_history_chunks)
  A7 chunk-header-number-label-mismatch          (prompts/plan_generator lee chunk_numbers)
  B1 stuck-terminal-failed-no-banner-no-deadletter (terminal escala via _escalate_unrecoverable_chunk)
  B2 sweep-omits-pua-failed                       (NOT EXISTS incluye pending_user_action + dead_lettered_at)
  C1 pause-status-transition-no-cas               (helper CAS + 3 callsites)
  C2 lock-insert-exception-fallthrough            (except lock_err → defer, no fallthrough)
  C3 lock-release-not-ownership-aware             (DELETE guardado por locked_at)
  D1 i2-missing-recovery-exhausted-chunks-update  (AND user_id en UPDATE)
  D2 i2-missing-recent-chunk-lessons-unblock      (AND user_id en UPDATE)
  E1 synth-overload-alert-never-resolves          (auto-resolver chunk_synthesis_overload)
  E2 degraded-rate-high-never-auto-resolves       (rama else resolver)
  E3 dual-processing-critical-no-liveness-tick     (tick pipeline_metrics)
  E4 indefinite-pause-unblock-telemetry-lost       (no literal "unknown")
  F1 gap-e-dedup-claim-no-op                       (dedup real última-ocurrencia)

Tooltip-anchor: P1-CHUNK-LEARN-AUDIT.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _REPO_ROOT / "backend"
_CRON = _BACKEND / "cron_tasks.py"
_ORCH = _BACKEND / "graph_orchestrator.py"
_PLANS = _BACKEND / "routers" / "plans.py"
_SYSTEM = _BACKEND / "routers" / "system.py"
_PLAN_GEN = _BACKEND / "prompts" / "plan_generator.py"
_APP = _BACKEND / "app.py"


@pytest.fixture(scope="module")
def cron_src() -> str:
    return _CRON.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def orch_src() -> str:
    return _ORCH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def plans_src() -> str:
    return _PLANS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def system_src() -> str:
    return _SYSTEM.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def plan_gen_src() -> str:
    return _PLAN_GEN.read_text(encoding="utf-8")


def _slice_fn(src: str, def_signature: str, max_lines: int = 4000) -> str:
    """Devuelve el cuerpo de una función desde `def_signature` hasta el siguiente
    `def ` al mismo nivel de indentación (o EOF)."""
    start = src.find(def_signature)
    assert start != -1, f"No encuentro `{def_signature}`"
    # indentación de la def
    line_start = src.rfind("\n", 0, start) + 1
    indent = start - line_start
    rest = src[start + len(def_signature):]
    # buscar siguiente `def ` con indentación <= la de esta def
    lines = rest.splitlines(keepends=True)
    body = [def_signature]
    for ln in lines:
        stripped = ln.lstrip()
        cur_indent = len(ln) - len(stripped)
        if stripped.startswith(("def ", "async def ")) and cur_indent <= indent:
            break
        body.append(ln)
        if len(body) > max_lines:
            break
    return "".join(body)


# ===========================================================================
# Cluster A — key-drift / pérdida silenciosa de aprendizaje
# ===========================================================================
def test_a1_strip_whitelist_includes_nudge_and_tone(orch_src):
    """A1: las 2 señales backend-only deben estar en _TRUSTED_INTERNAL_FORM_KEYS,
    o el strip las borra antes de llegar al prompt (escritas y leídas, pero
    eliminadas en medio — clase P1-CHUNK-3)."""
    m = re.search(r"_TRUSTED_INTERNAL_FORM_KEYS[^=\n]*=\s*frozenset\(", orch_src)
    assert m is not None, "No encuentro _TRUSTED_INTERNAL_FORM_KEYS = frozenset(...)"
    block = orch_src[m.start(): m.start() + 8000]
    # cortar en el cierre `})` del frozenset
    close = block.find("})")
    assert close != -1
    block = block[:close]
    assert '"_nudge_conversion_rates"' in block, (
        "A1: `_nudge_conversion_rates` falta en el whitelist → el strip la borra."
    )
    assert '"_successful_tone_strategies"' in block, (
        "A1: `_successful_tone_strategies` falta en el whitelist → el strip la borra."
    )


def test_a2_consumer_reads_both_pause_reason_keys(cron_src):
    """A2: el recovery cron debe tolerar ambas claves del snapshot."""
    assert 'snap.get("_pantry_pause_reason") or snap.get("_pause_reason")' in cron_src, (
        "A2: el consumer (_recover_pantry_paused_chunks) debe leer "
        "`_pantry_pause_reason or _pause_reason` para no degradar reasons a empty_pantry."
    )


def test_a2_anchor_producer_dual_writes(cron_src):
    """A2: el productor de missing_start_date_no_anchor debe escribir ambas claves.
    (El reason es único, así que la presencia del assignment a `_pantry_pause_reason`
    con ese valor prueba el dual-write — antes solo existía el `_pause_reason`.)"""
    assert '"_pause_reason"] = "missing_start_date_no_anchor"' in cron_src
    assert '"_pantry_pause_reason"] = "missing_start_date_no_anchor"' in cron_src, (
        "A2: el productor missing_start_date_no_anchor debe dual-escribir "
        "`_pantry_pause_reason` (la rama dedicada del recovery cron lo lee)."
    )


@pytest.mark.parametrize("reason", [
    "learning_proxy_exhausted",
    "all_prior_days_blocked_by_restrictions",
    "pantry_violation_post_merge",
])
def test_a3_collateral_producers_dual_write(cron_src, reason):
    """A3: los 3 productores colaterales dual-escriben _pantry_pause_reason."""
    assert f'"_pause_reason"] = "{reason}"' in cron_src, f"No encuentro el productor de {reason}"
    assert f'"_pantry_pause_reason"] = "{reason}"' in cron_src, (
        f"A3: el productor de {reason} debe dual-escribir `_pantry_pause_reason`."
    )


def test_a4_gate_reads_quality_history_chunks(orch_src):
    """A4: el gate de confianza lee la clave real `quality_history_chunks`."""
    assert 'actual_form_data.get("quality_history_chunks")' in orch_src, (
        "A4: el gate de Señales 7-9 debe leer `quality_history_chunks` (la clave "
        "real), no solo la muerta `quality_history`."
    )


def test_a5_ui_hint_reads_quality_history_chunks(plans_src):
    """A5: el hint de la UI lee la clave real."""
    assert 'hp.get("quality_history_chunks")' in plans_src, (
        "A5: last_learning_hint debe leer `quality_history_chunks` (floats 0-1)."
    )


def test_a6_admin_metric_reads_quality_history_chunks(system_src):
    """A6: la métrica admin consulta la clave real en el JSONB."""
    assert "quality_history_chunks" in system_src, (
        "A6: average_quality_score debe consultar `health_profile->>'quality_history_chunks'`."
    )
    # La query vieja (dead key sin sufijo, en el WHERE) no debe quedar como
    # único filtro. Verificamos que el sufijo _chunks esté presente en el SQL.
    assert "health_profile->>'quality_history_chunks'" in system_src


def test_a7_chunk_header_reads_plural_key(plan_gen_src):
    """A7: el builder del header lee la clave plural `chunk_numbers`."""
    assert 'chunk_lessons.get("chunk_numbers")' in plan_gen_src, (
        "A7: el header de lecciones debe derivar el label de `chunk_numbers` "
        "(la clave plural que el worker escribe), no de `chunk_number`."
    )


# ===========================================================================
# Cluster B — recovery / estado terminal
# ===========================================================================
def test_b1_stuck_terminal_uses_escalate_helper(cron_src):
    """B1: el path terminal de _detect_and_escalate_stuck_chunks escala via el
    helper canónico (dead_lettered_at + banner + alert), no un UPDATE crudo."""
    body = _slice_fn(cron_src, "def _detect_and_escalate_stuck_chunks():")
    assert "_escalate_unrecoverable_chunk(" in body, (
        "B1: el path terminal debe invocar `_escalate_unrecoverable_chunk` "
        "(setea dead_lettered_at + banner _user_action_required + alert)."
    )
    # Anti-regresión: ya no debe quedar el UPDATE masivo bulk a 'failed' por ANY(uuid[]).
    assert "SET status = 'failed', updated_at = NOW() WHERE id = ANY(%s::uuid[])" not in body, (
        "B1: el UPDATE masivo `status='failed'` sin dead-letter debe haberse reemplazado."
    )


def test_b2_sweep_alive_predicate_includes_pua_and_failed(cron_src):
    """B2: el NOT EXISTS del sweep huérfanos cuenta pending_user_action y
    failed-no-dead-lettered como 'vivo' (paridad con _finalize_zombie_partial_plans)."""
    body = _slice_fn(cron_src, "def _sweep_meal_plans_without_chunks() -> int:")
    assert "pending_user_action" in body, (
        "B2: el sweep debe tratar `pending_user_action` como chunk vivo."
    )
    assert "dead_lettered_at IS NULL" in body, (
        "B2: el sweep debe tratar `failed AND dead_lettered_at IS NULL` como vivo."
    )


# ===========================================================================
# Cluster C — concurrencia
# ===========================================================================
def test_c1_cas_pause_helper_exists_and_guards(cron_src):
    """C1: el helper CAS existe y guarda por attempts + status='processing'."""
    body = _slice_fn(cron_src, "def _cas_pause_chunk_to_pending_user_action(")
    assert "AND attempts = %s AND status = 'processing'" in body, (
        "C1: el CAS debe guardar la transición con `AND attempts=%s AND status='processing'`."
    )
    assert "RETURNING id" in body


@pytest.mark.parametrize("helper_def", [
    "def _pause_chunk_for_pantry_refresh(",
    "def _pause_chunk_for_final_inventory_validation(",
    "def _pause_chunk_for_synthesis_overload(",
])
def test_c1_pause_helpers_use_cas(cron_src, helper_def):
    """C1: los 3 helpers de pausa van por el CAS ownership-aware."""
    body = _slice_fn(cron_src, helper_def)
    assert "_cas_pause_chunk_to_pending_user_action(" in body, (
        f"C1: {helper_def} debe usar `_cas_pause_chunk_to_pending_user_action` "
        f"en lugar del UPDATE bare a 'pending_user_action'."
    )


def test_c2_lock_exception_defers_not_fallthrough(cron_src):
    """C2: el except del lock difiere (no cae a generación sin lock)."""
    idx = cron_src.find("except Exception as lock_err:")
    assert idx != -1
    window = cron_src[idx: idx + 1800]
    assert "_handle_heartbeat_start_failure(task_id, user_id)" in window, (
        "C2: el except del lock debe diferir vía `_handle_heartbeat_start_failure`."
    )
    # Debe haber un `return` antes del siguiente `try:` de generación.
    hb_pos = window.find("_handle_heartbeat_start_failure(task_id, user_id)")
    next_try = window.find("\n        try:", hb_pos)
    assert next_try != -1, "C2: no encuentro el `try:` de generación tras el except."
    assert "return" in window[hb_pos:next_try], (
        "C2: tras el defer debe haber `return` — no caer al bloque de generación."
    )


def test_c3_lock_release_is_ownership_aware(cron_src):
    """C3: el DELETE del lock se guarda por locked_at (no borra el lock de otro worker)."""
    assert "DELETE FROM chunk_user_locks WHERE locked_by_chunk_id = %s AND locked_at = %s" in cron_src, (
        "C3: el DELETE del lock en el finally debe guardar por `locked_at` "
        "para no borrar el lock del worker que nos desplazó."
    )
    # Y el INSERT debe capturar locked_at.
    assert "RETURNING user_id, locked_at;" in cron_src, (
        "C3: el INSERT del lock debe `RETURNING ... locked_at` para capturar nuestra fila."
    )


# ===========================================================================
# Cluster D — invariante I2 (filtro user_id)
# ===========================================================================
def test_d1_recovery_exhausted_update_has_user_id(cron_src):
    """D1: el UPDATE de _recovery_exhausted_chunks filtra por user_id."""
    body = _slice_fn(cron_src, "def _escalate_unrecoverable_chunk(")
    assert "WHERE id = %s AND user_id = %s" in body, (
        "D1: el UPDATE de _recovery_exhausted_chunks / _user_action_required debe "
        "incluir `AND user_id = %s` (I2 sobre campo learning-crítico)."
    )


def test_d2_recent_chunk_lessons_update_has_user_id(cron_src):
    """D2: el UPDATE de _recent_chunk_lessons (auto-unblock) filtra por user_id."""
    # Localizar el jsonb_set de _recent_chunk_lessons y verificar el WHERE cercano.
    idx = cron_src.find("'{_recent_chunk_lessons}'")
    assert idx != -1
    window = cron_src[idx: idx + 600]
    assert "WHERE id = %s AND user_id = %s" in window, (
        "D2: el UPDATE de _recent_chunk_lessons debe incluir `AND user_id = %s`."
    )


# ===========================================================================
# Cluster E — observabilidad / auto-resolución de alertas
# ===========================================================================
def test_e1_synth_overload_alert_has_resolver(cron_src):
    """E1: chunk_synthesis_overload tiene auto-resolver (modelo Auto explicit)."""
    body = _slice_fn(cron_src, "def _alert_high_synthesized_lesson_ratio() -> None:")
    assert "chunk_synthesis_overload:%" in body, (
        "E1: debe existir un resolver `UPDATE system_alerts ... LIKE 'chunk_synthesis_overload:%'`."
    )
    assert "resolved_at = NOW()" in body


def test_e2_degraded_rate_has_resolver_branch(cron_src):
    """E2: degraded_rate_high se auto-resuelve cuando el ratio cae bajo umbral."""
    body = _slice_fn(cron_src, "def _alert_if_degraded_rate_high():")
    assert "E2-DEGRADED-RESOLVE" in body, "E2: falta la rama resolver."
    assert "degraded_rate_high:" in body
    assert "resolved_at = NOW()" in body


def test_e3_dual_processing_emits_liveness_tick(cron_src):
    """E3: el detector critical de doble-procesamiento emite tick de liveness."""
    body = _slice_fn(cron_src, "def _alert_chunk_dual_processing() -> None:")
    assert "_alert_chunk_dual_processing_tick" in body, (
        "E3: debe emitir `_alert_chunk_dual_processing_tick` en pipeline_metrics "
        "SIEMPRE (incl. caso sano) para distinguir cron vivo de cron muerto."
    )
    # El tick debe emitirse ANTES del early-return `if not rows: return`.
    tick_pos = body.find("_alert_chunk_dual_processing_tick")
    notrows_pos = body.find("if not rows:")
    assert tick_pos != -1 and notrows_pos != -1 and tick_pos < notrows_pos, (
        "E3: el tick debe emitirse antes del early-return del caso sano."
    )


def test_e4_unblock_telemetry_no_unknown_literal(cron_src):
    """E4: la telemetría de unblock ya no pasa el literal 'unknown' (rechazado por el gate UUID)."""
    # Localizar el bloque del evento indefinite_pause_unblocked.
    idx = cron_src.find('event="indefinite_pause_unblocked"')
    assert idx != -1
    window = cron_src[idx - 400: idx + 200]
    assert 'if user_id else "unknown"' not in window, (
        "E4: no pasar el literal 'unknown' al recorder (el gate UUID lo rechaza); "
        "omitir limpio si falta user_id."
    )


# ===========================================================================
# Cluster F — robustez del worker
# ===========================================================================
def test_f1_gap_e_dedup_is_real(cron_src):
    """F1: el bloque GAP-E deduplica de verdad (no solo loguea)."""
    assert "F1-GAP-E-DEDUP" in cron_src, "F1: falta el anchor del fix."
    assert "_pos_by_original_day" in cron_src, (
        "F1: el dedup real debe colapsar por `_pos_by_original_day` (última ocurrencia gana)."
    )
    # Anti-regresión: el log viejo que afirmaba dedup sin hacerlo no debe quedar.
    assert "Deduplicando (última ocurrencia gana)." not in cron_src, (
        "F1: el log viejo (que mentía sobre la dedup) debe haberse corregido."
    )


# ===========================================================================
# Marker anchor
# ===========================================================================
def test_last_known_pfix_marker_bumped():
    """El marker de _LAST_KNOWN_PFIX debe reflejar este bundle O un sucesor.

    [Relajado NG-CRON-OPT-2 · 2026-05-30] La aserción de familia `P1-CHUNK-LEARN`
    se relajó a un FLOOR DE FECHA: el marker migró a sucesores fuera de familia
    (P2-CRON-OPT · 2026-05-29, NG-CRON-OPT-2 · 2026-05-30, …) y la aserción de familia
    rompía cada bump legítimo posterior. La freshness real la enforza
    `test_p3_1_last_known_pfix_freshness`; aquí solo exigimos no-stale (≥ fecha sesión)."""
    import re as _re
    from datetime import date as _date
    text = _APP.read_text(encoding="utf-8")
    m = _re.search(r'_LAST_KNOWN_PFIX\s*=\s*"[^"]*·\s*(\d{4})-(\d{2})-(\d{2})"', text)
    assert m, 'No se encontró marker `_LAST_KNOWN_PFIX = "... · YYYY-MM-DD"` válido.'
    marker_date = _date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    assert marker_date >= _date(2026, 5, 29), (
        f"Marker stale: {marker_date} < 2026-05-29 (fecha del audit P1-CHUNK-LEARN)."
    )
