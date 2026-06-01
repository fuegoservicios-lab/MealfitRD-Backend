"""[P2-CRON-OPT-4 · 2026-05-31] Anclas de la CUARTA pasada de optimización de
cron_tasks.py.

Re-audit multi-agente fresco (workflow, 38 agentes, 28 candidatos → 18 reales por
verificación adversaria + crítico de completitud; cada hallazgo re-verificado a mano
contra el código de prod ANTES de implementar, según la lección del repo "el veredicto
del workflow sobreestima"). Las 3 pasadas previas (P2-CRON-OPT-AUDIT 82, P2-CRON-OPT-2 86,
P2-CRON-OPT-3 89) habían cosechado lo monolith-free y DIFERIDO 3 veces el backlog del
monolito (process_plan_chunk_queue/_chunk_worker + _background_shift_plan_for_user). Esta
pasada finalmente ataca el subconjunto SEGURO de ese backlog (adiciones puras, read-only,
dead-code, anotaciones) + gaps nuevos, dejando explícitamente lo race-critical diferido.

Cada test ancla un fix concreto sobre el SOURCE de prod (parser-based, sin import/DB —
el entorno de test es DB-less: cron_tasks no importa por falta de psycopg/supabase) para
que un refactor que revierta el fix falle ANTES de degradar prod.

Gaps implementados (12):
  G-A1 (P2) _detect_chronic_deferrals: dedupe SELECT `alert_key = ANY(%s)` sin
        `AND resolved_at IS NULL` → tras el auto-resolve de G11 (P2-CRON-OPT-3) las filas
        resolved-but-recent suprimían re-notificación + re-open. Misma clase NG-2.
  G-A2 (P3) _check_chunk_learning_ready: dedupe SELECT de temporal_gate_proactive sin
        `AND resolved_at IS NULL` → misma clase NG-2 vs el resolver G12.
  G-A3 (P2) _chunk_worker failure-path: persist de learning_metrics sin CAS
        (`WHERE id=%s` a secas) → asimétrico con el commit T2 success (G4-T2-SUCCESS-CAS)
        y la escalation (P1-NEW-2); ventana lost-update vs zombie-rescue. Añadido
        `AND attempts=%s AND status='processing'` + RETURNING + log CAS-DISPLACED.
  G-A4 (P2) _background_shift_plan_for_user: rama pantry-pause re-escribía plan_data +
        re-enviaba push "Renovación pausada" CADA corrida diaria (target = usuarios
        inactivos → spam diario). Guard de idempotencia sobre 'expired_pending_pantry'.
  G-B1 (P2) inject_learning_signals_from_profile: fetch UN superset de consumed_meals
        reusado vía consumed= por los 3 scorers + cold-start (antes 4 round-trips).
        Espeja S09-2/GAP-3 del cron path; include_ingredients=True obligatorio.
  G-B2 (P3) _recover_pantry_paused_chunks: days_offset al batch SELECT → elimina 2
        re-queries por-fila al MISMO row (cron ~1 min). Columna inmutable en pending_user_action.
  G-B3 (P3) _inject_advanced_learning_signals: gate _quality_data_sufficient hoisteado
        ARRIBA del calculate_plan_quality_score → en cold-start (consumed<3) se evitan 3
        round-trips DB cuyo score se descartaba como falso-negativo.
  G-B4 (P3) _cleanup_orphan_chunks: `meal_plan_id::text NOT IN (SELECT id::text ...)` →
        anti-join `NOT EXISTS (... m.id = q.meal_plan_id)` (ambas uuid → PK index, null-safe).
  G-C1 (P2) _chunk_worker LLM branch: reusa chunk_consumed_records para la fatiga GAP-F
        SOLO cuando days_offset>=14 (su ventana cubre los 14d) — preserva comportamiento
        para chunks tempranos (mismatch de ventana, no dedup ciego).
  G-C2 (P2) _chunk_worker Smart-Shuffle: _get_pantry_tolerance_for_user hoisteado fuera
        del `for _shuffle_idx` (antes 1 SELECT a user_profiles por día generado).
  G-C3 (P3) _chunk_worker: dead-write per-día `_nd["_shuffle_validation_failed"]`
        eliminado (0 readers; el counter gobierna el abort) + comentario heartbeat
        ficticio corregido (el main flow NO lee last_heartbeat_at).
  G-D  (P3) I2-EXEMPT: 4 UPDATE plan_chunk_queue keyados por meal_plan_id anotados como
        trusted cron context (cerraba el re-flag perpetuo G7/G8/G9 de cada auditoría).

Detalle: ~/.claude/projects/.../memory/project_p2_cron_opt_4_2026_05_31.md
Tooltip-anchor: P2-CRON-OPT-4.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_PY = _BACKEND_ROOT / "cron_tasks.py"
_APP_PY = _BACKEND_ROOT / "app.py"


def _src() -> str:
    return _CRON_PY.read_text(encoding="utf-8")


def _slice_fn(src: str, header: str) -> str:
    """Cuerpo de la función `header` hasta el siguiente def top-level."""
    start = src.find(header)
    assert start >= 0, f"no se encontró {header!r}"
    after = src[start + len(header):]
    nxt = re.search(r"\n(?:def |async def )\w", after)
    return after[: nxt.start()] if nxt else after


def _code_only(src: str) -> str:
    """Quita el contenido de comentarios (de `#` a fin de línea) para que las aserciones
    NEGATIVAS no matcheen el propio comentario explicativo del fix — lección repetida del
    repo ("mis comentarios rompieron mis asserts negativos"). Las líneas SQL ancladas en
    positivo no llevan `#`, así que sobreviven intactas."""
    out = []
    for line in src.splitlines():
        h = line.find("#")
        out.append(line if h < 0 else line[:h])
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Cross-link: el marker pertenece a la familia P2-CRON-OPT y no retrocede.
# ---------------------------------------------------------------------------
def test_marker_is_p2_cron_opt_family():
    # Relajado a floor-de-fecha (igual que cron_opt_3): otras sesiones pueden bumpear el
    # marker hacia adelante. La freshness real la enforza test_p3_1_last_known_pfix_freshness;
    # aquí: no retroceder antes de 2026-05-31 (esta pasada).
    app = _APP_PY.read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"[^"]*(\d{4})-(\d{2})-(\d{2})"', app)
    assert m, "No se encontró _LAST_KNOWN_PFIX con fecha parseable."
    assert (int(m.group(1)), int(m.group(2)), int(m.group(3))) >= (2026, 5, 31), (
        "El marker _LAST_KNOWN_PFIX retrocedió antes de 2026-05-31 (P2-CRON-OPT-4)."
    )


# ---------------------------------------------------------------------------
# G-A1: chronic_deferrals dedupe SELECT lleva resolved_at IS NULL
# ---------------------------------------------------------------------------
def test_g_a1_chronic_deferrals_dedupe_resolved_at_null():
    body = _slice_fn(_src(), "def _detect_chronic_deferrals(")
    # El bulk dedupe usa `alert_key = ANY(%s)`; tras él debe venir resolved_at IS NULL.
    idx = body.find("alert_key = ANY(%s)")
    assert idx >= 0, "G-A1: no se encontró el bulk dedupe SELECT `alert_key = ANY(%s)`."
    window = body[idx: idx + 300]
    assert "resolved_at IS NULL" in window, (
        "G-A1: el dedupe SELECT de chronic_deferrals debe filtrar `AND resolved_at IS NULL` "
        "(si no, las filas resolved-but-recent del resolver G11 suprimen re-notificación)."
    )


# ---------------------------------------------------------------------------
# G-A2: temporal_gate_proactive dedupe SELECT lleva resolved_at IS NULL
# ---------------------------------------------------------------------------
def test_g_a2_temporal_gate_dedupe_resolved_at_null():
    body = _slice_fn(_src(), "def _check_chunk_learning_ready(")
    idx = body.find("_p13_existing = execute_sql_query(")
    assert idx >= 0, "G-A2: no se encontró el dedupe SELECT del push temporal_gate."
    window = body[idx: idx + 400]
    assert "resolved_at IS NULL" in window, (
        "G-A2: el dedupe SELECT de temporal_gate_proactive debe filtrar `AND resolved_at IS NULL`."
    )


# ---------------------------------------------------------------------------
# G-A3: failure-path learning_metrics persist lleva CAS guard
# ---------------------------------------------------------------------------
def test_g_a3_failure_path_learning_metrics_cas():
    body = _slice_fn(_src(), "def process_plan_chunk_queue(")
    # Anclar en la variable ÚNICA del failure-path (NO confundir con el preflight persist
    # `... WHERE id = %s AND learning_metrics IS NULL`, que es un write distinto e idempotente).
    idx = body.find("_lm_persist_res = execute_sql_write(")
    assert idx >= 0, "G-A3: no se encontró el persist CAS de learning_metrics (failure-path)."
    window = body[idx: idx + 260]
    assert "UPDATE plan_chunk_queue SET learning_metrics = %s::jsonb" in window, (
        "G-A3: _lm_persist_res debe persistir learning_metrics."
    )
    assert "AND attempts = %s AND status = 'processing'" in window, (
        "G-A3: el persist de learning_metrics tras fallo debe llevar el CAS guard "
        "`AND attempts = %s AND status = 'processing'` (simetría con T2 success + escalation)."
    )
    assert "CAS-DISPLACED" in body, "G-A3: debe loguear [G-A3/CAS-DISPLACED] cuando rowcount=0."


# ---------------------------------------------------------------------------
# G-A4: pantry-pause idempotency guard en _background_shift_plan_for_user
# ---------------------------------------------------------------------------
def test_g_a4_pantry_pause_idempotency_guard():
    body = _slice_fn(_src(), "def _background_shift_plan_for_user(")
    assert 'if plan_data.get("generation_status") == "expired_pending_pantry":' in body, (
        "G-A4: la rama pantry-pause debe tener guard de idempotencia sobre "
        "'expired_pending_pantry' para no re-escribir + re-pushear cada corrida diaria."
    )
    assert "PANTRY-PAUSE-IDEMPOTENT" in body, "G-A4: debe loguear el skip idempotente."


# ---------------------------------------------------------------------------
# G-B1: API inject reusa un superset de consumed_meals
# ---------------------------------------------------------------------------
def test_g_b1_api_inject_consumed_superset():
    body = _slice_fn(_src(), "def inject_learning_signals_from_profile(")
    assert "_api_consumed_super = get_consumed_meals_since(" in body, (
        "G-B1: debe fetchear UN superset _api_consumed_super."
    )
    # CRÍTICO: el superset debe traer ingredientes (la fatiga los necesita).
    sup_idx = body.find("_api_consumed_super = get_consumed_meals_since(")
    assert "include_ingredients=True" in body[sup_idx: sup_idx + 300], (
        "G-B1: el superset DEBE fetchearse con include_ingredients=True "
        "(calculate_ingredient_fatigue quedaría ciega sin ellos — regresión GAP-1)."
    )
    # Los 3 scorers reciben consumed=_api_consumed_super.
    assert body.count("consumed=_api_consumed_super") >= 3, (
        "G-B1: fatiga/success/adherence + cold-start deben reusar consumed=_api_consumed_super."
    )


# ---------------------------------------------------------------------------
# G-B2: _recover_pantry_paused_chunks lleva days_offset en el batch SELECT
#        y NO re-query-ea days_offset por fila.
# ---------------------------------------------------------------------------
def test_g_b2_recover_pantry_batch_days_offset():
    body = _slice_fn(_src(), "def _recover_pantry_paused_chunks(")
    assert "SELECT id, user_id, meal_plan_id, week_number, days_offset, pipeline_snapshot" in body, (
        "G-B2: el batch SELECT debe incluir days_offset."
    )
    assert "SELECT days_offset FROM plan_chunk_queue WHERE id = %s" not in body, (
        "G-B2: el re-query por-fila de days_offset (tz_unresolved) debe haberse eliminado."
    )
    assert "SELECT days_offset, week_number FROM plan_chunk_queue WHERE id = %s" not in body, (
        "G-B2: el re-query por-fila de days_offset/week_number (prev_chunk) debe haberse eliminado."
    )


# ---------------------------------------------------------------------------
# G-B3: quality gate hoisteado ARRIBA del calculate_plan_quality_score
# ---------------------------------------------------------------------------
def test_g_b3_quality_gate_hoisted_before_score():
    body = _slice_fn(_src(), "def _inject_advanced_learning_signals(")
    gate = body.find("_quality_data_sufficient = bool(consumed_records) and len(consumed_records) >= 3")
    call = body.find("calculate_plan_quality_score(user_id, {'days': days}, consumed_records, household_size)")
    assert gate >= 0, "G-B3: ancla del gate _quality_data_sufficient ausente."
    assert call >= 0, "G-B3: ancla del cálculo de quality_score ausente."
    assert gate < call, (
        "G-B3: el gate _quality_data_sufficient debe computarse ANTES de calculate_plan_quality_score "
        "(hoist) para evitar las 3 queries DB en cold-start."
    )
    # El cálculo es condicional al gate.
    assert "if _quality_data_sufficient else None" in body, (
        "G-B3: calculate_plan_quality_score debe ser condicional (None en cold-start)."
    )


# ---------------------------------------------------------------------------
# G-B4: orphan-detection usa NOT EXISTS sin cast ::text
# ---------------------------------------------------------------------------
def test_g_b4_cleanup_orphan_not_exists():
    body = _slice_fn(_src(), "def _cleanup_orphan_chunks(")
    assert "NOT EXISTS (SELECT 1 FROM meal_plans m WHERE m.id = q.meal_plan_id)" in body, (
        "G-B4: orphan-detection debe usar el anti-join NOT EXISTS (uuid, PK index)."
    )
    # Negativa sobre código (sin comentarios): el comentario del fix menciona el patrón viejo.
    assert "meal_plan_id::text NOT IN" not in _code_only(body), (
        "G-B4: el `meal_plan_id::text NOT IN (SELECT id::text ...)` (seq-scan, cast) debe eliminarse."
    )
    # Snapshot atómico preservado.
    assert "FOR UPDATE SKIP LOCKED" in body, "G-B4: el FOR UPDATE SKIP LOCKED debe preservarse."


# ---------------------------------------------------------------------------
# G-C1: fatiga del LLM branch reusa el superset SOLO cuando la ventana lo cubre
# ---------------------------------------------------------------------------
def test_g_c1_fatigue_conditional_reuse():
    body = _slice_fn(_src(), "def process_plan_chunk_queue(")
    assert "_fatigue_consumed = chunk_consumed_records if (int(days_offset or 0) >= 14) else None" in body, (
        "G-C1: la fatiga GAP-F debe reusar chunk_consumed_records SOLO si days_offset>=14 "
        "(su ventana max(7,offset) cubre los 14d) — si no, fetch propio (no estrechar la ventana)."
    )
    assert "calculate_ingredient_fatigue(user_id, tuning_metrics=tuning_metrics, consumed=_fatigue_consumed)" in body, (
        "G-C1: la fatiga debe recibir consumed=_fatigue_consumed."
    )


# ---------------------------------------------------------------------------
# G-C2: pantry_tolerance resuelto UNA vez, fuera del for _shuffle_idx
# ---------------------------------------------------------------------------
def test_g_c2_pantry_tolerance_hoisted():
    body = _slice_fn(_src(), "def process_plan_chunk_queue(")
    tol = body.find("_p1d_tolerance = _get_pantry_tolerance_for_user(user_id)")
    loop = body.find("for _shuffle_idx in range(days_count):")
    assert tol >= 0 and loop >= 0, "G-C2: anclas ausentes (tolerance call / shuffle loop)."
    assert tol < loop, (
        "G-C2: _get_pantry_tolerance_for_user debe resolverse ANTES del `for _shuffle_idx` (hoist)."
    )
    # Solo UNA resolución (no re-query por día).
    assert body.count("_p1d_tolerance = _get_pantry_tolerance_for_user(user_id)") == 1, (
        "G-C2: debe haber exactamente UNA resolución de pantry_tolerance (no por-día)."
    )


# ---------------------------------------------------------------------------
# G-C3: dead-write _shuffle_validation_failed eliminado + comentario heartbeat corregido
# ---------------------------------------------------------------------------
def test_g_c3_dead_write_and_heartbeat_comment():
    code = _code_only(_src())  # negativas sobre código sin comentarios (el fix los menciona)
    assert '_nd["_shuffle_validation_failed"] = True' not in code, (
        "G-C3: el dead-write per-día `_nd[\"_shuffle_validation_failed\"]` (0 readers) debe eliminarse."
    )
    # El counter que gobierna el abort se conserva.
    assert "_shuffle_validation_failed_count += 1" in code, (
        "G-C3: el counter _shuffle_validation_failed_count (que gobierna el abort) debe conservarse."
    )
    # El comentario heartbeat ficticio fue corregido (la frase exacta ya no aparece ni en comentarios).
    assert "El flujo principal lee `last_heartbeat_at` para detectar threads colgados" not in _src(), (
        "G-C3: el comentario ficticio sobre que el main flow lee last_heartbeat_at debe corregirse."
    )


# ---------------------------------------------------------------------------
# G-D: anotaciones I2-EXEMPT en los UPDATE plan_chunk_queue por meal_plan_id
# ---------------------------------------------------------------------------
def test_g_d_i2_exempt_annotations():
    src = _src()
    # Las 3 anotaciones G-D deben existir (cierran el re-flag perpetuo G7/G8/G9).
    assert src.count("G-D · P2-CRON-OPT-4") >= 3, (
        "G-D: deben existir las 3 anotaciones I2-EXEMPT G-D (P1-3 FIX block, _degraded, bshift)."
    )
