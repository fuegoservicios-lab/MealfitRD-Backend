"""[P2-CRON-OPT-3 · 2026-05-30] Anclas de la TERCERA pasada de optimización de
cron_tasks.py.

Re-audit multi-agente fresco (workflow, 79 agentes, 50 candidatos → 40 "confirmados"
por verificación adversaria, dedup → 22 gaps; veredicto 89/100). PERO la verificación
manual SQL-forensic-first del implementador (lección del repo: "no confiar ciegamente
en el finding") descartó/recortó varios:

  - G4 (adherence_history_rotations sin persistir en el inject path) = FALSO POSITIVO:
    el nightly (`_persist_nightly_learning_signals`) es el SSOT writer; añadir un write
    en el inject duplicaría el writer de tendencia (anti-patrón rechazado en
    P1-CHUNK-LEARN-2 G3). NO implementado.
  - G14 (filtros dead-lettered "dejan pasar" no-dicts) = NO es bug: el docstring dice
    explícitamente que las lecciones no-dead se conservan; preservar no-dicts es por
    diseño y se defienden downstream. NO implementado.
  - G13/G15/G17 + dead-writes cosméticos = bajo valor / no-bug → diferidos.
  - ~10 gaps tocan el monolito `_chunk_worker`/`_background_shift` (transaction-critical,
    37 tests parser anclan código adentro, reabre races lost-update) → DIFERIDOS al
    follow-up multi-PR dedicado, como en las 2 pasadas previas.

Implementados (7 gaps, todos monolith-free + verificados reales/seguros), anclados acá
sobre el SOURCE de prod (parser-based, sin import/DB) para que un revert falle ANTES de
degradar prod:

  G1  (P2) calculate_meal_level_adherence: `meal_type` consumed colapsa a 'snack'
      (default de log_consumed_meal) y jamás matchea los slots del plan ('Desayuno'...)
      → eaten=0 para todos → señal FALSA de "abandono total". Fail-safe: si hay
      consumed_records pero 0 intersección → return {} (incomputable, no all-zeros). El
      fix REAL es upstream (capturar meal_type al loguear) — follow-up.
  G2  (P2) get_similar_user_patterns: `cm.created_at` → `cm.consumed_at` (la columna
      semánticamente correcta + la que usa todo el resto del codebase).
  G3  (P2) _compute_chunk_delay_days: `totalDays` con fallback a `form_data` (antes solo
      raíz → 0 → adelanto del chunk final no disparaba en safety_margin mode).
  G11 (P2) chronic_deferrals: auto-resolve (antes resolved_at IS NULL para siempre).
  G12 (P2) temporal_gate_proactive: auto-resolve cuando el gate pasa.
  G16 (P2) misfire_grace_time en hot_table_bloat_tick + drain_pending_facts_queue.
  G18 (P2) tokens_estimated=0 en 2 ticks (antes recibían el count → drift semántico).

Detalle: ~/.claude/projects/.../memory/project_p2_cron_opt_3_2026_05_30.md
Tooltip-anchor: P2-CRON-OPT-3.
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
    """Devuelve el cuerpo de la función `header` hasta el siguiente def top-level."""
    start = src.find(header)
    assert start >= 0, f"no se encontró {header!r}"
    after = src[start + len(header):]
    nxt = re.search(r"\n(?:def |async def )\w", after)
    return after[: nxt.start()] if nxt else after


# ---------------------------------------------------------------------------
# Cross-link: el marker vive y pertenece a esta pasada
# ---------------------------------------------------------------------------
def test_marker_is_p2_cron_opt_3():
    # [P1-PROD-AUDIT-3 · 2026-05-30] Relajado a floor-fecha: audits posteriores
    # bumpean el marker; el exact-match `P2-CRON-OPT-3` rompía en el bump siguiente.
    # Freshness real → test_p3_1_last_known_pfix_freshness. Aquí: no retroceder
    # antes de 2026-05-30.
    app = _APP_PY.read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"[^"]*(\d{4})-(\d{2})-(\d{2})"', app)
    assert m, "No se encontró _LAST_KNOWN_PFIX con fecha parseable."
    assert (int(m.group(1)), int(m.group(2)), int(m.group(3))) >= (2026, 5, 30), (
        "El marker _LAST_KNOWN_PFIX retrocedió antes de 2026-05-30 (P2-CRON-OPT-3)."
    )


# ---------------------------------------------------------------------------
# G1: meal-level adherence fail-safe (evita señal falsa de abandono total)
# ---------------------------------------------------------------------------
def test_g1_meal_level_adherence_failsafe_returns_empty_on_no_intersection():
    body = _slice_fn(_src(), "def calculate_meal_level_adherence(")
    assert "_total_eaten = sum(s['eaten'] for s in meal_type_stats.values())" in body, (
        "G1: debe computar _total_eaten para el fail-safe."
    )
    assert "if consumed_records and _total_eaten == 0:" in body, (
        "G1: debe detectar consumed sin intersección con slots del plan."
    )
    # Tras el guard hay un `return {}` (no all-zeros).
    guard_idx = body.find("if consumed_records and _total_eaten == 0:")
    assert "return {}" in body[guard_idx: guard_idx + 600], (
        "G1: el fail-safe debe devolver {} (incomputable), NO el dict de all-zeros."
    )


# ---------------------------------------------------------------------------
# G2: get_similar_user_patterns usa consumed_at, no created_at
# ---------------------------------------------------------------------------
def test_g2_similar_user_patterns_uses_consumed_at():
    body = _slice_fn(_src(), "def get_similar_user_patterns(")
    assert "cm.consumed_at" in body, (
        "G2: la query de cold-start debe filtrar por cm.consumed_at."
    )
    assert "cm.created_at" not in body, (
        "G2: cm.created_at es la columna equivocada (drift); debe ser consumed_at."
    )


# ---------------------------------------------------------------------------
# G3: _compute_chunk_delay_days lee totalDays con fallback a form_data
# ---------------------------------------------------------------------------
def test_g3_chunk_delay_totaldays_form_data_fallback():
    body = _slice_fn(_src(), "def _compute_chunk_delay_days(")
    assert '(_snap.get("form_data", {}) or {}).get("totalDays")' in body, (
        "G3: totalDays debe tener fallback a form_data (snapshots dual-write/antiguos)."
    )


# ---------------------------------------------------------------------------
# G11: chronic_deferrals auto-resolve
# ---------------------------------------------------------------------------
def test_g11_chronic_deferrals_has_resolver_helper():
    src = _src()
    assert "def _resolve_cleared_chronic_deferral_alerts(" in src, (
        "G11: debe existir el helper de auto-resolve de chronic_deferrals."
    )
    helper = _slice_fn(src, "def _resolve_cleared_chronic_deferral_alerts(")
    assert "chronic_deferrals:%%" in helper and "resolved_at = NOW()" in helper, (
        "G11: el helper debe UPDATE ... resolved_at=NOW() sobre chronic_deferrals:%."
    )
    assert "alert_key <> ALL(%s::text[])" in helper, (
        "G11: debe resolver solo las claves NO activas (set vacío → resuelve todas)."
    )


def test_g11_resolver_wired_into_detector():
    body = _slice_fn(_src(), "def _detect_chronic_deferrals(")
    # Llamado en los 2 early-returns (sin activos) + tras el loop (con activos).
    assert body.count("_resolve_cleared_chronic_deferral_alerts(") >= 3, (
        "G11: el resolver debe invocarse en los early-returns y tras el loop."
    )
    assert "_resolve_cleared_chronic_deferral_alerts(set(deferrals_by_user.keys()))" in body, (
        "G11: tras el loop debe resolver las claves cuyo usuario ya no está activo."
    )


# ---------------------------------------------------------------------------
# G12: temporal_gate_proactive auto-resolve cuando el gate pasa
# ---------------------------------------------------------------------------
def test_g12_temporal_gate_proactive_auto_resolve():
    body = _slice_fn(_src(), "def _check_chunk_learning_ready(")
    assert "_p12_ready = (ratio_ready and not _signal_too_weak) or inventory_proxy_used" in body, (
        "G12: debe computar _p12_ready una sola vez."
    )
    assert 'f"temporal_gate_proactive:{user_id}:{meal_plan_id}:{int(week_number)}"' in body, (
        "G12: debe resolver el alert_key temporal_gate_proactive con el mismo formato del INSERT."
    )
    # El resolve está gateado por _p12_ready (gate pasó).
    idx = body.find("if _p12_ready:")
    assert idx >= 0 and "resolved_at = NOW()" in body[idx: idx + 700], (
        "G12: el resolve debe correr solo cuando el gate pasa (_p12_ready)."
    )


# ---------------------------------------------------------------------------
# G16: misfire_grace_time en los 2 crons agregadores
# ---------------------------------------------------------------------------
def test_g16_misfire_grace_on_aggregator_crons():
    src = _src()
    # Aislar cada bloque de registro y exigir misfire_grace dentro.
    for job_id in ('id="hot_table_bloat_tick"', 'id="drain_pending_facts_queue"'):
        idx = src.find(job_id)
        assert idx >= 0, f"no se encontró el registro {job_id}"
        block = src[idx: idx + 700]
        assert "misfire_grace_time=_aggregator_misfire_grace_s()" in block, (
            f"G16: {job_id} debe registrarse con misfire_grace_time (patrón P2-NEW-7)."
        )


# ---------------------------------------------------------------------------
# G18: tokens_estimated=0 en los ticks de sweep (drift semántico cerrado)
# ---------------------------------------------------------------------------
def test_g18_no_count_in_tokens_estimated_slot():
    src = _src()
    # El patrón drift era `VALUES (NULL, NULL, %s, 0, 0, %s, 0, %s::jsonb)` (count en
    # tokens_estimated). Tras el fix NINGÚN tick debe usarlo; todos usan 0 en ese slot.
    assert "VALUES (NULL, NULL, %s, 0, 0, %s, 0, %s::jsonb)" not in src, (
        "G18: ningún pipeline_metrics tick debe poner un count en tokens_estimated; "
        "la convención es 0 (el count vive en metadata)."
    )
    # Sanity: la convención correcta sí está presente en múltiples ticks.
    assert src.count("VALUES (NULL, NULL, %s, 0, 0, 0, 0, %s::jsonb)") >= 10, (
        "G18: los ticks de pipeline_metrics deben usar 0 en tokens_estimated."
    )
