"""[P2-CRON-OPT-2 · 2026-05-30] Anclas de la SEGUNDA pasada de optimización de
cron_tasks.py — gaps NUEVOS que el primer barrido (P2-CRON-OPT · 2026-05-29) no
encontró, hallados por un re-audit más profundo (workflow multi-agente, 81 agentes,
55 candidatos → 14 confirmados, veredicto 86/100).

Cada test ancla un fix concreto (NG-1..NG-8) sobre el SOURCE de prod (parser-based,
sin import/DB) para que un refactor que revierta el fix falle ANTES de degradar prod.

Resumen de los gaps:
  NG-1  (P1+P2) key-drift round 2:
        (A) calculate_meal_level_adherence leía 'created_at' (clave inexistente en el
            fetcher) → days_passed siempre 1 → adherencia por meal-type saturada.
        (B) calculate_plan_quality_score lee 'ingredients' pero 3 callsites no
            proyectaban la columna → diversity_score SIEMPRE 0 (20% del score).
        (cobertura conductual completa en test_gap_1_consumed_meals_fetcher_contract.py)
  NG-2  (P2) cooldown de `chunk_lesson_synth_ratio_high` sin `AND resolved_at IS NULL`
        → re-alertas suprimidas tras el auto-resolve del G6 (regresión live de ayer).
  NG-3  (P3) null-deref en el guard de fatigue_data (orden de evaluación).
  NG-4  (P3) I2: 8 UPDATE meal_plans sin `AND user_id=%s` (+ blind-spot multilínea del
        parser); cobertura del contrato en test_p2_new_8_cron_tasks_i2_contract.py.
  NG-5  (P3) _persist_plan_data return no chequeado → over-count de telemetría.
  NG-6  (P3) dead-code: re-import muerto, comentario stale (4→5 prefixes), columna
        lag_seconds muerta + dual-log engañoso.
  NG-7  (P3) doble fetch de health_profile en el path degradado (Smart-Shuffle).
  NG-8  (P3) ventana de inactividad '3 days' hardcoded → knob make_interval(days => %s).

Detalle: ~/.claude/projects/.../memory/project_p2_cron_opt_2_2026_05_30.md
Tooltip-anchor: P2-CRON-OPT-2.
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
    start = src.find(header)
    assert start >= 0, f"no se encontró {header!r}"
    after = src[start + len(header):]
    nxt = re.search(r"\n(?:def |async def )\w", after)
    return after[: nxt.start()] if nxt else after


# ---------------------------------------------------------------------------
# Cross-link: el marker vive y pertenece a esta pasada
# ---------------------------------------------------------------------------
def test_marker_is_p2_cron_opt_2():
    # [P2-CRON-OPT-3 · 2026-05-30] Relajado de exact-match a floor de familia.
    # [P2-GENCHUNK-SPEED · 2026-06-01] Re-relajado a floor-de-FECHA (igual que el
    # sibling ya-relajado test_p2_cron_opt_4::test_marker_is_p2_cron_opt_family):
    # pasadas POSTERIORES de OTRAS familias (P1-BUDGET-CUSTOM, P0-TIER-RLS-LOCK,
    # P3-BACKEND-AUDIT, P2-GENCHUNK-SPEED...) bumpean legítimamente el marker fuera
    # de la familia P2-CRON-OPT. Exigir la familia volvía este test rojo perpetuo
    # (thrash documentado). La freshness real (formato + floor) la enforza
    # test_p3_1_last_known_pfix_freshness; acá solo: el marker no retrocede.
    app = _APP_PY.read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"[^"]*(\d{4})-(\d{2})-(\d{2})"', app)
    assert m, "No se encontró _LAST_KNOWN_PFIX con fecha parseable."
    assert (int(m.group(1)), int(m.group(2)), int(m.group(3))) >= (2026, 5, 30), (
        "El marker _LAST_KNOWN_PFIX retrocedió antes de 2026-05-30 (P2-CRON-OPT-2/3)."
    )


# ---------------------------------------------------------------------------
# NG-1A: calculate_meal_level_adherence lee consumed_at, no created_at
# ---------------------------------------------------------------------------
def test_ng1a_meal_level_adherence_reads_consumed_at():
    body = _slice_fn(_src(), "def calculate_meal_level_adherence(")
    assert "_coerce_consumed_at_to_dt(record.get('consumed_at'))" in body, (
        "meal-level adherence debe leer consumed_at vía _coerce_consumed_at_to_dt"
    )
    assert "'created_at' in record" not in body, (
        "NG-1A regresión: meal-level adherence NO debe leer la clave inexistente "
        "'created_at' (→ days_passed siempre 1 → adherencia saturada)."
    )


# ---------------------------------------------------------------------------
# NG-1B: los 3 fetches que alimentan calculate_plan_quality_score opt-in ingredients
# ---------------------------------------------------------------------------
def test_ng1b_quality_callsites_include_ingredients():
    src = _src()
    n_since = src.count("get_consumed_meals_since(user_id, since_time, include_ingredients=True)")
    assert n_since >= 2, (
        f"NG-1B: ≥2 callsites `since_time, include_ingredients=True` (chunk worker + "
        f"persist nightly); got {n_since}."
    )
    assert "get_consumed_meals_since(user_id, plan_start_date_str, include_ingredients=True)" in src, (
        "NG-1B: trigger_incremental_learning debe fetchear con include_ingredients=True."
    )


# ---------------------------------------------------------------------------
# NG-2: cooldown de chunk_lesson_synth_ratio_high filtra resolved_at IS NULL
# ---------------------------------------------------------------------------
def test_ng2_synth_ratio_cooldown_filters_resolved():
    body = _slice_fn(_src(), "def _alert_high_synthesized_lesson_ratio(")
    # El cooldown SELECT (make_interval(hours => %s)) debe incluir AND resolved_at IS NULL.
    m = re.search(
        r"SELECT triggered_at FROM system_alerts.*?make_interval\(hours => %s\).*?LIMIT 1",
        body, re.S,
    )
    assert m, "No se encontró el cooldown SELECT de _alert_high_synthesized_lesson_ratio."
    assert "resolved_at IS NULL" in m.group(0), (
        "NG-2 regresión: el cooldown de chunk_lesson_synth_ratio_high debe filtrar "
        "`AND resolved_at IS NULL` (si no, suprime re-alertas tras el auto-resolve del G6)."
    )


# ---------------------------------------------------------------------------
# NG-3: guard null-safe de fatigue_data
# ---------------------------------------------------------------------------
def test_ng3_fatigue_data_null_safe():
    src = _src()
    assert '(fatigue_data or {}).get("fatigued_ingredients")' in src, (
        "NG-3: la comprehension de current_fatigued_pure debe ser None-safe "
        "`(fatigue_data or {}).get(...)`."
    )
    # El patrón buggy (guard ineficaz tras el iterable) no debe reaparecer.
    assert 'for f in (fatigue_data.get("fatigued_ingredients") or [])' not in src, (
        "NG-3 regresión: el iterable evalúa fatigue_data.get() ANTES del guard "
        "`if fatigue_data` → AttributeError si fatigue_data es None."
    )


# ---------------------------------------------------------------------------
# NG-4: I2 — los 2 UPDATEs que el blind-spot multilínea ocultaba ya cumplen
# (contrato completo en test_p2_new_8_cron_tasks_i2_contract.py, widened)
# ---------------------------------------------------------------------------
def test_ng4_heal_plan_start_date_filters_user_id():
    body = _slice_fn(_src(), "def _check_chunk_learning_ready(")
    # El heal de _plan_start_date (P0-A) debe filtrar AND user_id = %s.
    assert "'{_plan_start_date}'" in body
    m = re.search(r"UPDATE meal_plans.*?_plan_start_date.*?WHERE id = %s( AND user_id = %s)?", body, re.S)
    assert m and m.group(1), (
        "NG-4 regresión: el heal de _plan_start_date en _check_chunk_learning_ready "
        "debe filtrar `AND user_id = %s` (user_id está en scope)."
    )


def test_ng4_item_level_restock_has_i2_exempt():
    body = _slice_fn(_src(), "def _reactivate_shopping_list_after_perishable_cycle(")
    assert "restocked_items" in body
    assert "I2-EXEMPT" in body, (
        "NG-4: el UPDATE item-level de restocked_items (sweep cross-user system-wide) "
        "debe llevar marker `# I2-EXEMPT: <razón>`."
    )


# ---------------------------------------------------------------------------
# NG-5: _persist_plan_data return chequeado en el cron de coherencia
# ---------------------------------------------------------------------------
def test_ng5_persist_return_checked():
    body = _slice_fn(_src(), "def _shopping_coherence_alert_job(")
    assert "if _persist_plan_data(plan_id, plan_data, user_id=user_id):" in body, (
        "NG-5: el cron de coherencia debe chequear el retorno de _persist_plan_data "
        "antes de incrementar persisted_count (evita over-count en UPDATE de 0 filas)."
    )


# ---------------------------------------------------------------------------
# NG-6: dead-code cleanup
# ---------------------------------------------------------------------------
def test_ng6a_dead_reimport_removed():
    # Chequea el STATEMENT de import, no el bare identifier (un comentario podría
    # citarlo legítimamente — lección "comentario citando literal rompe substring-check").
    assert "from knobs import _env_int as " not in _src(), (
        "NG-6A: el re-import alias muerto `from knobs import _env_int as ...` debe eliminarse."
    )


def test_ng6b_kv_sweep_comment_says_5_prefixes():
    src = _src()
    assert "Limpia 4 prefixes" not in src, "NG-6B: comentario stale '4 prefixes' debe corregirse."
    assert "Limpia 5 prefixes" in src, "NG-6B: el comentario debe reflejar 5 prefixes (compressor_cache añadido)."


def test_ng6c_stuck_chunk_lag_reads_raw():
    body = _slice_fn(_src(), "def _detect_and_escalate_stuck_chunks(")
    assert "lag_h = round((r.get('lag_seconds') or 0) / 3600.0, 1)" in body, (
        "NG-6C: lag_h debe leer el `lag_seconds` raw (consume la columna antes muerta + "
        "hace el dual-log raw-vs-effective significativo)."
    )


# ---------------------------------------------------------------------------
# NG-7: reuso del health_profile en el path degradado
# ---------------------------------------------------------------------------
def test_ng7_degraded_health_profile_reused():
    src = _src()
    # init + set + reuse → al menos 3 menciones.
    assert src.count("_degraded_hp_cached") >= 3, (
        "NG-7: el path degradado debe cachear y reusar el health_profile "
        "(`_degraded_hp_cached`) en vez de re-fetchear el mismo row."
    )


# ---------------------------------------------------------------------------
# NG-8: ventana de inactividad como knob
# ---------------------------------------------------------------------------
def test_ng8_bg_refill_inactivity_knob():
    body = _slice_fn(_src(), "def trigger_background_rolling_refill(")
    assert "MEALFIT_BG_REFILL_INACTIVITY_DAYS" in body, (
        "NG-8: la ventana de inactividad debe leerse del knob MEALFIT_BG_REFILL_INACTIVITY_DAYS."
    )
    assert "make_interval(days => %s)" in body, (
        "NG-8: usar make_interval(days => %s) + bound param (NO interpolación Python en SQL)."
    )
    assert "INTERVAL '3 days'" not in body, "NG-8 regresión: el literal '3 days' debe estar parametrizado."
