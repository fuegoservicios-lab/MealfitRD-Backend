"""[P1-FALLBACK-CAUSE-SPLIT · 2026-07-29] Incidente corr=23c65543: owner clicked "Generar Nueva
Opción", the pipeline ran for 143.69s, the medical reviewer APPROVED the plan
(`review_passed=True`), clinical band scored 1.00/1.00 post-finalize — and NOTHING was saved. No
LLM outage occurred anywhere in the window (verified: circuit breaker never opened, zero
errors/timeouts/4xx/5xx/tracebacks).

Root cause chain:
  1. The planner LLM (temp 0.95, driven by `_is_same_day_reroll`) returned a 1-element
     `plan_skeleton.days` array when 3 were requested — `PlanSkeletonModel.days` has no length
     constraint, and nothing checked the count before ~140s of expensive work ran on the truncated
     skeleton. Only 1 day-generation worker was ever scheduled; days 2-3 never existed, so there was
     no exception to log — the incident took full production log forensics to even notice the gap.
  2. `_apply_final_defense_guardrails` correctly detected `days=1/3` and called
     `_repair_partial_plan`, which filled the 2 missing days with a deterministic math fallback and
     set `plan["_is_fallback"] = True` — an UNCONDITIONAL flag shared with genuine LLM-outage
     fallbacks (`_get_extreme_fallback_plan`).
  3. The 3 `FALLBACK-GUARD` sites in `routers/plans.py` treated ANY `_is_fallback=True` as
     "emergency — discard, don't persist, don't queue chunks", logging "LLM upstream caído" — which
     was FALSE (nothing about the LLM was unavailable; the reviewer had already approved the
     content that DID arrive).

Fix (this file's tests): `_is_fallback` keeps its EXISTING semantics untouched (it still gates the
deterministic clinical-safety re-application on synthetic fallback days in
`_apply_final_defense_guardrails` — that coupling is load-bearing and this fix does not touch it).
A new, ORTHOGONAL flag `_partial_repair=True` (+ `_repair_stats` + `_fallback_source`) is set ONLY
when (a) at least one day survived the repair untouched (`real_days > 0`) AND (b) the medical
reviewer approved the plan (`review_passed=True`) — i.e. exactly the incident's shape, genuinely
NOT an outage. The 3 `FALLBACK-GUARD` sites in `routers/plans.py` now carve out `_partial_repair`
from the discard branch: it falls through to normal persistence instead of being thrown away, and
the emergency branches for the 3 remaining, GENUINELY-empty-handed causes
(`pipeline_exception` / `guardrail_empty_result` / `guardrail_all_synthetic`) now log the actual
branch that fired instead of a hardcoded, sometimes-false "LLM upstream caído".

Also covered: the loud ERROR log when the planner returns fewer skeleton days than requested
(`generate_days_parallel_node`) — before this fix, a short skeleton produced ZERO log trace, which
is exactly what made this incident undebuggable without raw production logs.
"""
from __future__ import annotations

import pathlib

import graph_orchestrator as g


_NUTR = {"target_calories": 2000, "macros": {"protein_g": 150, "carbs_g": 200, "fats_g": 60}}
_FORM = {"mainGoal": "Salud General"}


def _valid_day(n=1):
    return {
        "day": n,
        "meals": [{"meal": "Desayuno", "name": "Avena con leche", "ingredients": ["1 taza Avena"], "cals": 300}],
    }


def _invalid_day(n=1):
    return {"day": n, "meals": []}


# ---------------------------------------------------------------------------
# `_repair_partial_plan`: `_repair_stats` + disclaimer honesto según severidad
# ---------------------------------------------------------------------------

def test_repair_stats_partial_real_content_survives():
    plan = {"days": [_valid_day(1)]}
    repaired = g._repair_partial_plan(plan, nutrition=_NUTR, requested_days=3, form_data=_FORM)
    assert repaired is True
    assert plan["_repair_stats"] == {
        "requested_days": 3, "real_days": 1, "replaced_count": 0, "filled_count": 2,
    }
    assert len(plan["days"]) == 3
    # `_is_fallback` NUNCA se toca por este fix — sigue disparando la capa clínica
    # determinista sobre los días sintéticos aguas abajo.
    assert plan["_is_fallback"] is True


def test_repair_disclaimer_honest_when_real_content_survives():
    plan = {"days": [_valid_day(1)]}
    g._repair_partial_plan(plan, nutrition=_NUTR, requested_days=3, form_data=_FORM)
    disclaimer = plan["_review_disclaimer"]
    assert "1 de 3" in disclaimer
    assert "indisponibilidad temporal de la IA" not in disclaimer, (
        "el disclaimer NO debe culpar a un outage de IA cuando sí hubo contenido real"
    )


def test_repair_stats_zero_real_days_keeps_old_disclaimer():
    plan = {"days": []}
    g._repair_partial_plan(plan, nutrition=_NUTR, requested_days=3, form_data=_FORM)
    assert plan["_repair_stats"]["real_days"] == 0
    assert "indisponibilidad temporal de la IA" in plan["_review_disclaimer"]


# ---------------------------------------------------------------------------
# `_apply_final_defense_guardrails`: `_partial_repair` / `_fallback_source`
# ---------------------------------------------------------------------------

def test_guardrail_marks_partial_repair_when_review_passed(monkeypatch):
    monkeypatch.setattr(g, "FALLBACK_CLINICAL_LAYER_ENABLED", False)  # aislar de Neon
    plan = {"days": [_valid_day(1)]}
    final_state = {"plan_result": plan, "review_passed": True}
    g._apply_final_defense_guardrails(final_state, nutrition=_NUTR, actual_form_data=_FORM, requested_days=3)
    pr = final_state["plan_result"]
    assert pr["_is_fallback"] is True
    assert pr["_partial_repair"] is True
    assert pr["_fallback_source"] == "guardrail_partial_repair"
    assert pr["_repair_stats"]["real_days"] == 1


def test_guardrail_does_not_mark_partial_repair_when_review_failed(monkeypatch):
    """Un repair sobre un plan que la revisión médica RECHAZÓ no debe colarse por
    la puerta de persistencia — ese caso ya lo cubren `_critical_rejection`/
    `_review_failed_but_delivered` en `_apply_critical_review_guardrails`."""
    monkeypatch.setattr(g, "FALLBACK_CLINICAL_LAYER_ENABLED", False)
    plan = {"days": [_valid_day(1)]}
    final_state = {"plan_result": plan, "review_passed": False}
    g._apply_final_defense_guardrails(final_state, nutrition=_NUTR, actual_form_data=_FORM, requested_days=3)
    pr = final_state["plan_result"]
    assert not pr.get("_partial_repair")
    assert pr["_fallback_source"] == "guardrail_all_synthetic"


def test_guardrail_all_synthetic_when_zero_real_days_survive(monkeypatch):
    monkeypatch.setattr(g, "FALLBACK_CLINICAL_LAYER_ENABLED", False)
    plan = {"days": [_invalid_day(1), _invalid_day(2)]}
    final_state = {"plan_result": plan, "review_passed": True}
    g._apply_final_defense_guardrails(final_state, nutrition=_NUTR, actual_form_data=_FORM, requested_days=3)
    pr = final_state["plan_result"]
    assert not pr.get("_partial_repair")
    assert pr["_fallback_source"] == "guardrail_all_synthetic"


def test_guardrail_empty_result_tags_source(monkeypatch):
    monkeypatch.setattr(g, "FALLBACK_CLINICAL_LAYER_ENABLED", False)
    final_state = {"plan_result": None, "review_passed": True}
    g._apply_final_defense_guardrails(final_state, nutrition=_NUTR, actual_form_data=_FORM, requested_days=3)
    pr = final_state["plan_result"]
    assert pr["_is_fallback"] is True
    assert pr["_fallback_source"] == "guardrail_empty_result"
    assert not pr.get("_partial_repair")


def test_guardrail_complete_plan_is_a_pure_noop(monkeypatch):
    """Sanity: un plan YA completo no debe adquirir NINGUNO de estos flags."""
    monkeypatch.setattr(g, "FALLBACK_CLINICAL_LAYER_ENABLED", False)
    plan = {"days": [_valid_day(1), _valid_day(2), _valid_day(3)]}
    final_state = {"plan_result": plan, "review_passed": True}
    g._apply_final_defense_guardrails(final_state, nutrition=_NUTR, actual_form_data=_FORM, requested_days=3)
    pr = final_state["plan_result"]
    assert not pr.get("_is_fallback")
    assert not pr.get("_partial_repair")
    assert not pr.get("_fallback_source")


# ---------------------------------------------------------------------------
# Skeleton corto: log ERROR en caliente (antes: cero rastro hasta el guardrail
# final, ~2.5 min después — tomó forensia completa de producción notarlo)
# ---------------------------------------------------------------------------

def test_skeleton_short_log_anchor_present_before_worker_loop():
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    fn_idx = src.index("async def generate_days_parallel_node")
    next_node_idx = src.index("\ndef _generate_candidate", fn_idx) if "\ndef _generate_candidate" in src[fn_idx:] else len(src)
    # `_generate_candidate` es una función NESTED (async def) dentro de este nodo;
    # usamos el callsite único del primer worker como límite derecho en su lugar.
    worker_idx = src.index('day_coros.append(_safe_gen(skel_day, day_num, temp_override))', fn_idx)
    log_idx = src.index("❌ [SKELETON-SHORT]", fn_idx)
    assert fn_idx < log_idx < worker_idx, (
        "el log de skeleton corto debe vivir DENTRO de generate_days_parallel_node y ANTES de que "
        "se programe el primer worker — si corre después, ya se gastó tiempo/costo sobre el "
        "esqueleto truncado antes de que quede rastro alguno."
    )
    # Ancla al `if` LITERAL (no solo a la comparación en cualquier parte del texto) — un
    # `if False and len(skeleton_days) < days_in_chunk:` (rama neutralizada pero con el texto
    # intacto) NO matchea este substring exacto, así que esta aserción SÍ detecta esa sabotage
    # específica (una mera búsqueda de la comparación, sin el `if` pegado, no la detectaría).
    condition_idx = src.index("if len(skeleton_days) < days_in_chunk:", fn_idx)
    assert condition_idx < log_idx


def test_skeleton_short_uses_error_level_not_debug_or_info():
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    log_idx = src.index("❌ [SKELETON-SHORT]")
    # retroceder hasta el `logger.<level>(` inmediatamente anterior
    call_start = src.rfind("logger.", max(0, log_idx - 200), log_idx)
    assert call_start != -1
    call_line = src[call_start:log_idx]
    assert "logger.error(" in call_line, (
        f"el log de skeleton corto debe ser logger.error (WARNING o superior por brief), no: {call_line!r}"
    )


# ---------------------------------------------------------------------------
# Ningún worker de día se pierde SIN log — regresión general sobre el loop de
# resultados de `_generate_candidate` (día ya cubierto por código preexistente,
# pero sin test de regresión explícito hasta ahora).
# ---------------------------------------------------------------------------

def test_failed_day_worker_always_logged_at_error_or_higher():
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    block_start = src.index("for day_num, result, err in results:")
    block_end = src.index("if failed_days and generated_days:", block_start)
    block = src[block_start:block_end]
    assert "failed_days.append(day_num)" in block
    assert "logger.error(" in block, (
        "un día que falla definitivamente tras hedging DEBE loguearse a nivel ERROR — "
        "un worker perdido sin rastro es indebuggable (costó forensia completa de producción "
        "en corr=23c65543)."
    )
    # El log debe ocurrir ANTES del append a failed_days (no un log post-hoc en otro lado
    # que pueda saltarse si algo excepciona entre medio).
    log_pos = block.index("logger.error(")
    append_pos = block.index("failed_days.append(day_num)")
    assert log_pos < append_pos
    # No basta con que la SUBCADENA "exc_info=" exista en el bloque — un `# exc_info=(...)`
    # comentado la contendría igual y este assert pasaría falsamente. Exigimos una línea NO
    # comentada cuyo texto stripeado empiece con `exc_info=`.
    exc_info_lines = [
        ln for ln in block.splitlines()
        if ln.strip().startswith("exc_info=")
    ]
    assert exc_info_lines, (
        "debe preservar traceback via `exc_info=` en una línea de código real (no comentada), "
        "no solo type(err).__name__"
    )


# ---------------------------------------------------------------------------
# Anchors de marker (cross-link con `_LAST_KNOWN_PFIX` / P2-HIST-AUDIT-14)
# ---------------------------------------------------------------------------

def test_marker_anchor_present_in_graph_orchestrator():
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "P1-FALLBACK-CAUSE-SPLIT" in src
    assert "_partial_repair" in src
    assert "_fallback_source" in src
    assert "guardrail_partial_repair" in src
    assert "guardrail_all_synthetic" in src
    assert "guardrail_empty_result" in src
    assert "pipeline_exception" in src


def test_marker_anchor_present_in_routers_plans():
    import routers.plans as rp
    src = pathlib.Path(rp.__file__).read_text(encoding="utf-8")
    assert src.count('and not result.get("_partial_repair")') + src.count(
        'and not _result.get("_partial_repair")') == 3, (
        "los 3 guard sites de FALLBACK-GUARD deben excluir `_partial_repair` del descarte"
    )
    assert src.count('.get("_partial_repair"):') >= 3, (
        "cada uno de los 3 guard sites debe tener su rama `elif ... _partial_repair` que "
        "cae al flujo normal de persistencia"
    )


def test_last_known_pfix_bumped():
    app_src = pathlib.Path(g.__file__).parent.joinpath("app.py").read_text(encoding="utf-8")
    assert 'P1-FALLBACK-CAUSE-SPLIT · 2026-07-29' in app_src
