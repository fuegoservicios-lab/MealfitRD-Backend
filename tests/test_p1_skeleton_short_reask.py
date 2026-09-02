"""[P1-SKELETON-SHORT-REASK · 2026-09-02] Esqueleto corto: re-pedir, reutilizar, y fechar lo rellenado.

Vivo (run fdbb56de, plan eb24dc95, 2026-09-02 09:16–09:23 UTC): intento 1 rechazado por el
revisor («ninguna preparación transformada»); en el intento 2 el planificador (glm-5.3-flash,
temp 0.95, `_is_same_day_reroll=True`) murió a los 45 s de timeout, su reintento automático
devolvió 1/3 días, nadie volvió a preguntar y el guardrail P0-2 rellenó los días 2 y 3 con menú
matemático — sin `date`, porque el estampado corre antes del guardrail. Hoy: 3 esqueletos cortos
en 6 planes (2 en los 9 días anteriores).

Tres cierres, cada uno con su test:
  A. re-pedir UNA vez con la orden explícita de N días, y si sigue corto reutilizar los días
     faltantes del esqueleto del intento anterior (helpers puros + parser del call site);
  B. timeout del planificador por knob (`MEALFIT_PLANNER_TIMEOUT_S`, default 90) en el cliente
     Y en `_safe_ainvoke` (el de 45 fijo era el que mataba a GLM con razonamiento);
  C. `date`/`day_name` en los días sintéticos del guardrail (misma aritmética que el estampado).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")

go = pytest.importorskip("graph_orchestrator")


# ---------------------------------------------------------------- A. helpers puros
def _skel(n, **extra):
    return {"day": n, "protein_pool": [f"P{n}"], "carb_pool": [f"C{n}"], **extra}


def test_merge_fills_only_missing_days_from_previous_skeleton():
    current = [_skel(1)]
    previous = [_skel(1, protein_pool=["viejo"]), _skel(2), _skel(3)]
    merged, reused = go._merge_short_skeleton_days(current, previous, 3)
    assert reused == [2, 3]
    assert [d["day"] for d in merged] == [1, 2, 3]
    assert merged[0]["protein_pool"] == ["P1"], "el día del LLM manda sobre el previo"
    assert merged[1]["_reused_from_previous_attempt"] is True
    assert "_reused_from_previous_attempt" not in merged[0]


def test_merge_without_previous_is_a_noop():
    merged, reused = go._merge_short_skeleton_days([_skel(1)], [], 3)
    assert reused == [] and [d["day"] for d in merged] == [1]


def test_merge_is_a_deep_copy_and_ignores_days_beyond_chunk():
    previous = [_skel(1), _skel(2), _skel(3), _skel(4)]
    merged, reused = go._merge_short_skeleton_days([_skel(2)], previous, 3)
    assert reused == [1, 3] and [d["day"] for d in merged] == [1, 2, 3]
    merged[0]["protein_pool"].append("mutado")
    assert previous[0]["protein_pool"] == ["P1"], "no se muta el esqueleto previo"


def test_merge_uses_position_when_llm_omits_day_number():
    merged, reused = go._merge_short_skeleton_days([{"protein_pool": ["x"]}], [_skel(1), _skel(2)], 2)
    assert reused == [2] and [d["day"] for d in merged] == [1, 2]


def test_reask_directive_names_the_exact_count_and_overrides_today_wording():
    txt = go._skeleton_short_reask_directive(1, 3)
    assert "1 de 3" in txt and "EXACTAMENTE 3" in txt and "day 1..3" in txt
    assert "hoy" in txt, "la instrucción de re-roll («opciones de hoy») fue la que encogió el esqueleto"


# ---------------------------------------------------------------- A. call site (parser)
def test_reask_block_sits_after_skeleton_is_built_and_before_scrub():
    i = _SRC.find("[P1-SKELETON-SHORT-REASK · 2026-09-02] Esqueleto corto: re-pedir una vez")
    built = _SRC.find("skeleton = response.dict()")
    scrub = _SRC.find('skeleton["_selected_techniques"] = selected_techniques')
    assert -1 not in (i, built, scrub) and built < i < scrub
    win = _SRC[i:scrub]
    assert "SKELETON_SHORT_REASK_ENABLED and len(_skel_days_now) < days_in_chunk" in win
    assert "_skeleton_short_reask_directive(len(_skel_days_now), days_in_chunk) + prompt_text" in win
    assert "_merge_short_skeleton_days(_skel_days_now, _prev_skel_days, days_in_chunk)" in win
    assert 'state.get("plan_skeleton")' in win, "el esqueleto previo vive en el state"
    assert "attempt > 1" in win, "solo hay esqueleto previo a partir del intento 2"


def test_reask_is_a_single_extra_call_and_never_shrinks_the_skeleton():
    i = _SRC.find("[P1-SKELETON-SHORT-REASK · 2026-09-02] Esqueleto corto: re-pedir una vez")
    win = _SRC[i:i + 3200]
    assert win.count("_safe_ainvoke(planner_llm, _reask_payload") == 1
    assert "if len(_reask_days) > len(_skel_days_now):" in win, "solo se adopta si trae MÁS días"


# ---------------------------------------------------------------- B. timeout por knob
def test_planner_timeout_is_a_knob_used_in_both_places():
    assert 'PLANNER_LLM_TIMEOUT_S = _env_int("MEALFIT_PLANNER_TIMEOUT_S", 90)' in _SRC
    assert 'SKELETON_SHORT_REASK_ENABLED = _env_bool("MEALFIT_SKELETON_SHORT_REASK", True)' in _SRC
    assert "timeout=PLANNER_LLM_TIMEOUT_S,  # [P1-SKELETON-SHORT-REASK] era 45 fijo" in _SRC
    assert "_safe_ainvoke(_llm, planner_payload, timeout=float(PLANNER_LLM_TIMEOUT_S + 5))" in _SRC
    i = _SRC.find("async def plan_skeleton_node(")
    assert not re.search(r"timeout=45,", _SRC[i:i + 20000]), "el 45 fijo volvió al cliente del planner"
    assert go.PLANNER_LLM_TIMEOUT_S >= 60


# ---------------------------------------------------------------- C. fecha en días sintéticos
def test_stamp_missing_day_dates_only_touches_days_without_date():
    plan = {"days": [
        {"day": 1, "date": "2026-09-02", "day_name": "Miércoles"},
        {"day": 2},
        {"day": 3, "day_name": "Fallback"},
    ]}
    fd = {"_plan_start_date": "2026-09-02T09:16:05+00:00", "tzOffset": 240, "_days_offset": 0}
    stamped = go._stamp_missing_day_dates(plan, fd)
    assert stamped == [2, 3]
    assert plan["days"][0]["date"] == "2026-09-02", "nunca pisa una fecha existente"
    assert plan["days"][1]["date"] == "2026-09-03" and plan["days"][1]["day_name"] == "Jueves"
    assert plan["days"][2]["date"] == "2026-09-04" and plan["days"][2]["day_name"] == "Fallback", \
        "day_name existente se respeta; solo se rellena si falta"


def test_stamp_respects_days_offset_and_local_timezone():
    # 2026-09-02 02:30 UTC con UTC-4 es todavía 1 de septiembre local
    plan = {"days": [{"day": 1}, {"day": 2}]}
    fd = {"_plan_start_date": "2026-09-02T02:30:00+00:00", "tz_offset_minutes": 240, "_days_offset": 3}
    assert go._stamp_missing_day_dates(plan, fd) == [1, 2]
    assert [d["date"] for d in plan["days"]] == ["2026-09-04", "2026-09-05"]


def test_stamp_falls_back_to_now_without_start_date():
    plan = {"days": [{"day": 1}]}
    now = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
    assert go._stamp_missing_day_dates(plan, {}, now=now) == [1]
    assert plan["days"][0]["date"] == "2026-09-02"


def test_guardrail_stamps_dates_right_after_repair():
    i = _SRC.find("[P1-SKELETON-SHORT-REASK · 2026-09-02] Los días sintéticos nacían sin")
    assert i != -1, "el estampado del guardrail desapareció"
    before = _SRC[max(0, i - 900):i]
    after = _SRC[i:i + 3000]
    assert "_repair_partial_plan(plan_final, nutrition=nutrition, requested_days=requested_days," in before,         "el estampado va JUSTO después del relleno del guardrail (rama plan incompleto)"
    assert "_stamp_missing_day_dates(plan_final, actual_form_data)" in after
    j_stamp = after.find("_stamp_missing_day_dates")
    j_rstats = after.find("_rstats = plan_final.get(")
    assert j_rstats != -1 and j_stamp < j_rstats, "antes de leer _repair_stats / marcar _partial_repair"
