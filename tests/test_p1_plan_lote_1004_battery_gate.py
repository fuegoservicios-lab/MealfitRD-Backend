"""[P1-PLAN-LOTE-1004 · 2026-11-14] (G-04 + G-15) Puerta automática corta de batería del coach y matriz de cobertura de reglas.

Verifica SIN gastar IA:
1. Todos los 19 casos del preset corto (`SHORT_PRESET_IDS`) existen en `battery.json` con `assert_spec` determinista.
2. `M1` (fusión multi-foto pan + jugo sobre cena) y `M2` (memoria lejana `search_deep_memory`) cubren sus huecos.
3. `evaluar_caso` detecta argumentos incorrectos, ausencia de pan en `M1`, herramientas prohibidas y fugas `[ID_REGISTRO_DIARIO]`.
4. `ledger.jsonl` genera el formato `batería N/M, antes N'/M` y `regression_detected` bloquea regresiones.
5. `rule_coverage.build_coverage_matrix()` indexa todas las reglas emitidas por `prompts/chat_agent.py` con 0 huecos y `comparar_reglas_prompt` detecta si se pierde alguna regla.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_COACH_BAT_DIR = _BACKEND_ROOT / "scripts" / "coach_battery"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))
if str(_COACH_BAT_DIR) not in sys.path:
    sys.path.insert(0, str(_COACH_BAT_DIR))

import evaluador  # noqa: E402
import rule_coverage  # noqa: E402


def _load_battery_by_id() -> dict[str, dict]:
    data = json.loads((_COACH_BAT_DIR / "battery.json").read_text(encoding="utf-8"))
    return {c["id"]: c for c in data["casos"]}


def test_short_preset_has_19_cases_with_deterministic_specs():
    assert len(evaluador.SHORT_PRESET_IDS) == 19
    by_id = _load_battery_by_id()
    for cid in evaluador.SHORT_PRESET_IDS:
        assert cid in by_id, f"Falta caso {cid} en battery.json"
        spec = (by_id[cid].get("expect") or {}).get("assert_spec")
        assert isinstance(spec, dict) and spec, f"Falta assert_spec determinista en {cid}"


def test_evaluador_passes_valid_m1_multiphoto_merge_and_catches_omitted_pan():
    case = _load_battery_by_id()["M1"]
    turns_ok = [
        {
            "tools": [{"tool": "log_consumed_meal", "args": {"meal_name": "Puré de plátano maduro con pollo", "meal_type": "cena", "calories": 600, "protein": 31}}],
            "errors": [],
            "contaminated": [],
            "final": "Anoté tu cena: ~600 kcal y 31 g de proteína.",
        },
        {
            "tools": [{
                "tool": "correct_consumed_meal",
                "args": {
                    "meal_id": "abc",
                    "meal_name": "Puré de plátano maduro con pollo, pan de agua y jugo de lechoza con azúcar",
                    "ingredients": ["Puré de plátano", "Pollo", "Pan de agua", "Jugo de lechoza con azúcar"],
                },
            }],
            "errors": [],
            "contaminated": [],
            "final": "Actualicé tu cena con el pan de agua y el jugo de lechoza con azúcar.",
        },
    ]
    res_ok = evaluador.evaluar_caso(case, turns_ok)
    assert res_ok["passed"], res_ok["failures"]

    # Regresión tipo captura: omitió el pan y solo actualizó lechoza
    turns_bad = [
        turns_ok[0],
        {
            "tools": [{
                "tool": "correct_consumed_meal",
                "args": {
                    "meal_id": "abc",
                    "meal_name": "Puré de plátano maduro con pollo y jugo de lechoza con azúcar",
                },
            }],
            "errors": [],
            "contaminated": [],
            "final": "La cena quedó actualizada con la lechoza con azúcar.",
        },
    ]
    res_bad = evaluador.evaluar_caso(case, turns_bad)
    assert not res_bad["passed"]
    assert any("correct_consumed_meal" in f for f in res_bad["failures"])


def test_evaluador_catches_leak_and_wrong_arguments():
    by_id = _load_battery_by_id()
    b6 = by_id["B6"]
    bad_b6 = [{
        "tools": [{"tool": "log_consumed_meal", "args": {"meal_name": "Chimi", "days_ago": 0, "calories": 550, "protein": 24}}],
        "errors": [],
        "contaminated": [],
        "final": "Anoté tu chimi [ID_REGISTRO_DIARIO: 123].",
    }]
    res = evaluador.evaluar_caso(b6, bad_b6)
    assert not res["passed"]
    assert any("days_ago=0" in f for f in res["failures"])
    assert any("internal_leak" in f for f in res["failures"])

    m2 = by_id["M2"]
    ok_m2 = [{
        "tools": [{"tool": "search_deep_memory", "args": {"user_id": "u1", "query": "inicio plan estómago"}}],
        "errors": [],
        "contaminated": [],
        "final": "Al principio del plan comentaste que los lácteos enteros te caían pesados.",
    }]
    assert evaluador.evaluar_caso(m2, ok_m2)["passed"]


def test_ledger_comparison_and_regression_detection(tmp_path):
    ledger_file = tmp_path / "ledger.jsonl"
    prev = evaluador.build_ledger_entry(
        label="antes",
        preset="short",
        model="gpt-6-luna",
        fallback_model="gemini-4-hydrogen",
        vision_model="gemini-4-hydrogen",
        passed=18,
        total=19,
        cost_usd=0.08,
        calls=32,
        cache_hit_pct=45.0,
        p50_secs=2.9,
        p90_secs=4.4,
        failed_ids=["A1"],
        git_sha="abc1234",
    )
    evaluador.append_ledger_entry(prev, ledger_file)
    entries = evaluador.load_ledger(ledger_file)
    baseline = evaluador.latest_baseline_entry(entries, preset="short", exclude_label="despues")
    assert baseline is not None

    assert evaluador.format_comparison_line(19, 19, baseline) == "batería 19/19, antes 18/19"
    assert not evaluador.regression_detected(19, 19, baseline, current_failed_ids=[])
    assert evaluador.regression_detected(17, 19, baseline, current_failed_ids=["A1", "B2"])
    # Incluso con igual N/M, si falló un caso distinto nuevo se marca regresión
    assert evaluador.regression_detected(18, 19, baseline, current_failed_ids=["B2"])


def test_rule_coverage_matrix_has_zero_uncovered_rules_and_detects_drop():
    matrix = rule_coverage.build_coverage_matrix()
    assert matrix["total_rules"] >= 54
    assert matrix["uncovered_rules"] == [], f"Reglas sin cobertura: {matrix['uncovered_rules']}"

    diff = rule_coverage.comparar_reglas_prompt(
        "Regla [BB-025 / P1-DIARY-COMPLEMENTS-SECTION] y [P0-CHAT-ALLERGY-SSOT].",
        "Regla [BB-025 / P1-DIARY-COMPLEMENTS-SECTION].",
    )
    assert diff["perdidas"] == ["P0-CHAT-ALLERGY-SSOT"]
