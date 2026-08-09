# [P1-VARIETY-REPORT-FRESH · 2026-08-09] El variety_report que los gates del review consumen
# (huevo hard-gate + _variety_repeat_gate_issues: proteína same-day, fruta, clash, base-dish)
# nace en _apply_deterministic_clinical_layer, pero el TAIL de assemble sigue mutando los días
# DESPUÉS (P0-BAND-PRE-REVIEW, P1-SAMEDAY-BURN-FIX, P1-FRUIT-SAVORY-BURN-FIX, phantom-dairy,
# grain-dry, dup-merge…). Medido en los N=20 del 2026-08-08/09: 10 correlaciones donde el
# re-autofix logueó "→ después: limpio" y 60-90s más tarde el gate rechazó por la MISMA
# repetición ya corregida (e67afefd y 66512b91 quemaron sus DOS intentos así, y la directiva
# de retry acusaba al LLM de un defecto que el plan ya no tenía). El gate debe medir LO QUE
# JUZGA: recompute puro (~ms) sobre los días actuales en review, misma exención de staples
# que el productor. La clase es la de P1-MICRO-REPORT-REFRESH: un panel que nadie refresca
# en el único punto donde se decide.
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("MEALFIT_DB_BACKEND", "neon")
os.environ.setdefault("NEON_DATABASE_URL", "postgresql://stub:stub@localhost:5432/stub")
os.environ.setdefault("NEON_DATABASE_URL_UNPOOLED", "postgresql://stub:stub@localhost:5432/stub")

import graph_orchestrator as go  # noqa: E402

_SRC = open(os.path.join(os.path.dirname(__file__), "..", "graph_orchestrator.py"),
            encoding="utf-8").read()


def _mk_plan(days):
    return {"days": days}


def _meal(name, ingredients):
    return {"name": name, "ingredients": ingredients, "meal": "almuerzo"}


def _clean_day():
    # 4 comidas, proteínas TODAS distintas (sin repetición same-day)
    return {
        "day": 1,
        "meals": [
            _meal("Revoltillo criollo", ["2 huevos", "1 tomate"]),
            _meal("Pechuga de pollo guisada", ["150 g de pechuga de pollo"]),
            _meal("Yogurt con fruta", ["170 g de yogurt griego"]),
            _meal("Filete de pescado al horno", ["150 g de filete de pescado blanco"]),
        ],
    }


def _dirty_day():
    # pollo en 2 comidas del MISMO día → el gate debe verlo
    return {
        "day": 1,
        "meals": [
            _meal("Pechuga de pollo guisada", ["150 g de pechuga de pollo"]),
            _meal("Wrap de pollo", ["100 g de pollo desmenuzado"]),
            _meal("Yogurt con fruta", ["170 g de yogurt griego"]),
        ],
    }


def test_helper_existe():
    assert hasattr(go, "_refresh_variety_report_for_gates"), (
        "el helper de refresh desapareció — sin él, review juzga con el reporte RANCIO "
        "de mitad de assemble (10 rechazos fantasma medidos en los N=20 del 2026-08-08/09)"
    )


def test_reporte_rancio_que_acusa_se_limpia():
    # El caso medido (corr=140dfe19/e67afefd/66512b91): los días YA están limpios
    # (el refix corrigió), pero el reporte viejo sigue diciendo "repetición".
    plan = _mk_plan([_clean_day()])
    plan["variety_report"] = {"same_day_protein_repeats": 1, "fruit_repeats": 2,
                              "egg_meals": 9, "total_meals": 4}
    go._refresh_variety_report_for_gates(plan, {})
    rep = plan["variety_report"]
    assert rep.get("same_day_protein_repeats") == 0, (
        "el refresh debe recomputar sobre los días ACTUALES — un reporte que acusa una "
        "repetición ya corregida quema el intento entero y le miente al retry"
    )
    assert rep.get("fruit_repeats") == 0


def test_reporte_rancio_que_absuelve_tambien_se_corrige():
    # Dirección simétrica (under-detection): un pase tardío INTRODUCE la repetición y el
    # reporte viejo dice "limpio" — el gate quedaría ciego. El refresh cierra ambas.
    plan = _mk_plan([_dirty_day()])
    plan["variety_report"] = {"same_day_protein_repeats": 0}
    go._refresh_variety_report_for_gates(plan, {})
    assert plan["variety_report"].get("same_day_protein_repeats", 0) >= 1, (
        "el refresh también debe VER una repetición nacida después del reporte original"
    )


def test_fail_safe_conserva_el_reporte_previo(monkeypatch):
    plan = _mk_plan([_clean_day()])
    stale = {"same_day_protein_repeats": 1, "marker": "stale"}
    plan["variety_report"] = stale

    def _boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(go, "build_variety_report", _boom)
    go._refresh_variety_report_for_gates(plan, {})  # no debe levantar
    assert plan["variety_report"] is stale, (
        "fail-safe: si el recompute falla, el reporte previo se conserva (rancio > ausente)"
    )


def test_sin_days_no_toca_nada():
    plan = {"days": [], "variety_report": {"same_day_protein_repeats": 3}}
    go._refresh_variety_report_for_gates(plan, {})
    assert plan["variety_report"] == {"same_day_protein_repeats": 3}, (
        "sin días no hay nada que medir — pisar el reporte con uno vacío sería inventar datos"
    )
    go._refresh_variety_report_for_gates(None, {})  # no debe levantar


def test_review_refresca_antes_de_consumir():
    # Estructural: review_plan_node debe invocar el refresh ANTES de leer plan.get("variety_report")
    # para los gates. Anclas estructurales (no ventanas de chars fijas — 6ª lección de la clase):
    # del inicio de review_plan_node al consumo `_vr = plan.get("variety_report")`.
    i_def = _SRC.index("async def review_plan_node")
    i_consume = _SRC.index('_vr = plan.get("variety_report")', i_def)
    body = _SRC[i_def:i_consume]
    assert "_refresh_variety_report_for_gates(plan, form_data)" in body, (
        "review_plan_node debe refrescar el variety_report ANTES de que el hard-gate de huevo "
        "y _variety_repeat_gate_issues lo consuman — si esto falta, el gate vuelve a juzgar "
        "con el reporte de mitad de assemble (rechazos fantasma medidos 2026-08-08/09)"
    )


def test_paridad_staples_con_el_productor():
    # El productor (clinical layer, línea P3-VARIETY) pasa user_staples=_user_staple_labels(form_data);
    # el refresh debe pasar LO MISMO o la exención staple+técnica-distinta divergiría entre
    # el reporte de assemble y el de review (dos mediciones honestas que discrepan = bug).
    i_h = _SRC.index("def _refresh_variety_report_for_gates")
    win = _SRC[i_h:_SRC.index("\ndef ", i_h + 10)]
    assert "_user_staple_labels(form_data)" in win, (
        "el refresh debe aplicar la MISMA exención de staples que el productor del reporte"
    )
