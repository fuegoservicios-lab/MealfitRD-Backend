"""[P1-SINGLE-TRIP-POLICY · 2026-09-05] «No, solo la compra grande» (`freshTopup=no` → `shopping.fresh_topup_days`
None con `main_cycle_days > 7`) solo lo respetaba la proyección. Ahora: (1) el prompt pide duraderos/congelados
del día 8 en adelante, (2) el validador de fidelidad avisa (`fresh_beyond_horizon`, warn) cuando un fresco
delicado cae fuera del horizonte, (3) la lista no se ventanea por semana y (4) el PDF dice «una sola compra».
"""
import json
from pathlib import Path

import pytest

import horizon as hz

_BACKEND = Path(__file__).resolve().parents[1]
_SINGLE = {"policy_hash": "h", "recurrence": {"global_mode": "balanced"},
           "shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "limited"}}
_TOPUPS = {"policy_hash": "h", "recurrence": {"global_mode": "balanced"},
           "shopping": {"main_cycle_days": 30, "fresh_topup_days": 7, "freezer_mode": "limited"}}


def _frontend(*parts):
    for base in (_BACKEND.parents[0], _BACKEND.parent):
        p = base.joinpath("frontend", "src", *parts)
        if p.exists():
            return p.read_text(encoding="utf-8").replace(chr(13), "")
    pytest.skip("frontend hermano no disponible")


def test_a_regla_unica():
    assert hz.single_trip_policy(_SINGLE) is True
    assert hz.single_trip_policy(_TOPUPS) is False
    assert hz.single_trip_policy({"shopping": {"main_cycle_days": 7, "fresh_topup_days": None}}) is False, "una semana es una compra por definición"
    assert hz.single_trip_policy(None) is False


def test_b_el_prompt_pide_duraderos_y_congelados_segun_el_congelador():
    lines = hz.single_trip_prompt_lines(_SINGLE)
    assert lines and "UNA sola compra para 30 días" in lines[0] and "repollo" in lines[0]
    assert "Congelador limitado" in lines[1]
    assert "Sin congelador" in hz.single_trip_prompt_lines({**_SINGLE, "shopping": {**_SINGLE["shopping"], "freezer_mode": "none"}})[1]
    assert "se congelan" in hz.single_trip_prompt_lines({**_SINGLE, "shopping": {**_SINGLE["shopping"], "freezer_mode": "full"}})[1]
    # bloque que empieza tras la semana de frescos: aviso explícito con sus días
    sl = {"days_offset": 14, "days": [{"day_index": 14}, {"day_index": 15}, {"day_index": 16}]}
    assert any("días 15–17" in l for l in hz.single_trip_prompt_lines(_SINGLE, sl))
    assert hz.single_trip_prompt_lines(_TOPUPS) == [], "con reposiciones semanales no se dice nada"
    block = hz.policy_prompt_block(_SINGLE, None, surface="test", enforced=True)
    assert "UNA sola compra" in block and block.index("UNA sola compra") < block.index("No «diversifiques»")


def _days(ingredients_by_day):
    return [{"meals": [{"type": "almuerzo", "ingredients": ings}]} for ings in ingredients_by_day]


def test_c_validador_warn_solo_tras_el_horizonte_y_solo_delicados():
    days = _days([["1 lechuga"], ["200 g de filete de pescado fresco"], ["1 lata de atún", "salsa de tomate"], ["2 tomates maduros"]])
    # bloque que empieza en el día 14 del ciclo: todo está fuera del horizonte de 7 días
    issues = hz.fresh_beyond_horizon_issues(days, {"days_offset": 14}, _SINGLE)
    codes = [(i["code"], i["day"]) for i in issues]
    # [F7-G] la proteína fresca fuera de la ventana de congelación tiene código propio (antes se confundía con un fresco)
    assert codes == [("fresh_beyond_horizon", 1), ("protein_beyond_freeze_window", 2), ("fresh_beyond_horizon", 4)], codes
    assert all(i["severity"] == "low" for i in issues), "modo warn: nunca bloquea"
    # el primer bloque (días 1-3) no avisa aunque use lechuga
    assert hz.fresh_beyond_horizon_issues(days, {"days_offset": 0}, _SINGLE) == []
    # con reposiciones semanales, nada
    assert hz.fresh_beyond_horizon_issues(days, {"days_offset": 14}, _TOPUPS) == []
    # el report los incluye
    src = (_BACKEND / "horizon.py").read_text(encoding="utf-8")
    assert "issues.extend(fresh_beyond_horizon_issues(days, sl, effective))" in src


def test_d_la_lista_no_se_ventanea_con_una_sola_compra(monkeypatch):
    import shopping_calculator as sc
    monkeypatch.setattr(sc, "_trip_windowed_perishables_enabled", lambda: True)
    plan = {"_plan_policy": {"effective": _SINGLE}, "days": [{"date": f"2026-09-{d:02d}"} for d in range(1, 15)], "grocery_start_date": "2026-09-01"}
    assert sc.plan_single_trip_policy(plan) is True
    assert sc.active_trip_window_days(plan) is None, "una sola compra: los perecederos van para todo el ciclo"
    plan_topups = {**plan, "_plan_policy": {"effective": _TOPUPS}}
    assert sc.plan_single_trip_policy(plan_topups) is False
    # el espejo del guard (ignore_knob) re-deriva del sello, no de la política: no se toca
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    assert "if not ignore_knob and plan_single_trip_policy(plan_data):" in src


def test_e_pdf_una_sola_compra_y_catalogos():
    dash = _frontend("pages", "Dashboard.jsx")
    assert "const isSingleTrip = !isWeekly && !!_policyShopping && Number(_policyShopping.main_cycle_days || 0) > 7 && !_policyShopping.fresh_topup_days;" in dash
    # [F7-G] el copy depende del congelador declarado (sin congelador / limitado / completo)
    assert "t('PERECEDEROS — UNA SOLA COMPRA (SIN CONGELADOR: CONSUME PRIMERO)')" in dash
    assert "t('PERECEDEROS — UNA SOLA COMPRA (CONGELA LO DE LA SEGUNDA SEMANA)')" in dash
    assert "t('COMPRA ESTA SEMANA — PERECEDEROS (REPITE CADA 7 DÍAS)')" in dash, "con reposiciones, el copy de siempre"
    for loc in ("en-US", "pt-BR", "fr-FR", "it-IT"):
        cat = json.loads(_frontend("i18n", "locales", f"{loc}.json"))
        # [F7-G] el rótulo depende del congelador: los tres textos viven en los cuatro catálogos
        for k in ("PERECEDEROS — UNA SOLA COMPRA (SIN CONGELADOR: CONSUME PRIMERO)",
                  "PERECEDEROS — UNA SOLA COMPRA (CONGELA LAS PROTEÍNAS EL DÍA DE LA COMPRA)",
                  "PERECEDEROS — UNA SOLA COMPRA (CONGELA LO DE LA SEGUNDA SEMANA)"):
            assert cat.get(k), (loc, k)
        assert "{duracion}" in cat["Elegiste reponer solo en la compra grande: estas cantidades cubren todo tu ciclo de {duracion}. Congela las proteínas y consume primero lo más delicado."], loc
