# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-288 · 2026-09-25] La meta del bariátrico que BAJA de peso respeta su propia regla clínica.

Bariátrica de la batería (95 kg, SOP, perder grasa gradual): meta 2.000 kcal con 93 g de grasa (42 %) contra la regla
que el sistema le da a la IA («1.400-1.700 kcal, aceite ≤ 1 cdta por comida»). El revisor la rechazaba por precisión de
macros y el plan salía con aviso tras ~11-13 minutos."""
from __future__ import annotations

import pathlib

import bariatrico_meta as bm
import nutrition_calculator as nc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_BARIATRICA = dict(age="35", weight="95", weightUnit="kg", height="165", gender="female", mainGoal="lose_fat",
                   activityLevel="light", goalPace="gradual", medicalConditions=["Cirugía Bariátrica", "SOP (PCOS)"])


def _grasa_pct(r):
    return 100.0 * r["macros"]["fats_g"] * 9 / r["target_calories"]


def test_el_techo_de_quien_pierde_es_el_de_su_regla():
    assert bm.techo("lose_fat", 2000) == 1700
    assert bm.techo("maintenance", 2000) == 2000 and bm.techo("gain_muscle", 2000) == 2000


def test_lo_liberado_por_la_proteina_no_infla_la_grasa():
    mac = {"protein_g": 150, "carbs_g": 150, "fats_g": 50}
    sobra = bm.capar_proteina(mac, 80, 1700, "lose_fat")
    assert mac["protein_g"] == 80 and sobra > 0
    final = 1700 - sobra
    assert abs(mac["fats_g"] * 9 - 0.35 * final) <= 9, (mac, final)          # 35 % de la meta FINAL (±1 g)
    mant = {"protein_g": 150, "carbs_g": 150, "fats_g": 50}
    assert bm.capar_proteina(mant, 80, 2000, "maintenance") == 0.0 and mant["fats_g"] == round(50 + 280 / 9)


def test_la_bariatrica_de_la_bateria():
    r = nc.get_nutrition_targets(dict(_BARIATRICA))
    assert 1400 <= r["target_calories"] <= 1700, r["target_calories"]        # antes 2.000
    assert r["macros"]["protein_g"] == 80
    assert _grasa_pct(r) <= 36.0, _grasa_pct(r)                               # antes 42 %
    m = r["macros"]
    assert abs(m["protein_g"] * 4 + m["carbs_g"] * 4 + m["fats_g"] * 9 - r["target_calories"]) <= 12   # la meta es la suma


def test_mantenimiento_y_no_bariatricos_no_cambian():
    r = nc.get_nutrition_targets(dict(_BARIATRICA, mainGoal="maintenance"))
    assert r["target_calories"] == 2000                                        # su decisión clínica, intacta
    r2 = nc.get_nutrition_targets(dict(_BARIATRICA, medicalConditions=["SOP (PCOS)"]))
    assert r2["target_calories"] > 1700 and r2["macros"]["protein_g"] > 80


def test_ganchos():
    src = (_BACKEND / "nutrition_calculator.py").read_text(encoding="utf-8")
    assert '_techo_b = __import__("bariatrico_meta").techo(goal, BARIATRIC_KCAL_CEILING_KCAL)' in src
    assert "_bm_288.capar_proteina(macros, _baria_cap, target_calories, goal)" in src
    assert "tooltip-anchor: P1-PLAN-LOTE-288-BARIATRICO-PIERDE" in (_BACKEND / "bariatrico_meta.py").read_text(encoding="utf-8")
