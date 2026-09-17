# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-77 · 2026-09-17] Lo que dejó la batería de 63 casos del coach con DeepSeek (17-sep, 02:00; 0 fallos duros):

1. El plan que ve el coach pesaba 38,7 KB de JSON en CADA turno, casi todo telemetría por comida (`_misalign_trace`,
   `_solver_raw_by_food`, `ingredients_raw`…) y sin las sumas del día: el modelo sumó 121 g de proteína donde el plan
   daba 127 g. Poda profunda + `totales_dia`.
2. Reglas M-P del prompt: una petición puntual no es perfil (C5 guardó «sin pescado esta noche» como rechazo permanente),
   nada que no pueda hacer (B7 ofreció recordatorios), idioma íntegro (I3/I4) y el saludo con la próxima comida (A3).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def _plan():
    return {
        "days": [
            {"day": 1, "date": "2026-09-16", "day_name": "Miércoles", "_day_index": 0, "_sodium_mg_est": 1200,
             "meals": [
                 {"meal": "Desayuno", "name": "Yaniqueques con huevo", "cals": 512, "protein": 26, "carbs": 58, "fats": 20,
                  "macros": ["P:26g", "C:58g", "G:20g"], "ingredients": ["2 huevos"], "ingredients_raw": [{"n": "huevo", "g": 100}],
                  "recipe": ["Bate los huevos"], "_display": {"en-US": {"name": "Johnnycakes"}},
                  "_misalign_trace": [{"stage": "x"}], "_solver_raw_by_food": {"huevo": 100}},
                 {"meal": "Cena", "name": "Pollo al horno", "cals": 668, "protein": 53, "carbs": 78, "fats": 15,
                  "ingredients": ["muslo de pollo"], "_closer_raw_by_food": {}},
                 "no-dict",
             ]},
            "no-dict",
        ],
        "calories": 2100, "_plan_policy": {"requested": "x"}, "_fidelity_report": {"score": 1},
        "aggregated_shopping_list": [{"name": "Huevo"}],
    }


def test_la_poda_es_profunda_y_no_muta_el_plan():
    import agent
    plan = _plan()
    antes = copy.deepcopy(plan)
    out = agent._prune_plan_for_chat(plan)
    assert plan == antes                                              # jamás se muta el plan vivo
    assert "_plan_policy" not in out and "_fidelity_report" not in out and "aggregated_shopping_list" not in out
    d = out["days"][0]
    assert "_day_index" not in d and "_sodium_mg_est" not in d and d["day_name"] == "Miércoles"
    m = d["meals"][0]
    for k in ("_display", "_misalign_trace", "_solver_raw_by_food", "ingredients_raw", "recipe"):
        assert k not in m, k                                          # la receta la sirve consultar_dia_del_plan
    for k in ("name", "cals", "protein", "macros", "ingredients"):
        assert k in m, k
    assert d["meals"][2] == "no-dict" and out["days"][1] == "no-dict"    # lo raro pasa intacto
    assert agent._prune_plan_for_chat("no-dict") == "no-dict"


def test_los_totales_del_dia_van_sumados():
    import agent
    d = agent._prune_plan_for_chat(_plan())["days"][0]
    assert d["totales_dia"] == {"kcal": 1180, "proteina_g": 79, "carbohidratos_g": 136, "grasas_g": 35}
    assert agent._totales_dia_para_chat([{"calories": "300", "protein": None}, "x"]) == {
        "kcal": 300, "proteina_g": 0, "carbohidratos_g": 0, "grasas_g": 0}


def test_la_poda_reduce_el_json_del_prompt():
    import agent
    plan = _plan()
    assert len(json.dumps(agent._prune_plan_for_chat(plan))) < len(json.dumps(plan)) * 0.7


def test_el_prompt_lleva_las_reglas_m_a_p():
    p = _src("prompts/chat_agent.py")
    for lit in ("M. LO PUNTUAL NO ES PERFIL [P1-PLAN-LOTE-77]", "no la guardes como rechazo ni como alergia",
                "N. NADA QUE NO PUEDAS HACER: no ofrezcas recordatorios", "no digas que la ubicaste ni la colocaste",
                "O. IDIOMA ÍNTEGRO", "P. SI SOLO SALUDA", "la próxima comida del plan POR SU NOMBRE", "techo, no meta"):
        assert lit in p, lit
    assert p.index("L. TEMAS DE RIESGO") < p.index("M. LO PUNTUAL NO ES PERFIL") < p.index("P. SI SOLO SALUDA")


def test_el_bloque_del_prompt_lo_dice():
    a = _src("agent.py")
    assert "`totales_dia` son las sumas del día ya hechas: úsalas, no sumes a mano." in a
    assert "Los pasos de cada receta NO vienen aquí: los da la herramienta `consultar_dia_del_plan`." in a


def test_el_stub_de_la_bateria_y_el_documento():
    assert "_nota_comidas_sin_registrar" in _src("scripts/coach_battery/run_battery.py")
    assert "P1-PLAN-LOTE-77" in _src("docs/coach_bateria_2026_09_15.md")
    assert 'P1-PLAN-LOTE-77 · 2026-09-17' in _src("app.py")
