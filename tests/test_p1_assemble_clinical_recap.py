"""[P1-ASSEMBLE-CLINICAL-RECAP · 2026-08-02] El assemble DEBE re-aplicar los caps clinicos
DM2/bariatrico DESPUES del refinador global (_rdi) y del recheck post-quantize.

Motivo (audit solver+seeder v7, gap gemelo de P1-CARBFLOOR-CLINICAL-RECAP): en
`assemble_plan_node`, `cap_dm2_high_gi_portions` y `cap_bariatric_portions` corren ANTES del
refinador global (`_rdi` alias de `refine_day_portions_integer`) y del recheck post-quantize
(`_rebalance_day_macros_to_target` bidireccional [0.3, 2.5]). Ambos pases son macro-agnosticos
a la clinica y pueden devolver gramos a una linea que la capa clinica ya habia recortado (una
batata DM2 capada a 150 g puede volver a 300-375 g) -- el propio docstring de
`reapply_clinical_portion_caps` documenta este modo de fallo para las superficies de update;
este test ancla el espejo del lado assemble/form-gen, donde nada re-capeaba en el happy path.

`reapply_clinical_portion_caps` es idempotente (`if grams <= cap: continue`) y no-op sin
condicion clinica (`form_data` vacio) -- llamarlo incondicionalmente tras el bloque
refine+postquantize-recheck es seguro y barato.
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as _f:
    SRC = _f.read()


# Anchor CONTIGUO real en el codigo. El anchor propuesto originalmente en el brief
# ("_rebalance_day_macros_to_target(_rq_meals") tiene un salto de linea entre el "(" y
# "_rq_meals" en el callsite real -- nunca hace match con `str.index`, así que se ancla en la
# condición completa que sí es una sola línea contigua.
# tooltip-anchor: P1-ASSEMBLE-CLINICAL-RECAP
_REFINE_IDX = SRC.index("if _drift and _rebalance_day_macros_to_target(")
_TAIL = SRC[_REFINE_IDX:_REFINE_IDX + 6000]


def test_recap_clinico_corre_despues_del_refine_del_assemble():
    recap_idx = _TAIL.find("reapply_clinical_portion_caps")
    assert recap_idx != -1 and "P1-ASSEMBLE-CLINICAL-RECAP" in _TAIL, (
        "El assemble no re-aplica caps clinicos despues del refine/postquantize-recheck: "
        "una batata DM2 capada a 150g puede volver a 300-375g y persistirse en form-gen."
    )


def test_recap_invoca_al_helper_ssot_con_form_data_y_surface_propio():
    """El callsite (dentro de la ventana post-refine, NO la definición del helper que aparece
    antes en el archivo) debe pasar el plan (`result`) + `form_data` SERVER-SIDE del assemble y
    un `surface` propio distinguible de las superficies de update -- si el surface fuera
    genérico o `form_data` un literal, la telemetría de `reapply_clinical_portion_caps` no
    podría diferenciar este callsite de swap-persist/chat-modify. Ventana amplia (no línea
    única): la llamada real envuelve argumentos en varias líneas."""
    i = _TAIL.index("reapply_clinical_portion_caps(")
    seg = _TAIL[i:i + 200]
    assert "result" in seg and "form_data" in seg
    assert "surface=\"assemble_post_refine\"" in seg or "surface='assemble_post_refine'" in seg


def test_recap_es_noop_sin_condicion_clinica():
    from graph_orchestrator import reapply_clinical_portion_caps
    plan = {"days": [{"meals": [{"ingredients": ["300 g de batata"], "ingredients_raw": ["300 g de batata"]}]}]}
    reapply_clinical_portion_caps(plan, {}, db={}, surface="assemble_post_refine")
    assert plan["days"][0]["meals"][0]["ingredients"] == ["300 g de batata"]


def test_knob_assemble_clinical_recap_default_true():
    import graph_orchestrator as go
    assert go.ASSEMBLE_CLINICAL_RECAP_ENABLED is True
