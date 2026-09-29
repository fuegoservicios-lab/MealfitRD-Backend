# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-815 · 2026-09-29] La fidelidad del tiempo de cocina ya puede fallar en el camino del modelo.

Refutación del informe de fidelidad (29-sep): la dimensión `prep_time` mide el `prep_time` DECLARADO; en el día
determinista ese número sale de la receta (`_prep_time_source`=`receta`), en el camino del modelo lo escribe el propio
modelo y nada lo marcaba. Plan 6594aae1 («Nada» = 10 min): 20 de 20 comidas «cumplían» por lo declarado; la cena del día
2 declara 10 min y sus pasos suman 21. Y el `score` (1 − issues/n_checks, un issue por comida) ponía a 0 un plan con 6
de 12 comidas fuera: por dimensión, 0,83.

Solo instrumento: sello `llm`, contraste por pasos FUERA del score y `score_v2` junto al `score` de siempre.
"""
from __future__ import annotations

import json
import pathlib
import sys
import types

import pytest

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_KNOB = "MEALFIT_FIDELITY_PREP_TIME_MEASURED"

# declara 10 min; los pasos: 4×2 (por lado, extremo bajo) + 3 + 10 = 21 min de fuego
_PASOS_21 = [
    "Mise en place: pica la cebolla y el ají; corta la batata en cubos.",
    "El Toque de Fuego: sella el pescado 4-5 min por lado. Saltea la cebolla y el ají 3 min. Hierve la batata en "
    "cubos 10 min hasta que esté tierna.",
    "Montaje: sirve el pescado sobre la batata con el sofrito.",
]


def _llm(name="Pescado sellado con batata", prep="10 min", pasos=None):
    return {"meal": "Cena", "name": name, "ingredients": ["150 g de pescado blanco", "1 batata"],
            "prep_time": prep, "recipe": list(pasos if pasos is not None else _PASOS_21)}


def _det(name="Pollo guisado", prep="35 min"):
    return {"meal": "Almuerzo", "name": name, "ingredients": ["150 g de Pollo"], "prep_time": prep,
            "_prep_time_source": "receta", "recipe": ["El Toque de Fuego: guisa el pollo 30 min."]}


def _sellado(meal):
    import fidelidad_tiempo as ft
    ft.sellar_llm(meal)
    return meal


# ─────────────────────────────────────────────────────────────── (1) el sello

def test_lo_que_declara_el_modelo_queda_sellado_llm(monkeypatch):
    monkeypatch.delenv(_KNOB, raising=False)
    import fidelidad_tiempo as ft
    m = _llm()
    ft.sellar_llm(m)
    assert m["_prep_time_source"] == "llm"
    d = _det()
    ft.sellar_llm(d)
    assert d["_prep_time_source"] == "receta", "lo que ya trae fuente (receta/técnica) no se toca"
    vacio = {"name": "x", "prep_time": ""}
    ft.sellar_llm(vacio)
    assert "_prep_time_source" not in vacio, "lo vacío es de recipe_library.fill_prep_time, no del modelo"
    monkeypatch.setenv(_KNOB, "false")
    apagado = _llm()
    ft.sellar_llm(apagado)
    assert "_prep_time_source" not in apagado, "knob apagado ⇒ lo de antes, byte a byte"


# ─────────────────────────────────────────────────────────────── (2) el contraste por pasos

def test_declara_diez_con_pasos_de_veintiuno_y_el_score_intacto(monkeypatch):
    import horizon
    monkeypatch.delenv(_KNOB, raising=False)
    days = [{"day": 1, "meals": [_det(), _sellado(_llm())]}]
    rep = horizon.fidelity_report(days, None, {"food_anchors": []}, surface="t", form_data={"cookingTime": "none"})
    pt = rep["prep_time_steps"]
    assert pt["meals"] == 1 and pt["by_source"] == {"llm": 1, "sin_sello": 0}, pt
    assert [(i["meal"], i["declared"], i["prep_time_measured"], i["prep_time_over"]) for i in pt["items"]] == [
        ("Cena", 10, 21, True)], pt["items"]
    assert pt["prep_time_measured"] == 1 and pt["prep_time_over"] == 1 and pt["over_by_declared"] == 0, pt
    # la comida determinista no se contrasta: su número ya es el de su receta
    assert all(i["name"] != "Pollo guisado" for i in pt["items"])

    # el score de siempre, issues, n_checks y checks_run: idénticos con el instrumento apagado
    monkeypatch.setenv(_KNOB, "false")
    days_off = [{"day": 1, "meals": [_det(), _llm()]}]
    rep_off = horizon.fidelity_report(days_off, None, {"food_anchors": []}, surface="t",
                                      form_data={"cookingTime": "none"})
    for k in ("score", "issues", "codes", "n_checks", "checks_run", "unmeasured"):
        assert rep[k] == rep_off[k], k
    assert rep["n_checks"] == len(rep["checks_run"]), "ancla de P1-PLAN-LOTE-3"
    assert [i["meal"] for i in rep["issues"] if i["code"] == "prep_time_over_budget"] == ["Almuerzo"], (
        "fuera del score: el contraste por pasos NO añade issues (el panel y el coach leen `issues`)")
    assert not ({"prep_time_steps", "score_v2", "score_v2_dims"} & set(rep_off)), "apagado ⇒ sin campos nuevos"


def test_sin_tope_mide_pero_no_juzga_y_sin_sello_se_cuenta_aparte(monkeypatch):
    import horizon
    monkeypatch.delenv(_KNOB, raising=False)
    sin_sello = _llm(name="Plato de un camino que borra marcas")          # plan d8b10b05: fuera de alcance
    days = [{"day": 1, "meals": [sin_sello, _sellado(_llm(name="Otro", prep="10 min"))]}]
    rep = horizon.fidelity_report(days, None, {"food_anchors": []}, surface="t", form_data={"cookingTime": "plenty"})
    pt = rep["prep_time_steps"]
    assert pt["budget"] is None and pt["prep_time_over"] == 0 and pt["prep_time_measured"] == 2, pt
    assert pt["by_source"] == {"llm": 1, "sin_sello": 1}, pt
    assert all(i["prep_time_over"] is None for i in pt["items"]), "sin tope no hay «fuera»"
    assert pt["understated"] == 2, "21 de pasos contra 10 declarados: se dice aunque no haya tope"


# ─────────────────────────────────────────────────────────────── (3) score_v2 por dimensión

def test_score_v2_por_dimension_junto_al_de_siempre(monkeypatch):
    import horizon
    monkeypatch.delenv(_KNOB, raising=False)
    days = []
    for d in range(3):
        meals = []
        for k, slot in enumerate(("Desayuno", "Almuerzo", "Merienda", "Cena")):
            fuera = (d * 4 + k) % 2 == 0                                  # 6 de 12 fuera
            meals.append({"meal": slot, "name": f"Plato {d}-{k}", "ingredients": [f"{50 + d * 4 + k} g de Arroz"],
                          "prep_time": "35 min" if fuera else "8 min", "_prep_time_source": "receta"})
        days.append({"day": d + 1, "meals": meals})
    rep = horizon.fidelity_report(days, None, {"food_anchors": []}, surface="t", form_data={"cookingTime": "none"})
    assert rep["score"] == 0.0, "el de siempre: 6 issues / 3 checks"
    assert rep["score_v2_dims"] == {"exact_repeat": 1.0, "ingredient_days": 1.0, "prep_time": 0.5}, rep["score_v2_dims"]
    assert rep["score_v2"] == 0.833
    assert rep["n_checks"] == len(rep["checks_run"])


# ─────────────────────────────────────────────────────────────── (4) pipeline_metrics

def test_la_metrica_lleva_el_contraste_aplanado(monkeypatch):
    import horizon
    monkeypatch.delenv(_KNOB, raising=False)
    days = [{"day": 1, "meals": [_det(), _sellado(_llm())]}]
    rep = horizon.fidelity_report(days, None, {"food_anchors": []}, surface="t", form_data={"cookingTime": "none"})
    escrito = []
    monkeypatch.setitem(sys.modules, "db", types.SimpleNamespace(execute_sql_write=lambda q, p: escrito.append(p)))
    horizon.emit_fidelity_metric("u1", "p1", rep, mode="shadow", gate="warn")
    assert escrito, "la métrica se escribió"
    meta = json.loads(escrito[0][-1])
    assert meta["score"] == rep["score"] and meta["score_v2"] == rep["score_v2"]
    assert (meta["prep_time_llm_meals"], meta["prep_time_measured"], meta["prep_time_over"],
            meta["prep_time_over_declared"]) == (1, 1, 1, 0), meta


# ─────────────────────────────────────────────────────────────── anclas

def test_anclas():
    go = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "fill_prep_time(m, form_data)" in go, "lo vacío sigue yendo a recipe_library (P1-AUDITORIA-ARQ-VERIFICADA)"
    assert '__import__("fidelidad_tiempo").sellar_llm(m)' in go and "P1-PLAN-LOTE-815" in go
    hz = (_BACKEND / "horizon.py").read_text(encoding="utf-8")
    assert '__import__("fidelidad_tiempo").telemetria(' in hz and '__import__("fidelidad_tiempo").metadata_plana(' in hz
    ft = (_BACKEND / "fidelidad_tiempo.py").read_text(encoding="utf-8")
    assert "tooltip-anchor: P1-PLAN-LOTE-815" in ft and _KNOB in ft and "minutos_de_fuego" in ft
    assert _KNOB in (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    f3 = (_BACKEND / "docs" / "plan_policy_f3.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-815" in f3 and "score_v2" in f3 and "prep_time_steps" in f3
