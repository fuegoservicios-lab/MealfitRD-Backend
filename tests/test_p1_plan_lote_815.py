# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-815 · 2026-09-29] La fidelidad del tiempo de cocina ya puede fallar en el camino del modelo.

Refutación del informe de fidelidad (29-sep): la dimensión `prep_time` mide el `prep_time` DECLARADO; en el día
determinista ese número sale de la receta (`_prep_time_source`=`receta`), en el camino del modelo lo escribe el propio
modelo y nada lo marcaba. Plan 6594aae1 («Nada» = 10 min): 20 de 20 comidas «cumplían» por lo declarado; la cena del día
2 declara 10 min y sus pasos suman 21. Y el `score` (1 − issues/n_checks, un issue por comida) ponía a 0 un plan con 6
de 12 comidas fuera: por dimensión, 0,83.

Solo instrumento: sello `llm`, contraste por pasos FUERA del score y `score_v2` junto al `score` de siempre.

Ronda de corrección (revisor): el día de contingencia (`_build_fallback_day`, «15 min» de plantilla, `_day_fallback`)
salía sellado `llm` al pasar por `assemble_plan_node`; `sin_sello` no aparece dentro del pipeline (assemble sella todo
lo que trae minutos); y un fallo del instrumento no puede tirar la fila de `pipeline_metrics` de siempre.
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
    sin_sello = _llm(name="Plato de un plan guardado antes del lote")     # solo FUERA del pipeline (replay)
    days = [{"day": 1, "meals": [sin_sello, _sellado(_llm(name="Otro", prep="10 min"))]}]
    rep = horizon.fidelity_report(days, None, {"food_anchors": []}, surface="t", form_data={"cookingTime": "plenty"})
    pt = rep["prep_time_steps"]
    assert pt["budget"] is None and pt["prep_time_over"] == 0 and pt["prep_time_measured"] == 2, pt
    assert pt["by_source"] == {"llm": 1, "sin_sello": 1}, pt
    assert all(i["prep_time_over"] is None for i in pt["items"]), "sin tope no hay «fuera»"
    assert pt["understated"] == 2, "21 de pasos contra 10 declarados: se dice aunque no haya tope"


def _contingencia(n=2):
    import graph_orchestrator as go
    fb = go._build_fallback_day({"target_calories": 2000, "macros": {}}, n, frozenset(), form_data={})
    fb["_day_fallback"] = True                                            # lo que hace generate_days_parallel
    return fb


def test_el_dia_de_contingencia_no_es_palabra_del_modelo(monkeypatch):
    """Revisor, cambio 1: `_build_fallback_day` pone «15 min» a mano (P1-AUDITORIA-ARQ-VERIFICADA lo llamó relleno).
    Ni se sella `llm` ni se contrasta con sus pasos: con «Nada» salían over_by_declared=3 y prep_time_over=2."""
    import fidelidad_tiempo as ft
    monkeypatch.delenv(_KNOB, raising=False)
    fb = _contingencia()
    for m in fb["meals"]:
        ft.sellar_llm(m, fb)
    assert [m.get("_prep_time_source") for m in fb["meals"]] == [None] * len(fb["meals"]), fb["meals"]
    llm = {"day": 1, "meals": [_llm()]}
    for m in llm["meals"]:
        ft.sellar_llm(m, llm)
    assert llm["meals"][0]["_prep_time_source"] == "llm", "un día del modelo se sigue sellando con el día al lado"
    pt = ft.contraste_por_pasos([llm, fb], {"cookingTime": "none"})
    assert (pt["meals"], pt["by_source"], pt["over_by_declared"], pt["prep_time_measured"], pt["prep_time_over"]) == (
        1, {"llm": 1, "sin_sello": 0}, 0, 1, 1), pt
    assert pt["fallback_meals"] == len(fb["meals"]) == 3, "la contingencia se cuenta aparte, no se contrasta"


def test_assemble_sella_al_modelo_y_no_a_la_contingencia(monkeypatch):
    """El sello, ejecutado de verdad en `assemble_plan_node` (no solo el ancla de texto). Sin DB ni LLM: corre en local."""
    import asyncio
    import db_core
    import graph_orchestrator as go
    monkeypatch.delenv(_KNOB, raising=False)
    for mod in (db_core, go):                                             # la métrica de assemble: a ningún sitio
        monkeypatch.setattr(mod, "execute_sql_write", lambda *a, **k: None, raising=False)
    vacio = _llm(name="Sin minutos", prep="")
    state = {"nutrition": {"target_calories": 2000, "goal_label": "Mantener",
                           "macros": {"protein_g": 150, "carbs_g": 200, "fats_g": 60,
                                      "protein_str": "150g", "carbs_str": "200g", "fats_str": "60g"}},
             "form_data": {"cookingTime": "none"},
             "plan_result": {"days": [{"day": 1, "meals": [_llm(), _det(), vacio]}, _contingencia()]}}
    days = asyncio.run(go.assemble_plan_node(state))["plan_result"]["days"]
    fuentes = {m["name"]: m.get("_prep_time_source") for d in days for m in d["meals"]}
    assert fuentes["Pescado sellado con batata"] == "llm", fuentes
    assert fuentes["Pollo guisado"] == "receta", "lo que ya trae fuente no se toca"
    assert fuentes["Sin minutos"] in ("registry", "unknown"), "lo vacío sigue en recipe_library.fill_prep_time"
    fb = [d for d in days if d.get("_day_fallback")]
    assert fb and all(m.get("_prep_time_source") is None for m in fb[0]["meals"]), fb
    pt = __import__("fidelidad_tiempo").contraste_por_pasos(days, {"cookingTime": "none"})
    assert pt["by_source"]["sin_sello"] == 0, "dentro del pipeline sin_sello no aparece: assemble sella todo"
    monkeypatch.setenv(_KNOB, "false")
    state["plan_result"]["days"] = [{"day": 1, "meals": [_llm()]}]
    off = asyncio.run(go.assemble_plan_node(state))["plan_result"]["days"]
    assert "_prep_time_source" not in off[0]["meals"][0], "knob apagado ⇒ sin sello"


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


def test_la_fila_de_siempre_no_cae_si_el_instrumento_no_importa(monkeypatch):
    """Revisor (no bloqueante): el `__import__` del instrumento estaba fuera de su propio try; si fallaba, el `except`
    de fuera tiraba la fila ENTERA de `pipeline_metrics` (score, codes, n_checks…), no solo los campos nuevos."""
    import horizon
    monkeypatch.delenv(_KNOB, raising=False)
    days = [{"day": 1, "meals": [_det(), _sellado(_llm())]}]
    rep = horizon.fidelity_report(days, None, {"food_anchors": []}, surface="t", form_data={"cookingTime": "none"})
    escrito = []
    monkeypatch.setitem(sys.modules, "db", types.SimpleNamespace(execute_sql_write=lambda q, p: escrito.append(p)))
    monkeypatch.setitem(sys.modules, "fidelidad_tiempo", None)          # ⇒ __import__ lanza ImportError
    horizon.emit_fidelity_metric("u1", "p1", rep, mode="shadow", gate="warn")
    assert escrito, "la fila de siempre se escribe aunque el instrumento nuevo no esté"
    meta = json.loads(escrito[0][-1])
    assert meta["score"] == rep["score"] and meta["n_checks"] == rep["n_checks"] and "score_v2" not in meta, meta


# ─────────────────────────────────────────────────────────────── anclas

def test_anclas():
    go = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "fill_prep_time(m, form_data)" in go, "lo vacío sigue yendo a recipe_library (P1-AUDITORIA-ARQ-VERIFICADA)"
    assert '__import__("fidelidad_tiempo").sellar_llm(m, d)' in go and "P1-PLAN-LOTE-815" in go
    hz = (_BACKEND / "horizon.py").read_text(encoding="utf-8")
    assert '__import__("fidelidad_tiempo").telemetria(' in hz and '__import__("fidelidad_tiempo").metadata_plana(' in hz
    ft = (_BACKEND / "fidelidad_tiempo.py").read_text(encoding="utf-8")
    assert "tooltip-anchor: P1-PLAN-LOTE-815" in ft and _KNOB in ft and "minutos_de_fuego" in ft
    assert _KNOB in (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    f3 = (_BACKEND / "docs" / "plan_policy_f3.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-815" in f3 and "score_v2" in f3 and "prep_time_steps" in f3
    # revisor, cambio 2: `sin_sello` NO se ve dentro del pipeline (assemble sella todo lo que trae minutos)
    for doc in (ft, f3):
        assert "_day_fallback" in doc and "fuera del pipeline" in doc
        assert "para que ese camino se vea en la métrica" not in doc and "sus comidas salen como `sin_sello`" not in doc
