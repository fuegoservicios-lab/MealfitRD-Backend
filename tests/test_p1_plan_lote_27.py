# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-27 · 2026-09-12] C5 (segunda parte, a) del plan de pendientes: CUL-P1-02 (estructura culinaria del plato
como contrato ligero de evaluación) y CUL-P1-04 (la cadena de reparación se mide por etapas).

Lo que se prueba:
  · `dish_structure`: familia por nombre (la «tortilla de trigo» es pan, no huevo), componentes sólo con lo que la lista
    declara en g/ml, y las tres relaciones del backlog con sus umbrales sacados de la biblioteca curada — sin falsos
    positivos sobre las 193 recetas ni sobre el corpus fijo;
  · V9 en el escáner, en la rúbrica del marcador estricto y en la cadena literal;
  · `repair_stage_diff`: fotos de entrada / tras caps / salida, hallazgos NUEVOS por etapa, knob, sin catálogo se dice;
  · los tres ganchos en `db_plans._finalize_plan_data_for_insert` (parser) y una cadena que introduce un defecto lo delata.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]

import dish_structure as ds  # noqa: E402
import repair_stage_diff as rsd  # noqa: E402
import culinary_coherence as cc  # noqa: E402

_CAT = [
    {"name": "Avena", "aliases": ["avena en hojuelas"], "category": "Granos", "prep_methods": ["cocido"]},
    {"name": "Leche descremada", "aliases": ["leche"], "category": "Lácteos", "prep_methods": ["ninguno"]},
    {"name": "Pechuga de pollo", "aliases": ["pollo"], "category": "Proteínas", "prep_methods": ["plancha"]},
    {"name": "Lentejas", "aliases": [], "category": "Legumbres", "prep_methods": ["guisado"]},
    {"name": "Tortilla de trigo", "aliases": ["tortilla de trigo integral"], "category": "Granos", "prep_methods": ["tostado"]},
    {"name": "Espinaca", "aliases": ["espinacas"], "category": "Vegetales", "prep_methods": ["salteado"]},
    {"name": "Tomate", "aliases": ["tomates"], "category": "Vegetales", "prep_methods": ["crudo"]},
    {"name": "Clara de huevo", "aliases": ["claras de huevo", "claras"], "category": "Proteínas", "prep_methods": ["cocido"]},
]


def _meal(name, ings, rec, **extra):
    m = {"meal": "Desayuno", "name": name, "ingredients": list(ings), "ingredients_raw": list(ings), "recipe": list(rec)}
    m.update(extra)
    return m


CREMA = _meal("Crema de lentejas", ["10 g de lentejas", "300 ml de leche descremada"], ["Licúa las lentejas con la leche hasta obtener una crema espesa."])
WRAP = _meal("Wrap de pollo", ["40 g de tortilla de trigo", "250 g de pechuga de pollo", "80 g de lechuga"], ["Rellena la tortilla con el pollo y la lechuga."])
TORT = _meal("Tortilla de claras con espinaca y tomate", ["4 claras de huevo", "60 g de espinaca", "50 g de tomate"],
             ["Bate las claras.", "Vierte en la sartén con la espinaca y el tomate crudos y cuaja."])


# ─────────────── familia y componentes ───────────────

def test_la_familia_sale_del_nombre_y_la_tortilla_de_trigo_es_pan():
    assert ds.familia({"name": "Revoltillo criollo con cebolla"}) == "tortilla_revuelto"
    assert ds.familia({"name": "Wrap de pollo en tortilla de trigo"}) == "tostada_wrap"
    assert ds.familia({"name": "Tortilla de trigo rellena de res"}) == "tostada_wrap"
    assert ds.familia({"name": "Batido de guineo y avena"}) == "batido_crema" and ds.familia({"name": "Pollo guisado con yuca"}) == "guiso"
    assert ds.familia({"name": "Panqueques de avena"}) == "panqueque" and ds.familia({"name": "Ensalada de atún"}) == "ensalada"
    assert ds.familia({"name": "Mangú con salami"}) == "otro" and ds.familia({}) == "otro"
    assert set(ds.FAMILIAS) >= {"tortilla_revuelto", "panqueque", "bowl", "guiso", "ensalada", "tostada_wrap", "batido_crema", "otro"}


def test_los_componentes_solo_pesan_lo_que_la_lista_declara():
    k = ds.componentes(WRAP)
    assert k["soporte"] == ("tortilla de trigo", 40.0) and k["principal"] == ("pechuga de pollo", 250.0)
    assert k["solidos_g"] == 370.0 and k["liquidos_ml"] == 0.0
    k2 = ds.componentes(CREMA)
    assert k2["liquidos_ml"] == 300.0 and k2["solidos_g"] == 10.0
    k3 = ds.componentes(_meal("X", ["2 tomates", "1 taza de avena", "45 g de queso blanco (45 g)"], []))
    assert k3["solidos_g"] == 45.0 and "tomates" in k3["vegetales_agua"], "las piezas sin gramos no pesan; el paréntesis sí"
    assert ds.contract(_meal("X", ["2 tomates", "1 taza de avena"], []))["confianza"] == "baja"
    assert ds.contract(WRAP)["confianza"] == "alta"


# ─────────────── las tres relaciones del backlog ───────────────

def test_la_crema_de_10_g_de_legumbre_y_300_ml_que_se_promete_espesa():
    tipos = [r["tipo"] for r in ds.relaciones(CREMA)]
    assert tipos == ["crema_sin_espesante"]
    r = ds.relaciones(CREMA)[0]
    assert "0.03" in r["detalle"] and "biblioteca curada" in r["evidencia"]
    con_proceso = dict(CREMA, recipe=["Cocina las lentejas con la leche a fuego bajo hasta que espese y licúa."])
    assert ds.relaciones(con_proceso) == [], "un proceso que espesa exime"
    con_espesante = dict(CREMA, ingredients=CREMA["ingredients"] + ["1 guineo maduro"])
    assert ds.relaciones(con_espesante) == [], "un espesante en la lista, aunque venga en piezas, exime"
    poco_liquido = dict(CREMA, ingredients=["10 g de lentejas", "100 ml de leche descremada"])
    assert ds.relaciones(poco_liquido) == [], "con menos de 150 ml no hay «cientos de ml»"


def test_el_wrap_con_relleno_desproporcionado_y_el_wrap_curado():
    assert [r["tipo"] for r in ds.relaciones(WRAP)] == ["wrap_desproporcionado"]
    assert "8.2×" in ds.relaciones(WRAP)[0]["detalle"] or "8.3×" in ds.relaciones(WRAP)[0]["detalle"]
    curado = dict(WRAP, ingredients=["60 g de tortilla de trigo", "120 g de pechuga de pollo", "30 g de lechuga"])
    assert ds.relaciones(curado) == [], "2,5× es la mediana de la biblioteca"
    assert ds.UMBRALES["wrap_relleno_por_pan_max"] == 5.0 and ds.UMBRALES["crema_solidos_por_ml_min"] == 0.2


def test_la_tortilla_de_claras_con_vegetales_de_agua_sin_preparar():
    assert [r["tipo"] for r in ds.relaciones(TORT)] == ["tortilla_vegetales_crudos"]
    prep = dict(TORT, recipe=["Saltea la espinaca y el tomate hasta que suelten el agua y escúrrelos.", "Bate las claras, viértelas encima y cuaja."])
    assert ds.relaciones(prep) == []
    mise = dict(TORT, recipe=["Bate las claras en un bol.", "Sofríe la espinaca y el tomate 3 minutos.", "Vierte las claras y cuaja."])
    assert ds.relaciones(mise) == [], "batir el huevo en el mise en place no es cocinarlo: el sofrito va antes del huevo que se vierte"
    aparte = dict(TORT, ingredients=["4 claras de huevo", "80 g de repollo"], recipe=["Bate las claras y cuaja la tortilla.", "Aliña el repollo y sírvelo al lado."])
    assert ds.relaciones(aparte) == [], "el repollo de la ensalada de al lado no va dentro de la tortilla"


def _corpus():
    fs = sorted(glob.glob(str(_BACKEND / "scripts" / "data" / "culinary_corpus_*.json")))
    if not fs:
        pytest.skip("sin corpus culinario fijo")
    return json.loads(Path(fs[-1]).read_text(encoding="utf-8"))


def test_cero_falsos_positivos_sobre_la_biblioteca_curada_y_el_corpus_fijo():
    import dish_registry as dr
    tpl = dr.templates_by_id("DO")
    pasos = dr.recipe_steps_index("do")
    disparos = []
    for tid, t in tpl.items():
        m = {"name": t["name"], "ingredients": [f"{float(x['grams']):g} g de {x['canonical']}" for x in t["constituents"]],
             "recipe": list(pasos.get(tid) or [])}
        disparos += [(t["name"], r["tipo"]) for r in ds.relaciones(m)]
    assert disparos == [], disparos
    from culinary_corpus import filas_para_medir
    c = _corpus()
    disparos_c = [(m["name"], r["tipo"]) for f in filas_para_medir(c) for d in f["plan_data"]["days"] for m in d["meals"] for r in ds.relaciones(m)]
    assert disparos_c == [], disparos_c


# ─────────────── V9 en el escáner ───────────────

def test_v9_en_el_escaner_la_rubrica_y_la_cadena():
    plan = {"days": [{"day": 1, "meals": [CREMA, WRAP, TORT, _meal("Avena", ["40 g de avena", "200 ml de leche descremada"], ["Cocina la avena con la leche 10 min."])]}]}
    viol, est = cc.culinary_contract_scan_status(plan, _CAT)
    v9 = sorted((v["meal_index"], v["food"]) for v in viol if v["check"] == "V9")
    assert v9 == [(0, "crema_sin_espesante"), (1, "wrap_desproporcionado"), (2, "tortilla_vegetales_crudos")]
    assert all(v["severity"] == "minor" and v["repairable"] is False and "evidencia:" in v["detail"] for v in viol if v["check"] == "V9")
    assert cc.CHECKS_CAPA1[-1] == "V9" and est["checks"] == list(cc.CHECKS_CAPA1)
    src = (_BACKEND / "culinary_coherence.py").read_text(encoding="utf-8")
    assert "out.extend(_v9_estructura(day, meal, index))" in src.split("def culinary_contract_scan(")[1]
    import importlib.util
    spec = importlib.util.spec_from_file_location("culinary_golden_score", _BACKEND / "scripts" / "culinary_golden_score.py")
    gsc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gsc)
    assert gsc.RUBRICA["estructura_del_plato"] == {"V9"} and "V9" in gsc._CODIGOS_DET


# ─────────────── la cadena de reparación se mide ───────────────

def _plan_limpio():
    return {"days": [{"day": 1, "meals": [_meal("Avena con leche", ["40 g de avena", "200 ml de leche descremada"], ["Cocina la avena con la leche 10 min."]),
                                          _meal("Pollo a la plancha", ["150 g de pechuga de pollo"], ["Asa el pollo a la plancha 8 min por lado."], meal="Almuerzo")]}]}


def test_fotos_por_etapa_y_los_hallazgos_nuevos_se_atribuyen_a_su_etapa(caplog):
    plan = _plan_limpio()
    ctx = rsd.start(plan, surface="t", catalog=_CAT)
    assert ctx and ctx["etapas"]["entrada"] == {} and ctx["comidas"] == 2
    # los «caps» convierten el pollo a la plancha en una crema imposible
    plan["days"][0]["meals"][1].update({"name": "Crema de lentejas", "ingredients": ["10 g de lentejas", "300 ml de leche descremada"],
                                        "recipe": ["Licúa hasta obtener una crema espesa."]})
    rsd.mark(ctx, "tras_caps", plan)
    # la cola deja el mise en place pesando más avena de la que la lista compra (V4)
    plan["days"][0]["meals"][0]["recipe"] = ["Mise en place: mide 90 g de avena y 200 ml de leche descremada.", "Cocina 10 min."]
    with caplog.at_level("WARNING"):
        inf = rsd.finish(ctx, plan)
    assert inf["estado"] == "medido" and inf["n_nuevos"] >= 2 and plan["_repair_stage_diff"] is inf
    etapas = {(n["etapa"], n["check"]) for n in inf["nuevos"]}
    assert ("tras_caps", "V9") in etapas and ("salida", "V4") in etapas
    assert inf["etapas"]["tras_caps"].get("V9") == 1 and inf["etapas"]["salida"].get("V9") == 1
    assert any("introdujo" in r.message for r in caplog.records)


def test_una_cadena_que_no_toca_nada_no_tiene_nuevos_y_resuelve_lo_que_arregla():
    plan = _plan_limpio()
    plan["days"][0]["meals"][0]["recipe"] = ["Mise en place: mide 90 g de avena y 200 ml de leche descremada.", "Cocina 10 min."]
    inf = rsd.medir_cadena(plan, lambda p: p["days"][0]["meals"][0].update(recipe=["Mise en place: mide 40 g de avena y 200 ml de leche descremada.", "Cocina 10 min."]), surface="b", catalog=_CAT)
    assert inf["n_nuevos"] == 0 and inf["resueltos"] == 1 and inf["etapas"]["entrada"] == {"V4": 1} and inf["etapas"]["salida"] == {}


def test_sin_catalogo_o_con_el_knob_apagado_se_dice_y_no_se_mide(monkeypatch):
    plan = _plan_limpio()
    monkeypatch.setattr(rsd, "_catalogo", lambda catalog=None: None)
    assert rsd.start(plan, surface="s") is None and plan["_repair_stage_diff"] == {"estado": "sin_catalogo", "surface": "s"}
    monkeypatch.setenv("MEALFIT_REPAIR_STAGE_DIFF", "0")
    assert rsd.start(_plan_limpio(), surface="s", catalog=_CAT) is None
    assert rsd.mark(None, "tras_caps", plan) is None and rsd.finish(None, plan) is None
    assert rsd.start({"days": []}, catalog=_CAT) is None


def test_los_tres_ganchos_viven_en_el_persist_boundary_en_orden():
    src = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    body = src.split("def _finalize_plan_data_for_insert(")[1].split("\ndef ")[0]
    i_start = body.index('_rsd_ctx = _rsd.start(_pd, surface=str(surface or "pre-INSERT"))')
    i_fpc = body.index("_n, _summ = _fpc(_pd[\"days\"]")
    i_caps = body.index('_rsd.mark(_rsd_ctx, "tras_caps", _pd)')
    i_tail = body.index("_rfc_tail_out = _rfc_tail(")
    i_fin = body.index("_rsd.finish(_rsd_ctx, _pd)")
    i_frz = body.index("if _frozen_token is not None:\n                    from graph_orchestrator import restore_past_days")
    assert i_start < i_fpc < i_caps < i_tail < i_fin < i_frz, "entrada antes de la coherencia, foto tras los caps, salida tras el contrato y antes de restaurar los días congelados"
    assert "P1-PLAN-LOTE-27-STAGE-DIFF" in body


# ─────────────── docs, knob, marcador ───────────────

def test_docs_knob_y_marcador():
    doc = (_BACKEND / "docs" / "culinary_coherence.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-27" in doc and "V9" in doc and "_repair_stage_diff" in doc and "mediana 0,90" in doc
    knobs = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    assert "| `MEALFIT_REPAIR_STAGE_DIFF` | `True` |" in knobs
    plan = (_BACKEND / "docs" / "plan_pendientes_2026_09_11.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-27" in plan
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 27 and m.group(2) >= "2026-09-12"
    for f, anchor in (("dish_structure.py", "P1-PLAN-LOTE-27-DISH-STRUCTURE"), ("repair_stage_diff.py", "P1-PLAN-LOTE-27-REPAIR-STAGE-DIFF")):
        assert anchor in (_BACKEND / f).read_text(encoding="utf-8")
