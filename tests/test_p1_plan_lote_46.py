# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-46 · 2026-09-14] Segunda prueba RD del dueño (plan 63eedc6b), tras el lote 45.

  1. Repetición entre planes: el revisor rechazó los 3 intentos por platos del plan de la mañana y el día determinista, que
     no miraba otros planes, armaba lo mismo cada vez (354 s, entregado sin aprobar). Ahora el día prefiere lo no reciente y
     respeta el tope del bloque, y el revisor no reintenta lo que un día determinista arma igual.
  2. El ingrediente que da nombre al plato: guacamole sin aguacate, «maní tostado» sin maní. Los re-trims no lo tocan y, si
     falta, vuelve al 25 % de la plantilla.
  3. Parches: «Cocina queso a la plancha o hervido» y el autofix que puso pollo sobre pasos escritos para un huevo.
"""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]

import culinary_coherence as cc  # noqa: E402
import deterministic_day as dd  # noqa: E402
import graph_orchestrator as go  # noqa: E402
import identidad_plato as idp  # noqa: E402

_REG = {t["template_id"]: t for t in json.loads(
    (_BACKEND / "data" / "registry" / "dish_registry_do_v1.json").read_text(encoding="utf-8"))["templates"]}
_LIB = json.loads((_BACKEND / "data" / "registry" / "recipe_library_do_v1.json").read_text(encoding="utf-8"))["por_id"]
GUACAMOLE = "tpl_e07ba75823ee"      # Guacamole criollo con casabe y huevo duro (aguacate 100 g, el más pesado)
MANI_PASAS = "tpl_74788cfcf505"     # Maní tostado con pasas
PICA_POLLO = "tpl_9dc9eb01b21f"     # Pica pollo casero al horno con tostones al aire (empaniza con harina)

_CAT = [{"name": n, "aliases": a, "category": c, "prep_methods": ["ninguno"]} for n, a, c in (
    ("Aguacate", [], "Frutas"), ("Maní", ["mani"], "Frutos secos"), ("Mantequilla de maní", [], "Grasas"),
    ("Pasas", [], "Frutas"), ("Queso mozzarella", ["mozzarella", "queso"], "Lácteos"), ("Tomate", ["tomates"], "Vegetales"),
    ("Cebolla", [], "Vegetales"), ("Limón", ["limon"], "Frutas"), ("Casabe", [], "Granos"),
    ("Pechuga de pollo", ["pollo"], "Proteínas"), ("Huevo", ["huevos"], "Proteínas"))]
_IDX = cc.build_culinary_index(_CAT)


def _guacamole(nombre="Guacamole criollo con casabe y Pechuga de pollo", **extra):
    ings = ["½ tomate", "20 g de cebolla", "½ limón", "60 g de casabe", "100 g de pechuga de pollo"]
    m = {"name": nombre, "_recipe_source": "library", "_template_id": GUACAMOLE, "_scale_factor": 1.525,
         "ingredients": list(ings), "ingredients_raw": list(ings)}
    m.update(extra)
    return m


# ─────────────── 2. el ingrediente que da nombre al plato ───────────────
def test_la_identidad_es_lo_que_el_nombre_nombra_y_lo_mas_pesado():
    ident = dict(idp.identidad(_guacamole(), _REG[GUACAMOLE]))
    assert "Aguacate" in ident, "el guacamole no dice «aguacate»: entra por ser lo más pesado de la plantilla"
    assert "Casabe" in ident and "Tomate" not in ident
    assert {n for n, _g in idp.identidad({"name": "Maní tostado con pasas"}, _REG[MANI_PASAS])} == {"Maní", "Pasas"}
    assert idp.nombrado("Leche evaporada", "Avena cocida con leche evaporada y maní")
    assert idp.nombrado("Dátiles", "Maní tostado con pasas y dátiles") and not idp.nombrado("Dátiles", "Maní con pasas")


def test_vuelve_lo_que_falta_al_25_por_ciento_con_raw_y_nada_mas():
    m = _guacamole()
    assert idp.restaurar_meal(m, _IDX) == ["+38 g de Aguacate"]          # 0,25 × 100 g × 1,525
    assert m["ingredients"][-1] == "38 g de Aguacate" and m["ingredients_raw"][-1] == "38 g de Aguacate"
    assert m["_identidad_restaurada"] == ["+38 g de Aguacate"]
    assert idp.restaurar_meal(m, _IDX) == [], "idempotente: ya está"
    mani = {"name": "Maní tostado con pasas y queso mozzarella", "_recipe_source": "library", "_template_id": MANI_PASAS,
            "_scale_factor": 1.13, "ingredients": ["40 g de pasas", "75 g de queso"]}
    assert idp.restaurar_meal(mani, _IDX) == ["+8 g de Maní"]


def test_lo_presente_aunque_sea_poco_no_se_toca():
    """La primera versión subía también lo que quedó pequeño: sobre el plan 63eedc6b llevaba la grasa al 116-122 % y el
    salami de 5 a 41 g. Sólo vuelve lo que FALTA."""
    mani = {"name": "Maní tostado con pasas", "_recipe_source": "library", "_template_id": MANI_PASAS,
            "_scale_factor": 1.0, "ingredients": ["5 g de maní", "40 g de pasas"]}
    assert idp.restaurar_meal(mani, _IDX) == [] and mani["ingredients"] == ["5 g de maní", "40 g de pasas"]


def test_no_vuelve_lo_sustituido_ni_lo_que_choca_con_una_alergia_ni_fuera_de_la_biblioteca():
    m = _guacamole("Guacamole criollo con casabe y huevo duro", _protein_autofix_applied="huevo->pollo")
    assert idp.restaurar_meal(m, _IDX) == ["+38 g de Aguacate"], "el huevo lo quitó el autofix a propósito: no vuelve"
    mani = {"name": "Maní tostado con pasas", "_recipe_source": "library", "_template_id": MANI_PASAS,
            "ingredients": ["40 g de pasas"]}
    assert idp.restaurar_meal(dict(mani, ingredients=["40 g de pasas"]), _IDX, allergies=["Maní"]) == []
    assert idp.restaurar_meal(dict(mani, _recipe_source="llm", ingredients=["40 g de pasas"]), _IDX) == []
    assert idp.restaurar_meal(dict(mani, _sodium_autofix_applied="swap_saltcured", ingredients=["40 g de pasas"]), _IDX) == []


def test_restaurar_identidad_recorre_los_dias_y_sin_catalogo_no_hace_nada(monkeypatch):
    dias = [{"day": 1, "meals": [_guacamole(), {"name": "Otro", "ingredients": ["1 huevo"]}]}]
    assert idp.restaurar_identidad(dias, index=_IDX) == 1
    assert idp.restaurar_identidad([{"meals": [_guacamole()]}], index={}) == 0
    monkeypatch.setenv("MEALFIT_DISH_IDENTITY_FLOOR", "false")
    assert idp.restaurar_identidad([{"meals": [_guacamole()]}], index=_IDX) == 0


def test_los_recortes_de_grasa_y_carbohidrato_no_tocan_la_identidad():
    m = _guacamole()
    assert go._identidad_protege(m, "10 g de aguacate") is True
    assert go._identidad_protege(m, "1 cdta de aceite de oliva") is False
    assert go._identidad_protege(dict(m, _recipe_source="llm"), "10 g de aguacate") is False
    for fn in (go._trim_day_fats_to_target, go._trim_day_carbs_to_target):
        assert "_identidad_protege(m, ing)" in inspect.getsource(fn), fn.__name__
    assert "_identidad_protege" not in inspect.getsource(go._close_carb_gap_for_day), "el cerrador SUBE carbos: no aplica"


def test_la_identidad_corre_antes_del_contrato_final_en_los_dos_guardados():
    fin = inspect.getsource(go.finalize_plan_data_coherence)
    i_id = fin.index("_idp.restaurar_identidad(days, db=db, allergies=allergies)")
    assert i_id < fin.index("(P2-C) Contract-lint per-meal en el persist boundary") < fin.index("_recipe_final_contract(days, db)")
    assert i_id < fin.index("if FINALIZE_TRUTHUP_ALL_ENABLED:"), "el truth-up final re-mide con la línea ya puesta"
    dbp = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    assert dbp.index("_idp_tail.restaurar_identidad(") < dbp.index("_rfc_tail_out = _rfc_tail("), "antes del contrato final de la cola"


# ─────────────── 3. parches ───────────────
def test_el_cerrador_no_manda_cocinar_un_queso():
    for nm in ("queso", "queso mozzarella", "mozzarella", "queso gouda"):
        paso = go._closer_protein_step_text(nm, False)
        assert "hervid" not in paso.lower() and "plancha" not in paso.lower(), (nm, paso)
    assert "plancha" in go._closer_protein_step_text("Pechuga de pollo", False), "lo que sí se cocina conserva su paso"
    assert "_CHEESE_WORDING_HINT" in inspect.getsource(go)


def test_el_autofix_no_reescribe_recetas_congeladas():
    assert go._receta_congelada({"_recipe_source": "library"}) is True
    assert go._receta_congelada({"_recipe_source": "llm"}) is False
    dias = [{"day": 3, "meals": [
        {"meal": "Almuerzo", "name": "Arroz con lentejas y huevo duro", "_recipe_source": "library",
         "ingredients": ["3 huevos", "100 g de lentejas"], "recipe": ["Hierve los huevos 10 minutos y pélalos."]},
        {"meal": "Cena", "name": "Guacamole criollo con casabe y huevo duro", "_recipe_source": "library",
         "ingredients": ["2 huevos", "60 g de aguacate"], "recipe": ["Hierve el huevo, pélalo y córtalo en tajadas."]}]}]
    antes = json.dumps(dias, ensure_ascii=False, sort_keys=True)
    assert go._protein_repeat_autofix(dias, form_data={}, db=None) == 0
    assert json.dumps(dias, ensure_ascii=False, sort_keys=True) == antes
    assert inspect.getsource(go._protein_repeat_autofix).count("_receta_congelada(") >= 4


def test_empanizar_es_usar_la_harina():
    pasos = list(_LIB[PICA_POLLO]["pasos"])
    assert any("por la harina" in p for p in pasos)
    dias = [{"meals": [{"name": "Pica pollo casero al horno con tostones al aire", "recipe": list(pasos),
                        "ingredients": ["150 g de pechuga de pollo", "20 g de harina de trigo"],
                        "ingredients_raw": ["150 g de pechuga de pollo", "20 g de harina de trigo"]}]}]
    assert go._strip_raw_flour_compliance_bolt(dias) == 0
    assert "20 g de harina de trigo" in dias[0]["meals"][0]["ingredients"]
    tostadas = [{"meals": [{"name": "Tostadas con mantequilla de maní", "recipe": ["Tuesta el pan.", "Unta la mantequilla."],
                            "ingredients": ["2 rebanadas de pan", "20 g de harina de trigo"]}]}]
    assert go._strip_raw_flour_compliance_bolt(tostadas) == 1, "la harina suelta sin masa sigue saliendo"


# ─────────────── 1. repetición entre planes y en el bloque ───────────────
def test_el_revisor_no_cuenta_los_repetidos_de_dias_deterministas(monkeypatch):
    dias = [{"_day_source": "deterministic", "meals": [{"name": "A"}]}, {"meals": [{"name": "B"}]}]
    assert go._anti_repeticion_sin_deterministas(["A", "B", "C"], dias) == ["B", "C"]
    monkeypatch.setattr(go, "ANTI_REPETITION_SKIP_DETERMINISTIC", False)
    assert go._anti_repeticion_sin_deterministas(["A", "B", "C"], dias) == ["A", "B", "C"]
    assert "_anti_repeticion_sin_deterministas(filtered_repeated, days)" in inspect.getsource(go.review_plan_node)


def _plan_huevo_x2(fuente):
    dia = {"day": 1, "meals": [
        {"meal": "Desayuno", "name": "Revoltillo de huevo con plátano", "ingredients": ["2 huevos", "150 g de plátano"]},
        {"meal": "Cena", "name": "Huevos hervidos con yuca", "ingredients": ["2 huevos", "200 g de yuca"]}]}
    if fuente:
        dia["_day_source"] = fuente
    return {"days": [dia]}


def test_el_gate_de_proteina_repetida_no_rechaza_un_dia_determinista(monkeypatch):
    rep = go.build_variety_report(_plan_huevo_x2("deterministic"))
    assert rep["same_day_protein_repeats"] == 1 and rep["same_day_protein_repeats_modelo"] == 0
    assert not any("MISMA PROTEÍNA" in i for i in go._variety_repeat_gate_issues(rep))
    rep_llm = go.build_variety_report(_plan_huevo_x2(None))
    assert rep_llm["same_day_protein_repeats_modelo"] == 1
    assert any("MISMA PROTEÍNA" in i for i in go._variety_repeat_gate_issues(rep_llm)), "el día del modelo sigue juzgado"
    monkeypatch.setattr(go, "VARIETY_GATE_SKIP_DETERMINISTIC", False)
    assert any("MISMA PROTEÍNA" in i for i in go._variety_repeat_gate_issues(rep))


def test_los_planes_recientes_se_leen_una_vez_y_solo_en_el_primer_bloque(monkeypatch):
    import db
    llamadas = []

    def _fake(q, params=None, **kw):
        llamadas.append(params)
        return [{"plan_data": {"days": [{"meals": [{"_template_id": "tpl_x"},
                                                   {"name": "Pica pollo casero al horno con tostones al aire"}]}]}},
                {"plan_data": json.dumps({"_archived_days": [{"meals": [{"_template_id": "tpl_y"}]}], "days": []})}]

    monkeypatch.setattr(db, "execute_sql_query", _fake)
    dd._RECIENTES_CACHE.clear()
    got = dd._plantillas_de_planes_recientes([], {}, "u1", _REG, 0)
    assert {"tpl_x", "tpl_y", PICA_POLLO} <= got
    assert llamadas and llamadas[0][1] == dd._PLANES_RECIENTES == 3, "el mismo número que el revisor"
    assert dd._plantillas_de_planes_recientes([], {}, "u1", _REG, 0) == got and len(llamadas) == 1, "caché"
    assert dd._plantillas_de_planes_recientes([], {}, "u1", _REG, 3) == set(), "un bloque que continúa no mira otros planes"
    assert dd._plantillas_de_planes_recientes([], {}, "guest", _REG, 0) == set()
    dd._RECIENTES_CACHE.clear()


def test_el_tope_del_bloque_y_su_conteo():
    assert dd._max_repeticion_bloque({}, {"days_count": 3}) == 1
    assert dd._max_repeticion_bloque({}, {"days_count": 7}) == 2
    assert dd._max_repeticion_bloque({}, None) is None
    mem = [{"meals": [{"_template_id": "a"}, {"_template_id": "b"}]}, {"meals": [{"_template_id": "a"}]},
           {"_persistido": True, "meals": [{"_template_id": "a"}]}]
    assert dd._conteo_bloque(mem) == {"a": 2, "b": 1}, "los días persistidos no son del bloque"


def test_la_familia_va_a_la_comida_principal_y_la_puerta_de_proteina_tiene_su_knob(monkeypatch):
    assert dd._familia_del_blueprint({"days": [{"day_index": 1, "protein": "Pescado"}]}, 1) == "Pescado"
    assert dd._familia_del_blueprint({"days": [{"day_index": 1, "protein": "Pescado"}]}, 2) is None
    src = inspect.getsource(dd.build_day_for_skeleton)
    for ancla in ("_fam_solo_principal", "_franja_principal", "_puerta_proteina",
                  "_plantillas_de_planes_recientes(memoria, _fd, _uid, por_id, _offset)",
                  "_max_repeticion_bloque(_fd, _sl)", "_conteo_bloque(memoria, por_id)"):
        assert ancla in src, ancla
    monkeypatch.delenv("MEALFIT_DETERMINISTIC_DAY_SAME_DAY_VARIETY", raising=False)
    assert dd._variedad_del_dia_on() is False, "la puerta de base ligera sigue apagada: sólo se enciende la de proteína"
    assert dd._knob_on("MEALFIT_DETERMINISTIC_DAY_SAME_DAY_PROTEIN") is True


# ─────────────── knobs, docs, marcador ───────────────
def test_knobs_docs_y_marcador():
    go_src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    for knob in ('_env_bool("MEALFIT_ANTI_REPETITION_SKIP_DETERMINISTIC", True)',
                 '_env_bool("MEALFIT_VARIETY_GATE_SKIP_DETERMINISTIC", True)',
                 '_env_bool("MEALFIT_PROTEIN_AUTOFIX_SKIP_LIBRARY", True)'):
        assert knob in go_src, knob
    knobs_doc = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    for k in ("MEALFIT_DETERMINISTIC_DAY_RECENT_PLANS", "MEALFIT_DETERMINISTIC_DAY_BLOCK_REPEAT",
              "MEALFIT_DETERMINISTIC_DAY_FAMILY_MAIN_SLOT_ONLY", "MEALFIT_DETERMINISTIC_DAY_SAME_DAY_PROTEIN",
              "MEALFIT_ANTI_REPETITION_SKIP_DETERMINISTIC", "MEALFIT_VARIETY_GATE_SKIP_DETERMINISTIC",
              "MEALFIT_PROTEIN_AUTOFIX_SKIP_LIBRARY", "MEALFIT_DISH_IDENTITY_FLOOR"):
        assert k in knobs_doc, k
    for doc in ("knobs_reference.md", "plan_pendientes_2026_09_11.md", "deterministic_day.md", "culinary_coherence.md",
                "plan_agente_lotes_38_43_2026_09_14.md"):
        assert "P1-PLAN-LOTE-46" in (_BACKEND / "docs" / doc).read_text(encoding="utf-8"), doc
    for f, ancla in (("identidad_plato.py", "P1-PLAN-LOTE-46-IDENTIDAD"),
                     ("graph_orchestrator.py", "P1-PLAN-LOTE-46-ANTI-REPETICION"),
                     ("graph_orchestrator.py", "P1-PLAN-LOTE-46-AUTOFIX-CONGELADA"),
                     ("graph_orchestrator.py", "P1-PLAN-LOTE-46-QUESO"),
                     ("graph_orchestrator.py", "P1-PLAN-LOTE-46-RECORTES-RESPETAN-IDENTIDAD"),
                     ("graph_orchestrator.py", "P1-PLAN-LOTE-46-IDENTIDAD-FINALIZE"),
                     ("db_plans.py", "P1-PLAN-LOTE-46-IDENTIDAD-COLA"),
                     ("deterministic_day.py", "P1-PLAN-LOTE-46-PLANES-RECIENTES"),
                     ("deterministic_day.py", "P1-PLAN-LOTE-46-FAMILIA-PRINCIPAL")):
        assert ancla in (_BACKEND / f).read_text(encoding="utf-8"), (f, ancla)
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 46
