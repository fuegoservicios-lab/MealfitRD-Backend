# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-26 · 2026-09-12] C5 (primera parte) del plan de pendientes: CUL-P1-01 (claras según persona),
CUL-P1-03 (cultura y horario como contexto) y CUL-P1-05 (tiempos y equipo ejecutables).

Lo que se prueba:
  · `culinary_context` lee lo declarado (equipo, básicos con franja, tiempo) y no finge lo que falta;
  · el gate de horario y el autofix del arroz de noche respetan un básico declarado PARA esa franja;
  · el juez recibe la regla de horario como contexto (RD) o neutra (beta), más lo que sabe de la persona;
  · el techo de comidas con huevo honra al básico y una clara de aglutinante no cuenta;
  · V8a (tiempo oculto) y V8b (equipo no disponible) en el escáner; almacenaje ≠ consumo; ml sólo con ml o con densidad;
  · el equipo declarado llega al prompt del día, al selector determinista y a la métrica de personalización;
  · con la política en sombra, la ración de piezas no honrada queda ESCRITA en `relaxations`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]

import culinary_context as cx  # noqa: E402
import culinary_coherence as cc  # noqa: E402
import graph_orchestrator as go  # noqa: E402

_SRC_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")

_CAT = [
    {"name": "Avena", "aliases": ["avena en hojuelas"], "category": "Granos", "prep_methods": ["cocido"]},
    {"name": "Leche descremada", "aliases": ["leche descremada", "leche"], "category": "Lácteos", "prep_methods": ["ninguno"]},
    {"name": "Pechuga de pollo", "aliases": ["pollo"], "category": "Proteínas", "prep_methods": ["plancha"]},
    {"name": "Habichuelas rojas", "aliases": ["habichuelas"], "category": "Legumbres", "prep_methods": ["guisado"]},
    {"name": "Miel", "aliases": ["miel de abeja"], "category": "Otros", "prep_methods": ["ninguno"]},
]
_IDX = cc.build_culinary_index(_CAT)


def _meal(ings, rec, name="Plato", slot="Almuerzo", **extra):
    m = {"meal": slot, "name": name, "ingredients": list(ings), "ingredients_raw": list(ings), "recipe": list(rec)}
    m.update(extra)
    return m


# ─────────────── equipo declarado ───────────────

def test_el_equipo_declarado_se_lee_del_panel_y_ausente_es_none_no_vacio():
    assert cx.declared_equipment({}) is None and cx.declared_equipment({"super_personalization": {"kitchenEquipment": []}}) is None
    d = cx.declared_equipment({"super_personalization": {"kitchenEquipment": ["Horno", "Licuadora", "Sartén / Caldero"]}})
    assert d == {"horno", "licuadora"}
    assert cx.declared_equipment({"health_profile": {"super_personalization": {"kitchenEquipment": ["Airfryer"]}}}) == {"airfryer"}
    assert cx.declared_equipment({"kitchenEquipment": ["Estufa"]}) == set(), "declaró que sólo tiene estufa: se sabe, y no hay horno"
    assert cx.declared_equipment_labels({"kitchenEquipment": ["Olla de presión", "Horno"]}) == ["horno", "olla de presión"]


def test_los_pasos_dicen_que_equipo_exigen_y_que_falta():
    assert cx.equipment_required(["Hornea 25 minutos a 200 °C."]) == {"horno"}
    assert cx.equipment_required(["Licúa la avena con la leche.", "Pásalo por la freidora de aire."]) == {"licuadora", "airfryer"}
    assert cx.equipment_required(["Sofríe la cebolla en la sartén."]) == set()
    m = _meal(["100 g de avena"], ["Hornea 20 min y sirve."])
    assert cx.missing_equipment(m, {"licuadora"}) == ["horno"] and cx.missing_equipment(m, {"horno"}) == []
    assert cx.missing_equipment(m, None) == [], "sin declaración no se acusa: se dice que no se midió"


def test_el_bloque_del_prompt_solo_aparece_con_equipo_declarado(monkeypatch):
    assert cx.equipment_block({}) == ""
    b = cx.equipment_block({"kitchenEquipment": ["Licuadora"]})
    assert "EQUIPO DE COCINA DISPONIBLE" in b and "licuadora" in b and "NO tienes" in b and "horno" in b
    monkeypatch.setenv("MEALFIT_CULINARY_CONTEXT", "0")
    assert cx.equipment_block({"kitchenEquipment": ["Licuadora"]}) == ""


# ─────────────── tiempo oculto ───────────────

def test_las_horas_de_espera_cuentan_y_la_conservacion_no():
    assert cx.hidden_wait_minutes(["Marina el pollo 2 horas en la nevera."])[0] == 120
    assert cx.hidden_wait_minutes(["Remoja las habichuelas toda la noche."])[0] == 480
    assert cx.hidden_wait_minutes(["Sirve caliente. Consume dentro de las 24 horas."])[0] == 0
    assert cx.hidden_wait_minutes(["Refrigera cualquier sobrante hasta 48 horas."])[0] == 0
    assert cx.hidden_wait_minutes(["Hornea 1 h a 180 °C."])[0] == 60
    assert cx.hidden_wait_minutes(["Congela porciones de 140 g hasta 3 meses."])[0] == 0


def test_el_tiempo_declarado_cubre_o_no_cubre_la_espera():
    assert cx.declared_prep_minutes({"prep_time": "40 min"}) == 40 and cx.declared_prep_minutes({"prep_time": "1 h 30 min"}) == 90
    assert cx.declared_prep_minutes({"prep_time": "1 h"}) == 60 and cx.declared_prep_minutes({}) is None
    m = _meal(["150 g de pechuga de pollo"], ["Marina el pollo 2 horas.", "Ásalo a la plancha 8 min."], prep_time="15 min")
    h = cx.check_hidden_time(m)
    assert h and h["espera_min"] == 120 and h["declarado_min"] == 15
    assert cx.check_hidden_time(dict(m, prep_time="3 h")) is None
    sin = cx.check_hidden_time(_meal(["100 g de habichuelas rojas"], ["Remoja las habichuelas toda la noche."]))
    assert sin and sin["declarado_min"] is None, "sin tiempo declarado y con víspera: también sale"
    assert cx.check_hidden_time(_meal(["100 g de avena"], ["Cocina 10 minutos."], prep_time="15 min")) is None


# ─────────────── básicos y horario ───────────────

_FORM_ARROZ_CENA = {"stapleAnchors": [{"name": "Arroz blanco", "slots": ["dinner"], "min_per_7d": 3, "max_per_7d": 7}]}


def test_los_basicos_declarados_traen_su_franja():
    st = cx.declared_staples({"stapleAnchors": [{"name": "Arroz blanco", "slots": ["dinner", "Cena"]}], "stapleFoods": ["Huevo", "Arroz blanco"]})
    assert st == [{"name": "Arroz blanco", "slots": ["cena"]}, {"name": "Huevo", "slots": []}]


def test_un_basico_declarado_para_esa_franja_exime_la_regla_de_horario_y_solo_esa(monkeypatch):
    label = "arroz/locrio/moro (\"arroz de noche\")"
    assert cx.slot_preference_exempt(label, "cena", _FORM_ARROZ_CENA) == "Arroz blanco"
    assert cx.slot_preference_exempt(label, "cena", {"stapleAnchors": [{"name": "Arroz blanco", "slots": ["lunch"]}]}) is None
    assert cx.slot_preference_exempt(label, "cena", {"stapleFoods": ["Arroz blanco"]}) == "Arroz blanco", "sin franja = todas"
    assert cx.slot_preference_exempt("sopón/guiso pesado de noche (sancocho/asopao/mondongo)", "cena", _FORM_ARROZ_CENA) is None
    assert cx.slot_preference_exempt(label, "cena", {}) is None
    monkeypatch.setenv("MEALFIT_CULINARY_CONTEXT", "0")
    assert cx.slot_preference_exempt(label, "cena", _FORM_ARROZ_CENA) is None


def test_el_gate_de_horario_no_acusa_el_arroz_de_noche_que_la_persona_pidio():
    days = [{"day": 1, "meals": [{"meal": "Cena", "name": "Arroz blanco con pollo guisado", "ingredients": ["150 g de arroz blanco"], "recipe": ["Cocina."]}]}]
    sin = go._detect_slot_appropriateness(days, {})
    con = go._detect_slot_appropriateness(days, _FORM_ARROZ_CENA)
    assert any("arroz" in i["label"] for i in sin), "sin declaración, la regla soft de siempre sigue viva"
    assert not any("arroz" in i["label"] for i in con)


def test_el_autofix_del_arroz_de_noche_respeta_al_basico_declarado_y_sus_llamadores_le_pasan_el_formulario():
    days = [{"day": 1, "meals": [{"meal": "Cena", "name": "Arroz blanco con pollo", "ingredients": ["150 g de arroz blanco"], "recipe": ["Cocina."]}]}]
    assert go._night_rice_autofix(days, form_data=_FORM_ARROZ_CENA) == 0
    assert days[0]["meals"][0]["ingredients"] == ["150 g de arroz blanco"]
    assert cx.rice_staple_for_slot(_FORM_ARROZ_CENA, "cena") == "Arroz blanco" and cx.rice_staple_for_slot(_FORM_ARROZ_CENA, "desayuno") is None
    assert _SRC_GO.count("_night_rice_autofix(") >= 4 and _SRC_GO.count("form_data=form_data)") >= 3
    for lit in ('_nr_layer = _night_rice_autofix(plan.get("days") or [], _db, country=_dcl_country, form_data=form_data)',
                "_nr_fixed = _night_rice_autofix(days, country=_apn_country, form_data=form_data)",
                '_nrc_fixed = _night_rice_autofix(plan.get("days", []), compound=True, country=_rpn_country, form_data=form_data)'):
        assert lit in _SRC_GO, lit


# ─────────────── el juez ───────────────

def test_la_regla_de_horario_del_juez_es_contexto_en_rd_y_neutra_en_beta():
    rd = go._culinary_judge_rubric_for_country("DO")
    assert go._JUDGE_SLOT_RULE_DO in rd and "NUNCA van en desayuno ni cena" not in rd
    assert "REGLA DE HORARIO — CONTEXTO, NO DOGMA" in rd and "PASTA de noche es legítima" in rd
    assert "componentes SEPARADOS" in rd and "avena salada" in rd
    for cc_ in ("ES", "MX", "US"):
        beta = go._culinary_judge_rubric_for_country(cc_)
        assert go._JUDGE_SLOT_RULE_NEUTRAL in beta and go._JUDGE_SLOT_RULE_DO not in beta
        assert "dominican" not in go._JUDGE_SLOT_RULE_NEUTRAL.lower()


def test_el_juez_recibe_lo_que_sabe_de_la_persona():
    ctx = cx.judge_context({"stapleAnchors": [{"name": "Arroz blanco", "slots": ["dinner"]}], "cultureProfiles": ["dominican_criolla", "mexican"],
                            "kitchenEquipment": ["Licuadora"], "cookingTime": "30min"}, "DO")["contexto"]
    assert ctx["basicos_declarados"] == ["Arroz blanco (cena)"] and ctx["cocinas_elegidas"] == ["dominican_criolla", "mexican"]
    assert ctx["equipo_disponible"] == ["licuadora"] and ctx["tiempo_de_cocina"] == "30min" and "slot_inapropiado" in ctx["lectura"]
    assert cx.judge_context({}, None) == {}
    assert "async def run_culinary_judge(plan: dict, country: str = \"DO\", form_data=None)" in _SRC_GO
    assert '{"meals": _meals, **_ctx_judge(form_data, country)}' in _SRC_GO
    assert "_cj = await run_culinary_judge(plan, _cj_country, form_data=form_data)" in _SRC_GO


def test_un_componente_separado_en_el_nombre_no_es_una_mezcla(monkeypatch):
    assert go._meal_has_sweet_savory_clash({"name": "Revoltillo de huevo con mango"}) is True
    assert go._meal_has_sweet_savory_clash({"name": "Revoltillo de huevo con mango al lado"}) is False
    assert go._meal_has_sweet_savory_clash({"name": "Brochetas de cerdo con ensalada de lechosa"}) is True
    assert go._meal_has_sweet_savory_clash({"name": "Pescado a la plancha con salsa de mango"}) is False, "proteína + fruta dulce era y sigue siendo aceptable"
    monkeypatch.setenv("MEALFIT_CULINARY_CONTEXT", "0")
    assert go._meal_has_sweet_savory_clash({"name": "Revoltillo de huevo con mango al lado"}) is True
    assert go._SWEET_DOMINANT_FRUITS is cx._SWEET_DOMINANT_FRUITS, "los nombres se re-exportan desde culinary_context"


# ─────────────── el huevo ───────────────

def test_una_clara_de_aglutinante_no_es_un_plato_de_huevo():
    croq = _meal(["200 g de yuca", "1 clara de huevo", "100 g de atún"], ["Mezcla la yuca con el atún y la clara hasta formar una masa."], name="Croquetas de yuca y atún")
    assert cx.egg_is_binder(croq) and cx.egg_pieces(croq) == 1.0
    assert not cx.egg_is_binder(_meal(["3 huevos"], ["Bate los huevos para la masa."], name="Torta de plátano"))
    assert not cx.egg_is_binder(_meal(["4 claras de huevo"], ["Bate las claras."], name="Tortilla de claras"))
    assert not cx.egg_is_binder(_meal(["100 g de avena"], ["Cocina."], name="Avena"))


def test_el_techo_de_comidas_con_huevo_honra_al_basico_declarado_y_nunca_baja(monkeypatch):
    assert cx.egg_meal_cap(28, 7, False, 3) == 3 and cx.egg_meal_cap(28, 7, True, 3) == 7 and cx.egg_meal_cap(12, 3, True, 5) == 5
    monkeypatch.setenv("MEALFIT_CULINARY_CONTEXT", "0")
    assert cx.egg_meal_cap(28, 7, True, 3) == 3
    assert cx.egg_staple_declared({"stapleFoods": ["Clara de huevo"]}) and not cx.egg_staple_declared({})


def test_el_gate_de_variedad_y_su_autofix_cuentan_y_topan_con_el_contexto():
    days = [{"day": i + 1, "meals": [{"meal": "Desayuno", "name": "Huevos revueltos", "ingredients": ["2 huevos"], "recipe": ["Revuelve."]},
                                     {"meal": "Cena", "name": "Croquetas de yuca y atún", "ingredients": ["200 g de yuca", "1 clara de huevo", "100 g de atún"],
                                      "recipe": ["Forma una masa con la yuca, el atún y la clara."]}]} for i in range(8)]
    sin = go.build_variety_report({"days": days})
    con = go.build_variety_report({"days": days}, user_staples={"huevo"})
    assert any(i.startswith("Huevo en 8/16") for i in sin["issues"]), sin["issues"]
    assert not any(i.startswith("Huevo en") for i in con["issues"]), con["issues"]
    assert "_meal_has_egg(m, _sa_egg) and not _ctx_egg_binder(m)" in _SRC_GO and "_ctx_egg_staple(_fd)" in _SRC_GO


# ─────────────── el escáner: V8a, V8b, almacenaje, ml ───────────────

def test_v8a_tiempo_oculto_y_v8b_equipo_en_el_escaner():
    plan = {"days": [{"day": 1, "meals": [
        _meal(["150 g de pechuga de pollo"], ["Marina el pollo 2 horas.", "Hornea 20 min."], prep_time="15 min"),
        _meal(["100 g de avena"], ["Cocina la avena 10 min."], prep_time="15 min"),
    ]}]}
    viol, est = cc.culinary_contract_scan_status(plan, _CAT)
    assert est["contexto"] == {"equipo": "no_declarado"} and {v["check"] for v in viol} >= {"V8a"} and not [v for v in viol if v["check"] == "V8b"]
    viol2, est2 = cc.culinary_contract_scan_status(plan, _CAT, form_data={"kitchenEquipment": ["Estufa"]})
    v8b = [v for v in viol2 if v["check"] == "V8b"]
    assert est2["contexto"] == {"equipo": "declarado"} and [v["food"] for v in v8b] == ["horno"] and v8b[0]["meal_index"] == 0
    plan["form_data"] = {"super_personalization": {"kitchenEquipment": ["Horno"]}}
    viol3, est3 = cc.culinary_contract_scan_status(plan, _CAT)
    assert est3["contexto"]["equipo"] == "declarado" and not [v for v in viol3 if v["check"] == "V8b"], "el equipo viaja dentro del plan persistido"
    assert cc.CHECKS_CAPA1.index("V8b") == cc.CHECKS_CAPA1.index("V8a") + 1 and est["checks"] == list(cc.CHECKS_CAPA1)


def test_el_almacenaje_no_es_consumo_para_los_checks_de_cantidad():
    assert "porciones de 140" not in cc._texto_de_consumo("Sirve 150 g de pollo. Congela porciones de 140 g para la semana.")
    assert cc._texto_de_consumo("Sirve 150 g de pollo.") == "Sirve 150 g de pollo."
    m = _meal(["150 g de pechuga de pollo"], ["Mise en place: pesa 150 g de pollo.", "Congela porciones de 140 g de pollo para otro día."])
    assert not cc._v4_cantidad_inconsistente(1, m, _IDX), "los 140 g congelados no contradicen los 150 g comprados"


def test_ml_se_compara_con_ml_y_cruzar_familias_exige_densidad_respaldada():
    idx = _IDX
    assert cc._v7d_masa_sobrante(1, _meal(["400 ml de avena"], ["Usa 100 ml de avena."]), idx), "misma unidad: comparable sin convertir"
    assert not cc._v7d_masa_sobrante(1, _meal(["400 ml de avena"], ["Usa 120 g de avena."]), idx), "ml de avena contra g de avena: nadie sabe la densidad"
    viol = cc._v7d_masa_sobrante(1, _meal(["420 ml de leche descremada"], ["Añade 250 g de leche descremada."]), idx)
    assert viol and "g" in viol[0]["detail"], "la leche tiene densidad respaldada: se cruza"
    assert cc._densidad_respaldada("Miel") == 1.42 and cc._densidad_respaldada("Avena") is None


# ─────────────── prompt del día, selector determinista, personalización ───────────────

def test_el_prompt_del_dia_lleva_el_equipo_declarado_y_sin_el_es_byte_identico():
    from prompts.day_generator import build_day_assignment_context as b
    sk = {"meal_types": ["Desayuno", "Almuerzo"], "protein_pool": ["Pollo"], "carb_pool": ["Arroz"], "fruit_pool": ["Guineo"]}
    base = b(sk, 1)
    assert b(sk, 1, kitchen_equipment=None) == base and b(sk, 1, kitchen_equipment=[]) == base
    con = b(sk, 1, kitchen_equipment=["licuadora"])
    assert "EQUIPO DE COCINA DISPONIBLE" in con and "licuadora" in con and con != base
    assert "kitchen_equipment=_ctx_equipment_labels(form_data)" in _SRC_GO and _SRC_GO.count("kitchen_equipment=_ctx_equipment_labels(form_data)") == 2
    assert "diet_type=(form_data or {}).get('dietType'), country=_critique_country)" in _SRC_GO


def test_el_selector_determinista_no_ofrece_platos_de_horno_a_quien_no_tiene_horno():
    import dish_registry as dr
    todos = dr.template_candidates("DO", "almuerzo", None, k=200)
    sin_horno = dr.template_candidates("DO", "almuerzo", None, k=200, available_equipment=set())
    assert todos and len(sin_horno) < len(todos)
    pasos = dr.recipe_steps_index("do")
    assert all(not cx.equipment_required(pasos.get(c["template_id"]) or []) for c in sin_horno)
    assert len(dr.template_candidates("DO", "almuerzo", None, k=200, available_equipment={"horno", "licuadora", "airfryer", "parrilla", "batidora", "microondas", "olla_presion", "olla_arrocera", "sandwichera"})) == len(todos)
    src = (_BACKEND / "deterministic_day.py").read_text(encoding="utf-8")
    assert "available_equipment=_equipo" in src and "_de_ctx(_fd)" in src


def test_la_metrica_de_personalizacion_mide_el_equipo_cuando_esta_declarado_y_lo_dice_cuando_no():
    import horizon
    days = [{"day": 1, "meals": [_meal(["100 g de avena"], ["Hornea 20 min."], prep_time="30 min")]}]
    i0, r0, u0 = horizon._equipment_issues(days, {})
    assert (i0, r0) == ([], []) and u0 == [{"check": "equipment", "reason": "equipment_not_declared"}]
    i1, r1, u1 = horizon._equipment_issues(days, {"kitchenEquipment": ["Estufa"]})
    assert r1 == ["equipment"] and u1 == [] and i1[0]["code"] == "equipment_unavailable" and i1[0]["missing"] == ["horno"]
    issues, run, unmeasured = horizon.personalization_issues(days, None, {"food_anchors": []}, form_data={"kitchenEquipment": ["Estufa"]})
    assert "equipment" in run and any(i["code"] == "equipment_unavailable" for i in issues)
    src = (_BACKEND / "horizon.py").read_text(encoding="utf-8")
    assert "form_has_no_equipment_field" not in src


# ─────────────── la política: la ración no honrada, escrita ───────────────

def test_con_la_politica_en_sombra_la_racion_de_piezas_no_honrada_queda_escrita():
    import plan_policy as pp
    compiled = {"enforced": False, "effective": {"food_anchors": [
        {"ingredient_id": "clara_de_huevo", "name": "Clara de huevo", "portion": {"qty": 10, "unit": "unidad"}},
        {"ingredient_id": "huevo", "name": "Huevo", "portion": {"qty": 2, "unit": "unidad"}},
        {"ingredient_id": "arroz", "name": "Arroz blanco", "portion": {"qty": 150, "unit": "g"}}]}, "relaxations": []}
    pp._note_portions_not_enforced(compiled)
    rels = compiled["relaxations"]
    assert len(rels) == 1 and rels[0]["reason_code"] == "portion_cap_default_not_enforced" and rels[0]["action"] == "deferred"
    assert rels[0]["requested"] == 10 and rels[0]["applied"] == 6 and rels[0]["field"] == "food_anchors[clara_de_huevo].portion"
    assert "10" in pp.explain_relaxations(rels)[0] and "6" in pp.explain_relaxations(rels)[0]
    vigente = {"enforced": True, "effective": compiled["effective"], "relaxations": []}
    pp._note_portions_not_enforced(vigente)
    assert vigente["relaxations"] == [], "en vigor, la ración se honra (E5): nada que anotar"
    assert "_note_portions_not_enforced(compiled)" in (_BACKEND / "plan_policy.py").read_text(encoding="utf-8")


# ─────────────── docs, knob, marcador, tope ───────────────

def test_docs_knob_marcador_y_el_god_file_no_subio_el_tope():
    assert len(_SRC_GO.splitlines()) <= 53100
    doc = (_BACKEND / "docs" / "culinary_coherence.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-26" in doc and "V8a" in doc and "V8b" in doc and "CONTEXTO, NO DOGMA" in doc
    knobs = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    assert "| `MEALFIT_CULINARY_CONTEXT` | `True` |" in knobs
    plan = (_BACKEND / "docs" / "plan_pendientes_2026_09_11.md").read_text(encoding="utf-8")
    assert re.search(r"^\| C5 \| 🟡 2026-09-12", plan, re.M) and "P1-PLAN-LOTE-26" in plan
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 26 and m.group(2) >= "2026-09-12"
    src = (_BACKEND / "culinary_context.py").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-26-CULINARY-CONTEXT" in src and re.search(r"def context_enabled\(\).*?MEALFIT_CULINARY_CONTEXT.*?True", src, re.S)
