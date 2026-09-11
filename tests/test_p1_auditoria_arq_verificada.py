# -*- coding: utf-8 -*-
"""[P1-AUDITORIA-ARQ-VERIFICADA · 2026-09-11] Verificación independiente de la auditoría de arquitectura
(`BioBoros_auditoria_arquitectura.md`, revisión auditada baf4163) y las correcciones que se justificaron.

Cada test lleva el número del hallazgo (H1…H16) que verifica. Expresan el comportamiento ESPERADO del
producto, no el defecto que había: ninguna aserción aquí fija un error como especificación.

Informe: `docs/auditoria_arquitectura_verificacion_2026_09_11.md`.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_REG = _BACKEND / "data" / "registry" / "dish_registry_do_v1.json"


@pytest.fixture(scope="module")
def dd():
    import deterministic_day
    return deterministic_day


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# Un mundo pequeño y determinista para armar días de verdad sin base ni LLM.
# ─────────────────────────────────────────────────────────────────────────────────────────────────

_CATALOGO = {
    "Pollo": {"name": "Pollo", "kcal_per_100g": 165, "protein_g_per_100g": 31, "carbs_g_per_100g": 0,
              "fats_g_per_100g": 3.6, "sodium_mg_per_100g": 74},
    "Arroz": {"name": "Arroz", "kcal_per_100g": 130, "protein_g_per_100g": 2.7, "carbs_g_per_100g": 28,
              "fats_g_per_100g": 0.3, "sodium_mg_per_100g": 1},
    "Arenque": {"name": "Arenque", "kcal_per_100g": 200, "protein_g_per_100g": 23, "carbs_g_per_100g": 0,
                "fats_g_per_100g": 12, "sodium_mg_per_100g": 900},
    "Yuca": {"name": "Yuca", "kcal_per_100g": 160, "protein_g_per_100g": 1.4, "carbs_g_per_100g": 38,
             "fats_g_per_100g": 0.3},   # sin sodio A PROPÓSITO: es el «sin dato»
    "Sal": {"name": "Sal", "kcal_per_100g": 0, "protein_g_per_100g": 0, "carbs_g_per_100g": 0,
            "fats_g_per_100g": 0, "sodium_mg_per_100g": 38758},
}


def _tpl(tid, nombre, slots, protein, constituents, sodio=None, prep=("receta", 45)):
    return {"template_id": tid, "name": nombre, "slots": list(slots), "protein": protein, "status": "ok",
            "constituents": [{"name": n, "grams": g} for n, g in constituents],
            "nutrition_per_serving": {"sodium_mg": sodio},
            "logistics": {"prep_minutes_source": prep[0], "prep_minutes_est": prep[1]}}


_POR_ID = {
    "tpl_a": _tpl("tpl_a", "Pollo con arroz (a)", ["desayuno", "almuerzo", "cena", "merienda"], "pollo",
                  [("Pollo", 200), ("Arroz", 200), ("Sal", 1)], sodio=150),
    "tpl_b": _tpl("tpl_b", "Pollo con arroz (b)", ["desayuno", "almuerzo", "cena", "merienda"], "pollo",
                  [("Pollo", 190), ("Arroz", 210), ("Sal", 1)], sodio=140, prep=("tecnica", 30)),
    "tpl_c": _tpl("tpl_c", "Arenque con yuca", ["almuerzo", "cena"], "pescado",
                  [("Arenque", 150), ("Yuca", 250)], sodio=1350),
    # una merienda de ~200 kcal: con 5 o 6 comidas la merienda pesa el 10-12 % del día
    "tpl_m": _tpl("tpl_m", "Arroz con leche pequeño", ["merienda"], "none",
                  [("Arroz", 150)], sodio=5),
}

# Un día de una sola franja recibe el 100 % de las kcal (el reparto se normaliza a 1,0): el objetivo
# de estos tests de una franja es de 600 kcal para que el plato de ~590 kcal entre en banda.
_NUT1 = {"target_calories": "600 kcal", "macros": {"protein": "60g", "carbs": "60g", "fats": "15g"}}


@pytest.fixture
def mundo(dd, monkeypatch):
    """Registry, catálogo y receta falsos; el knob encendido; el escáner del plato en verde."""
    import dish_registry as dr
    import shopping_calculator as sc
    import recipe_library as rl

    llamadas = []

    def _template_candidates(country, slot, family=None, **kw):
        llamadas.append({"country": country, "slot": slot, "family": family, "kw": dict(kw)})
        return [{"template_id": tid, "name": t["name"]} for tid, t in _POR_ID.items() if slot in t["slots"]]

    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY", "true")
    monkeypatch.setattr(dr, "template_candidates", _template_candidates)
    monkeypatch.setattr(dr, "templates_by_id", lambda country: dict(_POR_ID))
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: list(_CATALOGO.values()))
    monkeypatch.setattr(rl, "recipe_for_dish_name",
                        lambda name, country="DO": ["Cuece el pollo con tres tazas de agua.", "Sirve."])
    monkeypatch.setattr(dd, "verifica_comida", lambda meal, fd, cat: [])
    return llamadas


_NUT = {"target_calories": "2000 kcal", "macros": {"protein": "150g", "carbs": "200g", "fats": "60g"}}


def _esqueleto(meal_types, pool=("Pechuga de pollo",)):
    """La forma REAL: `DaySkeletonModel` (schemas.py), sin `slots` ni `protein`."""
    return {"day": 1, "assigned_technique": "Guisado", "protein_pool": list(pool), "carb_pool": ["Arroz"],
            "fruit_pool": ["Guineo"], "veggie_pool": [], "meal_types": list(meal_types),
            "breakfast_category": "Libre", "brief_concept": "x"}


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# H1 · el contrato del esqueleto: `meal_types`/`protein_pool`, no `slots`/`protein`
# ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_h1_las_franjas_salen_de_meal_types_con_el_reparto_ssot(dd):
    from nutrition_calculator import MEAL_SLOT_SPLITS
    fr = dd._franjas_del_dia(_esqueleto(["Desayuno", "Almuerzo", "Cena"]))
    assert [e for e, _, _ in fr] == ["Desayuno", "Almuerzo", "Cena"], "la etiqueta del esqueleto se conserva"
    assert [s for _, s, _ in fr] == ["desayuno", "almuerzo", "cena"]
    esperado = MEAL_SLOT_SPLITS[3]
    assert [round(f, 3) for _, _, f in fr] == [esperado["desayuno"], esperado["almuerzo"], esperado["cena"]]
    assert abs(sum(f for _, _, f in fr) - 1.0) < 1e-6


def test_h1_seis_comidas_clinicas_reciben_seis_franjas(dd):
    from nutrition_calculator import MEAL_SLOT_SPLITS, meal_types_for_count
    seis = meal_types_for_count(6)
    fr = dd._franjas_del_dia(_esqueleto(seis))
    assert len(fr) == 6 and [e for e, _, _ in fr] == seis
    assert [s for _, s, _ in fr] == ["desayuno", "merienda", "almuerzo", "merienda", "cena", "merienda"]
    sp = MEAL_SLOT_SPLITS[6]
    assert [round(f, 3) for _, _, f in fr] == [sp["desayuno"], sp["merienda_am"], sp["almuerzo"],
                                                sp["merienda_pm"], sp["cena"], sp["merienda_noche"]], (
        "las tres meriendas toman su cuota en orden AM → PM → noche, como el solver del camino del LLM")


def test_h1_la_forma_del_blueprint_sigue_valiendo_y_sin_nada_son_las_cuatro_canonicas(dd):
    from nutrition_calculator import meal_types_for_count
    fr = dd._franjas_del_dia({"slots": ["desayuno", "cena"]})
    assert [(e, s) for e, s, _ in fr] == [("Desayuno", "desayuno"), ("Cena", "cena")]
    fr4 = dd._franjas_del_dia({})
    assert [e for e, _, _ in fr4] == meal_types_for_count(4)


def test_h1_una_etiqueta_desconocida_cede_el_dia_al_llm(dd):
    assert dd._franjas_del_dia(_esqueleto(["Brunch", "Cena"])) is None
    assert dd.build_day_for_skeleton(_NUT, {}, {"slots": ["brunch"]}, 1) is None


def test_h1_la_familia_sale_del_pool_o_del_blueprint(dd):
    assert dd._familias_del_dia(_esqueleto(["Cena"], pool=("Pechuga de pollo", "Huevos"))) == ["Pechuga de pollo", "Huevos"]
    assert dd._familias_del_dia({"slots": ["cena"], "protein": "pollo"}) == ["pollo"]
    assert dd._familias_del_dia({}) == []


def test_h1_el_dia_real_pide_al_registry_las_franjas_y_la_familia_del_esqueleto(dd, mundo):
    dia = dd.build_day_for_skeleton(_NUT, {"user_id": "u"}, _esqueleto(["Desayuno", "Almuerzo", "Cena"]), 1)
    assert dia is not None and [m["meal"] for m in dia["meals"]] == ["Desayuno", "Almuerzo", "Cena"], (
        "un plan clínico de 3 comidas recibe 3 comidas con SUS etiquetas — no las 4 de una tabla propia")
    assert [c["slot"] for c in mundo] == ["desayuno", "almuerzo", "cena"]
    assert all(c["family"] == "Pechuga de pollo" for c in mundo), "el `protein_pool` viaja como familia"


def test_h1_cinco_comidas_producen_cinco_platos_con_merienda_am_y_pm(dd, mundo):
    from nutrition_calculator import meal_types_for_count
    cinco = meal_types_for_count(5)
    dia = dd.build_day_for_skeleton(_NUT, {"user_id": "u"}, _esqueleto(cinco), 1)
    assert dia is not None and [m["meal"] for m in dia["meals"]] == cinco
    assert sum(1 for c in mundo if c["slot"] == "merienda") == 2


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# H2 · filtros del registry, CandidateSet fijado y cocina del día
# ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_h2_los_filtros_que_horizon_ya_pasaba_viajan_tambien_desde_aqui(dd, mundo, monkeypatch):
    import horizon
    monkeypatch.setattr(horizon, "required_nutrients", lambda eff: ("phosphorus_mg", "potassium_mg"))
    monkeypatch.setattr(horizon, "_dur_kwargs", lambda eff, di: {"need_days": 25, "allow_frozen": True})
    fd = {"user_id": "u", "_plan_policy_effective": {"clinical": {"conditions": ["renal"]}},
          "health_profile": {"allergies": ["huevo"], "dietType": "omnivora"}}
    dia = dd.build_day_for_skeleton(_NUT1, fd, _esqueleto(["Almuerzo"]), 1)
    assert dia is not None
    kw = mundo[0]["kw"]
    assert kw["require_known_nutrients"] == ("phosphorus_mg", "potassium_mg"), "un renal no recibe un plato cuyo fósforo nadie midió"
    assert "market_country" in kw and kw["market_country"], "el MERCADO filtra lo que ese mercado no vende"
    assert kw["need_days"] == 25 and kw["allow_frozen"] is True, "la durabilidad de compra única viaja"
    assert kw["exclude_allergens"] == ["huevo"] and kw["diet"] == "omnivora"


def test_h2_el_candidateset_fijado_al_run_manda_sobre_la_consulta_viva(dd, mundo):
    fd = {"user_id": "u", "_blueprint_slice": {"days_offset": 7, "registry": {"candidates": {"7:desayuno": ["tpl_b"]}}}}
    dia = dd.build_day_for_skeleton(_NUT, fd, _esqueleto(["Desayuno", "Almuerzo", "Cena"]), 1)
    assert dia is not None and dia["_day_index"] == 7, "día 1 del bloque que empieza en el día 7 ⇒ índice 7"
    desayuno, almuerzo, cena = dia["meals"]
    assert desayuno["_template_id"] == "tpl_b" and desayuno["_candidate_source"] == "fijado"
    assert almuerzo["_candidate_source"] == "vivo" and cena["_candidate_source"] == "vivo"
    assert all(c["kw"]["rotate"] == 7 for c in mundo), "la rotación usa el índice ABSOLUTO del plan"


def test_h2_un_fijado_que_ya_no_pasa_los_filtros_reconsulta_y_lo_dice(dd, mundo):
    fd = {"user_id": "u", "_blueprint_slice": {"days_offset": 0, "registry": {"candidates": {"0:cena": ["tpl_retirada"]}}}}
    dia = dd.build_day_for_skeleton(_NUT1, fd, _esqueleto(["Cena"]), 1)
    assert dia is not None and dia["meals"][0]["_candidate_source"] == "vivo"


def test_h2_un_pool_sin_plantillas_pierde_la_familia_no_el_dia(dd, mundo, monkeypatch):
    import dish_registry as dr
    original = dr.template_candidates

    def _sin_familia(country, slot, family=None, **kw):
        return [] if family else original(country, slot, None, **kw)

    monkeypatch.setattr(dr, "template_candidates", _sin_familia)
    dia = dd.build_day_for_skeleton(_NUT1, {"user_id": "u"}, _esqueleto(["Cena"], pool=("Cabra",)), 1)
    assert dia is not None and dia["meals"][0]["_candidate_source"] == "vivo_sin_familia"


def test_h2_la_cocina_del_dia_recibe_el_indice_del_dia(dd, mundo, monkeypatch):
    import constants
    vistos = []
    original = constants.cultural_country_for_form_data

    def _spy(fd, day_index=None):
        vistos.append(day_index)
        return original(fd, day_index=day_index)

    monkeypatch.setattr(constants, "cultural_country_for_form_data", _spy)
    fd = {"user_id": "u", "_blueprint_slice": {"days_offset": 14}}
    assert dd.build_day_for_skeleton(_NUT1, fd, _esqueleto(["Cena"]), 3) is not None
    assert vistos == [16], "día 3 del bloque que empieza en 14 ⇒ la cocina del día 16"


def test_h2_las_docstrings_ya_no_dicen_que_el_dia_no_pasa_por_assemble(dd):
    src = Path(dd.__file__).read_text(encoding="utf-8")
    assert "se persiste sin pasar por `assemble_plan_node`" not in src
    assert "no pasa por `assemble_plan_node`, así que" not in src
    assert "aristas del grafo" in src, "la premisa corregida queda escrita donde estaba la falsa"


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# H3 · el tiempo del plato: del registry, nunca «15 min» inventado
# ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_h3_construir_comida_lleva_el_tiempo_que_declara_su_receta(dd, mundo):
    m = dd.construir_comida(_POR_ID["tpl_a"], 1.0, _CATALOGO, "almuerzo", "DO")
    assert m["prep_time"] == "45 min" and m["_prep_time_source"] == "receta"
    m2 = dd.construir_comida(_POR_ID["tpl_b"], 1.0, _CATALOGO, "almuerzo", "DO")
    assert m2["prep_time"] == "30 min" and m2["_prep_time_source"] == "tecnica"


def test_h3_el_relleno_por_defecto_no_se_convierte_en_dato(dd, mundo):
    t = _tpl("tpl_x", "Sin tiempo", ["cena"], "pollo", [("Pollo", 200), ("Arroz", 200)], prep=("defecto", 30))
    m = dd.construir_comida(t, 1.0, _CATALOGO, "cena", "DO")
    assert "prep_time" not in m, "el 30 de relleno del registry es otro número inventado: no se sirve"


def test_h3_fill_prep_time_del_registry_o_vacio_y_dicho():
    import recipe_library as rl
    meal = {"name": "Plato que no existe en ningún registro", "recipe": ["x"]}
    rl.fill_prep_time(meal, {})
    assert meal["prep_time"] == "" and meal["_prep_time_source"] == "unknown", (
        "sin dato, vacío (el frontend oculta el chip) y la fuente dicha — nunca «15 min»")
    ya = {"name": "x", "prep_time": "20 min"}
    rl.fill_prep_time(ya, {})
    assert ya["prep_time"] == "20 min" and "_prep_time_source" not in ya, "lo que el modelo declaró se respeta"


@pytest.mark.skipif(not _REG.exists(), reason="sin snapshot del registry")
def test_h3_prep_time_for_meal_lee_el_registry_real():
    import recipe_library as rl
    reg = json.loads(_REG.read_text(encoding="utf-8"))
    con_receta = next(t for t in reg["templates"]
                      if (t.get("logistics") or {}).get("prep_minutes_source") == "receta")
    esperado = f"{int(con_receta['logistics']['prep_minutes_est'])} min"
    assert rl.prep_time_for_meal({"_template_id": con_receta["template_id"]}, "DO") == esperado
    assert rl.prep_time_for_meal({"name": con_receta["name"]}, "DO") == esperado, "también por nombre exacto"
    assert rl.prep_time_for_meal({"name": "Plato inventado por el modelo"}, "DO") is None


def test_h3_assemble_ya_no_inventa_quince_minutos():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'm["prep_time"] = "15 min"' not in src
    assert "fill_prep_time(m, form_data)" in src


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# H4 · el agua medida de la receta congelada escala con la porción
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("paso,factor,esperado", [
    ("Cuécela en una olla con tres tazas de agua a fuego medio.", 0.6, "Cuécela en una olla con 1¾ tazas de agua a fuego medio."),
    ("Cuécela con tres tazas de agua.", 1.6, "Cuécela con 4¾ tazas de agua."),
    ("Hierve 3 litros de agua con sal.", 0.5, "Hierve 1,5 litros de agua con sal."),
    ("Añade media taza de agua.", 1.6, "Añade ¾ taza de agua."),
    ("Pon un litro y medio de agua.", 0.5, "Pon 750 ml de agua."),
    ("Agrega dos tazas y media de agua.", 1.2, "Agrega 3 tazas de agua."),
])
def test_h4_el_agua_medida_escala(dd, paso, factor, esperado):
    out, cambio = dd.escalar_agua_en_pasos([paso], factor)
    assert cambio and out == [esperado]


@pytest.mark.parametrize("paso,factor", [
    ("Usa dos tazas de agua por cada taza de arroz.", 0.6),       # proporción: ya escala sola
    ("Una cucharadita del agua de la cocción.", 1.6),             # no es volumen de cocción
    ("Cuécela con tres tazas de agua.", 1.0),                      # sin escala no hay nada que tocar
    ("Cuécela con tres tazas de agua.", 1.04),                     # dentro del 5 %
    ("Deja reposar hasta que el agua se absorba.", 0.6),           # sin cantidad
])
def test_h4_lo_que_no_es_agua_medida_absoluta_no_se_toca(dd, paso, factor):
    out, cambio = dd.escalar_agua_en_pasos([paso], factor)
    assert not cambio and out == [paso]


def test_h4_la_comida_armada_lleva_el_agua_escalada_y_lo_dice(dd, mundo):
    m = dd.construir_comida(_POR_ID["tpl_a"], 0.6, _CATALOGO, "almuerzo", "DO")
    assert m["_recipe_water_scaled"] is True and "1¾ tazas de agua" in m["recipe"][0]
    m1 = dd.construir_comida(_POR_ID["tpl_a"], 1.0, _CATALOGO, "almuerzo", "DO")
    assert "_recipe_water_scaled" not in m1 and "tres tazas de agua" in m1["recipe"][0]


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# H5 · «sal» por palabra, no por subcadena
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("nombre", ["Sal", "Sal marina", "Ajo", "Ajo en polvo", "Clavos de olor", "Orégano dominicano", "Sazón completa"])
def test_h5_los_condimentos_de_verdad_no_escalan(dd, nombre):
    assert dd._no_escala(nombre), nombre


@pytest.mark.parametrize("nombre", ["Salami", "Salsa de tomate", "Ensalada verde", "Salmón", "Salchicha",
                                    "Bacalao salado", "Ajonjolí", "Salvado de avena", "Pechuga de pollo"])
def test_h5_los_alimentos_que_contienen_sal_como_letras_si_escalan(dd, nombre):
    assert not dd._no_escala(nombre), f"{nombre!r} no es un condimento: su porción crece con la ración"


def test_h5_el_salami_del_guiso_se_escala_con_el_plato(dd, mundo):
    cat = dict(_CATALOGO, **{"Salami": {"name": "Salami", "kcal_per_100g": 336, "protein_g_per_100g": 22,
                                        "carbs_g_per_100g": 2, "fats_g_per_100g": 26, "sodium_mg_per_100g": 1740}})
    t = _tpl("tpl_s", "Salami guisado con yuca", ["almuerzo"], "cerdo", [("Salami", 90), ("Yuca", 250), ("Sal", 1)])
    m = dd.construir_comida(t, 0.6, cat, "almuerzo", "DO")
    assert "54 g de Salami" in m["ingredients"] and "1 g de Sal" in m["ingredients"]


def test_h5_el_retinol_tambien_casa_por_palabra(dd):
    cat = {"Hígado de res": {"vitamin_a_mcg_rae_per_100g": 4970}, "Empate de res": {"vitamin_a_mcg_rae_per_100g": 4970}}
    assert dd._retinol_preformado_mcg({"ingredients": ["120 g de Hígado de res"]}, cat) > 3000
    assert dd._retinol_preformado_mcg({"ingredients": ["120 g de Empate de res"]}, cat) == 0.0


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# H6 · H14 · el sodio es el del plato servido, y lo desconocido se dice
# ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_h6_el_sodio_se_mide_sobre_los_gramos_servidos(dd):
    comida = {"ingredients": ["200 g de Arenque", "100 g de Arroz"]}
    mg, sin = dd._sodio_de_comida(comida, _CATALOGO, _POR_ID["tpl_c"], 1.6)
    assert round(mg) == 1801 and sin == 0, "900×2 + 1×1 — no la ración base del registry"


def test_h6_una_linea_sin_dato_no_es_cero_a_secas(dd):
    comida = {"ingredients": ["240 g de Arenque", "400 g de Yuca"]}   # la yuca no publica sodio
    mg, sin = dd._sodio_de_comida(comida, _CATALOGO, _POR_ID["tpl_c"], 1.6)
    assert round(mg) == 2160 and sin == 1


def test_h6_sin_ningun_dato_cae_a_la_racion_del_registry_escalada(dd):
    comida = {"ingredients": ["400 g de Yuca"]}
    mg, sin = dd._sodio_de_comida(comida, _CATALOGO, _POR_ID["tpl_c"], 1.6)
    assert round(mg) == round(1350 * 1.6) and sin == 1


def test_h6_el_dia_carga_el_sodio_del_plato_escalado_y_anota_lo_desconocido(dd, mundo, monkeypatch):
    monkeypatch.setattr(dd, "_techo_sodio", lambda: 100000.0)
    dia = dd.build_day_for_skeleton(_NUT1, {"user_id": "u"}, _esqueleto(["Almuerzo"], pool=("Arenque",)), 1)
    assert dia is not None
    m = dia["meals"][0]
    assert m["_sodium_mg_est"] > 0 and dia["_sodium_mg_est"] == m["_sodium_mg_est"]
    if m["_template_id"] == "tpl_c":
        assert m["_sodium_unknown_lines"] == 1 and dia["_sodium_unknown_lines"] == 1, "la yuca no publica sodio: se dice"


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# H7 · el backstop clínico es fail-secure también aquí; el escáner corre completo
# ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_h7_un_backstop_que_no_se_puede_evaluar_rechaza_el_plato(dd, monkeypatch):
    import graph_orchestrator as go

    def _revienta(*a, **k):
        raise RuntimeError("catálogo caído")

    monkeypatch.setattr(go, "clinical_backstop_for_meal", _revienta)
    fuera = dd.verifica_comida({"meal": "Cena", "name": "x", "ingredients": ["100 g de Pollo"], "recipe": ["Cocina."]},
                               {"allergies": ["huevo"]}, _CATALOGO)
    assert any("no evaluable" in str(v) for v in fuera), "«no pude evaluar» no es «sin violaciones»"


def test_h7_el_escaner_culinario_corre_completo_no_una_lista_a_mano(dd):
    src = inspect.getsource(dd.verifica_comida)
    assert "culinary_contract_scan(" in src
    assert '"_v1_verbo_alimento", "_v2_estado_imposible"' not in src


def test_h7_un_techo_de_retinol_no_evaluable_rechaza(dd, monkeypatch):
    monkeypatch.setattr(dd, "_retinol_preformado_mcg", lambda meal, cat: (_ for _ in ()).throw(ValueError("x")))
    fuera = dd.verifica_comida({"meal": "Cena", "name": "x", "ingredients": [], "recipe": []}, {}, {})
    assert any("retinol no evaluable" in str(v) for v in fuera)


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# H9 · «no pude evaluar» deja marca en vez de aprobar; el gate culinario no muere por un formato
# ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_h9_el_log_del_gate_culinario_ya_no_formatea_none_con_porcentaje():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'cobertura={_cul_cov:.0%}' not in src
    assert "'no medible'" in src


def test_h9_el_guard_de_coherencia_marca_el_plan_cuando_no_puede_evaluar(monkeypatch):
    import shopping_calculator as sc

    def _revienta(*a, **k):
        raise KeyError("hidratación")

    monkeypatch.setattr(sc, "run_shopping_coherence_guard", _revienta)
    plan = {"days": []}
    div, block = sc.run_shopping_coherence_guard_and_append_history(plan)
    assert div == [] and block is False
    marca = plan["_shopping_coherence_unevaluable"]
    assert marca["stage"] == "run_shopping_coherence_guard" and marca["error"] == "KeyError" and marca["at"]


def test_h9_la_marca_no_lanza_ni_con_basura():
    import shopping_calculator as sc
    sc._mark_guard_unevaluable(None, "x", ValueError("y"))
    sc._mark_guard_unevaluable("no soy un dict", "x", ValueError("y"))


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# H12 · las mutaciones que cambian lo que la proyección lee la re-encolan; la Nevera entra en la huella
# ─────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("reason", ["shift_plan", "grocery_start_date", "restore_local", "restore", "restock", "inventory_consume"])
def test_h12_cada_mutacion_reencola_la_proyeccion(reason):
    src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    assert f'reason="{reason}"' in src, f"/{reason} cambia lo que la proyección lee y no la re-encolaba"


def test_h12_la_huella_de_la_nevera_cambia_con_la_nevera(monkeypatch):
    import shopping_calculator as sc
    from shopping.projection import reprojection as rp
    monkeypatch.setattr(sc, "fetch_inventory_and_consumed_for_plan",
                        lambda uid, pd, is_new_plan=False: ([{"name": "Pollo", "quantity": 2, "unit": "lb"}], []))
    a = rp.inventory_fingerprint("u", {})
    monkeypatch.setattr(sc, "fetch_inventory_and_consumed_for_plan",
                        lambda uid, pd, is_new_plan=False: ([{"name": "Pollo", "quantity": 1, "unit": "lb"}], []))
    b = rp.inventory_fingerprint("u", {})
    assert a and b and a != b

    def _revienta(*a, **k):
        raise RuntimeError("sin pool")

    monkeypatch.setattr(sc, "fetch_inventory_and_consumed_for_plan", _revienta)
    assert rp.inventory_fingerprint("u", {}) == "", "fail-open: sin Nevera medible, manda la huella de la lista"


def test_h12_misma_lista_pero_otra_nevera_encola(monkeypatch):
    import plan_jobs as pj
    import horizon
    import db
    from shopping.projection import reprojection as rp
    plan, user = "e45e649c-231d-493a-adbf-af8aa8b73ce8", "f47126cb-e137-4003-9db3-cbec22b02d59"
    pd = {"aggregated_shopping_list_weekly": [{"name": "Pollo", "base_qty": 1, "base_unit": "lb", "market_qty": 1, "market_unit": "lb"}],
          "days": [{}], "total_days_requested": 7}
    fp = rp.shopping_list_fingerprint(pd)
    monkeypatch.setattr(pj, "plan_jobs_enabled", lambda: True)
    monkeypatch.setattr(pj, "consumer_enabled", lambda jt: True)
    monkeypatch.setattr(horizon, "shopping_projection_jobs_enabled", lambda: True)
    monkeypatch.setattr(horizon, "effective_policy_for_plan", lambda p, form_data=None: {"policy_hash": "h", "shopping": {"main_cycle_days": 7, "fresh_topup_days": 7}})
    monkeypatch.setattr(pj, "current_plan_revision", lambda plan_id: 3)
    monkeypatch.setattr(rp, "inventory_fingerprint", lambda uid, p: "nevera-nueva")
    monkeypatch.setattr(db, "execute_sql_query", lambda *a, **k: {"status": "done", "fp": fp, "inv": "nevera-vieja"})
    visto = {}
    monkeypatch.setattr(pj, "enqueue_plan_job", lambda jt, pid, uid, **kw: visto.update(kw) or "job-1")
    assert rp.enqueue_shopping_reprojection(plan, user, reason="restock", plan_data=pd) == "job-1"
    assert visto["payload"]["inventory_fingerprint"] == "nevera-nueva"
    assert visto["dedup_key"].endswith(":nevera-n"), "la clave de dedupe distingue la Nevera"
    monkeypatch.setattr(db, "execute_sql_query", lambda *a, **k: {"status": "done", "fp": fp, "inv": "nevera-nueva"})
    assert rp.enqueue_shopping_reprojection(plan, user, reason="restock", plan_data=pd) is None, "misma lista y misma Nevera: idéntica"


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# H14 · una receta congelada no se reescribe en «3 pilares»
# ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_h14_expand_devuelve_la_receta_de_la_biblioteca_sin_llm():
    src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    i = src.index("expanded_steps = expand_recipe_agent(data)")
    bloque = src[i - 2500:i]
    assert "recipe_for_dish_name as _rl_recipe" in bloque and '"recipe_source": "library"' in bloque, (
        "la receta firmada se devuelve tal cual ANTES de pedirle al LLM que la reescriba")


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# H16 · el selector: rotación antes del corte, y la reserva menos salada
# ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_h16_la_rotacion_alcanza_a_los_candidatos_que_el_corte_escondia(dd, monkeypatch):
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY_TIE_MAX", "10")
    por_id = {f"tpl_{i:02d}": _tpl(f"tpl_{i:02d}", f"Plato {i:02d}", ["almuerzo"], "pollo",
                                   [("Pollo", 200), ("Arroz", 200)]) for i in range(12)}
    obj = {"kcal": 590, "protein_g": 67, "carbs_g": 56, "fats_g": 8}
    tids = sorted(por_id)
    sin_rotar = [t["template_id"] for t, _ in dd.elegir_plantillas(tids, obj, _CATALOGO, por_id, "almuerzo")]
    assert len(sin_rotar) == 10
    cabezas = {dd.elegir_plantillas(tids, obj, _CATALOGO, por_id, "almuerzo", rotacion=r)[0][0]["template_id"]
               for r in range(12)}
    assert cabezas == set(tids), "los 12 elegibles llegan a ser cabeza alguna vez: ninguno queda inalcanzable"


def test_h16_la_reserva_de_sodio_es_la_menos_salada(dd):
    src = inspect.getsource(dd.build_day_for_skeleton)
    bloque = src.split("if _sodio_dia + _na > _techo_sodio():")[1].split("continue")[0]
    assert "if _reserva is None:" in bloque and "elif _na < _reserva[2]:" in bloque
