# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-24 · 2026-09-12] C3 del plan de pendientes = CUL-P0-04: huevo entero, clara y yema no son intercambiables.

Aceptación del backlog, una a una:
  · «12 huevos sin yema» conserva 12 claras y NO hereda los macros de 12 enteros (ficha fijada en la prueba);
  · gramos/unidades concuerdan con esa ficha;
  · nada convierte una petición de claras en yemas (ni en la lista, ni en los pasos, ni en el prompt, ni en los pools);
  · la petición sobrevive a «enteros primero» y a los fallbacks;
  · las recetas crudas / incompatibles con restricciones fallan sus controles en las tres formas.
Más el hallazgo de orden: el contrato de C2 no era el último del persist boundary — ahora corre en la cola real.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_ROOT = _BACKEND.parent

import recipe_contract as rc  # noqa: E402
from culinary_coherence import build_culinary_index  # noqa: E402

# Ficha FIJADA en la prueba (mismos números que el catálogo vivo del 2026-09-12; la prueba no depende de la base).
_CAT = [
    {"name": "Huevo", "aliases": ["huevos", "huevos enteros"], "category": "Proteínas", "prep_methods": ["cocido"],
     "density_g_per_unit": 50, "kcal_per_100g": 138.9, "protein_g_per_100g": 12.6, "carbs_g_per_100g": 0.72, "fats_g_per_100g": 9.51},
    {"name": "Clara de huevo", "aliases": ["claras de huevo", "clara de huevo", "claras", "clara de huevos"], "category": "Proteínas",
     "prep_methods": ["cocido"], "density_g_per_unit": 33, "kcal_per_100g": 48.1, "protein_g_per_100g": 10.9, "carbs_g_per_100g": 0.73,
     "fats_g_per_100g": 0.17},
    {"name": "Yema de huevo", "aliases": ["yema de huevo", "yemas de huevo", "yemas", "yema de huevos"], "category": "Proteínas",
     "prep_methods": ["cocido"], "density_g_per_unit": 17, "kcal_per_100g": 316.5, "protein_g_per_100g": 15.9, "carbs_g_per_100g": 3.59,
     "fats_g_per_100g": 26.5},
    {"name": "Cebolla", "aliases": ["cebollas"], "category": "Vegetales", "prep_methods": ["crudo", "sofrito"]},
    {"name": "Pan integral", "aliases": ["pan integral familiar"], "category": "Granos", "prep_methods": ["tostado"]},
]
_INDEX = build_culinary_index(_CAT)


def _meal(ings, rec, **extra):
    m = {"name": "Plato", "meal": "Desayuno", "ingredients": list(ings), "ingredients_raw": list(ings), "recipe": list(rec)}
    m.update(extra)
    return m


class _FichaDB:
    """`macros_from_ingredient_string` con la ficha fijada: gramos = piezas × densidad; macros por 100 g."""

    _ROWS = {r["name"]: r for r in _CAT if "kcal_per_100g" in r}
    _RX = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s+(huevos?(?:\s+enteros?)?|claras?(?:\s+de\s+huevo)?|yemas?(?:\s+de\s+huevo)?)\s*$", re.I)

    def macros_from_ingredient_string(self, s):
        m = self._RX.match(str(s))
        if not m:
            return None
        n = float(m.group(1).replace(",", "."))
        noun = m.group(2).lower()
        row = self._ROWS["Clara de huevo" if noun.startswith("clara") else "Yema de huevo" if noun.startswith("yema") else "Huevo"]
        g = n * row["density_g_per_unit"]
        return {"name": row["name"], "grams": g, "kcal": g * row["kcal_per_100g"] / 100, "protein": g * row["protein_g_per_100g"] / 100,
                "carbs": g * row["carbs_g_per_100g"] / 100, "fats": g * row["fats_g_per_100g"] / 100}


# ─────────────── regla 1: la lista nombra la forma ───────────────

def test_doce_huevos_sin_yema_son_doce_claras_y_no_heredan_los_macros_de_doce_enteros():
    m = _meal(["12 huevos sin yema"], ["bate 12 huevos"])
    assert rc.canonicalize_egg_form_lines(m) == 1
    assert m["ingredients"] == ["12 claras de huevo"] and m["ingredients_raw"] == ["12 claras de huevo"]
    db = _FichaDB()
    claras = db.macros_from_ingredient_string(m["ingredients"][0])
    enteros = db.macros_from_ingredient_string("12 huevos")
    assert claras["grams"] == pytest.approx(396.0) and enteros["grams"] == pytest.approx(600.0)
    assert claras["fats"] < 1.0 < 50.0 < enteros["fats"], "12 claras no son 12 enteros: 0,67 g de grasa contra 57"
    assert claras["protein"] == pytest.approx(43.16, abs=0.1)


def test_las_variantes_de_la_forma_y_el_sentido_prohibido():
    m = _meal(["2 huevos (solo claras)", "1 huevo sin clara", "3 huevos", "4 claras de huevo", "1 huevo sin yemas (50 g)"], [])
    assert rc.canonicalize_egg_form_lines(m) == 3
    assert m["ingredients"] == ["2 claras de huevo", "1 yema de huevo", "3 huevos", "4 claras de huevo", "1 clara de huevo"]
    # nunca al revés: una línea de claras jamás se convierte en enteros ni en yemas
    m2 = _meal(["4 claras de huevo", "2 yemas de huevo"], [])
    assert rc.canonicalize_egg_form_lines(m2) == 0 and m2["ingredients"] == ["4 claras de huevo", "2 yemas de huevo"]


def test_raw_se_localiza_por_texto_nunca_por_indice():
    m = _meal(["12 huevos sin yema"], [])
    m["ingredients_raw"] = ["Sal al gusto", "12 huevos sin yema"]         # orden distinto: el índice mentiría
    rc.canonicalize_egg_form_lines(m)
    assert m["ingredients_raw"] == ["Sal al gusto", "12 claras de huevo"]
    m3 = _meal(["12 huevos sin yema"], [])
    m3["ingredients_raw"] = ["12 huevos sin yema", "12 huevos sin yema"]  # ambigua ⇒ raw no se toca
    rc.canonicalize_egg_form_lines(m3)
    assert m3["ingredients"] == ["12 claras de huevo"] and m3["ingredients_raw"] == ["12 huevos sin yema", "12 huevos sin yema"]


# ─────────────── regla 2: los pasos siguen a la forma comprada ───────────────

def test_el_tope_diario_parte_la_lista_y_los_pasos_reciben_el_reparto():
    # el caso del corpus: `_cap_daily_whole_eggs` dejó «3 huevos» + «2 claras de huevo» y el paso decía «casca 6 huevos»
    m = _meal(["3 huevos", "2 claras de huevo"], ["Mise en place: casca 6 huevos; pica ¼ de cebolla.",
                                                  "El Toque de Fuego: añade 6 huevos con la sal."], _egg_day_capped=True)
    r = rc.egg_forms_step_sync(m, _INDEX)
    assert r["reescritas"] == 2
    assert m["recipe"][0].startswith("Mise en place: casca 3 huevos y 2 claras de huevo;")
    assert "añade 3 huevos y 2 claras de huevo con la sal" in m["recipe"][1]
    # y el contrato completo, después, no vuelve a «casca 3 huevos» a secas (C2 ya ve 3 = 3 y 2 = 2)
    r2 = rc.reconcile_meal(m, _INDEX)
    assert "casca 3 huevos y 2 claras de huevo" in m["recipe"][0]
    assert r2["reescritas"] == 0 or all(c["familia"] != "huevo_forma" for c in r2["cambios"])


def test_c2_a_solas_borraba_las_claras_y_el_contrato_completo_ya_no():
    lista = ["3 huevos", "2 claras de huevo"]
    solo_c2 = _meal(lista, ["casca 6 huevos."])
    rc.reconcile_step_quantities(solo_c2, _INDEX)
    assert solo_c2["recipe"] == ["casca 3 huevos."], "C2 ciego a la forma: repara el número y borra las claras"
    completo = _meal(lista, ["casca 6 huevos."])
    rc.reconcile_meal(completo, _INDEX)
    assert completo["recipe"] == ["casca 3 huevos y 2 claras de huevo."]


def test_las_claras_compradas_que_ningun_paso_usa_entran_en_la_primera_mencion():
    m = _meal(["1 huevo", "3 claras de huevo"], ["sofríe la cebolla, incorpora 1 huevo y remueve."])
    rc.egg_forms_step_sync(m, _INDEX)
    assert m["recipe"] == ["sofríe la cebolla, incorpora 1 huevo y 3 claras de huevo y remueve."]
    desnudo = _meal(["3 huevos", "1 clara de huevo"], ["Cocina huevo a la plancha o hervido.", "Acompaña con huevo."])
    rc.egg_forms_step_sync(desnudo, _INDEX)
    assert desnudo["recipe"][0] == "Cocina 3 huevos y 1 clara de huevo a la plancha o hervido."
    assert desnudo["recipe"][1] == "Acompaña con huevo.", "sólo la primera mención recibe el reparto"


def test_lista_solo_de_claras_los_pasos_dejan_de_cascar_huevos():
    m = _meal(["4 claras de huevo"], ["Mise en place: bate 4 huevos.", "añade el huevo y remueve hasta que cuaje.",
                                       "⚠️ Seguridad alimentaria: cocina el huevo por completo (≥71°C, yema y clara firmes, sin partes líquidas) antes de servir."])
    r = rc.egg_forms_step_sync(m, _INDEX)
    assert r["reescritas"] == 3
    assert m["recipe"][0] == "Mise en place: bate 4 claras de huevo."
    assert m["recipe"][1] == "añade las claras y remueve hasta que cuaje."
    assert "la clara firme, sin partes líquidas" in m["recipe"][2] and "yema y clara firmes" not in m["recipe"][2]
    uno = _meal(["1 clara de huevo"], ["bate 1 huevo;", "añade el huevo."])
    rc.egg_forms_step_sync(uno, _INDEX)
    assert uno["recipe"] == ["bate 1 clara de huevo;", "añade la clara."]


def test_una_lista_de_enteros_con_pasos_que_hablan_de_la_clara_es_tecnica_no_contradiccion():
    m = _meal(["2 huevos"], ["cocina hasta que la clara esté cuajada y la yema tierna.", "separa las claras y bate."])
    antes = copy.deepcopy(m)
    assert rc.egg_forms_step_sync(m, _INDEX)["reescritas"] == 0 and m == antes
    assert rc.reconcile_meal(m, _INDEX)["reescritas"] == 0 and m == antes


def test_nada_convierte_claras_en_yemas_ni_toca_las_notas_de_procedencia():
    m = _meal(["4 claras de huevo"], ["bate 4 huevos.", "🌱 Nota del Nutricionista AI: esta receta usa solo la clara — NO botes las yemas.",
                                       "💡 se reemplazó huevo por yogur en la versión anterior."])
    rc.reconcile_meal(m, _INDEX)
    assert "yema" not in m["recipe"][0] and "clara" in m["recipe"][0]
    assert m["recipe"][1].startswith("🌱 Nota del Nutricionista AI: esta receta usa solo la clara") and "NO botes las yemas" in m["recipe"][1]
    assert m["recipe"][2] == "💡 se reemplazó huevo por yogur en la versión anterior."
    assert rc._es_nota(m["recipe"][1]), "la nota 🌱 no es un paso"


def test_idempotente_y_fail_open():
    m = _meal(["3 huevos", "2 claras de huevo"], ["casca 6 huevos.", "añade el huevo."])
    r1 = rc.reconcile_meal(m, _INDEX)
    r2 = rc.reconcile_meal(m, _INDEX)
    assert r1["reescritas"] >= 1 and r2["reescritas"] == 0 and r2.get("lista_reescrita", 0) == 0
    assert rc.egg_forms_step_sync({"ingredients": None, "recipe": "no es lista"}, _INDEX) == {"reescritas": 0, "cambios": []}
    assert rc.canonicalize_egg_form_lines({"ingredients": "x"}) == 0


def test_la_telemetria_conserva_la_forma_de_c2_y_solo_anade_lista_reescrita_si_cambio(monkeypatch):
    monkeypatch.setattr(rc, "_index_default", lambda db=None: _INDEX)
    monkeypatch.setenv("MEALFIT_RECIPE_FINAL_CONTRACT", "repair")
    m = _meal(["3 huevos"], ["casca 6 huevos."])
    rc.apply_final_contract_meal(m, None)
    assert set(m[rc.TELEMETRIA_KEY]) == {"modo", "reescritas", "familias", "sin_reparar"}
    m2 = _meal(["12 huevos sin yema"], ["bate 12 huevos."])
    rc.apply_final_contract_meal(m2, _FichaDB())
    assert m2[rc.TELEMETRIA_KEY]["lista_reescrita"] == 1 and m2["ingredients"] == ["12 claras de huevo"]
    assert m2["recipe"] == ["bate 12 claras de huevo."]


def test_al_canonizar_la_lista_se_re_miden_los_macros_con_el_truth_up_del_repo(monkeypatch):
    monkeypatch.setattr(rc, "_index_default", lambda db=None: _INDEX)
    monkeypatch.setenv("MEALFIT_RECIPE_FINAL_CONTRACT", "repair")
    llamadas = []
    import graph_orchestrator as go
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: llamadas.append(meal["ingredients"][0]) or True)
    m = _meal(["12 huevos sin yema"], ["bate 12 huevos."])
    rc.apply_final_contract([{"day": 1, "meals": [m]}], _FichaDB())
    assert llamadas == ["12 claras de huevo"], "la lista cambió ⇒ los macros se re-miden sobre la línea nueva"
    sin_cambio = _meal(["3 huevos"], ["casca 6 huevos."])
    rc.apply_final_contract([{"day": 1, "meals": [sin_cambio]}], _FichaDB())
    assert llamadas == ["12 claras de huevo"], "si la lista no cambió, no se re-mide"


# ─────────────── el orden: la cola REAL del persist boundary ───────────────

def test_el_contrato_corre_en_la_cola_del_shield_pre_insert_antes_de_restaurar_los_dias_congelados():
    src = (_BACKEND / "db_plans.py").read_text(encoding="utf-8", errors="ignore")
    i = src.index("def _finalize_plan_data_for_insert(")
    cuerpo = src[i:src.index("\ndef ", i + 10)]
    fpc = cuerpo.index("from graph_orchestrator import finalize_plan_data_coherence as _fpc")
    cdwe = cuerpo.index("_pass_n += _cdwe(_pd.get(\"days\") or [], db=_db_ins)")
    fica = cuerpo.index("_fica(_pd)")
    tail = cuerpo.index("_rfc_tail_out = _rfc_tail(_pd.get(\"days\") or [], _db_ins)")
    frz = cuerpo.index("_rpd_frz(_pd, _frozen_token)")
    dppv = cuerpo.index("_dppv(_pd)")
    assert fpc < cdwe < fica < tail < frz < dppv, (
        "el contrato va DESPUÉS del tope de huevos y del re-cuadre de conteos, ANTES de restaurar los pasados y de los detectores")
    assert "P1-PLAN-LOTE-24-FINAL-CONTRACT-TAIL" in cuerpo


def test_el_contrato_corre_al_final_del_mutator_de_swap_y_de_chat_modify():
    sw = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8", errors="ignore")
    a = sw.index("_ubp_sw(plan_data, surface=\"swap_persist\", pantry_limited=_pl_sw)")
    b = sw.index("_rfc_sw(plan_data.get(\"days\") or [], _NDB_rfc_sw())")
    c = sw.index("_rebuild_plan_shopping_lists_inline(\n                plan_data, verified_user_id, surface=\"swap_persist\"")
    assert a < b < c, "swap: tras la paridad de banda y ANTES del rebuild de las listas"
    tl = (_BACKEND / "tools.py").read_text(encoding="utf-8", errors="ignore")
    a2 = tl.index("_ubp_cm(plan_data_fresh, surface=\"chat_modify\", pantry_limited=_pl_cm)")
    b2 = tl.index("_rfc_cm(plan_data_fresh.get(\"days\") or [], _NDB_rfc_cm())")
    c2 = tl.index("plan_data_fresh[\"aggregated_shopping_list\"] = aggr_list")
    assert a2 < b2 < c2, "chat-modify: tras la paridad de banda y ANTES de escribir las listas"
    for s in (sw, tl):
        assert "P1-PLAN-LOTE-24-FINAL-CONTRACT-TAIL" in s


def test_los_ganchos_de_c2_siguen_en_los_finalizadores():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8", errors="ignore")
    assert "_rfc = _recipe_final_contract(days, db)" in src
    assert "total += _recipe_final_contract_meal(meal, db)" in src


# ─────────────── la petición de claras sobrevive a «enteros primero» y a los fallbacks ───────────────

def test_egg_staple_forms_lee_la_declaracion_por_sustantivo_cabecera():
    from plan_policy import egg_staple_forms, egg_white_staple_declared
    assert egg_staple_forms({}) == set() and egg_staple_forms(None) == set()
    assert egg_staple_forms({"stapleFoods": ["Clara de huevo", "Avena"]}) == {"clara"}
    assert egg_staple_forms({"stapleAnchors": [{"name": "Huevo", "slots": ["desayuno"]}]}) == {"entero"}
    assert egg_staple_forms({"staple_foods": ["Yema de huevo", "Clara de huevo"]}) == {"yema", "clara"}
    assert egg_white_staple_declared({"stapleFoods": ["Clara de huevo"]})
    assert not egg_white_staple_declared({"stapleFoods": ["Clara de huevo", "Huevo"]}), "pidió las dos formas: no se le impone la clara"
    assert not egg_white_staple_declared({"stapleFoods": ["Huevos"]})


def test_el_prompt_sustituye_enteros_primero_por_claras_primero_solo_para_quien_declaro_la_clara():
    from prompts.day_generator import build_day_generator_system_prompt, override_egg_form_preference
    base = build_day_generator_system_prompt()
    assert "HUEVOS: ENTEROS PRIMERO" in base
    assert override_egg_form_preference(base, False) is base, "sin declaración, el MISMO objeto (prompt-cache)"
    out = override_egg_form_preference(base, True)
    assert "HUEVOS: CLARAS PRIMERO" in out and "ENTEROS PRIMERO" not in out
    assert "2 huevos + 2 claras" not in out, "la regla espejo no conserva la preferencia por la mezcla con yema"
    assert "nunca «huevos sin yema»" in out
    assert out.count("HUEVOS:") == base.count("HUEVOS:"), "se SUSTITUYE la regla, no se añade otra (dos reglas contradictorias es el modo de fallo conocido)"
    assert override_egg_form_preference(out, True) == out, "idempotente"
    assert override_egg_form_preference("sin la regla", True) == "sin la regla"


def test_el_helper_del_orquestador_aplica_el_espejo_y_conserva_la_identidad_del_prompt_cacheado(monkeypatch):
    import graph_orchestrator as go
    assert go._day_system_instruction_for_diet({}) is go._DAY_SYSTEM_INSTRUCTION_CACHED
    con = go._day_system_instruction_for_diet({"stapleFoods": ["Clara de huevo"]})
    assert "HUEVOS: CLARAS PRIMERO" in con and con is not go._DAY_SYSTEM_INSTRUCTION_CACHED
    ambos = go._day_system_instruction_for_diet({"stapleFoods": ["Clara de huevo", "Huevo"]})
    assert "HUEVOS: ENTEROS PRIMERO" in ambos
    monkeypatch.setattr(go, "EGG_STAPLE_HONORED", False)
    assert go._day_system_instruction_for_diet({"stapleFoods": ["Clara de huevo"]}) is go._DAY_SYSTEM_INSTRUCTION_CACHED, "knob off ⇒ conducta anterior"


def test_el_diversificador_de_pools_no_toca_el_huevo_declarado_basico(monkeypatch):
    import graph_orchestrator as go
    skel = [{"day": i + 1, "protein_pool": ["Huevos", "Pollo"]} for i in range(5)]
    sin = copy.deepcopy(skel)
    assert go._diversify_egg_pools(sin, {}) >= 1, "sin declaración, diversifica como siempre"
    con = copy.deepcopy(skel)
    assert go._diversify_egg_pools(con, {"stapleFoods": ["Clara de huevo"]}) == 0 and con == skel
    con2 = copy.deepcopy(skel)
    assert go._diversify_egg_pools(con2, {"stapleAnchors": [{"name": "Huevo"}]}) == 0 and con2 == skel
    monkeypatch.setattr(go, "EGG_STAPLE_HONORED", False)
    off = copy.deepcopy(skel)
    assert go._diversify_egg_pools(off, {"stapleFoods": ["Clara de huevo"]}) >= 1, "knob off ⇒ conducta anterior"


def test_la_racion_pedida_llega_al_bloque_de_politica_con_su_forma():
    from horizon import policy_prompt_block, _portion_txt
    eff = {"recurrence": {"global_mode": "balanced"},
           "food_anchors": [{"ingredient_id": "clara_de_huevo", "name": "Clara de huevo", "slots": ["desayuno"], "min_per_7d": 5,
                             "max_per_7d": 7, "portion": {"qty": 10, "unit": "unidad"}},
                            {"ingredient_id": "avena", "name": "Avena", "slots": [], "min_per_7d": 2, "max_per_7d": 7, "portion": None}]}
    out = policy_prompt_block(eff, None, surface="test", enforced=True)
    assert "Clara de huevo (5–7 de cada 7 días, desayuno, 10 unidad por comida)" in out
    assert "Avena (2–7 de cada 7 días)" in out
    assert _portion_txt({"qty": 1.5, "unit": "g"}) == "1.5 g" and _portion_txt(None) == "" and _portion_txt({"qty": 0, "unit": "g"}) == ""


def test_el_ancla_clara_de_huevo_no_se_da_por_cumplida_con_huevos_enteros():
    from plan_policy import _matches
    from horizon import anchor_in_text
    for txt in ("2 huevos", "3 huevos enteros", "1 yema de huevo"):
        assert not _matches("Clara de huevo", txt) and not anchor_in_text("Clara de huevo", txt), txt
    assert _matches("Clara de huevo", "4 claras de huevo") and anchor_in_text("Clara de huevo", "4 claras de huevo")
    # Limitación conocida y fuera de este lote: «4 claras» a secas (sin «de huevo») NO casa con el ancla — la fidelidad
    # exige todos los tokens del nombre. El parser de nutrición y el del contrato sí la resuelven a `Clara de huevo`.
    assert not anchor_in_text("Clara de huevo", "4 claras")
    assert _matches("Huevo", "4 claras de huevo"), "quien pidió «Huevo» a secas acepta cualquier forma"


# ─────────────── las restricciones cubren las tres formas (ya cierto; ahora anclado) ───────────────

def test_alergia_vegano_y_huevo_crudo_ven_las_tres_formas():
    import graph_orchestrator as go
    from constants import strip_accents
    for t in ("huevo", "huevos", "clara", "claras", "yema", "yemas"):
        assert t in go._DIET_EGG_TERMS and t in go._RAW_EGG_TERMS
    src = (_BACKEND / "constants.py").read_text(encoding="utf-8", errors="ignore")
    i = src.index('if any(r in ["huevo", "huevos", "egg", "eggs"] for r in normalized_restrictions):')
    seg = src[i:i + 400]
    for t in ('"clara"', '"claras"', '"yema"', '"yemas"'):
        assert t in seg, "la alergia a huevo se expande a claras y yemas"
    for ing in ("4 claras de huevo", "1 yema de huevo", "2 huevos"):
        assert go._meal_has_egg({"ingredients": [ing]}, strip_accents)
    viol = go._scan_raw_egg_violations({"days": [{"meals": [{"name": "Batido de claras", "ingredients": ["4 claras de huevo"], "recipe": ["licúa todo"]}]}]})
    assert viol and viol[0][3] in ("blended", "no_cook"), "claras crudas en un batido: el escáner las ve"


# ─────────────── corpus fijo, medidor, docs, marker ───────────────

def test_sobre_el_corpus_fijo_las_contradicciones_de_forma_bajan_a_cero_y_es_idempotente():
    import importlib.util
    # por ruta, no por sys.path: `scripts/` en cabeza sombrea `plan_gym` (ratchet de P1-PLAN-LOTE-13)
    _spec = importlib.util.spec_from_file_location("medir_formas_huevo", _BACKEND / "scripts" / "medir_formas_huevo.py")
    mfh = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(mfh)
    r = mfh.medir_corpus(mfh.CORPUS)
    assert r["comidas"] == 64 and r["con_huevo"] == 14 and r["egg_day_capped"] == 6
    assert r["comidas_contradictorias_antes"] >= 5
    assert r["comidas_contradictorias_despues"] == 0
    assert r["idempotente"] and r["reescritas_forma"] >= 6


def test_el_medidor_existe_y_solo_lee():
    src = (_BACKEND / "scripts" / "medir_formas_huevo.py").read_text(encoding="utf-8")
    assert "conn.read_only = True" in src and "copy.deepcopy" in src
    assert not re.search(r"\b(INSERT|UPDATE|DELETE)\b", src), "el medidor no escribe"


def test_docs_plan_knobs_y_marker():
    doc = (_BACKEND / "docs" / "culinary_coherence.md").read_text(encoding="utf-8")
    assert "## Las tres formas del huevo (C3 · `P1-PLAN-LOTE-24` · 2026-09-12)" in doc
    assert "5 comidas contradictorias → 0" in doc and "P1-PLAN-LOTE-24-FINAL-CONTRACT-TAIL" in doc
    plan = (_BACKEND / "docs" / "plan_pendientes_2026_09_11.md").read_text(encoding="utf-8")
    assert "| C3 | ✅ 2026-09-12 · las tres formas del huevo · 👤 compra de claras en envase |" in plan
    knobs = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    assert "| `MEALFIT_EGG_STAPLE_HONORED` | `True` |" in knobs
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 24 and m.group(2) >= "2026-09-12"
    for rel in ("BACKLOG-P0-P3.md", "REVISION-DE-LOS-GAPS.md"):
        p = _ROOT / "docs" / "audits" / "2026-09-07-coherencia-culinaria" / rel
        if p.exists():
            assert "P1-PLAN-LOTE-24" in p.read_text(encoding="utf-8", errors="ignore")


def test_el_god_file_bajo_el_techo():
    n = sum(1 for _ in (_BACKEND / "graph_orchestrator.py").open(encoding="utf-8", errors="ignore"))
    assert n <= 53_100, f"graph_orchestrator.py {n} líneas: extraer, no subir el tope"
