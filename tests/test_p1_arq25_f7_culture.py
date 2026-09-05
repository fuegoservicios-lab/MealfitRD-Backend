"""[P1-ARQ25-F7-CULTURE · 2026-09-05] Fase 7 (subfase A): cultura separada del mercado (I16). Seis perfiles como
DATA sobre un motor genérico; mezcla principal + hasta dos secundarias con intensidad; reparto determinista por
día; las superficies CULTURALES (plantillas, inspiración, hábitos de franja, arroz nocturno, juez) leen la puerta
cultural y las de MERCADO (precios, catálogo, moneda, unidades, despensa) siguen leyendo `country_for_form_data`.
Gate del roadmap: market_country=US + dominican_criolla 0.7 ⇒ platos criollos con precios y catálogo de US.
"""
import re
from pathlib import Path

import pytest

import cultural_profiles as cp

_BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _knob(monkeypatch):
    monkeypatch.delenv("MEALFIT_CULTURAL_PROFILES", raising=False)


def test_a_seis_perfiles_como_data_y_ninguna_cadena_if_elif():
    assert set(cp.profile_ids()) == {"dominican_criolla", "puertorico_criolla", "mexico_casera", "colombia_casera", "spain_mediterranea", "us_everyday"}
    for pid, p in cp.PROFILES.items():
        assert p["library"] in ("do", "es", "mx", "co", "pr", "us") and p["market_default"] and p["name_es"]
        assert p["staples"] and p["dish_families"] and p["techniques"] and p["flavor_base"] and p["slot_affinity"]
    src = (_BACKEND / "cultural_profiles.py").read_text(encoding="utf-8")
    assert not re.search(r"if\s+\w+\s*==\s*['\"](dominican_criolla|mexico_casera|spain_mediterranea)['\"]", src), "§9.4: nada de un if/elif por cultura"


def test_b_mezcla_valida_principal_mayoritaria_y_hasta_dos_secundarias():
    ws = cp.normalize_weights([{"profile_id": "dominican_criolla", "weight": 0.7}, {"profile_id": "us_everyday", "weight": 0.3}])
    assert [w["profile_id"] for w in ws] == ["dominican_criolla", "us_everyday"] and abs(sum(w["weight"] for w in ws) - 1) < 1e-6
    # la principal nunca baja de 0.5; 4 perfiles → se recortan a 3; ids inválidos y duplicados fuera
    ws = cp.normalize_weights([{"profile_id": "us_everyday", "weight": 0.2}, {"profile_id": "mexico_casera", "weight": 0.2},
                               {"profile_id": "spain_mediterranea", "weight": 0.2}, {"profile_id": "colombia_casera", "weight": 0.2},
                               {"profile_id": "marte", "weight": 9}, {"profile_id": "us_everyday", "weight": 5}])
    assert len(ws) == 3 and ws[0]["weight"] >= 0.5 and abs(sum(w["weight"] for w in ws) - 1) < 1e-6
    assert cp.normalize_weights([]) == [{"profile_id": "dominican_criolla", "weight": 1.0}]
    assert cp.normalize_weights([{"profile_id": "nada", "weight": 1}], default_profile="mexico_casera")[0]["profile_id"] == "mexico_casera"


def test_c_campo_del_formulario_con_intensidades():
    ws = cp.weights_from_form_field({"main": "dominican_criolla", "secondary": [{"profile_id": "spain_mediterranea", "intensity": "frecuente"}]})
    assert ws[0]["profile_id"] == "dominican_criolla" and ws[0]["weight"] == pytest.approx(0.7) and ws[1]["weight"] == pytest.approx(0.3)
    # dos predominantes suman 0.9 → se recortan a 0.5 entre las dos; la principal conserva 0.5
    ws = cp.weights_from_form_field({"main": "us_everyday", "secondary": [{"profile_id": "mexico_casera", "intensity": "predominante"},
                                                                          {"profile_id": "colombia_casera", "intensity": "predominante"}]})
    assert ws[0]["weight"] == pytest.approx(0.5) and len(ws) == 3
    assert cp.weights_from_form_field(None) is None and cp.weights_from_form_field({"main": "x"}) is None
    # la principal repetida como secundaria se ignora
    assert cp.weights_from_form_field({"main": "us_everyday", "secondary": [{"profile_id": "us_everyday", "intensity": "frecuente"}]}) == [{"profile_id": "us_everyday", "weight": 1.0}]


def test_d_sin_eleccion_la_cocina_del_pais_de_compra_y_knob_apagado_legacy(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "1")  # la cocina por defecto es la del MERCADO, que respeta su knob
    assert cp.culture_weights_for_form({"country": "MX"}) == [{"profile_id": "mexico_casera", "weight": 1.0}]
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    assert cp.culture_weights_for_form({"country": "MX"}) == [{"profile_id": "dominican_criolla", "weight": 1.0}], "países apagado ⇒ mercado DO ⇒ cocina DO"
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "1")
    assert cp.culture_weights_for_form({}) == [{"profile_id": "dominican_criolla", "weight": 1.0}]
    form = {"country": "US", "cultureProfiles": {"main": "dominican_criolla", "secondary": [{"profile_id": "us_everyday", "intensity": "frecuente"}]}}
    assert cp.culture_weights_for_form(form)[0]["profile_id"] == "dominican_criolla"
    monkeypatch.setenv("MEALFIT_CULTURAL_PROFILES", "0")
    assert cp.culture_weights_for_form(form) == [{"profile_id": "us_everyday", "weight": 1.0}], "knob apagado ⇒ cultura = mercado"


def test_e_reparto_determinista_por_dia():
    ws = [{"profile_id": "dominican_criolla", "weight": 0.7}, {"profile_id": "us_everyday", "weight": 0.3}]
    seq = [cp.profile_for_day(ws, d) for d in range(10)]
    assert seq.count("dominican_criolla") == 7 and seq.count("us_everyday") == 3
    assert seq[0] == "dominican_criolla", "la principal abre"
    assert seq == [cp.profile_for_day(ws, d) for d in range(10)], "determinista"
    assert cp.profile_for_day([{"profile_id": "mexico_casera", "weight": 1.0}], 5) == "mexico_casera"


def test_f_gate_i16_mercado_us_con_cocina_dominicana():
    form = {"country": "US", "cultureProfiles": {"main": "dominican_criolla", "secondary": [{"profile_id": "us_everyday", "intensity": "frecuente"}]}}
    from constants import country_for_form_data, cultural_country_for_form_data
    import plan_policy as pp
    monkey_country = country_for_form_data(form)
    assert cultural_country_for_form_data(form) == "DO", "la cocina principal manda en las superficies culturales"
    assert cultural_country_for_form_data(form, day_index=0) == "DO"
    assert monkey_country in ("US", "DO"), "el mercado sigue saliendo de country_for_form_data (con el knob de países apagado cae a DO)"
    compiled = pp.compile_from_form(form)
    eff = (compiled or {}).get("effective") if isinstance(compiled, dict) else None
    assert eff and [w["profile_id"] for w in eff["culture_weights"]] == ["dominican_criolla", "us_everyday"]
    assert eff["market_country"] == monkey_country


def test_g_blueprint_y_prompt_llevan_la_cocina_del_dia(tmp_path, monkeypatch):
    import horizon as hz
    eff = {"policy_hash": "h", "recurrence": {"global_mode": "balanced"}, "market_country": "US", "diet": {"allergies": []},
           "culture_weights": [{"profile_id": "dominican_criolla", "weight": 0.7}, {"profile_id": "us_everyday", "weight": 0.3}],
           "shopping": {"main_cycle_days": 7, "fresh_topup_days": None}}
    bp = hz.build_blueprint(eff, total_days=10)
    cultures = [list(d["culture"].values())[0] for d in bp["days"]]
    assert cultures.count("dominican_criolla") == 7 and cultures.count("us_everyday") == 3
    assert bp["culture_profile"] == "dominican_criolla" and bp["culture_weights"][0]["profile_id"] == "dominican_criolla"
    assert bp["registry"]["snapshot_hash"] is None or isinstance(bp["registry"].get("library_hashes"), dict)
    # el encabezado de inspiración nombra la mezcla; un perfil solo conserva el literal histórico
    import dish_library as dl
    assert dl._inspiration_heading(None, eff["culture_weights"]).startswith("INSPIRACIÓN: COCINA DOMINICANA 70 %")
    assert dl._inspiration_heading("DO") == "INSPIRACIÓN DOMINICANA" == dl._inspiration_heading(None, [{"profile_id": "dominican_criolla", "weight": 1.0}])


def test_h_las_superficies_culturales_leen_la_puerta_cultural_y_las_de_mercado_no():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    for anchor in ("_cj_country = cultural_country_for_form_data(form_data)", "_apn_country = cultural_country_for_form_data(form_data)",
                   "_dcl_country = cultural_country_for_form_data(form_data)", "_critique_country = cultural_country_for_form_data(form_data)",
                   "_rules_table = slot_rules_for_country(_country)"):
        assert anchor in src, anchor
    i = src.index("_rules_table = slot_rules_for_country(_country)")
    assert "_country = cultural_country_for_form_data(form_data)" in src[i - 200:i]
    # el contexto de inspiración del día lleva la cocina del DÍA
    assert "country=cultural_country_for_form_data(form_data)" in src or "country=cultural_country_for_form_data(form_data, " in src
    # mercado intacto: la validación contra la despensa y los precios siguen con country_for_form_data
    assert "country=country_for_form_data(form_data),\n                )\n                if val_result is not True:" in src
    assert src.count("country_for_form_data(form_data)") >= 8, "las superficies de mercado no se tocaron"


def test_i_las_superficies_post_generacion_leen_la_cocina_sellada_en_el_plan(monkeypatch):
    """Cierre: swap, regenerar día y el coach mutan un plan YA generado; su cocina sale del sello `_plan_policy`,
    no del mercado. Sin sello (plan anterior a F7) caen al país de mercado del plan: legado byte-idéntico."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "1")
    from constants import cultural_country_for_plan, country_for_plan
    sealed = {"_country": "US", "_plan_policy": {"effective": {"culture_weights": [
        {"profile_id": "dominican_criolla", "weight": 0.7}, {"profile_id": "us_everyday", "weight": 0.3}]}}}
    assert country_for_plan(sealed, {}) == "US" and cultural_country_for_plan(sealed, {}) == "DO"
    assert cultural_country_for_plan(sealed, {}, day_index=1) == "US", "el reparto por día es el mismo del blueprint"
    legacy = {"_country": "ES"}
    assert cultural_country_for_plan(legacy, {}) == "ES" == country_for_plan(legacy, {})
    # sin sello, la elección viva del perfil; con el knob apagado, el mercado
    assert cultural_country_for_plan(legacy, {"cultureProfiles": {"main": "mexico_casera"}}) == "MX"
    monkeypatch.setenv("MEALFIT_CULTURAL_PROFILES", "0")
    assert cultural_country_for_plan(sealed, {}) == "US"
    monkeypatch.delenv("MEALFIT_CULTURAL_PROFILES", raising=False)
    # el motor del swap recibe los pesos ya compilados (`_culture_weights`) y los prefiere al perfil
    assert cultural_country_for_form_data_via_stamp() == "DO"


def cultural_country_for_form_data_via_stamp():
    from constants import cultural_country_for_form_data
    return cultural_country_for_form_data({"country": "US", "_culture_weights": [{"profile_id": "dominican_criolla", "weight": 1.0}],
                                           "cultureProfiles": {"main": "us_everyday"}})


def test_j_swap_regen_y_coach_cablean_la_puerta_cultural_y_dejan_el_mercado_en_su_sitio():
    agent = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    assert "_swap_culture = _ccffd_swap(form_data)" in agent
    for cultural in ("country=_swap_culture,", "_bmtr(meal_type, _swap_culture)", "build_swap_meal_prompt_template(_swap_culture)",
                     "slot_coherence_backstop_for_meal(_slot_dump, meal_type, _swap_culture)", "_swap_slot_feedback_suffix(_swap_culture,",
                     "_swap_raw_staple_feedback_suffix(_swap_culture,"):
        assert cultural in agent, cultural
    for market in ("swap_allergies, swap_dislikes, swap_diet, country=_swap_country", "_safe_high_density_proteins(allergies, _cl_db, country=_swap_country)"):
        assert market in agent, market
    tools = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    assert "_modify_culture = _ccfp_modify(plan_data, form_data)" in tools
    for cultural in ("slot_rules_for_country(_modify_culture)", "build_modify_meal_prompt_template(_modify_culture)",
                     "country=_modify_culture)", "build_meal_timing_rules(meal_type, _modify_culture)"):
        assert cultural in tools, cultural
    assert "ingreds, clean_ingredients, country=_modify_country" in tools, "la despensa sigue siendo de mercado"
    plans = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    assert plans.count("AS plan_culture") == 3, "los tres SELECT del sello leen también la cocina"
    assert 'data["_culture_weights"] = _cw_swap' in plans and 'data["_culture_weights"] = _cw_rd' in plans
    assert '"_culture_weights": data.get("_culture_weights")' in plans, "regenerar día reenvía el sello a cada swap"
    assert 'country=_swap_culture)' in plans and '_micro_form["country"] = _swap_country' in plans
