"""[P1-PLAN-LOTE-3 · 2026-09-11] Tercer lote del plan de pendientes (`docs/plan_pendientes_2026_09_11.md`).

  B5  Perfil `neutral` («Sin cocina en particular»): sin biblioteca ni mercado propios, la cocina ES el mercado.
      `diet.exclusions` llegan a `dish_registry.template_candidates(exclude_foods=)` desde el blueprint, el prompt y el
      día determinista. Cuando la cocina del día no tiene plato para una franja, el blueprint cae al mercado y lo
      ANOTA (`registry.culture_fallbacks`) y el informe de fidelidad lo publica (`culture_unavailable`).
  B1  El informe de fidelidad declara `checks_run`, `unmeasured` y `n_checks`, y mide tres dimensiones más:
      `culture_share_*`, `prep_time_over_budget`, `anchor_portion_*`. El equipo sigue sin medirse y se dice.
  B3  `computation` (+ `computation_hash`) en el informe y en la métrica: con QUÉ se calculó, separado de `input_hash`.
  B4  Semilla del run (`horizon.run_seed`): el prompt del esqueleto/día y el sembrador dejan de ser irrepetibles.

Cada test expresa el comportamiento ESPERADO. Ninguno codifica el defecto como especificación.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# B5 · perfil neutral
# ---------------------------------------------------------------------------

def test_b5_el_perfil_neutral_existe_y_no_tiene_pais_propio():
    import cultural_profiles as cp
    assert cp.is_profile("neutral") and cp.is_neutral_profile("neutral")
    assert cp.country_for_profile("neutral") is None
    assert cp.profile_name_es("neutral") == "Sin cocina en particular"
    assert "neutral" not in cp._PROFILE_BY_MARKET.values(), "ningún país sugiere el neutral solo"
    for cc in ("DO", "ES", "US"):
        assert cp.profile_for_market(cc) != "neutral"


def test_b5_la_cocina_del_neutral_es_el_mercado(monkeypatch):
    import cultural_profiles as cp
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.setenv("MEALFIT_CULTURAL_PROFILES", "true")
    ws = cp.weights_from_form_field({"main": "neutral", "secondary": []})
    assert ws == [{"profile_id": "neutral", "weight": 1.0}]
    assert cp.cultural_country_for_form_data({"country": "ES", "cultureProfiles": {"main": "neutral"}}) == "ES"
    assert cp.cultural_country_for_form_data({"country": "DO", "cultureProfiles": {"main": "neutral"}}) == "DO"
    # mezcla: la principal neutral cede el día al mercado; la secundaria conserva su país
    ws2 = cp.weights_from_form_field({"main": "neutral", "secondary": [{"profile_id": "spain_mediterranea", "intensity": "frecuente"}]})
    assert ws2[0]["profile_id"] == "neutral" and ws2[1]["profile_id"] == "spain_mediterranea"
    assert cp.staple_bases_for_day(ws, 0, ["Arroz", "Plátano verde"]) == [], "sin básicos propios no siembra nada"


def test_b5_el_blueprint_con_neutral_consulta_la_biblioteca_del_mercado(monkeypatch):
    import horizon
    import dish_registry as dr
    llamadas = []

    def _tc(country, slot, family=None, **kw):
        llamadas.append((country, slot, tuple(kw.get("exclude_foods") or ())))
        return [{"template_id": f"t_{country}_{slot}", "name": f"Plato {country} {slot}"}]
    monkeypatch.setattr(dr, "template_candidates", _tc)
    monkeypatch.setattr(dr, "registry_hash", lambda c=None: "h" + str(c))
    monkeypatch.setattr(dr, "registry_snapshot_version", lambda: 9)
    eff = {"market_country": "ES", "culture_weights": [{"profile_id": "neutral", "weight": 1.0}],
           "diet": {"type": "balanced", "allergies": [], "exclusions": ["hígado"]}}
    days = [{"day_index": 0, "slots": ["almuerzo"], "protein": "pollo", "culture": {"almuerzo": "neutral"}}]
    blk = horizon._registry_block_for_country(None, effective=eff, days_out=days)
    assert llamadas and llamadas[0][0] == "ES", "neutral ⇒ la biblioteca del mercado"
    assert llamadas[0][2] == ("hígado",), "las exclusiones viajan al selector"
    assert blk["candidates"] == {"0:almuerzo": ["t_ES_almuerzo"]}
    assert blk["culture_fallbacks"] == []


def test_b5_sin_plato_en_la_cocina_pedida_se_cae_al_mercado_y_queda_dicho(monkeypatch):
    import horizon
    import dish_registry as dr

    def _tc(country, slot, family=None, **kw):
        return [] if country == "DO" else [{"template_id": f"t_{country}", "name": f"Plato {country}"}]
    monkeypatch.setattr(dr, "template_candidates", _tc)
    monkeypatch.setattr(dr, "registry_hash", lambda c=None: "h" + str(c))
    monkeypatch.setattr(dr, "registry_snapshot_version", lambda: 9)
    eff = {"market_country": "ES", "culture_weights": [{"profile_id": "dominican_criolla", "weight": 1.0}],
           "diet": {"type": "vegan", "allergies": [], "exclusions": []}}
    days = [{"day_index": 3, "slots": ["cena"], "protein": None, "culture": {"cena": "dominican_criolla"}}]
    blk = horizon._registry_block_for_country("ES", effective=eff, days_out=days)
    assert blk["candidates"] == {"3:cena": ["t_ES"]}
    assert blk["culture_fallbacks"] == [{"day_index": 3, "slot": "cena", "profile": "dominican_criolla",
                                         "culture_country": "DO", "fallback_country": "ES", "found": 1}]
    # la rebanada conserva los de SUS días y el informe lo publica
    bp = {"days": [{"day_index": i, "slots": ["cena"], "anchors": []} for i in range(7)], "anchors": [],
          "recurrence": {"global_mode": "balanced"}, "registry": {**blk}}
    sl = horizon.slice_for_chunk(bp, 0, 7)
    assert sl["registry"]["culture_fallbacks"] == blk["culture_fallbacks"]
    assert horizon.slice_for_chunk(bp, 7, 7)["registry"]["culture_fallbacks"] == []
    issues, run, unmeasured = horizon.personalization_issues([{"meals": []}], sl, eff, {})
    assert [i["code"] for i in issues] == ["culture_unavailable"]
    assert issues[0]["day_index"] == 3 and issues[0]["slot"] == "cena"


def test_b5_el_selector_excluye_por_identidad_de_alimento_no_por_subcadena():
    import dish_registry as dr
    t = {"constituents": [{"name": "Queso fresco", "canonical": "Queso fresco"}, {"name": "Arroz"}]}
    assert dr._template_uses_excluded_food(t, ["res"], None) is False, "«res» ⊄ «queso fresco»"
    assert dr._template_uses_excluded_food(t, ["queso fresco"], None) is True
    assert dr._template_uses_excluded_food(t, ["QUESO FRESCO"], None) is True
    assert dr._template_uses_excluded_food(t, ["arroz"], None) is True
    assert dr._template_uses_excluded_food(t, ["pollo"], None) is False


def test_b5_el_dia_determinista_pasa_los_no_me_gusta_al_selector():
    src = _src("deterministic_day.py")
    assert "exclude_foods=_excluidos" in src
    assert 'dislikes' in src[src.find("_excluidos ="): src.find("_excluidos =") + 400]


def test_b5_el_chip_neutral_existe_en_el_frontend_y_en_los_cuatro_idiomas():
    js = (_FRONT / "src" / "config" / "cultures.js")
    if not js.exists():
        pytest.skip("frontend ausente en este checkout")
    s = js.read_text(encoding="utf-8")
    assert "{ id: 'neutral', labelKey: i18nKey('Sin cocina en particular'), marketDefault: null }" in s
    assert "neutral: t(" in s
    for loc in ("en-US", "pt-BR", "fr-FR", "it-IT"):
        cat = (_FRONT / "src" / "i18n" / "locales" / f"{loc}.json").read_text(encoding="utf-8")
        assert '"Sin cocina en particular":' in cat, loc
        assert '"Platos de cualquier cocina que venda tu mercado, sin sesgo":' in cat, loc


# ---------------------------------------------------------------------------
# B1 · métrica de personalización con cobertura declarada
# ---------------------------------------------------------------------------

def _plan(prep=("40 min", "registry")):
    return [{"day": 1, "meals": [
        {"meal": "almuerzo", "name": "Pollo guisado", "ingredients": ["2 huevos", "150 g de Pollo"],
         "prep_time": prep[0], "_prep_time_source": prep[1]},
        {"meal": "cena", "name": "Ensalada", "ingredients": ["3 huevos"], "prep_time": "10 min", "_prep_time_source": "registry"},
    ]}]


def test_b1_el_informe_declara_lo_medido_y_lo_no_medido():
    import horizon
    rep = horizon.fidelity_report(_plan(), None, {"food_anchors": []}, surface="t", form_data={"cookingTime": "1hour"})
    assert rep["checks_run"][:2] == ["exact_repeat", "ingredient_days"]
    assert "prep_time" in rep["checks_run"], "con techo (1 h) y platos con minutos declarados, el tiempo se mide"
    checks = {u["check"] for u in rep["unmeasured"]}
    assert "equipment" in checks, "el formulario no pregunta el equipo: se declara, no se finge"
    assert rep["n_checks"] == len(rep["checks_run"])
    assert "computation" in rep and rep["computation"]["computation_hash"]
    # `plenty` = sin techo: ni se mide ni se finge que falta el dato
    rep2 = horizon.fidelity_report(_plan(), None, {"food_anchors": []}, surface="t", form_data={"cookingTime": "plenty"})
    assert "prep_time" not in rep2["checks_run"] and "prep_time" not in {u["check"] for u in rep2["unmeasured"]}


def test_b1_tiempo_de_cocina_sobre_el_presupuesto_declarado():
    import horizon
    issues, run, unmeasured = horizon._prep_time_issues(_plan(("40 min", "registry")), {"cookingTime": "none"})
    assert run == ["prep_time"]
    assert [i["code"] for i in issues] == ["prep_time_over_budget"] and issues[0]["minutes"] == 40 and issues[0]["budget"] == 10
    issues2, run2, _ = horizon._prep_time_issues(_plan(("40 min", "registry")), {"cookingTime": "1hour"})
    assert issues2 == [] and run2 == ["prep_time"]
    # un tiempo inventado («unknown») no cuenta: no se mide contra un relleno
    _, run3, unmeasured3 = horizon._prep_time_issues([{"meals": [{"prep_time": "90 min", "_prep_time_source": "unknown"}]}], {"cookingTime": "none"})
    assert run3 == [] and unmeasured3 == [{"check": "prep_time", "reason": "no_meal_declares_minutes"}]
    # sin dato en el formulario: no medido, dicho
    assert horizon._prep_time_issues(_plan(), {})[2] == [{"check": "prep_time", "reason": "no_cooking_time_in_form"}]


def test_b1_raciones_de_las_anclas_en_piezas():
    import horizon
    eff = {"food_anchors": [{"ingredient_id": "huevo", "name": "Huevo", "portion": {"qty": 3, "unit": "unidades"}}]}
    issues, run, unmeasured = horizon._anchor_portion_issues(_plan(), eff)
    assert run == ["anchor_portion:huevo"] and unmeasured == []
    assert [(i["code"], i["day"], i["served"]) for i in issues] == [("anchor_portion_below", 1, 2.0)], issues
    # sin línea en piezas no se mide, y se dice
    eff2 = {"food_anchors": [{"ingredient_id": "pollo", "name": "Pollo", "portion": {"qty": 2, "unit": "unidades"}}]}
    _, run2, unmeasured2 = horizon._anchor_portion_issues(_plan(), eff2)
    assert run2 == [] and unmeasured2 == [{"check": "anchor_portion:pollo", "reason": "no_piece_line_found"}]


def test_b1_reparto_de_cocinas_servido_frente_al_pedido(monkeypatch):
    import horizon
    import recipe_library as rl
    idx = {"DO": {"mangu": "t1", "locrio": "t2", "sancocho": "t3"}, "ES": {"tortilla": "t4", "gazpacho": "t5", "cocido": "t6"}}
    monkeypatch.setattr(rl, "_registry_name_index", lambda country="DO": idx.get(country, {}))
    eff = {"culture_weights": [{"profile_id": "dominican_criolla", "weight": 0.7}, {"profile_id": "spain_mediterranea", "weight": 0.3}]}
    days = [{"meals": [{"name": n} for n in ("Tortilla", "Gazpacho", "Cocido", "Mangú", "Locrio")]}]
    issues, run, unmeasured = horizon._culture_share_issues(days, eff)
    assert run == ["culture_share"]
    codes = sorted(i["code"] for i in issues)
    assert codes == ["culture_share_above", "culture_share_below"], issues
    # con pocos platos identificados no se juzga
    _, run2, unmeasured2 = horizon._culture_share_issues([{"meals": [{"name": "Mangú"}]}], eff)
    assert run2 == [] and unmeasured2[0]["reason"] == "too_few_identified"
    # una sola cocina: nada que medir
    assert horizon._culture_share_issues(days, {"culture_weights": [{"profile_id": "dominican_criolla", "weight": 1.0}]}) == ([], [], [])


# ---------------------------------------------------------------------------
# B3 · huella de la computación
# ---------------------------------------------------------------------------

def test_b3_la_huella_de_la_computacion_cambia_con_el_sistema_no_con_el_usuario():
    import horizon
    a = horizon.computation_stamp({}, {"user_id": "u1", "_days_offset": 0}, attempt=1)
    b = horizon.computation_stamp({}, {"user_id": "u1", "_days_offset": 0}, attempt=1)
    assert a["computation_hash"] == b["computation_hash"], "determinista"
    assert set(a) >= {"registry_hash", "catalog_generation", "day_prompt_hash", "model", "seed", "code_marker", "computation_hash"}
    c = horizon.computation_stamp({}, {"user_id": "u1", "_days_offset": 0}, attempt=2)
    assert c["seed"] != a["seed"] and c["computation_hash"] != a["computation_hash"], "otro intento, otra semilla"
    assert a["day_prompt_hash"] and len(a["day_prompt_hash"]) == 16


def test_b3_la_metrica_lleva_la_huella():
    src = _src("horizon.py")
    i = src.find("def emit_fidelity_metric(")
    assert '"computation_hash": (report.get("computation") or {}).get("computation_hash")' in src[i:i + 3000]
    assert '"unmeasured":' in src[i:i + 3000] and '"checks_run":' in src[i:i + 3000]


# ---------------------------------------------------------------------------
# B4 · semilla del run
# ---------------------------------------------------------------------------

def test_b4_la_semilla_sale_del_run_y_varia_por_intento(monkeypatch):
    import horizon
    fd = {"_blueprint_slice": {"slice_hash": "abc123"}, "_days_offset": 14, "user_id": "u1"}
    s1 = horizon.run_seed(fd, attempt=1)
    assert s1 == horizon.run_seed(dict(fd), attempt=1) and 10000 <= s1 <= 99999
    assert horizon.run_seed(fd, attempt=2) != s1
    assert horizon.run_seed({**fd, "_days_offset": 15}, attempt=1) != s1, "otro bloque, otra semilla"
    assert horizon.run_seed({}, attempt=1) is None, "sin de qué derivarla, el azar de siempre"
    monkeypatch.setenv("MEALFIT_SEED_FROM_RUN", "false")
    assert horizon.run_seed(fd, attempt=1) is None


def test_b4_el_sembrador_usa_un_generador_sembrado_sin_cambiar_su_codigo():
    import ai_helpers
    src = _src("ai_helpers.py")
    assert "random = _seeder_rng(form_data)" in src
    assert "random.shuffle(unique_proteins)" in src, "el código del sembrador no cambia de letra (los anclajes viejos siguen)"
    rng = ai_helpers._seeder_rng({"_blueprint_slice": {"slice_hash": "z"}, "user_id": "u"})
    assert isinstance(rng, random.Random) and not isinstance(rng, type(random))
    assert ai_helpers._seeder_rng({}) is random
    a = ai_helpers._pick_light_anchor_candidates(list("abcdefgh"), 4, rng=random.Random(7))
    b = ai_helpers._pick_light_anchor_candidates(list("abcdefgh"), 4, rng=random.Random(7))
    assert a == b


def test_b4_el_prompt_del_esqueleto_y_del_dia_usan_la_semilla_del_run():
    src = _src("graph_orchestrator.py")
    assert src.count('run_seed(form_data, attempt=state.get("attempt")) or random.randint(10000, 99999)') == 2
    assert "random_seed = random.randint(10000, 99999)" not in src
    assert len(src.splitlines()) <= 53100


def test_marker_bumpeado():
    import app
    assert "[P1-PLAN-LOTE-3 · 2026-09-11]" in _src("app.py")
    # [P1-PLAN-LOTE-13 · 2026-09-12] «no anterior a este lote», no «igual a hoy»: el pin de la fecha y del prefijo
    # `P1-PLAN-` rompía 12 tests el primer día en que otro P-fix bumpeaba el marker.
    assert app._LAST_KNOWN_PFIX.split("·")[-1].strip() >= "2026-09-11", app._LAST_KNOWN_PFIX
