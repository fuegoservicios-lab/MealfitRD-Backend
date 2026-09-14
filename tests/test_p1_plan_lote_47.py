# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-47 · 2026-09-14] Tercera prueba RD del dueño (plan d8b10b05), tras el lote 46.

El plan salió aprobado al primer intento y sin repetir los planes anteriores (lo del 46 funcionó), pero ninguno de sus 3
días llegó como lo armó el día determinista: la autocrítica corrigió los días 1 y 2 con el LLM y la regeneración
quirúrgica el 3, y el corrector pegó encima días sin `_template_id` — 6 platos de biblioteca sobrevivieron por el NOMBRE
y las protecciones de los lotes 45 y 46 se apagaron (huevo→queso «revuelve queso blanco», tiempo en el paso del agua fría,
batata 200 → 10 g). Tres piezas:

  1. Re-elegir, no reescribir (`reeleccion_dia`): el día determinista señalado se rearma sin LLM; lo que el corrector deja
     igual vuelve con su procedencia.
  2. El armador aprende las reglas fijas de la autocrítica: proteína del día por lo que el plato LLEVA, cena fuerte en
     ganancia muscular, básicos / proteína pesada / plato-base entre días del bloque, franja y base ligera.
  3. Parches: el paso que enfría no lleva tiempo de fuego, el casabe se tuesta, el queso no se revuelve.
"""
from __future__ import annotations

import copy
import inspect
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]

import deterministic_day as dd  # noqa: E402
import graph_orchestrator as go  # noqa: E402
import pasos_sustitucion as ps  # noqa: E402
import reeleccion_dia as rd  # noqa: E402

_GM = {"mainGoal": "gain_muscle", "dietType": "balanced"}
_KNOBS_ARMADOR = ("MEALFIT_DETERMINISTIC_DAY_PROTEIN_BY_CONTENT", "MEALFIT_DETERMINISTIC_DAY_GAINMUSCLE_DINNER",
                  "MEALFIT_DETERMINISTIC_DAY_BLOCK_STAPLES", "MEALFIT_DETERMINISTIC_DAY_BLOCK_HEAVY_PROTEIN",
                  "MEALFIT_DETERMINISTIC_DAY_SLOT_COHERENCE", "MEALFIT_DETERMINISTIC_DAY_BLOCK_DISH_BASE",
                  "MEALFIT_DETERMINISTIC_DAY_LIGHT_BASE")


def _m(slot, nombre, ings, tid=None):
    m = {"meal": slot, "name": nombre, "ingredients": list(ings)}
    if tid:
        m.update({"_template_id": tid, "_recipe_source": "library", "recipe": ["Paso de biblioteca."]})
    return m


TORTILLA = _m("Desayuno", "Tortilla de harina de maíz con queso cheddar",
              ["65 g de Harina de maíz precocida", "40 g de Queso cheddar", "50 g de Huevo"])
MORO = _m("Almuerzo", "Moro de habichuelas negras con huevo frito",
          ["60 g de Arroz blanco", "50 g de Habichuelas negras", "50 g de Huevo"])
BOLLITOS = _m("Cena", "Bollitos de yuca rellenos de queso, horneados", ["180 g de Yuca", "60 g de Queso blanco", "25 g de Huevo"])
POLLO = _m("Cena", "Pollo guisado con batata", ["150 g de Pechuga de pollo", "200 g de Batata"])


# ─────────────── 2. las reglas de la autocrítica, en el armador ───────────────
def test_la_proteina_del_dia_se_lee_en_lo_que_el_plato_lleva():
    """La tortilla de maíz lleva 50 g de huevo con la etiqueta «queso»: la puerta de la etiqueta la dejaba pasar y el día 3
    salió con huevo en el desayuno y en el almuerzo. El detector del revisor, sobre el plato armado, sí lo ve."""
    assert dd._repite_proteina([TORTILLA], MORO, _GM) is True
    familia = dd._plato_de_familia("Huevo")
    assert dd._repite_proteina([familia], TORTILLA, _GM) is True, "la familia del almuerzo no se gasta en el desayuno"
    assert dd._repite_proteina([familia], BOLLITOS, _GM) is False, "el huevo que liga una masa no cuenta (como en el gate)"
    assert dd._repite_proteina([MORO, TORTILLA], POLLO, _GM) is False, "sólo cuenta lo que el candidato añade"
    assert dd._plato_de_familia("Queso") is None and dd._plato_de_familia("Pescado") is not None
    assert "_prot in _labels_var" in inspect.getsource(dd.build_day_for_skeleton), "con el knob apagado, la etiqueta"


def test_la_cena_debil_de_ganancia_muscular_con_el_detector_de_la_autocritica():
    assert dd._cena_debil(BOLLITOS, _GM) is True
    assert dd._cena_debil(POLLO, _GM) is False
    assert dd._cena_debil(BOLLITOS, {"mainGoal": "gain_muscle", "dietType": "vegetarian"}) is False, "insatisfacible en veg"
    assert dd._cena_debil(BOLLITOS, {"mainGoal": "lose_fat"}) is False


@pytest.mark.parametrize("comida", [BOLLITOS, TORTILLA, _m("Merienda", "Avena con lechosa y piña", ["50 g de avena", "80 g de lechosa"]),
                                    _m("Desayuno", "Yogurt griego con guineo", ["170 g de yogurt griego", "1 guineo"])])
def test_los_basicos_con_el_criterio_exacto_de_la_autocritica(comida):
    """Paridad con `_count_staple_repetitions`: dos días con el mismo plato devuelven justo sus básicos."""
    assert dd._basicos_de(comida) == set(go._count_staple_repetitions([{"meals": [comida]}, {"meals": [comida]}]))


def test_basicos_proteinas_pesadas_y_platos_base_del_bloque():
    dia = lambda *ms, **kw: dict({"meals": list(ms)}, **kw)  # noqa: E731
    mem = [dia(BOLLITOS), dia(_m("Cena", "Pollo al horno", ["150 g de pechuga de pollo"])),
           dia(_m("Cena", "Pollo a la plancha", ["150 g de pechuga de pollo"])), dia(BOLLITOS, _persistido=True)]
    assert dd._basicos_del_bloque(mem, {}) == {"yuca", "queso blanco"}
    assert dd._basicos_del_bloque([dia(BOLLITOS, _persistido=True)], {}) == set(), "un bloque anterior no es el bloque"
    rutina = {"_policy_enforced": True, "_plan_policy_effective": {"recurrence": {"global_mode": "routine"}}}
    assert dd._basicos_del_bloque(mem, rutina) == set(), "con rutina, repetir es lo pedido (el mismo filtro que la autocrítica)"
    assert dd._pesadas_vetadas(mem, {}) == {"pollo"}, "pollo en 2 días: un tercero es la monotonía"
    assert dd._pesadas_de(POLLO) == {"pollo"} and dd._pesadas_de(BOLLITOS) == set()
    tok = dd._plato_base_de({"name": "Guiso de res con papas"})
    assert tok and tok == dd._plato_base_de({"name": "Guiso de pollo criollo"})
    guisos = [dia({"meal": "Cena", "name": "Guiso de res con papas", "ingredients": []}),
              dia({"meal": "Cena", "name": "Guiso de pollo criollo", "ingredients": []})]
    assert tok in dd._platos_base_vetados(guisos, {})


def test_la_franja_con_los_detectores_de_la_autocritica():
    almuerzo = _m("Almuerzo", "Pinchos de pollo con yuca hervida", ["150 g de pechuga de pollo", "200 g de yuca"])
    assert dd._rompe_franja([almuerzo], _m("Cena", "Tilapia al horno con yuca", ["150 g de tilapia", "200 g de yuca"]), _GM)
    assert not dd._rompe_franja([almuerzo], _m("Cena", "Pescado al horno con batata",
                                               ["150 g de filete de pescado", "200 g de batata"]), _GM)


_TPL = {t["template_id"]: t for t in (
    {"template_id": "tpl_tortilla", "name": TORTILLA["name"], "protein": "queso", "slots": ["desayuno"],
     "constituents": [{"name": "Harina de maíz precocida", "grams": 65}, {"name": "Queso cheddar", "grams": 40},
                      {"name": "Huevo", "grams": 50}]},
    {"template_id": "tpl_mangu", "name": "Mangú con queso frito", "protein": "queso", "slots": ["desayuno"],
     "constituents": [{"name": "Plátano verde", "grams": 200}, {"name": "Queso blanco", "grams": 60}]},
    {"template_id": "tpl_avena", "name": "Avena con canela y leche", "protein": "none", "slots": ["desayuno"],
     "constituents": [{"name": "Avena", "grams": 50}, {"name": "Leche", "grams": 200}]},
    {"template_id": "tpl_moro", "name": MORO["name"], "protein": "huevo", "slots": ["almuerzo"],
     "constituents": [{"name": "Arroz blanco", "grams": 60}, {"name": "Habichuelas negras", "grams": 50},
                      {"name": "Huevo", "grams": 50}]},
    {"template_id": "tpl_yogur", "name": "Yogur natural con fresas", "protein": "yogur", "slots": ["merienda"],
     "constituents": [{"name": "Yogur natural", "grams": 150}, {"name": "Fresas", "grams": 80}]},
    {"template_id": "tpl_bollitos", "name": BOLLITOS["name"], "protein": "queso", "slots": ["cena"],
     "constituents": [{"name": "Yuca", "grams": 180}, {"name": "Queso blanco", "grams": 60}, {"name": "Huevo", "grams": 25}]},
    {"template_id": "tpl_pollo", "name": POLLO["name"], "protein": "pollo", "slots": ["cena"],
     "constituents": [{"name": "Pechuga de pollo", "grams": 150}, {"name": "Batata", "grams": 200}]},
)}


@pytest.fixture
def armador(monkeypatch):
    """El armador de verdad con un registro de juguete: el orden de los candidatos es el del registro, así que la
    primera opción de cada franja es la que el día del plan d8b10b05 sirvió (tortilla, bollitos)."""
    import dish_registry as dr
    import shopping_calculator as sc
    nombres = {c["name"] for t in _TPL.values() for c in t["constituents"]}
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: [{"name": n, "kcal_per_100g": 100} for n in sorted(nombres)])
    monkeypatch.setattr(dr, "templates_by_id", lambda country: dict(_TPL))
    monkeypatch.setattr(dr, "template_candidates", lambda country, slot, family=None, **kw: [
        {"template_id": tid} for tid, t in _TPL.items()
        if slot in t["slots"] and (not family or t["protein"] == str(family).lower())])
    monkeypatch.setattr(dd, "elegir_con_tiempo", lambda tids, obj, cat, por_id, slot, presupuesto=None, **kw: [
        (por_id[t], 1.0) for t in tids if t in por_id])

    def _comida(t, f, cat, slot, country, obj=None):
        ings = [f"{c['grams']:g} g de {c['name']}" for c in t["constituents"]]
        return {"meal": slot.capitalize(), "name": t["name"], "ingredients": ings, "ingredients_raw": list(ings),
                "calories": 500, "protein": "30g", "carbs": "50g", "fats": "15g", "_template_id": t["template_id"],
                "_recipe_source": "library", "recipe": ["Paso."]}
    monkeypatch.setattr(dd, "construir_comida", _comida)
    monkeypatch.setattr(dd, "verifica_comida", lambda meal, fd, cat: [])
    monkeypatch.setattr(dd, "_sodio_de_comida", lambda *a, **k: (100.0, 0))
    monkeypatch.setattr(dd, "_dias_previos_persistidos", lambda fd, uid: [])
    monkeypatch.setattr(dd, "_plantillas_de_planes_recientes", lambda *a, **k: set())
    monkeypatch.setattr(dd, "deterministic_day_for_user", lambda uid: True)
    fd = dict(_GM, user_id="u1", _blueprint_slice={"days_offset": 0, "days_count": 3,
                                                   "days": [{"day_index": 0, "protein": "Huevo"}]})
    nut = {"target_calories": "2100 kcal", "macros": {"protein": "123g", "carbs": "271g", "fats": "58g"}}
    esq = {"day": 1, "meal_types": ["Desayuno", "Almuerzo", "Merienda", "Cena"], "protein_pool": []}

    def _armar(**kw):
        dia = dd.build_day_for_skeleton(nut, fd, esq, 1, memoria=[], **kw)
        assert dia is not None
        return [m["name"] for m in dia["meals"]]
    return _armar


def test_el_dia_del_plan_d8b10b05_ya_no_se_arma_asi(armador, monkeypatch):
    """Día de familia huevo en ganancia muscular. Antes: tortilla (huevo escondido) + moro con huevo, y bollitos de queso
    de cena — lo que la autocrítica reescribió con el LLM. Ahora: mangú, moro, pollo."""
    for k in _KNOBS_ARMADOR:
        monkeypatch.setenv(k, "false")
    assert armador() == [TORTILLA["name"], MORO["name"], "Yogur natural con fresas", BOLLITOS["name"]]
    for k in _KNOBS_ARMADOR:
        monkeypatch.delenv(k, raising=False)
    assert armador() == ["Mangú con queso frito", MORO["name"], "Yogur natural con fresas", POLLO["name"]]
    assert armador(evitar={"plantillas": {"tpl_mangu"}}) == [
        "Avena con canela y leche", MORO["name"], "Yogur natural con fresas", POLLO["name"]], "lo que el llamador pide evitar"


def test_de_reserva_queda_el_que_menos_rompe_y_la_repeticion_pesa_a_medias():
    src = inspect.getsource(dd.build_day_for_skeleton)
    assert "_reserva_var = (_c, _t, _na)" in src and "_faltas < _reserva_faltas" in src
    assert "0.5 * bool(_max_rep and" in src, "la cuota agotada pesa menos que lo que hace saltar la autocrítica"
    for ancla in ("_repite_proteina(meals, _c, _fd)", "_cena_debil(_c, _fd)", "_rompe_franja(meals, _c, _fd)",
                  "_basicos_del_bloque(memoria, _fd)", "_pesadas_vetadas(memoria, _fd)", "_platos_base_vetados(memoria, _fd)",
                  "_evitar_t", "_reservada"):
        assert ancla in src, ancla


# ─────────────── 1. re-elegir, no reescribir ───────────────
def _dias_bloque():
    d1 = {"day": 1, "_day_source": "deterministic", "meals": [
        _m("Almuerzo", "Pinchos de pollo con yuca hervida", ["150 g de pechuga de pollo", "200 g de yuca"], "t_pinchos"),
        _m("Cena", "Pescado al horno con batata", ["150 g de filete de pescado", "200 g de batata"], "t_pescado")]}
    d2 = {"day": 2, "_day_source": "deterministic", "meals": [
        _m("Almuerzo", "Espaguetis con sardinas", ["80 g de pasta", "100 g de sardinas en lata"], "t_espaguetis"),
        dict(BOLLITOS, _template_id="t_bollitos", _recipe_source="library", recipe=["x"])]}
    d3 = {"day": 3, "_day_source": "deterministic", "meals": [
        _m("Desayuno", "Revoltillo de huevo con plátano", ["2 huevos", "150 g de plátano"], "t_revoltillo"),
        _m("Almuerzo", "Moro de habichuelas negras con huevo frito", ["60 g de arroz blanco", "50 g de huevo"], "t_moro")]}
    return [d1, d2, d3]


_NUEVO_D2 = {"day": 2, "_day_source": "deterministic", "meals": [
    _m("Almuerzo", "Espaguetis con sardinas", ["80 g de pasta", "100 g de sardinas en lata"], "t_espaguetis"),
    _m("Cena", "Res guisada con plátano", ["150 g de carne de res", "150 g de plátano"], "t_res")]}
_NUEVO_D3 = {"day": 3, "_day_source": "deterministic", "meals": [
    _m("Desayuno", "Mangú con queso frito", ["200 g de plátano verde", "60 g de queso blanco"], "t_mangu"),
    _m("Almuerzo", "Moro de habichuelas negras con huevo frito", ["60 g de arroz blanco", "50 g de huevo"], "t_moro")]}


def test_las_senales_son_las_de_la_autocritica_y_lo_que_se_pide_evitar():
    dias = _dias_bloque()
    assert rd.senales(dias, 1, _GM) == [("basico", "yuca")]
    tipos2 = {t for t, _ in rd.senales(dias, 2, _GM)}
    assert tipos2 == {"basico", "cena_debil"}
    assert ("proteina_del_dia", "") in rd.senales(dias, 3, _GM)
    ev = rd.evitar_para(dias, 2, _GM, ["Día 2: cambia 'Espaguetis con sardinas' por algo más ligero"])
    assert ev["plantillas"] == {"t_bollitos", "t_espaguetis"} and ev["basicos"] == {"yuca"}


def test_reelegir_mide_tras_cada_cambio_y_barre_los_no_nombrados(monkeypatch):
    llamadas = []

    def _fake(nutrition, form_data, skeleton_day, day_num, memoria=None, evitar=None):
        llamadas.append((day_num, set((evitar or {}).get("plantillas") or ())))
        return copy.deepcopy({2: _NUEVO_D2, 3: _NUEVO_D3}.get(day_num))
    monkeypatch.setattr(dd, "build_day_for_skeleton", _fake)
    dias = _dias_bloque()
    llm, rearmados, conservados = rd.reelegir_en_lugar(dias, [2, 1], nutrition={}, form_data=_GM, etiqueta="t")
    assert (llm, rearmados, conservados) == ([], [2], [1]), "sin la yuca del día 2, el día 1 ya no tiene señal"
    assert dias[1]["meals"][1]["name"] == "Res guisada con plátano" and dias[1]["_reelegido_por"] == ["basico", "cena_debil"]
    assert dias[2]["meals"][0]["name"] == "Revoltillo de huevo con plátano", "sin barrer, el día 3 no se toca"
    dias = _dias_bloque()
    _, rearmados, _ = rd.reelegir_en_lugar(dias, [2, 1], nutrition={}, form_data=_GM, etiqueta="t", barrer=True)
    assert rearmados == [2, 3] and dias[2]["meals"][0]["name"] == "Mangú con queso frito", "el huevo repetido, sin LLM"


def test_reelegir_conserva_lo_que_nadie_senala_y_cede_al_llm_lo_que_no_mejora(monkeypatch):
    dias = _dias_bloque()
    limpio = {"day": 4, "_day_source": "deterministic", "meals": [_m("Almuerzo", "Pescado a la plancha", ["150 g de tilapia"], "t_x")]}
    dias.append(limpio)
    monkeypatch.setattr(dd, "build_day_for_skeleton", lambda *a, **k: pytest.fail("nada verificable: no se rearma"))
    assert rd.reelegir(dias, 4, nutrition={}, form_data=_GM) == (limpio, "conservado")
    evitados = []

    def _igual(nutrition, form_data, skeleton_day, day_num, memoria=None, evitar=None):
        evitados.append(set(evitar["plantillas"]))
        d = copy.deepcopy(_dias_bloque()[1])
        d["meals"][0]["_template_id"] = "t_otro"          # cambia, pero la cena débil y la yuca siguen
        return d
    monkeypatch.setattr(dd, "build_day_for_skeleton", _igual)
    assert rd.reelegir(_dias_bloque(), 2, nutrition={}, form_data=_GM) == (None, "sin_mejora")
    assert evitados == [{"t_bollitos"}, {"t_bollitos", "t_espaguetis"}], "el 2.º intento libera también la comida principal"
    monkeypatch.setattr(dd, "build_day_for_skeleton", lambda *a, **k: None)
    assert rd.reelegir(_dias_bloque(), 2, nutrition={}, form_data=_GM) == (None, "sin_dia")
    assert rd.reelegir([{"day": 1, "meals": []}], 1, nutrition={}, form_data=_GM) == (None, "no_determinista")


def test_lo_que_el_corrector_deja_igual_vuelve_con_su_procedencia(monkeypatch):
    lib = dict(_template_id="t1", _recipe_source="library", recipe=["Paso congelado."], _scale_factor=1.2)
    orig = {"day": 1, "_day_source": "deterministic", "_day_index": 0, "meals": [dict(TORTILLA, **lib)]}
    corr = {"day": 1, "_critique_applied": True, "meals": [dict(TORTILLA, ingredients=[
        "70 g de harina de maíz precocida", "40 g de queso cheddar", "1 huevo"], recipe=["Otro texto."])]}
    assert rd.restaurar_procedencia(orig, corr) == 1
    assert corr["meals"][0]["_template_id"] == "t1" and corr["meals"][0]["recipe"] == ["Paso congelado."]
    assert corr["_day_source"] == "deterministic", "si vuelven todos, el día vuelve a ser determinista"
    otro = {"day": 1, "meals": [dict(TORTILLA, ingredients=["70 g de harina de maíz", "100 g de pechuga de pollo"])]}
    assert rd.restaurar_procedencia(orig, otro) == 0 and "_template_id" not in otro["meals"][0], "cambió un alimento"
    monkeypatch.setenv("MEALFIT_CRITIQUE_RESTORE_PROVENANCE", "false")
    assert rd.restaurar_procedencia(orig, {"day": 1, "meals": [dict(TORTILLA)]}) == 0


def test_la_autocritica_y_la_regeneracion_reeligen_antes_de_llamar_al_llm():
    sc = inspect.getsource(go.self_critique_node)
    i = sc.index("_reel.reelegir_en_lugar(")
    assert i < sc.index("tasks_by_day: dict = {}") and "barrer=True" in sc[i:i + 700]
    assert "corrected_any = bool(_reelegidos)" in sc
    assert sc.index("_reel.restaurar_procedencia(d, corrected_day)") < sc.index("days[i] = corrected_day")
    sr = inspect.getsource(go.surgical_marker_regen_node)
    assert sr.index("_reel.reelegir_en_lugar(") < sr.index("await asyncio.gather(")
    assert "_re_correct_one(d) for d in _para_llm_sr" in sr and "fixed_count = len(_reelegidos_sr)" in sr
    assert sr.index("_reel.restaurar_procedencia(d, corrected_day)") < sr.index("days[i] = corrected_day")


# ─────────────── 3. parches ───────────────
def _huevo_duro():
    return {"name": "Huevo duro con casabe y sal", "recipe": [
        "Mise en place: Cuece los huevos en agua hirviendo durante diez minutos exactos.",
        "El Toque de Fuego: Pásalos a agua fría para cortar la cocción y pélalos con facilidad.",
        "Montaje: Pártelos por la mitad, sálalos al gusto y acompáñalos con el casabe."]}


def test_el_paso_que_enfria_no_lleva_tiempo_de_fuego(monkeypatch):
    m = _huevo_duro()
    assert go._inject_recipe_time_temp_defaults(m) is False and m["recipe"] == _huevo_duro()["recipe"]
    sofrito = {"name": "Pollo guisado", "recipe": ["Mise en place: pica.", "El Toque de Fuego: Sofríe la cebolla y el pollo.",
                                                   "Montaje: sirve."]}
    assert go._inject_recipe_time_temp_defaults(sofrito) is True and "(~" in sofrito["recipe"][1]
    assert ps.paso_frio("El Toque de Fuego: Cocina el arroz y deja enfriar.") is False, "si también cocina, lleva tiempo"
    monkeypatch.setenv("MEALFIT_TIMETEMP_SKIP_COLD_STEP", "false")
    m = _huevo_duro()
    assert go._inject_recipe_time_temp_defaults(m) is True


def test_el_casabe_se_tuesta_aunque_venga_de_sustituir_un_grano(monkeypatch):
    pasos = ["Mise en place: Enjuaga 30 g de Casabe; separa la coliflor en floretes.",
             "El Toque de Fuego: Cocina Casabe en agua hasta que ablanden e incorpóralo al plato. Dora la coliflor.",
             "Montaje: Sirve Casabe como base."]
    nuevos, n = ps.tecnica_del_sustituto(pasos, "Casabe")
    txt = " ".join(nuevos).lower()
    assert n == 2 and "enjuaga" not in txt and "en agua" not in txt and txt.count("tuesta el casabe") == 1
    assert nuevos[0].startswith("Mise en place: Ten a mano 30 g de casabe") and "Sirve casabe como base" in nuevos[2]
    assert ps.tecnica_del_sustituto(["Cocina la batata en agua."], "Batata") == (["Cocina la batata en agua."], 0)

    class _DB:
        def grams_from_ingredient_string(self, s):
            return 100

    cena = {"meal": "Cena", "name": "Pollo guisado con arroz blanco",
            "ingredients": ["150 g de pechuga de pollo", "100 g de arroz blanco"],
            "recipe": ["Mise en place: Enjuaga el arroz blanco y pica la cebolla.",
                       "El Toque de Fuego: Hierve el arroz blanco en agua con sal unos 18 minutos. Sofríe el pollo.",
                       "Montaje: Sirve el pollo sobre el arroz blanco."]}
    dias = [{"day": 1, "meals": []}, {"day": 2, "meals": []}, {"day": 3, "meals": [cena]}]   # día 3 ⇒ casabe
    assert go._night_rice_autofix(dias, _DB()) == 1
    pasos = " ".join(cena["recipe"]).lower()
    assert "tuesta el casabe" in pasos and "hierve el casabe" not in pasos and "enjuaga" not in pasos
    monkeypatch.setenv("MEALFIT_CARB_SWAP_TECHNIQUE", "false")
    assert ps.tecnica_del_sustituto(pasos, "Casabe") == (pasos, 0)


def test_el_queso_no_se_revuelve():
    out, cambio = go._sanitize_swapped_protein_steps(
        ["Aparte, revuelve queso blanco en una sartén.", "Montaje: rellénalas con el queso gouda y queso blanco revuelto."],
        ["queso blanco"])
    assert cambio and out == ["Aparte, dora el queso blanco en una sartén.",
                              "Montaje: rellénalas con el queso gouda y queso blanco dorado."]
    assert ps.redaccion_queso("Bate el queso cottage con la avena.", "queso cottage") == "Mezcla el queso cottage con la avena."
    assert ps.redaccion_queso("Revuelve los huevos.", "queso blanco") == "Revuelve los huevos.", "sin el queso, nada"
    assert "_ps.redaccion_queso(s, nd_s)" in inspect.getsource(go._sanitize_swapped_protein_steps)


# ─────────────── knobs, docs, marcador, tope ───────────────
def test_knobs_docs_marcador_y_el_god_file_no_subio_el_tope():
    knobs_doc = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    for k in _KNOBS_ARMADOR + ("MEALFIT_CRITIQUE_REPICK_DETERMINISTIC", "MEALFIT_CRITIQUE_RESTORE_PROVENANCE",
                               "MEALFIT_TIMETEMP_SKIP_COLD_STEP", "MEALFIT_CARB_SWAP_TECHNIQUE", "MEALFIT_SWAP_CHEESE_WORDING"):
        assert k in knobs_doc, k
    for doc in ("knobs_reference.md", "plan_pendientes_2026_09_11.md", "deterministic_day.md", "culinary_coherence.md",
                "plan_agente_lotes_38_43_2026_09_14.md"):
        assert "P1-PLAN-LOTE-47" in (_BACKEND / "docs" / doc).read_text(encoding="utf-8"), doc
    for f, ancla in (("reeleccion_dia.py", "P1-PLAN-LOTE-47-REELEGIR"),
                     ("pasos_sustitucion.py", "P1-PLAN-LOTE-47-PASOS-SUSTITUCION"),
                     ("deterministic_day.py", "P1-PLAN-LOTE-47-REGLAS-DE-LA-AUTOCRITICA"),
                     ("graph_orchestrator.py", "P1-PLAN-LOTE-47-REELEGIR-AUTOCRITICA"),
                     ("graph_orchestrator.py", "P1-PLAN-LOTE-47-REELEGIR-REGEN")):
        assert ancla in (_BACKEND / f).read_text(encoding="utf-8"), (f, ancla)
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 47
    n = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8").count("\n")
    assert n <= 52_600, f"graph_orchestrator.py {n} líneas: extraer, no subir el tope"
