# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-49 · 2026-09-14] Quinta prueba RD del dueño (plan a059d7bb), tras el lote 48.

La mejor de la serie (144 s, 3/3 días deterministas, 12/12 recetas de biblioteca, juez 0, aprobado al primer intento), con
defectos a la vista:

  1. El ingrediente que da nombre, en migajas: guacamole con 5 g de aguacate, «maní tostado» con 5 g de maní y el casabe
     con 2,7 g de mantequilla de maní — y el día 1 al 71 % de su grasa. En la cola del guardado sube al piso si cabe.
  2. El cerrador: huevo al desayuno con huevo en la cena (dos días), 10 g de arenque en una merienda, y su paso metido en
     el primer paso con fuego («Añade camarones al guiso…» dentro del que hierve la yuca).
  3. El tope de huevos: la lista decía «3 claras de huevo» y la compra seguía con 165 g de huevo entero.
  4. «Nada» de tiempo: una cena de 65 min; el tiempo pasa a ser falta del armador.
  5. Los planes recientes contaban la fila vacía del plan en curso: 6 de 12 platos repetían el tercer plan.
  6. Texto: «ya escurridas» y «el líquido de la lata» tras cambiar las sardinas; «brevemente (~10-12 min)».
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND = Path(__file__).resolve().parents[1]

import cierres_con_receta as ccr  # noqa: E402
import culinary_coherence as cc  # noqa: E402
import deterministic_day as dd  # noqa: E402
import graph_orchestrator as go  # noqa: E402
import identidad_plato as idp  # noqa: E402
import pasos_sustitucion as ps  # noqa: E402

GUACAMOLE = "tpl_e07ba75823ee"      # Guacamole criollo con casabe y huevo duro (aguacate 100 g, el más pesado)
_CAT = [{"name": n, "aliases": a, "category": c, "prep_methods": ["ninguno"]} for n, a, c in (
    ("Aguacate", [], "Frutas"), ("Maní", ["mani"], "Frutos secos"), ("Tomate", ["tomates"], "Vegetales"),
    ("Cebolla", [], "Vegetales"), ("Limón", ["limon"], "Frutas"), ("Casabe", [], "Granos"),
    ("Huevo", ["huevos"], "Proteínas"))]
_IDX = cc.build_culinary_index(_CAT)


class _DB:
    """Macros por gramo de juguete (kcal, grasa); lo demás, 0,1 g por gramo."""
    _POR_G = (("mantequilla de mani", (5.88, 0.50)), ("aguacate", (1.60, 0.15)), ("mani", (5.67, 0.49)),
              ("queso cottage", (0.98, 0.04)), ("yogurt griego", (0.97, 0.05)), ("huevo", (1.55, 0.11)))

    def macros_from_ingredient_string(self, s):
        m = re.match(r"\s*([\d.]+)\s*g\s+de\s+(.+)$", str(s).strip(), re.IGNORECASE)
        if not m:
            return None
        g, nombre = float(m.group(1)), ccr._sa(m.group(2)).strip()
        kpg, fpg = next((v for k, v in self._POR_G if k in nombre), (1.0, 0.0))
        return {"name": nombre, "grams": g, "kcal": kpg * g, "protein": 0.1 * g, "carbs": 0.1 * g, "fats": fpg * g}

    def grams_from_ingredient_string(self, s):
        mac = self.macros_from_ingredient_string(s)
        return mac["grams"] if mac else None

    def __getattr__(self, _n):
        return lambda *a, **k: None


def _guac(**extra):
    m = {"meal": "Cena", "name": "Guacamole criollo con casabe y huevo duro", "_recipe_source": "library",
         "_template_id": GUACAMOLE, "_scale_factor": 1.525, "cals": 524, "fats": "16g",
         "ingredients": ["1 tomate", "½ cebolla", "2 tortas pequeño de casabe", "3 huevos", "5 g de aguacate"],
         "ingredients_raw": ["160 g de Tomate", "0.5 cebolla", "2 tortas pequeño de casabe", "3 huevos", "5 g de aguacate"]}
    m.update(extra)
    return m


# ─────────────── 1. la identidad, hasta el final ───────────────
def test_lo_presente_pero_pobre_sube_al_piso_si_al_dia_le_cabe():
    m = _guac()
    margen = {"kcal": 300.0, "grasa": 17.0}
    assert idp.restaurar_meal(m, _IDX, db=_DB(), margen=margen) == ["↑5→38 g de Aguacate"]   # 0,25 × 100 × 1,525
    assert "38 g de Aguacate" in m["ingredients"] and "5 g de aguacate" not in m["ingredients"]
    assert "38 g de Aguacate" in m["ingredients_raw"] and "5 g de aguacate" not in m["ingredients_raw"]
    assert margen["grasa"] == pytest.approx(17.0 - 33 * 0.15)
    assert idp.restaurar_meal(m, _IDX, db=_DB(), margen=margen) == [], "idempotente: ya está en su piso"
    corto = _guac()
    assert idp.restaurar_meal(corto, _IDX, db=_DB(), margen={"kcal": 300.0, "grasa": 2.0}) == [], "sin sitio, no sube"
    assert "5 g de aguacate" in corto["ingredients"]
    assert idp.restaurar_meal(_guac(), _IDX, db=_DB()) == [], "sin margen del día, lo del lote 46: sólo lo que falta"


def test_nunca_baja_lo_que_ya_pasa_del_piso_aunque_la_lista_lea_mal_los_decimales():
    """Réplica sobre el plan a059d7bb: la primera versión medía con el lector de la lista del contrato, que leyó «57.2 g de
    Soya texturizada» como 2 g y «subió» la soya… a 18 g. Se mide con el lector de la base, y nunca se baja."""
    cat = _CAT + [{"name": "Soya texturizada", "aliases": ["soya"], "category": "Proteínas", "prep_methods": ["ninguno"]},
                  {"name": "Berenjena", "aliases": [], "category": "Vegetales", "prep_methods": ["ninguno"]},
                  {"name": "Batata", "aliases": [], "category": "Granos", "prep_methods": ["ninguno"]}]
    m = {"meal": "Cena", "name": "Berenjena guisada con soya texturizada y batata", "_recipe_source": "library",
         "_template_id": "tpl_dbae3c086fa2", "_scale_factor": 1.326,
         "ingredients": ["125 g de Berenjena", "55 g de Soya texturizada", "1 batata mediana"],
         "ingredients_raw": ["125 g de Berenjena", "57.2 g de Soya texturizada", "230 g de Batata"]}
    antes = (list(m["ingredients"]), list(m["ingredients_raw"]))
    assert idp.restaurar_meal(m, cc.build_culinary_index(cat), db=_DB(), margen={"kcal": 500.0, "grasa": 30.0}) == []
    assert (m["ingredients"], m["ingredients_raw"]) == antes


def test_el_margen_es_el_del_dia_contra_los_objetivos_del_plan(monkeypatch):
    assert idp.objetivos_de({"calories": 2100, "macros": {"fats": "58g"}}) == {"kcal": 2100.0, "grasa": 58.0}
    assert idp.objetivos_de({"calories": 2100, "macros": {}}) is None
    obj = {"kcal": 2100.0, "grasa": 58.0}
    otro = {"meal": "Almuerzo", "name": "Pollo", "cals": 1300, "fats": "25g", "ingredients": ["200 g de pollo"]}
    dias = [{"day": 1, "meals": [_guac(), dict(otro)]}]                     # grasa del día 41 de 58 × 1,05
    assert idp.restaurar_identidad(dias, db=_DB(), index=_IDX, objetivos=obj) == 1
    assert "38 g de Aguacate" in dias[0]["meals"][0]["ingredients"]
    llenos = [{"day": 1, "meals": [_guac(), dict(otro, fats="50g")]}]      # 66 g: ya sobre el techo
    assert idp.restaurar_identidad(llenos, db=_DB(), index=_IDX, objetivos=obj) == 0
    monkeypatch.setenv("MEALFIT_DISH_IDENTITY_RAISE", "false")
    assert idp.restaurar_identidad([{"day": 1, "meals": [_guac(), dict(otro)]}], db=_DB(), index=_IDX, objetivos=obj) == 0


def test_la_cola_del_guardado_pasa_los_objetivos():
    dbp = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    i = dbp.index("_idp_tail.restaurar_identidad(")
    assert "objetivos=_idp_tail.objetivos_de(_pd)" in dbp[i:i + 400]
    assert i > dbp.index("from protein_floor_last_word import reencuadra_y_mide"), "después de todos los recortes"
    assert i < dbp.index("_rfc_tail_out = _rfc_tail("), "antes del contrato final, que así sincroniza los pasos"


# ─────────────── 2. el cerrador lee el día y la receta ───────────────
def _info(name, protein, kcal, carbs=1.0, fats=3.0):
    return SimpleNamespace(name=name, protein=protein, kcal=kcal, carbs=carbs, fats=fats)


_HUEVO = _info("Huevo", 13.0, 155.0, 1.0, 11.0)
_LACTEOS = [(0.11, "Queso cottage", _info("Queso cottage", 11.0, 98.0)),
            (0.10, "Yogurt griego", _info("Yogurt griego", 10.0, 97.0))]


@pytest.fixture
def cerrador(monkeypatch):
    monkeypatch.setattr(go, "PROTEIN_CLOSER_SCALE_FIRST", False)
    monkeypatch.setattr(go, "_scale_congruent_protein_line", lambda *a, **k: False)
    monkeypatch.setattr(go, "_safe_high_density_proteins", lambda *a, **k: list(_LACTEOS))
    return go


def _desayuno():
    return {"meal": "Desayuno", "name": "Casabe con mantequilla de maní y pasas", "protein": 5, "carbs": 60, "fats": 8,
            "cals": 330, "ingredients": ["½ torta de casabe", "25 g de mantequilla de maní", "25 g de pasas"],
            "ingredients_raw": ["½ torta de casabe", "25 g de mantequilla de maní", "25 g de pasas"],
            "recipe": ["Mise en place: mide todo.", "Unta el casabe con la mantequilla de maní.", "Montaje: sirve."]}


def test_en_franja_ligera_un_lacteo_que_el_dia_no_tenga_antes_que_repetir_el_huevo(cerrador, monkeypatch):
    m = _desayuno()
    assert go._close_protein_gap_for_meal(m, 25.0, _DB(), [(1.0, "Huevo", _HUEVO)], day_used_proteins={"huevo"},
                                          enforce_min_threshold=False) > 0
    blob = " ".join(m["ingredients"]).lower()
    assert "cottage" in blob and "huevo" not in blob, m["ingredients"]
    monkeypatch.setenv("MEALFIT_CLOSER_LIGHT_SLOT_CLEAN", "false")
    m2 = _desayuno()
    go._close_protein_gap_for_meal(m2, 25.0, _DB(), [(1.0, "Huevo", _HUEVO)], day_used_proteins={"huevo"},
                                   enforce_min_threshold=False)
    assert "huevo" in " ".join(m2["ingredients"]).lower(), "con el knob apagado, el piso gana con la proteína repetida"


def test_sin_base_de_datos_o_en_comida_fuerte_la_conducta_de_siempre(cerrador):
    m = _desayuno()
    go._close_protein_gap_for_meal(m, 25.0, None, [(1.0, "Huevo", _HUEVO)], day_used_proteins={"huevo"},
                                   enforce_min_threshold=False)
    assert "huevo" in " ".join(m["ingredients"]).lower(), "sin db no se busca otro lácteo (test del lote de julio)"
    almuerzo = dict(_desayuno(), meal="Almuerzo", name="Arroz blanco con vegetales")
    go._close_protein_gap_for_meal(almuerzo, 40.0, _DB(), [(1.0, "Huevo", _HUEVO)], day_used_proteins={"huevo"},
                                   enforce_min_threshold=False)
    assert "cottage" not in " ".join(almuerzo["ingredients"]).lower(), "sólo en desayuno y merienda"


def test_el_cerrador_no_anade_un_curado(cerrador, monkeypatch):
    cands = [(0.30, "Bacalao", _info("Bacalao", 29.0, 105.0)), (0.25, "Pechuga de pollo", _info("Pechuga de pollo", 31.0, 165.0))]
    m = dict(_desayuno(), meal="Almuerzo", name="Arroz blanco con vegetales")
    go._close_protein_gap_for_meal(m, 40.0, _DB(), list(cands), enforce_min_threshold=False)
    blob = " ".join(m["ingredients"]).lower()
    assert "pollo" in blob and "bacalao" not in blob, m["ingredients"]
    monkeypatch.setenv("MEALFIT_CLOSER_NO_SALTCURED", "false")
    m2 = dict(_desayuno(), meal="Almuerzo", name="Arroz blanco con vegetales")
    go._close_protein_gap_for_meal(m2, 40.0, _DB(), list(cands), enforce_min_threshold=False)
    assert "bacalao" in " ".join(m2["ingredients"]).lower()


def _salami(**extra):
    m = {"meal": "Cena", "name": "Salami guisado con yuca, cebolla y camarones", "_recipe_source": "library",
         "recipe": ["Mise en place: pela y mide todo.",
                    "El Toque de Fuego: Pela la yuca, córtala en trozos y hiérvela en agua con sal unos 20-25 minutos.",
                    "Corta el salami en rodajas y dóralas en el aceite a fuego medio-alto por ambas caras.",
                    "Baja el fuego, añade la cebolla en aros y el ajo, y cocina hasta que la cebolla se ablande.",
                    "Sirve el salami encebollado sobre la yuca escurrida.",
                    "💪 Añade camarones al guiso en los últimos minutos de cocción para que tome el sabor.",
                    "Montaje: Emplata «Salami guisado con yuca, cebolla y camarones» y sirve."]}
    m.update(extra)
    return m


def test_el_paso_del_cerrador_va_despues_del_ultimo_paso_con_fuego_y_el_al_lado_al_emplatado():
    m = _salami()
    go._integrate_complement_steps([{"meals": [m]}])
    r = m["recipe"]
    assert r[1].endswith("20-25 minutos."), "el paso que hierve la yuca queda como estaba"
    assert r[4] == "Añade camarones al guiso en los últimos minutos de cocción para que tome el sabor.", r
    assert not any("💪" in s for s in r)
    avena = {"meal": "Desayuno", "name": "Avena cocida con leche evaporada y queso cottage", "_recipe_source": "library",
             "recipe": ["El Toque de Fuego: Calienta la leche evaporada a fuego medio.", "Añade la avena y cocina 5 minutos.",
                        "💪 Sirve queso cottage al lado para acompañar.", "Montaje: Emplata y sirve."]}
    go._integrate_complement_steps([{"meals": [avena]}])
    assert avena["recipe"][0] == "El Toque de Fuego: Calienta la leche evaporada a fuego medio."
    assert avena["recipe"][-1] == "Montaje: Emplata y sirve. Sirve queso cottage al lado para acompañar."


def test_fuera_de_la_biblioteca_o_con_el_knob_apagado_la_fusion_de_siempre(monkeypatch):
    m = _salami(_recipe_source="llm")
    go._integrate_complement_steps([{"meals": [m]}])
    assert "Añade camarones al guiso" in m["recipe"][1], "receta del LLM: se funde en el Toque de Fuego (conducta previa)"
    monkeypatch.setenv("MEALFIT_CLOSER_STEP_PLACEMENT", "false")
    m2 = _salami()
    go._integrate_complement_steps([{"meals": [m2]}])
    assert "Añade camarones al guiso" in m2["recipe"][1]


# ─────────────── 3. el tope de huevos también en la compra ───────────────
def _dia_huevos():
    return [{"day": 1, "meals": [
        {"meal": "Desayuno", "name": "Casabe con mantequilla de maní, pasas y huevo", "ingredients": ["½ torta de casabe", "3 huevos"],
         "ingredients_raw": ["½ torta de casabe", "165 g de huevo cocido", "Sal al gusto"]},
        {"meal": "Cena", "name": "Guacamole criollo con casabe y huevo duro", "ingredients": ["4 huevos", "5 g de aguacate"],
         "ingredients_raw": ["4 huevos", "5 g de aguacate"]}]}]


def test_el_tope_de_huevos_reescribe_la_compra_aunque_no_se_emparejen_por_alimento(monkeypatch):
    monkeypatch.setattr(go, "_raw_idx_for_display", lambda *a, **k: None)   # el emparejamiento que falló en a059d7bb
    dias = _dia_huevos()
    assert go._cap_daily_whole_eggs(dias, db=_DB(), max_whole=4) >= 1
    des = dias[0]["meals"][0]
    assert des["ingredients"][1] == "3 claras de huevo"
    assert "165 g de huevo cocido" not in des["ingredients_raw"] and "3 claras de huevo" in des["ingredients_raw"]
    assert ccr.indice_huevo_entero_raw(["1 huevo", "2 claras de huevo"]) == 0
    assert ccr.indice_huevo_entero_raw(["1 huevo", "2 huevos cocidos"]) is None, "dos candidatas: no se adivina"
    monkeypatch.setenv("MEALFIT_EGGCAP_RAW_WHOLE_EGG", "false")
    dias = _dia_huevos()
    go._cap_daily_whole_eggs(dias, db=_DB(), max_whole=4)
    assert "165 g de huevo cocido" in dias[0]["meals"][0]["ingredients_raw"], "sin el respaldo: el defecto del plan"


# ─────────────── 4. «Nada» de tiempo, como falta del armador ───────────────
def _tpl(tid, nombre, slot, minutos, prot="none"):
    return {"template_id": tid, "name": nombre, "protein": prot, "slots": [slot],
            "logistics": {"prep_minutes_source": "receta", "prep_minutes_est": minutos},
            "constituents": [{"name": nombre.split()[0], "grams": 100}]}


def test_la_falta_de_tiempo_crece_con_el_exceso():
    t = lambda m: {"logistics": {"prep_minutes_source": "receta", "prep_minutes_est": m}}  # noqa: E731
    assert [dd._falta_tiempo(t(m), 10) for m in (10, 20, 30, 45, 65)] == [0.0, 0.0, 0.35, 1.0, 1.5]
    assert dd._falta_tiempo(t(65), None) == 0.0 and dd._falta_tiempo({"logistics": {}}, 10) == 0.0


_TPLS = {t["template_id"]: t for t in (
    _tpl("tpl_des", "Avena con leche", "desayuno", 5), _tpl("tpl_alm", "Arroz con pollo", "almuerzo", 15, "pollo"),
    _tpl("tpl_mer", "Yogur con fresas", "merienda", 5, "yogur"),
    _tpl("tpl_lenta", "Berenjena guisada con soya", "cena", 65, "legumbre"),
    _tpl("tpl_rapida", "Pescado a la plancha con ensalada", "cena", 30, "pescado"))}


@pytest.fixture
def armador(monkeypatch):
    import dish_registry as dr
    import shopping_calculator as sc
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: [{"name": n, "kcal_per_100g": 100}
                                                               for n in sorted({t["name"].split()[0] for t in _TPLS.values()})])
    monkeypatch.setattr(dr, "templates_by_id", lambda country: dict(_TPLS))
    monkeypatch.setattr(dr, "template_candidates", lambda country, slot, family=None, **kw: [
        {"template_id": tid} for tid, t in _TPLS.items() if slot in t["slots"]])
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
    nut = {"target_calories": "2000 kcal", "macros": {"protein": "120g", "carbs": "250g", "fats": "60g"}}
    esq = {"day": 1, "meal_types": ["Desayuno", "Almuerzo", "Merienda", "Cena"], "protein_pool": []}

    def _armar():
        dia = dd.build_day_for_skeleton(nut, {"mainGoal": "lose_fat", "cookingTime": "none", "user_id": "u1"}, esq, 1,
                                         memoria=[])
        assert dia is not None
        return [m["name"] for m in dia["meals"]]
    return _armar


def test_la_falta_de_tiempo_queda_apagada_y_encendida_cede_la_cena_de_65_minutos(armador, monkeypatch):
    """Medida en réplica: ningún peso mejora el tiempo sin romper otra regla (variedad o autocrítica). Queda apagada hasta
    que la biblioteca tenga almuerzos y cenas rápidos; encendida, la cena de 65 min cede ante una de 30."""
    assert armador()[-1] == "Berenjena guisada con soya", "apagada por defecto: el orden de siempre"
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY_TIME_FAULT", "true")
    assert armador()[-1] == "Pescado a la plancha con ensalada"


# ─────────────── 5. los planes recientes sin la fila del plan en curso ───────────────
def test_los_planes_recientes_son_los_que_tienen_dias():
    assert "_archived_days" in dd._SQL_CON_DIAS and "jsonb_array_length(plan_data->'days')" in dd._SQL_CON_DIAS
    assert '"SELECT plan_data FROM meal_plans WHERE user_id = %s AND " + _SQL_CON_DIAS' in inspect.getsource(
        dd._plantillas_de_planes_recientes)
    dbp = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    i = dbp.index("def get_recent_meals_from_plans")
    assert "jsonb_array_length(plan_data->'days')" in dbp[i:i + 1500], "el revisor compara con los mismos planes"


# ─────────────── 6. texto ───────────────
def _casabe():
    return {"name": "Casabe con mantequilla de maní y canela", "recipe": [
        "El Toque de Fuego: Tuesta el casabe brevemente para que quede crujiente.", "Úntalo con la mantequilla de maní."]}


def test_lo_breve_recibe_un_tiempo_breve_y_no_el_de_la_tecnica(monkeypatch):
    """La primera versión no ponía tiempo y el contrato de la receta lo acusó en 2 de las 193 recetas de biblioteca («Toque
    de Fuego sin tiempo/temperatura concreta»): el contrato pide un tiempo; «brevemente» pide que sea corto."""
    m = _casabe()
    assert go._inject_recipe_time_temp_defaults(m) is True
    assert m["recipe"][0].endswith(f"(~{ps.TIEMPO_BREVE}).") and "10-12" not in m["recipe"][0]
    assert go._inject_recipe_time_temp_defaults(m) is False, "idempotente: ya trae tiempo"
    monkeypatch.setenv("MEALFIT_TIMETEMP_SKIP_BRIEF_STEP", "false")
    m2 = _casabe()
    assert go._inject_recipe_time_temp_defaults(m2) is True and ps.TIEMPO_BREVE not in m2["recipe"][0]


def test_tras_cambiar_la_lata_por_pescado_fresco_no_quedan_ni_escurridas_ni_el_liquido(monkeypatch):
    p3 = ("Incorpora filete de pescado blanco, ya escurridas, deshechas en trozos grandes, remueve con cuidado y agrega "
          "el arroz, que se pesa en crudo.")
    p4 = ("Mientras el locrio reposa, mezcla en un bol el repollo y la zanahoria rallada, con un chorrito de agua y un "
          "pellizco del líquido de la lata de filete de pescado blanco si quieres darle sabor, y revuelve bien.")
    assert ps.quitar_clausula_enlatado(p3) == ("Incorpora filete de pescado blanco en trozos grandes, remueve con cuidado "
                                               "y agrega el arroz, que se pesa en crudo.")
    assert ps.quitar_clausula_enlatado(p4) == ("Mientras el locrio reposa, mezcla en un bol el repollo y la zanahoria "
                                               "rallada, con un chorrito de agua, y revuelve bien.")
    m = {"recipe": [p3, p4, "Montaje: sirve."]}
    assert ps.limpiar_pasos_enlatado(m) == 2
    src = inspect.getsource(go._day_sodium_autofix) if hasattr(go, "_day_sodium_autofix") else \
        (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index('"swap_saltcured" if _is_saltcured_swap else "swap_canned"')
    assert "_ps.limpiar_pasos_enlatado(_bm)" in src[i:i + 900], "sólo en el cambio de un enlatado"
    monkeypatch.setenv("MEALFIT_CANNED_SWAP_CLAUSES", "false")
    assert ps.quitar_clausula_enlatado(p3) == p3


# ─────────────── knobs, docs, marcador, tope ───────────────
def test_knobs_docs_marcador_y_el_god_file_no_subio_el_tope():
    knobs_doc = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    for k in ("MEALFIT_DISH_IDENTITY_RAISE", "MEALFIT_CLOSER_LIGHT_SLOT_CLEAN", "MEALFIT_CLOSER_NO_SALTCURED",
              "MEALFIT_CLOSER_STEP_PLACEMENT", "MEALFIT_EGGCAP_RAW_WHOLE_EGG", "MEALFIT_DETERMINISTIC_DAY_TIME_FAULT",
              "MEALFIT_TIMETEMP_SKIP_BRIEF_STEP", "MEALFIT_CANNED_SWAP_CLAUSES"):
        assert k in knobs_doc, k
    for doc in ("knobs_reference.md", "plan_pendientes_2026_09_11.md", "deterministic_day.md", "culinary_coherence.md",
                "plan_agente_lotes_38_43_2026_09_14.md"):
        assert "P1-PLAN-LOTE-49" in (_BACKEND / "docs" / doc).read_text(encoding="utf-8"), doc
    for f, ancla in (("identidad_plato.py", "P1-PLAN-LOTE-49-IDENTIDAD-HASTA-EL-FINAL"),
                     ("cierres_con_receta.py", "P1-PLAN-LOTE-49-CERRADOR-LEE"),
                     ("cierres_con_receta.py", "P1-PLAN-LOTE-49-PASO-EN-SU-SITIO"),
                     ("graph_orchestrator.py", "P1-PLAN-LOTE-49-FRANJA-LIGERA"),
                     ("deterministic_day.py", "P1-PLAN-LOTE-49-FALTA-TIEMPO"),
                     ("deterministic_day.py", "P1-PLAN-LOTE-49-PLANES-CON-DIAS"),
                     ("db_plans.py", "P1-PLAN-LOTE-49-PLANES-CON-DIAS")):
        assert ancla in (_BACKEND / f).read_text(encoding="utf-8"), (f, ancla)
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 49
    n = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8").count("\n")
    assert n <= 52_600, f"graph_orchestrator.py {n} líneas: extraer, no subir el tope"
