# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-48 · 2026-09-14] Cuarta prueba RD del dueño (plan 358a2cdf), la primera tras el lote 47.

11 de 12 comidas llegaron con su receta de biblioteca, pero los cerradores de macros les colgaron cosas encima y el
queso que añaden sale «queso» a secas:

  1. El armador prefiere el plato que ya llega al piso de proteína de su franja (el del cerrador); la base propia se
     escala en vez de colgar otra (mofongo + arroz); en un jugo el lácteo va al lado (cottage licuado con chinola).
  2. El «queso» genérico toma el nombre del queso del plato: la lista compraba queso blanco para platos «con queso cottage».
  3. El cambio del arenque por pescado fresco borraba la frase entera del locrio (se fueron el pescado y el arroz).
  4. La re-elección re-mide la cola al final: lo que otro rearmado ya arregló no va al LLM.
  5. La pulpa de chinola tiene techo propio: 335 g eran 14 chinolas para un jugo.
"""
from __future__ import annotations

import copy
import inspect
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]

import cierres_con_receta as ccr  # noqa: E402
import deterministic_day as dd  # noqa: E402
import graph_orchestrator as go  # noqa: E402
import pasos_sustitucion as ps  # noqa: E402
import reeleccion_dia as rd  # noqa: E402

_GM = {"mainGoal": "gain_muscle", "dietType": "balanced"}


# ─────────────── 1. el armador y el piso de proteína ───────────────
def test_el_piso_de_proteina_es_el_del_cerrador_y_solo_en_ganancia_muscular():
    piso = 25 * float(go.LIGHT_SLOT_PROTEIN_MIN_PCT)
    assert dd._bajo_piso_proteina({"protein": "5g"}, 25, _GM) is True
    assert dd._bajo_piso_proteina({"protein": f"{piso + 0.5:.1f}g"}, 25, _GM) is False
    assert dd._bajo_piso_proteina({"protein": "5g"}, 25, {"mainGoal": "lose_fat"}) is False, "fuera de ganancia no hay remiendo"
    src = inspect.getsource(dd.build_day_for_skeleton)
    assert '0.4 * bool(_piso_prot and _bajo_piso_proteina(_c, obj.get("protein_g"), _fd))' in src
    assert "0.5 * bool(_max_rep and" in src, "el piso pesa menos que la cuota de repetición (con 1,5, sardinas ×4 en 7 días)"


_TPL = {t["template_id"]: t for t in (
    {"template_id": "tpl_platano", "name": "Plátano maduro asado con mantequilla de maní", "protein": "none",
     "slots": ["desayuno"], "constituents": [{"name": "Plátano maduro", "grams": 170}, {"name": "Mantequilla de maní", "grams": 28}]},
    {"template_id": "tpl_mangu", "name": "Mangú con queso frito", "protein": "queso", "slots": ["desayuno"],
     "constituents": [{"name": "Plátano verde", "grams": 200}, {"name": "Queso blanco", "grams": 80}]},
    {"template_id": "tpl_pollo", "name": "Pollo guisado con arroz", "protein": "pollo", "slots": ["almuerzo"],
     "constituents": [{"name": "Pechuga de pollo", "grams": 150}, {"name": "Arroz blanco", "grams": 80}]},
    {"template_id": "tpl_yogur", "name": "Yogur natural con fresas", "protein": "yogur", "slots": ["merienda"],
     "constituents": [{"name": "Yogur natural", "grams": 200}, {"name": "Fresas", "grams": 80}]},
    {"template_id": "tpl_pescado", "name": "Pescado al horno con batata", "protein": "pescado", "slots": ["cena"],
     "constituents": [{"name": "Filete de pescado", "grams": 170}, {"name": "Batata", "grams": 200}]},
)}
_PROT = {"tpl_platano": "5g", "tpl_mangu": "22g", "tpl_pollo": "45g", "tpl_yogur": "20g", "tpl_pescado": "38g"}


@pytest.fixture
def armador(monkeypatch):
    import dish_registry as dr
    import shopping_calculator as sc
    nombres = {c["name"] for t in _TPL.values() for c in t["constituents"]}
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: [{"name": n, "kcal_per_100g": 100} for n in sorted(nombres)])
    monkeypatch.setattr(dr, "templates_by_id", lambda country: dict(_TPL))
    monkeypatch.setattr(dr, "template_candidates", lambda country, slot, family=None, **kw: [
        {"template_id": tid} for tid, t in _TPL.items() if slot in t["slots"]])
    monkeypatch.setattr(dd, "elegir_con_tiempo", lambda tids, obj, cat, por_id, slot, presupuesto=None, **kw: [
        (por_id[t], 1.0) for t in tids if t in por_id])

    def _comida(t, f, cat, slot, country, obj=None):
        ings = [f"{c['grams']:g} g de {c['name']}" for c in t["constituents"]]
        return {"meal": slot.capitalize(), "name": t["name"], "ingredients": ings, "ingredients_raw": list(ings),
                "calories": 500, "protein": _PROT[t["template_id"]], "carbs": "50g", "fats": "15g",
                "_template_id": t["template_id"], "_recipe_source": "library", "recipe": ["Paso."]}
    monkeypatch.setattr(dd, "construir_comida", _comida)
    monkeypatch.setattr(dd, "verifica_comida", lambda meal, fd, cat: [])
    monkeypatch.setattr(dd, "_sodio_de_comida", lambda *a, **k: (100.0, 0))
    monkeypatch.setattr(dd, "_dias_previos_persistidos", lambda fd, uid: [])
    monkeypatch.setattr(dd, "_plantillas_de_planes_recientes", lambda *a, **k: set())
    monkeypatch.setattr(dd, "deterministic_day_for_user", lambda uid: True)
    nut = {"target_calories": "2100 kcal", "macros": {"protein": "123g", "carbs": "271g", "fats": "58g"}}
    esq = {"day": 1, "meal_types": ["Desayuno", "Almuerzo", "Merienda", "Cena"], "protein_pool": []}

    def _armar(goal):
        dia = dd.build_day_for_skeleton(nut, dict(_GM, mainGoal=goal, user_id="u1"), esq, 1, memoria=[])
        assert dia is not None
        return [m["name"] for m in dia["meals"]]
    return _armar


def test_en_ganancia_el_desayuno_que_no_llega_cede_su_turno(armador, monkeypatch):
    """El plátano maduro con mantequilla de maní (5 g de proteína) salía primero y el cerrador le pegaba 185 g de edamame."""
    assert armador("gain_muscle")[0] == "Mangú con queso frito"
    assert armador("lose_fat")[0] == "Plátano maduro asado con mantequilla de maní", "fuera de ganancia, el orden de siempre"
    monkeypatch.setenv("MEALFIT_DETERMINISTIC_DAY_PROTEIN_FLOOR", "false")
    assert armador("gain_muscle")[0] == "Plátano maduro asado con mantequilla de maní"


# ─────────────── 1. la base propia y el jugo ───────────────
class _DB:
    """Macros por gramo de juguete: plátano 1,22 kcal/0,32 g carbos; el resto, proteína."""
    def macros_from_ingredient_string(self, s):
        m = re.match(r"\s*([\d.]+)\s*g\b", str(s))
        if not m:
            return None
        g = float(m.group(1))
        if "platano" in ccr._sa(s):
            return {"name": "Plátano verde", "grams": g, "kcal": 1.22 * g, "protein": 0.013 * g, "carbs": 0.32 * g, "fats": 0.004 * g}
        return {"name": "Pollo", "grams": g, "kcal": 1.65 * g, "protein": 0.31 * g, "carbs": 0.0, "fats": 0.036 * g}

    def grams_from_ingredient_string(self, s):
        mac = self.macros_from_ingredient_string(s)
        return mac["grams"] if mac else None


def _mofongo(**extra):
    m = {"meal": "Almuerzo", "name": "Mofongo de plátano verde al horno con pollo guisado", "_recipe_source": "library",
         "ingredients": ["180 g de plátano verde", "150 g de muslo de pollo"],
         "ingredients_raw": ["180 g de plátano verde", "150 g de muslo de pollo"]}
    m.update(extra)
    return m


def test_la_base_propia_crece_hasta_la_mitad_en_la_lista_y_en_la_compra(monkeypatch):
    m = _mofongo()
    k, c = ccr.escalar_base_propia(m, 150, 400, 80, _DB())
    assert m["ingredients"][0] == "270 g de plátano verde" and m["ingredients_raw"][0] == "270 g de plátano verde"
    assert round(k, 1) == 109.8 and round(c, 1) == 28.8 and m["_base_propia_escalada"] == 1.5
    poco = _mofongo()
    k2, _ = ccr.escalar_base_propia(poco, 150, 22, 80, _DB())
    assert 0 < k2 <= 22.5, "el techo de calorías del día manda"
    assert ccr.escalar_base_propia({"ingredients": ["150 g de muslo de pollo"]}, 150, 400, 80, _DB()) == (0.0, 0.0)
    monkeypatch.setenv("MEALFIT_GAINMUSCLE_FLOOR_OWN_BASE", "false")
    assert ccr.escalar_base_propia(_mofongo(), 150, 400, 80, _DB()) == (0.0, 0.0)


def test_el_piso_de_ganancia_no_le_cuelga_otra_base_a_la_biblioteca(monkeypatch):
    import nutrition_db
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _DB)
    cena = {"meal": "Cena", "name": "Pollo guisado con bollitos de plátano", "_recipe_source": "library",
            "ingredients": ["140 g de plátano verde", "100 g de pechuga de pollo"],
            "ingredients_raw": ["140 g de plátano verde", "100 g de pechuga de pollo"], "cals": 500, "carbs": 60}
    dias = [{"day": 1, "meals": [dict(_mofongo(), cals=600, carbs=70), cena,
                                 {"meal": "Desayuno", "name": "Avena", "ingredients": ["50 g de avena"], "cals": 300, "carbs": 50}]}]
    nut = {"macros": {"protein_g": 123, "carbs_g": 271, "fats_g": 58}}
    go._repair_gainmuscle_day_kcal(dias, nut, {"mainGoal": "gain_muscle"}, db=_DB())
    lineas = " ".join(" ".join(m["ingredients"]) for m in dias[0]["meals"][:2]).lower()
    assert "arroz" not in lineas and "batata" not in lineas, "ni arroz junto al mofongo ni batata junto a los bollitos"
    assert dias[0]["meals"][0]["ingredients"][0] != "180 g de plátano verde", "la base del mofongo creció"
    src = inspect.getsource(go._repair_gainmuscle_day_kcal)
    assert src.index("_ccr.escalar_base_propia(") < src.index("_otro_arroz = any(")


def test_en_un_jugo_el_lacteo_va_al_lado_y_en_una_batida_a_la_licuadora():
    assert ccr.es_jugo({"name": "Jugo de chinola natural sin azúcar con puñado de maní"})
    assert ccr.es_jugo({"name": "Limonada con jengibre"}) and not ccr.es_jugo({"name": "Batida de lechosa con avena"})
    pasos = ["Mete la pulpa en la licuadora con agua fría y licúa unos segundos.", "Sirve el jugo bien frío."]
    jugo = {"meal": "Merienda", "name": "Jugo de chinola natural sin azúcar", "recipe": list(pasos),
            "ingredients": ["80 g de chinola", "30 g de queso cottage"]}
    assert go._append_closer_protein_step(jugo, "queso cottage", True)
    paso = next(p for p in jugo["recipe"] if "💪" in p)
    assert "licuadora" not in paso and "al lado" in paso, paso
    batida = {"meal": "Merienda", "name": "Batida de lechosa", "recipe": list(pasos), "ingredients": ["30 g de queso cottage"]}
    assert go._append_closer_protein_step(batida, "queso cottage", True)
    assert any("a la licuadora" in p for p in batida["recipe"] if "💪" in p), "la batida conserva su licuadora"
    atun = {"meal": "Merienda", "name": "Jugo de chinola natural", "recipe": list(pasos), "ingredients": ["80 g de chinola"]}
    assert go._append_closer_protein_step(atun, "atún en lata", True)
    paso = next(p for p in atun["recipe"] if "💪" in p)
    assert paso[paso.index("💪"):].startswith("💪 Sirve atún en lata al lado"), "en un jugo nada se incorpora al vaso"


# ─────────────── 2. el queso con su nombre ───────────────
def test_el_queso_generico_toma_el_nombre_del_plato_en_la_lista_y_en_la_compra(monkeypatch):
    dias = [{"meals": [
        {"name": "Batida de lechosa con avena y queso cottage", "ingredients": ["50 g de lechosa", "30 g de queso"],
         "ingredients_raw": ["50 g de Lechosa", "30 g de queso"], "_display": {"x": 1}},
        {"name": "Tortilla de harina de maíz con queso cheddar", "ingredients": ["25 g de queso", "1 huevo"],
         "ingredients_raw": ["25 g de queso", "1 huevo"]},
        {"name": "Mangú con queso frito", "ingredients": ["60 g de queso"]},
        {"name": "Pasta con queso gouda y mozzarella", "ingredients": ["40 g de queso rallado"]},
        {"name": "Queso de hoja con vainitas", "ingredients": ["60 g de queso de hoja"]},
        {"name": "Guiso", "ingredients": ["40 g de queso"], "recipe": ["Mise en place: mide 40 g de queso gouda.",
                                                                     "Añade el queso cheddar rallado al final."]}]}]
    assert ccr.nombrar_quesos_genericos(dias) == 3
    b, t, mangu, pasta, hoja, guiso = dias[0]["meals"]
    assert b["ingredients"][1] == "30 g de queso cottage" and b["ingredients_raw"][1] == "30 g de queso cottage"
    assert "_display" not in b, "la capa de traducción espeja la lista: se regenera"
    assert t["ingredients"][0] == "25 g de queso cheddar"
    assert mangu["ingredients"] == ["60 g de queso"] and pasta["ingredients"] == ["40 g de queso rallado"], "ninguno o dos: no se adivina"
    assert hoja["ingredients"] == ["60 g de queso de hoja"]
    assert guiso["ingredients"] == ["40 g de queso cheddar"], "sin queso en el nombre, el de los pasos (no el de la Mise)"
    assert ccr.nombrar_quesos_genericos(dias) == 0, "idempotente"
    monkeypatch.setenv("MEALFIT_GENERIC_CHEESE_FROM_NAME", "false")
    assert ccr.nombrar_quesos_genericos([{"meals": [{"name": "Batida con queso cottage", "ingredients": ["30 g de queso"]}]}]) == 0


def test_el_queso_se_nombra_antes_del_lacteo_del_nombre_y_de_los_dos_barridos():
    asm = inspect.getsource(go.assemble_plan_node)
    assert asm.index("_ccr.nombrar_quesos_genericos(result.get(\"days\") or [])") < asm.index("_repair_name_phantom_dairy(")
    i_bar = asm.index("_barrer_lineas_muertas_de_raw(days)")
    assert asm.rfind("_ccr.nombrar_quesos_genericos(days)", 0, i_bar) != -1
    fin = inspect.getsource(go.finalize_plan_data_coherence)
    assert fin.index("_ccr.nombrar_quesos_genericos(days)") < fin.index("_barrer_lineas_muertas_de_raw(days)")


# ─────────────── 3. el desalado es una cláusula ───────────────
def test_el_cambio_del_arenque_no_se_lleva_el_arroz_del_locrio(monkeypatch):
    frase = ("Agrega el filete de pescado blanco (ya desalado y en trozos), remuévelo un momento con el sofrito, y añade "
             "el arroz blanco (se pesa en crudo). Remueve para que se impregne bien.")
    m = {"name": "Locrio de Pescado blanco", "recipe": ["Sofríe la cebolla.", frase,
                                                        "Mise en place: Desala el arenque remojándolo 2 horas. Escurre."]}
    assert go._strip_desalt_instructions(m) == 2
    assert m["recipe"][1].startswith("Agrega el filete de pescado blanco (en trozos)") and "arroz blanco" in m["recipe"][1]
    assert "desala" not in " ".join(m["recipe"]).lower() and m["recipe"][2] == "Mise en place: Escurre."
    assert ps.quitar_clausula_desalado("Incorpora el bacalao desalado y la papa.") == "Incorpora el bacalao y la papa."
    monkeypatch.setenv("MEALFIT_DESALT_CLAUSE_ONLY", "false")
    m2 = {"recipe": [frase]}
    go._strip_desalt_instructions(m2)
    assert "arroz" not in " ".join(m2["recipe"]), "con el knob apagado, la frase entera se va (conducta previa)"


# ─────────────── 4. la cola se re-mide ───────────────
def _det(n, *meals):
    return {"day": n, "_day_source": "deterministic", "meals": [
        {"meal": s, "name": nm, "ingredients": list(ings), "_template_id": tid, "_recipe_source": "library"}
        for s, nm, ings, tid in meals]}


def test_lo_que_otro_rearmado_arreglo_no_va_al_llm(monkeypatch):
    """Plan 358a2cdf: el día 3 nombrado no mejoraba solo; el barrido rehízo el día 1 sin yuca y el 3 igual fue al LLM."""
    d1 = _det(1, ("Almuerzo", "Pinchos de pollo con yuca hervida", ["150 g de pechuga de pollo", "200 g de yuca"], "t_pinchos"),
              ("Cena", "Pescado al horno con batata", ["150 g de filete de pescado", "200 g de batata"], "t_pescado"))
    d3 = _det(3, ("Almuerzo", "Espaguetis con sardinas", ["80 g de pasta", "100 g de sardinas en lata"], "t_esp"),
              ("Cena", "Tilapia al horno con yuca y cebolla", ["150 g de tilapia", "200 g de yuca"], "t_tilapia"))
    nuevo3 = _det(3, ("Almuerzo", "Espaguetis con sardinas", ["80 g de pasta", "100 g de sardinas en lata"], "t_esp"),
                  ("Cena", "Res guisada con yuca", ["150 g de carne de res", "200 g de yuca"], "t_res"))
    nuevo1 = _det(1, ("Almuerzo", "Pollo a la plancha con arroz", ["150 g de pechuga de pollo", "80 g de arroz blanco"], "t_pa"),
                  ("Cena", "Pescado al horno con batata", ["150 g de filete de pescado", "200 g de batata"], "t_pescado"))
    monkeypatch.setattr(dd, "build_day_for_skeleton",
                        lambda nutrition, fd, sk, n, memoria=None, evitar=None: copy.deepcopy({1: nuevo1, 3: nuevo3}[n]))
    dias = [d1, d3]
    llm, rearmados, conservados = rd.reelegir_en_lugar(dias, [3], nutrition={}, form_data=_GM, etiqueta="t", barrer=True)
    assert (llm, rearmados, conservados) == ([], [1], [3])
    assert dias[1] is d3, "el día 3 se conserva con su receta de biblioteca"
    assert "P1-PLAN-LOTE-48-REMEDIR-COLA" in Path(rd.__file__).read_text(encoding="utf-8")


# ─────────────── 5. la pulpa ───────────────
def test_la_pulpa_de_chinola_tiene_techo_propio():
    dias = [{"day": 1, "meals": [{"meal": "Merienda", "name": "Jugo de chinola", "ingredients": ["335 g de chinola", "15 g de maní"],
                                  "ingredients_raw": ["335 g de Chinola", "15 g de Maní"]},
                                 {"meal": "Merienda", "name": "Sandía fresca", "ingredients": ["300 g de sandía"]}]}]
    assert go._cap_unrealistic_portions(dias) >= 1
    assert dias[0]["meals"][0]["ingredients"][0] == f"{go.REALISM_PULP_CAP_G} g de chinola"
    assert dias[0]["meals"][0]["ingredients_raw"][0] == f"{go.REALISM_PULP_CAP_G} g de Chinola"
    assert dias[0]["meals"][1]["ingredients"] == ["300 g de sandía"], "la fruta de volumen conserva su techo"


# ─────────────── knobs, docs, marcador, tope ───────────────
def test_knobs_docs_marcador_y_el_god_file_no_subio_el_tope():
    knobs_doc = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    for k in ("MEALFIT_DETERMINISTIC_DAY_PROTEIN_FLOOR", "MEALFIT_GAINMUSCLE_FLOOR_OWN_BASE", "MEALFIT_CLOSER_JUICE_DAIRY_ASIDE",
              "MEALFIT_GENERIC_CHEESE_FROM_NAME", "MEALFIT_DESALT_CLAUSE_ONLY", "MEALFIT_REALISM_PULP_CAP_G"):
        assert k in knobs_doc, k
    for doc in ("knobs_reference.md", "plan_pendientes_2026_09_11.md", "deterministic_day.md", "culinary_coherence.md",
                "plan_agente_lotes_38_43_2026_09_14.md"):
        assert "P1-PLAN-LOTE-48" in (_BACKEND / "docs" / doc).read_text(encoding="utf-8"), doc
    for f, ancla in (("cierres_con_receta.py", "P1-PLAN-LOTE-48-CIERRES-CON-RECETA"),
                     ("deterministic_day.py", "P1-PLAN-LOTE-48-PISO-PROTEINA"),
                     ("graph_orchestrator.py", "P1-PLAN-LOTE-48-BASE-PROPIA"),
                     ("graph_orchestrator.py", "P1-PLAN-LOTE-48-QUESO-NOMBRADO"),
                     ("graph_orchestrator.py", "P1-PLAN-LOTE-48-PULPA")):
        assert ancla in (_BACKEND / f).read_text(encoding="utf-8"), (f, ancla)
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 48
    n = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8").count("\n")
    assert n <= 52_600, f"graph_orchestrator.py {n} líneas: extraer, no subir el tope"
