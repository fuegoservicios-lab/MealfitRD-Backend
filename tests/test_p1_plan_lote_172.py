# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-172 · 2026-09-23] La batería REAL del generador (camino del modelo, perfil RD «Nada» de tiempo, ganar
músculo, 30 días) entregó macros en banda en el pipeline… y el escudo pre-INSERT los rompió. Tres defectos, medidos
en el plan de la batería reproduciendo el escudo pase a pase:

1. **La taza de 200 g para todo** (`humanize_ingredients`): «115 g de avena» salía «½ taza» (la avena pesa 80 g por
   taza en el catálogo). El display manda (`_reconcile_display_raw_lines`) y la media taza —40 g— pasó a la compra y a
   los macros: el día 1 bajó de 2.119 a 1.987 kcal.
2. **El empate del cuantizador** (`nutrition_db.quantize_ingredient_string`): los discretos iban a (0, ½, 1) y el
   empate lo ganaba el PRIMERO de la tupla: ¾ de pechuga → ½ (−33 %), 1¼ → 1. Las tres pechugas del almuerzo
   (160-170 g) salieron en «½ pechuga (≈100 g)» y la proteína del día 1 bajó de 129 a 118 g.
3. **El peso fingido** del pulido final: «0,58 pechuga (≈116 g)» salía «½ pechuga (≈100 g)» con 124 g en la compra:
   el cuantizador reescalaba también el peso que el humanizador había puesto para que se pese.

Y en el camino del modelo, el prompt recibía `"cookingTime": "none"` en crudo: 8 de 12 comidas por encima del tope de
10 min que el propio sistema audita. Ahora el valor viaja en claro con el MISMO tope que mide la auditoría."""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


# ─────────────────────────── 1. La taza del catálogo ───────────────────────────

def test_la_avena_no_se_muestra_en_la_taza_de_200_g(monkeypatch):
    import humanize_ingredients as hz
    monkeypatch.setattr("shopping_calculator.get_master_ingredients",
                        lambda: [{"name": "Avena", "density_g_per_cup": 80}], raising=False)
    out = hz.humanize_ingredient("115 g de avena")
    assert out != "½ taza de avena", "la media taza de avena son 40 g, no 115"
    assert out.startswith("1½ taza"), out     # 115 / 80 = 1,44 → cuartos → 1½


def test_la_taza_sale_del_catalogo_por_nombre_y_por_cabecera(monkeypatch):
    import humanize_ingredients as hz
    filas = [{"name": "Avena", "density_g_per_cup": 80}, {"name": "Yogurt griego entero", "density_g_per_cup": 245},
             {"name": "Arroz blanco", "density_g_per_cup": 185}]
    monkeypatch.setattr("shopping_calculator.get_master_ingredients", lambda: filas, raising=False)
    assert hz._cup_grams_for("avena en hojuelas", "avena") == 80          # la cabecera del nombre
    assert hz._cup_grams_for("yogurt griego entero", "yogurt") == 245     # exacto
    assert hz._cup_grams_for("arroz blanco", "arroz") == 185


def test_sin_catalogo_el_respaldo_es_el_del_catalogo_de_hoy(monkeypatch):
    import humanize_ingredients as hz
    monkeypatch.setattr("shopping_calculator.get_master_ingredients", lambda: [], raising=False)
    assert hz._cup_grams_for("avena", "avena") == 80.0
    assert hz._cup_grams_for("pasta integral", "pasta") == 140.0
    assert hz._GRAIN_CUP_FALLBACK_G["avena"] < 200, "el respaldo no puede volver a la taza de 200 g para todo"


def test_la_taza_mostrada_pesa_lo_que_la_linea(monkeypatch):
    """Cierre: lo que el sistema VUELVE a leer de la taza (densidad del catálogo) es lo que había en gramos (±15 %)."""
    import humanize_ingredients as hz
    monkeypatch.setattr("shopping_calculator.get_master_ingredients",
                        lambda: [{"name": "Avena", "density_g_per_cup": 80}], raising=False)
    for g in (60, 80, 115, 160):
        out = hz.humanize_ingredient(f"{g} g de avena")
        m = re.match(r"^(\d*)([¼½¾]?)\s+taza", out)
        assert m, out
        cups = (int(m.group(1)) if m.group(1) else 0) + {"": 0, "¼": .25, "½": .5, "¾": .75}[m.group(2)]
        assert abs(cups * 80 - g) / g <= 0.15, (g, out)


# ─────────────────────────── 2. Cuartos de conteo ───────────────────────────

def test_tres_cuartos_de_pechuga_no_bajan_a_media():
    from nutrition_db import quantize_ingredient_string as q
    out, fac = q("¾ pechuga de pollo (≈150 g)")
    assert fac == 1.0 and out.startswith("¾ pechuga"), (out, fac)
    out, fac = q("1¼ pechugas de pollo (≈250 g)")
    assert fac == 1.0, (out, fac)
    out, fac = q("¾ plátano verde")
    assert fac == 1.0, (out, fac)


def test_los_indivisibles_siguen_enteros_y_lo_raro_sigue_prohibido():
    from nutrition_db import quantize_ingredient_string as q
    assert q("0.66 huevos enteros")[0] == "1 huevos enteros"
    assert q("1¾ huevos")[0].startswith("2 huevos")
    assert q("3.87 papas medianas (580.23g)")[0].startswith("4 papas")       # el hint del modelo: contrato de siempre
    assert q("0.15 platano maduro grande (29.91g)")[0].startswith("0.5 platano")


def test_el_conteo_con_peso_del_humanizador_no_reescribe_el_peso():
    from nutrition_db import quantize_ingredient_string as q
    out, fac = q("0.58 pechuga de pollo (≈116.13 g)")
    assert fac == 1.0, "el peso manda: el conteo es solo una guía"
    assert "(≈116 g)" in out, out
    assert out.startswith("0.5 pechuga"), out


def test_la_cuantizacion_del_plan_no_toca_los_macros_de_tres_cuartos_de_pechuga():
    from graph_orchestrator import _apply_portion_quantization

    class _DB:
        def macros_from_ingredient_string(self, s):
            return {"protein": 45.0, "carbs": 0.0, "fats": 5.0, "kcal": 225.0}

    meal = {"ingredients": ["¾ pechuga de pollo (≈150 g)"], "ingredients_raw": ["160 g de pechuga de pollo cocida"],
            "protein": 45, "carbs": 0, "fats": 5, "cals": 225}
    _apply_portion_quantization({"days": [{"meals": [meal]}]}, _DB())
    assert meal["protein"] == 45 and meal["cals"] == 225, meal
    assert meal["ingredients_raw"] == ["160 g de pechuga de pollo cocida"], meal


def test_knob_de_rollback_existe():
    src = _src("nutrition_db.py")
    assert 'MEALFIT_QUANTIZE_COUNT_QUARTER_GRID' in src
    assert "tooltip-anchor: P1-PLAN-LOTE-172-CUARTOS" in src


# ─────────────────────────── 3. El tiempo de cocina, en claro ───────────────────────────

def test_el_prompt_recibe_el_tiempo_en_claro_con_el_tope_que_se_audita():
    import horizon
    out = horizon.explain_form_codes_for_prompt({"cookingTime": "none", "budget": "low"})
    assert out["cookingTime"] != "none"
    assert f"{horizon._COOKING_TIME_BUDGET_MIN['none']} minutos" in out["cookingTime"]
    assert out["budget"] == "low", "solo se traduce lo que el modelo no puede adivinar"
    assert "30 minutos" in horizon.explain_form_codes_for_prompt({"cookingTime": "30min"})["cookingTime"]
    assert "60 minutos" in horizon.explain_form_codes_for_prompt({"cookingTime": "1hour"})["cookingTime"]


def test_sin_codigo_conocido_el_formulario_queda_igual_y_no_se_muta():
    import horizon
    f = {"cookingTime": "raro", "a": 1}
    assert horizon.explain_form_codes_for_prompt(f) == f
    g = {"cookingTime": "none"}
    horizon.explain_form_codes_for_prompt(g)
    assert g == {"cookingTime": "none"}, "copia: el backend sigue leyendo el código"
    assert horizon.explain_form_codes_for_prompt(None) is None


def test_el_sanitizador_del_prompt_pasa_por_la_traduccion():
    go = _src("graph_orchestrator.py")
    i = go.index("def _sanitize_form_data_for_prompt")
    fin = go.index("\n# [P3-PLAN-MODEL-KNOBS", i)
    cuerpo = go[i:fin]
    assert 'explain_form_codes_for_prompt(' in cuerpo


def test_sin_tiempo_el_planificador_no_elige_horno_ni_guiso():
    """Con el tiempo ya en claro, el día seguía saliendo con «Pastelón gratinado» (30 min) y tilapia al horno: la técnica
    la imponía `_select_techniques`, que elegía al azar del catálogo entero."""
    import constants as c
    lentas = {"Horneado Saludable", "Guiso o Estofado Ligero", "Croquetas o Tortitas al Horno",
              "Relleno (Ej. Canoas, Vegetales rellenos)", "Desmenuzado (Ropa Vieja)", "Asado a la Parrilla"}
    rapidas = c.techniques_for_cooking_time("none")
    assert rapidas and not (set(rapidas) & lentas), rapidas
    assert len({c.TECH_TO_FAMILY[t] for t in rapidas}) >= 3, "el selector pide 3 familias distintas"
    assert c.techniques_for_cooking_time("plenty") == list(c.ALL_TECHNIQUES)
    assert c.techniques_for_cooking_time(None) == list(c.ALL_TECHNIQUES)
    import graph_orchestrator as g
    for _ in range(20):
        assert not (set(g._select_techniques(None, cooking_time="none")) & lentas)
    go = _src("graph_orchestrator.py")
    assert 'cooking_time=form_data.get("cookingTime")' in go, "el planificador tiene que pasar el tiempo del formulario"


def test_las_habas_son_legumbre_y_no_acompanan_la_fruta():
    """Batería real (alergia a lácteos): las TRES meriendas de fruta salieron con «habas cocidas» y el paso «Cocina habas
    a la plancha o hervidas». El cerrador de proteína ya excluye legumbres en lo dulce, pero su lista no tenía «haba»."""
    import graph_orchestrator as g
    from constants import strip_accents as sa
    assert any(h in "habas cocidas" for h in g._LEGUME_PROTEIN_HINT)
    assert g._is_sweet_meal({"name": "Manzana fresca con maní tostado", "meal": "Merienda",
                             "ingredients": ["1 manzana mediana", "30 g de maní tostado"]}, sa)


def test_el_nombre_del_plato_del_modelo_marca_su_identidad():
    import identidad_plato as ip
    assert ip.nombrada_en_el_nombre("Maní tostado con pasas", "1.44 g de maní tostado sin sal")
    assert ip.nombrada_en_el_nombre("Panqueques de avena con queso fresco", "15 g de avena en hojuelas")
    assert ip.nombrada_en_el_nombre("Pollo guisado con arroz", "150 g de pechuga de pollo")
    assert not ip.nombrada_en_el_nombre("Queso blanco a la plancha", "90 g de arroz blanco"), "«blanco» no es identidad"
    assert not ip.nombrada_en_el_nombre("Pollo guisado con arroz", "1 cda de aceite de oliva")
    assert ip.protege_linea({"name": "Maní tostado con pasas"}, "5 g de maní tostado"), "plato del modelo: el nombre manda"


def test_el_solver_no_baja_el_mani_del_plato_de_mani_a_migajas():
    """«Maní tostado con pasas…» con 1,44 g de maní: el solver lo llevaba a su cota inferior para cuadrar la grasa."""
    from portion_solver import solve_meal_macros, SOLVER_IDENTITY_MIN_SCALE

    class _DB:
        _T = {"mani": (5.67, 0.26, 0.16, 0.49), "pasas": (2.99, 0.03, 0.79, 0.005)}

        def _row(self, s):
            return next((v for k, v in self._T.items() if k in s.lower().replace("í", "i")), None)

        def macros_from_ingredient_string(self, s):
            m = re.match(r"\s*(\d+(?:\.\d+)?)\s*g", s)
            r = self._row(s)
            if not (m and r):
                return None
            g = float(m.group(1))
            return {"kcal": r[0] * g, "protein": r[1] * g, "carbs": r[2] * g, "fats": r[3] * g}

    lineas = ["30 g de maní tostado", "40 g de pasas"]
    objetivo = {"kcal": 170, "protein": 3, "carbs": 30, "fats": 3}   # pide casi sin grasa: el maní es la grasa
    sin = solve_meal_macros(lineas, objetivo, db=_DB())
    con = solve_meal_macros(lineas, objetivo, db=_DB(), dish_name="Maní tostado con pasas")
    g_sin = float(re.match(r"\s*([\d.]+)", sin["ingredients"][0]).group(1))
    g_con = float(re.match(r"\s*([\d.]+)", con["ingredients"][0]).group(1))
    assert g_sin < 30 * 0.5, "precondición: sin nombre el solver lo hunde"
    assert g_con >= 30 * SOLVER_IDENTITY_MIN_SCALE - 0.5, (g_sin, g_con)


def test_embarazo_el_plan_dice_la_especie_del_pescado_y_que_el_queso_es_pasteurizado():
    """Batería real (Embarazo): dos rechazos CRÍTICOS del revisor —pescado sin especie, queso fresco sin pasteurizar— y el
    usuario recibió el plan de emergencia con la lista vacía."""
    import embarazo_seguro as e
    plan = {"days": [{"meals": [{"name": "Pescado blanco dorado en airfryer",
                                 "ingredients": ["150 g de filete de pescado blanco", "1¼ lonjas/pedazos de queso (≈31 g)",
                                                 "20 g de queso parmesano"],
                                 "ingredients_raw": ["150 g de filete de pescado blanco",
                                                     "30 g de queso blanco fresco bajo en sodio", "20 g de queso parmesano"],
                                 "_display": {"en-US": {}}}]}]}
    assert e.etiquetar(plan, {"medicalConditions": ["Embarazo"]}) == 1
    m = plan["days"][0]["meals"][0]
    assert m["name"] == "Tilapia dorada en airfryer"
    assert m["ingredients"][0] == "150 g de filete de tilapia" and m["ingredients_raw"][0] == "150 g de filete de tilapia"
    assert "queso pasteurizado (≈31 g)" in m["ingredients"][1]
    assert "pasteurizado" in m["ingredients_raw"][1]
    assert m["ingredients"][2] == "20 g de queso parmesano", "el queso curado no necesita la etiqueta"
    assert "_display" not in m, "la traducción espejaba las líneas viejas"
    assert e.etiquetar(plan, {"medicalConditions": ["Embarazo"]}) == 0, "idempotente"
    otro = {"days": [{"meals": [{"name": "Pescado guisado", "ingredients": ["150 g de pescado"]}]}]}
    assert e.etiquetar(otro, {"medicalConditions": ["Ninguna"]}) == 0, "sin embarazo no se toca nada"
    import graph_orchestrator as g
    import inspect
    assert "embarazo_seguro" in inspect.getsource(g._apply_condition_substitutions)


def test_el_cerrador_de_proteina_respeta_lo_que_no_le_gusta():
    """Batería real («No me gusta: Pescado»): el cerrador pegó «80 g de atún en agua» a un desayuno y renombró el plato."""
    import constants as c
    fd = {"allergies": ["Ninguna"], "dislikes": ["Pescado", "Berenjena"]}
    assert c.alergias_y_rechazos(fd) == ["Pescado", "Berenjena"]
    assert c.alergias_y_rechazos({"allergies": ["Ninguna"], "dislikes": ["Ninguno"]}) == []
    import graph_orchestrator as g
    assert "atun" in g._expand_allergy_declarations(c.alergias_y_rechazos(fd))
    go = _src("graph_orchestrator.py")
    assert go.count('__import__("constants").alergias_y_rechazos(form_data)') >= 7, \
        "los cuatro constructores de candidatos y los tres cerradores tienen que recibir también los rechazos"
    assert '_safe_high_density_proteins(form_data.get("allergies")' not in go
    assert '_safe_high_density_proteins((form_data or {}).get("allergies")' not in go


def test_el_sanitizador_real_traduce_el_tiempo():
    import graph_orchestrator as g
    out = g._sanitize_form_data_for_prompt({"cookingTime": "none", "_x": 1, "country": "DO"})
    assert "_x" not in out and "country" not in out
    assert out["cookingTime"].startswith("none = NO TIENE TIEMPO"), out


# ─────────────────────────── 4. La descripción no puede hablar de otro plato ───────────────────────────

def test_la_desc_de_un_pavo_guisado_no_dice_pollo():
    """Batería real: «Pavo guisado ligero…» describía «una preparación ligera de pollo». El pase de honestidad no
    conocía el pavo y además daba el pollo por PRESENTE porque «pechuga» (de pavo) casaba con su patrón."""
    import graph_orchestrator as g
    days = [{"meals": [{"name": "Pavo guisado ligero con cundeamor y maíz",
                        "desc": "Una preparación ligera de pollo con cundeamor, cebolla, ajo y maíz.",
                        "ingredients": ["140 g de pechuga de pavo", "270 g de maíz dulce en granos"]},
                       {"name": "Pollo guisado", "desc": "Pollo guisado con papas.",
                        "ingredients": ["150 g de pechuga de pollo", "100 g de papa"]}]}]
    assert g._desc_food_honesty_pass(days) == 1
    assert "de pavo con cundeamor" in days[0]["meals"][0]["desc"]
    assert days[0]["meals"][1]["desc"] == "Pollo guisado con papas.", "una pechuga a secas sigue siendo pollo"


def test_el_plato_determinista_trae_descripcion_propia_y_no_el_relleno():
    """«Comida saludable y balanceada.» era la descripción de TODO plato determinista (assemble la rellenaba)."""
    import deterministic_day as d
    txt = d._descripcion_de([[160, "Filete de pescado blanco", "proteico"], [70, "Arroz blanco", "carbo"],
                             [5, "Aceite de oliva", "otro"], [1, "Sal", "otro"], [40, "Cebolla", "otro"]],
                            "almuerzo", "25 min")
    assert txt == "Plato fuerte con filete de pescado blanco y arroz blanco. Se prepara en unos 25 min.", txt
    assert "aceite" not in txt.lower() and "sal " not in txt.lower()
    assert d._descripcion_de([], "cena") == "", "sin ingredientes no se inventa nada: assemble pone su relleno"
    cuerpo = _src("deterministic_day.py").split("def construir_comida", 1)[1].split("\ndef ", 1)[0]
    assert 'meal["desc"] = _d' in cuerpo, "construir_comida tiene que poner la descripción"


# ─────────────────────────── marcador ───────────────────────────

def test_marker_172():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src("app.py"))
    assert m and int(m.group(1)) >= 172 and m.group(2) >= "2026-09-23"
