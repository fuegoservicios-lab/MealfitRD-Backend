# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-854 · 2026-09-29] La ficha dice lo que el plato ES: ni lo que el prompt pidió ni lo que el plato fue.

G24 (29-sep, 6 planes reales con P1-PLAN-LOTE-815 desplegado): en los 6 países las descripciones repetían el
lenguaje del prompt («Una preparación transformada con identidad propia, distinta al almuerzo del día», «de categoría
Avena/Cereales», «sin repetir la base») y, tras las reparaciones, nombraban lo que el plato ya no lleva (ES D2 Cena
«pechuga de pollo» con pavo; MX D2 Merienda «manzana crujiente» con lechosa; PR D3 Merienda «sin lácteos… mango» con
lechosa y cottage; CO D2 Desayuno «cocida en leche» cocida en agua). Los textos de abajo son los de G24, literales.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import descripcion_veraz as dv  # noqa: E402


@pytest.fixture(autouse=True)
def _sistema_de_paises_encendido(monkeypatch):
    """Como en producción desde el flip del 18-ago: la puerta lee el país con `constants.country_for_plan`, que con
    `MEALFIT_COUNTRY_SYSTEM` apagado devuelve DO para todo plan (lo prueba el test de la sección 7)."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")

_META = re.compile(r"identidad propia|nombre propio|categor[ií]a|Avena/Cereales|tub[eé]rculo local|"
                   r"distint[ao]s? (?:a|al|de|del) |repetir|repetid|para variar|m[aá]s (?:ligera|suave) que|"
                   r"tema del d[ií]a|alternativa saciante al|transformad[ao](?! en)", re.IGNORECASE)


# ---------------------------------------------------------------- 1. metalenguaje (textos literales de G24)
@pytest.mark.parametrize("antes,despues", [
    # ES D1 Cena
    ("Cena de tortitas caseras de trigo doradas a la sartén y rellenas de pollo salteado con tomate, cebolla y "
     "auyama. Una preparación transformada con identidad propia, distinta al almuerzo del día.",
     "Cena de tortitas caseras de trigo doradas a la sartén y rellenas de pollo salteado con tomate, cebolla y "
     "auyama."),
    # US D2 Desayuno
    ("Avena caliente cocida en leche descremada con mango dulce, linaza molida y un toque de canela: un desayuno "
     "reconfortante de categoría Avena/Cereales que aporta energía estable para la mañana.",
     "Avena caliente cocida en leche descremada con mango dulce, linaza molida y un toque de canela: un desayuno "
     "reconfortante que aporta energía estable para la mañana."),
    # US D2 Merienda
    ("Merienda fresca y saciante: piña dorada a la parrilla con canela sobre yogurt griego natural y nueces "
     "troceadas. Sin avena y sin repetir la base del desayuno.",
     "Merienda fresca y saciante: piña dorada a la parrilla con canela sobre yogurt griego natural y nueces "
     "troceadas."),
    # MX D1 Cena
    ("Tortitas doradas de maíz rellenas de queso blanco, con calabacita y jitomate como acompañamiento. Una cena "
     "casera con una base distinta a la del almuerzo.",
     "Tortitas doradas de maíz rellenas de queso blanco, con calabacita y jitomate como acompañamiento."),
    # PR D2 Desayuno
    ("Un bowl frío y cremoso de mango con queso fresco, quinoa y chía; una alternativa saciante al desayuno de "
     "avena.",
     "Un bowl frío y cremoso de mango con queso fresco, quinoa y chía."),
    # PR D3 Almuerzo
    ("Tortitas doradas al horno de harina de yuca con atún y adobo, acompañadas de lechuga fresca y aguacate "
     "cremoso. Una preparación transformada, práctica y completa para el mediodía.",
     "Tortitas doradas al horno de harina de yuca con atún y adobo, acompañadas de lechuga fresca y aguacate "
     "cremoso. Una preparación práctica y completa para el mediodía."),
    # CO D1 Cena
    ("Cena colombiana sencilla y reconfortante: arepas de maíz asadas en el budare, rellenas de queso ricotta bajo "
     "en sodio y coronadas con tomate, cebolla roja y cilantro. Más ligera que el almuerzo, lista en menos de 25 "
     "minutos.",
     "Cena colombiana sencilla y reconfortante: arepas de maíz asadas en el budare, rellenas de queso ricotta bajo "
     "en sodio y coronadas con tomate, cebolla roja y cilantro. Lista en menos de 25 minutos."),
    # CO D2 Merienda
    ("Merienda fresca y saciante: yogur griego natural con fresas en rodajas y almendras laminadas, sin frutas "
     "tropicales para variar del resto del día.",
     "Merienda fresca y saciante: yogur griego natural con fresas en rodajas y almendras laminadas."),
    # CO D2 Cena
    ("Cena ligera y reconfortante: auyama guisada con cebolla, ají morrón y tomate, con huevo escalfado encima y "
     "plátano maduro asado al horno. Sin arepa ni maíz para no repetir el carbohidrato del almuerzo.",
     "Cena ligera y reconfortante: auyama guisada con cebolla, ají morrón y tomate, con huevo escalfado encima y "
     "plátano maduro asado al horno."),
    # ES D3 Desayuno
    ("Tortitas tiernas de quinoa y plátano, con un toque de canela: una opción de cereal sin avena para empezar el "
     "día.",
     "Tortitas tiernas de quinoa y plátano, con un toque de canela."),
])
def test_metalenguaje_de_g24_se_limpia(antes, despues):
    assert dv.limpiar_metalenguaje(antes) == despues
    assert dv.limpiar_metalenguaje(despues) == despues, "idempotente"


@pytest.mark.parametrize("etiqueta,resto", [
    ("Cena con identidad propia: ", "guiso rápido de frijoles horneados con pollo desmenuzado, tomate y especias."),
    ("Base de tubérculo local: ", "tortitas doradas de batata rallada a la plancha, con queso fresco y huevos."),
    ("Cena ligera de identidad propia: ", "pechuga de pollo magra marinada en ajo y limón, asada a la parrilla."),
])
def test_la_etiqueta_del_prompt_delante_de_los_dos_puntos_se_quita(etiqueta, resto):
    salida = dv.limpiar_metalenguaje(etiqueta + resto)
    assert salida == resto[0].upper() + resto[1:]


@pytest.mark.parametrize("igual", [
    "Huevos pericos jugosos con pimentón y cebolla, acompañados de arepa dorada y aguacate fresco.",
    "Uvas jugosas acompañadas de maní tostado para una merienda sencilla, sin yogur y fácil de llevar.",
    "Un almuerzo criollo transformado en tortitas doradas: pollo sazonado con ajo y cilantro.",
    "Plato fuerte criollo de lentejas suaves guisadas con papa. Acompaña con agua.",
    "Un desayuno de martes laboral, rápido de armar y con energía sostenida.",
])
def test_la_prosa_sin_metalenguaje_queda_igual(igual):
    assert dv.limpiar_metalenguaje(igual) == igual


def test_si_todo_es_metalenguaje_no_deja_la_ficha_vacia():
    solo_meta = "Cena con identidad propia, distinta al almuerzo."
    assert dv.limpiar_metalenguaje(solo_meta) == solo_meta, "vaciar la ficha es peor que dejarla"


# ---------------------------------------------------------------- 2. la ficha frente a los ingredientes
def _comida(desc, ingredientes, nombre="Plato"):
    return {"meal": "Cena", "name": nombre, "desc": desc, "ingredients": list(ingredientes)}


def test_es_d2_cena_pollo_en_la_ficha_pavo_en_el_plato():
    m = _comida("Pan plano casero de harina de trigo recién hecho a la sartén, relleno de pechuga de pollo salteada "
                "con calabacín, ají morrón y cebolla.",
                ["55 g de harina de trigo", "105 g de pechuga de pavo", "1½ calabacín", "1 ají morrón", "½ cebolla"],
                "Pan plano de harina con pavo salteado, calabacín y ají morrón")
    assert dv.alinear_con_ingredientes(m) >= 1
    assert "pechuga de pavo salteada" in m["desc"] and "pollo" not in m["desc"]


def test_us_d3_cena_pollo_desmenuzado_es_pavo():
    m = _comida("Guiso rápido de frijoles horneados con pollo desmenuzado, tomate y especias ahumadas.",
                ["160 g de pechuga de pavo", "1¼ tazas de frijoles horneados (de lata, escurridos)",
                 "⅔ taza de tomate picado"])
    dv.alinear_con_ingredientes(m)
    assert "con pavo desmenuzado" in m["desc"]


def test_mx_d2_merienda_manzana_es_lechosa():
    m = _comida("Una pausa sencilla y refrescante con manzana crujiente, almendras tostadas y un toque de canela.",
                ["½ lechosa mediana (405g)", "15 g de almendras fileteadas", "Canela en polvo al gusto",
                 "120 g de queso cottage"])
    dv.alinear_con_ingredientes(m)
    # [ronda 2 · revisor] la lechosa no cruje: el adjetivo era de la manzana y se va con ella
    assert m["desc"] == ("Una pausa sencilla y refrescante con lechosa, almendras tostadas y un toque de canela."), \
        m["desc"]


def test_pr_d3_merienda_sin_lacteos_falso_y_mango_es_lechosa():
    m = _comida("Merienda sin lácteos que combina mango dulce en cubos con maní tostado para un aporte de fruta "
                "fresca, fibra y grasas buenas.",
                ["300 g de lechosa", "10 g de maní tostado sin sal", "25 g de queso cottage"])
    dv.alinear_con_ingredientes(m)
    assert "sin lácteos" not in m["desc"], "lleva cottage: «sin lácteos» es falso"
    assert "lechosa dulce en cubos" in m["desc"] and "mango" not in m["desc"]


def test_co_d2_desayuno_cocida_en_leche_se_cuece_en_agua():
    m = _comida("Avena tibia cocida en leche con canela, coronada con mango dulce y queso fresco desmenuzado.",
                ["30 g de avena en hojuelas", "140 g de mango", "30 g de queso fresco", "¼ cdta de canela en polvo",
                 "72 g de yogurt griego entero", "110 ml de agua"])
    dv.alinear_con_ingredientes(m)
    assert "cocida en agua" in m["desc"]


@pytest.mark.parametrize("desc,ingredientes", [
    # la leche SÍ está: no se toca
    ("Avena caliente cocida en leche descremada con mango dulce.", ["30 g de avena", "505 ml de leche descremada",
                                                                   "60 g de mango"]),
    # sinónimo regional resuelto por el SSOT (papaya ↔ lechosa): no es un fantasma
    ("Omelet ligero de nopales y tomate, acompañado de papaya fresca.", ["3 huevos", "1¼ tazas de lechosa en cubos"]),
    # merluza/pescado: normalize_ingredient_for_tracking los une
    ("Pescado blanco a la plancha con ajo y pimentón dulce.", ["100 g de filete de merluza", "1 diente de ajo"]),
    # «res» NO es subcadena de «fresco» (bug conocido): sin res en la ficha no hay nada que hacer
    ("Queso fresco con mango.", ["30 g de queso fresco", "100 g de mango"]),
    # negada: honesta
    ("Una cena ligera, sin arroz y con mucho sabor.", ["½ pechuga de pollo", "1 papa"]),
    # dos candidatos de la misma familia: ambiguo, no se adivina
    ("Yogur con manzana y canela.", ["1 taza de yogurt griego", "100 g de mango", "100 g de piña"]),
])
def test_lo_verdadero_o_ambiguo_no_se_toca(desc, ingredientes):
    m = _comida(desc, ingredientes)
    dv.alinear_con_ingredientes(m)
    assert m["desc"] == desc


# ---------------------------------------------------------------- 3. el plan: gate por país, knob, idempotencia
def _plan(pais, desc="Cena con identidad propia: pechuga de pollo salteada con cebolla.",
          ingredientes=("105 g de pechuga de pavo", "½ cebolla")):
    return {"_country": pais, "days": [{"day": 1, "meals": [_comida(desc, ingredientes)]}]}


def test_beta_se_limpia_y_marca_cuantas():
    p = _plan("ES")
    n = dv.aplicar_plan(p)
    assert n == 1
    assert p["days"][0]["meals"][0]["desc"] == "Pechuga de pavo salteada con cebolla."
    assert p["_description_truth"]["fichas"] == 1


# [P1-PLAN-LOTE-858] DO se abrió por defecto tras el replay; este test ancla ahora el ROLLBACK: con el knob de DO en
# False, DO y los planes sin país vuelven a salir byte a byte.
@pytest.mark.parametrize("pais", ["DO", None, "", "XX"])
def test_do_y_planes_sin_pais_no_cambian_nada(monkeypatch, pais):
    monkeypatch.setenv("MEALFIT_DESCRIPTION_TRUTH_DO", "false")
    p = _plan(pais) if pais is not None else {"days": _plan("DO")["days"]}
    antes = json.dumps(p, ensure_ascii=False, sort_keys=True)
    assert dv.aplicar_plan(p) == 0
    assert json.dumps(p, ensure_ascii=False, sort_keys=True) == antes, "DO es el control: byte a byte"


def test_knob_do_opt_in(monkeypatch):
    monkeypatch.setenv("MEALFIT_DESCRIPTION_TRUTH_DO", "true")
    p = _plan("DO")
    assert dv.aplicar_plan(p) == 1


def test_knob_apagado_no_toca(monkeypatch):
    monkeypatch.setenv("MEALFIT_DESCRIPTION_TRUTH", "false")
    p = _plan("MX")
    antes = copy.deepcopy(p)
    assert dv.aplicar_plan(p) == 0
    assert p == antes


def test_idempotente_sobre_el_plan():
    p = _plan("CO")
    dv.aplicar_plan(p)
    una = copy.deepcopy(p)
    assert dv.aplicar_plan(p) == 0
    assert p["days"] == una["days"]


# ---------------------------------------------------------------- 4. cableado y contrato del repo
def test_cableado_en_la_cola_del_escudo_tras_el_pulido_y_antes_de_restaurar_dias():
    src = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    cuerpo = src.split("def _finalize_plan_data_for_insert(")[1].split("\ndef ")[0]
    i_pulido = cuerpo.index("P1-PLAN-LOTE-181-PULIDO-COLA")
    i_ficha = cuerpo.index("P1-PLAN-LOTE-854-FICHA-VERAZ")
    i_restore = cuerpo.index("_rpd_frz(_pd, _frozen_token)")
    assert i_pulido < i_ficha < i_restore
    assert "descripcion_veraz" in cuerpo[i_ficha - 200:i_ficha + 600]


def test_knobs_documentados():
    ref = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    assert "`MEALFIT_DESCRIPTION_TRUTH`" in ref and "`MEALFIT_DESCRIPTION_TRUTH_DO`" in ref


def test_cada_familia_del_metalenguaje_cita_su_frase_del_prompt():
    """«No inventes»: cada familia de la lista cerrada nombra la frase del prompt de la que sale, y esa frase existe
    en el código de los prompts (o del revisor, que devuelve su crítica al prompt)."""
    fuentes = ""
    for f in ("prompts/day_generator.py", "prompts/planner.py", "prompts/asignacion_pais.py", "graph_orchestrator.py",
              "cron_tasks.py", "desayuno_por_alergia.py", "dish_library.py", "prompts/preferences.py"):
        fuentes += (_BACKEND / f).read_text(encoding="utf-8")
    fuentes_n = fuentes.lower()
    assert len(dv.FAMILIAS_DEL_PROMPT) >= 8
    for fam in dv.FAMILIAS_DEL_PROMPT:
        assert fam["origen"].lower() in fuentes_n, f"{fam['id']}: «{fam['origen']}» no está en los prompts"


def test_la_tooltip_anchor_del_modulo():
    assert "tooltip-anchor: P1-PLAN-LOTE-854" in (_BACKEND / "descripcion_veraz.py").read_text(encoding="utf-8")


def test_no_quedan_frases_del_prompt_en_los_textos_de_g24_tras_limpiar():
    textos = [
        "Cena con identidad propia: guiso rápido de frijoles horneados con pollo desmenuzado. Una cena ligera, con "
        "identidad propia y más suave que el almuerzo.",
        "Pechuga a la parrilla. Base distinta al almuerzo (sin legumbres) y más ligera que el almuerzo.",
        "Atún guisado con apio y ají morrón aportando ese toque fresco y dulce que pide el tema del día.",
        "Piña con almendras y queso fresco bajo en sodio, sin batidos ni lácteos repetidos.",
    ]
    for t in textos:
        assert not _META.search(dv.limpiar_metalenguaje(t)), dv.limpiar_metalenguaje(t)


# ---------------------------------------------------------------- 5. ronda 2 del revisor (29-sep)
# (1) «con identidad propia y sin X»: la preposición era del modificador, no de la ausencia.
@pytest.mark.parametrize("antes,despues", [
    ("Cena reconfortante de lentejas guisadas con pollo, ají morrón y espinacas, servida sobre yuca hervida y "
     "terminada con limón; un plato de cuchara con identidad propia y sin lácteos.",
     "Cena reconfortante de lentejas guisadas con pollo, ají morrón y espinacas, servida sobre yuca hervida y "
     "terminada con limón; un plato de cuchara sin lácteos."),
    ("Merienda con maní tostado y guineo, con nombre propio y sin gluten.",
     "Merienda con maní tostado y guineo, sin gluten."),
])
def test_propio_y_sin_no_deja_con_sin(antes, despues):
    salida = dv.limpiar_metalenguaje(antes)
    assert salida == despues
    assert not re.search(r"\b(?:con|de) (?:sin|ni|no)\b", salida)


# (2) la ausencia CLÍNICA (alergia, dieta, sal, azúcar) es verdad y se queda; la de una base que rota se va.
@pytest.mark.parametrize("antes,despues", [
    ("Pescado a la plancha con tayota. Sin lácteos y sin repetir la base del almuerzo.",
     "Pescado a la plancha con tayota. Sin lácteos."),
    ("Pescado a la plancha con tayota. Sin gluten y sin repetir la base del almuerzo.",
     "Pescado a la plancha con tayota. Sin gluten."),
    ("Guiso de pollo con papas y zanahoria. Sin cerdo y sin repetir la proteína del almuerzo.",
     "Guiso de pollo con papas y zanahoria. Sin cerdo."),
    ("Tostadas integrales con aguacate y huevo; sin azúcar añadida y sin repetir la base del desayuno.",
     "Tostadas integrales con aguacate y huevo; sin azúcar añadida."),
    ("Pechuga de pollo jugosa horneada con ajo, orégano y limón, servida con yautía suave. Una cena reconfortante, "
     "sin gluten y con base distinta a la del almuerzo.",
     "Pechuga de pollo jugosa horneada con ajo, orégano y limón, servida con yautía suave. Una cena reconfortante y "
     "sin gluten."),
    ("Cena sin gluten con identidad propia: ñame horneado hasta quedar tierno y dorado, huevo cuajado a la sartén.",
     "Cena sin gluten: ñame horneado hasta quedar tierno y dorado, huevo cuajado a la sartén."),
    ("Cena horneada de sabor criollo: capas de plátano maduro majado con queso blanco fresco. Plato fuerte propio, "
     "sin pescado ni gluten, distinto al almuerzo.",
     "Cena horneada de sabor criollo: capas de plátano maduro majado con queso blanco fresco. Plato fuerte propio, "
     "sin pescado ni gluten."),
    ("Batata tierna con relleno cremoso de queso blanco, cebolla y cilantro; una cena reconfortante, sin ser pesada, "
     "y distinta al almuerzo.",
     "Batata tierna con relleno cremoso de queso blanco, cebolla y cilantro; una cena reconfortante, sin ser pesada."),
])
def test_la_ausencia_clinica_se_conserva(antes, despues):
    assert dv.limpiar_metalenguaje(antes) == despues
    assert dv.limpiar_metalenguaje(despues) == despues, "idempotente"


@pytest.mark.parametrize("texto", [
    "Merienda fresca y salada: casabe crujiente con queso fresco y pepino. Sin avena ni yogurt, para no repetir "
    "bases del día.",
    "Quinoa con berenjena estofada. Sin arroz y sin repetir la base del almuerzo.",
    "Yogurt natural con fresas y maní tostado. Sin pan ni avena, para no repetir la base del desayuno.",
])
def test_la_ausencia_de_una_base_que_rota_se_va(texto):
    salida = dv.limpiar_metalenguaje(texto)
    assert not re.search(r"\bsin (?:avena|arroz|pan|yogurt)\b", salida, re.IGNORECASE), salida


def test_la_ausencia_falsa_de_la_justificacion_no_se_conserva():
    """«Sin lácteos y sin aguacate, para variar…» con aguacate en la lista: se queda sólo la parte verdadera."""
    m = _comida("Desayuno de categoría pan/tostadas: tostadas de maíz precocida bien crujientes, huevo revuelto con "
                "tomate, y aguacate fresco. Sin lácteos y sin aguacate, para variar del resto de la semana.",
                ["25 g de harina de maíz precocida", "3 huevos", "½ tomate", "½ aguacate mediano"])
    p = {"_country": "MX", "days": [{"day": 1, "meals": [m]}]}
    dv.aplicar_plan(p)
    assert m["desc"] == ("Tostadas de maíz precocida bien crujientes, huevo revuelto con tomate, y aguacate fresco. "
                         "Sin lácteos."), m["desc"]


def test_plan_con_alergia_a_lacteos_conserva_su_sin_lacteos():
    m = _comida("Cena con identidad propia: filete de pescado blanco al horno con ajo y limón, majado cremoso de "
                "tayota. Sin lácteos y sin repetir la base del almuerzo.",
                ["1½ filetes de pescado (≈245 g)", "250 g de tayota", "1 limón"])
    p = {"_country": "MX", "days": [{"day": 1, "meals": [m]}]}
    dv.aplicar_plan(p)
    assert m["desc"] == ("Filete de pescado blanco al horno con ajo y limón, majado cremoso de tayota. Sin lácteos."), \
        m["desc"]


# (3) la leche vegetal también es leche: «cocida en leche» con leche de almendras no pasa a «agua».
@pytest.mark.parametrize("leche", ["200 ml de leche de almendras", "200 ml de Leche de soya", "½ taza de leche vegetal"])
def test_cocida_en_leche_vegetal_no_pasa_a_agua(leche):
    desc = "Avena cremosa cocida en leche con canela y fresas."
    m = _comida(desc, ["40 g de avena en hojuelas", leche, "100 ml de agua", "80 g de fresas"])
    dv.alinear_con_ingredientes(m)
    assert m["desc"] == desc


# (4) yogur y queso de coco (o veganos) no son lácteos: el «sin lácteos» es verdad.
@pytest.mark.parametrize("desc,ingredientes", [
    ("Merienda sin lácteos: yogur de coco con fresas y almendras.",
     ["150 g de yogur de coco", "80 g de fresas", "10 g de almendras"]),
    ("Merienda vegana y sin lácteos, con yogur de coco y mango.", ["150 g de Yogur de coco", "80 g de mango"]),
    ("Tostada vegana sin lácteos, con queso vegano y tomate.", ["1 rebanada de pan", "30 g de queso vegano",
                                                                "1 tomate"]),
    ("Casabe con mantequilla de maní, sin lácteos.", ["1 casabe", "1 cda de Mantequilla de maní"]),
])
def test_yogur_de_coco_no_es_lacteo(desc, ingredientes):
    m = _comida(desc, ingredientes)
    dv.alinear_con_ingredientes(m)
    assert m["desc"] == desc


def test_cuajada_y_mantequilla_si_son_lacteos():
    for ing in ("40 g de cuajada", "1 cdta de mantequilla"):
        m = _comida("Arepa asada con hogao, sin lácteos y con café.", ["1 arepa", ing])
        dv.alinear_con_ingredientes(m)
        assert "sin lácteos" not in m["desc"], (ing, m["desc"])


# (5) familia «dulce»: el adjetivo que no casa con la lista blanca era de la fruta vieja.
def test_el_participio_de_preparacion_se_queda_y_concuerda():
    m = _comida("Batido de mango licuado con leche fría.", ["300 g de lechosa", "200 ml de leche"])
    dv.alinear_con_ingredientes(m)
    assert m["desc"] == "Batido de lechosa licuada con leche fría.", m["desc"]


def test_un_verbo_tras_la_fruta_no_es_un_adjetivo():
    m = _comida("Merienda de manzana aporta fibra y frescor.", ["300 g de lechosa"])
    dv.alinear_con_ingredientes(m)
    assert m["desc"] == "Merienda de lechosa aporta fibra y frescor.", m["desc"]


# (6) el nombre no se toca: `services.py` calcula `meal_names` antes del escudo.
def test_el_nombre_no_se_toca():
    m = _comida("Mangú cremoso con huevo y queso frito.", ["2 plátanos verdes", "2 huevos", "30 g de queso"],
                "Mangú transformado con huevo y queso")
    p = {"_country": "PR", "days": [{"day": 1, "meals": [m]}]}
    dv.aplicar_plan(p)
    assert m["name"] == "Mangú transformado con huevo y queso"


# menores: requesón/cuajada cuentan como queso; «plátano» (banana en ES/MX) con guineo no se borra.
@pytest.mark.parametrize("desc,ingredientes", [
    ("Tostadas de pan integral con tomate rallado y queso.", ["2 rebanadas de pan integral", "1 tomate",
                                                              "60 g de requesón"]),
    ("Arepa asada con hogao y queso.", ["1 arepa", "40 g de cuajada", "hogao"]),
    ("Tortitas de avena y claras con plátano.", ["40 g de avena", "3 claras", "1 guineo"]),
])
def test_hiponimos_cuentan_como_presentes(desc, ingredientes):
    m = _comida(desc, ingredientes)
    dv.alinear_con_ingredientes(m)
    assert m["desc"] == desc


# el paréntesis: «(distinta al bulgur del almuerzo)» se va entero, sin dejar «(,».
def test_el_parentesis_de_comparacion_se_va_entero():
    antes = ("Cena ligera y rápida con base de maíz dulce en granos (distinta al bulgur del almuerzo), queso blanco "
             "fresco, zanahoria rallada, pepino y cilantro al limón. Sin aguacate para variar respecto a los otros días.")
    salida = dv.limpiar_metalenguaje(antes)
    assert salida.count("(") == salida.count(")") and "(," not in salida, salida
    assert salida.startswith("Cena ligera y rápida con base de maíz dulce en granos, queso blanco fresco"), salida


# ---------------------------------------------------------------- 6. lo que la lectura del replay de la ronda 2 cazó
def test_quitar_una_ausencia_falsa_no_deja_una_oracion_munon():
    m = _comida("Casabe crujiente untado con mantequilla de maní natural y espolvoreado con canela, acompañado de mango "
                "fresco. Merienda sin lácteo, distinta a las meriendas de los otros días del plan.",
                ["1 porción de casabe (30 g)", "1½ cdas de mantequilla de maní natural", "40 g de mango",
                 "¾ taza de yogurt natural entero"])
    p = {"_country": "MX", "days": [{"day": 1, "meals": [m]}]}
    dv.aplicar_plan(p)
    assert m["desc"] == ("Casabe crujiente untado con mantequilla de maní natural y espolvoreado con canela, acompañado "
                         "de mango fresco."), m["desc"]


def test_el_adjetivo_que_concuerda_con_otro_sustantivo_no_se_toca():
    m = _comida("Yogur natural batido con jugo de limón y cubos de melón bien fríos, sin cocción.",
                ["⅓ taza de yogurt natural", "215 g de lechosa", "1 limón"])
    dv.alinear_con_ingredientes(m)
    assert m["desc"] == "Yogur natural batido con jugo de limón y cubos de lechosa bien fríos, sin cocción.", m["desc"]


def test_la_enumeracion_de_ausencias_se_filtra_entera():
    m = _comida("Pan integral crujiente con queso blanco fresco, huevo cocido y guanábana fresca en cubos. Sin yogurt, "
                "sin avena y sin lechosa, para romper la repetición de los días anteriores.",
                ["3 rebanadas de pan integral", "20 g de queso blanco", "65 g de guanábana", "60 g de huevo"])
    p = {"_country": "CO", "days": [{"day": 1, "meals": [m]}]}
    dv.aplicar_plan(p)
    assert m["desc"].endswith("guanábana fresca en cubos. Sin lechosa."), m["desc"]


def test_la_ausencia_que_queda_se_verifica_con_los_hiponimos():
    m = _comida("Pasta integral con champiñones y una salsa cremosa de yogurt natural, limón y cilantro. Sin queso "
                "blanco ni aguacate, para variar los acompañamientos y moderar la sal.",
                ["40 g de pasta integral seca", "80 g de yogurt", "170 g de champiñones",
                 "70 g de queso cottage bajo en sodio"])
    p = {"_country": "US", "days": [{"day": 1, "meals": [m]}]}
    dv.aplicar_plan(p)
    assert m["desc"].endswith("limón y cilantro. Sin aguacate."), m["desc"]


def test_la_clausula_de_repeticion_no_se_traga_la_ausencia_clinica():
    m = _comida("Merienda sin lácteos ni cereal repetido: tostada integral con aguacate machacado, láminas de lechosa "
                "y maní triturado.",
                ["1 rebanada de pan integral", "½ aguacate mediano", "1 lechosa mediana (200g)", "15 g de maní"])
    p = {"_country": "ES", "days": [{"day": 1, "meals": [m]}]}
    dv.aplicar_plan(p)
    assert m["desc"] == ("Merienda sin lácteos: tostada integral con aguacate machacado, láminas de lechosa y maní "
                         "triturado."), m["desc"]


# ---------------------------------------------------------------- 7. ronda 3 del revisor (29-sep): frases rotas
# El detector «0» de la ronda 2 no miraba la palabra que queda colgando antes del signo ni la «y» delante de una
# aposición con artículo; con DO forzado dejaba 8 textos rotos. Estos son los del revisor (`edge_r2.py`), con sus
# ingredientes, más el «Su y evita…» del corpus.
@pytest.mark.parametrize("desc,ingredientes,despues", [
    # (1) «…, con identidad propia y muy distinta al almuerzo» dejaba «con muy.»: «muy» no cierra una frase.
    ("Casabe crujiente horneado con huevos al plato y ricotta: una cena ligera, con identidad propia y muy distinta "
     "al almuerzo de pescado.",
     ["1 pieza de casabe", "2 huevos", "1½ cdas de queso ricotta"],
     "Casabe crujiente horneado con huevos al plato y ricotta: una cena ligera."),
    # (1) «Su base es distinta a la…» dejaba «Su;» / «Su y evita…»: el posesivo sin su nombre es un muñón.
    ("Un plato cálido de lentejas y batata con berenjena. Su base es distinta a la de batatas del almuerzo; "
     "acompáñalo con agua.",
     ["100 g de lentejas", "½ batata", "1 berenjena"],
     "Un plato cálido de lentejas y batata con berenjena. Acompáñalo con agua."),
    ("Cena caliente pero ligera, con tortillas tostadas a la parrilla y gandules guisados con vegetales. Su base es "
     "distinta a la del almuerzo y evita repetir pescado; acompáñala con agua.",
     ["2 tortillas de maíz", "½ taza de gandules", "1 taza de vegetales mixtos"],
     "Cena caliente pero ligera, con tortillas tostadas a la parrilla y gandules guisados con vegetales. Evita "
     "repetir pescado; acompáñala con agua."),
    # (1) «…, en una cena distinta al bowl…» dejaba «en.».
    ("Remolacha tierna con huevo bien cocido y casabe calentado al momento, en una cena distinta al bowl del "
     "almuerzo.",
     ["1 remolacha", "2 huevos", "1 casabe"],
     "Remolacha tierna con huevo bien cocido y casabe calentado al momento."),
    # (2)(3) una aposición que empieza por artículo no es un miembro de la enumeración: la coma se queda.
    ("Quinoa suelta con alcachofa al vapor y queso blanco fresco, una cena ligera y distinta al almuerzo. "
     "Acompáñala con agua.",
     ["½ taza de quinoa", "1 alcachofa", "30 g de queso blanco"],
     "Quinoa suelta con alcachofa al vapor y queso blanco fresco, una cena ligera. Acompáñala con agua."),
    ("Tortilla de maíz rellena de pollo a la plancha y vegetales frescos, una cena sabrosa y más ligera que el "
     "almuerzo.",
     ["1 tortilla de maíz", "120 g de pechuga de pollo", "1 taza de lechuga"],
     "Tortilla de maíz rellena de pollo a la plancha y vegetales frescos, una cena sabrosa."),
    ("Manzana crujiente con mantequilla de maní y un toque de canela, una merienda sencilla y sin yogur.",
     ["½ manzana", "1 cda de mantequilla de maní", "canela", "½ taza de yogurt griego"],
     "Manzana crujiente con mantequilla de maní y un toque de canela, una merienda sencilla."),
])
def test_ronda_3_ni_palabra_colgante_ni_y_ante_la_aposicion(desc, ingredientes, despues):
    m = _comida(desc, ingredientes)
    p = {"_country": "MX", "days": [{"day": 1, "meals": [m]}]}
    dv.aplicar_plan(p)
    assert m["desc"] == despues, m["desc"]
    assert not re.search(r"\b(?:su|sus|en|muy|tan|por|con|de|y|e)\s*[.;:!?]", m["desc"], re.IGNORECASE)
    assert not re.search(r"\b(?:y|e)\s+(?:una?|el|la|los|las)\s+(?:cena|merienda|desayuno|almuerzo|comida)\b",
                         m["desc"], re.IGNORECASE)


# (recomendado) la puerta lee el país del plan con `constants.country_for_plan`: con el sistema de países apagado
# (rollback del flip) todo plan es DO y un sello «ES» que quedó en `plan_data` no la abre.
def test_con_el_sistema_de_paises_apagado_un_sello_beta_no_abre_la_puerta(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "false")
    monkeypatch.setenv("MEALFIT_DESCRIPTION_TRUTH_DO", "false")   # [P1-PLAN-LOTE-858] DO abierto por defecto
    p = _plan("ES")
    antes = copy.deepcopy(p)
    assert dv.aplicar_plan(p) == 0
    assert p == antes
    monkeypatch.setenv("MEALFIT_DESCRIPTION_TRUTH_DO", "true")
    assert dv.aplicar_plan(p) == 1, "el opt-in de DO sigue valiendo para todo plan que la puerta lee como DO"


def test_la_puerta_usa_country_for_plan():
    src = (_BACKEND / "descripcion_veraz.py").read_text(encoding="utf-8")
    cuerpo = src.split("def aplica_a(")[1].split("\ndef ")[0]
    assert "country_for_plan(plan_data, None)" in cuerpo and "canonicalize_country(" not in cuerpo
