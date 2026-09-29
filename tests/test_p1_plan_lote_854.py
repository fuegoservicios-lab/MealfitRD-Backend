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
    assert "con lechosa crujiente" in m["desc"] and "manzana" not in m["desc"]


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


@pytest.mark.parametrize("pais", ["DO", None, "", "XX"])
def test_do_y_planes_sin_pais_no_cambian_nada(pais):
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
