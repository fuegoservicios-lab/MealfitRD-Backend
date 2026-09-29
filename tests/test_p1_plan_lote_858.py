# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-858 · 2026-09-29] La ficha no sirve lo que el plato ya no lleva, y RD también se corrige.

(A) Batería de embarazo (otra sesión, 29-sep): una cena que ya lleva pechuga decía «servido con huevo duro» — el
reparador 426 cambió el huevo por pechuga y la ficha no se enteró. `descripcion_veraz.alinear_con_ingredientes`
devolvía 0 cambios: el huevo no tiene sustituto de su familia estrecha y la retirada del 854 sólo actuaba cuando la
mención CERRABA su cláusula («huevo duro y ensalada» no la cierra). Ahora se quita el miembro de la enumeración (o la
cláusula entera) y la frase queda gramatical. Las ausencias declaradas («sin huevo», «sin lácteos») no se tocan.

(B) El 854 no se aplicaba a los planes de RD (mercado principal) por cautela; medido con replay sin IA sobre los
planes guardados y los 13 planes DO vivos de producción, se abre por defecto (`MEALFIT_DESCRIPTION_TRUTH_DO`=True).
"""
from __future__ import annotations

import ast
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
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")


def _comida(desc, ingredientes, nombre="Pechuga a la plancha con ensalada"):
    return {"meal": "Cena", "name": nombre, "desc": desc, "ingredients": list(ingredientes)}


_PECHUGA = ["Pechuga de pollo", "Lechuga", "Tomate"]   # el caso real: tal cual lo dejó el reparador 426


# ---------------------------------------------------------------- (A) el alimento que ya no está, fuera de la ficha
@pytest.mark.parametrize("desc,despues", [
    # el caso de la batería de embarazo, literal
    ("Pechuga jugosa a la plancha, servido con huevo duro y ensalada fresca.",
     "Pechuga jugosa a la plancha con ensalada fresca."),
    # el participio que concuerda con el plato se queda; el que no («Pechuga…, servido») cede su sitio a «con»
    ("Pechuga jugosa a la plancha, servida con huevo duro y ensalada fresca.",
     "Pechuga jugosa a la plancha, servida con ensalada fresca."),
    ("Pechuga jugosa a la plancha, acompañada de huevo duro y ensalada fresca.",
     "Pechuga jugosa a la plancha, acompañada de ensalada fresca."),
    ("Pechuga jugosa a la plancha con huevo duro y ensalada fresca.",
     "Pechuga jugosa a la plancha con ensalada fresca."),
    # el fantasma es el último miembro, tras «y», con su adjetivo
    ("Pechuga jugosa a la plancha con ensalada fresca y huevo duro.",
     "Pechuga jugosa a la plancha con ensalada fresca."),
    # …o el del medio de una enumeración
    ("Pechuga jugosa a la plancha con lechuga, huevo duro y tomate.",
     "Pechuga jugosa a la plancha con lechuga y tomate."),
    ("Pechuga jugosa a la plancha con lechuga, tomate y huevo duro picado.",
     "Pechuga jugosa a la plancha con lechuga y tomate."),
    # el único miembro: se va la cláusula entera, y la oración siguiente queda
    ("Pechuga jugosa a la plancha, servida con huevo duro. Una cena ligera y fresca para la noche.",
     "Pechuga jugosa a la plancha. Una cena ligera y fresca para la noche."),
    ("Pechuga jugosa a la plancha con huevo duro, lista en 15 minutos.",
     "Pechuga jugosa a la plancha, lista en 15 minutos."),
    # con un «con» antes en la oración, el resto se une con «y» (no «…con tomate con lechuga»)
    ("Pechuga jugosa a la plancha con tomate, servido con huevo duro y lechuga fresca.",
     "Pechuga jugosa a la plancha con tomate y lechuga fresca."),
    # el participio concuerda con el núcleo de la oración («Un plato…: …, acompañado de»): se queda
    ("Un plato sencillo: pechuga jugosa con tomate, acompañado de huevo duro y lechuga fresca.",
     "Un plato sencillo: pechuga jugosa con tomate, acompañado de lechuga fresca."),
    # el miembro de antes ya lleva su «y»: la coma, no «…cebolla y tomate y lechuga»
    ("Cena ligera: pechuga salteada con cebolla y tomate, huevos revueltos y lechuga fresca.",
     "Cena ligera: pechuga salteada con cebolla y tomate, lechuga fresca."),
    # el participio que sigue a la «y» es otra cláusula: queda con su coma
    ("Quinoa guisada con cebolla morada y tomate, coronada con huevos escalfados y acompañada de lechuga fresca.",
     "Quinoa guisada con cebolla morada y tomate, acompañada de lechuga fresca."),
    # el propósito que colgaba del alimento quitado se va con él («para una fruta distinta»)
    ("Pechuga jugosa a la plancha con lechuga, acompañada de mango fresco en cubos para una fruta distinta.",
     "Pechuga jugosa a la plancha con lechuga."),
    # miembro seguido de coma: si lo que sigue es otro alimento, es la enumeración; si no, es otra cláusula
    ("Pechuga jugosa a la plancha rellena con huevo bien cocido, tomate y lechuga; lista en minutos.",
     "Pechuga jugosa a la plancha rellena con tomate y lechuga; lista en minutos."),
    ("Pechuga guisada con tomate y cebolla, coronada con huevo bien cocido, acompañada de lechuga fresca.",
     "Pechuga guisada con tomate y cebolla, acompañada de lechuga fresca."),
    # «un huevo», «rodajas de huevo»: el determinante y la forma de corte se van con el alimento
    ("Pechuga jugosa a la plancha, servido con un huevo duro y ensalada fresca.",
     "Pechuga jugosa a la plancha con ensalada fresca."),
    ("Pechuga jugosa a la plancha con rodajas de huevo duro y ensalada fresca.",
     "Pechuga jugosa a la plancha con ensalada fresca."),
])
def test_el_alimento_que_ya_no_esta_se_quita_y_la_frase_queda_gramatical(desc, despues):
    m = _comida(desc, _PECHUGA)
    assert dv.alinear_con_ingredientes(m) >= 1
    assert m["desc"] == despues, m["desc"]


def test_el_participio_sin_genero_claro_se_queda():
    m = _comida("Mangú suave de plátano verde, coronado con huevo revuelto y cebollita morada salteada.",
                ["½ plátano verde", "¼ taza de cebollita morada en plumas", "50 g de queso blanco"], "Mangú")
    dv.alinear_con_ingredientes(m)
    assert m["desc"] == "Mangú suave de plátano verde, coronado con cebollita morada salteada.", m["desc"]


def test_lo_que_sigue_a_la_y_sin_ser_un_alimento_del_plato_no_se_une():
    """«…coronado con plátano maduro y coco rallado» sin coco en la lista: lo que sigue a la «y» tampoco está; no se
    sabe qué es, y no se adivina."""
    desc = "Un bowl frío y cremoso de guanábana, coronado con plátano maduro y coco rallado."
    m = _comida(desc, ["105 g de guanábana", "⅓ taza de yogurt natural"], "Bowl de guanábana")
    dv.alinear_con_ingredientes(m)
    assert m["desc"] == desc


def test_fruta_y_base_sin_sustituto_tambien_se_quitan():
    m = _comida("Yogur griego natural con granola casera y fresas frescas.",
                ["170 g de yogurt griego natural", "30 g de granola"], "Yogur con granola")
    dv.alinear_con_ingredientes(m)
    assert m["desc"] == "Yogur griego natural con granola casera.", m["desc"]
    m = _comida("Pollo guisado criollo con arroz blanco y ensalada verde.",
                ["150 g de pechuga de pollo", "1 taza de lechuga", "½ pepino"], "Pollo guisado con ensalada")
    dv.alinear_con_ingredientes(m)
    assert m["desc"] == "Pollo guisado criollo con ensalada verde.", m["desc"]


@pytest.mark.parametrize("desc", [
    # las ausencias declaradas son verdad (ronda 2 del 854): se quedan
    "Pechuga jugosa a la plancha con ensalada fresca, sin huevo ni lácteos.",
    "Pechuga jugosa a la plancha con ensalada fresca y sin huevo.",
    "Cena sin lácteos: pechuga jugosa a la plancha con ensalada fresca.",
    # el fantasma es el núcleo de la frase o parte de un nombre compuesto: no hay cláusula que quitar
    "Huevo duro con ensalada fresca y pechuga a la plancha.",
    "Tortilla de huevo con pechuga desmenuzada y ensalada fresca.",
    # el tramo lleva otra cosa que no sabemos si está («salsa de mango»): la duda, intacta
    "Pechuga jugosa a la plancha con salsa de mango y ensalada fresca.",
    # el tramo que se iría arrastra una ausencia declarada o algo que el plato sí lleva: la duda, intacta
    "Pechuga jugosa a la plancha con lechuga, acompañada de mango fresco para una cena sin lácteos.",
    "Pechuga jugosa a la plancha con lechuga, acompañada de mango fresco para realzar el tomate.",
    # «de otra comida»: no afirma que ESTE plato lo lleve
    "Pechuga jugosa a la plancha con ensalada fresca, más ligera que el arroz del almuerzo.",
])
def test_lo_que_no_es_un_fantasma_quitable_queda_igual(desc):
    m = _comida(desc, _PECHUGA)
    dv.alinear_con_ingredientes(m)
    assert m["desc"] == desc, m["desc"]


def test_el_fantasma_con_sustituto_de_su_familia_se_cambia_no_se_quita():
    m = _comida("Pechuga de pollo a la plancha con ensalada fresca y tomate.",
                ["150 g de pechuga de pavo", "1 taza de lechuga", "1 tomate"])
    dv.alinear_con_ingredientes(m)
    assert m["desc"] == "Pechuga de pavo a la plancha con ensalada fresca y tomate.", m["desc"]


def test_quitar_nunca_deja_una_ficha_de_menos_de_cuatro_palabras():
    m = _comida("Pechuga con huevo duro.", _PECHUGA)
    dv.alinear_con_ingredientes(m)
    assert m["desc"] == "Pechuga con huevo duro."


def test_idempotente():
    m = _comida("Pechuga jugosa a la plancha, servido con huevo duro y ensalada fresca.", _PECHUGA)
    dv.alinear_con_ingredientes(m)
    una = m["desc"]
    assert dv.alinear_con_ingredientes(m) == 0
    assert m["desc"] == una


# ---------------------------------------------------------------- (B) RD, el mercado principal, también
def _plan(pais, desc="Cena con identidad propia: pechuga de pollo salteada con cebolla.",
          ingredientes=("105 g de pechuga de pavo", "½ cebolla")):
    return {"_country": pais, "days": [{"day": 1, "meals": [_comida(desc, ingredientes)]}]}


@pytest.mark.parametrize("pais", ["DO", None])
def test_do_se_corrige_por_defecto(monkeypatch, pais):
    monkeypatch.delenv("MEALFIT_DESCRIPTION_TRUTH_DO", raising=False)
    p = _plan(pais) if pais else {"days": _plan("DO")["days"]}
    assert dv.aplicar_plan(p) == 1
    assert p["days"][0]["meals"][0]["desc"] == "Pechuga de pavo salteada con cebolla."


def test_el_knob_de_do_sigue_siendo_el_rollback(monkeypatch):
    monkeypatch.setenv("MEALFIT_DESCRIPTION_TRUTH_DO", "false")
    p = _plan("DO")
    antes = json.dumps(p, ensure_ascii=False, sort_keys=True)
    assert dv.aplicar_plan(p) == 0
    assert json.dumps(p, ensure_ascii=False, sort_keys=True) == antes
    assert dv.aplicar_plan(_plan("MX")) == 1, "apagar DO no apaga beta"


def test_los_knobs_se_registran_al_importar_el_modulo():
    """Al ARRANQUE, no en la primera ficha: `/admin/knobs` debe decir si DO está abierto antes de que llegue un plan."""
    src = (_BACKEND / "descripcion_veraz.py").read_text(encoding="utf-8")
    arbol = ast.parse(src)
    top = [n for n in arbol.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    llamadas = {getattr(c.func, "id", getattr(c.func, "attr", "")) for n in top for c in ast.walk(n)
                if isinstance(c, ast.Call)}
    assert {"activo", "incluye_do"} <= llamadas, "las dos lecturas del knob deben correr a nivel de módulo"
    # …y el módulo se importa al arranque (db_plans lo carga la fachada `db` en el import de la app)
    db_src = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    db_top = [n for n in ast.parse(db_src).body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert any(isinstance(n, ast.Import) and any(a.name == "descripcion_veraz" for a in n.names) for n in db_top)
    from knobs import get_knobs_registry_snapshot
    snap = get_knobs_registry_snapshot()
    assert "MEALFIT_DESCRIPTION_TRUTH" in snap and "MEALFIT_DESCRIPTION_TRUTH_DO" in snap


def test_knob_documentado_con_default_true():
    ref = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    fila = next(ln for ln in ref.splitlines() if ln.startswith("| `MEALFIT_DESCRIPTION_TRUTH_DO`"))
    assert "| `True` |" in fila and "P1-PLAN-LOTE-858" in fila


def test_el_marker_del_lote():
    src = (_BACKEND / "descripcion_veraz.py").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-858" in src and "tooltip-anchor: P1-PLAN-LOTE-858" in src


# ---------------------------------------------------------------- (B) lo que el replay de DO encontró roto, arreglado
# Replay sin IA con DO forzado (corpus de 780 planes + los 13 planes DO vivos): de 1 058 cambios únicos se leyeron los
# 38 de «alimento corregido», los 63 de «cláusula quitada», los 116 de «otro» y 100 de metalenguaje, y el detector
# ampliado pasó por los 1 058. Estas frases salían ROTAS (textos literales del corpus); con ellas DO no se podía abrir.
@pytest.mark.parametrize("antes,despues", [
    # «con identidad propia y lista…» dejaba «con lista»: lo que sigue es un predicativo, la preposición se va
    ("Cena fresca con identidad propia y lista en 10 minutos: wrap integral tibio relleno de huevo entero bien cocido "
     "en rodajas y pepino en láminas.",
     "Cena fresca lista en 10 minutos: wrap integral tibio relleno de huevo entero bien cocido en rodajas y pepino en "
     "láminas."),
    ("Salmón sellado a la plancha con limón, envuelto en tortilla integral con kale tibio y pepino fresco: cena ligera, "
     "con identidad propia y lista en minutos. Acompaña con agua.",
     "Salmón sellado a la plancha con limón, envuelto en tortilla integral con kale tibio y pepino fresco: cena ligera, "
     "lista en minutos. Acompaña con agua."),
    # «que respeta la categoría de…» dejaba «que respeta.» / «que respeta y arranca»
    ("Avena remojada en frío con fresas maduras, acompañada de huevos revueltos con cilantro fresco. Un desayuno "
     "fresco, ligero y completo que respeta la categoría de avena/cereales y arranca el sábado con energía.",
     "Avena remojada en frío con fresas maduras, acompañada de huevos revueltos con cilantro fresco. Un desayuno "
     "fresco, ligero y completo que arranca el sábado con energía."),
    ("Revoltillo bien cocido con tomate, ají y cebolla, acompañado de casabe crujiente y aguacate fresco: un desayuno "
     "práctico, sabroso y completo que respeta la categoría de revoltillo asignada.",
     "Revoltillo bien cocido con tomate, ají y cebolla, acompañado de casabe crujiente y aguacate fresco: un desayuno "
     "práctico, sabroso y completo."),
    ("Avena cocida lentamente en leche pasteurizada con canela y níspero maduro, terminada con maní tostado triturado "
     "para dar crocancia. Un desayuno caliente, cremoso y sin lácteos fermentados, que respeta la categoría de cereales "
     "asignada.",
     "Avena cocida lentamente en leche pasteurizada con canela y níspero maduro, terminada con maní tostado triturado "
     "para dar crocancia. Un desayuno caliente, cremoso y sin lácteos fermentados."),
    ("Plátano maduro asado hasta caramelizar, untado con mantequilla de maní cremosa, acompañado de ciruelas frescas y "
     "huevo cocido: un desayuno criollo, cálido y saciante que respeta la categoría de cereales sin repetir la avena "
     "del Día 3.",
     "Plátano maduro asado hasta caramelizar, untado con mantequilla de maní cremosa, acompañado de ciruelas frescas y "
     "huevo cocido: un desayuno criollo, cálido y saciante."),
    # «y con la categoría de pan/tostadas asignada para el día» dejaba «y con para el día»
    ("Panqueques tiernos de avena y huevo, servidos con mango fresco en cubos y un toque de chía. Un desayuno dulce, "
     "saciante y con la categoría de pan/tostadas asignada para el día.",
     "Panqueques tiernos de avena y huevo, servidos con mango fresco en cubos y un toque de chía. Un desayuno dulce y "
     "saciante."),
    # el muñón tras el signo («: cena.», «; un desayuno.») se va con su signo
    ("Pechuga de pollo a la plancha con ajo y limón, acompañada de yuca tierna cocida en microondas y nabo salteado con "
     "cilantro: cena con identidad propia, sin repetir el wrap del almuerzo.",
     "Pechuga de pollo a la plancha con ajo y limón, acompañada de yuca tierna cocida en microondas y nabo salteado con "
     "cilantro."),
    ("Panqueques suaves de harina de trigo, sin sal añadida, servidos con pera en gajos, granada y un toque de canela; "
     "un desayuno de categoría pan/tostadas con identidad propia.",
     "Panqueques suaves de harina de trigo, sin sal añadida, servidos con pera en gajos, granada y un toque de canela."),
    # «Una cena, sin lácteos.»: la coma que quedó entre el núcleo y su ausencia
    ("Tiras de pechuga de pollo salteadas a fuego alto con zanahoria, cebolla y ají cubanela, servidas junto a plátano "
     "maduro asado al horno. Una cena con identidad propia, base distinta a la pasta del almuerzo y sin lácteos.",
     "Tiras de pechuga de pollo salteadas a fuego alto con zanahoria, cebolla y ají cubanela, servidas junto a plátano "
     "maduro asado al horno. Una cena sin lácteos."),
])
def test_do_las_frases_que_el_replay_encontro_rotas(antes, despues):
    assert dv.limpiar_metalenguaje(antes) == despues


@pytest.mark.parametrize("desc,ingredientes,despues", [
    # «para romper la repetición de queso y aportar energía» dejaba «, aportar energía»: el propósito quitado se lleva
    # su coordinada, como ya hacía con «para variar» / «para no repetir»
    ("Merienda fresca y ligera: mango y lechosa en cubos con un toque de limón y maní tostado picado, sin lácteos, para "
     "romper la repetición de queso y aportar energía entre comidas.",
     ["70 g de mango", "25 g de lechosa", "25 g de maní tostado", "1½ limones", "85 g de queso cottage"],
     "Merienda fresca y ligera: mango y lechosa en cubos con un toque de limón y maní tostado picado."),
    # la ausencia falsa quitada dejaba la aposición muñón «, una merienda.»
    ("Casabe tostado con una capa de mantequilla de maní y un toque aromático de canela, una merienda sin lácteos y "
     "distinta al yogurt del día anterior.",
     ["½ tortas pequeñas de casabe", "1½ cdas de mantequilla de maní", "½ cdta de canela molida",
      "90 g de queso mozzarella"],
     "Casabe tostado con una capa de mantequilla de maní y un toque aromático de canela."),
    # quitar el último de dos predicativos tras una coma no convierte la coma en «y» («…con pasas y práctica»)
    ("Una porción sencilla y crujiente de maní tostado con pasas, práctica para llevar y sin lácteos.",
     ["20 g de maní tostado sin sal", "20 g de pasas", "120 g de queso cottage"],
     "Una porción sencilla y crujiente de maní tostado con pasas, práctica para llevar."),
    ("Mezcla energética de maní tostado, pasas y dátiles picados. Se arma en 3 minutos, sin cocción y sin lácteos, "
     "perfecta para la tarde.",
     ["15 g de maní", "15 g de pasas", "20 g de dátiles", "95 g de queso cottage"],
     "Mezcla energética de maní tostado, pasas y dátiles picados. Se arma en 3 minutos, sin cocción, perfecta para la "
     "tarde."),
    # …pero una lista de calificativos del mismo número sí se cierra con «y»
    ("Plátano maduro asado con mantequilla de maní. Un desayuno dominicano cálido, dulce y sin lácteos, listo en 8 "
     "minutos.",
     ["½ plátano maduro mediano", "2 cdtas de mantequilla de maní", "66 g de yogurt griego entero"],
     "Plátano maduro asado con mantequilla de maní. Un desayuno dominicano cálido y dulce, listo en 8 minutos."),
])
def test_do_las_fichas_que_el_replay_encontro_rotas(monkeypatch, desc, ingredientes, despues):
    monkeypatch.setenv("MEALFIT_DESCRIPTION_TRUTH_DO", "true")
    p = {"_country": "DO", "days": [{"day": 1, "meals": [_comida(desc, ingredientes, "Merienda")]}]}
    dv.aplicar_plan(p)
    assert p["days"][0]["meals"][0]["desc"] == despues, p["days"][0]["meals"][0]["desc"]


# ---------------------------------------------------------------- ronda del revisor: 8 fichas rotas que DO publicaba
# El revisor leyó las juntas de corte de los 1 014 cambios únicos de DO (no con detectores de patrón fijo, que estas
# 8 esquivaban): seis clases, todas del metalenguaje del 854. Textos literales del corpus.
@pytest.mark.parametrize("antes,despues", [
    # (1) «como base distinta a…» dejaba «como.» / «como,»: la base es verdad, la comparación no
    ("Bowl frío y armable con tiras de pechuga de pollo salteadas al limón y ajo sobre kale, zanahoria y puerro "
     "crujientes, con tortilla integral tostada en tiras como base distinta a la del almuerzo.",
     "Bowl frío y armable con tiras de pechuga de pollo salteadas al limón y ajo sobre kale, zanahoria y puerro "
     "crujientes, con tortilla integral tostada en tiras como base."),
    ("Cena con identidad propia: quinoa guisada como base distinta al plátano del almuerzo, pollo desmenuzado en un "
     "guiso ligero de berenjena, vainitas y pimiento, terminado con cilantro fresco. Saciante y sin arroz.",
     "Quinoa guisada como base, pollo desmenuzado en un guiso ligero de berenjena, vainitas y pimiento, terminado con "
     "cilantro fresco. Saciante y sin arroz."),
    ("Cena de sartén única: salmón sellado al limón sobre arepitas de maíz doradas y nabo salteado al wok con ajo y "
     "cebolla. Proteína animal como protagonista y el maíz como base distinta del arroz del mediodía.",
     "Cena de sartén única: salmón sellado al limón sobre arepitas de maíz doradas y nabo salteado al wok con ajo y "
     "cebolla. Proteína animal como protagonista y el maíz como base."),
    # …y un «como» sin sustantivo que conservar no queda colgando
    ("Pechuga guisada con cebolla y tomate, como opción distinta al almuerzo.",
     "Pechuga guisada con cebolla y tomate."),
    # (2) «Se prepara sin repetir la base…;» dejaba «Se prepara;»: el verbo sin su complemento se va con él
    ("Una cena ligera de tortitas horneadas con granos de maíz dulce, acompañadas de una salsa fresca de yogurt al ajo y "
     "coles de Bruselas tiernas. Se prepara sin repetir la base de harina del almuerzo; acompaña con agua para mantener "
     "una hidratación adecuada.",
     "Una cena ligera de tortitas horneadas con granos de maíz dulce, acompañadas de una salsa fresca de yogurt al ajo y "
     "coles de Bruselas tiernas. Acompaña con agua para mantener una hidratación adecuada."),
    # (3) «…distinta a la del desayuno y el almuerzo» dejaba «y el almuerzo»: la segunda comida sigue la comparación
    ("Tortitas horneadas de yautía con queso blanco pasteurizado, acompañadas de tomate, lechuga y aguacate; una cena "
     "casera con una preparación distinta a la del desayuno y el almuerzo.",
     "Tortitas horneadas de yautía con queso blanco pasteurizado, acompañadas de tomate, lechuga y aguacate; una cena "
     "casera."),
    # (4) «Distinta al almuerzo en base y técnica» dejaba «Técnica;»: la dimensión de la comparación se va con ella
    ("Cena ligera con identidad propia: filete de pescado blanco horneado sobre vegetales al tomate y acompañado de "
     "bollitos de harina al horno. Distinta al almuerzo en base y técnica; acompaña con agua.",
     "Filete de pescado blanco horneado sobre vegetales al tomate y acompañado de bollitos de harina al horno. Acompaña "
     "con agua."),
    # (5) «…para cerrar el dia variado y ligero» dejaba «…vegetales asados y ligero»: el adjetivo era del tramo quitado
    ("Cena con identidad propia y cero soya: pinchos de pollo jugosos alternados con aji morron, cebolla y tomate, "
     "sellados a la plancha y servidos sobre yuca hervida cremosa. Proteina magra, vegetales asados y un carbohidrato "
     "distinto al del almuerzo para cerrar el dia variado y ligero.",
     "Cena con cero soya: pinchos de pollo jugosos alternados con aji morron, cebolla y tomate, sellados a la plancha y "
     "servidos sobre yuca hervida cremosa. Proteina magra y vegetales asados."),
    # (6) «Una opción de cereal sin yogur, horneada para…» dejaba «Horneada para…» sin su sustantivo (y sin concordar
    # con «Tortitas»): quitado el sujeto, el participio que lo calificaba se va con él
    ("Tortitas tiernas de avena con chía y canela, acompañadas de guayaba fresca. Una opción de cereal sin yogur, "
     "horneada para un desayuno práctico.",
     "Tortitas tiernas de avena con chía y canela, acompañadas de guayaba fresca."),
    # …pero «nada» no es un participio (replay de esta ronda: la oración entera se iba)
    ("Cena ligera y con identidad propia: filete de tilapia al airfryer con limón, ajo y orégano, plátano maduro asado "
     "que se carameliza solo y un encurtido fresco de repollo morado. Base distinta al almuerzo, nada de arroz de noche.",
     "Filete de tilapia al airfryer con limón, ajo y orégano, plátano maduro asado que se carameliza solo y un encurtido "
     "fresco de repollo morado. Nada de arroz de noche."),
    # …ni el adjetivo que se lee solo: «Lista en menos de 25 minutos.» (G24 CO) se queda
    ("Arepas de maíz asadas en el budare, rellenas de queso ricotta. Más ligera que el almuerzo, lista en menos de 25 "
     "minutos.",
     "Arepas de maíz asadas en el budare, rellenas de queso ricotta. Lista en menos de 25 minutos."),
])
def test_ronda_revisor_las_ocho_fichas_rotas_de_do(antes, despues):
    assert dv.limpiar_metalenguaje(antes) == despues


_VEGETARIANA = ("Cena vegetariana con identidad propia: tortilla dorada de plátano maduro y queso de hoja a la plancha, "
                "servida con ensalada crujiente de repollo, zanahoria y limón.")


def test_la_etiqueta_de_dieta_verificada_se_queda():
    """(9, recomendado) «Cena vegetariana con identidad propia:» perdía «vegetariana» (muñón genérico), mientras «Cena
    sin gluten:» se conservaba. La dieta es verdad… si los ingredientes no la desmienten (backstop SSOT de dieta)."""
    ings = ["½ plátano maduro mediano, en rodajas", "25 g de queso de hoja", "1 taza de repollo rallado",
            "½ zanahoria rallada", "½ limón"]
    assert dv.limpiar_metalenguaje(_VEGETARIANA, ingredientes=ings) == (
        "Cena vegetariana: tortilla dorada de plátano maduro y queso de hoja a la plancha, servida con ensalada "
        "crujiente de repollo, zanahoria y limón.")


@pytest.mark.parametrize("ings", [
    None,                                                                   # sin ingredientes no se puede verificar
    ["½ plátano maduro mediano", "100 g de pechuga de pollo", "1 taza de repollo rallado", "½ zanahoria rallada"],
])
def test_la_etiqueta_de_dieta_que_no_se_puede_verificar_se_va(ings):
    assert dv.limpiar_metalenguaje(_VEGETARIANA, ingredientes=ings) == (
        "Tortilla dorada de plátano maduro y queso de hoja a la plancha, servida con ensalada crujiente de repollo, "
        "zanahoria y limón.")


# ---------------------------------------------------------------- las juntas que leí después del arreglo (esta ronda)
# Tras arreglar las 8 del revisor, el replay se leyó por JUNTAS (una línea por corte, 1 081 distintas de los 857 cambios
# de metalenguaje de DO): salieron fichas rotas que ningún detector de patrón fijo marcaba. Textos del corpus.
@pytest.mark.parametrize("antes,despues", [
    # la segunda mitad de «rompe la repetición de pollo y yautía de otros días» quedaba como objeto de «aporta»
    ("Gandules guisados con cebolla, ajo y limón, acompañados de plátano verde hervido y espinacas. Un guiso vegetal "
     "que aporta proteína de legumbre y rompe la repetición de pollo y yautía de otros días.",
     "Gandules guisados con cebolla, ajo y limón, acompañados de plátano verde hervido y espinacas. Un guiso vegetal "
     "que aporta proteína de legumbre."),
    # «de tu [categoría asignada]», «según [la categoría asignada]»: el determinante o la preposición colgando
    ("Avena que se hidrata sola en leche durante la noche: base de cereal de tu categoría asignada, coronada con melón "
     "dulce y frío.",
     "Avena que se hidrata sola en leche durante la noche: base de cereal, coronada con melón dulce y frío."),
    ("Pan integral tostado con huevo cocido y aguacate. Base de pan/tostadas según la categoría asignada, con el huevo "
     "cocido como proteína magra.",
     "Pan integral tostado con huevo cocido y aguacate. Base de pan/tostadas, con el huevo cocido como proteína magra."),
    # el sujeto sin su verbo ni su objeto: «La porción de quinoa;», «Las arepitas aportan.»
    ("Una cena ligera de quinoa con vainitas salteadas y queso blanco fresco. La porción de quinoa aporta una base "
     "distinta a la del almuerzo; acompaña la cena con agua.",
     "Una cena ligera de quinoa con vainitas salteadas y queso blanco fresco. Acompaña la cena con agua."),
    ("Una cena caliente, vegetal y sin lácteos, con soya texturizada guisada en tomate, cebolla y cilantro. Las "
     "arepitas aportan una base criolla distinta a la del almuerzo.",
     "Una cena caliente, vegetal y sin lácteos, con soya texturizada guisada en tomate, cebolla y cilantro."),
    # el verbo o el gerundio sin objeto: «que da.», «usando.»
    ("Plátano verde majado y horneado en capas con queso blanco fresco y cilantro. Un plato caliente y reconfortante "
     "que da una preparación distinta a la ensalada fría del almuerzo.",
     "Plátano verde majado y horneado en capas con queso blanco fresco y cilantro. Un plato caliente y reconfortante."),
    ("Bollitos tiernos de harina de maíz cocidos en un caldo suave de tomate, cebolla y ajo. Se sirve con queso blanco "
     "fresco y aguacate al final, usando una base distinta a la del almuerzo.",
     "Bollitos tiernos de harina de maíz cocidos en un caldo suave de tomate, cebolla y ajo. Se sirve con queso blanco "
     "fresco y aguacate al final."),
    # el adverbio sin su adjetivo: «y claramente.»
    ("Pechuga de pollo guisada con tomate, cebolla roja y ajo, servida con yuca hervida y coliflor al limón: una cena "
     "completa, sin pasta y claramente distinta del almuerzo.",
     "Pechuga de pollo guisada con tomate, cebolla roja y ajo, servida con yuca hervida y coliflor al limón: una cena "
     "completa."),                   # «sin pasta» era la justificación (base que rota, regla del 854)
    # el sustantivo cuyo único calificativo era la comparación: «con un formato.», «con un perfil.», «con una porción.»
    ("Cena ligera y rápida con sardinas como plato fuerte, acompañadas de tortilla integral, repollo salteado y manzana; "
     "sin arroz por la noche y con un formato distinto al almuerzo.",
     "Cena ligera y rápida con sardinas como plato fuerte, acompañadas de tortilla integral, repollo salteado y "
     "manzana."),                    # ídem «sin arroz por la noche»
    ("Tiras magras de res horneadas con maíz dulce, zanahoria y pimiento, sin tomate, con un perfil más ligero que el "
     "almuerzo y base distinta al casabe.",
     "Tiras magras de res horneadas con maíz dulce, zanahoria y pimiento, sin tomate."),
    ("Cena ligera, sin arroz, con plátano verde tierno y maní tostado; una preparación sencilla para terminar el día con "
     "una porción de carbohidrato distinta al almuerzo.",
     "Cena ligera, sin arroz, con plátano verde tierno y maní tostado; una preparación sencilla para terminar el día."),
    # la dimensión de la comparación también tras la coma: «distinta al almuerzo en base, técnica y vegetales»
    ("Pechuga salteada con repollo y zanahoria. Cena con identidad propia, distinta al almuerzo en base, técnica y "
     "vegetales.",
     "Pechuga salteada con repollo y zanahoria."),
    # el propósito del sujeto quitado: «Para no duplicar el tubérculo…» sin nada que lo rija
    ("Yautía tierna guisada en salsa natural de tomate, cebolla y ajo, con una ensalada fresca de remolacha. Base de "
     "yautía distinta a la papa del almuerzo, para no duplicar el tubérculo en las comidas fuertes.",
     "Yautía tierna guisada en salsa natural de tomate, cebolla y ajo, con una ensalada fresca de remolacha."),
    # «con identidad propia, fibra soluble y proteína completa»: el «con» rige también la lista de sustantivos
    ("Cena tibia de cebada con apio, tomate y cebolla salteados. Un plato con identidad propia, fibra soluble y proteína "
     "completa, distinto al almuerzo.",
     "Cena tibia de cebada con apio, tomate y cebolla salteados. Un plato con fibra soluble y proteína completa."),
    ("Filete de mero horneado envuelto con puerro y cilantro. Cena con identidad propia, proteína magra y guarnición de "
     "bajo índice glucémico.",
     "Filete de mero horneado envuelto con puerro y cilantro. Cena con proteína magra y guarnición de bajo índice "
     "glucémico."),
    # «; identidad propia de sartén, …» dejaba el fragmento «; de sartén,»
    ("Arepitas de maíz doradas, huevo guisado con nabo y una ensalada fresca de repollo con aguacate y limón; identidad "
     "propia de sartén, sin arroz ni yogur.",
     "Arepitas de maíz doradas, huevo guisado con nabo y una ensalada fresca de repollo con aguacate y limón; sin arroz "
     "ni yogur."),
])
def test_juntas_rotas_de_esta_ronda(antes, despues):
    assert dv.limpiar_metalenguaje(antes) == despues


@pytest.mark.parametrize("antes,despues", [
    # torpes (no rotas), baratas de arreglar: la «y» que cerraba la enumeración delante de «que/para»
    ("Pechuga de pollo horneada sobre pimientos, acompañada de un puré de auyama. Una cena ligera, jugosa y con "
     "identidad propia que no repite el plato del mediodía.",
     "Pechuga de pollo horneada sobre pimientos, acompañada de un puré de auyama. Una cena ligera y jugosa que no repite "
     "el plato del mediodía."),
    ("Trozos de mapuey hervidos con queso blanco fresco dorado a la plancha. Cena ligera, criolla y con identidad propia "
     "para cerrar el día.",
     "Trozos de mapuey hervidos con queso blanco fresco dorado a la plancha. Cena ligera y criolla para cerrar el día."),
    # «: merienda distinta[, sin repetir…].»: el «distinta» era de la comparación quitada
    ("Tostadas de casabe con queso blanco fresco bajo en sodio, lechosa y granada fresca: merienda distinta, sin repetir "
     "la avena del desayuno ni la base del almuerzo.",
     "Tostadas de casabe con queso blanco fresco bajo en sodio, lechosa y granada fresca."),
    # la coma entre el núcleo genérico y su predicativo: «Un desayuno, ideal para…», «: cena, lista en…»
    ("Mangú dominicano cremoso de plátano verde, acompañado de una ensalada de aguacate y kiwi. Un desayuno con "
     "identidad propia, sin lácteos repetidos, ideal para arrancar el sábado.",
     "Mangú dominicano cremoso de plátano verde, acompañado de una ensalada de aguacate y kiwi. Un desayuno ideal para "
     "arrancar el sábado."),
    ("Wrap de pollo con puerro al limón: cena, distinta al bowl del almuerzo y lista en 10 minutos.",
     "Wrap de pollo con puerro al limón: cena lista en 10 minutos."),
])
def test_torpes_de_esta_ronda(antes, despues):
    assert dv.limpiar_metalenguaje(antes) == despues


# ---------------------------------------------------------------- ronda 3 · revisor: la «y» que se comía la coma
# «, acompañado de aguacate y melón fresco» sin melón: la regla que vuelve a unir la lista con «y» («con leche, chía y
# lechosa» → «con leche y chía») unía también el tramo del participio: «…cebollita y revoltillo y acompañado de
# aguacate». Las 4 fichas literales del corpus del revisor, quitando la fruta de los ingredientes.
@pytest.mark.parametrize("desc,ingredientes,despues", [
    ("Mangú suave con cebollita y revoltillo, acompañado de aguacate y melón fresco: un desayuno criollo y saciante "
     "para empezar el día con buen balance.",
     ["½ plátano verde mediano (89 g)", "1 huevo", "6 claras de huevo", "½ aguacate", "½ cebolla",
      "¾ cdta de aceite de oliva", "Sal al gusto"],
     "Mangú suave con cebollita y revoltillo, acompañado de aguacate: un desayuno criollo y saciante para empezar el "
     "día con buen balance."),
    ("Yautía majada con limón y cilantro, acompañada de huevo y fresas frescas: un desayuno dominicano sencillo y "
     "colorido.",
     ["¼ pedazo de yautía (≈60 g)", "3 huevos", "¼ cdta de aceite de oliva", "½ limón", "½ cda de cilantro picado",
      "Sal al gusto", "6 claras de huevo"],
     "Yautía majada con limón y cilantro, acompañada de huevo: un desayuno dominicano sencillo y colorido."),
    ("Bollitos tiernos de harina de Negrito, acompañados de huevo y naranja fresca para empezar el día con una comida "
     "criolla y sustanciosa.",
     ["25 g de Harina de Negrito", "3 huevos", "¼ cdta de polvo de hornear", "¼ cdta de aceite de oliva",
      "Sal al gusto", "6 claras de huevo"],
     "Bollitos tiernos de harina de Negrito, acompañados de huevo."),
    ("Avena tibia y saciante, acompañada de yogur y mango para empezar el día con fibra y energía estable.",
     ["35 g de avena", "¾ taza de yogurt griego sin azúcar", "¼ cdta de canela en polvo", "140 ml de agua"],
     "Avena tibia y saciante, acompañada de yogur."),
])
def test_ronda_3_el_tramo_del_participio_no_se_une_con_y(desc, ingredientes, despues):
    m = _comida(desc, ingredientes, "Desayuno")
    assert dv.alinear_con_ingredientes(m) >= 1
    assert m["desc"] == despues, m["desc"]


@pytest.mark.parametrize("desc,ingredientes,despues", [
    # la regla sigue uniendo la enumeración de alimentos (para la que se escribió)
    ("Avena cremosa cocida con leche, chía y lechosa fresca; un desayuno criollo y saciante.",
     ["40 g de avena", "1 taza de leche descremada", "1 cda de semillas de chía"],
     "Avena cremosa cocida con leche y chía; un desayuno criollo y saciante."),
    # «ensalada» tiene forma de participio y es un plato: se une
    ("Pechuga jugosa a la plancha con tomate, ensalada y mango fresco; una cena ligera.",
     ["Pechuga de pollo", "Lechuga", "Tomate"],
     "Pechuga jugosa a la plancha con tomate y ensalada; una cena ligera."),
    # replay de mutación (fichas literales del corpus): el alimento con «de» es un miembro, no una cláusula
    ("Desayuno práctico de pan integral tostado, mantequilla de maní y guayaba fresca; una combinación sencilla para "
     "empezar el día con energía.",
     ["1 rebanada de pan integral familiar", "1½ cucharadas de mantequilla de maní",
      "1 taza de yogurt griego entero pasteurizado"],
     "Desayuno práctico de pan integral tostado y mantequilla de maní; una combinación sencilla para empezar el día "
     "con energía."),
    ("Casabe dorado y crujiente con queso blanco fresco, rodajas de tomate y fresas jugosas: un desayuno dominicano de "
     "domingo, fresco y sin repetir el pan integral del resto del plan.",
     ["½ porción de casabe (15 g)", "30 g de queso blanco fresco", "1 tomate mediano", "1 taza de yogurt griego entero"],
     "Casabe dorado y crujiente con queso blanco fresco y rodajas de tomate: un desayuno dominicano de domingo, fresco "
     "y sin repetir el pan integral del resto del plan."),
    # una palabra sola con forma de participio («granada») sigue siendo un miembro
    ("Un desayuno fresco y sustancioso: avena suave cocida con leche y canela, coronada con lechosa, granada y queso "
     "fresco blanco desmenuzado para un contraste salado-dulce muy criollo.",
     ["30 g de avena", "60 g de lechosa", "60 g de guineo", "5 g de semillas de Linaza", "Canela en polvo al gusto",
      "80 g de yogurt griego entero", "110 ml de agua"],
     "Un desayuno fresco y sustancioso: avena suave cocida con leche y canela, coronada con lechosa y granada."),
])
def test_ronda_3_la_enumeracion_de_alimentos_se_sigue_uniendo(desc, ingredientes, despues):
    m = _comida(desc, ingredientes, "Plato")
    assert dv.alinear_con_ingredientes(m) >= 1
    assert m["desc"] == despues, m["desc"]
