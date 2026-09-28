# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-745 · 2026-09-28] Ruido de la capa 1 del escáner culinario, clase por clase.

Medido el 28-sep sobre 848 comidas recientes (67 planes ya pasados por la cola del 744): la capa 1 marcaba **169
comidas (19,9 %)** y casi todo era ruido. Cada clase se corrige con su mecanismo, y cada test de abajo lleva el texto
REAL que la disparaba (falso positivo) junto al verdadero positivo que debe seguir disparando:

  · V8a — la nota de levotiroxina («separa estos alimentos … al menos 4 horas de la dosis») se leía como 4 h de
    espera oculta. Es un consejo de medicación, no un tiempo de cocina: 36 de 37 V8a del corpus.
  · V2  — «(ya viene cocido)» acusaba a TODOS los alimentos frescos del paso. En «… revuelve los huevos hasta que
    cuajen. Escurre e incorpora atún en agua (ya viene cocido)…» el que viene cocido es el atún, no el huevo: 43 de
    43 V2 del corpus. Ahora el estado es de su DUEÑO: el alimento que lo precede en la misma oración.
  · V1  — el aceite/la sal/el orégano no son el objeto del verbo sino el medio o el aliño; un verbo NEGADO («sin
    freír», «ni lo hiervas», «sin dejar que hierva», «sin cocción») no es una instrucción; «hornear» dentro de «polvo
    de hornear» / «freír» dentro de «queso de freír» es parte de un NOMBRE; «cocción» es un sustantivo («agua de
    cocción»); y «dorar» no es sólo saltear: el casabe se dora tostándolo y el plátano maduro friéndolo.
  · V7a — el singular gramatical: «corta 300 g de filete» (se mide en masa), «2 tortas de casabe» (el número va en
    la TORTA, no en el casabe), «5 aceitunas» (el nombre ya es plural y se leía como singular), «cocina huevo a la
    plancha» (sin artículo no dice «uno»), la nota de seguridad («cocina el huevo por completo») y «corta el tomate
    en cubitos» (cortar en trozos lo vuelve colectivo, como ya lo hacía «pica»).

Lo que se conserva, con test: la avena de la víspera (V8a), el pescado FRESCO «(ya viene cocido)» (V2), «Cuece el
casabe» y «Dora el queso de hoja» (V1), y «hierve el huevo» con 2 huevos en la lista (V7a).
tooltip-anchor: P1-PLAN-LOTE-745
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import culinary_coherence as cc  # noqa: E402
import culinary_context as cx  # noqa: E402

# Filas del catálogo real (SELECT del 28-sep): nombre, métodos y listo-para-comer tal cual; los ALIAS van RECORTADOS a
# los que estos tests necesitan (el Queso blanco real, p. ej., también lleva «queso» a secas).
_CAT = [
    {"name": "Huevo", "prep_methods": ["hervir", "plancha", "freir", "hornear", "guisar", "saltear"], "ready_to_eat": False,
     "category": "Proteínas", "aliases": ["huevos", "huevos enteros"]},
    {"name": "Clara de huevo", "prep_methods": ["hervir", "plancha", "freir", "hornear", "guisar", "saltear"],
     "ready_to_eat": False, "category": "Proteínas", "aliases": ["claras de huevo", "clara de huevo", "claras", "clara de huevos"]},
    {"name": "Atún en agua", "prep_methods": ["ninguno", "plancha", "saltear"], "ready_to_eat": True, "category": "Proteínas",
     "aliases": ["atún", "atún enlatado", "atún en lata"]},
    {"name": "Casabe", "prep_methods": ["tostar", "ninguno"], "ready_to_eat": True, "category": "Despensa",
     "aliases": ["casabe tradicional", "casabe dominicano"]},
    {"name": "Pechuga de pollo", "prep_methods": ["hervir", "plancha", "freir", "hornear", "guisar", "saltear"],
     "ready_to_eat": False, "category": "Proteínas", "aliases": ["pollo", "pechuga", "pechuga de pollo deshuesada"]},
    {"name": "Muslo de pollo", "prep_methods": ["hervir", "plancha", "freir", "hornear", "guisar", "saltear"],
     "ready_to_eat": False, "category": "Proteínas", "aliases": ["muslos de pollo"]},
    {"name": "Filete de pescado blanco", "prep_methods": ["hervir", "plancha", "freir", "hornear", "guisar", "saltear"],
     "ready_to_eat": False, "category": "Proteínas",
     "aliases": ["pescado blanco", "filete de pescado", "chillo", "pescado fresco", "pescado"]},
    {"name": "Aceite de oliva", "prep_methods": ["ninguno"], "ready_to_eat": True, "category": "Despensa",
     "aliases": ["aceite oliva", "aceite de oliva extra virgen"]},
    {"name": "Orégano dominicano", "prep_methods": ["ninguno"], "ready_to_eat": True, "category": "Despensa",
     "aliases": ["orégano", "oregano dominicano", "oregano seco"]},
    {"name": "Pimienta negra", "prep_methods": ["ninguno"], "ready_to_eat": True, "category": "Despensa",
     "aliases": ["pimienta", "pimienta molida"]},
    {"name": "Sal", "prep_methods": ["ninguno"], "ready_to_eat": True, "category": "Despensa",
     "aliases": ["sal marina", "sal rosada", "sal de mesa"]},
    {"name": "Queso blanco", "prep_methods": ["ninguno", "crudo", "licuar"], "ready_to_eat": True, "category": "Lácteos",
     "aliases": ["queso blanco fresco", "queso de freír", "queso fresco"]},
    {"name": "Queso de hoja", "prep_methods": ["ninguno", "crudo"], "ready_to_eat": True, "category": "Lácteos",
     "aliases": ["queso hoja"]},
    {"name": "Espinacas", "prep_methods": ["hervir", "saltear", "plancha", "hornear", "guisar", "crudo"], "ready_to_eat": None,
     "category": "Vegetales", "aliases": ["espinaca"]},
    {"name": "Polvo de hornear", "prep_methods": ["ninguno"], "ready_to_eat": True, "category": "Despensa",
     "aliases": ["polvo para hornear", "baking powder", "royal"]},
    {"name": "Avena", "prep_methods": ["hervir", "ninguno", "tostar"], "ready_to_eat": False, "category": "Despensa",
     "aliases": ["avena en hojuelas"]},
    {"name": "Leche descremada", "prep_methods": ["ninguno", "crudo", "hervir"], "ready_to_eat": True, "category": "Lácteos",
     "aliases": ["leche desnatada"]},
    {"name": "Plátano maduro", "prep_methods": ["hervir", "freir", "hornear", "guisar"], "ready_to_eat": None,
     "category": "Víveres", "aliases": ["plátanos maduros", "maduro"]},
    {"name": "Guineo", "prep_methods": ["crudo", "licuar", "ninguno"], "ready_to_eat": True, "category": "Frutas",
     "aliases": ["banana", "banano"]},
    {"name": "Guineo verde", "prep_methods": ["hervir", "freir", "hornear", "guisar"], "ready_to_eat": None,
     "category": "Víveres", "aliases": ["guineos verdes"]},
    {"name": "Tomate", "prep_methods": ["hervir", "saltear", "plancha", "hornear", "guisar", "crudo"], "ready_to_eat": None,
     "category": "Vegetales", "aliases": ["tomates"]},
    {"name": "Cebolla", "prep_methods": ["hervir", "saltear", "plancha", "hornear", "guisar", "crudo"], "ready_to_eat": None,
     "category": "Vegetales", "aliases": ["cebolla blanca", "cebolla roja"]},
    {"name": "Limón", "prep_methods": ["crudo", "licuar", "ninguno"], "ready_to_eat": True, "category": "Frutas",
     "aliases": ["limones", "jugo de limón"]},
    {"name": "Laurel", "prep_methods": ["ninguno"], "ready_to_eat": True, "category": "Despensa",
     "aliases": ["laurel", "hojas de laurel", "hoja de laurel"]},
    {"name": "Aceitunas", "prep_methods": ["ninguno"], "ready_to_eat": True, "category": "Despensa",
     "aliases": ["aceitunas verdes", "aceitunas negras"]},
    {"name": "Tortilla integral", "prep_methods": ["tostar", "ninguno", "saltear"], "ready_to_eat": True,
     "category": "Despensa", "aliases": ["tortillas integrales", "tortillas de trigo integral"]},
    {"name": "Yuca", "prep_methods": ["hervir", "freir", "hornear", "guisar"], "ready_to_eat": None, "category": "Víveres",
     "aliases": ["mandioca"]},
]


@pytest.fixture(scope="module")
def idx():
    return cc.build_culinary_index(_CAT)


def _meal(recipe, ingredients=(), prep_time=None, meal="Desayuno"):
    m = {"meal": meal, "name": "Plato", "recipe": list(recipe), "ingredients": list(ingredients)}
    if prep_time is not None:
        m["prep_time"] = prep_time
    return m


def _v1(recipe, idx, ingredients=()):
    return {(v["food"]) for v in cc._v1_verbo_alimento(1, _meal(recipe, ingredients), idx)}


# ───────────────────────────────────────────── V8a · la nota de medicación no es una espera

_NOTA_LEVO = ("⚕️ Nota clínica: toma la levotiroxina en ayunas y separa estos alimentos "
              "(lácteos/soya/linaza/espinacas/café/toronja) al menos 4 horas de la dosis — interfieren su absorción.")


def test_v8a_la_nota_de_levotiroxina_no_es_tiempo_de_espera():
    """36 de las 37 V8a del corpus del 28-sep: «4 horas de la dosis» es separación de un fármaco, no una espera."""
    m = _meal(["El Toque de Fuego: calienta el aceite y revuelve los huevos 3-4 min.", _NOTA_LEVO], prep_time="15 min")
    assert cx.hidden_wait_minutes(m["recipe"])[0] == 0
    assert cx.check_hidden_time(m) is None
    assert cc._v8a_tiempo_oculto(1, m, {}) == []


def test_v8a_otra_redaccion_de_la_dosis_tampoco():
    """La misma regla con la redacción de `medication_rules` («… al menos 4 horas de la pastilla»)."""
    pasos = ["⚕️ Separa los lácteos/calcio, el hierro y la soya al menos 4 horas de la pastilla."]
    assert cx.hidden_wait_minutes(pasos)[0] == 0
    # y sin el marcador ⚕️: lo que manda es que hable de la medicación
    assert cx.hidden_wait_minutes(["Toma tu medicamento y espera 2 horas antes de desayunar."])[0] == 0


def test_v8a_la_avena_de_la_vispera_SIGUE_disparando():
    """El único V8a verdadero del corpus: «la noche anterior» con 5 min declarados."""
    m = _meal(["Mise en place: La noche anterior mezcla ½ taza de avena (50 g) con 170 ml de leche descremada, 30 g de "
               "queso fresco batido, ¼ cdta de vainilla y ¼ cdta de canela en polvo en un envase con tapa; refrigera. "
               "Al servir, corta 185 g de melón en cubos de 1 cm.", _NOTA_LEVO], prep_time="5 min")
    h = cx.check_hidden_time(m)
    assert h and h["espera_min"] == 480, h
    assert [v["check"] for v in cc._v8a_tiempo_oculto(1, m, {})] == ["V8a"]


def test_v8a_el_marinado_real_sigue_disparando():
    m = _meal(["Marina el pollo 2 horas en la nevera.", "Ásalo a la plancha 8 min."], prep_time="15 min")
    assert cx.check_hidden_time(m)["espera_min"] == 120
    # el remojo de la víspera del corpus de 426 planes, con 25 min declarados
    m = _meal(["Mise en place: remoja las lentejas desde la noche anterior; pica la cebolla (5 g), el tomate (40 g) y "
               "el ajo."], prep_time="25 min")
    assert cx.check_hidden_time(m)["espera_min"] == 480
    # reposar a temperatura ambiente SÍ es una espera: sólo el TOPE de seguridad («más de 2 horas») no lo es
    assert cx.hidden_wait_minutes(["Deja reposar la masa a temperatura ambiente 2 horas."])[0] == 120
    # la «pastilla de caldo» es cocina, no medicación
    assert cx.hidden_wait_minutes(["Disuelve la pastilla de caldo en agua y marina el pollo 2 horas."])[0] == 120


def test_v8a_el_tope_de_seguridad_alimentaria_no_es_espera():
    """Conservación, no espera (corpus de 426 planes): el límite de 2 h fuera de la nevera."""
    for paso in (
        "⚠️ Seguridad alimentaria: recalienta el pollo ya cocido hasta que esté humeante en el centro (~74°C) y no lo "
        "dejes a temperatura ambiente más de 2 horas.",
        "⚠️ Seguridad alimentaria: cocina los huevos hasta que la clara y la yema estén firmes (~71°C) y refrigera lo "
        "que sobre dentro de 2 horas.",
        "El Toque de Fuego: calienta el arroz blanco en sartén 2-3 minutos. Usa huevos cocidos y refrigerados "
        "correctamente, sin dejarlos a temperatura ambiente por más de 2 horas.",
    ):
        assert cx.hidden_wait_minutes([paso])[0] == 0, paso


def test_v8a_el_consejo_de_no_acostarse_no_es_espera():
    paso = ("Montaje: sirve la crema templada con las tortillas de maíz al lado. Acompaña con agua y evita acostarte "
            "durante las 2-3 horas posteriores a la cena.")
    assert cx.hidden_wait_minutes([paso])[0] == 0


def test_v8a_el_prep_time_que_nombra_el_reposo_ya_lo_declaro():
    m = _meal(["Montaje: refrigera la mezcla durante la noche; al servir, coloca encima el mango y el kiwi."],
              prep_time="10 min más reposo nocturno")
    assert cx.check_hidden_time(m) is None
    assert cx.check_hidden_time(dict(m, prep_time="10 min (más refrigeración)")) is None
    assert cx.check_hidden_time(dict(m, prep_time="10 min")) is not None


# ───────────────────────────────────────────── V2 · el estado «ya viene cocido» es de su dueño

_PASO_ATUN = ("El Toque de Fuego: calienta ½ cdta de aceite de oliva en una sartén antiadherente a fuego medio y "
              "revuelve los huevos y las claras de huevo 3-4 min hasta que cuajen por completo. Escurre e incorpora "
              "atún en agua (ya viene cocido) a la preparación antes de servir.")


def test_v2_el_atun_ya_cocido_no_acusa_al_huevo_de_otra_oracion(idx):
    """43 de 43 V2 del corpus: el atún (listo para comer) viene cocido; el huevo de la oración anterior, no."""
    assert cc._v2_estado_imposible(1, _meal([_PASO_ATUN]), idx) == []


def test_v2_el_casabe_ya_cocido_no_acusa_a_los_demas(idx):
    paso = ("El Toque de Fuego: hierve el huevo 10 min; calienta el casabe en un sartén seco 1-2 min por lado, hasta "
            "que esté crujiente (el casabe ya está cocido: solo se tuesta, nunca se remoja ni se hierve).")
    assert cc._v2_estado_imposible(1, _meal([paso]), idx) == []


def test_v2_el_pescado_FRESCO_ya_cocido_SIGUE_disparando(idx):
    """El caso real que motivó P1-CLOSER-NOTE-FUSED-FRESHCOCIDO: pescado crudo declarado cocido."""
    paso = ("El Toque de Fuego: revuelve los huevos 3 min. Escurre e incorpora filete de pescado blanco (ya viene "
            "cocido) a la preparación antes de servir.")
    v = cc._v2_estado_imposible(1, _meal([paso]), idx)
    assert [x["food"] for x in v] == ["Filete de pescado blanco"], v


def test_v2_la_pechuga_fresca_ya_cocida_en_la_lista_y_en_el_paso(idx):
    v = cc._v2_estado_imposible(1, _meal(["Sirve la Pechuga de pollo (ya viene cocida) sobre la ensalada."],
                                          ["150 g de pechuga de pollo (ya viene cocida)"]), idx)
    assert [x["food"] for x in v] == ["Pechuga de pollo", "Pechuga de pollo"], v


def test_v2_sin_dueno_delante_mira_al_que_sigue(idx):
    v = cc._v2_estado_imposible(1, _meal(["Ya viene cocido el filete de pescado blanco: sírvelo con arroz."]), idx)
    assert [x["food"] for x in v] == ["Filete de pescado blanco"], v


# ───────────────────────────────────────────── V1 · medio, negación, nombre, sustantivo y «dorar»

def test_v1_el_aceite_y_los_alinos_no_son_el_objeto_del_verbo(idx):
    """rd545b: «sazona el pollo con pimienta negra y cocínalo en plancha con el aceite de oliva…»."""
    assert _v1(["El Toque de Fuego: sazona el pollo con pimienta negra y cocínalo en plancha con el aceite de oliva a "
                "fuego medio-alto durante 6-8 minutos, volteándolo, hasta que alcance 74 °C."], idx) == set()
    assert _v1(["Rocía con el aceite de oliva y cocina en airfryer a 200 °C durante 12-15 minutos."], idx) == set()


def test_v1_el_condimento_como_OBJETO_directo_sigue_acusado(idx):
    """La exención es del medio y del aliño, no del condimento que el verbo trabaja: «hierve la canela en polvo» (el
    caso de P1-PLAN-LOTE-67) y «fríe el orégano» siguen siendo hallazgos. Lo cazó el test del lote 67 en rojo."""
    cat = _CAT + [{"name": "Canela en polvo", "prep_methods": ["ninguno"], "ready_to_eat": True,
                   "category": "Despensa", "aliases": []}]
    i2 = cc.build_culinary_index(cat)
    assert _v1(["El Toque de Fuego: hierve la canela en polvo 10 minutos."], i2) == {"Canela en polvo"}
    assert _v1(["Fríe el orégano dominicano en abundante aceite."], idx) == {"Orégano dominicano"}
    # y con preposición por medio vuelve a ser medio/aliño
    assert _v1(["Guisa 8-10 minutos con el orégano y la sal."], idx) == set()


def test_v1_el_agua_dentro_del_nombre_no_lo_vuelve_condimento():
    """La exención mira la CABEZA del nombre: «X en agua» no es agua. Texto de rdv592 («suma el atún desmenuzado, orégano y
    2 cdas de agua y guisa 5-6 min»): el orégano calla y el alimento se sigue juzgando.

    [P1-PLAN-LOTE-745 · ronda 1] Con un alimento SINTÉTICO: el test fijaba antes que el «Atún en agua» REAL se acusara
    de guisar, y el atún guisado es un plato dominicano normal — eso es un hueco del catálogo (lo decide el dueño), no
    un hallazgo que un test deba dar por bueno."""
    i2 = cc.build_culinary_index(_CAT + [{"name": "Conserva de prueba en agua", "prep_methods": ["ninguno"],
                                          "ready_to_eat": True, "category": "Despensa",
                                          "aliases": ["conserva de prueba"]}])
    assert _v1(["Aparte calienta 1½ cdas de aceite a fuego medio; suma la conserva de prueba desmenuzada, orégano y 2 "
                "cdas de agua y guisa 5-6 min."], i2) == {"Conserva de prueba en agua"}


def test_v1_un_verbo_negado_no_es_una_instruccion(idx):
    for paso in (
        "Calienta ¼ cda de aceite de oliva en sartén antiadherente y sella el queso blanco fresco 1-2 min por lado, "
        "solo hasta dorar la superficie (sin empanizar ni freír).",
        "El Toque de Fuego: calienta los triángulos de casabe en un sartén seco a fuego medio durante 1-2 minutos por "
        "lado, hasta que estén crujientes (no lo humedezcas ni lo hiervas).",
        "🫓 Acompaña con el casabe de tus ingredientes: está listo para comer, sin cocción.",
        "Añade el queso blanco fresco y calienta 1 minuto más, sin dejar que hierva.",
        "El Toque de Fuego: calienta el casabe en un sartén seco (el casabe ya está cocido: solo se tuesta, nunca se "
        "remoja ni se hierve).",
    ):
        assert _v1([paso], idx) == set(), paso


def test_v1_la_negacion_no_tapa_la_orden_siguiente(idx):
    """«No lo hiervas» no exime al «hierve» afirmativo de la oración siguiente."""
    assert _v1(["No lo hiervas todavía. Hierve el casabe 5 minutos."], idx) == {"Casabe"}


def test_v1_un_verbo_dentro_del_nombre_de_un_alimento_no_es_un_verbo(idx):
    assert _v1(["Mise en place: mide 30 g de avena, 15 ml de leche descremada y ½ cdta de polvo de hornear; bate 1 "
                "huevo."], idx) == set()
    assert _v1(["Mise en place: lava 1½ tazas de espinacas y ralla 50 g de queso de freír."], idx) == set()


def test_v1_coccion_es_un_sustantivo(idx):
    """«desecha el agua de cocción» no hierve el guineo."""
    assert _v1(["El Toque de Fuego: hierve el guineo verde en agua 15-18 min. Escurre el guineo y desecha el agua de "
                "cocción."], idx) == set()


def test_v1_dorar_se_satisface_con_cualquier_metodo_que_dore(idx):
    """El casabe se dora TOSTÁNDOLO; el plátano maduro, FRIÉNDOLO. Ninguno de los dos se «saltea» en el catálogo."""
    assert _v1(["El Toque de Fuego: dora el casabe en una sartén seca a fuego medio durante 1 minuto por lado, hasta "
                "que esté crujiente."], idx) == set()
    assert _v1(["Aparte, dora el plátano maduro en un sartén a fuego medio 4 minutos por lado hasta caramelizar."],
               idx) == set()


def test_v1_los_verdaderos_SIGUEN_disparando(idx):
    """El golden set («Cuece el casabe»), el queso de hoja dorado (catálogo: sólo crudo) y la pechuga licuada."""
    assert _v1(["Cuece el Casabe 10 minutos en agua."], idx) == {"Casabe"}
    assert _v1(["El Toque de Fuego: calienta la 1½ cdtas de aceite de oliva en un sartén a fuego medio y dora el queso "
                "de hoja 2-3 minutos por lado hasta que se marque."], idx) == {"Queso de hoja"}
    assert _v1(["Licúa la Pechuga de pollo hasta obtener una crema."], idx) == {"Pechuga de pollo"}
    assert _v1(["Fríe el Casabe en abundante aceite."], idx) == {"Casabe"}


def test_v1_el_scan_completo_ve_lo_mismo(idx):
    plan = {"days": [{"day": 1, "meals": [_meal(["Hierve el Casabe."], ["30 g de casabe"]),
                                          _meal(["Dora el casabe en una sartén seca 1 minuto por lado."], ["30 g de casabe"],
                                                meal="Merienda")]}]}
    v1 = [(v["meal"], v["food"]) for v in cc.culinary_contract_scan(plan, _CAT) if v["check"] == "V1"]
    assert v1 == [("Desayuno", "Casabe")], v1


# ───────────────────────────────────────────── V7a · el singular gramatical

def _v7a(ings, pasos, idx):
    return [v["food"] for v in cc._v7a_lista_compra_de_mas(1, _meal(pasos, ings), idx)]


def test_v7a_el_nombre_ya_plural_no_es_un_singular(idx):
    """«5 aceitunas» + «corta las aceitunas» salía «(aceitunas, siempre en singular)»."""
    assert _v7a(["5 aceitunas"], ["Corta las aceitunas en rodajas y mide el aceite de oliva.",
                                   "Montaje: sirve la quinoa distribuyendo el queso y las aceitunas."], idx) == []


def test_v7a_una_mencion_en_masa_vuelve_colectivo_el_singular(idx):
    assert _v7a(["2 filetes de pescado (≈300 g)"],
                ["Mise en place: corta 300 g de filete de pescado blanco en porciones.",
                 "Unta el filete con parte del aceite, el ajo y la sal."], idx) == []
    assert _v7a(["5 claras de huevo"], ["El Toque de Fuego: bate 1 huevo con 135 g de clara de huevo."], idx) == []


def test_v7a_el_numero_va_en_la_unidad_no_en_el_alimento(idx):
    """«2 tortas pequeñas de casabe»: lo que se cuenta son TORTAS. «el casabe» es masa, como «el pan»."""
    assert _v7a(["2 tortas pequeñas de casabe"],
                ["Tuesta el casabe en una sartén seca 1-2 min por lado.",
                 "Montaje: acompaña con el casabe tostado."], idx) == []


def test_v7a_sin_articulo_y_en_notas_no_hay_evidencia(idx):
    assert _v7a(["2 huevos"], ["💪 Cocina huevo a la plancha o hervido y sírvelo como proteína del plato.",
                               "⚠️ Seguridad alimentaria: cocina el huevo por completo (≥71°C, yema y clara firmes) "
                               "antes de servir; evita el huevo crudo o poco cocido."], idx) == []
    assert _v7a(["2 limones"], ["Exprime limón sobre el pescado."], idx) == []


def test_v7a_el_jugo_del_limon_es_masa(idx):
    assert _v7a(["2 limones (su jugo)"], ["Mise en place: sazona el pescado con sal y el jugo del limón."], idx) == []
    assert _v7a(["2 limones"], ["Montaje: rellena cada tortilla, rocía el jugo de limón y enrolla."], idx) == []


def test_v7a_cortar_en_trozos_vuelve_colectivo(idx):
    """Como «pica» y «ralla»: «corta el tomate en cubitos» con 3 tomates habla del tomate picado, no de uno."""
    assert _v7a(["3 tomates medianos"], ["Mise en place: lava y corta el tomate en cubitos y la cebolla en dados.",
                                         "Sofríe la cebolla y el tomate 3 minutos."], idx) == []


def test_v7a_el_punto_de_coccion_no_cuenta_piezas(idx):
    """«escalfa 3 huevos … hasta que la clara cuaje» habla de la clara de esos huevos, no de las 6 claras compradas."""
    assert _v7a(["6 claras de huevo"], ["Aparte, pocha 3 huevos en agua hirviendo con un chorrito de vinagre 3 min, "
                                        "hasta que la clara cuaje y la yema quede líquida."], idx) == []
    assert _v7a(["2 filetes de pescado (≈300 g)"],
                ["Incorpora el pescado, tapa y cocina 8-10 minutos, hasta que el centro del filete alcance 63 °C."],
                idx) == []


def test_v7a_cortar_en_piezas_o_porciones_tambien(idx):
    assert _v7a(["2 pechugas de pollo (≈340 g)"], ["Mise en place: corta la pechuga de pollo en piezas medianas."],
                idx) == []
    assert _v7a(["2 filetes de pescado (≈290 g)"], ["Mise en place: corta el filete de pescado blanco en porciones."],
                idx) == []


def test_v7a_el_detalle_nombra_el_singular_encontrado(idx):
    v = cc._v7a_lista_compra_de_mas(1, _meal(["Hierve el huevo 10 min y sírvelo."], ["2 huevos"]), idx)
    assert [x["detail"] for x in v] == ["la lista compra 2 y los pasos hablan de una sola (huevo, siempre en singular)"]


def test_v7a_los_verdaderos_SIGUEN_disparando(idx):
    """Un huevo que se lava, se hierve y se sirve —con 2 en la lista— y las 2 tortillas de las que se rellena una."""
    assert _v7a(["2 huevos"], ["Pela y corta ½ kiwi en cubos y lava el huevo.",
                               "El Toque de Fuego: hierve el huevo en agua durante 10-12 min.",
                               "Montaje: acompaña con el kiwi y el huevo duro pelado y cortado por la mitad."],
                idx) == ["Huevo"]
    assert _v7a(["2 tortillas integrales"], ["Calienta la tortilla integral en una sartén seca 1 min por lado.",
                                             "Montaje: rellena la tortilla integral con el pollo."], idx) == ["Tortilla integral"]
    assert _v7a(["4 claras de huevo"], ["Bate los huevos con la clara y pela la naranja."], idx) == ["Clara de huevo"]
    # cortar en MITADES no es trocear: el huevo duro sigue siendo uno
    assert _v7a(["2 huevos"], ["Hierve el huevo 10 min.", "Corta el huevo duro en mitades."], idx) == ["Huevo"]
    # trocear exige que el corte sea DE ese alimento: el tomate en cubos no vuelve colectiva a la tortilla
    assert _v7a(["2 tortillas integrales"], ["Coloca la tortilla integral y corta el tomate en cubos."],
                idx) == ["Tortilla integral"]
    # la unidad también cuenta: 2 hojas de laurel y un paso que usa «la hoja»
    assert _v7a(["2 hojas de laurel"], ["Cocina la yuca al vapor con la hoja de laurel 20 minutos."], idx) == ["Laurel"]


def test_v7a_la_rama_de_cifras_no_cambia(idx):
    assert _v7a(["1½ ajíes cubanela en rodajas", "2 huevos"], ["Rebana 2 huevos duros."], idx) == []
    assert _v7a(["3 huevos"], ["Bate 2 huevos."], idx) == ["Huevo"]


# ───────────────────────────────────────────── ronda 1 de la revisión: ninguna regla nueva se come un verdadero
# Cada caso de abajo disparaba en la base y la primera versión del lote lo callaba (pruebas de una línea del revisor).

@pytest.mark.parametrize("paso, espera", [
    # el tope de seguridad y el plazo de conservación comparten paso con un marinado de verdad
    ("Marina el pollo con ajo y limón 2 horas en la nevera; no lo dejes a temperatura ambiente más de 2 horas.", 120),
    ("Remoja las lentejas 8 horas (no las dejes a temperatura ambiente más de 2 horas).", 480),
    ("Marina el pollo 3 horas en la nevera y refrigera lo que sobre dentro de 2 horas.", 180),
    # «lo que sobre» sin plazo no es un tope: la masa sí reposa 1 hora
    ("Deja reposar la masa 1 hora tapada y refrigera lo que sobre.", 60),
    # la nota de la dosis en OTRA cláusula del mismo paso no borra el remojo
    ("Remoja las habichuelas 8 horas; si tomas levotiroxina, separa este plato 4 horas de la dosis.", 480),
    # «acuesta los filetes» es cocina; sólo «acostarte» es el consejo de no tumbarse tras cenar
    ("Acuesta los filetes sobre la cebolla y marina 2 horas en la nevera.", 120),
])
def test_v8a_las_exclusiones_nuevas_actuan_sobre_su_frase_no_sobre_el_paso(paso, espera):
    assert cx.hidden_wait_minutes([paso])[0] == espera, paso
    assert cx.check_hidden_time(_meal([paso, "Cocina 10 min."], prep_time="15 min")) is not None


def test_v8a_dentro_de_sin_verbo_de_conservacion_no_es_un_plazo():
    """«cocínalo dentro de 24 horas» no conserva nada: el paso sigue pidiendo 3 h de marinado (como en la base)."""
    assert cx.hidden_wait_minutes(["Marina el pollo 3 horas en la nevera y cocínalo dentro de 24 horas."])[0] >= 180


def test_v8a_la_conservacion_de_antes_del_lote_sigue_mirandose_por_paso_DECISION_PENDIENTE():
    """Decisión PENDIENTE del dueño, no un olvido: `_RE_ALMACEN` («guarda», «consume dentro», «sobras»…) calla el paso
    ENTERO, como en la base. Pasarlo a cláusula destapa 9 + 135 comidas de los dos corpus, todas la nota de la legumbre
    SECA de abajo. Si el dueño decide que ese remojo es tiempo oculto, este test se invierte a propósito."""
    nota = ("💡 Cocción previa: remoja las habichuelas rojas secas 8-12 h y hiérvelas 60-90 min hasta que estén tiernas, "
            "y escúrrelas (puedes cocinar la tanda de varios días y guardarla en la nevera hasta 4 días).")
    assert cx.hidden_wait_minutes([nota])[0] == 0
    # y la conservación sola sigue sin contar
    assert cx.hidden_wait_minutes(["Guarda las sobras en la nevera hasta 3 días."])[0] == 0
    assert cx.hidden_wait_minutes(["Consume dentro de 24 horas."])[0] == 0


@pytest.mark.parametrize("prep_time", ["20 min (sin reposo)", "25 min (no requiere remojo)",
                                       "40 min (guarda en la nevera)", "20 min sin marinar", "30 min (en la nevera)"])
def test_v8a_el_prep_time_que_NIEGA_o_solo_menciona_la_espera_no_la_declara(prep_time):
    m = _meal(["Marina el pollo 4 horas en la nevera.", "Ásalo 8 min."], prep_time=prep_time)
    assert cx.check_hidden_time(m) is not None, prep_time


@pytest.mark.parametrize("prep_time", ["10 min + refrigeración", "10 min (más refrigeración)", "10 min más reposo nocturno",
                                       "10 min más refrigeración", "15 min + toda la noche en remojo",
                                       "15 min y reposo en la nevera"])
def test_v8a_el_prep_time_que_SUMA_la_espera_la_declara(prep_time):
    """Las seis formas del corpus son aditivas («+ refrigeración», «más reposo»)."""
    m = _meal(["Montaje: refrigera la mezcla durante la noche; al servir, coloca encima el mango."], prep_time=prep_time)
    assert cx.check_hidden_time(m) is None, prep_time


def _v2(recipe, idx, ingredients=()):
    return [v["food"] for v in cc._v2_estado_imposible(1, _meal(recipe, ingredients), idx)]


def test_v2_el_dueno_salta_los_condimentos(idx):
    assert _v2(["Sirve la pechuga de pollo con sal y pimienta (ya está cocida)."], idx) == ["Pechuga de pollo"]


def test_v2_el_acompanante_con_con_no_le_quita_el_estado_a_la_cabeza(idx):
    """«la pechuga … con la cebolla (ya viene cocida)»: la cebolla es la más cercana, pero la frase habla de la pechuga."""
    assert _v2(["Añade la pechuga de pollo desmenuzada con la cebolla (ya viene cocida) y mezcla."], idx) == [
        "Pechuga de pollo"]
    # el caso del corpus no se reabre: el atún (listo para comer) es el dueño natural y el huevo de al lado no se acusa
    assert _v2(["Sirve el huevo revuelto con atún en agua (ya viene cocido)."], idx) == []


def test_v2_la_coordinacion_con_y_tambien_sube(idx):
    """«la pechuga de pollo y el filete de pescado blanco (ya viene cocido)»: la base acusaba a los dos crudos y la
    primera versión sólo al pescado (sonda del revisor)."""
    assert _v2(["Incorpora la pechuga de pollo y el filete de pescado blanco (ya viene cocido)."], idx) == [
        "Filete de pescado blanco", "Pechuga de pollo"]
    # un verbo por medio corta la coordinación: «revuelve los huevos y agrega el pescado» no dice nada del huevo
    assert _v2(["Revuelve los huevos y agrega el filete de pescado blanco (ya viene cocido)."], idx) == [
        "Filete de pescado blanco"]


def test_v2_sin_dueno_en_su_oracion_mira_la_anterior(idx):
    assert _v2(["Incorpora el filete de pescado blanco. Ya viene cocido, así que solo caliéntalo 1 minuto."], idx) == [
        "Filete de pescado blanco"]
    # y el sujeto pospuesto sigue siendo el de su oración, aunque la anterior nombre otro alimento
    assert _v2(["Revuelve los huevos 3 min. Ya viene cocido el filete de pescado blanco: sírvelo al lado."], idx) == [
        "Filete de pescado blanco"]


def test_v1_mientras_no_es_una_negacion(idx):
    """«Mientras se hornea el queso de hoja» SÍ hornea el queso de hoja (catálogo: sólo crudo)."""
    assert _v1(["Mientras se hornea el queso de hoja 10 minutos, prepara la ensalada."], idx) == {"Queso de hoja"}
    # el caso del corpus: el sujeto de «mientras se hornean» no está en el catálogo ⇒ no se acusa a los de la frase
    assert _v1(["El Toque de Fuego: hornea las papas 22-25 minutos. Mezcla el queso de hoja con el jugo de limón "
                "mientras se hornean."], idx) == set()


def test_v7a_el_adjetivo_de_tamano_no_es_la_palabra_contada(idx):
    """«2 pequeñas tortillas integrales» cuenta tortillas, no «pequeñas»."""
    for ing in ("2 pequeñas tortillas integrales", "2 medianas tortillas integrales", "2 grandes tortillas integrales"):
        assert _v7a([ing], ["Calienta la tortilla integral 1 min.", "Rellena la tortilla integral con el pollo."],
                    idx) == ["Tortilla integral"], ing


def test_v7a_el_salto_del_tamano_no_se_come_un_nombre():
    """«chic…» como prefijo se comía «chicharrones»: la palabra contada pasaba a ser «de»."""
    i2 = cc.build_culinary_index(_CAT + [{"name": "Chicharrón de cerdo", "prep_methods": ["freir"], "ready_to_eat": True,
                                          "category": "Proteínas", "aliases": ["chicharrones de cerdo"]}])
    contado: dict = {}
    cc._v7_piezas("2 chicharrones de cerdo", i2, contado=contado)
    assert contado == {"Chicharrón de cerdo": "chicharrones"}, contado
    contado = {}
    cc._v7_piezas("2 chiquitas tortillas integrales", i2, contado=contado)
    assert contado == {"Tortilla integral": "tortillas"}, contado


def test_v7a_trocear_en_rodajas_una_pieza_contable_DECISION_PENDIENTE(idx):
    """Decisión de producto PENDIENTE del dueño (revisión del 745, defecto 6): hoy «corta el huevo duro en rodajas» y
    «corta el guineo en rodajas» vuelven colectiva a la pieza, igual que «corta el tomate en cubitos» — y el check calla
    aunque la lista compre 2. «En mitades» sí sigue contando uno (test de arriba). En el corpus, los 74 casos que esta
    regla calla son casi todos tomate, pechuga o filete en masa: el efecto neto es bueno. Si el dueño decide que una
    pieza contable en rodajas sigue siendo UNA, este test se invierte a propósito."""
    assert _v7a(["2 huevos"], ["Hierve el huevo 10 min.", "Corta el huevo duro en rodajas."], idx) == []
    assert _v7a(["2 guineos"], ["Pela y corta el guineo en rodajas.", "Sirve con la avena."], idx) == []


# ───────────────────────────────────────────── anclas

def test_anclas_en_el_codigo():
    src = (_BACKEND / "culinary_coherence.py").read_text(encoding="utf-8")
    ctx = (_BACKEND / "culinary_context.py").read_text(encoding="utf-8")
    for a in ("P1-PLAN-LOTE-745-V1", "P1-PLAN-LOTE-745-V2", "P1-PLAN-LOTE-745-V7A"):
        assert a in src, a
    assert "P1-PLAN-LOTE-745-V8A" in ctx
