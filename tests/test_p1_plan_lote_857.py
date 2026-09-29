# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-857 · 2026-09-29] Dos instrumentos de variedad que no veían lo que había en el plato.

(1) Pescados de los países beta. `_MAIN_PROTEIN_ALIASES['pescado']` (el mapa del contador cross-día, del gate
same-day, de `reeleccion_dia` y del armador determinista) no conocía trucha, sardina, mojarra, bagre,
huachinango, lenguado, caballa ni boquerón, y SÍ contaba «dorado», que en el plato es casi siempre un adjetivo:
G24 CO D1 contaba pescado por «Plátano Maduro Dorado», no por sus sardinas, y la trucha de D1 y D3 no existía.
Las especies salen del vocabulario de pescado del escáner (`_ALLERGEN_SYNONYMS['pescado']`, que ya incluye
`vocabulario_mar.PESCADOS_EXTRA`), no de otra tabla. Knob `MEALFIT_BETA_FISH_SPECIES_COUNT`.

(2) `_count_staple_repetitions` casaba los básicos por subcadena: «pina» dentro de «espinaca» (G24 DO: `pina: 2`
sin una piña). Ahora con el resolvedor del gate same-day (`culinary_context._name_has_token`), y su espejo del
armador determinista (`deterministic_day._basicos_de`) con él. Knob `MEALFIT_STAPLE_TOKEN_MATCH`.
"""
from __future__ import annotations

import copy
import logging
import re
import subprocess
import sys
from pathlib import Path

import pytest

import graph_orchestrator as go

_BACKEND = Path(__file__).resolve().parent.parent


def _plan(textos_por_dia):
    """Un plan de N días, una comida por día cuyo nombre e ingrediente es el texto dado."""
    return [{"day": i + 1, "meals": [{"meal": "Almuerzo", "name": t, "ingredients": [f"120 g de {t}"]}]}
            for i, t in enumerate(textos_por_dia)]


# ── (1) pescados ─────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("especie", ["Trucha", "Sardinas", "Mojarra", "Bagre", "Huachinango", "Lenguado",
                                     "Caballa", "Boquerones", "Filete de dorada"])
def test_las_especies_beta_son_pescado(especie):
    assert "pescado" in go._protein_gate_labels_in_text(f"150 g de {especie}"), especie


def test_trucha_y_sardina_cuentan_entre_dias():
    dias = _plan(["Trucha a la plancha con patacones", "Ensalada estilo ceviche con sardinas",
                  "Trucha al horno con limón"])
    assert go._count_cross_day_heavy_protein_repetition(dias).get("pescado") == 3
    assert go.build_variety_report({"days": dias})["cross_day_proteins"].get("pescado") == 3


def test_dorado_adjetivo_no_es_pescado():
    """G24 CO D1: el «pescado» salía del plátano, no de las sardinas."""
    dias = _plan(["Plátano maduro dorado con queso", "Tostadas doradas con aguacate", "Papas doradas al horno"])
    assert "pescado" not in go._count_cross_day_heavy_protein_repetition(dias)
    assert "pescado" not in go._protein_gate_labels_in_text("Plátano Maduro Dorado")
    assert "pescado" in go._protein_gate_labels_in_text("150 g de filete de dorado"), "la frase inequívoca se queda"


def test_sardinas_en_el_almuerzo_y_trucha_en_la_cena_el_mismo_dia():
    dia = {"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Ensalada fresca con sardinas y garbanzos",
         "ingredients": ["120 g de Sardinas en lata", "80 g de Garbanzos"]},
        {"meal": "Cena", "name": "Arepa asada con ensalada fresca y trucha",
         "ingredients": ["1 Arepa de maíz", "100 g de Trucha"]},
    ]}
    assert go.build_variety_report({"days": [dia]})["same_day_protein_repeats"] >= 1
    assert go._days_with_same_day_protein_repeat({"days": [dia]}) == [1]


@pytest.mark.parametrize("texto", ["Ensalada César con pollo", "1 taza de fumet", "1 cda de salsa inglesa",
                                   "Carpaccio de remolacha", "Arroz con atún"])
def test_lo_que_no_es_un_pez_de_plato_no_cuenta(texto):
    """Del escáner salen sólo las especies: el aderezo César, el fondo, la salsa inglesa y los homónimos
    («carpa» ⊂ «carpaccio») se quedan fuera; el atún conserva su etiqueta."""
    assert "pescado" not in go._protein_gate_labels_in_text(texto), texto


def test_el_armador_determinista_ve_las_mismas_especies():
    import deterministic_day as dd
    assert "pescado" in dd._pesadas_de({"name": "Trucha al horno", "ingredients": ["150 g de Trucha"]})
    assert "pescado" not in dd._pesadas_de({"name": "Plátano maduro dorado", "ingredients": ["1 Plátano maduro"]})


def test_knob_de_especies_apagado_deja_el_mapa_como_estaba(monkeypatch):
    import pescado_especies as pe
    base = {"pescado": ["pescado", "tilapia", "dorado"], "atun": ["atun"], "pollo": ["pollo"]}
    vocab = ["pescado", "atun", "trucha", "sardina", "dorado", "cesar"]
    monkeypatch.setattr(pe, "BETA_FISH_SPECIES_COUNT", False)
    apagado = copy.deepcopy(base)
    assert pe.extender_pescado(apagado, vocab) == ([], [])
    assert apagado == base
    monkeypatch.setattr(pe, "BETA_FISH_SPECIES_COUNT", True)
    encendido = copy.deepcopy(base)
    anadidos, quitados = pe.extender_pescado(encendido, vocab)
    assert anadidos == ["trucha", "sardina", "filete de dorada"] and quitados == ["dorado"]
    assert encendido["pescado"] == ["pescado", "tilapia", "trucha", "sardina", "filete de dorada"]
    assert encendido["atun"] == ["atun"], "el atún no se muda de etiqueta"


# ── (2) básicos entre días ───────────────────────────────────────────────────────────────────────────────


def test_espinaca_no_es_pina():
    dias = _plan(["Ensalada de espinacas con uva", "Revoltillo con espinaca"])
    assert "pina" not in go._count_staple_repetitions(dias)


def test_pina_si_cuenta():
    dias = _plan(["Piña picada con yogurt", "Batido de piña"])
    assert go._count_staple_repetitions(dias).get("pina") == 2


def test_espejo_del_armador_determinista():
    import deterministic_day as dd
    assert "pina" not in dd._basicos_de({"name": "Ensalada de espinacas", "ingredients": ["2 tazas de Espinacas"]})
    assert "pina" in dd._basicos_de({"name": "Piña con yogurt", "ingredients": ["1 taza de Piña"]})


def test_knob_de_basicos_apagado_vuelve_a_la_subcadena(monkeypatch):
    import basicos_por_token as bpt
    dias = _plan(["Ensalada de espinacas con uva", "Revoltillo con espinaca"])
    monkeypatch.setattr(bpt, "STAPLE_TOKEN_MATCH", False)
    assert go._count_staple_repetitions(dias).get("pina") == 2      # la conducta previa, medida
    monkeypatch.setattr(bpt, "STAPLE_TOKEN_MATCH", True)
    assert "pina" not in go._count_staple_repetitions(dias)


# ── ronda 7 · el día con especie nueva va entero al gate; se reescribe sólo el pez que la receta cuece ─────────────
#
# Ronda 6: las especies nuevas cuentan para DETECTAR, pero el autofix no las reescribe, ni reescribe un pez listo, crudo
# o frío. La revisión r6 encontró dos huecos. (1) Con la especie nueva DELANTE, ella se quedaba de guardiana y el autofix
# reescribía el otro pez, uno de la base, en una repetición que la base no veía (d8b10b05 D2, «Espaguetis con sardinas»:
# la tilapia pasaba a pollo). (2) El filtro de crudo/frío dejaba pasar pollo crudo en listas («Marina la cebolla, el ají
# y el mero»), participios («se marina»), coordinadas, «leche de tigre», «sírvelo helado» y precocidos. Ronda 7: (1) el
# día cuya repetición de pescado incluye una especie nueva se salta ENTERO; (2) la regla positiva de V7f: un pez se
# reescribe sólo si una cláusula que lo nombra (o su enclítico) lo cuece — el blanqueo no cuenta — y «ya cocido»,
# «precocido» o «cocidos» junto al pez es precocido. La guarda de conservas sigue.


def _tilapia():
    return {"meal": "Almuerzo", "name": "Tilapia al horno con arroz",
            "ingredients": ["150 g de Filete de tilapia", "1 taza de Arroz"],
            "recipe": ["Hornea la tilapia 20 minutos a 200 °C."]}


def _comida(nombre, lineas, pasos, slot="Cena"):
    return {"meal": slot, "name": nombre, "ingredients": list(lineas), "ingredients_raw": list(lineas),
            "recipe": list(pasos)}


def _dia_dos_pescados(nombre, linea, pasos):
    return [{"day": 1, "meals": [_tilapia(), _comida(nombre, [linea, "1/2 Aguacate"], pasos)]}]


def _textos(meal):
    return [meal["name"], *meal["ingredients"], *(meal.get("ingredients_raw") or []), *(meal.get("recipe") or [])]


def _motivo(etiqueta, comida, ligero=False):
    import pescado_especies as pe
    return pe.motivo_para_no_reescribir(etiqueta, comida, go._MAIN_PROTEIN_ALIASES.get(etiqueta, ()),
                                        go._PRECOOKED_PROTEIN_HINT, go._diet_pool_item_banned, ligero)


_BLANQUEO = ("Blanquea {a} en agua hirviendo 1-2 minutos y escúrrelo bien antes de marinar (el cítrico solo marina, "
             "no cuece). Marina {a} en jugo de limón 20 minutos.")

# Las formas del revisor r5, con especie NUEVA: el día no se toca.
_ESPECIES_NUEVAS = {
    "sardinas-lata-en-el-paso": _comida("Arroz con sardinas", ["90 g de Sardinas", "1 taza de Arroz"],
                                        ["Escurre las sardinas de la lata y mézclalas con el arroz caliente."]),
    "sardinas-abre-la-lata": _comida("Casabe con sardinas", ["90 g de Sardinas", "1 Casabe"],
                                     ["Abre la lata y escurre las sardinas.", "Colócalas sobre el casabe."]),
    "sardinas-ya-viene-cocido": _comida("Arroz con vegetales y sardinas", ["1 taza de Arroz", "90 g de Sardinas"],
                                        ["Cocina el arroz 18 minutos.",
                                         "Escurre e incorpora sardinas (ya viene cocido) a la preparación."]),
    "lata-pequena-de-sardinas": _comida("Ensalada de sardinas", ["1 lata pequeña de sardinas", "1 tomate"],
                                        ["Desmenuza las sardinas y mézclalas con el tomate."]),
    "caballa-de-su-lata": _comida("Casabe con caballa", ["90 g de Caballa", "1 Casabe"],
                                  ["Escurre la caballa de su lata y colócala sobre el casabe."]),
    "ceviche-de-trucha-blanqueado": _comida("Ceviche de trucha", ["150 g de Trucha", "1 limón"],
                                            [_BLANQUEO.format(a="la trucha"), "Sirve frío."]),
    "trucha-cocida": _comida("Trucha a la plancha con papas", ["150 g de Trucha", "200 g de Papa"],
                             ["Cocina la trucha a la plancha 4 minutos por lado.", "Hierve las papas 15 minutos."]),
    "bagre-guisado": _comida("Bagre guisado con arroz", ["150 g de Bagre", "1 taza de Arroz"],
                             ["Guisa el bagre 20 minutos en salsa de tomate."]),
}


@pytest.mark.parametrize("tilapia_primero", [True, False], ids=["tilapia-primero", "especie-nueva-primero"])
@pytest.mark.parametrize("clave", list(_ESPECIES_NUEVAS))
def test_el_dia_con_especie_nueva_no_se_toca(clave, tilapia_primero):
    """Ni la comida de la especie nueva ni la tilapia de la base: el día entero va al gate."""
    otra = copy.deepcopy(_ESPECIES_NUEVAS[clave])
    dias = [{"day": 1, "meals": [_tilapia(), otra] if tilapia_primero else [otra, _tilapia()]}]
    antes = copy.deepcopy(dias)
    assert go._days_with_same_day_protein_repeat({"days": dias}) == [1], "la repetición se detecta"
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced", "country": "ES"}, None) == 0
    assert dias == antes


def test_la_especie_nueva_va_al_gate_con_su_motivo(caplog):
    dias = [{"day": 1, "meals": [_tilapia(), copy.deepcopy(_ESPECIES_NUEVAS["sardinas-lata-en-el-paso"])]}]
    antes = copy.deepcopy(dias)
    with caplog.at_level(logging.INFO):
        assert go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None) == 0
    assert dias == antes
    assert "reason=especie_nueva" in caplog.text
    assert go._days_with_same_day_protein_repeat({"days": dias}) == [1]


# La sonda de la revisión r6 (probe_nuevo.py): sardinas o trucha DELANTE y un pez de la base detrás. La base no veía la
# repetición y no tocaba nada; la ronda 6 sacaba pollo crudo en los 8 días.
_GUARDIANAS_NUEVAS = {
    "sardinas": _comida("Ensalada de sardinas con aguacate", ["90 g de Sardinas en lata", "1/2 Aguacate"],
                        ["Escurre las sardinas en lata y mézclalas con el aguacate."], slot="Almuerzo"),
    "trucha": _comida("Trucha a la plancha con papas", ["150 g de Trucha"],
                      ["Cocina la trucha a la plancha 4 minutos por lado."], slot="Almuerzo"),
}
_PECES_DE_LA_BASE_R6 = {
    "mero-lista-marinada": _comida("Mero al limón con cebolla", ["150 g de Filete de mero", "1 cebolla"],
                                   ["Corta el mero en cubos pequeños.",
                                    "Marina la cebolla, el ají y el mero en jugo de limón 30 minutos.", "Sirve con casabe."]),
    "cd1b2fd0-D3": _comida("Pescado blanco guisado en salsa de tomate", ["1 filete de pescado", "1/2 tomate"],
                           ["Mise en place: escurre bien filete de pescado blanco (240 g); pica 1 tomate.",
                            "El Toque de Fuego: sofríe el tomate 6-8 min; añade filete de pescado blanco, mezcla "
                            "suavemente y cocina 2-3 min más."]),
    "tilapia-se-marina": _comida("Tilapia al limón", ["150 g de Filete de tilapia"],
                                 ["La tilapia, cortada en cubos pequeños, se marina en limón 20 minutos.",
                                  "Sirve con aguacate."]),
    "sirve-helado": _comida("Ensalada de pescado", ["120 g de Filete de pescado"],
                            ["Desmenuza el pescado y mézclalo con la cebolla y el limón.",
                             "Refrigera 30 minutos y sírvelo helado."]),
}


@pytest.mark.parametrize("otra", list(_PECES_DE_LA_BASE_R6))
@pytest.mark.parametrize("guardiana", list(_GUARDIANAS_NUEVAS))
def test_la_especie_nueva_delante_no_deja_reescribir_el_pez_de_la_base(guardiana, otra):
    dias = [{"day": 1, "meals": [copy.deepcopy(_GUARDIANAS_NUEVAS[guardiana]), copy.deepcopy(_PECES_DE_LA_BASE_R6[otra])]}]
    antes = copy.deepcopy(dias)
    assert go._days_with_same_day_protein_repeat({"days": dias}) == [1]
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None) == 0
    assert dias == antes


def test_se_salta_el_dia_entero_no_solo_el_pescado():
    """El pollo repetido del mismo día tampoco se toca: el gate regenera el día y la base no veía este día como el de la
    rama (sin las sardinas no había repetición de pescado)."""
    dias = [{"day": 1, "meals": [
        copy.deepcopy(_GUARDIANAS_NUEVAS["sardinas"]),
        _comida("Tilapia al horno", ["150 g de Filete de tilapia"], ["Hornea la tilapia 20 minutos."]),
        _comida("Pollo guisado con arroz", ["150 g de Pechuga de pollo"], ["Guisa el pollo 25 minutos."], slot="Desayuno"),
        _comida("Wrap de pollo", ["100 g de Pechuga de pollo"], ["Cocina el pollo a la plancha 6 minutos por lado."],
                slot="Merienda"),
    ]}]
    antes = copy.deepcopy(dias)
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None) == 0
    assert dias == antes


@pytest.mark.parametrize("sardinas_primero", [True, False], ids=["orden-real", "trucha-primero"])
def test_g24_co_sardinas_y_trucha_van_al_gate(sardinas_primero):
    """G24 CO D1: las dos son especies nuevas; la ronda 5 reescribía la trucha a pollo. Ahora ninguna se toca."""
    sard = {"meal": "Almuerzo", "name": "Ensalada Fresca Estilo Ceviche con Sardinas, Garbanzos y Plátano Maduro Dorado",
            "ingredients": ["110 g de sardinas en lata escurridas",
                            "½ taza de garbanzos cocidos (de lata, enjuagados y escurridos)", "½ plátano maduro"],
            "recipe": ["Mise en place: escurre bien los 110 g de sardinas en lata y desmenúzalas.",
                       "Montaje: marina las sardinas 5 minutos con el jugo de limón y sirve."]}
    tru = {"meal": "Cena", "name": "Arepas de Maíz Asadas Rellenas de Ricotta con Ensalada Fresca y Trucha",
           "ingredients": ["35 g de harina de maíz precocida", "180 g de queso ricotta", "50 g de trucha cocida"],
           "recipe": ["El Toque de Fuego: forma 2 arepas y ásalas a fuego medio 5-6 minutos por lado. Cocina trucha "
                      "a la plancha o hervida y sírvela como proteína del plato."]}
    dias = [{"day": 1, "meals": [sard, tru] if sardinas_primero else [tru, sard]}]
    antes = copy.deepcopy(dias)
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced", "country": "CO"}, None) == 0
    assert dias == antes


def test_el_reescritor_usa_los_alias_de_antes_del_lote():
    """Revisión r5 (no bloquea): «emplatado bonito» salía «emplatado pechuga de pollo». El reescritor no conoce las
    especies nuevas: la tilapia pasa a pollo y el adjetivo se queda."""
    dias = [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Mero a la plancha", "ingredients": ["150 g de Filete de mero"],
         "recipe": ["Cocina el mero a la plancha 4 minutos por lado."]},
        {"meal": "Cena", "name": "Tilapia al horno con papas", "ingredients": ["150 g de Filete de tilapia"],
         "recipe": ["Hornea la tilapia 20 minutos.", "Corta las papas para un emplatado bonito y sirve."]},
    ]}]
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None) == 1
    cena = dias[0]["meals"][1]
    assert cena["_protein_autofix_applied"] == "pescado->pollo"
    assert cena["recipe"] == ["Hornea pechuga de pollo 20 minutos.", "Corta las papas para un emplatado bonito y sirve."]
    assert {"trucha", "sardina", "bonito", "caballa", "anchoa"} <= set(go._PEZ_NUEVO)
    assert not {"tilapia", "mero", "corvina", "pescado", "bacalao"} & set(go._PEZ_NUEVO), "la base no es nueva"


# (2) La regla positiva, sobre la base de antes del lote (atún, corvina, mero, camarones…): formas de las rondas 5 y 6.
_PEZ_LISTO_CRUDO_FRIO = {
    # la base: «Ensalada de atún… sirve frío» salía «Escurre pechuga de pollo y mézclala… Sirve frío»
    "atun-ensalada-sirve-frio": ("atun", _comida("Ensalada de atún", ["120 g de Atún", "1 taza de Lechuga"],
                                                 ["Escurre el atún y mézclalo con la lechuga.", "Sirve frío."])),
    "atun-en-agua": ("atun", _comida("Wrap de atún", ["120 g de atún en agua", "1 tortilla"],
                                     ["Rellena la tortilla con el atún."])),
    "lata-de-atun": ("atun", _comida("Arroz con atún", ["1 lata de atún (120 g)", "1 taza de Arroz"],
                                     ["Mezcla el atún con el arroz caliente."])),
    "atun-en-conserva": ("atun", _comida("Pasta con atún", ["120 g de Atún en conserva", "1 taza de Pasta"],
                                         ["Incorpora el atún a la pasta."])),
    "ceviche-de-corvina-blanqueado": ("pescado", _comida("Ceviche de corvina", ["150 g de Corvina", "1 limón"],
                                                         [_BLANQUEO.format(a="la corvina"), "Sirve frío."])),
    "tiradito-de-corvina": ("pescado", _comida("Tiradito de corvina", ["120 g de Corvina"],
                                               ["Corta la corvina en láminas finas y báñala con leche de tigre."])),
    "mero-marinado-sin-fuego": ("pescado", _comida("Mero marinado al limón", ["150 g de Filete de mero marinado"],
                                                   ["Marina el mero en jugo de limón 30 minutos y sirve."])),
    "salmon-ahumado": ("pescado", _comida("Tosta de salmón ahumado", ["80 g de Salmón ahumado", "1 pan"],
                                          ["Coloca el salmón ahumado sobre la tosta."])),
    "pescado-ya-viene-cocido": ("pescado", _comida("Arroz con pescado", ["120 g de Filete de pescado", "1 taza de Arroz"],
                                                   ["Escurre e incorpora filete de pescado (ya viene cocido) al arroz."])),
    "coctel-de-camarones-frio": ("camarones", _comida("Cóctel de camarones", ["120 g de Camarones cocidos"],
                                                      ["Mezcla los camarones con la salsa rosada y sirve frío."])),
}
# Las sondas de la revisión r6 (probe6.py), todas con un pez de la base. Ninguna se reescribe.
_SONDAS_R6 = {
    "atun-sandwich-sin-lata": ("atun", _comida("Sándwich de atún", ["90 g de Atún", "2 rebanadas de pan integral"], [
        "Escurre el atún y mézclalo con mayonesa ligera y cebolla picada.", "Unta la mezcla en el pan y sirve."])),
    "atun-al-natural": ("atun", _comida("Ensalada de atún con tomate", ["100 g de atún al natural", "1 tomate"],
                                        ["Desmenuza el atún al natural y mézclalo con el tomate y el aguacate."])),
    "atun-en-salmuera": ("atun", _comida("Wrap de atún", ["100 g de atún en salmuera", "1 tortilla"],
                                         ["Escurre el atún en salmuera y rellena la tortilla."])),
    "salmon-curado": ("pescado", _comida("Tostada con salmón curado", ["60 g de Salmón curado", "1 tostada"],
                                         ["Coloca el salmón curado sobre la tostada con queso crema."])),
    "camarones-cocidos-ensalada": ("camarones", _comida(
        "Ensalada de camarones con aguacate", ["120 g de Camarones cocidos", "1/2 Aguacate"],
        ["Pela los camarones cocidos y mézclalos con el aguacate y el limón.", "Sirve en copas."])),
    "camarones-ya-cocidos-calienta": ("camarones", _comida("Arroz con camarones", ["120 g de Camarones", "1 taza de Arroz"], [
        "Cocina el arroz 18 minutos.", "Agrega los camarones ya cocidos y calienta 1 minuto."])),
    "camarones-precocidos": ("camarones", _comida("Pasta con camarones", ["120 g de Camarones precocidos", "1 taza de Pasta"], [
        "Hierve la pasta 10 minutos.", "Incorpora los camarones precocidos a la pasta y mezcla."])),
    "pescado-ya-cocido-participio": ("pescado", _comida(
        "Arroz con pescado desmenuzado", ["120 g de Filete de pescado", "1 taza de Arroz"],
        ["Cocina el arroz 18 minutos.", "Incorpora el pescado ya cocido y desmenuzado al arroz."])),
    "pulpo-cocido-ensalada": ("pulpo", _comida("Ensalada de pulpo", ["100 g de Pulpo cocido", "1 papa"], [
        "Corta el pulpo en rodajas y alíñalo con aceite y pimentón.", "Hierve la papa 15 minutos."])),
    "cangrejo-ensalada": ("cangrejo", _comida("Ensalada de cangrejo", ["100 g de Carne de cangrejo", "1 taza de lechuga"],
                                              ["Desmenuza el cangrejo y mézclalo con la lechuga y mayonesa."])),
    "mero-lista-marinada": ("pescado", _PECES_DE_LA_BASE_R6["mero-lista-marinada"]),
    "tilapia-y-camarones-marinados": ("pescado", _comida(
        "Tilapia y camarones marinados al limón", ["120 g de Filete de tilapia", "60 g de Camarones"],
        ["Corta la tilapia en cubos y marínala con los camarones en jugo de limón 30 minutos.",
         "Sirve con cebolla morada."])),
    "corvina-leche-de-tigre": ("pescado", _comida("Corvina en leche de tigre", ["150 g de Corvina", "2 limones"], [
        "Corta la corvina en cubos y cúbrela con la leche de tigre 15 minutos.", "Sirve con camote."])),
    "tilapia-se-marina": ("pescado", _PECES_DE_LA_BASE_R6["tilapia-se-marina"]),
    "corvina-banada": ("pescado", _comida("Corvina al limón con ají", ["150 g de Corvina"], [
        "Corta la corvina en láminas y báñala en jugo de limón con ají 20 minutos.", "Sirve."])),
    "cebiche-mixto-coord": ("pescado", _comida("Mixto de mero y pulpo al limón", ["100 g de Filete de mero",
                                                                                   "50 g de Pulpo cocido"], [
        "Pica el mero y el pulpo, y déjalos marinando en limón 25 minutos.", "Sirve con cebolla."])),
    "tataki-atun": ("atun", _comida("Tataki de atún", ["150 g de Atún fresco"], [
        "Sella el atún 30 segundos por lado en sartén muy caliente.", "Corta en láminas y sirve."])),
    "sirve-helado": ("pescado", _PECES_DE_LA_BASE_R6["sirve-helado"]),
    "enfria-y-sirve": ("pescado", _comida("Ensalada de mero", ["120 g de Filete de mero"], [
        "Mezcla el mero con cebolla y limón.", "Enfría en la nevera antes de servir."])),
    "sirvela-bien-fria-coordinada": ("pescado", _comida("Ensalada de tilapia", ["120 g de Filete de tilapia"], [
        "Mezcla la tilapia con cebolla y sírvela, bien fría, con galletas."])),
    "lata-lista": ("atun", _comida("Ensalada de atún y huevo", ["120 g de Atún", "1 huevo"], [
        "Abre la lata, escurre el líquido y mezcla el atún con el huevo duro."])),
    "lata-en-otra-frase-atun": ("atun", _comida("Arroz con atún", ["120 g de Atún", "1 taza de Arroz"], [
        "Abre y escurre la lata.", "Mezcla el atún con el arroz caliente."])),
    "conserva-plural": ("pescado", _comida("Ensalada con filetes de merluza en conserva", ["120 g de Merluza en conserva"],
                                           ["Mezcla la merluza con tomate."])),
    "en-aceite-bacalao": ("pescado", _comida("Tostada con bacalao en aceite", ["80 g de Bacalao en aceite"],
                                             ["Coloca el bacalao sobre la tostada."])),
}
# Ronda 8: las sondas de la revisión r7 (probe7.py) que la ronda 7 aún reescribía, con el motivo de cada veto.
_SONDAS_R7 = {
    # (1) «cocido» en la LÍNEA sin la firma del cerrador en los pasos: el paso sólo calienta
    "linea-camarones-cocidos-saltea-1min": ("camarones", _comida("Arroz con camarones", [
        "120 g de Camarones cocidos", "1 taza de Arroz"], ["Cocina el arroz 18 minutos.",
                                                           "Saltea los camarones con ajo 1 minuto, sólo para calentarlos, "
                                                           "y mézclalos con el arroz."])),
    "linea-pescado-cocido-sarten-1min": ("pescado", _comida("Arroz con pescado", [
        "120 g de Filete de pescado cocido", "1 taza de Arroz"],
        ["Calienta el pescado en la sartén 1 minuto y mézclalo con el arroz."])),
    "linea-pulpo-cocido-plancha-1min": ("pulpo", _comida("Pulpo a la plancha", ["120 g de Pulpo cocido", "1 papa"], [
        "Pasa el pulpo por la plancha 1 minuto por lado.", "Hierve la papa 15 minutos."])),
    # (2) «crudo»/«en crudo» hasta dos palabras después del pez, en un paso o en la línea
    "arroz-y-mero-crudo": ("pescado", _comida("Arroz con mero en láminas", ["150 g de Filete de mero", "1 taza de Arroz"], [
        "Cocina el arroz 18 minutos y sírvelo con el mero crudo en láminas finas y limón."])),
    "atun-en-crudo-en-la-linea": ("atun", _comida("Arroz con atún y aguacate", ["120 g de Atún fresco en crudo",
                                                                               "1 taza de Arroz"], [
        "Cocina el arroz 18 minutos y sírvelo con el atún y el aguacate."])),
    # (3) el plato crudo nombrado en un PASO, no en el nombre
    "sebiche-en-pasos": ("pescado", _comida("Pescado al limón estilo peruano", ["150 g de Corvina", "1 camote"], [
        "Prepara el cebiche: corta la corvina en cubos, cúbrela con limón 15 minutos y hierve el camote 20 minutos.",
        "Sirve."])),
    # (4) el pez ya listo en un paso: ahumado, curado, sobrante, ya horneado/asado/hervido/frito/cocinado
    "ahumado-solo-en-paso": ("pescado", _comida("Tostada de salmón", ["60 g de Salmón", "1 pan"], [
        "Tuesta el pan en la sartén 2 minutos y coloca encima el salmón ahumado con queso crema."])),
    "ya-horneado-microondas": ("pescado", _comida("Arroz con pescado", ["120 g de Filete de pescado", "1 taza de Arroz"], [
        "Calienta el pescado ya horneado en el microondas 1 minuto y sírvelo con el arroz."])),
    "sobrante-sarten": ("pescado", _comida("Arroz con tilapia", ["120 g de Filete de tilapia", "1 taza de Arroz"], [
        "Usa la tilapia sobrante de la cena, desmenúzala y caliéntala en la sartén 2 minutos con el arroz."])),
    # (5) la cocción de OTRO alimento después de «mientras»
    "coma-marina-y-sofrie": ("pescado", _comida("Mero al limón con cebolla", ["150 g de Filete de mero", "1 cebolla"], [
        "Corta el mero en cubos, marínalo en jugo de limón 20 minutos y, mientras tanto, sofríe la cebolla 5 minutos.",
        "Sirve el mero con la cebolla encima."])),
    "marina-mientras-hierve": ("pescado", _comida("Tilapia al limón con arroz", ["150 g de Filete de tilapia",
                                                                                 "1 taza de Arroz"], [
        "Marina la tilapia en limón 20 minutos mientras hierve el arroz.", "Sirve la tilapia sobre el arroz."])),
    "tartar-pan-horno": ("pescado", _comida("Mero picado con aguacate y pan", ["150 g de Filete de mero", "1 aguacate",
                                                                               "2 rebanadas de pan"], [
        "Pica el mero finamente y mézclalo con el aguacate y el limón mientras el pan se tuesta en el horno 5 minutos.",
        "Sirve sobre el pan."])),
}
_MOTIVOS_R7 = {
    "linea-camarones-cocidos-saltea-1min": "pez_precocido", "linea-pescado-cocido-sarten-1min": "pez_precocido",
    "linea-pulpo-cocido-plancha-1min": "pez_precocido", "arroz-y-mero-crudo": "pez_crudo",
    "atun-en-crudo-en-la-linea": "pez_crudo", "sebiche-en-pasos": "pez_crudo", "ahumado-solo-en-paso": "pez_en_conserva",
    "ya-horneado-microondas": "pez_precocido", "sobrante-sarten": "pez_precocido", "coma-marina-y-sofrie": "pez_sin_coccion",
    "marina-mientras-hierve": "pez_sin_coccion", "tartar-pan-horno": "pez_sin_coccion",
}
# Lo que la ronda 8 deja pasar A SABIENDAS (medido en probe7.py): la regla positiva de V7f ve una cocción en la cláusula
# del pez, pero no es la del pez, o no alcanza para un ave. Si alguno deja de reescribirse, pasa a `_SONDAS_R7` y sale
# del docstring de `test_nunca_sale_carne_sin_coccion` y de docs/knobs_reference.md.
_RESIDUOS = {
    # otro alimento se cuece en la MISMA frase que el pez, sin «mientras»
    "sandwich-tuesta-el-pan": ("atun", _comida("Sándwich de atún", ["90 g de Atún", "2 rebanadas de pan"], [
        "Escurre el atún, mézclalo con la mayonesa y la cebolla y tuesta el pan en la sartén 2 minutos.",
        "Arma el sándwich y sirve."])),
    # tiempos que cuecen el pez y no el ave que lo sustituye
    "sellado-30-s": ("pescado", _comida("Salmón sellado con ensalada", ["150 g de Salmón"], [
        "Sella el salmón 30 segundos por lado en sartén muy caliente; córtalo en láminas.", "Sirve con la ensalada."])),
    "caldo-1-minuto": ("pescado", _comida("Sopa de mero", ["120 g de Filete de mero", "2 tazas de caldo"], [
        "Corta el mero en láminas muy finas y colócalas en el tazón.",
        "Vierte el caldo caliente de la olla sobre el mero y deja reposar 1 minuto."])),
    "cocina-1-minuto-mas": ("pescado", _comida("Arroz con pescado", ["120 g de Filete de pescado", "1 taza de Arroz"], [
        "Agrega el pescado hervido y desmenuzado al arroz y cocina 1 minuto más."])),
    "escabeche": ("pescado", _comida("Pescado en escabeche", ["150 g de Filete de pescado", "1 cebolla"], [
        "Fríe el pescado 3 minutos por lado; cúbrelo con el escabeche y refrigera 12 horas.", "Sirve frío."])),
    # dos peces en la comida: la cocción de uno se lleva al otro
    "dos-peces": ("pescado", _comida("Salmón al horno con ensalada de mero marinado", ["120 g de Salmón",
                                                                                     "60 g de Filete de mero"], [
        "Hornea el salmón 15 minutos.", "Marina el mero en limón 20 minutos y sírvelo al lado."])),
}
_COCINADOS = {
    "pescado": _tilapia(),
    "atun": _comida("Atún guisado con arroz", ["150 g de Atún fresco", "1 taza de Arroz"],
                    ["Guisa el atún 15 minutos en salsa de tomate."], slot="Almuerzo"),
    "camarones": _comida("Camarones al ajillo", ["150 g de Camarones", "1 taza de Arroz"],
                         ["Saltea los camarones con ajo 5 minutos."], slot="Almuerzo"),
    "pulpo": _comida("Pulpo guisado", ["150 g de Pulpo", "1 taza de Arroz"],
                     ["Guisa el pulpo 40 minutos en salsa de tomate."], slot="Almuerzo"),
    "cangrejo": _comida("Cangrejo guisado", ["150 g de Cangrejo", "1 taza de Arroz"],
                        ["Guisa el cangrejo 20 minutos."], slot="Almuerzo"),
}
_NO_SE_REESCRIBEN = {**_PEZ_LISTO_CRUDO_FRIO, **_SONDAS_R6, **_SONDAS_R7}


@pytest.mark.parametrize("clave", list(_SONDAS_R7))
def test_ronda_8_cada_veto_con_su_motivo(clave):
    etiqueta, comida = _SONDAS_R7[clave]
    assert _motivo(etiqueta, comida) == _MOTIVOS_R7[clave]


@pytest.mark.xfail(strict=True, reason="residuo documentado del lote 857: la regla positiva de V7f lo da por cocido")
@pytest.mark.parametrize("clave", list(_RESIDUOS))
def test_residuos_conocidos(clave):
    """Estas comidas SIGUEN reescribiéndose (xfail estricto): si una deja de hacerlo, el test avisa para sacarla de la
    lista de residuos de los docs."""
    etiqueta, comida = _RESIDUOS[clave]
    assert _motivo(etiqueta, comida) is not None


@pytest.mark.parametrize("cocinado_primero", [True, False], ids=["cocinado-primero", "sonda-primero"])
@pytest.mark.parametrize("clave", list(_NO_SE_REESCRIBEN))
def test_nunca_se_reescribe_un_pez_que_la_receta_no_cuece(clave, cocinado_primero):
    etiqueta, comida = _NO_SE_REESCRIBEN[clave]
    comida = copy.deepcopy(comida)
    antes = copy.deepcopy(comida)
    otro = copy.deepcopy(_COCINADOS[etiqueta])
    dias = [{"day": 1, "meals": [otro, comida] if cocinado_primero else [comida, otro]}]
    assert go._days_with_same_day_protein_repeat({"days": dias}) == [1]
    go._protein_repeat_autofix(dias, {"dietType": "balanced", "country": "DO"}, None)
    assert comida == antes, "el pez que la receta no cuece no se reescribe"


_REESCRIBIBLES = {
    "mero-plancha-control": ("pescado", _comida("Mero a la plancha", ["150 g de Filete de mero"],
                                                ["Cocina el mero a la plancha 4 minutos por lado."])),
    "camarones-salteados-control": ("camarones", _comida("Camarones salteados con vegetales",
                                                         ["150 g de Camarones", "1 taza de brócoli"],
                                                         ["Saltea los camarones con el brócoli 5 minutos."])),
    # cocido y servido frío: el pollo también se cuece (la ronda 6 lo dejaba al gate por «fría»)
    "ensalada-fria-de-merluza": ("pescado", _comida("Ensalada fría de merluza", ["120 g de Merluza", "1 tomate"],
                                                    ["Cocina la merluza al vapor 8 minutos.", "Mezcla con el tomate."])),
    "marina-y-hornea": ("pescado", _comida("Tilapia con arroz", ["150 g de Filete de tilapia marinada"],
                                           ["Marina la tilapia con limón 10 minutos; luego hornéala 20 minutos a 200 °C."])),
    # ronda 8: cortar en «mientras» no quita la cocción del pez que va ANTES, ni el enclítico de la frase siguiente
    "hornea-mientras-hierve": ("pescado", _comida("Tilapia al horno con arroz", ["150 g de Filete de tilapia"], [
        "Hornea la tilapia 20 minutos a 200 °C mientras hierve el arroz."])),
    "marina-mientras-y-hornea-despues": ("pescado", _comida("Tilapia al limón al horno", ["150 g de Filete de tilapia"], [
        "Marina la tilapia en limón 10 minutos mientras se calienta el horno; luego hornéala 20 minutos a 200 °C."])),
    # ronda 8: «cocido» en la línea CON la firma del cerrador en el paso es el peso cocido (1461aeca D3, 92328ff7 D9)
    "linea-cocido-con-la-firma-del-cerrador": ("pescado", _comida("Arroz con pescado", [
        "120 g de pescado cocido", "1 taza de Arroz"], [
        "Cocina el arroz 18 minutos.", "Cocina pescado a la plancha o hervido y sírvelo como proteína del plato."])),
}


@pytest.mark.parametrize("clave", list(_REESCRIBIBLES))
def test_el_pez_que_la_receta_cuece_si_se_reescribe(clave):
    etiqueta, comida = _REESCRIBIBLES[clave]
    dias = [{"day": 1, "meals": [copy.deepcopy(_COCINADOS[etiqueta]), copy.deepcopy(comida)]}]
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced", "country": "DO"}, None) == 1
    assert dias[0]["meals"][1].get("_protein_autofix_applied", "").startswith(etiqueta + "->")


_FORMAS_PELIGROSAS = list(_ESPECIES_NUEVAS.values()) + [c for _, c in _NO_SE_REESCRIBEN.values()]
_ALERTA_RE = re.compile(r"\blatas?\b|enlatad|ya\s+(?:viene\s+)?cocid|precocid|\bfr[ií][oa]s?\b|\bmarin|ceviche|tiradito"
                        r"|tataki|helad|enfr[ií]|leche de tigre|b[aá][nñ]ala|curad|al natural|salmuera")


@pytest.mark.parametrize("i", range(len(_FORMAS_PELIGROSAS)))
def test_nunca_sale_carne_sin_coccion(i):
    """Sobre las formas de ESTE fichero (las de las revisiones r5, r6 y r7), en los dos órdenes y junto a cada plato
    cocinado: no sale una carne en una comida que habla de lata, marinado, curado, frío o precocido, y toda carne que el
    autofix escribe la cuece una cláusula que la nombra.

    No es un «nunca» universal. La regla es la de V7f (un verbo de cocción, o fuego y tiempo, en la cláusula del pez), y
    deja pasar a sabiendas los residuos de `_RESIDUOS` (xfail estricto en `test_residuos_conocidos`):
      - otro alimento cocido en la misma frase que el pez, sin «mientras» («Escurre el atún, mézclalo con la mayonesa y
        tuesta el pan en la sartén 2 minutos»: el sándwich sale con pollo sin cocer);
      - tiempos que cuecen el pez y no el ave que lo sustituye: el sellado de 30 s, el caldo que reposa 1 minuto,
        «cocina 1 minuto más» sobre el pescado ya hervido, el escabeche (frito 3 minutos por lado y servido frío);
      - una comida con dos peces, donde la cocción de uno cuenta para el otro («Hornea el salmón… Marina el mero…»).
    El tiempo mínimo del ave que hereda los tiempos del pez no es de este lote: es de la sesión de PLATO
    (`pasos_cantidades.ave_a_74` sólo sube los °C)."""
    import pescado_especies as pe
    comida = _FORMAS_PELIGROSAS[i]
    for otro in _COCINADOS.values():
        for orden in (0, 1):
            m = [copy.deepcopy(otro), copy.deepcopy(comida)]
            dias = [{"day": 1, "meals": m if orden == 0 else m[::-1]}]
            go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None)
            for meal in dias[0]["meals"]:
                marca = meal.get("_protein_autofix_applied")
                if not marca:
                    continue
                blob = " ".join(str(t) for t in _textos(meal)).lower()
                assert not _ALERTA_RE.search(blob), (marca, blob[:300])
                destino = marca.split("->")[1]
                if destino in go._PROTEIN_TARGET_FORMS and destino != "queso":
                    rx = pe._rx_alias(go._PROTEIN_TARGET_FORMS[destino].values())
                    assert pe._lo_cuece([pe._norm(p) for p in meal.get("recipe") or []], rx), (marca, blob[:300])


@pytest.mark.parametrize("etiqueta,comida,motivo,ligero", [
    ("atun", _PEZ_LISTO_CRUDO_FRIO["atun-en-agua"][1], "pez_en_conserva", False),
    ("atun", _PEZ_LISTO_CRUDO_FRIO["lata-de-atun"][1], "pez_en_conserva", False),
    ("pescado", _PEZ_LISTO_CRUDO_FRIO["salmon-ahumado"][1], "pez_en_conserva", False),
    ("pescado", _PEZ_LISTO_CRUDO_FRIO["pescado-ya-viene-cocido"][1], "pez_en_conserva", False),
    ("atun", _SONDAS_R6["atun-al-natural"][1], "pez_en_conserva", False),
    ("atun", _SONDAS_R6["atun-en-salmuera"][1], "pez_en_conserva", False),
    ("pescado", _SONDAS_R6["en-aceite-bacalao"][1], "pez_en_conserva", False),
    ("pescado", _SONDAS_R6["conserva-plural"][1], "pez_en_conserva", False),
    ("camarones", _SONDAS_R6["camarones-ya-cocidos-calienta"][1], "pez_precocido", False),
    ("camarones", _SONDAS_R6["camarones-precocidos"][1], "pez_precocido", False),
    ("pescado", _SONDAS_R6["pescado-ya-cocido-participio"][1], "pez_precocido", False),
    ("camarones", _SONDAS_R6["camarones-cocidos-ensalada"][1], "pez_precocido", False),
    # «Saltea los camarones cocidos»: la ronda 6 los reescribía (el cerrador no lista «cocidos»)
    ("camarones", _comida("Arroz con camarones", ["150 g de Camarones cocidos"],
                          ["Saltea los camarones cocidos con ajo 2 minutos."]), "pez_precocido", False),
    ("pescado", _PEZ_LISTO_CRUDO_FRIO["ceviche-de-corvina-blanqueado"][1], "pez_crudo", False),
    ("pescado", _PEZ_LISTO_CRUDO_FRIO["tiradito-de-corvina"][1], "pez_crudo", False),
    ("atun", _SONDAS_R6["tataki-atun"][1], "pez_crudo", False),
    ("pescado", _PEZ_LISTO_CRUDO_FRIO["mero-marinado-sin-fuego"][1], "pez_sin_coccion", False),
    ("pescado", _SONDAS_R6["mero-lista-marinada"][1], "pez_sin_coccion", False),
    ("pescado", _SONDAS_R6["tilapia-se-marina"][1], "pez_sin_coccion", False),
    ("pescado", _SONDAS_R6["tilapia-y-camarones-marinados"][1], "pez_sin_coccion", False),
    ("pescado", _SONDAS_R6["cebiche-mixto-coord"][1], "pez_sin_coccion", False),
    ("pescado", _SONDAS_R6["corvina-leche-de-tigre"][1], "pez_sin_coccion", False),
    # ronda 8: «el salmón curado» en el paso es el producto listo
    ("pescado", _SONDAS_R6["salmon-curado"][1], "pez_en_conserva", False),
    ("pescado", _SONDAS_R6["sirve-helado"][1], "pez_sin_coccion", False),
    ("pescado", _SONDAS_R6["enfria-y-sirve"][1], "pez_sin_coccion", False),
    ("atun", _PEZ_LISTO_CRUDO_FRIO["atun-ensalada-sirve-frio"][1], "pez_sin_coccion", False),
    ("atun", _SONDAS_R6["lata-en-otra-frase-atun"][1], "pez_sin_coccion", False),
    # ronda 8: «cocido» en la LÍNEA es precocido salvo que un paso lleve la firma del cerrador («a la plancha o hervido»,
    # «como proteína del plato»), que es la convención del peso cocido (la ronda 7 dejaba decidir al paso)
    ("camarones", _PEZ_LISTO_CRUDO_FRIO["coctel-de-camarones-frio"][1], "pez_precocido", False),
    ("pulpo", _SONDAS_R6["pulpo-cocido-ensalada"][1], "pez_precocido", False),
    # el blanqueo antes de marinar no cuenta, aunque el nombre no diga «ceviche»
    ("pescado", _comida("Corvina al limón", ["150 g de Corvina"], [_BLANQUEO.format(a="la corvina")]),
     "pez_sin_coccion", False),
    # «Hornea 20 minutos» no nombra el pez: la regla no lo ve y decide el gate (falso positivo aceptado)
    ("pescado", _comida("Tilapia marinada al horno", ["150 g de tilapia marinada"], ["Hornea 20 minutos a 200 °C."]),
     "pez_sin_coccion", False),
    # una nota de seguridad no cuece el pez
    ("pescado", _comida("Tilapia al limón", ["150 g de Filete de tilapia"], [
        "Marina la tilapia en limón 20 minutos.", "⚠ Seguridad alimentaria: cocina el pescado hasta 63 °C."]),
     "pez_sin_coccion", False),
    # lo que se reescribe como en la base
    ("pescado", _tilapia(), None, False),
    ("pescado", _REESCRIBIBLES["marina-y-hornea"][1], None, False),
    ("pescado", _REESCRIBIBLES["ensalada-fria-de-merluza"][1], None, False),
    ("pescado", _comida("Merluza en salsa de tomate", ["150 g de Merluza"],
                        ["Cocina la merluza en salsa de tomate 10 minutos."]), None, False),
    ("pescado", _comida("Pescado al horno", ["150 g de Filete de pescado"],
                        ["Sazona el pescado con ajo; hornéalo 18-20 minutos a 200 °C."]), None, False),
    # la forma del cerrador (prod-1461aeca D3): «30 g de arenque cocido» y «Cocina arenque a la plancha o hervido»
    ("pescado", _comida("Queso blanco al horno con arenque", ["40 g de queso blanco", "30 g de arenque cocido"], [
        "Hornea el queso 12-15 min. Cocina arenque a la plancha o hervido y sírvelo como proteína del plato."]), None, False),
    ("atun", _COCINADOS["atun"], None, False),
    ("camarones", _COCINADOS["camarones"], None, False),
    # sin pasos no hay cláusula que deje el pez crudo: como la base
    ("camarones", _comida("Ensalada con camarones", ["120 g de Camarones", "1 taza de Lechuga"], []), None, False),
    # el único destino de la merienda es el queso: no hay nada que cocer (ronda 5)
    ("pescado", _comida("Casabe con tilapia desmenuzada", ["60 g de tilapia desmenuzada"],
                        ["Coloca la tilapia desmenuzada sobre el casabe."], slot="Merienda"), None, True),
    ("pescado", _comida("Casabe con tilapia desmenuzada", ["60 g de tilapia desmenuzada"],
                        ["Coloca la tilapia desmenuzada sobre el casabe."]), "pez_sin_coccion", False),
    # carne: la guarda es del mar
    ("pollo", _comida("Ensalada fría de pollo", ["120 g de pollo en lata"], ["Sirve frío."]), None, False),
])
def test_motivo_para_no_reescribir(etiqueta, comida, motivo, ligero):
    assert _motivo(etiqueta, comida, ligero) == motivo


def test_la_conserva_sale_de_la_lista_del_cerrador():
    """SSOT: `_PRECOOKED_PROTEIN_HINT` (el cerrador escribe «ya viene cocido» con ella). «sardina» sin «fresca» es
    conserva, como anchoa y mojama; «sardinas frescas» no. La cocción es la de V7f."""
    for linea in ("90 g de Sardinas", "40 g de Anchoas", "30 g de Mojama", "120 g de sardinas en aceite"):
        assert _motivo("pescado", _comida("X", [linea], ["Sirve."])) == "pez_en_conserva", linea
    assert _motivo("pescado", _comida("X", ["150 g de Sardinas frescas"],
                                      ["Cocina las sardinas a la plancha 3 minutos."])) is None
    src = (_BACKEND / "pescado_especies.py").read_text(encoding="utf-8")
    assert "_v7f_evidencia" in src and "_V7F_ENCLITICO_RE" in src, "la cocción se mide con el criterio de V7f"
    for retirado in ("def pez_en_conserva", "def pez_crudo", "def guardiana_y_conservas", "PECES_DE_LATA", "_ENTRE",
                     "_CRUDO_DESPUES", "_MARINA_ANTES", "_SERVIDO_FRIO_RE", "def _crudo"):
        assert retirado not in src, retirado


def test_el_autofix_llama_a_la_guarda_con_el_ssot():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    cuerpo = src.split("def _protein_repeat_autofix", 1)[1].split("\ndef ", 1)[0]
    assert "motivo_para_no_reescribir" in cuerpo and "_PRECOOKED_PROTEIN_HINT" in cuerpo and "_PEZ_NUEVO" in cuerpo
    assert "especie_nueva_en" in cuerpo, "el día con especie nueva se salta entero"
    assert "guardiana_y_conservas" not in cuerpo
    # la guardiana es la de la base: la primera comida con la proteína en el nombre
    assert "_keep_idx = next((i for i, (_, _in_name) in enumerate(hits) if _in_name), 0)" in cuerpo


def test_la_guarda_general_tiene_knob(monkeypatch):
    """`MEALFIT_PROTEIN_AUTOFIX_FISH_READY_GUARD=false` ⇒ la conducta de la base: el atún frío se reescribe."""
    import pescado_especies as pe
    ens = _PEZ_LISTO_CRUDO_FRIO["atun-ensalada-sirve-frio"][1]
    monkeypatch.setattr(pe, "PEZ_LISTO_GUARD", False)
    dias = [{"day": 1, "meals": [copy.deepcopy(_COCINADOS["atun"]), copy.deepcopy(ens)]}]
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None) == 1
    assert dias[0]["meals"][1]["_protein_autofix_applied"] == "atun->pollo"
    monkeypatch.setattr(pe, "PEZ_LISTO_GUARD", True)
    dias = [{"day": 1, "meals": [copy.deepcopy(_COCINADOS["atun"]), copy.deepcopy(ens)]}]
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None) == 0


def test_knob_de_especies_apagado_no_salta_el_dia(monkeypatch):
    import pescado_especies as pe
    comidas = [_tilapia(), _ESPECIES_NUEVAS["trucha-cocida"]]
    monkeypatch.setattr(pe, "BETA_FISH_SPECIES_COUNT", False)
    assert pe.especie_nueva_en(comidas, ["trucha"]) is None
    monkeypatch.setattr(pe, "BETA_FISH_SPECIES_COUNT", True)
    assert pe.especie_nueva_en(comidas, ["trucha"]) == "Trucha a la plancha con papas"
    assert pe.especie_nueva_en(comidas[1:], ["trucha"]) is None, "una sola comida no es repetición"


def test_el_pez_cocinado_de_la_base_se_reescribe_como_antes():
    dias = [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Mero a la plancha", "ingredients": ["150 g de Filete de mero"],
         "recipe": ["Cocina el mero a la plancha 4 minutos por lado."]},
        copy.deepcopy(_REESCRIBIBLES["marina-y-hornea"][1])]}]
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None) == 1
    assert dias[0]["meals"][1]["_protein_autofix_applied"] == "pescado->pollo"


def test_sin_compuestos_de_conserva_en_el_reescritor():
    assert go._PROTEIN_SOURCE_COMPOUNDS["pescado"] == ("filete de pescado blanco", "pescado blanco", "filete de pescado")


# ── ronda 4 · (2) la escalera respeta la dieta ───────────────────────────────────────────────────────────


_PESCETARIANOS = ("pescetariano", "Pescetariana", "pescatarian")


@pytest.mark.parametrize("dieta", _PESCETARIANOS)
@pytest.mark.parametrize("cena", [
    ("Mero a la plancha con ensalada", "150 g de Filete de mero", ["Cocina el mero a la plancha 4 minutos por lado."]),
    ("Ensalada de sardinas en lata con aguacate", "120 g de Sardinas en lata",
     ["Escurre las sardinas en lata y mézclalas con el aguacate.", "Montaje: sirve frío."]),
], ids=["mero", "sardinas-en-lata"])
def test_pescetariano_nunca_recibe_carne(dieta, cena):
    """Base: tilapia + mero con dieta pescetariana → «Pechuga de pollo a la plancha». El pescado sólo se cambia por
    otra proteína del mar; si la escalera no tiene ninguna, no se reescribe (decide el gate)."""
    dias = _dia_dos_pescados(*cena)
    antes = copy.deepcopy(dias)
    assert go._protein_repeat_autofix(dias, {"dietType": dieta, "country": "DO"}, None) == 0
    assert dias == antes
    assert go._scan_diet_violations({"days": dias}, dieta) == []


def test_pescetariano_atun_repetido_no_pasa_a_carne_ni_a_legumbre():
    dias = [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Ensalada de atún", "ingredients": ["120 g de Atún en agua", "1 taza de Lechuga"]},
        {"meal": "Cena", "name": "Arroz con vegetales", "ingredients": ["120 g de atún en agua", "1 taza de Arroz"]},
    ]}]
    go._protein_repeat_autofix(dias, {"dietType": "pescetariano"}, None)
    assert go._scan_diet_violations({"days": dias}, "pescetariano") == []
    assert "atún" in dias[0]["meals"][1]["ingredients"][0].lower()


def test_pescetariano_marisco_repetido_si_pasa_a_pescado():
    dias = [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Camarones al ajillo", "ingredients": ["150 g de Camarones", "1 taza de Arroz"]},
        {"meal": "Cena", "name": "Ensalada con camarones", "ingredients": ["120 g de Camarones", "1 taza de Lechuga"]},
    ]}]
    assert go._protein_repeat_autofix(dias, {"dietType": "pescetariano"}, None) == 1
    assert dias[0]["meals"][1]["_protein_autofix_applied"] == "camarones->pescado"


@pytest.mark.parametrize("dieta", _PESCETARIANOS)
def test_pescetariano_merienda_ligera_pasa_a_queso_como_en_la_base(dieta):
    """Revisión r4 (no bloquea): «del mar sólo a otra del mar» quitaba un arreglo que la base hacía bien — «Casabe con
    tilapia desmenuzada» en la merienda pasaba a queso, que la dieta permite. En la comida ligera o dulce el queso es el
    único destino que el autofix ya admite; en la principal sigue sin reescribirse (la legumbre de respaldo salía
    «50 g de habichuelas rojas guisadas cocida»)."""
    dias = [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Tilapia al horno con arroz",
         "ingredients": ["150 g de Filete de tilapia", "1 taza de Arroz"], "recipe": ["Hornea la tilapia 20 min."]},
        {"meal": "Merienda", "name": "Casabe con tilapia desmenuzada",
         "ingredients": ["1 pieza de Casabe", "60 g de tilapia desmenuzada"],
         "recipe": ["Coloca la tilapia desmenuzada sobre el casabe."]},
    ]}]
    assert go._protein_repeat_autofix(dias, {"dietType": dieta}, None) == 1
    assert dias[0]["meals"][1]["_protein_autofix_applied"] == "pescado->queso"
    assert go._scan_diet_violations({"days": dias}, dieta) == []


def test_omnivoro_sigue_como_antes():
    dias = _dia_dos_pescados("Mero a la plancha con ensalada", "150 g de Filete de mero", ["Cocina el mero."])
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None) == 1
    assert dias[0]["meals"][1]["name"] == "Pechuga de pollo a la plancha con ensalada"


def test_la_dieta_sale_del_ssot():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    cuerpo = src.split("def _protein_repeat_autofix", 1)[1].split("\ndef ", 1)[0]
    assert "destino_apto_para_la_dieta" in cuerpo and "_diet_pool_item_banned" in cuerpo
    pe_src = (_BACKEND / "pescado_especies.py").read_text(encoding="utf-8")
    assert "canonicalize_diet_type" in pe_src


# ── ronda 4 · (3) las anchoas ────────────────────────────────────────────────────────────────────────────


def test_anchoas_son_pescado():
    """Datos: la única comida del corpus con anchoas (ES, «Coca de vegetales con anchoas») lleva 40 g — la mitad de la
    proteína del plato — y el catálogo la tiene en «Proteínas». Anchoas y boquerones el mismo día son el mismo pez."""
    assert "pescado" in go._protein_gate_labels_in_text("Coca de vegetales con anchoas")
    dia = {"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Coca de vegetales con anchoas", "ingredients": ["40 g de Anchoas"]},
        {"meal": "Cena", "name": "Boquerones fritos", "ingredients": ["150 g de Boquerones"]},
    ]}
    assert go._days_with_same_day_protein_repeat({"days": [dia]}) == [1]
    assert "atun" not in go._protein_gate_labels_in_text("40 g de Anchoas")

# ── contrato ─────────────────────────────────────────────────────────────────────────────────────────────


def test_knobs_documentados():
    doc = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    for knob in ("MEALFIT_BETA_FISH_SPECIES_COUNT", "MEALFIT_STAPLE_TOKEN_MATCH",
                 "MEALFIT_PROTEIN_AUTOFIX_FISH_READY_GUARD"):
        assert knob in doc, knob
    assert "P1-PLAN-LOTE-857" in doc


def test_knobs_en_el_inventario_del_arranque():
    """Ronda 4: `basicos_por_token` se importaba en la primera llamada, así que su knob faltaba en el
    `[KNOBS/INVENTORY]` que `graph_orchestrator` emite al terminar de importarse. Se mide en un proceso limpio y sin
    importar antes los módulos del lote (importarlos primero es justo lo que escondía el fallo)."""
    codigo = (
        "import logging, os, sys\n"
        "os.environ['MEALFIT_DISABLE_SEMANTIC_CACHE'] = 'true'\n"
        "logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(message)s')\n"
        "import graph_orchestrator\n"
    )
    out = subprocess.run([sys.executable, "-c", codigo], cwd=_BACKEND, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    inventario = [ln for ln in out.stdout.splitlines() if "[KNOBS/INVENTORY]" in ln]
    assert inventario, "el arranque no emitió el inventario de knobs"
    for knob in ("MEALFIT_BETA_FISH_SPECIES_COUNT", "MEALFIT_STAPLE_TOKEN_MATCH",
                 "MEALFIT_PROTEIN_AUTOFIX_FISH_READY_GUARD"):
        assert knob + "=" in inventario[-1], knob


def test_marcador_y_anclas():
    for mod in ("pescado_especies.py", "basicos_por_token.py"):
        src = (_BACKEND / mod).read_text(encoding="utf-8")
        assert "[P1-PLAN-LOTE-857 · 2026-09-29]" in src and "tooltip-anchor: P1-PLAN-LOTE-857" in src, mod
    src_go = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    cuerpo = src_go.split("def _count_staple_repetitions", 1)[1].split("\ndef ", 1)[0]
    assert "basicos_por_token" in cuerpo and "any(a in text_norm for a in alias_list)" not in cuerpo
    assert 'pescado_especies").extender_pescado(_MAIN_PROTEIN_ALIASES, _ALLERGEN_SYNONYMS["pescado"])' in src_go
    src_dd = (_BACKEND / "deterministic_day.py").read_text(encoding="utf-8")
    assert "basicos_por_token" in src_dd.split("def _basicos_de", 1)[1].split("\ndef ", 1)[0]
