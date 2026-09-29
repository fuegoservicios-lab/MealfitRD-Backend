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


# ── ronda 5 · la conserva se queda; se cambia el pez que se cuece ────────────────────────────────────────


def _dia_dos_pescados(nombre, linea, pasos):
    return [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Tilapia al horno con arroz",
         "ingredients": ["150 g de Filete de tilapia", "1 taza de Arroz"],
         "recipe": ["Hornea la tilapia 20 minutos a 200 °C."]},
        {"meal": "Cena", "name": nombre, "ingredients": [linea, "1/2 Aguacate"],
         "ingredients_raw": [linea, "1/2 Aguacate"], "recipe": list(pasos)},
    ]}]


def _textos(meal):
    return [meal["name"], *meal["ingredients"], *(meal.get("ingredients_raw") or []), *meal["recipe"]]


_CONSERVAS = [
    # la sonda del revisor r4: fría, «Escurre… sirve frío» — salía pechuga de pollo cruda
    ("Ensalada de sardinas en lata con aguacate", "120 g de Sardinas en lata",
     ["Mise en place: pica el tomate y el aguacate.",
      "Escurre las sardinas en lata y mézclalas con el aguacate y el tomate.", "Montaje: sirve frío."]),
    ("Ensalada de sardinas con aguacate", "1 lata de Sardinas en lata (120 g)",
     ["Escurre las sardinas en lata y mézclalas con el aguacate.",
      "Añade un pellizco del líquido de la lata de sardinas si quieres darle sabor, y sirve."]),
    ("Arroz con sardinas", "2 latas de sardinas (250 g)", ["Abre las latas de sardinas y mézclalas con el arroz."]),
    # con fuego: es un RECALENTADO — 2-3 min no cuecen una pechuga cruda, y «Abre pechuga de pollo»
    ("Sardinas guisadas", "120 g de Sardinas en salsa de tomate",
     ["Calienta las sardinas en salsa de tomate en la sartén 3 minutos."]),
    ("Arroz con sardinas", "1 lata de sardinas (120 g)",
     ["Abre la lata de sardinas y caliéntalas en la sartén 3 minutos."]),
    ("Ensalada de sardinas", "110 g de sardinas en lata escurridas",
     ["Desmenuza las sardinas y saltéalas 2 minutos en la sartén."]),
    ("Sardinas en aceite con casabe", "120 g de Sardinas en aceite", ["Escurre las sardinas en aceite y sírvelas."]),
    ("Sardinas en agua con casabe", "120 g de Sardinas en agua", ["Escurre las sardinas en agua."]),
    ("Ensalada de bonito", "100 g de Bonito del norte en aceite", ["Desmenuza el bonito del norte en aceite."]),
    ("Ensalada de caballa", "1 lata de caballa en aceite de oliva (110 g)", ["Desmenuza la caballa en aceite de oliva."]),
    ("Caballa en escabeche", "120 g de Caballa en escabeche", ["Sirve la caballa en escabeche fría."]),
    ("Tosta de boquerones en vinagre", "80 g de Boquerones en vinagre", ["Coloca los boquerones en vinagre sobre el pan."]),
    ("Ensalada de trucha ahumada", "100 g de Trucha ahumada", ["Corta la trucha ahumada en tiras."]),
    ("Tosta de salmón ahumado", "80 g de Salmón ahumado", ["Coloca el salmón ahumado sobre la tosta."]),
    ("Coca con anchoas", "40 g de Anchoas", ["Reparte las anchoas sobre la coca."]),
]


@pytest.mark.parametrize("nombre,linea,pasos", _CONSERVAS, ids=[c[1] for c in _CONSERVAS])
def test_la_conserva_se_queda_y_se_cambia_el_pez_que_se_cuece(nombre, linea, pasos):
    """Revisión r4 (bloquea, seguridad alimentaria): tilapia + «Ensalada de sardinas en lata» salía «Escurre pechuga de
    pollo y mézclalas… sirve frío» — pollo crudo, y ni V7f ni el reparador del lote 68 lo veían. La receta de una conserva
    está escrita para un producto listo para comer: su fuego, si lo hay, es un recalentado. La conserva se queda y el
    autofix cambia el OTRO pez, el que la receta sí cuece; el plato reescrito no hereda ninguna lata."""
    dias = _dia_dos_pescados(nombre, linea, pasos)
    cena_antes = copy.deepcopy(dias[0]["meals"][1])
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced", "country": "DO"}, None) == 1
    alm, cena = dias[0]["meals"]
    assert cena == cena_antes, "la conserva no se toca"
    assert alm.get("_protein_autofix_applied") == "pescado->pollo"
    assert alm["name"] == "Pechuga de pollo al horno con arroz"
    assert alm["recipe"] == ["Hornea pechuga de pollo 20 minutos a 200 °C."]
    for texto in _textos(alm):
        assert not re.search(r"\blatas?\b|enlatad|escurrid", texto, re.IGNORECASE), texto
    assert go._days_with_same_day_protein_repeat({"days": dias}) == []


def _ceviche_de_sardinas_g24_co():
    """G24 CO D1, almuerzo tal cual (la conserva con garbanzos de lata en el mismo paso)."""
    return {"meal": "Almuerzo", "name": "Ensalada Fresca Estilo Ceviche con Sardinas, Garbanzos y Plátano Maduro Dorado",
            "ingredients": ["110 g de sardinas en lata escurridas",
                            "½ taza de garbanzos cocidos (de lata, enjuagados y escurridos)", "½ plátano maduro",
                            "1½ limones"],
            "recipe": [
                "Mise en place: escurre bien los 110 g de sardinas en lata y desmenúzalas; enjuaga y escurre los ½ taza "
                "de garbanzos cocidos (de lata, enjuagados y escurridos) (90 g). Corta los ½ plátano maduro (105 g) en "
                "rodajas de 1 cm y exprime los 1½ limones.",
                "El Toque de Fuego: en un sartén a fuego medio con 1 cdta de aceite de oliva, dora las rodajas de "
                "plátano maduro 3 minutos por lado hasta que caramelicen y estén tiernas.",
                "Montaje: marina las sardinas 5 minutos con el jugo de limón, sal, pimienta negra y el cilantro picado. "
                "Sirve la lechuga con los garbanzos, corona con las sardinas al limón y acompaña con el plátano."]}


def _arepas_con_trucha_g24_co():
    return {"meal": "Cena", "name": "Arepas de Maíz Asadas Rellenas de Ricotta con Ensalada Fresca y Trucha",
            "ingredients": ["35 g de harina de maíz precocida", "180 g de queso ricotta", "50 g de trucha cocida"],
            "recipe": ["El Toque de Fuego: forma 2 arepas y ásalas a fuego medio 5-6 minutos por lado. Cocina trucha a "
                       "la plancha o hervida y sírvela como proteína del plato.",
                       "Montaje: sirve las arepas rellenas con la ensalada fresca."]}


@pytest.mark.parametrize("trucha_primero", [True, False], ids=["trucha-primero", "orden-real"])
def test_g24_co_el_ceviche_de_sardinas_nunca_es_ceviche_de_pollo(trucha_primero):
    """Revisión r4: con la trucha delante, la guardiana era la trucha y salía «Ensalada Fresca Estilo Ceviche con Pechuga
    de pollo» + «marina pechuga de pollo 5 minutos con el jugo de limón». El plato SÍ tiene fuego (el del plátano): no
    basta con mirar si el plato tiene fuego. Y el paso de los garbanzos queda intacto (r4 dejaba «(de lata, enjuagados
    y ) (90 g)»), igual que «110 g de sardinas en lata escurridas» (r4: «pechuga de pollo escurridas»)."""
    sard, tru = _ceviche_de_sardinas_g24_co(), _arepas_con_trucha_g24_co()
    sard_antes = copy.deepcopy(sard)
    dias = [{"day": 1, "meals": [tru, sard] if trucha_primero else [sard, tru]}]
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced", "country": "CO"}, None) == 1
    assert sard == sard_antes
    assert "(de lata, enjuagados y escurridos) (90 g)" in sard["recipe"][0]
    assert tru["_protein_autofix_applied"] == "pescado->pollo"
    assert tru["name"].endswith("y Pechuga de pollo")
    assert "50 g de pechuga de pollo cocida" in tru["ingredients"]
    assert "Cocina pechuga de pollo a la plancha" in tru["recipe"][0]


def test_dos_conservas_el_mismo_dia_no_se_reescriben(caplog):
    """Sin un pez que la receta cueza, no hay a quién cambiar: decide el gate, y la impotencia queda en el log."""
    dias = [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Ensalada de sardinas", "ingredients": ["120 g de Sardinas en lata"],
         "recipe": ["Escurre las sardinas y sírvelas."]},
        {"meal": "Cena", "name": "Casabe con caballa", "ingredients": ["1 lata de caballa en aceite (110 g)"],
         "recipe": ["Desmenuza la caballa sobre el casabe."]},
    ]}]
    antes = copy.deepcopy(dias)
    with caplog.at_level(logging.INFO):
        assert go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None) == 0
    assert dias == antes
    assert "conserva_sin_fuego" in caplog.text


def test_el_pez_fresco_se_reescribe_sin_tocar_los_garbanzos_de_lata():
    """El pez fresco se sigue reescribiendo como en la base, y la lata de OTRO alimento del mismo paso no se toca."""
    dias = _dia_dos_pescados("Mero guisado con garbanzos", "150 g de Filete de mero",
                             ["Guisa el mero 15 minutos con los garbanzos cocidos (de lata, enjuagados y escurridos) "
                              "(90 g)."])
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None) == 1
    cena = dias[0]["meals"][1]
    assert cena["_protein_autofix_applied"] == "pescado->pollo"
    assert cena["recipe"] == ["Guisa pechuga de pollo 15 minutos con los garbanzos cocidos (de lata, enjuagados y "
                              "escurridos) (90 g)."]


def test_el_pez_fresco_en_salsa_sigue_como_antes():
    """«en salsa de tomate» sólo es la lata en los peces que se venden así: una merluza en salsa es una preparación."""
    dias = _dia_dos_pescados("Merluza en salsa de tomate", "150 g de Merluza",
                             ["Cocina la merluza en salsa de tomate 10 minutos."])
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None) == 1
    assert dias[0]["meals"][1]["name"] == "Pechuga de pollo en salsa de tomate"


@pytest.mark.parametrize("texto,esperado", [
    ("120 g de Sardinas en lata", True), ("1 lata de sardina (120 g)", True), ("2 latas de sardinas", True),
    ("120 g de sardinas (de lata)", True), ("120 g de Sardinas enlatadas", True), ("120 g de Sardinas en agua", True),
    ("100 g de Bonito del norte en aceite", True), ("80 g de Boquerones en vinagre", True),
    ("100 g de Trucha ahumada", True), ("120 g de Caballa en escabeche", True), ("40 g de Anchoas", True),
    ("30 g de Mojama", True),
    ("150 g de Sardinas frescas", False), ("150 g de Merluza en salsa de tomate", False),
    ("150 g de Tilapia con garbanzos de lata", False), ("150 g de Bacalao salado", False),
    ("150 g de Tilapia con pimentón ahumado", False), ("150 g de Filete de tilapia", False),
])
def test_pez_en_conserva(texto, esperado):
    import pescado_especies as pe
    assert pe.pez_en_conserva({"name": "Plato", "ingredients": [texto]}, go._MAIN_PROTEIN_ALIASES["pescado"]) is esperado


def test_pez_en_conserva_mira_los_pasos_y_no_depende_del_orden():
    """El vinagre o la lata del PASO también cuentan («marina las sardinas en vinagre» es un boquerón crudo); el aceite o
    el agua sólo en la lista (en un paso, «hierve las sardinas en agua» es una cocción). Pura: sin estado global
    (r4 mutaba `_CONSERVAS` y el resultado dependía del orden de los tests)."""
    import pescado_especies as pe
    pez = go._MAIN_PROTEIN_ALIASES["pescado"]
    assert pe.pez_en_conserva({"name": "X", "ingredients": ["150 g de Trucha"]}, ["trucha"]) is False
    assert pe.pez_en_conserva({"name": "Sardinas", "ingredients": ["150 g de Sardinas"],
                               "recipe": ["Marina las sardinas en vinagre 2 horas."]}, pez) is True
    assert pe.pez_en_conserva({"name": "Sardinas", "ingredients": ["150 g de Sardinas"],
                               "recipe": ["Hierve las sardinas en agua 10 minutos."]}, pez) is False
    assert pe.pez_en_conserva({"name": "X", "ingredients": ["120 g de Sardinas en lata"]}, pez) is True


def test_knob_apagado_no_mira_conservas(monkeypatch):
    import pescado_especies as pe
    monkeypatch.setattr(pe, "BETA_FISH_SPECIES_COUNT", False)
    assert pe.pez_en_conserva({"name": "X", "ingredients": ["120 g de Sardinas en lata"]}, ["sardina"]) is False
    assert pe.pez_crudo({"name": "Ceviche de trucha", "ingredients": ["150 g de Trucha"]}, ["trucha"]) is False
    assert pe.guardiana_y_conservas("pescado", [{"name": "Tilapia"}, {"name": "Sardinas en lata"}], ["sardina"], 0) \
        == (0, {})


# ── ronda 5 · el pez CRUDO tampoco pasa a pollo crudo ────────────────────────────────────────────────────


_CRUDOS = [
    # la sonda del propio revisor r4 (probe.py): salía «Sirve pechuga de pollo marinadas al limón»
    ("Sardinas marinadas", "120 g de Sardinas marinadas", ["Sirve las sardinas marinadas al limón."]),
    ("Ceviche de trucha", "150 g de Trucha", ["Marina la trucha en jugo de limón 20 minutos.", "Sirve frío."]),
    ("Tiradito de corvina", "120 g de Corvina", ["Corta la corvina en láminas finas y báñala con leche de tigre."]),
    ("Ensalada con boquerones", "80 g de Boquerones crudos", ["Limpia los boquerones y alíñalos con limón."]),
]


@pytest.mark.parametrize("nombre,linea,pasos", _CRUDOS, ids=[c[0] for c in _CRUDOS])
def test_el_pez_crudo_se_queda_y_se_cambia_el_pez_que_se_cuece(nombre, linea, pasos):
    dias = _dia_dos_pescados(nombre, linea, pasos)
    cena_antes = copy.deepcopy(dias[0]["meals"][1])
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced", "country": "DO"}, None) == 1
    alm, cena = dias[0]["meals"]
    assert cena == cena_antes, "el pez crudo no se toca"
    assert alm.get("_protein_autofix_applied") == "pescado->pollo"


def test_el_pez_marinado_que_luego_se_hornea_se_reescribe_como_antes():
    """La señal cruda no basta: si una cláusula lo cuece (aunque sea con un enclítico: «…; luego hornéala»), o el nombre
    declara la cocción, el pez se reescribe como en la base."""
    import pescado_especies as pe
    pez = go._MAIN_PROTEIN_ALIASES["pescado"]
    marinada = {"name": "Tilapia con arroz", "ingredients": ["150 g de Filete de tilapia marinada"],
                "recipe": ["Marina la tilapia con limón 10 minutos; luego hornéala 20 minutos a 200 °C."]}
    assert pe.pez_crudo(marinada, pez) is False
    assert pe.pez_crudo({"name": "Tilapia marinada al horno", "ingredients": ["150 g de tilapia marinada"],
                         "recipe": ["Hornea 20 minutos a 200 °C."]}, pez) is False
    # réplica 63eedc6b (locrio): el «crudo» es del arroz, no del pez — la señal tiene que ir PEGADA al pez
    assert pe.pez_crudo({"name": "Locrio de mero", "ingredients": ["150 g de Filete de mero"], "recipe": [
        "Incorpora el mero en trozos grandes, remueve con cuidado y agrega el arroz, que se pesa en crudo. Revuelve, "
        "cubre con agua y cocina a fuego alto hasta que rompa hervor; tapa y cocina unos 15-18 minutos."]}, pez) is False
    dias = [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Mero a la plancha", "ingredients": ["150 g de Filete de mero"],
         "recipe": ["Cocina el mero a la plancha 4 minutos por lado."]},
        {"meal": "Cena", **copy.deepcopy(marinada)}]}]
    assert go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None) == 1
    assert dias[0]["meals"][1]["_protein_autofix_applied"] == "pescado->pollo"
    assert "hornéala" in dias[0]["meals"][1]["recipe"][0]


def test_la_impotencia_nombra_el_motivo(caplog):
    dias = [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Ceviche de corvina", "ingredients": ["150 g de Corvina"],
         "recipe": ["Marina la corvina en limón 20 minutos."]},
        {"meal": "Cena", "name": "Ensalada de sardinas", "ingredients": ["120 g de Sardinas en lata"],
         "recipe": ["Escurre las sardinas y sírvelas."]},
    ]}]
    with caplog.at_level(logging.INFO):
        assert go._protein_repeat_autofix(dias, {"dietType": "balanced"}, None) == 0
    assert "reason=conserva_sin_fuego" in caplog.text


def test_sin_compuestos_de_conserva_en_el_reescritor():
    """r4 metía 2 078 frases en `_PROTEIN_SOURCE_COMPOUNDS['pescado']` para reescribir la lata entera; como la conserva
    ya no se reescribe, sobran (y con ellas el filtro de coste y la limpieza de pasos que rompía los garbanzos)."""
    assert go._PROTEIN_SOURCE_COMPOUNDS["pescado"] == ("filete de pescado blanco", "pescado blanco", "filete de pescado")


# ── ronda 4 · (2) la escalera respeta la dieta ───────────────────────────────────────────────────────────


_PESCETARIANOS = ("pescetariano", "Pescetariana", "pescatarian")


@pytest.mark.parametrize("dieta", _PESCETARIANOS)
@pytest.mark.parametrize("cena", [
    ("Mero a la plancha con ensalada", "150 g de Filete de mero", ["Cocina el mero a la plancha 4 minutos por lado."]),
    _CONSERVAS[0],
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
    for knob in ("MEALFIT_BETA_FISH_SPECIES_COUNT", "MEALFIT_STAPLE_TOKEN_MATCH"):
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
    for knob in ("MEALFIT_BETA_FISH_SPECIES_COUNT", "MEALFIT_STAPLE_TOKEN_MATCH"):
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
