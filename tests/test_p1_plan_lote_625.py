# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-625 · 2026-09-27] Un paso traducido que pierde la etiqueta de sección la RECUPERA en vez de caer entero
al español.

Medido en el único plan real traducido (fr-FR, 3957a669): 12 de 54 pasos de receta idénticos al original español
—7 «Mise en place», 4 «Toque de Fuego», 1 «Seguridad alimentaria»— y la telemetría decía «16 de 16 escritas». El
modelo escribe la tipografía francesa correcta («Mise en place : ») o traduce la etiqueta («Coup de feu : »), el
validador lo detecta (la etiqueta es un identificador: los parsers de pantalla casan el español pegado a los dos
puntos) y descartaba la LÍNEA ENTERA. La etiqueta es lo único que hay que conservar: se repone la española delante del
texto traducido y la línea vuelve a pasar por el mismo control. Si no hay dónde reponerla, cae al español como antes.

Además la directiva pide las cifras en dígitos: «trois œufs» por «3 huevos» también devolvía el paso al español.

tooltip-anchor: P1-PLAN-LOTE-625
"""
import pytest


@pytest.fixture(scope="module")
def mod():
    import plan_display_i18n as m
    return m


def _original(pasos):
    return {"day_idx": 0, "meal_idx": 0, "name": "Pollo guisado", "description": "Plato criollo.",
            "recipe": pasos, "ingredients": ["180 g de Pechuga de pollo"]}


def _traducido(pasos):
    return {"i": 0, "name": "Poulet mijoté", "description": "Plat créole.",
            "recipe": pasos, "ingredients": ["180 g de blanc de poulet (Pechuga de pollo)"]}


@pytest.mark.parametrize("orig,trad,esperado", [
    ("Mise en place: pica la cebolla.", "Mise en place : hache l'oignon.", "Mise en place: hache l'oignon."),
    ("Mise en place: pica la cebolla.", "Mise en place : hache l'oignon.", "Mise en place: hache l'oignon."),
    ("Mise en place: pica la cebolla.", "Mise en place : hache l'oignon.", "Mise en place: hache l'oignon."),
    ("El Toque de Fuego: dora el pollo 5 minutos.", "Le coup de feu : fais dorer le poulet 5 minutes.",
     "El Toque de Fuego: fais dorer le poulet 5 minutes."),
    ("Montaje: sirve caliente.", "Dressage : sers chaud.", "Montaje: sers chaud."),
    ("Seguridad alimentaria: cocina el pollo hasta 74 °C.", "Sécurité alimentaire : cuis le poulet jusqu'à 74 °C.",
     "Seguridad alimentaria: cuis le poulet jusqu'à 74 °C."),
    ("🔬 Nota del nutricionista: cubre el 30 % del hierro.", "🔬 Note du nutritionniste : couvre 30 % du fer.",
     "🔬 Nota del nutricionista: couvre 30 % du fer."),
])
def test_la_etiqueta_se_repone_y_el_texto_queda_traducido(mod, orig, trad, esperado):
    d = mod._validate_and_build_display(_original([orig]), _traducido([trad]))
    assert d["recipe"] == [esperado]
    assert mod._conserva_el_vocab_cerrado(orig, d["recipe"][0])


def test_sin_donde_reponerla_cae_al_espanol(mod):
    # sin dos puntos cerca del principio no hay etiqueta que sustituir: el español, como antes
    orig = ["Mise en place: pica la cebolla."]
    d = mod._validate_and_build_display(_original(orig), _traducido(["Hachez l'oignon finement avant tout."]))
    assert d["recipe"] == orig


def test_la_reparacion_no_salta_el_control_de_cifras(mod):
    orig = ["Mise en place: bate 3 huevos."]
    d = mod._validate_and_build_display(_original(orig), _traducido(["Mise en place : bats trois œufs."]))
    assert d["recipe"] == orig


def test_un_paso_sin_etiqueta_no_se_toca(mod):
    orig = ["Dora el pollo 5 minutos."]
    trad = ["Fais dorer le poulet : 5 minutes."]
    d = mod._validate_and_build_display(_original(orig), _traducido(trad))
    assert d["recipe"] == trad


@pytest.mark.parametrize("locale", ["en-US", "pt-BR", "fr-FR", "it-IT"])
def test_la_directiva_pide_las_cifras_en_digitos(mod, locale):
    d = mod._DISPLAY_LANGUAGE_DIRECTIVES[locale]
    assert '"3"' in d, locale
