# -*- coding: utf-8 -*-
"""[P1-STEP-NAME-TAIL · 2026-09-07] La cola del nombre colgando tras el paréntesis.

`sync_recipe_steps_to_household` armoniza las unidades entre la lista y los pasos: reescribe
«190 g de yogurt» a «1 taza de yogurt (190 g)» para que el usuario pueda cocinar con taza o con
balanza. Sustituye el tramo que casó `_STEP_GRAMS_MENTION_RE` — y esa regex captura el nombre del
alimento con una ventana de **tres palabras**. Con un nombre de cuatro la captura se queda corta y
la palabra sobrante sobrevive a la sustitución:

    «mide 1 taza de yogurt griego sin azúcar (190 g) azúcar»

Medido en la flota viva: **24 de 1.194 comidas (2,0 %)** con palabra residual y **32 (2,7 %)** con
dos gramajes que se contradicen. El dueño lo marcó a ciegas en 4 de 37 comidas del juicio del
duelo, sin saber qué lo causaba, llamándolo «texto corrupto» y «pesos contradictorios».

**Lo que NO se hizo**: ensanchar la ventana de la regex. Tragaría texto que no es el nombre — en
«190 g de yogurt reserva para el final», «reserva» entraría en la captura. El tramo se acota por
el nombre REAL del ingrediente, que la función ya tiene en `token_map`.

Estos tests anclan las dos direcciones del arreglo, que es donde vive el riesgo: que **consuma
todo el nombre** (el defecto) y que **no consuma ni una palabra más** (la regresión que un
arreglo perezoso introduciría).
"""
import re

import pytest

from humanize_ingredients import (_STEP_GRAMS_MENTION_RE, _fin_del_nombre,
                                  sync_recipe_steps_to_household)


def _meal(humanizada, raw, pasos):
    return {"ingredients": [humanizada], "ingredients_raw": [raw], "recipe": list(pasos)}


# ------------------------------------------------- la regex sigue cortando: eso NO se tocó

def test_la_regex_sigue_capturando_solo_tres_palabras():
    """El arreglo NO ensancha la ventana — se documenta que sigue corta a propósito.

    Si alguien la ensancha creyendo que ahí estaba el bug, este test se lo dice: el precio de
    ensancharla es tragar «reserva», «picado», «al gusto» y demás cola que no es el nombre.
    """
    m = _STEP_GRAMS_MENTION_RE.search("190 g de yogurt griego sin azucar")
    assert m.group(2) == "yogurt griego sin"


def test_la_regex_ya_frenaba_bien_en_las_preposiciones_de_preparacion():
    """«en lonjas» / «en cubos» nunca fueron el problema: la exclusión los frena."""
    m = _STEP_GRAMS_MENTION_RE.search("40 g de semillas de linaza en lonjas")
    assert m.group(2) == "semillas de linaza"


# ------------------------------------------------- el defecto: la cola sobrevivía

def test_el_nombre_de_cuatro_palabras_no_deja_cola():
    meal = _meal("1 taza de yogurt griego sin azúcar",
                 "190 g de yogurt griego sin azúcar",
                 ["Mise en place: mide 190 g de yogurt griego sin azúcar y reserva."])
    sync_recipe_steps_to_household(meal)
    paso = meal["recipe"][0]
    assert "(190 g) azúcar" not in paso, f"la cola del nombre sigue colgando: {paso!r}"
    assert paso.count("azúcar") == 1, f"«azúcar» aparece dos veces: {paso!r}"
    assert "1 taza de yogurt griego sin azúcar (190 g)" in paso


def test_el_nombre_de_cinco_palabras_tampoco():
    meal = _meal("2 cdas de mantequilla de maní sin azúcar añadida",
                 "30 g de mantequilla de maní sin azúcar añadida",
                 ["Añade 30 g de mantequilla de maní sin azúcar añadida y mezcla."])
    sync_recipe_steps_to_household(meal)
    paso = meal["recipe"][0]
    assert "añadida" in paso and paso.count("añadida") == 1, paso
    assert "g) sin" not in paso and "g) azúcar" not in paso, paso


# ------------------------------------------------- la regresión: no consumir de más

@pytest.mark.parametrize("cola", ["en cubos", "en lonjas", "picado finito", "y reserva"])
def test_no_se_traga_la_preparacion_que_sigue_al_nombre(cola):
    """El texto que NO es el nombre del alimento tiene que sobrevivir intacto.

    Es la mitad del contrato que un arreglo perezoso —ensanchar la ventana— rompería en silencio:
    el paso seguiría leyéndose bien, pero habría perdido la instrucción.
    """
    meal = _meal("1 taza de arroz blanco", "80 g de arroz blanco",
                 [f"Mide 80 g de arroz blanco {cola}."])
    sync_recipe_steps_to_household(meal)
    assert cola in meal["recipe"][0], meal["recipe"][0]


def test_no_se_traga_una_palabra_que_solo_PARECE_del_nombre():
    """«sin» abre la cola del nombre, pero sólo si el ingrediente lo lleva de verdad."""
    meal = _meal("1 taza de yogurt griego", "190 g de yogurt griego",
                 ["Mide 190 g de yogurt griego sin remover demasiado."])
    sync_recipe_steps_to_household(meal)
    assert "sin remover demasiado" in meal["recipe"][0], meal["recipe"][0]


# ------------------------------------------------- el doble gramaje

def test_no_escribe_dos_gramajes_que_se_contradicen():
    """Si la forma humanizada YA termina en gramos, no se le añade otra cifra.

    «Yogurt griego sin azúcar 150g (95 g)» son dos pesos en la misma frase y el usuario no tiene
    forma de saber a cuál obedecer. 32 comidas vivas lo tenían.
    """
    meal = _meal("½ taza de yogurt griego sin azúcar 150g",
                 "95 g de yogurt griego sin azúcar",
                 ["Mide 95 g de yogurt griego sin azúcar."])
    sync_recipe_steps_to_household(meal)
    paso = meal["recipe"][0]
    cifras = re.findall(r"\d+(?:[.,]\d+)?\s*g\b", paso)
    assert len(cifras) == 1, f"dos gramajes en la misma frase: {paso!r} -> {cifras}"


# ------------------------------------------------- el helper, y el fail-safe

def test_el_helper_no_extiende_cuando_no_hay_nada_que_extender():
    """El caso mayoritario: nombre de ≤3 palabras, la regex ya lo cubrió entero."""
    paso = "Mide 80 g de arroz blanco en cubos."
    m = _STEP_GRAMS_MENTION_RE.search(paso)
    assert _fin_del_nombre(paso, m.start(2), m.end(2), "arroz blanco") == m.end(2)


def test_el_helper_cae_al_comportamiento_previo_ante_basura():
    paso = "Mide 80 g de arroz blanco."
    m = _STEP_GRAMS_MENTION_RE.search(paso)
    for basura in ("", None, 12345, "   "):
        assert _fin_del_nombre(paso, m.start(2), m.end(2), basura) == m.end(2)


def test_es_idempotente():
    """La mención ya reescrita no vuelve a casar: correrla dos veces no acumula paréntesis."""
    meal = _meal("1 taza de yogurt griego sin azúcar",
                 "190 g de yogurt griego sin azúcar",
                 ["Mide 190 g de yogurt griego sin azúcar y reserva."])
    sync_recipe_steps_to_household(meal)
    una = meal["recipe"][0]
    sync_recipe_steps_to_household(meal)
    assert meal["recipe"][0] == una


def test_fail_safe_deja_el_texto_intacto():
    """Sin `ingredients_raw` la función no puede saber nada: devuelve 0 y no toca el paso."""
    meal = {"ingredients": ["1 taza de arroz"], "recipe": ["Mide 80 g de arroz blanco."]}
    antes = list(meal["recipe"])
    assert sync_recipe_steps_to_household(meal) == 0
    assert meal["recipe"] == antes
