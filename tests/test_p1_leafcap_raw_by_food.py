# -*- coding: utf-8 -*-
"""[P1-LEAFCAP-RAW-BY-FOOD · 2026-09-08] La lista compraba la hoja SIN capar.

`_cap_leaf_volume_in_meals` (P3-LEAF-VOLUME-CAP, 28-jun) recorta el volumen de hojas crudas que el
solver infla para clavar carbs: «1 taza de repollo morado» (89 g) baja al tope de 75 g. Escribía
`ings[i]` y **no mencionaba `ingredients_raw` ni una vez** en toda la función.

## Cómo apareció, y por qué el número honesto es pequeño

Salió del mismo barrido que `P1-NIGHTRICE-RAW-BY-FOOD`: derivar por AST los pases del finalizador y
correrlos uno a uno sobre las comidas ya alineadas. 46 de las 62 líneas de la clase «el display
mueve los gramos y raw se queda» son este pase; las otras 16 sólo emergen al componer pases y no se
le pueden atribuir a ninguno.

Severidad medida, no supuesta: **los macros se calculan sobre `meal['ingredients']`** — el lado que
el usuario lee—, así que el plato y sus macros concuerdan. El desvío es de COMPRA (89 g comprados
para una receta que usa 75), y sobre una hoja barata. No es el «lee batata / compra arroz».

## Por qué se arregló igual, siendo pequeño

Porque cuesta cero líneas y cierra una incoherencia entre superficies: en las dos cadenas de
generación los reconciliadores posteriores YA dejan raw capado (medido en los planes vivos: 252
líneas de hoja sanas, 1 divergente marginal), así que la misma receta se comportaba distinto según
llegara por generación o por swap. Esa diferencia es la que confunde a un operador.

`graph_orchestrator.py` está en 53.100 de 53.100: el arreglo reusa `sustituye_display_y_raw` —el
helper de `P1-NIGHTRICE-RAW-BY-FOOD`— sobre la línea que ya existía, y el import va pegado al que
ya había. Cero líneas nuevas.
"""
import inspect

import graph_orchestrator as go


def test_el_cap_de_hojas_escribe_raw():
    src = inspect.getsource(go._cap_leaf_volume_in_meals)
    assert "_sub_dr(m, ing, i, ings, new_ing)" in src, (
        "volvió a capar sólo el display: la lista compra la hoja sin capar")
    assert "sustituye_display_y_raw" in src, "perdió el import del helper"


def test_reusa_el_helper_no_escribe_otro():
    """Tercer pase que necesita lo mismo. Si aparece un cuarto, que use este helper: la familia
    `raw[idx]` costó siete P-fixes justamente por tener una implementación por sitio."""
    import constants

    src = inspect.getsource(go._cap_leaf_volume_in_meals)
    assert "_raw_idx_for_display" not in src, (
        "reimplementó el emparejado por alimento dentro del pase en vez de usar el SSOT")
    assert hasattr(constants, "sustituye_display_y_raw")


def test_capa_raw_a_la_MISMA_linea():
    """El caso real de la flota: «1 taza de repollo morado rallado» → tope de 75 g."""
    import constants

    meal = {"ingredients": ["1 taza de repollo morado rallado"],
            "ingredients_raw": ["1 taza de repollo morado rallado"]}
    constants.sustituye_display_y_raw(
        meal, "1 taza de repollo morado rallado", 0, meal["ingredients"],
        "0.84 taza de repollo morado rallado")
    assert meal["ingredients_raw"] == ["0.84 taza de repollo morado rallado"]


def test_el_cap_sigue_capando():
    """Un arreglo que apaga la función no es un arreglo."""
    src = inspect.getsource(go._cap_leaf_volume_in_meals)
    assert "LEAF_VOLUME_CAP_G / float(grams)" in src and "_LEAF_TOKENS" in src


def test_no_se_toco_el_factor():
    """Igual que en `P1-CAP-FALLBACK-MISMO-ALIMENTO`: esto es la escritura de raw, no la aritmética.
    Si alguien toca el factor del cap, que sea a sabiendas y con su propia medición."""
    src = inspect.getsource(go._cap_leaf_volume_in_meals)
    assert "new_ing = _resc(ing, factor)" in src, "cambió cómo se calcula la línea capada"
