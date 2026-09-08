# -*- coding: utf-8 -*-
"""[P1-NIGHTRICE-RAW-BY-FOOD · 2026-09-08] El usuario leía batata y la lista compraba arroz.

`_night_rice_autofix` sustituye el arroz de la CENA por un tubérculo (regla «sin arroz de noche»).
En sus 116 líneas escribía `ings[j]` y **no mencionaba `ingredients_raw` ni una vez**. Su gemelo
`_breakfast_rice_autofix` tenía el bloque idéntico, letra por letra.

## Cómo apareció: mi propia lista a mano estaba ciega

El experimento en positivo del 08-sep (correr cada pase sobre las comidas ya ALINEADAS y ver cuál
deja rota una limpia) lo corrí sobre **once pases que escribí a mano**. Derivar el conjunto por AST
dio **32**, y los dos mayores no estaban en mi lista. Es el mismo error que ese mismo día me mordió
con la lista heredada de ficheros emisores de alertas y con el inventario de índices por grep.

*Una lista escrita a mano hereda el límite de quien la escribió.*

## Por qué no se veía en producción

Las DOS cadenas de generación (`assemble_plan_node`, `finalize_plan_data_coherence`) terminan en
`_reconcile_display_missing_in_raw` / `_reconcile_raw_missing_in_display` /
`_reconcile_display_raw_lines`, que reparan raw. Por eso en los planes vivos hay **0** casos de «lee
batata / compra arroz» (65 sanos, 4 con ambos a propósito — moro con batata).

`finalize_single_meal_recipe_coherence` —swap, chat-modify, regenerar-día, **4 call sites de
producción**— NO tiene ninguno. El defecto sólo existía donde nadie estaba mirando. Y tres de los
cuatro llamadores usan el default `skip_night_rice=False`, así que el pase corre en el swap real.

## Por qué NO se arregló añadiendo los reconciliadores

Medido sobre las 986 comidas alineadas: añadirlos al finalizador baja el daño de 98 a 21 líneas,
**pero en 24 comidas `_reconcile_raw_missing_in_display` devuelve el arroz al display** y pelea con
la regla que el pase acaba de aplicar. Cambia «lee batata, compra arroz» por «lee arroz y batata».

El pase que cambia el display es quien debe llevarse raw — la doctrina de los siete P-fixes de la
familia `raw[idx]`. Sustituye EN EL SITIO (no apendea), así no queda la línea vieja comprando el
arroz retirado.
"""
import inspect

import graph_orchestrator as go


def test_los_dos_pases_de_arroz_escriben_raw():
    """El bloque era idéntico en los dos; un arreglo que sólo toque uno deja el gemelo abierto."""
    for nombre in ("_night_rice_autofix", "_breakfast_rice_autofix"):
        src = inspect.getsource(getattr(go, nombre))
        assert "_sub_dr(m, ing, j, ings," in src, (
            f"{nombre} volvió a escribir sólo `ingredients`: el usuario lee el tubérculo y la "
            f"lista compra el arroz")
        assert "sustituye_display_y_raw" in src, f"{nombre} perdió el import del helper"


def test_el_helper_empareja_por_ALIMENTO_no_por_indice():
    """La regla de la casa tras siete P-fixes: índice sólo con paralelismo VERIFICADO.

    `P2-RAW-PAIR-BY-FOOD` midió que «mismo largo» es cierto el 93,5 % de las veces y verdadero sólo
    el 48,1 %. Emparejar por índice aquí pondría la batata encima de la línea de otro alimento.
    """
    import constants

    src = inspect.getsource(constants.sustituye_display_y_raw)
    assert "_raw_idx_for_display" in src, "volvió el emparejado por índice"
    assert "raw[idx]" not in src


def test_sustituye_en_el_sitio_no_apendea():
    """Apendear dejaría la línea vieja: el usuario leería batata y la lista compraría LAS DOS."""
    import constants

    meal = {"ingredients": ["40 g de arroz blanco crudo"],
            "ingredients_raw": ["40 g de arroz blanco crudo"]}
    nueva = constants.sustituye_display_y_raw(
        meal, "40 g de arroz blanco crudo", 0, meal["ingredients"], "56 g de batata")
    assert nueva == "56 g de batata"
    assert meal["ingredients_raw"] == ["56 g de batata"], (
        "raw debe quedar con UNA línea (la nueva), no con la vieja ni con las dos")


def test_sin_pareja_no_toca_raw():
    """Si el alimento no resuelve o hay ambigüedad, raw se queda como estaba.

    «No escribo» es el fallo seguro; «escribo en la línea equivocada» deja al usuario comprando otra
    cosa, que es peor que no recortar ninguna (la doctrina que `_remove_one_raw_line_by_food` ya
    enunciaba y que el fallback del cap contradecía hasta `P1-CAP-FALLBACK-MISMO-ALIMENTO`).
    """
    import constants

    meal = {"ingredients": ["40 g de arroz blanco crudo"], "ingredients_raw": ["2 huevos"]}
    constants.sustituye_display_y_raw(
        meal, "40 g de arroz blanco crudo", 0, meal["ingredients"], "56 g de batata")
    assert meal["ingredients_raw"] == ["2 huevos"], "escribió la batata sobre la línea del huevo"


def test_meal_sin_raw_no_revienta():
    """Fail-safe: el pase corre en superficies donde `ingredients_raw` puede no existir."""
    import constants

    for meal in ({"ingredients": ["x"]}, {"ingredients": ["x"], "ingredients_raw": None}, None):
        assert constants.sustituye_display_y_raw(meal, "x", 0, ["x"], "y") == "y"


def test_el_helper_vive_FUERA_del_fichero_en_su_techo():
    """`graph_orchestrator.py` está en 53.100 de 53.100 y su test del techo dice que eso «no se
    arregla subiendo el número: se arregla extrayendo». El arreglo son 0 líneas netas."""
    import constants

    assert hasattr(constants, "sustituye_display_y_raw")
    assert "def sustituye_display_y_raw" not in inspect.getsource(go)


def test_la_superficie_de_swap_sigue_sin_reconciliadores_a_proposito():
    """Ancla la decisión medida: si alguien añade los reconciliadores al finalizador, 24 comidas
    recuperan el arroz en el display y la regla «sin arroz de noche» queda anulada en el swap.

    Si un día hacen falta, primero hay que resolver esa pelea — no añadirlos y ver.
    """
    src = inspect.getsource(go.finalize_single_meal_recipe_coherence)
    assert "_reconcile_raw_missing_in_display" not in src, (
        "devuelve al display lo que los pases acaban de quitar: medido en 24 de 986 comidas. "
        "Ver el docstring de `constants.sustituye_display_y_raw`.")
