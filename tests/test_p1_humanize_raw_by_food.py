# -*- coding: utf-8 -*-
"""[P1-HUMANIZE-RAW-BY-FOOD · 2026-09-07] El último pase que cruzaba las dos listas por índice.

`sync_recipe_steps_to_household` armoniza las unidades entre lista y pasos («190 g de yogurt» →
«1 taza de yogurt (190 g)»). Para saber qué escribir cruzaba `ingredients` con `ingredients_raw`
**por posición**, con la sola guarda de que midieran igual.

Eso ya estaba medido y resuelto en esta casa, pero no llegó aquí:

- `P2-RAW-PAIR-BY-FOOD · 2026-07-29`: «"Mismo largo" NUNCA fue evidencia de "mismo orden": el
  reconciliador reconstruye raw como `[conservadas] + [añadidas]`, o sea preserva el largo y
  cambia el orden. Medido: el 93,5 % de las comidas tiene largos iguales pero solo el 48,1 % de
  ESAS son paralelas por índice.»
- `P1-DM-RAW-BY-FOOD · 2026-07-31` migró el último pase de `graph_orchestrator` a ese contrato y
  lo llamó «el último que quedaba fuera». **No lo era**: éste vive en otro fichero.

Medido hoy sobre la flota: 699 de 1.172 comidas (59,6 %) con las listas desalineadas, y la
función escribía el **plato equivocado** en el paso:

    corta 115 g de lechosa  →  corta 25 g de avena (115 g)
    cortar 110 g de mango   →  cortar 1¼ tazas de yogurt griego sin azúcar (110 g)

Comprobado contra la versión ANTERIOR del módulo sobre las mismas comidas: idéntico. No era
regresión de `P1-STEP-NAME-TAIL`.

El arreglo **adopta el contrato existente** en vez de escribir otro comprobador (la lección de
`P1-DIET-CANON-SSOT`), así que la función sigue armonizando donde antes mentía, en vez de
abstenerse.
"""
import pytest

from humanize_ingredients import _pares_display_raw, sync_recipe_steps_to_household

_PARALELAS_DISPLAY = ["½ cebolla", "1 taza de lechosa", "2 cdas de aceite de oliva"]
_PARALELAS_RAW = ["0.5 cebolla", "115 g de lechosa", "30 g de aceite de oliva"]


# ---------------------------------------------------------------- el emparejador

def test_con_listas_paralelas_empareja_por_indice(monkeypatch):
    """El caso mayoritario y barato: si alinean, se conserva el pareo posicional."""
    pares = _pares_display_raw(_PARALELAS_DISPLAY, _PARALELAS_RAW)
    assert pares == list(zip(_PARALELAS_DISPLAY, _PARALELAS_RAW))


def test_con_listas_ROTADAS_no_empareja_lechosa_con_avena():
    """El caso vivo reducido: raw rotado respecto a display.

    Lo que NO puede pasar es que «115 g de lechosa» quede emparejado con la línea humanizada de la
    avena — que es justo lo que producía el pareo por índice.
    """
    disp = ["25 g de avena", "1 taza de lechosa"]
    raw = ["115 g de lechosa", "25 g de avena"]
    pares = dict((r, h) for h, r in _pares_display_raw(disp, raw))
    assert pares.get("115 g de lechosa") != "25 g de avena", pares


def test_una_linea_ambigua_no_entra_al_mapa():
    """Dos líneas display del mismo alimento: no se adivina cuál. 0 o >1 ⇒ fuera."""
    disp = ["100 g de pollo", "150 g de pollo"]
    raw = ["100 g de pollo"]
    for h, r in _pares_display_raw(disp, raw):
        assert not (r == "100 g de pollo" and h not in disp)


@pytest.mark.parametrize("basura", [None, 12345, "no soy una lista"])
def test_ante_basura_no_revienta(basura):
    """Fail-safe: devuelve algo iterable, nunca lanza — la función que lo usa es best-effort."""
    assert isinstance(_pares_display_raw(basura, ["1 taza de avena"]), list)


# ---------------------------------------------------------------- la función que lo usa

def test_no_escribe_el_alimento_equivocado_con_listas_rotadas():
    """El síntoma que se vio en la flota, reducido a un caso."""
    meal = {
        "ingredients": ["25 g de avena", "1 taza de lechosa"],
        "ingredients_raw": ["115 g de lechosa", "25 g de avena"],
        "recipe": ["Mise en place: corta 115 g de lechosa en cubos."],
    }
    sync_recipe_steps_to_household(meal)
    assert "avena" not in meal["recipe"][0], meal["recipe"][0]
    assert "lechosa" in meal["recipe"][0]


def test_sigue_armonizando_cuando_alinean():
    """La corrección no puede apagar la función donde ya funcionaba."""
    meal = {
        "ingredients": ["1 taza de yogurt griego sin azúcar"],
        "ingredients_raw": ["190 g de yogurt griego sin azúcar"],
        "recipe": ["Mide 190 g de yogurt griego sin azúcar y reserva."],
    }
    assert sync_recipe_steps_to_household(meal) >= 1
    assert "1 taza de yogurt griego sin azúcar (190 g)" in meal["recipe"][0]
    assert "reserva" in meal["recipe"][0]


def test_no_pisa_el_arreglo_de_la_cola_del_nombre():
    """`P1-STEP-NAME-TAIL` sigue vivo bajo el emparejamiento nuevo: nada de «(190 g) azúcar»."""
    meal = {
        "ingredients": ["1 taza de yogurt griego sin azúcar"],
        "ingredients_raw": ["190 g de yogurt griego sin azúcar"],
        "recipe": ["Mide 190 g de yogurt griego sin azúcar y reserva."],
    }
    sync_recipe_steps_to_household(meal)
    assert "(190 g) azúcar" not in meal["recipe"][0]


def test_usa_el_comprobador_de_la_casa_y_no_uno_propio():
    """Ancla estructural: si alguien reimplementa el paralelismo aquí, este test lo dice.

    Escribir un segundo comprobador es cómo nacieron las tres tablas de canonicalización de dieta
    que drifearon (`P1-DIET-CANON-SSOT`). El contrato vive en `graph_orchestrator`.
    """
    import inspect

    import humanize_ingredients

    src = inspect.getsource(humanize_ingredients._pares_display_raw)
    assert "_raw_display_parallel_by_food" in src
    assert "_resolve_line_food_grams" in src
