# -*- coding: utf-8 -*-
"""[P1-CLASH-HUEVO-Y-VIVERES · 2026-09-09] El vocabulario listaba las PREPARACIONES, no la cosa.

## De dónde sale

El dueño miró su plan vivo `0871ea93` y preguntó por un plato del desayuno:

    «Plátano verde horneado con huevo, queso fresco y mango»
    — ¿y el mango en esta combinación no es raro?

Lo era. Y el sistema tenía desde julio un autofix determinista para exactamente eso
(`_fruit_savory_autofix`), encendido, cuyo docstring pone **«huevo+mango» como PRIMER ejemplo
canónico** del pareo que corrige.

No saltó. Medido: `FRUIT_SAVORY_AUTOFIX_ENABLED` a True, `mango` presente en
`_SWEET_DOMINANT_FRUITS`, y aun así `_meal_has_sweet_savory_clash` devolvía **False**.

La causa: `_SAVORY_CLASH_TOKENS` listaba `revoltillo` y `revuelto` —las PREPARACIONES del
huevo— y no `huevo`. El plato decía «huevo» a secas, así que no casaba con nada.

**Una lista que enumera formas en vez de la cosa deja fuera la forma que nadie escribió.** Es la
misma familia que ya pagó `P1-DIET-CANON-SSOT` (la tabla del filtro olvidaba `vegetariana` y
`vegana` y servía pollo a vegetarianas) y `P1-PANTRY-NAME-RESOLUTION` (`"2 huevos"` contra la
fila `Huevo` devolvía éxito sin descontar).

## Por qué la ampliación se queda corta a propósito

Se añaden `huevo` y las bases saladas criollas que el vocabulario nunca tuvo (`mangu`,
`platano verde`, `tostones`, `mofongo`). **No** se añaden:

  · `platano` a secas — el maduro es dulce por definición y va con salado legítimamente
    (plátano maduro al vapor con sardinas es cocina real).
  · `casabe` — casabe con mantequilla de maní y fruta es un desayuno dominicano normal.

Ampliar de más convierte un guard en ruido, y un guard ruidoso acaba apagado. Los controles
negativos de abajo son la frontera, no un adorno.
"""
import pytest

import graph_orchestrator as go


#: El caso del dueño, textual, y sus hermanos de la misma familia.
DEBEN_MARCARSE = (
    "Plátano verde horneado con huevo, queso fresco y mango",
    "Mangú con huevo y mango",
    "Tostones con huevo y guayaba",
    "Mofongo con piña",
    "Revoltillo criollo con mango",          # el que YA funcionaba: no se rompe
    "Arroz blanco con piña",                 # idem
)

#: La frontera. Cada uno de estos es cocina legítima y marcarlo sería ruido.
NO_DEBEN_MARCARSE = (
    "Casabe con mantequilla de maní, lechosa y yogurt",   # desayuno dominicano normal
    "Yogurt con mango y granola",                          # la fruta va con yogur, dice el propio autofix
    "Plátano maduro al vapor con sardinas",                # el maduro SÍ va con salado
    "Pollo con piña a la parrilla",                        # exclusión vigente desde P1-MENU-COHERENCE-1
    "Batido de mango con leche",
    "Ensalada de espinacas con huevo",                     # salado sin fruta dulce
    "Mangú con los tres golpes",                           # sin fruta: jamás es clash
)


@pytest.mark.parametrize("nombre", DEBEN_MARCARSE)
def test_marca_el_pareo_chocante(nombre):
    assert go._meal_has_sweet_savory_clash({"name": nombre}) is True, (
        f"«{nombre}» debería detectarse como fruta dulce sobre base salada")


@pytest.mark.parametrize("nombre", NO_DEBEN_MARCARSE)
def test_no_marca_lo_que_es_cocina_legitima(nombre):
    assert go._meal_has_sweet_savory_clash({"name": nombre}) is False, (
        f"«{nombre}» es cocina legítima; marcarlo convierte el guard en ruido y acaba apagado")


def test_el_vocabulario_tiene_el_INGREDIENTE_no_solo_sus_preparaciones():
    """El defecto exacto, anclado: si alguien vuelve a dejar sólo `revoltillo`/`revuelto`, esto cae.

    El docstring de `_fruit_savory_autofix` promete «huevo+mango»; esta prueba exige que el
    vocabulario pueda cumplirlo.
    """
    assert "huevo" in go._SAVORY_CLASH_TOKENS, (
        "`huevo` salió del vocabulario salado: el autofix vuelve a prometer en su docstring un "
        "caso («huevo+mango») que no puede detectar")
    for base in ("mangu", "platano verde", "tostones", "mofongo"):
        assert base in go._SAVORY_CLASH_TOKENS, f"se perdió la base salada criolla {base!r}"


def test_el_docstring_del_autofix_sigue_prometiendo_este_caso():
    """Si alguien quita «huevo+mango» del docstring, que sea a sabiendas de que era el caso real."""
    import inspect

    src = inspect.getsource(go._fruit_savory_autofix)
    assert "huevo+mango" in src, (
        "el docstring dejó de nombrar el caso que originó este P-fix; si fue deliberado, "
        "actualiza también este test contando por qué")


def test_el_autofix_sustituye_de_verdad_y_es_idempotente():
    """El detector solo señala; lo que el usuario ve lo cambia el autofix."""
    plato = {
        "name": "Plátano verde horneado con huevo, queso fresco y mango",
        "ingredients": ["½ plátano verde mediano", "2 huevos", "½ mango", "15 g de queso blanco"],
        "ingredients_raw": ["½ plátano verde mediano", "2 huevos", "½ mango", "15 g de queso blanco"],
    }
    dias = [{"meals": [plato]}]
    n = go._fruit_savory_autofix(dias, form_data={"dislikes": [], "allergies": ["Ninguna"]})
    assert n == 1, "el autofix no sustituyó la fruta"
    nombre = dias[0]["meals"][0]["name"].lower()
    assert "mango" not in nombre and "aguacate" in nombre
    assert not any("mango" in str(i).lower() for i in dias[0]["meals"][0]["ingredients"]), (
        "la fruta salió del nombre pero se quedó en los ingredientes: la lista de compras la "
        "seguiría comprando")

    # segunda vuelta: ya no hay nada que arreglar
    assert go._fruit_savory_autofix(dias, form_data={"dislikes": [], "allergies": ["Ninguna"]}) == 0
