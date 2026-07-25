"""[P1-UNIT-NOUN-NOT-FOOD · 2026-07-25] Un ingrediente que se compra y ningún paso usa.

Plan vivo `1d3c6643`, cena del día 1: la comida se llama *"…con Salsa de Lechosa y Puré de Mapuey
y **Queso de Hoja**"*, lista `15g de queso de hoja cocido`, y **ningún paso ni el montaje dicen
qué hacer con él**. El usuario lo compra, lo paga y le cuenta en los macros.

`_ensure_ingredients_used_in_recipe` (P2-RECIPE-REVERSE-COHERENCE) existe justo para eso y no lo
vio: el chequeo de uso es `any(stem in pasos)`, y el stem `hoja` casaba con *"cocina al vapor con
la **hoja de laurel**"* — otro ingrediente del mismo plato. `hoja` es unidad, no alimento.

## Por qué esta regla y no una más ancha

Medido sobre 518 líneas de ingrediente de 8 planes reales ANTES de tocar el pase:

    regla                                          veredictos que cambian
    quitar sustantivos de unidad  (ésta)           1   ← sólo el caso real
    exigir >=2 tokens de identidad                41   ← 40 falsos positivos
    enmascarar frases de otros ingredientes        1   ← y NO era este caso

⚠️ La regla de "≥2 tokens" habría marcado como no-usados `1 cdta de canela en polvo` (el paso dice
"espolvorea canela"), `lechuga romana`, `4 huevos enteros`, `yogurt griego` y `3 rebanadas de pan
integral familiar` — este último es **exactamente** el falso positivo que cerró
P1-STEM-SHORT-FOOD-NOUN, o sea que lo habría regresado. Este pase lleva cinco P-fixes conteniendo
falsos positivos; endurecerlo a ojo es la forma de reabrirlos todos.
"""
import pytest

import graph_orchestrator as go


def _meal_cena_cerdo():
    """La cena del plan vivo, recortada a lo que importa."""
    return {
        "name": "Costilla de Cerdo al Vapor con Salsa de Lechosa y Puré de Mapuey y Queso de Hoja",
        "ingredients": ["75g de costilla de cerdo", "1 mapuey mediano (200g)",
                        "½ hoja de laurel", "15g de queso de hoja cocido", "Sal al gusto"],
        "recipe": [
            "Mise en place: separa las costillas. Pela el mapuey y córtalo en trozos.",
            "El Toque de Fuego: coloca las costillas en una vaporera con la hoja de laurel, "
            "sal y pimienta. Cocina al vapor 30 minutos. Hierve el mapuey y haz puré.",
            "Montaje: coloca el puré de mapuey como base y acomoda las costillas al lado."],
    }


def test_detecta_el_queso_que_ningun_paso_usa():
    meal = _meal_cena_cerdo()
    assert go._ensure_ingredients_used_in_recipe(meal) >= 1
    blob = " ".join(str(s) for s in meal["recipe"]).lower()
    assert "queso de hoja" in blob, meal["recipe"]


def test_el_laurel_NO_se_marca_como_no_usado():
    """`hoja de laurel` sí aparece en los pasos: el pase no puede inventarle un paso."""
    meal = _meal_cena_cerdo()
    go._ensure_ingredients_used_in_recipe(meal)
    nuevos = [s for s in meal["recipe"] if "complemento" in str(s).lower()]
    assert not any("laurel" in str(s).lower() for s in nuevos), nuevos


def test_idempotente():
    meal = _meal_cena_cerdo()
    assert go._ensure_ingredients_used_in_recipe(meal) >= 1
    assert go._ensure_ingredients_used_in_recipe(meal) == 0


# ───────────── lo que NO puede regresar (los 5 falsos positivos ya cerrados) ─────────────

@pytest.mark.parametrize("ingrediente,paso", [
    # el caso exacto de P1-STEM-SHORT-FOOD-NOUN
    ("3 rebanadas de pan integral familiar", "Tuesta las lonjas de pan por ambos lados."),
    ("1 cdta de canela en polvo", "Espolvorea canela por encima antes de servir."),
    ("1½ tazas de lechuga romana picada", "Mezcla la lechuga con el vinagre y sirve."),
    ("4 huevos enteros", "Bate los huevos y viértelos en la sartén."),
    ("1¼ tazas de yogurt griego sin azúcar", "Coloca el yogurt en un bowl y añade la fruta."),
    ("85 g de chivo (pulpa magra)", "Guisa el chivo por 40 minutos hasta que ablande."),
    ("1½ tomates medianos", "Agrega el tomate picado al sofrito."),
])
def test_no_reabre_falsos_positivos(ingrediente, paso):
    meal = {"name": "Plato", "ingredients": [ingrediente], "recipe": ["Mise en place: prepara.", paso]}
    assert go._ensure_ingredients_used_in_recipe(meal) == 0, meal["recipe"]


def test_un_alimento_que_ES_la_unidad_no_se_queda_sin_tokens():
    """El fallback: si TODOS los tokens son de unidad, se usan igual — si no, un alimento llamado
    'filete' quedaría sin identidad y el pase le inventaría un paso siempre."""
    meal = {"name": "Plato", "ingredients": ["2 filetes"],
            "recipe": ["Mise en place: sazona.", "Cocina los filetes 5 minutos por lado."]}
    assert go._ensure_ingredients_used_in_recipe(meal) == 0


def test_la_tabla_de_unidades_no_contiene_alimentos():
    """Si alguien mete un alimento aquí, deja de contar como identidad y el pase le inventará un
    paso de complemento en TODOS los platos que lo lleven."""
    for sospechoso in ("queso", "pan", "arroz", "pollo", "huevo", "leche", "papa", "yuca"):
        assert sospechoso not in go._UNIT_NOUN_NOT_FOOD
