"""[P1-GAINMUSCLE-NO-DOUBLE-CARB · 2026-07-26] Pasta y arroz en el mismo plato.

El refill del piso calórico de gain-muscle añade arroz blanco a una comida cuando el día queda
bajo el piso. Elegía la comida **solo por slot** (almuerzo → cena → desayuno), sin mirar si ese
plato ya tenía su base de carbohidrato. Plan vivo `0afa0ed5`, día 1:

    "Res Salteada al Wok con Pasta Integral y Vegetales"
        ¾ taza de pasta integral (peso seco)
        30 g de arroz blanco crudo                       ← añadido por el refill
        paso 4: "💡 Acompaña este plato con el arroz blanco cocido de tus ingredientes
                 para completar las calorías del día."

Pasta y arroz en el mismo almuerzo. Es el simétrico del `CLOSER_NO_DOUBLE_MAIN` de proteínas,
aplicado al carbohidrato.

⚠️ Yo había concluido antes que el arroz "lo escribe el generador desde el principio" porque
`_close_carb_gap_for_day` solo ESCALA ingredientes existentes. Era cierto de ese corrector y
falso del sistema: **este otro sí añade**. Comprobar todos los que pueden escribir, no el
primero que se lee.

## Reorden, no filtro

Las comidas SIN base de carbo van primero; el resto conserva su orden por slot detrás. Si TODAS
tienen base, el comportamiento es el de antes: el piso calórico de gain-muscle es **clínico** y
no se sacrifica por estética — se degrada honesto, igual que el cerrador de proteína.

El arroz NO cuenta como base conflictiva: el refill consolida en una línea de arroz existente
en vez de duplicarla, así que un plato que ya lleva arroz es el destino ideal, no un conflicto.

tooltip-anchor: P1-GAINMUSCLE-NO-DOUBLE-CARB
"""
from __future__ import annotations

import pytest

import graph_orchestrator as g
from constants import strip_accents


def _tiene(meal):
    return g._meal_has_conflicting_carb_base(meal, strip_accents)


# ───────────── 1. el caso vivo ─────────────

def test_el_almuerzo_de_pasta_es_conflictivo():
    meal = {"name": "Res Salteada al Wok con Pasta Integral y Vegetales",
            "ingredients": ["120 g de res en tiras", "¾ taza de pasta integral (peso seco)"]}
    assert _tiene(meal) is True


def test_una_cena_de_pollo_con_coliflor_no_lo_es():
    """El puré de coliflor no es un carbohidrato base: ahí el arroz sí cabría."""
    meal = {"name": "Muslo de Pollo al Horno con Piel Crujiente y Puré de Coliflor",
            "ingredients": ["½ muslo de pollo", "1 cabeza de coliflor mediana"]}
    assert _tiene(meal) is False


# ───────────── 2. la base puede no estar en el título ─────────────

def test_detecta_la_base_en_los_INGREDIENTES():
    """"Bowl criollo de pollo" no dice pasta en el nombre, pero la lleva."""
    meal = {"name": "Bowl Criollo de Pollo", "ingredients": ["1 taza de pasta integral"]}
    assert _tiene(meal) is True


@pytest.mark.parametrize("base", [
    "½ taza de bulgur", "200 g de yuca", "1 casabe", "2 lonjas de pan integral",
    "¾ taza de ñame rallado", "1 plátano maduro", "¼ taza de avena", "150 g de batata",
])
def test_bases_de_carbo_dominicanas(base):
    assert _tiene({"name": "Plato", "ingredients": [base]}) is True


# ───────────── 3. el arroz NO es conflicto ─────────────

def test_un_plato_que_ya_lleva_arroz_no_es_conflicto():
    """El refill consolida en la línea de arroz existente en vez de duplicarla, así que ese
    plato es el DESTINO IDEAL. Marcarlo conflictivo lo mandaría al final de la cola."""
    meal = {"name": "Pollo Guisado con Arroz Blanco",
            "ingredients": ["150 g de pollo", "1 taza de arroz blanco"]}
    assert _tiene(meal) is False


# ───────────── 4. el orden resultante ─────────────

def _orden(meals):
    """Reproduce la clave de ordenación del refill."""
    _slot_rank = {"almuerzo": 0, "cena": 1, "desayuno": 2}
    return sorted(meals, key=lambda mm: (
        1 if _tiene(mm) else 0,
        next((r for s, r in _slot_rank.items()
              if s in strip_accents(str(mm.get("meal", "")).lower())), 3)))


def test_la_comida_sin_base_gana_al_almuerzo_con_pasta():
    almuerzo = {"meal": "Almuerzo", "name": "Res con Pasta Integral",
                "ingredients": ["1 taza de pasta integral"]}
    cena = {"meal": "Cena", "name": "Pollo con Puré de Coliflor", "ingredients": ["coliflor"]}
    assert _orden([almuerzo, cena])[0] is cena


def test_si_TODAS_tienen_base_manda_el_slot_como_antes():
    """El piso calórico es clínico: no se deja de cumplir por estética."""
    almuerzo = {"meal": "Almuerzo", "name": "Res con Pasta", "ingredients": ["pasta integral"]}
    cena = {"meal": "Cena", "name": "Pollo con Yuca", "ingredients": ["yuca"]}
    assert _orden([cena, almuerzo])[0] is almuerzo, "vuelve a mandar el orden por slot"


def test_entre_dos_sin_base_sigue_mandando_el_slot():
    almuerzo = {"meal": "Almuerzo", "name": "Res con Ensalada", "ingredients": ["lechuga"]}
    cena = {"meal": "Cena", "name": "Pollo con Vegetales", "ingredients": ["brocoli"]}
    assert _orden([cena, almuerzo])[0] is almuerzo


# ───────────── 5. bordes ─────────────

@pytest.mark.parametrize("meal", [{}, {"name": None}, {"ingredients": None},
                                  {"name": "Plato", "ingredients": [None, 12]}])
def test_entradas_raras_no_rompen(meal):
    assert isinstance(_tiene(meal), bool)


def test_el_refill_consulta_el_predicado():
    """El helper puede ser correcto y no llamarse nunca — el modo de fallo de
    P1-CAPPED-STAPLE-HONESTY. Se ancla el callsite."""
    from pathlib import Path
    src = Path(g.__file__).resolve().read_text(encoding="utf-8")
    i = src.index("_slot_rank = {\"almuerzo\": 0")
    bloque = src[i:i + 2200]
    assert "_meal_has_conflicting_carb_base(mm" in bloque, \
        "el orden del refill debe consultar el predicado"
