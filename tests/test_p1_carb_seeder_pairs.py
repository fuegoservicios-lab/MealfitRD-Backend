"""[P1-CARB-SEEDER-PAIRS · 2026-07-27] Una sola base por día y dos comidas que la llevan.

## Lo que veía el owner

    Lunes   desayuno "Revoltillo de Claras con Papa" (299 g) + cena "Papa Rellena"  (300 g)
    Martes  almuerzo "Papas Rellenas al Horno"       (450 g) + cena "Tortilla de Papas" (374 g)
    Miércoles  yautía en almuerzo Y cena

824 gramos de papa en un día.

## La causa: el mismo contrato roto que la fruta, con la última categoría que quedaba

El seeder repartía:

    veggie_i + veggie_ib   -> DOS por día
    fruit_i  + fruit_ib    -> DOS por día  (P1-FRUIT-SEEDER-GATE-CONTRACT, 2026-07-26)
    carb_i                 -> UNO

Pero un día tiene almuerzo Y cena y ambos llevan base. Con una sola asignada, el day-gen la usa
en las dos. **No es un fallo del modelo: la asignación no daba para otra cosa** — exactamente la
lección de la fruta ("un modelo mejor se estrella igual").

Medido sobre 23 días de 8 planes vivos: **9 días (39%)** repiten una base en 2+ comidas. En el
plan del owner, 3 de 3.

## Por qué no un gate de rechazo

Con el 39% de días afectados, un gate dispararía regeneraciones completas — y hoy se midió lo que
cuestan (un rechazo = esqueleto + 3-4 días + revisión otra vez). Se previene en el sembrador, que
es determinista y no cuesta ni una llamada LLM.

## Coste en la lista de compras

`_rotate_pairs` reparte día i → (carbos[i], carbos[i+1]), así que 4 bases cubren 3 días con dos
por día en vez de comprar 6. Es el mismo reparto conservador que ya usa la fruta, y por eso se
EXTRAJO el helper en vez de copiarlo: dos implementaciones del mismo reparto divergen.

tooltip-anchor: P1-CARB-SEEDER-PAIRS
"""
from __future__ import annotations

import pytest

import ai_helpers as a


# ───────────── 1. el reparto ─────────────

def test_cada_dia_recibe_dos_bases_distintas():
    pares = a._rotate_pairs(["Papa", "Arroz", "Yautía", "Yuca"])
    assert pares is not None and len(pares) == 3
    for i, (p1, p2) in enumerate(pares):
        assert p1 != p2, f"el día {i} recibió la misma base dos veces: {p1}"


def test_las_bases_se_REUTILIZAN_entre_dias():
    """El punto que protege la lista de compras: 4 bases cubren 3 días con dos por día, no 6."""
    pares = a._rotate_pairs(["Papa", "Arroz", "Yautía", "Yuca"])
    distintas = {x for par in pares for x in par}
    assert len(distintas) <= 4, f"se están pidiendo {len(distintas)} bases distintas: {distintas}"


def test_con_tres_bases_tambien_funciona():
    pares = a._rotate_pairs(["Papa", "Arroz", "Yautía"])
    assert all(p1 != p2 for p1, p2 in pares), pares


@pytest.mark.parametrize("entrada", [None, [], ["Papa"], ["", "  "]])
def test_sin_material_suficiente_devuelve_None(entrada):
    """El caller cae a su fallback en vez de inventar un pool."""
    assert a._rotate_pairs(entrada) is None


# ───────────── 2. la fruta sigue igual (helper compartido) ─────────────

def test_la_fruta_sigue_repartiendose_en_pares():
    pares = a._rotate_fruit_pairs(["Guineo", "Mango", "Lechosa", "Níspero"])
    assert pares is not None and len(pares) == 3
    assert all(f1 != f2 for f1, f2 in pares), pares


def test_una_sola_implementacion_del_reparto():
    """Ancla de la CLASE. Copiar la rotación en vez de extraerla es cómo se acaba con dos
    predicados que divergen — P1-PANTRY-GATE-SSOT y P1-CLOSER-INTO-MONTAJE costaron eso."""
    import inspect
    src = inspect.getsource(a._rotate_fruit_pairs)
    assert "_rotate_pairs(" in src, (
        "`_rotate_fruit_pairs` debe reutilizar `_rotate_pairs`, no llevar su propia rotación"
    )


# ───────────── 3. el prompt recibe las dos bases ─────────────

def test_el_prompt_formatea_con_las_dos_bases():
    """Si falta un placeholder, `.format` revienta con KeyError y NO se genera ningún plan."""
    p = a.get_deterministic_variety_prompt("", {"mainGoal": "gain_muscle"}, None)
    assert p and len(p) > 500
    assert "{carb_" not in p, "quedó un placeholder sin sustituir en el prompt"


def test_el_prompt_prohibe_repetir_la_base_el_mismo_dia():
    from prompts.preferences import DETERMINISTIC_VARIETY_PROMPT as T
    assert "{carb_0b}" in T and "{carb_1b}" in T and "{carb_2b}" in T, (
        "el prompt no recibe la 2ª base: el day-gen volverá a repetir la única que tiene"
    )
    assert "NUNCA la misma base dos veces el mismo día" in T, (
        "falta la instrucción explícita; asignar la 2ª base sin decir que no repita no basta"
    )


def test_la_regla_de_bases_transformables_menciona_las_dos():
    from prompts.preferences import DETERMINISTIC_VARIETY_PROMPT as T
    i = T.index("BASES A TRANSFORMAR")
    assert "{carb_0b}" in T[:i], (
        "la regla de bases transformables debe listar las dos del día, o se contradice con la "
        "asignación de arriba"
    )
