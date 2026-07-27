"""[P1-STEP-LABEL-DEDUP · 2026-07-27] El usuario leía "El Toque de Fuego" dos veces en un paso.

## Lo que veía el owner

    El Toque de Fuego: Blanquea el filete de mero en agua hirviendo 1-2 minutos y escúrrelo
    bien antes de marinar (el cítrico solo marina, no cuece). El Toque de Fuego: En una olla
    pequeña, cocina el arroz blanco con ½ taza de agua…

El inyector del blanqueo antepone su frase **con el rótulo incluido** a un paso que ya empezaba
por ese mismo rótulo. Medido: **2 de 530 pasos** de 14 planes vivos, los dos en ceviches.

## La regla

Se conserva la PRIMERA aparición y se retiran las siguientes, **dejando la prosa intacta** — no se
reordena ni se borra contenido. La frase que queda huérfana del rótulo se recapitaliza: sin eso el
paso quedaba como "…pica todo. lava el kale".

tooltip-anchor: P1-STEP-LABEL-DEDUP
"""
from __future__ import annotations

import copy

import pytest

import humanize_ingredients as H

D = H.dedupe_step_label


# ───────────── 1. el caso del owner ─────────────

def test_el_paso_del_ceviche():
    out = D("El Toque de Fuego: Blanquea el filete de mero en agua hirviendo 1-2 minutos. "
            "El Toque de Fuego: En una olla, hierve las rodajas de papa 15-20 minutos.")
    assert out.lower().count("el toque de fuego") == 1
    assert "Blanquea el filete de mero" in out, "no se puede perder prosa"
    assert "hierve las rodajas de papa" in out


def test_conserva_el_rotulo_al_inicio():
    out = D("El Toque de Fuego: A. El Toque de Fuego: B.")
    assert out.startswith("El Toque de Fuego:")


def test_recapitaliza_la_frase_huerfana():
    """Sin esto el paso queda '…pica todo. lava el kale' — minúscula tras punto."""
    out = D("Mise en place: pica todo. Mise en place: lava el kale.")
    assert "Lava el kale" in out and "lava el kale" not in out


@pytest.mark.parametrize("rotulo", ["El Toque de Fuego", "Mise en place", "Montaje"])
def test_cubre_los_tres_rotulos(rotulo):
    out = D(f"{rotulo}: primero. {rotulo}: segundo.")
    assert out.lower().count(rotulo.lower()) == 1


# ───────────── 2. lo que NO debe tocar ─────────────

@pytest.mark.parametrize("paso", [
    "El Toque de Fuego: Sofríe la cebolla 2 minutos.",
    "Montaje: sirve y disfruta.",
    "Mise en place: pica la cebolla y el ají.",
    "Calienta el aceite. El Toque de Fuego: dora el pollo.",   # una sola aparición, aunque no inicial
    "Cocina 5 minutos y sirve.",
])
def test_una_sola_aparicion_queda_intacta(paso):
    assert D(paso) == paso


def test_idempotente():
    una = D("Montaje: A. Montaje: B.")
    assert D(una) == una


@pytest.mark.parametrize("basura", [None, 123, "", "   ", []])
def test_fail_safe(basura):
    D(basura)


# ───────────── 3. conectado al pipeline ─────────────

def test_corre_dentro_de_humanize_plan_ingredients():
    """Ancla 'código presente, efecto ausente'."""
    plan = {"days": [{"day": 1, "meals": [{
        "name": "Ceviche", "meal": "Almuerzo",
        "ingredients": ["170 g de filete de mero"],
        "recipe": ["El Toque de Fuego: Blanquea el mero. El Toque de Fuego: hierve la papa.",
                   "Montaje: sirve."],
    }]}]}
    out = H.humanize_plan_ingredients(copy.deepcopy(plan))
    pasos = out["days"][0]["meals"][0]["recipe"]
    assert pasos[0].lower().count("el toque de fuego") == 1, pasos[0]
    assert "Blanquea el mero" in pasos[0] and "hierve la papa" in pasos[0].lower()
