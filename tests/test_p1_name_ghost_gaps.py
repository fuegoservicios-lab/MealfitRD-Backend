# -*- coding: utf-8 -*-
"""[P1-NAME-GHOST-GAPS · 2026-09-06] El nombre cita un alimento que no está en la lista.

`nombre_no_corresponde` es la segunda bolsa del juez culinario: **40 violaciones en 7 días**, 19
de ellas el 05-sep. «Papas a la parrilla con tofu y ensalada de aceitunas» sin papa y sin
aceituna en `ingredients[]`; «Casabe con huevo, Aguacate y mantequilla de maní» sin aguacate. Ni
se cuentan en los macros ni se compran: la receta manda cortar algo que el usuario no tiene.

*Antes de construir la maquinaria, busca la maquinaria.* El pase ya existía —los ghosts de
carbo, fruto seco, fruta y vegetal— y probándolo contra los casos reales resultó que los frutos
secos **sí** se materializaban («…y pistachos» añade su línea) mientras que papa, maíz, aguacate
y aceituna no tenían entrada. Cuentas medidas del token ausente en los planes vivos:
aguacate 7, aceitunas 4, papas 3, maíz 2.

Las porciones son deliberadamente modestas (30 g de aguacate ≈ 48 kcal, del mismo orden que el
tope de ≤60 kcal del veg-guard): el objetivo es que la receta y la compra dejen de mentir, no
rehacer los macros del plato.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import graph_orchestrator as go  # noqa: E402


def _dias(nombre, ingredientes, pasos):
    return [{"day": 1, "meals": [{"meal": "Almuerzo", "name": nombre,
                                  "ingredients": list(ingredientes), "recipe": list(pasos)}]}]


def _correr(nombre, ingredientes, pasos):
    days = _dias(nombre, ingredientes, pasos)
    n = go._add_missing_recipe_step_carbs(days, db=None, allergies=None)
    return n, days[0]["meals"][0]["ingredients"]


# ── los cuatro huecos medidos ────────────────────────────────────────────────────────────────
def test_papa_y_aceituna_del_caso_real():
    """El plato del 05-sep, tal cual: el nombre promete papas y aceitunas y la lista no las trae."""
    n, ings = _correr("Papas a la parrilla con tofu y ensalada de aceitunas",
                      ["150 g de tofu", "2 tazas de lechuga"],
                      ["Mise en place: rebana las aceitunas y corta las papas.", "Montaje: sirve."])
    assert n == 2
    blob = " ".join(ings).lower()
    assert "papa" in blob and "aceituna" in blob


def test_aguacate_nombrado_pero_ausente():
    n, ings = _correr("Casabe con huevo, Aguacate y mantequilla de maní",
                      ["2 huevos", "30 g de casabe"],
                      ["Montaje: corona con el aguacate en láminas."])
    assert n == 1 and any("aguacate" in str(i).lower() for i in ings)


def test_maiz_en_grano():
    n, ings = _correr("Bowl de maíz con queso",
                      ["40 g de queso blanco"],
                      ["Montaje: mezcla el maíz con el queso."])
    assert n == 1 and any("maíz" in str(i).lower() or "maiz" in str(i).lower() for i in ings)


# ── los excludes: sin ellos el arreglo inventa comida ────────────────────────────────────────
@pytest.mark.parametrize("nombre, ings, pasos", [
    ("Arepitas de maíz con queso", ["70 g de harina de maíz", "40 g de queso blanco"],
     ["Mise en place: mezcla la harina de maíz con agua."]),
    ("Ensalada aliñada", ["2 tazas de lechuga", "1 cda de aceite de oliva"],
     ["Montaje: aliña con el aceite de oliva."]),
    ("Tostada con aceite de aguacate", ["2 rebanadas de pan integral", "1 cdta de aceite de aguacate"],
     ["Montaje: unta el aceite de aguacate."]),
])
def test_el_derivado_no_materializa_el_alimento(nombre, ings, pasos):
    """«harina de maíz» no es maíz en grano, «aceite de oliva» no es una aceituna y «aceite de
    aguacate» no es un aguacate. Sin estos excludes, cada plato con harina de maíz recibiría
    además 80 g de maíz que nadie va a cocinar."""
    n, salida = _correr(nombre, ings, pasos)
    assert n == 0, f"materializó de más: {salida}"
    assert salida == ings


def test_no_duplica_lo_que_ya_esta():
    n, ings = _correr("Ensalada con aguacate",
                      ["30 g de aguacate", "2 tazas de lechuga"],
                      ["Montaje: corona con el aguacate."])
    assert n == 0 and len(ings) == 2


def test_la_papa_no_se_confunde_con_la_batata():
    """`batata` tiene su propia entrada y su propia porción; «papa dulce» es batata, no papa."""
    n, ings = _correr("Batata asada", ["150g de batata"], ["Montaje: sirve la batata."])
    assert n == 0, f"la batata disparó el ghost de papa: {ings}"


# ── la tabla nueva comparte el motor, no lo duplica ──────────────────────────────────────────
def test_las_grasas_van_por_el_mismo_bucle():
    """Una tabla nueva no necesita un pase nuevo: si se separara, tendría su propio scan de
    alérgenos y sus propios excludes, y volveríamos a tener dos verdades sobre lo mismo."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.find("_ghosts = (_STEP_CARB_GHOSTS")
    assert i > 0, "cambió la composición de la tabla de ghosts"
    assert "_STEP_FAT_GHOSTS" in src[i:i + 700], "las grasas no entran en el bucle compartido"


def test_el_knob_apaga_solo_las_grasas(monkeypatch):
    monkeypatch.setattr(go, "RECIPE_STEP_FAT_GUARD_ENABLED", False)
    n, ings = _correr("Casabe con Aguacate", ["30 g de casabe"],
                      ["Montaje: corona con el aguacate."])
    assert n == 0 and ings == ["30 g de casabe"]
    # …y el resto del pase sigue vivo con el knob de grasas apagado
    n2, ings2 = _correr("Bowl de avena", ["1 taza de leche"],
                        ["Mise en place: mide la avena."])
    assert n2 == 1 and any("avena" in str(i).lower() for i in ings2)
