# -*- coding: utf-8 -*-
"""[P1-DURABILITY-FRESH-STATE · 2026-09-05] La tabla de durabilidad resuelve por el ALIMENTO y no miraba la
palabra de estado. Para cuatro pescados eso invertía el resultado, porque su nombre desnudo es a la vez el de la
conserva y el del pescado del mostrador:

    atún fresco / bacalao fresco / arenque fresco / anchoas frescas  →  despensa, 180 días

Consecuencia medida: en un ciclo de 30 días SIN congelador, un plato con atún fresco pasaba el guard el día 25.

Lo que este test también fija es lo que NO se tocó. «arroz cocido» sigue siendo despensa 180 y «claras de huevo»
frío 35 a propósito: el módulo responde «¿cuánto aguanta lo que el usuario COMPRA?», y lo que compra es arroz y
huevos. Tratarlos como sobras de nevera bloquearía platos correctos."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import pantry_durability as pd  # noqa: E402


@pytest.mark.parametrize("nombre", [
    "Atún fresco", "atun fresco", "Bacalao fresco", "Arenque fresco", "Anchoas frescas",
    "Sardinas frescas", "Atún crudo", "Filete de atún fresco",
])
def test_el_pescado_del_mostrador_no_es_una_lata(nombre):
    r = pd.classify(nombre)
    assert r["cls"] == "freezable", f"«{nombre}» → {r}"
    assert r["days_fresh"] == 3, r
    assert r["rule"] == "fresh_state", "el veto del calificativo, no la tabla"


@pytest.mark.parametrize("nombre, dias", [
    ("Atún en agua", 180), ("Atún", 180), ("Sardinas en lata", 180), ("Anchoas", 180),
])
def test_la_conserva_sigue_siendo_conserva(nombre, dias):
    """Sin calificativo de frescura, el nombre desnudo sigue significando lata."""
    r = pd.classify(nombre)
    assert r["cls"] == "pantry" and r["days_fresh"] == dias, r


@pytest.mark.parametrize("nombre, cls, dias", [
    ("arroz cocido", "pantry", 180),      # se compra arroz; se cocina ese día
    ("claras de huevo", "cold", 35),      # se compran huevos; se separan ese día
    ("pollo", "freezable", 3),
    ("salmón fresco", "freezable", 3),    # ya acertaba: «salmon» no tiene regla de conserva
    ("lechuga", "fresh", 3),
])
def test_lo_que_no_se_toca(nombre, cls, dias):
    r = pd.classify(nombre)
    assert r["cls"] == cls, f"«{nombre}» → {r}"
    if dias is not None and nombre != "lechuga":
        assert r["days_fresh"] == dias, r


def test_el_guard_bloquea_el_atun_fresco_en_el_dia_25():
    """El caso vivo completo: ciclo de 30 días, una sola compra, sin congelador."""
    eff = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "none"}}
    req = pd.single_trip_requirements(eff, 25)
    assert req and req["allow_frozen"] is False
    issue = pd.ingredient_issue_beyond_horizon("Atún fresco", 25, allow_frozen=req["allow_frozen"])
    assert issue == "protein_beyond_freeze_window", f"debía bloquear y devolvió {issue!r}"


def test_la_semana_de_frescos_sigue_exenta():
    """El roadmap 2.6 la señala como defecto y no lo es: es la semana de frescos, comprada el día 0."""
    eff = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "none"}}
    assert pd.single_trip_requirements(eff, 3) is None
    assert pd.single_trip_requirements(eff, 6) is None
    assert pd.single_trip_requirements(eff, 7) is not None


def test_todo_plazo_dice_de_donde_sale():
    """Procedencia: un plazo raro se audita sin releer la tabla entera."""
    for nombre in ("Atún fresco", "Arroz", "Pollo", "Ingrediente inexistente xyz"):
        assert pd.classify(nombre).get("rule"), nombre
    assert pd.classify("Ingrediente inexistente xyz")["rule"] == "category_default"
