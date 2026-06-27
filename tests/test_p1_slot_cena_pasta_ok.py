"""[P1-SLOT-CENA-PASTA-OK · 2026-06-27] Decisión de producto (owner): la PASTA/espagueti SÍ va en la
cena dominicana (carbo ligero de digestión rápida; "cenar espaguetis" es común) → NO se bloquea de noche.
Queda inapropiada SOLO en el DESAYUNO. El "arroz de noche" sigue bloqueado, y el cereal/panqueque/waffle
en la cena siguen soft-block ("evítalos a menos que no haya otra alternativa", que casi siempre la hay).

Ancla la decisión: si un refactor vuelve a meter la pasta en la lista de inapropiados de la cena, este
test falla ANTES de que el cambio cultural incorrecto llegue a producción.
"""
from __future__ import annotations

import inspect

import constants as c


def test_pasta_allowed_at_dinner():
    for name in ("Espaguetis con pollo y vegetales", "Pasta criolla con carne molida",
                 "Macarrones con queso y pollo", "Lasaña de vegetales", "Fideos con camarones"):
        assert c.slot_violations_for_meal_name(name, "cena") == [], f"{name} NO debería bloquearse en cena"


def test_pasta_still_blocked_at_breakfast():
    """En el desayuno la pasta sigue inapropiada (bloqueo duro)."""
    v = c.slot_violations_for_meal_name("Espaguetis con pollo", "desayuno")
    assert v and any(x["hard"] for x in v)


def test_rice_still_blocked_at_dinner():
    """El 'arroz de noche' sigue bloqueado (confirmado por el owner)."""
    assert c.slot_violations_for_meal_name("Pollo con arroz blanco", "cena")


def test_breakfast_foods_still_soft_blocked_at_dinner():
    """Cereal/panqueque/waffle en la cena: soft-block (evitar salvo que no haya alternativa)."""
    for name in ("Cereal con leche", "Panqueques con sirope", "Waffles con fruta"):
        v = c.slot_violations_for_meal_name(name, "cena")
        assert v and not any(x["hard"] for x in v), f"{name} debe ser soft-block en cena"


def test_marker_anchor_present():
    assert "P1-SLOT-CENA-PASTA-OK" in inspect.getsource(c)
