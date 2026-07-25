"""[P1-PHANTOM-RAW-PARITY · 2026-07-24] La invariante que casi convierte dos arreglos en medio.

`P1-PHANTOM-INGREDIENT` y `P1-COOKED-GRAIN-DRY` nacieron escribiendo sobre `meal["ingredients"]`
y tocando `ingredients_raw` **sólo cuando las dos listas estaban alineadas por índice**. Al mirar
el plan vivo `732588f8` para diseñar el dedupe apareció el dato que lo invalidaba:

    Casabe Tropical     ingredients = 4 líneas    ingredients_raw = 5 líneas
    Revoltillo          ingredients = 8 líneas    ingredients_raw = 9 líneas
    Bowl Poke           11 y 11 … pero los índices 8/9/10 en distinto orden

Las dos listas **no** son paralelas: `_restore_display_from_raw_orphans` (P1-DISPLAY-RESTORE-FROM-RAW)
las reconcilia por *stem* y trata `len(raw) > len(ings)` como estado esperado.

Y el consumidor que importa lee **raw primero**:

    shopping_calculator.py:3764   ingredients = meal.get("ingredients_raw") or meal.get("ingredients") or []
    shopping_calculator.py:9740   ingredients = meal.get("ingredients_raw") or meal.get("ingredients", [])

O sea: el guard de alineación se saltaba exactamente las comidas desalineadas, que son las que
tienen el problema. El fantasma habría quedado *visible pero sin comprarse* y el arroz habría
seguido comprándose 2.76× — media reparación cada uno, que es el defecto original.

Este archivo ancla el contrato transversal, no el comportamiento de cada pase (eso vive en
`test_p1_phantom_ingredient.py` y `test_p1_cooked_grain_dry.py`).
"""
from pathlib import Path

import pytest

import graph_orchestrator as go
import shopping_calculator as sc


def test_el_consumidor_real_lee_raw_primero():
    """Si esto cambia, el resto de este archivo deja de tener sentido — que se entere el test."""
    src = Path(sc.__file__).read_text(encoding="utf-8")
    assert 'meal.get("ingredients_raw") or meal.get("ingredients")' in src, (
        "la lista de compras resuelve sobre `ingredients_raw`; ese es el motivo de la paridad"
    )


def test_ningun_pase_condiciona_la_escritura_de_raw_a_la_alineacion():
    """Guard prohibido: `len(raw) == len(ings)` como condición para escribir raw."""
    src = Path(go.__file__).read_text(encoding="utf-8")
    for fn in ("_repair_declared_but_unlisted_ingredients", "_normalize_cooked_grain_lines"):
        i = src.index(f"def {fn}")
        body = src[i:i + 4200]
        assert "raw_aligned" not in body, (
            f"{fn} volvió a condicionar la escritura de raw a la alineación por índice"
        )


@pytest.fixture
def _catalogs(monkeypatch):
    monkeypatch.setattr(go, "_PHANTOM_CATALOG_INDEX_CACHE",
                        {"guanabana": "Guanábana", "avena": "Avena"}, raising=False)
    monkeypatch.setattr(go, "_COOKED_CATALOG_KCAL_CACHE",
                        {"arroz blanco": 358.6, "arroz": 358.6}, raising=False)
    yield
    monkeypatch.setattr(go, "_PHANTOM_CATALOG_INDEX_CACHE", None, raising=False)
    monkeypatch.setattr(go, "_COOKED_CATALOG_KCAL_CACHE", None, raising=False)


def test_ambos_pases_escriben_raw_con_listas_desalineadas(_catalogs):
    """El caso real: raw más larga que el display (Casabe: 4 vs 5)."""
    meal = {
        "name": "Plato desalineado",
        "ingredients": ["¼ taza de avena", "65g de arroz blanco cocido"],
        "ingredients_raw": ["¼ taza de avena", "65g de arroz blanco cocido", "Sal al gusto"],
        "recipe": ["Mise en place: 30 g de pulpa de guanábana en trozos."],
    }
    days = [{"day": 1, "meals": [meal]}]
    go._repair_declared_but_unlisted_ingredients(days)
    go._normalize_cooked_grain_lines(days)

    raw_blob = " | ".join(meal["ingredients_raw"]).lower()
    assert "guanábana" in raw_blob, "el fantasma tiene que llegar a la lista que se compra"
    assert "arroz blanco crudo" in raw_blob, "el arroz tiene que comprarse en gramos secos"
    assert "cocido" not in raw_blob
