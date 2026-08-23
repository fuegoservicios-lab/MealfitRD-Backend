"""G01: vocabulario alimentario beta comparte backstop y catálogo cerrado."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import graph_orchestrator as go


BACKEND_ROOT = Path(__file__).resolve().parents[1]
CATALOG_SNAPSHOT = BACKEND_ROOT / "tests" / "master_ingredients_names_2026_08_23.json"

_NEW_FOODS = [
    ("gluten", "bocadillo", "Bocadillo"),
    ("gluten", "baguette", "Baguette"),
    ("gluten", "coca", "Coca"),
    ("gluten", "mollete", "Mollete"),
    ("gluten", "torrija", "Torrija"),
    ("gluten", "fideua", "Fideuá"),
    ("gluten", "migas", "Migas"),
    ("gluten", "bolillo", "Bolillo"),
    ("gluten", "telera", "Telera"),
    ("gluten", "concha", "Concha"),
    ("gluten", "birote", "Birote"),
    ("gluten", "empanizado", "Empanizado"),
    ("mariscos", "sepia", "Sepia"),
]


@pytest.fixture
def catalogo_inyectado(monkeypatch):
    import shopping_calculator as sc

    rows = [
        {"name": display, "price_per_lb": 1, "price_per_unit": 0}
        for _clase, _term, display in _NEW_FOODS
    ] + [{"name": "Arroz blanco", "price_per_lb": 1, "price_per_unit": 0}]
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.setattr(sc, "_verified_ingredients_only_enabled", lambda: True)
    monkeypatch.setattr(sc, "get_master_ingredients", lambda *a, **k: list(rows))
    go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()
    yield
    go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()


@pytest.mark.parametrize(
    ("clase", "termino", "display"),
    _NEW_FOODS,
    ids=[term for _clase, term, _display in _NEW_FOODS],
)
def test_cada_alta_bloquea_backstop_y_catalogo_inyectado(
    catalogo_inyectado,
    clase: str,
    termino: str,
    display: str,
) -> None:
    meal = {"name": "Sonda", "ingredients": [f"100 g de {display}"]}
    violations = go.clinical_backstop_for_meal(meal, allergies=[clase])
    assert violations, f"{termino}: el backstop no lo reconoce como {clase}"

    block = go._get_verified_catalog_instruction(
        {"country": "ES", "allergies": [clase]}
    )
    assert display not in block, f"{termino}: el catálogo cerrado todavía lo ofrece"
    assert "Arroz blanco" in block


@pytest.mark.parametrize(
    "display",
    ["Pan", "Tostada", "Harina de trigo", "Galleta", "Espagueti"],
)
def test_cinco_controles_do_siguen_generando_una_violacion(display: str) -> None:
    plan = {
        "days": [{"meals": [{"name": "Control DO", "ingredients": [f"100 g de {display}"]}]}]
    }
    assert len(go._scan_allergen_violations(plan, ["gluten"])) == 1


def test_ninguna_alta_colisiona_con_las_347_filas_vivas_snapshot() -> None:
    names = json.loads(CATALOG_SNAPSHOT.read_text(encoding="utf-8"))
    assert len(names) == 347
    collisions = {}
    for _clase, term, _display in _NEW_FOODS:
        hits = [
            name
            for name in names
            if go._sinonimo_alimento_casa(go._norm_declaracion(name), term)
        ]
        if hits:
            collisions[term] = hits
    assert collisions == {}


@pytest.mark.parametrize(
    ("term", "inocuo"),
    [
        ("bocadillo", "Embocadillado"),
        ("coca", "Cocada"),
        ("migas", "Hormigas"),
        ("telera", "Pastelera"),
        ("concha", "Conchada"),
        ("sepia", "Sepiado"),
        ("empanizado", "Desempanizado"),
    ],
)
def test_altas_no_casan_dentro_de_otra_palabra(term: str, inocuo: str) -> None:
    assert not go._sinonimo_alimento_casa(go._norm_declaracion(inocuo), term)


def test_mutacion_quitar_bocadillo_pone_el_guard_rojo() -> None:
    assert "bocadillo" in go._ALLERGEN_SYNONYMS["gluten"]
    meal = {"name": "Sonda", "ingredients": ["Bocadillo con jamón"]}
    assert go.clinical_backstop_for_meal(meal, allergies=["gluten"])


def test_pfix_marker_cierra_g01() -> None:
    app = (BACKEND_ROOT / "app.py").read_text(encoding="utf-8")
    assert "P0-COUNTRY-ALLERGEN-FOOD-VOCAB · 2026-08-23" in app
