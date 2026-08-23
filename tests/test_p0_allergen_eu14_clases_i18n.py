"""G02: cuatro clases EU14 atraviesan backstop y catálogo en cinco idiomas."""

from __future__ import annotations

from pathlib import Path

import pytest

import graph_orchestrator as go


BACKEND_ROOT = Path(__file__).resolve().parents[1]

_CASES = [
    ("apio", "es", "Apio", "Apio"),
    ("apio", "en", "Celery", "Apio"),
    ("apio", "fr", "Céleri", "Apio"),
    ("apio", "it", "Sedano", "Apio"),
    ("apio", "pt", "Aipo", "Apio"),
    ("mostaza", "es", "Mostaza", "Mostaza"),
    ("mostaza", "en", "Mustard", "Mostaza"),
    ("mostaza", "fr", "Moutarde", "Mostaza"),
    ("mostaza", "it", "Senape", "Mostaza"),
    ("mostaza", "pt", "Mostarda", "Mostaza"),
    ("sulfitos", "es", "Sulfitos", "Vino"),
    ("sulfitos", "en", "Sulphites", "Vino"),
    ("sulfitos", "fr", "Sulfites", "Vino"),
    ("sulfitos", "it", "Solfiti", "Vino"),
    ("sulfitos", "pt", "Sulfitos", "Vino"),
    ("altramuces", "es", "Altramuz", "Altramuces"),
    ("altramuces", "en", "Lupin", "Altramuces"),
    ("altramuces", "fr", "Lupin", "Altramuces"),
    ("altramuces", "it", "Lupini", "Altramuces"),
    ("altramuces", "pt", "Tremoço", "Altramuces"),
]

_CATALOG = [
    {"name": "Apio", "price_per_lb": 1, "price_per_unit": 0},
    {"name": "Mostaza", "price_per_lb": 0, "price_per_unit": 1},
    {"name": "Vino", "price_per_lb": 0, "price_per_unit": 1},
    {"name": "Altramuces", "price_per_lb": 1, "price_per_unit": 0},
    {"name": "Camarones", "price_per_lb": 1, "price_per_unit": 0},
    {"name": "Arroz blanco", "price_per_lb": 1, "price_per_unit": 0},
]


@pytest.fixture
def catalogo_inyectado(monkeypatch):
    import shopping_calculator as sc

    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.setattr(sc, "_verified_ingredients_only_enabled", lambda: True)
    monkeypatch.setattr(sc, "get_master_ingredients", lambda *a, **k: list(_CATALOG))
    go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()
    yield
    go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()


@pytest.mark.parametrize(
    ("clase", "locale", "declaracion", "alimento"),
    _CASES,
    ids=[f"{clase}-{locale}" for clase, locale, _decl, _food in _CASES],
)
def test_eu14_bloquea_backstop_y_catalogo_en_cinco_idiomas(
    catalogo_inyectado,
    clase: str,
    locale: str,
    declaracion: str,
    alimento: str,
) -> None:
    meal = {"name": "Sonda EU14", "ingredients": [f"100 g de {alimento}"]}
    violations = go.clinical_backstop_for_meal(meal, allergies=[declaracion])
    assert violations, f"{clase}/{locale}: el backstop dejó pasar {alimento}"

    block = go._get_verified_catalog_instruction(
        {"country": "ES", "locale": locale, "allergies": [declaracion]}
    )
    assert alimento not in block, f"{clase}/{locale}: el catálogo todavía ofrece {alimento}"
    assert "Arroz blanco" in block, "el filtro se llevó por delante el control inocuo"


def test_cobertura_enumera_los_catorce_alergenos_del_reglamento() -> None:
    eu14_a_clase = {
        "cereales_con_gluten": "gluten",
        "crustaceos": "mariscos",
        "huevos": "huevo",
        "pescado": "pescado",
        "cacahuetes": "mani",
        "soja": "soya",
        "leche": "lacteos",
        "frutos_de_cascara": "frutos secos",
        "apio": "apio",
        "mostaza": "mostaza",
        "sesamo": "sesamo",
        "sulfitos": "sulfitos",
        "altramuces": "altramuces",
        "moluscos": "mariscos",
    }
    assert len(eu14_a_clase) == 14
    assert set(eu14_a_clase.values()) <= set(go._ALLERGEN_SYNONYMS)


def test_sulfitos_documenta_vehiculos_acotados_y_no_categorias_amplias() -> None:
    vehicles = set(go._ALLERGEN_SYNONYMS["sulfitos"])
    assert {"vino", "vinagre de vino", "orejones", "pasas"} <= vehicles
    assert vehicles.isdisjoint({"fruta", "frutas", "fruta seca", "condimento", "condimentos"})


def test_plural_z_ces_funciona_para_fallback_literal_en_ambas_capas(
    catalogo_inyectado,
    monkeypatch,
) -> None:
    import shopping_calculator as sc

    meal = {"name": "Sonda", "ingredients": ["100 g de Codornices"]}
    assert go.clinical_backstop_for_meal(meal, allergies=["Codorniz"])

    rows = list(_CATALOG) + [
        {"name": "Codornices", "price_per_lb": 1, "price_per_unit": 0},
    ]
    monkeypatch.setattr(sc, "get_master_ingredients", lambda *a, **k: rows)
    go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()
    block = go._get_verified_catalog_instruction(
        {"country": "ES", "allergies": ["Codorniz"]}
    )
    assert "Codornices" not in block


def test_sal_de_apio_no_dispara_mariscos() -> None:
    meal = {"name": "Sonda", "ingredients": ["2 g de Sal de apio"]}
    assert go.clinical_backstop_for_meal(meal, allergies=["Mariscos"]) == []


def test_do_sin_alergias_conserva_byte_identidad(catalogo_inyectado, monkeypatch) -> None:
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()
    antes = go._get_verified_catalog_instruction({"country": "DO"})
    go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()
    despues = go._get_verified_catalog_instruction({"country": "DO", "allergies": []})
    assert antes == despues


def test_pfix_marker_cierra_g02() -> None:
    app = (BACKEND_ROOT / "app.py").read_text(encoding="utf-8")
    assert "P0-ALLERGEN-EU14-CLASES-I18N · 2026-08-23" in app
