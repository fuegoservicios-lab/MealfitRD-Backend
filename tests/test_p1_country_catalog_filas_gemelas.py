"""[P1-COUNTRY-CATALOG-FILAS-GEMELAS · 2026-08-23]

El catálogo cerrado no ofrece dos filas para el mismo alimento. Las filas
persistidas se conservan: este guard observa únicamente la instrucción servida
al modelo y crece con el catálogo inyectado.
"""
from collections import defaultdict
from decimal import Decimal

import pytest

import graph_orchestrator as go
from constants import strip_accents


_ROWS = [
    {"name": "Frijoles pintos", "name_en": "Pinto beans", "category": "Despensa",
     "kcal_per_100g": 347, "density_g_per_cup": 193, "density_g_per_unit": None,
     "price_per_lb": 72, "price_per_unit": 127},
    {"name": "Judías pintas", "name_en": "Pinto beans", "category": "Despensa",
     "kcal_per_100g": 347, "density_g_per_cup": None, "density_g_per_unit": None,
     "price_per_lb": 0, "price_per_unit": 0},
    {"name": "Habichuelas blancas", "name_en": "White beans", "category": "Despensa",
     "kcal_per_100g": 333, "density_g_per_cup": 180, "density_g_per_unit": None,
     "price_per_lb": 0, "price_per_unit": 50},
    {"name": "Judías blancas", "name_en": "White beans", "category": "Despensa",
     "kcal_per_100g": 333, "density_g_per_cup": 180, "density_g_per_unit": None,
     "price_per_lb": 0, "price_per_unit": 0},
    {"name": "Queso ricotta", "name_en": "Ricotta cheese", "category": "Lácteos",
     "kcal_per_100g": 151, "density_g_per_cup": 246, "density_g_per_unit": None,
     "price_per_lb": 0, "price_per_unit": 245},
    {"name": "Requesón", "name_en": "Ricotta", "category": "Lácteos",
     "kcal_per_100g": 151, "density_g_per_cup": 246, "density_g_per_unit": None,
     "price_per_lb": 0, "price_per_unit": 0},
    {"name": "Camarones", "name_en": "Shrimp", "category": "Proteínas",
     "kcal_per_100g": 85, "density_g_per_cup": None, "density_g_per_unit": 8,
     "price_per_lb": 299, "price_per_unit": 299},
    {"name": "Gambas", "name_en": "Prawns", "category": "Proteínas",
     "kcal_per_100g": 71, "density_g_per_cup": None, "density_g_per_unit": None,
     "price_per_lb": 0, "price_per_unit": 0},
    {"name": "Pollo", "name_en": "Chicken", "category": "Proteínas",
     "kcal_per_100g": 165, "density_g_per_cup": None, "density_g_per_unit": None,
     "price_per_lb": 95, "price_per_unit": 0},
]
_SHADOWED = {"Judías pintas", "Judías blancas", "Requesón", "Gambas"}
_CANONICAL = {"Frijoles pintos", "Habichuelas blancas", "Queso ricotta", "Camarones"}


@pytest.fixture(autouse=True)
def catalog(monkeypatch):
    import shopping_calculator as sc

    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.setattr(sc, "_verified_ingredients_only_enabled", lambda: True)
    monkeypatch.setattr(sc, "get_master_ingredients", lambda *a, **k: list(_ROWS))
    go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()
    yield
    go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()


def _offered(country: str) -> set[str]:
    block = go._get_verified_catalog_instruction({"country": country})
    assert "uno de estos):\n" in block
    return set(block.rsplit("uno de estos):\n", 1)[1].split(", "))


def _norm(value) -> str:
    return " ".join(strip_accents(str(value or "").lower()).split())


def _duplicate_groups(country: str):
    offered = _offered(country)
    rows = [row for row in _ROWS if row["name"] in offered]
    groups = []

    by_name_en = defaultdict(list)
    for row in rows:
        key = _norm(row.get("name_en"))
        if key:
            by_name_en[key].append(row["name"])
    groups.extend(names for names in by_name_en.values() if len(names) > 1)

    by_nutrition = defaultdict(list)
    for row in rows:
        kcal = row.get("kcal_per_100g")
        densities = (row.get("density_g_per_cup"), row.get("density_g_per_unit"))
        if kcal is None or not any(value is not None for value in densities):
            continue
        signature = (Decimal(str(kcal)), _norm(row.get("category")), *densities)
        by_nutrition[signature].append(row["name"])
    groups.extend(names for names in by_nutrition.values() if len(names) > 1)
    return groups


@pytest.mark.parametrize("country", ("DO", "ES", "MX", "CO", "PR", "US"))
def test_blanket_ningun_pais_ofrece_filas_gemelas(country):
    assert _duplicate_groups(country) == []


def test_es_oculta_altas_redundantes_y_conserva_filas_canonicas():
    offered = _offered("ES")
    assert _SHADOWED.isdisjoint(offered)
    assert _CANONICAL <= offered


def test_do_conserva_identidades_canonicas_y_no_recibe_altas_es():
    offered = _offered("DO")
    assert _CANONICAL <= offered
    assert _SHADOWED.isdisjoint(offered)


def test_filtro_no_renombra_ni_contamina_el_ssot_de_compras():
    import shopping_calculator as sc

    assert go._COUNTRY_CATALOG_SHADOWED_TWINS == {"ES": frozenset(_SHADOWED)}
    assert not hasattr(sc, "_COUNTRY_CATALOG_SHADOWED_TWINS")
    assert all(row["name"] for row in _ROWS), "las filas inyectadas siguen conservando su identidad"
