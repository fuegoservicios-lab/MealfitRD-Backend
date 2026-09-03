"""[P1-COUNTRY-CAPS-DO-LEXICON · 2026-08-23]

Los topes de compra deben reconocer la misma familia aunque el catálogo use
el nombre beta. El guard observa la conducta del agregador sobre catálogo
inyectado: no congela el texto de display ni consulta producción.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import pytest

import shopping_calculator as sc
from constants import strip_accents


def _row(
    name: str,
    name_en: str,
    category: str,
    kcal: float,
    *,
    density_cup=None,
    price_lb=0,
    price_unit=0,
):
    return {
        "name": name,
        "name_en": name_en,
        "category": category,
        "kcal_per_100g": kcal,
        "density_g_per_cup": density_cup,
        "density_g_per_unit": None,
        "price_per_lb": price_lb,
        "price_per_unit": price_unit,
        "aliases": [],
        "container_weight_g": None,
        "container_unit": None,
    }


_ROWS = [
    _row("Frijoles pintos", "Pinto beans", "Despensa", 347, density_cup=193,
         price_lb=72, price_unit=127),
    _row("Judías pintas", "Pinto beans", "Despensa", 347),
    _row("Habichuelas blancas", "White beans", "Despensa", 333,
         density_cup=180, price_unit=50),
    _row("Judías blancas", "White beans", "Despensa", 333, density_cup=180),
    _row("Queso ricotta", "Ricotta cheese", "Lácteos", 151,
         density_cup=246, price_unit=245),
    _row("Requesón", "Ricotta", "Lácteos", 151, density_cup=246),
    _row("Nata", "Cream", "Lácteos", 340),
    _row("Crema agria", "Sour cream", "Lácteos", 198),
    _row("Crema mexicana", "Mexican cream", "Lácteos", 136),
    _row("Crema mitad y mitad", "Half and half", "Lácteos", 131),
    _row("Cuajada", "Curd cheese", "Lácteos", 98),
    _row("Natilla", "Custard", "Lácteos", 104),
    _row("Queso de papa", "Potato cheese", "Lácteos", 357),
    _row("Queso en hebras", "Shredded cheese", "Lácteos", 295),
    _row("Queso provolone", "Provolone cheese", "Lácteos", 351),
    _row("Suero costeño", "Coastal sour cream", "Lácteos", 83),
    _row("Suero de mantequilla", "Buttermilk", "Lácteos", 62),
    _row("Pollo", "Chicken", "Proteínas", 165, price_lb=95),
]


@pytest.fixture(autouse=True)
def catalog(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.setenv("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "true")
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: list(_ROWS))


def _norm(value) -> str:
    return " ".join(strip_accents(str(value or "").casefold()).split())


def _twin_groups():
    """Agrupa por glosa o firma nutricional; el test no enumera los pares."""
    groups: set[frozenset[str]] = set()
    by_name_en = defaultdict(list)
    by_nutrition = defaultdict(list)
    for row in _ROWS:
        gloss = _norm(row.get("name_en"))
        if gloss:
            by_name_en[gloss].append(row["name"])
        densities = (row.get("density_g_per_cup"), row.get("density_g_per_unit"))
        if any(value is not None for value in densities):
            signature = (
                Decimal(str(row["kcal_per_100g"])),
                _norm(row["category"]),
                *densities,
            )
            by_nutrition[signature].append(row["name"])
    for bucket in (*by_name_en.values(), *by_nutrition.values()):
        if len(bucket) > 1:
            groups.add(frozenset(bucket))
    return sorted(groups, key=lambda group: sorted(group))


def _aggregate(name: str):
    result = sc.aggregate_and_deduct_shopping_list(
        [f"800 g de {name}"] * 3,
        structured=True,
        num_days=3,
    )
    items = result.get("items") if isinstance(result, dict) else result
    item = next(i for i in items if _norm(i.get("name")) == _norm(name))
    caps = [
        cap for cap in sc.get_caps_applied_last_run()
        if _norm(cap.get("food")) == _norm(name)
    ]
    return item, caps


def _cap_signature(name: str):
    item, caps = _aggregate(name)
    return (
        float(item["base_qty"]),
        item["base_unit"],
        tuple((cap["reason"], round(float(cap["post_value"]), 3)) for cap in caps),
    )


def test_blanket_gemelos_reciben_el_mismo_tope():
    groups = _twin_groups()
    assert groups, "el catálogo inyectado debe contener gemelos observables"
    for group in groups:
        signatures = {_cap_signature(name) for name in group}
        assert len(signatures) == 1, f"topes asimétricos para {sorted(group)}: {signatures}"


def test_todos_los_lacteos_beta_disparan_el_cap_de_su_familia():
    beta_dairy = [
        row["name"] for row in _ROWS
        if _norm(row["category"]) == "lacteos"
        and not (row["price_per_lb"] or row["price_per_unit"])
    ]
    assert len(beta_dairy) == 12
    for name in beta_dairy:
        _item, caps = _aggregate(name)
        assert [cap["reason"] for cap in caps] == ["P6-LACTEOS-PERISHABLE-CAP"], name


@pytest.mark.parametrize("name,reason", [
    ("Frijoles pintos", "P6-LEGUMES-DRY-CAP"),
    ("Queso ricotta", "P6-LACTEOS-PERISHABLE-CAP"),
])
def test_control_do_conserva_su_tope_historico(name, reason):
    item, caps = _aggregate(name)
    assert float(item["base_qty"]) == pytest.approx(453.59, abs=0.01)
    assert [(cap["reason"], float(cap["post_value"])) for cap in caps] == [
        (reason, pytest.approx(453.592))
    ]


def test_el_cierre_queda_anclado_en_la_implementacion():
    source = Path(sc.__file__).read_text(encoding="utf-8")
    assert "P1-COUNTRY-CAPS-DO-LEXICON" in source
