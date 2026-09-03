"""Guards de P1-COUNTRY-CONDIMENT-PARITY-BETA.

El catálogo cerrado beta autoriza sazonadores locales. El gate de Nevera debe
reconocer exactamente los que pertenecen al país del plan, sin convertir el
pool beta completo (que también contiene proteínas y alimentos base) en una
lista de condimentos gratis.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import constants
import shopping_calculator


ROOT = Path(__file__).resolve().parents[1]


# Filas mínimas con la misma metadata que decide el código vivo. Incluimos
# etiquetas históricas imperfectas (hierbas/chiles bajo ``Vegetales``) porque
# son precisamente las filas que el catálogo verificado ofrece hoy.
CONDIMENTOS = {
    "ES": (
        ("Azafrán", "Despensa", "Saffron"),
        ("Alioli", "Despensa", "Garlic aioli"),
    ),
    "MX": (
        ("Epazote", "Vegetales", "Epazote herb"),
        ("Chile jalapeño", "Vegetales", "Jalapeño pepper"),
        ("Chile serrano", "Vegetales", "Serrano pepper"),
        ("Chile poblano", "Vegetales", "Poblano pepper"),
        ("Chile chipotle", "Despensa", "Chipotle chile"),
        ("Chile guajillo", "Despensa", "Guajillo chile"),
        ("Chile ancho", "Despensa", "Ancho chile"),
        ("Chile habanero", "Vegetales", "Habanero chile"),
        ("Chile de árbol", "Despensa", "Tree chile"),
        ("Chile pasilla", "Despensa", "Pasilla chile"),
        ("Chile mulato", "Despensa", "Mulato chile"),
        ("Achiote", "Despensa", "Achiote"),
        ("Aceite de achiote", "Despensa", "Achiote oil"),
        ("Hoja santa", "Vegetales", "Hoja santa leaf"),
        ("Sazón con culantro y achiote", "Despensa", "Culantro-achiote seasoning"),
    ),
    "CO": (
        ("Guascas", "Vegetales", "Guasca herb"),
    ),
    "PR": (
        ("Sofrito", "Despensa", "Sofrito sauce base"),
        ("Adobo", "Despensa", "Adobo seasoning"),
        ("Recao", "Vegetales", "Culantro"),
        ("Pique", "Despensa", "Hot pepper sauce"),
        ("Especias para arroz con dulce", "Despensa", "Rice pudding spices"),
        ("Aceite de achiote", "Despensa", "Achiote oil"),
        ("Sazón con culantro y achiote", "Despensa", "Culantro-achiote seasoning"),
    ),
    "US": (
        ("Aderezo ranch", "Despensa", "Ranch dressing"),
        ("Chile en polvo", "Despensa", "Chili powder"),
        ("Kétchup", "Despensa", "Ketchup"),
        ("Salsa barbacoa", "Despensa", "Barbecue sauce"),
        ("Salsa inglesa", "Despensa", "Worcestershire sauce"),
        ("Sazonador para tacos", "Despensa", "Taco seasoning"),
    ),
}

NO_CONDIMENTOS = {
    "ES": (
        ("Jamón serrano", "Proteínas", "Serrano ham"),
        ("Fideos", "Despensa", "Noodles"),
    ),
    "MX": (("Cecina", "Proteínas", "Dried cured meat"),),
    "CO": (
        ("Gallina criolla", "Proteínas", "Free-range hen"),
        ("Frijol cargamanto", "Despensa", "Cargamanto beans"),
    ),
    "PR": (("Pernil", "Proteínas", "Roast pork shoulder"),),
    "US": (("Chili con carne", "Despensa", "Chili con carne"),),
}


def _row(name: str, category: str, name_en: str) -> dict:
    return {
        "name": name,
        "category": category,
        "name_en": name_en,
        "aliases": [name.lower()],
        "price_per_lb": 0,
        "price_per_unit": 0,
    }


@pytest.fixture(autouse=True)
def catalogo_beta_inyectado(monkeypatch):
    membership = {}
    rows = []
    for country, triples in {**CONDIMENTOS}.items():
        for triple in triples + NO_CONDIMENTOS[country]:
            rows.append(_row(*triple))
            membership.setdefault(triple[0], set()).add(country)

    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda: rows)
    monkeypatch.setattr(
        shopping_calculator,
        "is_country_catalog_unpriced_item",
        lambda name, country=None: country in membership.get(name, set()),
    )
    cache = getattr(constants, "_COUNTRY_CATALOG_CONDIMENT_CACHE", None)
    if isinstance(cache, dict):
        cache.clear()
    yield
    if isinstance(cache, dict):
        cache.clear()


def _veredicto(name: str, country: str):
    return constants.validate_ingredients_against_pantry(
        [f"1 pizca de {name}"],
        ["500 g de Pechuga de pollo"],
        strict_quantities=False,
        return_unauthorized=True,
        country=country,
    )


@pytest.mark.parametrize(
    ("country", "name"),
    [(cc, row[0]) for cc, rows in CONDIMENTOS.items() for row in rows],
)
def test_cada_condimento_ofrecido_por_el_catalogo_del_pais_pasa(country, name):
    assert _veredicto(name, country) == (True, []), (
        f"{name} aparece en el catálogo cerrado de {country}; castigarlo por inexistente "
        "cobra un retry por obedecer el propio prompt"
    )


@pytest.mark.parametrize(
    ("country", "name"),
    [(cc, row[0]) for cc, rows in NO_CONDIMENTOS.items() for row in rows],
)
def test_el_pool_completo_no_se_convierte_en_condimento(country, name):
    verdict, unauthorized = _veredicto(name, country)
    assert verdict is not True
    assert unauthorized == [f"1 pizca de {name}"]


def test_exencion_es_por_pais_y_do_no_hereda_beta():
    assert _veredicto("Azafrán", "ES") == (True, [])
    for other in ("DO", "MX", "CO", "PR", "US"):
        verdict, unauthorized = _veredicto("Azafrán", other)
        assert verdict is not True, f"Azafrán ES se filtró a {other}"
        assert unauthorized == ["1 pizca de Azafrán"]


def test_do_no_construye_patrones_beta_y_aliases_desnudos_no_se_globalizan():
    assert constants._country_catalog_condiment_patterns("DO") == ()
    for naked in ("Azafrán", "Adobo", "Sofrito", "Guascas", "Mora"):
        verdict, unauthorized = _veredicto(naked, "DO")
        assert verdict is not True, f"{naked} se convirtió en exención global"
        assert unauthorized == [f"1 pizca de {naked}"]


def test_mutacion_sin_azafran_es_roja_y_control_do_permanece_rechazando(monkeypatch):
    original = shopping_calculator.get_master_ingredients
    monkeypatch.setattr(
        shopping_calculator,
        "get_master_ingredients",
        lambda: [r for r in original() if r["name"] != "Azafrán"],
    )
    constants._COUNTRY_CATALOG_CONDIMENT_CACHE.clear()

    assert _veredicto("Azafrán", "ES")[0] is not True
    assert _veredicto("Azafrán", "DO")[0] is not True


def test_todos_los_call_sites_productivos_propagan_country():
    files = ("agent.py", "cron_tasks.py", "graph_orchestrator.py", "tools.py")
    calls = []
    for filename in files:
        tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            called = fn.id if isinstance(fn, ast.Name) else (
                fn.attr if isinstance(fn, ast.Attribute) else ""
            )
            if called in {"validate_ingredients_against_pantry", "_vip"}:
                calls.append((filename, node.lineno, {kw.arg for kw in node.keywords}))

    assert len(calls) == 18, f"cambió el universo de call sites: {calls}"
    missing = [(f, line) for f, line, kwargs in calls if "country" not in kwargs]
    assert not missing, f"validadores sin país (el fix quedaría inerte en beta): {missing}"


def test_anclas_de_implementacion_y_marker_supersession_safe():
    src = (ROOT / "constants.py").read_text(encoding="utf-8")
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "tooltip-anchor: P1-COUNTRY-CONDIMENT-PARITY-BETA" in src
    assert "_COUNTRY_CATALOG_CONDIMENT_CACHE" in src
    assert "P1-COUNTRY-CONDIMENT-PARITY-BETA" in app
