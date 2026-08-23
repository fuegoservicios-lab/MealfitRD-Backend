"""[P1-COACH-SHOPLIST-BRAND-LEAK · 2026-08-23]

El modo beta ya ocultaba precios y la marca de ``display_qty``, pero dejaba la
misma marca dominicana en los dos canales que lee el coach:

* ``tools.check_shopping_list`` narra ``display_string`` de la forma estructurada.
* ``agent.py`` inserta directamente la forma plana ``list[str]`` en el prompt.

La propiedad de seguridad es el separador de tamaño/marca `` · ``; no una lista
de marcas conocidas. El nombre del alimento debe sobrevivir completo: cortar la
cadena en el separador convierte ``(... · marca) de Alimento`` en ``(...)``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import shopping_calculator as sc


_MASTER = [{
    "name": "Arroz blanco",
    "category": "Despensa",
    "market_container": "funda",
    "container_weight_g": 453.592,
    "price_per_lb": 40.0,
    "default_unit": "funda",
    "shelf_life_days": 365,
    "aliases": [],
}]

_DEFAULTS = {
    "arroz blanco": [{
        "grams": 453.592,
        "price": 42.0,
        "label": "Selecto 1 Lb · Wala",
        "unit": "funda",
        "id": "synthetic-wala-1lb",
    }],
}


@pytest.fixture()
def injected_catalog(monkeypatch):
    """Catálogo hermético: la prueba no depende de DB, red ni skip."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: list(_MASTER))
    monkeypatch.setattr(sc, "fetch_brand_default_packages", lambda: _DEFAULTS)
    monkeypatch.setattr(sc, "fetch_brand_pref_packages", lambda _user_id: {})
    sc.invalidate_master_cache()
    yield
    sc.invalidate_master_cache()


def _plan(*, beta: bool) -> dict:
    plan = {
        "country": "ES" if beta else "DO",
        "days": [{
            "day": 1,
            "meals": [{
                "meal": "Almuerzo",
                "ingredients": ["800 g de arroz blanco"],
                "ingredients_raw": ["800 g de arroz blanco"],
            }],
        }],
    }
    if beta:
        plan["_pricing_mode"] = "beta_no_prices"
    return plan


def _flatten(result):
    if isinstance(result, dict):
        return [item for values in result.values() for item in values]
    return list(result)


@pytest.mark.parametrize("categorize", [False, True])
def test_delta_beta_limpia_marca_en_forma_estructurada_y_preserva_alimento(
    injected_catalog, categorize,
):
    result = sc.get_shopping_list_delta(
        None, _plan(beta=True), is_new_plan=True, categorize=categorize,
        structured=True, multiplier=1.0,
    )
    items = _flatten(result)
    arroz = next((item for item in items if item.get("name") == "Arroz blanco"), None)
    assert arroz is not None, f"el alimento desapareció o cambió de identificador: {result!r}"
    for field in ("display_qty", "display_string", "sku_size_label"):
        value = str(arroz.get(field) or "")
        assert " · " not in value, f"{field} todavía filtra la marca del catálogo DO: {value!r}"
    assert "Arroz blanco" in arroz["display_string"], (
        "limpiar la marca truncó el sufijo que contiene el nombre del alimento"
    )


@pytest.mark.parametrize("categorize", [False, True])
def test_delta_beta_limpia_marca_en_texto_plano_del_prompt_y_preserva_alimento(
    injected_catalog, categorize,
):
    result = sc.get_shopping_list_delta(
        None, _plan(beta=True), is_new_plan=True, categorize=categorize,
        structured=False, multiplier=1.0,
    )
    lines = _flatten(result)
    assert lines, "el fixture debe producir texto para el prompt del coach"
    rendered = "\n".join(lines)
    assert " · " not in rendered, f"el prompt plano todavía recibe la marca: {rendered!r}"
    assert "Arroz blanco" in rendered, f"la limpieza se comió el nombre del alimento: {rendered!r}"


def test_strip_conserva_cada_unidad_y_sufijo_del_nombre():
    result = [{
        "name": "Arroz blanco",
        "display_qty": "3 funda (Selecto 1 Lb · Wala c/u)",
        "display_string": "3 funda (Selecto 1 Lb · Wala c/u) de Arroz blanco",
        "sku_size_label": "Selecto 1 Lb · Wala",
    }]
    sc._strip_prices_for_beta_pricing_mode(result)
    assert result[0]["display_qty"] == "3 funda (Selecto 1 Lb c/u)"
    assert result[0]["display_string"] == "3 funda (Selecto 1 Lb c/u) de Arroz blanco"
    assert result[0]["sku_size_label"] == "Selecto 1 Lb"


def test_control_do_conserva_bytes_y_marca(injected_catalog):
    """El choke point sólo corre para beta; DO conserva exactamente su payload con marca."""
    plan = _plan(beta=False)
    first = sc.get_shopping_list_delta(
        None, plan, is_new_plan=True, categorize=True, structured=True, multiplier=1.0,
    )
    second = sc.get_shopping_list_delta(
        None, plan, is_new_plan=True, categorize=True, structured=True, multiplier=1.0,
    )
    first_bytes = json.dumps(first, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    second_bytes = json.dumps(second, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert first_bytes == second_bytes, "el control DO dejó de ser byte determinista"
    assert b" \xc2\xb7 " in first_bytes, "fixture inválido: DO debe conservar la marca del catálogo"


def test_marker_tiene_formato_movil_y_el_anchor_vive_en_implementacion():
    backend = Path(__file__).resolve().parents[1]
    app_src = (backend / "app.py").read_text(encoding="utf-8")
    calc_src = (backend / "shopping_calculator.py").read_text(encoding="utf-8")
    assert '_LAST_KNOWN_PFIX = "P' in app_src and " · 2026-" in app_src
    assert "P1-COACH-SHOPLIST-BRAND-LEAK" in calc_src
