"""G04: el catálogo cerrado beta resta preparaciones DO inequívocas."""

from __future__ import annotations

from pathlib import Path

import pytest

import graph_orchestrator as go


BACKEND_ROOT = Path(__file__).resolve().parents[1]

_CURATED = {
    "Casabe",
    "Casabe albahaca",
    "Longaniza dominicana",
    "Orégano dominicano",
    "Queso de hoja",
    "Salami",
    "Harina de Negrito",
    "Cundeamor",
    "Mapuey",
}

_CATALOG = [
    *[
        {"name": name, "price_per_lb": 1, "price_per_unit": 0}
        for name in sorted(_CURATED)
    ],
    {"name": "Pollo", "price_per_lb": 1, "price_per_unit": 0},
    {"name": "Arroz blanco", "price_per_lb": 1, "price_per_unit": 0},
    # Se conservan hasta que existan destinos neutrales canónicos; retirarlos hoy
    # haría desaparecer comida bajo una lista cerrada.
    {"name": "Auyama", "price_per_lb": 1, "price_per_unit": 0},
    {"name": "Yautía", "price_per_lb": 1, "price_per_unit": 0},
    # Filas beta sin precio reconocidas por el SSOT por país.
    {"name": "Jamón serrano", "price_per_lb": 0, "price_per_unit": 0},
    {"name": "Boquerones", "price_per_lb": 0, "price_per_unit": 0},
]


@pytest.fixture
def catalogo(monkeypatch):
    import shopping_calculator as sc

    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.setattr(sc, "_verified_ingredients_only_enabled", lambda: True)
    monkeypatch.setattr(sc, "get_master_ingredients", lambda *a, **k: list(_CATALOG))
    go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()
    yield
    go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()


def _render(country: str) -> str:
    return go._get_verified_catalog_instruction({"country": country})


def test_es_resta_las_nueve_filas_curadas_y_conserva_beta(catalogo) -> None:
    es = _render("ES")
    for name in _CURATED:
        assert name not in es
    assert "Jamón serrano" in es
    assert "Boquerones" in es


def test_no_se_borran_regionalismos_sin_destino_canonico(catalogo) -> None:
    es = _render("ES")
    assert "Auyama" in es
    assert "Yautía" in es


def test_do_conserva_byte_identidad_y_las_nueve_filas(catalogo, monkeypatch) -> None:
    with_knob = _render("DO")
    for name in _CURATED:
        assert name in with_knob

    go._VERIFIED_CATALOG_INSTRUCTION_CACHE.clear()
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    without_knob = _render("DO")
    assert with_knob == without_knob


def test_predicado_nuevo_no_es_true_incondicional(catalogo) -> None:
    es = _render("ES")
    assert "Pollo" in es
    assert "Arroz blanco" in es


def test_cache_sigue_separando_paises(catalogo) -> None:
    do_first = _render("DO")
    es_after = _render("ES")
    assert do_first != es_after
    assert "Casabe" in do_first
    assert "Casabe" not in es_after
    assert any(
        isinstance(key, tuple) and len(key) == 2 and key[1] == "DO"
        for key in go._VERIFIED_CATALOG_INSTRUCTION_CACHE
    )
    assert any(
        isinstance(key, tuple) and len(key) == 2 and key[1] == "ES"
        for key in go._VERIFIED_CATALOG_INSTRUCTION_CACHE
    )


def test_lista_curada_es_exacta_y_no_contamina_ssot_de_compras() -> None:
    import shopping_calculator as sc

    assert set(go._BETA_CATALOG_DO_EXCLUSIVE_NAMES) == _CURATED
    assert not hasattr(sc, "_BETA_CATALOG_DO_EXCLUSIVE_NAMES")
    source = (BACKEND_ROOT / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "_DO_LEXICON_NEUTRAL" not in source[source.index(
        "_BETA_CATALOG_DO_EXCLUSIVE_NAMES"
    ):source.index("def _patron_termino_alergeno")]


def test_pfix_marker_cierra_g04() -> None:
    app = (BACKEND_ROOT / "app.py").read_text(encoding="utf-8")
    assert "P1-BETA-CATALOG-DO-EXCLUSIVE · 2026-08-23" in app
