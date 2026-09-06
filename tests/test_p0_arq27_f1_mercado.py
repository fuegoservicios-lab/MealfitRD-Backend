# -*- coding: utf-8 -*-
"""[ARQ27-P1-07 · 2026-09-06] El paso de disponibilidad en el mercado no corría nunca.

`compile_policy` tiene desde F2 un paso 3 —entre la dieta y el presupuesto— que descarta un ancla
que el mercado del usuario no vende. Ese paso pide `known_ingredients` en su contexto, y
**`compile_from_form`, el único constructor que usa producción, jamás se lo pasaba**: el contexto
llegaba con presupuesto y modo de precios, sin catálogo. Resultado: `market_check_skipped` en el
100 % de los planes (57 de 57 según el registro publicado de F3).

Coste: la cultura podía pedir un plato cuyo ingrediente ancla el país no vende y nadie se enteraba
hasta mucho más abajo. Caro sobre todo en perfiles veganos y en cocinas cruzadas —una cocina
dominicana comprando en Estados Unidos— que es justo lo que el sistema de países abrió.

Dos decisiones que estos tests defienden:

  · **Cultura ≠ mercado (I16).** El snapshot es del país de COMPRA. Una cocina dominicana en US
    consulta ingredientes de US y recetas de DO.
  · **Ausente ≠ vacío.** Catálogo ilegible ⇒ `None` ⇒ el compilador conserva `market_check_skipped`.
    Devolver `[]` diría «este país no vende nada» y borraría todas las anclas del usuario. Es la
    misma distinción que ARQ27-P0-03 hace con los nutrientes: no saber no es saber que no hay.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import catalog_capability as CC  # noqa: E402
import plan_policy as PP  # noqa: E402

_FILAS = [
    {"name": "Arroz blanco", "aliases": ["arroz"]},
    {"name": "Yuca", "aliases": []},
    {"name": "Filete de pescado blanco", "aliases": ["chillo", "mero"]},
]


@pytest.fixture(autouse=True)
def _cache_limpia():
    """El `monkeypatch` revierte el mock; la caché de módulo NO. Sin este teardown, un snapshot de
    tres filas construido aquí sobrevive y el test siguiente mide el catálogo falso creyendo que mide
    el vivo — que es exactamente lo que pasó con US."""
    CC.reset_cache()
    yield
    CC.reset_cache()


@pytest.fixture
def catalogo(monkeypatch):
    """Catálogo determinista: estos tests miden el CABLEADO, no el contenido de producción."""
    import shopping_calculator
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda *a, **k: list(_FILAS))
    return _FILAS


def _politica(form):
    return PP.compile_from_form(form)


# ── el snapshot de capacidad ──────────────────────────────────────────────────────────────────
def test_el_snapshot_trae_nombres_alias_y_huella(catalogo):
    s = CC.catalog_capability("DO")
    assert s["market_country"] == "DO" and s["count"] == 3
    assert "Arroz blanco" in s["names"] and "chillo" in s["aliases"]
    assert len(s["fingerprint"]) == 12 and s["source"] == "master_ingredients"


def test_un_alias_no_cuenta_como_alimento(catalogo):
    """`count` cuenta identidades, no formas de nombrarlas (ARQ27-P1-05). Tres filas con cinco
    nombres siguen siendo tres alimentos."""
    s = CC.catalog_capability("DO")
    assert s["count"] == 3 and len(s["aliases"]) == 3
    assert s["count"] == len(s["names"])


@pytest.mark.parametrize("rows", [[], None])
def test_catalogo_ilegible_es_desconocido_no_vacio(monkeypatch, rows):
    """La decisión que separa este gap de un incidente: `None`, jamás `[]`."""
    import shopping_calculator
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda *a, **k: rows)
    assert CC.catalog_capability("DO") is None
    assert CC.known_ingredient_names("DO") is None


def test_catalogo_que_revienta_es_desconocido(monkeypatch):
    import shopping_calculator

    def _boom(*a, **k):
        raise RuntimeError("db caída")
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", _boom)
    assert CC.catalog_capability("DO") is None


# ── el cableado: el paso 3 corre ──────────────────────────────────────────────────────────────
def test_el_paso_de_mercado_corre_y_deja_constancia(catalogo):
    """Antes: `market_check_skipped` siempre. La constancia POSITIVA hace falta porque un plan sin
    anclas descartadas es indistinguible de uno cuyo mercado nadie comprobó."""
    eff = _politica({"country": "DO", "dietType": "balanced"})["effective"]
    notas = eff.get("notes") or []
    assert "market_check_applied" in notas
    assert "market_check_skipped" not in notas


def test_sin_catalogo_el_paso_se_declara_omitido(monkeypatch):
    """La conducta previa se conserva EXACTAMENTE cuando no hay evidencia. No hay aprobación por
    omitir el contexto: hay una nota que dice que no se comprobó."""
    import shopping_calculator
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda *a, **k: [])
    eff = _politica({"country": "DO", "dietType": "balanced"})["effective"]
    notas = eff.get("notes") or []
    assert "market_check_skipped" in notas and "market_check_applied" not in notas


def test_un_ancla_que_el_mercado_no_vende_se_explica_antes_del_llm(catalogo):
    eff_out = _politica({"country": "DO", "dietType": "balanced",
                         "stapleFoods": ["Arroz blanco", "Zapote"]})
    nombres = [a.get("name") for a in (eff_out["effective"].get("food_anchors") or [])]
    assert "Arroz blanco" in nombres and "Zapote" not in nombres
    # `_relax` guarda el motivo en `reason_code`, NO en `reason`. Leer el campo equivocado ya
    # fabricó un gap inexistente una vez en este mismo roadmap (el recuento de `relaxations[]`),
    # así que el test lo nombra tal cual lo escribe el productor.
    fuera = [r for r in eff_out["relaxations"] if r.get("reason_code") == "anchor_not_in_market"]
    assert [r["requested"] for r in fuera] == ["Zapote"]
    assert fuera[0]["evidence"].get("market_country")


def test_un_ancla_que_solo_existe_por_alias_sigue_disponible(catalogo):
    """«Chillo» no es una fila del catálogo: es un ALIAS de «Filete de pescado blanco». El mercado lo
    tiene, con otro nombre. Descartarlo sería inventar una escasez."""
    eff = _politica({"country": "DO", "dietType": "balanced", "stapleFoods": ["Chillo"]})["effective"]
    assert [a.get("name") for a in (eff.get("food_anchors") or [])] == ["Chillo"]


def test_el_mercado_manda_sobre_la_cocina(catalogo, monkeypatch):
    """I16. Una cocina dominicana comprando en US consulta el catálogo de US; el snapshot lleva el
    país de COMPRA, no el de la biblioteca cultural.

    El knob del sistema de países va explícito: `conftest` lo deja en su default `False` y sin él
    `country_for_form_data` colapsa todo a DO — un test que no declara el flag que necesita mide otro
    producto (ARQ27-P1-06)."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "1")
    eff = _politica({"country": "US", "dietType": "balanced",
                     "cuisinePrimary": "dominican_criolla"})["effective"]
    assert eff.get("market_country") == "US"
    assert "market_check_applied" in (eff.get("notes") or [])
    assert CC.catalog_capability("US")["market_country"] == "US"


# ── contra el catálogo VIVO: que el filtro discrimine de verdad ───────────────────────────────
@pytest.mark.parametrize("cc", ["DO", "ES", "US", "MX", "CO", "PR"])
def test_los_seis_mercados_tienen_capacidad_no_vacia(cc):
    """Sin monkeypatch. Si un mercado saliera vacío el paso 3 borraría todas las anclas de sus
    usuarios, así que este test vale por sí solo aunque necesite la DB."""
    s = CC.catalog_capability(cc)
    if s is None:
        pytest.skip("catálogo no disponible en este entorno")
    assert s["count"] > 250, f"{cc} solo expone {s['count']} filas"


def test_el_matcher_no_aprueba_cualquier_cosa():
    """Un filtro que nunca rechaza es un filtro inerte. `_matches` es el SSOT existente; aquí solo se
    comprueba que sigue distinguiendo un alimento real de uno inventado."""
    conocidos = ["Arroz blanco", "Yuca", "arroz"]
    assert any(PP._matches("Arroz blanco", k) for k in conocidos)
    assert not any(PP._matches("Unicornio", k) for k in conocidos)
    assert not any(PP._matches("Zapote", k) for k in conocidos)
