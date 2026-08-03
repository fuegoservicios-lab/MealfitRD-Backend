"""[P3-NEW-12 · 2026-05-11] Canonicalizers RD adicionales — cítricos, tomate,
cebolla, quesos blancos, frutos secos.

Bug observado: variantes triviales del LLM ("limón verde", "limón persa",
"tomate cherry", "tomate criollo", "cebolla roja", "cebolla morada",
"queso frescal", "queso de freír", "almendra natural", "almendra tostada")
generaban N líneas separadas en la lista de compras cuando el usuario
compra UN producto (en la mayoría de casos). Algunos tipos DEBEN
preservarse como productos distintos (limón≠lima, tomate cherry≠tomate,
queso de freír≠queso blanco, almendras≠maní).

Fix: 5 canonicalizers nuevos paralelos a P2-NEW-A (frutas tropicales /
verduras hoja / aceites). Bilateral wiring:
  - guard side: `_canonicalize_for_coherence` (presence/absence + magnitude).
  - aggregator side: `aggregate_and_deduct_shopping_list` (lista output).

Estrategia del test:
    A) Behavioral per canonicalizer: variantes mapean al canónico esperado.
    B) Behavioral cross-tipo: NO se colapsan tipos distintos.
    C) Parser-based: bilateral wiring (cada fn invocada en ambos
       sitios).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _REPO_ROOT / "backend"
sys.path.insert(0, str(_BACKEND))


# ──────────────────────────────────────────────────────────────────────
# A) Behavioral — variantes mapean al canónico esperado.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name,expected", [
    # Cítricos
    ("Limón", "Limón"),
    ("limones", "Limón"),
    ("limón verde", "Limón"),
    ("limón criollo", "Limón"),
    ("limón persa", "Limón"),
    ("lima", "Lima"),
    ("Limas", "Lima"),
    ("naranja", "Naranja"),
    ("naranja agria", "Naranja"),
    ("naranja valencia", "Naranja"),
    ("mandarina", "Mandarina"),
    ("mandarinas", "Mandarina"),
    ("toronja", "Toronja"),
    ("pomelo", "Toronja"),
    ("grapefruit", "Toronja"),
    # No matches → None
    ("manzana", None),
    ("", None),
])
def test_canonicalize_citricos(name, expected):
    from shopping_calculator import canonicalize_citricos
    assert canonicalize_citricos(name) == expected


def test_citricos_does_not_collapse_cross_type():
    """Limón y lima son productos distintos."""
    from shopping_calculator import canonicalize_citricos
    assert canonicalize_citricos("limón") != canonicalize_citricos("lima")
    assert canonicalize_citricos("naranja") != canonicalize_citricos("mandarina")


@pytest.mark.parametrize("name,expected", [
    ("tomate", "Tomate"),
    ("Tomate", "Tomate"),
    ("tomates", "Tomate"),
    ("tomate criollo", "Tomate"),
    ("tomate perita", "Tomate"),
    ("tomate maduro", "Tomate"),
    ("tomate roma", "Tomate"),
    # Cherry/uva PRESERVADO como producto distinto.
    ("tomate cherry", "Tomate cherry"),
    ("tomates cherry", "Tomate cherry"),
    ("tomate uva", "Tomate cherry"),
    # No match
    # [reapuntado 2026-07-28 · P3-TOMATE-SAUCE-FIX 06-22] El "OK por simplicidad" era el
    # BUG: la salsa colapsaba a tomate fresco y se costeaba/capeaba como tomate de
    # ensalada (P5-VEG-CAP en vez de P6-SAUCE-CAP). Las formas procesadas conservan su
    # nombre y resuelven a su propio master item.
    ("salsa de tomate", None),
    ("pasta de tomate", None),
    ("tomate seco", None),
    ("zanahoria", None),
    ("", None),
])
def test_canonicalize_tomate(name, expected):
    from shopping_calculator import canonicalize_tomate
    assert canonicalize_tomate(name) == expected


def test_tomate_cherry_preserved_distinct():
    """Cherry NO debe colapsar con tomate normal — son productos distintos."""
    from shopping_calculator import canonicalize_tomate
    assert canonicalize_tomate("tomate cherry") != canonicalize_tomate("tomate criollo")


@pytest.mark.parametrize("name,expected", [
    ("cebolla", "Cebolla"),
    ("cebollas", "Cebolla"),
    ("cebolla roja", "Cebolla"),
    ("cebolla morada", "Cebolla"),
    ("cebolla blanca", "Cebolla"),
    ("cebolla amarilla", "Cebolla"),
    # Cebollín PRESERVADO como producto distinto.
    ("cebollín", "Cebollín"),
    ("cebollin", "Cebollín"),
    ("cebolla verde", "Cebollín"),
    ("cebolla de verdeo", "Cebollín"),
    ("cebolleta", "Cebollín"),
    ("cebolletas", "Cebollín"),
    # No match
    ("ajo", None),
    ("", None),
])
def test_canonicalize_cebolla(name, expected):
    from shopping_calculator import canonicalize_cebolla
    assert canonicalize_cebolla(name) == expected


def test_cebolla_vs_cebollin_distinct():
    """Cebollín es producto distinto (hierba aromática)."""
    from shopping_calculator import canonicalize_cebolla
    assert canonicalize_cebolla("cebolla") != canonicalize_cebolla("cebollín")


@pytest.mark.parametrize("name,expected", [
    # Queso blanco family colapsa
    ("queso blanco", "Queso blanco"),
    ("queso frescal", "Queso blanco"),
    ("queso fresco", "Queso blanco"),
    # Queso de freír PRESERVADO
    ("queso de freír", "Queso de freír"),
    ("queso de freir", "Queso de freír"),
    ("queso frito", "Queso de freír"),
    # Otros tipos distintos
    ("mozzarella", "Mozzarella"),
    ("mozarella", "Mozzarella"),
    ("queso crema", "Queso crema"),
    ("cheddar", "Cheddar"),
    ("queso cheddar", "Cheddar"),
    ("parmesano", "Parmesano"),
    ("parmegiano", "Parmesano"),
    # No match: "queso" genérico ambiguo
    ("queso", None),
    ("", None),
])
def test_canonicalize_quesos_blancos_rd(name, expected):
    from shopping_calculator import canonicalize_quesos_blancos_rd
    assert canonicalize_quesos_blancos_rd(name) == expected


def test_quesos_preserve_distinct_types():
    """Queso de freír, queso blanco, mozzarella, queso crema son productos
    distintos."""
    from shopping_calculator import canonicalize_quesos_blancos_rd as cq
    types = {cq("queso de freír"), cq("queso blanco"), cq("mozzarella"), cq("queso crema")}
    assert len(types) == 4, f"esperado 4 tipos distintos, got {types}"


@pytest.mark.parametrize("name,expected", [
    # Almendras
    ("almendra", "Almendras"),
    ("almendras", "Almendras"),
    ("almendra tostada", "Almendras"),
    ("almendras laminadas", "Almendras"),
    # Maní
    ("maní", "Maní"),
    ("mani", "Maní"),
    ("cacahuete", "Maní"),
    ("cacahuetes", "Maní"),
    ("cacahuate", "Maní"),
    # Nueces
    ("nuez", "Nueces"),
    ("nueces", "Nueces"),
    ("walnut", "Nueces"),
    ("walnuts", "Nueces"),
    # Avellanas
    ("avellana", "Avellanas"),
    ("avellanas", "Avellanas"),
    # Pistachos
    ("pistacho", "Pistachos"),
    ("pistachos", "Pistachos"),
    # Anacardos / marañón / cashew
    ("anacardo", "Anacardos"),
    ("anacardos", "Anacardos"),
    ("marañón", "Anacardos"),
    ("cashew", "Anacardos"),
    # Pecanas
    ("pecana", "Pecanas"),
    ("pecanas", "Pecanas"),
    ("nuez pecan", "Pecanas"),
    ("pecan", "Pecanas"),
    # No match: semillas (categoría distinta)
    ("chía", None),
    ("semillas de calabaza", None),
    # No match: pasas (categoría distinta)
    ("pasas", None),
    ("", None),
])
def test_canonicalize_frutos_secos(name, expected):
    from shopping_calculator import canonicalize_frutos_secos
    assert canonicalize_frutos_secos(name) == expected


def test_frutos_secos_no_cross_collapse():
    """Almendras ≠ maní ≠ nueces — alergenos distintos, precios distintos."""
    from shopping_calculator import canonicalize_frutos_secos as cf
    types = {cf("almendras"), cf("maní"), cf("nueces"), cf("pistachos"), cf("anacardos")}
    assert len(types) == 5, f"esperado 5 tipos distintos, got {types}"


def test_pecanas_vs_nueces_distinct():
    """Pecanas tienen su propia canónica para no colapsar con nueces."""
    from shopping_calculator import canonicalize_frutos_secos as cf
    assert cf("pecanas") == "Pecanas"
    assert cf("nueces") == "Nueces"
    assert cf("pecanas") != cf("nueces")


# ──────────────────────────────────────────────────────────────────────
# C) Parser-based — bilateral wiring (guard + aggregator).
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def src() -> str:
    return (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")


def test_all_5_canonicalizers_defined(src: str):
    for fn in (
        "canonicalize_citricos",
        "canonicalize_tomate",
        "canonicalize_cebolla",
        "canonicalize_quesos_blancos_rd",
        "canonicalize_frutos_secos",
    ):
        assert f"def {fn}(" in src, (
            f"P3-NEW-12 regresión: `{fn}` ya no está definido."
        )


def test_guard_side_wires_all_5(src: str):
    """`_canonicalize_for_coherence` debe invocar las 5 fns nuevas."""
    func_start = src.find("def _canonicalize_for_coherence(")
    func_end = src.find("\ndef run_shopping_coherence_guard(", func_start)
    body = src[func_start:func_end]
    for fn in (
        "canonicalize_citricos",
        "canonicalize_tomate",
        "canonicalize_cebolla",
        "canonicalize_quesos_blancos_rd",
        "canonicalize_frutos_secos",
    ):
        assert fn in body, (
            f"P3-NEW-12 regresión: `_canonicalize_for_coherence` ya no "
            f"invoca `{fn}`. Sin el wire del guard side, presence/absence "
            f"reportará drift falso entre receta y lista para variantes "
            f"colapsables."
        )


# [review final audit-v7-p1 · 2026-08-03] El lado LISTA DE COMPRAS del espejo vive hoy en
# `canonicalize_shopping_food_name`, no inline en `aggregate_and_deduct_shopping_list`.
#
# `P1-VEG-BACKFILL-HONESTY` extrajo la cadena entera a un SSOT para que el lado TEXTO del backstop
# de cantidades canonicalizara con EXACTAMENTE el mismo código que el lado comprado. El espejo que
# P3-NEW-12 vigila (guard ↔ lista) no cambió: cambió en qué función vive una de las dos mitades.
# Se delimita por la siguiente `def` de columna 0 (ancla estructural) en vez de por una ventana de
# 30.000 chars, que es cómo caducan estos guards, y se ancla además que el agregador DELEGUE — sin
# esa segunda mitad, mover el bloque a un tercer sitio volvería a pasar mudo.
_LISTA_SIDE_FUNC = "canonicalize_shopping_food_name"


def _cuerpo_de(src: str, fn_name: str) -> str:
    """Cuerpo de una función top-level, delimitado por la siguiente `def` de columna 0."""
    start = src.find(f"def {fn_name}(")
    assert start >= 0, f"función `{fn_name}` no encontrada"
    m = re.search(r"^def \w+", src[start + 1:], re.MULTILINE)
    end = (start + 1 + m.start()) if m else len(src)
    return src[start:end]


def test_aggregator_side_wires_all_5(src: str):
    """El lado LISTA DE COMPRAS debe invocar las 5 fns nuevas."""
    body = _cuerpo_de(src, _LISTA_SIDE_FUNC)
    for fn in (
        "canonicalize_citricos",
        "canonicalize_tomate",
        "canonicalize_cebolla",
        "canonicalize_quesos_blancos_rd",
        "canonicalize_frutos_secos",
    ):
        assert fn in body, (
            f"P3-NEW-12 regresión: `{_LISTA_SIDE_FUNC}` ya "
            f"no invoca `{fn}`. Sin el wire del aggregator side, la lista "
            f"de compras output mostrará N líneas separadas para variantes "
            f"que deberían colapsar."
        )


def test_el_agregador_delega_en_el_ssot_de_canonicalizacion(src: str):
    """[review final · 2026-08-03] La mitad que faltaba: si `aggregate_and_deduct_shopping_list`
    deja de delegar, las 5 familias dejan de aplicarse a la lista real aunque sigan escritas en el
    SSOT — y el espejo guard↔lista se rompe sin que ningún test lo diga."""
    agg = _cuerpo_de(src, "aggregate_and_deduct_shopping_list")
    assert f"{_LISTA_SIDE_FUNC}(name, master_map)" in agg, (
        "P3-NEW-12 regresión: el agregador dejó de delegar en el SSOT de canonicalización."
    )


def test_marker_present_in_both_sites(src: str):
    """`P3-NEW-12` aparece en el bloque del guard y en el bloque del lado lista
    (anchor para grep). Las dos mitades del espejo tienen que ser correlacionables."""
    guard_idx = src.find("def _canonicalize_for_coherence(")
    guard_end = src.find("\ndef run_shopping_coherence_guard(", guard_idx)
    guard_body = src[guard_idx:guard_end]
    assert "P3-NEW-12" in guard_body, (
        "P3-NEW-12 regresión: marker no presente en el bloque del guard."
    )

    assert "P3-NEW-12" in _cuerpo_de(src, _LISTA_SIDE_FUNC), (
        "P3-NEW-12 regresión: marker no presente en el bloque del lado "
        "lista de compras. Sin él un revisor no correlaciona ambas mitades del "
        "wiring."
    )
