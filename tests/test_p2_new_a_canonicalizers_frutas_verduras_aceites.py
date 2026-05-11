"""[P2-NEW-A · 2026-05-11] Tests parser-based + behavior tests para 3
canonicalizers nuevos: `canonicalize_frutas_tropicales`,
`canonicalize_verduras_hoja`, `canonicalize_aceites`.

Motivación:
    Pre-P2-NEW-A, los canonicalizers cubrían proteínas
    (`canonicalize_protein`, `canonicalize_fish_seafood`, `canonicalize_pavo`,
    `canonicalize_huevo`, `canonicalize_lacteo`), granos/legumbres
    (`canonicalize_grano`, `canonicalize_legumino`), tubérculos
    (`canonicalize_viveres`) y musáceas (`canonicalize_musaceae`). Faltaban
    3 familias cuyo gap silenciaba `cap_swallowed_modifier`:

      - Frutas tropicales RD: mango/piña/papaya-lechosa generaban N
        líneas en la lista de compras por preparación
        ("Mango verde" + "Mango maduro" → 2 líneas).
      - Verduras de hoja: variedades de lechuga (romana/americana/
        criolla) generaban N líneas — el usuario compra UNA lechuga.
      - Aceites: "aceite de oliva extra virgen" vs "aceite oliva"
        reportaba `cap_swallowed_modifier` falso positivo en el guard
        recetas↔lista.

Cierre P2-NEW-A:
    Cada canonicalizer es un helper que:
      - Acepta `name` (str o None).
      - Devuelve canonical fijo si matchea, `None` si no aplica
        (caller cae al siguiente canonicalizer).
      - Está wired bilateral: en `_canonicalize_for_coherence` del
        guard (símil presencia/magnitud) Y en
        `aggregate_and_deduct_shopping_list` (output al usuario).
        Sin esta simetría, presence diverge.

Tests:
    1. Cada helper existe en shopping_calculator.py con signatura
       `(name) -> str | None`.
    2. Behavior: ejemplos canónicos para cada familia.
    3. Cada helper está wired en AMBOS sitios bilaterales.
    4. Cross-link slug del marker.

Tooltip-anchor: P2-NEW-A-START | gap P2 audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_SC_PY = Path(__file__).resolve().parent.parent / "shopping_calculator.py"


def _extract_function_body(src: str, fn_name: str) -> str | None:
    pattern = re.compile(rf"def\s+{re.escape(fn_name)}\s*\(")
    m = pattern.search(src)
    if not m:
        return None
    start = m.start()
    next_def = re.search(r"\n(?:def\s|class\s)", src[start + 1:])
    end = (start + 1 + next_def.start()) if next_def else len(src)
    return src[start:end]


@pytest.fixture(scope="module")
def sc_src() -> str:
    return _SC_PY.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Existence — cada helper está definido con signature `(name) -> str | None`
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fn_name,expected_canonicals",
    [
        ("canonicalize_frutas_tropicales", {"Mango", "Piña", "Lechosa"}),
        ("canonicalize_verduras_hoja", {"Lechuga", "Espinaca", "Rúcula", "Acelga", "Berro"}),
        ("canonicalize_aceites", {"Aceite de oliva", "Aceite de girasol", "Aceite de coco", "Aceite de aguacate"}),
    ],
)
def test_canonicalizer_exists_with_expected_canonicals(sc_src, fn_name, expected_canonicals):
    """Cada canonicalizer debe existir + el cuerpo debe contener cada
    canonical esperado al menos una vez (return literal o regex de
    matching).

    Sin alguno de estos canonicals, el aggregator y el guard producirían
    nombres distintos al usuario y la simetría se rompe.
    """
    body = _extract_function_body(sc_src, fn_name)
    assert body is not None, (
        f"P2-NEW-A regresión: helper `{fn_name}` no existe en "
        f"shopping_calculator.py. Sin este canonicalizer, las variantes "
        f"de su familia generan N líneas en la lista de compras."
    )
    missing = {c for c in expected_canonicals if f"'{c}'" not in body and f'"{c}"' not in body}
    assert not missing, (
        f"P2-NEW-A regresión: `{fn_name}` no contiene los canonicals "
        f"esperados (return strings): {sorted(missing)}. Revisar las "
        f"reglas documentadas en el docstring del helper."
    )


# ---------------------------------------------------------------------------
# 2. Behavior tests por canonicalizer (cargado dinámicamente del source)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def loaded_helpers(sc_src) -> dict:
    """Carga las 3 funciones sin importar shopping_calculator completo
    (que arrastra deps pesadas). Aisladas en un namespace local con
    solo `re` disponible — suficiente para los 3 helpers.
    """
    ns: dict = {"re": re}
    for fn in (
        "canonicalize_frutas_tropicales",
        "canonicalize_verduras_hoja",
        "canonicalize_aceites",
    ):
        body = _extract_function_body(sc_src, fn)
        if body is None:
            raise AssertionError(f"Helper {fn} no encontrado para load")
        exec(body, ns)
    return ns


def test_frutas_tropicales_collapses_mango_variants(loaded_helpers):
    f = loaded_helpers["canonicalize_frutas_tropicales"]
    # Variantes que deben colapsar a Mango.
    for variant in ("Mango verde", "Mango maduro en almíbar", "1 mango", "mangos picados"):
        assert f(variant) == "Mango", f"esperaba Mango para {variant!r}, got {f(variant)!r}"


def test_frutas_tropicales_collapses_pina_with_and_without_tilde(loaded_helpers):
    f = loaded_helpers["canonicalize_frutas_tropicales"]
    assert f("piña") == "Piña"
    assert f("Pina rebanada") == "Piña"
    assert f("piñas frescas") == "Piña"


def test_frutas_tropicales_papaya_and_lechosa_collapse_to_lechosa(loaded_helpers):
    """`papaya` y `lechosa` ambos → "Lechosa" (canónico es-DO).
    LLM puede emitir cualquiera; el aggregator presenta el nombre criollo.
    """
    f = loaded_helpers["canonicalize_frutas_tropicales"]
    assert f("papaya") == "Lechosa"
    assert f("Lechosa madura") == "Lechosa"
    assert f("papayas frescas") == "Lechosa"


def test_frutas_tropicales_avoids_dairy_false_match(loaded_helpers):
    """`lechoso/lechosos` (adjetivo masculino lácteo) NO debe colapsar
    a Lechosa. Solo `lechosa/lechosas` (femenino, papaya en RD).
    """
    f = loaded_helpers["canonicalize_frutas_tropicales"]
    assert f("postre lechoso") is None
    assert f("yogurt lechoso") is None


def test_frutas_tropicales_avoids_papa_false_match(loaded_helpers):
    """`papa/papas` (tubérculo) está cubierto por canonicalize_viveres,
    NO por frutas. Evitar matchear prefijo de "papaya".
    """
    f = loaded_helpers["canonicalize_frutas_tropicales"]
    assert f("papa hervida") is None
    assert f("papas asadas") is None


def test_verduras_hoja_collapses_lechuga_varieties(loaded_helpers):
    """Variedades de lechuga → "Lechuga"."""
    vh = loaded_helpers["canonicalize_verduras_hoja"]
    for variant in (
        "Lechuga romana",
        "Lechuga americana picada",
        "lechuga criolla",
        "lechugas mixtas",
    ):
        assert vh(variant) == "Lechuga", f"variant {variant!r} got {vh(variant)!r}"


def test_verduras_hoja_other_families(loaded_helpers):
    vh = loaded_helpers["canonicalize_verduras_hoja"]
    assert vh("espinaca cocida") == "Espinaca"
    assert vh("rucula fresca") == "Rúcula"
    assert vh("rúcula") == "Rúcula"
    assert vh("acelgas salteadas") == "Acelga"
    assert vh("berro") == "Berro"


def test_verduras_hoja_returns_none_for_unrelated(loaded_helpers):
    vh = loaded_helpers["canonicalize_verduras_hoja"]
    assert vh("mango") is None
    assert vh("tomate") is None
    assert vh("") is None
    assert vh(None) is None


def test_aceites_oliva_variants_collapse(loaded_helpers):
    """Todas las variantes de oliva → "Aceite de oliva" incluyendo
    "AOVE" (extra virgen).
    """
    ac = loaded_helpers["canonicalize_aceites"]
    for variant in (
        "Aceite de oliva extra virgen",
        "aceite oliva",
        "AOVE",
        "aceite de oliva virgen prensado en frío",
        "1 cdita de aceite de oliva",
    ):
        assert ac(variant) == "Aceite de oliva", f"variant {variant!r} got {ac(variant)!r}"


def test_aceites_other_types_preserved_separately(loaded_helpers):
    """Cada tipo de aceite PRESERVADO (no colapsan entre sí)."""
    ac = loaded_helpers["canonicalize_aceites"]
    assert ac("aceite de girasol") == "Aceite de girasol"
    assert ac("aceite girasol") == "Aceite de girasol"
    assert ac("aceite de coco") == "Aceite de coco"
    assert ac("aceite de aguacate") == "Aceite de aguacate"


def test_aceites_vegetal_generic_not_canonicalized(loaded_helpers):
    """`aceite vegetal` genérico NO se canonicaliza — el LLM no
    especificó tipo y consolidarlo ocultaría esa ambigüedad.
    """
    ac = loaded_helpers["canonicalize_aceites"]
    assert ac("aceite vegetal") is None
    assert ac("aceite") is None  # demasiado genérico


def test_aceites_dairy_not_matched(loaded_helpers):
    ac = loaded_helpers["canonicalize_aceites"]
    assert ac("mantequilla") is None
    assert ac("margarina") is None
    assert ac("ghee") is None


# ---------------------------------------------------------------------------
# 3. Wiring bilateral — guard + aggregator referencian cada helper
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def guard_body(sc_src) -> str:
    body = _extract_function_body(sc_src, "_canonicalize_for_coherence")
    assert body is not None, (
        "P2-NEW-A sanity: `_canonicalize_for_coherence` no encontrada. "
        "El guard recetas↔lista cambió de nombre? Si sí, actualizar este test."
    )
    return body


@pytest.fixture(scope="module")
def aggregator_body(sc_src) -> str:
    body = _extract_function_body(sc_src, "aggregate_and_deduct_shopping_list")
    assert body is not None, (
        "P2-NEW-A sanity: `aggregate_and_deduct_shopping_list` no encontrada."
    )
    return body


@pytest.mark.parametrize(
    "fn_name",
    [
        "canonicalize_frutas_tropicales",
        "canonicalize_verduras_hoja",
        "canonicalize_aceites",
    ],
)
def test_canonicalizer_wired_in_guard(guard_body, fn_name):
    """Cada helper debe ser llamado dentro de `_canonicalize_for_coherence`.
    Sin esto, el guard reporta presence asimétrica vs el aggregator
    (recetas dicen X, el guard canonicaliza solo el lado aggregated).
    """
    assert fn_name in guard_body, (
        f"P2-NEW-A regresión: `{fn_name}` no está wired en "
        f"`_canonicalize_for_coherence`. El guard no canonicalizará las "
        f"variantes de esa familia → falsos positivos `cap_swallowed_modifier`."
    )


@pytest.mark.parametrize(
    "fn_name",
    [
        "canonicalize_frutas_tropicales",
        "canonicalize_verduras_hoja",
        "canonicalize_aceites",
    ],
)
def test_canonicalizer_wired_in_aggregator(aggregator_body, fn_name):
    """Cada helper debe ser llamado dentro de
    `aggregate_and_deduct_shopping_list`. Sin esto, la lista de compras
    muestra variantes separadas al usuario aunque el guard internamente
    las consolide.
    """
    assert fn_name in aggregator_body, (
        f"P2-NEW-A regresión: `{fn_name}` no está wired en "
        f"`aggregate_and_deduct_shopping_list`. La lista de compras "
        f"presentará variantes separadas (mango verde + mango maduro = "
        f"2 líneas) cuando shopping-wise son el mismo producto."
    )


# ---------------------------------------------------------------------------
# 4. Cross-link slug del marker
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    expected_slug = "p2_new_a"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "El nombre de este archivo debe contener el slug del P-fix "
        "(`p2_new_a`) para que el cross-link "
        "`test_p2_hist_audit_14_marker_test_link` lo matchee."
    )
