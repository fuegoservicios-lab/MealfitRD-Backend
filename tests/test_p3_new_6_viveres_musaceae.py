"""[P3-NEW-6 · 2026-05-11] Canonicalizers para víveres dominicanos
(yuca/yautía/batata/papa/auyama) y musáceas (plátano/guineo).

Bug original (audit 2026-05-11):
    Recetas con preparaciones múltiples de un mismo vívere se aggregaban
    como líneas separadas en la lista de compras:
      - "Yuca hervida", "Yuca con mojo", "Yuca al ajillo" → 3 líneas.
      - "Plátano verde para mangú", "Plátano maduro frito",
        "Plátano maduro en almíbar" → 3 líneas.
    Shopping-wise, son el MISMO producto. Inflaba la lista y degradaba
    la UX del usuario en supermercado (3 chequeos para 1 producto).

Cierre P3-NEW-6:
    Dos helpers paralelos a la familia P1-NEW-2 (`canonicalize_huevo`,
    `canonicalize_lacteo`, `canonicalize_grano`, `canonicalize_legumino`):
      - `canonicalize_viveres(name) -> "Yuca"/"Yautía"/"Batata"/"Papa"/
        "Auyama" | None`
      - `canonicalize_musaceae(name) -> "Plátano"/"Guineo" | None`

    Wired bilateral en `_canonicalize_for_coherence` (coherence side)
    y en `aggregate_and_deduct_shopping_list` (aggregator). El bilateral
    es OBLIGATORIO: si solo el aggregator consolida y el guard no,
    expected_sum_from_recipes ve "Yuca hervida" y "Yuca con mojo" como
    separadas pero el aggregator emite solo "Yuca" → guard reporta
    `cap_swallowed_modifier` falsos positivos.

Tooltip-anchor: P3-NEW-6-VIVERES-MUSACEAE | shopping consolidation
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest


os.environ.setdefault("MEALFIT_DISABLE_SEMANTIC_CACHE", "1")

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from shopping_calculator import canonicalize_viveres, canonicalize_musaceae  # noqa: E402


_SHOPPING_CALCULATOR_PY = _BACKEND_ROOT / "shopping_calculator.py"


def _extract_function_body(src: str, fn_name: str) -> str:
    m = re.search(rf"^def\s+{re.escape(fn_name)}\s*\(", src, re.MULTILINE)
    if not m:
        raise AssertionError(f"No se encontró def {fn_name}.")
    start = m.start()
    next_def = re.search(r"\n^def\s", src[start + 1:], re.MULTILINE)
    end = (start + 1 + next_def.start()) if next_def else len(src)
    return src[start:end]


# ---------------------------------------------------------------------------
# canonicalize_viveres — functional tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    # Yuca: prefijo + preparaciones comunes RD
    ("Yuca", "Yuca"),
    ("yuca", "Yuca"),
    ("Yucas", "Yuca"),
    ("Yuca hervida", "Yuca"),
    ("Yuca con mojo", "Yuca"),
    ("yuca al ajillo", "Yuca"),
    ("Yuca rallada", "Yuca"),
    ("Sancocho con yuca", "Yuca"),
    # Yautía: con y sin tilde, singular y plural
    ("Yautía", "Yautía"),
    ("yautia", "Yautía"),
    ("Yautías", "Yautía"),
    ("yautias amarillas", "Yautía"),
    # Batata
    ("Batata", "Batata"),
    ("Batatas", "Batata"),
    ("batata asada", "Batata"),
    # Papa: incluye preparaciones
    ("Papa", "Papa"),
    ("Papas", "Papa"),
    ("Papa hervida", "Papa"),
    ("Papas fritas", "Papa"),
    ("papa al horno", "Papa"),
    # Auyama (calabaza criolla)
    ("Auyama", "Auyama"),
    ("Auyamas", "Auyama"),
    ("Sopa de auyama", "Auyama"),
])
def test_viveres_canonicalizes_matching_names(name, expected):
    """Cada regla produce canónico fijo (case-insensitive)."""
    assert canonicalize_viveres(name) == expected, (
        f"P3-NEW-6 regresión: `{name}` debería canonicalizar a "
        f"`{expected}` pero el helper devolvió "
        f"`{canonicalize_viveres(name)}`."
    )


@pytest.mark.parametrize("name", [
    "Papaya",
    "Papaya madura",
    "Jugo de papaya",
    "PAPAYA",
    "papayas verdes",
])
def test_papaya_does_not_match_papa(name):
    """`papaya` es fruta, NO tubérculo. La regla Papa debe excluirla
    explícitamente (el regex `\\bpapas?\\b` matchearía "papa" dentro de
    "papaya" sin el guard `'papaya' not in n_low`)."""
    assert canonicalize_viveres(name) is None, (
        f"P3-NEW-6 regresión: `{name}` NO debe canonicalizar a 'Papa' "
        f"(es fruta tropical, categoría shopping distinta). El helper "
        f"devolvió `{canonicalize_viveres(name)}`."
    )


@pytest.mark.parametrize("name", [
    "Pollo",
    "Pavo",
    "Calabacín",  # NO es auyama
    "Calabacines",
    "Tayota",
    "Remolacha",
    "Zanahoria",
    "Plátano",   # musácea, no vívere
    "Guineo",    # musácea
    "Ñame",      # cubierto por _consolidate_inline_canon
    "Arroz",
    "",
    None,
])
def test_viveres_non_matches_return_none(name):
    """Inputs fuera del scope del helper → None. El caller cae al
    siguiente canonicalizer (musaceae) o al fallback."""
    assert canonicalize_viveres(name) is None


# ---------------------------------------------------------------------------
# canonicalize_musaceae — functional tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    # Plátano: cualquier estado o preparación
    ("Plátano", "Plátano"),
    ("platano", "Plátano"),
    ("Plátanos", "Plátano"),
    ("platanos", "Plátano"),
    ("Plátano verde", "Plátano"),
    ("Plátano maduro", "Plátano"),
    ("Plátano verde para mangú", "Plátano"),
    ("Plátano maduro frito", "Plátano"),
    ("Plátano maduro en almíbar", "Plátano"),
    ("Plátanos pintones", "Plátano"),
    ("PLATANO", "Plátano"),
    # Guineo (banano criollo)
    ("Guineo", "Guineo"),
    ("Guineos", "Guineo"),
    ("guineo verde", "Guineo"),
    ("Guineo maduro", "Guineo"),
])
def test_musaceae_canonicalizes_matching_names(name, expected):
    """Plátanos y guineos colapsan a su canónico fijo. Estado madurez
    es variable temporal, NO producto distinto."""
    assert canonicalize_musaceae(name) == expected, (
        f"P3-NEW-6 regresión: `{name}` debería canonicalizar a "
        f"`{expected}` pero el helper devolvió "
        f"`{canonicalize_musaceae(name)}`."
    )


@pytest.mark.parametrize("name", [
    "Pollo",
    "Yuca",
    "Papa",
    "Manzana",
    "Naranja",
    "Pera",
    "Banana",  # variante anglo — no normalizada
    "",
    None,
])
def test_musaceae_non_matches_return_none(name):
    """Inputs fuera del scope (no es plátano ni guineo) → None."""
    assert canonicalize_musaceae(name) is None


# ---------------------------------------------------------------------------
# Drift detection: ambos call sites usan los helpers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("helper_name", [
    "canonicalize_viveres",
    "canonicalize_musaceae",
])
def test_coherence_side_invokes_helper(helper_name):
    """`_canonicalize_for_coherence` (lado guard) debe invocar ambos
    canonicalizers en la cadena de fallback. Sin esto, el guard
    reporta false positives `cap_swallowed_modifier` cuando el
    aggregator consolida pero el guard sigue viendo nombres separados.
    """
    src = _SHOPPING_CALCULATOR_PY.read_text(encoding="utf-8")
    body = _extract_function_body(src, "_canonicalize_for_coherence")
    assert f"{helper_name}(" in body, (
        f"P3-NEW-6 regresión: `_canonicalize_for_coherence` no invoca "
        f"`{helper_name}(...)`. El bilateral con el aggregator es "
        f"obligatorio — si solo el aggregator consolida, el guard "
        f"reporta `cap_swallowed_modifier` falsos positivos."
    )


@pytest.mark.parametrize("helper_name", [
    "canonicalize_viveres",
    "canonicalize_musaceae",
])
def test_aggregator_invokes_helper(helper_name):
    """`aggregate_and_deduct_shopping_list` (aggregator) debe invocar
    ambos canonicalizers. Sin esto, las preparaciones múltiples del
    mismo producto siguen como líneas separadas en la shopping list.
    """
    src = _SHOPPING_CALCULATOR_PY.read_text(encoding="utf-8")
    body = _extract_function_body(src, "aggregate_and_deduct_shopping_list")
    assert f"{helper_name}(" in body, (
        f"P3-NEW-6 regresión: `aggregate_and_deduct_shopping_list` no "
        f"invoca `{helper_name}(...)`. Sin esto, recetas con "
        f"preparaciones múltiples del mismo vívere/musácea siguen "
        f"generando líneas separadas en la lista de compras (bug "
        f"original P3-NEW-6)."
    )


def test_aggregator_calls_happen_after_inline_canon():
    """En el aggregator, los nuevos helpers viven en la rama `else` del
    bloque `_consolidate_inline_canon` (después de la consolidación de
    Huevo/Ñame/Miel/Ajo). Esto preserva precedence:
      - Ñame ya tiene canónico fijo via inline_canon → no entra al
        viveres path (que NO incluye Ñame intencionalmente).
    """
    src = _SHOPPING_CALCULATOR_PY.read_text(encoding="utf-8")
    body = _extract_function_body(src, "aggregate_and_deduct_shopping_list")
    # Encontrar la posición del bloque `_consolidate_inline_canon`.
    inline_pos = body.find("_consolidate_inline_canon(canonical_name)")
    assert inline_pos > 0, (
        "P3-NEW-6 regresión: no se encontró call `_consolidate_inline_canon"
        "(canonical_name)` en aggregator. ¿Refactor masivo?"
    )
    viveres_pos = body.find("canonicalize_viveres(")
    assert viveres_pos > inline_pos, (
        "P3-NEW-6 regresión: `canonicalize_viveres` debe llamarse DESPUÉS "
        "del bloque `_consolidate_inline_canon` para preservar precedence "
        "(Ñame ya cubierto por inline_canon)."
    )


# ---------------------------------------------------------------------------
# Anchor cross-link
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    """Slug del filename matchea el marker `P3-NEW-6`."""
    expected_slug = "p3_new_6"
    assert expected_slug in __file__.replace("\\", "/").lower()
