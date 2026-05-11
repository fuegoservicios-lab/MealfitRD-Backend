"""[P2-NEW-8 · 2026-05-11] SSOT helper `_consolidate_inline_canon` para
4 reglas inline (Huevo / Ñame / Miel / Ajo) en shopping_calculator.py.

Antes de P2-NEW-8 las 4 reglas vivían duplicadas inline:
    - `_canonicalize_for_coherence` (cuerpo del guard recetas↔lista,
      ~3450) — usaba if/elif/elif/elif/else.
    - `aggregate_and_deduct_shopping_list` (aggregator que produce el
      output de la lista de compras, ~3865) — usaba 4 ifs secuenciales.

Drift risk pre-P2-NEW-8: si una regla se actualizaba en un sitio sin
tocar el otro, el guard reportaba false positives ("Huevo missing" o
"unit_mismatch") porque `expected_sum_from_recipes` y el aggregator
producían canónicos distintos para el mismo nombre. Pavo ya tenía
`test_p3_4_canonicalize_pavo_mirrors_aggregator` como espejo dedicado;
estas 4 reglas (Huevo, Ñame, Miel, Ajo) no tenían cobertura análoga.

Cierre P2-NEW-8:
    - Helper `_consolidate_inline_canon(name) -> str | None` colocado
      después de `canonicalize_legumino` para vecindad con los otros 7
      canonicalizers (pavo/protein/fish_seafood/huevo/lacteo/grano/
      legumino).
    - Ambos call sites refactorizados para invocar el helper.
    - Tests: funcionales (cada regla + exclusión `ajo en polvo` + no-
      matches) + parser-based drift detection (ambos sites usan helper +
      regex inline ya no vive fuera del helper).

Tooltip-anchor: P2-NEW-8-INLINE-CANON | espejo Huevo/Ñame/Miel/Ajo
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest


# El import de shopping_calculator dispara setup de master_list/embed cache.
# Desactivamos rutas no necesarias con env vars defensivas antes del import.
os.environ.setdefault("MEALFIT_DISABLE_SEMANTIC_CACHE", "1")

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from shopping_calculator import _consolidate_inline_canon  # noqa: E402


_SHOPPING_CALCULATOR_PY = _BACKEND_ROOT / "shopping_calculator.py"


def _extract_function_body(src: str, fn_name: str) -> str:
    """Extrae cuerpo de la función desde `def fn_name(` hasta el
    siguiente `\\ndef ` top-level. Mismo patrón que helpers paralelos
    en la suite."""
    m = re.search(rf"^def\s+{re.escape(fn_name)}\s*\(", src, re.MULTILINE)
    if not m:
        raise AssertionError(f"No se encontró def {fn_name}.")
    start = m.start()
    next_def = re.search(r"\n^def\s", src[start + 1:], re.MULTILINE)
    end = (start + 1 + next_def.start()) if next_def else len(src)
    return src[start:end]


# ---------------------------------------------------------------------------
# 1. Functional tests — las 4 reglas producen el canónico esperado
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    # Huevo: prefijo `huevo(s)?`, claras/yemas de huevo
    ("Huevo", "Huevo"),
    ("huevo", "Huevo"),
    ("Huevos", "Huevo"),
    ("HUEVOS", "Huevo"),
    ("Huevos enteros", "Huevo"),
    ("Claras de huevo", "Huevo"),
    ("clara de huevo", "Huevo"),
    ("Yemas de huevo", "Huevo"),
    ("yema de huevo", "Huevo"),
    # Ñame: con o sin tilde (LLM puede emitir cualquier forma)
    ("Ñame", "Ñame"),
    ("ñame", "Ñame"),
    ("Name", "Ñame"),
    ("Ñame blanco", "Ñame"),
    ("Name amarillo", "Ñame"),
    # Miel: prefijo
    ("Miel", "Miel"),
    ("miel", "Miel"),
    ("Miel cruda", "Miel"),
    ("Miel pura", "Miel"),
    # Ajo: prefijo o "diente(s) de ajo", sin 'polvo'
    ("Ajo", "Ajo"),
    ("ajo", "Ajo"),
    ("Ajo majado", "Ajo"),
    ("ajo triturado", "Ajo"),
    ("Diente de ajo", "Ajo"),
    ("dientes de ajo", "Ajo"),
])
def test_canonicalizes_matching_names(name, expected):
    """Cada regla produce su canónico fijo (case-insensitive sobre input,
    case-preserving sobre output)."""
    assert _consolidate_inline_canon(name) == expected, (
        f"P2-NEW-8 regresión: `{name}` debería canonicalizar a "
        f"`{expected}` pero el helper devolvió "
        f"`{_consolidate_inline_canon(name)}`."
    )


@pytest.mark.parametrize("name", [
    "Ajo en polvo",
    "Ajo polvo",
    "AJO EN POLVO",
    "ajo en polvo molido",
])
def test_ajo_en_polvo_excluded(name):
    """`ajo en polvo` es categoría distinta (especia seca) — no debe
    colapsar a 'Ajo' fresco. La exclusión por palabra 'polvo' es
    defensiva: shopping list separa especias secas vs frescos."""
    assert _consolidate_inline_canon(name) is None, (
        f"P2-NEW-8 regresión: `{name}` no debe canonicalizar a 'Ajo' "
        f"(es especia, categoría distinta). El helper devolvió "
        f"`{_consolidate_inline_canon(name)}`."
    )


@pytest.mark.parametrize("name", [
    "Pollo",
    "Pavo",
    "Yuca",
    "Plátano",
    "Pasta",
    "Arroz",
    "Salmón",
    "Lentejas",
    "",
    None,
])
def test_non_matches_return_none(name):
    """Inputs que no matchean ninguna de las 4 reglas → None. El caller
    mantiene el name original o cae a otros canonicalizers (pavo/protein/
    fish_seafood/huevo/lacteo/grano/legumino)."""
    assert _consolidate_inline_canon(name) is None


# ---------------------------------------------------------------------------
# 2. Drift detection — ambos call sites usan el helper SSOT
# ---------------------------------------------------------------------------
def test_both_call_sites_invoke_helper():
    """`_canonicalize_for_coherence` Y `aggregate_and_deduct_shopping_list`
    deben invocar `_consolidate_inline_canon`. Si alguno reintroduce las
    reglas inline, falsea el contrato SSOT."""
    src = _SHOPPING_CALCULATOR_PY.read_text(encoding="utf-8")
    for fn_name in ("_canonicalize_for_coherence", "aggregate_and_deduct_shopping_list"):
        body = _extract_function_body(src, fn_name)
        assert "_consolidate_inline_canon(" in body, (
            f"P2-NEW-8 regresión: `{fn_name}` no invoca "
            f"`_consolidate_inline_canon(...)`. Si las reglas inline se "
            f"reintrodujeron, falsea el contrato SSOT — drift entre call "
            f"sites es exactamente el bug que este helper cierra."
        )


# Patrones legacy que NO deben aparecer fuera del helper (cada regla).
_LEGACY_PATTERNS = {
    "huevo": r"\^\(huevos\?\|claras\?",
    "name":  r"\^\[ñn\]ame\\b",
    "miel":  r"\^miel\\b",
    "ajo":   r"\^ajo\\b",
}


@pytest.mark.parametrize("fn_name", [
    "_canonicalize_for_coherence",
    "aggregate_and_deduct_shopping_list",
])
def test_call_site_bodies_dont_contain_legacy_regex(fn_name):
    """Los cuerpos de los 2 call sites NO deben contener los regex
    inline legacy. Si aparecen, una copia legacy sigue viva y el SSOT
    está roto."""
    src = _SHOPPING_CALCULATOR_PY.read_text(encoding="utf-8")
    body = _extract_function_body(src, fn_name)
    for rule_name, pattern in _LEGACY_PATTERNS.items():
        assert not re.search(pattern, body), (
            f"P2-NEW-8 regresión: `{fn_name}` contiene regex legacy de "
            f"`{rule_name}` (`{pattern}`) — debería usar "
            f"`_consolidate_inline_canon(...)` en su lugar. SSOT roto."
        )


def test_helper_body_contains_all_four_rules():
    """`_consolidate_inline_canon` debe contener los 4 regex patterns
    (uno por cada regla). Si alguno se borra accidentalmente, el helper
    pierde cobertura y un call site puede empezar a regresar None
    cuando antes consolidaba."""
    src = _SHOPPING_CALCULATOR_PY.read_text(encoding="utf-8")
    body = _extract_function_body(src, "_consolidate_inline_canon")
    for rule_name, pattern in _LEGACY_PATTERNS.items():
        assert re.search(pattern, body), (
            f"P2-NEW-8 regresión: `_consolidate_inline_canon` NO "
            f"contiene la regla `{rule_name}` (patrón `{pattern}`). "
            f"Si la regla cambió de forma, actualizar este test al "
            f"mismo tiempo para mantener el invariante de cobertura."
        )


# ---------------------------------------------------------------------------
# 3. Anchor cross-link slug
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    """Slug del filename matchea el marker `P2-NEW-8`."""
    expected_slug = "p2_new_8"
    assert expected_slug in __file__.replace("\\", "/").lower()
