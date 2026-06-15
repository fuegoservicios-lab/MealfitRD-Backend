"""[P1-ALLERGEN-DERIVATIVES · 2026-06-14] Detección determinista de DERIVADOS / ingredientes
compuestos de alérgenos.

Cierra el gap P1 del audit clínico-de-precisión (2026-06-14): el swap quirúrgico y el backstop
`_scan_allergen_violations` no capturaban el alérgeno escondido en un compuesto (whey en un batido
para alérgico a lácteos, mayonesa para alérgico a huevo, miso/teriyaki para soya, cuscús/seitán
para gluten, anchoa/surimi para pescado, mazapán/nutella para frutos secos) → el único filtro era
el revisor LLM (falible). La extensión de `_ALLERGEN_SYNONYMS` es DETECCIÓN (no sustitución): el
backstop escala el residual a crítico → fallback allergen-free, manteniendo el sesgo de
sobre-detección intencional (un falso positivo es mejor que servir el alérgeno).

Cubre:
  A. `_scan_allergen_violations` detecta cada familia de derivados (funcional).
  B. Negativos: ingrediente inocuo + sin alergia NO disparan.
  C. Parser-based anchor (un renombre del marker o de las claves falla el test antes de prod).
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GO_PATH = _BACKEND_ROOT / "graph_orchestrator.py"


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


def _plan(ingredients):
    return {"days": [{"meals": [{"name": "Almuerzo", "ingredients": list(ingredients)}]}]}


# ════════════════════════════════════════════════════════════════════════════════════════════════
# A. Detección de derivados por familia (el alérgeno escondido en un compuesto)
# ════════════════════════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("allergy,ingredient", [
    # lácteos / lactosa
    ("lacteos", "1 scoop de whey"),
    ("lacteos", "30 g de proteina de suero"),
    ("lacteos", "2 bolas de helado de vainilla"),
    ("lacteos", "1 cda de dulce de leche"),
    ("lactosa", "1 taza de kefir"),
    # huevo
    ("huevo", "2 cdas de mayonesa"),
    ("huevo", "porcion de merengue"),
    ("huevo", "salsa holandesa"),
    # soya
    ("soya", "1 cda de miso"),
    ("soya", "tempeh a la plancha"),
    ("soya", "pollo en salsa teriyaki"),
    ("soya", "lecitina de soya"),
    # gluten
    ("gluten", "1 taza de cuscus"),
    ("gluten", "filete de seitan"),
    ("gluten", "sopa espesada con semola"),
    ("gluten", "vaso de malta morena"),
    # pescado
    ("pescado", "anchoas en aceite"),
    ("pescado", "palitos de surimi"),
    ("pescado", "un toque de salsa inglesa"),
    # frutos secos
    ("frutos secos", "barra de mazapan"),
    ("frutos secos", "1 cda de nutella"),
])
def test_derivative_is_detected(go, allergy, ingredient):
    """El backstop determinista flaggea el derivado para la alergia correspondiente."""
    violations = go._scan_allergen_violations(_plan([ingredient, "Arroz blanco"]), [allergy])
    assert violations, f"derivado '{ingredient}' NO detectado para alergia '{allergy}'"


# ════════════════════════════════════════════════════════════════════════════════════════════════
# B. Negativos — sin falsos positivos sobre ingredientes inocuos / sin alergia
# ════════════════════════════════════════════════════════════════════════════════════════════════
def test_innocuous_ingredient_not_flagged(go):
    """Un plato sin el alérgeno no dispara (arroz/pollo bajo alergia a lácteos)."""
    violations = go._scan_allergen_violations(
        _plan(["1 taza de arroz blanco", "150 g de pechuga de pollo", "ensalada verde"]),
        ["lacteos"])
    assert violations == [], violations


def test_no_allergy_no_violation(go):
    """Sin alergias declaradas → cero violaciones aunque haya derivados."""
    assert go._scan_allergen_violations(_plan(["1 scoop de whey", "2 cdas de mayonesa"]), []) == []


def test_word_boundary_no_substring_false_positive(go):
    """'miso' (soya) NO debe matchear dentro de 'permiso'/'compromiso' (frontera de palabra)."""
    violations = go._scan_allergen_violations(_plan(["plato sin compromiso de sabor"]), ["soya"])
    assert violations == [], violations


# ════════════════════════════════════════════════════════════════════════════════════════════════
# C. Parser-based anchor — un renombre del marker o de las claves derivadas falla el test
# ════════════════════════════════════════════════════════════════════════════════════════════════
def test_marker_and_derivative_keys_present():
    src = _GO_PATH.read_text(encoding="utf-8")
    assert "P1-ALLERGEN-DERIVATIVES" in src, "falta el marker P1-ALLERGEN-DERIVATIVES"
    # Términos derivados representativos de cada familia: si alguien revierte la extensión, falla.
    for term in ('"whey"', '"mayonesa"', '"miso"', '"seitan"', '"surimi"', '"mazapan"'):
        assert term in src, f"falta el derivado {term} en _ALLERGEN_SYNONYMS"
