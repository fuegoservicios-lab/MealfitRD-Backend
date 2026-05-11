"""[P1-AUDIT-2 · 2026-05-10] `canonicalize_fish_seafood` cubre el modo
de falso positivo no cubierto por `canonicalize_protein` (pollo/cerdo/res)
ni `canonicalize_pavo`.

Bug original (audit 2026-05-10):
    Recetas con pescados/mariscos (filete de salmón, camarones a la
    plancha, tilapia frita) producían `cap_swallowed_modifier` falso
    positivo: el aggregator agrupaba "Salmón" pero el guard veía
    "Filete de salmón guisado" en receta → canonical-de-receta ≠
    canonical-de-lista → silent miss del yield_uncovered.

Fix:
    Helper `canonicalize_fish_seafood` espejado a `canonicalize_protein`
    pero per-species (cada especie es su propio canónico). Knob
    `MEALFIT_COHERENCE_FISH_KEYWORDS` para extensibilidad runtime.
    Wired en `_canonicalize_for_coherence` después de protein y antes
    del fallback genérico.

Tests funcionales (no parser-based — el helper es PURE function):
    - Casos positivos: filete de salmón → 'Salmón', camarones plancha
      → 'Camarón', tilapia frita → 'Tilapia', langostinos al ajillo
      → 'Langostino'.
    - Plurales: camarones/camaron → 'Camarón'.
    - Tildes: salmon/salmón → 'Salmón'.
    - Multi-match patológico: "mero con salmón" → None.
    - Exclusiones: atún en lata, salmón ahumado, fingers de pescado,
      surimi → None.
    - Empty/None input → None.

Test parser-based:
    - Helper wired en _canonicalize_for_coherence después de protein.
    - Knob MEALFIT_COHERENCE_FISH_KEYWORDS leído via _knob_env_str.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from shopping_calculator import canonicalize_fish_seafood


# ---------------------------------------------------------------------------
# 1. Casos positivos canónicos
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    # Fish — cuts + de + especie
    ("filete de salmón", "Salmón"),
    ("filete de salmon", "Salmón"),  # sin tilde
    ("filete de tilapia", "Tilapia"),
    ("filete de mero", "Mero"),
    # Fish — cooking states
    ("salmón guisado", "Salmón"),
    ("tilapia frita", "Tilapia"),
    ("dorado al horno", "Dorado"),
    ("atún sellado", "Atún"),
    # Fish — exact match
    ("Salmón", "Salmón"),
    ("tilapia", "Tilapia"),
    ("MERO", "Mero"),
    # Fish — plural → singular canonical
    ("salmones", "Salmón"),
    ("tilapias", "Tilapia"),
    ("sardinas", "Sardina"),
    # Seafood — singular/plural
    ("camarón", "Camarón"),
    ("camarones", "Camarón"),
    ("camaron", "Camarón"),  # sin tilde
    ("langostino", "Langostino"),
    ("langostinos", "Langostino"),
    ("calamar", "Calamar"),
    ("calamares", "Calamar"),
    # Seafood — preparaciones (caso típico del bug)
    ("camarones a la plancha", "Camarón"),
    ("langostinos al ajillo", "Langostino"),
    ("calamares fritos", "Calamar"),
    ("almejas al vapor", "Almeja"),
    ("pulpo a la gallega", "Pulpo"),
    # Seafood — preserved species
    ("vieiras", "Vieira"),
    ("mejillones", "Mejillón"),
    ("jaibas", "Jaiba"),
])
def test_canonical_positives(name, expected):
    """Casos del bug del audit: el helper debe colapsar al canónico
    base independiente de preparación / plural / tilde."""
    assert canonicalize_fish_seafood(name) == expected, (
        f"P1-AUDIT-2: `canonicalize_fish_seafood({name!r})` "
        f"debe devolver {expected!r}. Si falla, el guard reportará "
        f"false positive cap_swallowed_modifier para esa receta."
    )


# ---------------------------------------------------------------------------
# 2. Multi-match patológico
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", [
    "mero con salmón",     # platillo mixto raro
    "ceviche tilapia mero",
    "atún y bacalao",
])
def test_multi_canonical_returns_none(name):
    """Si el nombre menciona ≥2 especies distintas, no canonicalizamos
    (no es claro cuál gana — patológico)."""
    assert canonicalize_fish_seafood(name) is None, (
        f"P1-AUDIT-2: `canonicalize_fish_seafood({name!r})` debe "
        f"devolver None — multi-match patológico."
    )


# ---------------------------------------------------------------------------
# 3. Exclusiones — productos derivados
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", [
    "atún en lata",
    "atún enlatado",
    "sardinas enlatadas",
    "salmón ahumado",
    "bacalao ahumado",
    "fingers de pescado",
    "palitos de pescado",
    "nuggets de tilapia",
    "croqueta de bacalao",
    "surimi",
    "sucedáneo de cangrejo",
])
def test_derived_products_return_none(name):
    """Productos derivados deli (enlatado, ahumado, fingers, surimi)
    NO equivalen al pescado fresco — el master_map debe canonicalizar
    eso por su lado. Aquí retornamos None para no colapsar magnitudes."""
    assert canonicalize_fish_seafood(name) is None, (
        f"P1-AUDIT-2: `canonicalize_fish_seafood({name!r})` debe "
        f"devolver None (producto derivado). Colapsarlo a fresh "
        f"causaría falso negativo en presence (lista de compra "
        f"diría 'compraste 200g salmón' cuando es enlatado)."
    )


# ---------------------------------------------------------------------------
# 4. Empty / None / nombres irrelevantes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", [
    None,
    "",
    "   ",
    "pollo guisado",  # no es pescado
    "ñame",
    "arroz blanco",
    "carne molida de res",
])
def test_non_fish_returns_none(name):
    assert canonicalize_fish_seafood(name) is None, (
        f"P1-AUDIT-2: `canonicalize_fish_seafood({name!r})` debe "
        f"devolver None — no es del dominio del helper."
    )


# ---------------------------------------------------------------------------
# 5. Knob runtime — extensibilidad sin redeploy
# ---------------------------------------------------------------------------
def test_knob_extra_keywords_added(monkeypatch):
    """Knob `MEALFIT_COHERENCE_FISH_KEYWORDS` permite añadir especies
    sin redeploy. Formato `kw:Canon,kw2:Canon2`."""
    monkeypatch.setenv(
        "MEALFIT_COHERENCE_FISH_KEYWORDS",
        "ostra:Ostra,ostras:Ostra,boquerón:Boquerón",
    )
    assert canonicalize_fish_seafood("ostras a la parrilla") == "Ostra"
    assert canonicalize_fish_seafood("boquerón frito") == "Boquerón"


def test_knob_empty_is_default(monkeypatch):
    """Default vacío: el knob no extiende el mapping."""
    monkeypatch.setenv("MEALFIT_COHERENCE_FISH_KEYWORDS", "")
    # "ostra" NO está en el default → None.
    assert canonicalize_fish_seafood("ostra a la parrilla") is None


def test_knob_malformed_skipped_silently(monkeypatch):
    """Items mal formados (sin `:` o vacíos) se ignoran. El knob mal
    escrito NO debe romper el guard."""
    monkeypatch.setenv(
        "MEALFIT_COHERENCE_FISH_KEYWORDS",
        "esto-no-tiene-colon, :Canon, kw:, ostra:Ostra",
    )
    # Solo el item válido se añadió.
    assert canonicalize_fish_seafood("ostra fresca") == "Ostra"
    # Defaults siguen funcionando.
    assert canonicalize_fish_seafood("camarones") == "Camarón"


# ---------------------------------------------------------------------------
# 6. Parser-based: wired en _canonicalize_for_coherence
# ---------------------------------------------------------------------------
def test_helper_wired_in_canonicalize_for_coherence():
    """El helper debe llamarse desde `_canonicalize_for_coherence`
    DESPUÉS de protein y ANTES del fallback genérico. Sin esto, el
    guard no usa la canonicalización aunque el helper exista."""
    _SC = Path(__file__).resolve().parent.parent / "shopping_calculator.py"
    src = _SC.read_text(encoding="utf-8")
    assert "canonicalize_fish_seafood(raw_name)" in src, (
        "P1-AUDIT-2 regresión: `canonicalize_fish_seafood` no se llama "
        "sobre `raw_name` en `_canonicalize_for_coherence`. El helper "
        "está definido pero no wired al pipeline — el guard sigue con "
        "el comportamiento previo (false positives en pescados)."
    )
    assert "canonicalize_fish_seafood(canonical)" in src, (
        "P1-AUDIT-2 regresión: `canonicalize_fish_seafood` no se llama "
        "sobre `canonical` en `_canonicalize_for_coherence`. Falta "
        "cobertura para el caso en que master_map ya canonicalizó."
    )


def test_knob_registers_with_knobs_registry():
    """Knob debe registrarse en `_KNOBS_REGISTRY` para aparecer en
    `/admin/knobs`. Bypasea `_knob_env_str` (case-sensitive) pero
    llama `_register_knob` manualmente."""
    _SC = Path(__file__).resolve().parent.parent / "shopping_calculator.py"
    src = _SC.read_text(encoding="utf-8")
    # Localizar la función helper.
    m = re.search(
        r"def\s+_get_extra_fish_seafood_keywords\s*\(\s*\)[^:]*:(.*?)(?=^def\s)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, "P1-AUDIT-2: `_get_extra_fish_seafood_keywords` no encontrada."
    body = m.group(1)
    assert "MEALFIT_COHERENCE_FISH_KEYWORDS" in body, (
        "P1-AUDIT-2 regresión: knob `MEALFIT_COHERENCE_FISH_KEYWORDS` "
        "ya no se referencia en `_get_extra_fish_seafood_keywords`."
    )
    assert "_register_knob" in body, (
        "P1-AUDIT-2 regresión: `_get_extra_fish_seafood_keywords` no "
        "llama `_register_knob`. Sin esto, el knob no aparece en "
        "`/admin/knobs` y rollback en caliente no es observable."
    )
