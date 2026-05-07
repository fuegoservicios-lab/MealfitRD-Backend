"""[P1-3] Tests para normalización de unidades híbridas (paquete↔g) en
`aggregate_and_deduct_shopping_list`.

Bug original (audit P1-3):
  El bloque que convierte unidades de contenedor (paquete/pote/lata/...)
  a gramos requería DOS condiciones AND:
    1. `container_weight_g > 0` en master_ingredients.
    2. La unidad textual está en un set hardcodeado limitado.
  Si master NO tenía el peso curado (común para SKUs sin curar) o el usuario
  tipeaba un alias no contemplado (ej. "1 caja de leche", "1 tetra de jugo",
  "1 galón de agua"), la unidad NO se normalizaba a gramos. Resultado: el
  inventario quedaba en `units['caja']=1` mientras el plan acumulaba
  `units['g']=500` → el item aparecía DUPLICADO en el delta (uno por peso,
  otro por contenedor) y el usuario compraba dos veces lo mismo.

Fix:
  1. `_CONTAINER_UNIT_ALIASES` (frozenset módulo) cubre TODOS los envases
     del mercado dominicano: paquete, pote, lata, cartón, envase, botella,
     funda, caja, tetra, galón, jarra, bolsa, sobre, etc.
  2. `_fallback_container_weight_g(category)` retorna un peso default
     conservador por categoría cuando master no tiene el dato curado.
  3. El bloque normalizador usa el fallback automáticamente — la unidad
     SIEMPRE se convierte si el alias matchea.

Cobertura:
  - test_container_aliases_set_includes_caja_tetra_galon
  - test_container_aliases_includes_legacy_paquete_pote_lata
  - test_fallback_returns_dairy_weight_for_lacteos
  - test_fallback_returns_default_for_unknown_category
  - test_fallback_handles_none_and_empty_category
  - test_aggregator_normalizes_caja_to_grams_via_fallback
  - test_aggregator_normalizes_tetra_via_alias
  - test_aggregator_no_phantom_duplicate_when_inventory_uses_caja
"""
import inspect

import pytest

import shopping_calculator
from shopping_calculator import (
    _CONTAINER_UNIT_ALIASES,
    _fallback_container_weight_g,
    aggregate_and_deduct_shopping_list,
)


# ---------------------------------------------------------------------------
# 1. Set de aliases — cobertura completa.
# ---------------------------------------------------------------------------
def test_container_aliases_set_includes_caja_tetra_galon():
    """Aliases NUEVOS añadidos por P1-3 que antes faltaban."""
    new_aliases = {'caja', 'cajas', 'tetra', 'tetrapak', 'galón', 'galon', 'galones', 'jarra', 'jarras', 'bolsa', 'bolsas'}
    missing = new_aliases - _CONTAINER_UNIT_ALIASES
    assert not missing, f"P1-3: faltan aliases nuevos: {missing}"


def test_container_aliases_includes_legacy_paquete_pote_lata():
    """Aliases LEGACY ya existentes deben preservarse para no romper
    inventarios persistidos antes del fix."""
    legacy = {'paquete', 'paquetes', 'pote', 'potes', 'lata', 'latas', 'cartón', 'carton', 'cartones', 'envase', 'envases', 'botella', 'botellas', 'funda', 'fundas'}
    missing = legacy - _CONTAINER_UNIT_ALIASES
    assert not missing, f"P1-3 regression: aliases legacy borrados: {missing}"


def test_container_aliases_is_frozen_for_safety():
    """`_CONTAINER_UNIT_ALIASES` debe ser frozenset (inmutable) para que
    nadie lo mute accidentalmente en runtime."""
    assert isinstance(_CONTAINER_UNIT_ALIASES, frozenset)


# ---------------------------------------------------------------------------
# 2. Fallback por categoría.
# ---------------------------------------------------------------------------
def test_fallback_returns_dairy_weight_for_lacteos():
    """Lácteos → ~1L (cartón leche típico)."""
    assert _fallback_container_weight_g("Lácteos") == 1000.0
    # Variantes case + sin tilde.
    assert _fallback_container_weight_g("lácteos") == 1000.0
    assert _fallback_container_weight_g("lacteos") == 1000.0


def test_fallback_returns_grain_weight_for_despensa():
    """Despensa/granos → ~450g (paquete arroz/pasta)."""
    assert _fallback_container_weight_g("Despensa") == 450.0
    assert _fallback_container_weight_g("Despensa y Granos") == 450.0


def test_fallback_returns_oil_weight_for_aceites():
    assert _fallback_container_weight_g("Aceites") == 950.0


def test_fallback_returns_default_for_unknown_category():
    """Categoría no listada → default genérico (500g)."""
    assert _fallback_container_weight_g("Categoria Inventada") == 500.0


def test_fallback_handles_none_and_empty_category():
    """Defensa: None/'' no debe lanzar."""
    assert _fallback_container_weight_g(None) == 500.0
    assert _fallback_container_weight_g("") == 500.0


# ---------------------------------------------------------------------------
# 3. Aggregator: la unidad híbrida se normaliza a gramos.
# ---------------------------------------------------------------------------
def test_aggregator_normalizes_caja_to_grams_via_alias():
    """Un item del inventario "1 caja de leche" debe ser comparable contra
    el plan en gramos (ej. 500g leche). Antes del fix, "caja" no estaba en
    el set de aliases y el inventario quedaba sin convertir → duplicación."""
    plan = ["500 g de leche"]
    consumed = ["1 caja de leche"]  # alias NUEVO añadido por P1-3
    result = aggregate_and_deduct_shopping_list(plan, consumed, structured=True)
    leche_items = [r for r in result if isinstance(r, dict) and "leche" in r.get("name", "").lower()]
    # 500g plan vs 1000g caja (fallback Lácteos) → consumed cubre todo el peso.
    # El bloque P0-11 (clamp) suprime el item.
    assert leche_items == [], (
        f"P1-3: caja de leche debe normalizar a gramos vía fallback. "
        f"Leche items: {leche_items}"
    )


def test_aggregator_normalizes_tetra_via_alias():
    """`tetra` (envase tipo tetrapak) debe ser reconocido."""
    plan = ["500 g de jugo"]
    consumed = ["1 tetra de jugo"]
    result = aggregate_and_deduct_shopping_list(plan, consumed, structured=True)
    jugo_items = [r for r in result if isinstance(r, dict) and "jugo" in r.get("name", "").lower()]
    assert jugo_items == [], "P1-3: tetra debe normalizar igual que paquete"


def test_aggregator_normalizes_galon_via_alias():
    """`galón` para líquidos (agua, leche industrial) debe normalizarse."""
    plan = ["500 g de leche"]
    consumed = ["1 galón de leche"]
    result = aggregate_and_deduct_shopping_list(plan, consumed, structured=True)
    leche_items = [r for r in result if isinstance(r, dict) and "leche" in r.get("name", "").lower()]
    assert leche_items == [], "P1-3: galón debe normalizar igual que cartón"


def test_aggregator_does_not_phantom_duplicate_for_known_alias():
    """Caso del bug: plan en gramos + inventario en alias híbrido → un item
    en la lista (no dos)."""
    plan = ["1000 g de arroz"]
    consumed = ["1 paquete de arroz"]  # legacy alias, fallback 450g
    result = aggregate_and_deduct_shopping_list(plan, consumed, structured=True)
    arroz_items = [r for r in result if isinstance(r, dict) and "arroz" in r.get("name", "").lower()]
    # Plan 1000g - paquete 450g = 550g restantes a comprar. Debe haber UN item.
    assert len(arroz_items) <= 1, (
        f"P1-3: NO debe haber items duplicados (peso + paquete). Got: {arroz_items}"
    )


# ---------------------------------------------------------------------------
# 4. Defensa estructural: el código fuente usa la constante y el fallback.
# ---------------------------------------------------------------------------
def test_aggregator_uses_constant_set_not_inline_list():
    """El bloque normalizador debe referenciar `_CONTAINER_UNIT_ALIASES`,
    no una lista inline. Defensa contra reintroducir el patrón roto."""
    src = inspect.getsource(aggregate_and_deduct_shopping_list)
    assert "_CONTAINER_UNIT_ALIASES" in src, (
        "P1-3: el aggregator debe usar `_CONTAINER_UNIT_ALIASES` (constante)"
    )
    # Debe invocar el fallback helper.
    assert "_fallback_container_weight_g" in src, (
        "P1-3: falta uso del fallback `_fallback_container_weight_g`"
    )


def test_documentation_p1_3_present():
    """Comentario `[P1-3]` documenta el rationale para futuros maintainers."""
    src = inspect.getsource(shopping_calculator)
    assert "[P1-3]" in src
