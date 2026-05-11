"""[P2-NEW-1 · 2026-05-10] Test del helper `canonicalize_protein` y su
integración en `_canonicalize_for_coherence`.

Bug original (audit 2026-05-10):
  Solo `canonicalize_pavo` existía. Pollo/cerdo/res caían al fallback
  master_map + heurística general. Si el LLM producía "pechuga de pollo
  desmenuzada" en una receta y la lista contenía "Pollo" (canónico),
  el guard veía dos foods distintos y reportaba magnitudes inválidas
  como divergencia.

Fix (P2-NEW-1):
  Helper `canonicalize_protein(name)` colapsa cualquier mención de
  pollo/cerdo/res con sus cortes y cooking-states a sus canónicos
  base ('Pollo', 'Cerdo', 'Res'). Integrado en
  `_canonicalize_for_coherence` después del check de pavo.

Diferencias vs. canonicalize_pavo:
  - No hay distinción fresh-vs-procesado (no aplica comercialmente).
  - Excluye productos derivados con canónico industrial propio:
    salchichas, longanizas, salamis, jamón, tocineta, chorizo,
    nuggets, etc. — el master_map los canonicaliza por separado.
"""
import pytest

import shopping_calculator
from shopping_calculator import (
    canonicalize_protein,
    _canonicalize_for_coherence,
)


@pytest.fixture(autouse=True)
def no_master_db(monkeypatch):
    """Stub `get_master_ingredients` a `[]` (mismo patrón que
    test_p1_shopping_recipe_coherence.py:33). Sin esto, los tests
    del guard intentan cargar el master_map desde DB."""
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda: [])


# ---------------------------------------------------------------------------
# 1. canonicalize_protein — happy path por proteína
# ---------------------------------------------------------------------------
class TestCanonicalizeProteinPositive:
    """Casos que SÍ deben canonicalizar."""

    @pytest.mark.parametrize("raw,expected", [
        # Pollo: cortes + cooking-states + fresh markers
        ("pollo", "Pollo"),
        ("Pollo", "Pollo"),
        ("pechuga de pollo", "Pollo"),
        ("pechuga de pollo fresca", "Pollo"),
        ("muslo de pollo desmenuzado", "Pollo"),
        ("filete de pollo cocido", "Pollo"),
        ("pollo asado", "Pollo"),
        ("pollo guisado", "Pollo"),
        ("pollo molido", "Pollo"),
        ("pierna de pollo horneada", "Pollo"),
        ("pollo orgánico", "Pollo"),
        # Cerdo
        ("cerdo", "Cerdo"),
        ("Cerdo", "Cerdo"),
        ("chuleta de cerdo", "Cerdo"),
        ("chuleta de cerdo guisada", "Cerdo"),
        ("lomo de cerdo asado", "Cerdo"),
        ("costilla de cerdo", "Cerdo"),
        ("cerdo desmenuzado", "Cerdo"),
        # Res / Carne de res
        ("res", "Res"),
        ("Res", "Res"),
        ("carne de res", "Res"),
        ("carne de res molida", "Res"),
        ("filete de res", "Res"),
        ("lomo de res asado", "Res"),
    ])
    def test_canonicalizes_to_base(self, raw, expected):
        assert canonicalize_protein(raw) == expected


# ---------------------------------------------------------------------------
# 2. canonicalize_protein — negative cases (devuelve None)
# ---------------------------------------------------------------------------
class TestCanonicalizeProteinNegative:
    """Casos que NO deben canonicalizar (devolver None)."""

    @pytest.mark.parametrize("raw", [
        # No menciona ninguna proteína target
        "tomate",
        "arroz",
        "manzanas",
        # Pavo (su propio helper)
        "pavo",
        "pechuga de pavo",
        # Productos derivados con canónico industrial propio — fuera
        # del dominio de este helper (master_map debe resolverlos).
        "salchicha de pollo",
        "longaniza de cerdo",
        "salami de res",
        "jamón de pollo",
        "tocineta de cerdo",
        "chorizo de cerdo",
        "nuggets de pollo",
        "pollo enlatado",
        # Multi-protein (no claro cuál gana)
        "guiso de pollo y cerdo",
        # Caldo (composición, no proteína directa)
        "caldo de pollo",
        "caldo de res",
        # Inputs degenerados
        "",
        None,
        "   ",
    ])
    def test_returns_none(self, raw):
        assert canonicalize_protein(raw) is None


# ---------------------------------------------------------------------------
# 3. Integración con _canonicalize_for_coherence
# ---------------------------------------------------------------------------
class TestCanonicalizeForCoherenceIntegration:
    """Verifica que `_canonicalize_for_coherence` aplica el mirror simétrico
    para pollo/cerdo/res. Los inputs simulan los lados receta y lista del
    coherence guard."""

    def test_pollo_recipe_vs_canonical_list_collapse_to_same(self, no_master_db):
        """Receta: 'pechuga de pollo fresca'. Lista: 'Pollo'.
        Deben canonicalizar al mismo 'Pollo' → guard no reporta presence."""
        recipe_side = _canonicalize_for_coherence({"pechuga de pollo fresca"})
        list_side = _canonicalize_for_coherence({"Pollo"})
        # Intersection no vacía: ambos producen 'Pollo'.
        assert "Pollo" in recipe_side
        assert "Pollo" in list_side
        assert recipe_side & list_side, (
            f"Recipe canonical={recipe_side!r}, list canonical={list_side!r} — "
            f"no comparten 'Pollo'; el guard reportaría falso positivo de presence."
        )

    def test_cerdo_chuleta_vs_canonical(self, no_master_db):
        recipe_side = _canonicalize_for_coherence({"chuleta de cerdo guisada"})
        list_side = _canonicalize_for_coherence({"Cerdo"})
        assert recipe_side & list_side, (
            f"Recipe={recipe_side!r}, list={list_side!r} — no comparten 'Cerdo'."
        )

    def test_res_carne_molida_vs_canonical(self, no_master_db):
        recipe_side = _canonicalize_for_coherence({"carne de res molida"})
        list_side = _canonicalize_for_coherence({"Res"})
        assert recipe_side & list_side, (
            f"Recipe={recipe_side!r}, list={list_side!r} — no comparten 'Res'."
        )

    def test_pavo_still_works_after_protein_added(self, no_master_db):
        """Regression: el path de pavo sigue activo, no fue desplazado por
        el nuevo path de proteína. canonicalize_pavo se chequea ANTES."""
        recipe_side = _canonicalize_for_coherence({"pechuga de pavo fresca"})
        list_side = _canonicalize_for_coherence({"Pechuga de pavo"})
        # Pavo siempre va a 'Pechuga de pavo' (fresh default), NO a 'Pavo' base.
        assert "Pechuga de pavo" in recipe_side
        assert "Pechuga de pavo" in list_side

    def test_non_protein_food_unaffected(self, no_master_db):
        """Foods sin pollo/cerdo/res/pavo no son tocados por el nuevo path."""
        result = _canonicalize_for_coherence({"tomate", "arroz", "Manzana"})
        # No asertamos forma exacta (depende de master_map + heurística), solo
        # que no se colapsaron a un canónico de proteína.
        assert not (result & {"Pollo", "Cerdo", "Res"}), (
            f"Result {result!r} se contaminó con canónicos de proteína."
        )

    def test_jamon_de_pollo_not_collapsed(self, no_master_db):
        """`jamón de pollo` es producto deli — NO debe colapsar a 'Pollo'.
        Master_map o aggregator lo canonicalizan independiente."""
        result = _canonicalize_for_coherence({"jamón de pollo"})
        assert "Pollo" not in result, (
            f"`jamón de pollo` se colapsó a 'Pollo' — pero es deli, no equivale "
            f"comercialmente. Result: {result!r}."
        )
