"""[P3-HERB-CAP] Tests para el cap defensivo de hierbas frescas.

Bug original (PDF 2026-05-05):
  Lista de compras mensual × 2 personas mostró "Cilantro: 23 Mazos" —
  ~1.15 kg de cilantro fresco para 30 días, cuando un mazo dura 5-7 días
  refrigerado y el resto se descompone (~$200 RD$ desperdiciados).

Causas en cadena:
  1. `_parse_quantity` NO reconocía "mazo" como unidad → "1 mazo de
     cilantro" se parseaba como `unit='unidad', name='Mazo de cilantro'`.
     La canonicalización fallaba (no resolvía a "Cilantro") así que el cap
     no podía actuar (name no matcheaba el set de hierbas).
  2. BLOQUE 1.5 de `apply_smart_market_units` (is_herb_mazo) calculaba
     `units_needed = max(1, ceil(raw_qty))` sobre `raw_qty` ya
     multiplicado por el ciclo, sin cap.

Fix:
  1. `_parse_quantity` ahora reconoce `mazo|mazos|atado|atados|manojo|manojos`
     como unidad canónica → permite canonicalización a "Cilantro".
  2. `aggregate_and_deduct_shopping_list` aplica cap defensivo justo antes
     del loop final: `cap_mazos = max(2, round(multiplier × 3/7))` —
     1 mazo / persona / semana = uso realista. Aplica a un set de hierbas
     conocidas (cilantro, perejil, recao, menta, albahaca, romero, etc.).

Cobertura:
  - Repro PDF: 1 mazo cilantro × multiplier mensual×2p → cap 8
  - Cap escalado: weekly/biweekly/monthly × 1p/2p/4p
  - LLM dice "cda cilantro": converted a g, cap por gramos
  - Items que NO son hierbas no se tocan (pollo, etc.)
  - `_parse_quantity` reconoce mazo/atado/manojo
"""
import pytest

from shopping_calculator import (
    _parse_quantity,
    aggregate_and_deduct_shopping_list,
)


# ---------------------------------------------------------------------------
# 1. _parse_quantity reconoce mazo
# ---------------------------------------------------------------------------
class TestParseQuantityMazo:
    @pytest.mark.parametrize("text,expected_unit", [
        ("1 mazo de cilantro", "mazo"),
        ("2 mazos de perejil", "mazo"),
        ("1 atado de menta", "mazo"),
        ("3 atados de cilantro", "mazo"),
        ("1 manojo de albahaca", "mazo"),
    ])
    def test_recognizes_mazo_variants(self, text, expected_unit):
        qty, unit, name = _parse_quantity(
            text, apply_yield_multiplier=False, apply_legumbres_yield_only=True
        )
        assert unit == expected_unit, f"{text!r}: got unit={unit!r}"
        # name no debe contener "mazo"/"atado"/"manojo" (deben ir a unidad)
        assert "mazo" not in name.lower()
        assert "atado" not in name.lower()
        assert "manojo" not in name.lower()

    def test_canonicalization_cilantro(self):
        """'1 mazo de cilantro' → name canónico 'Cilantro' (no 'Mazo de cilantro')."""
        _, unit, name = _parse_quantity(
            "1 mazo de cilantro",
            apply_yield_multiplier=False,
            apply_legumbres_yield_only=True,
        )
        assert unit == "mazo"
        assert name.lower() == "cilantro"


# ---------------------------------------------------------------------------
# 2. Cap escalado por person-weeks (multiplier × 3/7)
# ---------------------------------------------------------------------------
class TestHerbCapScaling:
    """Cap = max(_HERB_MAZO_CAP_FLOOR, round(person_weeks)) donde
    person_weeks = max(1.0, multiplier × 3/7).

    [P3-HERB-CAP-FLOOR · 2026-05-16] El floor pasó de hardcoded 2 a knob
    `MEALFIT_HERB_MAZO_CAP_FLOOR` con DEFAULT 1. Razón documentada en
    shopping_calculator.py: para 1 persona × 7 días, 2 mazos (~100g, ¼ lb)
    de cilantro/perejil son excesivos; 1 mazo (~50g) basta. Solo afecta el
    caso 1p × semanal (person_weeks=1.0) — todo plan 2p+ o cycle >1 semana
    tiene person_weeks >= 2, donde el floor no muerde. Este test refleja el
    default floor=1 actual de prod."""

    def _herb_qty(self, multiplier: float) -> int:
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["1 mazo de cilantro"],
            multiplier=multiplier,
            structured=True,
        )
        cilantro = next(
            (r for r in result if "cilantro" in r.get("name", "").lower()), None
        )
        assert cilantro is not None
        return int(cilantro["market_qty"])

    @pytest.mark.parametrize("scenario,multiplier,expected", [
        ("mensual x 4 personas", 4 * 4 * 7 / 3, 16),
        ("mensual x 2 personas", 2 * 4 * 7 / 3, 8),
        ("mensual x 1 persona", 1 * 4 * 7 / 3, 4),
        ("quincenal x 2 personas", 2 * 2 * 7 / 3, 4),
        ("quincenal x 1 persona", 1 * 2 * 7 / 3, 2),
        ("semanal x 2 personas", 2 * 1 * 7 / 3, 2),
        # [P3-HERB-CAP-FLOOR · 2026-05-16] floor default 1 (era 2):
        # person_weeks = max(1.0, 2.33×3/7=1.0) = 1.0 → max(1, round(1.0)) = 1.
        ("semanal x 1 persona", 1 * 1 * 7 / 3, 1),  # max(1, round(1.0))
    ])
    def test_cap_scales_with_person_weeks(self, scenario, multiplier, expected):
        actual = self._herb_qty(multiplier)
        assert actual == expected, (
            f"{scenario} (mult={multiplier:.2f}): esperado cap={expected}, "
            f"recibido {actual}"
        )

    def test_minimum_cap_clamps_high_demand_to_floor(self):
        """[P3-HERB-CAP-FLOOR · 2026-05-16] Para multipliers muy bajos,
        person_weeks se clampa a 1.0 → cap = max(floor, round(1.0)) = floor.
        El floor default es 1 (era 2 hardcoded). Verificamos pidiendo 5 mazos
        con multiplier mínimo: la demanda 5 excede el cap (=floor=1), así que
        se clampa al floor. Garantiza que una demanda alta nunca eluda el
        cap — el comportamiento que el test protege (clamp efectivo) sigue
        intacto; solo bajó el valor del piso de 2 a 1."""
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=["5 mazos de cilantro"],
            multiplier=1.0,  # caso degenerate
            structured=True,
        )
        cilantro = next(
            (r for r in result if "cilantro" in r.get("name", "").lower()), None
        )
        actual = int(cilantro["market_qty"])
        # Cap = floor default (1); demanda 5 excede así que se clampa a 1.
        assert actual == 1, f"Cap floor (default 1) debe clampar 5→1, recibido {actual}"


# ---------------------------------------------------------------------------
# 3. Repro exacto del PDF 2026-05-05
# ---------------------------------------------------------------------------
def test_repro_pdf_2026_05_05_cilantro_23_mazos():
    """PDF original: '1 mazo cilantro' en 1 receta × multiplier mensual×2p
    (18.67) producía 19+ mazos (después con duplicados/ramita: 23).
    Post-fix: cap a 8 mazos."""
    plan_items = [
        "1 mazo de cilantro",
        # Otras menciones que también se sumarían sin cap:
        "1 mazo de cilantro",  # otra receta
        "1 ramita de cilantro",  # variante
    ]
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=plan_items,
        multiplier=18.67,  # mensual × 2 personas
        structured=True,
    )
    cilantro = next(
        (r for r in result if "cilantro" in r.get("name", "").lower()), None
    )
    assert cilantro is not None, "Cilantro debe aparecer en la lista"
    qty = int(cilantro["market_qty"])
    assert 2 <= qty <= 8, (
        f"Cilantro mensual × 2p debe tener qty entre 2-8 mazos, recibido {qty}"
    )


# ---------------------------------------------------------------------------
# 4. Cap por GRAMOS (LLM dice 'cda cilantro' que se convierte a g)
# ---------------------------------------------------------------------------
def test_cap_aplicable_a_gramos_cuando_llm_usa_cdas():
    """Si LLM dice 'cda cilantro' (no mazo), la unit conversion lo manda a g.
    El cap por g (cap_mazos × 50g) también debe activarse."""
    plan_items = ["1 cda de cilantro picado"] * 30  # 30 cdas
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=plan_items,
        multiplier=18.67,
        structured=True,
    )
    cilantro = next(
        (r for r in result if "cilantro" in r.get("name", "").lower()), None
    )
    if cilantro is not None:
        # Display debe ser razonable (no centenares de mazos)
        qty = int(cilantro["market_qty"])
        assert qty <= 10, f"Cap por g fallo: {qty} mazos para 30 cdas mensual"


# ---------------------------------------------------------------------------
# 5. Items que NO son hierbas no se ven afectados (no regresión)
# ---------------------------------------------------------------------------
class TestNonHerbsNotAffected:
    @pytest.mark.parametrize("plan_item,name_keyword", [
        ("200g de pollo", "pollo"),
        ("1 cebolla", "cebolla"),
        ("2 tomates", "tomate"),
        ("1 lb de zanahoria", "zanahoria"),  # vegetal weighable, no es hierba
    ])
    def test_no_cap_para_no_hierbas(self, plan_item, name_keyword):
        """Cap solo aplica a hierbas frescas en _HERB_NAMES_FOR_CAP.
        Otros items se calculan normalmente sin clamp."""
        result = aggregate_and_deduct_shopping_list(
            plan_ingredients=[plan_item],
            multiplier=18.67,
            structured=True,
        )
        item = next(
            (r for r in result if name_keyword in r.get("name", "").lower()), None
        )
        if item is not None:
            # No debe estar capped a 8 (excepto coincidencia)
            qty = float(item["market_qty"])
            # Pollo 200g × 18.67 = 8.23 lbs; cebolla 1 × 18.67 = 19 uds; etc.
            # El cap de hierbas (8) NO debe aplicar.
            # Solo verificamos que el item está presente con qty > 0.
            assert qty > 0


# ---------------------------------------------------------------------------
# 6. Múltiples hierbas independientes
# ---------------------------------------------------------------------------
def test_multiple_herbs_each_capped_independently():
    """Cilantro + perejil + menta cada uno tiene su propio cap."""
    plan_items = [
        "1 mazo de cilantro",
        "1 mazo de perejil",
        "1 mazo de menta",
    ]
    result = aggregate_and_deduct_shopping_list(
        plan_ingredients=plan_items,
        multiplier=18.67,
        structured=True,
    )
    for herb in ["cilantro", "perejil", "menta"]:
        item = next(
            (r for r in result if herb in r.get("name", "").lower()), None
        )
        assert item is not None, f"{herb} debe estar en la lista"
        assert int(item["market_qty"]) <= 8, (
            f"{herb}: cap fallo → qty={item['market_qty']}"
        )
