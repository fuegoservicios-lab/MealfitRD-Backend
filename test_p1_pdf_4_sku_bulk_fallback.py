"""[P1-PDF-4] Tests para `_find_best_sku` cuando la necesidad excede al
SKU comercial más grande disponible.

Bug original (reproducido en producción 2026-05-04):
  Plan mensual × 2 personas, yogurt griego con
  `available_sizes_g=[150, 227, 453]`. La necesidad agregada era
  `g_total = 8.23 lbs ≈ 3733g`. En `_find_best_sku`:
    - Estrategia 1 (single SKU ≤ 2× con waste ≤ 5%): ningún size ≥ 3733 → skip.
    - Estrategia 2 (best-fit): el guard `if size < g_total * 0.15: continue`
      descartaba TODOS los sizes (3733 × 0.15 = 560 > 453).
    - `best_result` quedaba en `None`.
    - Fallback: `return (1, sizes[0])` → `(1, 150)` → PDF mostraba
      "1 pote (150g)" cuando el usuario realmente necesitaba ~25 potes
      (≈ 3.7 kg). Under-buy del 96%, compra duplicada cuando el usuario
      detectara el faltante a mitad de mes.

  Mismo modo de fallo aplicaba a TODO ingrediente cuyo SKU max < 6.67× la
  necesidad real (habichuelas, queso blanco, mozzarella en planes largos).

Fix (Estrategia 3 fallback):
  Si la estrategia 2 deja `best_result=None` (todos los sizes filtrados),
  caemos al SKU MÁS GRANDE con `ceil(g_total / size)` count. Matemática
  correcta, ningún under-buy silencioso.
"""
import math

import pytest

from shopping_calculator import _find_best_sku, apply_smart_market_units


# ---------------------------------------------------------------------------
# 1. Bug repro directo: yogurt mensual × 2 personas
# ---------------------------------------------------------------------------
def test_yogurt_monthly_2_personas_no_under_buy():
    """Repro del bug observable. Antes: (1, 150). Después: count que cubre."""
    g_total = 8.23 * 453.592  # 3733g — valor real del log de producción
    sizes = [150, 227, 453]   # SKUs reales de master_ingredients

    count, size = _find_best_sku(g_total, sizes, 0.02)

    # Pre-fix: count=1, size=150 → comprado=150g (cubre 4% de la necesidad)
    # Post-fix: usa el size más grande (453) con ceil → cubre ≥ 100%
    total_bought_g = count * size
    coverage = total_bought_g / g_total

    assert coverage >= 1.0, (
        f"Under-buy detectado: {count} × {size}g = {total_bought_g}g "
        f"vs necesidad {g_total:.0f}g (coverage={coverage:.1%}). "
        "El fix debe garantizar coverage >= 100%."
    )
    assert size == 453, (
        f"Esperado fallback al SKU más grande (453g), recibido {size}g. "
        "El fix debe usar `sizes[-1]` cuando todos los sizes quedan filtrados."
    )
    assert count == math.ceil(g_total / 453), (
        f"Conteo esperado ceil(3733/453)=9, recibido {count}."
    )


def test_habichuelas_rojas_monthly_2_personas_no_under_buy():
    """Mismo patrón con [340, 453]: ambos < 3733*0.15=560 → fallback bug."""
    g_total = 8.23 * 453.592  # 3733g
    sizes = [340, 453]

    count, size = _find_best_sku(g_total, sizes, 0.02)
    coverage = (count * size) / g_total

    assert coverage >= 1.0, f"Under-buy: {count}×{size}g={count*size}g vs {g_total:.0f}g"
    assert size == 453, f"Esperado 453 (max), recibido {size}"


def test_queso_blanco_no_under_buy_above_threshold():
    """Para que el bug se dispare con [227, 340, 453], g_total > 453/0.15 = 3020g.
    Plan grande con queso múltiple → simulamos 7 lbs = 3175g.
    """
    g_total = 7.0 * 453.592  # 3175g
    sizes = [227, 340, 453]

    count, size = _find_best_sku(g_total, sizes, 0.02)
    coverage = (count * size) / g_total

    assert coverage >= 1.0
    assert size == 453, f"Esperado SKU grande, recibido {size}g"


# ---------------------------------------------------------------------------
# 2. Casos NO afectados por el fix (Strategy 1 y 2 deben seguir intactos)
# ---------------------------------------------------------------------------
def test_strategy_1_single_pkg_within_tolerance():
    """Si un single SKU cubre con ≤ 5% waste, debe ganar — sin tocar fallback."""
    g_total = 0.99 * 453.592  # ≈ 449g, cabe en 1 paquete de 453g (waste 0.9%)
    sizes = [150, 227, 453]

    count, size = _find_best_sku(g_total, sizes, 0.02)

    assert (count, size) == (1, 453), (
        f"Strategy 1 debió retornar (1, 453), recibido ({count}, {size})"
    )


def test_strategy_2_best_fit_multi_pkg():
    """Necesidad mediana donde Strategy 2 elige el size con mejor score."""
    g_total = 1.5 * 453.592  # 680g
    sizes = [150, 227, 453]

    count, size = _find_best_sku(g_total, sizes, 0.02)
    coverage = (count * size) / g_total

    # Strategy 2 debe elegir 3×227g=681g (waste mínimo, count razonable)
    assert (count, size) == (3, 227), (
        f"Strategy 2 debió retornar (3, 227), recibido ({count}, {size})"
    )
    assert coverage >= 1.0


# ---------------------------------------------------------------------------
# 3. Edge cases del fix
# ---------------------------------------------------------------------------
def test_fix_uses_largest_not_smallest():
    """Validación explícita del bug clásico: NUNCA retornar (1, smallest)
    cuando todos los sizes son << g_total."""
    g_total = 10000.0  # 22 lbs — necesidad masiva
    sizes = [50, 100, 200]

    count, size = _find_best_sku(g_total, sizes, 0.02)

    assert size == 200, (
        f"Fallback debe usar size MAX (200), nunca el smallest. Recibido {size}."
    )
    # 10000/200 = 50 paquetes — alto pero matemáticamente correcto
    assert count >= math.ceil(10000 / 200)


def test_fix_preserves_coverage_at_boundary():
    """g_total justo en el límite donde el guard 0.15 empieza a filtrar."""
    # Para sizes=[453], el guard skipea cuando size < g_total*0.15
    # → g_total > 453/0.15 = 3020g
    g_total = 3050.0  # justo arriba del threshold
    sizes = [150, 227, 453]

    count, size = _find_best_sku(g_total, sizes, 0.02)
    coverage = (count * size) / g_total

    assert coverage >= 1.0
    assert size == 453


# ---------------------------------------------------------------------------
# 4. Test end-to-end vía apply_smart_market_units (forma del display_qty)
# ---------------------------------------------------------------------------
def test_apply_smart_market_units_display_qty_for_yogurt_monthly():
    """Verifica el contrato externo: el display_qty del PDF nunca debe decir
    '1 pote (150g)' cuando weight_in_lbs es 8+ lbs."""
    master_item = {
        "market_container": "pote",
        "container_weight_g": 453,
        "available_sizes_g": [150, 227, 453],
        "category": "Lácteos",
        "shelf_life_days": 14,
    }
    result = apply_smart_market_units(
        "Yogurt griego sin azúcar", 8.23, "lb", 0.0, master_item
    )

    display = result["display_qty"]
    market_qty = result["market_qty"]

    # El bug específico: nunca debe mostrar "1 pote (150g)" para 8 lbs
    assert "150g" not in display, (
        f"Regresión P1-PDF-4: display_qty={display!r} contiene '150g' para 8.23 lbs. "
        "El fallback de _find_best_sku no debe seleccionar el SKU más pequeño."
    )
    assert market_qty >= 8, (
        f"market_qty={market_qty} es demasiado bajo para 8.23 lbs (~3733g). "
        "Esperado >= 8 potes (de 453g) o equivalente."
    )
