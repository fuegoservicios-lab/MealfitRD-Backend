"""[P2-PDF-1] Tests para conversión cocido→seco de legumbres/granos en
el shopping aggregator.

Bug original (observado tras P1-PDF-4 + P1-PDF-5):
  Plan mensual × 2 personas con habichuelas rojas. `ingredients_raw`:
    - "150g de habichuelas rojas cocidas" (martes almuerzo)
    - "200g de habichuelas rojas cocidas" (martes cena)
  Total cocidas/ciclo: 350g. Multiplicador 18.67 (2 personas × 4 ciclos
  × 7/3 días) → 6534g cocidas ≈ 14.4 lbs.

  El aggregator pasaba `apply_yield_multiplier=False` (P1-2: simetría
  plan↔inventario para proteínas) sin distinguir legumbres/granos. El
  SKU comercial de habichuelas/lentejas/arroz/pasta es SECO; sin
  conversión cocido→seco (factor 0.35×), el aggregator computaba 14.4
  lbs en peso seco → producía "15 paquetes de 1 lb c/u" cuando el usuario
  realmente necesita ~5 lbs secas. Over-buy de 3×.

Fix:
  Nuevo flag `apply_legumbres_yield_only` en `_parse_quantity` (default
  False — preserva todos los call-sites históricos). Cuando True, activa
  SOLO la regla #1 de `_calculate_yield_multiplier` (legumbres/granos
  cocidos → 0.35×) sin reabrir las reglas #2-4 que P1-2 cerró por la
  asimetría plan↔inventario en proteínas.

  Regex corregida también: `\bhabichuela\b` no matcheaba "habichuelas"
  porque `s` es word char. Ahora `habichuelas?`, `lentejas?`, `pastas?`,
  `quinoas?` cubren plural simple, y `frijol(?:es)?`, `guandul(?:es)?`
  cubren plural en `-es`.

Cobertura:
  - Helper `_calculate_yield_multiplier(only_legumbres_grains=True)`
  - `_parse_quantity` con el nuevo flag (incluyendo plurales)
  - End-to-end: g_total real del plan reproduce conteo razonable post-fix
  - Aggregator no rompe simetría plan↔consumed
  - Items que NO son legumbres no se tocan (proteínas, lácteos)
"""
import math

import pytest

from shopping_calculator import (
    _calculate_yield_multiplier,
    _parse_quantity,
)


# ---------------------------------------------------------------------------
# 1. Yield multiplier en modo only_legumbres_grains
# ---------------------------------------------------------------------------
class TestYieldMultiplierLegumbresOnly:
    def test_habichuelas_singular(self):
        assert _calculate_yield_multiplier("habichuela cocida", only_legumbres_grains=True) == 0.35

    def test_habichuelas_plural(self):
        """Bug original: \\bhabichuela\\b no matcheaba 'habichuelas'."""
        assert _calculate_yield_multiplier("habichuelas rojas cocidas", only_legumbres_grains=True) == 0.35
        assert _calculate_yield_multiplier("habichuelas negras hervidas", only_legumbres_grains=True) == 0.35

    def test_lentejas_plural(self):
        assert _calculate_yield_multiplier("lentejas cocidas", only_legumbres_grains=True) == 0.35

    def test_arroz_singular(self):
        assert _calculate_yield_multiplier("arroz integral cocido", only_legumbres_grains=True) == 0.35
        assert _calculate_yield_multiplier("arroz blanco hervido", only_legumbres_grains=True) == 0.35

    def test_pasta_singular_y_plural(self):
        assert _calculate_yield_multiplier("pasta cocida", only_legumbres_grains=True) == 0.35
        assert _calculate_yield_multiplier("pastas hervidas", only_legumbres_grains=True) == 0.35

    def test_quinoa_singular_y_plural(self):
        assert _calculate_yield_multiplier("quinoa hervida", only_legumbres_grains=True) == 0.35
        assert _calculate_yield_multiplier("quinoas cocidas", only_legumbres_grains=True) == 0.35

    def test_frijol_singular_y_plural(self):
        """frijol/frijoles — plural en -es, no en -s."""
        assert _calculate_yield_multiplier("frijol negro cocido", only_legumbres_grains=True) == 0.35
        assert _calculate_yield_multiplier("frijoles negros cocidos", only_legumbres_grains=True) == 0.35

    def test_guandul_singular_y_plural(self):
        """guandul/guandules — plural en -es, no en -s."""
        assert _calculate_yield_multiplier("guandul cocido", only_legumbres_grains=True) == 0.35
        assert _calculate_yield_multiplier("guandules cocidos", only_legumbres_grains=True) == 0.35

    def test_no_aplicar_proteinas(self):
        """Bajo only_legumbres_grains la regla #2 (proteínas) NO debe activarse.
        Esto preserva la simetría plan↔inventario de P1-2."""
        assert _calculate_yield_multiplier("pollo cocido", only_legumbres_grains=True) == 1.0
        assert _calculate_yield_multiplier("pavo asado", only_legumbres_grains=True) == 1.0
        assert _calculate_yield_multiplier("pescado horneado", only_legumbres_grains=True) == 1.0

    def test_no_aplicar_merma_pelado(self):
        """Regla #3 (víveres pelados, +30%) tampoco debe activarse."""
        assert _calculate_yield_multiplier("yuca pelada", only_legumbres_grains=True) == 1.0
        assert _calculate_yield_multiplier("plátano pelado", only_legumbres_grains=True) == 1.0

    def test_no_aplicar_si_no_dice_cocido(self):
        """Sin la palabra 'cocido/hervido' el yield siempre es 1.0."""
        assert _calculate_yield_multiplier("habichuelas rojas", only_legumbres_grains=True) == 1.0
        assert _calculate_yield_multiplier("arroz integral", only_legumbres_grains=True) == 1.0

    def test_no_match_palabras_inventadas(self):
        """Defensa contra over-matching: 'frijole'/'guandule' no son palabras."""
        assert _calculate_yield_multiplier("frijole inventado cocido", only_legumbres_grains=True) == 1.0
        assert _calculate_yield_multiplier("guandule inventado cocido", only_legumbres_grains=True) == 1.0


# ---------------------------------------------------------------------------
# 2. Backward-compat: legacy mode preserva TODAS las reglas
# ---------------------------------------------------------------------------
class TestLegacyYieldUnchanged:
    """Sin el flag, el comportamiento debe ser idéntico al pre-P2-PDF-1.
    Los call-sites históricos (tools.py, cron_tasks.py, db_inventory.py)
    dependen de las 4 reglas activas."""

    def test_legacy_proteinas_yield_135(self):
        assert _calculate_yield_multiplier("pollo cocido") == 1.35
        assert _calculate_yield_multiplier("pavo asado") == 1.35

    def test_legacy_merma_pelado_yield_130(self):
        assert _calculate_yield_multiplier("yuca pelada") == 1.30

    def test_legacy_hueso_yield_140(self):
        assert _calculate_yield_multiplier("pollo sin hueso") == 1.40


# ---------------------------------------------------------------------------
# 3. _parse_quantity con apply_legumbres_yield_only
# ---------------------------------------------------------------------------
class TestParseQuantityLegumbresFlag:
    def test_habichuelas_cocidas_aplica_yield(self):
        qty, unit, name = _parse_quantity(
            "200g de habichuelas rojas cocidas (con su caldo)",
            apply_yield_multiplier=False, apply_legumbres_yield_only=True,
        )
        assert qty == pytest.approx(200 * 0.35), (
            f"200g cocidas debe convertirse a 70g secas, recibido qty={qty}"
        )
        assert unit == "g"
        assert name == "Habichuelas rojas"

    def test_lentejas_tazas_cocidas_aplica_yield(self):
        qty, unit, name = _parse_quantity(
            "2 tazas de lentejas cocidas (400g)",
            apply_yield_multiplier=False, apply_legumbres_yield_only=True,
        )
        assert qty == pytest.approx(2 * 0.35)
        assert unit == "taza"
        assert name == "Lentejas"

    def test_arroz_cocido_aplica_yield(self):
        qty, unit, name = _parse_quantity(
            "1.25 tazas de arroz integral cocido (280g)",
            apply_yield_multiplier=False, apply_legumbres_yield_only=True,
        )
        assert qty == pytest.approx(1.25 * 0.35)

    def test_proteina_cocida_NO_aplica_yield(self):
        """Pollo cocido bajo el flag legumbres-only: sin conversión."""
        qty, unit, name = _parse_quantity(
            "210g de pechuga de pollo previamente cocida",
            apply_yield_multiplier=False, apply_legumbres_yield_only=True,
        )
        assert qty == 210.0, (
            f"Pollo cocido NO debe llevar yield bajo legumbres-only, recibido qty={qty}"
        )

    def test_queso_NO_aplica_yield(self):
        qty, _, _ = _parse_quantity(
            "75 g de queso blanco fresco",
            apply_yield_multiplier=False, apply_legumbres_yield_only=True,
        )
        assert qty == 75.0

    def test_legacy_path_sin_flags_preserva_proteinas(self):
        """Path legacy (apply_yield_multiplier=True) SIGUE aplicando todas las reglas."""
        qty, _, _ = _parse_quantity("210g de pollo cocido", apply_yield_multiplier=True)
        assert qty == pytest.approx(210 * 1.35)

    def test_path_aggregator_sin_flag_legumbres_no_aplica(self):
        """Sin `apply_legumbres_yield_only=True`, el aggregator pre-P2 dejaba
        habichuelas en peso cocido (bug original)."""
        qty, _, _ = _parse_quantity(
            "200g de habichuelas rojas cocidas",
            apply_yield_multiplier=False,
        )
        assert qty == 200.0, (
            "Sin el flag P2, qty debe quedar en peso literal (preserva backwards compat)"
        )


# ---------------------------------------------------------------------------
# 4. End-to-end: reproducción del bug observado en producción
# ---------------------------------------------------------------------------
class TestProductionRepro:
    def test_habichuelas_monthly_2_personas_seco_realista(self):
        """Plan real (2026-05-04): 350g habichuelas cocidas/ciclo de 3 días.
        Multiplicador mensual × 2 personas = 18.67. Pre-fix: ~14 lbs. Post-fix:
        14 × 0.35 ≈ 5 lbs secas — coherente con SKU comercial."""
        # Simular agregación de los 2 platos del plan real
        plan_items = [
            "150g de habichuelas rojas cocidas",
            "200g de habichuelas rojas cocidas (con su caldo)",
        ]
        multiplier = (2 * 4) * (7 / 3)  # 18.67 (mensual × 2 personas, 3 días generados)

        total_g = 0.0
        for item in plan_items:
            qty, unit, name = _parse_quantity(
                item, apply_yield_multiplier=False, apply_legumbres_yield_only=True,
            )
            assert unit == "g"
            assert name == "Habichuelas rojas"
            total_g += qty * multiplier

        total_lbs = total_g / 453.592
        # Pre-fix: 6534g ≈ 14.4 lbs (over-buy 3×). Post-fix: ~2287g ≈ 5 lbs.
        assert 4.0 <= total_lbs <= 6.0, (
            f"Habichuelas mensual debe estar en rango realista 4-6 lbs secas, "
            f"recibido {total_lbs:.2f} lbs (g_total={total_g:.0f}g)"
        )

    def test_lentejas_monthly_2_personas_seco_realista(self):
        """Plan real: 2 tazas lentejas cocidas (400g) en 1 plato. × 18.67 mensual."""
        qty, unit, name = _parse_quantity(
            "2 tazas de lentejas cocidas (400g)",
            apply_yield_multiplier=False, apply_legumbres_yield_only=True,
        )
        # qty = 0.7 tazas — el aggregator multiplica × 18.67 y luego convierte
        # vía density (192g/taza para lentejas). 0.7 × 18.67 × 192 = 2510g ≈ 5.5 lbs
        total_tazas = qty * (2 * 4) * (7 / 3)
        total_g = total_tazas * 192  # density_g_per_cup de lentejas
        total_lbs = total_g / 453.592
        assert 4.0 <= total_lbs <= 7.0

    def test_pollo_NO_se_ve_afectado(self):
        """Pollo cocido en el aggregator sigue en peso literal — preserva
        simetría P1-2 plan↔inventario."""
        qty, _, _ = _parse_quantity(
            "350g de pavo molido",
            apply_yield_multiplier=False, apply_legumbres_yield_only=True,
        )
        assert qty == 350.0
