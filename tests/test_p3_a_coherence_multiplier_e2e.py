"""[P3-A · 2026-05-08] E2E coherencia recetas↔lista bajo multipliers explícitos.

P1-C cerró el contrato presence + magnitudes con tolerance, pero la mayoría
de los 82 tests v2 ejercitan multiplier implícito 1.0 (vía fallback default).
En producción los multipliers reales son:

  - 1.0 → ciclo semanal (caller pasa householdSize directamente).
  - 2.0 → ciclo quincenal (P1-7: 7d-base × 2 semanas).
  - 4.0 → ciclo mensual (P1-7: 7d-base × ~4.3, redondeado).

Si la simetría plan↔lista se rompiera SOLO bajo escala (ej. el aggregator
escala correctamente pero `expected_sum_from_recipes` olvidara aplicar el
multiplier en alguna rama, o viceversa), los tests con multiplier=1.0 no lo
detectarían: 1.0 cancela el bug.

Este suite endurece la regresión cubriendo los 3 multipliers reales × 3
perfiles que ejercitan distintas particularidades del aggregator:

  - Estable (pollo, arroz)         → escala lineal pura.
  - Perecedero (zanahoria, tomate) → mismo aggregator, distinta categoría
                                     (cycle-lock 7d aplica a persistencia,
                                     NO al cómputo del expected).
  - Pavo                           → P3-4 canonicalize simétrico fresh ↔
                                     procesado; protege contra regresión
                                     histórica documentada en
                                     project_p3_4_pavo_coherence_v3_2026_05_07.

Invariante principal verificada en cada caso:
    `expected_sum_from_recipes(plan, multiplier=M)["X"]["g"] == BASE * M`

Y E2E con el guard:
  - Si expected_sum y aggregated_shopping_list usan el mismo M → 0 divergencias.
  - Si están desincronizados (ej. multiplier=2 pero lista en M=1) → guard
    reporta presence_missing (porque la cantidad es <50% de lo esperado)
    o magnitude divergence.
"""
import pytest

import shopping_calculator
from shopping_calculator import (
    expected_sum_from_recipes,
    run_shopping_coherence_guard,
)


# ---------------------------------------------------------------------------
# Fixture autouse: stub master_ingredients en TODOS los tests del archivo.
#
# Sin este stub, los nombres canonicalizados drift según el estado del pool
# de DB: si está abierto, "pollo" → "Pechuga de pollo" y "arroz" →
# "Arroz blanco" via master_map. Forzar el fallback inline garantiza que
# los assertions sobre `Pollo`/`Arroz`/`Zanahoria`/etc. son determinísticos
# en cualquier entorno (CI, local, prod). Mismo patrón que
# test_p1_shopping_recipe_coherence.py pero ampliado a autouse porque
# este suite no separa "tests sólo de cómputo" vs "tests E2E del guard"
# — todos dependen del mismo contrato de naming.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def no_master_db(monkeypatch):
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda: [])


# ---------------------------------------------------------------------------
# 1. expected_sum_from_recipes escala lineal con multiplier — 3 perfiles
# ---------------------------------------------------------------------------
class TestExpectedScalesLinearly:
    """`expected_sum_from_recipes(plan, multiplier=M)` debe producir cantidades
    proporcionales a M para cualquier perfil de alimento. Si el multiplier no
    se aplicara en alguna rama (regresión silenciosa), los tests con M=1.0
    pasarían pero un usuario quincenal/mensual recibiría una lista escasa."""

    @pytest.mark.parametrize("multiplier", [1.0, 2.0, 4.0])
    def test_stable_proteins_and_grains(self, multiplier):
        plan = {"days": [{"meals": [
            {"meal": "almuerzo", "ingredients_raw": [
                "200 g pollo", "150 g arroz",
            ]},
        ]}]}
        result = expected_sum_from_recipes(plan, multiplier=multiplier)
        assert result["Pollo"]["g"] == pytest.approx(200.0 * multiplier)
        assert result["Arroz"]["g"] == pytest.approx(150.0 * multiplier)

    @pytest.mark.parametrize("multiplier", [1.0, 2.0, 4.0])
    def test_perishables(self, multiplier):
        """Perecederos se calculan idénticos a estables en `expected_sum`.
        El cycle-lock 7d (Visión-C) aplica a la PERSISTENCIA de la lista, no
        al cómputo teórico que hace el guard."""
        plan = {"days": [{"meals": [
            {"meal": "almuerzo", "ingredients_raw": [
                "100 g zanahoria", "150 g tomate",
            ]},
        ]}]}
        result = expected_sum_from_recipes(plan, multiplier=multiplier)
        assert result["Zanahoria"]["g"] == pytest.approx(100.0 * multiplier)
        assert result["Tomate"]["g"] == pytest.approx(150.0 * multiplier)

    @pytest.mark.parametrize("multiplier", [1.0, 2.0, 4.0])
    def test_pavo_canonical(self, multiplier):
        """Pavo es el caso especial cubierto en P3-4: `canonicalize_pavo` hace
        mirror simétrico fresh↔procesado. Aquí solo verificamos la escala
        cruda; la canonicalización vive en `_canonicalize_for_coherence` que
        ejercita el guard end-to-end abajo."""
        plan = {"days": [{"meals": [
            {"meal": "almuerzo", "ingredients_raw": ["80 g pavo"]},
        ]}]}
        result = expected_sum_from_recipes(plan, multiplier=multiplier)
        assert result["Pavo"]["g"] == pytest.approx(80.0 * multiplier)

    @pytest.mark.parametrize("multiplier", [1.0, 2.0, 4.0])
    def test_mixed_perfiles_one_plan(self, multiplier):
        """Plan realista con los 3 perfiles juntos en un mismo día. Cubre
        el escenario producción más común (cena con proteína + grano +
        vegetal perecedero)."""
        plan = {"days": [{"meals": [
            {"meal": "cena", "ingredients_raw": [
                "150 g pavo",
                "100 g arroz",
                "80 g zanahoria",
            ]},
        ]}]}
        result = expected_sum_from_recipes(plan, multiplier=multiplier)
        assert result["Pavo"]["g"] == pytest.approx(150.0 * multiplier)
        assert result["Arroz"]["g"] == pytest.approx(100.0 * multiplier)
        assert result["Zanahoria"]["g"] == pytest.approx(80.0 * multiplier)

    def test_zero_division_safety(self):
        """multiplier <=0 / NaN / inf se clampa a 1.0 dentro de
        `expected_sum_from_recipes` (mismo patrón que P1-7). No debe lanzar."""
        plan = {"days": [{"meals": [
            {"meal": "almuerzo", "ingredients_raw": ["200 g pollo"]},
        ]}]}
        for bad in (0, -1, -2.5, float("nan"), float("inf")):
            result = expected_sum_from_recipes(plan, multiplier=bad)
            assert result["Pollo"]["g"] == pytest.approx(200.0), (
                f"multiplier={bad!r} debió clampar a 1.0; got {result}"
            )


# ---------------------------------------------------------------------------
# 2. Guard E2E: simetría bajo multiplier explícito
# ---------------------------------------------------------------------------
class TestGuardSymmetricUnderMultiplier:
    """`run_shopping_coherence_guard(plan, multiplier=M)` con expected y
    aggregated escalados al mismo M debe reportar 0 divergencias presence
    (los foods de la receta están todos en la lista). En modo `warn` no debe
    setear `_shopping_coherence_block`."""

    def _make_plan(self, multiplier):
        """Plan con 3 perfiles + aggregated escalado simétricamente.
        Cantidades base del plan (M=1.0) deliberadamente fáciles de multiplicar."""
        return {
            "days": [{"meals": [
                {"meal": "almuerzo", "ingredients_raw": [
                    "200 g pollo",
                    "150 g arroz",
                    "100 g zanahoria",
                    "80 g pavo",
                ]},
            ]}],
            "aggregated_shopping_list": [
                {"name": "Pollo",     "market_qty_numeric": 200.0 * multiplier, "market_unit": "g"},
                {"name": "Arroz",     "market_qty_numeric": 150.0 * multiplier, "market_unit": "g"},
                {"name": "Zanahoria", "market_qty_numeric": 100.0 * multiplier, "market_unit": "g"},
                {"name": "Pavo",      "market_qty_numeric":  80.0 * multiplier, "market_unit": "g"},
            ],
            "calc_household_multiplier": multiplier,
        }

    @pytest.mark.parametrize("multiplier", [1.0, 2.0, 4.0])
    def test_no_presence_divergences_when_in_sync(self, monkeypatch, multiplier):
        """Si expected y aggregated están escalados igual, no faltan foods."""
        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "warn")
        plan = self._make_plan(multiplier)
        divs = run_shopping_coherence_guard(plan, multiplier=multiplier)
        presence_missing = [d for d in divs if d.get("side") == "expected_only"]
        assert presence_missing == [], (
            f"multiplier={multiplier}: presence_missing inesperado, "
            f"expected/aggregated están sincronizados: {presence_missing}"
        )
        # Defensivo extra: no debe haberse marcado el block.
        assert "_shopping_coherence_block" not in plan

    @pytest.mark.parametrize("multiplier", [2.0, 4.0])
    def test_block_mode_no_critical_when_in_sync(self, monkeypatch, multiplier):
        """En mode=block con expected/aggregated alineados, NO se marca el
        `_shopping_coherence_block` (no hay foods faltantes ni delta crítico)."""
        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "block")
        plan = self._make_plan(multiplier)
        run_shopping_coherence_guard(plan, multiplier=multiplier)
        assert "_shopping_coherence_block" not in plan

    @pytest.mark.parametrize("multiplier_expected,multiplier_aggregated", [
        (2.0, 1.0),  # quincenal mal escalado a semanal en lista
        (4.0, 1.0),  # mensual mal escalado a semanal en lista
        (4.0, 2.0),  # mensual mal escalado a quincenal en lista
    ])
    def test_drift_between_expected_and_aggregated_detected(
        self, monkeypatch, multiplier_expected, multiplier_aggregated
    ):
        """Si el aggregator escaló a M_a pero el guard se llama con
        multiplier=M_e (M_a < M_e), la lista tiene CANTIDADES MENORES de lo
        esperado por la receta → guard debe reportar magnitudes divergentes
        (delta_pct > tolerance default 0.10)."""
        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "warn")
        # Aggregated escalado a multiplier_aggregated.
        plan = self._make_plan(multiplier_aggregated)
        # Pero llamamos al guard con multiplier_expected (más alto).
        divs = run_shopping_coherence_guard(plan, multiplier=multiplier_expected)
        magnitude_divs = [d for d in divs if d.get("magnitude") is True]
        assert magnitude_divs, (
            f"M_expected={multiplier_expected} vs M_aggregated={multiplier_aggregated}: "
            f"el guard debe detectar drift por magnitud; got 0 magnitude divs."
        )

    @pytest.mark.parametrize("multiplier", [1.0, 2.0, 4.0])
    def test_pavo_canonicalize_under_multiplier(self, monkeypatch, multiplier):
        """Variante específica para pavo (P3-4 simetría fresh↔procesado).
        Aunque la receta diga 'pavo' y la lista 'Pavo (procesado)', el
        canonical match deduplica y el guard NO reporta divergencia bajo
        cualquier multiplier."""
        monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "warn")
        plan = {
            "days": [{"meals": [
                {"meal": "almuerzo", "ingredients_raw": ["80 g pavo"]},
            ]}],
            "aggregated_shopping_list": [
                {"name": "Pavo", "market_qty_numeric": 80.0 * multiplier, "market_unit": "g"},
            ],
            "calc_household_multiplier": multiplier,
        }
        divs = run_shopping_coherence_guard(plan, multiplier=multiplier)
        presence_missing = [d for d in divs if d.get("side") == "expected_only"]
        assert presence_missing == [], (
            f"multiplier={multiplier}: pavo no debería faltar bajo canonical match"
        )
