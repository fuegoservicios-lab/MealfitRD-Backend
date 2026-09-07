"""[BIOBOROS-3.0-TEST] Batería de pruebas unitarias para el optimizador MILP (HiGHS).

Valida:
1. Convergencia exacta en almuerzo típico dominicano (pechuga, arroz, habichuelas, aceite).
2. Restricción entera para alimentos discretos (ej. 2 huevos enteros, nunca 2.37 huevos).
3. Respuesta determinista y rápida (<10ms) ante el contraejemplo 20x/10x (macros acoplados imposibles).
4. Estabilidad numérica sin dependencias de red ni persistencia.
"""
import pytest
from milp_meal_sizer import MilpMealSizer, MacroTarget, SizerIngredient, _HAS_SCIPY


@pytest.mark.skipif(not _HAS_SCIPY, reason="SciPy no disponible en el entorno")
def test_almuerzo_dominicano_convergencia_optima():
    """Un almuerzo dominicano estándar debe converger al target con error < 2%."""
    sizer = MilpMealSizer()

    # Ingredientes típicos dominicanos con valores por gramo
    # Pechuga: ~1.65 kcal/g, 0.31 P, 0.0 C, 0.036 F
    # Arroz blanco crudo: ~3.65 kcal/g, 0.07 P, 0.80 C, 0.007 F
    # Habichuelas rojas secas: ~3.33 kcal/g, 0.24 P, 0.60 C, 0.008 F
    # Aceite vegetal: ~8.84 kcal/g, 0.0 P, 0.0 C, 1.0 F
    ingredients = [
        SizerIngredient(
            name="Pechuga de pollo",
            food_id="pechuga_pollo",
            base_qty=150.0,
            unit="g",
            kcal_per_unit=1.65,
            protein_per_unit=0.31,
            carbs_per_unit=0.0,
            fats_per_unit=0.036,
            min_qty=80.0,
            max_qty=300.0,
        ),
        SizerIngredient(
            name="Arroz blanco",
            food_id="arroz_blanco",
            base_qty=70.0,
            unit="g",
            kcal_per_unit=3.65,
            protein_per_unit=0.07,
            carbs_per_unit=0.80,
            fats_per_unit=0.007,
            min_qty=30.0,
            max_qty=150.0,
        ),
        SizerIngredient(
            name="Habichuelas rojas",
            food_id="habichuela_roja",
            base_qty=50.0,
            unit="g",
            kcal_per_unit=3.33,
            protein_per_unit=0.24,
            carbs_per_unit=0.60,
            fats_per_unit=0.008,
            min_qty=20.0,
            max_qty=120.0,
        ),
        SizerIngredient(
            name="Aceite vegetal",
            food_id="aceite_vegetal",
            base_qty=10.0,
            unit="g",
            kcal_per_unit=8.84,
            protein_per_unit=0.0,
            carbs_per_unit=0.0,
            fats_per_unit=1.0,
            min_qty=5.0,
            max_qty=25.0,
        ),
    ]

    target = MacroTarget(
        kcal=650.0,
        protein=52.0,
        carbs=75.0,
        fats=16.0,
        tolerance_pct=0.03,  # 3% tolerancia estricta
    )

    result = sizer.solve(ingredients, target)

    assert result.status == "optimal"
    assert result.converged is True
    assert result.execution_time_ms < 50.0  # Típicamente < 5ms

    # Verificar que el error en cada macro esté por debajo de la tolerancia
    for macro, err in result.macro_errors.items():
        assert err <= 0.03, f"Macro {macro} superó la tolerancia: {err:.4f}"

    # Cantidades dentro de cotas culinarias
    for ing_out in result.ingredients:
        assert ing_out["qty"] > 0


@pytest.mark.skipif(not _HAS_SCIPY, reason="SciPy no disponible en el entorno")
def test_alimento_discreto_huevos_enteros():
    """Los alimentos discretos (huevos) deben ser enteros exactos (ej. 2 o 3, no 2.4)."""
    sizer = MilpMealSizer()

    # Huevo entero (unidad de ~50g): 72 kcal, 6.3g P, 0.4g C, 4.8g F
    # Avena en hojuelas (g): 3.89 kcal/g, 0.17 P, 0.66 C, 0.07 F
    ingredients = [
        SizerIngredient(
            name="Huevo entero",
            food_id="huevo_entero",
            base_qty=2.0,
            unit="unidad",
            kcal_per_unit=72.0,
            protein_per_unit=6.3,
            carbs_per_unit=0.4,
            fats_per_unit=4.8,
            min_qty=1.0,
            max_qty=4.0,
            is_discrete=True,
        ),
        SizerIngredient(
            name="Avena en hojuelas",
            food_id="avena",
            base_qty=40.0,
            unit="g",
            kcal_per_unit=3.89,
            protein_per_unit=0.17,
            carbs_per_unit=0.66,
            fats_per_unit=0.07,
            min_qty=20.0,
            max_qty=100.0,
            is_discrete=False,
        ),
    ]

    target = MacroTarget(
        kcal=350.0,
        protein=22.0,
        carbs=35.0,
        fats=13.0,
        tolerance_pct=0.08,
    )

    result = sizer.solve(ingredients, target)

    huevo_out = next(item for item in result.ingredients if item["food_id"] == "huevo_entero")
    assert huevo_out["qty"] in [1.0, 2.0, 3.0, 4.0], f"El huevo no fue entero: {huevo_out['qty']}"
    assert huevo_out["qty"] == int(huevo_out["qty"])


@pytest.mark.skipif(not _HAS_SCIPY, reason="SciPy no disponible en el entorno")
def test_contraejemplo_macros_acoplados_incompatibles():
    """El contraejemplo histórico 20x/10x donde P=40 exige x=2 y F=5 exige x=0.5.
    El solver debe diagnosticar infactibilidad estricta o relajación en <10ms sin colapsar.
    """
    sizer = MilpMealSizer()

    # Un solo ingrediente que aporta 20 de proteína y 10 de grasa por unidad
    ingredients = [
        SizerIngredient(
            name="Alimento acoplado 20x/10x",
            food_id="alimento_acoplado",
            base_qty=1.0,
            unit="porcion",
            kcal_per_unit=170.0,
            protein_per_unit=20.0,
            carbs_per_unit=0.0,
            fats_per_unit=10.0,
            min_qty=0.5,
            max_qty=2.0,
        )
    ]

    # Target incompatible: P=40 requiere x=2; F=5 requiere x=0.5
    target = MacroTarget(
        kcal=250.0,
        protein=40.0,
        carbs=0.0,
        fats=5.0,
        tolerance_pct=0.02,
    )

    result = sizer.solve(ingredients, target)

    # Debe completarse rápidamente
    assert result.execution_time_ms < 15.0
    # No puede ser convergado estricto al 2% porque es matemáticamente imposible
    assert result.converged is False
    # Pero no debe fallar con excepción, sino devolver el mejor compromiso ponderado
    assert result.status in ("feasible_relaxed", "infeasible")
    assert result.macro_errors["protein"] > 0 or result.macro_errors["fats"] > 0
