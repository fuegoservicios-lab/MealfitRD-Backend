"""[P2-COH-1] Tests para sinónimos expandidos en el chequeo
recipe-vs-ingredient coherence (`_run_assembly_validations`).

Bug original (incidente 2026-05-05):
  El revisor médico flageó:
    "Día 3, Merluza Estofada Boca Chica: La receta indica 'pescado' pero
    no hay ningún ingrediente equivalente (ej. pescado, chillo, dorado)
    listado."
  El plan tenía "merluza" en `ingredients_raw` y "el pescado" en el texto
  de la receta. El check NO conocía merluza como sinónimo de pescado →
  falso positivo HIGH severity → plan rechazado.

Fix:
  Expandir `protein_synonyms["pescado"]` en `graph_orchestrator._run_assembly_validations`
  con los peces caribeños comunes (merluza, róbalo, pargo, corvina,
  mahi-mahi, lubina, carite, jurel, lambí). Tambien expandir cortes
  comunes de pollo, res, cerdo y pavo. Sincronizar con
  `constants.PROTEIN_SYNONYMS["pescado"]` para SSOT.

Cobertura:
  - Repro exacto del incidente (merluza + "el pescado")
  - Otros peces caribeños comunes
  - Otros cortes (cerdo costilla, res molida, pavo molido, pollo encuentros)
  - Negative tests: ingredients que NO matchean siguen flageándose
  - SSOT: las adiciones también están en constants.PROTEIN_SYNONYMS
"""
import re

import pytest

from constants import PROTEIN_SYNONYMS
from graph_orchestrator import _run_assembly_validations


# ---------------------------------------------------------------------------
# Helper para construir un plan mínimo que pase por el check
# ---------------------------------------------------------------------------
def _make_plan_with_meal(meal_name: str, recipe_text: str, ingredients: list[str]) -> dict:
    """Construye un plan con 1 día y 1 meal cumpliendo el contrato mínimo
    de `_run_assembly_validations`. Incluye `servir` en la receta para no
    disparar el check de "incompleto" — solo testeamos coherence."""
    return {
        "main_goal": "Test goal",
        "insights": [],
        "days": [
            {
                "day": 1,
                "day_name": "Lunes",
                "macros": {"protein_g": 100, "carbs_g": 100, "fats_g": 30, "calories": 1200},
                "meals": [
                    {
                        "meal": "Cena",
                        "name": meal_name,
                        "time": "20:00",
                        "desc": "Test",
                        "ingredients": ingredients,
                        "recipe": recipe_text + " Servir caliente.",
                        "macros": [],
                        "prep_time": "15 min",
                        "difficulty": "Fácil",
                        "cals": 600,
                        "protein": 40,
                        "carbs": 50,
                        "fats": 20,
                    }
                ],
            }
        ],
    }


def _run_check(meal_name: str, recipe: str, ingredients: list[str]) -> list[str]:
    """Corre las validaciones de assembly sobre un plan mínimo y devuelve
    los recipe_coherence_errors (lista vacía si pasa)."""
    plan = _make_plan_with_meal(meal_name, recipe, ingredients)
    _run_assembly_validations(plan, skeleton={}, affected_days_set=set())
    return plan.get("_recipe_coherence_errors", [])


# ---------------------------------------------------------------------------
# 1. Repro exacto del incidente 2026-05-05
# ---------------------------------------------------------------------------
def test_repro_incident_2026_05_05_merluza_es_pescado():
    """La receta del Día 3 mencionaba "el pescado" pero el ingrediente era
    "merluza". Pre-fix: rechazo HIGH → plan entregado roto. Post-fix: pasa."""
    errors = _run_check(
        meal_name="Merluza Estofada Boca Chica",
        recipe="Sazonar el pescado con limón. Cocinar el pescado en sartén.",
        ingredients=["250g de filete de merluza", "1/2 limón", "1 cdta aceite de oliva"],
    )
    # No debe haber error de "pescado no listado" — merluza ahora es synonym
    pescado_errors = [e for e in errors if "pescado" in e.lower() and "equivalente" in e.lower()]
    assert pescado_errors == [], (
        f"Esperado que merluza sea reconocido como pescado, recibido errores: {pescado_errors}"
    )


# ---------------------------------------------------------------------------
# 2. Otros peces caribeños comunes
# ---------------------------------------------------------------------------
class TestFishSynonyms:
    @pytest.mark.parametrize("fish", [
        "merluza", "róbalo", "robalo", "pargo", "corvina",
        "mahi-mahi", "mahi mahi", "lubina", "carite", "jurel",
        # Originales (sanity check no-regresión)
        "chillo", "dorado", "mero", "salmón", "salmon", "tilapia",
        "bacalao", "atún", "atun", "sardina",
    ])
    def test_fish_recognized_as_pescado(self, fish):
        """Cada fish específico debe satisfacer recipes que digan 'pescado'."""
        errors = _run_check(
            meal_name=f"{fish.capitalize()} a la Plancha",
            recipe=f"Sazonar el pescado. Cocinar el pescado a la plancha.",
            ingredients=[f"200g de {fish}", "1 limón"],
        )
        pescado_err = [e for e in errors if "pescado" in e.lower() and "equivalente" in e.lower()]
        assert pescado_err == [], (
            f"Pez '{fish}' no fue reconocido como pescado: {pescado_err}"
        )


# ---------------------------------------------------------------------------
# 3. Otros cortes y preparaciones añadidos
# ---------------------------------------------------------------------------
class TestExtendedSynonyms:
    def test_pollo_encuentros_recognized(self):
        errors = _run_check(
            meal_name="Encuentros Guisados",
            recipe="Sazonar el pollo. Guisar el pollo a fuego lento.",
            ingredients=["350g de encuentros de pollo", "1 cebolla"],
        )
        pollo_err = [e for e in errors if "pollo" in e.lower() and "equivalente" in e.lower()]
        assert pollo_err == []

    def test_pollo_alas_recognized(self):
        errors = _run_check(
            meal_name="Alas BBQ",
            recipe="Adobar el pollo. Hornear el pollo a 200°C.",
            ingredients=["6 alas de pollo", "salsa BBQ al gusto"],
        )
        assert not [e for e in errors if "pollo" in e.lower() and "equivalente" in e.lower()]

    def test_res_molida_recognized(self):
        errors = _run_check(
            meal_name="Albóndigas Caseras",
            recipe="Mezclar la carne con cebolla. Formar bolitas y cocinar la res.",
            ingredients=["300g de carne molida de res", "1/2 cebolla"],
        )
        assert not [e for e in errors if e.lower().startswith("día 1, albóndigas") and "res" in e.lower()]

    def test_cerdo_costilla_recognized(self):
        errors = _run_check(
            meal_name="Costillas BBQ",
            recipe="Adobar el cerdo. Hornear el cerdo lentamente.",
            ingredients=["500g de costilla de cerdo", "salsa BBQ"],
        )
        assert not [e for e in errors if "cerdo" in e.lower() and "equivalente" in e.lower()]

    def test_pavo_molido_recognized(self):
        errors = _run_check(
            meal_name="Hamburguesas de Pavo",
            recipe="Formar discos con el pavo. Cocinar el pavo a la plancha.",
            ingredients=["300g de pavo molido", "1 huevo"],
        )
        assert not [e for e in errors if "pavo" in e.lower() and "equivalente" in e.lower()]

    def test_camaron_singular_recognized(self):
        """LLM puede escribir 'camarón' (singular) en receta y 'camarones' en ingredients."""
        errors = _run_check(
            meal_name="Arroz con Camarones",
            recipe="Saltear el camarón en aceite. Mezclar el camarón con arroz.",
            ingredients=["200g de camarones limpios", "1 taza de arroz"],
        )
        assert not [e for e in errors if "camarón" in e.lower() and "equivalente" in e.lower()]


# ---------------------------------------------------------------------------
# 4. Negative tests: el check sigue flageando incoherencias REALES
# ---------------------------------------------------------------------------
class TestNegativeStillFlags:
    def test_recipe_dice_pescado_sin_ingrediente_de_pez(self):
        """Si la receta menciona pescado pero ingredients NO tiene ningún
        pez, el check debe flagear (regresión guard)."""
        errors = _run_check(
            meal_name="Plato Confuso",
            recipe="Cocinar el pescado a fuego alto. Servir caliente.",
            ingredients=["200g de pollo", "1 cebolla"],  # No hay pez
        )
        pescado_err = [e for e in errors if "pescado" in e.lower() and "equivalente" in e.lower()]
        assert len(pescado_err) >= 1, (
            f"Recipe sin ingredient de pez DEBE flagearse, recibido: {pescado_err}"
        )

    def test_recipe_dice_res_sin_ingrediente_de_res(self):
        errors = _run_check(
            meal_name="Plato Confuso",
            recipe="Cocinar la res a la plancha. Servir.",
            ingredients=["200g de tilapia", "1 cebolla"],
        )
        res_err = [e for e in errors if "'res'" in e.lower() and "equivalente" in e.lower()]
        assert len(res_err) >= 1


# ---------------------------------------------------------------------------
# 5. SSOT — verificar coherencia con constants.PROTEIN_SYNONYMS
# ---------------------------------------------------------------------------
class TestSSOTConsistency:
    """Las adiciones del fix también deben estar en constants.PROTEIN_SYNONYMS
    para que fact_extractor / agent / ai_helpers reconozcan los nuevos
    sinónimos al canonicalizar ingredientes."""

    @pytest.mark.parametrize("fish", [
        "merluza", "róbalo", "robalo", "pargo", "corvina",
        "mahi-mahi", "lubina", "carite", "jurel",
    ])
    def test_fish_present_in_constants_pescado(self, fish):
        """Los peces nuevos también deben estar en constants para SSOT."""
        pescado_synonyms = [s.lower() for s in PROTEIN_SYNONYMS.get("pescado", [])]
        assert fish.lower() in pescado_synonyms, (
            f"Sinónimo '{fish}' está en graph_orchestrator pero NO en "
            f"constants.PROTEIN_SYNONYMS['pescado']. Drift entre módulos."
        )
