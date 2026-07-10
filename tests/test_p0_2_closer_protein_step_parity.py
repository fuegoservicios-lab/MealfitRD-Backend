"""[P0-2-PROTEIN-STEP-PARITY · 2026-07-10] (recipe plausibility roadmap, item P0-2) Evidencia visual
(plan 564d6e4e): receta con ingredientes "0.5 pechuga de pollo (porción)" Y "40g de camarones cocido"
— los camarones JAMÁS aparecen en los pasos (Mise en Place / Toque de Fuego / Montaje solo mencionan
pollo). El closer que añadió los camarones (uno de varios en el pipeline: PROTEIN_FLOOR/topup/
protein-band-post-finalize/etc.) nunca invocó `_append_closer_protein_step` para esa línea.

En vez de rastrear CUÁL closer específico lo dejó huérfano (media docena de call sites candidatos a
través de assemble + el shield pre-INSERT), este fix es un SWEEP defensivo universal: recorre
`ingredients` de cada meal, identifica líneas proteína-dominantes (`_ingredient_is_protein_dominant`,
ya usado por P1-PROTEIN-BAND-POST-FINALIZE) y para cada una que el TdF/Montaje NO mencione, invoca el
`_append_closer_protein_step` YA EXISTENTE — que trae su propio dedup (paso duplicado / alimento ya
usado en pasos reales) y su propio wording correcto por tipo de alimento (cocido/lácteo-blando/
legumbre/licuado/plancha). Cero lógica nueva de wording — solo cierre de la brecha de cobertura.
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()
with open(os.path.join(_BACKEND, "db_plans.py"), encoding="utf-8") as f:
    _DBP = f.read()


# ───────────────────────── estructural ─────────────────────────

def test_marker_present():
    assert "P0-2-PROTEIN-STEP-PARITY" in _GO


def test_function_defined_and_reuses_existing_helpers():
    assert "def ensure_protein_step_parity(plan_data" in _GO
    i = _GO.index("def ensure_protein_step_parity(plan_data")
    body = _GO[i:i + 4200]
    assert "_ingredient_is_protein_dominant(" in body, (
        "debe reusar el detector de proteína-dominante ya usado por "
        "P1-PROTEIN-BAND-POST-FINALIZE, no reimplementar un umbral propio"
    )
    assert "_append_closer_protein_step(" in body, (
        "debe delegar en el appender YA EXISTENTE (con su dedup + wording por tipo), "
        "no reimplementar la construcción del paso"
    )


def test_called_in_dbplans_shield():
    assert "ensure_protein_step_parity" in _DBP
    i_rpb = _DBP.index("reconcile_protein_band_post_finalize")
    i_parity = _DBP.index("ensure_protein_step_parity")
    assert i_parity > i_rpb, (
        "debe correr DESPUÉS de reconcile_protein_band_post_finalize (que puede escalar "
        "porciones proteína-dominantes existentes) para barrer el estado final de ingredients"
    )


# ───────────────────────── funcional ─────────────────────────

def _meal_pollo_camarones_orphan():
    return {
        "meal": "Almuerzo", "name": "Papas al horno",
        "ingredients": [
            "1 huevo", "1.5 papas medianas (224 g)", "1 naranja",
            "0.5 pechuga de pollo (porción)", "40 g de camarones cocido",
        ],
        "recipe": [
            "Precalienta el horno a 200°C. Lava y corta la papa en rodajas.",
            "Hornea las papas 15 min, casca un huevo en cada hueco.",
            "💪 Cocina pechuga de pollo a la plancha o hervido y sírvelo como proteína del plato.",
            "Sirve el huevo al horno con las papas en un plato.",
        ],
    }


class _StubDB:
    """Resuelve pollo/camarones como proteína-dominante; el resto no resuelve."""
    _CATALOG = {
        "camarones": {"name": "Camarones cocido", "protein": 8.4, "carbs": 0.0, "fats": 0.4, "kcal": 40.0, "grams": 40},
        "pechuga": {"name": "Pechuga de pollo", "protein": 27.0, "carbs": 0.0, "fats": 3.0, "kcal": 140.0, "grams": 100},
    }

    def macros_from_ingredient_string(self, s):
        sl = str(s).lower()
        for tok, info in self._CATALOG.items():
            if tok in sl:
                return dict(info)
        return None

    def lookup(self, s):
        return self.macros_from_ingredient_string(s)

    def grams_from_ingredient_string(self, s):
        info = self.macros_from_ingredient_string(s)
        return info["grams"] if info else None


def test_functional_adds_step_for_orphaned_protein(monkeypatch):
    import graph_orchestrator as g
    meal = _meal_pollo_camarones_orphan()
    plan_data = {"days": [{"day": 1, "meals": [meal]}]}
    n = g.ensure_protein_step_parity(plan_data, db=_StubDB())
    assert n >= 1
    joined = " ".join(str(s) for s in meal["recipe"]).lower()
    assert "camaron" in joined, "camarones debe ganar un paso — ya tenía ingrediente pero cero mención"


def test_functional_skips_already_mentioned_protein(monkeypatch):
    """El pollo YA tiene su paso 💪 — el sweep NO debe duplicar (dedup del appender existente)."""
    import graph_orchestrator as g
    meal = _meal_pollo_camarones_orphan()
    plan_data = {"days": [{"day": 1, "meals": [meal]}]}
    g.ensure_protein_step_parity(plan_data, db=_StubDB())
    pollo_steps = [s for s in meal["recipe"] if "pollo" in str(s).lower()]
    assert len(pollo_steps) == 1, f"no debe duplicar el paso de pollo ya presente: {pollo_steps}"


def test_functional_noop_on_malformed_input():
    import graph_orchestrator as g
    assert g.ensure_protein_step_parity({}, db=_StubDB()) == 0
    assert g.ensure_protein_step_parity(None, db=_StubDB()) == 0
    assert g.ensure_protein_step_parity({"days": "nope"}, db=_StubDB()) == 0
