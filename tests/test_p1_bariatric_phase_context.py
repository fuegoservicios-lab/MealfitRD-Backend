"""[P1-BARIATRIC-PHASE-CONTEXT · 2026-06-27] Segunda iteración bariátrica tras re-test en vivo (corr=5b30b71f):
mis reglas (cap queso/yogurt + 6 comidas + variety relax) funcionaron, pero el reviewer médico seguía
rechazando crítico porque (1) asumía FASE POST-OP TEMPRANA (puré: rechazaba fibra/vegetales enteros/
leguminosas/granos integrales/mariscos — todos OK en mantenimiento) y (2) déficit de proteína (target 100g
inalcanzable en volumen bariátrico + planner usaba leguminosas de baja densidad).

Fixes de esta iteración (asumiendo fase MANTENIMIENTO, el caso común de un usuario de la app):
  (a)  Reviewer recibe CONTEXTO DE FASE (mantenimiento) → deja de aplicar reglas de fase temprana.
  (a') Generador: prompt_block bariátrico declara fase mantenimiento / dieta general.
  (b)  Proteína animal densa para bariátrica (skip garantía-leguminosa + high-density, como gain_muscle).
  (b') Cap del TARGET de proteína bariátrica a nivel achievable en el pouch (knob, default 90g/día).
"""
from __future__ import annotations

from pathlib import Path

import nutrition_calculator as nc

_BACKEND = Path(nc.__file__).resolve().parent


def _baria_form(**over):
    f = {"weight": 90, "weightUnit": "kg", "height": 170, "age": 45, "gender": "male",
         "activityLevel": "moderate", "mainGoal": "maintenance", "medicalConditions": ["Cirugía Bariátrica"]}
    f.update(over)
    return f


# ──────────────────────────── (b') cap del target de proteína ────────────────────────────

def test_bariatric_protein_target_capped():
    capped = nc.get_nutrition_targets(_baria_form())["macros"]["protein_g"]
    uncapped = nc.get_nutrition_targets(_baria_form(medicalConditions=["Ninguna"]))["macros"]["protein_g"]
    assert capped <= 90, f"proteína bariátrica no capeada: {capped}"
    assert uncapped > capped, f"control sin bariátrica debería ser mayor: {uncapped} vs {capped}"


def test_bariatric_protein_cap_via_freetext():
    capped = nc.get_nutrition_targets(_baria_form(medicalConditions=["Ninguna"],
                                                  otherConditions="bypass gastrico hace 1 año"))["macros"]["protein_g"]
    assert capped <= 90


def test_bariatric_protein_cap_redistributes_to_fat_not_carbs():
    """Las kcal liberadas del cap de proteína van a GRASA (no carbos → evita dumping).

    [reapuntado 2026-07-27] Comparaba GRAMOS absolutos y una feature posterior —el techo kcal
    bariátrico (`bariatric_kcal_ceiling_applied`: la capacidad del pouch hace inalcanzable el
    TDEE; 2000 vs 2700 kcal del control, deliberada y con nota clínica propia)— hizo esa
    comparación de manzanas con peras: menos kcal totales ⇒ menos grasa absoluta aunque la
    redistribución sea correcta. La invariante vive en las FRACCIONES: la grasa bariátrica pesa
    MÁS del total que en el control (recibe lo liberado) y los carbos NO pesan más (dumping).
    Medido al reapuntar: grasa 39% vs 30%, carbos 45% vs 45%.
    """
    b = nc.get_nutrition_targets(_baria_form())["macros"]
    c = nc.get_nutrition_targets(_baria_form(medicalConditions=["Ninguna"]))["macros"]

    def _kcal(m):
        return m["protein_g"] * 4 + m["carbs_g"] * 4 + m["fats_g"] * 9

    fat_frac_b = b["fats_g"] * 9 / _kcal(b)
    fat_frac_c = c["fats_g"] * 9 / _kcal(c)
    carb_frac_b = b["carbs_g"] * 4 / _kcal(b)
    carb_frac_c = c["carbs_g"] * 4 / _kcal(c)
    assert fat_frac_b >= fat_frac_c, (fat_frac_b, fat_frac_c)
    assert carb_frac_b <= carb_frac_c + 0.02, (
        f"los carbos bariátricos pesan más que en el control ({carb_frac_b:.0%} vs "
        f"{carb_frac_c:.0%}): las kcal del cap de proteína se fueron a carbos → riesgo dumping"
    )


def test_small_bariatric_patient_not_overcapped():
    # paciente pequeño cuyo target ya es < 90g no debe subir artificialmente
    small = nc.get_nutrition_targets(_baria_form(weight=50, gender="female"))["macros"]["protein_g"]
    assert small <= 90


# ──────────────────────────── (a)(a')(b) anchors estructurales ────────────────────────────

def test_reviewer_receives_maintenance_phase_note():
    go = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "P1-BARIATRIC-PHASE-CONTEXT" in go
    assert "_baria_note" in go and "FASE DE MANTENIMIENTO" in go
    # el note se inyecta en el prompt del reviewer
    assert "{_baria_note}" in go


def test_generator_prompt_declares_maintenance_phase():
    cr = (_BACKEND / "condition_rules.py").read_text(encoding="utf-8")
    # el prompt_block bariátrico declara fase mantenimiento / dieta general
    assert "MANTENIMIENTO" in cr and "dieta general" in cr.lower()


def test_bariatric_uses_dense_protein_path():
    ai = (_BACKEND / "ai_helpers.py").read_text(encoding="utf-8")
    assert "P1-BARIATRIC-PROTEIN-DENSITY" in ai
    assert "_is_bariatric" in ai
    # skip de la garantía de leguminosa y high-density extendidos a bariátrica
    assert "_GOALS_SKIP_LEGUME_GUARANTEE or _is_bariatric" in ai
    assert '_main_goal == "gain_muscle" or _is_bariatric' in ai


def test_protein_cap_knob_anchor():
    ncs = (_BACKEND / "nutrition_calculator.py").read_text(encoding="utf-8")
    assert "P1-BARIATRIC-PROTEIN-TARGET" in ncs
    assert "MEALFIT_BARIATRIC_PROTEIN_MAX_G" in ncs
