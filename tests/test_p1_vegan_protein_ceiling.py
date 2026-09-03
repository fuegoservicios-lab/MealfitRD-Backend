# [P1-VEGAN-PROTEIN-CEILING · 2026-08-08] Palanca 2 del 65% (issue #9/#14, OK del owner):
# el target proteico vegano se deriva con techo 1.8 g/kg (knob, clamp [1.0, 2.2]) en vez del
# estándar 2.2/2.6 — el perfil vegana_dm2 recibía 188g inalcanzables con fuentes vegetales
# del catálogo es-DO y el piso rechazaba SIEMPRE (déficit estructural del TARGET, no de la
# generación). Las kcal liberadas van a carbos por la redistribución C1 existente.
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nutrition_calculator import calculate_macros, _protein_ceiling_vegan_g_per_kg


def test_vegan_capea_a_1_8_gkg():
    assert _protein_ceiling_vegan_g_per_kg() == 1.8
    # 85 kg, hipercalórico para forzar proteína alta por %:
    vegan = calculate_macros(3000, "gain_muscle", weight_kg=85, diet="vegana")
    std = calculate_macros(3000, "gain_muscle", weight_kg=85)
    assert vegan["protein_g"] <= round(1.8 * 85), f"vegan sin capear: {vegan['protein_g']}g"
    assert vegan["protein_g"] < std["protein_g"], "el techo vegano debe quedar BAJO el estándar"
    # kcal conservadas: lo liberado va a carbos
    kcal_v = vegan["protein_g"] * 4 + vegan["carbs_g"] * 4 + vegan["fats_g"] * 9
    kcal_s = std["protein_g"] * 4 + std["carbs_g"] * 4 + std["fats_g"] * 9
    assert abs(kcal_v - kcal_s) <= 12, "la redistribución debe conservar las kcal (±redondeo)"


def test_no_vegan_intactos():
    # balanced / vegetarian / pescatarian: byte-idéntico al comportamiento sin diet
    # (huevo/lácteo/pescado sí alcanzan densidad — solo vegan se capea).
    for d in (None, "balanced", "vegetariana", "pescatarian"):
        m = calculate_macros(3000, "gain_muscle", weight_kg=85, diet=d)
        base = calculate_macros(3000, "gain_muscle", weight_kg=85)
        assert m == base, f"dieta {d!r} no debe alterar el target"


def test_vegan_bajo_el_techo_no_se_toca():
    # Si el % de proteína ya queda bajo 1.8 g/kg, el techo vegano no muerde.
    vegan = calculate_macros(1500, "maintenance", weight_kg=90, diet="vegan")
    base = calculate_macros(1500, "maintenance", weight_kg=90)
    assert vegan == base
