"""[P3-SLOT-DISTRIBUTION · 2026-06-13] Distribución intra-día de macros/kcal por slot.

Hallazgo HIGH de la auditoría clínica: el solver usaba el cal_share del LLM como target de
cada comida, propagando la distribución desbalanceada (D1 concentraba 48% kcal / 62% proteína
en el desayuno; 3 comidas bajo el umbral leucínico ~22g). `_canonical_slot_fractions` usa el
split fisiológico canónico (desayuno 20% / almuerzo 35% / merienda 15% / cena 30%) → el solver
redistribuye kcal+proteína equitativamente con cada comida principal por encima del umbral.

Cubre: (1) split canónico por nombre, (2) suma a 1.0, (3) independencia del orden, (4)
independencia del cal_share del LLM, (5) slots no-mapeados reciben el remanente, (6) 3 comidas.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_orchestrator import _canonical_slot_fractions


def _meals(*slots, cals=None):
    out = []
    for i, s in enumerate(slots):
        m = {"slot": s, "name": s}
        if cals:
            m["cals"] = cals[i]
        out.append(m)
    return out


def test_split_canonico_4_comidas():
    fr = _canonical_slot_fractions(_meals("Desayuno", "Almuerzo", "Merienda", "Cena"))
    assert fr == [0.20, 0.35, 0.15, 0.30]


def test_suma_a_uno():
    fr = _canonical_slot_fractions(_meals("Desayuno", "Almuerzo", "Merienda", "Cena"))
    assert abs(sum(fr) - 1.0) < 1e-9


def test_independiente_del_orden():
    # Mismo conjunto de slots en otro orden → cada slot conserva SU fracción.
    fr = _canonical_slot_fractions(_meals("Cena", "Desayuno", "Merienda", "Almuerzo"))
    assert fr[0] == 0.30  # Cena
    assert fr[1] == 0.20  # Desayuno
    assert fr[2] == 0.15  # Merienda
    assert fr[3] == 0.35  # Almuerzo


def test_independiente_del_cal_share_del_llm():
    # Aunque el LLM ponga 979 kcal en desayuno y poco en el resto, la fracción es la canónica.
    fr = _canonical_slot_fractions(
        _meals("Desayuno", "Almuerzo", "Merienda", "Cena", cals=[979, 542, 202, 327]))
    assert fr == [0.20, 0.35, 0.15, 0.30]  # NO refleja el 48% del desayuno


def test_ninguna_comida_supera_40pct():
    fr = _canonical_slot_fractions(_meals("Desayuno", "Almuerzo", "Merienda", "Cena"))
    assert max(fr) <= 0.40  # cierra el "ninguna comida >40% del total"


def test_split_3_comidas():
    fr = _canonical_slot_fractions(_meals("Desayuno", "Almuerzo", "Cena"))
    assert fr == [0.30, 0.40, 0.30]


def test_slot_no_mapeado_recibe_remanente():
    # 'Postre' no mapea → recibe parte igual del remanente; el vector sigue sumando 1.0.
    fr = _canonical_slot_fractions(_meals("Desayuno", "Almuerzo", "Postre", "Cena"))
    assert abs(sum(fr) - 1.0) < 1e-9
    assert fr[0] == 0.20 and fr[1] == 0.35 and fr[3] == 0.30
    assert fr[2] > 0  # Postre recibe el remanente (0.15)


def test_sin_comidas_lista_vacia():
    assert _canonical_slot_fractions([]) == []


def test_protein_por_slot_sobre_umbral_leucinico():
    # Con target diario 119g (techo C1 para ~54kg), cada comida PRINCIPAL queda sobre ~22g.
    fr = _canonical_slot_fractions(_meals("Desayuno", "Almuerzo", "Merienda", "Cena"))
    daily_p = 119.0
    prot = [round(f * daily_p, 1) for f in fr]
    # almuerzo (0.35) y cena (0.30) son las comidas principales
    assert prot[1] >= 22 and prot[3] >= 22
    # ninguna comida concentra >50% (el desayuno bajó de 62% a 20%)
    assert max(prot) / sum(prot) < 0.40


# ── Integración: el solver redistribuye la proteína a los targets canónicos ──
_ROWS = [
    {"name": "Yogurt griego", "aliases": ["yogurt griego", "yogur griego"], "kcal_per_100g": 59.0,
     "protein_g_per_100g": 10.0, "carbs_g_per_100g": 3.6, "fats_g_per_100g": 0.4, "density_g_per_cup": 245.0},
    {"name": "Avena", "aliases": ["avena"], "kcal_per_100g": 382.0,
     "protein_g_per_100g": 13.2, "carbs_g_per_100g": 67.7, "fats_g_per_100g": 6.5, "density_g_per_cup": 90.0},
    {"name": "Pechuga de pollo", "aliases": ["pollo", "pechuga"], "kcal_per_100g": 165.0,
     "protein_g_per_100g": 31.0, "carbs_g_per_100g": 0.0, "fats_g_per_100g": 3.6},
    {"name": "Arroz integral", "aliases": ["arroz", "arroz integral"], "kcal_per_100g": 112.0,
     "protein_g_per_100g": 2.6, "carbs_g_per_100g": 24.0, "fats_g_per_100g": 0.9, "density_g_per_cup": 195.0},
]


def test_solver_redistribuye_proteina_a_targets_canonicos():
    from nutrition_db import IngredientNutritionDB
    from graph_orchestrator import _apply_macro_solver_to_meal, _meal_macro_num
    db = IngredientNutritionDB(rows=_ROWS)
    # Día con proteína amontonada en el desayuno (réplica del hallazgo D1).
    meals = [
        {"slot": "Desayuno", "name": "Avena con Yogurt", "protein": 75, "carbs": 100, "fats": 10, "cals": 800,
         "ingredients": ["2 taza de yogurt griego (490g)", "1 taza de avena (90g)"]},
        {"slot": "Almuerzo", "name": "Pollo con Arroz", "protein": 12, "carbs": 40, "fats": 4, "cals": 300,
         "ingredients": ["50g de pechuga de pollo", "1 taza de arroz integral (195g)"]},
        {"slot": "Merienda", "name": "Yogurt", "protein": 8, "carbs": 6, "fats": 1, "cals": 90,
         "ingredients": ["0.5 taza de yogurt griego (120g)"]},
        {"slot": "Cena", "name": "Pollo", "protein": 14, "carbs": 5, "fats": 3, "cals": 150,
         "ingredients": ["80g de pechuga de pollo", "0.5 taza de arroz integral (98g)"]},
    ]
    daily = {"kcal": 1800.0, "protein": 119.0, "carbs": 180.0, "fats": 50.0}
    fr = _canonical_slot_fractions(meals)
    for m, f in zip(meals, fr):
        st = {k: daily[k] * f for k in ("kcal", "protein", "carbs", "fats")}
        _apply_macro_solver_to_meal(m, st, db)
    prot = [_meal_macro_num(m["protein"]) for m in meals]
    total = sum(prot)
    # El desayuno YA NO concentra la mayoría: bajó de 62% a ~20%.
    assert prot[0] / total < 0.30
    # Las comidas principales (almuerzo 35%, cena 30%) superan el umbral leucínico ~22g.
    assert prot[1] >= 30 and prot[3] >= 25
    # Ninguna comida concentra >40% del total.
    assert max(prot) / total < 0.40
