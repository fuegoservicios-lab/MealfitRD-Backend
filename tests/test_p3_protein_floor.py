"""[P3-PROTEIN-FLOOR · 2026-06-13] Cierre DURO del déficit de proteína (FS6).

La re-auditoría adversaria del plan fresco encontró que se entregaba solo 68% del target de
proteína (81g vs 119g) porque los fixes eran advisory, no restricción dura. FS6: el closer
rellena cada comida al target de proteína del slot con proteína de alta densidad allergen-safe
INTEGRADA (gramos, no nota), y el reconcile protein-preserving nivela kcal escalando SOLO
carbos/grasas (la proteína queda fija).

Cubre: (1) candidatos alta densidad allergen-safe, (2) closer rellena al target integrado,
(3) closer no-cook usa proteína segura (no carne cruda/huevo en batido), (4) closer respeta
alergias, (5) reconcile PRESERVA proteína + nivela kcal, (6) integración: día deficitario
→ día al target.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nutrition_db import IngredientNutritionDB
from graph_orchestrator import (
    _safe_high_density_proteins, _close_protein_gap_for_meal,
    _protein_preserving_day_reconcile, _meal_macro_num,
)

# Catálogo con nombres = los de DOMINICAN_PROTEINS para que _safe_high_density_proteins los vea.
_ROWS = [
    {"name": "Pollo", "aliases": ["pollo", "pechuga"], "kcal_per_100g": 165,
     "protein_g_per_100g": 31, "carbs_g_per_100g": 0, "fats_g_per_100g": 3.6},
    {"name": "Pescado", "aliases": ["pescado", "tilapia"], "kcal_per_100g": 128,
     "protein_g_per_100g": 26, "carbs_g_per_100g": 0, "fats_g_per_100g": 2.7},
    {"name": "Yogurt", "aliases": ["yogurt", "yogur griego", "yogurt griego"], "kcal_per_100g": 59,
     "protein_g_per_100g": 10, "carbs_g_per_100g": 3.6, "fats_g_per_100g": 0.4, "density_g_per_cup": 245},
    {"name": "Lentejas", "aliases": ["lentejas", "lenteja"], "kcal_per_100g": 116,
     "protein_g_per_100g": 9, "carbs_g_per_100g": 20, "fats_g_per_100g": 0.4, "density_g_per_cup": 198},
    {"name": "Camarones", "aliases": ["camarones", "camaron"], "kcal_per_100g": 99,
     "protein_g_per_100g": 24, "carbs_g_per_100g": 0.2, "fats_g_per_100g": 0.3},
    {"name": "Arroz blanco", "aliases": ["arroz", "arroz blanco"], "kcal_per_100g": 130,
     "protein_g_per_100g": 2.7, "carbs_g_per_100g": 28, "fats_g_per_100g": 0.3, "density_g_per_cup": 158},
    {"name": "Queso Mozzarella", "aliases": ["queso mozzarella", "mozzarella", "queso"], "kcal_per_100g": 280,
     "protein_g_per_100g": 22, "carbs_g_per_100g": 2.2, "fats_g_per_100g": 22},
]


def _db():
    return IngredientNutritionDB(rows=_ROWS)


# ---------- candidatos alta densidad ----------

def test_candidatos_alta_densidad_ordenados_por_magrez():
    cands = _safe_high_density_proteins(["Ninguna"], _db())
    names = [c[1] for c in cands]
    # Pollo/Pescado/Camarones/Yogurt no aparece yogurt (10g < 18 min) ni lentejas (9g) ni arroz
    assert "Pollo" in names and "Pescado" in names and "Camarones" in names
    assert "Lentejas" not in names and "Yogurt" not in names and "Arroz blanco" not in names
    # ordenado por proteína/kcal desc: camarones (24/99=0.24) > pescado (26/128=0.20) > pollo (31/165=0.19)
    assert names[0] == "Camarones"


def test_candidatos_respeta_alergia():
    # Alergia a mariscos → camarones excluido.
    cands = _safe_high_density_proteins(["mariscos"], _db())
    names = [c[1] for c in cands]
    assert "Camarones" not in names
    assert "Pollo" in names


# ---------- closer ----------

def test_closer_rellena_al_target_integrado():
    meal = {"name": "Lentejas Guisadas con Arroz", "protein": 18, "carbs": 70, "fats": 12, "cals": 460,
            "ingredients": ["1 taza de lentejas cocidas (198g)", "1 taza de arroz (158g)"]}
    cands = _safe_high_density_proteins(["Ninguna"], _db())
    added = _close_protein_gap_for_meal(meal, 42.0, _db(), cands, fill_pct=0.92)
    assert added > 0
    # proteína subió hacia ~0.92*42 = 38.6g
    assert meal["protein"] >= 36
    # el ingrediente proteico fue integrado a la lista (gramos), no solo una nota
    assert any("g de" in str(i) and ("camaron" in str(i).lower() or "pescado" in str(i).lower()
               or "pollo" in str(i).lower()) for i in meal["ingredients"])


def test_closer_no_cook_nunca_mete_carne_cruda():
    # Batido: NUNCA carne cruda/huevo crudo. Solo proteína no-cocción-safe (queso/yogur).
    meal = {"name": "Batido de Mango", "protein": 6, "carbs": 40, "fats": 2, "cals": 200,
            "ingredients": ["1 mango (200g)"]}
    cands = _safe_high_density_proteins(["Ninguna"], _db())  # pollo/pescado/camarones/queso
    _close_protein_gap_for_meal(meal, 18.0, _db(), cands, fill_pct=0.92)
    nuevo = " ".join(str(i) for i in meal["ingredients"]).lower()
    assert not any(t in nuevo for t in ("pollo", "camaron", "pescado", "huevo", "clara"))
    # si añadió algo, fue lácteo no-cocción-safe (queso)
    if len(meal["ingredients"]) > 1:
        assert "queso" in nuevo


def test_closer_no_cook_acepta_yogur_si_esta_en_candidatos():
    # Si el set incluye yogur (bajando min_protein), el batido SÍ lo acepta.
    db = _db()
    cands = _safe_high_density_proteins(["Ninguna"], db, min_protein=9.0)  # incluye yogur
    meal = {"name": "Batido Proteico", "protein": 6, "carbs": 40, "fats": 2, "cals": 200,
            "ingredients": ["1 mango (200g)"]}
    added = _close_protein_gap_for_meal(meal, 18.0, db, cands, fill_pct=0.92)
    assert added > 0
    assert any("yogur" in str(i).lower() for i in meal["ingredients"])


def test_closer_dish_fit_comida_ligera_evita_carne_exotica():
    # Desayuno (ligero): NO debe meter camarones; prefiere huevo/lácteo (queso).
    meal = {"name": "Revoltillo de Huevo con Avena", "protein": 12, "carbs": 60, "fats": 14,
            "cals": 12*4+60*4+14*9, "ingredients": ["2 huevos", "1 taza de avena"]}
    cands = _safe_high_density_proteins(["Ninguna"], _db())  # camarones es el más magro
    _close_protein_gap_for_meal(meal, 28.0, _db(), cands, fill_pct=0.92)
    nuevo = " ".join(str(i) for i in meal["ingredients"]).lower()
    assert "camaron" not in nuevo  # NO carne exótica en desayuno
    assert "queso" in nuevo  # eligió lácteo (dish-fit ligero)


def test_closer_congruencia_escala_proteina_del_plato():
    # Si el plato ya menciona la proteína (pollo), el closer la prefiere (escala el tema).
    meal = {"name": "Pollo Guisado con Arroz", "protein": 20, "carbs": 60, "fats": 12,
            "cals": 20*4+60*4+12*9, "ingredients": ["80g de pollo", "1 taza de arroz"]}
    cands = _safe_high_density_proteins(["Ninguna"], _db())
    _close_protein_gap_for_meal(meal, 42.0, _db(), cands, fill_pct=0.92)
    nuevo = " ".join(str(i) for i in meal["ingredients"]).lower()
    assert "pollo" in nuevo and "camaron" not in nuevo  # escaló pollo, no metió camarón


def test_closer_congruencia_por_token_con_queso():
    # "...con Queso..." debe escalar el queso (token), NO meter camarón ajeno.
    meal = {"name": "Revuelto de Huevos con Queso y Batata", "protein": 14, "carbs": 50, "fats": 16,
            "cals": 14*4+50*4+16*9, "ingredients": ["2 huevos", "50g de queso", "1 batata"]}
    cands = _safe_high_density_proteins(["Ninguna"], _db())  # camarones es el más magro
    _close_protein_gap_for_meal(meal, 28.0, _db(), cands, fill_pct=0.92)
    nuevo = " ".join(str(i) for i in meal["ingredients"]).lower()
    assert "queso" in nuevo and "camaron" not in nuevo


def test_closer_nota_sin_gramaje_hardcodeado():
    # La nota NO debe llevar un gramaje fijo (se desfasaría tras el trim); refiere al ingrediente.
    import re as _re
    meal = {"name": "Lentejas con Arroz", "protein": 18, "carbs": 70, "fats": 12, "cals": 460,
            "ingredients": ["1 taza de lentejas cocidas (198g)"], "recipe": ["Cocina."]}
    cands = _safe_high_density_proteins(["Ninguna"], _db())
    _close_protein_gap_for_meal(meal, 42.0, _db(), cands, fill_pct=0.92)
    nota = next((s for s in meal["recipe"] if "💪" in s), "")
    assert nota and "ingredientes" in nota
    assert not _re.search(r"\d+\s*g de", nota)  # sin "Xg de ..." en la nota


def test_closer_es_idempotente_no_re_anade_en_segunda_pasada():
    # Re-assemble (retries/surgical) re-ejecuta el solver block. El closer NO debe re-añadir
    # proteína (antes acumulaba "20g + 40g habichuelas" y duplicaba huevos → inflaba el día).
    meal = {"name": "Lentejas con Arroz", "protein": 18, "carbs": 70, "fats": 12, "cals": 460,
            "ingredients": ["1 taza de lentejas cocidas (198g)"]}
    cands = _safe_high_density_proteins(["Ninguna"], _db())
    a1 = _close_protein_gap_for_meal(meal, 42.0, _db(), cands, fill_pct=0.92)
    n_ings = len(meal["ingredients"])
    p_after = meal["protein"]
    a2 = _close_protein_gap_for_meal(meal, 42.0, _db(), cands, fill_pct=0.92)  # 2ª pasada
    assert a1 > 0 and a2 == 0  # la 2ª no añade
    assert len(meal["ingredients"]) == n_ings  # no duplica ingredientes
    assert meal["protein"] == p_after  # no infla la proteína


def test_closer_no_op_si_ya_alcanza_target():
    meal = {"name": "Pollo a la Plancha", "protein": 40, "carbs": 10, "fats": 8, "cals": 300,
            "ingredients": ["150g de pollo (150g)"]}
    cands = _safe_high_density_proteins(["Ninguna"], _db())
    assert _close_protein_gap_for_meal(meal, 42.0, _db(), cands, fill_pct=0.92) == 0


# ---------- reconcile protein-preserving ----------

def test_reconcile_preserva_proteina_y_nivela_kcal():
    # Día con kcal por ENCIMA del target (el closer añadió proteína) → reconcile baja
    # carbos/grasas, NO la proteína.
    meals = [
        {"name": "A", "protein": 40, "carbs": 60, "fats": 20, "cals": 40*4+60*4+20*9,
         "ingredients": ["120g de pollo (120g)", "1 taza de arroz (158g)"]},
        {"name": "B", "protein": 38, "carbs": 80, "fats": 18, "cals": 38*4+80*4+18*9,
         "ingredients": ["100g de pescado (100g)", "1.5 taza de arroz (237g)"]},
    ]
    P0 = sum(_meal_macro_num(m["protein"]) for m in meals)
    target = 1800.0
    ok = _protein_preserving_day_reconcile(meals, target, _db())
    assert ok
    P1 = sum(_meal_macro_num(m["protein"]) for m in meals)
    cals = sum(_meal_macro_num(m["cals"]) for m in meals)
    assert P1 == P0  # proteína INTACTA
    assert abs(cals - target) < 60  # kcal niveladas al target


def test_integracion_dia_deficitario_llega_al_target():
    # Día tipo auditado: comidas base bajas en proteína. Tras closer (a cada slot) +
    # reconcile, el día debe acercarse al target diario de proteína.
    db = _db()
    cands = _safe_high_density_proteins(["Ninguna"], db)
    daily_p, daily_cals = 119.0, 2200.0
    fracs = [0.20, 0.35, 0.15, 0.30]
    meals = [
        {"name": "Avena con Fruta", "protein": 10, "carbs": 60, "fats": 10, "cals": 10*4+60*4+10*9,
         "ingredients": ["1 taza de avena", "1 mango"]},
        {"name": "Lentejas con Arroz", "protein": 18, "carbs": 90, "fats": 14, "cals": 18*4+90*4+14*9,
         "ingredients": ["1 taza de lentejas cocidas (198g)", "1 taza de arroz (158g)"]},
        {"name": "Fruta con Almendras", "protein": 8, "carbs": 50, "fats": 16, "cals": 8*4+50*4+16*9,
         "ingredients": ["2 taza de fresas", "almendras"]},
        {"name": "Yuca con Ensalada", "protein": 9, "carbs": 70, "fats": 12, "cals": 9*4+70*4+12*9,
         "ingredients": ["1 taza de yuca", "ensalada"]},
    ]
    for m, fr in zip(meals, fracs):
        _close_protein_gap_for_meal(m, daily_p * fr, db, cands, fill_pct=0.92)
    _protein_preserving_day_reconcile(meals, daily_cals, db)
    P = sum(_meal_macro_num(m["protein"]) for m in meals)
    cals = sum(_meal_macro_num(m["cals"]) for m in meals)
    # antes era ~45g (10+18+8+9); ahora debe superar el 85% del target de 119g
    assert P >= 0.85 * daily_p, f"proteína {P} < 85% de {daily_p}"
    assert abs(cals - daily_cals) < 120  # kcal cerca del target


def test_trim_techo_baja_proteina_sobre_target():
    from graph_orchestrator import _trim_day_protein_to_ceiling
    # Día sobre-producido: 180g vs target 119g → trim al target.
    meals = [
        {"name": "A", "protein": 60, "carbs": 40, "fats": 15, "cals": 60*4+40*4+15*9,
         "ingredients": ["200g de pollo (200g)", "1 taza de arroz (158g)"]},
        {"name": "B", "protein": 60, "carbs": 50, "fats": 12, "cals": 60*4+50*4+12*9,
         "ingredients": ["230g de pescado (230g)", "1 taza de arroz (158g)"]},
        {"name": "C", "protein": 60, "carbs": 30, "fats": 10, "cals": 60*4+30*4+10*9,
         "ingredients": ["250g de camarones (250g)", "ensalada"]},
    ]
    trimmed = _trim_day_protein_to_ceiling(meals, 119.0, _db(), ceiling_pct=1.12)
    assert trimmed
    P = sum(_meal_macro_num(m["protein"]) for m in meals)
    assert abs(P - 119) <= 6  # bajó al target


def test_trim_no_op_dentro_de_la_banda():
    from graph_orchestrator import _trim_day_protein_to_ceiling
    meals = [{"name": "A", "protein": 119, "carbs": 100, "fats": 30, "cals": 1500,
              "ingredients": ["180g de pollo (180g)"]}]
    assert _trim_day_protein_to_ceiling(meals, 119.0, _db(), ceiling_pct=1.12) is False


def test_regla_piso_proteina_presente_en_prompt():
    # Ancla la regla de anclaje (A) en el prompt: un borrado la haría fallar aquí.
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "prompts", "preferences.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    assert "REGLA DE PISO DE PROTEÍNA" in src
    assert "alta densidad" in src.lower()
    assert "90%" in src  # umbral de rechazo explícito
