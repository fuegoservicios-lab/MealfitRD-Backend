"""[P2-FAT-LEAN-SWAP + P2-FAT-DEADZONE · 2026-07-05] (audit solver+seeder P2-1)

Zona muerta del rebalance de grasas: cuando la grasa NO-movible (embebida en líneas
proteína-dominantes: salmón, res 80/20, muslo/piel) ya excede el target del día,
`_rebalance_day_macros_to_target._one("fats")` retornaba en SILENCIO — cero palanca, cero señal.

Dos capas:
  1. Telemetría P2-FAT-DEADZONE (siempre ON): log ⚠ una vez por invocación cuando grasas > target
     con 0g movibles o con no-movible ≥ target. Serie para decidir el flip del lever.
  2. Lever `_swap_fat_dense_protein_to_lean_for_day` (OFF por default — cambia la identidad del
     plato): proteína grasa → forma magra de la MISMA especie (allergen-neutral por construcción),
     gramos intactos, nombre/pasos/raw reescritos, delta honesto + truth-up, marker
     `_protein_autofix_applied` (fidelity-discount lo reconoce). Jamás sobre-corrige (solo swapea
     si ACERCA al target).
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_GO = (_REPO_ROOT / "backend" / "graph_orchestrator.py").read_text(encoding="utf-8")


# ───────────────────────── parser-based ─────────────────────────

def test_knob_defined_default_off():
    m = re.search(r'FAT_LEAN_SWAP_ENABLED\s*=\s*_env_bool\(\s*"MEALFIT_FAT_LEAN_SWAP"\s*,\s*(\w+)\)', _GO)
    assert m and m.group(1) == "False", "el lever cambia identidad de plato → nace OFF (A/B antes de flip)"


def test_deadzone_telemetry_in_both_returns():
    """La telemetría vive en AMBOS returns silenciosos de `_one` (movable=0 y desired<=0)."""
    i = _GO.index("def _rebalance_day_macros_to_target")
    body = _GO[i:i + 7000]
    assert body.count("P2-FAT-DEADZONE") >= 2, "ambas ramas de la zona muerta deben loguear"
    assert "_fat_dz_logged" in body, "guard de una-línea-por-invocación (3 pases)"


def test_wired_in_engine_before_rebalance_gated():
    i_swap = _GO.index("[P2-FAT-LEAN-SWAP · 2026-07-05] (audit solver+seeder P2-1) Lever OFF-default")
    i_reb = _GO.index("[P3-MACRO-REBALANCE · 2026-06-19] Tras swap + cuantización")
    assert i_swap < i_reb, "el swap corre ANTES del rebalance (el motor dimensiona la proteína nueva)"
    win = _GO[i_swap:i_swap + 1200]
    assert "if FAT_LEAN_SWAP_ENABLED and _fg:" in win


# ───────────────────────── funcional ─────────────────────────

def _db():
    from nutrition_db import IngredientNutritionDB
    return IngredientNutritionDB(rows=[
        {"name": "Salmón", "aliases": ["salmon"], "kcal_per_100g": 208,
         "protein_g_per_100g": 20, "carbs_g_per_100g": 0, "fats_g_per_100g": 13},
        {"name": "Filete de pescado blanco", "aliases": ["pescado blanco", "pescado"],
         "kcal_per_100g": 82, "protein_g_per_100g": 18, "carbs_g_per_100g": 0, "fats_g_per_100g": 1.5},
        {"name": "Arroz blanco", "aliases": ["arroz"], "kcal_per_100g": 130,
         "protein_g_per_100g": 2.7, "carbs_g_per_100g": 28, "fats_g_per_100g": 0.3},
    ])


def _meals_salmon():
    # salmón 300g = 39g de grasa (proteína-dominante: 4·60=240 > 9·39=351? no: 351>240 → grasa-
    # dominante... el salmón real es P-dominante; para el TEST lo que importa es la mecánica del
    # swap por token, no el grupo). fats del día = 40.
    return [{"meal": "Cena", "name": "Salmón a la Plancha con Arroz",
             "protein": 62, "carbs": 42, "fats": 40, "cals": 950,
             "ingredients": ["300g de salmón", "150g de arroz blanco"],
             "ingredients_raw": ["300g de salmón", "150g de arroz blanco"],
             "recipe": ["Mise en place: seca el salmón.",
                        "El Toque de Fuego: cocina el salmón 8-10 min a fuego medio.",
                        "Montaje: sirve con el arroz."]}]


def test_swap_rewrites_line_raw_name_steps_and_macros():
    import graph_orchestrator as go
    meals = _meals_salmon()
    n = go._swap_fat_dense_protein_to_lean_for_day(meals, 12.0, _db(), {}, tol=0.10)
    assert n == 1
    m = meals[0]
    joined = " ".join(m["ingredients"]).lower()
    assert "salmón" not in joined and "salmon" not in joined
    assert "filete de pescado blanco" in joined
    assert "300g" in " ".join(m["ingredients"]), "los gramos quedan intactos (misma porción)"
    assert "filete de pescado blanco" in " ".join(m["ingredients_raw"]).lower(), "raw en lockstep"
    assert "salm" not in go_strip(m["name"]).lower(), f"el nombre debe reflejar el swap: {m['name']}"
    assert "salm" not in " ".join(m["recipe"]).lower(), "los pasos también se reescriben"
    assert str(m.get("_protein_autofix_applied", "")).startswith("salmon->")
    assert go._meal_macro_num(m.get("fats")) < 40, "la grasa del meal baja con el delta honesto"


def go_strip(s):
    from constants import strip_accents
    return strip_accents(str(s))


def test_no_swap_when_within_band():
    import graph_orchestrator as go
    meals = _meals_salmon()
    meals[0]["fats"] = 41
    assert go._swap_fat_dense_protein_to_lean_for_day(meals, 40.0, _db(), {}, tol=0.10) == 0
    assert "300g de salmón" in meals[0]["ingredients"]


def test_no_overcorrection_below_band():
    """Día apenas sobre banda: el swap completo lo dejaría MÁS lejos (por debajo) → no swapea."""
    import graph_orchestrator as go
    meals = _meals_salmon()
    # cur=40, target=36 (11% sobre): delta del swap ≈ 34.5g → after ≈ 5.5 (|5.5-36|=30.5 > |40-36|=4)
    assert go._swap_fat_dense_protein_to_lean_for_day(meals, 36.0, _db(), {}, tol=0.10) == 0
    assert "300g de salmón" in meals[0]["ingredients"], "jamás sobre-corregir bajo la banda"


def test_dislike_on_replacement_blocks_swap():
    import graph_orchestrator as go
    meals = _meals_salmon()
    n = go._swap_fat_dense_protein_to_lean_for_day(
        meals, 12.0, _db(), {"dislikes": ["pescado blanco"]}, tol=0.10)
    assert n == 0
    assert "300g de salmón" in meals[0]["ingredients"]
