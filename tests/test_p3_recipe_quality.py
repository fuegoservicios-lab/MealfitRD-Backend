"""[P3-RECIPE-QUALITY · 2026-06-28] El usuario: "las recetas son muy malas" (capturas de plan bariátrico) — se
sentían ensambladas por el solver: fracciones absurdas (2.5 huevos, 0.5 rebanada, 3.33 taza lechuga), proteína
PEGADA de relleno (10g camarón, chivo en yogurt) con paso robótico, cantidades triviales. Auditoría vía workflow
(3 auditores + verificación adversaria que corrigió la propuesta sobre-diseñada "quantize-first"). 6 fixes:

  Fix 1 (P3-HUMAN-WHOLE-DISCRETE): huevo/pan/rebanada/tostada/galleta → ENTERO (nunca 0.5/2.5); batata/guineo/
        aguacate/tomate/cebolla → siguen partiéndose en ½.
  Fix 2 (P3-HUMAN-LEAF-CUP): hojas (lechuga/espinaca) en taza SIN tercios (3.33 → 3.5); arroz/avena mantienen ⅓.
  Fix 3 (P3-PROTEIN-CLOSER-MIN-THRESHOLD): el closer NO pega proteína trivial (<20g alimento / <6g proteína).
  Fix 5 (P3-SWEET-GUARD-INGREDIENTS): el sweet-guard mira INGREDIENTES, no solo el nombre → 'Yogurt con Lechosa
        y Chivo' (nombre neutro) ahora se detecta dulce y bloquea la carne.
  Fix 6 (P3-CLOSER-RECIPE-INTEGRATE): paso de receta natural, no el robótico "incorpora el X ... para reforzar".
  Fix 7 (P3-PROTEIN-INTEGRATED): reglas en el prompt del generador.
"""
from __future__ import annotations

from pathlib import Path

from nutrition_db import quantize_ingredient_string as _q
import graph_orchestrator as g
from constants import strip_accents as _sa

_BACKEND = Path(g.__file__).resolve().parent


# ---- Fix 1 ----
def test_eggs_whole_no_half():
    out, _ = _q("2.5 huevos enteros")
    assert "2.5" not in out and "0.5" not in out
    assert out.startswith("2 ") or out.startswith("3 ")


def test_bread_slice_whole():
    out, _ = _q("0.5 rebanada de pan integral (15g)")
    assert out.startswith("1 ")


def test_divisible_keeps_half():
    for s in ("0.5 guineo mediano (40g)", "0.5 aguacate", "0.5 batata mediana"):
        out, _ = _q(s)
        assert out.startswith("0.5 "), out


# ---- Fix 2 ----
def test_leaf_cup_no_thirds():
    out, _ = _q("3.33 taza de lechuga romana picada (167g)")
    assert "3.33" not in out and "1/3" not in out.lower()


def test_grain_cup_keeps_thirds():
    out, _ = _q("1.33 taza de arroz integral")
    assert "1.33" in out or "1/3" in out  # arroz conserva tercios (1/3 taza es real)


# ---- Fix 3 ----
class _Info:
    def __init__(self, n, p, c=1.0, f=1.0, k=99.0):
        self.name, self.protein, self.carbs, self.fats, self.kcal = n, p, c, f, k


def _meal(cals=150, prot=5):
    return {"name": "Almuerzo", "protein": prot, "carbs": 10, "fats": 5, "cals": cals, "ingredients": ["base"]}


def test_closer_skips_trivial_protein():
    cands = [(0.2, "Camarones", _Info("Camarones", 20))]
    assert g._close_protein_gap_for_meal(_meal(), 7, None, cands) == 0  # gap ~2g prot → trivial → no pega


def test_closer_adds_real_protein():
    cands = [(0.2, "Camarones", _Info("Camarones", 20))]
    assert g._close_protein_gap_for_meal(_meal(), 30, None, cands) >= 20  # gap real → añade ≥ umbral


def test_closer_min_knobs_present():
    assert hasattr(g, "PROTEIN_CLOSER_MIN_GRAMS") and hasattr(g, "PROTEIN_CLOSER_MIN_PROTEIN_G")


# ---- Fix 5 ----
def test_sweet_guard_by_ingredients():
    m = {"name": "Merienda Proteica", "ingredients": ["25g de yogurt griego", "95g de lechosa en cubos", "linaza"]}
    assert g._is_sweet_meal(m, _sa) is True  # nombre neutro pero ingredientes dulces


def test_sweet_guard_savory_salad_false():
    m = {"name": "Ensalada de Pollo", "ingredients": ["pollo", "lechuga", "tomate"]}
    assert g._is_sweet_meal(m, _sa) is False


def test_chivo_blocked_in_yogurt_by_ingredients():
    m = {"name": "Merienda Proteica", "protein": 5, "carbs": 20, "fats": 3, "cals": 150,
         "ingredients": ["25g de yogurt griego", "95g de lechosa en cubos"]}
    cands = [(0.2, "Chivo", _Info("Chivo", 27))]
    assert g._close_protein_gap_for_meal(m, 25, None, cands) == 0  # dulce por ingredientes → no mete chivo


# ---- Fix 6 ----
def test_recipe_step_not_robotic():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "P3-CLOSER-RECIPE-INTEGRATE" in src
    assert "indicado en los ingredientes para reforzar la proteína" not in src  # frase robótica eliminada


# ---- Fix 7 ----
def test_prompt_human_portions_rule():
    src = (_BACKEND / "prompts" / "day_generator.py").read_text(encoding="utf-8")
    assert "P3-HUMAN-WHOLE-DISCRETE" in src and "P3-PROTEIN-INTEGRATED" in src


# ---- knobs/anchors ----
def test_quantize_knobs_anchor():
    src = (_BACKEND / "nutrition_db.py").read_text(encoding="utf-8")
    assert "P3-HUMAN-WHOLE-DISCRETE" in src and "P3-HUMAN-LEAF-CUP" in src
    assert "_WHOLE_ONLY_TOKENS" in src and "_LEAF_TOKENS" in src
