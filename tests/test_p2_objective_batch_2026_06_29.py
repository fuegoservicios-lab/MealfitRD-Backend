"""[P2-OBJECTIVE-BATCH · 2026-06-29] Test-ancla del batch de gaps P2 del audit-objetivo de plan-gen.

  P2-3  P2-SLOT-RICE-SYNONYMS        — el gate de cena/desayuno caza arroz nombrado distinto (chofán/paella/congrí…).
  P2-5  P2-QTY-PRESENCE-GUARD        — alimento verificado sin cantidad → porción default + degradado; exime sazonadores.
  P2-6  P2-RECIPE-REVERSE-COHERENCE  — ingrediente listado pero no usado en pasos → paso de uso añadido (idempotente).
  P2-7  P2-RECIPE-STEP-CONTRACT      — contrato de pasos (tiempo/temp, sustantivo) en form-gen + update prompts.
  P2-8  P2-NAME-HONESTY-DEGRADED     — nombre cárnico fantasma sin carne real → flag observable (no rename mentiroso).
  P2-9  P2-CHATMODIFY-CLASH-RETRY    — chat-modify presiona retry ante pareo fruta+salado no pedido.
  P2-10 P2-CHATMODIFY-CROSS-DAY-VARIETY — chat-modify siembra proteínas de otros días (advisory).

P2-1 (saturación→carb-closer) queda RESUELTO por P1-3 (carb-closer ON, granularidad día); P2-2 (platos compuestos
0-silencioso) es data-deferred (el gate nunca dispara a cobertura actual + verified-only/regla-8 mitigan); P2-4
(arroz-de-noche) cubierto por la detección de P2-3 + autofix existente + decisión de producto del owner.
"""
from pathlib import Path
from types import SimpleNamespace

import graph_orchestrator as g
from constants import strip_accents, slot_violations_for_meal_name

_BACKEND = Path(__file__).resolve().parent.parent
_TOOLS = (_BACKEND / "tools.py").read_text(encoding="utf-8")
_DAYGEN = (_BACKEND / "prompts" / "day_generator.py").read_text(encoding="utf-8")
_MEALOPS = (_BACKEND / "prompts" / "meal_operations.py").read_text(encoding="utf-8")


class _MockDB:
    """db mínima para P2-5: solo `lookup` (el truth-up que falla queda atrapado por el try del helper)."""
    def lookup(self, name):
        n = strip_accents((name or "").lower())
        if "pollo" in n or "pechuga" in n:
            return SimpleNamespace(name="Pechuga de pollo", protein=31.0, carbs=0.0, fats=3.6)
        if "arroz" in n:
            return SimpleNamespace(name="Arroz", protein=2.7, carbs=28.0, fats=0.3)
        if "aceite" in n:
            return SimpleNamespace(name="Aceite de oliva", protein=0.0, carbs=0.0, fats=100.0)
        return None


# ───────────────────────── P2-3: sinónimos de arroz en el gate ─────────────────────────
def test_p2_3_rice_synonyms_caught_in_cena():
    for name in ("Chofán de Pollo", "Paella Criolla", "Congrí", "Risotto de Hongos", "Mamposteao"):
        assert slot_violations_for_meal_name(name, "cena"), f"'{name}' debería violar el slot cena"
    # falso-positivo guard: un plato sin arroz NO se marca
    assert not slot_violations_for_meal_name("Pescado al Horno con Batata", "cena")
    assert "P2-SLOT-RICE-SYNONYMS" in (_BACKEND / "constants.py").read_text(encoding="utf-8")


# ───────────────────────── P2-5: garantía de cantidad ─────────────────────────
def test_p2_5_injects_default_portion_for_bare_verified_food():
    meal = {"name": "Pollo con Arroz",
            "ingredients": ["Pollo", "Arroz", "Sal", "Aceite de oliva al gusto"],
            "ingredients_raw": ["Pollo", "Arroz", "Sal", "Aceite de oliva al gusto"]}
    n = g._ensure_ingredient_quantities(meal, _MockDB())
    assert n == 2, "debe corregir Pollo + Arroz (no Sal sazonador, no 'al gusto')"
    assert meal["ingredients"][0] == "120 g de Pollo"      # proteína-dominante
    assert meal["ingredients"][1].startswith("90 g")        # carbo
    assert meal["ingredients"][2] == "Sal"                  # sazonador intacto
    assert "al gusto" in meal["ingredients"][3]             # condimento intacto
    assert meal["ingredients_raw"][0] == "120 g de Pollo"   # raw sincronizado
    assert meal.get("_dish_quality_degraded") is True
    # idempotente
    assert g._ensure_ingredient_quantities(meal, _MockDB()) == 0


def test_p2_5_knob_on():
    assert g.QTY_PRESENCE_GUARD_ENABLED is True


def test_p2_5_assemble_callsite_constructs_its_own_db():
    # regresión del NameError visto en prod (corr=e4f3febe 2026-06-30): assemble NO expone `db`/`_db` en ese
    # scope (el motor construye la suya después) → el callsite DEBE construir su propia IngredientNutritionDB.
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("P2-QTY-PRESENCE-GUARD] {_qg}")          # ancla al callsite de assemble (no el del finalizer)
    block = src[i - 600:i]
    assert "_QGDB()" in block or "IngredientNutritionDB()" in block, \
        "el callsite de assemble debe construir su propio db (assemble no expone db en ese scope)"
    assert "_ensure_ingredient_quantities(_m, db)" not in src, "no usar un `db` indefinido en assemble (NameError)"


# ───────────────────────── P2-6: coherencia inversa ─────────────────────────
def test_p2_6_adds_step_for_unused_ingredient():
    meal = {"name": "Bowl de Pollo",
            "ingredients": ["150 g de Pechuga de pollo", "100 g de Aguacate"],
            "recipe": ["Mise en place: corta el pollo en cubos",
                       "El Toque de Fuego: saltea el pollo 8 min a fuego medio",
                       "Montaje: sirve el pollo en el bowl"]}
    n = g._ensure_ingredients_used_in_recipe(meal)
    assert n == 1, "el aguacate está listado pero nunca usado en los pasos"
    assert any("aguacate" in strip_accents(str(s).lower()) for s in meal["recipe"])
    # idempotente: tras añadirlo ya aparece
    assert g._ensure_ingredients_used_in_recipe(meal) == 0


def test_p2_6_noop_when_all_used():
    meal = {"name": "Pollo a la Plancha",
            "ingredients": ["150 g de Pechuga de pollo"],
            "recipe": ["Mise en place: sazona el pollo", "El Toque de Fuego: cocina el pollo 10 min"]}
    assert g._ensure_ingredients_used_in_recipe(meal) == 0
    assert g.RECIPE_REVERSE_COHERENCE_ENABLED is True


# ───────────────────────── P2-7: contrato de pasos ─────────────────────────
def test_p2_7_recipe_step_contract_markers():
    assert "P2-RECIPE-STEP-CONTRACT" in _DAYGEN
    assert _MEALOPS.count("P2-RECIPE-STEP-CONTRACT") >= 2  # swap + modify
    for src in (_DAYGEN, _MEALOPS):
        low = src.lower()
        assert "temperatura" in low or "180" in low or "fuego" in low


# ───────────────────────── P2-8: honestidad de nombre ─────────────────────────
def test_p2_8_flags_phantom_dairy_name():
    meal = {"name": "Pollo a la Plancha", "ingredients": ["200 g de Yogur griego", "100 g de Ñame"]}
    assert g._fix_phantom_protein_in_name(meal, strip_accents) is False  # no hay carne real → no renombra
    assert meal.get("_name_honesty_degraded") is True                     # pero marca el nombre como no-honesto
    assert g.PHANTOM_PROTEIN_DEGRADE_FLAG is True


def test_p2_8_noop_when_protein_real():
    meal = {"name": "Pollo a la Plancha", "ingredients": ["150 g de Pechuga de pollo"]}
    g._fix_phantom_protein_in_name(meal, strip_accents)
    assert meal.get("_name_honesty_degraded") is not True


# ───────────────────────── P2-9 / P2-10: chat-modify (parser) ─────────────────────────
def test_p2_9_chatmodify_clash_retry_present():
    assert "P2-CHATMODIFY-CLASH-RETRY" in _TOOLS
    assert "_meal_has_sweet_savory_clash" in _TOOLS
    assert 'raise ValueError("SWEET_SAVORY_CLASH no pedido en modify")' in _TOOLS


def test_p2_10_chatmodify_cross_day_variety_present():
    assert "P2-CHATMODIFY-CROSS-DAY-VARIETY" in _TOOLS
    assert "MEALFIT_UPDATE_CROSS_DAY_VARIETY" in _TOOLS
