"""[P1-RECIPE-QUALITY-BATCH-2 · 2026-07-07] Cierre de los 3 diferidos del review visual del plan
766893f4 (30 días, ganancia muscular, generación en chunks). El forense SQL sobre el plan vivo reveló
que los 3 síntomas son una MISMA raíz + 2 residuos:

- P1-CHUNK-FINALIZE-PARITY (raíz): `finalize_plan_data_coherence` (el finalizer defensivo que procesa
  los días chunk/partial/degradado) corría `_cap_unrealistic_portions` pero NO `_relevel_fats_universal`
  ni `_cap_cheese_dumps_final` → day2 llegó a F141 total (58 target) con "Tostadas PB+Mango+Queso" 67g
  grasa SIN capear (el cheese-cap del batch previo solo vivía en assemble). Fix: paridad en el finalizer
  con target_fats derivado por el caller (chunk worker + shield pre-INSERT).
- P1-COUNT-UNICODE-FRAC (#3 avocado): "1½ aguacates (221 g)" evadía el count-cap `aguacate:1.0` porque
  el regex exigía espacio tras el dígito; la fracción unicode lo rompía. Fix: lead unicode-aware.
- P1-RAW-DISPLAY-RECONCILE-RECIPROCAL (#1 ajo): raw tenía "0.5 diente de ajo" dropeado del display.
- P3-9 (#2 seed-note): la nota 🌱 congelaba la qty del seed → triple divergencia. Fix: la nota nombra
  el alimento, la qty vive en la línea de ingrediente.
"""
import os
import re

import graph_orchestrator as g

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()
with open(os.path.join(_BACKEND, "cron_tasks.py"), encoding="utf-8") as f:
    _CRON = f.read()
with open(os.path.join(_BACKEND, "db_plans.py"), encoding="utf-8") as f:
    _DBP = f.read()


class _DB:
    """Fake DB: cheese fat-dominante, aguacate graso, huevo, resto neutro."""
    def macros_from_ingredient_string(self, s):
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)", str(s))
        gr = float(m.group(1).replace(",", ".")) if m else 0.0
        low = str(s).lower()
        if "queso" in low and "yogur" not in low:
            return {"protein": gr * 0.11, "carbs": gr * 0.03, "fats": gr * 0.30, "kcal": gr * 3.5}
        if "aguacate" in low:  # count-based ("1 aguacate") → gr≈1; grasa por unidad
            return {"protein": gr * 2.0, "carbs": gr * 1.0, "fats": gr * 15.0, "kcal": gr * 160.0}
        return {"protein": 0.0, "carbs": 0.0, "fats": 0.0, "kcal": 0.0}

    def micros_from_ingredient_string(self, s):
        return {}

    def _ingredient_macro_group(self, *a, **k):
        return None


# ───────────────────────── FIX A: chunk-finalize parity ─────────────────────────

def test_finalizer_signature_has_target_fats():
    assert "def finalize_plan_data_coherence(days: list, db=None, allergies=None, target_fats=None)" in _GO


def test_finalizer_wires_relevel_and_cheese_final():
    assert "P1-CHUNK-FINALIZE-PARITY" in _GO
    i = _GO.find("P1-CHUNK-FINALIZE-PARITY · 2026-07-07")
    assert i != -1
    seg = _GO[i:i + 1900]
    assert "_relevel_fats_universal(days, target_fats" in seg, "relevel gated por target_fats en el finalizer"
    assert "_cap_cheese_dumps_final(days" in seg, "cheese-final en el finalizer"
    assert "FATS_RELEVEL_UNIVERSAL_ENABLED and target_fats" in seg, "relevel skip sin target (fail-safe)"


def test_chunk_worker_derives_and_passes_target_fats():
    # el chunk worker (cron_tasks) deriva el target de plan_data['macros']['fats'] y lo pasa
    assert "P1-CHUNK-FINALIZE-PARITY" in _CRON
    assert "_fpc_chunk(plan_data[\"days\"], target_fats=_tf_ck)" in _CRON
    i = _CRON.find("_fpc_chunk(plan_data[\"days\"], target_fats=_tf_ck)")
    seg = _CRON[max(0, i - 600):i]
    assert '(plan_data.get("macros") or {}).get("fats")' in seg


def test_insert_shield_derives_and_passes_target_fats():
    # el shield pre-INSERT (db_plans) también deriva y pasa target_fats
    assert "_fpc(_pd[\"days\"], target_fats=_tf_ins)" in _DBP
    i = _DBP.find("_fpc(_pd[\"days\"], target_fats=_tf_ins)")
    seg = _DBP[max(0, i - 500):i]
    assert '(_pd.get("macros") or {}).get("fats")' in seg


def test_finalizer_caps_cheese_dump_end_to_end(monkeypatch):
    """Funcional: un 'Desayuno' dulce con 160g de queso pasado por el finalizer → capeado a 120g
    (MERIENDA, sweet-aware) por el cheese-final AHORA cableado en el path defensivo."""
    monkeypatch.setattr(g, "CHEESE_DUMP_FINAL_ENABLED", True)
    meal = {"name": "Tostadas de Mantequilla de Maní y Mango con Queso", "meal": "Desayuno",
            "ingredients": ["160 g de queso", "2 tortillas de trigo"],
            "ingredients_raw": ["160 g de queso", "2 tortillas de trigo"],
            "recipe": ["Mise en place: pesa el queso.", "Montaje: sirve."],
            "protein": 40, "carbs": 60, "fats": 55, "cals": 895}
    days = [{"meals": [meal]}]
    g.finalize_plan_data_coherence(days, _DB(), target_fats=58.0)
    assert meal["ingredients"][0].startswith("120"), meal["ingredients"][0]
    assert meal["ingredients_raw"][0].startswith("120")


# ───────────────────────── FIX B: unicode count parser (#3 avocado) ─────────────────────────

def test_avocado_unicode_fraction_capped():
    """'1½ aguacates (221 g)' → capeado a 1 aguacate (el count-cap aguacate=1.0 ahora ve la fracción)."""
    meal = {"name": "Bowl de Soya estilo Ceviche", "meal": "Almuerzo",
            "ingredients": ["65 g de soya texturizada", "1½ aguacates (221 g)"],
            "ingredients_raw": ["65 g de soya texturizada", "1.5 aguacate (221 g)"],
            "protein": 41, "carbs": 83, "fats": 8, "cals": 576}
    n = g._cap_unrealistic_portions([{"meals": [meal]}], db=_DB())
    assert n >= 1
    assert meal["ingredients"][1].startswith("1 aguacate"), meal["ingredients"][1]
    assert meal["ingredients_raw"][1].startswith("1 aguacate")  # lockstep raw


def test_egg_unicode_fraction_capped():
    """'4½ huevos' → 4 (cap huevo=4.0). Antes '4½' evadía el parser."""
    meal = {"name": "Revoltillo", "meal": "Desayuno",
            "ingredients": ["4½ huevos", "1 tomate"], "ingredients_raw": ["4½ huevos", "1 tomate"],
            "protein": 30, "carbs": 10, "fats": 20, "cals": 340}
    g._cap_unrealistic_portions([{"meals": [meal]}], db=_DB())
    assert meal["ingredients"][0].startswith("4 huevo"), meal["ingredients"][0]


def test_integer_single_space_count_not_broken(monkeypatch):
    """Regresión: '1 tomate' / '2 cebolla' (entero + 1 espacio) siguen parseando y NO se capean
    de más (el `\\s*` pre-fracción no debe comerse el único espacio)."""
    meal = {"name": "Ensalada", "meal": "Almuerzo",
            "ingredients": ["1 tomate mediano (100 g)", "2 cebolla morada (199 g)"],
            "ingredients_raw": ["1 tomate mediano (100 g)", "2 cebolla morada (199 g)"],
            "protein": 5, "carbs": 20, "fats": 2, "cals": 118}
    g._cap_unrealistic_portions([{"meals": [meal]}], db=_DB())
    assert meal["ingredients"][0].startswith("1 tomate"), "1 tomate ≤ cap 3 → intacto"
    assert meal["ingredients"][1].startswith("2 cebolla"), "2 cebolla ≤ cap 2 → intacto"


def test_count_lead_regex_unit_lines_ignored():
    """Líneas en gramos ('221 g de aguacate') → noun 'g' (no capeado); sin nº → None."""
    R = g._REALISM_COUNT_LEAD_RE
    m = R.match("221 g de aguacate")
    assert m and m.group(3) == "g"  # 'g' no es noun capeado → inocuo
    assert R.match("sal al gusto").group(3) == "sal"  # matchea pero el guard (grp1|grp2) lo salta
    m2 = R.match("sal al gusto")
    assert not (m2.group(1) or m2.group(2))  # sin nº ni fracción → no se procesa


# ───────────────────────── FIX C: reciprocal reconcile (#1 ajo) ─────────────────────────

def test_reciprocal_reconcile_restores_ajo_to_display(monkeypatch):
    monkeypatch.setattr(g, "RAW_DISPLAY_RECONCILE_RECIPROCAL_ENABLED", True)
    meal = {"name": "Puré de Sardinas",
            "ingredients": ["½ lata de sardinas", "½ cebolla picada (52g)", "⅔ taza de arroz"],
            "ingredients_raw": ["0.5 lata de sardinas", "0.5 cebolla picada (52g)",
                                "0.5 diente de ajo picado", "0.67 taza de arroz"]}
    n = g._reconcile_raw_missing_in_display([{"meals": [meal]}])
    assert n == 1, f"esperada 1 línea (ajo) restaurada, dio {n}"
    joined = " ".join(meal["ingredients"]).lower()
    assert "ajo" in joined, "el ajo del raw ahora aparece en el display"


def test_reciprocal_reconcile_idempotent(monkeypatch):
    monkeypatch.setattr(g, "RAW_DISPLAY_RECONCILE_RECIPROCAL_ENABLED", True)
    meal = {"name": "X", "ingredients": ["½ cebolla"],
            "ingredients_raw": ["0.5 cebolla", "0.5 diente de ajo picado"]}
    assert g._reconcile_raw_missing_in_display([{"meals": [meal]}]) == 1
    assert g._reconcile_raw_missing_in_display([{"meals": [meal]}]) == 0  # 2ª pasada = no-op


def test_reciprocal_reconcile_noop_when_aligned(monkeypatch):
    monkeypatch.setattr(g, "RAW_DISPLAY_RECONCILE_RECIPROCAL_ENABLED", True)
    meal = {"name": "X", "ingredients": ["1 pechuga de pollo", "80 g de arroz"],
            "ingredients_raw": ["1 pechuga de pollo", "80 g de arroz"]}
    assert g._reconcile_raw_missing_in_display([{"meals": [meal]}]) == 0


def test_reciprocal_reconcile_knob_off(monkeypatch):
    monkeypatch.setattr(g, "RAW_DISPLAY_RECONCILE_RECIPROCAL_ENABLED", False)
    meal = {"name": "X", "ingredients": ["½ cebolla"],
            "ingredients_raw": ["0.5 cebolla", "0.5 diente de ajo picado"]}
    assert g._reconcile_raw_missing_in_display([{"meals": [meal]}]) == 0
    assert len(meal["ingredients"]) == 1  # intacto


def test_reciprocal_reconcile_wired_both_surfaces():
    assert _GO.count("_reconcile_raw_missing_in_display(days)") >= 2  # finalizer + assemble


# ───────────────────────── FIX D: seed-note qty strip (#2) ─────────────────────────

def test_seed_note_uses_food_not_qty():
    assert "P3-9 · 2026-07-07" in _GO
    i = _GO.find("_seed_food = _re.sub(")
    assert i != -1, "el strip _seed_food debe existir"
    # la nota embebe _seed_food, NO _seed_line
    j = _GO.find("🌱 Nota del Nutricionista AI: {_seed_verb} {_seed_food}")
    assert j != -1, "la nota 🌱 debe usar {_seed_food} (sin cantidad), no {_seed_line}"


def test_seed_food_strip_drops_leading_qty():
    strip = lambda s: re.sub(
        r"^\s*\d+(?:[.,]\d+)?\s*(?:g|gr|gramos|ml|kg|taza|tazas|cda|cdas|cdta|cdtas|unidad|unidades)?"
        r"\s*(?:de\s+)?", "", s, flags=re.IGNORECASE).strip() or s
    assert strip("50 g de zanahoria rallada") == "zanahoria rallada"
    assert strip("10 g de semillas de linaza") == "semillas de linaza"
    assert strip("10 g de nueces") == "nueces"
