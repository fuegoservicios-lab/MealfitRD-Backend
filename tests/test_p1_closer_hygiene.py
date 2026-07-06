"""[P1-CLOSER-HYGIENE + P1-REVERSE-COH-PLURAL + P2s · 2026-07-06]

Review visual #7 (plan da7bb310) — la capa de coherencia del closer/reverse-coherence:
- Complemento ESPURIO "incorpora también tomates medianos" cuando los pasos SÍ usan "el tomate"
  (ceguera singular/plural del matcher, ×2 platos del mismo plan).
- Paso 💪 del closer DUPLICADO idéntico (corre en 2 seams sin chequear) y REDUNDANTE (guisantes
  ya guisados en el TdF; huevo extra en un revoltillo de 4 huevos).
- "½ taza de yogurt cocido" / "Cocina huevo cocido a la plancha" — sufijo/nombre "cocido" absurdo.
- "4 huevos" + "1 huevo" líneas sin consolidar (la fusión solo veía gramos).
- "¼ taza de arroz cocido (40g crudo)" — single-form 3× inflada + claim "40 g crudo rinden ~¼
  taza cocida" (falso: rinden ~¾).
- "8 tortillas integrales (240 g c/u)" — total etiquetado como cada-una.
- "Licúa... (~10-12 min a fuego medio)" — técnica licuado inexistente en el timetemp.
- "210 g de queso" en una merienda de fruta — re-inflado post-closer sin re-fire del snack-cap.
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


@pytest.fixture()
def go():
    import graph_orchestrator as g
    return g


# ───────────── reverse-coherence plural ─────────────

def test_plural_line_singular_steps_no_spurious_complement(go, monkeypatch):
    monkeypatch.setattr(go, "RECIPE_REVERSE_COHERENCE_ENABLED", True)
    meal = {
        "name": "Revoltillo con Casabe",
        "ingredients": ["4 huevos", "1½ tomates medianos", "½ cebolla"],
        "recipe": ["Mise en place: pica finamente el tomate y la cebolla.",
                   "El Toque de Fuego: agrega el tomate y cocina 2 minutos más. "
                   "Bate los huevos con sal y cuájalos.",
                   "Montaje: sirve."],
    }
    assert go._ensure_ingredients_used_in_recipe(meal) == 0, \
        "'tomates medianos' (plural) SÍ está en pasos como 'el tomate' → cero complemento espurio"
    assert not any("(complemento)" in str(s) for s in meal["recipe"])


def test_genuinely_missing_still_gets_step(go, monkeypatch):
    monkeypatch.setattr(go, "RECIPE_REVERSE_COHERENCE_ENABLED", True)
    meal = {
        "name": "Plato",
        "ingredients": ["150 g de pollo", "1 zanahoria"],
        "recipe": ["Mise en place: sazona el pollo.",
                   "El Toque de Fuego: cocina el pollo 8 min.", "Montaje: sirve."],
    }
    assert go._ensure_ingredients_used_in_recipe(meal) >= 1, \
        "ingrediente genuinamente ausente de los pasos → sigue ganando su paso de uso"


# ───────────── closer: dup + redundante + wording ─────────────

def test_closer_step_not_duplicated(go):
    meal = {"name": "Batata con Guisantes",
            "recipe": ["Mise en place: pela.", "Montaje: sirve."]}
    assert go._append_closer_protein_step(meal, "queso cottage", True) is True
    n1 = sum(1 for s in meal["recipe"] if "💪" in str(s))
    assert go._append_closer_protein_step(meal, "queso cottage", True) is False, \
        "2ª pasada (el closer corre en dos seams) → no duplica"
    assert sum(1 for s in meal["recipe"] if "💪" in str(s)) == n1 == 1


def test_closer_step_skipped_when_food_already_in_recipe(go):
    meal = {"name": "Batata con Guisantes",
            "recipe": ["Mise en place: escurre los guisantes secos cocidos.",
                       "El Toque de Fuego: añade los guisantes secos y cocina 5 minutos.",
                       "Montaje: sirve."]}
    assert go._append_closer_protein_step(meal, "guisantes secos", False) is False, \
        "la receta YA cocina los guisantes → el paso genérico es redundante"
    meal2 = {"name": "Revoltillo",
             "recipe": ["Mise en place: bate los huevos.",
                        "El Toque de Fuego: cuaja los huevos 3-4 min.", "Montaje: sirve."]}
    assert go._append_closer_protein_step(meal2, "huevo cocido", False) is False, \
        "huevo extra en un revoltillo de huevos → sin paso 'Cocina huevo cocido'"


def test_cocido_name_gets_soft_wording(go):
    txt = go._closer_protein_step_text("huevo cocido", False)
    assert txt.startswith("Incorpora"), \
        f"nombre con 'cocido' jamás recibe 'Cocina X a la plancha': {txt}"
    assert "Cocina huevo cocido" not in txt


def test_dairy_line_never_gets_cocido_suffix():
    i = _GO.index("[P1-CLOSER-HYGIENE · 2026-07-06] lácteos jamás llevan sufijo")
    win = _GO[i:i + 700]
    assert "_dairy_nm" in win and '"cocid" in _nm_strip' in win, \
        "'½ taza de yogurt cocido' muere en la composición de la línea"


# ───────────── consolidación de conteos ─────────────

def test_count_lines_consolidated(go, monkeypatch):
    monkeypatch.setattr(go, "INGREDIENT_LINE_CONSOLIDATE_ENABLED", True)
    days = [{"day": 1, "meals": [{
        "meal": "Desayuno", "name": "Revoltillo",
        "ingredients": ["4 huevos", "1½ tomates medianos", "1 huevo"],
        "ingredients_raw": ["4 huevos", "1½ tomates medianos", "1 huevo"],
        "recipe": ["Mise: x.", "Montaje: y."],
    }]}]
    assert go._consolidate_duplicate_gram_lines(days) >= 1
    ings = days[0]["meals"][0]["ingredients"]
    assert "5 huevos" in ings and "1 huevo" not in ings and "4 huevos" not in ings, ings
    assert days[0]["meals"][0]["ingredients_raw"] == ings, "lockstep raw"


def test_count_lines_different_tail_not_merged(go, monkeypatch):
    monkeypatch.setattr(go, "INGREDIENT_LINE_CONSOLIDATE_ENABLED", True)
    days = [{"day": 1, "meals": [{
        "meal": "Cena", "name": "X",
        "ingredients": ["2 papas medianas", "1 papa grande"],
        "ingredients_raw": ["2 papas medianas", "1 papa grande"],
        "recipe": ["Mise: x.", "Montaje: y."],
    }]}]
    go._consolidate_duplicate_gram_lines(days)
    assert len(days[0]["meals"][0]["ingredients"]) == 2, \
        "resto-de-nombre distinto (medianas vs grande) → conservador, no fusionar"


# ───────────── arroz single crudo + rinden + c/u ─────────────

def test_grain_single_raw_recomputed(go):
    days = [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Salteado con Arroz",
        "ingredients": ["¼ taza de arroz integral cocido (41 g crudo)"],
        "ingredients_raw": ["¼ taza de arroz integral cocido (41 g crudo)"],
        "recipe": ["Mise en place: cocina el arroz integral según instrucciones "
                   "(40 g crudo rinden ~¼ taza cocida)."],
    }]}]
    assert go._fix_cooked_raw_annotations(days) >= 2
    line = days[0]["meals"][0]["ingredients"][0]
    m = re.search(r"\((\d+)g crudos\)", line)
    assert m and int(m.group(1)) <= 20, f"¼ taza cocida ≈ 40g → crudo ≈ 14g (no 41): {line}"
    step = days[0]["meals"][0]["recipe"][0]
    m2 = re.search(r"(\d+) g crudos rinden", step)
    assert m2 and int(m2.group(1)) <= 20, f"el claim del paso se recomputa: {step}"


def test_cu_total_mislabel_fixed(go):
    days = [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Wrap",
        "ingredients": ["8 tortillas integrales (240 g c/u)"],
        "ingredients_raw": ["8 tortillas integrales (240 g c/u)"],
        "recipe": ["Mise: x."],
    }]}]
    assert go._fix_cooked_raw_annotations(days) >= 1
    line = days[0]["meals"][0]["ingredients"][0]
    assert "en total" in line and "c/u" not in line, f"240g es el total (30g/tortilla): {line}"


def test_cu_legit_per_unit_untouched(go):
    days = [{"day": 1, "meals": [{
        "meal": "Cena", "name": "X",
        "ingredients": ["2 batatas (200 g c/u)"],
        "ingredients_raw": ["2 batatas (200 g c/u)"],
        "recipe": ["Mise: x."],
    }]}]
    go._fix_cooked_raw_annotations(days)
    assert "c/u" in days[0]["meals"][0]["ingredients"][0], \
        "200g por batata es plausible per-unit (200/2=100 fuera del rango-total) → intacto"


# ───────────── licuado timetemp ─────────────

def test_licuado_injection_not_fuego(go):
    meal = {"name": "Batido de Guineo con Maní",
            "recipe": ["Mise en place: pela el guineo.",
                       "El Toque de Fuego: coloca todo en la licuadora y licúa a máxima velocidad.",
                       "Montaje: sirve frío."]}
    assert go._inject_recipe_time_temp_defaults(meal) is True
    _tdf = next(s for s in meal["recipe"] if "licúa" in str(s).lower())
    assert "licuado" in _tdf and "a fuego" not in _tdf.lower(), \
        f"batido de licuadora jamás gana '(~10-12 min a fuego medio)': {_tdf}"


def test_licuado_persisted_absurd_reclamped(go):
    meal = {"name": "Batido de Guineo",
            "recipe": ["Mise en place: pela.",
                       "El Toque de Fuego: licúa hasta obtener textura cremosa "
                       "(~10-12 min a fuego medio).",
                       "Montaje: sirve."]}
    assert go._clamp_recipe_time_temp_outliers(meal) is True
    _tdf = meal["recipe"][1]
    assert "de licuado" in _tdf and "a fuego" not in _tdf, \
        f"el absurdo persistido se re-clampa Y pierde el 'a fuego medio' residual: {_tdf}"


# ───────────── queso en merienda ─────────────

def test_snack_cheese_capped(go, monkeypatch):
    monkeypatch.setattr(go, "PORTION_REALISM_CAP_ENABLED", True)
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)

    class _NoopDB:
        def macros_from_ingredient_string(self, s):
            return None

        def lookup(self, s):
            return None

    days = [{"day": 1, "meals": [{
        "meal": "Merienda", "name": "Lechosa con Queso",
        "ingredients": ["210 g de queso", "½ lechosa"],
        "ingredients_raw": ["210 g de queso", "½ lechosa"],
        "recipe": ["Mise: x.", "Montaje: y."],
    }, {
        "meal": "Merienda", "name": "Bowl de Yogurt",
        "ingredients": ["200 g de yogurt griego"],
        "ingredients_raw": ["200 g de yogurt griego"],
        "recipe": ["Mise: x.", "Montaje: y."],
    }]}]
    assert go._cap_unrealistic_portions(days, db=_NoopDB()) == 1
    q = days[0]["meals"][0]["ingredients"][0]
    n = float(re.match(r"^\s*(\d+(?:[.,]\d+)?)", q).group(1).replace(",", "."))
    assert n <= float(go.SNACK_CHEESE_CAP_G) + 0.01, f"queso de merienda al techo: {q}"
    assert days[0]["meals"][1]["ingredients"][0].startswith("200 g de yogurt"), \
        "yogurt exento (un bowl de 200g es normal)"
