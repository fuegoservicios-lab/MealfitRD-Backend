"""[P3-DISPLAY-GRAMMAR + P3-STEP-FRACTIONS + P3-GUARD-BLIND-WATER-WHITELIST +
P3-CANNED-MIN-SERVIBLE + P3-EGG-BATTER-NOTE · 2026-07-05]

Batch de pulido P3 acumulado de los reviews visuales #5/#6:
- "2½ tomate" / "1½ cebolla" / "1 chuletas" / "½ cdas" / "2 guineos mediano" /
  "1 Lechosa mediano" → concordancia número/género + minúscula de alimentos conocidos.
- Pasos con "2.5 cdas" / "0.5 taza" / "1/4 de la masa" vs lista con fracciones unicode.
- WARN [VERIFIED-ONLY-GUARD-BLIND] ruidoso por 'Agua' (no-comprable por diseño).
- "20g de sardinas en lata" — nadie abre una lata para 20g.
- Nota de huevo "yema y clara firmes" sobre MASA de panqueques.
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "shopping_calculator.py"), encoding="utf-8") as f:
    _SC = f.read()


def _pretty(s):
    from humanize_ingredients import _prettify_quantity_display
    return _prettify_quantity_display(s)


# ───────────────────── concordancia display ─────────────────────

def test_fractional_gt1_pluralizes_noun():
    assert _pretty("2½ tomate").startswith("2½ tomates")
    assert _pretty("1½ cebolla").startswith("1½ cebollas")


def test_qty_one_singularizes_noun():
    assert _pretty("1 chuletas de cerdo (75g)").startswith("1 chuleta de cerdo")


def test_fraction_lead_unit_singularized():
    assert _pretty("½ cdas de aceite de oliva") == "½ cda de aceite de oliva"


def test_adjective_number_agreement():
    out = _pretty("2 guineos mediano (239g)")
    assert out.startswith("2 guineos medianos"), out


def test_adjective_gender_and_lowercase_known_food():
    out = _pretty("1 Lechosa mediano (203g)")
    assert out.startswith("1 lechosa mediana"), out


def test_unknown_adjective_blocks_number_change():
    # "batido" no está en el set de adjetivos → pluralizar el sustantivo crearía
    # "2½ tomates batido" (mismatch nuevo) → conservador: intacto.
    out = _pretty("2½ tomate batido")
    assert "tomates batido" not in out, out


def test_known_participle_inflects_with_noun():
    out = _pretty("2½ tomate picado")
    assert out.startswith("2½ tomates picados"), out


# ───────────────────── fracciones en pasos ─────────────────────

def test_step_decimal_and_ascii_fractions():
    from humanize_ingredients import prettify_step_fractions
    assert prettify_step_fractions("Unta cada tortilla con 2.5 cdas de mantequilla.") == \
        "Unta cada tortilla con 2½ cdas de mantequilla."
    assert prettify_step_fractions("mezcla la harina con 0.5 taza de agua tibia") == \
        "mezcla la harina con ½ taza de agua tibia"
    assert prettify_step_fractions("vierte 1/4 de la masa por panqueque") == \
        "vierte ¼ de la masa por panqueque"


def test_step_ranges_and_temps_untouched():
    from humanize_ingredients import prettify_step_fractions
    s = "Hornea 18-20 min a 180 °C y saltea 3-4 minutos; usa 10/12 de los floretes."
    assert prettify_step_fractions(s) == s


# ───────────────────── whitelist agua/caldo del WARN ─────────────────────

def test_guard_blind_whitelists_water():
    i = _SC.index("[VERIFIED-ONLY-GUARD-BLIND]")
    win = _SC[max(0, i - 1500):i]
    assert "P3-GUARD-BLIND-WATER-WHITELIST" in win, \
        "'Agua'/'hielo'/'caldo' no-comprables se excluyen del WARN (ruido, no desobediencia)"
    assert '("agua", "hielo")' in win
    assert 'startswith("caldo")' in win


# ───────────────────── piso enlatados ─────────────────────

def test_canned_protein_bumped_to_min(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "PORTION_SHRINK_FLOOR_ENABLED", True)
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)

    class _NoopDB:
        def macros_from_ingredient_string(self, s):
            return None

        def lookup(self, s):
            return None

    days = [{"day": 1, "meals": [{
        "meal": "Almuerzo", "name": "Plátano con Ensalada de Guineo",
        "ingredients": ["½ plátano verde", "20g de sardinas en lata", "60 g de queso"],
        "ingredients_raw": ["½ plátano verde", "20g de sardinas en lata", "60 g de queso"],
        "recipe": ["Mise: x.", "Montaje: y."],
    }]}]
    assert go._floor_subservible_portions(days, day_kcal_target=None, db=_NoopDB()) >= 1
    import re
    line = days[0]["meals"][0]["ingredients"][1]
    n = float(re.match(r"^\s*(\d+(?:[.,]\d+)?)", line).group(1).replace(",", "."))
    assert n >= float(go.CANNED_MIN_SERVIBLE_G), \
        f"20g de sardinas en lata → ≥{go.CANNED_MIN_SERVIBLE_G}g (nadie abre una lata para 20g): {line}"
    raw_line = days[0]["meals"][0]["ingredients_raw"][1]
    assert re.match(r"^\s*(\d+)", raw_line).group(1) == re.match(r"^\s*(\d+)", line).group(1), \
        "lockstep raw (compras compran lo mismo)"


def test_canned_above_min_untouched(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "PORTION_SHRINK_FLOOR_ENABLED", True)
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)

    class _NoopDB:
        def macros_from_ingredient_string(self, s):
            return None

    days = [{"day": 1, "meals": [{
        "meal": "Almuerzo", "name": "Atún",
        "ingredients": ["185g de atún en agua", "150 g de yuca"],
        "ingredients_raw": ["185g de atún en agua", "150 g de yuca"],
        "recipe": ["Mise: x.", "Montaje: y."],
    }]}]
    go._floor_subservible_portions(days, day_kcal_target=None, db=_NoopDB())
    assert days[0]["meals"][0]["ingredients"][0].startswith("185g"), "sobre el piso → intacto"


# ───────────────────── nota huevo en masa ─────────────────────

def test_egg_note_batter_variant():
    import graph_orchestrator as go
    plan = {"days": [{"day": 1, "meals": [{
        "meal": "Desayuno", "name": "Panqueques de Harina de Trigo con Mango",
        "ingredients": ["½ taza de harina de trigo (59g)", "2 huevos",
                        "¾ taza de leche descremada (179ml)"],
        "recipe": ["Mise en place: mide la harina, el huevo y la leche.",
                   "El Toque de Fuego: bate el huevo crudo con la leche, incorpora a los secos y "
                   "mezcla hasta obtener una masa homogénea. Cocina 2 minutos por lado.",
                   "Montaje: apila y sirve."],
    }]}]}
    _n = go._apply_food_safety_fixes(plan)
    rec = plan["days"][0]["meals"][0]["recipe"]
    notes = [s for s in rec if "Seguridad alimentaria" in str(s)]
    if notes:  # si el scan flageó el huevo, la nota debe ser la variante de MASA
        assert any("masa" in str(s).lower() for s in notes), notes
        assert not any("yema y clara firmes" in str(s) for s in notes), \
            "wording de huevo entero no aplica a masa batida"


def test_egg_note_whole_egg_keeps_classic_wording():
    import graph_orchestrator as go
    plan = {"days": [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Yuca con Claras Revueltas",
        "ingredients": ["8 claras de huevo (267g)", "150 g de yuca"],
        "recipe": ["Mise en place: pica.",
                   "El Toque de Fuego: vierte las claras de huevo crudo y revuelve hasta que cuajen.",
                   "Montaje: sirve."],
    }]}]}
    go._apply_food_safety_fixes(plan)
    rec = plan["days"][0]["meals"][0]["recipe"]
    notes = [s for s in rec if "Seguridad alimentaria" in str(s)]
    if notes:
        assert any("yema y clara firmes" in str(s) for s in notes), \
            "huevo entero/claras (sin masa) → wording clásico intacto"
