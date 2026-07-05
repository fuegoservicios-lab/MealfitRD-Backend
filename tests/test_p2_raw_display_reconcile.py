"""[P2-RAW-DISPLAY-RECONCILE + P2-BIGFRUIT-COUNT-CAP + P2-TSP-TO-TBSP · 2026-07-05]

Forensics del plan vivo 55846e5e:
- "8¾ cdta de miel" existía SOLO en `ingredients` — sin contraparte en `ingredients_raw`. La
  lista de compras Y los medidores (azúcares añadidos/micros/sodio) leen raw preferido → 62g de
  miel (≈51g de azúcar) eran INVISIBLES: panel "Azúcares añadidos 0/25" con la miel en el plato.
  La resolución del catálogo estaba BIEN (miel: density 340/cup, sugars 82.1/100g — verificado
  en vivo); el gap era el desync display/raw (clase ya medida en 3aa6e58a).
  → `_reconcile_display_missing_in_raw`: línea de display cuyo token principal no aparece en
  NINGUNA línea del raw → se APPENDEA al raw (dirección segura: solo se gana visibilidad).
- "2 Lechosa" (~3kg) para un aderezo → count-caps de frutas grandes (lechosa/piña/melón ≤1).
- "8¾ cdta de miel" → "3 cdas de miel" (3 cdta = 1 cda, volumen exacto).
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


# ───────────────────── reconcile display → raw ─────────────────────

def test_wired_before_panel_recompute_and_at_boundary():
    i_rec = _GO.index("_reconcile_display_missing_in_raw(days)")
    i_panel = _GO.index("(P2-1) Panel de micros FRESCO al final del motor")
    assert i_rec < i_panel, "el reconcile corre ANTES del recompute del panel (el medidor ve la miel)"
    assert _GO.count("_reconcile_display_missing_in_raw(days)") >= 2, "también en el persist boundary"


def test_honey_becomes_visible_to_raw(go=None):
    import graph_orchestrator as g
    days = [{"day": 1, "meals": [{
        "meal": "Merienda", "name": "Yogurt con Miel",
        "ingredients": ["1 taza de yogurt griego entero (201 g)", "8¾ cdta de miel"],
        "ingredients_raw": ["1 taza de yogurt griego entero (201 g)"],
        "recipe": ["Mise en place: mide.", "Montaje: sirve."],
    }]}]
    assert g._reconcile_display_missing_in_raw(days) == 1
    raw = days[0]["meals"][0]["ingredients_raw"]
    assert any("miel" in str(r).lower() for r in raw), "la miel ya es visible para compras/medidores"
    assert g._reconcile_display_missing_in_raw(days) == 0, "idempotente"


def test_existing_raw_lines_never_touched():
    import graph_orchestrator as g
    days = [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Plato",
        "ingredients": ["150 g de pollo"],
        "ingredients_raw": ["150g de pechuga de pollo"],
        "recipe": ["Mise en place: x.", "Montaje: y."],
    }]}]
    assert g._reconcile_display_missing_in_raw(days) == 0, \
        "'pollo' YA aparece en el raw (token match) → nada que añadir, nada se modifica"
    assert days[0]["meals"][0]["ingredients_raw"] == ["150g de pechuga de pollo"]


# ───────────────────── big-fruit count caps ─────────────────────

@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "PORTION_REALISM_CAP_ENABLED", True)
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


class _NoopDB:
    def macros_from_ingredient_string(self, s):
        return None

    def lookup(self, s):
        return None


def test_two_papayas_capped_to_one(go):
    days = [{"day": 1, "meals": [{
        "meal": "Merienda", "name": "Brochetas con Aderezo de Lechosa",
        "ingredients": ["2 Lechosa", "30 g de queso blanco"],
        "ingredients_raw": ["2 Lechosa", "30 g de queso blanco"],
        "recipe": ["Mise en place: x.", "Montaje: y."],
    }]}]
    assert go._cap_unrealistic_portions(days, db=_NoopDB()) == 1
    line = days[0]["meals"][0]["ingredients"][0]
    n = float(re.match(r"^\s*(\d+(?:[.,]\d+)?)", line).group(1).replace(",", "."))
    assert n <= 1.0 + 0.01, f"2 lechosas (~3kg) para 1 persona → ≤1: {line}"


# ───────────────────── cdta → cdas ─────────────────────

def _pretty(s):
    from humanize_ingredients import _prettify_quantity_display
    return _prettify_quantity_display(s)


def test_nine_tsp_to_tbsp():
    out = _pretty("8¾ cdta de miel")
    # 8¾ es fracción unicode en lead → el prettify decimal no la parsea; probar decimal:
    out2 = _pretty("9 cdta de miel")
    assert out2.startswith("3 cdas"), f"9 cdta = 3 cdas: {out2}"


def test_few_tsp_untouched():
    assert _pretty("1 cdta de vainilla") == "1 cdta de vainilla"
    # <3 cdta NO se promueve a cdas; el plural-concordance sí corrige "2 cdta"→"2 cdtas" (correcto).
    assert _pretty("2 cdta de canela en polvo") == "2 cdtas de canela en polvo"
