"""[P1-BARIATRIC-VOLUME-CAP · 2026-06-27] (iteración 4 — GAP2 del crítico adversario) Tras iter 3 el reviewer
bajó de crítico a HIGH y el plan se ENTREGA, pero quedaban porciones absurdas que el solver infla para clavar
kcal ignorando el pouch: '7.19 ciruelas (287g)', '690g de vegetales', '4.82 nísperos (144g)', '27.5 almendras'.

Fixes iter 4:
  - cap_bariatric_portions 2ª pasada: cap de VOLUMEN AGREGADO por comida (≤300g principal / ≤200g merienda),
    recortando ítems NO-proteicos por factor común; la proteína queda intacta (no re-infla volumen).
  - per-item: + tokens níspero/ciruela (fruta ≤80g) y nueces/semillas (≤20g).
  - target de proteína bariátrico bajado a 80g (clínico 60-80g) → cierra el déficit relativo (76/90→76/80).
"""
from __future__ import annotations

import re
from pathlib import Path

import nutrition_calculator as nc

_BACKEND = Path(nc.__file__).resolve().parent


class _StubDB:
    """pollo = proteína densa; resto = no-proteico bajo en cal (veg/fruta)."""
    def macros_from_ingredient_string(self, s):
        m = re.match(r"\s*(\d+(?:\.\d+)?)\s*g", str(s))
        g = float(m.group(1)) if m else None
        is_prot = any(t in str(s).lower() for t in ("pollo", "res", "pescado", "huevo"))
        return {"grams": g, "protein": (g or 0) * (0.25 if is_prot else 0.01),
                "carbs": (g or 0) * 0.03, "fats": (g or 0) * 0.02,
                "kcal": (g or 0) * (1.5 if is_prot else 0.25)}


def _tot(ings):
    return sum(float(re.match(r"\s*(\d+(?:\.\d+)?)\s*g", i).group(1)) for i in ings
              if re.match(r"\s*(\d+(?:\.\d+)?)\s*g", i))


def _grams(ing):
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*g", ing)
    return float(m.group(1)) if m else None


def _patch_group(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "_ingredient_macro_group",
                        lambda s, db: "protein" if any(t in s.lower() for t in ("pollo", "res", "pescado", "huevo")) else "veg")


def test_volume_cap_main_meal_trims_to_limit_keeps_protein(monkeypatch):
    import graph_orchestrator as g
    _patch_group(monkeypatch)
    days = [{"day": 1, "meals": [{"meal": "Cena", "name": "Ensalada",
            "ingredients": ["450g de Pepino", "240g de Tomate", "90g de Pollo"]}]}]
    g.cap_bariatric_portions(days, {"medicalConditions": ["Cirugía Bariátrica"]}, db=_StubDB())
    ings = days[0]["meals"][0]["ingredients"]
    assert _tot(ings) <= g.BARIATRIC_MEAL_VOLUME_G + 1, f"volumen no capeado: {_tot(ings)}"
    # la proteína (pollo) NO se recorta
    pollo = next(i for i in ings if "pollo" in i.lower())
    assert _grams(pollo) >= 90


def test_volume_cap_snack_stricter(monkeypatch):
    import graph_orchestrator as g
    _patch_group(monkeypatch)
    days = [{"day": 1, "meals": [{"meal": "Merienda PM", "name": "Snack",
            "ingredients": ["300g de Pepino", "120g de Zanahoria"]}]}]
    g.cap_bariatric_portions(days, {"medicalConditions": ["Cirugía Bariátrica"]}, db=_StubDB())
    assert _tot(days[0]["meals"][0]["ingredients"]) <= g.BARIATRIC_SNACK_VOLUME_G + 1


def test_per_item_caps_nispero_ciruela_and_nuts(monkeypatch):
    import graph_orchestrator as g
    _patch_group(monkeypatch)
    days = [{"day": 1, "meals": [{"meal": "Merienda AM", "name": "x", "ingredients": [
        "287g de Ciruela", "144g de Níspero", "33g de Almendras"]}]}]
    g.cap_bariatric_portions(days, {"medicalConditions": ["Cirugía Bariátrica"]}, db=_StubDB())
    ings = days[0]["meals"][0]["ingredients"]
    assert _grams(ings[0]) <= g.BARIATRIC_FRUIT_CAP_G, f"ciruela: {ings[0]}"
    assert _grams(ings[1]) <= g.BARIATRIC_FRUIT_CAP_G, f"níspero: {ings[1]}"
    assert _grams(ings[2]) <= g.BARIATRIC_NUT_CAP_G, f"almendras: {ings[2]}"


def test_volume_cap_noop_non_bariatric(monkeypatch):
    import graph_orchestrator as g
    _patch_group(monkeypatch)
    days = [{"day": 1, "meals": [{"meal": "Cena", "ingredients": ["450g de Pepino", "90g de Pollo"]}]}]
    assert g.cap_bariatric_portions(days, {"medicalConditions": ["Ninguna"]}, db=_StubDB()) == 0


def test_bariatric_protein_target_lowered_to_80():
    fd = {"weight": 90, "weightUnit": "kg", "height": 170, "age": 45, "gender": "male",
          "activityLevel": "moderate", "mainGoal": "maintenance", "medicalConditions": ["Cirugía Bariátrica"]}
    assert nc.get_nutrition_targets(fd)["macros"]["protein_g"] <= 80


def test_anchors():
    go = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "P1-BARIATRIC-VOLUME-CAP" in go
    assert "BARIATRIC_MEAL_VOLUME_G" in go and "BARIATRIC_SNACK_VOLUME_G" in go
    assert "_BARIATRIC_NUT_TOKENS" in go and "nispero" in go and "ciruela" in go
