"""[P1-BARIATRIC-PROTEIN-DENSITY · 2026-06-27] (iteración 3) Tras P1-BARIATRIC-PHASE-CONTEXT el reviewer bajó
de ~35 a ~8 rechazos (la fase mantenimiento desbloqueó los alimentos de dieta general). Los restantes eran
gaps DETERMINISTAS que el crítico adversario ya había predicho + una mala selección de proteína:
  - Longaniza (embutido graso) elegida como proteína-main → rechazo.
  - Guineo 240g / mango en merienda → dumping (porción/alto-IG).
  - Aguacate 0.5 unidad excede ≤1/4.
  - Déficit de proteína 79/90 (a 2g, por usar Longaniza/soya baja calidad).

Fixes iteración 3:
  1. cap_bariatric_portions extendido → cap determinista de FRUTA (≤80g) y AGUACATE (≤30g).
  2. Pool de proteína bariátrico penaliza embutidos (×0.1) + el swap high-density los reemplaza por animal magro.
  3. Pool de fruta bariátrico penaliza alto-IG (guineo/mango/uva/piña ×0.15) → prefiere fresa/lechosa/mandarina.
"""
from __future__ import annotations

import re
from pathlib import Path

import nutrition_calculator as nc

_BACKEND = Path(nc.__file__).resolve().parent


class _StubDB:
    def macros_from_ingredient_string(self, s):
        m = re.match(r"\s*(\d+(?:\.\d+)?)\s*g", str(s))
        g = float(m.group(1)) if m else None
        return {"grams": g, "protein": (g or 0) * 0.05, "carbs": (g or 0) * 0.2,
                "fats": (g or 0) * 0.05, "kcal": (g or 0) * 1.5}


def _grams(ing: str):
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*g", ing)
    return float(m.group(1)) if m else None


# ──────────────────────────── (1) cap determinista fruta + aguacate ────────────────────────────

def test_cap_bariatric_caps_fruit_and_avocado():
    import graph_orchestrator as g
    days = [{"day": 1, "meals": [{"name": "Merienda", "ingredients": [
        "240g de Guineo en cubos", "75g de Aguacate", "100g de Espinaca"]}]}]
    n = g.cap_bariatric_portions(days, {"medicalConditions": ["Cirugía Bariátrica"]}, db=_StubDB())
    ings = days[0]["meals"][0]["ingredients"]
    assert n >= 2
    assert _grams(ings[0]) <= g.BARIATRIC_FRUIT_CAP_G, f"guineo no capeado: {ings[0]}"
    assert _grams(ings[1]) <= g.BARIATRIC_AVOCADO_CAP_G, f"aguacate no capeado: {ings[1]}"


def test_cap_bariatric_does_not_touch_spinach_via_pina_token():
    # 'pina' (start-anchored) NO debe capear 'espinaca' como si fuera fruta
    import graph_orchestrator as g
    days = [{"day": 1, "meals": [{"name": "x", "ingredients": ["300g de Espinaca salteada", "90g de Pollo"]}]}]
    g.cap_bariatric_portions(days, {"medicalConditions": ["Cirugía Bariátrica"]}, db=_StubDB())
    # espinaca no es fruta → su gramaje no se recorta por el cap de fruta (puede subir por recovery, nunca a 80)
    assert _grams(days[0]["meals"][0]["ingredients"][0]) >= 300


def test_cap_bariatric_fruit_noop_non_bariatric():
    import graph_orchestrator as g
    days = [{"day": 1, "meals": [{"name": "x", "ingredients": ["240g de Guineo"]}]}]
    assert g.cap_bariatric_portions(days, {"medicalConditions": ["Ninguna"]}, db=_StubDB()) == 0


# ──────────────────────────── (2)(3) anchors selección proteína/fruta ────────────────────────────

def test_bariatric_penalizes_embutidos_in_protein_pool():
    ai = (_BACKEND / "ai_helpers.py").read_text(encoding="utf-8")
    # embutidos penalizados también para bariátrica (no solo por goal)
    assert "_GOALS_PENALIZE_PROCESSED or _is_bariatric" in ai
    # el swap high-density reemplaza embutidos como main para bariátrica
    assert "_should_replace_main" in ai and "_PROCESSED_MEAT_KEYWORDS" in ai


def test_bariatric_penalizes_high_gi_fruit_in_pool():
    ai = (_BACKEND / "ai_helpers.py").read_text(encoding="utf-8")
    assert "_HIGH_GI_FRUITS" in ai and "_is_bariatric" in ai


def test_cap_tokens_and_knobs_present():
    go = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "_BARIATRIC_FRUIT_TOKENS" in go and "_BARIATRIC_FAT_TOKENS" in go
    assert "MEALFIT_BARIATRIC_FRUIT_CAP_G" in go and "MEALFIT_BARIATRIC_AVOCADO_CAP_G" in go
