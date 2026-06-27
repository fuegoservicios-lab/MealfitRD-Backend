"""[P1-BARIATRIC-CAP-LABEL · 2026-06-27] (iteración 5) Tras iter 4 el reviewer ya no rechaza alimentos/porciones
bariátricas; las observaciones restantes eran artefactos de generación. Dos bariátrico-específicos cerrados aquí:
  #1 Etiqueta del cap incoherente: mi cap reescalaba "1 Piña mediano (240g)" a "0.33 Piña"/"1 Piña (80g)" (no
     medible). Fix `_resc_cap_coherent`: si la unidad líder queda <0.5, reescribe a gramos ("80 g de Piña mediano").
  #2 Nueces/semillas ENTERAS (riesgo obstructivo): prompt_block bariátrico exige molidas/mantequilla + el pool
     veg/grasa de ai_helpers las penaliza (×0.1) para bariátrica (las formas molida/mantequilla NO se penalizan).
"""
from __future__ import annotations

import re
from pathlib import Path

import nutrition_calculator as nc

_BACKEND = Path(nc.__file__).resolve().parent


class _StubDB:
    def macros_from_ingredient_string(self, s):
        m = re.search(r"\((\d+(?:\.\d+)?)\s*g", str(s)) or re.match(r"\s*(\d+(?:\.\d+)?)\s*g", str(s))
        g = float(m.group(1)) if m else None
        is_prot = "pollo" in str(s).lower()
        return {"grams": g, "protein": (g or 0) * (0.25 if is_prot else 0.01),
                "carbs": (g or 0) * 0.1, "fats": 0.0, "kcal": (g or 0) * (1.5 if is_prot else 0.5)}


# ──────────────────────────── #1 etiqueta coherente del cap ────────────────────────────

def test_cap_label_unit_fruit_rewritten_to_grams(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "_ingredient_macro_group", lambda s, db: "protein" if "pollo" in s.lower() else "fruta")
    days = [{"day": 1, "meals": [{"meal": "Merienda PM", "name": "x",
            "ingredients": ["1 Piña mediano (240 g)", "30 g de Pollo"]}]}]
    g.cap_bariatric_portions(days, {"medicalConditions": ["Cirugía Bariátrica"]}, db=_StubDB())
    pina = days[0]["meals"][0]["ingredients"][0]
    # NO debe quedar "1 Piña" ni "0.33 Piña"; debe ser gramos coherentes y ≤80
    assert "g de" in pina.lower() or pina.lower().strip().startswith(("80 g", "80g"))
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*g", pina)
    assert m and float(m.group(1)) <= 80, f"etiqueta incoherente: {pina}"
    assert not pina.lower().startswith("1 "), f"quedó unidad entera engañosa: {pina}"


def test_cap_label_helper_pure():
    import graph_orchestrator as g
    # gram-based: se reescala normal, sin reescribir a 'X g de' redundante raro
    out = g._resc_cap_coherent("150 g de Queso blanco (150 g)", 30.0 / 150.0, 30)
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*g", out)
    assert m and float(m.group(1)) <= 31, out
    # unidad fraccionaria → reescribe a gramos
    out2 = g._resc_cap_coherent("1 Piña mediano (240 g)", 80.0 / 240.0, 80)
    assert not out2.strip().startswith("1 "), out2
    assert "80 g de" in out2 or out2.strip().startswith("80 g"), out2


# ──────────────────────────── #2 nueces enteras: prompt + pool ────────────────────────────

def test_bariatric_prompt_requires_ground_nuts():
    import condition_rules as cr
    blk = cr.build_condition_prompt({"medicalConditions": ["Cirugía Bariátrica"]}).lower()
    assert ("molidas" in blk or "molida" in blk) and "mantequilla" in blk
    assert "obstrucci" in blk  # riesgo de obstrucción mencionado


def test_ai_helpers_penalizes_whole_nuts_for_bariatric():
    ai = (_BACKEND / "ai_helpers.py").read_text(encoding="utf-8")
    assert "_WHOLE_NUT_SEED_TOKENS" in ai
    # excluye explícitamente las formas seguras (mantequilla/molida/fileteada)
    assert '"mantequilla" in _vn or "molid" in _vn' in ai


def test_anchors():
    go = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "P1-BARIATRIC-CAP-LABEL" in go and "def _resc_cap_coherent" in go
