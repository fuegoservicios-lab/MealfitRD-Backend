"""[P1-BARIATRIC-TORONJA · 2026-06-27] (iter 7) Re-test corr=c42e0575 (run guest/pool-aleatorio) regresó a
CRÍTICO por foods que los caps de porción no cubrían:
  - TORONJA (contraindicación ABSOLUTA bariátrica: inhibe CYP3A4): la regla bariátrica reusaba solo
    _DM2_SUGAR_SUBS, no _DM2_GLYCEMIC_SUBS (que tiene toronja→Fresa). Bug de seguridad. Fix: combinar ambos.
  - Fruta ALTO-IG (mango 103g, guineo 83g) > 50g → dumping: cap más estricto ≤50g para alto-IG (vs 80g general).
  - Falso positivo de mi gen-sanity: dropeó "caldo de verduras" (legítimo). Fix: solo dropear GARBLE de 1 palabra.
  - Sub-bug: la recuperación de kcal re-inflaba ítems con cap (manzana 80→133). Fix: recovery no escala cappables.
"""
from __future__ import annotations

import re
from pathlib import Path

import condition_rules as cr
import nutrition_calculator as nc

_BACKEND = Path(nc.__file__).resolve().parent


class _StubDB:
    def macros_from_ingredient_string(self, s):
        m = re.search(r"\((\d+(?:\.\d+)?)\s*g", str(s)) or re.match(r"\s*(\d+(?:\.\d+)?)\s*g", str(s))
        # 'caldo'/'esguineocas' no resuelven a gramos (no en catálogo)
        bad = any(b in str(s).lower() for b in ("caldo", "esguineocas"))
        g = float(m.group(1)) if (m and not bad) else None
        return {"grams": g, "protein": 0.0, "carbs": (g or 0) * 0.1, "fats": 0.0, "kcal": (g or 0)}


def _grams(ing):
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*g", ing)
    return float(m.group(1)) if m else None


# ──────────────────────────── TORONJA (seguridad) ────────────────────────────

def test_bariatric_substitutes_toronja():
    subs = cr.collect_substitutions({"medicalConditions": ["Cirugía Bariátrica"]})
    blob = " ".join(str(s) for s in subs).lower()
    assert "toronja" in blob and "fresa" in blob, "toronja→fresa ausente en subs bariátricas"


def test_bariatric_substitutes_toronja_token_variants():
    # 'manga gastrica' (post-merge upstream → medicalConditions) también dispara la regla bariátrica.
    # (collect_substitutions lee medicalConditions; el merge de otherConditions ocurre aguas arriba, P1-FORM-6.)
    subs = cr.collect_substitutions({"medicalConditions": ["manga gastrica"]})
    blob = " ".join(str(s) for s in subs).lower()
    assert "toronja" in blob


# ──────────────────────────── fruta alto-IG ≤50g ────────────────────────────

def test_highgi_fruit_capped_to_50(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "_ingredient_macro_group", lambda s, db: "fruta")
    days = [{"day": 1, "meals": [{"meal": "Merienda AM", "name": "x",
            "ingredients": ["1 mango maduro (103 g)", "1 guineo (90 g)", "80g de Manzana"]}]}]
    g.cap_bariatric_portions(days, {"medicalConditions": ["Cirugía Bariátrica"]}, db=_StubDB())
    ings = days[0]["meals"][0]["ingredients"]
    assert _grams(ings[0]) <= g.BARIATRIC_HIGHGI_FRUIT_CAP_G, f"mango: {ings[0]}"
    assert _grams(ings[1]) <= g.BARIATRIC_HIGHGI_FRUIT_CAP_G, f"guineo: {ings[1]}"
    # manzana (bajo-IG) usa el cap general 80 y NO se re-infla por la recuperación
    assert _grams(ings[2]) <= g.BARIATRIC_FRUIT_CAP_G, f"manzana re-inflada: {ings[2]}"


def test_recovery_does_not_reinflate_cappable(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "_ingredient_macro_group", lambda s, db: "fruta")
    days = [{"day": 1, "meals": [{"meal": "Merienda", "name": "x",
            "ingredients": ["1 mango (103 g)", "80g de Manzana"]}]}]
    g.cap_bariatric_portions(days, {"medicalConditions": ["Cirugía Bariátrica"]}, db=_StubDB())
    assert _grams(days[0]["meals"][0]["ingredients"][1]) <= 80


# ──────────────────────────── gen-sanity falso positivo ────────────────────────────

def test_gen_sanity_keeps_multiword_broth(monkeypatch):
    import graph_orchestrator as g
    import shopping_calculator
    monkeypatch.setattr(shopping_calculator, "_is_verified_for_shopping", lambda n: False)
    plan = {"days": [{"day": 1, "meals": [{"meal": "Cena", "name": "Pollo",
            "ingredients": ["90g de Pollo", "0.5 taza de caldo de verduras (119g)", "EsGuineocas"]}]}]}
    g._generation_sanity_autofix(plan, db=_StubDB())
    ings = [i.lower() for i in plan["days"][0]["meals"][0]["ingredients"]]
    assert any("caldo de verduras" in i for i in ings), "caldo (multi-palabra) NO debe dropearse"
    assert not any("esguineocas" in i for i in ings), "EsGuineocas (1 palabra) sí debe dropearse"


def test_anchors():
    cr_src = (_BACKEND / "condition_rules.py").read_text(encoding="utf-8")
    assert "P1-BARIATRIC-TORONJA" in cr_src and "_DM2_SUGAR_SUBS + _DM2_GLYCEMIC_SUBS" in cr_src
    go = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "_BARIATRIC_HIGHGI_FRUIT_TOKENS" in go and "MEALFIT_BARIATRIC_HIGHGI_FRUIT_CAP_G" in go
