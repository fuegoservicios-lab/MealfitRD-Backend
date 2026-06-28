"""[P3-GEN-SANITY-AUTOFIX · 2026-06-27] Autofix determinista post-assemble de artefactos de generación del LLM
que el revisor médico bariátrico cazó (corr=579fb9a3) — NO bariátrico-específicos, afectan todos los planes:
  #4 ingrediente INCONGRUENTE (almidón/huevo) dentro de un batido/jugo → dropear ("½ papa en Batido de Lechosa").
  #5 nombre GLITCHEADO/irresoluble que no resuelve al catálogo verificado NI a gramos → dropear ("EsGuineocas").
(#3 "arroz de noche" se atacó en el prompt del day_generator §15d, no aquí — es soft-gate, no drop.)
Conservador: solo basura pura/incongruencia clara, nunca deja una comida vacía, recalcula macros.
"""
from __future__ import annotations

from pathlib import Path

import nutrition_calculator as nc

_BACKEND = Path(nc.__file__).resolve().parent


class _StubDB:
    # palabras que NO resuelven al catálogo (glitch del LLM) → lookup None
    _UNRESOLVABLE = ("esfresascas", "esguineocas", "frescas", "guineocas")

    def macros_from_ingredient_string(self, s):
        import re
        # Espeja el real (nutrition_db): lookup(name) PRIMERO → si no resuelve, None (el hint de gramos NO basta).
        if any(u in str(s).lower() for u in self._UNRESOLVABLE):
            return None
        m = re.match(r"\s*(\d+(?:\.\d+)?)\s*g", str(s))
        g = float(m.group(1)) if m else None
        return {"grams": g, "protein": (g or 0) * 0.1, "carbs": (g or 0) * 0.1, "fats": 0.0, "kcal": (g or 0)}

    def lookup(self, raw_name):
        # [P3-GEN-SANITY-GARBLE-MULTIWORD] None para mash-tokens irresolubles; objeto para palabras reales.
        return None if any(u in str(raw_name).lower() for u in self._UNRESOLVABLE) else object()


def _patch_verified(monkeypatch, unverified=("esguineocas",)):
    import shopping_calculator
    monkeypatch.setattr(shopping_calculator, "_is_verified_for_shopping",
                        lambda n: not any(u in str(n).lower() for u in unverified))


def test_drops_starch_in_batido(monkeypatch):
    import graph_orchestrator as g
    _patch_verified(monkeypatch)
    plan = {"days": [{"day": 1, "meals": [
        {"meal": "Merienda PM", "name": "Batido de Lechosa y Yogurt",
         "ingredients": ["120g de Lechosa", "0.5 papa mediana (75 g)", "120g de Yogurt"]}]}]}
    n = g._generation_sanity_autofix(plan, db=_StubDB())
    ings = plan["days"][0]["meals"][0]["ingredients"]
    assert n >= 1
    assert not any("papa" in i.lower() for i in ings), ings
    assert any("lechosa" in i.lower() for i in ings) and any("yogurt" in i.lower() for i in ings)


def test_drops_garbled_unresolvable_name(monkeypatch):
    import graph_orchestrator as g
    _patch_verified(monkeypatch)
    plan = {"days": [{"day": 1, "meals": [
        {"meal": "Cena", "name": "Pollo con Vegetales",
         "ingredients": ["90g de Pollo", "EsGuineocas", "100g de Vainitas"]}]}]}
    g._generation_sanity_autofix(plan, db=_StubDB())
    ings = plan["days"][0]["meals"][0]["ingredients"]
    assert not any("esguineocas" in i.lower() for i in ings), ings
    assert any("pollo" in i.lower() for i in ings)


def test_drops_multiword_garble_keeps_compound(monkeypatch):
    """[P3-GEN-SANITY-GARBLE-MULTIWORD · 2026-06-27] '75g de Esfresascas frescas' (mash multi-palabra) debe
    dropearse — antes sobrevivía porque solo se atacaba garble de 1 palabra. Un compuesto legítimo con palabra
    corta ('salsa criolla casera', 'salsa'<7) NO debe caer."""
    import graph_orchestrator as g
    _patch_verified(monkeypatch, unverified=("esfresascas", "frescas", "salsa"))
    plan = {"days": [{"day": 1, "meals": [
        {"meal": "Cena", "name": "Pollo",
         "ingredients": ["90g de Pollo", "75g de Esfresascas frescas", "Salsa criolla casera"]}]}]}
    g._generation_sanity_autofix(plan, db=_StubDB())
    ings = [i.lower() for i in plan["days"][0]["meals"][0]["ingredients"]]
    assert not any("esfresascas" in i for i in ings), f"garble multi-palabra debe dropearse: {ings}"
    assert any("salsa criolla" in i for i in ings), f"compuesto legítimo (palabra <7) NO debe caer: {ings}"
    assert any("pollo" in i for i in ings)


def test_normal_meal_untouched(monkeypatch):
    import graph_orchestrator as g
    _patch_verified(monkeypatch)
    plan = {"days": [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Pollo guisado con arroz integral",
         "ingredients": ["120g de Pollo", "50g de Arroz integral", "100g de Ensalada"]}]}]}
    before = list(plan["days"][0]["meals"][0]["ingredients"])
    n = g._generation_sanity_autofix(plan, db=_StubDB())
    assert n == 0
    assert plan["days"][0]["meals"][0]["ingredients"] == before


def test_arroz_in_batido_dropped_but_not_in_almuerzo(monkeypatch):
    # 'arroz' es incongruente en batido (drop) pero válido en almuerzo (keep)
    import graph_orchestrator as g
    _patch_verified(monkeypatch)
    plan = {"days": [{"day": 1, "meals": [
        {"meal": "Merienda", "name": "Batido proteico", "ingredients": ["120g de Yogurt", "40g de Arroz"]},
        {"meal": "Almuerzo", "name": "Bandera", "ingredients": ["50g de Arroz", "120g de Pollo"]}]}]}
    g._generation_sanity_autofix(plan, db=_StubDB())
    batido = plan["days"][0]["meals"][0]["ingredients"]
    almuerzo = plan["days"][0]["meals"][1]["ingredients"]
    assert not any("arroz" in i.lower() for i in batido), batido
    assert any("arroz" in i.lower() for i in almuerzo), almuerzo


def test_never_empties_a_meal(monkeypatch):
    import graph_orchestrator as g
    _patch_verified(monkeypatch, unverified=("todo",))
    plan = {"days": [{"day": 1, "meals": [
        {"meal": "Cena", "name": "x", "ingredients": ["TodoBasura1", "TodoBasura2"]}]}]}
    g._generation_sanity_autofix(plan, db=_StubDB())
    # ambos son glitch pero NO se vacía la comida (keep original)
    assert len(plan["days"][0]["meals"][0]["ingredients"]) >= 1


def test_free_items_never_dropped(monkeypatch):
    import graph_orchestrator as g
    _patch_verified(monkeypatch, unverified=("agua", "limon", "esguineocas"))
    plan = {"days": [{"day": 1, "meals": [
        {"meal": "Cena", "name": "Pescado", "ingredients": ["100g de Pescado", "Agua", "Jugo de limón", "EsGuineocas"]}]}]}
    g._generation_sanity_autofix(plan, db=_StubDB())
    ings = [i.lower() for i in plan["days"][0]["meals"][0]["ingredients"]]
    assert any("agua" in i for i in ings) and any("limon" in i or "limón" in i for i in ings)
    assert not any("esguineocas" in i for i in ings)


def test_batido_keeps_fruit_drops_tuber(monkeypatch):
    # [P1-BARIATRIC-TORONJA] falso positivo cerrado: 'papa'(token) matcheaba 'papaya'. Un batido CONSERVA frutas
    # (papaya/lechosa) y solo dropea tubérculos/granos/huevo (papa = potato).
    import graph_orchestrator as g
    _patch_verified(monkeypatch)
    plan = {"days": [{"day": 1, "meals": [{"meal": "Merienda", "name": "Batido de Lechosa", "ingredients": [
        "27g de lechosa (papaya) madura", "120g de Yogurt", "0.5 papa mediana (75g)"]}]}]}
    g._generation_sanity_autofix(plan, db=_StubDB())
    ings = [i.lower() for i in plan["days"][0]["meals"][0]["ingredients"]]
    assert any("papaya" in i or "lechosa" in i for i in ings), "papaya/lechosa NO debe dropearse de un batido"
    assert not any("papa mediana" in i for i in ings), "la papa (tubérculo) sí debe dropearse del batido"


def test_anchors():
    go = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "P3-GEN-SANITY-AUTOFIX" in go and "def _generation_sanity_autofix" in go
    dg = (_BACKEND / "prompts" / "day_generator.py").read_text(encoding="utf-8")
    assert "ARROZ DE NOCHE" in dg.upper()  # #3 prompt
