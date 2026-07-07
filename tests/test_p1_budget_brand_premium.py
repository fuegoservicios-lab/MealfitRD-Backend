"""[P1-BUDGET-BRAND-PREMIUM · 2026-07-07] Decisión de producto del owner (ablanda levemente P2-H): en
el banner 'excedido', surfacear cuánto CUESTA la elección de marcas premium del usuario vs la más
económica, como UN total accionable. Contexto: el review del plan 4e7b8dbb (24% sobre presupuesto)
reveló que el sobrecosto es en gran parte las preferencias de marca premium del usuario (Quaker avena
RD$99 vs Wala RD$47, Jif maní vs genérico...). P2-H excluía esos ítems de las sugerencias para no
molestar; ahora se acumula el premium y se muestra UN resumen (sin nag per-ítem).
"""
import os

import shopping_calculator as sc

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "shopping_calculator.py"), encoding="utf-8") as f:
    _SC = f.read()


def test_variant_price_per_g_both_formats():
    assert abs(sc._variant_price_per_g({"grams": 618, "price": 99}) - 99 / 618) < 1e-6
    assert abs(sc._variant_price_per_g(
        {"presentation": "Funda 650 gr", "price_rd": 47}) - 47 / 650) < 1e-6
    assert sc._variant_price_per_g({"grams": 0, "price": 10}) is None
    assert sc._variant_price_per_g(None) is None


def _wire(monkeypatch, pref_food="avena"):
    # el usuario tiene pref de marca para `pref_food` (Quaker 618g RD$99); lo más barato Wala 650g RD$47
    monkeypatch.setattr(sc, "fetch_brand_pref_packages", lambda uid: {pref_food: {"grams": 618, "price": 99}})
    monkeypatch.setattr(sc, "_resolve_brand_pref",
                        lambda name, prefs: {"grams": 618, "price": 99} if pref_food in name.lower() else None)
    monkeypatch.setattr(sc, "cheapest_supermarket_variant",
                        lambda name: ({"brand": "Wala", "presentation": "Funda 650 gr", "price_rd": 47}
                                      if pref_food in name.lower()
                                      else {"brand": "Genérico", "presentation": "Paq 1lb", "price_rd": 40}))


def test_brand_premium_total_surfaced(monkeypatch):
    monkeypatch.setattr(sc, "_budget_brand_premium_surface_enabled", lambda: True)
    _wire(monkeypatch)
    weekly = [{"name": "Avena", "estimated_cost_rd": 198}]
    sugs = sc.build_budget_suggestions(weekly, user_id="u1")
    prem = [s for s in sugs if s.get("type") == "marca_premium_total"]
    assert prem, f"esperado un resumen marca_premium_total, dio: {sugs}"
    # 198 * (1 - (47/650)/(99/618)) = 198 * 0.549 ≈ 109
    assert 95 <= prem[0]["saving_rd"] <= 120, prem[0]["saving_rd"]
    assert "marcas premium" in prem[0]["text"].lower()


def test_no_per_item_nag_for_preferred(monkeypatch):
    """P2-H preservado: NO hay sugerencia per-ítem 'marca' para el ítem con pref (solo el total)."""
    monkeypatch.setattr(sc, "_budget_brand_premium_surface_enabled", lambda: True)
    _wire(monkeypatch)
    sugs = sc.build_budget_suggestions([{"name": "Avena", "estimated_cost_rd": 198}], user_id="u1")
    per_item = [s for s in sugs if s.get("type") == "marca" and "avena" in s.get("item", "").lower()]
    assert not per_item, "no debe haber nag per-ítem para la marca elegida (P2-H)"


def test_knob_off_no_premium(monkeypatch):
    monkeypatch.setattr(sc, "_budget_brand_premium_surface_enabled", lambda: False)
    _wire(monkeypatch)
    sugs = sc.build_budget_suggestions([{"name": "Avena", "estimated_cost_rd": 198}], user_id="u1")
    assert not [s for s in sugs if s.get("type") == "marca_premium_total"]


def test_non_preferred_item_still_gets_marca_suggestion(monkeypatch):
    """Ítem SIN pref → sigue recibiendo la sugerencia normal de marca más económica (sin cambio)."""
    monkeypatch.setattr(sc, "_budget_brand_premium_surface_enabled", lambda: True)
    monkeypatch.setattr(sc, "fetch_brand_pref_packages", lambda uid: {})  # sin prefs
    monkeypatch.setattr(sc, "_resolve_brand_pref", lambda name, prefs: None)
    monkeypatch.setattr(sc, "cheapest_supermarket_variant",
                        lambda name: {"brand": "Wala", "presentation": "Funda 650 gr", "price_rd": 47})
    sugs = sc.build_budget_suggestions([{"name": "Cangrejo", "estimated_cost_rd": 599}], user_id="u1")
    assert [s for s in sugs if s.get("type") == "marca"], "ítem sin pref conserva la sugerencia normal"


def test_marker_anchored():
    assert "P1-BUDGET-BRAND-PREMIUM" in _SC
    assert 'MEALFIT_BUDGET_BRAND_PREMIUM_SURFACE' in _SC
