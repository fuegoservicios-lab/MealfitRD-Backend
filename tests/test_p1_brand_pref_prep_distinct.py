"""[P1-BRAND-PREF-PREP-DISTINCT · 2026-07-10] La escalera de preferencias de marca cruzaba
alimento BASE ↔ producto DERIVADO por la contención bidireccional (:2519): el usuario eligió
Jif Cremosa para 'Mantequilla de maní' y la lista de compras se lo aplicó TAMBIÉN al ítem 'Maní'
(maní entero tostado) — visto en el plan vivo 6d742f23: "Maní → 1 pote (Cremosa 16 Oz · Jif)
RD$295" cuando el catálogo tiene 19 SKUs propios de Maní (pote Wala 300g RD$185).

Es la lección P1-NUT-BUTTER-DISTINCT / P1-PREP-COLLAPSE-GUARD aplicada al brand matcher: una
PREPARACIÓN ("<prep> de <X>") es un producto DISTINTO de su base <X> — jamás cross-match, en
NINGUNA dirección. La contención legítima (pref 'arroz' → ítem 'arroz blanco premium') se conserva.

tooltip-anchor: P1-BRAND-PREF-PREP-DISTINCT
"""
import shopping_calculator as sc

_PKG_JIF = [{"label": "Cremosa 16 Oz · Jif", "price": 295, "grams": 454}]
_PKG_WALA = [{"label": "300 g · Wala", "price": 185, "grams": 300}]
_PKG_ARROZ = [{"label": "1 Lb · Campos", "price": 55, "grams": 454}]


def test_butter_pref_does_not_apply_to_whole_peanuts():
    prefs = {"mantequilla de mani": _PKG_JIF}
    assert sc._resolve_brand_pref("Maní", prefs) is None          # el caso vivo del plan 6d742f23


def test_butter_pref_still_applies_to_butter():
    prefs = {"mantequilla de mani": _PKG_JIF}
    assert sc._resolve_brand_pref("Mantequilla de maní", prefs) == _PKG_JIF


def test_peanut_pref_does_not_apply_to_butter():
    prefs = {"mani": _PKG_WALA}
    assert sc._resolve_brand_pref("Mantequilla de maní", prefs) is None  # dirección inversa


def test_vinegar_vs_apple_same_class():
    prefs = {"vinagre de manzana": _PKG_JIF}
    assert sc._resolve_brand_pref("Manzana", prefs) is None


def test_legit_containment_preserved():
    prefs = {"arroz": _PKG_ARROZ}
    assert sc._resolve_brand_pref("Arroz blanco premium", prefs) == _PKG_ARROZ


def test_default_resolver_also_guarded():
    defaults = {"mantequilla de mani": _PKG_JIF}
    assert sc._resolve_brand_default("Maní", defaults) is None
