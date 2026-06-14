"""[C2-ALLERGEN-GUARD · 2026-06-13] Backstop DETERMINISTA de seguridad de alérgenos sobre
el revisor LLM. Escanea ingredientes vs alergias declaradas (+ sinónimos DD) y rechaza-duro.

Gap clínico encontrado en validación de perfiles (C2): la seguridad de alérgenos dependía
100% del revisor LLM (que puede fallar). Este guard determinista nunca sirve un alérgeno.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_orchestrator import _scan_allergen_violations


def _plan(*ingredient_lists):
    return {"days": [{"day": 1, "meals": [
        {"name": f"Comida {i}", "ingredients": ings} for i, ings in enumerate(ingredient_lists)
    ]}]}


def test_detects_direct_allergen():
    plan = _plan(["150g de pechuga de pollo", "2 cucharadas de mantequilla de maní (30g)"])
    viol = _scan_allergen_violations(plan, ["maní"])
    assert viol, "debió detectar el maní"
    assert any("mani" in v[2] or "mani" in v[1].lower() for v in viol)


def test_detects_via_synonym():
    # Alergia 'lactosa' → debe detectar 'queso' (sinónimo).
    plan = _plan(["100g de queso blanco", "ensalada"])
    viol = _scan_allergen_violations(plan, ["lactosa"])
    assert viol and any("queso" in v[1].lower() for v in viol)


def test_mariscos_synonym():
    plan = _plan(["200g de camarones guisados", "arroz"])
    viol = _scan_allergen_violations(plan, ["mariscos"])
    assert viol and any("camaron" in v[2] for v in viol)


def test_no_violation_when_safe():
    plan = _plan(["150g de pollo", "arroz", "ensalada verde"])
    assert _scan_allergen_violations(plan, ["maní"]) == []


def test_sentinel_ninguna_no_violations():
    plan = _plan(["mantequilla de maní", "queso", "camarones"])
    assert _scan_allergen_violations(plan, ["Ninguna"]) == []
    assert _scan_allergen_violations(plan, []) == []


def test_freetext_allergy_literal_match():
    # Alergia free-text no en el mapa → match literal del término.
    plan = _plan(["100g de fresas frescas", "yogurt"])
    viol = _scan_allergen_violations(plan, ["fresa"])
    assert viol and any("fresa" in v[1].lower() for v in viol)


def test_allergy_mixed_with_sentinel_still_detected():
    # ['Ninguna', 'maní'] → la alergia real se detecta pese al sentinel.
    plan = _plan(["mantequilla de maní (30g)"])
    viol = _scan_allergen_violations(plan, ["Ninguna", "maní"])
    assert viol
