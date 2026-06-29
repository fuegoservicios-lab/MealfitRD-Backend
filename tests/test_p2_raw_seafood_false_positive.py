"""[P2-RAW-SEAFOOD-FALSE-POSITIVE · 2026-06-28] El banner food-safety de pescado/carne crudos se disparaba por la palabra
"ceviche" en el NOMBRE sin verificar proteína animal → falso-positivo en "Ceviche de yuca con edamame" (vegetariano). Fix:
"ceviche"/"cebiche" pasan a la rama AMBIGUA (exige proteína animal real, como tartar/carpaccio). sushi/sashimi/tiradito
siguen standalone (siempre pescado crudo). Caso legítimo (ceviche de pescado real) preservado.

Tests PUROS: _scan_raw_seafood_meat_violations (string logic, sin Neon).
"""
from __future__ import annotations

import graph_orchestrator as g


def _scan(name, ingredients, recipe=None):
    meal = {"name": name, "ingredients": ingredients, "recipe": recipe or []}
    return g._scan_raw_seafood_meat_violations({"days": [{"meals": [meal]}]})


def test_ceviche_yuca_vegetariano_no_banner():
    # el bug: ceviche de yuca (sin animal) NO debe disparar el banner de pescado crudo
    assert _scan("Ceviche de yuca con edamame", ["0.5 yuca", "225g edamame", "0.25 taza mango"]) == []


def test_ceviche_pescado_real_si_banner():
    # caso legítimo: ceviche de pescado real SÍ debe avisar
    assert len(_scan("Ceviche de Atún", ["150g atún fresco", "limón"])) == 1
    assert len(_scan("Ceviche de Corvina", ["150g corvina", "cebolla morada"])) == 1


def test_sushi_sashimi_tiradito_siguen_standalone():
    # inequívocamente pescado crudo → flag aunque no se itemice la especie
    assert len(_scan("Sushi variado", ["arroz", "alga nori", "pescado"])) == 1
    assert len(_scan("Sashimi", ["pescado fresco"])) == 1
    assert len(_scan("Tiradito de la casa", ["pescado blanco", "ají amarillo"])) == 1


def test_frases_explicitas_standalone():
    assert len(_scan("Tartar especial", ["100g atún crudo"])) == 1  # "atun crudo" standalone


def test_tartar_vegetal_no_banner():
    # regresión del caso ambiguo existente (no romperlo)
    assert _scan("Tartar de remolacha", ["remolacha", "alcaparras"]) == []


def test_anchor():
    import pathlib
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "P2-RAW-SEAFOOD-FALSE-POSITIVE" in src
    # ceviche fuera del standalone, dentro del ambiguo
    assert g._RAW_PREP_AMBIGUOUS.count("ceviche") == 1
    assert "ceviche" not in g._RAW_SEAFOOD_MEAT_TERMS
    # sushi/sashimi/tiradito SIGUEN standalone
    assert "sushi" in g._RAW_SEAFOOD_MEAT_TERMS and "sashimi" in g._RAW_SEAFOOD_MEAT_TERMS
