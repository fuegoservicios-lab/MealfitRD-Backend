"""[P0-ALLERGEN-SUBS · 2026-06-14] Sustitución QUIRÚRGICA determinista de alérgenos IgE.

Cierra el gap del audit clínico (2026-06-14): las alergias declaradas dependían del prompt + el
backstop romo `_scan_allergen_violations` (que NUKE el plan entero a fallback). Este P0 hace
quirúrgica la defensa: para los alérgenos con un reemplazo seguro que RESUELVE al catálogo es-DO
(fish/shellfish/soy/gluten) sustituye el ingrediente ofensor in-place conservando el plan rico.
`_scan_allergen_violations` queda como red de seguridad post-swap (cero regresión).

Cubre:
  A. `condition_rules.collect_allergen_substitutions` (detección + tablas; honesto: lácteos/huevo/
     maní/frutos secos NO se sustituyen).
  B. `graph_orchestrator._apply_allergen_substitutions` (swap, narrowness, negativos, preserve_qty,
     idempotencia, knob off).
  C. Regresión: `_apply_condition_substitutions` sigue funcionando tras extraer el motor compartido.
  D. Integración: `_apply_deterministic_clinical_layer` ejecuta el guard de alérgenos.
  E. Parser-based anchors (renombre falla el test antes de cambiar producción).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import condition_rules as cr

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GO_PATH = _BACKEND_ROOT / "graph_orchestrator.py"
_CR_PATH = _BACKEND_ROOT / "condition_rules.py"


def _meal(ingredients, **extra):
    m = {"name": "Almuerzo", "ingredients": list(ingredients),
         "protein": 30, "carbs": 40, "fats": 10, "cals": 400, "recipe": ["Cocina la proteína"]}
    m.update(extra)
    return m


def _plan(ingredients):
    return {"days": [{"meals": [_meal(ingredients)]}]}


def _ings(plan):
    return [str(i).lower() for i in plan["days"][0]["meals"][0]["ingredients"]]


# ════════════════════════════════════════════════════════════════════════════════════════════════
# A. collect_allergen_substitutions — detección + tablas (puro, sin importar graph_orchestrator)
# ════════════════════════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("allergy,expected_cat,expected_repl", [
    ("Pescado", "allergen:fish", "Pechuga de pollo"),
    ("alergia al pescado", "allergen:fish", "Pechuga de pollo"),
    ("Mariscos", "allergen:shellfish", "Pechuga de pollo"),
    ("camarones", "allergen:shellfish", "Pechuga de pollo"),
    ("Soya", "allergen:soy", "Pechuga de pollo"),
    ("tofu", "allergen:soy", "Pechuga de pollo"),
    ("Gluten", "allergen:gluten", "Casabe"),
    ("trigo", "allergen:gluten", "Casabe"),
    ("celiaquía", "allergen:gluten", "Casabe"),
])
def test_declared_allergy_yields_subs(allergy, expected_cat, expected_repl):
    subs = cr.collect_allergen_substitutions({"allergies": [allergy]})
    assert subs, f"alergia {allergy!r} no produjo sustituciones"
    cats = {s["condition"] for s in subs}
    assert expected_cat in cats
    repls = {s["replacement"] for s in subs}
    assert expected_repl in repls


@pytest.mark.parametrize("allergy", ["lácteos", "leche", "huevo", "maní", "frutos secos", "nueces"])
def test_unsupported_allergens_not_substituted(allergy):
    """DECISIÓN HONESTA: lácteos/huevo/maní/frutos secos NO se sustituyen (sin target que resuelva
    en el catálogo es-DO) → siguen por el path crítico→fallback. Si esto cambia (se añaden filas de
    leche/queso vegetal al catálogo), actualizar la decisión documentada en condition_rules.py."""
    assert cr.collect_allergen_substitutions({"allergies": [allergy]}) == []


@pytest.mark.parametrize("sentinel", ["Ninguna", "ninguna", "", "N/A", "sin alergias"])
def test_sentinels_yield_no_subs(sentinel):
    assert cr.collect_allergen_substitutions({"allergies": [sentinel]}) == []


def test_no_allergies_key_yields_no_subs():
    assert cr.collect_allergen_substitutions({}) == []
    assert cr.collect_allergen_substitutions({"allergies": []}) == []


def test_multiple_allergies_multiple_categories():
    subs = cr.collect_allergen_substitutions({"allergies": ["pescado", "gluten"]})
    cats = {s["condition"] for s in subs}
    assert "allergen:fish" in cats and "allergen:gluten" in cats


def test_gluten_subs_carry_negatives():
    subs = cr.collect_allergen_substitutions({"allergies": ["gluten"]})
    gluten = [s for s in subs if s["condition"] == "allergen:gluten"]
    assert gluten and all("de maiz" in s["negatives"] and "casabe" in s["negatives"] for s in gluten)


def test_sub_shape_matches_condition_subs():
    """Mismo shape que collect_substitutions → reusable por el motor compartido."""
    subs = cr.collect_allergen_substitutions({"allergies": ["pescado"]})
    keys = set(subs[0].keys())
    assert {"tokens", "replacement", "label", "negatives", "condition", "preserve_qty"} <= keys


# ════════════════════════════════════════════════════════════════════════════════════════════════
# B. _apply_allergen_substitutions — swap funcional (importa graph_orchestrator)
# ════════════════════════════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


def test_fish_swapped_to_chicken(go):
    plan = _plan(["Filete de pescado blanco", "Arroz blanco", "Ensalada"])
    n = go._apply_allergen_substitutions(plan, {"allergies": ["pescado"]})
    assert n == 1
    ings = _ings(plan)
    assert not any("pescado" in i for i in ings), ings
    assert any("pollo" in i for i in ings), ings


def test_shellfish_swapped_to_chicken(go):
    plan = _plan(["Camarones al ajillo", "Arroz blanco"])
    go._apply_allergen_substitutions(plan, {"allergies": ["mariscos"]})
    ings = _ings(plan)
    assert not any("camaron" in i for i in ings)
    assert any("pollo" in i for i in ings)


def test_soy_tofu_to_chicken_and_sauce_to_lemon(go):
    plan = _plan(["Tofu salteado", "Salsa de soya baja en sodio", "Vegetales"])
    go._apply_allergen_substitutions(plan, {"allergies": ["soya"]})
    ings = _ings(plan)
    assert not any("tofu" in i for i in ings)
    assert not any("soya" in i for i in ings)
    assert any("pollo" in i for i in ings)
    assert any("limon" in i or "limón" in i for i in ings)


@pytest.mark.parametrize("orig,token_gone,repl_token", [
    ("Pan de agua", "pan de agua", "casabe"),
    ("Harina de trigo", "harina de trigo", "maíz"),
    ("Espagueti a la boloñesa", "espagueti", "arroz"),
    ("Galletas de soda", "galletas de soda", "galletas de arroz"),
    # [live-fix] el revisor médico rechaza avena por contaminación cruzada → swap a Quinoa (GF)
    ("0.25 taza de avena", "avena", "quinoa"),
    ("Bowl Proteico de Avena con Chinola", "avena", "quinoa"),
])
def test_gluten_staples_swapped(go, orig, token_gone, repl_token):
    plan = _plan([orig, "Vegetales"])
    go._apply_allergen_substitutions(plan, {"allergies": ["gluten"]})
    ings = _ings(plan)
    assert not any(token_gone in i for i in ings), ings
    assert any(repl_token in i for i in ings), ings


def test_narrow_tokens_do_not_false_positive_breadfruit(go):
    """'pana' (fruta de pan) NO debe swappear bajo alergia a gluten (lección bug 'soya'/'pana')."""
    plan = _plan(["Pana hervida", "Pollo guisado"])
    n = go._apply_allergen_substitutions(plan, {"allergies": ["gluten"]})
    assert n == 0
    assert any("pana" in i for i in _ings(plan))


def test_gluten_free_variant_vetoed(go):
    """'Pan sin gluten' NO se swappea (negativo 'sin gluten')."""
    plan = _plan(["Pan sin gluten", "Huevo"])
    go._apply_allergen_substitutions(plan, {"allergies": ["gluten"]})
    assert any("pan sin gluten" in i for i in _ings(plan))


def test_preserve_quantity_prefix(go):
    plan = _plan(["150g de filete de pescado", "Arroz"])
    go._apply_allergen_substitutions(plan, {"allergies": ["pescado"]})
    ings = plan["days"][0]["meals"][0]["ingredients"]
    assert any(i.startswith("150g de") and "pollo" in i.lower() for i in ings), ings


def test_dairy_allergy_leaves_plan_untouched(go):
    """Lácteos NO se sustituyen → el plan no cambia (sigue por el path crítico→fallback)."""
    plan = _plan(["Queso blanco", "Leche", "Yogurt griego sin azúcar"])
    n = go._apply_allergen_substitutions(plan, {"allergies": ["lácteos"]})
    assert n == 0
    assert _ings(plan) == ["queso blanco", "leche", "yogurt griego sin azúcar"]


def test_knob_off_disables_swap(go, monkeypatch):
    monkeypatch.setattr(go, "ALLERGEN_SUBSTITUTION_ENABLED", False)
    plan = _plan(["Filete de pescado blanco", "Arroz"])
    n = go._apply_allergen_substitutions(plan, {"allergies": ["pescado"]})
    assert n == 0
    assert any("pescado" in i for i in _ings(plan))


def test_idempotent(go):
    plan = _plan(["Filete de pescado blanco", "Arroz"])
    go._apply_allergen_substitutions(plan, {"allergies": ["pescado"]})
    snapshot = _ings(plan)
    n2 = go._apply_allergen_substitutions(plan, {"allergies": ["pescado"]})
    assert n2 == 0
    assert _ings(plan) == snapshot


def test_no_allergy_no_swap(go):
    plan = _plan(["Filete de pescado blanco", "Arroz"])
    n = go._apply_allergen_substitutions(plan, {"allergies": ["Ninguna"]})
    assert n == 0
    assert any("pescado" in i for i in _ings(plan))


# ════════════════════════════════════════════════════════════════════════════════════════════════
# C. Regresión: el motor compartido NO rompió las sustituciones por condición
# ════════════════════════════════════════════════════════════════════════════════════════════════
def test_condition_dm2_sugar_still_substituted(go):
    plan = _plan(["1 cucharada de miel", "Avena", "Fresas"])
    n = go._apply_condition_substitutions(plan, {"medicalConditions": ["Diabetes tipo 2"]})
    assert n == 1
    ings = _ings(plan)
    assert not any("miel" in i for i in ings)
    assert any("stevia" in i for i in ings)
    assert plan["days"][0]["meals"][0].get("_dm2_sugar_fixed")  # flag compat preservado


def test_condition_hta_sodium_still_substituted(go):
    plan = _plan(["100g de longaniza dominicana", "Arroz", "Habichuelas"])
    n = go._apply_condition_substitutions(plan, {"medicalConditions": ["Hipertensión"]})
    assert n == 1
    ings = _ings(plan)
    assert not any("longaniza" in i for i in ings)
    assert any("pollo" in i for i in ings)


# ════════════════════════════════════════════════════════════════════════════════════════════════
# D. Integración: la capa clínica determinista corre el guard de alérgenos
# ════════════════════════════════════════════════════════════════════════════════════════════════
def test_clinical_layer_applies_allergen_swap(go):
    plan = _plan(["Filete de pescado blanco", "Arroz blanco", "Ensalada verde"])
    out = go._apply_deterministic_clinical_layer(
        plan, {"allergies": ["pescado"], "medicalConditions": ["Ninguna"]}, {})
    ings = [str(i).lower() for i in out["days"][0]["meals"][0]["ingredients"]]
    assert not any("pescado" in i for i in ings), ings
    assert any("pollo" in i for i in ings), ings
    assert out.get("_clinical_layer_applied") is True


# ════════════════════════════════════════════════════════════════════════════════════════════════
# E. Parser-based anchors (un renombre falla el test ANTES de cambiar producción)
# ════════════════════════════════════════════════════════════════════════════════════════════════
def test_anchor_shared_core_and_allergen_fn_present():
    src = _GO_PATH.read_text(encoding="utf-8")
    assert "def _apply_substitutions_core(" in src
    assert "def _apply_allergen_substitutions(" in src
    assert "def _apply_condition_substitutions(" in src


def test_anchor_guard_wired_in_clinical_layer():
    src = _GO_PATH.read_text(encoding="utf-8")
    assert "ALLERGEN_SUBSTITUTION_ENABLED" in src
    assert "_apply_allergen_substitutions(plan, form_data)" in src
    assert 'MEALFIT_ALLERGEN_SUBSTITUTION' in src


def test_anchor_collect_fn_in_condition_rules():
    src = _CR_PATH.read_text(encoding="utf-8")
    assert "def collect_allergen_substitutions(" in src
    assert "_ALLERGEN_SUBS_BY_CAT" in src
    assert "P0-ALLERGEN-SUBS" in src


def test_anchor_marker_in_app_py():
    # [drift-fix · 2026-06-15] Antes hardcodeaba `"P0-ALLERGEN-SUBS" in marker` (cierto SOLO cuando este
    # fix era el último mergeado). El marker `_LAST_KNOWN_PFIX` avanza con CADA P-fix → la aserción se
    # volvió stale por diseño. La presencia del CÓDIGO P0-ALLERGEN-SUBS ya la ancla
    # `test_anchor_collect_fn_in_condition_rules`; la freshness/formato del marker viven en
    # `test_p3_1_last_known_pfix_freshness`. Aquí solo verificamos que el marker exista y tenga el
    # formato `Pn-... · YYYY-MM-DD` (sanity, no un valor específico que se vuelve obsoleto).
    text = (_BACKEND_ROOT / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*["\']([^"\']+)["\']', text)
    assert m, "_LAST_KNOWN_PFIX no encontrado en app.py"
    assert re.search(r"·\s*\d{4}-\d{2}-\d{2}\s*$", m.group(1)), (
        f"marker sin fecha bien formada (`Pn-... · YYYY-MM-DD`): {m.group(1)!r}")
