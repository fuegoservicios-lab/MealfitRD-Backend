"""[P2-SOLVER-SEEDER-V3 · 2026-07-07] Audit solver+seeder v3 — regresión de los 5 gaps cerrados.

Cero P0/P1 (la math del solver y el andamiaje de guards clínicos ya eran sólidos tras v1/v2). Los 5
gaps de v3 son "datos incompletos DENTRO de un guard/sanitizer que sí existe" (por eso las refutaciones
estructurales de v2 no los cazaron):

  Sd-P2-d  DM2 GI guard solo bloqueaba FRUTA/azúcar → extendido a almidones alto-IG (papa/yuca/arroz
           blanco/pan blanco), los portadores DOMINANTES de K/vit-C en un plan dominicano.
  Sd-P2-e  Guard renal de calcio solo bloqueaba LÁCTEOS → extendido a portadores NO-lácteos alto-P/K
           (ajonjolí/almendra/espinaca/sardina/tofu).
  S-P2-d   Sanitizer de suplementos borraba el string pero no decrementaba los macros STORED → fantasma
           bajo la coverage-gate (helper `_decrement_stripped_supplement_macros`).
  S-P3-a   Knobs base del clamp/pesos del solver sin validator (sus hermanos sí) → swap fat-finger
           aceptado en silencio → validators + guard de inversión.
  Sd-P3-a  Piso de 25.0 kcal de colocación de semilla hardcodeado → knob MEALFIT_MICRO_SEED_MIN_BUDGET_KCAL.
"""
import importlib
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()
with open(os.path.join(_BACKEND, "portion_solver.py"), encoding="utf-8") as f:
    _PS = f.read()


# ═══════════════════════════ PARSER ANCHORS ═══════════════════════════

def test_markers_present():
    for m in ("P2-DM2-HIGHGI-STARCH-GUARD", "P2-RENAL-CA-NONDAIRY-GUARD",
              "P2-SUPP-STRIP-DECREMENT", "S-P3-a", "Sd-P3-a"):
        assert m in _GO or m in _PS, f"marker {m!r} ausente"


def test_sd_p2_d_tokens_and_branch_anchored():
    i = _GO.index("_DM2_HIGHGI_STARCH_TOKENS = (")
    win = _GO[i:i + 400]
    for t in ("papa", "yuca", "arroz blanco", "pan blanco", "batata"):
        assert t in win, f"almidón alto-IG '{t}' ausente"
    # word-boundary OBLIGATORIO (lección 'res'↔'fresas' / 'papa'⊄'papaya')
    assert "_dm2_starch_re = _re.compile(" in win and r")s?\b" in win
    body = _GO[_GO.index("def _ceiling_risky_contributor"):][:1400]
    assert "DM2_HIGHGI_STARCH_GUARD and _dm2_starch_re.search(_ing_low)" in body


def test_sd_p2_e_tokens_and_branch_anchored():
    i = _GO.index("_RENAL_CA_NONDAIRY_TOKENS = (")
    win = _GO[i:i + 400]
    for t in ("ajonjoli", "almendra", "espinaca", "sardina", "tofu"):
        assert t in win, f"portador Ca no-lácteo '{t}' ausente"
    assert "_renal_ca_nondairy_re = _re.compile(" in win and r")s?\b" in win
    body = _GO[_GO.index("def _ceiling_risky_contributor"):][:1400]
    assert "RENAL_CALCIUM_NONDAIRY_GUARD" in body and "_renal_ca_nondairy_re.search(_ing_low)" in body


def test_s_p2_d_helper_and_gate_anchored():
    assert "def _decrement_stripped_supplement_macros(" in _GO
    # el sanitizer llama al helper, gateado por el knob
    i = _GO.index("if not form_data.get(\"includeSupplements\"):")
    win = _GO[i:i + 2500]
    assert "SUPP_STRIP_DECREMENT_MACROS" in win
    assert "_decrement_stripped_supplement_macros(" in win
    assert "_removed_lines" in win


def test_s_p3_a_validators_anchored():
    # los knobs base del clamp/pesos ahora pasan validator (antes _envf(name, default) pelado)
    assert 'MEALFIT_SOLVER_MIN_SCALE", 0.3, lambda v:' in _PS
    assert 'MEALFIT_SOLVER_MAX_SCALE", 3.5, lambda v:' in _PS
    assert 'MEALFIT_SOLVER_MAX_SCALE_PROTEIN", 5.0, lambda v:' in _PS
    assert 'MEALFIT_SOLVER_W_KCAL", 1.2, lambda v:' in _PS
    assert 'MEALFIT_SOLVER_LSQ_REG", 0.10, lambda v:' in _PS
    # guard de inversión + fallback fallback local acepta validator
    assert "if not (SOLVER_MIN_SCALE < SOLVER_MAX_SCALE):" in _PS
    assert "def _envf(name: str, default: float, validator=None)" in _PS


def test_sd_p3_a_knob_replaces_literal():
    assert 'MEALFIT_MICRO_SEED_MIN_BUDGET_KCAL", 25' in _GO
    # ningún literal 25.0 pelado sobrevive en los gates de siembra
    assert "kcal_budget_left <= 25.0" not in _GO
    assert "kcal_budget_left > 25.0" not in _GO
    assert "_fat_seed_reserve > 25.0" not in _GO
    # ambos gates usan el knob
    seg = _GO[_GO.index("_seed_from_reserve = (k in _FAT_BASED_MICROS"):][:600]
    assert seg.count("MICRO_SEED_MIN_BUDGET_KCAL") >= 3


# ═══════════════════════════ FUNCIONAL: micro-closer ═══════════════════════════

def _grams_of(low: str) -> float:
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", low)
    return float(m.group(1).replace(",", ".")) if m else 0.0


class _FakeDB:
    """densities: {token: {micro_key: per_g}}. Resuelve por substring del string."""

    def __init__(self, densities):
        self.densities = densities

    def micros_from_ingredient_string(self, s):
        low = str(s).lower()
        g = _grams_of(low)
        for tok, mic in self.densities.items():
            if tok in low:
                return {k: v * g for k, v in mic.items()}
        return {}

    def macros_from_ingredient_string(self, s):
        return {"kcal": 10.0}

    def grams_from_ingredient_string(self, s):
        return _grams_of(str(s).lower())


@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "MICRONUTRIENT_CLOSER_ENABLED", True)
    monkeypatch.setattr(g, "MICRO_CLOSER_PERDAY_ENABLED", False)
    monkeypatch.setattr(g, "MICRO_SEED_ENABLED", False)
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


def _mk_plan(ingredients):
    return {"macros": {"fats": "60g"}, "days": [{"day": 1, "meals": [
        {"meal": "Almuerzo", "name": "Bowl", "cals": 500,
         "ingredients": list(ingredients),
         "ingredients_raw": list(ingredients),
         "recipe": ["Sirve."]},
    ]}]}


def _run(go, monkeypatch, plan, conditions, micro, floor):
    import micronutrients
    monkeypatch.setattr(micronutrients, "build_micronutrient_report", lambda *a, **kw: {
        "panel": [{"key": micro, "piso": floor, "valor": 0.0, "status": "bajo"}],
        "gaps": [{"key": micro, "piso": floor, "status": "bajo"}],
        "coverage": 1.0, "per_day_floors": {"flagged": False}})
    return go._close_micro_gaps_for_plan(plan, {"medicalConditions": conditions}, _FakeDB(_DENS))


# densidades compartidas (per gramo)
_DENS = {
    "papa": {"potassium_mg": 4.0},
    "papaya": {"potassium_mg": 4.0},
    "espinaca": {"potassium_mg": 2.0, "calcium_mg": 2.0},
    "ajonjoli": {"calcium_mg": 10.0},
}


def _grams(plan, food):
    line = next(s for s in plan["days"][0]["meals"][0]["ingredients"] if food in s)
    return float(re.match(r"^\s*(\d+(?:[.,]\d+)?)", line).group(1).replace(",", "."))


# ---- Sd-P2-d: DM2 almidón alto-IG ----

def test_sd_p2_d_dm2_skips_starch_scales_alternative(go, monkeypatch):
    plan = _mk_plan(["100 g de papa hervida", "100 g de espinaca"])
    n = _run(go, monkeypatch, plan, ["Diabetes tipo 2"], "potassium_mg", 900.0)
    assert n >= 1
    assert _grams(plan, "papa") == 100.0, "almidón alto-IG jamás es target de escalado en DM2"
    assert _grams(plan, "espinaca") > 100.0, "la espinaca (no-IG) toma el lugar richest-first"


def test_sd_p2_d_non_dm2_scales_starch_as_before(go, monkeypatch):
    plan = _mk_plan(["100 g de papa hervida", "100 g de espinaca"])
    _run(go, monkeypatch, plan, ["Ninguna"], "potassium_mg", 900.0)
    assert _grams(plan, "papa") > 100.0, "sin DM2 el richest-first (papa) se escala normal"


def test_sd_p2_d_wordboundary_papaya_not_blocked(go, monkeypatch):
    # 'papaya' NO es un almidón: '\\bpapa\\b' no debe false-matchear → sigue escalable en DM2
    plan = _mk_plan(["100 g de papaya", "100 g de espinaca"])
    _run(go, monkeypatch, plan, ["Diabetes tipo 2"], "potassium_mg", 900.0)
    assert _grams(plan, "papaya") > 100.0, "'papaya' no debe caer en el guard de almidón (word-boundary)"


def test_sd_p2_d_knob_off_reverts(go, monkeypatch):
    monkeypatch.setattr(go, "DM2_HIGHGI_STARCH_GUARD", False)
    plan = _mk_plan(["100 g de papa hervida", "100 g de espinaca"])
    _run(go, monkeypatch, plan, ["Diabetes tipo 2"], "potassium_mg", 900.0)
    assert _grams(plan, "papa") > 100.0, "knob OFF → comportamiento previo (papa escalable)"


# ---- Sd-P2-e: renal calcio no-lácteo ----

def test_sd_p2_e_renal_skips_nondairy_ca(go, monkeypatch):
    plan = _mk_plan(["20 g de ajonjoli"])
    _run(go, monkeypatch, plan, ["Enfermedad renal crónica"], "calcium_mg", 900.0)
    assert _grams(plan, "ajonjoli") == 20.0, "portador de Ca no-lácteo alto-P jamás se escala en ERC"


def test_sd_p2_e_non_renal_scales_nondairy_ca(go, monkeypatch):
    plan = _mk_plan(["20 g de ajonjoli"])
    _run(go, monkeypatch, plan, ["Ninguna"], "calcium_mg", 900.0)
    assert _grams(plan, "ajonjoli") > 20.0, "sin ERC el ajonjolí es un portador de Ca legítimo"


def test_sd_p2_e_knob_off_reverts(go, monkeypatch):
    monkeypatch.setattr(go, "RENAL_CALCIUM_NONDAIRY_GUARD", False)
    plan = _mk_plan(["20 g de ajonjoli"])
    _run(go, monkeypatch, plan, ["Enfermedad renal crónica"], "calcium_mg", 900.0)
    assert _grams(plan, "ajonjoli") > 20.0, "knob OFF → dairy-only (comportamiento previo)"


# ═══════════════════════════ FUNCIONAL: S-P2-d helper ═══════════════════════════

_SUPP_KWS = ("proteína en polvo", "proteina en polvo", "whey", "creatina")


class _SuppDB:
    def __init__(self, resolves=True):
        self.resolves = resolves

    def macros_from_ingredient_string(self, s):
        low = str(s).lower()
        if any(k in low for k in ("proteína en polvo", "proteina en polvo", "whey")):
            return {"protein": 20.0, "carbs": 2.0, "fats": 1.0, "kcal": 95.0} if self.resolves else {}
        return {}


def test_s_p2_d_decrements_stored_macros_and_filters_raw():
    import graph_orchestrator as g
    meal = {"protein": 45, "carbs": 60, "fats": 12, "cals": 500,
            "ingredients_raw": ["1 plato de sancocho", "1 scoop de proteína en polvo (25g)"],
            "macros": ["P:45g", "C:60g", "G:12g"]}
    unresolved = g._decrement_stripped_supplement_macros(
        meal, ["1 scoop de proteína en polvo (25g)"], _SuppDB(resolves=True), _SUPP_KWS)
    assert unresolved == 0
    assert meal["protein"] == 25 and meal["cals"] == 405
    assert meal["carbs"] == 58 and meal["fats"] == 11
    assert not any("proteína en polvo" in r.lower() for r in meal["ingredients_raw"])
    assert meal["macros"] == ["P:25g", "C:58g", "G:11g"]


def test_s_p2_d_unresolved_supplement_failsafe():
    import graph_orchestrator as g
    meal = {"protein": 45, "carbs": 60, "fats": 12, "cals": 500,
            "ingredients_raw": ["1 plato de sancocho", "1 scoop de whey (25g)"]}
    unresolved = g._decrement_stripped_supplement_macros(
        meal, ["1 scoop de whey (25g)"], _SuppDB(resolves=False), _SUPP_KWS)
    assert unresolved == 1
    # no resuelve → no resta (fail-safe: no empeora), pero SÍ filtra el raw
    assert meal["protein"] == 45 and meal["cals"] == 500
    assert not any("whey" in r.lower() for r in meal["ingredients_raw"])


def test_s_p2_d_clamps_at_zero():
    import graph_orchestrator as g
    meal = {"protein": 10, "carbs": 0, "fats": 0, "cals": 40, "ingredients_raw": []}
    g._decrement_stripped_supplement_macros(
        meal, ["1 scoop de whey (25g)"], _SuppDB(resolves=True), _SUPP_KWS)
    assert meal["protein"] == 0 and meal["cals"] == 0, "clamp a 0, jamás negativo"


# ═══════════════════════════ FUNCIONAL: S-P3-a validators ═══════════════════════════

def _reload_solver_with_env(**env):
    """Recarga portion_solver con env overrides, restaurando os.environ + módulo limpio al final."""
    import portion_solver
    _orig = dict(os.environ)
    try:
        for k, v in env.items():
            os.environ[k] = v
        importlib.reload(portion_solver)
        return (portion_solver.SOLVER_MIN_SCALE, portion_solver.SOLVER_MAX_SCALE,
                portion_solver.SOLVER_MAX_SCALE_PROTEIN)
    finally:
        os.environ.clear()
        os.environ.update(_orig)
        importlib.reload(portion_solver)  # restaura estado limpio del módulo


def test_s_p3_a_swap_falls_back_to_defaults():
    mn, mx, _ = _reload_solver_with_env(
        MEALFIT_SOLVER_MIN_SCALE="3.5", MEALFIT_SOLVER_MAX_SCALE="0.3")
    assert mn == 0.3 and mx == 3.5, "swap fat-finger → validators caen a defaults"


def test_s_p3_a_negative_max_falls_back():
    _, mx, _ = _reload_solver_with_env(MEALFIT_SOLVER_MAX_SCALE="-2")
    assert mx == 3.5, "MAX negativo fuera de rango → default"


def test_s_p3_a_protein_below_general_normalized():
    _, mx, prot = _reload_solver_with_env(
        MEALFIT_SOLVER_MAX_SCALE="3.5", MEALFIT_SOLVER_MAX_SCALE_PROTEIN="2.0")
    assert prot == mx == 3.5, "proteína < general → igualada al general (nunca escala menos)"


def test_s_p3_a_happy_path_unchanged():
    mn, mx, prot = _reload_solver_with_env(
        MEALFIT_SOLVER_MIN_SCALE="0.4", MEALFIT_SOLVER_MAX_SCALE="4.0",
        MEALFIT_SOLVER_MAX_SCALE_PROTEIN="6.0")
    assert (mn, mx, prot) == (0.4, 4.0, 6.0), "valores válidos pasan sin tocar"


# ═══════════════════════════ FUNCIONAL: Sd-P3-a knob ═══════════════════════════

def test_sd_p3_a_knob_registered_default_25():
    import graph_orchestrator as g
    assert g.MICRO_SEED_MIN_BUDGET_KCAL == 25
    from knobs import _KNOBS_REGISTRY
    assert any("MEALFIT_MICRO_SEED_MIN_BUDGET_KCAL" in str(k) for k in _KNOBS_REGISTRY), \
        "el knob debe auto-registrarse en _KNOBS_REGISTRY (visible en /health/version)"
