"""[P3-SOLVER-SEEDER-V4 · 2026-07-29] Los P3 del audit solver+seeder v4 (pulido / observabilidad).

Solver / refinador:
- P3-REFINE-WEIGHTS-KNOBS: los últimos pesos hardcodeados del motor de precisión → knobs.
- P3-REFINE-EXEMPT-BOUNDARY: `"sal"` ⊂ salmón/ensalada congelaba proteína, y los portadores que el
  micro-closer siembra quedaban movibles. 13ª mordida de la clase 'res'⊂'fresco'.
- P3-REFINE-OBSERVABILITY: el refinador agotaba `max_iters` en silencio.
- P3-SOLVER-CONVERGED-BAND: `converged` no era comparable con la banda que alimenta (+ per-macro).
- P3-SOLVER-FEASIBILITY: distingue "falta escalado" de "falta un PORTADOR".
- P3-SOLVER-WIRING-METRICS: la abstención deja rastro y sale serie propia `solver_convergence`.

Closer:
- P3-CLOSER-LINE-CLAMP-RAW-FALLBACK: el techo vitalicio se medía solo sobre display y se
  auto-desactivaba; ahora cae a raw y, sin gramos, acota por RATIO vitalicio.
- P3-SEED-REINFORCE-FOOD-MATCH: el refuerzo buscaba la forma pre-humanize (letra muerta en las
  superficies persistidas, que son justo donde la erosión ocurre).
- P3-MICRO-SEED-BUDGET-EXACT: el gate comparaba contra un piso fijo y debitaba el costo real.
- P3-MICRO-SEED-CEILING-GUARD: el candidato de siembra pasa por el guard clínico + UL.
- P3-CLOSER-PCOS-GLYCEMIC-GUARD: SOP/PCOS hereda el guard alto-IG de DM2.

Prompt / tests:
- P3-DAYGEN-SLOT-TARGETS: cuota por slot en el prompt del day-gen (OFF, nodo más caro).
- P3-REFINER-LOCKSTEP-FIXTURES / P3-REFINER-ASSEMBLE-ANCHOR: en test_p1_next_level_batch.py.
"""
from __future__ import annotations

import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel: str) -> str:
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as f:
        return f.read()


_PS = _read("portion_solver.py")
_GO = _read("graph_orchestrator.py")
_DG = _read(os.path.join("prompts", "day_generator.py"))
_NL = _read(os.path.join("tests", "test_p1_next_level_batch.py"))


# ═══════════════ P3-REFINE-WEIGHTS-KNOBS ═══════════════

def test_refine_weights_are_knobs_with_identical_defaults():
    import portion_solver as ps
    assert ps._REFINE_WEIGHTS == {"kcal": 1.0, "protein": 1.5, "carbs": 1.0, "fats": 1.2}, \
        "defaults IDÉNTICOS a los literales previos ⇒ cero cambio de comportamiento en el merge"
    for kn in ("MEALFIT_REFINE_W_KCAL", "MEALFIT_REFINE_W_PROTEIN",
               "MEALFIT_REFINE_W_CARBS", "MEALFIT_REFINE_W_FATS"):
        assert kn in _PS
    from knobs import get_knobs_registry_snapshot
    snap = get_knobs_registry_snapshot()
    assert "MEALFIT_REFINE_W_FATS" in snap, "auto-registrados → visibles en /health/version"


# ═══════════════ P3-REFINE-EXEMPT-BOUNDARY ═══════════════

@pytest.mark.parametrize("line,exempt", [
    ("150 g de salmon a la plancha", False),   # "sal" ⊂ salmon → NO debe congelar la proteína
    ("200 g de ensalada verde", False),        # "sal" ⊂ ensalada
    ("80 g de salchicha de pavo", False),      # "sal" ⊂ salchicha
    ("5 g de sal", True),                      # sal de verdad SÍ
    ("10 g de aceite de oliva", True),
    ("10 g de linaza", True),                  # portador del micro-closer: protegido
    ("30 g de mani", True),
    ("50 g de zanahoria rallada", True),
])
def test_exempt_matcher_word_boundary(line, exempt):
    import portion_solver as ps
    fn = ps._exempt_matcher(("sal", "aceite", "azucar"))
    assert fn(line) is exempt, f"{line!r}"


def test_exempt_matcher_is_cached():
    """Recompilar por línea sería O(líneas × tokens) en el hot path del greedy."""
    import portion_solver as ps
    toks = ("sal", "aceite")
    assert ps._exempt_matcher(toks) is ps._exempt_matcher(toks)


def test_micro_carrier_tokens_cover_the_seed_catalog():
    """Paridad con lo que el closer siembra/escala: si se añade una semilla nueva al catálogo y no
    entra aquí, el refinador podrá recortarla a la mitad y deshacer el cierre."""
    import portion_solver as ps
    for t in ("linaza", "girasol", "mani", "zanahoria", "auyama", "espinaca"):
        assert t in ps._MICRO_CARRIER_TOKENS, t


# ═══════════════ P3-REFINE-OBSERVABILITY ═══════════════

def test_refiner_logs_when_it_exhausts_iterations():
    assert "P3-REFINE-OBSERVABILITY" in _PS
    assert "_exhausted = True" in _PS
    assert "_exhausted = False" in _PS, "sale por convergencia real ⇒ no es agotamiento"
    assert "AGOTÓ" in _PS


# ═══════════════ P3-SOLVER-CONVERGED-BAND / P3-SOLVER-FEASIBILITY ═══════════════

def test_converged_report_defaults_to_previous_semantics():
    """El knob nace OFF: la serie de no-convergencia ya tiene línea base medida (57.2%) y cambiarle
    la semántica sin avisar rompería la comparabilidad histórica."""
    import portion_solver as ps
    assert ps.SOLVER_CONVERGED_USES_BAND is False
    ok, per = ps._converged_report({"protein": 45, "carbs": 70, "fats": 25, "kcal": 700},
                                   {"protein": 45, "carbs": 70, "fats": 25, "kcal": 700}, 0.10)
    assert ok is True and set(per) == {"protein", "carbs", "fats"}, "kcal fuera con el knob OFF"


def test_converged_report_exposes_which_macro_failed():
    """El `break` viejo perdía CUÁL macro falló — el dato que vuelve accionable la métrica."""
    import portion_solver as ps
    ok, per = ps._converged_report({"protein": 45, "carbs": 70, "fats": 40},
                                   {"protein": 45, "carbs": 70, "fats": 25}, 0.10)
    assert ok is False
    assert per["fats"] is False and per["protein"] is True


def test_converged_uses_band_when_enabled(monkeypatch):
    import portion_solver as ps
    monkeypatch.setattr(ps, "SOLVER_CONVERGED_USES_BAND", True)
    # 1.11× de proteína: FUERA de ±10% simétrico pero DENTRO de la banda [0.90, 1.12]
    ok, per = ps._converged_report({"protein": 50.0, "carbs": 70, "fats": 25, "kcal": 700},
                                   {"protein": 45.0, "carbs": 70, "fats": 25, "kcal": 700}, 0.10)
    assert per["protein"] is True, "la banda real la acepta; el criterio simétrico la rechazaba"
    assert "kcal" in per, "y kcal entra al criterio con su banda estrecha"
    assert ok is True


def test_feasibility_detects_missing_carrier():
    """Caso vivo: merienda sin portador de grasa — target 8.5 g y máximo alcanzable 6.0 g."""
    import portion_solver as ps
    ents = [{"macros": {"kcal": 88, "protein": 15, "carbs": 5, "fats": 0.6}, "group": "protein"},
            {"macros": {"kcal": 89, "protein": 1.1, "carbs": 22, "fats": 0.3}, "group": "carbs"}]
    infe = ps._feasibility_report(ents, {"kcal": 300, "protein": 24, "carbs": 30, "fats": 8.5},
                                  0.3, [5.0, 3.5])
    assert infe == {"fats": "high"}, f"debía marcar la grasa inalcanzable: {infe}"


def test_feasibility_silent_when_reachable():
    import portion_solver as ps
    ents = [{"macros": {"kcal": 165, "protein": 31, "carbs": 0, "fats": 3.6}, "group": "protein"}]
    assert ps._feasibility_report(ents, {"protein": 40.0}, 0.3, [5.0]) == {}


def test_solver_returns_the_new_signals():
    # [P3-AUDIT-V5-TESTDEBT · 2026-07-30] Era `count(key) == 2` sobre el archivo entero, con dos
    # modos de fallo: (a) COMPENSACIÓN — un docstring futuro que cite la clave entre comillas
    # mantiene el 2 aunque un `return` la pierda, o sea verde sobre una pérdida real de señal;
    # (b) FRAGILIDAD — un tercer return legítimo lo rompe por un cambio que no es el suyo.
    # Se afirma POR SITIO: cada return público debe traer las tres claves. Inmune a menciones
    # en prosa y robusto ante returns nuevos.
    for fn, fin in (("def solve_portion_macros(", "\ndef solve_meal_macros("),
                    ("def solve_meal_macros(", "\n# =")):
        i = _PS.index(fn)
        cuerpo = _PS[i:_PS.index(fin, i)]
        ret = cuerpo[cuerpo.rindex("return {"):]
        for key in ('"converged_per_macro"', '"infeasible"', '"residuals"'):
            assert key in ret, f"{key} falta en el return de {fn.strip()}"


# ═══════════════ P3-SOLVER-WIRING-METRICS ═══════════════

def test_abstention_leaves_a_trace_in_plan_data():
    seg = _GO[_GO.index("def _apply_macro_solver_to_meal"):]
    seg = seg[: seg.index("\ndef ", 1)]
    assert '_solver_abstained_coverage' in seg, \
        "una decisión que cambia el plan debe ser consultable por SQL, no arqueología de logs"
    assert "_solver_infeasible" in seg and "_solver_failed_macros" in seg


def test_solver_convergence_has_its_own_series():
    assert '"node": "solver_convergence"' in _GO
    for k in ('"abstained_coverage"', '"infeasible_meals"', '"not_converged"'):
        assert k in _GO, k
    assert 'SOLVER_WIRING_METRICS = _env_bool("MEALFIT_SOLVER_WIRING_METRICS", True)' in _GO


# ═══════════════ P3-CLOSER-LINE-CLAMP-RAW-FALLBACK ═══════════════

def test_line_clamp_falls_back_to_raw_then_to_ratio():
    seg = _GO[_GO.index("def _close_micro_gaps_for_plan"):]
    seg = seg[: seg.index("\ndef ", 1)]
    assert "P3-CLOSER-LINE-CLAMP-RAW-FALLBACK" in seg
    assert "_clamp_basis" in seg
    assert "_closer_unmeasured_growth" in seg, "techo VITALICIO por ratio cuando no hay gramos"


def test_lifetime_ratio_cap_never_below_per_pass_max():
    """Un cap vitalicio por debajo del máximo por pasada recortaría el escalado SANO de la primera
    pasada — la regresión que un test cazó al ponerlo en 1.5 con el per-pass en 1.6."""
    import graph_orchestrator as go
    assert go.MICRO_CLOSER_UNMEASURED_LIFETIME_CAP >= go.MICRONUTRIENT_CLOSER_MAX_SCALE
    assert "cap vitalicio" in _GO, "el guard de coherencia sigue vivo"


# ═══════════════ P3-SEED-REINFORCE-FOOD-MATCH ═══════════════

def test_reinforce_matches_by_food_and_reports_when_it_cannot():
    seg = _GO[_GO.index("def _close_micro_gaps_for_plan"):]
    seg = seg[: seg.index("\ndef ", 1)]
    assert "MICRO_SEED_REINFORCE_FOOD_MATCH" in seg
    assert "_reinforced_any" in seg, "la rama 'no encontré la semilla' ya no sale MUDA"
    assert "grams_from_ingredient_string(_rline)" in seg, \
        "lead casero ('⅔ zanahoria (≈50 g)') → gramos por el resolvedor"


# ═══════════════ P3-MICRO-SEED-BUDGET-EXACT / CEILING-GUARD ═══════════════

def test_seed_budget_gate_uses_the_real_price():
    seg = _GO[_GO.index("def _close_micro_gaps_for_plan"):]
    seg = seg[: seg.index("\ndef ", 1)]
    assert "MICRO_SEED_BUDGET_EXACT" in seg
    assert "_kc_seed > _budget_here" in seg, "gate por COSTO real, no por piso fijo"


def test_seed_candidate_passes_the_clinical_guard():
    seg = _GO[_GO.index("def _close_micro_gaps_for_plan"):]
    seg = seg[: seg.index("\ndef ", 1)]
    assert "MICRO_SEED_CEILING_GUARD and _ceiling_risky_contributor(k, _cand_low)" in seg, \
        "el candidato de SIEMBRA reusa el guard que ya protege el escalado (7 ramas gratis)"
    assert "_ul_seed" in seg, "y el techo UL que el seed no chequeaba"


# ═══════════════ P3-CLOSER-PCOS-GLYCEMIC-GUARD ═══════════════

def test_pcos_inherits_the_high_gi_guard():
    seg = _GO[_GO.index("def _close_micro_gaps_for_plan"):]
    seg = seg[: seg.index("\ndef ", 1)]
    assert "PCOS_CONDITION_TERMS" in seg
    assert 'CLOSER_PCOS_GLYCEMIC_GUARD = _env_bool("MEALFIT_CLOSER_PCOS_GLYCEMIC_GUARD", True)' in _GO
    from constants import PCOS_CONDITION_TERMS
    assert "sop" in PCOS_CONDITION_TERMS


def test_pcos_terms_go_through_condition_strings_not_raw_substring():
    """'sop' son 3 caracteres: por `_condition_strings` (accent-strip, mismo criterio que los otros
    detectores), NUNCA por substring crudo sobre texto libre."""
    seg = _GO[_GO.index("def _close_micro_gaps_for_plan"):]
    seg = seg[: seg.index("\ndef ", 1)]
    i = seg.index("PCOS_CONDITION_TERMS")
    assert "_conditions" in seg[i - 400:i + 400]


# ═══════════════ P3-DAYGEN-SLOT-TARGETS ═══════════════

def test_slot_targets_block_off_by_default_and_prompt_identical():
    from prompts.day_generator import build_day_assignment_context as _b
    sk = {"brief_concept": "x", "protein_pool": ["Pollo"], "carb_pool": ["Arroz"],
          "fruit_pool": ["Lechosa"], "meal_types": ["Desayuno", "Almuerzo", "Merienda", "Cena"]}
    out = _b(sk, 1, daily_targets={"kcal": 2050, "protein": 150, "carbs": 200, "fats": 68})
    assert "CUOTA POR COMIDA" not in out, "OFF ⇒ prompt byte-idéntico (nodo más caro del pipeline)"
    assert 'MEALFIT_DAYGEN_SLOT_TARGETS_IN_PROMPT' in _DG


def test_slot_targets_block_demands_a_fat_carrier():
    """13 de las 16 infactibilidades medidas eran de GRASA: la regla del portador ataca la causa."""
    from prompts.day_generator import build_slot_targets_block as _s
    blk = _s({"kcal": 2050, "protein": 150, "carbs": 200, "fats": 68},
             ["Desayuno", "Almuerzo", "Merienda", "Cena"])
    assert "CUOTA POR COMIDA" in blk
    assert "PORTADOR DE GRASA OBLIGATORIO" in blk
    assert "Merienda" in blk


def test_slot_targets_block_fails_open():
    from prompts.day_generator import build_slot_targets_block as _s
    assert _s({}, []) == ""
    assert _s(None, ["Desayuno"]) == ""


# ═══════════════ P3-REFINER-LOCKSTEP-FIXTURES / ASSEMBLE-ANCHOR ═══════════════

def test_lockstep_test_now_covers_the_misaligned_states():
    assert "_ref_meals_reordered" in _NL and "_ref_meals_raw_longer" in _NL, \
        "el fixture alineado certificaba el ÚNICO caso en que el bug no ocurre"
    assert "test_refiner_lockstep_never_scales_the_wrong_food" in _NL


def test_assemble_callsite_anchor_is_unique():
    assert _GO.count("tooltip-anchor: P1-NEXT-LEVEL-SOLVER-ASSEMBLE-CALLSITE") == 1
    assert "P1-NEXT-LEVEL-SOLVER-ASSEMBLE-CALLSITE" in _NL
