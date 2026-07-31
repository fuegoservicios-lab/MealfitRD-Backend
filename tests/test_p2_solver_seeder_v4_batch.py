"""[P2-SOLVER-SEEDER-V4 · 2026-07-29] Los P2 del audit solver+seeder v4.

Alineación raw↔display:
- P2-RAW-PAIR-BY-FOOD: "mismo largo" nunca fue evidencia de "mismo orden" (93.5% de las comidas
  tienen largo igual y solo el 48.1% de ESAS son paralelas). Solver + refinador + closer exigen
  paralelismo VERIFICADO por alimento antes de fiarse del índice.
- P2-MICRO-CLOSER-RAW-BY-FOOD: las 3 escrituras a raw del closer pasan por un helper SSOT; el
  refuerzo dejó de ASIGNAR el string de display a raw.
- P2-RECONCILE-AFTER-BAND-CLOSER: el shield pre-INSERT reconcilia DESPUÉS del band-closer.

Solver / refinador:
- P2-REFINE-COVERAGE-GATE: abstención con masa no-resuelta (el refinador inflaba las líneas
  resolubles de las OTRAS comidas para cubrir macros que ya estaban en el plato).
- P2-SOLVER-CONVERGENCE-METRIC: la señal de mayor volumen del solver (57.2%) ya tiene consumidor,
  y el swap deja de descartarla.
- P2-SWAP-NUM-MEALS: el slot-target del swap deja de asumir 4 comidas.

Closer / clínico:
- P2-MICRO-CLOSER-MEASURE-RAW-UNION: medir sobre la MISMA base que el panel del usuario.
- P2-MICRO-CLOSER-MACRO-DELTA: la masa añadida llega a los macros aunque el truth-up se abstenga.
- P2-CLOSER-MED-CONTRIBUTOR-GUARD: anticoagulante y K-elevador pasan de filtro-de-micro a
  filtro-de-CONTRIBUYENTE.
- P2-MICRO-SEED-CLINICAL-CTX: sin contexto clínico no se siembra (los filtros se gateaban por
  presencia de key ⇒ "campo ausente" se leía como "sin alergias").

Seeder de variedad:
- P2-FREQ-TRACKING-CHUNKED / P2-FREQ-LOOKUP-CANONICAL: el motor de frecuencias recibe datos y los
  lee por la misma clave con la que se escriben.
- P2-PROTEIN-ALIASES-SSOT: bacalao/arenque/mejillones dejan de ser invisibles al gate same-day.
- P2-PANTRY-ROTATION-FLOOR: la nevera es un piso, no una camisa de fuerza.
- P2-LIGHT-PROTEIN-SEED: capacidad implementada, lever OFF (prompt byte-idéntico).
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
_AI = _read("ai_helpers.py")
_AG = _read("agent.py")
_SV = _read("services.py")
_DBP = _read("db_plans.py")
_PLANS = _read(os.path.join("routers", "plans.py"))
_TOOLS = _read("tools.py")


# ═══════════════ P2-RAW-PAIR-BY-FOOD ═══════════════

def test_parallel_checker_rejects_permuted_equal_length():
    """EL bug: raw ROTADO con el MISMO largo. Caso vivo (plan fbe53a5b): display[0]='filete de
    pescado' / raw[0]='batata mediana'. El guard por largo lo daba por paralelo."""
    import graph_orchestrator as go
    disp = ["150 g de pollo", "200 g de arroz blanco", "15 g de aceite de oliva"]
    assert go._raw_display_parallel_by_food(disp, list(disp)) is True
    permuted = ["200 g de arroz blanco", "15 g de aceite de oliva", "150 g de pollo"]
    assert go._raw_display_parallel_by_food(disp, permuted) is False
    assert go._raw_display_parallel_by_food(disp, disp[:2]) is False, "largos distintos → no paralelo"


def test_solver_requires_verified_parallelism():
    # [P3 · 2026-07-29] fin de función, no ventana fija (la de 9000 caducó el mismo día).
    _i = _GO.index("def _apply_macro_solver_to_meal")
    body = _GO[_i:_i + _GO[_i:].index("\ndef ", 1)]
    assert "_raw_display_parallel_by_food(_ing_strs, raw)" in body
    assert "RAW_PAIR_BY_FOOD" in body
    assert 'RAW_PAIR_BY_FOOD = _env_bool("MEALFIT_RAW_PAIR_BY_FOOD", True)' in _GO


def test_refiner_requires_verified_parallelism():
    assert "_raw_display_parallel_by_food as _par_ok" in _PS
    assert "_parallel = len(raw) == len(_disp_orig)" in _PS


# ═══════════════ P2-MICRO-CLOSER-RAW-BY-FOOD ═══════════════

def test_closer_raw_writes_go_through_the_ssot_helper():
    """Las TRES escrituras a raw del closer (escalado, refuerzo, fatswap)."""
    seg = _GO[_GO.index("def _close_micro_gaps_for_plan"):]
    seg = seg[: seg.index("\ndef ", 1)]
    assert seg.count("_sync_one_raw_line(") == 3, "las 3 escrituras a raw pasan por el helper"
    # el refuerzo ya NO asigna el string de DISPLAY a raw (era erróneo incluso con listas paralelas)
    assert "_r_raw[_ri] = _new_line" not in seg


def test_sync_one_raw_line_scales_the_right_food_when_misaligned():
    import graph_orchestrator as go
    meal = {"name": "Pollo con arroz",
            "ingredients": ["150 g de pollo", "200 g de arroz blanco"],
            "ingredients_raw": ["30 g de cebolla roja", "150 g de pollo", "200 g de arroz blanco"]}
    ok = go._sync_one_raw_line(meal, 0, "150 g de pollo", 2.0)
    raw = meal["ingredients_raw"]
    assert ok is True
    assert "cebolla" in raw[0].lower() and raw[0].startswith("30"), \
        f"la cebolla NO debe moverse: {raw}"
    assert "pollo" in raw[1].lower() and not raw[1].startswith("150"), \
        f"el pollo del raw debía escalar: {raw}"


def test_sync_one_raw_line_is_conservative_when_ambiguous():
    """Alimento en varias líneas de raw → no adivinamos: raw intacto."""
    import graph_orchestrator as go
    meal = {"name": "x", "ingredients": ["100 g de pollo", "50 g de arroz blanco"],
            "ingredients_raw": ["60 g de pollo", "40 g de pollo", "50 g de arroz blanco"]}
    before = list(meal["ingredients_raw"])
    assert go._sync_one_raw_line(meal, 0, "100 g de pollo", 2.0) is False
    assert meal["ingredients_raw"] == before


def test_sync_one_raw_line_never_swallows_errors_silently():
    """El primer intento de este helper usaba `rescale_ingredient_string` como global (no existe a
    nivel de módulo) → NameError tragado por el except → raw NUNCA se escribía, en silencio.
    El except ahora LOGUEA; este test ancla el import local que lo cierra de raíz."""
    seg = _GO[_GO.index("def _sync_one_raw_line"):]
    seg = seg[: seg.index("\ndef ", 1)]
    assert "from nutrition_db import rescale_ingredient_string" in seg
    assert "logger.warning" in seg, "un fallo aquí desincroniza la lista de compras: nunca mudo"


# ═══════════════ P2-RECONCILE-AFTER-BAND-CLOSER ═══════════════

def test_reconcile_runs_after_the_band_closer_pre_insert():
    i_ramb = _DBP.index("reconcile_all_macros_band_post_finalize as _ramb")
    i_rec = _DBP.index("P2-RECONCILE-AFTER-BAND-CLOSER")
    i_rdp = _DBP.index("refire_display_polish_post_finalize as _rdp")
    assert i_ramb < i_rec < i_rdp, "reconciliar tras el band-closer y antes del re-pulido"
    assert 'RECONCILE_AFTER_BAND_CLOSER = _env_bool("MEALFIT_RECONCILE_AFTER_BAND_CLOSER", True)' in _GO


# ═══════════════ P2-REFINE-COVERAGE-GATE ═══════════════

class _PartialDB:
    """Resuelve SOLO las líneas de pollo/arroz — el resto es 'masa invisible' (plato criollo)."""

    def macros_from_ingredient_string(self, s):
        low = str(s).lower()
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", low)
        if not m or not any(t in low for t in ("pollo", "arroz")):
            return None
        g = float(m.group(1).replace(",", "."))
        return {"kcal": 1.6 * g, "protein": 0.2 * g, "carbs": 0.1 * g, "fats": 0.03 * g}

    def grams_from_ingredient_string(self, s):
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", str(s).lower())
        return float(m.group(1).replace(",", ".")) if m else 0.0


def _low_cov_day():
    return [{"name": "Sancocho", "ingredients": ["150 g de pollo", "200 g de sancocho criollo",
                                                 "180 g de vianda mixta", "120 g de auyama del país"]}]


def test_refiner_abstains_under_low_coverage():
    from portion_solver import refine_day_portions_integer
    meals = _low_cov_day()
    before = [list(m["ingredients"]) for m in meals]
    moves = refine_day_portions_integer(meals, {"kcal": 2000, "protein": 150, "carbs": 200, "fats": 67},
                                        _PartialDB(), floor_g=20.0, cap_g=400.0)
    assert moves == 0, "con 1/4 de cobertura el refinador debe abstenerse"
    assert [m["ingredients"] for m in meals] == before, "día INTACTO"


def test_refine_coverage_knob_rollback(monkeypatch):
    import portion_solver as ps
    from portion_solver import refine_day_portions_integer
    meals = _low_cov_day()
    monkeypatch.setattr(ps, "REFINE_MIN_COVERAGE", 0.0)
    assert refine_day_portions_integer(meals, {"kcal": 2000, "protein": 150, "carbs": 200, "fats": 67},
                                       _PartialDB(), floor_g=20.0, cap_g=400.0) > 0, \
        "con el gate en 0.0 vuelve a mover (rollback sin redeploy)"


def test_refine_coverage_benign_lines_excluded():
    """Agua y hierbas NO son masa oculta: no deben deprimir la cobertura (misma forma
    word-boundary que P1-SOLVER-COVERAGE-BENIGN — 'agua' ⊄ 'aguacate')."""
    assert "_BENIGN_RE" in _PS
    assert r"\b(?:agua|hielo|perejil|cilantro" in _PS
    assert 'REFINE_MIN_COVERAGE = _envf("MEALFIT_REFINE_MIN_COVERAGE", 0.6' in _PS


# ═══════════════ P2-SOLVER-CONVERGENCE-METRIC ═══════════════

def test_convergence_keys_added_to_existing_metric():
    i = _GO.index("[P2-SOLVER-CLAMP-ACTION · 2026-07-05]")
    blk = _GO[i:_GO.index("[P2-LOW-COVERAGE-MEALS", i)]
    for k in ('"not_converged_meals"', '"greedy_fallback_meals"', '"not_converged_ratio"'):
        assert k in blk, f"falta {k} en la métrica per-run"
    assert '"total_meals"' in blk, "el denominador de la tasa de flota sigue ahí"
    assert 'SOLVER_CONVERGENCE_METRIC = _env_bool("MEALFIT_SOLVER_CONVERGENCE_METRIC", True)' in _GO


def test_swap_no_longer_drops_solver_telemetry():
    """Sin esto la serie de no-convergencia arranca sesgada: un agujero justo en las comidas
    swapeadas (el solver corre sobre un `model_dump()` que se descartaba)."""
    for k in ("_solver_not_converged", "_solver_greedy_fallback", "_solver_clamp_saturated"):
        assert k in _AG, f"{k} debe volver al meal tras el swap"


# ═══════════════ P2-SWAP-NUM-MEALS ═══════════════

def test_swap_num_meals_derived_from_profile():
    assert 'SWAP_NUM_MEALS_FROM_PLAN = _env_bool("MEALFIT_SWAP_NUM_MEALS_FROM_PLAN", True)' in _AG
    assert "decide_meals_per_day as _dmpd8" in _AG
    assert "_num8 = _num8 or 4" in _AG, "el literal 4 sigue siendo el ÚLTIMO fallback"


# ═══════════════ P2-MICRO-CLOSER-MEASURE-RAW-UNION / MACRO-DELTA ═══════════════

def test_closer_measures_on_raw_union():
    seg = _GO[_GO.index("def _close_micro_gaps_for_plan"):]
    seg = seg[: seg.index("\ndef ", 1)]
    assert "MICRO_CLOSER_MEASURE_RAW_UNION" in seg
    assert "_measured_on_raw" in seg
    assert "day_total += float(_rc)" in seg
    assert 'MICRO_CLOSER_MEASURE_RAW_UNION = _env_bool("MEALFIT_MICRO_CLOSER_MEASURE_RAW_UNION", True)' in _GO


def test_closer_macro_delta_fallback_wired_at_all_four_sites():
    seg = _GO[_GO.index("def _close_micro_gaps_for_plan"):]
    seg = seg[: seg.index("\ndef ", 1)]
    # [P3-AUDIT-V5-TESTDEBT · 2026-07-30] `== 4` → `>= 4` por la misma razón que el hermano de
    # arriba: el conteo exacto castiga precisamente el uso correcto del SSOT.
    assert seg.count("_closer_sync_meal_macros(") >= 4, \
        "seed + refuerzo + escalado + fatswap sincronizan macros por el helper"
    assert "_truth_up_meal_macros_from_strings(" not in seg, \
        "ninguno debe llamar al truth-up pelado (se abstiene y deja los macros stale)"


class _AbstainDB:
    """Fuerza la abstención del truth-up: una línea resuelve por nombre y no por cantidad."""

    def macros_from_ingredient_string(self, s):
        low = str(s).lower()
        if "lata" in low:
            return None
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", low)
        if not m:
            return None
        g = float(m.group(1).replace(",", "."))
        return {"kcal": 5.3 * g, "protein": 0.18 * g, "carbs": 0.29 * g, "fats": 0.42 * g}


def test_macro_delta_applies_when_truthup_abstains(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda *a, **kw: False)
    meal = {"name": "Merienda", "cals": 300, "protein": 20, "carbs": 30, "fats": 8,
            "ingredients": ["1 lata de atún al agua", "10 g de semillas de linaza"]}
    assert go._closer_sync_meal_macros(meal, _AbstainDB(), new_line="10 g de semillas de linaza") is True
    assert meal["cals"] > 300, "las kcal de la semilla deben contarse"
    assert meal["fats"] > 8, "y su grasa también (el closer siembra material graso-denso)"


def test_macro_delta_not_applied_when_truthup_writes(monkeypatch):
    """Idempotencia: si el truth-up SÍ escribió, el delta no se suma encima."""
    import graph_orchestrator as go
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda *a, **kw: True)
    meal = {"name": "x", "cals": 300, "protein": 20, "carbs": 30, "fats": 8, "ingredients": []}
    assert go._closer_sync_meal_macros(meal, _AbstainDB(), new_line="10 g de semillas de linaza") is True
    assert meal["cals"] == 300


# ═══════════════ P2-CLOSER-MED-CONTRIBUTOR-GUARD ═══════════════

def test_med_guards_moved_to_contributor_level():
    seg = _GO[_GO.index("def _close_micro_gaps_for_plan"):]
    seg = seg[: seg.index("\ndef ", 1)]
    assert "_vitk_contrib_re.search(_ing_low)" in seg, "vit K por CONTRIBUYENTE, no solo por micro"
    assert "_highk_contrib_re.search(_ing_low)" in seg
    assert "from medication_rules import _HIGH_VIT_K_TERMS" in seg, \
        "SSOT importado, NO una lista re-escrita (esa fue la 12ª mordida de subcadena)"
    for kn in ("MEALFIT_CLOSER_ANTICOAG_CONTRIBUTOR_GUARD", "MEALFIT_CLOSER_KMED_CONTRIBUTOR_GUARD"):
        assert kn in _GO


def test_vitk_regex_matches_greens_and_not_unrelated_foods():
    """El guard es inútil si el regex no compila o si muerde de más."""
    import re as _re
    from constants import strip_accents as _sa
    from medication_rules import _HIGH_VIT_K_TERMS as _H
    rx = _re.compile(r"\b(?:" + "|".join(_re.escape(_sa(t)) for t in _H) + r")s?\b")
    assert rx.search(_sa("80 g de espinacas"))
    assert rx.search(_sa("100 g de brocoli"))
    assert not rx.search(_sa("150 g de pechuga de pollo"))


# ═══════════════ P2-MICRO-SEED-CLINICAL-CTX ═══════════════

def test_seed_requires_clinical_context():
    assert 'MICRO_SEED_REQUIRE_CLINICAL_CTX = _env_bool("MEALFIT_MICRO_SEED_REQUIRE_CLINICAL_CTX", True)' in _GO
    seg = _GO[_GO.index("def _close_micro_gaps_for_plan"):]
    seg = seg[: seg.index("\ndef ", 1)]
    assert '"allergies" not in _fd' in seg, \
        "presencia de KEY, no truthiness: `if _fd.get('allergies')` trata ausente como sin alergias"


def test_update_surfaces_carry_the_clinical_keys():
    """La otra mitad: los dos dicts hechos a mano omitían justo las 3 claves que el seed consulta."""
    for src, name in ((_PLANS, "routers/plans.py"), (_TOOLS, "tools.py")):
        assert '"allergies":' in src, name
        assert '"dietType":' in src, name
        assert '"dislikes":' in src, name


# ═══════════════ P2-FREQ-TRACKING-CHUNKED / P2-FREQ-LOOKUP-CANONICAL ═══════════════

def test_freq_tracking_helper_is_ssot_and_chunked_calls_it():
    assert "def _track_ingredient_frequencies(" in _SV
    assert "def extract_raw_ingredients_from_plan(" in _SV
    assert "_track_ingredient_frequencies(user_id, raw_ingredients)" in _SV, "el path no-chunked lo usa"
    assert "_track_ingredient_frequencies as _tif" in _PLANS, "y la rama chunked también"
    assert "P2-FREQ-TRACKING-CHUNKED" in _PLANS


def test_freq_lookup_uses_the_same_key_the_table_is_written_with():
    from constants import normalize_ingredient_for_tracking as _norm
    # los casos medidos: la clave canónica es OTRO ítem del pool → freq=0 permanente
    assert _norm("Maní") == "nueces/almendras"
    assert _norm("Salmón") == "pescado"
    assert "_freq_for_pool_item" in _AI
    assert "normalize_ingredient_for_tracking as _norm_freq" in _AI
    assert 'FREQ_LOOKUP_CANONICAL = _env_bool("MEALFIT_FREQ_LOOKUP_CANONICAL", True)' in _AI


# ═══════════════ P2-PROTEIN-ALIASES-SSOT ═══════════════

def test_fish_species_visible_to_the_same_day_gate():
    import graph_orchestrator as go
    for token in ("bacalao", "arenque"):
        assert token in go._MAIN_PROTEIN_ALIASES["pescado"], f"{token} seguía invisible al gate"


def test_ssot_union_does_not_reintroduce_the_pechuga_collision():
    """P6-SLOT-CROSS-PROTEIN quitó 'pechuga' a propósito: matchea dentro de 'pechuga de pavo' y
    etiquetaba pavo como pollo. La unión automática la reintrodujo y la regresión lo cazó."""
    import graph_orchestrator as go
    assert "pechuga" not in go._MAIN_PROTEIN_ALIASES["pollo"]
    assert "_ambiguous" in _GO, "el guard de ambigüedad sigue vivo"


def test_mejillones_labeled_and_has_a_swap_ladder():
    import graph_orchestrator as go
    from constants import strip_accents as sa
    assert "mejillones" in go._HEAVY_PROTEIN_LABELS
    assert go._dm_species_of(sa("Mejillones al Vapor".lower())) == {"mejillon"}
    assert "mejillones" in go._PROTEIN_REPEAT_SWAP_LADDER, \
        "detectar sin poder reparar = no_ladder_for_label = rechazo del plan completo"


def test_every_gate_label_now_has_a_ladder():
    """Rojo PRE-EXISTENTE que este batch cierra: P1-VARIETY-GATE-GAPS añadió 5 especies al gate sin
    escalera, dejando el autofix impotente."""
    from graph_orchestrator import _PROTEIN_REPEAT_SWAP_LADDER, _SAME_DAY_PROTEIN_GATE_LABELS
    missing = [l for l in _SAME_DAY_PROTEIN_GATE_LABELS
               if l != "huevo" and l not in _PROTEIN_REPEAT_SWAP_LADDER]
    assert not missing, f"labels del gate sin escalera: {missing}"


def test_res_keeps_its_word_boundary():
    """No re-abrir 'res' ⊂ 'fResco' al ampliar alias (P1-DM-SPECIES-BOUNDARY)."""
    import graph_orchestrator as go
    from constants import strip_accents as sa
    assert go._dm_species_of(sa("cilantro fresco para decorar")) == set()


# ═══════════════ P2-PANTRY-ROTATION-FLOOR ═══════════════

def test_pantry_floor_knob_and_conditional_lock():
    assert "PANTRY_ROTATION_MIN_PROTEINS" in _AI
    assert "MEALFIT_PANTRY_ROTATION_MIN_PROTEINS" in _AI
    assert "_pantry_sustains_rotation = bool(extracted_p) and len(extracted_p) >= _min_p" in _AI
    # [P3-CYCLE-LOCK-ADDITIVE · 2026-07-31] el veredicto de la nevera se ACUMULA sobre el lock
    # del grocery-cycle en vez de pisarlo (antes esta línea era `cycle_locked = ...` directo).
    assert "cycle_locked = cycle_locked or _pantry_sustains_rotation" in _AI, \
        "el lock deja de ser incondicional (una nevera con 1 proteína daba 3 días iguales)"
    import ai_helpers as ai
    assert ai.PANTRY_ROTATION_MIN_PROTEINS == 2


# ═══════════════ P2-SLOT-DRIFT-TELEMETRY ═══════════════

def test_slot_drift_is_observable():
    import graph_orchestrator as go
    plan = {"days": [{"meals": [
        {"meal_type": "Desayuno", "protein": 30, "carbs": 60, "fats": 15},
        {"meal_type": "Almuerzo", "protein": 50, "carbs": 80, "fats": 25},
        {"meal_type": "Merienda", "protein": 20, "carbs": 30, "fats": 10},
        {"meal_type": "Cena", "protein": 20, "carbs": 30, "fats": 10}]}]}
    drift = go._compute_slot_drift(plan)
    assert drift and "cena" in drift
    assert drift["cena"]["protein"] < 1.0, "la cena bajo su cuota debe ser VISIBLE"
    assert '"slot_drift"' in _GO, "y viajar en el payload de la banda"
    assert 'SLOT_AWARE_DAY_REPAIR = _env_bool("MEALFIT_SLOT_AWARE_DAY_REPAIR", False)' in _GO, \
        "el cambio de reparto FÍSICO nace OFF (exige A/B contra all4_ratio)"


def test_slot_drift_uses_the_canonical_meal_key_not_meal_type():
    """[P3-SLOT-DRIFT-KEY · 2026-07-29] En los planes vivos `meal_type` vale 'flexible' para TODAS
    las comidas: la v1 de este helper agrupaba el día entero en UN bucket y la telemetría —que existe
    para hacer visible el sesgo POR SLOT— no podía mostrarlo (verificado en el plan vivo 4218191f:
    `slot_drift = {'flexible': {...}}`, una sola fila). Misma clase que P1-SLOT-FRACTIONS-KEY, donde
    leer la clave equivocada dejó P3-SLOT-DISTRIBUTION inerte seis semanas."""
    import graph_orchestrator as go
    plan = {"days": [{"meals": [
        {"meal": "Desayuno", "meal_type": "flexible", "protein": 30, "carbs": 60, "fats": 15},
        {"meal": "Almuerzo", "meal_type": "flexible", "protein": 50, "carbs": 80, "fats": 25},
        {"meal": "Merienda", "meal_type": "flexible", "protein": 20, "carbs": 30, "fats": 10},
        {"meal": "Cena", "meal_type": "flexible", "protein": 20, "carbs": 30, "fats": 10}]}]}
    drift = go._compute_slot_drift(plan)
    assert set(drift) == {"desayuno", "almuerzo", "merienda", "cena"}, \
        f"debe distinguir los 4 slots, no colapsarlos en uno: {list(drift)}"
    assert drift["cena"]["protein"] < drift["desayuno"]["protein"], \
        "el sesgo contra la CENA debe quedar visible — es el motivo de esta telemetría"


def test_slot_drift_falls_back_to_canonical_names_without_meal_key():
    """Sin clave `meal` se etiqueta por ÍNDICE con el nombre canónico del split: nunca todo bajo la
    misma etiqueta, que es el modo de fallo que hace la métrica inútil."""
    import graph_orchestrator as go
    plan = {"days": [{"meals": [
        {"meal_type": "flexible", "protein": 30, "carbs": 60, "fats": 15},
        {"meal_type": "flexible", "protein": 50, "carbs": 80, "fats": 25},
        {"meal_type": "flexible", "protein": 20, "carbs": 30, "fats": 10},
        {"meal_type": "flexible", "protein": 20, "carbs": 30, "fats": 10}]}]}
    assert set(go._compute_slot_drift(plan)) == {"desayuno", "almuerzo", "merienda", "cena"}


def test_slot_drift_reuses_the_pipeline_ssot():
    """Nada de una 3ª lista paralela de fracciones: usa la misma función que reparte el día."""
    seg = _GO[_GO.index("def _compute_slot_drift"):]
    seg = seg[: seg.index("\ndef ", 1)]
    assert "_canonical_slot_fractions(_meals)" in seg


# ═══════════════ P2-LIGHT-PROTEIN-SEED ═══════════════

def test_light_protein_seed_born_off_and_prompt_byte_identical():
    import ai_helpers as ai
    assert ai.LIGHT_PROTEIN_SEED is False, "nace OFF esperando A/B"
    assert "{light_protein_block}" in _read(os.path.join("prompts", "preferences.py"))
    assert '_light_block = ""' in _AI, "con el knob OFF el bloque es vacío → prompt byte-idéntico"
