"""[P2-OBJECTIVE-BATCH-2 · 2026-07-01] Lote de ~18 P2 del audit objetivo 2026-07-01 (post P0+P1).

Cobertura por marker (cada uno con su propio tooltip-anchor en producción):
  Macros:  P2-TRUTHUP-KNOB-SPLIT · P2-REVIEW-PATCH-TRUTHUP · P2-MICROCLOSER-REQUANTIZE ·
           P2-FALLBACK-PHYSICAL-MACROS
  Micros:  P2-MICRO-CLOSER-KEYS-EXT (K/vitE/omega-3) · P2-MICRO-CLOSER-CEILINGS
  Slots:   P2-CHAT-EXPLICIT-SLOT-WISH · P2-UPDATE-SAMEDAY-VARIETY · P2-CROSSDAY-PREP-DIVERSITY ·
           P2-NAME-HONESTY-EXT · P2-SWAP-BAND-WARNING
  Recetas: P2-EXPAND-VEG-TRUTHUP · P2-RECIPE-STEP-CONTRACT-GATE · P2-STEP-INSERT-BEFORE-MONTAJE ·
           P2-PERSIST-BOUNDARY-COHERENCE · P2-FALLBACK-RECIPE-SLOT-TEMPLATE · P2-RECIPE-HOUSEHOLD-NOTE
"""
from __future__ import annotations

import inspect
from pathlib import Path

import graph_orchestrator as g

_BACKEND = Path(__file__).resolve().parent.parent
_GRAPH = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_TOOLS = (_BACKEND / "tools.py").read_text(encoding="utf-8")
_AGENT = (_BACKEND / "agent.py").read_text(encoding="utf-8")
_AIH = (_BACKEND / "ai_helpers.py").read_text(encoding="utf-8")
_FRONT = _BACKEND.parent / "frontend" / "src"


# ───────────────────────── markers presentes ─────────────────────────
def test_markers_present():
    for mk in ("P2-TRUTHUP-KNOB-SPLIT", "P2-REVIEW-PATCH-TRUTHUP", "P2-MICROCLOSER-REQUANTIZE",
               "P2-FALLBACK-PHYSICAL-MACROS", "P2-MICRO-CLOSER-KEYS-EXT", "P2-MICRO-CLOSER-CEILINGS",
               "P2-CROSSDAY-PREP-DIVERSITY", "P2-NAME-HONESTY-EXT", "P2-STEP-INSERT-BEFORE-MONTAJE",
               "P2-PERSIST-BOUNDARY-COHERENCE", "P2-FALLBACK-RECIPE-SLOT-TEMPLATE",
               "P2-RECIPE-STEP-CONTRACT-GATE", "P2-CHAT-EXPLICIT-SLOT-WISH"):
        assert mk in _GRAPH or mk in _TOOLS, f"marker {mk} ausente"
    assert "P2-SWAP-BAND-WARNING" in _PLANS and "P2-EXPAND-VEG-TRUTHUP" in _PLANS
    assert "P2-UPDATE-SAMEDAY-VARIETY" in _AGENT and "P2-UPDATE-SAMEDAY-VARIETY" in _TOOLS
    assert "P2-MICROCLOSER-REQUANTIZE" in _PLANS and "P2-MICROCLOSER-REQUANTIZE" in _TOOLS


# ───────────────────────── macros ─────────────────────────
def test_knob_defaults():
    assert g.PER_MEAL_MACRO_TRUTHUP_ENABLED is True
    assert g.MICROCLOSER_BAND_RECHECK_ENABLED is True
    assert g.FALLBACK_PHYSICAL_MACROS_ENABLED is True


def test_band_recheck_decoupled_from_carb_trim():
    """[P2-MICROCLOSER-REQUANTIZE] el re-check post-closer tiene knob PROPIO (antes gateado por
    CARB_TARGET_TRIM_ENABLED=False → muerto con defaults)."""
    i = _GRAPH.find("P2-MICROCLOSER-BAND-RECHECK")
    seg = _GRAPH[i:i + 1200]
    assert "if MICROCLOSER_BAND_RECHECK_ENABLED:" in seg, \
        "el re-check sigue acoplado a CARB_TARGET_TRIM_ENABLED"


def test_review_patch_truths_up_touched_meal():
    """[P2-REVIEW-PATCH-TRUTHUP] tras borrar ingredientes post-band-gate, truth-up del meal."""
    i = _GRAPH.find("def _auto_patch_ingredient_coherence")
    seg = _GRAPH[i:_GRAPH.find("\ndef ", i + 10)]
    assert "_truth_up_meal_macros_from_strings" in seg, \
        "el auto-patch de review borra ingredientes sin recomputar macros"


def test_fallback_rescales_and_truths_up():
    i = _GRAPH.find("def _build_fallback_day")
    seg = _GRAPH[i:_GRAPH.find("\ndef _get_extreme_fallback_plan", i)]
    assert "rescale_ingredient_string" in seg and "_truth_up_meal_macros_from_strings" in seg, \
        "el fallback sigue asertando macros sin física (P2-FALLBACK-PHYSICAL-MACROS)"
    assert "aproximado a" in _GRAPH, "el disclaimer del fallback debe ser honesto (aproximado, no 'SÍ cumple')"


def test_swap_and_chat_requantize_after_closer():
    i = _PLANS.find("P2-MICROCLOSER-REQUANTIZE")
    assert i != -1 and "_apply_portion_quantization" in _PLANS[i:i + 1500], \
        "swap-persist: closer sin re-quantize"
    assert _TOOLS.count("_apply_portion_quantization") >= 2, \
        "chat-modify: pre-listas y callback deben re-quantizar tras el closer"


# ───────────────────────── micros ─────────────────────────
def test_closer_keys_extended():
    for k in ("potassium_mg", "vit_e_mg", "omega3_g"):
        assert k in g._MICRO_CLOSER_KEYS, f"{k} debe ser cerrable (P2-MICRO-CLOSER-KEYS-EXT)"
        assert k in g._MICRO_CLOSER_INGREDIENT_KEY
    assert g._MICRO_CLOSER_UL.get("vit_e_mg") == 1000.0
    assert "potassium_mg" not in g._MICRO_CLOSER_UL, "K de comida no lleva UL (el guard es renal/K-med)"


def test_closer_potassium_skips_kelev_and_renal():
    i = _GRAPH.find('if k == "potassium_mg" and _k_elev:')
    assert i != -1, "falta el skip de potasio bajo medicamento K-elevador"
    assert '"potassium_mg", "vit_e_mg", "omega3_g"' in _GRAPH, \
        "las keys nuevas deben excluirse también en ERC"


def test_closer_ceiling_guard_present():
    assert "_ceiling_risky_contributor" in _GRAPH, \
        "falta el guard de techos (P2-MICRO-CLOSER-CEILINGS): dislipidemia/HTA no debe escalar queso/embutido"
    i = _GRAPH.find("def _ceiling_risky_contributor")
    seg = _GRAPH[i:i + 800]
    assert "_DAIRY_TOKENS" in seg or "calcium_mg" in seg, "falta la rama renal-calcio-lácteo"


# ───────────────────────── slots / apetecibilidad ─────────────────────────
def test_finalizer_skip_night_rice_param():
    sig = inspect.signature(g.finalize_single_meal_recipe_coherence)
    assert "skip_night_rice" in sig.parameters and sig.parameters["skip_night_rice"].default is False
    assert "skip_night_rice=_wish_slot" in _TOOLS, \
        "chat-modify no pasa la señal de deseo explícito al finalizer (P2-CHAT-EXPLICIT-SLOT-WISH)"


def test_sameday_backstop_in_swap_and_seed_in_chat():
    assert "SAME_DAY_PROTEIN_REPEAT" in _AGENT, "swap sin backstop determinista same-day"
    assert "RETRY VARIEDAD DEL DÍA" in _AGENT, "el retry same-day debe ser acotado (marker en prompt)"
    assert "VARIEDAD DEL DÍA (preferencia fuerte): las OTRAS comidas de ESE día" in _TOOLS, \
        "chat-modify no siembra las comidas del día propio"


def test_crossday_preps_reported():
    plan = {"days": [
        {"day": d, "meals": [{"name": "Panqueques de avena", "ingredients": ["50g de avena"]}]}
        for d in range(1, 6)
    ]}
    vr = g.build_variety_report(plan)
    assert "cross_day_preps" in vr
    assert vr["cross_day_preps"].get("panqueque") == 5, f"esperado panqueque en 5 días: {vr['cross_day_preps']}"


def test_name_honesty_extended_leaders():
    for k in ("chivo", "huevo", "queso", "jamon", "cordero", "conejo"):
        assert k in g._PHANTOM_PROTEIN_SYNS, f"líder '{k}' no detectable (P2-NAME-HONESTY-EXT)"
    assert "huevo" not in g._PHANTOM_PROTEIN_REPLACEMENT_OK, "huevo NUNCA como reemplazo (dairy no se toca)"
    assert "queso" not in g._PHANTOM_PROTEIN_REPLACEMENT_OK


def test_phantom_chivo_renamed_to_real_meat(monkeypatch):
    monkeypatch.setattr(g, "PHANTOM_PROTEIN_NAMEFIX_ENABLED", True)
    from constants import strip_accents
    meal = {"name": "Chivo guisado a la dominicana",
            "ingredients": ["150g de pollo", "1 taza de yuca"], "recipe": []}
    assert g._fix_phantom_protein_in_name(meal, strip_accents) is True
    assert "chivo" not in meal["name"].lower() and "pollo" in meal["name"].lower()


def test_phantom_never_renames_to_queso(monkeypatch):
    monkeypatch.setattr(g, "PHANTOM_PROTEIN_NAMEFIX_ENABLED", True)
    if hasattr(g, "PHANTOM_PROTEIN_DEGRADE_FLAG"):
        monkeypatch.setattr(g, "PHANTOM_PROTEIN_DEGRADE_FLAG", True)
    from constants import strip_accents
    meal = {"name": "Cerdo a la parrilla", "ingredients": ["80g de queso cheddar", "1 batata"], "recipe": []}
    assert g._fix_phantom_protein_in_name(meal, strip_accents) is False
    assert meal["name"] == "Cerdo a la parrilla"
    assert meal.get("_name_honesty_degraded") is True


def test_swap_band_warning_in_router_and_frontend():
    assert '"swap_quality_warning"' in _PLANS or "swap_quality_warning" in _PLANS
    ctx = (_FRONT / "context" / "AssessmentContext.jsx").read_text(encoding="utf-8")
    assert "swap_quality_warning" in ctx, "el frontend no renderiza el warning del swap"


# ───────────────────────── recetas ─────────────────────────
def test_insert_before_montaje_helper():
    steps = ["Mise en place: corta.", "El Toque de Fuego: cocina 10 min.", "Montaje: sirve."]
    out = g._insert_step_before_montaje(steps, "💪 añade pollo.")
    assert out.index("💪 añade pollo.") == 2, "el paso debe insertarse ANTES de Montaje"
    assert out[-1].startswith("Montaje")
    out2 = g._insert_step_before_montaje(["Paso único"], "extra")
    assert out2 == ["Paso único", "extra"]  # sin Montaje → append


def test_step_contract_lint():
    ok = {"recipe": ["Mise en place: pesa 50 g.", "El Toque de Fuego: 10 min a fuego medio.", "Montaje: sirve."]}
    assert g._recipe_step_contract_issues(ok) == []
    bad = {"recipe": ["Cocina.", "Sirve."]}
    issues = g._recipe_step_contract_issues(bad)
    assert issues and any("Mise" in i for i in issues)
    no_time = {"recipe": ["Mise en place: x.", "El Toque de Fuego: cocina bien.", "Montaje: sirve."]}
    assert any("tiempo" in i for i in g._recipe_step_contract_issues(no_time))
    report = g.compute_dish_quality_report({"days": [{"day": 1, "meals": [bad]}]})
    assert report.get("contract_meals") == 1 and "contract_ratio" in report


def test_persist_boundary_includes_reverse_and_offcatalog():
    i = _GRAPH.find("def finalize_plan_data_coherence")
    # [P2-AUDIT-V2-BATCH · 2026-07-01] ventana 7000→9000: el bloque qty-presence (P2-QTY-PRESENCE-PERSIST)
    # se insertó antes del strip; el contrato anclado (reverse + offcatalog en el boundary) no cambió.
    seg = _GRAPH[i:i + 9000]
    assert "_ensure_ingredients_used_in_recipe" in seg and "_strip_offcatalog_condiments_from_recipe" in seg, \
        "persist boundary sin reverse-coherence/off-catalog strip (P2-PERSIST-BOUNDARY-COHERENCE)"


def test_fallback_recipe_steps_by_category():
    steps = g._fallback_recipe_steps("Almuerzo", ["150g de pollo", "1 taza de arroz"])
    assert any("plancha" in s.lower() for s in steps) and any("min" in s.lower() for s in steps)
    shake = g._fallback_recipe_steps("Merienda", ["1 taza de leche", "1 guineo", "batido de fresa"])
    assert any("licúa" in s.lower() or "licua" in s.lower() for s in shake)
    oat = g._fallback_recipe_steps("Desayuno", ["1/2 taza de avena", "1 taza de leche"])
    assert any("avena" in s.lower() for s in oat)


def test_expand_requires_three_steps():
    assert "len(clean_steps) >= 3" in _AIH, \
        "la expansión debe exigir ≥3 pasos (una expansión degenerada no reemplaza la receta)"


def test_expand_veg_truthup():
    i = _PLANS.find("def _append_expand_veg")
    seg = _PLANS[i:i + 2200]
    assert "_truth_up_meal_macros_from_strings" in seg, \
        "el veg añadido en expand no recomputaba macros (P2-EXPAND-VEG-TRUTHUP)"


def test_household_note_in_recipe_views():
    for f in ("RecipesView.jsx", "MobileRecipes.jsx"):
        src = (_FRONT / "components" / "recipes" / f).read_text(encoding="utf-8")
        assert "Porciones para 1 persona" in src, f"{f}: falta la nota household (P2-RECIPE-HOUSEHOLD-NOTE)"
