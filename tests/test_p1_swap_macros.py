"""[P1-SWAP-MACROS · 2026-05-22] Tests del bundle Cambiar Plato production-ready.

Cierra los 3 gaps identificados en el audit del flow `/swap-meal`:

1. **Macros validation post-gen** — el prompt LLM solo inyectaba `target_calories`
   como hint soft, sin validación. LLM podía emitir 450kcal/8g protein vs
   target 350kcal/15g sin queja → macros semanales driftaban +28% kcal -47%
   protein. Fix: `validate_meal_macros_against_targets` + retry tenacity con
   feedback inyectado al prompt.

2. **Strict pantry mode** — razón `budget`/`pantry_first` del modal solo
   inyectaba hint cosmético al prompt, no restringía la fallback a pantry.
   Si LLM fallaba 3×, el fallback usaba ["Pollo", "Arroz", "Aguacate"]
   hardcodeados (pueden NO estar en nevera). Fix: en strict_pantry + sin
   pantry → 422 explícito; en strict_pantry + con pantry → fallback usa
   solo `clean_ingredients`.

3. **Coherence escalation post-swap** — `_recompute_aggregates_after_swap`
   solo emitía telemetría a `_shopping_coherence_block_history`; el cron
   diario P3-B alertaba 6-24h después. Fix: cuando `critical_count > 0`
   inyectamos `plan_data._swap_coherence_warnings` que el Dashboard renderea
   inline como banner amber (mismo lenguaje visual que `_quality_degraded`).

Tests organizados por fix con anchors de tooltip:
  - Section A: validador unitario (validate_meal_macros_against_targets)
  - Section B: integración structural en swap_meal (parser-based grep)
  - Section C: integración structural en execute_modify_single_meal
  - Section D: prompts inyectan los 4 targets (parser-based)
  - Section E: strict_pantry derivation + router 422 mapping
  - Section F: coherence escalation user-visible
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from unittest.mock import patch

import pytest


_BACKEND = Path(__file__).resolve().parents[1]


# =====================================================================
# Section A — Validador unitario `validate_meal_macros_against_targets`
# =====================================================================

def test_validate_macros_pass_within_tolerance():
    from nutrition_calculator import validate_meal_macros_against_targets as _v

    meal = {"cals": 360, "protein": 16, "carbs": 38, "fats": 14}
    target = {"cals": 350, "protein": 15, "carbs": 40, "fats": 15}
    passed, drifts, summary = _v(meal, target, tolerance_pct=0.15)

    assert passed is True
    assert summary == ""
    assert set(drifts.keys()) == {"cals", "protein", "carbs", "fats"}
    assert drifts["protein"]["delta_pct"] < 0.15


def test_validate_macros_fail_protein_drift():
    from nutrition_calculator import validate_meal_macros_against_targets as _v

    # Target 15g, actual 8g → 47% drift (clearly > 15%)
    meal = {"cals": 350, "protein": 8, "carbs": 40, "fats": 15}
    target = {"cals": 350, "protein": 15, "carbs": 40, "fats": 15}
    passed, drifts, summary = _v(meal, target, tolerance_pct=0.15)

    assert passed is False
    assert "protein" in summary.lower()
    assert "MACROS FUERA DE OBJETIVO" in summary
    assert drifts["protein"]["delta_pct"] >= 0.40


def test_validate_macros_cals_tolerance_15x_wider():
    """cals tolera 1.5× la base — kcal varía más por porciones/guarniciones."""
    from nutrition_calculator import validate_meal_macros_against_targets as _v

    # protein/carbs/fats exactos; cals 18% over (vs 15% base, 22.5% cals threshold)
    meal = {"cals": 413, "protein": 15, "carbs": 40, "fats": 15}
    target = {"cals": 350, "protein": 15, "carbs": 40, "fats": 15}
    passed, _, _ = _v(meal, target, tolerance_pct=0.15)
    assert passed is True, "cals dentro de 22.5% (1.5× base) debe pasar"

    # Ahora 25% over en cals → debe fallar
    meal["cals"] = 438
    passed, _, summary = _v(meal, target, tolerance_pct=0.15)
    assert passed is False
    assert "cals" in summary.lower()


def test_validate_macros_skip_when_target_zero():
    """Si target=0/None para una key, esa key se omite (no se enforce drift)."""
    from nutrition_calculator import validate_meal_macros_against_targets as _v

    meal = {"cals": 350, "protein": 100, "carbs": 40, "fats": 15}
    target = {"cals": 350, "protein": 0, "carbs": 40, "fats": 15}  # protein target=0
    passed, drifts, _ = _v(meal, target, tolerance_pct=0.15)

    assert passed is True
    assert "protein" not in drifts, "target=0 → no enforce → no drift report"


def test_validate_macros_accepts_alias_keys():
    """Acepta `fat` (singular), `calories`, `protein_g`, etc."""
    from nutrition_calculator import validate_meal_macros_against_targets as _v

    meal = {"calories": 350, "protein_g": 15, "carbs_g": 40, "fat": 15}
    target = {"cals": 350, "protein": 15, "carbs": 40, "fats": 15}
    passed, drifts, _ = _v(meal, target)
    assert passed is True
    assert drifts["fats"]["actual"] == 15


def test_validate_macros_kill_switch_default_true():
    """`_meal_macros_validate_enabled()` default True."""
    import os
    from nutrition_calculator import _meal_macros_validate_enabled

    _prev = os.environ.pop("MEALFIT_SWAP_MACROS_VALIDATE", None)
    try:
        assert _meal_macros_validate_enabled() is True
        os.environ["MEALFIT_SWAP_MACROS_VALIDATE"] = "false"
        assert _meal_macros_validate_enabled() is False
        os.environ["MEALFIT_SWAP_MACROS_VALIDATE"] = "true"
        assert _meal_macros_validate_enabled() is True
    finally:
        os.environ.pop("MEALFIT_SWAP_MACROS_VALIDATE", None)
        if _prev is not None:
            os.environ["MEALFIT_SWAP_MACROS_VALIDATE"] = _prev


def test_validate_macros_tolerance_knob_clamp():
    import os
    from nutrition_calculator import _meal_macros_tolerance_pct

    _prev = os.environ.pop("MEALFIT_SWAP_MACROS_TOLERANCE_PCT", None)
    try:
        assert _meal_macros_tolerance_pct() == 0.15  # default
        os.environ["MEALFIT_SWAP_MACROS_TOLERANCE_PCT"] = "0.01"
        assert _meal_macros_tolerance_pct() == 0.05  # clamped to lower bound
        os.environ["MEALFIT_SWAP_MACROS_TOLERANCE_PCT"] = "0.99"
        assert _meal_macros_tolerance_pct() == 0.50  # clamped to upper bound
        os.environ["MEALFIT_SWAP_MACROS_TOLERANCE_PCT"] = "garbage"
        assert _meal_macros_tolerance_pct() == 0.15  # fallback
    finally:
        os.environ.pop("MEALFIT_SWAP_MACROS_TOLERANCE_PCT", None)
        if _prev is not None:
            os.environ["MEALFIT_SWAP_MACROS_TOLERANCE_PCT"] = _prev


# =====================================================================
# Section B — Integración estructural en swap_meal (agent.py)
# =====================================================================

def _agent_source() -> str:
    return (_BACKEND / "agent.py").read_text(encoding="utf-8")


def test_swap_meal_imports_macros_validator():
    """[P1-SWAP-MACROS-VALIDATOR anchor] swap_meal debe importar el helper
    desde nutrition_calculator dentro del scope local (lazy import)."""
    src = _agent_source()
    assert "validate_meal_macros_against_targets" in src, (
        "swap_meal debe referenciar el validador (lazy import o callable)."
    )
    assert "_meal_macros_validate_enabled" in src, (
        "swap_meal debe respetar el kill switch."
    )


def test_swap_meal_invokes_validator_in_retry_loop():
    """El validador DEBE invocarse DENTRO del invoke_with_retry y, al fallar,
    DEBE inyectar el summary al prompt para el próximo retry."""
    src = _agent_source()
    # Localizar el bloque de swap_meal (delimitado por def + función siguiente)
    swap_block_match = re.search(
        r"def swap_meal\(form_data:.*?(?=^def |^# =+\n# ORQ)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert swap_block_match, "No se localizó el cuerpo de swap_meal."
    swap_block = swap_block_match.group(0)

    # Macros validator invoke
    assert "_validate_macros" in swap_block, "validator no invocado en swap_meal"
    assert "P1-SWAP-MACROS" in swap_block, "anchor del tooltip P1-SWAP-MACROS ausente"
    # Inyección de feedback al retry prompt
    assert "ATENCIÓN AL INTENTO FALLIDO ANTERIOR" in swap_block, (
        "el summary debe inyectarse al next-attempt prompt"
    )


# =====================================================================
# Section C — Integración estructural en execute_modify_single_meal (tools.py)
# =====================================================================

def _tools_source() -> str:
    return (_BACKEND / "tools.py").read_text(encoding="utf-8")


def test_execute_modify_single_meal_invokes_validator():
    src = _tools_source()
    block = re.search(
        r"def execute_modify_single_meal\(.*?(?=^@tool|^def )",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert block, "No se localizó el cuerpo de execute_modify_single_meal."
    body = block.group(0)
    assert "_validate_macros" in body
    assert "original_protein" in body, "targets per-meal deben pasarse al prompt"
    assert "P1-SWAP-MACROS" in body


# =====================================================================
# Section D — Prompts inyectan los 4 targets
# =====================================================================

def test_swap_prompt_includes_4_targets():
    src = (_BACKEND / "prompts" / "meal_operations.py").read_text(encoding="utf-8")
    swap_block = re.search(r"SWAP_MEAL_PROMPT_TEMPLATE\s*=\s*\"\"\"(.*?)\"\"\"", src, re.DOTALL)
    assert swap_block, "SWAP_MEAL_PROMPT_TEMPLATE no encontrado"
    body = swap_block.group(1)
    for token in ("{target_calories}", "{target_protein}", "{target_carbs}", "{target_fats}"):
        assert token in body, f"placeholder {token} ausente en SWAP_MEAL_PROMPT_TEMPLATE"


def test_modify_prompt_includes_4_targets():
    src = (_BACKEND / "prompts" / "meal_operations.py").read_text(encoding="utf-8")
    mod_block = re.search(r"MODIFY_MEAL_PROMPT_TEMPLATE\s*=\s*\"\"\"(.*?)\"\"\"", src, re.DOTALL)
    assert mod_block, "MODIFY_MEAL_PROMPT_TEMPLATE no encontrado"
    body = mod_block.group(1)
    for token in ("{original_cals}", "{original_protein}", "{original_carbs}", "{original_fats}"):
        assert token in body, f"placeholder {token} ausente en MODIFY_MEAL_PROMPT_TEMPLATE"


# =====================================================================
# Section E — Strict pantry: derivation + router 422 mapping
# =====================================================================

def test_swap_meal_derives_strict_pantry_from_reason():
    """[P3-SWAP-PANTRY-DEFAULT · 2026-05-22] Tras flip a strict-by-default,
    la derivación es ``swap_reason not in ("cravings", "weekend")`` —
    cravings/weekend opt-out, todo lo demás (incluido budget/pantry_first
    back-compat) → strict."""
    src = _agent_source()
    assert 'swap_reason not in ("cravings", "weekend")' in src, (
        "strict_pantry derivation post-P3-SWAP-PANTRY-DEFAULT debe ser "
        "`swap_reason not in (\"cravings\", \"weekend\")` (default strict)."
    )
    assert "SWAP_STRICT_PANTRY_NO_INVENTORY" in src, (
        "marker de raise para 422 ausente"
    )
    assert "P1-SWAP-STRICT-PANTRY" in src, "anchor del tooltip ausente"


def test_router_maps_strict_pantry_error_to_422():
    src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    # Localiza el handler api_swap_meal
    handler_match = re.search(
        r"def api_swap_meal\(.*?(?=@router\.|\ndef api_swap_meal_persist)",
        src,
        re.DOTALL,
    )
    assert handler_match, "api_swap_meal no encontrado"
    body = handler_match.group(0)
    assert "SWAP_STRICT_PANTRY_NO_INVENTORY" in body, (
        "router debe atrapar el marker para retornar 422"
    )
    assert "status_code=422" in body, "el handler debe responder 422 en strict pantry sin inventory"


# =====================================================================
# Section F — Coherence escalation user-visible
# =====================================================================

@pytest.fixture
def base_final_state():
    return {
        "plan_result": {
            "days": [{"day": 1, "meals": []}],
            "calc_household_multiplier": 1.0,
        },
        "form_data": {
            "user_id": None,
            "groceryDuration": "weekly",
        },
    }


@pytest.fixture
def stub_aggregates(monkeypatch):
    import shopping_calculator as sc
    monkeypatch.setattr(sc, "get_shopping_list_delta", lambda *a, **kw: [])
    monkeypatch.setattr(sc, "fetch_inventory_and_consumed_for_plan", lambda *a, **kw: ({}, {}))
    monkeypatch.setattr(sc, "_build_hybrid_shopping_list", lambda a, b: a or b or [])


def test_severe_coherence_injects_user_visible_warnings(base_final_state, stub_aggregates):
    """Cuando hay divergencias críticas (cap_swallowed_modifier o magnitude
    >30%), `plan_result._swap_coherence_warnings` debe quedar inyectado para
    que el Dashboard renderee banner amber."""
    fake_div = [
        {"ingredient": "Pollo", "hypothesis": "cap_swallowed_modifier", "magnitude": False, "delta_pct": 0.0},
        {"ingredient": "Arroz", "hypothesis": "magnitude_overflow", "magnitude": True, "delta_pct": 0.45},
        {"ingredient": "Tomate", "hypothesis": "yield_uncovered", "magnitude": False, "delta_pct": 0.0},
    ]

    import shopping_calculator as sc
    with patch.object(sc, "run_shopping_coherence_guard", lambda *a, **kw: fake_div):
        from graph_orchestrator import _recompute_aggregates_after_swap
        asyncio.run(_recompute_aggregates_after_swap(base_final_state))

    warnings = base_final_state["plan_result"].get("_swap_coherence_warnings")
    assert warnings is not None, "_swap_coherence_warnings debe inyectarse"
    assert warnings["critical_count"] == 2, "cap_swallowed + magnitude>30% = 2 criticos"
    assert len(warnings["summary"]) == 2
    assert any(s["hypothesis"] == "cap_swallowed_modifier" for s in warnings["summary"])
    assert "detected_at" in warnings


def test_no_severe_no_warnings_injection(base_final_state, stub_aggregates):
    """Sin divergencias críticas (solo yield_uncovered, magnitudes <30%) → NO
    se inyecta el campo user-visible. La telemetría warn-only sigue su curso."""
    fake_div = [
        {"ingredient": "X", "hypothesis": "yield_uncovered", "magnitude": False, "delta_pct": 0.0},
        {"ingredient": "Y", "hypothesis": "magnitude_minor", "magnitude": True, "delta_pct": 0.15},
    ]

    import shopping_calculator as sc
    with patch.object(sc, "run_shopping_coherence_guard", lambda *a, **kw: fake_div):
        from graph_orchestrator import _recompute_aggregates_after_swap
        asyncio.run(_recompute_aggregates_after_swap(base_final_state))

    assert "_swap_coherence_warnings" not in base_final_state["plan_result"], (
        "Sin divergencias críticas, NO debe inyectarse el banner. "
        "Solo cap_swallowed_modifier o magnitude>30% gatean el user-visible."
    )


def test_coherence_escalate_knob_kill_switch(base_final_state, stub_aggregates, monkeypatch):
    """Flip de `MEALFIT_SWAP_COHERENCE_BLOCK_SEVERE_ONLY=false` desactiva la
    inyección user-visible (revierte al telemetry-only pre-fix)."""
    fake_div = [
        {"ingredient": "Pollo", "hypothesis": "cap_swallowed_modifier", "magnitude": False, "delta_pct": 0.0},
    ]
    monkeypatch.setenv("MEALFIT_SWAP_COHERENCE_BLOCK_SEVERE_ONLY", "false")

    import shopping_calculator as sc
    with patch.object(sc, "run_shopping_coherence_guard", lambda *a, **kw: fake_div):
        from graph_orchestrator import _recompute_aggregates_after_swap
        asyncio.run(_recompute_aggregates_after_swap(base_final_state))

    assert "_swap_coherence_warnings" not in base_final_state["plan_result"], (
        "Knob OFF debe revertir al pre-fix telemetry-only."
    )


def test_dashboard_banner_consumes_swap_coherence_warnings():
    """Parser-based: el banner amber del Dashboard debe leer
    `planData._swap_coherence_warnings.critical_count > 0`."""
    src = (Path(__file__).resolve().parents[2] / "frontend" / "src" / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")
    assert "_swap_coherence_warnings" in src, (
        "Dashboard.jsx debe consumir el campo inyectado por backend"
    )
    assert "P1-SWAP-COHERENCE-ESCALATE" in src, "anchor del fix ausente en Dashboard.jsx"


def test_coherence_surfaces_table_documents_escalation():
    """La tabla canónica debe mencionar P1-SWAP-COHERENCE-ESCALATE + el knob
    + `_swap_coherence_warnings` para que un futuro auditor entienda la
    diferencia entre warn-only-pre-fix y user-visible-post-fix."""
    src = (_BACKEND / "docs" / "coherence_surfaces_table.md").read_text(encoding="utf-8")
    assert "_swap_coherence_warnings" in src
    assert "P1-SWAP-COHERENCE-ESCALATE" in src
    assert "MEALFIT_SWAP_COHERENCE_BLOCK_SEVERE_ONLY" in src


# =====================================================================
# Section G — Recipe-ingredients coherence per-meal (P1-SWAP-RECIPE-COHERENCE)
# Cierra la limitación dejada documentada en el bundle inicial — el
# user-facing "Cambiar Plato" ahora valida coherencia receta↔ingredientes
# ANTES de retornar el meal, gateando retry tenacity.
# =====================================================================

def test_recipe_coherence_clean_meal_passes():
    from nutrition_calculator import validate_meal_recipe_ingredients_coherence as _check
    meal = {
        "ingredients": ["200g pechuga de pollo", "1 taza de arroz integral"],
        "recipe": [
            "Mise en place: corta el pollo en cubos.",
            "El Toque de Fuego: dora el pollo y agrega el arroz.",
            "Montaje: sirve.",
        ],
    }
    passed, divs, summary = _check(meal)
    assert passed is True
    assert divs == {}
    assert summary == ""


def test_recipe_coherence_cap_swallowed_modifier_detected():
    """Receta menciona pollo pero ingredients lista pavo → debe rechazar."""
    from nutrition_calculator import validate_meal_recipe_ingredients_coherence as _check
    meal = {
        "ingredients": ["200g pavo molido", "1 taza de arroz"],
        "recipe": [
            "Mise en place: corta el pollo en cubos.",
            "El Toque de Fuego: dora el pollo.",
            "Montaje: sirve con arroz.",
        ],
    }
    passed, divs, summary = _check(meal)
    assert passed is False
    assert "pollo" in divs
    assert "RECETA MENCIONA INGREDIENTES NO LISTADOS" in summary


def test_recipe_coherence_alias_pechuga_counts_as_pollo():
    """Aliases de PROTEIN_SYNONYMS (pechuga = pollo canónico) cuentan."""
    from nutrition_calculator import validate_meal_recipe_ingredients_coherence as _check
    meal = {
        "ingredients": ["1 pechuga de pollo"],
        "recipe": [
            "Mise en place: prepara el pollo.",
            "El Toque de Fuego: dora.",
            "Montaje: sirve.",
        ],
    }
    passed, _, _ = _check(meal)
    assert passed is True


def test_recipe_coherence_accent_insensitive():
    """salmon (sin acento) vs salmón (con acento) match correctamente."""
    from nutrition_calculator import validate_meal_recipe_ingredients_coherence as _check
    meal = {
        "ingredients": ["200g salmon"],
        "recipe": [
            "Mise en place: limpia el salmón.",
            "El Toque de Fuego: cocina.",
            "Montaje: sirve.",
        ],
    }
    passed, _, _ = _check(meal)
    assert passed is True


def test_recipe_coherence_word_boundary_no_substring_false_positive():
    """`_alias_appears_as_word` debe rechazar match parcial 'res' en 'estresante'."""
    from nutrition_calculator import _alias_appears_as_word
    assert _alias_appears_as_word("res", "cocina en un proceso estresante") is False
    assert _alias_appears_as_word("res", "agrega la res molida") is True


def test_recipe_coherence_empty_inputs_passthrough():
    """Sin ingredients o sin recipe → passed=True (no signal)."""
    from nutrition_calculator import validate_meal_recipe_ingredients_coherence as _check
    assert _check({})[0] is True
    assert _check({"ingredients": [], "recipe": []})[0] is True
    assert _check({"ingredients": ["pollo"], "recipe": []})[0] is True


def test_recipe_coherence_kill_switch():
    import os
    from nutrition_calculator import _swap_recipe_coherence_enabled
    _prev = os.environ.pop("MEALFIT_SWAP_RECIPE_COHERENCE_VALIDATE", None)
    try:
        assert _swap_recipe_coherence_enabled() is True
        os.environ["MEALFIT_SWAP_RECIPE_COHERENCE_VALIDATE"] = "false"
        assert _swap_recipe_coherence_enabled() is False
    finally:
        os.environ.pop("MEALFIT_SWAP_RECIPE_COHERENCE_VALIDATE", None)
        if _prev is not None:
            os.environ["MEALFIT_SWAP_RECIPE_COHERENCE_VALIDATE"] = _prev


def test_recipe_coherence_multi_protein_partial_divergence():
    """huevos OK + pollo missing → solo pollo en divs."""
    from nutrition_calculator import validate_meal_recipe_ingredients_coherence as _check
    meal = {
        "ingredients": ["2 huevos", "1 taza de arroz"],
        "recipe": [
            "Mise en place: bate los huevos y prepara el pollo.",
            "El Toque de Fuego: cocina los huevos.",
            "Montaje: sirve.",
        ],
    }
    passed, divs, _ = _check(meal)
    assert passed is False
    assert "pollo" in divs
    assert "huevos" not in divs


def test_recipe_coherence_fallback_subset_of_canonical():
    """El _FALLBACK_PROTEIN_SYNONYMS debe ser subconjunto del canónico de
    constants.py — si alguien expande el fallback, debe estar también en
    constants. Si constants no es importable (langchain falta en local),
    skip el test sin fallar."""
    pytest.importorskip("langchain_google_genai", reason="constants.py requiere langchain")
    from constants import PROTEIN_SYNONYMS as CANONICAL
    from nutrition_calculator import _FALLBACK_PROTEIN_SYNONYMS as FALLBACK

    for canonical_key, fb_aliases in FALLBACK.items():
        # La canonical key del fallback puede ser una agrupación más amplia
        # (p.ej. "habichuelas" cubre rojas/negras/blancas). Aceptamos match
        # si existe AL MENOS UN canonical en CANONICAL que tenga aliases
        # con substring match contra los del fallback.
        fb_set = {a.lower() for a in fb_aliases}
        canonical_all_aliases = set()
        for v in CANONICAL.values():
            canonical_all_aliases.update(a.lower() for a in v)
        # Cada alias del fallback debe aparecer en algún alias canónico
        # (substring tolerado: "habichuelas" → "habichuelas rojas")
        for fb_alias in fb_set:
            if not any(fb_alias in c_alias or c_alias in fb_alias for c_alias in canonical_all_aliases):
                pytest.fail(
                    f"Fallback alias {fb_alias!r} (canonical={canonical_key!r}) "
                    f"no tiene match en constants.PROTEIN_SYNONYMS — drift detectado."
                )


def test_swap_meal_invokes_recipe_coherence_validator():
    """[P1-SWAP-RECIPE-COHERENCE anchor] swap_meal DEBE invocar el validador
    de recipe-coherence dentro del retry loop ANTES del macros validator."""
    src = _agent_source()
    swap_block_match = re.search(
        r"def swap_meal\(form_data:.*?(?=^def |^# =+\n# ORQ)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert swap_block_match
    swap_block = swap_block_match.group(0)
    assert "_validate_recipe_coh" in swap_block
    assert "_recipe_coh_enabled" in swap_block
    assert "P1-SWAP-RECIPE-COHERENCE" in swap_block


def test_execute_modify_single_meal_invokes_recipe_coherence_validator():
    src = _tools_source()
    block = re.search(
        r"def execute_modify_single_meal\(.*?(?=^@tool|^def )",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert block
    body = block.group(0)
    assert "_validate_recipe_coh" in body
    assert "P1-SWAP-RECIPE-COHERENCE" in body
