"""[P0-EXPAND-CLINICAL-GUARD · 2026-07-01] (audit P0-1 · /recipe/expand hardening clínico)

Dos modos de fallo del endpoint `/recipe/expand` (el Chef AI reemplaza `recipe[]` completo):

1. SIN capa clínica determinista: los pasos expandidos son prompt-trustable, NO enforced (misma
   clase que P0-AGENT-1) → podían re-introducir un alérgeno IgE o un prohibido de condición/dieta
   que la generación había sustituido ("endulza con miel" a un DM2, camarones a un alérgico).
2. BORRABA notas deterministas: las notas food-safety (⚠ huevo/pescado/víver crudos) y los
   disclaimers del solver (💡) viven COMO pasos de recipe[] → al reemplazar, un usuario con
   ceviche perdía la advertencia de cocción.

Fix:
  - Hidratación server-side de alergias/dieta/condiciones (`_enrich_clinical_from_profile`,
    UNION body+perfil) ANTES de usar los pasos.
  - Scan clínico determinista de los pasos expandidos (SSOT `clinical_backstop_for_meal`)
    ANTES de cobrar cuota → violación = soft-fail HTTP 200 (`clinical_check_failed`), sin
    cobro, sin persist, sin isExpanded.
  - `_set_expanded_recipe_preserving_notes` en el persist callback: preserva pasos-nota (⚠/💡)
    del recipe original + re-deriva food-safety desde ingredients (idempotente). La respuesta
    del endpoint refleja lo persistido.
  - Knob de rollback sin redeploy: MEALFIT_EXPAND_CLINICAL_GUARD (default ON).

Tests: (1) funcional del detector de pasos-nota; (2) parser-based del wiring/orden en el handler.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import graph_orchestrator as g

_BACKEND = Path(__file__).resolve().parent.parent
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")


def _extract_function_body(src: str, fn_name: str) -> str:
    pattern = re.compile(rf"def\s+{re.escape(fn_name)}\s*\(")
    m = pattern.search(src)
    assert m, f"No se encontró `def {fn_name}(` en plans.py"
    start = m.start()
    next_def = re.search(r"\n(?:@router\.|@app\.|def\s)", src[start + 1:])
    end = (start + 1 + next_def.start()) if next_def else len(src)
    return src[start:end]


@pytest.fixture(scope="module")
def expand_body() -> str:
    return _extract_function_body(_PLANS, "api_expand_recipe")


# ---------------------------------------------------------------------------
# 1. Funcional: detector de pasos-nota (SSOT de la preservación)
# ---------------------------------------------------------------------------
def test_note_step_detector():
    assert g._is_recipe_safety_note_step(
        "⚠️ Seguridad alimentaria: cocina el huevo por completo (≥71°C)."
    ) is True
    assert g._is_recipe_safety_note_step(
        "💡 Ajustamos ligeramente las porciones para que tus calorías del día cuadren."
    ) is True
    assert g._is_recipe_safety_note_step("Mise en place: corta el pollo en cubos de 2 cm.") is False
    assert g._is_recipe_safety_note_step(None) is False
    assert g._is_recipe_safety_note_step("") is False


# ---------------------------------------------------------------------------
# 2. Parser-based: hidratación + scan pre-cobro + soft-fail + preservación
# ---------------------------------------------------------------------------
def test_clinical_enrichment_hydrated_server_side(expand_body):
    assert "_enrich_clinical_from_profile(_expand_clin, user_id)" in expand_body, \
        "el handler no hidrata alergias/dieta/condiciones server-side (P0-EXPAND-CLINICAL-GUARD)"


def test_knob_present_default_on(expand_body):
    assert 'os.environ.get("MEALFIT_EXPAND_CLINICAL_GUARD", "true")' in expand_body, \
        "falta el knob de rollback MEALFIT_EXPAND_CLINICAL_GUARD con default ON"


def test_clinical_scan_runs_before_charge(expand_body):
    """El scan clínico corre ANTES de `log_api_usage` — una violación NO debe cobrar cuota
    (mismo contrato que el fail-signal y el check de coherencia)."""
    i_scan = expand_body.find("clinical_backstop_for_meal")
    i_charge = expand_body.find('log_api_usage(user_id, "llm_recipe_expand")')
    assert i_scan != -1, "el handler no invoca clinical_backstop_for_meal sobre los pasos expandidos"
    assert i_charge != -1
    assert i_scan < i_charge, "el scan clínico debe correr ANTES de cobrar cuota"


def test_violation_soft_fails_without_persist(expand_body):
    """Violación → soft-fail HTTP 200 con `clinical_check_failed` + receta original, ANTES del
    bloque de persistencia (patrón P3-SWAP-SOFT-FAIL-200, espejo del coherence check)."""
    i_flag = expand_body.find('"clinical_check_failed": True')
    i_persist = expand_body.find("update_plan_data_atomic")
    assert i_flag != -1, "el soft-fail no expone clinical_check_failed al cliente"
    assert i_persist != -1
    assert i_flag < i_persist, "el soft-fail clínico debe cortar ANTES de la persistencia"
    seg = expand_body[i_flag - 600: i_flag + 400]
    assert '"success": False' in seg and "req_recipe_original" in seg, \
        "el soft-fail debe devolver success=False + la receta original para display"


def test_persist_preserves_notes_both_paths(expand_body):
    """Ambos caminos del callback (índices + propagación por contenido) deben usar el helper
    de preservación de notas — cero asignaciones crudas `[\"recipe\"] = expanded_steps`."""
    n_calls = expand_body.count("_set_expanded_recipe_preserving_notes(")
    assert n_calls >= 3, (
        f"esperadas ≥3 apariciones del helper (1 def + 2 caminos), hay {n_calls} — "
        "¿un camino volvió a asignar recipe crudo?"
    )
    assert '["recipe"] = expanded_steps' not in expand_body, \
        "asignación cruda de recipe detectada — pierde las notas ⚠/💡 (P0-EXPAND-CLINICAL-GUARD)"


def test_persist_rederives_food_safety(expand_body):
    assert "food_safety_backstop_for_meal" in expand_body, \
        "el callback no re-deriva las notas food-safety sobre el meal expandido"


def test_response_includes_preserved_notes(expand_body):
    """La respuesta refleja lo persistido (pasos + notas preservadas), no los pasos crudos."""
    assert '"expanded_recipe": _resp_steps' in expand_body, \
        "la respuesta debe devolver _resp_steps (pasos + notas ⚠/💡 preservadas)"


def test_marker_anchor_present(expand_body):
    assert "P0-EXPAND-CLINICAL-GUARD" in expand_body, "falta el tooltip-anchor en el handler"
