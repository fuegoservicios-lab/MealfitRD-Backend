"""[P0-EXPAND-CONDITION-GUARD · 2026-07-01] (audit objetivo v2 · P0-1 · /recipe/expand condición)

Gap que cierra: el guard clínico de expand (P0-EXPAND-CLINICAL-GUARD) escanea los pasos del Chef AI
con `clinical_backstop_for_meal` — que cubre alérgeno IgE + dieta + mercurio-embarazo pero NO reglas
de CONDICIÓN (DM2/HTA/dislipidemia). El propio docstring del guard afirmaba proteger contra
"'endulza con miel' a un DM2", cobertura que la implementación no tenía: la expansión con
`"El Toque de Fuego: endulza con 2 cucharadas de miel"` para un DM2 pasaba el scan, cobraba cuota
y PERSISTÍA el paso contraindicado. El backstop sustitutivo (`condition_substitution_backstop_for_meal`)
existe pero opera sobre `ingredients[]`, no sobre prosa de pasos, y no se invocaba en expand.

Fix:
  - Detector SSOT `condition_prohibited_violations_for_meal` (graph_orchestrator): reusa los MISMOS
    tokens+negatives de `condition_rules.collect_substitutions` pero DETECTA en vez de sustituir
    (abortivo — rechazar es barato en expand: soft-fail 200 + retry; reescribir prosa es frágil).
    NO escanea `name` (pre-existente → bloquearía el retry para siempre) y SALTA pasos-nota ⚠/💡/⚕
    (la nota "se sustituyó miel → Stevia" contiene el token ofensor por diseño). Best-effort.
  - Wiring en `api_expand_recipe`: extiende `_exp_viols` ANTES de cobrar → cae en el mismo soft-fail
    `clinical_check_failed` del guard de alérgenos (sin cobro, sin persist, sin isExpanded).
  - `_expand_clin` seedea `medicalConditions` desde el body; `_enrich_clinical_from_profile` hidrata
    del perfil para autenticados (ya cubría medicalConditions + free-text plegado).
  - Knob de rollback sin redeploy: MEALFIT_EXPAND_CONDITION_GUARD (default ON).

Tests: (1) funcionales del detector (DM2/HTA/dislipidemia, negatives, notas, sin condiciones, knob);
(2) parser-based del wiring/orden en el handler.
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
# 1. Funcional: detector de términos contraindicados por condición
# ---------------------------------------------------------------------------
_DM2 = {"medicalConditions": ["Diabetes tipo 2"]}
_HTA = {"medicalConditions": ["Hipertensión"]}
_DYS = {"medicalConditions": ["Colesterol alto"]}


def test_dm2_honey_step_detected():
    """El escenario literal del audit: miel en un paso expandido para un DM2."""
    viols = g.condition_prohibited_violations_for_meal(
        {"ingredients": ["El Toque de Fuego: endulza con 2 cucharadas de miel."]}, _DM2)
    assert viols, "miel en paso expandido NO detectada para DM2 (P0-EXPAND-CONDITION-GUARD)"
    assert any("miel" in v for v in viols)


def test_dm2_negative_respected():
    """'sin azúcar' es negative del SSOT → no debe disparar falso positivo."""
    viols = g.condition_prohibited_violations_for_meal(
        {"ingredients": ["Endulza con canela y vainilla, sin azúcar añadida."]}, _DM2)
    assert viols == [], f"negative 'sin azúcar' ignorado: {viols}"


def test_hta_bouillon_step_detected():
    viols = g.condition_prohibited_violations_for_meal(
        {"ingredients": ["El Toque de Fuego: añade un cubito de pollo al agua hirviendo."]}, _HTA)
    assert viols, "cubito en paso expandido NO detectado para HTA"


def test_dyslipidemia_butter_detected_peanut_butter_safe():
    viols = g.condition_prohibited_violations_for_meal(
        {"ingredients": ["Termina con una cucharada de mantequilla derretida."]}, _DYS)
    assert viols, "mantequilla en paso expandido NO detectada para dislipidemia"
    safe = g.condition_prohibited_violations_for_meal(
        {"ingredients": ["Sirve con una cucharada de mantequilla de maní."]}, _DYS)
    assert safe == [], f"'mantequilla de maní' (grasa insaturada, negative SSOT) marcó falso positivo: {safe}"


def test_no_conditions_noop():
    assert g.condition_prohibited_violations_for_meal(
        {"ingredients": ["Endulza con 2 cucharadas de miel."]}, {}) == []
    assert g.condition_prohibited_violations_for_meal(
        {"ingredients": ["Endulza con miel."]}, {"medicalConditions": []}) == []


def test_note_steps_skipped():
    """La nota clínica determinista contiene el token ofensor POR DISEÑO — no debe re-detectarse
    (bloquearía el retry para siempre si el Chef la echoea)."""
    viols = g.condition_prohibited_violations_for_meal(
        {"ingredients": [
            "⚕️ Ajuste clínico (condición médica): se sustituyó miel por una alternativa segura.",
            "⚠️ Seguridad alimentaria: cocina el huevo por completo.",
            "💡 Ajustamos las porciones para que tus calorías cuadren (incluye miel de referencia).",
        ]}, _DM2)
    assert viols == [], f"pasos-nota ⚠/💡/⚕ no fueron saltados: {viols}"


def test_name_not_scanned():
    """El nombre es PRE-existente (validado por S1): escanearlo bloquearía el retry para siempre."""
    viols = g.condition_prohibited_violations_for_meal(
        {"name": "Pollo glaseado a la miel", "ingredients": ["Hornea el pollo 25 min a 200°C."]}, _DM2)
    assert viols == [], f"el detector escaneó el nombre del plato: {viols}"


def test_knob_off_disables(monkeypatch):
    monkeypatch.setattr(g, "EXPAND_CONDITION_GUARD", False)
    assert g.condition_prohibited_violations_for_meal(
        {"ingredients": ["Endulza con 2 cucharadas de miel."]}, _DM2) == []


def test_best_effort_on_error(monkeypatch):
    """Espejo P1-MERCURY-UPDATE-GUARD: error del scanner → [] (no bloquea todos los expand);
    el fail-secure abortivo queda para alérgenos IgE en clinical_backstop_for_meal."""
    import condition_rules
    def _boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(condition_rules, "collect_substitutions", _boom)
    assert g.condition_prohibited_violations_for_meal(
        {"ingredients": ["Endulza con miel."]}, _DM2) == []


# ---------------------------------------------------------------------------
# 2. Parser-based: wiring en el handler de expand
# ---------------------------------------------------------------------------
def test_condition_scan_wired_in_expand(expand_body):
    assert "condition_prohibited_violations_for_meal" in expand_body, \
        "el handler no invoca el detector de condición sobre los pasos expandidos"


def test_condition_scan_runs_before_charge(expand_body):
    """El scan de condición corre ANTES de `log_api_usage` — una violación NO debe cobrar cuota
    (mismo contrato que el scan de alérgenos del guard)."""
    i_scan = expand_body.find("condition_prohibited_violations_for_meal")
    i_charge = expand_body.find('log_api_usage(user_id, "llm_recipe_expand")')
    assert i_scan != -1 and i_charge != -1
    assert i_scan < i_charge, "el scan de condición debe correr ANTES de cobrar cuota"


def test_condition_violations_feed_same_soft_fail(expand_body):
    """Las violaciones de condición se SUMAN a `_exp_viols` → caen en el mismo soft-fail
    `clinical_check_failed` (sin persist, receta original, retry posible)."""
    i_extend = expand_body.find("_exp_viols.extend(_cpv_exp(")
    i_check = expand_body.find("if _exp_viols:")
    assert i_extend != -1, "el detector no alimenta _exp_viols (¿rama soft-fail propia? debe reusar la del guard)"
    assert i_check != -1 and i_extend < i_check, \
        "el extend debe ocurrir ANTES del check `if _exp_viols:` del soft-fail"


def test_expand_clin_seeds_medical_conditions(expand_body):
    assert '"medicalConditions": data.get("medicalConditions")' in expand_body, \
        "_expand_clin no seedea medicalConditions desde el body (guests/clients sin perfil)"


def test_marker_anchor_present(expand_body):
    assert "P0-EXPAND-CONDITION-GUARD" in expand_body, "falta el tooltip-anchor en el handler"


def test_knob_registered_default_on():
    assert getattr(g, "EXPAND_CONDITION_GUARD") is True, \
        "MEALFIT_EXPAND_CONDITION_GUARD debe defaultear ON (clínico)"
