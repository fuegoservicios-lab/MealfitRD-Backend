"""[P1-UPDATE-INTELLIGENCE-2 · 2026-06-24] Regresión de los 4 P1 de la RE-auditoría de inteligencia.

El cierre del 2026-06-23 scopeó el backstop clínico de updates SOLO a alérgenos+dieta. La re-auditoría
encontró 4 P1 abiertos en las superficies de UPDATE (swap S3 / regenerate-day S2 / chat-modify):

  P1-1  RENAL    — el cap renal KDIGO de proteína no se re-enforza en updates; peor, el retarget P1-3
                   podía SUBIR la proteína sobre el techo. Fix: `renal_protein_trim_for_update` + exclusión
                   de proteína del max(suma,meta) en perfiles renales + trim del día regenerado.
  P1-2  MERCURIO — pescado alto en mercurio (teratógeno) no es alérgeno IgE ni veg* → el backstop no lo
                   cazaba. Fix: `_scan_mercury_pregnancy_violations` dentro de `clinical_backstop_for_meal`
                   (abortivo, reusa `_PREGNANCY_MERCURY_SUBS` de S1). Knob MEALFIT_UPDATE_MERCURY_GUARD.
  P1-3  SURFACE  — el backend computa `day_quality_warning` pero el frontend lo descartaba. Fix: surfacing
                   en `regenerateDay` (toast ámbar accionable).
  P1-4  AI-PARTIAL — si la IA cae a mitad del loop de regenerate-day: persistía un día parcial, cobraba
                   crédito y PERDÍA platos (el break + full-replace truncaba). Fix: padear new_meals, NO
                   cobrar crédito, flag `ai_interrupted` + surfacing accionable.

Mayormente parser-based (el repo prueba estos puntos por parsing del source de prod, con tooltip-anchors
que fallan ante un renombre). Los funcionales que requieren importar `graph_orchestrator` se saltan en
entornos sin `langgraph` (corren en CI).
"""
import ast
import os
import re

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(BACKEND, rel), encoding="utf-8") as f:
        return f.read()


def _func_src(source, name):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"función {name!r} no encontrada")


AGENT = _read("agent.py")
TOOLS = _read("tools.py")
PLANS = _read("routers/plans.py")
ORCH = _read("graph_orchestrator.py")
ASSESS = _read("../frontend/src/context/AssessmentContext.jsx")
APP = _read("app.py")


# ── P1-1: cap renal en updates ────────────────────────────────────────────────
def test_p1_1_renal_trim_helper_exists_and_gated():
    assert "def renal_protein_trim_for_update" in ORCH, "falta el helper de trim renal de updates"
    src = _func_src(ORCH, "renal_protein_trim_for_update")
    assert "RENAL_CAP_ENABLED" in src, "el trim renal debe gatearse por RENAL_CAP_ENABLED (no por UPDATE_CLINICAL_GUARD)"
    assert "_trim_day_protein_to_ceiling" in src, "debe reusar la maquinaria validada de S1"
    assert "ceiling_pct=1.0" in src, "el techo es 1.0 (trima al cap, no lo excede)"


def test_p1_1_swap_runs_renal_trim_before_return():
    src = _func_src(AGENT, "swap_meal")
    assert "renal_protein_trim_for_update" in src, "swap debe trimar la proteína al techo renal"
    assert "_renal_capped" in src, "swap debe leer el flag renal del plan persistido"
    assert "renal_protein_cap" in src


def test_p1_1_modify_runs_renal_trim():
    src = _func_src(TOOLS, "execute_modify_single_meal")
    assert "renal_protein_trim_for_update" in src, "chat-modify debe trimar la proteína al techo renal"
    assert "renal_protein_cap" in src


def test_p1_1_regenerate_day_excludes_protein_from_retarget_when_renal():
    src = _func_src(PLANS, "api_regenerate_day")
    assert "_renal_capped" in src, "regenerate-day debe leer el flag renal del plan"
    # la proteína NO se sube hacia la meta en perfiles renales (sería iatrogénico)
    assert re.search(r'_kk\s*==\s*["\']protein_g["\']\s*and\s*_renal_capped', src), \
        "el retarget debe excluir protein_g del max() cuando el plan es renal-capeado"
    # defensa-en-profundidad: el día regenerado se trima al cap del día
    assert "renal_protein_trim_for_update" in src or "_rtu" in src


# ── P1-2: mercurio-embarazo en updates ────────────────────────────────────────
def test_p1_2_mercury_scanner_exists_and_reuses_s1_terms():
    assert "def _scan_mercury_pregnancy_violations" in ORCH
    src = _func_src(ORCH, "_scan_mercury_pregnancy_violations")
    assert "_is_pregnancy_or_lactation" in src, "solo debe disparar en embarazo/lactancia"
    assert "_PREGNANCY_MERCURY_SUBS" in src, "debe reusar los términos de S1 (SSOT, no duplicar la lista)"
    assert "MERCURY_UPDATE_GUARD" in src


def test_p1_2_mercury_knob_default_on():
    assert re.search(r'MERCURY_UPDATE_GUARD\s*=\s*_env_bool\(\s*["\']MEALFIT_UPDATE_MERCURY_GUARD["\']\s*,\s*True', ORCH), \
        "el guard de mercurio debe ser default ON (seguridad)"


def test_p1_2_backstop_accepts_form_data_and_scans_mercury():
    src = _func_src(ORCH, "clinical_backstop_for_meal")
    assert "form_data=None" in src, "el backstop debe aceptar form_data para detectar embarazo"
    assert "_scan_mercury_pregnancy_violations" in src


def test_p1_2_swap_passes_form_data_to_both_backstop_calls():
    src = _func_src(AGENT, "swap_meal")
    # ambos callsites (retry loop + guard final) pasan form_data
    assert src.count("form_data=form_data") >= 2, "swap debe pasar form_data a los 2 callsites del backstop"


def test_p1_2_modify_passes_form_data_with_conditions():
    src = _func_src(TOOLS, "execute_modify_single_meal")
    assert "form_data=_clin_form" in src, "modify debe pasar un form con condiciones al backstop"
    assert "medicalConditions" in src, "modify debe hidratar medicalConditions del perfil para el mercurio"


# ── P1-3: surfacing del aviso de calidad del día ──────────────────────────────
def test_p1_3_frontend_surfaces_day_quality_warning():
    src = _func_src(PLANS, "api_regenerate_day")
    assert "day_quality_warning" in src, "el backend debe seguir computando el aviso"
    # el frontend AHORA lo lee (antes lo descartaba)
    assert "day_quality_warning" in ASSESS, "regenerateDay debe surface el aviso (P1-3)"


# ── P1-4: IA caída a mitad → sin pérdida de platos, sin cobro, con aviso ───────
def test_p1_4_regenerate_day_pads_meals_on_ai_interrupt():
    src = _func_src(PLANS, "api_regenerate_day")
    # padea los platos restantes para no truncar el día con el full-replace
    assert "new_meals.extend(meals[len(new_meals):])" in src, "debe padear new_meals si la IA cayó a mitad"
    # NO cobra crédito en interrupción
    assert re.search(r'if\s+not\s+_ai_unavailable\s*:\s*\n\s*log_api_usage', src), \
        "no debe cobrar crédito cuando _ai_unavailable"
    # flag en la respuesta
    assert '"ai_interrupted"' in src
    assert "ai_interrupted_message" in src


def test_p1_4_frontend_surfaces_ai_interrupted():
    assert "ai_interrupted" in ASSESS, "regenerateDay debe surface la interrupción de IA con aviso accionable"


# ── marker + freshness ────────────────────────────────────────────────────────
def test_p1_marker_bumped():
    # [de-pin · 2026-06-26] `_LAST_KNOWN_PFIX` es single-valued: pinear el valor
    # específico de ESTE cierre lo vuelve stale apenas un P-fix posterior bumpea el
    # marker (pasó con P2-DISH-COHERENCE el 2026-06-25 y P1-ANALYZE-NO-CHARGE-ON-FALLBACK
    # el 2026-06-26). El contrato DURABLE del bump vive en
    # test_p3_1_last_known_pfix_freshness (formato + floor de fecha) y
    # test_p2_hist_audit_14_marker_test_link (cross-link slug↔test). Aquí solo
    # verificamos que el marker existe y está bien formado (`Pn-... · YYYY-MM-DD`).
    assert re.search(r'_LAST_KNOWN_PFIX\s*=\s*"P\d+-[A-Z0-9-]+ · \d{4}-\d{2}-\d{2}"', APP), \
        "_LAST_KNOWN_PFIX debe existir con formato `Pn-... · YYYY-MM-DD`"


# ── Funcional (guardado por import de graph_orchestrator) ─────────────────────
try:
    import graph_orchestrator as _GO
    _GO_ERR = None
except Exception as _e:  # pragma: no cover
    _GO = None
    _GO_ERR = _e

requires_go = pytest.mark.skipif(
    _GO is None, reason=f"graph_orchestrator no importable (¿falta langgraph?): {_GO_ERR}"
)


@requires_go
def test_mercury_scanner_fires_on_high_mercury_fish_when_pregnant():
    meal = {"name": "Tiburón a la plancha", "ingredients": ["Filete de tiburón", "Arroz", "Ensalada"]}
    viol = _GO._scan_mercury_pregnancy_violations(meal, {"medicalConditions": ["embarazo"]})
    assert viol and any("mercurio" in v.lower() for v in viol), "tiburón en embarazo debe violar"


@requires_go
def test_mercury_scanner_excludes_tuna_and_non_pregnant():
    # atún excluido a propósito (FDA Best/Good Choice en moderación)
    assert _GO._scan_mercury_pregnancy_violations(
        {"name": "Ensalada de atún", "ingredients": ["Atún en lata"]}, {"medicalConditions": ["embarazo"]}
    ) == []
    # no-embarazo → no dispara aunque haya tiburón
    assert _GO._scan_mercury_pregnancy_violations(
        {"name": "Tiburón guisado", "ingredients": ["Tiburón"]}, {"medicalConditions": []}
    ) == []


@requires_go
def test_clinical_backstop_surfaces_mercury_via_form_data():
    meal = {"name": "Pez espada", "ingredients": ["Filete de pez espada", "Vegetales"]}
    out = _GO.clinical_backstop_for_meal(meal, allergies=[], diet_type="balanced",
                                         form_data={"medicalConditions": ["embarazada"]})
    assert any("mercurio" in v.lower() for v in out), "el backstop debe surface el mercurio en embarazo"
    # sin form_data NO escanea mercurio (compat hacia atrás)
    assert _GO.clinical_backstop_for_meal(meal, allergies=[], diet_type="balanced") == []


@requires_go
def test_renal_trim_noop_when_not_renal_or_no_ceiling():
    meal = {"name": "x", "protein": 99, "carbs": 10, "fats": 5, "ingredients": ["Pechuga de pollo"]}
    assert _GO.renal_protein_trim_for_update([meal], 50.0, renal_capped=False) is False
    assert _GO.renal_protein_trim_for_update([meal], 0, renal_capped=True) is False
