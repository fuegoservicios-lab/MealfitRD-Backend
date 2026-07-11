"""[P1-PANTRY-WIZARD-STEP · 2026-07-11] La Nevera se prepara DENTRO del wizard.

Feedback del owner (iteración 3 del modo "Desde mi Nevera", mismo día): "en vez de
enviarme al dashboard... mejor hacerlo directo en el formulario, como la pregunta 21,
cada vez que se active la opción de la nevera se debe activar la pregunta 21".
Supersede el desvío a /dashboard/pantry (P1-PANTRY-BUILDER-FLOW) y su excepción en
ProtectedRoute (P1-PANTRY-BUILDER-GATE) — ambos eliminados, el guard vuelve a ser
uniforme y el flag sessionStorage desaparece.

Contrato:
1. Flow: paso final condicional `...(isPantryMode ? [{...QPantryBuilder...}] : [])`
   al FINAL del array (no mueve índices de fieldToStepIndex); `isPantryMode` excluye
   guests; QSupplements en modo pantry avanza (`nextStep`) con label "Siguiente";
   el submit único `submitAndGenerate` corre validación idéntica en ambos caminos.
2. QPantryBuilder: mismos endpoints que la página Nevera (/api/inventory, /api/catalog,
   POST items con 409→increment, increment, DELETE), invalida el cache singleton tras
   mutaciones (la página Nevera no debe abrir stale), medidor debounced de
   /api/plans/pantry-feasibility y CTA deshabilitado con Nevera vacía.
3. El desvío murió completo: cero referencias a `mealfit_pantry_plan_flow` en el
   frontend; ProtectedRoute sin excepción de pantry.

tooltip-anchor: P1-PANTRY-WIZARD-STEP
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend" / "src"

_FLOW = _FRONT / "components" / "assessment" / "InteractiveAssessmentFlow.jsx"
_QPB = _FRONT / "components" / "assessment" / "questions" / "QPantryBuilder.jsx"
_QSUP = _FRONT / "components" / "assessment" / "questions" / "QSupplements.jsx"

_FLOW_SRC = _FLOW.read_text(encoding="utf-8")
_QPB_SRC = _QPB.read_text(encoding="utf-8")
_QSUP_SRC = _QSUP.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Wizard: paso condicional al final + submit único
# ---------------------------------------------------------------------------

def test_pantry_step_is_conditional_and_last():
    assert "const isPantryMode = formData.planSource === 'pantry' && !isGuest;" in _FLOW_SRC, (
        "isPantryMode debe excluir guests (la Nevera requiere cuenta)"
    )
    m = re.search(r"\.\.\.\(isPantryMode \? \[\{.*?QPantryBuilder.*?\}\] : \[\]\)\s*\];", _FLOW_SRC, re.DOTALL)
    assert m, (
        "el paso Nevera debe ser el ÚLTIMO del array steps vía spread condicional — "
        "insertarlo en medio movería los índices de fieldToStepIndex (buildFieldToStepIndex "
        "se memoiza una vez por mount)"
    )


def test_supplements_advances_in_pantry_mode():
    assert re.search(r"onFinish=\{isPantryMode \? \(\) => nextStep\(\) : submitAndGenerate\}", _FLOW_SRC), (
        "QSupplements en modo pantry debe avanzar al paso Nevera, no generar"
    )
    assert "finishLabel={isPantryMode ? 'Siguiente' : undefined}" in _FLOW_SRC
    assert "finishLabel" in _QSUP_SRC and '(finishLabel || "Finalizar y Generar")' in _QSUP_SRC


def test_single_submit_used_by_both_paths():
    assert "const submitAndGenerate = async () => {" in _FLOW_SRC
    assert "<QPantryBuilder onFinish={submitAndGenerate}" in _FLOW_SRC, (
        "el CTA del paso Nevera debe disparar el MISMO submit (validación P0-B3 + "
        "piso de presupuesto incluidos) — no un camino paralelo"
    )
    # La validación de campos corre dentro del submit único.
    i_fn = _FLOW_SRC.find("const submitAndGenerate = async () => {")
    i_val = _FLOW_SRC.find("findFirstIncompleteField(formData)", i_fn)
    i_nav = _FLOW_SRC.find("navigate('/plan')", i_fn)
    assert i_fn < i_val < i_nav, "submitAndGenerate debe validar antes de navegar a /plan"


# ---------------------------------------------------------------------------
# 2. QPantryBuilder: endpoints reales + cache + CTA
# ---------------------------------------------------------------------------

def test_qpantrybuilder_uses_real_inventory_endpoints():
    for anchor in (
        "'/api/inventory'",
        "'/api/catalog'",
        "'/api/inventory/items'",
        "'/api/inventory/increment'",
        "`/api/inventory/items/${item.id}`",
        "'/api/plans/pantry-feasibility'",
    ):
        assert anchor in _QPB_SRC, f"QPantryBuilder perdió el endpoint {anchor}"
    assert "err?.status === 409" in _QPB_SRC, (
        "duplicado UNIQUE (user+nombre+unidad) debe degradar a increment, no a error"
    )


def test_qpantrybuilder_invalidates_shared_cache():
    assert _QPB_SRC.count("invalidateInventoryCache()") >= 3, (
        "cada mutación (add/qty/delete) debe invalidar el cache singleton — sin esto "
        "la página Nevera abre con inventario stale tras el wizard"
    )


def test_qpantrybuilder_feasibility_is_debounced():
    i_ep = _QPB_SRC.find("'/api/plans/pantry-feasibility'")
    window = _QPB_SRC[max(0, i_ep - 1500): i_ep + 1500]
    assert "setTimeout" in window and "clearTimeout" in window


def test_qpantrybuilder_cta_disabled_when_empty():
    assert "disabled={isSubmitting || count === 0}" in _QPB_SRC, (
        "Nevera vacía → CTA deshabilitado (queja original del owner: el modo con "
        "Nevera vacía era indistinguible del plan libre)"
    )
    assert "Crear mi plan con esta Nevera" in _QPB_SRC


# ---------------------------------------------------------------------------
# 3. El desvío murió completo
# ---------------------------------------------------------------------------

def test_detour_flag_fully_removed():
    hits = []
    for f in _FRONT.rglob("*.jsx"):
        if "mealfit_pantry_plan_flow" in f.read_text(encoding="utf-8"):
            hits.append(f.name)
    assert not hits, (
        f"referencias huérfanas al flag del desvío superseded: {hits} — el paso "
        "Nevera vive DENTRO del wizard, el flag no debe renacer"
    )


def test_protected_route_guard_is_uniform_again():
    guard = (_FRONT / "components" / "layout" / "ProtectedRoute.jsx").read_text(encoding="utf-8")
    assert "_pantryBuilderFlow" not in guard and "sessionStorage.getItem('mealfit_pantry_plan_flow')" not in guard, (
        "la excepción de pantry en ProtectedRoute era del desvío superseded — "
        "con el paso en el wizard, ningún usuario sin plan entra al dashboard"
    )


def test_marker_anchored_in_source():
    assert _FLOW_SRC.count("P1-PANTRY-WIZARD-STEP") >= 2
    assert _QPB_SRC.count("P1-PANTRY-WIZARD-STEP") >= 1
    assert _QSUP_SRC.count("P1-PANTRY-WIZARD-STEP") >= 1
