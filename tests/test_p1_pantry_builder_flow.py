"""[P1-PANTRY-BUILDER-FLOW · 2026-07-11] Modo "Desde mi Nevera" v2 — manual-first.

Feedback del owner (2026-07-11): "no funciona como esperaba, ya que la IA crea el plan
y en teoría esto debería ser más manual, el usuario debe tocar la nevera antes de crear
el plan sea de 30, 15 o 7 días". El submit del wizard ya NO genera directo en modo
pantry: DESVÍA a /pantry en "modo constructor" — banner con medidor de factibilidad en
vivo (re-consulta /api/plans/pantry-feasibility con debounce al cambiar el inventario)
y CTA "Crear mi plan con esta Nevera" que limpia el flag y navega a /plan (ahí el
useEffect de Plan.jsx dispara el SSE con el form completo persistido en el context).

Contrato (parser-based sobre el frontend):
1. Flow: la rama pantry setea `mealfit_pantry_plan_flow` en sessionStorage y navega a
   /pantry con return temprano — NUNCA navega a /plan dentro de esa rama.
2. El flag es session-scoped (muere con la pestaña). localStorage lo haría eterno: el
   banner re-aparecería semanas después en visitas normales a la Nevera.
3. Pantry: consume el flag, CTA deshabilitado con Nevera vacía (el modo sin alimentos
   es indistinguible del plan libre — queja original del owner), CTA limpia el flag,
   y escape hatch honesto ("Prefiero generar libre" → planSource='scratch').

tooltip-anchor: P1-PANTRY-BUILDER-FLOW
"""
from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend" / "src"
_FLOW = _FRONT / "components" / "assessment" / "InteractiveAssessmentFlow.jsx"
_PANTRY = _FRONT / "pages" / "Pantry.jsx"

_FLOW_SRC = _FLOW.read_text(encoding="utf-8")
_PANTRY_SRC = _PANTRY.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. El submit del wizard desvía a /pantry (no genera directo)
# ---------------------------------------------------------------------------

def test_flow_detours_to_pantry_before_generating():
    i_branch = _FLOW_SRC.find("formData.planSource === 'pantry'")
    assert i_branch > 0, "la rama pantry del submit desapareció"
    i_flag = _FLOW_SRC.find("sessionStorage.setItem('mealfit_pantry_plan_flow'", i_branch)
    # [P1-PANTRY-ROUTE-ALIAS] ruta canónica: '/pantry' a secas era el catch-all 404.
    i_nav_pantry = _FLOW_SRC.find("navigate('/dashboard/pantry')", i_branch)
    i_nav_plan = _FLOW_SRC.find("navigate('/plan')", i_branch)
    assert i_flag > 0, "la rama pantry debe setear el flag del modo constructor"
    assert i_nav_pantry > 0, "la rama pantry debe navegar a /pantry"
    assert i_flag < i_nav_pantry, "flag ANTES de navegar (si navega primero, Pantry no ve el modo)"
    assert i_nav_pantry < i_nav_plan, (
        "navigate('/plan') debe quedar FUERA (después) de la rama pantry — "
        "el modo Nevera ya no genera directo desde el submit"
    )


def test_flow_pantry_branch_returns_early():
    i_branch = _FLOW_SRC.find("formData.planSource === 'pantry'")
    i_nav_plan = _FLOW_SRC.find("navigate('/plan')", i_branch)
    branch_slice = _FLOW_SRC[i_branch:i_nav_plan]
    assert "return;" in branch_slice, (
        "la rama pantry debe hacer return temprano — sin él, el submit seguiría "
        "de largo y dispararía la generación libre además del desvío"
    )


def test_flag_is_session_scoped_never_localstorage():
    for src, name in ((_FLOW_SRC, "flow"), (_PANTRY_SRC, "Pantry")):
        assert "localStorage.setItem('mealfit_pantry_plan_flow'" not in src, (
            f"{name}: el flag en localStorage sería ETERNO — el banner constructor "
            "re-aparecería semanas después en visitas normales a la Nevera"
        )
    assert "sessionStorage.setItem('mealfit_pantry_plan_flow'" in _FLOW_SRC


# ---------------------------------------------------------------------------
# 2. Pantry: banner constructor + medidor + CTA
# ---------------------------------------------------------------------------

def test_pantry_consumes_flag_and_renders_builder():
    assert "sessionStorage.getItem('mealfit_pantry_plan_flow')" in _PANTRY_SRC
    assert "Crear mi plan con esta Nevera" in _PANTRY_SRC, "CTA del constructor desapareció"
    assert "sessionStorage.removeItem('mealfit_pantry_plan_flow')" in _PANTRY_SRC, (
        "el flag debe limpiarse (CTA/dismiss) — sin cleanup el banner es permanente"
    )
    assert "navigate('/plan')" in _PANTRY_SRC, "el CTA debe navegar a /plan (dispara el SSE)"


def test_pantry_cta_disabled_when_empty():
    assert "disabled={inventory.length === 0}" in _PANTRY_SRC, (
        "Nevera vacía → CTA deshabilitado (queja original del owner: el modo con "
        "Nevera vacía era indistinguible del plan libre)"
    )


def test_pantry_has_free_generation_escape_hatch():
    assert "updateData('planSource', 'scratch')" in _PANTRY_SRC, (
        "escape hatch 'Prefiero generar libre' debe re-marcar planSource='scratch' — "
        "sin él, el usuario generaría 'libre' con el backend aún en modo pantry"
    )


def test_pantry_feasibility_live_meter_is_debounced():
    i_ep = _PANTRY_SRC.find("'/api/plans/pantry-feasibility'")
    assert i_ep > 0, "el medidor en vivo debe consultar el pre-flight determinista"
    window = _PANTRY_SRC[max(0, i_ep - 1500): i_ep + 1500]
    assert "setTimeout" in window and "clearTimeout" in window, (
        "la re-consulta al cambiar inventario debe ir con debounce (RateLimiter "
        "server-side + el add/delete debe persistir antes de re-contar)"
    )


# ---------------------------------------------------------------------------
# 3. Markers anclados en el código fuente
# ---------------------------------------------------------------------------

def test_marker_anchored_in_source():
    assert _FLOW_SRC.count("P1-PANTRY-BUILDER-FLOW") >= 1
    assert _PANTRY_SRC.count("P1-PANTRY-BUILDER-FLOW") >= 2
