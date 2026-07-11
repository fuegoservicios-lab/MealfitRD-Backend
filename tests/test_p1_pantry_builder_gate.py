"""[P1-PANTRY-BUILDER-GATE · 2026-07-11] El guard de assessment-incompleto deja pasar
el modo constructor de la Nevera.

Bug vivo (screenshot del owner, 10:15): tras el desvío del wizard, ProtectedRoute veía
cuenta nueva SIN plan (`hasCompletedAssessment=false`) → rebote inmediato de
/dashboard/pantry a /assessment: el usuario nunca veía el banner constructor (solo el
toast). El modo constructor es el PRIMER flujo legítimo que lleva a un usuario sin plan
al dashboard — necesita excepción explícita, gateada por el flag sessionStorage
`mealfit_pantry_plan_flow` Y la ruta exacta de la Nevera (no abre el resto del
dashboard a cuentas sin onboarding).

Contrato:
1. ProtectedRoute lee el flag (sync, mismo patrón que `mealfit_plan_in_progress`) y
   exceptúa SOLO `/dashboard/pantry` con flag activo del rebote a /assessment.
2. El desvío del wizard NO aplica a guests (la Nevera requiere cuenta; un guest solo
   puede traer planSource='pantry' stale de una sesión autenticada previa).
3. La X "salir del modo" con usuario sin plan navega a /assessment (destino
   determinista — quedarse en la Nevera sin flag re-activaría el rebote del guard).

tooltip-anchor: P1-PANTRY-BUILDER-GATE
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend" / "src"

_GUARD_SRC = (_FRONT / "components" / "layout" / "ProtectedRoute.jsx").read_text(encoding="utf-8")
_FLOW_SRC = (_FRONT / "components" / "assessment" / "InteractiveAssessmentFlow.jsx").read_text(encoding="utf-8")
_PANTRY_SRC = (_FRONT / "pages" / "Pantry.jsx").read_text(encoding="utf-8")


def test_guard_reads_builder_flag():
    assert "sessionStorage.getItem('mealfit_pantry_plan_flow')" in _GUARD_SRC, (
        "ProtectedRoute debe leer el flag del modo constructor — sin él, el guard "
        "assessment-incompleto rebota el desvío del wizard a /assessment"
    )


def test_guard_exception_is_scoped_to_pantry_route():
    m = re.search(
        r"if \(!hasCompletedAssessment && !isOnAssessment && !isOnPlan && !isOnLanding && !isOnAccountSettings\s*"
        r"&& !\(isOnPantry && _pantryBuilderFlow\)\)",
        _GUARD_SRC,
    )
    assert m, (
        "la excepción del guard debe ser `!(isOnPantry && _pantryBuilderFlow)` — "
        "acotada a /dashboard/pantry CON flag (no abrir todo el dashboard a cuentas "
        "sin onboarding)"
    )
    assert "location.pathname === '/dashboard/pantry'" in _GUARD_SRC


def test_flow_detour_excludes_guests():
    assert "formData.planSource === 'pantry' && !isGuest" in _FLOW_SRC, (
        "el desvío del wizard debe excluir guests — la Nevera requiere cuenta y "
        "ProtectedRoute los rebotaría (planSource='pantry' stale es posible)"
    )


def test_builder_exit_has_deterministic_destination():
    i_exit = _PANTRY_SRC.find('aria-label="Salir del modo constructor"')
    assert i_exit > 0, "botón de salida del modo constructor desapareció"
    window = _PANTRY_SRC[i_exit:i_exit + 600]
    assert "if (!planData) navigate('/assessment')" in window, (
        "salir del modo sin plan debe llevar a /assessment — quedarse en la Nevera "
        "sin flag re-activa el rebote del guard en la próxima navegación"
    )


def test_marker_anchored_in_source():
    assert _GUARD_SRC.count("P1-PANTRY-BUILDER-GATE") >= 1
    assert _FLOW_SRC.count("P1-PANTRY-BUILDER-GATE") >= 1
    assert _PANTRY_SRC.count("P1-PANTRY-BUILDER-GATE") >= 1
