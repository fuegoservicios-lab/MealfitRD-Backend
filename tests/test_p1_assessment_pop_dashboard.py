"""[P1-ASSESSMENT-POP-DASHBOARD · 2026-08-20] El CTA del apex mandaba al
formulario a quien ya lo completó.

El apex (bioboros.com) es landing ESTÁTICA sin sesión: su "Crear mi plan"
aterriza SIEMPRE como carga de documento nueva en app.*/assessment. Con sesión
y assessment completado, `ProtectedRoute` solo guardaba la dirección "sin
assessment → /assessment" — la inversa no existía y el usuario veía el wizard
en vez de su dashboard.

El guard nuevo es POP-only (cold-start / redirect del apex / URL tecleada) con
exención `reload`, para NO tocar:
  - los `navigate('/assessment')` internos (renovar/regenerar desde
    Dashboard/History/Settings/DashboardTracking) — navegación PUSH;
  - el F5 a mitad de una renovación — POP con performance type 'reload'.

Este test parsea el source SIN comentarios (todas las aserciones son positivas
— "esto DEBE aparecer" — así que un stripping agresivo solo puede dar falso
ROJO, nunca falso verde; la dirección segura del filtro conservador).

Tooltip-anchor: P1-ASSESSMENT-POP-DASHBOARD
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PROTECTED_ROUTE = (
    _REPO_ROOT / "frontend" / "src" / "components" / "layout" / "ProtectedRoute.jsx"
)
_VITEST_COMPANION = (
    _REPO_ROOT
    / "frontend"
    / "src"
    / "__tests__"
    / "ProtectedRoute.assessment_pop_redirect.test.jsx"
)

_GUARD_RE = re.compile(r"isOnAssessment\s*&&\s*navigationType\s*===\s*'POP'")


def _code_without_comments() -> str:
    src = _PROTECTED_ROUTE.read_text(encoding="utf-8")
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"^\s*//.*$", "", src, flags=re.M)
    return src


def _guard_block(code: str) -> str:
    m = _GUARD_RE.search(code)
    assert m, (
        "P1-ASSESSMENT-POP-DASHBOARD: no existe el guard "
        "`isOnAssessment && navigationType === 'POP'` en ProtectedRoute.jsx "
        "(fuera de comentarios). Sin él, el CTA del apex vuelve a servir el "
        "formulario a usuarios con el assessment completado."
    )
    # Ventana acotada tras el gate: suficiente para el cuerpo del guard,
    # corta para no casar con el bloque landing-POP de más abajo.
    return code[m.start() : m.start() + 1200]


def test_guard_exists_and_redirects_plan_to_dashboard():
    block = _guard_block(_code_without_comments())
    plan_idx = block.find("planData")
    nav_idx = block.find('<Navigate to="/dashboard" replace />')
    assert plan_idx != -1 and nav_idx != -1 and plan_idx < nav_idx, (
        "P1-ASSESSMENT-POP-DASHBOARD: dentro del guard POP de /assessment debe "
        "existir la rama `if (planData)` que redirige con "
        '`<Navigate to="/dashboard" replace />`.'
    )


def test_guard_exempts_document_reload():
    block = _guard_block(_code_without_comments())
    assert re.search(r"\?\.type\s*!==\s*'reload'", block), (
        "P1-ASSESSMENT-POP-DASHBOARD: el guard perdió la exención `reload` "
        "(patrón LANDING-REFRESH-STAY). Sin ella, un F5 a mitad de una "
        "renovación del formulario expulsa al usuario al dashboard y pierde "
        "su progreso del wizard."
    )


def test_guard_covers_tracking_mode():
    block = _guard_block(_code_without_comments())
    assert "plan_mode" in block and "'tracking'" in block, (
        "P1-ASSESSMENT-POP-DASHBOARD: el guard perdió la rama de modo "
        "seguimiento. Un usuario tracking (perfil completo, sin plan A "
        "PROPÓSITO) debe ir a su dashboard-contador, no a re-contestar el "
        "formulario en cada llegada fría."
    )


def test_pending_recovery_guard_still_wins():
    """El guard de recovery (`mealfit_plan_in_progress` → /plan) debe evaluarse
    ANTES: un usuario con generación en curso que aterriza en /assessment debe
    ir a la pantalla de carga, no al dashboard (P1-GUEST-PLAN-RECOVERY)."""
    code = _code_without_comments()
    recovery = re.search(r"_hasPendingPlanRecovery\s*&&\s*!isOnPlan", code)
    guard = _GUARD_RE.search(code)
    assert recovery and guard and recovery.start() < guard.start(), (
        "P1-ASSESSMENT-POP-DASHBOARD: el guard POP de /assessment debe ir "
        "DESPUÉS del guard de recovery pendiente. Invertirlos manda al "
        "dashboard a quien tiene un plan generándose."
    )


def test_vitest_companion_anchors_both_bug_cases():
    assert _VITEST_COMPANION.exists(), (
        "P1-ASSESSMENT-POP-DASHBOARD: falta el test de conducta "
        "ProtectedRoute.assessment_pop_redirect.test.jsx (6 casos: plan→dashboard, "
        "tracking→dashboard, sin-plan se queda, cuenta nueva se queda, PUSH "
        "interno se queda, reload se queda)."
    )
    body = _VITEST_COMPANION.read_text(encoding="utf-8")
    assert "P1-ASSESSMENT-POP-DASHBOARD" in body
    assert "RENOVAR" in body, (
        "P1-ASSESSMENT-POP-DASHBOARD: el companion perdió el caso PUSH "
        "(renovar plan) — es el que protege los navigate('/assessment') "
        "internos de Dashboard/History/Settings."
    )
