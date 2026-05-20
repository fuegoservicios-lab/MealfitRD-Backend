"""[P3-LANDING-SKIP-POP-ONLY . 2026-05-20] Test anti-regresion del redirect
`/` -> `/dashboard` gated por `useNavigationType() === 'POP'`.

5a iteracion del dia sobre el mismo feature. Las anteriores:
    1. P3-MOBILE-LANDING-SKIP   — gated por matchMedia(768px). Mobile-only.
    2. P3-LANDING-SKIP-UNIVERSAL — sin gate viewport, bloqueaba "Inicio".
    3. P3-LANDING-SKIP-FIRST-VISIT v1 — flag module-level. Bug: timing.
    4. P3-LANDING-SKIP-FIRST-VISIT v2 — flag con guards. Bug: StrictMode.
    5. Este — useNavigationType, puro, sin side-effects en render.

Por que `useNavigationType` y NO un flag module-level:
    - StrictMode (re-habilitado en main.jsx, P2-STRICT-MODE-ENABLE) invoca
      el componente 2x por render en dev. Mutar un flag durante render
      lo deja inconsistente entre las 2 invocaciones — la 1a lo setea,
      la 2a ve el valor mutado y NO redirige. React commitea el resultado
      de la 2a invocacion -> el redirect queda silenciosamente descartado.
    - `useNavigationType()` es un hook declarativo de react-router. Devuelve
      'POP' / 'PUSH' / 'REPLACE'. Cold-start del PWA y refresh son 'POP'.
      Link clicks y navigate() programatico son 'PUSH' / 'REPLACE'.
    - Cero side-effects en render -> StrictMode-safe, idempotente.

Semantica del redirect:
    | Accion                      | navigationType | isOnLanding | Resultado     |
    |-----------------------------|----------------|-------------|---------------|
    | Cold-start PWA en `/`       | POP            | true        | Redirect      |
    | Refresh en `/`              | POP            | true        | Redirect      |
    | Browser back de /dashboard  | POP            | true        | Redirect      |
    | Click "Inicio" desde dash   | PUSH           | true        | Render landing|
    | Navigate replace -> `/`     | REPLACE        | true        | Render landing|
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PROTECTED_ROUTE_JSX = (
    _REPO_ROOT / "frontend" / "src" / "components" / "layout" / "ProtectedRoute.jsx"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip_js_comments(src: str) -> str:
    src = re.sub(r"/\*[\s\S]*?\*/", "", src)
    src = re.sub(r"//[^\n]*", "", src)
    return src


def test_marker_present_as_tooltip_anchor():
    """[P3-LANDING-SKIP-POP-ONLY] Marker presente en source."""
    src = _read(_PROTECTED_ROUTE_JSX)
    assert "P3-LANDING-SKIP-POP-ONLY" in src, (
        "Marker `P3-LANDING-SKIP-POP-ONLY` ausente en ProtectedRoute.jsx."
    )


def test_useNavigationType_imported_from_react_router():
    """[P3-LANDING-SKIP-POP-ONLY] El hook debe importarse de react-router-dom."""
    src = _read(_PROTECTED_ROUTE_JSX)
    import_match = re.search(
        r"import\s*\{[^}]*useNavigationType[^}]*\}\s*from\s*['\"]react-router-dom['\"]",
        src,
    )
    assert import_match, (
        "`useNavigationType` debe importarse de 'react-router-dom'. "
        "Si lo cambiaste a otro hook (e.g. useLocation custom), el contrato "
        "POP/PUSH/REPLACE se pierde."
    )


def test_useNavigationType_invoked_at_top_level():
    """[P3-LANDING-SKIP-POP-ONLY] El hook debe llamarse en el cuerpo del
    componente (no condicionalmente). React reglas-de-hooks."""
    src = _read(_PROTECTED_ROUTE_JSX)
    # Llamada al hook + asignacion a const
    invocation_match = re.search(
        r"const\s+navigationType\s*=\s*useNavigationType\s*\(\s*\)\s*;",
        src,
    )
    assert invocation_match, (
        "Esperaba `const navigationType = useNavigationType();` al tope del "
        "componente. Llamadas condicionales rompen las reglas de hooks."
    )


def test_redirect_gated_by_navigationType_pop():
    """[P3-LANDING-SKIP-POP-ONLY] El if del redirect debe checkear
    `navigationType === 'POP'`. Sin esto, Link clicks (PUSH) tambien
    serian redirigidos -> bloquearia "Inicio" del menu de cuenta."""
    src = _read(_PROTECTED_ROUTE_JSX)
    # Buscar el if completo del redirect / -> /dashboard
    if_match = re.search(
        r"if\s*\(\s*isOnLanding[^)]+\)\s*\{[\s\S]*?return\s+<Navigate\s+to=['\"]/dashboard['\"]",
        src,
    )
    assert if_match, "Bloque del redirect a /dashboard no encontrado."
    condition = if_match.group(0)
    assert "navigationType" in condition, (
        "Guard `navigationType` ausente. Sin el, todos los Link clicks a "
        "`/` serian redirigidos -> 'Inicio' del menu de cuenta queda bloqueado."
    )
    assert re.search(r"navigationType\s*===\s*['\"]POP['\"]", condition), (
        "Debe ser `navigationType === 'POP'`. 'POP' captura cold-start + "
        "refresh + browser back; 'PUSH' y 'REPLACE' son internal nav."
    )


def test_redirect_gated_by_landing_and_assessment():
    """[P3-LANDING-SKIP-POP-ONLY] Tres guards obligatorios: isOnLanding +
    hasCompletedAssessment + navigationType === 'POP'."""
    src = _read(_PROTECTED_ROUTE_JSX)
    if_match = re.search(
        r"if\s*\(\s*isOnLanding[^)]+\)\s*\{[\s\S]*?return\s+<Navigate\s+to=['\"]/dashboard['\"]",
        src,
    )
    condition = if_match.group(0)
    assert "isOnLanding" in condition
    assert "hasCompletedAssessment" in condition, (
        "Sin `hasCompletedAssessment`, users sin plan saltarian el wizard."
    )


def test_no_module_level_state_residual():
    """[P3-LANDING-SKIP-POP-ONLY] Anti-regresion contra iteraciones 3-4:
    NO debe haber `let _hasSpaSessionStarted` ni mutaciones module-level.
    Cualquier mutacion durante render rompe StrictMode."""
    src = _strip_js_comments(_read(_PROTECTED_ROUTE_JSX))
    assert "_hasSpaSessionStarted" not in src, (
        "Anti-regresion: `_hasSpaSessionStarted` presente en codigo. "
        "Bug de StrictMode: la mutacion durante render rompe la 2a "
        "invocacion del componente. Usa `useNavigationType()` en su lugar."
    )
    assert "isFirstSpaVisit" not in src, (
        "Anti-regresion: `isFirstSpaVisit` presente. Variable de la "
        "implementacion module-level previa. Reemplazada por navigationType."
    )


def test_no_viewport_gating_residual():
    """[P3-LANDING-SKIP-POP-ONLY] Anti-regresion contra iteracion 1
    (P3-MOBILE-LANDING-SKIP): no matchMedia / isMobile en codigo."""
    src = _strip_js_comments(_read(_PROTECTED_ROUTE_JSX))
    assert "matchMedia" not in src, "Anti-regresion: matchMedia presente."
    assert "isMobile" not in src, "Anti-regresion: isMobile presente."


def test_assessment_redirect_still_precedes():
    """[P3-LANDING-SKIP-POP-ONLY] Precedencia: /assessment antes que /dashboard."""
    src = _read(_PROTECTED_ROUTE_JSX)
    assessment_match = re.search(r"<Navigate\s+to=['\"]/assessment['\"]", src)
    dashboard_match = re.search(r"<Navigate\s+to=['\"]/dashboard['\"]", src)
    assert assessment_match and dashboard_match
    assert assessment_match.start() < dashboard_match.start(), (
        "Redirect a /assessment debe preceder al de /dashboard."
    )


def test_redirect_uses_replace():
    """[P3-LANDING-SKIP-POP-ONLY] `replace` obligatorio para evitar loops
    con browser back."""
    src = _read(_PROTECTED_ROUTE_JSX)
    assert re.search(
        r"<Navigate\s+to=['\"]/dashboard['\"]\s+replace\s*/>",
        src,
    ), "Navigate a /dashboard debe usar `replace`."


def test_protected_route_is_pure_no_side_effects():
    """[P3-LANDING-SKIP-POP-ONLY] StrictMode-safety: ningun side-effect
    durante render. No global mutations, no localStorage writes, no
    sessionStorage writes, no DOM manipulation."""
    src = _strip_js_comments(_read(_PROTECTED_ROUTE_JSX))
    # Buscar el cuerpo del componente.
    component_match = re.search(
        r"const\s+ProtectedRoute\s*=\s*\([^)]*\)\s*=>\s*\{([\s\S]*?)\n\};",
        src,
    )
    assert component_match, "Cuerpo del componente ProtectedRoute no encontrado."
    body = component_match.group(1)
    # Side-effects prohibidos durante render.
    forbidden = [
        ("localStorage.setItem", "localStorage write durante render"),
        ("sessionStorage.setItem", "sessionStorage write durante render"),
        ("document.cookie", "cookie write durante render"),
        ("window.history.push", "history mutation durante render"),
    ]
    for pattern, msg in forbidden:
        assert pattern not in body, (
            f"Anti-regresion StrictMode: {msg} ({pattern!r}). Mover a "
            f"useEffect o al handler del evento que lo causa."
        )
