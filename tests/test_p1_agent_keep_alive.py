"""[P1-AGENT-KEEP-ALIVE · 2026-05-20] Test anti-regresión del keep-alive
de `AgentPage` en `DashboardAnimatedLayout`.

Bug observado:
    Cada navegación Nevera/Plan/Recetas → Agente desmontaba AgentPage y
    re-montaba con state vacío. Pre-fixes #9/#10 (persist sessionId +
    cache messages en localStorage) mitigaban pero el flash visible del
    re-mount seguía. Reportado 2026-05-20: "lo siento igual o peor".

Fix:
    `DashboardAnimatedLayout` renderiza AgentPage RESIDENTE en el árbol
    (lazy-mounted al primer visit), con `display: none` cuando NO es la
    ruta activa. Cero re-mount → cero flash → cero refetch.

    `<Route path="/dashboard/agent" element={<></>}>` es un trampolin
    vacío para que React Router matchee el path (sin caer al wildcard
    Navigate("/")).
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_APP_JSX = _REPO_ROOT / "frontend" / "src" / "App.jsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_dashboard_layout_renders_agentpage_residente():
    """[P1-AGENT-KEEP-ALIVE] `DashboardAnimatedLayout` debe renderizar
    `<AgentPage />` directamente (no via Outlet) para que NO se desmonte
    al cambiar de ruta dentro de /dashboard/*."""
    src = _read(_APP_JSX)
    layout_match = re.search(
        r"const DashboardAnimatedLayout\s*=.*?\}\s*;",
        src,
        re.DOTALL,
    )
    assert layout_match, "DashboardAnimatedLayout no encontrado en App.jsx"
    body = layout_match.group(0)
    # Anchor: <AgentPage /> renderizado dentro del layout.
    assert re.search(r"<AgentPage\s*/?>", body), (
        "`<AgentPage />` NO se renderiza dentro de DashboardAnimatedLayout. "
        "Sin esto, el componente NO es residente. Ver P1-AGENT-KEEP-ALIVE."
    )
    # Anchor: display dinámico basado en isAgent.
    assert re.search(r"display:\s*isAgent\s*\?\s*['\"]block['\"]", body), (
        "`display: isAgent ? 'block' : 'none'` ausente — el componente "
        "siempre estaría visible. Ver P1-AGENT-KEEP-ALIVE."
    )


def test_dashboard_layout_uses_visited_ref_lazy():
    """[P1-AGENT-KEEP-ALIVE · drift-fix 2026-06-15] Lazy-mount: AgentPage solo se monta al primer visit
    (evita pagar ~300KB de chunk si el user nunca entra al chat). El guard se refactorizó de un useRef
    (`hasVisitedAgentRef`) a un useState con set en render-phase (`hasVisitedAgent` + `if (isAgent &&
    !hasVisitedAgent) setHasVisitedAgent(true)`), pero la GARANTÍA es idéntica."""
    src = _read(_APP_JSX)
    assert "hasVisitedAgent" in src, (
        "`hasVisitedAgent` ausente — sin lazy guard, AgentPage se monta siempre incluso si el user "
        "nunca abre /dashboard/agent. Ver P1-AGENT-KEEP-ALIVE."
    )
    # Sanity: el estado gatea el render condicional del AgentPage residente.
    assert re.search(
        r"\{\s*hasVisitedAgent\s*&&",
        src,
    ), "El estado `hasVisitedAgent` no se usa como guard del render condicional."


def test_lazy_mount_state_hook_imported():
    """[P1-AGENT-KEEP-ALIVE · drift-fix 2026-06-15] El lazy-mount se refactorizó de useRef a useState
    (`hasVisitedAgent` + `setHasVisitedAgent`). Verifica que el hook esté importado de react (antes
    `useRef`, ahora `useState`) y que el setter del guard exista — sin esto, crashea al runtime."""
    src = _read(_APP_JSX)
    assert re.search(
        r"import\s*\{[^}]*useState[^}]*\}\s*from\s*['\"]react['\"]",
        src,
    ), "`useState` no importado de 'react' — el guard del lazy-mount crashea al runtime."
    assert "setHasVisitedAgent" in src, "el setter `setHasVisitedAgent` del lazy-mount guard ausente."


def test_agent_route_is_trampolin():
    """[P1-AGENT-KEEP-ALIVE] La Route `/dashboard/agent` debe ser un
    trampolin con `element={<></>}` (Fragment vacío), NO `<AgentPage />`.
    Si renderiza AgentPage acá Y arriba en el layout, hay 2 instancias
    montadas simultáneamente."""
    src = _read(_APP_JSX)
    # Buscar la Route path="/dashboard/agent" — su element NO debe ser <AgentPage />.
    route_match = re.search(
        r'<Route\s+path="/dashboard/agent"\s+element=\{([^}]+)\}\s*/>',
        src,
    )
    assert route_match, (
        "Route `/dashboard/agent` no encontrada. Sin esta route, el "
        "wildcard `*` matchearía y redirigiría a `/`. Ver P1-AGENT-KEEP-ALIVE."
    )
    element_arg = route_match.group(1).strip()
    # Aceptamos: <></> | <Fragment /> | <Fragment></Fragment>.
    is_empty_fragment = (
        element_arg in ("<></>", "<Fragment />", "<Fragment/>")
        or element_arg.startswith("<>")
    )
    assert is_empty_fragment, (
        f"Route `/dashboard/agent` renderiza `{element_arg}` — debe ser "
        f"Fragment vacío `<></>`. AgentPage ya está renderizado residente "
        f"en DashboardAnimatedLayout. Ver P1-AGENT-KEEP-ALIVE · 2026-05-20."
    )
