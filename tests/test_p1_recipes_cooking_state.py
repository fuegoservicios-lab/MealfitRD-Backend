"""[P1-RECIPES-COOKING-STATE · 2026-08-21] Recetas hereda el vacío honesto del
Dashboard.

Reportado con captura: con el bloque generándose AHORA, /dashboard/recipes decía
«Aún no hay recetas para este día · Cuando tu plan esté completo…» — estático y
sin la animación de cocinado que el Dashboard ya tiene desde
P1-DASH-GENERATING-HONESTY.

Este test ancla la PARIDAD entre las dos escaleras (Dashboard.jsx ↔ Recipes.jsx):
mismos contadores, misma precedencia (pausado > cocinando > programado), mismos
títulos de catálogo (cero claves nuevas: Recetas REUTILIZA las del Dashboard, ya
traducidas a los 4 idiomas). Si el Dashboard cambia su escalera, este test exige
mover la de Recetas a la vez — es la alternativa deliberada a extraer un helper:
los anclas de test_p1_dash_generating_honesty.py clavan los literales DENTRO de
Dashboard.jsx, así que extraer rompería ese contrato.

La conducta (render de cada estado + el gate del fetch) vive en el companion
vitest Recipes.p1_recipes_cooking_state.test.jsx.

Tooltip-anchor: P1-RECIPES-COOKING-STATE
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RECIPES = _REPO_ROOT / "frontend" / "src" / "pages" / "Recipes.jsx"
_DASHBOARD = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"
_VITEST_COMPANION = (
    _REPO_ROOT / "frontend" / "src" / "__tests__" / "Recipes.p1_recipes_cooking_state.test.jsx"
)

_TITULOS = [
    "Tus próximos días están en pausa",
    "Estamos cocinando estos días",
    "Estos días aún no toca prepararlos",
]


def test_recipes_reads_the_three_counters():
    src = _RECIPES.read_text(encoding="utf-8")
    for counter in ("pending_user_action_count", "running_now_count", "scheduled_count"):
        assert counter in src, (
            f"P1-RECIPES-COOKING-STATE: Recipes.jsx dejó de leer `{counter}` — "
            "el vacío vuelve a ser estático mientras el servidor cocina."
        )


def test_precedence_matches_dashboard():
    """Pausado gana a cocinando; cocinando gana a programado — el MISMO orden que
    el Dashboard. Invertirlo diría «cocinando, no hagas nada» a un usuario cuya
    cola espera SU acción."""
    src = _RECIPES.read_text(encoding="utf-8")
    i_paused = src.find("_recPaused")
    i_running = src.find("_recCorriendoAhora")
    i_sched = src.find("if (_recProgramados)")
    assert -1 not in (i_paused, i_running, i_sched), (
        "P1-RECIPES-COOKING-STATE: desapareció alguna rama de la escalera de Recetas."
    )
    assert i_paused < i_running < i_sched, (
        "P1-RECIPES-COOKING-STATE: la precedencia de Recetas dejó de espejar la del "
        "Dashboard (pausado > cocinando > programado)."
    )


def test_titles_are_shared_with_dashboard():
    """Cero claves nuevas: las tres frases DEBEN existir en ambos ficheros. Si el
    Dashboard renombra una, este assert obliga a mover Recetas en el mismo paso
    (y los catálogos ya cubren la clave — P1-I18N-DASHBOARD)."""
    rec = _RECIPES.read_text(encoding="utf-8")
    dash = _DASHBOARD.read_text(encoding="utf-8")
    for titulo in _TITULOS:
        assert titulo in rec, f"P1-RECIPES-COOKING-STATE: Recetas perdió «{titulo}»"
        assert titulo in dash, (
            f"P1-RECIPES-COOKING-STATE: el Dashboard ya no usa «{titulo}» — si lo "
            "renombraste, renombra la copia de Recetas en el mismo commit."
        )


def test_cooking_state_is_live_animated():
    """Lo que pidió el dueño, literalmente: LA ANIMACIÓN. El EmptyState de
    cocinando debe llevar la prop `live` (icono girando, gated por
    prefers-reduced-motion en index.css)."""
    src = _RECIPES.read_text(encoding="utf-8")
    i_running = src.find("if (_recCorriendoAhora)")
    ventana = src[i_running : i_running + 400]
    assert "live" in ventana, (
        "P1-RECIPES-COOKING-STATE: el estado cocinando de Recetas perdió `live` — "
        "vuelve el vacío quieto que se lee como congelado."
    )


def test_fetch_gated_by_active_status_including_complete_partial():
    """`complete_partial` es la población DOMINANTE de producción (20 de 24 planes
    medidos 2026-08-04): dejarla fuera del gate haría el vacío ciego justo donde
    más se ve."""
    src = _RECIPES.read_text(encoding="utf-8")
    i_gate = src.find("_genStatusForChunks === 'partial'")
    assert i_gate != -1, "P1-RECIPES-COOKING-STATE: desapareció el gate de estado activo del fetch."
    ventana = src[i_gate : i_gate + 500]
    assert "'complete_partial'" in ventana
    assert "getPlanChunkStatus" in src


def test_vitest_companion_exists():
    assert _VITEST_COMPANION.exists(), (
        "P1-RECIPES-COOKING-STATE: falta el companion vitest (5 casos: cocinando "
        "live, precedencia de pausado, programado, estático, y harnesses hermanos "
        "sin llamada)."
    )
    assert "P1-RECIPES-COOKING-STATE" in _VITEST_COMPANION.read_text(encoding="utf-8")
