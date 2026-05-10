"""[P0-HIST-FIX-2 · 2026-05-09] Cross-link backend del fix UX al
bloque "días pendientes" del modal del Historial.

Bug original (reportado en producción 2026-05-09 — screenshot del
usuario "no entiendo este bloque"):
    El bloque mostraba:
      - Título: "Días 3–6 pendientes"
      - Chip: "4/6"  ← AMBIGUO
      - Body: "Generación en proceso — vuelve a abrir el plan en
              unos minutos."

    El chip "4/6" se leía ambiguamente: ¿4 hechos de 6 totales, o
    4 faltan de 6 totales? Sin label, el user tenía que inferir.
    Body genérico: "unos minutos" — ¿2? ¿20? ¿qué significa?

Fix (frontend-only, copy + UX):
    1. Counter reframed como progreso explícito: "2 de 6 listos"
       en vez de "4/6". Frase que comunica done de total.
    2. Título dice cuántos faltan: "Faltan 4 días por generar"
       — el número va en texto, no en chip ambiguo.
    3. Subtitle line con rango natural: "Faltan del día 3 al día 6."
    4. Body por tono con copy concreto:
         in_flight → "Mealfit los está generando ahora en segundo
                      plano. Cierra el modal y vuelve a abrirlo
                      en 2 a 5 minutos para verlos listos."
         exhausted/failed → "Pulsa 'Reactivar este Plan' abajo para
                             reintentarlo."
         pending_user_action → "Mealfit está esperando que actualices
                                algo (tu nevera, tu registro de
                                comidas, o la fecha del plan)."
    5. Icon variable por tono (⚠️/⏸️/🔄/📅) en lugar de 📅 hardcoded.

Cobertura backend (este test cierra el cross-link del marker —
P2-HIST-AUDIT-14 requiere tests/test_p0_hist_fix_2*.py):
    1. Anchor del marker en History.jsx Y CSS.
    2. Endpoint /history-list sigue exponiendo `total_days_requested`
       (input del cómputo del frontend) — protege contra refactor
       que rompa el contrato del payload.
"""
from __future__ import annotations

import inspect
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_HISTORY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "History.jsx"
_HISTORY_CSS = _REPO_ROOT / "frontend" / "src" / "pages" / "History.module.css"


# ---------------------------------------------------------------------------
# 1. Anchor del marker en frontend
# ---------------------------------------------------------------------------
def test_marker_present_in_history_jsx():
    assert _HISTORY_JSX.exists()
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert "[P0-HIST-FIX-2" in text, (
        "Marker `P0-HIST-FIX-2` debe aparecer en History.jsx donde "
        "vive el copy clarificado del bloque missing-days."
    )


def test_marker_present_in_css():
    assert _HISTORY_CSS.exists()
    text = _HISTORY_CSS.read_text(encoding="utf-8")
    assert "[P0-HIST-FIX-2" in text, (
        "Marker `P0-HIST-FIX-2` debe aparecer en History.module.css "
        "donde se declara `.missingDaysSubtitle`."
    )


# ---------------------------------------------------------------------------
# 2. Anti-pattern: chip "4/6" no debe regresar
# ---------------------------------------------------------------------------
def test_chip_uses_progress_framing_not_raw_fraction():
    """El render del chip NO debe usar `{_missingDays}/{_totalRequested}`
    crudo (forma antigua, ambigua). Debe usar progreso explícito
    `{_planDaysLen} de {_totalRequested} listos` para evitar la
    ambigüedad reportada por el usuario."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    # Antiguo formato no debe estar.
    assert "{_missingDays}/{_totalRequested}" not in text, (
        "El chip antiguo `{_missingDays}/{_totalRequested}` regresa "
        "al bug de ambigüedad. Usa el progreso explícito "
        "`{_planDaysLen} de {_totalRequested} listos`."
    )
    # Nuevo formato debe estar.
    assert "{_planDaysLen} de {_totalRequested} listos" in text, (
        "El chip debe usar `{_planDaysLen} de {_totalRequested} "
        "listos` como progreso explícito."
    )


# ---------------------------------------------------------------------------
# 3. Backend: payload contract preservado
# ---------------------------------------------------------------------------
def test_history_list_exposes_total_days_requested():
    """El frontend lee `plan.total_days_requested` (top-level) Y
    `plan_data.total_days_requested` (legacy) para computar
    missing_days. Ambos deben seguir disponibles en el payload."""
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    assert "total_days_requested" in src, (
        "Endpoint /history-list debe exponer `total_days_requested` "
        "para que el cómputo de missing_days en el frontend funcione."
    )


def test_history_list_exposes_chunk_in_flight_count():
    """El frontend usa `chunk_in_flight_count` para inferir el tono
    `info` con copy 'Mealfit los está generando ahora en segundo
    plano'. Sin este counter, el tono cae al fallback genérico
    'aún no se han generado' que no comunica progreso."""
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    assert "chunk_in_flight_count" in src, (
        "Endpoint /history-list debe exponer `chunk_in_flight_count` "
        "para que el bloque missing-days infiera tono `info` con copy "
        "concreto en lugar del fallback genérico."
    )
