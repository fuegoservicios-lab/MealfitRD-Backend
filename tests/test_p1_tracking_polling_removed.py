"""[P1-TRACKING-POLLING-REMOVED · 2026-05-20] Anti-regresión del polling
fijo de 15s en `TrackingProgress.jsx` (card "Progreso en Tiempo Real").

Bug observado:
    Cada `setInterval(fetchConsumed, 15000)` disparaba un `setConsumed({...})`
    con objeto nuevo (referencia distinta) → React rerender del card aunque
    los 4 valores numéricos fueran iguales → flicker visual cada 15s sin
    cambio aparente. UX molesta reportada 2026-05-20.

Fix:
    Eliminar `setInterval` y depender de 2 triggers reactivos:
      1. `mealfit:refresh-inventory` custom event (chat agent vía P1-CHAT-UI-ACTION-INVENTORY).
      2. `visibilitychange` (cross-tab/cross-device sync cuando user vuelve al tab).

    Mismo patrón que `WaterTracker.jsx` (sin polling fijo, solo event-driven).
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TRACKING_PROGRESS_JSX = _REPO_ROOT / "frontend" / "src" / "components" / "dashboard" / "TrackingProgress.jsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_tracking_progress_has_no_setInterval():
    """[P1-TRACKING-POLLING-REMOVED] `TrackingProgress.jsx` NO debe contener
    `setInterval(fetchConsumed, ...)` ni similar — el polling fijo causa
    rerenders sin cambio de datos.

    Anchor: el bug era específicamente `setInterval(fetchConsumed, 15000)`.
    Si alguien lo reintroduce con cualquier interval, este test falla.
    """
    src = _read(_TRACKING_PROGRESS_JSX)
    # Anti-pattern: setInterval que dispare fetchConsumed (o cualquier
    # callback que refetchee consumed_meals).
    matches = re.findall(r"setInterval\s*\(\s*fetchConsumed", src)
    assert not matches, (
        f"`setInterval(fetchConsumed, ...)` encontrado en TrackingProgress.jsx — "
        f"el polling fijo causa rerenders sin cambio de datos y molesta UX. "
        f"Usar `mealfit:refresh-inventory` event + visibilitychange listener "
        f"en su lugar. Ver P1-TRACKING-POLLING-REMOVED · 2026-05-20."
    )
    # Defensa adicional: cualquier setInterval > 5s en este componente
    # probablemente regresa al polling fixed pattern.
    interval_matches = re.findall(r"setInterval\s*\([^,]+,\s*(\d+)", src)
    for interval_ms_str in interval_matches:
        interval_ms = int(interval_ms_str)
        assert interval_ms < 5000, (
            f"`setInterval` con interval {interval_ms}ms encontrado — el "
            f"componente no debe hacer polling. Solo intervals <5s son OK "
            f"(usualmente animation/UI ticks, no data fetches). Ver "
            f"P1-TRACKING-POLLING-REMOVED · 2026-05-20."
        )


def test_tracking_progress_listens_visibility_change():
    """[P1-TRACKING-POLLING-REMOVED] Reemplazo del polling: el componente
    DEBE escuchar `visibilitychange` para refetch cuando el user vuelve al
    tab. Cubre el caso "mutación cross-tab" que el listener del custom
    event NO cubre."""
    src = _read(_TRACKING_PROGRESS_JSX)
    assert "visibilitychange" in src, (
        "TrackingProgress.jsx NO escucha `visibilitychange`. Sin esto, "
        "mutaciones cross-tab (user logueó comida desde otro browser) NO "
        "se reflejan hasta el próximo mount del componente. Ver "
        "P1-TRACKING-POLLING-REMOVED · 2026-05-20."
    )
    # Sanity: addEventListener + removeEventListener pareados.
    add = "addEventListener('visibilitychange'" in src or \
          'addEventListener("visibilitychange"' in src
    remove = "removeEventListener('visibilitychange'" in src or \
             'removeEventListener("visibilitychange"' in src
    assert add and remove, (
        "addEventListener + removeEventListener para visibilitychange "
        "deben estar pareados (cleanup correcto, no memory leak)."
    )


def test_tracking_progress_preserves_inventory_event_listener():
    """[P1-TRACKING-POLLING-REMOVED] El listener `mealfit:refresh-inventory`
    (P1-CHAT-UI-ACTION-INVENTORY · 2026-05-20) debe seguir intacto — es el
    trigger principal del refetch (covers ~99% del uso: chat agent logueando
    comida)."""
    src = _read(_TRACKING_PROGRESS_JSX)
    assert "mealfit:refresh-inventory" in src, (
        "Listener `mealfit:refresh-inventory` desapareció — el chat agent "
        "no puede refrescar el card tras log_consumed_meal. Ver "
        "P1-CHAT-UI-ACTION-INVENTORY · 2026-05-20."
    )
