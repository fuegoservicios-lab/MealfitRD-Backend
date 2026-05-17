"""[P3-DURATION-DROPDOWN-CLOSE-IMMEDIATE · 2026-05-17] Fix UX: el dropdown
"Duración del Plan" (7/15/30 días) en Dashboard quedaba abierto hasta que
terminara el recalc async (~1-3s); ahora se cierra inmediatamente al
seleccionar la opción.

Síntoma reportado por usuario (screenshot adjunto):
> "Quiero que cuando seleccione 7, 15 o 30 se cierre el menú con la
>  selección que hice ya que actualmente cuando selecciono algo se queda
>  abierto el menú de opciones"

Causa raíz: en `Dashboard.jsx` el `onClick` de cada opción del dropdown
ejecutaba secuencialmente:
  1. updateData('groceryDuration', opt.value)        // sync
  2. safeUpdateHealthProfile({ groceryDuration })    // sync
  3. await fetchWithAuth('/api/plans/recalculate...') // 1-3s async
  4. setShowDespensaDropdown(false)                  // SOLO aquí cerraba

La llamada al setter de visibilidad estaba al final del callback, por
fuera del bloque `if (userProfile?.id && planData) {...}` pero después
de `await withRecalcLock(...)`. Resultado: dropdown permanecía abierto
durante todo el recalc — usuario veía la opción seleccionada (✓ verde)
pero el menú no se cerraba, percibido como "no responde".

Fix: mover `setShowDespensaDropdown(false)` ANTES del `if` que dispara
el recalc. El toast.loading('Calculando lista...') sigue dando feedback
visible del trabajo en background; cerrar el dropdown es independiente
de que el recalc tenga éxito.
"""
from __future__ import annotations

from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DASHBOARD = (
    _BACKEND_ROOT.parent / "frontend" / "src" / "pages" / "Dashboard.jsx"
).read_text(encoding="utf-8")


def test_marker_present():
    assert "P3-DURATION-DROPDOWN-CLOSE-IMMEDIATE" in _DASHBOARD, (
        "Marker P3-DURATION-DROPDOWN-CLOSE-IMMEDIATE ausente — un refactor "
        "podría revertir el fix y reintroducir el delay percibido."
    )


def test_close_happens_before_recalc_if_block():
    """`setShowDespensaDropdown(false)` DEBE ejecutarse antes del
    `if (userProfile?.id && planData)` que dispara el recalc async.

    Si está después de ese bloque (o dentro del try/finally), el menú se
    queda abierto durante el recalc (~1-3s) — comportamiento pre-fix.
    """
    # Anchor: el callback del map de opciones [weekly/biweekly/monthly]
    # del dropdown de duración. Marcado por safeUpdateHealthProfile +
    # groceryDuration en la misma vecindad.
    anchor = "safeUpdateHealthProfile({ groceryDuration: opt.value });"
    idx = _DASHBOARD.find(anchor)
    assert idx > 0, (
        "Anchor `safeUpdateHealthProfile({ groceryDuration: opt.value })` "
        "ausente — el dropdown de duración fue refactorizado y este test "
        "perdió su punto de referencia. Re-anclar manualmente."
    )
    # Inspecciona los 1200 chars siguientes a safeUpdateHealthProfile.
    # En esa ventana debe aparecer setShowDespensaDropdown(false) ANTES
    # del `if (userProfile?.id && planData)`. La ventana cubre el comment
    # block del fix + el setter + la apertura del if.
    window = _DASHBOARD[idx : idx + 1200]
    close_pos = window.find("setShowDespensaDropdown(false)")
    if_pos = window.find("if (userProfile?.id && planData)")
    assert close_pos > 0, (
        "`setShowDespensaDropdown(false)` no aparece en la ventana inmediata "
        "tras safeUpdateHealthProfile — fix removido o desplazado."
    )
    assert if_pos > 0, (
        "Anchor `if (userProfile?.id && planData)` ausente en la ventana — "
        "estructura del callback cambió, re-anclar."
    )
    assert close_pos < if_pos, (
        "REGRESIÓN: `setShowDespensaDropdown(false)` está DESPUÉS del bloque "
        "`if (userProfile?.id && planData)` — el dropdown queda abierto "
        "durante el recalc async (~1-3s). Mover el setter ANTES del if."
    )


def test_no_duplicate_close_after_finally():
    """Defensa: NO debe haber un segundo `setShowDespensaDropdown(false)`
    al final del callback (después del `} finally { ... }`). Si alguien
    añadió un fix duplicado, el segundo call es redundante pero también
    señala que el primero podría haber sido removido sin querer."""
    anchor = "safeUpdateHealthProfile({ groceryDuration: opt.value });"
    idx = _DASHBOARD.find(anchor)
    assert idx > 0
    # Ventana extendida (callback completo + onClick close)
    window = _DASHBOARD[idx : idx + 3000]
    count = window.count("setShowDespensaDropdown(false)")
    assert count == 1, (
        f"Se esperaba exactamente 1 llamada a setShowDespensaDropdown(false) "
        f"en el callback de la opción de duración; encontradas {count}. "
        f"Múltiples llamadas sugieren un fix duplicado — limpiar."
    )
