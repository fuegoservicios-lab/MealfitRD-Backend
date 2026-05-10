"""[P0-HIST-FIX-4 · 2026-05-09] Cross-link backend del fix UX al
contador "X listos" — usar `_generatedTotal` (= _planDaysLen +
_expiredDays) en vez de `_planDaysLen` solo.

Bug reportado en producción 2026-05-09 (continuación de FIX-3):
    Plan de 7 días con primer chunk que generó 3 días (Vie+Sáb+Dom).
    Hoy es Sábado → Viernes ya expiró visualmente del array. El
    array `plan_data.days` tiene 2 entries (Sáb, Dom).

    Tras P0-HIST-FIX-3 el chip mostraba "2 de 7 listos" — usuario
    respondió "no debería decir 3 de 7?" porque el primer chunk
    generó 3 días reales (no 2). El día expirado FUE generado y
    debe contar en el numerador.

Fix:
    Nueva variable `_generatedTotal = _planDaysLen + _expiredDays`.
    Usada como numerador del chip "X de Y listos". Math que cuadra:
    `_generatedTotal + _missingDays = _displayTotal`
    (3 generados + 4 por generar = 7 totales ✓).

    Day numbering del missing range también cambia: ahora arranca
    en `_generatedTotal + 1` (no `_planDaysLen + 1`) — los días
    expirados ocupan los primeros slots, los faltantes vienen DESPUÉS.

Cobertura backend (cross-link del marker):
    1. Anchor del marker en History.jsx.
    2. Frontend declara la nueva variable `_generatedTotal`.
    3. Chip render usa _generatedTotal (no _planDaysLen).
    4. Range usa _generatedTotal + 1 como inicio.
"""
from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_HISTORY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "History.jsx"


def test_marker_present_in_history_jsx():
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert "[P0-HIST-FIX-4" in text, (
        "Marker `P0-HIST-FIX-4` debe aparecer en History.jsx donde "
        "se computa _generatedTotal."
    )


def test_history_jsx_declares_generated_total():
    """`_generatedTotal = _planDaysLen + _expiredDays` debe estar
    declarado para usar como numerador del chip."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert re.search(
        r"_generatedTotal\s*=\s*_planDaysLen\s*\+\s*_expiredDays",
        text,
    ), (
        "Variable `_generatedTotal = _planDaysLen + _expiredDays` "
        "debe declararse para reflejar lo que el primer chunk "
        "realmente generó (incluyendo días expirados)."
    )


def test_chip_uses_generated_total():
    """El chip debe usar `_generatedTotal` como numerador, no
    `_planDaysLen`. La diferencia es lo que el user vio en su
    screenshot: '2 de 7' vs '3 de 7'."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    m = re.search(
        r"\{(\w+)\}\s*de\s*\{_displayTotal\}\s*listos",
        text,
    )
    assert m is not None, (
        "Chip render `{X} de {_displayTotal} listos` no encontrado."
    )
    var_used = m.group(1)
    assert var_used == "_generatedTotal", (
        f"Chip usa `{{{var_used}}}` pero debe usar `_generatedTotal` "
        f"(= _planDaysLen + _expiredDays). Esto refleja los días que "
        f"el chunk realmente generó, incluyendo expirados visualmente."
    )


def test_missing_range_starts_from_generated_total_plus_one():
    """El missing range debe empezar después de los días generados
    (incluyendo expirados). Si _generatedTotal=3, el siguiente día
    pendiente es el 4, no el 3."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    # Singular: `el día ${_generatedTotal + 1}`.
    assert re.search(
        r"el\s+d[ií]a\s*\$\{_generatedTotal\s*\+\s*1\}",
        text,
    ), (
        "Missing range singular debe arrancar en `_generatedTotal + 1` "
        "(no en `_planDaysLen + 1`). El día expirado ocupa el slot "
        "ANTERIOR al primer día pendiente."
    )
    # Plural start: `del día ${_generatedTotal + 1}`.
    assert re.search(
        r"del\s+d[ií]a\s*\$\{_generatedTotal\s*\+\s*1\}",
        text,
    )
