r"""[P3-DASH-WINDOW-FROM-TODAY · 2026-05-18] El Dashboard rolling window
arranca SIEMPRE en hoy y nunca retrocede a días pasados.

Bug previo (pre-fix):
    Ventana fija de 3 tabs con anti-colapso al final del chunk vivo. Esto
    significa que cuando el user está en el día 3 del chunk (miércoles en
    plan 7d), la ventana retrocedía a [Lun, Mar, Mié] para preservar 3 tabs
    en vez de mostrar [Mié] solo. El user reportó:
        "cuando se acabe el lunes debe ser eliminado y seguir con martes y
        miércoles, cuando se acabe el día martes debe seguir con miércoles
        nadamas y cuando se acabe el miércoles deben aparecer los 4 planes
        que deberian estar encolados para crearse en el siguiente chunks
        y aparecer como: jueves, viernes, sabado y domingo"

Causa raíz arquitectónica:
    El `_WINDOW_SIZE=3` con `visibleStartIndex=min(today, planDays-_WINDOW)`
    era anti-colapso: cuando el cron del chunk N+1 estaba atrasado, evitaba
    que el slice produjera ventana vacía. Pero efecto colateral: tabs de
    días pasados quedaban visibles (tachados via `isPastDay`). El user los
    quiere ESCONDIDOS, no tachados.

Fix:
    1. `_MAX_WINDOW = 4` (era 3) — permite mostrar [J,V,S,D] cuando chunk 2
       entra (chunks 7d = [3, 4]).
    2. `visibleStartIndex = min(todayPlanDayIndex, max(0, planDays-1))` —
       siempre arranca desde hoy, NUNCA retrocede.
    3. `visiblePlanDays = planDays.slice(visibleStartIndex, +_MAX_WINDOW)` —
       slice se achica al cruzar cada día.
    4. Edge case "chunk N+1 atrasado": clamp del start a planDays-1 evita
       slice vacío. Si chunk 2 no llegó, muestra el último día (con
       `isPastDay=true` → tachado, comunicando "esperando chunk").

Comportamiento end-to-end (plan 7d con chunks [3, 4]):
    - Lunes (día 1):    [L, M, Mi]      ventana 3 (chunk 2 aún no listo)
    - Martes (día 2):   [M, Mi]          ventana 2 (se achica)
    - Miércoles (día 3):[Mi]              ventana 1 (último día del chunk 1)
    - Jueves (día 4):   [J, V, S, D]      ventana 4 (chunk 2 ya en planDays)
    - Viernes (día 5):  [V, S, D]         ventana 3

Cobertura del test:
    1. Marker presente en Dashboard.jsx.
    2. `_MAX_WINDOW = 4` declarado (no `_WINDOW_SIZE = 3`).
    3. `visibleStartIndex` usa `Math.min(todayPlanDayIndex, ...)` — siempre
       arranca en hoy o el último día disponible.
    4. NO existe `Math.min(todayPlanDayIndex, planDays.length - _WINDOW_SIZE)`
       (el patrón anti-colapso fue removido).
    5. Slice usa `_MAX_WINDOW` como cap (no `_WINDOW_SIZE`).
    6. Skeleton placeholder usa `_MAX_WINDOW` como referencia del faltante.
    7. Marker P3-DASH-WINDOW-FROM-TODAY bumpeado en app.py.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DASHBOARD = (_REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx").read_text(
    encoding="utf-8"
)
_APP_PY = (_REPO_ROOT / "backend" / "app.py").read_text(encoding="utf-8")
# [P3-DASH-WINDOW-TEST · 2026-05-29] La lógica de la ventana rolling se extrajo de Dashboard.jsx a
# `utils/planWindow.js` (función pura `computeRollingWindow` + cap `MAX_WINDOW`, con su propio test JS
# `planWindow.test.js`). Este test parser-based se actualizó para verificar que el refactor PRESERVA la
# garantía (arranca en hoy, cap=4, ventana que se achica, sin tabs de días pasados) apuntando a la
# ubicación ACTUAL del código, en vez de la implementación inline vieja en Dashboard.jsx.
_PLANWINDOW = (_REPO_ROOT / "frontend" / "src" / "utils" / "planWindow.js").read_text(encoding="utf-8")


def test_marker_present_in_dashboard():
    assert "P3-DASH-WINDOW-FROM-TODAY" in _DASHBOARD, (
        "Marker P3-DASH-WINDOW-FROM-TODAY ausente en Dashboard.jsx. "
        "Un revert silente reintroduciría tabs de días pasados visibles."
    )


def test_max_window_replaces_window_size():
    """`MAX_WINDOW = 4` (en utils/planWindow.js tras el refactor 2026-05-29), aliasado en Dashboard.jsx
    como `_MAX_WINDOW = MAX_WINDOW`; el `_WINDOW_SIZE` antiguo NO."""
    max_window_match = re.search(r"const\s+MAX_WINDOW\s*=\s*(\d+)", _PLANWINDOW)
    assert max_window_match, (
        "`MAX_WINDOW` no encontrado en utils/planWindow.js. El refactor del windowing fue revertido?"
    )
    assert int(max_window_match.group(1)) == 4, (
        f"`MAX_WINDOW` debe ser 4 (para mostrar [J,V,S,D] cuando chunk 2 entra). "
        f"Actual: {max_window_match.group(1)}."
    )
    # Dashboard.jsx aliasa el cap importado de planWindow.js.
    assert re.search(r"const\s+_MAX_WINDOW\s*=\s*MAX_WINDOW", _DASHBOARD), (
        "Dashboard.jsx debe aliasar `_MAX_WINDOW = MAX_WINDOW` (cap importado de planWindow.js)."
    )
    # `_WINDOW_SIZE` antiguo NO debe existir en código ejecutable (declaración).
    assert not re.search(r"^\s*const\s+_WINDOW_SIZE\s*=", _DASHBOARD, re.MULTILINE), (
        "Declaración de `_WINDOW_SIZE` sigue activa. Debe estar reemplazada por `_MAX_WINDOW`/`MAX_WINDOW`."
    )


def test_visible_start_index_starts_from_today():
    """La garantía 'arranca SIEMPRE en hoy, nunca retrocede' vive ahora en `computeRollingWindow`
    (utils/planWindow.js: `visibleStartIndex = Math.min(todayPlanDayIndex, lastIndex)`); Dashboard.jsx
    la consume. El patrón anti-colapso viejo (`planDays.length - _WINDOW_SIZE`) fue removido."""
    # La lógica pura: nunca retrocede más allá del último día disponible (lastIndex).
    new_pattern = re.search(
        r"visibleStartIndex\s*=\s*Math\.min\(\s*todayPlanDayIndex,\s*lastIndex\s*\)",
        _PLANWINDOW,
    )
    assert new_pattern, (
        "computeRollingWindow debe calcular `visibleStartIndex = Math.min(todayPlanDayIndex, lastIndex)` "
        "(arranca en hoy / último día disponible, nunca retrocede)."
    )
    # Dashboard.jsx CONSUME el helper (no reimplementa el cálculo inline).
    assert re.search(r"computeRollingWindow\(", _DASHBOARD), (
        "Dashboard.jsx debe usar `computeRollingWindow` (utils/planWindow.js) para la ventana."
    )
    # Anti-pattern: el cálculo viejo anti-colapso NO debe existir en ningún lado.
    assert re.search(r"planDays\.length\s*-\s*_WINDOW_SIZE", _DASHBOARD) is None, (
        "Patrón anti-colapso `planDays.length - _WINDOW_SIZE` sigue presente. Debe estar removido."
    )


def test_visible_plan_days_uses_max_window():
    """`visiblePlanDays = planDays.slice(visibleStartIndex, visibleStartIndex + _MAX_WINDOW)`."""
    pattern = re.search(
        r"const\s+visiblePlanDays\s*=\s*planDays\.slice\(\s*visibleStartIndex,\s*visibleStartIndex\s*\+\s*_MAX_WINDOW\s*\)",
        _DASHBOARD,
    )
    assert pattern, (
        "Slice de `visiblePlanDays` debe usar `_MAX_WINDOW` como cap. "
        "Esperado: `planDays.slice(visibleStartIndex, visibleStartIndex + _MAX_WINDOW)`."
    )


def test_skeleton_uses_max_window():
    """El cálculo de `_missingSlots` para skeleton placeholders debe usar
    `_MAX_WINDOW`, no `_WINDOW_SIZE` (que ya no existe)."""
    pattern = re.search(
        r"_missingSlots\s*=\s*_MAX_WINDOW\s*-\s*visiblePlanDays\.length",
        _DASHBOARD,
    )
    assert pattern, (
        "Skeleton placeholders deben calcular `_missingSlots` contra "
        "`_MAX_WINDOW`. Si quedó como `_WINDOW_SIZE`, va a romper en runtime "
        "(ReferenceError: _WINDOW_SIZE is not defined)."
    )


def test_use_effect_active_day_uses_max_window():
    """La re-selección del día activo cuando queda fuera de la ventana usa la lógica pura
    `shouldReselectActiveDay` (utils/planWindow.js: `windowEnd = visibleStartIndex + maxWindow`); el
    effect del Dashboard recalcula la ventana via `computeRollingWindow`."""
    assert re.search(
        r"const\s+windowEnd\s*=\s*visibleStartIndex\s*\+\s*maxWindow",
        _PLANWINDOW,
    ), ("`shouldReselectActiveDay` debe calcular `windowEnd = visibleStartIndex + maxWindow` (cap del slice).")
    # El Dashboard aplica el recálculo de la ventana via el helper puro (no `_WINDOW_SIZE` inline).
    assert re.search(r"computeRollingWindow\(", _DASHBOARD)


def test_marker_bumped_in_app_py():
    """El marker P-fix global debe reflejar este fix (fecha ≥ 2026-05-18)."""
    m = re.search(
        r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"',
        _APP_PY,
    )
    assert m
    marker = m.group(1)
    date_match = re.search(r"·\s*(\d{4}-\d{2}-\d{2})$", marker)
    assert date_match, f"Marker sin fecha: {marker!r}"
    from datetime import date
    assert date.fromisoformat(date_match.group(1)) >= date(2026, 5, 18), (
        f"Marker stale: {marker!r}. Este fix es 2026-05-18."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Simulación matemática del slice (sin necesidad de runtime JS)
# ─────────────────────────────────────────────────────────────────────────────


def _compute_visible(today_idx: int, plan_len: int, max_window: int = 4) -> tuple[int, int]:
    """Replica del cálculo JS para validación matemática del slice.

    Returns (start_index, end_index_exclusive). slice = planDays[start:end].
    """
    start = min(today_idx, max(0, plan_len - 1))
    end = min(start + max_window, plan_len)
    return start, end


@pytest.mark.parametrize("today,plan_len,expected_range", [
    # Plan 3d (solo chunk 1):
    (0, 3, (0, 3)),   # Lunes: [L,M,Mi]
    (1, 3, (1, 3)),   # Martes: [M,Mi]
    (2, 3, (2, 3)),   # Miércoles: [Mi]
    # Plan 7d (chunks [3,4] ambos listos):
    (0, 7, (0, 4)),   # Lunes: [L,M,Mi,J]   ← cap a 4
    (3, 7, (3, 7)),   # Jueves: [J,V,S,D]
    (4, 7, (4, 7)),   # Viernes: [V,S,D]
    (6, 7, (6, 7)),   # Domingo: [D]
    # Plan 15d (chunks [3,4,4,4]):
    (0, 15, (0, 4)),  # Lunes: [L,M,Mi,J]   ← cap a 4
    (10, 15, (10, 14)),  # Día 11: 4 días futuros
    # Edge case: cron atrasado, hoy > planDays (no debería ocurrir si triggerShift funciona):
    (5, 3, (2, 3)),   # Today=día 6, plan solo 3 días → muestra día 3 (último)
])
def test_slice_math_matches_jsx(today, plan_len, expected_range):
    start, end = _compute_visible(today, plan_len)
    assert (start, end) == expected_range, (
        f"today={today}, plan_len={plan_len}: esperado {expected_range}, got ({start}, {end})"
    )
