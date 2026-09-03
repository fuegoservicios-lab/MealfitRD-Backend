"""[P0-HIST-FIX-3 · 2026-05-09] Cross-link backend del fix UX al
chip "X de Y listos" del modal del Historial — split entre
`_activeTotal` (decrementado por shift_plan) y `_displayTotal`
(plan original, inmutable).

Bug original (reportado en producción 2026-05-09):
    Plan creado como 7 días con `legacy totalDays=7`. Tras 5 shifts
    (cron rolling), `total_days_requested` se decrementó a 6 y
    `plan_data.days` quedó con 2 entries (Sáb+Dom — Vie ya pasó y
    fue trimmeado). El frontend leía `total_days_requested` (6) para
    el chip → "2 de 6 listos". Usuario reportó "las semanas tienen
    7 días, no 6" — el chip contradecía el nombre del plan ("7 días").

Fix (frontend):
    Split en dos variables:
      - `_activeTotal`: lo que el backend ESPERA generar AHORA mismo
        (`total_days_requested`, decrementado por shift_plan). Usado
        para missing math (`_activeTotal - _planDaysLen`).
      - `_displayTotal`: el plan ORIGINAL (`legacy totalDays`),
        inmutable. Refleja el mental model del usuario. Usado para
        el chip "X de Y listos".
      - `_expiredDays = _displayTotal - _activeTotal`: días ya
        consumidos por shift_plan. Cuando > 0, subtitle agrega
        "(N día(s) anterior(es) ya pasó(aron))" para que la math
        cuadre visualmente (planDaysLen + missingDays + expired = displayTotal).

Cobertura backend (cross-link del marker — P2-HIST-AUDIT-14
requiere tests/test_p0_hist_fix_3*.py):
    1. Anchor del marker en History.jsx.
    2. Endpoint /history-list expone `total_days_requested` Y `totalDays`
       legacy en el payload — el frontend toma max() de ambos para el
       display.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

# [P1-I18N-HISTORY-FORENSE · 2026-08-21] El chip pasó por `t()`, así que la grafía
# `{_generatedTotal} de {_displayTotal} listos` ya no existe literalmente: ahora es
# `t('{hechos} de {total} listos', { hechos: _generatedTotal, total: _displayTotal })`.
#
# La conducta NO cambió —mismos números, mismo orden, mismo encuadre de progreso— así que
# estos guards se reanclan a la PROPIEDAD en vez de a la grafía. Es la misma lección que
# el chip de caducidad de la Nevera, el mismo día y reportada por otra sesión: si el
# guard se puede expresar por la propiedad, que no dependa de cómo se escriba.
#
# `_EXPR_CHIP` acota la ventana a la expresión del chip, para que «las dos variables
# aparecen en el fichero» no cuente como «aparecen juntas en el chip».
import re as _re

_EXPR_CHIP = _re.compile(
    r"t\(\s*'\{\w+\} de \{\w+\} listos'[^)]*\)", _re.S
)


def _expresion_del_chip(texto: str) -> str:
    m = _EXPR_CHIP.search(texto)
    assert m is not None, (
        "No encontré la expresión del chip de progreso. Tiene que seguir siendo UNA "
        "sola llamada `t('{…} de {…} listos', {…})`: partirla en «etiqueta» + valor "
        "fijaría el orden español, y hay idiomas donde el número va delante. "
        "[P1-I18N-HISTORY-FORENSE]"
    )
    return m.group(0)



_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_HISTORY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "History.jsx"


# ---------------------------------------------------------------------------
# 1. Anchor del marker en frontend
# ---------------------------------------------------------------------------
def test_marker_present_in_history_jsx():
    assert _HISTORY_JSX.exists()
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert "[P0-HIST-FIX-3" in text, (
        "Marker `P0-HIST-FIX-3` debe aparecer en History.jsx donde "
        "se splittea _activeTotal vs _displayTotal."
    )


def test_history_jsx_declares_split_variables():
    """El frontend debe declarar AMBAS variables (_activeTotal +
    _displayTotal) para el split. Sin esta separación, el chip
    cae al bug original (mostrar 6 cuando el plan es 7)."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    assert "_activeTotal" in text, (
        "Variable `_activeTotal` debe declararse para missing math."
    )
    assert "_displayTotal" in text, (
        "Variable `_displayTotal` debe declararse para el chip."
    )
    assert "_expiredDays" in text, (
        "Variable `_expiredDays` debe declararse para mention en subtitle."
    )


# ---------------------------------------------------------------------------
# 2. Backend: payload contract preservado
# ---------------------------------------------------------------------------
def test_history_list_exposes_total_days_requested():
    """El frontend lee `total_days_requested` para `_activeTotal`."""
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    assert "total_days_requested" in src, (
        "Endpoint /history-list debe seguir exponiendo "
        "`total_days_requested` (input de _activeTotal)."
    )


def test_history_list_does_not_strip_legacy_totalDays():
    """El payload debe permitir que el frontend lea `plan_data.totalDays`
    (legacy) si está presente. El endpoint no consume todo el plan_data
    como blob (proyección mínima por keys), pero NO debe excluir
    explícitamente `totalDays` — el frontend hace fallback al jsonb
    via supabase direct cuando abre el modal (P1-HIST-AUDIT-4 lazy
    load). Verificamos que el SELECT no oculte el campo."""
    from routers.plans import api_plans_history_list
    src = inspect.getsource(api_plans_history_list)
    # El endpoint extrae `totalDays` como input alternativo de
    # `total_days_requested` (línea ~5707).
    assert "totalDays" in src, (
        "Endpoint /history-list debe leer `totalDays` (legacy) como "
        "fallback del COALESCE de total_days_requested."
    )


# ---------------------------------------------------------------------------
# 3. Anti-pattern: chip viejo no debe regresar
# ---------------------------------------------------------------------------
def test_chip_uses_display_total_not_active_total():
    """El render del chip NO debe usar `_totalRequested` o
    `_activeTotal` directamente. Debe usar `_displayTotal` para
    reflejar el plan original que el usuario nombró.

    [stale-parser fix 2026-06-16] P0-HIST-FIX-4 refinó el numerador del
    chip de `_planDaysLen` a `_generatedTotal` (= `_planDaysLen +
    _expiredDays`). El denominador — la invariante real de este test —
    sigue siendo `_displayTotal` (plan original, no `_activeTotal`
    decrementado por shift_plan). El regex tolera ambos numeradores
    (`_planDaysLen` legacy o `_generatedTotal` actual). Ver History.jsx
    línea ~5301."""
    text = _HISTORY_JSX.read_text(encoding="utf-8")
    # Buscar el chip render: `{<numerador>} de {X} listos`. El numerador
    # puede ser `_planDaysLen` (legacy) o `_generatedTotal` (post-FIX-4).
    _expr = _expresion_del_chip(text)
    assert "_generatedTotal" in _expr or "_planDaysLen" in _expr, (
        "El numerador del chip ya no sale de `_generatedTotal`/`_planDaysLen`."
    )

    class _M:
        """El resto del test consulta `m.group(1)` esperando el DENOMINADOR."""

        @staticmethod
        def group(_n):
            mm = re.search(r"total:\s*(\w+)", _expr)
            return mm.group(1) if mm else ""

    m = _M
    var_used = m.group(1)
    assert var_used == "_displayTotal", (
        f"Chip usa `{{{var_used}}}` pero debe usar `_displayTotal` "
        f"para reflejar el plan original del usuario, no el activo "
        f"decrementado por shift_plan."
    )
