"""[P1-CHAT-MACRO-CONTEXT · 2026-07-12] El agente conoce las macros del día + panel superpers fail-closed.

Pregunta del owner: "¿el agente sabe la cantidad de macros acumuladas en el
día?" — sabía SOLO kcal (DIARIO DE HOY + alertas de presupuesto); proteína/
carbos/grasas no viajaban, así que no podía razonar "33g de 125g de proteína"
como la card 'Progreso en Tiempo Real'. `_macro_totals_line` las suma del
diario y anexa las metas del plan.

Además (mismo turno): el panel de Súper Personalización era FAIL-OPEN — una
carga fallida (red caída, 5xx sin toast) dejaba el formulario VACÍO editable;
guardar en ese estado sobrescribía los datos reales con vacío.
tooltip-anchor: P1-CHAT-MACRO-CONTEXT
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BACKEND)

from agent import _macro_totals_line  # noqa: E402

with open(os.path.join(_BACKEND, "agent.py"), encoding="utf-8") as f:
    _AG = f.read()


def test_macro_line_sums_and_targets():
    consumed = [
        {"protein": 22, "carbs": 48, "healthy_fats": 24},
        {"protein": 11.4, "carbs": 44, "healthy_fats": 21},
    ]
    plan = {"macros": {"protein": 125, "carbs": 269, "fats": 58}}
    line = _macro_totals_line(consumed, plan)
    assert "33g proteína (meta 125g)" in line
    assert "92g carbohidratos (meta 269g)" in line
    assert "45g grasas (meta 58g)" in line


def test_macro_line_without_plan_targets():
    line = _macro_totals_line([{"protein": 10, "carbs": 20, "healthy_fats": 5}], None)
    assert "10g proteína" in line and "meta" not in line


def test_macro_line_fail_open_empty():
    assert _macro_totals_line(None, None) == ""  # shape rara → "" (no rompe el prompt)


def test_injected_in_both_prompt_paths():
    assert _AG.count("system_prompt += _macro_totals_line(consumed_today, current_plan)") >= 2, \
        "ambos paths del DIARIO DE HOY (non-stream y stream) llevan las macros"


def test_superpers_panel_fail_closed():
    with open(os.path.join(_ROOT, "frontend", "src", "components", "settings",
                           "SuperPersonalizationPanel.jsx"), encoding="utf-8") as f:
        panel = f.read()
    assert "loadFailed" in panel
    assert "if (!res.ok) throw" in panel, \
        "HTTP no-ok también es fallo de carga (antes era silencio + panel vacío)"
    assert "loading || loadFailed" in panel, \
        "guardar sin carga exitosa pisaría los datos reales con vacío"
    assert "Reintentar" in panel, "el fallo ofrece retry, no formulario vacío editable"
    assert "load(attempt + 1)" in panel, "1 reintento automático antes de rendirse"
