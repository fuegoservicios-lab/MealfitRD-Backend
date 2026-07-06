"""[P2-NEVERA-COMPLETION-REMOVED · 2026-07-06] Panel "Para completar tu Nevera" eliminado.

Decisión del owner (2026-07-06): el panel de faltantes era redundante — la lista
de compras YA es "lo que te falta comprar" — y ocupaba demasiado espacio del hero
(30+ chips). Se eliminó el render + estado del Dashboard; el backend
(compute_pantry_completion_delta + campo `pantry_completion_list` del recalc,
knob OFF por default) queda INTACTO por si se revisita como tooltip/contador
compacto — ver test_p1_renewal_pantry_completion.py.

Si alguien re-añade el panel, este test falla: la decisión se revierte con
consenso del owner, no por accidente de merge.
"""
import os

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DASH = os.path.join(_BACKEND, "..", "frontend", "src", "pages", "Dashboard.jsx")


def _src() -> str:
    with open(_DASH, encoding="utf-8") as f:
        return f.read()


def test_panel_render_removed():
    src = _src()
    assert "Para completar tu Nevera\n" not in src.replace("panel \"Para completar tu Nevera\"", ""), (
        "el panel 'Para completar tu Nevera' fue eliminado por decisión del owner "
        "(redundante con la lista de compras + ocupaba el hero) — no re-añadir sin consenso."
    )
    assert "pantryCompletionList.slice" not in src, "render de chips del panel eliminado"
    assert "setPantryCompletionList(" not in src, "estado del panel eliminado"


def test_tombstone_documents_decision():
    src = _src()
    assert "P2-NEVERA-COMPLETION-REMOVED" in src, (
        "la lápida debe quedar en Dashboard.jsx para que el próximo lector sepa que "
        "fue decisión de producto, no código perdido."
    )


def test_backend_surface_intact():
    """El backend NO se toca: knob + compute + campo del recalc siguen (revisitable)."""
    sc = open(os.path.join(_BACKEND, "shopping_calculator.py"), encoding="utf-8").read()
    assert "def compute_pantry_completion_delta" in sc
    plans = open(os.path.join(_BACKEND, "routers", "plans.py"), encoding="utf-8").read()
    assert '"pantry_completion_list"' in plans
