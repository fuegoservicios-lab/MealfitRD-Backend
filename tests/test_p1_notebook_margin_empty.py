"""[P1-NOTEBOOK-MARGIN-EMPTY · 2026-08-21] El margen rojo del cuaderno atravesaba
el estado pausado.

Reportado con captura: con el día en pausa («Tus próximos días están en pausa» +
banner ámbar), las dos rayas rojas del margen (`.meals-container::before`,
pseudo-elemento POSICIONADO que pinta por encima de los hijos en flujo) cruzaban
el banner y el EmptyState. Un margen sin renglones que anotar es ruido.

El fix es un gate por clase (`meals-container--sin-filas`) puesto por el JSX con
la MISMA condición que elige EmptyState vs timeline — la decoración calibrada
del dueño (DASH-NOTEBOOK-SOFTEN, P1-NOTEBOOK-MARGIN-LIGHT) queda intacta cuando
hay platos. El detalle de conducta vive en el companion vitest
`Dashboard.p1_notebook_margin_empty.test.js`; este ancla protege la existencia
del gate y su orden en la cascada desde el gate de CI backend.

Tooltip-anchor: P1-NOTEBOOK-MARGIN-EMPTY
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DASHBOARD = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"
_VITEST_COMPANION = (
    _REPO_ROOT
    / "frontend"
    / "src"
    / "__tests__"
    / "Dashboard.p1_notebook_margin_empty.test.js"
)


def test_gate_rule_exists_and_after_base():
    src = _DASHBOARD.read_text(encoding="utf-8")
    base = src.find(".meals-container::before {")
    gate = src.find(".meals-container--sin-filas::before {")
    assert base != -1, "desapareció la regla base del margen del cuaderno"
    assert gate != -1, (
        "P1-NOTEBOOK-MARGIN-EMPTY: desapareció el gate .meals-container--sin-filas::before "
        "— el margen rojo vuelve a atravesar el banner de pausa y el EmptyState."
    )
    assert gate > base, (
        "P1-NOTEBOOK-MARGIN-EMPTY: el gate quedó ANTES de la base; misma "
        "especificidad ⇒ la base lo pisa y el margen reaparece."
    )
    cuerpo = src[gate : src.find("}", gate)]
    assert "display: none" in cuerpo


def test_class_is_conditional_on_meals():
    src = _DASHBOARD.read_text(encoding="utf-8")
    assert "dayHasMealCards ? '' : ' meals-container--sin-filas'" in src, (
        "P1-NOTEBOOK-MARGIN-EMPTY: la clase dejó de ser condicional. Fija apaga "
        "el margen TAMBIÉN con platos (borra la decoración calibrada del dueño); "
        "ausente lo deja atravesando el estado pausado."
    )


def test_vitest_companion_exists():
    assert _VITEST_COMPANION.exists(), (
        "P1-NOTEBOOK-MARGIN-EMPTY: falta el companion vitest con los 4 asserts "
        "(regla apaga, cascada, clase condicional, predicado único de suplementos)."
    )
    assert "P1-NOTEBOOK-MARGIN-EMPTY" in _VITEST_COMPANION.read_text(encoding="utf-8")
