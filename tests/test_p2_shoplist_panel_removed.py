"""[P2-SHOPLIST-PANEL-REMOVED · 2026-07-06] Panel "Lista de compras por pasillo" eliminado.

Decisión del owner: el panel expandible (P2-AUDIT-V7-BATCH P2-8, 2026-07-04)
duplicaba el detalle itemizado que ya vive en el PDF descargable y engordaba el
hero. Se conserva SOLO una línea mínima con el total "esta ida al súper" — dato
único (el banner de presupuesto muestra el CICLO completo, otro número) que
permite ver el efecto de cambiar marcas/duración sin descargar el PDF.

Si alguien re-añade el panel, este test falla: la decisión se revierte con
consenso del owner, no por accidente de merge.
"""
import os

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FRONTEND = os.path.join(os.path.dirname(_BACKEND), "frontend")
_DASH = os.path.join(_FRONTEND, "src", "pages", "Dashboard.jsx")


def _dash() -> str:
    with open(_DASH, encoding="utf-8") as f:
        return f.read()


def test_panel_component_deleted():
    assert not os.path.exists(os.path.join(
        _FRONTEND, "src", "components", "dashboard", "ShoppingListPanel.jsx"
    )), "ShoppingListPanel.jsx fue eliminado (el detalle vive en el PDF) — git history lo preserva"


def test_dashboard_does_not_mount_panel():
    src = _dash()
    assert "<ShoppingListPanel" not in src
    assert "import ShoppingListPanel" not in src


def test_minimal_total_line_survives():
    # [2026-07-06 · iteración 3 de la colocación] la línea suelta "se veía
    # huérfana" (owner) → el total vive DENTRO del banner de presupuesto (la
    # caja del dinero: ciclo arriba, esta-ida abajo, mismos colores).
    src = _dash()
    assert "P2-SHOPLIST-PANEL-REMOVED" in src, "lápida documentando la decisión del owner"
    # Anchor en el banner de la UI ("costo real") — el mismo comment existe en
    # variante PDF más arriba en el archivo (línea ~2783) y confundía el index.
    banner = src.index("Estado honesto del presupuesto: costo real")
    win = src[banner:banner + 9000]
    assert "Esta ida al súper:" in win, (
        "el total 'esta ida' es dato ÚNICO (≠ total del ciclo) — vive en el banner de presupuesto"
    )
    assert "estimated_cost_rd" in win, "suma desde la lista agregada (misma fuente que tenía el panel)"
    assert "el detalle está en el PDF" in win, "la línea apunta al PDF como fuente del detalle"
