"""[P1-PANTRY-SCAN-QTY · 2026-07-11] Cantidad sanitizada por clase de unidad + spinner real.

Bug vivo (primer scan del owner): foto de UN paquete de avena → el modelo devolvió el
peso impreso del empaque → clamp plano min(99) mostró **"99 paquete"**. Además el
spinner del botón usaba la clase global `animate-spin`, no disponible en el chunk del
wizard → ícono congelado ("cuando carga no tiene movimiento").

Contrato:
1. `_sane_scan_qty(qty, unit)`: envases discretos (paquete/lata/botella/funda/taza)
   con qty>12 = peso mal leído → colapsa a 1 (mejor subestimar; el usuario ajusta
   con +). 'unidad' tolera hasta 30 (huevos). lb ∈ [0.25, 10]; g ∈ [10, 5000].
2. El prompt instruye NÚMERO DE ENVASES, nunca el peso impreso.
3. Spinner con keyframes autocontenidos (qpb-spin) — cero dependencia de clases
   globales que el chunk del wizard no carga.

tooltip-anchor: P1-PANTRY-SCAN-QTY
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))
_QPB = (_BACKEND.parent / "frontend" / "src" / "components" / "assessment"
        / "questions" / "QPantryBuilder.jsx")


def test_package_weight_misread_collapses_to_one():
    from routers.user_data import _sane_scan_qty
    # El caso vivo: avena "500" con unidad paquete → 1 (no 99, no 12).
    assert _sane_scan_qty(500, "paquete") == 1.0
    assert _sane_scan_qty(99, "lata") == 1.0
    assert _sane_scan_qty(3, "paquete") == 3.0
    assert _sane_scan_qty(0, "botella") == 1.0


def test_unidad_tolerates_egg_cartons():
    from routers.user_data import _sane_scan_qty
    assert _sane_scan_qty(24, "unidad") == 24.0
    assert _sane_scan_qty(500, "unidad") == 30.0


def test_weight_units_keep_wide_range():
    from routers.user_data import _sane_scan_qty
    assert _sane_scan_qty(500, "g") == 500.0
    assert _sane_scan_qty(3, "lb") == 3.0
    assert _sane_scan_qty(99, "lb") == 10.0
    assert _sane_scan_qty("basura", "paquete") == 1.0


def test_scan_pipeline_uses_sanitizer():
    src = (_BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8")
    assert '_sane_scan_qty(it.get("quantity"), _unit)' in src, (
        "el pipeline del scan debe pasar por el sanitizador — el clamp plano "
        "min(99) produjo '99 paquete' en producción"
    )
    assert "NUNCA el peso o los gramos impresos" in src, (
        "el prompt debe instruir número de envases, no peso del empaque"
    )


def test_spinner_is_self_contained():
    src = _QPB.read_text(encoding="utf-8")
    assert "@keyframes qpb-spin" in src and "qpb-spin 1s linear infinite" in src, (
        "spinner con keyframes propios — la clase global animate-spin no existe "
        "en el chunk del wizard (ícono congelado, feedback owner)"
    )
    assert 'className="animate-spin"' not in src


def test_marker_anchored_in_source():
    src = (_BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8")
    assert src.count("P1-PANTRY-SCAN-QTY") >= 1
    assert _QPB.read_text(encoding="utf-8").count("P1-PANTRY-SCAN-QTY") >= 1
