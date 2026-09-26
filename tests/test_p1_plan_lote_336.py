# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-336 · 2026-09-26] La sonda del teclado mide pausas y traza quién mueve la página (solo frontend).

El video del dueño (25-sep, 60 fps) enseña, al abrir el teclado, 2-3 fotogramas con la página desplazada y una franja
negra arriba, y al cerrar la caja quieta ~100 ms y luego un salto. Antes de desactivar nada en el nativo de iOS hay que
saber QUIÉN mueve la página: la sonda traza por fotograma S (paneo de iOS), sy (scroll), top y caja, y anota pausas.
Tests de verdad: `frontend/src/__tests__/lote336.sonda_pausas.test.js`. Tooltip-anchor: P1-PLAN-LOTE-336
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 336


def test_la_sonda_mide_pausas_y_traza_por_fotograma():
    if not _FRONT.exists():
        return
    src = (_FRONT / "src" / "utils" / "keyboardProbe.js").read_text(encoding="utf-8")
    assert "export function detectorDePausas" in src and "trazaHasta" in src
