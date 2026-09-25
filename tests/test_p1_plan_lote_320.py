# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-320 · 2026-09-25] La Nevera abre al instante al venir del Agente (solo frontend).

El dueño: «a veces se tarda mucho en pasar del Agente a la Nevera». El backend responde en 4 ms (medido en el VPS);
lo lento era la primera visita (código de la página compilado en ese momento, con pantalla de carga) y el esqueleto
tras un cambio del chat. Medido en el arnés con build de producción (CPU x4, red 400 ms): 471-533 ms con pantalla de
carga -> 213-243 ms sin ella; 527-664 -> 126-156 ms. Tests de verdad: `frontend/src/__tests__/lote320.test.js`.
Tooltip-anchor: P1-PLAN-LOTE-320
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 320


def test_el_frontend_precarga_y_guarda_copia_vieja():
    if not _FRONT.exists():  # worktree del backend sin el frontend al lado
        return
    assert (_FRONT / "src" / "utils" / "precargaDePaginas.js").exists()
    cache = (_FRONT / "src" / "utils" / "pantryCache.js").read_text(encoding="utf-8")
    assert "export const getStaleInventory" in cache and "export const borrarCacheDeInventario" in cache
