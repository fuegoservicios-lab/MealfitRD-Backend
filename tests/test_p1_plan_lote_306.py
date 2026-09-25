# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-306 · 2026-09-25] Fluidez del teclado del chat y foto en Web Worker (solo frontend).

El dueño: «el teclado y abrir una foto con el teclado abierto no es tan fluido como Gemini». Medido en el arnés del chat
real (80 mensajes, CPU x4): `--kb-ms` escrita en <html> recalculaba el estilo de ~2 650 nodos (97 ms) y `--kb-inset`
heredada por todos los mensajes (57 ms); ahora son `@property … inherits:false` escritas en las piezas `data-kb-anima`.
La foto se prepara en un Web Worker (OffscreenCanvas) con caída al hilo principal. Los tests de verdad viven en
`frontend/src/__tests__/lote306.*.test.js`; este ancla el marcador y que esos ficheros existan.
Tooltip-anchor: P1-PLAN-LOTE-306
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 306


def test_el_frontend_trae_el_worker_y_las_variables_no_heredables():
    if not _FRONT.exists():  # worktree del backend sin el frontend al lado
        return
    assert (_FRONT / "src" / "workers" / "chatImage.worker.js").exists()
    css = (_FRONT / "src" / "index.css").read_text(encoding="utf-8")
    assert "@property --kb-ms { syntax: '*'; inherits: false; }" in css
    assert "@property --kb-inset { syntax: '*'; inherits: false; }" in css
