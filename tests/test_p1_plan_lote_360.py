# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-360 · 2026-09-26] La foto al volver de la galería ya no congela el iPhone; «Otra…» automático.

Sonda del dueño (paquete 20260926-034121): fLeida +5479 → la caja empieza a subir → `pausa 769ms` sin fotogramas,
iOS paneando la página (S=335) y recolocación de golpe; la preparación (prepW +6286) empezó DESPUÉS. En el arnés,
añadir la misma foto cuesta ≤117 ms de JS: el congelado era WebKit de iOS decodificando la foto GRANDE para la vista
previa. En iOS la caja espera a la miniatura (worker, sin los 650 ms de espera). Y «Otra…» recalcula solo al
terminar de escribir (Intro o al salir del campo) y se centra a la vista. Solo frontend:
`frontend/src/__tests__/lote360.test.js`, `lote347.test.jsx`. Tooltip-anchor: P1-PLAN-LOTE-360
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 360


def test_el_frontend_trae_la_regla_de_la_vista_previa():
    if not _FRONT.exists():
        return
    assert (_FRONT / "src" / "utils" / "vistaPreviaDelAdjunto.js").exists()
