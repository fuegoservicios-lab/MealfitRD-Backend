"""[P1-PLAN-LOTE-165 · 2026-09-22] Idiomas en la frontera con el servidor, y el teclado en la parte web.

El backend elegía la frase de progreso del coach de una lista ESPAÑOLA (`get_progress_msg`) y el cliente la pintaba tal
cual en los cinco idiomas. Ahora cada evento `progress` lleva también su `phase` (la clave de esa lista) y el cliente
pinta su frase traducida; `message` se queda por compatibilidad con paquetes OTA anteriores.

Tooltip-anchor: P1-PLAN-LOTE-165
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def test_cada_evento_de_progreso_lleva_su_fase():
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    eventos = re.findall(r"\{'type': 'progress', [^}]*\}", src)
    assert len(eventos) >= 8, eventos
    for ev in eventos:
        m = re.search(r"'phase': '([a-z_]+)', 'message': get_progress_msg\('([a-z_]+)'\)", ev)
        assert m and m.group(1) == m.group(2), f"evento sin fase (o fase distinta de su frase): {ev}"
    claves = set(re.findall(r'"([a-z_]+)": \[', src[src.index("def get_progress_msg"):src.index("def get_progress_msg") + 3000]))
    assert {"analizando", "generando_plan", "modificando_comida", "actualizando_bd",
            "registrando_progreso", "calculando_compras", "buscando_memoria"} <= claves


def test_el_cliente_pinta_la_frase_de_la_fase():
    p = _FRONT / "src" / "pages" / "AgentPage.jsx"
    if not p.exists():
        pytest.skip("frontend ausente")
    a = p.read_text(encoding="utf-8")
    i = a.index("if (dataObj.type === 'progress') {")
    assert "frasesDeFase[dataObj.phase]" in a[i:i + 400]


def test_el_marcador_esta_al_dia():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'^_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · ', src, re.M)
    assert m and int(m.group(1)) >= 165
