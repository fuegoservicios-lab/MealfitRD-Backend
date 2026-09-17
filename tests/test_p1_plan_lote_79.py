# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-79 · 2026-09-17] Topes de longitud del coach MEDIDOS con el proveedor alterno: el modelo se pasa del tope
entre un 5 y un 20 % y la regla P no lo movió (19-20 de 63 casos sobre el tope en dos corridas). Con 65/65/120 (el de
riesgo se queda en 90: es seguridad): 20 → 6 casos sobre el tope, ratio medio 0,89 → 0,77, 76 → 67 palabras de media y
0 fallos duros en la misma batería. Y los emojis de cabecera de una herramienta no se copian (D7 salía con 5)."""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def test_los_topes_medidos_y_el_de_riesgo_intacto():
    p = _src("prompts/chat_agent.py")
    assert "Charla, registro, pregunta puntual o dato del plan: 2-3 frases, MÁXIMO 65 palabras." in p
    assert "Queja o frustración: MÁXIMO 65 palabras" in p
    assert "explicación que el usuario PIDIÓ: MÁXIMO 120 palabras" in p
    assert "Tema de riesgo (regla L y regla I): MÁXIMO 90 palabras." in p          # seguridad: no se recorta
    assert "MÁXIMO 80 palabras" not in p and "MÁXIMO 150 palabras" not in p


def test_los_emojis_de_cabecera_de_una_herramienta_no_se_copian():
    p = _src("prompts/chat_agent.py")
    assert "los emojis de cabecera que traiga una herramienta (🥛, 🛒, 🥩…) NO se copian" in p
    assert "[P1-PLAN-LOTE-79]" in p


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', _src("app.py"))
    assert m and int(m.group(1)) >= 79
    assert "P1-PLAN-LOTE-79" in _src("docs/coach_bateria_2026_09_15.md")
