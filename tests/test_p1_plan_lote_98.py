# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-98 · 2026-09-18] Descartada, la invitación al plan no deja NADA en el contador.

El lote 91 la colapsaba a un enlace tenue al final («¿Quieres el plan completo? Enciéndelo aquí»). El dueño, con captura:
«quítalo, ya está el interruptor en configuración». `TurnOnPlanCard` devuelve null en las dos ofertas (encender y reanudar);
la puerta de vuelta es Configuración → Capacidades; el descarte sigue persistiendo. El hueco «plan» vacío sale del reparto
(`:empty`) y en el teléfono las áreas quedan en dos: una fila vacía PRIMERA cobraría 1,5rem de aire sobre el título."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"
_CSS = "src/components/dashboard/DashboardTracking.module.css"


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8")


def _sin_comentarios(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def test_descartada_no_queda_nada_y_el_descarte_persiste():
    jsx = _front("src/components/dashboard/DashboardTracking.jsx")
    assert len(re.findall(r"if \(dismissed\) return null;", jsx)) == 2, "las DOS ofertas (encender y reanudar)"
    assert "turnOnLink" not in jsx
    assert "Enciéndelo aquí" not in jsx and "Reanúdalo aquí" not in jsx
    assert "_DISMISS_KEY = 'mealfit_turnon_card_dismissed'" in jsx


def test_el_hueco_vacio_sale_del_reparto_y_sin_fila_fantasma():
    css = _sin_comentarios(_front(_CSS))
    plano = re.sub(r"\s+", " ", css)
    assert ".turnOnSlot:empty { display: none; }" in plano
    assert '.page:has(.turnOnSlot:empty) { grid-template-areas: "main" "side"; }' in plano
    assert "turnOnLink" not in css


def test_los_catalogos_no_conservan_las_claves_del_enlace():
    for loc in ("en-US", "pt-BR", "fr-FR", "it-IT"):
        cat = _front(f"src/i18n/locales/{loc}.json")
        assert "¿Quieres el plan completo? Enciéndelo aquí" not in cat
        assert "Tu plan está en pausa. Reanúdalo aquí" not in cat


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 98
    assert "P1-PLAN-LOTE-98" in (_BACKEND / "docs" / "modo_seguimiento_ui.md").read_text(encoding="utf-8")
