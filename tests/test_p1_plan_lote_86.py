# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-86 · 2026-09-17] El medidor de créditos no se muestra en modo seguimiento. Los créditos (`api_usage`)
solo los consumen acciones del generador (generar/analizar plan, cambiar plato, regenerar día, expandir receta, arreglar
sodio, reintentar bloques); el coach tiene su cuota aparte y escanear/anotar no cuentan. Con el plan en pausa nada los
toca: el contador deja de montarlo y el dashboard de plan lo conserva."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8")


def test_el_contador_de_seguimiento_no_monta_el_medidor_y_el_dashboard_de_plan_si():
    tracking = _front("src/components/dashboard/DashboardTracking.jsx")
    assert "import CreditsMeter from './CreditsMeter';" not in tracking and "<CreditsMeter" not in tracking
    assert "[P1-PLAN-LOTE-86" in tracking
    assert "<CreditsMeter" in _front("src/pages/Dashboard.jsx")


def test_lo_que_gasta_credito_sigue_siendo_solo_el_generador():
    """Si alguien cuelga `verify_api_quota` de una superficie del diario o del coach, la premisa del lote cae."""
    import routers.plans as rp, routers.diary as rd, routers.chat as rc, routers.plans_generation as rg
    for mod in (rd, rc):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert "Depends(verify_api_quota)" not in src, f"{mod.__name__} gasta crédito de plan"
    assert "Depends(verify_api_quota)" in Path(rg.__file__).read_text(encoding="utf-8")
    assert "Depends(verify_api_quota)" in Path(rp.__file__).read_text(encoding="utf-8")


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 86
    assert "P1-PLAN-LOTE-86" in (_BACKEND / "docs" / "modo_seguimiento_ui.md").read_text(encoding="utf-8")
