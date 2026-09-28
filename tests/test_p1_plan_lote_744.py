# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-744 · 2026-09-28] (G93) Playwright prueba el sistema de países con el build REAL.

`frontend/e2e/pais.spec.js`: un invitado llega al paso «¿En qué país haces la compra?» del wizard y ve los seis países.
Es el único sitio donde la app corre con `VITE_COUNTRY_SYSTEM` compilado desde `.env.production`. Verificado en los dos
sentidos el 28-sep: pasa con el build de producción y FALLA con un build con la bandera apagada. Aquí se ancla que el spec
exista y siga mirando el paso de país (no una superficie que el invitado no alcanza, como Configuración).

tooltip-anchor: P1-PLAN-LOTE-744
"""
from pathlib import Path

import pytest

_SPEC = Path(__file__).resolve().parents[2] / "frontend" / "e2e" / "pais.spec.js"


def test_el_spec_de_pais_mira_el_wizard():
    if not _SPEC.exists():
        pytest.skip("frontend ausente (repo hermano)")
    s = _SPEC.read_text(encoding="utf-8")
    assert "¿En qué país haces la compra?" in s and "page.goto('/assessment')" in s
    for nombre in ("República Dominicana", "España", "Estados Unidos", "México", "Puerto Rico", "Colombia"):
        assert nombre in s
