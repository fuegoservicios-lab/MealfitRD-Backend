# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-743 · 2026-09-28] (G36 paso 3) vitest corre con el sistema de países ENCENDIDO, como producción.

Producción construye con `VITE_COUNTRY_SYSTEM=true` desde el 18-ago; el runner no la definía, así que la suite entera
probaba el mundo de ANTES del flip. Pasos 1-2 (lote 706): las 7 aserciones que fijaban el valor del build pasan a anclar
la regla o a mockear la rama de rollback. Paso 3 (este): `test.env` en `vite.config.js` enciende la bandera. Medido con la
suite entera encendida: 529 ficheros, 4611 tests verdes y 3 rojos, los tres del fixture «formulario completo» de
`P0_12_household_defaults`, al que le faltaba la cocina (obligatoria con el sistema encendido; ya la trae).

tooltip-anchor: P1-PLAN-LOTE-743
"""
import re
from pathlib import Path

import pytest

_FRONT = Path(__file__).resolve().parents[2] / "frontend"


def test_el_runner_enciende_la_bandera_como_el_build():
    if not _FRONT.exists():
        pytest.skip("frontend ausente (repo hermano)")
    cfg = (_FRONT / "vite.config.js").read_text(encoding="utf-8")
    bloque = cfg[cfg.index("  test: {"):]
    assert re.search(r"env:\s*\{[^}]*VITE_COUNTRY_SYSTEM:\s*'true'", bloque)
    prod = (_FRONT / ".env.production").read_text(encoding="utf-8")
    assert re.search(r"^VITE_COUNTRY_SYSTEM=true$", prod, re.M)


def test_el_formulario_completo_trae_la_cocina():
    if not _FRONT.exists():
        pytest.skip("frontend ausente (repo hermano)")
    t = (_FRONT / "src" / "__tests__" / "P0_12_household_defaults.test.jsx").read_text(encoding="utf-8")
    assert "cultureProfiles: { main: 'dominicana', secondary: [] }" in t
