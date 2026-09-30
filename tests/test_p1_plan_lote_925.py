# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-925 · 2026-09-30] El paréntesis que da gramos COCIDOS se cuenta cocido aunque la línea diga «secas».

Medido sobre las 13.778 líneas distintas del corpus con el catálogo real: «¼ taza de habichuelas rojas secas (≈135 g
cocidas)» y «50 g de habichuelas rojas secas (≈135 g cocidas)» contaban 465 kcal — el resolvedor toma los 135 g del
paréntesis, que son COCIDOS, y «secas» cortaba la conversión a la base seca de la fila (344,7 kcal). Reales: ~170.
Lo que NO cambia (primera versión descartada: un test del 284 la paró): «⅓ taza de garbanzos secos (60 g) cocidos» son
60 g en seco, y «85 g de guisantes secos cocidos» (el cerrador, con la nota del 931 «hierve los guisantes secos…») son
gramos secos para hervir.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import cocido_en_catalogo as cc  # noqa: E402


class _Fila:
    def __init__(self, name, kcal, dens=None):
        self.name, self.kcal, self.density_g_per_cup = name, kcal, dens


class _DB:
    _FILAS = {"guisantes secos": _Fila("Guisantes secos", 341.0, 200.0),
              "habichuelas rojas": _Fila("Habichuelas rojas", 344.7, 184.0),
              "garbanzos": _Fila("Garbanzos", 364.0, 200.0)}

    def lookup(self, nombre):
        n = cc._norm(nombre)
        for k in sorted(self._FILAS, key=len, reverse=True):
            if k in n:
                return self._FILAS[k]
        return None


def test_el_parentesis_cocido_manda_aunque_la_linea_diga_secas():
    for linea in ("¼ taza de habichuelas rojas secas (≈135 g cocidas)", "50 g de habichuelas rojas secas (≈135 g cocidas)"):
        g = cc.en_base_de_la_fila(linea, 135.0, _DB())
        assert 47 <= g <= 52, (linea, g)             # 135 × 127/344,7 ≈ 49,7


def test_lo_que_esta_en_seco_no_cambia():
    db = _DB()
    assert cc.en_base_de_la_fila("⅓ taza de garbanzos secos (60 g) cocidos", 60.0, db) == 60.0
    assert cc.en_base_de_la_fila("85 g de guisantes secos cocidos", 85.0, db) == 85.0
    assert cc.en_base_de_la_fila("40 g de habichuelas rojas secas", 40.0, db) == 40.0
    assert cc.en_base_de_la_fila("1¼ tazas de habichuelas rojas cocidas (80 g en seco, bien cocidas)", 80.0, db) == 80.0


def test_knob_apagado_conducta_previa(monkeypatch):
    monkeypatch.setenv("MEALFIT_COOKED_HINT_WINS", "false")
    assert cc.en_base_de_la_fila("¼ taza de habichuelas rojas secas (≈135 g cocidas)", 135.0, _DB()) == 135.0


def test_ancla():
    assert "tooltip-anchor: P1-PLAN-LOTE-925" in (_BACKEND / "cocido_en_catalogo.py").read_text(encoding="utf-8")
