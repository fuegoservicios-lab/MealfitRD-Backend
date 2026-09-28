# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-655 · 2026-09-27] Dos premisas dominicanas que viajaban a todos los países (G72 + G86).

1. El bloque «CATÁLOGO VERIFICADO» abría diciendo «El usuario SOLO puede comprar alimentos con precio verificado en el
   supermercado» también a un usuario de España, cuyo régimen es literalmente `beta_no_prices`: sus filas entran
   PORQUE no tienen precio. Es la premisa de la orden que viene detrás, y para beta era falsa. RD conserva su frase.
2. El prompt de Dreaming (consolidación de memoria, hoy apagado) se presentaba como «coach nutricional dominicano» y
   pedía el perfil «en español dominicano» para todos los usuarios.

tooltip-anchor: P1-PLAN-LOTE-655
"""
import importlib.util
import inspect
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "_vcc", Path(__file__).resolve().parent / "test_p1_verified_catalog_country.py")
_vcc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vcc)
go, catalogo, knob_on, _bloque = _vcc.go, _vcc.catalogo, _vcc.knob_on, _vcc._bloque   # fixtures y helper de ese test


def test_el_pais_beta_no_lee_que_todo_tiene_precio(go, catalogo, knob_on):
    es = _bloque(go, "ES")
    assert "precio verificado en el supermercado" not in es
    assert "USA EXCLUSIVAMENTE ESTOS ALIMENTOS" in es


def test_rd_conserva_su_frase(go, catalogo, knob_on):
    assert "El usuario SOLO puede comprar alimentos con precio verificado en el supermercado." in _bloque(go, "DO")


def test_dreaming_no_se_presenta_como_coach_dominicano():
    import dreaming
    src = inspect.getsource(dreaming)
    assert "coach nutricional dominicano" not in src
    assert "español dominicano" not in src
