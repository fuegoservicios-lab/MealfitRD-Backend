# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-808 · 2026-09-29] El pescado del cerrador no se «revuelve hasta que cuaje».

Bloque REAL de producción (plan 92328ff7, semana 3, 29-sep 04:30): «Guineítos verdes con huevo desmenuzado criollo,
espinacas y filete de pescado blanco» — «Cocina huevo y filete de pescado blanco y clara de huevo en la sartén, revueltos,
hasta que cuajen por completo»: la fusión de los pasos del cerrador llegaba entera al lote 421 (claras del cerrador).
"""
from __future__ import annotations

import pasos_cantidades as pc


def _paso(texto):
    m = {"recipe": [texto, "Montaje: sirve."]}
    pc.claras_del_cerrador(m)
    return m["recipe"][0]


def test_el_pescado_lleva_su_punto_y_el_huevo_se_cuaja():
    p = _paso("El Toque de Fuego: cocina la cebolla 5 minutos. Cocina huevo y filete de pescado blanco y clara de huevo a la "
              "plancha o hervido y sírvelos como proteína del plato.")
    assert "pescado blanco en la sartén, revuelt" not in p and "pescado blanco y clara de huevo en la sartén" not in p, p
    assert "Cocina huevo y clara de huevo en la sartén, revueltos, hasta que cuajen por completo." in p, p
    assert "Cocina el filete de pescado blanco a la plancha 3-4 min por lado" in p and "63 °C" in p, p


def test_lo_enlatado_se_sirve_y_los_huevos_solos_como_siempre():
    p = _paso("Cocina clara de huevo y atún en agua a la plancha o hervido y sírvelos como proteína del plato.")
    assert "Sirve atún en agua (ya viene cocido) al lado." in p, p
    assert _paso("Cocina 3 huevos y 2 claras de huevo a la plancha o hervidos y sírvelos como proteína del plato.") == (
        "Cocina 3 huevos y 2 claras de huevo en la sartén, revueltos, hasta que cuajen por completo, y sírvelos como "
        "proteína del plato.")
