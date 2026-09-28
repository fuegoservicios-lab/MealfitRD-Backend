# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-741 · 2026-09-28] (G94, parte de producto) El coach sabe el número de emergencias del país del usuario.

La regla I del coach («síntomas… las señales de alarma que piden médico o emergencias») no da número, y la IA tiende al
9-1-1 de RD: a un usuario de España (112) o de Colombia (123) le daría uno que allí no es el de emergencias. El Aviso
Médico (texto legal) ya dice «si está en otro país, use el número de emergencias local»; nombrar 112/123 ahí sigue siendo
decisión del dueño. Esto es el lado del PRODUCTO: `COUNTRY_PROFILES[*]["emergency_number"]` (SSOT del país) y el bloque de
país del coach lo nombra. RD sigue sin bloque de país: su prompt no cambia ni un byte.

tooltip-anchor: P1-PLAN-LOTE-741
"""
import constants


def test_cada_pais_tiene_su_numero():
    esperado = {"DO": "911", "US": "911", "PR": "911", "MX": "911", "ES": "112", "CO": "123"}
    assert {c: p["emergency_number"] for c, p in constants.COUNTRY_PROFILES.items()} == esperado


def test_el_coach_lo_nombra_fuera_de_rd():
    assert "112" in constants.coach_country_context("ES")
    assert "123" in constants.coach_country_context("CO")
    assert "911" in constants.coach_country_context("MX")
    assert "número de emergencias" in constants.coach_country_context("ES")


def test_rd_no_cambia():
    assert constants.coach_country_context("DO") == ""
