"""[P1-COUNTRY-SYSTEM-F0 · 2026-08-16] Fase 0 del sistema de países: el DATO.

El país es ISO-3166 alpha-2 y canonicaliza en UN solo sitio
(`constants.canonicalize_country`) — la lección de P1-DIET-CANON-SSOT: eran 3
tablas a mano, driftaron, y una sirvió Pollo a vegetarianas. No escribas una 2ª.

Con el knob maestro apagado (default), TODO usuario resuelve a 'DO' y el motor
es byte-idéntico al de hoy. La spec exige que nada se encienda hasta que los 5
países tengan catálogo y vocabularios (decisión del dueño 2026-08-16).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import constants

_BACKEND = Path(__file__).resolve().parent.parent
_FRONTEND = _BACKEND.parent / "frontend"

_CODIGOS = ("DO", "ES", "US", "MX", "PR", "CO")


# ── canonicalize_country ─────────────────────────────────────────────────────

@pytest.mark.parametrize("crudo,esperado", [
    (None, "DO"),
    ("", "DO"),
    ("   ", "DO"),
    ("DO", "DO"),
    ("do", "DO"),
    ("Es", "ES"),
    ("US", "US"),
    ("mx", "MX"),
    ("PR", "PR"),
    ("co", "CO"),
    ("XX", "DO"),                    # desconocido ⇒ fail-safe al nativo
    ("República Dominicana", "DO"),  # texto humano JAMÁS es un código válido
    (123, "DO"),
    (["ES"], "DO"),
])
def test_canonicaliza_a_iso_o_do(crudo, esperado):
    assert constants.canonicalize_country(crudo) == esperado


def test_siempre_devuelve_un_codigo_del_perfil():
    """La salida SIEMPRE es clave de COUNTRY_PROFILES — ninguna rama del motor
    debe manejar un país sin perfil."""
    for crudo in (None, "xx", "ES", "us", "", "garbage", 0):
        assert constants.canonicalize_country(crudo) in constants.COUNTRY_PROFILES


# ── COUNTRY_PROFILES ─────────────────────────────────────────────────────────

def test_perfiles_los_seis_paises_con_claves_completas():
    assert set(constants.COUNTRY_PROFILES.keys()) == set(_CODIGOS)
    for cc, perfil in constants.COUNTRY_PROFILES.items():
        for clave in ("name_es", "currency", "is_beta", "has_native_prices", "default_tz_offset_min"):
            assert clave in perfil, f"{cc} sin {clave}"


def test_solo_rd_es_nativo_con_precios():
    assert constants.COUNTRY_PROFILES["DO"]["is_beta"] is False
    assert constants.COUNTRY_PROFILES["DO"]["has_native_prices"] is True
    for cc in ("ES", "US", "MX", "PR", "CO"):
        assert constants.COUNTRY_PROFILES[cc]["is_beta"] is True, cc
        assert constants.COUNTRY_PROFILES[cc]["has_native_prices"] is False, cc


def test_monedas_y_husos():
    esperado = {
        "DO": ("DOP", 240), "ES": ("EUR", -60), "US": ("USD", 300),
        "MX": ("MXN", 360), "PR": ("USD", 240), "CO": ("COP", 300),
    }
    for cc, (moneda, tz) in esperado.items():
        assert constants.COUNTRY_PROFILES[cc]["currency"] == moneda, cc
        # Convención getTimezoneOffset(): minutos POSITIVOS al oeste de UTC.
        # ES invierno = -60 (UTC+1). Es un DEFAULT por país, no el del usuario.
        assert constants.COUNTRY_PROFILES[cc]["default_tz_offset_min"] == tz, cc


# ── knob maestro ─────────────────────────────────────────────────────────────

def test_knob_maestro_nace_apagado():
    src = (_BACKEND / "constants.py").read_text(encoding="utf-8")
    assert re.search(
        r"COUNTRY_SYSTEM_ENABLED\s*=\s*_env_bool\(\s*\"MEALFIT_COUNTRY_SYSTEM\"\s*,\s*False\s*\)",
        src,
    ), (
        "El knob maestro debe ser _env_bool('MEALFIT_COUNTRY_SYSTEM', False): "
        "la spec prohíbe encender nada hasta que los 5 países estén completos."
    )
