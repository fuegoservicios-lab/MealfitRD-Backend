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


# ── el único lector preexistente queda tras knob APAGADO ────────────────────

def _cuerpo_similar_patterns() -> str:
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    ini = src.index("def get_similar_user_patterns")
    fin = src.find("\ndef ", ini + 10)
    cuerpo = src[ini: fin if fin != -1 else len(src)]
    return "\n".join(l for l in cuerpo.splitlines() if not l.strip().startswith("#"))


def test_segmentacion_coldstart_gateada_apagada():
    """Poblar `country` (Fase 0 escribe 'DO' por default) NO debe revivir la
    segmentación cultural del cold-start: con 1 usuario español el pool queda
    vacío, y el dominicano que rellena el campo se segmenta contra un pool que
    excluye a los legacy sin clave (el `=` no casa con clave ausente). Se
    enciende con datos, no de rebote."""
    cuerpo = _cuerpo_similar_patterns()
    assert "MEALFIT_COUNTRY_COLDSTART_SEGMENT" in cuerpo, (
        "La rama de país del cold-start perdió su knob: escribir country la "
        "reactiva sin que nadie lo haya decidido."
    )
    pos_knob = cuerpo.index("MEALFIT_COUNTRY_COLDSTART_SEGMENT")
    pos_filtro = cuerpo.find("health_profile->>'country'", pos_knob)
    assert pos_filtro != -1, (
        "El filtro por país ya no está DESPUÉS del knob: o se movió fuera del "
        "gate o se eliminó — ambos cambian conducta sin decisión."
    )
    assert re.search(
        r"_env_bool\(\s*\"MEALFIT_COUNTRY_COLDSTART_SEGMENT\"\s*,\s*False\s*\)", cuerpo
    ), "El knob debe nacer con default False."


# ── paridad frontend↔backend ─────────────────────────────────────────────────

def test_paridad_countries_js_con_country_profiles():
    """Un país añadido en un solo lado es la clase de drift que P1-DIET-CANON-SSOT
    pagó. Parser sobre el fuente JS (sin comentarios, CRLF-safe)."""
    src = (_FRONTEND / "src" / "config" / "countries.js").read_text(encoding="utf-8")
    sin_comentarios = "\n".join(
        re.sub(r"(^|\s)//.*$", r"\1", l)
        for l in re.split(r"\r?\n", re.sub(r"/\*.*?\*/", "", src, flags=re.S))
    )
    codigos_js = re.findall(r"code:\s*'([A-Z]{2})'", sin_comentarios)
    assert codigos_js, "No pude parsear los codes de countries.js"
    assert set(codigos_js) == set(constants.COUNTRY_PROFILES.keys())
    betas_js = dict(re.findall(r"code:\s*'([A-Z]{2})',[^}]*beta:\s*(true|false)", sin_comentarios))
    for cc, perfil in constants.COUNTRY_PROFILES.items():
        assert betas_js.get(cc) == ("true" if perfil["is_beta"] else "false"), cc


# ── el wizard y la rama contador ─────────────────────────────────────────────

def _js_sin_comentarios(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    return "\n".join(
        re.sub(r"(^|\s)//.*$", r"\1", l)
        for l in re.split(r"\r?\n", re.sub(r"/\*.*?\*/", "", src, flags=re.S))
    )


def test_qcountry_usa_el_ssot_y_val_iso():
    src = _js_sin_comentarios(
        _FRONTEND / "src" / "components" / "assessment" / "questions" / "QCountry.jsx"
    )
    assert "from '../../../config/countries'" in src, "QCountry debe leer el SSOT"
    assert "updateData('country'" in src
    assert re.search(r"value=\{[a-zA-Z_.]*code\}", src) or "value={c.code}" in src, (
        "El value del radio debe ser el CODE del SSOT — jamás un literal español."
    )
    assert "'República Dominicana'" not in src.replace("labelKey", ""), (
        "Nombre de país como literal en QCountry: los labels salen del SSOT."
    )


def test_paso_pais_antes_del_submit_y_gated():
    src = _js_sin_comentarios(
        _FRONTEND / "src" / "components" / "assessment" / "InteractiveAssessmentFlow.jsx"
    )
    # [CONTROLLER-RULING · 2026-08-16] `src.find("QCountry")` matchearía la línea
    # del IMPORT (arriba del todo), volviendo trivial el assert "antes del
    # submit". `component: <QCountry` es única al objeto del step.
    pos_country = src.find("component: <QCountry")
    pos_supplements = src.find("component: <QSupplements")
    assert pos_country != -1, "El paso QCountry no está en el flow."
    assert pos_country < pos_supplements, (
        "QCountry quedó DESPUÉS del paso que lleva el submit: el país se "
        "preguntaría después de generar el plan — o sea, nunca."
    )
    ini = src.rfind("COUNTRY_SYSTEM_UI", 0, pos_country)
    # [CONTROLLER-RULING · 2026-08-16] Ventana ampliada a 600 (era 400): el
    # bloque del step (comentario + title/subtitle/fields/component) no cabía
    # holgado en 400 chars tras stripping de comentarios.
    assert ini != -1 and pos_country - ini < 600, (
        "El paso QCountry no está gateado por COUNTRY_SYSTEM_UI: aparecería en "
        "producción antes del flip (la spec exige oscuro total)."
    )


def test_rama_contador_persiste_el_pais():
    src = _js_sin_comentarios(
        _FRONTEND / "src" / "components" / "assessment" / "questions" / "QTrackingFinish.jsx"
    )
    ini = src.index("for (const extra of [")
    bloque = src[ini: ini + 500]
    assert "'country'" in bloque, (
        "El bucle de acompañantes de QTrackingFinish no incluye 'country': en "
        "modo contador el país se cae AL SUELO en silencio (hallazgo del "
        "escéptico del mapa — la rama corta es ALLOWLIST, no spread)."
    )


# ── Task 5: selector en Configuración ────────────────────────────────────────

def test_settings_tiene_selector_de_pais_gateado():
    src = _js_sin_comentarios(_FRONTEND / "src" / "pages" / "Settings.jsx")
    assert "COUNTRY_SYSTEM_UI" in src and "coerceCountry" in src, (
        "Settings no monta el selector de país (o no está gateado en oscuro)."
    )
    ini = src.find("health_profile")
    assert re.search(r"health_profile:\s*\{\s*country", src), (
        "El selector debe persistir por PATCH /api/profile con "
        "{health_profile:{country}} — el mismo merge key-level que usa la rama "
        "contador. Un endpoint nuevo aquí sería plomería duplicada."
    )
