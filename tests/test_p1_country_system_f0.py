"""[P1-COUNTRY-SYSTEM-F0 · 2026-08-16] Fase 0 del sistema de países: el DATO.

El país es ISO-3166 alpha-2 y canonicaliza en UN solo sitio
(`constants.canonicalize_country`) — la lección de P1-DIET-CANON-SSOT: eran 3
tablas a mano, driftaron, y una sirvió Pollo a vegetarianas. No escribas una 2ª.

Con el knob maestro apagado (default), TODO usuario resuelve a 'DO' y el motor
es byte-idéntico al de hoy. La spec exige que nada se encienda hasta que los 5
países tengan catálogo y vocabularios (decisión del dueño 2026-08-16).
"""
from __future__ import annotations

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


# ── cierres de fase ──────────────────────────────────────────────────────────

def test_country_no_es_sensible_y_es_decision_explicita():
    """`country` NO es PII médica: va en mealfit_form plano. La decisión se
    ancla para que nadie lo mueva a SENSITIVE_FIELDS «por si acaso» (cifrar de
    más degrada a no-persistir en browsers sin crypto.subtle) ni un campo
    futuro se cuele sin clasificar citando este precedente."""
    src = _js_sin_comentarios(_FRONTEND / "src" / "config" / "secureFormStorage.js")
    ini = src.index("SENSITIVE_FIELDS = [")
    bloque = src[ini: src.index("]", ini)]
    assert "'country'" not in bloque


def test_el_dato_viaja_pero_el_motor_no_lo_lee_todavia():
    """Fase 0 = SOLO el dato. El día que graph_orchestrator consuma country,
    este test se REESCRIBE apuntando al SSOT (canonicalize_country) — si
    aparece un lector suelto antes de Fase 1, es un lector sin canonicalizar."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    lectores = re.findall(r"form_data(?:\.get\()?\s*\(?['\"]country['\"]", src)
    assert not lectores, (
        "graph_orchestrator lee form_data['country'] antes de la Fase 1: todo "
        "lector debe nacer detrás de canonicalize_country + COUNTRY_SYSTEM_ENABLED."
    )


def test_getcountrylabel_paridad_con_labelkey_de_countries_js():
    """`getCountryLabel` (Settings.jsx) traduce por switch con literales
    `t('...')` escritos a mano — NUNCA `t(c.labelKey)` — porque una clave
    DINÁMICA es invisible para `scripts/i18n-check.mjs` (mismo motivo que
    `sentinelLabel` en QAllergies.jsx; ver comentario en Settings.jsx). Si
    `countries.js` gana un país sin que alguien añada su `case` aquí, esa fila
    degrada en silencio a su código ISO crudo en Configuración — este test
    ancla la PARIDAD, no la grafía de cada línea."""
    settings_src = _js_sin_comentarios(_FRONTEND / "src" / "pages" / "Settings.jsx")
    ini = settings_src.index("const getCountryLabel = ")
    fin = settings_src.index("\n};", ini)
    cuerpo = settings_src[ini:fin]
    literales_t = set(re.findall(r"t\('([^']+)'\)", cuerpo))
    assert literales_t, "No pude parsear los t('...') de getCountryLabel."

    countries_src = _js_sin_comentarios(_FRONTEND / "src" / "config" / "countries.js")
    label_keys = set(re.findall(r"labelKey:\s*'([^']+)'", countries_src))
    assert label_keys, "No pude parsear los labelKey de countries.js."

    assert literales_t == label_keys, (
        "getCountryLabel (Settings.jsx) y labelKey (countries.js) divergen: un "
        "7º país degradaría en silencio a su código ISO crudo en Configuración. "
        f"Solo en getCountryLabel: {literales_t - label_keys}. "
        f"Solo en countries.js: {label_keys - literales_t}."
    )
