"""[P1-COUNTRY-SYSTEM-F1 · 2026-08-16] Fase 1 del sistema de países: la espina.

Fase 0 escribió el DATO (`country`, default 'DO') sin que el motor lo leyera
todavía. Fase 1 abre la ÚNICA puerta de lectura — `constants.country_for_form_data`
— que T2-T7 usarán para que el motor deje de forzar lo criollo (arroz+habichuela,
"pollo guisado", DOP) sobre los países en beta. Con el knob maestro apagado
(default) el motor sigue siendo BYTE-IDÉNTICO: `country_for_form_data` devuelve
'DO' sin importar lo que traiga `form_data`, exactamente igual que si Fase 0
nunca hubiera existido.

Esta fase también hereda y cierra el ruling "parked" que dejó Fase 0: el canal
sin nombre (`_sanitize_form_data_for_prompt`) excluía `'country'` SOLO en la
rama de trim, dejando la puerta abierta a que el país colara al prompt en
cuanto alguien apagara el kill-switch `MEALFIT_PROMPT_TRIM_FORM_DATA`. La
exclusión pasa a ser INCONDICIONAL — vive en ambas ramas — porque el país
viaja a prompts SOLO vía el sistema de variantes (F1-T2/T3), jamás como key
suelta de form_data.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import constants

_BACKEND = Path(__file__).resolve().parent.parent
_FRONTEND = _BACKEND.parent / "frontend"


# ── constants.country_for_form_data ──────────────────────────────────────────

def test_knob_apagado_todo_es_do(monkeypatch):
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    for fd in ({"country": "ES"}, {"country": "xx"}, {}, None, "no-dict"):
        assert constants.country_for_form_data(fd) == "DO"


def test_knob_encendido_canonicaliza(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    assert constants.country_for_form_data({"country": "es"}) == "ES"
    assert constants.country_for_form_data({"country": "basura"}) == "DO"
    assert constants.country_for_form_data({}) == "DO"


def test_no_dict_es_do_incluso_con_knob_encendido(monkeypatch):
    """El contrato de `Produces` es explícito: `form_data` no-dict ⇒ 'DO' bajo
    CUALQUIER estado del knob — no solo cuando está apagado."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    for fd in (None, "no-dict", ["ES"], 42, 3.14):
        assert constants.country_for_form_data(fd) == "DO"


def test_knob_se_lee_por_llamada_no_cacheado_al_importar(monkeypatch):
    """El helper NO debe cachear el knob en un módulo-level constant al
    import (esa es la razón exacta para NO usar `COUNTRY_SYSTEM_ENABLED`
    dentro del helper — ver Task brief). Togglear el env var entre llamadas,
    en el MISMO proceso, debe cambiar el resultado sin reimport."""
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    assert constants.country_for_form_data({"country": "es"}) == "DO"
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    assert constants.country_for_form_data({"country": "es"}) == "ES"
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    assert constants.country_for_form_data({"country": "es"}) == "DO"


# ── sanitizer: exclusión incondicional (descarga el ruling parked de F0) ────

def _sanitizer_cuerpo() -> str:
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    sin_comentarios = "\n".join(
        l for l in src.splitlines() if not l.strip().startswith("#")
    )
    ini = sin_comentarios.index("def _sanitize_form_data_for_prompt")
    fin = sin_comentarios.find("\ndef ", ini + 10)
    return sin_comentarios[ini: fin if fin != -1 else len(sin_comentarios)]


def test_sanitizer_excluye_country_en_passthrough_y_en_trim():
    """F1-T1: Fase 0 solo excluía `'country'` en la rama de TRIM
    (`PROMPT_TRIM_FORM_DATA=True`). El passthrough del kill-switch
    (`if not PROMPT_TRIM_FORM_DATA: return form_data`) devolvía el dict
    COMPLETO — un segundo canal sin gate, reservado justo para cuando alguien
    apague el trim. F1 cierra ese ruling parked: la exclusión debe estar
    presente en AMBAS ramas, incondicional.

    Parser-based, comentarios stripeados, CRLF-safe (mismo patrón que
    `test_el_dato_viaja_pero_el_motor_no_lo_lee_todavia` en F0). La ventana
    del passthrough se acota a la línea del `if` + su siguiente línea (el
    `return` de esa rama es una sola línea por diseño) para no confundirse
    con la exclusión de la rama de trim que viene después."""
    cuerpo = _sanitizer_cuerpo()
    patron = re.compile(r"k\s*!=\s*['\"]country['\"]")

    pos_if = cuerpo.index("if not PROMPT_TRIM_FORM_DATA")
    fin_linea_if = cuerpo.index("\n", pos_if)
    fin_return_if = cuerpo.index("\n", fin_linea_if + 1)
    rama_passthrough = cuerpo[pos_if:fin_return_if]
    assert patron.search(rama_passthrough), (
        "El `return` dentro de `if not PROMPT_TRIM_FORM_DATA:` no excluye "
        "'country' — la exclusión sigue viviendo SOLO en la rama de trim "
        "(conducta F0). Con el kill-switch apagado el país volvería a colar "
        "al prompt del LLM."
    )

    resto = cuerpo[fin_return_if:]
    assert patron.search(resto), (
        "La rama de trim perdió la exclusión de 'country' heredada de F0."
    )


# ── T2: render por país del day-gen (apilado sobre el de dieta) ─────────────
#
# `build_day_generator_system_prompt(diet, country)` apila DOS renders: primero dieta
# (_DIET_FRAGMENT_TABLE, preexistente), luego país (_BETA_FRAGMENT_TABLE, esta task). La
# sutileza de composición que el ledger pre-vuelo marcó "vigilar en review": los targets de
# _BETA_FRAGMENT_TABLE son los fragmentos TAL COMO QUEDARON tras el render de dieta — para
# vegetarian/vegan son la columna correspondiente de _DIET_FRAGMENT_TABLE, NUNCA el verbatim
# balanced. Si fuera al revés, un vegetariano español conservaría los patrones de almuerzo/cena
# BALANCED (con carne): el .replace() de país nunca encontraría ese texto (la dieta ya lo
# reemplazó) ni el texto vegetariano (nunca fue el target). 'DO'/None debe seguir tomando el
# camino EXACTO pre-T2 (ancla `is`) — discharge del ruling F0 #3 (name_es sin anclar, T2 es su
# primer consumidor) y del contrato del plan (para DO el retorno es byte-idéntico al actual).

_BETA_CCS = tuple(cc for cc, p in constants.COUNTRY_PROFILES.items() if p["is_beta"])


def test_build_do_o_none_es_byte_identico_is():
    """`country` None/'DO' ⇒ el retorno actual EXACTO — mismo objeto (ancla `is`). balanced/
    pescatarian devuelven la constante DAY_GENERATOR_SYSTEM_PROMPT; vegetarian/vegan el
    render cacheado en _DIET_PROMPT_RENDER_CACHE. En NINGÚN caso país=DO debe recomputar."""
    from prompts.day_generator import build_day_generator_system_prompt as build
    for diet in ("balanced", "pescatarian", "vegetarian", "vegan"):
        assert build(diet, "DO") is build(diet), f"país='DO' no byte-idéntico para diet={diet}"
        assert build(diet, None) is build(diet), f"país=None no byte-idéntico para diet={diet}"
        assert build(diet) is build(diet), f"llamada repetida sin país no es la misma para diet={diet}"


def test_build_country_desconocido_cae_a_do():
    """Un país fuera de COUNTRY_PROFILES (fail-safe de canonicalize_country) también debe
    tomar el camino DO — nunca queda 'huérfano' sin perfil ni dispara el render beta."""
    from prompts.day_generator import build_day_generator_system_prompt as build
    assert build("balanced", "XX") is build("balanced")
    assert build("vegan", "garbage") is build("vegan")


def test_beta_es_balanced_contiene_pais_no_criollo():
    from prompts.day_generator import build_day_generator_system_prompt as build
    out = build("balanced", "ES")
    assert "España" in out
    assert "Bandera:" not in out
    assert "Locrio" not in out
    assert "Mofongo" not in out
    # cabecera prepend + one-liner repetido tras el bloque de almuerzo (P1-DIET-BLIND-DIRECTIVES:
    # una directiva sola pierde contra órdenes específicas) ⇒ al menos 2 apariciones del país.
    assert out.count("España") >= 2, "la directiva de país debe repetirse tras el bloque de almuerzo"


def test_beta_fragment_table_vegan_sin_carnes():
    """Composición beta+vegan, SCOPED a los bloques reemplazados: itera las 4 filas de
    _BETA_FRAGMENT_TABLE (almuerzo/cena/§15-header-desayuno/§15-snacks), columna 'vegan'. NO
    se assertea sobre el prompt completo — legítimamente menciona pollo/res/cerdo/pescado en
    secciones no tocadas por esta task (§2 distinción ají morrón/cubanela con "pollo a la
    jardinera" como ejemplo, §12 caps de seguridad de embutidos/atún). 'res ' lleva espacio
    final a propósito (memoria: 'res' es substring de 'interesante')."""
    from prompts.day_generator import _BETA_FRAGMENT_TABLE
    forbidden = re.compile(r"pollo|res |cerdo|pescado")
    assert len(_BETA_FRAGMENT_TABLE) >= 2, "faltan filas mínimas (almuerzo + cena)"
    for i, (_target, repl) in enumerate(_BETA_FRAGMENT_TABLE):
        vegan_repl = repl.get("vegan")
        if not vegan_repl:
            continue
        hits = forbidden.findall(vegan_repl)
        assert not hits, f"fila beta #{i}: reemplazo vegano con carne/pescado {hits}: {vegan_repl!r}"


def test_beta_vegan_es_render_sin_criollo():
    """Mismo check que (b) pero para build('vegan', 'ES') — la composición dieta×país completa,
    no solo el balanced."""
    from prompts.day_generator import build_day_generator_system_prompt as build
    out = build("vegan", "ES")
    assert "Bandera:" not in out
    assert "Locrio" not in out
    assert "Mofongo" not in out
    assert "España" in out


def test_name_es_parity_primer_consumidor():
    """Ruling heredado de F0 #3 (name_es de COUNTRY_PROFILES sin anclar — anclarlo con su
    primer consumidor). T2 es ese primer consumidor: cada name_es debe aparecer en el render
    beta de su propio país."""
    from prompts.day_generator import build_day_generator_system_prompt as build
    assert _BETA_CCS, "no hay países beta en COUNTRY_PROFILES — el fixture de este test está vacío"
    for cc in _BETA_CCS:
        name_es = constants.COUNTRY_PROFILES[cc]["name_es"]
        out = build("balanced", cc)
        assert name_es in out, f"{cc}: name_es {name_es!r} ausente del render"


def test_beta_fragment_table_targets_existen_verbatim():
    """Si un futuro edit del prompt (dieta o §15) deriva un target de _BETA_FRAGMENT_TABLE, el
    .replace() del builder queda no-op EN SILENCIO — la misma clase de bug que
    test_fragmentos_balanced_existen_verbatim_en_la_constante ancla para _DIET_FRAGMENT_TABLE.
    Cada target debe existir verbatim en el render de SU columna de dieta (post-render de
    dieta, no el verbatim balanced a secas — la sutileza de composición del brief)."""
    from prompts.day_generator import (
        _BETA_FRAGMENT_TABLE,
        DAY_GENERATOR_SYSTEM_PROMPT,
        build_day_generator_system_prompt as build,
    )
    diet_rendered = {
        "balanced": DAY_GENERATOR_SYSTEM_PROMPT,
        "vegetarian": build("vegetarian"),
        "vegan": build("vegan"),
    }
    for i, (target, repl) in enumerate(_BETA_FRAGMENT_TABLE):
        for diet_key, text in diet_rendered.items():
            t = target.get(diet_key)
            r = repl.get(diet_key)
            assert t is not None and r is not None, f"fila beta #{i} sin columna {diet_key!r}"
            assert t in text, (
                f"fila beta #{i}/{diet_key}: target ya no existe verbatim en el render de esa dieta"
            )
            assert r != t, f"fila beta #{i}/{diet_key}: reemplazo idéntico al target (sin variante real)"


def test_cache_pais_dimensionado_maximo_15():
    """≤3×5 entradas: 3 columnas de dieta (pescatarian colapsa a 'balanced') × 5 países beta."""
    from prompts.day_generator import (
        build_day_generator_system_prompt as build,
        _COUNTRY_PROMPT_RENDER_CACHE,
    )
    for diet in ("balanced", "pescatarian", "vegetarian", "vegan"):
        for cc in _BETA_CCS:
            build(diet, cc)
    assert len(_COUNTRY_PROMPT_RENDER_CACHE) <= 15, (
        f"cache de país sobredimensionada: {len(_COUNTRY_PROMPT_RENDER_CACHE)} entradas"
    )


# ── T2: call sites (_day_system_instruction_for_diet + path sin cache) ──────

_RAW_COUNTRY_RX = re.compile(r"form_data(?:\.get\()?\s*\(?['\"]country['\"]")


def _assert_deriva_pais_via_ssot(cuerpo: str, label: str):
    assert "country_for_form_data(form_data)" in cuerpo, (
        f"{label}: no deriva el país via country_for_form_data(form_data) — la ÚNICA puerta (T1)"
    )
    assert not _RAW_COUNTRY_RX.search(cuerpo), (
        f"{label}: lee form_data['country']/form_data.get('country') crudo — el mismo drift "
        "que P1-DIET-CANON-SSOT pagó (3 tablas a mano, una sirvió Pollo a vegetarianas), "
        "aplicado a país."
    )


def _cuerpo_day_system_instruction_for_diet() -> str:
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    sin_comentarios = "\n".join(
        l for l in src.splitlines() if not l.strip().startswith("#")
    )
    ini = sin_comentarios.index("def _day_system_instruction_for_diet")
    fin = sin_comentarios.find("\ndef ", ini + 10)
    return sin_comentarios[ini: fin if fin != -1 else len(sin_comentarios)]


def _cuerpo_daygen_nocache_path() -> str:
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    sin_comentarios = "\n".join(
        l for l in src.splitlines() if not l.strip().startswith("#")
    )
    m = re.search(
        r"if PROMPT_CACHE_SYSTEM_MESSAGE:\s*\n\s*prompt_text\s*=\s*dynamic_day_prompt\s*\n\s*else:"
        r".*?prompt_text\s*=\s*dynamic_day_prompt\s*\+\s*_bdgsp_nc\(",
        sin_comentarios,
        re.DOTALL,
    )
    assert m, "no encontré la región sin-cache del day-gen (mismo patrón que test_p1_prompt_cache_systemmsg.py)"
    return m.group(0)


def test_day_system_instruction_for_diet_deriva_pais_via_ssot():
    _assert_deriva_pais_via_ssot(
        _cuerpo_day_system_instruction_for_diet(), "_day_system_instruction_for_diet"
    )


def test_daygen_nocache_path_deriva_pais_via_ssot():
    _assert_deriva_pais_via_ssot(_cuerpo_daygen_nocache_path(), "path sin cache del day-gen")


def test_day_system_instruction_for_diet_knob_off_ignora_country(monkeypatch):
    """Knob apagado (default) ⇒ country_for_form_data devuelve 'DO' SIN mirar form_data —
    _day_system_instruction_for_diet debe tomar el camino EXACTO pre-T2 sin importar qué
    'country' venga en form_data."""
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    import graph_orchestrator as go
    assert go._day_system_instruction_for_diet({"dietType": "balanced", "country": "ES"}) \
        is go._DAY_SYSTEM_INSTRUCTION_CACHED
    assert go._day_system_instruction_for_diet({"dietType": "vegana", "country": "ES"}) \
        is go._day_system_instruction_for_diet({"dietType": "vegana"})


def test_day_system_instruction_for_diet_beta_incluye_pais(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    import graph_orchestrator as go
    out = go._day_system_instruction_for_diet({"dietType": "balanced", "country": "es"})
    assert "España" in out
    assert out is not go._DAY_SYSTEM_INSTRUCTION_CACHED
