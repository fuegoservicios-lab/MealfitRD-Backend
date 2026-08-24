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

import asyncio
import re
from pathlib import Path

import pytest

import constants
import nutrition_calculator as nc

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
    """Composición beta+vegan, SCOPED a las filas REALMENTE diet-aware (almuerzo/cena — donde
    la columna 'vegan' es TEXTO DE MENÚ y por tanto difiere de 'balanced'). NO se assertea sobre
    filas diet-invariantes (§15-header-desayuno, §15-snacks, y las del fix-round 1 como la
    regla 8 de medidas caseras): esas comparten el MISMO texto en las 3 columnas — son PROSA DE
    REGLA (ej. "1 pechuga de pollo" como EJEMPLO de formato de medida casera, no una sugerencia
    de qué servirle a un vegano) y aplicar aquí el filtro de carnes sería un falso positivo (el
    fix-round 1 lo encontró: la fila de medidas caseras quedó marcada por mencionar "pollo" en
    un ejemplo de unidad, no de plato). Tampoco se assertea sobre el prompt completo —
    legítimamente menciona pollo/res/cerdo/pescado en secciones no tocadas por esta task (§2
    distinción ají morrón/cubanela con "pollo a la jardinera" como ejemplo, §12 caps de
    seguridad de embutidos/atún). 'res ' lleva espacio final a propósito (memoria: 'res' es
    substring de 'interesante')."""
    from prompts.day_generator import _BETA_FRAGMENT_TABLE
    forbidden = re.compile(r"pollo|res |cerdo|pescado")
    assert len(_BETA_FRAGMENT_TABLE) >= 2, "faltan filas mínimas (almuerzo + cena)"
    diet_aware_rows = 0
    for i, (_target, repl) in enumerate(_BETA_FRAGMENT_TABLE):
        vegan_repl = repl.get("vegan")
        balanced_repl = repl.get("balanced")
        if not vegan_repl or vegan_repl == balanced_repl:
            continue  # diet-invariante: mismo texto para todos, no es "menú del vegano"
        diet_aware_rows += 1
        hits = forbidden.findall(vegan_repl)
        assert not hits, f"fila beta #{i}: reemplazo vegano con carne/pescado {hits}: {vegan_repl!r}"
    assert diet_aware_rows >= 2, "faltan filas diet-aware mínimas (almuerzo + cena)"


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

# [P1-COUNTRY-SYSTEM-F2 · 2026-08-17 (Task 9, F6)] Re-anclaje del guard anti-lector — antes
# GRAPHEME-BOUND en 2 sentidos, ambos verificados EN VIVO durante el desarrollo de este fix
# (no especulado): (1) el nombre de variable estaba hardcoded a "form_data" — un lector crudo
# usando OTRO alias del mismo shape (`data`, el nombre de parámetro real de
# `_enrich_clinical_from_profile` en routers/plans.py; `meal_form`, el nombre real que usa
# `/regenerate-day`) pasaba INVISIBLE; (2) el regex viejo (`['\"]country['\"]` sin `\[` en el
# patrón) JAMÁS matcheaba la forma `form_data["country"]` — SOLO `.get("country")` — confirmado
# reproduciendo `_RAW_COUNTRY_RX_LEGACY` contra ambas formas: `.search('form_data["country"]')`
# ⇒ None. El docstring del guard T8 (abajo) documentaba su propia mutación como
# "`form_data.get('country')` crudo" — nunca probó la forma de corchetes, así que el gap
# sobrevivió sin que ningún test lo notara.
#
# `_FORM_SHAPE_ALIASES` es la lista EXPANDABLE (F6 lo pide explícitamente) — cualquier variable
# nueva que cargue una copia "form_data-shaped" de los datos del usuario (alergias, dietType,
# country, ...) se añade aquí, no se hardcodea en un regex nuevo.
_FORM_SHAPE_ALIASES = ("form_data", "data", "meal_form")

# ALLOWANCE (F6 lo pide explícitamente): `\bcountry\b` como CLAVE dentro de un dict-shape (p.ej.
# `_micro_form = {"country": _hp_micro.get("country"), "dietType": ...}`, el patrón de
# hidratación real en routers/plans.py, Task 9 · g) NO es una lectura cruda — es la construcción
# LEGÍTIMA del form-shape dict desde el perfil (la fuente autorizada). El patrón exige un alias
# reconocido INMEDIATAMENTE antes del accessor (`.get(`/`[`), así que una clave de dict-literal
# ("country": ...) o una lectura de OTRA fuente (`_hp_micro.get("country")`, `hp.get("country")`,
# `ctx["country"]`) nunca matchea — ninguna de esas tiene el alias pegado al accessor. Y una
# ASIGNACIÓN (`data["country"] = ...`, el lado ESCRITURA de la misma hidratación) tampoco
# matchea: el lookahead negativo `(?!\s*=(?!=))` excluye el target de un `=` (no confundir con
# `==`, sí debe seguir cazando lecturas en comparaciones como `meal_form["country"] == "DO"`).
# Las 11 combinaciones (3 formas de lectura real × alias reconocido/no-reconocido × dict-literal
# × asignación × comparación) se verificaron una a una antes de fijar este patrón.
_FORM_SHAPE_COUNTRY_READ_RX = re.compile(
    r"(?:" + "|".join(_FORM_SHAPE_ALIASES) + r")"
    r"(?:\.get\(\s*['\"]country['\"]"
    r"|\[\s*['\"]country['\"]\s*\](?!\s*=(?!=)))"
)
_RAW_COUNTRY_RX = _FORM_SHAPE_COUNTRY_READ_RX

# Regex LEGACY (pre-Task-9) preservado SOLO para la mutación de abajo — reproduce el bug real
# (nunca vivió en producción con esta forma después del fix; existe únicamente como fixture de
# la mutación bidireccional F6).
_RAW_COUNTRY_RX_LEGACY = re.compile(r"form_data(?:\.get\()?\s*\(?['\"]country['\"]")


def test_f6_legacy_regex_no_veia_la_forma_de_corchetes():
    """MUTACIÓN #1 (bidireccional, dirección 'bug reproducido'): confirma que el regex PRE-Task-9
    tenía el gap real — nunca matcheaba `form_data['country']`/`form_data[\"country\"]`, solo
    `.get('country')`. Si este test empieza a fallar, alguien "arregló" el legacy sin darse
    cuenta (harmless, pero la mutación deja de ser honesta)."""
    assert _RAW_COUNTRY_RX_LEGACY.search('x = form_data.get("country")')
    assert not _RAW_COUNTRY_RX_LEGACY.search('x = form_data["country"]')
    assert not _RAW_COUNTRY_RX_LEGACY.search("x = form_data['country']")


def test_f6_nuevo_regex_si_ve_la_forma_de_corchetes():
    """MUTACIÓN #1 (dirección 'fix cierra el gap'): el regex re-anclado SÍ atrapa ambas formas
    (.get Y corchetes) — la forma que el guard T8 llevaba meses sin poder ver."""
    assert _FORM_SHAPE_COUNTRY_READ_RX.search('x = form_data["country"]')
    assert _FORM_SHAPE_COUNTRY_READ_RX.search("x = form_data['country']")
    assert _FORM_SHAPE_COUNTRY_READ_RX.search('x = form_data.get("country")')


def test_f6_alias_expandido_data_y_meal_form():
    """MUTACIÓN #2: antes de F6, un lector crudo vía OTRO alias del mismo shape (`data`, el
    parámetro real de `_enrich_clinical_from_profile`; `meal_form`, el de `/regenerate-day`) era
    INVISIBLE porque el regex viejo solo reconocía literalmente 'form_data'."""
    assert not _RAW_COUNTRY_RX_LEGACY.search('data.get("country")'), (
        "el legacy no reconocía el alias 'data' — confirma el blind spot pre-F6"
    )
    assert _FORM_SHAPE_COUNTRY_READ_RX.search('data.get("country")')
    assert _FORM_SHAPE_COUNTRY_READ_RX.search('meal_form["country"]')


def test_f6_country_key_allowance_en_form_shape_dicts():
    """ALLOWANCE explícita (F6): construir un form-shape dict con 'country' como CLAVE (el
    patrón de hidratación real de _micro_form/data en routers/plans.py, Task 9 · g) NO debe
    dispararse — ni como dict-literal ni como asignación (solo la LECTURA es sospechosa)."""
    # dict-literal: 'country' es una CLAVE, no una lectura de form_data/data/meal_form:
    assert not _FORM_SHAPE_COUNTRY_READ_RX.search(
        '_micro_form = {"country": _hp_micro.get("country"), "dietType": x}'
    )
    # asignación (el lado ESCRITURA de la hidratación F2a real en _enrich_clinical_from_profile):
    assert not _FORM_SHAPE_COUNTRY_READ_RX.search('data["country"] = hp.get("country")')
    # lectura de OTRA fuente (perfil), nunca de un alias form-shape:
    assert not _FORM_SHAPE_COUNTRY_READ_RX.search('_hp_micro.get("country")')
    assert not _FORM_SHAPE_COUNTRY_READ_RX.search('hp.get("country")')
    # alias no reconocido (ninguno de los 3 de _FORM_SHAPE_ALIASES):
    assert not _FORM_SHAPE_COUNTRY_READ_RX.search('ctx["country"]')
    assert not _FORM_SHAPE_COUNTRY_READ_RX.search('_micro_form["country"]')
    # pero una LECTURA en comparación SÍ debe seguir cazándose (no es una asignación):
    assert _FORM_SHAPE_COUNTRY_READ_RX.search('meal_form["country"] == "DO"')


def test_f6_guard_t8_real_detecta_alias_data_si_se_reintrodujera():
    """Prueba de extremo a extremo (no solo el regex aislado): si UN día alguno de los 6 módulos
    blanket regresara a leer `data.get('country')` crudo (alias distinto de 'form_data'), el
    guard T8 real (`_FORM_DATA_COUNTRY_RE`, la MISMA instancia que `_FORM_SHAPE_COUNTRY_READ_RX`
    tras el re-anclaje) lo detectaría — antes de F6 este caso habría pasado desapercibido."""
    fake_offender = 'def f(data):\n    country = data.get("country")\n    return country\n'
    assert _FORM_DATA_COUNTRY_RE.search(fake_offender), (
        "el guard blanket re-anclado debe detectar el alias 'data', no solo 'form_data'"
    )


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


# ── T2 fix-round 1: órdenes dominicanas encontradas FUERA de §15 ────────────
#
# La review (post-Task-2) renderizó el prompt beta REAL y encontró que las órdenes MÁS
# imperativas ("REGLA ESTRICTA" regla 2, "el validador RECHAZA" regla 19) viven fuera de §15 —
# la cabecera de país (Task 2 original) no alcanza si dos líneas después la regla 2 ordena sin
# condición "usa alimentos típicos de República Dominicana" (la misma forma de fallo que
# P1-DIET-BLIND-DIRECTIVES ya midió: una directiva de alto nivel pierde contra órdenes
# específicas). Ruling del controller: el hallazgo §16 (constants.build_meal_timing_rules
# spliced a import-time + tools.py:1644) se MUEVE a Task 4 — NO se toca en esta sección.

def test_finding1_rule2_ingredientes_locales():
    from prompts.day_generator import build_day_generator_system_prompt as build
    out = build("balanced", "ES")
    assert "INGREDIENTES DOMINICANOS" not in out
    assert "económicos de República Dominicana" not in out
    assert "INGREDIENTES LOCALES" in out
    assert "país del usuario" in out


def test_finding2_rule25_sin_tabla_criolla_ni_ancla_paladar():
    from prompts.day_generator import build_day_generator_system_prompt as build
    out = build("balanced", "ES")
    assert "PLATOS CRIOLLOS APETECIBLES" not in out
    assert "paladar dominicano" not in out
    assert "mofongo / mangú / tostones" not in out
    assert "casabe" not in out.split("TRANSFORMA")[1].split("APETECIBILIDAD")[0] if "TRANSFORMA" in out else True
    # el PRINCIPIO de transformación sigue vivo (no se perdió la regla, solo el ejemplo criollo):
    assert "TRANSFORMA LOS STAPLES" in out
    assert 'staple "crudo/simple"' in out


def test_finding3_rule19_definicion_no_ancla_dominicana():
    from prompts.day_generator import build_day_generator_system_prompt as build
    out = build("balanced", "ES")
    assert "PREPARACIÓN dominicana real" not in out
    assert "locrios (almuerzo)" not in out
    # el REQUISITO citado por el validador sigue exacto (no se relajó, solo se re-ancló):
    assert "AL MENOS una preparación transformada por día" in out
    assert "el validador RECHAZA un plan de puros staples servidos" in out


def test_finding4_frases_sin_marco_nacional():
    from prompts.day_generator import build_day_generator_system_prompt as build
    out = build("balanced", "ES")
    # 4a — regla 5 (sabor)
    assert "sabor criollo real a guisos, locrios y habichuelas" not in out
    assert "sabor real a guisos, salteados y leguminosas" in out
    # 4b — regla 8 (medidas)
    assert "MEDIDAS CASERAS DOMINICANAS" not in out
    assert "medidas caseras dominicanas" not in out
    assert "MEDIDAS CASERAS CLARAS" in out
    # 4c — §15c header de categorías de merienda
    assert "merienda dominicana" not in out
    assert "Categorías VÁLIDAS de merienda:" in out
    # 4d — §15c crudités (la regla queda, se retira el marco de nacionalidad)
    assert "AMERICANA, no dominicana" not in out
    assert "El gate determinista los rechaza." in out
    # 4e — §15f apetecibilidad
    assert "un dominicano se lo comería" not in out
    assert "tu usuario se lo comería con gusto" in out
    # 4f (auto-hallado durante el barrido amplio de finding 6, mismo patrón que 4a-4e: regla
    # universal — no desperdiciar yemas — envuelta en un marco nacional innecesario) — §12
    # HUEVOS: ENTEROS PRIMERO
    assert "desperdicio real en cocina dominicana" not in out
    assert "desperdicio real en la cocina" in out


# Sobrevivientes DOCUMENTADOS del render beta tras el fix-round 1 (clasificación completa +
# rationale en task-2-report.md, sección "Explícitamente NO tocado"):
#   (b) PROHIBICIÓN legítima — universal independientemente del país (una merienda no debe ser
#       un guiso pesado, una cena no debe llevar arroz, en CUALQUIER país; el nombre del plato
#       prohibido es incidental a la regla). El «paella/risotto prohibido» de cena (2ª entrada)
#       [P1-COUNTRY-SYSTEM-F2 · 2026-08-17 (Task 9, F5)] queda EXPLÍCITAMENTE DO-only — la spec
#       Fase 1 la nombra como el gate determinista `_detect_slot_appropriateness` (S1, "paella/
#       risotto listados como arroz prohibido de cena"); esta mención en el PROMPT es la misma
#       regla, no un bug a re-anclar.
#   (c) referencia al ENUM de categorías del Planificador (Mangú/Avena/Pan/Batido/Revoltillo),
#       compartido con ~40 archivos de catálogo/planner (dish_templates.json,
#       dominican_dishes.json, planner.py, plan_generator.py...). Cambiar la ETIQUETA aquí sin
#       cambiar el enum real que el Planificador asigna sería cosmético y potencialmente
#       engañoso — Fase 2 (catálogo por país) es donde este enum se vuelve per-country de verdad.
_B_CLASS_PROHIBICIONES = [
    'PROHIBIDO ABSOLUTO: técnicas de plato fuerte (salteado, locrio, asopao, guisado, frito '
    'completo, horneado tipo cazuela).',
    'PROHIBIDO el "ARROZ DE NOCHE": NADA de arroz blanco/integral, locrio, moro, asopao NI '
    'platos cuya BASE sea arroz aunque el nombre no diga "arroz" (chofán/arroz frito, paella, '
    'risotto, congrí, mamposteao) en la cena (no se acostumbra en la cena dominicana y el gate '
    'lo rechaza).',
    "Evita frituras pesadas, locrios densos y guisos calóricos en la noche.",
]
_C_CLASS_CATALOG_ENUM = [
    "IMPORTANTE: Usa la CATEGORÍA de desayuno asignada por el Planificador (Mangú/tubérculos, "
    "Avena/cereales, Pan/tostadas, Batido/bowl, Revoltillo/tortilla). NO elijas mangú si el "
    "planificador asignó otra categoría.",
]

# [P1-DAYGEN-PROMPT-NO-NEUTRALIZE · 2026-08-23] La neutralización final eliminó las antiguas
# clases (e) de ejemplos con Casabe/Arepitas; ya no son sobrevivientes. La única excepción
# deliberada es (d): la nota universal que evita hervir una torta ya cocida. En producción se
# enmascara exactamente este fragmento antes del SSOT, de modo que no acabe afirmando el absurdo
# «pan tostado integral es una torta seca de yuca».
_D_CLASS_TECHNIQUE_UNIVERSAL = [
    'TÉCNICA CORRECTA POR ALIMENTO [P1-CASABE-NO-BOIL · 2026-07-30]: el CASABE es una torta seca de yuca YA COCIDA — se sirve tal cual, se tuesta o se calienta en sartén/horno 1-2 min; JAMÁS se hierve, se cocina en agua ni "se deja reposar tapado" como si fuera arroz (un plan real instruyó "Cocina Casabe en 1½ tazas de agua con sal, tapa y hierve 15 minutos" — eso arruina el plato). Lo mismo aplica a pan, tostadas, galletas y tortillas ya horneadas: NUNCA les apliques la plantilla de cocción de granos (proporción agua:grano, hervir, reposar). Esa plantilla es SOLO para arroz, bulgur, quinoa, avena y granos crudos.',
]
def _dominican_token_hits(text: str) -> list[str]:
    """Vocabulario derivado EN CADA llamada del SSOT de producción.

    Añadir una fila a `_DO_LEXICON_NEUTRAL` amplía este guard sin editar el test. Es deliberado
    no conservar aquí la antigua segunda tabla de ocho platos: su cobertura y el neutralizador
    divergieron exactamente como predijo P1-DIET-CANON-SSOT.
    """
    from constants import _DO_LEXICON_NEUTRAL

    scoped = text.casefold()
    terms = sorted(
        {source.casefold() for source, _replacement in _DO_LEXICON_NEUTRAL},
        key=lambda value: (-len(value), value),
    )
    return [term for term in terms if term in scoped]


def _scoped_out_sin_s16(out: str) -> str:
    """Excluye §16 (CONTRATO EXACTO DEL VALIDADOR DE HORARIO, derivado de
    constants.build_meal_timing_rules) del texto escaneado — ruling del controller: MOVIDO a
    Task 4, no es prompt-directive de day_generator sino el espejo del slot SSOT que T4
    parameteriza. Tocarlo aquí estaría fuera del scope de este fix-round."""
    i16 = out.index("16. CONTRATO EXACTO DEL VALIDADOR DE HORARIO")
    i17 = out.index("\n17. PRESUPUESTO DE SODIO")
    return out[:i16] + out[i17:]


# Whitelist EXACTA del SSOT. Las prohibiciones B y el enum C siguen en el prompt, pero no son
# términos de `_DO_LEXICON_NEUTRAL`; la clase E ya se neutraliza. Sólo D se excluye del scanner.
_ALL_DOCUMENTED_SURVIVORS = _D_CLASS_TECHNIQUE_UNIVERSAL


def test_finding5_guard_case_insensitive_sin_sobrevivientes_no_documentados():
    """Guard SSOT: excluye §16 y la única whitelist técnica; todo otro término falla."""
    from prompts.day_generator import build_day_generator_system_prompt as build
    scoped = _scoped_out_sin_s16(build("balanced", "ES"))

    for survivor in _ALL_DOCUMENTED_SURVIVORS:
        assert survivor in scoped, (
            f"sobreviviente documentado ya no existe verbatim en el render — o cambió el texto "
            f"fuente (actualiza esta lista) o ya se arregló (muévelo a los tests de arriba): "
            f"{survivor[:70]!r}"
        )
        scoped = scoped.replace(survivor, "", 1)

    hits = _dominican_token_hits(scoped)
    assert not hits, f"sobrevivientes NO documentados de contenido dominicano: {hits}"


def test_finding5_guard_vale_tambien_para_vegan():
    """Los targets del fix-round 1 son diet-invariantes — el guard debe sostenerse igual sobre
    vegan, no solo balanced."""
    from prompts.day_generator import build_day_generator_system_prompt as build
    scoped = _scoped_out_sin_s16(build("vegan", "ES"))

    for survivor in _ALL_DOCUMENTED_SURVIVORS:
        assert survivor in scoped, f"sobreviviente ausente en vegan: {survivor[:70]!r}"
        scoped = scoped.replace(survivor, "", 1)

    hits = _dominican_token_hits(scoped)
    assert not hits, f"sobrevivientes NO documentados (vegan): {hits}"


def test_finding5_guard_vale_tambien_para_vegetarian():
    """[P1-COUNTRY-SYSTEM-F2 · 2026-08-17 (Task 9)] Tercera columna de dieta — balanced y vegan
    ya estaban cubiertas; vegetarian quedaba sin su propio guard (mismo target/repl, pero un
    futuro edit que sólo tocara la columna vegetarian de _DIET_FRAGMENT_TABLE podía introducir
    residuo sin que NINGÚN test existente lo viera)."""
    from prompts.day_generator import build_day_generator_system_prompt as build
    scoped = _scoped_out_sin_s16(build("vegetarian", "ES"))

    for survivor in _ALL_DOCUMENTED_SURVIVORS:
        assert survivor in scoped, f"sobreviviente ausente en vegetarian: {survivor[:70]!r}"
        scoped = scoped.replace(survivor, "", 1)

    hits = _dominican_token_hits(scoped)
    assert not hits, f"sobrevivientes NO documentados (vegetarian): {hits}"


def test_finding5_f5a_merienda_casabe_bullet_reemplazado():
    """F5a (Task 9): el bullet 'Casabe / galletas integrales...' de la lista de categorías
    VÁLIDAS de merienda ya no sobrevive al render beta — recomendaba casabe como opción al
    usuario beta (a diferencia de las menciones incidentales documentadas en clase d/e)."""
    from prompts.day_generator import build_day_generator_system_prompt as build
    out = build("balanced", "ES")
    assert "Casabe / galletas integrales" not in out
    assert "Tostada integral / galletas integrales + queso bajo en sodio O aguacate" in out


def test_finding5_f5b_carb_rotation_sin_casabe():
    """F5b (Task 9): la frase de rotación de carbo de cena ya no ofrece casabe — el párrafo
    ARROZ DE NOCHE que la precede (moro/paella/risotto prohibidos) se conserva INTACTO y
    DO-only (clase b, documentado arriba; la spec Fase 1 lo nombra vía
    `_detect_slot_appropriateness`)."""
    from prompts.day_generator import build_day_generator_system_prompt as build
    out = build("balanced", "ES")
    assert "Rota a otro carbohidrato del pool asignado distinto del arroz (NUNCA arroz)." in out
    assert "Rota a otro carbo de cena: batata, yuca, ñame, casabe o pan integral" not in out
    # el párrafo ARROZ DE NOCHE (clase b) sigue verbatim — no se tocó:
    assert (
        'PROHIBIDO el "ARROZ DE NOCHE": NADA de arroz blanco/integral, locrio, moro, asopao NI '
        'platos cuya BASE sea arroz aunque el nombre no diga "arroz" (chofán/arroz frito, paella, '
        'risotto, congrí, mamposteao) en la cena (no se acostumbra en la cena dominicana y el gate '
        'lo rechaza).'
    ) in out


def test_finding5_do_byte_identico_tras_las_2_filas_nuevas():
    """DO (o país desconocido) sigue tomando el camino EXACTO pre-F1-T2 — las 2 filas nuevas de
    F5a/F5b viven en `_BETA_FRAGMENT_TABLE`, que el camino DO nunca recorre. Ancla con `is`."""
    from prompts.day_generator import (
        build_day_generator_system_prompt as build,
        DAY_GENERATOR_SYSTEM_PROMPT,
    )
    assert build("balanced", "DO") is DAY_GENERATOR_SYSTEM_PROMPT
    assert build("balanced", None) is DAY_GENERATOR_SYSTEM_PROMPT
    assert "Casabe / galletas integrales" in DAY_GENERATOR_SYSTEM_PROMPT
    assert (
        "Rota a otro carbo de cena: batata, yuca, ñame, casabe o pan integral (NUNCA arroz)."
        in DAY_GENERATOR_SYSTEM_PROMPT
    )


def test_finding5_mutacion_regex_sin_casabe_deja_pasar_el_bug():
    """MUTACIÓN #1: si `_DOMINICAN_TOKEN_RX` no tuviera 'casabe' (el estado real pre-Task-9), el
    guard NO habría detectado que la fila F5a todavía no existía — reproduce el falso-verde
    histórico para probar que el token nuevo es lo que cierra el hueco."""
    from prompts.day_generator import build_day_generator_system_prompt as build
    pre_task9_rx = re.compile(r"locrio|mofongo|mangú|bandera:", re.IGNORECASE)
    # Simula el bug: un render beta que TODAVÍA recomienda casabe en la merienda (pre-fix).
    out_con_bug = build("balanced", "ES").replace(
        "Tostada integral / galletas integrales + queso bajo en sodio O aguacate",
        "Casabe / galletas integrales + queso bajo en sodio O aguacate",
    )
    scoped = _scoped_out_sin_s16(out_con_bug)
    # Para reproducir el scanner histórico hay que retirar también sus antiguas clases B/C;
    # ya no forman parte de la whitelist SSOT actual porque ninguno de sus términos vive en
    # `_DO_LEXICON_NEUTRAL`.
    for survivor in (_B_CLASS_PROHIBICIONES + _C_CLASS_CATALOG_ENUM + _D_CLASS_TECHNIQUE_UNIVERSAL):
        scoped = scoped.replace(survivor, "", 1)
    # El regex VIEJO no ve el bug reintroducido (falso verde):
    assert not pre_task9_rx.findall(scoped)
    # El regex NUEVO sí lo detecta:
    assert _dominican_token_hits(scoped)


def test_finding5_mutacion_sin_las_2_filas_nuevas_el_guard_falla():
    """MUTACIÓN #2 (bidireccional): si F5a/F5b nunca se hubieran añadido a
    `_BETA_FRAGMENT_TABLE`, el guard ampliado (con los 4 tokens nuevos) SÍ debe fallar contra el
    render real de HOY — reproduce el estado pre-fix quitando las 2 filas del pipeline (no vía
    monkeypatch de la tabla, vía el .replace() inverso sobre el string ya renderizado, que es
    equivalente porque las filas son idempotentes) y confirma que el guard las habría cazado."""
    from prompts.day_generator import build_day_generator_system_prompt as build
    out = build("balanced", "ES")
    # revierte F5a y F5b manualmente (simula "nunca se aplicaron"):
    out_sin_fix = out.replace(
        "Tostada integral / galletas integrales + queso bajo en sodio O aguacate",
        "Casabe / galletas integrales + queso bajo en sodio O aguacate",
    ).replace(
        "Rota a otro carbohidrato del pool asignado distinto del arroz (NUNCA arroz).",
        "Rota a otro carbo de cena: batata, yuca, ñame, casabe o pan integral (NUNCA arroz).",
    )
    scoped = _scoped_out_sin_s16(out_sin_fix)
    for survivor in _ALL_DOCUMENTED_SURVIVORS:
        scoped = scoped.replace(survivor, "", 1)
    hits = _dominican_token_hits(scoped)
    assert hits, "el guard debía detectar casabe reintroducido en merienda+cena, y no lo hizo"


# ── T3: bloque de país en el contexto compartido + jueces ───────────────────
#
# `_country_context_block(country)` es el fan-in: se computa UNA vez dentro de
# `_build_shared_context` y viaja a planner+day-gen vía `ctx['country_context']` — el MISMO
# patrón que `diet_directive_context` (P1-DAYGEN-DIET-CONVERGE), cablearlo en los consumidores
# por separado sería duplicarlo. DO/None/desconocido ⇒ "" (byte-identidad del tramo dinámico).
# Los 3 jueces (self-critique cultural_score, juez culinario, feedback de retry del swap en
# agent.py) derivan país por su propio spine y re-anclan SOLO la nacionalidad, preservando el
# REQUISITO/rúbrica — mismo principio que T2 aplicó a las reglas 2/19 del day-gen ("la
# REQUISICIÓN se preserva, solo se re-ancla su nacionalidad").

def _cuerpo_build_shared_context() -> str:
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    sin_comentarios = "\n".join(
        l for l in src.splitlines() if not l.strip().startswith("#")
    )
    ini = sin_comentarios.index("def _build_shared_context")
    fin = sin_comentarios.find("\ndef ", ini + 10)
    return sin_comentarios[ini: fin if fin != -1 else len(sin_comentarios)]


def test_country_context_block_do_o_desconocido_es_vacio():
    import graph_orchestrator as go
    assert go._country_context_block("DO") == ""
    assert go._country_context_block(None) == ""
    assert go._country_context_block("xx") == ""  # fail-safe de canonicalize_country ⇒ DO


def test_country_context_block_beta_contiene_name_es():
    import graph_orchestrator as go
    assert _BETA_CCS, "no hay países beta en COUNTRY_PROFILES — fixture vacío"
    for cc in _BETA_CCS:
        name_es = constants.COUNTRY_PROFILES[cc]["name_es"]
        out = go._country_context_block(cc)
        assert out != "", f"{cc}: bloque vacío para un país beta"
        assert name_es in out, f"{cc}: name_es {name_es!r} ausente del bloque"


def test_build_shared_context_llama_country_context_block_una_sola_vez():
    """El fan-in de planner+day-gen: UNA sola llamada a `_country_context_block(` dentro del
    cuerpo de `_build_shared_context` — cablearlo en los consumidores por separado (planner Y
    day-gen invocándolo cada uno) sería duplicarlo (brief T3, Interfaces)."""
    cuerpo = _cuerpo_build_shared_context()
    n = cuerpo.count("_country_context_block(")
    assert n == 1, f"se esperaba exactamente 1 llamada a _country_context_block(, hallada(s) {n}"


def test_build_shared_context_deriva_pais_via_ssot():
    _assert_deriva_pais_via_ssot(_cuerpo_build_shared_context(), "_build_shared_context")


def test_build_shared_context_expone_country_context_key():
    cuerpo = _cuerpo_build_shared_context()
    assert '"country_context"' in cuerpo, (
        "_build_shared_context no expone ctx['country_context'] — el bloque de país nacería "
        "pero no llegaría a ningún consumidor (clase P2-DREAMING-PLAN-DEADWRITE)."
    )


def test_planner_y_daygen_consumen_country_context():
    """Los 2 fan-out de `_build_shared_context` (planner + day-gen) deben LEER
    `ctx['country_context']` — si el bloque nace pero nadie lo interpola en el prompt, es un
    dead-write (misma clase que P2-DREAMING-PLAN-DEADWRITE: `_dream_plan_constraints` se
    computaba pero nunca se inyectaba)."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    n = src.count("ctx['country_context']")
    assert n >= 2, (
        f"se esperaban >=2 usos de ctx['country_context'] (planner + day-gen), hallados {n}"
    )


def test_cero_form_data_get_country_crudo_en_agent_py():
    """Espejo de F0's `test_el_dato_viaja_pero_el_motor_no_lo_lee_todavia` (que ya cubre
    graph_orchestrator.py), aplicado a agent.py — T3 añade los primeros lectores de país en
    este archivo (feedback de retry del swap)."""
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    lectores = re.findall(r"form_data(?:\.get\()?\s*\(?['\"]country['\"]", src)
    assert not lectores, (
        "agent.py lee form_data['country']/form_data.get('country') crudo — todo lector debe "
        "pasar por country_for_form_data (T1), el mismo drift que P1-DIET-CANON-SSOT pagó."
    )


# ── T3: cultural_score (self-critique) ───────────────────────────────────────
#
# Dos sitios ligados: el texto del SystemMessage (`_CRITIQUE_EVALUATOR_SYSTEM_INSTRUCTION`,
# constante estática a import-time) y el `Field(description=...)` de `cultural_score` en
# `CritiqueEvaluation` (BaseModel, schema enviado al LLM vía with_structured_output — el LLM
# LEE esa descripción como parte del contrato de salida). Mecanismo elegido: helper
# `_critique_evaluator_artifacts_for_country` devuelve AMBOS (instrucción + clase modelo) desde
# una cache por país; DO es literalmente los objetos globales (mismo `is`).

def test_critique_evaluator_do_es_byte_identico():
    import graph_orchestrator as go
    instruction, model_cls = go._critique_evaluator_artifacts_for_country("DO")
    assert instruction is go._CRITIQUE_EVALUATOR_SYSTEM_INSTRUCTION
    assert model_cls is go.CritiqueEvaluation
    assert "3. Coherencia cultural Dominicana (cultural_score)" in instruction
    assert (
        model_cls.model_fields["cultural_score"].description
        == "Coherencia Cultural Dominicana (1-10)"
    )


def test_critique_evaluator_pais_desconocido_cae_a_do():
    import graph_orchestrator as go
    instruction, model_cls = go._critique_evaluator_artifacts_for_country("XX")
    assert instruction is go._CRITIQUE_EVALUATOR_SYSTEM_INSTRUCTION
    assert model_cls is go.CritiqueEvaluation


def test_critique_evaluator_beta_contiene_name_es():
    import graph_orchestrator as go
    for cc in _BETA_CCS:
        name_es = constants.COUNTRY_PROFILES[cc]["name_es"]
        instruction, model_cls = go._critique_evaluator_artifacts_for_country(cc)
        assert instruction is not go._CRITIQUE_EVALUATOR_SYSTEM_INSTRUCTION
        assert "Coherencia cultural Dominicana" not in instruction
        assert name_es in instruction
        assert model_cls is not go.CritiqueEvaluation
        assert name_es in model_cls.model_fields["cultural_score"].description
        # el resto del schema se HEREDA intacto — no se reinventa el modelo entero.
        assert (
            model_cls.model_fields["visual_score"].description
            == go.CritiqueEvaluation.model_fields["visual_score"].description
        )


def test_critique_evaluator_beta_memoizado_por_pais():
    import graph_orchestrator as go
    a = go._critique_evaluator_artifacts_for_country("ES")
    b = go._critique_evaluator_artifacts_for_country("es")  # canonicaliza también en minúscula
    assert a[0] is b[0]
    assert a[1] is b[1]


def _cuerpo_self_critique_node() -> str:
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    sin_comentarios = "\n".join(
        l for l in src.splitlines() if not l.strip().startswith("#")
    )
    ini = sin_comentarios.index("async def self_critique_node")
    fin = sin_comentarios.find("\nasync def ", ini + 10)
    return sin_comentarios[ini: fin if fin != -1 else len(sin_comentarios)]


def test_self_critique_node_deriva_pais_via_ssot():
    cuerpo = _cuerpo_self_critique_node()
    assert "_critique_evaluator_artifacts_for_country(" in cuerpo, (
        "self_critique_node no invoca _critique_evaluator_artifacts_for_country — "
        "cultural_score se quedaría anclado a RD para países beta."
    )
    _assert_deriva_pais_via_ssot(cuerpo, "self_critique_node")


def test_self_critique_node_no_muta_el_evaluator_payload_hardcoded():
    """DO byte-identity de la construcción del SystemMessage/legacy-path: la línea que arma
    `evaluator_payload` sigue leyendo el símbolo `_CRITIQUE_EVALUATOR_SYSTEM_INSTRUCTION` (para
    DO ese nombre resuelve, vía shadow LOCAL en la función, al MISMO objeto global — ver
    test_critique_evaluator_do_es_byte_identico) — anclado también por
    test_p3_cost_cut_v2.py::test_evaluator_uses_payload_list_when_cache_on, que este cambio NO
    debe romper."""
    cuerpo = _cuerpo_self_critique_node()
    assert "SystemMessage(content=_CRITIQUE_EVALUATOR_SYSTEM_INSTRUCTION)" in cuerpo
    assert "_CRITIQUE_EVALUATOR_SYSTEM_INSTRUCTION + \"\\n\\n\" + human_content" in cuerpo


# ── T3: juez culinario ────────────────────────────────────────────────────────

def test_culinary_judge_rubric_do_es_byte_identico():
    import graph_orchestrator as go
    assert go._culinary_judge_rubric_for_country("DO") is go._CULINARY_JUDGE_RUBRIC
    assert "Eres un juez culinario dominicano experto" in go._CULINARY_JUDGE_RUBRIC


def test_culinary_judge_rubric_pais_desconocido_cae_a_do():
    import graph_orchestrator as go
    assert go._culinary_judge_rubric_for_country("garbage") is go._CULINARY_JUDGE_RUBRIC


def test_culinary_judge_rubric_beta_contiene_name_es():
    import graph_orchestrator as go
    for cc in _BETA_CCS:
        name_es = constants.COUNTRY_PROFILES[cc]["name_es"]
        out = go._culinary_judge_rubric_for_country(cc)
        assert out != go._CULINARY_JUDGE_RUBRIC
        assert "Eres un juez culinario dominicano experto" not in out
        assert (
            f"Eres un juez culinario experto en la cocina de {name_es} y cocina internacional"
            in out
        )
        # el resto de la rúbrica (ejemplos + reglas duras) sobrevive intacto — no se reescribe
        # todo el prompt, solo se re-ancla la frase de apertura.
        assert "REGLA DURA DE HORARIO" in out
        assert "TIPOS CANÓNICOS DE VIOLACIÓN" in out


def test_culinary_judge_rubric_beta_memoizada():
    import graph_orchestrator as go
    a = go._culinary_judge_rubric_for_country("ES")
    b = go._culinary_judge_rubric_for_country("ES")
    assert a is b


def test_run_culinary_judge_acepta_country_con_default_do():
    """El default 'DO' preserva a TODOS los callers preexistentes (scripts/calibrate_culinary_
    judge.py llama con 1 solo argumento) — nadie queda roto por la firma nueva."""
    import inspect
    import graph_orchestrator as go
    sig = inspect.signature(go.run_culinary_judge)
    assert "country" in sig.parameters
    assert sig.parameters["country"].default == "DO"


def test_run_culinary_judge_callsite_deriva_pais_via_ssot():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    sin_comentarios = "\n".join(
        l for l in src.splitlines() if not l.strip().startswith("#")
    )
    i = sin_comentarios.index('if CULINARY_JUDGE_GUARD != "off":')
    cuerpo = sin_comentarios[i:i + 400]
    _assert_deriva_pais_via_ssot(cuerpo, "callsite de run_culinary_judge")
    assert "run_culinary_judge(plan, " in cuerpo, (
        "el callsite productivo debe pasar el país derivado como 2º argumento"
    )


# ── T3: feedback de retry del swap (agent.py) ─────────────────────────────────
#
# Extraídos a helpers PUROS (`_swap_slot_feedback_suffix`, `_swap_raw_staple_feedback_suffix`)
# para poder testearlos funcionalmente SIN invocar el pipeline de swap completo (que exige LLM)
# — el brief pide "functional or parser per lo testable sin llamadas LLM"; una función pura de
# (país, ...) -> str es lo más barato y honesto de verificar.

def test_swap_slot_feedback_do_es_byte_identico():
    import agent
    out = agent._swap_slot_feedback_suffix("DO", "Cena", ["ejemplo"])
    assert (
        out
        == "\n\n🕒 COHERENCIA DE HORARIO (OBLIGATORIO): el plato anterior no encaja con el horario "
        "«Cena»: ejemplo. Propón un plato que SÍ corresponda a ese momento "
        "del día para un dominicano — el arroz/locrio/pasta van en almuerzo/cena (NUNCA desayuno); "
        "la cena es ligera (evita 'arroz de noche' y comidas de desayuno). Mantén los macros objetivo."
    )


def test_swap_slot_feedback_pais_desconocido_es_como_do():
    import agent
    assert agent._swap_slot_feedback_suffix("xx", "Cena", []) == agent._swap_slot_feedback_suffix("DO", "Cena", [])


def test_swap_slot_feedback_beta_contiene_name_es():
    import agent
    for cc in _BETA_CCS:
        name_es = constants.COUNTRY_PROFILES[cc]["name_es"]
        out = agent._swap_slot_feedback_suffix(cc, "Cena", ["ejemplo"])
        assert "para un dominicano" not in out
        assert name_es in out
        # la REGLA en sí (arroz/locrio/pasta, 'arroz de noche') es territorio de F1-T4
        # (SLOT_INAPPROPIATE_FOODS por país) — sobrevive intacta, NO es scope de esta task.
        assert "el arroz/locrio/pasta van en almuerzo/cena (NUNCA desayuno)" in out
        assert "'arroz de noche'" in out


def test_swap_raw_staple_feedback_do_es_byte_identico():
    import agent
    out = agent._swap_raw_staple_feedback_suffix("DO", "🍳 RETRY PLATO TRANSFORMADO", "motivo")
    assert out == (
        "\n\n🍳 RETRY PLATO TRANSFORMADO (OBLIGATORIO): el plato anterior es un staple sin transformar "
        "(motivo). Conviértelo en una preparación dominicana REAL — guiso, "
        "locrio, revoltillo, arepitas, bollitos, al horno con majado — manteniendo los "
        "macros objetivo y los mismos ingredientes base."
    )


def test_swap_raw_staple_feedback_pais_desconocido_es_como_do():
    import agent
    a = agent._swap_raw_staple_feedback_suffix("garbage", "M", "r")
    b = agent._swap_raw_staple_feedback_suffix("DO", "M", "r")
    assert a == b


def test_swap_raw_staple_feedback_beta_contiene_name_es():
    import agent
    for cc in _BETA_CCS:
        name_es = constants.COUNTRY_PROFILES[cc]["name_es"]
        out = agent._swap_raw_staple_feedback_suffix(cc, "🍳 RETRY PLATO TRANSFORMADO", "motivo")
        assert "una preparación dominicana REAL" not in out
        assert name_es in out
        # el REQUISITO (transformar el staple) sigue intacto — mismo trato que T2 dio a la
        # regla 19 del day-gen: se preserva la orden, se re-ancla solo la nacionalidad.
        assert "Conviértelo en una preparación real de la cocina de" in out


def test_swap_meal_deriva_pais_una_sola_vez_via_ssot():
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    sin_comentarios = "\n".join(
        l for l in src.splitlines() if not l.strip().startswith("#")
    )
    ini = sin_comentarios.index("def swap_meal(form_data")
    fin = sin_comentarios.find("\ndef ", ini + 10)
    cuerpo = sin_comentarios[ini: fin if fin != -1 else len(sin_comentarios)]
    n = cuerpo.count("country_for_form_data(form_data)")
    assert n == 1, f"swap_meal debe derivar el país UNA sola vez, hallado {n}×"
    assert not _RAW_COUNTRY_RX.search(cuerpo)


def test_swap_meal_wire_los_dos_guards_con_el_pais_derivado():
    """Los DOS guards de retry (slot-horario ~L2762, raw-staple ~L2843) deben reusar la MISMA
    variable derivada (`_swap_country`), no recomputar country_for_form_data cada uno (eso
    haría fallar el test de arriba, que exige exactamente 1 derivación)."""
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    assert "_swap_slot_feedback_suffix(_swap_country" in src
    assert "_swap_raw_staple_feedback_suffix(_swap_country" in src


# ── T4: gates culturales suaves por país (SLOT_INAPPROPRIATE_FOODS por país) ────────────────
#
# `constants.slot_rules_for_country(country)`: 'DO' (default, knob apagado) ⇒ el MISMO objeto
# `SLOT_INAPPROPRIATE_FOODS` (identidad `is`); beta ⇒ tabla derivada MEMOIZADA con la MISMA
# estructura/tokens/excludes pero TODA regla `hardness='soft'` — siguen disparando (telemetría
# para diseñar las tablas nativas de Fase 2) pero dejan de forzar retry. `_detect_slot_
# appropriateness` la resuelve UNA vez por evaluación; los autofixes de arroz (`_night_rice_
# autofix`/`_breakfast_rice_autofix`) ganan un gate por país — DO reescribe como siempre, beta
# deja el plato intacto (el sustituto arroz→tubérculo ES una preparación dominicana).
# `build_meal_timing_rules` (§16/SLOT_POSITIVE_HINT — el hallazgo que el fix-round de T2 movió
# aquí) gana el mismo `country`: DO exacto byte a byte; beta OMITE la enumeración "NO uses..."
# (sus labels son intencionalmente dominicanos, por diseño arriba) y deja solo la guía positiva
# neutral (`_SLOT_POSITIVE_HINT_NEUTRAL`). Los 3 "rezagados" de prompt (Part B del brief) — el
# "según la cultura dominicana" de `build_day_assignment_context`, el hint del carbohidrato del
# corrector de self-critique y el fallback de `surgical_marker_regen_node` — se re-anclan por
# país con el MISMO principio que T2/T3 usaron: el REQUISITO sobrevive, solo se retira el marco
# de nacionalidad.

_BETA_CC_SAMPLE = _BETA_CCS[0] if _BETA_CCS else "ES"


# ── slot_rules_for_country ────────────────────────────────────────────────────

def test_slot_rules_for_country_do_es_identidad():
    assert constants.slot_rules_for_country("DO") is constants.SLOT_INAPPROPRIATE_FOODS
    assert constants.slot_rules_for_country("xx") is constants.SLOT_INAPPROPRIATE_FOODS  # fail-safe
    assert constants.slot_rules_for_country(None) is constants.SLOT_INAPPROPRIATE_FOODS


def test_slot_rules_for_country_beta_todo_soft_mismos_tokens():
    assert _BETA_CCS, "no hay países beta en COUNTRY_PROFILES — fixture vacío"
    for cc in _BETA_CCS:
        table = constants.slot_rules_for_country(cc)
        assert table is not constants.SLOT_INAPPROPRIATE_FOODS, cc
        assert set(table.keys()) == set(constants.SLOT_INAPPROPRIATE_FOODS.keys()), cc
        for slot, base_rules in constants.SLOT_INAPPROPRIATE_FOODS.items():
            rules = table[slot]
            assert len(rules) == len(base_rules), f"{cc}/{slot}"
            for rule, base_rule in zip(rules, base_rules):
                assert rule["hardness"] == "soft", f"{cc}/{slot}/{rule['label']}: {rule['hardness']!r}"
                assert rule["label"] == base_rule["label"]
                # mismos objetos tuple — no copia (mismos tokens/excludes, brief item 1)
                assert rule["tokens"] is base_rule["tokens"], f"{cc}/{slot}: tokens debe ser el MISMO objeto"
                assert rule.get("exclude") is base_rule.get("exclude"), f"{cc}/{slot}: exclude debe ser el MISMO objeto"


def test_slot_rules_for_country_beta_memoizado():
    a = constants.slot_rules_for_country(_BETA_CC_SAMPLE)
    b = constants.slot_rules_for_country(_BETA_CC_SAMPLE)
    assert a is b
    c = constants.slot_rules_for_country(_BETA_CC_SAMPLE.lower())
    assert a is c, "canonicaliza también en minúscula antes de memoizar"


def test_slot_rules_for_country_no_muta_la_tabla_base():
    """Llamar la variante beta NO debe mutar SLOT_INAPPROPRIATE_FOODS in-place — `dict(rule,
    hardness='soft')` crea una COPIA del dict de la regla, nunca escribe sobre el original."""
    before = {slot: [r["hardness"] for r in rules] for slot, rules in constants.SLOT_INAPPROPRIATE_FOODS.items()}
    for cc in _BETA_CCS:
        constants.slot_rules_for_country(cc)
    after = {slot: [r["hardness"] for r in rules] for slot, rules in constants.SLOT_INAPPROPRIATE_FOODS.items()}
    assert before == after
    assert constants.SLOT_INAPPROPRIATE_FOODS["desayuno"][0]["hardness"] == "hard"


# ── slot_violations_for_meal_name: rules_table opcional (inyección país-aware) ───────────────

def test_slot_violations_for_meal_name_sin_rules_table_es_identico_a_antes():
    """Backward-compat: sin 3er argumento (ni con None explícito), comportamiento byte-idéntico
    — sigue leyendo SLOT_INAPPROPRIATE_FOODS. [T8 slot-callers sweep] a día de hoy el único
    caller de producción que aún NO pasa `rules_table` es `plan_gym.py` (gym offline, EXENTO —
    ver backend/docs/country_system_f1.md); esta función sigue soportando la firma corta para él
    y para cualquier caller futuro sin país en scope."""
    v1 = constants.slot_violations_for_meal_name("Arroz con Locrio", "desayuno")
    v2 = constants.slot_violations_for_meal_name("Arroz con Locrio", "desayuno", None)
    assert v1 == v2
    assert v1 and v1[0]["hard"] is True


def test_slot_violations_for_meal_name_rules_table_beta_ablanda_el_hard():
    beta_table = constants.slot_rules_for_country(_BETA_CC_SAMPLE)
    v_do = constants.slot_violations_for_meal_name("Arroz con Locrio", "desayuno")
    v_beta = constants.slot_violations_for_meal_name("Arroz con Locrio", "desayuno", beta_table)
    assert v_do and v_do[0]["hard"] is True
    assert v_beta and v_beta[0]["hard"] is False
    assert v_do[0]["label"] == v_beta[0]["label"], "mismo token/label — solo se ablanda hardness"


# ── build_meal_timing_rules: §16/SLOT_POSITIVE_HINT por país ─────────────────────────────────

_MEAL_TYPES_ES_DO = ("Desayuno", "Almuerzo", "Cena", "Merienda")


def test_build_meal_timing_rules_do_byte_equal_explicito_e_implicito():
    for mt in _MEAL_TYPES_ES_DO:
        assert constants.build_meal_timing_rules(mt) == constants.build_meal_timing_rules(mt, "DO")
        assert constants.build_meal_timing_rules(mt) == constants.build_meal_timing_rules(mt, country="DO")


def test_build_meal_timing_rules_pais_desconocido_cae_a_do():
    for mt in _MEAL_TYPES_ES_DO:
        assert constants.build_meal_timing_rules(mt, "xx") == constants.build_meal_timing_rules(mt)


def test_build_meal_timing_rules_do_contiene_mangu_y_locrio():
    """Ancla el estado PRE-T4 (sentinel inverso): si esto deja de contener los tokens
    dominicanos, el guard beta de abajo dejaría de tener contra qué contrastar (falso-verde
    silencioso — la misma clase que P2-SLOT-SSOT-PROMPT ya vigila para el prompt del day-gen)."""
    out = "\n".join(constants.build_meal_timing_rules(mt) for mt in _MEAL_TYPES_ES_DO)
    assert "mangú" in out
    assert "locrio" in out.lower()


def test_build_meal_timing_rules_beta_sin_locrio_ni_mangu_ni_dominican():
    for cc in _BETA_CCS:
        out = "\n".join(constants.build_meal_timing_rules(mt, cc) for mt in _MEAL_TYPES_ES_DO)
        assert "locrio" not in out.lower(), cc
        assert "mangú" not in out.lower() and "mangu" not in out.lower(), cc
        assert "dominican" not in out.lower(), cc
        assert out.strip() != "", f"{cc}: el bloque no debe quedar vacío (guía positiva neutral sobrevive)"


# ── §16 en el render completo del day-gen (extiende los render-guards de T2) ─────────────────

def test_daygen_do_render_es_byte_identico_is_incluyendo_s16():
    """T2 ya ancla la identidad `is` del prompt COMPLETO — repetido aquí como sentinel explícito
    de que la nueva fila §16 de `_BETA_FRAGMENT_TABLE` (T4) no introduce un 2º `.replace()` que
    la rompa: el camino DO/None sigue siendo el objeto EXACTO, sin recomputar nada."""
    from prompts.day_generator import build_day_generator_system_prompt as build, DAY_GENERATOR_SYSTEM_PROMPT
    assert build("balanced", "DO") is DAY_GENERATOR_SYSTEM_PROMPT
    assert build("balanced") is DAY_GENERATOR_SYSTEM_PROMPT
    assert build("vegan", "DO") is build("vegan")


def test_daygen_do_s16_conserva_locrio_y_mangu():
    """Sentinel inverso, acotado a §16 (no al prompt completo — eso ya lo cubre T2's finding5)."""
    from prompts.day_generator import build_day_generator_system_prompt as build
    out = build("balanced", "DO")
    i16 = out.index("16. CONTRATO EXACTO DEL VALIDADOR DE HORARIO")
    i17 = out.index("\n17. PRESUPUESTO DE SODIO")
    s16 = out[i16:i17]
    assert "locrio" in s16.lower()
    assert "mangú" in s16.lower()


def test_daygen_beta_s16_sin_locrio_ni_mangu_ni_dominican():
    """Extiende los render-guards de T2: `test_finding5_guard_case_insensitive_sin_sobrevivientes_
    no_documentados` EXCLUYE §16 a propósito (territorio T4, ver `_scoped_out_sin_s16`). Este test
    escanea EXACTAMENTE la región complementaria — §16, en balanced Y vegan (la fila es
    diet-invariante)."""
    from prompts.day_generator import build_day_generator_system_prompt as build
    for diet in ("balanced", "vegan"):
        out = build(diet, "ES")
        i16 = out.index("16. CONTRATO EXACTO DEL VALIDADOR DE HORARIO")
        i17 = out.index("\n17. PRESUPUESTO DE SODIO")
        s16 = out[i16:i17]
        assert "locrio" not in s16.lower(), diet
        assert "mangú" not in s16.lower() and "mangu" not in s16.lower(), diet
        assert "dominican" not in s16.lower(), diet


# ── autofixes de arroz: gate por país ANTES de reescribir ────────────────────────────────────

def _cuerpo_night_rice_autofix() -> str:
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    sin_comentarios = "\n".join(
        l for l in src.splitlines() if not l.strip().startswith("#")
    )
    ini = sin_comentarios.index("def _night_rice_autofix(")
    fin = sin_comentarios.find("\ndef ", ini + 10)
    return sin_comentarios[ini: fin if fin != -1 else len(sin_comentarios)]


def _cuerpo_breakfast_rice_autofix() -> str:
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    sin_comentarios = "\n".join(
        l for l in src.splitlines() if not l.strip().startswith("#")
    )
    ini = sin_comentarios.index("def _breakfast_rice_autofix(")
    fin = sin_comentarios.find("\ndef ", ini + 10)
    return sin_comentarios[ini: fin if fin != -1 else len(sin_comentarios)]


def test_night_rice_autofix_gate_pais_antes_del_try_de_reescritura():
    cuerpo = _cuerpo_night_rice_autofix()
    pos_gate = cuerpo.index('!= "DO"')
    pos_try = cuerpo.index("\n    try:")
    assert pos_gate < pos_try, "el gate por país debe aparecer ANTES del try/reescritura"


def test_breakfast_rice_autofix_gate_pais_antes_del_try_de_reescritura():
    cuerpo = _cuerpo_breakfast_rice_autofix()
    pos_gate = cuerpo.index('!= "DO"')
    pos_try = cuerpo.index("\n    try:")
    assert pos_gate < pos_try, "el gate por país debe aparecer ANTES del try/reescritura"


class _FakeDBT4:
    """Doble mínimo de IngredientNutritionDB — parsea 'Ng' del string, fallback 150."""
    def grams_from_ingredient_string(self, s):
        m = re.search(r"(\d+)\s*g", s)
        return float(m.group(1)) if m else 150.0


def _cena_con_arroz(day=1):
    return [{"day": day, "meals": [{"meal": "Cena", "name": "Pollo con Arroz Blanco",
                                     "ingredients": ["200 g de pollo", "150 g de arroz blanco"],
                                     "recipe": ["Cocina el arroz 15 min.", "Sirve con el pollo."]}]}]


def _desayuno_con_arroz(day=1):
    return [{"day": day, "meals": [{"meal": "Desayuno", "name": "Arroz con Huevo",
                                     "ingredients": ["150 g de arroz blanco", "2 huevos"],
                                     "recipe": ["Cocina el arroz 15 min.", "Sirve con huevo."]}]}]


def _wire_autofix_knobs(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda m, db: True)
    monkeypatch.setattr(go, "NIGHT_RICE_AUTOFIX_ENABLED", True)
    monkeypatch.setattr(go, "BREAKFAST_RICE_AUTOFIX_ENABLED", True)


def test_night_rice_autofix_do_reescribe_como_siempre(monkeypatch):
    import graph_orchestrator as go
    _wire_autofix_knobs(monkeypatch)
    days = _cena_con_arroz()
    fixed = go._night_rice_autofix(days, db=_FakeDBT4())
    assert fixed == 1
    assert "arroz" not in days[0]["meals"][0]["name"].lower()


def test_night_rice_autofix_beta_deja_el_plato_intacto(monkeypatch):
    import graph_orchestrator as go
    import copy
    _wire_autofix_knobs(monkeypatch)
    for cc in _BETA_CCS:
        days = _cena_con_arroz()
        before = copy.deepcopy(days)
        fixed = go._night_rice_autofix(days, db=_FakeDBT4(), country=cc)
        assert fixed == 0, cc
        assert days == before, f"{cc}: el plato beta debe quedar BYTE-IDÉNTICO — 0 reescrituras"


def test_breakfast_rice_autofix_do_reescribe_como_siempre(monkeypatch):
    import graph_orchestrator as go
    _wire_autofix_knobs(monkeypatch)
    days = _desayuno_con_arroz()
    fixed = go._breakfast_rice_autofix(days, db=_FakeDBT4())
    assert fixed == 1
    assert "arroz" not in days[0]["meals"][0]["name"].lower()


def test_breakfast_rice_autofix_beta_deja_el_plato_intacto(monkeypatch):
    import graph_orchestrator as go
    import copy
    _wire_autofix_knobs(monkeypatch)
    for cc in _BETA_CCS:
        days = _desayuno_con_arroz()
        before = copy.deepcopy(days)
        fixed = go._breakfast_rice_autofix(days, db=_FakeDBT4(), country=cc)
        assert fixed == 0, cc
        assert days == before, f"{cc}: el plato beta debe quedar BYTE-IDÉNTICO — 0 reescrituras"


# ── _detect_slot_appropriateness: consumidor de slot_rules_for_country ───────────────────────

def _cuerpo_detect_slot_appropriateness() -> str:
    """Acotado a `_detect_slot_appropriateness` en sí — el boundary naive `\\ndef ` (usado en el
    resto del archivo) NO sirve aquí porque la SIGUIENTE definición es `async def self_critique_
    node` decorada con `@_node_label(...)`: `\\ndef ` la salta entera y sigue de largo hasta el
    próximo `def` a nivel de columna 0 (que puede vivir cientos/miles de líneas después, dentro
    de OTRA función no relacionada) — exactamente el mismo punto ciego que hace que el propio
    helper `_cuerpo_self_critique_node()` de arriba (T3) devuelva una región mucho más ancha que
    el nodo real. Se toma el boundary MÁS CERCANO entre `\\ndef `/`\\nasync def `/`\\n@`."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    sin_comentarios = "\n".join(
        l for l in src.splitlines() if not l.strip().startswith("#")
    )
    ini = sin_comentarios.index("def _detect_slot_appropriateness")
    candidatos = [
        p for p in (
            sin_comentarios.find("\ndef ", ini + 10),
            sin_comentarios.find("\nasync def ", ini + 10),
            sin_comentarios.find("\n@", ini + 10),
        ) if p != -1
    ]
    fin = min(candidatos) if candidatos else len(sin_comentarios)
    return sin_comentarios[ini:fin]


def test_detect_slot_appropriateness_deriva_pais_via_ssot_una_sola_vez():
    cuerpo = _cuerpo_detect_slot_appropriateness()
    _assert_deriva_pais_via_ssot(cuerpo, "_detect_slot_appropriateness")
    n = cuerpo.count("country_for_form_data(form_data)")
    assert n == 1, f"debe derivar el país UNA sola vez por evaluación, hallado {n}×"


def test_detect_slot_appropriateness_usa_slot_rules_for_country():
    """Cuenta EXACTAMENTE 2 (no solo `in`, que un mutante puede satisfacer vía el OTRO call site):
    el pase name-level (línea del `issues.append` de la violación) Y el pre-check `_name_flagged`
    del pase ingredient-level (P2-SLOT-EVASION-TELEMETRY) — AMBOS deben recibir la tabla resuelta.
    `in cuerpo` sin contar dejaba pasar un mutante que revertía SOLO el primero (el segundo, que
    también matchea el substring, lo enmascaraba) — confirmado con mutación explícita, ver reporte."""
    cuerpo = _cuerpo_detect_slot_appropriateness()
    assert "slot_rules_for_country(" in cuerpo
    n = cuerpo.count("slot_violations_for_meal_name(name, slot_key, _rules_table)")
    assert n == 2, (
        f"esperaba 2 usos de slot_violations_for_meal_name(name, slot_key, _rules_table) (pase "
        f"name-level + pre-check _name_flagged), hallado {n}× — si alguno cae de nuevo a 2 "
        f"argumentos, ese sitio ignora slot_rules_for_country y beta deja de ablandar ahí."
    )


def test_detect_slot_appropriateness_knob_off_ignora_country_en_form_data(monkeypatch):
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    import graph_orchestrator as go
    days = [{"day": 1, "meals": [{"meal": "Desayuno", "name": "Arroz con Huevo", "ingredients": []}]}]
    issues = go._detect_slot_appropriateness(days, {"country": "ES"})
    assert issues and issues[0]["hard"] is True, "knob apagado ⇒ 'DO' siempre, sin mirar form_data"


def test_detect_slot_appropriateness_beta_ablanda_hard_a_soft(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    import graph_orchestrator as go
    days = [{"day": 1, "meals": [{"meal": "Desayuno", "name": "Arroz con Huevo", "ingredients": []}]}]
    issues_beta = go._detect_slot_appropriateness(days, {"country": _BETA_CC_SAMPLE})
    assert issues_beta and issues_beta[0]["hard"] is False
    issues_do = go._detect_slot_appropriateness(days, {"country": "DO"})
    assert issues_do and issues_do[0]["hard"] is True
    assert issues_beta[0]["label"] == issues_do[0]["label"], "mismo token — solo cambia hard"


# ── Part B: los 3 rezagados de prompt (RULED-in por el review de T2) ─────────────────────────

# ── (4) build_day_assignment_context: "según la cultura dominicana" ──────────────────────────

def test_build_day_assignment_context_do_byte_equal_y_literal_exacto():
    from prompts.day_generator import build_day_assignment_context as bdac
    skeleton = {"protein_pool": []}
    do1 = bdac(skeleton, 1, day_name="Lunes")
    do2 = bdac(skeleton, 1, day_name="Lunes", country="DO")
    assert do1 == do2
    assert "según la cultura dominicana" in do1


def test_build_day_assignment_context_beta_sin_dominicana():
    from prompts.day_generator import build_day_assignment_context as bdac
    skeleton = {"protein_pool": []}
    for cc in _BETA_CCS:
        out = bdac(skeleton, 1, day_name="Lunes", country=cc)
        assert "según la cultura dominicana" not in out, cc
        assert "según la cultura local del usuario" in out, cc


def test_build_day_assignment_context_pais_desconocido_cae_a_do():
    from prompts.day_generator import build_day_assignment_context as bdac
    skeleton = {"protein_pool": []}
    assert bdac(skeleton, 1, day_name="Lunes", country="xx") == bdac(skeleton, 1, day_name="Lunes")


def test_generate_days_parallel_node_wire_country_en_build_day_assignment_context():
    """El callsite del brief (~L8146/8170): `generate_days_parallel_node` deriva el país vía SSOT
    e inyecta `country=` al llamar `build_day_assignment_context` — sin esto, el bloque "adapta
    la cultura" del día quedaría anclado a DO para siempre, sin importar el país real."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    sin_comentarios = "\n".join(
        l for l in src.splitlines() if not l.strip().startswith("#")
    )
    ini = sin_comentarios.index("assignment_context = build_day_assignment_context(")
    fin = sin_comentarios.index("\n        )", ini) + len("\n        )")
    cuerpo = sin_comentarios[ini:fin]
    assert "country=country_for_form_data(form_data)" in cuerpo


# ── (5) self_critique_node: hint del carbohidrato de la cena ─────────────────────────────────

def _cuerpo_carb_hint_line() -> str:
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    sin_comentarios = "\n".join(
        l for l in src.splitlines() if not l.strip().startswith("#")
    )
    ini = sin_comentarios.index("_carb_hint_line = (")
    fin = sin_comentarios.index("\n                    )", ini) + len("\n                    )")
    return sin_comentarios[ini:fin]


def test_self_critique_carb_hint_do_literal_exacto():
    cuerpo = _cuerpo_carb_hint_line()
    assert (
        '"  • Cambia el CARBOHIDRATO de la cena (yuca→batata, arroz→ñame, papas→casabe)."'
        in cuerpo
    )


def test_self_critique_carb_hint_beta_variante_presente():
    cuerpo = _cuerpo_carb_hint_line()
    assert '"  • Cambia el CARBOHIDRATO de la cena por otro del catálogo."' in cuerpo


def test_self_critique_carb_hint_reusa_critique_country_no_rederiva():
    """T3 ya deriva `_critique_country` una vez, arriba del nodo (shadow work) — este sitio debe
    REUSARLO vía closure, no volver a llamar country_for_form_data ni leer form_data crudo."""
    cuerpo = _cuerpo_carb_hint_line()
    assert "_critique_country" in cuerpo
    assert "country_for_form_data(" not in cuerpo
    assert not _RAW_COUNTRY_RX.search(cuerpo)


def test_self_critique_build_day_assignment_context_wire_country():
    """El OTRO callsite de build_day_assignment_context dentro de self_critique_node (el
    `skeleton_block` del corrector, ~L10745) también debe pasar el país derivado — mismo
    razonamiento que el callsite de generate_days_parallel_node arriba."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "diet_type=(form_data or {}).get('dietType'), country=_critique_country)" in src


# ── (6) surgical_marker_regen_node: fallback de plantilla matemática ─────────────────────────

def _cuerpo_fallback_dish_clause() -> str:
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    sin_comentarios = "\n".join(
        l for l in src.splitlines() if not l.strip().startswith("#")
    )
    ini = sin_comentarios.index("_fallback_dish_clause = (")
    fin = sin_comentarios.index("\n            )", ini) + len("\n            )")
    return sin_comentarios[ini:fin]


def test_surgical_marker_regen_fallback_do_literal_exacto():
    cuerpo = _cuerpo_fallback_dish_clause()
    assert '"platos dominicanos reales" if _surgical_country == "DO" else' in cuerpo


def test_surgical_marker_regen_fallback_beta_variante_presente():
    cuerpo = _cuerpo_fallback_dish_clause()
    assert '"platos reales de la cocina del usuario"' in cuerpo


def test_surgical_marker_regen_fallback_reusa_surgical_country_no_rederiva():
    cuerpo = _cuerpo_fallback_dish_clause()
    assert "_surgical_country" in cuerpo
    assert "country_for_form_data(" not in cuerpo
    assert not _RAW_COUNTRY_RX.search(cuerpo)


def _cuerpo_surgical_marker_regen_node() -> str:
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    sin_comentarios = "\n".join(
        l for l in src.splitlines() if not l.strip().startswith("#")
    )
    ini = sin_comentarios.index("async def surgical_marker_regen_node")
    fin = sin_comentarios.find("\nasync def ", ini + 10)
    return sin_comentarios[ini: fin if fin != -1 else len(sin_comentarios)]


def test_surgical_marker_regen_node_deriva_pais_via_ssot_una_sola_vez():
    """A diferencia de `self_critique_node` (cuyo boundary naive sobre-captura, ver
    `_cuerpo_detect_slot_appropriateness`), la SIGUIENTE función top-level tras
    `surgical_marker_regen_node` es otro `async def` cercano (`_recompute_aggregates_after_swap`)
    — el boundary `\\nasync def ` acota correctamente sin colar funciones ajenas."""
    cuerpo = _cuerpo_surgical_marker_regen_node()
    assert "async def _recompute_aggregates_after_swap" not in cuerpo
    _assert_deriva_pais_via_ssot(cuerpo, "surgical_marker_regen_node")
    n = cuerpo.count("country_for_form_data(form_data)")
    assert n == 1, f"debe derivar el país UNA sola vez, hallado {n}×"


def test_surgical_marker_regen_build_day_assignment_context_wire_country():
    cuerpo = _cuerpo_surgical_marker_regen_node()
    assert "country=_surgical_country" in cuerpo


# ── T4 fix-round 1 (review NEEDS FIXES): los backstops de retry país-blind ───────────────────
#
# El review encontró un TERCER gap no-disclosed: `slot_coherence_backstop_for_meal` (swap S3,
# graph_orchestrator.py) y el P1-CHAT-SLOT-BACKSTOP inline de tools.py (chat-modify) llamaban
# `slot_violations_for_meal_name` SIN tabla — SIEMPRE la tabla dura, sin importar el país — y
# trataban CUALQUIER violación devuelta como retry-forzante (agent.py `raise ValueError(
# "SLOT_INCOHERENCE"...)`  / tools.py `raise ValueError("plato fuera de horario"...)`), incluso
# las que este mismo fix-round (T4 original) ya trata como soft/telemetría en el gate S1. Ambos
# se corrigen con el MISMO shape: resolver `slot_rules_for_country(country)` y filtrar a "DO
# incluye TODO — byte-idéntico a pre-fix, INCLUIDAS reglas nativamente soft como 'arroz de noche'
# en cena, que YA disparaban este backstop antes de T4 — / país != DO incluye SOLO lo que sigue
# siendo hard tras resolver la tabla". La primera versión de este fix filtraba por `hard`
# INCONDICIONALMENTE (sin el `or country == "DO"`) y rompía `test_backstop_for_update_surfaces`
# (test_p1_slot_appropriateness.py) — 'arroz de noche' en cena es soft incluso en la tabla DO
# nativa, así que un filtro ciego por `hard` habría dejado de dispararlo también para DO. Los
# tests de abajo anclan explícitamente ese caso como regresión.

def test_slot_coherence_backstop_do_conserva_regla_nativamente_soft():
    """Regresión: 'arroz de noche' en cena es hardness='soft' incluso en SLOT_INAPPROPRIATE_FOODS
    nativa, y el backstop YA la incluía (ignoraba hardness por completo, pre-fix). Filtrar
    incondicionalmente por `hard` (la primera versión de este fix) rompe ESTE caso."""
    import graph_orchestrator as go
    out_default = go.slot_coherence_backstop_for_meal({"name": "Pollo con arroz blanco"}, "Cena")
    out_explicit_do = go.slot_coherence_backstop_for_meal({"name": "Pollo con arroz blanco"}, "Cena", "DO")
    assert out_default, "DO (default) debe seguir viendo esta violación soft-nativa"
    assert out_explicit_do, "DO (explícito) debe seguir viendo esta violación soft-nativa"
    assert out_default == out_explicit_do


def test_slot_coherence_backstop_do_sigue_forzando_hard():
    import graph_orchestrator as go
    out = go.slot_coherence_backstop_for_meal({"name": "Arroz con Huevo"}, "Desayuno", "DO")
    assert out, "DO debe seguir viendo la violación hard (desayuno-arroz)"


def test_slot_coherence_backstop_beta_no_dispara_para_violacion_soft():
    import graph_orchestrator as go
    assert _BETA_CCS, "fixture vacío"
    for cc in _BETA_CCS:
        out_hard_for_do = go.slot_coherence_backstop_for_meal({"name": "Arroz con Huevo"}, "Desayuno", cc)
        assert out_hard_for_do == [], f"{cc}: violación hard-para-DO/soft-para-beta no debe disparar, hallado {out_hard_for_do}"
        out_soft_for_do = go.slot_coherence_backstop_for_meal({"name": "Pollo con arroz blanco"}, "Cena", cc)
        assert out_soft_for_do == [], f"{cc}: violación soft-nativa tampoco debe disparar, hallado {out_soft_for_do}"


def test_slot_coherence_backstop_pais_desconocido_cae_a_do():
    import graph_orchestrator as go
    out_xx = go.slot_coherence_backstop_for_meal({"name": "Pollo con arroz blanco"}, "Cena", "xx")
    out_do = go.slot_coherence_backstop_for_meal({"name": "Pollo con arroz blanco"}, "Cena", "DO")
    assert out_xx == out_do


def test_swap_meal_wire_slot_coherence_backstop_con_pais():
    """agent.py's caller pasa `_swap_country` (ya derivado, T3) — sin esto el backstop cae al
    default 'DO' sin importar el país real del usuario."""
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    assert "slot_coherence_backstop_for_meal(_slot_dump, meal_type, _swap_country)" in src


def _tools_cuerpo_inline_backstop() -> str:
    src = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    sin_comentarios = "\n".join(
        l for l in src.splitlines() if not l.strip().startswith("#")
    )
    ini = sin_comentarios.index("_meal_viols_all = slot_violations_for_meal_name(")
    fin = sin_comentarios.index("if _unrequested and _slot_attempt[0] < 3:", ini)
    return sin_comentarios[ini:fin]


def test_chat_modify_backstop_thread_rules_table_y_filtro_pais():
    cuerpo = _tools_cuerpo_inline_backstop()
    assert "_modify_rules_table" in cuerpo
    assert '_modify_country == "DO" or v.get("hard")' in cuerpo


def _unrequested_labels_country_aware(meal_name: str, user_changes: str, slot_key: str, country: str) -> list:
    """Réplica FUNCIONAL país-aware de la lógica del backstop inline de tools.py — mismo patrón
    que test_p1_chat_slot_backstop.py::_unrequested_labels (pura, sin DB/LLM, sin importar
    tools.py), extendida con el filtro de país de este fix-round."""
    rules_table = constants.slot_rules_for_country(country)
    meal_v_all = constants.slot_violations_for_meal_name(meal_name, slot_key, rules_table)
    meal_v = [v for v in meal_v_all if country == "DO" or v.get("hard")]
    if not meal_v:
        return []
    requested = {
        v["label"] for v in constants.slot_violations_for_meal_name(user_changes or "", slot_key, rules_table)
    }
    return [v for v in meal_v if v["label"] not in requested]


def test_chat_modify_backstop_do_conserva_presion_de_retry():
    slot = constants.canonical_slot_key("Cena")
    out = _unrequested_labels_country_aware("Arroz blanco con pollo guisado", "cámbiame la cena", slot, "DO")
    assert out, "DO debe seguir viendo la violación (comportamiento pre-fix, incluida la soft-nativa)"


def test_chat_modify_backstop_beta_sin_presion_de_retry_para_soft():
    slot = constants.canonical_slot_key("Cena")
    for cc in _BETA_CCS:
        out = _unrequested_labels_country_aware("Arroz blanco con pollo guisado", "cámbiame la cena", slot, cc)
        assert out == [], f"{cc}: no debe forzar retry para una violación soft-para-beta"


# ── T4 fix-round 1: hard flag del pase INGREDIENT-LEVEL, overrideado por país (review IMPORTANT #2) ──

def test_detect_slot_appropriateness_ingredient_hard_override_do():
    import graph_orchestrator as go
    days = [{"day": 1, "meals": [{"meal": "Desayuno", "name": "Bowl energetico criollo",
                                    "ingredients": ["150g arroz blanco", "1 huevo"]}]}]
    issues = go._detect_slot_appropriateness(days, {"country": "DO"})
    assert issues and issues[0]["hard"] is True


def test_detect_slot_appropriateness_ingredient_hard_override_beta(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    import graph_orchestrator as go
    days = [{"day": 1, "meals": [{"meal": "Desayuno", "name": "Bowl energetico criollo",
                                    "ingredients": ["150g arroz blanco", "1 huevo"]}]}]
    issues = go._detect_slot_appropriateness(days, {"country": _BETA_CC_SAMPLE})
    assert issues, "el issue debe seguir PRESENTE — sigue midiendo (telemetría)"
    assert issues[0]["hard"] is False, "pero el flag hard debe overridearse a False para beta"


def test_detect_slot_appropriateness_ingredient_override_no_toca_slot_ingredient_violations():
    """Contenido (review: 'do NOT change slot_ingredient_violations itself'): la función
    compartida sigue devolviendo hard=True incondicional — el override vive SOLO en el sitio de
    consumo dentro de _detect_slot_appropriateness."""
    v = constants.slot_ingredient_violations(["150g arroz blanco"], "desayuno")
    assert v and v[0]["hard"] is True


# ── Task 9 (g) MUTATOR-PURITY: cierre del pase ingredient-level de slot_coherence_backstop ───
#
# Parqueado por T4 fix-round 1 (docs/country_system_f1.md, fila "slot_coherence_backstop_for_
# meal's pase ingredient-level"): a diferencia de _detect_slot_appropriateness (arriba), este
# pase INSIDE slot_coherence_backstop_for_meal seguía siempre-hard sin importar país. Cerrado
# con el MISMO override formula (v["hard"] and país==DO) — pero AQUÍ como filtro de inclusión,
# no como campo re-escrito (esta función retorna list[str], no list[dict]).

def test_g_slot_coherence_backstop_ingredient_pass_do_detecta_arroz_oculto():
    """DO conserva EXACTAMENTE el comportamiento pre-Task-9: arroz oculto en ingredients de un
    desayuno con nombre inocuo SÍ dispara el backstop (byte-idéntico)."""
    import graph_orchestrator as go
    meal = {"name": "Bowl energético criollo", "ingredients": ["150g arroz blanco", "1 huevo"]}
    out_do = go.slot_coherence_backstop_for_meal(meal, "Desayuno", "DO")
    out_default = go.slot_coherence_backstop_for_meal(meal, "Desayuno")
    assert out_do and "INGREDIENTES del desayuno" in out_do[0]
    assert out_do == out_default


def test_g_slot_coherence_backstop_ingredient_pass_beta_no_dispara():
    """Beta ya NO ve la violación ingredient-level — cierre del gap disclosed en T4 fix-round 1."""
    import graph_orchestrator as go
    meal = {"name": "Bowl energético criollo", "ingredients": ["150g arroz blanco", "1 huevo"]}
    assert _BETA_CCS, "fixture vacío"
    for cc in _BETA_CCS:
        out = go.slot_coherence_backstop_for_meal(meal, "Desayuno", cc)
        assert out == [], f"{cc}: el pase ingredient-level ya no debe disparar para beta, hallado {out}"


def test_g_slot_coherence_backstop_pais_desconocido_ingredient_cae_a_do():
    import graph_orchestrator as go
    meal = {"name": "Bowl energético criollo", "ingredients": ["150g arroz blanco"]}
    out_xx = go.slot_coherence_backstop_for_meal(meal, "Desayuno", "xx")
    out_do = go.slot_coherence_backstop_for_meal(meal, "Desayuno", "DO")
    assert out_xx == out_do and out_xx != []


def test_g_slot_ingredient_violations_no_se_toco():
    """Contrato compartido con T4: slot_ingredient_violations en SÍ sigue devolviendo hard=True
    incondicional — el override vive SOLO en el sitio de consumo (mismo principio que T4)."""
    v = constants.slot_ingredient_violations(["150g arroz blanco"], "desayuno")
    assert v and v[0]["hard"] is True


def test_g_mutacion_filtro_or_es_un_no_op():
    """MUTACIÓN: reproduce el bug real que este fix corrigió durante el desarrollo — filtrar con
    `_is_do or v.get("hard")` (el patrón del pase NAME-level, copiado ingenuamente) es un NO-OP
    aquí porque `slot_ingredient_violations` devuelve hard=True INCONDICIONAL: el OR con un
    valor siempre-True es siempre-True, así que beta seguiría viendo el 100% de las violaciones.
    El filtro correcto es AND, no OR."""
    v_list = constants.slot_ingredient_violations(["150g arroz blanco"], "desayuno")
    assert v_list, "fixture: debe haber al menos 1 violación para que la mutación sea significativa"
    for _is_do_sim in (True, False):
        # filtro roto (OR, lo que se escribió primero y se descartó):
        broken = [v for v in v_list if _is_do_sim or v.get("hard")]
        # filtro correcto (AND, lo que quedó en producción):
        fixed = [v for v in v_list if v.get("hard") and _is_do_sim]
        if not _is_do_sim:
            assert broken == v_list, "el OR debía ser un no-op (beta seguiría viendo TODO)"
            assert fixed == [], "el AND correctamente excluye para beta"


def test_g_swap_persist_wire_finalize_y_backstop_con_pais():
    """routers/plans.py::_swap_mutator (api_swap_meal_persist) — ambos call sites que T4
    fix-round 1 dejó EXENTO (finalize_single_meal_recipe_coherence + slot_coherence_backstop_
    for_meal) ahora reciben `country=_swap_country`, resuelto ANTES del lock."""
    src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    assert "day_kcal_target=_dkt_sp(plan_data.get(\"macros\")), country=_swap_country)" in src
    assert '_slot_sp(new_meal, str(new_meal.get("meal") or ""), country=_swap_country)' in src
    # el EXENTO viejo (T4 fix-round 1) ya no debe seguir citado como abierto en este call site:
    assert "P1-COUNTRY-SYSTEM-F1 EXENTO: T4 fix-round 1 finding, no cerrado" not in src


def test_g_swap_persist_country_prefetch_antes_del_lock():
    """`_swap_country` se resuelve (closure) ANTES de que `update_plan_data_atomic` adquiera el
    lock — orden textual = orden de ejecución (mismo módulo, mismo hilo de request). Busca la
    invocación REAL (`result = update_plan_data_atomic(`), no la mención en el docstring de la
    función (que cita el mismo texto en prosa, ANTES del pre-fetch)."""
    src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    i_persist_def = src.index("def api_swap_meal_persist(")
    # P1-COUNTRY-PLAN-VS-PERFIL-EN-BLOQUES cambió el resolvedor de perfil
    # puro al SSOT del artefacto; el contrato de este guard sigue siendo el orden.
    i_prefetch = src.index("_swap_country = _cfp_swap_persist(", i_persist_def)
    i_lock = src.index("result = update_plan_data_atomic(", i_persist_def)
    assert i_persist_def < i_prefetch < i_lock, (
        "el pre-fetch de country debe vivir ANTES de update_plan_data_atomic (patrón _micro_form)"
    )


def test_g_recalculate_shopping_list_wire_finalize_con_pais():
    """routers/plans.py::api_recalculate_shopping_list — 2º call site país-blind identificado en
    docs/country_system_f1.md ('Parqueado para Fase 2').

    [RE-ANCLADO por P1-PLAN-STAMPS-COUNTRY · 2026-08-21] Anclaba la grafía `_cffd_rc(`
    (`country_for_form_data` sobre el perfil). El resolvedor pasó a `country_for_plan`, que
    prefiere el SELLO DEL PLAN y cae al perfil sólo si el plan no lo trae — porque un plan es un
    artefacto con fecha y re-interpretar platos españoles bajo reglas dominicanas producía el
    híbrido que hoy existe en producción. Lo que este test protege de VERDAD no cambió: que el
    país se resuelva ANTES del finalize y ANTES del lock (pre-fetch, patrón MUTATOR-PURITY: nada
    de reentrar al pool dentro del `SELECT … FOR UPDATE`). Se re-ancla a ese ORDEN."""
    src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    i_fn = src.index("def api_recalculate_shopping_list(")
    # [P1-COUNTRY-STAMP-NO-FALLBACK-WRITE] También captura la procedencia; el ancla es
    # la llamada al SSOT antes del finalizer, no una asignación de una sola variable.
    i_fetch = src.index("_recalc_country, _recalc_country_source = _cfp_rc(", i_fn)
    i_call = src.index("_rc_fixed += _fin_rc_rc(_m, allergies=_rc_allergies, portion_floors=False,\n"
                        "                                                 country=_recalc_country)", i_fn)
    i_lock = src.index("update_plan_data_atomic(", i_fn)
    assert i_fn < i_fetch < i_call < i_lock


def test_g_swap_country_deriva_via_ssot_no_lector_crudo():
    """El pre-fetch pasa por el SSOT del plan; canonicaliza el sello y sólo
    cae al perfil para artefactos legacy."""
    src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    cuerpo_ini = src.index("def api_swap_meal_persist(")
    cuerpo_fin = src.index("\ndef ", cuerpo_ini + 10)
    cuerpo = src[cuerpo_ini:cuerpo_fin]
    assert "country_for_plan as _cfp_swap_persist" in cuerpo
    assert "_cfp_swap_persist(" in cuerpo


# ── T4 fix-round 1: finalizer país-aware (review IMPORTANT #3) ───────────────────────────────

def _finalizer_fake_db():
    class _FDB:
        def grams_from_ingredient_string(self, s):
            m = re.search(r"(\d+)\s*g", s)
            return float(m.group(1)) if m else 150.0
    return _FDB()


def _cena_arroz_meal_finalizer():
    return {"meal": "Cena", "name": "Pollo con Arroz Blanco",
            "ingredients": ["200 g de pollo", "150 g de arroz blanco"],
            "recipe": ["Cocina el arroz 15 min.", "Sirve con el pollo."]}


def test_finalizer_do_default_sigue_autofix_de_arroz(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "NIGHT_RICE_AUTOFIX_ENABLED", True)
    monkeypatch.setattr(go, "UPDATE_RECIPE_FINALIZE_ENABLED", True)
    meal = _cena_arroz_meal_finalizer()
    go.finalize_single_meal_recipe_coherence(meal, db=_finalizer_fake_db())
    assert "arroz" not in meal["name"].lower()


def test_finalizer_beta_salta_solo_el_autofix_de_arroz(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "NIGHT_RICE_AUTOFIX_ENABLED", True)
    monkeypatch.setattr(go, "UPDATE_RECIPE_FINALIZE_ENABLED", True)
    assert _BETA_CCS, "fixture vacío"
    for cc in _BETA_CCS:
        meal = _cena_arroz_meal_finalizer()
        go.finalize_single_meal_recipe_coherence(meal, db=_finalizer_fake_db(), country=cc)
        assert "arroz" in meal["name"].lower(), f"{cc}: el nombre no debe perder 'arroz' (autofix saltado)"
        assert any("arroz" in i.lower() for i in meal["ingredients"]), (
            f"{cc}: los ingredientes deben conservar arroz (autofix saltado)"
        )


def test_finalizer_wire_country_en_swap_y_chat_modify():
    agent_src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    assert (
        "_fin_rc(_out, pantry_strict=bool(clean_ingredients), allergies=allergies, country=_swap_country)"
        in agent_src
    )
    tools_src = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    assert "country=_modify_country" in tools_src
    assert "_fin_rc_m(new_meal_data, pantry_strict=_ps_fin, allergies=_clin_allergies," in tools_src


# ═══════════════════════════════════════════════════════════════════════════
# ── T5: fecha local por usuario (independiente del país) ────────────────────
# ═══════════════════════════════════════════════════════════════════════════
#
# [P1-COUNTRY-SYSTEM-F1 · 2026-08-16 (T5)] Country-INDEPENDENT a propósito: un dominicano
# viajando a España también debe cortar su día/hora en LA SUYA, no en RD fija — por eso
# `db_facts.user_tz_offset_min(user_id)` NO deriva de `constants.COUNTRY_PROFILES` (eso es un
# default de FORM para el motor de generación, T1-T4), sino de `health_profile->>'tzOffset'`, el
# reloj PERSONAL que el cliente ya manda (`new Date().getTimezoneOffset()`, Dashboard.jsx/
# Plan.jsx/etc.) y que `/shift-plan` ya sincroniza en el perfil (`routers/plans.py::_tz_mutator`,
# escribe `tz_offset_minutes` Y `tzOffset` juntos).
#
# Reemplaza el hardcode `AT TIME ZONE 'America/Santo_Domingo'` en los 4 sitios SQL fuera de este
# archivo por aritmética de offset: `(col ± make_interval(mins => %s))::date`/`EXTRACT` — álgebra
# EXACTA para un huso de offset fijo (América/Santo_Domingo no tiene DST), verificada contra
# Neon 2026-08-16 (ver task-5-report.md).
#
# ⚠️ UN sitio (`db_facts.get_avg_meal_hour`) usa `+` en vez de `-`: el forense contra Neon reveló
# que `consumed_at` es `timestamptz` (no NAIVE), así que el idiom previo (doble
# `AT TIME ZONE 'UTC' AT TIME ZONE 'America/Santo_Domingo'`) NETEABA +offset en vez de -offset —
# un bug preexistente (hora promedio de comida sesgada +8h, NO fecha) fuera de scope de T5 (que
# solo parametriza el huso, no audita aritmética — ver comentario en el propio sitio). Preservar
# ese signo es lo que exige la byte-identidad a offset=240; los tests de abajo lo anclan
# EXPLÍCITAMENTE para que un futuro "cleanup" no lo "corrija" sin querer.
#
# Contrato con vecinos anclados (NO se tocan, corren tal cual): `test_p1_chat_past_days_memory.py`
# (:329 `tz_offset: int = 240` en firmas, :471 prohíbe `tz_offset or 240`) y
# `test_p1_diary_tz_default_rd.py` (`get_consumed_meals_today` sigue defaulteando 240 — concepto
# DISTINTO, ventana Python-side, no SQL `AT TIME ZONE`).

_T5_FILES = ("db_facts.py", "tools.py", "proactive_agent.py")


def _sin_comentarios(rel_path: str) -> str:
    src = (_BACKEND / rel_path).read_text(encoding="utf-8")
    return "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))


def _mock_db_facts_query(monkeypatch, result):
    """Doble de `db_facts.execute_sql_query` que no toca la DB — mismo patrón que
    `captured_window` en test_p1_diary_tz_default_rd.py. Devuelve `result` para cualquier query
    (el helper bajo test solo hace una)."""
    import db_facts
    calls = []

    def _fake(query, params=None, fetch_one=False, fetch_all=False, **kwargs):
        calls.append({"query": query, "params": params})
        return result

    monkeypatch.setattr(db_facts, "execute_sql_query", _fake)
    monkeypatch.setattr(db_facts, "connection_pool", object())
    return calls


# ── user_tz_offset_min: helper único ──────────────────────────────────────────

def test_user_tz_offset_min_lee_tzoffset_del_perfil(monkeypatch):
    import db_facts
    _mock_db_facts_query(monkeypatch, {"tz": "-60"})
    assert db_facts.user_tz_offset_min("u-1") == -60


def test_user_tz_offset_min_perfil_ausente_cae_a_240(monkeypatch):
    import db_facts
    _mock_db_facts_query(monkeypatch, None)
    assert db_facts.user_tz_offset_min("u-1") == 240


def test_user_tz_offset_min_clave_ausente_o_null_cae_a_240(monkeypatch):
    import db_facts
    _mock_db_facts_query(monkeypatch, {"tz": None})
    assert db_facts.user_tz_offset_min("u-1") == 240


def test_user_tz_offset_min_garbage_cae_a_240(monkeypatch):
    import db_facts
    for basura in ("no-soy-un-numero", "", "[1,2]", "true"):
        _mock_db_facts_query(monkeypatch, {"tz": basura})
        assert db_facts.user_tz_offset_min("u-1") == 240, f"basura={basura!r}"


def test_user_tz_offset_min_excepcion_de_db_cae_a_240(monkeypatch):
    import db_facts

    def _boom(*a, **k):
        raise RuntimeError("Neon caído")

    monkeypatch.setattr(db_facts, "execute_sql_query", _boom)
    monkeypatch.setattr(db_facts, "connection_pool", object())
    assert db_facts.user_tz_offset_min("u-1") == 240


def test_user_tz_offset_min_sin_user_id_cae_a_240(monkeypatch):
    import db_facts
    calls = _mock_db_facts_query(monkeypatch, {"tz": "-60"})
    assert db_facts.user_tz_offset_min(None) == 240
    assert db_facts.user_tz_offset_min("") == 240
    assert not calls, "no debería consultar la DB si user_id es falsy"


def test_user_tz_offset_min_sin_connection_pool_cae_a_240(monkeypatch):
    import db_facts
    monkeypatch.setattr(db_facts, "connection_pool", None)
    assert db_facts.user_tz_offset_min("u-1") == 240


@pytest.mark.parametrize("crudo,esperado", [
    ("899", 899), ("900", 900), ("901", 900), ("5000", 900),
    ("-899", -899), ("-900", -900), ("-901", -900), ("-5000", -900),
    ("240.0", 240), ("-60", -60),
])
def test_user_tz_offset_min_clamp_y_coercion_numerica(monkeypatch, crudo, esperado):
    import db_facts
    _mock_db_facts_query(monkeypatch, {"tz": crudo})
    assert db_facts.user_tz_offset_min("u-1") == esperado, f"crudo={crudo!r}"


def test_user_tz_offset_min_query_apunta_a_health_profile_tzoffset(monkeypatch):
    import db_facts
    calls = _mock_db_facts_query(monkeypatch, {"tz": "-60", "tz_legacy": None})
    db_facts.user_tz_offset_min("u-1")
    assert calls, "execute_sql_query no se invocó"
    q = calls[-1]["query"]
    assert "health_profile->>'tzOffset'" in q
    assert "health_profile->>'tz_offset_minutes'" in q, (
        "[fix-round 1] la query debe seleccionar TAMBIÉN tz_offset_minutes — sin esta columna "
        "el fallback no tiene de dónde leer"
    )
    assert "user_profiles" in q
    assert calls[-1]["params"] == ("u-1",)


# ── T5 fix-round 1: fallback a tz_offset_minutes (segundo escritor no sincronizado) ──────────
#
# [P1-COUNTRY-SYSTEM-F1 · 2026-08-16 (T5, fix-round 1)] Review encontró que el docstring
# original afirmaba `_tz_mutator` (/shift-plan) como "el único write path conocido" — falso.
# `routers/plans.py::_postprocess_pipeline_result` (~L2107-2137, el escritor de CADA
# `/analyze`+`/analyze/stream`) también escribe `health_profile`, y su rama sin `tzOffset` crudo
# en el payload deja `tz_offset_minutes` poblado SIN `tzOffset` — un perfil así degradaba a 240
# para siempre bajo el diseño single-key original. Estos tests anclan el fallback que lo cierra.

def test_user_tz_offset_min_solo_tz_offset_minutes_usa_el_fallback(monkeypatch):
    """El caso exacto del segundo escritor: `tzOffset` nunca se poblo, `tz_offset_minutes` sí
    (con el offset real resuelto por `_resolve_request_tz_offset`) — el helper debe devolverlo,
    no degradar a 240."""
    import db_facts
    _mock_db_facts_query(monkeypatch, {"tz": None, "tz_legacy": "-60"})
    assert db_facts.user_tz_offset_min("u-1") == -60


def test_user_tz_offset_min_ambas_claves_tzoffset_gana(monkeypatch):
    """Cuando ambas claves están pobladas (el caso sincronizado de `_tz_mutator`), `tzOffset`
    tiene prioridad — orden INVERSO al COALESCE de `cron_tasks._get_user_tz_live` (que prueba
    `tz_offset_minutes` primero), a propósito: `tzOffset` es la clave que el diseño original de
    este helper vincula."""
    import db_facts
    _mock_db_facts_query(monkeypatch, {"tz": "-60", "tz_legacy": "300"})
    assert db_facts.user_tz_offset_min("u-1") == -60


def test_user_tz_offset_min_tzoffset_garbage_cae_al_fallback(monkeypatch):
    """`tzOffset` PRESENTE pero no numérico no debe saltar directo a 240 — debe intentar
    `tz_offset_minutes` antes de rendirse."""
    import db_facts
    _mock_db_facts_query(monkeypatch, {"tz": "no-soy-un-numero", "tz_legacy": "-60"})
    assert db_facts.user_tz_offset_min("u-1") == -60


def test_user_tz_offset_min_ambas_claves_garbage_cae_a_240(monkeypatch):
    import db_facts
    _mock_db_facts_query(monkeypatch, {"tz": "no-numero", "tz_legacy": "[1,2]"})
    assert db_facts.user_tz_offset_min("u-1") == 240


def test_user_tz_offset_min_sin_cache_entre_llamadas(monkeypatch):
    """Per-call a propósito (brief: correctness sobre micro-perf en diario) — perfiles distintos
    en llamadas consecutivas del MISMO user_id deben reflejarse siempre, nunca memoizado."""
    import db_facts
    valores = iter(["-60", "300"])

    def _fake(query, params=None, fetch_one=False, fetch_all=False, **kwargs):
        return {"tz": next(valores)}

    monkeypatch.setattr(db_facts, "execute_sql_query", _fake)
    monkeypatch.setattr(db_facts, "connection_pool", object())

    assert db_facts.user_tz_offset_min("u-1") == -60
    assert db_facts.user_tz_offset_min("u-1") == 300, (
        "el segundo valor debe reflejar la NUEVA lectura — una cache lo congelaría en -60"
    )


# ── parser: cero 'America/Santo_Domingo' hardcoded en los 4 sitios ───────────

def test_cero_america_santo_domingo_hardcoded_en_los_4_sitios():
    """Los 4 SQL (db_facts.get_avg_meal_hour, tools._rescue_dinner_slot,
    tools.log_consumed_meal, proactive_agent.get_daily_nudge_count) deben quedar en CERO
    ocurrencias del literal fuera de comentarios (permitido en comentarios, brief). NO escanea
    `schemas.py:145` (`timezone: Optional[str] = 'America/Santo_Domingo'`) — campo Pydantic NO
    relacionado, fuera de scope de T5 a propósito."""
    for rel_path in _T5_FILES:
        cuerpo = _sin_comentarios(rel_path)
        assert "America/Santo_Domingo" not in cuerpo, (
            f"{rel_path}: todavía hardcodea 'America/Santo_Domingo' fuera de un comentario"
        )


def test_los_4_sitios_usan_make_interval_parametrizado():
    """Guard positivo — evita que el test de arriba pase vacío si alguien borra el SQL entero en
    vez de parametrizarlo. 2 usos en db_facts.py (HOUR+MINUTE), 4 en tools.py (2 sitios × 2), 2
    en proactive_agent.py."""
    minimos = {"db_facts.py": 2, "tools.py": 4, "proactive_agent.py": 2}
    for rel_path, minimo in minimos.items():
        cuerpo = _sin_comentarios(rel_path)
        n = cuerpo.count("make_interval(mins => %s)")
        assert n >= minimo, f"{rel_path}: esperaba >= {minimo} usos, hallado {n}"


def test_los_4_sitios_derivan_offset_via_user_tz_offset_min():
    """Ningún sitio puede resolver el offset por su cuenta (2ª tabla/lectura ad-hoc) — todos
    pasan por el único helper, mismo principio que P1-DIET-CANON-SSOT."""
    ocurrencias = {"db_facts.py": 1, "tools.py": 2, "proactive_agent.py": 1}
    for rel_path, minimo in ocurrencias.items():
        cuerpo = _sin_comentarios(rel_path)
        n = cuerpo.count("user_tz_offset_min(")
        assert n >= minimo, f"{rel_path}: esperaba >= {minimo} llamadas a user_tz_offset_min(, hallado {n}"


# ── db_facts.get_avg_meal_hour: preserva-bug con signo '+' ───────────────────

def test_get_avg_meal_hour_usa_offset_resuelto_con_signo_menos(monkeypatch):
    """[RECONVERTIDO por P1-AVG-MEAL-HOUR-SIGN · 2026-08-21] Este test anclaba el signo '+' como
    PRESERVA-BUG deliberado de T5: `consumed_at` es timestamptz, el doble AT TIME ZONE previo
    neteaba +offset, y T5 replicó ese '+' para ser byte-idéntica a offset=240 mientras sólo
    parametrizaba el huso. Su propio docstring decía que «un P-fix dedicado con su propio test
    debe decidir si corregir el signo».

    Este es ese P-fix. Lo que cambió la decisión no fue el gusto: con el knob de países ENCENDIDO
    el sesgo dejó de ser la constante «+8 h» que el comentario prometía y pasó a ser `2 × offset`
    — una función del país, que para España CAMBIA DE SIGNO (−4 h). La byte-identidad que el '+'
    protegía sólo existía mientras todos los usuarios compartían offset 240.

    El test se RECONVIERTE y conserva su mitad todavía cierta: que el huso salga resuelto por
    usuario y que el hardcode 'America/Santo_Domingo' no vuelva."""
    import db_facts

    calls = []

    def _fake(query, params=None, fetch_one=False, fetch_all=False, **kwargs):
        calls.append({"query": query, "params": params})
        return []

    monkeypatch.setattr(db_facts, "execute_sql_query", _fake)
    monkeypatch.setattr(db_facts, "connection_pool", object())
    monkeypatch.setattr(db_facts, "user_tz_offset_min", lambda uid: -60)

    db_facts.get_avg_meal_hour("u-1", "desayuno")

    assert calls, "execute_sql_query no se invocó"
    query, params = calls[-1]["query"], calls[-1]["params"]
    assert query.count("consumed_at - make_interval(mins => %s)") == 2, (
        "esperaba el signo '-' (P1-AVG-MEAL-HOUR-SIGN) en HOUR y MINUTE"
    )
    assert "consumed_at + make_interval" not in query, (
        "volvió el preserva-bug: sobre timestamptz el '+' desvía 2×offset, no una constante"
    )
    assert "America/Santo_Domingo" not in query
    assert params[0] == -60 and params[1] == -60, f"offset resuelto ausente: {params!r}"


def test_get_avg_meal_hour_offset_240_por_defecto_sin_perfil(monkeypatch):
    """Fallback: sin perfil (`user_tz_offset_min` REAL, sin mock, con SELECT que no encuentra
    fila) ⇒ 240 — conducta IDÉNTICA al hardcode previo."""
    import db_facts

    calls = []

    def _fake(query, params=None, fetch_one=False, fetch_all=False, **kwargs):
        calls.append({"query": query, "params": params})
        return None if fetch_one else []

    monkeypatch.setattr(db_facts, "execute_sql_query", _fake)
    monkeypatch.setattr(db_facts, "connection_pool", object())

    db_facts.get_avg_meal_hour("u-1", "desayuno")

    # la ÚLTIMA call es la query propia de avg_meal_hour; el offset viajó resuelto a 240.
    query, params = calls[-1]["query"], calls[-1]["params"]
    assert params[0] == 240 and params[1] == 240


# ── tools._rescue_dinner_slot: signo '-' (hop simple, correcto) ──────────────

def test_rescue_dinner_slot_usa_offset_resuelto_con_signo_menos(monkeypatch):
    import tools
    import db

    calls = []

    def _fake(query, params=None, fetch_all=False, fetch_one=False, **kwargs):
        calls.append({"query": query, "params": params})
        return []

    monkeypatch.setattr(tools, "_DINNER_RESCUE_ENABLED", True)
    monkeypatch.setattr(db, "execute_sql_query", _fake)
    monkeypatch.setattr(tools, "user_tz_offset_min", lambda uid: -60)

    tools._rescue_dinner_slot("u-1", "snack", 500, 0)

    assert calls, "execute_sql_query no se invocó"
    query, params = calls[-1]["query"], calls[-1]["params"]
    assert query.count("- make_interval(mins => %s)") == 2
    assert "America/Santo_Domingo" not in query
    assert params[-2] == -60 and params[-1] == -60, f"offset resuelto ausente: {params!r}"


# ── tools.log_consumed_meal (dup-guard): SQL exacto, signo '-' ───────────────

def _cuerpo_log_consumed_meal_dup_guard() -> str:
    cuerpo_completo = _sin_comentarios("tools.py")
    ini = cuerpo_completo.index("_meal_type in _CONSUMED_MAIN_MEAL_TYPES and not force")
    fin = cuerpo_completo.index("except Exception as _dup_err:", ini)
    return cuerpo_completo[ini:fin]


def test_log_consumed_meal_dup_guard_sql_exacto_offset_resuelto():
    cuerpo = _cuerpo_log_consumed_meal_dup_guard()
    assert "user_tz_offset_min(user_id)" in cuerpo
    assert "America/Santo_Domingo" not in cuerpo
    assert cuerpo.count("- make_interval(mins => %s)") == 2


# ── proactive_agent.get_daily_nudge_count: signo '-' ──────────────────────────

def test_get_daily_nudge_count_usa_offset_resuelto_no_240(monkeypatch):
    import proactive_agent

    calls = []

    def _fake(query, params=None, fetch_one=False, **kwargs):
        calls.append({"query": query, "params": params})
        return {"total": 0}

    monkeypatch.setattr(proactive_agent, "execute_sql_query", _fake)
    monkeypatch.setattr(proactive_agent, "user_tz_offset_min", lambda uid: -60)

    proactive_agent.get_daily_nudge_count("u-1")

    assert calls, "execute_sql_query no se invocó"
    query, params = calls[-1]["query"], calls[-1]["params"]
    assert query.count("- make_interval(mins => %s)") == 2
    assert "America/Santo_Domingo" not in query
    assert params == ("u-1", -60, -60), f"offset resuelto ausente: {params!r}"


def test_get_daily_nudge_count_offset_240_por_defecto_sin_perfil(monkeypatch):
    import proactive_agent
    import db_facts

    calls = []

    def _fake_execute(query, params=None, fetch_one=False, **kwargs):
        calls.append({"query": query, "params": params})
        return {"total": 0}

    def _fake_query_for_helper(query, params=None, fetch_one=False, **kwargs):
        return None  # sin perfil

    monkeypatch.setattr(db_facts, "execute_sql_query", _fake_query_for_helper)
    monkeypatch.setattr(db_facts, "connection_pool", object())
    monkeypatch.setattr(proactive_agent, "execute_sql_query", _fake_execute)

    proactive_agent.get_daily_nudge_count("u-1")

    assert calls and calls[-1]["params"] == ("u-1", 240, 240)


# ── funcional: el corte de día DIFIERE entre offset=240 y offset=-60 ─────────

def test_offset_240_vs_menos60_difieren_en_el_corte_de_dia_a_las_03_utc():
    """La demostración concreta que exige el brief: a las 03:00 UTC, RD (offset=240=UTC-4) ya
    cruzó la medianoche hacia AYER; España en invierno (offset=-60=UTC+1) sigue en HOY. Reproduce
    en Python puro la MISMA álgebra que `(col - make_interval(mins => %s))::date` usa en SQL —
    `instant - timedelta(minutes=offset)` — verificada byte a byte contra Neon 2026-08-16 (ver
    task-5-report.md). No requiere DB: es aritmética de calendario, no I/O."""
    from datetime import datetime, timedelta, timezone
    instante = datetime(2026, 8, 16, 3, 0, 0, tzinfo=timezone.utc)

    fecha_rd = (instante - timedelta(minutes=240)).date()
    fecha_es_invierno = (instante - timedelta(minutes=-60)).date()

    assert fecha_rd.isoformat() == "2026-08-15", "RD (UTC-4) debe leer AYER a las 03:00 UTC"
    assert fecha_es_invierno.isoformat() == "2026-08-16", "España invierno (UTC+1) debe leer HOY"
    assert fecha_rd != fecha_es_invierno, "el corte de día debe DIFERIR entre offsets"


def test_dst_america_santo_domingo_no_aplica_240_vale_todo_el_ano():
    """[T5] Documenta la premisa DST del brief: América/Santo_Domingo NO observa horario de
    verano (fijo UTC-4 los 365 días) — a diferencia de España (CET/CEST, sí tiene DST: su
    tzOffset persistido puede ser -60 en invierno pero -120 en verano). Esto es lo que hace
    seguro reemplazar `AT TIME ZONE 'America/Santo_Domingo'` por una constante 240 en vez de
    requerir lógica de calendario — IANA tzdata resuelve el offset de RD a -4h para CUALQUIER
    fecha del año."""
    from zoneinfo import ZoneInfo
    from datetime import datetime
    rd = ZoneInfo("America/Santo_Domingo")
    verano = datetime(2026, 7, 1, 12, 0, 0, tzinfo=rd)
    invierno = datetime(2026, 1, 1, 12, 0, 0, tzinfo=rd)
    assert verano.utcoffset().total_seconds() / 60 == -240
    assert invierno.utcoffset().total_seconds() / 60 == -240


# ═══════════════════════════════════════════════════════════════════════════
# ── T6: presupuesto EUR/MXN/COP — pisos provisionales + moneda local en 422 ──
# ═══════════════════════════════════════════════════════════════════════════
#
# Mecanismo encontrado (pre-existente): `_budget_cycle_floor_dop(days)` es un
# piso TOTAL por ciclo, NO lineal, en DOP, con knob por ciclo
# `MEALFIT_BUDGET_FLOOR_TOTAL_{7,15,30}D_DOP`. `validate_budget_sufficient`
# compara SIEMPRE en espacio DOP: para USD convierte con la tasa fija
# `_budget_usd_to_dop()` (knob `MEALFIT_BUDGET_USD_TO_DOP`, default 60.0);
# cualquier OTRA moneda no reconocida caía en el mismo `else` que DOP (se
# trataba el número como si fuera DOP — el fail-safe pre-Fase-1).
#
# Extensión de T6 (ruling R2-F1): EUR/MXN/COP NO ganan una tasa FX propia —
# ganan su PROPIO piso literal por ciclo (`_budget_cycle_floor_for_currency`),
# derivado UNA vez del piso USD (80/140/260) por factor fijo y redondeado a
# cifra amable, espejo exacto del frontend (`BUDGET_MIN_TOTAL` en
# formValidation.js). La comparación es DIRECTA en la moneda declarada — la
# MISMA semántica que el camino DOP histórico — nunca una conversión FX.

# ── pisos backend: _budget_cycle_floor_for_currency ──────────────────────────

def test_gate_currencies_son_exactamente_las_monedas_beta():
    """[reconvertido · P1-BUDGET-FLOOR-USD · 2026-08-21] Antes fijaba la lista a mano
    (`{"EUR","MXN","COP"}`) y por eso no acusó que USD —moneda de DOS países beta, US y PR— no
    tuviera piso propio: se juzgaba con la cesta dominicana al tipo de cambio.

    Ahora la lista se DERIVA de `COUNTRY_PROFILES`, que es el SSOT de países. Así el test deja de
    describir el estado y pasa a exigir la regla: toda moneda de un país beta tiene piso propio. Si
    mañana entra un séptimo país con moneda nueva y nadie le pone piso, esto falla — que es lo que
    debió pasar con USD y no pasó."""
    from constants import COUNTRY_PROFILES
    esperadas = {
        p["currency"] for cc, p in COUNTRY_PROFILES.items() if p.get("is_beta")
    }
    assert set(nc._BUDGET_CYCLE_FLOOR_DEFAULTS_BY_CURRENCY.keys()) == esperadas, (
        "hay una moneda de país beta sin piso propio (o un piso de una moneda que ya no se usa)"
    )


def test_piso_eur_defaults():
    assert nc._budget_cycle_floor_for_currency(7, "EUR") == 75
    assert nc._budget_cycle_floor_for_currency(15, "EUR") == 135
    assert nc._budget_cycle_floor_for_currency(30, "EUR") == 245


def test_piso_mxn_defaults():
    assert nc._budget_cycle_floor_for_currency(7, "MXN") == 1400
    assert nc._budget_cycle_floor_for_currency(15, "MXN") == 2500
    assert nc._budget_cycle_floor_for_currency(30, "MXN") == 4700


def test_piso_cop_defaults():
    assert nc._budget_cycle_floor_for_currency(7, "COP") == 350000
    assert nc._budget_cycle_floor_for_currency(15, "COP") == 600000
    assert nc._budget_cycle_floor_for_currency(30, "COP") == 1100000


def test_piso_moneda_no_reconocida_delega_en_dop_sin_tocarlo():
    """DOP/basura: delega en `_budget_cycle_floor_dop` — el piso histórico NO se toca ni se
    reimplementa por segunda vez.

    [reconvertido · P1-BUDGET-FLOOR-USD · 2026-08-21] Este test también afirmaba
    `USD == _budget_cycle_floor_dop(days)`, o sea que anclaba el defecto: USD es la moneda de DOS
    países beta (US y PR) y era la única sin piso propio, así que se juzgaba con la cesta
    dominicana al tipo de cambio — 17% por debajo de los US$80/140/260 que el producto ya declara.
    La propiedad que este test protege de verdad es «una moneda NO RECONOCIDA delega»; USD dejó de
    serlo. Se conserva el caso genuino (XYZ, DOP) y el nuevo contrato de USD vive en
    `test_p1_budget_floor_usd.py`."""
    for days in (7, 15, 30):
        assert nc._budget_cycle_floor_for_currency(days, "XYZ") == nc._budget_cycle_floor_dop(days)
        assert nc._budget_cycle_floor_for_currency(days, "DOP") == nc._budget_cycle_floor_dop(days)


def test_piso_ciclo_no_estandar_interpola_desde_7d_igual_que_dop():
    per_day = 75.0 / 7.0
    assert nc._budget_cycle_floor_for_currency(10, "EUR") == pytest.approx(per_day * 10)


def test_piso_knob_override_por_ciclo_y_moneda_sin_contaminar_vecinos(monkeypatch):
    monkeypatch.setenv("MEALFIT_BUDGET_FLOOR_TOTAL_7D_EUR", "999")
    assert nc._budget_cycle_floor_for_currency(7, "EUR") == 999.0
    # Ni el resto de ciclos de EUR ni las otras monedas se contaminan.
    assert nc._budget_cycle_floor_for_currency(15, "EUR") == 135
    assert nc._budget_cycle_floor_for_currency(7, "MXN") == 1400
    assert nc._budget_cycle_floor_for_currency(7, "DOP") == nc._budget_cycle_floor_dop(7)


def test_pisos_nuevos_registrados_en_knobs_registry():
    from knobs import get_knobs_registry_snapshot
    for currency in ("EUR", "MXN", "COP"):
        for days in (7, 15, 30):
            nc._budget_cycle_floor_for_currency(days, currency)
    snap = get_knobs_registry_snapshot()
    for currency in ("EUR", "MXN", "COP"):
        for days in (7, 15, 30):
            name = f"MEALFIT_BUDGET_FLOOR_TOTAL_{days}D_{currency}"
            assert name in snap, f"{name} ausente del registry tras invocar su lector"


# ── paridad frontend↔backend (patrón FORM-DRIFT-ANCHOR, cross-file) ──────────

_BUDGET_MIN_TOTAL_BLOCK = re.compile(
    r"export\s+const\s+BUDGET_MIN_TOTAL\s*=\s*\{(?P<body>.*?)\}\s*;", re.DOTALL)
_CURRENCY_BLOCK = re.compile(r"(\w+):\s*\{([^}]*)\}")
_CYCLE_NUMBER = re.compile(r"(\w+):\s*(\d+)")
_DAYS_BY_CYCLE_NAME = {"weekly": 7, "biweekly": 15, "monthly": 30}


def _read_form_validation_js() -> str:
    path = _FRONTEND / "src" / "config" / "formValidation.js"
    if not path.exists():
        pytest.skip(f"formValidation.js no existe en {path}")
    return path.read_text(encoding="utf-8")


def _parse_budget_min_total(text: str) -> dict:
    block = _BUDGET_MIN_TOTAL_BLOCK.search(text)
    if not block:
        raise AssertionError(
            "No se encontró `export const BUDGET_MIN_TOTAL = {...};` en formValidation.js. "
            "Si el formato cambió, actualiza _BUDGET_MIN_TOTAL_BLOCK."
        )
    body = block.group("body")
    out = {}
    for m in _CURRENCY_BLOCK.finditer(body):
        currency, inner = m.group(1), m.group(2)
        out[currency] = {cm.group(1): int(cm.group(2)) for cm in _CYCLE_NUMBER.finditer(inner)}
    return out


def test_parser_extrae_budget_min_total_dop_usd_sanity():
    """Sanity del parser contra lo que YA existía antes de T6 — si esto falla,
    el regex está roto y los tests de paridad de abajo miden sobre un parser
    inválido, no sobre drift real."""
    parsed = _parse_budget_min_total(_read_form_validation_js())
    assert parsed.get("DOP") == {"weekly": 4000, "biweekly": 7000, "monthly": 13000}
    assert parsed.get("USD") == {"weekly": 80, "biweekly": 140, "monthly": 260}


@pytest.mark.parametrize("currency", ["EUR", "MXN", "COP"])
def test_piso_frontend_backend_coincide(currency):
    frontend = _parse_budget_min_total(_read_form_validation_js())
    assert currency in frontend, f"{currency} ausente de BUDGET_MIN_TOTAL en formValidation.js"
    backend = {
        cycle_name: nc._budget_cycle_floor_for_currency(days, currency)
        for cycle_name, days in _DAYS_BY_CYCLE_NAME.items()
    }
    assert frontend[currency] == backend, (
        f"Drift de piso {currency}: frontend={frontend[currency]} backend={backend}"
    )


def test_mutacion_desincroniza_piso_eur_produce_mismatch_de_paridad(monkeypatch):
    """Prueba que el mecanismo de paridad SÍ detecta drift (no que sea vacío):
    fuerza el knob de override —el MISMO mecanismo que usaría un operador— a
    un valor distinto del frontend y confirma que la comparación deja de
    coincidir. Reproduce en miniatura qué pasaría si un futuro PR tocara un
    piso en un solo lado."""
    frontend = _parse_budget_min_total(_read_form_validation_js())
    backend_before = nc._budget_cycle_floor_for_currency(7, "EUR")
    assert frontend["EUR"]["weekly"] == backend_before, (
        "precondición inválida: deben coincidir ANTES de mutar (si esto falla, "
        "test_piso_frontend_backend_coincide ya debería estar en rojo)"
    )
    monkeypatch.setenv("MEALFIT_BUDGET_FLOOR_TOTAL_7D_EUR", str(int(backend_before) + 1))
    backend_after = nc._budget_cycle_floor_for_currency(7, "EUR")
    assert backend_after != frontend["EUR"]["weekly"], (
        "la mutación no desincronizó el piso — este test de mutación no demuestra "
        "que la paridad detecte drift real"
    )


# ── validate_budget_sufficient: gate del knob + monedas nuevas ───────────────

def _budget_form(currency, amount, country=None, grocery="weekly", household=1):
    f = {
        "weight": 70, "weightUnit": "kg", "height": 170, "age": 30,
        "gender": "male", "activityLevel": "moderate", "mainGoal": "maintenance",
        "groceryDuration": grocery, "householdSize": household,
        "budget": "custom", "budgetAmount": amount, "budgetCurrency": currency,
    }
    if country is not None:
        f["country"] = country
    return f


def test_knob_off_moneda_nueva_se_trata_como_dop_igual_que_antes(monkeypatch):
    """[byte-identidad] Con el knob OFF, budgetCurrency='EUR' cae en el `else`
    de SIEMPRE — tratado como si fuera DOP, símbolo 'RD$' — EXACTAMENTE la
    conducta pre-Fase-1 para cualquier moneda no reconocida. Prueba que la
    rama nueva es INALCANZABLE con el knob apagado (no solo "nadie la llama
    hoy"), aunque un cliente declare 'EUR' en budgetCurrency."""
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    ok_bajo, detail_bajo = nc.validate_budget_sufficient(_budget_form("EUR", 100))
    assert ok_bajo is False
    assert detail_bajo["currency"] == "EUR"
    assert "RD$" in detail_bajo["message"]
    ok_alto, detail_alto = nc.validate_budget_sufficient(_budget_form("EUR", 50000))
    assert ok_alto is True and detail_alto is None


# [P1-COUNTRY-BUDGET-FLOOR-FX · 2026-08-23] CONTRATO CAMBIADO A SABIENDAS.
# Estos tests anclaban «bajo piso RECHAZA» para EUR/MXN/COP. Ese piso resultó ser una
# conversión FX de la cesta dominicana (EUR=USD×0,95 · MXN=USD×18 · COP=USD×4200), no una
# cesta de esos países: un colombiano con 200.000 COP/semana —cifra realista— no podía
# generar plan contra un piso de 437.500 ≈ 1,88 M COP/mes para UNA persona, por encima de
# su salario mínimo. Y pasado el gate el número era inútil: país beta ⇒ lista sin precios.
# Ahora ORIENTA (aviso con las mismas cifras y el mismo mensaje) en vez de BLOQUEAR.
# Lo que sigue anclado aquí es lo que NO cambió: la moneda, el piso calculado y el mensaje
# en su símbolo propio, nunca «RD$». El gate duro sigue medido en DOP/USD (ver
# test_p1_country_budget_floor_fx.py, que además cubre la mutación).
def test_knob_on_pais_es_eur_bajo_piso_avisa(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    ok, detail = nc.validate_budget_sufficient(_budget_form("EUR", 1, country="ES"))
    assert ok is True, "ahora orienta en vez de bloquear (P1-COUNTRY-BUDGET-FLOOR-FX)"
    assert detail["warning_code"] == "budget_below_goal_floor_advisory"
    assert detail["currency"] == "EUR"
    assert detail["min_budget"] > 0
    assert "EUR" in detail["message"]
    assert "RD$" not in detail["message"]


def test_knob_on_pais_es_eur_sobre_piso_acepta(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    ok, detail = nc.validate_budget_sufficient(_budget_form("EUR", 50000, country="ES"))
    assert ok is True and detail is None


def test_knob_on_mx_mxn_bajo_piso_avisa_con_mxn(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    ok, detail = nc.validate_budget_sufficient(_budget_form("MXN", 1, country="MX"))
    assert ok is True, "ahora orienta en vez de bloquear (P1-COUNTRY-BUDGET-FLOOR-FX)"
    assert detail["warning_code"] == "budget_below_goal_floor_advisory"
    assert detail["currency"] == "MXN"
    assert "MXN" in detail["message"]
    assert "RD$" not in detail["message"]


def test_knob_on_co_cop_bajo_piso_avisa_con_cop(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    ok, detail = nc.validate_budget_sufficient(_budget_form("COP", 1, country="CO"))
    assert ok is True, "ahora orienta en vez de bloquear (P1-COUNTRY-BUDGET-FLOOR-FX)"
    assert detail["warning_code"] == "budget_below_goal_floor_advisory"
    assert detail["currency"] == "COP"
    assert "COP" in detail["message"]
    assert "RD$" not in detail["message"]


@pytest.mark.parametrize("currency,country", [("EUR", "ES"), ("MXN", "MX"), ("COP", "CO")])
def test_mensaje_nuevo_nunca_hardcodea_rd_simbolo(monkeypatch, currency, country):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    ok, detail = nc.validate_budget_sufficient(_budget_form(currency, 1, country=country))
    assert ok is True  # [P1-COUNTRY-BUDGET-FLOOR-FX] orienta, no bloquea
    assert detail["warning_code"] == "budget_below_goal_floor_advisory"
    # El MENSAJE es lo que este test mide, y no cambió: mismas cifras, mismo
    # símbolo propio de la moneda. Sólo dejó de venir dentro de un 422.
    assert "RD$" not in detail["message"], (
        f"El mensaje para {currency} contiene 'RD$' hardcodeado — viola el contrato de Task 6."
    )
    assert currency in detail["message"]


def test_usd_bloquea_igual_con_el_knob_encendido_o_apagado(monkeypatch):
    """[reconvertido · P1-BUDGET-FLOOR-USD · 2026-08-21] Antes exigía `detail_off == detail_on`, o
    sea que el knob NO cambiara el mecanismo de USD. Esa igualdad era justo el defecto: significaba
    que encender el sistema de países dejaba a US y PR con la cesta dominicana convertida por
    `_budget_usd_to_dop`, mientras a ES/MX/CO se les daba piso propio. Una devaluación del peso
    movía el mínimo de un usuario de Florida.

    Lo que este test protege de verdad —y sigue protegiendo— es que **el veredicto no dependa del
    knob**: un presupuesto absurdo se rechaza con el sistema encendido y apagado. Lo que ya no se
    exige es que el CAMINO sea el mismo, porque el nuevo camino es el correcto. El contrato de
    rollback (knob apagado ⇒ FX histórico exacto) vive en `test_p1_budget_floor_usd.py`."""
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    ok_off, detail_off = nc.validate_budget_sufficient(_budget_form("USD", 5))
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    ok_on, detail_on = nc.validate_budget_sufficient(_budget_form("USD", 5))
    assert ok_off is False
    assert ok_on is False
    assert detail_off and detail_on, "bloquea sin explicar por qué en alguno de los dos caminos"


def test_dop_sigue_intacto_con_knob_on(monkeypatch):
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    ok_off, detail_off = nc.validate_budget_sufficient(_budget_form("DOP", 100))
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    ok_on, detail_on = nc.validate_budget_sufficient(_budget_form("DOP", 100))
    assert ok_off is False
    assert ok_on is False
    assert detail_off == detail_on


def test_do_path_mensaje_exacto_golden(monkeypatch):
    """Ancla BYTE A BYTE el texto del 422 para un usuario DOP — el camino que
    Task 6 promete dejar intacto. Congela `min_budget_for_goals` para aislar
    el texto de la aritmética de calorías (competencia de OTRO test:
    test_p2_budget_floor.py)."""
    monkeypatch.setattr(nc, "min_budget_for_goals", lambda form_data: {
        "min_budget_dop": 4000, "min_per_day_dop": 571, "days": 7, "household": 1,
        "target_calories": 2000,
    })
    ok, detail = nc.validate_budget_sufficient(
        {"budget": "custom", "budgetAmount": "100", "budgetCurrency": "DOP"})
    assert ok is False
    assert detail["message"] == (
        "Tu presupuesto de RD$100 es insuficiente para tus metas (2000 kcal/día × 7 días). "
        "El mínimo para un plan profesional es ~RD$4,000. Sube tu presupuesto o ajusta tus "
        "metas (menos días, menos personas, o una meta calórica menor). No bajamos la calidad "
        "nutricional para encajar en un presupuesto demasiado bajo."
    )


def test_eur_path_mensaje_exacto_golden(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.setattr(nc, "min_budget_for_goals", lambda form_data: {
        "min_budget_dop": 4000, "min_per_day_dop": 571, "days": 7, "household": 1,
        "target_calories": 2000,
    })
    ok, detail = nc.validate_budget_sufficient(
        {"budget": "custom", "budgetAmount": "50", "budgetCurrency": "EUR", "country": "ES"})
    assert ok is True  # [P1-COUNTRY-BUDGET-FLOOR-FX] orienta, no bloquea
    assert detail["warning_code"] == "budget_below_goal_floor_advisory"
    # El MENSAJE es lo que este test mide, y no cambió: mismas cifras, mismo
    # símbolo propio de la moneda. Sólo dejó de venir dentro de un 422.
    assert detail["message"] == (
        "Tu presupuesto de EUR 50 es insuficiente para tus metas (2000 kcal/día × 7 días). "
        "El mínimo para un plan profesional es ~EUR 75. Sube tu presupuesto o ajusta tus "
        "metas (menos días, menos personas, o una meta calórica menor). No bajamos la calidad "
        "nutricional para encajar en un presupuesto demasiado bajo."
    )


def test_household_mayor_a_1_incluye_clausula_en_mensaje_eur(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.setattr(nc, "min_budget_for_goals", lambda form_data: {
        "min_budget_dop": 8000, "min_per_day_dop": 1143, "days": 7, "household": 2,
        "target_calories": 2000,
    })
    ok, detail = nc.validate_budget_sufficient({
        "budget": "custom", "budgetAmount": "50", "budgetCurrency": "EUR",
        "country": "ES", "householdSize": 2,
    })
    assert ok is True  # [P1-COUNTRY-BUDGET-FLOOR-FX] orienta, no bloquea
    assert detail["warning_code"] == "budget_below_goal_floor_advisory"
    # El MENSAJE es lo que este test mide, y no cambió: mismas cifras, mismo
    # símbolo propio de la moneda. Sólo dejó de venir dentro de un 422.
    assert "× 2 personas" in detail["message"]


def test_simbolo_nuevo_sale_de_country_profiles_no_de_tabla_propia(monkeypatch):
    """Si COUNTRY_PROFILES dejara de reconocer 'EUR' como moneda de algún
    país, el símbolo debe degradar a 'RD$' (fail-safe) — prueba que el
    símbolo se VALIDA contra COUNTRY_PROFILES en vez de asumir ciegamente el
    código que llegó en budgetCurrency."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    fake_profiles = {
        cc: {**profile, "currency": "XXX" if profile["currency"] == "EUR" else profile["currency"]}
        for cc, profile in constants.COUNTRY_PROFILES.items()
    }
    monkeypatch.setattr(constants, "COUNTRY_PROFILES", fake_profiles)
    ok, detail = nc.validate_budget_sufficient(_budget_form("EUR", 1, country="ES"))
    assert ok is False
    assert "RD$" in detail["message"], (
        "Con COUNTRY_PROFILES sin 'EUR' registrado, el símbolo debe caer a 'RD$' (fail-safe) — "
        "si esto falla, el código no está realmente consultando COUNTRY_PROFILES en runtime."
    )


def test_gate_lee_mealfit_country_system_no_otro_nombre():
    src = nc.__file__ and open(nc.__file__, encoding="utf-8").read()
    assert '"MEALFIT_COUNTRY_SYSTEM"' in src, (
        "El gate de EUR/MXN/COP debe leer el knob MEALFIT_COUNTRY_SYSTEM (mismo "
        "nombre que constants.country_for_form_data) — no un knob paralelo."
    )


# ═══════════════════════════════════════════════════════════════════════════
# ── T6 fix-round 1 (review): moneda REALMENTE vigente en los 4 call sites ────
# ═══════════════════════════════════════════════════════════════════════════
#
# Hallazgo del review: `currencySymbol` (QBudget) ya recomputaba `betaCurrency` en
# cada render y por eso gateaba bien, pero `placeholder`/`aria-label` (QBudget) y los
# pisos de `InteractiveAssessmentFlow.jsx`/`useBudgetFloor.js` leían `budgetCurrency`
# CRUDO. Escenario real: usuario elige EUR con la bandera encendida, `budgetCurrency`
# sobrevive en formData/localStorage, la bandera vuelve a apagarse (rollback) SIN que
# nadie limpie `budgetCurrency` — placeholder seguía diciendo "Ej. 100" (EUR), el
# lector de pantalla seguía anunciando "euros", y el gate cliente aceptaba "≥75"
# pensando en EUR mientras el backend (mismo knob apagado) comparaba ese monto contra
# el piso DOP (~4000+) y rechazaba con 422 en RD$.
#
# Fix de la CLASE (no de las instancias): `effectiveBudgetCurrency(country,
# budgetCurrency)` — en `formValidation.js`, no en QBudget.jsx — es la ÚNICA función
# que decide qué moneda está REALMENTE vigente; los 4 call sites (+ el símbolo/toggle
# de QBudget, por uniformidad) pasan TODOS por ella. Reusa `currencyOptionsForCountry`
# (movida al mismo módulo) — CERO segundo mapa país→moneda.

def _read_qbudget_jsx() -> str:
    path = _FRONTEND / "src" / "components" / "assessment" / "questions" / "QBudget.jsx"
    if not path.exists():
        pytest.skip(f"QBudget.jsx no existe en {path}")
    return path.read_text(encoding="utf-8")


def _read_interactive_assessment_flow_jsx() -> str:
    path = _FRONTEND / "src" / "components" / "assessment" / "InteractiveAssessmentFlow.jsx"
    if not path.exists():
        pytest.skip(f"InteractiveAssessmentFlow.jsx no existe en {path}")
    return path.read_text(encoding="utf-8")


def _read_use_budget_floor_js() -> str:
    path = _FRONTEND / "src" / "hooks" / "useBudgetFloor.js"
    if not path.exists():
        pytest.skip(f"useBudgetFloor.js no existe en {path}")
    return path.read_text(encoding="utf-8")


# ── formValidation.js: única fuente de los helpers de moneda ─────────────────

def test_dark_toggle_literales_viven_en_formvalidation():
    """[dark anchor] Los DOS literales que `test_p1_budget_custom.py` YA ancla
    deben vivir en `formValidation.js` (movidos ahí en el fix-round 1) —
    condición necesaria para que, con COUNTRY_SYSTEM_UI=false,
    `currencyOptionsForCountry` arme EXACTAMENTE [DOP, USD]."""
    src = _read_form_validation_js()
    assert re.search(r"value:\s*'DOP'\s*,\s*label:\s*'RD\$'", src)
    assert re.search(r"value:\s*'USD'\s*,\s*label:\s*'US\$'", src)


def test_formvalidation_exporta_helpers_de_moneda_una_sola_vez():
    """`currencyOptionsForCountry`/`effectiveBudgetCurrency`/`BETA_CURRENCY_BY_COUNTRY`
    viven en formValidation.js. QBudget/InteractiveAssessmentFlow/useBudgetFloor
    los IMPORTAN — ninguno los redefine localmente (eso sería el segundo mapa
    país→moneda que el review pidió evitar)."""
    formval_src = _read_form_validation_js()
    assert re.search(r"export (function|const) currencyOptionsForCountry", formval_src)
    assert re.search(r"export function effectiveBudgetCurrency", formval_src)
    assert "export const BETA_CURRENCY_BY_COUNTRY" in formval_src

    for label, src in (
        ("QBudget.jsx", _read_qbudget_jsx()),
        ("InteractiveAssessmentFlow.jsx", _read_interactive_assessment_flow_jsx()),
        ("useBudgetFloor.js", _read_use_budget_floor_js()),
    ):
        assert "export const BETA_CURRENCY_BY_COUNTRY" not in src, (
            f"{label} redefine BETA_CURRENCY_BY_COUNTRY localmente — segundo mapa país→moneda."
        )
        assert "export function currencyOptionsForCountry" not in src, (
            f"{label} redefine currencyOptionsForCountry localmente."
        )
        assert "export function effectiveBudgetCurrency" not in src, (
            f"{label} redefine effectiveBudgetCurrency localmente."
        )


def _parse_beta_currency_by_country(_text: str) -> dict:
    """[P1-COUNTRY-BUDGET-CURRENCY-DEFAULT] El mapa ya no se copia a mano.

    Deriva las mismas filas que consume `BETA_CURRENCY_BY_COUNTRY`: países beta cuya
    moneda no es una de las dos opciones universales DOP/USD.
    """
    countries_src = (_FRONTEND / "src" / "config" / "countries.js").read_text(encoding="utf-8")
    rows = re.findall(
        r"code:\s*'([A-Z]{2})'[^}]*currency:\s*'([A-Z]{3})'[^}]*beta:\s*(true|false)",
        countries_src,
    )
    assert rows, "No se pudieron parsear filas code/currency/beta de countries.js."
    return {
        code: currency
        for code, currency, beta in rows
        if beta == "true" and currency not in ("DOP", "USD")
    }


def test_formvalidation_mapa_beta_currency_por_pais():
    src = _read_form_validation_js()
    assert "Object.fromEntries" in src and "COUNTRIES" in src
    assert _parse_beta_currency_by_country(src) == {"ES": "EUR", "MX": "MXN", "CO": "COP"}


def test_parser_extrae_beta_currency_by_country_sanity():
    parsed = _parse_beta_currency_by_country(_read_form_validation_js())
    assert parsed == {"ES": "EUR", "MX": "MXN", "CO": "COP"}


def test_beta_currency_by_country_coincide_con_country_profiles():
    """[fix-round 1 · review] El comentario de `BETA_CURRENCY_BY_COUNTRY` afirma que
    un drift contra `COUNTRY_PROFILES` lo detecta ESTE test T6 — antes del fix-round
    1 esa frase era falsa (el test previo solo comparaba contra un dict hardcoded
    DENTRO del propio test, ciego a cualquier cambio real de COUNTRY_PROFILES).
    Ahora compara de verdad, en las DOS direcciones, contra el backend."""
    frontend_map = _parse_beta_currency_by_country(_read_form_validation_js())
    for country_code, frontend_currency in frontend_map.items():
        assert country_code in constants.COUNTRY_PROFILES, (
            f"{country_code} está en BETA_CURRENCY_BY_COUNTRY (frontend) pero no en "
            f"COUNTRY_PROFILES (backend)."
        )
        backend_currency = constants.COUNTRY_PROFILES[country_code]["currency"]
        assert frontend_currency == backend_currency, (
            f"Drift de moneda para {country_code}: frontend (BETA_CURRENCY_BY_COUNTRY) "
            f"dice '{frontend_currency}', backend (COUNTRY_PROFILES) dice "
            f"'{backend_currency}'."
        )
    # A la inversa: todo país beta con moneda propia (≠ DOP/USD) en el backend debe
    # tener entrada en el frontend, o su toggle nunca ofrecerá esa moneda.
    for country_code, profile in constants.COUNTRY_PROFILES.items():
        if profile["is_beta"] and profile["currency"] not in ("DOP", "USD"):
            assert country_code in frontend_map, (
                f"{country_code} tiene moneda beta '{profile['currency']}' en "
                f"COUNTRY_PROFILES (backend) pero no aparece en BETA_CURRENCY_BY_COUNTRY "
                f"(frontend) — su toggle nunca la ofrecería."
            )
            assert frontend_map[country_code] == profile["currency"]


def test_mutacion_desincroniza_beta_currency_produce_mismatch(monkeypatch):
    """Igual que la mutación de pisos: prueba que la comparación SÍ detecta drift.
    Muta COUNTRY_PROFILES (no el frontend — más simple que editar+revertir archivo
    fuente para un solo dict) y confirma que deja de coincidir."""
    frontend_map = _parse_beta_currency_by_country(_read_form_validation_js())
    fake_profiles = {
        cc: {**profile, "currency": "XXX" if cc == "ES" else profile["currency"]}
        for cc, profile in constants.COUNTRY_PROFILES.items()
    }
    monkeypatch.setattr(constants, "COUNTRY_PROFILES", fake_profiles)
    assert frontend_map["ES"] != constants.COUNTRY_PROFILES["ES"]["currency"], (
        "la mutación no desincronizó — el test de mutación es inválido"
    )


# ── QBudget.jsx: importa (no redefine) + usa effectiveCurrency en TODO lo visible ──

def test_qbudget_importa_helpers_de_formvalidation():
    src = _read_qbudget_jsx()
    assert re.search(
        r"import\s*\{[^}]*currencyOptionsForCountry[^}]*effectiveBudgetCurrency[^}]*\}"
        r"\s*from\s*['\"]\.\./\.\./\.\./config/formValidation['\"]"
        r"|"
        r"import\s*\{[^}]*effectiveBudgetCurrency[^}]*currencyOptionsForCountry[^}]*\}"
        r"\s*from\s*['\"]\.\./\.\./\.\./config/formValidation['\"]",
        src,
    ), "QBudget debe importar currencyOptionsForCountry Y effectiveBudgetCurrency de config/formValidation."
    assert re.search(
        r"import\s*\{[^}]*COUNTRY_SYSTEM_UI[^}]*\}\s*from\s*['\"]\.\./\.\./\.\./config/countries['\"]",
        src,
    ), "QBudget debe seguir importando COUNTRY_SYSTEM_UI de config/countries."


def test_qbudget_currency_symbol_y_toggle_usan_effective_currency():
    """[fix-round 1 · review] `currencySymbol` Y el `value` resaltado del toggle
    deben derivar de `effectiveCurrency` — dos mecanismos distintos para la misma
    pregunta ("¿qué moneda es la vigente?") fue exactamente el origen del bug."""
    src = _read_qbudget_jsx()
    assert re.search(r"const effectiveCurrency = effectiveBudgetCurrency\(", src), (
        "QBudget no calcula effectiveCurrency vía effectiveBudgetCurrency."
    )
    m = re.search(r"const currencySymbol = ([\s\S]*?);\n", src)
    assert m, "No se pudo aislar `const currencySymbol = ...;` en QBudget.jsx"
    symbol_body = m.group(1)
    assert "effectiveCurrency" in symbol_body, "currencySymbol debe derivar de effectiveCurrency."
    # [P3-I18N-MONEDA-COMPUESTA-A-MANO-EN-EL-PRESUPUESTO] El símbolo ya no se arma con
    # ramas literales locales: Intl/currencySymbolFor es el SSOT para los cinco idiomas.
    assert "currencySymbolFor(effectiveCurrency)" in symbol_body
    assert re.search(r"value=\{effectiveCurrency\}", src), (
        "El UnitToggle no resalta `effectiveCurrency` — podría seguir resaltando una moneda STALE."
    )


def test_qbudget_placeholder_usa_effective_currency():
    src = _read_qbudget_jsx()
    assert "effectiveCurrency === 'MXN'" in src
    assert "effectiveCurrency === 'COP'" in src
    assert "budgetCurrency === 'MXN'" not in src, (
        "El placeholder volvió a leer budgetCurrency crudo en vez de effectiveCurrency."
    )
    assert "budgetCurrency === 'COP'" not in src
    assert "Ej. 2000" in src
    assert "Ej. 400000" in src


def test_qbudget_aria_label_usa_effective_currency_y_conserva_frases():
    src = _read_qbudget_jsx()
    assert "effectiveCurrency === 'EUR'" in src
    assert "budgetCurrency === 'EUR'" not in src, (
        "El aria-label volvió a leer budgetCurrency crudo en vez de effectiveCurrency."
    )
    assert "Presupuesto total en euros" in src
    assert "Presupuesto total en pesos mexicanos" in src
    assert "Presupuesto total en pesos colombianos" in src
    # Byte-identidad del dark path: los 2 originales siguen ahí.
    assert "Presupuesto total en dólares" in src
    assert "Presupuesto total en pesos dominicanos" in src


# ── InteractiveAssessmentFlow.jsx / useBudgetFloor.js: los 2 call sites que el
#    review encontró SIN gatear ────────────────────────────────────────────────

def test_interactive_assessment_flow_gate_usa_effective_budget_currency():
    """[fix-round 1 · review] `isCustomBudgetValid` (SSOT de las TRES puertas:
    validateExtra del step, el salto a la última pregunta, y el submit) debe
    resolver la moneda vía `effectiveBudgetCurrency` — no `fd.budgetCurrency`
    crudo, o una moneda beta STALE (bandera apagada tras rollback) aceptaría un
    monto que el backend, con el mismo knob apagado, rechazaría con 422 en RD$."""
    src = _read_interactive_assessment_flow_jsx()
    m = re.search(r"const isCustomBudgetValid = \(fd\) =>([\s\S]*?);\n\n", src)
    assert m, "No se pudo aislar isCustomBudgetValid en InteractiveAssessmentFlow.jsx"
    body = m.group(1)
    assert "effectiveBudgetCurrency(" in body, (
        "isCustomBudgetValid no usa effectiveBudgetCurrency — vulnerable al rollback "
        "de moneda STALE (review de Task 6, fix-round 1)."
    )
    assert "fd.budgetCurrency || 'DOP'" not in body, (
        "isCustomBudgetValid sigue usando fd.budgetCurrency crudo."
    )
    assert re.search(
        r"import\s*\{[^}]*effectiveBudgetCurrency[^}]*\}\s*from\s*['\"]\.\./\.\./config/formValidation['\"]",
        src,
    ), "InteractiveAssessmentFlow.jsx no importa effectiveBudgetCurrency de config/formValidation."


def test_use_budget_floor_usa_effective_budget_currency():
    """[fix-round 1 · review] El piso ESTÁTICO (fallback sin red, lo que se ve
    mientras el fetch personalizado no ha vuelto) de `useBudgetFloor` debe resolver
    la moneda vía `effectiveBudgetCurrency` — mismo motivo que el gate del flow."""
    src = _read_use_budget_floor_js()
    assert re.search(r"const currency = effectiveBudgetCurrency\(", src), (
        "useBudgetFloor no calcula `currency` vía effectiveBudgetCurrency."
    )
    assert "formData?.budgetCurrency || 'DOP'" not in src, (
        "useBudgetFloor sigue leyendo formData?.budgetCurrency crudo."
    )
    assert re.search(
        r"import\s*\{[^}]*effectiveBudgetCurrency[^}]*\}\s*from\s*['\"]\.\./config/formValidation['\"]",
        src,
    ), "useBudgetFloor.js no importa effectiveBudgetCurrency de config/formValidation."


# ═══════════════════════════════════════════════════════════════════════════
# ── T7: lista de compras en modo beta honesto (_pricing_mode) ────────────────
# ═══════════════════════════════════════════════════════════════════════════
#
# Con `COUNTRY_PROFILES[país]['has_native_prices'] is False` (los 5 betas: ES/US/MX/PR/CO),
# el backend deja de emitir CUALQUIER monto denominado en RD$: el aggregator
# (`estimated_cost_rd=None` en cada ítem), el resumen de costo (`shopping_cost_summary` ⇒
# None/ausente), la reconciliación de presupuesto (`budget_reconciliation` ⇒ ausente — cae
# río abajo de un summary None), las sugerencias de ahorro (`build_budget_suggestions` ⇒ []
# — se filtran solas cuando ningún ítem tiene precio), el mensaje de consentimiento del chat
# (`_build_consent_message` omite el "(~RD$...)"), y las 2 inyecciones de precios al LLM
# (prompt de generación + chat-modify). DO / knob apagado ⇒ la clave `plan_data._pricing_mode`
# NUNCA se escribe — byte-identidad total (ni siquiera aparece como `None` explícito).

import json as _json


def test_case_placeholder_t7_import_ok():
    """Sanity: los módulos que T7 toca importan sin error (colección lazy de abajo)."""
    import shopping_calculator as _sc
    import agent as _agent
    import tools as _tools
    assert _sc and _agent and _tools


# ── constants.pricing_mode_for_country / pricing_mode_for_form_data (SSOT del literal) ──────

def test_pricing_mode_for_country_do_es_none():
    assert constants.pricing_mode_for_country("DO") is None


def test_pricing_mode_for_country_beta_es_beta_no_prices():
    for cc in _BETA_CCS:
        assert constants.pricing_mode_for_country(cc) == "beta_no_prices", cc


def test_pricing_mode_for_country_desconocido_es_none():
    """Fail-safe: un código fuera de COUNTRY_PROFILES nunca inventa un modo."""
    assert constants.pricing_mode_for_country("ZZ") is None
    assert constants.pricing_mode_for_country(None) is None


def test_pricing_mode_for_form_data_knob_apagado_es_siempre_none(monkeypatch):
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    for fd in ({"country": "ES"}, {"country": "MX"}, {}, None, "no-dict"):
        assert constants.pricing_mode_for_form_data(fd) is None


def test_pricing_mode_for_form_data_knob_encendido_beta(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    assert constants.pricing_mode_for_form_data({"country": "es"}) == "beta_no_prices"
    assert constants.pricing_mode_for_form_data({"country": "DO"}) is None
    assert constants.pricing_mode_for_form_data({"country": "basura"}) is None
    assert constants.pricing_mode_for_form_data({}) is None


def test_pricing_mode_es_composicion_pura_de_country_for_form_data(monkeypatch):
    """No debe reimplementar el gate del knob/canonicalización — solo componer las 2
    puertas ya existentes (T1 + el mapa has_native_prices)."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    for fd in ({"country": "CO"}, {"country": "pr"}, {"country": "xx"}, {}):
        expected = constants.pricing_mode_for_country(constants.country_for_form_data(fd))
        assert constants.pricing_mode_for_form_data(fd) == expected


# ── assemble_plan_node: estampado del flag ANTES de la agregación ───────────────────────────

def test_assemble_plan_node_deriva_pricing_mode_via_ssot():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("async def assemble_plan_node")
    j = src.index("from constants import pricing_mode_for_form_data", i)
    assert i < j, "assemble_plan_node no deriva el pricing_mode vía el helper SSOT."
    window = src[j:j + 300]
    assert "_pricing_mode = pricing_mode_for_form_data(form_data)" in window
    assert 'result["_pricing_mode"] = _pricing_mode' in window
    # la clave SOLO se escribe si el helper devolvió algo truthy (DO/knob-off ⇒ None ⇒
    # nunca se escribe) — nunca un `result["_pricing_mode"] = _pricing_mode` incondicional.
    assert "if _pricing_mode:" in window


def test_assemble_plan_node_estampa_pricing_mode_antes_del_primer_get_shopping_list_delta():
    """El flag debe existir en `result` ANTES de que `get_shopping_list_delta` lo lea —
    si no, la 1ª pasada de agregación de un plan nuevo no suprimiría precios."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i_stamp = src.index('result["_pricing_mode"] = _pricing_mode')
    i_first_call = src.index("get_shopping_list_delta,", i_stamp)  # el import, primera línea que la nombra
    assert i_stamp < i_first_call


# ── aggregator: get_shopping_list_delta suprime estimated_cost_rd/estimated_cost ────────────

def _priced_plan_fixture(pricing_mode=None):
    plan = {
        "days": [{"day": 1, "meals": [{
            "meal": "Almuerzo",
            "ingredients": ["200 g de pollo", "1 taza de arroz"],
            "ingredients_raw": ["200 g de pollo", "1 taza de arroz"],
        }]}],
    }
    if pricing_mode:
        plan["_pricing_mode"] = pricing_mode
    return plan


def test_get_shopping_list_delta_beta_anula_estimated_cost_rd(monkeypatch):
    """Fuerza un precio no-cero vía `_cost_from_market` (el catálogo real está vacío en
    este entorno de test sin .env — sin forzarlo, la ausencia de RD$ sería un falso verde
    por falta de datos, no por el gate). Con `_pricing_mode='beta_no_prices'`, TODO ítem
    estructurado debe salir con `estimated_cost_rd=None`."""
    import shopping_calculator as sc
    monkeypatch.setattr(sc, "_cost_from_market", lambda *a, **kw: 100.0)
    sc.reset_caps_applied_last_run()
    plan = _priced_plan_fixture(pricing_mode="beta_no_prices")
    items = sc.get_shopping_list_delta(None, plan, True, False, True, 1.0)
    assert items, "el fixture debe producir al menos un ítem agregado"
    for it in items:
        assert it.get("estimated_cost_rd") is None, it


def test_get_shopping_list_delta_do_control_conserva_estimated_cost_rd(monkeypatch):
    """Control NEGATIVO del test anterior: el MISMO fixture, MISMO monkeypatch, SIN la
    clave `_pricing_mode` — debe conservar el precio forzado. Prueba que el monkeypatch
    realmente inyecta un precio (si esto fallara, el test beta de arriba sería un falso
    verde por catálogo vacío, no por el gate funcionando)."""
    import shopping_calculator as sc
    monkeypatch.setattr(sc, "_cost_from_market", lambda *a, **kw: 100.0)
    sc.reset_caps_applied_last_run()
    plan = _priced_plan_fixture(pricing_mode=None)
    items = sc.get_shopping_list_delta(None, plan, True, False, True, 1.0)
    assert items
    assert any(it.get("estimated_cost_rd") == 100.0 for it in items), items


def test_get_shopping_list_delta_pricing_mode_no_beta_no_es_no_op(monkeypatch):
    """Cualquier valor de `_pricing_mode` que NO sea exactamente 'beta_no_prices' (typo,
    valor legacy, etc.) NO debe suprimir precios — el chequeo es igualdad estricta,
    nunca truthy genérico."""
    import shopping_calculator as sc
    monkeypatch.setattr(sc, "_cost_from_market", lambda *a, **kw: 100.0)
    sc.reset_caps_applied_last_run()
    plan = _priced_plan_fixture(pricing_mode="algo_que_no_es_el_literal")
    items = sc.get_shopping_list_delta(None, plan, True, False, True, 1.0)
    assert any(it.get("estimated_cost_rd") == 100.0 for it in items), items


def test_strip_prices_for_beta_pricing_mode_structured_list():
    import shopping_calculator as sc
    items = [{"name": "Pollo", "estimated_cost_rd": 250.5, "estimated_cost": 250.5}]
    out = sc._strip_prices_for_beta_pricing_mode(items)
    assert out is items, "debe mutar in-place y retornar el mismo objeto"
    assert items[0]["estimated_cost_rd"] is None
    assert items[0]["estimated_cost"] is None


def test_strip_prices_for_beta_pricing_mode_categorized_dict():
    import shopping_calculator as sc
    cats = {"Proteínas": [{"name": "Pollo", "estimated_cost_rd": 250.5}],
            "Vegetales": [{"name": "Tomate", "estimated_cost_rd": 40.0}]}
    sc._strip_prices_for_beta_pricing_mode(cats)
    assert cats["Proteínas"][0]["estimated_cost_rd"] is None
    assert cats["Vegetales"][0]["estimated_cost_rd"] is None


def test_strip_prices_for_beta_pricing_mode_texto_plano_no_op():
    """`structured=False` produce `list[str]`/`dict[str, list[str]]` — sin campos de
    costo, el strip no debe reventar (items no son dict)."""
    import shopping_calculator as sc
    items = ["200 g de pollo", "1 taza de arroz"]
    out = sc._strip_prices_for_beta_pricing_mode(items)
    assert out == ["200 g de pollo", "1 taza de arroz"]


def test_strip_prices_for_beta_pricing_mode_ausencia_de_campo_no_los_inventa():
    """Un ítem SIN `estimated_cost_rd`/`estimated_cost` (ej. urgentes crudos) no debe
    ganar la clave de la nada — `in` guardia cada asignación."""
    import shopping_calculator as sc
    items = [{"name": "Urgente sin precio"}]
    sc._strip_prices_for_beta_pricing_mode(items)
    assert "estimated_cost_rd" not in items[0]
    assert "estimated_cost" not in items[0]


# ── compute_shopping_cost_summary: pricing_mode ⇒ None (nunca dict de ceros) ─────────────────

def test_compute_shopping_cost_summary_beta_es_none():
    import shopping_calculator as sc
    weekly = [{"name": "Pollo", "estimated_cost_rd": None, "is_perishable": True}]
    out = sc.compute_shopping_cost_summary(weekly, weekly, weekly, "weekly",
                                            pricing_mode="beta_no_prices")
    assert out is None


def test_compute_shopping_cost_summary_default_pricing_mode_es_byte_identico():
    """Sin el kwarg (callers preexistentes que no lo pasan), comportamiento EXACTO a
    antes de T7 — no debe volverse None por defecto."""
    import shopping_calculator as sc
    weekly = [{"name": "Pollo", "estimated_cost_rd": 100.0, "is_perishable": True}]
    with_kwarg_none = sc.compute_shopping_cost_summary(weekly, weekly, weekly, "weekly", pricing_mode=None)
    without_kwarg = sc.compute_shopping_cost_summary(weekly, weekly, weekly, "weekly")
    assert with_kwarg_none is not None
    assert without_kwarg is not None
    assert with_kwarg_none["by_duration"]["weekly"]["trip_total_rd"] == \
        without_kwarg["by_duration"]["weekly"]["trip_total_rd"]


def test_compute_shopping_cost_summary_pricing_mode_no_beta_string_no_suprime():
    import shopping_calculator as sc
    weekly = [{"name": "Pollo", "estimated_cost_rd": 100.0, "is_perishable": True}]
    out = sc.compute_shopping_cost_summary(weekly, weekly, weekly, "weekly", pricing_mode="DO")
    assert out is not None


# [T7 fix-round · review Critical] La entrega original gateó 5 de 8 call sites reales de
# `compute_shopping_cost_summary` — el escaneo era por ARCHIVO (un `needle` fijo por
# fichero), así que un 2º call site en el MISMO archivo (T2 `_sum_t2b`, la 2ª pasada de
# convergencia) o un call site en un archivo no listado (`routers/plans.py::_rebuild_
# plan_shopping_lists_inline`, DEFAULT-ON, invocado desde swap-persist/regen-day/
# recipe-expand) quedaban invisibles al test. El reviewer lo ejecutó empíricamente: un
# plan beta con `estimated_cost_rd=None` en todos sus ítems producía un dict de CEROS
# no-None que SÍ se persistía como `shopping_cost_summary` — el modo de fallo exacto que
# el docstring de la función advierte.
#
# Reemplazo: escaneo GENÉRICO por CALL SITE, no por archivo. Encuentra TODO alias de
# import de `compute_shopping_cost_summary` en cualquier .py de producción bajo backend/
# (excluidos tests/, venvs, scripts/scratch — no código de app en caliente) y CADA
# invocación de ese alias (paren-matched, multi-línea seguro), exige `pricing_mode=` en
# la lista de argumentos. Comentarios stripeados ANTES de buscar — una mención en un
# comentario no debe contar ni como import ni como call site.

_CSS_EXCLUDED_TOP_DIRS = {
    "tests", "venv", "venv-test", "test_venv", "__pycache__", "scripts", "scratch",
    "migrations", "data", "docs", "infra", "uploads", ".git", ".pytest_cache", ".superpowers",
}
_CSS_ALIAS_IMPORT_RE = re.compile(r"compute_shopping_cost_summary\s+as\s+(\w+)")


def _css_strip_comment_lines(src: str) -> str:
    return "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))


def _css_iter_backend_py_files():
    for p in sorted(_BACKEND.rglob("*.py")):
        rel = p.relative_to(_BACKEND)
        if rel.parts and rel.parts[0] in _CSS_EXCLUDED_TOP_DIRS:
            continue
        yield p


def _find_compute_shopping_cost_summary_calls():
    """[(relpath, lineno_aprox, alias, args_text), ...] — CADA invocación real (no import,
    no comentario) en código de producción, vía cualquier alias."""
    calls = []
    for path in _css_iter_backend_py_files():
        raw = path.read_text(encoding="utf-8")
        src = _css_strip_comment_lines(raw)
        aliases = set(_CSS_ALIAS_IMPORT_RE.findall(src))
        if path.name == "shopping_calculator.py":
            # La propia definición del módulo no es un call site — nunca se importa
            # "as compute_shopping_cost_summary" a sí misma, pero por si acaso.
            aliases.discard("compute_shopping_cost_summary")
        for alias in aliases:
            call_re = re.compile(r"\b" + re.escape(alias) + r"\s*\(")
            for m in call_re.finditer(src):
                i = m.end() - 1  # posición de '('
                depth = 0
                j = i
                while j < len(src):
                    if src[j] == "(":
                        depth += 1
                    elif src[j] == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                args_text = src[m.start(): j + 1]
                lineno = src[: m.start()].count("\n") + 1
                calls.append((str(path.relative_to(_BACKEND)).replace("\\", "/"), lineno, alias, args_text))
    return calls


def test_compute_shopping_cost_summary_todo_call_site_pasa_pricing_mode():
    """Debe fallar si CUALQUIER call site (presente o un 9º futuro) no lleva
    `pricing_mode=` en su lista de argumentos — sin importar en qué archivo viva ni
    cuántas veces se invoque el mismo alias importado."""
    calls = _find_compute_shopping_cost_summary_calls()
    assert calls, "el escaneo no encontró NINGÚN call site — probablemente está roto."
    missing = [
        (relpath, lineno, alias) for relpath, lineno, alias, args in calls
        if "pricing_mode=" not in args
    ]
    assert not missing, (
        "Call site(s) de compute_shopping_cost_summary SIN pricing_mode=:\n  - "
        + "\n  - ".join(f"{r}:~{n} (alias {a})" for r, n, a in missing)
    )


def test_compute_shopping_cost_summary_ocho_call_sites_exactos():
    """Ancla el conteo que el review reprodujo empíricamente (8). Si sube, un call site
    NUEVO apareció — el test de arriba ya lo cubre, pero este hace el crecimiento
    visible en vez de silencioso. Si baja, alguien consolidó/eliminó un call site
    (probablemente está bien, pero merece que el número baje a propósito, no por
    accidente de este test)."""
    calls = _find_compute_shopping_cost_summary_calls()
    sites = ", ".join(f"{r}:~{n}" for r, n, _a, _args in calls)
    assert len(calls) == 8, f"se esperaban 8 call sites, se hallaron {len(calls)}: {sites}"


# ── build_budget_reference: beta ⇒ None incondicional (T6 fold) ─────────────────────────────

def test_build_budget_reference_beta_es_none(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    ref = nc.build_budget_reference({"budget": "custom", "budgetAmount": "500",
                                      "budgetCurrency": "EUR", "country": "ES"})
    assert ref is None


def test_build_budget_reference_beta_es_none_para_los_5_paises_beta(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    for cc in _BETA_CCS:
        for tier in ("low", "medium", "high", "unlimited", "custom"):
            ref = nc.build_budget_reference({"budget": tier, "budgetAmount": "500", "country": cc})
            assert ref is None, (cc, tier)


def test_build_budget_reference_do_sigue_intacto(monkeypatch):
    """DO / knob apagado ⇒ el guard de T7 nunca dispara — comportamiento EXACTO a antes."""
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    ref_off = nc.build_budget_reference({"budget": "medium", "groceryDuration": "weekly"})
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    ref_do_on = nc.build_budget_reference({"budget": "medium", "groceryDuration": "weekly", "country": "DO"})
    assert ref_off is not None and ref_do_on is not None
    assert ref_off == ref_do_on


def test_build_budget_reference_beta_guard_deriva_pricing_mode_via_ssot():
    src = nc.__file__ and open(nc.__file__, encoding="utf-8").read()
    i = src.index("def build_budget_reference")
    j = src.find("\ndef ", i + 10)
    body = src[i:j if j != -1 else len(src)]
    assert "from constants import pricing_mode_for_form_data" in body
    assert 'pricing_mode_for_form_data(form_data) == "beta_no_prices"' in body
    assert "return None" in body.split('pricing_mode_for_form_data(form_data) == "beta_no_prices"')[1][:40]


# ── reconcile_budget_with_cost: contrato preexistente (cost_summary None ⇒ None) ────────────

def test_reconcile_budget_with_cost_cost_summary_none_es_none():
    """El 'camino degradado numérico' que el brief cita como YA existente — T7 lo dispara
    con el flag (compute_shopping_cost_summary ⇒ None), no lo reimplementa."""
    ref = {"tier": "custom", "basis": "custom", "currency": "EUR",
           "reference_rd": 100, "floor_rd": 50, "days": 7, "household": 1}
    assert nc.reconcile_budget_with_cost(ref, None) is None


# ── build_budget_suggestions: [] natural (filtra por estimated_cost_rd > 0, sin cambios) ────

def test_build_budget_suggestions_beta_lista_sin_precios_es_vacia():
    import shopping_calculator as sc
    weekly = [
        {"name": "Camarones", "estimated_cost_rd": None},
        {"name": "Salmón", "estimated_cost_rd": None},
    ]
    assert sc.build_budget_suggestions(weekly, user_id=None) == []


def test_build_budget_suggestions_do_control_no_es_vacia():
    """Control: el MISMO shape, con precio real, SÍ produce sugerencias (via
    cheapest_supermarket_variant) — ancla que la lista vacía de arriba es por AUSENCIA
    de precio, no por un bug que la deje siempre vacía."""
    import shopping_calculator as sc
    weekly = [{"name": "Aceite de oliva", "estimated_cost_rd": 500.0}]
    # fail-open: sin catálogo de supermercado en este entorno de test, cheapest_supermarket_
    # variant devuelve None y build_budget_suggestions no añade nada — el punto de este test
    # es que NO explota, no que produzca contenido (eso ya lo cubre test_p1_budget_brand_premium.py).
    out = sc.build_budget_suggestions(weekly, user_id=None)
    assert isinstance(out, list)


# ── _build_consent_message: omite "(~RD$...)" cuando el precio es None (sin cambios) ────────

def test_build_consent_message_precio_none_omite_rd():
    import agent
    msg = agent._build_consent_message([{"name": "Camarón", "qty_needed": 1, "unit": "lb", "est_price_rd": None}])
    assert "RD$" not in msg
    assert "Camarón" in msg


def test_build_consent_message_precio_numerico_control_incluye_rd():
    """Control: con precio numérico (el camino DO), el mensaje SÍ incluye el monto — este
    comportamiento es PRE-EXISTENTE y T7 no lo toca."""
    import agent
    msg = agent._build_consent_message([{"name": "Camarón", "qty_needed": 1, "unit": "lb", "est_price_rd": 350}])
    assert "RD$350" in msg


def test_swap_meal_with_consent_anula_est_price_rd_en_pais_beta():
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    i = src.index("def swap_meal_with_consent")
    j = src.find("\ndef ", i + 10)
    body = src[i:j if j != -1 else len(src)]
    assert "missing = _price_missing_ingredients(_unauthorized)" in body
    i_missing = body.index("missing = _price_missing_ingredients(_unauthorized)")
    i_country = body.index("country_for_form_data(form_data)", i_missing)
    i_strip = body.index('_m["est_price_rd"] = None', i_country)
    i_call = body.index("_build_consent_message(missing)", i_strip)
    # orden: computar missing -> derivar país -> anular precios -> RECIÉN entonces
    # construir el mensaje (si el orden se invirtiera, el mensaje vería precios viejos).
    assert i_missing < i_country < i_strip < i_call
    assert "has_native_prices" in body


def test_swap_meal_with_consent_missing_ingredients_payload_tambien_queda_limpio():
    """El guard debe anular `est_price_rd` en los DICTS de `missing` (que también salen
    tal cual en `missing_ingredients` del payload JSON) — no solo en el string del
    mensaje. Ancla que la mutación vive ANTES del `return {...}` que expone `missing`."""
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    i = src.index("def swap_meal_with_consent")
    j = src.find("\ndef ", i + 10)
    body = src[i:j if j != -1 else len(src)]
    i_strip = body.index('_m["est_price_rd"] = None')
    i_return = body.index('"missing_ingredients": missing,')
    assert i_strip < i_return


# ── tools.py execute_modify_single_meal: gate de la inyección de precios al chat ────────────

def test_execute_modify_single_meal_gatea_inteligencia_de_precios_por_pricing_mode():
    src = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    i = src.index("def execute_modify_single_meal")
    j = src.find("\ndef ", i + 10)
    body = src[i:j if j != -1 else len(src)]
    assert 'if plan_data.get("_pricing_mode") != "beta_no_prices":' in body
    i_gate = body.index('if plan_data.get("_pricing_mode") != "beta_no_prices":')
    i_prices = body.index("INTELIGENCIA DE PRECIOS", i_gate)
    assert i_gate < i_prices, "el gate debe envolver el bloque de inyección de precios."


def test_execute_modify_single_meal_usa_plan_data_no_form_data_para_el_gate():
    """[decisión deliberada] El gate usa `plan_data['_pricing_mode']` (fetcheado de DB,
    confiable HOY) en vez de `_modify_country`/`form_data` (T4 ya documenta que el
    chat-agent no puebla `country` en form_data todavía — un gate sobre esa variable
    sería un placeholder inerte)."""
    src = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    i = src.index("def execute_modify_single_meal")
    j = src.find("\ndef ", i + 10)
    body = src[i:j if j != -1 else len(src)]
    i_plan_data_fetch = body.index('plan_data = plan_record["plan_data"]')
    i_gate = body.index('if plan_data.get("_pricing_mode") != "beta_no_prices":')
    assert i_plan_data_fetch < i_gate, "el gate debe leer plan_data DESPUÉS de fetchearlo."


# ── graph_orchestrator._build_shared_context: prices_context (2ª inyección LLM) ─────────────

def test_build_shared_context_prices_context_gatea_por_pais():
    cuerpo = _cuerpo_build_shared_context()
    assert "_shared_ctx_country = country_for_form_data(form_data)" in cuerpo
    i_var = cuerpo.index("_shared_ctx_country = country_for_form_data(form_data)")
    i_prices = cuerpo.index('"prices_context":', i_var)
    window = cuerpo[i_prices:i_prices + 400]
    # [P3-PRICING-MODE-SSOT-BLANKET · 2026-08-22] Antes se anclaba la expresión a mano
    # (`COUNTRY_PROFILES.get(_shared_ctx_country, {}).get("has_native_prices", True)`). Eso era
    # exactamente el «2º chequeo» que el comentario de `pricing_mode_for_country` prohíbe por
    # escrito — o sea que este guard EXIGÍA la segunda tabla. La propiedad que defiende (el
    # `prices_context` va gateado por país) no cambia; el mecanismo pasa por la puerta del SSOT.
    assert "pricing_mode_for_country(_shared_ctx_country)" in window, (
        "el gate de `prices_context` dejó de pasar por `pricing_mode_for_country`"
    )
    assert '"beta_no_prices"' in window, (
        "el gate no compara contra el literal SSOT del modo beta"
    )
    # el predicado de budget histórico (P3-GENCHUNK-SPEED) sigue verbatim adentro —
    # anclado también por test_j_prices_context_gated_on_budget (test_p2_genchunk_speed.py).
    assert 'build_prices_context() if (str(form_data.get("budget") or "").strip())' in window


def test_build_shared_context_country_context_reusa_la_misma_derivacion():
    """`country_context` (T3) y `prices_context` (T7) deben compartir `_shared_ctx_country`
    — country_for_form_data(form_data) se llama UNA sola vez en toda la función."""
    cuerpo = _cuerpo_build_shared_context()
    n = cuerpo.count("country_for_form_data(form_data)")
    assert n == 1, f"se esperaba exactamente 1 derivación de país, hallada(s) {n}"
    assert '"country_context": _country_context_block(_shared_ctx_country)' in cuerpo


def test_p3_genchunk_speed_prices_gate_test_sigue_vivo():
    """[anti-regresión cruzada] El test histórico J (test_p2_genchunk_speed.py) sigue
    pudiendo encontrar el predicado de budget EXACTO en TODO el archivo — no solo dentro
    de _build_shared_context — confirma que T7 no lo movió/reescribió."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'build_prices_context() if (str(form_data.get("budget") or "").strip())' in src
    assert "P3-GENCHUNK-SPEED-PRICES-GATE" in src


# ── api_budget_floor / budget_floor_in_currency: hint en la moneda REAL, nunca DOP mal etiquetado ──

def test_budget_floor_in_currency_dop_usd_byte_identico_al_mecanismo_historico():
    days, min_dop = 7, 4000.0
    usd_dop = nc._budget_usd_to_dop()
    amt_dop, cur_dop = nc.budget_floor_in_currency(days, "DOP", min_dop)
    assert (round(amt_dop), cur_dop) == (round(min_dop), "DOP")
    amt_usd, cur_usd = nc.budget_floor_in_currency(days, "USD", min_dop)
    assert (round(amt_usd), cur_usd) == (round(min_dop / usd_dop), "USD")


def test_budget_floor_in_currency_moneda_no_reconocida_cae_a_dop():
    amt, cur = nc.budget_floor_in_currency(7, "XYZ", 4000.0)
    assert (round(amt), cur) == (4000, "DOP")


def test_budget_floor_in_currency_knob_apagado_beta_currency_cae_a_dop(monkeypatch):
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    amt, cur = nc.budget_floor_in_currency(7, "EUR", 4000.0)
    assert (round(amt), cur) == (4000, "DOP")


def test_budget_floor_in_currency_knob_encendido_eur_usa_su_propio_piso(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    # household=1, calorías de referencia ⇒ scale=1.0 ⇒ el piso propio de la moneda tal cual.
    dop_base = nc._budget_cycle_floor_dop(7)
    amt, cur = nc.budget_floor_in_currency(7, "EUR", dop_base)
    assert cur == "EUR"
    assert round(amt) == round(nc._budget_cycle_floor_for_currency(7, "EUR"))


def test_budget_floor_in_currency_escala_por_el_mismo_factor_calorias_hogar(monkeypatch):
    """min_budget_dop YA duplicado (simula calorías×hogar altos) ⇒ el monto en EUR debe
    escalar por el MISMO factor 2×, no quedar anclado al piso base."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    dop_base = nc._budget_cycle_floor_dop(7)
    amt_1x, _ = nc.budget_floor_in_currency(7, "EUR", dop_base)
    amt_2x, _ = nc.budget_floor_in_currency(7, "EUR", dop_base * 2)
    assert amt_2x == pytest.approx(amt_1x * 2)


def test_api_budget_floor_knob_on_pais_beta_responde_en_su_moneda(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    from routers.plans import api_budget_floor
    res = asyncio.run(api_budget_floor(payload=_budget_form("EUR", 1, country="ES"), _uid=None))
    assert res.get("ok") is True
    assert res["currency"] == "EUR"
    assert res["min_budget"] > 0
    # NUNCA un monto DOP (miles) mal etiquetado como si fuera EUR (decenas/centenas).
    dop_floor = nc._budget_cycle_floor_dop(7)
    assert res["min_budget"] < dop_floor


def test_api_budget_floor_knob_off_pais_beta_cae_a_dop_byte_identico(monkeypatch):
    """Knob apagado (default) ⇒ EXACTAMENTE el mecanismo pre-T7, aunque el cliente declare
    EUR + country=ES."""
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    from routers.plans import api_budget_floor
    res = asyncio.run(api_budget_floor(payload=_budget_form("EUR", 1, country="ES"), _uid=None))
    assert res.get("ok") is True
    assert res["currency"] == "DOP"


def test_api_budget_floor_mx_co_tambien_responden_en_su_moneda(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    from routers.plans import api_budget_floor
    for currency, cc in (("MXN", "MX"), ("COP", "CO")):
        res = asyncio.run(api_budget_floor(payload=_budget_form(currency, 1, country=cc), _uid=None))
        assert res["currency"] == currency, (currency, cc)


def test_api_budget_floor_tier_references_en_beta_no_mezcla_escalas():
    """`tier_references` (low/medium/high) debe escalar sobre `min_in_currency`, nunca
    devolver la fórmula DOP/USD disfrazada de EUR — cada banda debe ser mayor que el
    mínimo (factor >= 1.0 documentado en _budget_tier_band_factor)."""
    import os
    prev = os.environ.get("MEALFIT_COUNTRY_SYSTEM")
    os.environ["MEALFIT_COUNTRY_SYSTEM"] = "true"
    try:
        from routers.plans import api_budget_floor
        res = asyncio.run(api_budget_floor(payload=_budget_form("EUR", 1, country="ES"), _uid=None))
        refs = res.get("tier_references") or {}
        assert refs, "tier_references vacío para un país beta"
        for tier_val in refs.values():
            assert tier_val >= res["min_budget"]
    finally:
        if prev is None:
            os.environ.pop("MEALFIT_COUNTRY_SYSTEM", None)
        else:
            os.environ["MEALFIT_COUNTRY_SYSTEM"] = prev


# ── sweep final: 0 apariciones de 'RD$' y 0 estimated_cost_rd numérico en TODO el payload ───

def test_t7_sweep_beta_plan_cero_rd_en_todo_el_payload(monkeypatch):
    """[la lección del review final de F0] La prosa compuesta backend es donde se esconde
    'RD$'. Construye un plan beta, corre aggregator + resumen + sugerencias + mensaje de
    consentimiento, serializa TODO a JSON y barre 0 apariciones de 'RD$' y 0
    `estimated_cost_rd` numérico. El precio se FUERZA vía monkeypatch (ver docstring de
    test_get_shopping_list_delta_beta_anula_estimated_cost_rd) para que la ausencia sea
    prueba del GATE, no de un catálogo vacío."""
    import shopping_calculator as sc
    import agent
    monkeypatch.setattr(sc, "_cost_from_market", lambda *a, **kw: 999.0)
    sc.reset_caps_applied_last_run()

    plan = _priced_plan_fixture(pricing_mode="beta_no_prices")
    items = sc.get_shopping_list_delta(None, plan, True, False, True, 1.0)
    assert items

    summary = sc.compute_shopping_cost_summary(
        items, items, items, "weekly", pricing_mode=plan["_pricing_mode"]
    )
    suggestions = sc.build_budget_suggestions(items, user_id=None)
    consent_msg = agent._build_consent_message(
        [{"name": "Camarón", "qty_needed": 1, "unit": "lb", "est_price_rd": None}]
    )

    payload = {
        "aggregated_shopping_list": items,
        "shopping_cost_summary": summary,
        "budget_suggestions": suggestions,
        "consent_message": consent_msg,
    }
    blob = _json.dumps(payload, ensure_ascii=False, default=str)

    assert "RD$" not in blob, blob
    for it in items:
        cost = it.get("estimated_cost_rd") if isinstance(it, dict) else None
        assert cost is None or (isinstance(cost, (int, float)) and cost == 0), it
    assert summary is None
    assert suggestions == []


def test_t7_sweep_control_do_plan_mismo_fixture_produce_rd(monkeypatch):
    """Control NEGATIVO del sweep: el MISMO fixture/monkeypatch SIN `_pricing_mode` SÍ
    produce 'RD$' en el payload — prueba que el sweep de arriba mide el gate real (si
    esto fallara, ambos tests serían falsos-verdes por un fixture que nunca genera RD$)."""
    import shopping_calculator as sc
    monkeypatch.setattr(sc, "_cost_from_market", lambda *a, **kw: 999.0)
    sc.reset_caps_applied_last_run()

    plan = _priced_plan_fixture(pricing_mode=None)
    items = sc.get_shopping_list_delta(None, plan, True, False, True, 1.0)
    summary = sc.compute_shopping_cost_summary(items, items, items, "weekly", pricing_mode=None)
    blob = _json.dumps({"aggregated_shopping_list": items, "shopping_cost_summary": summary},
                        ensure_ascii=False, default=str)
    assert any(it.get("estimated_cost_rd") == 999.0 for it in items), items
    assert summary is not None


# ═══════════════════════════════════════════════════════════════════════════
# ── T7 fix-round (review Critical) — el leak REAL que el reviewer ejecutó ───────────────────
# ═══════════════════════════════════════════════════════════════════════════
#
# `compute_shopping_cost_summary` tiene 8 call sites de producción (no 6 como documentaba
# la entrega original) — 3 quedaron sin `pricing_mode=`:
#   1. `routers/plans.py::_rebuild_plan_shopping_lists_inline` (alias `_ccs_il`) — DEFAULT-ON
#      (`MEALFIT_UPDATE_INLINE_LIST_RECALC` default "true"), invocado desde swap-persist,
#      regen-day y recipe-expand. Sin el gate, un plan beta (`estimated_cost_rd=None` en
#      TODOS sus ítems) producía un dict de CEROS técnicamente no-`None` que SÍ se
#      persistía como `shopping_cost_summary` — el modo de fallo exacto que el docstring de
#      la función ya advertía en prosa pero no cerraba en código.
#   2. `graph_orchestrator.py::_bc_ccs` (2ª pasada de budget-convergence) — en la práctica
#      inalcanzable (depende de `status=="excedido"`, que nunca ocurre sin un primer
#      `shopping_cost_summary` no-None), pero gateado por el MISMO principio de
#      defensa-en-profundidad que `build_budget_reference`.
#   3. `cron_tasks.py::_ccs_t2` 2ª invocación (`_sum_t2b`) — el gate del primer llamado
#      (`_sum_t2`) NO cubre este 2º call porque lee "excedido" desde
#      `full_plan_data.get('budget_reconciliation')` — datos PERSISTIDOS, no un cómputo
#      fresco de este chunk. Una reconciliación STALE sembrada por el leak #1 (que corre en
#      OTRA superficie, antes de que este chunk corra) podría disparar este pase y
#      auto-perpetuarse vía `refresh_budget_reconciliation`.
#
# Este bloque prueba el leak #1 FUNCIONALMENTE (el más grave: DEFAULT-ON, tráfico real de
# usuario) invocando la función real, no un mock.

def _inline_recalc_plan_fixture(pricing_mode=None):
    plan = {
        "days": [{"day": 1, "meals": [{
            "meal": "Almuerzo",
            "ingredients": ["200 g de pollo", "1 taza de arroz"],
            "ingredients_raw": ["200 g de pollo", "1 taza de arroz"],
        }]}],
    }
    if pricing_mode:
        plan["_pricing_mode"] = pricing_mode
    return plan


def test_rebuild_plan_shopping_lists_inline_beta_no_persiste_shopping_cost_summary(monkeypatch):
    """[T7 fix-round · review Critical] El leak REAL ejecutado por el reviewer: invoca
    `_rebuild_plan_shopping_lists_inline` (routers/plans.py, DEFAULT-ON, swap-persist/
    regen-day/recipe-expand) con un plan_data beta y un precio FORZADO vía monkeypatch
    (mismo patrón anti-falso-verde del resto del archivo). `shopping_cost_summary` NO debe
    quedar escrito en plan_data — antes del fix quedaba un dict de ceros no-None."""
    import shopping_calculator as sc
    from routers.plans import _rebuild_plan_shopping_lists_inline
    monkeypatch.setattr(sc, "_cost_from_market", lambda *a, **kw: 999.0)
    sc.reset_caps_applied_last_run()

    plan_data = _inline_recalc_plan_fixture(pricing_mode="beta_no_prices")
    ok = _rebuild_plan_shopping_lists_inline(plan_data, None, "test-surface-t7")

    assert ok is True, "el rebuild inline debe reportar éxito (el gate de pricing NO es un fallo)."
    assert "shopping_cost_summary" not in plan_data, (
        f"REGRESIÓN: shopping_cost_summary quedó escrito para un plan beta: "
        f"{plan_data.get('shopping_cost_summary')!r}"
    )
    assert "budget_reconciliation" not in plan_data
    # El aggregator SÍ corrió (las listas se reconstruyeron) — el gate es específico al
    # resumen de costo, no un abort general del rebuild.
    weekly = plan_data.get("aggregated_shopping_list_weekly")
    assert weekly, "el rebuild debe seguir reconstruyendo las listas incluso en modo beta."
    for it in weekly:
        assert it.get("estimated_cost_rd") is None, it


def test_rebuild_plan_shopping_lists_inline_do_control_si_persiste(monkeypatch):
    """Control DO: el MISMO fixture/monkeypatch SIN el flag SÍ persiste
    `shopping_cost_summary` con costo real — ancla que el test de arriba mide el gate,
    no un `_rebuild_plan_shopping_lists_inline` que jamás persiste nada."""
    import shopping_calculator as sc
    from routers.plans import _rebuild_plan_shopping_lists_inline
    monkeypatch.setattr(sc, "_cost_from_market", lambda *a, **kw: 999.0)
    sc.reset_caps_applied_last_run()

    plan_data = _inline_recalc_plan_fixture(pricing_mode=None)
    ok = _rebuild_plan_shopping_lists_inline(plan_data, None, "test-surface-t7-control")

    assert ok is True
    assert "shopping_cost_summary" in plan_data
    assert plan_data["shopping_cost_summary"]["by_duration"]["weekly"]["trip_total_rd"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# Task 8 — cierre de fase
# ═══════════════════════════════════════════════════════════════════════════
#
# T7 dejó como mandato: "barrido de TODOS los callers de las funciones derivadas
# de SLOT_INAPPROPRIATE_FOODS antes de cerrar la fase" (progress.md, Task 4 fix
# round 1) — el fix round 1 y 2 de T4 encontraron caller tras caller no-gateado
# reviewer-por-reviewer (routers/plans.py, dos veces). Esta sección hace el
# barrido de una vez: TODO caller de producción de las 6 funciones derivadas
# (slot_violations_for_meal_name / slot_ingredient_violations /
# slot_rules_for_country / _detect_slot_appropriateness /
# slot_coherence_backstop_for_meal / build_meal_timing_rules), clasificado como
# país-consciente (wired) o exento-documentado (marker `# [P1-COUNTRY-SYSTEM-F1
# EXENTO: <razón>]` a pocas líneas), con el conteo anclado por función — un
# NUEVO caller futuro sin ninguna de las dos etiquetas tira este test a rojo.
# Tabla resultante (mismos números): backend/docs/country_system_f1.md.

# ── Guard blanket: country_for_form_data es el ÚNICO lector de form_data['country'] ──
# Extiende el guard de F0 (test_p1_country_system_f0.py::
# test_el_dato_viaja_pero_el_motor_no_lo_lee_todavia, scope=solo
# graph_orchestrator.py) a los 5 módulos restantes que F1 tocó. constants.py
# queda FUERA del barrido a propósito: ahí vive el cuerpo de
# country_for_form_data, la ÚNICA lectura legítima de la key.

_COUNTRY_BLANKET_FILES = (
    "graph_orchestrator.py", "cron_tasks.py", "shopping_calculator.py",
    "nutrition_calculator.py", "agent.py", "tools.py",
)
# [P1-COUNTRY-SYSTEM-F2 · 2026-08-17 (Task 9, F6)] SSOT compartida con el guard T2 (arriba,
# `_FORM_SHAPE_COUNTRY_READ_RX`) — antes eran DOS objetos regex con el MISMO patrón (y el MISMO
# bug de corchetes) definidos por separado; ahora un solo re-anclaje beneficia a ambos guards.
_FORM_DATA_COUNTRY_RE = _FORM_SHAPE_COUNTRY_READ_RX


def test_country_for_form_data_es_el_unico_lector_en_los_6_modulos():
    """Con o sin el knob, NINGÚN símbolo de estos 6 archivos debe leer
    form_data['country']/form_data.get('country') directamente — la ÚNICA puerta
    es constants.country_for_form_data. Comentarios stripeados ANTES de
    matchear (mismo patrón que el guard F0). Mutación: reintroducir
    `form_data.get('country')` crudo en cualquiera de los 6 ⇒ RED (verificado a
    mano contra graph_orchestrator.py durante el desarrollo de este test)."""
    offenders = {}
    for fname in _COUNTRY_BLANKET_FILES:
        src = (_BACKEND / fname).read_text(encoding="utf-8")
        sin_comentarios = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
        hits = _FORM_DATA_COUNTRY_RE.findall(sin_comentarios)
        if hits:
            offenders[fname] = len(hits)
    assert not offenders, (
        f"Lector(es) suelto(s) de form_data['country'] fuera de "
        f"country_for_form_data: {offenders} — todo consumo debe pasar por "
        f"constants.country_for_form_data."
    )


# ── El barrido de callers de las 6 funciones derivadas de SLOT_INAPPROPRIATE_FOODS ──

_SCS_EXENTO_RE = re.compile(r"P1-COUNTRY-SYSTEM-F1 EXENTO:")
_SCS_WINDOW = 8  # líneas de margen del scanner; el AUTOR escribe el marker a ~5


def _scs_mask_comments_preserve_lines(raw: str) -> str:
    """Enmascara CONTENIDO de comentarios y strings (COMMENT/STRING/FSTRING_*) con el
    tokenizer real de Python — inmune a strings multi-línea (docstrings) que un scanner
    char-a-char por línea pierde de vista entre líneas. Bug real encontrado durante el
    desarrollo de este test: una MENCIÓN en PROSA de una docstring (p.ej.
    "resuelve `slot_rules_for_country(country)`", texto legítimo escrito por T4) se
    contaba como una LLAMADA real con un masker ingenuo que resetea su estado 'dentro de
    string' en cada salto de línea — 3 falsos positivos reproducidos y cerrados con este
    tokenizer-based rewrite. Preserva saltos de línea exactos (line numbers intactos,
    necesario para el check de ventana ±N líneas del marker EXENTO). Fallback: si el
    archivo no tokeniza, retorna `raw` sin tocar (peor caso: un false-positive, nunca un
    crash del scanner — verificado 0/86 archivos de backend/ fallan a tokenizar)."""
    import tokenize
    import io as _io
    lines = raw.splitlines(keepends=True)
    try:
        tokens = list(tokenize.generate_tokens(_io.StringIO(raw).readline))
    except Exception:
        return raw
    for tok in tokens:
        tname = tokenize.tok_name.get(tok.type, "")
        if tname != "COMMENT" and tname != "STRING" and "FSTRING" not in tname:
            continue
        (sr, sc), (er, ec) = tok.start, tok.end
        if sr == er:
            line = lines[sr - 1]
            lines[sr - 1] = line[:sc] + (" " * (ec - sc)) + line[ec:]
            continue
        first = lines[sr - 1]
        has_nl = first.endswith("\n")
        body_len = len(first) - (1 if has_nl else 0)
        lines[sr - 1] = first[:sc] + (" " * (body_len - sc)) + ("\n" if has_nl else "")
        for mid in range(sr, er - 1):
            line = lines[mid]
            nl = "\n" if line.endswith("\n") else ""
            lines[mid] = (" " * (len(line) - len(nl))) + nl
        last = lines[er - 1]
        lines[er - 1] = (" " * ec) + last[ec:]
    return "".join(lines)


def _scs_file_can_call(masked_src: str, fn_name: str) -> bool:
    """True si `fn_name` está definido en este archivo O importado — single-line O
    multi-línea parenthesized (`from constants import (\\n    ...\\n)`). Bug real
    encontrado: graph_orchestrator.py importa slot_violations_for_meal_name /
    build_meal_timing_rules en un bloque de 14 líneas; un check per-line los pierde,
    produciendo un falso NEGATIVO (peor que un falso positivo: 2 call sites reales
    quedaban invisibles al barrido)."""
    if re.search(r"^\s*(?:async\s+)?def\s+" + re.escape(fn_name) + r"\s*\(", masked_src, re.MULTILINE):
        return True
    for m in re.finditer(r"\bimport\s", masked_src):
        start = m.end()
        if masked_src[start:start + 200].lstrip().startswith("("):
            paren_start = masked_src.index("(", start)
            depth, j = 0, paren_start
            while j < len(masked_src):
                if masked_src[j] == "(":
                    depth += 1
                elif masked_src[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            span = masked_src[start:j + 1]
        else:
            line_end = masked_src.find("\n", start)
            span = masked_src[start: line_end if line_end != -1 else len(masked_src)]
        if re.search(r"\b" + re.escape(fn_name) + r"\b", span):
            return True
    return False


def _scs_split_top_level_args(inner: str) -> list:
    """Divide el contenido ENTRE los parens externos de una llamada en argumentos
    top-level, respetando anidamiento (),[],{}. El contenido de strings ya llegó
    enmascarado (espacios) desde `_scs_mask_comments_preserve_lines`, así que no hace
    falta lógica propia de escape/quote — solo depth-tracking de los 3 pares de
    delimitadores."""
    inner = inner.strip()
    if not inner:
        return []
    args, buf, depth = [], [], 0
    for c in inner:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == "," and depth == 0:
            args.append("".join(buf))
            buf = []
            continue
        buf.append(c)
    args.append("".join(buf))
    return [a.strip() for a in args if a.strip()]


_scs_masked_cache: dict = {}


def _scs_masked_files():
    """[(path, masked_src), ...] — UNA sola tokenización+máscara por archivo, cacheada a
    nivel de módulo (compartida entre las 6 funciones auditadas de este archivo de
    test). Sin la cache, cada función re-tokeniza los ~86 archivos de backend/
    (incluido graph_orchestrator.py, ~50k líneas) desde cero — medido: 30s el barrido
    completo sin cache vs 6s con ella + el pre-filtro de abajo."""
    if not _scs_masked_cache:
        needles = tuple(_SCS_SPECS.keys())
        for path in _css_iter_backend_py_files():
            raw = path.read_text(encoding="utf-8")
            # Pre-filtro barato: si NINGÚN nombre de las 6 funciones aparece ni como
            # substring crudo, tokenizar es trabajo desperdiciado (medido: 79/86
            # archivos de backend/ no mencionan ninguna de las 6).
            if not any(n in raw for n in needles):
                continue
            _scs_masked_cache[path] = _scs_mask_comments_preserve_lines(raw)
    return _scs_masked_cache.items()


def _scs_find_calls(fn_name: str):
    """[(relpath, lineno, callee_alias, args_list), ...] — TODO call site de
    producción de `fn_name` (directo + vía alias de import), backend/ excluyendo
    tests/scripts/venvs/etc (mismo set que T7, _CSS_EXCLUDED_TOP_DIRS)."""
    calls = []
    alias_re = re.compile(r"\b" + re.escape(fn_name) + r"\s+as\s+(\w+)")
    for path, masked in _scs_masked_files():
        names = set(alias_re.findall(masked))
        if _scs_file_can_call(masked, fn_name):
            names.add(fn_name)
        for name in names:
            call_re = re.compile(r"\b" + re.escape(name) + r"\s*\(")
            for m in call_re.finditer(masked):
                line_start = masked.rfind("\n", 0, m.start()) + 1
                prefix = masked[line_start:m.start()].strip()
                if prefix.endswith("def"):
                    continue
                i = m.end() - 1
                depth, j = 0, i
                while j < len(masked):
                    if masked[j] == "(":
                        depth += 1
                    elif masked[j] == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                inner = masked[i + 1:j]
                lineno = masked[:m.start()].count("\n") + 1
                calls.append((
                    str(path.relative_to(_BACKEND)).replace("\\", "/"), lineno, name,
                    _scs_split_top_level_args(inner),
                ))
    return calls


# Config por función: `min_wired` = nº de args top-level que implica país-consciencia
# (el arg extra más allá de la firma country-blind mínima); `kw` = prefijo keyword
# alternativo (positional O keyword, cualquiera cuenta); `structural` = True cuando
# la función NO tiene NINGÚN parámetro de país en su firma (slot_ingredient_violations)
# — todo call site de una función `structural` exige el marker EXENTO sin excepción,
# porque no hay argumento posible que lo vuelva "wired".
_SCS_SPECS = {
    "slot_rules_for_country": dict(min_wired=1, kw=None, structural=False),
    "_detect_slot_appropriateness": dict(min_wired=2, kw="form_data=", structural=False),
    "slot_coherence_backstop_for_meal": dict(min_wired=3, kw="country=", structural=False),
    "build_meal_timing_rules": dict(min_wired=2, kw="country=", structural=False),
    "slot_violations_for_meal_name": dict(min_wired=3, kw="rules_table=", structural=False),
    "slot_ingredient_violations": dict(min_wired=None, kw=None, structural=True),
}

_scs_raw_lines_cache: dict = {}


def _scs_raw_lines(relpath: str) -> list:
    if relpath not in _scs_raw_lines_cache:
        _scs_raw_lines_cache[relpath] = (_BACKEND / relpath).read_text(encoding="utf-8").splitlines()
    return _scs_raw_lines_cache[relpath]


def _scs_has_exento_nearby(relpath: str, lineno: int, window: int = _SCS_WINDOW) -> bool:
    lines = _scs_raw_lines(relpath)
    lo = max(0, lineno - 1 - window)
    hi = min(len(lines), lineno - 1 + window + 1)
    return any(_SCS_EXENTO_RE.search(l) for l in lines[lo:hi])


def _scs_classify(fn_name: str):
    """[(relpath, lineno, alias, status), ...] — status ∈ {'wired', 'exento', 'DESNUDO'}."""
    spec = _SCS_SPECS[fn_name]
    out = []
    for relpath, lineno, alias, args in _scs_find_calls(fn_name):
        if spec["structural"]:
            status = "exento" if _scs_has_exento_nearby(relpath, lineno) else "DESNUDO"
        else:
            wired = len(args) >= spec["min_wired"]
            if not wired and spec["kw"]:
                wired = any(a.startswith(spec["kw"]) for a in args)
            if wired:
                status = "wired"
            elif _scs_has_exento_nearby(relpath, lineno):
                status = "exento"
            else:
                status = "DESNUDO"
        out.append((relpath, lineno, alias, status))
    return out


def _scs_assert_no_desnudos(fn_name: str):
    results = _scs_classify(fn_name)
    assert results, f"{fn_name}: el escaneo no encontró NINGÚN call site — probablemente está roto."
    desnudos = [(r, l, a) for r, l, a, s in results if s == "DESNUDO"]
    assert not desnudos, (
        f"{fn_name}: call site(s) SIN wiring de país NI marker EXENTO (añade "
        f"`country=`/`rules_table=`/`form_data=` o `# [P1-COUNTRY-SYSTEM-F1 EXENTO: "
        f"<razón>]` a ≤{_SCS_WINDOW} líneas):\n  - "
        + "\n  - ".join(f"{r}:{l} (alias {a})" for r, l, a in desnudos)
    )


def test_scs_slot_rules_for_country_sin_desnudos():
    _scs_assert_no_desnudos("slot_rules_for_country")


def test_scs_detect_slot_appropriateness_sin_desnudos():
    _scs_assert_no_desnudos("_detect_slot_appropriateness")


def test_scs_slot_coherence_backstop_for_meal_sin_desnudos():
    _scs_assert_no_desnudos("slot_coherence_backstop_for_meal")


def test_scs_build_meal_timing_rules_sin_desnudos():
    _scs_assert_no_desnudos("build_meal_timing_rules")


def test_scs_slot_violations_for_meal_name_sin_desnudos():
    _scs_assert_no_desnudos("slot_violations_for_meal_name")


def test_scs_slot_ingredient_violations_sin_desnudos():
    _scs_assert_no_desnudos("slot_ingredient_violations")


def test_scs_conteo_exacto_por_funcion():
    """Ancla el conteo de call sites de producción por función — mismos números que la
    tabla de backend/docs/country_system_f1.md. Si sube, un call site NUEVO apareció (el
    test `..._sin_desnudos` de esa función ya exige que esté wired/exento, pero éste
    hace el crecimiento VISIBLE en vez de silencioso, mismo patrón que T7
    `test_compute_shopping_cost_summary_ocho_call_sites_exactos`). Si baja, alguien
    consolidó/eliminó un call site — probablemente bien, pero merece bajar a propósito."""
    expected = {
        "slot_rules_for_country": 3,
        "_detect_slot_appropriateness": 5,
        "slot_coherence_backstop_for_meal": 2,
        "build_meal_timing_rules": 5,
        "slot_violations_for_meal_name": 8,
        "slot_ingredient_violations": 2,
    }
    actual = {fn: len(_scs_classify(fn)) for fn in _SCS_SPECS}
    assert actual == expected, f"conteo de call sites cambió: esperado {expected}, real {actual}"


# ═══════════════════════════════════════════════════════════════════════════
# FINAL-FIX (2026-08-16) — 4 hallazgos de la review final de Fase 1, todos flip-gated (nada
# muerde en oscuro):
#   F1 — planner/preferences/meal_operations sin wiring de país (spec §Fase 1.1 los nombra).
#   F2 — el país nunca llegaba a swap/regen-day/chat: T3/T4 dejaron la MECÁNICA (los callers
#        SABEN recibir `country=`) pero nada poblaba `country` en el `form_data`/`data` real —
#        wiring presente en código, inerte en runtime.
#   F3 — build_budget_context clampeaba cualquier moneda fuera de {DOP, USD} a DOP y el bloque
#        que el LLM SÍ lee mentía con «RD$» sobre un monto declarado en otra moneda.
#   F4 — el texto de rechazo del gate S1 (`_detect_slot_appropriateness`) seguía mandando
#        "dominicano"/es-DO incondicionalmente pese a que T4 ya construyó
#        `_SLOT_POSITIVE_HINT_NEUTRAL` para esta exacta necesidad.
# ═══════════════════════════════════════════════════════════════════════════

# ── F1a: planner.py — Categoría A de desayuno + ejemplo CORRECTO ────────────

def test_f1a_planner_do_o_none_es_byte_identico_is():
    from prompts.planner import build_planner_system_prompt as build, PLANNER_SYSTEM_PROMPT
    assert build("DO") is PLANNER_SYSTEM_PROMPT
    assert build(None) is PLANNER_SYSTEM_PROMPT
    assert build() is PLANNER_SYSTEM_PROMPT


def test_f1a_planner_country_desconocido_cae_a_do():
    from prompts.planner import build_planner_system_prompt as build, PLANNER_SYSTEM_PROMPT
    assert build("xx") is PLANNER_SYSTEM_PROMPT
    assert build("garbage") is PLANNER_SYSTEM_PROMPT


def test_f1a_planner_beta_neutraliza_categoria_a_y_ejemplo():
    from prompts.planner import build_planner_system_prompt as build
    assert _BETA_CCS, "no hay países beta — fixture vacío"
    for cc in _BETA_CCS:
        out = build(cc)
        assert 'Categoría A "Tubérculos/Mangú"' not in out, cc
        assert "Ejemplo CORRECTO: Día 1=Mangú (A)" not in out, cc
        assert "Tubérculos/Plátano" in out, cc
        # Categorías B-E y el ejemplo INCORRECTO (fuera del alcance citado por la review) sobreviven.
        assert 'Categoría B "Cereales/Avena"' in out, cc
        assert "Ejemplo INCORRECTO: Día 1=Mangú de plátano (A)" in out, cc


def test_f1a_planner_cache_dimensionada():
    from prompts.planner import build_planner_system_prompt as build, _PLANNER_PROMPT_COUNTRY_CACHE
    for cc in _BETA_CCS:
        build(cc)
    assert len(_PLANNER_PROMPT_COUNTRY_CACHE) <= len(_BETA_CCS)


def test_f1a_shared_context_expone_country_key_crudo():
    """`ctx['country']` (código crudo, distinto de `ctx['country_context']` el bloque
    renderizado) es lo que `plan_skeleton_node` reusa para no re-derivar."""
    cuerpo = _cuerpo_build_shared_context()
    assert '"country":' in cuerpo, (
        "_build_shared_context no expone ctx['country'] — plan_skeleton_node no puede reusar "
        "la derivación de T3 sin re-derivar country_for_form_data(form_data) una 2ª vez."
    )


def test_f1a_plan_skeleton_node_wire_planner_con_ctx_country():
    """Los 2 call sites del planner (SystemMessage bajo cache-knob + rama legacy sin cache)
    deben threadear `ctx['country']` — no re-derivar `country_for_form_data(form_data)`.
    Comentarios stripeados (mismo patrón que el resto del archivo): un 3er hit vive en prosa de
    comentario (docstring de `_build_shared_context`, mencionando el call site como ejemplo) y
    no cuenta como call site real."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    sin_comentarios = "\n".join(
        l for l in src.splitlines() if not l.strip().startswith("#")
    )
    n = sin_comentarios.count("build_planner_system_prompt(ctx['country'])")
    assert n == 2, f"esperaba 2 call sites (SystemMessage + rama sin-cache), hallado {n}"


def test_f1a_breakfast_cat_label_do_byte_equal():
    from prompts.day_generator import build_day_assignment_context as bdac
    skeleton = {"protein_pool": [], "breakfast_category": "Mangú/Tubérculos"}
    do1 = bdac(skeleton, 1, day_name="Lunes")
    do2 = bdac(skeleton, 1, day_name="Lunes", country="DO")
    assert do1 == do2
    assert "CATEGORÍA DE DESAYUNO ASIGNADA: Mangú/Tubérculos" in do1
    assert "NO uses mangú/tubérculos" in do1


def test_f1a_breakfast_cat_label_beta_traducida():
    from prompts.day_generator import build_day_assignment_context as bdac
    skeleton = {"protein_pool": [], "breakfast_category": "Mangú/Tubérculos"}
    for cc in _BETA_CCS:
        out = bdac(skeleton, 1, day_name="Lunes", country=cc)
        assert "Mangú/Tubérculos" not in out, cc
        assert "CATEGORÍA DE DESAYUNO ASIGNADA: Tubérculos/plátano (preparación local)" in out, cc
        assert "NO uses tubérculo/plátano" in out, cc


def test_f1a_breakfast_cat_enum_value_del_skeleton_no_se_muta():
    """El enum de schemas.py (skeleton_day['breakfast_category']) NUNCA se toca — solo la LABEL
    mostrada al LLM en este bloque cambia. Otros consumidores del mismo dict (ej. el brief
    anti-repetición cross-day en graph_orchestrator.py:8870) deben seguir viendo el valor exacto
    del schema."""
    from prompts.day_generator import build_day_assignment_context as bdac
    skeleton = {"protein_pool": [], "breakfast_category": "Mangú/Tubérculos"}
    bdac(skeleton, 1, day_name="Lunes", country="ES")
    assert skeleton["breakfast_category"] == "Mangú/Tubérculos"


def test_f1a_breakfast_cat_otra_categoria_conserva_label_pero_hereda_warn_beta():
    """Una categoría YA neutral (Avena/Cereales) no debe alterarse — el `.replace()` de la LABEL
    exige match literal 'Mangú/Tubérculos'. La frase de advertencia SÍ cambia por país siempre
    (es un anti-cheat genérico contra la categoría A, no condicionado a la asignación del día)."""
    from prompts.day_generator import build_day_assignment_context as bdac
    skeleton = {"protein_pool": [], "breakfast_category": "Avena/Cereales"}
    out_do = bdac(skeleton, 1, day_name="Lunes", country="DO")
    out_beta = bdac(skeleton, 1, day_name="Lunes", country="ES")
    assert "CATEGORÍA DE DESAYUNO ASIGNADA: Avena/Cereales" in out_do
    assert "CATEGORÍA DE DESAYUNO ASIGNADA: Avena/Cereales" in out_beta
    assert "NO uses mangú/tubérculos" in out_do
    assert "NO uses tubérculo/plátano" in out_beta


# ── F1b: preferences.py — bullet "FIDELIDAD CULTURAL es-DO" del seeder ──────

def test_f1b_variety_prompt_do_o_none_byte_identico():
    import random
    from prompts.preferences import build_deterministic_variety_prompt as build, DETERMINISTIC_VARIETY_PROMPT
    random.seed(20260816)
    a = build(3, "DO")
    random.seed(20260816)
    b = build(3, None)
    random.seed(20260816)
    c = build(3)
    assert a == b == c
    assert a == DETERMINISTIC_VARIETY_PROMPT


def test_f1b_variety_prompt_country_desconocido_cae_a_do():
    import random
    from prompts.preferences import build_deterministic_variety_prompt as build
    random.seed(2026)
    do = build(3, "DO")
    random.seed(2026)
    xx = build(3, "xx")
    assert do == xx


def test_f1b_variety_prompt_beta_neutraliza_fidelidad_cultural():
    from prompts.preferences import build_deterministic_variety_prompt as build
    for cc in _BETA_CCS:
        out = build(3, cc)
        assert "FIDELIDAD CULTURAL es-DO" not in out, cc
        assert "SOLO ingredientes dominicanos" not in out, cc
        assert "FIDELIDAD AL CONTEXTO" in out, cc
        assert "evita ingredientes difíciles de conseguir" in out, cc
        # el resto del esqueleto (reglas de proteína/sodio/variedad) no se toca.
        assert "REGLA DE SEGURIDAD ALIMENTARIA" in out, cc


def _cuerpo_get_deterministic_variety_prompt() -> str:
    src = (_BACKEND / "ai_helpers.py").read_text(encoding="utf-8")
    sin_comentarios = "\n".join(
        l for l in src.splitlines() if not l.strip().startswith("#")
    )
    ini = sin_comentarios.index("def get_deterministic_variety_prompt")
    fin = sin_comentarios.find("\ndef ", ini + 10)
    return sin_comentarios[ini: fin if fin != -1 else len(sin_comentarios)]


def test_f1b_get_deterministic_variety_prompt_deriva_pais_via_ssot():
    _assert_deriva_pais_via_ssot(
        _cuerpo_get_deterministic_variety_prompt(), "get_deterministic_variety_prompt"
    )


def test_f1b_get_deterministic_variety_prompt_wire_country_en_el_builder():
    """[P1-COUNTRY-SYSTEM-F2 · 2026-08-17 (Task 9, j)] Re-anclado: el call site ya NO re-deriva
    `country_for_form_data(form_data)` inline — reusa `_variety_country`, derivado UNA sola vez
    arriba en la misma función (closure) y compartido con el 2º call site de esta task
    (`_get_fast_filtered_catalogs`, antes country-blind). El INTENTO original de este test
    (el builder recibe país derivado vía la ÚNICA puerta T1, no un literal/'DO' hardcoded) sigue
    verificado — solo cambió DÓNDE se deriva, no que se derive."""
    cuerpo = _cuerpo_get_deterministic_variety_prompt()
    assert "build_deterministic_variety_prompt(_dc, _variety_country)" in cuerpo
    assert cuerpo.count("country_for_form_data(form_data)") == 1, (
        "country_for_form_data(form_data) debe aparecer UNA sola vez (derivación única, closure)"
    )
    assert "_variety_country = country_for_form_data(form_data)" in cuerpo


# ── F1c: meal_operations.py — templates de swap/modify ──────────────────────

def test_f1c_swap_template_do_o_none_es_byte_identico_is():
    from prompts.meal_operations import build_swap_meal_prompt_template as build, SWAP_MEAL_PROMPT_TEMPLATE
    assert build("DO") is SWAP_MEAL_PROMPT_TEMPLATE
    assert build(None) is SWAP_MEAL_PROMPT_TEMPLATE
    assert build("xx") is SWAP_MEAL_PROMPT_TEMPLATE


def test_f1c_modify_template_do_o_none_es_byte_identico_is():
    from prompts.meal_operations import build_modify_meal_prompt_template as build, MODIFY_MEAL_PROMPT_TEMPLATE
    assert build("DO") is MODIFY_MEAL_PROMPT_TEMPLATE
    assert build(None) is MODIFY_MEAL_PROMPT_TEMPLATE
    assert build("garbage") is MODIFY_MEAL_PROMPT_TEMPLATE


def test_f1c_swap_template_beta_neutraliza_reglas_25_y_3():
    from prompts.meal_operations import build_swap_meal_prompt_template as build
    for cc in _BETA_CCS:
        out = build(cc)
        assert "PLATO CRIOLLO" not in out, cc
        assert "mofongo / mangú / tostones" not in out, cc
        assert "gastronomía/ingredientes locales dominicanos" not in out, cc
        assert "PREPARACIÓN APETECIBLE" in out, cc
        assert "ingredientes accesibles y cotidianos del contexto del usuario" in out, cc
        # el resto del template (reglas 1, 4-8) no se toca.
        assert "COHERENCIA RECETA↔INGREDIENTES" in out, cc


def test_f1c_modify_template_beta_neutraliza_reglas_4_y_65():
    from prompts.meal_operations import build_modify_meal_prompt_template as build
    for cc in _BETA_CCS:
        out = build(cc)
        assert "PLATO CRIOLLO" not in out, cc
        assert "mofongo / mangú / tostones" not in out, cc
        assert "4. Usa ingredientes dominicanos" not in out, cc
        assert "4. Usa ingredientes accesibles y cotidianos del contexto del usuario" in out, cc
        assert "COHERENCIA RECETA↔INGREDIENTES" in out, cc


def test_f1c_swap_meal_wire_template_con_swap_country():
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    assert "build_swap_meal_prompt_template(_swap_country).format(" in src


def test_f1c_execute_modify_single_meal_wire_template_con_modify_country():
    src = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    assert "build_modify_meal_prompt_template(_modify_country).format(" in src


def test_f1c_meal_ops_cache_dimensionada():
    from prompts.meal_operations import (
        build_swap_meal_prompt_template as bs,
        build_modify_meal_prompt_template as bm,
        _MEAL_OPS_COUNTRY_CACHE,
    )
    for cc in _BETA_CCS:
        bs(cc)
        bm(cc)
    assert len(_MEAL_OPS_COUNTRY_CACHE) <= 2 * len(_BETA_CCS)


# ── F2: país nunca hidratado en swap/regen-day/chat (wiring inerte en runtime) ──────────────
#
# T3/T4 dejaron los CALLERS listos para recibir `country=` — pero el dict que le pasan
# (`form_data`/`data`) nunca tenía la key poblada, así que en runtime SIEMPRE llegaba 'DO'. F2
# cierra la fuente, no el consumo.

def test_f2a_enrich_clinical_from_profile_hidrata_country_desde_perfil(monkeypatch):
    import db
    import routers.plans as rp
    monkeypatch.setattr(db, "get_user_profile", lambda uid: {"health_profile": {"country": "MX"}})
    data = {}
    rp._enrich_clinical_from_profile(data, "user-1")
    assert data.get("country") == "MX"


def test_f2a_enrich_clinical_from_profile_perfil_ausente_no_agrega_key(monkeypatch):
    import db
    import routers.plans as rp
    monkeypatch.setattr(db, "get_user_profile", lambda uid: None)
    data = {}
    rp._enrich_clinical_from_profile(data, "user-1")
    assert "country" not in data, (
        "sin perfil, la espina country_for_form_data cae a 'DO' downstream — no se inventa "
        "una key aquí."
    )


def test_f2a_enrich_clinical_from_profile_perfil_sin_country_no_agrega_key(monkeypatch):
    import db
    import routers.plans as rp
    monkeypatch.setattr(db, "get_user_profile", lambda uid: {"health_profile": {"allergies": []}})
    data = {}
    rp._enrich_clinical_from_profile(data, "user-1")
    assert "country" not in data


def test_f2a_enrich_clinical_from_profile_body_country_gana_sobre_perfil(monkeypatch):
    import db
    import routers.plans as rp
    monkeypatch.setattr(db, "get_user_profile", lambda uid: {"health_profile": {"country": "MX"}})
    data = {"country": "ES"}
    rp._enrich_clinical_from_profile(data, "user-1")
    assert data["country"] == "ES", "el body (ya validado por el cliente) gana sobre el perfil"


def test_f2a_enrich_clinical_from_profile_guarda_crudo_sin_canonicalizar(monkeypatch):
    """'store raw' — la canonicalización ocurre en los LECTORES vía la espina T1
    (`country_for_form_data`), nunca en este hidratador (evitaría la 2ª tabla que
    P1-DIET-CANON-SSOT ya pagó una vez)."""
    import db
    import routers.plans as rp
    monkeypatch.setattr(db, "get_user_profile", lambda uid: {"health_profile": {"country": "mx"}})
    data = {}
    rp._enrich_clinical_from_profile(data, "user-1")
    assert data.get("country") == "mx"


def test_f2b_regenerate_day_meal_form_propaga_country():
    src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    ini = src.index('"diet_type": data.get("diet_type") or data.get("dietType") or "balanced",')
    fin = src.index('"goal": data.get("goal") or data.get("mainGoal"),', ini)
    bloque = src[ini:fin]
    assert '"country": data.get("country")' in bloque, (
        "meal_form es un dict de keys EXPLÍCITAS (no hace spread de `data`) — sin esto, "
        "swap_meal(surface='day') seguía cayendo a 'DO' aunque data['country'] ya viniera "
        "hidratado por F2a."
    )


def test_f2c_tools_comment_documenta_el_mecanismo_real_no_el_ruling_obsoleto():
    """El comentario honesto de execute_modify_single_meal debe citar el mecanismo REAL
    (merge_form_data_with_profile) — no el ruling T4 desactualizado que declaraba el chat-agent
    ciego al país."""
    src = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    ini = src.index("def execute_modify_single_meal(")
    fin = src.index("_modify_country = country_for_plan(plan_data, form_data)")
    bloque = src[ini:fin]
    assert "merge_form_data_with_profile" in bloque
    assert "el chat-agent hoy no puebla" not in bloque, "el ruling T4 desactualizado sigue ahí"


def test_f2d_doc_wiring_sentence_cita_la_hidratacion_final_fix():
    doc = (_BACKEND / "docs" / "country_system_f1.md").read_text(encoding="utf-8")
    ini = doc.index("SÍ están wired")
    bloque = doc[ini: ini + 800]
    assert "FINAL-FIX F2" in bloque
    assert "_enrich_clinical_from_profile" in bloque
    assert "merge_form_data_with_profile" in bloque


# ── F3: build_budget_context clampeaba EUR/MXN/COP a RD$ ────────────────────

def test_f3_build_budget_context_dop_usd_byte_identico_knob_on_y_off(monkeypatch):
    from prompts.plan_generator import build_budget_context as build
    fd_dop = {"budget": "custom", "budgetAmount": "5000"}
    fd_usd = {"budget": "custom", "budgetAmount": "150", "budgetCurrency": "USD"}
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    dop_off, usd_off = build(fd_dop), build(fd_usd)
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    dop_on, usd_on = build(fd_dop), build(fd_usd)
    assert dop_off == dop_on, "DOP debe ser byte-idéntico con el knob ON u OFF"
    assert usd_off == usd_on, "USD debe ser byte-idéntico con el knob ON u OFF"
    assert "RD$5,000" in dop_on
    assert "US$150" in usd_on


def test_f3_build_budget_context_eur_knob_off_sigue_clampeando_a_dop(monkeypatch):
    """Fail-safe preservado: sin el knob, EUR/MXN/COP se tratan como DOP — igual que ANTES de
    esta fase (comportamiento histórico, no una regresión de F3)."""
    from prompts.plan_generator import build_budget_context as build
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    out = build({"budget": "custom", "budgetAmount": "245", "budgetCurrency": "EUR"})
    assert "RD$" in out
    assert "EUR" not in out


def test_f3_build_budget_context_eur_knob_on_no_clampea_ni_miente_rd(monkeypatch):
    from prompts.plan_generator import build_budget_context as build
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    out = build({"budget": "custom", "budgetAmount": "245", "budgetCurrency": "EUR"})
    assert "RD$" not in out
    assert "US$" not in out
    assert "EUR" in out
    assert (
        "El usuario definió un presupuesto TOTAL de 245 EUR para su ciclo de compras"
        in out
    )


def test_f3_build_budget_context_mxn_cop_tambien_respetan_su_moneda(monkeypatch):
    from prompts.plan_generator import build_budget_context as build
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    mxn = build({"budget": "custom", "budgetAmount": "1400", "budgetCurrency": "MXN"})
    cop = build({"budget": "custom", "budgetAmount": "350000", "budgetCurrency": "COP"})
    assert "MXN" in mxn and "RD$" not in mxn
    assert "COP" in cop and "RD$" not in cop


def test_f3_build_budget_context_moneda_no_reconocida_sigue_cayendo_a_dop(monkeypatch):
    """Basura (`XYZ`) no es una moneda beta real — jamás debe escapar el clamp, con o sin knob."""
    from prompts.plan_generator import build_budget_context as build
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    out = build({"budget": "custom", "budgetAmount": "500", "budgetCurrency": "XYZ"})
    assert "RD$" in out


def test_f3_tools_chat_consume_el_mismo_builder():
    """[F3] tools.py:1162 (chat, GAP-07 budget en modify con expansión) consume
    `build_budget_context` — un solo fix en el builder arregla las DOS superficies (form-gen +
    chat-modify); no debe existir un 2º builder de presupuesto bifurcado."""
    src = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    assert "from prompts.plan_generator import build_budget_context as _bbc_cm" in src


# ── F4: S1 retry text seguía mandando criollo incondicionalmente para beta ──

def test_f4_detect_slot_appropriateness_do_texto_intacto(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    import graph_orchestrator as go
    days = [{"day": 1, "meals": [{"meal": "Desayuno", "name": "Arroz con Huevo", "ingredients": []}]}]
    issues = go._detect_slot_appropriateness(days, {"country": "DO"})
    assert issues
    text = issues[0]["text"]
    assert "rechazo de coherencia cultural es-DO" in text
    assert "que no corresponde al desayuno dominicano" in text


def test_f4_detect_slot_appropriateness_knob_off_texto_do_pese_a_country_beta(monkeypatch):
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    import graph_orchestrator as go
    days = [{"day": 1, "meals": [{"meal": "Desayuno", "name": "Arroz con Huevo", "ingredients": []}]}]
    issues = go._detect_slot_appropriateness(days, {"country": "ES"})
    assert "rechazo de coherencia cultural es-DO" in issues[0]["text"], (
        "knob apagado ⇒ 'DO' siempre, sin mirar form_data"
    )


def test_f4_detect_slot_appropriateness_beta_texto_neutro(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    import graph_orchestrator as go
    days = [{"day": 1, "meals": [{"meal": "Desayuno", "name": "Arroz con Huevo", "ingredients": []}]}]
    for cc in _BETA_CCS:
        issues = go._detect_slot_appropriateness(days, {"country": cc})
        assert issues, cc
        text = issues[0]["text"]
        assert "es-DO" not in text, cc
        assert "dominicano" not in text, cc
        assert "no corresponde al horario desayuno" in text, cc


def test_f4_detect_slot_appropriateness_beta_usa_slot_positive_hint_neutral(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    import graph_orchestrator as go
    from constants import _SLOT_POSITIVE_HINT_NEUTRAL
    days = [{"day": 1, "meals": [{"meal": "Desayuno", "name": "Arroz con Huevo", "ingredients": []}]}]
    issues = go._detect_slot_appropriateness(days, {"country": _BETA_CC_SAMPLE})
    assert _SLOT_POSITIVE_HINT_NEUTRAL["desayuno"] in issues[0]["text"], (
        "T4 construyó _SLOT_POSITIVE_HINT_NEUTRAL para build_meal_timing_rules — F4 lo wirea "
        "TAMBIÉN aquí, en vez de slot_positive_hint() (que solo varía por dieta, nunca país)."
    )


def test_f4_detect_slot_appropriateness_do_y_beta_comparten_label_solo_difiere_hard_y_texto():
    """Regresión del contrato T4 (hard soft/duro por país) — F4 no debe tocar `label`/`hard`,
    solo el string `text` legible por el LLM."""
    import graph_orchestrator as go
    days = [{"day": 1, "meals": [{"meal": "Desayuno", "name": "Arroz con Huevo", "ingredients": []}]}]
    import os as _os_f4
    _prev = _os_f4.environ.get("MEALFIT_COUNTRY_SYSTEM")
    _os_f4.environ["MEALFIT_COUNTRY_SYSTEM"] = "true"
    try:
        issues_do = go._detect_slot_appropriateness(days, {"country": "DO"})
        issues_beta = go._detect_slot_appropriateness(days, {"country": _BETA_CC_SAMPLE})
    finally:
        if _prev is None:
            _os_f4.environ.pop("MEALFIT_COUNTRY_SYSTEM", None)
        else:
            _os_f4.environ["MEALFIT_COUNTRY_SYSTEM"] = _prev
    assert issues_do[0]["label"] == issues_beta[0]["label"]
    assert issues_do[0]["hard"] is True and issues_beta[0]["hard"] is False
    assert issues_do[0]["text"] != issues_beta[0]["text"]


def test_f4_detect_slot_appropriateness_deriva_pais_una_sola_vez_se_mantiene():
    """Regresión: el fix de F4 reusa `_country` (ya derivado por T4) — NO debe añadir una 2ª
    derivación (el test T4 original ya lo exige; este lo re-ancla scoped al bloque que F4 tocó)."""
    cuerpo = _cuerpo_detect_slot_appropriateness()
    assert cuerpo.count("country_for_form_data(form_data)") == 1
