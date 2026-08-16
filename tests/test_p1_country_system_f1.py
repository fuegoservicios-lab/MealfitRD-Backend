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
#       prohibido es incidental a la regla).
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
_DOMINICAN_TOKEN_RX = re.compile(r"locrio|mofongo|mangú|bandera:", re.IGNORECASE)


def _scoped_out_sin_s16(out: str) -> str:
    """Excluye §16 (CONTRATO EXACTO DEL VALIDADOR DE HORARIO, derivado de
    constants.build_meal_timing_rules) del texto escaneado — ruling del controller: MOVIDO a
    Task 4, no es prompt-directive de day_generator sino el espejo del slot SSOT que T4
    parameteriza. Tocarlo aquí estaría fuera del scope de este fix-round."""
    i16 = out.index("16. CONTRATO EXACTO DEL VALIDADOR DE HORARIO")
    i17 = out.index("\n17. PRESUPUESTO DE SODIO")
    return out[:i16] + out[i17:]


def test_finding5_guard_case_insensitive_sin_sobrevivientes_no_documentados():
    """Guard HONESTO (no solo verde): escanea el render beta case-insensitive por los 4 tokens
    duros, excluye §16 (T4) y los sobrevivientes DOCUMENTADOS arriba (clase b/c), y falla si
    queda CUALQUIER OTRO hit — el mecanismo que impide que un futuro edit reintroduzca una orden
    dominicana sin que nadie se entere."""
    from prompts.day_generator import build_day_generator_system_prompt as build
    scoped = _scoped_out_sin_s16(build("balanced", "ES"))

    for survivor in _B_CLASS_PROHIBICIONES + _C_CLASS_CATALOG_ENUM:
        assert survivor in scoped, (
            f"sobreviviente documentado ya no existe verbatim en el render — o cambió el texto "
            f"fuente (actualiza esta lista) o ya se arregló (muévelo a los tests de arriba): "
            f"{survivor[:70]!r}"
        )
        scoped = scoped.replace(survivor, "", 1)

    hits = _DOMINICAN_TOKEN_RX.findall(scoped)
    assert not hits, f"sobrevivientes NO documentados de contenido dominicano: {hits}"


def test_finding5_guard_vale_tambien_para_vegan():
    """Los targets del fix-round 1 son diet-invariantes — el guard debe sostenerse igual sobre
    vegan, no solo balanced."""
    from prompts.day_generator import build_day_generator_system_prompt as build
    scoped = _scoped_out_sin_s16(build("vegan", "ES"))

    for survivor in _B_CLASS_PROHIBICIONES + _C_CLASS_CATALOG_ENUM:
        assert survivor in scoped, f"sobreviviente ausente en vegan: {survivor[:70]!r}"
        scoped = scoped.replace(survivor, "", 1)

    hits = _DOMINICAN_TOKEN_RX.findall(scoped)
    assert not hits, f"sobrevivientes NO documentados (vegan): {hits}"
