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
    — sigue leyendo SLOT_INAPPROPRIATE_FOODS. Ningún caller preexistente (tools.py chat-backstop,
    plan_gym.py scoring, agent.py backstop) pasa este argumento."""
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

def test_get_avg_meal_hour_usa_offset_resuelto_con_signo_mas(monkeypatch):
    """[T5] `consumed_at` es timestamptz (no NAIVE) — el doble AT TIME ZONE previo neteaba
    +offset, no -offset (forense Neon 2026-08-16). La sustitución usa `+` para ser BYTE-IDÉNTICA
    a offset=240; un '-' aquí sería una regresión silenciosa de conducta (aunque 'arreglaría' un
    bug no pedido por T5). Ver task-5-report.md para el argumento completo."""
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
    assert query.count("consumed_at + make_interval(mins => %s)") == 2, (
        "esperaba el signo '+' (preserva-bug) en HOUR y MINUTE"
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

def test_gate_currencies_son_exactamente_eur_mxn_cop():
    assert set(nc._BUDGET_CYCLE_FLOOR_DEFAULTS_BY_CURRENCY.keys()) == {"EUR", "MXN", "COP"}


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
    """DOP/USD/basura: delega en `_budget_cycle_floor_dop` — el piso histórico
    NO se toca ni se reimplementa por segunda vez."""
    for days in (7, 15, 30):
        assert nc._budget_cycle_floor_for_currency(days, "XYZ") == nc._budget_cycle_floor_dop(days)
        assert nc._budget_cycle_floor_for_currency(days, "DOP") == nc._budget_cycle_floor_dop(days)
        assert nc._budget_cycle_floor_for_currency(days, "USD") == nc._budget_cycle_floor_dop(days)


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


def test_knob_on_pais_es_eur_bajo_piso_rechaza(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    ok, detail = nc.validate_budget_sufficient(_budget_form("EUR", 1, country="ES"))
    assert ok is False
    assert detail["currency"] == "EUR"
    assert detail["min_budget"] > 0
    assert "EUR" in detail["message"]
    assert "RD$" not in detail["message"]


def test_knob_on_pais_es_eur_sobre_piso_acepta(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    ok, detail = nc.validate_budget_sufficient(_budget_form("EUR", 50000, country="ES"))
    assert ok is True and detail is None


def test_knob_on_mx_mxn_bajo_piso_rechaza_con_mxn(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    ok, detail = nc.validate_budget_sufficient(_budget_form("MXN", 1, country="MX"))
    assert ok is False
    assert detail["currency"] == "MXN"
    assert "MXN" in detail["message"]
    assert "RD$" not in detail["message"]


def test_knob_on_co_cop_bajo_piso_rechaza_con_cop(monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    ok, detail = nc.validate_budget_sufficient(_budget_form("COP", 1, country="CO"))
    assert ok is False
    assert detail["currency"] == "COP"
    assert "COP" in detail["message"]
    assert "RD$" not in detail["message"]


@pytest.mark.parametrize("currency,country", [("EUR", "ES"), ("MXN", "MX"), ("COP", "CO")])
def test_mensaje_nuevo_nunca_hardcodea_rd_simbolo(monkeypatch, currency, country):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    ok, detail = nc.validate_budget_sufficient(_budget_form(currency, 1, country=country))
    assert ok is False
    assert "RD$" not in detail["message"], (
        f"El mensaje para {currency} contiene 'RD$' hardcodeado — viola el contrato de Task 6."
    )
    assert currency in detail["message"]


def test_usd_sigue_intacto_con_knob_on(monkeypatch):
    """El país-system ON no debe tocar el mecanismo USD histórico (conversión
    por _budget_usd_to_dop)."""
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    ok_off, detail_off = nc.validate_budget_sufficient(_budget_form("USD", 5))
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    ok_on, detail_on = nc.validate_budget_sufficient(_budget_form("USD", 5))
    assert ok_off is False
    assert ok_on is False
    assert detail_off == detail_on


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
    assert ok is False
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
    assert ok is False
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


# ── QBudget.jsx: parser — dark intacto, lit condicionado, wiring completo ────

def _read_qbudget_jsx() -> str:
    path = _FRONTEND / "src" / "components" / "assessment" / "questions" / "QBudget.jsx"
    if not path.exists():
        pytest.skip(f"QBudget.jsx no existe en {path}")
    return path.read_text(encoding="utf-8")


def test_qbudget_dark_toggle_literales_originales_intactos():
    """[dark anchor] Los DOS literales que `test_p1_budget_custom.py` YA
    ancla (`test_budget_currency_toggle_defaults_to_dop`) deben seguir
    presentes verbatim — condición necesaria para que, con
    COUNTRY_SYSTEM_UI=false, el toggle sea EXACTAMENTE [DOP, USD] de hoy."""
    src = _read_qbudget_jsx()
    assert re.search(r"value:\s*'DOP'\s*,\s*label:\s*'RD\$'", src)
    assert re.search(r"value:\s*'USD'\s*,\s*label:\s*'US\$'", src)


def test_qbudget_importa_country_system_ui_y_coerce_country():
    src = _read_qbudget_jsx()
    assert re.search(
        r"import\s*\{[^}]*COUNTRY_SYSTEM_UI[^}]*\}\s*from\s*['\"]\.\./\.\./\.\./config/countries['\"]",
        src,
    ), "QBudget debe importar COUNTRY_SYSTEM_UI de config/countries (el flag dark del frontend)."
    assert "coerceCountry" in src


def test_qbudget_mapa_beta_currency_por_pais():
    src = _read_qbudget_jsx()
    assert re.search(r"ES:\s*'EUR'", src)
    assert re.search(r"MX:\s*'MXN'", src)
    assert re.search(r"CO:\s*'COP'", src)


def test_qbudget_usa_helper_puro_exportado_para_las_opciones():
    """El helper puro (probado en vitest sin montar el componente) debe
    EXISTIR y ser lo que arma `currencyOptions` — no una copia inline que
    pueda driftear de lo que el test de JS ejercita."""
    src = _read_qbudget_jsx()
    assert re.search(r"export (function|const) currencyOptionsForCountry", src), (
        "El helper puro currencyOptionsForCountry debe estar exportado para el test vitest."
    )
    assert src.count("currencyOptionsForCountry(") >= 2, (
        "El componente debe LLAMAR a currencyOptionsForCountry (definición + uso), no solo definirlo."
    )


def test_qbudget_currency_symbol_ramifica_por_beta_currency():
    src = _read_qbudget_jsx()
    m = re.search(r"const currencySymbol = ([\s\S]*?);\n", src)
    assert m, "No se pudo aislar `const currencySymbol = ...;` en QBudget.jsx"
    body = m.group(1)
    assert "'USD'" in body and "US$" in body, "El símbolo USD debe seguir intacto."
    assert "betaCurrency" in body, "El símbolo debe ramificar por `betaCurrency` para EUR/MXN/COP."
    assert "'RD$'" in body, "El fallback DOP debe seguir siendo 'RD$'."


def test_qbudget_placeholder_mxn_cop_con_ejemplo_propio():
    src = _read_qbudget_jsx()
    assert "budgetCurrency === 'MXN'" in src
    assert "budgetCurrency === 'COP'" in src
    assert "Ej. 2000" in src
    assert "Ej. 400000" in src


def test_qbudget_aria_label_cubre_monedas_nuevas_y_conserva_originales():
    src = _read_qbudget_jsx()
    assert "Presupuesto total en euros" in src
    assert "Presupuesto total en pesos mexicanos" in src
    assert "Presupuesto total en pesos colombianos" in src
    # Byte-identidad del dark path: los 2 originales siguen ahí.
    assert "Presupuesto total en dólares" in src
    assert "Presupuesto total en pesos dominicanos" in src
