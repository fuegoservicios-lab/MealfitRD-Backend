"""[P1-COUNTRY-SYSTEM-F2 · 2026-08-17] Fase 2 del sistema de países: Task 1 — el harness
semántico + auditoría de drops (LA MEDICIÓN).

Fase 1 (`test_p1_country_system_f1.py`) dejó el motor listo para dejar de forzar lo criollo en
beta, con el knob maestro apagado (byte-identidad DO). Fase 2 es catálogo y seguridad por país:
Addendum del dueño 2026-08-17 §1 — "catálogo por completitud MEDIDA, no por cuota" — así que ANTES
de dar de alta un solo alimento (T5-T8), hace falta MEDIR el hueco real. Esta task construye ese
instrumento: `backend/scripts/country_catalog_gap.py`, un harness CLI sin LLM que pasa listas
curadas de alimentos típicos por país por el pipeline REAL de resolución de ingredientes
(`shopping_calculator.normalize_name`) contra el catálogo vivo, y clasifica cada ítem en
RESUELVE-BIEN / SUSTITUCION-SILENCIOSA / DROP — más un modo `--rd-drops` que agrega la telemetría
`record_verified_only_drop` para el top-up de RD.

Spec: `docs/superpowers/specs/2026-08-16-sistema-paises-design.md` §Fase 2 + Addendum §1.
Plan: `docs/superpowers/plans/2026-08-17-paises-fase-2.md` Task 1.

Secciones:
  A. Unit — `classify_food` con `shopping_calculator.normalize_name`/`_get_verified_shopping_name_set`
     MOCKEADOS (los 3 veredictos + el edge fuzzy 0.857 documentado en el brief de esta task +
     el caso semántico "debe NUNCA clasificar RESUELVE-BIEN").
  B. Parser — el script abre el pool ANTES de tocar el catálogo (ambos modos); reusa los
     resolvers de producción (no reimplementa difflib/cosine_similarity); CLI vía argparse.
  C. Smoke — las 5 listas curadas (≥60 items, sin duplicados).
  D. Unit — `_aggregate_rd_drops` (función pura, sin DB).
  E. Task 2 — preselección IANA: paridad TZ→país con COUNTRY_PROFILES.
  F. Task 3 — coach en tu idioma, comida en español.
  G. Task 4 — los 4 vocabularios de alérgenos/dieta ×país + drift RD (mejillón/vieira/arenque)
     + el guard de paridad + el alta-hook contra el catálogo vivo. T4 consideró y RECHAZÓ sumar
     'avena' a gluten (colisión con P1-ALLERGEN-NEGATION-EXCUSE, solo-prefijo); fix-round 1
     (post-review) REVIRTIÓ esa decisión con una excusa FORWARD scoped-a-gluten — ver G1.

Task 1-F (secciones A-F): ningún test toca Neon — todo lo que necesita catálogo/DB va mockeado
vía `monkeypatch`. La corrida REAL contra el catálogo vivo (`--country ES`, `--rd-drops`) es un
paso manual documentado en el reporte de la task, no parte de la suite.

Task 4 (sección G) introduce la ÚNICA excepción: `test_backstop_conoce_cada_alimento_peligroso_del_catalogo_vivo`
(el alta-hook, contrato T4 ítem 3) SÍ toca Neon de verdad —`master_ingredients` real, read-only,
pool abierto explícitamente— porque su propósito específico es verificar que el backstop conoce
CADA alimento del catálogo que existe HOY, no una fixture. Marcado `@pytest.mark.e2e` (igual que
el resto de la suite, `tests/conftest.py::_guard_test_writes_to_prod`) para que el gate rápido
(`-m "not e2e"`) no dependa de conectividad DB; se salta con `pytest.skip` si el pool no está
disponible, nunca falla por infraestructura ausente.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import re
from pathlib import Path

import pytest

import constants
import prompts.chat_agent as chat_agent_prompts

_BACKEND = Path(__file__).resolve().parent.parent
_FRONTEND = _BACKEND.parent / "frontend"
_SCRIPT = _BACKEND / "scripts" / "country_catalog_gap.py"
_AGENT_PY = _BACKEND / "agent.py"
_PROACTIVE_AGENT_PY = _BACKEND / "proactive_agent.py"
_CHAT_AGENT_PY = _BACKEND / "prompts" / "chat_agent.py"
_HELP_BOT_PY = _BACKEND / "prompts" / "help_bot.py"
_DISH_TEMPLATES_ES_JSON = _BACKEND / "data" / "dish_templates_es.json"
_DISH_TEMPLATES_MX_JSON = _BACKEND / "data" / "dish_templates_mx.json"
_DISH_TEMPLATES_CO_JSON = _BACKEND / "data" / "dish_templates_co.json"
_CRON_TASKS_PY = _BACKEND / "cron_tasks.py"
_SHOPPING_CALCULATOR_PY = _BACKEND / "shopping_calculator.py"


def _load():
    """Carga `country_catalog_gap.py` como módulo fresco (mismo patrón que
    `test_p0_clinical_validation_export.py`). No toca Neon: el script solo abre el pool
    dentro de `_open_pool()`/`run_*_mode()`, nunca a nivel de módulo."""
    spec = importlib.util.spec_from_file_location("country_catalog_gap", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _source() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def _source_sin_comentarios() -> str:
    # CRLF-safe: `splitlines()` trata `\r\n` y `\n` por igual (no se asume un separador).
    return "\n".join(
        line for line in _source().splitlines() if not line.strip().startswith("#")
    )


def test_script_existe_y_expone_el_contrato():
    assert _SCRIPT.exists(), "backend/scripts/country_catalog_gap.py debe existir"
    mod = _load()
    for name in (
        "classify_food", "run_country_mode", "run_rd_drops_mode", "_aggregate_rd_drops",
        "_open_pool", "CURATED_FOODS_BY_COUNTRY", "main",
    ):
        assert hasattr(mod, name), f"falta {name}"


# ════════════════════════════════════════════════════════════════════════════════════════════
# A. Unit — classify_food, resolvers mockeados
# ════════════════════════════════════════════════════════════════════════════════════════════

def test_resuelve_bien_lexico_exact(monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod.sc, "normalize_name", lambda name: "Tomate")
    monkeypatch.setattr(mod.sc, "_get_verified_shopping_name_set", lambda: {"tomate"})

    r = mod.classify_food("tomate", semantic_tier_active=True)

    assert r["verdict"] == "RESUELVE-BIEN"
    assert r["tier"] == "exact"
    assert r["matched"] == "Tomate"
    assert r["score"] is None


def test_resuelve_bien_lexico_synonym(monkeypatch):
    """El query NO es igual (normalizado) al canónico devuelto → vino de un alias, no de
    una igualdad literal. Ningún log de fuzzy/semantic disparó → tier léxico (INTENTO 1-4)."""
    mod = _load()
    monkeypatch.setattr(mod.sc, "normalize_name", lambda name: "Tomate")
    monkeypatch.setattr(mod.sc, "_get_verified_shopping_name_set", lambda: {"tomate"})

    r = mod.classify_food("jitomate", semantic_tier_active=True)

    assert r["verdict"] == "RESUELVE-BIEN"
    assert r["tier"] == "synonym"
    assert r["matched"] == "Tomate"


def test_resuelve_bien_fuzzy_couscous_cuscus_0_857(monkeypatch):
    """Edge documentado en el brief de Task 1: 'couscous'→'cuscus' con ratio 0.857 debe
    aterrizar RESUELVE-BIEN(fuzzy). El clasificador NO re-valida ese ratio contra ningún
    umbral propio (0.857 es incluso MENOR que `_FUZZY_MATCH_THRESHOLD`=0.87 de producción,
    a propósito): la decisión de "hubo match" ya la tomó `normalize_name` — este test prueba
    que el clasificador LEE esa decisión desde el log ('[Fuzzy Match]') en vez de
    recalcularla, que es justamente el contrato "no reimplementar scoring"."""
    mod = _load()

    def fake_normalize_name(name):
        logging.info("🔤 [Fuzzy Match] 'couscous' -> 'Cuscús' (ratio 0.857)")
        return "Cuscús"

    monkeypatch.setattr(mod.sc, "normalize_name", fake_normalize_name)
    monkeypatch.setattr(mod.sc, "_get_verified_shopping_name_set", lambda: {"cuscus"})

    r = mod.classify_food("couscous", semantic_tier_active=True)

    assert r["verdict"] == "RESUELVE-BIEN"
    assert r["tier"] == "fuzzy"
    assert r["matched"] == "Cuscús"
    assert r["score"] == pytest.approx(0.857)


def test_sustitucion_silenciosa_paella_papaya_nunca_es_resuelve_bien(monkeypatch):
    """Estilo del brief: 'paella'→'papaya' a score 0.50 (un no-match evidente) NO debe
    clasificar jamás como RESUELVE-BIEN — vino del tier semántico (INTENTO 6), que por
    diseño es el que puede convertir un alimento en "otra cosa sin ruido" (spec §Fase 2).
    Cualquier resolución vía ese tier es SUSTITUCION-SILENCIOSA, sin excepción por score."""
    mod = _load()

    def fake_normalize_name(name):
        logging.info("🧠 [Semantic Search] Resuelto: 'paella' -> 'Papaya' con score 0.500")
        return "Papaya"

    monkeypatch.setattr(mod.sc, "normalize_name", fake_normalize_name)
    monkeypatch.setattr(mod.sc, "_get_verified_shopping_name_set", lambda: {"papaya"})

    r = mod.classify_food("paella", semantic_tier_active=True)

    assert r["verdict"] == "SUSTITUCION-SILENCIOSA"
    assert r["verdict"] != "RESUELVE-BIEN"
    assert r["matched"] == "Papaya"
    assert r["score"] == pytest.approx(0.50)


def test_drop_sin_match_verificado(monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod.sc, "normalize_name", lambda name: "Cosa Sin Catalogar")
    monkeypatch.setattr(mod.sc, "_get_verified_shopping_name_set", lambda: {"papaya", "cuscus"})

    r = mod.classify_food("laurel", semantic_tier_active=True)

    assert r["verdict"] == "DROP"
    assert r["matched"] is None
    assert r["tier"] is None


def test_drop_con_tier_semantico_inactivo_marca_unknown(monkeypatch):
    """Sin COHERE_API_KEY (o init fallido) el tier 6 nunca corre dentro de `normalize_name` —
    un DROP en esas condiciones podría, en realidad, ser una sustitución silenciosa nunca
    evaluada. El contrato pide clasificar ESE tier como UNKNOWN y decirlo (brief Task 1)."""
    mod = _load()
    monkeypatch.setattr(mod.sc, "normalize_name", lambda name: "Cosa Sin Catalogar")
    monkeypatch.setattr(mod.sc, "_get_verified_shopping_name_set", lambda: set())

    r = mod.classify_food("comino", semantic_tier_active=False)

    assert r["verdict"] == "DROP"
    assert r["semantic_tier_status"] == "unknown"


def test_drop_con_tier_semantico_activo_marca_checked(monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod.sc, "normalize_name", lambda name: "Cosa Sin Catalogar")
    monkeypatch.setattr(mod.sc, "_get_verified_shopping_name_set", lambda: set())

    r = mod.classify_food("comino", semantic_tier_active=True)

    assert r["semantic_tier_status"] == "checked"


def test_resolver_exception_clasifica_drop_sin_reventar(monkeypatch):
    mod = _load()

    def boom(name):
        raise RuntimeError("catálogo caído")

    monkeypatch.setattr(mod.sc, "normalize_name", boom)

    r = mod.classify_food("cualquiera", semantic_tier_active=True)

    assert r["verdict"] == "DROP"


# ════════════════════════════════════════════════════════════════════════════════════════════
# B. Parser — pool-open, reuso de resolvers de producción, CLI
# ════════════════════════════════════════════════════════════════════════════════════════════

def test_open_pool_llama_connection_pool_open():
    src = _source_sin_comentarios()
    assert "def _open_pool" in src
    assert "connection_pool.open()" in src


def test_run_country_mode_abre_el_pool_antes_del_cache_semantico():
    """`_open_pool()` DEBE aparecer antes de `get_semantic_cache(` en el cuerpo de
    `run_country_mode` — de lo contrario `get_master_ingredients()` (llamada internamente
    por el cache semántico) mide el catálogo VACÍO en vez del real. Mutación (ejecutada
    manualmente durante la implementación de esta task, evidencia en el reporte): comentar
    la línea `_open_pool()` de `run_country_mode` deja este test en RED."""
    src = _source_sin_comentarios()
    ini = src.index("def run_country_mode")
    fin = src.index("\ndef ", ini + 10)
    cuerpo = src[ini:fin]

    assert "_open_pool()" in cuerpo, "run_country_mode no abre el pool"
    pos_open = cuerpo.index("_open_pool()")
    pos_cache = cuerpo.index("get_semantic_cache(")
    assert pos_open < pos_cache, "_open_pool() debe preceder a get_semantic_cache()"


def test_run_rd_drops_mode_abre_el_pool_antes_de_la_query():
    src = _source_sin_comentarios()
    ini = src.index("def run_rd_drops_mode")
    fin = src.index("\ndef ", ini + 10)
    cuerpo = src[ini:fin]

    assert "_open_pool()" in cuerpo, "run_rd_drops_mode no abre el pool"
    pos_open = cuerpo.index("_open_pool()")
    pos_query = cuerpo.index("execute_sql_query(")
    assert pos_open < pos_query, "_open_pool() debe preceder a la query de pipeline_metrics"


def test_usa_los_resolvers_de_produccion():
    src = _source_sin_comentarios()
    assert "import shopping_calculator as sc" in src
    assert "sc.normalize_name(" in src
    assert "sc._get_verified_shopping_name_set(" in src
    assert "sc._FUZZY_MATCH_THRESHOLD" in src


def test_no_reimplementa_scoring():
    """Guard estructural del contrato "REUSE the production functions, do not reimplement
    scoring": el script no debe traer su propia calculadora de similitud — ni `difflib`
    (el motor del tier fuzzy) ni `cosine_similarity` (el motor del tier semántico). El
    clasificador determina el tier LEYENDO el log que `normalize_name` ya emite
    (`_resolve_with_tier`/`_TierLogCapture`), nunca recalculando un ratio.

    Busca el USO real como TOKEN de código (`import difflib` al inicio de línea, o
    `difflib.<algo>(`/`cosine_similarity(` como llamada), no la palabra suelta — el propio
    docstring del módulo la menciona en PROSA (dos veces: describiendo el pipeline que
    reusa, y documentando esta misma ausencia) y esa mención no es una violación del
    contrato. `(?m)^\\s*import difflib\\b` exige que sea una sentencia real, no texto."""
    src = _source_sin_comentarios()
    assert re.search(r"(?m)^\s*import difflib\b", src) is None, "no debe importar difflib"
    assert re.search(r"\bdifflib\.\w+\(", src) is None, "no debe llamar a difflib.*()"
    assert re.search(r"(?<!def )\bcosine_similarity\(", src) is None, "no debe recalcular cosine_similarity"
    # Contrapartida positiva: el mecanismo real usado SÍ está presente.
    assert "_TierLogCapture" in src
    assert "addHandler" in src


def test_cli_usa_argparse_con_dos_modos_mutuamente_exclusivos():
    src = _source_sin_comentarios()
    assert "import argparse" in src
    assert "add_mutually_exclusive_group(required=True)" in src
    assert '"--country"' in src
    assert '"--rd-drops"' in src


# ════════════════════════════════════════════════════════════════════════════════════════════
# C. Smoke — listas curadas por país
# ════════════════════════════════════════════════════════════════════════════════════════════

def test_listas_curadas_cubren_los_5_paises_beta():
    mod = _load()
    assert set(mod.CURATED_FOODS_BY_COUNTRY) == {"ES", "MX", "CO", "PR", "US"}


@pytest.mark.parametrize("cc", ["ES", "MX", "CO", "PR", "US"])
def test_lista_curada_tiene_al_menos_60_items_sin_duplicados(cc):
    mod = _load()
    items = mod.CURATED_FOODS_BY_COUNTRY[cc]

    assert 60 <= len(items) <= 130, f"{cc}: {len(items)} items — fuera del rango ~60-120 del brief"
    assert all(isinstance(x, str) and x.strip() for x in items), f"{cc} tiene un item vacío/no-str"

    lowered = [x.strip().lower() for x in items]
    dupes = sorted({x for x in lowered if lowered.count(x) > 1})
    assert not dupes, f"{cc} tiene duplicados: {dupes}"


# ════════════════════════════════════════════════════════════════════════════════════════════
# D. Unit — _aggregate_rd_drops (función pura, sin DB)
# ════════════════════════════════════════════════════════════════════════════════════════════

def test_aggregate_rd_drops_suma_a_traves_de_filas():
    mod = _load()
    rows = [
        {"metadata": {"top_verified_only_drops": [["laurel", 3], ["comino", 2]]}},
        {"metadata": {"top_verified_only_drops": [["laurel", 5], ["curcuma", 1]]}},
    ]

    out = mod._aggregate_rd_drops(rows)
    by_food = {d["food"]: d["count"] for d in out}

    assert by_food["laurel"] == 8
    assert by_food["comino"] == 2
    assert by_food["curcuma"] == 1
    assert out[0]["food"] == "laurel"  # ordenado desc por count


def test_aggregate_rd_drops_tolera_metadata_como_json_string():
    """El mismo patrón defensivo que `_creativity_kpi_job` aplica a `dish_quality_report`
    (`isinstance(dqr, str): json.loads(dqr)`) — algunos paths de lectura devuelven la
    columna JSONB ya parseada, otros como string cruda."""
    mod = _load()
    rows = [{"metadata": '{"top_verified_only_drops": [["oregano", 4]]}'}]

    out = mod._aggregate_rd_drops(rows)

    assert out == [{"food": "oregano", "count": 4}]


def test_aggregate_rd_drops_filas_vacias_o_malformadas_no_crashea():
    mod = _load()
    assert mod._aggregate_rd_drops([]) == []
    assert mod._aggregate_rd_drops([{"metadata": None}, {"metadata": {}}]) == []
    assert mod._aggregate_rd_drops([{"metadata": "no es json{{{"}]) == []
    assert mod._aggregate_rd_drops([{"metadata": {"top_verified_only_drops": [["solo_nombre"]]}}]) == []


# ════════════════════════════════════════════════════════════════════════════════════════════
# E. Task 2 — preselección IANA (Addendum §4): paridad TZ→país con COUNTRY_PROFILES
# ════════════════════════════════════════════════════════════════════════════════════════════
#
# `frontend/src/config/countries.js` gana `countryFromTimeZone(tzName)` — traduce el NOMBRE
# de la zona horaria IANA del navegador a un código de país, JAMÁS el offset (RD y Puerto
# Rico comparten -240 los 365 días: serían indistinguibles — la razón que el propio Addendum
# cita para prohibirlo). Este backend no ejecuta JS: igual que
# `test_paridad_countries_js_con_country_profiles` (F0), parsea el FUENTE con regex
# (comentarios fuera, CRLF-safe) y verifica la propiedad — nunca la grafía de cada línea.

def _js_sin_comentarios(path: Path) -> str:
    """Mismo stripping que test_p1_country_system_f0.py: bloque `/* */` fuera primero
    (re.S), luego `// ...` línea a línea — `splitlines()`-equivalente vía split CRLF-safe
    (no asume qué separador usa el archivo)."""
    src = path.read_text(encoding="utf-8")
    return "\n".join(
        re.sub(r"(^|\s)//.*$", r"\1", l)
        for l in re.split(r"\r?\n", re.sub(r"/\*.*?\*/", "", src, flags=re.S))
    )


def _countries_js_sin_comentarios() -> str:
    return _js_sin_comentarios(_FRONTEND / "src" / "config" / "countries.js")


def _tz_country_codes_from_js() -> set:
    """Códigos que `countryFromTimeZone` puede emitir por zona DEDICADA: los valores de
    `TZ_COUNTRY_EXACT` + el 2º elemento de cada par `[prefijo, código]` de
    `TZ_COUNTRY_PREFIXES`. (El fallback `DEFAULT_COUNTRY`='DO' para lo desconocido/ausente
    vive en la firma de la función, no en estas tablas — pero DO igual aparece aquí porque
    `America/Santo_Domingo` tiene su propia fila explícita, contrato Task 2 punto 1.)"""
    src = _countries_js_sin_comentarios()

    ini_exact = src.index("const TZ_COUNTRY_EXACT")
    fin_exact = src.index("};", ini_exact)
    bloque_exact = src[ini_exact:fin_exact]
    matches_exact = re.findall(r":\s*'([A-Z]{2})'", bloque_exact)
    # [fix-round 1 · review] Contar FILAS crudas ANTES de deduplicar a `set`: un
    # set es CIEGO a una truncación PARCIAL que deja ≥1 fila por código — el test
    # de igualdad de abajo (`test_paridad_tz_country_map_con_country_profiles`)
    # seguiría en verde aunque el bloque perdiera 15 de sus 25 filas, mientras
    # sobreviva al menos una por código. Es EXACTAMENTE el modo de fallo que el
    # propio accidente del comment-stripper de esta misma task habría dejado
    # pasar en silencio si hubiera devorado solo un tramo intermedio en vez del
    # bloque completo (ver reporte de Task 2, fix-round 1). 25 = las filas de
    # `TZ_COUNTRY_EXACT` hoy (DO + PR + ES×2 + CO + 11 MX + 9 US) — el piso es
    # `>=`, no `==`, porque el número solo puede CRECER con más zonas.
    assert len(matches_exact) >= 25, (
        f"esperaba ≥25 filas en TZ_COUNTRY_EXACT, parseé {len(matches_exact)} — posible "
        "truncación parcial del bloque (mismo modo de fallo que el accidente del "
        "comment-stripper de esta task)."
    )
    codigos = set(matches_exact)

    ini_prefix = src.index("const TZ_COUNTRY_PREFIXES")
    fin_prefix = src.index("];", ini_prefix)
    codigos |= set(re.findall(r"'([A-Z]{2})'\s*\]", src[ini_prefix:fin_prefix]))

    assert codigos, "No pude parsear ningún código de TZ_COUNTRY_EXACT/TZ_COUNTRY_PREFIXES"
    return codigos


def test_countryfromtimezone_existe_y_toma_un_nombre_de_zona():
    src = _countries_js_sin_comentarios()
    assert "export function countryFromTimeZone(tzName)" in src, (
        "countryFromTimeZone debe existir y tomar un NOMBRE de zona (string) — Addendum §4 "
        "prohíbe inferir el país desde el offset."
    )


def test_paridad_tz_country_map_con_country_profiles():
    """Task 2, contrato del brief: 'every code the TZ map emits exists in
    constants.COUNTRY_PROFILES'. Un código que la tabla TZ→país pudiera devolver sin perfil
    backend es exactamente el drift que P1-DIET-CANON-SSOT pagó (tres tablas de dieta a mano,
    driftaron, una sirvió Pollo a vegetarianas) — aquí el motor recibiría un país sin
    moneda/piso/tz default.

    Igualdad, no solo subconjunto: ancla ADEMÁS que Task 2 mapeó los 6 países con al menos
    una zona propia (los 5 beta + DO vía `America/Santo_Domingo`) — no solo que lo que hay es
    válido. Es el guard backend que complementa la mutación pedida en el brief (quitar la fila
    de Puerto Rico ⇒ vitest RED): si esa misma fila desapareciera, el set deja de cubrir 'PR'
    y este test también cae."""
    codigos_tz = _tz_country_codes_from_js()
    perfiles = set(constants.COUNTRY_PROFILES.keys())
    assert codigos_tz == perfiles, (
        f"TZ_COUNTRY_EXACT/PREFIXES vs COUNTRY_PROFILES divergen. "
        f"Sin perfil backend: {codigos_tz - perfiles}. "
        f"Perfilados sin zona dedicada: {perfiles - codigos_tz}."
    )


# ════════════════════════════════════════════════════════════════════════════════════════════
# F. Task 3 — Coach en tu idioma, comida en español (Addendum §2)
# ════════════════════════════════════════════════════════════════════════════════════════════
#
# `prompts.chat_agent.build_language_directive(locale)` es el SSOT: una directiva de idioma
# derivada de `user_profiles.locale` (los 5 valores de P1-I18N-DASHBOARD: es-DO, en-US, pt-BR,
# fr-FR, it-IT), inyectada en AMBAS copias del coach (`agent.py::chat_with_agent` /
# `chat_with_agent_stream`) Y en el agente proactivo (`proactive_agent.py::run_proactive_checks`,
# notificaciones LLM user-facing — chat + push body). FRONTERA DURA (Addendum §2, nombrada dos
# veces por el dueño): nombres de alimentos/platos y tool calls SIEMPRE en español canónico — la
# propia directiva lo instruye Y las tool-instructions (`build_tools_instructions*`) NUNCA se
# parametrizan por locale (test F8, "las tool calls no ganan ningún camino de traducción").
#
# `country` (cocina/precios) es un eje INDEPENDIENTE de `locale` (idioma) — spec Fase 2
# "Limitaciones aceptadas": "no se infieren mutuamente". Nada aquí debe leer/reutilizar
# `country_for_form_data`.
#
# Parser, NUNCA `import agent`/`import proactive_agent`: ambos módulos cargan LangGraph/DB pool/
# .env a nivel de import (mismo motivo que test_p0_agent_1_user_id_override.py y
# test_p1_prod_audit_3.py leen agent.py como texto plano). `prompts.chat_agent` SÍ se importa en
# vivo para las pruebas funcionales de `build_language_directive` — su único import de módulo es
# `datetime`/`typing` (cero DB/LLM), confirmado inocuo (mismo patrón que test_p3_chat_identity.py).
#
# Decisiones de alcance (documentadas, no gaps — ver reporte de la task para el detalle):
#   - `prompts/help_bot.py` (el bot de "Obtener ayuda", marketing/producto): FUERA de alcance.
#     Su propio docstring lo dice: "NO tiene tools, NO recibe user_id, NO toca DB" — no hay
#     `locale` que leer (test F9).
#   - `vision_agent.py` (escaneo de comida): el prompt de vision produce un INVENTARIO
#     ESTRUCTURADO de ingredientes en español dominicano (alimenta `pantry_names_match`), no
#     prosa conversacional — es "contenido", no "coach"; frontera del Addendum lo excluye.
#   - `generate_chat_title_background`/`TITLE_GENERATION_PROMPT` (título del chat en el
#     sidebar): es una etiqueta de navegación de 2-4 palabras, no "prosa del coach" ni
#     "notificación" — más cercano a "chrome del dashboard" (categoría YA cubierta por el i18n
#     de UI). El contrato de esta task no lo nombra explícitamente entre los archivos a tocar;
#     queda fuera, documentado para que un futuro P-fix lo retome si el dueño lo pide.


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _fn_body(src: str, def_line: str, end_marker: str | None = None) -> str:
    """Mismo helper que `test_p1_prod_audit_3.py::_fn_body` — cuerpo de una función desde
    `def_line` hasta `end_marker` (si se da) o el próximo `\\ndef ` top-level, o EOF."""
    start = src.index(def_line)
    if end_marker:
        return src[start: src.index(end_marker, start)]
    nxt = src.find("\ndef ", start + 1)
    return src[start: nxt if nxt != -1 else len(src)]


# ── F1. build_language_directive existe (RED si el Task 3 no se implementó) ───────────────

def test_build_language_directive_existe_y_es_invocable():
    assert hasattr(chat_agent_prompts, "build_language_directive")
    assert chat_agent_prompts.build_language_directive("en-US") is not None


# ── F2. es-DO ⇒ "" byte-idéntico, is-anchored (contrato Task 3: "is/== exact") ─────────────

def test_es_do_devuelve_cadena_vacia_is_e_igualdad():
    r = chat_agent_prompts.build_language_directive("es-DO")
    assert r == ""
    # [fix-round 1 · 2026-08-17] `r is ""` es CIERTO en CPython (el string vacío está
    # interned) pero el propio intérprete emite SyntaxWarning por comparar identidad
    # contra un literal — la garantía es real, la SINTAXIS que la exhibe no. `len(r) == 0`
    # confirma la misma propiedad (vacío de verdad, no un str truthy-vacío raro) sin activar
    # el warning del parser.
    assert r == "" and len(r) == 0


@pytest.mark.parametrize("basura", [
    None, "", "xx-XX", "ES-do", "es-do", "es", 123, 4.5, ["en-US"], {"locale": "en-US"},
])
def test_locale_none_vacio_o_basura_retorna_cadena_vacia_fail_safe(basura):
    """Locale garbage/NULL ⇒ es-DO (fail-safe) — incluye tipos no-string (nunca debe lanzar)."""
    assert chat_agent_prompts.build_language_directive(basura) == ""


# ── F3. en-US ⇒ directiva + excepción de nombres + regla de tool calls ─────────────────────

def test_en_us_contiene_directiva_excepcion_de_nombres_y_regla_tool_calls():
    r = chat_agent_prompts.build_language_directive("en-US")
    assert "SIEMPRE en English" in r
    assert "nombres de alimentos y platos" in r and "español" in r
    assert "en las tool calls usa EXCLUSIVAMENTE los nombres canónicos en español" in r


@pytest.mark.parametrize("locale,idioma", [
    ("pt-BR", "Português"),
    ("fr-FR", "Français"),
    ("it-IT", "Italiano"),
])
def test_los_otros_3_idiomas_contienen_su_propia_directiva(locale, idioma):
    r = chat_agent_prompts.build_language_directive(locale)
    assert f"SIEMPRE en {idioma}" in r
    assert "nombres de alimentos y platos" in r and "español" in r
    assert "en las tool calls usa EXCLUSIVAMENTE los nombres canónicos en español" in r


# ── F4. Variante-cacheada (patrón T2-F1, _COUNTRY_PROMPT_RENDER_CACHE) ─────────────────────

def test_cacheado_por_variante_misma_instancia_segunda_llamada():
    r1 = chat_agent_prompts.build_language_directive("en-US")
    r2 = chat_agent_prompts.build_language_directive("en-US")
    assert r1 is r2, "debe reusar el string cacheado, no reconstruirlo por llamada"


def test_cache_no_colisiona_entre_idiomas():
    en = chat_agent_prompts.build_language_directive("en-US")
    pt = chat_agent_prompts.build_language_directive("pt-BR")
    assert en != pt
    assert "English" in en and "Português" in pt


# ── F5. Ambas copias del coach cargan la directiva (parser, ancla por propiedad) ───────────
#
# Lección P1-CHAT-PAUSED-PROMPT-BLOCKS (2026-08-14): "un bloque arreglado y otro contradice"
# es un bug real de este mismo archivo. Las dos funciones se verifican POR SEPARADO — no basta
# con que la subcadena exista UNA vez en todo agent.py.

def test_chat_with_agent_inline_llama_build_language_directive():
    src = _read(_AGENT_PY)
    body = _fn_body(src, "def chat_with_agent(session_id: str", end_marker="def chat_with_agent_stream(")
    assert "build_language_directive(" in body


def test_chat_with_agent_stream_llama_build_language_directive():
    src = _read(_AGENT_PY)
    body = _fn_body(src, "def chat_with_agent_stream(session_id: str")  # última función del archivo → EOF
    assert "build_language_directive(" in body


def test_agent_importa_build_language_directive_de_prompts_chat_agent():
    src = _read(_AGENT_PY)
    ini = src.index("from prompts.chat_agent import (")
    fin = src.index("\n)", ini)
    bloque = src[ini:fin]
    assert "build_language_directive" in bloque


# ── F6. Guests ⇒ es-DO (nunca leen `locale`, se quedan en el default) — parser + funcional ──

@pytest.mark.parametrize("def_line,end_marker", [
    ("def chat_with_agent(session_id: str", "def chat_with_agent_stream("),
    ("def chat_with_agent_stream(session_id: str", None),
])
def test_guest_nunca_gana_locale_default_precede_al_guard_de_autenticado(def_line, end_marker):
    """`_coach_locale = "es-DO"` DEBE quedar asignado ANTES del
    `if user_id and user_id != session_id and user_id != "guest":` que lo sobre-escribe con el
    valor real del perfil. Un guest (o user_id==session_id) NUNCA entra a ese `if`, así que su
    locale se queda en el default 'es-DO' estructuralmente — sin branch propio que pueda
    olvidarse de excluir al guest."""
    src = _read(_AGENT_PY)
    body = _fn_body(src, def_line, end_marker=end_marker)
    pos_default = body.index('_coach_locale = "es-DO"')
    pos_guard = body.index('if user_id and user_id != session_id and user_id != "guest":')
    assert pos_default < pos_guard


@pytest.mark.parametrize("def_line,end_marker", [
    ("def chat_with_agent(session_id: str", "def chat_with_agent_stream("),
    ("def chat_with_agent_stream(session_id: str", None),
])
def test_coach_locale_del_profile_tiene_fallback_es_do_si_es_falsy(def_line, end_marker):
    """Segunda capa de fail-safe (además de la de `build_language_directive` misma): si
    `profile.get("locale")` viniera None/"" de la DB, el `or "es-DO"` evita propagar un
    falsy al builder."""
    src = _read(_AGENT_PY)
    body = _fn_body(src, def_line, end_marker=end_marker)
    assert '_profile_for_prompt.get("locale") or "es-DO"' in body


def test_build_language_directive_de_guest_es_cadena_vacia_extremo_funcional():
    """Extremo funcional del F6 parser: el valor que un guest estructuralmente conserva
    ('es-DO', el default) produce "" — cierra el circuito parser+funcional pedido por el
    contrato ('Guests ⇒ es-DO path')."""
    assert chat_agent_prompts.build_language_directive("es-DO") == ""


# ── F7. Reuso del profile ya leído — cero round-trips DB extra (paridad con el patrón country) ─

@pytest.mark.parametrize("def_line,end_marker", [
    ("def chat_with_agent(session_id: str", "def chat_with_agent_stream("),
    ("def chat_with_agent_stream(session_id: str", None),
])
def test_locale_se_lee_del_mismo_get_user_profile_que_full_name(def_line, end_marker):
    """El contrato de la task pide TRAZAR primero cómo llega `locale` y REUSAR si el endpoint
    ya lee el perfil. `chat_with_agent`/`_stream` YA llamaban `get_user_profile(user_id)` para
    `full_name` (P3-CHAT-IDENTITY) — el fix debe capturar ESE resultado (`_profile_for_prompt`)
    y leer `locale` de ahí, no añadir un segundo SELECT."""
    src = _read(_AGENT_PY)
    body = _fn_body(src, def_line, end_marker=end_marker)
    assert body.count("get_user_profile(user_id)") == 1, (
        "debe haber EXACTAMENTE una llamada a get_user_profile(user_id) en este bloque — "
        "full_name y locale comparten la misma lectura, cero round-trips nuevos"
    )
    assert "_profile_for_prompt = get_user_profile(user_id)" in body
    assert '_id_name = _profile_for_prompt.get("full_name")' in body


# ── F8. Las tool calls NUNCA ganan un camino de traducción (frontera dura, parser) ─────────

def test_build_tools_instructions_no_gana_parametro_locale():
    """`build_tools_instructions`/`_stream` (las instrucciones de tool-calling que el coach
    recibe) deben conservar EXACTAMENTE su firma pre-Task-3 — sin parámetro `locale`. Si
    alguien threadeara locale hasta aquí para "traducir" las instrucciones de herramientas,
    este test lo atrapa: la frontera dura del Addendum exige que las tool calls sigan en
    español canónico SIEMPRE, sin ningún camino condicional por idioma."""
    src = _read(_CHAT_AGENT_PY)
    assert "def build_tools_instructions(user_id: str, plan_en_pausa: bool = False) -> str:" in src
    assert "def build_tools_instructions_stream(user_id: str, plan_en_pausa: bool = False) -> str:" in src

    body_inline = _fn_body(src, "def build_tools_instructions(user_id: str, plan_en_pausa: bool = False) -> str:",
                            end_marker="def build_tools_instructions_stream(")
    body_stream = _fn_body(src, "def build_tools_instructions_stream(user_id: str, plan_en_pausa: bool = False) -> str:",
                            end_marker="def build_inventory_context(")
    assert "locale" not in body_inline
    assert "locale" not in body_stream
    assert "build_language_directive" not in body_inline
    assert "build_language_directive" not in body_stream


# ── F9. help_bot (marketing/producto) — FUERA de alcance, documentado ──────────────────────

def test_help_bot_no_gana_la_directiva_de_idioma():
    """`prompts/help_bot.py` es el Q&A de producto público de "Obtener ayuda": cero tools,
    cero DB, cero user_id en el prompt (su propio docstring lo declara). No hay `locale` que
    leer — la directiva NO debe importarse ni mencionarse ahí."""
    src = _read(_HELP_BOT_PY)
    assert "build_language_directive" not in src
    assert "locale" not in src


# ── F10. proactive_agent — notificaciones LLM user-facing (Addendum §2, "las notificaciones") ─
#
# `run_proactive_checks` arma DOS prompts LLM cuyo output es prosa mostrada al usuario (se
# persiste como mensaje del coach vía `save_message(..., "model", content)` Y viaja como body
# de la Web Push): el f-string de "Resumen del día" (usuario sin ningún registro hoy) y
# `PROACTIVE_PROMPT.format(...)` (comida específica olvidada). `classify_nudge_sentiment` NO
# entra aquí: su output es JSON estructurado interno (sentiment/meal_logged/causal_reason),
# nunca se muestra al usuario.

def test_proactive_agent_importa_build_language_directive():
    src = _read(_PROACTIVE_AGENT_PY)
    assert "from prompts.chat_agent import build_language_directive" in src


def test_run_proactive_checks_aplica_directiva_en_ambos_prompts_antes_del_invoke():
    src = _read(_PROACTIVE_AGENT_PY)
    body = _fn_body(src, "def run_proactive_checks():", end_marker="def _trigger_week2_background_generation(")
    n = body.count("build_language_directive(")
    assert n >= 2, f"esperaba ≥2 call sites (Resumen del día + PROACTIVE_PROMPT), encontré {n}"
    # La directiva se aplica ANTES de invocar el LLM que consumirá `prompt` — si se apilara
    # DESPUÉS del primer invoke sería demasiado tarde para ese branch.
    primer_directive = body.index("build_language_directive(")
    primer_invoke = body.index("chat_llm.invoke(prompt)")
    assert primer_directive < primer_invoke


def test_run_proactive_checks_locale_viene_del_profile_ya_leido_sin_query_extra():
    """`profile = get_user_profile(user_id)` YA se lee en esta función (para `scheduleType` /
    turno nocturno) — el locale del nudge debe leer del MISMO `profile`, cero round-trips
    nuevos. Mismo criterio de reuso que F7 para agent.py."""
    src = _read(_PROACTIVE_AGENT_PY)
    body = _fn_body(src, "def run_proactive_checks():", end_marker="def _trigger_week2_background_generation(")
    assert body.count("get_user_profile(user_id)") == 1
    pos_profile = body.index("profile = get_user_profile(user_id)")
    pos_locale = body.index('_nudge_locale = profile.get("locale")')
    assert pos_profile < pos_locale


def test_classify_nudge_sentiment_no_toca_la_directiva():
    """`classify_nudge_sentiment` clasifica la RESPUESTA del usuario a un JSON interno — nunca
    genera prosa mostrada al usuario. No debe ganar la directiva (scope creep innecesario)."""
    src = _read(_PROACTIVE_AGENT_PY)
    body = _fn_body(src, "def classify_nudge_sentiment(user_reply: str) -> dict:", end_marker="def handle_nudge_response(")
    assert "build_language_directive" not in body


# ════════════════════════════════════════════════════════════════════════════════════════════
# G. Task 4 — Los 4 vocabularios de alérgenos/dieta ×país + drift RD (INNEGOCIABLE)
# ════════════════════════════════════════════════════════════════════════════════════════════
#
# CUATRO vocabularios paralelos de alérgenos/dieta viven en el repo, con propósitos distintos y
# SIN un mecanismo que los mantenga sincronizados:
#
#   #1 `graph_orchestrator._ALLERGEN_SYNONYMS` (~:14214) — DETECCIÓN determinista post-generación
#      (C2-ALLERGEN-GUARD). `_scan_allergen_violations` lo usa para escanear el plan YA generado
#      contra las alergias IgE declaradas; es la red de seguridad final (`clinical_backstop_for_meal`
#      lo reusa para swap/regenerate-day/chat-modify). Sesgo a SOBRE-detectar.
#   #2 `graph_orchestrator._DIET_*_TERMS` (~:14335-14360, cuatro tuplas: FLESH/SEAFOOD/EGG/DAIRY)
#      — DETECCIÓN determinista de producto ANIMAL para dietType (vegano/vegetariano/pescetariano),
#      vía `_scan_diet_violations`. Eje ortogonal a #1 (una dieta no es una alergia IgE), pero
#      SEAFOOD/EGG/DAIRY nombran ingredientes concretos que #1 TAMBIÉN nombra bajo sus categorías
#      mariscos+pescado/huevo/lácteos — esas tres son las que deben coincidir.
#   #3 Los catch-alls de categoría inline en `constants._get_fast_filtered_catalogs` (~:3369-3430)
#      — PRE-FILTRADO: cuando un chip de alergia/dislike es una CATEGORÍA ("Mariscos", "Gluten"),
#      expande a los nombres concretos del catálogo curado (`DOMINICAN_PROTEINS`/`CARBS`/
#      `VEGGIES_FATS`/`FRUITS`) para no ofrecerlos en los pools de variedad. Su paridad contra #1
#      YA la enforza `test_p2_catalog_filter_ssot.py::test_paridad_filtro_vs_escaner_canonico`
#      (usa `_scan_allergen_violations` — o sea #1 — como oráculo contra los sobrevivientes del
#      filtro). Este archivo NO duplica ese test; lo cita, lo mantiene verde (barrido de
#      vecinos), y ancla con un caso puntual que las altas de #1 de este task ya estaban
#      cubiertas ahí (ver `test_altas_de_este_task_ya_estaban_cubiertas_por_constants_catchall`).
#      `constants.py` NO se tocó en este task — un intento con 'avena' SÍ lo tocó y se revirtió
#      junto con la entrada gemela en #1 (ver G1/G4: colisiona con P1-ALLERGEN-NEGATION-EXCUSE).
#   #4 `condition_rules._ALLERGEN_DETECT` / `_ALLERGEN_*_SUBS` (fish/shellfish/soy/gluten,
#      condition_rules.py ~:697-760) — SUSTITUCIÓN QUIRÚRGICA proactiva (P0-ALLERGEN-SUBS): ANTES
#      de que el plan se persista, reemplaza el ingrediente ofensor por uno seguro que resuelve al
#      catálogo (p.ej. camarón→pollo), conservando el plan rico del LLM. Documentado como
#      INTENCIONALMENTE más ESTRECHO que #1 en dos ejes: (a) por DISEÑO cubre solo 4 categorías
#      (fish/shellfish/soy/gluten) — lácteos/huevo/maní/frutos secos quedan FUERA a propósito
#      ("DECISIÓN HONESTA", condition_rules.py ~L685: el catálogo es-DO no tiene un target libre
#      del alérgeno que resuelva); (b) sus tokens son deliberadamente ESTRECHOS ("lección del bug
#      'soya'/'pana'") — lo que un token estrecho no atrape lo recoge el backstop #1. Que #4 sea
#      MÁS ESTRECHO que #1 es la relación esperada; que #4 conozca un alimento como riesgo y #1 NO
#      (el sentido que sí importa) es un agujero real — ver sección G4.
#
# DRIFT VIVO medido en este task (T4, pre-fix): 'mejillón'/'vieira' viven en #2
# (`_DIET_SEAFOOD_TERMS`, comentario `P1-VARIETY-CATALOG-POOLS`) pero NO en #1 — un plan con
# "Mejillones" para un alérgico a mariscos pasaba el backstop determinista limpio. Confirmado
# EN VIVO contra `master_ingredients` (206 filas): 'Mejillones' y 'Arenque' SON alimentos
# catalogados hoy; `clinical_backstop_for_meal(..., allergies=["mariscos"|"pescado"])` no los
# marcaba — ver G5. El diff completo (algoritmo `_uncovered`, abajo) contra las clases
# mariscos+pescado/lácteos/huevo aparece documentado en cada assert de G2.
#
# Checklist-anchor para T5-T8 (el alta-hook, contrato del task): cuando un país nuevo dé de alta
# un alimento que sea alérgeno o clase-dieta, este archivo tiene DOS guards que deben seguir en
# verde:
#   1. `test_paridad_dieta_alergeno_bidireccional` (G2) — si el alimento nuevo entra a
#      `_DIET_*_TERMS` (o a `_ALLERGEN_SYNONYMS`) sin su espejo en el otro, este test se pone rojo.
#   2. `test_backstop_conoce_cada_alimento_peligroso_del_catalogo_vivo` (G5, `@pytest.mark.e2e`)
#      — si el alimento nuevo entra a `master_ingredients` y CUALQUIER vocabulario hermano ya lo
#      reconoce como peligroso pero `_ALLERGEN_SYNONYMS` no, este test se pone rojo con el nombre
#      exacto del alimento y la clase que falta.
# La corrección en ambos casos es la MISMA: añadir el término faltante a
# `_ALLERGEN_SYNONYMS[<clase>]` (graph_orchestrator.py) y/o `_DIET_*_TERMS` según corresponda —
# NUNCA borrar un término existente (dirección de seguridad: solo se añade, ver docstring de cada
# vocabulario para la categoría correcta).


@pytest.fixture(scope="module")
def go():
    """`graph_orchestrator` es un módulo de ~38k líneas que importa LangGraph/DB a nivel de
    import — el mismo motivo por el que `test_p1_allergen_derivatives.py` lo carga vía fixture
    en vez de `import graph_orchestrator` a nivel de módulo del archivo de test."""
    import graph_orchestrator as _go
    return _go


@pytest.fixture(scope="module")
def condrules():
    import condition_rules as _cr
    return _cr


def _term_matches(term: str, text: str) -> bool:
    """El MISMO matcher que producción usa en `_scan_allergen_violations`/`_scan_diet_violations`:
    `\\b<term>(?:s|es)?\\b` sobre texto accent-stripped + lower. Reusar este matcher (no un `in`
    plano, no un `set()` de strings crudos) es lo que evita DOS clases de falso-positivo al
    diffear vocabularios en este archivo:
      (a) singular/plural — 'camaron' en un vocabulario y 'camarones' en el otro NO son un gap
          real, porque el sufijo `(?:s|es)?` ya los hace equivalentes en producción;
      (b) frase compuesta redundante — 'salsa de pescado' ausente de un vocabulario NO es un gap
          si ese vocabulario ya tiene la raíz 'pescado' (la frase completa la cubre por substring
          igual que un ingrediente real la cubriría).
    Término vacío nunca matchea (evita el bug F31 de `constants.py`: una alternativa vacía en el
    regex matchearía cualquier posición)."""
    t = constants.strip_accents(str(term)).lower().strip()
    if not t:
        return False
    return re.search(r"\b" + re.escape(t) + r"(?:s|es)?\b",
                      constants.strip_accents(str(text)).lower()) is not None


def _covered(terms, probe: str) -> bool:
    """True si ALGÚN término de `terms` reconocería `probe` (nombre/frase completa) como
    ingrediente ofensor — el mismo criterio que corre en producción contra un ingrediente real."""
    return any(_term_matches(t, probe) for t in terms)


def _uncovered(source_terms, target_terms) -> list:
    """Términos de `source_terms` que `target_terms` NO reconocería si escaneara ese término como
    si fuera el texto completo de un ingrediente. Es la unidad de comparación de TODA esta
    sección — deliberadamente NO es `set(source) - set(target)` (eso cuenta 'camaron'≠'camarones'
    y 'salsa de pescado'≠'pescado' como gaps falsos, ver `_term_matches`)."""
    return sorted({t for t in source_terms if not _covered(target_terms, t)})


# ── G1. El drift vivo: mejillón/vieira (el TDD RED de este task) ──────────────────────────────

@pytest.mark.parametrize("ingrediente", ["Mejillones", "1 libra de mejillón", "Vieiras a la plancha"])
def test_mejillon_vieira_flageados_como_mariscos(go, ingrediente):
    """[G1 · el drift nombrado por el task] Pre-fix, `_ALLERGEN_SYNONYMS['mariscos']` no conocía
    'mejillon'/'mejillones'/'vieira' — este assert es la RED que el fix cierra. Funcional, no
    estructural: prueba el camino real (`_scan_allergen_violations`), igual que
    `test_p1_allergen_derivatives.py`."""
    plan = {"days": [{"meals": [{"name": "Cena", "ingredients": [ingrediente, "Arroz blanco"]}]}]}
    violaciones = go._scan_allergen_violations(plan, ["mariscos"])
    assert violaciones, f"'{ingrediente}' no fue detectado como marisco por _ALLERGEN_SYNONYMS"


def test_arenque_flageado_como_pescado(go):
    """Segundo hallazgo del diff (no nombrado por el task, encontrado diffeando la clase
    pescado): 'arenque' vive en `_DIET_SEAFOOD_TERMS` desde P1-VARIETY-CATALOG-POOLS pero
    faltaba en `_ALLERGEN_SYNONYMS['pescado']`. 'Arenque' es fila real de `master_ingredients`
    (confirmado en vivo, ver G5) — no es un caso hipotético."""
    plan = {"days": [{"meals": [{"name": "Almuerzo", "ingredients": ["Arenque ahumado"]}]}]}
    assert go._scan_allergen_violations(plan, ["pescado"])


# [fix-round 1 · 2026-08-17 · REVIERTE la decisión de T4 de arriba] El review de T4 confirmó por
# EJECUCIÓN DIRECTA (el harness abajo, reproducido tal cual antes de tocar código) que 'avena'
# bare NO tenía NINGÚN backstop determinista en 4 superficies vivas que dependen EXCLUSIVAMENTE
# de `clinical_backstop_for_meal`: swap individual (agent.py), regenerate-day (agent.py), chat
# modify (tools.py::execute_modify_single_meal) y el tamiz degradado sin LLM
# (`cron_tasks._sieve_catalog_for_safety`/`_degraded_safety_violations`) — NINGUNA de las cuatro
# pasa por `_apply_deterministic_clinical_layer` (solo generación inicial), así que la sustitución
# proactiva de `condition_rules.py` (que SÍ conoce 'sin gluten' vía `_ALLERGEN_GLUTEN_NEGATIVES`)
# nunca corre ahí. Razón por la que T4 rechazó 'avena': `_ALLERGEN_NEGATION_PREFIX_RX` excusa SOLO
# por PREFIJO (mira hacia atrás) y en "avena certificada sin gluten" la negación SIGUE a 'avena'.
# El cierre no reintenta lo mismo: añade una excusa FORWARD nueva (`_GLUTEN_FORWARD_EXCUSE_RX`,
# graph_orchestrator.py), scoped a la categoría gluten ÚNICAMENTE, que mira ADELANTE del match —
# mismo mecanismo estructural que `_PLANT_ADJ_EXCUSE_RX` ya usa para plant-adjacency, aplicado a
# negación en vez de a adyacencia vegetal.
#
#     Bare Avena, allergies=[gluten] via clinical_backstop_for_meal: []          ← el hueco (RED)
#     Avena cocida (no GF claim): []                                             ← también el hueco
#     Pan integral (control): ["alérgeno 'pan integral' ..."]                    ← el matcher funciona


def test_avena_bare_flageada_como_gluten_fix_round_1(go):
    """[fix-round 1 · el RED de este fix-round] Bare 'avena' (sin claim 'sin gluten') DEBE violar
    gluten — la excusa forward nunca excusa un término desnudo, solo uno seguido de una negación
    explícita dentro de la ventana corta. Ancla el estado GREEN post-fix del hallazgo del harness
    de arriba, incluida la superficie real (`clinical_backstop_for_meal`, no solo el scanner
    interno) que expuso el hueco en las 4 superficies vivas."""
    plan_bare = {"days": [{"meals": [{"name": "Desayuno", "ingredients": ["Avena"]}]}]}
    plan_cocida = {"days": [{"meals": [{"name": "Desayuno", "ingredients": ["Avena cocida"]}]}]}
    assert go._scan_allergen_violations(plan_bare, ["gluten"]), (
        "'Avena' bare (sin claim GF) debe violar gluten — sin esto, swap/regenerate-day/"
        "chat-modify/tamiz-degradado sirven avena sin backstop a un alérgico"
    )
    assert go._scan_allergen_violations(plan_cocida, ["gluten"]), (
        "'Avena cocida' (sin claim GF) debe violar gluten — mismo hueco, otra grafía"
    )
    meal_bare = {"name": "Desayuno", "ingredients": ["Avena"]}
    assert go.clinical_backstop_for_meal(meal_bare, allergies=["gluten"]), (
        "clinical_backstop_for_meal (swap/regenerate-day/chat-modify) debe bloquear avena bare"
    )


def test_avena_certificada_sin_gluten_sigue_excusada_tras_incluir_avena(go):
    """[fix-round 1 · ancla (b) del contrato: debe seguir verde ANTES y DESPUÉS] La razón por la
    que T4 rechazó 'avena' (colisión con P1-ALLERGEN-NEGATION-EXCUSE, solo-prefijo) queda cerrada
    por la excusa FORWARD nueva. Ancla en ESTE archivo (no solo en
    `test_p1_allergen_negation_excuse.py`) que el caso medido original — y la grafía sin
    'certificada' — siguen sin violar."""
    for ingrediente in ("20 g de avena certificada sin gluten", "Avena sin gluten (panqueques)"):
        plan = {"days": [{"meals": [{"name": "Desayuno", "ingredients": [ingrediente]}]}]}
        assert go._scan_allergen_violations(plan, ["gluten"]) == [], (
            f"'{ingrediente}' es CUMPLIMIENTO (avena GF certificada) — no debe violar"
        )


def test_leche_sin_lactosa_no_se_excusa_por_la_excusa_forward_de_gluten(go):
    """[fix-round 1 · ancla (c): sin cambio de conducta, control negativo del scoping] La excusa
    forward nueva está SCOPED a la categoría gluten ÚNICAMENTE (gateada por
    `_ALLERGEN_GLUTEN_TERM_SET` en el callsite) — NUNCA generalizar entre categorías. 'leche sin
    lactosa' DEBE seguir violando lácteos: la alergia es a la PROTEÍNA (caseína/whey), no al
    azúcar — 'lactosa' quedaría negada pero 'leche' no (mismo criterio que
    `test_leche_sin_lactosa_sigue_violando_lacteos` en test_p1_allergen_negation_excuse.py; este
    test ancla específicamente que la excusa NUEVA no se filtró fuera de su scope)."""
    plan = {"days": [{"meals": [{"name": "Cena", "ingredients": ["200 ml de leche sin lactosa"]}]}]}
    v = go._scan_allergen_violations(plan, ["Lácteos"])
    assert v and v[0][2] == "leche", "leche sin lactosa debe seguir violando lácteos (no gluten-scoped)"


def test_tostada_sobre_detecta_almendras_tostadas_aceptado(go):
    """[finding 4 del fix-round 1 · documented-accept, NO remover el término] 'tostada' bare
    (gluten) también matchea 'almendras tostadas' (frutos secos tostados, sin relación con
    gluten) — mismo token pre-existe en `condition_rules._ALLERGEN_GLUTEN_SUBS` (swap 'pan
    tostado'/'tostada'→Casabe). Sesgo a SOBRE-detectar es la dirección de seguridad declarada de
    este vocabulario (docstring de `_ALLERGEN_SYNONYMS`, C2-ALLERGEN-GUARD) — el costo (un plan
    con almendras tostadas cae a fallback para un alérgico a gluten, aunque las almendras no
    tengan gluten) se documenta aquí para que sea VISIBLE, no silencioso. Comportamiento
    PRE-EXISTENTE (no introducido por este fix-round) — verificado idéntico antes y después."""
    plan = {"days": [{"meals": [{"name": "Snack", "ingredients": ["Almendras tostadas"]}]}]}
    v = go._scan_allergen_violations(plan, ["gluten"])
    assert v and v[0][2] == "tostada", (
        "'Almendras tostadas' debe seguir disparando 'tostada' — sobre-detección aceptada, "
        "NO remover el término"
    )


# ── G1-bis. fix-round 2: el leak del relleno libre `{0,2}` en `_GLUTEN_FORWARD_EXCUSE_RX` ──────
#
# [fix-round 2 · 2026-08-17 · re-review] El `{0,2}` de relleno de fix-round 1 aceptaba CUALQUIER
# token (`\S+`) antes de la negación — no solo adjetivos de claim GF. Consecuencia: en un
# ingrediente con DOS términos de gluten separados por una conjunción, la excusa se filtraba
# HACIA ATRÁS sobre el término SIN claim propio. Medido por ejecución directa (reproducido tal
# cual antes de tocar código, harness del re-review):
#
#     'Trigo y avena sin gluten' + [gluten] -> []   ← trigo (glutinoso incondicional) excusado mal
#     'Avena y pan sin gluten'   + [gluten] -> []
#     'Avena y agua sin gluten'  + [gluten] -> []   ← CUALQUIER 2 tokens de relleno se tragan
#
# Dirección peligrosa (fail-open) — exactamente lo que el sesgo de sobre-detección de este
# vocabulario prohíbe. Ruling del controller: WHITELIST de tokens-adjetivo evidenciados, NO
# blacklist de conjunciones/términos (una blacklist deja unknown-unknowns sin cubrir — el mismo
# error de diseño en dirección opuesta).

@pytest.mark.parametrize("ingrediente", [
    "Trigo y avena sin gluten",
    "Avena y pan sin gluten",
    "Avena y agua sin gluten",
])
def test_relleno_libre_no_excusa_termino_vecino_sin_claim_propio(go, ingrediente):
    """[fix-round 2 · el RED de este re-review] Los 3 casos exactos del leak reportado. Cada uno
    DEBE violar gluten — el relleno de la excusa forward ya NO acepta tokens arbitrarios
    (conjunción 'y' + un término/palabra cualquiera), solo la whitelist evidenciada de adjetivos
    de claim GF ('certificada'). Ninguno de los 3 tiene 'certificada' cerca, así que ninguno debe
    excusarse — el claim 'sin gluten' pertenece SOLO al término que lo tiene inmediatamente
    adyacente (avena/pan en los casos 2-3; en el caso 1 NINGÚN término tiene claim propio: 'trigo'
    no, porque lo que lo sigue es 'y avena', y aunque 'avena' sí se excusa a sí misma, eso no
    excusa a 'trigo')."""
    plan = {"days": [{"meals": [{"name": "Desayuno", "ingredients": [ingrediente]}]}]}
    v = go._scan_allergen_violations(plan, ["gluten"])
    assert v, (
        f"'{ingrediente}' debe violar gluten — el relleno libre NO debe excusar un término sin "
        f"claim GF propio adyacente (leak de fix-round 1, cerrado en fix-round 2)"
    )


def test_trigo_con_avena_certificada_sin_gluten_excusa_solo_avena(go):
    """[fix-round 2 · control de precisión] Cuando SÍ hay un claim legítimo adyacente a UN
    término pero no al otro, cada término se evalúa por SU PROPIO contexto inmediato — no por
    'algún claim en algún lugar de la frase'. 'trigo' sigue sin backstop GF real (no existe tal
    cosa como trigo sin gluten) y debe violar; el hecho de que 'avena certificada sin gluten' sea
    legítima justo después no lo excusa retroactivamente."""
    plan = {"days": [{"meals": [{"name": "Desayuno", "ingredients": ["Trigo y avena certificada sin gluten"]}]}]}
    v = go._scan_allergen_violations(plan, ["gluten"])
    assert v and v[0][2] == "trigo", (
        "'trigo' debe seguir violando aunque 'avena certificada sin gluten' (legítimo) esté al lado"
    )


@pytest.mark.parametrize("ingrediente", [
    "Avena con un poco sin gluten",       # 3+ fillers — ya fallaba estructuralmente, debe seguir
    "Trigo, avena sin gluten",             # la coma rompe la excusa (sin espacio inicial válido)
])
def test_controles_del_leak_siguen_correctos_tras_whitelist(go, ingrediente):
    """[fix-round 2 · controles nombrados por el re-review, deben seguir en verde] Ninguno de
    estos dependía del relleno libre para funcionar correctamente — confirmán que la whitelist no
    los rompió."""
    plan = {"days": [{"meals": [{"name": "Desayuno", "ingredients": [ingrediente]}]}]}
    v = go._scan_allergen_violations(plan, ["gluten"])
    assert v, f"'{ingrediente}' debe seguir violando (control pre-existente, no debe romperse)"


def test_cereal_con_trigo_sigue_flageado(go):
    """[fix-round 2 · control nombrado por el re-review] Sin ninguna negación cerca, 'trigo' debe
    seguir flagged — no hay excusa que pudiera aplicar en ninguna versión del regex."""
    plan = {"days": [{"meals": [{"name": "Desayuno", "ingredients": ["Cereal con trigo"]}]}]}
    v = go._scan_allergen_violations(plan, ["gluten"])
    assert v and v[0][2] == "trigo"


def test_camarones_sin_gluten_sigue_violando_mariscos(go):
    """[fix-round 2 · ancla de paridad #3] 'camarones sin gluten' es una categoría DISTINTA
    (mariscos) — la excusa forward está scoped a gluten únicamente, así que esta alergia ni
    siquiera pasa por `_ALLERGEN_GLUTEN_TERM_SET`. Control de que la whitelist nueva no tiene
    ningún efecto fuera de su categoría."""
    plan = {"days": [{"meals": [{"name": "Cena", "ingredients": ["Camarones sin gluten"]}]}]}
    v = go._scan_allergen_violations(plan, ["mariscos"])
    # `forbidden` es un SET y 'camaron' Y 'camarones' son DOS entradas literales separadas del
    # vocabulario — cuál se reporta depende del orden de iteración (hash aleatorio por proceso,
    # mismo caso que `test_gluten_real_sigue_violando` ya documenta para 'trigo'/'harina de
    # trigo'). Ambas contienen 'camaron' como substring — ese es el criterio estable.
    assert v and "camaron" in v[0][2]


# ── G2. EL GUARD DE PARIDAD — dieta ↔ alérgeno bidireccional (el producto real de este task) ──
#
# Clases con contraparte en AMBOS vocabularios (nombran alimentos de origen animal Y son alergia
# IgE declarable): mariscos+pescado ↔ SEAFOOD, lácteos ↔ DAIRY, huevo ↔ EGG. Frutos secos/maní/
# gluten/soya son EXCLUSIVAMENTE de #1 (no son producto animal — una dieta vegana no los prohíbe);
# carne es EXCLUSIVAMENTE de #2 (este sistema no modela alergia IgE a carne/alfa-gal). Esa
# asimetría de CATEGORÍA es la "lista de excepciones documentadas" a nivel estructural que pide
# el contrato — no participan del cross-check porque no tienen con qué cruzarse.
_SEAFOOD, _DAIRY, _EGG = "mariscos+pescado (seafood)", "lacteos (dairy)", "huevo (egg)"
_CORRESPONDING_CLASSES = (_SEAFOOD, _DAIRY, _EGG)

# Excepciones documentadas POR TÉRMINO (no por categoría) — vacío hoy porque este task cerró
# cada asimetría real que encontró (ver reporte de la task para el diff completo antes/después).
# El mecanismo queda vivo para el día en que una asimetría legítima aparezca: añadir aquí con su
# razón, JAMÁS borrar el término del vocabulario que sí lo tiene.
_PARITY_TERM_EXCEPTIONS = {
    _SEAFOOD: {"solo_allergen": set(), "solo_diet": set()},
    _DAIRY: {"solo_allergen": set(), "solo_diet": set()},
    _EGG: {"solo_allergen": set(), "solo_diet": set()},
}


def _vocab_pair(clase, go):
    if clase == _SEAFOOD:
        return (list(go._ALLERGEN_SYNONYMS["mariscos"]) + list(go._ALLERGEN_SYNONYMS["pescado"]),
                list(go._DIET_SEAFOOD_TERMS))
    if clase == _DAIRY:
        return (list(go._ALLERGEN_SYNONYMS["lacteos"]), list(go._DIET_DAIRY_TERMS))
    if clase == _EGG:
        return (list(go._ALLERGEN_SYNONYMS["huevo"]), list(go._DIET_EGG_TERMS))
    raise ValueError(f"clase sin mapeo: {clase!r}")


@pytest.mark.parametrize("clase", _CORRESPONDING_CLASSES)
def test_paridad_dieta_alergeno_bidireccional(go, clase):
    """EL GUARD (contrato T4, ítem 2). Para cada clase con contraparte en ambos vocabularios:
    todo término de `_ALLERGEN_SYNONYMS` debe ser reconocido por `_DIET_*_TERMS` y viceversa,
    salvo excepción documentada en `_PARITY_TERM_EXCEPTIONS`.

    RED pre-fix (mejillón/vieira, T4): `_uncovered(diet_terms, allergen_terms)` para
    `_SEAFOOD` incluía 'mejillon'/'mejillones'/'vieira'/'arenque'. RED futuro (T5-T8): si una
    alta añade un alimento nuevo a un solo vocabulario de un par correspondiente, ese término
    aparece en uno de los dos `_uncovered(...)` de abajo y el assert falla con el nombre exacto."""
    allergen_terms, diet_terms = _vocab_pair(clase, go)
    excepciones = _PARITY_TERM_EXCEPTIONS[clase]

    solo_allergen = set(_uncovered(allergen_terms, diet_terms)) - excepciones["solo_allergen"]
    solo_diet = set(_uncovered(diet_terms, allergen_terms)) - excepciones["solo_diet"]

    assert not solo_allergen, (
        f"[{clase}] estos términos de _ALLERGEN_SYNONYMS no los reconoce _DIET_*_TERMS ni una "
        f"excepción documentada en _PARITY_TERM_EXCEPTIONS: {sorted(solo_allergen)}"
    )
    assert not solo_diet, (
        f"[{clase}] estos términos de _DIET_*_TERMS no los reconoce _ALLERGEN_SYNONYMS ni una "
        f"excepción documentada en _PARITY_TERM_EXCEPTIONS: {sorted(solo_diet)}"
    )


def test_lactosa_es_mas_estrecha_que_lacteos_a_proposito(go):
    """`_ALLERGEN_SYNONYMS['lactosa']` (intolerancia — solo importa el AZÚCAR) es
    deliberadamente más estrecho que `['lacteos']` (alergia a la PROTEÍNA — importa todo
    derivado): 'ghee' (mantequilla clarificada, lactosa removida en el proceso) y 'caseina'/
    'caseinato'/'proteina de suero'/'proteina de leche' (proteínas, no azúcar) están en
    `lacteos` pero NO en `lactosa` a propósito. Control negativo: NO es la misma asimetría que
    el drift de mejillón — aquí las DOS categorías viven dentro del vocabulario #1, no hay
    contraparte de dieta que deba igualarlas, y encogerla haría que un intolerante a lactosa
    evitara innecesariamente proteína de suero aislada (que SÍ suele ser baja/libre de lactosa).
    Este test ancla que la asimetría es intencional, no que deba cerrarse."""
    lactosa = set(t.lower() for t in go._ALLERGEN_SYNONYMS["lactosa"])
    lacteos = set(t.lower() for t in go._ALLERGEN_SYNONYMS["lacteos"])
    assert lactosa < lacteos, "lactosa debe seguir siendo subconjunto ESTRICTO de lacteos"
    assert "ghee" in lacteos and "ghee" not in lactosa
    assert "caseinato" in lacteos and "caseinato" not in lactosa


# ── G3. Vocabulario #3 (constants.py catch-alls) — cita al guard existente + confirma neighbors ─
#
# constants.py NO se tocó en este task (el intento con 'avena' se revirtió, ver G1/G4) — su
# paridad contra #1 ya la enforza
# `test_p2_catalog_filter_ssot.py::test_paridad_filtro_vs_escaner_canonico` (oráculo
# `_scan_allergen_violations`, barrido en "neighbors green"). Este test confirma PUNTUALMENTE que
# las altas que SÍ se quedaron (mejillón/vieira en mariscos, arenque en pescado) no generan un
# sobreviviente-violación nuevo: 'Mejillones' y 'Arenque' YA vivían en el catch-all de
# constants.py (P1-VARIETY-CATALOG-POOLS, anterior a este task) — el oráculo reforzado no
# encuentra nada nuevo que excluir, así que el test de OTRO archivo no se ve afectado.

def test_altas_de_este_task_ya_estaban_cubiertas_por_constants_catchall():
    """[vocabulario #3] Control negativo: si esto fallara, algo más (no este task) movió el
    catch-all de `constants._get_fast_filtered_catalogs`."""
    from constants import _get_fast_filtered_catalogs

    con_mariscos = [x for pool in _get_fast_filtered_catalogs(("Mariscos",), (), "") for x in pool]
    assert not any("mejillon" in str(x).lower() for x in con_mariscos), (
        "'Mejillones' sobrevive al chip 'Mariscos' en constants.py"
    )
    con_pescado = [x for pool in _get_fast_filtered_catalogs(("Pescado",), (), "") for x in pool]
    assert not any(str(x).lower() == "arenque" for x in con_pescado), (
        "'Arenque' sobrevive al chip 'Pescado' en constants.py"
    )


# ── G4. Vocabulario #4 (condition_rules.py) — el backstop conoce cada objetivo de sustitución ──
#
# `_ALLERGEN_SHELLFISH_SUBS`/`_ALLERGEN_FISH_SUBS`/`_ALLERGEN_SOY_SUBS`/`_ALLERGEN_GLUTEN_SUBS`
# son las ÚNICAS 4 categorías que el motor de sustitución quirúrgica modela (por diseño,
# "DECISIÓN HONESTA" ~condition_rules.py:685 — lácteos/huevo/maní/frutos secos quedan fuera:
# sin target GF/libre-de-alérgeno en el catálogo es-DO). Esa exclusión de CATEGORÍA es la
# excepción documentada para #4; DENTRO de sus 4 categorías, todo lo que #4 trata como objetivo
# de sustitución (evidencia de que el LLM SÍ puede generar ese texto) debe tener backstop en #1.
_V4_EXTRACTORS = {
    # [fix-round 1 · finding 3] 'mariscos'/'pescado' indexaban SOLO la fila [0] (hardcode) mientras
    # soya/gluten ya iteraban TODAS las filas — exactamente la clase de hueco silencioso-si-la-
    # tabla-crece que este archivo existe para prevenir. Hoy ambas tienen 1 fila (`[0][0]` y
    # `[t for sub in ... for t in sub[0]]` son equivalentes ahora), pero si `_ALLERGEN_SHELLFISH_SUBS`/
    # `_ALLERGEN_FISH_SUBS` ganan una 2ª fila (p.ej. T5-T8 añadiendo un swap nuevo) el hardcode
    # anterior habría dejado de verla en silencio — las 4 clases usan la misma forma ahora.
    "mariscos": lambda cr: [t for sub in cr._ALLERGEN_SHELLFISH_SUBS for t in sub[0]],
    "pescado": lambda cr: [t for sub in cr._ALLERGEN_FISH_SUBS for t in sub[0]],
    "soya": lambda cr: [t for sub in cr._ALLERGEN_SOY_SUBS for t in sub[0]],
    "gluten": lambda cr: [t for sub in cr._ALLERGEN_GLUTEN_SUBS for t in sub[0]],
}

# Excepción documentada POR TÉRMINO (mismo mecanismo que `_PARITY_TERM_EXCEPTIONS` en G2) — mecanismo
# vivo para el día en que una asimetría legítima aparezca, JAMÁS para silenciar un gap real.
# [fix-round 1 · 2026-08-17] 'avena' (bare + sus 3 compuestos) YA NO es excepción: fix-round 1
# sumó 'avena' a `_ALLERGEN_SYNONYMS['gluten']` con una excusa forward scoped-a-gluten (ver
# graph_orchestrator.py `_GLUTEN_FORWARD_EXCUSE_RX`) — los 4 compuestos matchean por substring
# 'avena' y quedan CUBIERTOS, no excepcionados. Vacío en las 4 clases: cero gaps conocidos hoy.
_V4_TERM_EXCEPTIONS = {
    "mariscos": set(), "pescado": set(), "soya": set(), "gluten": set(),
}


@pytest.mark.parametrize("clase_allergen", list(_V4_EXTRACTORS.keys()))
def test_backstop_cubre_los_objetivos_de_sustitucion_de_condition_rules(go, condrules, clase_allergen):
    """[G4 · vocabulario #4] Si `collect_allergen_substitutions` falla en sustituir (bug, texto
    del LLM que no matchea sus tokens estrechos a propósito), `_scan_allergen_violations` es la
    ÚNICA red que queda. Pre-fix (T4) esta clase estaba rota para 'gluten' (tostada/macarrón/
    coditos/fideo/tallarín/penne/ravioli/ñoqui/tortilla de harina, y avena — cerrada en
    fix-round 1) y 'mariscos'/'pescado' (gamba/arenque)."""
    v4_terms = _V4_EXTRACTORS[clase_allergen](condrules)
    v1_terms = go._ALLERGEN_SYNONYMS[clase_allergen]
    faltan = set(_uncovered(v4_terms, v1_terms)) - _V4_TERM_EXCEPTIONS[clase_allergen]
    assert not faltan, (
        f"condition_rules sustituye estos términos como riesgo de {clase_allergen!r} pero "
        f"_ALLERGEN_SYNONYMS[{clase_allergen!r}] no los reconoce ni hay excepción documentada en "
        f"_V4_TERM_EXCEPTIONS: {sorted(faltan)}"
    )


def test_v4_no_modela_lacteos_huevo_mani_frutos_secos_a_proposito(condrules):
    """Control negativo de la excepción de CATEGORÍA: confirma que la ausencia es la
    'DECISIÓN HONESTA' documentada (sin target GF/libre-de-alérgeno en el catálogo es-DO), no un
    olvido — si algún día alguien añade `_ALLERGEN_DAIRY_SUBS` sin querer decidirlo a propósito,
    este test deja de fallar EN SILENCIO (no hay assert que lo prohíba estructuralmente porque
    prohibir código futuro no es el trabajo de un test; lo que ancla es la RAZÓN documentada)."""
    src = condrules.__file__
    texto = Path(src).read_text(encoding="utf-8")
    assert "DECISIÓN HONESTA" in texto
    assert "lácteos, huevo, maní y frutos secos NO se sustituyen aquí" in texto


# ── G5. EL ALTA-HOOK — el backstop conoce cada alimento peligroso del catálogo VIVO ────────────
#
# [P1-COUNTRY-SYSTEM-F2 · T4 · e2e] Único test de TODO este archivo que toca Neon (el resto es
# mockeado/parser, ver docstring del módulo) — lección del repo: pool abierto explícitamente
# (`db_core.connection_pool.open()`), `SELECT` read-only, marcado `@pytest.mark.e2e` para que el
# gate rápido (`-m "not e2e"`) no dependa de conectividad DB, consistente con el resto de la
# suite (`tests/conftest.py::_guard_test_writes_to_prod`). Este es el CHECKLIST-ANCHOR que T5-T8
# deben mantener verde: si el nombre de un alimento nuevo (cualquier país) matchea una clase de
# seguridad en OTRO vocabulario (dieta, sustitución de condition_rules) pero `_ALLERGEN_SYNONYMS`
# no lo reconoce, este test lo nombra explícitamente — la corrección es añadir el sinónimo
# faltante a `_ALLERGEN_SYNONYMS[<clase>]` ANTES de mergear esa alta.

# Alternativas PLANT-BASED de un producto animal — la MISMA excusa de adyacencia que
# `_PLANT_ADJ_EXCUSE_RX` aplica en producción (`_scan_allergen_violations`): 'Leche de coco' no
# viola una alergia a LÁCTEOS (el alérgico a coco matchea por su propio término). No son gaps del
# backstop, es la excusa funcionando — documentadas aquí para que el test no las reporte como
# falsas alarmas.
_G5_EXCUSADOS_PLANT_ADJ = {
    ("lacteos", "leche de almendras"), ("lacteos", "leche de avena"), ("lacteos", "leche de coco"),
    ("lacteos", "leche de soya"), ("lacteos", "mantequilla de almendras"),
    ("lacteos", "yogur de coco"), ("lacteos", "mantequilla de mani"),
    ("lactosa", "leche de almendras"), ("lactosa", "leche de avena"), ("lactosa", "leche de coco"),
    ("lactosa", "leche de soya"), ("lactosa", "mantequilla de almendras"),
    ("lactosa", "yogur de coco"), ("lactosa", "mantequilla de mani"),
    # 'Mantequilla de maní' NO se excusa para la clase 'mani': ES el alérgeno (maní no es la
    # base plant-adjacent que excusa OTRO alérgeno, es el alérgeno mismo).
}

# [fix-round 1 · 2026-08-17 · el hueco de arriba SE CERRÓ, no se excusó] T4 tenía aquí
# `_G5_EXCUSADOS_AVENA_GLUTEN_DECISION = {("gluten", "avena"), ("gluten", "leche de avena")}`:
# 'Avena'/'Leche de avena' matcheaban el probe pero `_ALLERGEN_SYNONYMS['gluten']` no las conocía
# a propósito (T4 había rechazado sumar 'avena' bare — colisión con P1-ALLERGEN-NEGATION-EXCUSE).
# El review de fix-round 1 sumó 'avena' con una excusa FORWARD scoped-a-gluten
# (`_GLUTEN_FORWARD_EXCUSE_RX`, graph_orchestrator.py) — verificado EN VIVO contra el catálogo
# real (script de prueba, no fixture): ambas filas SÍ disparan `clinical_backstop_for_meal` hoy.
# El set de exclusión queda ELIMINADO (no vaciado): dejarlo vacío invitaría a repoblarlo con
# "excepciones" que en realidad son gaps — la dirección de este guard es sumar cobertura, nunca
# exceptions. Si esto revive como gap real, `faltantes` lo nombrará explícitamente.


@pytest.mark.e2e
def test_backstop_conoce_cada_alimento_peligroso_del_catalogo_vivo():
    """[G5 · el alta-hook, contrato T4 ítem 3] Query read-only a `master_ingredients` (pool
    abierto explícitamente). Para cada clase de seguridad, une los tokens que CUALQUIER
    vocabulario hermano (#2 dieta, #4 sustitución) ya reconoce como peligrosos + los propios de
    #1, y verifica que todo nombre de catálogo que matchee alguno de esos tokens SÍ dispare
    `clinical_backstop_for_meal` para la alergia correspondiente.

    Hallazgo EN VIVO de este task (T4 pre-fix, 206 filas en `master_ingredients`): 'Mejillones'
    (mariscos) y 'Arenque' (pescado) son alimentos catalogados HOY cuyo nombre ya vivía en un
    vocabulario hermano (#2 dieta) pero `_ALLERGEN_SYNONYMS` no los reconocía —
    `clinical_backstop_for_meal` los dejaba pasar en silencio; ambos cerrados en T4. 'Yogur de
    coco'/'Mantequilla de maní' matchean el probe pero son EXCUSA correcta (plant-adjacency), no
    gap — ver `_G5_EXCUSADOS_PLANT_ADJ`. 'Avena'/'Leche de avena' (gluten) SÍ tenían el mismo hueco
    (T4 las dejó sin backstop a propósito) — fix-round 1 lo CERRÓ (ver comentario arriba); ya no
    hay exclusión que las cubra, así que este test las verifica como cualquier otro alimento."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — faltan NEON_DATABASE_URL/.env (e2e, no bloquea el gate)")
    db_core.connection_pool.open()
    from db_core import execute_sql_query
    import graph_orchestrator as go
    import condition_rules as cr

    rows = execute_sql_query("SELECT name FROM master_ingredients", fetch_all=True)
    assert rows, (
        "master_ingredients vino vacío con el pool abierto — si esto falla, el catálogo real "
        "tiene 0 filas o el pool no abrió de verdad (lección del repo: mides el vacío, no el "
        "sistema)"
    )
    nombres = [r["name"] for r in rows if r.get("name")]

    # [fix-round 1 · auto-detectado corriendo el test en vivo] 'mariscos' y 'pescado' NO pueden
    # unir `_DIET_SEAFOOD_TERMS` completo: esa tupla mezcla pescado+marisco A PROPÓSITO (una
    # dieta vegana prohíbe ambos por igual, P1-VARIETY-CATALOG-POOLS), pero `_ALLERGEN_SYNONYMS`
    # SÍ distingue las dos alergias IgE. Unirla aquí habría hecho que el probe de 'mariscos'
    # incluyera "bacalao"/"salmón" (peces, no mariscos) y viceversa — falsos positivos, no gaps
    # reales (el G2 de arriba YA garantiza `mariscos ∪ pescado` ⊇ `_DIET_SEAFOOD_TERMS`, así que
    # re-unirla aquí es además redundante). Cada clase usa SOLO su propio vocabulario #1 + el
    # sub-conjunto de #4 que le corresponde (shellfish/fish ya vienen separados en condition_rules).
    clases_tokens = {
        "mariscos": set(cr._ALLERGEN_SHELLFISH_SUBS[0][0]) | set(go._ALLERGEN_SYNONYMS["mariscos"]),
        "pescado": set(cr._ALLERGEN_FISH_SUBS[0][0]) | set(go._ALLERGEN_SYNONYMS["pescado"]),
        "gluten": {t for sub in cr._ALLERGEN_GLUTEN_SUBS for t in sub[0]}
                  | set(go._ALLERGEN_SYNONYMS["gluten"]),
        "lacteos": set(go._DIET_DAIRY_TERMS) | set(go._ALLERGEN_SYNONYMS["lacteos"]),
        "lactosa": set(go._DIET_DAIRY_TERMS) | set(go._ALLERGEN_SYNONYMS["lactosa"]),
        "huevo": set(go._DIET_EGG_TERMS) | set(go._ALLERGEN_SYNONYMS["huevo"]),
        "soya": {t for sub in cr._ALLERGEN_SOY_SUBS for t in sub[0]} | set(go._ALLERGEN_SYNONYMS["soya"]),
        "frutos secos": set(go._ALLERGEN_SYNONYMS["frutos secos"]),
        "mani": set(go._ALLERGEN_SYNONYMS["mani"]),
    }

    faltantes = []
    for clase, tokens in clases_tokens.items():
        for nombre in nombres:
            if not _covered(tokens, nombre):
                continue
            # accent-stripped: 'Mantequilla de maní' vs la entrada escrita a mano sin tilde.
            clave = (clase, constants.strip_accents(nombre).strip().lower())
            if clave in _G5_EXCUSADOS_PLANT_ADJ:
                continue
            meal = {"name": "probe", "ingredients": [nombre]}
            if not go.clinical_backstop_for_meal(meal, allergies=[clase], diet_type=None):
                faltantes.append((clase, nombre))

    assert not faltantes, (
        f"{len(faltantes)} alimento(s) del catálogo VIVO matchean una clase de seguridad en un "
        f"vocabulario hermano pero _ALLERGEN_SYNONYMS no los reconoce (el backstop los dejaría "
        f"pasar en silencio): {faltantes}. Añade el sinónimo faltante a "
        f"_ALLERGEN_SYNONYMS[<clase>] (graph_orchestrator.py) antes de mergear esta alta."
    )


# ════════════════════════════════════════════════════════════════════════════════════════════
# H. Task 5 — Catálogo España (dirigido por el JSON de T1, sin cuota: 32 DROP = 32 altas)
# ════════════════════════════════════════════════════════════════════════════════════════════
#
# T1 clasificó 80 alimentos/platos curados de ES contra el catálogo vivo: 48 RESUELVE-BIEN, 0
# SUSTITUCION-SILENCIOSA, 32 DROP (`backend/data/country_gaps/es.json`). Esta sección ancla el
# cierre exacto de esos 32 DROP:
#   H1-H3  `dish_templates_es.json` (55 plantillas, constituents en nombres EXACTOS del catálogo
#          + gramos crudos) — espejo de `dish_templates.json` (RD), consumido por
#          `_culinary_judge_rubric_for_country`.
#   H4-H5  Golden fixture: tortilla española conserva su huevo en `ingredients`/constituents Y
#          sobrevive al agregador de compras; Jamón serrano (SIN precio RD) sobrevive vía la
#          generalización de P1-BAKING-STAPLES en vez de dropearse en silencio.
#   H6     `is_country_catalog_unpriced_item` reconoce las 32 altas.
#   H7-H9  `COUNTRY_POOLS['ES']` + `_get_fast_filtered_catalogs(..., country=)` — byte-identidad
#          sin `country`/con 'DO', pool propio con ES.
#   H10-H11 `_build_filtered_edge_recipe_day` gana `country` (default 'DO') y los 4 call sites de
#          `cron_tasks.py` lo derivan UNA vez y lo reusan (parser, mismo patrón T2).
#   H12-H13 `_culinary_judge_rubric_for_country`/`_dish_templates_path_for_country`: DO `is`-
#          idéntico a `_CULINARY_JUDGE_RUBRIC`; ES sustituye el bloque de ejemplos; país sin
#          archivo propio (MX) cae al fallback RD.
#   H14    Golden fixture: un día ES con arroz fuera de horario pasa como SOFT (hard=False) — la
#          MISMA combinación dish/slot es HARD para DO (control de que el mecanismo T4 realmente
#          discrimina, no solo que ES no truene).
#   H15    `pantry_names_match` reconoce 5 de las altas con prefijos de cantidad/plural/case.
#   H16    Anchor NARROW de las altas T5 en los 4 vocabularios (más preciso que el G2 genérico:
#          si alguien revierte SOLO mis términos, este test falla con el nombre exacto).
#   H17    e2e: las 32 filas existen en `master_ingredients` vivo, SIN precio, con `fdc_id` real.

# ── H0. El fix del harness: "verificado" en modo --country NO debe exigir precio RD ────────────
#
# [hallazgo real, medido re-corriendo el harness contra el catálogo YA con las 32 altas] La
# primera re-corrida de `country_catalog_gap.py --country ES` tras insertar las 32 filas seguía
# reportando 32 DROP — IDÉNTICO al pre-alta. Causa raíz: `classify_food` marcaba "verificado" con
# `sc._get_verified_shopping_name_set()`, que exige `price_per_lb>0 OR price_per_unit>0` — el
# MISMO gate que `MEALFIT_VERIFIED_INGREDIENTS_ONLY` usa en producción para decidir qué entra a
# una lista de COMPRAS. Las 32 altas de T5 son SIN precio RD A PROPÓSITO (España es país beta,
# `pricing_mode='beta_no_prices'`) — nunca iban a "tener precio", así que ese gate las clasificaría
# DROP para siempre, sin importar cuán bien resuelva `normalize_name`. La pregunta de Task 1
# ("¿el catálogo tiene este alimento con nutrición real?") es DISTINTA de la pregunta de
# `MEALFIT_VERIFIED_INGREDIENTS_ONLY` ("¿esto se puede costear en una lista RD?") — nunca se tocó
# ese mecanismo (Global Constraint del plan); se le dio a `classify_food` un segundo criterio de
# "verificado" OPCIONAL (`catalog_name_set`, default `None` ⇒ comportamiento IDÉNTICO al pre-fix,
# las 8 unit tests de la sección A que mockean `_get_verified_shopping_name_set` directamente
# siguen verdes sin tocarlas) que `run_country_mode` puebla con TODO `master_ingredients`, precio
# incluido o no.

def test_classify_food_sin_catalog_name_set_preserva_comportamiento_pre_fix(monkeypatch):
    """Default `catalog_name_set=None` ⇒ sigue consultando `sc._get_verified_shopping_name_set()`
    — byte-identidad con las 8 unit tests de la sección A (ninguna se tocó)."""
    mod = _load()
    monkeypatch.setattr(mod.sc, "normalize_name", lambda name: "Tomate")
    monkeypatch.setattr(mod.sc, "_get_verified_shopping_name_set", lambda: {"tomate"})
    r_sin_precio = mod.classify_food("tomate")
    monkeypatch.setattr(mod.sc, "_get_verified_shopping_name_set", lambda: set())
    r_no_verificado = mod.classify_food("tomate")
    assert r_sin_precio["verdict"] == "RESUELVE-BIEN"
    assert r_no_verificado["verdict"] == "DROP"


def test_classify_food_con_catalog_name_set_ignora_get_verified_shopping_name_set(monkeypatch):
    """Cuando se pasa `catalog_name_set` explícito, ESE set decide — `_get_verified_shopping_name_set`
    (precio-RD) queda IGNORADO. Simula la alta ES: 'jamon serrano' resuelve exacto pero NO tiene
    precio (el mock de _get_verified_shopping_name_set está VACÍO a propósito)."""
    mod = _load()
    monkeypatch.setattr(mod.sc, "normalize_name", lambda name: "Jamón serrano")
    monkeypatch.setattr(mod.sc, "_get_verified_shopping_name_set", lambda: set())  # sin precio

    r = mod.classify_food("Jamón serrano", catalog_name_set={"jamon serrano"})

    assert r["verdict"] == "RESUELVE-BIEN", (
        "con catalog_name_set explícito, un alimento SIN precio pero EN el catálogo debe "
        "resolver bien — la pregunta de Task 1 no es sobre precio"
    )
    assert r["tier"] == "exact"


def test_classify_food_con_catalog_name_set_sigue_dropeando_lo_genuinamente_ausente(monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod.sc, "normalize_name", lambda name: "Cosa Sin Catalogar")
    r = mod.classify_food("laurel", catalog_name_set={"jamon serrano", "gambas"})
    assert r["verdict"] == "DROP"


@pytest.mark.e2e
def test_catalog_name_set_including_unpriced_incluye_las_32_altas_y_mas_que_el_priced():
    """[e2e] Contra el catálogo VIVO: el set 'incluye-sin-precio' debe ser un SUPERSET estricto
    del set 'verificado-con-precio' — las 32 altas T5 aparecen en uno pero no en el otro."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    mod = _load()

    con_precio = mod.sc._get_verified_shopping_name_set()
    sin_precio_incluido = mod._catalog_name_set_including_unpriced()

    assert con_precio < sin_precio_incluido, "debe ser superset ESTRICTO (las 32 altas T5 sin precio)"
    assert "jamon serrano" in sin_precio_incluido
    assert "jamon serrano" not in con_precio


_DISH_TEMPLATES_ES_NAMES = frozenset({
    "Jamón serrano", "Jamón ibérico", "Chorizo español", "Morcilla", "Lomo embuchado",
    "Panceta ibérica", "Gambas", "Almejas", "Boquerones", "Anchoas", "Cordero", "Requesón",
    "Cuajada", "Nata", "Judías blancas", "Judías pintas", "Acelgas", "Fideos", "Membrillo",
    "Higo", "Azafrán", "Alioli", "Turrón", "Mazapán", "Sobrasada", "Butifarra", "Percebes",
    "Vieira", "Chistorra", "Piñones", "Almendra marcona", "Membrillo dulce",
})

# [P1-COUNTRY-SYSTEM-F2 · T6 · 2026-08-17] Las 46 altas de catálogo MX/CO de esta task (nombre
# CANÓNICO de fila — Achiote y Panela cuentan UNA vez cada una aunque satisfagan un item curado
# de ambos países). Espejo de `_DISH_TEMPLATES_ES_NAMES` de arriba — mismo propósito: excluir del
# sweep de "no debe reconocerlas" (son las que SÍ deben reconocerse).
_DISH_TEMPLATES_MX_CO_NAMES = frozenset({
    "Tortilla de maíz", "Chile jalapeño", "Chile serrano", "Chile poblano", "Chile chipotle",
    "Chile guajillo", "Chile ancho", "Chile habanero", "Chile de árbol", "Chile pasilla",
    "Chile mulato", "Nopal", "Jícama", "Epazote", "Chorizo mexicano", "Chorizo verde", "Cecina",
    "Frijoles refritos", "Crema mexicana", "Tuna de nopal", "Flor de Jamaica", "Xoconostle",
    "Achiote", "Hoja santa", "Chocolate de mesa", "Panela", "Huitlacoche", "Chicharrón",
    "Chorizo santarrosano", "Trucha", "Chontaduro", "Frijol cargamanto", "Suero costeño",
    "Guascas", "Arracacha", "Lulo", "Curuba", "Uchuva", "Arequipe", "Natilla", "Champús",
    "Gallina criolla", "Borojó", "Feijoa", "Granadilla", "Mora",
})


@pytest.fixture(scope="module")
def sc():
    """`shopping_calculator` — mismo motivo que `country_catalog_gap.py` lo importa por módulo
    (no toca Neon a nivel de import, solo dentro de funciones que abren el pool explícitamente)."""
    import shopping_calculator as _sc
    return _sc


def _load_dish_templates_es() -> dict:
    with open(_DISH_TEMPLATES_ES_JSON, encoding="utf-8") as f:
        return json.load(f)


# ── H1-H3. dish_templates_es.json — forma, constituents, golden de la tortilla ─────────────────

def test_dish_templates_es_json_existe_con_forma_esperada():
    assert _DISH_TEMPLATES_ES_JSON.exists(), "backend/data/dish_templates_es.json debe existir"
    data = _load_dish_templates_es()
    templates = data.get("templates")
    assert isinstance(templates, list)
    assert 40 <= len(templates) <= 60, f"{len(templates)} plantillas — fuera del rango ~40-60 del brief"
    nombres = [t.get("name") for t in templates]
    assert all(isinstance(n, str) and n.strip() for n in nombres), "una plantilla sin name"
    assert len(nombres) == len(set(nombres)), "nombres de plantilla duplicados"
    for t in templates:
        assert isinstance(t.get("slots"), list) and t["slots"], f"{t.get('name')!r} sin slots"
        assert set(t["slots"]) <= {"desayuno", "almuerzo", "cena", "merienda"}, (
            f"{t.get('name')!r} tiene un slot fuera del canon de 4"
        )
        constituents = t.get("constituents")
        assert isinstance(constituents, list) and constituents, f"{t.get('name')!r} sin constituents"
        for c in constituents:
            assert isinstance(c.get("name"), str) and c["name"].strip(), f"{t['name']!r}: constituent sin name"
            assert isinstance(c.get("grams"), (int, float)) and c["grams"] > 0, (
                f"{t['name']!r}: constituent {c.get('name')!r} sin gramos > 0"
            )


def test_dish_templates_es_arroz_pasta_como_base_nunca_en_desayuno_ni_cena():
    """Mismo SSOT que la regla dura del juez (`_build_culinary_judge_rubric`): arroz/pasta como
    BASE nunca van en desayuno ni cena. Ancla que las plantillas curadas de T5 (paella, sopa de
    fideos) respetan la regla que su propio `_note` dice heredar — si alguien añade una plantilla
    nueva con `base` arroz/pasta mal slotteada, este test la atrapa antes que el juez LLM."""
    data = _load_dish_templates_es()
    ofensoras = [
        t["name"] for t in data["templates"]
        if t.get("base") in ("arroz", "pasta") and set(t.get("slots", [])) & {"desayuno", "cena"}
    ]
    assert not ofensoras, f"plantillas con base arroz/pasta en desayuno/cena: {ofensoras}"


def test_tortilla_espanola_conserva_su_huevo_en_constituents():
    """[Golden fixture · contrato de la task] Las variantes de tortilla española del archivo
    ('Tortilla española...' y 'Tortilla de patatas...') DEBEN listar 'Huevo' entre sus
    constituents con gramos > 0 — es el ingrediente que le da nombre al plato (regla
    `nombre_no_corresponde` del propio juez culinario)."""
    data = _load_dish_templates_es()
    tortillas = [t for t in data["templates"] if t["name"].startswith(("Tortilla española", "Tortilla de patatas"))]
    assert len(tortillas) >= 2, "esperaba al menos 2 variantes de tortilla en el archivo"
    for t in tortillas:
        huevos = [c for c in t["constituents"] if c["name"] == "Huevo"]
        assert huevos and huevos[0]["grams"] > 0, (
            f"{t['name']!r} debe conservar 'Huevo' en constituents con gramos > 0"
        )


@pytest.mark.e2e
def test_dish_templates_es_constituents_resuelven_al_catalogo_vivo():
    """[H2 · e2e] Contrato "nombres EXACTOS del catálogo" — cada `constituents[].name` de las 55
    plantillas debe ser un `name` LITERAL de `master_ingredients` (no un alias que resuelva vía
    `normalize_name`: EXACTO, para que un futuro consumidor pueda indexar por igualdad directa).
    Mismo patrón e2e que G5 (pool abierto explícito, skip si no hay conectividad)."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — faltan NEON_DATABASE_URL/.env (e2e, no bloquea el gate)")
    db_core.connection_pool.open()
    from db_core import execute_sql_query

    data = _load_dish_templates_es()
    nombres_usados = {c["name"] for t in data["templates"] for c in t["constituents"]}
    rows = execute_sql_query("SELECT name FROM master_ingredients", fetch_all=True)
    assert rows, "master_ingredients vino vacío con el pool abierto"
    catalogo = {r["name"] for r in rows if r.get("name")}

    faltantes = sorted(nombres_usados - catalogo)
    assert not faltantes, (
        f"{len(faltantes)} nombre(s) de constituents en dish_templates_es.json NO son un `name` "
        f"exacto de master_ingredients: {faltantes}"
    )


# ── H4-H5. Golden fixture funcional: tortilla sobrevive, Jamón serrano no se dropea en silencio ─

@pytest.mark.e2e
def test_huevo_de_la_tortilla_sobrevive_al_agregador_de_compras(sc):
    """[Golden fixture · funcional] El ingrediente 'Huevo' de 'Tortilla española con patata y
    cebolla' (constituents reales del archivo) llega intacto al agregador — nunca fue el riesgo
    (Huevo ya tenía precio antes de T5), pero ancla el camino completo plantilla→lista."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()

    data = _load_dish_templates_es()
    tortilla = next(t for t in data["templates"] if t["name"] == "Tortilla española con patata y cebolla")
    plan_ingredients = [f"{c['grams']} g de {c['name']}" for c in tortilla["constituents"]]

    result = sc.aggregate_and_deduct_shopping_list(plan_ingredients, structured=True)
    items = result if isinstance(result, list) else (result.get("items") or [])
    nombres = [it.get("name") for it in items]
    assert "Huevo" in nombres, f"'Huevo' no sobrevivió al agregador: {nombres}"


@pytest.mark.e2e
def test_jamon_serrano_no_se_dropea_en_silencio_via_unpriced_keep(sc, monkeypatch):
    """[Golden fixture · el gap real que T5 cierra] 'Jamón serrano' (alta T5, SIN precio RD a
    propósito) llega al agregador exactamente como cualquier ingrediente off-catálogo — sin el
    keep generalizado de P1-BAKING-STAPLES, `_is_verified_for_shopping` lo trataría como
    inventado por el LLM y lo dropearía en silencio (el modo de fallo original de
    P1-BAKING-STAPLES, ahora a escala de país). Verifica el nombre Y que quede SIN precio
    (`estimated_cost_rd` None) bajo su categoría propia — nunca con un precio RD inventado.

    `MEALFIT_VERIFIED_INGREDIENTS_ONLY` monkeypatcheado a 'true': el baseline de la suite lo
    fija 'false' (`conftest.py`, P1-VERIFIED-ONLY-DEFAULT-ON) precisamente para que tests de
    coherencia con ingredientes sintéticos no disparen el drop — este test SÍ quiere ejercer esa
    puerta, mismo patrón que `test_p3_verified_ingredients_only` (citado en el propio conftest)."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    monkeypatch.setenv("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "true")

    result = sc.aggregate_and_deduct_shopping_list(["40 g de Jamón serrano"], structured=True)
    items = result if isinstance(result, list) else (result.get("items") or [])
    jamon = next((it for it in items if it.get("name") == "Jamón serrano"), None)
    assert jamon is not None, "'Jamón serrano' fue dropeado del agregador — el keep no-op"
    assert jamon.get("estimated_cost_rd") is None, (
        "'Jamón serrano' no debe llevar un costo RD inventado"
    )
    assert jamon.get("display_category") == "CATÁLOGO SIN PRECIO"


@pytest.mark.e2e
def test_jamon_serrano_se_dropea_si_el_knob_de_keep_esta_apagado(sc, monkeypatch):
    """[Mutación viva vía knob · rollback documentado] Control negativo: con
    MEALFIT_VERIFIED_INGREDIENTS_ONLY=true (la puerta que activa el drop/keep, ver test de
    arriba) Y MEALFIT_COUNTRY_CATALOG_UNPRICED_KEEP=false el comportamiento REVIERTE al pre-T5
    (drop + WARNING) — confirma que el keep de arriba depende REALMENTE del mecanismo nuevo y no
    de otra vía (p.ej. que 'Jamón serrano' ya tuviera precio por accidente)."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    monkeypatch.setenv("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "true")
    monkeypatch.setenv("MEALFIT_COUNTRY_CATALOG_UNPRICED_KEEP", "false")

    result = sc.aggregate_and_deduct_shopping_list(["40 g de Jamón serrano"], structured=True)
    items = result if isinstance(result, list) else (result.get("items") or [])
    nombres = [it.get("name") for it in items]
    assert "Jamón serrano" not in nombres, (
        "con el knob apagado 'Jamón serrano' debe dropearse (comportamiento pre-T5) — si sigue "
        "presente, el keep no respeta MEALFIT_COUNTRY_CATALOG_UNPRICED_KEEP"
    )


# ── H6. is_country_catalog_unpriced_item reconoce las 32 altas ─────────────────────────────────

@pytest.mark.parametrize("nombre", sorted(_DISH_TEMPLATES_ES_NAMES))
def test_is_country_catalog_unpriced_item_reconoce_cada_alta(sc, nombre):
    assert sc.is_country_catalog_unpriced_item(nombre), f"{nombre!r} no reconocido como unpriced keep"


def test_is_country_catalog_unpriced_item_no_reconoce_alimento_do_generico(sc):
    """Control negativo: un alimento RD normal (con precio real) no debe entrar por accidente al
    keep — sería inofensivo (el gate previo, `_is_verified_for_shopping`, ya lo captura primero)
    pero confirmaría que el matcher es demasiado laxo."""
    assert not sc.is_country_catalog_unpriced_item("Pollo")
    assert not sc.is_country_catalog_unpriced_item("Arroz blanco")


def test_country_catalog_unpriced_keep_knob_default_true(sc):
    assert sc._country_catalog_unpriced_keep_enabled() is True


# ── H6-bis. fix-round 1 (review IMPORTANT): la colisión de substring 'pinones' ⊂ 'champinones' ──
#
# [fix-round 1 · 2026-08-17] El reviewer barrió los 32 tokens contra el catálogo pre-T5 completo
# (206 filas) + los 4 pools DOMINICAN_* (145 nombres) y encontró: `strip_accents('Champiñones')`
# = 'champinones' CONTIENE 'pinones' (Piñones ⊂ Champiñones) como substring bare — el matcher
# original (`tok in low`) marcaba 'Champiñones' (fila RD PRICED real, `DOMINICAN_VEGGIES_FATS`)
# como si fuera una alta sin precio de T5. 17ª colisión de substring documentada en el proyecto
# (sal⊂salsa, pollo⊂repollo, res⊂fresco...). Blast radius bajo hoy (el gate externo
# `not _is_verified_for_shopping` ya protege nombres bien formados — Champiñones SÍ tiene precio,
# así que nunca llega a esta rama en producción), pero el bug es real y el fix debe ser de raíz.

def test_champinones_no_colisiona_con_pinones_regresion(sc):
    """[el RED de este fix-round] 'Champiñones' (fila RD PRICED, `DOMINICAN_VEGGIES_FATS`) NUNCA
    debe reconocerse como alimento de catálogo-país sin precio — 'pinones' (Piñones, alta T5) es
    un substring bare de 'champinones' (accent-stripped) pero NO un token completo dentro de él."""
    assert not sc.is_country_catalog_unpriced_item("Champiñones"), (
        "'Champiñones' colisiona con el token 'pinones' (Piñones) por substring bare — "
        "is_country_catalog_unpriced_item debe matchear por TOKEN completo (word-boundary), no `in`"
    )


@pytest.mark.e2e
def test_is_country_catalog_unpriced_item_no_colisiona_con_ningun_nombre_del_catalogo_vivo_ni_pools(sc):
    """[el guard durable · review IMPORTANT] Sweep COMPLETO (no solo el caso puntual reportado):
    para cada nombre REAL del catálogo vivo (`master_ingredients`, sin importar precio) Y cada
    nombre de los 4 pools `DOMINICAN_*` + `COUNTRY_POOLS['ES']`, si ese nombre NO es una de las 32
    altas T5, `is_country_catalog_unpriced_item` debe ser False. Esto es lo que atrapa la
    colisión #18 antes de producción — el caso puntual de Champiñones (arriba) solo prueba que
    ESE caso está cerrado, este test prueba que la CLASE de bug está cerrada."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()

    rows = sc.get_master_ingredients() or []
    nombres_catalogo = {r.get("name") for r in rows if r.get("name")}

    nombres_pools = set()
    for pool in (constants.DOMINICAN_PROTEINS, constants.DOMINICAN_CARBS,
                 constants.DOMINICAN_VEGGIES_FATS, constants.DOMINICAN_FRUITS):
        nombres_pools.update(pool)
    for pool in constants.COUNTRY_POOLS.values():
        for key in ("proteins", "carbs", "veggies_fats", "fruits"):
            nombres_pools.update(pool.get(key) or [])

    # [P1-COUNTRY-SYSTEM-F2 · T6 · 2026-08-17] extendido: también excluye las 46 altas MX/CO —
    # ESAS sí deben reconocerse (son el propósito de sus tokens nuevos en
    # `_COUNTRY_CATALOG_UNPRICED_TOKENS`), así que no son "falsos positivos" si aparecen True.
    candidatos = (nombres_catalogo | nombres_pools) - _DISH_TEMPLATES_ES_NAMES - _DISH_TEMPLATES_MX_CO_NAMES

    falsos_positivos = sorted(n for n in candidatos if sc.is_country_catalog_unpriced_item(n))
    assert not falsos_positivos, (
        f"{len(falsos_positivos)} nombre(s) NO son una alta T5/T6 pero "
        f"is_country_catalog_unpriced_item los reconoce igual (colisión de substring/token): "
        f"{falsos_positivos}"
    )


# ── H7-H9. COUNTRY_POOLS['ES'] + _get_fast_filtered_catalogs(..., country=) ────────────────────

def test_country_pools_es_estructura():
    pool = constants.COUNTRY_POOLS.get("ES")
    assert isinstance(pool, dict)
    for key in ("proteins", "carbs", "veggies_fats", "fruits"):
        assert isinstance(pool.get(key), list) and pool[key], f"COUNTRY_POOLS['ES'][{key!r}] vacío"
        assert all(isinstance(x, str) and x.strip() for x in pool[key])


def test_get_fast_filtered_catalogs_sin_country_es_byte_identico_a_country_none_y_do():
    """[Byte-identidad, contrato global del plan] `country` es kwarg NUEVO — todo call site
    preexistente (ai_helpers.py/agent.py, y los tests de este mismo repo que llaman con 3
    posicionales) sigue devolviendo EXACTAMENTE el pool DOMINICAN_*."""
    casos = [((), (), ""), (("mariscos",), (), ""), ((), ("pescado",), ""), ((), (), "vegano")]
    for allergies, dislikes, diet in casos:
        base = constants._get_fast_filtered_catalogs(allergies, dislikes, diet)
        con_none = constants._get_fast_filtered_catalogs(allergies, dislikes, diet, country=None)
        con_do = constants._get_fast_filtered_catalogs(allergies, dislikes, diet, country="DO")
        assert base == con_none == con_do, f"diverge para {(allergies, dislikes, diet)!r}"


def test_get_fast_filtered_catalogs_es_usa_su_propio_pool():
    proteins_do, _, _, _ = constants._get_fast_filtered_catalogs((), (), "")
    proteins_es, carbs_es, veg_es, fruits_es = constants._get_fast_filtered_catalogs((), (), "", country="ES")
    assert proteins_es != proteins_do
    assert "Jamón serrano" in proteins_es
    assert "Gambas" in proteins_es
    assert set(proteins_es) == set(constants.COUNTRY_POOLS["ES"]["proteins"])
    assert set(carbs_es) == set(constants.COUNTRY_POOLS["ES"]["carbs"])
    assert set(veg_es) == set(constants.COUNTRY_POOLS["ES"]["veggies_fats"])
    assert set(fruits_es) == set(constants.COUNTRY_POOLS["ES"]["fruits"])


def test_get_fast_filtered_catalogs_es_sigue_aplicando_el_filtro_de_alergias():
    """El filtrado de alergias/dislikes/dieta corre ENCIMA del pool seleccionado — no es
    exclusivo de DOMINICAN_*. 'pescado' como dislike debe seguir excluyendo peces del pool ES
    (Bacalao, Pulpo — mariscos NO, ver P1-PESCADO-CATCHALL: 'pescado' a secas solo excluye peces)
    y conservar las carnes/embutidos que no son pescado."""
    proteins, _, _, _ = constants._get_fast_filtered_catalogs((), ("pescado",), "", country="ES")
    assert "Bacalao" not in proteins
    assert "Jamón serrano" in proteins
    assert "Gambas" in proteins, "'pescado' (a secas) no debe excluir mariscos — solo peces"


def test_get_fast_filtered_catalogs_es_vegano_deja_pasar_algunas_carnes_es_al_primer_filtro():
    """[hallazgo real, documentado — NO un gap de seguridad] Los catch-alls de dieta de
    `_get_fast_filtered_catalogs` ('carne'/'mariscos' expanden a tokens RD-específicos:
    'jamon'/'pollo'/'camaron'/etc, P1-VARIETY-CATALOG-POOLS 2026-06-27) NO conocen los nombres
    españoles nuevos (Chorizo español, Gambas, Morcilla...) — se les escapan al primer filtro,
    igual que 'al filtro aún se le escapan plurales' que P0-DEGRADED-SAFETY-SCAN ya documenta
    para el pool RD. Este test ANCLA el hallazgo (no lo esconde); el siguiente test prueba que
    la RED que sí los atrapa (`_sieve_catalog_for_safety`, segunda malla) los limpia todos."""
    proteins, _, _, _ = constants._get_fast_filtered_catalogs((), (), "vegano", country="ES")
    escapan = set(proteins) & set(constants.COUNTRY_POOLS["ES"]["proteins"])
    assert escapan, (
        "si esto se vacía, el primer filtro aprendió los nombres ES — actualiza este test para "
        "reflejar la mejora en vez de dejarlo como documentación de un hallazgo que ya no existe"
    )


def test_p0_degraded_safety_scan_limpia_lo_que_el_primer_filtro_de_vegano_deja_pasar_en_es():
    """[el mecanismo real de seguridad] `_sieve_catalog_for_safety` (P0-DEGRADED-SAFETY-SCAN, la
    'segunda malla' que YA existía antes de T5) usa `clinical_backstop_for_meal` →
    `_scan_diet_violations` → `_DIET_FLESH_TERMS`/`_DIET_SEAFOOD_TERMS` — vocabularios que T5 SÍ
    actualizó (H16). El pool ES para un vegano debe quedar VACÍO después de esta segunda malla,
    aunque el primer filtro (test de arriba) deje pasar charcutería/mariscos españoles."""
    import cron_tasks as ct
    proteins, _, _, _ = constants._get_fast_filtered_catalogs((), (), "vegano", country="ES")
    sieved = ct._sieve_catalog_for_safety(proteins, (), "vegano")
    assert sieved == [], f"la segunda malla debe vaciar el pool ES para un vegano: {sieved}"


# ── H10-H11. cron_tasks.py — _build_filtered_edge_recipe_day gana country (parser) ─────────────

def _cron_tasks_source() -> str:
    return _CRON_TASKS_PY.read_text(encoding="utf-8")


def _sin_comentarios(src: str) -> str:
    return "\n".join(line for line in src.splitlines() if not line.strip().startswith("#"))


def test_build_filtered_edge_recipe_day_gana_country_default_do():
    src = _sin_comentarios(_cron_tasks_source())
    assert 'country: str = "DO",' in src, (
        "_build_filtered_edge_recipe_day debe ganar country default 'DO' (preserva callers)"
    )
    ini = src.index("def _build_filtered_edge_recipe_day(")
    fin = src.index("\ndef ", ini + 10)
    cuerpo = src[ini:fin]
    assert "_get_fast_filtered_catalogs(" in cuerpo
    assert "country=country," in cuerpo, (
        "_build_filtered_edge_recipe_day debe threadear su propio country a _get_fast_filtered_catalogs"
    )


def test_edge_recipe_country_derivado_una_vez_y_reusado_en_los_4_callsites():
    """[T5 · patrón T2] `_edge_recipe_country` se deriva UNA vez (`country_for_form_data(form_data)`)
    y se reusa en los 4 call sites de `_build_filtered_edge_recipe_day` de este bloque del chunk
    worker — no 4 derivaciones independientes que podrían driftar."""
    src = _sin_comentarios(_cron_tasks_source())
    n_derivaciones = src.count("_edge_recipe_country = _country_for_form_data(form_data)")
    assert n_derivaciones == 1, f"esperaba 1 derivación de _edge_recipe_country, hay {n_derivaciones}"
    n_callsites = src.count("country=_edge_recipe_country,")
    assert n_callsites == 4, f"esperaba 4 call sites usando _edge_recipe_country, hay {n_callsites}"
    pos_derivacion = src.index("_edge_recipe_country = _country_for_form_data(form_data)")
    for m in re.finditer(r"country=_edge_recipe_country,", src):
        assert m.start() > pos_derivacion, "un call site usa _edge_recipe_country antes de derivarlo"


def test_edge_recipe_country_usa_country_for_form_data_ssot():
    """La derivación debe pasar por la ÚNICA puerta de lectura de país (`country_for_form_data`),
    no leer `form_data['country']` crudo — knob apagado ⇒ siempre 'DO' (byte-identidad)."""
    src = _sin_comentarios(_cron_tasks_source())
    assert "from constants import country_for_form_data as _country_for_form_data" in src


# ── H12-H13. _culinary_judge_rubric_for_country / _dish_templates_path_for_country ─────────────

def test_culinary_judge_rubric_do_es_is_identico_al_cacheado(go):
    assert go._culinary_judge_rubric_for_country("DO") is go._CULINARY_JUDGE_RUBRIC


def test_culinary_judge_rubric_es_sustituye_ejemplos_y_encabezado(go):
    rubric_es = go._culinary_judge_rubric_for_country("ES")
    rubric_do = go._culinary_judge_rubric_for_country("DO")
    assert rubric_es != rubric_do
    assert "Tortilla española" in rubric_es
    assert "Mangú" not in rubric_es, "los ejemplos dominicanos no deben sobrevivir en la variante ES"
    assert "PLATOS DE ESPAÑA" in rubric_es
    assert "cocina de España" in rubric_es


def test_dish_templates_path_for_country_es_usa_su_archivo_mx_cae_a_rd(go):
    """[actualizado T6 · 2026-08-17] MX y CO GANAN archivo propio en esta task (antes caían al
    fallback RD, como PR/US siguen haciendo hoy) — ver Sección I. PR (sin archivo dedicado, sin
    task que se lo dé todavía) es el control vivo de que el fallback explícito sigue funcionando
    para un país beta cualquiera sin archivo propio."""
    assert go._dish_templates_path_for_country("ES") == str(_DISH_TEMPLATES_ES_JSON)
    assert go._dish_templates_path_for_country("MX") == str(_DISH_TEMPLATES_MX_JSON)
    assert go._dish_templates_path_for_country("CO") == str(_DISH_TEMPLATES_CO_JSON)
    assert go._dish_templates_path_for_country("PR") == go._DO_DISH_TEMPLATES_PATH
    assert go._dish_templates_path_for_country("DO") == go._DO_DISH_TEMPLATES_PATH


def test_culinary_judge_rubric_pr_sin_archivo_propio_cae_a_ejemplos_rd(go):
    """[actualizado T6 · 2026-08-17] País beta SIN `dish_templates_<cc>.json` dedicado (PR, hoy
    — MX/CO YA NO son ejemplo de esto tras T6, ver Sección I) conserva los ejemplos dominicanos —
    fallback explícito, NO una excepción que rompa la rúbrica."""
    rubric_pr = go._culinary_judge_rubric_for_country("PR")
    assert "Mangú" in rubric_pr
    assert "Tortilla española" not in rubric_pr
    assert "Tacos de pollo" not in rubric_pr
    assert "Ajiaco" not in rubric_pr


# ── H14. Golden fixture: slot soft (ES) vs hard (DO) para el MISMO día ─────────────────────────

def _dia_es_con_paella_en_desayuno() -> list:
    return [{
        "day": 1,
        "meals": [
            {"meal": "Desayuno", "name": "Paella de mariscos con gambas y almejas",
             "ingredients": ["Gambas", "Almejas", "Arroz blanco"]},
            {"meal": "Almuerzo", "name": "Cocido madrileño con garbanzos y chorizo",
             "ingredients": ["Garbanzos", "Chorizo español"]},
            {"meal": "Cena", "name": "Tortilla española con patata y cebolla",
             "ingredients": ["Huevo", "Papa", "Cebolla"]},
            {"meal": "Merienda", "name": "Higos con jamón serrano",
             "ingredients": ["Higo", "Jamón serrano"]},
        ],
    }]


def test_dia_es_arroz_fuera_de_horario_pasa_como_soft_sin_forzar_retry(go, monkeypatch):
    """[Golden fixture · el contrato de la task] Un día ES con 'Paella...' (arroz) en Desayuno —
    violación real de la regla dura de horario — DEBE detectarse como SOFT (hard=False) para
    país ES: `slot_rules_for_country('ES')` softea TODA regla (T4). Soft no fuerza retry en el
    intento final del gate (`should_retry`) — 'pasa' significa exactamente esto: el mecanismo
    NO trata el día como un bloqueo duro, aunque la violación se siga reportando (telemetría).
    Requiere el knob ENCENDIDO (`country_for_form_data` solo lee `form_data['country']` con
    `MEALFIT_COUNTRY_SYSTEM=true` — knob apagado ⇒ 'DO' siempre, byte-identidad)."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    dias = _dia_es_con_paella_en_desayuno()
    violaciones = go._detect_slot_appropriateness(dias, {"country": "ES"})
    assert violaciones, "el día de prueba debe producir AL MENOS una violación (control positivo)"
    duras = [v for v in violaciones if v["hard"]]
    assert not duras, f"país ES no debe producir violaciones HARD: {duras}"


def test_el_mismo_dia_es_hard_para_do_control_de_que_el_mecanismo_discrimina(go, monkeypatch):
    """Control negativo del golden fixture de arriba: el MISMO día/dish/slot, con country='DO',
    debe producir violaciones HARD — si este test también diera soft, H14 no probaría nada (el
    detector podría estar simplemente roto para TODOS los países, no funcionando para ES)."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    dias = _dia_es_con_paella_en_desayuno()
    violaciones = go._detect_slot_appropriateness(dias, {"country": "DO"})
    assert violaciones, "control: el día debe violar también en DO"
    duras = [v for v in violaciones if v["hard"]]
    assert duras, "país DO debe seguir produciendo violaciones HARD (byte-identidad del mecanismo T4)"


def test_dia_es_con_knob_apagado_es_byte_identico_a_do_ignora_country_form_data(go):
    """Control de byte-identidad: SIN monkeypatch del knob (default apagado en este proceso de
    test), el mismo día con `form_data={'country': 'ES'}` debe comportarse EXACTAMENTE como DO
    — `country_for_form_data` no lee el campo `country` en absoluto con el knob apagado."""
    dias = _dia_es_con_paella_en_desayuno()
    violaciones_es_knob_off = go._detect_slot_appropriateness(dias, {"country": "ES"})
    violaciones_do = go._detect_slot_appropriateness(dias, {"country": "DO"})
    duras_es = [v["hard"] for v in violaciones_es_knob_off]
    duras_do = [v["hard"] for v in violaciones_do]
    assert duras_es == duras_do == [True, True], (
        "con el knob apagado, form_data['country']='ES' NO debe cambiar nada — mismo resultado que DO"
    )


# ── H15. pantry_names_match reconoce las altas T5 ───────────────────────────────────────────────

@pytest.mark.parametrize("texto_libre,fila_catalogo", [
    ("1 lata de anchoas", "Anchoas"),
    ("gambas", "Gambas"),
    ("200g de cuajada", "Cuajada"),
    ("2 higos", "Higo"),
    ("almendras marconas", "Almendra marcona"),
])
def test_pantry_names_match_reconoce_las_altas_es(texto_libre, fila_catalogo):
    assert constants.pantry_names_match(texto_libre, fila_catalogo), (
        f"pantry_names_match({texto_libre!r}, {fila_catalogo!r}) debe ser True"
    )


# ── H16. Anchor NARROW de las altas T5 en los 4 vocabularios ────────────────────────────────────

@pytest.mark.parametrize("clase,termino", [
    ("mariscos", "percebe"), ("mariscos", "percebes"),
    ("pescado", "boqueron"), ("pescado", "boquerones"),
    ("lacteos", "cuajada"), ("lactosa", "cuajada"), ("lactosa", "nata"),
    ("frutos secos", "pinon"), ("frutos secos", "pinones"),
])
def test_altas_t5_presentes_en_allergen_synonyms(go, clase, termino):
    assert termino in go._ALLERGEN_SYNONYMS[clase], f"{termino!r} ausente de _ALLERGEN_SYNONYMS[{clase!r}]"


@pytest.mark.parametrize("termino", ["percebe", "percebes", "boqueron", "boquerones"])
def test_altas_t5_presentes_en_diet_seafood_terms(go, termino):
    assert termino in go._DIET_SEAFOOD_TERMS


def test_cuajada_presente_en_diet_dairy_terms(go):
    assert "cuajada" in go._DIET_DAIRY_TERMS


@pytest.mark.parametrize("termino", [
    "morcilla", "panceta", "embuchado", "sobrasada", "butifarra", "chistorra", "cordero",
])
def test_altas_t5_charcuteria_presentes_en_diet_flesh_terms(go, termino):
    """Carne/embutidos NO son alérgeno IgE en este sistema (solo vocabulario #2 dieta) — sin
    esto, un plan ES con 'Morcilla'/'Sobrasada'/etc. pasaría el scan vegano/vegetariano limpio."""
    assert termino in go._DIET_FLESH_TERMS


@pytest.mark.parametrize("ingrediente", [
    "Sobrasada a la plancha", "Butifarra con alubias", "200 g de morcilla",
    "Chistorra frita", "Panceta ibérica curada", "Pierna de cordero asada",
])
def test_scan_diet_violations_detecta_las_altas_t5_charcuteria(go, ingrediente):
    """Funcional (no solo estructural): un vegano NO debe recibir estos alimentos — prueba el
    camino real, no solo que el token viva en la tupla."""
    plan = {"days": [{"meals": [{"name": "Almuerzo", "ingredients": [ingrediente]}]}]}
    assert go._scan_diet_violations(plan, "vegano"), f"'{ingrediente}' no fue detectado para vegano"


# ── H17. e2e — las 32 altas existen en el catálogo vivo, SIN precio, con fdc_id real ────────────

@pytest.mark.e2e
def test_32_altas_es_existen_en_catalogo_vivo_sin_precio_con_fdc_id():
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    from db_core import execute_sql_query

    rows = execute_sql_query(
        "SELECT name, price_per_lb, price_per_unit, fdc_id, nutrition_source "
        "FROM master_ingredients WHERE name = ANY(%s)",
        (list(_DISH_TEMPLATES_ES_NAMES),),
        fetch_all=True,
    ) or []
    por_nombre = {r["name"]: r for r in rows}

    faltantes = sorted(_DISH_TEMPLATES_ES_NAMES - set(por_nombre))
    assert not faltantes, f"altas T5 ausentes del catálogo vivo: {faltantes}"

    con_precio = [n for n, r in por_nombre.items()
                  if float(r["price_per_lb"] or 0) > 0 or float(r["price_per_unit"] or 0) > 0]
    assert not con_precio, f"altas T5 con precio RD (deberían estar en 0): {con_precio}"

    sin_fdc = [n for n, r in por_nombre.items() if not r.get("fdc_id")]
    assert not sin_fdc, f"altas T5 sin fdc_id (fuente no auditable): {sin_fdc}"

    no_usda = [n for n, r in por_nombre.items() if r.get("nutrition_source") != "usda"]
    assert not no_usda, f"altas T5 con nutrition_source != 'usda': {no_usda}"


# ══════════════════════════════════════════════════════════════════════════════════════════════
# SECCIÓN I (Task 6) — Catálogo México + Colombia, dirigido por el JSON de T1
# ══════════════════════════════════════════════════════════════════════════════════════════════
#
# Mismo contrato que T5/Sección H, para dos países a la vez. `country_gaps/mx.json` (30 DROP de
# 76) y `co.json` (26 DROP de 74) — medidos contra el catálogo POST-T5 (238 filas) — son la lista
# de trabajo. Regla FILA-vs-SINÓNIMO (el eje nuevo de esta task, ausente en T5 porque España no
# tenía homógrafos con el catálogo RD): mismo alimento con nombre regional ⇒ alias sobre la fila
# YA EXISTENTE (Jitomate→Tomate, Mazorca/Choclo→Maíz dulce en granos, Malanga→Yautía, Cuchuco de
# trigo→Bulgur, Chile cuaresmeño→Chile jalapeño, Piloncillo→Panela, Color (bijol)→Achiote — estos
# 3 últimos viven EMBEBIDOS en los aliases de una fila NUEVA del mismo lote, no en
# `scripts/data/synonyms_mx_co_2026_08_17.json`); alimento genuinamente distinto (aunque cercano
# nutricionalmente a otro, ej. Frijol cargamanto vs Frijoles pintos) ⇒ fila nueva. 46 filas nuevas
# + 6 operaciones de sinónimo vía script (+ 3 embebidas en altas del mismo lote) resuelven los 56
# DROP. 3 filas SIN fdc_id real (`nutrition_source='manual'`): Achiote, Flor de Jamaica, Hoja
# santa — ver docstring de `scripts/add_foods_mx_co_2026_08_17.py` y el reporte de la task.
#
# El SSOT real de "¿este sinónimo resuelve?" es `shopping_calculator.normalize_name` (lo que
# `classify_food`/el harness llaman) — NO `constants.GLOBAL_REVERSE_MAP` (construido desde
# PROTEIN_SYNONYMS/CARB_SYNONYMS/VEGGIE_FAT_SYNONYMS/FRUIT_SYNONYMS, consumido SOLO por
# `normalize_ingredient_for_tracking`, el sistema de fatiga/variedad — CERO call sites en
# `shopping_calculator.py`, verificado por grep). Los 6 sinónimos de esta task viven en el array
# `aliases` de `master_ingredients` (la fila EXISTENTE gana un alias nuevo), que es lo que
# `_construir_indice_alias` indexa — esta sección lo verifica contra el resolver REAL, no una
# tabla muerta. `test_global_reverse_map_...` (I8) ancla que la tabla NO se tocó (control negativo
# de que el trabajo real vive donde debe).

def _load_dish_templates_mx() -> dict:
    with open(_DISH_TEMPLATES_MX_JSON, encoding="utf-8") as f:
        return json.load(f)


def _load_dish_templates_co() -> dict:
    with open(_DISH_TEMPLATES_CO_JSON, encoding="utf-8") as f:
        return json.load(f)


# ── I1. dish_templates_mx.json / dish_templates_co.json — forma + regla dura de horario ─────────

@pytest.mark.parametrize("cc,loader", [("MX", _load_dish_templates_mx), ("CO", _load_dish_templates_co)])
def test_dish_templates_json_existe_con_forma_esperada(cc, loader):
    data = loader()
    templates = data.get("templates")
    assert isinstance(templates, list)
    assert 40 <= len(templates) <= 60, f"[{cc}] {len(templates)} plantillas — fuera del rango ~40-60 del brief"
    nombres = [t.get("name") for t in templates]
    assert all(isinstance(n, str) and n.strip() for n in nombres), f"[{cc}] una plantilla sin name"
    assert len(nombres) == len(set(nombres)), f"[{cc}] nombres de plantilla duplicados"
    for t in templates:
        assert isinstance(t.get("slots"), list) and t["slots"], f"[{cc}] {t.get('name')!r} sin slots"
        assert set(t["slots"]) <= {"desayuno", "almuerzo", "cena", "merienda"}, (
            f"[{cc}] {t.get('name')!r} tiene un slot fuera del canon de 4"
        )
        constituents = t.get("constituents")
        assert isinstance(constituents, list) and constituents, f"[{cc}] {t.get('name')!r} sin constituents"
        for c in constituents:
            assert isinstance(c.get("name"), str) and c["name"].strip(), (
                f"[{cc}] {t['name']!r}: constituent sin name"
            )
            assert isinstance(c.get("grams"), (int, float)) and c["grams"] > 0, (
                f"[{cc}] {t['name']!r}: constituent {c.get('name')!r} sin gramos > 0"
            )


@pytest.mark.parametrize("cc,loader", [("MX", _load_dish_templates_mx), ("CO", _load_dish_templates_co)])
def test_dish_templates_arroz_como_base_nunca_en_desayuno_ni_cena(cc, loader):
    """Mismo SSOT que la regla dura del juez — arroz como BASE nunca en desayuno ni cena. MX
    incluye platos rice-based (mole con arroz, camarones al mojo de ajo) TODOS en almuerzo; CO
    incluye 'Calentado paisa' (culturalmente también desayuno) deliberadamente SOLO en almuerzo
    para no contradecir la regla dura que la propia rúbrica del juez declara — ver `_note` del
    archivo, mismo criterio que T5 corrigió para 'Sopa de fideos' en su §10."""
    data = loader()
    ofensoras = [
        t["name"] for t in data["templates"]
        if t.get("base") in ("arroz", "pasta") and set(t.get("slots", [])) & {"desayuno", "cena"}
    ]
    assert not ofensoras, f"[{cc}] plantillas con base arroz/pasta en desayuno/cena: {ofensoras}"


@pytest.mark.e2e
@pytest.mark.parametrize("cc,loader", [("MX", _load_dish_templates_mx), ("CO", _load_dish_templates_co)])
def test_dish_templates_constituents_resuelven_al_catalogo_vivo(cc, loader):
    """[e2e] Contrato "nombres EXACTOS del catálogo" — cada `constituents[].name` debe ser un
    `name` LITERAL de `master_ingredients` post-altas T6 (no un alias que resuelva vía
    `normalize_name`). Mismo patrón que el equivalente de T5 para ES."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — faltan NEON_DATABASE_URL/.env (e2e, no bloquea el gate)")
    db_core.connection_pool.open()
    from db_core import execute_sql_query

    data = loader()
    nombres_usados = {c["name"] for t in data["templates"] for c in t["constituents"]}
    rows = execute_sql_query("SELECT name FROM master_ingredients", fetch_all=True)
    assert rows, "master_ingredients vino vacío con el pool abierto"
    catalogo = {r["name"] for r in rows if r.get("name")}

    faltantes = sorted(nombres_usados - catalogo)
    assert not faltantes, (
        f"[{cc}] {len(faltantes)} nombre(s) de constituents NO son un `name` exacto de "
        f"master_ingredients: {faltantes}"
    )


# ── I2. _dish_templates_path_for_country + _culinary_judge_rubric_for_country MX/CO ──────────────

def test_dish_templates_path_for_country_mx_co_usan_su_archivo_propio(go):
    assert go._dish_templates_path_for_country("MX") == str(_DISH_TEMPLATES_MX_JSON)
    assert go._dish_templates_path_for_country("CO") == str(_DISH_TEMPLATES_CO_JSON)


def test_culinary_judge_rubric_mx_sustituye_ejemplos_y_encabezado(go):
    rubric_mx = go._culinary_judge_rubric_for_country("MX")
    rubric_do = go._culinary_judge_rubric_for_country("DO")
    assert rubric_mx != rubric_do
    assert "Tacos de pollo al pastor ligero" in rubric_mx
    assert "Mangú" not in rubric_mx, "los ejemplos dominicanos no deben sobrevivir en la variante MX"
    assert "PLATOS DE M" in rubric_mx.upper()
    assert "cocina de México" in rubric_mx


def test_culinary_judge_rubric_co_sustituye_ejemplos_y_encabezado(go):
    rubric_co = go._culinary_judge_rubric_for_country("CO")
    rubric_do = go._culinary_judge_rubric_for_country("DO")
    assert rubric_co != rubric_do
    assert "Ajiaco" in rubric_co
    assert "Mangú" not in rubric_co, "los ejemplos dominicanos no deben sobrevivir en la variante CO"
    assert "PLATOS DE C" in rubric_co.upper()
    assert "cocina de Colombia" in rubric_co


def test_culinary_judge_rubric_mx_co_es_do_no_se_contaminan_entre_si(go):
    """Cada variante de país usa SOLO sus propios ejemplos — un plato MX nunca debe aparecer en
    la rúbrica CO (ni viceversa), y ninguno de los dos debe traer platos ES ni DO. Prueba
    puntual de que la sustitución de bloque (`rendered.replace(_do_block, _country_block)`) no
    deja restos cruzados entre las 4 variantes ya cacheadas en el mismo proceso."""
    rubric_mx = go._culinary_judge_rubric_for_country("MX")
    rubric_co = go._culinary_judge_rubric_for_country("CO")
    rubric_es = go._culinary_judge_rubric_for_country("ES")
    assert "Ajiaco" not in rubric_mx and "Bandeja paisa" not in rubric_mx
    assert "Tacos de pollo" not in rubric_co and "Pozole" not in rubric_co
    assert "Tortilla española" not in rubric_mx and "Tortilla española" not in rubric_co
    assert "Tacos de pollo" not in rubric_es and "Ajiaco" not in rubric_es


def test_culinary_judge_rubric_do_sigue_byte_identico_tras_mx_co(go):
    """Control de no-regresión: dar de alta MX/CO no debe tocar la ruta DO — mismo objeto
    cacheado (identidad, no solo igualdad), verificado DESPUÉS de resolver MX/CO/ES en este
    mismo proceso (el cache es un dict módulo-level; esto confirma que no hay mutación cruzada)."""
    go._culinary_judge_rubric_for_country("MX")
    go._culinary_judge_rubric_for_country("CO")
    assert go._culinary_judge_rubric_for_country("DO") is go._CULINARY_JUDGE_RUBRIC


# ── I3. COUNTRY_POOLS['MX']/['CO'] + _get_fast_filtered_catalogs(country=) ───────────────────────

@pytest.mark.parametrize("cc", ["MX", "CO"])
def test_country_pools_mx_co_estructura(cc):
    pool = constants.COUNTRY_POOLS.get(cc)
    assert isinstance(pool, dict)
    for key in ("proteins", "carbs", "veggies_fats", "fruits"):
        assert isinstance(pool.get(key), list) and pool[key], f"COUNTRY_POOLS[{cc!r}][{key!r}] vacío"
        assert all(isinstance(x, str) and x.strip() for x in pool[key])


@pytest.mark.parametrize("cc", ["MX", "CO"])
def test_get_fast_filtered_catalogs_usa_su_propio_pool(cc):
    proteins_do, _, _, _ = constants._get_fast_filtered_catalogs((), (), "")
    proteins, carbs, veg, fruits = constants._get_fast_filtered_catalogs((), (), "", country=cc)
    assert proteins != proteins_do
    assert set(proteins) == set(constants.COUNTRY_POOLS[cc]["proteins"])
    assert set(carbs) == set(constants.COUNTRY_POOLS[cc]["carbs"])
    assert set(veg) == set(constants.COUNTRY_POOLS[cc]["veggies_fats"])
    assert set(fruits) == set(constants.COUNTRY_POOLS[cc]["fruits"])


def test_get_fast_filtered_catalogs_sin_country_mx_co_sigue_siendo_do_byte_identico():
    """Dar de alta MX/CO no debe tocar el fallback — `country=None`/'DO'/país sin pool siguen
    devolviendo EXACTAMENTE `DOMINICAN_*` (byte-idéntico, mismo test que H8 para ES)."""
    casos = [((), (), ""), (("mariscos",), (), ""), ((), (), "vegano")]
    for allergies, dislikes, diet in casos:
        base = constants._get_fast_filtered_catalogs(allergies, dislikes, diet)
        con_none = constants._get_fast_filtered_catalogs(allergies, dislikes, diet, country=None)
        con_pr = constants._get_fast_filtered_catalogs(allergies, dislikes, diet, country="PR")
        assert base == con_none == con_pr, f"diverge para {(allergies, dislikes, diet)!r}"


# ── I4. unpriced-keep: las 46 altas T6 reconocidas por su propio token ───────────────────────────
# (el sweep e2e que prueba "nada MÁS se reconoce por accidente" ya se extendió arriba, I0 —
# test_is_country_catalog_unpriced_item_no_colisiona_con_ningun_nombre_del_catalogo_vivo_ni_pools)

@pytest.mark.parametrize("nombre", sorted(_DISH_TEMPLATES_MX_CO_NAMES))
def test_is_country_catalog_unpriced_item_reconoce_cada_alta_t6(sc, nombre):
    assert sc.is_country_catalog_unpriced_item(nombre), f"{nombre!r} no reconocido como unpriced keep"


def test_chile_serrano_no_colisiona_con_jamon_serrano_en_unpriced_keep(sc):
    """[el mismo tipo de colisión que el fix-round 1 de T5, encontrada y evitada EN ESTA task
    antes de commitear] El token bare 'serrano' habría colisionado con los aliases de 'Jamón
    serrano' (T5: 'jamon serrano'/'serrano ham') — se eligió el token de 2 palabras 'chile
    serrano' a propósito. Este test ancla que AMBOS resuelven a su propio alimento sin pisarse."""
    assert sc.is_country_catalog_unpriced_item("Chile serrano")
    assert sc.is_country_catalog_unpriced_item("Jamón serrano")
    # 'Jamón serrano' NO debe matchear por el token 'chile serrano' (no lo contiene) — y
    # viceversa, confirmando que son dos reconocimientos INDEPENDIENTES, no uno colándose en otro.
    assert "chile serrano" not in constants.strip_accents("Jamón serrano").lower()
    assert "jamon serrano" not in constants.strip_accents("Chile serrano").lower()


# ── I5. Los 4 vocabularios — anchors narrow de las altas T6 ───────────────────────────────────────

@pytest.mark.parametrize("clase,termino", [
    ("pescado", "trucha"), ("pescado", "truchas"),
    ("lacteos", "arequipe"), ("lacteos", "suero costeno"),
    ("lactosa", "arequipe"), ("lactosa", "suero costeno"),
])
def test_altas_t6_presentes_en_allergen_synonyms(go, clase, termino):
    assert termino in go._ALLERGEN_SYNONYMS[clase], f"{termino!r} ausente de _ALLERGEN_SYNONYMS[{clase!r}]"


@pytest.mark.parametrize("termino", ["trucha", "truchas"])
def test_altas_t6_presentes_en_diet_seafood_terms(go, termino):
    assert termino in go._DIET_SEAFOOD_TERMS


@pytest.mark.parametrize("termino", ["arequipe", "suero costeno"])
def test_altas_t6_presentes_en_diet_dairy_terms(go, termino):
    assert termino in go._DIET_DAIRY_TERMS


def test_cecina_presente_en_diet_flesh_terms(go):
    """'chicharron'/'gallina'/'chorizo' YA vivían en `_DIET_FLESH_TERMS` (cubren Chicharrón-CO/
    Gallina criolla-CO/Chorizo mexicano-verde-santarrosano-MX-CO por substring) — 'cecina' es la
    ÚNICA alta T6 de esta clase sin ningún término existente que la matchee."""
    assert "cecina" in go._DIET_FLESH_TERMS


@pytest.mark.parametrize("termino,catalogo_esperado", [
    ("gallina", "Gallina criolla"), ("chicharron", "Chicharrón"),
    ("chorizo", "Chorizo mexicano"), ("chorizo", "Chorizo verde"), ("chorizo", "Chorizo santarrosano"),
])
def test_terminos_preexistentes_ya_cubren_las_altas_t6_sin_cambios(go, termino, catalogo_esperado):
    """[control negativo, honestidad de la task] Estas altas NO necesitaron ningún término nuevo
    — ya vivían cubiertas por substring. Documenta explícitamente qué NO se tocó, para que un
    futuro lector no asuma que toda alta requirió una entrada nueva en el vocabulario."""
    assert termino in go._DIET_FLESH_TERMS
    assert _term_matches(termino, catalogo_esperado)


@pytest.mark.parametrize("ingrediente", ["Trucha a la plancha", "200 g de truchas"])
def test_scan_allergen_violations_detecta_trucha_como_pescado(go, ingrediente):
    plan = {"days": [{"meals": [{"name": "Cena", "ingredients": [ingrediente]}]}]}
    v = go._scan_allergen_violations(plan, ["pescado"])
    assert v, f"{ingrediente!r} debe violar la alergia a pescado"


@pytest.mark.parametrize("ingrediente", ["Arequipe con queso", "40 g de suero costeño"])
def test_scan_allergen_violations_detecta_arequipe_suero_costeno_como_lacteos(go, ingrediente):
    plan = {"days": [{"meals": [{"name": "Merienda", "ingredients": [ingrediente]}]}]}
    v = go._scan_allergen_violations(plan, ["lacteos"])
    assert v, f"{ingrediente!r} debe violar la alergia a lácteos"


def test_scan_diet_violations_detecta_cecina_para_vegano(go):
    plan = {"days": [{"meals": [{"name": "Desayuno", "ingredients": ["Cecina con huevo"]}]}]}
    assert go._scan_diet_violations(plan, "vegano"), "'Cecina con huevo' no fue detectado para vegano"


# ── I6. Regla FILA-vs-SINÓNIMO — el contrato central de esta task ────────────────────────────────

def test_global_reverse_map_no_se_toco_en_esta_task():
    """[requerido por el contrato de la task] `GLOBAL_REVERSE_MAP` (PROTEIN/CARB/VEGGIE_FAT/
    FRUIT_SYNONYMS) es el vocabulario de `normalize_ingredient_for_tracking` (sistema de fatiga/
    variedad) — CERO call sites en `shopping_calculator.py` (verificado por grep), así que NO es
    el resolver que `classify_food`/el harness usan. Pin de que esta task no lo tocó: los 10
    sinónimos de fila-vs-sinónimo (I6 abajo) viven en `aliases` de `master_ingredients`, no aquí.
    Si este test se pone rojo, algo escribió en GLOBAL_REVERSE_MAP — hay que decidir a propósito
    si eso es correcto (rompería el criterio de esta task: el resolver real es normalize_name)."""
    src = _SHOPPING_CALCULATOR_PY.read_text(encoding="utf-8")
    assert "GLOBAL_REVERSE_MAP" not in src, (
        "shopping_calculator.py referencia GLOBAL_REVERSE_MAP — el resolver real de "
        "normalize_name debe seguir siendo independiente de ese vocabulario"
    )
    # Pins puntuales (invariantes desde ANTES de esta task, verificados en vivo): si CAMBIAN,
    # algo tocó PROTEIN/CARB/VEGGIE_FAT/FRUIT_SYNONYMS de forma que afecta resoluciones RD
    # existentes. 'papaya' YA era variante de 'lechosa' en FRUIT_SYNONYMS ANTES de esta task
    # (T6 no tocó ese dict) — el homógrafo MX 'Papaya' (RESUELVE-BIEN vía normalize_name, no vía
    # esta tabla) coexiste sin conflicto porque son DOS sistemas independientes con el mismo
    # resultado por coincidencia, no por compartir mecanismo.
    assert constants.GLOBAL_REVERSE_MAP.get("banana") == "guineo"
    assert constants.GLOBAL_REVERSE_MAP.get("pechuga") == "pollo"
    assert constants.GLOBAL_REVERSE_MAP.get("papaya") == "lechosa"


@pytest.mark.e2e
@pytest.mark.parametrize("query,esperado", [
    # Los 6 sinónimos vía script (aliases sobre fila YA EXISTENTE):
    ("Jitomate", "Tomate"), ("Jitomates", "Tomate"),
    ("Mazorca", "Maíz dulce en granos"), ("Choclo", "Maíz dulce en granos"),
    ("Malanga", "Yautía"), ("Cuchuco de trigo", "Bulgur"),
    # Los 3 sinónimos embebidos en aliases de una fila NUEVA del mismo lote:
    ("Chile cuaresmeño", "Chile jalapeño"), ("Piloncillo", "Panela"), ("Color (bijol)", "Achiote"),
    ("Chile morrón", "Ají morrón"),
])
def test_sinonimos_t6_resuelven_via_normalize_name_real(sc, query, esperado):
    """[el guard central del contrato] Verifica el CAMINO REAL (`sc.normalize_name`, lo que
    `classify_food`/producción llaman) — no una tabla muerta. Marcado e2e porque
    `normalize_name` lee `get_master_ingredients()` (catálogo vivo)."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    assert sc.normalize_name(query) == esperado, (
        f"normalize_name({query!r}) debe resolver a {esperado!r} (sinónimo fila-vs-sinónimo T6)"
    )


@pytest.mark.e2e
@pytest.mark.parametrize("query,esperado", [
    ("Jamón serrano", "Jamón serrano"), ("mora azul", "Arándanos"), ("Guayaba", "Guayaba"),
    ("Piña", "Piña"), ("Higo", "Higo"), ("Leche", "Leche"), ("Carne de res", "Carne de res"),
    ("Cerdo", "Cerdo"), ("Apio", "Apio"), ("Zanahoria", "Zanahoria"),
    ("Elote", "Maíz dulce en granos"), ("Chinola", "Chinola"), ("Atún en agua", "Atún en agua"),
    ("Yautía", "Yautía"), ("Bulgur", "Bulgur"), ("Ají morrón", "Ají morrón"), ("Tomate", "Tomate"),
])
def test_pin_resoluciones_rd_es_no_cambiaron_tras_las_altas_t6(sc, query, esperado):
    """[el pin explícito que pide el contrato: "los aciertos RD actuales no cambian"] Cada una de
    estas resoluciones PRE-EXISTE a T6 (RD nativo o alta T5/ES) y comparte un token/substring con
    alguna alta o alias nuevo de T6 (Jamón serrano↔Chile serrano, mora azul↔Mora, Guayaba/Piña↔
    'guayaba piña' descartado de Feijoa, Higo↔'higo chumbo' descartado de Tuna de nopal, Leche↔
    'dulce de leche' de Arequipe, Carne de res/Cerdo↔aliases de Cecina/Chicharrón, Apio/Zanahoria↔
    aliases de Arracacha, Elote↔Mazorca/Choclo recién añadidos a la MISMA fila, Chinola↔Curuba/
    Granadilla compartiendo fdc, Atún en agua↔homógrafo 'tuna' de Tuna de nopal, Yautía↔alias
    Malanga, Bulgur↔alias Cuchuco de trigo, Ají morrón/Tomate↔alias Chile morrón/Jitomate). RED si
    CUALQUIERA de estas 17 cambia de resultado."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    assert sc.normalize_name(query) == esperado, (
        f"REGRESIÓN: normalize_name({query!r}) debía seguir resolviendo a {esperado!r} — una "
        f"alta/alias de T6 rompió un acierto RD/ES pre-existente"
    )


@pytest.mark.e2e
def test_homografo_tortilla_de_maiz_resuelve_a_su_propia_fila(sc):
    """[homógrafo con más riesgo del task, citado explícito en el brief] 'tortilla' nombra TRES
    alimentos distintos en el sistema: Tortilla de trigo/integral (RD, harina de trigo), Tortilla
    española (ES/T5, huevo+patata) y Tortilla de maíz (MX/T6, maíz sin gluten) — ninguna alias
    comparte token con otra."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    assert sc.normalize_name("Tortillas de maíz") == "Tortilla de maíz"
    assert sc.normalize_name("tortilla de maiz picada") == "Tortilla de maíz"
    assert sc.normalize_name("Tortilla de trigo") == "Tortilla de trigo"
    assert sc.normalize_name("Tortilla integral") == "Tortilla integral"


def test_resolve_preparation_distinct_tortilla_maiz_canoniza_a_fila_real(sc):
    """[la MUTACIÓN de esta task, ver reporte] Pre-T6, `resolve_preparation_distinct` forzaba
    `(True, None)` (pass-through, DROP) para 'tortilla de maíz' porque el catálogo solo tenía
    tortillas de TRIGO — comentario original: "el catálogo solo tiene tortillas de TRIGO". Con
    la alta real T6 debe CANONIZAR a la fila, no pasar de largo. RED-first: revertir la línea a
    `return (True, None)` reproduce el DROP (ver reporte §Mutaciones)."""
    handled, canonical = sc.resolve_preparation_distinct("Tortillas de maíz")
    assert handled is True
    assert canonical == "Tortilla de maíz", (
        f"resolve_preparation_distinct debe canonizar a 'Tortilla de maíz', devolvió {canonical!r} "
        f"— si es None, la mutación pre-T6 (pass-through) revivió"
    )


@pytest.mark.e2e
def test_homografo_tuna_de_nopal_no_contamina_atun_en_agua(sc):
    """['tuna' en México/RD = fruta del nopal; 'atún' = el pescado — palabras DISTINTAS que un
    hispanohablante no confundiría, pero el alias bare 'tuna' (necesario porque normalize_name
    stripea paréntesis ANTES de comparar: "Tuna (fruta de nopal)" -> "tuna") podría en teoría
    colisionar si 'Atún en agua' alguna vez ganara un alias en inglés. Pin de que NO lo tiene."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    assert sc.normalize_name("Tuna (fruta de nopal)") == "Tuna de nopal"
    assert sc.normalize_name("Tuna") == "Tuna de nopal"
    assert sc.normalize_name("Atún en agua") == "Atún en agua"
    assert sc.normalize_name("atún") == "Atún en agua"


@pytest.mark.e2e
def test_homografo_mora_no_contamina_mora_azul_de_arandanos(sc):
    """['mora' sola en es-LatAm = mora/zarzamora (blackberry); 'mora azul' (alias PRE-EXISTENTE
    de 'Arándanos', blueberry) es un término DISTINTO con calificativo — jamás confundidos porque
    los aliases de 'Mora' nunca incluyen 'azul'. Fibra 5.3g (Mora) vs 2.4g (Arándanos) confirma
    que además son alimentos macro-distintos, no solo nombres distintos (regla >15% del contrato)."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    assert sc.normalize_name("Mora") == "Mora"
    assert sc.normalize_name("mora azul") == "Arándanos"
    assert sc.normalize_name("Arándanos") == "Arándanos"


@pytest.mark.parametrize("item,fila_o_alias_esperado,tipo", [
    ("Jitomate", "Tomate", "sinónimo"), ("Chile morrón", "Ají morrón", "sinónimo"),
    ("Mazorca", "Maíz dulce en granos", "sinónimo"), ("Choclo", "Maíz dulce en granos", "sinónimo"),
    ("Malanga", "Yautía", "sinónimo"), ("Cuchuco de trigo", "Bulgur", "sinónimo"),
    ("Chile cuaresmeño", "Chile jalapeño", "sinónimo"), ("Piloncillo", "Panela", "sinónimo"),
    ("Color (bijol)", "Achiote", "sinónimo"),
    ("Frijol cargamanto", "Frijol cargamanto", "fila nueva"), ("Chicharrón", "Chicharrón", "fila nueva"),
    ("Trucha", "Trucha", "fila nueva"), ("Tortilla de maíz", "Tortilla de maíz", "fila nueva"),
])
def test_tabla_fila_vs_sinonimo_estructural(item, fila_o_alias_esperado, tipo):
    """[tabla estructural, sin DB] Documenta en código la decisión fila-vs-sinónimo por item —
    complementa (no reemplaza) el test funcional `test_sinonimos_t6_resuelven_via_normalize_name_real`
    de arriba: aquí se ancla la DECISIÓN (qué se decidió), allá el RESULTADO (que funciona)."""
    if tipo == "sinónimo":
        assert item != fila_o_alias_esperado, f"{item!r} está marcado sinónimo pero apunta a sí mismo"
        assert fila_o_alias_esperado in _T6_NUEVAS_FILAS_O_PREEXISTENTES(), (
            f"el destino {fila_o_alias_esperado!r} de {item!r} debe ser una fila real conocida"
        )
    else:
        assert item == fila_o_alias_esperado, f"{item!r} está marcado fila nueva pero mapea a otro nombre"
        assert item in _DISH_TEMPLATES_MX_CO_NAMES, f"{item!r} marcado fila nueva debe estar en las 46 altas"


def _T6_NUEVAS_FILAS_O_PREEXISTENTES() -> frozenset:
    return _DISH_TEMPLATES_MX_CO_NAMES | frozenset({
        "Tomate", "Ají morrón", "Maíz dulce en granos", "Yautía", "Bulgur",
    })


# ── I7. e2e — las 46 altas T6 existen en el catálogo vivo, SIN precio, con fdc_id o 'manual' ─────

@pytest.mark.e2e
def test_46_altas_t6_existen_en_catalogo_vivo_sin_precio_con_fdc_id_o_manual():
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    from db_core import execute_sql_query

    rows = execute_sql_query(
        "SELECT name, price_per_lb, price_per_unit, fdc_id, nutrition_source "
        "FROM master_ingredients WHERE name = ANY(%s)",
        (list(_DISH_TEMPLATES_MX_CO_NAMES),),
        fetch_all=True,
    ) or []
    por_nombre = {r["name"]: r for r in rows}

    faltantes = sorted(_DISH_TEMPLATES_MX_CO_NAMES - set(por_nombre))
    assert not faltantes, f"altas T6 ausentes del catálogo vivo: {faltantes}"

    con_precio = [n for n, r in por_nombre.items()
                  if float(r["price_per_lb"] or 0) > 0 or float(r["price_per_unit"] or 0) > 0]
    assert not con_precio, f"altas T6 con precio RD (deberían estar en 0): {con_precio}"

    # [T6 · a diferencia de T5] 3 filas SIN fdc_id real (nutrition_source='manual' en su lugar) —
    # ver docstring de add_foods_mx_co_2026_08_17.py. Las otras 43 SÍ exigen fdc_id + 'usda'.
    _MANUAL = {"Achiote", "Flor de Jamaica", "Hoja santa"}
    con_usda = {n: r for n, r in por_nombre.items() if n not in _MANUAL}
    sin_fdc = [n for n, r in con_usda.items() if not r.get("fdc_id")]
    assert not sin_fdc, f"altas T6 (no-manual) sin fdc_id: {sin_fdc}"
    no_usda = [n for n, r in con_usda.items() if r.get("nutrition_source") != "usda"]
    assert not no_usda, f"altas T6 (no-manual) con nutrition_source != 'usda': {no_usda}"

    manuales = {n: r for n, r in por_nombre.items() if n in _MANUAL}
    assert set(manuales) == _MANUAL, f"esperaba exactamente {_MANUAL} como manual, hay {set(manuales)}"
    no_manual = [n for n, r in manuales.items() if r.get("nutrition_source") != "manual"]
    assert not no_manual, f"{no_manual} deberían tener nutrition_source='manual'"


# ── I8. Golden fixture: un día MX y un día CO pasan slots suaves (mismo patrón que H14 para ES) ──

def _dia_mx_con_mole_en_desayuno() -> list:
    return [{
        "day": 1,
        "meals": [
            {"meal": "Desayuno", "name": "Mole ligero de pollo con arroz",
             "ingredients": ["Pechuga de pollo", "Chocolate de mesa", "Arroz blanco"]},
            {"meal": "Almuerzo", "name": "Tacos de pollo al pastor ligero",
             "ingredients": ["Pechuga de pollo", "Tortilla de maíz", "Piña"]},
            {"meal": "Cena", "name": "Enchiladas de queso al horno",
             "ingredients": ["Tortilla de maíz", "Queso blanco", "Chile guajillo"]},
            {"meal": "Merienda", "name": "Jícama con chile y limón",
             "ingredients": ["Jícama", "Chile de árbol", "Limón"]},
        ],
    }]


def _dia_co_con_arroz_en_desayuno() -> list:
    return [{
        "day": 1,
        "meals": [
            {"meal": "Desayuno", "name": "Arroz con pollo colombiano",
             "ingredients": ["Pechuga de pollo", "Arroz blanco", "Ají cubanela"]},
            {"meal": "Almuerzo", "name": "Ajiaco santafereño con pollo",
             "ingredients": ["Pechuga de pollo", "Papa", "Guascas"]},
            {"meal": "Cena", "name": "Trucha al horno con limón",
             "ingredients": ["Trucha", "Limón", "Ajo"]},
            {"meal": "Merienda", "name": "Jugo natural de lulo",
             "ingredients": ["Lulo"]},
        ],
    }]


@pytest.mark.parametrize("cc,builder", [("MX", _dia_mx_con_mole_en_desayuno), ("CO", _dia_co_con_arroz_en_desayuno)])
def test_dia_arroz_fuera_de_horario_pasa_como_soft_sin_forzar_retry(go, monkeypatch, cc, builder):
    """[Golden fixture] Un día con un plato base-arroz en Desayuno — violación real de la regla
    dura — DEBE detectarse SOFT (hard=False) para país MX/CO: `slot_rules_for_country` softea
    toda regla en Fase 1 (T4), igual que ES en H14."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    dias = builder()
    violaciones = go._detect_slot_appropriateness(dias, {"country": cc})
    assert violaciones, f"[{cc}] el día de prueba debe producir AL MENOS una violación (control positivo)"
    duras = [v for v in violaciones if v["hard"]]
    assert not duras, f"[{cc}] no debe producir violaciones HARD: {duras}"


@pytest.mark.parametrize("cc,builder", [("MX", _dia_mx_con_mole_en_desayuno), ("CO", _dia_co_con_arroz_en_desayuno)])
def test_el_mismo_dia_mx_co_hard_para_do_control_de_que_el_mecanismo_discrimina(go, monkeypatch, cc, builder):
    """Control negativo (mismo patrón que H14): el MISMO día, con country='DO', debe seguir
    produciendo HARD — confirma que el mecanismo distingue países en vez de estar simplemente roto."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    dias = builder()
    violaciones = go._detect_slot_appropriateness(dias, {"country": "DO"})
    assert violaciones, f"[{cc}→DO] control: el día debe violar también en DO"
    duras = [v for v in violaciones if v["hard"]]
    assert duras, f"[{cc}→DO] debe seguir produciendo violaciones HARD (byte-identidad del mecanismo)"


# ── I9. Cierre medible — mx.json/co.json committed: cero DROP, cero SUSTITUCION-SILENCIOSA ───────

@pytest.mark.parametrize("cc,fname", [("MX", "mx.json"), ("CO", "co.json")])
def test_harness_mx_co_cierra_en_cero_drops_cero_silenciosas(cc, fname):
    """[el criterio de cierre del contrato] `mx.json`/`co.json` (sobrescritos por la re-corrida
    final del harness post-altas, committed en el repo) deben reportar counts.DROP == 0 y
    counts.SUSTITUCION-SILENCIOSA == 0 — el mismo criterio de salida que T5 cerró para ES."""
    path = _BACKEND / "data" / "country_gaps" / fname
    assert path.exists(), f"[{cc}] {fname} debe existir committed en el repo"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("country") == cc
    counts = data.get("counts") or {}
    assert counts.get("DROP") == 0, f"[{cc}] DROP debe ser 0 en el cierre, es {counts.get('DROP')}"
    assert counts.get("SUSTITUCION-SILENCIOSA") == 0, (
        f"[{cc}] SUSTITUCION-SILENCIOSA debe ser 0 en el cierre, es {counts.get('SUSTITUCION-SILENCIOSA')}"
    )
    assert counts.get("RESUELVE-BIEN") == data.get("total_items"), (
        f"[{cc}] RESUELVE-BIEN debe cubrir el 100% de total_items en el cierre"
    )
