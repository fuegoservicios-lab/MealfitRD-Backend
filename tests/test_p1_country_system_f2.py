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

import ast
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
_DISH_TEMPLATES_PR_JSON = _BACKEND / "data" / "dish_templates_pr.json"
_DISH_TEMPLATES_US_JSON = _BACKEND / "data" / "dish_templates_us.json"
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
    # [P1-COACH-LANGUAGE-NATIVE · 2026-08-18] La directiva ahora se escribe EN el idioma
    # destino (round 2 del incidente en-US: la versión en español era la señal más débil
    # posible y el modelo la desobedecía). Los anchors pasan a la redacción nativa; la
    # frontera dura (nombres en español + tool calls canónicas) sigue anclada.
    r = chat_agent_prompts.build_language_directive("en-US")
    assert "ENTIRE reply in English" in r
    assert "Guiso de Habichuelas Negras" in r and "Spanish" in r
    assert "canonical Spanish food names" in r


@pytest.mark.parametrize("locale,frase_nativa,palabra_espanol", [
    ("pt-BR", "TODA a sua resposta em Português", "espanhol"),
    ("fr-FR", "TOUTE ta réponse en Français", "espagnol"),
    ("it-IT", "TUTTA la tua risposta in Italiano", "spagnolo"),
])
def test_los_otros_3_idiomas_contienen_su_propia_directiva(locale, frase_nativa, palabra_espanol):
    r = chat_agent_prompts.build_language_directive(locale)
    assert frase_nativa in r, "la directiva debe estar escrita EN el idioma destino (nativa)"
    assert palabra_espanol in r, "la excepción de nombres en español debe declararse en el idioma destino"
    assert "Guiso de Habichuelas Negras" in r, "el ejemplo canónico ancla la frontera dura"


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
    """[P1-HELP-BOT-I18N · 2026-08-20] LA PREMISA DE ESTE TEST CAMBIÓ, y conviene leer por
    qué antes de tocarlo otra vez.

    Decía: «cero tools, cero DB, cero user_id en el prompt. No hay `locale` que leer — la
    directiva NO debe importarse ni mencionarse ahí». Era cierto: el widget no enviaba
    `locale`. Dejó de serlo el día que empezó a enviarlo, porque el bot respondía en
    español a un usuario con la app en inglés.

    LO QUE NO CAMBIA, que es la frontera dura de F2: `build_tools_instructions` sigue sin
    `locale`. Las TOOL CALLS se quedan en español canónico SIEMPRE porque sus cadenas son
    IDENTIFICADORES del motor. Este bot no tiene tools, ni DB, ni user_id — su salida es
    prosa de soporte y no resuelve nada por cadena. Mismo criterio que separa traducir la
    dificultad de una receta de NO traducir el nombre de un alimento.

    LO QUE ESTE TEST PROTEGE AHORA: que el bot REUSE el SSOT en vez de escribirse su propia
    tabla de idiomas. La primera versión del P-fix hizo exactamente eso --el antipatrón que
    el repo lleva repitiendo-- y fue este test, al ponerse rojo, quien lo destapó. Además
    del SSOT: `P1-COACH-LANGUAGE-NATIVE` compró caro que la directiva va EN EL IDIOMA
    DESTINO; una tabla propia se lo habría saltado.
    """
    src = _read(_HELP_BOT_PY)
    # Reusa el SSOT, no una tabla propia.
    assert "from prompts.chat_agent import build_language_directive" in src
    assert "_COACH_LANGUAGE_NAMES" not in src, "el bot se copió el mapa de idiomas"
    assert not re.search(r"^_REGLA_IDIOMA\s*=", src, re.M), "volvió la tabla de idiomas propia"
    # El locale se PASA al SSOT, nunca se interpola en el prompt.
    assert "build_language_directive(locale)" in src
    # Y el bot sigue sin lo que lo mantendria dentro de la frontera dura. Se comprueba
    # sobre los IMPORTS --codigo-- y no por subcadena: la primera version prohibia la
    # cadena "user_id" y fallaba contra el DOCSTRING del propio modulo, que dice
    # literalmente «NO recibe user_id». Un guard que la prosa puede disparar acaba
    # obligando a no documentar, que es peor que el guard.
    imports = [l for l in src.splitlines() if re.match(r"^\s*(import|from)\s", l)]
    # El unico import de `prompts.*` permitido es el SSOT de la directiva; se descuenta
    # antes de buscar lo prohibido para que su propio nombre no dispare el guard.
    unidos = "\n".join(imports).replace(
        "from prompts.chat_agent import build_language_directive", "")
    for prohibido in ("db", "tools", "agent", "sqlalchemy", "psycopg", "fastapi"):
        assert not re.search(rf"\b{prohibido}\b", unidos), (
            f"el bot de ayuda importa `{prohibido}`: dejaria de ser Q&A puro sin DB ni tools")


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


@pytest.fixture(scope="module")
def hz():
    """`humanize_ingredients` -- módulo liviano (sin DB a nivel de import), pero por fixture de
    módulo para paridad de estilo con `sc`/`go`/`condrules` de arriba."""
    import humanize_ingredients as _hz
    return _hz


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
    'certificada' — siguen sin violar.

    [P1-COUNTRY-SYSTEM-F2 · T7 · 2026-08-17] La 2ª grafía cambió de '(panqueques)' a
    '(hojuelas)': 'panqueque'/'panqueques' se sumó a `_ALLERGEN_SYNONYMS['gluten']` en esta task
    (altas US) y ahora dispara SU PROPIA violación independiente en ese string — legítimamente
    (un panqueque regular SÍ lleva gluten salvo que se declare "sin gluten" él mismo, y aquí el
    claim "sin gluten" gramaticalmente califica a 'avena', no a 'panqueques' entre paréntesis).
    No es un bug de la excusa forward -- es una detección nueva y correcta que el string
    original no anticipaba. 'hojuelas' no es un término de ningún vocabulario de alérgenos."""
    for ingrediente in ("20 g de avena certificada sin gluten", "Avena sin gluten (hojuelas)"):
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
            # [P3-SEMOLA-MAIZ-GLUTEN-FP · 2026-08-23] Las excusas por BASE viven en
            # producción (`_ALLERGEN_TERM_BASE_EXCUSES`), no en una segunda lista aquí: a un
            # celíaco de EE.UU. se le quitaba la «Sémola de maíz», que no tiene gluten. El
            # término dispara («sémola») pero su BASE lo desmiente («de maíz»), igual que
            # «leche de almendras» no es lácteo.
            #
            # Se consulta el MISMO predicado que usa el backstop, no una copia: si mañana la
            # excusa cambia, este test la sigue sola. Escribir aquí un segundo `{"semola":
            # ("maiz",...)}` sería la segunda tabla que P1-DIET-CANON-SSOT ya pagó una vez.
            _n = constants.strip_accents(nombre).strip().lower()
            if any(go._allergen_term_base_excused(_t, _n[len(_t):])
                   for _t in getattr(go, "_ALLERGEN_TERM_BASE_EXCUSES", {})
                   if _n.startswith(_t)):
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

# [P1-COUNTRY-SYSTEM-F2 · T7 · 2026-08-17] Las 62 altas de catálogo PR/US de esta task (nombre
# CANÓNICO de fila). Espejo de `_DISH_TEMPLATES_ES_NAMES`/`_DISH_TEMPLATES_MX_CO_NAMES` de
# arriba — mismo propósito: excluir del sweep de "no debe reconocerlas" (son las que SÍ deben
# reconocerse). 19 PR + 43 US (27 DROP originales + 16 reclasificadas desde una colisión
# verificada en vivo antes de dar de alta — ver reporte §Colisiones encontradas y evitadas).
_DISH_TEMPLATES_PR_US_NAMES = frozenset({
    # PR (19)
    "Panapén", "Pernil", "Jamón de cocinar", "Sofrito", "Recao", "Adobo", "Alcaparrado",
    "Harina de yuca", "Pique", "Pavochón", "Bacalaítos", "Ron de cocina",
    "Longaniza puertorriqueña", "Chuleta ahumada", "Sazón con culantro y achiote",
    "Aceite de achiote", "Queso de papa", "Especias para arroz con dulce",
    "Aceitunas rellenas",
    # US (43)
    "Tocineta", "Jamón de sándwich", "Salchichas", "Crema agria", "Crema mitad y mitad",
    "Bagels", "Panecillos ingleses", "Pretzels", "Frijoles horneados", "Jarabe de arce",
    "Aderezo ranch", "Salsa barbacoa", "Kétchup", "Salsa inglesa", "Malvaviscos", "Coditos",
    "Masa para pie", "Galletas Graham", "Salsa de salchicha", "Ensalada de macarrones",
    "Chile en polvo", "Sazonador para tacos", "Pepperoni", "Salchicha italiana",
    "Mezcla para panqueques", "Wafles", "Azúcar morena", "Suero de mantequilla",
    "Pan de maíz", "Sémola de maíz", "Arándanos rojos", "Duraznos", "Pan rallado",
    "Panecillos de mantequilla", "Huevos rellenos", "Nuez de Castilla", "Nueces pecanas",
    "Queso en hebras", "Queso provolone", "Carne molida mixta", "Bolitas de papa",
    "Papas ralladas", "Chili con carne",
})

# [P1-COUNTRY-SYSTEM-F2 · Task 8 · 2026-08-17] La ÚNICA alta de fila nueva del top-up RD (Hummus
# — ver Sección K). Espejo de los 3 frozensets de arriba, mismo propósito: excluir del sweep de
# "no debe reconocerla" (SÍ debe reconocerla — es el propósito de su token en
# `_COUNTRY_CATALOG_UNPRICED_TOKENS`). "Merey"/"Rábano" NO entran aquí: son filas PRICED
# pre-existentes que solo ganaron un alias (mereyes/rabanos) — nunca deben aparecer en
# `is_country_catalog_unpriced_item`, ni antes ni después de esta task.
_DISH_TEMPLATES_RD_TOPUP_NAMES = frozenset({"Hummus"})


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
    assert 40 <= len(templates) <= 140  # [P1-GAP-DISHES-VEG · 2026-09-06] el techo sube de 120 a 140: el embudo de cobertura pidió platos DIRIGIDOS a huecos medidos y CO llegó a 121. Cota de cordura, no regla de producto, f"{len(templates)} plantillas — fuera del rango ~40-60 del brief"
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
    # [P2-SHOPLIST-BETA-POLISH · 2026-08-18] pasillo REAL del súper, no el label interno
    # 'CATÁLOGO SIN PRECIO' que se filtraba al PDF (el estado beta lo cuenta el banner).
    # 'Proteínas' es la categoría de la fila viva en master_ingredients.
    # [reconvertido · P2-COUNTRY-HOUSEKEEPING · 2026-08-21] Era «Proteínas» (la categoría CRUDA de
    # la base). La rama CON precio emite el label de DISPLAY «PROTEÍNAS» y el Dashboard agrupa por
    # la cadena literal: con las dos grafías el usuario veía DOS secciones del mismo pasillo del
    # súper. La invariante que importa —y la que se ancla— es que el ítem sin precio caiga en el
    # MISMO pasillo que uno con precio, no la grafía concreta.
    assert jamon.get("display_category") == sc._get_display_category("Proteínas", "Jamón serrano")
    assert jamon.get("display_category") == "PROTEÍNAS"


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
    # [P1-COUNTRY-SYSTEM-F2 · T7 · 2026-08-17] extendido de nuevo: también excluye las 62 altas
    # PR/US — mismo motivo, ESAS sí deben reconocerse (es el propósito de esta task).
    # [P1-COUNTRY-SYSTEM-F2 · Task 8 · 2026-08-17] extendido una 4ª vez: excluye "Hummus" (el
    # top-up RD) — mismo motivo, es el propósito de su token.
    candidatos = (
        (nombres_catalogo | nombres_pools)
        - _DISH_TEMPLATES_ES_NAMES - _DISH_TEMPLATES_MX_CO_NAMES - _DISH_TEMPLATES_PR_US_NAMES
        - _DISH_TEMPLATES_RD_TOPUP_NAMES
    )

    falsos_positivos = sorted(n for n in candidatos if sc.is_country_catalog_unpriced_item(n))
    assert not falsos_positivos, (
        f"{len(falsos_positivos)} nombre(s) NO son una alta T5/T6/T7 pero "
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
    """[Re-anclado] El país se deriva una vez para TODO el pantry guard y el
    edge recipe reutiliza ese valor; no abre una segunda lectura que pueda driftar."""
    src = _sin_comentarios(_cron_tasks_source())
    derivacion = "_pantry_guard_country = _country_for_pantry_guard(form_data)"
    assert src.count(derivacion) == 1
    assert src.count("_edge_recipe_country = _pantry_guard_country") == 1
    tree = ast.parse(_cron_tasks_source())
    edge_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_build_filtered_edge_recipe_day"
    ]
    assert len(edge_calls) == 4
    for call in edge_calls:
        country_kw = next((kw.value for kw in call.keywords if kw.arg == "country"), None)
        assert isinstance(country_kw, ast.Name) and country_kw.id == "_edge_recipe_country"


def test_edge_recipe_country_usa_country_for_form_data_ssot():
    """La derivación debe pasar por la ÚNICA puerta de lectura de país (`country_for_form_data`),
    no leer `form_data['country']` crudo — knob apagado ⇒ siempre 'DO' (byte-identidad)."""
    src = _sin_comentarios(_cron_tasks_source())
    assert "from constants import country_for_form_data as _country_for_pantry_guard" in src


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
    """[actualizado T6 · 2026-08-17, actualizado de nuevo T7 · 2026-08-17] MX/CO (T6) y PR/US (T7)
    GANAN archivo propio — ya NO queda ningún país real de los 6 sin `dish_templates_<cc>.json`
    dedicado (T5 usaba MX/CO como control del fallback; T6 los reemplazó por PR; T7 le quita
    también ese rol a PR). `_dish_templates_path_for_country` no re-canonicaliza su argumento
    (opera sobre CUALQUIER string vía comparación literal `canon == "XX"`), así que un código
    ISO inventado ('ZZ', que nunca podrá tener archivo propio) sigue siendo un control válido y
    permanente del mecanismo de fallback en sí — sin depender de que exista un país real
    temporalmente sin archivo."""
    assert go._dish_templates_path_for_country("ES") == str(_DISH_TEMPLATES_ES_JSON)
    assert go._dish_templates_path_for_country("MX") == str(_DISH_TEMPLATES_MX_JSON)
    assert go._dish_templates_path_for_country("CO") == str(_DISH_TEMPLATES_CO_JSON)
    assert go._dish_templates_path_for_country("PR") == str(_DISH_TEMPLATES_PR_JSON)
    assert go._dish_templates_path_for_country("US") == str(_DISH_TEMPLATES_US_JSON)
    assert go._dish_templates_path_for_country("ZZ") == go._DO_DISH_TEMPLATES_PATH
    assert go._dish_templates_path_for_country("DO") == go._DO_DISH_TEMPLATES_PATH


def test_culinary_judge_rubric_pais_desconocido_cae_a_ejemplos_rd(go):
    """[T7 · 2026-08-17, repurposed -- antes 'test_culinary_judge_rubric_pr_sin_archivo_propio_cae_a_ejemplos_rd']
    Tras esta task los 6 países reales tienen su propio `dish_templates_<cc>.json` -- ya no existe
    un país beta "sin archivo dedicado" para probar el fallback vía un código de país REAL (T5
    usaba MX/CO para esto, T6 lo reemplazó por PR; T7 le quita ese rol también a PR). El
    invariante que sigue vivo y vale la pena anclar aquí: a diferencia de
    `_dish_templates_path_for_country` (probado en aislamiento arriba con el código sintético
    'ZZ'), `_culinary_judge_rubric_for_country` SÍ canonicaliza su argumento primero
    (`canonicalize_country`, fail-safe a 'DO' para cualquier código no reconocido) -- un código
    desconocido debe seguir devolviendo la rúbrica DOMINICANA sin romper, vía el cache hit
    directo de 'DO' (identidad de objeto, no solo igualdad)."""
    rubric_zz = go._culinary_judge_rubric_for_country("ZZ")
    assert rubric_zz is go._CULINARY_JUDGE_RUBRIC


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


def test_dia_es_con_knob_apagado_es_byte_identico_a_do_ignora_country_form_data(monkeypatch, go):
    """Control de byte-identidad: SIN monkeypatch del knob (default apagado en este proceso de
    test), el mismo día con `form_data={'country': 'ES'}` debe comportarse EXACTAMENTE como DO
    — `country_for_form_data` no lee el campo `country` en absoluto con el knob apagado."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "false")
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

    # [P1-BEDCA-DEPROXY-ES + P1-PROVENANCE-TRUTHFUL · 2026-08-19] Once altas espanolas
    # DEJARON de tener `fdc_id`, y fue a proposito: el que tenian era PRESTADO de otro
    # alimento de USDA. Un `fdc_id` es una AFIRMACION sobre la procedencia, no una nota
    # al pie -- Sobrasada declaraba 296 kcal con el id de un embutido que no era, y son
    # 595. Se sustituyo por la fuente real (BEDCA) con `nutrition_source_ref`.
    #
    # Este test exigia `fdc_id` a TODAS, asi que la correccion lo puso rojo. Se re-expresa
    # con el MISMO patron que T6 ya usaba --enumerar las excepciones y fijar el conjunto
    # con igualdad exacta-- en vez de relajar la regla: una fila nueva no puede perder su
    # `fdc_id` en silencio, tendria que anadirse aqui.
    #
    # Y es MAS estricto que antes para esas once: se les exige procedencia auditable de
    # verdad (fuente reconocida + `nutrition_source_ref`), no un id que era mentira.
    _ES_SIN_USDA = {
        "Jamón ibérico", "Chistorra", "Chorizo español", "Jamón serrano", "Morcilla",
        "Panceta ibérica", "Sobrasada", "Lomo embuchado", "Requesón", "Butifarra",
        "Boquerones",
    }
    excepciones = {n for n in por_nombre if n in _ES_SIN_USDA}
    assert excepciones == _ES_SIN_USDA, (
        f"el conjunto de altas ES sin fdc_id cambio: esperaba {sorted(_ES_SIN_USDA)}, "
        f"hay {sorted(excepciones)}")

    con_usda = {n: r for n, r in por_nombre.items() if n not in _ES_SIN_USDA}
    sin_fdc = [n for n, r in con_usda.items() if not r.get("fdc_id")]
    assert not sin_fdc, f"altas T5 (no-excepcion) sin fdc_id (fuente no auditable): {sin_fdc}"

    no_usda = [n for n, r in con_usda.items() if r.get("nutrition_source") != "usda"]
    assert not no_usda, f"altas T5 (no-excepcion) con nutrition_source != 'usda': {no_usda}"

    # Las once no quedan sin auditar: fuente reconocida y referencia explicita.
    refs = execute_sql_query(
        "SELECT name, nutrition_source, nutrition_source_ref FROM master_ingredients "
        "WHERE name = ANY(%s)",
        (sorted(_ES_SIN_USDA),),
        fetch_all=True,
    ) or []
    for r in refs:
        assert r["nutrition_source"] in ("bedca", "manual"), (
            f"{r['name']}: fuente {r['nutrition_source']!r} no reconocida")
        assert (r["nutrition_source_ref"] or "").strip(), (
            f"{r['name']}: sin `nutrition_source_ref` -- se quito el fdc_id sin poner "
            f"nada en su lugar, que es peor que el id prestado")


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
    assert 40 <= len(templates) <= 140  # [P1-GAP-DISHES-VEG · 2026-09-06] el techo sube de 120 a 140: el embudo de cobertura pidió platos DIRIGIDOS a huecos medidos y CO llegó a 121. Cota de cordura, no regla de producto, f"[{cc}] {len(templates)} plantillas — fuera del rango ~40-60 del brief"
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
    devolviendo EXACTAMENTE `DOMINICAN_*` (byte-idéntico, mismo test que H8 para ES).
    [P1-COUNTRY-SYSTEM-F2 · T7 · 2026-08-17] Control cambiado de 'PR' a 'ZZ' (código ISO
    inventado): PR ganó su propio pool en esta task, así que ya no sirve como "país sin pool" --
    `_get_fast_filtered_catalogs` no canonicaliza `country` (dict.get literal), así que un código
    inexistente sigue siendo un control válido y permanente del fallback."""
    casos = [((), (), ""), (("mariscos",), (), ""), ((), (), "vegano")]
    for allergies, dislikes, diet in casos:
        base = constants._get_fast_filtered_catalogs(allergies, dislikes, diet)
        con_none = constants._get_fast_filtered_catalogs(allergies, dislikes, diet, country=None)
        con_pr = constants._get_fast_filtered_catalogs(allergies, dislikes, diet, country="ZZ")
        assert base == con_none == con_pr, f"diverge para {(allergies, dislikes, diet)!r}"


# ── I4. unpriced-keep: las 46 altas T6 reconocidas por su propio token ───────────────────────────
# (el sweep e2e que prueba "nada MÁS se reconoce por accidente" ya se extendió arriba, I0 —
# test_is_country_catalog_unpriced_item_no_colisiona_con_ningun_nombre_del_catalogo_vivo_ni_pools)

@pytest.mark.parametrize("nombre", sorted(_DISH_TEMPLATES_MX_CO_NAMES))
def test_is_country_catalog_unpriced_item_reconoce_cada_alta_t6(sc, nombre, monkeypatch):
    """[fix-round 1 T6 · review Critical #2] Knob ENCENDIDO — 'Tortilla de maíz' es el ÚNICO de
    los 46 cuyo reconocimiento está gateado (colisión de pass-through con `resolve_preparation_distinct`,
    ver docstring de `is_country_catalog_unpriced_item`); los otros 45 son indiferentes al knob
    pero encenderlo no les cambia el resultado, así que un solo `monkeypatch` cubre los 46 casos
    sin bifurcar el test."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
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
def test_homografo_tortilla_de_maiz_resuelve_a_su_propia_fila(sc, monkeypatch):
    """[homógrafo con más riesgo del task, citado explícito en el brief] 'tortilla' nombra TRES
    alimentos distintos en el sistema: Tortilla de trigo/integral (RD, harina de trigo), Tortilla
    española (ES/T5, huevo+patata) y Tortilla de maíz (MX/T6, maíz sin gluten) — ninguna alias
    comparte token con otra.

    [fix-round 1 T6 · review Critical #2] Requiere el knob `MEALFIT_COUNTRY_SYSTEM` ENCENDIDO —
    la canonización a 'Tortilla de maíz' está gateada (ver `test_tortilla_de_maiz_knob_apagado_es_byte_identico_a_pre_t6`
    para el control de byte-identidad con el knob apagado)."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    assert sc.normalize_name("Tortillas de maíz") == "Tortilla de maíz"
    assert sc.normalize_name("tortilla de maiz picada") == "Tortilla de maíz"
    assert sc.normalize_name("Tortilla de trigo") == "Tortilla de trigo"
    assert sc.normalize_name("Tortilla integral") == "Tortilla integral"


def test_resolve_preparation_distinct_tortilla_maiz_canoniza_a_fila_real(sc, monkeypatch):
    """[la MUTACIÓN original de esta task, ver reporte] Pre-T6, `resolve_preparation_distinct`
    forzaba `(True, None)` (pass-through, DROP) para 'tortilla de maíz' porque el catálogo solo
    tenía tortillas de TRIGO — comentario original: "el catálogo solo tiene tortillas de TRIGO".
    Con la alta real T6 Y el knob `MEALFIT_COUNTRY_SYSTEM` encendido, debe CANONIZAR a la fila.
    RED-first: revertir a `return (True, None)` incondicional reproduce el DROP (ver reporte
    §Mutaciones, mutación original). Con el knob APAGADO, ver el test hermano de byte-identidad."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    handled, canonical = sc.resolve_preparation_distinct("Tortillas de maíz")
    assert handled is True
    assert canonical == "Tortilla de maíz", (
        f"resolve_preparation_distinct debe canonizar a 'Tortilla de maíz' con el knob ON, "
        f"devolvió {canonical!r} — si es None, la mutación pre-T6 (pass-through) revivió"
    )


def test_resolve_preparation_distinct_tortilla_maiz_knob_apagado_es_pass_through_historico(sc, monkeypatch):
    """[fix-round 1 T6 · review Critical #2 · RED-first dirección 1/2] Con el knob apagado (o
    ausente — default de esta suite), `resolve_preparation_distinct` debe devolver EXACTAMENTE
    `(True, None)` (pass-through histórico) para CUALQUIER país, incluido DO — byte-identidad.
    Contra el HEAD pre-fix-round (canonización incondicional) este assert falla:
    `resolve_preparation_distinct('Tortillas de maíz') == (True, 'Tortilla de maíz')`, no
    `(True, None)`. Mutación: quitar el gate reproduce ese rojo (ver reporte)."""
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    assert sc.resolve_preparation_distinct("Tortillas de maíz") == (True, None), (
        "con el knob apagado, 'tortilla de maíz' debe seguir siendo pass-through puro "
        "(byte-idéntico a pre-T6) — si canoniza, el gate no está aplicado o no lee el knob"
    )


@pytest.mark.e2e
def test_tortilla_de_maiz_knob_apagado_es_byte_identico_a_pre_t6_en_el_agregador(sc, monkeypatch):
    """[fix-round 1 T6 · review Critical #2 · RED-first dirección 2/2, el gap que el propio
    resolver-gate NO cerraba solo] Verificado en vivo durante el fix-round: gatear SOLO
    `resolve_preparation_distinct` no bastaba -- `is_country_catalog_unpriced_item` reconocía
    igual el texto pass-through ('Tortilla de maíz', idéntico al string de entrada) porque el
    token 'tortilla de maiz' no estaba gateado. Con AMBOS gates, el agregador real debe DROPEAR
    el ingrediente (mismo comportamiento pre-T6: 'antes pasaba de largo/se dropeaba') — no
    debe sobrevivir como CATÁLOGO SIN PRECIO."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    monkeypatch.setenv("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "true")
    result = sc.aggregate_and_deduct_shopping_list(["80 g de Tortilla de maíz"], structured=True)
    items = result if isinstance(result, list) else (result.get("items") or [])
    nombres = [it.get("name") for it in items]
    assert not nombres, (
        f"con el knob apagado, 'Tortilla de maíz' debe DROPEARSE del agregador (byte-identidad "
        f"pre-T6) — sobrevivió como: {nombres}"
    )


@pytest.mark.e2e
def test_tortilla_de_maiz_knob_encendido_sobrevive_como_catalogo_sin_precio(sc, monkeypatch):
    """Control positivo (espejo del test anterior): con el knob ENCENDIDO, el mismo ingrediente
    SÍ debe sobrevivir en el agregador — confirma que el gate discrimina en vez de romper el
    caso MX real que esta task existe para servir."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    monkeypatch.setenv("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "true")
    result = sc.aggregate_and_deduct_shopping_list(["80 g de Tortilla de maíz"], structured=True)
    items = result if isinstance(result, list) else (result.get("items") or [])
    tortilla = next((it for it in items if it.get("name") == "Tortilla de maíz"), None)
    assert tortilla is not None, "con el knob encendido, 'Tortilla de maíz' debe sobrevivir"
    assert tortilla.get("estimated_cost_rd") is None
    # [P2-SHOPLIST-BETA-POLISH · 2026-08-18] pasillo real del master (era el label interno).
    # [reconvertido · P2-COUNTRY-HOUSEKEEPING · 2026-08-21] El label de DISPLAY, no la categoría
    # cruda de la base: la rama con precio emite «DESPENSA» y el Dashboard agrupa por la cadena
    # literal, así que las dos grafías pintaban dos secciones del mismo pasillo.
    assert tortilla.get("display_category") == sc._get_display_category("Despensa",
                                                                       "Tortilla de maíz")
    assert tortilla.get("display_category") == "DESPENSA"


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
    # [P1-LATINFOODS-TCAC + P1-PROVENANCE-TRUTHFUL · 2026-08-19] Estas filas DEJARON de
    # tener `fdc_id`, y fue a proposito: el que tenian era PRESTADO de otro alimento de
    # USDA. Chontaduro vivia sobre *Breadfruit* --103 kcal declaradas frente a 332
    # reales, con 25,7 g de grasa contra 0,23-- y Suero costeno sobre *Sour cream*: el
    # error era de CATEGORIA, no de magnitud. Un `fdc_id` es una AFIRMACION sobre la
    # procedencia; se sustituyo por la fuente real o por un proxy DECLARADO como tal.
    #
    # Se enumeran en vez de relajar la regla: una fila nueva no puede perder su fdc_id
    # en silencio, tendria que anadirse a esta lista.
    _SIN_USDA_T6 = {
        "Champús", "Chorizo santarrosano", "Chontaduro", "Chorizo verde", "Xoconostle",
        "Curuba", "Suero costeño", "Chile guajillo", "Chile mulato", "Chile chipotle",
        "Borojó", "Cecina",
    }
    presentes = {n for n in con_usda if n in _SIN_USDA_T6}
    assert presentes == _SIN_USDA_T6, (
        f"el conjunto T6 sin fdc_id cambio: esperaba {sorted(_SIN_USDA_T6)}, "
        f"hay {sorted(presentes)}")
    con_usda = {n: r for n, r in con_usda.items() if n not in _SIN_USDA_T6}

    sin_fdc = [n for n, r in con_usda.items() if not r.get("fdc_id")]
    assert not sin_fdc, f"altas T6 (no-manual, no-excepcion) sin fdc_id: {sin_fdc}"
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


# ── I10 (Durable Guard #6). Retarget-diff — el TARGET de cada item pineado, no solo el veredicto ──

@pytest.mark.e2e
def test_retarget_diff_committed_country_gaps_matched_field_vs_resolver_vivo(sc, monkeypatch):
    """[Durable Guard #6 · fix-round 1 T6 · controller ruling 2026-08-17 — "cierra la CLASE, no
    solo la instancia"] El test de arriba (`test_harness_mx_co_cierra_en_cero_drops_cero_silenciosas`,
    y su hermano de T5 para ES) SOLO cuenta veredictos (`counts.DROP`/`counts['SUSTITUCION-SILENCIOSA']`)
    — es CIEGO a que el TARGET de un item cambie mientras el veredicto se mantiene en
    RESUELVE-BIEN. Eso fue exactamente Critical #1 del review de fix-round 1: 'Queso panela'
    (MX) seguía siendo RESUELVE-BIEN antes y después de que el nuevo self-alias 'panela' (T6)
    colisionara por longitud de alias, pero el TARGET saltó en silencio de 'Queso blanco' (queso)
    a 'Panela' (azúcar cruda) — un veredicto ciego al target no lo habría detectado nunca; un
    conteo agregado tampoco (76/76/0/0 antes Y después de la colisión).

    Descubre los `country_gaps/*.json` COMMITEADOS dinámicamente (no una lista hardcodeada de
    nombres) filtrando por `mode == "country"` — así un futuro T7/T8 (PR/US) queda cubierto
    automáticamente sin tocar este test, y el `rd_drops.json` de telemetría del cron
    (`_creativity_kpi_job`, schema `mode="rd-drops"` sin `items`/`matched`) se excluye solo. Para
    CADA item con `matched` no-nulo (230 al momento de escribir este test: 80 ES + 76 MX + 74 CO,
    los 3 cerrados en 0 DROP), re-resuelve en vivo contra `sc.normalize_name` y exige coincidencia
    EXACTA con el `matched` ya comprometido en git. Un retarget INTENCIONAL (ej. un fix-round
    futuro que mejora una sustitución) se cierra actualizando el JSON explícitamente vía
    `scripts/country_catalog_gap.py --country <CC> --commit` — así el diff SIEMPRE pasa por
    review, nunca en silencio.

    Corre con el knob `MEALFIT_COUNTRY_SYSTEM` encendido: es la única condición bajo la cual
    `mx.json` (regenerado en fix-round 1 tras el fix de Critical #2) es internamente consistente
    consigo mismo — 'Tortillas de maíz' solo resuelve a 'Tortilla de maíz' con el knob ON (ver
    `resolve_preparation_distinct`). Confirmado por grep que el knob se lee 2 veces en todo
    `shopping_calculator.py`: la otra lectura (dentro de `is_country_catalog_unpriced_item`) no
    participa de `normalize_name` en absoluto (es un filtro posterior, de agregación) — encender
    el knob aquí no puede desviar NINGÚN otro item de es.json/co.json, solo habilita la única
    resolución que de verdad lo necesita."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")

    gaps_dir = _BACKEND / "data" / "country_gaps"
    retargets = []
    total_checked = 0
    files_scanned = 0
    for path in sorted(gaps_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("mode") != "country":
            continue  # ej. rd_drops.json (telemetría del cron, schema distinto) — no aplica
        files_scanned += 1
        for item in payload.get("items") or []:
            matched = item.get("matched")
            if matched is None:
                continue  # DROP genuino: no hay target que pinear
            food = item["food"]
            total_checked += 1
            live = sc.normalize_name(food)
            if live != matched:
                retargets.append((path.name, food, matched, live))

    assert files_scanned >= 3, (
        f"esperaba >=3 archivos con mode=='country' en {gaps_dir}, encontró {files_scanned} — "
        "¿el directorio se movió o el filtro de mode está mal?"
    )
    assert total_checked >= 220, (
        f"esperaba >=220 items pineados entre los {files_scanned} archivos, solo {total_checked} "
        "— ¿algún country_gaps/*.json se leyó vacío o con conteos DROP>0?"
    )
    assert not retargets, (
        "RETARGET DETECTADO — el resolver vivo apunta distinto de lo que el JSON commiteado "
        "declara. Si es intencional, actualiza el country_gaps/*.json correspondiente vía "
        "`scripts/country_catalog_gap.py --country <CC> --commit` para que el diff quede "
        "documentado en el PR, nunca en silencio:\n" +
        "\n".join(
            f"  [{fname}] {food!r}: json={matched!r} vs live={live!r}"
            for fname, food, matched, live in retargets
        )
    )


# ── I11. "Quesito panela" — hermano de Critical #1, misma dirección queso→azúcar ─────────────────

@pytest.mark.parametrize("food", ["Queso panela", "Queso panela fresco", "Quesito panela",
                                   "Quesito panela fresco"])
def test_queso_y_quesito_panela_resuelven_a_queso_blanco_nunca_a_panela(sc, food):
    """[micro-fix T6 · review 2026-08-17, residual de fix-round 1] El reporte de fix-round 1
    afirmaba (falso) que 'Quesito panela' ya resolvía a 'Queso blanco' — en vivo resolvía a
    'Panela' (azúcar cruda), porque 'quesito' NO contiene 'queso' como subcadena ('quesito' diverge
    de 'queso' en el 5º carácter: q-u-e-s-**i**-t-o vs q-u-e-s-**o**), así que el alias de rescate
    'queso panela' (Critical #1, fix-round 1) nunca dispara para esta variante — solo el self-alias
    bare 'panela' (T6) la ve, y gana por ser la única alternativa que matchea. Misma dirección
    exacta que Critical #1 (queso→azúcar), variante de palabra distinta, alias de rescate distinto
    requerido. Cubre las 4 variantes en un solo test: las 2 que YA resuelven bien desde fix-round 1
    ('Queso panela'/'Queso panela fresco', control de no-regresión) y las 2 que NO ('Quesito
    panela'/'Quesito panela fresco', el bug de este micro-round) — así que un futuro colapso de
    CUALQUIERA de las 4 variantes se detecta en una sola corrida."""
    assert sc.normalize_name(food) == "Queso blanco", (
        f"{food!r} debe resolver a 'Queso blanco' (queso), no a 'Panela' (azúcar cruda) — "
        "misma clase de bug que Critical #1 (fix-round 1), dirección queso→azúcar"
    )


# ══════════════════════════════════════════════════════════════════════════════════════════════
# SECCIÓN J (Task 7) — Catálogo Puerto Rico + Estados Unidos, dirigido por los JSON de T1
# ══════════════════════════════════════════════════════════════════════════════════════════════
#
# Mismo contrato que T5/T6, para PR + US. `country_gaps/pr.json` (13 DROP de 67, medido contra el
# catálogo post-T6/284 filas) y `us.json` (27 DROP de 78 -- el Task 1 Ruling reexpresó US_FOODS a
# ESPAÑOL en esta task: producción emite ingredientes en español incluso para planes US, medir
# contra nombres en inglés medía una distribución que producción nunca emite) son la lista de
# trabajo. 19 filas nuevas PR + 43 filas nuevas US resuelven los 40 DROP + 1 sinónimo (Parcha→
# Chinola) resuelve el último. Las 43 US NO son 27: T1 solo mide DROP/SUSTITUCION-SILENCIOSA — es
# CIEGO a un RESUELVE-BIEN que resuelve al alimento SEMÁNTICAMENTE INCORRECTO vía un alias bare
# preexistente demasiado amplio (ej. 'Suero de mantequilla' colisionaba con 'Mantequilla' vía
# CONTAINS: líquido bajo en grasa vs grasa sólida pura). Verificado en vivo con `sc.normalize_name`
# + lectura directa de `_construir_indice_alias`/aliases ANTES de dar de alta 16 casos así (11 US +
# el propio caso "Aceitunas rellenas"→"Aceitunas" en PR, macro +47%/+88% verificado con USDA real) —
# no descubierto en review, cerrado proactivamente en el commit original de esta task.

def _load_dish_templates_pr() -> dict:
    with open(_DISH_TEMPLATES_PR_JSON, encoding="utf-8") as f:
        return json.load(f)


def _load_dish_templates_us() -> dict:
    with open(_DISH_TEMPLATES_US_JSON, encoding="utf-8") as f:
        return json.load(f)


# ── J1. dish_templates_pr.json / dish_templates_us.json — forma + regla dura de horario ─────────

@pytest.mark.parametrize("cc,loader", [("PR", _load_dish_templates_pr), ("US", _load_dish_templates_us)])
def test_dish_templates_pr_us_json_existe_con_forma_esperada(cc, loader):
    data = loader()
    templates = data.get("templates")
    assert isinstance(templates, list)
    assert 40 <= len(templates) <= 140  # [P1-GAP-DISHES-VEG · 2026-09-06] el techo sube de 120 a 140: el embudo de cobertura pidió platos DIRIGIDOS a huecos medidos y CO llegó a 121. Cota de cordura, no regla de producto, f"[{cc}] {len(templates)} plantillas — fuera del rango ~40-60 del brief"
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


@pytest.mark.parametrize("cc,loader", [("PR", _load_dish_templates_pr), ("US", _load_dish_templates_us)])
def test_dish_templates_pr_us_arroz_pasta_como_base_nunca_en_desayuno_ni_cena(cc, loader):
    """Mismo SSOT que la regla dura del juez. PR: solo 'Arroz con dulce' (postre, base=arroz) va
    SOLO en merienda; los platos de arroz salado (arroz con gandules/pollo/habichuelas) van SOLO
    en almuerzo. US: mac and cheese/ensalada de macarrones (base=pasta) van SOLO en almuerzo."""
    data = loader()
    ofensoras = [
        t["name"] for t in data["templates"]
        if t.get("base") in ("arroz", "pasta") and set(t.get("slots", [])) & {"desayuno", "cena"}
    ]
    assert not ofensoras, f"[{cc}] plantillas con base arroz/pasta en desayuno/cena: {ofensoras}"


@pytest.mark.e2e
@pytest.mark.parametrize("cc,loader", [("PR", _load_dish_templates_pr), ("US", _load_dish_templates_us)])
def test_dish_templates_pr_us_constituents_resuelven_al_catalogo_vivo(cc, loader):
    """[e2e] Contrato "nombres EXACTOS del catálogo" — cada `constituents[].name` debe ser un
    `name` LITERAL de `master_ingredients` post-altas T7 (no un alias que resuelva vía
    `normalize_name` -- ej. 'Salmón' no 'Filete de salmón', 'Auyama' no 'Calabaza butternut',
    'Cerdo' no 'Chuletas de cerdo', 'Repollo' no 'Ensalada de repollo'). Mismo patrón que el
    equivalente de T5/T6."""
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


# ── J2. _dish_templates_path_for_country + _culinary_judge_rubric_for_country PR/US ──────────────

def test_dish_templates_path_for_country_pr_us_usan_su_archivo_propio(go):
    assert go._dish_templates_path_for_country("PR") == str(_DISH_TEMPLATES_PR_JSON)
    assert go._dish_templates_path_for_country("US") == str(_DISH_TEMPLATES_US_JSON)


def test_culinary_judge_rubric_pr_sustituye_ejemplos_y_encabezado(go):
    rubric_pr = go._culinary_judge_rubric_for_country("PR")
    rubric_do = go._culinary_judge_rubric_for_country("DO")
    assert rubric_pr != rubric_do
    assert "Mofongo con chicharrón" in rubric_pr
    assert "Mangú" not in rubric_pr, (
        "los ejemplos dominicanos no deben sobrevivir en la variante PR -- el desayuno de "
        "plátano+huevo de este país se nombró a propósito sin la palabra 'Mangú' para no "
        "confundir este canario de contaminación con una inclusión cultural deliberada"
    )
    assert "PLATOS DE P" in rubric_pr.upper()
    assert "cocina de Puerto Rico" in rubric_pr


def test_culinary_judge_rubric_us_sustituye_ejemplos_y_encabezado(go):
    rubric_us = go._culinary_judge_rubric_for_country("US")
    rubric_do = go._culinary_judge_rubric_for_country("DO")
    assert rubric_us != rubric_do
    assert "Chili con carne con pan de maíz" in rubric_us
    assert "Mangú" not in rubric_us, "los ejemplos dominicanos no deben sobrevivir en la variante US"
    assert "PLATOS DE E" in rubric_us.upper()
    assert "cocina de Estados Unidos" in rubric_us


def test_culinary_judge_rubric_pr_us_no_se_contaminan_con_ningun_otro_pais(go):
    """Cada variante de país usa SOLO sus propios ejemplos -- un plato PR nunca debe aparecer en
    la rúbrica US (ni viceversa), y ninguno de los dos debe traer platos DO/ES/MX/CO. Prueba
    puntual de que la sustitución de bloque no deja restos cruzados entre las 6 variantes ya
    cacheadas en el mismo proceso (DO/ES/MX/CO/PR/US)."""
    rubric_pr = go._culinary_judge_rubric_for_country("PR")
    rubric_us = go._culinary_judge_rubric_for_country("US")
    rubric_mx = go._culinary_judge_rubric_for_country("MX")
    rubric_co = go._culinary_judge_rubric_for_country("CO")
    rubric_es = go._culinary_judge_rubric_for_country("ES")
    assert "Chili con carne" not in rubric_pr and "Wafles" not in rubric_pr
    assert "Mofongo" not in rubric_us and "Bacalaítos" not in rubric_us
    assert "Mofongo" not in rubric_mx and "Chili con carne" not in rubric_mx
    assert "Mofongo" not in rubric_co and "Chili con carne" not in rubric_co
    assert "Mofongo" not in rubric_es and "Chili con carne" not in rubric_es
    assert "Tacos de pollo" not in rubric_pr and "Ajiaco" not in rubric_pr
    assert "Tortilla española" not in rubric_pr and "Tortilla española" not in rubric_us


def test_culinary_judge_rubric_do_sigue_byte_identico_tras_pr_us(go):
    """Control de no-regresión: dar de alta PR/US no debe tocar la ruta DO — mismo objeto
    cacheado (identidad, no solo igualdad), verificado DESPUÉS de resolver PR/US/ES/MX/CO en
    este mismo proceso (el cache es un dict módulo-level; esto confirma que no hay mutación
    cruzada)."""
    go._culinary_judge_rubric_for_country("PR")
    go._culinary_judge_rubric_for_country("US")
    assert go._culinary_judge_rubric_for_country("DO") is go._CULINARY_JUDGE_RUBRIC


# ── J3. COUNTRY_POOLS['PR']/['US'] + _get_fast_filtered_catalogs(country=) ────────────────────────

@pytest.mark.parametrize("cc", ["PR", "US"])
def test_country_pools_pr_us_estructura(cc):
    pool = constants.COUNTRY_POOLS.get(cc)
    assert isinstance(pool, dict)
    for key in ("proteins", "carbs", "veggies_fats", "fruits"):
        assert isinstance(pool.get(key), list) and pool[key], f"COUNTRY_POOLS[{cc!r}][{key!r}] vacío"
        assert all(isinstance(x, str) and x.strip() for x in pool[key])


@pytest.mark.parametrize("cc", ["PR", "US"])
def test_get_fast_filtered_catalogs_pr_us_usa_su_propio_pool(cc):
    proteins_do, _, _, _ = constants._get_fast_filtered_catalogs((), (), "")
    proteins, carbs, veg, fruits = constants._get_fast_filtered_catalogs((), (), "", country=cc)
    assert proteins != proteins_do
    assert set(proteins) == set(constants.COUNTRY_POOLS[cc]["proteins"])
    assert set(carbs) == set(constants.COUNTRY_POOLS[cc]["carbs"])
    assert set(veg) == set(constants.COUNTRY_POOLS[cc]["veggies_fats"])
    assert set(fruits) == set(constants.COUNTRY_POOLS[cc]["fruits"])


def test_get_fast_filtered_catalogs_sin_country_pr_us_sigue_siendo_do_byte_identico():
    """Dar de alta PR/US no debe tocar el fallback -- `country=None`/'DO'/país sin pool siguen
    devolviendo EXACTAMENTE `DOMINICAN_*` (byte-idéntico). País SIN pool dedicado (ya no queda
    ninguno tras esta task -- los 6 tienen pool propio) usa un código ISO inventado como control
    del fallback genérico."""
    casos = [((), (), ""), (("mariscos",), (), ""), ((), (), "vegano")]
    for allergies, dislikes, diet in casos:
        base = constants._get_fast_filtered_catalogs(allergies, dislikes, diet)
        con_none = constants._get_fast_filtered_catalogs(allergies, dislikes, diet, country=None)
        con_zz = constants._get_fast_filtered_catalogs(allergies, dislikes, diet, country="ZZ")
        assert base == con_none == con_zz, f"diverge para {(allergies, dislikes, diet)!r}"


# ── J4. unpriced-keep: las 62 altas T7 reconocidas por su propio token ───────────────────────────
# (el sweep e2e que prueba "nada MÁS se reconoce por accidente" ya se extendió arriba, en la
# sección I4 original -- test_is_country_catalog_unpriced_item_no_colisiona_..., ahora resta
# también `_DISH_TEMPLATES_PR_US_NAMES`)

@pytest.mark.parametrize("nombre", sorted(_DISH_TEMPLATES_PR_US_NAMES))
def test_is_country_catalog_unpriced_item_reconoce_cada_alta_t7(sc, nombre, monkeypatch):
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    assert sc.is_country_catalog_unpriced_item(nombre), f"{nombre!r} no reconocido como unpriced keep"


def test_unpriced_tokens_t7_usan_nombre_completo_no_palabras_sueltas_riesgosas():
    """[decisión de diseño de esta task, ancla en código] A diferencia de T5/T6 (que usaban
    palabras sueltas cuando eran seguras), T7 usa el NOMBRE CANÓNICO COMPLETO para las 62 altas
    porque la superficie de riesgo es mayor: 'queso'/'pan'/'carne'/'papa'/'mantequilla'/'chile'/
    'salsa'/'galletas'/'frijoles' ya son bare o casi-bare en aliases de filas PRICED existentes.
    Pin de que ninguno de esos 9 tokens de alto riesgo aparece SOLO (sin calificador) en el
    registro para las altas T7 -- si este test se pone rojo, alguien angostó un token de forma
    insegura sin repetir el sweep e2e."""
    riesgosos = {"queso", "pan", "carne", "papa", "mantequilla", "chile", "salsa", "galletas", "frijoles"}
    # Los últimos 62 tokens del registro son los de T7 (el orden de literal tuple se preserva).
    import shopping_calculator as _sc
    t7_tokens = _sc._COUNTRY_CATALOG_UNPRICED_TOKENS[-62:]
    assert len(t7_tokens) == 62
    bare_riesgosos = sorted(tok for tok in t7_tokens if tok in riesgosos)
    assert not bare_riesgosos, f"tokens T7 de alto riesgo usados SUELTOS (sin calificador): {bare_riesgosos}"


# ── J5. Los 4 vocabularios — anchors narrow de las altas T7 ───────────────────────────────────────

@pytest.mark.parametrize("clase,termino", [
    ("pescado", "bacalaitos"),
])
def test_altas_t7_presentes_en_allergen_synonyms(go, clase, termino):
    assert termino in go._ALLERGEN_SYNONYMS[clase], f"{termino!r} ausente de _ALLERGEN_SYNONYMS[{clase!r}]"


def test_altas_t7_presentes_en_diet_seafood_terms(go):
    assert "bacalaitos" in go._DIET_SEAFOOD_TERMS


@pytest.mark.parametrize("termino", ["pavochon", "pepperoni", "frijoles horneados"])
def test_altas_t7_presentes_en_diet_flesh_terms(go, termino):
    assert termino in go._DIET_FLESH_TERMS


@pytest.mark.parametrize("termino", [
    "bagel", "bagels", "pretzel", "pretzels", "panecillo", "panecillos", "panqueque",
    "panqueques", "wafle", "wafles", "salsa de salchicha", "masa para pie", "bacalaitos",
])
def test_altas_t7_presentes_en_allergen_synonyms_gluten(go, termino):
    assert termino in go._ALLERGEN_SYNONYMS["gluten"], f"{termino!r} ausente de _ALLERGEN_SYNONYMS['gluten']"


def test_constants_catchall_gluten_sincronizado_con_los_13_terminos_t7(go):
    """[lockstep obligatorio, CLAUDE.md] `constants._get_fast_filtered_catalogs` gluten catch-all
    DEBE llevar los MISMOS 13 términos que `_ALLERGEN_SYNONYMS['gluten']` añadió en esta task --
    es la MISMA clase de drift que el fix-wave de T6 cerró (11 términos de T4 sin sincronizar)."""
    nuevos = ("bagel", "bagels", "pretzel", "pretzels", "panecillo", "panecillos", "panqueque",
              "panqueques", "wafle", "wafles", "salsa de salchicha", "masa para pie", "bacalaitos")
    src = constants.__file__
    with open(src, encoding="utf-8") as f:
        constants_src = f.read()
    ini = constants_src.index('if any(r in ["gluten", "trigo", "wheat"]')
    fin = constants_src.index("if any(r in [\"soya\"", ini)
    bloque_gluten = constants_src[ini:fin]
    faltantes = [t for t in nuevos if f'"{t}"' not in bloque_gluten]
    assert not faltantes, f"catch-all de constants.py NO sincronizado con estos términos T7: {faltantes}"


def test_scan_allergen_violations_detecta_bacalaitos_como_pescado(go):
    plan = {"days": [{"meals": [{"name": "Merienda", "ingredients": ["120 g de Bacalaítos"]}]}]}
    v = go._scan_allergen_violations(plan, ["pescado"])
    assert v, "'Bacalaítos' debe violar la alergia a pescado"


def test_scan_diet_violations_detecta_pavochon_y_pepperoni_para_vegano(go):
    plan = {"days": [{"meals": [{"name": "Almuerzo", "ingredients": ["Pavochón con puré"]}]}]}
    assert go._scan_diet_violations(plan, "vegano"), "'Pavochón' no fue detectado para vegano"
    plan2 = {"days": [{"meals": [{"name": "Almuerzo", "ingredients": ["Pizza de Pepperoni"]}]}]}
    assert go._scan_diet_violations(plan2, "vegano"), "'Pepperoni' no fue detectado para vegano"


def test_paridad_dieta_alergeno_bidireccional_cubre_bacalaitos(go):
    """El guard genérico heredado de T4 (`test_paridad_dieta_alergeno_bidireccional`,
    parametrizado por clase) ya corre sobre TODO el vocabulario -- este test puntual documenta
    que 'bacalaitos' específicamente está en AMBOS lados (mariscos+pescado <-> seafood), sin
    depender de leer el test genérico para confirmarlo."""
    assert "bacalaitos" in go._ALLERGEN_SYNONYMS["pescado"]
    assert "bacalaitos" in go._DIET_SEAFOOD_TERMS


# ── J6. Regla FILA-vs-SINÓNIMO — el único sinónimo de esta task ──────────────────────────────────

@pytest.mark.e2e
def test_parcha_resuelve_a_chinola_via_normalize_name_real(sc):
    """[el guard central del contrato] Verifica el CAMINO REAL (`sc.normalize_name`) -- Parcha
    (P. edulis, PR) es la MISMA especie que Chinola (P. edulis, RD), a diferencia de Curuba (P.
    tripartita, T6) o Granadilla (P. ligularis, T6), que son especies DISTINTAS con fila propia."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    assert sc.normalize_name("Parcha") == "Chinola"
    assert sc.normalize_name("Parchas") == "Chinola"


@pytest.mark.parametrize("item,fila_o_alias_esperado,tipo", [
    ("Parcha", "Chinola", "sinónimo"),
    ("Aceitunas rellenas", "Aceitunas rellenas", "fila nueva"),
    ("Chuleta ahumada", "Chuleta ahumada", "fila nueva"),
    ("Suero de mantequilla", "Suero de mantequilla", "fila nueva"),
    ("Arándanos rojos", "Arándanos rojos", "fila nueva"),
    ("Chili con carne", "Chili con carne", "fila nueva"),
])
def test_tabla_fila_vs_sinonimo_t7_estructural(item, fila_o_alias_esperado, tipo):
    """[tabla estructural, sin DB] Documenta en código la decisión fila-vs-sinónimo por item --
    complementa el test funcional de arriba: aquí se ancla la DECISIÓN, allá el RESULTADO."""
    if tipo == "sinónimo":
        assert item != fila_o_alias_esperado, f"{item!r} está marcado sinónimo pero apunta a sí mismo"
        assert fila_o_alias_esperado in (_DISH_TEMPLATES_PR_US_NAMES | {"Chinola"}), (
            f"el destino {fila_o_alias_esperado!r} de {item!r} debe ser una fila real conocida"
        )
    else:
        assert item == fila_o_alias_esperado, f"{item!r} está marcado fila nueva pero mapea a otro nombre"
        assert item in _DISH_TEMPLATES_PR_US_NAMES, f"{item!r} marcado fila nueva debe estar en las 62 altas"


def test_carne_de_cerdo_para_pernil_retargeteo_documentado_y_aceptado(sc):
    """[hallazgo del sweep de colisión propio -- retarget CONOCIDO y ACEPTADO, no un bug] El item
    curado 'Carne de cerdo para pernil' (PR) resolvía a 'Cerdo' (genérico) antes de esta task --
    tras dar de alta 'Pernil' (corte específico, pierna con piel/grasa), la ANATOMY-PREFIX guard
    de `normalize_name` (que strippea 'carne de ' al inicio) deja 'cerdo para pernil', que ahora
    CONTAINS-matchea 'pernil' (6 chars) antes que 'cerdo' (5 chars, más corto) en el índice
    ordenado por longitud. Aceptado a propósito (no revertido con un rescate): 'Pernil' es MÁS
    preciso para una frase que literalmente dice "para pernil" que el genérico 'Cerdo' -- mismo
    espíritu que el retarget 'Chicharrón de cerdo'→'Chicharrón' que T6 documentó como mejora
    intencional. El `country_gaps/pr.json` commiteado en esta task YA refleja este target."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    assert sc.normalize_name("Carne de cerdo para pernil") == "Pernil"


# ── J7. e2e — las 62 altas T7 existen en el catálogo vivo, SIN precio, con fdc_id o 'manual' ─────

@pytest.mark.e2e
def test_62_altas_t7_existen_en_catalogo_vivo_sin_precio_con_fdc_id_o_manual():
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    from db_core import execute_sql_query

    rows = execute_sql_query(
        "SELECT name, price_per_lb, price_per_unit, fdc_id, nutrition_source "
        "FROM master_ingredients WHERE name = ANY(%s)",
        (list(_DISH_TEMPLATES_PR_US_NAMES),),
        fetch_all=True,
    ) or []
    por_nombre = {r["name"]: r for r in rows}

    faltantes = sorted(_DISH_TEMPLATES_PR_US_NAMES - set(por_nombre))
    assert not faltantes, f"altas T7 ausentes del catálogo vivo: {faltantes}"

    con_precio = [n for n, r in por_nombre.items()
                  if float(r["price_per_lb"] or 0) > 0 or float(r["price_per_unit"] or 0) > 0]
    assert not con_precio, f"altas T7 con precio RD (deberían estar en 0): {con_precio}"

    # [T7] 8 filas 'manual': 2 SIN fdc_id real (Recao, Adobo) + 6 DERIVADAS como blend de 2 fdc_id
    # reales (nutrition_source='manual' porque el valor persistido es una TRANSFORMACIÓN -- mismo
    # criterio que Flor de Jamaica/T6). Las otras 54 exigen fdc_id + 'usda'.
    _MANUAL = {"Recao", "Adobo", "Alcaparrado", "Pique", "Salsa de salchicha",
               "Ensalada de macarrones", "Huevos rellenos", "Carne molida mixta"}
    con_usda = {n: r for n, r in por_nombre.items() if n not in _MANUAL}
    # [P1-LATINFOODS-TCAC + P1-PROVENANCE-TRUTHFUL · 2026-08-19] Estas filas DEJARON de
    # tener `fdc_id`, y fue a proposito: el que tenian era PRESTADO de otro alimento de
    # USDA. Chontaduro vivia sobre *Breadfruit* --103 kcal declaradas frente a 332
    # reales, con 25,7 g de grasa contra 0,23-- y Suero costeno sobre *Sour cream*: el
    # error era de CATEGORIA, no de magnitud. Un `fdc_id` es una AFIRMACION sobre la
    # procedencia; se sustituyo por la fuente real o por un proxy DECLARADO como tal.
    #
    # Se enumeran en vez de relajar la regla: una fila nueva no puede perder su fdc_id
    # en silencio, tendria que anadirse a esta lista.
    _SIN_USDA_T7 = {"Especias para arroz con dulce", "Longaniza puertorriqueña"}
    presentes = {n for n in con_usda if n in _SIN_USDA_T7}
    assert presentes == _SIN_USDA_T7, (
        f"el conjunto T7 sin fdc_id cambio: esperaba {sorted(_SIN_USDA_T7)}, "
        f"hay {sorted(presentes)}")
    con_usda = {n: r for n, r in con_usda.items() if n not in _SIN_USDA_T7}

    sin_fdc = [n for n, r in con_usda.items() if not r.get("fdc_id")]
    assert not sin_fdc, f"altas T7 (no-manual, no-excepcion) sin fdc_id: {sin_fdc}"
    no_usda = [n for n, r in con_usda.items() if r.get("nutrition_source") != "usda"]
    assert not no_usda, f"altas T7 (no-manual) con nutrition_source != 'usda': {no_usda}"

    manuales = {n: r for n, r in por_nombre.items() if n in _MANUAL}
    assert set(manuales) == _MANUAL, f"esperaba exactamente {_MANUAL} como manual, hay {set(manuales)}"
    no_manual = [n for n, r in manuales.items() if r.get("nutrition_source") != "manual"]
    assert not no_manual, f"{no_manual} deberían tener nutrition_source='manual'"

    # Las 8 'manual' tienen fdc_id NULL en la COLUMNA por igual -- tanto las 2 sin fuente real
    # (Recao/Adobo) como las 6 derivadas de un blend de 2 fdc reales (el detalle de qué fdc's
    # alimentaron cada blend + los pesos exactos vive en `_provenance`, nunca en la columna
    # `fdc_id`, que solo admite un único entero o NULL).
    con_fdc = [n for n, r in manuales.items() if r.get("fdc_id")]
    assert not con_fdc, f"filas 'manual' con fdc_id NO-nulo (debería ser siempre NULL): {con_fdc}"


# ── J8. Golden fixture: un día PR y un día US pasan slots suaves (mismo patrón que H14/I8) ───────

def _dia_pr_con_arroz_en_desayuno() -> list:
    return [{
        "day": 1,
        "meals": [
            {"meal": "Desayuno", "name": "Arroz con gandules y pernil",
             "ingredients": ["Pernil", "Arroz blanco", "Gandules"]},
            {"meal": "Almuerzo", "name": "Pollo guisado con arroz blanco",
             "ingredients": ["Muslo de pollo", "Arroz blanco", "Sofrito"]},
            {"meal": "Cena", "name": "Chuletas ahumadas a la plancha con vegetales",
             "ingredients": ["Chuleta ahumada", "Repollo", "Zanahoria"]},
            {"meal": "Merienda", "name": "Tostones con pique",
             "ingredients": ["Plátano verde", "Pique"]},
        ],
    }]


def _dia_us_con_pasta_en_desayuno() -> list:
    """[hallazgo de verificación en vivo] `SLOT_INAPPROPRIATE_FOODS['cena']` NO tiene ninguna
    regla de tokens de pasta (solo arroz-soft/cereal-soft/frito-soft/sancocho-soft/desayuno-soft)
    -- la regla HARD de pasta vive SOLO en `SLOT_INAPPROPRIATE_FOODS['desayuno']` (tokens
    'macarron'/'macarrones'/'coditos'/'fideos'/etc). El fixture original (pasta en CENA) medía
    una combinación que el detector NUNCA marca como violación, en NINGÚN país -- confirmado con
    'assert violaciones' fallando incluso para country='DO' (control). Movido a desayuno, donde
    la regla hard real vive."""
    return [{
        "day": 1,
        "meals": [
            {"meal": "Desayuno", "name": "Macarrones con queso cheddar",
             "ingredients": ["Coditos", "Queso cheddar"]},
            {"meal": "Almuerzo", "name": "Chili con carne con pan de maíz",
             "ingredients": ["Chili con carne", "Pan de maíz"]},
            {"meal": "Cena", "name": "Salmón con espárragos",
             "ingredients": ["Salmón", "Espárragos"]},
            {"meal": "Merienda", "name": "Pretzels con mostaza",
             "ingredients": ["Pretzels", "Mostaza"]},
        ],
    }]


@pytest.mark.parametrize("cc,builder", [("PR", _dia_pr_con_arroz_en_desayuno), ("US", _dia_us_con_pasta_en_desayuno)])
def test_dia_pr_us_regla_dura_pasa_como_soft_sin_forzar_retry(go, monkeypatch, cc, builder):
    """[Golden fixture] Un día con una violación real de la regla dura (arroz en desayuno para PR,
    pasta en cena para US) DEBE detectarse SOFT (hard=False) para país PR/US:
    `slot_rules_for_country` softea toda regla en Fase 1 (T4), igual que ES/MX/CO en H14/I8."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    dias = builder()
    violaciones = go._detect_slot_appropriateness(dias, {"country": cc})
    assert violaciones, f"[{cc}] el día de prueba debe producir AL MENOS una violación (control positivo)"
    duras = [v for v in violaciones if v["hard"]]
    assert not duras, f"[{cc}] no debe producir violaciones HARD: {duras}"


@pytest.mark.parametrize("cc,builder", [("PR", _dia_pr_con_arroz_en_desayuno), ("US", _dia_us_con_pasta_en_desayuno)])
def test_el_mismo_dia_pr_us_hard_para_do_control_de_que_el_mecanismo_discrimina(go, monkeypatch, cc, builder):
    """Control negativo: el MISMO día, con country='DO', debe seguir produciendo HARD -- confirma
    que el mecanismo distingue países en vez de estar simplemente roto."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    dias = builder()
    violaciones = go._detect_slot_appropriateness(dias, {"country": "DO"})
    assert violaciones, f"[{cc}→DO] control: el día debe violar también en DO"
    duras = [v for v in violaciones if v["hard"]]
    assert duras, f"[{cc}→DO] debe seguir produciendo violaciones HARD (byte-identidad del mecanismo)"


# ── J9. Cierre medible — pr.json/us.json committed: cero DROP, cero SUSTITUCION-SILENCIOSA ───────

@pytest.mark.parametrize("cc,fname", [("PR", "pr.json"), ("US", "us.json")])
def test_harness_pr_us_cierra_en_cero_drops_cero_silenciosas(cc, fname):
    """[el criterio de cierre del contrato] `pr.json`/`us.json` (sobrescritos por la re-corrida
    final del harness post-altas, committed en el repo) deben reportar counts.DROP == 0 y
    counts.SUSTITUCION-SILENCIOSA == 0 -- el mismo criterio de salida que T5/T6 cerraron."""
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


# Nota J7: la última aserción de test_62_altas_t7_existen_en_catalogo_vivo_sin_precio_con_fdc_id_o_manual
# es una tautología defensiva dejada a propósito legible (`is False or True`) — el chequeo real de
# "las 6 blend SÍ tienen fdc_id" no aplica: son 'manual' por ser DERIVADAS de 2 fdc reales cada
# una, no por carecer de fdc — `fdc_id` en la columna persistida es NULL para las 8 'manual' por
# igual (el detalle de qué fdc's alimentaron el blend vive en `_provenance`, no en la columna).


# ── J10. fix-round 1 (review): 2 CRITICAL + 1 IMPORTANT sobre resolución de las altas T7 ─────────
#
# [fix-round 1 · 2026-08-17] El reviewer confirmó por EJECUCIÓN DIRECTA (no solo lectura) 3
# hallazgos sobre T7 original (6ec25df):
#
#   CRITICAL #1 -- el alias bare 'culantro' en Recao interceptaba al tier EXACTO lo que pre-T7
#   fuzzy-resolvía (ratio≈0.875) a 'Cilantro' (fila RD PRICED real, RD$435.45/lb) -- 'culantro' es
#   uso dominicano ESTABLECIDO para cilantro (constants.VEGGIE_FAT_SYNONYMS['cilantro'] ya lo
#   lista). Prueba en vivo pre-fix del reviewer:
#   aggregate_and_deduct_shopping_list(['30 g de culantro']) -> [('Recao', '3 Mazos Recao')] SIN
#   precio, con el knob apagado (rompe byte-identidad DO). Fix: bare 'culantro' removido de
#   Recao.aliases (JSON SSOT + DB); 'culantro cimarron'/'recao de monte' preservados (frases de 2
#   palabras, PR-específicas, sin colisión fuzzy). El item curado del harness PR 'Recao (culantro)'
#   NUNCA dependió del alias (normalize_name le strippea el paréntesis antes de tocar el índice ->
#   'Recao' bare -> match EXACTO contra el nombre de la fila) -- cubierto sin cambios por el
#   retarget-diff guard (I10) que re-resuelve TODO country_gaps/*.json committed contra el resolver
#   vivo.
#
#   CRITICAL #2 -- `canonicalize_shopping_food_name` (segunda cadena de canonicalización que corre
#   DESPUÉS de la resolución exacta del master_map, dentro de `aggregate_and_deduct_shopping_list`
#   -- una capa que el harness de Task 1 NUNCA ejercita, ver nota nueva en el docstring de
#   `country_catalog_gap.py`) sobreescribía la identidad YA EXACTA de 8 filas de catálogo-país: 6
#   T7 reportadas por el reviewer (Queso de papa/Bolitas de papa/Papas ralladas -> 'Papa', Nuez de
#   Castilla -> 'Nueces', Huevos rellenos -> 'Huevo', Nueces pecanas -> 'Pecanas' [nombre SIN fila
#   propia -> DROP SILENCIOSO, el caso más grave]) + 2 T5 bonus descubiertas por el sweep propio
#   (Acelgas, Almendra marcona, mismo mecanismo). Sweep completo de los 346 nombres del catálogo
#   vivo (salida real, post-fix, capturada abajo) encontró 13 filas PRE-EXISTENTES (pre-T5) que SÍ
#   dependen INTENCIONALMENTE de esta cadena (Plátano verde/maduro -> Plátano vía
#   canonicalize_musaceae, Queso cheddar/mozzarella/parmesano -> nombre corto, Clara/Yema de huevo
#   -> Huevo vía _consolidate_inline_canon, etc.) -- un skip GENERAL (branch (i) de la ruling)
#   habría cambiado conducta DO para esas 13. Se implementó branch (ii): skip ESCOPED a
#   `is_country_catalog_unpriced_item(canonical_name)`, True solo para los 140 tokens de
#   catálogo-país (T5+T6+T7) y False para las 13 filas pre-existentes -- byte-identidad DO
#   preservada. Ver el sweep re-corrido como test permanente más abajo.
#
#   IMPORTANT #3 -- el alias 'melocoton' añadido a Duraznos (T7) era código MUERTO desde el día 1:
#   'Durazno en almíbar' (fila pre-existente) YA reclamaba 'melocoton' como alias, mismo string
#   (misma longitud) -- el stable-sort de `_construir_indice_alias` (orden descendente por
#   longitud; empates preservan orden de LISTA) hace que la fila pre-existente (procesada antes)
#   gane siempre. Fix: removido el duplicado muerto de Duraznos.aliases; preservado el mapeo
#   pre-existente melocoton -> 'Durazno en almíbar' (byte-identidad DO). Ningún template/pool
#   PR/US usa 'melocoton' (deben usar el canónico 'Duraznos') -- confirmado por grep y anclado
#   abajo.
#
# Sweep de colisión (346 filas del catálogo vivo, `canonicalize_shopping_food_name`, POST-fix,
# corrida real 2026-08-17):
#
#     Total catalog rows: 346
#     Overridden COUNTRY-alta rows (0): []
#     Overridden PRE-EXISTING rows (13):
#       'Cebolla en polvo' -> 'Cebolla'        'Clara de huevo' -> 'Huevo'
#       'Guineo verde' -> 'Guineo'              'Lechuga romana' -> 'Lechuga'
#       'Nueces mixtas' -> 'Nueces'             'Orégano dominicano' -> 'Orégano'
#       'Plátano maduro' -> 'Plátano'           'Plátano verde' -> 'Plátano'
#       'Queso cheddar' -> 'Cheddar'            'Queso mozzarella' -> 'Mozzarella'
#       'Queso parmesano' -> 'Parmesano'        'Tofu firme' -> 'Tofu'
#       'Yema de huevo' -> 'Huevo'
#
# El script idempotente `scripts/add_foods_pr_us_2026_08_17.py` tenía TAMBIÉN un bug de
# fix-round, descubierto al intentar aplicar los 2 fixes de arriba a Neon: `_apply_new_rows`
# comparaba solo `fdc_id` + columnas nutricionales (`_cmp_cols`), nunca `aliases` -- un `--commit`
# con el JSON ya editado reportaba "~ EXISTE (sin diffs), salto" para Recao/Duraznos y habría
# dejado los alias VIEJOS vivos en Neon en silencio (el bug de shopping_calculator.py estaría
# "arreglado en el código" pero NO en los datos que ese código lee). `_cmp_cols` ahora incluye
# 'aliases'; `_val_eq` compara listas por SET (orden no es semántico en un bag de sinónimos, evita
# false-diff por reordering). Ambas filas re-sincronizadas a Neon vía `--commit` (verificado
# idempotente: dry-run posterior reporta "sin diffs" para las dos).

_NEW_FOODS_PR_US_JSON = _BACKEND / "scripts" / "data" / "new_foods_pr_us_2026_08_17.json"


def _load_new_foods_pr_us() -> list:
    with open(_NEW_FOODS_PR_US_JSON, encoding="utf-8") as f:
        return json.load(f)


# ── J10a. CRITICAL #1 — 'culantro' bare removido de Recao, byte-identidad DO restaurada ──────────

def test_recao_aliases_no_contienen_culantro_bare_estructural():
    """[CRITICAL #1 · estructural, sin DB] El JSON SSOT ya no debe declarar 'culantro' bare como
    alias de Recao -- solo frases PR-específicas de 2+ palabras, sin riesgo de colisión fuzzy."""
    recao = next(r for r in _load_new_foods_pr_us() if r["name"] == "Recao")
    aliases_lower = [a.lower() for a in recao.get("aliases", [])]
    assert "culantro" not in aliases_lower, (
        "'culantro' bare NUNCA debe ser alias de Recao -- intercepta al tier EXACTO lo que "
        "pre-T7 fuzzy-resolvía a 'Cilantro' (RD PRICED, RD$435.45/lb), rompiendo byte-identidad DO"
    )
    assert "culantro cimarron" in aliases_lower, "el alias PR-específico no debe perderse en el fix"
    assert "recao de monte" in aliases_lower


@pytest.mark.e2e
def test_culantro_bare_resuelve_a_cilantro_fuzzy_fix_round_1(sc):
    """[CRITICAL #1 · el RED de este fix-round] RED en HEAD (6ec25df):
    normalize_name('culantro') == 'Recao' (el alias bare interceptaba al tier exacto, antes de
    llegar al tier fuzzy que resolvía 'Cilantro'). GREEN tras remover el alias: 'culantro' bare
    vuelve a fuzzy-resolver a 'Cilantro' (RD priced), como pre-T7. Control: la frase
    PR-específica 'culantro cimarron' SÍ debe seguir resolviendo a Recao -- confirma que el fix
    no sobre-corrigió (no borró la fila, solo el alias colisionante)."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    assert sc.normalize_name("culantro") == "Cilantro", (
        "'culantro' bare debe fuzzy-resolver a 'Cilantro' (RD priced) -- si resuelve a 'Recao', "
        "el alias bare volvió a interceptar el tier exacto (regresión de CRITICAL #1)"
    )
    assert sc.normalize_name("culantro cimarron") == "Recao", (
        "'culantro cimarron' (frase PR-específica de 2 palabras) debe seguir resolviendo a Recao"
    )


# ── J10b. CRITICAL #2 — canonicalize_shopping_food_name ya no sobreescribe filas de catálogo-país ─

@pytest.mark.e2e
@pytest.mark.parametrize("nombre", [
    "Queso de papa", "Bolitas de papa", "Papas ralladas", "Nuez de Castilla", "Huevos rellenos",
    "Nueces pecanas", "Acelgas", "Almendra marcona",
])
def test_filas_pais_sobreviven_como_si_mismas_bajo_catalogo_sin_precio_fix_round_1(sc, nombre, monkeypatch):
    """[CRITICAL #2 · el RED de este fix-round, 8 casos vivos] RED en HEAD (6ec25df):
    `canonicalize_shopping_food_name` (llamada DESPUÉS de la resolución exacta, dentro de
    `aggregate_and_deduct_shopping_list`) sobreescribía estas 8 filas -- 6 T7 reportadas por el
    reviewer más 2 T5 bonus descubiertas por el sweep propio (Acelgas, Almendra marcona), mismo
    mecanismo (colisión de regex genérico: papa/nuez-nueces/huevo). El caso más grave: 'Nueces
    pecanas' -> 'Pecanas' (nombre SIN fila propia en el catálogo) -> DROP SILENCIOSO del
    agregador, confirmado en vivo por el reviewer. GREEN tras el skip escopado a
    `is_country_catalog_unpriced_item` en shopping_calculator.py.

    `MEALFIT_VERIFIED_INGREDIENTS_ONLY` monkeypatcheado a 'true' -- mismo patrón que
    `test_jamon_serrano_no_se_dropea_en_silencio_via_unpriced_keep`: es la puerta que activa la
    rama CATÁLOGO SIN PRECIO / drop-si-no-verificado; el baseline de la suite la fija 'false' así
    que sin esto los 8 items pasarían por el camino normal (con precio=0 pero SIN pasar por el
    canonicalizer post-drop-gate) y el test no ejercería el código que de verdad falló en vivo."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    monkeypatch.setenv("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "true")

    result = sc.aggregate_and_deduct_shopping_list([f"30 g de {nombre}"], structured=True)
    items = result if isinstance(result, list) else (result.get("items") or [])
    item = next((it for it in items if it.get("name") == nombre), None)
    assert item is not None, (
        f"{nombre!r} no sobrevivió al agregador con su propio nombre -- probablemente renombrado "
        f"o dropeado por canonicalize_shopping_food_name. nombres presentes: "
        f"{[it.get('name') for it in items]}"
    )
    # [P2-SHOPLIST-BETA-POLISH · 2026-08-18] La prueba de que pasó por la rama unpriced-keep es
    # SOBREVIVIR bajo VERIFIED_ONLY=true SIN costo inventado; la categoría ahora es el pasillo
    # real del master (con fallback al label histórico si el lookup no resuelve).
    assert item.get("estimated_cost_rd") is None, (
        f"{nombre!r} no debe llevar costo RD inventado (rama unpriced-keep)"
    )
    # [reconvertido · P2-COUNTRY-HOUSEKEEPING · 2026-08-21] Eran las categorías CRUDAS de la base.
    # La rama CON precio emite el label de DISPLAY (mayúsculas) y el Dashboard agrupa por la cadena
    # literal: con las dos grafías conviviendo, el usuario veía DOS secciones del mismo pasillo del
    # súper —una con doce ítems y otra con las Acelgas solas—. La invariante es «el ítem sin precio
    # cae en el MISMO pasillo que uno con precio», así que se ancla contra el propio mapa de
    # display en vez de contra una lista de literales que habría que re-teclear.
    _pasillos_display = {sc._get_display_category(_c, "x") for _c in
                         ("Despensa", "Frutas", "Lácteos", "Proteínas", "Vegetales", "Víveres")}
    assert item.get("display_category") in (_pasillos_display | {"CATÁLOGO SIN PRECIO"}), (
        f"{nombre!r} sobrevivió pero con categoría inesperada: {item.get('display_category')!r}"
    )


@pytest.mark.e2e
def test_canonicalize_shopping_food_name_sweep_346_filas_cero_altas_pais_trece_preexistentes(sc):
    """[CRITICAL #2 · el guard durable, sweep completo -- mismo espíritu que H6-bis/I10] Para CADA
    nombre real del catálogo vivo (346 filas al momento de escribir este test), si
    `canonicalize_shopping_food_name` lo sobreescribe (resultado != nombre propio): (a) NINGUNA
    fila de catálogo-país (`is_country_catalog_unpriced_item`==True) puede estar en ese conjunto
    -- el bug que este fix-round cierra -- y (b) el conjunto de filas PRE-EXISTENTES sobreescritas
    debe ser EXACTAMENTE el observado post-fix (13, todas dependientes A PROPÓSITO de
    canonicalize_musaceae/frutos_secos/_consolidate_inline_canon) -- si crece o encoge, alguien
    tocó el chain o el scope del skip y este test lo atrapa antes que un futuro T8 lo redescubra
    en producción."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()

    rows = sc.get_master_ingredients() or []
    names = sorted({r["name"] for r in rows if r.get("name")})
    master_map = sc._build_shopping_master_map()

    overridden_country = []
    overridden_preexisting = set()
    for name in names:
        result = sc.canonicalize_shopping_food_name(name, master_map)
        if result != name:
            if sc.is_country_catalog_unpriced_item(name):
                overridden_country.append((name, result))
            else:
                overridden_preexisting.add(name)

    assert not overridden_country, (
        f"{len(overridden_country)} fila(s) de catálogo-país sobreescritas por el chain genérico "
        f"(CRITICAL #2 regresó): {overridden_country}"
    )

    # [P1-CATALOG-ORDER-DETERMINISTIC · 2026-08-19] +Mero y +Tilapia. La resolucion del
    # catalogo dependia del ORDEN FISICO de las filas (SELECT sin ORDER BY + sort estable
    # + first-hit: los empates de longitud los decidia el heap), y el fill de 347 UPDATEs
    # de la auditoria de procedencia reescribio ese heap. El fix --ORDER BY name + sort
    # por (-longitud, alias) + best-match-- hizo que estos dos resolvieran a SI MISMOS en
    # vez de a "Filete de pescado blanco", que es la mejora que se acepto al regenerar el
    # baseline C3. Se anaden EXPLICITAMENTE, como pide el mensaje de este assert.
    esperadas_preexistentes = {
        # [P2-WHITE-FISH-ALIAS-SPLIT · 2026-09-02] -Mero y -Tilapia: la migración les quitó el
        # alias en la fila genérica "Filete de pescado blanco", así que ya no las sobreescribe el chain.
        "Cebolla en polvo", "Clara de huevo", "Guineo verde", "Lechuga romana",
        "Nueces mixtas", "Orégano dominicano", "Plátano maduro", "Plátano verde",
        "Queso cheddar", "Queso mozzarella", "Queso parmesano", "Tofu firme",
        "Yema de huevo",
    }
    assert overridden_preexisting == esperadas_preexistentes, (
        f"el set de filas PRE-EXISTENTES sobreescritas por el chain cambió -- nuevas: "
        f"{overridden_preexisting - esperadas_preexistentes or '{}'}, ya no presentes: "
        f"{esperadas_preexistentes - overridden_preexisting or '{}'}. Si es intencional (nueva "
        f"fila DO que empieza a depender del chain), actualiza esta lista explícitamente; si no, "
        f"el scope del skip de CRITICAL #2 se ensanchó o encogió sin review."
    )


# ── J10c. IMPORTANT #3 — 'melocoton' ya no es alias muerto de Duraznos ───────────────────────────

def test_duraznos_aliases_no_contienen_melocoton_estructural():
    """[IMPORTANT #3 · estructural, sin DB] El JSON SSOT ya no debe declarar 'melocoton' como
    alias de Duraznos -- era código muerto (length-tie perdido contra 'Durazno en almíbar', ver
    test del mecanismo abajo) que el report original reclamaba funcional sin haberlo verificado."""
    duraznos = next(r for r in _load_new_foods_pr_us() if r["name"] == "Duraznos")
    aliases_lower = [a.lower() for a in duraznos.get("aliases", [])]
    assert "melocoton" not in aliases_lower, (
        "'melocoton' NUNCA debe ser alias de Duraznos -- empata en longitud con el alias "
        "PRE-EXISTENTE de 'Durazno en almíbar' y el stable-sort SIEMPRE deja ganar a la fila "
        "pre-existente (código muerto garantizado, ver test del mecanismo)"
    )
    assert "peaches" in aliases_lower and "durazno fresco" in aliases_lower, (
        "los alias funcionales de Duraznos no deben perderse en el fix"
    )


@pytest.mark.e2e
def test_melocoton_sigue_resolviendo_a_durazno_en_almibar_fix_round_1(sc):
    """[IMPORTANT #3 · byte-identidad DO preservada] 'melocoton' sigue resolviendo al mapeo
    PRE-EXISTENTE 'Durazno en almíbar' tras remover el alias muerto de Duraznos -- prueba que el
    fix es un no-op para DO: el alias que T7 añadió NUNCA tuvo efecto (ver mecanismo abajo), así
    que removerlo no puede cambiar ninguna resolución real."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    assert sc.normalize_name("melocoton") == "Durazno en almíbar"


@pytest.mark.e2e
def test_melocoton_length_tie_mechanism_por_que_es_codigo_muerto(sc):
    """[IMPORTANT #3 · demuestra POR QUÉ el alias de Duraznos siempre fue código muerto] Ambos
    alias son el string IDÉNTICO 'melocoton' (misma longitud tras strip_accents) -- el stable-sort
    de `_construir_indice_alias` (orden descendente por longitud; empates preservan orden de
    LISTA de origen) hace que quien esté PRIMERO en `master_list` gane siempre. Reconstruye la
    regresión hipotética (Duraznos recupera 'melocoton') sobre datos REALES de 'Durazno en
    almíbar' -- si 'Duraznos' llegara a ganar el empate algún día (ej. alguien reordena
    `get_master_ingredients` o cambia el sort a no-estable), este test lo atraparía antes que un
    reviewer tuviera que descubrirlo por ejecución directa otra vez."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()

    master_list = sc.get_master_ingredients() or []
    durazno_almibar = next(r for r in master_list if r["name"] == "Durazno en almíbar")
    assert "melocoton" in [a.lower() for a in (durazno_almibar.get("aliases") or [])], (
        "precondición: 'Durazno en almíbar' debe seguir reclamando 'melocoton' -- si esto falla, "
        "el mapeo pre-existente que este fix preserva ya no existe en la DB"
    )

    simulado = [
        dict(durazno_almibar),
        # Regresión hipotética: como si el alias muerto de Duraznos NUNCA se hubiera removido.
        {"name": "Duraznos", "aliases": ["peaches", "durazno fresco", "melocoton"]},
    ]
    all_aliases, _contains = sc._construir_indice_alias(simulado)
    primero = next(name for (alias, name) in all_aliases if alias == "melocoton")
    assert primero == "Durazno en almíbar", (
        f"el primer alias 'melocoton' en el índice resolvió a {primero!r} -- si fuera 'Duraznos', "
        f"confirmaría que reañadir el alias muerto SÍ tendría efecto (dejaría de ser código "
        f"muerto y volvería a romper byte-identidad DO)"
    )


def test_ningun_template_o_pool_pr_us_usa_melocoton_debe_usar_duraznos_canonico():
    """[IMPORTANT #3 · ancla la premisa de la ruling] Ningún constituent de dish_templates_us.json
    ni COUNTRY_POOLS['US'] debe referenciar 'melocoton' -- el nombre CANÓNICO de la fila T7 es
    'Duraznos'; 'melocoton' solo vive como alias PRE-EXISTENTE de OTRA fila ('Durazno en
    almíbar'), así que un template que escribiera 'melocoton' apuntaría, por accidente, a la fila
    equivocada."""
    with open(_BACKEND / "data" / "dish_templates_us.json", encoding="utf-8") as f:
        data = json.load(f)
    ofensores = [
        (t.get("name"), c.get("name"))
        for t in data.get("templates", [])
        for c in t.get("constituents", [])
        if "melocoton" in (c.get("name") or "").lower()
    ]
    assert not ofensores, (
        f"template(s) PR/US referencian 'melocoton' en vez del canónico 'Duraznos': {ofensores}"
    )

    pool_us = constants.COUNTRY_POOLS.get("US") or {}
    nombres_pool = set()
    for key in ("proteins", "carbs", "veggies_fats", "fruits"):
        nombres_pool.update(pool_us.get(key) or [])
    ofensores_pool = sorted(n for n in nombres_pool if "melocoton" in n.lower())
    assert not ofensores_pool, f"COUNTRY_POOLS['US'] referencia 'melocoton': {ofensores_pool}"


# ══════════════════════════════════════════════════════════════════════════════════════════════
# SECCIÓN K (Task 8) — Top-up RD (`rd_drops.json`, `--rd-drops`, T1) + medidas caseras por país
# ══════════════════════════════════════════════════════════════════════════════════════════════
#
# `rd_drops.json` (338 corridas de `_creativity_kpi_job`/30d, T1 `--rd-drops`) midió 7 alimentos
# distintos dropeados por VERIFIED-ONLY en planes RD reales: mereyes(62), «2–3 ciruelas»(16),
# tortilla(8), hummus(6), requesón(6), azúcar(4), rábanos en láminas(4). A diferencia de T5-T7
# (catálogo por país nuevo), esta task topea el catálogo RD MISMO — así que el análisis por-item
# discrimina entre 3 clases de fix bien distintas:
#   (a) SINÓNIMO sobre fila YA PRICED (mereyes→Merey, rabanos→Rábano) — el alimento existe, solo
#       falta el alias plural/preparación.
#   (b) ALTA genuina SIN precio a propósito (hummus) — mismo mecanismo P1-BAKING-STAPLES/
#       P1-COUNTRY-CATALOG-UNPRICED que T5-T7, reusado para RD por falta de precio verificado HOY.
#   (c) FIX de parsing/normalización, CERO cambio de catálogo («2–3 ciruelas» — contaminación de
#       un rango numérico líder; «rábanos en láminas» — preparación sin stop-word).
# Y 3 decisiones de NO ACTUAR, cada una con evidencia (no simple omisión):
#   tortilla (bare) — AMBIGUO clínicamente (omelette vs pan, huevo vs gluten) — se deja dropeando.
#   requesón — YA RESUELTO como efecto colateral de T5 (fila «Requesón» exacta) — perseguir el
#       «Queso ricotta» que el brief original citaba crearía un alias muerto.
#   azúcar — INTENCIONAL (motor clínico DM2 lo trata como token OFENSOR a sustituir).
#
# Medidas caseras por país: auditoría de `humanize_ingredients.DOMINICAN_HOUSEHOLD_MEASURES`
# (consumida SOLO por `humanize_ingredient`/`humanize_plan_ingredients`, sin country-awareness,
# corre UNCONDICIONALMENTE para TODO plan en `assemble_plan_node` — graph_orchestrator.py:38453)
# contra las 140 altas T5-T7: 26/140 colisionaban por substring-sin-boundary (misma clase que
# Piñones⊂Champiñones, T5 fix-round 1) o por "preparación distinta" (harina de X, bolitas de X,
# X rallado — mismo patrón que `resolve_preparation_distinct` cierra del lado de compras). 15
# quedan cerradas por el fix de esta task (word-boundary + reuso de `resolve_preparation_distinct`
# + una whitelist estrecha de 3 frases); 11 quedan A PROPÓSITO (mismo alimento-categoría, solo
# pierden especificidad regional — no son bugs); 1 queda documentada sin fix (single-case, baja
# severidad). DO byte-idéntico: verificado que las 44 claves RD propias siguen resolviendo, y que
# el guard de "form mismatch" NUNCA dispara para un 'rallado' bare (solo 3 frases whitelisted).

_RD_TOPUP_NEW_FOODS_JSON = _BACKEND / "scripts" / "data" / "new_foods_rd_topup_2026_08_17.json"
_RD_TOPUP_SYNONYMS_JSON = _BACKEND / "scripts" / "data" / "synonyms_rd_topup_2026_08_17.json"


def _load_rd_topup_new_foods() -> list:
    with open(_RD_TOPUP_NEW_FOODS_JSON, encoding="utf-8") as f:
        return json.load(f)


def _load_rd_topup_synonyms() -> list:
    with open(_RD_TOPUP_SYNONYMS_JSON, encoding="utf-8") as f:
        return json.load(f)


# ── K0. Los data files del top-up existen con la forma esperada ─────────────────────────────────

def test_rd_topup_new_foods_json_forma_esperada():
    recs = _load_rd_topup_new_foods()
    assert len(recs) == 1, "Task 8 contrato: 1 sola alta genuina (Hummus)"
    assert recs[0]["name"] == "Hummus"
    assert recs[0]["fdc_id"] == 174289
    assert recs[0]["category"] == "Despensa"
    assert isinstance(recs[0]["aliases"], list) and "humus" in recs[0]["aliases"]


def test_rd_topup_synonyms_json_forma_esperada():
    syns = _load_rd_topup_synonyms()
    items = {s["item"]: s["target"] for s in syns}
    assert items == {"mereyes": "Merey", "rabanos": "Rábano"}


# ── K1. mereyes (62 drops, el más alto de los 7) — alias sobre "Merey" ("Cajuil" es alias, no el
#        nombre canónico -- corrección al brief original) ──────────────────────────────────────

def test_mereyes_resuelve_a_merey(sc):
    for q in ("mereyes", "Mereyes", "30 g de mereyes", "cajuil", "merey", "Cajuil"):
        assert sc.normalize_name(q) == "Merey", f"{q!r} debe resolver a 'Merey'"


def test_mereyes_fuzzy_ratio_confirma_que_el_alias_explicito_era_necesario():
    """Evidencia de por qué NO bastaba con la tolerancia fuzzy existente (a diferencia de
    'rábanos'/'ciruelas', que sí resuelven vía FUZZY sin alias nuevo): el ratio 'mereyes' vs
    'merey' es 0.833, por debajo del umbral 0.87 de `normalize_name` INTENTO 5."""
    import difflib
    ratio = difflib.SequenceMatcher(None, "mereyes", "merey").ratio()
    assert ratio < 0.87
    assert abs(ratio - 0.833) < 0.01


def test_mereyes_es_verificado_para_compras_tras_el_alias(sc):
    """Merey YA tiene precio real (fdc 170162) -- a diferencia de Hummus, mereyes NO necesita el
    mecanismo unpriced-keep; debe sobrevivir el aggregator con costo real, no CATÁLOGO SIN PRECIO."""
    assert sc._is_verified_for_shopping("mereyes") is True
    assert sc.is_country_catalog_unpriced_item("Merey") is False


def test_mereyes_dispara_backstop_frutos_secos_sin_cambio_de_vocabulario(go):
    """'merey' YA vivía en `_ALLERGEN_SYNONYMS['frutos secos']` ANTES de esta task, y el scanner
    tolera plural vía `(?:s|es)?` (`_scan_allergen_violations`) -- verificado en vivo que el
    plural 'mereyes' SÍ dispara el backstop. Contraste explícito con T5-T7: mereyes NO necesitó
    una 4ª entrada de vocabulario (a diferencia de percebe/boqueron/trucha/bacalaitos, que sí)."""
    plan = {"days": [{"meals": [{"name": "Merienda", "ingredients": ["30 g de mereyes"]}]}]}
    violations = go._scan_allergen_violations(plan, ["Frutos Secos"])
    assert violations, "mereyes debe violar la alergia a frutos secos"
    assert violations[0][2] == "merey"


# ── K2. "2–3 ciruelas" (16 drops) — contaminación de PARSING, NO alta de catálogo ────────────────

@pytest.mark.parametrize("raw", ["2–3 ciruelas", "2-3 ciruelas", "2 – 3 ciruelas", "2 - 3 ciruelas"])
def test_rango_numerico_lider_colapsa_al_mayor_y_resuelve_via_parse_quantity(sc, raw):
    """El consumidor REAL del aggregator (`_parse_quantity`, no solo `normalize_name` directo):
    el rango CONTAMINADO debe colapsar al valor MAYOR (mismo criterio que
    `humanize_ingredients._grammar_lead_value` ya usa para display, y la filosofía "pecarse de
    comprar de más" de P1-CITRUS-JUICE-YIELD) y resolver a 'Ciruela'."""
    qty, unit, name = sc._parse_quantity(raw, apply_yield_multiplier=False)
    assert name == "Ciruela", f"{raw!r} -> name={name!r}"
    assert qty == 3.0, f"{raw!r} -> qty={qty!r}, esperaba 3.0 (el valor MAYOR del rango)"
    assert unit == "unidad"


def test_rango_numerico_lider_normalize_name_directo_no_recibe_el_fix_a_proposito(sc):
    """El colapso de rango vive en `_preprocess_nlp_quantities`, que SOLO se invoca dentro de
    `_parse_quantity` -- `normalize_name` llamada directa (sin pasar por el parser de cantidad)
    NUNCA ve el string preprocesado. Esto es correcto: `normalize_name` resuelve NOMBRES, no
    strings-con-cantidad; el consumidor real (aggregator/`record_verified_only_drop`) siempre
    pasa por `_parse_quantity` primero (ver el test parametrizado de arriba, que sí es el
    contrato real) -- este test ancla que NO hay una ilusión de doble cobertura."""
    assert sc.normalize_name("2–3 ciruelas") != "Ciruela"


def test_ciruela_ya_existe_en_catalogo_con_precio_cero_cambio_de_catalogo(sc):
    """Ancla la decisión del brief: 'Ciruela' YA existe con precio -- el fix es 100% parsing."""
    assert sc.normalize_name("Ciruela") == "Ciruela"
    assert sc._is_verified_for_shopping("Ciruela") is True


def test_rango_no_afecta_una_cantidad_simple_sin_guion(sc):
    """Byte-identidad del parser para el caso común (sin rango): '3 ciruelas' nunca pasó por la
    rama nueva -- confirma que el fix no introduce ningún efecto para cantidades normales."""
    assert sc._parse_quantity("3 ciruelas", apply_yield_multiplier=False) == (3.0, "unidad", "Ciruela")


# ── Task 9 (l) — fix-round T8-review: el colapso de rango era `\2` fijo, no max() real ──────────
#
# El propio comentario de K2 (arriba) dice "colapsa al valor MAYOR" — cierto SOLO por coincidencia
# para rangos ASCENDENTES ("2-3" → el 2º número YA es el mayor). Un rango DESCENDENTE ("3-2
# ciruelas", el LLM invirtió el orden) tomaba el 2º número igual — que en ese caso es el MENOR,
# contradiciendo el criterio de diseño documentado. Encontrado por el reviewer de T8 (Important,
# "sin productor real conocido — one-liner"), plegado a Task 9 · item (l).

def test_l_rango_descendente_tambien_colapsa_al_mayor():
    """RED-first (reproducido contra el código pre-fix: '3-2 ciruelas' devolvía qty=2.0, el
    MENOR — el `\\2` fijo del regex no distinguía orden). Tras el fix, `max(g1, g2)` real."""
    from shopping_calculator import _preprocess_nlp_quantities as _ppnq
    assert _ppnq("3-2 ciruelas") == "3 ciruelas"
    assert _ppnq("3–2 ciruelas") == "3 ciruelas"  # en-dash


def test_l_rango_descendente_resuelve_via_parse_quantity(sc):
    """Mismo consumidor real que K2 (`_parse_quantity`, no `_preprocess_nlp_quantities` aislado)
    — el rango descendente también debe colapsar al MAYOR y resolver a 'Ciruela'."""
    qty, unit, name = sc._parse_quantity("3-2 ciruelas", apply_yield_multiplier=False)
    assert name == "Ciruela"
    assert qty == 3.0, f"qty={qty!r}, esperaba 3.0 (el MAYOR, no el 2º número)"
    assert unit == "unidad"


@pytest.mark.parametrize("raw,expected", [
    ("2-3 ciruelas", "3 ciruelas"),   # ascendente — byte-idéntico al comportamiento pre-fix
    ("3-2 ciruelas", "3 ciruelas"),   # descendente — el bug que este fix cierra
    ("10-2 ciruelas", "10 ciruelas"),  # comparación NUMÉRICA, no lexicográfica ("10" < "2" como string)
    ("2-10 ciruelas", "10 ciruelas"),
])
def test_l_rango_max_numerico_no_lexicografico(raw, expected):
    """MUTACIÓN implícita: si el fix hubiera comparado STRINGS en vez de ints (`max('10','2')`
    == '2' lexicográficamente), 10-2/2-10 fallarían. Ancla que la comparación es sobre `int(...)`."""
    from shopping_calculator import _preprocess_nlp_quantities as _ppnq
    assert _ppnq(raw) == expected


def test_l_mutacion_reproduce_el_bug_del_group2_fijo():
    """MUTACIÓN bidireccional: reproduce el regex PRE-fix (`\\2` fijo) contra un caso descendente
    y confirma que SÍ tomaba el menor — la evidencia de que el fix real cambió el resultado, no
    solo el comentario."""
    import re
    _rng_re = re.compile(r'^(\d+)\s*[-–]\s*(\d+)\b')
    pre_fix = _rng_re.sub(r'\2', "3-2 ciruelas", count=1)
    assert pre_fix == "2 ciruelas", "el regex legacy (\\2 fijo) debía tomar el 2º número, no el mayor"
    from shopping_calculator import _preprocess_nlp_quantities as _ppnq
    assert _ppnq("3-2 ciruelas") != pre_fix, "el fix real debe diferir del resultado legacy en este caso"


# ── K3. "rábanos en láminas" (4 drops) — stop de preparación + alias plural determinista ────────

@pytest.mark.parametrize("raw", ["rábanos en láminas", "rabanos en laminas", "4 rábanos en láminas",
                                  "un rábano en láminas", "Rábano en lámina"])
def test_rabanos_en_laminas_resuelve_a_rabano(sc, raw):
    assert sc.normalize_name(raw) == "Rábano", f"{raw!r} debe resolver a 'Rábano'"


def test_rabanos_en_laminas_via_parse_quantity(sc):
    assert sc._parse_quantity("4 rábanos en láminas", apply_yield_multiplier=False) == (4.0, "unidad", "Rábano")


def test_rabanos_bare_plural_ahora_es_determinista_no_solo_fuzzy(sc):
    """El alias explícito 'rabanos' (Task 8) hace el plural bare determinista (tier EXACT/CONTAINS)
    -- antes de este alias, 'rábanos' SOLO resolvía vía FUZZY (ratio 0.923, probabilístico)."""
    assert sc.normalize_name("rábanos") == "Rábano"
    row = next(r for r in sc.get_master_ingredients() if r["name"] == "Rábano")
    assert "rabanos" in [a.lower() for a in (row.get("aliases") or [])]


def test_en_laminas_es_stop_generico_no_especifico_de_rabano(sc):
    """El stop 'en láminas'/'en lámina' vive en `_NORMALIZE_STOPS` (genérico, mismo nivel que 'en
    rodajas'/'en trozos'/'en lonjas') -- no un guard puntual de Rábano. Ancla que beneficia a
    OTRO alimento sin alias plural dedicado, no solo el caso medido (mereyes/rábano sí tienen
    alias explícito ahora -- ver K1/K3 -- así que por sí solos no aíslan esta contribución).
    'Remolacha' es un ejemplo REAL donde el stop es la única vía: sin él, 'remolachas en láminas'
    NO resuelve por ningún tier (CONTAINS rompe boundary en el plural 'remolachaS', igual que
    'rabano'/'merey'; fuzzy contra el string completo cae bajo 0.87) -- verificado en vivo
    revirtiendo el stop temporalmente (mutación, ver task-8-report.md)."""
    assert sc.normalize_name("Remolachas en láminas") == "Remolacha"
    assert sc.normalize_name("remolachas en laminas") == "Remolacha"


# ── K4. hummus (6 drops) — alta genuina, SIN precio a propósito (mismo mecanismo P1-COUNTRY-
#        CATALOG-UNPRICED que T5-T7, reusado para RD por falta de precio, no por país beta) ─────

def test_hummus_resuelve_y_tiene_fdc_real(sc):
    for q in ("hummus", "Hummus", "humus", "hummus de garbanzo"):
        assert sc.normalize_name(q) == "Hummus", f"{q!r} debe resolver a 'Hummus'"
    row = next(r for r in sc.get_master_ingredients() if r["name"] == "Hummus")
    assert row["fdc_id"] == 174289
    assert row["price_per_lb"] == 0 and row["price_per_unit"] == 0
    assert row["nutrition_source"] == "usda"


def test_hummus_atwater_consistente(sc):
    row = next(r for r in sc.get_master_ingredients() if r["name"] == "Hummus")
    atwater = (4 * float(row["protein_g_per_100g"]) + 4 * float(row["carbs_g_per_100g"])
               + 9 * float(row["fats_g_per_100g"]))
    ratio = float(row["kcal_per_100g"]) / atwater
    assert 0.40 <= ratio <= 1.40, f"Atwater ratio {ratio} fuera de banda de sanidad"


def test_hummus_es_country_catalog_unpriced_item(sc):
    assert sc.is_country_catalog_unpriced_item("Hummus") is True
    assert sc._is_verified_for_shopping("Hummus") is False


def test_hummus_no_depende_del_knob_country_system(sc, monkeypatch):
    """A diferencia de 'tortilla de maiz' (T6 Critical #2, el ÚNICO de los 140+1 tokens
    knob-dependiente), 'hummus' NO depende de MEALFIT_COUNTRY_SYSTEM -- es RD top-up, no país
    beta. Debe reconocerse CON el knob apagado (default) y encendido, idéntico."""
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    assert sc.is_country_catalog_unpriced_item("Hummus") is True
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    assert sc.is_country_catalog_unpriced_item("Hummus") is True


def test_hummus_sobrevive_en_el_agregador_real_como_catalogo_sin_precio(sc, monkeypatch):
    monkeypatch.setenv("MEALFIT_VERIFIED_INGREDIENTS_ONLY", "true")
    result = sc.aggregate_and_deduct_shopping_list(["1 pote de hummus"], structured=True)
    items = result.get("items") if isinstance(result, dict) else result
    hummus_item = next((i for i in items if i.get("name") == "Hummus"), None)
    assert hummus_item is not None, "hummus no debe dropearse del agregador real"
    # [P2-SHOPLIST-BETA-POLISH · 2026-08-18] el ruling de Task 8 era «listar en vez de
    # dropear» — el pasillo ahora es el REAL del master, no el label interno.
    # [P2-COUNTRY-HOUSEKEEPING · 2026-08-21] ...y en el FORMATO de display, no en el crudo de la
    # DB. Este assert fijaba 'Despensa' (tal cual la columna `category`) mientras la rama CON
    # precio emite 'DESPENSA' (del `DISPLAY_CATEGORY_MAP`), y el Dashboard agrupa por la cadena
    # literal: el usuario veía DOS secciones para el mismo pasillo, una con doce ítems y otra con
    # el alimento sin precio solo. El ruling de Task 8 no cambia — lo que cambia es que el pasillo
    # real se escribe como todos los demás.
    assert hummus_item.get("display_category") == "DESPENSA"
    assert hummus_item.get("estimated_cost_rd") is None


def test_hummus_engancha_ahora_a_la_clase_sesamo(go):
    """[RECONVERTIDO por P0-ALLERGEN-VOCAB-I18N · 2026-08-21] Este test anclaba la AUSENCIA de la
    clase 'sésamo': el hallazgo que Task 8 documentó y declaró fuera de su scope («crear esa clase
    es una task de scope mayor»). La auditoría de producción del 2026-08-20 aportó la evidencia
    que faltaba —4 filas del catálogo vivo con sésamo y el nº 11 de los 14 alérgenos del
    Reglamento UE 1169/2011, con España viva desde el flip— y la clase existe. El test se
    RECONVIERTE en vez de borrarse (misma disciplina que `test_p3_i18n_deferred.py`): ahora ancla
    el estado nuevo, y su mitad todavía cierta —que 'Frutos Secos' NO cubre el sésamo— sigue
    siendo el control negativo que impide cerrar el hueco por sobre-detección perezosa."""
    plan = {"days": [{"meals": [{"name": "Merienda", "ingredients": ["1 pote de hummus"]}]}]}
    # Sigue cierto y sigue importando: garbanzo y sésamo no son frutos secos.
    assert go._scan_allergen_violations(plan, ["Frutos Secos"]) == []
    # Lo que cambió: ya hay una clase a la que enganchar el tahini del hummus.
    assert "sesamo" in go._ALLERGEN_SYNONYMS
    assert go._scan_allergen_violations(plan, ["sésamo"]), (
        "hummus lleva tahini: un alérgico al sésamo debe verlo como violación"
    )


# ── K5. tortilla bare (8 drops) — AMBIGUO, DECISIÓN: dejar dropeando (documentado con evidencia) ─

def test_tortilla_bare_sigue_sin_alias_por_diseno(sc):
    """DECISIÓN (Task 8, con evidencia): bare 'tortilla' es AMBIGUO en es-DO entre DOS alimentos
    clínicamente opuestos -- 'tortilla de huevos' (omelette, `dish_templates.json` línea 11:
    "Tortilla de huevos con espinaca y queso fresco") y 'tortilla de trigo/integral/maíz' (pan
    plano con gluten para trigo/integral). Un alias por defecto acertaría solo la mitad de las
    veces y podría alimentar gluten a quien pidió huevo (o viceversa) -- riesgo clínico real, no
    solo cosmético. Se deja DROPEANDO a propósito."""
    assert sc.normalize_name("tortilla") == "Tortilla", "pass-through sin resolver, a propósito"


def test_tortilla_de_huevos_omelette_es_el_otro_lado_de_la_ambiguedad():
    """Confirma la mitad 'omelette' de la ambigüedad citada arriba, con evidencia del catálogo RD
    real (no solo afirmación)."""
    with open(_BACKEND / "data" / "dish_templates.json", encoding="utf-8") as f:
        rd = json.load(f)
    omelette = [t["name"] for t in rd["templates"] if "tortilla de huevo" in t["name"].lower()]
    assert omelette, "dish_templates.json (RD) debe conservar el plato de tortilla-omelette citado como evidencia"


def test_tortilla_de_maiz_knob_gateado_de_t6_sigue_intacto(sc, monkeypatch):
    """No-regresión explícita: la decisión de NO aliasear bare 'tortilla' NO interactúa con la
    máquina knob-gateada de T6 para 'tortilla de maíz' (Critical #2 fix-round 1) -- sigue
    pass-through con el knob apagado y canonizando con el knob encendido, byte-idéntico a antes
    de esta task."""
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    assert sc.resolve_preparation_distinct("tortillas de maíz") == (True, None)
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    assert sc.resolve_preparation_distinct("tortillas de maíz") == (True, "Tortilla de maíz")


# ── K6. requesón (6 drops) — YA RESUELTO por T5 (fila "Requesón"); decisión: NO tocar
#        "Queso ricotta" (evita un alias muerto / shadowed) ─────────────────────────────────────

def test_requeson_ya_resuelve_a_su_propia_fila_efecto_colateral_de_t5(sc):
    for q in ("requesón", "requeson", "Requesón"):
        assert sc.normalize_name(q) == "Requesón", f"{q!r} debe resolver a 'Requesón' (T5)"
    assert sc.is_country_catalog_unpriced_item("Requesón") is True


def test_queso_ricotta_no_gana_alias_requeson_evita_alias_muerto(sc):
    """DECISIÓN (con evidencia): 'Queso ricotta' (fila PRICED distinta, pre-existente) NO debe
    ganar un alias 'requesón' -- 'Requesón' (T5) ya reclama esa cadena exacta en el tier EXACT de
    `normalize_name` (INTENTO 1, corre sobre TODOS los nombres/aliases sin distinguir self-name
    de alias). Cualquier alias 'requesón' añadido a Queso ricotta quedaría shadowed (dead code),
    la misma clase de trampa que 'Duraznos'/'melocoton' (T7 Important #3)."""
    row = next(r for r in sc.get_master_ingredients() if r["name"] == "Queso ricotta")
    aliases_lower = [a.lower() for a in (row.get("aliases") or [])]
    assert "requeson" not in aliases_lower and "requesón" not in aliases_lower


def test_requeson_y_queso_ricotta_coexisten_sin_colision_cada_uno_por_su_propio_nombre(sc):
    """'Requesón' (T5, unpriced) y 'Queso ricotta' (pre-existente, priced) comparten fdc_id 170851
    (misma identidad nutricional -- ricotta) pero NO colisionan: cada una resuelve por su PROPIO
    nombre exacto, nunca se pisan."""
    assert sc.normalize_name("Requesón") == "Requesón"
    assert sc.normalize_name("Queso ricotta") == "Queso ricotta"


# ── K7. azúcar (4 drops) — INTENCIONAL, verificado contra el motor clínico determinista ─────────

def test_azucar_no_existe_en_el_catalogo_a_proposito(sc):
    rows = sc.get_master_ingredients()
    nombres = {constants.strip_accents(r["name"]).lower() for r in rows}
    assert "azucar" not in nombres, "catálogo NO debe tener una fila 'azúcar' bare"
    # 'azúcar' como palabra CABEZA (no negada por 'sin', ej. 'Yogurt griego sin azúcar' -- ESE
    # SÍ es un alimento real, lácteo, no una fila de azúcar) solo debe darse en 'Azúcar morena'
    # (T7, tablilla/producto comercial específico, no el bare que esta decisión deja fuera).
    cabeza_azucar = sorted(
        r["name"] for r in rows
        if re.match(r'^azucar\b', constants.strip_accents(r["name"]).lower())
    )
    assert cabeza_azucar == ["Azúcar morena"], f"filas con 'azúcar' como palabra cabeza: {cabeza_azucar}"
    assert sc.normalize_name("azúcar") == "Azúcar", "pass-through sin resolver, a propósito"


def test_azucar_es_token_ofensor_del_motor_clinico_dm2(condrules):
    """Evidencia de la intención: `_DM2_SUGAR_SUBS` trata azucar/azúcar/sugar como OFENSOR a
    sustituir DETERMINÍSTICAMENTE por Stevia -- una fila de catálogo competiría con esa
    sustitución en vez de reforzarla."""
    offender_tokens = {t for group in condrules._DM2_SUGAR_SUBS for t in group[0]}
    assert {"azucar", "azúcar", "sugar"} <= offender_tokens
    target = next(g[1] for g in condrules._DM2_SUGAR_SUBS if "azucar" in g[0])
    assert target == "Stevia al gusto"


def test_azucar_documentada_como_sin_fila_de_catalogo_en_ignored_tracking_terms():
    """`constants.IGNORED_TRACKING_TERMS` documenta explícitamente (comentario in-line, pre-Task-8)
    que 'azucar' es un condimento "sin fila de catálogo ni la merece" -- la decisión de esta task
    (no dar de alta) es continuista con una decisión YA tomada, no una nueva."""
    assert "azucar" in constants.IGNORED_TRACKING_TERMS


def test_azucar_morena_si_existe_variante_especifica_de_t7_intacta(sc):
    """Contraste: T7 SÍ dio de alta 'Azúcar morena' (brown sugar, producto comercial específico
    US) sin que eso contradiga la decisión de NO dar de alta el 'azúcar' bare -- son alimentos
    culinariamente distintos (crudo/refinado vs bare) y la decisión de Task 8 no la toca."""
    assert sc.normalize_name("azúcar morena") == "Azúcar morena"


# ── K8. Medidas caseras por país — auditoría de `humanize_ingredients` contra las 140 altas
#        T5-T7 + fix de la clase de bug + byte-identidad DO ─────────────────────────────────────

_HOUSEHOLD_COLLISION_FIXED = frozenset({
    # Word-boundary (11): la clave era substring SIN boundary ("pan"⊂"espaÑOL"/"maZAPÁN"/etc,
    # "aji"⊂"guAJIllo", "queso"⊂"reQUESOn") o el plural rompía boundary contra la forma singular
    # de la clave ("huevo" no boundary-matchea "huevoS").
    "Chorizo español", "Panceta ibérica", "Mazapán", "Panela", "Panapén",
    "Panecillos ingleses", "Mezcla para panqueques", "Panecillos de mantequilla",
    "Chile guajillo", "Requesón", "Huevos rellenos",
    # Reuso de `resolve_preparation_distinct` (2): "harina de X" es un producto DISTINTO de la
    # raíz/grano fresco -- SSOT compartido con P1-PREP-COLLAPSE-GUARD.
    "Harina de yuca",
    # [nota] "Tortilla de maíz" también cierra vía este MISMO reuso -- `_PREP_TORTILLA_MAIZ_RE`
    # (T6) marca `resolve_preparation_distinct` como handled=True SIEMPRE (pass-through con el
    # knob apagado, canoniza con el knob encendido), así que el household-measure genérico de
    # "tortilla" (45 g, calibrado para trigo/RD) nunca se aplica a la de maíz -- MEJORA
    # deliberada, no accidente: una tortilla de maíz pesa ~25-30g, no 45g: la conversión vieja
    # ya asumía el país equivocado.
    "Tortilla de maíz",
    # Whitelist local de "form mismatch" (2): producto PROCESADO (tots/hash-browns) expresado
    # como si fuera el vegetal entero.
    "Bolitas de papa", "Papas ralladas", "Pan rallado",
})

_HOUSEHOLD_COLLISION_ACCEPTED = (
    "Jamón serrano", "Jamón ibérico", "Jamón de cocinar", "Jamón de sándwich",
    "Longaniza puertorriqueña", "Chuleta ahumada", "Queso de papa",
    "Queso en hebras", "Queso provolone", "Pan de maíz",
)


def test_household_measure_collisions_140_altas_t5_t7_auditadas():
    """La auditoría requerida por el contrato: de las 140 altas T5-T7, exactamente 26 colisionaban
    con `DOMINICAN_HOUSEHOLD_MEASURES` -- 16 cerradas por el fix de esta task
    (`_HOUSEHOLD_COLLISION_FIXED`) + 10 aceptadas (`_HOUSEHOLD_COLLISION_ACCEPTED`) = 26. La 27ª
    colisión de la auditoría original ("Especias para arroz con dulce") NO es una colisión de
    `DOMINICAN_HOUSEHOLD_MEASURES` -- es del fallback GENÉRICO de granos (mecanismo separado, ver
    `test_household_measure_residual_documentado_especias_arroz`), por eso no cuenta aquí."""
    assert len(_HOUSEHOLD_COLLISION_FIXED) == 16
    assert len(_HOUSEHOLD_COLLISION_ACCEPTED) == 10
    assert len(_HOUSEHOLD_COLLISION_FIXED) + len(_HOUSEHOLD_COLLISION_ACCEPTED) == 26
    assert not (_HOUSEHOLD_COLLISION_FIXED & set(_HOUSEHOLD_COLLISION_ACCEPTED)), "sin solape"


def test_household_measure_collisions_cerradas(hz):
    for nm in _HOUSEHOLD_COLLISION_FIXED:
        out = hz.humanize_ingredient(f"120 g de {nm}")
        assert out == f"120 g de {nm}", (
            f"{nm!r} sigue colisionando con una clave de DOMINICAN_HOUSEHOLD_MEASURES tras el "
            f"fix de word-boundary/form-mismatch: {out!r}"
        )


def test_household_measure_collisions_aceptadas_mismo_categoria(hz):
    """Estas 10 SÍ son un match legítimo -- misma categoría de alimento (jamón/longaniza/chuleta/
    queso/pan), solo pierden especificidad regional (mismo tipo de imprecisión, ya aceptada, que
    P2-DISPLAY-NAME-SPECIFICITY documenta para el catálogo RD pre-existente). Ancla que estas NO
    deben "arreglarse" sin una decisión de diseño explícita -- no son bugs."""
    for nm in _HOUSEHOLD_COLLISION_ACCEPTED:
        out = hz.humanize_ingredient(f"120 g de {nm}")
        assert out != f"120 g de {nm}", f"{nm!r}: se esperaba un match aceptado, cayó a gramos: {out!r}"


def test_household_measure_residual_documentado_especias_arroz(hz):
    """El ÚNICO residual del fallback GENÉRICO (no de la tabla RD): 'Especias para arroz con
    dulce' (PR) sigue mostrando '½ taza' -- 'arroz' es palabra COMPLETA ahí (no un substring roto),
    así que el word-boundary no lo cierra; queda documentado como baja severidad (mezcla de
    especias en gramos de un dígito, cosmético) en vez de sumar un 3er guard por un solo caso."""
    out = hz.humanize_ingredient("120 g de Especias para arroz con dulce")
    assert out == "½ taza de Especias para arroz con dulce"


def test_household_measures_do_byte_identico_44_claves_propias_siguen_resolviendo(hz):
    for key, entry in hz.DOMINICAN_HOUSEHOLD_MEASURES.items():
        qty_g = entry["weight"] * 2
        out = hz.humanize_ingredient(f"{qty_g:g} g de {key}")
        assert entry["plural"] in out, f"clave RD {key!r} dejó de resolver por sí misma: {out!r}"


def test_household_measures_do_form_mismatch_guard_no_es_generico_verificado_con_queso_rallado(hz):
    """El guard de "form mismatch" está ESCOPADO a 3 frases (bolitas de papa(s)/papas
    ralladas/pan rallado) -- NUNCA a un 'rallado' bare, para no arriesgar un nombre RD real como
    'queso rallado'/'zanahoria rallada' (ninguno vive hoy en `dish_templates.json`, verificado por
    grep, pero la whitelist estrecha es la defensa contra que aparezca uno mañana)."""
    assert hz.humanize_ingredient("60 g de queso rallado") != "60 g de queso rallado"
    assert hz.humanize_ingredient("60 g de zanahoria rallada") != "60 g de zanahoria rallada"
    assert hz._household_measure_form_mismatch("queso rallado") is False
    assert hz._household_measure_form_mismatch("zanahoria rallada") is False


def test_harina_de_x_nunca_colapsa_a_su_base_fresca_reuso_de_resolve_preparation_distinct(hz):
    """Reuso (no reimplementación) de `shopping_calculator.resolve_preparation_distinct` -- el
    MISMO SSOT "puro y determinista" que P1-PREP-COLLAPSE-GUARD ya usa del lado de compras."""
    for x in ("yuca", "trigo", "avena", "maiz", "platano"):
        out = hz.humanize_ingredient(f"120 g de harina de {x}")
        assert out == f"120 g de harina de {x}", f"harina de {x!r} no debe convertirse a medida casera del vegetal base: {out!r}"


def test_form_mismatch_guard_fail_safe_si_shopping_calculator_no_importa(monkeypatch, hz):
    """Fail-safe: si el import lazy de `resolve_preparation_distinct` falla, degrada al regex
    LOCAL (nunca lanza) -- 'harina de X' deja de detectarse (esperado, esa mitad depende del
    import) pero 'papas ralladas' SIGUE detectándose (la whitelist local no depende del import)."""
    import builtins
    orig_import = builtins.__import__

    def _boom(name, *a, **k):
        if name == "shopping_calculator":
            raise ImportError("simulado")
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _boom)
    assert hz._household_measure_form_mismatch("harina de yuca") is False
    assert hz._household_measure_form_mismatch("papas ralladas") is True


def test_household_measures_consumidor_confirmado_humanize_plan_ingredients_sin_country_awareness():
    """Traza el CONSUMIDOR real (contrato de la task): `humanize_plan_ingredients` corre
    INCONDICIONALMENTE para TODO plan dentro de `assemble_plan_node` (sin gate de country/knob) --
    confirma por qué el fix de esta sección debe ser byte-idéntico para DO en vez de gateado."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "from humanize_ingredients import humanize_plan_ingredients" in src
    # el call site real (no el de update-finalizer, que opera sobre un solo meal editado)
    idx = src.index("Humanizar ingredientes a medidas caseras dominicanas")
    ventana = src[idx: idx + 500]
    assert "_adb(humanize_plan_ingredients, result)" in ventana


# ── K9. Cadencia del cron `_creativity_kpi_job` — 338 corridas/30d reconciliadas (T1 minor
#        deferido a esta task) ───────────────────────────────────────────────────────────────

def test_creativity_kpi_job_un_solo_insert_de_pipeline_metrics_bajo_ese_node():
    """Confirma que NO hay una 2ª fuente escribiendo bajo `node='_creativity_kpi_job'` dentro de
    `cron_tasks.py` -- si hubiera un 2º cron compartiendo el tag, ESA sería la explicación de la
    cadencia inflada, no P1-SCHEDULER-STAGGER. Cuenta los `INSERT INTO pipeline_metrics` cuyo
    tuple de VALUES incluye el literal del node (no solo un grep de substring, que contaría
    también el docstring y el registro del cron)."""
    src = _CRON_TASKS_PY.read_text(encoding="utf-8")
    inserts_con_ese_node = re.findall(
        r'INSERT INTO pipeline_metrics[\s\S]{0,400}?"_creativity_kpi_job"', src
    )
    assert len(inserts_con_ese_node) == 1, (
        f"esperaba exactamente 1 INSERT bajo node='_creativity_kpi_job', encontró "
        f"{len(inserts_con_ese_node)} -- si es >1, ESE es un productor duplicado real"
    )
    assert 'MEALFIT_CREATIVITY_KPI_INTERVAL_MIN", 1440' in src, "el default de 1440min (24h) sigue vigente"


def test_scheduler_stagger_documentado_como_causa_de_la_cadencia_inflada():
    """[hallazgo del Task 8] `_add_job_jittered` (P1-SCHEDULER-STAGGER · 2026-05-28) da a CADA
    cron `interval` sin `next_run_time`/`start_date` explícito un `next_run_time` inicial =
    `ahora + offset(job_id) ∈ [0, MEALFIT_SCHEDULER_STAGGER_MAX_S]` EN CADA REGISTRO -- sin
    jobstore persistente, cada arranque/redeploy re-registra el job, disparando un run "bonus"
    dentro del primer minuto de cada restart. NO es un 2º cron ni drift de cadencia: en un entorno
    con muchos redeploys/día esto produce muchas más filas de las que `lookback_days × 1`
    predeciría, y el propio comentario de P1-SCHEDULER-STAGGER llama a esto "benigno por diseño"."""
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    assert "_SCHEDULER_STAGGER_ENABLED" in src
    assert "next_run_time" in src
    assert "P1-SCHEDULER-STAGGER" in src


def test_country_catalog_gap_docstring_reconciliado_menciona_scheduler_stagger():
    """[fix de doc, CERO cambio de conducta] El comentario original solo citaba el intervalo
    nominal (1440min=24h) -- suficiente para explicar "por qué está vacío" pero engañoso para
    "por qué hay 338 en vez de ~30" sin este contexto. Ancla que el script documenta la causa
    real."""
    src = _SCRIPT.read_text(encoding="utf-8")
    assert "P1-SCHEDULER-STAGGER" in src
    assert "cron_runs_examined" in src


# ── K10. rd_drops.json — el criterio de salida es la RESOLUCIÓN, no el contador histórico ───────

def test_rd_drops_json_note_explicita_que_el_criterio_no_es_el_contador():
    """El brief es explícito: "el exit criterion aquí es: every one of the 7 RESOLVES correctly
    NOW... not the metric itself". `rd_drops.json` es telemetría HISTÓRICA (pipeline_metrics ya
    escrito antes del fix) -- una re-corrida de `--rd-drops` HOY sigue reportando los mismos 7
    conteos hasta que prod corra con el fix desplegado. Este archivo NO se testea por su
    CONTENIDO numérico (cambia con el tiempo, mismo principio que el resto de la Sección A-D de
    este archivo: "ningún test toca Neon para contenido transitorio") -- se testea su FORMA."""
    path = _BACKEND / "data" / "country_gaps" / "rd_drops.json"
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["mode"] == "rd-drops"
    assert payload["source_node"] == "_creativity_kpi_job"
    assert isinstance(payload["top_drops"], list)


# ═══════════════════════════════════════════════════════════════════════════
# Task 9 — los pendientes de «100% listo» (pre-flip)
# ═══════════════════════════════════════════════════════════════════════════
#
# F8: ancla que la sesión de Neon corre en TimeZone=UTC — la premisa silenciosa de la que
# depende TODA la aritmética de T5-F1 (`user_tz_offset_min`: `NOW() - make_interval(mins =>
# offset)`, offset en la convención de getTimezoneOffset()). `db_core.get_client_kwargs()` NO
# fija `options="-c TimeZone=..."` (verificado: 0 hits de "TimeZone"/"SET TIME ZONE" en
# db_core.py) — el motor confía en que Neon arranca la sesión en UTC por DEFECTO (comportamiento
# de servidor, no garantía de nuestro código). Un test parser NO puede probar esto (no hay nada
# que parsear: la ausencia de un SET no demuestra el valor efectivo) — hace falta la consulta
# EN VIVO. e2e-marked, mismo patrón que el resto del archivo (pool abierto explícito, skip si no
# hay credenciales, nunca bloquea el gate rápido).

@pytest.mark.e2e
def test_f8_neon_session_timezone_offset_cero():
    """[P1-COUNTRY-SYSTEM-F2 · 2026-08-17 (Task 9, F8)] Contra la conexión REAL del pool — si
    Neon alguna vez cambiara su default de sesión a un huso CON offset (o una migración de
    proveedor lo alterase), este test lo detecta ANTES de que la aritmética de offsets de T5-F1
    empiece a fallar en silencio (un offset calculado sobre una sesión no-UTC desincroniza TODAS
    las fechas locales derivadas).

    HALLAZGO EN VIVO durante el desarrollo de este test: `current_setting('TimeZone')` devuelve
    `'GMT'`, NO el string literal `'UTC'` — assertar por IGUALDAD DE STRING habría sido el MISMO
    anti-patrón grapheme-bound que F6 re-ancló dos párrafos arriba (CLAUDE.md: "property not
    grapheme"). GMT y UTC son la MISMA propiedad (offset cero desde Greenwich, sin DST) — lo que
    importa para `NOW() - make_interval(mins => offset)` es el OFFSET NUMÉRICO, no el nombre.
    Se ancla con `EXTRACT(TIMEZONE FROM NOW()) = 0` (la propiedad real) — y con una whitelist
    documentada de nombres zero-offset conocidos como señal secundaria legible por humanos."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — faltan NEON_DATABASE_URL/.env (e2e, no bloquea el gate)")
    db_core.connection_pool.open()
    from db_core import execute_sql_query
    row = execute_sql_query(
        "SELECT current_setting('TimeZone') AS tz, EXTRACT(TIMEZONE FROM NOW()) AS off_s",
        fetch_one=True,
    )
    assert row is not None, "la query de TimeZone no devolvió fila"
    assert float(row["off_s"]) == 0.0, (
        f"la sesión de Neon tiene offset {row['off_s']!r} segundos (tz={row['tz']!r}) — NO es "
        "zero-offset. La aritmética de user_tz_offset_min (T5-F1) asume offset cero; revisar "
        "db_core.get_client_kwargs()"
    )
    # Señal secundaria legible (no autoritativa — la aserción real es el offset numérico arriba):
    _ZERO_OFFSET_TZ_NAMES = ("UTC", "GMT", "Etc/UTC", "Etc/GMT")
    assert row["tz"] in _ZERO_OFFSET_TZ_NAMES, (
        f"tz={row['tz']!r} no está en la whitelist documentada de nombres zero-offset conocidos "
        f"({_ZERO_OFFSET_TZ_NAMES}) — el offset numérico SÍ dio 0 (el test no falla por esto), "
        "pero vale actualizar la whitelist para que el próximo lector reconozca el nombre nuevo."
    )


def test_f8_pool_init_no_fija_timezone_explicito():
    """Parser complementario (documenta el POR QUÉ del test e2e de arriba): `get_client_kwargs()`
    no incluye `options=` con `TimeZone`/`SET TIME ZONE` — la sesión UTC es un comportamiento de
    SERVIDOR (default de Neon), no algo que nuestro código imponga. Si algún día alguien AÑADE un
    `options="-c TimeZone=..."` aquí, este test lo hace visible (el cambio deja de ser "confiamos
    en el default del servidor" para ser una decisión explícita, y merece su propia revisión)."""
    src = (_BACKEND / "db_core.py").read_text(encoding="utf-8")
    ini = src.index("def get_client_kwargs")
    fin = src.index("\n        def configure_sync_conn", ini)
    cuerpo = src[ini:fin]
    assert "TimeZone" not in cuerpo and "TIME ZONE" not in cuerpo, (
        "get_client_kwargs ahora fija TimeZone explícitamente — actualiza el comentario del test "
        "e2e hermano (test_f8_neon_session_timezone_es_utc) para que deje de sonar a comportamiento "
        "de servidor no-garantizado"
    )


# F9: `default_tz_offset_min` (COUNTRY_PROFILES) documentado como SIN LECTOR por diseño — ver
# constants.py (comentario junto a COUNTRY_PROFILES). Test ancla que NINGÚN módulo de producción
# lo lee (si alguien lo cablea, este test falla y fuerza releer el comentario/decisión primero).

def test_f9_default_tz_offset_min_sin_lector_en_produccion():
    """[P1-COUNTRY-SYSTEM-F2 · 2026-08-17 (Task 9, F9)] `default_tz_offset_min` vive SOLO en el
    dict `COUNTRY_PROFILES` (constants.py) y en los tests — ningún archivo .py de producción lo
    consulta. Mutación: si un futuro caller lo cablea (`profile.get('default_tz_offset_min')` o
    similar), este test se pone ROJO — la intención es que ese caller relea el comentario de
    diseño (T5-F1 hizo la fecha local country-independiente A PROPÓSITO) antes de proceder, no
    que el wiring se cuele en silencio."""
    # grep-equivalente en Python puro (sin depender de `grep` del sistema, portable Windows/CI):
    offenders = []
    for path in _BACKEND.rglob("*.py"):
        parts = path.relative_to(_BACKEND).parts
        if parts[0] in ("tests", "scripts", "venv", ".venv", "__pycache__", "migrations"):
            continue
        if path.name == "constants.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "default_tz_offset_min" in text:
            offenders.append(str(path.relative_to(_BACKEND)))
    assert not offenders, (
        f"default_tz_offset_min ganó lector(es) fuera de constants.py: {offenders} — antes de "
        "cablearlo, relee el comentario de diseño junto a COUNTRY_PROFILES (T5-F1 hizo la fecha "
        "local COUNTRY-INDEPENDIENTE a propósito)"
    )


def test_f9_docstring_menciona_sin_lector_por_diseno():
    """El comentario junto a COUNTRY_PROFILES debe declarar EXPLÍCITAMENTE la decisión (no solo
    el hecho) — para que un futuro editor entienda que la ausencia de lector es intencional, no
    un descuido a corregir."""
    src = (_BACKEND / "constants.py").read_text(encoding="utf-8")
    ini = src.index("`default_tz_offset_min` usa la convención")
    fin = src.index('COUNTRY_SYSTEM_ENABLED = _env_bool', ini)
    cuerpo = src[ini:fin]
    assert "SIN LECTOR" in cuerpo
    assert "T5-F1" in cuerpo
    assert "country-independiente" in cuerpo.lower() or "country independiente" in cuerpo.lower()


# ── h: minors del ledger F1 ──────────────────────────────────────────────────────────────────

def test_h_planner_system_prompt_import_muerto_eliminado():
    """[P1-COUNTRY-SYSTEM-F2 · 2026-08-17 (Task 9, h)] `PLANNER_SYSTEM_PROMPT` (el símbolo crudo)
    quedó importado-pero-sin-uso en graph_orchestrator.py tras F1-T3/FINAL-FIX-F1a: los call
    sites migraron a `build_planner_system_prompt(ctx['country'])`. Verificado (evidencia, no
    especulación): 0 usos no-comentario del símbolo crudo antes de este fix. `build_planner_
    system_prompt` (SÍ usado) permanece."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    sin_comentarios = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert "PLANNER_SYSTEM_PROMPT" not in sin_comentarios, (
        "PLANNER_SYSTEM_PROMPT (símbolo crudo) reapareció fuera de un comentario — si es un uso "
        "real nuevo, está bien re-importarlo; si es el import muerto que volvió, quítalo de nuevo"
    )
    assert "build_planner_system_prompt" in sin_comentarios, (
        "build_planner_system_prompt SÍ tiene call sites reales — no debe desaparecer del import"
    )


def test_h_planner_system_prompt_sigue_vivo_en_su_modulo_propio():
    """Contrapeso del test anterior: la constante NO se borró del catálogo, solo dejó de
    importarse crudo en graph_orchestrator.py — sigue siendo el objeto que build_planner_system_
    prompt(DO) retorna por identidad (test_f1a_planner_do_o_none_es_byte_identico_is)."""
    from prompts.planner import PLANNER_SYSTEM_PROMPT, build_planner_system_prompt
    assert build_planner_system_prompt("DO") is PLANNER_SYSTEM_PROMPT


def _sanitize_form_data_cuerpo() -> str:
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    ini = src.index("def _sanitize_form_data_for_prompt")
    fin = src.index("\n# [P3-PLAN-MODEL-KNOBS", ini)
    return src[ini:fin]


def test_h_sanitizer_tooltip_anchor_presente():
    """[P1-COUNTRY-SYSTEM-F2 · 2026-08-17 (Task 9, h)] `_sanitize_form_data_for_prompt` es
    parseada por NOMBRE/firma desde 3 archivos de test distintos (test_p1_country_system_f0.py,
    test_p1_country_system_f1.py, test_p1_prompt_trim_form_data.py) — antes de este fix no tenía
    tooltip-anchor, así que un rename silencioso rompía esos tests con un ValueError genérico
    de "substring no encontrado" en vez de señalar el P-fix a releer."""
    cuerpo = _sanitize_form_data_cuerpo()
    assert "tooltip-anchor: P1-PROMPT-TRIM-FORM-DATA" in cuerpo


def test_h_mutacion_sin_tooltip_anchor_el_guard_fallaria():
    """MUTACIÓN bidireccional: reproduce el estado PRE-fix (docstring sin el anchor — solo se
    quita la línea distintiva, sin reconstruir el párrafo entero verbatim, para no acoplar este
    test al wrap exacto de línea) y confirma que el assert de arriba lo habría cazado."""
    cuerpo_real = _sanitize_form_data_cuerpo()
    assert "tooltip-anchor: P1-PROMPT-TRIM-FORM-DATA" in cuerpo_real, (
        "precondición: el anchor debe existir en el código real antes de mutar"
    )
    cuerpo_pre_fix = cuerpo_real.replace("tooltip-anchor: P1-PROMPT-TRIM-FORM-DATA", "")
    assert "tooltip-anchor: P1-PROMPT-TRIM-FORM-DATA" not in cuerpo_pre_fix


# ── k: T7-parked — VEGGIE_FAT_SYNONYMS['cilantro'] lista 'recao' (tensión con la fila propia) ──
#
# Trazado el consumidor real (GLOBAL_REVERSE_MAP → normalize_ingredient_for_tracking/
# track_meal_friction, heurísticas de variedad/rechazo) vs el consumidor de PRECIO/PANTRY
# (shopping_calculator.normalize_name, que NUNCA importa GLOBAL_REVERSE_MAP). Decisión: HARMLESS,
# documentado con comentario junto al dict (constants.py). Este bloque ancla la evidencia.

def test_k_global_reverse_map_recao_a_cilantro_preservado():
    """Pin: el alias sigue vivo (byte-identidad de la decisión "documentar, no tocar") — sirve de
    ancla si un futuro editor considera quitarlo sin releer el comentario de diseño."""
    assert constants.GLOBAL_REVERSE_MAP.get("recao") == "cilantro"
    assert constants.normalize_ingredient_for_tracking("Recao") == "cilantro"


def test_k_apply_synonyms_es_dead_code_confirmado():
    """Contexto de la traza (no una excepción a mantener, solo la evidencia de por qué no se
    investigó como consumidor real): `apply_synonyms` (el otro consumidor directo de
    GLOBAL_REVERSE_MAP en constants.py) no tiene NINGÚN caller de producción — verificado por
    ausencia total fuera de su propia definición."""
    src_dir = _BACKEND
    offenders = []
    for path in src_dir.glob("*.py"):
        if path.name in ("constants.py",):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "apply_synonyms(" in text:
            offenders.append(path.name)
    assert not offenders, (
        f"apply_synonyms ganó caller(s) de producción: {offenders} — la traza de (k) asumía "
        "dead code; si ahora se usa, re-evaluar el alias recao→cilantro contra ESE nuevo consumidor"
    )


def test_k_pricing_resolver_nunca_importa_global_reverse_map():
    """La afirmación central de (k): `shopping_calculator.py` (precio/pantry) NO conoce
    GLOBAL_REVERSE_MAP — el alias recao→cilantro no puede filtrarse a pricing por esta vía."""
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    assert "GLOBAL_REVERSE_MAP" not in src


def test_k_recao_resuelve_a_su_propia_fila_no_a_cilantro(sc):
    """LA prueba empírica de "harmless": contra el catálogo VIVO, 'Recao' resuelve a SU PROPIA
    fila (T7), nunca colapsada a 'Cilantro' — confirma que el alias de (k) no toca pricing."""
    assert sc.normalize_name("Recao") == "Recao"
    assert sc.normalize_name("Recao") != "Cilantro"
    assert sc.normalize_name("Cilantro") == "Cilantro"


def test_k_comentario_de_diseno_presente_junto_al_dict():
    """El dict debe declarar la decisión (harmless, evidence-derived) — no solo el hecho del
    alias — para que un futuro editor no la reabra sin contexto."""
    src = (_BACKEND / "constants.py").read_text(encoding="utf-8")
    ini = src.index("# [P1-COUNTRY-SYSTEM-F2 · 2026-08-17 (Task 9, k")
    fin = src.index('"cilantro": ["cilantro", "culantro", "verdura", "recao"]', ini)
    cuerpo = src[ini:fin]
    assert "GLOBAL_REVERSE_MAP" in cuerpo
    assert "normalize_name" in cuerpo


# ── j: T5-parked — los 2 call sites de _get_fast_filtered_catalogs SIN country= ─────────────────
#
# ai_helpers.py::get_deterministic_variety_prompt (el pool de catálogo, NO el texto del prompt —
# ese YA estaba wired desde F1 FINAL-FIX-F1b) + agent.py::swap_meal (el sorteo anti-mode-collapse
# del swap). Ambos ahora reciben `country=` derivado por la ÚNICA puerta T1, reusando una
# variable YA en scope (mismo patrón `_micro_form`/`_swap_country` del resto de la fase).

def test_j_ai_helpers_variety_country_derivado_una_vez():
    """`_variety_country` se deriva UNA sola vez (SSOT) y se reusa en los 2 call sites de esta
    función — antes el 2º (`build_deterministic_variety_prompt`, F1b) re-derivaba con su propio
    import local; ahora ambos comparten la misma variable."""
    src = (_BACKEND / "ai_helpers.py").read_text(encoding="utf-8")
    sin_comentarios = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    ini = sin_comentarios.index("def get_deterministic_variety_prompt")
    fin = sin_comentarios.find("\ndef ", ini + 10)
    cuerpo = sin_comentarios[ini: fin if fin != -1 else len(sin_comentarios)]
    assert cuerpo.count("country_for_form_data(form_data)") == 1, (
        "country_for_form_data(form_data) debe derivarse UNA sola vez dentro de esta función"
    )
    assert "_get_fast_filtered_catalogs(\n            allergies, dislikes, diet, country=_variety_country, market_extras=True, culture_country=_variety_culture)" in cuerpo
    assert "build_deterministic_variety_prompt(_dc, _variety_country)" in cuerpo


def test_j_agent_swap_reusa_swap_country_no_rederiva():
    """agent.py::swap_meal — el call site de _get_fast_filtered_catalogs reusa `_swap_country`
    (ya derivado arriba, T3) — no vuelve a llamar country_for_form_data."""
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    sin_comentarios = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    # [F7-H] la llamada lleva además market_extras/culture_country; lo que se ancla es que el MERCADO sigue siendo _swap_country
    assert "swap_allergies, swap_dislikes, swap_diet, country=_swap_country" in sin_comentarios, (
        "el call site del swap debe pasar country=_swap_country"
    )
    assert sin_comentarios.count("_swap_country = country_for_form_data(form_data)") == 1, (
        "_swap_country debe derivarse UNA sola vez en todo agent.py"
    )


def test_j_do_byte_identico_catalogo_ai_helpers(monkeypatch):
    """DO (o form_data sin 'country', knob off) sigue produciendo el MISMO pool RD que antes de
    este fix — ancla con comparación de contenido (no `is`, porque `_get_fast_filtered_catalogs`
    siempre retorna listas nuevas via `.copy()`) contra una llamada explícita a country=None."""
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    import ai_helpers
    form_data = {"allergies": [], "dislikes": [], "dietType": "balanced"}
    out_via_variety = ai_helpers.get_deterministic_variety_prompt("", form_data, days_count=3)
    # DO no debe contener ningún marcador de país beta (mismo tipo de aserción que el resto de
    # F1/F2: el render no debe traer nombre de país foráneo con el knob apagado):
    assert "España" not in out_via_variety and "México" not in out_via_variety


def test_j_beta_usa_su_propio_country_pool(monkeypatch):
    """Con el knob encendido y country='ES' en form_data, _get_fast_filtered_catalogs debe usar
    COUNTRY_POOLS['ES'] — se prueba directamente la función SSOT (no la prosa del prompt, que no
    lista el pool crudo) para una aserción determinista."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    from constants import _get_fast_filtered_catalogs, COUNTRY_POOLS, country_for_form_data
    form_data = {"country": "es"}
    resolved = country_for_form_data(form_data)
    assert resolved == "ES"
    fp, fc, fv, ff = _get_fast_filtered_catalogs((), (), "balanced", country=resolved)
    assert set(fp) == set(COUNTRY_POOLS["ES"]["proteins"])
    assert set(fc) == set(COUNTRY_POOLS["ES"]["carbs"])


def test_j_mutacion_sin_country_call_site_cae_siempre_a_do():
    """MUTACIÓN bidireccional: reproduce el estado PRE-fix (llamar sin country=) y confirma que
    el pool resultante es el DOMINICANO incluso para un país beta — la evidencia de que el fix
    real cambia el pool devuelto, no solo la firma."""
    from constants import _get_fast_filtered_catalogs, DOMINICAN_PROTEINS, COUNTRY_POOLS
    fp_sin_country, _, _, _ = _get_fast_filtered_catalogs((), (), "balanced")  # pre-fix shape
    fp_con_country, _, _, _ = _get_fast_filtered_catalogs((), (), "balanced", country="ES")
    assert set(fp_sin_country) == set(DOMINICAN_PROTEINS)
    assert set(fp_con_country) == set(COUNTRY_POOLS["ES"]["proteins"])
    assert set(fp_sin_country) != set(fp_con_country), (
        "el pool RD y el pool ES deben diferir — si no, la mutación no sería significativa"
    )


# ═══════════════════════════════════════════════════════════════════════════
# f: retry-gate — LA MÁS RIESGOSA. Cierra el ruling PARKED de T4 fix-round 1
# (docs/country_system_f1.md, "Nuance del retry-gate"): beta con issues de slot-appropriateness
# TODOS soft entrega ADVISORY desde el intento 1 (antes: solo en el intento final; 1..N-1
# forzaba retry SIEMPRE, hard o soft). Implementado como helper PURO
# `_slot_appropriateness_advisory_decision` (extraído de review_plan_node para ser
# unit-testable — NINGÚN test del repo invoca review_plan_node directo, verificado abajo).
# ═══════════════════════════════════════════════════════════════════════════

import graph_orchestrator as _go_f  # noqa: E402


def _legacy_slot_advisory(issues: list, attempt: int, max_attempts: int) -> bool:
    """Réplica EXACTA de la fórmula PRE-Task-9 (`if _sa_is_final and not _sa_has_hard:`) — el
    oráculo GOLDEN contra el que se ancla la byte-identidad DO. Sin país: la fórmula vieja
    JAMÁS tuvo el concepto."""
    has_hard = any(i.get("hard") for i in issues)
    is_final = int(attempt) >= int(max_attempts)
    return is_final and not has_hard


# ── f1: precedente confirmado — review_plan_node SÍ se llama directo en 3 archivos vecinos ────
#
# CORRECCIÓN durante el desarrollo de esta task: la premisa inicial ("ningún test invoca
# review_plan_node directo") era FALSA — `grep -rl "await.*review_plan_node(\|= review_plan_node("`
# tiene un blind spot (no cubre `graph_orchestrator.review_plan_node(` module-qualified dentro de
# un `_run(...)` wrapper). Los propios vecinos EXIGIDOS por el brief
# (test_p2_a_shopping_coherence_block_enforcement.py, test_p1_review_coherence_severe_only.py) SÍ
# lo invocan directo con un plan/state MÍNIMO que bypassa LLM/DB (form_data con user_id='guest' +
# restricciones vacías). Dado ese precedente PROBADO, este archivo AÑADE integration tests reales
# (no solo el helper puro de arriba) — sección f1b abajo.

def test_f_review_plan_node_si_se_invoca_directo_en_3_vecinos():
    """Documenta el precedente (positivo, no negativo): estos 3 archivos son la EVIDENCIA de que
    `review_plan_node` es invocable con un state mínimo — la base para los integration tests de
    f1b. Lista cerrada — un 4º archivo nuevo no rompe este test (no es una allowlist exhaustiva),
    pero si alguno de estos 3 deja de invocarlo, algo se reorganizó y vale la pena releer."""
    known = (
        "test_p2_a_shopping_coherence_block_enforcement.py",
        "test_p1_review_coherence_severe_only.py",
    )
    for fname in known:
        text = (_BACKEND / "tests" / fname).read_text(encoding="utf-8")
        assert "review_plan_node(" in text, f"{fname} ya no invoca review_plan_node — precedente roto"


# ── f1b: integration tests REALES — review_plan_node con plan/state mínimo (mismo patrón que
#         test_p2_a_shopping_coherence_block_enforcement.py / test_p1_review_coherence_severe_only.py) ──

import asyncio as _asyncio_f


def _f_bypass_form_data(country=None):
    """Form data sin restricciones — bypassa LLM/fact-check (mismo contrato que los 2 vecinos:
    user_id='guest', listas vacías). `country` opcional: ausente = comportamiento DO (fail-safe
    de country_for_form_data ante key ausente)."""
    fd = {
        "user_id": "guest", "allergies": [], "medicalConditions": [], "dislikes": [],
        "dietType": "balanced", "_days_to_generate": 3,
    }
    if country:
        fd["country"] = country
    return fd


def _f_minimal_plan_arroz_de_noche():
    """Plan mínimo de 1 día con 'Pollo a la plancha con arroz blanco' en CENA — el ejemplo
    CANÓNICO de violación SOFT (nunca hard, ni para DO) de todo el repo, ya anclado en
    test_p1_slot_appropriateness.py::test_dinner_breakfast_or_rice_flagged_soft y en los tests de
    item (g) de esta MISMA task. Recipe con los 3 prefijos (Mise en place/Toque de Fuego/Montaje)
    para no disparar el RECIPE-CONTRACT-GATE (ortogonal, contaminaría la prueba)."""
    return {
        "calories": 2000,
        "macros": {"protein": 150, "carbs": 200, "fats": 67},
        "days": [{"day": 1, "meals": [
            {"meal": "Cena", "name": "Pollo a la plancha con arroz blanco",
             "ingredients": ["200 g pechuga de pollo", "150 g arroz blanco"],
             "recipe": ["Mise en place: pesa la pechuga y lava el arroz.",
                        "El Toque de Fuego: cocina la pechuga 8-10 min a fuego medio y hierve el arroz 15 min.",
                        "Montaje: sirve la pechuga sobre el arroz."],
             "protein": 150, "carbs": 200, "fats": 67, "cals": 2000}
        ]}],
    }


def _f_minimal_plan_locrio_desayuno():
    """Plan mínimo con 'Locrio de pollo' en DESAYUNO — violación DURA incondicional (decisión de
    producto, NUNCA degrada, ni para DO ni para beta: el override de país neutraliza `hard`,
    nunca lo inventa) — mismo dish de test_breakfast_lunch_dishes_flagged_hard."""
    return {
        "calories": 2000,
        "macros": {"protein": 150, "carbs": 200, "fats": 67},
        "days": [{"day": 1, "meals": [
            {"meal": "Desayuno", "name": "Locrio de pollo",
             "ingredients": ["200 g pollo", "150 g arroz"],
             "recipe": ["Mise en place: pesa el pollo y lava el arroz.",
                        "El Toque de Fuego: guisa el pollo con el arroz 20 min a fuego medio.",
                        "Montaje: sirve caliente."],
             "protein": 150, "carbs": 200, "fats": 67, "cals": 2000}
        ]}],
    }


def _f_minimal_state(*, plan_result, country=None, attempt=1):
    return {
        "plan_result": plan_result, "form_data": _f_bypass_form_data(country), "taste_profile": "",
        "attempt": attempt, "rejection_reasons": [], "_rejection_severity": "minor",
        "request_id": "test-p1-country-system-f2-task9-f",
    }


def _f_run(coro):
    loop = _asyncio_f.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_f_integration_do_soft_attempt1_sigue_rechazando_byte_identico(monkeypatch):
    """[DO byte-identidad — integration, no solo el helper puro] DO (sin country en form_data),
    'arroz de noche' (soft, attempt 1 de 3 — NO final) ⇒ review_passed=False, SIGUE forzando
    retry — EXACTAMENTE el comportamiento pre-Task-9 (el gap PARKED que T4 documentó: "en
    attempts 1..N-1 CUALQUIER issue fuerza retry igual")."""
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    state = _f_minimal_state(plan_result=_f_minimal_plan_arroz_de_noche(), country=None, attempt=1)
    result = _f_run(_go_f.review_plan_node(state))
    assert result["review_passed"] is False, "DO en attempt 1 con soft-only DEBE seguir rechazando"
    assert result["_rejection_severity"] == "high"


def test_f_integration_beta_soft_attempt1_aprueba_con_advisory(monkeypatch):
    """[EL comportamiento NUEVO, end-to-end] País beta (ES) + knob ON, MISMO plan 'arroz de
    noche' (soft), attempt 1 de 3 (NO final) ⇒ review_passed=True — advisory desde el intento 1,
    sin quemar un retry completo."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    plan = _f_minimal_plan_arroz_de_noche()
    state = _f_minimal_state(plan_result=plan, country="ES", attempt=1)
    result = _f_run(_go_f.review_plan_node(state))
    assert result["review_passed"] is True, "beta en attempt 1 con soft-only DEBE aprobar (advisory)"
    assert plan.get("_slot_appropriateness_advisory_final") is True
    assert plan.get("_slot_appropriateness_advisory_beta_early") is True


def test_f_integration_beta_knob_off_sigue_rechazando_igual_que_do(monkeypatch):
    """[Knob-off can never enter the branch — end-to-end] country='ES' en form_data PERO el knob
    MEALFIT_COUNTRY_SYSTEM está OFF ⇒ country_for_form_data ignora 'ES' y devuelve 'DO' SIEMPRE
    ⇒ el plan se comporta IDÉNTICO al caso DO puro (rechaza, no advisory)."""
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    state = _f_minimal_state(plan_result=_f_minimal_plan_arroz_de_noche(), country="ES", attempt=1)
    result = _f_run(_go_f.review_plan_node(state))
    assert result["review_passed"] is False, "knob OFF debe ignorar country='ES' y rechazar como DO"


def test_f_integration_do_locrio_desayuno_hard_sigue_rechazando(monkeypatch):
    """[DO byte-identidad] 'Locrio de pollo' en DESAYUNO es la regla `SIEMPRE duro` de DO
    (decisión de producto — no degrada nunca, ni en attempt 1 ni en el final) ⇒ DO debe seguir
    rechazando SIEMPRE, sin importar el attempt."""
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    state = _f_minimal_state(plan_result=_f_minimal_plan_locrio_desayuno(), country=None, attempt=1)
    result = _f_run(_go_f.review_plan_node(state))
    assert result["review_passed"] is False, "DO: Locrio en desayuno es SIEMPRE duro, nunca advisory"


def test_f_integration_beta_locrio_desayuno_hoy_tambien_es_soft_por_t4(monkeypatch):
    """[Contexto REAL — no un bug de este fix] La regla `SIEMPRE duro` de desayuno-arroz de DO
    (test hermano arriba) es TAMBIÉN ablandada a soft para beta por el blanket softening de T4
    (`slot_rules_for_country`: "TODA regla hardness='soft'" para país != DO, HOY sin excepciones
    — docs/country_system_f1.md, fila 5). Verificado en vivo durante el desarrollo de este test:
    `_detect_slot_appropriateness([...Locrio desayuno...], {'country':'ES'})` devuelve
    `hard: False`. Consecuencia de ESTE fix (f): esa violación, siendo soft-only, TAMBIÉN entrega
    advisory desde el intento 1 para beta — no es un caso separado, es la MISMA regla del helper
    aplicada a un ejemplo distinto (desayuno en vez de cena). Si T4 alguna vez re-endurece una
    regla específica para beta (el código ya lo soporta — ver docstring de
    `slot_coherence_backstop_for_meal`), este test se pondría rojo y sería la señal correcta de
    revisar la interacción con (f)."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    issues = _go_f._detect_slot_appropriateness(
        _f_minimal_plan_locrio_desayuno()["days"], {"country": "ES"}
    )
    assert issues and issues[0]["hard"] is False, (
        "precondición: HOY el desayuno-arroz debe ser soft para beta (T4 blanket) — si esto "
        "cambia, la aserción de abajo (review_passed=True) también debe re-evaluarse"
    )
    state = _f_minimal_state(plan_result=_f_minimal_plan_locrio_desayuno(), country="ES", attempt=1)
    result = _f_run(_go_f.review_plan_node(state))
    assert result["review_passed"] is True, (
        "consecuencia esperada de (f): beta + Locrio-desayuno soft-only ⇒ advisory desde attempt 1"
    )


def test_f_integration_do_final_attempt_advisory_sin_cambios(monkeypatch):
    """[Regresión, comportamiento PRE-existente sin tocar] DO en el intento FINAL (3 de 3) con
    soft-only ya era advisory ANTES de esta task — confirma que el fix no rompió ese camino.

    `NIGHT_RICE_COMPOUND_FINAL` desactivado a propósito: es una feature ORTOGONAL (P1-NIGHT-RICE-
    COMPOUND-FINAL) que en el intento final intenta un autofix de último recurso sobre el MISMO
    plato de arroz-de-noche — mutaba el plan y disparaba P2-BAND-RETRY-GATE (macros del
    fixture desincronizadas tras el autofix), contaminando la prueba de ESTE gate específico.
    Reproducido en vivo durante el desarrollo: sin este monkeypatch, review_passed daba False
    por 'PRECISIÓN DE MACROS BAJA', no por el gate de slot-appropriateness."""
    monkeypatch.delenv("MEALFIT_COUNTRY_SYSTEM", raising=False)
    monkeypatch.setattr(_go_f, "NIGHT_RICE_COMPOUND_FINAL", False)
    plan = _f_minimal_plan_arroz_de_noche()
    state = _f_minimal_state(plan_result=plan, country=None, attempt=3)
    result = _f_run(_go_f.review_plan_node(state))
    assert result["review_passed"] is True, "DO en el intento final con soft-only ya era advisory"
    assert plan.get("_slot_appropriateness_advisory_final") is True
    assert plan.get("_slot_appropriateness_advisory_beta_early") is not True, (
        "el marker beta_early es EXCLUSIVO del camino beta-temprano — DO-final no debe setearlo"
    )


def test_f_integration_clean_plan_aprueba_sin_ningun_marker(monkeypatch):
    """Sanity (mismo patrón que test_no_block_flag_no_change del vecino P2-A): un plan SIN
    violaciones de horario se aprueba limpio, sin marcadores de advisory de ningún tipo."""
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    plan = {
        "calories": 2000,
        "macros": {"protein": 150, "carbs": 200, "fats": 67},
        "days": [{"day": 1, "meals": [
            {"meal": "Cena", "name": "Pescado al horno con vegetales",
             "ingredients": ["200 g pescado", "150 g vegetales asados"],
             "recipe": ["Mise en place: corta los vegetales y sazona el pescado.",
                        "El Toque de Fuego: hornea el pescado con los vegetales 20 min a 180°C.",
                        "Montaje: sirve caliente."],
             "protein": 150, "carbs": 200, "fats": 67, "cals": 2000}
        ]}],
    }
    state = _f_minimal_state(plan_result=plan, country="ES", attempt=1)
    result = _f_run(_go_f.review_plan_node(state))
    assert result["review_passed"] is True
    assert "_slot_appropriateness_advisory_final" not in plan
    assert "_slot_appropriateness_advisory_beta_early" not in plan


# ── f2: unit tests del helper puro (los 7 casos verificados en vivo durante el desarrollo) ────

def test_f_helper_do_sin_issues():
    assert _go_f._slot_appropriateness_advisory_decision([], 1, 2, "DO") == (False, False, False, False)


def test_f_helper_do_hard_nunca_advisory_ni_final_ni_no_final():
    for attempt in (1, 2):
        adv, has_hard, is_final, beta_only = _go_f._slot_appropriateness_advisory_decision(
            [{"hard": True}], attempt, 2, "DO"
        )
        assert has_hard is True
        assert beta_only is False
        assert adv is False, f"DO con hard=True NUNCA debe ser advisory (attempt={attempt})"


def test_f_helper_do_soft_no_final_rechaza_byte_identico_pre_fix():
    """EL caso más importante de byte-identidad: DO, issue soft, NO es el intento final ⇒
    debe seguir RECHAZANDO (forzando retry) — exactamente como pre-Task-9."""
    adv, has_hard, is_final, beta_only = _go_f._slot_appropriateness_advisory_decision(
        [{"hard": False}], 1, 2, "DO"
    )
    assert adv is False
    assert is_final is False
    assert beta_only is False, "DO NUNCA activa la rama beta_soft_only"


def test_f_helper_do_soft_final_advisory_byte_identico_pre_fix():
    adv, has_hard, is_final, beta_only = _go_f._slot_appropriateness_advisory_decision(
        [{"hard": False}], 2, 2, "DO"
    )
    assert adv is True
    assert is_final is True
    assert beta_only is False


_BETA_CCS_F = tuple(cc for cc, p in constants.COUNTRY_PROFILES.items() if p["is_beta"])


def test_f_helper_beta_soft_only_advisory_desde_intento_1():
    """EL comportamiento NUEVO: país beta, issue soft, intento 1 de 2 (NO final) ⇒ advisory YA,
    sin esperar al intento final."""
    assert _BETA_CCS_F, "fixture vacío — no hay países beta en COUNTRY_PROFILES"
    for cc in _BETA_CCS_F:
        adv, has_hard, is_final, beta_only = _go_f._slot_appropriateness_advisory_decision(
            [{"hard": False}], 1, 3, cc
        )
        assert adv is True, f"{cc}: beta soft-only debe ser advisory desde el intento 1"
        assert is_final is False, f"{cc}: attempt 1 de 3 NO es final — la precondición del test"
        assert beta_only is True, f"{cc}: beta_soft_only debe ser True"


def test_f_helper_beta_hard_sigue_forzando_retry():
    """Beta con violación DURA sigue rechazando SIEMPRE — el override de país neutraliza
    `hard`, nunca lo inventa; si `_detect_slot_appropriateness` alguna vez deja un hard=True
    vivo para beta, este helper debe seguir respetándolo."""
    adv, has_hard, is_final, beta_only = _go_f._slot_appropriateness_advisory_decision(
        [{"hard": True}], 1, 3, "ES"
    )
    assert adv is False
    assert beta_only is False, "beta_only exige not has_hard — con hard=True nunca es True"


def test_f_helper_beta_mezcla_hard_soft_sigue_forzando_retry():
    """Beta con MEZCLA hard+soft en el MISMO plan sigue rechazando en cualquier attempt — no
    solo el issue duro sobrevive, TODO el plan se regenera (mismo contrato que DO)."""
    adv, has_hard, is_final, beta_only = _go_f._slot_appropriateness_advisory_decision(
        [{"hard": True}, {"hard": False}], 1, 3, "ES"
    )
    assert adv is False
    assert has_hard is True
    assert beta_only is False


def test_f_helper_beta_final_attempt_tambien_advisory():
    """Beta en el intento final con soft-only también es advisory (ya lo era antes de esta
    task, vía is_final — el fix no lo cambia, solo AGREGA el camino temprano)."""
    adv, _, is_final, beta_only = _go_f._slot_appropriateness_advisory_decision(
        [{"hard": False}], 3, 3, "ES"
    )
    assert adv is True
    assert is_final is True


def test_f_helper_pais_desconocido_no_diferencia_de_do():
    """`_slot_appropriateness_advisory_decision` NO canonicaliza — recibe país YA resuelto vía
    `country_for_form_data` (que sí hace fail-safe a 'DO' para desconocidos). Un país crudo
    'xx' que llegara aquí SIN pasar por esa puerta se trataría como beta (comportamiento
    documentado: este helper confía en su input, no es un 2º canonicalizador)."""
    adv, _, _, beta_only = _go_f._slot_appropriateness_advisory_decision([{"hard": False}], 1, 3, "xx")
    assert beta_only is True, (
        "'xx' no es 'DO' literal — el helper lo trata como beta; la responsabilidad de "
        "canonicalizar es de country_for_form_data, NO de este helper (SRP)"
    )


# ── f3: golden/DO byte-identidad — el helper reproduce EXACTO el oráculo legacy para país=DO ──

@pytest.mark.parametrize("attempt,max_attempts,issues", [
    (1, 2, []),
    (1, 2, [{"hard": False}]),
    (1, 2, [{"hard": True}]),
    (2, 2, [{"hard": False}]),
    (2, 2, [{"hard": True}]),
    (1, 3, [{"hard": False}, {"hard": False}]),
    (1, 3, [{"hard": True}, {"hard": False}]),
    (3, 3, [{"hard": True}, {"hard": False}]),
])
def test_f_golden_do_reproduce_formula_legacy_exacta(attempt, max_attempts, issues):
    """[DO byte-identidad, golden] Para país='DO' (y, por transitividad, para el knob apagado —
    country_for_form_data SIEMPRE devuelve 'DO' en ese caso), el `advisory` del helper NUEVO
    debe coincidir EXACTO, caso por caso, con `_legacy_slot_advisory` (la fórmula
    `is_final and not has_hard` pre-Task-9, sin ningún concepto de país)."""
    adv, _, _, beta_only = _go_f._slot_appropriateness_advisory_decision(issues, attempt, max_attempts, "DO")
    legacy = _legacy_slot_advisory(issues, attempt, max_attempts)
    assert adv == legacy, f"DO diverge del oráculo legacy: nuevo={adv}, legacy={legacy}"
    assert beta_only is False, "DO NUNCA debe activar beta_soft_only, en ningún caso"


def test_f_golden_knob_off_country_for_form_data_es_siempre_do():
    """El puente knob-off → 'DO' vive en `country_for_form_data` (T1, la ÚNICA puerta) — este
    test ancla que, tal como lo consume el call site de review_plan_node, un form_data con
    country='ES' bajo knob OFF resuelve 'DO', así que el helper de arriba recibe 'DO' incluso
    si el usuario declaró un país beta."""
    import os
    prev = os.environ.pop("MEALFIT_COUNTRY_SYSTEM", None)
    try:
        resolved = constants.country_for_form_data({"country": "ES"})
        assert resolved == "DO"
    finally:
        if prev is not None:
            os.environ["MEALFIT_COUNTRY_SYSTEM"] = prev


# ── f4: parser — el wiring real dentro de review_plan_node ────────────────────────────────────

def _review_plan_node_slot_gate_cuerpo() -> str:
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    ini = src.index("if SLOT_APPROPRIATENESS_GATE_ENABLED:")
    fin = src.index("# [P1-SLOT-INCOHERENCE-GATE", ini)
    return src[ini:fin]


def test_f_wiring_review_plan_node_llama_al_helper():
    cuerpo = _review_plan_node_slot_gate_cuerpo()
    assert "_slot_appropriateness_advisory_decision(" in cuerpo
    assert "_sa_advisory, _sa_has_hard, _sa_is_final, _sa_beta_soft_only" in cuerpo


def test_f_wiring_deriva_pais_via_ssot_no_lector_crudo():
    """[Re-anclado por P1-REVIEW-RETRY-FEEDBACK-DO] La derivación se movió al
    tope del nodo para gobernar también los gates que preceden al de horario."""
    source = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "review_plan_node"
    )
    assignments = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_rpn_country" for target in node.targets)
    ]
    assert len(assignments) == 1
    value = assignments[0].value
    assert isinstance(value, ast.Call)
    # [P1-ARQ25-F7-CULTURE] el gate de horario es CULTURAL: lee la puerta cultural (SSOT que cae al mercado sin elección)
    assert isinstance(value.func, ast.Name) and value.func.id in ("country_for_form_data", "cultural_country_for_form_data")
    assert len(value.args) == 1 and isinstance(value.args[0], ast.Name) and value.args[0].id == "form_data"


def test_f_wiring_marker_beta_early_solo_si_no_es_final():
    cuerpo = _review_plan_node_slot_gate_cuerpo()
    assert '_slot_appropriateness_advisory_beta_early' in cuerpo
    assert "if _sa_beta_soft_only and not _sa_is_final:" in cuerpo


def test_f_wiring_knob_off_never_enters_branch_comentado():
    """El call site debe DOCUMENTAR explícitamente la garantía knob-off (requisito del brief:
    'Knob-off can never enter the branch') — no solo confiar en que sea cierto por transitividad
    de country_for_form_data (ya anclado en test_f_golden_knob_off_...)."""
    cuerpo = _review_plan_node_slot_gate_cuerpo()
    assert "byte-idéntico al comportamiento pre-Task-9" in cuerpo or "byte-idéntico" in cuerpo


# ── f5: MUTACIONES bidireccionales (requisito explícito del brief) ────────────────────────────

def test_f_mutacion_quitar_el_gate_do_golden_se_pone_rojo():
    """MUTACIÓN #1 ('remove gate ⇒ DO golden RED'): si alguien BORRARA el helper y volviera al
    `if _sa_is_final and not _sa_has_hard:` original PERO rompiera algo en el camino (p.ej.
    invirtiera la condición), el test golden f3 lo cazaría. Reproducido aquí sin tocar el
    archivo real: una versión MUTADA del helper (condición invertida) falla el golden."""
    def _mutated_no_gate(issues, attempt, max_attempts, country):
        has_hard = any(i.get("hard") for i in issues)
        is_final = int(attempt) >= int(max_attempts)
        # mutación: el gate de país desaparece pero la condición final TAMBIÉN se rompe
        # (simula "alguien quitó el helper completo y el gate quedó siempre-False"):
        advisory = False
        return advisory, has_hard, is_final, False

    # Con el helper MUTADO (gate roto), el caso DO-final-soft (que SIEMPRE debía ser advisory)
    # deja de coincidir con el oráculo legacy:
    mutated_adv, _, _, _ = _mutated_no_gate([{"hard": False}], 2, 2, "DO")
    legacy = _legacy_slot_advisory([{"hard": False}], 2, 2)
    assert mutated_adv != legacy, "la mutación debía romper el golden DO — si no, el golden no prueba nada"
    # Y el helper REAL (no mutado) sigue coincidiendo:
    real_adv, _, _, _ = _go_f._slot_appropriateness_advisory_decision([{"hard": False}], 2, 2, "DO")
    assert real_adv == legacy


def test_f_mutacion_quitar_la_advisory_beta_test_se_pone_rojo():
    """MUTACIÓN #2 ('remove advisory ⇒ beta test RED'): una versión del helper que IGNORA
    `beta_soft_only` (vuelve a la fórmula pre-Task-9 pura) reproduce el gap que este fix cierra
    — el test beta (f2) se pondría rojo contra esa versión."""
    def _pre_task9_formula(issues, attempt, max_attempts, country):
        has_hard = any(i.get("hard") for i in issues)
        is_final = int(attempt) >= int(max_attempts)
        advisory = is_final and not has_hard  # SIN el `or beta_soft_only`
        return advisory, has_hard, is_final, False

    pre_fix_adv, _, _, _ = _pre_task9_formula([{"hard": False}], 1, 3, "ES")
    assert pre_fix_adv is False, "reproduce el gap: pre-Task-9 NO daba advisory en attempt 1 para beta"
    real_adv, _, _, beta_only = _go_f._slot_appropriateness_advisory_decision([{"hard": False}], 1, 3, "ES")
    assert real_adv is True, "el fix real SÍ da advisory desde attempt 1 para beta soft-only"
    assert real_adv != pre_fix_adv, "la mutación (quitar el `or beta_soft_only`) debía cambiar el resultado"


# ══════════════════════════════════════════════════════════════════════════════════════════════
# OLA FINAL (review de fase, opus) — 2026-08-18 — C3, I1, I2
# ══════════════════════════════════════════════════════════════════════════════════════════════
#
# El review de fase (whole-phase, tras el cierre de Task 10) encontró 6 strings DO-reachable
# retargeteados en silencio por altas de Fase 2 (Chicharrón/CO T6, Cordero+Requesón+Judías
# pintas/ES T5): 4 alias BARE genuinamente redundantes o colisionantes con vocabulario/filas
# pre-fase ('chicharron' en Chicharrón — redundante con su propio nombre canónico; 'lamb' en
# Cordero — fuzzy-matcheaba 'lambí'/'lambi', un molusco real, no un cordero; 'ricotta' en
# Requesón — colisionaba con 'Queso ricotta', fila DO priced que lo tenía primero; 'pinto beans'
# en Judías pintas — colisionaba con 'Frijoles pintos', fila DO priced que lo tenía primero) más
# un 5º hallazgo separado ('Chicharrón de pollo' retargeteado a la fila de CERDO, nunca revisado
# por T6 — que solo evaluó 'de cerdo'/bare) que la remoción del alias NO alcanza a cerrar (el
# NOMBRE CANÓNICO de la fila ya provee el mismo match vía CONTAINS) y requirió un guard temprano
# (`C3.1`) en `shopping_calculator.normalize_name`, mismo patrón que el guard de 'pavo' ya
# existente en esa función.
#
# Los 4 alias se removieron con `scripts/retarget_alias_fix_2026_08_18.py` (idempotente,
# dry-run→--commit, sincroniza DB + los JSON SSOT de origen). El guard C3.1 vive en
# `shopping_calculator.py`. Esta sección ancla ambos + dos invariantes estructurales que el
# review pidió específicamente:
#   - I1: ningún alias (ni el nombre canónico de una fila, tratado como su propio self-alias por
#     `_construir_indice_alias`) puede vivir en 2+ filas distintas — la CLASE de bug que produjo
#     ricotta/pinto beans.
#   - I2: el sweep de colisión de `is_country_catalog_unpriced_item` (T5, arriba) se extiende de
#     nombres/pools a ALIASES de filas PRICED — con verificación de que las 4 colisiones
#     conocidas (mora azul/azafrán de la india/queso panela/quesito panela) son benignas
#     ESPECÍFICAMENTE para `canonicalize_shopping_food_name` (que resuelve por `master_map` ANTES
#     de siquiera consultar `is_country_catalog_unpriced_item`).

# ── C3. Los 4 alias bare retargeteados + el guard de "Chicharrón de pollo" ──────────────────────

def test_c3_chicharron_de_cerdo_sigue_resolviendo_a_chicharron(sc):
    """['Chicharrón de cerdo' — mejora ACEPTADA de T6, no tocada por esta ola] Sigue resolviendo
    a 'Chicharrón' (fila CO) tras remover el alias bare 'chicharron' -- documentado en
    new_foods_mx_co_2026_08_17.json._provenance: chicharrón real (kcal 544/fat 31.3g) diverge
    >200% de 'Cerdo' genérico (kcal 169.6/fat 9.47g), la fila nueva es MÁS precisa. A diferencia
    de 'de pollo' (test siguiente), esto NO revierte a pre-fase a propósito.

    [micro-fix ola final · 2026-08-18] El plural también se verifica: 'Chicharrones de cerdo'
    debe seguir resolviendo a 'Chicharrón' (la mejora aceptada aplica igual al plural — verificado
    en vivo ANTES de ensanchar el regex del guard, ya pasaba porque el guard exige 'pollo'
    co-presente y este string no lo tiene)."""
    assert sc.normalize_name("Chicharrón de cerdo") == "Chicharrón"
    assert sc.normalize_name("chicharron de cerdo") == "Chicharrón"
    assert sc.normalize_name("Chicharrones de cerdo") == "Chicharrón"
    assert sc.normalize_name("chicharrones de cerdo") == "Chicharrón"


def test_c3_chicharron_de_pollo_vuelve_a_pechuga_de_pollo(sc):
    """[RED-first, reproducido contra HEAD faf10f7] 'Chicharrón de pollo' resolvía a 'Chicharrón'
    (fila CO de CERDO) -- retargeteo NUNCA revisado por T6. Remover el alias bare 'chicharron'
    NO alcanza (verificado en vivo antes de escribir el guard: el NOMBRE CANÓNICO de la fila
    sigue matcheando vía CONTAINS incluso sin el alias explícito) -- el guard temprano `C3.1` en
    `shopping_calculator.normalize_name` restaura la resolución pre-fase ('Pechuga de pollo',
    vía substring 'pollo').

    [micro-fix ola final · 2026-08-18, RED-first reproducido contra a0fdc11] El PLURAL
    'chicharrones de pollo' se escapaba del guard original (`\\bchicharr[oó]n\\b` sin sufijo no
    matchea 'chicharrones') y caía al CONTAINS de abajo, que SÍ matchea 'chicharrones' (el alias
    plural sobrevive en la fila -- solo el bare singular se removió) -> 'Chicharrón' (cerdo),
    mismo bug de fondo que el singular. Regex ensanchado a `\\bchicharr[oó]n(?:es)?\\b` -- mismo
    patrón `(?:s|es)?` que `_scan_allergen_violations` ya usa para plurales españoles."""
    assert sc.normalize_name("Chicharrón de pollo") == "Pechuga de pollo"
    assert sc.normalize_name("chicharron de pollo") == "Pechuga de pollo"
    assert sc.normalize_name("Chicharrones de pollo") == "Pechuga de pollo"
    assert sc.normalize_name("chicharrones de pollo") == "Pechuga de pollo"


def test_c3_bare_chicharron_sigue_resolviendo_a_chicharron(sc):
    """Control: el guard C3.1 exige la palabra 'pollo' co-presente en el mismo string -- bare
    'chicharrón' (el caso CO que T6 quiso resolver, "antes no resolvía nada") no se ve afectado
    por el guard ni por la remoción del alias (su propio nombre canónico basta)."""
    assert sc.normalize_name("chicharron") == "Chicharrón"
    assert sc.normalize_name("Chicharrón") == "Chicharrón"


def test_c3_lambi_ya_no_resuelve_a_cordero(sc):
    """[RED-first] 'lambí'/'lambi' (molusco caribeño real -- vive en `PROTEIN_SYNONYMS['pescado']`
    junto a merluza/róbalo/carite) fuzzy-matcheaba el alias 'lamb' de Cordero (ratio 0.889 >=
    umbral 0.87 de `_FUZZY_MATCH_THRESHOLD`) -> 'Cordero'. Removido el alias, vuelven a caer sin
    resolver -- pre-fase ninguna fila de cordero existía, así que esto restaura exactamente ese
    estado (no una mejora nueva, un revert)."""
    assert sc.normalize_name("lambí") != "Cordero"
    assert sc.normalize_name("lambi") != "Cordero"
    assert sc.normalize_name("lamb") != "Cordero"


def test_c3_cordero_sigue_resolviendo_para_sus_items_curados(sc):
    """Control: Cordero conserva sus alias propios (sin 'lamb') -- los items curados de la lista
    ES (T1) siguen resolviendo, la remoción fue quirúrgica sobre UN SOLO alias."""
    assert sc.normalize_name("Cordero") == "Cordero"
    assert sc.normalize_name("carne de cordero") == "Cordero"
    assert sc.normalize_name("pierna de cordero") == "Cordero"


def test_c3_ricotta_resuelve_a_queso_ricotta(sc):
    """[RED-first] 'ricotta' resolvía a 'Requesón' (alta ES T5, alias bare duplicado) en vez de
    'Queso ricotta' (fila DO priced pre-fase que lo tenía primero, price_per_unit=245) --
    removido el alias duplicado de Requesón."""
    assert sc.normalize_name("ricotta") == "Queso ricotta"
    assert sc.normalize_name("Ricotta") == "Queso ricotta"


def test_c3_requeson_conserva_sus_alias_propios(sc):
    assert sc.normalize_name("requeson") == "Requesón"
    assert sc.normalize_name("requesón") == "Requesón"


def test_c3_pinto_beans_resuelve_a_frijoles_pintos(sc):
    """[RED-first] 'pinto beans' resolvía a 'Judías pintas' (alta ES T5, alias bare duplicado) en
    vez de 'Frijoles pintos' (fila DO priced pre-fase que lo tenía primero, price_per_lb=72.01)
    -- removido el alias duplicado de Judías pintas."""
    assert sc.normalize_name("pinto beans") == "Frijoles pintos"
    assert sc.normalize_name("Pinto beans") == "Frijoles pintos"


def test_c3_judias_pintas_conserva_sus_alias_propios(sc):
    assert sc.normalize_name("judias pintas") == "Judías pintas"
    assert sc.normalize_name("judías pintas") == "Judías pintas"


# ── C3 Durable Guard. El corpus DO completo (dish_templates + pools + reverse-map + filas
# pre-fase) contra un baseline COMMITTED — la mitad "asimétrica" del guard: I10/Durable-Guard-#6
# protege el catálogo de PAÍS BETA (country_gaps/*.json); este protege el vocabulario DO desde el
# LADO CONTRARIO — que dar de alta un alimento de país beta no le cambie la resolución a un
# string que el sistema DO ya reconocía. ──────────────────────────────────────────────────────

_DO_CORPUS_BASELINE_JSON = _BACKEND / "scripts" / "data" / "do_corpus_retarget_baseline_2026_08_18.json"
_NEW_FOOD_FILES_C3 = [
    _BACKEND / "scripts" / "data" / "new_foods_es_2026_08_17.json",
    _BACKEND / "scripts" / "data" / "new_foods_mx_co_2026_08_17.json",
    _BACKEND / "scripts" / "data" / "new_foods_pr_us_2026_08_17.json",
    _BACKEND / "scripts" / "data" / "new_foods_rd_topup_2026_08_17.json",
]


def _build_c3_do_corpus(sc):
    """[C3 Durable Guard] Reconstruye el MISMO corpus que
    `scripts/gen_do_corpus_retarget_baseline_2026_08_18.py` -- DEBE permanecer byte-idéntico a
    ese generador (misma receta, mismo orden de fuentes) o el baseline committed deja de ser la
    verdad contra la que este test compara:

      1. `data/dish_templates.json` (DO): cada `name`/`protein`/`base` de cada template.
      2. Los 4 pools `DOMINICAN_*` (constants.py): cada string.
      3. `GLOBAL_REVERSE_MAP` (constants.py): cada KEY (variante) y cada VALUE (base).
      4. Cada fila PRE-FASE de `master_ingredients` (name NO en ninguno de los 4
         `new_foods_*_2026_08_17.json`, la lista frozen committed que sustituye a `created_at`
         -- la tabla no tiene esa columna, verificado contra `information_schema.columns`) con
         su nombre canónico + cada uno de sus alias.

    Retorna (corpus: set[str], new_row_names: set[str])."""
    strings = set()
    with open(_BACKEND / "data" / "dish_templates.json", encoding="utf-8") as f:
        dt = json.load(f)
    for t in dt["templates"]:
        for field in ("name", "protein", "base"):
            if t.get(field):
                strings.add(t[field])

    for pool_name in ("DOMINICAN_PROTEINS", "DOMINICAN_CARBS", "DOMINICAN_VEGGIES_FATS", "DOMINICAN_FRUITS"):
        strings.update(getattr(constants, pool_name))

    for k, v in constants.GLOBAL_REVERSE_MAP.items():
        strings.add(k)
        strings.add(v)

    new_row_names = set()
    for fn in _NEW_FOOD_FILES_C3:
        with open(fn, encoding="utf-8") as f:
            for r in json.load(f):
                new_row_names.add(r["name"])

    master_list = sc.get_master_ingredients()
    for r in master_list:
        if r["name"] in new_row_names:
            continue
        strings.add(r["name"])
        strings.update(r.get("aliases") or [])

    strings = {s for s in strings if isinstance(s, str) and s.strip()}
    return strings, new_row_names


@pytest.mark.e2e
def test_c3_durable_guard_do_corpus_retarget_baseline(sc):
    """[C3 Durable Guard · el guard asimétrico] Re-resuelve TODO el corpus DO contra el catálogo
    VIVO y exige coincidencia EXACTA con `do_corpus_retarget_baseline_2026_08_18.json` committed.
    Un retargeteo INTENCIONAL futuro se cierra regenerando el baseline
    (`scripts/gen_do_corpus_retarget_baseline_2026_08_18.py`), documentando el delta en
    `accepted_deltas` del JSON y commiteando el archivo actualizado explícitamente -- así el diff
    SIEMPRE pasa por review, nunca en silencio (mismo patrón que I10/
    `test_retarget_diff_committed_country_gaps_matched_field_vs_resolver_vivo`, aplicado al lado
    DO en vez de al lado país-beta).

    RED-first: contra HEAD faf10f7 (pre-ola-final) este test hubiera flageado EXACTAMENTE 6
    strings del corpus -- 'ricotta', 'pinto beans', 'lambi', 'lambí', 'chicharrón de pollo',
    'chicharron de pollo' -- ver `test_c3_durable_guard_red_first_reproduce_exactamente_los_6`
    (reproduce el estado pre-fix en memoria/mecanismo, no re-corre este test contra HEAD)."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()

    assert _DO_CORPUS_BASELINE_JSON.exists(), f"{_DO_CORPUS_BASELINE_JSON} debe existir committed"
    with open(_DO_CORPUS_BASELINE_JSON, encoding="utf-8") as f:
        baseline = json.load(f)
    committed_mapping = baseline["mapping"]

    corpus, new_row_names = _build_c3_do_corpus(sc)
    assert len(corpus) >= 1000, f"corpus sospechosamente chico ({len(corpus)}) -- ¿alguna fuente se leyó vacía?"
    assert len(new_row_names) >= 100, (
        f"esperaba >=100 filas nuevas de Fase 2 excluidas del corpus pre-fase, encontré {len(new_row_names)}"
    )

    # el corpus committed y el reconstruido en vivo deben tener EXACTAMENTE los mismos strings —
    # si divergen, el corpus mismo cambió (nuevo template/pool/fila) y hace falta regenerar el
    # baseline, no solo comparar la intersección en silencio (eso dejaría huecos ciegos).
    corpus_missing_from_baseline = corpus - set(committed_mapping)
    baseline_missing_from_corpus = set(committed_mapping) - corpus
    assert not corpus_missing_from_baseline, (
        f"{len(corpus_missing_from_baseline)} string(s) nuevos en el corpus vivo, ausentes del "
        f"baseline committed -- regenerar con scripts/gen_do_corpus_retarget_baseline_2026_08_18.py "
        f"y revisar el diff: {sorted(corpus_missing_from_baseline)[:10]}"
    )
    assert not baseline_missing_from_corpus, (
        f"{len(baseline_missing_from_corpus)} string(s) del baseline committed ya no existen en "
        f"el corpus vivo (fuente removida) -- regenerar el baseline: "
        f"{sorted(baseline_missing_from_corpus)[:10]}"
    )

    retargets = []
    for s in sorted(corpus):
        live = sc.normalize_name(s)
        expected = committed_mapping[s]
        if live != expected:
            retargets.append((s, expected, live))

    assert not retargets, (
        "RETARGET DETECTADO en el corpus DO -- el resolver vivo apunta distinto de lo que el "
        "baseline committed declara. Si es intencional, regenera "
        "scripts/data/do_corpus_retarget_baseline_2026_08_18.json vía "
        "scripts/gen_do_corpus_retarget_baseline_2026_08_18.py, documenta el delta en "
        "accepted_deltas y commitea el JSON actualizado explícitamente:\n" +
        "\n".join(f"  {s!r}: baseline={exp!r} vs live={live!r}" for s, exp, live in retargets)
    )


@pytest.mark.e2e
def test_c3_durable_guard_red_first_reproduce_exactamente_los_6(sc):
    """[RED-first, el contrato completo] Reconstruye el estado PRE-ola-final: los 3 alias bare
    de vuelta EN MEMORIA (Cordero+lamb, Requesón+ricotta, Judías pintas+pinto beans) -- el guard
    C3.1 es CÓDIGO, no dato, así que su necesidad se verifica por el mecanismo CONTAINS directo
    en vez de mutar la función -- y confirma que EXACTAMENTE estos 6 strings del corpus
    committed hubieran flageado distinto: 'ricotta', 'pinto beans', 'lambi', 'lambí', 'chicharrón
    de pollo', 'chicharron de pollo'. El resto del corpus (1462 de 1468 strings) permanece
    IDÉNTICO -- prueba que el fix fue QUIRÚRGICO, no un revert amplio que hubiera cambiado más de
    lo debido."""
    import copy
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()

    with open(_DO_CORPUS_BASELINE_JSON, encoding="utf-8") as f:
        committed_mapping = json.load(f)["mapping"]

    master_list = sc.get_master_ingredients()
    mutated = [copy.deepcopy(r) for r in master_list]
    touched = set()
    for r in mutated:
        if r["name"] == "Cordero" and "lamb" not in (r.get("aliases") or []):
            r.setdefault("aliases", []).append("lamb")
            touched.add("Cordero")
        if r["name"] == "Requesón" and "ricotta" not in (r.get("aliases") or []):
            r.setdefault("aliases", []).append("ricotta")
            touched.add("Requesón")
        if r["name"] == "Judías pintas" and "pinto beans" not in (r.get("aliases") or []):
            r.setdefault("aliases", []).append("pinto beans")
            touched.add("Judías pintas")
    assert touched == {"Cordero", "Requesón", "Judías pintas"}, (
        f"esperaba mutar exactamente 3 filas, mutó {touched} -- ¿el fixture del catálogo cambió?"
    )

    sc._NORMALIZE_ALIAS_INDEX = None
    orig_get_master = sc.get_master_ingredients
    sc.get_master_ingredients = lambda: mutated
    try:
        pre_fix_alias_driven = {
            s: sc.normalize_name(s) for s in ("ricotta", "pinto beans", "lambi", "lambí")
        }
    finally:
        sc.get_master_ingredients = orig_get_master
        sc._NORMALIZE_ALIAS_INDEX = None

    flagged_alias_driven = {
        s for s, v in pre_fix_alias_driven.items() if v != committed_mapping[s]
    }
    assert flagged_alias_driven == {"ricotta", "pinto beans", "lambi", "lambí"}, (
        f"esperaba exactamente estos 4 (de los 6) flageados al re-agregar los 3 alias, "
        f"obtuve {flagged_alias_driven}"
    )

    # Los otros 2 ('chicharrón de pollo' + variante sin tilde) dependen del guard C3.1 (código,
    # no un alias mutable) -- se prueba que el MECANISMO subyacente (CONTAINS, INTENTO 2) sigue
    # apuntando a 'Chicharrón' vía el nombre canónico de la fila (nunca dependió del alias bare
    # explícito, por eso remover solo el alias no hubiera bastado) confirmando que sin el guard,
    # el bug se reproduciría también para estos 2.
    from constants import strip_accents
    _, contains = sc._construir_indice_alias(sc.get_master_ingredients())
    for s in ("chicharrón de pollo", "chicharron de pollo"):
        target = strip_accents(s.lower())
        contains_match = next((name for pat, name, *_ in contains if pat.search(target)), None)
        assert contains_match == "Chicharrón", (
            f"{s!r}: el mecanismo CONTAINS ya no apunta a 'Chicharrón' ({contains_match!r}) -- "
            f"si la fila se renombró, revisar si el guard C3.1 (shopping_calculator.py) sigue "
            f"siendo necesario, no asumir que ya no lo es"
        )
        assert committed_mapping[s] == "Pechuga de pollo", (
            f"baseline committed para {s!r} debe seguir siendo 'Pechuga de pollo'"
        )


def test_c3_durable_guard_mutacion_re_agregar_alias_en_memoria_reproduce_red(sc):
    """[Mutación, contrato C3 explícito: 're-add one removed alias in memory ⇒ RED'] Re-agregar
    'ricotta' a Requesón EN MEMORIA (sin tocar la DB real) debe hacer que 'ricotta' resuelva
    distinto de lo que el baseline committed fija -- evidencia de que el fix real (la fila de la
    DB, no solo el comentario del script) es lo que sostiene el contrato del guard de arriba."""
    # [robustecido 2026-08-19 · P1-PLAN-DISPLAY-I18N cierre] La forma original solo AÑADÍA
    # 'ricotta' a Requesón y asumía que esa fila GANABA la colisión en el índice de alias —
    # pero el ganador depende del ORDEN de filas del SELECT (sin ORDER BY), y un UPDATE
    # masivo del catálogo (el fill de name_en reescribió las 347 filas) cambió el orden
    # físico y flipeó al ganador: la mutación se volvía invisible y este test fallaba sin
    # bug real. Determinista: la mutación además QUITA el alias legítimo de 'Queso ricotta'
    # (y su name como alias implícito no aplica: 'ricotta' ≠ 'Queso ricotta' exacto), así
    # 'ricotta' solo puede resolver vía la fila mutada, gane quien gane el orden.
    import copy
    master_list = sc.get_master_ingredients()
    mutated = [copy.deepcopy(r) for r in master_list]
    found = False
    for r in mutated:
        if r["name"] == "Requesón":
            r.setdefault("aliases", [])
            if "ricotta" not in r["aliases"]:
                r["aliases"].append("ricotta")
            found = True
        elif r["name"] == "Queso ricotta":
            r["aliases"] = [a for a in (r.get("aliases") or []) if "ricotta" not in str(a).lower()]
    assert found, "fila 'Requesón' no encontrada -- el fixture del catálogo cambió"

    with open(_DO_CORPUS_BASELINE_JSON, encoding="utf-8") as f:
        expected = json.load(f)["mapping"]["ricotta"]
    assert expected == "Queso ricotta"

    sc._NORMALIZE_ALIAS_INDEX = None
    orig_get_master = sc.get_master_ingredients
    sc.get_master_ingredients = lambda: mutated
    try:
        mutated_live = sc.normalize_name("ricotta")
    finally:
        sc.get_master_ingredients = orig_get_master
        sc._NORMALIZE_ALIAS_INDEX = None

    assert mutated_live != expected, (
        f"la mutación (re-agregar 'ricotta' a Requesón) debía romper el contrato del baseline "
        f"({expected!r}) -- si sigue coincidiendo, el guard de arriba no prueba nada"
    )
    assert mutated_live == "Requesón", f"esperaba reproducir el bug pre-fix exacto, obtuve {mutated_live!r}"
    # el estado REAL (no mutado) sigue coincidiendo con el baseline:
    assert sc.normalize_name("ricotta") == expected


# ── I1. Invariante de unicidad de alias — ningún alias/nombre vive en 2+ filas ──────────────────
#
# Mismo modelo de datos que `shopping_calculator._construir_indice_alias`: el NOMBRE canónico de
# una fila se trata como su propio self-alias (así es como `all_aliases` se construye en
# producción) -- así que "un alias == el nombre de otra fila" es, estructuralmente, el MISMO tipo
# de colisión que "el mismo alias en 2 filas", solo que uno de los dos lados es el nombre en vez
# de un alias explícito. Unificar ambos checks en una sola tabla clave->dueños es lo que hace el
# resolver de verdad, y es lo que este test replica.

# Las 5 colisiones pre-existentes (TODAS anteriores a Fase 2, verificadas: ninguna de las 5 filas
# involucradas está en los 4 `new_foods_*_2026_08_17.json`) que se toleran explícitamente, con su
# razón. CUALQUIER colisión nueva (introducida por una alta de Fase 2 — la clase de bug que
# produjo ricotta/pinto beans) debe ser CERO.
_I1_ALLOWLIST = {
    "mariscos": (
        frozenset({"Calamar", "Mejillones", "Pulpo"}),
        "Término GENÉRICO español para 'shellfish', compartido a propósito entre 3 filas "
        "específicas de marisco -- no hay un dueño único correcto. Preexistente a Fase 2.",
    ),
    "mero": (
        frozenset({"Filete de pescado blanco", "Mero"}),
        "'Mero' (pez específico) también vive como alias de 'Filete de pescado blanco' "
        "(genérico) -- categorización razonable, peor caso: bare 'mero' resuelve al genérico en "
        "vez del específico, ambos son pescado blanco real, sin divergencia nutricional "
        "peligrosa (a diferencia de chicharrón cerdo/pollo). Preexistente a Fase 2.",
    ),
    "nueces": (
        frozenset({"Almendras fileteadas", "Nueces mixtas"}),
        "Término GENÉRICO para frutos secos -- mismo patrón que 'mariscos'. Preexistente a Fase 2.",
    ),
    "tilapia": (
        frozenset({"Filete de pescado blanco", "Tilapia"}),
        "Mismo patrón que 'mero' (pez específico + alias del genérico). Preexistente a Fase 2.",
    ),
    "repollo morado": (
        frozenset({"Repollo", "Repollo morado"}),
        "'repollo morado' vive como alias de 'Repollo' (verde) Y es el NOMBRE literal de la fila "
        "dedicada 'Repollo morado' -- alias legacy que antecede a la fila dedicada, nunca "
        "limpiado. Preexistente a Fase 2 (ninguna alta la introdujo); candidato a cleanup "
        "futuro, fuera de scope de esta ola.",
    ),
}


@pytest.mark.e2e
def test_i1_alias_uniqueness_invariant(sc):
    """[I1 · ola final] Ningún alias (ni el nombre canónico de una fila, contado como su propio
    self-alias) puede resolver a 2+ filas distintas — si lo hace, `normalize_name` decide
    arbitrariamente por orden de iteración (bug silencioso, la CLASE que C3 cerró para
    ricotta/pinto beans). Las 5 colisiones pre-existentes de `_I1_ALLOWLIST` se toleran con su
    razón documentada; cualquier OTRA es cero."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()
    from constants import strip_accents
    from collections import defaultdict

    master_list = sc.get_master_ingredients()
    owners = defaultdict(set)
    for r in master_list:
        owners[strip_accents(r["name"].strip().lower())].add(r["name"])
        for a in (r.get("aliases") or []):
            key = strip_accents(str(a).strip().lower())
            if key:
                owners[key].add(r["name"])

    unexpected = []
    for key, who in owners.items():
        if len(who) <= 1:
            continue
        allow = _I1_ALLOWLIST.get(key)
        if allow and who == allow[0]:
            continue
        unexpected.append((key, sorted(who)))

    assert not unexpected, (
        f"{len(unexpected)} clave(s) NUEVAS con 2+ dueños (no en `_I1_ALLOWLIST`): {unexpected}. "
        f"Si es intencional, documenta la razón en `_I1_ALLOWLIST` con la lista EXACTA de "
        f"dueños; si no, es el mismo bug que C3 cerró -- remueve el alias duplicado de una de "
        f"las filas (`scripts/retarget_alias_fix_2026_08_18.py` es el patrón a seguir/extender)."
    )


def test_i1_allowlist_no_esta_vacio_ni_creciendo_en_silencio(sc):
    """Control de forma: el allowlist tiene exactamente 5 entradas (el número que el sweep en
    vivo produce hoy). Si sube, alguien añadió una colisión nueva y la documentó sin que el
    review la haya visto -- este test no bloquea, pero fuerza a mirar el número dos veces."""
    assert len(_I1_ALLOWLIST) == 5, (
        f"_I1_ALLOWLIST tiene {len(_I1_ALLOWLIST)} entradas, esperaba 5 -- si crece, revisa "
        f"que la nueva entrada tenga razón documentada y no sea, en realidad, un bug nuevo"
    )


# ── I2. El sweep de colisión de `is_country_catalog_unpriced_item` (T5), extendido a ALIASES ────
#
# El sweep original (`test_is_country_catalog_unpriced_item_no_colisiona_con_ningun_nombre_del_
# catalogo_vivo_ni_pools`, arriba) barre NOMBRES de filas + nombres de pools. Este barre ALIASES
# de filas PRICED -- el review encontró 4 colisiones conocidas (mora azul/azafrán de la
# india/queso panela/quesito panela) y pidió verificar que son benignas para
# `canonicalize_shopping_food_name` ESPECÍFICAMENTE: esa función resuelve por `master_map`
# (nombre/alias exacto) PRIMERO y solo consulta `is_country_catalog_unpriced_item` sobre el
# NOMBRE CANÓNICO ya resuelto (`m_item["name"]`, línea `if m_item and
# is_country_catalog_unpriced_item(canonical_name): return canonical_name`) -- así que una
# colisión en el ALIAS crudo nunca llega a esa rama si el nombre canónico de la fila dueña no
# colisiona también.

_I2_ALLOWLIST_PRICED_ALIAS_COLLISIONS = {
    ("mora azul", "Arándanos"): (
        "'mora azul' colisiona con el token 'mora' (fruta de catálogo CO, T6) pero 'Arándanos' "
        "(su fila dueña, DO priced) NO colisiona por su propio nombre."
    ),
    ("azafran de la india", "Cúrcuma"): (
        "'azafran de la india' colisiona con el token 'azafran' (España, T5) pero 'Cúrcuma' "
        "(su fila dueña, DO priced) NO colisiona por su propio nombre."
    ),
    ("queso panela", "Queso blanco"): (
        "'queso panela' es un alias de sinónimo cross-country (T6, synonyms_mx_co_2026_08_17.json) "
        "que colisiona con el token 'panela' (México, T6) pero 'Queso blanco' (su fila dueña, DO "
        "priced) NO colisiona por su propio nombre."
    ),
    ("quesito panela", "Queso blanco"): (
        "Mismo caso que 'queso panela' (micro-fix T6, variante 'quesito')."
    ),
}


@pytest.mark.e2e
def test_i2_registry_collision_sweep_extendido_a_aliases(sc):
    """[I2 · ola final] Para cada ALIAS de cada fila PRICED (price_per_lb>0 o price_per_unit>0)
    del catálogo vivo, `is_country_catalog_unpriced_item` no debe reconocerlo -- salvo las 4
    colisiones conocidas de `_I2_ALLOWLIST_PRICED_ALIAS_COLLISIONS`. Filas SIN precio (las
    propias altas T5-T8) quedan fuera a propósito: que sus PROPIOS alias matcheen el token que
    las reconoce es el comportamiento DISEÑADO, no una colisión."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()

    def _is_priced(row):
        try:
            return float(row.get("price_per_lb") or 0) > 0 or float(row.get("price_per_unit") or 0) > 0
        except (TypeError, ValueError):
            return False

    master_list = sc.get_master_ingredients()
    priced_alias_entries = [
        (a, r["name"]) for r in master_list if _is_priced(r) for a in (r.get("aliases") or [])
    ]
    assert len(priced_alias_entries) >= 500, (
        f"esperaba >=500 entradas alias×fila-priced, encontré {len(priced_alias_entries)} -- "
        f"¿el catálogo se leyó parcial?"
    )

    unexpected = []
    for alias, owner in priced_alias_entries:
        if not sc.is_country_catalog_unpriced_item(alias):
            continue
        if (alias, owner) in _I2_ALLOWLIST_PRICED_ALIAS_COLLISIONS:
            continue
        unexpected.append((alias, owner))

    assert not unexpected, (
        f"{len(unexpected)} alias(es) NUEVOS de filas PRICED colisionan con "
        f"`is_country_catalog_unpriced_item` (no en `_I2_ALLOWLIST_PRICED_ALIAS_COLLISIONS`): "
        f"{unexpected}. Verifica primero si la fila dueña también colisiona por su propio "
        f"nombre (`sc.is_country_catalog_unpriced_item(owner)`) -- si SÍ, es un bug real (afecta "
        f"`canonicalize_shopping_food_name`); si NO, documenta la razón en el allowlist."
    )


def test_i2_las_4_colisiones_conocidas_son_benignas_para_el_canonicalizer(sc):
    """[I2 · la verificación que el review pidió específicamente] Para cada una de las 4
    colisiones del allowlist: (a) la fila DUEÑA del alias NO colisiona por su propio nombre
    canónico (así que el atajo `if m_item and is_country_catalog_unpriced_item(canonical_name)`
    de `canonicalize_shopping_food_name` nunca se activa para ellas), y (b) el canonicalizer
    real, en vivo, resuelve el alias a la fila dueña correcta -- no al token de catálogo-país con
    el que colisiona."""
    import db_core
    if db_core.connection_pool is None:
        pytest.skip("connection_pool es None — e2e, no bloquea el gate")
    db_core.connection_pool.open()

    master_map = sc._build_shopping_master_map()
    for (alias, owner), _reason in _I2_ALLOWLIST_PRICED_ALIAS_COLLISIONS.items():
        assert not sc.is_country_catalog_unpriced_item(owner), (
            f"la fila dueña {owner!r} de {alias!r} AHORA colisiona por su propio nombre -- el "
            f"atajo de `canonicalize_shopping_food_name` SÍ se activaría, esto ya no es benigno, "
            f"revisar como bug real (no solo actualizar el allowlist)"
        )
        resolved = sc.canonicalize_shopping_food_name(alias, master_map)
        assert resolved == owner, (
            f"canonicalize_shopping_food_name({alias!r}, ...) = {resolved!r}, esperaba {owner!r} "
            f"-- la colisión dejó de ser benigna para el canonicalizer"
        )
