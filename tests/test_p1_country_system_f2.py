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
     + el guard de paridad + el alta-hook contra el catálogo vivo. Incluye un hallazgo
     CONSIDERADO Y RECHAZADO (avena/gluten, colisiona con P1-ALLERGEN-NEGATION-EXCUSE) — ver G1.

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


def test_avena_bare_deliberadamente_no_es_termino_de_gluten(go):
    """[hallazgo CONSIDERADO Y RECHAZADO — no un gap, documentado para que nadie lo reintente]

    Diffeando `condition_rules._ALLERGEN_GLUTEN_SUBS` (vocabulario #4) contra `_ALLERGEN_SYNONYMS`
    se encuentra que la sustitución quirúrgica trata la avena como riesgo de contaminación cruzada
    de gluten — el primer instinto es añadir 'avena' bare al backstop #1 también. Se probó en este
    task y se REVIRTIÓ: rompe `test_p1_allergen_negation_excuse.py::test_avena_certificada_sin_gluten_no_viola`
    + `test_pool_scrub_ya_no_roba_la_avena_sin_gluten`. Razón estructural, no solo dos tests
    rojos: `_ALLERGEN_NEGATION_PREFIX_RX` excusa por PREFIJO (mira hacia atrás desde el match) —
    en "avena certificada sin gluten" la negación SIGUE a 'avena', nunca la precede, así que un
    token bare 'avena' NUNCA podría beneficiarse de esa excusa y volvería a castigar el
    CUMPLIMIENTO que P1-ALLERGEN-NEGATION-EXCUSE cerró (corr=abb71a1d). `condition_rules.py` ya
    resuelve esto con su PROPIA lista `_ALLERGEN_GLUTEN_NEGATIVES` (incluye "sin gluten") antes de
    sustituir — el backstop #1 no tiene ese mecanismo por-término, solo el genérico de negación.

    Este test ancla el estado DECIDIDO: 'avena' NO está en `_ALLERGEN_SYNONYMS['gluten']` (control
    negativo — si alguien la reintenta sin leer este comentario, este test sigue verde pero
    `test_p1_allergen_negation_excuse.py` cae, la misma señal que detuvo este task)."""
    assert "avena" not in [t.lower() for t in go._ALLERGEN_SYNONYMS["gluten"]]
    # Verificación cruzada en vivo: el caso medido sigue sin violar HOY.
    plan = {"days": [{"meals": [{"name": "Desayuno", "ingredients": ["20 g de avena certificada sin gluten"]}]}]}
    assert go._scan_allergen_violations(plan, ["gluten"]) == []


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
    "mariscos": lambda cr: list(cr._ALLERGEN_SHELLFISH_SUBS[0][0]),
    "pescado": lambda cr: list(cr._ALLERGEN_FISH_SUBS[0][0]),
    "soya": lambda cr: [t for sub in cr._ALLERGEN_SOY_SUBS for t in sub[0]],
    "gluten": lambda cr: [t for sub in cr._ALLERGEN_GLUTEN_SUBS for t in sub[0]],
}

# Excepción documentada POR TÉRMINO (mismo mecanismo que `_PARITY_TERM_EXCEPTIONS` en G2): la
# avena (bare + sus 3 compuestos, que #4 lista aparte) es la ÚNICA familia que #4 sustituye pero
# #1 NO puede seguir a ciegas — ver `test_avena_bare_deliberadamente_no_es_termino_de_gluten` (G1)
# para la razón completa (`_ALLERGEN_NEGATION_PREFIX_RX` es solo-prefijo; un token bare 'avena'
# reintroduce el falso-positivo que P1-ALLERGEN-NEGATION-EXCUSE cerró). `condition_rules.py`
# resuelve esto con su propia `_ALLERGEN_GLUTEN_NEGATIVES` antes de sustituir — mecanismo que #1
# no tiene por-término, solo el genérico. Único hueco conocido y ACEPTADO de todo este archivo.
_V4_TERM_EXCEPTIONS = {
    "mariscos": set(), "pescado": set(), "soya": set(),
    "gluten": {"avena", "harina de avena", "hojuelas de avena", "salvado de avena"},
}


@pytest.mark.parametrize("clase_allergen", list(_V4_EXTRACTORS.keys()))
def test_backstop_cubre_los_objetivos_de_sustitucion_de_condition_rules(go, condrules, clase_allergen):
    """[G4 · vocabulario #4] Si `collect_allergen_substitutions` falla en sustituir (bug, texto
    del LLM que no matchea sus tokens estrechos a propósito), `_scan_allergen_violations` es la
    ÚNICA red que queda. Pre-fix esta clase estaba rota para 'gluten' (tostada/macarrón/coditos/
    fideo/tallarín/penne/ravioli/ñoqui/tortilla de harina — sin contar la avena, excepción
    documentada) y 'mariscos'/'pescado' (gamba/arenque)."""
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

# 'Avena'/'Leche de avena' bajo la clase 'gluten': razón DISTINTA de la plant-adjacency de
# arriba — no es que la avena sea segura por ser vegetal, es que `_ALLERGEN_SYNONYMS['gluten']`
# DELIBERADAMENTE no lleva 'avena' bare (ver G1/G4: reintroduciría el falso-positivo que
# P1-ALLERGEN-NEGATION-EXCUSE cerró contra "avena certificada sin gluten"). `condition_rules.py`
# SÍ la trata como riesgo de contaminación cruzada, así que el probe de 'gluten' la sigue
# marcando 'covered' — este set le dice al test que la NO-detección aquí es la decisión, no un gap.
_G5_EXCUSADOS_AVENA_GLUTEN_DECISION = {("gluten", "avena"), ("gluten", "leche de avena")}


@pytest.mark.e2e
def test_backstop_conoce_cada_alimento_peligroso_del_catalogo_vivo():
    """[G5 · el alta-hook, contrato T4 ítem 3] Query read-only a `master_ingredients` (pool
    abierto explícitamente). Para cada clase de seguridad, une los tokens que CUALQUIER
    vocabulario hermano (#2 dieta, #4 sustitución) ya reconoce como peligrosos + los propios de
    #1, y verifica que todo nombre de catálogo que matchee alguno de esos tokens SÍ dispare
    `clinical_backstop_for_meal` para la alergia correspondiente.

    Hallazgo EN VIVO de este task (pre-fix, 206 filas en `master_ingredients`): 'Mejillones'
    (mariscos) y 'Arenque' (pescado) son alimentos catalogados HOY cuyo nombre ya vivía en un
    vocabulario hermano (#2 dieta) pero `_ALLERGEN_SYNONYMS` no los reconocía —
    `clinical_backstop_for_meal` los dejaba pasar en silencio; ambos cerrados. 'Yogur de coco'/
    'Mantequilla de maní' matchean el probe pero son EXCUSA correcta (plant-adjacency), no gap —
    ver `_G5_EXCUSADOS_PLANT_ADJ`. 'Avena'/'Leche de avena' (gluten) SÍ quedan sin backstop, mismo
    criterio que G1/G4 — ver `_G5_EXCUSADOS_AVENA_GLUTEN_DECISION`."""
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
            if clave in _G5_EXCUSADOS_PLANT_ADJ or clave in _G5_EXCUSADOS_AVENA_GLUTEN_DECISION:
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
