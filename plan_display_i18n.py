# backend/plan_display_i18n.py
"""Motor de enriquecimiento de la capa de display i18n del plan.

tooltip-anchor: P1-PLAN-DISPLAY-I18N

Decisión arquitectónica (spec `docs/superpowers/specs/2026-08-19-plan-display-i18n-design.md`):
el plan se GENERA y PERSISTE siempre en español canónico — los nombres de alimentos y platos
son identificadores del sistema (`pantry_names_match`, coherence guard, backstop de alergias
resuelven por esos strings exactos; P1-I18N-DASHBOARD). Este módulo escribe un campo
paralelo de solo-lectura para el frontend:

    meal["_display"][locale] = {"name", "description", "recipe", "ingredients"}

que el frontend consume con fallback campo a campo al original si falta (fase 2, fuera de
este módulo). es-DO jamás lleva `_display` — byte-idéntico a hoy.

[P3-I18N-DISPLAY-DOCSTRING-LEE-DISPLAY · 2026-08-22] Este párrafo decía que el módulo «NUNCA
lee ese campo de vuelta ni condiciona su propia conducta a él». Desde
`P2-DISPLAY-REDESPACHO-SIN-FILTRO` hace exactamente eso, y con razón: `_ya_traducido_*` lee
su propio `_display` para no re-pagar una traducción que ya está — y para exigir que sea
USABLE, no sólo que exista. Sin esa lectura, un display a medias dado por bueno deja esa
comida en español para siempre porque nadie la reintenta.

La frontera real, que sigue intacta, es otra: `_display` **jamás influye en el dato canónico
ni en una decisión del motor**. Ni el generador, ni los guards, ni la resolución de nevera,
ni el backstop clínico lo miran. Lo único que condiciona es si este módulo vuelve a gastar
en traducir. Decirlo mal importa: un lector que crea que la regla es «nadie lo lee» borrará
esa comprobación creyendo que restaura una invariante.

Contrato:
    enrich_plan_display(plan_id, user_id, locale, day_indices=None) -> dict
        {"enriched_meals": int, "skipped": str | None}. Jamás lanza (fail-open TOTAL — spec
        "Global Constraints": cualquier excepción del enriquecimiento es un warning + el plan
        se queda sin `_display` + el frontend cae a español, nunca rompe generación/swap/lectura).

    schedule_plan_display_enrichment(plan_id, user_id, locale, day_indices=None) -> None
        Wrapper fire-and-forget: thread background + dedupe. Los 5 disparadores (Task 2/3)
        llaman a este, no a `enrich_plan_display` directo.

Validación determinista (spec, sección "Contrato de datos"):
    - `recipe` e `ingredients` de salida deben tener la MISMA longitud que el original, en el
      mismo orden (alineados por índice) — si no, el meal ENTERO se descarta (no se persiste
      _display para ese meal).
    - Cada línea de `ingredients` debe contener el nombre canónico español del alimento como
      substring accent-insensitive (extraído del original — ver `_extract_canonical_name`). Si
      la línea traducida lo pierde, SOLO esa línea cae de vuelta al original español (fallback
      per-línea, NO descarta el meal — "un gloss que pierde el identificador es peor que no
      tener gloss"). Si el original no tiene canónico identificable, la línea pasa sin check.

[Fix round 1 · 2026-08-19, review `task-1-review.md`] Tres cambios de contrato sobre Task 1:
    - Lotes: el plan se trocea en lotes de `MEALFIT_PLAN_DISPLAY_I18N_BATCH_DAYS` días por
      llamada LLM (Finding 6) — un solo lote gigante (todo el plan en una llamada) truncaba la
      salida y quemaba el costo sin recuperación parcial ni registro.
    - TOCTOU: `targets` se lee FUERA del row-lock (no se puede sostener un `FOR UPDATE` durante
      los ~segundos que dura la llamada LLM). El mutator de `_persist_batch` compara el nombre
      canónico del meal en esa posición contra el snapshot leído ANTES de escribir — si difiere
      (swap/regenerate-day concurrente), SALTA ese meal (Finding 5). Sin esto, la promesa de la
      spec ("stale display imposible por construcción") era falsa bajo carrera.
    - Lock cross-worker: se libera SIEMPRE en el `finally` (antes solo tenía TTL, Finding 4) y su
      clave incluye los `day_indices` del lote (dos disparadores sobre días distintos del mismo
      plan+locale ya no se bloquean entre sí).

Costo: NUNCA a `api_usage` (cero crédito de usuario) — telemetría a `llm_usage_events` vía
`db.log_llm_usage_event(node="plan_display_i18n")`, emitida INMEDIATAMENTE tras cada `invoke`
exitoso (Finding 7) — antes solo se emitía en el camino feliz completo, dejando ciegos los
modos de fallo post-invoke (JSON roto, 0 meals válidos, persist fallido).

Knobs (auto-registrados en `_KNOBS_REGISTRY` vía `knobs._env_bool/_env_str/_env_float/_env_int`):
    MEALFIT_PLAN_DISPLAY_I18N                default True  — apaga el motor y el attach
                                             del gloss. NO revierte lo ya persistido ni
                                             toca la búsqueda: ver «Qué apaga el kill
                                             switch, y qué NO» en
                                             docs/plan_display_i18n.md
                                             [P2-I18N-KILLSWITCH-NO-REVIERTE].
    MEALFIT_PLAN_DISPLAY_I18N_MODEL          default flash — convención P3-PREVIEW-MODEL-KNOB.
    MEALFIT_PLAN_DISPLAY_I18N_TIMEOUT_S      default 60.0  — timeout del cliente LLM.
    MEALFIT_PLAN_DISPLAY_I18N_BATCH_DAYS     default 4     — TOPE DURO en días por lote
                                                            (clamp 1-30). Ya no es la unidad
                                                            del troceo, ver abajo.
    MEALFIT_PLAN_DISPLAY_I18N_MAX_OUTPUT_TOKENS default 8000 — cota de salida por lote.

[P1-DISPLAY-LOTE-POR-COMIDAS · 2026-08-21] El lote se dimensiona por el TAMAÑO PROYECTADO
de la salida, no por días. Medido sobre las 271 comidas de los 6 planes vivos: el lote
ordinario (4 días × 4-5 platos = 16-20 comidas) proyecta 6.600-8.300 tokens contra el tope
de 8.000, y pasarse no degradaba — la salida se truncaba, el JSON no parseaba y el tramo
entero se descartaba con la llamada ya cobrada. Ahora:

    - `_particionar_targets` reparte por tokens estimados, con `BATCH_DAYS × las comidas
      del día más cargado DE ESTE PLAN` como tope duro. El knob conserva su significado:
      «BATCH_DAYS=1» sigue queriendo decir un día, traiga ese día una comida o seis.
    - `_dividir_lote` + la pila de pendientes: un lote que no parsea se parte en dos y se
      reintenta. Sólo una comida sola que sigue fallando es pérdida definitiva, y entonces
      se cuenta en `targets_perdidos` y se registra a nivel `error`.
    - `_max_invocaciones_por_ciclo` acota el split recursivo.
"""

import json
import logging
import re
import threading
import time
import unicodedata
from typing import Optional

from knobs import _env_bool, _env_str, _env_float, _env_int
from constants import strip_accents
from prompts.chat_agent import _COACH_LANGUAGE_NAMES
from llm_provider import build_chat_llm, DEEPSEEK_FLASH
from langchain_core.messages import SystemMessage
from db import update_plan_data_atomic, log_llm_usage_event, execute_sql_query, execute_sql_write

logger = logging.getLogger(__name__)

# ============================================================
# Knobs
# ============================================================


def _plan_display_i18n_enabled() -> bool:
    return _env_bool("MEALFIT_PLAN_DISPLAY_I18N", True)


def _plan_display_i18n_max_inflight() -> int:
    """Cuántos enriquecimientos pueden estar en vuelo a la vez, en TODO el proceso.

    [P3-I18N-DISPLAY-HILO-SIN-TECHO · 2026-08-22] El dedupe por (plan, idioma) impide dos
    hilos para el mismo par; el cruce entre planes no lo acotaba nada. Cada hilo puede
    vivir 20-29 minutos hablando con un proveedor pago.

    Por debajo de 1 el techo dejaría de tener sentido y apagaría la feature por accidente
    —un cero silencioso es peor que un knob mal puesto—, así que se acota abajo.
    """
    return max(1, _env_int("MEALFIT_PLAN_DISPLAY_I18N_MAX_INFLIGHT", 4))


# `BoundedSemaphore` y no `Semaphore`: si un `release()` de más se colara, el acotado lo
# convierte en un `ValueError` ruidoso en vez de subir el techo en silencio para siempre.
# Un techo que se relaja solo no es un techo.
_INFLIGHT_SEMAPHORE = threading.BoundedSemaphore(_plan_display_i18n_max_inflight())


def _plan_display_i18n_model_name() -> str:
    return _env_str("MEALFIT_PLAN_DISPLAY_I18N_MODEL", DEEPSEEK_FLASH)


def _plan_display_i18n_timeout_s() -> float:
    return _env_float("MEALFIT_PLAN_DISPLAY_I18N_TIMEOUT_S", 60.0)


def should_enrich_locale(locale) -> bool:
    """[Fix round 1 · F10] SSOT exportado del gate de locale — reemplaza el literal
    `locale != "es-DO"` duplicado en los 3 call sites de mutador (swap-persist,
    regenerate-day, chat-modify) que pre-filtran ANTES de despachar el thread
    background (evitar el import+thread cuando es obviamente innecesario; el gate
    real y autoritativo sigue viviendo DENTRO de `enrich_plan_display` /
    `schedule_plan_display_enrichment`, este helper solo espeja esa misma condición
    para que el call site no reimplemente el conocimiento del motor).

    `locale not in _COACH_LANGUAGE_NAMES` YA excluye es-DO (nunca está en ese dict,
    P1-I18N-DASHBOARD) — por eso un simple `!= "es-DO"` en el call site bastaba en la
    práctica, pero duplicaba el conocimiento de "qué es un locale válido" fuera del
    motor. `constants.strip_accents`/`canonicalize_diet_type` ya pagaron esta lección
    (P1-DIET-CANON-SSOT): un 2º/3º/4º lugar que reimplementa la misma regla driftea.
    """
    return isinstance(locale, str) and locale in _COACH_LANGUAGE_NAMES


def _plan_display_i18n_batch_days() -> int:
    """[Finding 6 · fix round 1] Días por llamada LLM. `day_indices=None` (todo
    el plan, disparador 4 de la spec — cambio de idioma con plan ya creado) es
    justo el caso que motivó la feature Y el que más probablemente reventaba:
    ~28 días × 4-5 meals en UNA sola llamada trunca la salida."""
    return _env_int(
        "MEALFIT_PLAN_DISPLAY_I18N_BATCH_DAYS",
        4,
        validator=lambda v: 1 <= v <= 30,
    )


# [P1-DISPLAY-LOTE-POR-COMIDAS · 2026-08-21] Comidas por dia con las que `BATCH_DAYS`
# se traduce a un tope duro de comidas. Es un FALLBACK: cuando el plan esta delante se
# mide sobre el (ver `_comidas_por_dia_del_plan`), porque un tope constante le cambiaria
# el significado al knob —«BATCH_DAYS=1» tiene que seguir queriendo decir «un dia»,
# aunque ese dia traiga una sola comida—. Medido en los planes vivos: 4,00 comidas/dia
# en cinco de los seis y 5,00 en el mas cargado (76a6836d).
_COMIDAS_POR_DIA_TOPE = 5


def _comidas_por_dia_del_plan(days: list, day_indices: list) -> int:
    """El dia mas cargado de los que se van a traducir, con suelo 1.

    Se toma el MAXIMO y no la media a proposito: el tope tiene que acotar el peor lote
    posible, y con la media un plan irregular (2 comidas un dia, 6 el siguiente) daria
    un tope que el dia cargado se salta.
    """
    idx = set(day_indices or [])
    mayor = 0
    for i, d in enumerate(days or []):
        if idx and i not in idx:
            continue
        if isinstance(d, dict) and isinstance(d.get("meals"), list):
            mayor = max(mayor, len(d["meals"]))
    return mayor or _COMIDAS_POR_DIA_TOPE


def _plan_display_i18n_max_output_tokens() -> int:
    return _env_int(
        "MEALFIT_PLAN_DISPLAY_I18N_MAX_OUTPUT_TOKENS",
        8000,
        validator=lambda v: 500 <= v <= 32000,
    )


# [P3-I18N-DISPLAY-KNOBS-PEREZOSOS · 2026-08-22] Los cinco knobs, declarados en el IMPORT.
#
# `knobs._env_*` registra en `_KNOBS_REGISTRY` al ser LLAMADO. Estos cinco viven dentro de
# funciones que sólo corren cuando hay algo que traducir, así que
# `get_knobs_registry_snapshot()` —lo que un operador consulta para saber qué puede tocar sin
# redeploy— no los conocía hasta que la feature se ejecutaba por primera vez. Y esta capa,
# medido el 2026-08-22, se ha ejecutado CINCO veces en toda su historia: en la práctica los
# knobs eran invisibles siempre.
#
# Son cinco lecturas de entorno en el import y NO cachean nada: cada accesor sigue leyendo en
# vivo, que es lo que permite el rollback sin redeploy. Esto sólo los DECLARA.
# [P3-I18N-DISPLAY-KNOBS-TODOS-EN-EL-REGISTRY · 2026-08-23] El bloque se movió al FINAL del
# módulo. Aquí sólo podía enumerar los accesores ya definidos, y por eso se quedó en cinco:
# `_plan_display_i18n_max_inflight` y `_max_locales_display` nacen más abajo. Al final del
# fichero están todos en ámbito, y el bloque deja de depender del orden de definición.


def _circuit_breaker_can_proceed(model_name: str) -> bool:
    """[Finding 10 · fix round 1] Consulta al circuit breaker ANTES de gastar
    threads/latencia en un provider ya caído — mismo patrón que el callsite de
    referencia (`agent.py:4771-4779`, título del chat). Import LOCAL: evitar que
    `graph_orchestrator` (módulo pesado) entre en el costo de import de este
    módulo (contrato verificado: import sin efectos DB/LLM).

    Fail-open: si el CB no se puede resolver (import falla, DB caída), asumimos
    que se puede proceder — un circuit breaker que bloquea por su propia
    indisponibilidad sería peor que no tenerlo.
    """
    try:
        from graph_orchestrator import _get_circuit_breaker
        return _get_circuit_breaker(model_name).can_proceed()
    except Exception as e:
        logger.debug(f"[P1-PLAN-DISPLAY-I18N] circuit breaker check falló (fail-open): {e!r}")
        return True


# ============================================================
# Dedupe: lock in-process + marker cross-worker en app_kv_store
# (mismo patrón que `agent.py::_try_claim_title_lock_cross_worker`)
#
# [Finding 4 · fix round 1] Dos cambios sobre Task 1:
#   1. El marker cross-worker se LIBERA en el `finally` de `enrich_plan_display`
#      (antes solo expiraba por TTL a los 5 min) — con 5 disparadores sobre el
#      mismo (plan_id, locale) el TTL viejo bloqueaba un re-enriquecimiento
#      legítimo (p.ej. swap 3 min después del persist inicial) durante toda la
#      ventana, sin reintento ni alerta.
#   2. La clave incluye los `day_indices` del lote — dos disparadores sobre
#      DÍAS DISTINTOS del mismo plan+locale ya no se bloquean entre sí.
# El set in-process (`_INFLIGHT`) es un pre-filtro barato de proceso (NO
# incluye day_indices a propósito: es solo una optimización para evitar un
# roundtrip a la DB cuando el mismo worker ya tiene ALGO en vuelo para ese
# plan+locale; la autoridad real es el marker cross-worker de abajo).
# ============================================================

_INFLIGHT: set = set()
_INFLIGHT_LOCK = threading.Lock()
_ENRICH_LOCK_TTL_S = 300  # Red de seguridad si el proceso muere ANTES del finally.


def _inflight_key(plan_id: str, locale: str, day_indices) -> tuple:
    """[Ola final · FF-3] Clave del set in-process, CON los días — paridad con la clave
    del KV cross-worker.

    Por qué: el comentario de arriba clasifica `_INFLIGHT` como "solo una optimización"
    cuya autoridad real es el marker cross-worker. Eso vale en un despliegue
    multi-worker; aquí NO: FastAPI + APScheduler + `_chunk_worker` + los handlers de
    swap/expand/chat-modify viven en el MISMO proceso, así que el set in-process era el
    gate DOMINANTE y su clave gruesa `(plan_id, locale)` reimponía exactamente el bloqueo
    que la clave del KV se diseñó para evitar. El perdedor no se encolaba ni se
    reintentaba: devolvía `dedupe_inprocess` y moría ahí, dejando ESOS días en español
    (p.ej. un swap durante el enriquecimiento plan-wide del cambio de idioma — el flujo
    que motivó la feature).

    `day_indices=None` (el pre-filtro del scheduler cuando el caller no declara días)
    produce una clave que JAMÁS matchea una clave del motor: el pre-filtro simplemente
    deja de filtrar ese caso y el gate real de `enrich_plan_display` (que ya normalizó
    contra `len(days)`) decide. Es la asimetría segura: filtrar de menos duplica una
    llamada flash barata; filtrar de más pierde días.
    """
    if day_indices is None:
        return (plan_id, locale, None)
    return (plan_id, locale, tuple(day_indices))


def _enrich_lock_kv_key(plan_id: str, locale: str, day_indices: list) -> str:
    """Determinista: el mismo lote de días (en cualquier orden de entrada,
    `enrich_plan_display` ya normaliza a sorted+dedup antes de llamar aquí)
    produce siempre la misma key."""
    scope = ",".join(str(i) for i in day_indices) if day_indices else "none"
    if len(scope) > 200:
        import hashlib
        scope = "h" + hashlib.sha256(scope.encode("utf-8")).hexdigest()[:16]
    return f"plan_display_enrich:{plan_id}:{locale}:{scope}"


def _try_claim_enrich_lock_cross_worker(plan_id: str, locale: str, day_indices: list) -> bool:
    """Claim atómico cross-worker vía UPSERT en `app_kv_store`. Espejo de
    `agent.py::_try_claim_title_lock_cross_worker`. Best-effort: si la DB no
    responde, retorna True (fail-open) — preferimos duplicar una llamada
    flash barata a bloquear el enriquecimiento por un outage del KV.
    """
    try:
        _now_ts = time.time()
        _kv_key = _enrich_lock_kv_key(plan_id, locale, day_indices)
        result = execute_sql_query(
            """
            INSERT INTO app_kv_store (key, value)
            VALUES (%s, jsonb_build_object('started_at', %s::float))
            ON CONFLICT (key) DO UPDATE SET
                value = jsonb_build_object('started_at', %s::float),
                updated_at = NOW()
            WHERE COALESCE((app_kv_store.value->>'started_at')::float, 0)
                  < %s::float
            RETURNING key
            """,
            (_kv_key, _now_ts, _now_ts, _now_ts - _ENRICH_LOCK_TTL_S),
            fetch_all=True,
        )
        return bool(result)
    except Exception as e:
        logger.warning(
            f"[P1-PLAN-DISPLAY-I18N] claim de lock cross-worker falló (fail-open) "
            f"plan={plan_id} locale={locale}: {e!r}"
        )
        return True


def _release_enrich_lock_cross_worker(plan_id: str, locale: str, day_indices: list) -> None:
    """[Finding 4 · fix round 1] DELETE best-effort del marker — libera el slot
    para el PRÓXIMO disparador legítimo dentro de la ventana TTL en vez de
    obligarlo a esperar hasta 5 min. El TTL se conserva como red de seguridad
    para el caso donde el proceso muere ANTES de llegar a este `finally`."""
    try:
        _kv_key = _enrich_lock_kv_key(plan_id, locale, day_indices)
        execute_sql_write("DELETE FROM app_kv_store WHERE key = %s", (_kv_key,))
    except Exception as e:
        logger.debug(
            f"[P1-PLAN-DISPLAY-I18N] release de lock cross-worker falló (best-effort) "
            f"plan={plan_id} locale={locale}: {e!r}"
        )


# ============================================================
# Lectura del plan (ownership AND user_id, mismo patrón de routers/plans.py)
# ============================================================


def _fetch_plan_data(plan_id: str, user_id: str) -> Optional[dict]:
    # tooltip-anchor: I2 user_id filter (test_p1_plan_display_i18n.py sección "ownership").
    try:
        row = execute_sql_query(
            "SELECT plan_data FROM meal_plans WHERE id = %s AND user_id = %s",
            (plan_id, user_id),
            fetch_one=True,
        )
    except Exception as e:
        logger.warning(f"[P1-PLAN-DISPLAY-I18N] SELECT plan falló plan={plan_id}: {e!r}")
        return None
    if not row:
        return None
    pd = row.get("plan_data")
    if isinstance(pd, str):
        try:
            pd = json.loads(pd)
        except Exception:
            return None
    if not isinstance(pd, dict):
        return None
    return pd


def _fetch_plan_name_column(plan_id: str, user_id: str) -> Optional[str]:
    """[P1-PLAN-TITLE-I18N · 2026-08-20] El nombre del plan vive en la COLUMNA
    `meal_plans.name`, NO dentro de `plan_data`.

    Fase 1c leia el titulo de `plan_data["name"]` y ese campo solo existe si el plan
    fue RENOMBRADO alguna vez (`PATCH /api/plans/{id}/name` escribe la columna Y el
    jsonb). Medido en produccion: `plan_data->>'name'` es NULL en los 8 planes vivos,
    asi que `plan_name_pending` salia siempre None, al LLM nunca se le pedia
    `plan_name` y el bloque que escribe `pd["_display"]` jamas se ejecutaba. La
    funcionalidad estaba INERTE: el Historial en ingles seguia mostrando el titulo en
    espanol mientras los meals SI se traducian (esos si tienen su `_display`).

    Los tests no lo vieron porque sus fixtures traen `plan_data={"name": ...}` — una
    forma que la base de datos no produce. Un fixture que no se parece a produccion
    prueba el codigo contra un mundo que no existe.

    SELECT propio (no ampliar `_fetch_plan_data`) para no cambiarle la firma: los
    tests la monkeypatchean como seam y devolveria una tupla donde esperan un dict.
    Es una query trivial por enriquecimiento, en hilo de fondo.

    tooltip-anchor: I2 user_id filter (test_p1_plan_title_i18n.py).
    """
    try:
        row = execute_sql_query(
            "SELECT name FROM meal_plans WHERE id = %s AND user_id = %s",
            (plan_id, user_id),
            fetch_one=True,
        )
    except Exception as e:
        logger.warning(
            f"[P1-PLAN-DISPLAY-I18N] SELECT name fallo plan={plan_id}: {e!r}")
        return None
    if not row:
        return None
    nombre = row.get("name")
    return nombre if isinstance(nombre, str) and nombre.strip() else None


def _normalize_day_indices(day_indices, days_len: int) -> list:
    """`None` -> todos los días del plan. Lista arbitraria -> dedup + sorted +
    solo enteros reales (nunca bool, que es subclase de int en Python)."""
    if day_indices is None:
        return list(range(days_len))
    try:
        cleaned = {int(i) for i in day_indices if isinstance(i, int) and not isinstance(i, bool)}
    except TypeError:
        return []
    return sorted(cleaned)


# ============================================================
# Extracción del nombre canónico.
#
# [Finding 1 · fix round 1] La v1 de Task 1 devolvía "todo el resto de la
# línea" tras quitar cantidad+unidad — contra el corpus real (formato
# dominante "<cantidad> <unidad> de <alimento> <modificador>") eso arrastraba
# el "de " conector Y los modificadores de estado de cocción ("secas",
# "cocidas", "picadas", "rallada", "al gusto"), que el LLM NUNCA reproduce
# dentro de un gloss porque no son parte del nombre del alimento. El prompt
# pedía "Nombre canónico" y el validador comparaba contra "de Nombre canónico
# modificador" — descartaba ~10 de cada 11 líneas reales con gloss perfecto.
#
# Fix en 3 pasos, medido contra las 11 líneas reales del review (11/11,
# mínimo pactado 10/11 — ver test_canonical_extraction_validates_against_real_corpus):
#   1. Prefijo de cantidad/unidad (igual que v1, + "lonjas/rebanadas/rodajas",
#      que faltaban — day_generator.py:48 las usa).
#   2. UN SOLO conector ("de"/"del"/"la"/"el"/"los"/"las") inmediatamente tras
#      la cantidad/unidad — nombres compuestos como "semillas de linaza" o
#      "Pechuga de pollo" conservan su "de" INTERNO porque solo se quita el
#      PRIMER match.
#   3. Modificador de estado de cocción/preparación al FINAL de la línea
#      ("secas", "cocidas", "picadas", "rallada", "al gusto"...). Los
#      modificadores que SÍ cambian la identidad del alimento ("blanco" en
#      "arroz blanco", "pintas" en "judías pintas", "dominicano" en "Oregano
#      dominicano") NO están en esta lista — se conservan a propósito.
# ============================================================

# [P1-I18N-DISPLAY-CANONICO-PARTITIVO · 2026-08-22] Dos ampliaciones, las dos por lo mismo:
# lo que NO se consume como cantidad acaba DENTRO del «nombre canónico», y entonces el check
# rechaza un gloss correcto y esa línea se queda en español.
#
#   '2 dientes de ajo'          -> canónico 'dientes de ajo'   (debería ser 'ajo')
#   '½ pedazo de ñame (≈150 g)' -> canónico 'pedazo de ñame'   (debería ser 'ñame')
#   '⅓ taza de yogurt griego'   -> la fracción no se consumía siquiera
#
# MEDIDO con una corrida dirigida contra el modelo real sobre días reales: fr-FR tenía 8 de
# 186 líneas de ingrediente caídas al español (4,3 %), y SEIS de esas ocho son de estas dos
# clases. El ajo aparece en casi todos los platos salados, así que además son siempre las
# MISMAS líneas: el usuario ve su receta en francés con una de cada ~20 líneas en español, y
# siempre la del ajo. Es justo el «mitad francés mitad español en la misma pantalla» que el
# fallback per-línea existe para evitar, produciéndose por un defecto del extractor.
#
#   (1) Fracciones vulgares COMPLETAS y guiones de rango. Antes sólo ½¼¾, así que '⅓' y el
#       '–' de '1–2 dientes' bloqueaban el prefijo entero.
#   (2) Los sustantivos PARTITIVOS entran en la lista de unidades. No es una categoría nueva:
#       la lista ya contenía a sus hermanos 'lonjas/rebanadas/rodajas/lascas', que son
#       exactamente lo mismo (una porción contable de un alimento que no es el alimento).
#
# NO se toca el `\b` del final ni el orden: la lección de alternación-no-longest-match que
# documenta el comentario de abajo sigue viva, y los partitivos nuevos comparten prefijo
# entre sí ('pedazo'/'pedazos') igual que 'unidad'/'unidades'.
_QTY_UNIT_PREFIX_RE = re.compile(
    r"^\s*[\d.,/½¼¾⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞\s–—-]*\s*"
    r"(kg|kilos?|g|gr|gramos?|ml|mls?|l|litros?|tazas?|taza|cditas?|cdtas?|cdas?|cucharadas?|"
    r"cucharaditas?|unidad(?:es)?|unids?|piezas?|pza|oz|onzas?|lb|lbs|libras?|"
    r"lonjas?|rebanadas?|rodajas?|lascas?|"
    # Partitivos: porción contable que NO es el alimento. Mismo papel que 'rodajas'.
    r"dientes?|pedazos?|tortas?|ramitas?|ramas?|hojas?|tallos?|filetes?|"
    r"pu[ñn]ados?|gajos?|cabezas?|latas?|paquetes?|sobres?|potes?|fundas?|manojos?)?\b\s*",
    # [Finding 1 fix, iteración 2] `\b` tras el grupo de unidad: sin esto la
    # alternativa CORTA "l"/"g"/"lb" (que aparece antes en la lista por orden
    # alfabético natural) consume solo el prefijo de una palabra más larga
    # ("l" de "lonjas" -> deja "onjas de queso"), el mismo defecto de
    # alternación-no-longest-match que ya mordió a "unidades?" vs "unid" en
    # la ronda anterior. `\b` fuerza backtracking hacia la alternativa que
    # consume la PALABRA completa.
    re.IGNORECASE,
)
_LEADING_LINKER_RE = re.compile(r"^(?:de|del|la|el|los|las)\s+", re.IGNORECASE)
_PARENTHETICAL_RE = re.compile(r"\([^)]*\)")
_TRAILING_MODIFIER_RE = re.compile(
    r"\s+(?:al gusto"
    r"|en cubos|en trozos|en rodajas|en tiras|en juliana"
    r"|secas?|secos?"
    r"|cocidas?|cocidos?"
    r"|picadas?|picados?"
    r"|ralladas?|rallados?"
    r"|molidas?|molidos?"
    r"|trituradas?|triturados?"
    r"|cortadas?|cortados?"
    r"|deshuesadas?|deshuesados?"
    r"|desmenuzadas?|desmenuzados?"
    r"|peladas?|pelados?"
    r"|lavadas?|lavados?)\s*$",
    re.IGNORECASE,
)


def _extract_canonical_name(ingredient_line: str) -> str:
    if not isinstance(ingredient_line, str):
        return ""
    s = ingredient_line.strip()
    if not s:
        return ""
    s = _QTY_UNIT_PREFIX_RE.sub("", s, count=1).strip()
    s = _LEADING_LINKER_RE.sub("", s, count=1).strip()
    s = _PARENTHETICAL_RE.sub("", s).strip()
    s = _TRAILING_MODIFIER_RE.sub("", s).strip()
    return s


# ============================================================
# Vocabulario cerrado de las recetas (P1-DISPLAY-VOCAB-CERRADO · 2026-08-21)
# ============================================================

# Un paso de receta NO es prosa lisa: empieza por una etiqueta de seccion, o es una
# ANOTACION en vez de una accion. Los tres parsers del frontend —`RecipesView.jsx`,
# `MobileRecipes.jsx`, `utils/recipeSteps.js`— casan ESPANOL LITERAL, asi que una
# etiqueta traducida deja el paso sin reconocer.
#
# MEDIDO sobre los 1.904 pasos vivos (2026-08-21): 1.816 llevan una de estas marcas
# —el 95,4 %—; 599 «Montaje», 598 «Mise en place», 484 «Toque de fuego» y 135
# anotaciones. Y lo caro no es el formato: una anotacion que pierde su etiqueta pasa a
# NUMERARSE como accion de cocina, que es el defecto que `P2-RECIPE-NOTES-NOT-STEPS`
# cerro en su dia — «Nota del nutricionista» convertida en «Step 2».
#
# El criterio es el mismo que ya rige para los nombres de alimento: esto es un
# IDENTIFICADOR, no prosa. Se conserva literal en el DATO y se traduce en la PANTALLA.
#
# UN SOLO SITIO, a proposito: la leccion de P1-DIET-CANON-SSOT, donde tres tablas a mano
# driftearon y la del filtro olvido 'vegetariana'. Si anades una etiqueta aqui, hay que
# anadirla tambien a los parsers del frontend Y al catalogo de traduccion — el test
# `test_p1_display_vocab_cerrado.py` verifica las dos puntas.
#
# Los patrones replican EXACTAMENTE los del frontend, incluida la tolerancia a emoji
# inicial y la variante sin «El» de «Toque de Fuego». El `:` va PEGADO a proposito: la
# tipografia francesa emite «Mise en place : » con espacio delante y el parser no lo
# reconoce, asi que esa forma tiene que contar como PERDIDA, no como conservada.
_VOCAB_CERRADO = (
    ("mise_en_place", re.compile(r"^mise en place:", re.IGNORECASE)),
    ("toque_de_fuego", re.compile(r"^(el\s+)?toque de fuego:", re.IGNORECASE)),
    ("montaje", re.compile(r"^montaje:", re.IGNORECASE)),
    ("nota_nutricionista", re.compile(r"nota del nutricionista", re.IGNORECASE)),
    ("seguridad_alimentaria", re.compile(r"seguridad alimentaria\s*:", re.IGNORECASE)),
    ("porciones_ajustadas", re.compile(r"ajustamos ligeramente las porciones",
                                       re.IGNORECASE)),
)

# Los parsers limpian la cabeza antes de casar (un paso puede venir con emoji delante),
# asi que aqui se limpia igual: comparar sobre la linea cruda dejaria fuera las 62
# «🔬 Nota del nutricionista» del corpus.
_CABEZA_NO_ALFANUM = re.compile(r"^[^\w\u00C0-\u024F]+", re.UNICODE)


def _marca_de_vocab_cerrado(linea) -> Optional[str]:
    """Devuelve la marca de vocabulario cerrado de esta linea, o `None`.

    Sobre los primeros 80 caracteres, que es la misma ventana que usa
    `isRecipeAnnotation` — buscar «nota del nutricionista» en la linea entera marcaria
    un paso de cocina que la mencione de pasada.
    """
    if not isinstance(linea, str) or not linea.strip():
        return None
    cabeza = _CABEZA_NO_ALFANUM.sub("", linea.strip())[:80]
    for marca, rx in _VOCAB_CERRADO:
        if rx.search(cabeza):
            return marca
    return None


def _conserva_el_vocab_cerrado(original: str, traducido: str) -> bool:
    """¿La traduccion conserva la marca que traia el original?

    Si el original no lleva marca, no hay nada que proteger y la linea pasa — el 4,6 %
    de pasos que son prosa lisa se traduce y punto.
    """
    marca = _marca_de_vocab_cerrado(original)
    if marca is None:
        return True
    return _marca_de_vocab_cerrado(traducido) == marca


# ============================================================
# Directivas de idioma NATIVAS por locale + prompt (UN lote por llamada)
#
# [Finding 3 · fix round 1] La v1 de Task 1 escribía la directiva ENTERA en
# español y decía el literal "English gloss" a los 4 locales — contradictorio
# para pt-BR/fr-FR/it-IT. Lección ya pagada dos veces el 2026-08-18
# (P1-COACH-LANGUAGE-NATIVE): una directiva escrita en español pidiendo otro
# idioma es la señal más débil posible; se reescribe EN el idioma destino,
# instrucción Y demostración (el ejemplo de gloss) a la vez. Mismo patrón que
# `prompts/chat_agent.py::_TITLE_LANGUAGE_DIRECTIVES`.
#
# El bloque de DATOS originales (nombres/descripciones/recetas del plan) se
# mantiene en español — eso es correcto, es lo que hay que traducir.
# ============================================================

_DISPLAY_LANGUAGE_DIRECTIVES = {
    "en-US": (
        "Translate these Dominican meal-plan dishes into English, for the USER to read. "
        "The system keeps operating in canonical Spanish internally — this is ONLY a "
        "parallel display layer.\n\n"
        "STRICT RULES:\n"
        "1. Write 'name'/'description'/'recipe' EXCLUSIVELY in English. EXCEPTION inside "
        "'ingredients': each line carries the food name in the format \"English gloss "
        "(Nombre canónico en español)\" — example: \"30 g dried red beans (Habichuelas "
        "rojas)\". The Spanish canonical name MUST appear literally, untranslated, exactly "
        "as in the original (it is a system identifier).\n"
        "2. The output arrays 'recipe' and 'ingredients' MUST have EXACTLY the same number "
        "of elements as the original, in the SAME order (aligned by index).\n"
        "3. Some 'recipe' lines start with a SECTION label or are a NOTE, not a cooking action. These labels are system identifiers, exactly like the canonical food name: copy them VERBATIM in Spanish, with the same punctuation and no space before the colon — \"Mise en place:\", \"El Toque de Fuego:\", \"Montaje:\", \"Nota del nutricionista:\", \"Seguridad alimentaria:\". Translate only the text AFTER the label. The app renders the label in English on screen; a translated label makes the app show a nutritionist note as if it were a numbered cooking step.\n"
        "4. Reply with ONLY valid JSON, no markdown, no text outside the JSON, with this "
        "exact contract:\n"
        '{"meals":[{"i":0,"name":"...","description":"...","recipe":["...","..."],'
        '"ingredients":["...","..."]}]}'
    ),
    "pt-BR": (
        "Traduza estes pratos de um plano alimentar dominicano para o Português, para "
        "LEITURA do usuário. O sistema continua operando em espanhol canônico internamente "
        "— isto é APENAS uma camada de exibição paralela.\n\n"
        "REGRAS ESTRITAS:\n"
        "1. Escreva 'name'/'description'/'recipe' EXCLUSIVAMENTE em Português. EXCEÇÃO "
        "dentro de 'ingredients': cada linha traz o nome do alimento no formato \"gloss em "
        "português (Nome canônico em espanhol)\" — exemplo: \"30 g feijão vermelho seco "
        "(Habichuelas rojas)\". O nome canônico em espanhol DEVE aparecer literalmente, sem "
        "traduzir, exatamente como no original (é um identificador do sistema).\n"
        "2. Os arrays 'recipe' e 'ingredients' de saída DEVEM ter EXATAMENTE a mesma "
        "quantidade de elementos que o original, na MESMA ordem (alinhados por índice).\n"
        "3. Algumas linhas de 'recipe' comecam com um rotulo de SECAO ou sao uma NOTA, nao uma acao de cozinha. Esses rotulos sao identificadores do sistema, exatamente como o nome canonico do alimento: copie-os LITERALMENTE em espanhol, com a mesma pontuacao e sem espaco antes dos dois-pontos — \"Mise en place:\", \"El Toque de Fuego:\", \"Montaje:\", \"Nota del nutricionista:\", \"Seguridad alimentaria:\". Traduza somente o texto DEPOIS do rotulo. O aplicativo exibe o rotulo em portugues na tela; um rotulo traduzido faz o app mostrar uma nota do nutricionista como se fosse um passo numerado.\n"
        "4. Responda APENAS com JSON válido, sem markdown, sem texto fora do JSON, com este "
        "contrato exato:\n"
        '{"meals":[{"i":0,"name":"...","description":"...","recipe":["...","..."],'
        '"ingredients":["...","..."]}]}'
    ),
    "fr-FR": (
        "Traduis ces plats d'un plan alimentaire dominicain en Français, pour la LECTURE de "
        "l'utilisateur. Le système continue de fonctionner en espagnol canonique en interne "
        "— ceci est SEULEMENT une couche d'affichage parallèle.\n\n"
        "RÈGLES STRICTES :\n"
        "1. Rédige 'name'/'description'/'recipe' EXCLUSIVEMENT en Français. EXCEPTION à "
        "l'intérieur de 'ingredients' : chaque ligne porte le nom de l'aliment au format "
        "« gloss en français (Nom canonique en espagnol) » — exemple : « 30 g de haricots "
        "rouges secs (Habichuelas rojas) ». Le nom canonique espagnol DOIT apparaître "
        "littéralement, sans traduction, exactement comme dans l'original (c'est un "
        "identifiant du système).\n"
        "2. Les tableaux 'recipe' et 'ingredients' de sortie DOIVENT avoir EXACTEMENT le "
        "même nombre d'éléments que l'original, dans le MÊME ordre (alignés par indice).\n"
        "3. Certaines lignes de 'recipe' commencent par une etiquette de SECTION ou sont une NOTE, pas une action de cuisine. Ces etiquettes sont des identifiants du systeme, exactement comme le nom canonique de l'aliment : recopie-les LITTERALEMENT en espagnol, avec la meme ponctuation et SANS espace avant les deux-points — \"Mise en place:\", \"El Toque de Fuego:\", \"Montaje:\", \"Nota del nutricionista:\", \"Seguridad alimentaria:\". Attention : la typographie francaise mettrait une espace avant les deux-points ; ici il ne faut PAS. Traduis uniquement le texte APRES l'etiquette. L'application affiche l'etiquette en francais a l'ecran ; une etiquette traduite fait afficher une note du nutritionniste comme une etape numerotee.\n"
        "4. Réponds UNIQUEMENT avec du JSON valide, sans markdown, sans texte hors du JSON, "
        "avec ce contrat exact :\n"
        '{"meals":[{"i":0,"name":"...","description":"...","recipe":["...","..."],'
        '"ingredients":["...","..."]}]}'
    ),
    "it-IT": (
        "Traduci questi piatti di un piano alimentare dominicano in Italiano, per la "
        "LETTURA dell'utente. Il sistema continua a operare in spagnolo canonico "
        "internamente — questo è SOLO uno strato di visualizzazione parallelo.\n\n"
        "REGOLE RIGIDE:\n"
        "1. Scrivi 'name'/'description'/'recipe' ESCLUSIVAMENTE in Italiano. ECCEZIONE "
        "dentro 'ingredients': ogni riga porta il nome dell'alimento nel formato \"gloss in "
        "italiano (Nome canonico in spagnolo)\" — esempio: \"30 g di fagioli rossi secchi "
        "(Habichuelas rojas)\". Il nome canonico spagnolo DEVE apparire letteralmente, "
        "senza tradurlo, esattamente come nell'originale (è un identificatore del "
        "sistema).\n"
        "2. Gli array 'recipe' e 'ingredients' in uscita DEVONO avere ESATTAMENTE lo stesso "
        "numero di elementi dell'originale, nello STESSO ordine (allineati per indice).\n"
        "3. Alcune righe di 'recipe' iniziano con un'etichetta di SEZIONE o sono una NOTA, non un'azione di cucina. Queste etichette sono identificatori di sistema, esattamente come il nome canonico dell'alimento: copiale ALLA LETTERA in spagnolo, con la stessa punteggiatura e senza spazio prima dei due punti — \"Mise en place:\", \"El Toque de Fuego:\", \"Montaje:\", \"Nota del nutricionista:\", \"Seguridad alimentaria:\". Traduci solo il testo DOPO l'etichetta. L'app mostra l'etichetta in italiano sullo schermo; un'etichetta tradotta fa comparire una nota del nutrizionista come se fosse un passo numerato.\n"
        "4. Rispondi SOLO con JSON valido, senza markdown, senza testo fuori dal JSON, con "
        "questo contratto esatto:\n"
        '{"meals":[{"i":0,"name":"...","description":"...","recipe":["...","..."],'
        '"ingredients":["...","..."]}]}'
    ),
}

_DISPLAY_DATA_HEADER = {
    "en-US": "ORIGINAL DISHES (Spanish — do not translate food names outside the "
    "ingredients gloss):",
    "pt-BR": "PRATOS ORIGINAIS (espanhol — não traduza os nomes dos alimentos fora do "
    "gloss de ingredients):",
    "fr-FR": "PLATS ORIGINAUX (espagnol — ne traduis pas les noms des aliments en dehors "
    "du gloss d'ingredients) :",
    "it-IT": "PIATTI ORIGINALI (spagnolo — non tradurre i nomi degli alimenti al di fuori "
    "del gloss di ingredients):",
}

_JSON_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


# ============================================================
# [P1-PLAN-DISPLAY-I18N · fase 1c] Nombre del PLAN — MISMA llamada LLM que los
# meals, un campo más en el prompt/contrato JSON. Addendum NATIVO por locale
# (misma lección P1-COACH-LANGUAGE-NATIVE que `_DISPLAY_LANGUAGE_DIRECTIVES`:
# una instrucción en español pidiendo otro idioma es la señal más débil
# posible), apendeado a la directiva SOLO cuando hay un plan_name que
# traducir — un plan sin nombre nunca ve este bloque.
# ============================================================

_PLAN_NAME_ADDENDUM = {
    "en-US": (
        "\n4. The data below may include an extra line \"PLAN NAME: <text>\". If present, "
        "ALSO translate it into English and add it as a top-level \"plan_name\" key in the "
        "SAME JSON reply (sibling of \"meals\"): "
        '{"meals":[...],"plan_name":"..."}. '
        "Any food name that appears inside the plan title stays in Spanish canonical form "
        "(same rule as the ingredients gloss) — translate only the surrounding descriptive/"
        "creative text. If there is no \"PLAN NAME\" line, omit the \"plan_name\" key entirely."
    ),
    "pt-BR": (
        "\n4. Os dados abaixo podem incluir uma linha extra \"PLAN NAME: <texto>\". Se "
        "presente, TAMBÉM traduza para o Português e adicione como uma chave de nível "
        "superior \"plan_name\" na MESMA resposta JSON (irmã de \"meals\"): "
        '{"meals":[...],"plan_name":"..."}. '
        "Qualquer nome de alimento que apareça dentro do título do plano permanece na forma "
        "canônica em espanhol (mesma regra do gloss de ingredients) — traduza apenas o texto "
        "descritivo/criativo ao redor. Se não houver linha \"PLAN NAME\", omita a chave "
        "\"plan_name\" completamente."
    ),
    "fr-FR": (
        "\n4. Les données ci-dessous peuvent inclure une ligne supplémentaire "
        "« PLAN NAME : <texte> ». Si présente, traduis-la AUSSI en Français et ajoute-la "
        "comme clé de premier niveau « plan_name » dans la MÊME réponse JSON (sœur de "
        "« meals ») : "
        '{"meals":[...],"plan_name":"..."}. '
        "Tout nom d'aliment apparaissant dans le titre du plan reste en forme canonique "
        "espagnole (même règle que le gloss d'ingredients) — traduis seulement le texte "
        "descriptif/créatif environnant. S'il n'y a pas de ligne « PLAN NAME », omets "
        "complètement la clé « plan_name »."
    ),
    "it-IT": (
        "\n4. I dati sottostanti possono includere una riga extra \"PLAN NAME: <testo>\". "
        "Se presente, traducila ANCHE in Italiano e aggiungila come chiave di primo livello "
        "\"plan_name\" nella STESSA risposta JSON (sorella di \"meals\"): "
        '{"meals":[...],"plan_name":"..."}. '
        "Qualsiasi nome di alimento che appare nel titolo del piano resta nella forma "
        "canonica spagnola (stessa regola del gloss di ingredients) — traduci solo il testo "
        "descrittivo/creativo circostante. Se non c'è una riga \"PLAN NAME\", ometti "
        "completamente la chiave \"plan_name\"."
    ),
}


# [P1-INSIGHTS-I18N · 2026-08-20] El RAZONAMIENTO del plan (`plan_data.insights`) —
# el panel «Diagnóstico / Plan de Acción / Tip del Chef». Los TITULOS ya pasaban por
# `t()`; el CUERPO lo escribe el LLM y se quedaba en español con la app en inglés.
#
# Va en la MISMA llamada que los meals y el nombre, como un campo más del contrato JSON:
# una llamada extra por plan seria pagar dos veces por el mismo lote.
#
# `insights` es un ARRAY ALINEADO POR INDICE, así que su contrato es el de `recipe`, no
# el de `plan_name`: misma longitud y mismo orden, o se descarta ENTERO. El panel rotula
# cada entrada por posicion (0=Diagnóstico, 1=Plan de Acción, 2=Tip del Chef), de modo
# que una traduccion con un elemento de menos no seria «peor texto»: pondria el consejo
# del chef bajo el titulo de diagnostico.
_INSIGHTS_ADDENDUM = {
    "en-US": (
        "\n5. The data below may include an \"INSIGHTS\" block with numbered lines. If "
        "present, ALSO translate every line into English and add them as a top-level "
        "\"insights\" array in the SAME JSON reply, in the SAME ORDER and with the SAME "
        "NUMBER of items: {\"meals\":[...],\"insights\":[\"...\",\"...\"]}. "
        "Food names inside stay in Spanish canonical form. If there is no \"INSIGHTS\" "
        "block, omit the \"insights\" key entirely."
    ),
    "pt-BR": (
        "\n5. Os dados abaixo podem incluir um bloco \"INSIGHTS\" com linhas numeradas. Se "
        "presente, TAMBÉM traduza cada linha para o Português e adicione-as como um array "
        "de nível superior \"insights\" na MESMA resposta JSON, na MESMA ORDEM e com a "
        "MESMA QUANTIDADE de itens. Nomes de alimentos permanecem em espanhol canônico. "
        "Se não houver bloco \"INSIGHTS\", omita a chave \"insights\"."
    ),
    "fr-FR": (
        "\n5. Les données ci-dessous peuvent inclure un bloc « INSIGHTS » avec des lignes "
        "numérotées. Si présent, traduis AUSSI chaque ligne en français et ajoute-les "
        "comme un tableau de premier niveau « insights » dans la MÊME réponse JSON, dans "
        "le MÊME ORDRE et avec le MÊME NOMBRE d'éléments. Les noms d'aliments restent en "
        "espagnol canonique. S'il n'y a pas de bloc « INSIGHTS », omets la clé « insights »."
    ),
    "it-IT": (
        "\n5. I dati qui sotto possono includere un blocco \"INSIGHTS\" con righe numerate. "
        "Se presente, traduci ANCHE ogni riga in italiano e aggiungile come array di primo "
        "livello \"insights\" nella STESSA risposta JSON, nello STESSO ORDINE e con lo "
        "STESSO NUMERO di elementi. I nomi degli alimenti restano in spagnolo canonico. "
        "Se non c'è il blocco \"INSIGHTS\", ometti del tutto la chiave \"insights\"."
    ),
}


def _insights_already_translated(plan_data, locale) -> bool:
    """True si `plan_data["_display"][locale]["insights"]` ya existe y esta bien formado.

    Evita pagar la traduccion del mismo razonamiento en cada enriquecimiento. No mira la
    LONGITUD contra el original a proposito: si el plan se regenera con otro numero de
    insights, el guard TOCTOU de `_persist_batch` lo detecta y descarta -- duplicar aqui
    esa comprobacion seria una segunda regla que puede drifear de la primera.
    """
    disp = plan_data.get("_display") if isinstance(plan_data, dict) else None
    if not isinstance(disp, dict):
        return False
    entrada = disp.get(locale)
    if not isinstance(entrada, dict):
        return False
    ya = entrada.get("insights")
    if not (isinstance(ya, list) and bool(ya) and all(
            isinstance(x, str) and x.strip() for x in ya)):
        return False
    # [P1-I18N-DISPLAY-INSIGHTS-SIN-DEFENSA-DE-ECO · 2026-08-23] Un eco YA PERSISTIDO no
    # cuenta como traducido. Sin esto, los planes que guardaron el español antes de que
    # existiera la defensa de `_validate_insights` no se repararían nunca: el gate diría
    # «ya está» en cada ciclo. Es la mitad que `_plan_name_already_translated` ya tenía y
    # que a este le faltaba — el mismo par de funciones, la misma defensa, un campo menos.
    original = plan_data.get("insights") if isinstance(plan_data, dict) else None
    if isinstance(original, list) and len(original) == len(ya) and ya and all(
            isinstance(o, str) and _eco_del_original(x, o) for x, o in zip(ya, original)):
        return False
    return True


def _validate_insights(value, original) -> Optional[list]:
    """Contrato de ARRAY ALINEADO (el de `recipe`), no el de `plan_name`.

    Misma longitud y mismo orden o se descarta ENTERO: el panel rotula por POSICION, asi
    que un elemento de menos no degrada el texto -- mueve el «Tip del Chef» bajo el
    titulo «Diagnostico». Fail-open: cualquier forma inesperada devuelve None y el panel
    se queda en espanol, que es correcto aunque no sea lo pedido.
    """
    if not isinstance(original, list) or not original:
        return None
    if not isinstance(value, list) or len(value) != len(original):
        return None
    fuera = []
    for linea in value:
        if not isinstance(linea, str) or not linea.strip():
            return None
        fuera.append(linea.strip())
    # [P1-I18N-DISPLAY-INSIGHTS-SIN-DEFENSA-DE-ECO · 2026-08-23] Defensa contra el ECO: que
    # el modelo devuelva el español tal cual. Los otros dos campos que escribe ESTA MISMA
    # llamada —el contenido de la comida y el nombre del plan— la ganaron el 21 y el 23 de
    # agosto; los insights se quedaron fuera, y son el único de los tres que, una vez
    # persistido el eco, NO se reintenta jamás: `_insights_already_translated` pasa a decir
    # «ya está» y el panel se queda en español para siempre.
    #
    # DOS SEÑALES, no una. Una línea puede coincidir legítimamente con su original —un
    # tecnicismo, un nombre propio— y tumbar el lote por eso tiraría traducciones buenas. Se
    # descarta cuando TODAS son eco, que es la firma de «devolvió el original». Mismo
    # criterio que `_validate_and_build_display` usa para la comida.
    if fuera and all(_eco_del_original(a, b) for a, b in zip(fuera, original)):
        return None
    return fuera


def _build_prompt(targets: list, locale: str, plan_name: Optional[str] = None,
                  insights: Optional[list] = None) -> str:
    directive = _DISPLAY_LANGUAGE_DIRECTIVES.get(locale)
    header = _DISPLAY_DATA_HEADER.get(locale)
    if directive is None or header is None:
        # `enrich_plan_display` ya descarta locales fuera de `_COACH_LANGUAGE_NAMES`
        # antes de llegar aquí — este fallback es solo para que `_build_prompt` no
        # reviente si algún día se llama aislado con un locale sin directiva nativa.
        idioma = _COACH_LANGUAGE_NAMES.get(locale, locale)
        directive = (
            f"Translate into {idioma}. Reply with ONLY valid JSON matching the meals "
            f"contract: "
            '{"meals":[{"i":0,"name":"...","description":"...","recipe":["..."],'
            '"ingredients":["..."]}]}'
        )
        header = "ORIGINAL DISHES (Spanish):"

    if plan_name:
        directive = f"{directive}{_PLAN_NAME_ADDENDUM.get(locale, '')}"
    if insights:
        directive = f"{directive}{_INSIGHTS_ADDENDUM.get(locale, '')}"

    lines = []
    if plan_name:
        lines.append(f"PLAN NAME: {plan_name}")
    if insights:
        lines.append("INSIGHTS:")
        for ins_idx, ins in enumerate(insights):
            lines.append(f"  [{ins_idx}] {ins}")
    for i, t in enumerate(targets):
        lines.append(f"{i}. NAME: {t['name']}")
        lines.append(f"   DESCRIPTION: {t['description']}")
        lines.append("   RECIPE:")
        for step_idx, step in enumerate(t["recipe"]):
            lines.append(f"     [{step_idx}] {step}")
        lines.append("   INGREDIENTS:")
        for ing_idx, ing in enumerate(t["ingredients"]):
            lines.append(f"     [{ing_idx}] {ing}")
    meals_block = "\n".join(lines)

    return f"{directive}\n\n{header}\n{meals_block}"


def _plan_name_already_translated(
    plan_data: dict, locale: str, original: Optional[str] = None
) -> bool:
    """[fase 1c] True si `plan_data["_display"][locale]["name"]` ya existe y SIRVE.

    Evita retraducir el nombre del plan en CADA disparador (swap/regenday/chatmod pueden
    llamar a `enrich_plan_display` decenas de veces sobre el mismo plan) — el rename manual
    (`api_rename_plan`) popea esta key (`- '_display'` a nivel plan), así que un rename real
    SÍ dispara una retraducción en el próximo enriquecimiento.

    [P1-DISPLAY-ECO-PERSISTIDO · 2026-08-23] Un ECO no cuenta como traducido.

    `P2-DISPLAY-ECO-NOMBRE` impide que un nombre devuelto SIN traducir se PERSISTA. Pero
    este gate —el que decide si hay algo que reintentar— sólo miraba que la clave existiera
    y no estuviera vacía. O sea que un eco que entró por cualquier otra vía (un plan
    anterior al fix, un camino que no pase por `_validate_plan_name`) queda **permanente**:
    el gate dice «ya está» y nadie lo reintenta jamás.

    No es hipotético. Medido el 2026-08-23 en el ÚNICO plan de producción con `_display`:

        en-US -> "Strong Flavor, Life in Balance"      (traducido)
        pt-BR -> "Sabor Forte, Vida em Equilíbrio"     (traducido)
        fr-FR -> "Sazón Fuerte, Vida en Equilibrio"    ← el ESPAÑOL, tal cual

    Un eco persistido es peor que una ausencia: la ausencia se reintenta sola.

    `original` es opcional para no romper llamadas viejas, pero sin él la comprobación se
    degrada a la de antes — pásalo siempre que lo tengas.
    """
    disp = plan_data.get("_display") if isinstance(plan_data, dict) else None
    if not isinstance(disp, dict):
        return False
    entry = disp.get(locale)
    if not (isinstance(entry, dict)
            and isinstance(entry.get("name"), str)
            and entry.get("name").strip()):
        return False
    if isinstance(original, str) and original.strip():
        # Mismo comparador que usa el validador al persistir: tolerante a caja y acentos,
        # porque «SAZON fuerte» tampoco es una traducción.
        return not _eco_del_original(entry["name"], original)
    return True


def _eco_del_original(a: str, b: str) -> bool:
    """¿`a` es el mismo texto que `b`, ignorando caja, acentos y espacio sobrante?

    [P2-DISPLAY-ECO-NOMBRE · 2026-08-21] Tolerante a propósito: «HABICHUELAS
    guisadas» tampoco es una traducción. Se normaliza con NFKD + descarte de
    combinantes, que es la forma barata de comparar sin acentos sin arrastrar
    dependencias.
    """
    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return " ".join(s.lower().split())
    return _norm(a) == _norm(b)


def _validate_plan_name(value, original: Optional[str] = None) -> Optional[str]:
    """Sin check de canónico (a diferencia de `_validate_and_build_display` para
    ingredients): el nombre del plan es texto creativo del LLM sin un alimento
    identificable de forma determinista — se confía en la instrucción del
    addendum nativo. `None`/no-string/vacío-tras-strip -> None (omitido, jamás
    error, fail-open total del motor).

    [P2-DISPLAY-ECO-NOMBRE · 2026-08-21] Y un ECO tampoco vale. Un LLM que devuelve
    el nombre SIN traducir —lo más común cuando es un plato criollo que no sabe
    traducir— pasaba esta validación, se persistía como `_display[locale].name`, y a
    partir de ahí el gate de «¿ya está traducido?» respondía SÍ: el nombre se quedaba
    en español para siempre y nadie lo reintentaba.

    Devolver `None` para un nombre que legítimamente no cambia entre idiomas (una
    marca, un nombre propio) es CORRECTO y no una pérdida: significa «no hay
    traducción que aportar», el motor cae al español, y el usuario ve exactamente lo
    mismo que vería con el eco. Lo que se gana es que el gate deja de mentir.

    `original` es opcional para no romper a los llamadores que no lo pasen.
    """
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v:
        return None
    if isinstance(original, str) and original.strip() and _eco_del_original(v, original):
        return None
    return v


def _parse_json_response(raw: str) -> Optional[dict]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    cleaned = _JSON_CODE_FENCE_RE.sub("", raw).strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


# ============================================================
# Recolección de targets (SIN row-lock — ver aviso TOCTOU en `_persist_batch`)
# ============================================================


def _fingerprint_lines(value) -> list:
    """[Ola final · FF-2] Normalización SSOT de los dos arrays que `_display[locale]`
    espeja por índice (`ingredients`, `recipe`). La usan LAS DOS puntas: el snapshot
    (`_collect_targets`) y la verificación de identidad del mutator (`_persist_batch`),
    así que la comparación es simétrica por construcción — si divergieran, un `recipe`
    legacy de tipo string (que aquí colapsa a `[]`, igual que antes) contaría como
    "cambió" en cada persist y nada se escribiría nunca.

    La COPIA (`list(value)`) es load-bearing, no cosmética: los re-cuantizadores
    mutan `ingredients` IN-PLACE (`ings[i] = new_ing`). Si el snapshot guardara la
    MISMA lista del meal, la mutación viajaría también al snapshot y la comparación
    de FF-2 pasaría siempre — el guard nacería muerto.
    """
    return list(value) if isinstance(value, list) else []


def _display_ya_usable(meal: dict, locale: Optional[str]) -> bool:
    """¿Este meal YA tiene un `_display[locale]` que sirve?

    [P2-DISPLAY-REDESPACHO-SIN-FILTRO · 2026-08-21] Se exige USABLE, no presente. Es
    la misma lección que P1-I18N-GATE-VALOR dejó en el validador de catálogos: medir
    que la clave existe no es medir que sirve. Un display a medias dado por bueno deja
    esa comida en español para siempre, porque nadie la reintenta.

    Los arrays se comparan por LONGITUD porque el espejo es por índice: un `_display`
    con menos líneas que el meal pinta el gramaje de un ingrediente junto al nombre de
    otro.
    """
    if not locale:
        return False
    disp = meal.get("_display")
    if not isinstance(disp, dict):
        return False
    entrada = disp.get(locale)
    if not isinstance(entrada, dict):
        return False
    if not isinstance(entrada.get("name"), str) or not entrada["name"].strip():
        return False
    for campo in ("recipe", "ingredients"):
        original = meal.get(campo)
        traducido = entrada.get(campo)
        if isinstance(original, list) and original:
            if not isinstance(traducido, list) or len(traducido) != len(original):
                return False
    return True


def _collect_targets(days: list, day_indices_batch: list, locale: Optional[str] = None) -> list:
    """[P2-DISPLAY-REDESPACHO-SIN-FILTRO · 2026-08-21] `locale` es opcional a
    propósito: sin él la conducta es la de antes (reunir todo). Con él, se saltan las
    comidas ya traducidas a ESE idioma.

    Antes esta función no tenía ni una referencia a `_display`, así que cada
    re-disparo del enriquecimiento —cambio de idioma, chunk nuevo, recovery— volvía a
    mandar al LLM lo que ya estaba hecho. Filtrar aquí convierte además la reanudación
    en barata: un enriquecimiento cortado a la mitad retoma sólo lo que falta.
    """
    idx_set = set(day_indices_batch)
    targets = []
    for day_idx, day in enumerate(days):
        if day_idx not in idx_set:
            continue
        if not isinstance(day, dict):
            continue
        meals = day.get("meals")
        if not isinstance(meals, list):
            continue
        for meal_idx, meal in enumerate(meals):
            if not isinstance(meal, dict):
                continue
            if _display_ya_usable(meal, locale):
                continue
            recipe = meal.get("recipe")
            ingredients = meal.get("ingredients")
            targets.append(
                {
                    "day_idx": day_idx,
                    "meal_idx": meal_idx,
                    "name": meal.get("name") or "",
                    # [P1-DESC-KEY-DEAD · 2026-07-24 · fix round 2] La clave real del meal
                    # persistido es `desc`, no `description` — leer solo `description` deja
                    # esto muerto (mismo bug histórico, ver test_p1_desc_key_dead.py).
                    "description": meal.get("desc") or meal.get("description") or "",
                    # [Ola final · FF-2] Snapshot de los arrays espejados: es a la vez el
                    # material del prompt Y la huella con la que `_persist_batch` verifica
                    # que el meal no cambió bajo los pies (misma normalización en ambas
                    # puntas, vía `_fingerprint_lines`).
                    "recipe": _fingerprint_lines(recipe),
                    "ingredients": _fingerprint_lines(ingredients),
                }
            )
    return targets


# ============================================================
# Troceo del trabajo (P1-DISPLAY-LOTE-POR-COMIDAS · 2026-08-21)
# ============================================================

# El lote se medía en DÍAS y el coste lo fijan las COMIDAS. Con el default de 4 días y
# los 4-5 platos diarios de un plan normal, el lote ordinario son 16-20 comidas.
#
# MEDIDO sobre las 271 comidas de los 6 planes vivos con días (2026-08-21),
# reconstruyendo la forma exacta del JSON de respuesta sobre el texto fuente:
#
#     16 comidas  ~6.642 tok de media · ~10.544 en el peor caso   -> se pasa
#     20 comidas  ~8.302 tok de media · ~12.963 en el peor caso   -> se pasa DE MEDIA
#
# Contra un tope de salida de 8.000. Un plan con 5 comidas al día (existe: 76a6836d,
# 55 comidas en 11 días) se pasaba de media, y pasarse no degradaba: la salida se
# truncaba, el JSON no parseaba y el lote entero se descartaba con el gasto ya pagado
# —`_emit_usage_telemetry` corre justo tras el invoke—.
#
# Los dos números de abajo son cotas CONSERVADORAS a propósito. Equivocarse por lo bajo
# cuesta una llamada de más; equivocarse por lo alto cuesta el lote entero.
_CHARS_POR_TOKEN = 3.0        # es/fr/pt con acentos tokenizan peor que el inglés
_INFLACION_TRADUCCION = 1.20  # el francés y el portugués se alargan sobre el español
_FRACCION_UTIL_DEL_TOPE = 0.75


def _tokens_estimados(targets: list) -> float:
    """Tokens de salida que el LLM tendrá que emitir para estos targets.

    Se mide sobre el JSON serializado —no sobre la suma de los campos— porque las
    comillas, las comas y los nombres de clave también se emiten, y en un array de
    ingredientes con líneas cortas esa estructura no es ruido: es un tercio del texto.
    """
    if not targets:
        return 0.0
    chars = sum(len(json.dumps(t, ensure_ascii=False)) for t in targets)
    return chars * _INFLACION_TRADUCCION / _CHARS_POR_TOKEN


def _particionar_targets(targets: list, max_output_tokens: int,
                         tope_comidas: Optional[int] = None) -> list:
    """Trocea la lista de comidas en lotes que quepan en la salida.

    DOS límites, y los dos hacen falta:

      - el proyectado, que es el que responde al contenido real;
      - un tope duro de comidas (`BATCH_DAYS` × las comidas del día más cargado), porque
        una estimación que se equivoca por lo alto no tiene suelo. El knob de días no se
        sustituye: se reinterpreta como cota superior.

    Una comida que por sí sola ya se pasa sale igualmente, en su propio lote. Que el LLM
    la trunque es otro problema; hacerla desaparecer del reparto sería este mismo bug
    otra vez, y más callado.
    """
    if not targets:
        return []
    presupuesto = max(1.0, float(max_output_tokens) * _FRACCION_UTIL_DEL_TOPE)
    if tope_comidas is None:
        tope_comidas = _plan_display_i18n_batch_days() * _COMIDAS_POR_DIA_TOPE
    tope_comidas = max(1, int(tope_comidas))

    lotes: list = []
    actual: list = []
    acumulado = 0.0
    for t in targets:
        coste = _tokens_estimados([t])
        if actual and (acumulado + coste > presupuesto or len(actual) >= tope_comidas):
            lotes.append(actual)
            actual, acumulado = [], 0.0
        actual.append(t)
        acumulado += coste
    if actual:
        lotes.append(actual)
    return lotes


def _dividir_lote(lote: list) -> tuple:
    """Parte un lote en dos para reintentarlo. `([], [])` significa INDIVISIBLE.

    El caso base es load-bearing: sin él, partir un lote de una sola comida devuelve
    `([], [x])` y el reintento vuelve a encolar exactamente el mismo trabajo para
    siempre. Una comida sola que no parsea es pérdida definitiva, y así hay que
    declararla — no reintentarla en bucle.
    """
    if len(lote) < 2:
        return [], []
    mitad = (len(lote) + 1) // 2   # la primera mitad nunca es la pequeña
    return lote[:mitad], lote[mitad:]


def _max_invocaciones_por_ciclo(n_lotes_iniciales: int) -> int:
    """Techo de llamadas al LLM en un ciclo, con split-and-retry activo.

    El split es recursivo: un modelo que devuelve basura para TODO convertiría 4 lotes
    en una bajada comida a comida. Partir en dos añade como mucho una llamada por nivel
    y lote, y con lotes de ~7 comidas tres niveles agotan la división — de ahí el ×3
    más un colchón fijo para los lotes de un solo elemento.
    """
    return max(1, int(n_lotes_iniciales)) * 3 + 2


# ============================================================
# Validación por meal + construcción del `_display` final
# ============================================================


# [P2-DISPLAY-RETENCION-LOCALES · 2026-08-21] Cuántos idiomas conserva `_display`.
#
# El mapa sólo AÑADÍA: nada evacuaba nunca un idioma abandonado, así que un plan de 30
# días visitado en los cinco guardaba cinco copias completas del texto dentro de
# `plan_data` —el mismo jsonb que el comentario de `user_data.py` ya describe como «de
# cientos de KB a MB con 30 días de recetas expandidas»—. Medido a ojo de servilleta:
# ~500 B por comida y locale, ×4 comidas ×30 días ≈ 60 KB por idioma.
#
# Se pone un TOPE en vez de desalojar el anterior, y la diferencia importa: desalojar
# haría re-pagar la traducción entera cada vez que alguien alterna entre dos idiomas,
# que es justo lo que P2-DISPLAY-REDESPACHO-SIN-FILTRO acaba de evitar. Los dos arreglos
# tiran en direcciones opuestas y 2 es donde se cruzan — cubre el ir y venir real
# (idioma nuevo + el de antes) y acota el crecimiento a 2× en vez de 5×.
#
# El activo NUNCA se poda; se descarta el resto por orden de inserción, que en un dict
# de Python es el orden en que se visitaron.
def _max_locales_display() -> int:
    """Cuántos idiomas conserva el `_display` de un plan. Default **2**, a propósito.

    [P2-DISPLAY-RETENCION-LOCALES · 2026-08-21] El 2 no es arbitrario y su razonamiento está
    escrito: desalojar el idioma anterior obligaría a re-pagar la traducción entera cada vez
    que alguien alterna entre dos, y conservarlos todos multiplica por cinco un jsonb que ya
    va «de cientos de KB a MB» (~60 KB por idioma en un plan de 30 días). Dos es donde se
    cruzan las dos presiones: cubre el ir y venir real y acota el crecimiento a 2×.

    [P1-DISPLAY-PODA-TIRA-TRABAJO-PAGADO · 2026-08-23] Llegué a subirlo a 4 tras ver, en una
    ejecución contra un plan REAL, que un plan con tres idiomas salía con dos y perdía el
    `en-US` con sus insights. La observación es correcta —se descarta trabajo pagado— pero la
    conclusión no: eso es el COSTE CONOCIDO de la decisión de arriba, no un descuido. Subir
    el tope revierte un cruce que alguien ya calculó, y no me toca a mí.

    Lo que sí faltaba y queda: que sea un KNOB. Antes era una constante, así que ajustar el
    equilibrio exigía redeploy — y este es justo el tipo de umbral que se querría mover
    mirando datos reales. Ahora se mueve sin desplegar, en cualquiera de las dos direcciones.
    """
    return max(1, _env_int("MEALFIT_PLAN_DISPLAY_I18N_MAX_LOCALES", 2))


def _podar_locales(disp_map: dict, activo: str) -> dict:
    tope = _max_locales_display()
    if not isinstance(disp_map, dict) or len(disp_map) <= tope:
        return disp_map
    # [P2-I18N-DISPLAY-PODA-INERTE-EN-SU-VALOR-MINIMO · 2026-08-23] `[-(tope - 1):]` con
    # `tope = 1` es `lista[-0:]`, y `-0 == 0`: el slice arranca en el PRINCIPIO y devuelve la
    # lista entera. O sea que el ajuste más agresivo del knob —el que un operador elige justo
    # cuando el jsonb se le está yendo— dejaba la poda en no-op y conservaba los cinco
    # idiomas, pareciendo que había hecho algo.
    #
    # `max(0, ...)` en vez de tocar el slice: deja explícito que con tope=1 no se conserva
    # NINGÚN idioma extra, sólo el activo que se añade justo debajo.
    extra = max(0, tope - 1)
    otros = [k for k in disp_map if k != activo]
    conservar = otros[-extra:] if extra else []
    conservar.append(activo)
    return {k: disp_map[k] for k in disp_map if k in conservar}


_NUMEROS = re.compile(r"\d+(?:[.,]\d+)?")


def _cifras_de(linea: str) -> list:
    """El multiconjunto de números de una línea, con el decimal normalizado.

    [P2-DISPLAY-VALIDADOR-SIN-CIFRAS · 2026-08-21] El separador se normaliza porque
    «1.5» → «1,5» es lo que un francés espera leer: tratarlo como cifra perdida
    convertiría el guard en un generador de falsos positivos justo en el idioma que
    más lo necesita. Se ordena porque el orden de las cifras dentro de la frase SÍ
    puede cambiar legítimamente al reordenar la sintaxis.
    """
    return sorted(m.group(0).replace(",", ".") for m in _NUMEROS.finditer(linea or ""))


def _conserva_las_cifras(original_line: str, translated_line: str) -> bool:
    return _cifras_de(original_line) == _cifras_de(translated_line)


# [P2-I18N-DISPLAY-VALIDADOR-CIEGO-A-LA-UNIDAD · 2026-08-23] Las unidades de MAGNITUD, y sólo
# ésas. «180 g» → «180 oz» conserva todas las cifras y es cinco veces más comida: el control
# de cifras lo dejaba pasar como traducción buena, y el usuario leía una cantidad falsa en su
# receta (el motor no: el canónico español sigue dentro de la línea).
#
# NO entran las unidades de cocina traducibles —taza→cup, cda→tbsp, diente→clove— porque ésas
# SE TRADUCEN y exigirlas iguales tiraría todas las traducciones buenas. Entra lo que es una
# magnitud física: si cambia, cambia la cantidad. Cada grupo es una clase de equivalencia
# (g ≡ gr ≡ gramos) para que «100 gr» → «100 g» no cuente como cambio.
_UNIDAD_MAGNITUD = re.compile(
    r"(?<![A-Za-z])(?:"
    r"(?P<mg>mg|miligramos?|milligrams?|milligrammes?|milligrammi)|"
    r"(?P<kg>kg|kilos?|kilogramos?|kilograms?|kilogrammes?|chilogrammi|quilos?)|"
    r"(?P<g>g|gr|gramos?|grams?|grammes?|grammi)|"
    r"(?P<ml>ml|mls?|mililitros?|milliliters?|millilitres?|millilitri)|"
    r"(?P<cl>cl|centilitros?|centiliters?|centilitres?|centilitri)|"
    r"(?P<l>l|litros?|liters?|litres?|litri)|"
    r"(?P<floz>fl\.?\s?oz)|"
    r"(?P<oz>oz|onzas?|ounces?|onces?|once)|"
    r"(?P<lb>lb|lbs|libras?|pounds?|livres?|libbre|libbra)"
    r")(?![A-Za-z])",
    re.IGNORECASE,
)


def _magnitudes_de(linea: str) -> list:
    """Las clases de unidad de magnitud que aparecen en la línea, en orden."""
    fuera = []
    for m in _UNIDAD_MAGNITUD.finditer(linea or ""):
        fuera.append(m.lastgroup)
    return fuera


def _conserva_la_unidad(original_line: str, translated_line: str) -> bool:
    """¿La traducción conserva la MAGNITUD de la línea original?

    Sólo se compara cuando el original lleva una unidad de magnitud; una línea sin unidad
    («Sal al gusto») o con una traducible («2 tazas») no tiene nada que conservar y pasa.
    """
    orig = _magnitudes_de(original_line)
    if not orig:
        return True
    return orig == _magnitudes_de(translated_line)


def _canonico_presente(canonical: str, linea: str) -> bool:
    """¿El nombre canónico aparece en la línea COMO PALABRA?

    [P3-DISPLAY-SUBSTRING-SIN-FRONTERA · 2026-08-21] Antes esto era un `in` pelado sobre
    las dos cadenas normalizadas, y validaba por accidente. Es la clase de defecto que
    este repo ya pagó tres veces: «sal» dentro de Salami, «pollo» dentro de repollo,
    «res» dentro de fResco.

    MEDIDO sobre los 347 nombres del catálogo: 17 son subcadena de otro sin ser palabra
    —«Ajo» en «Ajonjolí», «Piña» en «Espinacas», «Uva» en «Uchuva», y «Sal» en NUEVE— y
    4 caen dentro de palabras inglesas corrientes: «Sal» en *salad*, *salmon* y *salt*;
    «Piña» en *spinach*.

    En concreto: el LLM devolvía «1 tsp salt» —sin el paréntesis con el canónico, que es
    justo lo que este check existe para exigir— y pasaba, porque «sal» está dentro de
    «salt». La línea se persistía sin identificador y la Nevera dejaba de descontar esa
    fila, en silencio.

    EL PRECIO, aceptado: un gloss que PLURALICE el canónico («(Huevos)» por «(Huevo)»)
    deja de validar y esa línea cae al español. Es lo correcto —la directiva pide el
    canónico «literalmente, exactamente como en el original»— y caer al español es
    degradación, no corrupción. Tolerar variaciones sería inventar una regla de
    morfología por idioma que nadie puede validar.
    """
    if not canonical:
        return False
    c = strip_accents(canonical).lower().strip()
    if not c:
        return False
    return re.search(rf"(?<!\w){re.escape(c)}(?!\w)", strip_accents(linea).lower()) is not None


def _validate_and_build_display(original: dict, item: dict) -> Optional[dict]:
    name = item.get("name")
    description = item.get("description")
    recipe = item.get("recipe")
    ingredients = item.get("ingredients")

    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(description, str) or not description.strip():
        return None
    if not isinstance(recipe, list) or len(recipe) != len(original["recipe"]):
        return None
    if not isinstance(ingredients, list) or len(ingredients) != len(original["ingredients"]):
        return None

    # [P1-DISPLAY-ECO-CONTENIDO · 2026-08-23] Un lote que vuelve SIN traducir no se persiste.
    #
    # `P2-DISPLAY-ECO-NOMBRE` cubría el nombre del PLAN. El contenido de cada comida no tenía
    # esta defensa: se comprobaban tipos y longitudes, nada más. Así que un lote donde el
    # modelo devuelve el español tal cual pasaba la validación entera, se persistía como si
    # fuera la traducción, y el gate de «ya traducido» pasaba a decir SÍ — el plan se quedaba
    # en español para siempre y nadie lo reintentaba.
    #
    # MEDIDO el 2026-08-23 ejecutando contra un plan REAL de producción: devolvió
    # `enriched_meals: 4` y las cuatro comidas quedaron en español, ingredientes incluidos
    # (`¼ taza de avena (Avena)` donde debía leerse en francés). Cuatro éxitos declarados,
    # cero traducciones. Sin ejecutar contra datos reales esto no se ve: los tests usan
    # dobles que devuelven texto traducido por construcción.
    #
    # Se juzga por la DESCRIPCIÓN y no por el nombre: un nombre puede coincidir
    # legítimamente entre idiomas (una marca, un sustantivo propio como «Mangú»), pero una
    # frase entera que sobrevive intacta a un cambio de idioma no es una traducción.
    # Se exigen DOS señales, no una. Un solo campo idéntico es evidencia débil: el nombre
    # puede coincidir legítimamente entre idiomas (una marca, «Mangú») y una descripción muy
    # corta también (un «Desc.» de una palabra, o un plato descrito con un solo sustantivo).
    # Que el nombre Y la descripción sobrevivan los dos intactos es otra cosa: es el lote sin
    # traducir, que es justo lo que se midió en producción — allí echaban name, description,
    # ingredientes y receta a la vez.
    #
    # Descartar por una sola señal rompía siete casos legítimos de
    # `P3-DISPLAY-SUBSTRING-FRONTERA`, donde la descripción es un marcador de posición
    # compartido y lo que se está probando es el fallback POR LÍNEA de los ingredientes.
    _desc_original = original.get("description") or ""
    _nombre_original = original.get("name") or ""
    if (_desc_original.strip() and _nombre_original.strip()
            and _eco_del_original(description, _desc_original)
            and _eco_del_original(name, _nombre_original)):
        return None

    final_ingredients = []
    for idx, translated_line in enumerate(ingredients):
        translated_line = translated_line if isinstance(translated_line, str) else ""
        original_line = original["ingredients"][idx]
        original_line = original_line if isinstance(original_line, str) else str(original_line)
        # [P2-DISPLAY-VALIDADOR-SIN-CIFRAS · 2026-08-21] La cantidad va PRIMERO,
        # antes del canonico. Un «180 g» convertido a «1 cup» pasaba entero: el
        # usuario cocina con la cantidad equivocada y el motor sigue calculando los
        # macros sobre el original en espanol, asi que la pantalla y el calculo
        # dejan de contar lo mismo sin que nada avise.
        if not _conserva_las_cifras(original_line, translated_line):
            final_ingredients.append(original_line)
            continue
        # [P2-I18N-DISPLAY-VALIDADOR-CIEGO-A-LA-UNIDAD · 2026-08-23] Y la MAGNITUD, no sólo
        # las cifras: «180 g» → «180 oz» conserva los números y es cinco veces más comida.
        if not _conserva_la_unidad(original_line, translated_line):
            final_ingredients.append(original_line)
            continue
        canonical = _extract_canonical_name(original_line)
        if not canonical:
            # Sin canónico identificable en el original: la línea pasa sin check
            # (spec: "si el original no tiene canónico identificable, la línea
            # pasa sin check").
            final_ingredients.append(translated_line.strip() or original_line)
            continue
        if _canonico_presente(canonical, translated_line):
            final_ingredients.append(translated_line)
        else:
            # Un gloss que pierde el identificador es peor que no tener gloss:
            # se descarta ESA línea (no el meal) -> fallback al original español.
            final_ingredients.append(original_line)

    # [P2-DISPLAY-VALIDADOR-SIN-CIFRAS · 2026-08-21] `recipe` no tenia NINGUN check
    # per-linea: el array entraba tal cual con solo mirar su longitud. Y los tiempos y
    # las temperaturas viven ahi («Hornear 45 minutos a 180 grados»), que es dato que
    # el usuario ejecuta con las manos. Mismo fallback per-linea que ingredients:
    # se descarta la LINEA, no el meal, para no perder la traduccion de todo lo demas.
    final_recipe = []
    for idx, step in enumerate(recipe):
        step = step if isinstance(step, str) else str(step)
        original_step = original["recipe"][idx]
        original_step = original_step if isinstance(original_step, str) else str(original_step)
        # [P1-DISPLAY-VOCAB-CERRADO · 2026-08-21] Dos checks per-linea, el mismo
        # fallback: se descarta LA LINEA (no el meal) y se cae al espanol.
        #   - cifras: un «180 g» convertido a «1 cup» descuadra pantalla y calculo.
        #   - vocabulario cerrado: sin el prefijo, el parser de pantalla no reconoce la
        #     seccion, y sin la etiqueta de nota una ANOTACION pasa a numerarse como
        #     accion de cocina.
        ok = (_conserva_las_cifras(original_step, step)
              and _conserva_la_unidad(original_step, step)       # P2-I18N-DISPLAY-VALIDADOR-CIEGO-A-LA-UNIDAD
              and _conserva_el_vocab_cerrado(original_step, step))
        final_recipe.append(step if ok else original_step)

    return {
        "name": name.strip(),
        "description": description.strip(),
        "recipe": final_recipe,
        "ingredients": final_ingredients,
    }


# [P3-I18N-DISPLAY-METRICA-SIN-LECTOR · 2026-08-22] Razones que NO son un incidente.
#
# `dedupe_locked` es el caso NORMAL bajo concurrencia (otro worker ya está con ese par) e
# `inflight_cap` es el techo de hilos HACIENDO su trabajo. Alertar por cualquiera de las dos
# fabricaría una tasa de error que no existe — el mismo error que `P2-I18N-OBSERVABILIDAD-CERO`
# evitó al contar `SUPERSEDED` aparte de los fallos.
_RAZONES_BENIGNAS = frozenset({"dedupe_locked", "inflight_cap", "disabled", "ok"})


def _emit_degraded_alert(plan_id: str, user_id: str, locale: str, razon: str) -> None:
    """Deja fila en `system_alerts` cuando el enriquecimiento se cae por algo real.

    Por qué una alerta EMITIDA y no un cron que agregue: medido el 2026-08-22,
    `pipeline_metrics` no tiene ni una fila con `node='plan_display_i18n'` (contra 14.835 de
    la semana). Un cron sobre esa tabla seria un panel que reporta cero indefinidamente, y un
    panel que siempre dice cero es un panel que nadie mira el dia que deja de decirlo.

    Emitida cuesta cero mientras la capa no corra, y el dia que corra y falle deja rastro sin
    esperar a la siguiente pasada del cron. Es el modelo «Auto (implicit)» de la politica de
    `system_alerts`: el productor re-emite mientras la condicion exista.

    `alert_key` por LOCALE y no por plan: lo que un operador necesita saber es «el frances
    esta cayendo», no tener 40 filas de 40 planes. Best-effort de verdad — esto jamas puede
    tumbar el enriquecimiento, igual que sus dos hermanas de telemetria.
    """
    if razon in _RAZONES_BENIGNAS:
        return
    try:
        execute_sql_write(
            """
            INSERT INTO system_alerts
                (alert_key, alert_type, severity, title, message, metadata, affected_user_ids)
            VALUES (%s, 'plan_display_i18n_degraded', 'warning', %s, %s, %s::jsonb, %s::jsonb)
            ON CONFLICT (alert_key) DO UPDATE
            SET triggered_at = NOW(),
                metadata = EXCLUDED.metadata,
                resolved_at = NULL
            """,
            (
                f"plan_display_i18n_degraded:{locale}",
                f"La traduccion del plan se cayo en {locale}",
                (
                    f"Motivo: {razon}. El plan {plan_id} se sirve en espanol canonico. "
                    "El usuario ve la app en su idioma y el CONTENIDO del plan en espanol: "
                    "es una degradacion silenciosa, no un error visible."
                ),
                json.dumps({"plan_id": str(plan_id), "locale": locale, "reason": razon},
                           ensure_ascii=False),
                json.dumps([str(user_id)] if user_id else []),
            ),
        )
    except Exception as e:  # noqa: BLE001 — una alerta jamas tumba lo que vigila
        logger.warning(f"[P1-PLAN-DISPLAY-I18N] no se pudo persistir alerta degraded: {e!r}")


def _emit_result_telemetry(plan_id: str, user_id: str, locale: str, resumen: dict) -> None:
    """[P2-DISPLAY-SIN-TELEMETRIA-RESULTADO · 2026-08-21] El RESULTADO, no el coste.

    El módulo ya instrumentaba lo que se GASTA (`_emit_usage_telemetry` →
    `llm_usage_events`) y nada de lo que PASA: cero referencias a `pipeline_metrics`,
    cero a `system_alerts` y cero `logger.error` en todo el fichero. Así que un
    enriquecimiento que se descarta entero —JSON malformado, lote pasado del tope,
    todos los meals con mismatch TOCTOU— era indistinguible de uno que nunca se
    disparó: el usuario ve su plan en español y en el servidor no queda rastro.

    Y con Sentry en `DEFAULT_EVENT_LEVEL=ERROR`, un `logger.warning` tampoco llega:
    la elección de nivel decide si alguien se entera.

    Best-effort de verdad: esto NUNCA puede tumbar el enriquecimiento. Un fallo aquí
    se traga, igual que su hermano de coste.
    """
    _razon = str(resumen.get("reason") or "")
    if _razon:
        _emit_degraded_alert(plan_id, user_id, locale, _razon)
    try:
        execute_sql_write(
            "INSERT INTO pipeline_metrics (user_id, session_id, node, "
            "duration_ms, retries, tokens_estimated, confidence, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)",
            (
                user_id, plan_id, "plan_display_i18n",
                int(resumen.get("duration_ms") or 0),
                int(resumen.get("batches_failed") or 0),
                0,
                None,
                json.dumps({"locale": locale, **resumen}, ensure_ascii=False),
            ),
        )
    except Exception as e:
        logger.debug(f"[P1-PLAN-DISPLAY-I18N] pipeline_metrics falló (best-effort): {e!r}")


def _emit_usage_telemetry(plan_id: str, user_id: str, model_name: str, response) -> None:
    """Best-effort. NUNCA toca `api_usage` — solo `llm_usage_events` (libro de
    costo, node="plan_display_i18n"). Cualquier fallo se traga en silencio."""
    try:
        usage = getattr(response, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
        output_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
        log_llm_usage_event(
            user_id=user_id,
            plan_id=plan_id,
            model=model_name,
            node="plan_display_i18n",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except Exception as e:
        logger.debug(f"[P1-PLAN-DISPLAY-I18N] telemetría falló (best-effort): {e!r}")


# ============================================================
# Persistencia de UN lote validado
# ============================================================


def _persist_batch(
    plan_id: str,
    user_id: str,
    locale: str,
    targets: list,
    valid_by_index: dict,
    plan_name_snapshot: Optional[str] = None,
    plan_name_display: Optional[str] = None,
    insights_snapshot: Optional[list] = None,
    insights_display: Optional[list] = None,
) -> tuple:
    """Persiste un lote ya validado. Retorna `(written_count, mismatch_count)`.

    [fase 1c] `plan_name_snapshot`/`plan_name_display`: cuando no son `None`, este
    MISMO mutator también escribe `pd["_display"][locale]["name"]` (nivel PLAN,
    hermano del `_display` por-meal) — incluido solo en el lote donde se intentó
    la traducción del nombre (ver `enrich_plan_display`, `plan_name_pending`).
    Mismo guard TOCTOU que los meals: si `pd["name"]` ya no coincide con el
    snapshot leído antes de la llamada LLM (un rename concurrente corrió primero),
    el plan_name traducido se descarta (no se escribe) — nunca se pega una
    traducción sobre un nombre que ya cambió.

    El `mutator` corre DENTRO del `SELECT … FOR UPDATE` de `update_plan_data_atomic`
    (contrato P2-MUTATOR-PURITY, `db_plans.py:563`): pura, CPU-only, sin IO, sin
    llamadas LLM/DB, sin re-entrada al pool. Acumular en los dicts `counters`
    (closures Python en memoria) NO viola esa pureza.

    [Finding 5 · fix round 1] TOCTOU: `targets` se construyó en `enrich_plan_display`
    ANTES de este lock (leyendo `plan_data` fuera de cualquier lock — sostener un
    `FOR UPDATE` durante los ~segundos que dura la llamada LLM no es viable). Si el
    meal en esa posición CAMBIÓ entre la lectura y este persist (un
    swap/regenerate-day concurrente escribió ahí primero), el `name` canónico ya no
    coincide con el snapshot leído — escribir de todos modos pegaría la traducción
    del plato VIEJO sobre el plato NUEVO, exactamente el "stale display" que la spec
    declara imposible por construcción. Se compara el `name` (identificador estable
    del meal, en español canónico) y se SALTA si difiere — el meal se queda sin
    `_display` para este locale hasta el próximo disparador legítimo.
    """
    counters = {"written": 0, "mismatch": 0, "plan_name_written": False,
                "plan_name_mismatch": False, "insights_written": False,
                "insights_mismatch": False}

    def _mutator(pd: dict):
        pd_days = pd.get("days") if isinstance(pd, dict) else None
        if not isinstance(pd_days, list):
            return pd
        for i, display in valid_by_index.items():
            t = targets[i]
            try:
                day = pd_days[t["day_idx"]]
                meal = day.get("meals", [])[t["meal_idx"]]
            except (IndexError, TypeError, KeyError, AttributeError):
                continue
            if not isinstance(meal, dict):
                continue
            # [Ola final · FF-2] La verificación TOCTOU comparaba SOLO el nombre del plato,
            # pero `_display[locale]` espeja `ingredients` y `recipe` POR ÍNDICE — y TODO
            # re-cuantizador del motor (macro engine, caps clínicos DM2/bariátrico, carb-floor,
            # qty-sync) cambia gramos CONSERVANDO el nombre por construcción. Con la comparación
            # solo-name, un enriquecimiento en vuelo resucitaba el `_display` construido con el
            # snapshot PRE-mutación encima del pop del mutador: mentira permanente sobre gramos,
            # en el idioma del usuario. La ventana no son los segundos de una llamada LLM: `days`
            # se lee UNA vez y se reutiliza para TODOS los lotes. Igualdad simple de listas de
            # strings, CPU-only ⇒ P2-MUTATOR-PURITY intacta.
            if (meal.get("name") != t["name"]
                    or _fingerprint_lines(meal.get("ingredients")) != t["ingredients"]
                    or _fingerprint_lines(meal.get("recipe")) != t["recipe"]):
                counters["mismatch"] += 1
                continue
            disp_map = meal.get("_display")
            if not isinstance(disp_map, dict):
                disp_map = {}
            disp_map[locale] = display
            meal["_display"] = _podar_locales(disp_map, locale)
            counters["written"] += 1

        # [fase 1c] Escritura del nombre de PLAN — nivel `pd["_display"]`, hermano
        # del `_display` por-meal de arriba (misma key top-level, distinto scope).
        # Mismo guard TOCTOU (comparación por nombre) que los meals: si `pd["name"]`
        # ya no matchea el snapshot leído antes de la llamada LLM (rename manual
        # concurrente), la traducción se descarta silenciosamente — el próximo
        # enriquecimiento (disparado por el propio rename, que ya popeó `_display`)
        # la retraduce contra el nombre nuevo.
        if plan_name_display is not None:
            if pd.get("name") == plan_name_snapshot:
                plan_disp = pd.get("_display")
                if not isinstance(plan_disp, dict):
                    plan_disp = {}
                locale_entry = plan_disp.get(locale)
                if not isinstance(locale_entry, dict):
                    locale_entry = {}
                locale_entry["name"] = plan_name_display
                plan_disp[locale] = locale_entry
                # [P3-I18N-DISPLAY-PODA-SOLO-POR-COMIDA · 2026-08-22] El tope de idiomas se
                # aplicaba SOLO al `_display` por comida; el de nivel plan acumulaba los
                # cinco. Menos volumen, mismo argumento: nada lo evacuaba nunca.
                pd["_display"] = _podar_locales(plan_disp, locale)
                counters["plan_name_written"] = True
            else:
                counters["plan_name_mismatch"] = True

        # [P1-INSIGHTS-I18N · 2026-08-20] El RAZONAMIENTO, hermano del nombre y con el
        # MISMO guard TOCTOU: si `pd["insights"]` ya no coincide con el snapshot leido
        # antes de la llamada LLM (una regeneracion escribio otro razonamiento mientras
        # traduciamos), la traduccion se descarta. Pegar la vieja seria peor que no
        # traducir: el panel diria una cosa y el plan otra.
        if insights_display is not None:
            if pd.get("insights") == insights_snapshot:
                plan_disp = pd.get("_display")
                if not isinstance(plan_disp, dict):
                    plan_disp = {}
                locale_entry = plan_disp.get(locale)
                if not isinstance(locale_entry, dict):
                    locale_entry = {}
                locale_entry["insights"] = insights_display
                plan_disp[locale] = locale_entry
                # [P3-I18N-DISPLAY-PODA-SOLO-POR-COMIDA · 2026-08-22] Idem que el nombre.
                pd["_display"] = _podar_locales(plan_disp, locale)
                counters["insights_written"] = True
            else:
                counters["insights_mismatch"] = True
        return pd

    try:
        persisted = update_plan_data_atomic(plan_id, _mutator, user_id=user_id)
    except Exception as e:
        logger.warning(
            f"[P1-PLAN-DISPLAY-I18N] persist falló plan={plan_id} locale={locale}: {e!r}"
        )
        return 0, 0, False, False
    if not persisted:
        logger.warning(
            f"[P1-PLAN-DISPLAY-I18N] persist retornó vacío (plan desapareció o ownership "
            f"no matcheó) plan={plan_id} locale={locale}"
        )
        return 0, 0, False, False
    if counters["plan_name_mismatch"]:
        logger.warning(
            f"[P1-PLAN-DISPLAY-I18N] nombre del plan omitido por TOCTOU (cambió entre "
            f"lectura y persist) plan={plan_id} locale={locale}"
        )
    return (counters["written"], counters["mismatch"], counters["plan_name_written"],
            counters["insights_written"])


# ============================================================
# Motor
# ============================================================


def enrich_plan_display(
    plan_id: str,
    user_id: str,
    locale: str,
    day_indices: Optional[list] = None,
) -> dict:
    """Enriquece `_display[locale]` de los meals de `plan_id` (ownership
    `AND user_id`). JAMÁS lanza — fail-open total (ver docstring del módulo).

    Trocea en lotes de `MEALFIT_PLAN_DISPLAY_I18N_BATCH_DAYS` días (Finding 6):
    cada lote es UNA llamada LLM + UN persist independiente, así que un lote que
    falla (JSON roto, timeout) no tumba a los demás — recuperación parcial.

    Returns:
        {"enriched_meals": int, "skipped": str | None}
    """
    # [Ola final · FF-3] La clave real se arma tras normalizar los días (abajo); este
    # placeholder solo existe para que el `finally` tenga siempre algo que descartar.
    key = _inflight_key(plan_id, locale, None)
    try:
        if not _plan_display_i18n_enabled():
            return {"enriched_meals": 0, "skipped": "knob_off"}
        if not isinstance(locale, str) or locale not in _COACH_LANGUAGE_NAMES:
            # Cubre es-DO (nunca en el dict) y cualquier locale inválido.
            return {"enriched_meals": 0, "skipped": "locale"}

        plan_data = _fetch_plan_data(plan_id, user_id)
        if plan_data is None:
            return {"enriched_meals": 0, "skipped": "not_found"}

        days = plan_data.get("days")
        if not isinstance(days, list) or not days:
            return {"enriched_meals": 0, "skipped": "no_days"}

        requested_day_indices = _normalize_day_indices(day_indices, len(days))
        if not requested_day_indices:
            return {"enriched_meals": 0, "skipped": "no_meals"}

        # [Ola final · FF-3] MISMA granularidad que el marker cross-worker: dos
        # disparadores sobre DÍAS DISTINTOS del mismo plan+locale ya no se descartan
        # entre sí (`requested_day_indices` ya viene sorted+dedup de la normalización).
        key = _inflight_key(plan_id, locale, requested_day_indices)

        with _INFLIGHT_LOCK:
            if key in _INFLIGHT:
                return {"enriched_meals": 0, "skipped": "dedupe_inprocess"}
            _INFLIGHT.add(key)

        cross_worker_claimed = False
        try:
            if not _try_claim_enrich_lock_cross_worker(plan_id, locale, requested_day_indices):
                # [P3-I18N-DISPLAY-BREAKER-SIN-FILA · 2026-08-22] Fila también aquí. Los
                # otros seis caminos la dejan; estos dos salían mudos, así que en la
                # telemetría un plan bloqueado y un plan que nunca se pidió eran
                # indistinguibles — y el dedupe es el caso NORMAL bajo concurrencia.
                _emit_result_telemetry(plan_id, user_id, locale, {
                    "enriched_meals": 0, "reason": "dedupe_locked",
                })
                return {"enriched_meals": 0, "skipped": "dedupe_locked"}
            cross_worker_claimed = True

            model_name = _plan_display_i18n_model_name()
            if not _circuit_breaker_can_proceed(model_name):
                logger.info(
                    f"[P1-PLAN-DISPLAY-I18N] circuit breaker abierto model={model_name!r} "
                    f"plan={plan_id} locale={locale} — skip silente."
                )
                # [P3-I18N-DISPLAY-BREAKER-SIN-FILA · 2026-08-22] Este es el que más
                # falta hacía: el breaker abierto significa que el proveedor está caído, y
                # era justo el estado que no dejaba rastro en `pipeline_metrics`.
                _emit_result_telemetry(plan_id, user_id, locale, {
                    "enriched_meals": 0, "reason": "circuit_breaker_open",
                })
                return {"enriched_meals": 0, "skipped": "circuit_breaker_open"}

            batch_size = _plan_display_i18n_batch_days()
            max_tokens = _plan_display_i18n_max_output_tokens()
            timeout_s = _plan_display_i18n_timeout_s()
            # [P1-DISPLAY-LOTE-POR-COMIDAS · 2026-08-21] Se recogen TODOS los targets
            # de una vez y se trocean por el tamano proyectado de la salida. Antes el
            # troceo era `requested_day_indices[i:i+batch_days]` — dias, cuando el coste
            # lo fijan las comidas: el lote ordinario (16-20 platos) proyecta 6.600-8.300
            # tokens contra un tope de 8.000, se truncaba y se descartaba entero.
            #
            # `_collect_targets` ya filtra por `locale` lo que esta traducido, asi que
            # recogerlo todo junto no trae trabajo de mas; lo que si trae es la unica
            # forma de repartir por tamano, que necesita ver el conjunto.
            _todos_los_targets = _collect_targets(days, requested_day_indices, locale=locale)
            lotes_iniciales = _particionar_targets(
                _todos_los_targets,
                max_output_tokens=max_tokens,
                tope_comidas=batch_size * _comidas_por_dia_del_plan(
                    days, requested_day_indices),
            )
            # Pila: el split-and-retry devuelve las mitades aqui, y las mitades de una
            # mitad tambien. Se invierte para que el orden de consumo sea el natural.
            _pendientes = list(reversed(lotes_iniciales))
            _presupuesto_invocaciones = _max_invocaciones_por_ciclo(len(lotes_iniciales))
            targets_perdidos = 0

            total_written = 0
            mismatch_total = 0
            last_skip_reason = "no_meals"

            # [fase 1c] Nombre del PLAN — MISMA llamada LLM que los meals, intentado
            # UNA sola vez (el primer lote que efectivamente llega al LLM), no una vez
            # por lote: `plan_name_pending` se limpia tras el primer intento (éxito o
            # fallo de parseo/validación), así un plan de 28 días en 7 lotes no gasta
            # 7 traducciones del mismo título. `_plan_name_already_translated` evita
            # el intento por completo cuando ya hay una traducción vigente para este
            # locale (el rename manual la popea — ver `api_rename_plan` — así que un
            # rename real SÍ vuelve a disparar la traducción en el próximo enriquecimiento).
            # [P1-PLAN-TITLE-I18N · 2026-08-20] DOS valores distintos, y confundirlos
            # era el bug: el TEXTO a traducir y el SNAPSHOT del guard TOCTOU.
            #   - texto:    `plan_data["name"]` si existe (plan renombrado), y si no la
            #               COLUMNA `meal_plans.name`, que es donde vive de verdad.
            #   - snapshot: SIEMPRE el valor del jsonb, aunque sea None. El mutator
            #               compara `pd.get("name")` contra el, y un rename concurrente
            #               CREA ese campo -> None != "Nuevo" -> mismatch detectado. Pasar
            #               ahi el texto de la columna romperia el guard al reves: None
            #               nunca igualaria al titulo y no se escribiria jamas.
            _plan_name_pd = plan_data.get("name")
            _plan_name_snapshot_pd = _plan_name_pd if isinstance(_plan_name_pd, str) else None
            _plan_name_texto = (
                _plan_name_snapshot_pd
                if _plan_name_snapshot_pd and _plan_name_snapshot_pd.strip()
                else _fetch_plan_name_column(plan_id, user_id)
            )
            plan_name_pending = (
                _plan_name_texto
                if isinstance(_plan_name_texto, str)
                and _plan_name_texto.strip()
                and not _plan_name_already_translated(
                    plan_data, locale, original=_plan_name_texto
                )
                else None
            )

            # [P1-INSIGHTS-I18N · 2026-08-20] A diferencia del nombre, `insights` SI vive
            # dentro de `plan_data` -- no hace falta un SELECT extra. El snapshot es el
            # valor tal cual, que es lo que el mutator comparara.
            _insights_pd = plan_data.get("insights")
            _insights_snapshot = _insights_pd if isinstance(_insights_pd, list) else None
            insights_pending = (
                _insights_snapshot
                if _insights_snapshot
                and all(isinstance(x, str) and x.strip() for x in _insights_snapshot)
                and not _insights_already_translated(plan_data, locale)
                else None
            )

            # [P1-I18N-DISPLAY-NIVEL-PLAN-SIN-VIA · 2026-08-22] El nombre del plan y los
            # insights viajaban de POLIZÓN en un lote de comidas, y no tenían vía propia.
            #
            # El troceo se construye SÓLO desde los targets de comida. En el estado normal
            # —todas las comidas ya traducidas para este locale— `_collect_targets` devuelve
            # [], `_particionar_targets` devuelve [] y el `while` de abajo no llega a correr
            # ni una vez. MEDIDO con los dobles del repo (plan con 1 comida ya traducida a
            # en-US + nombre e insights pendientes):
            #
            #     enrich_plan_display(...) -> {'enriched_meals': 0, 'skipped': 'no_meals'}
            #     invocaciones LLM: 0   ·   _display de nivel plan: None
            #
            # Lo que eso significa para el usuario: renombra su plan en el Historial —el
            # rename popea `_display`, y popea el de TODOS los locales— y a partir de ahí el
            # título vuelve al español PARA SIEMPRE, y con él se va el panel «Diagnóstico /
            # Plan de Acción / Tip del Chef». No hay disparador que lo recupere, porque el
            # único que existe cuelga de que aparezca trabajo de comidas.
            #
            # El arreglo es una línea: si no hay lotes pero SÍ hay algo de nivel plan
            # pendiente, encolar un lote VACÍO. Todo lo demás ya lo soportaba —el `continue`
            # de abajo distingue explícitamente «lote vacío sin nada pendiente» de «lote
            # vacío con nombre/insights», `_build_prompt(targets=[])` lo admite y
            # `_persist_batch` también—; lo que faltaba era que alguien lo encolara.
            #
            # El presupuesto no se resiente: `_max_invocaciones_por_ciclo(0)` da 5.
            if not _pendientes and (plan_name_pending is not None or insights_pending is not None):
                _pendientes.append([])

            while _pendientes:
                if _presupuesto_invocaciones <= 0:
                    # El split es recursivo; sin techo, un modelo que devuelve basura
                    # para todo baja hasta una llamada por comida. Lo que queda en la
                    # pila se contabiliza como perdido en vez de desaparecer.
                    perdidas = sum(len(x) for x in _pendientes)
                    targets_perdidos += perdidas
                    logger.error(
                        f"[P1-PLAN-DISPLAY-I18N] techo de invocaciones agotado "
                        f"plan={plan_id} locale={locale}: {perdidas} comida(s) sin "
                        f"traducir en {len(_pendientes)} lote(s) pendiente(s)."
                    )
                    _pendientes = []
                    last_skip_reason = "invocation_budget_exhausted"
                    break

                targets = _pendientes.pop()
                if not targets and plan_name_pending is None and insights_pending is None:
                    continue
                _presupuesto_invocaciones -= 1

                prompt = _build_prompt(
                    targets, locale,
                    plan_name=plan_name_pending,
                    insights=insights_pending,
                )
                try:
                    llm = build_chat_llm(
                        model_name,
                        temperature=0.2,
                        timeout=timeout_s,
                        max_output_tokens=max_tokens,
                    )
                    response = llm.invoke([SystemMessage(content=prompt)])
                except Exception as e:
                    # [P1-I18N-DISPLAY-LOTE-PERDIDO-SIN-SENAL · 2026-08-22] Antes esto era
                    # un `continue` seco: el lote desaparecía sin reintento Y sin contarse.
                    #
                    # El resultado era la peor combinación posible. Un timeout transitorio
                    # de 60 s dejaba media traducción persistida y la otra media en
                    # español, PERMANENTEMENTE —el disparador 4 sólo mira el primer y el
                    # último día, así que no vuelve salvo que el usuario cambie de idioma
                    # otra vez— y `targets_perdidos` reportaba 0, de modo que la telemetría
                    # decía éxito limpio. Un fallo que no se reintenta y encima no se
                    # cuenta es indistinguible de que no hubiera pasado nada.
                    #
                    # Se reencola mientras quede presupuesto (el techo
                    # `_max_invocaciones_por_ciclo` ya existe, así que no hay riesgo de
                    # bucle: si el proveedor está caído de verdad, el techo lo para y la
                    # rama de arriba lo contabiliza). Sin presupuesto, se suma a
                    # `targets_perdidos` igual que hace la rama de JSON no parseable —
                    # que es el mismo fallo con otra causa y ya se contaba bien.
                    logger.warning(
                        f"[P1-PLAN-DISPLAY-I18N] LLM invoke falló plan={plan_id} "
                        f"locale={locale} comidas={len(targets)}: {e!r}"
                    )
                    last_skip_reason = "llm_exception"
                    if _presupuesto_invocaciones > 0:
                        _pendientes.append(targets)
                    else:
                        targets_perdidos += len(targets)
                        logger.error(
                            f"[P1-PLAN-DISPLAY-I18N] lote perdido tras fallo de invoke y "
                            f"sin presupuesto plan={plan_id} locale={locale}: "
                            f"{len(targets)} comida(s) se quedan en español."
                        )
                    continue

                # [Finding 7 · fix round 1] Telemetría INMEDIATAMENTE tras el invoke
                # exitoso — el gasto ya ocurrió aquí, sin importar que el parseo,
                # la validación o el persist de abajo fallen después.
                _emit_usage_telemetry(plan_id, user_id, model_name, response)

                raw_content = getattr(response, "content", "") or ""
                parsed = _parse_json_response(raw_content)
                if parsed is None or not isinstance(parsed.get("meals"), list):
                    # [P1-DISPLAY-LOTE-POR-COMIDAS · 2026-08-21] Antes esto era un
                    # `continue` y el tramo se perdia entero con el gasto ya pagado. La
                    # causa dominante de un JSON que no parsea es la salida TRUNCADA, y
                    # media salida si cabe: se parte y se reintenta cada mitad.
                    izq, der = _dividir_lote(targets)
                    if izq:
                        _pendientes.append(der)
                        _pendientes.append(izq)
                        logger.info(
                            f"[P1-PLAN-DISPLAY-I18N] JSON no parseable con "
                            f"{len(targets)} comida(s) plan={plan_id} locale={locale} "
                            f"— partido en {len(izq)}+{len(der)} y reintentado."
                        )
                    else:
                        # Una sola comida que sigue sin parsear: perdida definitiva.
                        # Es el evento que antes no dejaba rastro ninguno.
                        targets_perdidos += len(targets)
                        logger.error(
                            f"[P1-PLAN-DISPLAY-I18N] comida indivisible sin traducir "
                            f"plan={plan_id} locale={locale} "
                            f"day={targets[0].get('day_idx') if targets else '?'} "
                            f"meal={targets[0].get('meal_idx') if targets else '?'} "
                            f"— JSON no parseable tras el split."
                        )
                    last_skip_reason = "json_parse_error"
                    continue

                valid_by_index = {}
                for item in parsed["meals"]:
                    if not isinstance(item, dict):
                        continue
                    i = item.get("i")
                    if not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= len(targets):
                        continue
                    display = _validate_and_build_display(targets[i], item)
                    if display is not None:
                        valid_by_index[i] = display

                # [fase 1c] El intento de traducir el nombre del plan se consume aquí —
                # UNA vez, éxito o fallo — sin importar cómo termine este lote. Si el
                # LLM no devolvió `plan_name` (plan sin nombre) o vino inválido,
                # `_validate_plan_name` retorna `None` y `_persist_batch` simplemente
                # no escribe nada a nivel plan (los args quedan `None`).
                _batch_plan_name_snapshot = None
                _batch_plan_name_display = None
                if plan_name_pending is not None:
                    # El snapshot es el valor del JSONB (puede ser None a proposito):
                    # es lo que el mutator compara contra `pd.get("name")`.
                    _batch_plan_name_snapshot = _plan_name_snapshot_pd
                    # [P2-DISPLAY-ECO-NOMBRE · 2026-08-21] Con el original delante, un
                    # nombre devuelto SIN traducir se descarta en vez de persistirse:
                    # si se persiste, el gate de «ya traducido» dice SI y nadie lo
                    # reintenta nunca.
                    _batch_plan_name_display = _validate_plan_name(
                        parsed.get("plan_name"), original=plan_name_pending
                    )
                    plan_name_pending = None

                # [P1-INSIGHTS-I18N] Mismo ciclo de vida que el nombre: se intenta UNA vez
                # (el primer lote que llega al LLM) y se limpia el pendiente pase lo que
                # pase, para que un plan de 28 dias en 7 lotes no pague 7 traducciones del
                # mismo razonamiento.
                _batch_insights_snapshot = None
                _batch_insights_display = None
                if insights_pending is not None:
                    _batch_insights_snapshot = _insights_snapshot
                    _batch_insights_display = _validate_insights(
                        parsed.get("insights"), _insights_snapshot)
                    insights_pending = None

                if (not valid_by_index and _batch_plan_name_display is None
                        and _batch_insights_display is None):
                    last_skip_reason = "no_valid_meals"
                    continue

                written, mismatches, _plan_name_written, _insights_written = _persist_batch(
                    plan_id, user_id, locale, targets, valid_by_index,
                    plan_name_snapshot=_batch_plan_name_snapshot,
                    plan_name_display=_batch_plan_name_display,
                    insights_snapshot=_batch_insights_snapshot,
                    insights_display=_batch_insights_display,
                )
                total_written += written
                mismatch_total += mismatches
                if written or _plan_name_written or _insights_written:
                    last_skip_reason = None

            if mismatch_total:
                logger.warning(
                    f"[P1-PLAN-DISPLAY-I18N] {mismatch_total} meal(s) omitidos por TOCTOU "
                    f"(cambiaron entre lectura y persist) plan={plan_id} locale={locale}"
                )

            # [P2-DISPLAY-SIN-TELEMETRIA-RESULTADO · 2026-08-21] Una fila por
            # enriquecimiento, salga bien o mal. Antes sólo se instrumentaba el COSTE:
            # un ciclo que se descartaba entero era indistinguible de uno que nunca se
            # disparó, y el único síntoma era un usuario viendo su plan en español.
            _resumen = {
                "meals_written": total_written,
                "batches": len(lotes_iniciales),
                "targets": len(_todos_los_targets),
                "targets_perdidos": targets_perdidos,
                "mismatch": mismatch_total,
                # [P1-I18N-DISPLAY-LOTE-PERDIDO-SIN-SENAL · 2026-08-22] `reason` ya no
                # colapsa a None en cuanto se escribió ALGO. Con `total_written > 0` y
                # `targets_perdidos > 0` a la vez, el usuario tiene el plan MEDIO
                # traducido y permanentemente así — y la fila decía `reason: None`, o sea
                # éxito limpio. Media traducción no es un éxito: es el único estado que
                # el fallback per-línea existe para evitar, servido de golpe.
                "reason": (
                    "partial_loss" if (total_written > 0 and targets_perdidos > 0)
                    else None if total_written > 0
                    else ("persist_stale_mismatch" if mismatch_total else last_skip_reason)
                ),
            }
            _emit_result_telemetry(plan_id, user_id, locale, _resumen)

            if total_written > 0:
                if targets_perdidos:
                    # [P1-I18N-DISPLAY-LOTE-PERDIDO-SIN-SENAL · 2026-08-22] `error`, no
                    # `info`, y la elección de nivel ES el arreglo: con
                    # `DEFAULT_EVENT_LEVEL=ERROR` un `info` no sube a Sentry, así que un
                    # plan medio traducido para siempre no dejaba ni una señal que alguien
                    # pudiera ver. Mismo criterio que la rama de cero escrituras de abajo.
                    logger.error(
                        f"[P1-PLAN-DISPLAY-I18N] enriquecimiento PARCIAL plan={plan_id} "
                        f"locale={locale}: {total_written} meal(s) traducidos y "
                        f"{targets_perdidos} perdidos — el usuario ve el plan mitad "
                        f"traducido y el disparador no vuelve solo."
                    )
                else:
                    logger.info(
                        f"[P1-PLAN-DISPLAY-I18N] enriquecidos {total_written} meal(s) en "
                        f"{len(lotes_iniciales)} lote(s) plan={plan_id} locale={locale} perdidas={targets_perdidos}"
                    )
                return {
                    "enriched_meals": total_written,
                    "skipped": "partial_loss" if targets_perdidos else None,
                }

            if mismatch_total:
                last_skip_reason = "persist_stale_mismatch"
            # Cero escrituras con lotes despachados es DEGRADACIÓN, no rutina: se
            # levanta a `error` para que Sentry lo recoja. Con `DEFAULT_EVENT_LEVEL=ERROR`
            # un `warning` no sube, y la elección de nivel es la que decide si alguien
            # se entera. Se excluye el caso «no había nada que hacer» (0 lotes).
            if lotes_iniciales:
                logger.error(
                    f"[P1-PLAN-DISPLAY-I18N] enriquecimiento SIN escrituras "
                    f"plan={plan_id} locale={locale} lotes={len(lotes_iniciales)} "
                    f"mismatch={mismatch_total} motivo={last_skip_reason!r}"
                )
            return {"enriched_meals": 0, "skipped": last_skip_reason}
        finally:
            with _INFLIGHT_LOCK:
                _INFLIGHT.discard(key)
            if cross_worker_claimed:
                _release_enrich_lock_cross_worker(plan_id, locale, requested_day_indices)
    except Exception as e:
        logger.warning(
            f"[P1-PLAN-DISPLAY-I18N] enrich_plan_display excepción no controlada "
            f"(fail-open) plan={plan_id} locale={locale}: {e!r}"
        )
        return {"enriched_meals": 0, "skipped": "exception"}


def schedule_plan_display_enrichment(
    plan_id: str,
    user_id: str,
    locale: str,
    day_indices: Optional[list] = None,
) -> None:
    """Wrapper fire-and-forget: dispara `enrich_plan_display` en un thread
    background. Nunca lanza — cualquier fallo (incluso al lanzar el thread)
    queda en warning. Los 5 disparadores de la spec llaman a este helper,
    nunca al motor directo, así el caller no bloquea su propio request.
    """
    try:
        if not _plan_display_i18n_enabled():
            return
        if not isinstance(locale, str) or locale not in _COACH_LANGUAGE_NAMES:
            return

        # Pre-check barato in-process: evita levantar un thread si ya hay uno en vuelo
        # para el mismo (plan_id, locale, MISMOS días). No sustituye el dedupe real
        # (cross-worker + re-check) que vive dentro de `enrich_plan_display`.
        # [Ola final · FF-3] La clave incluye los días — un disparador sobre OTROS días
        # ya no se descarta aquí. Con `day_indices=None` la clave no matchea ninguna del
        # motor (que normaliza contra `len(days)`) ⇒ este pre-filtro no filtra ese caso
        # a propósito: el gate interno + el KV siguen decidiendo.
        _prefilter_key = _inflight_key(
            plan_id, locale,
            None if day_indices is None else _normalize_day_indices(day_indices, 0),
        )
        with _INFLIGHT_LOCK:
            if _prefilter_key in _INFLIGHT:
                logger.debug(
                    f"[P1-PLAN-DISPLAY-I18N] schedule skip — ya en vuelo "
                    f"plan={plan_id} locale={locale} days={day_indices}"
                )
                return

        # [P3-I18N-DISPLAY-HILO-SIN-TECHO · 2026-08-22] Techo GLOBAL de hilos en vuelo.
        #
        # El dedupe de arriba impide dos hilos para el MISMO (plan, idioma). Lo que no
        # acotaba nada es el cruce entre planes: cada uno arranca su propio
        # `threading.Thread` crudo, y cada hilo puede vivir 20-29 minutos hablando con el
        # proveedor. Con una cola de generación activa eso es un abanico sin tope sobre un
        # recurso pago.
        #
        # `acquire(blocking=False)` a propósito: bloquear congelaría el hilo del request
        # que programa el enriquecimiento. Esto es una conveniencia —el plan se sirve en
        # español si no hay hueco— y una conveniencia jamás bloquea al que la pide.
        if not _INFLIGHT_SEMAPHORE.acquire(blocking=False):
            logger.info(
                f"[P1-PLAN-DISPLAY-I18N] techo de hilos alcanzado "
                f"({_plan_display_i18n_max_inflight()}) — plan={plan_id} locale={locale} "
                f"queda en español"
            )
            _emit_result_telemetry(plan_id, user_id, locale, {
                "enriched_meals": 0, "reason": "inflight_cap",
            })
            return

        def _run():
            try:
                result = enrich_plan_display(
                    plan_id, user_id, locale, day_indices=day_indices
                )
                logger.info(
                    f"[P1-PLAN-DISPLAY-I18N] background enrich plan={plan_id} "
                    f"locale={locale} result={result}"
                )
            except Exception as e:
                # enrich_plan_display ya es fail-open y no debería lanzar, pero
                # el thread background no debe morir ruidoso bajo ningún caso.
                logger.warning(
                    f"[P1-PLAN-DISPLAY-I18N] background enrich excepción "
                    f"(fail-open) plan={plan_id} locale={locale}: {e!r}"
                )
            finally:
                # El `finally` es la mitad que importa: sin él, una excepción por la que
                # nadie suelta el permiso convierte el techo en un candado permanente —
                # la feature se apagaría sola y en silencio tras N fallos.
                _INFLIGHT_SEMAPHORE.release()

        # [P2-I18N-DISPLAY-SEMAFORO-SE-FUGA-SI-EL-HILO-NO-ARRANCA · 2026-08-23] El
        # `finally` de `_run` cubre todo lo que pase DENTRO del hilo, pero no este tramo: si
        # `start()` lanza («can't start new thread»), `_run` no corre, nadie suelta el
        # permiso, y con MAX_INFLIGHT fallos así la feature queda apagada para todo el
        # proceso — con «reason: inflight_cap» en cada intento, que apunta a un techo
        # alcanzado y no a uno fugado. Aquí se suelta si el hilo no llegó a existir. NO es
        # un `finally` alrededor del `start()`: eso lo soltaría también cuando SÍ arranca,
        # y el hilo lo soltaría otra vez (BoundedSemaphore lanza en el segundo release).
        try:
            threading.Thread(target=_run, daemon=True).start()
        except Exception:
            _INFLIGHT_SEMAPHORE.release()
            raise
    except Exception as e:
        logger.warning(
            f"[P1-PLAN-DISPLAY-I18N] schedule_plan_display_enrichment falló "
            f"(fail-open) plan={plan_id} locale={locale}: {e!r}"
        )


# ---------------------------------------------------------------------------------------
# [P3-I18N-DISPLAY-KNOBS-PEREZOSOS · 2026-08-22 · ampliado y RELOCALIZADO
#  P3-I18N-DISPLAY-KNOBS-TODOS-EN-EL-REGISTRY · 2026-08-23] Los SIETE knobs, en el IMPORT.
#
# `knobs._env_*` registra en `_KNOBS_REGISTRY` al ser LLAMADO. Estos accesores viven dentro
# de funciones que sólo corren cuando hay algo que traducir, así que
# `get_knobs_registry_snapshot()` —lo que un operador consulta para saber qué puede tocar sin
# redeploy— no los conocía hasta la primera ejecución. Y esta capa, medido el 2026-08-23, se
# ha ejecutado NUEVE veces en toda su historia: en la práctica eran invisibles siempre.
#
# Va al FINAL del módulo y no junto a los primeros accesores, que es donde estaba: allí sólo
# alcanzaban los cinco ya definidos, y los dos que nacen más abajo —`MAX_INFLIGHT` y
# `MAX_LOCALES`— se quedaban fuera. Aquí no depende del orden de definición.
#
# Son siete lecturas de entorno y NO cachean nada: cada accesor sigue leyendo en vivo, que es
# lo que permite el rollback sin redeploy. Esto sólo los DECLARA.
for _declarar in (
    _plan_display_i18n_enabled,
    _plan_display_i18n_model_name,
    _plan_display_i18n_timeout_s,
    _plan_display_i18n_batch_days,
    _plan_display_i18n_max_output_tokens,
    _plan_display_i18n_max_inflight,
    _max_locales_display,
):
    try:
        _declarar()
    except Exception:  # noqa: BLE001 — declarar un knob jamás puede tumbar el import
        pass
del _declarar
