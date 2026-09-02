import re
import math
import os
import random
from collections import defaultdict
import logging
from fractions import Fraction
from db_core import _storage_client, connection_pool, execute_sql_query
from canonical_units import canonicalize_unit, to_base_amount as _to_base_amount  # [P1-shop-coh-1] SSOT de unidades; [P1-NEW-10] conversor base

import time as _time


# [CABEZA-GUARD] Vegetales que se venden por peso/unidad pero NUNCA por "cabeza".
# Lista positiva (no incluye lechuga/coliflor/repollo/brócoli/ajo, que sí son
# nativamente "cabeza"). El guard al final de `apply_smart_market_units` la usa
# para detectar cualquier path interno que asignó erróneamente "Cabezas" a estos
# items y reconstruir el display_qty como peso (lbs) + sub-conteo en unidades.
_NON_CABEZA_NAMES_RE = re.compile(
    r'\b(zanahorias?|tomates?|pimientos?|aj[ií]es?|cebollas?|chiles?|berenjenas?|'
    r'papas?|yucas?|batatas?|tayotas?|remolachas?|calabac[ií]nes?|calabac[ií]n|'
    r'auyamas?|[ñn]ames?|yaut[ií]as?|vegetales)\b',
    re.IGNORECASE,
)


# [P1.4] Backoff exponencial con jitter para 429 / RESOURCE_EXHAUSTED de Gemini.
# Sin esto, una ráfaga puntual de quota tira el cache semántico (embed_documents)
# o pierde matches por ingrediente (embed_query), degradando la lista de compras
# de cualquier plan en curso. langchain_google_genai cambia el wrapping de la
# excepción entre versiones (a veces ResourceExhausted, a veces ClientError con
# code=429), así que detectamos por substring del mensaje + nombre de clase.
def _is_gemini_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    if exc.__class__.__name__ == "ResourceExhausted":
        return True
    return (
        "429" in msg
        or "resource_exhausted" in msg
        or "resourceexhausted" in msg
        or "quota" in msg
    )


def _is_gemini_spending_cap_error(exc: Exception) -> bool:
    """[P0-EMBED-SPENDING-CAP · 2026-05-16] Detecta el 429 específico de
    "spending cap" — la cuenta de AI Studio agotó su cap mensual. A diferencia
    del rate-limit transitorio (que se libera en segundos), el spending cap
    queda activo hasta que el operador suba el cap O hasta que ruede el
    ciclo de billing (hasta 30 días). Reintentar es desperdicio puro.

    El mensaje canónico de Google:
      "Your project has exceeded its monthly spending cap. Please go to AI
       Studio at https://ai.studio/spend to manage your project spend cap."
    """
    msg = str(exc).lower()
    return (
        "spending cap" in msg
        or "monthly spending" in msg
        or "ai.studio/spend" in msg
    )


# [P0-EMBED-SPENDING-CAP · 2026-05-16] Backoff GLOBAL en memoria para evitar
# que cada caller de embeddings reintente 3 veces durante el cap activo.
# Cuando se detecta el primer spending-cap, marcamos hasta `_BACKOFF_S` segundos
# adelante; durante ese window cualquier llamada a `_gemini_call_with_retry`
# salta los 3 intentos + 10s de backoff y raise inmediatamente.
#
# Trade-off: si el operador resuelve el cap durante el window, el sistema
# seguirá saltando hasta que expire. Mitigación: el reset es módulo-level
# (un restart del backend lo limpia) + el window default es corto (1800s).
_GEMINI_SPENDING_CAP_BACKOFF_S = 1800  # 30 min
_gemini_spending_cap_backoff_until: float = 0.0


def _gemini_call_with_retry(fn, *args, _label: str = "gemini_call", **kwargs):
    """Llama `fn(*args, **kwargs)` reintentando 429 con backoff + jitter.

    3 intentos máximo. Delays base 2s y 8s, cada uno con jitter ±25%. Errores
    no relacionados con quota se propagan inmediatamente.

    [P0-EMBED-SPENDING-CAP · 2026-05-16] Dos optimizaciones:
      1. Backoff global activo → raise inmediato (sin intentar). Cuando el
         project tiene `spending cap` activo, todas las llamadas fallarán
         igual; reintentar gasta wall-clock sin ganar nada.
      2. Si la primera respuesta es `spending cap` (no rate-limit), set
         backoff global + raise inmediato (no esperar a intento 2/3).

    Logs en INFO (no WARNING/ERROR): el caller maneja el fallo cayendo a un
    fast-path determinista; no es una condición crítica. Los warnings/errors
    quedaban señalando como roto algo que el sistema ya degrada graciosamente.
    """
    global _gemini_spending_cap_backoff_until

    # (1) Fast-fail si estamos en backoff por spending cap detectado antes.
    if _gemini_spending_cap_backoff_until > _time.time():
        remaining = int(_gemini_spending_cap_backoff_until - _time.time())
        # Logging suave — el caller ya sabe que cae a fast-path.
        logging.info(
            f"[GEMINI/QUOTA] {_label} skipped: spending cap activo "
            f"(~{remaining}s restantes hasta retry). Fast-fail to fast-path."
        )
        raise RuntimeError(
            f"Gemini spending cap active; fast-fail in {_label} (no retries)."
        )

    delays = (2.0, 8.0)
    for attempt in range(3):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not _is_gemini_quota_error(exc):
                raise
            # (2) Spending cap detectado → set backoff global + raise inmediato.
            if _is_gemini_spending_cap_error(exc):
                _gemini_spending_cap_backoff_until = (
                    _time.time() + _GEMINI_SPENDING_CAP_BACKOFF_S
                )
                logging.warning(
                    f"[GEMINI/QUOTA] {_label}: spending cap detectado. "
                    f"Activando fast-fail global por {_GEMINI_SPENDING_CAP_BACKOFF_S}s. "
                    f"Resolver en https://ai.studio/spend o esperar ciclo billing."
                )
                raise
            if attempt == 2:
                logging.info(
                    f"[GEMINI/QUOTA] {_label} agotó 3 intentos por 429 — "
                    f"upstream sin quota; el caller cae al fast-path."
                )
                raise
            base = delays[attempt]
            delay = base * (0.75 + 0.5 * random.random())
            logging.info(
                f"[GEMINI/QUOTA] {_label} 429 (intento {attempt + 1}/3); "
                f"backoff {delay:.1f}s."
            )
            _time.sleep(delay)

_master_cache = None
_master_cache_ts = 0
_MASTER_CACHE_TTL = 300  # 5 minutos de TTL para que aliases nuevos se refresquen
_semantic_cache = None

# Negative cache: cuando la inicialización del caché semántico falla (típicamente
# 429 RESOURCE_EXHAUSTED de Gemini), recordamos el fallo durante este TTL para no
# reintentar inmediatamente. Sin esto, cada llamada a `get_semantic_cache()`
# disparaba otros 3 reintentos × ~10s de backoff y spammeaba 3 logs ERROR.
# El sistema downstream tiene Regex Fast-Path como fallback, así que devolver
# None rápidamente es preferible a bloquear.
#
# Knob `MEALFIT_SEMANTIC_INIT_FAIL_COOLDOWN_S` (default 600s):
# - 300s era optimista para cuotas diarias de Gemini Free Tier; al agotarse el
#   límite diario de embeddings, esperar 5 min y reintentar solo gasta más
#   tokens en 429s sin recuperarse hasta el reset 24h después.
# - 600s reduce a la mitad las re-tentativas malgastadas durante un día con
#   cuota agotada, sin penalizar la recuperación tras un flap minute-window
#   (que en la práctica ya se libera entre reintentos del mismo pipeline).
# - Si la cuenta sube a paid tier, bajar el knob a 180s.
#
# [P2-1 · 2026-05-08] Migrado de `_env_int_local`/`_env_float_local` (que NO
# registraban en `_KNOBS_REGISTRY`) a los helpers compartidos de `knobs.py`.
# Antes los 3 knobs SEMANTIC_INIT/EMBED_INIT eran invisibles en `/health/version`.
# El import a top-level es seguro porque `knobs.py` no depende de este módulo
# (cero ciclo: graph_orchestrator lazy-importa shopping_calculator dentro de
# funciones, nunca a top-level).
from knobs import (
    _env_int as _knob_env_int,
    _env_float as _knob_env_float,
    _env_bool as _knob_env_bool,
    _env_str as _knob_env_str,
)

_SEMANTIC_INIT_FAIL_COOLDOWN_S = max(0, _knob_env_int("MEALFIT_SEMANTIC_INIT_FAIL_COOLDOWN_S", 600))


# [P2-LLM-TIMEOUT-SWEEP · 2026-05-30 · P0-LLM-PROVIDER-MIGRATION · 2026-06-12]
# El deadline del cliente de embeddings (init `embed_documents` + runtime
# `embed_query` del semantic cache) ahora vive en `embeddings_provider`
# (`_embeddings_timeout_s`, mismo knob `MEALFIT_EMBEDDING_LLM_TIMEOUT_S`) —
# este módulo ya no construye su propio cliente; consume
# `get_embeddings_client()`.

# Batching del cache init de embeddings para no saturar RPM del modelo. Modelos
# *-preview (ej. gemini-embedding-2-preview) tienen cuotas Tier 1 conservadoras
# (~30-100 RPM). master_ingredients tiene 50-100+ ítems; mandarlos en una sola
# ráfaga vía `embed_documents([...])` cuenta como N requests en milisegundos y
# pulveriza el RPM. Particionar en batches de 10 + delay 0.5s mantiene RPM
# < 60 con master_list de 100, y elimina el 429 sin cambiar de modelo.
# Trade-off: +5-10s en la primera inicialización; después está cacheado en
# Redis y cero costo. Knobs:
#   MEALFIT_EMBED_INIT_BATCH_SIZE   (default 10): ítems por llamada.
#   MEALFIT_EMBED_INIT_BATCH_DELAY_S (default 0.5): pausa entre batches.
# Si subes a un modelo estable con RPM alto, puedes poner BATCH_SIZE=999 y
# DELAY=0 para volver al comportamiento de ráfaga única (más rápido).
EMBED_INIT_BATCH_SIZE     = max(1, _knob_env_int  ("MEALFIT_EMBED_INIT_BATCH_SIZE",      10))
EMBED_INIT_BATCH_DELAY_S  = max(0.0, _knob_env_float("MEALFIT_EMBED_INIT_BATCH_DELAY_S",   0.5))
# [P2-CAP-LOG-LEVEL · 2026-07-29] Ratio post/pre por debajo del cual un tope de perecedero SÍ es
# señal (habla del menú, no del tope) y se queda en WARNING. Por encima, INFO: era el 74,6% del
# journal. `=1.0` devuelve todos a WARNING (rollback sin redeploy). Ver `_log_cap_applied`.
_CAP_LOG_SEVERE_RATIO = _knob_env_float("MEALFIT_CAP_LOG_SEVERE_RATIO", 0.5,
                                        lambda v: 0.0 <= v <= 1.0)
# [P1-EMBED-INIT-DEADLINE · 2026-07-08] Bound wall-clock GLOBAL de la init del semantic cache.
# La init embebe los ~203 nombres del catálogo contra Cohere en batches seriales; con el proveedor
# lento/rate-limiting el loop se arrastra minutos SIN tope global. El lock non-blocking (0.05s) protege
# al usuario síncrono (cae a fast-path si otro thread inicializa), pero el thread que SÍ hace la init en
# frío (startup warmer, o el primer request antes de que el warmer termine) queda bloqueado hasta que
# los 21 batches + retries terminen. Este deadline acota ese wall-clock: al excederlo, `_batched_embed_
# documents` lanza TimeoutError → `get_semantic_cache` lo captura → cooldown 600s + Regex Fast-Path.
# Default 30s: holgado para la init normal (~15-20s con 203 items) y acota el caso patológico a 30s en
# vez de minutos. Clamp mínimo 5s para no auto-sabotear la init sana. Rollback: subirlo muy alto (=999).
EMBED_INIT_DEADLINE_S     = max(5.0, _knob_env_float("MEALFIT_EMBED_INIT_DEADLINE_S",     30.0))
# [P1-EMBED-WARM-DEADLINE · 2026-07-25] Plazo del calentador de arranque (daemon thread, nadie
# espera su resultado). Debe superar `ceil(catálogo/BATCH_SIZE) × BATCH_DELAY_S` + el tiempo real
# de Cohere; con los 204 alimentos vivos, lotes de 3 y 3 s entre lotes eso son ~204 s de piso.
# 600 s deja holgura para que el catálogo casi triplique sin volver a romperse, y no cuelga el
# thread para siempre. Bajarlo a ≤30 reabre el bug: la init nunca termina, nunca persiste a Redis
# y toda la resolución de ingredientes cae al Regex Fast-Path.
EMBED_WARM_DEADLINE_S     = max(60.0, _knob_env_float("MEALFIT_EMBED_WARM_DEADLINE_S",    600.0))


def _batched_embed_documents(client, all_texts, batch_size, delay_s, retry_label, deadline=None):
    """Particiona `embed_documents` en batches para no saturar RPM del modelo.

    Cada batch va envuelto en `_gemini_call_with_retry`, así un 429 transitorio
    en el batch K solo reintenta ese batch (los anteriores ya están en `out` y
    no se pierden). Si todos los textos caben en un batch, comportamiento
    idéntico al pre-fix (sin overhead).

    [P1-EMBED-INIT-DEADLINE · 2026-07-08] `deadline` (monotonic timestamp, opcional):
    si se provee, antes de cada batch del loop multi-batch se verifica el bound
    wall-clock GLOBAL. Al excederlo se lanza `TimeoutError` → el caller
    (`get_semantic_cache`) lo captura, activa el cooldown y cae al Regex Fast-Path.
    El caso single-batch NO chequea deadline (el per-request timeout del provider
    ya lo acota; el deadline existe para topar la ACUMULACIÓN de N batches seriales).
    `deadline=None` → comportamiento pre-fix idéntico (backward-compat).
    """
    if len(all_texts) <= batch_size:
        return _gemini_call_with_retry(
            client.embed_documents, all_texts, _label=retry_label
        )
    out = []
    n_batches = (len(all_texts) + batch_size - 1) // batch_size
    logging.info(
        f"🧠 [P6-EMBED-BATCH] Cache init particionado en {n_batches} batches "
        f"de hasta {batch_size} ítems con delay {delay_s:.2f}s entre batches."
    )
    for i in range(0, len(all_texts), batch_size):
        # [P1-EMBED-INIT-DEADLINE] Abortar si la init GLOBAL excedió el bound wall-clock.
        # Chequeo al TOPE de cada iteración: el primer batch siempre corre (acotado por el
        # per-request timeout del provider); son los batches ACUMULADOS los que se topan.
        if deadline is not None and _time.monotonic() > deadline:
            raise TimeoutError(
                f"{retry_label}: init excedió el deadline wall-clock "
                f"({EMBED_INIT_DEADLINE_S:.0f}s) tras {len(out)}/{len(all_texts)} textos "
                f"({i // batch_size}/{n_batches} batches); cae a Regex Fast-Path + cooldown."
            )
        chunk = all_texts[i:i + batch_size]
        chunk_idx = (i // batch_size) + 1
        chunk_vectors = _gemini_call_with_retry(
            client.embed_documents, chunk,
            _label=f"{retry_label} batch {chunk_idx}/{n_batches}",
        )
        out.extend(chunk_vectors)
        if i + batch_size < len(all_texts) and delay_s > 0:
            _time.sleep(delay_s)
    return out
_semantic_cache_failed_until = 0.0


# ============================================================
# [P6-SEMANTIC-SKIP] Kill-switch para el caché semántico
# ------------------------------------------------------------
# Cuando el quota de embed_documents está permanentemente exhausto
# (caso real corrida 2026-05-05: cada pipeline desperdicia ~14s en
# 3×retries+backoff de 429s sin éxito porque Redis nunca se logra
# poblar — chicken-and-egg: persist solo corre tras Gemini exitoso).
#
# Activar este knob (`MEALFIT_DISABLE_SEMANTIC_CACHE=true`) hace que
# `get_semantic_cache` retorne None instantáneamente, saltando TODOS
# los intentos a Gemini. El sistema cae al Regex Fast-Path que ya
# cubre el ~95% de casos comunes de matching de ingredientes.
#
# Trade-off:
#   - PRO: ahorra ~14s/pipeline cuando quota está exhausto.
#   - CON: pierdes matching semántico fuzzy (ej. "cebollín verde fresco"
#     no matchea con master "Cebollín" si el regex no lo cubre).
# Para el operador en quota tight, el PRO domina. Default False para
# preservar comportamiento histórico (intentar semantic primero).
#
# Lectura inline (no en module-init): tests pueden cambiar el env via
# monkeypatch sin reload. Costo: 1 lookup string por llamada — trivial.
# ============================================================
def _semantic_cache_disabled() -> bool:
    """True si el operador desactivó el semantic cache via env var.
    Acepta '1', 'true', 'yes', 'on' (case-insensitive)."""
    # [P2-1 · 2026-05-08] `_env_bool` registra en `_KNOBS_REGISTRY`.
    return _knob_env_bool("MEALFIT_DISABLE_SEMANTIC_CACHE", False)

# ============================================================
# [P5-EMBED-CACHE-E] Persistencia de vectores en Redis
# ------------------------------------------------------------
# El caché semántico es in-process: cada worker (Gunicorn fork, container
# restart, deploy) re-fetcha embeddings desde Gemini. Como master_ingredients
# tiene ~50 items y cada embedding cuesta una llamada API, la inicialización
# pega contra el quota minute-window y dispara 429 (visible en cada corrida
# como "embed_documents agotó 3 intentos por 429"). El sistema cae al Regex
# Fast-Path graciosamente, pero se desperdicia ~14s por pipeline en backoffs
# y se aumenta presión sobre el quota compartido.
#
# Solución: cachear los vectores en Redis con key = hash estable de la
# master_list. Si master_ingredients no cambia (caso típico — items se
# añaden manualmente, ritmo semanal a lo más), Redis sirve los vectores
# instantáneamente y el primer worker que arranca no necesita Gemini.
# Cuando la lista cambia, el hash cambia → cache miss → re-fetch (una vez)
# → re-persist. TTL 7 días para que cualquier ingrediente nuevo se refleje
# en una semana incluso sin invalidación explícita.
#
# Tamaño: ~50 vectores × 768 floats × ~10 chars JSON c/u ≈ 384 KB por entry.
# Trivial para Redis. Versionamos la key con `v1` para invalidaciones futuras
# (cambio de modelo de embedding, nueva normalización de texto, etc.).
# ============================================================
_REDIS_EMBED_CACHE_KEY_PREFIX = "embed:master_ingredients:v1"
_REDIS_EMBED_CACHE_TTL_S = 7 * 24 * 3600  # 7 días


def _master_list_hash(master_list: list) -> str:
    """Hash estable de la lista para invalidación cuando cambia el contenido.

    Considera `name` + `aliases` + `category` — los campos que afectan el
    texto que se embebe (ver `texts = [f"{m['name']} - Categoría: ..."]`
    en `get_semantic_cache`). Si cualquiera de estos cambia, el embedding
    debe regenerarse para mantener la semántica del vector.

    Sortea los items por nombre para que el hash sea independiente del
    orden de devolución de Postgres (el SELECT no garantiza orden estable
    sin ORDER BY)."""
    import hashlib
    parts = []
    for m in sorted(master_list, key=lambda x: x.get("name", "")):
        name = m.get("name", "")
        category = m.get("category", "") or ""
        aliases = "|".join(sorted(m.get("aliases") or []))
        parts.append(f"{name}::{category}::{aliases}")
    blob = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _model_hash(model_name: str) -> str:
    """Hash corto del nombre de modelo para inyectar en la Redis key.

    [2026-05-06] Asegura que vectores cacheados con un modelo no se
    confundan con vectores de otro modelo (espacios vectoriales distintos).
    [P0-LLM-PROVIDER-MIGRATION · 2026-06-12] El ID viene de
    `embeddings_provider.get_embeddings_model_id()` (knob
    `MEALFIT_EMBEDDINGS_MODEL`); si cambias de provider/modelo, las entradas
    Redis viejas quedan ignoradas y se regeneran automáticamente.
    """
    import hashlib
    return hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:8]


def _redis_embed_cache_key(master_list: list) -> str:
    from embeddings_provider import get_embeddings_model_id
    return (
        f"{_REDIS_EMBED_CACHE_KEY_PREFIX}:"
        f"{_model_hash(get_embeddings_model_id())}:"
        f"{_master_list_hash(master_list)}"
    )


def _try_load_embed_vectors_from_redis(master_list: list):
    """Intenta cargar los vectores cacheados de Redis. Retorna None si:
       - Redis no está disponible
       - No hay entry para este hash
       - El JSON está corrupto o el shape no matchea
    Defensivo: nunca lanza, los errores degradan a None y el caller
    procede al fast-fetch desde Gemini."""
    try:
        from cache_manager import redis_client
        if not redis_client:
            return None
        key = _redis_embed_cache_key(master_list)
        raw = redis_client.get(key)
        if not raw:
            return None
        import json as _json
        vectors = _json.loads(raw)
        # Validar shape: lista de listas de floats, mismo length que master_list.
        if not isinstance(vectors, list):
            return None
        if len(vectors) != len(master_list):
            logging.info(
                f"🟡 [P5-EMBED-CACHE-E] Redis vectors length mismatch "
                f"({len(vectors)} vs {len(master_list)}); ignorando entry."
            )
            return None
        return vectors
    except Exception as exc:
        logging.info(
            f"🟡 [P5-EMBED-CACHE-E] Redis read fallo "
            f"({type(exc).__name__}); cae a Gemini fetch."
        )
        return None


def _persist_embed_vectors_to_redis(master_list: list, vectors: list) -> bool:
    """Persiste los vectores en Redis. Retorna True si OK, False si falló
    (Redis down, vectors no serializable, etc.). Nunca lanza."""
    try:
        from cache_manager import redis_client
        if not redis_client:
            return False
        key = _redis_embed_cache_key(master_list)
        import json as _json
        redis_client.setex(key, _REDIS_EMBED_CACHE_TTL_S, _json.dumps(vectors))
        return True
    except Exception as exc:
        logging.info(
            f"🟡 [P5-EMBED-CACHE-E] Redis write fallo "
            f"({type(exc).__name__}); siguiendo sin persistir."
        )
        return False

# Lock para serializar inicializaciones concurrentes. Sin esto, cuando el shopping
# list se calcula 3 veces en paralelo (mult ×2/×4/×8), las 3 disparan el fetch
# de embeddings simultáneamente — triplicando consumo de quota y latencia.
import threading as _threading
_semantic_cache_lock = _threading.Lock()


def invalidate_master_cache():
    """Invalida el caché de master_ingredients para forzar recarga desde DB."""
    global _master_cache, _master_cache_ts, _semantic_cache, _semantic_cache_failed_until
    _master_cache = None
    _master_cache_ts = 0
    _semantic_cache = None
    _semantic_cache_failed_until = 0.0

def get_semantic_cache(deadline_s: float | None = None):
    """Devuelve el caché semántico (master_list + vectors + embeddings_client).

    `deadline_s` es el tope wall-clock de la inicialización desde cero. Por defecto usa
    `EMBED_INIT_DEADLINE_S` (30 s), que es lo correcto para un camino que atiende a un usuario:
    más vale caer al Regex Fast-Path que bloquear la petición.

    [P1-EMBED-WARM-DEADLINE · 2026-07-25] El **calentador de arranque** debe pasar un plazo
    proporcional al trabajo real, porque con el default es matemáticamente imposible terminar
    (valores vivos en prod: 204 alimentos, lotes de 3, 3 s entre lotes):

        204 alimentos ÷ EMBED_INIT_BATCH_SIZE (3)   = 68 lotes
        68 lotes × EMBED_INIT_BATCH_DELAY_S (3,0 s) = 204 s de espera deliberada
        …contra un deadline de 30 s

    Consecuencia medida en prod: la init aborta SIEMPRE → nunca persiste a Redis → Redis nunca
    tiene los vectores (verificado: 100 claves, ninguna del catálogo) → cada proceso reintenta
    cada 10 min y quema 30 s, y **toda** la resolución de ingredientes cae al Regex Fast-Path.
    En el chunk que expiró el 17-jul esos 30 s salieron del presupuesto de 600 s.

    El deadline de 30 s se añadió en P1-EMBED-INIT-DEADLINE (2026-07-08) para que Cohere lento no
    bloqueara minutos — correcto para peticiones, pero se aplicó también al warmer, que es
    justamente el único caller que SÍ puede esperar (daemon thread, nadie aguarda su resultado).

    Orden de resolución (importante por interacción cooldown ↔ Redis):
      1. In-process cache hit → fast return.
      2. Lock + re-check.
      3. **Redis read FIRST** (no cuesta quota Gemini, vector data es válido
         incluso bajo cooldown). Si hit, retornamos sin tocar Gemini.
      4. Redis miss → AHORA chequear cooldown. Si activo, return None.
      5. Cooldown OK → llamar Gemini, persistir a Redis, retornar.

    [P6-EMBED-CACHE-FIX] Pre-fix: el cooldown check estaba ANTES del Redis
    read, así que cualquier 429 reciente bloqueaba Redis lookup por 300s.
    Caso real corrida 2026-05-05 14:01: 429 a las 14:01:02, cooldown hasta
    14:06:02 — pero Redis tenía vectores válidos persistidos en la corrida
    13:33. La cache nunca se servía aunque existiera.

    [P6-SEMANTIC-SKIP] Kill-switch: si `MEALFIT_DISABLE_SEMANTIC_CACHE` está
    on, retornamos None inmediatamente. Para entornos con quota Gemini
    permanentemente exhausto donde el Regex Fast-Path basta. Ahorra ~14s
    por pipeline en esa configuración.
    """
    # [P6-SEMANTIC-SKIP] Kill-switch antes de TODO: ni siquiera intentar
    # cache lookup ni Gemini call. Operador desactivó vía env.
    if _semantic_cache_disabled():
        return None

    global _semantic_cache, _semantic_cache_failed_until
    if _semantic_cache is not None:
        return _semantic_cache

    # [P3-EMBED-CACHE-STARTUP-WARM · 2026-05-16] Non-blocking lock acquire para
    # synchronous user-facing paths. Pre-fix: si el startup warmer (background
    # thread, ~100s en cold init) tenía el lock, una request del usuario
    # esperaba bloqueando hasta que el warmer terminara. Resultado: misma
    # latencia ~100s que sin warmer → recalc-shopping-list timeout 500/CORS.
    # Post-fix: si el lock está ocupado, asumimos que OTRO thread está
    # inicializando y caemos al regex fast-path (P6-SEMANTIC-SKIP) en lugar
    # de bloquear. La próxima call (post-init) leerá el cache instantáneo.
    # Timeout 0.05s (50ms) cubre la ventana de race entre 2 threads ambos
    # intentando inicializar legítimamente; en práctica el lock se libera
    # casi inmediato si nadie lo tiene.
    acquired = _semantic_cache_lock.acquire(timeout=0.05)
    if not acquired:
        logging.info(
            "🟡 [P3-EMBED-CACHE-STARTUP-WARM] Lock ocupado (otro thread inicializando "
            "semantic cache). Fast-path Regex será usado para esta query."
        )
        return None
    try:
        # Re-check tras adquirir el lock (otro thread pudo haber inicializado).
        if _semantic_cache is not None:
            return _semantic_cache

        master_list = get_master_ingredients()
        if not master_list:
            return None

        # Cliente embeddings: barato instanciar (sin quota cost), necesario
        # tanto para Redis-hit (downstream `embed_query` runtime) como para
        # el fetch inicial (init de `embed_documents`).
        # [P0-LLM-PROVIDER-MIGRATION · 2026-06-12] Via capa pluggable. Con
        # provider `disabled` retorna None → fast-path Regex (path graceful
        # pre-existente, mismo comportamiento que un fallo de instanciación).
        from embeddings_provider import get_embeddings_client
        embeddings = get_embeddings_client()
        if embeddings is None:
            logging.info(
                "🟡 [P6-EMBED-CACHE-FIX] Embeddings provider disabled/no "
                "instanciable; fast-path Regex será usado."
            )
            return None

        # [P6-EMBED-CACHE-FIX] PASO 1 — Try Redis FIRST. Vector data
        # cacheada es válida incluso si Gemini está en cooldown por 429
        # — son sistemas independientes.
        vectors = _try_load_embed_vectors_from_redis(master_list)
        if vectors is not None:
            _semantic_cache = {
                "master_list": master_list,
                "vectors": vectors,
                "embeddings_client": embeddings,
            }
            logging.info(
                f"🧠 [P5-EMBED-CACHE-E] Caché semántico cargado desde Redis "
                f"({len(vectors)} vectores, hash={_master_list_hash(master_list)[:8]}) "
                f"— Gemini embed_documents evitado."
            )
            return _semantic_cache

        # [P6-EMBED-CACHE-FIX] PASO 2 — Redis miss. AHORA sí chequear el
        # cooldown de Gemini (movido aquí para que Redis tenga su chance).
        if _time.time() < _semantic_cache_failed_until:
            return None

        try:
            texts = [f"{m['name']} - Categoría: {m.get('category','')}. Alias: {', '.join(m.get('aliases') or [])}" for m in master_list]
            vectors = _batched_embed_documents(
                embeddings, texts,
                EMBED_INIT_BATCH_SIZE, EMBED_INIT_BATCH_DELAY_S,
                retry_label="embed_documents (master_ingredients cache init)",
                # [P1-EMBED-INIT-DEADLINE · 2026-07-08] tope wall-clock GLOBAL de la init: si Cohere
                # se arrastra, aborta a los EMBED_INIT_DEADLINE_S y cae al fast-path (el except de abajo
                # activa el cooldown). Sin esto, el thread de init podía bloquear minutos.
                # [P1-EMBED-WARM-DEADLINE · 2026-07-25] …salvo que el caller declare otro plazo: el
                # warmer de arranque necesita ~96 s sólo de delay entre lotes (ver docstring).
                deadline=_time.monotonic() + float(
                    deadline_s if deadline_s and deadline_s > 0 else EMBED_INIT_DEADLINE_S),
            )

            _semantic_cache = {
                "master_list": master_list,
                "vectors": vectors,
                "embeddings_client": embeddings
            }
            # Persistir para los próximos workers/restarts.
            _persist_embed_vectors_to_redis(master_list, vectors)
            logging.info("🧠 Caché semántico local inicializado con éxito por primera vez.")
            return _semantic_cache
        except Exception as e:
            _semantic_cache_failed_until = _time.time() + _SEMANTIC_INIT_FAIL_COOLDOWN_S
            # INFO en vez de ERROR: el sistema cae al Regex Fast-Path y sigue trabajando.
            # Solo es notable la PRIMERA vez del cooldown; las llamadas siguientes
            # devuelven None instantáneamente sin loggear nada.
            logging.info(
                f"🟡 Caché semántico no disponible ({type(e).__name__}); "
                f"usando Regex Fast-Path. Reintentos pausados {_SEMANTIC_INIT_FAIL_COOLDOWN_S}s."
            )
            return None
    finally:
        # [P3-EMBED-CACHE-STARTUP-WARM · 2026-05-16] Release explícito: el
        # `with _semantic_cache_lock:` original liberaba automáticamente al
        # salir del bloque, pero el non-blocking acquire (timeout=0.05s) de
        # arriba requiere release explícito para evitar deadlock permanente.
        _semantic_cache_lock.release()


def cosine_similarity(v1, v2):
    dot = sum(a*b for a,b in zip(v1, v2))
    mag1 = math.sqrt(sum(a*a for a in v1))
    mag2 = math.sqrt(sum(a*a for a in v2))
    if mag1 == 0 or mag2 == 0: return 0
    return dot / (mag1 * mag2)

def get_master_ingredients():
    global _master_cache, _master_cache_ts
    now = _time.time()
    if _master_cache is None or (now - _master_cache_ts) > _MASTER_CACHE_TTL:
        if connection_pool:
            try:
                # [P1-CATALOG-ORDER-DETERMINISTIC · 2026-08-19] ORDER BY name: sin él, el
                # ganador de CADA colisión del índice de resolución (alias/contains/keys
                # normalizadas duplicadas) era el orden FÍSICO del heap — comportamiento
                # indefinido que un UPDATE masivo re-baraja. Medido: el fill del gloss del
                # catálogo (347 UPDATEs, 2026-08-19) flipeó 4 resoluciones REALES del corpus DO («Pollo horneado
                # al limón...» pasó de Pechuga de pollo a Arroz blanco). El orden alfabético
                # DESC no es capricho: el índice es first-wins y DESC restaura los 7
                # ganadores del baseline C3 committeado (verificado delta a delta:
                # Filete>Batata, Pechuga>Arroz, Repollo morado>Repollo, Pulpo>Calamar,
                # Tofu firme>Salsa de soya, Yuca>Atún) — el contrato revisado en F2 se
                # conserva Y queda estable para siempre.
                res = execute_sql_query("SELECT * FROM master_ingredients ORDER BY name DESC", fetch_all=True)
                # [P1-CATALOG-INDEX-NO-STICKY · 2026-07-29] `res or []` aceptaba como catálogo
                # CUALQUIER objeto truthy y le sellaba `_master_cache_ts` → 5 minutos sirviendo
                # basura como si fuera la tabla verificada. Cómo se destapó: un test parchea
                # `db_core.connection_pool` con un MagicMock, `execute_sql_query` lo lee en tiempo
                # de llamada, y el MagicMock (truthy) quedaba cacheado como el catálogo; aguas
                # abajo `_phantom_catalog_index` lo iteraba, lanzaba, y cacheaba su fallo. El
                # mock es el mensajero, pero el agujero es de producción: un cursor a medio
                # consumir o cualquier retorno inesperado del driver entra igual.
                if isinstance(res, list):
                    _master_cache = res
                    _master_cache_ts = now
                else:
                    logging.error(
                        f"❌ [P1-CATALOG-INDEX-NO-STICKY] master_ingredients devolvió "
                        f"{type(res).__name__}, no una lista — NO se cachea (se reintenta en la "
                        f"próxima llamada en vez de servir 5 min de basura).")
                    if _master_cache is None:
                        _master_cache = []
            except Exception as e:
                logging.error(f"Error fetching master_ingredients via pool: {e}")
                if _master_cache is None:
                    _master_cache = []
        else:
            logging.error("No connection_pool available to fetch master_ingredients")
            if _master_cache is None:
                _master_cache = []
    return _master_cache


# ============================================================
# [P3-VERIFIED-INGREDIENTS-ONLY · 2026-06-20] Enforcement: SOLO alimentos
# verificados con precio La Sirena (los ~202 verificados de master_ingredients (era 119 pre-expansion 2026-06-26)) pueden
# aparecer en la lista de compras. Decisión del owner: "no quiero que el LLM
# invente alimentos; solo los verificados del catalogo (~202) deben estar en la lista".
# Dos puntos consumen la MISMA `_is_verified_for_shopping` → simetría garantizada:
#   (1) drop en `aggregate_and_deduct_shopping_list` (excluye de la lista), y
#   (2) espejo en `run_shopping_coherence_guard` (filtra expected_raw) — sin el
#       espejo, dropear un ingrediente inventado generaría divergencia
#       `expected_only` → en modo=block fuerza retry costoso del plan.
# Gateado por knob; flip a False revierte sin redeploy.
# Tooltip-anchor: P3-VERIFIED-INGREDIENTS-ONLY.
# ------------------------------------------------------------
def _verified_ingredients_only_enabled() -> bool:
    # [P1-VERIFIED-ONLY-DEFAULT-ON · 2026-07-02] Default ON en CÓDIGO. Antes era OFF-en-código /
    # ON-solo-en-.env-del-VPS → un deploy limpio que resetee el .env lo apagaba EN SILENCIO y el
    # LLM volvía a inventar alimentos off-catálogo (drops de lista, costeo incompleto, palancas de
    # presupuesto sin base). El knob es load-bearing para la garantía "coherente + costeable" desde
    # 2026-06-20 (prod) → el default de código ahora refleja el contrato real. El baseline histórico
    # de los tests se preserva vía tests/conftest.py (setdefault a false — los tests del knob lo
    # activan con monkeypatch). Rollback sin redeploy: MEALFIT_VERIFIED_INGREDIENTS_ONLY=false.
    # Solo los alimentos verificados con precio del catálogo (~202 tras la expansión 2026-06-26)
    # pueden aparecer en la lista. [P3-STALE-119-COMMENTS · 2026-07-01]
    return _knob_env_bool("MEALFIT_VERIFIED_INGREDIENTS_ONLY", True)


# [P2-VERIFIED-DROP-TELEMETRY · 2026-07-01] (audit v2 creatividad GAP-5, batch P2-AUDIT-V2-BATCH)
# Contador in-process (bounded) de los ingredientes dropeados por VERIFIED-ONLY. El WARN grep-able
# (P1-VERIFIED-ONLY-OBSERVABILITY) sigue siendo la fuente forense; este sink da el AGREGADO que faltaba:
# el cron `_creativity_kpi_job` (cron_tasks) toma snapshot+reset y emite el top-N a pipeline_metrics →
# "qué synonyms/altas de catálogo faltan" deja de requerir grep de logs. In-memory a propósito (drops
# son raros por diseño; un restart pierde el parcial del día — aceptable para telemetría direccional).
_VERIFIED_ONLY_DROP_COUNTS: dict = {}
_VERIFIED_ONLY_DROP_MAX_KEYS = 200


def record_verified_only_drop(name) -> None:
    """Suma 1 al contador del ingrediente dropeado (key lower/trim, cap 200 keys anti-runaway)."""
    try:
        key = str(name or "").strip().lower()[:80]
        if not key:
            return
        if key not in _VERIFIED_ONLY_DROP_COUNTS and len(_VERIFIED_ONLY_DROP_COUNTS) >= _VERIFIED_ONLY_DROP_MAX_KEYS:
            return
        _VERIFIED_ONLY_DROP_COUNTS[key] = _VERIFIED_ONLY_DROP_COUNTS.get(key, 0) + 1
    except Exception:
        pass


def snapshot_and_reset_verified_only_drops() -> dict:
    """Devuelve el contador acumulado y lo resetea (consumido por el cron de KPI de creatividad)."""
    global _VERIFIED_ONLY_DROP_COUNTS
    snap = _VERIFIED_ONLY_DROP_COUNTS
    _VERIFIED_ONLY_DROP_COUNTS = {}
    return snap


_VERIFIED_SHOPPING_NAMES = None
_VERIFIED_SHOPPING_NAMES_TS = 0.0


def _get_verified_shopping_name_set() -> set:
    """Set de nombres canónicos (accent-stripped, lower) de los master_ingredients
    CON precio La Sirena verificado (price_per_lb>0 OR price_per_unit>0). Cacheado
    con el mismo TTL que get_master_ingredients (refresca precios/aliases nuevos)."""
    global _VERIFIED_SHOPPING_NAMES, _VERIFIED_SHOPPING_NAMES_TS
    now = _time.time()
    if _VERIFIED_SHOPPING_NAMES is None or (now - _VERIFIED_SHOPPING_NAMES_TS) > _MASTER_CACHE_TTL:
        from constants import strip_accents as _sa
        rows = get_master_ingredients() or []
        _VERIFIED_SHOPPING_NAMES = {
            _sa(str(r.get("name") or "").lower().strip())
            for r in rows
            if (r.get("price_per_lb") or 0) > 0 or (r.get("price_per_unit") or 0) > 0
        }
        _VERIFIED_SHOPPING_NAMES_TS = now
    return _VERIFIED_SHOPPING_NAMES


def _is_verified_for_shopping(name) -> bool:
    """True si `name` resuelve (vía normalize_name, el resolver SSOT recetas→master)
    a un master_ingredients con precio verificado. Usado IDÉNTICAMENTE por el drop
    del aggregator y el espejo del coherence guard — la simetría drop↔espejo está
    garantizada por construcción (misma función), así un alimento inventado (laurel,
    comino, cúrcuma) se excluye de la lista Y del set esperado, sin forzar retry."""
    try:
        canon = normalize_name(name)
    except Exception:
        return False
    from constants import strip_accents as _sa
    return _sa(str(canon).lower().strip()) in _get_verified_shopping_name_set()


def _seasoning_catalog_keep_enabled() -> bool:
    # [P2-SEASONING-CATALOG-KEEP · 2026-06-22] Default ON. Un ingrediente del CATÁLOGO verificado
    # (cilantro, orégano, etc.) que el LLM emitió SOLO en cantidad nominal (pizca/al gusto, sin peso)
    # se LISTA con 1 empaque mínimo en vez de dropearse — la receta lo usa y es un alimento comprable.
    # Cierra la lista-incompleta para sazones de catálogo (caso visto en vivo 2026-06-22: cilantro +
    # orégano dominicano caídos por "pizca"). NO afecta no-catálogo (esos siguen el drop +
    # observabilidad VERIFIED-ONLY). Flip a False revierte al drop. Tooltip-anchor: P2-SEASONING-CATALOG-KEEP.
    return _knob_env_bool("MEALFIT_SEASONING_CATALOG_KEEP", True)


# Peso por defecto (~1 empaque pequeño de sazón) cuando el master no trae container_weight_g/density.
_SEASONING_DEFAULT_G = 40.0


# [P1-SHOPLIST-SANITY-CAP · 2026-08-21] Umbral de envase por debajo del cual una fila de Despensa
# es una PRESENTACIÓN DE CONDIMENTO. No existe categoría «especias» en `master_ingredients` —
# orégano, arroz y maíz en lata comparten 'Despensa'—, pero el envase las separa limpiamente:
# especias 14-100 g, comida de despensa 425 g (lata de maíz) y 907 g (paquete de arroz). 120 g
# deja margen sobre el frasco más grande medido (Laurel, 100 g) sin acercarse a la lata.
_CONDIMENT_MAX_CONTAINER_G = 120.0


# ============================================================
# [P1-UNIT-SYSTEM-BY-COUNTRY · 2026-08-21] La lista del español venía en libras.
#
# Medido sobre los 2 planes beta vivos: 14/25 y 26/48 ítems en unidades imperiales. En ES/MX/CO la
# carne se vende por kilos y la báscula da gramos; ¼ lb = 113 g no es un número que nadie pida. La
# misma decisión ACIERTA para DO/US/PR, donde la libra es la unidad real de compra.
#
# DOS MEDICIONES ACOTARON EL ARREGLO, y las dos lo hicieron más pequeño:
#   1. Las recetas YA son métricas (0 de 96 líneas del plan español usan libras): el problema no
#      está en la generación, vive entero en este agregador determinista.
#   2. La mitad de los «lb» visibles NO son una instrucción de pesar sino el RÓTULO de un envase
#      real ("1 funda (Selecto 1 Lb · Wala)"). Convertir eso sería falsificar una etiqueta y el
#      usuario no encontraría el producto. Sólo se convierte cuando la unidad de mercado ES el peso.
#
# DISPLAY-ONLY, SIN EXCEPCIÓN. Se reescriben `display_qty` y `display_string`; jamás `market_unit`,
# `market_qty_numeric`, `base_qty` ni `base_unit`. Razón concreta: `/restock` («ya compré la lista»)
# construye las filas de `user_inventory` con `market_qty_numeric` + `market_unit`, así que
# convertir el DATO metería gramos donde la deducción espera libras — la Nevera descontaría mal y
# en silencio. Es la misma trampa por la que se descartó el arreglo propuesto para P1-5.
#
# Y hay UN camino por el que el display sí toca el dato: `Dashboard.jsx:4398` cae a
# `parseMarketQty(display_qty)` cuando `resolveShopQty(ing)` devuelve 0. Por eso la conversión se
# niega a actuar sobre un ítem sin cantidad numérica — es exactamente el caso donde ese fallback
# dispara. La guarda no es genérica: cubre el único hueco medido.
# tooltip-anchor: P1-UNIT-SYSTEM-BY-COUNTRY
# ============================================================
_G_POR_LB = 453.592
_G_POR_OZ = 28.3495
# Unidades de mercado que son una ORDEN DE PESAR (no el rótulo de un envase).
_UNIDADES_DE_PESO_IMPERIAL = ("lb", "lbs", "libra", "libras", "oz", "onza", "onzas")
# La cantidad va SIEMPRE al principio de la línea; anclarlo protege los rótulos entre paréntesis.
_RX_QTY_IMPERIAL = re.compile(
    r"^\s*[\d.,/¼½¾\s]+(lbs?|libras?|oz|onzas?)\b", re.I)


def _unit_system_by_country_enabled() -> bool:
    """Camino caliente de la lista ⇒ knob propio, según la convención del repo."""
    return _knob_env_bool("MEALFIT_UNIT_SYSTEM_BY_COUNTRY", True)


def _etiqueta_metrica(gramos: float) -> str:
    """454 g · 1,4 kg. Coma decimal: la lista se lee en español."""
    if gramos >= 1000:
        kg = round(gramos / 1000.0, 1)
        txt = f"{kg:.1f}".replace(".", ",")
        if txt.endswith(",0"):
            txt = txt[:-2]
        return f"{txt} kg"
    return f"{int(round(gramos))} g"


def unit_system_for_country_safe(country) -> str:
    """Espejo local del SSOT (`constants.unit_system_for_country`) con import tolerante.

    NO es una segunda tabla: delega. Existe sólo para que el camino caliente de la lista no
    reviente si el import falla, y para que el fallback ('imperial' = la conducta de hoy) esté
    escrito una vez.
    """
    try:
        from constants import unit_system_for_country as _usfc
        return _usfc(country)
    except Exception:
        return "imperial"


def _project_units_over_result(res, country) -> int:
    """Recorre el resultado del agregador (lista plana o dict por categoría) y proyecta el display
    de cada ítem. Devuelve cuántos convirtió. Espejo estructural de
    `_strip_prices_for_beta_pricing_mode`, que ya hace este mismo recorrido en el mismo sitio."""
    n = 0
    try:
        grupos = res.values() if isinstance(res, dict) else [res]
        for grupo in grupos:
            if not isinstance(grupo, list):
                continue
            for item in grupo:
                if _project_display_units_for_country(item, country):
                    n += 1
    except Exception:
        return n
    return n


def _project_display_units_for_country(market_obj, country) -> bool:
    """Reescribe SÓLO el display de un ítem a unidades métricas. True si convirtió."""
    try:
        if not isinstance(market_obj, dict) or not _unit_system_by_country_enabled():
            return False
        from constants import unit_system_for_country
        if unit_system_for_country(country) != "metric":
            return False
        unidad = str(market_obj.get("market_unit") or "").strip().lower().rstrip(".")
        if unidad not in _UNIDADES_DE_PESO_IMPERIAL:
            return False
        try:
            qty = float(market_obj.get("market_qty_numeric") or 0)
        except (TypeError, ValueError):
            return False
        if qty <= 0:
            # Ver el bloque de arriba: sin numérico, la Nevera parsea el display.
            return False
        gramos = qty * (_G_POR_OZ if unidad.startswith(("oz", "onza")) else _G_POR_LB)
        etiqueta = _etiqueta_metrica(gramos)
        tocado = False
        for campo in ("display_qty", "display_string"):
            actual = market_obj.get(campo)
            if not isinstance(actual, str) or not actual:
                continue
            nuevo, n = _RX_QTY_IMPERIAL.subn(etiqueta, actual, count=1)
            if n:
                market_obj[campo] = nuevo
                tocado = True
        return tocado
    except Exception:
        # Corre por cada ítem: una excepción aquí rompe la lista entera.
        return False


def _shoplist_sanity_cap_enabled() -> bool:
    """[P1-SHOPLIST-SANITY-CAP · 2026-08-21] Kill switch del tope de envases de condimento.

    tooltip-anchor: MEALFIT_SHOPLIST_SANITY_CAP (test_p1_shoplist_sanity_cap.py)"""
    return _knob_env_bool("MEALFIT_SHOPLIST_SANITY_CAP", True)


def _is_condiment_presentation(display_category, container_weight_g) -> bool:
    """[P1-SHOPLIST-SANITY-CAP · 2026-08-21] ¿Esta fila se vende como condimento?

    Estrecho por DOS lados a propósito: categoría de despensa Y envase pequeño. Capar comida de
    verdad sería el error opuesto y peor —el usuario compraría de menos y se quedaría sin cenar—,
    así que la alcachofa (Vegetales) y el maíz en lata (Despensa, 425 g) quedan fuera.

    Sin `container_weight_g` no sabemos qué es: fail-open, no capar. La asimetría es clara — el
    coste de no capar es un ítem feo; el de capar a ciegas, una compra corta.

    Predicado sobre el DATO, no lista de nombres: una lista habría que mantenerla cada vez que el
    catálogo crece y su fallo sería silencioso.

    tooltip-anchor: _is_condiment_presentation (test_p1_shoplist_sanity_cap.py)"""
    try:
        _g = float(container_weight_g or 0)
    except (TypeError, ValueError):
        return False
    if _g <= 0 or _g > _CONDIMENT_MAX_CONTAINER_G:
        return False
    try:
        from constants import strip_accents as _sa_c
        _cat = _sa_c(str(display_category or "").strip().lower())
    except Exception:
        _cat = str(display_category or "").strip().lower()
    return _cat.startswith("despensa")


def _apply_condiment_sanity_cap(market_obj, master_item, display_category, cycle_days) -> bool:
    """[P1-SHOPLIST-SANITY-CAP · 2026-08-21] Acota los envases de un CONDIMENTO y arrastra el
    costo. Muta `market_obj` in-place; devuelve True si recortó.

    POR QUÉ CAPAR AQUÍ SÍ ES HONESTO, y en P1-COUNTRY-KEEP-RESPECT-QTY no lo era: allí el default
    fijo IGNORABA una demanda real (653 g de almejas) y el usuario compraba de menos. Aquí la
    demanda ESTIMADA es la que está mal — un frasco de orégano de 90 g dura meses, y «1 orégano»
    repetido 30 días no son 30 frascos. El consumo real de un condimento no escala con el número
    de recetas que lo mencionan.

    EL COSTO SE RECORTA CON LA CANTIDAD. Si sólo se capara la cantidad, `shopping_cost_summary`
    seguiría contando los RD$810 de orégano y el banner de presupuesto marcaría «excedido» por un
    especiero — o sea que el defecto que más duele sobreviviría al arreglo.

    NO emite la nota de cobertura de P1-CAPPED-STAPLE-HONESTY, y es deliberado: esa nota existe
    para los caps que SÍ dejan corto (4 latas de atún que cubren 5,5 días de 30). Aquí el frasco
    cubre el ciclo de sobra, así que avisar sería crying wolf — y una nota que grita siempre se
    deja de leer, que es justo lo que hace inservible a un detector.

    tooltip-anchor: _apply_condiment_sanity_cap (test_p1_shoplist_sanity_cap.py)"""
    if not _shoplist_sanity_cap_enabled() or not isinstance(market_obj, dict):
        return False
    _envase = (master_item or {}).get("container_weight_g") if isinstance(master_item, dict) else None
    if not _is_condiment_presentation(display_category, _envase):
        return False
    try:
        _qty = float(market_obj.get("market_qty_numeric") or 0)
    except (TypeError, ValueError):
        return False
    _tope = _condiment_package_cap(cycle_days)
    if _qty <= _tope:
        return False
    _factor = _tope / _qty if _qty else 1.0
    market_obj["market_qty_numeric"] = float(_tope)
    market_obj["market_qty"] = str(_tope)
    _unidad = str(market_obj.get("market_unit") or "").strip()
    if _unidad:
        market_obj["display_qty"] = f"{_tope} {_unidad}{'s' if _tope > 1 and not _unidad.endswith('s') else ''}"
    for _k in ("estimated_cost_rd", "estimated_cost"):
        try:
            _c = market_obj.get(_k)
            if _c:
                market_obj[_k] = round(float(_c) * _factor, 2)
        except (TypeError, ValueError):
            pass
    logging.info(
        "🧂 [P1-SHOPLIST-SANITY-CAP] '%s': %.0f → %d %s (envase %.0f g, ciclo %s d). "
        "El consumo de un condimento no escala con el nº de recetas que lo mencionan.",
        market_obj.get("name"), _qty, _tope, _unidad or "envases",
        float(_envase or 0), cycle_days,
    )
    return True


def _condiment_package_cap(cycle_days) -> int:
    """[P1-SHOPLIST-SANITY-CAP · 2026-08-21] Máximo de envases de condimento para un ciclo.

    Una pizca son ~0,3 g y un sobre 14 g: tres comidas al día durante un mes son ~27 g, o sea DOS
    sobres. El tope da tres — generoso frente al consumo real y a años luz de los quince que la
    lista viva pedía.

    Nunca 0: eso borraría el condimento de la lista, que es el defecto CONTRARIO (lista incompleta
    sin aviso, el miedo explícito del dueño). Nunca revienta: corre en el camino caliente del
    agregador y una excepción aquí rompe la lista entera.

    tooltip-anchor: _condiment_package_cap (test_p1_shoplist_sanity_cap.py)"""
    import math as _math_cap
    try:
        _d = int(cycle_days or 0)
    except (TypeError, ValueError):
        _d = 0
    if _d <= 0:
        return 1
    return max(1, min(4, int(_math_cap.ceil(_d / 10.0))))


# [P1-BAKING-STAPLES · 2026-07-01] (audit v3 creatividad GAP-3) "Despensa básica" de horneado: agentes
# leudantes/aroma que los transforms insignia del owner (panqueques de avena/harina, bollos de yuca,
# arepitas) NECESITAN y que no están en el catálogo verificado con precio. VERIFIED-ONLY los amputaba en
# SILENCIO de la lista (la receta los usa, la compra no los trae → receta no cocinable tal cual; el
# backstop de texto solo cubría salsas). En vez de dropear: se listan como ~1 empaque pequeño SIN precio
# (estimated_cost_rd=None) bajo "DESPENSA BÁSICA". El coherence guard NO los escala a crítico (fantasma
# delta=inf excluido por diseño — ver "pueden ser staples no marcados" en run_shopping_coherence_guard).
# Si el owner los sube al catálogo con precio, resuelven como items normales y este keep es no-op
# (_is_verified_for_shopping gana primero). Rollback: MEALFIT_BAKING_STAPLES_KEEP=false → drop histórico.
# tooltip-anchor: P1-BAKING-STAPLES
_BAKING_PANTRY_STAPLE_TOKENS = (
    "polvo de hornear", "polvo para hornear", "levadura", "bicarbonato",
    "extracto de vainilla", "esencia de vainilla", "vainilla",
)
_BAKING_STAPLE_DEFAULT_G = 100.0


def _baking_staples_keep_enabled() -> bool:
    return _knob_env_bool("MEALFIT_BAKING_STAPLES_KEEP", True)


def is_baking_pantry_staple(name) -> bool:
    """True si `name` es un staple de horneado de la despensa básica (match substring accent-insensitive).
    Usado por el keep del aggregator (P1-BAKING-STAPLES); NO altera el filtro expected del guard (los
    staples quedan warn-only fantasma a propósito — jamás fuerzan retry)."""
    try:
        from constants import strip_accents as _sa
        low = _sa(str(name or "").lower())
        return any(tok in low for tok in _BAKING_PANTRY_STAPLE_TOKENS)
    except Exception:
        return False


# [P1-COUNTRY-SYSTEM-F2 · T5 · 2026-08-17] Generalización de P1-BAKING-STAPLES: MISMO problema,
# ámbito distinto. Los 32 alimentos que Task 5 dio de alta en `master_ingredients` para España
# (`country_gaps/es.json`, T1 — Jamón serrano, Gambas, Cordero, etc.) llevan SIN precio RD a
# propósito: España es país beta (`COUNTRY_PROFILES['ES']['is_beta']`, P1-COUNTRY-SYSTEM-F1) y su
# lista de compras corre en `pricing_mode='beta_no_prices'` (T7, `_strip_prices_for_beta_pricing_mode`
# borra el precio de TODO el aggregate igual) — no hay mercado RD que cotizar. Sin este keep,
# `_is_verified_for_shopping` (que exige precio>0) trataría estos nombres como si el LLM se los
# hubiera inventado y los dropearía en SILENCIO de la lista — el MISMO modo de fallo que motivó
# P1-BAKING-STAPLES (receta los usa, compra no los trae), ahora para un país entero en vez de 4
# staples de horneado. Mismo mecanismo (keep unpriced, ~1 paquete estimado, categoría propia),
# SEGUNDO registro de tokens con SU PROPIO knob de rollback — nunca toca
# `_BAKING_PANTRY_STAPLE_TOKENS`/`is_baking_pantry_staple` (byte-identidad DO/knob-off intacta:
# estos nombres NUNCA aparecen en un plan DO — `dish_templates.json` no los referencia). Si el
# owner sube alguno con precio real, `_is_verified_for_shopping` gana primero y este keep queda
# no-op (mismo contrato que P1-BAKING-STAPLES).
# [P1-COUNTRY-SYSTEM-F2 · ola final · 2026-08-18 · M3] CORRECCIÓN de una claim previa de este
# comentario ("el coherence guard NO necesita tocarse"): eso es CIERTO solo para el BLOQUEO (el
# carve-out `delta_pct != inf` en `run_shopping_coherence_guard` sí excusa el fantasma de delta
# infinito y nunca fuerza retry por esto) pero FALSO para el WARN — el MIRROR de ese guard (el
# filtro `expected_raw` que replica el drop del aggregator) solo llama `_is_verified_for_shopping`,
# NUNCA `is_country_catalog_unpriced_item` (ni su hermano `is_baking_pantry_staple`, el mismo
# blind spot desde P1-BAKING-STAPLES · 2026-07-01, 6 semanas antes de que Fase 2 existiera).
# Confirmado en vivo dos veces (Task 10 §5, QA con LLM real): "Tortilla de maíz"/MX y "Recao"/PR
# producen el WARN GUARD-BLIND de verified-only + `[COH-GUARD/warn] ... [aggregated_only]` — WARN
# (marker citado SIN corchetes a propósito: test_guard_blind_whitelists_water ancla con .index()
# al PRIMER literal del marker y debe seguir cayendo en el código real, no en este comentario.)
# espurio ("ausente de la lista de compras sin aviso" cuando SÍ está, solo sin precio), no un
# block. Pre-existente, de severidad baja, NO cerrado por esta ola (comparte función con 89+24+18
# tests en 3 archivos — cerrarlo bien necesita su propia ronda TDD, ver reporte de Task 10 §5).
# Rollback: MEALFIT_COUNTRY_CATALOG_UNPRICED_KEEP=false → drop histórico (mismo comportamiento pre-T5).
# tooltip-anchor: P1-COUNTRY-CATALOG-UNPRICED
_COUNTRY_CATALOG_UNPRICED_BY_COUNTRY: "dict[str, tuple[str, ...]]" = {
    # [P1-COUNTRY-CATALOG-BY-COUNTRY · 2026-08-21] La agrupación por país YA ESTABA escrita
    # aquí — en los comentarios de bloque de abajo (T5=ES, T6=MX/CO, T7=PR/US, Task 8=RD).
    # Estructura real, hecha a mano, que ningún programa podía leer: el `_vc_comprable` del
    # catálogo verificado preguntaba a la tupla PLANA, así que le ofrecía huitlacoche a un
    # español y percebes a un mexicano (medido: los renders de ES, MX y US eran el MISMO
    # string de 5777 chars — el catálogo había pasado de «sólo dominicano» a «no-dominicano»).
    # Esto NO reasigna ningún token: promueve a dato la partición que los bloques ya
    # declaraban. La tupla plana se DERIVA de aquí, así que las dos vistas no pueden
    # driftear. tooltip-anchor: P1-COUNTRY-CATALOG-BY-COUNTRY
    "ES": (
        "jamon serrano", "jamon iberico", "chorizo espanol", "morcilla", "lomo embuchado",
        "panceta iberica", "gambas", "almejas", "boquerones", "anchoas", "cordero", "requeson",
        "cuajada", "nata", "judias blancas", "judias pintas", "acelgas", "fideos", "membrillo",
        "higo", "azafran", "alioli", "turron", "mazapan", "sobrasada", "butifarra", "percebes",
        "vieira", "chistorra", "pinones", "almendra marcona", "membrillo dulce",
    ),
    # [P1-COUNTRY-SYSTEM-F2 · T6 · 2026-08-17] Mismas 46 altas de catálogo MX/CO de este task —
    # también SIN precio RD a propósito (mismo motivo que ES: países beta,
    # `pricing_mode='beta_no_prices'`). Tokens elegidos por 2 palabras cuando la palabra sola es
    # demasiado genérica o colisiona con una fila PRICED existente ('serrano' solo ⊂ 'Jamón
    # serrano'; 'ancho'/'crema'/'frijoles'/'gallina' solos son términos comunes) — verificado con
    # el MISMO sweep e2e de fix-round 1 de T5
    # (`test_is_country_catalog_unpriced_item_no_colisiona_con_ningun_nombre_del_catalogo_vivo_ni_pools`,
    # extendido para cubrir estas 46 también) contra el catálogo vivo (284 filas) + los 6 pools
    # (`DOMINICAN_*` + `COUNTRY_POOLS['ES'/'MX'/'CO']`): cero falsos positivos.
    "MX": (
        "tortilla de maiz", "jalapeno", "chile serrano", "poblano", "chipotle", "guajillo",
        "chile ancho", "habanero", "chile de arbol", "pasilla", "mulato", "nopal", "jicama",
        "epazote", "chorizo mexicano", "chorizo verde", "cecina", "frijoles refritos",
        "crema mexicana", "tuna de nopal", "flor de jamaica", "xoconostle", "achiote",
        "hoja santa", "chocolate de mesa", "panela", "huitlacoche", "chicharron",
    ),
    "CO": (
        "chorizo santarrosano", "trucha", "chontaduro", "frijol cargamanto", "suero costeno",
        "guascas", "arracacha", "lulo", "curuba", "uchuva", "arequipe", "natilla", "champus",
        "gallina criolla", "borojo", "feijoa", "granadilla", "mora",
    ),
    # [P1-COUNTRY-SYSTEM-F2 · T7 · 2026-08-17] 62 altas de catálogo PR/US de este task — también
    # SIN precio RD a propósito (países beta, `pricing_mode='beta_no_prices'`). A diferencia de
    # T5/T6, aquí se usa el NOMBRE CANÓNICO COMPLETO de cada fila como token (nunca una palabra
    # suelta) — la superficie de riesgo de esta task es más alta que T5/T6: 'queso'/'pan'/'carne'/
    # 'papa'/'mantequilla'/'chile'/'salsa'/'galletas'/'frijoles'/'aceitunas' son todas palabras
    # que YA aparecen bare o casi-bare en aliases de filas PRICED existentes (Queso blanco lleva
    # 'queso' bare; 'Mantequilla'/'Pan blanco familiar'/'Carne de res'/'Papa' son sus propios
    # nombres bare; 'Chile X' son 9 filas de T6; 'Galletas de soda' ya existe). El nombre COMPLETO
    # es, por construcción, único (dos filas no pueden compartir `name`) — mismo principio que
    # 'jamon serrano'/'chile serrano' de T5/T6, aplicado sin excepción a las 62. Verificado con el
    # MISMO sweep e2e extendido (`test_is_country_catalog_unpriced_item_no_colisiona_...`) contra
    # el catálogo vivo (346 filas) + los 7 pools (`DOMINICAN_*` + `COUNTRY_POOLS['ES'/'MX'/'CO'/
    # 'PR'/'US']`): cero falsos positivos.
    "PR": (
        "panapen", "pernil", "jamon de cocinar", "sofrito", "recao", "adobo", "alcaparrado",
        "harina de yuca", "pique", "pavochon", "bacalaitos", "ron de cocina",
        "longaniza puertorriquena", "chuleta ahumada", "sazon con culantro y achiote",
        "aceite de achiote", "queso de papa", "especias para arroz con dulce",
        "aceitunas rellenas",
    ),
    "US": (
        "tocineta", "jamon de sandwich", "salchichas", "crema agria", "crema mitad y mitad",
        "bagels", "panecillos ingleses", "pretzels", "frijoles horneados", "jarabe de arce",
        "aderezo ranch", "salsa barbacoa", "ketchup", "salsa inglesa", "malvaviscos", "coditos",
        "masa para pie", "galletas graham", "salsa de salchicha", "ensalada de macarrones",
        "chile en polvo", "sazonador para tacos", "pepperoni", "salchicha italiana",
        "mezcla para panqueques", "wafles", "azucar morena", "suero de mantequilla",
        "pan de maiz", "semola de maiz", "arandanos rojos", "duraznos", "pan rallado",
        "panecillos de mantequilla", "huevos rellenos", "nuez de castilla", "nueces pecanas",
        "queso en hebras", "queso provolone", "carne molida mixta", "bolitas de papa",
        "papas ralladas", "chili con carne",
    ),
    # [P1-COUNTRY-SYSTEM-F2 · Task 8 (RD top-up) · 2026-08-17] "Hummus" — drop real RD medido
    # (6/30d en rd_drops.json), genuinamente ausente del catálogo (USDA SÍ lo tiene: "Hummus,
    # commercial", fdc 174289). A diferencia de las altas T5-T7 (países BETA sin mercado RD que
    # cotizar), esta es una alta para RD MISMO — el mecanismo se reusa a propósito (mismo motivo
    # de fondo: SIN precio La Sirena verificado hoy) más que porque el país sea beta. Ruling
    # explícito del contrato de la task: listar como CATÁLOGO SIN PRECIO en vez de dropear, para
    # que el supermercado artificial (`supermarket_products`) pueda precificarlo después en vez
    # de perder el alimento en silencio de la lista.
    "DO": (
        "hummus",
    ),
}

# Vista plana derivada (orden estable, dedupe conservando el primero). La usan los 4 call
# sites del agregador, que NO preguntan por país a propósito: si un alimento español acaba en
# la lista de la compra hay que conservarlo venga de donde venga — ahí el fallo caro es perder
# comida en silencio, no ofrecer de más.
_COUNTRY_CATALOG_UNPRICED_TOKENS = tuple(dict.fromkeys(
    _t for _ts in _COUNTRY_CATALOG_UNPRICED_BY_COUNTRY.values() for _t in _ts))
_COUNTRY_CATALOG_UNPRICED_DEFAULT_G = 150.0

# [P1-COUNTRY-KEEP-RESPECT-QTY · 2026-08-21] Unidades que el agregador sabe convertir a peso más
# abajo (`if 'g' in units: ...`). Si la receta emitió CUALQUIERA de ellas, hay demanda real y el
# default de arriba no debe pisarla. Es la MISMA lista que consume el bloque de conversión — si
# alguien añade una unidad allí y la olvida aquí, ese alimento vuelve a salir a 150 g fijos.
_CONVERTIBLE_QTY_UNITS = ("g", "kg", "oz", "lb", "ml", "l")
# [P1-COUNTRY-KEEP-COUNT-UNITS · 2026-08-23] Conteos que no tienen una masa
# universal y deben sobrevivir hasta `apply_smart_market_units`. Son las formas
# canónicas del SSOT `canonical_units.CANONICAL_UNIT_MAP`; no incluye envases ni
# cantidades nominales (`pizca`/`al gusto`).
_COUNTABLE_QTY_UNITS = frozenset((
    "unidad", "cabeza", "diente", "hoja", "rebanada", "mazo",
))


def _country_keep_has_recipe_qty(units) -> bool:
    """¿La fila beta sin precio conserva una demanda física comprable?

    Peso/volumen continúan por el path existente. Los conteos permanecen como
    conteos: convertirlos al default de 150 g inventaría masa y rompería el
    escalamiento con el número de días.
    """
    if not _country_keep_respect_recipe_qty_enabled():
        return False
    for raw_unit, raw_qty in (units or {}).items():
        try:
            if float(raw_qty) <= 0.0001:
                continue
        except (TypeError, ValueError):
            continue
        unit = canonicalize_unit(raw_unit) or str(raw_unit or "").strip().lower()
        if unit in _CONVERTIBLE_QTY_UNITS or unit in _COUNTABLE_QTY_UNITS:
            return True
    return False


def _survives_shopping_list(name) -> bool:
    """[P1-COHERENCE-MIRROR-KEEP · 2026-08-21] «¿Este nombre sobrevive a la lista de compras?» —
    UNA pregunta, UNA respuesta, las dos orillas del coherence guard la hacen.

    EL DEFECTO QUE CIERRA. El lado AGREGADO (el agregador) tiene tres ramas: fila con precio,
    staple de horneado (P1-BAKING-STAPLES) y catálogo-país sin precio (P1-COUNTRY-SYSTEM-F2 T5).
    El lado ESPERADO (el filtro de `expected_raw` en `run_shopping_coherence_guard`) sólo replicaba
    la PRIMERA: llamaba a `_is_verified_for_shopping`, que exige precio > 0. Resultado: toda fila
    conservada-sin-precio quedaba, por construcción, «en la lista y ausente de las recetas» —
    `unknown` / `aggregated_only`, para siempre y en cada recálculo.

    Los conteos casan 1:1 en producción: el plan ES tiene 4 ítems sin precio y 4 fantasmas; el
    plan US tiene 3 y `_shopping_coherence_block_history` registra `{'unknown': 3}` en 13 entradas
    consecutivas del 2026-08-20. La doc lo atribuía a «vocabulario DO-tuned» y mandaba a ampliar
    un léxico: el mecanismo no tiene nada que ver con vocabulario.

    Orden de las ramas idéntico al `if/elif/else` del agregador a propósito — es lo que hace que
    esto sea un espejo y no una segunda opinión. `test_el_ssot_replica_las_mismas_tres_ramas_del_agregador`
    lo ancla: una cuarta rama de keep allí que se olvide aquí vuelve a romper el espejo.

    tooltip-anchor: _survives_shopping_list (test_p1_coherence_mirror_keep.py)"""
    if _is_verified_for_shopping(name):
        return True
    if _baking_staples_keep_enabled() and is_baking_pantry_staple(name):
        return True
    if _country_catalog_unpriced_keep_enabled() and is_country_catalog_unpriced_item(name):
        return True
    return False


def _filter_expected_to_shopping_survivors(expected_raw, emit_blind_warning: bool = False) -> dict:
    """[P1-COHERENCE-MIRROR-KEEP · 2026-08-21] Filtra el lado ESPERADO del guard con el mismo
    criterio que decide qué sobrevive a la lista, y —opcionalmente— emite el WARN
    VERIFIED-ONLY-GUARD-BLIND sobre lo que de verdad desapareció.

    (El tag va SIN corchetes en esta prosa a propósito: `test_guard_blind_whitelists_water`
    localiza la PRIMERA aparición de la forma con corchetes y exige la whitelist de agua en las
    1500 posiciones anteriores. Citarlo entre corchetes aquí secuestraba el ancla y ponía el guard
    rojo contra código correcto — la 8ª vez que un comentario derrota a un guard en este repo, y
    el mismo remedio que aplicó el commit 7b2a2ec.)

    Ese WARN es la ÚNICA señal que existe para «la lista de compras salió incompleta sin aviso»
    (el miedo explícito del dueño, P1-VERIFIED-ONLY-OBSERVABILITY). En país beta era 100% falsos
    positivos: acusaba al LLM de desobedecer con alimentos que estaban perfectamente en la lista.
    Un detector que grita siempre se apaga en una semana, y entonces la amputación real pasa
    desapercibida — así que arreglar el espejo le devuelve el significado sin trabajo extra."""
    if not isinstance(expected_raw, dict):
        return expected_raw
    _antes = set(expected_raw.keys())
    _filtrado = {k: v for k, v in expected_raw.items() if _survives_shopping_list(k)}
    if emit_blind_warning:
        _caidos = _antes - set(_filtrado.keys())
        # [P3-GUARD-BLIND-WATER-WHITELIST · 2026-07-05] "Agua"/"hielo"/"caldo..." NO son comprables
        # (agua de grifo): su drop del catálogo verificado es comportamiento correcto, no
        # desobediencia del LLM. Ruido medido en vivo (plan e49d44c3: WARN ×2 solo por 'Agua').
        # Match EXACTO para agua/hielo ('aguacate' no matchea); prefijo para caldos.
        # [P3-GUARD-BLIND-WATER-WHITELIST v2 · 2026-07-06] variantes de agua ("Agua fría/tibia/
        # para hervir" — vivas en el WARN) también son no-comprables; startswith("agua ") no
        # matchea 'aguacate' (exige el espacio).
        # [P1-COHERENCE-MIRROR-KEEP · 2026-08-21] Los DOS markers de arriba se conservan literales
        # al mover el bloque a este helper: `test_water_variants_whitelisted` localiza el v2 por
        # texto y `test_guard_blind_whitelists_water` exige el v1 en las 1500 posiciones ANTERIORES
        # al tag del WARN. Condensarlos en una sola línea rompió los dos — la suite completa lo
        # cazó, no el despliegue.
        _caidos = {
            x for x in _caidos
            if str(x).strip().lower() not in ("agua", "hielo")
            and not str(x).strip().lower().startswith("agua ")
            and not str(x).strip().lower().startswith("caldo")
        }
        if _caidos:
            logging.warning(
                "[VERIFIED-ONLY-GUARD-BLIND] %d ingrediente(s) de RECETAS fuera del catálogo "
                "verificado → ausentes de la lista de compras sin aviso (LLM desobedeció el "
                "prompt upstream): %s",
                len(_caidos), sorted(_caidos)[:25],
            )
    return _filtrado


def _country_keep_respect_recipe_qty_enabled() -> bool:
    """[P1-COUNTRY-KEEP-RESPECT-QTY · 2026-08-21] Kill switch del respeto a la cantidad de la
    receta en la rama de catálogo-país. `false` ⇒ vuelve el 150 g fijo (conducta T5). Toca el
    camino caliente del agregador (categoría/peso/SKU/costo), así que lleva knob propio en vez de
    hardcode, según la convención del repo.

    tooltip-anchor: MEALFIT_COUNTRY_KEEP_RESPECT_RECIPE_QTY (test_p1_country_keep_respect_qty.py)"""
    return _knob_env_bool("MEALFIT_COUNTRY_KEEP_RESPECT_RECIPE_QTY", True)


def _country_catalog_unpriced_keep_enabled() -> bool:
    return _knob_env_bool("MEALFIT_COUNTRY_CATALOG_UNPRICED_KEEP", True)


# ============================================================
# [P1-PLAN-DISPLAY-I18N · Task 5 · 2026-08-19] Glosses display-only
# de un ingrediente ("Black beans (Habichuelas negras)" / "Lechosa (papaya)") — fase 1b de
# docs/superpowers/specs/2026-08-19-plan-display-i18n-design.md, regla de
# oro: la lista de compras es SIEMPRE bilingüe, jamás inglés puro. El
# docstring de la función siguiente trae la RESTRICCIÓN DURA completa.
# ============================================================


def _display_name_en_for_item(master_item: dict) -> "str | None":
    """Gloss en inglés de la fila del master, si existe y no está vacío.

    RESTRICCIÓN DURA: esta función es el ÚNICO lugar de este archivo donde
    `name_en` puede aparecer. NUNCA lo uses en `normalize_name`, ningún
    alias, ningún matcher ni en `_is_verified_for_shopping` — la identidad
    de un ingrediente sigue resolviendo EXCLUSIVAMENTE por `name` (español
    canónico). Un campo de display que se cuela a un matcher es exactamente
    la clase de bug que P1-PANTRY-NAME-RESOLUTION cerró con escopeta; el
    test grep-proof en test_p1_plan_display_i18n.py (sección "catálogo")
    vigila que esta zona sea la única.

    Display-only: el caller lo adjunta a `market_obj["display_name_en"]`,
    nunca a `name`/`display_category`/ninguna clave que participe en
    matching. `None` cuando el master no trae el campo (catálogo aún sin
    poblar por `scripts/fill_catalog_name_en.py`) — el frontend cae en
    silencio al nombre español (mismo contrato que `_display[locale]`).

    [Ola final · FF-6] Gateado por el MISMO knob que el motor de enriquecimiento
    (`MEALFIT_PLAN_DISPLAY_I18N`, default True): la feature se documenta con UN kill
    switch y antes ese switch cubría media feature — con el knob en `false` un usuario
    en-US volvía a ver Plan y Recetas en español PERO su PDF de la lista seguía saliendo
    «Black beans (Habichuelas negras)». Estado mixto que nadie diseñó ni probó, y que
    aparece justo cuando alguien revierte por un incidente. Se lee el env aquí (helper
    local `_knob_env_bool`, mismo registro `_KNOBS_REGISTRY`) en vez de importar
    `plan_display_i18n`: acoplar el aggregator al motor no compra nada y sí arrastra
    un import pesado a un camino caliente.

    tooltip-anchor: P1-PLAN-DISPLAY-I18N
    """
    if not _knob_env_bool("MEALFIT_PLAN_DISPLAY_I18N", True):
        return None
    try:
        gloss = master_item.get("name_en") if isinstance(master_item, dict) else None
    except Exception:
        return None
    if not isinstance(gloss, str):
        return None
    gloss = gloss.strip()
    return gloss or None


def _display_gloss_es_for_item(master_item: dict) -> "str | None":
    """[P1-COUNTRY-GLOSS-SOLO-INGLES · 2026-08-23] Gloss panhispánico.

    Display-only, igual que el gloss inglés hermano: solo lee ``gloss_es`` y
    nunca cambia ``name``, aliases, categoría ni ninguna clave de matching.
    ``None`` mantiene byte-idéntico el item si la migración aún no existe o si
    el nombre canónico no es un regionalismo.
    """
    try:
        gloss = master_item.get("gloss_es") if isinstance(master_item, dict) else None
    except Exception:
        return None
    if not isinstance(gloss, str):
        return None
    gloss = gloss.strip()
    return gloss or None


def _master_category_for_unpriced_item(name) -> "str | None":
    """[P2-SHOPLIST-BETA-POLISH · 2026-08-18] Categoría REAL del master para un ítem
    unpriced-keep, para que 'Acelgas' caiga en VEGETALES y 'Membrillo' en FRUTAS en vez
    del pseudo-pasillo 'CATÁLOGO SIN PRECIO' (label interno que se filtraba al PDF del
    usuario — un comprador agrupa por pasillo del súper, no por estado de precios del
    catálogo; el estado beta lo cuenta el banner de la lista, una sola vez). Equality
    accent/case-insensitive contra `get_master_ingredients()` (cacheado, TTL) con puente
    de plural s/es — NO usa el mapa global de alias del chat (colapsa identidades a
    propósito, P1-PANTRY-NAME-RESOLUTION; y dos guards de F2 prohíben que su nombre
    aparezca siquiera en este archivo) ni re-implementa `normalize_name`. Solo corre
    para los pocos ítems del branch unpriced-keep, nunca en el camino con precio.
    `None` si no resuelve ⇒ el caller conserva el label histórico (fail-open display).
    tooltip-anchor: P2-SHOPLIST-BETA-POLISH"""
    try:
        from constants import strip_accents
        target = strip_accents(str(name or "").strip().lower())
        if not target:
            return None
        variants = {target, target + "s", target + "es"}
        if target.endswith("es"):
            variants.add(target[:-2])
        if target.endswith("s"):
            variants.add(target[:-1])
        for row in (get_master_ingredients() or []):
            rn = strip_accents(str(row.get("name") or "").strip().lower())
            if rn and rn in variants:
                cat = str(row.get("category") or "").strip()
                # [P2-COUNTRY-HOUSEKEEPING · 2026-08-21] El label de DISPLAY, no la categoría
                # cruda del master. La rama con precio devuelve 'VEGETALES' (del mapa) y ésta
                # devolvía 'Vegetales' (de la DB); el Dashboard agrupa por la cadena literal, así
                # que el usuario veía DOS secciones para el mismo pasillo del súper — una con doce
                # ítems y otra con Acelgas sola. Igual con PROTEÍNAS/Proteínas (Almejas) y
                # FRUTAS/Frutas (Membrillo).
                return _get_display_category(cat, str(name or "")) if cat else None
    except Exception:
        return None
    return None


def is_country_catalog_unpriced_item(name, country=None) -> bool:
    """True si `name` es uno de los alimentos de catálogo-país sin precio RD.

    [P1-COUNTRY-CATALOG-BY-COUNTRY · 2026-08-21] `country` es OPCIONAL y aditivo. Sin él la
    conducta es la histórica: se pregunta a la unión de los 6 países. Con él, sólo a los tokens de
    ese país. La asimetría es deliberada — los 4 call sites del agregador NO pasan país porque ahí
    conservar de más es correcto (si un alimento español acaba en la lista, hay que conservarlo:
    el fallo caro es perder comida en silencio); el único que pregunta por país es el catálogo
    verificado del generador, que es una decisión de QUÉ OFRECER. Un país no canónico cae a la
    unión: este predicado corre en el camino caliente del agregador y un país que no reconozco no
    puede estrechar el filtro.

    [fix-round 1 · review IMPORTANT · 2026-08-17] Match por TOKEN completo (word-boundary,
    accent-insensitive, tolerante a plural — mismo patrón que `_scan_allergen_violations`/
    `pantry_names_match`: `\\b<token>(?:s|es)?\\b`), NUNCA `tok in low` (substring bare). El bare
    `in` original dejaba pasar `'pinones' in 'champinones'` (Piñones ⊂ Champiñones,
    accent-stripped) — `Champiñones` es una fila RD PRICED real de `DOMINICAN_VEGGIES_FATS`, así
    que el bug marcaba un alimento de precio real como si fuera una alta sin precio de T5. 17ª
    colisión de substring documentada en el proyecto (sal⊂salsa, pollo⊂repollo, res⊂fresco...).
    Usado por el keep del aggregator (generalización de P1-BAKING-STAPLES, T5).

    [fix-round 1 T6 · review Critical #2 · 2026-08-17] `"tortilla de maiz"` es el ÚNICO de los
    78 tokens (32 T5 + 46 T6) con un camino de FALSO POSITIVO independiente de si la fila
    realmente resolvió: `resolve_preparation_distinct` intercepta CUALQUIER "tortilla(s) de
    maiz" ANTES de los tiers normales, y con el knob `MEALFIT_COUNTRY_SYSTEM` apagado devuelve
    `(True, None)` (pass-through histórico — ver ese docstring) — el pass-through ECOA el texto
    original ("Tortilla de maíz" tal cual lo escribió el usuario/la receta), que ESTE matcher
    reconocería igual sin que hubiera resuelto de verdad a la fila. Verificado en vivo contra el
    agregador real: con el gate del resolver YA puesto pero SIN este segundo gate,
    `aggregate_and_deduct_shopping_list(['80 g de Tortilla de maíz'])` con el knob apagado SEGUÍA
    sobreviviendo como CATÁLOGO SIN PRECIO — la fila no existía para efectos de DO antes de esta
    task (`db_inventory.py` PANTRY_UNIT_HINTS ya anticipaba el string en flujos de Nevera DO), así
    que debía dropearse, byte-idéntico. Ningún otro de los 78 tokens comparte este riesgo (ningún
    otro nombre calza los 3 regex pre-existentes de `resolve_preparation_distinct` —
    harina-de-X/caldo-de-X/crema-de-coco — que son los únicos que devuelven `(True, None)`
    incondicionalmente antes de esta task)."""
    try:
        from constants import strip_accents as _sa
        low = _sa(str(name or "").lower())
        tokens = _COUNTRY_CATALOG_UNPRICED_TOKENS
        if country is not None:
            # `canonicalize_country` es el ÚNICO SSOT de países (lección P1-DIET-CANON-SSOT): aquí
            # no nace una segunda tabla. Si no reconoce el valor, se queda la unión (fail-open).
            try:
                from constants import canonicalize_country as _cc_iccui
                _cc = _cc_iccui(country)
                # `canonicalize_country` cae a 'DO' ante CUALQUIER basura, así que su resultado no
                # distingue «el usuario es dominicano» de «no entendí el valor». Sin este
                # round-trip, un país mal tecleado estrecharía el filtro al único token de DO y
                # borraría comida de la lista en silencio — el fallo caro exacto que este
                # predicado existe para evitar. Con él, lo no reconocido se queda en la unión.
                if str(country).strip().upper() != _cc:
                    _cc = None
            except Exception:
                _cc = None
            _propios = _COUNTRY_CATALOG_UNPRICED_BY_COUNTRY.get(_cc) if _cc else None
            if _propios:
                tokens = _propios
        if not _knob_env_bool("MEALFIT_COUNTRY_SYSTEM", False):
            tokens = tuple(t for t in tokens if t != "tortilla de maiz")
        return any(
            re.search(r"\b" + re.escape(tok) + r"(?:s|es)?\b", low)
            for tok in tokens
        )
    except Exception:
        return False


DEFAULT_G_PER_TAZA = 150

# ============================================================
# [P1-3] Aliases de unidades de contenedor + fallback de peso por categoría.
# ------------------------------------------------------------
# El aggregator de la lista de compras necesita normalizar unidades híbridas
# tipo "1 paquete de arroz" a gramos para deducir contra el inventario que
# está en peso (g/lb). El bloque normalizador requiere DOS condiciones:
#   1. La unidad textual está en el set `_CONTAINER_UNIT_ALIASES`.
#   2. El item tiene `container_weight_g > 0` en master_ingredients (poblado
#      manualmente por el operador para SKUs estandarizados como
#      "Arroz Marca X 1 lb / 453g").
#
# ANTES, ambas condiciones eran AND estricto. Si master no tenía
# `container_weight_g` (común para SKUs sin curar) o el usuario tipeaba un
# alias no contemplado (ej. "1 caja de leche"), la unidad quedaba sin
# convertir → el inventario seguía como `units['paquete']=1` mientras el
# plan acumulaba `units['g']=500`. Resultado: el item APARECÍA en la lista
# de compras dos veces (uno por peso, otro por paquete) y el delta no se
# calculaba — el usuario compraba duplicado.
#
# AHORA:
#   - El set de aliases se amplía para cubrir 'caja', 'cajas', 'tetra',
#     'tetrapak', 'galón', 'galones', 'jarra', 'jarras', 'bolsa', 'bolsas'.
#     Estos son envases reales del mercado dominicano que el LLM o el
#     usuario pueden usar.
#   - Si `container_weight_g` no está en master, el helper
#     `_fallback_container_weight_g(category)` retorna un peso estimado
#     conservador por categoría (mejor estimar que dejar el item sin
#     normalizar).
# ============================================================
_CONTAINER_UNIT_ALIASES = frozenset({
    'paquete', 'paquetes', 'pqte', 'pqtes',
    'pote', 'potes', 'tarro', 'tarros',
    'lata', 'latas',
    'cartón', 'carton', 'cartones', 'cartones.', 'cartón.',
    'envase', 'envases',
    'botella', 'botellas', 'botellita', 'botellitas',
    'frasco', 'frascos',
    'funda', 'fundas', 'fundita', 'funditas',
    'caja', 'cajas',
    'tetra', 'tetrapak', 'tetra-pak',
    'galón', 'galon', 'galones',
    'jarra', 'jarras',
    'bolsa', 'bolsas', 'bolsita', 'bolsitas',
    'sobre', 'sobres', 'sobrecito', 'sobrecitos',
})

# Pesos default por categoría cuando master_ingredients NO tiene
# `container_weight_g` poblado. Defaults conservadores que reflejan tamaños
# típicos del mercado dominicano (cartón de leche 1L, paquete de arroz 1lb,
# pote de mantequilla 250g, etc.). Mejor under-estimate que dejar el item
# sin normalizar (lo que produciría duplicación en el delta).
_FALLBACK_CONTAINER_WEIGHT_G_BY_CATEGORY = {
    "lácteos":         1000.0,  # cartón leche 1L, yogur grande
    "lacteos":         1000.0,
    "bebidas":         1000.0,  # tetra jugo 1L
    "despensa":         450.0,  # paquete arroz / pasta 1lb
    "despensa y granos": 450.0,
    "víveres":          450.0,
    "viveres":          450.0,
    "granos":           450.0,
    "aceites":          950.0,  # botella aceite 1L
    "salsas":           250.0,  # frasco salsa mediano
    "especias":          50.0,  # sobre/frasquito condimento
    "proteínas":        500.0,  # paquete embutido
    "proteinas":        500.0,
    "frutas":           500.0,
    "vegetales":        500.0,
    "suplementos":      500.0,
}
_DEFAULT_FALLBACK_CONTAINER_WEIGHT_G = 500.0  # genérico cuando categoría no matchea


def _fallback_container_weight_g(category: str | None) -> float:
    """[P1-3] Estima el peso por contenedor por categoría cuando
    master_ingredients no tiene el dato curado. Defensivo: nunca lanza."""
    if not category:
        return _DEFAULT_FALLBACK_CONTAINER_WEIGHT_G
    cat_norm = str(category).strip().lower()
    return _FALLBACK_CONTAINER_WEIGHT_G_BY_CATEGORY.get(
        cat_norm, _DEFAULT_FALLBACK_CONTAINER_WEIGHT_G
    )


# ============================================================
# [VISIÓN-C / HYBRID-SHOPPING-LIST] Clasificación de items en
# 'staple' (despensa, compras mensuales) vs 'perishable' (compras
# semanales por shelf-life corto).
# ------------------------------------------------------------
# Ver discusión 2026-05-06: la lista mensual extrapolaba ×9.33 todos
# los items del chunk 1, produciendo cantidades absurdas en perecederos
# (9 lbs fresas, 6 lbs yogurt) y faltantes de chunks 2-8 con menús
# distintos. La solución Visión-C combina:
#   - Staples (paleta base reutilizada por GROCERY-CYCLE-LOCK) →
#     extrapolación mensual completa (multiplier × cycle_weeks).
#   - Perishables → multiplier de 1 semana (rotan según chunk vigente).
#
# Heurística de clasificación:
#   1. category in {'Despensa'} → staple (granos, aceites, especias,
#      conservas, harinas — shelf > 30 días típicamente).
#   2. category in {'Frutas','Vegetales'} → perishable (3-14 días).
#   3. category in {'Lácteos'}: depende. Yogurt/queso fresco → perishable;
#      leche UHT/queso curado → staple. Decidir por shelf_life_days.
#   4. category in {'Proteínas','Víveres'}: idem mixto. Carnes/pescados
#      frescos → perishable; tubérculos enteros → staple si shelf >= 21.
#   5. shelf_life_days >= STAPLE_SHELF_THRESHOLD_DAYS → staple.
#   6. shelf_life_days < STAPLE_SHELF_THRESHOLD_DAYS → perishable.
#
# Conservador: si dudas, perishable (mejor sub-comprar y rotar que
# sobre-comprar y desperdiciar).
# ============================================================
# [P2-1 · 2026-05-08] `_knob_env_int` registra en `_KNOBS_REGISTRY`.
STAPLE_SHELF_THRESHOLD_DAYS = max(7, _knob_env_int("MEALFIT_STAPLE_SHELF_THRESHOLD_DAYS", 21))

_STAPLE_CATEGORIES = {
    'despensa', 'granos', 'cereales', 'conservas', 'enlatados',
    'aceites', 'salsas', 'especias', 'condimentos',
}
_PERISHABLE_CATEGORIES = {
    'frutas', 'vegetales', 'hierbas', 'verduras',
}

# [P1-PAN-PERECEDERO · 2026-05-16] Excepciones a `_STAPLE_CATEGORIES`:
# items con `category="Despensa"` que SON realmente perecederos en RD.
#
# Bug observado en lista de compras del plan aeb25e1c: "Pan integral 1 paquete
# (1.3 lbs)" aparecía en sección "DESPENSA — ESTABLES +7 DÍAS" junto a aceite,
# arroz, sal. Pero pan integral fresco dura 5-7 días en cocina (~10d refrigerado).
# El usuario podía pensar "tengo 14+ días para usarlo" y se le mohecía.
#
# Causa: master_ingredients tiene Pan integral con category="Despensa" +
# shelf_life_days=14 (default genérico — NO refleja realidad de panes frescos).
# El matcher de `_classify_perishability` retorna "staple" al matchear cat
# en `_STAPLE_CATEGORIES`, ANTES de evaluar el shelf_life_days real.
#
# Solución: substring match contra el nombre (post strip_accents + lowercase)
# DENTRO de la rama _STAPLE_CATEGORIES. Items canónicamente catalogados como
# Despensa pero con shelf_life real ≤7d se rerutean a "perishable".
#
# Casabe (cracker totalmente deshidratado) SÍ es staple verdadero — dura meses.
# Galletas de soda 90d, galletas de arroz 30d → también staple. Solo los panes
# blandos frescos (sin proceso de horneado prolongado + bajo contenido de
# humedad) caen en esta excepción.
_DESPENSA_PERISHABLE_EXCEPTIONS = frozenset({
    'pan integral',
    'pan de agua',
    'pan blanco',
    'pan dulce',
    # [P1-TORTILLA-PERECEDERO · 2026-07-06] (review visual del plan de 30 días) Las
    # tortillas/wraps integrales de trigo (Toufayan, etc.) tienen category="Despensa"
    # en master pero son pan blando: duran ~1 semana refrigeradas, se enmohecen mucho
    # antes de los 30 días. Aparecían en "DESPENSA DEL MES — COMPRA UNA SOLA VEZ"
    # (12 wraps comprados el día 1 para todo el mes) → se dañan. El substring 'tortilla'
    # cubre "tortilla integral"/"tortilla de trigo"/"tortilla de maíz" (todas frescas).
    'tortilla',
    # NO incluir: casabe (deshidratado), galletas (selladas, secas), pan tostado,
    # tostones/chips de tortilla (secos, sellados — no contienen 'tortilla' salvo
    # "chips de tortilla", ausente del catálogo verificado RD).
})

# Heurística por nombre cuando category es ambigua (Lácteos/Víveres/Proteínas).
_PERISHABLE_NAME_HINTS = (
    'fresc', 'crud', 'congelad',  # 'fresca', 'fresco', 'cruda', 'congelado'
    'yogurt', 'yogur', 'queso fresco', 'queso de hoja', 'queso de freir',
    'queso de freír', 'queso blanco',
    'leche fresca', 'crema', 'mantequilla',
    'pollo', 'pechuga', 'pavo', 'res', 'carne', 'cerdo', 'chuleta',
    'pescado', 'tilapia', 'mero', 'salmon', 'salmón', 'camaron', 'camarón',
    'mariscos', 'atun fresco',
)
_STAPLE_NAME_HINTS = (
    'leche uht', 'leche en polvo', 'leche evaporada',
    'queso parmesano', 'queso curado',
    # [2026-05-07] Variantes adicionales de enlatados. master_ingredients
    # tiene 'Atún en agua' / 'Atún en aceite' como nombre canónico (no
    # 'Atún en lata'), y los hints originales solo cubrían "en lata".
    # Añadidas variantes "en agua" / "en aceite" para que el classifier
    # las marque como staple en hybrid (path biweekly/monthly).
    'atun en lata', 'atún en lata', 'atun enlatado', 'atún enlatado',
    'atun en agua', 'atún en agua', 'atun en aceite', 'atún en aceite',
    'pollo en lata', 'pollo enlatado',
    'sardinas', 'salmon en lata', 'salmón en lata',
    'arroz', 'pasta', 'lenteja', 'garbanzo', 'frijol', 'habichuela',
    'gandules', 'avena', 'harina',
    'aceite', 'vinagre', 'sal', 'azucar', 'azúcar', 'estevia',
    'salsa de tomate', 'pasta de tomate',
    'canela', 'oregano', 'orégano', 'comino', 'pimienta', 'sazon', 'sazón',
    # [P1-PAN-PERECEDERO · 2026-05-16] 'pan integral' REMOVIDO. Panes
    # frescos cubiertos por `_DESPENSA_PERISHABLE_EXCEPTIONS` (rerutea a
    # perishable aunque category=Despensa). Casabe (cracker deshidratado)
    # y galletas (selladas, secas) siguen siendo staples reales.
    'casabe', 'galletas',
    'mantequilla de mani', 'mantequilla de maní',
    'almendras', 'nueces',
)


def _classify_perishability(name: str, master_item: dict | None = None) -> str:
    """Clasifica un ingrediente como 'staple' o 'perishable'.

    Orden de precedencia (alto → bajo):
      1. Category exacta (alta confianza: 'Despensa', 'Frutas', etc.).
      2. Heurística por nombre (substrings curados).
      3. shelf_life_days del master (>= STAPLE_SHELF_THRESHOLD_DAYS → staple).
      4. Default: perishable (conservador).

    [2026-05-06 FIX] Antes shelf_life_days corría primero. master_ingredients
    persiste shelf_life_days=14 como default genérico para casi todos los
    items de Despensa (pan, aceite, miel, almendras, especias, granos…) —
    valor incorrecto pero ampliamente desplegado. Con threshold=21, ese 14
    devolvía "perishable" para staples obvios y `_build_hybrid_shopping_list`
    los marcaba `is_perishable=True`, contaminando la sección "Compra esta
    semana — Perecederos" del PDF con items que el usuario sabe que duran
    meses (aceite, miel, especias). Mover category/name hints adelante deja
    que la señal fuerte (cat='Despensa' del master, nombres canónicos como
    'pan integral'/'arroz'/'aceite') gane sobre el dato shelf default.
    shelf_life_days sigue siendo señal cuando NO hay categoría ni nombre
    reconocible (cubre items raros del LLM no registrados en master).
    """
    from constants import strip_accents
    name_lower = (name or "").lower().strip()
    name_norm = strip_accents(name_lower)
    # [DESCRIPTOR-FIX] Eliminar descriptores negativos antes del match por
    # palabra. "Yogurt sin azúcar" no es azúcar; "Leche bajo en grasa" no es
    # grasa. Si dejamos esos modificadores en el string, hints como "azucar"
    # / "sal" hacen match falso positivo y un yogurt termina como staple.
    name_for_hints = re.sub(r'\bsin\s+\w+', '', name_norm)
    name_for_hints = re.sub(r'\b(bajo|reducid[oa]|libre)\s+(de|en)\s+\w+', '', name_for_hints)
    name_for_hints = name_for_hints.strip()

    # 1. Category exacta (cuando es inequívoca). master_ingredients.category
    # es producto de curación humana — gana sobre datos numéricos default.
    cat = ""
    if isinstance(master_item, dict):
        cat = strip_accents(str(master_item.get("category", "") or "").lower().strip())
    if cat in _STAPLE_CATEGORIES:
        # [P1-PAN-PERECEDERO · 2026-05-16] Excepciones: items catalogados como
        # Despensa pero perecederos en realidad (panes frescos). Sin esta
        # excepción, pan integral terminaba en "Despensa estables +7 días"
        # cuando debe estar en "Perecederos esta semana".
        if any(exc in name_norm for exc in _DESPENSA_PERISHABLE_EXCEPTIONS):
            return "perishable"
        return "staple"
    if cat in _PERISHABLE_CATEGORIES:
        return "perishable"

    # 2. Heurística por nombre (más específica primero).
    # Staples más específicos: si el nombre contiene "atun en lata" / "leche uht"
    # tiene precedencia sobre el match genérico de "atun" / "leche" perishable.
    # Usamos `name_for_hints` (sin "sin X" / "bajo en X") para evitar falsos
    # positivos como "yogurt sin azúcar" → staple por azúcar.
    for hint in _STAPLE_NAME_HINTS:
        if hint in name_for_hints:
            return "staple"
    for hint in _PERISHABLE_NAME_HINTS:
        if hint in name_for_hints:
            return "perishable"

    # 3. shelf_life_days como fallback cuando ni cat ni nombre dieron señal.
    if isinstance(master_item, dict):
        shelf = master_item.get("shelf_life_days")
        if shelf is not None:
            try:
                shelf_int = int(shelf)
                if shelf_int >= STAPLE_SHELF_THRESHOLD_DAYS:
                    return "staple"
                else:
                    return "perishable"
            except (TypeError, ValueError):
                pass

    # 4. Default conservador.
    return "perishable"


def _build_hybrid_shopping_list(
    weekly_items: list,
    period_items: list,
    master_map: dict | None = None,
    restocked_at_iso: str | None = None,
    restocked_items: dict | None = None,
) -> list:
    """[VISIÓN-C] Combina lista semanal y lista del periodo (quincenal/mensual)
    en una lista híbrida:
      - Items 'staple' → cantidad del periodo completo (compra una vez).
      - Items 'perishable' → cantidad semanal (compra recurrente).

    Cada item en la salida lleva un campo `is_perishable: bool` (alineado con
    el SSOT P1-PDF-2 que el frontend ya consume vía `item_ref.is_perishable`)
    para que pueda renderizar 2 secciones separadas sin cambios.

    Si un item está SOLO en uno de los dos sets, se incluye con su clasificación
    (raro pero posible si caps cambian la composición entre multipliers).

    [RIESGO-1 FIX] Si `restocked_at_iso` está presente y la última compra de
    perecederos fue hace <`MEALFIT_PERISHABLE_CYCLE_DAYS` (default 7), los
    perecederos se EXCLUYEN del output. Razón: los chunks merge cada 3 días
    pero el usuario compra perecederos cada 7. Sin este filtro, las recalc
    intermedias muestran "compra 0.43kg pollo" porque el delta inventario
    refleja consumo parcial. Con el filtro, perecederos se mantienen ocultos
    hasta que toque el próximo ciclo de compra.

    [P1-2 FIX] `restocked_items: {ingredient_name_norm: iso_ts}` permite supresión
    item-level. Si el usuario solo compró fresas el lunes, solo "fresas" se suprime
    durante el ciclo; pollo/yogurt siguen visibles si no fueron comprados.
    Precedencia: `restocked_items` (item-level) > `restocked_at_iso` (blanket legacy).
    """
    from constants import strip_accents
    from datetime import datetime, timezone

    # [P1-6] Knobs de cycle (compartidos entre rama blanket y rama item-level).
    # [P1-A · 2026-05-08] Lazy import de `_env_int` para auto-registrar en
    # `_KNOBS_REGISTRY` (mismo patrón que `_get_coherence_tolerance_pct`/
    # `_get_coherence_guard_mode` aquí mismo). Fallback defensivo a
    # [P2-1 · 2026-05-08] Helpers `_knob_env_int` ya importados a top-level
    # desde `knobs.py` (cero ciclo); no requiere lazy import / fallback.
    _max_cap = max(7, min(_knob_env_int("MEALFIT_PERISHABLE_CYCLE_DAYS_MAX", 30), 90))
    cycle_days = max(1, min(_knob_env_int("MEALFIT_PERISHABLE_CYCLE_DAYS", 7), _max_cap))
    now_utc = datetime.now(timezone.utc)

    def _ts_within_cycle(iso_ts: str) -> bool:
        """True si `iso_ts` cae dentro del ciclo activo (suprimir el item)."""
        if not iso_ts or not isinstance(iso_ts, str):
            return False
        try:
            ts = iso_ts.replace("Z", "+00:00") if iso_ts.endswith("Z") else iso_ts
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_days = (now_utc - dt).total_seconds() / 86400.0
            return age_days < cycle_days
        except (ValueError, TypeError):
            return False

    # [P1-2] Item-level: precedencia sobre restocked_at_iso.
    suppress_by_item: dict[str, bool] = {}
    if isinstance(restocked_items, dict) and restocked_items:
        for raw_name, iso_ts in restocked_items.items():
            if not isinstance(raw_name, str):
                continue
            key = strip_accents(raw_name.lower().strip())
            if key and _ts_within_cycle(iso_ts):
                suppress_by_item[key] = True

    # [RIESGO-1] Blanket: aplica a TODOS los perecederos cuando no hay item-level.
    # Si hay item-level, ignoramos el blanket — el usuario eligió granularidad.
    suppress_perishables_blanket = (
        not suppress_by_item
        and bool(restocked_at_iso)
        and _ts_within_cycle(restocked_at_iso)
    )
    if not isinstance(weekly_items, list):
        weekly_items = []
    if not isinstance(period_items, list):
        period_items = []
    master_map = master_map or {}

    def _name_key(item):
        if not isinstance(item, dict):
            return ""
        return strip_accents(str(item.get("name", "")).lower().strip())

    weekly_by_name = {_name_key(i): i for i in weekly_items if isinstance(i, dict)}
    period_by_name = {_name_key(i): i for i in period_items if isinstance(i, dict)}

    all_names = set(weekly_by_name.keys()) | set(period_by_name.keys())
    hybrid = []

    for name_key in all_names:
        weekly_item = weekly_by_name.get(name_key)
        period_item = period_by_name.get(name_key)

        # Tomar nombre canónico del item disponible (period_item primero por ser
        # el contexto del usuario; weekly como fallback).
        ref_item = period_item or weekly_item
        name_canon = ref_item.get("name", "") if ref_item else ""
        # [RIESGO-2 FIX] master_map normalmente está vacío en producción porque
        # los call-sites no lo pasan. Pero P1-PDF-2 ya inyecta `shelf_life_days`,
        # `category` e `is_perishable` directamente en el item. Por eso el item
        # mismo es un master_item válido — usarlo como fallback evita que el
        # clasificador caiga al default conservador "perishable" cuando hay
        # data confiable en el propio item.
        master_item = (
            master_map.get(name_key)
            or master_map.get(name_canon.lower())
            or ref_item
            or {}
        )

        perishability = _classify_perishability(name_canon, master_item)

        if perishability == "perishable":
            chosen = weekly_item or period_item  # weekly preferido
        else:
            chosen = period_item or weekly_item  # period (mensual) preferido

        if not chosen:
            continue
        # [P1-2] Supresión item-level: si este nombre fue restocked dentro del
        # ciclo, ocultarlo (independiente de su clasificación de perecedero —
        # un staple que el usuario marcó como recién comprado tampoco debería
        # aparecer hasta el próximo ciclo).
        if suppress_by_item.get(name_key):
            continue
        # [RIESGO-1] Blanket legacy: si no hay item-level, suprimir todos los
        # perecederos durante el ciclo.
        if suppress_perishables_blanket and perishability == "perishable":
            continue
        # Marca para el frontend.
        out_item = dict(chosen)  # copia superficial
        out_item["is_perishable"] = (perishability == "perishable")
        out_item["_perishability"] = perishability
        # [P1-OVERCOVER-LABEL · 2026-07-28] El costo del ciclo YA sabe que un envase que
        # cubre varias semanas no se recompra semanal (P1-CYCLE-REPURCHASE-HONEST), pero el
        # USUARIO no: el tarro de cottage 16 Oz (cover ~5×) salía en la sección semanal SIN
        # letrero (caso vivo, plan 9af221fb) — recompra implícita cada ida. Con cobertura
        # ≥2× en un perecedero, el display declara cuántos días alcanza de verdad (capado
        # por la vida útil del abierto). Va AQUÍ y no al armar el result: `is_perishable`
        # nace en esta etapa (el intento anterior corría antes y nunca la veía).
        try:
            _ocr = float(out_item.get("pkg_cover_ratio") or 0)
            if (out_item["is_perishable"] and _ocr >= 2.0
                    and "alcanza" not in str(out_item.get("display_qty", ""))):
                _cubre_d = 7.0 * _ocr
                _vida_d = float(out_item.get("shelf_life_days") or 0)
                if _vida_d > 0:
                    _cubre_d = min(_cubre_d, _vida_d)
                _cubre_i = max(8, int(round(_cubre_d)))
                out_item["display_qty"] = (
                    f"{out_item.get('display_qty', '')} · alcanza ~{_cubre_i} días — "
                    f"no recompres cada semana")
        except (TypeError, ValueError):
            pass
        hybrid.append(out_item)

    return hybrid


# ============================================================
# [P1-TRIP-WINDOWED-PERISHABLES · 2026-08-02] La lista del viaje traía el PROMEDIO
# del plan, no la semana que el usuario va a cocinar.
# ------------------------------------------------------------
# `get_shopping_list_delta` promedia TODOS los días materializados y proyecta a 7
# (`base_duration_scale = 7.0 / num_days`). Eso es correcto mientras las semanas del
# plan sean intercambiables — y DEJARON de serlo: el seeder hace las semanas
# deliberadamente distintas (freq-tracking cross-chunk). Con 14 días materializados
# (pollo la semana 1, pescado la semana 2) el viaje 1 traía la MITAD del pollo que sí
# se cocina esta semana y la mitad del pescado que no se cocina hasta dentro de 8 días
# (se daña en la nevera). Con 27 días la fracción es peor.
#
# El guard de coherencia era estructuralmente CIEGO a esto: espeja la misma fórmula
# (`P1-COHERENCE-DAY-BASIS` escala el lado esperado ×7/num_days), así que el promedio
# se cancelaba a ambos lados y la divergencia jamás aparecía.
#
# Fix: SOLO los PERECEDEROS se ventanean a los 7 días del viaje activo. Los ESTABLES
# siguen saliendo del agregado del periodo completo, que es lo correcto: se compran UNA
# vez para todo el ciclo (misma partición que `_build_hybrid_shopping_list` ya hace
# entre lista semanal y lista de periodo, con el MISMO clasificador
# `_classify_perishability` — dos criterios distintos se contradirían: un ítem podría
# ventanearse aquí y tomarse del periodo allá).
#
# Cobertura de los días 8..N: el plan es una ventana RODANTE. El shift
# (`_background_shift_plan_for_user` / `/shift-plan`) poda los días ya consumidos, deja
# `days[0] = hoy` y reescribe `grocery_start_date`; cada chunk que aterriza dispara el
# rebuild T2 de las listas. Así el viaje k ve la semana k. La suma de los N viajes sigue
# cubriendo el plan (se re-reparte en el tiempo, no se recorta): antes cada viaje traía
# el promedio ×7 días, ahora cada viaje trae SUS 7 días. Ver la sección de riesgos del
# reporte del P-fix.
#
# ------------------------------------------------------------
# [P1-TRIP-WINDOWED-PERISHABLES · ronda 1 · 2026-08-02] DEFAULT **OFF**. Capacidad
# dormida, NO código muerto: la ventana solo entra cuando `len(days) > 7`, y esa
# condición HOY no se cumple nunca en producción.
#
# Medición contra la DB de producción (2026-08-02, SELECT sobre los 40 planes más
# recientes; 23 con datos): `n_days=2` en 3 planes, `n_days=3` en 20 planes.
# **0 planes por encima de 3 días materializados.** El shift poda los días consumidos
# a la misma velocidad a la que los chunks los añaden, así que `len(days)` orbita 2-3
# de forma permanente y `active_trip_window_days` devuelve `None` siempre.
#
# O sea: hoy el ventaneo es INERTE y sus riesgos (abajo) sí son reales. Se deja
# implementado, probado y apagado hasta que alguien cierre TODOS los prerequisitos
# enumerados abajo y vuelva a medir la distribución de `n_days`.
#
# [review final audit-v7-p1 · 2026-08-03 · KNOB-3] Aquí (y en dos sitios más) decía «los 3
# prerequisitos» mientras la lista enumeraba CUATRO: el (d) se añadió al corregir el reporte
# de la ronda 1 y el conteo se quedó atrás. Un operador que lea el número cierra (a), (b) y (c)
# y enciende con (d) abierto — el más silencioso de los cuatro. Se deja de escribir el número:
# un conteo literal al lado de una lista es deuda que caduca sola.
#
# ### Condiciones para ENCENDER (`MEALFIT_TRIP_WINDOWED_PERISHABLES=true`)
#
# (a) **El shift debe reconstruir —o marcar para recálculo— la lista de compras.**
#     Hoy NINGUNO de los dos paths del shift toca `aggregated_shopping_list`
#     (`_background_shift_plan_for_user` en cron_tasks.py y `api_shift_plan` en
#     routers/plans.py: 0 referencias en ambos cuerpos). Sin la ventana, una lista
#     stale post-shift es un promedio viejo benigno; CON la ventana, la lista habla de
#     una semana que ya pasó y el guard produce una divergencia SEVERA
#     (`Pescado expected_only` / `cap_swallowed_modifier`, que escala warn→block por
#     P2-COHERENCE-1 y puede desatar retries). Mecanismo candidato que YA existe para
#     drenarlo: marcar el plan como `partial_no_shopping` en el shift para que el cron
#     `_process_pending_shopping_lists` (housekeeping al tope de
#     `process_plan_chunk_queue`, ~cada minuto) lo re-agregue con el `days` ya avanzado.
#
# (b) **`cycle_total_rd` pasa a extrapolar la SEMANA 1 al ciclo**, no el promedio del
#     plan (`compute_shopping_cost_summary` multiplica los perecederos del viaje ×
#     semanas). Ese número alimenta `budget_reconciliation`, y un
#     `status == "excedido"` dispara la convergencia de presupuesto, que **sustituye
#     alimentos en el plan** (`apply_budget_convergence_for_days`). Un plan cuya semana
#     cara sea la 3 puede dejar de converger (o converger de más si la cara es la 1).
#     Hay que MEDIR el impacto sobre la distribución de `status` antes de encender.
#
# (c) **El último chunk de un plan de 30 días no tiene rebuild posterior**: el tramo
#     final del plan no se re-agrega, así que su ventana nunca avanza.
#
# (d) [hallazgo al corregir el reporte, ronda 1] Los dos callsites read-only de tools.py
#     (`check_shopping_list` —la tool del coach «qué me falta comprar»; el nombre
#     `calculate_shopping_list` que este bloque citaba NO existe en el repo, review final
#     2026-08-03— y `mark_shopping_list_purchased`) NO están cableados y
#     SÍ piden `structured=True` — con la ventana activa reconstruirían el promedio del
#     plan mientras el usuario compró la lista ventaneada. Para `mark_shopping_list_
#     purchased` eso significa cargar a la despensa una lista distinta de la comprada.
#     Cablearlos (o hacerlos leer la lista persistida) es parte de encender.
#
# El knob gobierna **cómo se construyen listas nuevas**. NUNCA cómo se interpretan las
# ya construidas: el espejo del guard se dispara por el SELLO `trip_window_days` de la
# propia lista, sin consultar el knob (ver `_mirror_trip_window_expected` y el
# `ignore_knob` de `active_trip_window_days`). Sin esa asimetría, apagar el knob con
# listas ya selladas en DB fabricaba exactamente la divergencia severa que el espejo
# existe para evitar — medido: knob ON → `[]`, knob OFF → `Pescado expected_only` +
# `Pollo 700 vs 1400`.
# tooltip-anchor: P1-TRIP-WINDOWED-PERISHABLES
# ============================================================
TRIP_WINDOW_DAYS = 7


def _trip_windowed_perishables_enabled() -> bool:
    """[P1-TRIP-WINDOWED-PERISHABLES · 2026-08-02] Knob de CONSTRUCCIÓN de listas nuevas.

    Default **False** (ronda 1): el ventaneo solo aplica con `len(days) > 7` y la
    medición contra producción del 2026-08-02 da 0/23 planes vivos por encima de 3 días
    materializados — encenderlo hoy no cambiaría ninguna lista, pero sí expondría los
    riesgos documentados en el bloque de arriba (son CUATRO, (a)-(d); el bloque los enumera
    y ya no los cuenta). `True` reactiva el ventaneo.

    NO gobierna la INTERPRETACIÓN de listas ya construidas: el espejo del guard usa el
    sello `trip_window_days` y jamás este knob.
    """
    return _knob_env_bool("MEALFIT_TRIP_WINDOWED_PERISHABLES", False)


def _parse_plan_day_date(value):
    """`'YYYY-MM-DD'` o ISO completo → `date`. `None` si no parsea (los planes
    pre-P1-CHAT-PAST-DAYS no traen `date` en los días)."""
    if not value:
        return None
    from datetime import date as _date_cls, datetime as _dt_cls
    if isinstance(value, _dt_cls):
        return value.date()
    if isinstance(value, _date_cls):
        return value
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return _dt_cls.fromisoformat(raw.replace("Z", "+00:00")).date()
    except (TypeError, ValueError):
        pass
    try:
        return _dt_cls.strptime(raw[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def active_trip_window_days(
    plan_data: dict,
    window_len: int = TRIP_WINDOW_DAYS,
    *,
    ignore_knob: bool = False,
) -> list | None:
    """[P1-TRIP-WINDOWED-PERISHABLES · 2026-08-02] Los `window_len` días del viaje
    ACTIVO, o `None` si ventanear es un no-op.

    `ignore_knob=True` [ronda 1]: salta la consulta al knob. Lo usa ÚNICAMENTE el espejo
    del guard, que no decide si ventanear —eso ya lo decidió quien construyó la lista— sino
    que RE-DERIVA la ventana que esa lista declara vía su sello `trip_window_days`. Con el
    knob apagado y listas selladas vivas en DB, respetar el knob aquí devolvía `None`, el
    espejo se volvía identidad y el guard fabricaba la divergencia severa que el espejo
    existe para evitar. Los constructores de listas NUNCA pasan este flag.

    Devuelve `None` (y el caller conserva el comportamiento de siempre) cuando:
      - el knob está apagado (salvo `ignore_knob`),
      - el plan no tiene días,
      - el plan cabe entero en la ventana (`len(days) <= window_len`) — el caso de
        producción del viaje 1, donde solo hay 2-3 días materializados y "la ventana"
        y "el plan completo" son lo MISMO.

    Anclaje: `grocery_start_date` (el campo que el shift reescribe a hoy siguiendo a
    `days[0]`) → se seleccionan los días cuya `date` cae en `[ancla, ancla+window_len)`.
    Si el plan no trae fechas parseables (legacy) o el filtro sale vacío, cae a
    `days[:window_len]` — que tras el shift ES el viaje activo, porque el shift poda
    los días consumidos y deja `days[0] = hoy`.
    """
    if not ignore_knob and not _trip_windowed_perishables_enabled():
        return None
    if not isinstance(plan_data, dict):
        return None
    days = plan_data.get("days")
    if not isinstance(days, list) or not days:
        return None
    try:
        win = max(1, int(window_len))
    except (TypeError, ValueError):
        win = TRIP_WINDOW_DAYS
    if len(days) <= win:
        return None

    anchor = _parse_plan_day_date(plan_data.get("grocery_start_date"))
    if anchor is None and isinstance(days[0], dict):
        anchor = _parse_plan_day_date(days[0].get("date"))
    if anchor is not None:
        from datetime import timedelta as _td_win
        limit = anchor + _td_win(days=win)
        selected = []
        for d in days:
            if not isinstance(d, dict):
                continue
            day_date = _parse_plan_day_date(d.get("date"))
            if day_date is not None and anchor <= day_date < limit:
                selected.append(d)
        if selected and len(selected) < len(days):
            return selected
    return [d for d in days[:win] if isinstance(d, dict)] or None


def _aggregated_trip_window_len(aggregated_list) -> int | None:
    """[P1-TRIP-WINDOWED-PERISHABLES · 2026-08-02] Lee el sello `trip_window_days` que
    `get_shopping_list_delta` estampa en cada ítem de una lista ventaneada.

    Es lo que permite que el espejo del guard sea AUTO-SINCRONIZADO: el guard no
    necesita saber qué superficie construyó la lista (assemble, rebuild T2, recalc,
    cron), solo si la lista que tiene delante es un promedio del plan o el viaje activo.
    Sin ese sello el guard mirroreaba a ciegas y fabricaba divergencias falsas masivas
    contra cualquier lista producida por una superficie sin ventanear."""
    for item in aggregated_list or []:
        if not isinstance(item, dict):
            continue
        raw = item.get("trip_window_days")
        if not raw:
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _protein_yield_seal_applied(aggregated_list) -> bool:
    """[P2-PROTEIN-YIELD-CANONICAL · 2026-08-03 · ronda 1] Lee el sello
    `protein_yield_applied` que `aggregate_and_deduct_shopping_list` estampa en cada ítem
    cuando construyó la lista con la regla #2 (proteínas cocidas → 1.35× crudo) activa.

    Gemela exacta de `_aggregated_trip_window_len`/`trip_window_days` (Task 6,
    P1-TRIP-WINDOWED-PERISHABLES): el guard debe espejar cómo se CONSTRUYÓ la lista que
    tiene delante, NUNCA el knob `MEALFIT_PROTEIN_YIELD_ON_CANONICAL` vigente en el momento
    de re-evaluarla. El knob gobierna la CONSTRUCCIÓN de listas nuevas (get_shopping_list_
    delta lo lee para decidir si aplicar yield); leerlo también aquí, en la INTERPRETACIÓN
    de una lista ya construida, fabrica divergencias fantasma en dos direcciones:
      - A/B: plan construido con knob OFF, re-evaluado (cron diario, rebuild) con knob ON
        → el guard ve el lado esperado subir 1.35× mientras el comprado se queda igual.
      - Rollback: knob ON→OFF a mitad de camino → listas ya sembradas con yield real
        dejan de ser reconocidas por el guard.
    Medido por el revisor: lista real de 1.435 g construida con knob OFF, evaluada con
    knob ON → 25,9% de divergencia + `magnitude=True`. Con el sello, cero divergencia en
    ambas direcciones (test `test_p2_protein_yield_canonical.py::TestGuardSealComposition`).
    """
    for item in aggregated_list or []:
        if isinstance(item, dict) and item.get("protein_yield_applied") is True:
            return True
    return False


def _pantry_deduction_seal(aggregated_list) -> bool | None:
    """[P2-GUARD-UNDERSUPPLY-CANONICAL · 2026-08-03] Lee el sello `pantry_deduction_applied`
    que `aggregate_and_deduct_shopping_list` estampa en cada ítem: ¿esta lista se construyó
    restando nevera/consumidos, o es canónica?

    Tercera del linaje `trip_window_days` (P1-TRIP-WINDOWED-PERISHABLES) /
    `protein_yield_applied` (P2-PROTEIN-YIELD-CANONICAL): el guard debe espejar cómo se
    CONSTRUYÓ la lista que tiene delante, no adivinar por la superficie que lo invoca. Aquí
    la consecuencia de adivinar mal es el agujero que este P-fix cierra — dar por hecho que
    hubo deducción convierte todo sub-suministro real en `pantry_overdeduct` exento.

    TRI-ESTADO a propósito, y por eso el sello se estampa también cuando vale `False`:
      - `True`  → hubo deducción efectiva (>0) de nevera/consumidos.
      - `False` → lista CANÓNICA declarada: no se restó nada.
      - `None`  → la lista no lleva sello (persistida antes de este P-fix, o construida por
                  una superficie que no pasa por el aggregator). "No sé" NO es "no dedujo":
                  el caller cae al default conservador `True` y conserva el comportamiento
                  previo. Colapsar ausencia con `False` haría que el cron diario empezara a
                  marcar severas las listas viejas CON deducción legítima.

    Matiz conocido: «deducción efectiva» se mide por CANTIDAD (`qty > 0`), así que una Nevera
    de puros condimentos «al gusto» (qty 0) sella `False` aunque `P2-SEASONING-RESTOCK-CLEAR`
    SÍ haya usado esos nombres para dropear ítems de la lista por presencia. Inocuo hoy y a
    propósito: lo que el sello alimenta es la pregunta «¿pudo el inventario RESTAR gramos de
    este alimento?», y un drop por presencia no resta gramos de ningún otro — el alimento
    dropeado desaparece de la lista (no queda con cantidad corta que malinterpretar), y los
    demás quedan intactos. Si algún día el keep/drop por presencia empezara a recortar
    cantidades, este sello tendría que mirar también ese camino.
    """
    _seen_false = False
    for item in aggregated_list or []:
        if not isinstance(item, dict):
            continue
        raw = item.get("pantry_deduction_applied")
        if raw is True:
            return True
        if raw is False:
            _seen_false = True
    return False if _seen_false else None


def _merge_trip_windowed_result(full_res, window_res, *, window_len: int):
    """[P1-TRIP-WINDOWED-PERISHABLES · 2026-08-02] Perecederos de la ventana + estables
    del periodo, conservando la forma del contenedor (lista estructurada o dict por
    categoría). Estampa `trip_window_days` en cada ítem del resultado.

    Reglas (espejo exacto de `_build_hybrid_shopping_list`, mismo clasificador):
      - estable  → se toma del agregado del PERIODO (se compra una vez para el ciclo).
      - perecedero presente en la ventana → cantidades de la VENTANA, categoría del
        periodo (para no mover el ítem de sección del PDF).
      - perecedero AUSENTE de la ventana → fuera de este viaje (es el pescado de la
        semana 3: comprarlo hoy es tirarlo).
      - perecedero SOLO en la ventana → entra (la deducción de inventario puede
        borrarlo del agregado del periodo, que pide menos por día, y no del de la
        ventana, que pide el consumo real de estos 7 días).
    """
    from constants import strip_accents

    def _flatten(res):
        if isinstance(res, dict):
            out = []
            for cat, lst in res.items():
                if not isinstance(lst, list):
                    return None
                for it in lst:
                    if not isinstance(it, dict):
                        return None
                    out.append((cat, it))
            return out
        if isinstance(res, list):
            out = []
            for it in res:
                if not isinstance(it, dict):
                    return None
                out.append((None, it))
            return out
        return None

    full_items = _flatten(full_res)
    window_items = _flatten(window_res)
    if full_items is None or window_items is None:
        logging.warning(
            "[P1-TRIP-WINDOWED-PERISHABLES] resultado no estructurado (%s/%s): se "
            "conserva el agregado del periodo sin ventanear.",
            type(full_res).__name__, type(window_res).__name__,
        )
        return full_res

    def _key(item):
        return strip_accents(str(item.get("name", "")).lower().strip())

    def _is_perishable(item):
        return _classify_perishability(item.get("name", "") or "", item) == "perishable"

    window_by_key = {}
    for _cat, item in window_items:
        window_by_key.setdefault(_key(item), item)
    full_keys = {_key(item) for _cat, item in full_items}

    merged = []
    dropped = []
    windowed = 0
    for cat, item in full_items:
        if not _is_perishable(item):
            merged.append((cat, dict(item)))
            continue
        win_item = window_by_key.get(_key(item))
        if win_item is None:
            dropped.append(str(item.get("name") or "?"))
            continue
        out_item = dict(win_item)
        # La categoría manda la sección del PDF: la del periodo es la que el resto de
        # la lista ya usa para este alimento.
        if item.get("category") is not None:
            out_item["category"] = item.get("category")
        if item.get("display_category") is not None:
            out_item["display_category"] = item.get("display_category")
        merged.append((cat, out_item))
        windowed += 1
    for cat, item in window_items:
        if _key(item) in full_keys:
            continue
        if _is_perishable(item):
            merged.append((cat, dict(item)))
            windowed += 1

    for _cat, item in merged:
        item["trip_window_days"] = int(window_len)

    logging.info(
        "🗓️ [P1-TRIP-WINDOWED-PERISHABLES] ventana=%dd → %d perecedero(s) del viaje, "
        "%d fuera de este viaje (%s)",
        int(window_len), windowed, len(dropped), ", ".join(sorted(dropped)[:12]) or "-",
    )

    # [ronda 1 · 2026-08-02] SIN re-ordenar. El agregador ya ordenó por `display_string`,
    # que empieza por la CANTIDAD ("1 ½ lbs de Arroz"), así que re-ordenar con las
    # cantidades nuevas de la ventana reorganizaba la lista visible por texto de cantidad
    # —ni por nombre ni por categoría— sustituyendo el orden nativo del agregador.
    # `merged` conserva el orden de `full_res`; los perecederos que solo existen en la
    # ventana van al final, en el orden en que la ventana los produjo.
    if isinstance(full_res, dict):
        out_dict: dict = {}
        for cat, item in merged:
            out_dict.setdefault(cat, []).append(item)
        return out_dict
    return [item for _cat, item in merged]


def parse_fraction(val: str) -> float:
    val = val.strip()
    try:
        if ' ' in val:
            parts = val.split(' ')
            if '/' in parts[1]:
                num, den = parts[1].split('/')
                return float(parts[0]) + float(num)/float(den)
        if '/' in val:
            num, den = val.split('/')
            return float(num)/float(den)
        return float(val)
    except Exception:
        return 0.0

# [P2-GUARD-PERF-REGEXCACHE · 2026-07-10] Los ~140 stop-words de normalize_name se compilaban como
# ~140 regexes DINÁMICOS POR LLAMADA (loop `re.sub(r'\b'+s+r'\b')`) — con 160 llamadas por corrida
# del coherence guard eso producía ~20k compilaciones re.* (el cache LRU de `re` tiene 512 slots →
# thrash total). Perfil cProfile en VPS (plan real 12 meals): 4.2s de 4.5s dentro de normalize_name;
# el guard reportaba 9.3-9.7s en prod (umbral 5s). UNA alternación precompilada a nivel módulo,
# ordenada por longitud DESC para que las frases multi-palabra ('bajo en grasa', 'hecha puré')
# matcheen antes que sus sub-tokens. Semántica equivalente al loop secuencial (sub global de cada
# stop; validado por las suites de coherencia/shopping). tooltip-anchor: P2-GUARD-PERF-REGEXCACHE
# [P1-COUNTRY-SYSTEM-F2 · Task 8 · 2026-08-17] 'en láminas'/'en lámina' faltaban del hermano
# de corte ('en rodajas'/'en trozos'/'en lonjas' ya cubrían la misma familia de preparación).
# Drop real medido: "rábanos en láminas" (4/30d en rd_drops.json) — "rábano" YA existe en el
# catálogo con precio, pero "rabanos en laminas" no matchea NINGÚN tier léxico/CONTAINS (el
# sufijo de plural rompe el boundary de la palabra "rabano") y la FUZZY del INTENTO 5 mide
# contra el string COMPLETO (ratio 0.50 << 0.87). Con este stop, `clean_n` colapsa a "rabanos"
# (bare, plural) — que SÍ resuelve vía FUZZY (ratio 0.923) contra "rabano", cerrando el drop sin
# tocar el catálogo. AMBAS formas (con tilde 'en láminas' Y sin tilde 'en laminas'): `n` en este
# punto de `normalize_name` está lowercased pero NO accent-stripped (mismo motivo por el que
# 'pequeño'/'puré' arriba llevan sus tildes) — con SOLO la forma acentuada, un input SIN tilde
# ("rabanos en laminas", plausible si el LLM/usuario omite diacríticos) no la habría matcheado y
# el drop seguiría vivo para esa variante (medido en vivo durante la verificación de esta task:
# con solo 'en láminas', "rabanos en laminas" quedaba sin resolver mientras "rábanos en láminas"
# sí). El alias explícito 'rabanos' (síncrono, ver synonyms_rd_topup_2026_08_17.json) YA cierra
# esta variante de forma robusta independientemente de este stop — este stop de todos modos gana
# la forma sin tilde para CUALQUIER OTRO alimento sin alias dedicado (ver
# test_en_laminas_es_stop_generico_no_especifico_de_rabano, "Remolacha en láminas").
_NORMALIZE_STOPS = ['cortado', 'cortada', 'cortados', 'cortadas', 'picado', 'picada', 'picados', 'picadas', 'picadito', 'picadita', 'picaditos', 'picaditas', 'pelado', 'pelada', 'pelados', 'peladas', 'hervido', 'hervida', 'hervidos', 'hervidas', 'cocido', 'cocida', 'cocidos', 'cocidas', 'asado', 'asada', 'asados', 'asadas', 'crudo', 'cruda', 'crudos', 'crudas', 'horneado', 'horneada', 'horneados', 'horneadas', 'desmenuzado', 'desmenuzada', 'desmenuzados', 'desmenuzadas', 'rallado', 'rallada', 'rallados', 'ralladas', 'guisado', 'guisada', 'guisados', 'guisadas', 'frito', 'frita', 'fritos', 'fritas', 'majado', 'majada', 'majados', 'majadas', 'triturado', 'triturada', 'triturados', 'trituradas', 'hecha puré', 'hecho puré', 'puré', 'en julianas', 'en tiras', 'en cubos', 'en hojuelas', 'en dados', 'en aros', 'en trozos', 'en rodajas', 'en porciones', 'en lonjas', 'en lonja', 'en láminas', 'en lámina', 'en laminas', 'en lamina', 'finamente', 'muy', 'pequeño', 'pequeña', 'pequeños', 'pequeñas', 'grande', 'grandes', 'mediano', 'mediana', 'medianos', 'medianas', 'maduro', 'madura', 'maduros', 'maduras', 'fresco', 'fresca', 'frescos', 'frescas', 'firme', 'firmes', 'entero', 'entera', 'enteros', 'enteras', 'fina', 'finas', 'gruesa', 'gruesas', 'magro', 'magra', 'magros', 'magras', 'natural', 'naturales', 'bajo en grasa', 'bajas en grasa', 'bajos en grasa', 'bajo en sodio', 'bajas en sodio', 'bajos en sodio', 'descremado', 'descremada', 'descremados', 'descremadas', 'sin sal', 'con sal', 'sin piel', 'sin hueso', 'para rebozar', 'al gusto', 'pizca de', 'rodajas de', 'de la despensa', 'ralladura y jugo de 1/2', 'la', 'el', 'los', 'las']
_NORMALIZE_STOPS_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(s) for s in sorted(_NORMALIZE_STOPS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE,
)
# Prefijos líderes (antes también re-compilados por llamada): precompilados una vez.
_NORMALIZE_PAREN_RE = re.compile(r'\(.*?\)')
_NORMALIZE_CONTAINER_PREFIX_RE = re.compile(
    r'^(cda|cdta|cdita|cucharada|cucharadita|taza|vaso|pizca|chorrito|puñado|atado|manojo|scoop|lonja|loncha|paquete|paquetico|funda|lata|sobre|sobrecito|chin|toque)(s)?\s*(de\s+|del\s+)?',
    re.IGNORECASE)
_NORMALIZE_ANATOMY_PREFIX_RE = re.compile(
    r'^(pechuga|filete|muslo|trozo|chuleta|pieza|corte|ración|racion|porción|porcion|filetico|medallón|medallones|carne)(s)?\s+(de\s+|del\s+)',
    re.IGNORECASE)
_NORMALIZE_DE_PREFIX_RE = re.compile(r'^(de\s+|del\s+)', re.IGNORECASE)



# [P1-MODIFIER-ONLY-ALIAS · 2026-07-26] Palabras que describen un ESTADO o CORTE, nunca un
# alimento. Un alias del catálogo que sea solo una de estas no puede resolver un alimento
# dentro de un texto que ya nombra otro (ver el bloque del INTENTO 2). Sin acentos: el caller
# compara contra `strip_accents`.
_MODIFIER_ONLY_ALIASES = frozenset({
    "maduro", "madura", "maduros", "maduras",
    "verde", "verdes", "fresco", "fresca", "frescos", "frescas",
    "crudo", "cruda", "crudos", "crudas", "cocido", "cocida", "cocidos", "cocidas",
    "molido", "molida", "molidos", "molidas", "rallado", "rallada",
    "picado", "picada", "entero", "entera", "enteros", "enteras",
    "seco", "seca", "secos", "secas", "integral", "integrales",
    "grande", "mediano", "pequeno", "magro", "magra",
})


# [P1-COHERENCE-ALIAS-INDEX · 2026-08-14] El índice de alias, construido UNA vez
# por catálogo en lugar de una vez por llamada.
#
# El coherence guard rebasó su umbral de 5s 17 veces en 7 días (hasta 11,5s), una
# de ellas dentro de un `/recalculate-shopping-list` SÍNCRONO. Perfilado contra el
# plan real cb361844 (26 días, 104 comidas): 17,4s de los 19,4s del guard vivían en
# `expected_sum_from_recipes`, y de esos 14,8s eran **compilar regex** — 481.240
# llamadas a `re._compile`.
#
# La causa: los INTENTOS 2 y 4 recorrían los ~700 alias del catálogo construyendo
# `r'\b' + re.escape(alias) + r'\b'` en caliente. La caché interna de `re` guarda
# 512 patrones; con más alias que huecos se vacía sola y cada llamada recompila casi
# todo. Y encima `all_aliases` se reconstruía y se REORDENABA (700 elementos) en cada
# una de las 973 llamadas. Nada de ese trabajo depende del texto a normalizar: depende
# solo del catálogo.
#
# Invalidación por IDENTIDAD (`is`), no por TTL. `get_master_ingredients()` devuelve
# el mismo objeto mientras su caché viva, así que la identidad detecta la recarga sin
# ventana ciega — y un test que parchea el catálogo no hereda el índice del test
# anterior, que es justo la fuga que un TTL habría creado aquí.
_NORMALIZE_ALIAS_INDEX: dict | None = None

# Umbral del tier fuzzy (INTENTO 5 de `normalize_name`). Vive en UNA constante
# porque la poda por longitud lo lee también: si la aceptación bajara a 0.80 y la
# poda se quedara en 0.87, la poda empezaría a descartar matches VÁLIDOS sin que
# nadie lo note. Un umbral duplicado es un umbral que ya drifteó.
_FUZZY_MATCH_THRESHOLD = 0.87

# [P2-CHICHARO-CHICHARRON · 2026-08-21] Pares (consulta, destino) que el fuzzy JAMÁS puede unir
# porque son alimentos distintos, no variantes del mismo. Una entrada, y la brevedad está MEDIDA:
# de 57 términos regionales barridos, 43 resolvieron, 9 cayeron en Proteínas y sólo éste era falso.
# Añadir aquí exige la misma evidencia — un par que el barrido de `test_p2_chicharo_chicharron.py`
# demuestre; si no, el arreglo correcto casi siempre es dar de alta la fila que falta.
_FUZZY_COLISIONES_PROHIBIDAS = (
    (r"\bchicharos?\b", r"chicharr"),   # guisante (MX) ≠ corteza de cerdo
)


def _construir_indice_alias(master_list: list) -> tuple[list, list]:
    """`(all_aliases, contains_compilados)` para un catálogo dado.

    `all_aliases` conserva el orden descendente por longitud del código original:
    es load-bearing, no cosmético — con orden arbitrario 'maduro' (6) le gana a
    'mango' (5) y un desayuno de mango mete PLÁTANO en la lista (P1-MODIFIER-ONLY-ALIAS).
    """
    from constants import strip_accents

    all_aliases = []
    for master in master_list:
        master_name = master["name"]
        all_aliases.append((strip_accents(master_name.strip().lower()), master_name))
        for alias in (master.get("aliases") or []):
            all_aliases.append((strip_accents(alias.strip().lower()), master_name))
    # [P1-CATALOG-ORDER-DETERMINISTIC · 2026-08-19] Desempate ALFABÉTICO tras la longitud:
    # el sort estable heredaba el orden de FILAS en los empates de longitud ('arroz'=5 vs
    # 'pollo'=5) — comportamiento indefinido que el fill masivo del gloss del catálogo re-barajó,
    # flipeando resoluciones reales del corpus DO. Con (len desc, alias asc) el índice es
    # idéntico sea cual sea el orden físico del SELECT.
    all_aliases.sort(key=lambda x: (-len(x[0]), x[0]))

    contains = [
        (re.compile(r'\b' + re.escape(alias_stripped) + r'\b', re.IGNORECASE), master_name, alias_stripped)
        for (alias_stripped, master_name) in all_aliases
        if alias_stripped and alias_stripped not in _MODIFIER_ONLY_ALIASES
    ]
    return all_aliases, contains


def _get_normalize_alias_index(master_list: list) -> tuple[list, list]:
    """Índice cacheado; se reconstruye solo si el catálogo es otro objeto (o cambió
    de tamaño, por si alguien lo muta en sitio)."""
    global _NORMALIZE_ALIAS_INDEX
    cache = _NORMALIZE_ALIAS_INDEX
    if (
        cache is not None
        and cache["src"] is master_list
        and cache["len"] == len(master_list)
    ):
        return cache["all_aliases"], cache["contains"]
    all_aliases, contains = _construir_indice_alias(master_list)
    # Se guarda la referencia al catálogo, no solo su id(): mantenerlo vivo es lo
    # que impide que un id reciclado por el GC valide un índice ajeno.
    _NORMALIZE_ALIAS_INDEX = {
        "src": master_list,
        "len": len(master_list),
        "all_aliases": all_aliases,
        "contains": contains,
    }
    return all_aliases, contains


def _best_contains_match(text: str, patterns) -> "str | None":
    """[P1-CATALOG-ORDER-DETERMINISTIC · 2026-08-19] Mejor match CONTAINS por
    (posición del match en el string, longitud del alias desc, alias asc). Ver el
    comentario del INTENTO 2 en normalize_name para el porqué semántico.
    tooltip-anchor: P1-CATALOG-ORDER-DETERMINISTIC"""
    # Orden del desempate: LONGITUD primero (la semántica histórica del índice — 'pernil'
    # le gana a 'cerdo' en «cerdo para pernil», retarget F2 documentado), POSICIÓN en el
    # string después (en empates de longitud la identidad del plato encabeza: 'pollo' a
    # posición 0 le gana a 'arroz' en «Pollo horneado ... con arroz»), alfabético al final
    # (determinismo total: el heap ya no decide nada).
    best_key = None
    best_name = None
    for _pat, master_name, alias_stripped in patterns:
        m = _pat.search(text)
        if not m:
            continue
        key = (-len(alias_stripped), m.start(), alias_stripped, master_name)
        if best_key is None or key < best_key:
            best_key = key
            best_name = master_name
    return best_name


def normalize_name(orig_name: str) -> str:
    n = str(orig_name).lower().strip()
    n = _NORMALIZE_PAREN_RE.sub('', n).strip()
    # Limpieza de prefijos contenedores o medidas informales
    n = _NORMALIZE_CONTAINER_PREFIX_RE.sub('', n)
    # Nueva mejora: Limpieza estricta de pseudo-unidades anatómicas LATINAS SOLO si están seguidas de 'de'
    n = _NORMALIZE_ANATOMY_PREFIX_RE.sub('', n)
    n = _NORMALIZE_DE_PREFIX_RE.sub('', n)

    # [P2-GUARD-PERF-REGEXCACHE] una sola pasada con la alternación precompilada (ver arriba)
    clean_n = _NORMALIZE_STOPS_RE.sub('', n)
        
    # Limpiar conjunciones o preposiciones que quedan colgadas al quitar los stops al inicio o al final
    clean_n = re.sub(r'^\s*(y|e|o|en|con|de|del|para)\b', '', clean_n, flags=re.IGNORECASE)
    clean_n = re.sub(r'\b(y|e|o|en|con|de|del|para)\s*$', '', clean_n, flags=re.IGNORECASE)
    clean_n = re.sub(r'\s+', ' ', clean_n).replace(',', '').strip()
    
    master_list = get_master_ingredients()
    from constants import strip_accents

    # [P3-PROTEIN-CAP-2] Guard pre-alias para distinguir productos de pavo:
    # el alias lookup downstream puede mapear "pechuga de pavo" / "filete de
    # pavo" a "Jamón de pavo" cuando master_list tiene esas frases listadas
    # como alias del producto procesado (caso real en environments con master
    # poblado desde constants.PROTEIN_SYNONYMS). Sin este guard, fresh y
    # molido se conflatarían con deli procesado, costando al usuario
    # ~$70 RD$/lb extra y nutrición peor (sodio 4× mayor en deli).
    #
    # Reglas, en orden de precedencia (alineadas con la canonicalización
    # del aggregator):
    #   1. fresh marker explícito + pechuga/filete → Pechuga de pavo
    #   2. processed marker explícito (jamón de pavo, lonjas, procesado) →
    #      Jamón de pavo
    #   3. pavo molido / carne de pavo → Pavo molido
    #   4. pechuga de pavo / filete de pavo (sin marker procesado) →
    #      Pechuga de pavo (default seguro fresh)
    #   5. else: cae al alias lookup (master decide)
    _opl = str(orig_name).lower()
    if re.search(r'\bpavo\b', _opl):
        _has_fresh = bool(re.search(r'\bfresc[oa]s?\b|\bfresh\b', _opl))
        _has_processed = bool(re.search(
            r'jam[oó]n\s+de\s+pavo|pavo\s+en\s+lonjas?|lonjas?\s+de\s+pavo|'
            r'pavo\s+procesado|pavo\s+en\s+rebanadas?',
            _opl
        ))
        if _has_fresh and re.search(r'\b(pechuga|filete)\s+de\s+pavo\b', _opl):
            return 'Pechuga de pavo'
        if _has_processed:
            return 'Jamón de pavo'
        if re.search(r'\bpavo\s+molido\b|\bcarne\s+de\s+pavo\b', _opl):
            return 'Pavo molido'
        if re.search(r'\b(pechuga|filete)\s+de\s+pavo\b', _opl):
            return 'Pechuga de pavo'
        # Fallback: "pavo" sin más descriptores → canonical "Pavo" (no
        # auto-canonicalizar a Jamón de pavo via alias lookup, que es la
        # trampa que justamente queremos evitar). Default seguro: tratar
        # como pavo genérico fresh.
        return 'Pavo'

    # [P3-YOGURT-CONSOLIDATE · 2026-06-22] Todo yogurt (griego/natural/entero/
    # sin azúcar/0%/light) resuelve a UN solo ítem de compra: "Yogurt". El LLM
    # emite variantes ("yogurt griego entero", "yogurt griego sin azúcar") que
    # antes resolvían a master rows distintos → 2+ líneas duplicadas en la lista
    # de compras (pedido del owner: "que diga solo yogurt, 1 solo"). Guard
    # temprano determinista (mismo patrón que el guard de pavo arriba) — evita la
    # colisión ambigua de alias same-length entre las filas variantes y es
    # simétrico con canonicalize_lacteo (coherencia → 'Yogur').
    #
    # SOLO afecta la AGREGACIÓN/DISPLAY de la lista de compras: el master row
    # "Yogurt" provee precio/envase (pote). La distinción nutricional entero vs
    # nonfat (P2-3: fat 4g vs 0.37g) se PRESERVA porque nutrition_db resuelve las
    # variantes por sus aliases en Tier-1/2 ANTES de delegar a normalize_name
    # (Tier-3). Tooltip-anchor: P3-YOGURT-CONSOLIDATE.
    if re.search(r'\byogur(t)?\b', _opl):
        return 'Yogurt'

    # [P1-COUNTRY-SYSTEM-F2 · ola final (review de fase) · 2026-08-18 · C3.1] "Chicharrón" (fila CO,
    # Task 6) hace que CUALQUIER frase con la palabra "chicharrón" resuelva a esa fila vía CONTAINS
    # (INTENTO 2/4) — no por el alias explícito 'chicharron' (removido en esta ola), sino porque el
    # NOMBRE CANÓNICO de toda fila se añade a `all_aliases` incondicionalmente (`_construir_indice_alias`
    # arriba). Verificado en vivo: quitar el alias NO cambia nada para "Chicharrón de pollo" — sigue
    # resolviendo a 'Chicharrón' (cerdo) mientras la fila exista con ese nombre. Pre-fase (fila
    # inexistente) 'chicharrón de cerdo' resolvía 'Cerdo' genérico y 'chicharrón de pollo' resolvía
    # 'Pechuga de pollo' (ambos por substring). El review de T6 mejoró 'de cerdo' A PROPÓSITO (chicharrón
    # real: kcal 544/fat 31,3g vs Cerdo genérico kcal 169,6/fat 9,47g, >200% de diferencia — ver
    # `_provenance` en new_foods_mx_co_2026_08_17.json) pero NUNCA evaluó 'de pollo' — que colisionaba en
    # silencio con la nutrición del CERDO (chicharrón de pollo real es ~muslo/pechuga frita, macro
    # totalmente distinto). Este guard restaura el pre-fase SOLO para 'de pollo', preservando intacta la
    # mejora aceptada de 'de cerdo'/bare (que sigue cayendo a los tiers de abajo → 'Chicharrón').
    # [micro-fix ola final · 2026-08-18] `(?:es)?` — el plural "Chicharrones de pollo" se escapaba: el
    # alias 'chicharrones' (plural) SOBREVIVE en la fila tras remover el bare 'chicharron' (solo el
    # singular se quitó), así que el CONTAINS de abajo lo seguía matcheando y el guard, con `\b...n\b`
    # sin sufijo, nunca disparaba para la forma plural. Mismo patrón `(?:s|es)?` que
    # `_scan_allergen_violations` ya usa para el plural español (fresa→fresas, pan→panes) y que
    # `_COUNTRY_CATALOG_UNPRICED_TOKENS`/fix-round 1 de esta misma ola asumía en otros tokens.
    if re.search(r'\bchicharr[oó]n(?:es)?\b', _opl) and re.search(r'\bpollo\b', _opl):
        return 'Pechuga de pollo'

    # [P1-PREP-COLLAPSE-GUARD · 2026-07-01] Preparaciones "harina de X"/"tortilla de maíz"/"crema de coco"
    # son PRODUCTOS DISTINTOS del alimento base (lección P1-NUT-BUTTER-DISTINCT generalizada). Sin este guard
    # temprano, el alias 'harina' (Harina de trigo) ganaba en el Tier-2 → "harina de avena" resolvía a TRIGO
    # (gluten para un celíaco) y "harina de plátano" a Plátano fresco (macros ~3× sub-contados). Con
    # equivalente real → canoniza; sin fila propia → pass-through (no matchea master → verified-only lo
    # dropea; coherencia simétrica porque ambos lados pasan por aquí). tooltip-anchor: P1-PREP-COLLAPSE-GUARD
    _prep_handled, _prep_canon = resolve_preparation_distinct(orig_name)
    if _prep_handled:
        if _prep_canon:
            return _prep_canon
        _prep_disp = re.sub(r'\(.*?\)', '', str(orig_name)).strip()
        if _prep_disp:
            return _prep_disp[0].upper() + _prep_disp[1:]
        return n

    n_stripped = strip_accents(n)
    clean_n_stripped = strip_accents(clean_n)
    
    # Recolectar todos los aliases + nombres canónicos para búsqueda,
    # ordenados por longitud (más largos primero) para evitar que 
    # 'platano' se trague 'platano maduro' o 'queso' se trague 'queso cottage'
    # [P1-COHERENCE-ALIAS-INDEX · 2026-08-14] Construido una vez por catálogo (ver
    # el helper): antes se rearmaba y reordenaba en CADA llamada, 973 veces por guard.
    all_aliases, _aliases_for_contains = _get_normalize_alias_index(master_list)

    # [P1-MODIFIER-ONLY-ALIAS · 2026-07-26] Un alias que es SOLO un modificador no puede
    # resolver un alimento por su cuenta dentro de un texto más grande.
    #
    # `Plátano maduro` tiene en el catálogo el alias **'maduro'** (a secas). Los tiers 2 y 4
    # buscan cada alias como palabra completa DENTRO del texto, recorriéndolos por longitud
    # DESCENDENTE — así que 'maduro' (6) se evalúa antes que 'mango' (5) y gana:
    #
    #     "½ mango maduro"  ->  Plátano maduro     (plan vivo 01d63a5b, desayuno día 1)
    #     "1 kiwi maduro"   ->  Plátano maduro
    #
    # No es que faltara el mango en la lista de compras: es que la lista traía PLÁTANO en su
    # lugar. El usuario compra plátanos para un desayuno de mango. Y como el aggregator y el
    # coherence guard usan este MISMO parser, ambos lados coinciden en el error y la guarda no
    # puede verlo — coherentemente equivocados.
    #
    # Afecta a todo alimento masculino cuyo nombre sea más corto que el modificador. Las
    # femeninas se salvan por casualidad ("pera madura" ≠ 'maduro'), lo que no es una defensa.
    #
    # Los tiers de match EXACTO (1 y 3) los conservan: si alguien escribe literalmente
    # "maduro" a secas, resolver a plátano maduro es defendible. Lo que se prohíbe es que un
    # modificador secuestre un texto que ya nombra otro alimento.
    # El filtro vive dentro del índice cacheado (P1-COHERENCE-ALIAS-INDEX), que además
    # trae ya compilado el patrón de cada alias superviviente.

    # ── INTENTO 1: Match Exacto sobre el texto RAW (sin mutilar por stops) ──
    # Esto es CRÍTICO porque los stops eliminan palabras como 'natural', 'descremado',
    # 'bajo en grasa' que son parte de aliases legítimos como 'yogurt griego natural'.
    for alias_stripped, master_name in all_aliases:
        if n_stripped == alias_stripped:
            return master_name

    # ── INTENTO 2: Regex sobre el texto RAW (sin mutilar) ──
    # Buscar "queso mozzarella bajo en grasa" dentro de "queso mozzarella bajo en grasa rallado"
    # [P1-MODIFIER-ONLY-ALIAS] lista filtrada: un modificador suelto no secuestra el texto.
    # [P1-CATALOG-ORDER-DETERMINISTIC · 2026-08-19] Best-match por POSICIÓN en el string
    # (luego longitud desc, luego alfabético) en vez de first-hit por orden del índice: en
    # un plato multi-alimento la identidad ENCABEZA («Pollo horneado al limón con arroz» es
    # pollo, no arroz; «Chillo al horno con... batata» es el pescado). El first-hit hacía
    # ganar al alias más largo y, en empates de longitud, al azar del heap. Los ~1400
    # patrones están precompilados (P1-COHERENCE-ALIAS-INDEX): el full-scan es sub-ms.
    _best = _best_contains_match(n_stripped, _aliases_for_contains)
    if _best is not None:
        return _best

    # ── INTENTO 3: Match Exacto sobre clean_n (texto limpio, fallback) ──
    for alias_stripped, master_name in all_aliases:
        if clean_n_stripped == alias_stripped:
            return master_name

    # ── INTENTO 4: Regex sobre clean_n (último recurso antes de fuzzy/semántica) ──
    # [P1-MODIFIER-ONLY-ALIAS] misma lista filtrada que el INTENTO 2.
    _best = _best_contains_match(clean_n_stripped, _aliases_for_contains)
    if _best is not None:
        return _best

    # ── INTENTO 5 [P4-UNIFIED-RESOLVER · 2026-06-14]: Fuzzy (difflib) ANTES de gastar un embedding.
    # Atrapa typos y variantes menores ("platanno"→"plátano", "yogur griego"→"yogurt griego") que los
    # tiers regex no cubren, sin costo de API. Conservador (ratio ≥ 0.87) para no introducir falsos
    # positivos; los casos semánticos reales (sinónimos no-léxicos) los sigue cubriendo el embedding. ──
    import difflib
    # Formas candidatas a comparar (los strippers de prefijo/stop-words a veces dejan el query corto o
    # le quitan contexto: "platanno maduro"→"platanno", "pechuga de poyo"→"poyo"). Comparamos contra el
    # crudo, el limpio Y el original (solo parens removidos) y tomamos el mejor ratio por alias.
    _orig_fuzz = strip_accents(re.sub(r'\(.*?\)', '', str(orig_name).lower()).strip())
    # [P2-MIXED-FRACTION-PARSE · 2026-07-06] Forma SIN prefijo de cantidad
    # ("1½ cebollas pequeñas" → "cebollas pequeñas"): callers que pasan display
    # strings crudos (coherence guard, inventario tipeado) no deben perder el
    # match léxico por el número — ni pagar el embedding de INTENTO 6 con un
    # query contaminado.
    _noqty = re.sub(r'^[\d\s.,/½¼¾⅓⅔⅕]+(?:de\s+)?', '', _orig_fuzz).strip()
    # [P1-QUALIFIER-STRIP-FUZZY · 2026-07-30] Forma SIN calificativo de cola: "nisperos sin semilla"
    # → "nisperos". Medido en prod (18 drops en 40 min): 'Nísperos sin semilla' NO resolvía mientras
    # 'Nísperos' → 'Níspero' resolvía de sobra — el calificativo tiraba el ratio fuzzy bajo 0.87 y
    # el alimento quedaba FUERA de la lista de compras ("un calificativo que no se compra por
    # separado no debería poder volver incomprable el alimento"). Solo entra al pool de formas
    # fuzzy: el umbral 0.87 sigue mandando, así que no introduce snaps nuevos por sí sola.
    # Solo calificativos NEGATIVOS ("sin semilla", "bajo en sodio", "libre de gluten"). "con X" se
    # queda fuera a propósito: "yogurt con fresas" nombra OTRO producto y recortarlo cambiaría el
    # alimento resuelto, no solo su presentación.
    _noqual = re.sub(r'\bsin\s+\w+.*$|\b(?:bajo|baja|libre)s?\s+(?:en|de)\s+\w+.*$',
                     '', _noqty).strip()
    _fuzz_forms = {f for f in (n_stripped, clean_n_stripped, _orig_fuzz, _noqty, _noqual)
                   if f and len(f) > 3}
    if _fuzz_forms:
        _fuzz_best, _fuzz_name = 0.0, None
        for alias_stripped, master_name in all_aliases:
            if not alias_stripped:
                continue
            # [P1-COHERENCE-ALIAS-INDEX · 2026-08-14] Poda por longitud ANTES de
            # gastar un difflib. No es una heurística: es una cota. Como
            # `ratio = 2·M/(la+lf)` y los caracteres casados `M` no pueden superar
            # `min(la, lf)`, el ratio nunca pasa de `2·min/(la+lf)`. Si esa cota ya
            # queda bajo el umbral, ese par NO puede ser un match — comparar es
            # trabajo tirado. Los pares que sí alcanzan el umbral tienen cota ≥ su
            # propio ratio, así que jamás se podan: el veredicto es idéntico, no
            # aproximado (test `test_la_poda_es_equivalente_no_aproximada`).
            # Medido: era el 45% del guard tras quitar la tormenta de regex.
            _la = len(alias_stripped)
            _r = 0.0
            for f in _fuzz_forms:
                _lf = len(f)
                if (2.0 * min(_la, _lf)) / (_la + _lf) < _FUZZY_MATCH_THRESHOLD:
                    continue
                _rr = difflib.SequenceMatcher(None, f, alias_stripped).ratio()
                if _rr > _r:
                    _r = _rr
            if _r > _fuzz_best:
                _fuzz_best, _fuzz_name = _r, master_name
        if _fuzz_best >= _FUZZY_MATCH_THRESHOLD and _fuzz_name:
            # [P2-CHICHARO-CHICHARRON · 2026-08-21] Un fuzzy alto puede cruzar de alimento.
            #
            # `chicharo` vs el alias `chicharron` da ratio 0,889 sobre un umbral de 0,87: pasa. Y
            # como el destino ES una fila real del catálogo, SOBREVIVE al filtro de verified-only —
            # no se cae de la lista, se COMPRA. Un mexicano que pide chícharos recibe corteza de
            # cerdo, y si además es vegetariano, musulmán o judío el plato es inaceptable por
            # razones que la nutrición no cubre. 18ª colisión de subcadena/fuzzy del proyecto.
            #
            # POR QUÉ UNA LISTA DE PARES Y NO UNA REGLA GENERAL: se barrieron 57 términos
            # regionales de ES/MX/CO/PR. 43 resolvieron, 9 cayeron en Proteínas y OCHO eran
            # correctos (gamba→Gambas, atún→Atún en agua, res→Carne de res…). El único falso
            # positivo era éste. Una regla de «cruce de categoría» habría exigido un clasificador
            # de «esto es carne» que no existe como SSOT — sería la cuarta tabla a mano, la lección
            # de P1-DIET-CANON-SSOT, para atrapar un caso. La defensa de CLASE es el barrido de
            # `test_p2_chicharo_chicharron.py`, que corre sobre el catálogo vivo y falla si un alta
            # futura crea otra colisión de esta forma.
            #
            # No cierra que el catálogo NO tenga fila de guisante fresco (la única de la familia es
            # `Guisantes secos`, 341 kcal, otro alimento). Tras el guard «chícharo» no resuelve a
            # nada: peor que lo ideal, mejor que cerdo. El alta con procedencia verificable es
            # curación de datos. tooltip-anchor: P2-CHICHARO-CHICHARRON
            _fz_q = strip_accents(str(orig_name).lower())
            _fz_r = strip_accents(str(_fuzz_name).lower())
            _colision = any(
                re.search(_q_rx, _fz_q) and re.search(_r_rx, _fz_r)
                for _q_rx, _r_rx in _FUZZY_COLISIONES_PROHIBIDAS
            )
            if _colision:
                logging.warning(
                    f"🛡 [P2-CHICHARO-CHICHARRON] fuzzy rechazado: '{orig_name}' -> "
                    f"'{_fuzz_name}' (ratio {_fuzz_best:.3f}) son alimentos DISTINTOS"
                )
            else:
                logging.info(f"🔤 [Fuzzy Match] '{orig_name}' -> '{_fuzz_name}' (ratio {_fuzz_best:.3f})")
                return _fuzz_name

    # Intento 6: Búsqueda de Similitud Semántica Vectorial (Cohere v4, Fallback Local)
    # Solo vale la pena gastar un request si la palabra no fue encontrada en absoluto y tiene suficiente longitud
    if len(n) > 3:
        cache = get_semantic_cache()
        if cache:
            try:
                # [P2-MIXED-FRACTION-PARSE] query sin prefijo de cantidad — un
                # "1½ " al frente solo mete ruido al embedding.
                _sem_q = _noqty if len(_noqty) > 3 else n
                # Calculamos el vector del texto no reconocido
                query_vector = _gemini_call_with_retry(
                    cache["embeddings_client"].embed_query, _sem_q,
                    _label=f"embed_query (semantic match: {_sem_q[:40]!r})",
                )
                best_score = -1.0
                best_match = None
                
                # Buscamos matemáticamente contra toda la tabla en milisegundos de RAM
                for i, master_vector in enumerate(cache["vectors"]):
                    score = cosine_similarity(query_vector, master_vector)
                    if score > best_score:
                        best_score = score
                        best_match = cache["master_list"][i]["name"]
                
                # Umbral de confianza estricto (0.70 o 70% de similitud)
                if best_score >= 0.70:
                    logging.info(f"🧠 [Semantic Search] Resuelto: '{orig_name}' -> '{best_match}' con score {best_score:.3f}")
                    return best_match
            except Exception as e:
                logging.error(f"Error en búsqueda semántica de '{orig_name}': {e}")

    if len(clean_n) > 0:
        return clean_n[0].upper() + clean_n[1:]
    return n

def _preprocess_nlp_quantities(s: str) -> str:
    s_lower = str(s).lower().strip()
    
    # Soporte nativo para fracciones Unicode al inicio
    fraction_map = {
        u"\u00BD": "1/2",  # ½
        u"\u00BC": "1/4",  # ¼
        u"\u00BE": "3/4",  # ¾
        u"\u2153": "1/3",  # ⅓
        u"\u2154": "2/3",  # ⅔
        u"\u2155": "1/5"   # ⅕
    }
    _frac_touched = False
    for k, v in fraction_map.items():
        if s_lower.startswith(k):
            s_lower = s_lower.replace(k, v + " ", 1)
            _frac_touched = True

    # [P2-MIXED-FRACTION-PARSE · 2026-07-06] Número MIXTO con fracción unicode
    # PEGADA ("1½ cebollas pequeñas"): el startswith de arriba solo cubre la
    # fracción SOLA ("½ taza"). Sin esto, el regex principal de _parse_quantity
    # no matchea → qty=0.0 nominal (la cebolla se SUB-CUENTA en la lista) y el
    # string ENTERO entra como nombre a normalize_name, cuyos tiers léxicos
    # fallan por el prefijo numérico → quemaba una llamada Cohere por recalc
    # (timeout de 15s observado en prod, corr=2eeca23b · 2026-07-06).
    # "1½" → "1 1/2" (el parser principal ya soporta mixtos con espacio).
    _mx = re.match(r'^(\d+)\s*([½¼¾⅓⅔⅕])', s_lower)
    if _mx:
        _out = f"{_mx.group(1)} {fraction_map[_mx.group(2)]}{s_lower[_mx.end():]}"
        return re.sub(r'\s{2,}', ' ', _out).strip()

    # [P1-COUNTRY-SYSTEM-F2 · Task 8 · 2026-08-17] Rango numérico LÍDER ("2–3 ciruelas",
    # "2-3 ciruelas"): la regex principal de `_parse_quantity` (línea ~2589) solo captura un
    # `\d+` simple al inicio — sin este colapso, el match falla POR COMPLETO (no hay forma de
    # consumir "–3 ciruelas" tras el primer dígito) y el fallback `if not match: return 0.0,
    # 'cantidad necesaria', normalize_name(s).strip()` pasa el string CONTAMINADO entero
    # ("2–3 ciruelas") a `normalize_name`, que no lo resuelve (ni exacto/CONTAINS ni fuzzy:
    # ratio 0.737 << 0.87 contra "ciruela") — drop real medido: 16/30d en rd_drops.json, el 2º
    # alimento más dropeado tras mereyes. Colapsa al valor MAYOR del rango (mismo criterio que
    # `humanize_ingredients._grammar_lead_value` ya usa para el DISPLAY: "el valor que concuerda
    # es el MAYOR" — y la misma filosofía "pecarse de comprar de más" de P1-CITRUS-JUICE-YIELD),
    # dejando "3 ciruelas" para que el resto del pipeline (regex principal + FUZZY plural de
    # `normalize_name`, ratio 0.933) resuelva -> Ciruela. `[-–]` (guion ASCII + en-dash, mismo
    # char-class que `_GRAMMAR_LEAD_RE` de humanize_ingredients.py) — nunca em-dash, sin
    # evidencia de ese caso en el corpus real.
    _rng = re.match(r'^(\d+)\s*[-–]\s*(\d+)\b', s_lower)
    if _rng:
        # [P1-COUNTRY-SYSTEM-F2 · 2026-08-17 (Task 9, l · fix-round T8-review)] El `\2` fijo de
        # abajo sustituía SIEMPRE por el segundo número — "colapsa al MAYOR" solo por COINCIDENCIA
        # cuando el rango viene ASCENDENTE ("2-3" → 3, sí es el mayor). Un rango DESCENDENTE
        # ("3-2 ciruelas", typo/orden invertido del LLM) tomaba el 2 — el MENOR, contradiciendo el
        # propio comentario de diseño de arriba. `re.sub` con función-repl computa max() real,
        # sin importar el orden de los 2 números. Byte-idéntico para el caso ascendente (el único
        # medido en producción — rd_drops.json).
        s_lower = re.sub(
            r'^(\d+)\s*[-–]\s*(\d+)\b',
            lambda _m: str(max(int(_m.group(1)), int(_m.group(2)))),
            s_lower,
            count=1,
        )
        return re.sub(r'\s{2,}', ' ', s_lower).strip()

    replacements = [
        # [JUICE-PREFIX-FIX 2026-05-06] Strip de prefijos descriptivos que no
        # son cantidades. El LLM emite "Zumo de 1 limón" / "Jugo de 1 limón" /
        # "Ralladura de 1 limón" como ingredientes. El regex principal de
        # `_parse_quantity` espera el string empezando con número, así que
        # estos caían al fallback `(0.0, 'cantidad necesaria', ...)` y el
        # aggregator los descartaba — el limón nunca aparecía en la lista
        # de compras aunque la receta lo usara. Strippeando el prefijo deja
        # "1 limón" → parser lo extrae correctamente.
        (r'^zumo\s+de\s+', ''),
        (r'^jugo\s+de\s+', ''),
        (r'^ralladura\s+de\s+', ''),
        (r'^c[aá]scara\s+de\s+', ''),
        (r'^un cuarto de\b', '1/4 de'),
        (r'^un cuarto\b', '1/4'),
        (r'^1 cuarto de\b', '1/4 de'),
        (r'^1 cuarto\b', '1/4'),
        (r'^tres cuartos de\b', '3/4 de'),
        (r'^tres cuartos\b', '3/4'),
        (r'^3 cuartos de\b', '3/4 de'),
        (r'^3 cuartos\b', '3/4'),
        (r'^un tercio de\b', '1/3 de'),
        (r'^un tercio\b', '1/3'),
        (r'^1 tercio de\b', '1/3 de'),
        (r'^1 tercio\b', '1/3'),
        (r'^media\b', '1/2'),
        (r'^medio\b', '1/2'),
        (r'^mitad de\b', '1/2 de'),
        (r'^mitad\b', '1/2'),
        (r'^un octavo de\b', '1/8 de'),
        (r'^un octavo\b', '1/8'),
        (r'^(cantidad necesaria|al gusto|al ojo)\s+(de\s+)?', '1 pizca de '),
        (r'^(un\s+)?chin\s+(de\s+)?', '1 chin de '),
        (r'^(un\s+)?chorrito\s+(de\s+)?', '1 chorrito de '),
        (r'^(un\s+)?toque\s+(de\s+)?', '1 toque de '),
        (r'^(una\s+)?pizca\s+(de\s+)?', '1 pizca de '),
        (r'^una\b', '1'),
        (r'^un\b', '1'),
        (r'^uno\b', '1'),
        (r'^dos\b', '2'),
        (r'^tres\b', '3'),
        (r'^cuatro\b', '4'),
        (r'^cinco\b', '5'),
        (r'^seis\b', '6'),
        (r'^siete\b', '7'),
        (r'^ocho\b', '8'),
        (r'^nueve\b', '9'),
        (r'^diez\b', '10')
    ]
    
    for pattern, repl in replacements:
        new_s = re.sub(pattern, repl, s_lower, count=1)
        if new_s != s_lower:
            return new_s.strip()

    # [P2-MIXED-FRACTION-PARSE · 2026-07-06] La expansión de fracción unicode
    # SOLA ("½ taza" → "1/2 taza") estaba MUERTA desde siempre: el loop de
    # arriba mutaba s_lower pero este return devolvía el `s` ORIGINAL cuando
    # ningún replacement posterior matcheaba → "½ taza de arroz" caía al
    # fallback qty=0.0 'cantidad necesaria' (ítem nominal sub-contado).
    if _frac_touched:
        return re.sub(r'\s{2,}', ' ', s_lower).strip()
    return s.strip()

# [P1-CITRUS-JUICE-YIELD · 2026-07-24] Rendimiento de jugo sobre fruta entera. En RD "limón"
# es el limón criollo (lima ácida), que rinde menos que un lemon: ~35% de su peso en jugo.
# Knob para poder ajustarlo sin redeploy. El multiplicador es 1/rendimiento (75 g de jugo →
# ~214 g de fruta). Se prefiere pecarse de comprar de más: un limón de sobra cuesta centavos,
# quedarse sin limón a mitad de ciclo rompe la receta.
CITRUS_JUICE_YIELD = _knob_env_float("MEALFIT_CITRUS_JUICE_YIELD", 0.35,
                                     validator=lambda v: 0.15 <= v <= 1.0)
CITRUS_JUICE_YIELD_MULT = round(1.0 / CITRUS_JUICE_YIELD, 4)
_CITRUS_JUICE_BUYABLE_CACHE: "dict | None" = None


def _citrus_juice_is_buyable(fruit_token: str) -> bool:
    """¿El catálogo vende el JUGO como producto propio? Entonces no hay nada que convertir.

    Hoy devuelve False para todos (no existe ninguna fila 'Jugo de …' — `jugo de limón` es
    alias de la fruta entera). Existe para que la regla se auto-desactive el día que se
    añada el producto, en vez de doble-contar en silencio.

    Fail-open hacia APLICAR la conversión: si el catálogo no responde, el estado conocido es
    'no existe fila de jugo', y no convertir es el bug que estamos cerrando.
    """
    global _CITRUS_JUICE_BUYABLE_CACHE
    if _CITRUS_JUICE_BUYABLE_CACHE is None:
        idx = {}
        try:
            from constants import strip_accents as _sa_cj
            for row in get_master_ingredients() or []:
                nm = _sa_cj(str(row.get("name") or "").strip().lower())
                if nm.startswith("jugo de ") or nm.startswith("zumo de "):
                    idx[nm[8:].strip()] = True
        except Exception as e:
            logging.debug(f"[P1-CITRUS-JUICE-YIELD] catálogo no disponible: {e}")
        _CITRUS_JUICE_BUYABLE_CACHE = idx
    from constants import strip_accents as _sa_cj2
    tok = _sa_cj2(str(fruit_token).lower()).rstrip("s").rstrip("e")
    return any(k.startswith(tok) for k in _CITRUS_JUICE_BUYABLE_CACHE)


def _protein_yield_on_canonical_enabled() -> bool:
    """[P2-PROTEIN-YIELD-CANONICAL · 2026-08-03] Knob del A/B: reactiva la regla #2
    (proteínas cocidas → 1.35× crudo) SOLO en la lista CANÓNICA (`is_new_plan=True`,
    sin lado inventario que la asimetría P1-2 deba proteger).

    Medición contra los 23 planes vivos (SELECT-only, 2026-08-03, ver
    `scripts/measure_cooked_protein_lines.py`): 12/5.899 líneas de `ingredients_raw`
    (0,203%) matchean la regla #2, pero **5/23 planes (~22%) tienen al menos una** —
    ejemplos reales (las 4 líneas MEDIDAS en la tabla de `test_p3_protein_yield_decision.py`,
    ninguna inventada): «160 g de pescado cocido», «100 g de cerdo magro cocido y
    desmenuzado», «45 g de costilla de cerdo cocida y desmenuzada», «40 g de pechuga de pollo
    cocido». [M-1 · review final] Aparte —EXCLUIDA del yield, control negativo con Δ=0, NO
    una de las 4 medidas—: «205 g de pollo cocido y desmenuzado (del almuerzo o preparado
    extra)», con el paréntesis COMPLETO: es justo lo que dispara `_PROTEIN_REUSE_PAREN_RE`.
    Cada match no-reuso es ~26% de under-buy de proteína en ese alimento (1 lb cocida
    declarada ⇒ solo 0,74 lb cruda comprada).

    [P3-PROTEIN-YIELD-DECISION · 2026-08-04] Decisión delegada: **FLIP a `True`**. De las
    12 líneas, 1 es de REUSO (ya excluida por `_PROTEIN_REUSE_PAREN_RE`) → 11 realmente
    afectadas. Medido OFFLINE ejecutando `get_shopping_list_delta` (misma convención
    `num_days=1` → `base_duration_scale=7` que ancla `test_p2_protein_yield_canonical.py`)
    sobre las 4 líneas no-reuso de las que tenemos texto exacto + precios RD$/lb del
    catálogo VERSIONADO (`scripts/add_foods_batch1_2026_06_26.py`: Muslo de pollo 68;
    `scripts/add_foods_batch2_2026_06_26.py`: Costilla de cerdo 189;
    `seed_supermarket_2026_07_02.py`: Cerdo genérico 115, Filete pechuga de pollo 135;
    rango real RD$68–290/lb según corte):

        160 g pescado cocido        → Δ392.00 g/sem × RD$127.5/lb ≈ RD$110.2
        100 g cerdo magro cocido     → Δ245.00 g/sem × RD$115/lb   ≈ RD$62.1
        45 g costilla de cerdo       → Δ110.25 g/sem × RD$189/lb   ≈ RD$45.9
        40 g pechuga de pollo cocido → Δ98.00 g/sem  × RD$135/lb   ≈ RD$29.2
        (control: la línea de REUSO medida da Δ=0 con el sello intacto)

    Promedio ≈ RD$61.85/línea × 2.2 líneas/plan (11 líneas / 5 planes afectados) ⇒ delta
    semanal PROMEDIO por plan afectado ≈ **RD$136**; peor caso CONSTRUIDO (cota superior:
    las 4 proteínas sumadas, ninguna de las 5 planes medidos tuvo de hecho las 4 a la
    vez) ≈ **RD$247**. Ambos números son una fracción menor (<10%) del costo semanal
    típico de una lista (RD$3.000–6.000, CLAUDE.md) — bajo el umbral ~RD$200 de la
    decisión delegada (la cota superior lo roza pero sigue siendo ruido frente al
    presupuesto semanal). El sello `protein_yield_applied` ya blinda al guard de
    coherencia en cualquier dirección del A/B (`TestGuardSealNotLiveKnob`), así que
    encender no reintroduce el bug que ese sello cerró.

    Default `True`: leído inline (no cacheado a nivel de módulo) para que los tests
    puedan togglear via `monkeypatch.setenv` sin `importlib.reload` — mismo patrón que
    `_semantic_cache_disabled`/`_trip_windowed_perishables_enabled`. Rollback sin
    redeploy: `MEALFIT_PROTEIN_YIELD_ON_CANONICAL=false`.
    """
    return _knob_env_bool("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", True)


# [P2-PROTEIN-YIELD-CANONICAL · 2026-08-03] Regex de la regla #2, extraídas a constantes
# de módulo para que el chequeo temprano (dentro de `only_legumbres_grains`) y el chequeo
# default (sin `only_legumbres_grains`) usen EXACTAMENTE el mismo criterio — cero riesgo
# de que un futuro edit añada un adjetivo/proteína a un lado y no al otro.
_PROTEIN_COOKED_ADJ_RE = re.compile(r'\b(cocid[oa]|hervid[oa]|asad[oa]|hornead[oa]|desmenuzad[oa]|frit[oa])\b')
_PROTEIN_FOOD_WORDS_RE = re.compile(r'\b(pollo|carne|res|pescado|cerdo|camar|pavo|salm[oó]n|filete)\b')

# [P2-PROTEIN-YIELD-CANONICAL · 2026-08-03] Caso borde medido en datos reales: «205 g de
# pollo cocido y desmenuzado (del almuerzo o preparado extra)» — línea de REUSO, la
# proteína ya se compró para otra comida del mismo ciclo. Aplicarle yield la
# sobre-compraría (el usuario NO necesita comprar 1.35× de algo que no va a cocinar de
# cero). Patrón derivado del dato real: paréntesis que menciona otro slot de comida
# (desayuno/almuerzo/cena/merienda) o una frase explícita de reuso (sobra/sobrante/
# preparado extra/día anterior).
#
# [ronda 1 · 2026-08-03] `sobras?` (stem "sobra"/"sobras", la forma coloquial MÁS común
# en RD — distinta de "sobrante"/"sobrantes", ya cubierto) y `d[ií]a\s+anterior` («del día
# anterior») añadidos. Esto es DETECCIÓN BEST-EFFORT CONSERVADORA sobre el vocabulario de
# reuso observado hasta ahora, no un intento de cobertura exhaustiva — un futuro sinónimo
# no listado aquí simplemente recibe yield (sobre-compra leve, ~26%), nunca al revés
# (nunca se infla silenciosamente una línea que SÍ es de reuso real y detectada).
_PROTEIN_REUSE_PAREN_RE = re.compile(
    r'\([^)]*\b(?:desayuno|almuerzo|cena|merienda|sobras?|sobrantes?|'
    r'preparad[oa]\s+extra|d[ií]a\s+anterior)\b[^)]*\)',
    re.IGNORECASE,
)


def _calculate_yield_multiplier(raw_name: str, *, only_legumbres_grains: bool = False,
                                 apply_protein_yield: bool = False) -> float:
    """Devuelve el multiplicador de yield (cocido↔crudo) para `raw_name`.

    Reglas (en orden):
      1. Legumbres/granos cocidos → 0.35× (1 taza seca rinde ~3 tazas cocidas)
      2. Proteínas cocidas        → 1.35× (peso cocido pierde ~25% a humedad)
      3. Víveres pelados          → 1.30× (merma de cáscara)
      4. Carnes sin hueso         → 1.40× (merma de hueso)
      Default                     → 1.0×

    [P2-PDF-1] `only_legumbres_grains` activa SOLO la regla #1, ignorando
    el resto. Usado por el shopping aggregator vía `_parse_quantity` para
    convertir "200g habichuelas cocidas" → 70g secas — el SKU comercial
    de habichuelas/lentejas/arroz/pasta es SECO, así que sin esta
    conversión el aggregator computaba en peso cocido (~3× sobre-estimado)
    y producía conteos exagerados de paquetes (15 paquetes de habichuelas
    cuando realmente se necesitan ~5 lbs secas).

    Por qué SOLO esta regla: el aggregator pasa `apply_yield_multiplier=
    False` por la asimetría P1-2 plan↔inventario (proteínas cocidas
    descritas en plan vs. inventario en peso literal sin "cocido" sesgan
    el delta hacia over-buy). Para PROTEÍNAS la asimetría es ~25%
    (aceptable). Para LEGUMBRES/GRANOS es 3× (material) y los SKUs son
    SECOS — la regla #1 cierra el gap sin reintroducir la asimetría #2.

    [P2-PROTEIN-YIELD-CANONICAL · 2026-08-03] `apply_protein_yield` reabre
    SELECTIVAMENTE la regla #2 dentro de `only_legumbres_grains=True`
    (modo aggregator). Solo tiene efecto cuando el caller es la lista
    CANÓNICA (`is_new_plan=True`): ahí no existe lado inventario que la
    asimetría P1-2 deba proteger, así que "1 lb de pollo cocido" compra el
    equivalente crudo (1.35 lb) en vez de comprar 1 lb literal que rinde
    solo ~0.74 lb cocidas tras la cocción (~26% menos proteína de la que
    el plan calculó). Excluye líneas con marcador de REUSO (paréntesis
    "(del almuerzo/de la cena/...)" o "preparado extra"/"sobrante"): esa
    proteína ya se compró para otra comida, aplicarle yield sobre-compraría.
    """
    n = raw_name.lower()
    # 1. Pastas y Granos cocidos (Expanden, necesitas menos crudo)
    # [P2-PDF-1] Soporte de plural agregado: antes la regex `\bhabichuela\b`
    # NO matcheaba "habichuelas" porque `\b` requiere boundary y `s` es word
    # char → "habichuelas cocidas" salía con yield=1.0 silenciosamente. Para
    # palabras cuyo plural agrega `s` simple (lenteja→lentejas, habichuela→
    # habichuelas, pasta→pastas, quinoa→quinoas) usamos sufijo `s?`. Para
    # los que pluralizan con `es` (frijol→frijoles, guandul→guandules) usamos
    # `(?:es)?` para no match accidentes como "frijole". Para `arroz` añadimos
    # `(?:es)?` defensivo (raramente plural).
    #
    # [P2-PDF-3] `garbanzo(s)?`, `soya`, `tofu` añadidos del PDF 2026-05-05:
    # "250g garbanzos cocidos" se aggregaba sin yield → 11 paquetes (1 lb)
    # en lugar de los ~5 lbs secas reales (over-buy 2×). `soya` y `tofu`
    # incluidos por simetría — la soya texturizada y el tofu firme también
    # se hidratan ~3× al cocinarse desde su forma comercial seca.
    if bool(re.search(r'\b(cocid[oa]s?|hervid[oa]s?)\b', n)) and bool(re.search(r'\b(arroz(?:es)?|pastas?|quinoas?|lentejas?|habichuelas?|frijol(?:es)?|guandul(?:es)?|garbanzos?|soyas?|tofu)\b', n)):
        return 0.35

    # 1b. [P1-CITRUS-JUICE-YIELD · 2026-07-24] Jugo de cítrico → FRUTA ENTERA.
    #
    # El catálogo no tiene ninguna fila de jugo: "jugo de limón" es un ALIAS de `Limón`,
    # la fruta entera (verificado en Neon). Así que "2 cdas de jugo de limón" se agregaba
    # como si fueran 30 g de limón entero, cuando exprimir 30 g de jugo exige ~86 g de
    # fruta. Plan vivo 732588f8: 5 cdas de jugo (~75 g) en 3 días → la lista compró
    # 2 limones, ~0.3× de lo necesario. El usuario se queda sin limón a mitad del ciclo.
    #
    # Va en el tramo compartido con la regla #1 (antes del early-return del aggregator)
    # porque es el MISMO tipo de desajuste: la receta habla en una forma y el SKU se vende
    # en otra. Y no reintroduce la asimetría P1-2 que motivó ese early-return: el inventario
    # del usuario también habla de limones enteros (nace de esta misma lista), así que tras
    # la conversión ambos lados usan la misma unidad.
    #
    # Se auto-desactiva si algún día el catálogo incorpora el jugo como producto comprable
    # (ahí ya no hay nada que convertir) — mismo patrón que el factor cocido→seco.
    if re.search(r'\b(jugo|zumo)\b', n):
        _fruit = re.search(r'\b(lim[oó]n(?:es)?|lima|naranjas?|toronjas?|mandarinas?)\b', n)
        if _fruit and not _citrus_juice_is_buyable(_fruit.group(1)):
            return CITRUS_JUICE_YIELD_MULT

    if only_legumbres_grains:
        # Modo aggregator: NO aplicar reglas #2-4 para preservar la simetría
        # plan↔inventario establecida en P1-2.
        #
        # [P2-PROTEIN-YIELD-CANONICAL · 2026-08-03] EXCEPCIÓN: el caller canónico
        # (is_new_plan=True, sin lado inventario) puede pedir explícitamente que la
        # regla #2 SÍ aplique — ahí no existe simetría que proteger. Reusa el mismo
        # regex de la regla #2 default (constantes de módulo), y excluye líneas de
        # REUSO ("(del almuerzo/de la cena/...)" / "preparado extra"/"sobrante"): esa
        # proteína ya se compró para otra comida, aplicarle yield sobre-compraría.
        if (apply_protein_yield and not _PROTEIN_REUSE_PAREN_RE.search(n)
                and bool(_PROTEIN_COOKED_ADJ_RE.search(n)) and bool(_PROTEIN_FOOD_WORDS_RE.search(n))):
            return 1.35
        return 1.0

    # 2. Proteínas cocidas (Se encogen por humedad, necesitas más crudo)
    if bool(_PROTEIN_COOKED_ADJ_RE.search(n)) and bool(_PROTEIN_FOOD_WORDS_RE.search(n)):
        return 1.35

    # 3. Merma de Cáscara/Limpieza (Víveres y Mariscos pelados)
    if bool(re.search(r'\b(pelad[oa]|limpi[oa]|sin piel|sin c[aá]scara)\b', n)) and bool(re.search(r'\b(yuca|platano|pl[aá]tano|batata|papa|guineo|camar[oó]n|manzana|pera)\b', n)):
        return 1.30

    # 4. Merma de Hueso (comprar sin hueso es más carne, pero si la receta pide carne magra y el ingrediente en lista es estándar)
    if bool(re.search(r'\b(sin hueso|deshuesad[oa])\b', n)) and bool(re.search(r'\b(pollo|muslo|carne|chuleta)\b', n)):
        return 1.40

    return 1.0

# [P1-DOUBLE-QTY-PARSE · 2026-07-27] Dos cantidades pegadas al frente: una fracción unicode
# seguida de una fracción ASCII ("1½ 1/2 cdas de mantequilla de maní"). Se conserva la primera.
# No toca el mixto legítimo "1 1/2 cdas" (el grupo exige terminar en fracción unicode) ni
# "½ taza de agua" (tras la fracción viene una palabra, no un dígito).
_DOUBLE_LEAD_QTY_RE = re.compile(r"^\s*(\d*[¼½¾⅓⅔])\s+\d+\s*/\s*\d+\s+")


_HINT_TRUMPS_QTY = _knob_env_bool("MEALFIT_GRAM_HINT_TRUMPS_QTY", True)
_HINT_TRUMPS_RATIO = _knob_env_float("MEALFIT_GRAM_HINT_TRUMPS_RATIO", 5.0,
                                     lambda v: 1.5 <= v <= 100.0)
# Pista de gramos que la propia línea declara: "(4 g)", "(39g)", "(≈147 g)", "(aprox. 204 g)",
# "(149g, lavadas)". Solo gramos: un "(2 tazas)" no es una declaración de peso.
_GRAM_HINT_RX = re.compile(r"\(\s*(?:[≈~]|aprox\.?|unos)?\s*(\d+(?:[.,]\d+)?)\s*g(?:r|ramos)?\b",
                            re.IGNORECASE)
# Conversión GRUESA a gramos, solo para detectar contradicciones de ORDEN DE MAGNITUD. No sirve
# para calcular nada: una taza de lechuga no pesa 240 g. Por eso el umbral es de 5× y no del 20%.
_COARSE_G_PER_UNIT = {"g": 1.0, "gr": 1.0, "gramo": 1.0, "gramos": 1.0, "ml": 1.0,
                      "cda": 15.0, "cdas": 15.0, "cucharada": 15.0, "cucharadas": 15.0,
                      "cdta": 5.0, "cdtas": 5.0, "cucharadita": 5.0, "cucharaditas": 5.0,
                      "taza": 240.0, "tazas": 240.0}


def _reconcile_qty_with_gram_hint(raw_line, qty, unit):
    """[P1-GRAM-HINT-TRUMPS-QTY · 2026-07-30] Si la línea se CONTRADICE a sí misma, gana la pista.

    Caso vivo del owner (plan 307395c7, desayuno del día 2):

        ingredients      →  '¼ cda de mantequilla de maní (4 g)'     ← lo que el usuario LEE
        ingredients_raw  →  '30 cdas de mantequilla de maní (4 g)'   ← lo que la lista COMPRA

    La lista de compras lee `ingredients_raw`. 30 cdas ≈ 450 g por aparición × el multiplicador del
    ciclo = 4.515 g ⇒ **10 potes de mantequilla de maní, RD$1.170**, la línea más cara del ciclo,
    para una tostada que lleva 4 g. El display decía ¼ cda.

    Lo que hace este guard: la línea raw trae DENTRO la prueba de que está mal — su propio
    paréntesis dice `(4 g)` y `_parse_quantity` lo tiraba a la basura. Cuando la cantidad parseada
    convierte a algo ≥`RATIO`× distinto de lo que la propia línea declara pesar, se cree a la pista.

    Por qué la pista y no la cantidad: el paréntesis es lo que el usuario ve, es de donde salieron
    los macros del plato, y en este producto denota SIEMPRE el peso total de la cantidad indicada
    ('2¼ tazas de fresas (338g)', '½ guineo (102g)'). Una cantidad sin pista no se toca nunca.

    Por qué 5× y no 20%: la conversión a gramos es GRUESA (1 taza = 240 g) y una taza de lechuga no
    pesa eso. El umbral tiene que dejar pasar el error honesto de densidad y cazar solo la
    contradicción de orden de magnitud — aquí fue de 112×.

    ⚠️ Va DENTRO del parser, no en el agregador, a propósito: `expected_sum_from_recipes` parsea las
    MISMAS líneas para el guard de coherencia. Corregir solo en el agregador haría que los dos lados
    discreparan y el guard bloquearía el plan por un arreglo.
    tooltip-anchor: P1-GRAM-HINT-TRUMPS-QTY"""
    if not _HINT_TRUMPS_QTY:
        return qty, unit
    try:
        if qty is None or float(qty) <= 0:
            return qty, unit
        m = _GRAM_HINT_RX.search(str(raw_line or ""))
        if not m:
            return qty, unit
        hint_g = float(m.group(1).replace(",", "."))
        if hint_g <= 0:
            return qty, unit
        factor = _COARSE_G_PER_UNIT.get(str(unit or "").strip().lower())
        if not factor:
            return qty, unit          # unidad no convertible (unidad/lata/pote…): sin veredicto
        approx_g = float(qty) * factor
        if approx_g <= 0:
            return qty, unit
        ratio = max(approx_g / hint_g, hint_g / approx_g)
        if ratio < _HINT_TRUMPS_RATIO:
            return qty, unit
        logging.warning(
            f"⚠️ [P1-GRAM-HINT-TRUMPS-QTY] línea contradictoria: '{str(raw_line)[:70]}' declara "
            f"{hint_g:.0f} g pero {qty:g} {unit} ≈ {approx_g:.0f} g ({ratio:.0f}× de diferencia) — "
            f"se usa la pista. Sin esto la lista compra por la cantidad, no por el peso.")
        return hint_g, "g"
    except Exception as _e_hint:
        logging.warning(f"[P1-GRAM-HINT-TRUMPS-QTY] no-op: {type(_e_hint).__name__}: {_e_hint}")
        return qty, unit


def _parse_quantity(s, *, apply_yield_multiplier: bool = True, apply_legumbres_yield_only: bool = False,
                     apply_protein_yield: bool = False):
    """[P1-2] Parsea un string de ingrediente a (qty, unit, name).

    `apply_yield_multiplier` controla si `_calculate_yield_multiplier` se
    aplica al qty extraído (default True para preservar el comportamiento
    de todos los call-sites históricos: tools.py, cron_tasks.py,
    db_inventory.py, etc. que dependen de yield→peso-crudo).

    El aggregator de la lista de compras (`aggregate_and_deduct_shopping_list`)
    lo invoca con `apply_yield_multiplier=False` para evitar la asimetría
    documentada en P1-2: el plan_ingredients del LLM frecuentemente describe
    el plato cocido ("1 lb pollo cocido") y `_parse_quantity` aplicaría
    yield 1.35 → 1.35 lb crudo. PERO el `physical_inventory` que el usuario
    tipea en su Nevera está SIEMPRE en peso literal sin "cocido" → yield 1.0
    → 1.0 lb. Esa asimetría textual sesgaba el delta plan-inventario hacia
    OVER-BUYING. Operando en peso literal en ambos lados, el delta refleja
    fielmente la diferencia descrita por LLM/usuario sin conversiones
    asimétricas.

    [P2-PDF-1] `apply_legumbres_yield_only` re-activa SOLO la regla
    legumbres/granos cocidos→secos (factor 0.35×) en el path del aggregator.
    Justificación: el SKU comercial de habichuelas/lentejas/arroz/pasta es
    SECO; sin esta conversión, "200g habichuelas cocidas" → 200g se
    aggregaba como si fuera 200g secas, sobreestimando 3× el conteo de
    paquetes en la lista de compras. La asimetría plan↔inventario que
    P1-2 cerró aplica a PROTEÍNAS (25% delta, simétrico aceptable);
    para LEGUMBRES la asimetría es 3× y se cierra solo en este lado
    porque el inventario también se canonicaliza al name seco antes de
    deducir.

    [P2-PROTEIN-YIELD-CANONICAL · 2026-08-03] `apply_protein_yield` reabre
    SELECTIVAMENTE la regla #2 (proteínas cocidas) dentro del modo
    `apply_legumbres_yield_only=True`. Solo tiene efecto real cuando el
    caller pasa AMBOS flags juntos (ver `aggregate_and_deduct_shopping_list`
    con `is_new_plan=True` y el knob `MEALFIT_PROTEIN_YIELD_ON_CANONICAL`
    encendido) — con `apply_yield_multiplier=True` (callers históricos) esta
    regla ya corre incondicionalmente y este flag no cambia nada.
    """
    # [P1-DOUBLE-QTY-PARSE · 2026-07-27] "1½ 1/2 cdas de mantequilla de maní (39 g)" degradaba en
    # silencio: la unidad caía de 'cda' a 'unidad' y el NOMBRE salía como "1/2 cdas de mantequilla
    # de maní". Como "una unidad" de mantequilla de maní es un pote, la lista compró UN POTE POR
    # CUCHARADA — 14 potes, RD$1.638, la mitad del sobrecoste del ciclo de 30 días.
    #
    # ⚠️ El productor de esa forma NO se pudo reproducir: ni `_prettify_quantity_display` ni
    # `_collapse_double_fraction` la generan, y el modelo escribe el mixto legítimo "1 1/2 cdas",
    # que este parser YA resolvía bien. Se endurece el CONSUMIDOR, que es donde se pierde el dinero
    # y donde devolver un nombre basura es un defecto por sí mismo, venga la entrada de donde venga.
    if isinstance(s, str):
        s = _DOUBLE_LEAD_QTY_RE.sub(r"\1 ", s, count=1)

    if isinstance(s, dict):
        # [P3-PARSE-QTY-DICT-GUARD · 2026-05-30] Blindar simétricamente con la
        # rama string (que ya cae a 0.0 vía parse_fraction). Un futuro caller
        # que pase un dict crudo con quantity='½'/'dos'/None/'inf' lanzaría
        # ValueError/TypeError o propagaría NaN/Inf a la lista. Sin import math:
        # `qty != qty` detecta NaN; `in (inf,-inf)` detecta Inf.
        try:
            qty = float(s.get("quantity", 0))
            if qty != qty or qty in (float("inf"), float("-inf")):
                qty = 0.0
        except (TypeError, ValueError):
            qty = 0.0
        unit = s.get("unit", "unidad")
        if unit:
            unit = str(unit).strip().lower()
        if not unit:
            unit = "unidad"
        name_raw = s.get("name") or s.get("ingredient_name") or s.get("item_name") or "Desconocido"
        return qty, unit, normalize_name(name_raw).strip()

    s_lower = str(s).lower().strip()
    
    # Mejora 3: Si contiene términos puramente informales SIN NÚMEROS (ej: "sal al gusto")
    # los mandaremos como nominal 0.0 para no alterar matemáticamente la despensa pero sí listarlos.
    abstract_terms = ['al gusto', 'al ojo', 'cantidad necesaria']
    for term in abstract_terms:
        if term in s_lower and not any(char.isdigit() for char in s_lower):
            clean_s = s_lower.replace(term, '').replace(' de ', ' ').strip()
            return 0.0, 'pizca', normalize_name(clean_s).strip()
            
    s = _preprocess_nlp_quantities(s)
    # Limpieza previa: si el AI genera "1 Ud." o "2 Uds.", limpiar el punto
    s = re.sub(r'\b([Uu]ds?)\.', r'\1', s)
    match = re.search(r'^(\d+(?:\s+\d+\/\d+|\/\d+|\.\d+)?)\s*(?:de\s+)?([a-zA-ZáéíóúÁÉÍÓÚñÑ]+)?(?:\s+(.*))?$', s)
    if not match:
        return 0.0, 'cantidad necesaria', normalize_name(s).strip()
    
    qty_str = match.group(1)
    unit_str = match.group(2)
    rest_str = match.group(3) or ""
    
    raw_qty = parse_fraction(qty_str)

    # [P1-2] yield_mult solo se aplica si el caller lo pidió explícitamente.
    # El aggregator pasa False para evitar la asimetría plan-vs-inventory
    # cuando solo el plan describe productos cocidos.
    # [P2-PDF-1] `apply_legumbres_yield_only` activa SELECTIVAMENTE la regla
    # legumbres/granos (0.35×) sin reabrir la asimetría de proteínas (#2-4).
    if apply_yield_multiplier:
        yield_mult = _calculate_yield_multiplier(rest_str)
    elif apply_legumbres_yield_only:
        yield_mult = _calculate_yield_multiplier(
            rest_str, only_legumbres_grains=True, apply_protein_yield=apply_protein_yield,
        )
    else:
        yield_mult = 1.0
    qty = raw_qty * yield_mult
    
    # [P1-shop-coh-1 · 2026-05-07] Lookup contra SSOT en `canonical_units.py`.
    # Antes era cadena if/elif duplicada con `db_inventory._CANONICAL_UNIT_MAP`;
    # divergencia silenciosa entre los dos hacía que aliases nuevos sólo
    # canonicalizaran de un lado, generando mismatches plan↔inventario.
    # Histórico de aliases que vivieron aquí (preservados en el SSOT):
    #   - cdas/cdtas plurales (P6-CDA-PLURAL-FIX 2026-05-07)
    #   - frascos plural (P5-OLIVE-CAP)
    #   - caja/bolsa/tetra/galón/jarra (P1-3 container aliases)
    #   - mazo/atado/manojo (P3-HERB-CAP)
    if unit_str:
        canonical = canonicalize_unit(unit_str)
        if canonical is not None:
            unit_str = canonical
        else:
            # Alias desconocido: la regex extrajo como `unit_str` algo que
            # en realidad pertenece al name. Rebobinar y caer a 'unidad'.
            rest_str = unit_str + (" " + rest_str if rest_str else "")
            unit_str = 'unidad'
    else:
        unit_str = 'unidad'

    # [P1-GRAM-HINT-TRUMPS-QTY · 2026-07-30] Último paso: si la línea declara su peso y la cantidad
    # parseada lo contradice por orden de magnitud, gana la pista. Ver el helper.
    qty, unit_str = _reconcile_qty_with_gram_hint(s, qty, unit_str)

    return qty, unit_str, normalize_name(rest_str).strip()
    
# [P1-PDF-LIST-POLISH · 2026-09-02] SSOT de plurales de envase/unidad. La tabla vivía
# inline dentro de `get_plural_unit` y le faltaban envases REALES del catálogo (medido en
# Neon 2026-09-02: 'funda' ×6 filas, 'malla', 'manojo', 'libra', 'litro') ⇒ el PDF
# imprimía «3 funda (…)». Un envase nuevo en master_ingredients se añade AQUÍ y en el
# glosario del PDF (`shoppingHelpers.js`, test_p2_i18n_pdf_categorias lo exige).
UNIT_PLURALS = {
    'lb': 'lbs', 'lbs': 'lbs', 'libra': 'libras', 'litro': 'litros',
    'paquete': 'paquetes', 'pote': 'potes', 'unidad': 'unidades',
    'lata': 'latas', 'cabeza': 'cabezas', 'diente': 'dientes',
    'cartón': 'cartones', 'carton': 'cartones',
    'sobre': 'sobres', 'sobrecito': 'sobrecitos',
    'botella': 'botellas', 'frasco': 'frascos',
    'funda': 'fundas', 'fundita': 'funditas', 'malla': 'mallas',
    'mazo': 'mazos', 'manojo': 'manojos', 'envase': 'envases',
    'tarro': 'tarros', 'barrita': 'barritas',  # [P3-PKG-DAIRY-VEG · 2026-06-22] mantequilla
    'rebanada': 'rebanadas', 'hoja': 'hojas',
    'cda': 'cdas', 'cdta': 'cdtas', 'taza': 'tazas',
    'ud.': 'Uds.',
}

def get_plural_unit(num, u):
    if num <= 1 or not u: return u
    # [P1-EGG-CARTON-SIZES · 2026-06-22] Unidades con sufijo parentético, p.ej.
    # 'cartón (30 uds.)' → pluralizar SOLO la palabra-cabeza y re-anexar el sufijo
    # ('2 cartones (30 uds.)', no '2 cartón (30 uds.)'). Para unidades simples (sin
    # paréntesis) el comportamiento es idéntico al previo.
    _paren_suffix = ""
    _m_paren = re.match(r'^\s*([^(]+?)\s*(\(.*\))\s*$', u)
    if _m_paren:
        u = _m_paren.group(1).strip()
        _paren_suffix = " " + _m_paren.group(2)
    u_lower = u.lower()
    result = UNIT_PLURALS.get(u_lower, u)
    # Preservar capitalización del input: si "Pote" → "Potes", si "pote" → "potes"
    if len(result) > 0 and u[0].isupper() and result[0].islower():
        result = result[0].upper() + result[1:]
    return result + _paren_suffix

# Mínimos comprables en mercado/colmado dominicano
MARKET_MINIMUMS = {
    "lb": 0.25,       # No se vende menos de 1/4 lb
    "lbs": 0.25,
    "pote": 1,        # No puedes comprar "medio pote"
    "paquete": 1,     # Siempre se compra entero  
    "fundita": 1,
    "mazo": 1,
    "lata": 1,
    "sobre": 1,
    "sobrecito": 1,
    "frasco": 1,
    "botella": 1,
    "cartón": 1,
    "carton": 1,
    "envase": 1,
    "tarro": 1,
    "barrita": 1,
    "cabeza": 1,
    "ud.": 1,
    "ud": 1,
}

# Mapeo canónico de categorías DB → categorías de display para PDF
DISPLAY_CATEGORY_MAP = {
    "Proteínas":        "PROTEÍNAS",
    "Lácteos":          "LÁCTEOS",
    "Frutas":           "FRUTAS",
    "Vegetales":        "VEGETALES",
    "Víveres":          "VÍVERES",
    "Despensa":         "DESPENSA",
    "Despensa y Granos": "DESPENSA",
    "Especias":         "ESPECIAS",
    "Suplementos":      "SUPLEMENTOS",
}

# ============================================================
# [P1-PDF-2] Clasificación canónica perecedero vs estable.
# ------------------------------------------------------------
# Antes, el PDF de la lista de compras tenía la heurística DUPLICADA:
#   - Frontend (`Dashboard.jsx`):
#     `cat.toLowerCase().includes('proteína'|'lácteo'|'vegetal'|'fruta')`
#   - Backend: ninguna — el frontend tomaba la decisión sin SSOT.
# Si `_get_display_category` devolvía una variante con typo o sin tilde
# ("Proteinas" sin acento, "vegetales" plural), la heurística de substring
# fallaba silenciosamente y items perecederos quedaban en la sección estable
# del PDF — riesgo concreto para el usuario que compra carne para "más de
# 7 días" porque el PDF la presentó como "+7 días almacén".
#
# Ahora el backend persiste `is_perishable: bool` por item en
# `aggregated_shopping_list`. El frontend lee el flag directo; mantiene la
# heurística como fallback defensivo solo para planes legacy persistidos
# antes de este fix.
#
# Reglas (en orden de precedencia):
#   1. `shelf_life_days` ≤ PERISHABLE_SHELF_LIFE_THRESHOLD_DAYS → perecedero
#      (señal más confiable, viene de master_ingredients o `_infer_shelf_life_days`).
#   2. Categoría (case-insensitive, accent-aware vía substring) coincide con
#      uno de PERISHABLE_CATEGORY_PREFIXES → perecedero.
#   3. Items urgentes (`category='🚨 Compra Urgente'`) → siempre perecedero
#      (semántica del flag: "comprar pronto").
#   4. Sino → estable (default conservador para "DESPENSA", "VÍVERES",
#      "ESPECIAS", "SUPLEMENTOS").
#
# Mantenimiento: si se añade una categoría nueva (ej. "Embutidos"), evaluar
# si entra en este set Y actualizar `_infer_shelf_life_days` en
# `db_inventory.py` para coherencia con la regla 1.
# ============================================================
PERISHABLE_CATEGORY_PREFIXES = frozenset({
    "proteína",
    "lácteo",
    "vegetal",
    "fruta",
    # [2026-05-06] Añadidos `víver` y `hierba` tras el bug de la lista
    # weekly: tubérculos frescos (Batata, Yautía, Plátano verde) viven en
    # cat='Víveres' en master, y hierbas frescas (Cilantro) en cat='Hierbas'.
    # Ambos son perecederos (7-14 días en clima tropical) pero antes caían
    # al fallback de shelf=14 → False (stable). Mismas categorías están en
    # `_PERISHABLE_CATEGORIES` que usa `_classify_perishability`.
    "víver",
    "hierba",
})

PERISHABLE_SHELF_LIFE_THRESHOLD_DAYS = 7


def is_perishable_category(category: str | None, shelf_life_days=None, name: str | None = None) -> bool:
    """[P1-PDF-2] Determina si un item de la lista de compras es perecedero.

    Helper canónico que reemplaza la heurística de substring duplicada en
    `Dashboard.jsx`. Devuelve `True` si el item debe agruparse en la sección
    "COMPRA INMEDIATA" del PDF (perecederos 1-7 días).

    Args:
        category: categoría cruda (`master_ingredients.category` o
            `display_category`). Tolerante a None, mayúsculas, acentos
            y formato plural ("PROTEÍNAS" vs "Proteína").
        shelf_life_days: días de shelf life del item. None / no parseable →
            cae a la regla de categoría.

    [2026-05-06 FIX] Antes shelf_life_days corría primero. master_ingredients
    persiste shelf_life_days=14 como default genérico para casi TODOS los
    items frescos (cerdo, lechosa, mango, queso blanco, brócoli, tomate,
    yautía, batata…) — valor incorrecto pero ampliamente desplegado.
    Con threshold=7, ese 14 devolvía False (stable) y `aggregated_shopping_list_weekly`
    (que NO pasa por `_build_hybrid_shopping_list`, va directo del aggregator)
    contaminaba la sección "DESPENSA — ESTABLES" del PDF weekly con carnes,
    frutas y vegetales que claramente son perecederos.
    Ya alineamos `_classify_perishability` con esta misma precedencia
    (cat → shelf → default); aquí replicamos para que el path weekly
    quede consistente con biweekly/monthly.

    Precedencia (alta → baja):
      1. Categoría stable explícita (`_STAPLE_CATEGORIES`: despensa, granos,
         conservas, especias, etc.) → False. Cubre canned proteins / sauces /
         spices que comparten cat raíz pero son estables.
      2. Categoría perecedera explícita ("urgente" o substring de
         `PERISHABLE_CATEGORY_PREFIXES`: proteína, lácteo, vegetal, fruta) → True.
         master.category es señal humana curada — gana sobre datos numéricos
         default-14.
      3. shelf_life_days fallback (categoría desconocida, e.g., "Otros"):
         shelf ≤ 7 → True, sino False.
      4. Default → False (conservador).
    """
    from constants import strip_accents
    cat_lower = str(category or "").strip().lower()
    cat_norm = strip_accents(cat_lower)

    # [P1-TORTILLA-PERECEDERO · 2026-07-06] Override por nombre — espejo del que
    # `_classify_perishability` aplica dentro de la rama `_STAPLE_CATEGORIES`.
    # Este clasificador (SSOT del flag `is_perishable` que consume el frontend/PDF,
    # path weekly) era category-only y NO tenía hook por nombre: panes/tortillas
    # frescos catalogados como "Despensa" caían a la Regla 1 (stable) SIEMPRE.
    # Sin este override, la tortilla integral quedaba en "estables — compra una
    # sola vez" en el weekly aunque `_classify_perishability` ya la trate como
    # perecedera en biweekly/monthly → inconsistencia visible cross-duración.
    if name:
        _name_norm = strip_accents(str(name).lower().strip())
        if any(exc in _name_norm for exc in _DESPENSA_PERISHABLE_EXCEPTIONS):
            return True

    # Pre-parse shelf_life para usar en múltiples reglas.
    shelf_int = None
    if shelf_life_days is not None:
        try:
            shelf_int = int(shelf_life_days)
        except (TypeError, ValueError):
            pass

    # Regla 1: categorías stable explícitas → siempre estable.
    # Atún en lata / aceitunas / salsa de soya viven aquí (cat='Despensa' /
    # 'Conservas') y NO en la categoría de su proteína fuente.
    if cat_norm in _STAPLE_CATEGORIES:
        return False

    # Regla 2: shelf_life largo (≥30 días) override — cubre proteínas/lácteos
    # enlatados o curados que viven en cat='Proteínas'/'Lácteos' por su origen
    # alimentario pero realmente son estables en almacén.
    # Ejemplos:
    #   - Atún en agua (cat=Proteínas, shelf=730) → enlatado, durabilidad 2 años
    #   - Leche UHT (cat=Lácteos, shelf=180) → tetra brik
    #   - Queso parmesano (cat=Lácteos, shelf=120) → curado
    # Sin esta regla, el match de categoría perishable (Regla 3) los enviaría
    # incorrectamente a la sección "Compra cada 7 días" del PDF weekly.
    # Threshold 30d filtra defaults dudosos (14d) sin afectar enlatados reales.
    _STAPLE_BY_LONG_SHELF_DAYS = 30
    if shelf_int is not None and shelf_int >= _STAPLE_BY_LONG_SHELF_DAYS:
        return False

    # Regla 3a: items urgentes siempre perecederos.
    if "urgente" in cat_lower:
        return True

    # Regla 3b: categoría perecedera explícita (substring para tolerar
    # plurales: "VEGETALES" contiene "vegetal", "FRUTAS" contiene "fruta").
    # Con shelf_life_days=14 default en DB para casi todos los frescos,
    # este match de categoría es la señal de verdad — la curación humana
    # gana sobre el dato numérico genérico.
    if any(prefix in cat_lower for prefix in PERISHABLE_CATEGORY_PREFIXES):
        return True

    # Regla 4: shelf_life_days como fallback cuando la categoría no da señal
    # clara (ej. "Otros", "Suplementos", categoría nueva no listada).
    if shelf_int is not None:
        return shelf_int <= PERISHABLE_SHELF_LIFE_THRESHOLD_DAYS

    # Regla 5: default conservador.
    return False

def _get_display_category(db_category: str, name: str = "") -> str:
    """Resuelve la categoría de display para el PDF. Server-side, elimina regex del frontend."""
    if db_category in DISPLAY_CATEGORY_MAP:
        return DISPLAY_CATEGORY_MAP[db_category]
    # Fallback NLP para ingredientes sin categoría en DB
    n = name.lower()
    if re.search(r'pollo|carne|pescado|\bres\b|cerdo|huevo|camar|at[uú]n|sardina|pavo|jam[oó]n|tocineta|salchicha|longaniza|salami', n):
        return "PROTEÍNAS"
    if re.search(r'queso|leche|yogur|crema|ricotta|cottage|mozzarella|mantequilla|margarina', n):
        return "LÁCTEOS"
    if re.search(r'manzana|guineo|naranja|fresa|chinola|mango|pi[ñn]a|lechosa|aguacate|lim[oó]n|pera|uva|mel[oó]n|sand[ií]a|kiwi|cereza|durazno|banana', n):
        return "FRUTAS"
    if re.search(r'tomate|cebolla|aj[ií]|zanahoria|br[oó]coli|espinaca|lechuga|pepino|ajo|cilantro|apio|repollo|coliflor|tayota|berenjena|vainita|molondr|auyama|jengibre|r[aá]bano|pimiento|habichuel[ií]ta', n):
        return "VEGETALES"
    if re.search(r'pl[aá]tano|papa|yuca|batata|yaut[ií]a|[ñn]ame|guine[ií]to', n):
        return "VÍVERES"
    if re.search(r'arroz|pasta|avena|harina|habichuela|frijol|lenteja|garbanzo|quinoa|guand[uú]l|\bpan\b', n):
        return "DESPENSA"
    if re.search(r'aceite|\bsal\b|pimienta|or[eé]gano|canela|comino|vinagre|miel|salsa|semilla|almendra|nuez|man[ií]|ch[ií]a|az[uú]car|caf[eé]|saz[oó]n', n):
        return "DESPENSA"
    return "OTROS"

# [P1-SKU-COVER-HONESTY · 2026-08-02] La rama `under_buy_g < over_buy_g` (⟺ frac<0.5) de los 3
# selectores de envase de abajo permitía retener el floor (comprar 1 envase de menos) con un
# under-buy de hasta ~50% del total sin ningún aviso — medido en prod: 18/22 planes con ≥1 ítem
# cover<0.9 sin nota (arroz 0.69-0.81, aceite 0.70-0.93, camarones 0.76-0.79). Acota ese under-buy
# absoluto a `SKU_FLOOR_MAX_UNDER_PCT` del total (default 10%, alineado con la tolerancia del
# coherence guard) — por encima, se compra el paquete extra (ceil). La rama previa
# `frac <= ANTI_WASTE_THRESHOLD` (colchón de 2% para errores de coma flotante, SKU-OVERSHOOT-FIX)
# se conserva intacta: es política anti-desperdicio deliberada, no el bug.
#
# [P1-SKU-COVER-HONESTY-R1 · 2026-08-02] (ronda 1 de revisión) El bound puro
# `under_buy_g <= g_total * PCT` NO es uniformemente más estricto que el criterio viejo: escala
# con `floor_units`, así que para conteos altos (`floor_units>=5` con PCT=0.10) permite MÁS
# under-buy relativo que `under_buy < over_buy`, y desde `floor_units>=9` es directamente vacuo
# (nunca fuerza ceil). Medido: Sazón 137g/sobre 14g → floor_units=9, el bound puro retiene 9
# sobres (cover 0.92) donde el criterio viejo hacía ceil a 10 (cover 1.022) — un déficit NUEVO
# que el código pre-fix no tenía. Fix: exigir AMBAS condiciones (`under_buy <= bound AND
# under_buy < over_buy`), de modo que el conjunto de casos donde se retiene el floor es un
# SUBCONJUNTO estricto del criterio viejo — nunca puede retener floor donde el código viejo
# hacía ceil, sólo puede convertir floor→ceil donde el viejo criterio permitía under-buy
# excesivo. Ver `test_bound_no_introduce_deficit_nuevo_en_floor_alto` (verificado con barrido
# aleatorio, 0% de déficits nuevos — ver report). Clamp [0, 0.5]: en 0, el bound nunca se
# satisface (con `under_buy>0`) y sólo el colchón anti-desperdicio retiene floor (máxima
# corrección). En 0.5: como `under_buy = frac*size` y el bound equivale a
# `frac <= PCT*floor_units/(1-PCT)`, en PCT=0.5 eso es `frac <= floor_units`, SIEMPRE cierto
# (`frac<1<=floor_units` para floor_units>=1) — el AND se reduce EXACTAMENTE a `under_buy <
# over_buy`, idéntico byte a byte al comportamiento pre-fix (no una aproximación).
SKU_FLOOR_MAX_UNDER_PCT = _knob_env_float(
    "MEALFIT_SKU_FLOOR_MAX_UNDER_PCT", 0.10, lambda v: 0.0 <= v <= 0.5)

# [P1-SKU-COVER-HONESTY · 2026-08-02] Umbral de `pkg_cover_ratio` bajo el cual se avisa
# "alcanza ~N de M días — recompra" (mismo formato que P1-CAPPED-STAPLE-HONESTY) si el ítem no
# tiene ya un aviso de cap (`capped_by` manda — no duplicar sufijos, decisión #3). Clamp [0, 1].
#
# [P1-SKU-COVER-HONESTY-R1 · 2026-08-02] Con el bound corregido, cuando SÍ se retiene el floor
# (rama nueva), la cobertura mínima garantizada es exactamente `1 - SKU_FLOOR_MAX_UNDER_PCT`
# (default 0.10 → cover_min=0.90). Si este umbral fuera <= ese mínimo garantizado (ej. el 0.9
# original), la nota queda INALCANZABLE por construcción para cualquier ítem recién resuelto por
# estos 3 sitios — medido: 60.000 casos aleatorios, 0 disparos legítimos con 0.9. Default subido
# a 0.95 para que la nota cubra la banda real 5%-10% de déficit que el bound SÍ permite (entre
# `1-SKU_FLOOR_MAX_UNDER_PCT` y este umbral) sin ser inerte ni volverse ruidosa.
# `test_pkg_cover_note_min_no_es_inalcanzable_por_construccion` ancla `PKG_COVER_NOTE_MIN >
# 1 - SKU_FLOOR_MAX_UNDER_PCT` para que un futuro cambio de default no la vuelva inerte en
# silencio.
PKG_COVER_NOTE_MIN = _knob_env_float(
    "MEALFIT_PKG_COVER_NOTE_MIN", 0.95, lambda v: 0.0 <= v <= 1.0)

# [P1-VEG-BACKFILL-HONESTY · 2026-08-02] Umbral bajo el cual la cantidad FINAL resuelta por
# `apply_smart_market_units` (`base_qty`, gramos) queda por debajo de lo que las recetas piden
# (`text_demand_g`, mismo parse que usa el guard — `expected_sum_from_recipes` threaded desde
# `get_shopping_list_delta`) y NINGÚN cap conocido (`_CAPS_APPLIED_LAST_RUN`) lo explica. Caso
# medido en prod: plan 5f4bb17e, receta "600 g de espárragos" en una cena vs lista semanal
# 583.33 g (103% de la compra semanal en una sola cena) con `capped_by=null` — espárragos no
# vive en ningún dict de cap por categoría (P5-VEG-CAP, P6-*), así que el déficit llegaba mudo.
# Cuando dispara, estampa `capped_by="qty_reconcile_v7"` SINTÉTICO para que el bloque de
# P1-CAPPED-STAPLE-HONESTY (que ya sabe componer el sufijo "alcanza ~N de M días") haga el
# trabajo — no se reimplementa el copy. Clamp (0, 1]: en 1.0 sólo dispara con déficit exacto (no
# realista); no se permite <=0 (compraría 0 y aun así "cubriría").
QTY_SHORTFALL_NOTE_MIN = _knob_env_float(
    "MEALFIT_QTY_SHORTFALL_NOTE_MIN", 0.9, lambda v: 0.0 < v <= 1.0)

# [P1-VEG-BACKFILL-HONESTY · 2026-08-03 · review final] Razón del sello SINTÉTICO, como constante
# y no como literal disperso. La necesita el productor (`apply_smart_market_units`) Y el consumidor
# (`_extract_aggregated_food_dict`, que debe EXCLUIRLA de la sustitución `base_qty ← capped_pre`).
# Dos literales iguales en dos puntas del archivo son exactamente cómo se pierde una exclusión.
#
# ⚠️ POR QUÉ EL SELLO SINTÉTICO NO ESCRIBE `capped_pre` (bug crítico del review final, reproducido
# ejecutando): el guard de coherencia sustituye la cantidad realmente comprada por `capped_pre`
# para los ítems con `capped_by` (P1-COHERENCE-CAPPED-PRE). Eso es correcto para un cap REAL,
# donde `capped_pre` es lo que el AGREGADOR calculó por su cuenta antes del tope — un número
# INDEPENDIENTE del lado esperado, así que si el agregador calcula mal la divergencia sigue
# saliendo. Para el sello sintético `pre_value` ES la demanda de las recetas
# (`expected_sum_from_recipes(..., multiplier=effective_multiplier)`), literalmente la misma
# función y el mismo factor que el lado ESPERADO del guard: el guard acababa comparando el
# esperado contra sí mismo. Medido: drift de magnitud 2× (recetas 2000 g, lista 1000 g) →
# divergencias `[{'delta_pct': 0.5}]` sin el sello y `[]` con él. Y como el guard corre en modo
# `block` por default, ese ítem dejaba de escalar, de reintentar y de aparecer en
# `_shopping_coherence_block_history`. El déficit sintético viaja ahora por claves PROPIAS
# (`shortfall_text_g` / `shortfall_bought_g`), que ningún consumidor del guard lee.
QTY_RECONCILE_SYNTHETIC_REASON = "qty_reconcile_v7"

# ═══════════════════════════════════════════════════════════════
# Helpers para SKU-Aware Sizing (P3)
# ═══════════════════════════════════════════════════════════════
def _find_best_sku(g_total: float, available_sizes_g: list, anti_waste_pct: float = 0.10):
    """Encuentra la combinación óptima de SKUs para minimizar desperdicio.

    Estrategias (en orden de prioridad):
      1. Single-SKU: paquete más pequeño que cubre la necesidad (≤20% waste, ≤2x tamaño)
      2. Best-Fit Multi: prueba TODOS los tamaños, elige el que minimiza desperdicio
      3. Fallback bulk: si TODOS los sizes son << g_total (necesidad >> SKU más
         grande, e.g. plan mensual × 2 personas con yogurt: 3733g vs SKU max
         453g), usar el size MÁS GRANDE con `ceil(g_total / size)` count.

    Returns: (count, size_g) — cuántos paquetes de qué tamaño
    """
    import math
    sizes = sorted([float(s) for s in available_sizes_g])  # ascendente

    # Estrategia 1: Un solo paquete que cubre la necesidad
    # Tolerancia muy ajustada (5%) para obligar escalar visualmente cuando aumentan personas.
    SINGLE_PKG_TOLERANCE = 0.05
    for size in sizes:
        if size >= g_total and size <= g_total * 2:
            waste_pct = (size - g_total) / size
            if waste_pct <= SINGLE_PKG_TOLERANCE:
                return 1, size

    # Estrategia 2: Prueba cada tamaño disponible, elige el mejor
    # Criterio: mínimo desperdicio con mínimo conteo de paquetes
    best_result = None
    best_waste = float('inf')

    for size in sizes:
        if size < g_total * 0.15:  # Skip tamaños ridículamente pequeños
            continue
        raw_count = g_total / size
        floor_count = math.floor(raw_count)
        frac = raw_count - floor_count

        # [2026-05-06 SKU-OVERSHOOT-FIX] Mismo principio que el standard path
        # en `apply_smart_market_units`: si el under-buy del floor es ABSOLUTAMENTE
        # menor que el over-buy del ceil, preferir floor aunque exceda el
        # `anti_waste_pct` umbral. Evita que items con `g_total ≈ container`
        # (ej. Pan integral 600g vs container 567g) salten al doble por
        # estrechez del threshold cuando el under-buy es marginal.
        if floor_count >= 1:
            under_buy = g_total - (floor_count * size)
            over_buy = ((floor_count + 1) * size) - g_total
            # [P1-SKU-COVER-HONESTY · 2026-08-02] `under_buy < over_buy` retenía el floor con
            # under-buy de hasta 50% del total sin aviso. Acotado a `SKU_FLOOR_MAX_UNDER_PCT`
            # del total — la rama `frac <= anti_waste_pct` (colchón anti-desperdicio) intacta.
            # [P1-SKU-COVER-HONESTY-R1 · 2026-08-02] El bound puro (sin `and under_buy < over_buy`)
            # NO es uniformemente más estricto: escala con `floor_count`, así que para conteos
            # altos (>=9 con el default 10%) es vacuo y PERMITE un déficit que el criterio viejo
            # no permitía. Exigir ambas condiciones garantiza que el resultado es subconjunto del
            # criterio viejo — nunca peor, sólo más estricto.
            if frac <= anti_waste_pct or (under_buy <= g_total * SKU_FLOOR_MAX_UNDER_PCT and under_buy < over_buy):
                count = floor_count
                total_g = count * size
                waste = max(0, g_total - total_g)
            else:
                count = floor_count + 1
                total_g = count * size
                waste = total_g - g_total
        else:
            count = max(1, math.ceil(raw_count))
            total_g = count * size
            waste = total_g - g_total

        waste_score = waste / g_total if g_total > 0 else 0
        # Penalizar conteos altos exponencialmente: 1 paquete siempre > N paquetes
        # count^1.5: 1→0.04, 2→0.11, 3→0.21, 4→0.32, 5→0.45
        score = waste_score + (count ** 1.5 * 0.04)

        if score < best_waste:
            best_waste = score
            best_result = (count, size)

    if best_result is not None:
        return best_result

    # ── Estrategia 3: Fallback bulk ──
    # Antes: `return (1, sizes[0])`. Catastrófico cuando ALL sizes quedaron
    # filtrados por el guard `size < g_total * 0.15`: el usuario necesita
    # mucho más que el SKU más grande, y devolver "1 paquete del MÁS
    # PEQUEÑO" produce under-buy del 90-99%. Bug observable: yogurt griego
    # `available_sizes=[150, 227, 453]` con g_total=3733g (mensual × 2
    # personas) → el guard descartaba los 3 sizes (3733 × 0.15 = 560 > 453)
    # → fallback retornaba (1, 150) → PDF mostraba "1 pote (150g)" cuando
    # el usuario necesita ~25 potes (≈3.7 kg). Mismo modo de fallo aplica a
    # habichuelas, queso blanco, y cualquier item cuyo SKU max < 6.67× la
    # necesidad real. Ahora usamos el size MÁS GRANDE con `ceil(g_total /
    # size)` — matemática correcta, ningún under-buy silencioso.
    largest_size = sizes[-1]
    fallback_count = max(1, math.ceil(g_total / largest_size))
    return fallback_count, largest_size


def _select_market_package(g_total: float, market_packages, anti_waste_pct: float = 0.02):
    """[P1-PKG-DURATION-PRICING · 2026-06-22] Elige el envase REAL (tamaño + precio) para
    cubrir `g_total` gramos y devuelve también el PRECIO del tamaño elegido.

    `anti_waste_pct` por defecto 0.02 = mismo umbral estricto que el path SKU legacy de
    `apply_smart_market_units` (ANTI_WASTE_THRESHOLD local), para que la selección sea
    idéntica a la que ya producía buenos tamaños por duración.

    Reutiliza la MISMA heurística de SKU (`_find_best_sku`: mínimo desperdicio + mínimo
    conteo) sobre los tamaños declarados en `market_packages`, de modo que la selección
    sigue siendo duración-aware (g_total ya viene escalado por `base_duration_scale`):
    7 días → paquete chico, 30 días → paquete grande. La diferencia vs el path legacy es
    que ahora cada tamaño trae su PRECIO real, cerrando el sobrecobro por precio plano
    (ej. arroz 30 días: 10 lb × 55 = RD$550 → 1 paquete 10 lb = RD$327, descuento por
    volumen verificado in-store).

    `market_packages`: lista [{"grams": N, "price": RD$, "label": "..."}].
    Returns dict {count, grams, price, label} o None si no hay datos usables.
    Tooltip-anchor: P1-PKG-DURATION-PRICING.
    """
    if not market_packages or not isinstance(market_packages, list):
        return None
    pkgs = []
    for p in market_packages:
        if not isinstance(p, dict):
            continue
        try:
            g = float(p.get("grams"))
            pr = float(p.get("price"))
        except (TypeError, ValueError):
            continue
        if g > 0 and pr >= 0:
            # [P1-EGG-CARTON-SIZES · 2026-06-22] `unit` opcional por envase: la FORMA del
            # paquete (lata/paquete/sobre/pote) cuando difiere del market_container genérico.
            # Cierra el wart "1 lata (800 g seco)" cuando un mismo ítem (habichuelas) se vende
            # en lata Y en bolsa. Sin `unit` → fallback a db_container (comportamiento previo).
            # [P1-BRAND-DEFAULT-PRESELECTED · 2026-07-06] `id` opcional (producto del súper)
            # se arrastra al envase elegido → item.brand_product_id → picker pre-selecciona.
            pkgs.append((g, pr, str(p.get("label") or ""), str(p.get("unit") or ""), str(p.get("id") or "")))
    if not pkgs:
        return None
    sizes = [g for (g, _, _, _, _) in pkgs]
    if g_total <= 0:
        # Sin necesidad calculable: 1 unidad del envase más pequeño.
        g_sel, pr_sel, lbl_sel, unit_sel, id_sel = min(pkgs, key=lambda t: t[0])
        return {"count": 1, "grams": g_sel, "price": pr_sel, "label": lbl_sel, "unit": unit_sel, "id": id_sel}
    if len(sizes) > 1:
        # [P1-PKG-COST-OPTIMAL · 2026-06-22] Elegir por COSTO total mínimo, no por
        # desperdicio+conteo (el path legacy `_find_best_sku` penaliza el nº de paquetes,
        # lo que hacía comprar 2 four-packs de yogurt griego RD$730 cuando 6 potes sueltos
        # RD$600 —exacto— eran más baratos). Para cada tamaño contamos con el MISMO floor
        # anti-desperdicio que _find_best_sku, acotado por `SKU_FLOOR_MAX_UNDER_PCT`
        # (P1-SKU-COVER-HONESTY), y tomamos el de MENOR costo total; desempate: menos
        # desperdicio, menos paquetes, envase más grande.
        # [P1-SKU-COVER-HONESTY-R1 · 2026-08-02] El comentario previo decía "no elige 5 lb de
        # arroz para una necesidad de ~2 lb" — eso era cierto SÓLO mientras el floor permitía el
        # under-buy silencioso de 907g/1050g (13,6% corto). Con el bound corregido, 1050g SÍ
        # elige 1×5lb (RD$235, más barato Y cubre) en vez de 1×2lb (RD$165, corto) o 2×2lb
        # (RD$330) — ver `test_arroz_unchanged`→renombrado en `test_p1_pkg_cost_optimal.py`.
        # Sigue verificado: arroz 15d→5lb/30d→10lb y habichuelas mantienen su selección; yogurt
        # 900g → 6 potes (no 2 four-packs). Tooltip-anchor: P1-PKG-COST-OPTIMAL.
        best_key = None
        chosen = None
        for (g, pr, lbl, unit, pid) in pkgs:
            raw = g_total / g
            floor_c = math.floor(raw)
            if floor_c >= 1:
                under = g_total - floor_c * g
                over = (floor_c + 1) * g - g_total
                frac = raw - floor_c
                # [P1-SKU-COVER-HONESTY · 2026-08-02] `under < over` (min-costo) FAVORECÍA el
                # floor con under-buy de hasta 50% sin aviso. Mismo bound que los otros 2 sitios.
                # [P1-SKU-COVER-HONESTY-R1 · 2026-08-02] `and under < over`: el bound puro solo
                # (sin este AND) es vacuo para floor_c>=9 y permitiría un déficit NUEVO que el
                # criterio viejo no permitía — ver knob doc arriba.
                count_c = floor_c if (frac <= anti_waste_pct or (under <= g_total * SKU_FLOOR_MAX_UNDER_PCT and under < over)) else floor_c + 1
            else:
                count_c = 1
            cost_c = count_c * pr
            waste = max(0.0, count_c * g - g_total)
            key = (round(cost_c, 4), round(waste, 4), count_c, -g)
            if best_key is None or key < best_key:
                best_key = key
                chosen = {"count": int(count_c), "grams": g, "price": pr, "label": lbl, "unit": unit, "id": pid}
        return chosen
    else:
        size_g = sizes[0]
        count = max(1, math.ceil(g_total / size_g))
        g_sel, pr_sel, lbl_sel, unit_sel, id_sel = min(pkgs, key=lambda t: abs(t[0] - size_g))
        return {"count": int(count), "grams": g_sel, "price": pr_sel, "label": lbl_sel, "unit": unit_sel, "id": id_sel}


def _choose_egg_carton(total_eggs: float, egg_packages):
    """[P1-EGG-CARTON-SIZES · 2026-06-22] Elige el CARTÓN de huevos cost-óptimo para cubrir
    `total_eggs`, sobre los tamaños declarados en `market_packages` con campo `units`
    (no `grams` — los huevos se cuentan por unidad discreta, no por peso).

    Los huevos NO pasan por el weight-path de apply_smart_market_units (se consolidan a
    'cartón (N uds.)' antes), así que el path market_packages por gramos no aplica. Este
    selector es el equivalente para huevos: elige el tamaño de cartón con MENOR costo total
    (count × precio_cartón), prefiriendo menos cartones y luego el cartón más grande en empate.

    Ejemplos con [{units:20,price:200},{units:30,price:295}]:
      14 huevos → cartón 20 (1×200=200, vs 1×295) — ahorra RD$95 en planes de 7 días.
      22 huevos → cartón 30 (1×295, vs 2×20=400).
      56 huevos → 2×cartón 30 (590, vs 3×20=600).

    `egg_packages`: lista [{"units": N, "price": RD$, "label": "..."}].
    Returns dict {units, count, price, label} o None si no hay datos usables.
    Tooltip-anchor: P1-EGG-CARTON-SIZES.
    """
    if not egg_packages or not isinstance(egg_packages, list):
        return None
    cartons = []
    for p in egg_packages:
        if not isinstance(p, dict):
            continue
        try:
            u = int(float(p.get("units")))
            pr = float(p.get("price"))
        except (TypeError, ValueError):
            continue
        if u > 0 and pr >= 0:
            cartons.append((u, pr, str(p.get("label") or "")))
    if not cartons:
        return None
    need = max(1.0, float(total_eggs))
    best = None
    for u, pr, lbl in cartons:
        count = max(1, math.ceil(need / u))
        cost = count * pr
        key = (cost, count, -u)  # menor costo, luego menos cartones, luego cartón más grande
        if best is None or key < best[0]:
            best = (key, u, count, pr, lbl)
    return {"units": best[1], "count": best[2], "price": best[3], "label": best[4]}


# ============================================================
# [P1-SUPERMARKET-COSTING · 2026-07-02] Marca preferida del súper → costeo real.
# ------------------------------------------------------------
# Fase 3 de la conexión Supermercado RD ↔ lista de compras: si el usuario eligió
# una marca/presentación en el panel "Marcas del súper" (tabla
# `user_brand_preferences`, P1-SUPERMARKET-PREFS), el costeo de ese ítem usa ESE
# envase (tamaño + precio de `supermarket_products`) en lugar del market_package
# genérico. Implementación mínimamente invasiva: overlay del `market_packages`
# del master_item con UNA entrada (la preferida) justo antes de
# `apply_smart_market_units` → `_select_market_package` compra
# ceil(g_total/grams) del envase elegido y `_cost_from_market` costea con su
# precio real. El efecto en cantidades es de la MISMA clase que el redondeo por
# envase que ya existe (P1-PKG) — el coherence guard lo tolera igual.
#
# Reversible sin redeploy: MEALFIT_BRAND_PREF_COSTING=false.
# Fail-open SIEMPRE: cualquier error (DB caída, presentación no parseable, food
# sin match) → costeo estándar sin la preferencia.
# Tooltip-anchor: P1-SUPERMARKET-COSTING. Test: test_p1_supermarket_costing.py.
# ============================================================

def _brand_pref_costing_enabled() -> bool:
    # [P2-1] `_env_bool` registra en `_KNOBS_REGISTRY`.
    return _knob_env_bool("MEALFIT_BRAND_PREF_COSTING", True)


def _norm_pref_food(value) -> str:
    """Normalización SIMÉTRICA a routers/supermarket.py::_norm_food y al
    foodKeyOf del frontend (minúsculas + sin acentos + espacios colapsados).
    Las claves de `user_brand_preferences.food_key` nacen de _norm_food —
    mantener las tres en sync."""
    import unicodedata
    s = unicodedata.normalize("NFD", str(value or "").strip().lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(s.split())


def _singular_pref_key(s: str) -> str:
    """Heurística ligera es-DO (misma que routers/supermarket.py::_singular)."""
    if len(s) > 4 and s.endswith("es"):
        return s[:-2]
    if len(s) > 3 and s.endswith("s"):
        return s[:-1]
    return s


# Tamaño del envase desde el texto libre de `presentation`. NOTA: la "L" suelta
# es AMBIGUA en el catálogo (PDF usa "L" para libra en produce Y litro en leche)
# → NO se parsea (fail-open al costeo estándar). Los líquidos del catálogo con
# marca usan "Lt"/"Ml" explícitos. El lookbehind `(?<![/\d.,])` evita que la
# notación de carnicería "80/20 Lb" / "96/4 Lb" (ratio magro/grasa) se lea como
# "20 Lb"/"4 Lb" — esos caen al fallback venta-por-libra (453.59 g).
_PRES_SIZE_RX = re.compile(
    r"(?<![/\d.,])(\d+(?:[.,]\d+)?)\s*(kg|grs?|g|onz|oz|lbs?|libras?|ml|lts?)\b",
    re.IGNORECASE,
)
_PRES_UNIT_G = {
    "kg": 1000.0, "g": 1.0, "gr": 1.0, "grs": 1.0,
    "oz": 28.3495, "onz": 28.3495,
    "lb": 453.592, "lbs": 453.592, "libra": 453.592, "libras": 453.592,
    # Líquidos ≈ densidad 1 (leches/aceites 0.9-1.03 — margen aceptable de costeo).
    "ml": 1.0, "lt": 1000.0, "lts": 1000.0,
}
_PRES_CONTAINER_WORDS = {
    "funda", "lata", "paquete", "botella", "frasco", "tarro", "pote", "caja",
    "carton", "brik", "sobre", "bandeja", "clamshell", "cubo", "barra", "tubo",
    "pieza", "malla", "tetra",
}


def _parse_presentation_grams(presentation) -> float | None:
    """'Funda 800 gr'→800 · 'Lata 15 Oz'→425.2 · '1.47 Lb'→666.8 · 'Brik 290 Ml'→290
    · 'Botella 1 Lt'→1000 · 'Criolla Lb' (venta por libra)→453.6 · resto→None."""
    text = str(presentation or "")
    if not text.strip():
        return None
    m = _PRES_SIZE_RX.search(text)
    if m:
        try:
            qty = float(m.group(1).replace(",", "."))
        except ValueError:
            return None
        base = _PRES_UNIT_G.get(m.group(2).lower())
        if base and qty > 0:
            grams = qty * base
            # Sanity: 1 g – 50 kg (fuera de eso = parse errado, mejor fail-open).
            if 1.0 <= grams <= 50000.0:
                return grams
        return None
    # [P1-BRAND-SIZE-FILTER · 2026-07-06] "L" suelta TRAS ENVASE SÓLIDO = libra.
    # La "L" sigue siendo ambigua en general (libra en produce / litro en leche),
    # pero los líquidos del catálogo usan "Lt"/"Ml" explícitos (ver nota arriba de
    # _PRES_SIZE_RX) — un "Paquete 2L"/"Funda 5L" de staple seco es LIBRA con alta
    # confianza. Cierra el gap donde los genéricos del sync (arroz "Paquete 2L")
    # perdían el overlay de costeo Y quedaban fuera del filtro por tamaño del
    # picker de marcas. "Botella 2L" sigue fail-open (litro probable).
    m2 = re.search(r"(?<![/\d.,])(\d+(?:[.,]\d+)?)\s*l\b", text, re.IGNORECASE)
    if m2:
        first = _norm_pref_food(text).split(" ")[0] if text.strip() else ""
        if first in ("paquete", "funda", "saco", "sobre", "caja"):
            try:
                qty = float(m2.group(1).replace(",", "."))
            except ValueError:
                return None
            grams = qty * 453.592
            if 1.0 <= grams <= 50000.0:
                return grams
        return None
    # "Lb"/"Libra" sin número = venta por libra (carnicería/produce).
    if re.search(r"\b(lb|libra)\b", text, re.IGNORECASE):
        return 453.592
    return None


def _pref_container_word(presentation) -> str:
    """Primera palabra de la presentación si es un envase conocido; 'libra' para
    venta-por-libra; fallback 'paquete'."""
    text = _norm_pref_food(presentation)
    first = text.split(" ")[0] if text else ""
    if first in _PRES_CONTAINER_WORDS:
        return first
    if re.search(r"\b(lb|libra)\b", text) and not _PRES_SIZE_RX.search(text):
        return "libra"
    return "paquete"


def fetch_brand_pref_packages(user_id: str) -> dict:
    """Preferencias de marca del usuario como packages costeables:
    {food_key_normalizado: {"grams", "price", "label", "unit"}}.
    Doble clave: el `food_key` persistido Y el food_name del producto (por si
    difieren). Solo productos activos con precio y presentación parseable.
    Fail-open: {} ante cualquier error."""
    if not user_id:
        return {}
    try:
        rows = execute_sql_query(
            """
            SELECT sp.id::text AS id, p.food_key, sp.food_name, sp.brand, sp.presentation,
                   sp.price_rd::float8 AS price_rd,
                   sp.size_grams::float8 AS size_grams
            FROM public.user_brand_preferences p
            JOIN public.supermarket_products sp ON sp.id = p.product_id
            WHERE p.user_id = %s AND sp.active AND sp.price_rd IS NOT NULL
            """,
            (user_id,),
            fetch_all=True,
        ) or []
    except Exception as exc:
        logging.warning(f"⚠️ [P1-SUPERMARKET-COSTING] fetch prefs falló (fail-open): {exc}")
        return {}
    out: dict = {}
    for r in rows:
        pkg = _pkg_from_product_row(r)
        if pkg is None:
            logging.info(
                f"🏷️ [P1-SUPERMARKET-COSTING] pref '{r.get('food_key')}' sin tamaño/precio usable "
                f"('{r.get('presentation')}', size_grams NULL) → costeo estándar para ese ítem. "
                f"Lever: poblar size_grams en /supermercado (P2-BRANDPREF-SIZE-COLUMN)."
            )
            continue
        for key in {_norm_pref_food(r.get("food_key")), _norm_pref_food(r.get("food_name"))}:
            if key:
                out.setdefault(key, pkg)
    return out


def _pkg_from_product_row(r) -> dict | None:
    """[P1-BRAND-LIST-VISIBILITY · 2026-07-06] Producto del súper → package costeable
    `{"grams","price","label","unit"}` o None (fail-open del ítem). SSOT compartido entre
    `fetch_brand_pref_packages` (marca ELEGIDA por el usuario) y
    `fetch_brand_default_packages` (marcas default). El `label` "tamaño · Marca"
    (ej. "800 gr · La Sanjuanera") es lo que el display del ítem enseña entre
    paréntesis — la MARCA visible en lista y PDF.

    [P2-BRANDPREF-SIZE-COLUMN · 2026-07-02] `size_grams` explícito (admin UI) es la fuente
    AUTORITATIVA del tamaño del envase; el parser del texto libre `presentation` queda como
    fallback (la "L" suelta es ambigua libra/litro → antes esos productos PERDÍAN el overlay)."""
    try:
        grams = float(r.get("size_grams") or 0) or None
    except (TypeError, ValueError):
        grams = None
    if grams is not None and not (1.0 <= grams <= 50000.0):
        grams = None  # sanity fuera de rango → fallback al parser
    if grams is None:
        grams = _parse_presentation_grams(r.get("presentation"))
    if not grams:
        return None
    try:
        price = float(r.get("price_rd"))
    except (TypeError, ValueError):
        return None
    if price <= 0:
        return None
    pres = str(r.get("presentation") or "").strip()
    # [P1-BRAND-GENERIC-LABEL · 2026-07-06] Producto sin marca en el catálogo =
    # la opción "Genérico" (mismo fallback que enseña el picker de marcas). El
    # owner pidió que la lista SIEMPRE diga la marca que está usando — un label
    # solo-tamaño ("2 lb") era indistinguible del costeo sin marcas.
    brand = (r.get("brand") or "").strip() or "Genérico"
    # Label para el display "(...)": tamaño + marca — ej. "800 gr · La Sanjuanera".
    size_part = pres.split(" ", 1)[1] if (
        pres and _norm_pref_food(pres).split(" ")[0] in _PRES_CONTAINER_WORDS and " " in pres
    ) else pres
    _first_word = _norm_pref_food(pres).split(" ")[0] if pres else ""
    if _first_word in ("paquete", "funda", "saco", "sobre", "caja"):
        # [P1-BRAND-DEFAULT-GUARDS · 2026-07-06] La "L" que el parser leyó como
        # LIBRA (envase sólido) no puede quedarse como "2 L" en el display — se
        # leería litros. Normalizar el label a "2 lb".
        size_part = re.sub(r"(?i)(?<![/\d.,])(\d+(?:[.,]\d+)?)\s*l\b", r"\1 lb", size_part)
    label = f"{size_part} · {brand}" if brand else size_part
    # [P1-BRAND-DEFAULT-GUARDS · 2026-07-06] Venta-por-libra (presentación "Lb"
    # sin tamaño explícito): marcar el pkg. El DEFAULT lo excluye — es mostrador
    # fresco (carnes/produce, casi siempre Genérico), forzarlo como "envase" de
    # 1 lb rompe los conteos nativos (Uds/Cabezas/¼ lb) y sobre-compra (guayaba
    # ¼ lb → 1 lb, chivo ¼ lb RD$75 → 1 lb RD$299). La PREFERENCIA manual sí lo
    # respeta (elección explícita del usuario — comportamiento pre-existente).
    _explicit_size = bool(r.get("size_grams")) or bool(_PRES_SIZE_RX.search(pres)) or (
        _first_word in ("paquete", "funda", "saco", "sobre", "caja")
        and re.search(r"(?<![/\d.,])\d+(?:[.,]\d+)?\s*l\b", pres, re.IGNORECASE)
    )
    return {
        "grams": grams,
        "price": price,
        "label": label.strip(" ·"),
        "unit": _pref_container_word(pres),
        "per_lb": not _explicit_size,
        # [P1-BRAND-DEFAULT-PRESELECTED · 2026-07-06] identidad del producto del
        # súper — viaja al ítem como `brand_product_id` para que el picker
        # pre-seleccione la marca que la lista está usando.
        "id": str(r.get("id") or "") or None,
    }


# ============================================================
# [P1-BRAND-LIST-VISIBILITY · 2026-07-06] Marcas default del súper → lista/PDF.
# ------------------------------------------------------------
# Pedido del owner: (a) la lista debe ENSEÑAR la marca de cada alimento, no solo
# el envase genérico; (b) por default la IA usa la marca MÁS BARATA; (c) si el
# usuario elige otra marca en "Marcas del súper", esa gana y el total sube.
#
# Implementación: para los ítems SIN preferencia del usuario, el overlay del
# aggregator reemplaza los `market_packages` genéricos del master con los
# productos REALES (marca + presentación + precio vivo) de `supermarket_products`.
# `_select_market_package` ya es costo-óptimo (P1-PKG-COST-OPTIMAL) → elige la
# marca/tamaño más barato para la necesidad del ciclo, y su `label` ("2 lb · La
# Garza") fluye a `display_qty` → panel por pasillo + PDF enseñan la marca.
# La preferencia manual del usuario (P1-SUPERMARKET-COSTING) SIEMPRE gana.
#
# Las marcas default son GLOBALES (no per-user) → cache in-process con TTL para
# no re-consultar ~2k productos en cada recalc/cron.
# Reversible sin redeploy: MEALFIT_BRAND_DEFAULT_PACKAGES=false.
# Fail-open SIEMPRE: error/DB caída → {} (costeo estándar sin marcas).
# Tooltip-anchor: P1-BRAND-LIST-VISIBILITY. Test: test_p1_brand_list_visibility.py.
# ============================================================

def _brand_default_packages_enabled() -> bool:
    return _knob_env_bool("MEALFIT_BRAND_DEFAULT_PACKAGES", True)


_BRAND_DEFAULTS_TTL_S = 600.0  # 10 min — precios del súper cambian a ritmo de cron, no de request
_BRAND_DEFAULTS_MAX_PER_FOOD = 12  # techo de variantes por alimento que ve el selector
_brand_defaults_cache: dict = {"at": 0.0, "data": None}


def fetch_brand_default_packages() -> dict:
    """Packages costeables de TODOS los productos activos del súper, agrupados por
    alimento: {food_key_normalizado: [pkg, ...]} con cada lista ordenada por precio
    ascendente (la más barata primero) y capada a `_BRAND_DEFAULTS_MAX_PER_FOOD`.
    Cache global TTL 10 min. Fail-open: {} ante cualquier error."""
    import time as _time
    now = _time.monotonic()
    cached = _brand_defaults_cache.get("data")
    if cached is not None and (now - _brand_defaults_cache.get("at", 0.0)) < _BRAND_DEFAULTS_TTL_S:
        return cached
    try:
        rows = execute_sql_query(
            """
            SELECT sp.id::text AS id, sp.food_name, sp.brand, sp.presentation,
                   sp.price_rd::float8 AS price_rd,
                   sp.size_grams::float8 AS size_grams
            FROM public.supermarket_products sp
            WHERE sp.active AND sp.price_rd IS NOT NULL
            """,
            (),
            fetch_all=True,
        ) or []
    except Exception as exc:
        logging.warning(f"⚠️ [P1-BRAND-LIST-VISIBILITY] fetch defaults falló (fail-open): {exc}")
        return {}
    out: dict = {}
    _canned_keys: dict = {}
    for r in rows:
        pkg = _pkg_from_product_row(r)
        if pkg is None:
            continue  # sin tamaño/precio usable → ese producto no participa del default
        if pkg.get("per_lb"):
            # [P1-BRAND-DEFAULT-GUARDS] mostrador fresco (venta por libra) fuera
            # del default — ver nota en _pkg_from_product_row.
            continue
        key = _norm_pref_food(r.get("food_name"))
        if key:
            # [P1-FRESH-OVER-CANNED-DEFAULT] la forma se detecta en la PRESENTATION cruda
            # (el label compuesto la recorta: "Lata Entero 185 gr" → "Entero 185 gr").
            # `_shelf` = CUALQUIER envase de estantería (lata/frasco/pote/tetra): solo si
            # existe una forma SIN envase de estantería (bandeja/paquete fresco) el alimento
            # califica como "fresco disponible" — sin esto, el FRASCO de atún Tonnino
            # contaba como fresco y filtraba todas las latas baratas (98→469 RD$, cazado
            # en la verificación).
            _pres = str(r.get("presentation") or "").lower()
            pkg["_canned"] = ("lata" in _pres) or ("enlatad" in _pres)
            pkg["_shelf"] = any(w in _pres for w in ("lata", "frasco", "pote", "tetra",
                                                     "enlatad", "conserva"))
            out.setdefault(key, []).append(pkg)
    for key, pkgs in out.items():
        # [P1-FRESH-OVER-CANNED-DEFAULT · 2026-07-28] "Champiñones frescos en láminas" en la
        # receta pero la lista compraba "Lata Trozos y Tallos · Roland" (caso vivo, plan
        # ab2b0a16): el selector es costo-óptimo TOTAL (ordenar no le manda) y elegía lata
        # sobre la bandeja FRESCA por RD$0.02/g. Cuando un alimento tiene formas MIXTAS
        # (fresca y lata), la lata SALE del set default — honestidad de forma sobre
        # centavos. La elección MANUAL del usuario va por product_id directo y sigue
        # ganando (puede elegir la lata). Alimentos solo-lata (atún, calamar) idénticos.
        _has_truly_fresh = any(not p.get("_shelf") for p in pkgs)
        if _has_truly_fresh:
            pkgs[:] = [p for p in pkgs if not p.get("_canned")]
        for p in pkgs:
            p.pop("_canned", None)
            p.pop("_shelf", None)
        pkgs.sort(key=lambda p: (p["price"], p["grams"]))
        del pkgs[_BRAND_DEFAULTS_MAX_PER_FOOD:]
    _brand_defaults_cache["data"] = out
    _brand_defaults_cache["at"] = now
    return out


# [P1-BRAND-DEFAULT-GUARDS · 2026-07-06] Guards del matching de DEFAULTS (la
# preferencia manual sigue usando la escalera completa de _resolve_brand_pref):
#  - Hierbas frescas se venden por MAZO (path nativo) — un default las convertía
#    en frasco de perejil MOLIDO Badia RD$215 o semillas de cilantro (verificado
#    contra plan vivo ff673061). Jamás overlay.
#  - Contención SOLO food⊆nombre ("arroz blanco premium" → "arroz blanco"). La
#    dirección inversa (nombre⊆food) hacía que "Yogurt" agarrara "Yogurt de
#    cabra" en frascos 4 Oz → 8 frascos RD$880 vs pote 1.96 kg RD$220.
#  - Variantes con MODIFICADOR ajeno al nombre del ítem fuera ("Maní" ≠ "Con
#    Pasas"; "Semillas de girasol" sí admite "semillas").
_BRAND_DEFAULT_HERB_RX = re.compile(
    r"\b(cilantro|cilantrico|puerro|perejil|menta|albahaca|romero|verdura|verdurita|recao|eneldo)\b"
)
_BRAND_DEFAULT_MODIFIER_TOKENS = (
    "molido", "molida", "semilla", "semillas", "con pasas", "con sal", "con azucar",
    "azucarado", "endulzado", "sazonado", "adobado", "ahumado", "frito", "frita",
    # [verificado vs plan vivo] "Maní Japonés" (recubierto) no es maní a secas;
    # "Baby" (zanahoria baby 12 Oz RD$175 vs ½ lb RD$14) es specialty. El
    # catálogo mezcla descriptores en inglés ("Roasted Salted") — "salted" es
    # el "con sal" inglés (sodio); "honey" cubre honey-roasted/glazed.
    "japones", "japonesa", "baby", "salted", "honey",
    # [plan cd4ae3c3] "Aceite de oliva" → "Blend Girasol 750 Ml · Wala": una
    # MEZCLA con girasol no es aceite de oliva (perfil graso distinto — el plan
    # cuenta MUFA de oliva). blend/mezcla/girasol fuera del default salvo que el
    # nombre del ítem los pida.
    "blend", "mezcla", "girasol",
)


def _resolve_brand_default(name: str, defaults: dict):
    """Packages default para un ítem del plan, o None. Más conservador que
    `_resolve_brand_pref` — un default equivocado degrada la lista solo; una
    preferencia la eligió el usuario a sabiendas."""
    if not defaults:
        return None
    key = _norm_pref_food(name)
    if not key or _BRAND_DEFAULT_HERB_RX.search(key):
        return None
    pkgs = None
    for probe in (key, _singular_pref_key(key)):
        if probe in defaults:
            pkgs = defaults[probe]
            break
    if pkgs is None and len(key) >= 4:
        padded_name = f" {key} "
        best_k = None
        for k in defaults:
            if len(k) >= 4 and f" {k} " in padded_name:
                # [P1-BRAND-PREF-PREP-DISTINCT] base↔preparación nunca cross-matchean
                if _brand_prep_distinct_conflict(key, k):
                    continue
                if best_k is None or len(k) > len(best_k):
                    best_k = k
        if best_k:
            pkgs = defaults[best_k]
    if not pkgs:
        return None
    # [P2-LEGUME-NO-LATA-DEFAULT · 2026-07-06] Legumbres: el aggregator convierte
    # las cantidades COCIDAS de la receta a SECAS (yield 0.35×, P2-PDF-1) porque
    # el SKU histórico es seco — un default en LATA (producto ya cocido) hereda
    # esa necesidad seca y sub-compra ~3× ("Habichuelas rojas: 1 lata 15 Oz"
    # para ~10 porciones del ciclo, plan cd4ae3c3). Latas fuera del default de
    # legumbres; la funda seca del master sigue siendo la base coherente.
    _is_legume = bool(re.search(
        r"\b(habichuela|frijol|lenteja|garbanzo|guandul|arveja)", key))
    # [P2-SPICE-MOLIDO-DEFAULT · 2026-07-06] (review #13) Especias: la forma MOLIDA es
    # la canónica de cocina (la receta pide "comino molido") — excluirla por el token
    # 'molido' dejaba como default "Comino Entero 1 Oz" (el entero no sustituye al
    # molido sin molinillo). Para especias: 'molido/molida' deja de ser modifier, y
    # 'entero/entera' pasa a descartarse cuando existe alternativa (fail-open si solo
    # hay entero).
    _is_spice = bool(re.search(
        r"\b(comino|oregano|pimienta|canela|curcuma|jengibre|nuez moscada|clavo)\b", key))
    _mod_toks = _BRAND_DEFAULT_MODIFIER_TOKENS if not _is_spice else tuple(
        t for t in _BRAND_DEFAULT_MODIFIER_TOKENS if t not in ("molido", "molida"))
    out = []
    for p in pkgs:
        lbl = _norm_pref_food(p.get("label"))
        if any(t in lbl and t not in key for t in _mod_toks):
            continue
        if _is_legume and str(p.get("unit") or "").strip().lower() == "lata":
            continue
        out.append(p)
    if _is_spice and out:
        _sin_entero = [p for p in out
                       if not re.search(r"\benter[oa]s?\b", _norm_pref_food(p.get("label")))]
        if _sin_entero:
            out = _sin_entero
    return out or None


# [P1-BRAND-PREF-PREP-DISTINCT · 2026-07-10] Una PREPARACIÓN ("<prep> de <X>") es un producto
# DISTINTO de su alimento base <X> (lección P1-NUT-BUTTER-DISTINCT / P1-PREP-COLLAPSE-GUARD,
# ahora en el brand matcher): la contención bidireccional aplicaba la preferencia Jif Cremosa
# de 'Mantequilla de maní' al ítem 'Maní' (maní entero) — plan vivo 6d742f23 compraba crema de
# maní (RD$295) en vez del pote de maní tostado (RD$185). Mismo vector: manzana↔vinagre de
# manzana, trigo↔harina de trigo, coco↔leche/crema de coco.
_BRAND_PREP_HEADS_RX = re.compile(
    r"^(mantequilla|crema|harina|leche|aceite|salsa|jugo|pasta|tortilla|pan|vinagre|dulce|mermelada)\s+de\s+")


def _brand_prep_distinct_conflict(a: str, b: str) -> bool:
    """True si `a` y `b` (keys ya normalizadas) son base↔preparación del mismo alimento
    ("mani" vs "mantequilla de mani") → JAMÁS cross-match de marca en ninguna dirección.
    tooltip-anchor: P1-BRAND-PREF-PREP-DISTINCT"""
    for shorter, longer in ((a, b), (b, a)):
        m = _BRAND_PREP_HEADS_RX.match(longer)
        if m and f" {shorter} " in f" {longer[m.end():]} ":
            return True
    return False


def _resolve_brand_pref(name: str, prefs: dict):
    """Resuelve el nombre de un ítem del plan contra las preferencias — misma
    escalera que POST /api/supermarket/match: exacto → singular → contención
    word-boundary bidireccional (padding con espacios: 'sal' NO matchea
    'salsa'). Empate → la clave más larga (más específica).
    [P1-BRAND-PREF-PREP-DISTINCT] base↔preparación nunca cross-matchean."""
    if not prefs:
        return None
    key = _norm_pref_food(name)
    if not key:
        return None
    for probe in (key, _singular_pref_key(key)):
        if probe in prefs:
            return prefs[probe]
    if len(key) < 4:
        return None
    padded = f" {key} "
    best_k = None
    for k in prefs:
        if len(k) >= 4 and (f" {k} " in padded or f" {key} " in f" {k} "):
            if _brand_prep_distinct_conflict(key, k):
                continue  # mantequilla de maní ≠ maní (producto derivado)
            if best_k is None or len(k) > len(best_k):
                best_k = k
    return prefs[best_k] if best_k else None


# ============================================================
# [P1-BUDGET-COST-SSOT · 2026-07-02] Resumen de costo de la lista — SSOT backend.
# ------------------------------------------------------------
# Antes: `aggregate_and_deduct_shopping_list` acumulaba `total_estimated_cost`
# y lo DESCARTABA (los `return` solo devuelven ítems); el frontend re-sumaba
# `estimated_cost_rd` por su cuenta (Dashboard.jsx `_shopTotalCost` +
# `_fullCycleCost`) → drift posible y CERO punto backend contra el cual
# comparar el presupuesto del formulario (P1-BUDGET-RECONCILE).
# Ahora el backend computa y persiste `plan_data.shopping_cost_summary` en
# los mismos persist-sites que escriben `aggregated_shopping_list*`
# (generación / recalc / chat-modify / swap / regen-day).
# Semántica de ciclo (espejo del frontend P3-CYCLE-COST-TOTAL): las listas
# biweekly/monthly son HÍBRIDAS (estables del periodo completo + perecederos
# de 1 semana) → costo real del ciclo = estables 1× + perecederos × semanas.
# Reversible sin redeploy: MEALFIT_SHOPPING_COST_SUMMARY=false.
# Fail-open SIEMPRE: cualquier error → None (el caller no persiste la key).
# Tooltip-anchor: P1-BUDGET-COST-SSOT. Test: test_p1_budget_intelligence.py.
# ============================================================

# [P1-CYCLE-COVERAGE-FRACTIONAL · 2026-07-06] (review visual plan 30d) El mapa
# hardcoded {monthly:4} era floor(30/7): 4 semanas = 28 días → los días 29-30 del
# ciclo quedaban sin costear NI cubrir (biweekly igual: floor(15/7)=2 → día 15 sin
# cubrir). Modelo honesto que separa DOS conceptos:
#   - Multiplicador de COSTO = días/7 FRACCIONAL (30/7=4.286): costo real de los
#     perecederos consumidos en el ciclo, sin desperdicio artificial. NO usar ceil
#     aquí: comprar 5 semanas completas para 30 días sobre-estima ~RD$1.9k y podría
#     disparar un banner "excedido" falso contra el presupuesto.
#   - Nº de IDAS al súper (display) = ceil(días/7) (30d=5 idas, la 5ª parcial cubre
#     días 29-30): cuántas veces el usuario físicamente recompra perecederos.
# Espejo frontend: Dashboard.jsx (_cycleCostMultiplier + _cycleTrips).
_CYCLE_DAYS_BY_DURATION = {"weekly": 7, "biweekly": 15, "monthly": 30}


# [P1-COHERENCE-UNIT-MISMATCH-SYM · 2026-07-25] Marcar `unit_mismatch` también cuando el lado
# RECETA no tiene la unidad (espejo de P2-COHERENCE-PACKAGE-UNITS). Rollback: =false vuelve a
# emitir esas divergencias sin tagear, con `hypothesis=unknown` y `delta_pct=inf`.
COHERENCE_UNIT_MISMATCH_SYM = _knob_env_bool("MEALFIT_COHERENCE_UNIT_MISMATCH_SYM", True)


# [P1-CYCLE-REPURCHASE-HONEST · 2026-07-25] Un perecedero cuyo envase mínimo cubre varias semanas
# NO se re-compra cada semana. Rollback sin redeploy: =false vuelve al ×semanas plano.
CYCLE_REPURCHASE_HONEST = _knob_env_bool("MEALFIT_CYCLE_REPURCHASE_HONEST", True)


def _item_cycle_repurchases(item: dict, cycle_days: int, trip_days: int = 7) -> float:
    """Veces que se compra ESTE ítem durante el ciclo.

    `pkg_cover_ratio` dice cuántas veces cabe la necesidad de una ida dentro del envase mínimo,
    así que el envase alcanza para `trip_days × ratio` días. El límite lo pone lo que ocurra
    primero: que se acabe **o que se dañe** (`shelf_life_days`) — una funda de 3 lb de manzanas
    puede cubrir el mes en consumo y no aguantarlo en la nevera.

    Sin datos (ítem viejo, sin envase resuelto) devuelve el comportamiento previo: ciclo/ida.
    Nunca menos de 1. Cuando `ratio>=1` (el envase alcanza o sobra), nunca más que el plano — este
    caso sólo puede BAJAR el costo declarado, jamás inventar compras (comportamiento original de
    P1-CYCLE-REPURCHASE-HONEST). [P1-SKU-COVER-HONESTY · 2026-08-02] Cuando `ratio<1` (el envase
    mínimo NO alcanza ni una ida — el under-buy que ese fix deja de esconder en silencio), el
    clamp `min(plano, ...)` invertía la intención: escondía que se necesitan MÁS recompras que el
    plano, no menos. Ya no se clampa en ese caso — el costo declarado del ciclo sube para reflejar
    la recompra real. tooltip-anchor: P1-CYCLE-REPURCHASE-HONEST
    """
    plano = max(1.0, float(cycle_days) / max(1, trip_days))
    if not CYCLE_REPURCHASE_HONEST or not isinstance(item, dict):
        return plano
    try:
        ratio = float(item.get("pkg_cover_ratio") or 0)
    except (TypeError, ValueError):
        ratio = 0.0
    if ratio <= 0:
        return plano                      # sin señal → no adivinar
    cubre_dias = max(1.0, float(trip_days) * ratio)
    try:
        vida = float(item.get("shelf_life_days") or 0)
        if vida > 0:
            cubre_dias = min(cubre_dias, vida)
    except (TypeError, ValueError):
        pass
    recompras = float(cycle_days) / cubre_dias
    if ratio < 1.0:
        return max(1.0, recompras)
    return max(1.0, min(plano, recompras))


def _perishable_cycle_cost(items, cycle_days: int, weeks_flat: float) -> tuple[float, float]:
    """Costo de perecederos del ciclo sumando ítem a ítem sus re-compras reales.

    Devuelve `(costo, ahorro_vs_plano)`. El ahorro es telemetría: mide cuánto sobre-declaraba el
    ×semanas plano, que es lo que veía el usuario en "Costo real del ciclo".
    tooltip-anchor: P1-CYCLE-REPURCHASE-HONEST"""
    total = plano = 0.0
    for it in items or []:
        if not isinstance(it, dict) or it.get("is_perishable") is not True:
            continue
        try:
            cost = float(it.get("estimated_cost_rd"))
        except (TypeError, ValueError):
            continue
        if not (cost > 0) or math.isnan(cost) or math.isinf(cost):
            continue
        plano += cost * weeks_flat
        total += cost * _item_cycle_repurchases(it, cycle_days)
    return total, max(0.0, plano - total)


def _cycle_cost_multiplier(duration: str) -> float:
    """Semanas-equivalentes FRACCIONALES de perecederos consumidos en el ciclo
    (días/7). monthly=4.286, biweekly=2.143, weekly=1.0. NO redondear a entero:
    el costo del ciclo debe reflejar los días reales, no floor/ceil."""
    return _CYCLE_DAYS_BY_DURATION.get(str(duration or "").strip().lower(), 7) / 7.0


def _cycle_trip_count(duration: str) -> int:
    """Nº de idas al súper por perecederos en el ciclo = ceil(días/7). monthly=5
    (la 5ª ida cubre los días 29-30), biweekly=3, weekly=1. Es el número que se
    muestra al usuario ('la compras N veces') — distinto del multiplicador de costo."""
    return int(math.ceil(_CYCLE_DAYS_BY_DURATION.get(str(duration or "").strip().lower(), 7) / 7.0))


def _cycle_qty_fractional_enabled() -> bool:
    """Knob de la CANTIDAD del periodo — hermano de `_cost_summary_enabled` pero para
    CANTIDAD, no COSTO. True (default): `cycle_qty_multiplier` escala días/7 FRACCIONAL.
    False: rollback sin redeploy a los literales viejos 2.0/4.0 (14/28 días) que este
    P-fix reemplaza. tooltip-anchor: P1-CYCLE-QTY-FRACTIONAL."""
    return _knob_env_bool("MEALFIT_CYCLE_QTY_FRACTIONAL", True)


# [P1-CYCLE-QTY-FRACTIONAL · 2026-08-02] Hermano de `_cycle_cost_multiplier` (COSTO, ya
# fraccional desde P1-CYCLE-COVERAGE-FRACTIONAL) pero para la CANTIDAD comprada. Los
# callsites que construyen las listas biweekly/monthly llamaban a
# `get_shopping_list_delta(..., multiplier=household_multiplier * N, ...)` con `N`
# HARDCODEADO en 2.0/4.0 — 2/4 SEMANAS ENTERAS = 14/28 días — para ciclos declarados de
# 15/30 días. El delta de `get_shopping_list_delta` YA proyecta a 7 días
# (`base_duration_scale = 7.0/num_days`); multiplicar por 2.0/4.0 compra 14/28 días de
# estables (arroz, aceite, avena) para un ciclo que promete 15/30: déficit sistemático
# ~6.7%, invisible al guard de coherencia (compara contra la base SEMANAL con tolerancia
# 10%, nunca ve el ciclo completo). La sección del PDF "DESPENSA DEL MES — compra una
# sola vez" mentía 2 días.
#
# Entra ANTES del redondeo a envases comprables: multiplier → effective_multiplier
# (`get_shopping_list_delta`) → qty escalada (`aggregate_and_deduct_shopping_list`,
# "plan_ingredients: qty * multiplier") → weight_in_lbs → `apply_smart_market_units`.
# El redondeo a paquete sigue absorbiendo el ~7% de aumento igual que absorbe cualquier
# otro valor de multiplier — este fix no introduce un salto de envase nuevo.
#
# Los caps `_person_weeks` (P1-PERSON-WEEKS-CYCLE-AWARE) heredan el multiplicador nuevo
# automáticamente: `_person_weeks = multiplier * num_days / 7.0` cancela exactamente el
# `base_duration_scale = 7/num_days` que ya viene aplicado en `multiplier`, así que
# `_person_weeks = household × cycle_qty_multiplier(duration)` sin cambios en esa fórmula.
#
# Duración desconocida → 1.0 (fail-safe: nunca infla una compra por un valor que no
# entiende). Público (sin underscore): lo importan graph_orchestrator.py, cron_tasks.py,
# tools.py y routers/plans.py. Rollback sin redeploy:
# MEALFIT_CYCLE_QTY_FRACTIONAL=false reproduce los literales viejos exactos (2.0/4.0).
# tooltip-anchor: P1-CYCLE-QTY-FRACTIONAL. Test: test_p1_cycle_qty_fractional.py.
def cycle_qty_multiplier(duration: str) -> float:
    d = str(duration or "").strip().lower()
    if not _cycle_qty_fractional_enabled():
        return {"biweekly": 2.0, "monthly": 4.0}.get(d, 1.0)
    if d not in _CYCLE_DAYS_BY_DURATION:
        return 1.0
    return _CYCLE_DAYS_BY_DURATION[d] / 7.0


# [P1-SKU-COVER-HONESTY-R2 · 2026-08-02] SSOT hermano de `cycle_qty_multiplier`: misma tabla
# (`_CYCLE_DAYS_BY_DURATION`), mismo string `duration`, para que un callsite que YA pasa
# `duration` a `cycle_qty_multiplier(...)` derive `cycle_days` de la MISMA fuente en vez de
# escribir un literal `15`/`30` suelto al lado (dos SSOT que puedan driftear). Consumido por los
# ~43 callsites de `get_shopping_list_delta` (routers/plans.py, cron_tasks.py, tools.py,
# graph_orchestrator.py, agent.py) para que la nota "alcanza ~N de M días" de
# `apply_smart_market_units` (parámetro `cycle_days`) diga el ciclo real, no el 7 fijo default.
# Desconocida → 7 (mismo fail-safe que `cycle_qty_multiplier` → 1.0, aquí el equivalente en
# días es la semana, el caso más común). Público (sin underscore): mismos 5 importadores que
# `cycle_qty_multiplier`. Test: test_p1_sku_cover_honesty.py (sección "plumbing end-to-end").
def cycle_days_for_duration(duration: str) -> int:
    d = str(duration or "").strip().lower()
    return int(_CYCLE_DAYS_BY_DURATION.get(d, 7))


def _cost_summary_enabled() -> bool:
    return _knob_env_bool("MEALFIT_SHOPPING_COST_SUMMARY", True)


def _sum_shopping_list_costs(items) -> dict:
    """Suma `estimated_cost_rd` de una lista estructurada, particionando por
    `is_perishable` (flag SSOT P1-PDF-2; ítems sin flag → estables, mismo
    default que el aggregator). Ítems sin precio NO suman (honesto)."""
    total = stable = perishable = 0.0
    priced = 0
    count = 0
    for it in items or []:
        if not isinstance(it, dict):
            continue
        count += 1
        raw_cost = it.get("estimated_cost_rd")
        try:
            cost = float(raw_cost)
        except (TypeError, ValueError):
            cost = 0.0
        if not (cost > 0) or math.isnan(cost) or math.isinf(cost):
            continue
        priced += 1
        total += cost
        if it.get("is_perishable") is True:
            perishable += cost
        else:
            stable += cost
    return {
        "total": total,
        "stable": stable,
        "perishable": perishable,
        "priced": priced,
        "count": count,
    }


# [P1-BUDGET-TIER-LEVERS · 2026-07-02] Piso de precios del Supermercado RD:
# variante ACTIVA más barata por alimento (food_name + master_food_name como
# claves), para las sugerencias de ahorro de la reconciliación de presupuesto.
# Cache TTL 300s (mismo orden que el cache del catálogo master). Fail-open: {}.
_SUPERMARKET_FLOOR_CACHE: dict = {"at": 0.0, "map": None}
_SUPERMARKET_FLOOR_TTL_S = 300.0


def _fetch_supermarket_price_floor_map() -> dict:
    now = _time.time()
    cached = _SUPERMARKET_FLOOR_CACHE.get("map")
    if cached is not None and (now - _SUPERMARKET_FLOOR_CACHE.get("at", 0.0)) < _SUPERMARKET_FLOOR_TTL_S:
        return cached
    floor_map: dict = {}
    try:
        rows = execute_sql_query(
            """
            SELECT food_name, master_food_name, brand, presentation,
                   price_rd::float8 AS price_rd
            FROM public.supermarket_products
            WHERE active AND price_rd IS NOT NULL AND price_rd > 0
            """,
            fetch_all=True,
        ) or []
        for r in rows:
            try:
                price = float(r.get("price_rd"))
            except (TypeError, ValueError):
                continue
            if price <= 0:
                continue
            entry = {
                "food_name": r.get("food_name"),
                "brand": (r.get("brand") or "").strip(),
                "presentation": (r.get("presentation") or "").strip(),
                "price_rd": price,
            }
            for key_src in (r.get("food_name"), r.get("master_food_name")):
                key = _norm_pref_food(key_src)
                if not key:
                    continue
                current = floor_map.get(key)
                if current is None or price < current["price_rd"]:
                    floor_map[key] = entry
    except Exception as exc:
        logging.warning(f"⚠️ [P1-BUDGET-TIER-LEVERS] price-floor del súper no disponible (fail-open): {exc}")
        return cached or {}
    _SUPERMARKET_FLOOR_CACHE["map"] = floor_map
    _SUPERMARKET_FLOOR_CACHE["at"] = now
    return floor_map


def cheapest_supermarket_variant(item_name: str) -> dict | None:
    """Variante activa más barata del Supermercado RD para un ítem de la lista
    (misma escalera de matching que las preferencias de marca). None si no hay
    match o el catálogo no está disponible."""
    try:
        floor_map = _fetch_supermarket_price_floor_map()
        if not floor_map:
            return None
        return _resolve_brand_pref(item_name, floor_map)
    except Exception:
        return None


def estimate_new_ingredient_price_rd(item_name: str, qty_grams: float) -> float | None:
    """[P1-PANTRY-STRICT-CONSENT · 2026-08-02] Precio aproximado RD$ de UN ingrediente
    nuevo (fuera de la Nevera física) para el mensaje de consentimiento de
    "Nevera estricta" (swap/regen-day/fix-sodium-day). Reusa el MISMO piso de precios
    del Supermercado RD que ya cotiza la lista de compras (`cheapest_supermarket_variant`
    + `_variant_price_per_g`) — NO un estimador nuevo, para no driftear de lo que el
    usuario ve en la lista. Fail-open: `None` si el catálogo no tiene match o falta
    peso/precio — el caller omite el precio del mensaje en vez de inventar uno.
    Tooltip-anchor: P1-PANTRY-STRICT-CONSENT."""
    try:
        if not item_name or not qty_grams or float(qty_grams) <= 0:
            return None
        variant = cheapest_supermarket_variant(item_name)
        if not variant:
            return None
        per_g = _variant_price_per_g(variant)
        if not per_g or per_g <= 0:
            return None
        return round(per_g * float(qty_grams), 2)
    except Exception:
        return None


# [P1-BUDGET-BRAND-PREMIUM · 2026-07-07] (decisión de producto del owner, ablanda levemente P2-H): en
# el banner 'excedido' — y SOLO ahí, este helper solo corre en ese caso — informar cuánto CUESTA la
# elección de marcas premium del usuario vs la opción más económica, como UN total accionable ("RD$X de
# tu sobrecosto son tus marcas premium"). NO nag per-ítem (respeta el espíritu de P2-H: no molestar con
# cada marca elegida); solo un resumen cuando el premium es material. Reversible: MEALFIT_BUDGET_BRAND_PREMIUM_SURFACE=false.
def _budget_brand_premium_surface_enabled() -> bool:
    return _knob_env_bool("MEALFIT_BUDGET_BRAND_PREMIUM_SURFACE", True)


_BUDGET_BRAND_PREMIUM_MIN_RD = 40.0  # premium total mínimo (RD$) para surfacearlo (ruido si menos)


def _variant_price_per_g(v) -> float | None:
    """Precio/gramo de un variant del súper. Acepta ambos formatos: pkg ({grams, price}) y
    floor-variant ({price_rd, presentation → gramos parseables}). None si no resuelve."""
    if not isinstance(v, dict):
        return None
    price = v.get("price") if v.get("price") is not None else v.get("price_rd")
    grams = v.get("grams") or v.get("size_grams") or _parse_presentation_grams(v.get("presentation"))
    try:
        price = float(price)
        grams = float(grams)
    except (TypeError, ValueError):
        return None
    return price / grams if (price > 0 and grams > 0) else None


def build_budget_suggestions(weekly_list, limit: int = 5, user_id=None) -> list:
    """[P1-BUDGET-CONVERGENCE · 2026-07-03] (audit v6 · P1-3) Sugerencias de ahorro accionables
    para el banner `excedido`: variante más barata del Supermercado RD para los ítems más caros
    de la lista SEMANAL. Helper SSOT extraído del inline de assemble para que el refresh de
    recalc/updates (nutrition_calculator.refresh_budget_reconciliation) recompute sugerencias
    del estado ACTUAL en vez de reusar las stale de la generación. Fail-open: [].
    [P2-AUDIT-V6-BATCH · 2026-07-03] (P2-H) brand-aware: si el usuario YA eligió marca para un
    ítem (user_brand_preferences), NO se sugiere el piso absoluto de ese ítem — sugerir contra
    su elección es ruido. Guests/None → sin filtro (comportamiento previo).
    tooltip-anchor: P1-BUDGET-CONVERGENCE"""
    try:
        prefs = {}
        if user_id:
            try:
                prefs = fetch_brand_pref_packages(user_id) or {}
            except Exception:
                prefs = {}
        priced = [
            it for it in (weekly_list or [])
            if isinstance(it, dict) and (it.get("estimated_cost_rd") or 0) > 0
        ]
        priced.sort(key=lambda x: x.get("estimated_cost_rd") or 0, reverse=True)
        sugs = []
        _brand_premium_total = 0.0  # [P1-BUDGET-BRAND-PREMIUM] costo extra de las marcas premium elegidas
        for it in priced[: max(1, int(limit)) * 3]:
            name = str(it.get("name") or "").strip()
            if not name:
                continue
            _pref_var = _resolve_brand_pref(name, prefs) if prefs else None
            if _pref_var:
                # [P1-BUDGET-BRAND-PREMIUM · 2026-07-07] el usuario ELIGIÓ marca para este ítem. NO
                # sugerimos per-ítem (P2-H: no molestar con su elección), pero SÍ acumulamos cuánto le
                # cuesta esa elección vs la más económica → un solo resumen accionable al final.
                if _budget_brand_premium_surface_enabled():
                    try:
                        _cheap = cheapest_supermarket_variant(name)
                        _pppg = _variant_price_per_g(_pref_var)
                        _cppg = _variant_price_per_g(_cheap)
                        if _cheap and _pppg and _cppg and _cppg < _pppg * 0.92:
                            _brand_premium_total += float(it.get("estimated_cost_rd") or 0) * (1.0 - _cppg / _pppg)
                    except Exception:
                        pass
                continue  # respetar la elección del usuario (sin sugerencia per-ítem)
            var = cheapest_supermarket_variant(name)
            if var and var.get("brand"):
                sugs.append({
                    "type": "marca",
                    "item": name,
                    "text": (
                        f"{name}: la opción más económica del súper es "
                        f"{var['brand']} {var['presentation']} "
                        f"(RD${var['price_rd']:.0f})"
                    ),
                })
            if len(sugs) >= max(1, int(limit)):
                break
        # [P1-BUDGET-BRAND-PREMIUM] resumen accionable del sobrecosto por marcas premium elegidas.
        # PREPEND (no append): el frontend muestra solo las primeras 3 sugerencias (Dashboard.jsx
        # `_sugs.slice(0,3)`) → este resumen es el mensaje más importante, va primero.
        if _brand_premium_total >= _BUDGET_BRAND_PREMIUM_MIN_RD:
            sugs.insert(0, {
                "type": "marca_premium_total",
                "saving_rd": round(_brand_premium_total),
                "text": (f"~RD${round(_brand_premium_total)} de tu sobrecosto son tus marcas premium "
                         f"elegidas — cámbialas en /supermercado por la opción más económica para ahorrar."),
            })
        return sugs
    except Exception:
        return []


def compute_shopping_cost_summary(
    weekly_list,
    biweekly_hybrid_list,
    monthly_hybrid_list,
    active_duration: str = "weekly",
    *,
    pricing_mode: "str | None" = None,
) -> dict | None:
    """SSOT del costo de la lista para las 3 duraciones. `cycle_total_rd` =
    estables 1× + perecederos × semanas del ciclo (las listas 15/30 ya son
    híbridas: estables al periodo, perecederos semanales). `trip_total_rd` =
    lo que cuesta ESTA ida al súper (suma cruda de la lista).

    [P1-COUNTRY-SYSTEM-F1 · 2026-08-16 (T7)] `pricing_mode='beta_no_prices'` ⇒ `None` SIN
    computar nada — un plan beta no tiene `estimated_cost_rd` en sus ítems (ver
    `get_shopping_list_delta`/`_strip_prices_for_beta_pricing_mode`), así que sumarlos daría
    un dict de CEROS técnicamente correcto pero engañoso (parece "sin costo" en vez de "sin
    dato de costo"). `None`/ausente es el contrato honesto que ya usan el resto de los
    fail-opens de esta función — `shopping_cost_summary` sale AUSENTE del plan, y todo lo que
    depende de él (`budget_reconciliation`, `build_budget_suggestions`) queda río abajo
    inalcanzable (los call sites productivos son todos `if summary: ...`). Keyword-only,
    default `None` (comportamiento previo byte-idéntico para callers que no lo pasen).
    Cada call site productivo pasa `pricing_mode=<plan_data>.get('_pricing_mode')` — el MISMO
    dict del que ya leen `weekly_list`/etc., nunca un 2º chequeo de país.

    tooltip-anchor: compute_shopping_cost_summary pricing_mode (test_p1_country_system_f1.py)
    """
    if pricing_mode == "beta_no_prices":
        return None
    if not _cost_summary_enabled():
        return None
    try:
        by_duration: dict = {}
        for duration, items in (
            ("weekly", weekly_list),
            ("biweekly", biweekly_hybrid_list),
            ("monthly", monthly_hybrid_list),
        ):
            sums = _sum_shopping_list_costs(items)
            # [P1-CYCLE-COVERAGE-FRACTIONAL · 2026-07-06] costo = perecederos × (días/7)
            # fraccional (honesto para los 30 días); idas mostradas = ceil(días/7).
            weeks = _cycle_cost_multiplier(duration)
            trips = _cycle_trip_count(duration)
            # [P1-CYCLE-REPURCHASE-HONEST · 2026-07-25] …pero POR ÍTEM: el ×semanas plano cobraba
            # 5 veces la funda de 3 lb de manzanas que dura el mes entero.
            _cycle_days = _CYCLE_DAYS_BY_DURATION.get(duration, 7)
            _perish_cycle, _ahorro = _perishable_cycle_cost(items, _cycle_days, weeks)
            cycle_total = sums["stable"] + _perish_cycle
            by_duration[duration] = {
                "trip_total_rd": round(sums["total"], 2),
                "stable_rd": round(sums["stable"], 2),
                "perishable_rd": round(sums["perishable"], 2),
                "cycle_weeks": round(weeks, 3),
                "cycle_trips": trips,
                "cycle_total_rd": round(cycle_total, 2),
                "cycle_repurchase_saving_rd": round(_ahorro, 2),
                "items_priced": sums["priced"],
                "items_total": sums["count"],
            }
        active = str(active_duration or "weekly").strip().lower()
        if active not in _CYCLE_DAYS_BY_DURATION:
            active = "weekly"
        from datetime import datetime, timezone
        summary = {
            "version": 1,
            "source": "backend",
            "active_duration": active,
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "by_duration": by_duration,
        }
        _active_totals = by_duration[active]
        logging.info(
            f"💵 [P1-BUDGET-COST-SSOT] summary {active}: trip=RD${_active_totals['trip_total_rd']:.0f} "
            f"cycle=RD${_active_totals['cycle_total_rd']:.0f} "
            f"({_active_totals['items_priced']}/{_active_totals['items_total']} con precio)"
        )
        _sav = _active_totals.get("cycle_repurchase_saving_rd") or 0
        if _sav > 0:
            logging.info(
                f"🧺 [P1-CYCLE-REPURCHASE-HONEST] {active}: RD${_sav:.0f} que el ×semanas plano "
                f"cobraba de más (envases que cubren varias idas al súper)."
            )
        return summary
    except Exception as exc:
        logging.warning(f"⚠️ [P1-BUDGET-COST-SSOT] summary falló (fail-open): {exc}")
        return None


def to_unicode_fraction(frac_str: str) -> str:
    mapping = {"1/4": "¼", "1/2": "½", "3/4": "¾"}
    return mapping.get(frac_str, frac_str)


# ============================================================
# [P1-PDF-5] Sufijo parentizado SIN ambigüedad para `display_qty`.
# ------------------------------------------------------------
# Antes el formato era literal `f"({sku_label})"` para todos los casos.
# Cuando count > 1, el usuario no podía distinguir si la cantidad en
# paréntesis era el peso/tamaño TOTAL o POR EMPAQUE:
#   - "16 paquetes (1 lb)"   ¿16 paquetes que SUMAN 1 lb? ¿O 16 paquetes
#                             de 1 lb c/u = 16 lbs?
#   - "13 sobres (14g)"      Físicamente imposible: 13 sobres no caben en
#                             14g; el `14g` es POR sobre. Lectura errónea
#                             llevaba al usuario a comprar de menos.
#   - "9 potes (16 oz)"      Tras P1-PDF-4 fix: 9 potes de 16 oz c/u =
#                             ~9 lbs; sin sufijo, ambiguo.
#
# Convención dominicana de supermercado: "c/u" = "cada uno" (etiquetas
# de góndola). Convención simétrica para totales aproximados: prefijo
# "~" + sufijo " total" — ya implícito por mega-frutas (lechosa,
# aguacate) donde `~X lbs` representa el peso TOTAL agregado, no por
# unidad.
#
# Reglas:
#   - count <= 1 → `(label)` (no hay ambigüedad con 1 unidad)
#   - count >  1 + label inicia con "~" → `(label total)` (mega-frutas:
#                                          peso TOTAL aproximado)
#   - count >  1 + label exacto         → `(label c/u)` (containers:
#                                          tamaño POR EMPAQUE)
# ============================================================
def _format_pkg_suffix(count, label: str) -> str:
    """Devuelve el sufijo parentizado con disambiguación per-package vs total.

    Sin sufijo (`""`) si `label` está vacío. Acepta `count` numérico o string
    convertible a float; degrada a `count_int=1` (sin "c/u") ante valores
    no-parseables.
    """
    if not label:
        return ""
    try:
        count_int = int(float(count))
    except (TypeError, ValueError):
        count_int = 1
    if count_int <= 1:
        return f"({label})"
    if label.startswith("~"):
        return f"({label} total)"
    return f"({label} c/u)"


def _has_pkg_suffix(display_qty: str, label: str) -> bool:
    """True si `display_qty` ya contiene cualquier variante del sufijo
    (legacy `(label)` o nuevas `(label c/u)` / `(label total)`).

    Usado por el wrapper de cierre para no duplicar sufijos cuando un
    bloque previo ya los añadió.
    """
    if not label or not display_qty:
        return False
    return any(v in display_qty for v in (
        f"({label})", f"({label} c/u)", f"({label} total)"
    ))


# [P0-3] Decimal canónico para las únicas fracciones que el motor de
# pesos dominicanos genera ("1/4", "1/2", "3/4"). Se usa para construir
# `market_qty` SIEMPRE como float, dejando el string fraccional unicode
# ("¼ lb", "1 ½ lbs") sólo en `display_qty`. Antes el campo era de
# tipo mixto (a veces float, a veces string como "1/2"/"1 1/2"), lo
# que rompía consumers numéricos (Restock que persiste a `user_inventory`,
# pricing, agregadores, frontend con `parseFloat(market_qty)`).
_FRACTION_DECIMAL = {"1/4": 0.25, "1/2": 0.5, "3/4": 0.75}

# [P2-MARKET-FRACTION-NO-SHORTFALL · 2026-07-31] Déficit máximo tolerado al redondear el peso a
# cuartos de libra. Por debajo de esto se permite bajar (evita encarecer una proteína cara por unos
# gramos); por encima se sube al siguiente cuarto. Clamp [0, 0.15]: con 0 se compra siempre de más,
# con 0.15 se vuelve al comportamiento previo de "cuarto más cercano".
MARKET_FRACTION_SHORTFALL_TOL = _knob_env_float(
    "MEALFIT_MARKET_FRACTION_SHORTFALL_TOL", 0.05, lambda v: 0.0 <= v <= 0.15)

_MARKET_FRACTION_LADDER = ("", "1/4", "1/2", "3/4")


def _lbs_to_market_fraction(lbs: float) -> "tuple[int, str]":
    """[P2-MARKET-FRACTION-NO-SHORTFALL · 2026-07-31] Peso en libras → (enteros, fracción de mercado).

    Antes esto era una escalera inline que redondeaba al cuarto MÁS CERCANO
    (`frac < 0.15 → ""`, `<= 0.35 → "1/4"`, …). El problema no es el redondeo en sí: es que el número
    que recibe (`lbs_for_weighable`) YA pasó por el anti-desperdicio y un `ceil`, o sea que la
    cantidad era correcta y esta última capa la bajaba por debajo del requisito. Medido en el plan
    fe788498: 5 ítems de 34 por debajo de lo que las recetas piden, hasta −22,9% (molondrones).

    En una lista de compras la asimetría importa — que sobre queda en la nevera, que falte te deja
    sin cocinar — pero subir SIEMPRE encarecería de más una proteína cara por un déficit trivial
    (el chivo se queda a −2,8%). Por eso el criterio es una tolerancia y no un `ceil` duro.

    Pura y determinista. tooltip-anchor: P2-MARKET-FRACTION-NO-SHORTFALL"""
    try:
        _lbs = max(0.0, float(lbs))
    except Exception:
        return 0, ""
    whole = math.floor(_lbs)
    frac_w = _lbs - whole

    # candidato "más cercano" (comportamiento histórico)
    if frac_w < 0.15:
        fraction_str = ""
    elif frac_w <= 0.35:
        fraction_str = "1/4"
    elif frac_w <= 0.65:
        fraction_str = "1/2"
    elif frac_w <= 0.85:
        fraction_str = "3/4"
    else:
        fraction_str = ""
        whole += 1

    # ¿el candidato deja al usuario corto más allá de la tolerancia? → siguiente peldaño
    _elegido = whole + _FRACTION_DECIMAL.get(fraction_str, 0.0)
    if _lbs > 0 and _elegido < _lbs * (1.0 - MARKET_FRACTION_SHORTFALL_TOL):
        _i = _MARKET_FRACTION_LADDER.index(fraction_str)
        if _i + 1 < len(_MARKET_FRACTION_LADDER):
            fraction_str = _MARKET_FRACTION_LADDER[_i + 1]
        else:
            whole += 1
            fraction_str = ""
    return int(whole), fraction_str


def _sku_size_label(size_g: float, unit_hint: str = None) -> str:
    """Convierte gramos a etiqueta legible de mercado dominicano.
    
    453g → '1lb', 908g → '2lb', 473g → '473ml', 946g → '946ml', 200g → '200g'
    Con soporte especial para potes/frascos en onzas fluidas.
    """
    if size_g is None:
        return ""
    size_g = float(size_g)
    if unit_hint and unit_hint.lower() in ['cartón', 'carton', 'botella', 'ml', 'l', 'galón', 'envase', 'lata']:
        # Tamaños de volumen conocidos (leche, jugos — se venden por ml, no por peso)
        VOLUME_LABELS = {250: "250ml", 473: "473ml", 946: "946ml", 1000: "1L", 1892: "1/2 Galón"}
        for vol_g, label in VOLUME_LABELS.items():
            if abs(size_g - vol_g) < 10:
                return label
        # [BOTELLA-ML-FALLBACK] Si el contenedor es una botella/lata pero el peso
        # no matchea ninguno de los tamaños canónicos (e.g. aceite de oliva 500g),
        # NO debemos caer al fallback genérico que produciría "500g". Los líquidos
        # de cocina (aceite, vinagre, salsas) tienen densidad ≈1 g/ml, así que
        # mostrar el mismo número como "ml" es correcto y mucho más legible que
        # "500g" en una botella de aceite (visto 2026-05-06).
        if unit_hint.lower() in ['botella', 'ml', 'l', 'galón']:
            if size_g >= 1000:
                # Convertir a litros con un decimal cuando es ≥1L (1500g → "1.5L")
                liters = size_g / 1000
                if abs(liters - round(liters)) < 0.05:
                    return f"{round(liters):d}L"
                return f"{liters:.1f}L"
            return f"{int(round(size_g))}ml"
            
    if unit_hint and unit_hint.lower() in ['pote', 'frasco']:
        # Mapeos típicos de onzas para potes (yogurt, queso crema, aceitunas)
        if abs(size_g - 453.592) < 15: return "16 oz"
        if abs(size_g - 226.796) < 15: return "8 oz"
        if abs(size_g - 340.194) < 15: return "12 oz"
    
    lbs = size_g / 453.592
    # Libras enteras limpias — threshold estricto (±2%) para no confundir 473g con 1lb
    if abs(lbs - round(lbs)) < 0.05 and round(lbs) >= 1:
        return f"{round(lbs)} lb" if round(lbs) == 1 else f"{round(lbs)} lbs"
    # Media libra
    if abs(lbs - 0.5) < 0.05:
        return "½ lb"
    if abs(lbs - 0.25) < 0.05:
        return "¼ lb"
        
    # Mejorar la etiqueta para pesos de mega frutas o porciones grandes (ej. 800g -> ~1.8 lbs)
    if lbs > 1.2:
        return f"{round(lbs, 1):g} lbs"
        
    # Todo lo demás en gramos
    return f"{int(size_g)}g"


# [P1-COHERENCE-BASE-QTY · 2026-07-26] Cantidad en unidad BASE del item de la lista, en el
# mismo idioma en que hablan las recetas (g / taza / cda / unidad). Es lo que permite al
# coherence guard emparejar; con `market_qty` (pote/sobre/paquete/mazo) no hay pareja posible.
#
# Las DOS rutas del aggregator entregan la cantidad en sitios distintos:
#   - por unidades: raw_qty>0, unit_str='unidad'/'taza'/... -> se usa tal cual
#   - por peso:     raw_qty=0.0 y la cantidad viaja en weight_in_lbs -> se convierte a gramos
# Sin cubrir la segunda, el 96% de los items se quedaba sin base (medido: 2 de 48).
_LB_TO_G = 453.592


def _coherence_base_fields(raw_qty, unit_str, weight_in_lbs) -> dict:
    """Devuelve `{'base_qty', 'base_unit'}` o `{}` si no se puede determinar. Fail-safe."""
    try:
        if isinstance(raw_qty, (int, float)) and float(raw_qty) > 0 and unit_str:
            return {"base_qty": float(raw_qty), "base_unit": str(unit_str).strip().lower()}
        if isinstance(weight_in_lbs, (int, float)) and float(weight_in_lbs) > 0:
            return {"base_qty": round(float(weight_in_lbs) * _LB_TO_G, 2), "base_unit": "g"}
    except Exception:
        pass
    return {}



def _purchase_covers_need(item: dict, need_g: float) -> bool:
    """[P1-COVERAGE-VS-PURCHASE · 2026-07-27] True si lo que el usuario COMPRA cubre lo que el
    plan necesita (antes del tope). Puro y fail-safe: ante cualquier duda devuelve False, o sea
    que el aviso se conserva — nunca peor que el comportamiento previo.

    Se calcula en gramos desde el envase (`package_grams` × cantidad) o desde la propia unidad
    cuando ya es de peso (lb/kg/g). Sin datos suficientes → False.
    """
    try:
        need = float(need_g or 0)
        if need <= 0:
            return False
        qty = float(item.get("market_qty_numeric") or 0)
        if qty <= 0:
            return False
        pkg = item.get("package_grams")
        if isinstance(pkg, (int, float)) and float(pkg) > 0:
            return (qty * float(pkg)) >= need
        unit = str(item.get("market_unit") or "").strip().lower()
        if unit.startswith("lb"):
            return (qty * _LB_TO_G) >= need
        if unit == "kg":
            return (qty * 1000.0) >= need
        if unit in ("g", "gr", "gramos"):
            return qty >= need
        return False
    except Exception:
        return False

def apply_smart_market_units(name: str, weight_in_lbs: float, unit_str: str, raw_qty: float, master_item: dict = None, cycle_days: int = 7, text_demand_g: float = None):
    """Motor determinístico de unidades de mercado dominicano.

    Flujo de resolución (4 bloques, sin hardcoded weights):
      1. DB Container: market_container + container_weight_g → Potes, Paquetes, Cartones, etc.
         1a. SKU-Aware: si hay available_sizes_g, optimiza tamaño de empaque
      2. DB Density:   density_g_per_unit → Unidades físicas (frutas, vegetales, huevos)
      3. Dominican Lbs: Fracciones de libra (1/4, 1/2, 3/4) para carnes, quesos, granel
      4. Raw Fallback:  Cantidades crudas del AI sin conversión

    Returns dict con confidence_score (1.0=DB+SKU, 0.95=DB, 0.85=density, 0.75=lbs, 0.5=raw)

    `cycle_days` [P1-SKU-COVER-HONESTY-R1 · 2026-08-02]: cuántos días de necesidad representa
    `weight_in_lbs`/`raw_qty` — default 7 (semanal), el comportamiento histórico. El caller es
    responsable de pasar el valor real (15/30) cuando construye la necesidad para una lista
    biweekly/monthly (el multiplicador de ciclo ya viene aplicado ANTES de esta función, así que
    esta función no puede inferirlo). Sólo afecta el copy de la nota "alcanza ~N de M días —
    recompra"; NO afecta ninguna decisión de cantidad/floor-ceil.

    `text_demand_g` [P1-VEG-BACKFILL-HONESTY · 2026-08-02]: demanda en GRAMOS que las recetas
    piden para este alimento (mismo parse que usa el coherence guard —
    `expected_sum_from_recipes` + `_normalize_food_units_to_base`, threaded desde
    `get_shopping_list_delta`/`aggregate_and_deduct_shopping_list`). Default `None` = no-op
    (comportamiento previo byte-idéntico). Si viene y NINGÚN cap real ya explica el déficit
    (`_cap_hit` de P1-CAPPED-STAPLE-HONESTY sigue `None`), y `base_qty` resuelve por debajo de
    `QTY_SHORTFALL_NOTE_MIN × text_demand_g`, se estampa `capped_by="qty_reconcile_v7"` sintético
    para que el bloque de abajo componga la misma nota "alcanza ~N de M días — recompra".
    ⚠️ El caller DEBE pasar demanda homogénea con lo que compra: si su lista es un DELTA (resta la
    Nevera / lo consumido), el mapa de texto tiene que venir vacío o la nota es falsa por
    construcción — ver el gate `_tdg_para_agg` en `get_shopping_list_delta`.
    ⚠️ El sello sintético NO escribe `capped_pre`/`capped_post` (usa `shortfall_text_g` /
    `shortfall_bought_g`): esos dos campos son el canal que el coherence guard usa para SUSTITUIR
    la cantidad comprada, y el `pre_value` del sintético es el propio lado esperado del guard.
    """
    import math
    from constants import UNIT_WEIGHTS
    import unicodedata
    n_lower = name.lower()
    
    if master_item is None:
        master_item = {}
        
    cat = (master_item.get("category") or "").lower()
    density_per_u = master_item.get("density_g_per_unit")
    if density_per_u is not None:
        density_per_u = float(density_per_u)

    # Fallback Semántico si no hay densidad en la DB
    # [P3-WEIGHT-DEFAULT-NO-UNITIZE · 2026-06-22] Si el item está DECLARADO por peso
    # (default_unit ∈ lb/kg/g) y la DB no tiene densidad, NO inventar una densidad-unidad
    # desde UNIT_WEIGHTS — ese fallback forzaba unitización absurda: sandía con "L=30/lb"
    # (owner) + density NULL caía a UNIT_WEIGHTS["sandia"]=3000g → "1 sandía entera (~6.6 lbs)
    # RD$198" para una necesidad de 200g, en vez de costear por libra (0.5 lb = RD$15). Carnes
    # ya están excluidas de BLOQUE 2 por is_meat_seafood; este guard cubre frutas/víveres
    # vendidos por peso (sandía picada por libra). Tooltip-anchor: P3-WEIGHT-DEFAULT-NO-UNITIZE.
    _du_weight = (master_item.get("default_unit") or "").strip().lower() in ('lb', 'lbs', 'kg', 'g', 'gr', 'gramo', 'gramos')
    if not density_per_u and not _du_weight:
        from constants import UNIT_WEIGHTS
        n_clean = ''.join(c for c in unicodedata.normalize('NFD', n_lower) if unicodedata.category(c) != 'Mn')
        # Búsqueda exacta o como palabra entera para evitar "agua" == "pan de agua"
        for k, v in UNIT_WEIGHTS.items():
            if k == n_clean or (re.search(rf'\b{re.escape(k)}(s|es)?\b', n_clean)):
                density_per_u = v
                break
        # Fallback plurales multi-palabra: "guineitos verdes" → "guineito verde"
        if not density_per_u:
            n_singular = re.sub(r'(es|s)\b', '', n_clean).strip()
            for k, v in UNIT_WEIGHTS.items():
                if k == n_singular or n_singular.startswith(k) or k.startswith(n_singular):
                    density_per_u = v
                    break

    # Autocorrección de Alucinaciones (unidades líquidas para sólidos)
    if unit_str.lower() in ['ml', 'l', 'lt', 'oz', 'onzas'] and re.search(r'queso|pollo|cerdo|carne|arroz|avena|lenteja|habichuela|almendra', n_lower):
        if weight_in_lbs <= 0 and raw_qty > 0:
            weight_in_lbs = raw_qty / 453.59 if unit_str.lower() in ['g', 'ml'] else raw_qty / 16.0
        unit_str = 'lb'
        
    was_unitarized = False
    display_qty = ""
    market_qty = weight_in_lbs if weight_in_lbs > 0 else raw_qty
    market_unit = "lbs" if weight_in_lbs > 0 else unit_str
    confidence = 0.5  # Default: raw fallback
    sku_label = None   # None = no SKU optimization applied
    # [P1-BRAND-SIZE-FILTER · 2026-07-06] Tamaño en gramos del ENVASE elegido —
    # viaja al ítem como `package_grams` para que el picker de marcas filtre las
    # variantes del súper al MISMO tamaño que la lista muestra (ej. 2 lb).
    _pkg_size_g = None
    # [P1-BRAND-DEFAULT-PRESELECTED · 2026-07-06] Identidad del producto del súper
    # que el costeo usó (default más barato O preferencia) → `brand_product_id`
    # del ítem → el picker lo pre-selecciona ("la marca que tu lista está usando").
    _pkg_product_id = None
    # [P1-SKU-COVER-HONESTY-R1 · 2026-08-02] Señal para la nota de under-buy: cuando
    # P2-PACK-UNITS-MATCH recuenta `sku_count` por UNIDADES reales del envase (no por gramos),
    # el `pkg_cover_ratio` en gramos (que usa la density del MASTER, no la del SKU real) deja de
    # ser una medida válida de cobertura — puede leer <0.9 con el conteo por-unidades ya
    # correcto. Medido: ratio 0.712 con el conteo de unidades exacto. La nota se suprime cuando
    # esta bandera queda en True (ver bloque de la nota, más abajo).
    _pkg_units_recounted = False

    # Guards mínimos para Bloques 2 y 3 (solo 2 regex, eliminados los 15+ anteriores)
    is_meat_seafood = bool(re.search(r'\b(pollo|cerdo|carne|res|pescado|camar[oó]n|camarones|mariscos?|filetes?|chuletas?|longanizas?|salamis?|jam[oó]n|pavo|tocineta|bacon|salchichas?)\b', n_lower))
    is_cheese = bool(re.search(r'\b(quesos?|mozzarella|cheddar|parmesano|gouda|dan[eé]s)\b', n_lower)) and not re.search(r'\b(crema|mantequilla)\b', n_lower)

    # Nuevas clasificaciones Nivel de Producción (Actualizado con plurales y más alimentos)
    is_native_countable = bool(re.search(r'\b(pl[aá]tanos?|guineos?|lim[oó]n|limones|huevos?|manzanas?|naranjas?|peras?|chinolas?|mandarinas?|kiwis?|duraznos?)\b', n_lower))
    is_mega_fruit = bool(re.search(r'\b(aguacates?|pi[ñn]as?|sand[ií]as?|mel[oó]n|melones|lechosas?|papayas?)\b', n_lower))
    is_native_weighable = bool(re.search(r'\b(zanahorias?|tomates?|aj[ií]es?|cebollas?|papas?|yucas?|batatas?|berenjenas?|tayotas?|remolachas?|calabac[ií]nes?|calabac[ií]n|auyamas?|vegetales|[ñn]ames?|yaut[ií]as?|pimientos?|chiles?)\b', n_lower))
    is_native_cabeza = bool(re.search(r'\b(br[oó]colis?|coliflor|repollos?|lechugas?)\b', n_lower))
    # [P1-CEBOLLIN-HERB-GARNISH · 2026-07-07] cebollín (chives) es hierba aromática de
    # guarnición — el clasificador ya lo trata como 'Cebollín' aparte, pero estaba OMITIDO
    # de is_herb_mazo (y de _HERB_NAMES_FOR_CAP) → "cebollín al gusto" caía a SEASONING-KEEP
    # con el paquete completo del master (375g) → "1 paquete (25 unid) RD$229" × ciclo. Con
    # esto va al BLOQUE 1.5 (mazo ~50g) + al cap por persona-semana. `\bcebollin` NO matchea
    # 'cebolla' (onion). tooltip-anchor: P1-CEBOLLIN-HERB-GARNISH
    is_herb_mazo = bool(re.search(r'\b(cilantro|cilantrico|puerro|perejil|menta|albahaca|romero|verdura|verdurita|recao|eneldo|cebollin|cebollín|cebollines|cebollínes)\b', n_lower))

    # ═══════════════════════════════════════════════════════════════
    # BLOQUE 1: Resolución Data-Driven (PRIORIDAD MÁXIMA)
    # Usa market_container + container_weight_g directamente de la DB.
    # Cubre: Lácteos(Pote/Cartón), Despensa(Paquete/Fundita/Botella),
    #         Especias(Sobre), Vegetales(Mazo/Cabeza/Lata), etc.
    # Anti-desperdicio (Ahora estricto): 2% de colchón para errores de coma flotante. 
    # Forzará compras mayores a la mínima escalada matemática (Ej: 4 personas vs 6).
    ANTI_WASTE_THRESHOLD = 0.02

    db_container = master_item.get("market_container")
    # [P1-POTE-PRICING · 2026-06-22] Fallback a `default_unit` cuando `market_container`
    # es NULL pero el item se vende en ENVASE (default_unit ∈ _CONTAINER_UNIT_ALIASES)
    # con `container_weight_g` poblado. Sin esto, items como la Mantequilla de maní
    # (pote, 454g, price_per_lb>0) NO se unitarizaban → caían al display "X lb" + cobro
    # a GRANEL (½ lb ≈ RD$29) en vez de por POTE (1 pote = RD$117). Solo aplica a
    # unidades de envase (NO pesos lb/g/kg) → no "envasa" un staple a granel. Cubre
    # además Leche evaporada/lata, Galletas de soda/paquete, Harina de trigo/paquete,
    # Yogurt griego/pote (mismo gap de datos: market_container NULL).
    if not db_container and (master_item.get("default_unit") or "").strip().lower() in _CONTAINER_UNIT_ALIASES:
        db_container = master_item.get("default_unit")
    db_container_weight_g = master_item.get("container_weight_g")
    available_sizes = master_item.get("available_sizes_g")
    market_packages = master_item.get("market_packages")
    # [P1-PKG-DURATION-PRICING · 2026-06-22] Precio del envase elegido (RD$/paquete) — se
    # propaga al market_obj para que `_cost_from_market` costee por tamaño real, no plano.
    market_pkg_price = None

    if db_container and db_container_weight_g and weight_in_lbs > 0:
        g_total = weight_in_lbs * 453.592

        # ── [P1-PKG-DURATION-PRICING] Path por market_packages (tamaño + PRECIO real) ──
        # Tiene prioridad sobre available_sizes_g: usa la MISMA heurística de SKU sobre los
        # tamaños con precio, y arrastra el precio del tamaño elegido (descuento por volumen).
        _mp_sel = _select_market_package(g_total, market_packages)
        if _mp_sel is not None:
            sku_count = _mp_sel["count"]
            sku_size_g = _mp_sel["grams"]
            sku_label = _mp_sel["label"] or _sku_size_label(sku_size_g, db_container)
            market_pkg_price = _mp_sel["price"]
            # [P2-PACK-UNITS-MATCH · 2026-07-06] (review #13) El envase declara sus UNIDADES
            # ("Burrito 5 unid 356 gr") y la demanda del ítem nació CONTABLE (el aggregator
            # convirtió unidades→gramos con la density del MASTER). Si esa density (48g/
            # tortilla) difiere de la real del SKU (356/5 = 71g), contar por gramos
            # sub-compra: 4 paquetes = 20 tortillas para ~30 necesarias. Recontar por
            # unidades reales: ceil(unidades_necesarias / unid_por_envase).
            _upp_m = re.search(r"(\d+)\s*unid", str(sku_label), re.IGNORECASE)
            if _upp_m and density_per_u and float(density_per_u) > 0:
                try:
                    _upp = int(_upp_m.group(1))
                    _units_needed = g_total / float(density_per_u)
                    if _upp > 0 and _units_needed >= 1.0:
                        _cnt_u = max(1, math.ceil(_units_needed / _upp - 0.02))
                        if _cnt_u != sku_count:
                            logging.info(
                                f"🧮 [P2-PACK-UNITS-MATCH] '{name}': {sku_count}→{_cnt_u} paquete(s) "
                                f"por unidades reales del envase ({_upp}/envase, necesita "
                                f"≈{_units_needed:.1f} uds; density master {float(density_per_u):.0f}g "
                                f"vs SKU {sku_size_g / _upp:.0f}g)")
                            sku_count = _cnt_u
                            # [P1-SKU-COVER-HONESTY-R1 · 2026-08-02] el conteo ya no viene del
                            # floor/ceil en gramos — `pkg_cover_ratio` (en gramos, density del
                            # MASTER) deja de medir cobertura real; suprimir la nota de under-buy.
                            _pkg_units_recounted = True
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
            _pkg_size_g = sku_size_g
            _pkg_product_id = _mp_sel.get("id") or None
            # [P1-EGG-CARTON-SIZES · 2026-06-22] Unidad de DISPLAY = la forma del envase
            # elegido (lata vs paquete) cuando difiere del market_container genérico. Sin
            # `unit` en el package → fallback a db_container (comportamiento previo).
            _pkg_unit = _mp_sel.get("unit") or db_container
            display_qty = (
                f"{sku_count} {get_plural_unit(sku_count, _pkg_unit)} "
                f"{_format_pkg_suffix(sku_count, sku_label)}"
            ).rstrip()
            market_qty = sku_count
            market_unit = _pkg_unit
            was_unitarized = True
            confidence = 1.0
        # ── SKU-Aware Path: múltiples tamaños disponibles ──
        elif available_sizes and isinstance(available_sizes, list) and len(available_sizes) > 1:
            sku_count, sku_size_g = _find_best_sku(g_total, available_sizes, ANTI_WASTE_THRESHOLD)
            sku_label = _sku_size_label(sku_size_g, db_container)
            _pkg_size_g = sku_size_g
            # [P1-PDF-5] Sufijo "c/u" cuando count > 1 para evitar lectura
            # ambigua: "9 potes (16 oz)" → "9 potes (16 oz c/u)".
            display_qty = (
                f"{sku_count} {get_plural_unit(sku_count, db_container)} "
                f"{_format_pkg_suffix(sku_count, sku_label)}"
            ).rstrip()
            market_qty = sku_count
            market_unit = db_container
            was_unitarized = True
            confidence = 1.0
        else:
            # ── Standard Path: tamaño único de envase ──
            container_weight_g = float(db_container_weight_g)
            if container_weight_g > 0:
                raw_units = g_total / container_weight_g
                floor_units = math.floor(raw_units)
                frac = raw_units - floor_units
                # [2026-05-06 SKU-OVERSHOOT-FIX] Cuando container ≈ g_total
                # (ej. Pan integral: g_total=600g cap monthly, container=567g),
                # `frac=0.058` superaba `ANTI_WASTE_THRESHOLD=0.02` y forzaba
                # ceil → 2 paquetes (1134g, 89% sobre el cap). El cap se
                # respeta en gramos pero el SKU resolver inflaba al doble.
                # Regla nueva: si comprar el floor (under-buy) deja una
                # carencia ABSOLUTA menor que la que generaría comprar uno
                # más (over-buy), preferir floor — el usuario queda más
                # cerca del target real. La heurística <=2% sigue como
                # gateway primario para preservar el comportamiento previo
                # cuando ambas opciones son razonables (e.g., g_total=920,
                # container=454 → floor=2 cubre 99%, ceil=3 sobra 36%).
                if floor_units >= 1:
                    under_buy_g = g_total - (floor_units * container_weight_g)
                    over_buy_g = ((floor_units + 1) * container_weight_g) - g_total
                    # [P1-SKU-COVER-HONESTY · 2026-08-02] `under_buy_g < over_buy_g` retenía el
                    # floor con under-buy de hasta 50% del total sin aviso (medido en prod: arroz
                    # cover 0.69-0.81, aceite 0.70-0.93, camarones 0.76-0.79 sobre 18/22 planes).
                    # Acotado a `SKU_FLOOR_MAX_UNDER_PCT` del total; `frac <= ANTI_WASTE_THRESHOLD`
                    # (colchón anti-desperdicio de coma flotante, SKU-OVERSHOOT-FIX) intacto.
                    # [P1-SKU-COVER-HONESTY-R1 · 2026-08-02] `and under_buy_g < over_buy_g`: el
                    # bound puro solo (sin este AND) escala con `floor_units` y es vacuo desde
                    # floor_units>=9 (default 10%) — permitiría un déficit NUEVO que el criterio
                    # viejo no permitía (medido: Sazón 137g/sobre 14g). El AND garantiza que el
                    # resultado es subconjunto estricto del criterio viejo: nunca peor.
                    if frac <= ANTI_WASTE_THRESHOLD or (under_buy_g <= g_total * SKU_FLOOR_MAX_UNDER_PCT and under_buy_g < over_buy_g):
                        units_needed = floor_units
                    else:
                        units_needed = floor_units + 1
                else:
                    units_needed = max(1, math.ceil(raw_units))
                
                sku_label = _sku_size_label(container_weight_g, db_container)
                _pkg_size_g = container_weight_g
                # [P1-PDF-5] Sufijo "c/u" cuando units_needed > 1 — ver
                # docstring de `_format_pkg_suffix`. Antes "13 sobres (14g)"
                # leía como "14g totales", ahora "13 sobres (14g c/u)".
                display_qty = (
                    f"{units_needed} {get_plural_unit(units_needed, db_container)} "
                    f"{_format_pkg_suffix(units_needed, sku_label)}"
                ).rstrip()
                market_qty = units_needed
                market_unit = db_container
                was_unitarized = True
                confidence = 0.95

    # ═══════════════════════════════════════════════════════════════
    # BLOQUE 1.5: Intercepción de Hierbas Flexibles (Nivel 5)
    # Siempre se compran por mazo o atadito en RD, evitando "1/4 lb" o "15g"
    # ═══════════════════════════════════════════════════════════════
    if not was_unitarized and is_herb_mazo:
        g_total = (weight_in_lbs * 453.592) if weight_in_lbs > 0 else 0
        if unit_str.lower() in ['mazo', 'mazos', 'atado', 'atados']:
            units_needed = max(1, math.ceil(raw_qty))
        else:
            units_needed = max(1, math.ceil(g_total / 50.0))  # 1 mazo ≈ 50g
            
        display_qty = f"{units_needed} {'Mazo' if units_needed == 1 else 'Mazos'}"
        market_qty = units_needed
        market_unit = "Mazo"
        was_unitarized = True
        confidence = 0.90

    # ═══════════════════════════════════════════════════════════════
    # BLOQUE 2: Conversión Matemática → Unidades Físicas
    # Para items vendidos por unidad con density_g_per_unit (frutas,
    # vegetales unitarios, huevos, plátanos, etc.)
    # Excluye carnes/quesos (se venden por peso en RD).
    # Guard anti-absurdo: items muy pequeños (vainitas 10g, molondrones 15g)
    # con conteos altos → mejor por libra.
    # ═══════════════════════════════════════════════════════════════
    if not was_unitarized and weight_in_lbs > 0 and density_per_u and not re.search(r'lata|envase|ud|frasco|pote|caja', unit_str.lower()):
        if not is_meat_seafood and not is_cheese:
            g_total = weight_in_lbs * 453.592
            raw_count = g_total / density_per_u
            floor_count = math.floor(raw_count)
            frac = raw_count - floor_count
            # Anti-desperdicio: si necesitas <10% de una unidad extra, no comprarla
            if frac <= ANTI_WASTE_THRESHOLD and floor_count >= 1:
                units_count = floor_count
            else:
                units_count = max(1, math.ceil(raw_count))
            
            # Guard: "20 vainitas" no tiene sentido → "1/2 lb de vainitas"
            # También, si la densidad es extremadamente baja (<= 15g) como vainitas, molondrones, fresas,
            # nunca debería venderse por unidad a menos que sea ajo (que se calcula por cabeza/diente).
            is_absurd = (units_count > 6 and density_per_u < 50) or (density_per_u <= 15 and "ajo" not in n_lower)
            
            if not is_absurd:
                if is_native_weighable:
                    # Enfoque Híbrido Priorizado a Peso: "1 lb (~5 Uds)"
                    lbs_for_weighable = (units_count * density_per_u) / 453.592
                    whole, fraction_str = _lbs_to_market_fraction(lbs_for_weighable)

                    if whole == 0 and not fraction_str:
                        # Si es muy ligero, forzar a "1/4 lb" o unidades puras si es excepcionalmente pequeño
                        unit_text = "Ud." if units_count == 1 else "Uds."
                        display_qty = f"{units_count} {unit_text}"
                        market_qty = units_count
                        market_unit = "Ud."
                        sku_label = None
                    else:
                        # [P0-3] `market_qty` SIEMPRE float. El display
                        # fraccional ("1 ½ lbs") vive en `display_qty`. Antes
                        # se asignaba string ("1 1/2") creando tipo mixto que
                        # rompía consumers numéricos.
                        frac_decimal = _FRACTION_DECIMAL.get(fraction_str, 0.0)
                        if whole > 0 and fraction_str:
                            weight_lbl = f"{whole} {to_unicode_fraction(fraction_str)} lbs"
                            market_qty_val = float(whole) + frac_decimal
                        elif whole > 0:
                            weight_lbl = f"{whole} {'lb' if whole == 1 else 'lbs'}"
                            market_qty_val = float(whole)
                        else:
                            weight_lbl = f"{to_unicode_fraction(fraction_str)} lb"
                            market_qty_val = frac_decimal

                        # Limpiamos visualmente
                        display_qty = f"{weight_lbl} (~{units_count} {'Ud.' if units_count == 1 else 'Uds.'})"
                        market_qty = market_qty_val
                        market_unit = "lb" if whole <= 1 and not (whole==1 and fraction_str) else "lbs"
                        sku_label = None
                        
                    was_unitarized = True
                    confidence = 0.85

                else:
                    unit_text = "Ud." if units_count == 1 else "Uds."
                    if is_native_cabeza or re.search(r'\bajo\b', n_lower): unit_text = "Cabeza" if units_count == 1 else "Cabezas"
                    
                    if is_native_countable:
                        # Sin sufijo para "plátanos" o "huevos"
                        sku_label = None
                    else:
                        # Mega Frutas y demás tendrán su etiqueta de peso estimado (~X lbs)
                        approx_weight_label = _sku_size_label(density_per_u * units_count)
                        if approx_weight_label:
                            sku_label = f"~{approx_weight_label}"
                        else:
                            sku_label = None

                    display_qty = f"{units_count} {unit_text}"
                    # [P1-PDF-5] Mega-frutas: el sku_label ya inicia con "~"
                    # (peso TOTAL aproximado). Helper añade " total" cuando
                    # count > 1 → "10 Uds. (~33.1 lbs total)" en vez del
                    # ambiguo "(~33.1 lbs)".
                    if sku_label:
                        suffix = _format_pkg_suffix(units_count, sku_label)
                        if suffix:
                            display_qty += f" {suffix}"

                    market_qty = units_count
                    market_unit = "Ud." if "Cabeza" not in unit_text else "Cabeza"
                    was_unitarized = True
                    confidence = 0.85

    # ═══════════════════════════════════════════════════════════════
    # BLOQUE 3: Escala Mercado Dominicano para Pesos
    # Para carnes, quesos, y cualquier item sin envase estándar.
    # Redondea a fracciones de libra reales: 1/4, 1/2, 3/4
    # ═══════════════════════════════════════════════════════════════
    if not was_unitarized and weight_in_lbs > 0:
        if weight_in_lbs < 0.23:
            # Mínimo comprable en colmado dominicano: 1/4 lb
            display_qty = "¼ lb"
            # [P0-3] float (antes "1/4" string).
            market_qty = 0.25
            market_unit = "lb"
            confidence = 0.75
        else:
            whole = math.floor(weight_in_lbs)
            frac = weight_in_lbs - whole
            fraction_str = ""

            if frac < 0.15: fraction_str = ""
            elif frac <= 0.35: fraction_str = "1/4"
            elif frac <= 0.65: fraction_str = "1/2"
            elif frac <= 0.85: fraction_str = "3/4"
            else:
                fraction_str = ""
                whole += 1

            # [P0-3] `market_qty` SIEMPRE float. El display fraccional Unicode
            # vive en `display_qty`. Antes este bloque emitía strings tipo
            # "1 1/2" / "3/4", causando tipo mixto que rompía consumers
            # numéricos (Restock, pricing, agregadores).
            frac_decimal = _FRACTION_DECIMAL.get(fraction_str, 0.0)
            if whole > 0 and fraction_str:
                display_qty = f"{whole} {to_unicode_fraction(fraction_str)} lbs"
                market_qty = float(whole) + frac_decimal
                market_unit = "lbs"
            elif whole > 0:
                display_qty = f"{whole} {'lb' if whole == 1 else 'lbs'}"
                market_qty = float(whole)
                market_unit = "lb" if whole == 1 else "lbs"
            elif fraction_str:
                display_qty = f"{to_unicode_fraction(fraction_str)} lb"
                market_qty = frac_decimal
                market_unit = "lb"
            else:
                display_qty = "¼ lb"
                market_qty = 0.25
                market_unit = "lb"
            confidence = 0.75

    # ═══════════════════════════════════════════════════════════════
    # BLOQUE 4: Fallback para formatos crudos sin peso aplicable
    # ═══════════════════════════════════════════════════════════════
    if not display_qty:
        if raw_qty > 0:
            if unit_str in ['unidad', 'unidades', 'paquete', 'paquetes', 'lata', 'latas', 'sobre', 'sobres', 'frasco', 'pote', 'potes', 'cartón', 'carton', 'botella', 'botellas', 'envase', 'envases', 'funda', 'fundas', 'fundita', 'funditas', 'mazo', 'mazos', 'cabeza', 'cabezas']:
                q_rounded = f"{math.ceil(raw_qty)}"
            else:
                q_rounded = f"{raw_qty:.2f}".rstrip('0').rstrip('.')
            if q_rounded == "": q_rounded = "1"
            
            if unit_str == 'unidad' or unit_str == 'unidades':
                if db_container:
                     # [P3-OLIVE-RENDER-FIX · 2026-05-16] Detectar "X items
                     # pequeños que caben en N envases" (aceitunas 5g/oliva en
                     # frasco 340g; almendras 1.2g/almendra en bolsa 113g; etc).
                     # Sin esto, el LLM emite "X aceitunas" (unidades
                     # individuales) y BLOQUE 4 lo renderiza como "X frascos"
                     # asumiendo 1 unidad = 1 envase. Bug observable PDF
                     # 2026-05-16 plan 4cc91584: "Aceitunas: 24/47/68 frascos
                     # (12 oz c/u)" para ciclos 7d/15d/30d × 1 persona = 18
                     # a 51 lbs de aceitunas. Realidad: 1 frasco basta.
                     #
                     # Heurística: density_g_per_unit < 50g (unit individual
                     # ligero) Y container_weight_g >= density × 5 (container
                     # contiene >=5 unidades). Convertir a gramos totales y
                     # dividir por container para obtener N envases reales.
                     # Items afectados (positivamente): aceitunas, almendras,
                     # nueces, semillas, pasas. Items NO afectados (siguen
                     # comportamiento legacy): yogurt (density por pote),
                     # leche (density por cartón), huevos (density por carton
                     # o por huevo individual donde container_weight ya está
                     # alineado), etc.
                     _small_unit_in_big_container = (
                         density_per_u and density_per_u < 50.0
                         and db_container_weight_g
                         and db_container_weight_g >= density_per_u * 5.0
                     )
                     if _small_unit_in_big_container:
                         try:
                             _raw_qty_num = float(q_rounded) if '.' in q_rounded else int(q_rounded)
                             _total_g = _raw_qty_num * float(density_per_u)
                             _container_count = max(1, math.ceil(_total_g / float(db_container_weight_g)))
                             # CRÍTICO: reescribir q_rounded para que line ~2110
                             # `market_qty = float(q_rounded)` recoja el container
                             # count (1), no el raw count (68). Sin esto, el
                             # fallthrough abajo OVERRIDE market_qty con el valor
                             # crudo en unidades y rompe el escalamiento
                             # downstream (cost calc, restock, etc.).
                             q_rounded = str(_container_count)
                             display_qty = (
                                 f"{_container_count} "
                                 f"{get_plural_unit(_container_count, db_container)}"
                             )
                             market_qty = float(_container_count)
                             market_unit = db_container
                             sku_label = _sku_size_label(db_container_weight_g, db_container)
                             if sku_label:
                                 suffix = _format_pkg_suffix(_container_count, sku_label)
                                 if suffix:
                                     display_qty += f" {suffix}"
                             was_unitarized = True
                             confidence = 0.95
                         except (TypeError, ValueError):
                             # Fallback al comportamiento legacy si algo
                             # falla (ej. q_rounded no parsea).
                             _small_unit_in_big_container = False
                     if not _small_unit_in_big_container:
                         display_qty = f"{q_rounded} {get_plural_unit(float(q_rounded) if '.' in q_rounded else int(q_rounded), db_container)}"
                         market_unit = db_container
                         sku_label = _sku_size_label(db_container_weight_g, db_container)
                         # [P1-PDF-5] Sufijo c/u para fallback. `q_rounded` es
                         # str numérico — el helper coerce vía int(float()).
                         if sku_label:
                             suffix = _format_pkg_suffix(q_rounded, sku_label)
                             if suffix:
                                 display_qty += f" {suffix}"
                else:
                     display_qty = f"{q_rounded} {'Ud.' if str(q_rounded) == '1' else 'Uds.'}"
                     market_unit = "Ud."
            else:
                display_qty = f"{q_rounded} {get_plural_unit(raw_qty, unit_str)}"
                
            market_qty = float(q_rounded) if '.' in q_rounded else int(q_rounded)
        else:
            display_qty = "Al gusto"
            market_qty = 0
            market_unit = "Al gusto"

    # [CABEZA-GUARD] Items que NUNCA deben llevar "Cabeza" como unidad de mercado.
    # Mi test directo de `apply_smart_market_units` para zanahoria/tomate/pimiento
    # con density_g_per_unit poblado retorna "lbs" correctamente, pero en producción
    # el PDF mostraba "X Cabezas (~Y Uds.)" para esos items. El path que dispara el
    # bug es probablemente el Bloque 1 (Data-Driven) cuando master_item tiene
    # `market_container='cabeza'` para un veg que no es nativo cabeza — o un cache
    # de display de un build viejo. Guard defensivo case-insensitive: si llegamos
    # al final con "cabeza/Cabezas/cabezas" en display_qty o market_unit y el name
    # matchea la lista excluida, reconstruimos como peso (lbs) usando weight_in_lbs
    # y density del master.
    _has_cabeza = (
        bool(re.search(r'\bcabezas?\b', display_qty, re.IGNORECASE))
        or (isinstance(market_unit, str) and 'cabeza' in market_unit.lower())
    )
    if _has_cabeza and _NON_CABEZA_NAMES_RE.search(name):
        logging.warning(
            f"[CABEZA-GUARD] '{name}' tenía display_qty='{display_qty}' "
            f"(Cabezas inválido para este vegetal). Reconstruyendo como peso."
        )
        _lbs = weight_in_lbs
        _whole = math.floor(_lbs)
        _frac = _lbs - _whole
        _frac_str = ""
        if _frac < 0.15: _frac_str = ""
        elif _frac <= 0.35: _frac_str = "1/4"
        elif _frac <= 0.65: _frac_str = "1/2"
        elif _frac <= 0.85: _frac_str = "3/4"
        else:
            _frac_str = ""
            _whole += 1
        if _whole > 0 and _frac_str:
            _weight_lbl = f"{_whole} {to_unicode_fraction(_frac_str)} lbs"
            market_qty = float(_whole) + _FRACTION_DECIMAL.get(_frac_str, 0.0)
        elif _whole > 0:
            _weight_lbl = f"{_whole} {'lb' if _whole == 1 else 'lbs'}"
            market_qty = float(_whole)
        else:
            _weight_lbl = f"{to_unicode_fraction(_frac_str or '1/4')} lb"
            market_qty = _FRACTION_DECIMAL.get(_frac_str or '1/4', 0.25)
        market_unit = "lbs" if market_qty > 1 else "lb"
        # Subtítulo "(~N Uds.)" si tenemos density del master.
        _density = (master_item or {}).get('density_g_per_unit')
        try:
            _density = float(_density) if _density else 0.0
        except (TypeError, ValueError):
            _density = 0.0
        if _density > 0 and weight_in_lbs > 0:
            _units_count = max(1, math.ceil(weight_in_lbs * 453.592 / _density))
            display_qty = f"{_weight_lbl} (~{_units_count} {'Ud.' if _units_count == 1 else 'Uds.'})"
        else:
            display_qty = _weight_lbl
        # Limpiar sku_label para que el bloque post-format no anexe sufijos del
        # path corrupto (ej. "(150g c/u)" del market_container='cabeza' viejo).
        sku_label = None
        # [P1-SKU-COVER-HONESTY-R1 · 2026-08-02] El guard reconstruye el ítem como PESO (lbs)
        # pero dejaba `_pkg_size_g` con el tamaño del envase 'cabeza' descartado — el bloque de
        # `pkg_cover_ratio` (más abajo) seguía viendo `_pkg_size_g` no-None y mezclaba unidades
        # (gramos del envase 'cabeza' vs `weight_in_lbs` reconstruido), produciendo un
        # `pkg_cover_ratio` sin sentido y disparando la nota de under-buy con un número falso.
        # Reproducido: Zanahoria 900g con `market_container='cabeza'` → "2 lbs · alcanza ~2 de 7
        # días — recompra" (cover 0.333) y encima inflaba `_item_cycle_repurchases` a ~12.9
        # recompras contra un plano de 4.29. Limpiar junto con `sku_label`.
        _pkg_size_g = None
        confidence = 0.80  # Bajamos confianza: hubo path bug detectado.

    # ═══ Formato Final ═══
    if "Al gusto" in display_qty or "Pizca" in display_qty:
        final_str = f"{display_qty} de {name}"
    elif market_unit in ["Ud.", "Uds.", "Cabeza", "Cabezas", "Mazo", "Mazos"]:
        final_str = f"{display_qty} {name}"
    else:
        final_str = f"{display_qty} de {name}"

    final_str = final_str.replace(" de de ", " de ")

    # [P0-3] `market_qty` ahora SIEMPRE es numérico (BLOQUES 1-4 emiten int/float;
    # los strings fraccionales se eliminaron). Antes este cast intentaba normalizar
    # un tipo mixto. Ahora el `round(..., 2)` simplemente preserva precisión de
    # display y se le asigna un valor float defensivo si por alguna razón llegara
    # un tipo inesperado (LLM hallucinations, futuro consumer sub-clase, etc.).
    if isinstance(market_qty, (int, float)):
        formatted_market_qty = round(float(market_qty), 2)
    else:
        # Defensa: tipo inesperado → forzar a float vía parser; si falla → 0.0.
        try:
            formatted_market_qty = round(float(market_qty), 2)
        except (TypeError, ValueError):
            formatted_market_qty = 0.0

    def _parse_market_qty(mq):
        if isinstance(mq, (int, float)):
            return float(mq)
        if isinstance(mq, str) and '/' in mq:
            try:
                parts = mq.strip().split()
                if len(parts) == 2:
                    num, den = parts[1].split('/')
                    return float(parts[0]) + float(num)/float(den)
                else:
                    num, den = mq.strip().split('/')
                    return float(num)/float(den)
            except (ValueError, IndexError, ZeroDivisionError, TypeError):
                return 0.0
        return 0.0

    numeric_qty = _parse_market_qty(formatted_market_qty)

    # Enforcement de mínimos comprables interactuando con reglas culturales
    if numeric_qty > 0 and market_unit.lower() in MARKET_MINIMUMS:
        min_qty = MARKET_MINIMUMS[market_unit.lower()]
        
        # Nivel de Producción: Carnes crudas mínimo 1/2 libra (excepto embutidos/deli)
        if market_unit.lower() in ['lb', 'lbs'] and is_meat_seafood and not re.search(r'\b(jam[oó]n|tocineta|bacon|salami|longaniza)\b', n_lower):
            min_qty = 0.5
            
        if numeric_qty < min_qty:
            formatted_market_qty = min_qty
            market_qty = min_qty
            if market_unit.lower() in ['lb', 'lbs']:
                frac_str = ""
                whole_min = math.floor(min_qty)
                frac_min = min_qty - whole_min
                if abs(frac_min - 0.25) < 0.1: frac_str = "1/4"
                elif abs(frac_min - 0.5) < 0.1: frac_str = "1/2"
                elif abs(frac_min - 0.75) < 0.1: frac_str = "3/4"
                
                if whole_min > 0 and frac_str: display_qty = f"{whole_min} {to_unicode_fraction(frac_str)} lbs"
                elif whole_min > 0: display_qty = f"{whole_min} {'lb' if whole_min == 1 else 'lbs'}"
                elif frac_str: display_qty = f"{to_unicode_fraction(frac_str)} lb"
                else: display_qty = f"{min_qty} lb"

                # [P0-3] Antes este bloque "resincronizaba" `formatted_market_qty`
                # a un string fraccional ("0 1/4" / "1 1/2") tras el bump de
                # MARKET_MINIMUMS, contradiciendo el `formatted_market_qty = min_qty`
                # de la línea 1048 (que dejaba float). Resultado: tipo mixto que
                # rompía el frontend Restock al persistir a `user_inventory`.
                # Ahora preservamos `min_qty` (float) — el display fraccional
                # ya está cubierto por `display_qty` arriba.

            else:
                display_qty = f"{int(min_qty)} {market_unit}"
                
            if market_unit.lower() in ["ud.", "uds.", "cabeza", "cabezas", "mazo", "mazos"]:
                final_str = f"{display_qty} {name}"
            else:
                final_str = f"{display_qty} de {name}"

    # Preservar la cadena híbrida construida a la perfección (ej: "1/2 lb (~5 Uds.)")
    # El código antiguo sobreescribía esta variable robando inteligencia.
    display_qty_final = display_qty

    # Nivel de Producción: Si logramós extraer un sku_size_label útil (tamaño paquete o aprox peso), anexarlo
    # [P1-PDF-5] `_has_pkg_suffix` reconoce las 3 variantes (legacy `(label)`,
    # nueva `(label c/u)`, y `(label total)`) → no duplica el sufijo si un
    # bloque previo ya lo añadió. `market_qty` es la fuente de count para
    # el path `MARKET_MINIMUMS-bumped` que cae acá sin haber añadido sufijo.
    if sku_label and not _has_pkg_suffix(display_qty_final, sku_label):
        suffix = _format_pkg_suffix(market_qty, sku_label)
        if suffix:
            display_qty_final = f"{display_qty_final} {suffix}"

    # [P0-2] `market_qty` puede ser un string fraccional ("1 1/2", "3/4", "1/4")
    # construido por los bloques 2/3 para preservar fidelidad al mercado dominicano,
    # pero el frontend antes hacía `parseFloat(item.market_qty)` directamente:
    # `parseFloat("1 1/2") → 1` y `parseFloat("1/2") → 0`, subdimensionando el
    # delta lista↔nevera. Ahora SIEMPRE exponemos `market_qty_numeric: float`
    # con el valor real (re-parseado tras MARKET_MINIMUMS, que muta
    # `formatted_market_qty`). El frontend prefiere este campo; `market_qty`
    # sigue siendo el string display-friendly para no romper consumers legacy.
    market_qty_numeric_final = _parse_market_qty(formatted_market_qty)

    # [P1-CAPPED-STAPLE-HONESTY · 2026-07-26] Si un cap de realismo recortó este alimento, DECIRLO.
    #
    # Caso medido (plan vivo 1070ceb1): el agregador calculó 3.547 g de atún para el ciclo de 30
    # días —correcto, casi exacto a la necesidad real de las recetas— y `P6-CANNED-PROTEIN-CAP` lo
    # recortó a 736 g (4 latas) porque nadie compra 22 latas de una vez. El recorte es RAZONABLE.
    # Lo que no es razonable es que esas 4 latas aparezcan bajo el encabezado «DESPENSA DEL MES —
    # COMPRA UNA SOLA VEZ · Cantidad calculada para todo el periodo»: cubren **5,5 días**. El usuario
    # compra la lista, cree que está abastecido el mes y se queda sin atún en menos de una semana,
    # sin un solo aviso.
    #
    # El dato del cap YA existía (`_CAPS_APPLIED_LAST_RUN`) pero sólo lo consumía el coherence
    # guard: nunca llegaba al item, así que la pantalla no podía saberlo. Se adjunta aquí —
    # `capped_by`/`capped_pre_g`/`capped_post_g` para tooling, y el sufijo en `display_qty` para que
    # la UI y el PDF sean honestos sin tocar el frontend (ambos renderean ese string tal cual).
    #
    # NO se sube la cantidad: comprar 22 latas es peor consejo que comprar 4 y recomprar. Lo que se
    # arregla es la MENTIRA, no el número. tooltip-anchor: P1-CAPPED-STAPLE-HONESTY
    _cap_hit = None
    if CAPPED_STAPLE_HONESTY:
        try:
            # ⚠️ `strip_accents` NO es un nombre de módulo en este archivo — se importa DENTRO de
            # cada función (líneas 688, 709, 753, 980). La primera versión de este bloque lo usaba
            # como global: lanzaba NameError, el `except` de abajo se lo tragaba, y el resultado era
            # que `get_caps_applied_last_run()` NUNCA se llamaba. Medido con un spy sobre un plan
            # real: `apply_smart_market_units` corría 50 veces y la lectura de caps 0.
            from constants import strip_accents as _sa_cap
            _nlow = _sa_cap(str(name).lower()).strip()
            for _c in get_caps_applied_last_run():
                if _c.get("food_lower", "").strip() and (
                        _sa_cap(_c["food_lower"]).strip() == _nlow):
                    if float(_c.get("post_value") or 0) < float(_c.get("pre_value") or 0):
                        _cap_hit = _c
                    break
        except Exception as _cap_e:
            # NO callar: un `except: pass` aquí fue justo lo que convirtió el NameError en un
            # no-op invisible y me hizo reportar como arreglado algo que no hacía nada.
            logging.warning(f"[P1-CAPPED-STAPLE-HONESTY] lookup de cap falló para "
                            f"'{str(name)[:40]}': {type(_cap_e).__name__}: {str(_cap_e)[:120]}")
            _cap_hit = None

    # [P1-VEG-BACKFILL-HONESTY · 2026-08-02] Backstop: si NINGÚN cap real explicó el déficit
    # (`_cap_hit` sigue `None` — no pisamos un cap real, decisión #1 del riesgo), comparar la
    # cantidad final resuelta (`base_qty`, en gramos) contra lo que las recetas piden
    # (`text_demand_g`, mismo parse que usa el guard). Caso medido en prod: espárragos no vive en
    # `_VEG_PER_WEEK_PER_PERSON` (P5-VEG-CAP) ni en ningún otro cap por categoría — el déficit
    # llegaba mudo (`capped_by=null`) aunque una sola cena agotara el 103% de la compra semanal.
    # Reusa `_coherence_base_fields` (misma fuente que `result["base_qty"]` más abajo) en vez de
    # re-derivar la conversión: una sola definición de "cuánto se está comprando en gramos".
    #
    # [P1-VEG-BACKFILL-HONESTY · 2026-08-03 · ronda de revisión] Este backstop ALIMENTA el mismo
    # `_cap_hit`/nota que P1-CAPPED-STAPLE-HONESTY — así que respeta el MISMO kill switch
    # `CAPPED_STAPLE_HONESTY` ("Flip a False si el copy molesta; el número NO cambia con el knob,
    # sólo se deja de decir"). Reproducido pre-fix: con el knob en `False` el lookup de arriba ni
    # corre (queda dentro de su propio `if CAPPED_STAPLE_HONESTY:`), pero este bloque nuevo corría
    # de todas formas y la nota "alcanza..." seguía saliendo — un operador que apaga el knob en un
    # incidente no lograba callarla. Simétrico con el cap REAL: cuando el knob está OFF, ni el
    # cap real ni este sintético dejan `capped_by` en el ítem (mismo invariante "sin capped_by ⟺
    # knob off o nada que reportar" para ambos caminos).
    _base_fields = _coherence_base_fields(raw_qty, unit_str, weight_in_lbs)
    if _cap_hit is None and text_demand_g is not None and CAPPED_STAPLE_HONESTY:
        try:
            _td_g = float(text_demand_g)
        except (TypeError, ValueError):
            _td_g = 0.0
        if _td_g > 0:
            _bq = _base_fields.get("base_qty")
            _bu = str(_base_fields.get("base_unit") or "").strip().lower()
            _bq_g = None
            if _bq is not None:
                if _bu in ("g", "gr", "gramo", "gramos"):
                    _bq_g = float(_bq)
                elif _bu in ("lb", "lbs"):
                    _bq_g = float(_bq) * _LB_TO_G
            if _bq_g is not None and _bq_g < QTY_SHORTFALL_NOTE_MIN * _td_g:
                _cap_hit = {
                    "food_lower": str(name).lower(),
                    "reason": QTY_RECONCILE_SYNTHETIC_REASON,
                    "pre_value": _td_g,
                    "post_value": _bq_g,
                }

    result = {
        "name": name,
        "market_qty": formatted_market_qty,
        "market_qty_numeric": market_qty_numeric_final,
        "market_unit": market_unit,
        "display_qty": display_qty_final,
        "display_string": final_str,
        "confidence_score": confidence,
        "shelf_life_days": master_item.get("shelf_life_days") if master_item else None,
        # [P1-COHERENCE-BASE-QTY · 2026-07-26] Cantidad ANTES de convertir a unidad de mercado.
        #
        # El coherence guard compara la suma de las recetas contra la lista emparejando por
        # (alimento, UNIDAD). Las recetas hablan en g/taza/cda; la lista, tras esta función,
        # solo guardaba `market_qty`/`market_unit` — pote, sobre, paquete, mazo. Sin unidad
        # común el emparejamiento fallaba y `expected_qty` salía 0.0 para TODOS:
        #
        #     Miel     pote     esperado=0.0  lista=1.0  ratio=inf  -> unknown
        #     Orégano  sobre    esperado=0.0  lista=1.0  ratio=inf  -> unknown
        #
        # Medido en el plan vivo 01d63a5b: 41 divergencias, 39 de ellas `unknown` por esta
        # causa. El guard no estaba detectando incoherencias reales — estaba comparando dos
        # idiomas distintos y llamando "desconocido" al resultado. El propio docstring de
        # `compare_expected_vs_aggregated` ya avisaba: el caller debe comparar ANTES de esta
        # conversión. Preservar la base aquí es la forma no-invasiva de cumplirlo.
        #
        # Aditivo: nadie que lea `market_*` se entera.
        #
        # ⚠️ HAY DOS RUTAS DE ENTRADA y solo una trae `raw_qty`:
        #   - por UNIDADES  (línea ~9858): `apply_smart_market_units(name, 0.0, u, q, ...)`
        #   - por PESO      (línea ~9816): `apply_smart_market_units(name, lbs, 'lb', 0.0, ...)`
        # La primera versión de este bloque solo miraba `raw_qty`, así que en la ruta de peso
        # guardaba 0.0 y el extractor lo descartaba: medido sobre el plan vivo fbe53a5b,
        # **2 de 48 items** tenían base. El fix quedaba inerte justo donde más items hay.
        # Es el mismo modo de fallo que dejó muerto P1-CAPPED-STAPLE-HONESTY: código presente,
        # efecto ausente, y solo se ve midiendo el resultado.
        # tooltip-anchor: P1-COHERENCE-BASE-QTY
        # [P1-VEG-BACKFILL-HONESTY · 2026-08-02] Reusa `_base_fields` (ya computado arriba para el
        # backstop de texto) en vez de llamar `_coherence_base_fields` una segunda vez — misma
        # función pura, mismo resultado, sin doble trabajo.
        **_base_fields,
    }
    if _cap_hit:
        try:
            _pre = float(_cap_hit.get("pre_value") or 0)
            _post = float(_cap_hit.get("post_value") or 0)
            result["capped_by"] = _cap_hit.get("reason")
            # [P1-VEG-BACKFILL-HONESTY · 2026-08-03 · review final] Dos canales, no uno.
            #
            # `capped_pre`/`capped_post` son el canal de los caps REALES (`_CAPS_APPLIED_LAST_RUN`):
            # el guard de coherencia los usa para SUSTITUIR la cantidad comprada por la que el
            # agregador calculó antes del tope, y eso sólo es legítimo porque ese número es un
            # cómputo INDEPENDIENTE del lado esperado (si el agregador se equivoca, la divergencia
            # sigue apareciendo — «no es un mute», P1-COHERENCE-CAPPED-PRE).
            #
            # El déficit SINTÉTICO no cumple esa condición: su `pre_value` ES la demanda de las
            # recetas, la misma `expected_sum_from_recipes(..., multiplier=effective_multiplier)`
            # con la que el guard construye el lado esperado. Escribirlo en `capped_pre` hacía que
            # el guard comparara el esperado contra sí mismo y no viera NINGUNA divergencia de
            # magnitud (reproducido: recetas 2000 g / lista 1000 g → `[]`). Va por claves propias.
            # tooltip-anchor: P1-QTY-SHORTFALL-OWN-CHANNEL
            _es_sintetico = (_cap_hit.get("reason") == QTY_RECONCILE_SYNTHETIC_REASON)
            if _es_sintetico:
                result["shortfall_text_g"] = round(_pre, 1)
                result["shortfall_bought_g"] = round(_post, 1)
            else:
                result["capped_pre"] = round(_pre, 1)
                result["capped_post"] = round(_post, 1)
            # Fracción del ciclo que cubre lo comprado. Se expresa en % y no en días porque el cap
            # se aplica sobre la unidad del agregador (g/latas/paquetes) y no siempre hay días.
            _frac = (_post / _pre) if _pre > 0 else 0.0
            result["capped_cycle_fraction"] = round(_frac, 3)
            # [P1-COVERAGE-VS-PURCHASE · 2026-07-27] El aviso comparaba lo CAPADO contra lo
            # necesario e ignoraba lo que el usuario realmente COMPRA.
            #
            # El envase de mercado redondea hacia ARRIBA, y muchas veces por encima de lo que
            # hacía falta. Entonces el usuario leía "recompra" de algo que le sobra:
            #
            #   Yogurt   necesita 1480.7 g   compra 1960.0 g   -> avisaba "alcanza ~18/30"
            #   Cúrcuma  necesita   43.6 g   compra  113.4 g   -> avisaba "alcanza ~19/30"
            #   Puerro   necesita  207.7 g   compra  300.0 g   -> avisaba "alcanza ~7/30"
            #
            # (los ejemplos NO reproducen el sufijo literal a propósito: `test_p1_capped_staple_
            # honesty` cuenta sus apariciones en este bloque y un comentario que lo cite lo rompe —
            # ya pasó con P1-RICE-STEP-HONEST)
            #
            # Medido sobre 6 planes vivos: **6 de 39** avisos (15%) eran así, y el yogurt salía
            # 3 veces. Un aviso que manda recomprar lo que sobra es peor que no avisar: enseña al
            # usuario a ignorar el resto de avisos, que sí son ciertos.
            #
            # Se calcula lo comprado en gramos (envase × cantidad, o la propia unidad si ya es de
            # peso) y se calla el aviso cuando cubre lo necesario ANTES del tope. Fail-open: si no
            # se puede calcular —10 de 39 no traen datos suficientes— se conserva el aviso, que es
            # el comportamiento de hoy.
            _tapa_avisa = not _purchase_covers_need(result, _pre)
            if not _tapa_avisa:
                result["coverage_ok_by_package"] = True
            # [P1-CAPPED-STAPLE-HONESTY · 2026-08-03 · review final] El denominador era un literal
            # fijo de un mes, mientras la nota GEMELA de `pkg_cover_ratio` (60 líneas abajo, misma
            # función) ya es paramétrica con `cycle_days` desde P1-SKU-COVER-HONESTY-R1. En una
            # lista SEMANAL el usuario leía un ciclo mensual sobre una compra de 7 días, al lado de
            # otra línea del MISMO PDF que declaraba el ciclo correcto; en la quincenal las dos
            # notas se contradecían con dos ciclos distintos. Y el numerador tampoco era cierto:
            # con cobertura 0,76 de la SEMANA la nota anunciaba tres semanas de autonomía.
            # `cycle_days` es parámetro de esta función (default 7) y la ronda 2 de
            # P1-SKU-COVER-HONESTY lo cableó a los ~26 callsites duration-aware justamente para
            # esto. Se conserva `round` (no `math.floor` como la gemela): con el gate `_frac < 0.9`
            # el redondeo nunca puede llegar a `cycle_days` (haría falta `_frac ≥ 1-0,5/cycle_days`,
            # o sea ≥0,93 con 7 días), así que la nota no puede emitir el absurdo «~N de N».
            #
            # (el comentario NO reproduce el sufijo literal a propósito: `test_p1_capped_staple_
            # honesty` cuenta sus apariciones en este bloque y un comentario que lo cite lo rompe —
            # ya pasó con P1-RICE-STEP-HONEST, y volvió a pasar al escribir ESTE comentario)
            _dias_alcanza = max(1, int(round(_frac * cycle_days)))
            if _frac and _frac < 0.9 and _tapa_avisa:
                result["display_qty"] = (
                    f"{display_qty_final} · alcanza ~{_dias_alcanza} de {cycle_days} días — recompra")
                result["display_string"] = (
                    f"{final_str} (alcanza ~{_dias_alcanza} de {cycle_days} días — recompra)")
        except Exception:
            pass
    if sku_label:
        result["sku_size_label"] = sku_label
    # [P1-BRAND-SIZE-FILTER · 2026-07-06] Tamaño del envase elegido → el picker de
    # marcas (SupermarketBrands) filtra variantes del súper al tamaño de la lista.
    if _pkg_size_g:
        try:
            result["package_grams"] = round(float(_pkg_size_g), 2)
        except (TypeError, ValueError):
            pass
        # [P1-CYCLE-REPURCHASE-HONEST · 2026-07-25] Cuántas veces cabe la NECESIDAD dentro de lo
        # que obliga a comprar el envase mínimo. Es el único punto del pipeline donde conviven las
        # dos cifras; más abajo el ítem ya solo lleva el envase y la necesidad se perdió.
        #
        # Lo consume el costo del ciclo: hoy multiplica TODOS los perecederos por las semanas del
        # ciclo (×4,29 en mensual) como si cada uno se re-comprara cada semana. La funda mínima de
        # manzanas es de 3 lb y el plan usa ~1 manzana por semana: se cobraba 5 veces una funda que
        # dura el mes. Medido en el plan vivo 1d3c6643 — manzana 6,5×, mozzarella 4,9×, avena 4,3×,
        # ricotta 3,7× de sobre-oferta.
        try:
            _need_g = float(weight_in_lbs) * 453.592
            _pkg_total_g = float(_pkg_size_g) * max(1.0, float(market_qty_numeric_final or 1))
            if _need_g > 0 and _pkg_total_g > 0:
                result["pkg_cover_ratio"] = round(_pkg_total_g / _need_g, 3)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
        # [P1-SKU-COVER-HONESTY · 2026-08-02] `pkg_cover_ratio` se calculaba y persistía pero
        # nadie lo consumía para avisar under-buy (solo sobre-cobertura ≥2×, P1-OVERCOVER-LABEL).
        # Medido en prod: 18/22 planes con ≥1 ítem cover<0.9 sin nota. Mismo formato que
        # P1-CAPPED-STAPLE-HONESTY ("alcanza ~N de M días — recompra"). Si el ítem YA tiene
        # `capped_by` (P1-CAPPED-STAPLE-HONESTY), esa nota manda — no se duplica sufijo
        # (decisión #3).
        #
        # [P1-SKU-COVER-HONESTY-R1 · 2026-08-02] Dos correcciones de la ronda 1:
        #
        # (1) `M` ya NO es un `7` fijo. El multiplicador de ciclo (household ×
        # `cycle_qty_multiplier(duration)` × `base_duration_scale`) entra en `weight_in_lbs`
        # ANTES de esta función — así que en listas quincenal/mensual `pkg_cover_ratio` mide
        # contra la necesidad de 15/30 días, no de 7. Hardcodear "de 7 días" ahí leía "alcanza
        # ~5 de 7 días" sobre un arroz que en realidad dura ~21 de 30. `cycle_days` (parámetro
        # nuevo, default 7 — mismo comportamiento previo para callers que no lo pasan) es la M
        # correcta. Callers duration-aware deben pasarlo explícitamente (ver docstring del
        # parámetro); si no lo hacen, el default 7 preserva el comportamiento pre-existente
        # (correcto para listas semanales, la mayoría de callers hoy).
        #
        # (2) Se excluye cuando `_pkg_units_recounted` (P2-PACK-UNITS-MATCH): el conteo ahí se
        # deriva por UNIDADES reales del envase, no por gramos — el `pkg_cover_ratio` en gramos
        # (density del MASTER) deja de medir cobertura real y el único disparo vivo bajo el
        # default anterior (0.9) era justo este falso positivo (ratio 0.712 con conteo correcto).
        #
        # [P1-SKU-COVER-HONESTY-R2 · 2026-08-02] `round()` se contradecía a sí mismo: Leche
        # 1000g/cartón 946g (el caso original de SKU-OVERSHOOT-FIX), cover=0,946 < 0,95 (el
        # umbral subido en la ronda 1) → `round(7*0,946)=round(6,62)=7` → "alcanza ~7 de 7 días
        # — recompra": afirma cobertura COMPLETA (7 de 7) y en la misma frase pide recomprar.
        # Texto sin sentido para el usuario. `math.floor` en vez de `round` da 6 (0,946*7=6,62,
        # floor=6) — "alcanza ~6 de 7 días" es cierto y accionable (falta 1 día). Como defensa
        # adicional (no alcanzable hoy con `PKG_COVER_NOTE_MIN<1`, pero sí si ese knob se
        # relaja a 1.0 en el futuro): si el floor iguala o supera `cycle_days`, la cobertura es
        # efectivamente completa — no hay nada que avisar, se suprime la nota entera en vez de
        # emitir "~N de N".
        try:
            _cover = result.get("pkg_cover_ratio")
            if (_cover is not None and float(_cover) < PKG_COVER_NOTE_MIN
                    and not result.get("capped_by") and not _pkg_units_recounted):
                _dias_cubiertos = math.floor(cycle_days * float(_cover))
                if _dias_cubiertos < cycle_days:
                    _dias_cubiertos = max(1, _dias_cubiertos)
                    result["display_qty"] = (
                        f"{result['display_qty']} · alcanza ~{_dias_cubiertos} de {cycle_days} días — recompra")
                    result["display_string"] = (
                        f"{result['display_string']} (alcanza ~{_dias_cubiertos} de {cycle_days} días — recompra)")
        except (TypeError, ValueError):
            pass
    # [P1-BRAND-DEFAULT-PRESELECTED · 2026-07-06] producto del súper que la lista usa.
    if _pkg_product_id:
        result["brand_product_id"] = _pkg_product_id
    # [P1-PKG-DURATION-PRICING] Precio del envase real elegido (RD$/paquete). Consumido por
    # `_cost_from_market` para costear por tamaño (count × precio), cerrando el sobrecobro
    # de precio plano en staples con descuento por volumen.
    if market_pkg_price is not None:
        result["market_pkg_price_rd"] = market_pkg_price
    return result


_MEAL_AGG_EXCLUDED_KEYWORDS_CACHE: tuple[tuple[str, ...], str] | None = None


def _meal_aggregation_excluded_keywords() -> tuple[str, ...]:
    """[P2-4 · 2026-05-08] SSOT de keywords excluidos en agregación de comidas.

    Devuelve la tupla normalizada (lowercase, stripped, sin vacíos) de
    keywords que disparan `_should_skip_meal_for_aggregation`. Lee
    `MEALFIT_COHERENCE_EXCLUDED_MEAL_KEYWORDS` (comma-separated) con default
    `"suplemento"`. Cachea por valor crudo: si el env-var cambia entre
    invocaciones (test isolation, reload), recomputa.

    Histórico: hasta 2026-05-07, los 3 sitios (`expected_sum_from_recipes`,
    `get_shopping_list_delta`, extractor de facts) duplicaban inline
    `if "suplemento" in meal.get("meal", "").lower(): continue`. Si una
    rama añadía un keyword nuevo (ej. `"infusión"` en el aggregator pero no
    en el guard), capa B de coherencia reportaba divergencias falsas — el
    mismo patrón que causó el bug de caps_asymmetry. SSOT lo previene.
    """
    global _MEAL_AGG_EXCLUDED_KEYWORDS_CACHE
    # [P2-1 · 2026-05-08] `_knob_env_str` registra en `_KNOBS_REGISTRY` y devuelve
    # ya normalizado (lower+strip). El cache local se queda para evitar el split
    # de keywords en cada llamada hot-path; el registro al registry es idempotente.
    raw = _knob_env_str("MEALFIT_COHERENCE_EXCLUDED_MEAL_KEYWORDS", "suplemento")
    if _MEAL_AGG_EXCLUDED_KEYWORDS_CACHE is not None and _MEAL_AGG_EXCLUDED_KEYWORDS_CACHE[1] == raw:
        return _MEAL_AGG_EXCLUDED_KEYWORDS_CACHE[0]
    parts = tuple(
        kw.strip()
        for kw in raw.split(",")
        if kw.strip()
    )
    if not parts:
        parts = ("suplemento",)
    _MEAL_AGG_EXCLUDED_KEYWORDS_CACHE = (parts, raw)
    return parts


def _should_skip_meal_for_aggregation(meal: dict) -> bool:
    """[P2-4 · 2026-05-08] Único punto de decisión "saltar esta comida en
    agregación de ingredientes". Llamado por `expected_sum_from_recipes`
    (capa B coherence guard), `get_shopping_list_delta` (aggregator
    principal) y el extractor de facts. Garantiza simetría entre el lado
    "expected" y el lado "aggregated" del coherence guard."""
    if not isinstance(meal, dict):
        return True
    name = str(meal.get("meal", "")).lower()
    for kw in _meal_aggregation_excluded_keywords():
        if kw in name:
            return True
    return False


def shopping_source_days(plan_data) -> list:
    """[P0-SHOPPING-CYCLE-DAYS · 2026-08-22] SSOT de "desde qué días se agrega la lista".

    Une `_archived_days` (lo que el shift rodante ya podó) con `days` (la ventana viva).
    ANTES, builder y guard leían `plan_data["days"]` a pelo, y esa ventana ENCOGE con
    cada shift: en el plan real 2245eb45 los 3 días generados dan 48 alimentos y el
    último día superviviente da 25 — y esos 25 son EXACTAMENTE lo que quedó publicado
    tras el siguiente recálculo. El usuario marcó "ya compré la lista", su nevera nació
    como espejo de esa lista mutilada (una sola proteína: Huevo; sin cebolla; sin
    almidón básico) y el chunk siguiente murió contra el gate de despensa.

    Lo usan LOS DOS lados a propósito (`get_shopping_list_delta` y
    `expected_sum_from_recipes`): mientras el lado ESPERADO del coherence guard leyera
    el mismo `days` encogido que el lado COMPRADO, ambos se recortaban a la vez y la
    divergencia se cancelaba — medido: la telemetría del plan bajó de 31 divergencias a
    6 justo DESPUÉS de la amputación, o sea que mutilar la lista MEJORABA la métrica.

    Agregar más días NO infla la compra: el total es
    `Σ(ingredientes) × (7/num_days) × cycle_qty_multiplier` = `promedio_por_día ×
    días_del_ciclo`, invariante en `num_days`. Con más días el promedio es mejor
    estimador, no mayor.

    Acota la unión al ciclo VIVO porque `_archived_days` no se vacía ni al renovar
    (ver `chat_history_context.py:204`): sin el filtro, un plan renovado arrastraría a
    la lista los alimentos de la temporada anterior. Los días sin `date` se conservan
    (fail-open: perder menú es peor que arrastrarlo).

    Rollback sin redeploy: `MEALFIT_SHOPPING_SOURCE_INCLUDES_ARCHIVED=false` restaura
    la conducta previa (sólo `days`). Tooltip-anchor: P0-SHOPPING-CYCLE-DAYS.
    """
    if not isinstance(plan_data, dict):
        return []
    vivos = plan_data.get("days")
    vivos = [d for d in vivos if isinstance(d, dict)] if isinstance(vivos, list) else []

    if not _knob_env_bool("MEALFIT_SHOPPING_SOURCE_INCLUDES_ARCHIVED", True):
        return vivos

    archivados = plan_data.get("_archived_days")
    archivados = [d for d in archivados if isinstance(d, dict)] if isinstance(archivados, list) else []
    if not archivados:
        return vivos

    # Filtro de ciclo: fuera los días anteriores al arranque del plan vivo.
    _cycle = plan_data.get("cycle_start_date") or plan_data.get("grocery_start_date")
    if isinstance(_cycle, str) and len(_cycle) >= 10:
        _corte = _cycle[:10]
        archivados = [
            d for d in archivados
            if not (isinstance(d.get("date"), str) and len(d["date"]) >= 10 and d["date"][:10] < _corte)
        ]

    union = archivados + vivos

    # Techo por si el plan lleva meses acumulando archivados: nos quedamos con los MÁS
    # RECIENTES, que son los que describen el ciclo de compra actual.
    try:
        _tope = int(plan_data.get("total_days_requested") or 0)
    except (TypeError, ValueError):
        _tope = 0
    if _tope <= 0:
        _tope = 30
    if len(union) > _tope:
        union = union[-_tope:]
    return union


def expected_sum_from_recipes(plan_data: dict, *, apply_yield: bool = False, multiplier: float = 1.0,
                               apply_protein_yield: bool = False) -> dict:
    """[P1-shop-coh-1 · 2026-05-07] Suma esperada de ingredientes desde el plan.

    Recorre `plan_data["days"][*]["meals"][*]` aplicando el MISMO contrato de
    parseo que `aggregate_and_deduct_shopping_list` (línea 2244):
    `_parse_quantity(item, apply_yield_multiplier=apply_yield, apply_legumbres_yield_only=True)`,
    misma corrección "ola"/"olas" → "Cebolla", mismo skip de comidas con
    "suplemento" en el nombre, mismo fallback `ingredients_raw` → `ingredients`
    → `recipe.ingredients`.

    El propósito es exponer la suma teórica de las recetas para que un
    consumidor (Paso 3 del plan P1-shop-coh-1) la contraste contra la lista
    de compras agregada y detecte divergencias. NO aplica master_map ni la
    canonicalización por nombre (huevos/ñame/miel/ajo/pavo) — esa capa vive
    inline en el aggregator y debe aplicarse simétricamente a ambos lados
    desde el comparador, no aquí.

    Args:
        plan_data: dict con shape `{"days": [{"meals": [...]}, ...]}`.
        apply_yield: default False, espejo del aggregator (peso literal en
            ambos lados → delta plan↔inventario simétrico, ver P1-2).
        multiplier: [P1-C 2026-05-07] escala las cantidades crudas por el
            household multiplier (`calc_household_multiplier` cacheado en
            plan_data por P1-3). El aggregator escala internamente; sin esta
            simetría, comparar magnitudes producía ratios espurios. Default
            1.0 preserva el comportamiento v1 (presence/absence). Acepta
            int|float; valores inválidos (NaN/inf/<=0) se clampan a 1.0.
        apply_protein_yield: [P2-PROTEIN-YIELD-CANONICAL · 2026-08-03] espejo
            OBLIGATORIO del mismo flag en `aggregate_and_deduct_shopping_list`
            — cuando el caller (aggregator canónico o el guard) aplica yield
            1.35× a proteínas cocidas del lado COMPRADO, este lado ESPERADO
            debe aplicarlo también para no reventar la tolerancia del
            coherence guard (±10% default, 35% de brecha si un solo lado
            yieldea). Default `False` preserva el comportamiento previo.

    Returns:
        `{food_name: {canonical_unit: total_qty}}`. Vacío si no hay días.
    """
    if not isinstance(plan_data, dict):
        return {}
    # [P0-SHOPPING-CYCLE-DAYS · 2026-08-22] Mismo SSOT que el lado COMPRADO. Leer
    # `plan_data["days"]` a pelo dejaba este lado en `{}` tras el shift, y con el
    # esperado vacío NINGUNA ausencia podía producir divergencia `expected_only`:
    # el guard no podía ver que faltaba el pollo ni aunque faltara.
    days = shopping_source_days(plan_data)
    if not days:
        return {}

    try:
        _mult = float(multiplier)
        if math.isnan(_mult) or math.isinf(_mult) or _mult <= 0:
            _mult = 1.0
    except (TypeError, ValueError):
        _mult = 1.0

    aggregated = defaultdict(lambda: defaultdict(float))
    for day in days:
        if not isinstance(day, dict):
            continue
        for meal in day.get("meals") or []:
            if not isinstance(meal, dict):
                continue
            # [P2-4] SSOT: helper compartido con get_shopping_list_delta y
            # el extractor de facts. Evita drift entre los 3 sitios.
            if _should_skip_meal_for_aggregation(meal):
                continue
            ingredients = meal.get("ingredients_raw") or meal.get("ingredients") or []
            if not ingredients:
                recipe = meal.get("recipe")
                if isinstance(recipe, dict):
                    ingredients = recipe.get("ingredients") or []
            for ing in ingredients:
                if isinstance(ing, str):
                    raw = ing
                elif isinstance(ing, dict):
                    q = ing.get("quantity", 0)
                    u = ing.get("unit", "unidad")
                    n = ing.get("name") or ing.get("item_name") or ing.get("display_name") or "Desconocido"
                    if q > 0 or u in ("pizca", "al gusto", "cantidad necesaria", "chin", "toque", "chorrito"):
                        raw = f"{q} {u} de {n}"
                    else:
                        raw = n
                else:
                    continue
                if not raw or len(str(raw)) < 3:
                    continue
                qty, unit, name = _parse_quantity(
                    raw,
                    apply_yield_multiplier=apply_yield,
                    apply_legumbres_yield_only=True,
                    apply_protein_yield=apply_protein_yield,
                )
                if not name:
                    continue
                if name.lower() in ("ola", "olas"):
                    name = "Cebolla"
                aggregated[name][unit] += float(qty) * _mult

    return {name: dict(units) for name, units in aggregated.items()}


def _mirror_trip_window_expected(
    plan_result: dict,
    expected_raw: dict,
    *,
    mult: float,
    window_len: int,
    day_basis_applied: bool,
    apply_protein_yield: bool = False,
) -> dict:
    """[P1-TRIP-WINDOWED-PERISHABLES · 2026-08-02] Espejo del ventaneo en el lado ESPERADO.

    Reconstruye `expected_raw` con la MISMA partición que `_merge_trip_windowed_result`
    aplicó a la lista: perecederos desde los días del viaje activo (escalados
    `7/len(ventana)` — el mismo factor que el agregador), estables desde el plan completo.
    Un perecedero ausente de la ventana desaparece del set esperado: no está en la lista
    porque no se cocina esta semana, no porque falte.

    `day_basis_applied` replica la condicionalidad de P1-COHERENCE-DAY-BASIS: si esa
    normalización está apagada por knob, el lado esperado no lleva escala de días y el
    ventaneo tampoco debe llevarla (mirroring, no una heurística nueva).

    La clasificación reusa `_classify_perishability` con el ítem de la lista agregada como
    `master_item` (mismo fallback que `_build_hybrid_shopping_list`: el ítem ya trae
    `category`/`shelf_life_days` inyectados por P1-PDF-2). Sin match, cae a los hints por
    nombre — el mismo camino que el agregador usó para decidir.

    [P2-PROTEIN-YIELD-CANONICAL · 2026-08-03] `apply_protein_yield` se propaga al
    `expected_sum_from_recipes` de la ventana — mismo espejo que el resto del guard.
    """
    if not isinstance(expected_raw, dict) or not expected_raw:
        return expected_raw
    # `ignore_knob=True` [ronda 1]: re-derivar la ventana que la lista YA declara no es
    # una decisión de construcción — es leer su sello. Respetar el knob aquí rompía el
    # rollback (ver el bloque de cabecera del P-fix).
    window = active_trip_window_days(plan_result, window_len=window_len, ignore_knob=True)
    if not window:
        return expected_raw

    window_scale = (7.0 / float(len(window))) if day_basis_applied else 1.0
    expected_window = expected_sum_from_recipes(
        {"days": window}, apply_yield=False, multiplier=mult * window_scale,
        apply_protein_yield=apply_protein_yield,
    )

    from constants import strip_accents
    aggregated_list = (plan_result.get("aggregated_shopping_list_weekly")
                       or plan_result.get("aggregated_shopping_list") or [])
    master_by_name: dict = {}
    for item in aggregated_list:
        if isinstance(item, dict) and item.get("name"):
            master_by_name.setdefault(
                strip_accents(str(item["name"]).lower().strip()), item
            )

    def _is_perishable(food) -> bool:
        key = strip_accents(str(food).lower().strip())
        return _classify_perishability(str(food), master_by_name.get(key)) == "perishable"

    mirrored: dict = {}
    dropped = []
    for food, units in expected_raw.items():
        if not _is_perishable(food):
            mirrored[food] = units
            continue
        if food in expected_window:
            mirrored[food] = expected_window[food]
        else:
            dropped.append(str(food))
    for food, units in expected_window.items():
        if food not in expected_raw and _is_perishable(food):
            mirrored[food] = units

    logging.info(
        "[COH-GUARD/P1-TRIP-WINDOWED-PERISHABLES] espejo ventana=%dd · %d alimento(s) "
        "esperados, %d perecedero(s) fuera de este viaje (%s)",
        len(window), len(mirrored), len(dropped), ", ".join(sorted(dropped)[:12]) or "-",
    )
    return mirrored


def _classify_divergence_hypothesis(
    exp_qty: float,
    act_qty: float,
    exp_units: dict,
    act_units: dict,
    food: str = "",
    pantry_deduction_applied: bool = True,
) -> str:
    """Heurístico de clasificación para `compare_expected_vs_aggregated`.

    Las hipótesis son orientativas para el reviewer humano/operacional; no
    sustituyen verificación. Orden de precedencia:
      1. cap_swallowed_modifier > 2. unit_mismatch > 3. yield_uncovered
      4. pantry_overdeduct / magnitude_undersupply > 5. unknown.

    [P2-AUDIT-1 · 2026-05-10] `food` opcional (default ''): cuando se provee
    y resuelve a pescado/mariscos vía `canonicalize_fish_seafood`, se usan
    bandas yield más estrechas (cooking loss menor que carnes rojas/blancas).
    Backward-compat: callers que no pasen `food` siguen con las bandas
    clásicas de carne/legumbre.

    [P2-GUARD-UNDERSUPPLY-CANONICAL · 2026-08-03] `pantry_deduction_applied`: si la
    lista contra la que se compara NO dedujo nevera/consumidos, el paso 4 no puede ser
    «la nevera dedujo de más» — es sub-suministro real. Ver el comentario del paso 4.
    Default `True` = comportamiento previo byte-idéntico para callers no migrados.
    """
    has_any_in_aggregated = any((q or 0) > 0 for q in act_units.values())

    # 1. food existe en expected pero TOTALMENTE ausente en aggregated.
    if exp_qty > 0 and not has_any_in_aggregated:
        return "cap_swallowed_modifier"

    # 2. la unit específica falta en aggregated pero el food sí aparece en
    # otra unit (típico: expected en `cda`, aggregated convertido a `g`, o
    # cap exact-match engulló el modificador — ver caps_asymmetry).
    if exp_qty > 0 and act_qty == 0 and has_any_in_aggregated:
        return "unit_mismatch"

    # 3. yield no aplicado: ratio típico de proteína cocida (1.35×) o
    # legumbre cocida (0.35×) que el aggregator no convirtió.
    # [P2-AUDIT-1 · 2026-05-10] Pescados/mariscos pierden menos agua al
    # cocinar que carnes (estimación literatura nutricional RD):
    #   - Pescado fileteado (tilapia, salmón, mero): 15-25% pérdida → 1.15-1.30×.
    #   - Mariscos (camarones, calamares, almejas): 5-20% pérdida → 1.05-1.20×.
    # Sin bandas separadas, divergencias 1.10-1.30 caían a `unknown` →
    # operador no veía la causa. Si `food` canonicaliza a fish/seafood,
    # usamos esas bandas; caso contrario las clásicas.
    if exp_qty > 0 and act_qty > 0:
        ratio = act_qty / exp_qty
        # Bandas clásicas (carne/legumbre).
        if 1.30 <= ratio <= 1.40 or 0.30 <= ratio <= 0.40:
            return "yield_uncovered"
        # Bandas fish/seafood — solo cuando `food` resuelve.
        if food:
            try:
                _fish_canon = canonicalize_fish_seafood(food)
            except Exception:
                _fish_canon = None
            if _fish_canon is not None:
                # Fish + seafood: rango combinado 1.05-1.30 (cooking loss
                # 5-25%). Más estrecho que carne porque la pérdida de
                # peso por cocción es menor en proteína acuática.
                if 1.05 <= ratio <= 1.30:
                    return "yield_uncovered"

    # 4. nevera/consumed dedujo de más: actual < expected/2 sin caer en
    # rangos de yield ni en zero (caso 2).
    #
    # [P3-NEW-5 · 2026-05-10] Threshold 0.5 (50%) es conservador por diseño
    # (deferred, sin code change inmediato):
    #
    #   Caso real auditado: receta espera 3kg pollo + nevera promete 2kg →
    #   ratio=0.67 > 0.5 → cae al `unknown` final, NO se reporta como
    #   `pantry_overdeduct`. Una propuesta del audit 2026-05-10 era subir
    #   el threshold a 0.75 para capturar también ese caso.
    #
    #   Razón para NO accionar sin evidencia: subir el threshold amplía
    #   el bucket `pantry_overdeduct` a expensas del `unknown`. Sin datos
    #   de cuántos `unknown` actuales son realmente overdeducts vs ruido
    #   genuino, el cambio puede inflar falsos positivos del cron alert.
    #
    #   Trigger para actuar:
    #     - SRE observa en pipeline_metrics WHERE node = '_shopping_coherence_alert'
    #       que >25% de los `unknown` correlacionan con sobrededucción real
    #       (verificar con user logs / consumed_meals).
    #     - O: usuarios reportan que pantry "olvida" items pero el guard
    #       no los flaggea como sobrededucción.
    #
    #   Si se observa: subir 0.5 → 0.75 sin redeploy via
    #   `MEALFIT_PANTRY_OVERDEDUCT_RATIO_THRESHOLD`.
    #
    # [P3-AUDIT-2 · 2026-05-10] El knob ya está implementado (antes
    # deferred). Default 0.5 preserva comportamiento histórico; subirlo
    # a 0.75 amplía el bucket (ver trigger arriba). Clamp [0.0, 1.0]:
    # valores fuera de ese rango caen al default + log warning.
    overdeduct_threshold = _knob_env_float(
        "MEALFIT_PANTRY_OVERDEDUCT_RATIO_THRESHOLD",
        0.5,
        validator=lambda v: 0.0 < v < 1.0,
    )
    #
    # [P2-GUARD-UNDERSUPPLY-CANONICAL · 2026-08-03] …pero SOLO si de verdad hubo deducción.
    # El umbral por sí solo no distingue «la nevera dedujo de más» de «la lista compra la
    # mitad de lo que las recetas exigen», y las superficies del guard comparan listas
    # CANÓNICAS (`is_new_plan=True` fuerza `physical_inventory=[]` y `consumed_ingredients=[]`,
    # P3-CANONICAL-AGG-WEEKLY): ahí no existe lado inventario, la hipótesis es IMPOSIBLE por
    # construcción, y el sub-suministro real heredaba la exención de escalada que
    # `_has_severe_divergence` le da a `pantry_overdeduct` (P1-COHERENCE-SEVERE-NO-NOISE).
    # Todo el rango `0 < ratio < umbral` quedaba invisible.
    #
    # Caso vivo del audit: espárragos 583,33 g comprados contra 1.400 g exigidos (41,7%) en
    # una lista canónica, archivado como «la nevera dedujo de más» con la nevera fuera de la
    # ecuación. `magnitude_undersupply` es la misma medición SIN la coartada.
    #
    # La exención original queda intacta donde era correcta (con deducción REAL el ratio bajo
    # sigue siendo el artefacto conocido del delta y sigue sin forzar retry).
    #
    # [ronda 1] El desdoble en DOS `return` no es estilo: `test_p1_3_coherence_labels_cross_
    # language.py` extrae las hipótesis con `return\s+["\']([a-z_]+)["\']`, que solo ve el
    # literal PEGADO al `return`. Con un ternario, `magnitude_undersupply` era invisible para
    # el parser — el drift test cross-language quedaba ciego justo con la hipótesis nueva, y
    # cuando el frontend añadiese su label saltaría `test_no_orphan_hypothesis_in_js` ("el JS
    # tiene hipótesis que el backend no emite") con un mensaje que acusa al lado equivocado.
    # Companions es-DO ya aterrizados en el repo frontend: `coherenceLabels.js`
    # (`COHERENCE_HYPOTHESIS_LABELS`) y `renderCoherenceWarnings.js` (`_ACTIONABLE_HYPOTHESES`,
    # espejo del set de `summarize_divergences_for_ui` que gobierna el toast histórico).
    if exp_qty > 0 and 0 < act_qty < exp_qty * overdeduct_threshold:
        if not pantry_deduction_applied:
            return "magnitude_undersupply"
        return "pantry_overdeduct"

    # 4-bis. [P1-COHERENCE-MILD-SHORT · 2026-08-05] Compra POR DEBAJO de lo que piden las
    # recetas, pero lejos del sub-suministro severo: el hueco entre el umbral de overdeduct
    # (0.5) y la tolerancia (~0.9). Nada lo nombraba, así que caía en `unknown`.
    #
    # `_bucket_unknown_magnitude_ratios` (P1-COHERENCE-UNKNOWN-RATIO-TELEMETRY) existe justo
    # para exigir ver la FORMA antes de inventar categorías — el propio código advierte "NO
    # añadir categorías sin ver la forma de esos ratios". Medida sobre el historial persistido
    # de 25 planes / 228 evaluaciones: `unknown` es el 28,2% de TODAS las hipótesis (202 de
    # 717), el segundo bucket; y de las incógnitas con ratio registrado, **128 de 130 (98,5%)
    # caen en la banda 0.5-0.9**. No es una nube dispersa: es un hueco único y bien definido.
    #
    # Es SOLO una etiqueta —mismo linaje que P1-COHERENCE-UNQUANTIFIED-LABEL, que rebautizó
    # 831 de 879 divergencias sin tocar comportamiento—. NO entra en `_ACTIONABLE_HYPOTHESES`:
    # comprar un 20% por debajo es ruido de envase y redondeo, no algo que el usuario deba
    # corregir a mano. Nombrarlo es lo que permite que las cifras de coherencia signifiquen
    # algo: un medidor que responde "no sé" el 28% de las veces no sostiene ninguna afirmación.
    if exp_qty > 0 and 0 < act_qty < exp_qty:
        return "magnitude_mild_short"

    # 5. [P1-COHERENCE-UNQUANTIFIED-LABEL · 2026-07-26] El alimento está en la lista pero las
    # recetas NO le ponen cantidad. Es el caso de los condimentos: "Sal al gusto" parsea a
    # `0.0 pizca`, cantidad cero, así que `expected` no lo tiene y sale `delta_pct = inf`.
    #
    # Ya estaba excluido del subset que BLOQUEA (fantasma delta=inf, exclusión por diseño — ver
    # `_BAKING_PANTRY_STAPLE_TOKENS`), pero se reportaba como `unknown`, que es la etiqueta de
    # "no sé qué pasó aquí". Medido con el guard REAL sobre 19 planes vivos: **831 de 879
    # divergencias** eran esto. Quien lee la alerta diaria veía 831 incógnitas donde había 831
    # condimentos sin cuantificar.
    #
    # Es SOLO una etiqueta: no cambia qué se reporta ni qué bloquea, cambia cómo se lee. Separar
    # este bucket es además el prerequisito para que los umbrales del cron
    # (`MEALFIT_COH_ALERT_CAP_RATIO`) midan señal en vez de ruido conocido.
    if exp_qty <= 0 and act_qty > 0:
        if any(float(v or 0) > 0 for v in exp_units.values()):
            # El alimento SÍ está en las recetas, pero en otra unidad (receta en `taza`, lista
            # en `pote`). Es simétrico del caso 2, que solo cubría `exp_qty > 0`. Medido: ~792
            # de las 879 divergencias sobre 19 planes vivos caían aquí como `unknown`, y son
            # casi todas de planes persistidos ANTES de `base_qty` — el desajuste que
            # P1-COHERENCE-BASE-QTY ya cierra para los planes nuevos.
            return "unit_mismatch"
        return "recipe_unquantified"

    return "unknown"


def _bucket_unknown_magnitude_ratios(divergences: list) -> dict:
    """[P1-COHERENCE-UNKNOWN-RATIO-TELEMETRY · 2026-07-08] Distribución de ratios actual/expected de las
    divergencias de MAGNITUD clasificadas 'unknown'. Evidencia forense para decidir si vale la pena añadir
    una categoría de hipótesis nueva: `_classify_divergence_hypothesis` cae a 'unknown' cuando la magnitud
    no encaja en yield/unit_mismatch/pantry_overdeduct, y el propio código (P3-NEW-5) advierte NO añadir
    categorías sin ver la FORMA de esos ratios (forense plan vivo 70f802ec: `{'unknown': 32}` sin insight).
    NO cambia el gate — solo telemetría (log del guard + block-history). Puro, fail-safe → {}.
    tooltip-anchor: P1-COHERENCE-UNKNOWN-RATIO-TELEMETRY"""
    buckets = {"<0.5": 0, "0.5-0.9": 0, "0.9-1.1": 0, "1.1-1.5": 0, "1.5-2": 0, "2-4": 0, ">=4": 0}
    try:
        for d in divergences or []:
            if not isinstance(d, dict):
                continue
            if (d.get("hypothesis") or "unknown") != "unknown":
                continue
            try:
                exp = float(d.get("expected_qty") or 0)
                act = float(d.get("actual_qty") or 0)
            except (TypeError, ValueError):
                continue
            if exp <= 0 or act < 0:
                continue
            r = act / exp
            if r < 0.5:
                buckets["<0.5"] += 1
            elif r < 0.9:
                buckets["0.5-0.9"] += 1
            elif r < 1.1:
                buckets["0.9-1.1"] += 1
            elif r < 1.5:
                buckets["1.1-1.5"] += 1
            elif r < 2:
                buckets["1.5-2"] += 1
            elif r < 4:
                buckets["2-4"] += 1
            else:
                buckets[">=4"] += 1
        return {k: v for k, v in buckets.items() if v > 0}
    except Exception:
        return {}


def compare_expected_vs_aggregated(
    expected: dict,
    aggregated: dict,
    *,
    tolerance: float = 0.05,
    pantry_deduction_applied: bool = True,
) -> list:
    """[P1-shop-coh-1 · 2026-05-07] Detecta divergencias `Σrecetas` ↔ `lista`.

    Compara dos dicts del mismo shape `{food: {unit: qty}}`. Una divergencia
    se reporta si `|actual - expected| > expected * tolerance`. Si
    `expected == 0` y `actual > 0`, siempre se reporta con `delta_pct = inf`
    (fantasma en la lista de compras).

    El caller es responsable de:
      - construir `aggregated` ANTES de la conversión `apply_smart_market_units`
        (para evitar falsos positivos por SKU mapping cda→g).
      - canonicalizar nombres simétricos en ambos lados (la canonicalización
        master_map del aggregator vive inline; este helper no la replica).

    Returns:
        list de dicts `{food, unit, expected_qty, actual_qty, delta_pct, hypothesis}`.
        Ordenada por `delta_pct` descendente (inf primero, luego peor a mejor).
        Vacía si no hay divergencias.

    Hipótesis posibles (ver `_classify_divergence_hypothesis`):
        unit_mismatch · yield_uncovered · cap_swallowed_modifier ·
        pantry_overdeduct · magnitude_undersupply · recipe_unquantified · unknown.

    [P2-GUARD-UNDERSUPPLY-CANONICAL · 2026-08-03] `pantry_deduction_applied`: se pasa tal
    cual al clasificador. `False` = la lista `aggregated` es CANÓNICA (no se le restó
    nevera ni consumidos), así que un `actual < expected/2` es sub-suministro real y NO
    «la nevera dedujo de más». El caller lo deriva del sello `pantry_deduction_applied` que
    el aggregator estampa (ver `_pantry_deduction_seal`). Default `True` = conservador.
    """
    if not isinstance(expected, dict):
        expected = {}
    if not isinstance(aggregated, dict):
        aggregated = {}

    # [P1-NEW-10 · 2026-05-11] Pre-normalización a unidad base dentro del
    # mismo sistema físico ANTES de iterar. Sin esto, `{Arroz: {kg: 1.0}}`
    # vs `{Arroz: {g: 1000.0}}` se reportaban como dos divergencias
    # (fantasma kg + fantasma g) en lugar de cero. La normalización es
    # simétrica: si el knob está OFF (default canary), pasamos los dicts
    # tal cual y el comportamiento es idéntico a v1 (preservar contrato
    # bajo regresión accidental del knob).
    if _get_coherence_unit_converter_enabled():
        try:
            expected = {
                food: _normalize_food_units_to_base(u or {})
                for food, u in expected.items()
            }
            aggregated = {
                food: _normalize_food_units_to_base(u or {})
                for food, u in aggregated.items()
            }
        except Exception as _norm_err:
            # Best-effort: si normalización falla, caer al comportamiento
            # v1 en vez de abortar el guard entero.
            logging.warning(
                f"[P1-NEW-10] unit_converter falló en pre-normalización: "
                f"{_norm_err}. Cayendo a comparación raw."
            )

    divergences = []
    all_foods = set(expected.keys()) | set(aggregated.keys())

    for food in all_foods:
        exp_units = expected.get(food) or {}
        act_units = aggregated.get(food) or {}
        if not isinstance(exp_units, dict):
            exp_units = {}
        if not isinstance(act_units, dict):
            act_units = {}
        all_units = set(exp_units.keys()) | set(act_units.keys())

        for unit in all_units:
            try:
                exp_qty = float(exp_units.get(unit) or 0)
                act_qty = float(act_units.get(unit) or 0)
            except (TypeError, ValueError):
                continue

            if exp_qty == 0 and act_qty == 0:
                continue

            if exp_qty == 0:
                # Fantasma: aggregated tiene algo que las recetas no piden.
                # [P1-COHERENCE-UNIT-MISMATCH-SYM · 2026-07-25] …o NO es fantasma y sólo son
                # unidades incomparables. Espejo exacto de P2-COHERENCE-PACKAGE-UNITS, que cerró
                # la dirección `act_qty == 0` (abajo) y dejó ésta abierta.
                #
                # Medido sobre el plan entregado 0bfe19ac: **37 de 40 divergencias eran esto.**
                #
                #     Harina de trigo  receta {'cda': 0.73, 'taza': 0.49}  lista "1 paquete"
                #     Orégano          receta {'cdta': ...}                lista "1 sobre"
                #     Batata           receta {'unidad': ...}              lista "680 g"
                #
                # El alimento SÍ está en las recetas, sólo que en otra unidad; buscar "paquete"
                # entre las unidades de la receta da 0 → `delta_pct = inf` → divergencia con
                # `hypothesis=unknown`. El guard ahogaba su propia señal: en ese plan sólo 3 de
                # las 40 eran comparaciones reales (Pulpo +152%, Limón −44%, Sardinas −18%), y
                # esa es exactamente la inestabilidad de conteos que P1-REVIEW-COHERENCE-SEVERE-ONLY
                # y P1-COHERENCE-COUNT-MATERIAL llevan meses conteniendo aguas abajo.
                #
                # Un fantasma DE VERDAD (alimento ausente de toda receta) tiene `exp_units` vacío
                # o a cero → `unit_mismatch` False → sigue siendo divergencia real. La detección
                # no se debilita; se deja de contar ruido como señal.
                _exp_unit_mismatch = bool(
                    COHERENCE_UNIT_MISMATCH_SYM
                    and any(float(v or 0) > 0 for v in exp_units.values()))
                divergences.append({
                    "food": food,
                    "unit": unit,
                    "expected_qty": 0.0,
                    "actual_qty": act_qty,
                    "delta_pct": float("inf"),
                    "unit_mismatch": _exp_unit_mismatch,
                    "hypothesis": _classify_divergence_hypothesis(
                        exp_qty, act_qty, exp_units, act_units, food=food,
                        pantry_deduction_applied=pantry_deduction_applied),
                })
                continue

            delta_pct = abs(act_qty - exp_qty) / exp_qty
            if delta_pct > tolerance:
                # [P2-COHERENCE-PACKAGE-UNITS · 2026-06-22] (audit fresco P2-15) Falso-positivo de magnitud
                # por unidades de ENVASE no convertibles: la receta pide el alimento en una unidad convertible
                # (g/ml/lb) pero `apply_smart_market_units` lo presenta en la lista como envase (pote/frasco/
                # Ud.). El conversor (`_normalize_food_units_to_base`) solo unifica g/ml/lb → para la unidad
                # esperada act_qty=0 aunque el alimento SÍ está en la lista bajo otra unidad → delta_pct=1.0
                # finito → entraba al subset crítico B → block + retry FALSO. Lo detectamos (alimento presente
                # bajo OTRA unidad) y lo tageamos `unit_mismatch`: sigue como telemetría warn pero NO es crítico
                # (no se puede comparar magnitud entre "1 pote" y "200 g"). NO debilita la detección real: cuando
                # las unidades SÍ son comparables, act_qty refleja la cantidad real. tooltip-anchor: P2-COHERENCE-PACKAGE-UNITS
                _unit_mismatch = (act_qty == 0 and any(float(v or 0) > 0 for v in act_units.values()))
                divergences.append({
                    "food": food,
                    "unit": unit,
                    "expected_qty": exp_qty,
                    "actual_qty": act_qty,
                    "delta_pct": delta_pct,
                    "unit_mismatch": _unit_mismatch,
                    "hypothesis": _classify_divergence_hypothesis(
                        exp_qty, act_qty, exp_units, act_units, food=food,
                        pantry_deduction_applied=pantry_deduction_applied),
                })

    # `inf` es mayor que cualquier float → ordena primero con `-delta_pct`.
    divergences.sort(key=lambda d: -d["delta_pct"])
    return divergences


def _get_coherence_guard_mode() -> str:
    """[P1-shop-coh-1 · 2026-05-07] Lee `MEALFIT_SHOPPING_COHERENCE_GUARD`.

    Valores válidos:
      - "off"   → guard no se invoca (compatibilidad backward).
      - "warn"  → invoca `compare_expected_vs_aggregated`, loggea divergencias
                  y deja seguir el pipeline. Modo canary / debugging local.
      - "block" → si `max(delta_pct) > MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT`,
                  aborta persistencia del plan (caller reintenta con Pro o
                  degrada según política). DEFAULT producción (P1-NEW-1).

    [P1-NEW-1 · 2026-05-10] Default bumpeado de "warn" a "block". Razón: el
    sistema producía listas incoherentes (cap_swallowed_modifier crítico —
    pollo en receta ausente en lista — entre otros) y solo lo loggeaba en
    `warn`. Ahora `review_plan_node` reintenta o degrada según
    `MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION` (default reject_minor).
    Rollback: `export MEALFIT_SHOPPING_COHERENCE_GUARD=warn` sin redeploy.

    Cualquier valor distinto cae al default con log de warning. Releído en
    cada invocación por preferencia operacional (cambio sin redeploy).
    """
    # [P2-1 · 2026-05-08] `_knob_env_str` ya importado a top-level desde `knobs.py`
    # (cero deps, sin riesgo de circular). El fallback try/except dejó de hacer
    # falta tras extraer los helpers a un módulo aislado.
    return _knob_env_str(
        "MEALFIT_SHOPPING_COHERENCE_GUARD",
        "block",
        choices={"off", "warn", "block"},
    )


def _get_coherence_liquid_keywords() -> set[str]:
    """[P1-1 · 2026-05-10] Lee `MEALFIT_COHERENCE_LIQUID_KEYWORDS` (CSV).

    Items cuyo nombre canónico (lower) contenga alguna de estas keywords
    reciben tolerancia ampliada en el chequeo de magnitudes — son
    condimentos/líquidos donde el escalado por household_multiplier es
    super-lineal en receta pero el usuario rara vez compra el equivalente
    (un hogar de 4 no compra 4× aceite).

    Default: keywords más comunes de condimento líquido es-DO. Knob CSV
    permite añadir/sustituir sin redeploy.

    [P3-NEW-4 · 2026-05-10] Anchor para review anual de keywords (deferred,
    sin code change inmediato):

      Cron `_shopping_coherence_alert_job` (cron_tasks.py:676) re-evalúa
      planes activos en mode=warn. Si reporta consistentemente
      `cap_swallowed_modifier` o `yield_uncovered` para items que SON
      líquidos (aceite/vinagre/salsas/etc.) pero NO están en el default,
      añadir el keyword al knob:

        export MEALFIT_COHERENCE_LIQUID_KEYWORDS="aceite,vinagre,...,nuevo_keyword"

      Candidatos a vigilar en es-DO (no añadidos por defecto hasta que
      la telemetría justifique):
        - "agrio" / "agrio de naranja" (marinada típica RD).
        - "mojo" (preparación de aliño criollo).
        - "miel" (jarabe — pero ya tiene canonical inline).
        - "leche de coco" (super-lineal en sancocho/asopao).

      Frecuencia sugerida de review: trimestral. Owner: el SRE que
      mire pipeline_metrics WHERE node = '_shopping_coherence_alert'.
    """
    raw = _knob_env_str(
        "MEALFIT_COHERENCE_LIQUID_KEYWORDS",
        "aceite,vinagre,salsa de soya,salsa soya,salsa picante",
    )
    out = set()
    for kw in str(raw).split(","):
        kw_clean = kw.strip().lower()
        if kw_clean:
            out.add(kw_clean)
    return out


def _get_coherence_liquid_tolerance_pct() -> float:
    """[P1-1 · 2026-05-10] Lee `MEALFIT_COHERENCE_LIQUID_TOLERANCE_PCT`.

    Tolerancia ampliada para items que matchean `_get_coherence_liquid_keywords`.
    Default 0.50 (50%): cubre el caso "receta escala 4× pero hogar compra 1×".
    Si está por debajo de la tolerancia base, se ignora (la base manda).
    """
    return _knob_env_float(
        "MEALFIT_COHERENCE_LIQUID_TOLERANCE_PCT",
        0.50,
        validator=lambda v: 0.0 < v < 5.0,
    )


def _is_liquid_food(food_name: str, liquid_keywords: set[str]) -> bool:
    """[P1-1 · 2026-05-10] True si el nombre canónico contiene alguna keyword."""
    if not food_name or not liquid_keywords:
        return False
    n_low = str(food_name).strip().lower()
    return any(kw in n_low for kw in liquid_keywords)


def _get_coherence_tolerance_pct() -> float:
    """[P1-shop-coh-1 · 2026-05-07] Lee `MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT`.

    Float en (0, 1). Default 0.10 (10%). Usado por el guard en modo `block`
    como umbral por encima del cual se aborta persistencia. Modo `warn`
    sigue usando la `tolerance` por defecto de `compare_expected_vs_aggregated`
    (5%) — éste knob es estrictamente para el blocking threshold, más laxo
    para evitar falsos abortos.
    """
    # [P2-1 · 2026-05-08] `_knob_env_float` ya importado a top-level desde
    # `knobs.py` (cero deps). El fallback try/except dejó de hacer falta.
    return _knob_env_float(
        "MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT",
        0.10,
        validator=lambda v: 0.0 < v < 1.0,
    )


def _get_coherence_compare_capped_pre_knob() -> bool:
    """[P1-COHERENCE-CAPPED-PRE · 2026-07-26] Lee `MEALFIT_COHERENCE_COMPARE_CAPPED_PRE`.

    Knob default **True** (opt-out). Cuando True, para los items con `capped_by` el guard
    compara las recetas contra `capped_pre` (lo calculado ANTES del tope) en vez de contra la
    cantidad topada.

    Los topes de perecederos (`P5-VEG-CAP`, `P6-LACTEOS-PERISHABLE-CAP`,
    `P6-FRUITS-LARGE-CAP`) son decisión de producto —nadie compra 30 días de tomate fresco de
    una vez— y ya se le comunican al usuario en el propio item ("alcanza ~12 de 30 días —
    recompra"). Reportarlos como incoherencia recetas↔lista es un falso positivo.

    Lo que el guard verifica con esto es que el agregador calculó BIEN desde las recetas. Si
    calcula mal, `capped_pre` diverge y se reporta igual: NO es un mute.

    Rollback sin redeploy: `MEALFIT_COHERENCE_COMPARE_CAPPED_PRE=false`.

    Tooltip-anchor: P1-COHERENCE-CAPPED-PRE-KNOB
    """
    return _knob_env_bool("MEALFIT_COHERENCE_COMPARE_CAPPED_PRE", True)


def _get_coherence_day_basis_norm_knob() -> bool:
    """[P1-COHERENCE-DAY-BASIS · 2026-07-26] Lee `MEALFIT_COHERENCE_DAY_BASIS_NORM`.

    Knob default **True** (opt-out). Cuando True, el guard escala el lado ESPERADO por
    `7.0 / días_materializados` antes de comparar contra `aggregated_shopping_list_weekly` —
    el MISMO factor que `get_shopping_list_delta` aplica al construir la lista
    (`base_duration_scale = 7.0 / num_days`, línea ~10174).

    Sin esto, el guard compara los 3 días de recetas que existen contra una lista proyectada a
    7 días y TODO diverge por 7/3 = 2.33. Medido en 19 planes vivos, el factor encaja al
    decimal (Pescado 574.7/3×7 = 1341.0 contra 1341.0 en la lista) — es estructural, no una
    incoherencia. Era la razón por la que el guard no podía estar en modo `block`.

    Nace en True (no OFF como los gates de calidad) porque NO añade rechazos: elimina falsos
    positivos de una comparación que estaba mal planteada. Rollback sin redeploy:
    `MEALFIT_COHERENCE_DAY_BASIS_NORM=false` restaura la comparación pre-fix.

    Tooltip-anchor: P1-COHERENCE-DAY-BASIS-KNOB
    """
    return _knob_env_bool("MEALFIT_COHERENCE_DAY_BASIS_NORM", True)


def _get_coherence_t2_block_severe_only_knob() -> bool:
    """[P2-COHERENCE-1 · 2026-05-11] Lee `MEALFIT_COHERENCE_T2_BLOCK_SEVERE_ONLY`.

    Knob default True (opt-out). Cuando True, las surfaces auxiliares que
    invocan el helper con `block_severe_only=True` (`_chunk_worker T2` por
    ahora) ESCALAN mode warn → block selectivo si el guard reportó al menos
    una divergencia "severa":
      - `cap_swallowed_modifier` (presence absent: receta menciona alimento,
        lista lo omite). Ejemplo: receta dice pollo, lista no tiene pollo.
      - magnitud con `delta_pct > 0.50` (lista tiene la mitad o el doble
        de lo que la receta requiere).

    Para el resto de divergencias (unknown extras, magnitudes leves <50%,
    pantry_overdeduct), el comportamiento sigue siendo warn-only.

    Rollback rápido: setear `MEALFIT_COHERENCE_T2_BLOCK_SEVERE_ONLY=false`
    sin redeploy. Restaura el comportamiento warn-only puro pre-P2-COHERENCE-1.

    Tooltip-anchor: P2-COHERENCE-1-KNOB
    """
    return _knob_env_bool(
        "MEALFIT_COHERENCE_T2_BLOCK_SEVERE_ONLY",
        True,
    )


# [P2-COHERENCE-1 · 2026-05-11] Threshold para magnitudes "severas".
# delta_pct > 0.50 = lista tiene la mitad / el doble / más de lo que la
# receta requiere. <0.50 son drift menores que el cron diario captura
# post-hoc sin necesidad de retry.
_COHERENCE_SEVERE_MAGNITUDE_THRESHOLD = 0.50


def _get_guard_undersupply_severe_knob() -> bool:
    """[P2-GUARD-UNDERSUPPLY-CANONICAL · 2026-08-03 · default False en ronda 1] Lee
    `MEALFIT_GUARD_UNDERSUPPLY_SEVERE`. **Default `False`: telemetry-first.**

    Qué NO está gateado (llega desde el día uno, sin tocar el knob):
      - la clasificación `magnitude_undersupply` en vez de `pantry_overdeduct`,
      - su presencia en `_shopping_coherence_block_history`, en el log del guard, en el
        `Counter` por hipótesis del cron diario y en el banner accionable.
      Es decir: la OBSERVABILIDAD del agujero se enciende inmediatamente. Lo único gateado
      es que la divergencia cuente como *severa* para escalar warn→block.

    Por qué el default arranca apagado (evidencia medida en la ronda 1, contra el brief que
    pedía `True`): la consecuencia real en la surface #3 (`_chunk_worker` T2 con
    `block_severe_only`) es que un `block_set` lanza `RuntimeError` DENTRO de un retry loop
    que re-corre un cómputo **determinista** — mismos días, mismo inventario, mismo catálogo
    ⇒ mismo resultado. Los 3 intentos fallan igual → `shopping_list_ok=False` → re-encolado
    con backoff → `CHUNK_MAX_FAILURE_ATTEMPTS` → **dead letter**. Y la población que llega a
    T2 con sub-oferta ≥0.5 es justo la que la surface #1 ya no pudo arreglar regenerando.
    Es el mismo mecanismo de P1-COHERENCE-SEVERE-NO-NOISE (2026-07-07): una escalada T2 en
    falso que quemó 3 retries + re-encolado. Reintroducirlo a ciegas para arreglar una
    exención a ciegas sería cambiar un error por su simétrico.

    Cómo encenderlo (secuencia, no fecha): medir en producción el volumen real de
    `magnitude_undersupply` en `_shopping_coherence_block_history` — cuántos planes, cuántos
    alimentos por plan, y cuántos de ellos vienen de T2. Si el volumen es el esperado
    (unidades, no decenas) y el retry tiene alguna posibilidad de converger,
    `MEALFIT_GUARD_UNDERSUPPLY_SEVERE=true` sin redeploy.

    [P3-UNDERSUPPLY-VISIBILITY · 2026-08-04] "Medir el volumen" no tenía SELECT
    programado — el default se habría vuelto permanente por inercia (nadie mira una
    tabla a mano todos los días). El cron diario `_shopping_coherence_alert_job`
    (cron_tasks.py) ahora expone `undersupply_count` como campo EXPLÍCITO en el tick
    `_shopping_coherence_alert_job_tick` (mismo patrón que `cap_count`) + una línea de
    log dedicada con este mismo marker — la serie diaria completa (ceros incluidos)
    queda visible sin SELECT manual.

    **Criterio de encendido con el baseline**: encender cuando la serie diaria muestre
    `magnitude_undersupply` estable y bajo (p.ej. <5% de las entries diarias, sin
    ráfagas concentradas en un solo plan — una ráfaga sugiere un bug puntual, no el
    volumen difuso que este knob asume). Baseline medido 2026-08-04 (~1,4h post-deploy,
    22 planes con history, 186 entries): **0 sobre 186 entries históricas** — la
    hipótesis literalmente no ha aparecido todavía (aún sin tráfico de listas nuevas
    construidas con el sello `pantry_deduction_applied` que necesita). Sigue en
    observación: encender requiere que la serie diaria acumule volumen suficiente para
    juzgar "estable y bajo" contra algo que no sea ruido de muestra pequeña.

    Tooltip-anchor: P2-GUARD-UNDERSUPPLY-CANONICAL-KNOB
    """
    return _knob_env_bool("MEALFIT_GUARD_UNDERSUPPLY_SEVERE", False)


def _has_severe_divergence(divergences: list) -> bool:
    """[P2-COHERENCE-1 · 2026-05-11] True si la lista contiene al menos
    una divergencia "severa" según el contrato del knob T2_BLOCK_SEVERE_ONLY.

    Severas:
      - hypothesis == 'cap_swallowed_modifier' (food de receta ausente en
        lista). Es la categoría más visible al usuario.
      - magnitude=True AND delta_pct > _COHERENCE_SEVERE_MAGNITUDE_THRESHOLD.

    NO severas (warn-only):
      - hypothesis == 'unknown' (food de lista que no aparece en recetas —
        normalmente staples o noise; bloquear retry sería ruidoso).
      - hypothesis == 'pantry_overdeduct' (caso conocido del aggregator).
      - hypothesis == 'unit_mismatch' / 'yield_uncovered' con delta menor.

    [P2-GUARD-UNDERSUPPLY-CANONICAL · 2026-08-03 · default False en ronda 1]
    `magnitude_undersupply` es el mismo rango de magnitud que `pantry_overdeduct` pero sobre
    una lista que NO dedujo nevera, así que no hay artefacto del delta que lo explique: la
    lista compra menos de la mitad de lo que las recetas exigen y el plato no sale. Cuenta
    como severa SOLO con `MEALFIT_GUARD_UNDERSUPPLY_SEVERE=true`; el default (`False`) la
    mantiene exenta mientras se mide el volumen en producción — ver el docstring del knob
    para el modo de fallo (dead-letter determinista en T2) que motivó arrancar apagado.
    """
    if not divergences:
        return False
    # [P2-GUARD-UNDERSUPPLY-CANONICAL] Una sola lectura del knob por llamada (el default no
    # cambia a mitad de una lista de divergencias) en vez de una por ítem.
    # [P1-COHERENCE-MILD-SHORT · 2026-08-05] `magnitude_mild_short` hereda la exención
    # de `unknown`, que es de donde salió. Sin esto el reetiquetado NO sería "solo una
    # etiqueta": esas divergencias pasarían de exentas a candidatas a escalar.
    #
    # Hoy no escalarían por un margen fino —la banda 0.5-0.9 produce |delta| ≤ 0.49 y el
    # check de severidad exige > 0.50—, pero eso es una coincidencia aritmética, no un
    # diseño: basta bajar `MEALFIT_PANTRY_OVERDEDUCT_RATIO_THRESHOLD` a 0.3 (knob que
    # existe y tiene tests propios) para que la banda llegue a |delta| 0.7 y empiece a
    # forzar retries. Sería el modo de fallo que P1-COHERENCE-SEVERE-NO-NOISE cerró:
    # sobre-oferta de envase escalando T2 warn→block en falso, 3 retries + re-encolado.
    #
    # Y de paso alinea el código con lo que la nota de abajo YA afirmaba: "`unknown` de
    # magnitud es SIEMPRE sobre-oferta". Era falso mientras el sub-suministro leve vivía
    # ahí dentro; con la banda separada, ahora es cierto.
    _exempt_hypotheses = ["unknown", "pantry_overdeduct", "magnitude_mild_short"]
    if not _get_guard_undersupply_severe_knob():
        _exempt_hypotheses.append("magnitude_undersupply")
    for d in divergences:
        if not isinstance(d, dict):
            continue
        if d.get("hypothesis") == "cap_swallowed_modifier":
            return True
        if d.get("magnitude") is True:
            # [P1-COHERENCE-SEVERE-NO-NOISE · 2026-07-07] (plan vivo 72c8b965 wk2: 67
            # divergencias `unknown` de SOBRE-oferta de envase — arroz/lechuga/ajo/calamar
            # comprados por paquete — escalaban el chunk T2 warn→block en FALSO, 3 retries +
            # re-encolado, quemando GLM). El docstring de arriba YA declara `unknown` y
            # `pantry_overdeduct` NO-severas, pero este check de magnitud las capturaba igual
            # cuando |delta|>0.50. Semántica: `unknown` de magnitud es SIEMPRE sobre-oferta
            # (act>exp; el sub-suministro severo se clasifica `pantry_overdeduct`), y la
            # sobre-oferta de envase NUNCA hace el plan incocinable → no debe forzar retry.
            # Solo cap_swallowed (falta real, arriba) + magnitudes severas de tipos accionables
            # (yield_uncovered/unit_mismatch) escalan. Alinea el CÓDIGO con el docstring.
            if d.get("hypothesis") in _exempt_hypotheses:
                continue
            try:
                delta = float(d.get("delta_pct") or 0)
            except (TypeError, ValueError):
                delta = 0.0
            # [P1-COHERENCE-INF-NOT-SEVERE · 2026-07-30] Un delta INFINITO no es una magnitud: es
            # un denominador que falta. El guard escribe `delta_pct = float("inf")` en la única
            # rama donde `expected_qty = 0` — la receta no pide NADA de ese alimento y la lista
            # tiene algo (sobre-oferta por construcción). `abs(inf) > 0.50` es verdadero
            # trivialmente, así que "la receta no dijo cuánto" se leía como la magnitud más severa
            # posible.
            #
            # Caso vivo (plan 4d2c1111, semana 2, 2026-07-30): `Pimienta negra` y `Sal` con
            # hipótesis `recipe_unquantified` ("sal al gusto", sin gramos) escalaron T2 warn→block,
            # el chunk agotó sus 3 intentos y **el usuario se quedó sin la lista de compras de la
            # semana 2**. Un plan entero bloqueado por sal y pimienta.
            #
            # Es el mismo endurecimiento que P1-COHERENCE-SEVERE-NO-NOISE (2026-07-07) aplicó a
            # `unknown`/`pantry_overdeduct`, y la misma clase que P1-TRANSFORM-GATE-PARITY: DOS
            # tests de severidad para el mismo concepto y el endurecimiento aterrizó en uno solo.
            # La ruta de review ya era inmune porque pasa por `_coherence_finite_abs_delta`
            # (graph_orchestrator), que mapea inf/NaN → 0.0; esta no lo hacía. Aquí no se puede
            # importar ese helper (graph_orchestrator importa de este módulo — sería circular), así
            # que se replica la semántica y un test ancla la PARIDAD entre las dos.
            #
            # NO debilita la detección: el caso que de verdad rompe un plan — la receta menciona un
            # alimento y la lista lo omite — es `cap_swallowed_modifier`, que sale por el `return
            # True` de arriba por NOMBRE de hipótesis, sin mirar el delta.
            if delta != delta or abs(delta) == float("inf"):
                continue
            if abs(delta) > _COHERENCE_SEVERE_MAGNITUDE_THRESHOLD:
                return True
    return False


def summarize_divergences_for_ui(divergences: list, max_items: int = 5) -> list:
    """[P2-COHERENCE-1 · 2026-05-11] Compacta la lista de divergencias del
    guard a un shape consumible por el frontend para renderear toasts.

    Retorna los primeros `max_items` items con shape estable:
      `{food, hypothesis, side, magnitude, delta_pct?}`
    Skipea entries no-dict y campos ausentes (resilient a evolución
    futura del guard sin romper UI).

    [P1-COHERENCE-BANNER-NOISE · 2026-06-22] El banner UI ("Lista revisada — N
    items pueden necesitar ajuste manual") debe surfacear SOLO divergencias
    ACCIONABLES por el usuario:
      - `cap_swallowed_modifier`: el alimento está en las recetas pero AUSENTE de
        la lista → "se te puede olvidar comprarlo" (accionable).
      - `pantry_overdeduct`: sub-suministro SEVERO (actual < expected/2, la nevera
        dedujo de más) → "te puedes quedar corto" (accionable).
      - `magnitude_undersupply` [P2-GUARD-UNDERSUPPLY-CANONICAL · 2026-08-03]: el MISMO
        sub-suministro severo pero sobre una lista que no dedujo nevera. Pre-fix estas
        divergencias salían etiquetadas `pantry_overdeduct` y YA aparecían en el banner;
        renombrarlas sin añadirlas aquí las habría borrado de la UI en silencio — es
        exactamente lo que el usuario necesita ver ("te puedes quedar corto").
    Se OMITEN del banner las divergencias de MAGNITUD benignas — `unknown`,
    `unit_mismatch`, `yield_uncovered` — que en la práctica son artefactos NO
    accionables: el alimento SÍ está en la lista, solo difiere por unidad de
    compra (ajo receta "dientes" / lista "cabezas"; cilantro "g" / "mazo"),
    rendimiento cocido↔crudo (cerdo, camarón), o compra por unidad entera
    (plátano, guineo, aguacate). En un re-escalado (cambio de duración) estas
    divergencias no representan drift real recetas↔lista — escalan ambos lados.
    Sin este filtro, cambiar la lista a 15/30 días disparaba un banner alarmante
    de 5 "Causa indeterminada" sobre alimentos que estaban correctos en la lista.

    NO se oculta nada accionable: "ausente" y "sub-suministro severo" siguen
    surfaceándose. La telemetría/historial
    (`run_shopping_coherence_guard_and_append_history`) conserva TODAS las
    divergencias para post-mortem; este filtro es solo para el banner UI.
    Knob `MEALFIT_COHERENCE_BANNER_ACTIONABLE_ONLY` (default True) revierte sin
    redeploy. Tooltip-anchor: P1-COHERENCE-BANNER-NOISE.
    """
    if not divergences:
        return []
    _actionable_only = _knob_env_bool("MEALFIT_COHERENCE_BANNER_ACTIONABLE_ONLY", True)
    _ACTIONABLE_HYPOTHESES = {
        "cap_swallowed_modifier", "pantry_overdeduct",
        "magnitude_undersupply",  # [P2-GUARD-UNDERSUPPLY-CANONICAL]
    }
    out = []
    for d in divergences:
        if not isinstance(d, dict):
            continue
        if _actionable_only and (d.get("hypothesis") or "unknown") not in _ACTIONABLE_HYPOTHESES:
            # Benigno/no-accionable (magnitud por unidad/yield/cap): fuera del banner.
            continue
        item = {
            "food": d.get("food") or d.get("name") or "",
            "hypothesis": d.get("hypothesis") or "unknown",
            "side": d.get("side") or "",
            "magnitude": bool(d.get("magnitude")),
        }
        if d.get("magnitude"):
            try:
                item["delta_pct"] = round(float(d.get("delta_pct") or 0), 3)
            except (TypeError, ValueError):
                pass
        out.append(item)
        if len(out) >= max_items:
            break
    return out


def _get_coherence_unit_converter_enabled() -> bool:
    """[P1-NEW-10 · 2026-05-11 · P2-UNIT-CONV-1 default flip · 2026-05-11]
    Lee `MEALFIT_COHERENCE_UNIT_CONVERTER_ENABLED`.

    Knob ACTIVE (default True post-P2-UNIT-CONV-1). Cuando True (default),
    `compare_expected_vs_aggregated` pre-normaliza ambos dicts
    (expected/aggregated) a unidad base dentro del mismo sistema físico
    vía `canonical_units.to_base_amount` antes de comparar. Resuelve
    falsos positivos del tipo:
        receta: `{Arroz: {kg: 1.0}}`  vs  lista: `{Arroz: {g: 1000.0}}`
        → ambos se normalizan a `{Arroz: {g: 1000.0}}` → no drift.

    Histórico:
      - P1-NEW-10 (2026-05-11): introducido como CANARY default False.
        Razón: prod no observaba esta divergencia (cron diario reportaba 0%
        `unit_mismatch` por aliasing kg↔g). Fix preventivo para drift
        futuro de LLM/prompt.
      - P2-UNIT-CONV-1 (2026-05-11): flip default a True. Audit prod via
        MCP confirmó 0 entries en `_shopping_coherence_block_history` en
        las últimas horas (3 planes total, todos abandoned). Sin datos
        reales, la decisión se basa en el contrato del converter:
          - Solo unifica unidades del MISMO sistema físico (peso↔g,
            volumen↔ml). NO hace cross-system (kg↔ml requiere densidad).
          - Tests `test_p1_new_10_*` cubren la matemática.
          - El mecanismo de "drift" detectado pre-fix era PURAMENTE false
            positive (ambas representaciones eran semánticamente correctas).
        Knob queda como kill switch: setear
        `MEALFIT_COHERENCE_UNIT_CONVERTER_ENABLED=false` revierte sin redeploy.

    Tooltip-anchor: P2-UNIT-CONV-1-DEFAULT
    """
    return _knob_env_bool(
        "MEALFIT_COHERENCE_UNIT_CONVERTER_ENABLED",
        True,
    )


def _normalize_food_units_to_base(units_dict: dict) -> dict:
    """[P1-NEW-10 · 2026-05-11] Convierte `{unit: qty}` a `{base_unit: qty}`
    consolidando aliases del mismo sistema físico.

    Ejemplos:
      {kg: 1.0}              → {g: 1000.0}
      {g: 100, kg: 0.5}      → {g: 600.0}             (merge mismo base)
      {taza: 2, cda: 4}      → {ml: 540.0}            (2*240 + 4*15)
      {kg: 0.5, ml: 200}     → {g: 500.0, ml: 200.0}  (sistemas distintos preservados)
      {unidad: 3}            → {unidad: 3.0}          (no convertible, pass-through)
      {kg: "bad"}            → {kg: "bad"}            (no numérico, pass-through)

    Args:
        units_dict: dict `{unit: qty}` (qty numérico o castable).

    Returns:
        Nuevo dict con las mismas semánticas pero con unidades convertidas
        a base + entries de unidades no convertibles preservadas. SIEMPRE
        devuelve dict nuevo (no mutates el input).
    """
    if not isinstance(units_dict, dict):
        return {}
    out = defaultdict(float)
    preserved = {}
    for unit, qty in units_dict.items():
        try:
            qty_f = float(qty)
        except (TypeError, ValueError):
            preserved[unit] = qty
            continue
        qty_base, base_unit = _to_base_amount(qty_f, unit)
        # Si el helper devolvió la unidad ORIGINAL sin convertir (no
        # convertible o desconocida), preservamos sin merge.
        if base_unit not in _CONVERTIBLE_BASE_UNITS:
            preserved[base_unit if base_unit else unit] = qty_base
            continue
        out[base_unit] += qty_base
    # Combinar resultados convertidos + preservados. Las keys son disjuntas
    # por construcción (preserved nunca contiene 'g' ni 'ml' base).
    merged = dict(out)
    for k, v in preserved.items():
        # Edge case: si por algún motivo una unidad preservada colisiona
        # con una base ('g' o 'ml'), priorizamos la del lado convertido.
        if k in merged:
            continue
        merged[k] = v
    return merged


_CONVERTIBLE_BASE_UNITS = frozenset({"g", "ml"})  # P1-NEW-10


def _extract_aggregated_food_dict(aggregated_list, *, exclude_pavo: bool = False) -> dict:
    """[P1-C 2026-05-07] Extrae `{food: {unit: qty}}` desde aggregated_shopping_list.

    Aplica los mismos filtros que `run_shopping_coherence_guard` v1 (skip
    `is_staple=True` y `category` con "urgente"). Lee `market_qty_numeric` /
    `market_unit` con fallback a `quantity` / `unit` cuando faltan.

    Args:
        aggregated_list: lista de items del shopping list. Cada item dict.
        exclude_pavo: si True, omite items cuyo nombre matchea `^pavo`. La
            regla fresh-vs-procesado del aggregator sobre pavo (50+ líneas)
            no se replica aquí; comparar magnitudes sin replicar produciría
            falsos positivos. Presence/absence sigue capturando pavo.

    Returns:
        dict `{name_strip: {unit_lower: qty}}`. Vacío si la lista no es válida.
    """
    out = defaultdict(lambda: defaultdict(float))
    if not aggregated_list or not isinstance(aggregated_list, list):
        return {}
    for item in aggregated_list:
        if not isinstance(item, dict):
            continue
        cat = str(item.get("category") or item.get("display_category") or "").lower()
        if "urgente" in cat:
            continue
        if item.get("is_staple") is True:
            continue
        name = item.get("name") or item.get("display_name")
        if not name:
            continue
        name_str = str(name).strip()
        if exclude_pavo and re.match(r'^pavo\b', name_str.lower()):
            continue
        # [P1-COHERENCE-BASE-QTY · 2026-07-26] Preferir la cantidad en unidad BASE cuando el
        # item la trae. Es el mismo idioma en el que hablan las recetas (g/taza/cda), así que
        # el emparejamiento por (alimento, unidad) del comparador por fin encuentra pareja.
        # Con `market_qty` el esperado salía 0.0 para todos —pote/sobre/paquete no existen en
        # una receta— y las 39 divergencias del plan 01d63a5b caían a `unknown` por eso.
        # Fallback al comportamiento previo para listas legacy sin `base_qty`.
        qty = None
        unit = None
        _bq, _bu = item.get("base_qty"), item.get("base_unit")
        # [P1-COHERENCE-CAPPED-PRE · 2026-07-26] Si el item fue TOPADO a propósito, comparar
        # contra `capped_pre` — la cantidad que el agregador calculó ANTES del tope.
        #
        # Los perecederos llevan un tope deliberado con mensaje al usuario:
        #
        #   Yogurt   capped_by=P6-LACTEOS-PERISHABLE-CAP  pre=2324.8  post=907.2
        #            display: "1 pote (1.96 kg) · alcanza ~12 de 30 días — recompra"
        #   Cebolla  capped_by=P5-VEG-CAP                 pre=1459.5  post=600.0
        #   Tomate   capped_by=P5-VEG-CAP                 pre=1575.0  post=750.0
        #   Lechosa  capped_by=P6-FRUITS-LARGE-CAP        pre=3613.3  post=3000.0
        #
        # Comparar las recetas contra `post` reporta el TOPE como si fuera incoherencia: no lo
        # es, es una decisión de producto (no se compran 30 días de tomate fresco de golpe) y
        # además se le comunica al usuario. Lo que el guard debe verificar es que el agregador
        # calculó BIEN a partir de las recetas; el tope se aplica después y por diseño.
        #
        # Tras P1-COHERENCE-DAY-BASIS, `capped_pre` coincide con el lado esperado **al decimal**
        # en los cuatro casos de arriba — confirmación independiente de que la base de días
        # quedó bien, desde un campo que el guard no miraba.
        #
        # Si el agregador calcula mal, `capped_pre` diverge y el guard lo sigue viendo: esto NO
        # es un mute. Knob de rollback: MEALFIT_COHERENCE_COMPARE_CAPPED_PRE=false.
        #
        # ⚠️ [P1-VEG-BACKFILL-HONESTY · 2026-08-03 · review final] EXCLUSIÓN del sello SINTÉTICO.
        # La sustitución de arriba es legítima sólo cuando `capped_pre` es un cómputo del AGREGADOR
        # independiente del lado esperado. El sello `qty_reconcile_v7` (déficit de texto sin cap
        # real que lo explique) tiene como `pre_value` la propia demanda de las recetas — la misma
        # `expected_sum_from_recipes` con el mismo multiplicador que construye el lado ESPERADO —
        # así que sustituir convertía la comparación en una tautología y el guard dejaba de ver
        # justo los ítems que compran de menos (modo `block` por default: sin divergencia no hay
        # retry, no hay degradación y no hay fila en `_shopping_coherence_block_history`).
        # Hoy el productor ya no escribe `capped_pre` para el sintético (usa `shortfall_*`), pero
        # la exclusión se mantiene aquí porque el sello queda PERSISTIDO en las listas: una lista
        # construida por una versión anterior seguiría cegando al guard al releerla.
        # tooltip-anchor: P1-QTY-SHORTFALL-OWN-CHANNEL
        if (item.get("capped_by")
                and item.get("capped_by") != QTY_RECONCILE_SYNTHETIC_REASON
                and _get_coherence_compare_capped_pre_knob()):
            _cp = item.get("capped_pre")
            if isinstance(_cp, (int, float)) and float(_cp) > 0 and _bu:
                _bq = float(_cp)
        if isinstance(_bq, (int, float)) and float(_bq) > 0 and _bu:
            qty, unit = float(_bq), str(_bu).strip().lower()
        if qty is None:
            try:
                qty = float(item.get("market_qty_numeric") or item.get("quantity") or 0)
            except (TypeError, ValueError):
                qty = 0.0
            unit = str(item.get("market_unit") or item.get("unit") or "").strip().lower()
        if qty <= 0:
            continue
        if not unit:
            unit = "unidad"
        out[name_str][unit] += qty
    return {n: dict(u) for n, u in out.items()}


def _normalize_food_dict_to_grams(food_dict: dict) -> dict:
    """[P1-COHERENCE-GRAM-NORM · 2026-07-26] Lleva `{food: {unit: qty}}` a gramos.

    El guard empareja por (alimento, UNIDAD). Medido sobre el plan vivo fbe53a5b, los dos
    lados hablan idiomas distintos y por eso `expected_qty` salía 0.0 en casi todo:

        lado RECETAS:  taza×8, unidad×15, cda×9, cdta×8, g×11, pizca, diente, hoja, rebanada
        lado LISTA:    g (tras P1-COHERENCE-BASE-QTY)

    Solo los 11 que ya venían en gramos podían casar. Preservar la cantidad base en la lista
    fue necesario pero NO suficiente: faltaba convertir el lado de las recetas al mismo
    idioma. Se usa `convert_amount` (SSOT de db_inventory, con densidad del catálogo) — el
    mismo motor que ya usa el inventario, no una tabla nueva.

    Lo que NO se puede convertir (pizca, `cantidad necesaria`, densidad ausente en modo
    strict) conserva su unidad original: es preferible una divergencia sin pareja a una
    conversión inventada. tooltip-anchor: P1-COHERENCE-GRAM-NORM
    """
    if not isinstance(food_dict, dict) or not food_dict:
        return {}
    try:
        from db_inventory import convert_amount as _conv
    except Exception:
        return food_dict
    # [P1-COHERENCE-CANON-DENSITY · 2026-07-26] La búsqueda NO puede ser solo por nombre exacto.
    #
    # Este normalizador corre DESPUÉS de `_canonicalize_food_dict_for_coherence`, así que las
    # claves que recibe ya son etiquetas canónicas — y una etiqueta canónica muchas veces NO es
    # una fila del catálogo. `Plátano verde` y `Plátano maduro` colapsan ambos a **`Plátano`**,
    # que no existe en `master_ingredients`; `Yogurt`/`Yogurt griego …` colapsan a **`Yogur`**,
    # que tampoco. El lookup fallaba, `convert_amount` recibía `{}` y avisaba
    # `item='<unknown>' sin density_g_per_unit` → devolvía None → la fila se quedaba en `unidad`
    # mientras la lista hablaba en `g` → `unit_mismatch` con `expected_qty=0.0`: un FANTASMA
    # inventado por el propio guard.
    #
    # Medido en el plan vivo fbe53a5b: `Plátano` esp=0.0 lista=1400.0 reportado como fantasma,
    # cuando ambas filas de plátano traen `density_g_per_unit=280` y la conversión era posible
    # (2.5 uds × 280 = 700 g contra 1400 g = divergencia REAL de ×2, que es lo que hay que
    # reportar). Mismo caso en Yogurt (`density_g_per_cup=245`).
    #
    # Se indexa por nombre, por alias y por forma canónica, sin sobreescribir: el nombre exacto
    # siempre gana. Dentro de un grupo canónico las densidades son equivalentes por construcción
    # —es la razón por la que colapsan— así que tomar la primera fila del grupo es correcto.
    try:
        master_map = {}
        _canon_pend = []
        for m in (get_master_ingredients() or []):
            nm = str(m.get("name", "")).strip()
            if not nm:
                continue
            master_map.setdefault(nm.lower(), m)
            for _al in (m.get("aliases") or []):
                if _al:
                    master_map.setdefault(str(_al).strip().lower(), m)
            _canon_pend.append((nm, m))
        for nm, m in _canon_pend:
            try:
                _c = sorted(_canonicalize_for_coherence({nm}))
            except Exception:
                continue
            if _c:
                master_map.setdefault(_c[0].strip().lower(), m)
    except Exception:
        master_map = {}
    out = {}
    for food, units in food_dict.items():
        if not isinstance(units, dict):
            continue
        _mi = master_map.get(str(food).lower()) or {}
        acc = {}
        grams = 0.0
        for unit, qty in units.items():
            try:
                q = float(qty)
            except (TypeError, ValueError):
                continue
            if q <= 0:
                continue
            u = str(unit).strip().lower()
            if u in ("g", "gr", "gramo", "gramos"):
                grams += q
                continue
            _g = None
            try:
                _g = _conv(q, u, "g", _mi)
            except Exception:
                _g = None
            if isinstance(_g, (int, float)) and _g > 0:
                grams += float(_g)
            else:
                acc[u] = acc.get(u, 0.0) + q      # no convertible → se conserva tal cual
        if grams > 0:
            acc["g"] = round(grams, 2)
        if acc:
            out[food] = acc
    return out


def _canonicalize_food_dict_for_coherence(food_dict: dict) -> dict:
    """[P1-C 2026-05-07] Canonicaliza las keys de un dict `{food: {unit: qty}}`
    aplicando la misma lógica que `_canonicalize_for_coherence` y sumando
    units cuando 2 nombres originales mapean al mismo canónico (e.g.,
    "huevo" y "claras de huevo" → ambos a "Huevo").
    """
    if not isinstance(food_dict, dict) or not food_dict:
        return {}
    raw_names = list(food_dict.keys())
    canonical_set = _canonicalize_for_coherence(raw_names)
    # Re-build mapping {raw → canonical} aplicando mismas reglas que el set.
    # Truco: pasar uno por uno y leer el único elemento del set retornado.
    out = defaultdict(lambda: defaultdict(float))
    for raw in raw_names:
        canon_set = _canonicalize_for_coherence([raw])
        if not canon_set:
            continue
        canonical = next(iter(canon_set))
        units = food_dict.get(raw) or {}
        if not isinstance(units, dict):
            continue
        for unit, qty in units.items():
            try:
                out[canonical][unit] += float(qty or 0)
            except (TypeError, ValueError):
                continue
    # Sanity check: garantiza que el set canónico calculado bulk coincide
    # con las keys del dict construido item-by-item (defensivo, no bloqueante).
    if set(out.keys()) != canonical_set:
        logging.debug(
            f"[COH-GUARD/v2] canonical drift: bulk={canonical_set} item={set(out.keys())}"
        )
    return {n: dict(u) for n, u in out.items()}


# [P2-NEW-1 · 2026-05-10] Canonical base names para los 3 proteínas centrales
# es-DO. Mantenidos como constantes para que test parser-based + futuro refactor
# tengan un anchor estable. El aggregator NO consolida estos como hace con pavo
# (no hay fresh-vs-procesado distinction comercial relevante), pero el coherence
# guard sí necesita simetría: "pechuga de pollo desmenuzada" en receta vs.
# "Pollo" en lista debe matchear.
_PROTEIN_CANONICAL_POLLO = 'Pollo'
_PROTEIN_CANONICAL_CERDO = 'Cerdo'
_PROTEIN_CANONICAL_RES = 'Res'


def canonicalize_protein(name) -> str | None:
    """[P2-NEW-1 · 2026-05-10] Canonicaliza nombres de pollo/cerdo/res a su
    nombre base para uso simétrico en el coherence guard.

    Cubre el modo de falso positivo no atrapado por pavo (que tiene su propio
    helper) ni por el fallback genérico (que solo strippea modificadores
    explícitos en `_TRAILING_MODIFIERS_ES`):
      - "pechuga de pollo fresca" (receta) vs. "Pollo" (lista) → ambos a 'Pollo'.
      - "muslo de pollo desmenuzado" vs. "Pollo" → ambos a 'Pollo'.
      - "carne de res molida" vs. "Res" → ambos a 'Res' (preserved como caso
        más conservador; el aggregator tampoco distingue "Res molida" como
        canónico aparte salvo en master_map explícito).
      - "chuleta de cerdo guisada" vs. "Cerdo" → ambos a 'Cerdo'.

    Diferencias vs. `canonicalize_pavo`:
      - NO hay distinción fresh-vs-procesado (no existe deli comercial
        equivalente para pollo/cerdo/res en RD que justifique el split).
      - NO preserva "molido" como canónico aparte (el aggregator a menudo
        usa "Carne molida" o el master_map alias resuelve; aquí colapsamos
        al genérico para que el guard compare magnitudes sumadas).
      - NO mata el caso "X enlatado" (productos ya procesados industrial
        deli) — el master_map debe canonicalizar eso por su lado.

    Reglas, en orden de precedencia:
      1. corte (`pechuga|muslo|filete|chuleta|pierna|lomo|costilla`) + de + X →
         canonical(X). Cubre "pechuga de pollo", "chuleta de cerdo", etc.
      2. X (`pollo|cerdo|res`) + cooking-state (`cocido|asado|hervido|
         desmenuzado|guisado|frito|horneado|molido|asada|frita|...`) → canonical(X).
      3. X + fresh/procesado markers (`fresco|fresca|orgánico|natural`) → canonical(X).
      4. exact match `pollo|cerdo|res|carne de res` → canonical correspondiente.
      5. Otros casos → None (no es el dominio de este helper).

    Returns:
      'Pollo' / 'Cerdo' / 'Res' / None.

    Nota: el aggregator NO tiene reglas equivalentes (verificado contra
    `shopping_calculator.py:3216-3238` donde solo pavo tiene canonicalización
    explícita). Este helper unilateralmente canonicaliza para el guard, lo
    que implica que el guard ahora trata "Pollo desmenuzado" en la lista y
    "Pollo" en la receta como mismo food para magnitudes. Esto es el
    comportamiento deseado (yield_uncovered / magnitudes se evalúan sobre
    el total acumulado del protein), no un falso positivo.
    """
    if not name:
        return None
    n_low = str(name).strip().lower()

    # Detectar cuál proteína está presente (mutuamente excluyentes — no
    # esperamos "pollo de res").
    has_pollo = bool(re.search(r'\bpollo\b', n_low))
    has_cerdo = bool(re.search(r'\bcerdo\b', n_low))
    has_res = bool(re.search(r'\b(res|carne\s+de\s+res)\b', n_low))

    # Multi-match raro: si más de uno, no canonicalizamos (no es claro qué
    # gana, ej. "pollo a la carne de res" — patológico).
    if sum([has_pollo, has_cerdo, has_res]) != 1:
        return None

    # Excluir composiciones que NO son del dominio (caldos, picadillos
    # mixtos, productos enlatados, embutidos derivados que tienen su
    # propio canónico industrial).
    if re.search(
        r'caldo|consomé|picadillo\s+(mixto|de\s+\w+)|enlatad[oa]|salchich|'
        r'longaniza|salami|jam[oó]n\b|tocineta|bacon|chorizo|nugget',
        n_low,
    ):
        return None

    if has_pollo:
        return _PROTEIN_CANONICAL_POLLO
    if has_cerdo:
        return _PROTEIN_CANONICAL_CERDO
    if has_res:
        return _PROTEIN_CANONICAL_RES
    return None


# [P1-AUDIT-2 · 2026-05-10] Canonical mapping para pescados y mariscos es-DO.
# Hardcoded por especie porque (a diferencia de pollo/cerdo/res, donde son 3
# canonicales) aquí cada especie es su propio canonical. El aggregator NO
# normaliza: "Filete de salmón" vs "Salmón" llegan distintos al guard sin
# este helper → false positive `cap_swallowed_modifier`. Cubre tilde / sin
# tilde + plural / singular de cada especie.
_FISH_SEAFOOD_CANONICAL = {
    # Fish — pescados de uso común RD
    'pescado': 'Pescado',
    'pescados': 'Pescado',
    'tilapia': 'Tilapia',
    'tilapias': 'Tilapia',
    'salmón': 'Salmón',
    'salmon': 'Salmón',
    'salmones': 'Salmón',
    'mero': 'Mero',
    'meros': 'Mero',
    'dorado': 'Dorado',
    'dorados': 'Dorado',
    'atún': 'Atún',
    'atun': 'Atún',
    'atunes': 'Atún',
    'bacalao': 'Bacalao',
    'bacalaos': 'Bacalao',
    'sardina': 'Sardina',
    'sardinas': 'Sardina',
    'lisa': 'Lisa',
    'lisas': 'Lisa',
    'carite': 'Carite',
    'carites': 'Carite',
    'robalo': 'Robalo',
    'robalos': 'Robalo',
    # Seafood — mariscos de uso común RD
    'camarón': 'Camarón',
    'camaron': 'Camarón',
    'camarones': 'Camarón',
    'langosta': 'Langosta',
    'langostas': 'Langosta',
    'langostino': 'Langostino',
    'langostinos': 'Langostino',
    'calamar': 'Calamar',
    'calamares': 'Calamar',
    'pulpo': 'Pulpo',
    'pulpos': 'Pulpo',
    'almeja': 'Almeja',
    'almejas': 'Almeja',
    'cangrejo': 'Cangrejo',
    'cangrejos': 'Cangrejo',
    'jaiba': 'Jaiba',
    'jaibas': 'Jaiba',
    'mejillón': 'Mejillón',
    'mejillon': 'Mejillón',
    'mejillones': 'Mejillón',
    'vieira': 'Vieira',
    'vieiras': 'Vieira',
}


def _get_extra_fish_seafood_keywords() -> dict[str, str]:
    """[P1-AUDIT-2 · 2026-05-10] Knob `MEALFIT_COHERENCE_FISH_KEYWORDS` para
    extensibilidad runtime sin redeploy.

    Formato: `kw1:Canonical1,kw2:Canonical2` (pares separados por coma).

    Ejemplo:
      export MEALFIT_COHERENCE_FISH_KEYWORDS="ostra:Ostra,ostras:Ostra,boquerón:Boquerón"

    Releído en cada llamada para permitir ajustes en caliente. Items con
    formato inválido (sin `:` o keys vacías) se ignoran silenciosamente —
    no rompemos el coherence guard por un knob mal escrito.

    Nota: bypassea `_knob_env_str` deliberadamente porque ese helper
    normaliza el valor a lowercase, lo que rompería el canonical
    case-sensitive (queremos 'Salmón', no 'salmón'). Registramos
    manualmente en `_KNOBS_REGISTRY` vía `_register_knob` para que
    el knob siga visible en `/admin/knobs`.
    """
    knob_name = "MEALFIT_COHERENCE_FISH_KEYWORDS"
    raw = os.environ.get(knob_name, "")
    try:
        from knobs import _register_knob
        _register_knob(knob_name, "str", "", raw, raw)
    except Exception:
        # Registro best-effort: si knobs.py cambió signature, no
        # rompemos el coherence guard.
        pass
    if not raw:
        return {}
    out: dict[str, str] = {}
    for pair in str(raw).split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        kw, canon = pair.split(":", 1)
        kw = kw.strip().lower()
        canon = canon.strip()
        if kw and canon:
            out[kw] = canon
    return out


def canonicalize_fish_seafood(name) -> str | None:
    """[P1-AUDIT-2 · 2026-05-10] Canonicaliza nombres de pescados y mariscos
    a su nombre base para uso simétrico en el coherence guard.

    Cubre el modo de falso positivo no atrapado por `canonicalize_protein`
    (que solo cubre pollo/cerdo/res) ni por el fallback genérico:
      - "filete de salmón guisado" (receta) vs. "Salmón" (lista) → ambos 'Salmón'.
      - "camarones a la plancha" vs. "Camarones" → ambos 'Camarón'.
      - "tilapia frita" vs. "Tilapia" → ambos 'Tilapia'.
      - "langostinos al ajillo" vs. "Langostino" → ambos 'Langostino'.

    Diferencias vs. `canonicalize_protein` y `canonicalize_pavo`:
      - NO hay 3 canonicales fijos: cada especie tiene su propio canonical
        (Salmón, Tilapia, Camarón, etc.). La diversidad zoológica del
        dominio justifica el mapping per-species.
      - SÍ singulariza: "camarones" → 'Camarón', "langostinos" → 'Langostino'
        (a diferencia de canonicalize_pavo que preserva preparaciones como
        canónicos aparte).
      - NO distingue fresh-vs-procesado dentro del guard: enlatado /
        ahumado deli / fingers / palitos / croquetas → None (productos
        derivados que NO equivalen al pescado fresco, master_map los
        canonicaliza por su lado si están definidos).

    Reglas:
      1. Buscar keywords del mapping en `name` (word-boundary regex).
      2. Si NO match → None (no es del dominio).
      3. Si TODOS los matches resuelven al MISMO canonical → ese canonical.
      4. Si resuelven a CANONICALES DISTINTOS (ej. "mero con salmón" —
         platillo mixto patológico) → None.
      5. Si el nombre indica producto derivado (enlatado, fingers, etc.) →
         None.

    Returns:
      Canonical string ('Salmón', 'Camarón', 'Tilapia', ...) o None.

    Knob `MEALFIT_COHERENCE_FISH_KEYWORDS` permite añadir especies regional
    o de marca sin redeploy (formato `kw:Canon,kw2:Canon2`).
    """
    if not name:
        return None
    n_low = str(name).strip().lower()

    full_map = {**_FISH_SEAFOOD_CANONICAL, **_get_extra_fish_seafood_keywords()}

    matched_canonicals: set[str] = set()
    for kw, canonical in full_map.items():
        if re.search(rf'\b{re.escape(kw)}\b', n_low):
            matched_canonicals.add(canonical)

    if len(matched_canonicals) != 1:
        return None

    # Excluir productos derivados — el master_map debe canonicalizar eso
    # por su lado (ej. "Atún en lata" ≠ "Atún fresco" para coherence).
    if re.search(
        r'\benlatad[oa]s?\b|\ben\s+lata\b|\bfingers?\b|\bpalitos?\b|'
        r'\bnuggets?\b|\bahumad[oa]s?\b|\bcroquetas?\b|\bbastones?\b|'
        r'\bsurimi\b|\bsucedáneo\b|\bsucedaneo\b',
        n_low,
    ):
        return None

    return next(iter(matched_canonicals))


def canonicalize_pavo(name) -> str | None:
    """[P3-4 · 2026-05-07] Canonicaliza un nombre que referencia pavo a uno
    de los cuatro canónicos del aggregator: 'Pechuga de pavo', 'Jamón de
    pavo', 'Pavo molido', 'Pavo'. Devuelve None si el nombre no menciona
    pavo o cae en un caso ambiguo (sin descriptor reconocido).

    Mirror simétrico de la regla fresh-vs-procesado del aggregator
    (`shopping_calculator.py:2865-2920`, [P3-PROTEIN-CAP-2]). El propósito
    es que el guard recetas↔lista pueda comparar magnitudes de productos
    de pavo sin caer en falsos positivos por divergencia entre el nombre
    de la receta ("pechuga de pavo fresca") y el nombre canonicalizado
    en la lista ("Pechuga de pavo").

    Reglas, en orden de precedencia (idéntico al aggregator):
      1. fresh marker (`fresca`/`fresh`) → 'Pechuga de pavo'
      2. processed marker (`jamón de pavo`, `pavo en lonjas`, `pavo
         procesado`, `pavo en rebanadas`) → 'Jamón de pavo'
      3. `pavo molido` o `carne de pavo` → 'Pavo molido'
      4. `pechuga de pavo` o `filete de pavo` (sin marker fresh/procesado)
         → 'Pechuga de pavo' (default seguro fresh)
      5. exact `'pavo'` (lower-stripped) → 'Pavo'
      6. cualquier otro caso (ej. "pavo guisado") → None.

    Nota: NO modifica el comportamiento del aggregator ni de
    `normalize_name`. Es una réplica de su contrato para uso simétrico
    desde el path de coherencia. Si el aggregator cambia su regla, este
    helper debe actualizarse — el test
    `test_p3_4_canonicalize_pavo_mirrors_aggregator` verifica el mirror.
    """
    if not name:
        return None
    n_low = str(name).strip().lower()
    if not re.search(r'\bpavo\b', n_low):
        return None
    if re.search(r'\bfresc[oa]s?\b|\bfresh\b', n_low):
        return 'Pechuga de pavo'
    if re.search(
        r'jam[oó]n\s+de\s+pavo|pavo\s+en\s+lonjas?|lonjas?\s+de\s+pavo|'
        r'pavo\s+procesado|pavo\s+en\s+rebanadas?',
        n_low,
    ):
        return 'Jamón de pavo'
    if re.search(r'\bpavo\s+molido\b|\bcarne\s+de\s+pavo\b', n_low):
        return 'Pavo molido'
    if re.search(r'\b(pechuga|filete)\s+de\s+pavo\b', n_low):
        return 'Pechuga de pavo'
    if n_low == 'pavo':
        return 'Pavo'
    return None


# ============================================================
# [P1-NEW-2 · 2026-05-11] Canonicalizers paralelos a `canonicalize_pavo`
# para 4 categorías que el guard recetas↔lista trataba como falsos
# positivos por equivalencia de presentaciones:
#
#   - canonicalize_huevo    (claras, yema, enteros → "Huevo")
#   - canonicalize_lacteo   (entera/descremada/light → producto base)
#   - canonicalize_grano    (integral/blanco/refinado → "Arroz"/"Avena")
#   - canonicalize_legumino (rojas/negras/blancas/secas/cocidas → base)
#
# Mismo contrato: devuelve canónico si hay match claro; None si el nombre
# NO menciona la categoría O cae en caso ambiguo. Defensivo by design:
# el guard sigue funcionando como antes cuando el helper retorna None.
#
# Tests E2E paralelos a `test_p3_4_pavo_coherence_v3.py`.
# ============================================================


def canonicalize_huevo(name) -> str | None:
    """[P1-NEW-2 · 2026-05-11] Canonicaliza nombres de huevo y derivados
    (claras, yemas, huevos enteros) a un único canónico 'Huevo'.

    Por qué existe:
      Pre-fix: una receta podía pedir "Claras de huevo (200g)" y la lista
      de compras agregar bajo "Huevos (3 unidades)" — el guard reportaba
      "Huevo missing" o "unit_mismatch" cuando, semánticamente, son el
      MISMO ingrediente shopping (el usuario compra huevos enteros, los
      separa). Este helper colapsa la equivalencia para que el guard
      compare cantidades sobre la misma key canónica.

    Reglas (orden de precedencia):
      1. Contiene `claras` (de huevo o solo) → 'Huevo' (claras vienen del
         huevo entero; el shopping list pide huevos enteros).
      2. Contiene `yema` (singular o plural) → 'Huevo' (idem).
      3. Contiene `huevo` (singular/plural, con/sin "de gallina") → 'Huevo'.
      4. Cualquier otro caso → None.

    NO toca productos derivados con nombre propio (tortilla, omelette,
    huevos endiablados) — esos son comidas, no ingredientes shopping.
    """
    if not name:
        return None
    n_low = str(name).strip().lower()
    # [P2-TRIAGE-REALBUGS · 2026-06-16] Exclusión de platos compuestos PRIMERO.
    # "tortilla", "omelette", "endiablado" son comidas, NO ingredientes shopping.
    # DEBE correr ANTES de los branches claras/yema: antes el bug de orden hacía
    # que "Omelette de claras" matcheara 'claras' y devolviera 'Huevo' en vez de
    # None (el docstring siempre documentó la intención de excluir estos platos).
    if re.search(r'\b(tortilla|omelette|omelete|endiablad)', n_low):
        return None
    # Claras (lab/preparados): "claras de huevo", "claras pasteurizadas".
    if re.search(r'\bclaras?\b', n_low):
        return 'Huevo'
    # Yemas: "yema de huevo", "yemas".
    if re.search(r'\byemas?\b', n_low):
        return 'Huevo'
    # Huevo en sus formas básicas.
    if re.search(r'\bhuevos?\b', n_low):
        return 'Huevo'
    return None


def canonicalize_lacteo(name) -> str | None:
    """[P1-NEW-2 · 2026-05-11] Canonicaliza nombres de lácteos a sus
    canónicos shopping eliminando presentaciones equivalentes
    (entera/descremada/light/deslactosada).

    Devuelve uno de:
      - 'Leche'  (cubre entera/descremada/semidescremada/deslactosada/light)
      - 'Yogur'  (cubre natural/griego/light/sin azúcar)
      - 'Queso fresco'  (default fresco si NO se identifica tipo madurado)
      - None  para cualquier otro lácteo (mantequilla, crema, productos
              compuestos como flan, helado — esos son shopping items
              distintos con cantidades propias).

    NO maneja marcas — eso lo cubre `_strip_dairy_brand` (P2-AUDIT-2).
    Este helper opera sobre el nombre POST-brand-strip.

    Conservador con quesos: si el nombre menciona un tipo concreto
    (mozzarella, cheddar, parmesano, manchego), retorna ese tipo
    capitalizado en lugar de colapsar a 'Queso fresco' — son shopping
    items distintos en RD.
    """
    if not name:
        return None
    n_low = str(name).strip().lower()

    # Leche: cualquier variante con descriptor entera/descremada/light.
    # No matchear "leche de coco" / "leche evaporada" / "leche condensada"
    # (productos distintos con cantidades propias).
    # [P1-NUT-MILK-DISTINCT · 2026-07-07] `almendras?` (plural) + maní/anacardo/arroz: la
    # receta dice "leche de almendraS" y el `almendra\b` singular NO matcheaba el plural →
    # caía a "Leche" (leche de vaca). Las leches vegetales NO son leche de vaca ni el fruto.
    if re.search(r'\bleche\s+de\s+(coco|almendras?|soja|soya|avena|man[íi]|anacardos?|arroz)\b', n_low):
        return None
    if re.search(r'\bleche\s+(evaporada|condensada|en\s+polvo)\b', n_low):
        return None
    if re.search(r'\bleche\b', n_low):
        return 'Leche'

    # Yogur: variantes natural/griego/light.
    # No matchear "yogur bebible saborizado" como mismo — generalmente
    # van separados en pantry. Pero "yogur natural" y "yogur griego" sí
    # comparten shopping key.
    if re.search(r'\byogur\b|\byogurt\b', n_low):
        return 'Yogur'

    # Queso: tipos concretos primero (no colapsar).
    queso_tipos = [
        'mozzarella', 'cheddar', 'parmesano', 'manchego', 'feta',
        'gouda', 'provolone', 'roquefort', 'brie', 'camembert',
        'ricotta', 'mascarpone', 'azul', 'gorgonzola',
    ]
    for tipo in queso_tipos:
        if re.search(rf'\b{tipo}\b', n_low):
            return tipo.capitalize()
    # Default: queso fresco / blanco / rallar — colapsar.
    if re.search(r'\bqueso\b', n_low):
        return 'Queso fresco'

    return None


def canonicalize_grano(name) -> str | None:
    """[P1-NEW-2 · 2026-05-11] Canonicaliza granos (arroz, avena, quinoa)
    a sus canónicos shopping eliminando presentaciones equivalentes
    (blanco/integral/refinado).

    Devuelve uno de:
      - 'Arroz'   (blanco/integral/parboiled colapsados — el usuario
                  compra el saco genérico y elige presentación).
      - 'Avena'   (hojuelas/molida/instantánea colapsados).
      - 'Quinoa'  (blanca/roja/tricolor colapsados).
      - None      para otros (cebada, mijo, etc. — sin demanda histórica).

    NO incluye trigo/pan/harina — esos son shopping items distintos
    (harina_integral vs harina_blanca tienen masas distintas en planes
    RD; el guard los trata como diferentes correctamente).
    """
    if not name:
        return None
    n_low = str(name).strip().lower()

    # Arroz: cualquier variante de presentación.
    if re.search(r'\barroz\b', n_low):
        return 'Arroz'
    # Avena: cualquier variante.
    if re.search(r'\bavena\b', n_low):
        return 'Avena'
    # Quinoa: cualquier color.
    if re.search(r'\bquinoa\b|\bquinua\b', n_low):
        return 'Quinoa'
    return None


def canonicalize_legumino(name) -> str | None:
    """[P1-NEW-2 · 2026-05-11] Canonicaliza legumbres (habichuelas,
    frijoles, lentejas, garbanzos) a sus canónicos shopping eliminando
    presentación (color/seco/cocido/enlatado).

    Devuelve uno de:
      - 'Habichuelas'  (rojas/negras/blancas/pintas → un solo canónico
                       — el usuario compra el saco; las recetas RD
                       intercambian colores libremente).
      - 'Lentejas'     (cualquier color).
      - 'Garbanzos'    (cualquier presentación).
      - None           para otras leguminosas (gandules, judías verdes —
                       en realidad gandules merece su propio canónico
                       en RD; lo retornamos aparte si está presente).

    NOTA es-DO: en RD "habichuelas" y "frijoles" son sinónimos
    intercambiables. Ambos colapsan a 'Habichuelas' (el canónico más
    frecuente en menús dominicanos).

    Gandules (Cajanus cajan) NO son habichuelas en sentido estricto
    — son leguminosa propia. Pre P1-NEW-2 el aggregator los listaba
    aparte; mantenemos ese contrato emitiendo 'Gandules' canónico.
    """
    if not name:
        return None
    n_low = str(name).strip().lower()

    # Gandules — propio canónico (no colapsar con habichuelas).
    if re.search(r'\bgandules?\b', n_low):
        return 'Gandules'

    # Habichuelas / frijoles — sinónimos RD, colapsan al canónico mayoritario.
    if re.search(r'\bhabichuelas?\b|\bfrijoles?\b|\bporotos?\b', n_low):
        return 'Habichuelas'

    # Lentejas — cualquier color/presentación.
    if re.search(r'\blentejas?\b', n_low):
        return 'Lentejas'

    # Garbanzos — cualquier presentación.
    if re.search(r'\bgarbanzos?\b', n_low):
        return 'Garbanzos'

    return None


# [P1-PREP-COLLAPSE-GUARD · 2026-07-01] (audit creatividad G3/G4, confirmado en vivo contra el catálogo)
# Una PREPARACIÓN "harina de X" es un PRODUCTO DISTINTO de X fresco (SKU, precio, macros ~3× en harinas de
# vívere) — generalización de la lección P1-NUT-BUTTER-DISTINCT. Sin este guard: (a) el alias 'harina'
# (Harina de trigo) ganaba por longest-first en el Tier-2 de normalize_name/_match_row → "harina de avena"
# resolvía a HARINA DE TRIGO (lista con gluten para un celíaco sin que el allergen-guard dispare — el string
# no dice "trigo"); (b) "harina de plátano" → Plátano verde fresco (macros sub-contados ~3×) vía Tier-2 y
# vía canonicalize_musaceae; (c) "tortilla de maíz" → Maíz dulce en granos (producto distinto; el catálogo
# solo tiene tortillas de trigo). Resolución explícita: preparaciones con equivalente real en el catálogo
# se canonizan (harina de avena→Avena molida ≈ mismos macros; harina de maíz→Harina de maíz precocida;
# harina de trigo→su propia fila); el resto se marca DISTINTO-sin-fila → normalize_name devuelve el nombre
# passthrough (no matchea master → verified-only lo dropea y el guard de coherencia ve ambos lados igual) y
# nutrition_db devuelve None (no computar macros del producto equivocado). tooltip-anchor: P1-PREP-COLLAPSE-GUARD
_PREP_FLOUR_RE = re.compile(r"\bharinas?\s+de\s+([a-z]+)")
# [P1-PREP-HEAD-GUARD · 2026-07-27] unidades que pueden preceder a "de harina..." sin cambiar el
# sustantivo cabeza ("1 taza de harina de trigo" ES harina; "tortilla de harina de trigo" NO).
_PREP_HEAD_UNITS = frozenset((
    "taza", "tazas", "cda", "cdas", "cdta", "cdtas", "cucharada", "cucharadas",
    "cucharadita", "cucharaditas", "g", "gr", "gramos", "kg", "ml", "l", "lb", "lbs",
    "libra", "libras", "oz", "onza", "onzas", "paquete", "paquetes", "funda", "fundas",
    "sobre", "sobres", "pizca", "porcion", "porciones", "mitad", "resto", "parte",
))
_PREP_FLOUR_CANON = {
    "avena": "Avena",
    "maiz": "Harina de maíz precocida",
    "trigo": "Harina de trigo",
}
_PREP_FLOUR_DISTINCT = (
    "platano", "platanos", "yuca", "arroz", "coco",
    "almendra", "almendras", "garbanzo", "garbanzos", "cebada", "quinoa",
)
_PREP_TORTILLA_MAIZ_RE = re.compile(r"\btortillas?\s+de\s+maiz\b")
# [P1-BROTH-NOT-MEAT · 2026-07-28] caldo de <lo que sea>: producto distinto de su ingrediente.
_PREP_BROTH_RE = re.compile(r"\bcaldos?\s+de\s+[a-z]")
_PREP_CREMA_COCO_RE = re.compile(r"\bcremas?\s+de\s+coco\b")


def resolve_preparation_distinct(name) -> tuple:
    """[P1-PREP-COLLAPSE-GUARD · 2026-07-01] Devuelve `(handled, canonical)`:
      - `(True, "<master name>")` → la preparación tiene equivalente real en el catálogo; resolver ahí.
      - `(True, None)` → producto DISTINTO sin fila propia; NO colapsar al alimento base (drop/pass-through).
      - `(False, None)` → no es una preparación cubierta; seguir con los tiers normales.
    Puro y determinista (sin catálogo) para que normalize_name, nutrition_db._match_row y los
    canonicalizers compartan exactamente las mismas reglas. tooltip-anchor: P1-PREP-COLLAPSE-GUARD"""
    if not name:
        return (False, None)
    try:
        from constants import strip_accents as _sa_prep
        low = _sa_prep(str(name).lower())
    except Exception:
        low = str(name).lower()
    m = _PREP_FLOUR_RE.search(low)
    if m:
        # [P1-PREP-HEAD-GUARD · 2026-07-27] "TORTILLA de harina de trigo (wrap, 60g)" resolvía a
        # HARINA DE TRIGO cruda: el regex casa "harina de X" en cualquier posición y el guard
        # secuestraba productos cuyo sustantivo cabeza es otro (tortilla/bollitos/pan — la harina
        # es el material, no el producto). Si antes de "harina" viene "<palabra> de ", solo es
        # producto-harina cuando esa palabra es una UNIDAD de medida ("1 taza de harina de trigo").
        _pre_m = re.search(r"([a-z]\w*)\s+de\s+$", low[:m.start()])
        if _pre_m and _pre_m.group(1) not in _PREP_HEAD_UNITS:
            return (False, None)  # cabeza ajena (tortilla/pan/bollitos) → tiers normales
        base = m.group(1)
        if base in _PREP_FLOUR_CANON:
            return (True, _PREP_FLOUR_CANON[base])
        if base in _PREP_FLOUR_DISTINCT:
            return (True, None)
        return (False, None)  # "harina de negrito" etc. → resuelven por su propia fila en los tiers
    if _PREP_TORTILLA_MAIZ_RE.search(low):
        # [P1-COUNTRY-SYSTEM-F2 · T6 · 2026-08-17] Antes de esta task el catálogo solo tenía
        # tortillas de TRIGO — el guard forzaba pass-through (True, None) para NO colapsar
        # "tortilla de maíz" a "Maíz dulce en granos" (kernel de maíz, macro ~4x distinto de una
        # tortilla horneada). Con la alta real "Tortilla de maíz" (USDA, T6) el guard PODRÍA
        # canonizar a su fila propia — pero "tortilla de maíz" es un string que YA vive en rutas
        # DO pre-existentes sin relación con esta task (`db_inventory.py` PANTRY_UNIT_HINTS línea
        # ~1907, `P6-CARBS-CAP`) y este resolver no recibe country: canonizar sin gate cambiaría
        # el comportamiento DO con el knob apagado (rompe byte-identidad — fix-round 1, review
        # Critical #2). Gateado por el MISMO knob maestro que `country_for_form_data`, leído
        # POR LLAMADA (no cacheado a nivel de módulo) — knob apagado ⇒ pass-through histórico
        # SIEMPRE, para cualquier país; knob encendido ⇒ canoniza (sigue sin colapsar al maíz
        # crudo en ningún caso).
        if _knob_env_bool("MEALFIT_COUNTRY_SYSTEM", False):
            return (True, "Tortilla de maíz")
        return (True, None)
    if _PREP_CREMA_COCO_RE.search(low):
        return (True, None)  # crema de coco ≠ coco fresco (SKU distinto)
    # [P1-BROTH-NOT-MEAT · 2026-07-28] "caldo de pollo" resolvía a PECHUGA DE POLLO y "caldo de
    # res" a CARNE DE RES (la subcadena 'pollo'/'res' ganaba el matching): la lista compraba
    # CARNE cuando la receta pedía caldo — 838 kg de pechuga en el caso extremo del test de caps.
    # El caldo es un producto DISTINTO sin fila propia en master_ingredients: se marca handled
    # sin canónico y sigue el mismo camino honesto que caldo de hueso/vegetales (drop del
    # verified-only con WARN de observabilidad), jamás la carne.
    if _PREP_BROTH_RE.search(low):
        return (True, None)
    return (False, None)


def canonicalize_viveres(name) -> str | None:
    """[P3-NEW-6 · 2026-05-11] Canonicaliza víveres dominicanos
    (tubérculos y raíces) a un canónico shopping fijo.

    Bug original (audit 2026-05-11): recetas con preparaciones múltiples
    de un mismo vívere ("Yuca hervida", "Yuca con mojo", "Yuca al
    ajillo") generaban 3 líneas separadas en la lista de compras aunque
    shopping-wise sean el mismo producto. Inflaba la lista y degradaba
    la UX de compras (más líneas que productos reales).

    Decisión: TODOS los yucas/yautías/batatas/papas/auyamas en
    cualquier preparación colapsan a su canónico fijo.

    Reglas (orden importa solo por early-return; prefijos son mutex):
      - yuca / yucas → "Yuca"
      - yautía / yautia / yautías / yautias → "Yautía"
      - batata / batatas → "Batata"
      - papa / papas → "Papa" (EXCEPTO si name contiene "papaya" —
        fruta, no tubérculo; falsa coincidencia de prefijo)
      - auyama / auyamas → "Auyama" (calabaza criolla RD, distinta
        de calabacín — no colapsan entre sí)

    NO incluye:
      - Ñame: ya cubierto por `_consolidate_inline_canon` desde P2-NEW-8.
      - Plátanos/guineos: musáceas, ver `canonicalize_musaceae`.
      - Tayota/remolacha/zanahoria: vegetales con shopping behavior
        distinto (rotación, presentación) — no víveres tradicionales
        RD.

    Args:
        name: candidato (str o `None`). Case-insensitive.

    Returns:
        Canonical name fijo si matchea; `None` si no aplica → el caller
        cae al siguiente canonicalizer o al fallback singularize/strip.
    """
    if not name:
        return None
    # [P1-PREP-COLLAPSE-GUARD · 2026-07-01] "harina de yuca" NO es yuca fresca — no colapsar.
    if resolve_preparation_distinct(name)[0]:
        return None
    n_low = str(name).lower()
    if re.search(r'\byucas?\b', n_low):
        return 'Yuca'
    if re.search(r'\byaut[ií]as?\b', n_low):
        return 'Yautía'
    if re.search(r'\bbatatas?\b', n_low):
        return 'Batata'
    if re.search(r'\bpapas?\b', n_low) and 'papaya' not in n_low:
        return 'Papa'
    if re.search(r'\bauyamas?\b', n_low):
        return 'Auyama'
    return None


def canonicalize_musaceae(name) -> str | None:
    """[P3-NEW-6 · 2026-05-11] Canonicaliza musáceas (plátano, guineo)
    a un canónico shopping fijo.

    Bug original (audit 2026-05-11): "Plátano verde para mangú",
    "Plátano maduro frito" y "Plátano maduro en almíbar" generaban 3
    líneas separadas en la lista de compras. El usuario compra los
    MISMOS plátanos — la madurez es variable temporal del producto (un
    plátano verde se convierte en maduro a los 5-7 días en cocina),
    no producto distinto.

    Decisión: TODOS los plátanos (cualquier estado o preparación)
    colapsan a "Plátano". Análogo al patrón de `canonicalize_viveres`
    (preparaciones múltiples → un canónico shopping).

    Reglas:
      - plátano / platano / plátanos / platanos → "Plátano"
      - guineo / guineos → "Guineo" (banano criollo, distinto del
        plátano — diferencia botánica + comercial real, no colapsa)

    Args:
        name: candidato (str o `None`). Case-insensitive. Acepta
              tildes y versiones sin tilde (el LLM puede emitir
              cualquier forma).

    Returns:
        Canonical name fijo si matchea; `None` si no aplica.
    """
    if not name:
        return None
    # [P1-PREP-COLLAPSE-GUARD · 2026-07-01] "harina de plátano" NO es plátano fresco (~3× kcal) — no colapsar.
    if resolve_preparation_distinct(name)[0]:
        return None
    n_low = str(name).lower()
    if re.search(r'\bpl[áa]tanos?\b', n_low):
        return 'Plátano'
    if re.search(r'\bguineos?\b', n_low):
        return 'Guineo'
    return None


def canonicalize_frutas_tropicales(name) -> str | None:
    """[P2-NEW-A · 2026-05-11] Canonicaliza frutas tropicales RD a un
    canónico shopping fijo.

    Bug observado en audit 2026-05-11: "Ensalada de mango con limón",
    "Mango verde rallado" y "Mango maduro en almíbar" generaban 3
    líneas separadas en la lista. Mismo modo de fallo que
    `canonicalize_viveres`/`canonicalize_musaceae`: preparaciones
    múltiples del MISMO producto inflan la lista de compras aunque
    shopping-wise sean idénticas.

    Reglas (orden por early-return; prefijos mutex):
      - mango / mangos → "Mango" (también en preparaciones: verde,
        maduro, en almíbar, etc.)
      - piña / pina / piñas / pinas → "Piña" (acepta sin tilde)
      - papaya / lechosa: AMBOS a "Lechosa" (canónico es-DO; en RD
        "lechosa" es el nombre común — incluido "papaya" porque el
        LLM puede emitir cualquiera). Solo matchea "lechosa/lechosas"
        en femenino, NO "lechoso/lechosos" (adjetivo lácteo) para
        evitar conflicto con `_strip_dairy_brand`.

    NO incluye:
      - Guineo/plátano: musáceas, ver `canonicalize_musaceae`.
      - Coco: tiene shopping behavior distinto (entero vs. rallado vs.
        leche de coco). NO colapsa.
      - Aguacate: aunque es fruta tropical, su shopping unit (unidad)
        difiere de las frutas que SÍ colapsan (lb). Mantenido separado.

    Args:
        name: candidato (str o `None`). Case-insensitive.

    Returns:
        Canonical name fijo si matchea; `None` si no aplica → el caller
        cae al siguiente canonicalizer o al fallback singularize/strip.
    """
    if not name:
        return None
    n_low = str(name).lower()
    if re.search(r'\bmangos?\b', n_low):
        return 'Mango'
    if re.search(r'\bpi[ñn]as?\b', n_low):
        return 'Piña'
    # `papaya`/`lechosa` (femenino solo) → "Lechosa". Match con singular
    # opcional. `\b` evita matchear `lechosamente` o similares.
    if re.search(r'\bpapayas?\b', n_low) or re.search(r'\blechosas?\b', n_low):
        return 'Lechosa'
    return None


def canonicalize_verduras_hoja(name) -> str | None:
    """[P2-NEW-A · 2026-05-11] Canonicaliza verduras de hoja verde a un
    canónico shopping fijo.

    Bug observado: variantes de lechuga ("lechuga romana", "lechuga
    americana", "lechuga criolla") generaban 3 líneas en la lista,
    pero el usuario compra UNA misma lechuga (o cualquiera que
    encuentre). La variedad es preferencia del LLM, no requisito
    del usuario; consolidar las 3 a "Lechuga" simplifica el shopping.

    Reglas:
      - lechuga / lechugas (cualquier variedad) → "Lechuga"
      - espinaca / espinacas → "Espinaca"
      - rúcula / rucula / rúculas / ruculas → "Rúcula"
      - acelga / acelgas → "Acelga"
      - berro / berros → "Berro"

    NO incluye:
      - Repollo: shopping unit (unidad) distinto de las hojas sueltas.
      - Col rizada / kale: poca presencia en planes RD; añadir cuando
        aparezca un caso real (orden de keywords está pensado para
        extensión sin re-orden).

    Args:
        name: candidato. Case-insensitive, acepta tildes opcionales.

    Returns:
        Canonical name fijo si matchea; `None` si no aplica.
    """
    if not name:
        return None
    n_low = str(name).lower()
    if re.search(r'\blechugas?\b', n_low):
        return 'Lechuga'
    if re.search(r'\bespinacas?\b', n_low):
        return 'Espinaca'
    if re.search(r'\br[úu]culas?\b', n_low):
        return 'Rúcula'
    if re.search(r'\bacelgas?\b', n_low):
        return 'Acelga'
    if re.search(r'\bberros?\b', n_low):
        return 'Berro'
    return None


def canonicalize_aceites(name) -> str | None:
    """[P2-NEW-A · 2026-05-11] Canonicaliza aceites a un canónico shopping
    fijo (preserva tipo de aceite — son productos distintos, NO colapsan
    entre sí).

    Bug observado: "aceite de oliva extra virgen", "aceite oliva
    prensado en frío", "AOVE" reportaban `cap_swallowed_modifier`
    falso positivo en el guard recetas↔lista — el master_map no listaba
    todas las variantes como aliases, y el aggregator tampoco las
    consolidaba. Resultado: la lista mostraba 2-3 líneas de oliva con
    cantidades fraccionadas en lugar de 1 línea sumada.

    Reglas (cada tipo PRESERVADO, solo se eliminan variantes
    cosméticas):
      - "aceite de oliva" / "aceite oliva" / "AOVE" (en cualquier
        forma: "extra virgen", "virgen", "prensado en frío",
        "primera prensada") → "Aceite de oliva"
      - "aceite de girasol" / "aceite girasol" → "Aceite de girasol"
      - "aceite de coco" / "aceite coco" → "Aceite de coco"
      - "aceite de aguacate" / "aceite aguacate" → "Aceite de aguacate"

    NO se colapsa "aceite de oliva" con "aceite de girasol": son
    productos distintos (precio, perfil graso, usos culinarios).

    NO incluye:
      - "aceite vegetal" genérico: ambiguo, no se canonicaliza para no
        ocultar al usuario que el LLM no especificó el tipo.
      - Mantequilla / margarina / ghee: shopping unit distinto (paquete),
        no son aceites en sentido shopping.

    Args:
        name: candidato. Case-insensitive.

    Returns:
        Canonical name fijo si matchea; `None` si no aplica.
    """
    if not name:
        return None
    n_low = str(name).lower()
    # Orden importa: "aceite de aguacate" antes que "aceite de" prefijos
    # genéricos. Cada tipo es mutex con los demás.
    if (
        re.search(r'\baceite\s+(?:de\s+)?oliva\b', n_low)
        or re.search(r'\baove\b', n_low)
    ):
        return 'Aceite de oliva'
    if re.search(r'\baceite\s+(?:de\s+)?girasol\b', n_low):
        return 'Aceite de girasol'
    if re.search(r'\baceite\s+(?:de\s+)?coco\b', n_low):
        return 'Aceite de coco'
    if re.search(r'\baceite\s+(?:de\s+)?aguacate\b', n_low):
        return 'Aceite de aguacate'
    return None


def canonicalize_citricos(name) -> str | None:
    """[P3-NEW-12 · 2026-05-11] Canonicaliza cítricos a canónicos shopping
    fijos (preserva tipo — son productos distintos, NO colapsan entre sí).

    Bug observado: "limón verde", "limón criollo", "limón persa" generaban
    3 líneas separadas en la lista, pero el usuario compra UN limón
    (cualquiera que encuentre). Variantes son preferencia del LLM, no
    requisito del usuario.

    Reglas (cada tipo PRESERVADO):
      - limón / limones (cualquier variedad: criollo, persa, verde) → "Limón"
      - lima / limas → "Lima"
      - naranja / naranjas (cualquier variedad: agria, dulce, valencia) → "Naranja"
      - mandarina / mandarinas → "Mandarina"
      - toronja / toronjas / pomelo(s) / grapefruit → "Toronja"

    NO colapsa cross-tipo: limón ≠ lima (precio + uso distintos).
    NO incluye:
      - Cidra / yuzu: poca presencia en RD; añadir cuando aparezca caso real.

    Tooltip-anchor: P3-NEW-12-CITRICOS

    Args:
        name: candidato. Case-insensitive, acepta tildes opcionales.

    Returns:
        Canonical name fijo si matchea; `None` si no aplica.
    """
    if not name:
        return None
    n_low = str(name).lower()
    if re.search(r'\blim[óo]n(?:es)?\b', n_low):
        return 'Limón'
    if re.search(r'\blimas?\b', n_low):
        return 'Lima'
    if re.search(r'\bnaranjas?\b', n_low):
        return 'Naranja'
    if re.search(r'\bmandarinas?\b', n_low):
        return 'Mandarina'
    if (
        re.search(r'\btoronjas?\b', n_low)
        or re.search(r'\bpomelos?\b', n_low)
        or re.search(r'\bgrapefruit\b', n_low)
    ):
        return 'Toronja'
    return None


def canonicalize_tomate(name) -> str | None:
    """[P3-NEW-12 · 2026-05-11] Canonicaliza variedades de tomate a "Tomate"
    (colapsado — son intercambiables para shopping en RD).

    Bug observado: "tomate perita", "tomate cherry", "tomate criollo",
    "tomate maduro" generaban 4 líneas en la lista, pero el usuario
    compra "tomate" en el supermercado/colmado sin pedir variedad
    específica (excepto cherry que SÍ es producto distinto).

    Reglas:
      - tomate cherry / tomates cherry / tomate uva / tomates uva
        → "Tomate cherry" (producto distinto, presentación pequeña)
      - tomate / tomates (cualquier OTRA variedad: perita, criollo,
        maduro, roma, ciruelo, manzano, italiano, plum) → "Tomate"

    NO incluye:
      - Tomate seco / sun-dried: producto procesado distinto.
      - Pasta/salsa de tomate: ya canonicalizados en el master_map.

    Tooltip-anchor: P3-NEW-12-TOMATE
    """
    if not name:
        return None
    n_low = str(name).lower()
    # [P3-TOMATE-SAUCE-FIX · 2026-06-22] Las formas PROCESADAS de tomate NO son tomate fresco:
    # salsa/pasta/puré/ketchup tienen su propio item de catálogo ("Salsa de tomate") y su propio
    # cap (P6-SAUCE-CAP, no P5-VEG-CAP); tomate seco/deshidratado es otro producto. El docstring
    # decía que se excluían "ya en el master_map" pero el `\btomates?\b` de abajo NO está anclado y
    # las capturaba → en planes reales "salsa de tomate" se canonicalizaba a "Tomate", se mezclaba
    # y costeaba como tomate de ensalada (disparaba P5-VEG-CAP en vez de P6-SAUCE-CAP). Early-return
    # None para que conserven su nombre y resuelvan a su master item. El callsite L6058 ya estaba
    # bien (ancla `^tomates?\b` + excluye salsa/pasta). Tooltip-anchor: P3-TOMATE-SAUCE-FIX.
    if "tomate" in n_low and re.search(r'\b(?:salsa|pasta|pur[eé]|ketchup|k[eé]tchup|catsup)\b', n_low):
        return None
    if re.search(r'\btomates?\s+(?:seco|secos|deshidratad)', n_low):
        return None
    # Cherry/uva PRIMERO (preserva como producto distinto).
    if re.search(r'\btomates?\s+(?:cherry|uva)\b', n_low):
        return 'Tomate cherry'
    if re.search(r'\btomates?\b', n_low):
        return 'Tomate'
    return None


def canonicalize_cebolla(name) -> str | None:
    """[P3-NEW-12 · 2026-05-11] Canonicaliza variedades de cebolla a
    "Cebolla" (colapsado — intercambiables para shopping RD).

    Bug observado: "cebolla roja", "cebolla morada", "cebolla blanca",
    "cebolla amarilla" generaban 4 líneas, pero el usuario compra
    "cebolla" sin pedir color específico (RD: cebolla roja es lo común).

    Reglas:
      - cebollín / cebollin / cebolla verde / cebolla de verdeo
        / cebolleta(s) → "Cebollín" (producto distinto — hierba aromática)
      - cebolla / cebollas (cualquier color: roja/morada/blanca/amarilla)
        → "Cebolla"

    NO incluye:
      - Ajo: ya canonicalizado en `_consolidate_inline_canon` (P2-NEW-8).
      - Puerro / leek: producto distinto, baja presencia.

    Tooltip-anchor: P3-NEW-12-CEBOLLA
    """
    if not name:
        return None
    n_low = str(name).lower()
    # Cebollín/cebolla verde PRIMERO (preserva como producto distinto).
    # Regex: alternación explícita porque `cebolli?nes?` falla con
    # "cebollin" (i required + e required en es?) y con "cebollín"
    # (tilde no en [i]). Mejor enumerar las variantes válidas.
    if (
        re.search(r'\b(?:cebollines|cebollínes|cebollín|cebollin)\b', n_low)
        or re.search(r'\bcebolla\s+verde\b', n_low)
        or re.search(r'\bcebolla\s+de\s+verdeo\b', n_low)
        or re.search(r'\bcebolletas?\b', n_low)
    ):
        return 'Cebollín'
    if re.search(r'\bcebollas?\b', n_low):
        return 'Cebolla'
    return None


def canonicalize_quesos_blancos_rd(name) -> str | None:
    """[P3-NEW-12 · 2026-05-11] Canonicaliza quesos blancos RD a un canónico
    shopping (colapsado bajo "Queso blanco" — variantes locales
    intercambiables).

    Bug observado: "queso frescal", "queso de freír", "queso blanco",
    "queso fresco" generaban 4 líneas, pero el usuario compra UN tipo
    de queso blanco RD (depende del supermercado local). Variantes son
    LLM-side, no shopping-side.

    Reglas:
      - queso de freír / queso frito → "Queso de freír" (producto
        distinto — alto punto fusión, para freír específicamente)
      - queso frescal / queso fresco / queso blanco → "Queso blanco"
      - mozzarella / mozarela → "Mozzarella" (producto distinto)
      - queso crema → "Queso crema" (producto distinto, untable)
      - cheddar / queso cheddar → "Cheddar"
      - parmesano / parmegiano / parmiggiano → "Parmesano"

    NO incluye:
      - "Queso" genérico sin modificador: ambiguo, no canonicaliza.
      - Quesos artesanales locales (de hoja, de pinitos, etc.): baja
        presencia, requieren caso real.

    Tooltip-anchor: P3-NEW-12-QUESOS-BLANCOS-RD
    """
    if not name:
        return None
    n_low = str(name).lower()
    # Orden importa: queso de freír antes que "queso blanco" genérico.
    if (
        re.search(r'\bqueso\s+(?:de\s+)?fre[íi]r\b', n_low)
        or re.search(r'\bqueso\s+frito\b', n_low)
    ):
        return 'Queso de freír'
    if re.search(r'\bqueso\s+crema\b', n_low):
        return 'Queso crema'
    if re.search(r'\bmozz?arell?a\b', n_low):
        return 'Mozzarella'
    if re.search(r'\bcheddar\b', n_low):
        return 'Cheddar'
    # Parmesano/parmegiano/parmigiano (incluye typos comunes con 'g').
    if (
        re.search(r'\bparmes(?:ano|iano)\b', n_low)
        or re.search(r'\bparmeg(?:ano|iano)\b', n_low)
        or re.search(r'\bparmigg?iano\b', n_low)
    ):
        return 'Parmesano'
    if (
        re.search(r'\bqueso\s+fresc?al\b', n_low)
        or re.search(r'\bqueso\s+fresco\b', n_low)
        or re.search(r'\bqueso\s+blanco\b', n_low)
    ):
        return 'Queso blanco'
    return None


def canonicalize_frutos_secos(name) -> str | None:
    """[P3-NEW-12 · 2026-05-11] Canonicaliza frutos secos a canónicos
    shopping fijos (preserva tipo — productos distintos, NO colapsan
    entre sí, mismo patrón que `canonicalize_aceites`).

    Bug observado: "almendra natural", "almendra tostada", "almendra
    laminada" generaban 3 líneas para el mismo producto base. Las
    preparaciones (tostado/laminado) son LLM detail; el shopping unit
    es "almendras" sin distinción.

    Reglas (cada tipo PRESERVADO):
      - almendra(s) (cualquier preparación) → "Almendras"
      - maní / mani / cacahuete(s) / cacahuate(s) → "Maní"
      - nuez / nueces (incluye nuez de castilla, walnut) → "Nueces"
      - avellana(s) → "Avellanas"
      - pistacho(s) → "Pistachos"
      - anacardo(s) / marañón(es) / cashew(s) → "Anacardos"
      - pecana(s) / nuez pecan / pecan(s) → "Pecanas"

    NO colapsa cross-tipo: almendra ≠ maní (precio + perfil graso
    distintos, alérgenos distintos).
    NO incluye:
      - Semillas (chía, lino, calabaza, girasol): categoría distinta
        (semillas, no nueces), requiere helper separado.
      - Frutos secos deshidratados (pasas, dátiles, ciruelas pasas):
        producto distinto (fruta deshidratada), no nuez.

    Tooltip-anchor: P3-NEW-12-FRUTOS-SECOS
    """
    if not name:
        return None
    n_low = str(name).lower()
    # [P1-NUT-BUTTER-DISTINCT · 2026-06-21] La mantequilla/crema/pasta de un fruto seco es un
    # PRODUCTO DISTINTO del fruto seco crudo (SKU, precio y presentación distintos): la
    # "Mantequilla de maní" (frasco RD$117) NO es "Maní" crudo (frasco RD$185); la crema de
    # almendra NO es "Almendras". NO consolidar a la nuez base — devolver None para que el
    # master_map resuelva el producto distinto por su propio nombre. Sin esto, el `\bman[íi]\b`
    # de abajo matcheaba "maní" DENTRO de "mantequilla de maní" → "Maní", y la lista de compras
    # contradecía la receta (la receta decía mantequilla de maní, la lista mostraba maní crudo
    # — el usuario compraría el producto equivocado). Simétrico en el coherence guard porque
    # ambos lados (recetas y lista) pasan por esta misma función. Tooltip-anchor: P1-NUT-BUTTER-DISTINCT.
    # [P1-NUT-MILK-DISTINCT · 2026-07-07] (review visual: "leche de almendras" → "Almendras
    # fileteadas" × 19 paquetes = RD$5,491, reventando el presupuesto) La LECHE de un fruto
    # seco es una BEBIDA distinta (cartón, cat. Lácteos, RD$260), NO el fruto seco crudo.
    # Sin este guard, "leche de almendras" caía al `\balmendras?\b` de abajo → "Almendras" →
    # consolidada a "Almendras fileteadas". Añadido "leche" al guard existente de
    # mantequilla/crema/pasta → None para que el master resuelva "Leche de almendras" exacto.
    if re.search(r'\b(mantequilla|crema|pasta|leche)\s+de\s+\w', n_low) or 'peanut butter' in n_low or 'almond butter' in n_low or 'almond milk' in n_low:
        return None
    if re.search(r'\balmendras?\b', n_low):
        return 'Almendras'
    if (
        re.search(r'\bman[íi]\b', n_low)
        or re.search(r'\bcacahuetes?\b', n_low)
        or re.search(r'\bcacahuates?\b', n_low)
    ):
        return 'Maní'
    if (
        re.search(r'\bpecanas?\b', n_low)
        or re.search(r'\bnuez\s+pecan\b', n_low)
        or re.search(r'\bpecans?\b', n_low)
    ):
        return 'Pecanas'
    if (
        re.search(r'\bnueces\b', n_low)
        or re.search(r'\bnuez\b', n_low)
        or re.search(r'\bwalnuts?\b', n_low)
    ):
        return 'Nueces'
    if re.search(r'\bavellanas?\b', n_low):
        return 'Avellanas'
    if re.search(r'\bpistachos?\b', n_low):
        return 'Pistachos'
    if (
        re.search(r'\banacardos?\b', n_low)
        or re.search(r'\bmarañ[óo]n(?:es)?\b', n_low)
        or re.search(r'\bcashews?\b', n_low)
    ):
        return 'Anacardos'
    return None


def _consolidate_inline_canon(name) -> str | None:
    """[P2-NEW-8 · 2026-05-11] SSOT para 4 reglas inline de canonicalización
    (Huevo / Ñame / Miel / Ajo) que antes vivían duplicadas en
    `_canonicalize_for_coherence` (cuerpo del guard recetas↔lista) y en
    `aggregate_and_deduct_shopping_list` (aggregator que produce el
    output de la lista de compras).

    Drift risk pre-P2-NEW-8: cuando una regla se actualizaba en un sitio
    sin tocar el otro, el guard reportaba false positives ("Huevo missing")
    porque expected_sum_from_recipes usaba la regla nueva y el aggregator
    seguía con la vieja (o viceversa). Pavo ya tenía test
    `test_p3_4_canonicalize_pavo_mirrors_aggregator` como espejo dedicado;
    estos 4 no tenían cobertura análoga.

    Reglas (orden importa por mutex de prefijos; no hay overlap pero
    early-return acelera el path común):
      1. Prefijo `huevo(s)?`, `clara(s) de huevo`, `yema(s) de huevo`
         → "Huevo".
      2. Prefijo `ñame` o `name` (palabra) → "Ñame".
      3. Prefijo `miel` (palabra) → "Miel".
      4. Prefijo `ajo` (palabra) o `diente(s) de ajo` → "Ajo", EXCEPTO
         si el nombre contiene 'polvo' (`ajo en polvo` es categoría
         distinta — se preserva como está).

    Args:
        name: nombre del alimento candidato. Aceptamos `str`, `None`,
              o cualquier objeto stringificable (lo casteamos a str
              internamente). Case-insensitive.

    Returns:
        Canonical name (string fijo: "Huevo" / "Ñame" / "Miel" / "Ajo")
        si alguna regla matchea. `None` si ninguna aplica — el caller
        mantiene el name original o cae a otros canonicalizers
        (canonicalize_pavo, canonicalize_protein, etc.).

    Tests: `tests/test_p2_new_8_inline_canon_ssot.py`.
    """
    if not name:
        return None
    n_low = str(name).lower()
    if re.search(r'^(huevos?|claras?\s+de\s+huevo|yemas?\s+de\s+huevo)', n_low):
        return 'Huevo'
    if re.search(r'^[ñn]ame\b', n_low):
        return 'Ñame'
    if re.search(r'^miel\b', n_low):
        return 'Miel'
    if (re.search(r'^ajo\b', n_low) or re.search(r'dientes?\s+de\s+ajo', n_low)) and 'polvo' not in n_low:
        return 'Ajo'
    return None


# [P1-1-COHERENCE-EDGE · 2026-05-10] Plurales irregulares es-DO frecuentes en
# nombres de comida. La regla heurística (strip `-s` cuando la previa es vocal)
# falla para palabras cuyo plural es `-es` y singular termina en consonante.
# Mapping explícito gana siempre.
_IRREGULAR_PLURALS_ES = {
    "limones": "limón",
    "jamones": "jamón",
    "frijoles": "frijol",
    "camarones": "camarón",
    "salmones": "salmón",
    "panes": "pan",
    "flores": "flor",
    "mariscos": "marisco",   # sí cae en heurística, pero explícito por uso frecuente
    "lácteos": "lácteo",
    "huevos": "huevo",
    "yogures": "yogur",
}

# [P1-1-COHERENCE-EDGE · 2026-05-10] Modificadores triviales que el master_map
# no siempre cubre como aliases. Se strippean SOLO si aparecen como sufijo
# trailing (último o penúltimo token) y solo si el resultado del strip queda
# ≥3 caracteres (evita degenerar "pan integral" → "pan integ" si el match
# fuera parcial, o nombres de 1-2 letras que serían ambiguos).
#
# Conservador por diseño: NO incluye "sin sal", "bajo en X", "light" — esos son
# productos diferentes a efectos nutricionales. Solo cubre presentaciones
# variantes del MISMO ingrediente shopping.
_TRAILING_MODIFIERS_ES = frozenset({
    # Presentación / preparación
    "fresco", "fresca", "frescos", "frescas",
    "congelado", "congelada", "congelados", "congeladas",
    "enlatado", "enlatada", "enlatados", "enlatadas",
    "natural", "naturales",
    "orgánico", "orgánica", "orgánicos", "orgánicas",
    # Color / cualidad cromática (variedades intercambiables a efectos shopping)
    "blanco", "blanca", "blancos", "blancas",
    "rojo", "roja", "rojos", "rojas",
    "verde", "verdes",
    "amarillo", "amarilla", "amarillos", "amarillas",
    "negro", "negra", "negros", "negras",
    # Procesamiento (lácteos)
    "descremado", "descremada",
    "semidescremado", "semidescremada",
    "entero", "entera",
    "deslactosado", "deslactosada",
    # Refinamiento (granos)
    "integral", "integrales",
    "refinado", "refinada", "refinados", "refinadas",
})


def _build_shopping_master_map() -> dict:
    """[P1-VEG-BACKFILL-HONESTY · 2026-08-03] SSOT del índice nombre/alias -> fila de
    `master_ingredients`. Extraído de `aggregate_and_deduct_shopping_list` (bloque "RESOLUCIÓN DE
    FRICCIÓN DE UNIDADES") para que otros consumidores (`get_shopping_list_delta::text_demand_g_
    map`) puedan canonicalizar con la MISMA tabla que usa el lado comprado, sin reimplementar el
    índice ni arriesgar drift."""
    master_map: dict = {}
    for m in get_master_ingredients():
        master_map[m["name"]] = m
        for alias in (m.get("aliases") or []):
            master_map[alias.strip().lower()] = m
            master_map[alias.strip().title()] = m
    return master_map


def canonicalize_shopping_food_name(name: str, master_map: dict) -> str:
    """[P1-VEG-BACKFILL-HONESTY · 2026-08-03] SSOT de la cadena de canonicalización por-nombre
    que `aggregate_and_deduct_shopping_list` aplicaba INLINE dentro de su loop de agregación
    (`master_map` lookup → `_consolidate_inline_canon` → familias de canonicalizers →
    pavo fresh/procesado → 13 regex de consolidación de cola). Extraída (ronda de revisión
    2026-08-03 de esta misma tarea) para que el backstop de texto
    (`get_shopping_list_delta::text_demand_g_map`, P1-VEG-BACKFILL-HONESTY) resuelva el MISMO
    nombre final que el lado comprado — antes cada lado tenía su propia copia y podían divergir
    silenciosamente.

    Medido en vivo: sin este SSOT, "300 g de tomates" parseaba a `'Tomates'` (el parser NO
    singulariza) y el `text_demand_g_map` quedaba con esa key — pero el lado comprado, que SÍ pasa
    por esta cadena, resolvía `'Tomate'` (vía `canonicalize_tomate`). El backstop quedaba MUDO
    para cualquier receta en plural (la mitad de las veces, así escriben los LLM) — mismo modo de
    fallo para "cebollas" (`canonicalize_cebolla` → 'Cebolla') y "espinacas"
    (`canonicalize_verduras_hoja` → 'Espinaca', singular).

    NO cubre el post-proceso CROSS-KEY de `aggregate_and_deduct_shopping_list` (PLURAL-MERGE /
    P6-LACTEOS-MERGE): esos dos pasos deciden entre variantes que YA coexisten como keys en el
    dict agregado completo (ej. "si además existe la key hermana en singular, cuál gana") — no
    son una función pura de un solo nombre, así que replicarlos aquí exigiría construir el dict
    completo del lado texto, que es más de lo que este backstop necesita (fail-open: sin esa
    fusión final, el peor caso es que el backstop no dispare para esa fusión específica, nunca
    que dispare mal).

    Args:
        name: nombre crudo (post-parse, pre-canonicalización) — mismo `name` que devuelve
            `_parse_quantity`.
        master_map: índice de `_build_shopping_master_map()` (o el que ya tenga el caller — el
            aggregator reusa el suyo, ya construido para esta misma llamada).

    Returns:
        Nombre canónico. Igual a `name` si ninguna regla de la cadena aplica.
    """
    m_item = master_map.get(name) or master_map.get(name.lower()) or master_map.get(name.title())
    canonical_name = m_item["name"] if m_item else name

    # [P1-COUNTRY-SYSTEM-F2 · T7 fix-round 1 · 2026-08-17] La identidad EXACTA de una fila de
    # catálogo-país (altas T5/T6/T7, `is_country_catalog_unpriced_item`) es AUTORITATIVA — salta
    # la cadena de canonicalizers genéricos de abajo, que fue diseñada para PREPARACIONES/variantes
    # de un mismo alimento RD (viveres, musáceas, quesos blancos...), no para decidir la identidad
    # de un alimento de otro país que ya resolvió exacto. Sin este salto, 8 filas país (6 T7 + 2 T5
    # encontradas por el mismo sweep) quedaban sobreescritas DESPUÉS de resolver correctamente —
    # 'Nueces pecanas' incluso se perdía por completo (→ 'Pecanas', string sin fila real,
    # DROPEADO en silencio por el gate verified-only aguas abajo).
    #
    # Sweep de las 346 filas vivas (`task-7-report.md` §Fix round 1) confirmó que la alternativa
    # — saltar la cadena para CUALQUIER match exacto — NO es segura: 13 filas PRE-EXISTENTES (no
    # de catálogo-país) dependen de esta cadena a propósito para colapsar variantes/preparación a
    # un display de compra más simple (ej. 'Plátano verde'/'Plátano maduro' → 'Plátano' vía
    # `canonicalize_musaceae`, 'Queso cheddar' → 'Cheddar', 'Clara de huevo'/'Yema de huevo' →
    # 'Huevo' vía `_consolidate_inline_canon`) — saltar la cadena para esas 13 rompería ese
    # comportamiento DO ya establecido. El scope a `is_country_catalog_unpriced_item` dejа esas 13
    # intactas (ninguna es un token de catálogo-país) y solo activa el atajo para los 140 tokens
    # de ES/MX/CO/PR/US, indiferente al estado del knob salvo 'tortilla de maiz' (mismo criterio
    # que la propia función ya aplica) — un ingrediente DO nunca matchea ninguno de esos 140
    # tokens (ya verificado por el sweep de colisión T7: 140/140 exactos, 0 falsos positivos
    # contra el catálogo+pools completo), así que esta rama es un no-op byte-idéntico para DO.
    if m_item and is_country_catalog_unpriced_item(canonical_name):
        return canonical_name

    _inline_canon = _consolidate_inline_canon(canonical_name)
    if _inline_canon is not None:
        canonical_name = _inline_canon
    else:
        _viv = canonicalize_viveres(canonical_name)
        if _viv is not None:
            canonical_name = _viv
        else:
            _mus = canonicalize_musaceae(canonical_name)
            if _mus is not None:
                canonical_name = _mus
            else:
                # [P2-NEW-A · 2026-05-11] Frutas tropicales / verduras de hoja / aceites: tres
                # familias más cuyas variantes inflan la lista si no consolidamos. Mismo orden y
                # patrón que el guard (mirror): primer match gana, cada uno mutex.
                _fr = canonicalize_frutas_tropicales(canonical_name)
                if _fr is not None:
                    canonical_name = _fr
                else:
                    _vh = canonicalize_verduras_hoja(canonical_name)
                    if _vh is not None:
                        canonical_name = _vh
                    else:
                        _ac = canonicalize_aceites(canonical_name)
                        if _ac is not None:
                            canonical_name = _ac
                        else:
                            # [P3-NEW-12 · 2026-05-11] 5 canonicalizers adicionales (cítricos,
                            # tomate, cebolla, quesos blancos RD, frutos secos). Mismo patrón
                            # mirror que P2-NEW-A. Sin estos, variantes triviales como "limón
                            # verde" vs "limón persa" o "tomate criollo" vs "tomate maduro" se
                            # quedan en líneas separadas en la lista de compras.
                            #
                            # [review final · 2026-08-03] Este comentario —y el de P2-NEW-A de
                            # arriba— viajaron con el bloque al extraerlo del agregador a esta
                            # función: `test_p3_new_12_canonicalizers_rd_extra::test_marker_
                            # present_in_both_sites` exige el marker en las DOS mitades del
                            # espejo (guard + lista) para que un revisor las correlacione por
                            # grep, y se había quedado sin la mitad de la lista. Tercera vez que
                            # esta rama pierde un marker al mover código: los comentarios se
                            # mueven CON él.
                            _cit = canonicalize_citricos(canonical_name)
                            if _cit is not None:
                                canonical_name = _cit
                            else:
                                _tom = canonicalize_tomate(canonical_name)
                                if _tom is not None:
                                    canonical_name = _tom
                                else:
                                    _ceb = canonicalize_cebolla(canonical_name)
                                    if _ceb is not None:
                                        canonical_name = _ceb
                                    else:
                                        _qb = canonicalize_quesos_blancos_rd(canonical_name)
                                        if _qb is not None:
                                            canonical_name = _qb
                                        else:
                                            _fs = canonicalize_frutos_secos(canonical_name)
                                            if _fs is not None:
                                                canonical_name = _fs

    # Ver comentario original en `aggregate_and_deduct_shopping_list`: SOLO `name.lower()` (raw
    # del parser), NO el post-master_map — un alias que ya canonicalizó "Pechuga de pavo" →
    # "Jamón de pavo" auto-activaría la regex procesada aunque el LLM dijo "pechuga fresca".
    _orig_name_lower = name.lower()
    if re.search(r'\bpavo\b', _orig_name_lower):
        _has_fresh_marker = bool(re.search(r'\bfresc[oa]s?\b|\bfresh\b', _orig_name_lower))
        _has_processed_marker = bool(re.search(
            r'jam[oó]n\s+de\s+pavo|'
            r'pavo\s+en\s+lonjas?|'
            r'lonjas?\s+de\s+pavo|'
            r'pavo\s+procesado|'
            r'pavo\s+en\s+rebanadas?',
            _orig_name_lower
        ))
        if _has_fresh_marker:
            canonical_name = 'Pechuga de pavo'
        elif _has_processed_marker:
            canonical_name = 'Jamón de pavo'
        elif re.search(r'pavo\s+molido|carne\s+de\s+pavo', _orig_name_lower):
            canonical_name = 'Pavo molido'
        elif re.search(r'pechuga\s+de\s+pavo|filete\s+de\s+pavo', _orig_name_lower):
            canonical_name = 'Pechuga de pavo'
        elif _orig_name_lower.strip() == 'pavo':
            canonical_name = 'Pavo'

    # [P0-SHOPPING-CALC-NAMEERROR · 2026-05-15] `_can_lower` se usa en las 13 regex de
    # consolidación de abajo (Fresas, Almendras, Orégano, Tortilla, Tomate, Cebolla, Espinacas,
    # Zanahoria, Vainitas, Habichuelas, Tofu, Perejil). Pre-fix la variable nunca se asignaba en
    # este scope → `NameError: name '_can_lower' is not defined` en cada plan generado, lo que
    # tumbaba toda la agregación (`aggregate_and_deduct_shopping_list` lanzaba) y dejaba la lista
    # de compras vacía/incompleta. Síntoma user-facing: el coherence guard reportaba 35
    # "divergencias críticas" (todos los ingredientes de las recetas marcados como
    # `presence=expected_only`) y el plan llegaba con `_shopping_coherence_block` no resuelto.
    # IMPORTANTE: se calcula DESPUÉS del bloque pavo porque el pavo puede mutar `canonical_name`;
    # las 13 regex de abajo necesitan ver el `canonical_name` post-pavo.
    #
    # [P1-VEG-BACKFILL-HONESTY · 2026-08-03 · review final] Este comentario, y el del orégano de
    # tres reglas más abajo, se perdieron al extraer el bloque desde
    # `aggregate_and_deduct_shopping_list` a esta función: dos tests parser-based anclaban en ellos
    # y quedaron rojos, y con ellos se fue la razón escrita de una decisión de producto no obvia.
    # Es la inversión exacta de la convención del repo («incluir tooltip-anchor en el código fuente
    # para que un renombre falle el test antes de cambiar producción»): el anchor se borró y el
    # test murió con él. Al mover código, los comentarios se mueven CON él.
    # tooltip-anchor: P0-SHOPPING-CALC-NAMEERROR
    _can_lower = canonical_name.lower()

    # Consolidación: Fresas variantes (congeladas, frescas) → Fresas
    if re.search(r'^fresas?\b', _can_lower):
        canonical_name = 'Fresas'
    # Consolidación: Almendras variantes → Almendras fileteadas
    if re.search(r'^almendras?\b', _can_lower) and 'mantequilla' not in _can_lower:
        canonical_name = 'Almendras fileteadas'
    # [P3-OREGANO-DISPLAY-NAME · 2026-06-20] Variantes de orégano (seco, dominicano) → 'Orégano'
    # (display; el owner pidió quitar 'dominicano', redundante en es-DO). Este literal ES el
    # nombre mostrado/almacenado en `aggregated_shopping_list` (NO `master_ingredients.name`).
    # 'Orégano' resuelve en `master_map` para el lookup de precio/envase vía el alias
    # 'orégano'.title()='Orégano' del catálogo (slug='oregano'). NO revertir sin re-alinear.
    # tooltip-anchor: P3-OREGANO-DISPLAY-NAME
    if re.search(r'^or[eé]gano\b', _can_lower):
        canonical_name = 'Orégano'
    if re.search(r'^tortillas?\s+integral', _can_lower):
        canonical_name = 'Tortilla integral'
    if re.search(r'^tomates?\b', _can_lower) and 'pasta' not in _can_lower and 'salsa' not in _can_lower:
        canonical_name = 'Tomate'
    if re.search(r'^cebollas?\s+(blanca|roja|morada|amarilla)', _can_lower):
        canonical_name = 'Cebolla'
    if re.search(r'^espinacas?$', _can_lower):
        canonical_name = 'Espinacas'
    if re.search(r'^zanahorias?$', _can_lower):
        canonical_name = 'Zanahoria'
    if re.search(r'^vainitas?$', _can_lower):
        canonical_name = 'Vainitas'
    if re.search(r'^habichuelas?$', _can_lower):
        canonical_name = 'Habichuelas'
    if re.search(r'^tofu\b', _can_lower):
        canonical_name = 'Tofu'
    if re.search(r'\bperejil\b', _can_lower):
        canonical_name = 'Perejil'

    return canonical_name


def _singularize_food_es(name: str) -> str:
    """[P1-1 · 2026-05-10] Singulariza un nombre de comida en español.

    Estrategia:
      1. Mapping explícito de plurales irregulares (`limones → limón`).
      2. Heurística defensiva: si termina en `-s` Y el char previo es vocal
         (a/e/i/o/u + acentuadas) Y el resultado queda ≥3 caracteres → strip
         la `-s` final (preservando case original).
      3. En cualquier otro caso, devolver el nombre intacto (case-preserved).

    Es-DO conservador. Acepta el riesgo de no singularizar formas que la
    heurística no cubre (`papas fritas` → no toca, `arroces` → no toca por no
    estar en mapping; el guard caerá a presence/absence en ese caso).
    """
    if not name or not isinstance(name, str):
        return name
    stripped = name.strip()
    if not stripped:
        return name
    n_low = stripped.lower()
    # 1. Mapping explícito (devuelve canónico en lower — los plurales irregulares
    #    siempre mapean al singular conocido en lowercase).
    if n_low in _IRREGULAR_PLURALS_ES:
        return _IRREGULAR_PLURALS_ES[n_low]
    # 2. Heurística vowel-before-s sobre el string ORIGINAL (preserva case).
    if len(stripped) >= 4 and stripped.endswith(("s", "S")):
        prev = stripped[-2].lower()
        if prev in "aeiouáéíóú":
            return stripped[:-1]
    # 3. Sin transformación → return case-preserved.
    return stripped


def _strip_trailing_modifier_es(name: str) -> str:
    """[P1-1 · 2026-05-10] Quita modificador trivial al final del nombre.

    Ej: `pollo orgánico` → `pollo`; `arroz integral` → `arroz`. Solo strippea
    si el modificador está en `_TRAILING_MODIFIERS_ES` y el resultado queda
    ≥3 caracteres. Si no hay modificador trailing reconocido, devuelve el
    nombre sin tocar.

    Aplica máximo UN strip por invocación (cubre el caso común "X color"
    sin colapsar "X fresco orgánico" — formas más complejas son raras y
    el riesgo de over-stripping las hace no rentables).
    """
    if not name or not isinstance(name, str):
        return name
    parts = name.strip().split()
    if len(parts) < 2:
        return name
    last = parts[-1].lower()
    if last not in _TRAILING_MODIFIERS_ES:
        return name
    remainder = " ".join(parts[:-1]).strip()
    if len(remainder) < 3:
        return name
    return remainder


# [P2-AUDIT-2 · 2026-05-10] Marcas comerciales de lácteos en RD que la
# receta del LLM puede emitir como modificador del nombre del ingrediente
# ("Leche Induvaca entera", "Yogurt Rica") pero el aggregator agrupa en el
# canónico base ("Leche", "Yogurt"). Sin strip de marca, el guard ve
# nombres distintos y reporta `cap_swallowed_modifier` falso positivo.
#
# Comprehensive es-DO (auditado contra catálogo de supermercados típicos
# Nacional/Jumbo/PriceSmart). Extensiones runtime via
# `MEALFIT_COHERENCE_DAIRY_BRANDS`.
_DAIRY_BRANDS_ES_DO = frozenset({
    "induvaca",
    "rica",
    "sosúa", "sosua",
    "yoplait",
    "parmalat",
    "pasteurizadora rica",
    "cofadel",
    "río san juan", "rio san juan",
    "santa clara",
    "milky",
    "lala",
    "yogu",
})

# Productos lácteos: si el nombre menciona alguno de estos, intentamos
# strip de marca. Si el nombre NO menciona lácteo, no tocamos (evita
# falsos positivos: "rica salsa" donde "rica" es adjetivo, no marca).
_DAIRY_PRODUCT_KEYWORDS = frozenset({
    "leche", "yogurt", "yogur", "yoghurt",
    "queso", "quesos", "mantequilla", "crema",
    "natilla", "kéfir", "kefir", "requesón", "requeson",
})


def _get_extra_dairy_brands() -> set[str]:
    """[P2-AUDIT-2 · 2026-05-10] Knob `MEALFIT_COHERENCE_DAIRY_BRANDS` (CSV)
    para añadir marcas regionales/nuevas sin redeploy.

    Formato: `marca1,marca2,marca con espacios`. Todo se lowercased.
    Default vacío. Releído en cada llamada.

    Nota: bypasea `_knob_env_str` porque ese helper aplica lowercase
    normalization sobre TODO el string (incluyendo separadores). Aquí
    necesitamos preservar comas como separadores, hacer trim + lower
    item por item. Registro manual via `_register_knob`.
    """
    knob_name = "MEALFIT_COHERENCE_DAIRY_BRANDS"
    raw = os.environ.get(knob_name, "")
    try:
        from knobs import _register_knob
        _register_knob(knob_name, "str", "", raw, raw)
    except Exception:
        pass
    if not raw:
        return set()
    out: set[str] = set()
    for brand in str(raw).split(","):
        b = brand.strip().lower()
        if b:
            out.add(b)
    return out


def _strip_dairy_brand(name: str) -> str:
    """[P2-AUDIT-2 · 2026-05-10] Quita marca comercial de lácteo del nombre
    si y solo si el nombre menciona un producto lácteo.

    Por qué la condición doble:
      Sin el gate de keyword lácteo, strippear `rica` de "rica salsa
      picante" rompería el canonical de salsas (adjective vs brand).
      Solo aplicamos strip cuando el contexto es lácteo (lowercased).

    Ejemplo:
      "Leche Induvaca entera"   → "Leche entera"
      "Yogurt Rica griego"      → "Yogurt griego"
      "Queso Sosúa rallado"     → "Queso rallado"
      "Rica salsa picante"      → "Rica salsa picante"  (no lácteo → no toca)
      "Mantequilla"             → "Mantequilla"  (sin marca → no toca)

    Returns: nombre con marca strippeada (case-preserved del resto), o el
    nombre original si no aplica.
    """
    if not name or not isinstance(name, str):
        return name
    n_low = name.lower()
    # Gate: solo strippeamos si el nombre menciona producto lácteo.
    has_dairy_keyword = any(
        re.search(rf"\b{re.escape(kw)}\b", n_low)
        for kw in _DAIRY_PRODUCT_KEYWORDS
    )
    if not has_dairy_keyword:
        return name
    # Set combinado: defaults + extensiones por knob.
    all_brands = set(_DAIRY_BRANDS_ES_DO) | _get_extra_dairy_brands()
    result = name
    # Strip cada marca con word-boundary. Ordenar por longitud DESC para
    # que "pasteurizadora rica" se intente antes que "rica" (evita
    # strip parcial del primer match).
    for brand in sorted(all_brands, key=len, reverse=True):
        # Case-insensitive replace preservando el resto.
        pattern = re.compile(rf"\b{re.escape(brand)}\b", re.IGNORECASE)
        if pattern.search(result):
            result = pattern.sub("", result)
            # Colapsar dobles espacios + trim.
            result = re.sub(r"\s+", " ", result).strip()
    return result


# [P1-CAPS-COHERENCE-RECONCILE · 2026-05-16] Tracker module-level de los caps
# aplicados durante el último run de `aggregate_and_deduct_shopping_list`.
#
# Motivación: los caps (P3-HERB-CAP, P5-VEG-CAP, P6-LEGUMES-DRY-CAP, P6-EGGS-AGGREGATE-CAP,
# P6-LACTEOS-PERISHABLE-CAP, P6-SPICE-CAP, etc.) recortan magnitudes
# INTENCIONALMENTE por storage realism (cilantro 933g→100g porque no se
# almacena >1 semana; gandules 2333g→907g porque 1 paquete 1lb es suficiente
# para 1 person-week). Pre-fix, el coherence guard comparaba
# `expected_sum_from_recipes` (sin caps) vs `aggregated_shopping_list` (con
# caps) → las magnitudes divergían → guard reportaba "61 divergencias críticas"
# como falsos positivos legítimos → UI mostraba "Verificación médica con
# observaciones" en planes válidos.
#
# Fix: cada cap registra metadata aquí. El guard consulta la lista y filtra
# divergencias `magnitude` cuyo food matchea un cap aplicado (canonicalmente).
# Knob kill switch `MEALFIT_COHERENCE_CAP_AWARE` (default True).
#
# [P3-CAPS-COHERENCE-RECONCILE-3 · 2026-05-30] CLASE CERRADA: los 16 caps están
# instrumentados en TODAS sus ramas de magnitud (HERB, VEG, OLIVE, CITRUS, SPICE,
# LEGUMES-DRY, CANNED-PROTEIN, EGGS-AGGREGATE, LACTEOS-PERISHABLE, FRUITS-LARGE,
# FRUITS-PERISHABLE, CARBS, SAUCE, OIL, SWEETENER, BROTHS). Registrar es aditivo y
# dirección-segura (los caps solo reducen over-buy) → el guard nunca ve un FP de
# magnitud de un cap, sin importar la unidad nativa que emitió el LLM. Si añades un
# cap NUEVO, registra `_record_cap_applied(name, pre, post, "MARKER")` en CADA rama
# que modifique `_units[...]` y añade el marker al parametrize de
# `test_p1_caps_coherence_reconcile.py::test_cap_callsite_records_metadata`.
# [P1-CAPPED-STAPLE-HONESTY · 2026-07-26] Kill switch del sufijo "alcanza ~N de 30 días" en los
# items que un cap de realismo recortó. Flip a False si el copy molesta; el número NO cambia con el
# knob (el cap sigue aplicándose), sólo se deja de decir.
CAPPED_STAPLE_HONESTY = _knob_env_bool("MEALFIT_CAPPED_STAPLE_HONESTY", True)

_CAPS_APPLIED_LAST_RUN: list = []


def reset_caps_applied_last_run() -> None:
    """[P1-CAPS-COHERENCE-RECONCILE · 2026-05-16] Limpia el tracker antes de
    cada nuevo run de `aggregate_and_deduct_shopping_list`. Sin esto, runs
    consecutivos acumulan caps de ejecuciones previas → el guard ve "caps
    fantasmas" que no aplicaron a este plan."""
    _CAPS_APPLIED_LAST_RUN.clear()


def _cap_log(msg: str) -> None:
    """[P2-CAP-LOG-LEVEL · 2026-07-29] Canal de los topes de perecederos: INFO, no WARNING.

    Los emisores de cap narraban con `logging.warning` una decisión de PRODUCTO ya tomada ("no se
    compran 30 días de tomate fresco de golpe"), consumida aguas abajo y además comunicada al usuario
    en el display del item. Medido en 8 h de producción: **343 de 460 WARNING = 74,6% del journal**
    eran esto (P5-VEG-CAP 110, P6-LACTEOS-PERISHABLE-CAP 70, P6-CITRUS-CAP 62, P3-HERB-CAP 48…). Un
    operador que abre el journal tras un incidente ve cientos de líneas de una decisión SANA, y las
    6 señales reales del día quedan enterradas entre ellas.

    **Un evento de diseño que ocurre cientos de veces no puede ser WARNING: el nivel deja de
    significar "mira esto".**

    El mensaje NO cambia (misma información, mismo marker, greppable igual); solo baja de nivel. Lo
    que sí es señal se emite UNA vez por corrida en `_log_severe_caps_summary`, en vez de repetirse
    por ítem. Rollback: MEALFIT_CAP_LOG_SEVERE_RATIO=1.0 → todo vuelve a WARNING.
    tooltip-anchor: P2-CAP-LOG-LEVEL"""
    if _CAP_LOG_SEVERE_RATIO >= 1.0:
        logging.warning(msg)     # rollback: comportamiento previo
    else:
        logging.info(msg)


_LAST_SEVERE_CAPS_SIG: set = set()
_LAST_SEVERE_CAPS_AT: float = 0.0
_SEVERE_CAPS_DEDUP_TTL_S = _knob_env_float("MEALFIT_CAP_SUMMARY_DEDUP_TTL_S", 120.0,
                                           lambda v: 0.0 <= v <= 3600.0)


def _log_severe_caps_summary() -> None:
    """[P2-CAP-LOG-LEVEL · 2026-07-29] UN warning por corrida con los topes que sí son señal.

    Un cap que recorta a la MITAD o más ya no habla del tope: habla del MENÚ (el planificador pidió
    tanto de un perecedero que el tope tuvo que quitar la mayoría). Eso merece el canal ruidoso —
    pero una vez, agregado, no 343 veces por ítem. La fuente es `_CAPS_APPLIED_LAST_RUN`, que ya se
    puebla siempre y es lo que consumen el guard de coherencia y el reconcile."""
    if _CAP_LOG_SEVERE_RATIO >= 1.0:
        return
    try:
        _sev = []
        _claves = set()
        for _c in (_CAPS_APPLIED_LAST_RUN or []):
            _pre = float(_c.get("pre_value") or 0.0)
            _post = float(_c.get("post_value") or 0.0)
            if _pre > 0 and (_post / _pre) < _CAP_LOG_SEVERE_RATIO:
                _sev.append(f"{_c.get('food')} {_pre:.0f}→{_post:.0f}g "
                            f"({_post / _pre:.0%}, {_c.get('reason')})")
                _claves.add(f"{_c.get('food')}|{_c.get('reason')}")
        if not _sev:
            return
        # [P2-CAP-LOG-LEVEL · 2026-07-29 · v2] De-dup por CONTENIDO dentro de una RÁFAGA:
        # `aggregate_and_deduct_shopping_list` corre una vez por variante de lista (semanal /
        # quincenal / mensual) sobre el MISMO plan, así que el resumen salía 2-3 veces idéntico y a
        # veces en el mismo segundo (medido en prod: 18 líneas en ~7 min, dos a las 21:28:40).
        # Repetir la misma línea es la versión pequeña del problema que este bloque vino a resolver.
        #
        # El TTL acota el de-dup a la ráfaga que lo motivó. La firma es un global de módulo, así que
        # sin él dos planes DISTINTOS con un set de topes severos idéntico (mismos alimentos, mismos
        # gramos) se silenciarían mutuamente para siempre: improbable, pero el modo de fallo sería
        # "tragarse la señal de otro usuario", que es justo lo contrario de lo que persigue este
        # bloque. Con TTL, lo peor que pasa es perder un duplicado dentro de la misma ráfaga.
        # Rollback: MEALFIT_CAP_SUMMARY_DEDUP_TTL_S=0 → sin de-dup (vuelven las 2-3 repeticiones).
        # [P2-CAP-LOG-LEVEL · 2026-07-30 · v3] La firma va por (ALIMENTO, RAZÓN), no por gramos.
        #
        # ⚠️ Corrección de la v2, medida en producción: la v2 firmaba el texto completo — gramos
        # incluidos — y los gramos son EXACTAMENTE lo que cambia entre variantes de lista. Cada
        # variante escala el mismo tope ('Puerro 415→200g' / 'Puerro 104→50g' / 'Puerro 208→100g'),
        # así que la firma nunca coincidía y el de-dup era **incapaz de disparar por construcción**
        # — peor que no ayudar. Medido: 53 resúmenes en 40 min, 3-4 por plan, cuando yo había
        # afirmado "como máximo uno por plan".
        #
        # Y los resúmenes de un mismo plan son ACUMULATIVOS (el 2º contiene las entradas del 1º más
        # otras), así que además se compara por SUBCONJUNTO: si lo que traigo ya estaba contado en
        # el resumen anterior de esta ráfaga, callo; si aporta un tope de un alimento nuevo, hablo.
        # Así el operador ve el resumen más COMPLETO y no sus tres prefijos.
        # El estado se guarda como SET, no como string unido: la clave ya contiene `|` por dentro
        # ("Puerro|P3-HERB-CAP"), así que unir con `|` y volver a partir por `|` deshacía las claves
        # en trozos y el subconjunto no casaba nunca. Sin separador no hay colisión posible.
        global _LAST_SEVERE_CAPS_SIG, _LAST_SEVERE_CAPS_AT
        _now = _time.time()
        _vigente = (_now - _LAST_SEVERE_CAPS_AT) < _SEVERE_CAPS_DEDUP_TTL_S
        _previas = _LAST_SEVERE_CAPS_SIG if _vigente else set()
        if _vigente and _claves.issubset(_previas):
            return
        _LAST_SEVERE_CAPS_SIG = _claves | _previas
        _LAST_SEVERE_CAPS_AT = _now
        logging.warning(
            f"🧺 [P2-CAP-LOG-LEVEL] {len(_sev)} tope(s) de perecedero recortaron "
            f">{int((1 - _CAP_LOG_SEVERE_RATIO) * 100)}%: "
            f"{'; '.join(_sev[:8])}{' …' if len(_sev) > 8 else ''} — esto habla del MENÚ "
            f"(demasiado perecedero pedido), no del tope.")
    except Exception as _e:
        # [P2-CAP-LOG-LEVEL · 2026-07-29 · v2] NO `pass`. Este mismo helper acaba de demostrar por
        # qué: un `_time` mal referenciado dentro del try convirtió "el resumen falla" en "el
        # resumen no tiene nada severo que decir" — indistinguibles desde fuera, y el test que lo
        # cazó lo hizo por el número equivocado. Un canal de telemetría que se rompe en silencio es
        # el modo de fallo que este bloque entero vino a atacar. Cuesta cero cuando funciona.
        logging.warning(f"⚠️ [P2-CAP-LOG-LEVEL] resumen de topes severos falló: {_e!r}")


def _record_cap_applied(name: str, pre_value: float, post_value: float, reason: str) -> None:
    """[P1-CAPS-COHERENCE-RECONCILE · 2026-05-16] Registra metadata de un cap
    aplicado por el aggregator. Best-effort: excepciones se silencian para
    no romper la cadena del aggregator si la metadata es inválida."""
    try:
        _CAPS_APPLIED_LAST_RUN.append({
            "food": str(name).strip(),
            "food_lower": str(name).strip().lower(),
            "pre_value": float(pre_value),
            "post_value": float(post_value),
            "reason": str(reason),
        })
    except Exception:
        pass


def get_caps_applied_last_run() -> list:
    """Retorna copia de la lista de caps del último run. El coherence guard
    consume esto para filtrar divergencias de magnitud que son legítimas."""
    return list(_CAPS_APPLIED_LAST_RUN)


# [P2-COHERENCE-GUARD-PERF · 2026-05-16] Cache del alias_map construido desde
# `get_master_ingredients()`. Pre-fix el coherence guard reconstruía este map
# en cada call a `_canonicalize_for_coherence`, y `_canonicalize_food_dict_for_coherence`
# llamaba a esta función N+1 veces (una bulk para el set + una per-item para
# deducir el mapping inverso raw→canonical). Para 33 items × 35 recipes el
# guard tardaba 3323ms (umbral 1000ms emitido por `_emit_coherence_guard_metric`).
# Con cache TTL=300s las invocaciones subsecuentes son O(1) lookup + O(N)
# iteración sobre food_names. master_ingredients rara vez cambia en runtime
# (dataset estático del repo) por lo que TTL alto es seguro; el restart natural
# del backend lo invalida.
_COHERENCE_ALIAS_MAP_CACHE: dict | None = None
_COHERENCE_ALIAS_MAP_CACHE_AT: float = 0.0
_COHERENCE_ALIAS_MAP_CACHE_SIZE: int = 0
_COHERENCE_ALIAS_MAP_TTL_S = 300.0


def _get_coherence_alias_map_cached() -> dict:
    """Retorna el alias_map (alias_lower → canonical) construido desde
    `get_master_ingredients()`, cacheado con TTL=300s. Ver bloque de docstring
    arriba. Excepciones devuelven dict vacío (fail-soft: el guard sigue
    funcionando con canonicalización inline pavo/protein/fish).
    """
    global _COHERENCE_ALIAS_MAP_CACHE, _COHERENCE_ALIAS_MAP_CACHE_AT, _COHERENCE_ALIAS_MAP_CACHE_SIZE
    import time as _time_alias
    now = _time_alias.time()
    if (
        _COHERENCE_ALIAS_MAP_CACHE is not None
        and (now - _COHERENCE_ALIAS_MAP_CACHE_AT) < _COHERENCE_ALIAS_MAP_TTL_S
    ):
        return _COHERENCE_ALIAS_MAP_CACHE
    try:
        master_list = get_master_ingredients() or []
    except Exception as e:
        logging.debug(f"[COH-GUARD] master_map fetch falló: {e}")
        master_list = []
    alias_map: dict = {}
    for m in master_list:
        canonical = m.get("name") or ""
        if not canonical:
            continue
        alias_map[canonical.strip().lower()] = canonical
        for alias in (m.get("aliases") or []):
            if alias:
                alias_map[str(alias).strip().lower()] = canonical
    _COHERENCE_ALIAS_MAP_CACHE = alias_map
    _COHERENCE_ALIAS_MAP_CACHE_AT = now
    _COHERENCE_ALIAS_MAP_CACHE_SIZE = len(alias_map)
    return alias_map


def _canonicalize_for_coherence(food_names) -> set:
    """[P1-shop-coh-1 · 2026-05-07] Canonicaliza un set de food names usando
    master_map + reglas inline simples del aggregator (huevo/ñame/miel/ajo).

    [P3-4 · 2026-05-07] Ahora también aplica `canonicalize_pavo` para que
    los productos de pavo sean simétricos entre expected (receta) y
    aggregated (lista). Limitación v1/v2 (pavo como falso positivo)
    cerrada.

    [P1-1 · 2026-05-10] Tras el match con master_map y reglas pavo, aplica
    fallback genérico: strip de modificador trivial trailing
    (`pollo orgánico → pollo`) + singularización es-DO (`manzanas → manzana`).
    Cierra dos modos de falso positivo `cap_swallowed_modifier` documentados:
    plurales y modificadores no listados como aliases.

    Réplica subset de la canonicalización en `aggregate_and_deduct_shopping_list`
    (línea ~2280) para que el guard compare nombres simétricos en ambos lados.
    """
    if not food_names:
        return set()
    # [P2-COHERENCE-GUARD-PERF · 2026-05-16] alias_map cacheado (ver helper
    # `_get_coherence_alias_map_cached`). Pre-fix esta función reconstruía el
    # alias_map en CADA call iterando ~100-200 items del master_list +
    # aliases. Y `_canonicalize_food_dict_for_coherence` la llamaba N+1 veces
    # (una bulk + una per-item para deducir el mapping inverso) → el guard
    # tardaba 3323ms para 33 items × 35 recetas. Con cache TTL=300s, todas
    # las invocaciones subsecuentes son O(N) lookups.
    alias_map = _get_coherence_alias_map_cached()

    out = set()
    for raw_name in food_names:
        if not raw_name:
            continue
        # [P2-AUDIT-2 · 2026-05-10] Strip marca comercial de lácteos ANTES
        # del lookup en master_map. "Leche Induvaca entera" → "Leche entera"
        # → master_map alias → "Leche". Sin esto, master_map no encontraba
        # match (no listamos todas las marcas como aliases — explota la
        # cardinalidad) y el guard reportaba false positive.
        raw_name = _strip_dairy_brand(str(raw_name))
        n_low = str(raw_name).strip().lower()
        canonical = alias_map.get(n_low, str(raw_name).strip())
        # [P2-NEW-8 · 2026-05-11] SSOT: las 4 reglas inline Huevo/Ñame/Miel/Ajo
        # ahora viven en `_consolidate_inline_canon` (sin esto, drift contra
        # el aggregator. Pre-P2-NEW-8 vivían duplicadas aquí y allá).
        _inline_canon = _consolidate_inline_canon(canonical)
        if _inline_canon is not None:
            canonical = _inline_canon
        else:
            # [P3-4 · 2026-05-07] Pavo: aplicar mirror del aggregator. Aplica
            # sobre raw_name (preserva intent del LLM) Y sobre canonical
            # (cubre el caso en que master_map ya canonicalizó). Si alguno
            # produce un canónico, ese gana; si ninguno → keep canonical.
            pavo_from_raw = canonicalize_pavo(raw_name)
            pavo_from_canon = canonicalize_pavo(canonical)
            if pavo_from_raw is not None:
                canonical = pavo_from_raw
            elif pavo_from_canon is not None:
                canonical = pavo_from_canon
            else:
                # [P2-NEW-1 · 2026-05-10] Pollo/cerdo/res: canonicalización
                # unilateral del coherence guard (el aggregator no consolida
                # estos; pero el guard sí necesita simetría para magnitudes).
                # "pechuga de pollo desmenuzada" en receta + "Pollo" en lista
                # → ambos a 'Pollo' → magnitude check trabaja sobre el total.
                protein_from_raw = canonicalize_protein(raw_name)
                protein_from_canon = canonicalize_protein(canonical)
                if protein_from_raw is not None:
                    canonical = protein_from_raw
                elif protein_from_canon is not None:
                    canonical = protein_from_canon
                else:
                    # [P1-AUDIT-2 · 2026-05-10] Pescados/mariscos: mismo patrón
                    # que pollo/cerdo/res pero per-species. "Filete de salmón
                    # guisado" en receta + "Salmón" en lista → ambos a 'Salmón'.
                    # Cierra el silent miss observado en audit (yield_uncovered
                    # NO se disparaba porque presence ya divergía).
                    fish_from_raw = canonicalize_fish_seafood(raw_name)
                    fish_from_canon = canonicalize_fish_seafood(canonical)
                    if fish_from_raw is not None:
                        canonical = fish_from_raw
                    elif fish_from_canon is not None:
                        canonical = fish_from_canon
                    # [P1-NEW-2 · 2026-05-11] 4 canonicalizers nuevos
                    # (huevo, lacteo, grano, legumino) — paralelos al
                    # pattern P2-NEW-1/P1-AUDIT-2. Try cada uno en orden
                    # antes del fallback genérico singularize/strip.
                    # Primer match gana — el orden refleja frecuencia en
                    # planes RD.
                    elif (
                        (huevo := canonicalize_huevo(raw_name)) is not None
                        or (huevo := canonicalize_huevo(canonical)) is not None
                    ):
                        canonical = huevo
                    elif (
                        (lact := canonicalize_lacteo(raw_name)) is not None
                        or (lact := canonicalize_lacteo(canonical)) is not None
                    ):
                        canonical = lact
                    elif (
                        (gr := canonicalize_grano(raw_name)) is not None
                        or (gr := canonicalize_grano(canonical)) is not None
                    ):
                        canonical = gr
                    elif (
                        (lg := canonicalize_legumino(raw_name)) is not None
                        or (lg := canonicalize_legumino(canonical)) is not None
                    ):
                        canonical = lg
                    # [P3-NEW-6 · 2026-05-11] Víveres y musáceas: paralelos a
                    # canonicalize_grano / canonicalize_legumino pero para
                    # tubérculos (yuca/yautía/batata/papa/auyama) y musáceas
                    # (plátano/guineo). Sin estos, "Yuca hervida" + "Yuca con
                    # mojo" se aggregaban como 2 líneas, inflando la lista
                    # de compras. Bilateral con el aggregator (mirror).
                    elif (
                        (viv := canonicalize_viveres(raw_name)) is not None
                        or (viv := canonicalize_viveres(canonical)) is not None
                    ):
                        canonical = viv
                    elif (
                        (mus := canonicalize_musaceae(raw_name)) is not None
                        or (mus := canonicalize_musaceae(canonical)) is not None
                    ):
                        canonical = mus
                    # [P2-NEW-A · 2026-05-11] Frutas tropicales / verduras de
                    # hoja / aceites: tres familias adicionales paralelas a
                    # viveres/musaceae. Sin estas:
                    #   - "Mango verde" + "Mango maduro" → 2 líneas (deberían
                    #     ser 1 línea "Mango").
                    #   - "Lechuga romana" + "Lechuga americana" → 2 líneas
                    #     ("Lechuga" canónica colapsa variedades).
                    #   - "Aceite de oliva extra virgen" + "Aceite oliva" →
                    #     2 líneas con qty fraccionada (Aceite de oliva en
                    #     una sola línea sumada).
                    # Bilateral con el aggregator (mirror).
                    elif (
                        (fr := canonicalize_frutas_tropicales(raw_name)) is not None
                        or (fr := canonicalize_frutas_tropicales(canonical)) is not None
                    ):
                        canonical = fr
                    elif (
                        (vh := canonicalize_verduras_hoja(raw_name)) is not None
                        or (vh := canonicalize_verduras_hoja(canonical)) is not None
                    ):
                        canonical = vh
                    elif (
                        (ac := canonicalize_aceites(raw_name)) is not None
                        or (ac := canonicalize_aceites(canonical)) is not None
                    ):
                        canonical = ac
                    # [P3-NEW-12 · 2026-05-11] 5 canonicalizers nuevos
                    # (cítricos, tomate, cebolla, quesos blancos RD, frutos
                    # secos). Paralelos al patrón P2-NEW-A. Cierran los
                    # últimos 5 buckets `unknown` documentados en P3-OPEN-3.
                    # Sin estos, "limón verde" + "limón persa" → 2 líneas;
                    # "tomate perita" + "tomate criollo" → 2 líneas; etc.
                    # Bilateral con el aggregator (mirror).
                    elif (
                        (cit := canonicalize_citricos(raw_name)) is not None
                        or (cit := canonicalize_citricos(canonical)) is not None
                    ):
                        canonical = cit
                    elif (
                        (tom := canonicalize_tomate(raw_name)) is not None
                        or (tom := canonicalize_tomate(canonical)) is not None
                    ):
                        canonical = tom
                    elif (
                        (ceb := canonicalize_cebolla(raw_name)) is not None
                        or (ceb := canonicalize_cebolla(canonical)) is not None
                    ):
                        canonical = ceb
                    elif (
                        (qb := canonicalize_quesos_blancos_rd(raw_name)) is not None
                        or (qb := canonicalize_quesos_blancos_rd(canonical)) is not None
                    ):
                        canonical = qb
                    elif (
                        (fs := canonicalize_frutos_secos(raw_name)) is not None
                        or (fs := canonicalize_frutos_secos(canonical)) is not None
                    ):
                        canonical = fs
                    # [P1-1 · 2026-05-10] Fallback genérico para los modos de
                    # falso positivo conocidos. Orden: strip modifier → singularizar.
                    # Si el master_map ya entregó un canónico distinto del raw
                    # (n_low != alias_map.get(n_low,...)), respetamos su veredicto
                    # y NO aplicamos fallback (master tiene contexto que la heurística
                    # no tiene). Solo cuando master no aportó (canonical == raw
                    # stripped) ejercitamos el fallback.
                    elif alias_map.get(n_low) is None:
                        stripped = _strip_trailing_modifier_es(canonical)
                        if stripped != canonical:
                            canonical = stripped
                        canonical = _singularize_food_es(canonical)
        out.add(canonical)
    return out


# [P1-COHERENCE-OVERSUPPLY-STAPLES · 2026-07-07] (extiende P1-COHERENCE-PACKAGING-NOISE)
# En el coherence guard, la SOBRE-oferta (lista > receta) es RUIDO de granularidad de
# envase para CASI TODO: granos/staples (arroz/avena/pasta/cebada), condimentos/aceites/
# semillas, y vegetales/aromáticos por unidad entera (ajo/repollo/lechuga) se compran por
# bolsa o cabeza → SIEMPRE hay "de sobra" y la receta es cocinable. La ÚNICA sobre-oferta
# que sigue siendo SEÑAL es la de PROTEÍNA vendida por PESO (carne/pescado/aves): comprar
# 4× el pollo de la receta es un over-buy real de costo (test_pavo_v3). Enlatados/huevos
# (rounding de lata/cartón) y frutas/veg (caps P5/P6 + cap-aware) NO están aquí → su
# sobre-oferta se filtra. Antes solo filtrábamos condimentos → arroz/repollo/ajo/avena
# quedaban como warn ruidoso (67/plan). `\w*` cubre plurales/compuestos ("filetes",
# "pechuga de pollo"). Solo FILTRA sobre-oferta; jamás oculta una FALTA (under-supply).
_COHERENCE_OVERSUPPLY_PROTEIN_KEEP_RE = re.compile(
    r"\b(pollo|pavo|cerdo|res|carne|bistec|chuleta|lomo|molid|costilla|longaniza|"
    r"salchicha|salami|tocineta|jamon|pechuga|muslo|pescado|tilapia|mero|salmon|atun|"
    r"sardina|bacalao|camaron|mariscos|mejillon|calamar|pulpo|cangrejo|langosta|"
    r"filete|higado)\w*"
)


def run_shopping_coherence_guard(plan_result: dict, *, mode_override: str = None, multiplier: float = None) -> list:
    """[P1-shop-coh-1 · 2026-05-07 / P1-C 2026-05-07 v2] Guard recetas↔lista.
    Honra `MEALFIT_SHOPPING_COHERENCE_GUARD` (off|warn|block).

    v2 cubre dos capas:
      A) **Presence/absence** (heredado de v1):
         - food en recetas y ausente de la lista → `cap_swallowed_modifier`.
         - food en la lista y ausente de las recetas → `unknown`.
      B) **Magnitudes** (P1-C, requiere multiplier):
         - escala expected por household multiplier antes de comparar.
         - excluye pavo del lado aggregated para evitar falsos positivos
           (regla fresh-vs-procesado de 50+ líneas no replicada).
         - usa `compare_expected_vs_aggregated` con tolerance leído del knob
           `MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT` (default 0.10) →
           ejercita las hipótesis `yield_uncovered`, `pantry_overdeduct`,
           `unit_mismatch` además de `cap_swallowed_modifier` (qty mitad).

    Items con `is_staple=True` o categoría "Urgente" se filtran del lado
    aggregated en ambas capas (no provienen de recetas, son ruido).

    [P2-PROTEIN-YIELD-CANONICAL · 2026-08-03 · ronda 1] El lado esperado espeja el SELLO
    `protein_yield_applied` que la lista ya lleva (estampado por
    `aggregate_and_deduct_shopping_list` al construirla), NUNCA el knob
    `MEALFIT_PROTEIN_YIELD_ON_CANONICAL` vigente en el momento de re-evaluar. Mismo
    criterio que el sello `trip_window_days` (P1-TRIP-WINDOWED-PERISHABLES): leer el
    knob en vez del sello fabrica divergencias fantasma cuando el guard re-valida un
    plan persistido (cron diario, rebuild) bajo un estado de knob distinto al que
    construyó la lista — en cualquiera de las dos direcciones del A/B (encendido
    después de construir, o rollback después de encender). Ver
    `_protein_yield_seal_applied`.

    Args:
        plan_result: dict con `days` y `aggregated_shopping_list`. Opcional
            `calc_household_multiplier` (cacheado por P1-3).
        mode_override: si se pasa, ignora el env var. Útil para el cron
            (Paso 7) que re-evalúa planes ya persistidos en modo `warn` para
            evitar mutar `_shopping_coherence_block` retroactivamente.
        multiplier: [P1-C] override del household multiplier. Si None, lee de
            `plan_result["calc_household_multiplier"]` con fallback 1.0.
            Pasar `multiplier=1.0` desactiva la simetría de escala (útil para
            tests v1 retro-compatibles).

    Modos:
      off:   no-op.
      warn:  log estructurado de divergencias + Counter por hipótesis.
      block: igual que warn + setea `plan_result["_shopping_coherence_block"]`
             con el subset crítico:
               - foods de receta AUSENTES en lista (presence), Y/O
               - divergencias de magnitud con delta_pct > tolerance que
                 NO sean fantasmas (delta=inf desde lado aggregated).
             [P2-A · 2026-05-07] El flag es CONSUMIDO por `review_plan_node`
             (graph_orchestrator) que rechaza el plan según
             `MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION` (default `reject_minor`).
             Antes de ese fix, mode=block era no-op silencioso (flag persistía
             pero nada lo accionaba). Ver memoria
             `project_p2_a_shopping_coherence_block_enforced`.

    Returns:
        Lista `[{food, side, hypothesis, ...}]`. Items presence/absence tienen
        `magnitude=False`. Items v2 magnitud tienen `magnitude=True` + campos
        `unit, expected_qty, actual_qty, delta_pct`. Vacía si guard `off` o
        sin divergencias.
    """
    # [P2-COHERENCE-GUARD-PERF · 2026-05-15] Wrap timing.
    # ANTES, no había métrica `duration_ms` persistida por call. Un refactor
    # accidental que volviese O(n²) (e.g. doble loop sobre ingredientes
    # canonicalizados) pasaba inadvertido hasta que la latencia del
    # `assemble_plan_node` saltase user-facing. Ahora cada call emite a
    # `pipeline_metrics(node='coherence_guard_validation')` con duration_ms
    # + recipe_count + ingredient_count + divergence_count.
    import time as _time_coh
    _coh_started_at = _time_coh.time()
    _coh_recipe_count = 0
    _coh_ingredient_count = 0
    _coh_divergence_count = 0
    _coh_emit_node = "coherence_guard_validation"

    if mode_override is not None:
        mode = str(mode_override).strip().lower()
        if mode not in ("off", "warn", "block"):
            mode = "warn"
    else:
        mode = _get_coherence_guard_mode()
    if mode == "off":
        # Emit metric even for off-mode (visibilidad: ver cuántas calls llegan
        # con el guard desactivado por knob).
        _emit_coherence_guard_metric(
            duration_ms=int((_time_coh.time() - _coh_started_at) * 1000),
            mode=mode,
            recipe_count=0,
            ingredient_count=0,
            divergence_count=0,
        )
        return []

    # [P1-C] Resolver multiplier: arg explícito > plan_result cacheado > 1.0.
    if multiplier is None:
        try:
            mult = float(plan_result.get("calc_household_multiplier") or 1.0)
        except (TypeError, ValueError):
            mult = 1.0
    else:
        try:
            mult = float(multiplier)
        except (TypeError, ValueError):
            mult = 1.0
    if math.isnan(mult) or math.isinf(mult) or mult <= 0:
        mult = 1.0

    # [P1-COHERENCE-DAY-BASIS · 2026-07-26] El guard comparaba 3 días de recetas contra una
    # lista de 7 días. El agregador PROYECTA a propósito (get_shopping_list_delta:10172):
    #
    #     # Si hay 3 días generados, representan un ciclo rotativo. Promediamos por día
    #     # y proyectamos a 7 días.
    #     base_duration_scale = 7.0 / num_days
    #     effective_multiplier = multiplier * base_duration_scale
    #
    # `expected_sum_from_recipes` NO espejaba ese factor, así que TODO divergía por
    # 7/num_days. Medido sobre 19 planes vivos —los 19 con ≤3 días materializados, porque el
    # guard corre en `assemble_plan_node` ANTES de que los chunks llenen los días 4+— el
    # factor encaja al decimal, no aproximadamente:
    #
    #     Pescado    574.7 / 3 × 7 = 1341.0   ← la lista dice 1341.0
    #     Cangrejo   225.0 / 3 × 7 =  525.0   ← la lista dice  525.0
    #
    # Dos alimentos sin relación con ratio idéntico ⇒ factor estructural, no incoherencia.
    # En modo `block` esto rechazaba casi cualquier plan, y por eso el guard estaba forzado a
    # `warn` en producción. La premisa de `P2-COH-WEEKLY-BASIS` ("la lista semanal ES la misma
    # base que expected") solo es cierta con la semana COMPLETA materializada.
    #
    # Se espeja la MISMA fórmula, no una heurística nueva. Solo cuando se va a comparar contra
    # la lista SEMANAL: la lista activa (quincenal/mensual) tiene otra base y es el caso que
    # P2-COH-WEEKLY-BASIS evita precisamente por eso.
    _basis_scale = 1.0
    _day_basis_applied = False
    if _get_coherence_day_basis_norm_knob() and (plan_result.get("aggregated_shopping_list_weekly")):
        try:
            # [P1-COH-BASIS-SSOT · 2026-08-22] MISMA fuente que el agregador (SSOT
            # `shopping_source_days`). Leer `plan_result["days"]` aquí inflaba el esperado
            # ×7/3 en planes con días archivados: 46 alimentos del plan 2245eb45 con ratio
            # 0.424-0.431. Detalle: docs/shopping_list_cycle_days.md § el espejo a medias.
            _n_days_basis = len(shopping_source_days(plan_result))
        except Exception:
            _n_days_basis = 0
        if _n_days_basis > 0:
            _basis_scale = 7.0 / float(_n_days_basis)
            _day_basis_applied = True
            if abs(_basis_scale - 1.0) > 1e-9:
                logging.info(
                    "[COH-GUARD/DAY-BASIS] días materializados=%d → escalando el lado esperado "
                    "×%.4f para igualar la proyección a 7 días del agregador",
                    _n_days_basis, _basis_scale,
                )

    # [P2-PROTEIN-YIELD-CANONICAL · 2026-08-03 · ronda 1] El espejo sigue al SELLO
    # `protein_yield_applied` que la lista lleva, NUNCA al knob VIGENTE — exactamente el
    # mismo criterio que el espejo del ventaneo unas líneas más abajo (sello
    # `trip_window_days`, comentario "el sello ES la evidencia de cómo se construyó ESTA
    # lista"). Leer el knob aquí en vez del sello fabrica divergencias fantasma en las DOS
    # direcciones del A/B: lista construida con knob OFF + cron re-evaluando con knob ON
    # (medido: 25,9% de divergencia sobre un ítem de 1.435 g), y el rollback simétrico
    # (knob ON→OFF con listas ya sembradas). Ver `_protein_yield_seal_applied`.
    #
    # [P2-GUARD-UNDERSUPPLY-CANONICAL · 2026-08-03 · ronda 1] La variable se llama
    # `_guard_agg_list` y no `_protein_yield_agg_list` porque ya la leen DOS sellos (proteína
    # y nevera) y previsiblemente los que vengan: es "la lista contra la que este guard va a
    # comparar", no la lista de una regla concreta.
    _guard_agg_list = (
        plan_result.get("aggregated_shopping_list_weekly")
        or plan_result.get("aggregated_shopping_list") or []
    )
    _apply_protein_yield = _protein_yield_seal_applied(_guard_agg_list)

    # [P2-GUARD-UNDERSUPPLY-CANONICAL · 2026-08-03] Misma disciplina de sello para la otra
    # pregunta que el clasificador no podía responder: ¿a esta lista se le restó nevera?
    # Sin la respuesta, `_classify_divergence_hypothesis` etiquetaba `pantry_overdeduct`
    # cualquier `0 < act < exp/2` por umbral puro, y `_has_severe_divergence` exime esa
    # hipótesis → el sub-suministro real de una lista CANÓNICA (donde la deducción es
    # imposible por construcción) heredaba la exención para siempre.
    _pantry_seal = _pantry_deduction_seal(_guard_agg_list)
    _pantry_deduction_applied = True if _pantry_seal is None else _pantry_seal
    if _pantry_seal is False:
        # `debug` y no `info`: la lista canónica es el caso NORMAL de este guard (toda
        # construcción con `is_new_plan=True`), así que a nivel info serían 4 líneas por
        # corrida describiendo que todo va como debe. Se loguea la anomalía, no la rutina.
        logging.debug(
            "[COH-GUARD/P2-GUARD-UNDERSUPPLY-CANONICAL] lista CANÓNICA (sello "
            "pantry_deduction_applied=False): el sub-suministro por debajo del 50% de lo "
            "que exigen las recetas se clasifica `magnitude_undersupply`, no "
            "`pantry_overdeduct`."
        )

    try:
        expected_raw = expected_sum_from_recipes(
            plan_result, apply_yield=False, multiplier=mult * _basis_scale,
            apply_protein_yield=_apply_protein_yield,
        )
    except Exception as e:
        logging.warning(f"[COH-GUARD] expected_sum_from_recipes falló: {e}")
        return []

    # [P1-TRIP-WINDOWED-PERISHABLES · 2026-08-02] ESPEJO OBLIGATORIO del ventaneo.
    # Si la lista contra la que vamos a comparar se construyó con la ventana del viaje
    # (perecederos = 7 días activos, estables = periodo completo), el lado ESPERADO tiene
    # que hacer exactamente la misma partición. Sin este espejo el guard reportaría
    # `expected_only` para TODO perecedero que no se cocina esta semana (el pescado de la
    # semana 3) y una divergencia de magnitud ~2× para los que sí — divergencias falsas
    # masivas, y en modo `block` un retry-storm. Es la simétrica de P1-COHERENCE-DAY-BASIS
    # (que espeja `7/num_days`) para la base temporal NUEVA.
    #
    # El disparador es el SELLO `trip_window_days` que la propia lista lleva, no un
    # parámetro del caller: así el espejo sigue a la lista que tiene delante (assemble,
    # rebuild T2, recalc, cron re-validando un plan persistido) en vez de asumir que
    # todas las superficies ventanean a la vez.
    #
    # [ronda 1 · 2026-08-02] Y explícitamente SIN consultar
    # `_trip_windowed_perishables_enabled()`: el sello ES la evidencia de cómo se
    # construyó ESTA lista. Apagar el knob detiene la construcción de listas
    # ventaneadas nuevas, pero las ya persistidas siguen siendo ventaneadas y su
    # esperado debe ventanearse igual. Con la consulta al knob aquí, el rollback
    # fabricaba `Pescado expected_only` (`cap_swallowed_modifier`, severa → warn/block)
    # sobre cada lista sellada viva — el rollback se volvía más peligroso que el fix.
    try:
        _trip_win_len = _aggregated_trip_window_len(
            plan_result.get("aggregated_shopping_list_weekly")
            or plan_result.get("aggregated_shopping_list") or []
        )
        if _trip_win_len:
            expected_raw = _mirror_trip_window_expected(
                plan_result, expected_raw, mult=mult,
                window_len=_trip_win_len, day_basis_applied=_day_basis_applied,
                apply_protein_yield=_apply_protein_yield,
            )
    except Exception as _mirror_exc:
        logging.warning(
            f"[COH-GUARD/P1-TRIP-WINDOWED-PERISHABLES] espejo de la ventana falló "
            f"(se compara contra la base del plan): {type(_mirror_exc).__name__}: {_mirror_exc}"
        )

    # [P3-VERIFIED-INGREDIENTS-ONLY · 2026-06-20] ESPEJO del drop del aggregator:
    # filtra el lado ESPERADO (recetas) a ingredientes verificados con la MISMA
    # `_is_verified_for_shopping`. Sin esto, un ingrediente inventado (laurel) que el
    # aggregator dropeó de la lista seguiría en expected_raw → divergencia
    # `expected_only` → en modo=block fuerza retry. Filtrar expected_raw aquí cubre
    # AMBAS capas (presence en :4797 y magnitude en :4831, que derivan de expected_raw).
    if _verified_ingredients_only_enabled() and isinstance(expected_raw, dict):
        # [P1-VERIFIED-ONLY-OBSERVABILITY · 2026-06-21] El filtro evita un retry-storm
        # (decisión: no bloquear por condimentos raros como laurel/comino), pero ANTES
        # era 100% silencioso: si el LLM desobedeció la instrucción upstream
        # (_get_verified_catalog_instruction) y metió un ingrediente SUSTANTIVO fuera de
        # el catalogo verificado, desaparecía de la lista Y del lado esperado del guard → cero señal →
        # "lista de compras incompleta entregada sin aviso" (el miedo del owner). Ahora
        # capturamos lo filtrado ANTES de descartarlo y emitimos un WARNING grep-able para
        # medir la tasa real de desobediencia (si es alta, hay que ampliar catálogo o
        # forzar retry; si es ~0, el sistema cumple). Tooltip-anchor: P1-VERIFIED-ONLY-OBSERVABILITY.
        # [P1-COHERENCE-MIRROR-KEEP · 2026-08-21] El filtro y su WARN viven ahora en
        # `_filter_expected_to_shopping_survivors`, que pregunta por `_survives_shopping_list` —
        # las TRES ramas del agregador, no sólo la del precio. Aquí estaba el mecanismo real de
        # la costura (a): las filas conservadas-sin-precio salían del lado esperado y quedaban
        # como fantasmas `unknown` en el lado agregado, 1:1 con los ítems sin precio del plan.
        expected_raw = _filter_expected_to_shopping_survivors(expected_raw, emit_blind_warning=True)

    # [P2-COH-WEEKLY-BASIS · 2026-07-04] Base CANÓNICA del guard = lista SEMANAL.
    # `expected_sum_from_recipes` suma los días del plan (~1 semana de recetas) ×
    # household — comparar eso contra la lista ACTIVA de un usuario quincenal/mensual
    # (híbrida: estables ×2/×4 semanas) hacía diverger ~100-300% a TODOS los estables +
    # una fila `inf` por split de unidad → 71 divergencias fantasma en un plan recién
    # renovado (caso vivo 2026-07-04, plan c5d800fd: 38 unknown + 33 unit_mismatch,
    # todas ruido de base). La lista semanal ES la misma base que expected; la activa
    # queda como fallback para fixtures/planes legacy sin la key semanal.
    aggregated_list = (plan_result.get("aggregated_shopping_list_weekly")
                       or plan_result.get("aggregated_shopping_list") or [])
    aggregated_names_raw = set()
    for item in aggregated_list:
        if not isinstance(item, dict):
            continue
        cat = str(item.get("category") or item.get("display_category") or "").lower()
        if "urgente" in cat:
            continue
        if item.get("is_staple") is True:
            continue
        nm = item.get("name") or item.get("display_name")
        if nm:
            aggregated_names_raw.add(str(nm).strip())

    expected_names = _canonicalize_for_coherence(set(expected_raw.keys()))
    aggregated_names = _canonicalize_for_coherence(aggregated_names_raw)

    missing_in_agg = expected_names - aggregated_names
    extra_in_agg = aggregated_names - expected_names

    divergences = []
    for food in sorted(missing_in_agg):
        divergences.append({
            "food": food,
            "side": "expected_only",
            "hypothesis": "cap_swallowed_modifier",
            "magnitude": False,
        })
    for food in sorted(extra_in_agg):
        divergences.append({
            "food": food,
            "side": "aggregated_only",
            "hypothesis": "unknown",
            "magnitude": False,
        })

    # [P1-C] Capa B: magnitudes. Solo se ejercita si tenemos expected y la
    # lista tiene items (early-out evita work inútil sobre planes vacíos).
    magnitude_divs = []
    if expected_raw and aggregated_list:
        try:
            tolerance_pct = _get_coherence_tolerance_pct()
            # [P3-4 · 2026-05-07] exclude_pavo=False ahora que
            # `canonicalize_pavo` (en `_canonicalize_for_coherence`) hace
            # el mirror simétrico de la regla fresh-vs-procesado del
            # aggregator. Antes se excluía pavo de ambos lados para
            # evitar falsos positivos por divergencia de canónico.
            agg_dict = _extract_aggregated_food_dict(aggregated_list, exclude_pavo=False)
            expected_canonical = _canonicalize_food_dict_for_coherence(expected_raw)
            aggregated_canonical = _canonicalize_food_dict_for_coherence(agg_dict)
            # [P1-COHERENCE-GRAM-NORM · 2026-07-26] Ambos lados al MISMO idioma antes de
            # comparar. Sin esto el emparejamiento por (alimento, unidad) fallaba —
            # "0.33 taza de avena" contra "26.4 g de avena" no casan— y el 100% de las
            # divergencias salía `unknown` con expected_qty=0.0. Simétrico a propósito:
            # normalizar un solo lado crearía el sesgo inverso.
            expected_canonical = _normalize_food_dict_to_grams(expected_canonical)
            aggregated_canonical = _normalize_food_dict_to_grams(aggregated_canonical)
            raw_mags = compare_expected_vs_aggregated(
                expected_canonical,
                aggregated_canonical,
                tolerance=tolerance_pct,
                pantry_deduction_applied=_pantry_deduction_applied,
            )
            # Filtrar `cap_swallowed_modifier` con act_qty=0 ya capturados por
            # presence/absence: evita doble-reporte del mismo food. Mantenemos
            # los casos donde act_qty>0 (qty mitad u otra deficiencia parcial).
            #
            # [P1-1 · 2026-05-10] Tolerancia ampliada para líquidos/condimentos.
            # Items que matchean keywords (`aceite`, `vinagre`, etc.) reciben
            # `MEALFIT_COHERENCE_LIQUID_TOLERANCE_PCT` (default 0.50) en lugar
            # de la tolerancia base. Cierra falsos positivos del modo "receta
            # escala lineal pero usuario compra ~constante".
            try:
                liquid_kws = _get_coherence_liquid_keywords()
                liquid_tol = _get_coherence_liquid_tolerance_pct()
            except Exception:
                liquid_kws = set()
                liquid_tol = 0.0
            for d in raw_mags:
                food = d["food"]
                # Caso ya cubierto por capa A: food faltante completo.
                if d["actual_qty"] == 0 and food in missing_in_agg:
                    continue
                # Caso ya cubierto por capa A: fantasma puro.
                if d["expected_qty"] == 0 and food in extra_in_agg:
                    continue
                # [P1-1] Líquidos: si el delta cae dentro de la tolerancia
                # ampliada, no es divergencia accionable. Solo se aplica al
                # caso magnitud-finita (no a fantasmas/missing).
                if (
                    liquid_kws
                    and liquid_tol > tolerance_pct
                    and _is_liquid_food(food, liquid_kws)
                    and d.get("delta_pct") not in (float("inf"), None)
                    and d.get("expected_qty", 0) > 0
                    and float(d["delta_pct"]) <= liquid_tol
                ):
                    continue
                d2 = dict(d)
                d2["side"] = "magnitude"
                d2["magnitude"] = True
                magnitude_divs.append(d2)
        except Exception as e:
            logging.warning(f"[COH-GUARD/v2] magnitudes falló (no aborta): {e}")

    # [P1-CAPS-COHERENCE-RECONCILE · 2026-05-16] Filtrar magnitude divs cuyo
    # food matchea un cap aplicado durante este run del aggregator. Los caps
    # recortan magnitudes intencionalmente por storage realism (cilantro
    # 933g→100g, gandules 2333g→907g, yogurt 5717g→2722g) y el guard NO
    # debe reportarlas como divergencias críticas — son por diseño.
    #
    # Matching canónico: el cap registra el food name pre-canonicalización
    # (e.g. "Cilantro"), pero el guard ya canonicalizó al food del divergence
    # (vía `_canonicalize_for_coherence`). Comparamos canónicos a ambos lados
    # para evitar drift por aliasing del master_map.
    try:
        import os as _os_cap_aware
        _cap_aware_env = _os_cap_aware.environ.get("MEALFIT_COHERENCE_CAP_AWARE", "true").strip().lower()
        _cap_aware_enabled = _cap_aware_env not in ("false", "0", "off", "no")
    except Exception:
        _cap_aware_enabled = True
    if _cap_aware_enabled and magnitude_divs:
        try:
            _caps_applied = get_caps_applied_last_run()
            if _caps_applied:
                _capped_foods_canonical = set()
                _capped_food_raw_names = [c["food"] for c in _caps_applied if c.get("food")]
                if _capped_food_raw_names:
                    _capped_foods_canonical = _canonicalize_for_coherence(_capped_food_raw_names)
                if _capped_foods_canonical:
                    _pre_filter = len(magnitude_divs)
                    magnitude_divs = [
                        d for d in magnitude_divs
                        if d.get("food") not in _capped_foods_canonical
                    ]
                    _filtered = _pre_filter - len(magnitude_divs)
                    if _filtered > 0:
                        logging.info(
                            f"🛒 [COH-GUARD/cap-aware] Filtradas {_filtered} divergencias "
                            f"magnitud por caps intencionales (P1-CAPS-COHERENCE-RECONCILE). "
                            f"Caps aplicados: {[c['reason'] for c in _caps_applied]}"
                        )
        except Exception as e:
            logging.warning(f"[COH-GUARD/cap-aware] filter falló (no aborta): {e}")

    # [P1-COHERENCE-PACKAGING-NOISE · 2026-07-07] (review visual plan 30d: 61-69
    # divergencias warn, presence=0, hipótesis {unknown, unit_mismatch}) Descartar
    # del set de magnitud el RUIDO ESTRUCTURAL de granularidad de envase, que NO es
    # una falta accionable:
    #   (a) unit_mismatch — el alimento está en la lista bajo una unidad de envase
    #       no convertible ("3 lonjas de pan"→745g, "1 sobre" pimienta, funda de
    #       mejillón vs g). Ya se excluye del block (P2-COHERENCE-PACKAGE-UNITS); es
    #       igual de ruidoso en warn.
    #   (b) [P1-COHERENCE-OVERSUPPLY-STAPLES] SOBRE-oferta (actual > expected) de cualquier
    #       alimento que NO sea proteína por peso: la receta pide 20g de semillas / 3 lonjas
    #       de pan / ½ cabeza de repollo pero el mínimo vendible es la funda/paquete/cabeza
    #       entera → la lista TIENE de sobra, la receta es cocinable, no hay divergencia real.
    #       La sobre-oferta de PROTEÍNA por peso (pollo/pescado/pavo 4×) SÍ se preserva (over-buy).
    # Se PRESERVAN: yield_uncovered (banda diagnóstica), pantry_overdeduct,
    # magnitude_undersupply [P2-GUARD-UNDERSUPPLY-CANONICAL] y unknown POR DEBAJO (falta real
    # "qty mitad"), sobre-oferta de PROTEÍNA por peso, y toda la capa presence.
    # Reversible: MEALFIT_COHERENCE_PACKAGING_NOISE_FILTER=false.
    if _knob_env_bool("MEALFIT_COHERENCE_PACKAGING_NOISE_FILTER", True) and magnitude_divs:
        try:
            from constants import strip_accents as _sa_pkg
            def _is_packaging_noise(d):
                if d.get("hypothesis") == "unit_mismatch":
                    return True  # unidad de envase no convertible (lonja/sobre/funda vs g)
                try:
                    _exp_q = float(d.get("expected_qty") or 0)
                    _act_q = float(d.get("actual_qty") or 0)
                except (TypeError, ValueError):
                    return False
                # Solo sobre-oferta (la lista tiene MÁS que la receta).
                if not (_exp_q > 0 and _act_q > _exp_q):
                    return False
                _food_norm = _sa_pkg(str(d.get("food") or "").lower())
                # Sobre-oferta de PROTEÍNA real (por peso) = over-buy → SEÑAL, no ruido.
                if _COHERENCE_OVERSUPPLY_PROTEIN_KEEP_RE.search(_food_norm):
                    return False
                # Cualquier otra sobre-oferta (grano/staple/veg/lácteo/condimento por envase
                # o unidad entera) = granularidad de envase → ruido.
                return True
            _pre_pkg = len(magnitude_divs)
            _pkg_noise = [d for d in magnitude_divs if _is_packaging_noise(d)]
            if _pkg_noise:
                magnitude_divs = [d for d in magnitude_divs if not _is_packaging_noise(d)]
                logging.info(
                    f"🛒 [COH-GUARD/pkg-noise] Filtradas {len(_pkg_noise)} divergencias magnitud "
                    f"estructurales (unit_mismatch + sobre-oferta de envase de no-proteína) de "
                    f"{_pre_pkg} (P1-COHERENCE-PACKAGING-NOISE / P1-COHERENCE-OVERSUPPLY-STAPLES). "
                    f"Restantes accionables: {len(magnitude_divs)}"
                )
        except Exception as e:
            logging.warning(f"[COH-GUARD/pkg-noise] filter falló (no aborta): {e}")

    divergences.extend(magnitude_divs)

    if divergences:
        from collections import Counter
        by_hyp = Counter(d["hypothesis"] for d in divergences)
        sample = "; ".join(f"{d['food']} [{d['side']}]" for d in divergences[:6])
        # [P1-COHERENCE-UNKNOWN-RATIO-TELEMETRY · 2026-07-08] Cuando el bucket 'unknown' domina, loguear la
        # distribución de ratios act/exp para que el operador vea la FORMA (sub-oferta vs over-oferta) y
        # decida con evidencia si añadir una categoría (P3-NEW-5: no categoría sin datos). NO cambia el gate.
        _unknown_ratios = {}
        if _knob_env_bool("MEALFIT_COHERENCE_UNKNOWN_RATIO_TELEMETRY", True) and by_hyp.get("unknown"):
            _unknown_ratios = _bucket_unknown_magnitude_ratios(divergences)
        _ratio_txt = f" unknown_ratios={_unknown_ratios}" if _unknown_ratios else ""
        logging.warning(
            f"🛒 [COH-GUARD/{mode}] {len(divergences)} divergencias "
            f"(presence={len(missing_in_agg)+len(extra_in_agg)}, magnitude={len(magnitude_divs)}, "
            f"multiplier={mult}). Hipótesis: {dict(by_hyp)}.{_ratio_txt} Sample: {sample}"
        )
        if mode == "block":
            critical = []
            # Crítico A: foods de receta ausentes en lista (presence).
            critical.extend(d for d in divergences if d["side"] == "expected_only")
            # Crítico B: divergencias de magnitud con delta finito > tolerance
            # (excluir fantasmas con delta=inf — pueden ser staples no marcados).
            # [P2-COHERENCE-PACKAGE-UNITS · 2026-06-22] Excluir `unit_mismatch` (alimento presente bajo una
            # unidad de envase no convertible: "1 pote" vs "200 g" no es una divergencia de magnitud real →
            # bloquearía + reintentaría en falso). Sigue contado como divergencia warn (telemetría).
            critical.extend(
                d for d in magnitude_divs
                if d.get("delta_pct") != float("inf") and d.get("expected_qty", 0) > 0
                and not d.get("unit_mismatch")
            )
            if critical:
                plan_result["_shopping_coherence_block"] = critical
                logging.error(
                    f"🛒 [COH-GUARD/block] {len(critical)} divergencias críticas "
                    f"(presence_missing + magnitude_delta) → marcado para review."
                )
    else:
        logging.info(
            f"🛒 [COH-GUARD/{mode}] OK: 0 divergencias (presence+magnitude, multiplier={mult})."
        )

    # [P2-COHERENCE-GUARD-PERF · 2026-05-15] Emit duration + cardinality
    # antes del return (cubre todos los paths exit normales del guard).
    _coh_recipe_count = len(expected_raw) if expected_raw else 0
    _coh_ingredient_count = len(aggregated_list) if aggregated_list else 0
    _coh_divergence_count = len(divergences)
    _emit_coherence_guard_metric(
        duration_ms=int((_time_coh.time() - _coh_started_at) * 1000),
        mode=mode,
        recipe_count=_coh_recipe_count,
        ingredient_count=_coh_ingredient_count,
        divergence_count=_coh_divergence_count,
    )

    return divergences


def _emit_coherence_guard_metric(
    *,
    duration_ms: int,
    mode: str,
    recipe_count: int,
    ingredient_count: int,
    divergence_count: int,
) -> None:
    """[P2-COHERENCE-GUARD-PERF · 2026-05-15 · umbral 1000→3000→5000 P3-COH-GUARD-PERF-THRESHOLD 2026-06-22]
    Best-effort INSERT a `pipeline_metrics` con perf del coherence guard. Knob umbral:
    `MEALFIT_COHERENCE_GUARD_SLOW_MS` (default 5000) — log warning si excede para que un refactor
    accidental O(n²) sea detectable sin esperar a tail-latency en user-facing.

    [P3-COH-GUARD-PERF-THRESHOLD · 2026-06-22] Default subido 1000→3000ms y luego 3000→5000ms con datos
    más ricos. La distribución de `pipeline_metrics WHERE node='coherence_guard_validation'` reveló DOS
    poblaciones: (a) guards de plan-pequeño/per-chunk ~0-15ms (p50=1ms, mayoría de las calls), y (b) guards
    de PLAN COMPLETO (47 recetas × 47 ingredientes + ~70 divergencias) que cuestan ~3-3.8s de forma
    CONSISTENTE (medición 9 calls hora 21:00 2026-06-22: p50=3027, p90=3676, max=3844). El costo del guard
    de plan completo es la canonicalización O(recetas×ingredientes) — constante para la carga, NO regresión.
    Con umbral 3000ms el warning disparaba en CADA recálculo de plan completo (baseline ~3.0-3.8s) → ruido.
    5000ms queda por encima del p90 observado (3676) con headroom, así solo avisa regresiones reales (>5s,
    ~1.5× el baseline). El metric numérico SIEMPRE se persiste (telemetría per-plan intacta); esto solo
    modula el WARNING. Optimización pendiente (no regresión): cachear la canonicalización del guard para
    bajar el baseline de plan completo de ~3s. Rollback: MEALFIT_COHERENCE_GUARD_SLOW_MS=3000.
    """
    try:
        import os as _os_coh
        try:
            _slow_threshold_ms = int(_os_coh.environ.get("MEALFIT_COHERENCE_GUARD_SLOW_MS", "5000"))
        except (TypeError, ValueError):
            _slow_threshold_ms = 5000

        if duration_ms > _slow_threshold_ms:
            logging.warning(
                f"[P2-COHERENCE-GUARD-PERF] guard tardó {duration_ms}ms "
                f"(umbral {_slow_threshold_ms}ms). recipes={recipe_count} "
                f"ingredients={ingredient_count} divergences={divergence_count} "
                f"mode={mode}. Posible regresión perf — investigar."
            )

        from db_core import execute_sql_write
        import json as _json_coh
        execute_sql_write(
            """
            INSERT INTO pipeline_metrics
                (user_id, session_id, node, duration_ms, retries,
                 tokens_estimated, confidence, metadata)
            VALUES (NULL, NULL, %s, %s, 0, 0, 0, %s::jsonb)
            """,
            (
                "coherence_guard_validation",
                int(duration_ms),
                _json_coh.dumps({
                    "mode": mode,
                    "recipe_count": int(recipe_count),
                    "ingredient_count": int(ingredient_count),
                    "divergence_count": int(divergence_count),
                }, ensure_ascii=False),
            ),
        )
    except Exception:
        # Silent — el guard NO debe fallar por una métrica de telemetry.
        pass


def run_shopping_coherence_guard_and_append_history(
    plan_result: dict,
    *,
    multiplier: float = None,
    mode_override: str = None,
    attempt: int = 1,
    action_taken: str = None,
    plan_id_hint: str = None,
    block_severe_only: bool = False,
) -> tuple:
    """[P1-NEXT-2 · 2026-05-11] SSOT que invoca `run_shopping_coherence_guard`
    Y appendea entry a `plan_result["_shopping_coherence_block_history"]`
    (cap configurable vía `MEALFIT_COHERENCE_BLOCK_HISTORY_CAP`).

    Cierra el gap detectado en el audit 2026-05-11:
        El guard solo se invocaba en `assemble_plan_node` (LangGraph
        full-pipeline, planes ≤7d). Los siguientes surfaces construían
        `aggregated_shopping_list*` sin invocar el guard:
          - `_chunk_worker` T2 (cron_tasks.py, multi-week plans).
          - `/recalculate-shopping-list` (routers/plans.py, recalc cliente
            tras Pantry mutations).
          - `tools.modify_single_meal` (agent tool).
        Resultado: planes multi-week + recalcs podían shipearse con
        divergencias recetas↔lista sin retry ni telemetría — solo
        capturados (post-hoc, sin mutar) por el cron diario 04:00 UTC
        `_shopping_coherence_alert_job` en mode=warn.

    El helper centraliza el bloque que vivía inline en
    `assemble_plan_node` (graph_orchestrator.py:6948-7016): invocar guard
    → si divergencias → construir entry con hipótesis-counter + block_set
    + attempt → appendear con cap. Idempotente respecto al estado: si la
    invocación del guard explota o no encuentra divergencias, no muta
    `plan_result` más allá de lo que ya hace `run_shopping_coherence_guard`
    (que puede setear `_shopping_coherence_block` en mode=block).

    Args:
        plan_result: dict con `days` y `aggregated_shopping_list*`. Debe
            contener `calc_household_multiplier` o pasarse explícito.
        multiplier: override del household multiplier. Si None, lee
            `plan_result["calc_household_multiplier"]`.
        mode_override: 'off' | 'warn' | 'block'. Si None, lee env var
            `MEALFIT_SHOPPING_COHERENCE_GUARD` (default 'block' post-P1-NEW-1).
        attempt: contador de attempt LangGraph (para telemetría).
            Surfaces fuera del pipeline (cron, recalc, agent) pasan 1.
        action_taken: si el caller sabe qué acción se va a tomar (e.g.,
            `"warn_only_chunked_plan"` para T2 / recalc / agent que NO
            retry), lo persiste directo. Si None, se usa el placeholder
            P2-2 (`"not_applicable"` cuando block_set=False, None cuando
            block_set=True para que review_plan_node lo hidrate).
        plan_id_hint: opcional, para el log de truncamiento.

    Returns:
        Tupla `(divergences, block_set)`:
          - `divergences`: lista de divergencias retornadas por el guard.
          - `block_set`: True si el guard seteó `_shopping_coherence_block`
            (mode=block + critical present). El caller decide si abortar
            la persistencia, re-encolar, devolver 400, etc.

    Tooltip-anchor: P1-NEXT-2-HELPER-START | test_p1_next_2_guard_at_persist_sites
    """
    try:
        divergences = run_shopping_coherence_guard(
            plan_result,
            mode_override=mode_override,
            multiplier=multiplier,
        ) or []
    except Exception as e:
        logging.warning(f"[COH-GUARD/HELPER] excepción en guard (no aborta): {e}")
        return [], False

    block_set = bool(plan_result.get("_shopping_coherence_block"))

    # [P2-COHERENCE-1 · 2026-05-11] Escalación selectiva warn → block.
    # Cuando el caller pasa `block_severe_only=True` (típicamente el
    # `_chunk_worker T2` que ya tiene su propio retry loop con backoff),
    # promovemos divergencias críticas (cap_swallowed_modifier o magnitudes
    # >50%) a block para forzar retry. Se respeta el knob
    # `MEALFIT_COHERENCE_T2_BLOCK_SEVERE_ONLY` (default True) como kill
    # switch — flip a False sin redeploy revierte al comportamiento warn-only.
    #
    # Solo se escala cuando mode efectivo es "warn" (no machacamos un
    # block que ya viene del guard original; tampoco escalamos si el
    # caller declaró mode_override="off").
    if (
        block_severe_only
        and not block_set
        and divergences
        and _get_coherence_t2_block_severe_only_knob()
    ):
        try:
            effective_mode = (
                str(mode_override).strip().lower() if mode_override is not None
                else _get_coherence_guard_mode()
            )
        except Exception:
            effective_mode = "warn"
        if effective_mode == "warn" and _has_severe_divergence(divergences):
            plan_result["_shopping_coherence_block"] = True
            block_set = True
            logging.warning(
                f"[COH-GUARD/HELPER/P2-COH-1] block_severe_only escaló warn→block "
                f"(plan_id_hint={plan_id_hint!r}, divergences={len(divergences)})."
            )

    if divergences:
        try:
            from datetime import datetime as _dt, timezone as _tz
            from collections import Counter as _Counter

            prior_history = plan_result.get("_shopping_coherence_block_history") or []
            if not isinstance(prior_history, list):
                prior_history = []

            try:
                attempt_n = int(attempt)
            except (TypeError, ValueError):
                attempt_n = 1

            hyp_counter = _Counter(
                str(d.get("hypothesis") or "unknown") for d in divergences
            )

            if action_taken is not None:
                effective_action = str(action_taken)
            else:
                # Mismo placeholder P2-2 que assemble_plan_node usa:
                # - block_set=True → None (review_plan_node lo hidrata)
                # - block_set=False → "not_applicable" (no entrará al branch)
                effective_action = None if block_set else "not_applicable"

            entry = {
                "ts": _dt.now(_tz.utc).isoformat(),
                "attempt": attempt_n,
                "divergence_count": len(divergences),
                "presence_count": sum(
                    1 for d in divergences if not d.get("magnitude")
                ),
                "magnitude_count": sum(
                    1 for d in divergences if d.get("magnitude")
                ),
                "hypotheses": dict(hyp_counter),
                "block_set": block_set,
                "action_taken": effective_action,
            }
            # [P1-COHERENCE-UNKNOWN-RATIO-TELEMETRY · 2026-07-08] Persistir la distribución de ratios del
            # bucket 'unknown' para queries forenses (¿los unknown son sub-oferta 0.5-0.9 o over-oferta?).
            if _knob_env_bool("MEALFIT_COHERENCE_UNKNOWN_RATIO_TELEMETRY", True) and hyp_counter.get("unknown"):
                _ur = _bucket_unknown_magnitude_ratios(divergences)
                if _ur:
                    entry["unknown_ratios"] = _ur

            # Lazy import para evitar ciclo: graph_orchestrator ya importa
            # de shopping_calculator (módulo cargado primero), así que un
            # import top-level acá rompe el orden. Lazy resuelve runtime.
            try:
                from graph_orchestrator import _apply_coherence_history_cap as _cap_helper
                new_history = _cap_helper(
                    prior_history,
                    entry,
                    plan_id_hint=plan_id_hint or plan_result.get("id") or plan_result.get("plan_id"),
                )
            except ImportError:
                # Fallback inline si el helper se mueve/borra: cap=20 por
                # default coincide con `_COHERENCE_BLOCK_HISTORY_CAP_DEFAULT`.
                new_history = list(prior_history) + [entry]
                if len(new_history) > 20:
                    new_history = new_history[-20:]

            plan_result["_shopping_coherence_block_history"] = new_history
        except Exception as _hist_e:
            logging.debug(
                f"[COH-GUARD/HELPER/HISTORY] no-op (telemetría): {_hist_e}"
            )

    return divergences, block_set


# [P3-AGG-CLEAN-LEADING-PUNCT · 2026-05-23] Caso real verificado log
# 2026-05-23 00:33-00:35: el aggregator emitió `/pedazos de queso` con
# `/` corrupto. El LLM ve ese item en la pantry list y trata de usarlo,
# pero el pantry guard busca exact-match (`queso`) y no matchea
# `/pedazos de queso` → unauthorized → retry. 3 retries seguidos
# fallaron por este mismo modo → 422 (gracias al fix P3-SWAP-LLM-RETRIES-422).
#
# Cleanup defensivo: strip leading punctuation/bullets/símbolos al
# inicio del name extraído por `_parse_quantity`. Esto cubre el caso
# verificado + futuras corrupciones similares (caracteres como `-`,
# `*`, `•`, `·`, `▪` que el LLM puede emitir como list-item markers).
#
# Emite log warning cuando aplica para visibilidad operacional —
# permite detectar upstream bugs sin romper el flujo runtime.
_LEADING_PUNCT_RE = re.compile(r"^[\s/\-\*•·▪▫◦‣⁃▸◾◽■□]+")


def _clean_leading_punct_from_name(name: str) -> str:
    """Strip leading punctuation/bullets de un ingredient name.

    Idempotente: ``"queso"`` → ``"queso"`` (sin cambios).
    Limpia: ``"/pedazos de queso"`` → ``"pedazos de queso"``,
            ``"- arroz"`` → ``"arroz"``,
            ``"• cebolla"`` → ``"cebolla"``.
    """
    if not isinstance(name, str) or not name:
        return name
    cleaned = _LEADING_PUNCT_RE.sub("", name)
    if cleaned != name:
        logging.warning(
            f"[P3-AGG-CLEAN-LEADING-PUNCT] Name normalizado: {name!r} → "
            f"{cleaned!r}. Upstream emitió punctuation/bullet inicial."
        )
    return cleaned


# [P3-AGG-PRESENTATION-MODIFIERS · 2026-05-23] Caso real verificado log
# 2026-05-23 00:45-00:47: tras limpiar el `/` corrupto, el LLM seguía
# fallando con `"Pedazos de queso"` (sin slash) porque ese name NO está
# en master_ingredients (canónico es "queso blanco" o "queso de freír"),
# Y Vector Search caía con 429 RESOURCE_EXHAUSTED → fallback regex
# exact-match → rechazo del pantry guard.
#
# Strip de presentation modifiers SEGUROS (no son aliases canónicos en
# PROTEIN_SYNONYMS / CARB_SYNONYMS / VEGGIE_FAT_SYNONYMS):
#
#   pedazos/pedazo, trozos/trozo, rebanadas/rebanada, rodajas/rodaja,
#   porciones/porción, tajadas/tajada, cubos/cubo, tiras/tira,
#   dados/dado, lonjas/lonja.
#
# NO incluimos "filete de" / "lomo de" / "carne molida de" porque ESOS
# SÍ son aliases canónicos en PROTEIN_SYNONYMS — stripearlos rompería
# la canonicalización legítima.
_PRESENTATION_MODIFIER_PREFIXES_RE = re.compile(
    r"^(pedazos?|trozos?|rebanadas?|rodajas?|porci(?:ón|on|ones)|"
    r"tajadas?|cubos?|tiras?|dados?|lonjas?)\s+de\s+",
    re.IGNORECASE,
)


def _strip_presentation_modifier_prefix(name: str) -> str:
    """Strip prefijos de presentación tipo "pedazos de X" → "X".

    Mantiene names canónicos que CONTIENEN "de" como parte legítima del
    canónico (ej. "queso de freír", "filete de pollo") porque su prefijo
    no está en la lista controlada.

    Idempotente: ``"queso"`` → ``"queso"`` (sin cambios).
    Limpia: ``"pedazos de queso"`` → ``"queso"``,
            ``"Rebanadas de pan"`` → ``"pan"``,
            ``"trozos de pollo"`` → ``"pollo"``.
    """
    if not isinstance(name, str) or not name:
        return name
    cleaned = _PRESENTATION_MODIFIER_PREFIXES_RE.sub("", name, count=1)
    if cleaned != name:
        logging.warning(
            f"[P3-AGG-PRESENTATION-MODIFIERS] Name normalizado: {name!r} → "
            f"{cleaned!r}. Upstream emitió modifier de presentación "
            f"(pedazos/trozos/etc) como parte del nombre canónico."
        )
    return cleaned


def _cost_from_market(market_obj, master_item, price_per_lb, price_per_unit):
    """[P3-PRICE-MARKET-COVERAGE · 2026-06-20] Costea sobre el DISPLAY real — lo que el
    usuario COMPRA (el paquete/cartón/Ud que apply_smart_market_units ya redondeó), no
    sobre el peso CRUDO de la receta. Cierra el desajuste donde staples por-peso
    (arroz/habichuelas/nueces) sub-costeaban (cobraban los gramos de la receta cuando el
    display dice '1 paquete (2 lb)') y el huevo sobre-costeaba (cobraba cartón completo
    por 'medio cartón'). Reemplaza al fallback P3-PRICE-UNIT-COVERAGE (solo-si-0).
    Devuelve el costo RD$ de la cantidad mostrada. Tooltip-anchor: P3-PRICE-MARKET-COVERAGE."""
    try:
        mqty = float(market_obj.get("market_qty") or 0)
    except (TypeError, ValueError):
        mqty = 0.0
    if mqty <= 0:
        return 0.0

    # (0) [P1-PKG-DURATION-PRICING · 2026-06-22] Precio por TAMAÑO de envase real. Si
    # apply_smart_market_units eligió un market_package (tamaño+precio de la tabla
    # verificada), costear con count × precio_del_paquete — descuento por volumen real,
    # NO price_per_lb plano. Usa market_qty_numeric (float fiel) cuando exista; cualquier
    # bump posterior (MARKET_MINIMUMS) se refleja porque el precio es por-paquete.
    _pkg_price = market_obj.get("market_pkg_price_rd")
    if _pkg_price is not None:
        try:
            _pp = float(_pkg_price)
            _mq = float(market_obj.get("market_qty_numeric") or mqty or 0)
            if _pp >= 0 and _mq > 0:
                return _mq * _pp
        except (TypeError, ValueError):
            pass

    munit = str(market_obj.get("market_unit") or "").lower().strip()
    _mi = master_item or {}
    try:
        container_g = float(_mi.get("container_weight_g") or 0)
    except (TypeError, ValueError):
        container_g = 0.0
    try:
        density_g = float(_mi.get("density_g_per_unit") or 0)
    except (TypeError, ValueError):
        density_g = 0.0

    # (D) UNIDADES EMPAQUETADAS — 'cartón (N uds.)' (huevos) o 'paquete (N uds.)' (ajo 4-pack).
    # El pre-process consolida en buckets: huevos vía `_choose_egg_carton`, ajo vía P1-AJO-4PACK.
    # [P1-EGG-CARTON-SIZES + P1-AJO-4PACK · 2026-06-22] Costo con el PRECIO REAL del paquete
    # elegido desde market_packages (por `units` == N): huevos cartón 20=RD$200/30=RD$295; ajo
    # paquete 4=RD$60. El precio NO escala linealmente, así que derivar de price_per_unit sería
    # incorrecto. Genérico: cualquier '(N uds.)' busca su precio en market_packages.units (solo
    # huevos y ajo producen ese sufijo). Fallback legacy (price_per_unit por cartón de 30) SOLO
    # para cartón de huevos ('cart'). Tooltip-anchor: P1-EGG-CARTON-SIZES | P1-AJO-4PACK.
    _units_m = re.search(r'\((\d+)\s*uds?\.?\)', munit)
    if _units_m:
        _pack_n = int(_units_m.group(1))
        _pkgs = _mi.get("market_packages")
        if isinstance(_pkgs, list):
            for _p in _pkgs:
                try:
                    if isinstance(_p, dict) and int(float(_p.get("units"))) == _pack_n and _p.get("price") is not None:
                        return mqty * float(_p["price"])
                except (TypeError, ValueError):
                    continue
        if 'cart' in munit and price_per_unit > 0:
            return mqty * float(_pack_n) * (price_per_unit / 30.0)

    # (A) Display en LIBRAS: market_qty ya está en libras → price_per_lb directo.
    if munit in ("lb", "lbs"):
        if price_per_lb > 0:
            return mqty * price_per_lb
        if price_per_unit > 0 and container_g > 0:
            return max(1, math.ceil(mqty * 453.592 / container_g)) * price_per_unit
        return 0.0

    # (C) Display por UNIDAD NATURAL (Ud./Cabeza: lechosa, melón, plátano enteros).
    if munit in ("ud.", "uds.", "ud", "uds", "unidad", "unidades", "cabeza", "cabezas"):
        if price_per_unit > 0:
            return mqty * price_per_unit
        if price_per_lb > 0 and density_g > 0:
            return mqty * density_g / 453.592 * price_per_lb
        return 0.0

    # Sin unidad costeable.
    if munit in ("al gusto", ""):
        return 0.0

    # (B) Display por ENVASE NOMBRADO (paquete/pote/botella/sobre/lata/funda/frasco/mazo).
    # market_qty = nº de ESE envase. price_per_unit es por envase; si solo hay price_per_lb
    # + peso del envase, convertir envases→libras (1 paquete arroz 907g → 2 lb × price_per_lb).
    if price_per_unit > 0:
        return mqty * price_per_unit
    if price_per_lb > 0 and container_g > 0:
        return mqty * container_g / 453.592 * price_per_lb
    return 0.0


def aggregate_and_deduct_shopping_list(plan_ingredients: list[str], consumed_ingredients: list[str] = None, categorize: bool = False, structured: bool = False, multiplier: float = 1.0, brand_prefs: dict | None = None, brand_defaults: dict | None = None, num_days: int | None = None, cycle_days: int | None = None, text_demand_g_map: dict | None = None, apply_protein_yield: bool = False):
    # [P2-PROTEIN-YIELD-CANONICAL · 2026-08-03] `apply_protein_yield`: el caller
    # (`get_shopping_list_delta`) lo activa SOLO cuando `is_new_plan=True` (lista
    # CANÓNICA, sin lado inventario) Y el knob `MEALFIT_PROTEIN_YIELD_ON_CANONICAL`
    # está ON. Reabre la regla #2 de `_calculate_yield_multiplier` (proteínas
    # cocidas → 1.35× crudo) SOLO en el loop de `plan_ingredients` de abajo — el
    # loop de `consumed_ingredients` (Nevera/consumido real) NUNCA la recibe,
    # incluso si un caller la pasara por error: en modo canónico ese loop está
    # vacío por construcción (ver `get_shopping_list_delta`), y aplicar yield a
    # inventario/consumido real reintroduciría la asimetría P1-2 que el
    # aggregator existe para evitar. Default `False` → comportamiento previo
    # byte-idéntico.
    # [P1-SKU-COVER-HONESTY-R1 · 2026-08-02] `cycle_days` (NO confundir con `num_days`, que es
    # días GENERADOS del plan/chunk para `base_duration_scale`): días que representa la
    # necesidad total ya escalada (7/15/30, según `duration` weekly/biweekly/monthly) — viaja a
    # `apply_smart_market_units` sólo para el copy de la nota "alcanza ~N de M días". Opcional;
    # `None` conserva el default 7 de `apply_smart_market_units` (comportamiento previo, correcto
    # para listas semanales). Callers duration-aware (`get_shopping_list_delta` y sus 15+
    # call-sites en cron_tasks.py/routers/plans.py/tools.py) NO pasan este valor todavía — ver
    # report de P1-SKU-COVER-HONESTY, ronda 1, sección de seguimiento.
    _cycle_days_for_note = int(cycle_days) if cycle_days else 7
    # [P1-CAPS-COHERENCE-RECONCILE · 2026-05-16] Reset del tracker de caps al
    # inicio de cada run del aggregator. Los caps que se apliquen durante
    # este run (P3-HERB-CAP, P5-VEG-CAP, P6-LEGUMES-DRY-CAP, P6-EGGS-AGGREGATE-CAP,
    # P6-LACTEOS-PERISHABLE-CAP, P6-SPICE-CAP) se acumulan en `_CAPS_APPLIED_LAST_RUN`
    # via `_record_cap_applied`. El coherence guard consulta esa lista para
    # ignorar divergencias de magnitud que corresponden a un cap intencional
    # (storage realism), no a un bug de generación del LLM.
    reset_caps_applied_last_run()
    aggregated = defaultdict(lambda: defaultdict(float))

    if consumed_ingredients is None:
        consumed_ingredients = []

    # [P1-7] Guard contra `multiplier` patológico (NaN/Infinity/cero/negativo).
    # Causas reales observables:
    #   - `householdSize=0` por perfil corrupto → caller pasa `1.0 * 0 = 0`
    #     → todo plan_ingredients se anula (lista vacía falsa).
    #   - `num_days=0` en plan vacío persistido a medias → div-zero al
    #     calcular `base_duration_scale = 7/num_days` (mitigado por
    #     `num_days = max(1, ...)` en `get_shopping_list_delta`, pero
    #     callers terceros pueden pasar effective_multiplier directo).
    #   - Float overflow en multiplicaciones encadenadas → `inf` que
    #     produce qty=`inf` en `aggregated` → cualquier cálculo posterior
    #     (clampear, redondear, formatear) revienta o produce strings
    #     "inf"/"nan" en el shopping list.
    # Clampeamos a `[0.01, 50.0]`:
    #   - Mín 0.01 evita anular el plan completo si llegó multiplier=0;
    #     el valor real de la lista es proporcional (1% del plan), pero
    #     el sistema sigue produciendo una lista renderizable y SRE
    #     detecta el log warning para investigar.
    #   - Max 50.0 cubre el peor caso legítimo (12 personas × 4 ciclos
    #     mensuales × 1 = 48); cualquier valor mayor es bug del caller.
    try:
        _mult = float(multiplier)
    except (TypeError, ValueError):
        _mult = 1.0
    if math.isnan(_mult) or math.isinf(_mult) or _mult <= 0:
        logging.warning(
            f"[P1-7/MULTIPLIER] multiplier={multiplier!r} inválido "
            f"(NaN/Inf/<=0). Clampeando a 1.0 para preservar lista renderizable."
        )
        _mult = 1.0
    elif _mult > 50.0:
        logging.warning(
            f"[P1-7/MULTIPLIER] multiplier={_mult} excede cap 50.0. "
            f"Clampeando a 50.0; bug probable en el caller."
        )
        _mult = 50.0
    multiplier = _mult
    
    # [P1-2] Convención de simetría plan↔inventario:
    #
    # El aggregator opera en PESO LITERAL (la cantidad textual descrita por
    # LLM/usuario) sin convertir cocido→crudo vía `_calculate_yield_multiplier`.
    # ANTES, `_parse_quantity` aplicaba yield 1.35× a cualquier match de
    # /\b(cocido|asado|hervido)\b\s+(pollo|carne|...)/  para convertir el
    # peso final descrito a peso crudo necesario. PERO esta conversión solo
    # disparaba cuando el TEXTO contenía el adjetivo:
    #   - plan_ingredients del LLM frecuentemente: "1 lb pollo cocido" → 1.35 lb
    #   - physical_inventory tipeado por user: "5 lb pollo" → 5.0 lb
    # La asimetría textual sesgaba el delta hacia OVER-BUYING (plan inflado a
    # peso crudo, inventario en peso literal sin compensación).
    #
    # AHORA ambos lados llaman `_parse_quantity` con `apply_yield_multiplier=False`
    # → todos los textos se tratan en peso literal y son comparables. El
    # multiplier por ciclo (semanal/quincenal/mensual) sigue aplicándose solo
    # al plan (correcto: consumed/inventario son cantidades absolutas reales).
    #
    # [P2-PDF-1] EXCEPCIÓN: legumbres/granos (`apply_legumbres_yield_only=True`)
    # mantienen su yield 0.35× (cocido→seco) porque su SKU comercial es SECO.
    # Sin esta excepción, "200g habichuelas cocidas" se aggregaba como peso
    # seco → producía 15 paquetes (1 lb c/u) cuando el usuario realmente
    # necesita ~5 lbs secas. La asimetría plan↔inventario que P1-2 cerró
    # NO se reabre: las proteínas cocidas (regla #2) siguen sin yield, y el
    # inventario de habichuelas se almacena con name canónico "Habichuelas
    # rojas" SIN "cocidas" → yield=1.0 → comparado simétricamente vs el
    # plan ya convertido a peso seco.
    # [P2-NEW-11 · 2026-05-11] CONTRATO DE ASIMETRÍA `multiplier` (NO TOCAR
    # sin leer este bloque entero):
    #
    #   plan_ingredients:    qty * multiplier  (escalado)
    #   consumed_ingredients: qty            (sin escalado)
    #
    # Esta asimetría es SEMÁNTICAMENTE CORRECTA, no un bug:
    #
    #   - `plan_ingredients` viene del plan generado por el LLM en
    #     PORCIONES BASE (recetas para 1 persona/comida). El `multiplier`
    #     (`calc_household_multiplier`) infla a la realidad familiar
    #     (3 personas × 7 días = 21 porciones por receta original).
    #
    #   - `consumed_ingredients` viene de `user_inventory` (pantry físico)
    #     o `recipe_consumed` (consumo registrado). YA son CANTIDADES
    #     ABSOLUTAS REALES — el LLM no escaló nada aquí.
    #
    # Ejemplo concreto:
    #   - Plan dice "100g arroz/porción", multiplier=21 → necesitamos 2100g.
    #   - Pantry tiene "500g arroz" físicos.
    #   - Lista correcta = 2100 - 500 = 1600g.
    #   - Si por error aplicáramos `* multiplier` al consumed:
    #     2100 - (500*21) = 2100 - 10500 = -8400 → "tienes excedente",
    #     no agregar a lista. RESULTADO: el usuario nunca compra arroz.
    #
    # Si un futuro refactor cambia el contrato de pantry (ej. almacenar
    # qty_per_person en lugar de cantidad real), AMBOS lados deben
    # migrar simultáneamente. El test parser-based
    # `test_p2_new_11_aggregate_multiplier_asymmetry_contract` ancla
    # esta decisión: detecta si alguien añade `* multiplier` al consumed
    # loop sin documentar la migración.
    plan_names = set()
    for item in plan_ingredients:
        if not item or len(item) < 3: continue
        qty, unit, name = _parse_quantity(
            item, apply_yield_multiplier=False, apply_legumbres_yield_only=True,
            apply_protein_yield=apply_protein_yield,
        )
        if not name: continue
        # [P3-AGG-CLEAN-LEADING-PUNCT · 2026-05-23] Strip bullets/punct al
        # inicio del name; cierra modo de fallo donde el LLM emite
        # "/pedazos de queso" y el pantry guard nunca matchea.
        name = _clean_leading_punct_from_name(name)
        # [P3-AGG-PRESENTATION-MODIFIERS · 2026-05-23] Strip prefijos de
        # presentación ("pedazos de queso" → "queso"). Aplicado DESPUÉS
        # del punct cleanup para que "/pedazos de queso" → "pedazos de
        # queso" → "queso" en cascada.
        name = _strip_presentation_modifier_prefix(name)
        if not name: continue
        if name.lower() in ["ola", "olas"]: name = "Cebolla"
        aggregated[name][unit] += float(qty) * float(multiplier)  # P2-NEW-11: escalado intencional
        plan_names.add(name)

    logging.info(f"🛒 [AGGREGATE] {len(plan_ingredients)} raw items → {len(plan_names)} unique names: {sorted(plan_names)[:30]}...")

    # [P2-SEASONING-RESTOCK-CLEAR · 2026-06-29] Set de nombres (normalizados) que el usuario YA tiene en su
    # Nevera (consumed/inventario). Lo consume el SEASONING-CATALOG-KEEP más abajo para NO re-listar un
    # condimento verificado que la Nevera ya cubre: el plan suele emitir el condimento de forma NOMINAL
    # ("al gusto"/"pizca", sin peso), así que la deducción por peso no lo resta y el seasoning-keep lo
    # re-inyectaba como "1 empaque" aunque ya esté comprado (caso Vainilla tras restock; clase
    # P3-RESTOCK-LECHE-UNIT — asimetría de unidad lista↔inventario). tooltip-anchor: P2-SEASONING-RESTOCK-CLEAR
    _consumed_name_set = set()
    # [P2-GUARD-UNDERSUPPLY-CANONICAL · 2026-08-03] ¿Esta corrida restó ALGO de verdad?
    # `consumed_ingredients` no vacío no basta: una Nevera de puros condimentos «al gusto»
    # parsea a qty=0 y no mueve una sola cantidad. El sello que el guard leerá debe declarar
    # la deducción EFECTIVA, no la intención — con qty=0 la lista sigue siendo canónica y un
    # sub-suministro sobre ella sigue siendo real.
    _pantry_deduction_effective = False
    for item in consumed_ingredients:
        if not item or len(item) < 3: continue
        # [P2-PDF-1] Mismo yield para consumed: si el plato consumido fue
        # "200g habichuelas cocidas", la deducción del inventario debe ser
        # 70g secas (mismo SKU físico que se sumó al plan).
        qty, unit, name = _parse_quantity(item, apply_yield_multiplier=False, apply_legumbres_yield_only=True)
        if not name: continue
        # [P3-AGG-CLEAN-LEADING-PUNCT · 2026-05-23] Mismo cleanup que el
        # plan loop arriba — la asimetría plan/consumed (P2-NEW-11) NO
        # se rompe: solo limpiamos punctuation, no escalamos.
        name = _clean_leading_punct_from_name(name)
        # [P3-AGG-PRESENTATION-MODIFIERS · 2026-05-23] Mismo strip de
        # modifiers que el plan loop — la simetría plan↔consumed
        # requiere identical normalization para que el delta funcione.
        name = _strip_presentation_modifier_prefix(name)
        if not name: continue
        if name.lower() in ["ola", "olas"]: name = "Cebolla"
        # [P2-SEASONING-RESTOCK-CLEAR] registra el nombre normalizado de lo que el usuario ya tiene.
        try:
            _consumed_name_set.add(normalize_name(name))
        except Exception:
            _consumed_name_set.add(str(name).strip().lower())
        aggregated[name][unit] -= float(qty)  # P2-NEW-11: SIN multiplier (ver contrato arriba)
        if float(qty) > 0:
            _pantry_deduction_effective = True  # [P2-GUARD-UNDERSUPPLY-CANONICAL]

    # --- RESOLUCIÓN DE FRICCIÓN DE UNIDADES (Híbridas) ---
    # [P1-VEG-BACKFILL-HONESTY · 2026-08-03] `master_map` + la resolución de nombre canónico se
    # extrajeron a `_build_shopping_master_map()`/`canonicalize_shopping_food_name()` (SSOT
    # compartido con `get_shopping_list_delta::text_demand_g_map` — ver docstring de esa función
    # para el bug que motivó la extracción: plurales rompían el emparejamiento en silencio).
    master_map = _build_shopping_master_map()

    # ── Re-agrupación por Nombre Canónico ──
    # Si el LLM devolvió "Huevo", "Huevos" y "Huevos enteros", el agregador original
    # los tiene como 3 llaves. Aquí los fusionamos en la llave canónica oficial ("Huevos")
    # para que su volumen se sume correctamente antes de calcular empaques comerciales.
    canonical_aggregated = defaultdict(lambda: defaultdict(float))
    for name, units in aggregated.items():
        canonical_name = canonicalize_shopping_food_name(name, master_map)
        for u, q in units.items():
            canonical_aggregated[canonical_name][u] += q

    # ── Post-proceso: Fusionar variantes plural/singular que escaparon las reglas explícitas ──
    # Cubre casos como "Brócoli"/"Brócolis", "Tomate"/"Tomates", etc.
    # Estrategia: si existe tanto la forma sin 's' final como con 's', conservar la que
    # esté en master_map; si ambas o ninguna está, conservar la plural.
    _keys_snapshot = list(canonical_aggregated.keys())
    for key in _keys_snapshot:
        if key not in canonical_aggregated:
            continue  # ya fue fusionada
        k_lower = key.lower()
        # Generar variante hermana (singular↔plural simple)
        if k_lower.endswith('es') and len(k_lower) > 4:
            sister = k_lower[:-2]
        elif k_lower.endswith('s') and not k_lower.endswith('ss') and len(k_lower) > 3:
            sister = k_lower[:-1]
        else:
            sister = k_lower + 's'

        # Buscar la variante hermana en el dict (case-insensitive)
        sister_key = next(
            (k for k in canonical_aggregated if k.lower() == sister),
            None
        )
        if not sister_key or sister_key == key:
            continue

        # Decidir cuál es el nombre canónico: preferir el que esté en master_map
        in_master_key = bool(master_map.get(key) or master_map.get(key.lower()) or master_map.get(key.title()))
        in_master_sister = bool(master_map.get(sister_key) or master_map.get(sister_key.lower()) or master_map.get(sister_key.title()))

        if in_master_sister and not in_master_key:
            target, source = sister_key, key
        elif in_master_key and not in_master_sister:
            target, source = key, sister_key
        else:
            # Ninguna o ambas en master: conservar la plural (más legible en RD)
            target, source = (key, sister_key) if k_lower.endswith('s') else (sister_key, key)

        for u, q in canonical_aggregated[source].items():
            canonical_aggregated[target][u] += q
        del canonical_aggregated[source]
        logging.info(f"🔀 [PLURAL-MERGE] '{source}' → '{target}'")

    # [P6-LACTEOS-MERGE] Mergear "Yogurt" genérico en variante específica
    # ("Yogurt griego sin azúcar", "Yogurt natural", etc.). Bug observable
    # PDF 2026-05-05 22:42: lista mostró "Yogurt griego: 13 potes" Y
    # "Yogurt: 7 Uds" como items separados → suma real 20 potes (>>cap 12).
    # Causa: master_map canonicaliza nombres distintos pero el shopping
    # cap aplica por key independiente. Si hay variante específica, el
    # genérico se folds dentro (más realista — el LLM emite "yogurt" como
    # shorthand del item específico del plan).
    from constants import strip_accents as _strip_accents_merge
    _generic_yogurt_keys = [
        k for k in canonical_aggregated
        if _strip_accents_merge(k.lower()).strip() == 'yogurt'
    ]
    _specific_yogurt_keys = [
        k for k in canonical_aggregated
        if 'yogurt' in _strip_accents_merge(k.lower()) and _strip_accents_merge(k.lower()).strip() != 'yogurt'
    ]
    if _generic_yogurt_keys and _specific_yogurt_keys:
        _target = _specific_yogurt_keys[0]
        for _source in _generic_yogurt_keys:
            if _source == _target:
                continue
            for u, q in canonical_aggregated[_source].items():
                canonical_aggregated[_target][u] += q
            del canonical_aggregated[_source]
            logging.info(
                f"🔀 [P6-LACTEOS-MERGE] '{_source}' → '{_target}' "
                f"(yogurt genérico folds en variante específica)"
            )

    aggregated = canonical_aggregated

    for name, units in aggregated.items():
        master_item = master_map.get(name) or master_map.get(name.lower()) or master_map.get(name.title()) or {}
        
        # --- Normalización Universal por Peso ---
        # Si un ingrediente se contabilizó en conteos/volúmenes o incluso en contenedores (pote, lata)
        # pero tenemos constancia en BD de su peso (density/container), lo sumamos hacia el gramo
        # para que fluya hacia el Bloque 1/2 y asigne empaques matemáticamente exactos.
        g_per_taza = float(master_item.get("density_g_per_cup") or 0)
        g_per_u = float(master_item.get("density_g_per_unit") or 0)
        
        # [Fallback] Si no hay densidad en la BD, buscamos en constants
        if g_per_u <= 0 or g_per_taza <= 0:
            from constants import UNIT_WEIGHTS, strip_accents, VOLUMETRIC_DENSITIES
            n_clean = strip_accents(name.lower())
            
            # [P3-WEIGHT-DEFAULT-NO-UNITIZE · 2026-06-22] Espejo del guard de apply_smart_market_units:
            # un item DECLARADO por peso (default_unit ∈ lb/kg/g) NO recibe densidad-unidad fantasma
            # de UNIT_WEIGHTS tampoco AQUÍ. Sin esto el aggregator unitizaría sandía a 3000g (1 melón
            # entero) internamente mientras el display la costea por libra → ambos paths divergen y el
            # coherence guard podría flaggear magnitud. Mantiene aggregator↔display coherentes.
            _du_weight_agg = (master_item.get("default_unit") or "").strip().lower() in ('lb', 'lbs', 'kg', 'g', 'gr', 'gramo', 'gramos')
            if g_per_u <= 0 and not _du_weight_agg:
                for k, v in UNIT_WEIGHTS.items():
                    if k == n_clean or (re.search(rf'\b{re.escape(k)}(s|es)?\b', n_clean)):
                        g_per_u = v
                        break
                # Fallback para plurales multi-palabra: singularizar cada palabra del input
                # Ej: "guineitos verdes" → "guineito verde" para matchear UNIT_WEIGHTS
                if g_per_u <= 0:
                    n_singular = re.sub(r'(es|s)\b', '', n_clean).strip()
                    for k, v in UNIT_WEIGHTS.items():
                        if k == n_singular or n_singular.startswith(k) or k.startswith(n_singular):
                            g_per_u = v
                            break
                        
            if g_per_taza <= 0:
                for k, v in VOLUMETRIC_DENSITIES.items():
                    if k == n_clean or (re.search(rf'\b{re.escape(k)}(s|es)?\b', n_clean)):
                        # VOLUMETRIC_DENSITIES es g/ml, 1 taza = 236.588 ml
                        g_per_taza = v * 236.588
                        break
        
        if g_per_taza <= 0:
            g_per_taza = DEFAULT_G_PER_TAZA

        container_weight_g = float(master_item.get("container_weight_g") or 0)
        db_container = (master_item.get("market_container") or "").lower()
        
        # Guardamos llaves en lista para modificar diccionario on-the-fly
        
        # Consolidation para Ajo
        if name.lower() == 'ajo':
            u_dientes = 0
            for k in list(units.keys()):
                if k.strip().lower() in ['diente', 'dientes', 'diente.', 'dientes.']:
                    u_dientes += units.pop(k)
            if u_dientes > 0:
                units['cabeza'] = units.get('cabeza', 0) + (u_dientes / 10.0)
            # [P1-AJO-4PACK · 2026-06-22] El ajo en RD se vende en PAQUETES de 4 cabezas
            # (RD$60 el paquete, verificado in-store por el owner) — no por cabeza suelta.
            # Redondear las cabezas necesarias HACIA ARRIBA a paquetes de 4 (mismo patrón que
            # el cartón de huevos) para que un plan que necesite 1-2 cabezas cueste el paquete
            # completo (RD$60), no 15-30. El egg/units-cost branch de _cost_from_market lee el
            # precio real del 4-pack desde Ajo.market_packages [{units:4, price:60}].
            _cab = units.pop('cabeza', 0)
            if _cab and _cab > 0:
                units['paquete (4 uds.)'] = units.get('paquete (4 uds.)', 0) + math.ceil(_cab / 4.0)

        # [P1-LAUREL-LEAF-UNIT · 2026-07-06] "N hojas de laurel" → gramos.
        # Las recetas piden laurel en HOJAS (count unit) y ninguna conversión lo
        # llevaba a peso → el Bloque 1 de envases (master: pote 100 g, RD$150,
        # market_packages poblado) jamás corría y el costeo caía al fallback
        # count × price_per_unit: "4.67 hojas × RD$150 (precio del POTE aplicado
        # POR HOJA) = RD$701" en el PDF del owner. Con la conversión (density
        # 0.6 g/hoja del master; fallback 0.5) el flujo normal compra 1 pote
        # (100 g) = RD$150 una sola vez, como cualquier especia.
        if 'laurel' in name.lower():
            _leaf_qty = 0.0
            for k in list(units.keys()):
                if k.strip().lower() in ('hoja', 'hojas'):
                    _leaf_qty += units.pop(k)
            if _leaf_qty > 0:
                _g_leaf = g_per_u if 0 < g_per_u <= 5.0 else 0.5
                units['g'] = units.get('g', 0.0) + _leaf_qty * _g_leaf
                logging.info(
                    f"🌿 [P1-LAUREL-LEAF-UNIT] {name}: {_leaf_qty:.2f} hojas → "
                    f"{_leaf_qty * _g_leaf:.1f}g (density {_g_leaf} g/hoja) → path de envases"
                )

        # [P1-CASABE-HOJA-UNIT · 2026-07-07] "N hojas de casabe" → gramos. MISMA clase
        # que P1-LAUREL-LEAF-UNIT: el casabe se pide en HOJAS (count unit) pero se vende
        # por PAQUETE (master: paquete 283 g, RD$94, market_packages poblado). Sin
        # conversión a peso, el Bloque 1 de envases jamás corría (exige weight_in_lbs>0)
        # y el costeo caía al fallback count × price_per_unit del path B de
        # `_cost_from_market`: "18.67 hojas × RD$94 (precio del PAQUETE aplicado POR HOJA)
        # = RD$1,755" en el PDF del owner (plan 5f80f797, review visual 30d). Con la
        # conversión (density del master g/hoja; fallback 15) el flujo normal compra
        # 1-2 paquetes = RD$94-199, como cualquier despensa (paridad con el primer plan
        # que mostró "1 paquete RD$85" cuando la receta usó "torta"). Tooltip-anchor:
        # P1-CASABE-HOJA-UNIT.
        if 'casabe' in name.lower():
            _casabe_hojas = 0.0
            for k in list(units.keys()):
                if k.strip().lower() in ('hoja', 'hojas'):
                    _casabe_hojas += units.pop(k)
            if _casabe_hojas > 0:
                _g_hoja = g_per_u if g_per_u and g_per_u > 0 else 15.0
                units['g'] = units.get('g', 0.0) + _casabe_hojas * _g_hoja
                logging.info(
                    f"🍘 [P1-CASABE-HOJA-UNIT] {name}: {_casabe_hojas:.2f} hojas → "
                    f"{_casabe_hojas * _g_hoja:.1f}g (density {_g_hoja} g/hoja) → path de envases"
                )

        # Empaque comercial mínimo para Huevos (Cartones en RD)
        # PRE-PASO: Convertir cualquier peso/volumen de huevos a unidades
        # (ej: "150ml de claras de huevo" ≈ 5 huevos, "100g de huevo" ≈ 2 huevos)
        # Esto evita que claras generen una entrada duplicada por el bloque de peso.
        if name.lower() in ['huevo', 'huevos']:
            egg_weight_g = 50  # 1 huevo entero ≈ 50g
            egg_white_ml = 30  # 1 clara ≈ 30ml
            extra_eggs_from_weight = 0
            
            for k in list(units.keys()):
                k_lower = k.strip().lower()
                if k_lower == 'g':
                    extra_eggs_from_weight += units.pop(k) / egg_weight_g
                elif k_lower == 'ml':
                    extra_eggs_from_weight += units.pop(k) / egg_white_ml
                elif k_lower == 'kg':
                    extra_eggs_from_weight += (units.pop(k) * 1000) / egg_weight_g
                elif k_lower == 'oz':
                    extra_eggs_from_weight += (units.pop(k) * 28.35) / egg_weight_g
                elif k_lower == 'lb':
                    extra_eggs_from_weight += (units.pop(k) * 453.592) / egg_weight_g
                elif k_lower == 'taza':
                    extra_eggs_from_weight += (units.pop(k) * g_per_taza) / egg_weight_g
                elif k_lower in ['cda', 'cdas', 'cucharada', 'cucharadas']:
                    extra_eggs_from_weight += (units.pop(k) * (g_per_taza / 16.0)) / egg_weight_g
                    
            if extra_eggs_from_weight > 0:
                units['unidad'] = units.get('unidad', 0) + math.ceil(extra_eggs_from_weight)
            
            # Ahora consolidar TODAS las unidades en cartones
            u_qty = 0
            for k in list(units.keys()):
                if k.strip().lower() in ['unidad', 'unidades', 'ud', 'uds', 'ud.', 'uds.', 'u', 'u.', 'pieza', 'piezas']:
                    u_qty += units.pop(k)
                elif hasattr(k, 'lower') and 'ud' in k.lower():
                    # Fallback agresivo para atrapar ' Uds.' o cualquier sufijo
                    u_qty += units.pop(k)
            if u_qty > 0:
                # [P3-EGG-REAL-CARTONS · 2026-06-20 · ampliado P1-EGG-CARTON-SIZES 2026-06-22]
                # Los huevos se compran por cartón completo (no por separado). El owner verificó
                # in-store DOS tamaños reales en el mercado DR: cartón de 20 uds (RD$200, mejor
                # para planes de 7 días) y cartón de 30 uds (RD$295, mejor valor/huevo para 15-30
                # días). `_choose_egg_carton` elige el tamaño cost-óptimo según los huevos
                # necesarios, leyendo Huevo.market_packages [{units,price,label}]. Sin datos →
                # fallback al cartón de 30 (comportamiento previo P3-EGG-REAL-CARTONS).
                _egg_sel = _choose_egg_carton(u_qty, master_item.get("market_packages"))
                if _egg_sel:
                    _egg_key = f"cartón ({_egg_sel['units']} uds.)"
                    units[_egg_key] = units.get(_egg_key, 0) + _egg_sel["count"]
                else:
                    units['cartón (30 uds.)'] = units.get('cartón (30 uds.)', 0) + math.ceil(u_qty / 30.0)

        for u in list(units.keys()):
            q = units[u]
            u_lower = u.lower()
            mapped_to_g = False
            
            # 1. Volúmenes
            if u_lower == 'taza':
                units['g'] = units.get('g', 0) + q * g_per_taza
                mapped_to_g = True
            elif u_lower in ['cda', 'cdas', 'cucharada', 'cucharadas']:
                units['g'] = units.get('g', 0) + q * (g_per_taza / 16.0)
                mapped_to_g = True
            elif u_lower in ['cdta', 'cdtas', 'cdita', 'cucharadita']:
                units['g'] = units.get('g', 0) + q * (g_per_taza / 48.0)
                mapped_to_g = True
                
            # 2. Unidades Físicas
            elif u_lower in ['unidad', 'unidades', 'ud', 'uds']:
                if g_per_u > 0:
                    units['g'] = units.get('g', 0) + q * g_per_u
                    mapped_to_g = True
            elif u_lower in ['rebanada', 'rebanadas', 'lonja', 'lonjas']:
                r_weight = 25 if 'pan' in name.lower() else (g_per_u if g_per_u > 0 else 25)
                units['g'] = units.get('g', 0) + q * r_weight
                mapped_to_g = True
                
            # 3. Contenedores Estándar — normalizar a gramos.
            # [P1-3] Antes esto requería `container_weight_g > 0` Y un alias
            # del set hardcodeado. Si master no tenía el peso curado o el
            # usuario tipeaba "1 caja de leche", la unidad NO se normalizaba
            # y el item aparecía duplicado en el delta (uno por peso del
            # plan, otro por paquete del inventario). AHORA:
            #   - `_CONTAINER_UNIT_ALIASES` cubre todos los envases del
            #     mercado dominicano (paquete, pote, lata, cartón, caja,
            #     tetra, galón, jarra, bolsa, sobre, etc).
            #   - Si master no tiene `container_weight_g`, usamos el
            #     fallback por categoría (conservador, mejor under-estimate
            #     que duplicar el item en el delta).
            else:
                is_container_alias = (u_lower == db_container) or (u_lower in _CONTAINER_UNIT_ALIASES)
                if is_container_alias:
                    effective_g = (
                        container_weight_g if container_weight_g > 0
                        else _fallback_container_weight_g(master_item.get("category"))
                    )
                    if effective_g > 0:
                        units['g'] = units.get('g', 0) + q * effective_g
                        mapped_to_g = True
            
            # Borrar la unidad original si logramos migrarla a gramos
            if mapped_to_g:
                del units[u]

    results = []
    categorized_results = defaultdict(list)
    total_estimated_cost = 0.0
    
    PANTRY_STAPLES = {
        'Sal y ajo en polvo', 'Aceite de oliva', 'Aceite de coco', 
        'Aceite de sésamo o maní', 'Salsa de soya', 'Orégano', 
        'Canela', 'Pimienta', 'Sal', 'Vinagre', 'Ajo en polvo'
    }
    # [P2-PDF-2] Items que NO van a la lista de compras: agua del grifo,
    # hielo. Pre-fix era match LITERAL contra el set ('agua', 'hielo',
    # 'agua potable', 'cubos de hielo'): variantes como "agua fría",
    # "agua tibia", "agua caliente", "agua mineral", "agua filtrada" NO
    # estaban listadas y entraban al PDF como items a comprar (caso real
    # 2026-05-05: "Agua fría — 3 lbs" en la sección OTROS). Ahora el
    # check es por palabra-prefix normalizada: nombre debe ser exactamente
    # el prefix o empezar con prefix + espacio (boundary de palabra).
    # Esto evita falso-skip de nombres como "aguaymanto" (fruta) que
    # también empieza con "agua" pero no es agua.
    #
    # Excepción consciente: "agua de coco" se ignora aunque sea producto
    # comprable. Si en algún plan futuro aparece como ingrediente real a
    # comprar, mover a allowlist explícita.
    from constants import strip_accents

    _IGNORE_SHOPPING_PREFIXES = ('agua', 'hielo')
    _IGNORE_SHOPPING_EXACT = {'cubos de hielo'}

    def _should_ignore_shopping(name_str: str) -> bool:
        n = strip_accents(name_str.lower()).strip()
        if not n:
            return True
        if n in _IGNORE_SHOPPING_EXACT:
            return True
        for prefix in _IGNORE_SHOPPING_PREFIXES:
            if n == prefix or n.startswith(prefix + " "):
                return True
        return False

    # ============================================================
    # [P3-HERB-CAP] Cap defensivo de hierbas frescas
    # ------------------------------------------------------------
    # Las hierbas frescas (cilantro, perejil, recao, menta, etc.) NO
    # escalan linealmente con el ciclo del plan: 1 mazo dura 5-7 días
    # refrigerado y >90% se descompone si compras 1 mes de golpe. PDF
    # real (2026-05-05) mostró "Cilantro: 23 Mazos" para mensual × 2
    # personas — culinariamente absurdo y caro (~$200 RD$ en hojas que
    # se botan).
    #
    # Causa: BLOQUE 1.5 de `apply_smart_market_units` calcula
    # `units_needed = max(1, ceil(raw_qty))` sobre el `raw_qty` ya
    # multiplicado por el ciclo. Si el LLM dice "1 mazo cilantro" en
    # 1 receta y el multiplier es 18.67, resulta en 19 mazos.
    #
    # Convención del cap: 1 mazo / persona / semana = uso realista
    # (incluye margen de 1-2 cdas por comida × 3 comidas/día × 7 días).
    # `multiplier × 3/7` deshace el `base_duration_scale = 7/days_generated`
    # aplicado upstream → recuperamos `person_weeks` efectivos del ciclo:
    #   - 2p mensual: 18.67 × 3/7 = 8.0 person-weeks → cap 8 mazos ✓
    #   - 2p quincenal: 9.33 × 3/7 = 4.0 → cap 4 mazos ✓
    #   - 2p semanal: 4.67 × 3/7 = 2.0 → cap 2 mazos ✓
    #   - 1p semanal: 2.33 × 3/7 = 1.0 → cap max(2, 1) = 2 mazos ✓
    #
    # `max(2, ...)` evita cap=1 absurdo para usuarios solo (a veces
    # comprar 1 mazo no es suficiente si la receta dice "1 mazo entero").
    # ============================================================
    _HERB_NAMES_FOR_CAP = {
        'cilantro', 'cilantrico', 'culantro', 'puerro', 'perejil',
        'menta', 'albahaca', 'romero', 'verdura', 'verdurita',
        'recao', 'eneldo', 'tomillo', 'laurel',
        # [P1-CEBOLLIN-HERB-GARNISH · 2026-07-07] clave sin acento (strip_accents aplicado
        # abajo) — cap por persona-semana para que cebollín de guarnición no se multiplique
        # ×ciclo como paquete de 375g. Ver is_herb_mazo (apply_smart_market_units).
        'cebollin',
    }
    _HERB_MAZO_GRAMS = 50.0  # 1 mazo de hierba ≈ 50g

    # [P1-PERSON-WEEKS-CYCLE-AWARE · 2026-07-30] El `3` estaba HARDCODEADO y solo es correcto en
    # planes de 3 días.
    #
    # El comentario de arriba declara la intención: "`multiplier × 3/7` deshace el
    # `base_duration_scale = 7/days_generated` aplicado upstream". Si la inversa de `7/num_days` es
    # `num_days/7`, el `3` solo la deshace cuando `num_days == 3` — y sus cuatro ejemplos trabajados
    # (18.67, 9.33, 4.67, 2.33) están TODOS calculados sobre un ciclo de 3 días. La intención estaba
    # bien escrita al lado de la línea que la traiciona.
    #
    # Efecto por longitud de ciclo (person_weeks obtenidos ÷ correctos):
    #   num_days=2 → ×1,5 (topes demasiado FLOJOS, y encima invisibles: al no dispararse el tope no
    #                      hay ni aviso)      · num_days=3 → exacto  · num_days=4 → ×0,75
    #   num_days=14 → ×0,214 (4,7× APRETADO)  · num_days=25 → ×0,12 (8,3×)
    #
    # Medido en el journal: con `days_len=14 base_scale=0.5`, `person_weeks` queda CLAVADO en 1.0
    # mientras el multiplicador va 1/2/4 (Yogurt 1155→907g, 2311→907g, 4622→907g: el mismo tope para
    # tres duraciones distintas). Con `days_len=3` sí sigue a 1/2/4, que es la firma del hardcode.
    #
    # Muerde a TODOS los topes que leen `_person_weeks`, incluidos los NO perecederos —
    # P6-CANNED-PROTEIN-CAP, P6-OIL-CAP, P6-SPICE, P6-SWEETENER — donde no hay coartada de realismo
    # de almacenamiento: que la lista mensual traiga 2 latas de atún para ~1.600 g de demanda no lo
    # explica la nevera, lo causa esta división.
    #
    # `num_days=None` conserva el comportamiento histórico (3) para cualquier callsite que todavía
    # no lo pase, en vez de cambiarle el cap por debajo sin avisar.
    _pw_days = float(num_days) if num_days and float(num_days) > 0 else 3.0
    _person_weeks = max(1.0, float(multiplier) * _pw_days / 7.0)
    # `round()` (vs `ceil()`) absorbe ruido de floating point: para
    # multiplier=18.67 (display rounded), person_weeks calc = 8.0014... →
    # ceil = 9 (off-by-one). round = 8 ✓. En producción multiplier es
    # `household × cycle × 7/days_generated` con valores exactos así que
    # person_weeks suele caer en entero limpio (2, 4, 8) — `round`
    # equivale al comportamiento esperado sin off-by-ones.
    #
    # [P3-HERB-CAP-FLOOR · 2026-05-16] Floor configurable. Default 1 (era
    # hardcoded 2). Razón: para 1 persona × 7 días, 2 mazos (≈100g, ¼ lb)
    # de cilantro/perejil/etc. son excesivos — 1 mazo (≈50g) basta para
    # uso casual durante una semana. El floor=2 original venía de "evitar
    # cap=1 absurdo si receta dice mazo entero", pero raramente las recetas
    # consumen un mazo COMPLETO; típicamente 1-2 cdas por comida. Para
    # planes 2p+ o cycles >1 semana, person_weeks >= 2 ya elige max(1, 2)=2,
    # así que bajar floor 2→1 solo afecta el caso 1p × 7d (que es donde
    # el usuario reportó "¼ lb es alto"). Operador con plan vegetariano
    # heavy puede bumpear a 2 sin redeploy.
    _HERB_MAZO_CAP_FLOOR = max(1, _knob_env_int("MEALFIT_HERB_MAZO_CAP_FLOOR", 1))
    _herb_cap_mazos = max(_HERB_MAZO_CAP_FLOOR, int(round(_person_weeks)))
    _herb_cap_g = _herb_cap_mazos * _HERB_MAZO_GRAMS

    for _name, _units in list(aggregated.items()):
        if strip_accents(_name.lower()).strip() not in _HERB_NAMES_FOR_CAP:
            continue
        # Cap unidad mazo: BLOQUE 1.5 de `apply_smart_market_units` lo
        # convierte directamente a `units_needed` sin más conversión.
        if 'mazo' in _units and _units['mazo'] > _herb_cap_mazos:
            _old_mazos = _units['mazo']
            _units['mazo'] = float(_herb_cap_mazos)
            _cap_log(
                f"[P3-HERB-CAP] '{_name}' mazo cap: {_old_mazos:.1f} → "
                f"{_herb_cap_mazos} (person_weeks={_person_weeks:.1f}; "
                f"hierbas frescas no se almacenan >1 semana)"
            )
            # [P2-CAPS-COHERENCE-RECONCILE-2 · 2026-05-30] La rama 'mazo' (el
            # disparador COMÚN — el LLM emite '1 mazo de cilantro') no registraba
            # el cap, solo la rama 'g' (abajo, caso raro 'cda'). Sin el registro,
            # el coherence guard ve la divergencia de magnitud en unidad 'mazo'
            # (37→8) como crítica y fuerza un retry innecesario en mode=block
            # (default prod). El test pasaba en falso porque 'found_any' lo
            # satisfacía la rama 'g'. Registrar AMBAS ramas cierra el FP.
            _record_cap_applied(_name, _old_mazos, _units['mazo'], "P3-HERB-CAP")
        # Cap por gramos: BLOQUE 1.5 también convierte g_total → mazos
        # vía `ceil(g_total / 50)`. Si LLM dijo "1 cda cilantro" eso ya
        # se convirtió a g en el loop anterior; cap aquí evita 23 mazos
        # equivalentes en peso.
        if 'g' in _units and _units['g'] > _herb_cap_g:
            _old_g = _units['g']
            _units['g'] = float(_herb_cap_g)
            _cap_log(
                f"[P3-HERB-CAP] '{_name}' peso cap: {_old_g:.0f}g → "
                f"{_herb_cap_g:.0f}g (equivalente a {_herb_cap_mazos} mazos)"
            )
            _record_cap_applied(_name, _old_g, _units['g'], "P3-HERB-CAP")

    # ============================================================
    # [P5-OLIVE-CAP] Cap defensivo de aceitunas
    # ------------------------------------------------------------
    # Las aceitunas se usan como guarnición/topping (~5-15g/serving):
    # 1 frasco de 12 oz (340g) cubre ~25-60 servings → suficiente para
    # uso casi diario de 2 personas durante un mes con margen de 2x.
    #
    # PDF real (2026-05-05): "Aceitunas: 75 frascos (12 oz c/u)" para
    # 2p × mes = 25 kg de aceitunas, ~$15,000 RD$ gastados en algo que
    # se descompondrá antes de consumir 5%. Causa probable: el LLM emite
    # "1 frasco de aceitunas" o pequeños gramajes en varias comidas como
    # garnish; el aggregator suma raw × multiplier 18.67 (mensual×2p) sin
    # cap por categoría salsa/encurtido. Mismo modo de fallo que P3-HERB-CAP
    # pero para encurtidos.
    #
    # Cap: 1 frasco / (3 person-weeks) — cubre uso intensivo (~daily)
    # con margen. Ejemplos:
    #   - 2p mensual (8 person_weeks) → cap 3 frascos
    #   - 2p quincenal (4 pw) → cap 1 frasco (suficiente para 2 sem)
    #   - 2p semanal (2 pw) → cap max(1, 0.67) = 1 frasco
    # Aplica a unidades 'frasco'/'botella'/'pote' Y al peso 'g' (este
    # último a través de un cap-equivalente en gramos de N × 340g).
    # ============================================================
    # [P6-OLIVE-CAP-FIX] Match por SUBSTRING en nombre Y unit, no literal exact.
    # Bug observable PDF 2026-05-05 19:36 ([8b0f351d]): 187 frascos de aceitunas
    # uncapped pese a P5-OLIVE-CAP existente. Causa: en producción master_map
    # canonicaliza a variantes ("Aceitunas Manzanilla", "Aceitunas Verdes")
    # que no estaban en el set literal `{'aceituna', 'aceitunas'}`. Y unit_key
    # puede emitirse como 'frasco (12 oz)' tras formateo con sufijo.
    # Mismo modo de fallo que el cap de huevos cartón con suffix (ver fix-2).
    #
    # [P6-OLIVE-CAP-FIX-3] (corrida 20:36 [265055c3]): pese a FIX-1, lista
    # mostró "94 frascos". Causa: el cap solo cubre `'g'` y unit substring
    # `'frasco'`/'botella'/'pote'. Pero LLM emite "12 oz aceitunas" → unit_key
    # es `'oz'` → no matchea ninguna substring → cap silenciosamente skipped.
    # Después loop de weight_in_lbs (línea 2888) suma 'oz'+'lb'+'kg'+'ml'+'l'
    # y BLOQUE 1 de apply_smart_market_units divide por 340g/frasco → 94.
    # Fix: sumar TODOS los units de peso a gramos equivalentes y capear el
    # total. Si excede, vaciar weight units y setear 'g' al cap.
    _OLIVE_SUBSTRINGS = ('aceituna', 'olive')
    _OLIVE_UNIT_SUBSTRINGS = ('frasco', 'botella', 'pote')
    _OLIVE_FRASCO_GRAMS_DEFAULT = 340.194  # 12 oz — fallback si el catálogo no resuelve container
    _WEIGHT_UNIT_TO_G = {
        'g': 1.0, 'kg': 1000.0, 'oz': 28.3495,
        'lb': 453.592, 'lbs': 453.592, 'ml': 1.0, 'l': 1000.0,
    }

    _olive_cap_frascos = max(1, int(round(_person_weeks / 3.0)))

    for _name, _units in list(aggregated.items()):
        _name_norm = strip_accents(_name.lower()).strip()
        if not any(s in _name_norm for s in _OLIVE_SUBSTRINGS):
            continue
        # [P1-OLIVE-CAP-REAL-CONTAINER · 2026-07-08] (SQL forense vivo) `_OLIVE_FRASCO_GRAMS`
        # estaba hardcodeado a 340.194g (12oz) desde P5-OLIVE-CAP original, pero el catálogo
        # REAL de `master_ingredients` tiene 'Aceitunas' con `container_weight_g=142` (frasco
        # más chico) — un cap calibrado a "3 frascos" con la constante vieja calculaba
        # `_olive_cap_g=1021g`, que `apply_smart_market_units` (BLOQUE 1, container REAL)
        # convertía de vuelta a ~8 frascos reales de 142g — el cap "funcionaba" en gramos pero
        # mentía en unidades de compra (2.7x más frascos de los pretendidos). Resuelve el
        # tamaño REAL desde `master_map` (ya indexado arriba, línea ~7002) y solo cae al
        # default hardcodeado si el catálogo no tiene el dato — nunca vuelve a driftear.
        _olive_m_item = (master_map.get(_name) or master_map.get(_name.lower())
                          or master_map.get(_name.title()))
        _OLIVE_FRASCO_GRAMS = _OLIVE_FRASCO_GRAMS_DEFAULT
        if _olive_m_item:
            try:
                _real_container_g = float(_olive_m_item.get("container_weight_g") or 0)
                if _real_container_g > 0:
                    _OLIVE_FRASCO_GRAMS = _real_container_g
            except (TypeError, ValueError):
                pass
        _olive_cap_g = _olive_cap_frascos * _OLIVE_FRASCO_GRAMS
        # Cap unit-based ('frasco', 'botella', 'pote' substring)
        for _unit_key in list(_units.keys()):
            if not isinstance(_unit_key, str):
                continue
            _unit_lower = _unit_key.lower()
            if not any(u in _unit_lower for u in _OLIVE_UNIT_SUBSTRINGS):
                continue
            if _units[_unit_key] > _olive_cap_frascos:
                _old = _units[_unit_key]
                _units[_unit_key] = float(_olive_cap_frascos)
                _record_cap_applied(_name, _old, _units[_unit_key], "P5-OLIVE-CAP")
                _cap_log(
                    f"[P5-OLIVE-CAP] '{_name}' {_unit_key!r} cap: {_old:.1f} → "
                    f"{_olive_cap_frascos} (person_weeks={_person_weeks:.1f}; "
                    f"olivas son guarnición, no main course)"
                )
        # [P6-OLIVE-CAP-FIX-3] Cap total de peso (sumando g/kg/oz/lb/ml/l).
        # Captura el caso donde LLM emite "X oz aceitunas" → unit_key 'oz'
        # no matchea substring 'frasco' pero igual produce 94 frascos en
        # display vía conversión a peso → BLOQUE 1.
        _total_weight_g = sum(
            _units.get(u, 0) * _WEIGHT_UNIT_TO_G[u]
            for u in _WEIGHT_UNIT_TO_G
            if u in _units
        )
        if _total_weight_g > _olive_cap_g:
            _present_units = {u: _units[u] for u in _WEIGHT_UNIT_TO_G if u in _units}
            for _wu in list(_present_units.keys()):
                del _units[_wu]
            _units['g'] = float(_olive_cap_g)
            _record_cap_applied(_name, _total_weight_g, _olive_cap_g, "P5-OLIVE-CAP")
            _cap_log(
                f"[P5-OLIVE-CAP] '{_name}' peso total cap: {_total_weight_g:.0f}g "
                f"(de {_present_units}) → {_olive_cap_g:.0f}g "
                f"(≈{_olive_cap_frascos} frascos de {_OLIVE_FRASCO_GRAMS:.0f}g; "
                f"person_weeks={_person_weeks:.1f})"
            )
        # [P6-OLIVE-CAP-FIX-4] Cap por COUNT cuando LLM emite "X aceitunas"
        # como conteo de unidades. Bug observable PDF 2026-05-05 21:34:
        # 234 frascos uncapped pese a FIX-3. Causa: LLM emite "5 aceitunas
        # verdes" → unit_key 'unidad'/'unidades' → no matchea substring
        # 'frasco' Y no está en _WEIGHT_UNIT_TO_G → silenciosamente skipped.
        # Después apply_smart_market_units BLOQUE 2 multiplica por density
        # (~5g/aceituna) y BLOQUE 1 divide por container_weight_g (340g) →
        # 234 frascos display.
        # Cap_count = cap_g / density_per_olive (5g/unidad estándar).
        _OLIVE_DENSITY_G_PER_UNIT = 5.0
        _olive_cap_count = max(2, int(round(_olive_cap_g / _OLIVE_DENSITY_G_PER_UNIT)))
        for _unit_key in ('unidad', 'unidades', 'ud', 'uds'):
            if _unit_key in _units and _units[_unit_key] > _olive_cap_count:
                _old = _units[_unit_key]
                _units[_unit_key] = float(_olive_cap_count)
                _record_cap_applied(_name, _old, _units[_unit_key], "P5-OLIVE-CAP")
                _cap_log(
                    f"[P5-OLIVE-CAP] '{_name}' {_unit_key} count cap: "
                    f"{_old:.0f} → {_olive_cap_count} (≈{_olive_cap_frascos} "
                    f"frascos × {int(_OLIVE_FRASCO_GRAMS/_OLIVE_DENSITY_G_PER_UNIT)} "
                    f"olivas/frasco; person_weeks={_person_weeks:.1f})"
                )
        # [P6-OLIVE-CAP-FIX-5 2026-05-07] Cap por unidades VOLUMÉTRICAS
        # (taza/cda/cdta). Bug observable PDF 2026-05-07 00:49 (plan
        # 4374fb17): "Aceitunas: 47 frascos (12 oz c/u)" = 16 kg para 1p×mes.
        # Causa: LLM emitió "X taza/cda de aceitunas" en varias comidas →
        # unit_key 'taza'/'cda'/'cdta' → NO matchea substring frasco/etc,
        # NO está en _WEIGHT_UNIT_TO_G, NO está en unidad/unidades → escapa
        # los 3 caps anteriores. Después apply_smart_market_units multiplica
        # taza/cda por densidad volumétrica → frascos display.
        # Convertir taza/cda/cdta a gramos equivalentes y capear si exceden.
        _VOLUMETRIC_TO_G = {
            'taza': 130.0,   # 1 taza ≈ 130g aceitunas drained
            'tazas': 130.0,
            'cda': 14.0,     # 1 cda ≈ 14g aceitunas
            'cdas': 14.0,
            'cucharada': 14.0,
            'cucharadas': 14.0,
            'cdta': 5.0,     # 1 cdta ≈ 5g aceitunas (~1 oliva)
            'cdtas': 5.0,
            'cucharadita': 5.0,
            'cucharaditas': 5.0,
        }
        _vol_total_g = sum(
            _units.get(u, 0) * _VOLUMETRIC_TO_G[u]
            for u in _VOLUMETRIC_TO_G
            if u in _units
        )
        if _vol_total_g > _olive_cap_g:
            _vol_present = {u: _units[u] for u in _VOLUMETRIC_TO_G if u in _units}
            for _vu in list(_vol_present.keys()):
                del _units[_vu]
            # Sumamos al peso 'g' existente (defensa: si ya hay 'g' del weight
            # path, no perdemos; si no, creamos nuevo)
            _units['g'] = _units.get('g', 0.0) + float(_olive_cap_g)
            _record_cap_applied(_name, _vol_total_g, _olive_cap_g, "P5-OLIVE-CAP")
            _cap_log(
                f"[P5-OLIVE-CAP] '{_name}' volumétrico cap: {_vol_total_g:.0f}g "
                f"(de {_vol_present}) → {_olive_cap_g:.0f}g "
                f"(≈{_olive_cap_frascos} frascos de {_OLIVE_FRASCO_GRAMS:.0f}g; "
                f"person_weeks={_person_weeks:.1f})"
            )

    # ============================================================
    # [P6-CITRUS-CAP] Cap defensivo para cítricos perecederos
    # ------------------------------------------------------------
    # PDF 2026-05-05 19:36 ([8b0f351d]): "Limón: 51 Uds." para 2p × mes
    # = ~1 limón/día/persona. Excesivo para uso típico (sazón, aderezo,
    # bebida): ½ limón/día/persona suficiente. Limón dura 2-3 semanas en
    # nevera, así que el problema NO es waste por descomposición sino
    # over-buying matemático: el LLM emite "jugo de 1/2 limón" en varias
    # comidas → suma raw × 18.67 = 30-50 limones en lista final.
    #
    # Cap: 4/persona/sem = uso intensivo (pescado, ensaladas, agua
    # citronizada). Para 2p × mes (8 person_weeks): cap 32 limones
    # (vs 51 PDF; reducción 37%).
    # Aplica a 'unidad'/'unidades' Y a 'g' (× 60g/limón promedio).
    # ============================================================
    # [P6-CITRUS-CAP-TIGHTEN 2026-05-06] Bajado 4→3 limones/persona/semana.
    # PDF mostraba 20 limones para 2p×mes (~2.5/persona/sem) y el cap previo
    # (4×8=32) no se activaba. 3/sem = 12/persona/mes = 24 para 2p — uso
    # intensivo realista (jugos, marinados, ensaladas). Si un usuario hace
    # mojo dominicano frecuente y necesita más, el cap se puede subir vía
    # config de master_ingredients sin tocar este default.
    _CITRUS_PER_WEEK_PER_PERSON = {
        'limon':       (3, 60.0),
        'limones':     (3, 60.0),
        'lima':        (3, 60.0),
        'limas':       (3, 60.0),
        # Naranja para jugo: 3/persona/sem (~½ vaso jugo/día). Naranja
        # entera de comer es categoría aparte (P6-FRUITS-LARGE-CAP),
        # pero por safety capeamos también el unit count global.
        'naranja':   (3, 200.0),
        'naranjas':  (3, 200.0),
    }

    for _name, _units in list(aggregated.items()):
        _name_norm = strip_accents(_name.lower()).strip()
        if _name_norm not in _CITRUS_PER_WEEK_PER_PERSON:
            continue
        _per_week, _density_default = _CITRUS_PER_WEEK_PER_PERSON[_name_norm]
        # Prefer master_ingredients.density_g_per_unit cuando esté poblado:
        # Naranja master tiene 180 g/ud (no 200), Limón 50 g/ud (no 60). Usar
        # el default hardcoded sin reconciliar producía cap_g = 3 × 200 = 600g
        # pero apply_smart_market_units divide después por master.density (180)
        # → ceil(600/180) = 4 unidades, off-by-1 vs el cap intencional de 3.
        _master_for_density = (
            master_map.get(_name)
            or master_map.get(_name.lower())
            or master_map.get(_name.title())
            or {}
        )
        try:
            _master_density = float(_master_for_density.get("density_g_per_unit") or 0)
        except (TypeError, ValueError):
            _master_density = 0.0
        _density = _master_density if _master_density > 0 else _density_default
        _citrus_cap_units = max(2, int(round(_per_week * _person_weeks)))
        _citrus_cap_g = _citrus_cap_units * _density

        for _unit_key in ('unidad', 'unidades'):
            if _unit_key in _units and _units[_unit_key] > _citrus_cap_units:
                _old = _units[_unit_key]
                _units[_unit_key] = float(_citrus_cap_units)
                _record_cap_applied(_name, _old, _units[_unit_key], "P6-CITRUS-CAP")
                _cap_log(
                    f"[P6-CITRUS-CAP] '{_name}' {_unit_key} cap: {_old:.1f} → "
                    f"{_citrus_cap_units} (person_weeks={_person_weeks:.1f}; "
                    f"~{_per_week}/persona/semana es uso intensivo realista)"
                )
        if 'g' in _units and _units['g'] > _citrus_cap_g:
            _old_g = _units['g']
            _units['g'] = float(_citrus_cap_g)
            _record_cap_applied(_name, _old_g, _units['g'], "P6-CITRUS-CAP")
            _cap_log(
                f"[P6-CITRUS-CAP] '{_name}' peso cap: {_old_g:.0f}g → "
                f"{_citrus_cap_g:.0f}g (≈{_citrus_cap_units} unidades)"
            )

    # ============================================================
    # [P5-VEG-CAP] Cap realista de vegetales perecederos sobre-asignados
    # ------------------------------------------------------------
    # Algunos vegetales (cebolla en particular) se acumulan en la lista
    # mensual a niveles matemáticamente correctos pero realísticamente
    # excesivos: el LLM puede pedir "1 cebolla picada" en cada comida →
    # 1 cebolla × 4 comidas/día × 30 días × 2p ≈ 240 cebollas raw, que
    # tras consolidación llegan a 70+ unidades. PDF 2026-05-05 mostró
    # "Cebolla: 23 lbs (~70 Uds.)" — coherente con el plan generado pero
    # 2-3× lo que se compraría realísticamente para almacenar (cebolla
    # cruda dura 3-4 semanas en clima tropical).
    #
    # Cap por person-week con valores realistas de uso semanal por
    # persona. Aplica al unit count si presente; si hay peso 'g',
    # también se cap usando density_g_per_unit del master_item.
    #
    # Convención conservadora: solo capear ingredientes con consumo
    # definido. Extender el dict cuando se observen otros casos en
    # producción (NO un cap blanket por categoría — el riesgo de
    # under-supply es alto si capeas algo que sí necesita uso intensivo).
    # ============================================================
    # Convención: tupla (units/persona/semana, density_g_default).
    # `density_g_default` se usa cuando master_item no tiene
    # `density_g_per_unit` (caso común en test sin DB; o ingredientes
    # nuevos sin curar). Valores reflejan tamaño promedio dominicano.
    #
    # [P5-VEG-CAP] cebolla (corrida 2026-05-05 13:11)
    # [P6-VEG-EXT] papa, plátano maduro, zanahoria, coliflor (PDF 13:33)
    #   - Papa 44 Uds/mes para 2p → cap 40
    #   - Plátano maduro 66 Uds/mes para 2p → cap 40 (storage realismo:
    #     plátano se pasa en 4-7 días, comprar 66 garantiza waste)
    #   - Zanahoria 35 Uds/mes para 2p → cap 32
    #   - Coliflor 12 cabezas/mes para 2p → cap 8 (vendida por cabeza,
    #     el cap loop chequea 'cabeza'/'cabezas' además de 'unidad')
    # Variantes incluidas para cubrir AMBOS environments:
    # - Producción: master_map canonicaliza ('Papa blanca' → 'Papa').
    # - Test sin DB: normalize_name strippa stopwords pero conserva
    #   forma plural ('Papas blancas' → 'Papas blancas'). El dict
    #   incluye plurales y formas con adjetivos comunes para no requerir
    #   master_map en pipelines de prueba.
    # Densities alineadas con `constants.UNIT_WEIGHTS` (mantener en sync
    # para que el cap_g produzca el cap_units correcto vía density del
    # path BLOQUE 2 — divergencia produciría off-by-density-ratio).
    _VEG_PER_WEEK_PER_PERSON = {
        # cebolla: 4/persona/sem = sofrito diario + 2 ensaladas/sem.
        # Para 2p × mes (8 person_weeks): cap 32 cebollas (vs 70 pre-cap).
        'cebolla':  (4, 110.0),
        'cebollas': (4, 110.0),
        # papa: 5/persona/sem = uso intensivo (estofado, sopa, asada).
        # Density 150g (UNIT_WEIGHTS["papa"]). Para 2p × mes: cap 40
        # papas (vs 44 PDF; reducción modest pero realista).
        'papa':           (5, 150.0),
        'papas':          (5, 150.0),
        'papa blanca':    (5, 150.0),
        'papas blancas':  (5, 150.0),
        # plátano maduro: 5/persona/sem = casi 1/día como acompañamiento.
        # Density 280g (UNIT_WEIGHTS["platano maduro"]). Storage upper
        # bound — más de eso garantiza waste por maduración.
        # Para 2p × mes: cap 40 plátanos (vs 66 PDF; reducción 39%).
        'platano':           (5, 280.0),
        'platanos':          (5, 280.0),
        'platano maduro':    (5, 280.0),
        'platanos maduros':  (5, 280.0),
        # zanahoria: 4/persona/sem = ensaladas + sofrito + jugos.
        # Density 75g (UNIT_WEIGHTS["zanahoria"]). Para 2p × mes:
        # cap 32 zanahorias (vs 35 PDF; reducción modest).
        'zanahoria':  (4, 75.0),
        'zanahorias': (4, 75.0),
        # coliflor: 1/persona/sem (cabeza ~500g rinde 2 porciones).
        # Density 500g (UNIT_WEIGHTS["coliflor"]). Para 2p × mes:
        # cap 8 cabezas (vs 12 PDF; reducción 33%). Storage realismo:
        # 1 cabeza dura 7-14 días refrigerada — comprar 12 a la vez
        # garantiza que la mitad se pase antes de consumir.
        'coliflor':   (1, 500.0),
        'coliflores': (1, 500.0),
        # [P6-VEG-EXT-2] auyama, plátano verde, berenjena
        # PDF 2026-05-05 19:36 ([8b0f351d]):
        #   - Auyama: 34¾ lbs (~31 Uds.) para 2p×mes = ~15 kg, absurdo
        #   - Plátano verde: 28 Uds. para 2p×mes = ~3.5/sem/persona
        #   - Berenjena: 12½ lbs (~19 Uds.) = ~3 berenjenas/sem/persona
        # Auyama: 1/persona/sem (puré, sopa, ensalada — uso moderado).
        # Density 1100g (típico DR squash pequeña). Para 2p × mes: cap 8
        # unidades (~17.6 lbs, vs 31 Uds PDF; reducción 74%).
        'auyama':  (1, 1100.0),
        'auyamas': (1, 1100.0),
        # Plátano verde: 3/persona/sem = mangú/tostones/mofongo 2-3×/sem.
        # Density 280g (UNIT_WEIGHTS["platano"]). Para 2p × mes: cap 24
        # plátanos (vs 28 PDF; reducción modest pero realista).
        'platano verde':    (3, 280.0),
        'platanos verdes':  (3, 280.0),
        # Berenjena: 2/persona/sem = parrillada o salteado 2×/sem.
        # Density 300g (berenjena dominicana mediana). Para 2p × mes:
        # cap 16 berenjenas (vs 19 PDF; reducción modest).
        'berenjena':  (2, 300.0),
        'berenjenas': (2, 300.0),
        # [P6-VEG-EXT-3] Batata (PDF 2026-05-05 21:12: 51 unidades para
        # 2p × mes — absurdo). Batata es starchy, uso 2-3×/sem como carbo.
        # 3/persona/sem = 24 max para 2p × mes (vs 51 PDF; reducción 53%).
        # Density 200g (batata dominicana mediana, smaller than papa).
        'batata':  (3, 200.0),
        'batatas': (3, 200.0),
        # [P6-VEG-EXT-4] Yuca (PDF 2026-05-05 21:34: 17 unidades para 2p × mes
        # = ~7 kg de yuca, alto). Yuca es staple del DR como carbo, uso
        # 2-3×/sem en almuerzo principal. 3/persona/sem = 24 max para
        # 2p × mes. Density 400g (yuca dominicana mediana — más grande
        # que papa/batata).
        'yuca':  (3, 400.0),
        'yucas': (3, 400.0),
        # [P6-VEG-EXT-5] Guineo (PDF 2026-05-05 21:50: 56 unidades para
        # 2p × mes — excesivo, ~7 guineos/sem/persona). Guineo (banana
        # común DR) se usa típicamente en desayuno o merienda. Distinto a
        # plátano maduro (cocinable) — guineo NO entra en su entry.
        # 4/persona/sem = 32 max para 2p × mes (vs 56 PDF; reducción 43%).
        # Density 120g (guineo DR mediano).
        'guineo':  (4, 120.0),
        'guineos': (4, 120.0),
        # [P6-VEG-EXT-5-FIX] Guineo verde (PDF 2026-05-05 23:12: 168 Uds
        # para 2p × mes — absurdo, ~21/sem/persona). 'Guineo verde' es
        # un item distinto a 'Guineo' en master_map (guineo verde para
        # mangú/sancocho, guineo común para postre/fruta). Sin esta
        # entry el cap de 'guineo' no captura la variante 'verde' (exact
        # match, no substring). Density 120g (similar a guineo común).
        'guineo verde':  (4, 120.0),
        'guineos verdes': (4, 120.0),
        # [P6-TOFU-CAP] Tofu (PDF 2026-05-05 23:33: 31 lbs para 2p × mes
        # = ~14 kg de tofu, absurdo). Tofu es proteína vegana de uso
        # 2-3×/sem como sustituto de carne. 1 lb/persona/sem = 8 lbs
        # max para 2p × mes. Density 454g (paquete típico 1 lb).
        # NOTA: tofu por unidad común es paquete; cap aplica en lbs y g.
        'tofu':         (1, 454.0),
        'tofu firme':   (1, 454.0),
        'tofu suave':   (1, 454.0),
        # [P6-VEG-EXT-6] Tomate y ñame
        # PDF 2026-05-05 21:50: Tomate 38 Uds, Ñame 12 Uds.
        # Tomate: uso constante en sofrito + ensaladas. 5/persona/sem = 40
        # max para 2p × mes (vs 38 PDF — está al límite, deja margen).
        # Density 100g (tomate DR mediano).
        'tomate':  (5, 100.0),
        'tomates': (5, 100.0),
        # Ñame: starchy similar a yuca. Uso 1-2×/sem como carbo. 2/persona/sem
        # = 16 max para 2p × mes. Density 600g (ñame DR es grande, +/-
        # similar a yuca pero más alargado).
        'ñame':  (2, 600.0),
        'ñames': (2, 600.0),
        'name':  (2, 600.0),  # sin tilde (strip_accents)
        'names': (2, 600.0),
        # [P6-VEG-EXT-7] Brócoli (PDF 2026-05-05 22:42: 14 cabezas para
        # 2p × mes — excesivo). Brócoli se usa 1-2×/sem como acompañante.
        # 1/persona/sem = 8 cabezas max para 2p × mes. Density 500g
        # (cabeza DR mediana, similar a coliflor).
        'brocoli':   (1, 500.0),
        'brocolis':  (1, 500.0),
        # [P2-RABANO-CAP · 2026-07-06] (review #14, plan 17c3fa8f) Rábano es GUARNICIÓN (rodajas
        # encurtidas al lado): la receta pedía 135g/porción × ~13 repeticiones en 30 días → 9
        # paquetes = RD$765 por un adorno. 2/persona/sem, density default 40g (rábano pequeño DR;
        # el master corrigió su density errónea de 4.5g→25g, paquete 200g/8 unid). Cap ~1 paquete.
        'rabano':   (2, 40.0),
        'rabanos':  (2, 40.0),
    }

    for _name, _units in list(aggregated.items()):
        _name_norm = strip_accents(_name.lower()).strip()
        if _name_norm not in _VEG_PER_WEEK_PER_PERSON:
            continue
        _per_week, _default_density = _VEG_PER_WEEK_PER_PERSON[_name_norm]
        _veg_cap_units = max(2, int(round(_per_week * _person_weeks)))

        # [P6-VEG-EXT] Cap unit count para path BLOQUE 1/4. Incluye
        # 'cabeza'/'cabezas' para coliflor/repollo/lechugas cuando el
        # aggregator usa esas unidades nativas.
        for _unit_key in ('unidad', 'unidades', 'cabeza', 'cabezas'):
            if _unit_key in _units and _units[_unit_key] > _veg_cap_units:
                _old = _units[_unit_key]
                _units[_unit_key] = float(_veg_cap_units)
                _record_cap_applied(_name, _old, _units[_unit_key], "P5-VEG-CAP")
                _cap_log(
                    f"[P5-VEG-CAP] '{_name}' {_unit_key} cap: {_old:.1f} → "
                    f"{_veg_cap_units} (person_weeks={_person_weeks:.1f}; "
                    f"realismo de almacenamiento + uso semanal por persona)"
                )

        # Cap por gramos: aplica cuando el aggregator ya convirtió
        # 'unidad' → 'g' usando density (caso típico en BLOQUE 2).
        # Density preferida: master_item.density_g_per_unit; fallback al
        # default del dict (no rompemos cap si DB no está disponible).
        if 'g' in _units:
            _master_item = (
                master_map.get(_name)
                or master_map.get(_name.lower())
                or master_map.get(_name.title())
            )
            _density = _default_density
            if _master_item:
                _master_density = float(_master_item.get('density_g_per_unit') or 0)
                if _master_density > 0:
                    _density = _master_density
            _veg_cap_g = _veg_cap_units * _density
            if _units['g'] > _veg_cap_g:
                _old_g = _units['g']
                _units['g'] = float(_veg_cap_g)
                _cap_log(
                    f"[P5-VEG-CAP] '{_name}' peso cap: {_old_g:.0f}g → "
                    f"{_veg_cap_g:.0f}g (≈{_veg_cap_units} unidades a "
                    f"{_density:.0f}g c/u)"
                )
                _record_cap_applied(_name, _old_g, _units['g'], "P5-VEG-CAP")

    # ============================================================
    # [P6-SPICE-CAP] Cap defensivo para especias en sobres
    # ------------------------------------------------------------
    # Especias como pimienta y orégano son condimentos de uso CONSTANTE
    # pero CANTIDAD MÍNIMA por dish (~0.5g). El LLM las menciona como
    # "1 pizca de pimienta" o "1 sobre" en CADA comida del plan;
    # aggregator suma raw × multiplier 18.67 (mensual×2p) → 38 sobres
    # de pimienta. PDF real (2026-05-05 13:33).
    #
    # Realmente 1 sobre estándar de 28g de pimienta/orégano dura 2-6
    # MESES para uso normal. Comprar 38 sobres = ~1 kg = más pimienta
    # que toda la cocina dominicana junta usa en 6 meses.
    #
    # Cap: 1 sobre por cada 4 person-weeks. Conservador para no quedarse
    # corto si el operador realmente cocina con especia intensiva:
    #   - 2p mensual (8 pw) → cap 2 sobres (~56g, dura 2-4 meses)
    #   - 2p quincenal (4 pw) → cap 1 sobre
    #   - 4p mensual (16 pw) → cap 4 sobres
    #
    # Aplica a especias secas comunes en cocina dominicana. Especias que
    # se usan crudas/frescas (cilantro, perejil) ya están cubiertas por
    # P3-HERB-CAP en su unidad nativa (mazo).
    # ============================================================
    # [P6-SPICE-CAP-FIX-3] Renombrado de set→tuple substring (mismo patrón
    # que P6-SAUCE-CAP-FIX). Bug observable PDF 2026-05-06 01:11-01:16:
    # "Canela en polvo: 19 sobres (28g c/u) = 532g" pese a tener 'canela'
    # y 'canela en polvo' en el set anterior. Causa: master_map / aggregator
    # canonicaliza con modificadores no anticipados (e.g. "canela en polvo
    # molida", "canela ceylán", "canela molida fina") → exact match `not in`
    # falla silenciosamente. Mismo síntoma que "salsa de soya baja en sodio"
    # documentado en P6-SAUCE-CAP-FIX. Solución: substring match con bases
    # cortas. Para fresh vs polvo (ajo, cebolla, jengibre, laurel, nuez)
    # mantenemos frase completa para evitar false-positive en frescos.
    _SPICE_SUBSTRINGS = (
        'pimienta',         # cubre negra/blanca/cayena/de jamaica/etc
        'oregano',          # cubre dominicano/seco/orejón
        'canela',           # cubre 'en polvo'/molida/ceylán/fina
        'comino',           # cubre molido/en polvo/entero
        'paprika',
        'pimenton',         # `pimentón` normalizado por strip_accents
        'curcuma',          # `cúrcuma` normalizado
        'sazon',            # `sazón` normalizado
        'nuez moscada',     # frase completa (NO 'nuez' → almendras/nueces)
        'ajo en polvo',     # frase completa (NO 'ajo' → ajo fresco cabeza)
        'cebolla en polvo', # frase completa (NO 'cebolla' → fresca)
        'jengibre en polvo',# frase completa (NO 'jengibre' → fresco)
        'laurel en polvo',  # frase completa (NO 'laurel' → hojas enteras)
    )
    _SPICE_SOBRE_GRAMS = 28.0  # sobre estándar dominicano

    _spice_cap_sobres = max(1, int(round(_person_weeks / 4.0)))
    _spice_cap_g = _spice_cap_sobres * _SPICE_SOBRE_GRAMS

    # [P6-SPICE-CAP-FIX-2] Mismo bug que P6-OLIVE-CAP-FIX-3: el cap solo
    # cubría `'g'` y substring sobre/s. Bug observable PDF 2026-05-05 21:12:
    # "Canela en polvo: 19 sobres (28g c/u)" = 532g uncapped. Causa: LLM
    # emite "1 oz canela" → unit_key 'oz' → no matchea 'sobre' ni 'g'.
    # Fix: sumar TOTAL de peso (g/kg/oz/lb/ml/l) y capear el total.
    for _name, _units in list(aggregated.items()):
        _name_norm = strip_accents(_name.lower()).strip()
        if not any(_s in _name_norm for _s in _SPICE_SUBSTRINGS):
            continue
        for _unit_key in ('sobre', 'sobres', 'sobrecito', 'sobrecitos'):
            if _unit_key in _units and _units[_unit_key] > _spice_cap_sobres:
                _old = _units[_unit_key]
                _units[_unit_key] = float(_spice_cap_sobres)
                _record_cap_applied(_name, _old, _units[_unit_key], "P6-SPICE-CAP")
                _cap_log(
                    f"[P6-SPICE-CAP] '{_name}' {_unit_key} cap: {_old:.1f} → "
                    f"{_spice_cap_sobres} (person_weeks={_person_weeks:.1f}; "
                    f"especia dura meses, condimento de cantidad mínima)"
                )
        # [P6-SPICE-CAP-FIX-2] Cap por peso TOTAL (cubre LLM emitting 'oz')
        _total_weight_g = sum(
            _units.get(u, 0) * _WEIGHT_UNIT_TO_G[u]
            for u in _WEIGHT_UNIT_TO_G
            if u in _units
        )
        if _total_weight_g > _spice_cap_g:
            _present_units = {u: _units[u] for u in _WEIGHT_UNIT_TO_G if u in _units}
            for _wu in list(_present_units.keys()):
                del _units[_wu]
            _units['g'] = float(_spice_cap_g)
            _record_cap_applied(_name, _total_weight_g, _spice_cap_g, "P6-SPICE-CAP")
            _cap_log(
                f"[P6-SPICE-CAP] '{_name}' peso total cap: {_total_weight_g:.0f}g "
                f"(de {_present_units}) → {_spice_cap_g:.0f}g "
                f"(≈{_spice_cap_sobres} sobres 28g; "
                f"person_weeks={_person_weeks:.1f})"
            )

    # ============================================================
    # [P6-SWEETENER-CAP] Cap defensivo para edulcorantes (estevia, sucralosa,
    # eritritol, etc.)
    # ------------------------------------------------------------
    # Edulcorantes son condimentos de uso CONSTANTE pero CANTIDAD MÍNIMA por
    # porción (~0.1-1g). Una caja de 50g de estevia dura 2-3 MESES en uso
    # normal. PDF 2026-05-06 17:36 mostró "Estevia: 3 caja (50g c/u)" para
    # 1 persona × 1 mes — equivale a 6-9 meses de stock.
    #
    # Cap: 1 caja de 50g por cada 8 person-weeks. Conservador para usuarios
    # que realmente endulzan a diario. Trade-off: si alguien hornea con
    # estevia industrialmente, queda corto — pero ese caso es raro y la
    # subestimación se nota inmediatamente vs la sobre-compra invisible.
    #   - 1p × mes (4 pw)        → cap 1 caja (50g, ~2-3 meses stock)
    #   - 1p × quincenal (2 pw)  → cap 1 caja
    #   - 2p × mes (8 pw)        → cap 1 caja
    #   - 4p × mes (16 pw)       → cap 2 cajas
    #
    # Aplica a edulcorantes acalóricos comunes en cocina dominicana. Azúcar
    # tradicional NO entra aquí (es ingrediente real, va en su propio cap).
    # ============================================================
    _SWEETENER_SUBSTRINGS = (
        'estevia',          # `stevia` normalizado por strip_accents
        'stevia',
        'sucralosa',
        'eritritol',
        'monk fruit',
        'edulcorante',
        'splenda',          # marca común
        'sweet n low',
        'allulosa',
    )
    _SWEETENER_BOX_GRAMS = 50.0  # caja estándar dominicana

    _sweetener_cap_boxes = max(1, int(round(_person_weeks / 8.0)))
    _sweetener_cap_g = _sweetener_cap_boxes * _SWEETENER_BOX_GRAMS

    for _name, _units in list(aggregated.items()):
        _name_norm = strip_accents(_name.lower()).strip()
        if not any(_s in _name_norm for _s in _SWEETENER_SUBSTRINGS):
            continue
        # Cap por unit-key (caja/cajas/cajita)
        for _unit_key in ('caja', 'cajas', 'cajita', 'cajitas'):
            if _unit_key in _units and _units[_unit_key] > _sweetener_cap_boxes:
                _old = _units[_unit_key]
                _units[_unit_key] = float(_sweetener_cap_boxes)
                _record_cap_applied(_name, _old, _units[_unit_key], "P6-SWEETENER-CAP")
                _cap_log(
                    f"[P6-SWEETENER-CAP] '{_name}' {_unit_key} cap: {_old:.1f} → "
                    f"{_sweetener_cap_boxes} (person_weeks={_person_weeks:.1f}; "
                    f"edulcorante 50g dura meses, uso mínimo por porción)"
                )
        # Cap por peso TOTAL (cubre LLM emitting 'g'/'sobre'/'oz')
        _total_weight_g = sum(
            _units.get(u, 0) * _WEIGHT_UNIT_TO_G[u]
            for u in _WEIGHT_UNIT_TO_G
            if u in _units
        )
        if _total_weight_g > _sweetener_cap_g:
            _present_units = {u: _units[u] for u in _WEIGHT_UNIT_TO_G if u in _units}
            for _wu in list(_present_units.keys()):
                del _units[_wu]
            _units['g'] = float(_sweetener_cap_g)
            _record_cap_applied(_name, _total_weight_g, _sweetener_cap_g, "P6-SWEETENER-CAP")
            _cap_log(
                f"[P6-SWEETENER-CAP] '{_name}' peso total cap: {_total_weight_g:.0f}g "
                f"(de {_present_units}) → {_sweetener_cap_g:.0f}g "
                f"(≈{_sweetener_cap_boxes} cajas 50g; "
                f"person_weeks={_person_weeks:.1f})"
            )

    # ============================================================
    # [P6-SAUCE-CAP] Cap defensivo para salsas/condimentos en lata/frasco
    # ------------------------------------------------------------
    # PDF 2026-05-05 21:12: "Salsa de tomate: 11 latas (425g c/u)" = ~4.7 kg
    # para 2p × mes. Salsa de tomate se usa ~30-50g/dish (sofrito, base
    # cocina). LLM emite "1 lata salsa" en cada receta → suma raw × 18.67
    # = 11+ latas. Realmente 1 lata 425g cubre 8-10 platos = >2 semanas.
    #
    # Cap: 1 lata por cada 4 person-weeks. Para 2p × mes (8 pw): cap 2 latas
    # = ~850g (suficiente para uso intensivo ~2-3×/semana). Aplica también
    # a salsas similares (mayonesa, mostaza, ketchup) que tienen mismo
    # patrón: condimento de uso ocasional pero LLM las pide en cada plato.
    # ============================================================
    # [P6-SAUCE-CAP-FIX] Match por SUBSTRING: PDF 2026-05-05 23:33 mostró
    # "Salsa de soya baja en sodio: 10 botellas" pese a tener 'salsa de soya'
    # en el set. Causa: master_map preserva el modificador "baja en sodio"
    # → exact match falla. Estrategia: si name contiene cualquier substring
    # del set, capear. Patrón análogo a P6-OLIVE-CAP-FIX (substring).
    _SAUCE_NAME_SUBSTRINGS = (
        'salsa de tomate', 'pasta de tomate', 'pure de tomate',
        'tomato sauce', 'tomato paste',
        'mayonesa', 'mayonnaise',
        'mostaza', 'mustard',
        'ketchup',
        'salsa inglesa', 'worcestershire',
        'salsa de soya', 'soy sauce', 'salsa soya',
    )
    _SAUCE_LATA_GRAMS = 425.0  # lata estándar dominicana de tomate

    _sauce_cap_latas = max(1, int(round(_person_weeks / 4.0)))
    _sauce_cap_g = _sauce_cap_latas * _SAUCE_LATA_GRAMS

    for _name, _units in list(aggregated.items()):
        _name_norm = strip_accents(_name.lower()).strip()
        if not any(s in _name_norm for s in _SAUCE_NAME_SUBSTRINGS):
            continue
        for _unit_key in ('lata', 'latas', 'frasco', 'frascos', 'botella', 'botellas'):
            if _unit_key in _units and _units[_unit_key] > _sauce_cap_latas:
                _old = _units[_unit_key]
                _units[_unit_key] = float(_sauce_cap_latas)
                _record_cap_applied(_name, _old, _units[_unit_key], "P6-SAUCE-CAP")
                _cap_log(
                    f"[P6-SAUCE-CAP] '{_name}' {_unit_key} cap: {_old:.1f} → "
                    f"{_sauce_cap_latas} (person_weeks={_person_weeks:.1f}; "
                    f"salsas/condimentos de uso ocasional)"
                )
        # Cap por peso TOTAL (cubre 'g'/'oz'/'lb'/'kg'/'ml'/'l'). Mismo
        # patrón que P6-OLIVE-CAP-FIX-3: LLM puede emitir "X oz salsa".
        _total_weight_g = sum(
            _units.get(u, 0) * _WEIGHT_UNIT_TO_G[u]
            for u in _WEIGHT_UNIT_TO_G
            if u in _units
        )
        if _total_weight_g > _sauce_cap_g:
            _present_units = {u: _units[u] for u in _WEIGHT_UNIT_TO_G if u in _units}
            for _wu in list(_present_units.keys()):
                del _units[_wu]
            _units['g'] = float(_sauce_cap_g)
            _record_cap_applied(_name, _total_weight_g, _sauce_cap_g, "P6-SAUCE-CAP")
            _cap_log(
                f"[P6-SAUCE-CAP] '{_name}' peso total cap: {_total_weight_g:.0f}g "
                f"(de {_present_units}) → {_sauce_cap_g:.0f}g "
                f"(≈{_sauce_cap_latas} latas 425g; "
                f"person_weeks={_person_weeks:.1f})"
            )

    # ============================================================
    # [P6-OIL-CAP] Cap defensivo para aceites de cocina
    # ------------------------------------------------------------
    # PDF 2026-05-07 00:30 (plan d119b6b7): "Aceite vegetal: 14 botellas
    # (946ml c/u)" = 13.2 LITROS para 1 persona × 1 mes — absurdo. Una
    # familia de 4 usa ~500ml/mes en uso normal. El bug: P6-OLIVE-CAP
    # cubre aceitunas (encurtidos), NO aceite. Aceite de oliva sale
    # "1 botella (250ml)" naturalmente porque master tiene container_weight=250
    # y el LLM emite poco (cdtas), pero "Aceite vegetal" tiene container 946ml
    # y el LLM emite suficientemente más volumen → SKU resolver multiplica.
    #
    # Causa-raíz del LLM emitting más vegetal oil: lo usa para "freir" /
    # "saltear" en pasos de cocina (~1-2 cdas por receta), mientras aceite
    # de oliva se reserva para finishing/aderezos (~1 cdta).
    #
    # Cap: 1 botella (~946ml estándar dominicano) por cada 4 person-weeks.
    # Equivale a ~1 botella por persona por mes — cubre uso normal con
    # margen. Substring match para capturar todos los variantes:
    #   aceite de oliva, aceite vegetal, aceite de canola, aceite de coco,
    #   aceite de girasol, aceite de maíz, aceite de sésamo, etc.
    # Excluye 'aceitunas' (que YA cubre P6-OLIVE-CAP) y 'aceite de hígado'
    # (suplemento, no cocina) por exact match en exclusión.
    # ============================================================
    _OIL_NAME_SUBSTRINGS = ('aceite',)
    _OIL_NAME_EXCLUDE = (
        'aceitunas', 'aceituna',  # encurtidos — cubierto por P6-OLIVE-CAP
        'aceite de higado', 'aceite de hígado',  # suplemento, no cocina
    )
    _OIL_BOTTLE_GRAMS = 946.0  # botella estándar 32 oz (~946ml) en colmado DR

    _oil_cap_botellas = max(1, int(round(_person_weeks / 4.0)))
    _oil_cap_g = _oil_cap_botellas * _OIL_BOTTLE_GRAMS

    for _name, _units in list(aggregated.items()):
        _name_norm = strip_accents(_name.lower()).strip()
        if not any(s in _name_norm for s in _OIL_NAME_SUBSTRINGS):
            continue
        if any(excl in _name_norm for excl in _OIL_NAME_EXCLUDE):
            continue
        # Cap por unit count (botella/frasco)
        for _unit_key in ('botella', 'botellas', 'frasco', 'frascos'):
            if _unit_key in _units and _units[_unit_key] > _oil_cap_botellas:
                _old = _units[_unit_key]
                _units[_unit_key] = float(_oil_cap_botellas)
                _record_cap_applied(_name, _old, _units[_unit_key], "P6-OIL-CAP")
                _cap_log(
                    f"[P6-OIL-CAP] '{_name}' {_unit_key} cap: {_old:.1f} → "
                    f"{_oil_cap_botellas} (person_weeks={_person_weeks:.1f}; "
                    f"aceite cocina dura ~1 mes/persona por botella 946ml)"
                )
        # Cap por peso TOTAL (cubre 'g'/'oz'/'lb'/'kg'/'ml'/'l').
        _total_weight_g = sum(
            _units.get(u, 0) * _WEIGHT_UNIT_TO_G[u]
            for u in _WEIGHT_UNIT_TO_G
            if u in _units
        )
        if _total_weight_g > _oil_cap_g:
            _present_units = {u: _units[u] for u in _WEIGHT_UNIT_TO_G if u in _units}
            for _wu in list(_present_units.keys()):
                del _units[_wu]
            _units['g'] = float(_oil_cap_g)
            _record_cap_applied(_name, _total_weight_g, _oil_cap_g, "P6-OIL-CAP")
            _cap_log(
                f"[P6-OIL-CAP] '{_name}' peso total cap: {_total_weight_g:.0f}g "
                f"(de {_present_units}) → {_oil_cap_g:.0f}g "
                f"(≈{_oil_cap_botellas} botellas 946ml; "
                f"person_weeks={_person_weeks:.1f})"
            )

    # ============================================================
    # [P6-CARBS-CAP] Cap defensivo para carbos packageados (tortillas, pan)
    # ------------------------------------------------------------
    # PDF 2026-05-05 22:42: "Tortilla integral: 7 paquetes (288g c/u)" =
    # ~2 kg tortillas para 2p × mes — excesivo. Tortillas se usan como
    # vehículo (wrap, burrito) ~2-3×/sem como sustituto de pan.
    # Pan integral típicamente 1 paquete dura ~1 sem para 2p (depende del
    # tamaño). Cap: 1 paquete por 2 person-weeks = 4 paquetes para 2p × mes.
    # ============================================================
    _CARBS_PACKAGE_NAMES_FOR_CAP = {
        'tortilla integral', 'tortillas integrales',
        'tortilla de trigo', 'tortillas de trigo',
        'tortilla de maiz', 'tortillas de maiz',  # strip_accents normalizado
        'pan integral', 'pan de molde', 'pan multigrano',
        'pan de centeno', 'pan blanco', 'pan',
        # [P1-CAP-CANON-DRIFT · 2026-07-28] la expansión del catálogo movió los canónicos a los
        # SKUs "familiar" ('pan integral' resuelve hoy a 'Pan integral familiar') y este set de
        # nombres EXACTOS dejó de matchear: caps de pan muertos en producción sin que nada
        # fallara. Mismo drift que las sardinas ('Sardinas en lata') en el cap de enlatados.
        'pan integral familiar', 'pan blanco familiar', 'pan familiar',
        'pan pita', 'pita integral',
        # [P6-CARBS-CAP-CRACKERS 2026-05-06] Visto en PDF: 9¼ lbs galletas
        # de soda para 2p × mes (~4 lbs/persona) — absurdo. Se usan como
        # snack ligero, no como base. Mismo cap que pan.
        'galletas', 'galletas de soda', 'galletas saladas',
        'galletas integrales', 'crackers',
    }
    # [CAP-RECALIBRATION 2026-05-07] Pan integral master tiene
    # container_weight_g=567 (no 300 asumido). Cap viejo de 1 paq/2pw × 300g
    # daba monthly=600g/persona = ~20g/día = ½ lonja/día (3× menos que el
    # consumo realista de 2-3 lonjas/día = 60g/día = 1800g/mes).
    # Nuevo: container 450g promedio (más cerca a real DR) y formula ×1pw
    # (en vez de /2pw) → monthly = 4 × 450 = 1800g cap, que con SKU 567g
    # se resuelve a 3 paq (1701g, ~57g/día = 2 lonjas, realista).
    # Knob MEALFIT_CARBS_CAP_GRAMS_PER_PW: gramos por person-week. Default 450
    # (real-world). Operador puede subir a 600 para usuarios pan-heavy o
    # bajar a 300 para reducir desperdicio.
    # [P2-1 · 2026-05-08] `_knob_env_float` registra en `_KNOBS_REGISTRY`.
    _CARBS_PACKAGE_GRAMS = max(150.0, _knob_env_float("MEALFIT_CARBS_CAP_GRAMS_PER_PW", 450.0))

    _carbs_cap_packages = max(1, int(round(_person_weeks)))
    _carbs_cap_g = _carbs_cap_packages * _CARBS_PACKAGE_GRAMS

    for _name, _units in list(aggregated.items()):
        _name_norm = strip_accents(_name.lower()).strip()
        if _name_norm not in _CARBS_PACKAGE_NAMES_FOR_CAP:
            continue
        for _unit_key in ('paquete', 'paquetes', 'bolsa', 'bolsas'):
            if _unit_key in _units and _units[_unit_key] > _carbs_cap_packages:
                _old = _units[_unit_key]
                _units[_unit_key] = float(_carbs_cap_packages)
                _record_cap_applied(_name, _old, _units[_unit_key], "P6-CARBS-CAP")
                _cap_log(
                    f"[P6-CARBS-CAP] '{_name}' {_unit_key} cap: {_old:.1f} → "
                    f"{_carbs_cap_packages} (person_weeks={_person_weeks:.1f}; "
                    f"carbos packageados con shelf-life moderada)"
                )
        # Cap por peso TOTAL (cubre 'g'/'oz'/'lb'/'kg' del LLM).
        _total_weight_g = sum(
            _units.get(u, 0) * _WEIGHT_UNIT_TO_G[u]
            for u in _WEIGHT_UNIT_TO_G
            if u in _units
        )
        if _total_weight_g > _carbs_cap_g:
            _present_units = {u: _units[u] for u in _WEIGHT_UNIT_TO_G if u in _units}
            for _wu in list(_present_units.keys()):
                del _units[_wu]
            _units['g'] = float(_carbs_cap_g)
            _record_cap_applied(_name, _total_weight_g, _carbs_cap_g, "P6-CARBS-CAP")
            _cap_log(
                f"[P6-CARBS-CAP] '{_name}' peso total cap: {_total_weight_g:.0f}g "
                f"(de {_present_units}) → {_carbs_cap_g:.0f}g "
                f"(≈{_carbs_cap_packages} paquetes {int(_CARBS_PACKAGE_GRAMS)}g; "
                f"person_weeks={_person_weeks:.1f})"
            )

    # ============================================================
    # [P6-LEGUMES-DRY-CAP] Cap defensivo para legumbres secas (paquetes 1lb)
    # ------------------------------------------------------------
    # PDF 2026-05-06 03:14: "Habichuelas rojas: 6 paquetes (1 lb c/u)" para
    # 2p × mes — 3 lbs/persona/mes es alto. Las legumbres secas no se
    # arruinan (duran años en despensa) pero el LLM tiende a pedirlas como
    # proteína vegetal en múltiples comidas → se acumulan al ×18.66 del
    # eff_mult mensual sin cap.
    #
    # Uso típico: legumbres como base proteica 2-3×/semana (1 plato familiar
    # rinde ~300-400g cocidos = ~120-150g secos por persona). Para 2p:
    #   - 1 paquete 1lb (453g) rinde ~1.5kg cocidas → 4-5 platos para 2p.
    #   - Cap razonable: 1 paquete por 2 person-weeks.
    # Para 2p × mes (8 person-weeks): cap 4 paquetes (vs 6 PDF; reducción 33%).
    # ============================================================
    # Substrings (no equality) porque el LLM emite "habichuelas rojas secas",
    # "frijoles negros cocidos", "lentejas rojas peladas", etc. Substring match
    # es más robusto que equality contra cada combinación. Los modificadores
    # ('secas','cocidas','peladas','rojas','negras', etc.) no cambian el cap:
    # legumbres → 1 paquete/2p×sem indistintamente.
    _LEGUMES_DRY_SUBSTRINGS_FOR_CAP = (
        'habichuela',   # habichuelas rojas/blancas/negras/pintas
        'frijol',       # frijoles rojos/negros/blancos
        # [P1-COUNTRY-CAPS-DO-LEXICON · 2026-08-23] Misma familia, nombre ES.
        # Se queda en la tabla ÚNICA de legumbres: no nace una variante por país.
        'judia',        # judías blancas/pintas
        'gandules',
        'lentejas',
        'garbanzos',
    )
    _LEGUMES_PACKAGE_GRAMS = 453.592  # 1 lb estándar mercado dominicano

    # [CAP-RECALIBRATION 2026-05-07] Cap viejo de 1 paq/2pw daba monthly =
    # 2 paq (907g) para 1 persona = 30g/día raw → ~90g cocido/día. Realista
    # para "legumbres ocasionales" pero corto si el planner las elige como
    # proteína principal del día (caso real cuando el goal es gain_muscle
    # plant-based o cuando habichuelas es uno de los 3 chosen_proteins).
    # En ese caso, una persona come ~200g cocido/día (1 taza) = ~70g raw/día
    # = ~1.96 kg/mes raw — bien sobre el cap viejo de 907g.
    # Nuevo: 1 paq por person-week (en vez de /2pw). Monthly = 4 paq (1816g)
    # = 60g/día raw → 180g cocido/día. Realistic para legume-heavy diet.
    # Knob MEALFIT_LEGUMES_PACKS_PER_PW: paquetes por person-week (default 1.0).
    # [P2-1 · 2026-05-08] `_knob_env_float` registra en `_KNOBS_REGISTRY`.
    _legumes_packs_per_pw = max(0.25, _knob_env_float("MEALFIT_LEGUMES_PACKS_PER_PW", 1.0))
    _legumes_cap_packages = max(1, int(round(_person_weeks * _legumes_packs_per_pw)))
    _legumes_cap_g = _legumes_cap_packages * _LEGUMES_PACKAGE_GRAMS

    for _name, _units in list(aggregated.items()):
        _name_norm = strip_accents(_name.lower()).strip()
        if not any(sub in _name_norm for sub in _LEGUMES_DRY_SUBSTRINGS_FOR_CAP):
            continue
        for _unit_key in ('paquete', 'paquetes', 'bolsa', 'bolsas', 'lata', 'latas'):
            if _unit_key in _units and _units[_unit_key] > _legumes_cap_packages:
                _old = _units[_unit_key]
                _units[_unit_key] = float(_legumes_cap_packages)
                _record_cap_applied(_name, _old, _units[_unit_key], "P6-LEGUMES-DRY-CAP")
                _cap_log(
                    f"[P6-LEGUMES-DRY-CAP] '{_name}' {_unit_key} cap: {_old:.1f} → "
                    f"{_legumes_cap_packages} (person_weeks={_person_weeks:.1f}; "
                    f"~1 paquete cocido rinde 4-5 platos para 2p)"
                )
        # Cap por peso TOTAL (cubre 'g'/'oz'/'lb'/'kg' del LLM).
        _total_weight_g = sum(
            _units.get(u, 0) * _WEIGHT_UNIT_TO_G[u]
            for u in _WEIGHT_UNIT_TO_G
            if u in _units
        )
        if _total_weight_g > _legumes_cap_g:
            _present_units = {u: _units[u] for u in _WEIGHT_UNIT_TO_G if u in _units}
            for _wu in list(_present_units.keys()):
                del _units[_wu]
            _units['g'] = float(_legumes_cap_g)
            _cap_log(
                f"[P6-LEGUMES-DRY-CAP] '{_name}' peso total cap: {_total_weight_g:.0f}g "
                f"(de {_present_units}) → {_legumes_cap_g:.0f}g "
                f"(≈{_legumes_cap_packages} paquetes 1lb; "
                f"person_weeks={_person_weeks:.1f})"
            )
            _record_cap_applied(_name, _total_weight_g, _legumes_cap_g, "P6-LEGUMES-DRY-CAP")

    # ============================================================
    # [P6-CANNED-PROTEIN-CAP] Cap defensivo para proteínas en lata
    # ------------------------------------------------------------
    # PDF 2026-05-05 22:42: "Atún en agua: 19 latas (184g c/u)" = ~3.5 kg
    # de atún para 2p × mes. Atún en lata se conserva mucho pero el LLM
    # tiende a pedirlo en cada comida proteica como fallback fácil.
    # 19 latas = ~3 latas/sem/persona — alto.
    # Cap: 1 lata / persona / semana = 8 latas para 2p × mes.
    # ============================================================
    _CANNED_PROTEIN_NAMES_FOR_CAP = {
        'atun', 'atun en agua', 'atun en aceite',  # strip_accents
        'sardinas', 'sardina',
        # [P1-CAP-CANON-DRIFT · 2026-07-28] la resolución canónica da 'Sardinas en lata' desde la
        # expansión del catálogo — el nombre suelto ya no matchea y las latas de sardina iban SIN
        # cap en producción. Mismo drift que el pan "familiar" en P6-CARBS-CAP.
        'sardinas en lata', 'sardina en lata',
        'salmon en lata', 'salmon enlatado',
        'pollo en lata', 'pollo enlatado',
    }
    _CANNED_PROTEIN_GRAMS = 184.0  # lata estándar atún

    _canned_cap_latas = max(2, int(round(_person_weeks)))
    _canned_cap_g = _canned_cap_latas * _CANNED_PROTEIN_GRAMS

    for _name, _units in list(aggregated.items()):
        _name_norm = strip_accents(_name.lower()).strip()
        if _name_norm not in _CANNED_PROTEIN_NAMES_FOR_CAP:
            continue
        for _unit_key in ('lata', 'latas'):
            if _unit_key in _units and _units[_unit_key] > _canned_cap_latas:
                _old = _units[_unit_key]
                _units[_unit_key] = float(_canned_cap_latas)
                _cap_log(
                    f"[P6-CANNED-PROTEIN-CAP] '{_name}' {_unit_key} cap: {_old:.0f} → "
                    f"{_canned_cap_latas} (person_weeks={_person_weeks:.1f}; "
                    f"~1 lata/persona/sem es uso intensivo realista)"
                )
                # [P2-CAPS-COHERENCE-RECONCILE-2 · 2026-05-30] Registrar el cap
                # (atún/sardinas mencionados por receta) para no disparar retries
                # falsos del coherence guard en mode=block.
                _record_cap_applied(_name, _old, _units[_unit_key], "P6-CANNED-PROTEIN-CAP")
        _total_weight_g = sum(
            _units.get(u, 0) * _WEIGHT_UNIT_TO_G[u]
            for u in _WEIGHT_UNIT_TO_G
            if u in _units
        )
        if _total_weight_g > _canned_cap_g:
            _present_units = {u: _units[u] for u in _WEIGHT_UNIT_TO_G if u in _units}
            for _wu in list(_present_units.keys()):
                del _units[_wu]
            _units['g'] = float(_canned_cap_g)
            _record_cap_applied(_name, _total_weight_g, _canned_cap_g, "P6-CANNED-PROTEIN-CAP")
            _cap_log(
                f"[P6-CANNED-PROTEIN-CAP] '{_name}' peso total cap: {_total_weight_g:.0f}g "
                f"(de {_present_units}) → {_canned_cap_g:.0f}g "
                f"(≈{_canned_cap_latas} latas 184g; "
                f"person_weeks={_person_weeks:.1f})"
            )

    # ============================================================
    # [P6-EGGS-AGGREGATE-CAP] Cap defensivo para huevos en lista de compras
    # ------------------------------------------------------------
    # P6-EGGS-CAP (en day_generator prompt) reduce las RECETAS a
    # ~3 enteros + ~6 claras por día. El reviewer médico acepta esto
    # porque mide RECETAS, no shopping list.
    #
    # PERO el aggregator suma claras + enteros como huevos comprables
    # (1 clara = 1 huevo, hay que comprar el huevo entero para sacar la
    # clara). Resultado real (PDF 2026-05-05 14:35): 5 enteros + 10.5
    # claras = 15.5 huevos / 3 días × multiplier 18.67 → 290 huevos
    # → 11 cartones para 2p × mes. ~5.5 huevos/persona/día = visualmente
    # excesivo aunque las recetas estén dentro del cap del reviewer.
    #
    # Realismo de uso: la mayoría de usuarios NO descarta yemas — cuando
    # compran 4 huevos para usar 4 claras, las 4 yemas se incorporan en
    # otras comidas (revoltillo, mayonesa, repostería). Comprar 11
    # cartones para usar 8 reales = ~$300 RD$ desperdiciados.
    #
    # Cap: derivado del knob `MEALFIT_EGGS_PER_PERSON_PER_DAY` (default 2).
    # Antes el cap era `max(2, round(person_weeks))` cartones, fórmula que
    # daba ~4 huevos/persona/día (2p mensual = 8 cartones = 240 huevos).
    # Reportado por usuario 2026-05-06: para 1p × mes generaba 4 cartones
    # (120 huevos = 4/día/persona) — alto aún para `gain_muscle`.
    # Nuevo cap (default 2/día/persona):
    #   - 1p mensual  (4 pw) → 2 × 4 × 7 = 56 huevos → 2 cartones
    #   - 2p mensual  (8 pw) → 2 × 8 × 7 = 112 huevos → 4 cartones
    #   - 2p semanal  (2 pw) → 2 × 2 × 7 = 28 huevos → 1 cartón (clamp a 2 mín)
    # Si el caso de uso es body-builder pesado: `MEALFIT_EGGS_PER_PERSON_PER_DAY=4`
    # restaura el comportamiento anterior. Knob facilita reversión sin redeploy.
    # ============================================================
    _EGGS_NAMES_FOR_CAP = {'huevo', 'huevos'}
    _EGG_DENSITY_G = 50.0  # UNIT_WEIGHTS["huevo"]
    _HUEVOS_PER_CARTON = 30
    # [P2-1 · 2026-05-08] `_knob_env_float` registra en `_KNOBS_REGISTRY`.
    _EGGS_PER_PERSON_PER_DAY = max(0.5, _knob_env_float("MEALFIT_EGGS_PER_PERSON_PER_DAY", 2.0))

    _eggs_cap_units = max(
        _HUEVOS_PER_CARTON,  # mínimo 1 cartón aunque pw sea bajo
        int(round(_EGGS_PER_PERSON_PER_DAY * _person_weeks * 7.0)),
    )
    _eggs_cap_cartones = max(2, math.ceil(_eggs_cap_units / _HUEVOS_PER_CARTON))
    _eggs_cap_g = _eggs_cap_units * _EGG_DENSITY_G

    for _name, _units in list(aggregated.items()):
        if strip_accents(_name.lower()).strip() not in _EGGS_NAMES_FOR_CAP:
            continue
        # [P6-EGGS-AGGREGATE-CAP-FIX] Splits unidades vs cartones porque
        # tienen DIFERENTES thresholds. Bug previo (PDF 2026-05-05 15:34
        # mostró 22 cartones uncapped): el loop comparaba 'cartón' value
        # contra el threshold-de-unidades (240). Para `units['cartón']=22`,
        # el chequeo `22 > 240` era False → no cap. Resultado: 22 cartones
        # × 30 = 660 huevos llegaban al usuario.
        #
        # Ahora 2 loops separados con threshold correcto en cada unidad.
        for _unit_key in ('unidad', 'unidades'):
            if _unit_key in _units and _units[_unit_key] > _eggs_cap_units:
                _old = _units[_unit_key]
                _units[_unit_key] = float(_eggs_cap_units)
                _record_cap_applied(_name, _old, _units[_unit_key], "P6-EGGS-AGGREGATE-CAP")
                _cap_log(
                    f"[P6-EGGS-AGGREGATE-CAP] '{_name}' {_unit_key} cap: "
                    f"{_old:.0f} → {_eggs_cap_units} (≈{_eggs_cap_cartones} "
                    f"cartones × {_HUEVOS_PER_CARTON} huevos; "
                    f"person_weeks={_person_weeks:.1f}; cap previene "
                    f"sobre-compra de claras + enteros sumados)"
                )
        # [P6-EGGS-AGGREGATE-CAP-FIX-2] Cartón keys con suffix de tamaño.
        # El bloque huevo-específico (línea ~2021) crea keys con sufijo:
        # 'cartón (30 uds.)', 'cartón (6 uds.)', 'medio cartón (15 uds.)'.
        # Antes mi cap solo matched ('cartón','carton','cartones') exactos
        # → no detectaba estos keys con suffix. Resultado: PDF mostraba
        # 22 cartones uncapped en corrida 2026-05-05 15:34.
        # Ahora detectamos cualquier key con 'cartón'/'carton' substring
        # y parseamos el tamaño del suffix '(N uds.)' para calcular cap
        # equivalente (8 cartones × 30 huevos = 240; pero 16 medios × 15
        # huevos = 240 también).
        for _unit_key in list(_units.keys()):
            if not isinstance(_unit_key, str):
                continue
            k_lower = _unit_key.lower()
            if 'cartón' not in k_lower and 'carton' not in k_lower:
                continue
            # Extract huevos-per-unit del suffix si está presente
            _suffix_match = re.search(r'\((\d+)\s*uds?\.?\)', k_lower)
            _huevos_per_unit = int(_suffix_match.group(1)) if _suffix_match else _HUEVOS_PER_CARTON
            _cap_for_this_size = max(1, math.ceil(_eggs_cap_units / _huevos_per_unit))
            if _units[_unit_key] > _cap_for_this_size:
                _old = _units[_unit_key]
                _units[_unit_key] = float(_cap_for_this_size)
                _cap_log(
                    f"[P6-EGGS-AGGREGATE-CAP] '{_name}' {_unit_key!r} cap: "
                    f"{_old:.0f} → {_cap_for_this_size} (≈{_eggs_cap_units} "
                    f"huevos / {_huevos_per_unit} huevos por unit; "
                    f"person_weeks={_person_weeks:.1f})"
                )
                _record_cap_applied(_name, float(_old), float(_cap_for_this_size), "P6-EGGS-AGGREGATE-CAP")
        # Cap por gramos: si el aggregator convirtió 'unidad' → 'g' via
        # density (BLOQUE 2), también capear ahí. 50g/huevo es estándar.
        if 'g' in _units and _units['g'] > _eggs_cap_g:
            _old_g = _units['g']
            _units['g'] = float(_eggs_cap_g)
            _cap_log(
                f"[P6-EGGS-AGGREGATE-CAP] '{_name}' peso cap: {_old_g:.0f}g "
                f"→ {_eggs_cap_g:.0f}g (≈{_eggs_cap_cartones} cartones)"
            )
            _record_cap_applied(_name, _old_g, _units['g'], "P6-EGGS-AGGREGATE-CAP")

    # ============================================================
    # [P6-FRUITS-LARGE-CAP] Cap defensivo para frutas grandes
    # ------------------------------------------------------------
    # Frutas grandes (melón, sandía, piña, lechosa, papaya) producen
    # múltiples servings por unidad PERO no se almacenan más de
    # 5-7 días refrigeradas enteras. PDF real (2026-05-05 15:09):
    # 24 melones para 2p × mes = 35 kg de melón = ~80% se descompone
    # antes de consumir.
    #
    # Causa: aggregator suma "1 taza de melón en cubos" × N comidas
    # × multiplier 18.67 sin entender que cada melón rinde 6-8 tazas.
    #
    # Cap por persona-semana (calibrado por densidad/rendimiento típico):
    #   - melón ~1.2kg → 6-8 servings → 1/persona/sem
    #   - sandía ~3kg → 15-20 servings → 0.5/persona/sem (más rendimiento)
    #   - piña ~1.5kg → 8-10 servings → 1/persona/sem
    #   - lechosa/papaya ~800g → 4-5 servings → 1/persona/sem
    #
    # Para 2p × mes (8 person_weeks):
    #   - melón: 8 unidades (vs 24 PDF; reducción 67%)
    #   - sandía: 4 unidades
    #   - piña/lechosa: 8 unidades
    # ============================================================
    _FRUITS_LARGE_PER_WEEK_PER_PERSON = {
        # melón: 1/persona/sem. UNIT_WEIGHTS["melon"]=1200g.
        'melon':   (1, 1200.0),
        'melones': (1, 1200.0),
        # sandía: 0.5/persona/sem (rinde 15-20 servings).
        # UNIT_WEIGHTS["sandia"]=3000g.
        'sandia':  (0.5, 3000.0),
        'sandias': (0.5, 3000.0),
        # piña: 1/persona/sem. UNIT_WEIGHTS["pina"]=1500g.
        'pina':    (1, 1500.0),
        'pinas':   (1, 1500.0),
        # lechosa/papaya: 1/persona/sem. UNIT_WEIGHTS["lechosa"]=800g.
        'lechosa': (1, 800.0),
        'lechosas': (1, 800.0),
        'papaya':  (1, 800.0),
        'papayas': (1, 800.0),
    }

    for _name, _units in list(aggregated.items()):
        _name_norm = strip_accents(_name.lower()).strip()
        if _name_norm not in _FRUITS_LARGE_PER_WEEK_PER_PERSON:
            continue
        _per_week, _default_density = _FRUITS_LARGE_PER_WEEK_PER_PERSON[_name_norm]
        _fruit_cap_units = max(2, int(round(_per_week * _person_weeks)))

        # Cap unit count
        for _unit_key in ('unidad', 'unidades'):
            if _unit_key in _units and _units[_unit_key] > _fruit_cap_units:
                _old = _units[_unit_key]
                _units[_unit_key] = float(_fruit_cap_units)
                _cap_log(
                    f"[P6-FRUITS-LARGE-CAP] '{_name}' {_unit_key} cap: "
                    f"{_old:.1f} → {_fruit_cap_units} "
                    f"(person_weeks={_person_weeks:.1f}; storage realismo: "
                    f"frutas grandes duran 5-7 días refrigeradas enteras)"
                )
                # [P2-CAPS-COHERENCE-RECONCILE-2 · 2026-05-30] Registrar el cap
                # para que el coherence guard (default block en prod) NO trate
                # esta divergencia de magnitud como crítica y fuerce un retry.
                _record_cap_applied(_name, _old, _units[_unit_key], "P6-FRUITS-LARGE-CAP")

        # Cap por gramos: aplica si el aggregator convirtió 'unidad' → 'g'
        if 'g' in _units:
            _master_item = (
                master_map.get(_name)
                or master_map.get(_name.lower())
                or master_map.get(_name.title())
            )
            _density = _default_density
            if _master_item:
                _master_density = float(_master_item.get('density_g_per_unit') or 0)
                if _master_density > 0:
                    _density = _master_density
            _fruit_cap_g = _fruit_cap_units * _density
            if _units['g'] > _fruit_cap_g:
                _old_g = _units['g']
                _units['g'] = float(_fruit_cap_g)
                _cap_log(
                    f"[P6-FRUITS-LARGE-CAP] '{_name}' peso cap: {_old_g:.0f}g "
                    f"→ {_fruit_cap_g:.0f}g (≈{_fruit_cap_units} unidades a "
                    f"{_density:.0f}g c/u)"
                )
                _record_cap_applied(_name, _old_g, _units['g'], "P6-FRUITS-LARGE-CAP")

    # ============================================================
    # [P6-FRUITS-PERISHABLE-CAP] Cap defensivo para frutas perecederas
    # vendidas por LIBRAS (fresas, arándanos, moras, frambuesas).
    # ------------------------------------------------------------
    # A diferencia de las frutas grandes (melón, sandía) que se compran
    # por unidad, estas se compran por libras/paquetes y son extremadamente
    # perecederas (3-5 días refrigeradas).
    #
    # PDF real (2026-05-05 15:34): "Fresas: 25 paquetes (1 lb c/u)" para
    # 2p × mes = 25 lbs ≈ 11 kg. Comprar 11kg de fresa de una vez =
    # ~80% se descompone antes de consumir. Mismo modo de fallo que melón
    # pre-cap pero diferente unidad de compra.
    #
    # Cap por LIBRAS (no por unidades):
    #   - fresa: 1 lb/persona/sem (ej. smoothie diario ~64g/persona)
    #   - arándanos/moras/frambuesas: 0.5 lb/persona/sem (más caras, menor volumen)
    #
    # Para 2p × mes (8 person_weeks):
    #   - fresa cap: 8 lbs (vs 25 PDF; reducción 68%)
    #   - berries: 4 lbs
    # ============================================================
    _FRUITS_PERISHABLE_LBS_PER_WEEK_PER_PERSON = {
        'fresa':  1.0,
        'fresas': 1.0,
        'arandano':  0.5,  # blueberries
        'arandanos': 0.5,
        'mora':  0.5,      # blackberries
        'moras': 0.5,
        'frambuesa':  0.5,  # raspberries
        'frambuesas': 0.5,
    }
    _LB_TO_G = 453.592
    _PAQUETE_LB_DEFAULT = 1.0  # 1 paquete estándar = 1 lb en RD

    for _name, _units in list(aggregated.items()):
        _name_norm = strip_accents(_name.lower()).strip()
        if _name_norm not in _FRUITS_PERISHABLE_LBS_PER_WEEK_PER_PERSON:
            continue
        _per_week_lbs = _FRUITS_PERISHABLE_LBS_PER_WEEK_PER_PERSON[_name_norm]
        _cap_lbs = max(1.0, float(round(_per_week_lbs * _person_weeks)))
        _cap_g = _cap_lbs * _LB_TO_G

        # Cap por gramos (path principal del aggregator después de
        # convertir lbs/oz a g en el main loop downstream)
        if 'g' in _units and _units['g'] > _cap_g:
            _old_g = _units['g']
            _units['g'] = float(_cap_g)
            _cap_log(
                f"[P6-FRUITS-PERISHABLE-CAP] '{_name}' peso cap: {_old_g:.0f}g "
                f"→ {_cap_g:.0f}g (≈{_cap_lbs:.0f} lbs; "
                f"person_weeks={_person_weeks:.1f}; storage realismo: "
                f"frutas perecederas duran 3-5 días)"
            )
            # [P2-CAPS-COHERENCE-RECONCILE-2 · 2026-05-30] Registrar el cap
            # (perecedero no-staple que SÍ llega al guard) para no disparar
            # retries falsos en mode=block. Registramos en cada rama porque la
            # que dispara depende de la unidad nativa que emitió el LLM (g/lb/paquete).
            _record_cap_applied(_name, _old_g, _units['g'], "P6-FRUITS-PERISHABLE-CAP")
        # Cap por libras (si LLM emitió "X lb de fresas")
        for _unit_key in ('lb', 'lbs', 'libra', 'libras'):
            if _unit_key in _units and _units[_unit_key] > _cap_lbs:
                _old = _units[_unit_key]
                _units[_unit_key] = float(_cap_lbs)
                _cap_log(
                    f"[P6-FRUITS-PERISHABLE-CAP] '{_name}' {_unit_key} cap: "
                    f"{_old:.1f} → {_cap_lbs:.0f} lbs"
                )
                _record_cap_applied(_name, _old, _units[_unit_key], "P6-FRUITS-PERISHABLE-CAP")
        # Cap por paquetes (1 paquete = 1 lb estándar dominicano)
        for _unit_key in ('paquete', 'paquetes'):
            if _unit_key in _units and _units[_unit_key] > _cap_lbs:
                _old = _units[_unit_key]
                _units[_unit_key] = float(_cap_lbs)
                _cap_log(
                    f"[P6-FRUITS-PERISHABLE-CAP] '{_name}' {_unit_key} cap: "
                    f"{_old:.0f} → {_cap_lbs:.0f} paquetes (1 paq ≈ 1 lb)"
                )
                _record_cap_applied(_name, _old, _units[_unit_key], "P6-FRUITS-PERISHABLE-CAP")

    # ============================================================
    # [P6-LACTEOS-PERISHABLE-CAP] Cap defensivo para lácteos perecederos
    # ------------------------------------------------------------
    # Yogurt y otros lácteos abiertos duran ~14 días refrigerados.
    # PDF real (2026-05-05 18:33): "Yogurt griego sin azúcar: 21 potes
    # (16 oz c/u)" para 2p × mes = 9.5 kg. Logísticamente:
    #   - 21 potes no caben en una nevera promedio
    #   - Los últimos potes se acercan al límite de caducidad (28+ días)
    #   - Realistic shopping pattern: re-stock semanal o quincenal
    #
    # Cap: 1.5 lb/persona/sem (≈1 pote 16oz cada 5 días).
    #   - 2p mensual (8 pw) → 12 lbs ≈ 12 potes (vs 21 PDF; reducción 43%)
    #   - 2p quincenal (4 pw) → 6 lbs ≈ 6 potes
    #   - 2p semanal (2 pw) → 3 lbs ≈ 3 potes
    #
    # Match por substring para cubrir variantes ('yogurt griego sin azúcar',
    # 'yogur natural', 'yogurt griego', etc.) sin enumerar manualmente.
    # ============================================================
    _LACTEOS_PERISHABLE_LBS_PER_WEEK_PER_PERSON = {
        'yogurt': 1.5,
        'yogur': 1.5,  # variante sin 't' final
        # [P6-LACTEOS-EXT] Queso ricotta (PDF 2026-05-05 21:50: 6 potes
        # 425g c/u = 2.55 kg). Ricotta es lácteo perecedero similar a
        # yogurt: dura 7-14 días refrigerado tras abrir. Uso típico
        # ~50-100g por dish (relleno, postre, ensalada). 1 lb/persona/sem
        # = uso intensivo (~daily). Para 2p × mes: cap 8 lbs ≈ 8 potes
        # 16oz (vs 6 potes 425g = ~5.6 lbs PDF — está bajo cap, pero
        # entry previene escalada en futuras corridas).
        'ricotta': 1.0,
        # Cottage cheese: similar a ricotta en uso/perishability.
        'cottage': 1.0,
        # [P6-LACTEOS-EXT-2] Queso mozzarella (PDF 2026-05-05 22:42:
        # 5 paquetes 1lb = 5 lbs para 2p × mes). Mozzarella es lácteo
        # perecedero similar a ricotta — abierto dura ~7-14 días.
        # 0.5 lb/persona/sem = uso moderado. Para 2p × mes: cap 4 lbs.
        'mozzarella': 0.5,
        # [P6-LACTEOS-EXT-3] Queso blanco / queso fresco (PDF 2026-05-05
        # 23:12: 9 paquetes 1lb = 9 lbs para 2p × mes — excesivo).
        # Queso blanco (estilo cottage DR) se usa más que mozzarella
        # como acompañante (con casabe, en desayuno, en arepa).
        # 0.75 lb/persona/sem = uso intensivo. Para 2p × mes: cap 6 lbs.
        'queso blanco': 0.75,
        'queso fresco': 0.75,
        # [P6-LACTEOS-EXT-4 2026-05-07] Leche (PDF 2026-05-07 plan 7ab9a552:
        # 3 cartones 946ml = 2.8 LITROS para 1p × sem ≈ 400ml/día). Para
        # alguien que toma leche en café/cereal/batido, ~250ml/día = 1.75
        # L/sem es uso intensivo realista pero 2.8L es excesivo.
        # Cap: 1.75 lb/persona/sem (≈800ml/sem ≈ 1 cartón pequeño 946ml).
        # Para 1p × mes (4pw): 7 lbs ≈ 3.2 L = 3-4 cartones (vs los 12
        # potenciales cartones que el LLM emitiría sin cap).
        # IMPORTANT: 'leche' substring también matchea 'leche en polvo' /
        # 'leche evaporada' / 'leche UHT' que son ESTABLES (long shelf).
        # Pero el cap es ALL-CAUSE: si compras 3L de leche en cualquier
        # forma, está bien. Lácteos son perishable post-apertura, así que
        # cap por volumen total tiene sentido.
        'leche': 1.75,
        # [P1-COUNTRY-CAPS-DO-LEXICON · 2026-08-23] Vocabulario beta de la
        # MISMA familia. Los valores reutilizan las clases de arriba: queso
        # fresco/cremoso, queso semiblando y lácteo líquido. No dependen del
        # país y por eso un gemelo recibe el mismo tope que su fila canónica.
        'requeson': 1.0,            # gemelo nutricional de ricotta
        'cuajada': 0.75,
        'queso de papa': 0.75,
        'queso en hebras': 0.5,
        'queso provolone': 0.5,
        'crema agria': 1.0,
        'crema mexicana': 1.0,
        'crema mitad y mitad': 1.75,
        'natilla': 1.0,             # antes de `nata`: match por substring
        'nata': 1.0,
        'suero costeno': 1.0,
        'suero de mantequilla': 1.75,
    }

    for _name, _units in list(aggregated.items()):
        _name_norm = strip_accents(_name.lower()).strip()
        _matched_key = next(
            (k for k in _LACTEOS_PERISHABLE_LBS_PER_WEEK_PER_PERSON if k in _name_norm),
            None,
        )
        if not _matched_key:
            continue
        _per_week_lbs = _LACTEOS_PERISHABLE_LBS_PER_WEEK_PER_PERSON[_matched_key]
        _cap_lbs = max(1.0, float(round(_per_week_lbs * _person_weeks)))
        _cap_g = _cap_lbs * 453.592

        # [LACTEOS-CAP-FIX 2026-05-07] Cap viejo solo chequeaba 'g' como unit
        # de peso. Bug observable plan 7ab9a552: Leche 3 cartones (2.8L)
        # weekly. La leche se emite en 'ml', que escapa el check de 'g'.
        # Fix: sumar TODAS las unidades de peso/volumen (`_WEIGHT_UNIT_TO_G`
        # ya cubre g/kg/oz/lb/lbs/ml/l) y capear el total — mismo patrón
        # que P6-OIL-CAP / P6-SAUCE-CAP.
        _total_weight_g = sum(
            _units.get(u, 0) * _WEIGHT_UNIT_TO_G[u]
            for u in _WEIGHT_UNIT_TO_G
            if u in _units
        )
        if _total_weight_g > _cap_g:
            _present_units = {u: _units[u] for u in _WEIGHT_UNIT_TO_G if u in _units}
            for _wu in list(_present_units.keys()):
                del _units[_wu]
            _units['g'] = float(_cap_g)
            _record_cap_applied(_name, _total_weight_g, _cap_g, "P6-LACTEOS-PERISHABLE-CAP")
            _cap_log(
                f"[P6-LACTEOS-PERISHABLE-CAP] '{_name}' peso total cap: "
                f"{_total_weight_g:.0f}g (de {_present_units}) → {_cap_g:.0f}g "
                f"(≈{_cap_lbs:.0f} lbs; person_weeks={_person_weeks:.1f}; "
                f"storage realismo: lácteos abiertos duran ~14 días refrigerado)"
            )
        for _unit_key in ('lb', 'lbs', 'libra', 'libras'):
            if _unit_key in _units and _units[_unit_key] > _cap_lbs:
                _old = _units[_unit_key]
                _units[_unit_key] = float(_cap_lbs)
                _record_cap_applied(_name, _old, _units[_unit_key], "P6-LACTEOS-PERISHABLE-CAP")
                _cap_log(
                    f"[P6-LACTEOS-PERISHABLE-CAP] '{_name}' {_unit_key} cap: "
                    f"{_old:.1f} → {_cap_lbs:.0f} lbs"
                )
        # [LACTEOS-CAP-FIX 2026-05-07] Cap por count para potes/cartones —
        # cubre el caso "X cartones de leche" emitido por el LLM como conteo.
        # Conversión aproximada: 1 cartón ≈ 1 lb (16oz ~ 454g leche).
        for _unit_key in ('pote', 'potes', 'carton', 'cartones', 'cartón'):
            if _unit_key in _units and _units[_unit_key] > _cap_lbs:
                _old = _units[_unit_key]
                _units[_unit_key] = float(_cap_lbs)
                _record_cap_applied(_name, _old, _units[_unit_key], "P6-LACTEOS-PERISHABLE-CAP")
                _cap_log(
                    f"[P6-LACTEOS-PERISHABLE-CAP] '{_name}' {_unit_key} cap: "
                    f"{_old:.0f} → {_cap_lbs:.0f} {_unit_key} (1 unidad ≈ 16oz/1 lb)"
                )

    # ============================================================
    # [P6-BROTHS-CAP] Cap defensivo para caldos / stocks
    # ------------------------------------------------------------
    # Caldos se usan como saborizante (cubitos 8-10g, líquido 1L cartón).
    # PDF real (2026-05-05 18:33): "Caldo de vegetales: 3 lbs" para 2p ×
    # mes = ~45g/día = 5+ cubitos/día. Excesivo (1-2 cubitos/día/2p es
    # uso normal). Format en libras también es weird (caldo es líquido o
    # cubitos, no lbs literales).
    #
    # Causa: aggregator suma "1 cda de caldo" × N comidas × multiplier
    # como peso seco. Realmente caldo concentrado: 1 cubito (10g) por
    # receta; o líquido pre-hecho (1L cartón) usado en sopas/guisos.
    #
    # Cap: 0.125 lb/persona/sem (~57g/sem = 5-6 cubitos).
    #   - 2p mensual (8 pw) → 1 lb ≈ 50 cubitos = 1.5 cubitos/día/2p
    #   - 2p quincenal (4 pw) → 0.5 lb (mín)
    #   - 2p semanal (2 pw) → 0.5 lb (mín)
    # Match substring para caldos de vegetales/pollo/res/hueso/marisco.
    # ============================================================
    _BROTHS_LBS_PER_WEEK_PER_PERSON = {
        'caldo': 0.125,
    }

    for _name, _units in list(aggregated.items()):
        _name_norm = strip_accents(_name.lower()).strip()
        _matched_key = next(
            (k for k in _BROTHS_LBS_PER_WEEK_PER_PERSON if k in _name_norm),
            None,
        )
        if not _matched_key:
            continue
        _per_week_lbs = _BROTHS_LBS_PER_WEEK_PER_PERSON[_matched_key]
        # Cap a 0.5 lb mínimo para que el operador siempre tenga al
        # menos 1 cartón/sobre. Round to nearest 0.5 lb para shopping
        # realismo (caldo se vende en aproximaciones medias).
        _cap_lbs = max(0.5, float(round(_per_week_lbs * _person_weeks * 2) / 2))
        _cap_g = _cap_lbs * 453.592

        if 'g' in _units and _units['g'] > _cap_g:
            _old_g = _units['g']
            _units['g'] = float(_cap_g)
            _record_cap_applied(_name, _old_g, _units['g'], "P6-BROTHS-CAP")
            _cap_log(
                f"[P6-BROTHS-CAP] '{_name}' peso cap: {_old_g:.0f}g → "
                f"{_cap_g:.0f}g (≈{_cap_lbs:.1f} lbs ≈ "
                f"{int(_cap_g / 10)} cubitos de 10g; "
                f"person_weeks={_person_weeks:.1f})"
            )
        for _unit_key in ('lb', 'lbs', 'libra', 'libras'):
            if _unit_key in _units and _units[_unit_key] > _cap_lbs:
                _old = _units[_unit_key]
                _units[_unit_key] = float(_cap_lbs)
                _record_cap_applied(_name, _old, _units[_unit_key], "P6-BROTHS-CAP")
                _cap_log(
                    f"[P6-BROTHS-CAP] '{_name}' {_unit_key} cap: "
                    f"{_old:.1f} → {_cap_lbs:.1f} lbs"
                )

    # [2026-05-06 PROTEIN-UNIT-FALLBACK] Fallback portion para proteínas que
    # llegan en unidades sueltas sin peso explícito.
    # ------------------------------------------------------------
    # PDF 2026-05-06 22:53: "Cerdo: 4 Uds." con ⚠ low-confidence.
    # Causa: el LLM emitió "1 chuleta de cerdo" / "1 lonja de cerdo" /
    # "1 unidad de cerdo" en alguna comida (sin peso). `_parse_quantity`
    # no reconoce 'chuleta' como unit canónico → cae a 'unidad' y mueve
    # 'chuleta' al name. master_map resuelve 'chuleta de cerdo' → 'Cerdo'
    # vía alias, pero la unidad ya quedó como 'unidad'. master.Cerdo tiene
    # `default_unit='lb'` y `density_g_per_unit=null` (idéntico para todas
    # las proteínas: pollo, res, pescado, pavo) — no hay fallback en master
    # para convertir unit→peso, así que el aggregator stored 'unidad' as-is.
    # Resultado: "Cerdo: 4 Uds" — semánticamente vacío para el usuario,
    # las proteínas se compran SIEMPRE por peso en supermercado dominicano.
    #
    # Fix: cuando el item es proteína (master.default_unit='lb',
    # master.category='Proteínas') Y SOLO trae count-units sin peso ni
    # densidad nativa, convertir cada unit count → gramos usando un
    # tamaño de porción típico (200g = chuleta/lonja/filete promedio en RD).
    # Knob: MEALFIT_PROTEIN_UNIT_FALLBACK_G.
    # ============================================================
    # [P2-1 · 2026-05-08] `_knob_env_int` registra en `_KNOBS_REGISTRY`.
    _PROTEIN_UNIT_FALLBACK_G = max(50, _knob_env_int("MEALFIT_PROTEIN_UNIT_FALLBACK_G", 200))
    _COUNT_UNITS_FOR_PROTEIN = ('unidad', 'unidades', 'rebanada', 'rebanadas',
                                 'paquete', 'paquetes', 'lonja', 'lonjas')
    _WEIGHT_UNITS_FOR_PROTEIN = ('g', 'kg', 'oz', 'lb', 'lbs', 'libra', 'libras', 'ml', 'l')

    for name, units in aggregated.items():
        master_item = master_map.get(name) or master_map.get(name.lower()) or master_map.get(name.title()) or {}

        # Evitar líquidos comunes/ilimitados en casa
        if _should_ignore_shopping(name):
            continue

        # [P1-SUPERMARKET-COSTING · 2026-07-02] Overlay de la marca preferida:
        # si el usuario eligió marca/presentación para este alimento, su envase
        # (tamaño+precio del súper) REEMPLAZA los market_packages del master —
        # `_select_market_package` comprará N de ESE envase y el costo sale de
        # su precio real. Copia superficial del master_item para no mutar el
        # cache global. Sin preferencia (o knob off) → comportamiento idéntico.
        _pref_pkg = None
        if brand_prefs:
            _pref_pkg = _resolve_brand_pref(name, brand_prefs)
            if _pref_pkg:
                master_item = dict(master_item)
                master_item["market_packages"] = [_pref_pkg]
                # El path por packages exige container+peso — completarlos con
                # los del envase elegido cuando el master no los tenga.
                if not master_item.get("market_container"):
                    master_item["market_container"] = _pref_pkg.get("unit") or "paquete"
                if not master_item.get("container_weight_g"):
                    master_item["container_weight_g"] = _pref_pkg["grams"]
        # [P1-BRAND-LIST-VISIBILITY · 2026-07-06] Sin preferencia manual → los
        # productos REALES del súper (marca + precio vivo) reemplazan el envase
        # genérico. `_select_market_package` (costo-óptimo, P1-PKG-COST-OPTIMAL)
        # elige la marca/tamaño MÁS BARATO para la necesidad del ciclo y su label
        # "2 lb · La Garza" fluye a display_qty → la lista y el PDF enseñan la
        # marca. La preferencia del usuario (bloque de arriba) SIEMPRE gana.
        if _pref_pkg is None and brand_defaults:
            # [P1-BRAND-DEFAULT-GUARDS] resolución conservadora (exacto/singular/
            # food⊆nombre + guards hierbas/modificadores) — NO la escalera de prefs.
            # Gate adicional (verificado vs plan vivo ff673061): el default SOLO
            # aplica a ítems cuyo master YA se vende en ENVASE. El mostrador
            # fresco/deli fraccional conserva sus paths nativos — sin este gate,
            # Zanahoria ½ lb RD$14 → "funda Baby 12 Oz" RD$175, Cebolla 1¼ lb
            # RD$59 → "3 mallas Perla Roja" RD$750, Queso blanco ¼ lb RD$68 →
            # "paquete 1 lb" RD$270 (la variante empacada del catálogo es
            # specialty, no el producto que el plan costea a granel).
            _mi_has_container = bool(master_item.get("market_container")) or (
                (str(master_item.get("default_unit") or "").strip().lower() in _CONTAINER_UNIT_ALIASES)
                and bool(master_item.get("container_weight_g"))
            )
            _def_pkgs = _resolve_brand_default(name, brand_defaults) if _mi_has_container else None
            if _def_pkgs and isinstance(_def_pkgs, list):
                master_item = dict(master_item)
                master_item["market_packages"] = list(_def_pkgs)
                if not master_item.get("market_container"):
                    master_item["market_container"] = _def_pkgs[0].get("unit") or "paquete"
                if not master_item.get("container_weight_g"):
                    master_item["container_weight_g"] = _def_pkgs[0]["grams"]

        weight_in_lbs = 0.0
        has_weight = False
        cat = master_item.get("category") or "Otros"
        display_cat = _get_display_category(cat, name)

        price_per_lb = float(master_item.get("price_per_lb", 0) or 0)
        price_per_unit = float(master_item.get("price_per_unit", 0) or 0)

        # [P3-VERIFIED-INGREDIENTS-ONLY · 2026-06-20] Solo alimentos verificados con
        # precio La Sirena (los ~202 verificados de master_ingredients (era 119 pre-expansion 2026-06-26)) pueden aparecer en la lista.
        # Un ingrediente inventado por el LLM (laurel, comino, cúrcuma...) que NO resuelve
        # a master con precio se EXCLUYE. El espejo en run_shopping_coherence_guard filtra
        # expected_raw con la MISMA `_is_verified_for_shopping`, así que este drop es un
        # SUBCONJUNTO del filtro esperado → cero divergencias `expected_only` → cero retry.
        if (_verified_ingredients_only_enabled()
                and not _is_verified_for_shopping(name)
                and price_per_lb <= 0 and price_per_unit <= 0):
            # [P1-BAKING-STAPLES · 2026-07-01] Staple de horneado (polvo de hornear/levadura/bicarbonato/
            # vainilla) → NO dropear: listar como ~1 empaque sin precio bajo "DESPENSA BÁSICA" para que la
            # receta insignia (panqueques/bollos) sea comprable tal cual. Cae al procesamiento normal.
            if _baking_staples_keep_enabled() and is_baking_pantry_staple(name):
                weight_in_lbs = _BAKING_STAPLE_DEFAULT_G / 453.592
                has_weight = True
                units = {}
                display_cat = "DESPENSA BÁSICA"
                logging.info(
                    f"🧁 [P1-BAKING-STAPLES] '{name}' fuera del catálogo con precio pero es staple de "
                    f"horneado → listado como ~1 empaque (~{_BAKING_STAPLE_DEFAULT_G:.0f}g, sin precio) "
                    f"en DESPENSA BÁSICA en vez de dropearlo."
                )
            elif _country_catalog_unpriced_keep_enabled() and is_country_catalog_unpriced_item(name):
                # [P1-COUNTRY-SYSTEM-F2 · T5 · 2026-08-17] generalización de P1-BAKING-STAPLES —
                # ver docstring de `is_country_catalog_unpriced_item`.
                # [P1-COUNTRY-KEEP-RESPECT-QTY · 2026-08-21] El `units = {}` de esta rama tiraba al
                # suelo la demanda REAL de las recetas: los 7 ítems de catálogo-país de los 2
                # planes beta vivos salían a 150,0 g exactos («¼ lb») para recetas que pedían 653 g
                # de almejas, 504 g de acelgas o 443 g de membrillo — y en 4 de los 7 sin siquiera
                # la nota de cobertura, porque el déficit en tazas/cucharadas no se puede calcular
                # en gramos: sub-compra MUDA. El default se diseñó para el caso «al gusto / sin
                # cantidad» y acabó ganando siempre; aquí se invierte la precedencia y queda como
                # último recurso. La rama de horneado de arriba NO cambia: 100 g de polvo de
                # hornear ES la respuesta correcta a «1 cdta» porque ahí se compra el ENVASE.
                _ccu_has_qty = _country_keep_has_recipe_qty(units)
                if not _ccu_has_qty:
                    weight_in_lbs = _COUNTRY_CATALOG_UNPRICED_DEFAULT_G / 453.592
                    has_weight = True
                    units = {}
                # [P2-SHOPLIST-BETA-POLISH · 2026-08-18] pasillo REAL del súper (Vegetales/
                # Frutas/...) en vez del label interno 'CATÁLOGO SIN PRECIO' que se filtraba
                # al PDF — ver docstring de `_master_category_for_unpriced_item`. Fallback al
                # label histórico si el master no resuelve.
                display_cat = _master_category_for_unpriced_item(name) or "CATÁLOGO SIN PRECIO"
                # [P3-COUNTRY-KEEP-LOG-VOLUME · 2026-08-21] a DEBUG: eran 3 líneas por ítem por
                # cada recálculo de lista, en bucle sobre las 141 filas.
                logging.debug(
                    "🌍 [P1-COUNTRY-CATALOG-UNPRICED] '%s' fuera del catálogo con precio pero es "
                    "alimento de catálogo-país (sin precio RD a propósito, país beta) → listado en "
                    "'%s' en vez de dropearlo (cantidad: %s).",
                    name, display_cat,
                    "de la receta" if _ccu_has_qty else f"~{_COUNTRY_CATALOG_UNPRICED_DEFAULT_G:.0f}g por defecto",
                )
            else:
                # [P1-VERIFIED-ONLY-OBSERVABILITY · 2026-06-21] WARNING (no info) para que el
                # drop sea grep-able en prod: este es el punto exacto donde un ingrediente de
                # receta fuera del catalogo verificado desaparece de la lista. Espejo del guard (P1-VERIFIED-ONLY-OBSERVABILITY).
                # [P2-OFF-CATALOG-SNAP-RESOLVED · 2026-06-29] (re-audit objetivo · P2 F4) El "snap fuzzy al master más
                # cercano" que un audit podría proponer YA ocurrió: `_is_verified_for_shopping(name)` → `normalize_name`
                # aplica regex + FUZZY difflib (INTENTO 5, ratio≥0.87) + embedding ANTES de devolver False. Si llegamos
                # aquí, NINGÚN tier resolvió el nombre a un master verificado → es off-catálogo GENUINO (garble/alimento
                # no vendido), y dropear es lo CORRECTO (costear un fantasma con price=0 descuadraría el total). La
                # defensa primaria es upstream (P2-VERIFIED-ONLY-UPDATE inyecta el catálogo al swap/chat-modify). NO
                # re-implementar un snap aquí: arriesgaría mis-snap en un subsistema de coherencia. tooltip-anchor: P2-OFF-CATALOG-SNAP-RESOLVED
                logging.warning(
                    "[VERIFIED-ONLY-DROP] '%s' excluido de la lista: fuera del catálogo "
                    "verificado (ni regex/fuzzy/embedding lo resolvieron → off-catálogo genuino). "
                    "Si es sustantivo, la lista de compras queda incompleta (defensa primaria: prompt upstream).",
                    name,
                )
                # [P2-VERIFIED-DROP-TELEMETRY · 2026-07-01] (audit v2 creatividad GAP-5, batch P2-AUDIT-V2-BATCH)
                # sink estructurado además del WARN grep-able: el cron `_creativity_kpi_job` agrega el top-N de
                # drops a pipeline_metrics → el owner decide synonyms/altas de catálogo con datos, no greps.
                record_verified_only_drop(name)
                continue

        # [PROTEIN-UNIT-FALLBACK] Aplica ANTES de la extracción de peso.
        # Solo convierte si: (a) cat=Proteínas, (b) master.default_unit='lb',
        # (c) NO hay density_g_per_unit nativo en master, (d) NO hay unidades
        # de peso en `units` (no queremos doblar contar items que el LLM
        # emitió en ambas formas), (e) hay al menos 1 count-unit relevante.
        _is_protein = strip_accents(str(cat).lower()).strip() in ('proteinas', 'proteína', 'proteinas')
        _master_default_unit = str(master_item.get("default_unit") or "").lower().strip()
        _has_master_density = bool(master_item.get("density_g_per_unit"))
        _has_weight_unit = any(_wu in units for _wu in _WEIGHT_UNITS_FOR_PROTEIN)
        if (_is_protein and _master_default_unit == 'lb'
                and not _has_master_density and not _has_weight_unit):
            _converted_g = 0.0
            _converted_from = []
            for _cu in _COUNT_UNITS_FOR_PROTEIN:
                if _cu in units and units[_cu] > 0.0001:
                    _converted_g += units[_cu] * _PROTEIN_UNIT_FALLBACK_G
                    _converted_from.append(f"{units[_cu]:.1f}{_cu}")
                    del units[_cu]
            if _converted_g > 0:
                units['g'] = units.get('g', 0.0) + _converted_g
                logging.info(
                    f"[PROTEIN-UNIT-FALLBACK] '{name}': "
                    f"{','.join(_converted_from)} → {_converted_g:.0f}g "
                    f"(porción default={_PROTEIN_UNIT_FALLBACK_G}g/ud; "
                    f"master.density vacía → no se podía convertir nativamente)"
                )

        if 'g' in units:
            weight_in_lbs += units['g'] / 453.592
            has_weight = True
            del units['g']
        if 'kg' in units:
            weight_in_lbs += units['kg'] * 2.20462
            has_weight = True
            del units['kg']
        if 'oz' in units:
            weight_in_lbs += units['oz'] / 16.0
            has_weight = True
            del units['oz']
        if 'lb' in units:
            weight_in_lbs += units['lb']
            has_weight = True
            del units['lb']
        # Líquidos: ml ≈ gramos (densidad ≈ 1 para leche, jugos, aceites)
        # Esto permite que 450ml de leche → peso → Bloque 1 → "1 Cartón"
        if 'ml' in units:
            weight_in_lbs += units['ml'] / 453.592  # 1ml ≈ 1g
            has_weight = True
            del units['ml']
        if 'l' in units:
            weight_in_lbs += (units['l'] * 1000) / 453.592
            has_weight = True
            del units['l']

        # [P0-11] Clamp defensivo: si `consumed > plan` en peso (todas las
        # unidades de peso suman a un net negativo), `weight_in_lbs` queda
        # negativo. La línea `if weight_in_lbs > 0.0001` más abajo evita que
        # se agregue una entrada por peso (correcto), PERO el for sobre
        # `units` que sigue puede agregar una entrada residual por unidad
        # ("1 Ud.") aunque el peso planificado ya esté cubierto al 100%
        # por el consumed. Resultado: "fantasma" en la lista de compras
        # del usuario.
        #
        # Fix: clampear a 0 y vaciar `units`. La consumed cubrió todo el
        # aporte planificado para este ingrediente — no hay nada que
        # comprar. Si el LLM expresó el mismo aporte como peso + unidad
        # (caso clásico: "1 cebolla mediana" + "200g cebolla"), ambas
        # representaciones quedan suprimidas simétricamente.
        if has_weight and weight_in_lbs < 0:
            logging.info(
                f"[P0-11/CLAMP] {name}: weight_in_lbs={weight_in_lbs:.4f} "
                f"(consumed cubrió/excedió plan). Clamp a 0 + reset units "
                f"residuales={list(units.keys())} para evitar entrada fantasma."
            )
            weight_in_lbs = 0.0
            units = {}

        added = False
        
        # DEDUP: Si el ingrediente tiene cantidades reales (peso ó unidades concretas),
        # eliminar las entradas nominales (pizca, al gusto) porque son redundantes.
        nominal_units = {'pizca', 'al gusto', 'cantidad necesaria', 'chin', 'toque', 'chorrito'}
        has_real_qty = has_weight or any(
            u not in nominal_units and q > 0.0001 
            for u, q in list(units.items())
        )
        if has_real_qty:
            # Tiene cantidades reales → borrar las nominales redundantes
            for nom_u in list(units.keys()):
                if nom_u in nominal_units:
                    del units[nom_u]
        
        # Si SOLO quedan nominales (pizca, al gusto) y no hay peso → saltar ingrediente
        # No aporta a una lista de compras real
        remaining_real = any(u not in nominal_units for u in units) or has_weight
        if not remaining_real:
            # [P2-SEASONING-CATALOG-KEEP · 2026-06-22] Si el ingrediente RESUELVE al catálogo
            # verificado (cilantro, orégano dominicano, etc.) pero el LLM lo emitió SOLO en cantidad
            # nominal (pizca/al gusto, sin peso), NO lo dropees: la receta lo usa y es un alimento
            # comprable. Le asignamos el peso de 1 empaque (container_weight_g → density → default) y
            # dejamos que caiga al path normal de abajo → apply_smart_market_units lo lista como
            # "1 frasco/mazo". Cierra la lista-incompleta para sazones de catálogo (caso en vivo
            # 2026-06-22). Los NO-catálogo ya se dropearon arriba (VERIFIED-ONLY) o caen al drop de abajo.
            _keep_seasoning = False
            if _seasoning_catalog_keep_enabled():
                try:
                    _keep_seasoning = _is_verified_for_shopping(name) and (price_per_lb > 0 or price_per_unit > 0)
                except Exception:
                    _keep_seasoning = False
            # [P2-SEASONING-RESTOCK-CLEAR · 2026-06-29] Si el usuario YA tiene este condimento en su Nevera
            # (consumed/inventario), NO lo re-listes como "1 empaque": el plan lo emitió nominal ("al gusto"),
            # la deducción por peso no lo restó, y el seasoning-keep lo re-inyectaba aunque ya esté comprado
            # (caso Vainilla tras restock; clase P3-RESTOCK-LECHE-UNIT). Match por nombre normalizado, fail-open.
            if _keep_seasoning:
                try:
                    if normalize_name(name) in _consumed_name_set:
                        _keep_seasoning = False
                        logging.info(
                            f"🧊 [P2-SEASONING-RESTOCK-CLEAR] '{name}' ya está en tu Nevera "
                            f"(consumed/inventario) → no se re-lista como empaque."
                        )
                except Exception:
                    pass
            if _keep_seasoning:
                try:
                    _seas_g = float(master_item.get("container_weight_g") or 0) or float(master_item.get("density_g_per_unit") or 0)
                except (TypeError, ValueError):
                    _seas_g = 0.0
                if _seas_g <= 0:
                    _seas_g = _SEASONING_DEFAULT_G
                weight_in_lbs = _seas_g / 453.592
                has_weight = True
                units = {}
                logging.info(
                    f"🌿 [SEASONING-CATALOG-KEEP] '{name}' es de catálogo pero el LLM lo emitió solo como "
                    f"cantidad nominal (pizca/al gusto) → listado como ~1 empaque (~{_seas_g:.0f}g) en vez "
                    f"de dropearlo (la receta lo usa). Cae al procesamiento normal de peso."
                )
                # NO continue: cae al `if has_weight:` de abajo y se procesa como ítem normal.
            else:
                # [P2-AGGREGATE-DROP-DIAG · 2026-05-16] Diagnostic logging.
                # Cuando un ingrediente aparece en `aggregated` (visible en log
                # `🛒 [AGGREGATE]`) pero NO en `aggregated_shopping_list` final
                # (log `🛒 [AGGREGATE FINAL]`), el coherence guard lo reporta
                # como `expected_only` divergence. Sin este log, debugging
                # requiere agregar instrumentación cada vez. Caso observado
                # 2026-05-16 plan 4cc91584: Avena emitida por receta pero
                # dropeada porque sus únicas unidades eran nominales (pizca/
                # al gusto). Este log captura el modo de fallo para que el
                # próximo incidente sea diagnosticable from log only.
                logging.info(
                    f"🛒 [AGGREGATE-DROP] '{name}' dropeado: sin peso "
                    f"(weight_in_lbs={weight_in_lbs:.4f}) y todas las unidades "
                    f"eran nominales (pizca/al gusto/etc). Units pre-dedup: "
                    f"{list(units.keys()) if units else '(vacío)'}. Si esperabas "
                    f"que el item apareciera en la lista, revisar la receta "
                    f"upstream: probablemente el LLM emitió cantidad nominal "
                    f"sin peso/unidad concreta."
                )
                continue
            
        if has_weight:
            if weight_in_lbs > 0.0001:
                _n_lower = name.lower()
                if any(kw in _n_lower for kw in ['pechuga', 'pavo', 'yogurt', 'lechosa', 'aguacate', 'arroz']):
                    logging.info(f"  🔬 [RAW LBS] {name}: {weight_in_lbs:.4f} lbs (mult={multiplier})")
                market_obj = apply_smart_market_units(name, weight_in_lbs, 'lb', 0.0, master_item, cycle_days=_cycle_days_for_note, text_demand_g=(text_demand_g_map or {}).get(name))
                # [P3-PRICE-MARKET-COVERAGE · 2026-06-20] Costo desde el DISPLAY redondeado (lo que
                # se compra), no desde weight_in_lbs crudo -> cierra el sub-costeo de staples por-peso.
                item_cost = _cost_from_market(market_obj, master_item, price_per_lb, price_per_unit)
                total_estimated_cost += item_cost
                market_obj["category"] = cat
                market_obj["display_category"] = display_cat
                # [P1-PLAN-DISPLAY-I18N · Task 5] Gloss bilingüe display-only para
                # la lista de compras — ver _display_name_en_for_item arriba.
                _name_en = _display_name_en_for_item(master_item)
                if _name_en:
                    market_obj["display_name_en"] = _name_en
                # [P1-COUNTRY-GLOSS-SOLO-INGLES · 2026-08-23] Gloss panhispánico
                # display-only, hermano del inglés: «Lechosa (papaya)».
                _gloss_es = _display_gloss_es_for_item(master_item)
                if _gloss_es:
                    market_obj["display_gloss_es"] = _gloss_es
                market_obj["is_staple"] = False
                # [P1-PDF-2] Cierra el drift de la heurística substring que vivía
                # SOLO en frontend. Backend es ahora SSOT para perishable
                # classification; el frontend lee este flag directo.
                # [PEPINO-FIX 2026-05-07] Cuando el item NO está en master
                # (cat="Otros" por default), usar `display_cat` como fallback.
                # `display_cat` viene de regex sobre el nombre y captura
                # variantes que master no registra (ej: Pepino → "VEGETALES").
                # Sin esto, items missing-from-master caían al default
                # "estable" mientras hybrid los marcaba "perecedero" → list
                # weekly mostraba Pepino en estables y biweekly/monthly en
                # perecederos (inconsistencia visible).
                _cat_for_perish = cat if (cat and cat.lower() != "otros") else display_cat
                market_obj["is_perishable"] = is_perishable_category(
                    _cat_for_perish, market_obj.get("shelf_life_days"), name=name
                )
                # [P3-PRICE-MARKET-COVERAGE · 2026-06-20] El costo ya viene de _cost_from_market
                # (sobre el display redondeado real); reemplaza al fallback P3-PRICE-UNIT-COVERAGE solo-si-0.
                market_obj["estimated_cost_rd"] = round(item_cost, 2) if item_cost > 0 else None
                # [P1-SHOPLIST-SANITY-CAP · 2026-08-21] La lista viva pedía 15 sobres de pimienta
                # y 10 frascos de orégano (RD$810, que contaminaban el banner de presupuesto).
                _apply_condiment_sanity_cap(market_obj, master_item, display_cat, cycle_days)
                item_val = market_obj if structured else market_obj["display_string"]
                results.append(item_val)
                categorized_results[display_cat].append(item_val)
                added = True

        for u, q in list(units.items()):
            # Saltar entradas nominales
            if u in nominal_units:
                continue
            if q > 0.0001:
                # DEDUP: Si este ingrediente ya fue agregado por peso (has_weight path),
                # y esta unidad residual es contable (unidad, cabeza, diente, mazo) que no se pudo convertir a gramos,
                # NO agregarlo de nuevo — ya está representado en la entrada de peso.
                if added and u.lower() in ['unidad', 'unidades', 'ud', 'uds', 'ud.', 'uds.', 'cabeza', 'cabezas', 'diente', 'dientes', 'mazo', 'mazos']:
                    logging.info(f"🔀 [DEDUP] Saltando entrada duplicada por {u} para '{name}' (ya tiene entrada por peso)")
                    continue
                market_obj = apply_smart_market_units(name, 0.0, u, q, master_item, cycle_days=_cycle_days_for_note, text_demand_g=(text_demand_g_map or {}).get(name))
                # [P3-PRICE-MARKET-COVERAGE · 2026-06-20] Costo desde el DISPLAY (envase/carton
                # redondeado); cubre huevo medio-carton (parsea "(N uds.)" x precio/30) y envases.
                item_cost = _cost_from_market(market_obj, master_item, price_per_lb, price_per_unit)
                total_estimated_cost += item_cost
                market_obj["category"] = cat
                market_obj["display_category"] = display_cat
                # [P1-PLAN-DISPLAY-I18N · Task 5] Mismo gloss bilingüe que el path
                # por peso arriba — ver _display_name_en_for_item.
                _name_en = _display_name_en_for_item(master_item)
                if _name_en:
                    market_obj["display_name_en"] = _name_en
                # [P1-COUNTRY-GLOSS-SOLO-INGLES · 2026-08-23] Gloss panhispánico
                # display-only, hermano del inglés: «Lechosa (papaya)».
                _gloss_es = _display_gloss_es_for_item(master_item)
                if _gloss_es:
                    market_obj["display_gloss_es"] = _gloss_es
                market_obj["is_staple"] = False
                # [P1-PDF-2] Mismo flag que arriba — todo item entrando a
                # `aggregated_shopping_list` debe tener `is_perishable` para que
                # el frontend nunca caiga al fallback de substring matching.
                # [PEPINO-FIX 2026-05-07] Cuando el item NO está en master
                # (cat="Otros" por default), usar `display_cat` como fallback.
                # `display_cat` viene de regex sobre el nombre y captura
                # variantes que master no registra (ej: Pepino → "VEGETALES").
                # Sin esto, items missing-from-master caían al default
                # "estable" mientras hybrid los marcaba "perecedero" → list
                # weekly mostraba Pepino en estables y biweekly/monthly en
                # perecederos (inconsistencia visible).
                _cat_for_perish = cat if (cat and cat.lower() != "otros") else display_cat
                market_obj["is_perishable"] = is_perishable_category(
                    _cat_for_perish, market_obj.get("shelf_life_days"), name=name
                )
                # [P3-PRICE-MARKET-COVERAGE · 2026-06-20] El costo ya viene de _cost_from_market
                # (sobre el display redondeado real); reemplaza al fallback P3-PRICE-UNIT-COVERAGE solo-si-0.
                market_obj["estimated_cost_rd"] = round(item_cost, 2) if item_cost > 0 else None
                # [P1-SHOPLIST-SANITY-CAP · 2026-08-21] La lista viva pedía 15 sobres de pimienta
                # y 10 frascos de orégano (RD$810, que contaminaban el banner de presupuesto).
                _apply_condiment_sanity_cap(market_obj, master_item, display_cat, cycle_days)
                item_val = market_obj if structured else market_obj["display_string"]
                results.append(item_val)
                categorized_results[display_cat].append(item_val)
                added = True
                
        # Removido el PANTRY_STAPLES force-add ("Disponible").
        # Si un alimento (incluyendo los estables) se deduce al 100%,
        # no debe irrumpir en la lista de compras del supermercado.

    # [P2-PROTEIN-YIELD-CANONICAL · 2026-08-03 · ronda 1] SELLO `protein_yield_applied` en
    # cada ítem, mismo patrón que `trip_window_days` (Task 6, P1-TRIP-WINDOWED-PERISHABLES):
    # el guard debe espejar cómo se CONSTRUYÓ la lista que tiene delante, no el knob VIGENTE
    # en el momento de re-evaluarla. Sin este sello, un cron re-validando un `plan_data`
    # persistido (o un rollback de knob ON→OFF, o un A/B ON→OFF a mitad de camino) fabrica
    # divergencias de magnitud fantasma: medido por el revisor, lista construida con knob OFF
    # (1.435 g) re-evaluada con knob ON → 25,9% de divergencia + `magnitude=True`. `results` y
    # `categorized_results[*]` comparten los MISMOS dicts (`item_val` se appendea a ambos
    # arriba), así que estampar aquí cubre las dos formas de salida.
    if structured and apply_protein_yield:
        for _it in results:
            if isinstance(_it, dict):
                _it["protein_yield_applied"] = True

    # [P2-GUARD-UNDERSUPPLY-CANONICAL · 2026-08-03] SELLO `pantry_deduction_applied`, tercero
    # del linaje (`trip_window_days`, `protein_yield_applied`). A diferencia de los otros dos,
    # este se estampa SIEMPRE — con `True` y con `False` — porque el valor informativo está
    # justamente en el `False`: es lo que le permite al guard afirmar «esta lista es canónica,
    # aquí no hay nevera que pueda haber deducido de más» en vez de asumirlo. Un sello ausente
    # (lista vieja, superficie que no pasa por el aggregator) es un tercer estado distinto que
    # `_pantry_deduction_seal` devuelve como `None` → default conservador.
    # `results` y `categorized_results[*]` comparten los MISMOS dicts (`item_val` se appendea a
    # ambos arriba), así que estampar aquí cubre las dos formas de salida.
    if structured:
        for _it in results:
            if isinstance(_it, dict):
                _it["pantry_deduction_applied"] = bool(_pantry_deduction_effective)

    results.sort(key=lambda x: x["display_string"] if structured else x)
    
    result_names = [r["name"] if structured and isinstance(r, dict) else str(r) for r in results]
    logging.info(f"🛒 [AGGREGATE FINAL] {len(results)} output items: {result_names[:20]}...")
    # [P2-CAP-LOG-LEVEL · 2026-07-29] UN warning agregado con los topes que sí son señal, en vez de
    # los ~343 per-item que copaban el 74,6% del journal. Corre al final del agregador, cuando
    # `_CAPS_APPLIED_LAST_RUN` ya está completo para ESTE run (se limpió al entrar, L7791).
    _log_severe_caps_summary()
    
    if categorize:
        for k in categorized_results:
            categorized_results[k].sort(key=lambda x: x["display_string"] if structured else x)
        return dict(categorized_results)
        
    return results

def aggregate_shopping_list(
    ingredients_list: list[str], *, num_days: int | None = None, multiplier: float = 1.0
) -> list[str]:
    """[P3-AGG-NUM-DAYS-PROPAGATE · 2026-08-04] Wrapper delgado — plumbing puro, sin
    lógica propia. ANTES llamaba a `aggregate_and_deduct_shopping_list` sin `num_days`
    ni `multiplier`: el agregador caía al fallback `_pw_days=3.0` (línea ~9948) →
    `_person_weeks = max(1.0, 1.0*3.0/7.0) = 1.0` SIEMPRE, sin importar cuántas personas
    ni qué duración (semanal/quincenal/mensual) tenga el plan real. Los caps P6 (latas de
    atún, aceite, especias, endulzantes...) leen `_person_weeks` para fijar su techo, así
    que recortaban una demanda mensual/multi-persona al mismo techo que una semanal de 1
    persona — verificado: atún household=2 mensual capaba a 2 latas (368g) en vez de las
    9 (1656g) que le corresponden.

    `num_days`/`multiplier` son keyword-only con default = comportamiento histórico exacto
    (`None`/`1.0`, byte-idéntico a antes de este fix) — un caller que no los pase no ve
    cambiar su resultado. Callers reales (`agent.py::swap_meal`, `chat_with_agent`,
    `chat_with_agent_stream`) los derivan del plan vía
    `agent._virtual_pantry_num_days_and_multiplier` (mismo SSOT que
    `get_shopping_list_delta`/`routers/plans.py::scaled_30`: `num_days` = días REALMENTE
    generados, `multiplier` = `household × cycle_qty_multiplier(duración) × 7/num_days`)."""
    return aggregate_and_deduct_shopping_list(ingredients_list, [], num_days=num_days, multiplier=multiplier)

def get_aggregated_shopping_list_for_plan(plan_result: dict) -> list[str]:
    return get_realtime_pantry(plan_result, [])

def fetch_inventory_and_consumed_for_plan(user_id: str, plan_result: dict, is_new_plan: bool = False) -> tuple:
    """[P1-5] Fetch one-shot del inventario físico + consumidos para un plan.

    Devuelve `(physical_inventory, consumed_ingredients)` listo para pasar
    como overrides a `get_shopping_list_delta`. Cuando un caller necesita
    invocar el delta con N multiplicidades distintas (1.0, 2.0, 4.0 para
    weekly/biweekly/monthly), debe llamar este helper UNA vez y pasar el
    resultado vía `inventory_override` + `consumed_override`. Esto evita
    que las queries a `user_inventory` (Realtime channel) y
    `consumed_meals` cambien entre las N llamadas y produzcan deltas
    inconsistentes.

    Para `user_id=None`/`"guest"`, retorna `([], [])`.
    """
    physical_inventory: list = []
    consumed_ingredients: list = []

    if not user_id or user_id == "guest":
        return physical_inventory, consumed_ingredients

    try:
        from db_inventory import get_raw_user_inventory
        from datetime import datetime
        raw_inventory = get_raw_user_inventory(user_id)
        if raw_inventory:
            master_list = get_master_ingredients()
            master_map = {m["name"]: m for m in master_list}
            PANTRY_STAPLES = {
                'Sal y ajo en polvo', 'Aceite de oliva', 'Aceite de coco',
                'Aceite de sésamo o maní', 'Salsa de soya', 'Orégano',
                'Canela', 'Pimienta', 'Sal', 'Vinagre', 'Ajo en polvo'
            }
            for item in raw_inventory:
                qty = float(item.get("quantity", 0))
                if qty <= 0:
                    continue
                name = item.get("ingredient_name", "")
                is_expired = False
                if name not in PANTRY_STAPLES:
                    created_at_str = item.get("created_at")
                    if created_at_str:
                        try:
                            item_date = datetime.strptime(created_at_str[:10], "%Y-%m-%d").date()
                            days_old = (datetime.now().date() - item_date).days
                            mi = master_map.get(name, {})
                            shelf_life = mi.get("shelf_life_days")
                            if shelf_life is None:
                                from db_inventory import _infer_shelf_life_days
                                shelf_life = _infer_shelf_life_days(name, mi.get("category", ""))
                            if (shelf_life - days_old) < 0:
                                is_expired = True
                        except Exception:
                            pass
                if not is_expired:
                    physical_inventory.append(item)

        if not is_new_plan:
            from db_plans import get_latest_meal_plan_with_id
            from db_facts import get_consumed_meals_since
            plan_record = get_latest_meal_plan_with_id(user_id)
            if plan_record and plan_record.get("plan_data"):
                plan_created_at = plan_record.get("created_at")
                if plan_created_at:
                    consumed_meals = get_consumed_meals_since(user_id, plan_created_at)
                    for cm in consumed_meals:
                        ings = cm.get("ingredients") or []
                        if isinstance(ings, list):
                            consumed_ingredients.extend(ings)
    except Exception as e:
        logging.error(f"[P1-5] Error en fetch_inventory_and_consumed_for_plan: {e}")

    return physical_inventory, consumed_ingredients


# [P1-COUNTRY-SYSTEM-F1 · 2026-08-16 (T7)] Único punto donde el aggregator deja de emitir
# montos en RD$ para un plan en modo beta (`plan_data['_pricing_mode'] == 'beta_no_prices'`).
#
# `aggregate_and_deduct_shopping_list` (arriba en este archivo) es la función de ~2000
# líneas que CALCULA `estimated_cost_rd` por ítem — pero solo la invocan `structured=True`
# los 3 call sites DENTRO de `get_shopping_list_delta` (pase principal, ventana de viaje,
# canonicalización de urgentes); sus otros 2 call sites (`aggregate_shopping_list`,
# `get_realtime_pantry`, "nevera virtual" del swap/chat) NUNCA piden `structured=True` — sus
# ítems son texto plano sin campos de costo. Verificado por grep cross-archivo antes de
# escribir este comentario: no hay una 4ª vía por la que un dict con `estimated_cost_rd`
# salga de este módulo sin pasar por aquí.
#
# Por eso NO se tocó el agregador de ~2000 líneas (ni sus 15+ call sites indirectos vía
# `get_shopping_list_delta`, documentados en su propio docstring): TODO caller de este
# archivo pasa `plan_result` como el `plan_data` PERSISTIDO (o, para la generación inicial,
# el `result` en construcción de `assemble_plan_node`, que ya lleva el flag estampado ANTES
# de estas llamadas — ver ese nodo) — nunca un dict ad-hoc sin la clave. Confirmado
# leyendo los ~15 call sites reales: agent.py (swap/chat), cron_tasks.py (T1/T2/GAP-F),
# routers/plans.py (/recalculate-shopping-list), tools.py (chat-modify) — todos pasan
# `plan_data`/`full_plan_data`/`plan_record["plan_data"]`, jamás un dict recortado.
def _strip_prices_for_beta_pricing_mode(res):
    """Muta `res` IN-PLACE anulando `estimated_cost_rd`/`estimated_cost` (si están presentes)
    en cada ítem — `list[dict]` (`structured=True, categorize=False`) o
    `dict[categoria, list[dict]]` (`structured=True, categorize=True`). En texto plano no hay
    campos de costo, pero sí puede viajar el label de marca dentro de `display_string`, por lo
    que también se sanea cada `str`. Retorna `res` para poder encadenarse inline en el `return`
    del caller.

    tooltip-anchor: _strip_prices_for_beta_pricing_mode (test_p1_country_system_f1.py)
    """
    # [P1-BETA-PRICE-LEAKS · 2026-08-21] Un precio no viaja sólo como número: viaja también como
    # el SKU del que salió. `display_qty` lleva «1 cartón (1 Lt · Wala)» — Wala, Zerca, Sosua,
    # Jazma y Rica son marcas de casa de supermercados DOMINICANOS que no existen en España, y
    # «funda» es el dominicanismo de bolsa. Medido en producción: 30 de 48 ítems del plan ES y 15
    # de 25 del US llevaban marca, y el usuario se lleva ese PDF al súper. No las eligió (son los
    # defaults más baratos del catálogo RD) y no puede quitarlas, porque el panel que las
    # gestionaría está oculto justamente por ser beta.
    #
    # Se quita la MARCA, no la presentación: «1 cartón (1 Lt)» sigue diciéndole qué comprar. El
    # separador ' · ' es el que `_pkg_from_product_row` usa para pegar tamaño y marca.
    _BRAND_SEP = " · "

    # [P1-COACH-SHOPLIST-BRAND-LEAK · 2026-08-23] `display_qty` era el único campo saneado,
    # pero ninguna de las dos rutas del coach lo narra: tools.py usa `display_string` de la
    # salida estructurada y agent.py inserta la salida plana (que ES ese display_string) en el
    # prompt. Además, el `rfind("(")` + truncado previo se comía TODO lo posterior al paréntesis:
    # limpiar «1 cartón (1 Lt · marca) de Leche» devolvía «1 cartón (1 Lt)» y perdía el alimento.
    # Se transforma únicamente el segmento parentético que contiene el separador SSOT, preservando
    # el resto byte por byte y el sufijo `c/u`. `sku_size_label` es la única forma legítimamente
    # desnuda (`tamaño · marca`), así que sólo ese campo habilita el fallback sin paréntesis.
    _BRANDED_PARENS_RX = re.compile(r"\([^()]*\)")
    _EACH_SUFFIX_RX = re.compile(r"(?:^|\s)c/u\s*$", re.IGNORECASE)

    def _strip_brand(value, *, allow_bare=False):
        if not isinstance(value, str) or _BRAND_SEP not in value:
            return value

        _changed = False

        def _clean_parens(match):
            nonlocal _changed
            _segment = match.group(0)
            if _BRAND_SEP not in _segment:
                return _segment
            _size, _brand_and_suffix = _segment.split(_BRAND_SEP, 1)
            _brand = _brand_and_suffix[:-1].strip()  # quitar el `)` para inspeccionar `c/u`
            _suffix = " c/u" if _EACH_SUFFIX_RX.search(_brand) else ""
            _changed = True
            return f"{_size.rstrip()}{_suffix})"

        _cleaned = _BRANDED_PARENS_RX.sub(_clean_parens, value)
        if _changed:
            return _cleaned
        if allow_bare and "(" not in value and ")" not in value:
            _size, _brand = value.rsplit(_BRAND_SEP, 1)
            _suffix = " c/u" if _EACH_SUFFIX_RX.search(_brand) else ""
            return f"{_size.rstrip()}{_suffix}"
        return value

    def _strip_item(it):
        if isinstance(it, dict):
            if "estimated_cost_rd" in it:
                it["estimated_cost_rd"] = None
            if "estimated_cost" in it:
                it["estimated_cost"] = None
            if "display_qty" in it:
                it["display_qty"] = _strip_brand(it.get("display_qty"))
            if "display_string" in it:
                it["display_string"] = _strip_brand(it.get("display_string"))
            if "sku_size_label" in it:
                it["sku_size_label"] = _strip_brand(
                    it.get("sku_size_label"), allow_bare=True)

    def _strip_items(items):
        for idx, it in enumerate(items):
            if isinstance(it, str):
                items[idx] = _strip_brand(it)
            else:
                _strip_item(it)

    if isinstance(res, dict):
        for items in res.values():
            if isinstance(items, list):
                _strip_items(items)
    elif isinstance(res, list):
        _strip_items(res)
    return res


def get_shopping_list_delta(
    user_id: str,
    plan_result: dict,
    is_new_plan: bool = False,
    categorize: bool = False,
    structured: bool = False,
    multiplier: float = 1.0,
    *,
    inventory_override: list | None = None,
    consumed_override: list | None = None,
    cycle_days: int | None = None,
    window_days: list | None = None,
):
    """Calcula el verdadero Delta: Ingredientes Totales del Plan - Inventario Físico Actual - (Opcional) Consumidos.

    [P1-5] Si el caller necesita N multiplicidades del mismo plan (típico:
    weekly/biweekly/monthly), debe llamar `fetch_inventory_and_consumed_for_plan`
    UNA vez y pasar el resultado vía `inventory_override` + `consumed_override`.
    Sin estos overrides, cada invocación re-consulta `user_inventory` (que
    puede cambiar entre llamadas por Realtime channel, restock, cron) y
    `consumed_meals_since` — produciendo deltas inconsistentes que el
    frontend muestra al usuario al cambiar `groceryDuration`.

    `cycle_days` [P1-SKU-COVER-HONESTY-R1 · 2026-08-02]: días que representa esta llamada
    (7/15/30) — se propaga a `apply_smart_market_units` sólo para el copy de la nota "alcanza
    ~N de M días". Opcional, default None → 7 (comportamiento previo). Los callers de HOY
    (routers/plans.py, cron_tasks.py, tools.py) invocan esta función 3 veces por surface
    (`aggr_7`/`aggr_15`/`aggr_30`) con el `multiplier` ya calculado por duración — ninguno pasa
    todavía `cycle_days` explícito; es un seguimiento pendiente, no de este archivo.

    `window_days` [P1-TRIP-WINDOWED-PERISHABLES · 2026-08-02]: días del viaje ACTIVO
    (típicamente `active_trip_window_days(plan_result)` → los 7 días desde
    `grocery_start_date`/`days[0]` tras el shift). Si viene, los PERECEDEROS se agregan
    SOLO de esa ventana (escala `7/len(window)`) y los ESTABLES siguen del agregado del
    periodo completo. `None` (default) = comportamiento previo: promedio de todos los días
    materializados proyectado a 7. Requiere `structured=True` (la partición necesita el
    nombre de cada ítem); con salida en texto plano se ignora con un WARNING.

    Gateado por `MEALFIT_TRIP_WINDOWED_PERISHABLES`, **default `False`** desde la ronda 1:
    con el knob apagado `window_days` no tiene efecto alguno (los callsites pasan el `None`
    que `active_trip_window_days` devuelve, y este método lo ignoraría igual). Ver el
    bloque de cabecera del P-fix para la medición y los prerequisitos (a)-(d) de encendido.
    """
    # [P1-SUPERMARKET-COSTING · 2026-07-02] Marca preferida del usuario → costeo
    # con el envase elegido. Fetch UNA vez por run (todas las superficies —
    # generación, recalc, chat, crons — pasan por este cuello). Guests/errores →
    # None (fail-open, costeo estándar). Knob: MEALFIT_BRAND_PREF_COSTING.
    brand_prefs = None
    if _brand_pref_costing_enabled():
        try:
            brand_prefs = fetch_brand_pref_packages(user_id) or None
        except Exception as _bp_exc:
            logging.warning(f"⚠️ [P1-SUPERMARKET-COSTING] prefs no disponibles (fail-open): {_bp_exc}")
            brand_prefs = None
    # [P1-BRAND-LIST-VISIBILITY · 2026-07-06] Marcas default (más baratas) para los
    # ítems sin preferencia manual — la lista/PDF enseñan marca en cada alimento
    # con productos del súper. Global + cacheado (TTL 10 min). Fail-open: None.
    brand_defaults = None
    if _brand_default_packages_enabled():
        try:
            brand_defaults = fetch_brand_default_packages() or None
        except Exception as _bd_exc:
            logging.warning(f"⚠️ [P1-BRAND-LIST-VISIBILITY] defaults no disponibles (fail-open): {_bd_exc}")
            brand_defaults = None

    all_ingredients = []
    # [P0-SHOPPING-CYCLE-DAYS · 2026-08-22] SSOT de la fuente de días (incluye los que el
    # shift archivó). Con `plan_result["days"]` a pelo, cada recálculo posterior a un
    # shift reconstruía la lista desde una ventana más corta y la SOBRESCRIBÍA: 48
    # alimentos → 25 en el plan real 2245eb45, dejando al usuario sin proteína que
    # cocinar y matando el chunk siguiente contra el gate de despensa.
    days = shopping_source_days(plan_result)
    if not days and plan_result.get("meals"):
        days = [{"day": 1, "meals": plan_result.get("meals")}] 
    if not days and plan_result.get("perfectDay"):
        days = [{"day": 1, "meals": plan_result.get("perfectDay")}]

    # Si hay 3 días generados, representan un ciclo rotativo. Promediamos por día y proyectamos a 7 días.
    num_days = max(1, len(days))
    base_duration_scale = 7.0 / num_days

    # [P1-7] Defensa numérica en cascada. Si `multiplier` llega como NaN/Inf
    # (perfil corrupto, overflow en composiciones del caller),
    # `aggregate_and_deduct_shopping_list` también lo clampa pero queremos
    # detectar y loguear el caller upstream aquí, donde tenemos contexto
    # (user_id, num_days). El clamp final a [0.01, 50.0] vive en el
    # aggregator — aquí solo normalizamos NaN/Inf a un default seguro.
    try:
        _raw_mult = float(multiplier)
    except (TypeError, ValueError):
        _raw_mult = 1.0
    if math.isnan(_raw_mult) or math.isinf(_raw_mult):
        logging.warning(
            f"[P1-7/DELTA-MULT] multiplier={multiplier!r} no-finito desde caller. "
            f"Defaulteando a 1.0; bug upstream probable."
        )
        _raw_mult = 1.0
    multiplier = _raw_mult

    effective_multiplier = multiplier * base_duration_scale

    logging.info(f"🔄 [SHOPPING MATH] days_len={num_days} base_scale={base_duration_scale} raw_mult={multiplier} eff_mult={effective_multiplier}")


    # [P1-TRIP-WINDOWED-PERISHABLES · 2026-08-02] Extracción de ingredientes extraída a
    # un helper local para poder correrla DOS veces sobre subconjuntos distintos de días
    # (plan completo → estables; ventana del viaje → perecederos) sin duplicar el
    # contrato de parseo. Cuerpo idéntico al loop previo.
    def _collect_ingredients(day_list):
        collected = []
        meals_seen = 0
        for day in day_list:
            if not isinstance(day, dict):
                continue
            for meal in day.get("meals", []):
                # [P2-4] SSOT: helper compartido con expected_sum_from_recipes y
                # el extractor de facts. Garantiza simetría capa-B del coherence
                # guard (expected ≡ aggregated en cuanto a qué meals contribuyen).
                if _should_skip_meal_for_aggregation(meal):
                    continue
                meals_seen += 1
                # [P1-4] Preferir `ingredients_raw` (pre-humanización) sobre
                # `ingredients` (display-friendly). El humanize convierte
                # "200g pechuga de pollo" → "1 pechuga de pollo (porción)" para
                # la UI; al re-agregar el plan persistido, la versión humanizada
                # pierde la unidad métrica y `_parse_quantity` cae a unit='unidad'
                # con qty=1, perdiendo el peso real. `humanize_plan_ingredients`
                # preserva el original en `ingredients_raw` desde P1-4.
                # Fallback al humanizado solo si el plan es legacy (pre-P1-4).
                ingredients = meal.get("ingredients_raw") or meal.get("ingredients", [])
                if not ingredients:
                    # Fallback: check if ingredients are inside a 'recipe' dict
                    recipe = meal.get("recipe")
                    if isinstance(recipe, dict):
                        ingredients = recipe.get("ingredients", [])
                for i in ingredients:
                    if isinstance(i, str):
                        collected.append(i)
                    elif isinstance(i, dict):
                        q = i.get("quantity", 0)
                        u = i.get("unit", "unidad")
                        n = i.get("name") or i.get("item_name") or i.get("display_name") or "Desconocido"
                        if q > 0 or u in ['pizca', 'al gusto', 'cantidad necesaria', 'chin', 'toque', 'chorrito']:
                            collected.append(f"{q} {u} de {n}")
                        else:
                            collected.append(n)
        return collected, meals_seen

    all_ingredients, meal_count = _collect_ingredients(days)

    logging.info(f"🛒 [SHOPPING EXTRACT] {len(days)} days, {meal_count} meals, {len(all_ingredients)} raw ingredients")

    # [P1-TRIP-WINDOWED-PERISHABLES · 2026-08-02] Validación de la ventana del viaje.
    # Se descarta (no-op, comportamiento previo byte-idéntico) cuando: knob apagado, no
    # vino ventana, la ventana ES el plan completo (caso de producción del viaje 1: solo
    # 2-3 días materializados) o la salida no es estructurada (la partición
    # perecedero/estable necesita el nombre de cada ítem, imposible sobre strings).
    _trip_window: list | None = None
    if window_days and _trip_windowed_perishables_enabled():
        try:
            _candidate = [d for d in window_days if isinstance(d, dict)]
        except TypeError:
            _candidate = []
        if not structured:
            if _candidate:
                logging.warning(
                    "[P1-TRIP-WINDOWED-PERISHABLES] window_days ignorada: requiere "
                    "structured=True (salida en texto plano no permite particionar "
                    "perecedero/estable)."
                )
        elif 0 < len(_candidate) < num_days:
            _trip_window = _candidate

    # [P1-5] Inventario + consumidos: si el caller pasó overrides (typical
    # cuando invoca el delta con N multiplicidades), reutilizamos su snapshot
    # para garantizar consistencia entre las N listas. Sin override, hacemos
    # el fetch aquí (caso 1 invocación: agente, tools, recalc).
    #
    # [P3-CANONICAL-AGG-WEEKLY · 2026-05-18] Si `is_new_plan=True`, forzamos
    # listas vacías ANTES del check del override. Esto cierra el bug del
    # refactor canónico anterior: callers que querían canonical pasaban
    # `is_new_plan=True` Y `inventory_override=_inv_snap` (porque _inv_snap
    # se reusa downstream para self-heal). Antes, el override ganaba y se
    # producía delta en vez de canonical. Ahora is_new_plan tiene precedencia
    # explícita sobre el override (semánticamente: "this is canonical, don't
    # deduct anything").
    if is_new_plan:
        physical_inventory = []
        consumed_ingredients = []
    elif inventory_override is not None or consumed_override is not None:
        physical_inventory = list(inventory_override) if inventory_override is not None else []
        consumed_ingredients = list(consumed_override) if consumed_override is not None else []
    else:
        physical_inventory, consumed_ingredients = fetch_inventory_and_consumed_for_plan(
            user_id, plan_result, is_new_plan
        )
            
    items_to_deduct = []
    if physical_inventory:
        items_to_deduct.extend([f"{item.get('quantity', 0)} {item.get('unit', 'unidad')} de {item.get('ingredient_name')}" for item in physical_inventory])
    if consumed_ingredients:
        items_to_deduct.extend(consumed_ingredients)

    # [P2-PROTEIN-YIELD-CANONICAL · 2026-08-03] Solo la lista CANÓNICA (`is_new_plan=True`,
    # sin lado inventario) puede reabrir la regla #2 de yield (proteínas cocidas → 1.35×
    # crudo) — con `is_new_plan=False` el delta sigue siendo peso literal en ambos lados
    # (asimetría P1-2 intacta, byte-idéntico). Gateado además por el knob
    # `MEALFIT_PROTEIN_YIELD_ON_CANONICAL` — [P3-PROTEIN-YIELD-DECISION · 2026-08-04]
    # default `True` tras medir el delta real (~RD$136/semana promedio por plan afectado,
    # ver docstring de `_protein_yield_on_canonical_enabled`); rollback sin redeploy con
    # `MEALFIT_PROTEIN_YIELD_ON_CANONICAL=false`.
    _apply_protein_yield = bool(is_new_plan) and _protein_yield_on_canonical_enabled()

    # [P1-VEG-BACKFILL-HONESTY · 2026-08-02] Demanda de las RECETAS por alimento, en gramos —
    # mismo parse que usa el coherence guard (`expected_sum_from_recipes`, misma función que
    # `compare_expected_vs_aggregated` consume) y misma normalización de unidades
    # (`_normalize_food_units_to_base`, que unifica g/kg/ml/taza/cda dentro del mismo sistema
    # físico). Se calcula con el MISMO `effective_multiplier` que escala `all_ingredients` abajo —
    # ambos lados de la comparación quedan a la misma escala. Threaded hasta
    # `apply_smart_market_units` para que, si la cantidad final resuelta cae por debajo de
    # `QTY_SHORTFALL_NOTE_MIN × texto` sin que ningún cap real lo explique, se estampe
    # `capped_by="qty_reconcile_v7"` sintético. Fail-open: si el parse falla, el mapa queda vacío
    # y el mecanismo nuevo es no-op (comportamiento previo).
    #
    # [P1-VEG-BACKFILL-HONESTY · 2026-08-03 · ronda de revisión] `expected_sum_from_recipes`
    # devuelve nombres CRUDOS (post-parse, pre-canonicalización) pero el lado comprado (dentro de
    # `aggregate_and_deduct_shopping_list`) resuelve por nombre CANÓNICO
    # (`canonicalize_shopping_food_name`, mismo SSOT que ahora usa ese aggregator). Sin pasar los
    # nombres de este mapa por la MISMA cadena, "300 g de tomates" quedaba con la key 'Tomates'
    # mientras el ítem comprado resolvía 'Tomate' -> `.get(name)` fallaba SIEMPRE que la receta
    # usara plural (medido: tomate/cebolla/espinaca — la mitad del vocabulario real de un LLM).
    # `canonicalize_shopping_food_name` puede colapsar varias keys crudas al mismo canónico (ej.
    # "Tomate" y "Tomates" en dos líneas de receta distintas) — se SUMAN, no se sobreescriben.
    try:
        _tdg_master_map = _build_shopping_master_map()
        _tdg_raw_units = {
            f: _normalize_food_units_to_base(u or {})
            for f, u in (expected_sum_from_recipes(
                plan_result, apply_yield=False, multiplier=effective_multiplier,
                # [P2-PROTEIN-YIELD-CANONICAL · 2026-08-03] Espejo obligatorio: si el
                # aggregator (más abajo) compra 1.35× para proteínas cocidas, la demanda
                # de texto que alimenta el backstop de Task 8 (P1-VEG-BACKFILL-HONESTY)
                # debe subir la MISMA 1.35× — de lo contrario el backstop vería "se
                # compró 135% del texto" y fabricaría una nota de recompra falsa sobre
                # una compra que en realidad es correcta.
                apply_protein_yield=_apply_protein_yield,
            ) or {}).items()
        }
        _text_demand_g_map: dict = {}
        for _food, _units in _tdg_raw_units.items():
            _g = _units.get("g")
            if not (isinstance(_g, (int, float)) and _g > 0):
                continue
            _canon_food = canonicalize_shopping_food_name(_food, _tdg_master_map)
            _text_demand_g_map[_canon_food] = _text_demand_g_map.get(_canon_food, 0.0) + float(_g)
    except Exception as _tdg_exc:
        logging.warning(f"[P1-VEG-BACKFILL-HONESTY] text_demand_g_map falló (fail-open, "
                        f"no-op): {type(_tdg_exc).__name__}: {_tdg_exc}")
        _text_demand_g_map = {}

    # [P1-VEG-BACKFILL-HONESTY · 2026-08-03 · review final] El backstop sólo puede comparar
    # magnitudes HOMOGÉNEAS.
    #
    # `_text_demand_g_map` es demanda BRUTA: sale de las recetas y no resta nada. Pero cuando hay
    # `items_to_deduct` (Nevera + consumidos), el lado comprado es un DELTA — el agregador hace
    # `aggregated[name][unit] -= qty` antes de resolver unidades de mercado. Comparar neto contra
    # bruto dispara el sello sobre EXACTAMENTE lo que el usuario ya tiene en casa: reproducido con
    # receta 2100 g + nevera 500 g + compra 1600 g → nota de recompra falsa. Y el falso positivo
    # crece cuanto mejor use el usuario la Nevera Inteligente, que es el incentivo invertido.
    #
    # Las superficies afectadas NO son marginales: `is_new_plan` es False POR DEFAULT en la firma,
    # y con ese modo corren la tool del coach (`check_shopping_list`), `mark_shopping_list_purchased`
    # (que además reinyecta el `display_string` a `restock_inventory`), los dos callsites de
    # `agent.py` que meten el delta en el system prompt, y `get_pantry_completion_list` — cuyo
    # propósito literal es «lo que el plan necesita MENOS lo que ya tiene», donde el falso positivo
    # sería del 100%.
    #
    # Se gatea por `items_to_deduct` y no por `is_new_plan` porque la condición REAL es «¿hubo
    # deducción?»: con `is_new_plan=True` la lista es canónica y `items_to_deduct` está vacío por
    # construcción (arriba), pero un delta con la nevera vacía también es homogéneo y ahí el
    # mecanismo sigue siendo correcto. Alternativa descartada: netear el mapa de texto con los
    # mismos `items_to_deduct` — reimplementa la deducción (parse de unidades, canonicalización,
    # orden de aplicación) en un segundo sitio, que es la clase de duplicación que este repo paga
    # cada vez que la escribe. Fail-safe: en la duda se calla la nota, nunca se inventa.
    # tooltip-anchor: P1-QTY-SHORTFALL-HOMOGENEO
    _tdg_para_agg = _text_demand_g_map if not items_to_deduct else {}
    if _text_demand_g_map and items_to_deduct:
        logging.info(
            f"[P1-VEG-BACKFILL-HONESTY] backstop de texto OMITIDO: la lista es un delta "
            f"({len(items_to_deduct)} ítems deducidos) y la demanda de recetas es bruta"
        )

    # [P1-PERSON-WEEKS-CYCLE-AWARE · 2026-07-30] `num_days` viaja al agregador porque los topes por
    # persona-semana necesitan deshacer el `base_duration_scale = 7/num_days` que se aplica tres
    # líneas más arriba. Sin él, `_person_weeks` usaba un `3` hardcodeado y los topes salían 4,7×
    # apretados en un ciclo de 14 días.
    res = aggregate_and_deduct_shopping_list(all_ingredients, items_to_deduct, categorize=categorize, structured=structured, multiplier=effective_multiplier, brand_prefs=brand_prefs, brand_defaults=brand_defaults, num_days=num_days, cycle_days=cycle_days, text_demand_g_map=_tdg_para_agg, apply_protein_yield=_apply_protein_yield)

    # [P1-TRIP-WINDOWED-PERISHABLES · 2026-08-02] Segunda pasada SOLO cuando hay ventana
    # de viaje: mismo agregador, mismos descuentos de inventario, misma aritmética —
    # únicamente cambia la base temporal (`7/len(ventana)` en vez de `7/num_days`). El
    # merge se queda los perecederos de esta pasada y los estables de la del periodo.
    # `num_days=len(ventana)` mantiene los topes por persona-semana idénticos
    # (`_person_weeks = multiplier_eff * num_days / 7` cancela la escala en ambas ramas).
    # Correr el agregador dos veces con el MISMO `items_to_deduct` no doble-descuenta:
    # cada ítem del resultado sale de exactamente UNA de las dos pasadas.
    if _trip_window:
        try:
            _window_ingredients, _window_meals = _collect_ingredients(_trip_window)
            _window_scale = 7.0 / float(len(_trip_window))
            logging.info(
                f"🗓️ [P1-TRIP-WINDOWED-PERISHABLES] ventana={len(_trip_window)}d de "
                f"{num_days}d materializados · meals={_window_meals} · "
                f"scale={_window_scale:.4f} (plan={base_duration_scale:.4f})"
            )
            # [P1-VEG-BACKFILL-HONESTY · 2026-08-03 · ronda de revisión] El mapa ya está calculado
            # sobre el plan COMPLETO (arriba) — sin pasarlo aquí, los ítems que esta segunda
            # pasada resuelve (perecederos de la ventana del viaje) nunca reciben el backstop.
            # Bug latente (este camino está OFF por default, `MEALFIT_TRIP_WINDOWED_PERISHABLES`)
            # que dejarlo coherente evita para cuando se encienda.
            _res_window = aggregate_and_deduct_shopping_list(
                _window_ingredients, items_to_deduct, categorize=categorize,
                structured=structured, multiplier=multiplier * _window_scale,
                brand_prefs=brand_prefs, brand_defaults=brand_defaults,
                num_days=len(_trip_window), cycle_days=cycle_days,
                text_demand_g_map=_tdg_para_agg, apply_protein_yield=_apply_protein_yield,
            )
            res = _merge_trip_windowed_result(res, _res_window, window_len=len(_trip_window))
        except Exception as _tw_exc:
            # Fail-open: cualquier fallo del ventaneo deja la lista del periodo intacta
            # (comportamiento previo). Jamás abortar la construcción de la lista.
            logging.warning(
                f"⚠️ [P1-TRIP-WINDOWED-PERISHABLES] ventaneo falló, se conserva el "
                f"agregado del periodo: {type(_tw_exc).__name__}: {_tw_exc}"
            )


    # [P0-3] Inyectar items de compra urgente si el plan superó validación de despensa en flexible_mode
    # [P1-URGENT-LIST-CANONICAL · 2026-08-09] Los urgentes son LÍNEAS DE RECETA por-comida
    # («95 g de mango en cubos», «1 cdta de pimentón») — inyectarlas VERBATIM infló la lista del
    # owner a 104 ítems (33 pseudo-productos), el contador «Marcas del súper» los contaba y el
    # PDF mostraba absurdos («20 g de avena en hojuelas · 0.87 ud»). Ahora pasan por el MISMO
    # agregador que reduce las líneas del plan a productos canónicos (funde duplicados: 3 líneas
    # de mango → 1 Mango con su cantidad real). Fail-open: si el agregador falla o devuelve
    # vacío con urgentes presentes, cae a la inyección cruda de siempre (mejor crudo que ausente
    # — son compras de seguridad del modo flexible). tooltip-anchor: P1-URGENT-LIST-CANONICAL
    urgent_items = plan_result.get("_pantry_supplement_required", [])
    if urgent_items:
        def _tag_urgent(entry):
            if isinstance(entry, dict):
                entry = dict(entry)
                entry["category"] = "🚨 Compra Urgente"
                entry["display_category"] = "🚨 Compra Urgente"
                entry["is_staple"] = False
                # [P1-PDF-2] urgentes = perecederos ("comprar pronto"), explícito.
                entry["is_perishable"] = True
                # [P0-2] contrato: espejo numérico SIEMPRE presente (el frontend no parsea
                # "1 1/2"). Si la entrada del agregador no lo trae, se deriva o cae a 1.0.
                if not isinstance(entry.get("market_qty_numeric"), (int, float)):
                    try:
                        entry["market_qty_numeric"] = float(entry.get("market_qty"))
                    except (TypeError, ValueError):
                        entry["market_qty_numeric"] = 1.0
                _ds = str(entry.get("display_string") or entry.get("name") or "")
                if not _ds.startswith("⚠️"):
                    entry["display_string"] = f"⚠️ {_ds}"
                return entry
            return f"⚠️ {entry}"

        def _raw_urgent(item):
            return {
                "name": item,
                "market_qty": 1,
                # [P0-2] Espejo numérico siempre presente (el frontend no parsea "1 1/2").
                "market_qty_numeric": 1.0,
                "market_unit": "ud",
                "display_qty": item,
                "display_string": f"⚠️ {item}",
                "category": "🚨 Compra Urgente",
                "display_category": "🚨 Compra Urgente",
                "is_staple": False,
                "is_perishable": True,
            } if structured else f"⚠️ {item}"

        _urgent_entries = None
        try:
            _canon = aggregate_and_deduct_shopping_list(
                [str(i) for i in urgent_items], [],
                categorize=False, structured=structured, multiplier=1.0)
            if isinstance(_canon, list) and _canon:
                _urgent_entries = [_tag_urgent(e) for e in _canon]
                logging.info(
                    f"🧺 [P1-URGENT-LIST-CANONICAL] {len(urgent_items)} línea(s) urgente(s) "
                    f"→ {len(_urgent_entries)} producto(s) canónicos en la lista")
        except Exception as _uc_exc:
            logging.warning(
                f"[P1-URGENT-LIST-CANONICAL] canonicalización falló, inyección cruda "
                f"(fail-open): {type(_uc_exc).__name__}: {_uc_exc}")
        if _urgent_entries is None:
            _urgent_entries = [_raw_urgent(item) for item in urgent_items]

        if categorize:
            if isinstance(res, dict):
                res["🚨 Compra Urgente"] = list(_urgent_entries)
        else:
            if isinstance(res, list):
                res.extend(_urgent_entries)

    # [P1-COUNTRY-SYSTEM-F1 · 2026-08-16 (T7)] `plan_result` es el `plan_data` persistido (o
    # el `result` en construcción de assemble_plan_node, que ya lleva la clave estampada
    # ANTES de llamar aquí) — ver el comentario de `_strip_prices_for_beta_pricing_mode`.
    if isinstance(plan_result, dict) and plan_result.get("_pricing_mode") == "beta_no_prices":
        _strip_prices_for_beta_pricing_mode(res)

    # [P1-UNIT-SYSTEM-BY-COUNTRY · 2026-08-21] Proyección métrica del DISPLAY, en el último paso.
    #
    # Aquí y no dentro del agregador por una razón medida: `_cost_from_market` calcula el costo
    # PARSEANDO el display redondeado («costo desde el DISPLAY, no desde weight_in_lbs crudo» —
    # P3-PRICE-MARKET-COVERAGE). Convertir antes le daría gramos a un parser que espera libras y
    # el costo saldría mal. Este es el único punto de salida de la función, así que también es el
    # único sitio donde está garantizado que todo el cálculo ya terminó.
    #
    # El país sale del SELLO del plan (`country_for_plan`, P1-PLAN-STAMPS-COUNTRY), que ya viaja
    # en `plan_result`: así los 26 call sites de esta función no cambian — 26 sitios donde
    # olvidarse de pasar un `country=` nuevo. Un plan sin sello (todos los anteriores al sello)
    # cae a 'DO' y conserva exactamente la conducta de hoy.
    try:
        from constants import country_for_plan as _cfp_units
        _pais_lista = _cfp_units(plan_result if isinstance(plan_result, dict) else {}, None)
        if unit_system_for_country_safe(_pais_lista) == "metric":
            _project_units_over_result(res, _pais_lista)
    except Exception as _us_exc:
        logging.warning(
            f"[P1-UNIT-SYSTEM-BY-COUNTRY] proyección métrica no-op (fail-open): "
            f"{type(_us_exc).__name__}: {_us_exc}")

    return res


def compute_pantry_completion_delta(
    user_id: str,
    plan_result: dict,
    multiplier: float = 1.0,
    *,
    inventory_override: list | None = None,
    categorize: bool = True,
    structured: bool = True,
):
    """[P1-RENEWAL-PANTRY-AWARE · 2026-06-28 · Fase 2] Lista de FALTANTES para
    "completar la nevera al 100%" para este plan: lo que el plan NECESITA MENOS lo
    que el usuario YA TIENE en la nevera (resta CUANTITATIVA real). Es decir, lo que
    debe comprar para que su nevera cubra el plan ("te faltan 2L de leche, 0.4kg de
    pollo").

    Es READ-ONLY y DERIVADO: reusa get_shopping_list_delta(is_new_plan=False) que
    deduce el inventario físico cuantitativamente. NUNCA toca la lista canónica
    (aggregated_shopping_list_*, que se persiste con is_new_plan=True). Deduce SOLO
    el inventario físico actual (la nevera) — NO 'consumidos' (irrelevante para un
    plan recién renovado). Si el caller ya tiene el snapshot de nevera (p.ej. el
    recalc lo fetcha una vez como `_inv_snap`), pasarlo vía `inventory_override`
    evita un segundo fetch a user_inventory.

    Gating: el caller debe respetar constants.PANTRY_COMPLETION_LIST_ENABLED (default
    OFF). Esta función NO consulta el knob (es pura/reutilizable).
    """
    if not user_id or user_id == "guest":
        return {} if categorize else []
    inv = inventory_override
    if inv is None:
        inv, _ = fetch_inventory_and_consumed_for_plan(user_id, plan_result, is_new_plan=False)
    return get_shopping_list_delta(
        user_id,
        plan_result,
        is_new_plan=False,
        categorize=categorize,
        structured=structured,
        multiplier=multiplier,
        inventory_override=inv,
        consumed_override=[],
    )


def get_realtime_pantry(
    plan_result: dict, consumed_ingredients: list[str], *, num_days: int | None = None, multiplier: float = 1.0
) -> list[str]:
    """[P3-AGG-NUM-DAYS-PROPAGATE · 2026-08-04] Wrapper delgado — plumbing puro. ANTES
    llamaba a `aggregate_and_deduct_shopping_list` sin `num_days`/`multiplier`: la
    «nevera virtual» que ve el LLM del swap (path PRIMARIO, `agent.py::swap_meal`) caía
    al fallback `_pw_days=3.0`/`_person_weeks=1.0` y quedaba capada a 1 persona-semana en
    CUALQUIER plan multi-semana/household>1 (ver docstring de `aggregate_shopping_list`
    para los números verificados). `num_days`/`multiplier` keyword-only, default = `None`/
    `1.0` (comportamiento histórico exacto, byte-idéntico para callers que no los pasen).
    Callers reales derivan ambos del plan vía `agent._virtual_pantry_num_days_and_multiplier`."""
    all_ingredients = []
    days = plan_result.get("days", [])
    if not days and plan_result.get("meals"):
        days = [{"day": 1, "meals": plan_result.get("meals")}] 
    if not days and plan_result.get("perfectDay"):
        days = [{"day": 1, "meals": plan_result.get("perfectDay")}]


    for day in days:
        for meal in day.get("meals", []):
            # [P2-4] SSOT: mismo helper que expected/delta para evitar drift.
            if _should_skip_meal_for_aggregation(meal):
                continue
            ingredients = meal.get("ingredients", [])
            for i in ingredients:
                if isinstance(i, str):
                    all_ingredients.append(i)
                elif isinstance(i, dict):
                    q = i.get("quantity", 0)
                    u = i.get("unit", "unidad")
                    n = i.get("name") or i.get("item_name") or i.get("display_name") or "Desconocido"
                    if q > 0 or u in ['pizca', 'al gusto', 'cantidad necesaria', 'chin', 'toque', 'chorrito']:
                        all_ingredients.append(f"{q} {u} de {n}")
                    else:
                        all_ingredients.append(n)

    return aggregate_and_deduct_shopping_list(
        all_ingredients, consumed_ingredients, num_days=num_days, multiplier=multiplier
    )
