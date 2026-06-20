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


# [P2-LLM-TIMEOUT-SWEEP · 2026-05-30 · P0-DEEPSEEK-MIGRATION · 2026-06-12]
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


def _batched_embed_documents(client, all_texts, batch_size, delay_s, retry_label):
    """Particiona `embed_documents` en batches para no saturar RPM del modelo.

    Cada batch va envuelto en `_gemini_call_with_retry`, así un 429 transitorio
    en el batch K solo reintenta ese batch (los anteriores ya están en `out` y
    no se pierden). Si todos los textos caben en un batch, comportamiento
    idéntico al pre-fix (sin overhead).
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
    [P0-DEEPSEEK-MIGRATION · 2026-06-12] El ID viene de
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

def get_semantic_cache():
    """Devuelve el caché semántico (master_list + vectors + embeddings_client).

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
        # [P0-DEEPSEEK-MIGRATION · 2026-06-12] Via capa pluggable. Con
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
                res = execute_sql_query("SELECT * FROM master_ingredients", fetch_all=True)
                _master_cache = res or []
                _master_cache_ts = now
            except Exception as e:
                logging.error(f"Error fetching master_ingredients via pool: {e}")
                if _master_cache is None:
                    _master_cache = []
        else:
            logging.error("No connection_pool available to fetch master_ingredients")
            if _master_cache is None:
                _master_cache = []
    return _master_cache

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
    # NO incluir: casabe (deshidratado), galletas (selladas, secas), pan tostado.
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
        hybrid.append(out_item)

    return hybrid

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

def normalize_name(orig_name: str) -> str:
    n = str(orig_name).lower().strip()
    n = re.sub(r'\(.*?\)', '', n).strip()
    # Limpieza de prefijos contenedores o medidas informales
    n = re.sub(r'^(cda|cdta|cdita|cucharada|cucharadita|taza|vaso|pizca|chorrito|puñado|atado|manojo|scoop|lonja|loncha|paquete|paquetico|funda|lata|sobre|sobrecito|chin|toque)(s)?\s*(de\s+|del\s+)?', '', n, flags=re.IGNORECASE)
    # Nueva mejora: Limpieza estricta de pseudo-unidades anatómicas LATINAS SOLO si están seguidas de 'de'
    n = re.sub(r'^(pechuga|filete|muslo|trozo|chuleta|pieza|corte|ración|racion|porción|porcion|filetico|medallón|medallones|carne)(s)?\s+(de\s+|del\s+)', '', n, flags=re.IGNORECASE)
    n = re.sub(r'^(de\s+|del\s+)', '', n, flags=re.IGNORECASE)
    
    stops = ['cortado', 'cortada', 'cortados', 'cortadas', 'picado', 'picada', 'picados', 'picadas', 'picadito', 'picadita', 'picaditos', 'picaditas', 'pelado', 'pelada', 'pelados', 'peladas', 'hervido', 'hervida', 'hervidos', 'hervidas', 'cocido', 'cocida', 'cocidos', 'cocidas', 'asado', 'asada', 'asados', 'asadas', 'crudo', 'cruda', 'crudos', 'crudas', 'horneado', 'horneada', 'horneados', 'horneadas', 'desmenuzado', 'desmenuzada', 'desmenuzados', 'desmenuzadas', 'rallado', 'rallada', 'rallados', 'ralladas', 'guisado', 'guisada', 'guisados', 'guisadas', 'frito', 'frita', 'fritos', 'fritas', 'majado', 'majada', 'majados', 'majadas', 'triturado', 'triturada', 'triturados', 'trituradas', 'hecha puré', 'hecho puré', 'puré', 'en julianas', 'en tiras', 'en cubos', 'en hojuelas', 'en dados', 'en aros', 'en trozos', 'en rodajas', 'en porciones', 'en lonjas', 'en lonja', 'finamente', 'muy', 'pequeño', 'pequeña', 'pequeños', 'pequeñas', 'grande', 'grandes', 'mediano', 'mediana', 'medianos', 'medianas', 'maduro', 'madura', 'maduros', 'maduras', 'fresco', 'fresca', 'frescos', 'frescas', 'firme', 'firmes', 'entero', 'entera', 'enteros', 'enteras', 'fina', 'finas', 'gruesa', 'gruesas', 'magro', 'magra', 'magros', 'magras', 'natural', 'naturales', 'bajo en grasa', 'bajas en grasa', 'bajos en grasa', 'bajo en sodio', 'bajas en sodio', 'bajos en sodio', 'descremado', 'descremada', 'descremados', 'descremadas', 'sin sal', 'con sal', 'sin piel', 'sin hueso', 'para rebozar', 'al gusto', 'pizca de', 'rodajas de', 'de la despensa', 'ralladura y jugo de 1/2', 'la', 'el', 'los', 'las']
    clean_n = n
    for s in stops:
        clean_n = re.sub(r'\b' + s + r'\b', '', clean_n, flags=re.IGNORECASE)
        
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

    n_stripped = strip_accents(n)
    clean_n_stripped = strip_accents(clean_n)
    
    # Recolectar todos los aliases + nombres canónicos para búsqueda,
    # ordenados por longitud (más largos primero) para evitar que 
    # 'platano' se trague 'platano maduro' o 'queso' se trague 'queso cottage'
    all_aliases = []
    for master in master_list:
        # El nombre canónico también cuenta como alias para búsqueda exacta
        master_name = master["name"]
        all_aliases.append((strip_accents(master_name.strip().lower()), master_name))
        for alias in (master.get("aliases") or []):
            all_aliases.append((strip_accents(alias.strip().lower()), master_name))
            
    all_aliases.sort(key=lambda x: len(x[0]), reverse=True)

    # ── INTENTO 1: Match Exacto sobre el texto RAW (sin mutilar por stops) ──
    # Esto es CRÍTICO porque los stops eliminan palabras como 'natural', 'descremado',
    # 'bajo en grasa' que son parte de aliases legítimos como 'yogurt griego natural'.
    for alias_stripped, master_name in all_aliases:
        if n_stripped == alias_stripped:
            return master_name

    # ── INTENTO 2: Regex sobre el texto RAW (sin mutilar) ──
    # Buscar "queso mozzarella bajo en grasa" dentro de "queso mozzarella bajo en grasa rallado"
    for alias_stripped, master_name in all_aliases:
        if re.search(r'\b' + re.escape(alias_stripped) + r'\b', n_stripped, flags=re.IGNORECASE):
            return master_name

    # ── INTENTO 3: Match Exacto sobre clean_n (texto limpio, fallback) ──
    for alias_stripped, master_name in all_aliases:
        if clean_n_stripped == alias_stripped:
            return master_name

    # ── INTENTO 4: Regex sobre clean_n (último recurso antes de fuzzy/semántica) ──
    for alias_stripped, master_name in all_aliases:
        if re.search(r'\b' + re.escape(alias_stripped) + r'\b', clean_n_stripped, flags=re.IGNORECASE):
            return master_name

    # ── INTENTO 5 [P4-UNIFIED-RESOLVER · 2026-06-14]: Fuzzy (difflib) ANTES de gastar un embedding.
    # Atrapa typos y variantes menores ("platanno"→"plátano", "yogur griego"→"yogurt griego") que los
    # tiers regex no cubren, sin costo de API. Conservador (ratio ≥ 0.87) para no introducir falsos
    # positivos; los casos semánticos reales (sinónimos no-léxicos) los sigue cubriendo el embedding. ──
    import difflib
    # Formas candidatas a comparar (los strippers de prefijo/stop-words a veces dejan el query corto o
    # le quitan contexto: "platanno maduro"→"platanno", "pechuga de poyo"→"poyo"). Comparamos contra el
    # crudo, el limpio Y el original (solo parens removidos) y tomamos el mejor ratio por alias.
    _orig_fuzz = strip_accents(re.sub(r'\(.*?\)', '', str(orig_name).lower()).strip())
    _fuzz_forms = {f for f in (n_stripped, clean_n_stripped, _orig_fuzz) if f and len(f) > 3}
    if _fuzz_forms:
        _fuzz_best, _fuzz_name = 0.0, None
        for alias_stripped, master_name in all_aliases:
            if not alias_stripped:
                continue
            _r = max(difflib.SequenceMatcher(None, f, alias_stripped).ratio() for f in _fuzz_forms)
            if _r > _fuzz_best:
                _fuzz_best, _fuzz_name = _r, master_name
        if _fuzz_best >= 0.87 and _fuzz_name:
            logging.info(f"🔤 [Fuzzy Match] '{orig_name}' -> '{_fuzz_name}' (ratio {_fuzz_best:.3f})")
            return _fuzz_name

    # Intento 6: Búsqueda de Similitud Semántica Vectorial (Cohere v4, Fallback Local)
    # Solo vale la pena gastar un request si la palabra no fue encontrada en absoluto y tiene suficiente longitud
    if len(n) > 3:
        cache = get_semantic_cache()
        if cache:
            try:
                # Calculamos el vector del texto no reconocido
                query_vector = _gemini_call_with_retry(
                    cache["embeddings_client"].embed_query, n,
                    _label=f"embed_query (semantic match: {n[:40]!r})",
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
    for k, v in fraction_map.items():
        if s_lower.startswith(k):
            s_lower = s_lower.replace(k, v + " ", 1)
            
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
            
    return s.strip()

def _calculate_yield_multiplier(raw_name: str, *, only_legumbres_grains: bool = False) -> float:
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

    if only_legumbres_grains:
        # Modo aggregator: NO aplicar reglas #2-4 para preservar la simetría
        # plan↔inventario establecida en P1-2.
        return 1.0

    # 2. Proteínas cocidas (Se encogen por humedad, necesitas más crudo)
    if bool(re.search(r'\b(cocid[oa]|hervid[oa]|asad[oa]|hornead[oa]|desmenuzad[oa]|frit[oa])\b', n)) and bool(re.search(r'\b(pollo|carne|res|pescado|cerdo|camar|pavo|salm[oó]n|filete)\b', n)):
        return 1.35

    # 3. Merma de Cáscara/Limpieza (Víveres y Mariscos pelados)
    if bool(re.search(r'\b(pelad[oa]|limpi[oa]|sin piel|sin c[aá]scara)\b', n)) and bool(re.search(r'\b(yuca|platano|pl[aá]tano|batata|papa|guineo|camar[oó]n|manzana|pera)\b', n)):
        return 1.30

    # 4. Merma de Hueso (comprar sin hueso es más carne, pero si la receta pide carne magra y el ingrediente en lista es estándar)
    if bool(re.search(r'\b(sin hueso|deshuesad[oa])\b', n)) and bool(re.search(r'\b(pollo|muslo|carne|chuleta)\b', n)):
        return 1.40

    return 1.0

def _parse_quantity(s, *, apply_yield_multiplier: bool = True, apply_legumbres_yield_only: bool = False):
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
    """
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
        yield_mult = _calculate_yield_multiplier(rest_str, only_legumbres_grains=True)
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
        
    return qty, unit_str, normalize_name(rest_str).strip()
    
def get_plural_unit(num, u):
    if num <= 1 or not u: return u
    u_lower = u.lower()
    PLURALS = {
        'lb': 'lbs', 'lbs': 'lbs',
        'paquete': 'paquetes', 'pote': 'potes', 'unidad': 'unidades',
        'lata': 'latas', 'cabeza': 'cabezas', 'diente': 'dientes',
        'cartón': 'cartones', 'carton': 'cartones',
        'sobre': 'sobres', 'sobrecito': 'sobrecitos',
        'botella': 'botellas', 'frasco': 'frascos',
        'fundita': 'funditas', 'mazo': 'mazos', 'envase': 'envases',
        'rebanada': 'rebanadas', 'hoja': 'hojas',
        'cda': 'cdas', 'cdta': 'cdtas', 'taza': 'tazas',
        'ud.': 'Uds.',
    }
    result = PLURALS.get(u_lower, u)
    # Preservar capitalización del input: si "Pote" → "Potes", si "pote" → "potes"
    if len(result) > 0 and u[0].isupper() and result[0].islower():
        result = result[0].upper() + result[1:]
    return result

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


def is_perishable_category(category: str | None, shelf_life_days=None) -> bool:
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
            if frac <= anti_waste_pct or under_buy < over_buy:
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


def apply_smart_market_units(name: str, weight_in_lbs: float, unit_str: str, raw_qty: float, master_item: dict = None):
    """Motor determinístico de unidades de mercado dominicano.
    
    Flujo de resolución (4 bloques, sin hardcoded weights):
      1. DB Container: market_container + container_weight_g → Potes, Paquetes, Cartones, etc.
         1a. SKU-Aware: si hay available_sizes_g, optimiza tamaño de empaque
      2. DB Density:   density_g_per_unit → Unidades físicas (frutas, vegetales, huevos)
      3. Dominican Lbs: Fracciones de libra (1/4, 1/2, 3/4) para carnes, quesos, granel
      4. Raw Fallback:  Cantidades crudas del AI sin conversión
    
    Returns dict con confidence_score (1.0=DB+SKU, 0.95=DB, 0.85=density, 0.75=lbs, 0.5=raw)
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
    if not density_per_u:
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

    # Guards mínimos para Bloques 2 y 3 (solo 2 regex, eliminados los 15+ anteriores)
    is_meat_seafood = bool(re.search(r'\b(pollo|cerdo|carne|res|pescado|camar[oó]n|camarones|mariscos?|filetes?|chuletas?|longanizas?|salamis?|jam[oó]n|pavo|tocineta|bacon|salchichas?)\b', n_lower))
    is_cheese = bool(re.search(r'\b(quesos?|mozzarella|cheddar|parmesano|gouda|dan[eé]s)\b', n_lower)) and not re.search(r'\b(crema|mantequilla)\b', n_lower)

    # Nuevas clasificaciones Nivel de Producción (Actualizado con plurales y más alimentos)
    is_native_countable = bool(re.search(r'\b(pl[aá]tanos?|guineos?|lim[oó]n|limones|huevos?|manzanas?|naranjas?|peras?|chinolas?|mandarinas?|kiwis?|duraznos?)\b', n_lower))
    is_mega_fruit = bool(re.search(r'\b(aguacates?|pi[ñn]as?|sand[ií]as?|mel[oó]n|melones|lechosas?|papayas?)\b', n_lower))
    is_native_weighable = bool(re.search(r'\b(zanahorias?|tomates?|aj[ií]es?|cebollas?|papas?|yucas?|batatas?|berenjenas?|tayotas?|remolachas?|calabac[ií]nes?|calabac[ií]n|auyamas?|vegetales|[ñn]ames?|yaut[ií]as?|pimientos?|chiles?)\b', n_lower))
    is_native_cabeza = bool(re.search(r'\b(br[oó]colis?|coliflor|repollos?|lechugas?)\b', n_lower))
    is_herb_mazo = bool(re.search(r'\b(cilantro|cilantrico|puerro|perejil|menta|albahaca|romero|verdura|verdurita|recao|eneldo)\b', n_lower))

    # ═══════════════════════════════════════════════════════════════
    # BLOQUE 1: Resolución Data-Driven (PRIORIDAD MÁXIMA)
    # Usa market_container + container_weight_g directamente de la DB.
    # Cubre: Lácteos(Pote/Cartón), Despensa(Paquete/Fundita/Botella),
    #         Especias(Sobre), Vegetales(Mazo/Cabeza/Lata), etc.
    # Anti-desperdicio (Ahora estricto): 2% de colchón para errores de coma flotante. 
    # Forzará compras mayores a la mínima escalada matemática (Ej: 4 personas vs 6).
    ANTI_WASTE_THRESHOLD = 0.02

    db_container = master_item.get("market_container")
    db_container_weight_g = master_item.get("container_weight_g")
    available_sizes = master_item.get("available_sizes_g")
    
    if db_container and db_container_weight_g and weight_in_lbs > 0:
        g_total = weight_in_lbs * 453.592
        
        # ── SKU-Aware Path: múltiples tamaños disponibles ──
        if available_sizes and isinstance(available_sizes, list) and len(available_sizes) > 1:
            sku_count, sku_size_g = _find_best_sku(g_total, available_sizes, ANTI_WASTE_THRESHOLD)
            sku_label = _sku_size_label(sku_size_g, db_container)
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
                    if frac <= ANTI_WASTE_THRESHOLD or under_buy_g < over_buy_g:
                        units_needed = floor_units
                    else:
                        units_needed = floor_units + 1
                else:
                    units_needed = max(1, math.ceil(raw_units))
                
                sku_label = _sku_size_label(container_weight_g, db_container)
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
                    whole = math.floor(lbs_for_weighable)
                    frac_w = lbs_for_weighable - whole
                    fraction_str = ""
                    if frac_w < 0.15: fraction_str = ""
                    elif frac_w <= 0.35: fraction_str = "1/4"
                    elif frac_w <= 0.65: fraction_str = "1/2"
                    elif frac_w <= 0.85: fraction_str = "3/4"
                    else: 
                        fraction_str = ""
                        whole += 1
                        
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

    result = {
        "name": name,
        "market_qty": formatted_market_qty,
        "market_qty_numeric": market_qty_numeric_final,
        "market_unit": market_unit,
        "display_qty": display_qty_final,
        "display_string": final_str,
        "confidence_score": confidence,
        "shelf_life_days": master_item.get("shelf_life_days") if master_item else None
    }
    if sku_label:
        result["sku_size_label"] = sku_label
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


def expected_sum_from_recipes(plan_data: dict, *, apply_yield: bool = False, multiplier: float = 1.0) -> dict:
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

    Returns:
        `{food_name: {canonical_unit: total_qty}}`. Vacío si no hay días.
    """
    if not isinstance(plan_data, dict):
        return {}
    days = plan_data.get("days")
    if not days or not isinstance(days, list):
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
                )
                if not name:
                    continue
                if name.lower() in ("ola", "olas"):
                    name = "Cebolla"
                aggregated[name][unit] += float(qty) * _mult

    return {name: dict(units) for name, units in aggregated.items()}


def _classify_divergence_hypothesis(
    exp_qty: float,
    act_qty: float,
    exp_units: dict,
    act_units: dict,
    food: str = "",
) -> str:
    """Heurístico de clasificación para `compare_expected_vs_aggregated`.

    Las hipótesis son orientativas para el reviewer humano/operacional; no
    sustituyen verificación. Orden de precedencia:
      1. cap_swallowed_modifier > 2. unit_mismatch > 3. yield_uncovered
      4. pantry_overdeduct > 5. unknown.

    [P2-AUDIT-1 · 2026-05-10] `food` opcional (default ''): cuando se provee
    y resuelve a pescado/mariscos vía `canonicalize_fish_seafood`, se usan
    bandas yield más estrechas (cooking loss menor que carnes rojas/blancas).
    Backward-compat: callers que no pasen `food` siguen con las bandas
    clásicas de carne/legumbre.
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
    if exp_qty > 0 and 0 < act_qty < exp_qty * overdeduct_threshold:
        return "pantry_overdeduct"

    return "unknown"


def compare_expected_vs_aggregated(
    expected: dict,
    aggregated: dict,
    *,
    tolerance: float = 0.05,
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
        pantry_overdeduct · unknown.
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
                divergences.append({
                    "food": food,
                    "unit": unit,
                    "expected_qty": 0.0,
                    "actual_qty": act_qty,
                    "delta_pct": float("inf"),
                    "hypothesis": _classify_divergence_hypothesis(exp_qty, act_qty, exp_units, act_units, food=food),
                })
                continue

            delta_pct = abs(act_qty - exp_qty) / exp_qty
            if delta_pct > tolerance:
                divergences.append({
                    "food": food,
                    "unit": unit,
                    "expected_qty": exp_qty,
                    "actual_qty": act_qty,
                    "delta_pct": delta_pct,
                    "hypothesis": _classify_divergence_hypothesis(exp_qty, act_qty, exp_units, act_units, food=food),
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
    """
    if not divergences:
        return False
    for d in divergences:
        if not isinstance(d, dict):
            continue
        if d.get("hypothesis") == "cap_swallowed_modifier":
            return True
        if d.get("magnitude") is True:
            try:
                delta = float(d.get("delta_pct") or 0)
            except (TypeError, ValueError):
                delta = 0.0
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
    """
    if not divergences:
        return []
    out = []
    for d in divergences:
        if not isinstance(d, dict):
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
        try:
            qty = float(item.get("market_qty_numeric") or item.get("quantity") or 0)
        except (TypeError, ValueError):
            qty = 0.0
        if qty <= 0:
            continue
        unit = str(item.get("market_unit") or item.get("unit") or "").strip().lower()
        if not unit:
            unit = "unidad"
        out[name_str][unit] += qty
    return {n: dict(u) for n, u in out.items()}


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
    if re.search(r'\bleche\s+de\s+(coco|almendra|soja|soya|avena)\b', n_low):
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
_CAPS_APPLIED_LAST_RUN: list = []


def reset_caps_applied_last_run() -> None:
    """[P1-CAPS-COHERENCE-RECONCILE · 2026-05-16] Limpia el tracker antes de
    cada nuevo run de `aggregate_and_deduct_shopping_list`. Sin esto, runs
    consecutivos acumulan caps de ejecuciones previas → el guard ve "caps
    fantasmas" que no aplicaron a este plan."""
    _CAPS_APPLIED_LAST_RUN.clear()


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

    try:
        expected_raw = expected_sum_from_recipes(plan_result, apply_yield=False, multiplier=mult)
    except Exception as e:
        logging.warning(f"[COH-GUARD] expected_sum_from_recipes falló: {e}")
        return []

    aggregated_list = plan_result.get("aggregated_shopping_list") or []
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
            raw_mags = compare_expected_vs_aggregated(
                expected_canonical,
                aggregated_canonical,
                tolerance=tolerance_pct,
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

    divergences.extend(magnitude_divs)

    if divergences:
        from collections import Counter
        by_hyp = Counter(d["hypothesis"] for d in divergences)
        sample = "; ".join(f"{d['food']} [{d['side']}]" for d in divergences[:6])
        logging.warning(
            f"🛒 [COH-GUARD/{mode}] {len(divergences)} divergencias "
            f"(presence={len(missing_in_agg)+len(extra_in_agg)}, magnitude={len(magnitude_divs)}, "
            f"multiplier={mult}). Hipótesis: {dict(by_hyp)}. Sample: {sample}"
        )
        if mode == "block":
            critical = []
            # Crítico A: foods de receta ausentes en lista (presence).
            critical.extend(d for d in divergences if d["side"] == "expected_only")
            # Crítico B: divergencias de magnitud con delta finito > tolerance
            # (excluir fantasmas con delta=inf — pueden ser staples no marcados).
            critical.extend(
                d for d in magnitude_divs
                if d.get("delta_pct") != float("inf") and d.get("expected_qty", 0) > 0
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
    """[P2-COHERENCE-GUARD-PERF · 2026-05-15] Best-effort INSERT a
    `pipeline_metrics` con perf del coherence guard. Knob umbral:
    `MEALFIT_COHERENCE_GUARD_SLOW_MS` (default 1000) — log warning si
    excede para que un refactor accidental O(n²) sea detectable sin
    esperar a tail-latency en user-facing.
    """
    try:
        import os as _os_coh
        try:
            _slow_threshold_ms = int(_os_coh.environ.get("MEALFIT_COHERENCE_GUARD_SLOW_MS", "1000"))
        except (TypeError, ValueError):
            _slow_threshold_ms = 1000

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


def aggregate_and_deduct_shopping_list(plan_ingredients: list[str], consumed_ingredients: list[str] = None, categorize: bool = False, structured: bool = False, multiplier: float = 1.0):
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
        qty, unit, name = _parse_quantity(item, apply_yield_multiplier=False, apply_legumbres_yield_only=True)
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
        aggregated[name][unit] -= float(qty)  # P2-NEW-11: SIN multiplier (ver contrato arriba)

    # --- RESOLUCIÓN DE FRICCIÓN DE UNIDADES (Híbridas) ---
    master_list = get_master_ingredients()
    # Mapeo por nombre canónico + aliases para resolución robusta
    master_map = {}
    for m in master_list:
        master_map[m["name"]] = m
        # Indexar todos los aliases para resolución fuzzy
        for alias in (m.get("aliases") or []):
            master_map[alias.strip().lower()] = m
            # También indexar con capitalización Title
            master_map[alias.strip().title()] = m

    # ── Re-agrupación por Nombre Canónico ──
    # Si el LLM devolvió "Huevo", "Huevos" y "Huevos enteros", el agregador original
    # los tiene como 3 llaves. Aquí los fusionamos en la llave canónica oficial ("Huevos")
    # para que su volumen se sume correctamente antes de calcular empaques comerciales.
    canonical_aggregated = defaultdict(lambda: defaultdict(float))
    for name, units in aggregated.items():
        m_item = master_map.get(name) or master_map.get(name.lower()) or master_map.get(name.title())
        canonical_name = m_item["name"] if m_item else name

        # [P2-NEW-8 · 2026-05-11] SSOT: las 4 reglas inline Huevo/Ñame/Miel/Ajo
        # ahora viven en `_consolidate_inline_canon` (paralelo a la llamada en
        # `_canonicalize_for_coherence`). Sin SSOT, un drift entre los dos
        # sitios producía false positives del guard recetas↔lista.
        _inline_canon = _consolidate_inline_canon(canonical_name)
        if _inline_canon is not None:
            canonical_name = _inline_canon
        else:
            # [P3-NEW-6 · 2026-05-11] Víveres y musáceas: consolidar variantes
            # del mismo producto en una sola línea de shopping. "Yuca hervida"
            # + "Yuca con mojo" → 1 línea "Yuca". "Plátano verde" + "Plátano
            # maduro" → 1 línea "Plátano" (madurez es estado temporal, no
            # producto distinto). Bilateral con el guard (mirror en
            # `_canonicalize_for_coherence`).
            _viv = canonicalize_viveres(canonical_name)
            if _viv is not None:
                canonical_name = _viv
            else:
                _mus = canonicalize_musaceae(canonical_name)
                if _mus is not None:
                    canonical_name = _mus
                else:
                    # [P2-NEW-A · 2026-05-11] Frutas tropicales / verduras de
                    # hoja / aceites: tres familias más cuyas variantes inflan
                    # la lista si no consolidamos. Mismo orden y patrón que el
                    # guard (mirror): primer match gana, cada uno mutex.
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
                                # [P3-NEW-12 · 2026-05-11] 5 canonicalizers
                                # adicionales (cítricos, tomate, cebolla,
                                # quesos blancos RD, frutos secos). Mismo
                                # patrón mirror que P2-NEW-A. Sin estos,
                                # variantes triviales como "limón verde" vs
                                # "limón persa" o "tomate criollo" vs
                                # "tomate maduro" se quedan en líneas
                                # separadas en la lista de compras.
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

        # [P3-PROTEIN-CAP-2] Consolidación de pavo con distinción
        # FRESH vs PROCESADO. Antes la regla colapsaba CUALQUIER
        # "pavo + (pechuga|lonjas|rebanada|picadito)" a "Jamón de pavo",
        # tratando pechuga FRESCA como deli procesado. Diferencia real
        # crítica para el usuario:
        #   - Pechuga fresca: ~$80 RD$/lb, sodio bajo, proteína magra clean.
        #   - Jamón de pavo en lonjas: ~$150 RD$/lb, sodio 4× mayor,
        #     contiene nitritos y conservantes.
        # Caso real 2026-05-05 02:14: LLM pidió "pechuga de pavo fresca"
        # → aggregator mostraba "27 lbs de Jamón de pavo" en la lista
        # → usuario compraría producto equivocado, costo +90% y nutrición
        # peor.
        #
        # Reglas, en orden de precedencia:
        #   1. Marker EXPLÍCITO de fresh (`fresca`, `fresh`) → Pechuga de pavo
        #      (gana sobre cualquier indicador de presentación)
        #   2. Marker explícito de procesado (`jamón de pavo`, `pavo en
        #      lonjas`, `pavo procesado`) sin fresh → Jamón de pavo
        #   3. `pavo molido` o `carne de pavo` → Pavo molido (lean ground,
        #      no procesado deli)
        #   4. `pechuga de pavo` o `filete de pavo` (sin marker explícito de
        #      procesado) → Pechuga de pavo (default seguro: en RD el
        #      consumidor que dice "pechuga de pavo" usualmente quiere fresca)
        #   5. Else: deja canonical_name del master_map sin tocar.
        # Usar SOLO `name.lower()` (raw del parser) para el matching, NO
        # `_can_lower` (post-master_map). Razón: master_map puede tener
        # alias que canonicaliza "Pechuga de pavo" → "Jamón de pavo"
        # (alias en PROTEIN_SYNONYMS / DB). Si hiciéramos matching sobre
        # `_can_lower`, la regex `jam[oó]n de pavo` se autoactivaría aunque
        # el LLM dijo "pechuga fresca", produciendo la conflación que
        # justamente queremos evitar. El raw name preserva la intención
        # original del LLM/usuario.
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
                # Default seguro: pechuga de pavo (sin marker procesado) → fresca
                canonical_name = 'Pechuga de pavo'
            elif _orig_name_lower.strip() == 'pavo':
                # "Pavo" solo, sin descriptores → canonical genérico (no
                # auto-conflate via master alias a Jamón de pavo).
                canonical_name = 'Pavo'

        # [P0-SHOPPING-CALC-NAMEERROR · 2026-05-15] `_can_lower` se usa en
        # las 13 regex de consolidación de abajo (Fresas, Almendras, Orégano,
        # Tortilla, Tomate, Cebolla, Espinacas, Zanahoria, Vainitas,
        # Habichuelas, Tofu, Perejil). Pre-fix la variable nunca se asignaba
        # en este scope → `NameError: name '_can_lower' is not defined` en
        # cada plan generado, lo que tumbaba toda la agregación
        # (`aggregate_and_deduct_shopping_list` lanzaba) y dejaba la lista
        # de compras vacía/incompleta. Síntoma user-facing: coherence guard
        # reportaba 35 "divergencias críticas" (todos los ingredientes de
        # las recetas marcados como `presence=expected_only`) y el plan
        # llegaba con `_shopping_coherence_block` no resuelto.
        # IMPORTANTE: se calcula DESPUÉS del bloque pavo porque el pavo
        # puede mutar canonical_name; las 13 regex de abajo necesitan ver
        # el canonical_name post-pavo.
        _can_lower = canonical_name.lower()

        # Consolidación: Fresas variantes (congeladas, frescas) → Fresas
        if re.search(r'^fresas?\b', _can_lower):
            canonical_name = 'Fresas'

        # Consolidación: Almendras variantes → Almendras fileteadas
        if re.search(r'^almendras?\b', _can_lower) and 'mantequilla' not in _can_lower:
            canonical_name = 'Almendras fileteadas'

        # Consolidación: Orégano variantes (seco, dominicano) → Orégano dominicano
        if re.search(r'^or[eé]gano\b', _can_lower):
            canonical_name = 'Orégano dominicano'

        # Consolidación: Tortilla/Tortillas integral/integrales → Tortilla integral
        if re.search(r'^tortillas?\s+integral', _can_lower):
            canonical_name = 'Tortilla integral'

        # Consolidación: Tomate variantes (de ensalada, bugalú, sin semillas, etc.) → Tomate
        if re.search(r'^tomates?\b', _can_lower) and 'pasta' not in _can_lower and 'salsa' not in _can_lower:
            canonical_name = 'Tomate'

        # Consolidación: Cebolla variantes (blanca, roja, morada) → Cebolla
        if re.search(r'^cebollas?\s+(blanca|roja|morada|amarilla)', _can_lower):
            canonical_name = 'Cebolla'

        # Consolidación: Espinaca/Espinacas → Espinacas
        if re.search(r'^espinacas?$', _can_lower):
            canonical_name = 'Espinacas'

        # Consolidación: Zanahoria/Zanahorias → Zanahoria
        if re.search(r'^zanahorias?$', _can_lower):
            canonical_name = 'Zanahoria'

        # Consolidación: Vainita/Vainitas → Vainitas
        if re.search(r'^vainitas?$', _can_lower):
            canonical_name = 'Vainitas'

        # Consolidación: Habichuela variantes sin adjetivo (solo habichuela/habichuelas) → Habichuelas
        if re.search(r'^habichuelas?$', _can_lower):
            canonical_name = 'Habichuelas'

        # Consolidación: Tofu variantes (ahumado, firme, suave) → Tofu
        if re.search(r'^tofu\b', _can_lower):
            canonical_name = 'Tofu'

        # Consolidación: Perejil variantes → Perejil
        if re.search(r'\bperejil\b', _can_lower):
            canonical_name = 'Perejil'

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
            
            if g_per_u <= 0:
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
                if u_qty <= 6:
                    units['cartón (6 uds.)'] = units.get('cartón (6 uds.)', 0) + 1
                elif u_qty <= 15:
                    units['medio cartón (15 uds.)'] = units.get('medio cartón (15 uds.)', 0) + 1
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
    }
    _HERB_MAZO_GRAMS = 50.0  # 1 mazo de hierba ≈ 50g

    _person_weeks = max(1.0, float(multiplier) * 3.0 / 7.0)
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
            logging.warning(
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
            logging.warning(
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
    _OLIVE_FRASCO_GRAMS = 340.194  # 12 oz frasco estándar dominicano
    _WEIGHT_UNIT_TO_G = {
        'g': 1.0, 'kg': 1000.0, 'oz': 28.3495,
        'lb': 453.592, 'lbs': 453.592, 'ml': 1.0, 'l': 1000.0,
    }

    _olive_cap_frascos = max(1, int(round(_person_weeks / 3.0)))
    _olive_cap_g = _olive_cap_frascos * _OLIVE_FRASCO_GRAMS

    for _name, _units in list(aggregated.items()):
        _name_norm = strip_accents(_name.lower()).strip()
        if not any(s in _name_norm for s in _OLIVE_SUBSTRINGS):
            continue
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
                logging.warning(
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
            logging.warning(
                f"[P5-OLIVE-CAP] '{_name}' peso total cap: {_total_weight_g:.0f}g "
                f"(de {_present_units}) → {_olive_cap_g:.0f}g "
                f"(≈{_olive_cap_frascos} frascos 12oz; "
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
                logging.warning(
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
            logging.warning(
                f"[P5-OLIVE-CAP] '{_name}' volumétrico cap: {_vol_total_g:.0f}g "
                f"(de {_vol_present}) → {_olive_cap_g:.0f}g "
                f"(≈{_olive_cap_frascos} frascos 12oz; "
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
                logging.warning(
                    f"[P6-CITRUS-CAP] '{_name}' {_unit_key} cap: {_old:.1f} → "
                    f"{_citrus_cap_units} (person_weeks={_person_weeks:.1f}; "
                    f"~{_per_week}/persona/semana es uso intensivo realista)"
                )
        if 'g' in _units and _units['g'] > _citrus_cap_g:
            _old_g = _units['g']
            _units['g'] = float(_citrus_cap_g)
            _record_cap_applied(_name, _old_g, _units['g'], "P6-CITRUS-CAP")
            logging.warning(
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
                logging.warning(
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
                logging.warning(
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
                logging.warning(
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
            logging.warning(
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
                logging.warning(
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
            logging.warning(
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
                logging.warning(
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
            logging.warning(
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
                logging.warning(
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
            logging.warning(
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
                logging.warning(
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
            logging.warning(
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
                logging.warning(
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
            logging.warning(
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
                logging.warning(
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
            logging.warning(
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
                logging.warning(
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
                logging.warning(
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
            logging.warning(
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
                logging.warning(
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
                logging.warning(
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
            logging.warning(
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
                logging.warning(
                    f"[P6-FRUITS-PERISHABLE-CAP] '{_name}' {_unit_key} cap: "
                    f"{_old:.1f} → {_cap_lbs:.0f} lbs"
                )
                _record_cap_applied(_name, _old, _units[_unit_key], "P6-FRUITS-PERISHABLE-CAP")
        # Cap por paquetes (1 paquete = 1 lb estándar dominicano)
        for _unit_key in ('paquete', 'paquetes'):
            if _unit_key in _units and _units[_unit_key] > _cap_lbs:
                _old = _units[_unit_key]
                _units[_unit_key] = float(_cap_lbs)
                logging.warning(
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
            logging.warning(
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
                logging.warning(
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
                logging.warning(
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
            logging.warning(
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
                logging.warning(
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

        weight_in_lbs = 0.0
        has_weight = False
        cat = master_item.get("category") or "Otros"
        display_cat = _get_display_category(cat, name)

        price_per_lb = float(master_item.get("price_per_lb", 0) or 0)
        price_per_unit = float(master_item.get("price_per_unit", 0) or 0)

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
                item_cost = weight_in_lbs * price_per_lb
                total_estimated_cost += item_cost
                market_obj = apply_smart_market_units(name, weight_in_lbs, 'lb', 0.0, master_item)
                market_obj["category"] = cat
                market_obj["display_category"] = display_cat
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
                    _cat_for_perish, market_obj.get("shelf_life_days")
                )
                # [P3-PRICE-UNIT-COVERAGE · 2026-06-20] Si el costo quedó en 0 (ítem que se vende por ENVASE:
                # price_per_unit>0, price_per_lb=0 — aceite/miel/huevo/yogurt entran por peso y daban 0),
                # costear desde el conteo de envases del DISPLAY (market_qty = lo que el usuario REALMENTE compra:
                # 1 botella, 1 cartón, 2 Ud) × price_per_unit. Usa market_qty (no el conteo crudo, que puede estar
                # en otra unidad que el precio: 30 huevos vs precio por cartón). Cierra ~40% de ítems caros sin precio.
                if item_cost <= 0 and price_per_unit > 0:
                    try:
                        _mq_disp = float(market_obj.get("market_qty") or 0)
                    except (TypeError, ValueError):
                        _mq_disp = 0.0
                    if _mq_disp > 0:
                        item_cost = _mq_disp * price_per_unit
                        total_estimated_cost += item_cost
                market_obj["estimated_cost_rd"] = round(item_cost, 2) if item_cost > 0 else None
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
                item_cost = 0.0
                if u in ['unidad', 'unidades', 'lata', 'latas', 'paquete', 'paquetes']:
                    item_cost = q * price_per_unit
                    total_estimated_cost += item_cost
                market_obj = apply_smart_market_units(name, 0.0, u, q, master_item)
                market_obj["category"] = cat
                market_obj["display_category"] = display_cat
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
                    _cat_for_perish, market_obj.get("shelf_life_days")
                )
                # [P3-PRICE-UNIT-COVERAGE · 2026-06-20] Si el costo quedó en 0 (ítem que se vende por ENVASE:
                # price_per_unit>0, price_per_lb=0 — aceite/miel/huevo/yogurt entran por peso y daban 0),
                # costear desde el conteo de envases del DISPLAY (market_qty = lo que el usuario REALMENTE compra:
                # 1 botella, 1 cartón, 2 Ud) × price_per_unit. Usa market_qty (no el conteo crudo, que puede estar
                # en otra unidad que el precio: 30 huevos vs precio por cartón). Cierra ~40% de ítems caros sin precio.
                if item_cost <= 0 and price_per_unit > 0:
                    try:
                        _mq_disp = float(market_obj.get("market_qty") or 0)
                    except (TypeError, ValueError):
                        _mq_disp = 0.0
                    if _mq_disp > 0:
                        item_cost = _mq_disp * price_per_unit
                        total_estimated_cost += item_cost
                market_obj["estimated_cost_rd"] = round(item_cost, 2) if item_cost > 0 else None
                item_val = market_obj if structured else market_obj["display_string"]
                results.append(item_val)
                categorized_results[display_cat].append(item_val)
                added = True
                
        # Removido el PANTRY_STAPLES force-add ("Disponible"). 
        # Si un alimento (incluyendo los estables) se deduce al 100%, 
        # no debe irrumpir en la lista de compras del supermercado.

    results.sort(key=lambda x: x["display_string"] if structured else x)
    
    result_names = [r["name"] if structured and isinstance(r, dict) else str(r) for r in results]
    logging.info(f"🛒 [AGGREGATE FINAL] {len(results)} output items: {result_names[:20]}...")
    
    if categorize:
        for k in categorized_results:
            categorized_results[k].sort(key=lambda x: x["display_string"] if structured else x)
        return dict(categorized_results)
        
    return results

def aggregate_shopping_list(ingredients_list: list[str]) -> list[str]:
    return aggregate_and_deduct_shopping_list(ingredients_list, [])

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
):
    """Calcula el verdadero Delta: Ingredientes Totales del Plan - Inventario Físico Actual - (Opcional) Consumidos.

    [P1-5] Si el caller necesita N multiplicidades del mismo plan (típico:
    weekly/biweekly/monthly), debe llamar `fetch_inventory_and_consumed_for_plan`
    UNA vez y pasar el resultado vía `inventory_override` + `consumed_override`.
    Sin estos overrides, cada invocación re-consulta `user_inventory` (que
    puede cambiar entre llamadas por Realtime channel, restock, cron) y
    `consumed_meals_since` — produciendo deltas inconsistentes que el
    frontend muestra al usuario al cambiar `groceryDuration`.
    """
    all_ingredients = []
    days = plan_result.get("days", [])
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


    meal_count = 0
    for day in days:
        for meal in day.get("meals", []):
            # [P2-4] SSOT: helper compartido con expected_sum_from_recipes y
            # el extractor de facts. Garantiza simetría capa-B del coherence
            # guard (expected ≡ aggregated en cuanto a qué meals contribuyen).
            if _should_skip_meal_for_aggregation(meal):
                continue
            meal_count += 1
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
                    all_ingredients.append(i)
                elif isinstance(i, dict):
                    q = i.get("quantity", 0)
                    u = i.get("unit", "unidad")
                    n = i.get("name") or i.get("item_name") or i.get("display_name") or "Desconocido"
                    if q > 0 or u in ['pizca', 'al gusto', 'cantidad necesaria', 'chin', 'toque', 'chorrito']:
                        all_ingredients.append(f"{q} {u} de {n}")
                    else:
                        all_ingredients.append(n)
                    
    logging.info(f"🛒 [SHOPPING EXTRACT] {len(days)} days, {meal_count} meals, {len(all_ingredients)} raw ingredients")

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
        
    res = aggregate_and_deduct_shopping_list(all_ingredients, items_to_deduct, categorize=categorize, structured=structured, multiplier=effective_multiplier)
    
    # [P0-3] Inyectar items de compra urgente si el plan superó validación de despensa en flexible_mode
    urgent_items = plan_result.get("_pantry_supplement_required", [])
    if urgent_items:
        if categorize:
            if isinstance(res, dict):
                res["🚨 Compra Urgente"] = []
                for item in urgent_items:
                    res["🚨 Compra Urgente"].append({
                        "name": item,
                        "market_qty": 1,
                        # [P0-2] Espejo numérico siempre presente para que el
                        # frontend nunca tenga que parsear `market_qty` (que
                        # en otros items puede venir como "1 1/2"/"1/2").
                        "market_qty_numeric": 1.0,
                        "market_unit": "ud",
                        "display_qty": item,
                        "display_string": f"⚠️ {item}",
                        "category": "🚨 Compra Urgente",
                        "display_category": "🚨 Compra Urgente",
                        "is_staple": False,
                        # [P1-PDF-2] Items urgentes son siempre perecederos
                        # ("comprar pronto"). El helper también lo deriva por
                        # substring "urgente" pero lo marcamos explícito para
                        # robustez (independiente de cualquier renombre futuro).
                        "is_perishable": True,
                    } if structured else f"⚠️ {item}")
        else:
            if isinstance(res, list):
                for item in urgent_items:
                    res.append({
                        "name": item,
                        "market_qty": 1,
                        # [P0-2] Espejo numérico — ver comentario equivalente arriba.
                        "market_qty_numeric": 1.0,
                        "market_unit": "ud",
                        "display_qty": item,
                        "display_string": f"⚠️ {item}",
                        "category": "🚨 Compra Urgente",
                        "display_category": "🚨 Compra Urgente",
                        "is_staple": False,
                        # [P1-PDF-2] Ver comentario equivalente arriba.
                        "is_perishable": True,
                    } if structured else f"⚠️ {item}")
    
    return res

def get_realtime_pantry(plan_result: dict, consumed_ingredients: list[str]) -> list[str]:
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
                    
    return aggregate_and_deduct_shopping_list(all_ingredients, consumed_ingredients)
