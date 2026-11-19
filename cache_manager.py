import os
import asyncio
import json
import hashlib
import functools
import threading
import time as _time_mod
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Intentar importar redis
try:
    import redis
    import redis.asyncio as redis_async
    from dotenv import load_dotenv
    load_dotenv()
    
    REDIS_URL = os.environ.get("REDIS_URL")
    if REDIS_URL:
        # Usamos decode_responses=True + timeouts cortos para fail-fast a caché local si Redis cuelga.
        redis_client = redis.from_url(
            REDIS_URL, decode_responses=True, socket_timeout=2.0, socket_connect_timeout=2.0
        )
        redis_async_client = redis_async.from_url(
            REDIS_URL, decode_responses=True, socket_timeout=2.0, socket_connect_timeout=2.0
        )
        logger.info("🔗 [CACHE] Conectado a Redis exitosamente (Sync y Async).")
    else:
        redis_client = None
        redis_async_client = None
except ImportError:
    redis_client = None
    redis_async_client = None
    logger.warning("⚠️ [CACHE] Redis no instalado, se usará caché en memoria local.")
except Exception as e:
    redis_client = None
    redis_async_client = None
    logger.warning(f"⚠️ [CACHE] Error conectando a Redis, se usará caché local: {e}")


def get_redis_async():
    """[P2-REDIS-ASYNC-PERLOOP · 2026-06-17] Cliente `redis.asyncio` ligado al
    event-loop ACTUAL.

    `redis.asyncio` cachea sus conexiones en el event-loop que primero las await-ea.
    El cliente module-global `redis_async_client` se crea al import (un loop), pero
    la generación de planes corre en SU PROPIO loop (vía run_until_complete/asyncio.run
    desde el handler SSE), que luego se cierra. Reusar el cliente global en un loop
    distinto/muerto rompe con `RuntimeError: Event loop is closed` (observado en el
    semáforo per-usuario, que entonces caía a fallback local).

    Fix: un cliente por loop, cacheado como atributo del PROPIO loop → su ciclo de
    vida se ata al loop (sin leak, sin reuse de id()). Si no hay REDIS_URL, no hay
    loop activo, o falla la creación → devuelve None y el caller usa su fallback
    local (mismo comportamiento que antes, pero ahora el path Redis SÍ funciona en
    el loop de generación). Fail-soft: nunca lanza.
    """
    _ra = globals().get("redis_async")
    _url = globals().get("REDIS_URL")
    if _ra is None or not _url:
        return None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    client = getattr(loop, "_mf_redis_async_client", None)
    if client is None:
        try:
            client = _ra.from_url(
                _url, decode_responses=True, socket_timeout=2.0, socket_connect_timeout=2.0
            )
        except Exception as _e:
            logger.warning(f"⚠️ [CACHE] get_redis_async: no se pudo crear cliente per-loop: {_e}")
            return None
        try:
            setattr(loop, "_mf_redis_async_client", client)
        except Exception:
            pass  # uvloop/C-loops pueden prohibir atributos; el cliente igual sirve (sin cache)
    return client

# Fallback local cache (LRU con TTL real utilizando OrderedDict)
_local_cache = OrderedDict()
_cache_lock = threading.Lock()

def centralized_cache(ttl_seconds=3600, maxsize=1000, cache_empty=True):
    """
    Decorador que intenta usar Redis (distribuido y compartido entre workers).
    Si Redis no está configurado, hace fallback a una caché local en memoria CON TTL real.
    Asume que los argumentos y el valor de retorno de la función decorada
    son 100% serializables a JSON.

    [P2-EMBED-NO-CACHE-EMPTY · 2026-05-30] `cache_empty` (default True =
    comportamiento histórico): si es False, un resultado FALSY (`[]`, `None`,
    `{}`, `""`, `0`) NO se persiste en caché — se devuelve directo y el próximo
    llamado re-ejecuta la función. Para funciones cuyo valor falsy representa un
    FALLO transitorio (embeddings que caen a `return []` por timeout/429), esto
    evita envenenar el caché con el fallo bajo TTL largo. NO usar en funciones
    cuyo `[]`/`{}` vacío sea un resultado legítimo y caro de recomputar.
    """
    def decorator(func):
        _func_local_cache = OrderedDict()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 1. Crear un Hash de los argumentos
            try:
                args_repr = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
            except Exception as j_err:
                logger.warning(f"⚠️ [CACHE] Argumentos no serializables en {func.__name__}, saltando caché: {j_err}")
                return func(*args, **kwargs)

            args_hash = hashlib.md5(args_repr.encode("utf-8")).hexdigest()
            cache_key = f"{func.__name__}:{args_hash}"  # misma clave que antes: no tirar la caché de visión en Redis
            
            # 2. Intentar buscar en Redis; si falla o no está, caer a caché local
            if redis_client:
                try:
                    cached_val = redis_client.get(cache_key)
                    if cached_val is not None:
                        return json.loads(cached_val)
                except Exception as e:
                    logger.warning(f"⚠️ [CACHE] Error Leyendo de Redis para {cache_key}: {e}")
            
            with _cache_lock:
                if cache_key in _func_local_cache:
                    stored_time, stored_json = _func_local_cache[cache_key]
                    if (_time_mod.time() - stored_time) < ttl_seconds:
                        _func_local_cache.move_to_end(cache_key)
                        return json.loads(stored_json)
                    else:
                        _func_local_cache.pop(cache_key, None)

            # 4. Cache Miss: Ejecutar función pesada original
            result = func(*args, **kwargs)

            if not cache_empty and not result:
                return result

            # 5. Guardar en Caché (serializar una sola vez para Redis y copia inmutable local)
            result_json = json.dumps(result)
            if redis_client:
                try:
                    redis_client.setex(cache_key, ttl_seconds, result_json)
                    return result
                except Exception as e:
                    logger.warning(f"⚠️ [CACHE] Error Escribiendo a Redis para {cache_key}: {e}")

            with _cache_lock:
                _func_local_cache[cache_key] = (_time_mod.time(), result_json)
                _func_local_cache.move_to_end(cache_key)
                while len(_func_local_cache) > maxsize:
                    _func_local_cache.popitem(last=False)
                        
            return result
        return wrapper
    return decorator

