import os
import json
import hashlib
import functools
import threading

# Intentar importar redis
try:
    import redis
    from dotenv import load_dotenv
    load_dotenv()
    
    REDIS_URL = os.environ.get("REDIS_URL")
    if REDIS_URL:
        # Usamos decode_responses=True para que devuelva strings en lugar de bytes
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        print("🔗 [CACHE] Conectado a Redis exitosamente.")
    else:
        redis_client = None
except ImportError:
    redis_client = None
    print("⚠️ [CACHE] Redis no instalado, se usará caché en memoria local.")
except Exception as e:
    redis_client = None
    print(f"⚠️ [CACHE] Error conectando a Redis, se usará caché local: {e}")

# Fallback local cache (LRU artesanal simple)
_local_cache = {}
_cache_lock = threading.Lock()

def centralized_cache(ttl_seconds=3600, maxsize=1000):
    """
    Decorador que intenta usar Redis (distribuido y compartido entre workers).
    Si Redis no está configurado, hace fallback a una caché local en memoria.
    Asume que los argumentos y el valor de retorno de la función decorada
    son 100% serializables a JSON.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 1. Crear un Hash de los argumentos
            # Convertir tuplas a listas para json.dumps
            try:
                args_repr = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
            except Exception as j_err:
                print(f"⚠️ [CACHE] Argumentos no serializables en {func.__name__}, saltando caché: {j_err}")
                return func(*args, **kwargs)

            # Usar MD5 (es suficientemente rápido para keys de caché cortas)
            args_hash = hashlib.md5(args_repr.encode("utf-8")).hexdigest()
            cache_key = f"{func.__name__}:{args_hash}"
            
            # 2. Intentar buscar en Redis
            if redis_client:
                try:
                    cached_val = redis_client.get(cache_key)
                    if cached_val:
                        return json.loads(cached_val)
                except Exception as e:
                    print(f"⚠️ [CACHE] Error Leyendo de Redis para {cache_key}: {e}")
            
            # 3. Intentar buscar en Local Cache (Fallback)
            else:
                with _cache_lock:
                    if cache_key in _local_cache:
                        # Aunque sea un diccionario eterno sin un TTL estricto cronológico,
                        # protegemos contra desbordamiento con maxsize.
                        return _local_cache[cache_key]

            # 4. Cache Miss: Ejecutar función pesada original
            result = func(*args, **kwargs)

            # 5. Guardar en Caché
            if redis_client:
                try:
                    redis_client.setex(cache_key, ttl_seconds, json.dumps(result))
                except Exception as e:
                    print(f"⚠️ [CACHE] Error Escribiendo a Redis para {cache_key}: {e}")
            else:
                with _cache_lock:
                    _local_cache[cache_key] = result
                    # Lógica LRU pobre (Evicción completa si pasa el maxsize para no tumbar la RAM)
                    if len(_local_cache) > maxsize:
                        _local_cache.clear()
                        _local_cache[cache_key] = result
                        print("🧹 [CACHE LOCAL] Limpieza de caché local forzada (capacidad alcanzada).")
                        
            return result
        return wrapper
    return decorator
