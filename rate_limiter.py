import time as _time
from collections import defaultdict as _defaultdict
from typing import Optional
from fastapi import Depends, HTTPException
import logging
from cache_manager import redis_client

from auth import get_verified_user_id

logger = logging.getLogger(__name__)

# --- Rate limiter ligero (in-memory, sliding window) ---
# Zero dependencias extra. Para multi-worker (gunicorn), cada worker tiene su propio contador,
# lo cual es aceptable (el rate real es N * max_calls). Para rate-limiting distribuido,
# usar Redis + slowapi.

class RateLimiter:
    """Sliding-window rate limiter como dependencia FastAPI reutilizable.
    Uso: Depends(RateLimiter(max_calls=10, period_seconds=60))"""
    def __init__(self, max_calls: int = 10, period_seconds: int = 60):
        self.max_calls = max_calls
        self.period = period_seconds
        self._hits: dict = _defaultdict(list)  # user_id → [timestamps]

    def __call__(self, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
        uid = verified_user_id or "anon"
        
        # Opcional: Soporte Redis para Rate Limiting Distribuido (#Mejora 3)
        if redis_client:
            now = _time.time()
            key = f"rl:{self.max_calls}:{self.period}:{uid}"
            window_start = now - self.period
            try:
                pipe = redis_client.pipeline()
                pipe.zremrangebyscore(key, 0, window_start) # Eliminar timestamps viejos
                pipe.zcard(key)                             # Contar peticiones en la ventana
                pipe.zadd(key, {str(now): now})             # Añadir petición actual
                pipe.expire(key, self.period)               # Expirar key para no consumir RAM eterna
                results = pipe.execute()
                
                count = results[1]
                if count >= self.max_calls:
                    raise HTTPException(
                        status_code=429,
                        detail=f"Demasiadas solicitudes. Máximo {self.max_calls} por {self.period}s. Intenta de nuevo en unos segundos."
                    )
                return verified_user_id
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"⚠️ [RATE LIMIT] Error en Redis, cambiando a memoria local transparente: {e}")
                # Hacemos fallback transparente a memoria local

        # --- Fallback Memoria Local ---
        now_mono = _time.monotonic()
        
        # Ocasionalmente (1% de tolerancia) limpiar llaves inactivas globales para evitar fugas de RAM
        import random
        if random.random() < 0.01:
            expired_keys = [
                k for k, timestamps in self._hits.items() 
                if not timestamps or (now_mono - timestamps[-1]) > self.period
            ]
            for k in expired_keys:
                del self._hits[k]

        # Purga timestamps viejos (fuera de la ventana)
        self._hits[uid] = [t for t in self._hits[uid] if now_mono - t < self.period]
        if len(self._hits[uid]) >= self.max_calls:
            raise HTTPException(
                status_code=429,
                detail=f"Demasiadas solicitudes. Máximo {self.max_calls} por {self.period}s. Intenta de nuevo en unos segundos."
            )
        self._hits[uid].append(now_mono)
        return verified_user_id


