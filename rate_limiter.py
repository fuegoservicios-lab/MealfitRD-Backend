import time as _time
import math as _math
from collections import defaultdict as _defaultdict
from typing import Optional
from fastapi import Depends, HTTPException, Request
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
    Uso: Depends(RateLimiter(max_calls=10, period_seconds=60))

    [P1-6] Bucket key: prefer `verified_user_id` cuando el usuario está
    autenticado; fall back a `ip:<client_ip>` para anónimos. Antes el código
    usaba un único bucket "anon" compartido entre TODOS los visitantes —
    bastaba un mal actor para agotar el cupo y bloquear a todos los demás.
    Con IP-fallback, cada IP anon tiene su propio cupo y los abusadores se
    aíslan. Para clientes detrás de proxy/CDN, idealmente el deploy expone
    `X-Forwarded-For` que FastAPI/Starlette respeta vía `request.client.host`
    cuando se configura `--proxy-headers`.
    """
    def __init__(self, max_calls: int = 10, period_seconds: int = 60):
        self.max_calls = max_calls
        self.period = period_seconds
        self._hits: dict = _defaultdict(list)  # user_id → [timestamps]

    def __call__(self, request: Request, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
        if verified_user_id:
            uid = verified_user_id
        else:
            # IP fallback. `request.client` puede ser `None` en clientes raros
            # (e.g., uvicorn con socket UNIX); en ese caso retornamos al
            # bucket compartido "anon" como último recurso.
            client_ip = request.client.host if request.client else None
            uid = f"ip:{client_ip}" if client_ip else "anon"
        
        # Opcional: Soporte Redis para Rate Limiting Distribuido (#Mejora 3)
        if redis_client:
            now = _time.time()
            key = f"rl:{self.max_calls}:{self.period}:{uid}"
            window_start = now - self.period
            try:
                # [P1-FORM-5] Añadido `zrange(0, 0, withscores=True)` para
                # obtener el hit más viejo (smallest score) dentro de la
                # ventana. Permite calcular un `Retry-After` exacto en lugar
                # de obligar al cliente a defaultear a 60s.
                pipe = redis_client.pipeline()
                pipe.zremrangebyscore(key, 0, window_start) # 0: Eliminar timestamps viejos
                pipe.zcard(key)                             # 1: Contar peticiones en la ventana
                pipe.zrange(key, 0, 0, withscores=True)     # 2: [P1-FORM-5] hit más viejo
                pipe.zadd(key, {str(now): now})             # 3: Añadir petición actual
                pipe.expire(key, self.period)               # 4: Expirar key para no consumir RAM eterna
                results = pipe.execute()

                count = results[1]
                oldest_list = results[2]
                if count >= self.max_calls:
                    # [P1-FORM-5] Retry-After exacto = tiempo hasta que el hit
                    # más viejo deje la ventana sliding. Antes el limiter solo
                    # devolvía 429 sin header → frontend defaulteaba a 60s
                    # incluso cuando la ventana se liberaría en 10s → UX
                    # confusa: usuario espera 60s, reintenta antes y vuelve a
                    # 429. Ahora el cliente ve el countdown real.
                    if oldest_list:
                        oldest_score = float(oldest_list[0][1])
                        retry_after = max(1, int(_math.ceil((oldest_score + self.period) - now)))
                    else:
                        # Fallback defensivo: si el set quedó vacío entre el
                        # zcard y el zrange (race muy improbable), usar el
                        # period completo. El cliente espera lo seguro.
                        retry_after = self.period
                    raise HTTPException(
                        status_code=429,
                        detail=f"Demasiadas solicitudes. Máximo {self.max_calls} por {self.period}s. Reintenta en {retry_after}s.",
                        headers={"Retry-After": str(retry_after)},
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
            # [P1-FORM-5] Retry-After exacto desde el hit más viejo en
            # monotonic time. Tras la purga arriba, `_hits[uid][0]` es el
            # hit válido más antiguo (lista preserva orden de inserción
            # cronológica). max(1, ...) garantiza al menos 1s para evitar
            # countdown=0 que el cliente interpretaría como "reintentar ya".
            oldest_mono = self._hits[uid][0]
            retry_after = max(1, int(_math.ceil((oldest_mono + self.period) - now_mono)))
            raise HTTPException(
                status_code=429,
                detail=f"Demasiadas solicitudes. Máximo {self.max_calls} por {self.period}s. Reintenta en {retry_after}s.",
                headers={"Retry-After": str(retry_after)},
            )
        self._hits[uid].append(now_mono)
        return verified_user_id


