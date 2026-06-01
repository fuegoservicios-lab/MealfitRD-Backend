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

            # [P2-RATELIMITER-BUCKET-METRICS · 2026-05-15] Emit cardinalidad
            # + alert sobre saturación de buckets. ANTES, el cleanup no
            # emitía métricas — si un bug (hash collision en user_id,
            # botnet generando IPs distintas) inflaba `self._hits` sin
            # límite, memoria leak silente hasta OOM. Best-effort:
            # cualquier fallo de DB se silencia para no afectar el
            # request en curso.
            try:
                _bucket_count = len(self._hits)
                _expired_count = len(expired_keys)
                _bucket_limit_warn = _env_int_safe(
                    "MEALFIT_RATE_LIMITER_BUCKET_LIMIT_WARN", 100000
                )
                _emit_rl_cleanup_metric(_bucket_count, _expired_count)
                if _bucket_limit_warn > 0 and _bucket_count > _bucket_limit_warn:
                    _emit_rl_saturation_alert(_bucket_count, _bucket_limit_warn)
                else:
                    # [P1-PROD-AUDIT-2 · 2026-05-30] La cardinalidad volvió bajo el
                    # umbral → auto-resolver el alert si estaba abierto. Pre-fix
                    # `rate_limiter_bucket_saturation` no tenía NINGÚN resolver →
                    # rojo-permanente tras un único spike transitorio.
                    _resolve_rl_saturation_alert()
            except Exception as _metric_err:
                try:
                    logger.debug(
                        f"[P2-RATELIMITER-BUCKET-METRICS] tick falló (best-effort): {_metric_err}"
                    )
                except Exception:
                    pass

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


# ---------------------------------------------------------------------------
# [P2-RATELIMITER-BUCKET-METRICS · 2026-05-15] Helpers para observabilidad
# del cleanup del rate limiter local (in-memory).
#
# Por qué módulo-level (no método de la clase):
#   El cleanup corre raramente (1% de requests). Mantener los helpers
#   fuera de la hot path del __call__ evita imports/binding overhead. El
#   tick es best-effort — un fallo de DB NO debe afectar el rate-limit
#   decision (que YA se tomó antes del bloque cleanup).
# ---------------------------------------------------------------------------
def _env_int_safe(name: str, default: int) -> int:
    """Lectura defensiva de env var entero. No registra en
    `_KNOBS_REGISTRY` para evitar circular import con graph_orchestrator
    (rate_limiter es upstream); el knob se documenta en CLAUDE.md /
    memoria del bundle."""
    import os
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _emit_rl_cleanup_metric(bucket_count: int, expired_count: int) -> None:
    """Best-effort INSERT a `pipeline_metrics` con cardinalidad del
    rate limiter. Lazy import de `db_core.execute_sql_write` para evitar
    circular import en module load (db_core importa rate_limiter? no, pero
    defensivo)."""
    try:
        from db_core import execute_sql_write
        import json as _json
        execute_sql_write(
            """
            INSERT INTO pipeline_metrics
                (user_id, session_id, node, duration_ms, retries,
                 tokens_estimated, confidence, metadata)
            VALUES (NULL, NULL, %s, 0, 0, %s, 0, %s::jsonb)
            """,
            (
                "rate_limiter_cleanup",
                int(bucket_count),
                _json.dumps({
                    "expired_buckets_removed": int(expired_count),
                    "remaining_bucket_count": int(bucket_count),
                }, ensure_ascii=False),
            ),
        )
    except Exception:
        # Silent — el cleanup no debe afectar requests legítimos.
        pass


def _emit_rl_saturation_alert(bucket_count: int, threshold: int) -> None:
    """Best-effort UPSERT a `system_alerts` cuando bucket_count > threshold."""
    try:
        from db_core import execute_sql_write
        import json as _json
        execute_sql_write(
            """
            INSERT INTO system_alerts
                (alert_key, alert_type, severity, title, message, metadata, affected_user_ids)
            VALUES (%s, 'rate_limiter_bucket_saturation', 'warning', %s, %s, %s::jsonb, %s::jsonb)
            ON CONFLICT (alert_key) DO UPDATE
            SET triggered_at = NOW(),
                metadata = EXCLUDED.metadata,
                resolved_at = NULL
            """,
            (
                "rate_limiter_bucket_saturation",
                "RateLimiter local: cardinalidad de buckets sobre umbral",
                (
                    f"In-memory `_hits` dict acumuló {bucket_count} buckets "
                    f"(umbral knob `MEALFIT_RATE_LIMITER_BUCKET_LIMIT_WARN`="
                    f"{threshold}). Posible memory leak (hash collision en "
                    f"user_id, botnet con muchas IPs distintas). "
                    f"Investigar antes de OOM. Bumpear knob solo si la "
                    f"causa raíz es legítima volumen orgánico."
                ),
                _json.dumps({
                    "bucket_count": int(bucket_count),
                    "threshold": int(threshold),
                }, ensure_ascii=False),
                _json.dumps([]),
            ),
        )
    except Exception:
        pass


def _resolve_rl_saturation_alert() -> None:
    """[P1-PROD-AUDIT-2 · 2026-05-30] Auto-resolve de `rate_limiter_bucket_saturation`.

    Pre-fix esta alert se emitía con `resolved_at = NULL` y NO tenía NINGÚN
    resolver (Auto/Handler/sweep) → quedaba rojo-permanente en el panel tras un
    único spike transitorio de cardinalidad. La llamamos en el cleanup tick cuando
    `_bucket_count <= threshold`. Best-effort + idempotente; `AND resolved_at IS NULL`
    para no re-tocar filas ya cerradas (clase NG-2/S12-2)."""
    try:
        from db_core import execute_sql_write
        execute_sql_write(
            "UPDATE system_alerts SET resolved_at = NOW() "
            "WHERE alert_key = 'rate_limiter_bucket_saturation' AND resolved_at IS NULL",
        )
    except Exception:
        pass
