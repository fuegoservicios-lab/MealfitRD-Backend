"""[H2 / P3-CORRELATION-ID · 2026-05-20] Correlation ID per-request +
propagación context-aware a través de async tasks y thread pool.

Por qué existe (audit `docs/gaps-audit-2026-05.md` H2):
    Pre-fix, debuggear un fallo del chat agent o del plan pipeline requería
    grep manual del `user_id` (cuando estaba) a través de logs de
    `agent.py` → `tools.py` → `db_plans.py` → `cron_tasks.py`. Cada uno
    loguea con su propio formato y NO hay un identificador único que
    correlacione un request HTTP con todos sus efectos cascada (background
    tasks, llamadas LLM, mutaciones DB).

    Síntoma típico: usuario reporta "el plan generó mal hoy a las 3pm".
    Sin correlation ID:
      - Filtrar logs por user_id (si llegó a loguearse) → 47 líneas mezcladas
        con otros workers procesando crons.
      - Ordenar por timestamp → orden parcial, contaminado por las 14k
        líneas de graph_orchestrator emitiendo en paralelo.
      - Reconstruir el flow manualmente cruzando timestamps + node names.
      - Tiempo medio: 30-60min por incident.

    Con correlation ID:
      - `grep corr=a7b3 logs.txt` → línea por línea del request entero
        en orden, sin contaminación cruzada.
      - Tiempo medio: 5-10min.

Diseño:
    1. `_correlation_id: ContextVar[str]` — Python contextvars (PEP 567)
       provides automatic propagation through `asyncio.create_task`,
       `asyncio.to_thread`, y cualquier coroutine. NO se propaga
       automáticamente a `concurrent.futures.ThreadPoolExecutor`
       (workers tienen sus propias threads sin el contextvar parent).
       Solución: `bg_executor.submit_bg_task` captura
       `contextvars.copy_context()` en el caller y ejecuta el callable
       via `ctx.run(...)` en el worker.

    2. `CorrelationIdFilter(logging.Filter)` — inyecta `record.correlation_id`
       a CADA log record desde el contextvar actual. Se instala sobre el
       root logger en `app.py` post-basicConfig — afecta todos los
       handlers/loggers/módulos sin tocar callsites.

    3. FastAPI middleware en `app.py`:
         - Lee `X-Correlation-ID` del request header (si el cliente lo
           propaga — útil para SPA tracing) o genera uno nuevo.
         - Set context var con token; reset token al fin del request
           (defensa contra leak entre requests del mismo worker thread).
         - Echo `X-Correlation-ID` en response header — cliente puede
           citarlo al reportar un bug.

    4. Formato: `[corr=<8chars>]` en cada log line.
       - 8 chars suficiente para distinguir requests en un día (~256M
         combinaciones).
       - Default `-` cuando no hay request activo (cron, init, shutdown
         — para distinguirlos visualmente de los logs request-scoped).

Limitaciones honestas:
    - `bg_executor.submit_bg_task` propaga el ID PERO el watcher daemon
      thread NO — corre en su propio thread sin contextvar. Eso es
      intencional: el watcher es operacional (timeout/alert), no parte
      del request lógico. Sus logs llevan `corr=-` lo cual es correcto.
    - APScheduler crons corren en threads pre-existentes (background
      scheduler) sin request scope → todos llevan `corr=-` también.
      Para correlation cron-internal, futuro P-fix podría asignar
      `corr=cron:<job_name>:<run_id>` al entry-point del cron.
    - Si un caller NO usa `submit_bg_task` y hace `threading.Thread`
      directo, el ID no se propaga (no hay forma de interceptar). Audit
      cross-codebase 2026-05-20: 0 uses de `threading.Thread` directos en
      paths productivos post-bg_executor (cron tasks pre-existentes
      siempre crean el thread via APScheduler).

Tooltip-anchor: P3-CORRELATION-ID.
Test: `tests/test_p3_correlation_id.py`.
"""
from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Iterator, Optional


# ---------------------------------------------------------------------------
# ContextVar — la única fuente de verdad del ID activo en el async/sync flow.
# ---------------------------------------------------------------------------
#
# Default `"-"`:
#   Distintivo visual cuando un log ocurre fuera de scope (init module,
#   cron, shutdown). Si el default fuera vacío "", el formato quedaría
#   `[corr=]` que es confuso. Si fuera "system", se confundiría con
#   un correlation ID legítimo de un request del usuario "system".
#
# Tipo `str` (no `Optional`):
#   Simplifica el filtro logging — siempre puede hacer `record.correlation_id = get()`
#   sin checks de None.
_correlation_id: ContextVar[str] = ContextVar("mealfit_correlation_id", default="-")


def get_correlation_id() -> str:
    """Retorna el correlation_id activo en el ContextVar actual.

    Safe en cualquier contexto (request, cron, init, thread pool worker
    siempre que el caller haya usado `ctx.run()`). Si no hay scope
    activo, retorna `"-"`.
    """
    return _correlation_id.get()


def set_correlation_id(value: str) -> Token:
    """Setea el correlation_id en el ContextVar actual y retorna el Token
    para reset posterior. NO usar directamente — preferir `with_correlation_id`
    o el middleware FastAPI que gestiona el lifecycle.

    Retorna:
        `Token` que el caller DEBE pasar a `_correlation_id.reset(token)`
        cuando termine el scope. Sin reset, el valor persiste en el
        worker thread y se puede leak a un request siguiente del MISMO
        worker (modo de fallo silencioso bajo carga).
    """
    return _correlation_id.set(value or "-")


def reset_correlation_id(token: Token) -> None:
    """Reset del ContextVar. Llamar en finally tras `set_correlation_id`."""
    try:
        _correlation_id.reset(token)
    except (LookupError, ValueError):
        # Token de otro contexto — sucede si caller llama reset fuera de
        # orden o tras un await que cruza tasks. Ignorar: peor caso, el
        # contextvar queda con el value pre-existente (que pronto será
        # overwriteado por el próximo set_correlation_id del próximo request).
        pass


def new_correlation_id() -> str:
    """Genera un nuevo correlation_id corto (8 chars hex, ~256M combinaciones).

    Diseño:
      - 8 chars = uuid4 truncado al primer block. Suficiente para
        unicidad en un día (>>1M requests sin colisión esperada).
      - Hex lowercase: fácil de grep, no se confunde con prefijos.
      - NO usar timestamp — un correlation ID debe ser opaco para que
        no haya tentación de derivar info del valor (e.g. ordering
        global). Para ordering usar el `asctime` del log.
    """
    return uuid.uuid4().hex[:8]


@contextmanager
def with_correlation_id(value: Optional[str] = None) -> Iterator[str]:
    """Context manager para scope manual del correlation ID.

    Uso típico (worker thread o cron que quiere su propio scope):

        from correlation import with_correlation_id

        with with_correlation_id() as cid:
            logger.info("starting batch process")
            ...

    Si `value=None`, genera uno nuevo. Si `value=""`, también genera uno
    nuevo (vacío sería visualmente confuso en logs).
    """
    cid = value if value else new_correlation_id()
    token = set_correlation_id(cid)
    try:
        yield cid
    finally:
        reset_correlation_id(token)


# ---------------------------------------------------------------------------
# Logging filter — inyecta `record.correlation_id` a cada log record.
# ---------------------------------------------------------------------------


class CorrelationIdFilter(logging.Filter):
    """Filtro que añade `correlation_id` a cada `LogRecord` desde el
    ContextVar activo. Se instala una sola vez en el root logger
    (post-`logging.basicConfig`) y propaga automáticamente a todos los
    loggers hijos (`logging.getLogger("foo.bar")` etc.) porque comparten
    el handler raíz.

    Filtros de logging son thread-safe per design — la única condición
    es que `filter()` NO lance excepciones. Si por alguna razón el
    contextvar lookup falla (no debería), defaultea a `"-"`.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.correlation_id = _correlation_id.get()
        except Exception:
            record.correlation_id = "-"
        return True  # Nunca dropear el record


def install_log_filter(root_logger: Optional[logging.Logger] = None) -> None:
    """Instala el `CorrelationIdFilter` en el logger dado (default: root).

    Idempotente: si ya hay un `CorrelationIdFilter` instalado, no añade
    otro. Útil para tests que importan app.py múltiples veces.

    El caller debe haber configurado el `format=` del basicConfig con
    `%(correlation_id)s` o el formato no expondrá el atributo.
    """
    target = root_logger if root_logger is not None else logging.getLogger()
    # Evitar duplicación si ya está instalado
    for f in target.filters:
        if isinstance(f, CorrelationIdFilter):
            return
    target.addFilter(CorrelationIdFilter())

    # CRÍTICO: los handlers HEREDAN filters del logger SOLO si están
    # attached al mismo logger. Como `basicConfig` añade el handler al
    # root, también necesitamos el filter en cada handler para que los
    # records que pasan por el handler tengan el atributo antes del
    # formato. Si el filter solo está en el logger, los records emitidos
    # por loggers HIJOS NO pasan por el filtro del root (propagan solo
    # los handlers, no los filters).
    for handler in target.handlers:
        # Misma protección de idempotencia
        already = any(isinstance(f, CorrelationIdFilter) for f in handler.filters)
        if not already:
            handler.addFilter(CorrelationIdFilter())
