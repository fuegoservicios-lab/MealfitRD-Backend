# backend/llm_telemetry.py
"""[P1-PLAN-LOTE-64 · 2026-09-15] (E7 / ARQ30-P2-01, 2.ª extracción) La telemetría de la llamada LLM y la caché de
prompts: los dos contextvars de atribución (`_current_node_var`, `user_id_var`), la métrica de timeout
(`_emit_llm_timeout_metric`), el emit idempotente del usage-event (`_USAGE_EMIT_SEEN`, `_usage_was_emitted`,
`_mark_usage_emitted`, `_emit_llm_usage_event_best_effort`) y la caché persistente (`PersistentLLMCache`, `_LLM_CACHE`,
knob `MEALFIT_LLM_CACHE_TTL_S`).

Vivían en `graph_orchestrator.py` y se movieron TAL CUAL (mismo texto, mismos comentarios; sólo cambian los imports); el
grafo los re-exporta, así que quien los importaba de allí sigue igual. Se queda en el grafo el despachador
`_submit_best_effort_metric` con su `_METRICS_EXECUTOR`: es la política de DÓNDE corre el emit y la comparten otros ocho
sitios del pipeline.

El logger conserva el nombre `graph_orchestrator` (misma razón que en `llm_concurrency.py` y `llm_circuit_breaker.py`:
el formato de producción imprime `%(name)s` y los `caplog` de la suite filtran por él). Para cambiar lo que ESTE código
ve en un test (`execute_sql_query`, `redis_client`, `get_redis_async`, `_get_be_db_cb`…), parchea `llm_telemetry.<nombre>`:
parchear `graph_orchestrator.<nombre>` ya no lo alcanza.
"""
import contextvars
import json
import logging
import weakref
from typing import Optional

from cache_manager import get_redis_async, redis_client
from db_core import aexecute_sql_query, aexecute_sql_write, execute_sql_query, execute_sql_write
from knobs import _env_int
from llm_attribution import plan_id_var
from llm_circuit_breaker import _get_be_db_cb, _is_pool_timeout_error

logger = logging.getLogger("graph_orchestrator")


# ============================================================
# [P1-COST-INSTRUMENTATION-PHASE2 · 2026-05-16] ContextVar para etiquetar
# llamadas LLM con el NODO del pipeline donde se originan. Phase 1 dejó la
# columna `llm_usage_events.node` 100% NULL ("unknown") porque la signature
# de `_safe_ainvoke` no incluía el caller — modificar 30+ callsites era
# invasivo. Phase 2 usa contextvar: cada nodo de LangGraph setea su nombre
# al entrar (via decorator `@_node_label("nombre")`), y propaga
# automáticamente a CUALQUIER `_safe_ainvoke` invocada en su scope
# (incluso a través de `asyncio.create_task` por la semántica de
# Python 3.7+ que copia el contexto a tasks hijas).
#
# El `_emit_llm_usage_event_best_effort` lee el var y lo pasa a
# `db_profiles.log_llm_usage_event(node=...)`. Si una llamada LLM ocurre
# FUERA del pipeline (e.g. chat agent tools), el var queda en None → la
# fila persiste con `node=NULL` (mismo comportamiento que phase 1 — no
# regresión).
#
# Tooltip-anchor: P1-COST-INSTRUMENTATION-PHASE2-CONTEXTVAR
_current_node_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "mealfit_current_node", default=None
)

# P1-NEW-1: ContextVar para rate-limit per-user. Lo setea `arun_plan_pipeline`
# al entrar y se propaga automáticamente a tasks hijas (asyncio.Task hereda
# el contexto al crearse) y a callbacks despachados vía `asyncio.to_thread`
# (`run_in_executor` con default executor preserva contextvars desde 3.7+).
# Default `None` → bypass del rate limit (preservar comportamiento de cron
# jobs, batch y otros callers que no setean el var).
user_id_var = contextvars.ContextVar("user_id", default=None)

# --- LLM cache TTL ---
LLM_CACHE_TTL_S             = _env_int  ("MEALFIT_LLM_CACHE_TTL_S",             300)

def _emit_llm_timeout_metric(
    *,
    node: str,
    timeout_threshold_s: float,
    actual_wait_s: float,
    llm=None,
    extra_metadata: dict | None = None,
) -> None:
    """[P1-LLM-TIMEOUT-METRICS · 2026-05-15] Helper SSOT para emitir tick a
    `pipeline_metrics` en TimeoutError catches del LLM.

    Best-effort: cualquier fallo de DB se silencia para no enmascarar el
    TimeoutError original al caller. Extrae model name del llm (si tiene
    `.model` attr — ChatGoogleGenerativeAI lo expone) sin lanzar.
    """
    try:
        model_name = None
        if llm is not None:
            for attr in ("model", "model_name", "_model"):
                try:
                    cand = getattr(llm, attr, None)
                    if isinstance(cand, str) and cand:
                        model_name = cand
                        break
                except Exception:
                    continue
        meta = {
            "timeout_threshold_s": float(timeout_threshold_s),
            "actual_wait_s": float(actual_wait_s),
            "model": model_name,
        }
        if extra_metadata:
            meta.update(extra_metadata)
        execute_sql_write(
            """
            INSERT INTO pipeline_metrics
                (user_id, session_id, node, duration_ms, retries,
                 tokens_estimated, confidence, metadata)
            VALUES (NULL, NULL, %s, %s, 0, 0, 0, %s::jsonb)
            """,
            (node, int(actual_wait_s * 1000), json.dumps(meta, ensure_ascii=False)),
        )
    except Exception as _e_metric:
        try:
            logger.debug(
                f"[P1-LLM-TIMEOUT-METRICS] emit falló (best-effort): {_e_metric!r}"
            )
        except Exception:
            pass

# [P2-ORCH-14 · 2026-05-28] Idempotencia del emit de usage-events. La misma
# AIMessage puede pasar por DOS capas de emit: el override del LLM raw
# (`ChatGoogleGenerativeAI.ainvoke/astream`, ~1264) Y `_safe_ainvoke` (~2737).
# Sin guard, los callers RAW (compressor/fact_checker que pasan el subclass crudo
# a `_safe_ainvoke`) doble-contarían el costo. La única razón de que hoy no
# doble-cuenten es incidental (los structured runnables no exponen `.model`).
# Stamp per-result (atributo o WeakSet de fallback) → emit exactamente una vez por
# objeto-resultado real. Tooltip-anchor: P2-ORCH-14.
_USAGE_EMIT_SEEN: "weakref.WeakSet" = weakref.WeakSet()


def _usage_was_emitted(result) -> bool:
    """[P2-ORCH-14] True si `result` ya fue contabilizado (no re-emitir)."""
    try:
        if getattr(result, "_mealfit_usage_emitted", False):
            return True
    except Exception:
        pass
    try:
        return result in _USAGE_EMIT_SEEN
    except Exception:
        return False


def _mark_usage_emitted(result) -> None:
    """[P2-ORCH-14] Marca `result` como ya-contabilizado. Intenta atributo
    (bypass de frozen vía object.__setattr__); si falla, WeakSet de identidad."""
    try:
        object.__setattr__(result, "_mealfit_usage_emitted", True)
        return
    except Exception:
        pass
    try:
        _USAGE_EMIT_SEEN.add(result)
    except Exception:
        # Objeto no weakref-able/hashable → no se puede trackear; mejor permitir
        # un emit (best-effort) que perder la fila por completo.
        pass


def _emit_llm_usage_event_best_effort(*, llm, result, duration_s: float, node: str = None) -> None:
    """[P1-COST-INSTRUMENTATION · 2026-05-15] Extrae model + usage_metadata
    de un response exitoso de LangChain ChatGoogleGenerativeAI y persiste
    a `llm_usage_events` via `db_profiles.log_llm_usage_event`.

    Best-effort wrapper: cualquier fallo (parse, DB) se silencia. No
    enmascara el response exitoso al caller.

    Acceso a usage_metadata es defensivo — LangChain expone:
      - `result.usage_metadata` (dict con input_tokens/output_tokens/total_tokens)
      - `result.response_metadata.usage_metadata` (fallback path en algunas
        versiones del SDK).
    Cached tokens vienen como `cached_content_token_count` (Gemini) o
    `input_token_details.cache_read` (LangChain canonical).

    [P3-CHAT-NODE-EXPLICIT · 2026-05-20] `node` ahora aceptable como kwarg
    explícito. Pre-fix: el helper solo resolvía desde `_current_node_var`
    (ContextVar). El chat-flow NO setea ese var → todas sus filas iban
    con `node=NULL` → SRE no podía filtrar costos chat vs plan-gen.
    Caller del chat ahora pasa `node='chat_call_model'` explícito; el
    ContextVar sigue siendo fallback para callsites del pipeline plan-gen
    que ya lo gestionan.
    """
    try:
        # [P2-ORCH-14] Idempotencia: si esta result ya fue contabilizada por otra
        # capa de emit (override raw vs _safe_ainvoke), abortar para no doble-contar.
        if result is not None and _usage_was_emitted(result):
            return
        model_name = None
        for attr in ("model", "model_name", "_model"):
            try:
                cand = getattr(llm, attr, None)
                if isinstance(cand, str) and cand:
                    model_name = cand
                    break
            except Exception:
                continue
        if not model_name:
            return

        usage = None
        try:
            usage = getattr(result, "usage_metadata", None)
        except Exception:
            usage = None
        if not usage:
            try:
                resp_meta = getattr(result, "response_metadata", None) or {}
                usage = resp_meta.get("usage_metadata") if isinstance(resp_meta, dict) else None
            except Exception:
                usage = None
        if not usage or not isinstance(usage, dict):
            return

        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        cached_tokens = 0
        details = usage.get("input_token_details")
        if isinstance(details, dict):
            cached_tokens = details.get("cache_read") or details.get("cached") or 0
        if not cached_tokens:
            cached_tokens = usage.get("cached_content_token_count", 0) or 0

        # [P1-COST-INSTRUMENTATION-PHASE2 · 2026-05-16] Inyecta el nombre
        # del nodo desde el ContextVar `_current_node_var`. Si la llamada
        # ocurrió fuera de un nodo etiquetado (e.g. agent tools, scripts
        # admin), `node` queda None → fila DB con `node=NULL` (legacy phase 1).
        # [P3-CHAT-NODE-EXPLICIT · 2026-05-20] Si el caller pasó `node`
        # explícito (kwarg), úsalo — tiene prioridad sobre el ContextVar.
        # El chat-flow (agent.py:call_model) lo pasa como 'chat_call_model'
        # porque ese flow NO setea el ContextVar.
        if node:
            current_node = node
        else:
            try:
                current_node = _current_node_var.get()
            except Exception:
                current_node = None

        from db_profiles import log_llm_usage_event
        # [P2-ORCH-14] Marcar ANTES del log (una sola fila por objeto-resultado).
        if result is not None:
            _mark_usage_emitted(result)
        # [P1-PLAN-LOTE-15 · 2026-09-12] plan_id directo desde el contexto (worker de chunks, /swap-meal, /regenerate-day):
        # en la cola el placeholder ya tiene id ANTES de generar, así que el canje por `corr` de abajo no dispara.
        try:
            _attr_pid = plan_id_var.get()
        except Exception:
            _attr_pid = None
        # [P1-COST-ATTRIBUTION · 2026-07-31] La "phase 2" que la docstring de
        # `log_llm_usage_event` prometía desde 2026-05-15 y nunca se hizo: sin
        # esto el libro de COSTO no se puede cruzar con nada. Medido el
        # 2026-07-31: user_id NULL en 6.061 de 6.063 filas y plan_id NULL en
        # las 6.063 → imposible responder "¿cuánto cuesta un plan?" o comparar
        # dos modelos por plan, que es justo lo que el índice de calidad
        # necesita para servir de guía.
        #   · user_id: del ContextVar que ya usa el router de modelos.
        #   · corr: el id de correlación del request/pipeline. Va en metadata
        #     (jsonb, sin migración) porque durante la GENERACIÓN el plan
        #     todavía no tiene id — nace del INSERT posterior (invariante I1).
        #     `attach_plan_id_to_usage_events` lo canjea por el plan_id real
        #     en cuanto ese INSERT ocurre.
        try:
            _attr_uid = user_id_var.get()
        except Exception:
            _attr_uid = None
        try:
            from correlation import get_correlation_id
            _attr_corr = get_correlation_id()
        except Exception:
            _attr_corr = None
        log_llm_usage_event(
            user_id=_attr_uid,
            plan_id=_attr_pid,
            model=model_name,
            node=current_node,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=int(cached_tokens) if cached_tokens else 0,
            metadata={
                "duration_s": round(float(duration_s), 3),
                **({"corr": _attr_corr} if _attr_corr and _attr_corr != "-" else {}),
            },
        )
    except Exception as _e_emit:
        try:
            logger.debug(
                f"[P1-COST-INSTRUMENTATION] emit usage event falló "
                f"(best-effort): {_e_emit!r}"
            )
        except Exception:
            pass


class PersistentLLMCache:
    """Implementa un diccionario persistente para reemplazar la caché en memoria"""
    def __init__(self, ttl_seconds=300):
        self.ttl = ttl_seconds

    def __contains__(self, key):
        return self.get(key) is not None

    def get(self, key, default=None):
        if redis_client:
            try:
                val = redis_client.get(key)
                if val:
                    return json.loads(val)
            except Exception as e:
                logger.warning(f"Redis cache read error: {e}")
        try:
            # [P0-4] Antes: `interval '%s seconds'` con `params=(key, self.ttl)`.
            # Psycopg NO sustituye `%s` dentro de literales SQL — el parser solo
            # cuenta UN placeholder real (`key = %s`) pero recibimos DOS params,
            # disparando `ProgrammingError` SIEMPRE. Resultado: cuando Redis
            # cae, el fallback DB del cache LLM tiene 100% miss garantizado y
            # cada lectura paga el costo del query + parse error silenciado.
            # Fix: `make_interval(secs => %s)` parametriza correctamente.
            res = execute_sql_query(
                "SELECT value FROM app_kv_store WHERE key = %s AND updated_at > now() - make_interval(secs => %s)",
                (key, self.ttl), fetch_one=True
            )
            if res:
                return res["value"] if isinstance(res["value"], list) or isinstance(res["value"], dict) else json.loads(res["value"])
        except Exception as e:
            logger.warning(f"DB cache read error: {e}")
        return default

    async def aget(self, key, default=None):
        # [P1-REDIS-ASYNC-PERLOOP-CB · 2026-07-07] cliente per-loop (ver LLMCircuitBreaker).
        _rc = get_redis_async()
        if _rc:
            try:
                val = await _rc.get(key)
                if val:
                    return json.loads(val)
            except Exception as e:
                logger.warning(f"Redis async cache read error: {e}")
        # [P1-BESTEFFORT-DB-CB · 2026-05-21] CB local: si el pool tuvo
        # ≥3 timeouts seguidos, skipea esta call los próximos 60s en lugar
        # de gastar 8-12s del pool timeout en una operación cosmética.
        _be_cb = _get_be_db_cb("llm_cache_aget")
        if _be_cb.is_open():
            return default
        try:
            # [P0-4] Ver comentario equivalente en `get` arriba — mismo bug,
            # mismo fix vía `make_interval(secs => %s)`.
            res = await aexecute_sql_query(
                "SELECT value FROM app_kv_store WHERE key = %s AND updated_at > now() - make_interval(secs => %s)",
                (key, self.ttl), fetch_one=True
            )
            _be_cb.record_success()
            if res:
                return res["value"] if isinstance(res["value"], list) or isinstance(res["value"], dict) else json.loads(res["value"])
        except Exception as e:
            if _is_pool_timeout_error(e):
                _be_cb.record_pool_timeout()
            # [P3-LOG-CLARITY · 2026-05-16] Bajado de warning→info: cache MISS
            # por pool saturado es BEST-EFFORT (el LLM call procede normal sin
            # cache hit; solo perdemos la optimización de evitar 1 prompt
            # duplicado). El plan NO falla por esto.
            logger.info(f"[LLM-CACHE] DB async miss (best-effort, LLM call procede): {e}")
        return default

    def __getitem__(self, key):
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def __setitem__(self, key, value):
        if redis_client:
            try:
                redis_client.setex(key, self.ttl, json.dumps(value))
            except Exception as e:
                logger.warning(f"Redis cache write error: {e}")
        try:
            execute_sql_write(
                "INSERT INTO app_kv_store (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                (key, json.dumps(value))
            )
        except Exception as e:
            logger.warning(f"DB cache write error: {e}")

    async def aset(self, key, value):
        # [P1-REDIS-ASYNC-PERLOOP-CB · 2026-07-07] cliente per-loop (ver LLMCircuitBreaker).
        _rc = get_redis_async()
        if _rc:
            try:
                await _rc.setex(key, self.ttl, json.dumps(value))
            except Exception as e:
                logger.warning(f"Redis async cache write error: {e}")
        # [P1-BESTEFFORT-DB-CB · 2026-05-21] Mismo gate que aget: fail-fast
        # cuando pool está saturado.
        _be_cb = _get_be_db_cb("llm_cache_aset")
        if _be_cb.is_open():
            return
        try:
            await aexecute_sql_write(
                "INSERT INTO app_kv_store (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                (key, json.dumps(value))
            )
            _be_cb.record_success()
        except Exception as e:
            if _is_pool_timeout_error(e):
                _be_cb.record_pool_timeout()
            # [P3-LOG-CLARITY · 2026-05-16] Bajado de warning→info: cache SET
            # fallido es BEST-EFFORT (perdemos opt de cachear este prompt
            # para próximas calls; no afecta el plan actual).
            logger.info(f"[LLM-CACHE] DB async set skipped (best-effort, no afecta plan): {e}")

# Brecha 1 Fix: Estado Persistente (Redis / DB)
# P1-NEW-2: TTL configurable vía `MEALFIT_LLM_CACHE_TTL_S`. Default 300s.
_LLM_CACHE = PersistentLLMCache(ttl_seconds=LLM_CACHE_TTL_S)
CACHE_TTL_SECONDS = LLM_CACHE_TTL_S
