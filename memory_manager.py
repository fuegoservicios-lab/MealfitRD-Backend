# backend/memory_manager.py
"""
Gestión de Memoria y Resúmenes Progresivos.
Proceso asíncrono que resume conversaciones antiguas para mantener
el contexto del usuario vivo sin saturar la ventana de tokens de Gemini.
"""

import os
import time
import json
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import RemoveMessage

logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

from db import (
    get_memory,
    get_summaries,
    save_summary,
    delete_old_messages,
    get_recent_messages,
    delete_summaries,
    archive_summaries,
    acquire_summarizing_lock,
    release_summarizing_lock,
    connection_pool
)

load_dotenv()

# ============================================================
# CONSTANTES
# ============================================================
MAX_CHAR_THRESHOLD = 4000   # A partir de cuántos caracteres en total se dispara el resumen
KEEP_RECENT = 10         # Cuántos mensajes recientes conservar sin resumir
MAX_SUMMARIES = 5        # Umbral para condensar resúmenes en un Master Summary

# [P1-18] Modelo de Gemini usado por `summarize_and_prune` para generar
# resúmenes y master summaries. ANTES estaba hardcodeado a
# `"gemini-3.1-flash-lite-preview"` — un ID que NO corresponde a ningún
# modelo público de Gemini conocido (la familia 3.x no existe; los
# modelos públicos son 1.5/2.0/2.5). Si el SDK lo rechazaba con 404, el
# resumen fallaba y el `except Exception` lo loggeaba como warning
# silencioso → el historial de la conversación seguía creciendo sin
# podarse, agotando la ventana de contexto del agente.
#
# Ahora:
#   - Configurable vía env var con default a `gemini-2.5-flash` (modelo
#     válido y económico para resúmenes).
#   - Cualquier mismatch con la nomenclatura oficial se detecta vía el
#     contador `_summarize_failures` y se promueve a `logger.error`
#     (en lugar de warning silencioso) para que SRE alertee.
MEMORY_SUMMARY_MODEL = os.environ.get(
    "MEMORY_SUMMARY_MODEL", "gemini-2.5-flash"
).strip() or "gemini-2.5-flash"

# [P1-18] Contador in-memory de fallos del resumen. Se incrementa en cada
# excepción de `summarize_and_prune` y se loggea a nivel `error` el primer
# fallo + cada N para detectar degradación sistémica sin spamear logs.
# Reset al primer éxito (auto-recovery sin restart del proceso).
#
# [P1-NEW-6 · 2026-05-11] Extendido con `last_attempt_at` para backoff
# exponencial: cuando hay racha de fallos, las próximas invocaciones se
# saltean hasta que pase la ventana de retry calculada. Sin esto, durante
# outage de Gemini (cuota / API key rotada / modelo eliminado) el cron
# invoca `summarize_and_prune` cada N min × M sesiones activas, quemando
# API calls (cost), saturando el pool de threads (otros crons MISSED), y
# llenando logs sin valor. El backoff cierra ese loop sin pausar
# permanentemente — al primer éxito se resetea el estado y todo vuelve a
# operación normal.
_summarize_failures: dict = {"count": 0, "last_error": None, "last_attempt_at": 0.0}

# [P1-19] Threshold de fallos consecutivos a partir del cual persistimos
# una row en `system_alerts` (además del log error). Esto permite que SRE
# alerte en dashboard sin tener que parsear logs. Dedupe por alert_key
# para que la primera y subsiguientes alertas reescriban el mismo row
# en lugar de crecer sin límite. Defensa contra una racha sostenida
# (modelo inválido, API key rotada, tabla missing, cuota agotada) que
# pasaría inadvertida si solo confiamos en logs durante outages.
_SUMMARY_FAILURE_ALERT_THRESHOLD = 5
_SUMMARY_FAILURE_ALERT_REPEAT_EVERY = 50  # re-persistir cada N adicionales


# [P1-NEW-6 · 2026-05-11] Tabla de backoff exponencial. Mapa `(min_count,
# max_count) → wait_seconds`. Aplica una vez que el contador
# `_summarize_failures["count"]` cruza el primer umbral.
#   - count 1-4:   sin skip (try harder, fallos esporádicos).
#   - count 5-9:   skip si < 10min desde last_attempt.
#   - count 10-19: skip si < 30min.
#   - count 20-39: skip si < 1h.
#   - count >=40:  skip si < 4h (degradación sistémica clara — Sentry/SRE).
# Cada entry es `(threshold_inclusive, wait_seconds)`.
_SUMMARY_BACKOFF_TABLE: list[tuple[int, int]] = [
    (5, 10 * 60),
    (10, 30 * 60),
    (20, 60 * 60),
    (40, 4 * 60 * 60),
]


def _summary_backoff_should_skip() -> bool:
    """[P1-NEW-6 · 2026-05-11] Decide si saltar la invocación actual de
    `summarize_and_prune` por backoff exponencial.

    Lee `_summarize_failures["count"]` y `_summarize_failures["last_attempt_at"]`.
    Si NO hay fallos recientes (count == 0) retorna False (proceed).
    Si los hay, busca la entry de _SUMMARY_BACKOFF_TABLE aplicable y
    verifica si ha pasado el wait_seconds desde el último intento.

    Knob kill-switch `MEALFIT_SUMMARY_BACKOFF_ENABLED` (default true).
    Set a `false` durante debugging para forzar todas las invocaciones."""
    try:
        from knobs import _env_bool
        if not _env_bool("MEALFIT_SUMMARY_BACKOFF_ENABLED", True):
            return False
    except Exception:
        # Si el knob loader falla, default seguro: backoff activo.
        pass

    count = _summarize_failures.get("count", 0)
    if count <= 0:
        return False  # path feliz; no hay backoff.

    last_at = _summarize_failures.get("last_attempt_at", 0.0)
    if not isinstance(last_at, (int, float)) or last_at <= 0:
        return False  # estado inconsistente; no podemos calcular wait.

    # Encontrar el wait_seconds más alto cuyo threshold sea <= count.
    # _SUMMARY_BACKOFF_TABLE está ordenada ascendente; iteramos en reverse
    # para encontrar el umbral mayor aplicable.
    wait_s: int = 0
    for threshold, w in reversed(_SUMMARY_BACKOFF_TABLE):
        if count >= threshold:
            wait_s = w
            break
    if wait_s <= 0:
        return False  # count < primer umbral; no skip todavía.

    elapsed = time.time() - last_at
    if elapsed < wait_s:
        return True
    return False


def _persist_summary_failure_alert(session_id: str, error_str: str, count: int) -> None:
    """[P1-19] Persiste / refresca alerta en `system_alerts` cuando el
    contador de fallos consecutivos cruza el threshold. Best-effort: si la
    DB también está caída, fallamos silenciosamente (el log error ya
    capturó la señal arriba). Defensa: no levantamos nunca para no
    duplicar el except del caller."""
    try:
        from db_core import execute_sql_write
        alert_key = "memory_summary_failures"
        execute_sql_write(
            """
            INSERT INTO system_alerts (alert_key, alert_type, severity, title, message, metadata)
            VALUES (%s, 'memory_summary_failure', 'warning', %s, %s, %s::jsonb)
            ON CONFLICT (alert_key) DO UPDATE
            SET triggered_at = NOW(),
                message = EXCLUDED.message,
                metadata = EXCLUDED.metadata,
                resolved_at = NULL
            """,
            (
                alert_key,
                f"summarize_and_prune fallando ({count} consecutivos)",
                (
                    f"`memory_manager.summarize_and_prune` lleva {count} fallos "
                    f"consecutivos. Posible modelo inválido "
                    f"(MEMORY_SUMMARY_MODEL={MEMORY_SUMMARY_MODEL!r}), API key "
                    f"rotada, cuota agotada o DB blip. La memoria del agente "
                    f"NO se está podando — riesgo de explosión de tokens y "
                    f"costos disparados."
                ),
                json.dumps({
                    "consecutive_failures": count,
                    "last_error": error_str[:500],
                    "last_session_id": session_id,
                    "model": MEMORY_SUMMARY_MODEL,
                }, ensure_ascii=False),
            ),
        )
    except Exception as alert_err:
        # No re-lanzar: el log error del caller ya capturó la señal.
        # Pero loggeamos a debug para que en dev podamos detectar si la
        # tabla está missing o el schema cambió.
        logger.debug(
            f"[MEMORY MANAGER/P1-19] No se pudo persistir alerta a system_alerts: {alert_err}"
        )


# ============================================================
# SINCRONIZACIÓN: Purgar Checkpoint de LangGraph
# ============================================================

# [P1-17] Cache lazy-singleton del grafo dummy usado para manipular el state
# del PostgresSaver sin importar agent.py. ANTES, cada llamada a
# `purge_langgraph_checkpoint` (disparada por `summarize_and_prune`)
# instanciaba un nuevo `StateGraph` + `PostgresSaver` + `compile`. En
# producción con N sesiones activas y resúmenes frecuentes, esto era
# CPU/memoria desperdiciados (objetos pesados de LangGraph) + potencial
# leak de los compiled graphs.
#
# AHORA cacheamos el grafo a nivel módulo. Invalidación por `id(pool)`
# para tests donde `connection_pool` se mockea/reemplaza — en producción
# el pool nunca cambia tras el startup, así que el cache se carga UNA
# vez por proceso.
import threading as _p117_threading

_DUMMY_PURGE_GRAPH_CACHE: dict = {"graph": None, "pool_id": None}
_DUMMY_PURGE_GRAPH_LOCK = _p117_threading.Lock()


def _get_dummy_purge_graph():
    """[P1-17] Devuelve el grafo dummy cacheado para `purge_langgraph_checkpoint`.

    Reusa la misma instancia entre llamadas si el `connection_pool` no
    cambió. Thread-safe vía double-checked locking. Retorna `None` si
    no hay pool disponible (caller hace early-return)."""
    if not connection_pool:
        return None
    pool_id = id(connection_pool)

    cached = _DUMMY_PURGE_GRAPH_CACHE
    if cached["graph"] is not None and cached["pool_id"] == pool_id:
        return cached["graph"]

    with _DUMMY_PURGE_GRAPH_LOCK:
        # Double-check: otro thread pudo haber inicializado mientras
        # esperábamos el lock.
        cached = _DUMMY_PURGE_GRAPH_CACHE
        if cached["graph"] is not None and cached["pool_id"] == pool_id:
            return cached["graph"]

        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.graph import StateGraph, START
        from langgraph.graph.message import MessagesState

        builder = StateGraph(MessagesState)
        builder.add_node("dummy", lambda state: state)
        builder.add_edge(START, "dummy")
        checkpointer = PostgresSaver(connection_pool)
        graph = builder.compile(checkpointer=checkpointer)

        _DUMMY_PURGE_GRAPH_CACHE["graph"] = graph
        _DUMMY_PURGE_GRAPH_CACHE["pool_id"] = pool_id
        return graph


def purge_langgraph_checkpoint(
    session_id: str,
    keep_recent: int = KEEP_RECENT,
    *,
    raise_on_failure: bool = False,
):
    """
    Purga los mensajes antiguos del checkpoint de LangGraph (PostgresSaver)
    para mantenerlo sincronizado con la tabla agent_messages de Supabase.

    Sin esta función, LangGraph acumula TODOS los mensajes históricos en su
    propio estado inmutable y los re-inyecta silenciosamente al modelo en cada
    turno, causando explosión de tokens y costos disparados.

    Usa la API RemoveMessage de LangChain para eliminar mensajes del state.

    [P1-17] El grafo dummy se cachea a nivel módulo (lazy singleton). Antes
    se instanciaba `StateGraph + PostgresSaver + compile` en cada llamada,
    desperdiciando CPU/memoria con potencial leak.

    [P1-20] `raise_on_failure` (kw-only, default False) controla si una
    excepción del SDK de LangGraph propaga al caller. Default False
    preserva el contrato histórico (best-effort, log warning silencioso).
    `summarize_and_prune` lo invoca con `True` para que un fallo del purge
    aborte la subsiguiente `delete_old_messages` y mantenga consistencia
    entre Supabase ↔ LangGraph (sin esto, A commiteaba el delete y B fallaba
    silenciosamente → mensajes borrados de Supabase pero LangGraph aún los
    retenía → re-inyección al LLM en el siguiente turno).
    """
    if not connection_pool:
        logger.warning("⚠️ [MEMORY MANAGER] No hay connection_pool, no se puede purgar checkpoint de LangGraph.")
        if raise_on_failure:
            raise RuntimeError(
                "[P1-20] purge_langgraph_checkpoint requirido pero connection_pool no disponible"
            )
        return

    try:
        from langchain_core.messages import RemoveMessage

        graph = _get_dummy_purge_graph()
        if graph is None:
            logger.warning("⚠️ [MEMORY MANAGER] _get_dummy_purge_graph retornó None. Skip.")
            return

        config = {"configurable": {"thread_id": session_id}}
        state = graph.get_state(config)
        
        if not state or not state.values:
            logger.info("➡️ [MEMORY MANAGER] No hay estado de LangGraph para esta sesión. Nada que purgar.")
            return
        
        messages = state.values.get("messages", [])
        
        if len(messages) <= keep_recent:
            logger.info(f"➡️ [MEMORY MANAGER] Solo {len(messages)} mensajes en checkpoint, no se necesita purga.")
            return
        
        # Calcular qué mensajes eliminar (todos excepto los keep_recent más recientes)
        messages_to_remove = messages[:-keep_recent]
        
        # Crear RemoveMessage para cada mensaje a eliminar
        remove_messages = [RemoveMessage(id=m.id) for m in messages_to_remove if hasattr(m, 'id') and m.id]
        
        if not remove_messages:
            logger.info("➡️ [MEMORY MANAGER] No se encontraron IDs de mensajes para purgar.")
            return
        
        logger.info(f"🔄 [MEMORY MANAGER] Purgando {len(remove_messages)} mensajes del checkpoint de LangGraph...")
        
        # Snapshot de seguridad antes de modificar
        pre_update_snapshot = graph.get_state(config)
        
        try:
            # Usar update_state para aplicar las eliminaciones
            graph.update_state(config, {"messages": remove_messages})
            logger.info(f"✅ [MEMORY MANAGER] {len(remove_messages)} mensajes eliminados del checkpoint de LangGraph. State sincronizado.")
        except Exception as update_err:
            logger.error(f"🚨 [MEMORY MANAGER] Fallo parcial en update_state: {update_err}. Restaurando snapshot...")
            if pre_update_snapshot and hasattr(pre_update_snapshot, 'values'):
                # Forzar la re-escritura del estado original
                graph.update_state(config, pre_update_snapshot.values)
                logger.info("♻️ [MEMORY MANAGER] Snapshot restaurado tras fallo.")
            # Propagar para que el except global lo capture
            raise update_err
            
    except Exception as e:
        # Nunca bloquear al usuario por un error de sincronización (default).
        logger.warning(f"⚠️ [MEMORY MANAGER] Error no-crítico purgando checkpoint de LangGraph: {e}")
        # [P1-20] Si el caller exigió fail-fast (e.g., `summarize_and_prune`
        # para preservar consistencia con `delete_old_messages`), propagar.
        if raise_on_failure:
            raise


# ============================================================
# PROMPTS DE RESUMEN (importados del paquete prompts/)
# ============================================================
from prompts.memory import (
    SUMMARY_PROMPT,
    MASTER_SUMMARY_PROMPT,
    PRIOR_STATE_INSTRUCTION_WITH_DATA,
    PRIOR_STATE_INSTRUCTION_EMPTY,
)

# ============================================================
# SCHEMA PYDANTIC: Estado Evolutivo del Paciente
# ============================================================
class EvolutionaryFeedback(BaseModel):
    le_encanta: List[str] = Field(description="Lista de cosas que al usuario le encantan (sabores, platos, texturas)")
    rechaza: List[str] = Field(description="Lista de cosas que el usuario rechaza o no le gustan")
    preferencias_horario: str = Field(description="Preferencias de horario de comidas")

class EvolutionaryState(BaseModel):
    tendencias_adherencia: str = Field(description="Cómo ha sido su adherencia al plan a lo largo del tiempo")
    nivel_estres: str = Field(description="Patrones de estrés observados")
    patrones_cocina: str = Field(description="Hábitos de cocina: tiempo, herramientas, estilo")
    feedback_recurrente: EvolutionaryFeedback = Field(description="Feedback recurrente sobre comidas")
    evolucion_dieta: str = Field(description="Cómo han cambiado sus objetivos o enfoque dietético")
    contexto_personal: str = Field(description="Contexto de vida relevante: trabajo, horarios, rutinas")
    notas_chef: str = Field(description="Insights culinarios específicos para este paciente")






# ============================================================
# FUNCIÓN PRINCIPAL: Resumir y Podar
# ============================================================
def summarize_and_prune(session_id: str):
    """
    Proceso asíncrono que:
    1. Verifica si la longitud total de caracteres del historial supera MAX_CHAR_THRESHOLD.
    2. Resume el bloque de mensajes más antiguo con Gemini.
    3. Guarda el resumen en conversation_summaries.
    4. Elimina los mensajes ya resumidos de agent_messages.

    [P1-NEW-6 · 2026-05-11] Guard de backoff exponencial: si hay racha
    consecutiva de fallos del resumen (Gemini down, cuota, modelo
    inválido), las próximas invocaciones se saltean hasta que pase la
    ventana de wait calculada (10min/30min/1h/4h según severity).
    """
    # [P1-NEW-6 · 2026-05-11] `global` declarado al inicio de la función
    # — el reset del dict en el path de éxito (más abajo) requiere rebind,
    # y Python 3.10+ exige que `global` preceda al primer uso de la
    # variable en el scope. Sin esta línea aquí, `py_compile --strict`
    # falla con `name '_summarize_failures' is used prior to global
    # declaration`.
    global _summarize_failures

    # [P1-NEW-6 · 2026-05-11] Backoff exponencial — skip si racha activa.
    if _summary_backoff_should_skip():
        count = _summarize_failures.get("count", 0)
        last_err = _summarize_failures.get("last_error")
        logger.info(
            f"⏰ [MEMORY MANAGER/P1-NEW-6] Skip resumen sesión "
            f"{str(session_id)[:8]}… — backoff activo ({count} fallos "
            f"consecutivos, último error: {last_err!r}). Próximo intento "
            f"tras ventana de espera. Set MEALFIT_SUMMARY_BACKOFF_ENABLED=false "
            f"para forzar."
        )
        return

    if not acquire_summarizing_lock(session_id):
        logger.warning(f"⚠️ [MEMORY MANAGER] Resumen ya en progreso para sesión {session_id}. Mitigando Condición de Carrera.")
        return

    try:
        all_messages = get_memory(session_id)
        
        if not all_messages:
            return
            
        total_chars = sum(len(msg.get("content", "")) for msg in all_messages)
        
        if total_chars <= MAX_CHAR_THRESHOLD:
            return  # No hay suficientes caracteres para detonar el resumen
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🧠 [MEMORY MANAGER] Sesión {str(session_id)[:8]}... tiene {len(all_messages)} mensajes. Iniciando resumen...")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        # Separar: mensajes a resumir vs. mensajes recientes a conservar
        messages_to_summarize = all_messages[:-KEEP_RECENT]
        
        if len(messages_to_summarize) == 0:
            logger.info("➡️  No hay mensajes suficientes para resumir después de reservar los recientes.")
            return
        
        # Formatear el bloque de conversación para el prompt
        conversation_block = "\n".join([
            f"[{msg['role'].upper()}] ({msg.get('created_at', 'N/A')}): {msg['content']}"
            for msg in messages_to_summarize
        ])
        
        # Obtener timestamps del bloque
        messages_start = messages_to_summarize[0].get("created_at")
        messages_end = messages_to_summarize[-1].get("created_at")
        message_count = len(messages_to_summarize)
        
        # Invocar Gemini para generar el resumen.
        # [P1-18] Modelo configurable vía `MEMORY_SUMMARY_MODEL` env var
        # con default `gemini-2.5-flash` (válido y económico). ANTES
        # hardcoded a `gemini-3.1-flash-lite-preview` — ID inválido que
        # rechaza el SDK con 404, fallando el resumen silenciosamente.
        summary_llm = ChatGoogleGenerativeAI(
            model=MEMORY_SUMMARY_MODEL,
            temperature=0.1,
            google_api_key=os.environ.get("GEMINI_API_KEY")
        )
        
        prompt = SUMMARY_PROMPT.format(conversation_block=conversation_block)
        response = summary_llm.invoke(prompt)
        summary_text = response.content
        
        # [P1-20] Orden crítico Supabase ↔ LangGraph para evitar carrera
        # de rollback parcial inconsistente.
        #
        # ANTES el flujo era:
        #   1. save_summary (Supabase)          → COMMIT A
        #   2. delete_old_messages (Supabase)   → COMMIT A'
        #   3. purge_langgraph_checkpoint       → COMMIT B (otro pool)
        # Si A/A' commiteaban pero B fallaba (DB blip, schema mismatch,
        # SDK rejection), los mensajes estaban borrados de Supabase
        # PERO LangGraph aún los retenía → re-inyección al LLM en el
        # siguiente turno de la conversación, fugando contexto privado
        # ya supuestamente resumido.
        #
        # AHORA:
        #   1. save_summary (idempotente sobre messages_end timestamp)
        #   2. purge_langgraph_checkpoint(raise_on_failure=True) ANTES del
        #      delete_old_messages. Si LangGraph falla, propaga al except,
        #      `delete_old_messages` NUNCA se ejecuta → Supabase conserva
        #      los mensajes (estado consistente: ambos lados los retienen,
        #      next cron tick lo retoma desde el principio).
        #   3. delete_old_messages SOLO si la purga ya tuvo éxito.
        #
        # Peor caso: paso 2 falla → redundancia (summary persistido +
        # mensajes en Supabase + LangGraph) pero NO inconsistencia de
        # safety/contexto.

        # 1. Guardar el resumen en la base de datos.
        save_summary(
            session_id=session_id,
            summary=summary_text,
            messages_start=messages_start,
            messages_end=messages_end,
            message_count=message_count
        )

        # 2. 🔗 SINCRONIZACIÓN CRÍTICA: Purgar checkpoint de LangGraph
        # ANTES de borrar de Supabase (P1-20). Si el purge falla, el
        # `raise_on_failure=True` propaga la excepción al except global
        # del summarize_and_prune, ABORTAMOS el delete_old_messages, y
        # la racha cuenta como fallo (P1-18 logger + P1-19 alert si N).
        purge_langgraph_checkpoint(
            session_id, keep_recent=KEEP_RECENT, raise_on_failure=True
        )

        # 3. Eliminar los mensajes ya resumidos de Supabase. Solo se
        # ejecuta si la purga de LangGraph tuvo éxito → consistencia
        # garantizada.
        delete_old_messages(session_id, before_timestamp=messages_end)
        
        # === LÓGICA DE RESUMEN JERÁRQUICO (MASTER SUMMARY → JSON EVOLUTIVO) ===
        summaries = get_summaries(session_id)
        if summaries and len(summaries) >= MAX_SUMMARIES:
            logger.info(f"🔄 [MEMORY MANAGER] Condensando {len(summaries)} resúmenes en un Estado Evolutivo JSON...")
            
            # Detectar si ya existe un Master Summary JSON previo entre los summaries
            prior_state = None
            narrative_summaries = []
            for s in summaries:
                summary_text = s.get('summary', '').strip()
                if summary_text.startswith('{'):
                    try:
                        prior_state = json.loads(summary_text)
                        logger.info("📋 [MEMORY MANAGER] Estado Evolutivo anterior detectado. Se actualizará incrementalmente.")
                    except json.JSONDecodeError:
                        narrative_summaries.append(s)
                else:
                    narrative_summaries.append(s)
            
            # Construir el bloque de resúmenes narrativos (solo los no-JSON)
            summaries_block = "\n\n".join([f"- {s.get('summary', '')}" for s in narrative_summaries])
            
            # Instrucción de estado previo
            if prior_state:
                prior_instruction = PRIOR_STATE_INSTRUCTION_WITH_DATA.format(
                    prior_state=json.dumps(prior_state, ensure_ascii=False, indent=2)
                )
            else:
                prior_instruction = PRIOR_STATE_INSTRUCTION_EMPTY
            
            master_prompt = MASTER_SUMMARY_PROMPT.format(
                summaries_block=summaries_block,
                prior_state_instruction=prior_instruction
            )
            
            # Usar .with_structured_output() para garantizar JSON perfecto (0% fallos de parseo).
            # [P1-18] Mismo modelo configurable que el summary_llm de arriba.
            structured_summary_llm = ChatGoogleGenerativeAI(
                model=MEMORY_SUMMARY_MODEL,
                temperature=0.1,
                google_api_key=os.environ.get("GEMINI_API_KEY")
            ).with_structured_output(EvolutionaryState)
            
            master_response = structured_summary_llm.invoke(master_prompt)
            
            # Pydantic garantiza el schema — convertir a JSON string para guardar
            if hasattr(master_response, 'model_dump'):
                master_dict = master_response.model_dump()
            else:
                master_dict = master_response.dict()
            
            master_summary_text = json.dumps(master_dict, ensure_ascii=False, indent=2)
            logger.info("✅ [MEMORY MANAGER] Estado Evolutivo generado con Structured Output (Pydantic). 0% riesgo de parseo.")
            
            # Las fechas del Master Summary cubren desde el primero hasta el último
            master_start = summaries[0].get("messages_start")
            master_end = summaries[-1].get("messages_end")
            master_count = sum([s.get("message_count", 0) for s in summaries])
            
            summary_ids = [s.get("id") for s in summaries if s.get("id")]
            if summary_ids:
                # 1. Archivar los resúmenes originales en cold storage para no perder detalles finos
                archive_summaries(summaries)
                logger.info(f"📦 [MEMORY MANAGER] {len(summaries)} resúmenes originales enviados a cold storage.")
                
                # 2. Borrarlos de la tabla activa de memoria de trabajo
                delete_summaries(summary_ids)
                
            save_summary(
                session_id=session_id,
                summary=master_summary_text,
                messages_start=master_start,
                messages_end=master_end,
                message_count=master_count
            )
            logger.info("✅ [MEMORY MANAGER] Estado Evolutivo JSON guardado exitosamente.")
        # =======================================================
        
        duration = round(time.time() - start_time, 2)
        
        logger.info(f"✅ [MEMORY MANAGER] Resumen completado en {duration}s")
        logger.info(f"   📝 {message_count} mensajes resumidos y eliminados")
        logger.info(f"   💾 {KEEP_RECENT} mensajes recientes conservados")
        logger.info(f"   📋 Resumen: {summary_text[:150]}...")
        logger.info(f"{'='*60}\n")
        
    except Exception as e:
        # [P1-18] Nunca bloquear al usuario por un error de resumen, PERO
        # promovemos a logger.error con contador in-memory para que SRE
        # detecte degradación sistémica (modelo inválido, API key rotada,
        # cuota agotada, tabla missing). El primer fallo + cada 10
        # subsiguientes salen a `error`; los intermedios a `warning` para
        # no spamear pero mantener visibilidad.
        error_str = str(e)
        if "Server disconnected" not in error_str:
            _summarize_failures["count"] += 1
            _summarize_failures["last_error"] = repr(e)
            # [P1-NEW-6 · 2026-05-11] Timestamp para backoff exponencial.
            # Persistido aquí (NO en el path feliz) — _summary_backoff_should_skip
            # lo lee para calcular `elapsed > wait_seconds`.
            _summarize_failures["last_attempt_at"] = time.time()
            n = _summarize_failures["count"]
            if n == 1 or n % 10 == 0:
                logger.error(
                    f"[MEMORY MANAGER/P1-18] summarize_and_prune falló (#{n}) — "
                    f"posible modelo inválido (`MEMORY_SUMMARY_MODEL={MEMORY_SUMMARY_MODEL!r}`), "
                    f"API key rotada, cuota agotada, o DB blip. "
                    f"session_id={session_id} error={error_str}"
                )
            else:
                logger.warning(
                    f"[MEMORY MANAGER/P1-18] summarize failure #{n}: {error_str}"
                )

            # [P1-19] Persistir alerta en `system_alerts` cuando se cruza
            # el threshold de fallos consecutivos. Re-disparada cada
            # `_SUMMARY_FAILURE_ALERT_REPEAT_EVERY` para que SRE vea que
            # la situación persiste si ya cerró la alerta inicial.
            if (
                n == _SUMMARY_FAILURE_ALERT_THRESHOLD
                or (n > _SUMMARY_FAILURE_ALERT_THRESHOLD
                    and (n - _SUMMARY_FAILURE_ALERT_THRESHOLD) % _SUMMARY_FAILURE_ALERT_REPEAT_EVERY == 0)
            ):
                _persist_summary_failure_alert(session_id, error_str, n)
    else:
        # [P1-18] Reset del contador en path de éxito (auto-recovery sin
        # restart del proceso). Si el resumen funcionó tras una racha de
        # fallos, lo loggeamos a info para que SRE vea la recuperación.
        # (`global _summarize_failures` movido al inicio de la función
        # por P1-NEW-6 — Python 3.10+ exige declararlo antes del primer uso.)
        if _summarize_failures["count"] > 0:
            prior_count = _summarize_failures["count"]
            logger.info(
                f"[MEMORY MANAGER/P1-18] Resumen recuperado tras "
                f"{prior_count} fallo(s). "
                f"Último error: {_summarize_failures['last_error']!r}"
            )
            # [P1-NEW-6 · 2026-05-11] Reset incluye `last_attempt_at`
            # para que el backoff arranque limpio si futura racha aparece.
            _summarize_failures = {"count": 0, "last_error": None, "last_attempt_at": 0.0}
            # [P1-19] Marcar la alerta como resuelta en system_alerts si
            # la habíamos persistido (cruzó el threshold). Best-effort:
            # mismo patrón defensivo que `_persist_summary_failure_alert`.
            if prior_count >= _SUMMARY_FAILURE_ALERT_THRESHOLD:
                try:
                    from db_core import execute_sql_write
                    execute_sql_write(
                        "UPDATE system_alerts SET resolved_at = NOW() "
                        "WHERE alert_key = %s AND resolved_at IS NULL",
                        ("memory_summary_failures",),
                    )
                except Exception as _resolve_err:
                    logger.debug(
                        f"[MEMORY MANAGER/P1-19] No se pudo resolver alerta: {_resolve_err}"
                    )
    finally:
        release_summarizing_lock(session_id)


# ============================================================
# FUNCIÓN: Construir Contexto de Memoria para Prompts
# ============================================================
def _format_evolutionary_state(state_json: dict) -> str:
    """
    Transforma un JSON de Estado Evolutivo en texto legible para inyectar en el prompt del agente.
    """
    parts: list[str] = []
    
    if state_json.get("tendencias_adherencia"):
        parts.append(f"📊 ADHERENCIA: {state_json['tendencias_adherencia']}")
    
    if state_json.get("nivel_estres"):
        parts.append(f"🧠 ESTRÉS: {state_json['nivel_estres']}")
    
    if state_json.get("patrones_cocina"):
        parts.append(f"🍳 COCINA: {state_json['patrones_cocina']}")
    
    feedback = state_json.get("feedback_recurrente", {})
    if feedback:
        if feedback.get("le_encanta"):
            parts.append(f"❤️ LE ENCANTA: {', '.join(feedback['le_encanta'])}")
        if feedback.get("rechaza"):
            parts.append(f"🚫 RECHAZA: {', '.join(feedback['rechaza'])}")
        if feedback.get("preferencias_horario"):
            parts.append(f"🕐 HORARIOS: {feedback['preferencias_horario']}")
    
    if state_json.get("evolucion_dieta"):
        parts.append(f"📈 EVOLUCIÓN DIETA: {state_json['evolucion_dieta']}")
    
    if state_json.get("contexto_personal"):
        parts.append(f"👤 CONTEXTO PERSONAL: {state_json['contexto_personal']}")
    
    if state_json.get("notas_chef"):
        parts.append(f"👨‍🍳 NOTAS CHEF: {state_json['notas_chef']}")
    
    return "\n".join(parts)


def build_memory_context(session_id: str) -> dict:
    """
    Construye el contexto completo de memoria para inyectar en los prompts.
    Soporta tanto resúmenes narrativos (texto) como Estado Evolutivo (JSON).
    
    Retorna:
        dict con:
        - "summary_context": str — Resúmenes previos concatenados (puede ser "")
        - "recent_messages": list — Mensajes recientes no resumidos
        - "full_context_str": str — Texto listo para inyectar en un prompt
    """
    summaries = get_summaries(session_id)
    recent_messages = get_recent_messages(session_id, limit=KEEP_RECENT)
    
    # Construir el string de resúmenes
    summary_context = ""
    if summaries:
        summary_parts: list[str] = []
        for s in summaries:
            summary_text = s.get('summary', '').strip()
            period = f"({s.get('messages_start', '?')} → {s.get('messages_end', '?')})"
            
            # Detectar si es un JSON de Estado Evolutivo
            if summary_text.startswith('{'):
                try:
                    state_json = json.loads(summary_text)
                    summary_parts.append(
                        f"[ESTADO EVOLUTIVO DEL PACIENTE — Período {period}]:\n```json\n{json.dumps(state_json, ensure_ascii=False, indent=2)}\n```"
                    )
                    continue
                except json.JSONDecodeError:
                    pass  # Tratar como texto normal
            
            # Resumen narrativo normal
            summary_parts.append(f"[Período {period}]:\n{summary_text}")
        
        summary_context = (
            "\n--- MEMORIA A LARGO PLAZO (Resúmenes de conversaciones anteriores) ---\n"
            + "\n\n".join(summary_parts)
            + "\n---------------------------------------------------------------------\n"
        )
    
    # Construir contexto completo como string
    recent_str = ""
    if recent_messages:
        recent_parts = [
            f"[{msg['role'].upper()}]: {msg['content']}"
            for msg in recent_messages
        ]
        recent_str = (
            "\n--- HISTORIAL RECIENTE (Últimos mensajes) ---\n"
            + "\n".join(recent_parts)
            + "\n----------------------------------------------\n"
        )
    
    full_context = summary_context + recent_str
    
    return {
        "summary_context": summary_context,
        "recent_messages": recent_messages,
        "full_context_str": full_context,
    }
