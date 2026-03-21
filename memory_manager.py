# backend/memory_manager.py
"""
Gestión de Memoria y Resúmenes Progresivos.
Proceso asíncrono que resume conversaciones antiguas para mantener
el contexto del usuario vivo sin saturar la ventana de tokens de Gemini.
"""

import os
import time
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import RemoveMessage
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


# ============================================================
# SINCRONIZACIÓN: Purgar Checkpoint de LangGraph
# ============================================================
def purge_langgraph_checkpoint(session_id: str, keep_recent: int = KEEP_RECENT):
    """
    Purga los mensajes antiguos del checkpoint de LangGraph (PostgresSaver)
    para mantenerlo sincronizado con la tabla agent_messages de Supabase.
    
    Sin esta función, LangGraph acumula TODOS los mensajes históricos en su
    propio estado inmutable y los re-inyecta silenciosamente al modelo en cada
    turno, causando explosión de tokens y costos disparados.
    
    Usa la API RemoveMessage de LangChain para eliminar mensajes del state.
    """
    if not connection_pool:
        print("⚠️ [MEMORY MANAGER] No hay connection_pool, no se puede purgar checkpoint de LangGraph.")
        return
    
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from agent import chat_builder
        
        checkpointer = PostgresSaver(connection_pool)
        graph = chat_builder.compile(checkpointer=checkpointer)
        
        config = {"configurable": {"thread_id": session_id}}
        state = graph.get_state(config)
        
        if not state or not state.values:
            print("➡️ [MEMORY MANAGER] No hay estado de LangGraph para esta sesión. Nada que purgar.")
            return
        
        messages = state.values.get("messages", [])
        
        if len(messages) <= keep_recent:
            print(f"➡️ [MEMORY MANAGER] Solo {len(messages)} mensajes en checkpoint, no se necesita purga.")
            return
        
        # Calcular qué mensajes eliminar (todos excepto los keep_recent más recientes)
        messages_to_remove = messages[:-keep_recent]
        
        # Crear RemoveMessage para cada mensaje a eliminar
        remove_messages = [RemoveMessage(id=m.id) for m in messages_to_remove if hasattr(m, 'id') and m.id]
        
        if not remove_messages:
            print("➡️ [MEMORY MANAGER] No se encontraron IDs de mensajes para purgar.")
            return
        
        print(f"🔄 [MEMORY MANAGER] Purgando {len(remove_messages)} mensajes del checkpoint de LangGraph...")
        
        # Usar update_state para aplicar las eliminaciones
        graph.update_state(config, {"messages": remove_messages})
        
        print(f"✅ [MEMORY MANAGER] {len(remove_messages)} mensajes eliminados del checkpoint de LangGraph. State sincronizado.")
        
    except Exception as e:
        # Nunca bloquear al usuario por un error de sincronización
        print(f"⚠️ [MEMORY MANAGER] Error no-crítico purgando checkpoint de LangGraph: {e}")


# ============================================================
# PROMPT DE RESUMEN
# ============================================================
SUMMARY_PROMPT = """Eres el Agente de Memoria de MealfitRD. Tu trabajo es condensar la NARRATIVA y el FEEDBACK de la conversación reciente en un resumen conciso y estructurado.

REGLAS CRÍTICAS DE ESPECIALIZACIÓN:
1. IGNORA LOS HECHOS DUROS: NO captures alergias, condiciones médicas, macros numéricos, ni dietas rígidas. (Otro agente ya está guardando esto en una base de datos vectorial estricta).
2. ENFÓCATE EN LA NARRATIVA: Captura cómo se sintió el usuario, su nivel de estrés, su adherencia al plan o si tuvo poco tiempo para cocinar.
3. CAPTURA FEEDBACK ESPECÍFICO: Registra qué opinó sobre platos recientes (ej: "Le pareció muy pesada la cena", "Le encantó la receta de pollo caribeño", "Quiere opciones más dulces en el desayuno").
4. NO incluyas detalles innecesarios como timestamps, IDs o metadata técnica.
5. Escribe el resumen en español con BULLET POINTS organizados así:
   • ESTADO ANÍMICO: ...
   • ADHERENCIA: ...
   • FEEDBACK COMIDA: ...
   • CONTEXTO PERSONAL: ...
6. Máximo 200 palabras.

BLOQUE DE CONVERSACIÓN A RESUMIR:
{conversation_block}

Genera el resumen estructurado ahora."""

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


MASTER_SUMMARY_PROMPT = """Eres el Administrador de Memoria a Largo Plazo de MealfitRD.
Tu tarea es tomar una lista cronológica de pequeños resúmenes y condensarlos en un Estado Evolutivo del paciente.

REGLAS CRÍTICAS:
1. IGNORA HECHOS DUROS: NO incluyas alergias, condiciones médicas ni macros numéricos exactos (eso vive en otra DB vectorial).
2. CONSOLIDA LA EVOLUCIÓN: Captura cómo ha evolucionado su relación con la dieta, patrones de estrés, tiempo disponible, y qué preparaciones le han funcionado.
3. CONSOLIDA EL FEEDBACK: Identifica patrones de rechazo hacia texturas, sabores o ingredientes, y preferencias recurrentes.
4. Elimina redundancias. Si un dato viejo fue reemplazado por algo más reciente, usa el reciente.
5. Escribe todos los valores en español.

{prior_state_instruction}

RESÚMENES A CONDENSAR:
{summaries_block}

Genera el Estado Evolutivo ahora."""

PRIOR_STATE_INSTRUCTION_WITH_DATA = """ESTADO EVOLUTIVO ANTERIOR (ACTUALÍZALO, NO LO REEMPLACES DESDE CERO):
Ya existe un Estado Evolutivo previo. Tu tarea es ACTUALIZARLO con la nueva información de los resúmenes.
- Preserva todos los datos anteriores que sigan siendo válidos.
- Agrega nueva información de los resúmenes recientes.
- Si hay conflicto, prioriza la información más reciente.

Estado anterior:
{prior_state}"""

PRIOR_STATE_INSTRUCTION_EMPTY = """No hay un Estado Evolutivo anterior. Crea uno nuevo desde cero basándote en los resúmenes proporcionados."""


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
    """
    if not acquire_summarizing_lock(session_id):
        print(f"⚠️ [MEMORY MANAGER] Resumen ya en progreso para sesión {session_id}. Mitigando Condición de Carrera.")
        return

    try:
        all_messages = get_memory(session_id)
        
        if not all_messages:
            return
            
        total_chars = sum(len(msg.get("content", "")) for msg in all_messages)
        
        if total_chars <= MAX_CHAR_THRESHOLD:
            return  # No hay suficientes caracteres para detonar el resumen
        
        print(f"\n{'='*60}")
        print(f"🧠 [MEMORY MANAGER] Sesión {str(session_id)[:8]}... tiene {len(all_messages)} mensajes. Iniciando resumen...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Separar: mensajes a resumir vs. mensajes recientes a conservar
        messages_to_summarize = all_messages[:-KEEP_RECENT]
        
        if len(messages_to_summarize) == 0:
            print("➡️  No hay mensajes suficientes para resumir después de reservar los recientes.")
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
        
        # Invocar Gemini para generar el resumen
        summary_llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-pro-preview",  # Modelo rápido y económico para resúmenes
            temperature=0.1,
            google_api_key=os.environ.get("GEMINI_API_KEY")
        )
        
        prompt = SUMMARY_PROMPT.format(conversation_block=conversation_block)
        response = summary_llm.invoke(prompt)
        summary_text = response.content
        
        # Guardar el resumen en la base de datos
        save_summary(
            session_id=session_id,
            summary=summary_text,
            messages_start=messages_start,
            messages_end=messages_end,
            message_count=message_count
        )
        
        # Eliminar los mensajes ya resumidos de Supabase
        delete_old_messages(session_id, before_timestamp=messages_end)
        
        # 🔗 SINCRONIZACIÓN CRÍTICA: Purgar también el checkpoint de LangGraph
        # Sin esto, LangGraph re-inyecta silenciosamente los mensajes borrados
        purge_langgraph_checkpoint(session_id, keep_recent=KEEP_RECENT)
        
        # === LÓGICA DE RESUMEN JERÁRQUICO (MASTER SUMMARY → JSON EVOLUTIVO) ===
        summaries = get_summaries(session_id)
        if summaries and len(summaries) >= MAX_SUMMARIES:
            print(f"🔄 [MEMORY MANAGER] Condensando {len(summaries)} resúmenes en un Estado Evolutivo JSON...")
            
            # Detectar si ya existe un Master Summary JSON previo entre los summaries
            prior_state = None
            narrative_summaries = []
            for s in summaries:
                summary_text = s.get('summary', '').strip()
                if summary_text.startswith('{'):
                    try:
                        prior_state = json.loads(summary_text)
                        print("📋 [MEMORY MANAGER] Estado Evolutivo anterior detectado. Se actualizará incrementalmente.")
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
            
            # Usar .with_structured_output() para garantizar JSON perfecto (0% fallos de parseo)
            structured_summary_llm = ChatGoogleGenerativeAI(
                model="gemini-3.1-pro-preview",
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
            print("✅ [MEMORY MANAGER] Estado Evolutivo generado con Structured Output (Pydantic). 0% riesgo de parseo.")
            
            # Las fechas del Master Summary cubren desde el primero hasta el último
            master_start = summaries[0].get("messages_start")
            master_end = summaries[-1].get("messages_end")
            master_count = sum([s.get("message_count", 0) for s in summaries])
            
            summary_ids = [s.get("id") for s in summaries if s.get("id")]
            if summary_ids:
                # 1. Archivar los resúmenes originales en cold storage para no perder detalles finos
                archive_summaries(summaries)
                print(f"📦 [MEMORY MANAGER] {len(summaries)} resúmenes originales enviados a cold storage.")
                
                # 2. Borrarlos de la tabla activa de memoria de trabajo
                delete_summaries(summary_ids)
                
            save_summary(
                session_id=session_id,
                summary=master_summary_text,
                messages_start=master_start,
                messages_end=master_end,
                message_count=master_count
            )
            print("✅ [MEMORY MANAGER] Estado Evolutivo JSON guardado exitosamente.")
        # =======================================================
        
        duration = round(time.time() - start_time, 2)
        
        print(f"✅ [MEMORY MANAGER] Resumen completado en {duration}s")
        print(f"   📝 {message_count} mensajes resumidos y eliminados")
        print(f"   💾 {KEEP_RECENT} mensajes recientes conservados")
        print(f"   📋 Resumen: {summary_text[:150]}...")
        print(f"{'='*60}\n")
        
    except Exception as e:
        # Nunca bloquear al usuario por un error de resumen
        error_str = str(e)
        if "Server disconnected" not in error_str:
            print(f"⚠️  [MEMORY MANAGER] Error no-crítico durante resumen: {error_str}")
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
                    formatted = _format_evolutionary_state(state_json)
                    summary_parts.append(
                        f"[ESTADO EVOLUTIVO DEL PACIENTE — Período {period}]:\n{formatted}"
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
