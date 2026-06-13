import os
import logging
from langchain_core.tools import tool
# [P0-DEEPSEEK-MIGRATION · 2026-06-12] Gemini → DeepSeek.
from llm_provider import ChatDeepSeek, DEEPSEEK_FLASH
from langchain_core.messages import SystemMessage, HumanMessage
from knobs import _env_float

logger = logging.getLogger(__name__)


# [P2-LLM-TIMEOUT-SWEEP · 2026-05-30] Cohorte omitida del sweep original. La tool
# `consultar_base_datos_medica` se despacha al `_FACT_CHECK_EXECUTOR`
# (ThreadPoolExecutor, default 2 workers; graph_orchestrator.py) cuyo thread NO se
# puede matar. El `asyncio.wait_for(asyncio.shield(future), 20s)` libera al caller
# async pero deja el future corriendo "para que pueda completar y liberar el slot".
# Si `clinical_llm.invoke` cuelga (socket Gemini sin respuesta) sin timeout, el slot
# NUNCA se libera → tras 2 colgadas ambos workers quedan ocupados forever → todo
# fact-check médico bloquea + degrada al fallback precautorio SIN alerta. El
# `timeout=` debe ser < el cap downstream de 20s para que el deadline del LLM
# dispare primero y libere el slot del pool.
def _medical_tool_llm_timeout_s() -> float:
    return _env_float(
        "MEALFIT_MEDICAL_TOOL_LLM_TIMEOUT_S",
        15.0,
        validator=lambda v: 0.0 < v <= 60.0,
    )


def _medical_tool_model_name() -> str:
    """[P0-DEEPSEEK-MIGRATION · 2026-06-12] Modelo de la herramienta médica
    via knob. Default DeepSeek V4 Flash: la tool es Q&A clínico
    determinístico (temp=0.0), fact-lookup-style sobre interacciones
    medicamentos↔alimentos / alergias cruzadas / contraindicaciones — NO
    requiere razonamiento creativo ni multimodal. El system prompt fuerza
    brevedad clínica y el fallback explícito a "Sin contraindicaciones
    médicas conocidas" cubre casos de baja confianza.

    Rollback / escalado sin redeploy: `MEALFIT_MEDICAL_TOOL_MODEL=deepseek-v4-pro`.
    """
    return os.environ.get("MEALFIT_MEDICAL_TOOL_MODEL", DEEPSEEK_FLASH)


@tool
def consultar_base_datos_medica(query: str) -> str:
    """
    Simula una consulta a una base de datos de nutrición clínica y médica.
    Usa esta herramienta para investigar interacciones de medicamentos con alimentos,
    alergias cruzadas (ej. látex-aguacate), contraindicaciones de enfermedades específicas o
    lineamientos nutricionales estrictos.

    Args:
        query: La pregunta clínica específica o el alimento/condición a investigar.
               (Ej: '¿Cuáles son las reacciones cruzadas comunes con alergia al látex?' o
               '¿Puede un hipertenso comer plátano con sal?').

    Returns:
        Un resumen clínico basado en hechos médicos.
    """
    logger.info(f"🏥 [MEDICAL TOOL] Consultando DB médica sobre: '{query}'")

    # Utilizamos un LLM ligero pero capaz de razonar como base de datos clínica
    try:
        clinical_llm = ChatDeepSeek(
            model=_medical_tool_model_name(),
            temperature=0.0, # Determinista para evitar alucinaciones
            max_retries=1,
            # [P2-LLM-TIMEOUT-SWEEP · 2026-05-30] deadline < 20s del _FACT_CHECK
            # _TOOL_TIMEOUT para que el LLM corte primero y libere el slot del pool.
            timeout=_medical_tool_llm_timeout_s(),
        )
        
        sys_prompt = """
        Eres un motor de búsqueda de una base de datos médica y de nutrición clínica.
        Tu único propósito es proveer hechos médicos concretos, comprobados y basados en ciencia.
        No des consejos médicos generales ni saludes.
        Responde de manera directa, listando contraindicaciones, alergias cruzadas o efectos metabólicos 
        relevantes según la evidencia médica actual.
        Si no hay evidencia de interacción o riesgo, decláralo explícitamente: 'Sin contraindicaciones médicas conocidas'.
        Sé breve, preciso y estrictamente clínico.
        """
        
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"Consulta clínica: {query}")
        ]
        
        response = clinical_llm.invoke(messages)
        return response.content
        
    except Exception as e:
        logger.error(f"❌ [MEDICAL TOOL] Error consultando DB médica: {e}")
        return f"Error en el sistema de consulta médica. Asume riesgo precautorio si hay dudas clínicas. Error: {str(e)}"
