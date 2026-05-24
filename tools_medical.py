import os
import logging
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


def _medical_tool_model_name() -> str:
    """[P1-FLASH-MODEL-GA · 2026-05-21 · P3-COST-CUT-AUX · 2026-05-22]
    Modelo de la herramienta médica via knob. Histórico:
      - Pre-fix: hardcoded `gemini-3-flash-preview` → sujeto a cuota free-tier
        20 RPD aunque billing esté activo.
      - P1-FLASH-MODEL-GA (2026-05-21): default `gemini-3.5-flash` (GA, paid-tier).
      - P3-COST-CUT-AUX (2026-05-22): default `gemini-3.1-flash-lite`. Razón:
        la tool es Q&A clínico determinístico (temp=0.0), fact-lookup-style
        sobre interacciones medicamentos↔alimentos / alergias cruzadas /
        contraindicaciones. NO requiere razonamiento creativo ni multimodal.
        Lite tiene base de conocimiento clínico suficiente para los patterns
        comunes; el system prompt fuerza brevedad clínica. Fallback explícito
        a "Sin contraindicaciones médicas conocidas" cubre casos donde lite
        no tiene confianza. Ahorro: 6× cheaper per call.

    Rollback sin redeploy: `MEALFIT_MEDICAL_TOOL_MODEL=gemini-3.5-flash` (o
    `gemini-3-flash-preview` para path histórico). Test parser-based:
    `test_p3_cost_cut_aux.py::test_medical_tool_default_is_lite`.
    """
    return os.environ.get("MEALFIT_MEDICAL_TOOL_MODEL", "gemini-3.1-flash-lite")


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
        clinical_llm = ChatGoogleGenerativeAI(
            model=_medical_tool_model_name(),
            temperature=0.0, # Determinista para evitar alucinaciones
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            max_retries=1
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
