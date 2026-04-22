import os
import logging
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

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
            model="gemini-3-flash-preview",
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
