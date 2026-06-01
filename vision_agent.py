import os
import io
import base64
from cache_manager import centralized_cache
from knobs import _env_str, _env_float  # [P3-VISION-MODEL-KNOB · 2026-05-20] / [P2-LLM-TIMEOUT-SWEEP · 2026-05-30]
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from db import save_visual_entry
import logging

logger = logging.getLogger(__name__)  # [P2-LOGGER-MIGRATION · 2026-05-12]


# [P3-VISION-MODEL-KNOB · 2026-05-20] Knob para overridear el modelo Gemini
# Vision sin redeploy. Cierra el gap "hardcoded preview model" identificado
# en `docs/gaps-audit-2026-05.md` (D3 / R2): el modelo
# `gemini-3.1-pro-preview` ya causó CB stale 4.4 días en 2026-05-11 (CLAUDE.md
# convención P3-PREVIEW-MODEL-KNOB). Si Google deprecia el preview sin aviso,
# `vision_agent` rompe → la cadena entera de Diario Visual + cruce-de-silos
# con `fact_extractor` queda silenciada hasta el próximo redeploy.
#
# Con el knob:
#   - Default = current production model (cero cambio de comportamiento).
#   - SRE puede setear `MEALFIT_VISION_MODEL=gemini-3.1-pro` (stable, sin
#     `-preview`) en EasyPanel y reiniciar el worker — vision retoma operación.
#
# Auto-registry en `_KNOBS_REGISTRY` vía `_env_str` → visible en
# `/health/version` (admin gated). Patrón espejo de `proactive_agent.py`
# (P3-PREVIEW-MODEL-KNOB).
# Tooltip-anchor: P3-VISION-MODEL-KNOB.
def _vision_model_name() -> str:
    # [P1-ALL-MODELS-GA · 2026-05-21] Default migrado de `gemini-3.1-pro-preview`
    # a `gemini-3.5-flash`. Vision pierde capacidad multimodal Pro pero gana
    # GA stability + paid-tier directo. Rollback: `MEALFIT_VISION_MODEL=gemini-3.1-pro-preview`.
    return _env_str("MEALFIT_VISION_MODEL", "gemini-3.5-flash")


# [P2-LLM-TIMEOUT-SWEEP · 2026-05-30] Timeout per-invoke del LLM Vision.
# Pre-fix: el constructor de `ChatGoogleGenerativeAI` se creaba SIN `timeout=`,
# así que un Gemini colgado (socket estancado, sobrecarga del provider, quota
# silenciosa) hacía que `await llm.ainvoke(...)` bloqueara indefinidamente. Como
# `process_image_with_vision` es awaitada en el handler async `api_diary_upload`
# (routers/diary.py) sobre el ÚNICO worker uvicorn, un invoke colgado estanca el
# event loop para TODAS las requests. El SDK default es `timeout=None` (espera
# infinita en sockets colgados). El constructor `timeout=` propaga al deadline
# del request gRPC; al exceder, raise DeadlineExceeded → lo captura el
# `except Exception` (línea ~91) que ya retorna el fallback graceful. Vision usa
# ventana holgada (30s) porque el análisis multimodal es más lento que texto.
# Knob auto-registrado vía `_env_float`. Tooltip-anchor: P2-LLM-TIMEOUT-SWEEP.
def _vision_llm_timeout_s() -> float:
    return _env_float(
        "MEALFIT_VISION_LLM_TIMEOUT_S",
        30.0,
        validator=lambda v: 0.0 < v <= 120.0,
    )


# [P2-LLM-TIMEOUT-SWEEP · 2026-05-30] Cohorte omitida: el `GoogleGenerativeAIEmbeddings`
# multimodal no acepta `timeout=` — solo `client_args` llega al cliente httpx
# y acota el deadline. Sin esto, un socket de embedding colgado bloquea el
# event loop / thread. Mismo knob que fact_extractor/constants.
def _embeddings_llm_timeout_s() -> float:
    return _env_float(
        "MEALFIT_EMBEDDING_LLM_TIMEOUT_S",
        15.0,
        validator=lambda v: 0.0 < v <= 60.0,
    )


# Definimos el modelo de salida estructurada para capturar la descripción
class ImageDescription(BaseModel):
    description: str = Field(description="Descripción concisa de los alimentos, ingredientes o comida visible en la imagen.")
    # [P2-DIARY-SCAN-MACROS · 2026-05-30] Nombre corto del platillo para el
    # flujo "Escanear comida → registrar macros" (modal del Dashboard). El
    # `description` es verboso (se persiste en visual_diary); `meal_name` es el
    # label corto que precarga el campo editable del modal. Tooltip-anchor:
    # P2-DIARY-SCAN-MACROS.
    meal_name: str = Field(description="Nombre corto del platillo en español dominicano (máx ~6 palabras), p.ej. 'Mangú con salami y queso frito'. Vacío si no es comida.")
    is_food: bool = Field(description="¿Contiene esta imagen comida, ingredientes o una nevera?")
    calories: int = Field(description="Estimación de calorías totales en la imagen. Usa 0 si no es comida.")
    protein: int = Field(description="Estimación de gramos de proteína totales en la imagen. Usa 0 si no es comida.")
    carbs: int = Field(description="Estimación de gramos de carbohidratos totales en la imagen. Usa 0 si no es comida.")
    healthy_fats: int = Field(description="Estimación de gramos de grasas saludables totales en la imagen. Usa 0 si no es comida.")

async def process_image_with_vision(image_bytes: bytes) -> dict:
    """
    Toma los bytes de una imagen, usa Gemini Vision para extraer una descripción
    y determina si contiene alimentos usando structured output.
    """
    try:
        # [P3-VISION-MODEL-KNOB · 2026-05-20] Modelo via knob (no hardcoded).
        # Default preserva el modelo preview actual; SRE puede swap sin redeploy
        # si Google deprecia. Tooltip-anchor: P3-VISION-MODEL-KNOB.
        llm = ChatGoogleGenerativeAI(
            model=_vision_model_name(),
            temperature=0.1,
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            timeout=_vision_llm_timeout_s(),  # [P2-LLM-TIMEOUT-SWEEP · 2026-05-30] no colgar el event loop
        ).with_structured_output(ImageDescription)
        
        # Convertimos los bytes a base64 para enviarlo a la API de LangChain/Gemini
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe detalladamente todos los alimentos, ingredientes o platillos que ves en esta imagen. Si es una nevera, lista el contenido visible. Si no hay comida, indícalo. Da también un nombre corto del platillo en español dominicano (`meal_name`, máx ~6 palabras). También proporciona una estimación de las calorías, gramos de proteína, gramos de carbohidratos y gramos de grasas saludables (solo el número) totales en la imagen (usa 0 si no es comida)."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        )

        response = await llm.ainvoke([message])

        description = response.description if response and hasattr(response, 'description') else "Imagen sin descripción clara."
        is_food = response.is_food if response and hasattr(response, 'is_food') else False

        # [P2-DIARY-SCAN-MACROS · 2026-05-30] Las macros se computaban pero solo
        # se retornaba `calories` — el modal "Escanear comida" necesita las 4
        # macros + el nombre corto. Defaults a 0/"" si no es comida (la API
        # nunca devuelve None para estos campos → frontend no se rompe).
        calories = 0
        protein = 0
        carbs = 0
        healthy_fats = 0
        meal_name = ""

        if is_food:
            calories = response.calories if hasattr(response, 'calories') else 0
            protein = response.protein if hasattr(response, 'protein') else 0
            carbs = response.carbs if hasattr(response, 'carbs') else 0
            healthy_fats = response.healthy_fats if hasattr(response, 'healthy_fats') else 0
            meal_name = (response.meal_name if hasattr(response, 'meal_name') else "") or ""
            if calories > 0 or protein > 0:
                description += f" (Estimación: Calorías: {calories}, Proteína: {protein}g, Carbohidratos: {carbs}g, Grasas Saludables: {healthy_fats}g)"
        return {
            "description": description,
            "is_food": is_food,
            "meal_name": meal_name,
            "calories": calories,
            "protein": protein,
            "carbs": carbs,
            "healthy_fats": healthy_fats,
        }

    except Exception as e:
        # [P3-VISION-FAIL-ERROR-LOG · 2026-05-30] error-level (NO warning) para
        # que Sentry/log-aggregation capturen una caída SOSTENIDA de Gemini
        # Vision (quota 429 RESOURCE_EXHAUSTED / DeadlineExceeded / modelo
        # deprecado). Pre-fix solo `logger.warning` → invisible al threshold
        # default de Sentry (convención del repo: error/trace lo captura, warn
        # no); un outage del "Escanear comida" + diario visual degradaba en
        # silencio para todos los usuarios sin señal operativa. Mismo patrón que
        # get_embedding (fact_extractor.py). El soft-fail return (analysis_failed)
        # se mantiene — esto solo sube el nivel de log. Tooltip-anchor: P3-VISION-FAIL-ERROR-LOG.
        logger.error(f"❌ [VISION] process_image_with_vision falló (modelo={_vision_model_name()!r}): {type(e).__name__}: {e}")
        # [P2-DIARY-SCAN-MACROS · 2026-05-30] `analysis_failed=True` distingue
        # "el analizador falló" (timeout, 429 RESOURCE_EXHAUSTED, provider caído)
        # de "no es comida" (is_food=False en el path de éxito). Sin esta señal
        # el modal del Dashboard mostraría "No detectamos comida" cuando en
        # realidad la IA no pudo correr — mensaje engañoso. El path de éxito NO
        # lleva esta key (default False en los consumidores).
        return {
            "description": "Error analizando imagen.",
            "is_food": False,
            "analysis_failed": True,
            "meal_name": "",
            "calories": 0,
            "protein": 0,
            "carbs": 0,
            "healthy_fats": 0,
        }

# [2026-05-06] Modelo de embeddings MULTIMODAL para visual diary y chat agent.
# Configurable vía env. Default `gemini-embedding-2` (GA estable, multimodal —
# texto + imágenes). Mismo benchmark que la variante preview (MTEB Multilingual
# 69.9, TextCaps 89.6, Docci 93.4) pero sin las restricciones de cuota del
# preview ni cambios disruptivos en futuras revisiones.
#
# Si Google saca un nuevo modelo multimodal mejor (ej. gemini-embedding-3),
# cambias el knob sin tocar código. Si la cuota se agota crónicamente, puedes
# temporalmente apuntar al text-only y aceptar perder multimodalidad hasta el
# reset diario.
GEMINI_EMBEDDING_MULTIMODAL_MODEL = os.environ.get(
    "MEALFIT_GEMINI_EMBEDDING_MULTIMODAL_MODEL",
    "models/gemini-embedding-2",
)


def get_multimodal_embedding(text: str) -> list:
    """
    Genera un embedding de la descripción usando Gemini.
    Usa el mismo patrón de caché centralizado que fact_extractor.get_embedding().
    """
    result = _cached_multimodal_embedding(text)
    return list(result) if result else None

@centralized_cache(ttl_seconds=3153600000, maxsize=10000, cache_empty=False)
def _cached_multimodal_embedding(text: str):
    """Wrapper cacheado para embeddings multimodales (Redis o local OrderedDict)."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_MULTIMODAL_MODEL,
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            # [P2-LLM-TIMEOUT-SWEEP · 2026-05-30] deadline httpx (ver helper).
            client_args={"timeout": _embeddings_llm_timeout_s()},
        )
        emb = embeddings.embed_query(text)
        logger.info(f"🔑 [VISUAL EMBEDDING CACHE] MISS → Generado embedding para: '{text[:50]}...'")
        return list(emb[:768])
    except Exception as e:
        # [P3-VISION-FAIL-ERROR-LOG · 2026-05-30] error-level para Sentry (ver
        # process_image_with_vision). Mismo gap de observabilidad.
        logger.error(f"❌ [VISION EMBEDDING] _cached_multimodal_embedding falló: {type(e).__name__}: {e}")
        return None

async def async_process_and_save_visual_entry(user_id: str, file_bytes: bytes, image_url: str, user_message: str = ""):
    """
    Procesador en segundo plano (Background Task).
    1. Analiza la imagen con Gemini Vision
    2. Si es comida, saca el embedding de la descripción
    3. Lo guarda en la tabla visual_diary.
    4. Cruce de silos: Extrae hechos nutricionales basados en la foto y el comentario del usuario.
    """
    from fact_extractor import async_extract_and_save_facts

    logger.info("\n-------------------------------------------------------------")
    logger.info("📸 [VISION AGENT] Procesando nueva imagen subida...")
    
    # Paso 1: Visión
    vision_result = await process_image_with_vision(file_bytes)
    
    if not vision_result.get("is_food"):
        logger.info("➡️ La imagen fue ignorada porque no se detectaron alimentos.")
        return

    description = vision_result.get("description", "")
    logger.info(f"✅ Descripción generada: '{description}'")
    
    # Paso 2: Embedding
    embedding = get_multimodal_embedding(description)
    
    if not embedding:
        logger.warning("⚠️ No se pudo vectorizar la imagen. Abortando guardado.")
        return
        
    # Paso 3: Base de Datos Visual Diary
    logger.info(f"📦 Guardando entrada visual en la DB (Vector 768d)...")
    save_visual_entry(
        user_id=user_id,
        image_url=image_url,
        description=description,
        embedding=embedding
    )
    logger.info("✅ ¡Imagen registrada en el Diario Visual con éxito!")

    # Paso 4: Cruce de Silos Multimodal (Diario Visual -> Hechos de Usuario)
    # Combinamos lo que dijo el usuario con lo que la IA vio en la foto
    # Ej: "Me cayó pesado esto" + "Plato de mangú con salami, queso frito y cebolla verde"
    combined_context = f"Comentario del usuario sobre su comida actual: '{user_message}'. Lo que estaba comiendo (según análisis de imagen): '{description}'"
    
    logger.info("🔄 [VISION AGENT] Enviando contexto combinado al Extractor de Hechos...")
    # async_extract_and_save_facts es síncrona — usamos asyncio.to_thread para no bloquear el event loop
    import asyncio
    await asyncio.to_thread(async_extract_and_save_facts, user_id, combined_context)
