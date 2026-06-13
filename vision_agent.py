import os
import io
import base64
from cache_manager import centralized_cache
from knobs import _env_str, _env_float  # [P3-VISION-MODEL-KNOB · 2026-05-20] / [P2-LLM-TIMEOUT-SWEEP · 2026-05-30]
# [P0-DEEPSEEK-MIGRATION · 2026-06-12] Gemini eliminado. ChatDeepSeek acepta
# base_url/api_key explícitos — el path vision lo reusa apuntando a CUALQUIER
# provider OpenAI-compatible con soporte de imágenes.
from llm_provider import ChatDeepSeek
from embeddings_provider import get_text_embedding
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from db import save_visual_entry
import logging

logger = logging.getLogger(__name__)  # [P2-LOGGER-MIGRATION · 2026-05-12]


# [P0-DEEPSEEK-MIGRATION · 2026-06-12] Vision como provider PLUGGABLE.
# DeepSeek-V4 NO acepta input de imágenes (verificado api-docs 2026-06-12),
# así que el análisis visual queda detrás de un provider OpenAI-compatible
# configurable (Qwen-VL, GLM-4V, moonshot, etc.) — mismo patrón que
# embeddings_provider. Mientras `MEALFIT_VISION_PROVIDER=disabled` (default
# hasta que el owner elija provider), `process_image_with_vision` retorna el
# payload `analysis_failed=True` SIN llamar a ningún API: el frontend ya
# distingue ese estado ("la IA no pudo analizar") de "no es comida".
#
# Para activar sin tocar código:
#   MEALFIT_VISION_PROVIDER=openai_compatible
#   MEALFIT_VISION_MODEL=<model-id-con-vision>
#   MEALFIT_VISION_BASE_URL=<base-url-openai-compatible>
#   VISION_API_KEY=<key>   (env var, NUNCA hardcodeada)
# Tooltip-anchor: P3-VISION-MODEL-KNOB (knob model preservado).
_VISION_PROVIDER_DISABLED = "disabled"
_VISION_PROVIDER_OPENAI_COMPATIBLE = "openai_compatible"
_warned_vision_disabled = False


def _vision_provider() -> str:
    return _env_str(
        "MEALFIT_VISION_PROVIDER",
        _VISION_PROVIDER_DISABLED,
        choices={_VISION_PROVIDER_DISABLED, _VISION_PROVIDER_OPENAI_COMPATIBLE},
    )


def _vision_model_name() -> str:
    return _env_str("MEALFIT_VISION_MODEL", "")


def _vision_base_url() -> str:
    return _env_str("MEALFIT_VISION_BASE_URL", "")


def is_vision_enabled() -> bool:
    """True si hay provider de visión activo con config mínima completa."""
    return (
        _vision_provider() == _VISION_PROVIDER_OPENAI_COMPATIBLE
        and bool(_vision_model_name())
        and bool(_vision_base_url())
    )


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


# [P0-DEEPSEEK-MIGRATION · 2026-06-12] El timeout de embeddings
# (`MEALFIT_EMBEDDING_LLM_TIMEOUT_S`) ahora vive en `embeddings_provider`.


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

def _vision_disabled_payload() -> dict:
    """Payload soft-fail cuando el provider de visión está deshabilitado.
    Misma shape que el except path — el frontend ya maneja `analysis_failed`."""
    return {
        "description": "Análisis de imagen no disponible temporalmente.",
        "is_food": False,
        "analysis_failed": True,
        "meal_name": "",
        "calories": 0,
        "protein": 0,
        "carbs": 0,
        "healthy_fats": 0,
    }


async def process_image_with_vision(image_bytes: bytes) -> dict:
    """
    Toma los bytes de una imagen, usa el provider de visión configurado para
    extraer una descripción y determina si contiene alimentos usando
    structured output. Con `MEALFIT_VISION_PROVIDER=disabled` retorna el
    payload `analysis_failed` sin tocar ningún API.
    """
    global _warned_vision_disabled
    if not is_vision_enabled():
        if not _warned_vision_disabled:
            logger.warning(
                "⚠️ [VISION] Provider de visión DESACTIVADO "
                "(MEALFIT_VISION_PROVIDER=disabled — DeepSeek no acepta "
                "imágenes; provider pendiente de configurar). El Diario "
                "Visual y 'Escanear comida' responderán analysis_failed. "
                "Este aviso se emite una vez por proceso."
            )
            _warned_vision_disabled = True
        return _vision_disabled_payload()

    try:
        # [P3-VISION-MODEL-KNOB · 2026-05-20] Modelo via knob (no hardcoded).
        # [P0-DEEPSEEK-MIGRATION] Cliente OpenAI-compatible con base_url/key
        # del provider de visión configurado.
        llm = ChatDeepSeek(
            model=_vision_model_name(),
            temperature=0.1,
            base_url=_vision_base_url(),
            api_key=(os.environ.get("VISION_API_KEY") or "").strip() or None,
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

# [P0-DEEPSEEK-MIGRATION · 2026-06-12 → P1-COHERE-EMBED-V4] El embedding
# "multimodal" siempre vectorizó el TEXTO de la descripción (no la imagen),
# así que delega al provider de `embeddings_provider` (hoy Cohere Embed v4,
# que ADEMÁS soporta imágenes si en el futuro se quiere búsqueda
# imagen-a-imagen). Con provider inactivo, retorna None y
# `async_process_and_save_visual_entry` aborta el guardado con warning
# (path pre-existente).


def get_multimodal_embedding(text: str, purpose: str = "query") -> list:
    """
    Genera un embedding de la descripción via `embeddings_provider`.
    Usa el mismo patrón de caché centralizado que fact_extractor.get_embedding().

    [P1-COHERE-EMBED-V4] `purpose="document"` SOLO en los paths que
    PERSISTEN a `visual_diary.embedding` (este módulo + routers/diary);
    las búsquedas del agente usan el default `"query"` — Embed v4 es
    asimétrico y esa distinción es la palanca de precisión del retrieval.
    """
    from embeddings_provider import get_embeddings_model_id

    if purpose not in ("query", "document"):
        purpose = "query"
    result = _cached_multimodal_embedding(text, get_embeddings_model_id(), purpose)
    return list(result) if result else None

@centralized_cache(ttl_seconds=3153600000, maxsize=10000, cache_empty=False)
def _cached_multimodal_embedding(text: str, model_id: str, purpose: str):
    """Wrapper cacheado para embeddings del visual diary (Redis o local
    OrderedDict). `model_id`/`purpose` son args para VERSIONAR la cache key
    por espacio vectorial y lado (P1-COHERE-EMBED-V4 — sin esto un switch
    de provider serviría vectores stale del espacio anterior desde Redis)."""
    try:
        emb = get_text_embedding(text, purpose=purpose)
        if not emb:
            return None
        logger.info(f"🔑 [VISUAL EMBEDDING CACHE] MISS → Generado embedding ({model_id}/{purpose}) para: '{text[:50]}...'")
        return list(emb)
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
    
    # Paso 2: Embedding (se PERSISTE en visual_diary → lado document)
    embedding = get_multimodal_embedding(description, purpose="document")
    
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
