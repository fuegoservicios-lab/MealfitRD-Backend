import os
import io
import base64
import asyncio
import threading
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
# [P1-MEAL-SCAN-GEMMA · 2026-07-12] Tercer provider: Ollama local (gemma4:12b
# vía túnel SSH reverso laptop→VPS, MISMO transporte que el escáner de Nevera
# de routers/user_data.py — knobs compartidos MEALFIT_VISION_PROVIDER=ollama,
# MEALFIT_OLLAMA_BASE_URL, MEALFIT_VISION_MODEL, MEALFIT_VISION_TIMEOUT_S).
# Antes: `_env_str(choices={disabled, openai_compatible})` degradaba el valor
# prod `ollama` a `disabled` con WARNING → "Escanear comida" muerto mientras
# "Escanear mi nevera" (que lee el env crudo) funcionaba con el mismo knob.
_VISION_PROVIDER_OLLAMA = "ollama"
# "off" es el alias que usa el escáner de Nevera para apagar — aceptarlo aquí
# evita el split-brain de un mismo env var con dos vocabularios.
_VISION_PROVIDER_OFF = "off"
_warned_vision_disabled = False

# [P1-MEAL-SCAN-GEMMA · 2026-07-12] Single-flight COMPARTIDO de todo análisis
# de visión local: la GPU de gemma (4GB VRAM) no soporta concurrencia, y tanto
# "Escanear comida" (Dashboard) como "Escanear mi nevera" (Pantry) golpean el
# mismo Ollama. Un solo Lock module-level para ambos surfaces.
_VISION_SINGLE_FLIGHT_LOCK = threading.Lock()


def get_vision_single_flight_lock() -> threading.Lock:
    """Lock único para llamadas al modelo de visión local (Ollama)."""
    return _VISION_SINGLE_FLIGHT_LOCK


def _vision_provider() -> str:
    return _env_str(
        "MEALFIT_VISION_PROVIDER",
        _VISION_PROVIDER_DISABLED,
        choices={
            _VISION_PROVIDER_DISABLED,
            _VISION_PROVIDER_OFF,
            _VISION_PROVIDER_OPENAI_COMPATIBLE,
            _VISION_PROVIDER_OLLAMA,
        },
    )


def _vision_model_name() -> str:
    return _env_str("MEALFIT_VISION_MODEL", "")


def _vision_base_url() -> str:
    return _env_str("MEALFIT_VISION_BASE_URL", "")


def _ollama_base_url() -> str:
    # Mismo default que routers/user_data.py (túnel SSH reverso en el VPS).
    return (os.environ.get("MEALFIT_OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")


def _ollama_model_name() -> str:
    # Crudo (sin _env_str) por paridad exacta con el escáner de Nevera.
    return os.environ.get("MEALFIT_VISION_MODEL") or "gemma4:12b"


def _ollama_timeout_s() -> int:
    """Timeout del roundtrip Ollama — mismo knob/clamps que user_data.py.
    gemma local por túnel es LENTO (30-120s por foto); el clamp de 30s del
    knob MEALFIT_VISION_LLM_TIMEOUT_S (pensado para providers cloud) mataría
    cada scan."""
    try:
        v = int(os.environ.get("MEALFIT_VISION_TIMEOUT_S", "240"))
    except ValueError:
        return 240
    return min(600, max(30, v))


def is_vision_enabled() -> bool:
    """True si hay provider de visión activo con config mínima completa."""
    provider = _vision_provider()
    if provider == _VISION_PROVIDER_OLLAMA:
        # Defaults completos (base URL + modelo) — siempre operable.
        return True
    return (
        provider == _VISION_PROVIDER_OPENAI_COMPATIBLE
        and bool(_vision_model_name())
        and bool(_vision_base_url())
    )


def is_vision_local() -> bool:
    """[P1-MEAL-SCAN-GEMMA] True si el análisis corre en el modelo LOCAL
    (gemma vía Ollama, costo cero). routers/diary.py lo usa para NO quemar
    crédito del cap mensual (`log_api_usage`) en scans gratis — doctrina
    P1-NEVERA-QUOTA-EXEMPT: el paywall es para costo LLM real."""
    return _vision_provider() == _VISION_PROVIDER_OLLAMA


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
        # [P1-CHAT-VISION-GEMMA] Shape estable con el path v4 (clasificación).
        "photo_kind": "otro",
        "items": [],
        "meal_name": "",
        "calories": 0,
        "protein": 0,
        "carbs": 0,
        "healthy_fats": 0,
    }


# ---------------------------------------------------------------------------
# [P1-MEAL-SCAN-GEMMA · 2026-07-12] Análisis de PLATO con gemma local (Ollama).
# Espejo del escáner de Nevera (user_data._ollama_vision_scan) pero con schema
# de comida: is_food + nombre corto + macros TOTALES del plato visible. El
# `format` JSON-schema de Ollama fuerza salida parseable; think=False porque
# gemma4 arranca en thinking-mode y devuelve content vacío sin él.
# ---------------------------------------------------------------------------

# [P1-CHAT-VISION-GEMMA · 2026-07-12] v4: la MISMA pasada clasifica la foto.
# 'plato' → macros para registrar en el diario; 'items' → lista de alimentos
# sueltos/compra lista para la Nevera (el chat-agent la ofrece vía
# modify_pantry_inventory); 'otro' → no es comida. Un solo roundtrip a la GPU
# cubre ambos poderes del escáner en el chat del Agente.
_MEAL_VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "photo_kind": {"type": "string", "enum": ["plato", "items", "otro"]},
        "is_food": {"type": "boolean"},
        "meal_name": {"type": "string"},
        "description": {"type": "string"},
        "calories": {"type": "number"},
        "protein": {"type": "number"},
        "carbs": {"type": "number"},
        "healthy_fats": {"type": "number"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "quantity": {"type": "number"},
                    "unit": {"type": "string"},
                },
                "required": ["name", "quantity", "unit"],
            },
        },
    },
    "required": ["photo_kind", "is_food", "meal_name", "description",
                 "calories", "protein", "carbs", "healthy_fats", "items"],
}

# [P1-MEAL-SCAN-DR-DISHES · 2026-07-12] Desambiguacion de platos criollos.
# Vivo (owner, 2 scans del mismo tres golpes): v1 nombro el mangu 'arroz';
# v2 dejo de confundirlo pero lo OMITIO por completo (nombre y macros) — el
# cap de 6 palabras recortaba justo la base del plato. v3: inventario completo
# de componentes OBLIGATORIO en description, nombres propios de platos
# criollos, base de carbohidrato primero en el nombre, y la suma de macros
# amarrada a TODOS los componentes listados (540 kcal para un plato de ~750
# era el mangu ausente de la suma).
_MEAL_VISION_PROMPT = (
    "Eres un nutricionista dominicano. Mira la foto y clasifica 'photo_kind': "
    "'plato' si es comida PREPARADA o SERVIDA lista para comer (plato, bowl, "
    "sandwich, desayuno servido); 'items' si son alimentos SUELTOS o una "
    "compra (funda del super, productos con empaque, frutas o verduras "
    "crudas, el interior de una nevera o despensa); 'otro' si no hay comida. "
    "SI ES 'otro': is_food=false, macros en 0, meal_name vacio, items vacio. "
    "SI ES 'plato' (deja items vacio): haz un INVENTARIO en 'description': "
    "lista TODOS los componentes visibles sin omitir ninguno - la base de "
    "carbohidrato (mangu, arroz, yuca, platano, pan), las proteinas (huevo, "
    "salami, queso frito, pollo, carne) y las guarniciones (cebolla roja "
    "encurtida, aguacate, ensalada). "
    "OJO: el MANGU es un pure COMPACTO y LISO de platano verde majado (masa "
    "color crema, casi siempre coronada con cebolla roja encurtida) - NO lo "
    "confundas con arroz, que se ve como granos sueltos y separados, y NUNCA "
    "lo omitas: si ves esa masa con cebollita encurtida, el plato lleva "
    "mangu. Para 'meal_name' (max 8 palabras): si el plato es un clasico "
    "dominicano con nombre propio, usalo (ej: 'Los tres golpes' = mangu + "
    "huevo frito + salami + queso frito; 'La bandera' = arroz + habichuelas "
    "+ carne; mofongo, sancocho); si no, nombra los componentes principales "
    "EMPEZANDO por la base (ej: 'Mangu con huevo, salami y queso'). "
    "Macros: estima calorias, proteina, carbohidratos y grasas de CADA "
    "componente del inventario POR SEPARADO y SUMA los totales de la porcion "
    "visible (no por 100g; ej: mangu ~300 kcal, huevo frito ~110, 2 rodajas "
    "de salami ~180, queso frito ~150). El total debe cubrir TODOS los "
    "componentes de description - un componente listado pero fuera de la "
    "suma es un error. Se realista con porciones dominicanas. "
    "SI ES 'items' (macros en 0, meal_name vacio): llena 'items' con cada "
    "alimento visible e identificable con certeza: 'name' generico en "
    "espanol dominicano SIN marca (ej: 'pechuga de pollo', 'arroz blanco', "
    "'platano verde', 'huevos', 'leche'), 'quantity' = NUMERO DE ENVASES O "
    "PIEZAS visibles (1 paquete, 2 latas, 6 huevos) - NUNCA el peso impreso "
    "en el empaque, 'unit' una de: unidad, lb, g, paquete, botella, lata, "
    "taza, funda. En 'description' resume la compra. NO inventes alimentos "
    "que no se vean claramente; si dudas, omitelo. "
    "Responde SOLO el JSON."
)

# Clamps espejo de ConsumedMealRequest (routers/diary.py) — el registro final
# los revalida, esto solo evita precargar absurdos en el modal.
_MEAL_MACRO_CAPS = {"calories": 10000, "protein": 1000, "carbs": 2000, "healthy_fats": 1000}


def _sane_item_qty(qty, unit) -> float:
    """[P1-CHAT-VISION-GEMMA] Espejo de user_data._sane_scan_qty (lección
    P1-PANTRY-SCAN-QTY): envase discreto con qty absurda (>12) casi siempre es
    el peso impreso mal leído → colapsar a 1."""
    try:
        q = float(qty or 1)
    except (TypeError, ValueError):
        q = 1.0
    u = str(unit or "").strip().lower()
    if u in ("g", "gramo", "gramos"):
        return max(10.0, min(5000.0, q))
    if u in ("lb", "libra", "libras"):
        return max(0.25, min(10.0, q))
    if u in ("unidad", "unidades"):
        return float(max(1, min(30, round(q))))
    q = round(q)
    if q > 12:
        return 1.0
    return float(max(1, q))


def _fmt_item_phrase(name: str, qty: float, unit: str) -> str:
    """Frase '2 unidades de Manzana' — EXACTO el formato que documenta
    `tools.modify_pantry_inventory(items_to_add=...)`, para que el chat-agent
    pueda copiar los items del análisis directo a la tool sin reinterpretar."""
    q = int(qty) if float(qty).is_integer() else qty
    u = str(unit or "unidad").strip().lower()
    if q != 1 and u in ("unidad", "paquete", "botella", "lata", "taza", "funda", "libra"):
        # Plural español: consonante final → +es (unidad→unidades), vocal → +s.
        u += "es" if u[-1] not in "aeiou" else "s"
    return f"{q} {u} de {name}"


def _coerce_meal_scan(data: dict) -> dict:
    """Normaliza la salida cruda de gemma al contrato de process_image_with_vision.
    Pura (sin IO) para testearla directo: clamps, is_food=False ⇒ macros 0,
    strings capadas, items sanitizados. Fail-open a 0 en macros no numéricas."""
    def _macro(key: str) -> int:
        try:
            n = int(round(float(data.get(key) or 0)))
        except (TypeError, ValueError):
            n = 0
        return max(0, min(_MEAL_MACRO_CAPS[key], n))

    is_food = bool(data.get("is_food"))
    description = str(data.get("description") or "").strip()[:600]
    meal_name = str(data.get("meal_name") or "").strip()[:120]

    kind = str(data.get("photo_kind") or "").strip().lower()
    if kind not in ("plato", "items", "otro"):
        # Compat: salida vieja/parcial sin photo_kind → derivar de is_food.
        kind = "plato" if is_food else "otro"

    # ---- Modo ITEMS: compra/alimentos sueltos → lista para la Nevera ----
    if kind == "items":
        items = []
        for it in (data.get("items") or [])[:30]:
            name = str((it or {}).get("name") or "").strip()[:60]
            if not name:
                continue
            unit = str((it or {}).get("unit") or "unidad").strip().lower()[:20]
            qty = _sane_item_qty((it or {}).get("quantity"), unit)
            items.append({"name": name, "quantity": qty, "unit": unit})
        if not items:
            # Clasificó 'items' pero no identificó nada usable → tratar como
            # no-comida honesta en vez de una compra vacía.
            return {
                "photo_kind": "otro",
                "is_food": False,
                "items": [],
                "description": description or "No se detectaron alimentos identificables en la imagen.",
                "meal_name": "",
                "calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0,
            }
        frases = ", ".join(_fmt_item_phrase(i["name"], i["quantity"], i["unit"]) for i in items)
        return {
            "photo_kind": "items",
            "is_food": True,
            "items": items,
            # Description determinista desde los items SANITIZADOS (no el texto
            # libre de gemma): es lo que el chat-agent copia a la tool.
            "description": f"Alimentos detectados (compra/items, no es un plato servido): {frases}."[:900],
            "meal_name": "",
            "calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0,
        }

    # ---- Modo OTRO / no-comida ----
    if kind == "otro" or not is_food:
        return {
            "photo_kind": "otro",
            "is_food": False,
            "items": [],
            "description": description or "No se detectó comida en la imagen.",
            "meal_name": "",
            "calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0,
        }

    # ---- Modo PLATO: contrato v3 intacto ----
    result = {
        "photo_kind": "plato",
        "is_food": True,
        "items": [],
        "description": description or "Comida detectada en la foto.",
        "meal_name": meal_name,
        "calories": _macro("calories"),
        "protein": _macro("protein"),
        "carbs": _macro("carbs"),
        "healthy_fats": _macro("healthy_fats"),
    }
    if result["calories"] > 0 or result["protein"] > 0:
        # Paridad con el path openai_compatible: la estimación viaja también en
        # la description que se persiste al Diario Visual (contexto del coach).
        result["description"] += (
            f" (Estimación: Calorías: {result['calories']}, Proteína: {result['protein']}g, "
            f"Carbohidratos: {result['carbs']}g, Grasas Saludables: {result['healthy_fats']}g)"
        )
    return result


def _ollama_meal_scan(image_b64: str) -> dict:
    """Roundtrip síncrono a Ollama (invocar via asyncio.to_thread). Lanza en
    error de red/JSON — el caller mapea al payload analysis_failed."""
    import httpx
    import json as _json
    body = {
        "model": _ollama_model_name(),
        "stream": False,
        "think": False,  # gemma4: thinking ON por default → content vacío sin esto
        "format": _MEAL_VISION_SCHEMA,
        "options": {"temperature": 0.1, "num_ctx": 8192},
        "messages": [{"role": "user", "content": _MEAL_VISION_PROMPT, "images": [image_b64]}],
    }
    resp = httpx.post(f"{_ollama_base_url()}/api/chat", json=body, timeout=_ollama_timeout_s())
    resp.raise_for_status()
    content = ((resp.json().get("message") or {}).get("content")) or "{}"
    return _coerce_meal_scan(_json.loads(content))


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

    # [P1-MEAL-SCAN-GEMMA · 2026-07-12] Provider LOCAL: gemma vía Ollama.
    # Single-flight compartido con el escáner de Nevera (misma GPU): si hay
    # otro análisis en vuelo devolvemos `busy=True` al instante en vez de
    # encolar minutos — el modal muestra "escáner ocupado, reintenta".
    if _vision_provider() == _VISION_PROVIDER_OLLAMA:
        lock = get_vision_single_flight_lock()
        if not lock.acquire(blocking=False):
            payload = _vision_disabled_payload()
            payload["busy"] = True
            payload["description"] = "El escáner está procesando otra foto."
            return payload
        try:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            # to_thread: httpx síncrono con timeout de minutos NO puede correr
            # en el event loop (workers=1 — estancaría TODAS las requests).
            return await asyncio.to_thread(_ollama_meal_scan, image_b64)
        except Exception as e:
            # [P3-VISION-FAIL-ERROR-LOG] error-level para Sentry (paridad con
            # el except del path openai_compatible más abajo).
            logger.error(
                f"❌ [VISION] meal-scan gemma falló (modelo={_ollama_model_name()!r}, "
                f"base={_ollama_base_url()!r}): {type(e).__name__}: {e}"
            )
            payload = _vision_disabled_payload()
            payload["description"] = "Error analizando imagen."
            return payload
        finally:
            lock.release()

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
