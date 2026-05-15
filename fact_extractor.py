import os
import json
import logging
import hashlib
import threading
import time as _time_module
from typing import Any, Callable
from cache_manager import centralized_cache
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

from knobs import _env_str, _env_float

logger = logging.getLogger(__name__)

from db import (
    save_user_fact, search_user_facts, delete_user_fact, search_user_facts_hybrid,
    get_user_facts_by_metadata, acquire_fact_lock, release_fact_lock,
    enqueue_pending_fact, dequeue_pending_facts, delete_pending_facts
)

# ============================================================
# [P3-FACT-SHADOW-AB · 2026-05-14] Shadow A/B PRO → FLASH
# ============================================================
# Los 2 callsites de structured extraction de este módulo
# (extract_facts + _run_fact_pipeline batch contradiction) están
# hardcoded a `gemini-3.1-pro-preview`. Histórico el costo de
# correr PRO en cada mensaje del chat es la fuente más cara del
# bill LLM tras la generación del plan. Structured extraction
# (con `.with_structured_output()`) es el sweet spot histórico
# donde FLASH suele igualar PRO porque el schema constrains el
# output — pero migrar a ciegas arriesga regresiones silenciosas
# en personalización (omisión de facts, canonicalización mala,
# merges incorrectos que corrompen datos del usuario).
#
# Mecánica: `_invoke_with_shadow` corre PRO sync (UX truth) y, si
# el knob está activado y el user pasa el sampling hash-based,
# dispara FLASH en daemon thread. Compara outputs estructuralmente
# y persiste a `pipeline_metrics` con node='fact_extractor_shadow_diff'.
# Cero impacto UX (siempre retorna PRO). Tras 1 semana de data:
#   - exact match ≥95% → migrar default a FLASH con confianza.
#   - 80-95% → analizar field_diffs, ajustar prompt si patrón sistémico.
#   - <80% → mantener PRO, descartar opción.
#
# Knobs:
#   - MEALFIT_FACT_EXTRACTOR_SHADOW_MODEL (str, default '' = off)
#       Setear a 'gemini-3-flash-preview' (o 'gemini-3.1-flash-lite-preview'
#       para test más agresivo) para activar.
#   - MEALFIT_FACT_EXTRACTOR_SHADOW_SAMPLE_RATE (float, default 0.1)
#       Fracción de users que ejecutan el shadow (estable por hash(user_id)).
_FACT_SHADOW_MODEL = _env_str("MEALFIT_FACT_EXTRACTOR_SHADOW_MODEL", "")
_FACT_SHADOW_SAMPLE_RATE = _env_float(
    "MEALFIT_FACT_EXTRACTOR_SHADOW_SAMPLE_RATE",
    0.1,
    validator=lambda v: 0.0 <= v <= 1.0,
)

# ============================================================
# [P2-NEW-FACTEX-PRIMARY-MODEL-KNOB · 2026-05-15] Knobs para los modelos
# PRIMARY (truth-path PRO) y ROUTER (gate flash-lite). Pre-fix los 3
# callsites tenían el modelo hardcoded inline (extract_facts, batch
# contradiction de _run_fact_pipeline, y el router should_extract_facts).
# El SHADOW ya tenía knob (P3-FACT-EXTRACTOR-SHADOW-AB · 2026-05-14), pero
# el path productivo PRIMARY no — convención `P3-PREVIEW-MODEL-KNOB ·
# 2026-05-12` exige knob para TODOS los modelos preview de Google ("CB row
# stale por preview model durante 4.4 días" en audit 2026-05-11). Sin
# estos knobs, una deprecation de Google tira la extracción de hechos
# hasta redeploy (45min Easypanel cold start). Tooltip-anchor:
# P2-NEW-FACTEX-PRIMARY-MODEL-KNOB.
_FACT_EXTRACTOR_PRIMARY_MODEL = _env_str(
    "MEALFIT_FACT_EXTRACTOR_PRIMARY_MODEL",
    "gemini-3.1-pro-preview",
)
_FACT_EXTRACTOR_ROUTER_MODEL = _env_str(
    "MEALFIT_FACT_EXTRACTOR_ROUTER_MODEL",
    "gemini-3.1-flash-lite-preview",
)


def _fact_extractor_primary_model_name() -> str:
    """[P2-NEW-FACTEX-PRIMARY-MODEL-KNOB] Helper SSOT — devuelve el modelo
    PRO usado en `extract_facts` y `_run_fact_pipeline` (batch contradicción).
    Centralizar en helper permite a tests parser-based escanear el callsite
    y validar que NO hay literal hardcoded fuera de este módulo."""
    return _FACT_EXTRACTOR_PRIMARY_MODEL


def _fact_extractor_router_model_name() -> str:
    """[P2-NEW-FACTEX-PRIMARY-MODEL-KNOB] Helper SSOT — modelo flash-lite
    del router `should_extract_facts` (gate cheap-first antes del PRO call)."""
    return _FACT_EXTRACTOR_ROUTER_MODEL


def _should_run_shadow(user_id: Optional[str]) -> bool:
    """Decide determinísticamente si correr el shadow para `user_id`.

    - Knob `MEALFIT_FACT_EXTRACTOR_SHADOW_MODEL` debe estar set y non-empty.
    - `user_id` debe ser truthy (sin user_id no podemos samplear estable).
    - Bucket determinístico vía SHA-256: mismo user_id siempre cae en el
      mismo bucket (evita ruido por sesión, A/B reproducible).
    """
    if not _FACT_SHADOW_MODEL or not user_id:
        return False
    if _FACT_SHADOW_SAMPLE_RATE <= 0.0:
        return False
    if _FACT_SHADOW_SAMPLE_RATE >= 1.0:
        return True
    h = hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 100
    return bucket < int(_FACT_SHADOW_SAMPLE_RATE * 100)


def _persist_shadow_diff(user_id: Optional[str], payload: dict) -> None:
    """Best-effort insert a `pipeline_metrics`. Silencia excepciones —
    el shadow NO debe romper el path productivo."""
    try:
        from db_core import execute_sql_write
        execute_sql_write(
            """
            INSERT INTO pipeline_metrics
                (user_id, session_id, node, duration_ms, retries,
                 tokens_estimated, confidence, metadata)
            VALUES (%s, NULL, %s, 0, 0, 0, 0, %s::jsonb)
            """,
            (
                user_id,
                "fact_extractor_shadow_diff",
                json.dumps(payload, ensure_ascii=False, default=str),
            ),
        )
    except Exception as e:
        logger.debug(f"[FACT-SHADOW] persist fallo (no fatal): {e}")


def _diff_facts_model(pro_result: Any, flash_result: Any) -> dict:
    """Compara outputs de `FactsModel` por las dimensiones load-bearing
    downstream: cantidad de facts + set de (category, ingrediente_canonico).
    Esas son las claves que afectan filtering crítico (alergia/condicion_medica)."""
    pro_facts = (pro_result.facts if pro_result and hasattr(pro_result, "facts") else []) or []
    flash_facts = (flash_result.facts if flash_result and hasattr(flash_result, "facts") else []) or []

    def _tuples(facts):
        out = set()
        for f in facts:
            md = getattr(f, "metadata", None)
            if md is None:
                continue
            cat = getattr(md, "category", "")
            canon = (getattr(md, "ingrediente_canonico", None) or "").lower()
            out.add((cat, canon))
        return out

    pro_set = _tuples(pro_facts)
    flash_set = _tuples(flash_facts)

    return {
        "pro_count": len(pro_facts),
        "flash_count": len(flash_facts),
        "count_match": len(pro_facts) == len(flash_facts),
        "category_canonico_set_match": pro_set == flash_set,
        "only_in_pro": [list(t) for t in sorted(pro_set - flash_set)],
        "only_in_flash": [list(t) for t in sorted(flash_set - pro_set)],
    }


def _diff_contradiction_result(pro_result: Any, flash_result: Any) -> dict:
    """Compara outputs de `BatchContradictionResult`. Las claves
    load-bearing son `ids_to_delete` (esto borra datos del user) y la
    cantidad de contradicciones/merges (sobre/sub-detección)."""
    def _extract(r):
        if r is None:
            return [], [], set()
        contradictions = getattr(r, "contradictions", None) or []
        merges = getattr(r, "merges", None) or []
        ids_to_delete = set()
        for c in contradictions:
            for _id in (getattr(c, "ids_to_delete", None) or []):
                ids_to_delete.add(str(_id))
        for m in merges:
            for _id in (getattr(m, "ids_to_delete", None) or []):
                ids_to_delete.add(str(_id))
        return contradictions, merges, ids_to_delete

    pro_c, pro_m, pro_ids = _extract(pro_result)
    flash_c, flash_m, flash_ids = _extract(flash_result)

    return {
        "pro_contradictions_count": len(pro_c),
        "pro_merges_count": len(pro_m),
        "flash_contradictions_count": len(flash_c),
        "flash_merges_count": len(flash_m),
        "ids_to_delete_set_match": pro_ids == flash_ids,
        "only_in_pro_ids": sorted(pro_ids - flash_ids),
        "only_in_flash_ids": sorted(flash_ids - pro_ids),
    }


def _invoke_with_shadow(
    *,
    prompt: str,
    output_schema: Any,
    pro_model: str,
    pro_temperature: float,
    callsite_tag: str,
    user_id: Optional[str],
    differ: Callable[[Any, Any], dict],
) -> Any:
    """[P3-FACT-SHADOW-AB] Invoca PRO sync (UX truth path) y, si el shadow
    está habilitado y el user pasa el sampling, dispara el modelo shadow
    en daemon thread para A/B. Retorna SIEMPRE el resultado de PRO.

    El thread shadow nunca bloquea — best-effort, exception → log debug
    + skip. Cero impacto sobre la respuesta del endpoint.
    """
    pro_t0 = _time_module.monotonic()
    pro_llm = ChatGoogleGenerativeAI(
        model=pro_model,
        temperature=pro_temperature,
        google_api_key=os.environ.get("GEMINI_API_KEY"),
    ).with_structured_output(output_schema)
    pro_result = pro_llm.invoke(prompt)
    pro_duration_ms = int((_time_module.monotonic() - pro_t0) * 1000)

    if not _should_run_shadow(user_id):
        return pro_result

    user_id_hash = hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()[:12] if user_id else None

    def _shadow_worker(pro_res=pro_result, pro_ms=pro_duration_ms):
        try:
            flash_t0 = _time_module.monotonic()
            flash_llm = ChatGoogleGenerativeAI(
                model=_FACT_SHADOW_MODEL,
                temperature=pro_temperature,
                google_api_key=os.environ.get("GEMINI_API_KEY"),
            ).with_structured_output(output_schema)
            flash_res = flash_llm.invoke(prompt)
            flash_ms = int((_time_module.monotonic() - flash_t0) * 1000)
            diff = differ(pro_res, flash_res)
            payload = {
                "callsite": callsite_tag,
                "pro_model": pro_model,
                "flash_model": _FACT_SHADOW_MODEL,
                "duration_ms_pro": pro_ms,
                "duration_ms_flash": flash_ms,
                "user_id_hash": user_id_hash,
                **diff,
            }
            _persist_shadow_diff(user_id, payload)
        except Exception as e:
            logger.warning(f"[FACT-SHADOW] worker fallo (no fatal): {e}")

    t = threading.Thread(target=_shadow_worker, daemon=True, name="fact-shadow")
    t.start()
    return pro_result
# ============================================================

# Categorías canónicas alineadas con extract_facts() prompt y graph_orchestrator.py CATEGORY_PRIORITY_WEIGHTS
FactCategoryLiteral = Literal[
    "alergia", "condicion_medica", "dieta", "rechazo",
    "preferencia", "objetivo", "sintoma_temporal"
]


class FactMetadata(BaseModel):
    category: FactCategoryLiteral = Field(description="Categoría del hecho: 'alergia', 'condicion_medica', 'dieta', 'rechazo', 'preferencia', 'objetivo' o 'sintoma_temporal'.")
    ingrediente_canonico: Optional[str] = Field(description="ID de catálogo universal del ingrediente en minúsculas. OBLIGATORIO usar los IDs definidos en el Catálogo Canónico (ej. 'peanut' para maní/cacahuete). Si no está en el catálogo, usa su traducción al inglés en minúscula y singular.", default=None)
    ingrediente: str = Field(description="Ingrediente principal si aplica, ej: 'mani', 'camarones'. Vacío si no aplica.")
    intensidad: int = Field(description="Intensidad del sentimiento (1 a 5). 1=Rechazo/Odio, 2=No le gusta, 3=Neutral/Info, 4=Le gusta, 5=Le fascina", default=3)

class FactItem(BaseModel):
    fact: str = Field(description="El hecho en sí expresado de forma clara.")
    metadata: FactMetadata = Field(description="Metadatos estructurados para clasificación exacta.")

class FactsModel(BaseModel):
    facts: List[FactItem] = Field(description="Lista de hechos nutricionales puntuales extraídos del mensaje.")

# --- Modelos para Batching de Contradicciones y Fusiones ---

class BatchContradictionItem(BaseModel):
    new_fact: str = Field(description="El texto exacto del nuevo hecho que genera la contradicción.")
    ids_to_delete: List[str] = Field(description="IDs (UUID) de los hechos existentes que este nuevo hecho contradice y deben ser borrados.")

class BatchMergeItem(BaseModel):
    merged_fact: str = Field(description="El hecho fusionado resultante (más completo que los individuales).")
    ids_to_delete: List[str] = Field(description="IDs (UUID) de los hechos existentes redundantes que serán reemplazados por el fusionado.")
    skip_new_fact: str = Field(description="El texto del nuevo hecho que fue absorbido en la fusión (para no guardarlo por separado).")

class BatchContradictionResult(BaseModel):
    contradictions: List[BatchContradictionItem] = Field(
        description="Lista de contradicciones encontradas. Vacía si no hay ninguna."
    )
    merges: List[BatchMergeItem] = Field(
        description="Lista de fusiones de hechos redundantes/complementarios. Vacía si no hay ninguna.",
        default=[]
    )

class RouterResult(BaseModel):
    has_relevant_info: bool = Field(description="True si el mensaje contiene datos médicos, preferencias alimenticias, alergias, síntomas u objetivos. False si es conversación casual (ej: hola, gracias, ok).")
    confidence_score: int = Field(description="Nivel de confianza de tu decisión, de 1 a 10. 10=Completamente seguro. 1=Muy inseguro/confuso (ej. sarcasmo, texto ambiguo).", default=10)

def should_extract_facts(user_message: str) -> bool:
    """Verifica rápidamente si el mensaje vale la pena analizarse para extraer hechos."""
    if not user_message or len(user_message.strip()) < 5:
        return False
        
    prompt = f"""
    Analiza este mensaje y responde True SOLO si el usuario menciona algo sobre:
    - Preferencias alimenticias (gustos, rechazos)
    - Hábitos de consumo o registro de comidas recientes (qué comió o qué bebió)
    - Alergias o condiciones médicas
    - Síntomas temporales o estado de salud
    - Objetivos de salud, condición física o peso
    
    Responde False si es conversación general, saludos, agradecimientos o texto sin datos útiles sobre el perfil del usuario.
    
    ES VITAL que también asignes un 'confidence_score' del 1 al 10. Si el mensaje es ambiguo, sarcástico, o no estás 100% seguro de si contiene un hecho médico velado, asigna un score bajo (ej. 1 a 6).
    
    Mensaje: "{user_message}"
    """
    
    llm = ChatGoogleGenerativeAI(
        model=_fact_extractor_router_model_name(),
        temperature=0.0,
        google_api_key=os.environ.get("GEMINI_API_KEY")
    ).with_structured_output(RouterResult)
    
    try:
        res = llm.invoke(prompt)
        if not res:
            return False
            
        logger.info(f"🚦 [ROUTER LITE] has_info: {res.has_relevant_info} | confidence: {res.confidence_score}/10")
        
        # Fallback de confianza: Si tiene info, lo pasamos. 
        # Si dice que NO tiene info, pero está inseguro (< 8), forzamos pasarlo al extractor pesado por seguridad.
        if res.has_relevant_info:
            return True
        elif res.confidence_score < 8:
            logger.warning(f"⚠️ [ROUTER FALLBACK] Score bajo ({res.confidence_score}/10). Enviando a extractor pesado por si acaso.")
            return True
        else:
            return False
            
    except Exception as e:
        logger.warning(f"⚠️ [ROUTER FALLBACK] Error analizando: {e}")
        return True # Fallback seguro: ante la duda, extraemos


def _build_ingredient_catalog() -> str:
    """Genera el catálogo canónico de ingredientes dinámicamente desde constants.py (SSoT)."""
    from constants import PROTEIN_SYNONYMS, CARB_SYNONYMS, VEGGIE_FAT_SYNONYMS, FRUIT_SYNONYMS
    
    lines = ["Catálogo Canónico de Ingredientes y Alérgenos (USA ESTOS IDs SIEMPRE QUE APLIQUE):"]
    
    # Alérgenos comunes (hardcoded porque son clínicos y no tienen sinónimos en constants)
    allergens = {
        "peanut": "maní, cacahuete",
        "dairy": "leche, queso, lactosa, yogur, mantequilla",
        "egg": "huevo",
        "shellfish": "mariscos, camarones, langosta, cangrejo, lambí",
        "fish": "pescado, salmón, atún, bacalao, arenque",
        "soy": "soja, soya",
        "wheat": "trigo, gluten, pan, pasta, harina",
        "tree_nut": "nueces, almendras, cajuil, macadamia",
    }
    for canon_id, synonyms_str in allergens.items():
        lines.append(f"- {canon_id} ({synonyms_str})")
    
    # Generar desde los synonym maps del sistema
    all_maps = {
        "PROTEÍNAS": PROTEIN_SYNONYMS,
        "CARBOHIDRATOS": CARB_SYNONYMS,
        "VEGETALES/GRASAS": VEGGIE_FAT_SYNONYMS,
        "FRUTAS": FRUIT_SYNONYMS,
    }
    for _group_name, syn_dict in all_maps.items():
        for base_name, variants in syn_dict.items():
            # Usar el nombre base como ID canónico y las primeras 4 variantes como ejemplos
            sample = ", ".join(variants[:4])
            lines.append(f"- {base_name} ({sample})")
    
    return "\n".join(lines)

DOMINICAN_INGREDIENT_CATALOG = _build_ingredient_catalog()

def extract_facts(user_message: str, recent_history: str = "", user_id: Optional[str] = None):
    """
    Analiza el mensaje del usuario y extrae "hechos" (facts) permanentes
    junto con sus metadatos estructurados (JSON) sobre sus preferencias,
    salud, alergias u objetivos.

    [P3-FACT-SHADOW-AB · 2026-05-14] `user_id` (opcional) habilita el
    shadow A/B PRO→FLASH determinístico. Sin user_id el shadow se salta
    (no podemos samplear estable); con user_id, el bucket se decide vía
    `_should_run_shadow`. Cero impacto UX — siempre retorna el output de PRO.
    """
    if not user_message or len(user_message.strip()) < 5:
        return []

    logger.info("\n-------------------------------------------------------------")
    logger.info("🔍 [EXTRACTOR DE HECHOS] Analizando mensaje para vectorizar y etiquetar...")
    
    history_context = f"\n    Contexto reciente (últimos mensajes):\n    {recent_history}\n" if recent_history else ""

    prompt = f"""
    Eres un Analista Nutricional que extrae "Hechos" (Facts) de los mensajes de los pacientes y los clasifica.
    Tu objetivo es leer un mensaje y determinar si contiene información útil para el perfil del usuario:
    - Preferencias alimenticias (gustos, rechazos fuertes) -> category: 'preferencia' o 'rechazo'
    - Hábitos de consumo o registro de comidas recientes -> category: 'dieta' o 'preferencia'
    - Alergias o condiciones crónicas -> category: 'alergia' o 'condicion_medica'
    - Síntomas o estados pasajeros (ej. "estómago revuelto esta semana", "estoy resfriado") -> category: 'sintoma_temporal'
    - Objetivos o rutinas -> category: 'objetivo'
    
    Además, clasifica la 'intensidad' (del 1 al 5) del sentimiento si aplica:
    - 1: Odio absoluto, rechazo frontal ("no soporto el brócoli")
    - 2: Desagrado leve ("no me gusta mucho el pescado")
    - 3: Neutral o dato médico objetivo ("tengo diabetes", "peso 80kg", "ayer comí arroz")
    - 4: Le gusta bastante ("qué buena la pasta")
    - 5: Pasión o adicción ("amo el plátano", "no puedo vivir sin café")

    IMPORTANTE - CANONICALIZACIÓN DE INGREDIENTES:
    Para el campo `ingrediente_canonico`, DEBES usar OBLIGATORIAMENTE uno de los siguientes IDs si el ingrediente coincide, especialmente para alergias y condiciones médicas:
    {DOMINICAN_INGREDIENT_CATALOG}
    Si el ingrediente no está en la lista, usa su nombre en inglés en minúsculas y singular (ej. "broccoli", "apple").

    {history_context}
    Mensaje del usuario: "{user_message}"
    
    Usa el contexto reciente SOLO para entender a qué se refiere el mensaje actual (ej. si dice "y también me dio alergia", el contexto te dirá a qué alimento). Pero EXTRAE hechos principalmente basados en el último mensaje.
    Si el mensaje NO contiene hechos importantes (ej. "hola", "gracias", "ok"), devuelve una lista vacía [].
    Si contiene hechos, extráelos como oraciones cortas, precisas (en tercera persona sobre el usuario).
    Asegúrate de incluir los metadatos de categoría, intensidad, e 'ingrediente' si aplica (y el 'ingrediente_canonico' rigurosamente guiado por el catálogo).
    """

    try:
        # [P3-FACT-SHADOW-AB · 2026-05-14] Pasamos por el helper que corre
        # PRO sync (truth) y opcionalmente FLASH en daemon thread si el
        # knob está activo. Helper devuelve siempre el resultado de PRO.
        response = _invoke_with_shadow(
            prompt=prompt,
            output_schema=FactsModel,
            pro_model=_fact_extractor_primary_model_name(),
            pro_temperature=0.1,
            callsite_tag="extract_facts",
            user_id=user_id,
            differ=_diff_facts_model,
        )
        facts = response.facts if response and hasattr(response, 'facts') else []
        if facts:
            logger.info(f"✅ Se encontraron {len(facts)} hechos estructurados.")
        else:
            logger.info("➡️ No se encontraron hechos relevantes.")
        return facts
    except Exception as e:
        logger.warning(f"⚠️ Error al extraer hechos: {e}")
        return []

CACHE_TTL_PERMANENT = 3153600000  # ~100 years — embeddings are deterministic for the same input

@centralized_cache(ttl_seconds=CACHE_TTL_PERMANENT, maxsize=10000)
def get_embedding(text: str) -> list:
    """Genera un vector embedding usando Gemini embedding (Caché Distribuido).

    Modelo TEXT-ONLY configurable vía `constants.GEMINI_EMBEDDING_TEXT_MODEL`
    (knob `MEALFIT_GEMINI_EMBEDDING_TEXT_MODEL`). Default `gemini-embedding-001`
    estable. Caso de uso: knowledge graph de facts del usuario — todo texto.

    [P3-3 · 2026-05-10] Política de fallo: fail-fast con `return []`.
    Decisión deliberada (no se implementa fallback a modelo alternativo):
      1. **Modelo único**: el knob `GEMINI_EMBEDDING_TEXT_MODEL` define UN
         modelo. Cambiar a fallback automático requeriría un segundo knob
         + versionado del cache (las dimensiones del vector pueden diferir
         entre modelos y romper similaridad search).
      2. **Cache persistente**: `CACHE_TTL_PERMANENT = ~100 años` —
         re-embedeos del mismo texto son raros. Una excepción transitoria
         se ve solo una vez por texto único; el cache absorbe el rebound.
      3. **Downstream tolera []**: los consumidores (`async_extract_and_save_facts`,
         hybrid search en `agent.py`, `proactive_agent.py`) chequean
         `if not emb: skip/degrade` y siguen el flujo sin propagar.

    El error queda visible vía `logger.error` (P3-3 — antes era `print(...)`
    invisible a Sentry / log aggregation). Si la frecuencia de error sube en
    `pipeline_metrics` o en Sentry, abrir P-fix dedicado con la decisión
    fallback explícita (ej. degradar a modelo legacy con dim conversion).
    """
    _model_name = "<unknown>"
    try:
        from constants import GEMINI_EMBEDDING_TEXT_MODEL
        _model_name = GEMINI_EMBEDDING_TEXT_MODEL
        embeddings = GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_TEXT_MODEL,
            google_api_key=os.environ.get("GEMINI_API_KEY")
        )
        emb = embeddings.embed_query(text)
        logger.debug(f"[EMBEDDING CACHE] MISS → Generado embedding para: '{text[:50]}...'")
        return list(emb[:768])
    except Exception as e:
        # [P3-3 · 2026-05-10] logger.error (no print) — feed Sentry/alerting.
        # Si esto se ve con frecuencia, el knob fail-fast deja un trail
        # diagnosticable sin sentry_sdk.capture explícito (sentry captura
        # error-level logs por default). `_model_name` se bindea ANTES del
        # import, así que aunque el import de constants falle, el log es
        # informativo (model=<unknown>).
        logger.error(
            f"[EMBEDDING] Falló get_embedding (modelo={_model_name!r}, "
            f"text_len={len(text)}): {type(e).__name__}: {e}"
        )
        return []

CRITICAL_CATEGORIES = {"condicion_medica", "alergia", "dieta", "objetivo"}


def _run_fact_pipeline(user_id: str, fact_items: list, log_prefix: str = ""):
    """
    Pipeline compartido de Fases 1-3: Preparar hechos, verificar contradicciones/fusiones en batch,
    borrar obsoletos y guardar nuevos. Usado por async_extract_and_save_facts y _process_single_extraction.
    """
    # FASE 1: Generar embeddings y buscar similares para TODOS los hechos
    prepared_facts = []

    for item in fact_items:
        if isinstance(item, dict):
            fact_text = item.get("fact", "")
            metadata = item.get("metadata", {})
        else:
            fact_text = getattr(item, "fact", "")
            metadata = item.metadata.model_dump() if hasattr(item, 'metadata') else {}
        
        if not fact_text:
            continue
        
        emb = get_embedding(fact_text)
        if not emb:
            continue
        
        category = metadata.get("category", "")
        filter_meta = {"category": category} if category in CRITICAL_CATEGORIES else None
        
        similar_facts = search_user_facts_hybrid(user_id, emb, filter_metadata=filter_meta, threshold=0.6, limit=5)
        
        if filter_meta and similar_facts:
            logger.info(f"{log_prefix}🔎 [HIBRID SEARCH] Búsqueda optimizada por categoría crítica '{category}'. Recuperados: {len(similar_facts)}")
        
        prepared_facts.append({
            "item": item,
            "fact_text": fact_text,
            "metadata": metadata,
            "emb": emb,
            "similar_facts": similar_facts
        })

    if not prepared_facts:
        return

    # FASE 2: Verificar contradicciones en BATCH (una sola llamada LLM)
    facts_with_similar = [pf for pf in prepared_facts if pf["similar_facts"]]
    
    ids_to_delete_all = set()
    merged_facts_to_save = []
    skipped_new_facts = set()

    if facts_with_similar:
        logger.info(f"{log_prefix}🔄 [BATCH] Verificando contradicciones para {len(facts_with_similar)} hechos...")
        
        sections = []
        for idx, pf in enumerate(facts_with_similar, 1):
            existing_str = "\n".join(
                [f"    ID: {f['id']} - Hecho: {f['fact']}" for f in pf["similar_facts"] if 'id' in f and 'fact' in f]
            )
            sections.append(
                f"  NUEVO HECHO #{idx}: \"{pf['fact_text']}\"\n"
                f"  Hechos existentes relacionados:\n{existing_str}"
            )
        
        all_sections = "\n\n".join(sections)
        
        batch_prompt = f"""
        Analiza TODOS los nuevos hechos y compáralos con sus hechos existentes correspondientes.
        Para cada nuevo hecho, determina:
    
        A) CONTRADICCIÓN: Si el nuevo hecho INVALIDA directamente uno viejo.
           Ejemplo: "no come pescado" vs "ahora le gusta el pescado" → CONTRADICCIÓN.
    
        B) REDUNDANCIA/FUSIÓN: Si el nuevo hecho es COMPLEMENTARIO o REDUNDANTE con uno existente.
           Ejemplo: "Le gusta el pollo" + "Amo el pollo asado" → FUSIÓN: "Al usuario le encanta el pollo, especialmente asado".
           Ejemplo: "Es alérgico al maní" + "Tiene alergia a los cacahuetes" → FUSIÓN: "El usuario es alérgico al maní/cacahuetes".
    
        Reglas:
        - Si un hecho nuevo contradice uno viejo, ponlo en "contradictions" con los IDs a borrar.
        - Si un hecho nuevo es redundante/complementario, ponlo en "merges":
          crea un "merged_fact" combinado en tercera persona.
          Incluye "ids_to_delete" de los viejos y "skip_new_fact" con el texto del nuevo.
        - Si un hecho nuevo es INDEPENDIENTE, NO lo incluyas en ninguna lista.
        - NO fusiones hechos de temas distintos.

        HECHOS A ANALIZAR:
        {all_sections}
        """
        
        try:
            # [P3-FACT-SHADOW-AB · 2026-05-14] Helper que corre PRO sync
            # y opcionalmente shadow FLASH en daemon thread. El user_id
            # del scope superior se usa para el sampling determinístico.
            response = _invoke_with_shadow(
                prompt=batch_prompt,
                output_schema=BatchContradictionResult,
                pro_model=_fact_extractor_primary_model_name(),
                pro_temperature=0.0,
                callsite_tag="contradiction_merge",
                user_id=user_id,
                differ=_diff_contradiction_result,
            )
        
            if response and response.contradictions:
                for contradiction in response.contradictions:
                    if contradiction.ids_to_delete:
                        logger.warning(f"{log_prefix}⚠️ [CONTRADICCIÓN] \"{contradiction.new_fact}\" → Borrar IDs: {contradiction.ids_to_delete}")
                        ids_to_delete_all.update(contradiction.ids_to_delete)
        
            if response and response.merges:
                for merge in response.merges:
                    if merge.ids_to_delete and merge.merged_fact:
                        logger.info(f"{log_prefix}🔀 [FUSIÓN] \"{merge.merged_fact}\" ← Absorbe IDs: {merge.ids_to_delete}")
                        ids_to_delete_all.update(merge.ids_to_delete)
                        skipped_new_facts.add(merge.skip_new_fact)
                    
                        original_metadata = {}
                        for pf in facts_with_similar:
                            if pf["fact_text"] == merge.skip_new_fact:
                                original_metadata = pf["metadata"]
                                break
                    
                        merged_facts_to_save.append({
                            "fact_text": merge.merged_fact,
                            "metadata": original_metadata
                        })
                    
        except Exception as e:
            logger.warning(f"{log_prefix}⚠️ [Error en validación batch de contradicciones/fusiones]: {e}")

    # FASE 3: Borrar contradictorios/redundantes y guardar hechos
    if ids_to_delete_all:
        logger.info(f"{log_prefix}🗑️ [BATCH] Borrando {len(ids_to_delete_all)} hechos (contradictorios + redundantes)...")
        for f_id in ids_to_delete_all:
            delete_user_fact(f_id)

    saved_count = 0
    for pf in prepared_facts:
        if pf["fact_text"] in skipped_new_facts:
            logger.info(f"{log_prefix}⏭️ Hecho absorbido en fusión: '{pf['fact_text']}'")
            continue
        save_user_fact(user_id, pf["fact_text"], pf["emb"], metadata=pf["metadata"])
        logger.info(f"{log_prefix}📦 Nuevo hecho guardado: '{pf['fact_text']}' | Metadatos: {pf['metadata']}")
        saved_count += 1
    
    merge_count = 0
    for mf in merged_facts_to_save:
        merged_emb = get_embedding(mf["fact_text"])
        if merged_emb:
            save_user_fact(user_id, mf["fact_text"], merged_emb, metadata=mf["metadata"])
            logger.info(f"{log_prefix}🔀 Hecho fusionado guardado: '{mf['fact_text']}' | Metadatos: {mf['metadata']}")
            merge_count += 1
    
    total_deleted = len(ids_to_delete_all)
    logger.info(f"{log_prefix}✅ [BATCH COMPLETO] {saved_count} nuevos + {merge_count} fusionados, {total_deleted} eliminados.")


def async_extract_and_save_facts(user_id: str, message: str, recent_history: str = ""):
    """
    Función orquestadora para ser ejecutada en background.
    Extrae hechos estructurados de un mensaje, revisa contradicciones con la DB
    usando BATCHING (una sola llamada a Gemini para todos los hechos),
    borra los obsoletos y guarda los nuevos vectorizados y etiquetados.
    """
    try:
        if not should_extract_facts(message):
            logger.info("⏭️ [ROUTER] Mensaje ignorado. No contiene hechos relevantes para el perfil.")
            return

        import time
        max_retries = 5
        retry_delay = 2
        lock_acquired = False
        
        for attempt in range(max_retries):
            if acquire_fact_lock(user_id):
                lock_acquired = True
                break
            logger.warning(f"⚠️ [FACT EXTRACTOR] Extracción en progreso para el usuario {user_id}. Esperando {retry_delay}s ({attempt+1}/{max_retries})...")
            time.sleep(retry_delay)
            
        if not lock_acquired:
            # ====== COLA PERSISTENTE: Nunca perder datos clínicos ======
            enqueue_pending_fact(user_id, message, recent_history)
            logger.info(f"📋 [FACT EXTRACTOR] Mensaje encolado en Supabase para procesamiento posterior.")
            return
            
        fact_items = extract_facts(message, recent_history, user_id=user_id)
        if not fact_items:
            release_fact_lock(user_id)
            return

        _run_fact_pipeline(user_id, fact_items, log_prefix="")

    except Exception as e:
        import traceback
        logger.error(f"❌ [CRÍTICO] Fallo general en orquestación de hechos: {e}")
        track = traceback.format_exc()
        logger.info(f"Trazabilidad extendida de error: {track}")
    finally:
        # Liberar el lock de BD para que la respuesta de FastAPI/UI no se bloquee ni devuelva timeout
        release_fact_lock(user_id)
        
        # PROCESAR COLA PERSISTENTE (SUPABASE WEBHOOKS)
        # El hilo asíncrono daemonizado (Fire-and-Forget local) fue removido debido a inestabilidad 
        # en entornos serverless/PaaS donde el proceso muere tras devolver la respuesta HTTP.
        # Ahora, un TRIGGER AFTER INSERT en Supabase llamará de forma robusta a nuestro endpoint especial.
        logger.info(f"✅ Extracción en línea terminada. Webhook externo procesará la cola si quedaron pendientes.")


def process_pending_queue_sync(user_id: str):
    """Worker síncrono para drenar la cola de pendientes. Llamado por el Webhook de Supabase de manera robusta."""
    
    if not acquire_fact_lock(user_id):
        logger.warning(f"⚠️ [WEBHOOK QUEUE] Lock ocupado para {user_id}. Se procesará luego.")
        return
        
    try:
        pending_items = dequeue_pending_facts(user_id)
        if not pending_items:
            logger.info("➡️ [WEBHOOK QUEUE] No hay hechos pendientes en cola.")
            return
            
        logger.info(f"\n📋 [FACT EXTRACTOR WEBHOOK] Iniciando drenaje estructurado para {len(pending_items)} mensajes pendientes...")
        processed_ids = []
        for idx, pending in enumerate(pending_items, 1):
            try:
                logger.info(f"   📋 [{idx}/{len(pending_items)}] Procesando: '{pending['message'][:50]}...'")
                _process_single_extraction(user_id, pending["message"], pending.get("recent_history", ""))
                processed_ids.append(pending["id"])
            except Exception as pe:
                logger.warning(f"   ⚠️ Error procesando pendiente #{idx}: {pe}")
                processed_ids.append(pending["id"]) # Evitar loop infinito fallido
        
        if processed_ids:
            delete_pending_facts(processed_ids)
        logger.info(f"✅ [FACT EXTRACTOR] Cola pendiente finalizada en Hilo Secundario.")
    except Exception as qe:
        logger.warning(f"⚠️ Error general en hilo secundario de cola: {qe}")
    finally:
        release_fact_lock(user_id)


def _process_single_extraction(user_id: str, message: str, recent_history: str = ""):
    """
    Procesa una sola extracción de hechos SIN manejar el lock.
    Usado internamente para drenar la cola de pendientes.
    Delega al pipeline compartido _run_fact_pipeline.
    """
    if not should_extract_facts(message):
        return
    
    fact_items = extract_facts(message, recent_history, user_id=user_id)
    if not fact_items:
        return

    _run_fact_pipeline(user_id, fact_items, log_prefix="   ")
