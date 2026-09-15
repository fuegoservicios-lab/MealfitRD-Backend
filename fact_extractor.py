import os
import json
import logging
import hashlib
import threading
import time as _time_module
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from cache_manager import centralized_cache
# [P0-LLM-PROVIDER-MIGRATION · 2026-06-12] Gemini → GLM (router por tier en
# llm_provider). Embeddings via capa pluggable embeddings_provider.
from llm_provider import ChatGLM, GLM_FLASH
from embeddings_provider import get_text_embedding
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

from knobs import _env_str, _env_float, _env_int

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
#       Setear a un model ID alternativo (e.g. 'glm-5.3' para medir
#       si PRO extrae mejor que el default flash) para activar el A/B.
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
# hasta redeploy (45min de cold start del VPS Oracle). Tooltip-anchor:
# P2-NEW-FACTEX-PRIMARY-MODEL-KNOB.
# [P0-LLM-PROVIDER-MIGRATION · 2026-06-12] Defaults GLM-5.3 Flash: la
# extracción de hechos es tarea de background (structured extraction a
# schema) — corre en el modelo barato para TODOS los tiers. Override sin
# redeploy via knob (e.g. `glm-5.3` si la calidad de extracción
# clínica degrada visiblemente).
_FACT_EXTRACTOR_PRIMARY_MODEL = _env_str(
    "MEALFIT_FACT_EXTRACTOR_PRIMARY_MODEL",
    GLM_FLASH,
)
_FACT_EXTRACTOR_ROUTER_MODEL = _env_str(
    "MEALFIT_FACT_EXTRACTOR_ROUTER_MODEL",
    GLM_FLASH,
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


# [P2-LLM-TIMEOUT-SWEEP · 2026-05-30] Timeouts per-invoke del fact-extractor.
# Pre-fix: los 3 constructores `ChatGoogleGenerativeAI` (pro sync, shadow en
# daemon thread, router flash-lite) se creaban SIN `timeout=`. El PRO invoke
# corre síncrono en el thread del threadpool que sirve `async_extract_and_save_facts`;
# un Gemini colgado bloqueaba ese thread indefinidamente Y mantenía tomado el
# fact-lock (liberado solo en el `finally`), bloqueando futuras extracciones del
# mismo usuario. El `timeout=` propaga al deadline gRPC → DeadlineExceeded,
# capturado por los `except` existentes (extract_facts→[], should_extract_facts→True).
# Floor 10s: Gemini API rechaza deadlines <10s con HTTP 400 INVALID_ARGUMENT
# (lección P1-CHAT-EMPTY-RESPONSE). Knobs auto-registrados.
# Tooltip-anchor: P2-LLM-TIMEOUT-SWEEP.
def _fact_extractor_llm_timeout_s() -> float:
    return _env_float(
        "MEALFIT_FACT_EXTRACTOR_LLM_TIMEOUT_S",
        30.0,
        validator=lambda v: 10.0 <= v <= 120.0,
    )


# [P0-LLM-PROVIDER-MIGRATION · 2026-06-12] El timeout de embeddings
# (`MEALFIT_EMBEDDING_LLM_TIMEOUT_S`, lección P2-LLM-TIMEOUT-SWEEP) ahora se
# aplica DENTRO de `embeddings_provider._embeddings_timeout_s` — un solo
# punto para todos los surfaces de embeddings.


def _fact_extractor_router_llm_timeout_s() -> float:
    return _env_float(
        "MEALFIT_FACT_EXTRACTOR_ROUTER_LLM_TIMEOUT_S",
        12.0,
        validator=lambda v: 10.0 <= v <= 120.0,
    )


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


# [P1-CHAT-FACTS-AUDIT · 2026-09-14] Observabilidad del gasto LLM del extractor.
# Antes ni el router (`should_extract_facts`) ni el extractor ni el batch de
# contradicciones escribían en `llm_usage_events`: el coste del coach en segundo
# plano era invisible. Best-effort: un fallo de telemetría jamás tumba la extracción.
# Import diferido: los tests que stubbean `db` sin esta función importan este módulo.
def _log_fact_llm_usage(raw: Any, *, model: str, node: str, user_id: Optional[str]) -> None:
    try:
        usage = getattr(raw, "usage_metadata", None) or {}
        if not isinstance(usage, dict):
            usage = {}
        details = usage.get("input_token_details") or {}
        cached = details.get("cache_read") if isinstance(details, dict) else None
        from db import log_llm_usage_event
        log_llm_usage_event(
            user_id=user_id or None,
            model=model,
            node=node,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cached_tokens=cached,
        )
    except Exception as e:
        logger.debug(f"[P1-CHAT-FACTS-AUDIT] log de uso LLM falló (no fatal): {e}")


def _unwrap_structured(result: Any, *, model: str, node: str, user_id: Optional[str]) -> Any:
    """[P1-CHAT-FACTS-AUDIT · 2026-09-14] Desenvuelve la salida de
    `with_structured_output(..., include_raw=True)` (`{raw, parsed, parsing_error}`),
    registra los tokens del `raw` y devuelve `parsed`.

    Un error de parseo se RELANZA: mismo contrato que sin `include_raw` (el caller lo
    trata como fallo, no como «no había nada»). Un objeto ya parseado (mocks) pasa tal cual.
    """
    if isinstance(result, dict) and "parsed" in result and "raw" in result:
        _log_fact_llm_usage(result.get("raw"), model=model, node=node, user_id=user_id)
        err = result.get("parsing_error")
        if err is not None and result.get("parsed") is None:
            raise err if isinstance(err, BaseException) else ValueError(str(err))
        return result.get("parsed")
    return result


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
    # [P1-CHAT-FACTS-AUDIT · 2026-09-14] `include_raw=True` para leer los tokens del
    # AIMessage y registrarlos en `llm_usage_events`; `_unwrap_structured` devuelve el
    # objeto parseado igual que antes (el contrato de los callers no cambia).
    pro_llm = ChatGLM(
        model=pro_model,
        temperature=pro_temperature,
        timeout=_fact_extractor_llm_timeout_s(),  # [P2-LLM-TIMEOUT-SWEEP · 2026-05-30]
    ).with_structured_output(output_schema, include_raw=True)
    pro_result = _unwrap_structured(
        pro_llm.invoke(prompt),
        model=pro_model,
        node=f"fact_extractor_{callsite_tag}",
        user_id=user_id,
    )
    pro_duration_ms = int((_time_module.monotonic() - pro_t0) * 1000)

    if not _should_run_shadow(user_id):
        return pro_result

    user_id_hash = hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()[:12] if user_id else None

    def _shadow_worker(pro_res=pro_result, pro_ms=pro_duration_ms):
        try:
            flash_t0 = _time_module.monotonic()
            flash_llm = ChatGLM(
                model=_FACT_SHADOW_MODEL,
                temperature=pro_temperature,
                timeout=_fact_extractor_llm_timeout_s(),  # [P2-LLM-TIMEOUT-SWEEP · 2026-05-30]
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

def should_extract_facts(user_message: str, user_id: Optional[str] = None) -> bool:
    """Verifica rápidamente si el mensaje vale la pena analizarse para extraer hechos.

    [P1-CHAT-FACTS-AUDIT · 2026-09-14] `user_id` solo atribuye el gasto en
    `llm_usage_events` (node `fact_extractor_router`)."""
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
    
    llm = ChatGLM(
        model=_fact_extractor_router_model_name(),
        temperature=0.0,
        timeout=_fact_extractor_router_llm_timeout_s(),  # [P2-LLM-TIMEOUT-SWEEP · 2026-05-30]
    ).with_structured_output(RouterResult, include_raw=True)  # [P1-CHAT-FACTS-AUDIT] tokens

    try:
        res = _unwrap_structured(
            llm.invoke(prompt),
            model=_fact_extractor_router_model_name(),
            node="fact_extractor_router",
            user_id=user_id,
        )
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

    [P1-CHAT-FACTS-AUDIT · 2026-09-14] Devuelve `None` si la llamada al LLM FALLA
    y `[]` si no había hechos. Antes ambos casos eran `[]`: una alergia dicha
    durante una caída del proveedor se perdía sin rastro y la cola marcaba el
    mensaje como procesado. Los llamadores reintentan ante `None`.
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
        # [P1-CHAT-FACTS-AUDIT · 2026-09-14] `None` = FALLÓ (≠ `[]` = no había nada).
        logger.warning(f"⚠️ [P1-CHAT-FACTS-AUDIT] Error al extraer hechos (se reintentará): {e}")
        return None

CACHE_TTL_PERMANENT = 3153600000  # ~100 years — embeddings are deterministic for the same input

@centralized_cache(ttl_seconds=CACHE_TTL_PERMANENT, maxsize=10000, cache_empty=False)
def _cached_text_embedding(text: str, model_id: str, purpose: str) -> list:
    """[P1-COHERE-EMBED-V4 · 2026-06-12] Capa cacheada REAL del embedding.

    La cache key del decorador se construye con (func, args) — por eso
    `model_id` y `purpose` son ARGUMENTOS: versionan el caché Redis (TTL
    ~100 años) por espacio vectorial (`embed-v4.0@1536`) y por lado
    (`query`/`document`, Embed v4 es asimétrico). Sin esto, un switch de
    provider serviría vectores del espacio ANTERIOR desde Redis — mezcla
    silenciosa que rompe toda similarity (bug detectado en la migración
    Gemini→Cohere).

    El error queda visible vía `logger.error` (P3-3): el del provider lo
    emite `embeddings_provider.get_text_embedding` (un solo punto para
    todos los surfaces); este wrapper conserva un except defensivo con
    contexto `text_len` por si el provider mismo es inimportable.
    """
    try:
        emb = get_text_embedding(text, purpose=purpose)
        if not emb:
            return []
        logger.debug(
            f"[EMBEDDING CACHE] MISS → Generado embedding ({model_id}/{purpose}) "
            f"para: '{text[:50]}...'"
        )
        return list(emb)
    except Exception as e:
        # [P3-3 · 2026-05-10] logger.error (no print) — feed Sentry/alerting.
        logger.error(
            f"[EMBEDDING] Falló get_embedding (provider import/runtime, "
            f"text_len={len(text)}): {type(e).__name__}: {e}"
        )
        return []


def get_embedding(text: str, purpose: str = "query") -> list:
    """Genera un vector embedding via `embeddings_provider` (Caché Distribuido).

    [P1-COHERE-EMBED-V4 · 2026-06-12] Provider de producción: Cohere Embed
    v4 (`embed-v4.0`, dim 1536 — igual que las columnas pgvector tras la
    migración `p1_cohere_embed_v4_vector_dims`). `purpose="document"` SOLO
    para textos que se PERSISTEN para retrieval asimétrico (user_facts);
    `"query"` (default) para todos los lados de búsqueda y comparaciones
    simétricas. La asimetría input_type es la palanca de precisión del RAG.

    [P3-3 · 2026-05-10] Política de fallo: fail-fast con `return []`.
    Decisión deliberada (no se implementa fallback a modelo alternativo):
      1. **Modelo único**: el knob del provider define UN modelo. Fallback
         automático requeriría doble versionado del cache y mezcla de
         espacios vectoriales en pgvector.
      2. **Cache persistente**: `CACHE_TTL_PERMANENT = ~100 años` para los
         embeddings EXITOSOS (deterministas para el mismo input + modelo +
         purpose). El fallo `[]` NO se cachea — `cache_empty=False`
         (P2-EMBED-NO-CACHE-EMPTY · 2026-05-30): un fallo transitorio se
         re-intenta en el siguiente llamado del mismo texto.
      3. **Downstream tolera []**: los consumidores (`async_extract_and_save_facts`,
         hybrid search en `agent.py`, `proactive_agent.py`) chequean
         `if not emb: skip/degrade` y siguen el flujo sin propagar.
    """
    from embeddings_provider import get_embeddings_model_id

    # model_id en la key versiona el caché por espacio vectorial; con
    # provider inactivo retorna "disabled" → el resultado [] NO se cachea
    # (cache_empty=False), así que esa key jamás acumula entradas.
    model_id = get_embeddings_model_id()
    if purpose not in ("query", "document"):
        purpose = "query"
    return _cached_text_embedding(text, model_id, purpose)

CRITICAL_CATEGORIES = {"condicion_medica", "alergia", "dieta", "objetivo"}


# ============================================================
# [P1-CHAT-FACTS-AUDIT · 2026-09-14] Protección de los hechos CLÍNICOS en el
# pipeline en línea. Antes, un hecho de `preferencia` podía «contradecir» una
# alergia y `delete_user_fact` la desactivaba, y una fusión heredaba la metadata
# del hecho NUEVO (una alergia fusionada perdía `category='alergia'`). Espejo de
# la exención de Dreaming (`dreaming._soft_delete_facts`): un hecho clínico
# existente NO lo retira ni lo absorbe un hecho no clínico. Retirar una alergia
# es una acción del usuario en Configuración (ver P0-CHAT-ALLERGY-MERGE en tools.py).
# ============================================================
def _clinical_categories() -> Optional[tuple]:
    """SSOT: `dreaming.CLINICAL_CATEGORIES` (no una lista nueva). Import diferido:
    dreaming arrastra `db`/`db_core` y hay tests que stubbean `db` e importan este
    módulo. Si el import falla devuelve `None` y los llamadores tratan TODO hecho
    existente como protegido (fail-secure)."""
    try:
        from dreaming import CLINICAL_CATEGORIES
        return tuple(CLINICAL_CATEGORIES)
    except Exception as e:
        logger.error(
            f"🚨 [P1-CHAT-FACTS-AUDIT] no se pudo importar dreaming.CLINICAL_CATEGORIES "
            f"({type(e).__name__}: {e}); se protegen TODOS los hechos existentes."
        )
        return None


def _is_clinical_new_fact(category: str, clinical: Optional[tuple]) -> bool:
    """¿El hecho NUEVO es clínico? Sin SSOT disponible, se asume que sí (fail-secure:
    se guarda aunque falte el embedding)."""
    return clinical is None or category in clinical


def _existing_fact_protected(existing_cat: str, new_cat: str, clinical: Optional[tuple]) -> bool:
    """Un hecho existente clínico no lo retira ni lo absorbe uno nuevo no clínico."""
    if clinical is None:
        return True
    return existing_cat in clinical and new_cat not in clinical


def _fact_metadata_dict(md: Any) -> dict:
    if isinstance(md, dict):
        return dict(md)
    if isinstance(md, str):
        try:
            _parsed = json.loads(md)
            return _parsed if isinstance(_parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _filter_deletable_ids(
    ids: Any,
    new_cat: str,
    existing_cat_by_id: dict,
    clinical: Optional[tuple],
    *,
    log_prefix: str = "",
    motivo: str = "",
) -> list:
    """Filtra los ids que el LLM propone retirar. Solo sobreviven los que (a) se le
    MOSTRARON (un id inventado no se toca) y (b) no son un hecho clínico protegido."""
    out: list = []
    for raw_id in ids or []:
        fid = str(raw_id)
        if fid in out:
            continue
        if fid not in existing_cat_by_id:
            logger.warning(
                f"{log_prefix}🛡 [P1-CHAT-FACTS-AUDIT] {motivo}: el id {fid} no estaba entre "
                f"los hechos mostrados al LLM — no se toca."
            )
            continue
        if _existing_fact_protected(existing_cat_by_id[fid], new_cat, clinical):
            logger.warning(
                f"{log_prefix}🛡 [P1-CHAT-FACTS-AUDIT] {motivo}: el hecho {fid} "
                f"(category={existing_cat_by_id[fid] or '?'}) está protegido frente a un hecho "
                f"nuevo de category={new_cat or '?'} — NO se retira; retirar una alergia o una "
                f"condición es una acción del usuario en Configuración."
            )
            continue
        out.append(fid)
    return out


def _embeddings_enabled() -> bool:
    """¿Hay proveedor de embeddings configurado? Sin él, reintentar no sirve de nada
    (degradación de diseño: el RAG cae a keyword/recency). Ante la duda → True
    (se trata como fallo transitorio y se reintenta, acotado por la cola)."""
    try:
        from embeddings_provider import is_embeddings_enabled
        return bool(is_embeddings_enabled())
    except Exception:
        return True


def _run_fact_pipeline(user_id: str, fact_items: list, log_prefix: str = "") -> bool:
    """
    Pipeline compartido de Fases 1-3: Preparar hechos, verificar contradicciones/fusiones en batch,
    guardar nuevos y retirar obsoletos. Usado por async_extract_and_save_facts y _process_single_extraction.

    [P1-CHAT-FACTS-AUDIT · 2026-09-14] Devuelve `True` si todo quedó persistido y
    `False` si algún hecho quedó sin guardar y conviene reintentar el mensaje.
    """
    clinical = _clinical_categories()
    complete = True

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
        
        # [P1-COHERE-EMBED-V4] purpose="document": este vector se PERSISTE en
        # user_facts.embedding para retrieval asimétrico (las queries del RAG
        # del chat/pipeline llegan con search_query). La similar-search de
        # abajo compara doc-vs-doc (simétrica dentro del espacio document).
        category = metadata.get("category", "")
        emb = get_embedding(fact_text, purpose="document")
        if not emb:
            # [P1-CHAT-FACTS-AUDIT · 2026-09-14] Antes: `continue` — el hecho se
            # descartaba en silencio si Cohere caía o faltaba COHERE_API_KEY, alergias
            # incluidas. Ahora:
            #   · clínico, o proveedor desactivado (reintentar no cambiaría nada) → se
            #     GUARDA sin embedding: invisible al RAG vectorial, pero visible para
            #     `get_user_facts_by_metadata` (el filtro de alergias) y para la UI.
            #     Sin búsqueda de similares: no hay vector con qué buscar.
            #   · no clínico con fallo transitorio → `complete=False` y el caller
            #     re-encola el mensaje; el reintento traerá su embedding.
            _degradado = not _embeddings_enabled()
            if _degradado or _is_clinical_new_fact(category, clinical):
                _saved = save_user_fact(user_id, fact_text, None, metadata=metadata)
                if _saved:
                    logger.warning(
                        f"{log_prefix}⚠️ [P1-CHAT-FACTS-AUDIT] embedding vacío: hecho guardado SIN "
                        f"embedding (category={category}, proveedor_activo={not _degradado}): '{fact_text}'"
                    )
                else:
                    logger.error(
                        f"{log_prefix}🚨 [P1-CHAT-FACTS-AUDIT] embedding vacío y save sin embedding "
                        f"falló (category={category}): '{fact_text}'"
                    )
                    if not _degradado:
                        complete = False
            else:
                logger.warning(
                    f"{log_prefix}⚠️ [P1-CHAT-FACTS-AUDIT] embedding vacío (fallo transitorio) para "
                    f"hecho no clínico (category={category}) — se reintentará el mensaje: '{fact_text}'"
                )
                complete = False
            continue

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
        return complete

    # FASE 2: Verificar contradicciones en BATCH (una sola llamada LLM)
    facts_with_similar = [pf for pf in prepared_facts if pf["similar_facts"]]

    contradiction_ids_by_text: dict = {}
    merged_facts_to_save = []
    skipped_new_facts = set()

    # [P1-CHAT-FACTS-AUDIT · 2026-09-14] Categoría de cada hecho EXISTENTE que se le
    # muestra al LLM: decide qué puede retirarse (ver `_filter_deletable_ids`).
    existing_cat_by_id: dict = {}
    existing_md_by_id: dict = {}
    for pf in facts_with_similar:
        for f in pf["similar_facts"]:
            if isinstance(f, dict) and f.get("id") is not None:
                _md = _fact_metadata_dict(f.get("metadata"))
                existing_md_by_id[str(f["id"])] = _md
                existing_cat_by_id[str(f["id"])] = str(_md.get("category") or "")
    new_cat_by_text = {
        pf["fact_text"]: str((pf["metadata"] or {}).get("category") or "") for pf in prepared_facts
    }

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
                    if not contradiction.ids_to_delete:
                        continue
                    # [P1-CHAT-FACTS-AUDIT] un hecho no clínico no retira uno clínico.
                    _new_cat = new_cat_by_text.get(contradiction.new_fact, "")
                    _ids_ok = _filter_deletable_ids(
                        contradiction.ids_to_delete, _new_cat, existing_cat_by_id, clinical,
                        log_prefix=log_prefix, motivo="contradicción",
                    )
                    if _ids_ok:
                        logger.warning(f"{log_prefix}⚠️ [CONTRADICCIÓN] \"{contradiction.new_fact}\" → Borrar IDs: {_ids_ok}")
                        contradiction_ids_by_text.setdefault(contradiction.new_fact, set()).update(_ids_ok)

            if response and response.merges:
                for merge in response.merges:
                    if not (merge.ids_to_delete and merge.merged_fact):
                        continue
                    # [P1-CHAT-FACTS-AUDIT] una fusión que absorbería un hecho clínico
                    # desde uno no clínico (o un id que no se le mostró) se descarta ENTERA:
                    # el hecho nuevo se guarda por separado y los viejos se conservan.
                    _new_cat = new_cat_by_text.get(merge.skip_new_fact, "")
                    _ids_req = {str(_i) for _i in merge.ids_to_delete}
                    _ids_ok = _filter_deletable_ids(
                        merge.ids_to_delete, _new_cat, existing_cat_by_id, clinical,
                        log_prefix=log_prefix, motivo="fusión",
                    )
                    if len(_ids_ok) != len(_ids_req):
                        logger.warning(
                            f"{log_prefix}🛡 [P1-CHAT-FACTS-AUDIT] fusión descartada: "
                            f"\"{merge.merged_fact}\" — el hecho nuevo se guarda por separado."
                        )
                        continue
                    logger.info(f"{log_prefix}🔀 [FUSIÓN] \"{merge.merged_fact}\" ← Absorbe IDs: {_ids_ok}")
                    skipped_new_facts.add(merge.skip_new_fact)

                    original_metadata = {}
                    for pf in facts_with_similar:
                        if pf["fact_text"] == merge.skip_new_fact:
                            original_metadata = dict(pf["metadata"] or {})
                            break
                    # [P1-CHAT-FACTS-AUDIT] si absorbe un hecho clínico, la fusión CONSERVA
                    # la categoría clínica (antes heredaba la metadata del nuevo, o `{}`).
                    if clinical:
                        _md_clinico = next(
                            (existing_md_by_id[_i] for _i in _ids_ok
                             if existing_cat_by_id.get(_i) in clinical),
                            None,
                        )
                        if _md_clinico is not None and original_metadata.get("category") not in clinical:
                            _base = dict(_md_clinico)
                            _base.update({k: v for k, v in original_metadata.items() if k != "category"})
                            original_metadata = _base

                    merged_facts_to_save.append({
                        "fact_text": merge.merged_fact,
                        "metadata": original_metadata,
                        "ids": set(_ids_ok),
                    })

        except Exception as e:
            logger.warning(f"{log_prefix}⚠️ [Error en validación batch de contradicciones/fusiones]: {e}")

    # FASE 3: GUARDAR primero y RETIRAR después.
    # [P1-CHAT-FACTS-AUDIT · 2026-09-14] Antes se soft-borraban los contradictorios y
    # absorbidos ANTES de guardar sus reemplazos: un save fallido (o un embedding vacío
    # en la fusión) dejaba el perfil sin el hecho viejo NI el nuevo. Ahora un hecho
    # viejo solo se retira si su reemplazo quedó persistido.

    # [P2-FACT-SAVE-FAIL-LOUD · 2026-05-30] Chequear el return de
    # save_user_fact ANTES de contar el éxito. Pre-fix: FASE 3 soft-borraba los
    # hechos contradictorios/fusionados (delete_user_fact arriba) y luego
    # guardaba el reemplazo SIN verificar el return; `save_user_fact` traga la
    # excepción de DB y `return None` (db_facts.py:253-255). El contador se
    # incrementaba incondicionalmente y el log "[BATCH COMPLETO]" reportaba
    # éxito aunque el INSERT no persistiera → en un blip transitorio entre el
    # delete y el save, el hecho viejo queda soft-deleted y el corregido NUNCA
    # persiste = pérdida NETA de un hecho de perfil (incluidas
    # alergias/condiciones médicas), con telemetría falsamente positiva. El
    # caller orquestador (async_extract_and_save_facts) NO reintenta un fallo
    # de save dentro del pipeline (su retry solo cubre la adquisición del lock).
    # Fix: contar solo en éxito + log.error VISIBLE (Sentry captura error-level)
    # marcando la categoría crítica para que SRE vea la pérdida.
    # Tooltip-anchor: P2-FACT-SAVE-FAIL-LOUD.
    saved_count = 0
    failed_new_texts = set()
    for pf in prepared_facts:
        if pf["fact_text"] in skipped_new_facts:
            logger.info(f"{log_prefix}⏭️ Hecho absorbido en fusión: '{pf['fact_text']}'")
            continue
        _saved = save_user_fact(user_id, pf["fact_text"], pf["emb"], metadata=pf["metadata"])
        if _saved:
            logger.info(f"{log_prefix}📦 Nuevo hecho guardado: '{pf['fact_text']}' | Metadatos: {pf['metadata']}")
            saved_count += 1
        else:
            failed_new_texts.add(pf["fact_text"])
            complete = False
            _cat = (pf.get("metadata") or {}).get("category")
            logger.error(
                f"{log_prefix}🚨 [P2-FACT-SAVE-FAIL-LOUD] save_user_fact NO persistió "
                f"'{pf['fact_text']}' (category={_cat}, critical={_cat in CRITICAL_CATEGORIES}) "
                f"— si reemplazaba un hecho contradictorio ya soft-deleted, el perfil quedó SIN ese dato."
            )

    ids_to_delete_all = set()
    merge_count = 0
    for mf in merged_facts_to_save:
        merged_emb = get_embedding(mf["fact_text"], purpose="document")  # [P1-COHERE-EMBED-V4] se persiste
        if not merged_emb:
            # [P1-CHAT-FACTS-AUDIT] antes: `continue` con los absorbidos YA borrados.
            # Ahora se guarda sin embedding (el hecho vale más que su vector).
            _cat = (mf.get("metadata") or {}).get("category")
            logger.warning(
                f"{log_prefix}⚠️ [P2-FACT-SAVE-FAIL-LOUD] embedding vacío para hecho fusionado "
                f"'{mf['fact_text']}' (category={_cat}) — se guarda SIN embedding."
            )
            merged_emb = None
        _saved = save_user_fact(user_id, mf["fact_text"], merged_emb, metadata=mf["metadata"])
        if _saved:
            logger.info(f"{log_prefix}🔀 Hecho fusionado guardado: '{mf['fact_text']}' | Metadatos: {mf['metadata']}")
            merge_count += 1
            ids_to_delete_all.update(mf.get("ids") or ())
        else:
            complete = False
            _cat = (mf.get("metadata") or {}).get("category")
            logger.error(
                f"{log_prefix}🚨 [P2-FACT-SAVE-FAIL-LOUD] save_user_fact NO persistió hecho "
                f"fusionado '{mf['fact_text']}' (category={_cat}, critical={_cat in CRITICAL_CATEGORIES}) "
                f"— los hechos absorbidos se CONSERVAN (P1-CHAT-FACTS-AUDIT)."
            )

    for _new_text, _ids in contradiction_ids_by_text.items():
        if _new_text in failed_new_texts:
            logger.warning(
                f"{log_prefix}⚠️ [P1-CHAT-FACTS-AUDIT] '{_new_text}' no se guardó: se conservan "
                f"los hechos que contradecía ({sorted(_ids)})."
            )
            continue
        ids_to_delete_all.update(_ids)

    if ids_to_delete_all:
        logger.info(f"{log_prefix}🗑️ [BATCH] Retirando {len(ids_to_delete_all)} hechos (contradictorios + redundantes)...")
        for f_id in ids_to_delete_all:
            delete_user_fact(f_id, user_id)  # [P1-CHAT-FACTS-AUDIT] I2: filtra por user_id

    total_deleted = len(ids_to_delete_all)
    logger.info(f"{log_prefix}✅ [BATCH COMPLETO] {saved_count} nuevos + {merge_count} fusionados, {total_deleted} eliminados.")
    return complete


class FactExtractionIncomplete(Exception):
    """[P1-CHAT-FACTS-AUDIT · 2026-09-14] La extracción falló (LLM) o dejó hechos sin
    persistir. `process_pending_queue_sync` NO marca como procesado el ítem que la lanza."""


def _is_guest_user_id(user_id: Any) -> bool:
    """[P1-CHAT-FACTS-AUDIT · 2026-09-14] Sin usuario real no hay a quién guardarle
    hechos: `user_facts` y el lock de `user_profiles` son por cuenta. Un invitado
    gastaba el router LLM, no conseguía el lock (no tiene fila en `user_profiles`) y
    su mensaje acababa en la cola persistente. Defensa en profundidad: el router de
    chat ya no debería pasar session_ids."""
    return not user_id or str(user_id).strip().lower() == "guest"


# [P1-CHAT-FACTS-AUDIT · 2026-09-14] Cola persistente: tope y espaciado de reintentos.
# Antes, un ítem que fallaba se marcaba como procesado ("evitar loop infinito") y el
# hecho se perdía. `pending_facts_queue` no tiene contador de intentos (y el DDL en
# runtime está prohibido), así que el tope va por ANTIGÜEDAD (`created_at`) y el
# espaciado vive en memoria del proceso (el cron del drenaje corre cada minuto).
def _pending_fact_max_age_h() -> int:
    return _env_int("MEALFIT_PENDING_FACT_MAX_AGE_H", 72, validator=lambda v: 1 <= v <= 720)


def _pending_fact_retry_backoff_s() -> int:
    return _env_int("MEALFIT_PENDING_FACT_RETRY_BACKOFF_S", 300, validator=lambda v: 0 <= v <= 86400)


_PENDING_RETRY_NOT_BEFORE: dict = {}
_PENDING_RETRY_GUARD = threading.Lock()


def _pending_item_too_old(pending: Any) -> bool:
    created = pending.get("created_at") if isinstance(pending, dict) else None
    if isinstance(created, str):
        try:
            created = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except Exception:
            return False
    if not isinstance(created, datetime):
        return False
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - created > timedelta(hours=_pending_fact_max_age_h())


def async_extract_and_save_facts(user_id: str, message: str, recent_history: str = ""):
    """
    Función orquestadora para ser ejecutada en background.
    Extrae hechos estructurados de un mensaje, revisa contradicciones con la DB
    usando BATCHING (una sola llamada LLM para todos los hechos),
    guarda los nuevos vectorizados y etiquetados y retira los obsoletos.
    """
    # [P1-CHAT-FACTS-AUDIT · 2026-09-14] Guarda de invitado ANTES de gastar LLM.
    if _is_guest_user_id(user_id):
        logger.info("⏭️ [P1-CHAT-FACTS-AUDIT] Extracción omitida: sin usuario real (invitado).")
        return

    # [P1-CHAT-FACTS-AUDIT · 2026-09-14] Token del lock: se libera SOLO si se adquirió,
    # y solo el lock propio. Antes el `finally` llamaba a `release_fact_lock` también en
    # los caminos «el router dice que no hay nada» y «no conseguí el lock, encolo»:
    # le quitaba el lock a la extracción concurrente que sí lo tenía.
    lock_token = None
    try:
        if not should_extract_facts(message, user_id=user_id):
            logger.info("⏭️ [ROUTER] Mensaje ignorado. No contiene hechos relevantes para el perfil.")
            return

        import time
        max_retries = 5
        retry_delay = 2

        for attempt in range(max_retries):
            _token = acquire_fact_lock(user_id)
            if _token:
                lock_token = _token
                break
            logger.warning(f"⚠️ [FACT EXTRACTOR] Extracción en progreso para el usuario {user_id}. Esperando {retry_delay}s ({attempt+1}/{max_retries})...")
            time.sleep(retry_delay)

        if not lock_token:
            # ====== COLA PERSISTENTE: Nunca perder datos clínicos ======
            enqueue_pending_fact(user_id, message, recent_history)
            logger.info(f"📋 [FACT EXTRACTOR] Mensaje encolado en la DB para procesamiento posterior.")
            return

        fact_items = extract_facts(message, recent_history, user_id=user_id)
        if fact_items is None:
            # [P1-CHAT-FACTS-AUDIT] El LLM FALLÓ (≠ «no había hechos»): se encola.
            enqueue_pending_fact(user_id, message, recent_history)
            logger.warning("⚠️ [P1-CHAT-FACTS-AUDIT] Extracción fallida; mensaje encolado para reintento.")
            return
        if not fact_items:
            return

        if _run_fact_pipeline(user_id, fact_items, log_prefix="") is False:
            enqueue_pending_fact(user_id, message, recent_history)
            logger.warning("⚠️ [P1-CHAT-FACTS-AUDIT] Hechos sin persistir; mensaje encolado para reintento.")

    except Exception as e:
        import traceback
        logger.error(f"❌ [CRÍTICO] Fallo general en orquestación de hechos: {e}")
        track = traceback.format_exc()
        logger.info(f"Trazabilidad extendida de error: {track}")
        if lock_token:
            # [P1-CHAT-FACTS-AUDIT] murió a medias con el lock tomado: se encola para
            # no perder un dato clínico (la cola tiene tope de antigüedad).
            try:
                enqueue_pending_fact(user_id, message, recent_history)
            except Exception:
                pass
    finally:
        if lock_token:
            release_fact_lock(user_id, lock_token)

        # PROCESAR COLA PERSISTENTE: la drena el cron `drain_pending_facts_queue`
        # (cron_tasks.py) vía `process_pending_queue_sync`.
        logger.info(f"✅ Extracción en línea terminada. El cron procesará la cola si quedaron pendientes.")


def process_pending_queue_sync(user_id: str):
    """Worker síncrono para drenar la cola de pendientes (cron `drain_pending_facts_queue`
    y endpoint webhook).

    [P1-CHAT-FACTS-AUDIT · 2026-09-14] Solo se borran de la cola los ítems procesados
    con ÉXITO (o descartados por superar `MEALFIT_PENDING_FACT_MAX_AGE_H`, con
    `logger.error`). Un ítem que falla se conserva y no se reintenta antes de
    `MEALFIT_PENDING_FACT_RETRY_BACKOFF_S`.
    """
    lock_token = acquire_fact_lock(user_id)
    if not lock_token:
        logger.warning(f"⚠️ [WEBHOOK QUEUE] Lock ocupado para {user_id}. Se procesará luego.")
        return

    try:
        pending_items = dequeue_pending_facts(user_id)
        if not pending_items:
            logger.info("➡️ [WEBHOOK QUEUE] No hay hechos pendientes en cola.")
            return

        logger.info(f"\n📋 [FACT EXTRACTOR WEBHOOK] Iniciando drenaje estructurado para {len(pending_items)} mensajes pendientes...")
        processed_ids = []
        now_m = _time_module.monotonic()
        with _PENDING_RETRY_GUARD:
            for _k in [k for k, v in _PENDING_RETRY_NOT_BEFORE.items() if v < now_m - 86400]:
                _PENDING_RETRY_NOT_BEFORE.pop(_k, None)
        for idx, pending in enumerate(pending_items, 1):
            _pid = str(pending["id"])
            if _pending_item_too_old(pending):
                logger.error(
                    f"🚨 [P1-CHAT-FACTS-AUDIT] Pendiente {_pid} descartado tras "
                    f"{_pending_fact_max_age_h()}h sin procesarse con éxito: "
                    f"'{str(pending.get('message', ''))[:80]}'"
                )
                processed_ids.append(pending["id"])
                with _PENDING_RETRY_GUARD:
                    _PENDING_RETRY_NOT_BEFORE.pop(_pid, None)
                continue
            with _PENDING_RETRY_GUARD:
                _not_before = _PENDING_RETRY_NOT_BEFORE.get(_pid)
            if _not_before is not None and now_m < _not_before:
                continue
            try:
                logger.info(f"   📋 [{idx}/{len(pending_items)}] Procesando: '{pending['message'][:50]}...'")
                _process_single_extraction(user_id, pending["message"], pending.get("recent_history", ""))
                processed_ids.append(pending["id"])
                with _PENDING_RETRY_GUARD:
                    _PENDING_RETRY_NOT_BEFORE.pop(_pid, None)
            except Exception as pe:
                # [P1-CHAT-FACTS-AUDIT] antes: se marcaba procesado igual y el hecho se perdía.
                logger.warning(f"   ⚠️ Error procesando pendiente #{idx} (se conserva para reintento): {pe}")
                with _PENDING_RETRY_GUARD:
                    _PENDING_RETRY_NOT_BEFORE[_pid] = now_m + _pending_fact_retry_backoff_s()

        if processed_ids:
            delete_pending_facts(processed_ids)
        logger.info(f"✅ [FACT EXTRACTOR] Cola pendiente finalizada en Hilo Secundario.")
    except Exception as qe:
        logger.warning(f"⚠️ Error general en hilo secundario de cola: {qe}")
    finally:
        release_fact_lock(user_id, lock_token)


def _process_single_extraction(user_id: str, message: str, recent_history: str = ""):
    """
    Procesa una sola extracción de hechos SIN manejar el lock.
    Usado internamente para drenar la cola de pendientes.
    Delega al pipeline compartido _run_fact_pipeline.

    [P1-CHAT-FACTS-AUDIT · 2026-09-14] Lanza `FactExtractionIncomplete` si el LLM
    falla o quedan hechos sin persistir: el ítem se conserva en la cola.
    """
    if not should_extract_facts(message, user_id=user_id):
        return

    fact_items = extract_facts(message, recent_history, user_id=user_id)
    if fact_items is None:
        raise FactExtractionIncomplete("extract_facts falló (LLM)")
    if not fact_items:
        return

    if _run_fact_pipeline(user_id, fact_items, log_prefix="   ") is False:
        raise FactExtractionIncomplete("hechos sin persistir")
