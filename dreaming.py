"""[P1-DREAMING-1 · 2026-06-13] Motor de consolidación de memoria ("Dreaming").

Capa híbrida sobre el RAG existente (user_facts + Cohere embed-v4 + match_user_facts):
un ciclo OFFLINE tipo "sueño" que de-duplica facts globalmente (lo que el pipeline
online no cruza porque solo ve K vecinos), aplica salience + decay, resuelve
contradicciones cross-sesión, y sintetiza UNA "memoria semántica de alto nivel"
por usuario (user_memory_profile) con evidencia FK-verificada (anti-confabulación).

INVARIANTES DE SEGURIDAD:
  - Categorías clínicas (alergia, condicion_medica) JAMÁS se auto-mergean ni
    pierden salience (floor 1.0, fail-secure). Solo preferencias no-críticas.
  - Toda query filtra `AND user_id = %s` (I2, anti-IDOR).
  - `acquire_fact_lock(user_id)` serializa el dream contra el extractor online
    (cierra double-write) y contra otros workers del dream.
  - Merges = soft-delete REVERSIBLE (is_active=FALSE + metadata.consolidated_into),
    nunca hard DELETE; el dream_consolidation_log guarda lo borrado para revertir.
  - El user_model SOLO se persiste si su evidence_fact_ids referencia facts reales
    del MISMO user_id (verificado en runtime, no prompt-trustable — espíritu P0-AGENT-1).
  - Master kill-switch `MEALFIT_DREAMING_ENABLED` (default False): el cron hace
    early-return → cero costo, sistema idéntico a hoy (rollback sin redeploy).

Doc canónica: backend/docs/dreaming_consolidation.md. Tests: test_p1_dreaming_1_*.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from pydantic import BaseModel, Field

# Helpers DB via la fachada pública (P3-DB-IMPORTS-FACADE) + el pool para el
# claim leader-safe (FOR UPDATE SKIP LOCKED necesita control transaccional).
from db import execute_sql_query, execute_sql_write
import db_core
from psycopg.types.json import Jsonb

# Knobs auto-registrados en `_KNOBS_REGISTRY` (P3-NEW-D).
from knobs import _env_int, _env_float, _env_bool, _env_str

logger = logging.getLogger(__name__)

# Categorías médicas EXENTAS de merge/decay (fail-secure).
CLINICAL_CATEGORIES = ("alergia", "condicion_medica")

# ===========================================================================
# Knobs
# ===========================================================================
def _dreaming_enabled() -> bool:
    return _env_bool("MEALFIT_DREAMING_ENABLED", False)

def _dreaming_retrieval_enabled() -> bool:
    return _env_bool("MEALFIT_DREAMING_RETRIEVAL_ENABLED", False)

def _dreaming_inject_plan_enabled() -> bool:
    return _env_bool("MEALFIT_DREAMING_INJECT_PLAN_ENABLED", False)

def _dreaming_model_name() -> str:
    # Override per-feature gana sobre el router por tier (P3-PREVIEW-MODEL-KNOB).
    # NUNCA pro: offline, no médico-crítico.
    from llm_provider import DEEPSEEK_FLASH
    return _env_str("MEALFIT_DREAMING_MODEL", DEEPSEEK_FLASH) or DEEPSEEK_FLASH

def _dreaming_interval_hours() -> int:
    return _env_int("MEALFIT_DREAMING_CONSOLIDATION_INTERVAL_HOURS", 24,
                    validator=lambda v: 6 <= v <= 168)

def _dreaming_max_users_per_night() -> int:
    return _env_int("MEALFIT_DREAMING_MAX_USERS_PER_NIGHT", 200,
                    validator=lambda v: 0 <= v <= 5000)

def _dreaming_max_cost_usd_per_night() -> float:
    return _env_float("MEALFIT_DREAMING_MAX_COST_USD_PER_NIGHT", 2.0,
                      validator=lambda v: 0.0 <= v <= 100.0)

def _dreaming_max_facts_per_call() -> int:
    return _env_int("MEALFIT_DREAMING_MAX_FACTS_PER_CALL", 60,
                    validator=lambda v: 10 <= v <= 200)

def _dreaming_salience_decay_rate() -> float:
    return _env_float("MEALFIT_DREAMING_SALIENCE_DECAY_RATE", 0.05,
                      validator=lambda v: 0.0 <= v <= 0.5)

def _dreaming_contradiction_alerts() -> bool:
    return _env_bool("MEALFIT_DREAMING_CONTRADICTION_ALERTS", True)

def _dreaming_batch() -> int:
    return _env_int("MEALFIT_DREAMING_BATCH", 50, validator=lambda v: 1 <= v <= 500)

def _dreaming_llm_timeout_s() -> float:
    return _env_float("MEALFIT_DREAMING_LLM_TIMEOUT_S", 60.0,
                      validator=lambda v: 5.0 <= v <= 180.0)

# Precio blended flash USD/1k tokens (cap de seguridad, NO billing exacto).
def _dreaming_usd_per_1k() -> float:
    return _env_float("MEALFIT_DREAMING_USD_PER_1K_TOKENS", 0.0003,
                      validator=lambda v: 0.0 <= v <= 1.0)


# ===========================================================================
# Schema del LLM (1 call por usuario: dedup + contradicciones + síntesis)
# ===========================================================================
class DreamMerge(BaseModel):
    canonical_fact: str = Field(description="El hecho canónico fusionado (más completo que los individuales).")
    source_fact_ids: List[str] = Field(description="IDs (UUID) de los hechos redundantes que reemplaza el canónico.")

class DreamContradiction(BaseModel):
    description: str = Field(description="Descripción breve de la contradicción detectada entre hechos del usuario.")
    fact_ids: List[str] = Field(description="IDs (UUID) de los hechos en conflicto.")

class DreamConsolidationResult(BaseModel):
    merges: List[DreamMerge] = Field(default_factory=list,
        description="Fusiones de preferencias NO-clínicas redundantes. Vacío si no hay.")
    contradictions: List[DreamContradiction] = Field(default_factory=list,
        description="Contradicciones detectadas (NO se auto-resuelven). Vacío si no hay.")
    user_model: str = Field(default="",
        description="6-8 frases en español dominicano que describen al usuario para personalizar su nutrición.")
    evidence_fact_ids: List[str] = Field(default_factory=list,
        description="IDs (UUID) de los hechos que respaldan el user_model. Cada afirmación debe tener base.")


# ===========================================================================
# Cola de trabajo (dream_work_queue)
# ===========================================================================
def enqueue_dream_work(user_id: str, reason: str = "manual") -> None:
    """Encola un usuario para consolidación. Idempotente: el unique partial index
    (user_id WHERE processed_at IS NULL) deduplica vía ON CONFLICT DO NOTHING.
    Best-effort: nunca propaga (no debe romper el caller — p.ej. el hook post-summary)."""
    if not _dreaming_enabled():
        return
    try:
        execute_sql_write(
            "INSERT INTO dream_work_queue (user_id, trigger_reason) VALUES (%s, %s) "
            "ON CONFLICT (user_id) WHERE processed_at IS NULL DO NOTHING",
            (user_id, reason),
        )
    except Exception as e:
        logger.warning(f"[P1-DREAMING-1] enqueue_dream_work({str(user_id)[:8]}, {reason}) falló (best-effort): {e}")


def _enqueue_dirty_users(cap: int) -> int:
    """Encola usuarios "dream-dirty" (>=2 facts activos + nunca consolidados o
    stale por el intervalo), ordenados por staleness. ON CONFLICT DO NOTHING."""
    interval_h = _dreaming_interval_hours()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=interval_h)
    try:
        res = execute_sql_write(
            """
            INSERT INTO dream_work_queue (user_id, trigger_reason)
            SELECT uf.user_id, 'nightly_sweep'
            FROM user_facts uf
            WHERE uf.is_active = TRUE
            GROUP BY uf.user_id
            HAVING COUNT(*) >= 2
               AND (MAX(uf.last_consolidated_at) IS NULL OR MAX(uf.last_consolidated_at) < %s)
            ORDER BY MAX(uf.last_consolidated_at) ASC NULLS FIRST
            LIMIT %s
            ON CONFLICT (user_id) WHERE processed_at IS NULL DO NOTHING
            RETURNING id
            """,
            (cutoff, int(cap)),
            returning=True,
        )
        return len(res) if res else 0
    except Exception as e:
        logger.warning(f"[P1-DREAMING-1] _enqueue_dirty_users falló (best-effort): {e}")
        return 0


def _claim_next_dream_work() -> Optional[dict]:
    """Reclama atómicamente el siguiente trabajo pendiente: SELECT ... FOR UPDATE
    SKIP LOCKED LIMIT 1 (leader-safe multi-worker) + bump de attempts. La tx es
    CORTA (no abarca el LLM call), así que el lock se libera de inmediato; la
    exclusión real por-usuario la da `acquire_fact_lock`. Devuelve {id, user_id}
    o None si no hay trabajo."""
    if not db_core.connection_pool:
        return None
    try:
        with db_core.connection_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id::text, user_id::text FROM dream_work_queue "
                    "WHERE processed_at IS NULL ORDER BY enqueued_at "
                    "FOR UPDATE SKIP LOCKED LIMIT 1"
                )
                row = cur.fetchone()
                if not row:
                    return None
                qid, uid = row[0], row[1]
                cur.execute("UPDATE dream_work_queue SET attempts = attempts + 1 WHERE id = %s", (qid,))
            return {"id": qid, "user_id": uid}
    except Exception as e:
        logger.warning(f"[P1-DREAMING-1] _claim_next_dream_work falló: {e}")
        return None


def _mark_dream_work_done(work_id: str, error: Optional[str] = None) -> None:
    try:
        execute_sql_write(
            "UPDATE dream_work_queue SET processed_at = now(), last_error = %s WHERE id = %s",
            (str(error)[:500] if error else None, work_id),
        )
    except Exception as e:
        logger.warning(f"[P1-DREAMING-1] _mark_dream_work_done({work_id}) falló: {e}")


def dream_backlog_size() -> int:
    try:
        r = execute_sql_query(
            "SELECT COUNT(*) AS n FROM dream_work_queue WHERE processed_at IS NULL",
            fetch_one=True,
        )
        return int(r["n"]) if r and r.get("n") is not None else 0
    except Exception:
        return -1


# ===========================================================================
# Budget diario (app_kv_store, key con fecha = reset natural sin cron extra)
# ===========================================================================
def _budget_key() -> str:
    return f"dreaming_budget_spent:{datetime.now(timezone.utc).date().isoformat()}"

def _get_budget_spent() -> float:
    try:
        r = execute_sql_query("SELECT value FROM app_kv_store WHERE key = %s", (_budget_key(),), fetch_one=True)
        if r and isinstance(r.get("value"), dict):
            return float(r["value"].get("usd", 0.0) or 0.0)
    except Exception:
        pass
    return 0.0

def _add_budget_spent(usd: float) -> None:
    try:
        execute_sql_write(
            "INSERT INTO app_kv_store (key, value, updated_at) VALUES (%s, %s::jsonb, NOW()) "
            "ON CONFLICT (key) DO UPDATE SET "
            "value = jsonb_set(app_kv_store.value, '{usd}', "
            "  to_jsonb(COALESCE((app_kv_store.value->>'usd')::float, 0) + %s)), updated_at = NOW()",
            (_budget_key(), Jsonb({"usd": round(usd, 6)}), round(usd, 6)),
        )
    except Exception as e:
        logger.warning(f"[P1-DREAMING-1] _add_budget_spent falló: {e}")


# ===========================================================================
# user_memory_profile (capa de reflexión, 1 fila/usuario)
# ===========================================================================
def get_user_memory_profile(user_id: str) -> Optional[dict]:
    """Lee el user_model vivo del usuario (para inyección al prompt). None si no existe."""
    if not user_id:
        return None
    try:
        return execute_sql_query(
            "SELECT user_id::text, user_model, (embedding IS NOT NULL) AS has_embedding, "
            "facts_synthesized_from, updated_at "
            "FROM user_memory_profile WHERE user_id = %s AND is_active = TRUE",
            (user_id,),
            fetch_one=True,
        )
    except Exception as e:
        logger.warning(f"[P1-DREAMING-1] get_user_memory_profile({str(user_id)[:8]}) falló: {e}")
        return None


def build_plan_constraints_block(user_id: Optional[str]) -> str:
    """[P1-DREAMING-1 · 2026-06-13 · Fase 4] Bloque de restricciones dinámicas
    para el GENERADOR DE PLANES, derivado del user_model consolidado. Cierra el
    silo conversación→plan: el plan se diseña conociendo las preferencias/
    restricciones sintetizadas por el Dreaming. Gateado por
    MEALFIT_DREAMING_INJECT_PLAN_ENABLED (default OFF → '' → plan idéntico a hoy).
    Fail-open: cualquier error → '' (jamás bloquea la generación del plan)."""
    if not user_id or not _dreaming_inject_plan_enabled():
        return ""
    try:
        prof = get_user_memory_profile(user_id)
        model = (prof or {}).get("user_model") if prof else None
        if not model:
            return ""
        cap = _env_int("MEALFIT_DREAMING_PROMPT_MAX_CHARS", 1200,
                       validator=lambda v: 0 <= v <= 4000)
        return (
            "\n--- 🧠 MODELO DEL USUARIO (memoria consolidada — respétalo al diseñar el plan) ---\n"
            + str(model)[:cap]
            + "\n--------------------------------------------------------------------------------\n"
        )
    except Exception as e:
        logger.debug(f"[P1-DREAMING-1] build_plan_constraints_block falló (fail-open): {e}")
        return ""


def _verify_evidence_fact_ids(user_id: str, evidence_ids: List[str]) -> List[str]:
    """ANTI-CONFABULACIÓN: filtra evidence_ids a los que existan en user_facts del
    MISMO user_id (activos). Devuelve solo los válidos (enforced, no prompt-trustable)."""
    if not evidence_ids:
        return []
    try:
        rows = execute_sql_query(
            "SELECT id::text AS id FROM user_facts "
            "WHERE id::text = ANY(%s) AND user_id = %s AND is_active = TRUE",
            ([str(x) for x in evidence_ids], user_id),
            fetch_all=True,
        ) or []
        return [r["id"] for r in rows]
    except Exception as e:
        logger.warning(f"[P1-DREAMING-1] _verify_evidence_fact_ids falló: {e}")
        return []


def _upsert_user_memory_profile(user_id: str, user_model: str, embedding: Optional[list],
                                evidence_ids: List[str], model_id: str, facts_count: int) -> bool:
    emb_str = f"[{','.join(map(str, embedding))}]" if embedding else None
    try:
        execute_sql_write(
            "INSERT INTO user_memory_profile "
            "(user_id, user_model, embedding, evidence_fact_ids, source_model, facts_synthesized_from, is_active, updated_at) "
            "VALUES (%s, %s, %s::extensions.vector, %s::uuid[], %s, %s, TRUE, now()) "
            "ON CONFLICT (user_id) DO UPDATE SET "
            "user_model = EXCLUDED.user_model, embedding = EXCLUDED.embedding, "
            "evidence_fact_ids = EXCLUDED.evidence_fact_ids, source_model = EXCLUDED.source_model, "
            "facts_synthesized_from = EXCLUDED.facts_synthesized_from, is_active = TRUE, updated_at = now()",
            (user_id, user_model, emb_str, [str(x) for x in evidence_ids], model_id, int(facts_count)),
        )
        return True
    except Exception as e:
        logger.error(f"[P1-DREAMING-1] _upsert_user_memory_profile({str(user_id)[:8]}) falló: {e}")
        return False


# ===========================================================================
# Núcleo: consolidación por usuario
# ===========================================================================
def _get_active_facts(user_id: str) -> List[dict]:
    return execute_sql_query(
        "SELECT id::text AS id, fact, metadata, salience_score, "
        "COALESCE(metadata->>'category','') AS category "
        "FROM user_facts WHERE user_id = %s AND is_active = TRUE "
        "ORDER BY created_at ASC",
        (user_id,),
        fetch_all=True,
    ) or []


def _soft_delete_facts(user_id: str, fact_ids: List[str], canonical_text: str) -> int:
    """Soft-delete REVERSIBLE de facts redundantes. NUNCA toca clínicos (defensa
    redundante: el LLM no debería proponerlos, pero el filtro SQL lo garantiza)."""
    if not fact_ids:
        return 0
    now_iso = datetime.now(timezone.utc).isoformat()
    res = execute_sql_write(
        "UPDATE user_facts SET is_active = FALSE, "
        "metadata = COALESCE(metadata,'{}'::jsonb) || %s "
        "WHERE user_id = %s AND id::text = ANY(%s) "
        "AND COALESCE(metadata->>'category','') NOT IN ('alergia','condicion_medica') "
        "RETURNING id::text",
        (Jsonb({"consolidated_into": canonical_text[:300], "consolidated_at": now_iso}),
         user_id, [str(x) for x in fact_ids]),
        returning=True,
    )
    return len(res) if res else 0


def _insert_canonical_fact(user_id: str, fact: str, salience: float) -> Optional[str]:
    """Inserta el fact canónico fusionado con su embedding (purpose='document')
    + salience boost + consolidation_source. Reusa get_embedding (cache Cohere)."""
    from fact_extractor import get_embedding
    emb = get_embedding(fact, purpose="document")  # asimetría: persistido => search_document
    emb_str = f"[{','.join(map(str, emb))}]" if emb else None
    try:
        res = execute_sql_write(
            "INSERT INTO user_facts (user_id, fact, embedding, metadata, salience_score, "
            "consolidation_source, last_consolidated_at) "
            "VALUES (%s, %s, %s::extensions.vector, %s, %s, 'dream_merge_canonical', now()) "
            "RETURNING id::text",
            (user_id, fact, emb_str,
             Jsonb({"category": "preferencia", "source": "dream_consolidation"}),
             float(salience)),
            returning=True,
        )
        # invalida el cache RAG del usuario (mismo contrato que save_user_fact)
        try:
            from db_facts import _invalidate_rag_cache
            _invalidate_rag_cache(user_id)
        except Exception:
            pass
        return res[0]["id"] if res else None
    except Exception as e:
        logger.error(f"[P1-DREAMING-1] _insert_canonical_fact falló: {e}")
        return None


def _apply_salience_maintenance(user_id: str, decay_rate: float) -> None:
    """Floor clínico 1.0 (fail-secure) + decay de no-clínicos no reforzados."""
    try:
        # 1) Floor médico: alergia/condicion_medica nunca decaen.
        execute_sql_write(
            "UPDATE user_facts SET salience_score = 1.0 "
            "WHERE user_id = %s AND is_active = TRUE "
            "AND COALESCE(metadata->>'category','') IN ('alergia','condicion_medica') "
            "AND salience_score < 1.0",
            (user_id,),
        )
        # 2) Decay de no-clínicos (clamp >= 0).
        execute_sql_write(
            "UPDATE user_facts SET salience_score = GREATEST(0, salience_score - %s) "
            "WHERE user_id = %s AND is_active = TRUE "
            "AND COALESCE(metadata->>'category','') NOT IN ('alergia','condicion_medica') "
            "AND consolidation_source IS DISTINCT FROM 'dream_merge_canonical'",
            (float(decay_rate), user_id),
        )
    except Exception as e:
        logger.warning(f"[P1-DREAMING-1] _apply_salience_maintenance falló: {e}")


def _build_dream_prompt(facts: List[dict]) -> str:
    lines = []
    for f in facts:
        cat = f.get("category") or "?"
        lines.append(f'- id={f["id"]} [{cat}] {f["fact"]}')
    facts_block = "\n".join(lines)
    return f"""Eres el "sistema de consolidación de memoria" (estilo sueño) de un coach nutricional dominicano.
Recibes TODOS los hechos activos de UN usuario. Tu trabajo OFFLINE:

1) FUSIONAR (merges): agrupa hechos de PREFERENCIA redundantes/complementarios en UN hecho canónico
   más completo. Indica los source_fact_ids fusionados. NO fusiones hechos de categorías 'alergia' o
   'condicion_medica' — esos son clínicos y NUNCA se tocan.
2) CONTRADICCIONES: marca (sin resolver) pares de hechos que se contradigan, sobre todo los clínicos.
   Solo describe y lista fact_ids; NO los borres.
3) USER_MODEL: sintetiza 6-8 frases en español dominicano que describan al usuario para personalizar
   su nutrición (preferencias fuertes, rechazos, hábitos, objetivos, restricciones). Sé concreto y
   citable: cada afirmación debe basarse en hechos reales de la lista. Lista en evidence_fact_ids los
   IDs que respaldan tu síntesis. NO inventes preferencias sin base. NO inventes alergias/condiciones.

Hechos del usuario:
{facts_block}

Responde SOLO con el schema estructurado. Si no hay nada que fusionar, deja merges vacío."""


def _estimate_cost_usd(prompt: str, output_obj) -> float:
    try:
        out_len = len(str(output_obj))
    except Exception:
        out_len = 0
    tokens = (len(prompt) + out_len) / 4.0
    return (tokens / 1000.0) * _dreaming_usd_per_1k()


def consolidate_user(user_id: str) -> dict:
    """Ciclo de Dreaming para UN usuario. Devuelve un dict de telemetría con
    status ∈ {ok, skipped_few_facts, skipped_locked, budget_exhausted,
    breaker_open, error}. Best-effort: no propaga excepciones."""
    from fact_extractor import get_embedding
    from db_facts import acquire_fact_lock, release_fact_lock

    result = {"user_id": user_id, "status": "ok", "facts_in": 0, "merges": 0,
              "contradictions": 0, "profile_updated": False, "cost_usd": 0.0}

    # Exclusión por-usuario contra el extractor online Y otros workers del dream.
    if not acquire_fact_lock(user_id):
        result["status"] = "skipped_locked"
        return result

    try:
        facts = _get_active_facts(user_id)
        result["facts_in"] = len(facts)
        if len(facts) < 2:
            # Nada que consolidar: marca consolidado para no re-evaluar pronto.
            execute_sql_write(
                "UPDATE user_facts SET last_consolidated_at = now() WHERE user_id = %s AND is_active = TRUE",
                (user_id,),
            )
            result["status"] = "skipped_few_facts"
            return result

        # Budget global del día (cap de seguridad).
        if _get_budget_spent() >= _dreaming_max_cost_usd_per_night():
            result["status"] = "budget_exhausted"
            return result

        model_id = _dreaming_model_name()

        # Circuit breaker del modelo (reusa el KV global por-modelo).
        try:
            from graph_orchestrator import LLMCircuitBreaker
            if not LLMCircuitBreaker(model_id).can_proceed():
                result["status"] = "breaker_open"
                return result
        except Exception:
            pass  # si el CB no está disponible, no bloqueamos el dream

        # Trocea por cap de facts/call (mantiene 1-call-ideal en el caso típico).
        facts = facts[: _dreaming_max_facts_per_call()]
        prompt = _build_dream_prompt(facts)

        from llm_provider import ChatDeepSeek
        llm = ChatDeepSeek(model=model_id, temperature=0.2,
                           timeout=_dreaming_llm_timeout_s(), max_retries=0
                           ).with_structured_output(DreamConsolidationResult)
        try:
            parsed: DreamConsolidationResult = llm.invoke(prompt)
            try:
                from graph_orchestrator import LLMCircuitBreaker
                LLMCircuitBreaker(model_id).record_success()
            except Exception:
                pass
        except Exception as llm_err:
            try:
                from graph_orchestrator import LLMCircuitBreaker
                LLMCircuitBreaker(model_id).record_failure()
            except Exception:
                pass
            raise llm_err

        if parsed is None:
            result["status"] = "error"
            return result

        cost = _estimate_cost_usd(prompt, parsed)
        result["cost_usd"] = round(cost, 6)
        _add_budget_spent(cost)

        valid_ids = {f["id"] for f in facts}

        # --- 1) Aplicar merges (soft-delete reversible + canónico) ---
        soft_deleted = []
        merges_applied = 0
        for m in (parsed.merges or []):
            src = [s for s in (m.source_fact_ids or []) if s in valid_ids]
            if len(src) < 2 or not (m.canonical_fact or "").strip():
                continue  # un merge necesita >=2 fuentes reales
            # snapshot para revertir
            for f in facts:
                if f["id"] in src:
                    soft_deleted.append({"fact_id": f["id"], "fact_text": f["fact"]})
            n = _soft_delete_facts(user_id, src, m.canonical_fact)
            if n > 0:
                # salience del canónico crece con cuántos hechos resume
                boost = min(1.0, 0.5 + 0.12 * n)
                _insert_canonical_fact(user_id, m.canonical_fact.strip(), boost)
                merges_applied += 1
        result["merges"] = merges_applied

        # --- 2) Salience: floor clínico + decay no-clínico ---
        _apply_salience_maintenance(user_id, _dreaming_salience_decay_rate())

        # --- 3) Contradicciones: alerta, NUNCA muta facts clínicos ---
        contradictions = [c for c in (parsed.contradictions or []) if c.fact_ids]
        result["contradictions"] = len(contradictions)
        if contradictions and _dreaming_contradiction_alerts():
            try:
                from cron_tasks import _persist_dream_contradiction_alert
                _persist_dream_contradiction_alert(user_id, contradictions)
            except Exception:
                pass

        # --- 4) Reflexión: user_model con evidencia FK-verificada ---
        user_model = (parsed.user_model or "").strip()
        if user_model:
            valid_evidence = _verify_evidence_fact_ids(user_id, parsed.evidence_fact_ids or [])
            if valid_evidence:
                emb = get_embedding(user_model, purpose="document")
                ok = _upsert_user_memory_profile(
                    user_id, user_model, emb if emb else None,
                    valid_evidence, model_id, len(facts),
                )
                result["profile_updated"] = ok
            else:
                logger.warning(
                    f"[P1-DREAMING-1] user_model descartado para {str(user_id)[:8]}: "
                    f"0 evidencia válida (anti-confabulación)."
                )

        # --- 5) Marca consolidado + audit log ---
        execute_sql_write(
            "UPDATE user_facts SET last_consolidated_at = now() WHERE user_id = %s AND is_active = TRUE",
            (user_id,),
        )
        try:
            execute_sql_write(
                "INSERT INTO dream_consolidation_log "
                "(user_id, facts_in, merges_applied, facts_soft_deleted, contradictions_detected, "
                " profile_updated, model_id, tokens_estimated, cost_usd) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (user_id, len(facts), merges_applied, Jsonb(soft_deleted),
                 len(contradictions), result["profile_updated"], model_id,
                 int((len(prompt) + len(str(parsed))) / 4.0), result["cost_usd"]),
            )
        except Exception as log_err:
            logger.warning(f"[P1-DREAMING-1] audit log falló (no fatal): {log_err}")

        return result
    except Exception as e:
        logger.error(f"[P1-DREAMING-1] consolidate_user({str(user_id)[:8]}) error: {e}")
        result["status"] = "error"
        result["error"] = str(e)[:300]
        return result
    finally:
        release_fact_lock(user_id)


def run_dream_cycle() -> dict:
    """Punto de entrada del cron. Encola dirty users + procesa hasta el cap nocturno.
    Early-return neutral si el knob master está OFF. Devuelve telemetría agregada."""
    agg = {"enabled": _dreaming_enabled(), "enqueued": 0, "processed": 0,
           "merges": 0, "contradictions": 0, "profiles": 0, "cost_usd": 0.0,
           "backlog": 0, "budget_exhausted": False}
    if not _dreaming_enabled():
        return agg

    max_users = _dreaming_max_users_per_night()
    if max_users <= 0:
        agg["backlog"] = dream_backlog_size()
        return agg

    agg["enqueued"] = _enqueue_dirty_users(max_users)

    for _ in range(max_users):
        claim = _claim_next_dream_work()
        if not claim:
            break
        res = consolidate_user(claim["user_id"])
        if res.get("status") == "budget_exhausted":
            agg["budget_exhausted"] = True
            # NO marcar processed → reintento la próxima noche.
            break
        _mark_dream_work_done(claim["id"], res.get("error"))
        agg["processed"] += 1
        agg["merges"] += res.get("merges", 0)
        agg["contradictions"] += res.get("contradictions", 0)
        agg["profiles"] += 1 if res.get("profile_updated") else 0
        agg["cost_usd"] = round(agg["cost_usd"] + res.get("cost_usd", 0.0), 6)

    agg["backlog"] = dream_backlog_size()
    return agg
