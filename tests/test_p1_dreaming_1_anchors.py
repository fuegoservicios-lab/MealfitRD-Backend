"""[P1-DREAMING-1 · 2026-06-13] Anclas del sistema híbrido RAG + Dreaming.

Cubre (parser-based + funcional liviano — sin DB viva salvo el smoke neutral):
1. Migración idempotente + Neon-safe (sin auth.users / RLS / auth.uid()).
2. SSOT dual-dir byte-idéntico (P3-MIGRATIONS-SSOT).
3. Scoping user_id (I2) + exención clínica + FOR UPDATE SKIP LOCKED en dreaming.py.
4. Firma `build_memory_context(session_id, user_id)` + propagación en agent.py.
5. Knobs MEALFIT_DREAMING_* presentes + defaults seguros (OFF) + neutralidad flag-OFF.
6. RPC match_user_memory clona el estilo Neon (search_path, sin SECURITY DEFINER/REVOKE).
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent  # workspace root (tiene migrations espejo)
_MIG_NAME = "p1_dreaming_1_salience_profile_2026_06_13.sql"


def _read(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def _strip_sql_comments(sql: str) -> str:
    """Quita comentarios `-- ...` para chequear DDL real, no la prosa explicativa
    (la migración menciona auth.users/auth.uid() en comentarios para decir que NO
    los usa — eso no debe confundir a las aserciones de Neon-safety)."""
    return "\n".join(line.split("--")[0] for line in sql.splitlines())


# ---------------------------------------------------------------------------
# 1. Migración idempotente + Neon-safe
# ---------------------------------------------------------------------------
def test_migration_idempotent_and_additive():
    sql = _read(f"migrations/{_MIG_NAME}")
    # Idempotencia
    assert "ADD COLUMN IF NOT EXISTS salience_score" in sql
    assert "DROP CONSTRAINT IF EXISTS user_facts_salience_range" in sql
    assert "CREATE TABLE IF NOT EXISTS public.user_memory_profile" in sql
    assert "CREATE TABLE IF NOT EXISTS public.dream_work_queue" in sql
    assert "CREATE TABLE IF NOT EXISTS public.dream_consolidation_log" in sql
    assert "CREATE INDEX IF NOT EXISTS" in sql
    # dedup leader-safe: unique partial index del trabajo pendiente
    assert "CREATE UNIQUE INDEX IF NOT EXISTS idx_dream_queue_user_pending" in sql
    assert "WHERE processed_at IS NULL" in sql
    # Sanity DO $$ que falla ruidoso
    assert "RAISE EXCEPTION" in sql
    # RPC con search_path qualified (estilo Neon)
    assert "CREATE OR REPLACE FUNCTION public.match_user_memory" in sql
    assert "SET search_path TO 'public', 'extensions'" in sql


def test_migration_is_neon_safe_no_supabase_auth_or_rls():
    raw = _read(f"migrations/{_MIG_NAME}")
    sql = _strip_sql_comments(raw)  # chequea DDL real, no la prosa de los comentarios
    # Neon: NO existe auth.users → las FKs van a public.user_profiles.
    assert "auth.users" not in sql, "Neon no tiene auth.users; FK debe ir a user_profiles"
    assert "REFERENCES public.user_profiles(id)" in sql
    # Neon: sin RLS / auth.uid() (scoping app-side AND user_id=%s).
    assert "auth.uid()" not in sql, "auth.uid() es Supabase-ism; no existe en Neon"
    assert "ROW LEVEL SECURITY" not in sql.upper()
    # F0-NEUTRAL: NO redefine match_user_facts (cuyo ORDER BY es por distancia cruda).
    assert "FUNCTION public.match_user_facts(" not in sql


def test_migration_uses_extensions_vector_1536():
    sql = _read(f"migrations/{_MIG_NAME}")
    assert "extensions.vector(1536)" in sql
    assert "extensions.vector_cosine_ops" in sql  # HNSW opclass qualified


# ---------------------------------------------------------------------------
# 2. SSOT dual-dir byte-idéntico
# ---------------------------------------------------------------------------
def test_migration_ssot_dual_dir_identical():
    backend_sql = (_BACKEND / "migrations" / _MIG_NAME).read_bytes()
    root_path = _ROOT / "migrations" / _MIG_NAME
    assert root_path.exists(), f"Falta la copia SSOT en {root_path} (P3-MIGRATIONS-SSOT)"
    assert backend_sql == root_path.read_bytes(), (
        "La migración de Dreaming NO es byte-idéntica entre backend/migrations "
        "y migrations (P3-MIGRATIONS-SSOT dual-dir)."
    )


# ---------------------------------------------------------------------------
# 3. dreaming.py: scoping user_id (I2) + exención clínica + SKIP LOCKED
# ---------------------------------------------------------------------------
def test_dreaming_user_id_scoping_and_locks():
    src = _read("dreaming.py")
    # Exclusión por-usuario reusando el lock del extractor online.
    assert "acquire_fact_lock(user_id)" in src
    assert "release_fact_lock(user_id)" in src
    # Pickup leader-safe multi-worker.
    assert "FOR UPDATE SKIP LOCKED" in src
    # Toda mutación user-scoped filtra user_id (muestreo de los UPDATE/INSERT clave).
    for needle in (
        "UPDATE user_facts SET is_active = FALSE",      # soft-delete
        "WHERE user_id = %s",                            # filtro genérico presente
        "INSERT INTO user_facts (user_id,",             # canónico
        "ON CONFLICT (user_id) DO UPDATE",              # upsert profile
    ):
        assert needle in src, f"dreaming.py debe contener {needle!r} (scoping/contrato)"


def test_dreaming_medical_categories_exempt():
    src = _read("dreaming.py")
    assert 'CLINICAL_CATEGORIES = ("alergia", "condicion_medica")' in src
    # Floor clínico 1.0 + el soft-delete EXCLUYE categorías clínicas.
    assert "salience_score = 1.0" in src
    assert "NOT IN ('alergia','condicion_medica')" in src


def test_dreaming_evidence_fk_verified_anti_confabulation():
    src = _read("dreaming.py")
    # La evidencia del user_model se valida contra user_facts reales del MISMO user_id.
    assert "_verify_evidence_fact_ids" in src
    assert "id::text = ANY(%s) AND user_id = %s AND is_active = TRUE" in src
    # Reflexión sin evidencia válida NO se persiste.
    assert "anti-confabulación" in src or "0 evidencia válida" in src


# ---------------------------------------------------------------------------
# 4. build_memory_context firma + propagación de user_id
# ---------------------------------------------------------------------------
def test_build_memory_context_accepts_user_id():
    src = _read("memory_manager.py")
    assert re.search(
        r"def build_memory_context\(session_id: str, user_id: Optional\[str\] = None\)", src
    ), "build_memory_context debe aceptar user_id (P1-DREAMING-1)"
    assert "_get_user_model_block(user_id)" in src


def test_agent_propagates_user_id_to_memory_context():
    src = _read("agent.py")
    # Ambos callsites (chat_with_agent + _stream) pasan user_id.
    assert "build_memory_context(session_id, user_id)" in src
    assert "build_memory_context(session_id)" not in re.sub(
        r"build_memory_context\(session_id, user_id\)", "", src
    ), "Quedó un callsite legacy build_memory_context(session_id) sin user_id"


# ---------------------------------------------------------------------------
# 5. Knobs presentes + defaults seguros + neutralidad flag-OFF
# ---------------------------------------------------------------------------
def test_dreaming_knobs_present_in_source():
    src = _read("dreaming.py")
    for knob in (
        "MEALFIT_DREAMING_ENABLED", "MEALFIT_DREAMING_RETRIEVAL_ENABLED",
        "MEALFIT_DREAMING_INJECT_PLAN_ENABLED", "MEALFIT_DREAMING_MODEL",
        "MEALFIT_DREAMING_CONSOLIDATION_INTERVAL_HOURS", "MEALFIT_DREAMING_MAX_USERS_PER_NIGHT",
        "MEALFIT_DREAMING_MAX_COST_USD_PER_NIGHT", "MEALFIT_DREAMING_SALIENCE_DECAY_RATE",
        "MEALFIT_DREAMING_MAX_FACTS_PER_CALL", "MEALFIT_DREAMING_BATCH",
    ):
        assert knob in src, f"Knob {knob} ausente en dreaming.py"
    # El knob del cache del user_model vive en el lado de lectura (memory_manager).
    assert "MEALFIT_DREAMING_USER_MODEL_CACHE_TTL_S" in _read("memory_manager.py")


def test_dreaming_flag_off_is_neutral_noop():
    """Smoke funcional: con MEALFIT_DREAMING_ENABLED OFF (default), run_dream_cycle
    es un no-op que NO toca la DB y reporta enabled=False."""
    import os
    os.environ.pop("MEALFIT_DREAMING_ENABLED", None)  # asegura default
    import importlib
    import dreaming
    importlib.reload(dreaming) if False else None  # defaults leídos at call-time
    assert dreaming._dreaming_enabled() is False
    assert dreaming._dreaming_retrieval_enabled() is False
    agg = dreaming.run_dream_cycle()
    assert agg["enabled"] is False
    assert agg["processed"] == 0


def test_plan_injection_wired_fase4():
    """[Fase 4] El user_model se inyecta al generador de planes via
    build_adherence_context (gateado por MEALFIT_DREAMING_INJECT_PLAN_ENABLED)."""
    src_d = _read("dreaming.py")
    assert "def build_plan_constraints_block(" in src_d
    assert "_dreaming_inject_plan_enabled()" in src_d
    src_pg = _read("prompts/plan_generator.py")
    assert "dynamic_user_constraints" in src_pg, (
        "build_adherence_context debe aceptar dynamic_user_constraints (Fase 4)"
    )
    src_go = _read("graph_orchestrator.py")
    assert "build_plan_constraints_block(_uid)" in src_go, (
        "el callsite del prompt de planes debe pasar el user_model consolidado"
    )


def test_marker_is_dreaming():
    src = _read("app.py")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m and m.group(1).startswith("P1-DREAMING-1"), (
        f"_LAST_KNOWN_PFIX debe reflejar el cierre de Dreaming, es {m.group(1) if m else None!r}"
    )
