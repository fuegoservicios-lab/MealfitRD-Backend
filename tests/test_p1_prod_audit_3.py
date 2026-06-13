"""[P1-PROD-AUDIT-3 · 2026-05-30] Regresión del bundle del 3er audit backend de
production-readiness (workflow 12 dimensiones + verificación adversaria; 0 P0,
5 P1, 6 P2, 13 P3). Cubre los gaps IMPLEMENTADOS; 3 quedaron DEFERIDOS con
rationale documentado (ver test_deferred_items_documented al final).

Mayormente parser-based (el venv de test no resuelve langgraph/supabase). El
único funcional importa cron_tasks (como test_p3_chunk_deferrals_fk_discard).

Correr: py -3 -m pytest tests/test_p1_prod_audit_3.py --noconftest -q
"""
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(*parts):
    with open(os.path.join(_BACKEND, *parts), "r", encoding="utf-8") as fh:
        return fh.read()


def _fn_body(src: str, def_line: str, end_marker: str | None = None) -> str:
    start = src.index(def_line)
    if end_marker:
        return src[start: src.index(end_marker, start)]
    nxt = src.find("\ndef ", start + 1)
    return src[start: nxt if nxt != -1 else len(src)]


# ════════════════════════════════════════════════════════════════════ P1 #1
def test_p1_diary_upload_idor_no_session_id_shortcircuit():
    """El guard IDOR de /upload NO debe tener el short-circuit `user_id != session_id`
    (permitía cross-user write con user_id==session_id==victim). Se chequea el
    fragmento EXACTO del guard viejo (no aparece en comentarios)."""
    src = _read("routers", "diary.py")
    body = _fn_body(src, "async def api_diary_upload(", end_marker="declared_ct =")
    assert 'user_id != "guest" and user_id != session_id' not in body, (
        "Regresión IDOR P1-PROD-AUDIT-3: volvió el short-circuit `user_id != session_id` "
        "que salteaba el ownership check."
    )
    assert 'if user_id and user_id != "guest":' in body
    assert "verified_user_id != user_id" in body and "status_code=401" in body


def test_p1_diary_upload_uuid_validation():
    """P2: /upload valida UUID de user_id (y session_id si presente) — paridad con
    /consumed GET + cierra inyección de path en la Storage key."""
    src = _read("routers", "diary.py")
    body = _fn_body(src, "async def api_diary_upload(", end_marker="declared_ct =")
    assert "assert_valid_uuid(user_id, allow_guest=True)" in body
    assert "assert_valid_uuid(session_id, allow_guest=True)" in body


# ════════════════════════════════════════════════════════════════════ P1 #2
def test_p1_diary_consumed_fail_loud():
    """/consumed debe capturar el return de log_consumed_meal y fallar loud (500)
    en None/falsy, y NO programar learning sobre data fantasma."""
    src = _read("routers", "diary.py")
    body = _fn_body(src, "def api_log_consumed_meal(", end_marker="@router.get")
    assert "_logged_ok = log_consumed_meal(" in body
    assert "if not _logged_ok:" in body
    assert "status_code=500" in body
    # El add_task de learning ocurre DESPUÉS del check de éxito (no sobre fantasma).
    assert body.index("if not _logged_ok:") < body.index("background_tasks.add_task(trigger_incremental_learning")


# ════════════════════════════════════════════════════════════════════ P1 #3
def test_p1_billing_cancelled_revoker_no_end_date():
    """El revocador SSOT (graceful degradation) debe degradar CANCELLED incluso sin
    subscription_end_date (PayPal omite next_billing_time → NULL → acceso perpetuo)."""
    src = _read("db_profiles.py")
    body = _fn_body(src, "def get_user_profile(", end_marker="def _invalidate_stale_chunks")
    # Ya NO exige `subscription_end_date` truthy en la condición de entrada.
    assert 'if profile.get("subscription_status") == "CANCELLED":' in body
    assert 'profile.get("subscription_status") == "CANCELLED" and profile.get("subscription_end_date")' not in body
    # Rama sin end_date degrada (fail-secure).
    assert "should_downgrade = True" in body
    assert '"plan_tier": "gratis"' in body and '"subscription_status": "INACTIVE"' in body


# ════════════════════════════════════════════════════════════════════ P1 #4 (live)
def test_p1_lesson_flush_classifier_treats_fk_and_check_as_terminal():
    """El clasificador de descarte de _flush_pending_lesson_telemetry debe tratar FK
    (incidente vivo) y CHECK como terminales — paridad con el de deferrals."""
    src = _read("cron_tasks.py")
    body = _fn_body(src, "def _flush_pending_lesson_telemetry(")
    assert '"violates foreign key" in _err_msg' in body, (
        "Incidente vivo P1: el flush de lesson_telemetry no descarta FK violations → "
        "re-INSERT forever de records de planes borrados (burst de ERRORs Postgres)."
    )
    assert '"violates check constraint" in _err_msg' in body
    assert '"violates not-null" in _err_msg' in body  # regresión guard
    assert '"invalid input syntax" in _err_msg' in body


# NOTA: la cobertura FUNCIONAL del FK-discard (mock execute_sql_write con FK error →
# discarded_invalid++, buffer vacío; y transient → preservado) vive en
# test_p3_chunk_deferrals_fk_discard.py para el flush HERMANO de deferrals — código
# idéntico y mirroreado. Importar cron_tasks requiere el venv full-deps (langgraph/
# supabase); en el venv parser-only de CI estos imports no resuelven, así que la
# regresión del clasificador del lesson flush se ancla parser-based arriba.


# ════════════════════════════════════════════════════════════════════ P2 timeouts
def test_p2_shopping_calculator_embeddings_timeout():
    # [P0-DEEPSEEK-MIGRATION · 2026-06-12] shopping_calculator ya no
    # construye su propio cliente de embeddings: delega a la capa pluggable
    # `embeddings_provider.get_embeddings_client()`, cuyo constructor acota
    # el deadline (`"timeout": _embeddings_timeout_s()`).
    src = _read("shopping_calculator.py")
    assert "from embeddings_provider import get_embeddings_client" in src
    provider_src = _read("embeddings_provider.py")
    assert "def _embeddings_timeout_s(" in provider_src
    assert '"timeout": _embeddings_timeout_s()' in provider_src
    assert "P2-LLM-TIMEOUT-SWEEP" in provider_src


def test_p2_tools_medical_clinical_llm_timeout():
    src = _read("tools_medical.py")
    assert "def _medical_tool_llm_timeout_s(" in src
    assert "timeout=_medical_tool_llm_timeout_s()" in src
    assert "P2-LLM-TIMEOUT-SWEEP" in src


def test_p2_chat_tts_db_writes_offloaded():
    """Los INSERTs sync del handler async de TTS deben ir por asyncio.to_thread."""
    src = _read("routers", "chat.py")
    body = _fn_body(src, "async def api_chat_tts(", end_marker="@router.post(\"/feedback\")")
    assert 'await asyncio.to_thread(log_api_usage, verified_user_id, "elevenlabs_tts")' in body
    assert "await asyncio.to_thread(\n                execute_sql_write," in body


# ════════════════════════════════════════════════════════════════════ P2 ENV normalize
def test_p2_is_production_helper_ssot():
    knobs_src = _read("knobs.py")
    assert "def is_production() -> bool:" in knobs_src
    assert '.strip().lower() == "production"' in knobs_src

    billing_src = _read("routers", "billing.py")
    assert "from knobs import" in billing_src and "is_production" in billing_src
    assert "is_sandbox = not is_production()" in billing_src
    assert 'os.environ.get("ENVIRONMENT") != "production"' not in billing_src, (
        "Regresión: quedó un is_sandbox con exact-match crudo en billing.py."
    )

    app_src = _read("app.py")
    assert "_IS_PRODUCTION = is_production()" in app_src
    assert '_IS_PRODUCTION = os.environ.get("ENVIRONMENT") == "production"' not in app_src


# ════════════════════════════════════════════════════════════════════ P3
def test_p3_fact_lock_fail_closed():
    src = _read("db_facts.py")
    body = _fn_body(src, "def acquire_fact_lock(", end_marker="def release_fact_lock")
    # La rama de excepción ahora retorna False (fail-closed). El comentario menciona
    # `return True` como descripción del bug previo → se chequea el statement REAL:
    # la función termina en `return False`.
    tail = body[body.index("except Exception as e:"):]
    assert tail.rstrip().endswith("return False"), (
        "Regresión: acquire_fact_lock debe fail-CLOSED (return False) en la rama de "
        "excepción, no fail-open (return True)."
    )


def test_p3_billing_verify_checks_row_matched():
    # [P1-NEON-DB-MIGRATION · 2026-06-12] Re-anclado: el check de filas
    # matcheadas pasó de `if not getattr(res, "data", None):` (PostgREST) a
    # `if not updated_rows:` sobre el resultado del UPDATE con
    # `RETURNING id` + `returning=True`. Misma propiedad fail-loud: si el
    # UPDATE de /verify matcheó 0 filas tras el cobro PayPal → alert + 500.
    src = _read("routers", "billing.py")
    assert "RETURNING id" in src
    assert "returning=True" in src
    assert "if not updated_rows:" in src
    # El check ocurre DESPUÉS del UPDATE con RETURNING (mismo flujo).
    assert src.index("RETURNING id") < src.index("if not updated_rows:")
    assert "billing_profile_not_found_on_upgrade" in src


def test_p3_per_user_semaphore_gc():
    src = _read("graph_orchestrator.py")
    assert 'LLM_PER_USER_LOCAL_CACHE_MAX = _env_int(' in src
    body = _fn_body(src, "def _get_local_sync(")
    assert "LLM_PER_USER_LOCAL_CACHE_MAX" in body
    assert 'getattr(_s, "_value", 0) >= self.max_per_user' in body


def test_p3_failed_deduction_prunes_succeeded_items():
    """El UPDATE de fallo parcial debe reescribir `ingredients` con `still_failing`
    (poda items ya deducidos → no re-deducción no-idempotente)."""
    src = _read("cron_tasks.py")
    fn = _fn_body(src, "def _process_failed_inventory_deductions_queue(")
    assert "still_failing = [" in fn
    assert "succeeded_idx" in fn
    assert "SET attempts = %s, ingredients = %s::jsonb" in fn
    # El test parser previo exige `SET attempts` — el orden lo preserva.


def test_p3_inventory_consume_cap():
    src = _read("routers", "plans.py")
    body = _fn_body(src, "def api_consume_inventory(")
    assert 'MEALFIT_CONSUME_MAX_ITEMS' in body
    assert "demasiado grande" in body


def test_p3_deferrals_backlog_tracker_and_doc():
    src = _read("cron_tasks.py")
    assert "def _track_deferrals_backlog(" in src
    assert 'alert_key="deferrals_flush_backlog"' in src
    # Llamado en ambos returns de _flush_pending_deferrals.
    fn = _fn_body(src, "def _flush_pending_deferrals(")
    assert fn.count("_track_deferrals_backlog(stats)") >= 2
    doc = _read("docs", "system_alerts_resolution_table.md")
    assert "deferrals_flush_backlog" in doc


def test_p3_partial_no_shopping_stranded_alert():
    src = _read("cron_tasks.py")
    body = _fn_body(src, "def _alert_stranded_partial_plans(")
    assert "partial_no_shopping" in body
    assert "gen_status" in body


def test_p3_heartbeat_comment_corrected():
    src = _read("cron_tasks.py")
    # El comentario sobre-optimista fue corregido a LOG-ONLY.
    idx = src.index("_chunk_heartbeat_start_failures: dict = {")
    pre = src[idx - 700: idx]
    assert "LOG-ONLY" in pre
    assert "NO machine-readable" in pre


def test_p3_chunk_pipeline_timeout_clamped():
    src = _read("constants.py")
    assert 'CHUNK_PIPELINE_TIMEOUT_SECONDS = _env_int(' in src
    assert "validator=lambda v: 30 <= v <= 1800" in src


def test_p3_memory_summary_model_knob_registered():
    src = _read("memory_manager.py")
    assert "_env_str" in src
    assert "MEMORY_SUMMARY_MODEL" in src
    # Ya NO es un os.environ.get crudo para el modelo.
    assert 'MEMORY_SUMMARY_MODEL = os.environ.get(' not in src


# ════════════════════════════════════════════════════════════════════ Deferidos
def test_deferred_items_documented():
    """3 findings DEFERIDOS con rationale (no son olvidos): reservation CAS (P2,
    bounded/self-heal, análogo al upsert revertido en PROD-AUDIT-2), PAYMENT_RETRYING
    expiry (P3, requiere migración con hazard de orden de deploy), y _persist_billing_alert
    async (P3, helper sync usado por tests, path raro). Documentados en la memoria
    project_prod_audit_3_2026_05_30.md. Este test es un marcador de intención."""
    # Sanity: las funciones deferidas siguen existiendo (no se tocaron a medias).
    inv = _read("db_inventory.py")
    assert "def release_chunk_reservations(" in inv
    assert "def _consume_reserved_inventory(" in inv
