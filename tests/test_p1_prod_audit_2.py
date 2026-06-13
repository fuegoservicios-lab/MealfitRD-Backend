"""[P1-PROD-AUDIT-2 · 2026-05-30] Tests de regresión parser-based del bundle del
audit prod-readiness 2026-05-30 (segunda pasada: workflow 16 dimensiones +
verificación adversaria). Cubre los gaps confirmados (5 P1 + 6 P2 + 8 P3).

Parser-based (lee source, no importa módulos prod): el venv de test no resuelve
langgraph/supabase. Correr: py -3 -m pytest tests/test_p1_prod_audit_2.py --noconftest -q
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BACKEND)


def _read(*parts):
    with open(os.path.join(*parts), "r", encoding="utf-8") as fh:
        return fh.read()


# ─────────────────────────────────────────────── P1 #1: chat IDOR (3er hermano)
def test_p1_chat_api_chat_idor_guard():
    src = _read(_BACKEND, "routers", "chat.py")
    start = src.index("def api_chat(")
    block = src[start: src.index('save_message(session_id, "user"', start)]
    assert "get_session_owner(session_id)" in block
    assert "_sess_owner != verified_user_id" in block and "status_code=403" in block


# ─────────────────────────────────────────────── P1 #2: probe LLM timeout
def test_p1_chunk_probe_llm_timeout():
    src = _read(_BACKEND, "cron_tasks.py")
    assert "def _chunk_probe_llm_timeout_s(" in src
    # [P0-DEEPSEEK-MIGRATION] el constructor del probe usa el modelo FREE
    # via knob (no hardcode) y debe llevar timeout=
    probe_at = src.index("probe_llm = ChatDeepSeek(")
    block = src[probe_at: probe_at + 600]
    assert "model=model_free_tier()" in block
    assert "timeout=_chunk_probe_llm_timeout_s()" in block


# ─────────────────────────────────────────────── P2 #6: embeddings + sentiment timeouts
def test_p2_embeddings_client_args_timeout():
    # [P0-DEEPSEEK-MIGRATION · 2026-06-12] El deadline de embeddings se
    # consolidó en embeddings_provider (un solo constructor, kwarg nativo
    # `timeout=` de OpenAIEmbeddings) — misma lección P2-LLM-TIMEOUT-SWEEP.
    src = _read(_BACKEND, "embeddings_provider.py")
    assert "def _embeddings_timeout_s(" in src, "helper falta en embeddings_provider"
    assert '"timeout": _embeddings_timeout_s()' in src, (
        "el constructor de embeddings debe acotar deadline via timeout="
    )


def test_p2_sentiment_timeout():
    src = _read(_BACKEND, "sentiment_classifier.py")
    assert "def _sentiment_llm_timeout_s(" in src
    assert "timeout=_sentiment_llm_timeout_s()" in src
    assert "MEALFIT_SENTIMENT_MODEL" in src  # modelo a knob


# ─────────────────────────────────────────────── P1 #3 + P2 #9/#10: billing webhook
def test_p1_billing_reactivate_not_cancelled():
    src = _read(_BACKEND, "routers", "billing.py")
    assert "P1-BILLING-REACTIVATE-NOT-CANCELLED" in src
    assert '.neq(\n                            "subscription_status", "CANCELLED"\n                        )' in src \
        or '.neq("subscription_status", "CANCELLED")' in src
    assert 'q.eq("subscription_status", "PAYMENT_RETRYING")' in src


def test_p2_billing_webhook_infra_503():
    src = _read(_BACKEND, "routers", "billing.py")
    # el except HTTPException: raise debe preceder al except Exception del webhook
    assert "except HTTPException:\n        # [P2-WEBHOOK-INFRA-503" in src
    # OAuth !=200 ya no retorna 200 silencioso
    assert "raise HTTPException(status_code=503, detail=\"PayPal auth transient failure; retry.\")" in src
    assert "Webhook processing transient error; retry." in src


def test_p2_billing_webhook_idempotency():
    src = _read(_BACKEND, "routers", "billing.py")
    assert "P2-WEBHOOK-IDEMPOTENCY" in src
    assert "paypal_webhook:" in src
    assert "ON CONFLICT (key) DO NOTHING RETURNING key" in src
    # el prefijo está en el sweep de retención KV
    cron = _read(_BACKEND, "cron_tasks.py")
    assert '"prefix": "paypal_webhook:"' in cron


# ─────────────────────────────────────────────── P1 #4: diary upload rate-limiter
def test_p1_diary_upload_ratelimiter():
    src = _read(_BACKEND, "routers", "diary.py")
    assert "_VISION_UPLOAD_LIMITER = RateLimiter(" in src
    assert "Depends(_VISION_UPLOAD_LIMITER)" in src


# ─────────────────────────────────────────────── P2 #7: alert resolvers
def test_p2_alert_resolvers():
    rl = _read(_BACKEND, "rate_limiter.py")
    assert "def _resolve_rl_saturation_alert(" in rl
    assert "_resolve_rl_saturation_alert()" in rl  # invocado en cleanup tick
    cron = _read(_BACKEND, "cron_tasks.py")
    assert "bg_task_timeout:%%" in cron and "MEALFIT_BG_TASK_TIMEOUT_ALERT_TTL_H" in cron
    # ambas keys documentadas en la tabla canónica
    doc = _read(_BACKEND, "docs", "system_alerts_resolution_table.md")
    assert "`bg_task_timeout:<task_name>`" in doc
    assert "`rate_limiter_bucket_saturation`" in doc


# ─────────────────────────────────────────────── P2 #8: pipeline_metrics index (SSOT dual-dir)
def test_p2_pipeline_metrics_index_migration_ssot():
    fname = "p2_pipeline_metrics_user_id_idx_2026_05_30.sql"
    root_mig = os.path.join(_ROOT, "supabase", "migrations", fname)
    backend_mig = os.path.join(_BACKEND, "supabase", "migrations", fname)
    assert os.path.exists(root_mig), "migración falta en supabase/migrations/ (workspace root)"
    assert os.path.exists(backend_mig), "migración falta en backend/supabase/migrations/"
    a = _read(root_mig)
    b = _read(backend_mig)
    assert a == b, "las dos copias de la migración deben ser idénticas (P3-MIGRATIONS-SSOT)"
    assert "idx_pipeline_metrics_user_id_created" in a
    assert "WHERE user_id IS NOT NULL" in a
    assert "CREATE INDEX IF NOT EXISTS" in a  # idempotente


# ─────────────────────────────────────────────── P1 #5: borrado de cuenta / PII
def test_p1_delete_account_data():
    src = _read(_BACKEND, "db_profiles.py")
    assert "def delete_account_data(" in src
    assert "def _purge_visual_diary_storage(" in src
    assert "_USER_SCOPED_TABLES_USERID" in src
    # checkpoints LangGraph purgados por thread_id
    assert "FROM agent_sessions WHERE user_id = %s" in src
    # endpoint admin gateado
    sys_src = _read(_BACKEND, "routers", "system.py")
    assert '"/admin/account/purge-data"' in sys_src
    assert "_verify_admin_token" in sys_src
    assert "delete_account_data" in sys_src


def test_p2_reset_clears_visual_diary():
    src = _read(_BACKEND, "db_profiles.py")
    reset_at = src.index("def reset_user_account_preferences(")
    end = src.index("def delete_account_data(", reset_at)
    block = src[reset_at:end]
    assert "DELETE FROM visual_diary WHERE user_id = %s" in block
    assert "_purge_visual_diary_storage" in block


# ─────────────────────────────────────────────── P2 #11: ThreadPoolExecutor antipattern
def test_p2_inventory_fetch_executor_no_with():
    src = _read(_BACKEND, "cron_tasks.py")
    assert "_inv_exec.shutdown(wait=False, cancel_futures=True)" in src


# ─────────────────────────────────────────────── P3 #17/#18: concurrencia
def test_p3_inventory_insert_race_documented_accepted():
    # El upsert ON CONFLICT se DIFIRIÓ (race narrow sin corrupción — la unique
    # constraint previene duplicados; reescribir el mock-based test cuesta más que
    # el beneficio). Aquí solo anclamos que el comentario engañoso fue corregido.
    src = _read(_BACKEND, "db_inventory.py")
    assert 'El comentario legacy "El INSERT no tiene race"' in src


def test_p3_visual_diary_atomic_increment():
    src = _read(_BACKEND, "db_facts.py")
    assert "SET frequency = frequency + 1" in src


# ─────────────────────────────────────────────── P3 #15/#16: caps de tamaño
def test_p3_size_caps():
    src = _read(_BACKEND, "routers", "plans.py")
    assert "MEALFIT_RESTOCK_MAX_ITEMS" in src
    assert "MEALFIT_MAX_PLAN_DATA_BYTES" in src
    assert src.count("MEALFIT_MAX_PLAN_DATA_BYTES") >= 2  # restore-local + swap-meal


# ─────────────────────────────────────────────── P3 #19/#20: startup/shutdown
def test_p3_app_pool_close_and_connect_timeout():
    src = _read(_BACKEND, "app.py")
    assert "chat_checkpoint_pool.close()" in src
    assert src.count("connect_timeout=5") >= 2  # leader lock + PostgresSaver setup


# ─────────────────────────────────────────────── P3 #14: chat-tool persist alert
def test_p3_chat_tool_persist_alert():
    src = _read(_BACKEND, "tools.py")
    assert src.count("_persist_plan_persist_failed_alert") >= 2  # else + except


# ─────────────────────────────────────────────── marker bump
def test_marker_bumped():
    # [P1-PROD-AUDIT-3 · 2026-05-30] Relajado a floor-fecha: cada audit posterior
    # bumpea `_LAST_KNOWN_PFIX`, y el assert exact-match `P1-PROD-AUDIT-2` rompía en
    # el siguiente bump (P1-PROD-AUDIT-3). El contrato real (marker fresco + formato)
    # lo ancla test_p3_1_last_known_pfix_freshness; aquí solo exigimos que el marker
    # NO retroceda antes de esta pasada (2026-05-30).
    import re as _re
    src = _read(_BACKEND, "app.py")
    m = _re.search(r'_LAST_KNOWN_PFIX\s*=\s*"[^"]*(\d{4})-(\d{2})-(\d{2})"', src)
    assert m, "No se encontró _LAST_KNOWN_PFIX con fecha parseable."
    assert (int(m.group(1)), int(m.group(2)), int(m.group(3))) >= (2026, 5, 30), (
        "_LAST_KNOWN_PFIX retrocedió antes de 2026-05-30 (audit PROD-AUDIT-2)."
    )
