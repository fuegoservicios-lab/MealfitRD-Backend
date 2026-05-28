"""[P2-AUDIT-IMPL · 2026-05-28] Tests de regresión parser-based del bundle de
implementación de los P2 del audit prod-readiness 2026-05-28 (8 items).

  - P2-1 P2-CHAT-WRITE-IDOR      : ownership pre-check en /message + /stream
  - P2-2 P2-HEALTH-KNOBS-COUNT   : /health/version expone conteo, no knobs_diff
  - P2-3 P2-CONSUMED-DEDUP       : dedup anti doble-tap en log_consumed_meal
  - P2-5 P2-CHUNK-METRICS-RETENTION : cron de retención de plan_chunk_metrics
  - P2-6 P2-READY-DB-CHECK       : /ready valida DB (tolerante, knob)
  - P2-7 P2-CRON-CORRELATION     : correlation_id por ejecución de cron
  - P2-8 P2-MEAL-PLANS-GENSTATUS-IDX : índice funcional generation_status (SSOT dual-dir)

Parser-based (lee source, no importa módulos prod): el venv de test no resuelve
langgraph/supabase. Correr: py -3 -m pytest tests/test_p2_audit_impl.py --noconftest -q
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_BACKEND)


def _read(*parts):
    with open(os.path.join(*parts), "r", encoding="utf-8") as fh:
        return fh.read()


# -------------------------------------------------------------------- P2-1
def test_p2_chat_write_idor_message_and_stream():
    src = _read(_BACKEND, "routers", "chat.py")
    assert src.count("P2-CHAT-WRITE-IDOR") >= 2, "el pre-check debe estar en /message Y /stream"
    assert src.count("get_session_owner(session_id)") >= 2, "ownership pre-check faltante"
    # el pre-check debe rechazar con 403 cuando el dueño no coincide
    assert "_sess_owner != verified_user_id" in src
    assert "status_code=403" in src


# -------------------------------------------------------------------- P2-2
def test_p2_health_version_knobs_count_not_diff():
    src = _read(_BACKEND, "app.py")
    assert '"knobs_overrides_count"' in src, "debe exponer el conteo de overrides"
    assert '"knobs_diff":' not in src, "ya NO debe serializar knobs_diff (nombres+valores) en público"
    assert "P2-HEALTH-KNOBS-COUNT" in src


# -------------------------------------------------------------------- P2-3
def test_p2_consumed_meal_dedup():
    src = _read(_BACKEND, "db_facts.py")
    assert "P2-CONSUMED-DEDUP" in src
    assert "MEALFIT_CONSUMED_MEAL_DEDUP_WINDOW_S" in src
    # debe consultar filas idénticas recientes antes del INSERT
    assert "FROM consumed_meals WHERE user_id" in src and "make_interval(secs" in src


# -------------------------------------------------------------------- P2-5
def test_p2_plan_chunk_metrics_retention():
    src = _read(_BACKEND, "cron_tasks.py")
    assert "def _purge_old_plan_chunk_metrics(" in src, "falta la función de purga"
    assert "MEALFIT_CHUNK_METRICS_RETENTION_DAYS" in src
    assert 'id="purge_old_plan_chunk_metrics"' in src, "el cron debe estar registrado"
    assert "DELETE FROM plan_chunk_metrics" in src


# -------------------------------------------------------------------- P2-6
def test_p2_ready_db_check():
    src = _read(_BACKEND, "app.py")
    assert "P2-READY-DB-CHECK" in src
    assert "MEALFIT_READY_REQUIRE_DB" in src
    # SELECT 1 con timeout corto, tolerante
    assert "SELECT 1" in src
    assert 'connection(timeout=' in src
    # el body de /ready ahora incluye el estado db
    assert '"db": db_ok' in src


# -------------------------------------------------------------------- P2-7
def test_p2_cron_correlation():
    src = _read(_BACKEND, "cron_tasks.py")
    assert "P2-CRON-CORRELATION" in src
    assert "MEALFIT_CRON_CORRELATION_ENABLED" in src
    assert "with_correlation_id(f\"cron:" in src, "el wrapper debe setear corr=cron:<job>:<run>"


# -------------------------------------------------------------------- P2-8
_GENSTATUS_MIG = "p2_meal_plans_generation_status_idx_2026_05_28.sql"


def test_p2_genstatus_idx_migration_dual_dir_identical():
    backend_mig = _read(_BACKEND, "supabase", "migrations", _GENSTATUS_MIG)
    root_mig = _read(_ROOT, "supabase", "migrations", _GENSTATUS_MIG)
    assert backend_mig == root_mig, "SSOT dual-dir: deben ser idénticas"
    assert "CREATE INDEX IF NOT EXISTS idx_meal_plans_generation_status" in backend_mig
    assert "plan_data->>'generation_status'" in backend_mig
    assert "RAISE EXCEPTION" in backend_mig


# -------------------------------------------------------------------- marker
def test_p2_marker_bumped():
    src = _read(_BACKEND, "app.py")
    assert "P2-AUDIT-IMPL" in src, "_LAST_KNOWN_PFIX debe estar bumpeado a P2-AUDIT-IMPL"
