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
    # [P2-CHAT-WRITE-IDOR · 2026-05-30] Umbral subido 2→3: el tercer hermano
    # `POST /api/chat` (api_chat) había omitido el guard. Los TRES write-handlers
    # que llaman save_message con session_id del body deben tener el pre-check.
    assert src.count("P2-CHAT-WRITE-IDOR") >= 3, "el pre-check debe estar en /message, /stream Y POST /api/chat"
    assert src.count("get_session_owner(session_id)") >= 3, "ownership pre-check faltante en alguno de los 3 write-handlers de chat"
    # el pre-check debe rechazar con 403 cuando el dueño no coincide
    assert "_sess_owner != verified_user_id" in src
    assert "status_code=403" in src


def test_p2_chat_write_idor_api_chat_handler_guarded():
    """[P2-CHAT-WRITE-IDOR · 2026-05-30] El handler `api_chat` (POST /api/chat),
    tercer write-handler que invoca save_message con session_id del body, debe
    contener el pre-check de ownership entre su definición y su primer
    save_message — antes solo lo tenían /message y /stream."""
    src = _read(_BACKEND, "routers", "chat.py")
    # Localiza el bloque del handler api_chat y verifica que el guard aparezca
    # antes del save_message del prompt del usuario.
    start = src.index("def api_chat(")
    end = src.index("save_message(session_id, \"user\"", start)
    block = src[start:end]
    assert "get_session_owner(session_id)" in block, (
        "api_chat (POST /api/chat) debe tener el pre-check get_session_owner ANTES "
        "de save_message — sin él, user_id==session_id abre IDOR de escritura."
    )
    assert "_sess_owner != verified_user_id" in block
    assert "status_code=403" in block


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
    backend_mig = _read(_BACKEND, "migrations", _GENSTATUS_MIG)
    root_mig = _read(_ROOT, "migrations", _GENSTATUS_MIG)
    assert backend_mig == root_mig, "SSOT dual-dir: deben ser idénticas"
    assert "CREATE INDEX IF NOT EXISTS idx_meal_plans_generation_status" in backend_mig
    assert "plan_data->>'generation_status'" in backend_mig
    assert "RAISE EXCEPTION" in backend_mig


# -------------------------------------------------------------------- marker
def test_p2_marker_bumped():
    # [P1-PROD-AUDIT-2 · 2026-05-30] Relajado de hardcode `"P2-AUDIT-IMPL"` a un
    # floor de fecha: el marker `_LAST_KNOWN_PFIX` avanza con cada bundle (ya pasó
    # por P1-CHAT-GUEST-IDOR → P1-PROD-AUDIT-2 desde el P2-AUDIT-IMPL del
    # 2026-05-28). Anclar el string exacto del bundle 2026-05-28 convertía este
    # test en un falso-rojo permanente tras cualquier bump posterior. La frescura
    # del marker la valida `test_p3_1_last_known_pfix_freshness.py`; aquí solo
    # exigimos que NO retroceda por debajo del bundle que este archivo cubre.
    import re
    src = _read(_BACKEND, "app.py")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*["\']([^"\']+)["\']', src)
    assert m, "_LAST_KNOWN_PFIX no encontrado en app.py"
    md = re.search(r"(\d{4})-(\d{2})-(\d{2})", m.group(1))
    assert md, f"marker sin fecha parseable: {m.group(1)!r}"
    assert (int(md.group(1)), int(md.group(2)), int(md.group(3))) >= (2026, 5, 28), (
        f"_LAST_KNOWN_PFIX {m.group(1)!r} es anterior al bundle P2-AUDIT-IMPL (2026-05-28)."
    )
