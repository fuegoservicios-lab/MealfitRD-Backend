"""[P1-ARQ25-F1-LIFECYCLE · 2026-09-02] Fase 1 del roadmap 2.5 «Núcleo único».

Ancla las invariantes nuevas del lifecycle (roadmap §3.2/§3.3) sin DB:

  I9   idempotencia   — `create_or_replay_run`: crea / reproduce / 409 por fingerprint.
  I10  fencing        — `attempts` es el token: el CAS del worker y el del chunk 0 lo usan;
                        un UPDATE de 0 filas NO completa el run.
  I12  revisión       — la migración crea `meal_plans.revision` + trigger `IS DISTINCT FROM`,
                        idéntica en los dos directorios SSOT.
  I19  autoridad única— el `_chunk_worker` tiene la rama `initial` ANTES del guard legacy y
                        el camino SSE legacy no crece (un solo `create_task(run_pipeline())`).
  H5   disponibilidad — `derive_availability` jamás da PLAN_READY con `days=[]`.
  H1   gate de gasto  — el pickup conserva el token `__PLAN_MODE_GATE__` (ya anclado por
                        test_p1_plan_mode; aquí se comprueba que el chunk 0 entra POR ESE pickup).
  Drain              — `_drain_aware` no arranca ticks con drain pedido; `request_worker_drain`
                        vuelve cuando no hay ticks en vuelo.

Los tests que importan `cron_tasks` se saltan si el entorno no puede importarlo
(sin deps del LLM): las anclas parser-based cubren ese caso.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

BACKEND = Path(__file__).resolve().parent.parent
MIGRATION_NAME = "arq25_f1_lifecycle_expand_2026_09_02.sql"


def _src(rel: str) -> str:
    return (BACKEND / rel).read_text(encoding="utf-8")


def _sql_without_comments(sql: str) -> str:
    return "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))


def _import_or_skip(modname: str):
    """`pytest.importorskip` sólo salta en ImportError; `cron_tasks` puede fallar con
    TypeError (constructor del provider LLM sin deps exactas). Aquí se salta con cualquiera."""
    try:
        return __import__(modname, fromlist=["_"])
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{modname} no importable en este entorno: {type(e).__name__}: {str(e)[:80]}")


# ---------------------------------------------------------------------------
# Migración (I12 + expand)
# ---------------------------------------------------------------------------

def test_migration_exists_and_is_idempotent_and_has_revision_trigger():
    sql = _src(f"migrations/{MIGRATION_NAME}")
    assert "CREATE TABLE IF NOT EXISTS public.plan_generation_runs" in sql
    assert "CREATE TABLE IF NOT EXISTS public.plan_jobs" in sql
    assert "ADD COLUMN IF NOT EXISTS revision INTEGER NOT NULL DEFAULT 1" in sql
    assert "ADD COLUMN IF NOT EXISTS run_id" in sql
    assert "ADD COLUMN IF NOT EXISTS claimed_by" in sql
    assert "ADD COLUMN IF NOT EXISTS input_hash" in sql
    assert "ADD COLUMN IF NOT EXISTS output_hash" in sql
    # I12 por trigger, sólo cuando plan_data cambia de verdad
    assert "BEFORE UPDATE OF plan_data ON public.meal_plans" in sql
    assert "NEW.plan_data IS DISTINCT FROM OLD.plan_data" in sql
    assert "DROP TRIGGER IF EXISTS meal_plans_bump_revision_trg" in sql
    # Patrón P3-NEW-2 para functions
    assert "SET search_path = ''" in sql
    # I9: la clave es única por usuario
    assert "plan_generation_runs_user_idem_uq" in sql and "(user_id, idempotency_key)" in sql
    # No se añade lease_token: attempts sigue siendo el token (I10). Se mira el SQL sin comentarios.
    assert "lease_token" not in _sql_without_comments(sql)
    assert "RAISE EXCEPTION" in sql  # sanity DO $$


def test_migration_is_byte_identical_in_root_ssot_dir_when_present():
    root = BACKEND.parent / "migrations" / MIGRATION_NAME
    if not root.exists():
        pytest.skip("workspace-root migrations/ no visible desde este checkout (worktree aislado)")
    assert root.read_bytes() == (BACKEND / "migrations" / MIGRATION_NAME).read_bytes()


# ---------------------------------------------------------------------------
# Estados derivados (H5 / §5.2) — puros
# ---------------------------------------------------------------------------

def test_derive_availability_never_plan_ready_without_days():
    from generation_lifecycle import AVAIL_NONE, AVAIL_PLAN_READY, AVAIL_PREVIEW_READY, derive_availability
    assert derive_availability({"days": [], "generation_status": "complete"}, 7, preview_days=3) == AVAIL_NONE
    assert derive_availability({"days": [{}, {}, {}]}, 7, preview_days=3) == AVAIL_PREVIEW_READY
    assert derive_availability({"days": [{}, {}]}, 7, preview_days=3) == AVAIL_NONE
    assert derive_availability({"days": [{}] * 7}, 7, preview_days=3) == AVAIL_PLAN_READY
    assert derive_availability({"days": [{}] * 4, "generation_status": "complete_partial"}, 7, preview_days=3) == AVAIL_PLAN_READY
    assert derive_availability(None, 7, preview_days=3) == AVAIL_NONE


def test_derive_run_status_precedence():
    from generation_lifecycle import (
        RUN_CANCELLED, RUN_COMPLETED, RUN_FAILED, RUN_PAUSED, RUN_PENDING, RUN_RUNNING,
        RUN_WAITING_RETRY, RUN_WAITING_USER, derive_run_status,
    )
    future = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    base_run = {"requested_days": 7}
    # placeholder recién encolado, chunk pending vencido → RUNNING (alguien lo va a tomar)
    assert derive_run_status(plan_data={"days": [], "generation_status": "generating"},
                             chunk_rows=[{"status": "pending", "execute_after": None}], run_row=base_run) == RUN_RUNNING
    assert derive_run_status(plan_data={"days": []}, chunk_rows=[{"status": "processing"}], run_row=base_run) == RUN_RUNNING
    assert derive_run_status(plan_data={"days": [{}] * 3}, chunk_rows=[{"status": "pending", "execute_after": future}], run_row=base_run) == RUN_WAITING_RETRY
    assert derive_run_status(plan_data={"days": [{}] * 3}, chunk_rows=[{"status": "failed", "dead_lettered_at": None}], run_row=base_run) == RUN_WAITING_RETRY
    assert derive_run_status(plan_data={"days": [{}] * 3}, chunk_rows=[{"status": "pending_user_action"}], run_row=base_run) == RUN_WAITING_USER
    assert derive_run_status(plan_data={"days": [{}] * 3}, chunk_rows=[{"status": "pending"}], run_row=base_run, plan_mode="tracking") == RUN_PAUSED
    assert derive_run_status(plan_data={"days": []}, chunk_rows=[{"status": "failed", "dead_lettered_at": "x"}], run_row=base_run) == RUN_FAILED
    assert derive_run_status(plan_data={"days": [{}] * 7, "generation_status": "complete"}, chunk_rows=[{"status": "completed"}], run_row=base_run) == RUN_COMPLETED
    assert derive_run_status(plan_data={"days": [{}] * 3}, chunk_rows=[{"status": "cancelled"}], run_row={**base_run, "cancel_requested_at": "x"}) == RUN_CANCELLED
    assert derive_run_status(plan_data={"days": [], "generation_status": "generating"}, chunk_rows=[], run_row=base_run) == RUN_PENDING


# ---------------------------------------------------------------------------
# I9 — idempotencia
# ---------------------------------------------------------------------------

def test_request_fingerprint_ignores_volatile_and_internal_keys():
    from generation_lifecycle import request_fingerprint
    a = {"weight": 70, "totalDays": 7, "idempotency_key": "k1", "tzOffset": -240, "_internal": 1}
    b = {"totalDays": 7, "weight": 70, "idempotency_key": "k2", "tzOffset": 0, "_other": 2}
    assert request_fingerprint(a) == request_fingerprint(b)
    assert request_fingerprint({**a, "weight": 71}) != request_fingerprint(a)


def test_create_or_replay_run_created_replayed_conflict():
    import generation_lifecycle as gl
    row = {"id": "run-1", "user_id": "u1", "plan_id": None, "idempotency_key": "k", "request_fingerprint": "fp",
           "requested_days": 7, "cancel_requested_at": None, "created_at": None, "completed_at": None}
    with patch("db_core.execute_sql_write", return_value=[row]) as w, patch("db_core.execute_sql_query") as q:
        run, created = gl.create_or_replay_run(user_id="u1", idempotency_key="k", fingerprint="fp",
                                               requested_days=7, market_country="DO", locale="es-DO", input_snapshot={})
    assert created is True and run["id"] == "run-1"
    assert "ON CONFLICT (user_id, idempotency_key) DO NOTHING" in w.call_args[0][0]
    q.assert_not_called()

    with patch("db_core.execute_sql_write", return_value=[]), patch("db_core.execute_sql_query", return_value=row):
        run, created = gl.create_or_replay_run(user_id="u1", idempotency_key="k", fingerprint="fp",
                                               requested_days=7, market_country=None, locale=None, input_snapshot={})
    assert created is False and run["id"] == "run-1"

    with patch("db_core.execute_sql_write", return_value=[]), patch("db_core.execute_sql_query", return_value=row):
        with pytest.raises(gl.RunFingerprintConflict) as ei:
            gl.create_or_replay_run(user_id="u1", idempotency_key="k", fingerprint="OTRO",
                                    requested_days=7, market_country=None, locale=None, input_snapshot={})
    assert ei.value.run_id == "run-1"


# ---------------------------------------------------------------------------
# I10 — fencing: attempts es el token
# ---------------------------------------------------------------------------

def test_fencing_token_is_attempts_and_worker_commit_uses_it():
    from generation_lifecycle import FENCING_TOKEN_COLUMN
    assert FENCING_TOKEN_COLUMN == "attempts"
    src = _src("cron_tasks.py")
    # el commit T2 del worker y el helper CAS
    assert "WHERE id = %s AND attempts = %s AND status = 'processing'" in src
    assert "WHERE id = %s AND attempts = %s AND status = %s" in src  # _cas_update_chunk_status
    # el chunk 0 entra por el mismo helper (busca la llamada en generation_lifecycle)
    gl = _src("generation_lifecycle.py")
    assert "_cas_update_chunk_status(" in gl
    assert 'task_id, int(pickup_attempts), "completed"' in gl


def test_run_initial_chunk_displaced_worker_does_not_complete_run(monkeypatch):
    """Dos reclamos del mismo chunk: el worker viejo (attempts stale) hace un CAS de 0 filas
    y NO marca el run completado ni publica `complete`."""
    import types
    import sys
    gl = _import_or_skip("generation_lifecycle")

    # cron_tasks puede no importar sin deps LLM: inyectamos un stub mínimo
    stub = types.ModuleType("cron_tasks")
    calls = {"cas": []}
    def _cas(task_id, expected_attempts, new_status, expected_status="processing", extra_set_clauses=None):
        calls["cas"].append((task_id, expected_attempts, new_status))
        return False  # desplazado
    stub._cas_update_chunk_status = _cas
    monkeypatch.setitem(sys.modules, "cron_tasks", stub)

    go = types.ModuleType("graph_orchestrator")
    go.run_plan_pipeline = lambda *a, **k: {"days": [{"meals": [{"name": "x"}]}]}
    monkeypatch.setitem(sys.modules, "graph_orchestrator", go)

    rp = types.ModuleType("routers.plans")
    rp._resolve_live_pantry = lambda uid, data: []
    rp._run_pantry_validation_for_initial_chunk = lambda **kw: kw["result"]
    rp._postprocess_pipeline_result = lambda **kw: {**kw["result"], "id": kw["existing_plan_id"]}
    routers_pkg = types.ModuleType("routers")
    routers_pkg.plans = rp
    monkeypatch.setitem(sys.modules, "routers", routers_pkg)
    monkeypatch.setitem(sys.modules, "routers.plans", rp)

    completed = {"n": 0}
    monkeypatch.setattr(gl, "mark_run_completed", lambda rid: completed.__setitem__("n", completed["n"] + 1))
    monkeypatch.setattr(gl, "clear_run_error", lambda rid: None)
    monkeypatch.setattr(gl, "run_cancel_requested", lambda rid: False)
    monkeypatch.setattr(gl, "RunProgressPublisher", lambda rid: (lambda ev: None))
    monkeypatch.setattr("db_core.execute_sql_write", lambda *a, **k: True)

    task = {"id": "chunk-0", "user_id": "u1", "meal_plan_id": "plan-1", "attempts": 1}
    snap = {"_run_id": "run-1", "raw_data": {"totalDays": 3}, "use_chunking": False, "totalDays": 3}
    gl.run_initial_chunk(task=task, snap=snap, form_data={"_plan_start_date": "2026-09-02T04:00:00+00:00"}, pickup_attempts=1)

    assert calls["cas"] == [("chunk-0", 1, "completed")]
    assert completed["n"] == 0, "un worker desplazado no puede completar el run"


# ---------------------------------------------------------------------------
# I19 — autoridad única / rama initial en el worker
# ---------------------------------------------------------------------------

def test_worker_has_initial_branch_before_legacy_guard():
    src = _src("cron_tasks.py")
    branch = src.find('if str(chunk_kind or "") == "initial":')
    # El comentario `[GAP 3 FIX` queda pegado al `try:` (ancla de otro test); lo que importa
    # es que la rama va ANTES del SELECT del guard legacy y DENTRO del mismo try.
    guard_comment = src.find("# [GAP 3 FIX: GUARD validar plan activo y no-fallido]")
    guard_sql = src.find("active_plan = execute_sql_query(", guard_comment)
    assert 0 < guard_comment < branch < guard_sql, "la rama initial va entre el `try:` y el guard legacy"
    assert "from generation_lifecycle import run_initial_chunk" in src[branch:guard_sql]
    main_try = src.rfind("\n        try:\n", 0, guard_comment)
    assert main_try > 0 and src[main_try:guard_comment].strip() == "try:", "la rama vive dentro del try del lock release"
    # el pickup sigue gateado (H1): el chunk 0 pasa por el mismo UPDATE ... __PLAN_MODE_GATE__
    assert src.count("__PLAN_MODE_GATE__") >= 2


def test_legacy_sse_path_does_not_grow_i19():
    src = _src("routers/plans.py")
    # exactamente UNA creación de la tarea legacy hasta que la Fase 9 la retire
    assert src.count("asyncio.create_task(run_pipeline())") == 1
    # ningún archivo nuevo de la Fase 1 escribe generation_status fuera del placeholder/dead-letter
    gl = _src("generation_lifecycle.py")
    assert gl.count("generation_status") <= 12


def test_postprocess_and_persist_accept_existing_plan_id():
    assert "existing_plan_id: Optional[str] = None" in _src("routers/plans.py")
    sv = _src("services.py")
    assert sv.count("existing_plan_id") >= 6
    dp = _src("db_plans.py")
    fn = dp[dp.find("def fill_placeholder_meal_plan_atomic"):dp.find("def save_new_meal_plan_robust")]
    assert "FOR UPDATE" in fn and "acquire_meal_plan_advisory_lock(cursor, plan_id" in fn  # I7
    assert "AND user_id = %s" in fn  # I2
    assert '!= "generating"' in fn  # sólo rellena placeholders


# ---------------------------------------------------------------------------
# Drain + wake
# ---------------------------------------------------------------------------

def test_drain_aware_skips_tick_and_wait_returns_when_idle():
    ct = _import_or_skip("cron_tasks")
    ran = {"n": 0}
    @ct._drain_aware
    def fake_tick(target_plan_id=None):
        ran["n"] += 1
        return "ok"
    ct._DRAIN_EVENT.clear()
    try:
        assert fake_tick() == "ok" and ran["n"] == 1
        assert ct.worker_ticks_in_flight() == 0
        assert ct.request_worker_drain(timeout_s=1) is True
        assert fake_tick() is None and ran["n"] == 1, "con drain pedido no arranca un tick"
    finally:
        ct._DRAIN_EVENT.clear()


def test_wake_chunk_worker_modifies_next_run_time():
    ct = _import_or_skip("cron_tasks")
    class _Job:
        def __init__(self): self.next = None
        def modify(self, **kw): self.next = kw.get("next_run_time")
    class _Sched:
        def __init__(self): self.job = _Job()
        def get_job(self, jid): return self.job if jid == "process_plan_chunk_queue" else None
    old = ct._SCHEDULER_REF
    try:
        ct._SCHEDULER_REF = None
        assert ct.wake_chunk_worker("t") is False
        s = _Sched()
        ct._SCHEDULER_REF = s
        assert ct.wake_chunk_worker("t") is True
        assert s.job.next is not None
    finally:
        ct._SCHEDULER_REF = old


def test_process_plan_chunk_queue_keeps_wrapped_impl():
    ct = _import_or_skip("cron_tasks")
    impl = getattr(ct.process_plan_chunk_queue, "__wrapped__", None)
    assert impl is not None and impl.__name__ == "process_plan_chunk_queue"


# ---------------------------------------------------------------------------
# Endpoints + knob
# ---------------------------------------------------------------------------

def test_router_paths_and_knob_default_off():
    pg = _import_or_skip("routers.plans_generation")
    paths = {r.path for r in pg.router.routes}
    assert paths == {
        "/api/plans/generation-runs",
        "/api/plans/generation-runs/{run_id}",
        "/api/plans/generation-runs/{run_id}/cancel",
        "/api/plans/generation-runs/{run_id}/events",
    }
    from generation_lifecycle import initial_via_queue_enabled
    with patch.dict(os.environ, {"MEALFIT_INITIAL_VIA_QUEUE": "", "MEALFIT_INITIAL_VIA_QUEUE_USERS": ""}):
        os.environ.pop("MEALFIT_INITIAL_VIA_QUEUE", None)
        os.environ.pop("MEALFIT_INITIAL_VIA_QUEUE_USERS", None)
        assert initial_via_queue_enabled() is False
        assert initial_via_queue_enabled("u-1") is False
    # canary por allowlist: solo esos usuarios, sin encender el global
    with patch.dict(os.environ, {"MEALFIT_INITIAL_VIA_QUEUE": "false", "MEALFIT_INITIAL_VIA_QUEUE_USERS": "U-1, u-2"}):
        assert initial_via_queue_enabled("u-1") is True
        assert initial_via_queue_enabled("u-3") is False
        assert initial_via_queue_enabled() is False
    with patch.dict(os.environ, {"MEALFIT_INITIAL_VIA_QUEUE": "true", "MEALFIT_INITIAL_VIA_QUEUE_USERS": ""}):
        assert initial_via_queue_enabled("cualquiera") is True


def test_app_mounts_generation_router_and_marker_bumped():
    app_src = _src("app.py")
    assert "from routers.plans_generation import router as plans_generation_router" in app_src
    assert "app.include_router(plans_generation_router)" in app_src
    # el marker avanza con cada P-fix; aquí basta con que no sea anterior a esta fase
    m = re.search(r'_LAST_KNOWN_PFIX = "[^"]*· (\d{4}-\d{2}-\d{2})"', app_src)
    assert m and m.group(1) >= "2026-09-02", "el marker no puede retroceder por detrás de la Fase 1"
    # drain antes de apagar el scheduler
    d = app_src.find("_req_drain")
    s = app_src.find("scheduler.shutdown(wait=False)")
    assert 0 < d < s


def test_progress_publisher_throttles_and_flushes(monkeypatch):
    import generation_lifecycle as gl
    writes = []
    monkeypatch.setattr(gl, "_write_progress", lambda rid, payload: writes.append(payload))
    monkeypatch.setenv("MEALFIT_RUN_PROGRESS_THROTTLE_S", "30")
    p = gl.RunProgressPublisher("run-x")
    p({"event": "a"}); p({"event": "b"}); p({"event": "c"})
    assert len(writes) == 1 and writes[0]["seq"] == 1  # las dos siguientes quedan pendientes
    p.flush({"event": "complete"})
    assert writes[-1]["event"]["event"] == "complete" and writes[-1]["seq"] == 4


def test_initial_chunk_is_due_immediately():
    """[P1-ARQ25-F1-LIFECYCLE · 2026-09-02, medido en el primer run real] `_enqueue_plan_chunk`
    programa con margen (+60 s); el wake del worker llegaba y el pickup descartaba el chunk 0
    por `execute_after <= NOW()`. El encolado del Bloque 1 lo pone a NOW() en el mismo UPDATE
    que estampa `run_id`."""
    gl = _src("generation_lifecycle.py")
    i = gl.find("UPDATE plan_chunk_queue SET run_id = %s, input_hash = %s, execute_after = NOW()")
    assert i > 0, "el chunk 0 debe quedar vencido al encolarse"
    assert "chunk_kind = %s AND status = 'pending'" in gl[i:i + 300]
