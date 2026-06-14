"""[P2-HIST-5 · 2026-05-09] Tests del cron `_gc_orphan_chunk_telemetry`
y de su registro en `register_plan_chunk_scheduler`.

Bug original (audit historial 2026-05-08):
    Las tablas `chunk_lesson_telemetry` y `chunk_deferrals` carecían
    de FK a `meal_plans`, así que al borrar planes los rows quedaban
    huérfanos sin GC. P0-HIST-3 cerró la primera mitad añadiendo FK
    con `ON DELETE SET NULL` (preserva contenido analítico, libera
    el FK constraint). P2-HIST-5 cierra la segunda mitad con un cron
    mensual que purga orphans más antiguos que la ventana de
    retención (default 90d).

Cobertura:
    - Cron es callable directamente (ejecutable bajo APScheduler).
    - SQL contract: `meal_plan_id IS NULL` Y `created_at < NOW() - interval Nd`.
    - Solo afecta las DOS tablas (chunk_lesson_telemetry + chunk_deferrals).
    - Cap por LIMIT (max_rows_per_run) para evitar lock prolongado.
    - Knob kill switch `MEALFIT_ORPHAN_TELEMETRY_GC_ENABLED=false` → no-op.
    - Knob retention <= 0 → no-op (defensivo, log warning).
    - Una falla en una tabla no aborta la otra.
    - Cron registrado en `register_plan_chunk_scheduler` con default 720h.
"""
import json
import inspect
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# 1. Anchor / contrato del cron
# ---------------------------------------------------------------------------
def test_p2_hist_5_anchor_in_function():
    from cron_tasks import _gc_orphan_chunk_telemetry
    src = inspect.getsource(_gc_orphan_chunk_telemetry)
    assert "P2-HIST-5" in src
    # SQL contract: ambas tablas + filtro orphan + retention.
    assert "chunk_lesson_telemetry" in src
    assert "chunk_deferrals" in src
    assert "meal_plan_id IS NULL" in src
    assert "make_interval(days =>" in src
    # Cap por LIMIT (no DELETE sin tope — protege contra lock).
    assert "LIMIT" in src


def test_function_is_callable():
    """Sanity: la función debe ser importable y callable sin args."""
    from cron_tasks import _gc_orphan_chunk_telemetry
    assert callable(_gc_orphan_chunk_telemetry)
    sig = inspect.signature(_gc_orphan_chunk_telemetry)
    assert len(sig.parameters) == 0


# ---------------------------------------------------------------------------
# 2. Knobs / kill-switches
# ---------------------------------------------------------------------------
def test_gc_skips_when_kill_switch_disabled(monkeypatch):
    """`MEALFIT_ORPHAN_TELEMETRY_GC_ENABLED=false` → no DELETE emitido."""
    monkeypatch.setenv("MEALFIT_ORPHAN_TELEMETRY_GC_ENABLED", "false")
    from cron_tasks import _gc_orphan_chunk_telemetry

    with patch("db_core.execute_sql_write") as mock_write:
        _gc_orphan_chunk_telemetry()

    # No DELETE debe haberse llamado. (Posible INSERT a pipeline_metrics
    # en otros paths, pero el path skip retorna ANTES de cualquier write).
    assert mock_write.call_count == 0


def test_gc_skips_when_retention_zero_or_negative(monkeypatch):
    """retention_days <= 0 borraría TODO row huérfano sin buffer
    post-mortem. El cron debe detectar y skipar (con log)."""
    monkeypatch.setenv("MEALFIT_ORPHAN_TELEMETRY_GC_ENABLED", "true")
    monkeypatch.setenv("MEALFIT_ORPHAN_TELEMETRY_RETENTION_DAYS", "0")
    from cron_tasks import _gc_orphan_chunk_telemetry

    with patch("db_core.execute_sql_write") as mock_write:
        _gc_orphan_chunk_telemetry()
    assert mock_write.call_count == 0

    monkeypatch.setenv("MEALFIT_ORPHAN_TELEMETRY_RETENTION_DAYS", "-5")
    with patch("db_core.execute_sql_write") as mock_write:
        _gc_orphan_chunk_telemetry()
    assert mock_write.call_count == 0


def test_gc_uses_default_retention_when_knob_unset(monkeypatch):
    """Sin knob, retention=90 días (default)."""
    monkeypatch.delenv("MEALFIT_ORPHAN_TELEMETRY_RETENTION_DAYS", raising=False)
    monkeypatch.setenv("MEALFIT_ORPHAN_TELEMETRY_GC_ENABLED", "true")
    from cron_tasks import _gc_orphan_chunk_telemetry

    captured = []
    def _spy(sql, params=None, **kw):
        captured.append({"sql": sql, "params": params})
        return []

    with patch("db_core.execute_sql_write", side_effect=_spy):
        _gc_orphan_chunk_telemetry()

    # Solo DELETEs (excluye el INSERT a pipeline_metrics).
    deletes = [c for c in captured if "DELETE FROM" in c["sql"]]
    assert all(d["params"][0] == 90 for d in deletes), (
        f"Esperaba todos los DELETEs con retention=90d (default); deletes: {deletes}"
    )
    assert len(deletes) >= 1


# ---------------------------------------------------------------------------
# 3. SQL contract en success path
# ---------------------------------------------------------------------------
def test_gc_emits_two_deletes_one_per_table(monkeypatch):
    """Una DELETE para chunk_lesson_telemetry + otra para chunk_deferrals.
    Las DELETE deben tener el mismo retention_days y max_rows."""
    monkeypatch.setenv("MEALFIT_ORPHAN_TELEMETRY_GC_ENABLED", "true")
    monkeypatch.setenv("MEALFIT_ORPHAN_TELEMETRY_RETENTION_DAYS", "60")
    monkeypatch.setenv("MEALFIT_ORPHAN_TELEMETRY_GC_MAX_ROWS", "1000")
    from cron_tasks import _gc_orphan_chunk_telemetry

    captured = []
    def _spy(sql, params=None, **kw):
        captured.append({"sql": sql, "params": params, "returning": kw.get("returning")})
        return []  # 0 rows deleted

    with patch("db_core.execute_sql_write", side_effect=_spy):
        _gc_orphan_chunk_telemetry()

    # Filtrar solo los DELETEs (puede haber INSERT a pipeline_metrics también).
    deletes = [c for c in captured if "DELETE FROM" in c["sql"]]
    assert len(deletes) == 2, f"Esperaba 2 DELETEs; got {len(deletes)}: {deletes}"

    tables_targeted = [d["sql"] for d in deletes]
    assert any("chunk_lesson_telemetry" in s for s in tables_targeted)
    assert any("chunk_deferrals" in s for s in tables_targeted)

    # Cada DELETE debe usar params (retention=60, max_rows=1000).
    for d in deletes:
        assert d["params"][0] == 60
        assert d["params"][1] == 1000
        # SQL contract: filter orphan + retention + LIMIT.
        assert "meal_plan_id IS NULL" in d["sql"]
        assert "make_interval" in d["sql"]
        assert "LIMIT" in d["sql"]
        # RETURNING para count del rowcount.
        assert "RETURNING" in d["sql"]
        assert d["returning"] is True


def test_gc_continues_when_one_table_fails(monkeypatch):
    """Si DELETE en chunk_lesson_telemetry falla, debe seguir
    intentando chunk_deferrals (best-effort)."""
    monkeypatch.setenv("MEALFIT_ORPHAN_TELEMETRY_GC_ENABLED", "true")
    monkeypatch.setenv("MEALFIT_ORPHAN_TELEMETRY_RETENTION_DAYS", "30")
    from cron_tasks import _gc_orphan_chunk_telemetry

    call_count = {"n": 0}
    def _spy(sql, params=None, **kw):
        call_count["n"] += 1
        if "chunk_lesson_telemetry" in sql and "DELETE" in sql:
            raise RuntimeError("simulated DB hiccup")
        return []

    # No debe propagarse — el cron es best-effort.
    with patch("db_core.execute_sql_write", side_effect=_spy):
        _gc_orphan_chunk_telemetry()

    # Debió intentar las DOS DELETEs (la primera tira error, la segunda igual corre).
    # call_count >= 2 (deletes) + 1 (pipeline_metrics insert posiblemente).
    assert call_count["n"] >= 2


def test_gc_uses_default_max_rows_when_knob_unset(monkeypatch):
    """Sin MEALFIT_ORPHAN_TELEMETRY_GC_MAX_ROWS, default 50000."""
    monkeypatch.delenv("MEALFIT_ORPHAN_TELEMETRY_GC_MAX_ROWS", raising=False)
    monkeypatch.setenv("MEALFIT_ORPHAN_TELEMETRY_GC_ENABLED", "true")
    from cron_tasks import _gc_orphan_chunk_telemetry

    captured = []
    def _spy(sql, params=None, **kw):
        captured.append({"sql": sql, "params": params})
        return []

    with patch("db_core.execute_sql_write", side_effect=_spy):
        _gc_orphan_chunk_telemetry()

    # Solo DELETEs (el INSERT a pipeline_metrics tiene shape distinto).
    deletes = [c for c in captured if "DELETE FROM" in c["sql"]]
    assert len(deletes) >= 2, f"Esperaba 2 DELETEs; got {len(deletes)}"
    for d in deletes:
        # params == (retention_days, max_rows). max_rows debe ser 50000 default.
        assert d["params"][1] == 50000, (
            f"Esperaba max_rows=50000 (default); got {d['params']}"
        )


# ---------------------------------------------------------------------------
# 4. Pipeline metrics emit (observabilidad)
# ---------------------------------------------------------------------------
def test_gc_emits_pipeline_metrics_with_counts(monkeypatch):
    """Cada run debe emitir 1 INSERT a pipeline_metrics con counts."""
    monkeypatch.setenv("MEALFIT_ORPHAN_TELEMETRY_GC_ENABLED", "true")
    monkeypatch.setenv("MEALFIT_ORPHAN_TELEMETRY_RETENTION_DAYS", "30")
    from cron_tasks import _gc_orphan_chunk_telemetry

    inserted_payloads = []
    def _spy(sql, params=None, **kw):
        if "INSERT INTO pipeline_metrics" in sql:
            inserted_payloads.append(params)
        return []

    with patch("db_core.execute_sql_write", side_effect=_spy):
        _gc_orphan_chunk_telemetry()

    assert len(inserted_payloads) == 1, "Esperaba 1 INSERT a pipeline_metrics"
    metadata_json = inserted_payloads[0][-1]  # último param es metadata jsonb
    metadata = json.loads(metadata_json)
    assert "retention_days" in metadata
    assert metadata["retention_days"] == 30
    assert "deleted_chunk_lesson_telemetry" in metadata
    assert "deleted_chunk_deferrals" in metadata
    assert "total_deleted" in metadata
    # Node correcto.
    assert inserted_payloads[0][2] == "_gc_orphan_chunk_telemetry"


# ---------------------------------------------------------------------------
# 5. Registro en el scheduler
# ---------------------------------------------------------------------------
def test_cron_registered_in_register_plan_chunk_scheduler():
    """`register_plan_chunk_scheduler` debe agregar el job
    `gc_orphan_chunk_telemetry` con el helper add_job."""
    import cron_tasks
    src = inspect.getsource(cron_tasks.register_plan_chunk_scheduler)
    # Anchor presente.
    assert "P2-HIST-5" in src
    # ID del job.
    assert "gc_orphan_chunk_telemetry" in src
    # Knob de intervalo.
    assert "MEALFIT_ORPHAN_TELEMETRY_GC_INTERVAL_HOURS" in src


def test_register_plan_chunk_scheduler_adds_job():
    """Cuando `register_plan_chunk_scheduler` se llama con un scheduler
    real (mock APScheduler), debe llamar add_job para nuestro cron."""
    import cron_tasks

    # Mock scheduler con get_job que devuelve None (job no existe → registrar).
    mock_scheduler = MagicMock()
    mock_scheduler.get_job.return_value = None

    cron_tasks.register_plan_chunk_scheduler(mock_scheduler)

    # Inspeccionar las llamadas a add_job, encontrando la que tiene
    # id="gc_orphan_chunk_telemetry".
    add_job_calls = mock_scheduler.add_job.call_args_list
    ids = [c.kwargs.get("id") for c in add_job_calls]
    assert "gc_orphan_chunk_telemetry" in ids, (
        f"Esperaba job 'gc_orphan_chunk_telemetry' registrado; ids: {ids}"
    )

    # Verificar config defensiva.
    gc_call = next(
        c for c in add_job_calls
        if c.kwargs.get("id") == "gc_orphan_chunk_telemetry"
    )
    assert gc_call.kwargs.get("max_instances") == 1
    assert gc_call.kwargs.get("coalesce") is True
    assert gc_call.kwargs.get("replace_existing") is True
    # Trigger interval (no cron literal — más fácil de tunear vía knob).
    assert gc_call.args[1] == "interval" or gc_call.kwargs.get("trigger") == "interval"


def test_register_plan_chunk_scheduler_default_interval_720h(monkeypatch):
    """Sin knob, el cron se registra cada 720h (≈30 días)."""
    monkeypatch.delenv("MEALFIT_ORPHAN_TELEMETRY_GC_INTERVAL_HOURS", raising=False)
    import cron_tasks

    mock_scheduler = MagicMock()
    mock_scheduler.get_job.return_value = None

    cron_tasks.register_plan_chunk_scheduler(mock_scheduler)

    gc_call = next(
        c for c in mock_scheduler.add_job.call_args_list
        if c.kwargs.get("id") == "gc_orphan_chunk_telemetry"
    )
    assert gc_call.kwargs.get("hours") == 720


def test_register_plan_chunk_scheduler_clamps_invalid_interval(monkeypatch):
    """Knob con valor 0/negativo → fallback al default 720h (no
    intervalo inválido en APScheduler)."""
    monkeypatch.setenv("MEALFIT_ORPHAN_TELEMETRY_GC_INTERVAL_HOURS", "0")
    import cron_tasks

    mock_scheduler = MagicMock()
    mock_scheduler.get_job.return_value = None

    cron_tasks.register_plan_chunk_scheduler(mock_scheduler)

    gc_call = next(
        c for c in mock_scheduler.add_job.call_args_list
        if c.kwargs.get("id") == "gc_orphan_chunk_telemetry"
    )
    assert gc_call.kwargs.get("hours") == 720


def test_register_plan_chunk_scheduler_idempotent(monkeypatch):
    """Si el job ya existe (`get_job` devuelve not-None), NO se re-registra."""
    import cron_tasks

    mock_scheduler = MagicMock()
    # get_job retorna un job existente para 'gc_orphan_chunk_telemetry'.
    def _get_job(job_id):
        if job_id == "gc_orphan_chunk_telemetry":
            return MagicMock()  # job exists
        return None
    mock_scheduler.get_job.side_effect = _get_job

    cron_tasks.register_plan_chunk_scheduler(mock_scheduler)

    # No debió añadir el GC job (ya existía).
    add_job_calls = mock_scheduler.add_job.call_args_list
    ids = [c.kwargs.get("id") for c in add_job_calls]
    assert "gc_orphan_chunk_telemetry" not in ids


# ---------------------------------------------------------------------------
# 6. Migration / FK constraints — SSOT closed reference
# ---------------------------------------------------------------------------
def test_p0_hist_3_migration_still_present():
    """[P2-HIST-5] La FK SET NULL en chunk_lesson_telemetry +
    chunk_deferrals fue aplicada en P0-HIST-3 (migración SSOT). Este
    test referencia la dependencia: el GC depende de que la FK exista
    (sin ella, los `meal_plan_id` post-DELETE seguirían apuntando a
    rows fantasma y el filter `IS NULL` nunca matchearía).
    """
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent.parent
    mig = repo_root / "migrations" / "p0_hist_3_telemetry_orphan_fk.sql"
    assert mig.is_file(), (
        f"Falta migración P0-HIST-3 ({mig.name}). El cron _gc_orphan_chunk_telemetry "
        "depende de que la FK SET NULL exista."
    )
    sql = mig.read_text(encoding="utf-8")
    assert "chunk_lesson_telemetry" in sql
    assert "chunk_deferrals" in sql
    assert "ON DELETE SET NULL" in sql
