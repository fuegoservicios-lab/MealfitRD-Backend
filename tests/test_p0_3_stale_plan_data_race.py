"""[P0-3 FIX] Tests para la recuperación de plan_data stale en el gate temporal.

Race que el fix cierra:
    Bajo READ COMMITTED (default de PostgreSQL), las dos lecturas en el path del gate
    viven en transacciones distintas:
      1. El caller (`_chunk_worker` o `_recover_pantry_paused_chunks`) lee plan_data
         al tiempo T (sin lock).
      2. El gate ejecuta su queue check al tiempo T+2 (sin lock, transacción nueva).
    Si entre ambos commits ocurren T1+T2 del chunk previo:
      - plan_data upstream NO tiene los días del chunk previo (snapshot stale).
      - queue check ve N como 'completed' (post-T2) → 0 active prior chunks.
      - Pre-fix: gate retornaba `ready=True, reason='missing_previous_chunk_days'`
        y el chunk N+1 dispatcheaba con `previous_chunk_days=[]`, perdiendo las
        lecciones reales de N — exactamente el bug que P0-3 cierra.
      - Post-fix: el gate detecta el caso (queue confirma 0 active pero
        previous_chunk_days vacío) y re-lee plan_data fresca. Si los días aparecen,
        continúa con datos actualizados; si no, conserva el fail-open original.

La pickup query (`process_plan_chunk_queue`) ya serializa por meal_plan_id, pero
los call sites de recovery (`_recover_pantry_paused_chunks`) no pasan por el
pickup — esa es la ventana real donde la race aparece.

Ejecutar:
    cd backend && python -m pytest tests/test_p0_3_stale_plan_data_race.py -v
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stale_plan_data_no_prev_days():
    """plan_data como lo leería el caller ANTES del commit T1+T2 del chunk previo:
    sólo con snapshot/form_data, sin los días del chunk previo todavía mergeados."""
    return {
        "days": [],  # chunk previo aún no mergeó.
        "total_days_requested": 7,
    }


def _fresh_plan_data_with_prev_days():
    """plan_data como queda DESPUÉS del commit T1+T2: días 1..3 del chunk previo
    presentes en plan_data.days. Esto es lo que el caller debería ver pero no ve
    por la race."""
    return {
        "days": [
            {"day": 1, "meals": [{"name": "Pollo"}]},
            {"day": 2, "meals": [{"name": "Carne"}]},
            {"day": 3, "meals": [{"name": "Pescado"}]},
        ],
        "total_days_requested": 7,
        # Lección persistida por T1 atómico (P0-1 FIX).
        "_last_chunk_learning": {"chunk": 1, "rejection_violations": 0},
    }


def _form_data_chunk2():
    return {
        "form_data": {
            "totalDays": 7,
            "_plan_start_date": "2026-04-25",
            "tz_offset_minutes": 0,
        },
        "totalDays": 7,
    }


# ---------------------------------------------------------------------------
# 1. Race confirmada: queue dice 0 active, plan_data fresca tiene días → recovery
# ---------------------------------------------------------------------------
@patch("cron_tasks._record_chunk_deferral")
@patch("cron_tasks.execute_sql_query")
def test_gate_recovers_from_stale_upstream_plan_data(mock_query, mock_deferral):
    """[P0-3 FIX] Si el caller pasó plan_data stale (sin días del chunk previo) Y
    queue check ve 0 active prior chunks, el gate debe re-leer plan_data fresca.
    Si encuentra los días en la lectura fresca, continúa la evaluación normal en
    lugar de retornar `ready=True, reason='missing_previous_chunk_days'`.
    """
    from cron_tasks import _check_chunk_learning_ready

    # Queue check: 0 active. Re-read fresca: tiene los días.
    def query_side_effect(sql, *args, **kwargs):
        if "FROM plan_chunk_queue" in sql and "COUNT" in sql:
            return {"n": 0}
        if "SELECT plan_data FROM meal_plans" in sql:
            return {"plan_data": _fresh_plan_data_with_prev_days()}
        if "FROM meal_plans WHERE id" in sql and "grocery_start_date" in sql:
            return {"gsd": "2026-04-25", "created_at": None}
        return None

    mock_query.side_effect = query_side_effect

    res = _check_chunk_learning_ready(
        user_id="u1",
        meal_plan_id="plan-stale-1",
        week_number=2,
        days_offset=3,
        plan_data=_stale_plan_data_no_prev_days(),  # stale upstream
        snapshot=_form_data_chunk2(),
    )

    # NO debe ser fail-open con missing_previous_chunk_days: la race fue detectada
    # y el gate continúa la evaluación normal con los días recuperados.
    assert res.get("reason") != "missing_previous_chunk_days", (
        f"El gate retornó fail-open con plan_data stale a pesar de que la re-lectura "
        f"fresca tenía los días: {res!r}"
    )
    # NO debe deferir por prev_chunk_still_active (queue dijo 0).
    assert res.get("reason") != "prev_chunk_still_active_in_queue", res

    # Telemetría: debe registrar el deferral con reason 'stale_plan_data_recovered'.
    deferral_reasons = [
        c.kwargs.get("reason") for c in mock_deferral.call_args_list
    ]
    assert "stale_plan_data_recovered" in deferral_reasons, (
        f"Telemetría P0-3/STALE-RECOVERY no registrada. Reasons: {deferral_reasons}"
    )


# ---------------------------------------------------------------------------
# 2. No hay race: queue dice 0 active y plan_data fresca también vacía → fail-open
# ---------------------------------------------------------------------------
@patch("cron_tasks._record_chunk_deferral")
@patch("cron_tasks.execute_sql_query")
def test_gate_returns_fail_open_when_fresh_plan_data_also_empty(
    mock_query, mock_deferral
):
    """Si la re-lectura fresca tampoco tiene los días, conservamos el fail-open
    legacy (ready=True, missing_previous_chunk_days). Cubre el caso legítimo de
    chunk previo failed permanentemente, plan corrupto, etc.
    """
    from cron_tasks import _check_chunk_learning_ready

    empty_pd = _stale_plan_data_no_prev_days()

    def query_side_effect(sql, *args, **kwargs):
        if "FROM plan_chunk_queue" in sql and "COUNT" in sql:
            return {"n": 0}
        if "SELECT plan_data FROM meal_plans" in sql:
            return {"plan_data": empty_pd}  # también vacía en DB.
        if "FROM meal_plans WHERE id" in sql and "grocery_start_date" in sql:
            return {"gsd": "2026-04-25", "created_at": None}
        return None

    mock_query.side_effect = query_side_effect

    res = _check_chunk_learning_ready(
        user_id="u2",
        meal_plan_id="plan-stale-2",
        week_number=2,
        days_offset=3,
        plan_data=empty_pd,
        snapshot=_form_data_chunk2(),
    )

    assert res.get("ready") is True
    assert res.get("reason") == "missing_previous_chunk_days"


# ---------------------------------------------------------------------------
# 3. Queue check dice prior active → defer ANTES de la re-lectura fresca
# ---------------------------------------------------------------------------
@patch("cron_tasks._record_chunk_deferral")
@patch("cron_tasks.execute_sql_query")
def test_gate_defers_when_queue_shows_active_prior_chunks(
    mock_query, mock_deferral
):
    """Si el queue check ve chunks activos pre-N+1, NO debe llegar a la re-lectura
    fresca: debe deferir inmediatamente (path P0-3 original). La re-lectura es
    sólo para el caso post-T2 donde queue dice 0 pero plan_data está stale.
    """
    from cron_tasks import _check_chunk_learning_ready

    plan_data_read_calls = []

    def query_side_effect(sql, *args, **kwargs):
        if "FROM plan_chunk_queue" in sql and "COUNT" in sql:
            return {"n": 1}  # hay un chunk activo
        if "SELECT plan_data FROM meal_plans" in sql:
            plan_data_read_calls.append(args)
            return {"plan_data": _stale_plan_data_no_prev_days()}
        if "FROM meal_plans WHERE id" in sql and "grocery_start_date" in sql:
            return {"gsd": "2026-04-25", "created_at": None}
        return None

    mock_query.side_effect = query_side_effect

    res = _check_chunk_learning_ready(
        user_id="u3",
        meal_plan_id="plan-stale-3",
        week_number=2,
        days_offset=3,
        plan_data=_stale_plan_data_no_prev_days(),
        snapshot=_form_data_chunk2(),
    )

    assert res.get("ready") is False
    assert res.get("reason") == "prev_chunk_still_active_in_queue"
    # NO debe haber re-lectura fresca de plan_data en el path defer.
    assert len(plan_data_read_calls) == 0, (
        "El gate hizo re-lectura fresca cuando ya había deferido por queue activo "
        "— ineficiente y rompe el orden lógico del fix."
    )


# ---------------------------------------------------------------------------
# 4. Error en la re-lectura fresca → fail-open (no crashear)
# ---------------------------------------------------------------------------
@patch("cron_tasks._record_chunk_deferral")
@patch("cron_tasks.execute_sql_query")
def test_gate_falls_open_when_fresh_reread_errors(mock_query, mock_deferral):
    """Si execute_sql_query lanza durante la re-lectura fresca, debemos caer al
    fail-open (ready=True, missing_previous_chunk_days). Mejor permitir generación
    con el snapshot stale que crashear el worker.
    """
    from cron_tasks import _check_chunk_learning_ready

    def query_side_effect(sql, *args, **kwargs):
        if "FROM plan_chunk_queue" in sql and "COUNT" in sql:
            return {"n": 0}
        if "SELECT plan_data FROM meal_plans" in sql:
            raise RuntimeError("simulated DB connection blip")
        if "FROM meal_plans WHERE id" in sql and "grocery_start_date" in sql:
            return {"gsd": "2026-04-25", "created_at": None}
        return None

    mock_query.side_effect = query_side_effect

    res = _check_chunk_learning_ready(
        user_id="u4",
        meal_plan_id="plan-stale-4",
        week_number=2,
        days_offset=3,
        plan_data=_stale_plan_data_no_prev_days(),
        snapshot=_form_data_chunk2(),
    )

    assert res.get("ready") is True
    assert res.get("reason") == "missing_previous_chunk_days"


# ---------------------------------------------------------------------------
# 5. plan_data upstream YA tiene los días → no se necesita re-lectura
# ---------------------------------------------------------------------------
@patch("cron_tasks._record_chunk_deferral")
@patch("cron_tasks.execute_sql_query")
def test_gate_uses_upstream_plan_data_when_already_populated(
    mock_query, mock_deferral
):
    """Happy path: si plan_data upstream YA tiene los días del chunk previo
    (caller en pickup post-T2 sin race), el gate ni siquiera entra al bloque
    P0-3 — la re-lectura fresca es exclusiva del caso `previous_chunk_days=[]`.
    """
    from cron_tasks import _check_chunk_learning_ready

    fresh_reads = []

    def query_side_effect(sql, *args, **kwargs):
        if "FROM plan_chunk_queue" in sql and "COUNT" in sql:
            return {"n": 0}
        if "SELECT plan_data FROM meal_plans" in sql:
            fresh_reads.append(args)
            return {"plan_data": _fresh_plan_data_with_prev_days()}
        if "FROM meal_plans WHERE id" in sql and "grocery_start_date" in sql:
            return {"gsd": "2026-04-25", "created_at": None}
        return None

    mock_query.side_effect = query_side_effect

    res = _check_chunk_learning_ready(
        user_id="u5",
        meal_plan_id="plan-stale-5",
        week_number=2,
        days_offset=3,
        plan_data=_fresh_plan_data_with_prev_days(),  # upstream ya está al día
        snapshot=_form_data_chunk2(),
    )

    # No re-lectura: el bloque del fix sólo entra cuando previous_chunk_days=[].
    assert len(fresh_reads) == 0, (
        f"El gate hizo re-lectura fresca innecesaria con plan_data upstream ya "
        f"populada: {fresh_reads!r}"
    )
    # El gate continúa la evaluación normal (no fail-open prematuro).
    assert res.get("reason") != "missing_previous_chunk_days"


# ---------------------------------------------------------------------------
# 6. Smoke test del invariante: el código fuente del gate tiene la rama
# ---------------------------------------------------------------------------
def test_gate_source_has_p0_3_stale_recovery_branch():
    """Contrato: el gate `_check_chunk_learning_ready` debe contener la rama
    P0-3/STALE-RECOVERY entre el queue check y el fail-open original. Sin esta
    rama, la race upstream-vs-queue dispara `ready=True` con previous_chunk_days
    vacío y el chunk N+1 dispatcha sin lecciones reales del chunk N.
    """
    src_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cron_tasks.py",
    )
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()

    assert "[P0-3/STALE-RECOVERY]" in source, (
        "El gate no tiene la rama de recovery de plan_data stale. Antes del fix "
        "P0-3, una race entre la lectura upstream y el commit T1+T2 dejaba "
        "previous_chunk_days vacío y el gate retornaba ready=True con datos "
        "incompletos."
    )
    assert "stale_plan_data_recovered" in source, (
        "El gate no registra telemetría 'stale_plan_data_recovered' para la "
        "race detectada — sin esta señal no se puede medir frecuencia del race "
        "en producción."
    )
