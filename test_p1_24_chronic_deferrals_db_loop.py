"""[P1-24] Tests para que `_detect_chronic_deferrals` no haga N round-trips
DB en el loop de notificación.

Bug original (audit P1-24):
  La SQL agrupa `chunk_deferrals` por (user_id, meal_plan_id, week_number)
  para evaluar el threshold por chunk concreto (HAVING COUNT >= MIN_COUNT).
  Pero el alert_key del cron es per-USER (`chronic_deferrals:{user_id}`).

  Pre-fix: para cada row del GROUP BY, el loop:
    1. Hacía un SELECT contra `system_alerts` (dedupe).
    2. Si pasaba el dedupe, INSERT/ON CONFLICT (los N-1 siguientes eran
       no-ops por el UNIQUE constraint).
    3. Disparaba `_dispatch_push_notification` (mitigado por el dedupe
       SELECT pero aún era doble work).

  Si un mismo usuario tenía deferrals crónicos en 4 (plan, week)
  distintos, producía 4 filas → 4 SELECT dedupe + hasta 4 INSERTs +
  hasta 4 dispatches. Multiplicado por N usuarios crónicos en cada
  tick (cada 6h), el cron desperdiciaba I/O y emitía push duplicados
  en la ventana entre el primer INSERT y la siguiente iteración.

Fix:
  1. Pre-aggregate `rows` por user_id en Python ANTES del loop. La
     entrada con `deferral_count` máximo se queda como representativa
     (señal más fuerte para metadata). Reduce la cardinalidad efectiva
     del loop de O(uniq_user_plan_weeks) a O(uniq_users).
  2. Bulk dedupe: un solo SELECT con `alert_key = ANY(%s)` trae todas
     las alertas dentro del cooldown en una sola query. Membership en
     Python (`set lookup`, O(1)) en lugar de SELECT per-iteration.
  3. Best-effort: si el bulk dedupe falla (DB blip), seguimos sin
     pre-filtro — el ON CONFLICT garantiza que no se duplican rows.

Cobertura:
  - test_no_rows_returns_early_no_dedupe_select
  - test_pre_aggregation_collapses_per_user_rows
  - test_bulk_dedupe_select_uses_any_alert_keys
  - test_bulk_dedupe_skips_users_already_alerted
  - test_per_user_loop_does_not_select_per_iteration
  - test_dedupe_failure_is_best_effort_inserts_continue
  - test_user_with_multiple_chunks_picks_max_deferral_count_for_metadata
  - test_documentation_p1_24_present
"""
import inspect
import json
from unittest.mock import patch, MagicMock

import pytest

import cron_tasks
from cron_tasks import _detect_chronic_deferrals


_SRC = inspect.getsource(cron_tasks._detect_chronic_deferrals)


def _make_row(user_id, plan_id, week, count, last_at="2026-01-01T00:00:00Z"):
    return {
        "user_id": user_id,
        "meal_plan_id": plan_id,
        "week_number": week,
        "deferral_count": count,
        "last_at": last_at,
    }


# ---------------------------------------------------------------------------
# 1. Comportamiento sin candidatos.
# ---------------------------------------------------------------------------
def test_no_rows_returns_early_no_dedupe_select():
    """Si la SQL inicial no devuelve filas, el cron retorna sin disparar
    el bulk dedupe SELECT. Ahorra round-trip cuando no hay nada que hacer."""
    queries = []

    def fake_query(sql, params=None, **kw):
        queries.append((sql, params))
        return []

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write"), \
         patch("cron_tasks._ensure_quality_alert_schema"), \
         patch("cron_tasks._dispatch_push_notification"):
        _detect_chronic_deferrals()

    # Solo el SELECT inicial sobre chunk_deferrals; NO el bulk dedupe.
    assert len(queries) == 1
    assert "FROM chunk_deferrals" in queries[0][0]


# ---------------------------------------------------------------------------
# 2. Pre-aggregación por user_id.
# ---------------------------------------------------------------------------
def test_pre_aggregation_collapses_per_user_rows():
    """Si un usuario tiene deferrals crónicos en 3 (plan, week) distintos,
    el loop de notificación corre UNA vez (no 3) — verificado vía cantidad
    de dispatches y INSERTs en system_alerts."""
    rows = [
        _make_row("user-A", "plan-1", 1, 5),
        _make_row("user-A", "plan-1", 2, 7),
        _make_row("user-A", "plan-2", 3, 6),
        _make_row("user-B", "plan-3", 1, 5),
    ]
    queries = [rows, []]  # Primer SELECT (chunk_deferrals), segundo (bulk dedupe).

    def fake_query(sql, params=None, **kw):
        return queries.pop(0) if queries else []

    push_calls = []

    def fake_push(**kwargs):
        push_calls.append(kwargs.get("user_id"))

    insert_calls = []

    def fake_write(sql, params=None, **kw):
        if "INSERT INTO system_alerts" in sql:
            insert_calls.append(params)

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._ensure_quality_alert_schema"), \
         patch("cron_tasks._dispatch_push_notification", side_effect=fake_push):
        _detect_chronic_deferrals()

    # Solo 2 pushes (user-A + user-B), no 4.
    assert sorted(push_calls) == ["user-A", "user-B"], (
        f"P1-24: pre-agregación falló — push_calls={push_calls}"
    )
    # Solo 2 INSERTs (uno por user_id único).
    assert len(insert_calls) == 2, (
        f"P1-24: esperaba 2 INSERTs (1 por user único), vio {len(insert_calls)}"
    )


def test_user_with_multiple_chunks_picks_max_deferral_count_for_metadata():
    """Cuando un usuario tiene varias filas, la elegida para metadata es
    la de mayor `deferral_count` (señal más fuerte). El metadata.deferral_
    count y el push body deben reflejarlo."""
    rows = [
        _make_row("user-A", "plan-1", 1, 5),    # weak
        _make_row("user-A", "plan-1", 2, 12),   # strongest — debería ganar
        _make_row("user-A", "plan-2", 3, 8),    # mid
    ]
    queries = [rows, []]

    def fake_query(sql, params=None, **kw):
        return queries.pop(0) if queries else []

    push_kwargs = []
    insert_metadata = []

    def fake_push(**kwargs):
        push_kwargs.append(kwargs)

    def fake_write(sql, params=None, **kw):
        if "INSERT INTO system_alerts" in sql:
            # params[3] es el JSON metadata.
            metadata = json.loads(params[3])
            insert_metadata.append(metadata)

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._ensure_quality_alert_schema"), \
         patch("cron_tasks._dispatch_push_notification", side_effect=fake_push):
        _detect_chronic_deferrals()

    assert len(insert_metadata) == 1
    assert insert_metadata[0]["deferral_count"] == 12, (
        f"P1-24: la fila representativa debe ser la de max deferral_count "
        f"(12), vio {insert_metadata[0]['deferral_count']}"
    )
    # El push body debe mencionar el conteo más fuerte.
    assert "12" in push_kwargs[0].get("body", ""), (
        f"P1-24: push body debe reflejar el max deferral_count: "
        f"{push_kwargs[0].get('body')!r}"
    )


# ---------------------------------------------------------------------------
# 3. Bulk dedupe.
# ---------------------------------------------------------------------------
def test_bulk_dedupe_select_uses_any_alert_keys():
    """El bulk dedupe debe ejecutarse con `ANY(%s)` y la lista de alert_keys
    de los usuarios candidatos. Sin esto, el cron seguiría haciendo SELECT
    por-usuario en el loop."""
    rows = [
        _make_row("user-A", "plan-1", 1, 5),
        _make_row("user-B", "plan-2", 1, 6),
    ]
    captured = []

    def fake_query(sql, params=None, **kw):
        captured.append((sql, params))
        if "FROM chunk_deferrals" in sql:
            return rows
        # Bulk dedupe.
        return []

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write"), \
         patch("cron_tasks._ensure_quality_alert_schema"), \
         patch("cron_tasks._dispatch_push_notification"):
        _detect_chronic_deferrals()

    bulk = [
        (sql, params) for sql, params in captured
        if "FROM system_alerts" in sql and "ANY(%s)" in sql
    ]
    assert bulk, (
        f"P1-24: esperaba un SELECT bulk con `alert_key = ANY(%s)`. "
        f"Queries: {[s[:120] for s, _ in captured]}"
    )
    # El primer parámetro debe ser la lista de alert_keys.
    bulk_sql, bulk_params = bulk[0]
    keys = bulk_params[0]
    assert sorted(keys) == [
        "chronic_deferrals:user-A",
        "chronic_deferrals:user-B",
    ], f"P1-24: alert_keys del bulk dedupe inesperados: {keys}"


def test_bulk_dedupe_skips_users_already_alerted():
    """Si el bulk dedupe devuelve un alert_key, ese usuario NO debe ser
    notificado en este tick (cooldown). Los demás sí."""
    rows = [
        _make_row("user-A", "plan-1", 1, 5),
        _make_row("user-B", "plan-2", 1, 6),
        _make_row("user-C", "plan-3", 1, 7),
    ]

    def fake_query(sql, params=None, **kw):
        if "FROM chunk_deferrals" in sql:
            return rows
        # Bulk dedupe: user-A y user-C ya alertados recientemente.
        return [
            {"alert_key": "chronic_deferrals:user-A"},
            {"alert_key": "chronic_deferrals:user-C"},
        ]

    push_calls = []
    insert_calls = []

    def fake_push(**kwargs):
        push_calls.append(kwargs.get("user_id"))

    def fake_write(sql, params=None, **kw):
        if "INSERT INTO system_alerts" in sql:
            insert_calls.append(params)

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._ensure_quality_alert_schema"), \
         patch("cron_tasks._dispatch_push_notification", side_effect=fake_push):
        _detect_chronic_deferrals()

    # Solo user-B debe pasar el dedupe.
    assert push_calls == ["user-B"], (
        f"P1-24: dedupe debió saltar user-A y user-C, vio: {push_calls}"
    )
    assert len(insert_calls) == 1


def test_per_user_loop_does_not_select_per_iteration():
    """Defensa estructural: el cuerpo del loop NO debe contener un SELECT
    de dedupe. Sin esto, alguien podría reintroducir el bug haciendo
    `SELECT triggered_at FROM system_alerts WHERE alert_key = %s` dentro
    del loop. Verificamos vía conteo de queries observadas."""
    rows = [
        _make_row(f"user-{i}", "plan-x", 1, 5) for i in range(20)
    ]
    queries = []

    def fake_query(sql, params=None, **kw):
        queries.append(sql)
        if "FROM chunk_deferrals" in sql:
            return rows
        return []

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write"), \
         patch("cron_tasks._ensure_quality_alert_schema"), \
         patch("cron_tasks._dispatch_push_notification"):
        _detect_chronic_deferrals()

    # Esperamos exactamente 2 queries: chunk_deferrals + bulk dedupe.
    # Sin el fix P1-24 habría 1 + 20 = 21 queries.
    assert len(queries) == 2, (
        f"P1-24: esperaba 2 SELECTs totales (chunk_deferrals + bulk dedupe), "
        f"vio {len(queries)}. Si > 2, el loop está haciendo SELECT por "
        f"iteración (regresión)."
    )


# ---------------------------------------------------------------------------
# 4. Defensa contra fallos del bulk dedupe.
# ---------------------------------------------------------------------------
def test_dedupe_failure_is_best_effort_inserts_continue():
    """Si el bulk dedupe falla (DB blip), el cron debe continuar y emitir
    los INSERTs. El UNIQUE constraint + ON CONFLICT en `system_alerts`
    garantiza que no hay duplicados."""
    rows = [_make_row("user-A", "plan-1", 1, 5)]

    def fake_query(sql, params=None, **kw):
        if "FROM chunk_deferrals" in sql:
            return rows
        # Bulk dedupe falla.
        raise RuntimeError("simulated DB blip")

    inserted = []

    def fake_write(sql, params=None, **kw):
        if "INSERT INTO system_alerts" in sql:
            inserted.append(params)

    push_calls = []

    def fake_push(**kwargs):
        push_calls.append(kwargs.get("user_id"))

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._ensure_quality_alert_schema"), \
         patch("cron_tasks._dispatch_push_notification", side_effect=fake_push):
        # No debe lanzar.
        _detect_chronic_deferrals()

    # Tras el fail del dedupe, el INSERT y el push siguen.
    assert push_calls == ["user-A"]
    assert len(inserted) == 1


# ---------------------------------------------------------------------------
# 5. Documentación.
# ---------------------------------------------------------------------------
def test_documentation_p1_24_present():
    """Comentario `[P1-24]` debe documentar el rationale del bulk dedupe
    + pre-agregación."""
    full_src = inspect.getsource(cron_tasks)
    assert "[P1-24]" in full_src, (
        "P1-24: falta marker que documente la pre-agregación + bulk dedupe."
    )


def test_documentation_mentions_perf_or_round_trips():
    """El comentario debe mencionar el problema de I/O o round-trips para
    que un futuro lector entienda por qué el código tiene esta forma y no
    la versión "más simple" del loop con SELECT per-iteration."""
    idx = _SRC.find("[P1-24]")
    assert idx > -1
    window = _SRC[idx : idx + 2000]
    needles = ["round-trip", "round trip", "i/o", "n round", "bulk", "agregar", "per-iteration"]
    assert any(n in window.lower() for n in needles), (
        f"P1-24: el comentario debe explicar el problema (round-trips / "
        f"bulk / per-iteration). Encontrado: {window[:300]!r}"
    )
