"""[P3-B · 2026-05-08] Tests del cron `_aggregate_coherence_block_history_metrics`.

Verifica el contrato de la métrica horaria que agrega
`_shopping_coherence_block_history` (escrita por P3-NEW-C en
`graph_orchestrator.assemble_plan_node`) a `pipeline_metrics`.

Cobertura:
  1. Conteo correcto de cada categoría de `action_taken` (incluido el
     invariant violation `null_block_set` y el legacy `none_other`).
  2. `confidence=0.0` cuando hay anomalías; `1.0` cuando todo normal.
  3. Resiliencia ante plan_data corrupto (string JSON inválido) y entradas
     no-dict en el historial.
  4. Tick con skip_reason si el pool SQL no está inicializado (no crash).
  5. INSERT a pipeline_metrics es best-effort (un fail de DB no debe
     enmascarar el motivo real del cron, pero tampoco crashea el scheduler).
  6. Knob `MEALFIT_COHERENCE_METRICS_LOOKBACK_H` con defensas contra
     valores patológicos (NaN/inf/<=0 → 1.0).

[P1-NEON-DB-MIGRATION · 2026-06-12] Re-anclado: el cron ya no usa el
builder PostgREST (`supabase.table(...).select(...).gte(...)`) sino
`execute_sql_query` contra el pool psycopg (Neon). Los stubs mockean
`cron_tasks.execute_sql_query` (fetch de meal_plans) + `db_core.connection_pool`
(guard de disponibilidad). Los rows del fetch reflejan la paridad de tipos
del SQL nuevo: `id::text`/`user_id::text` → str, `plan_data` jsonb → dict.
El skip_reason `supabase_not_initialized` se PRESERVÓ deliberadamente
(continuidad de dashboards) aunque el guard sea ahora `connection_pool`.
"""
import math
import pytest


# ---------------------------------------------------------------------------
# Helpers para construir los plan_data de prueba (rows del fetch SQL)
# ---------------------------------------------------------------------------


def _hist_entry(action_taken, block_set=False):
    """Helper para construir una entry mínima del historial."""
    return {
        "ts": "2026-05-08T12:00:00+00:00",
        "attempt": 1,
        "divergence_count": 0,
        "presence_count": 0,
        "magnitude_count": 0,
        "hypotheses": {},
        "block_set": block_set,
        "action_taken": action_taken,
    }


def _plan(history, plan_id="abc"):
    return {
        "id": plan_id,
        "user_id": "u1",
        "plan_data": {"_shopping_coherence_block_history": history},
    }


@pytest.fixture
def captured_inserts(monkeypatch):
    """Captura los INSERTS a pipeline_metrics en lugar de pegarle a la DB."""
    captured = []

    def _fake_execute_sql_write(sql, params, **kwargs):
        captured.append({"sql": sql, "params": params})
        return None

    import cron_tasks
    monkeypatch.setattr(cron_tasks, "execute_sql_write", _fake_execute_sql_write)
    return captured


@pytest.fixture
def install_plans(monkeypatch):
    """Factory: stubea el fetch SQL de meal_plans con los plans dados.

    [P1-NEON-DB-MIGRATION] `_aggregate_coherence_block_history_metrics` hace
    `from db_core import connection_pool` cada vez (no lo cachea) y llama
    `execute_sql_query` (binding módulo-level de cron_tasks). Patcheamos:
      - `db_core.connection_pool` → objeto truthy (pasa el guard de
        disponibilidad sin depender de DB real del entorno).
      - `cron_tasks.execute_sql_query` → retorna los rows (dicts con
        id/user_id str + plan_data dict, paridad del SELECT con ::text).
    """
    def _install(plans):
        import db_core
        import cron_tasks
        monkeypatch.setattr(db_core, "connection_pool", object())
        monkeypatch.setattr(
            cron_tasks,
            "execute_sql_query",
            lambda sql, params=None, **kwargs: list(plans),
        )
    return _install


# ---------------------------------------------------------------------------
# 1. Conteos por categoría de action_taken
# ---------------------------------------------------------------------------
class TestActionTakenCounts:
    def test_all_normal_actions_counted(self, install_plans, captured_inserts):
        """Las 5 categorías "esperadas" se cuentan correctamente."""
        install_plans([
            _plan([_hist_entry("not_applicable")], plan_id="p1"),
            _plan([_hist_entry("degrade", block_set=True)], plan_id="p2"),
            _plan([_hist_entry("reject_minor", block_set=True)], plan_id="p3"),
            _plan([_hist_entry("reject_high", block_set=True)], plan_id="p4"),
            _plan([_hist_entry("hydration_error", block_set=True)], plan_id="p5"),
        ])
        from cron_tasks import _aggregate_coherence_block_history_metrics
        _aggregate_coherence_block_history_metrics()
        assert len(captured_inserts) == 1
        meta = _parse_metadata(captured_inserts[0])
        c = meta["counts"]
        assert c["not_applicable"]  == 1
        assert c["degrade"]         == 1
        assert c["reject_minor"]    == 1
        assert c["reject_high"]     == 1
        assert c["hydration_error"] == 1
        # Anomalous porque hay hydration_error.
        assert meta["anomalous"] is True

    def test_normal_only_no_anomaly(self, install_plans, captured_inserts):
        """Sin invariant violations → confidence=1.0, anomalous=False."""
        install_plans([
            _plan([_hist_entry("not_applicable")], plan_id="p1"),
            _plan([_hist_entry("degrade", block_set=True)], plan_id="p2"),
        ])
        from cron_tasks import _aggregate_coherence_block_history_metrics
        _aggregate_coherence_block_history_metrics()
        assert len(captured_inserts) == 1
        # Los params son una tupla (.., confidence, metadata).
        params = captured_inserts[0]["params"]
        confidence_val = params[6]  # 7º placeholder
        assert confidence_val == 1.0
        meta = _parse_metadata(captured_inserts[0])
        assert meta["anomalous"] is False

    def test_null_block_set_invariant_violation(self, install_plans, captured_inserts):
        """[P2-2 invariant] block_set=True + action_taken=None es bug:
        debe contar como `null_block_set` y disparar anomaly gate."""
        install_plans([
            _plan([_hist_entry(None, block_set=True)], plan_id="p1"),
        ])
        from cron_tasks import _aggregate_coherence_block_history_metrics
        _aggregate_coherence_block_history_metrics()
        meta = _parse_metadata(captured_inserts[0])
        assert meta["counts"]["null_block_set"] == 1
        assert meta["anomalous"] is True
        confidence_val = captured_inserts[0]["params"][6]
        assert confidence_val == 0.0

    def test_none_other_legacy_path(self, install_plans, captured_inserts):
        """action_taken=None + block_set=False NO debe pasar bajo P2-2; si
        aparece, contar como `none_other` (regresión a investigar)."""
        install_plans([
            _plan([_hist_entry(None, block_set=False)], plan_id="p1"),
        ])
        from cron_tasks import _aggregate_coherence_block_history_metrics
        _aggregate_coherence_block_history_metrics()
        meta = _parse_metadata(captured_inserts[0])
        assert meta["counts"]["none_other"] == 1
        assert meta["anomalous"] is True

    def test_unexpected_action_value_not_counted(self, install_plans, captured_inserts):
        """Un `action_taken` con valor fuera de las 5 categorías esperadas
        NO debe inflar el dict (un typo en review_plan_node no debería
        dañar la calidad de la métrica)."""
        install_plans([
            _plan([_hist_entry("typoed_action", block_set=True)], plan_id="p1"),
            _plan([_hist_entry("degrade", block_set=True)], plan_id="p2"),
        ])
        from cron_tasks import _aggregate_coherence_block_history_metrics
        _aggregate_coherence_block_history_metrics()
        meta = _parse_metadata(captured_inserts[0])
        # No se añade key nueva al dict de counts.
        # [P2-B · 2026-05-08] +`post_swap_revalidation` añadido como bucket
        # dedicado para entries emitidas por `_recompute_aggregates_after_swap`.
        assert set(meta["counts"].keys()) == {
            "not_applicable", "degrade", "reject_minor", "reject_high",
            "hydration_error", "null_block_set", "none_other",
            "post_swap_revalidation",
            # [P2-2 · 2026-05-08] sub-bucket de post_swap_revalidation con
            # entry.alerted=True (escaló a system_alerts).
            "post_swap_critical_alerted",
            # [P3-NEXT-4 · 2026-05-11] Buckets per-surface auxiliar añadidos
            # cuando P1-NEXT-2 wirearon el helper en T2 / recalc / agent /
            # cron diario. Cada surface tiene su propio bucket para detectar
            # drift por origen.
            "warn_only_chunk_t2",
            "warn_only_recalc",
            "warn_only_agent_tool",
            "warn_only_cron_daily",
        }
        # Solo "degrade" sumó.
        assert meta["counts"]["degrade"] == 1


# ---------------------------------------------------------------------------
# 2. Resiliencia: plan_data corrupto / entries no-dict
# ---------------------------------------------------------------------------
class TestResilience:
    def test_string_plan_data_invalid_json_counted_as_parse_error(
        self, install_plans, captured_inserts
    ):
        """plan_data como string no-JSON: cuenta como parse_error y NO crashea."""
        install_plans([
            {"id": "p1", "user_id": "u1", "plan_data": "{bogus json"},
            _plan([_hist_entry("degrade", block_set=True)], plan_id="p2"),
        ])
        from cron_tasks import _aggregate_coherence_block_history_metrics
        _aggregate_coherence_block_history_metrics()
        meta = _parse_metadata(captured_inserts[0])
        assert meta["parse_errors"] == 1
        assert meta["counts"]["degrade"] == 1

    def test_history_with_non_dict_entries_skipped(self, install_plans, captured_inserts):
        """Entradas que no son dict (e.g. None) se saltan sin crash."""
        history = [None, "string", _hist_entry("degrade", block_set=True), 42]
        install_plans([_plan(history, plan_id="p1")])
        from cron_tasks import _aggregate_coherence_block_history_metrics
        _aggregate_coherence_block_history_metrics()
        meta = _parse_metadata(captured_inserts[0])
        assert meta["counts"]["degrade"] == 1
        assert meta["total_entries"] == 1  # Solo el dict válido.

    def test_no_history_field_skipped(self, install_plans, captured_inserts):
        """Plans sin `_shopping_coherence_block_history` se cuentan en
        plans_examined pero no en plans_with_history."""
        install_plans([
            {"id": "p1", "user_id": "u1", "plan_data": {}},  # no field
            _plan([_hist_entry("degrade", block_set=True)], plan_id="p2"),
        ])
        from cron_tasks import _aggregate_coherence_block_history_metrics
        _aggregate_coherence_block_history_metrics()
        meta = _parse_metadata(captured_inserts[0])
        assert meta["plans_examined"] == 2
        assert meta["plans_with_history"] == 1


# ---------------------------------------------------------------------------
# 3. Skip cuando el pool SQL no está disponible
# ---------------------------------------------------------------------------
class TestSupabaseUnavailable:
    def test_supabase_none_emits_tick_with_skip_reason(self, monkeypatch, captured_inserts):
        """[P1-CRON-TOP-LEVEL-TRY · 2026-05-15] Si el pool SQL es None, el
        cron DEBE emitir tick observable a pipeline_metrics con
        `skip_reason='supabase_not_initialized'`. Pre-fix, hacía silent skip
        — el watchdog no podía distinguir "cron OK sin anomalías" de
        "cron silenciosamente abortando por pool=None 3h seguidas".

        Post-fix, el tick siempre se emite (via finally), con skip_reason
        que discrimina los 5 paths canónicos.

        [P1-NEON-DB-MIGRATION · 2026-06-12] El guard es ahora
        `db_core.connection_pool` (no `db_core.supabase`), pero el literal
        `supabase_not_initialized` se preservó deliberadamente por
        continuidad de dashboards — NO renombrarlo.
        """
        import db_core
        monkeypatch.setattr(db_core, "connection_pool", None)
        from cron_tasks import _aggregate_coherence_block_history_metrics
        # No debe lanzar.
        _aggregate_coherence_block_history_metrics()
        # Tras P1-CRON-TOP-LEVEL-TRY: el INSERT SÍ debe emitirse, con
        # skip_reason='supabase_not_initialized' en metadata.
        assert len(captured_inserts) == 1, (
            f"P1-CRON-TOP-LEVEL-TRY: el tick observable debe emitirse aunque "
            f"connection_pool=None. captured_inserts={captured_inserts!r}"
        )
        meta = _parse_metadata(captured_inserts[0])
        assert meta.get("skip_reason") == "supabase_not_initialized", (
            f"P1-CRON-TOP-LEVEL-TRY: skip_reason debe ser "
            f"'supabase_not_initialized' (literal legacy preservado) cuando "
            f"connection_pool=None. Got: {meta.get('skip_reason')!r}"
        )


# ---------------------------------------------------------------------------
# 4. Knob lookback con defensas contra valores patológicos
# ---------------------------------------------------------------------------
class TestLookbackKnob:
    @pytest.mark.parametrize("raw,expected", [
        (None, 1.0),
        ("1", 1.0),
        ("2.5", 2.5),
        ("garbage", 1.0),
        ("-3", 1.0),     # <=0 clampa a 1.0
        ("0", 1.0),      # boundary
        ("nan", 1.0),    # NaN
        ("inf", 1.0),    # inf
    ])
    def test_lookback_knob_robust(
        self, raw, expected, install_plans, captured_inserts, monkeypatch
    ):
        if raw is None:
            monkeypatch.delenv("MEALFIT_COHERENCE_METRICS_LOOKBACK_H", raising=False)
        else:
            monkeypatch.setenv("MEALFIT_COHERENCE_METRICS_LOOKBACK_H", raw)
        install_plans([])  # vacío basta para ver el knob en metadata
        from cron_tasks import _aggregate_coherence_block_history_metrics
        _aggregate_coherence_block_history_metrics()
        # El cron emite INSERT incluso con 0 plans (mantiene serie temporal en métricas).
        meta = _parse_metadata(captured_inserts[0])
        assert meta["lookback_h"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 5. Best-effort del INSERT
# ---------------------------------------------------------------------------
class TestInsertBestEffort:
    def test_insert_failure_does_not_crash_cron(self, monkeypatch, install_plans):
        """Si execute_sql_write tira excepción, el cron NO debe propagarla
        (sino el scheduler la marca como ERROR y dispara la alerta crítica
        del listener P2-NEW-D — falso positivo). Best-effort silencioso."""
        def _failing_insert(*args, **kwargs):
            raise RuntimeError("DB connection refused")
        import cron_tasks
        monkeypatch.setattr(cron_tasks, "execute_sql_write", _failing_insert)
        install_plans([_plan([_hist_entry("degrade", block_set=True)], plan_id="p1")])
        from cron_tasks import _aggregate_coherence_block_history_metrics
        # No debe lanzar.
        _aggregate_coherence_block_history_metrics()


# ---------------------------------------------------------------------------
# Helper local
# ---------------------------------------------------------------------------
def _parse_metadata(insert_record):
    """Extrae metadata jsonb del INSERT capturado."""
    import json
    params = insert_record["params"]
    metadata_raw = params[7]  # 8º placeholder
    return json.loads(metadata_raw)
