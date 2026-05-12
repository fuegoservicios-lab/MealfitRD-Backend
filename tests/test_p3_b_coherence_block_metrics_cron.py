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
  4. Skip silencioso si supabase no está inicializado (no crash).
  5. INSERT a pipeline_metrics es best-effort (un fail de DB no debe
     enmascarar el motivo real del cron, pero tampoco crashea el scheduler).
  6. Knob `MEALFIT_COHERENCE_METRICS_LOOKBACK_H` con defensas contra
     valores patológicos (NaN/inf/<=0 → 1.0).
"""
import math
import pytest


# ---------------------------------------------------------------------------
# Helpers para construir el mock de supabase y los plan_data de prueba
# ---------------------------------------------------------------------------
class _StubExecuteResult:
    def __init__(self, data):
        self.data = data


class _StubTable:
    def __init__(self, plans):
        self._plans = plans

    def select(self, _cols):
        return self

    def gte(self, _col, _val):
        return self

    def limit(self, _n):
        return self

    def execute(self):
        return _StubExecuteResult(self._plans)


class _StubSupabase:
    def __init__(self, plans):
        self._plans = plans

    def table(self, _name):
        return _StubTable(self._plans)


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
def install_supabase(monkeypatch):
    """Factory: instala un stub de supabase con los plans dados."""
    def _install(plans):
        # `_aggregate_coherence_block_history_metrics` hace `from db_core
        # import supabase` cada vez (no lo cachea), así que monkeypatch del
        # atributo del módulo basta.
        import db_core
        monkeypatch.setattr(db_core, "supabase", _StubSupabase(plans))
    return _install


# ---------------------------------------------------------------------------
# 1. Conteos por categoría de action_taken
# ---------------------------------------------------------------------------
class TestActionTakenCounts:
    def test_all_normal_actions_counted(self, install_supabase, captured_inserts):
        """Las 5 categorías "esperadas" se cuentan correctamente."""
        install_supabase([
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

    def test_normal_only_no_anomaly(self, install_supabase, captured_inserts):
        """Sin invariant violations → confidence=1.0, anomalous=False."""
        install_supabase([
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

    def test_null_block_set_invariant_violation(self, install_supabase, captured_inserts):
        """[P2-2 invariant] block_set=True + action_taken=None es bug:
        debe contar como `null_block_set` y disparar anomaly gate."""
        install_supabase([
            _plan([_hist_entry(None, block_set=True)], plan_id="p1"),
        ])
        from cron_tasks import _aggregate_coherence_block_history_metrics
        _aggregate_coherence_block_history_metrics()
        meta = _parse_metadata(captured_inserts[0])
        assert meta["counts"]["null_block_set"] == 1
        assert meta["anomalous"] is True
        confidence_val = captured_inserts[0]["params"][6]
        assert confidence_val == 0.0

    def test_none_other_legacy_path(self, install_supabase, captured_inserts):
        """action_taken=None + block_set=False NO debe pasar bajo P2-2; si
        aparece, contar como `none_other` (regresión a investigar)."""
        install_supabase([
            _plan([_hist_entry(None, block_set=False)], plan_id="p1"),
        ])
        from cron_tasks import _aggregate_coherence_block_history_metrics
        _aggregate_coherence_block_history_metrics()
        meta = _parse_metadata(captured_inserts[0])
        assert meta["counts"]["none_other"] == 1
        assert meta["anomalous"] is True

    def test_unexpected_action_value_not_counted(self, install_supabase, captured_inserts):
        """Un `action_taken` con valor fuera de las 5 categorías esperadas
        NO debe inflar el dict (un typo en review_plan_node no debería
        dañar la calidad de la métrica)."""
        install_supabase([
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
        self, install_supabase, captured_inserts
    ):
        """plan_data como string no-JSON: cuenta como parse_error y NO crashea."""
        install_supabase([
            {"id": "p1", "user_id": "u1", "plan_data": "{bogus json"},
            _plan([_hist_entry("degrade", block_set=True)], plan_id="p2"),
        ])
        from cron_tasks import _aggregate_coherence_block_history_metrics
        _aggregate_coherence_block_history_metrics()
        meta = _parse_metadata(captured_inserts[0])
        assert meta["parse_errors"] == 1
        assert meta["counts"]["degrade"] == 1

    def test_history_with_non_dict_entries_skipped(self, install_supabase, captured_inserts):
        """Entradas que no son dict (e.g. None) se saltan sin crash."""
        history = [None, "string", _hist_entry("degrade", block_set=True), 42]
        install_supabase([_plan(history, plan_id="p1")])
        from cron_tasks import _aggregate_coherence_block_history_metrics
        _aggregate_coherence_block_history_metrics()
        meta = _parse_metadata(captured_inserts[0])
        assert meta["counts"]["degrade"] == 1
        assert meta["total_entries"] == 1  # Solo el dict válido.

    def test_no_history_field_skipped(self, install_supabase, captured_inserts):
        """Plans sin `_shopping_coherence_block_history` se cuentan en
        plans_examined pero no en plans_with_history."""
        install_supabase([
            {"id": "p1", "user_id": "u1", "plan_data": {}},  # no field
            _plan([_hist_entry("degrade", block_set=True)], plan_id="p2"),
        ])
        from cron_tasks import _aggregate_coherence_block_history_metrics
        _aggregate_coherence_block_history_metrics()
        meta = _parse_metadata(captured_inserts[0])
        assert meta["plans_examined"] == 2
        assert meta["plans_with_history"] == 1


# ---------------------------------------------------------------------------
# 3. Skip cuando supabase no está disponible
# ---------------------------------------------------------------------------
class TestSupabaseUnavailable:
    def test_supabase_none_skips_silently(self, monkeypatch, captured_inserts):
        """Si supabase es None (caso CI/local sin DB), el cron logea y skipea
        sin crashear el scheduler ni intentar INSERT a pipeline_metrics."""
        import db_core
        monkeypatch.setattr(db_core, "supabase", None)
        from cron_tasks import _aggregate_coherence_block_history_metrics
        # No debe lanzar.
        _aggregate_coherence_block_history_metrics()
        # Y NO debe intentar el INSERT (no llegamos a esa rama).
        assert captured_inserts == []


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
        self, raw, expected, install_supabase, captured_inserts, monkeypatch
    ):
        if raw is None:
            monkeypatch.delenv("MEALFIT_COHERENCE_METRICS_LOOKBACK_H", raising=False)
        else:
            monkeypatch.setenv("MEALFIT_COHERENCE_METRICS_LOOKBACK_H", raw)
        install_supabase([])  # vacío basta para ver el knob en metadata
        from cron_tasks import _aggregate_coherence_block_history_metrics
        _aggregate_coherence_block_history_metrics()
        # El cron emite INSERT incluso con 0 plans (mantiene serie temporal en métricas).
        meta = _parse_metadata(captured_inserts[0])
        assert meta["lookback_h"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 5. Best-effort del INSERT
# ---------------------------------------------------------------------------
class TestInsertBestEffort:
    def test_insert_failure_does_not_crash_cron(self, monkeypatch, install_supabase):
        """Si execute_sql_write tira excepción, el cron NO debe propagarla
        (sino el scheduler la marca como ERROR y dispara la alerta crítica
        del listener P2-NEW-D — falso positivo). Best-effort silencioso."""
        def _failing_insert(*args, **kwargs):
            raise RuntimeError("DB connection refused")
        import cron_tasks
        monkeypatch.setattr(cron_tasks, "execute_sql_write", _failing_insert)
        install_supabase([_plan([_hist_entry("degrade", block_set=True)], plan_id="p1")])
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
