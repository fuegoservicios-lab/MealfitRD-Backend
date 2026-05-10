"""[P2-B · 2026-05-08] Tests de re-validación de coherencia post-swap.

Bug original (audit 2026-05-08):
  `_recompute_aggregates_after_swap` (graph_orchestrator.py) recomputa
  aggregated_shopping_list_* tras swap a `best_attempt` pero NO invocaba
  `run_shopping_coherence_guard` ni emitía entry en
  `_shopping_coherence_block_history`. Si el `best_attempt` original tenía
  divergencias en sus aggregates pre-swap, las recomputadas tras swap
  podían tener divergencias DIFERENTES — y ninguna de las dos figuraba en
  la telemetría visible al cron P3-B (`_aggregate_coherence_block_history_metrics`).

Riesgo: cron P3-B subestimaba la tasa real de divergencias en planes
entregados al usuario.

Fix:
  Al final de `_recompute_aggregates_after_swap`:
    1. Invocar `run_shopping_coherence_guard(plan_result, mode_override="warn",
       multiplier=household)` — telemetría pura, NO mutamos el flag block
       (review ya pasó, no podemos rechazar).
    2. Si hay divergencias, append entry a `_shopping_coherence_block_history`
       con `action_taken="post_swap_revalidation"`, `swap=True`, multipliers
       y hipótesis. Cap 20 entries preservado.
    3. Si guard explota → log warning, no aborta.

Cobertura:
  - Guard se invoca con mode_override="warn" (no `block` aunque env var lo diga).
  - Si hay divergencias → entry añadida con campos correctos.
  - Si NO hay divergencias → no se modifica el history.
  - Si guard lanza → warning loggeado, plan_result intacto.
  - History cap 20 preservado tras append.
  - Flag `_shopping_coherence_block` NO se muta nunca aquí (telemetría pura).
"""
from __future__ import annotations

import asyncio
import logging
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def base_final_state():
    """final_state mínimo aceptado por _recompute_aggregates_after_swap.

    Tiene `days` (necesarios para que la función no haga early return) y
    `form_data` con `user_id=None` (rama guest → aggr listas vacías, sin
    pegarle a DB).
    """
    return {
        "plan_result": {
            "days": [{"day": 1, "meals": []}],
            "calc_household_multiplier": 1.0,
        },
        "form_data": {
            "user_id": None,
            "groceryDuration": "weekly",
        },
    }


@pytest.fixture
def stub_aggregates(monkeypatch):
    """Stub get_shopping_list_delta + fetch_inventory + _build_hybrid para
    aislar la función del DB y de la lógica de cálculo. No nos interesa
    QUÉ aggrega; nos interesa QUE re-valide al final."""
    from shopping_calculator import (
        get_shopping_list_delta,
        fetch_inventory_and_consumed_for_plan,
    )
    import shopping_calculator as sc

    monkeypatch.setattr(sc, "get_shopping_list_delta", lambda *a, **kw: [])
    monkeypatch.setattr(sc, "fetch_inventory_and_consumed_for_plan", lambda *a, **kw: ({}, {}))
    monkeypatch.setattr(sc, "_build_hybrid_shopping_list", lambda a, b: a or b or [])


# ---------------------------------------------------------------------------
# 1. Guard se invoca con mode_override="warn"
# ---------------------------------------------------------------------------
def test_guard_invoked_with_warn_override(base_final_state, stub_aggregates):
    """[Contrato] post-swap revalidation NO debe usar el env var (que podría
    ser `block`) — siempre warn-mode para telemetría."""
    captured_calls = []

    def _fake_guard(plan_result, *, mode_override=None, multiplier=None):
        captured_calls.append({"mode_override": mode_override, "multiplier": multiplier})
        return []  # Sin divergencias → no entry

    import shopping_calculator as sc
    with patch.object(sc, "run_shopping_coherence_guard", _fake_guard):
        from graph_orchestrator import _recompute_aggregates_after_swap
        asyncio.run(_recompute_aggregates_after_swap(base_final_state))

    assert len(captured_calls) == 1, (
        "Guard debe invocarse exactamente 1 vez en la rama post-swap."
    )
    assert captured_calls[0]["mode_override"] == "warn", (
        "mode_override DEBE ser 'warn' — NO leer el env var (review ya pasó, "
        "no podemos rechazar; queremos solo telemetría)."
    )


# ---------------------------------------------------------------------------
# 2. Divergencias → entry añadida al history con campos correctos
# ---------------------------------------------------------------------------
def test_divergences_append_history_entry(base_final_state, stub_aggregates):
    """Si guard reporta divergencias, una entry con action_taken=
    'post_swap_revalidation' y swap=True debe añadirse al history."""
    fake_div = [
        {"food": "Pollo", "hypothesis": "cap_swallowed_modifier", "magnitude": False},
        {"food": "Arroz", "hypothesis": "yield_uncovered", "magnitude": True},
        {"food": "Tomate", "hypothesis": "cap_swallowed_modifier", "magnitude": False},
    ]

    import shopping_calculator as sc
    with patch.object(sc, "run_shopping_coherence_guard", lambda *a, **kw: fake_div):
        from graph_orchestrator import _recompute_aggregates_after_swap
        asyncio.run(_recompute_aggregates_after_swap(base_final_state))

    history = base_final_state["plan_result"].get("_shopping_coherence_block_history") or []
    assert len(history) == 1, "Una entry debe haberse añadido."
    entry = history[0]
    assert entry["action_taken"] == "post_swap_revalidation"
    assert entry["swap"] is True
    assert entry["divergence_count"] == 3
    assert entry["presence_count"] == 2  # los 2 con magnitude=False
    assert entry["magnitude_count"] == 1
    assert entry["hypotheses"] == {"cap_swallowed_modifier": 2, "yield_uncovered": 1}
    assert "ts" in entry  # ISO timestamp presente
    # block_set refleja el estado actual (False — no había flag previo)
    assert entry["block_set"] is False


# ---------------------------------------------------------------------------
# 3. Sin divergencias → history NO se modifica
# ---------------------------------------------------------------------------
def test_no_divergences_no_history_mutation(base_final_state, stub_aggregates):
    """Si guard retorna [], NO se debe añadir nada al history (evita inflar
    plan_data con entries vacías)."""
    base_final_state["plan_result"]["_shopping_coherence_block_history"] = [
        {"action_taken": "not_applicable", "ts": "2026-05-08T10:00:00+00:00"}
    ]

    import shopping_calculator as sc
    with patch.object(sc, "run_shopping_coherence_guard", lambda *a, **kw: []):
        from graph_orchestrator import _recompute_aggregates_after_swap
        asyncio.run(_recompute_aggregates_after_swap(base_final_state))

    history = base_final_state["plan_result"]["_shopping_coherence_block_history"]
    assert len(history) == 1, (
        "Sin divergencias, el history NO debe crecer (evita ruido en P3-B)."
    )
    assert history[0]["action_taken"] == "not_applicable"


# ---------------------------------------------------------------------------
# 4. Guard explota → warning loggeado, plan_result intacto
# ---------------------------------------------------------------------------
def test_guard_exception_swallowed(base_final_state, stub_aggregates, caplog):
    """Si guard lanza por bug interno, NO debe propagar — el aggregator
    ya completó su trabajo, la telemetría es best-effort."""
    def _exploding_guard(*a, **kw):
        raise RuntimeError("synthetic guard failure")

    import shopping_calculator as sc
    with patch.object(sc, "run_shopping_coherence_guard", _exploding_guard):
        from graph_orchestrator import _recompute_aggregates_after_swap
        with caplog.at_level(logging.WARNING):
            # NO debe propagar
            asyncio.run(_recompute_aggregates_after_swap(base_final_state))

    # Aggregates ya estaban seteados (asignación previa al try guard)
    assert "aggregated_shopping_list" in base_final_state["plan_result"]
    # El warning quedó loggeado para diagnóstico
    assert any("ROLLBACK-COH-REVALIDATE" in r.message for r in caplog.records), (
        "Excepción del guard debe loggearse como WARNING para que ops pueda ver el motivo."
    )


# ---------------------------------------------------------------------------
# 5. History cap 20 preservado tras append
# ---------------------------------------------------------------------------
def test_history_cap_preserved_at_20(base_final_state, stub_aggregates):
    """Si el history ya tiene 20 entries, el append debe descartar la más
    vieja para mantener el cap. P3-NEW-C cap rule."""
    # Pre-pobla 20 entries
    base_final_state["plan_result"]["_shopping_coherence_block_history"] = [
        {"action_taken": "not_applicable", "ts": f"2026-05-08T0{i // 10}:0{i % 10}:00+00:00", "_legacy_idx": i}
        for i in range(20)
    ]

    fake_div = [{"food": "X", "hypothesis": "unknown", "magnitude": False}]
    import shopping_calculator as sc
    with patch.object(sc, "run_shopping_coherence_guard", lambda *a, **kw: fake_div):
        from graph_orchestrator import _recompute_aggregates_after_swap
        asyncio.run(_recompute_aggregates_after_swap(base_final_state))

    history = base_final_state["plan_result"]["_shopping_coherence_block_history"]
    assert len(history) == 20, "Cap 20 debe respetarse."
    # La entry más vieja (idx 0) debe haber sido descartada; la nueva
    # post_swap_revalidation debe estar al final.
    assert history[-1]["action_taken"] == "post_swap_revalidation"
    assert history[-1]["swap"] is True
    # Las idx 1..19 sobreviven
    legacy_indices = [e.get("_legacy_idx") for e in history if "_legacy_idx" in e]
    assert legacy_indices == list(range(1, 20))


# ---------------------------------------------------------------------------
# 6. Flag _shopping_coherence_block NO se muta aquí
# ---------------------------------------------------------------------------
def test_block_flag_not_mutated(base_final_state, stub_aggregates):
    """Aunque haya divergencias, post-swap NO debe setear ni limpiar
    `_shopping_coherence_block`. Telemetría pura — review ya pasó."""
    # Inicia con flag=None (no setado)
    fake_div = [{"food": "Pollo", "hypothesis": "cap_swallowed_modifier", "magnitude": False}]
    import shopping_calculator as sc
    with patch.object(sc, "run_shopping_coherence_guard", lambda *a, **kw: fake_div):
        from graph_orchestrator import _recompute_aggregates_after_swap
        asyncio.run(_recompute_aggregates_after_swap(base_final_state))

    # Flag NO se setea desde post-swap
    assert "_shopping_coherence_block" not in base_final_state["plan_result"], (
        "post-swap revalidation debe ser telemetría pura — NO setear el flag "
        "block (review ya pasó, no podemos rechazar)."
    )

    # Caso 2: si el flag YA estaba set (heredado del best_attempt), tampoco
    # se debe limpiar — refleja el estado real para queries del cron.
    base_final_state["plan_result"]["_shopping_coherence_block"] = ["existing"]
    base_final_state["plan_result"].pop("_shopping_coherence_block_history", None)
    with patch.object(sc, "run_shopping_coherence_guard", lambda *a, **kw: fake_div):
        asyncio.run(_recompute_aggregates_after_swap(base_final_state))

    assert base_final_state["plan_result"].get("_shopping_coherence_block") == ["existing"], (
        "Si el flag block estaba set pre-swap, post-swap NO debe limpiarlo."
    )
    history = base_final_state["plan_result"]["_shopping_coherence_block_history"]
    assert history[-1]["block_set"] is True, (
        "El campo `block_set` de la entry refleja el estado actual del flag."
    )


# ---------------------------------------------------------------------------
# 7. P3-B cron reconoce post_swap_revalidation como bucket dedicado
# ---------------------------------------------------------------------------
def test_p3b_cron_counts_post_swap_revalidation():
    """[P2-B → P3-B integration] El nuevo valor de action_taken debe tener
    su propio bucket en el cron `_aggregate_coherence_block_history_metrics`,
    NO caer en `none_other` ni inflar `null_block_set`."""
    import importlib, cron_tasks
    importlib.reload(cron_tasks)  # asegura código fresh

    # Construye plans con history que contiene post_swap_revalidation
    plans = [{
        "id": "p1",
        "user_id": "u1",
        "plan_data": {
            "_shopping_coherence_block_history": [
                {
                    "action_taken": "post_swap_revalidation",
                    "swap": True,
                    "block_set": False,
                    "ts": "2026-05-08T12:00:00+00:00",
                    "divergence_count": 2,
                },
            ]
        },
    }]

    captured = []

    def _fake_execute_sql_write(sql, params, **kwargs):
        captured.append({"sql": sql, "params": params})

    class _StubResult:
        def __init__(self, data): self.data = data

    class _StubTable:
        def __init__(self, plans): self._plans = plans
        def select(self, _): return self
        def gte(self, *_): return self
        def limit(self, _): return self
        def execute(self): return _StubResult(self._plans)

    class _StubSupabase:
        def __init__(self, plans): self._plans = plans
        def table(self, _): return _StubTable(self._plans)

    import db_core
    with patch.object(db_core, "supabase", _StubSupabase(plans)), \
         patch.object(cron_tasks, "execute_sql_write", _fake_execute_sql_write):
        cron_tasks._aggregate_coherence_block_history_metrics()

    assert len(captured) == 1, "Cron debe haber emitido 1 INSERT a pipeline_metrics."
    import json
    metadata = json.loads(captured[0]["params"][7])  # 8º placeholder = metadata jsonb
    counts = metadata["counts"]
    assert counts.get("post_swap_revalidation") == 1, (
        f"Bucket post_swap_revalidation debe contar la entry, "
        f"got counts={counts}"
    )
    assert counts.get("none_other", 0) == 0, (
        "post_swap_revalidation NO debe caer en bucket none_other "
        "(eso confundiría telemetría con bug genuino)."
    )
    # post_swap_revalidation NO es anomalous por sí solo (es observabilidad)
    # — pero si null_block_set/hydration_error/none_other > 0, sí.
    # Aquí ninguno está set → confidence=1.0
    confidence_val = captured[0]["params"][6]
    assert confidence_val == 1.0, (
        "post_swap_revalidation solo NO debe disparar anomaly gate; "
        "es bucket de observabilidad, no de error."
    )
