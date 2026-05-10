"""[P2-2 · 2026-05-08] Tests de la alerta inline post-swap.

Bug observado en el audit 2026-05-08:
  P2-B (2026-05-08) cerró el gap de telemetría — `_recompute_aggregates_after_swap`
  emite entries en `_shopping_coherence_block_history` con
  `action_taken="post_swap_revalidation"` cuando el guard en mode=warn detecta
  divergencias post-swap. Pero el cron P3-B trata ese bucket como observabilidad
  pura (NO anomalous), así que un plan entregado al usuario con divergencias
  CRÍTICAS (cap_swallowed_modifier o magnitudes >30%) no levantaba alerta.

Fix:
  - Helper `_emit_post_swap_coherence_alert` en graph_orchestrator.py inserta
    a `system_alerts` cuando `critical_count >= MEALFIT_POST_SWAP_DIVERGENCE_ALERT_THRESHOLD`.
  - Knobs: ENABLED (kill switch), THRESHOLD (default 3), COOLDOWN_HOURS (default 6).
  - Entry de history gana flag `alerted: bool` para correlación con cron.
  - Cron P3-B suma sub-bucket `post_swap_critical_alerted` para cross-check.

Cobertura:
  1. Threshold no alcanzado → no INSERT, alerted=False.
  2. Threshold alcanzado + critical_cap → INSERT + alerted=True.
  3. Threshold alcanzado por magnitud >30% (sin cap) → INSERT.
  4. Magnitud <30% NO cuenta como crítica.
  5. Kill switch ENABLED=False → no INSERT aunque threshold alcanzado.
  6. Cooldown activo → no re-INSERT.
  7. DB error en INSERT → alerted=False, no aborta caller.
  8. Entry gana campo `critical_count` para diagnóstico.
  9. Helper retorna False si user_id+plan_id ambos None y crítico (alert_key="unknown").
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def base_final_state():
    """final_state mínimo aceptado por _recompute_aggregates_after_swap.

    user_id=None (rama guest → aggr listas vacías). El alert_key cae a
    plan_id-prefix → "unknown" si plan_id también None.
    """
    return {
        "plan_result": {
            "days": [{"day": 1, "meals": []}],
            "calc_household_multiplier": 1.0,
            "id": "test-plan-uuid-12345",
        },
        "form_data": {
            "user_id": "user-abc-123",
            "groceryDuration": "weekly",
        },
    }


@pytest.fixture
def stub_aggregates(monkeypatch):
    """Stub get_shopping_list_delta + fetch + hybrid (pegan a DB)."""
    import shopping_calculator as sc
    monkeypatch.setattr(sc, "get_shopping_list_delta", lambda *a, **kw: [])
    monkeypatch.setattr(
        sc, "fetch_inventory_and_consumed_for_plan", lambda *a, **kw: ({}, {})
    )
    monkeypatch.setattr(sc, "_build_hybrid_shopping_list", lambda a, b: a or b or [])


@pytest.fixture
def reset_post_swap_knobs(monkeypatch):
    """Reset env vars de P2-2 para que cada test parta de defaults limpios."""
    for k in (
        "MEALFIT_POST_SWAP_DIVERGENCE_ALERT_ENABLED",
        "MEALFIT_POST_SWAP_DIVERGENCE_ALERT_THRESHOLD",
        "MEALFIT_POST_SWAP_ALERT_COOLDOWN_HOURS",
    ):
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------------------
# 1. Threshold no alcanzado → no INSERT, alerted=False
# ---------------------------------------------------------------------------
def test_below_threshold_no_alert(base_final_state, stub_aggregates, reset_post_swap_knobs):
    """2 cap_swallowed_modifier < threshold(3) → no escala."""
    fake_div = [
        {"food": "Pollo", "hypothesis": "cap_swallowed_modifier", "magnitude": False},
        {"food": "Tomate", "hypothesis": "cap_swallowed_modifier", "magnitude": False},
        {"food": "Lechuga", "hypothesis": "yield_uncovered", "magnitude": True, "delta_pct": 0.05},
    ]
    captured_sql = []

    def _fake_write(sql, params=None):
        captured_sql.append((sql, params))

    import shopping_calculator as sc
    import graph_orchestrator as go

    with patch.object(sc, "run_shopping_coherence_guard", lambda *a, **kw: fake_div):
        with patch.object(go, "execute_sql_write", _fake_write):
            with patch.object(go, "execute_sql_query", lambda *a, **kw: None):
                asyncio.run(go._recompute_aggregates_after_swap(base_final_state))

    history = base_final_state["plan_result"]["_shopping_coherence_block_history"]
    assert len(history) == 1
    assert history[-1]["alerted"] is False
    assert history[-1]["critical_count"] == 2
    # Sin INSERT a system_alerts (sólo divergencias benignas + 2 críticas).
    sa_inserts = [s for s, _ in captured_sql if "system_alerts" in s]
    assert sa_inserts == [], f"No esperaba INSERT a system_alerts; got: {sa_inserts}"


# ---------------------------------------------------------------------------
# 2. Threshold alcanzado por cap_swallowed → INSERT + alerted=True
# ---------------------------------------------------------------------------
def test_threshold_reached_by_cap_swallowed_inserts_alert(
    base_final_state, stub_aggregates, reset_post_swap_knobs
):
    """3 cap_swallowed_modifier ≥ threshold(3) → INSERT + alerted=True."""
    fake_div = [
        {"food": "Pollo", "hypothesis": "cap_swallowed_modifier", "magnitude": False},
        {"food": "Tomate", "hypothesis": "cap_swallowed_modifier", "magnitude": False},
        {"food": "Cebolla", "hypothesis": "cap_swallowed_modifier", "magnitude": False},
    ]
    captured_sql = []

    def _fake_write(sql, params=None):
        captured_sql.append((sql, params))

    import shopping_calculator as sc
    import graph_orchestrator as go

    with patch.object(sc, "run_shopping_coherence_guard", lambda *a, **kw: fake_div):
        with patch.object(go, "execute_sql_write", _fake_write):
            with patch.object(go, "execute_sql_query", lambda *a, **kw: None):
                asyncio.run(go._recompute_aggregates_after_swap(base_final_state))

    history = base_final_state["plan_result"]["_shopping_coherence_block_history"]
    assert history[-1]["alerted"] is True
    assert history[-1]["critical_count"] == 3
    sa_inserts = [(s, p) for s, p in captured_sql if "system_alerts" in s]
    assert len(sa_inserts) == 1, f"Esperaba 1 INSERT system_alerts; got: {len(sa_inserts)}"
    sql, params = sa_inserts[0]
    assert "post_swap_coherence" in sql
    assert "ON CONFLICT" in sql
    # alert_key debe contener el user_id (no "unknown" — tenemos user_id).
    assert "user-abc-123" in params[0], (
        f"alert_key debe per-user para dedupe; got: {params[0]}"
    )


# ---------------------------------------------------------------------------
# 3. Threshold alcanzado por magnitudes >30% → INSERT
# ---------------------------------------------------------------------------
def test_threshold_reached_by_severe_magnitudes_inserts_alert(
    base_final_state, stub_aggregates, reset_post_swap_knobs
):
    """3 magnitudes con |delta|>0.30 ≥ threshold(3) → INSERT."""
    fake_div = [
        {"food": "Arroz", "hypothesis": "yield_uncovered", "magnitude": True, "delta_pct": 0.45},
        {"food": "Frijol", "hypothesis": "pantry_overdeduct", "magnitude": True, "delta_pct": -0.55},
        {"food": "Aceite", "hypothesis": "unit_mismatch", "magnitude": True, "delta_pct": 0.32},
    ]
    captured_sql = []

    def _fake_write(sql, params=None):
        captured_sql.append((sql, params))

    import shopping_calculator as sc
    import graph_orchestrator as go

    with patch.object(sc, "run_shopping_coherence_guard", lambda *a, **kw: fake_div):
        with patch.object(go, "execute_sql_write", _fake_write):
            with patch.object(go, "execute_sql_query", lambda *a, **kw: None):
                asyncio.run(go._recompute_aggregates_after_swap(base_final_state))

    history = base_final_state["plan_result"]["_shopping_coherence_block_history"]
    assert history[-1]["alerted"] is True
    assert history[-1]["critical_count"] == 3


# ---------------------------------------------------------------------------
# 4. Magnitudes <30% NO cuentan como críticas
# ---------------------------------------------------------------------------
def test_minor_magnitudes_below_30pct_not_critical(
    base_final_state, stub_aggregates, reset_post_swap_knobs
):
    """5 magnitudes con |delta|≤0.30 son ruido, NO escalan aunque sean ≥3."""
    fake_div = [
        {"food": "Arroz", "hypothesis": "yield_uncovered", "magnitude": True, "delta_pct": 0.10},
        {"food": "Frijol", "hypothesis": "yield_uncovered", "magnitude": True, "delta_pct": 0.15},
        {"food": "Aceite", "hypothesis": "yield_uncovered", "magnitude": True, "delta_pct": 0.20},
        {"food": "Cebolla", "hypothesis": "yield_uncovered", "magnitude": True, "delta_pct": 0.28},
        {"food": "Sal", "hypothesis": "yield_uncovered", "magnitude": True, "delta_pct": -0.30},
    ]
    captured_sql = []

    import shopping_calculator as sc
    import graph_orchestrator as go

    with patch.object(sc, "run_shopping_coherence_guard", lambda *a, **kw: fake_div):
        with patch.object(go, "execute_sql_write", lambda *a, **kw: captured_sql.append(a)):
            with patch.object(go, "execute_sql_query", lambda *a, **kw: None):
                asyncio.run(go._recompute_aggregates_after_swap(base_final_state))

    history = base_final_state["plan_result"]["_shopping_coherence_block_history"]
    assert history[-1]["alerted"] is False
    # 0.30 (boundary) NO cuenta como crítica (estrictamente >0.30).
    assert history[-1]["critical_count"] == 0
    sa_inserts = [a for a in captured_sql if a and len(a) >= 1 and "system_alerts" in str(a[0])]
    assert sa_inserts == []


# ---------------------------------------------------------------------------
# 5. Kill switch desactiva la alerta
# ---------------------------------------------------------------------------
def test_kill_switch_disables_alert(
    base_final_state, stub_aggregates, reset_post_swap_knobs, monkeypatch
):
    """ENABLED=False → no escala aunque threshold alcanzado."""
    monkeypatch.setenv("MEALFIT_POST_SWAP_DIVERGENCE_ALERT_ENABLED", "false")

    fake_div = [
        {"food": "Pollo", "hypothesis": "cap_swallowed_modifier", "magnitude": False},
        {"food": "Tomate", "hypothesis": "cap_swallowed_modifier", "magnitude": False},
        {"food": "Cebolla", "hypothesis": "cap_swallowed_modifier", "magnitude": False},
        {"food": "Ajo", "hypothesis": "cap_swallowed_modifier", "magnitude": False},
    ]
    captured_sql = []

    import shopping_calculator as sc
    import graph_orchestrator as go

    with patch.object(sc, "run_shopping_coherence_guard", lambda *a, **kw: fake_div):
        with patch.object(go, "execute_sql_write", lambda *a, **kw: captured_sql.append(a)):
            with patch.object(go, "execute_sql_query", lambda *a, **kw: None):
                asyncio.run(go._recompute_aggregates_after_swap(base_final_state))

    history = base_final_state["plan_result"]["_shopping_coherence_block_history"]
    assert history[-1]["alerted"] is False
    assert history[-1]["critical_count"] == 4  # contar sigue, solo no escala
    sa_inserts = [a for a in captured_sql if a and "system_alerts" in str(a[0])]
    assert sa_inserts == []


# ---------------------------------------------------------------------------
# 6. Cooldown activo previene re-INSERT
# ---------------------------------------------------------------------------
def test_cooldown_skips_alert(
    base_final_state, stub_aggregates, reset_post_swap_knobs
):
    """Si ya hay alerta abierta dentro del cooldown window, skip INSERT."""
    fake_div = [
        {"food": "Pollo", "hypothesis": "cap_swallowed_modifier", "magnitude": False},
        {"food": "Tomate", "hypothesis": "cap_swallowed_modifier", "magnitude": False},
        {"food": "Cebolla", "hypothesis": "cap_swallowed_modifier", "magnitude": False},
    ]
    captured_sql = []

    # Mock: existing query retorna fila → cooldown activo.
    def _fake_query(sql, params=None, fetch_one=False):
        return (1,)  # truthy → existing alert

    import shopping_calculator as sc
    import graph_orchestrator as go

    with patch.object(sc, "run_shopping_coherence_guard", lambda *a, **kw: fake_div):
        with patch.object(go, "execute_sql_write", lambda *a, **kw: captured_sql.append(a)):
            with patch.object(go, "execute_sql_query", _fake_query):
                asyncio.run(go._recompute_aggregates_after_swap(base_final_state))

    history = base_final_state["plan_result"]["_shopping_coherence_block_history"]
    assert history[-1]["alerted"] is False  # skip por cooldown
    sa_inserts = [a for a in captured_sql if a and "system_alerts" in str(a[0])]
    assert sa_inserts == []


# ---------------------------------------------------------------------------
# 7. DB error en INSERT → alerted=False, no aborta el caller
# ---------------------------------------------------------------------------
def test_db_insert_failure_returns_false_no_abort(
    base_final_state, stub_aggregates, reset_post_swap_knobs
):
    """INSERT falla → alerted=False, history persiste, función no aborta."""
    fake_div = [
        {"food": "Pollo", "hypothesis": "cap_swallowed_modifier", "magnitude": False},
        {"food": "Tomate", "hypothesis": "cap_swallowed_modifier", "magnitude": False},
        {"food": "Cebolla", "hypothesis": "cap_swallowed_modifier", "magnitude": False},
    ]

    def _exploding_write(sql, params=None):
        if "system_alerts" in sql:
            raise RuntimeError("DB caída")
        # otros writes (no debería haber otros aquí) pasan

    import shopping_calculator as sc
    import graph_orchestrator as go

    with patch.object(sc, "run_shopping_coherence_guard", lambda *a, **kw: fake_div):
        with patch.object(go, "execute_sql_write", _exploding_write):
            with patch.object(go, "execute_sql_query", lambda *a, **kw: None):
                # No debe propagar
                asyncio.run(go._recompute_aggregates_after_swap(base_final_state))

    history = base_final_state["plan_result"]["_shopping_coherence_block_history"]
    assert len(history) == 1
    assert history[-1]["alerted"] is False  # error swallowed → False
    assert history[-1]["action_taken"] == "post_swap_revalidation"


# ---------------------------------------------------------------------------
# 8. Knobs P2-2 registrados en _KNOBS_REGISTRY tras llamada
# ---------------------------------------------------------------------------
def test_p2_2_knobs_registered():
    """ENABLED, THRESHOLD, COOLDOWN_HOURS deben aparecer en `_KNOBS_REGISTRY`
    tras invocar el helper directamente."""
    import graph_orchestrator as go
    from knobs import get_knobs_registry_snapshot

    # Trigger directo del helper con threshold no alcanzado (no INSERT).
    go._emit_post_swap_coherence_alert(
        user_id="user-test",
        plan_id="plan-test",
        divergences=[],
        hyp_counter={},
        critical_total=0,
        household=1.0,
    )
    snap = get_knobs_registry_snapshot()
    expected = {
        "MEALFIT_POST_SWAP_DIVERGENCE_ALERT_ENABLED": "bool",
        "MEALFIT_POST_SWAP_DIVERGENCE_ALERT_THRESHOLD": "int",
        # COOLDOWN_HOURS solo se lee si threshold alcanzado, así que
        # con critical_total=0 no aparece. Probamos por separado abajo.
    }
    for name, type_label in expected.items():
        assert name in snap, f"Knob {name} no registrado en _KNOBS_REGISTRY"
        assert snap[name]["type"] == type_label


def test_p2_2_cooldown_knob_registered_when_threshold_hit():
    """COOLDOWN_HOURS sólo se lee cuando threshold alcanzado — verificar
    que aparece en registry tras un trigger crítico."""
    import graph_orchestrator as go
    from knobs import get_knobs_registry_snapshot

    # Mock SQL para evitar DB real; threshold default=3, mandamos critical_total=5.
    import unittest.mock as mock
    with mock.patch.object(go, "execute_sql_query", return_value=None):
        with mock.patch.object(go, "execute_sql_write", lambda *a, **kw: None):
            go._emit_post_swap_coherence_alert(
                user_id="user-test-cd",
                plan_id="plan-test-cd",
                divergences=[
                    {"food": "X", "hypothesis": "cap_swallowed_modifier", "magnitude": False}
                ] * 5,
                hyp_counter={"cap_swallowed_modifier": 5},
                critical_total=5,
                household=1.0,
            )

    snap = get_knobs_registry_snapshot()
    assert "MEALFIT_POST_SWAP_ALERT_COOLDOWN_HOURS" in snap
    assert snap["MEALFIT_POST_SWAP_ALERT_COOLDOWN_HOURS"]["type"] == "int"


# ---------------------------------------------------------------------------
# 9. Helper retorna False si todos los inputs son None y crítico
# ---------------------------------------------------------------------------
def test_helper_handles_none_user_and_plan_with_unknown_key():
    """user_id=None + plan_id=None → alert_key cae a 'unknown'. Sigue
    emitiendo (no es un error), sólo perdemos granularidad de dedupe."""
    import graph_orchestrator as go
    import unittest.mock as mock

    captured_params = []
    with mock.patch.object(go, "execute_sql_query", return_value=None):
        with mock.patch.object(
            go, "execute_sql_write",
            lambda sql, params=None: captured_params.append(params),
        ):
            ok = go._emit_post_swap_coherence_alert(
                user_id=None,
                plan_id=None,
                divergences=[
                    {"food": "X", "hypothesis": "cap_swallowed_modifier", "magnitude": False}
                ] * 5,
                hyp_counter={"cap_swallowed_modifier": 5},
                critical_total=5,
                household=1.0,
            )

    assert ok is True
    assert len(captured_params) == 1
    alert_key = captured_params[0][0]
    assert alert_key == "post_swap_critical_divergence_unknown"


# ---------------------------------------------------------------------------
# 10. Cron P3-B suma sub-bucket post_swap_critical_alerted
# ---------------------------------------------------------------------------
def test_cron_p3b_counts_alerted_subbucket():
    """Una entry con action_taken=post_swap_revalidation + alerted=True debe
    incrementar BOTH `post_swap_revalidation` (legacy bucket) AND
    `post_swap_critical_alerted` (sub-bucket P2-2)."""
    import unittest.mock as mock
    import cron_tasks

    # Stub el supabase fetch: 1 plan con 1 entry alerted=True + 1 entry alerted=False.
    plan_data = {
        "_shopping_coherence_block_history": [
            {"action_taken": "post_swap_revalidation", "alerted": True, "block_set": False},
            {"action_taken": "post_swap_revalidation", "alerted": False, "block_set": False},
            {"action_taken": "not_applicable", "block_set": False},
        ]
    }

    fake_response = mock.MagicMock()
    fake_response.data = [{"id": "plan-1", "user_id": "u-1", "plan_data": plan_data}]
    fake_table = mock.MagicMock()
    fake_table.select.return_value.gte.return_value.limit.return_value.execute.return_value = fake_response

    fake_supabase = mock.MagicMock()
    fake_supabase.table.return_value = fake_table

    # Capturar metadata del INSERT a pipeline_metrics para inspeccionar counts.
    captured_metadata = []

    def _fake_write(sql, params=None):
        if "pipeline_metrics" in sql and params:
            captured_metadata.append(params[-1])  # último param = metadata json

    # `_aggregate_coherence_block_history_metrics` hace `from db_core import supabase`
    # localmente; mockeamos ahí, no en cron_tasks.
    with mock.patch("db_core.supabase", fake_supabase):
        with mock.patch("cron_tasks.execute_sql_write", _fake_write):
            cron_tasks._aggregate_coherence_block_history_metrics()

    assert len(captured_metadata) == 1
    import json as _json
    md = _json.loads(captured_metadata[0])
    counts = md["counts"]
    assert counts["post_swap_revalidation"] == 2, (
        f"Esperaba 2 entries post_swap_revalidation; got {counts['post_swap_revalidation']}"
    )
    assert counts["post_swap_critical_alerted"] == 1, (
        f"Esperaba 1 entry alerted=True; got {counts['post_swap_critical_alerted']}"
    )
    assert counts["not_applicable"] == 1
