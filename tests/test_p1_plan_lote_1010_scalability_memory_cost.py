"""[P1-PLAN-LOTE-1010 · 2026-11-14] Tests for G-10 (chat concurrency isolation), G-11/G-30 (superseded checkpoints + heartbeat metrics purge), G-12/G-21 (context TTL cache + context_build_ms), G-14 (fact extractor signals), G-18 (hourly burst + 3x cost spike), and G-19 (user daily USD cap)."""
from __future__ import annotations

import os
import sys
import threading
import time
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_g10_chat_concurrency_semaphore_isolated_from_plan_gen(monkeypatch):
    """El semáforo del coach (`MEALFIT_CHAT_MAX_CONCURRENT=16`) está aislado de `LLM_SEMAPHORE` del plan-gen (`4`)."""
    import agent
    import graph_orchestrator as go

    assert go.LLM_MAX_CONCURRENT == 4
    assert agent.chat_max_concurrent() == 16

    # Saturar todos los slots locales de plan-gen (`LLM_SEMAPHORE._local_semaphore`).
    sem_pg = go.LLM_SEMAPHORE._local_semaphore
    acquired = []
    for _ in range(go.LLM_MAX_CONCURRENT):
        assert sem_pg.acquire(timeout=0.5)
        acquired.append(1)

    try:
        completed = []
        errors = []
        barrier = threading.Barrier(3, timeout=2.0)

        def _worker(idx: int):
            try:
                with agent._acquire_chat_llm_slot():
                    barrier.wait()
                    completed.append(idx)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3.0)

        assert not errors
        assert sorted(completed) == [0, 1, 2]
    finally:
        for _ in acquired:
            sem_pg.release()


def test_g11_g30_sweep_superseded_checkpoints_and_metrics_gc(monkeypatch):
    """Poda segura de checkpoints superados + purga de heartbeats sin sufijo `_tick` en `pipeline_metrics`."""
    import cron_tasks
    import db
    import db_chat
    import db_core

    exec_calls: list[tuple[str, tuple]] = []

    class _FakeCursor:
        def __init__(self):
            self.rowcount = 0
            self._last_sql = ""

        def execute(self, sql, params=()):
            norm = " ".join(str(sql).split())
            self._last_sql = norm
            exec_calls.append((norm, params))
            if "DELETE FROM public.checkpoint_writes" in norm:
                self.rowcount = 4
            elif "DELETE FROM public.checkpoints" in norm:
                self.rowcount = 2
            elif "DELETE FROM public.checkpoint_blobs" in norm:
                self.rowcount = 3
            else:
                self.rowcount = 0

        def fetchall(self):
            if "FROM ranked" in self._last_sql:
                return [
                    {"thread_id": "t-1", "checkpoint_ns": "", "checkpoint_id": "ck-1"},
                    {"thread_id": "t-1", "checkpoint_ns": "", "checkpoint_id": "ck-2"},
                ]
            return []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _FakeTx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _FakeConn:
        def transaction(self):
            return _FakeTx()

        def cursor(self, row_factory=None):
            return _FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _FakePool:
        def connection(self):
            return _FakeConn()

    monkeypatch.setattr(db_chat, "connection_pool", _FakePool())
    monkeypatch.setattr(db_core, "_guard_test_write_to_prod", None)

    res = db_chat.sweep_superseded_thread_checkpoints(keep_latest_per_thread=2, min_age_hours=1, batch_rows=500)
    assert res == {"checkpoints": 2, "checkpoint_writes": 4, "checkpoint_blobs": 3}
    assert any("channel_versions" in sql and "NOT EXISTS" in sql for sql, _ in exec_calls)

    # Verifica `pipeline_metrics` GC ampliado + hook en `_sweep_orphan_chat_checkpoints`
    writes: list[tuple[str, tuple]] = []
    monkeypatch.setattr(
        db_core,
        "execute_sql_write",
        lambda sql, params=(), **k: writes.append((" ".join(sql.split()), params)) or ([{"id": 1}] if k.get("returning") else True),
    )
    cron_tasks._purge_old_pipeline_metrics()
    assert any(
        "llm_breaker_open_watch" in sql and "auth_jwt_verify_fail" in sql and "node LIKE '%%_tick'" in sql
        for sql, _ in writes
    )

    called = {}
    monkeypatch.setattr(db, "sweep_orphan_chat_checkpoints", lambda d, b: {"threads": ["t-orph"]})
    monkeypatch.setattr(
        db,
        "sweep_superseded_thread_checkpoints",
        lambda k, a, b: called.update(keep=k, age=a, batch=b) or {"checkpoints": 5, "checkpoint_writes": 1, "checkpoint_blobs": 2},
    )
    assert cron_tasks._sweep_orphan_chat_checkpoints() == 1
    assert called == {"keep": 2, "age": 1, "batch": 500}


def test_g14_fact_extractor_signals_routine_and_goal_recall(monkeypatch):
    """`should_extract_facts` deja pasar rutinas/gym/metas/cocina al router y filtra comidas puntuales."""
    import fact_extractor as fe

    monkeypatch.setattr(fe, "connection_pool", None, raising=False)
    # Comidas aisladas sin señales duraderas -> fast-path False (sin tocar LLM)
    assert fe.should_extract_facts("Desayuné 2 huevos con mangú") is False
    assert fe.should_extract_facts("Almorcé pollo guisado con arroz") is False

    # Stub ChatGLM para medir qué mensajes pasan el fast-path y llegan al router
    router_calls: list[str] = []

    class _RouterStub:
        def with_structured_output(self, *a, **k):
            return self

        def invoke(self, prompt_text):
            router_calls.append(str(prompt_text))
            return {"parsed": fe.RouterResult(has_relevant_info=True, confidence_score=9), "raw": None}

    monkeypatch.setattr(fe, "ChatGLM", lambda *a, **k: _RouterStub())

    # Mensaje que empieza como comida pero incluye rutina de entrenamiento + déficit -> debe evaluar True
    assert fe.should_extract_facts("Desayuné avena porque entreno gym a las 6am y estoy en déficit calórico") is True
    assert fe.should_extract_facts("No tengo horno, solo tengo airfryer para cocinar en casa") is True
    assert len(router_calls) == 2


def test_g18_hourly_fallback_burst_and_cost_spike_alerts():
    """`include_hourly_and_cost=True` detecta ráfaga 1h de respaldo y pico de coste >3x con piso $0.015."""
    import chat_observability_alerts as coa

    writes: list[tuple[str, tuple]] = []

    def _q_burst(sql, params=(), **kwargs):
        if "INTERVAL '1 hour'" in sql and "chat_turn_phases" in sql:
            return {"turns_1h": 10, "fallback_1h": 5}
        if "llm_usage_events" in sql:
            return {"calls_1h": 6, "avg_1h_usd": 0.025, "calls_7d": 100, "avg_7d_usd": 0.005}
        if "chat_turn_phases" in sql:
            return {"turns": 20, "error_turns": 0, "fallback_turns": 0}
        return {"votes": 10, "down_votes": 1}

    res = coa.evaluar_salud_coach_y_alertar(
        execute_sql_query=_q_burst,
        execute_sql_write=lambda sql, params=(), **k: writes.append((" ".join(sql.split()), params)) or True,
        include_hourly_and_cost=True,
    )
    assert res["fallback_hourly_burst"]["alerted"] is True
    assert res["cost_spike_3x"]["alerted"] is True
    alerted_keys = {p[0] for s, p in writes if "INSERT INTO system_alerts" in s}
    assert {"chat_fallback_hourly_burst", "chat_cost_spike_3x"} <= alerted_keys

    # Con coste < piso $0.015 (ej. $0.012 aunque >3x $0.003) NO dispara pico de costo
    writes2: list[tuple[str, tuple]] = []

    def _q_safe(sql, params=(), **kwargs):
        if "INTERVAL '1 hour'" in sql and "chat_turn_phases" in sql:
            return {"turns_1h": 10, "fallback_1h": 1}
        if "llm_usage_events" in sql:
            return {"calls_1h": 6, "avg_1h_usd": 0.012, "calls_7d": 100, "avg_7d_usd": 0.003}
        if "chat_turn_phases" in sql:
            return {"turns": 20, "error_turns": 0, "fallback_turns": 0}
        return {"votes": 10, "down_votes": 1}

    res2 = coa.evaluar_salud_coach_y_alertar(
        execute_sql_query=_q_safe,
        execute_sql_write=lambda sql, params=(), **k: writes2.append((" ".join(sql.split()), params)) or True,
        include_hourly_and_cost=True,
    )
    assert res2["fallback_hourly_burst"]["alerted"] is False
    assert res2["cost_spike_3x"]["alerted"] is False


def test_g19_user_daily_chat_budget_soft_and_hard_caps(monkeypatch):
    """Tope diario por usuario USD: ok (<80%), soft_80 (sin Terra), y hard_100 (respuesta cortés sin LLM)."""
    import agent
    import chat_hard_turn as cht

    monkeypatch.setenv("MEALFIT_CHAT_USER_DAILY_BUDGET_USD", "0.25")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    monkeypatch.setenv("MEALFIT_CHAT_AGENT_MODEL", "gpt-6-luna")

    r_ok = cht.check_user_daily_chat_budget("u-1", execute_sql_query=lambda *a, **k: {"spent_micros": 100_000})
    assert r_ok["status"] == "ok" and r_ok["allow_escalation"] is True and r_ok["block_turn"] is False

    r_soft = cht.check_user_daily_chat_budget("u-1", execute_sql_query=lambda *a, **k: {"spent_micros": 210_000})
    assert r_soft["status"] == "soft_80" and r_soft["allow_escalation"] is False and r_soft["block_turn"] is False

    r_hard = cht.check_user_daily_chat_budget("u-1", execute_sql_query=lambda *a, **k: {"spent_micros": 260_000})
    assert r_hard["status"] == "hard_100" and r_hard["block_turn"] is True

    # `evaluar_turno_dificil` bloquea escalado a Terra cuando `chat_budget_soft_degraded=True`
    hard_state = {
        "form_data": {"allergies": "maní, gluten", "medicalConditions": "diabetes"},
        "messages": [HumanMessage(content="¿Puedo comer avena con mis alergias?")],
        "chat_budget_soft_degraded": True,
    }
    eff_model, info = cht.evaluar_turno_dificil(hard_state, hard_state["messages"], "gpt-6-luna", [])
    assert eff_model == "gpt-6-luna" and info["escalated"] is False and info["reason"] == "user_daily_budget_soft_cap"

    # `chat_with_agent` y `chat_with_agent_stream` responden cortésmente en hard_100 sin tocar LangGraph
    monkeypatch.setattr(cht, "check_user_daily_chat_budget", lambda uid: dict(r_hard))
    monkeypatch.setattr(agent.chat_builder, "compile", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no graph")))

    reply, upd, nplan = agent.chat_with_agent("s-cap", "hola", user_id="u-1", form_data={"locale": "es-DO"})
    assert "límite de consultas" in reply and upd == {} and nplan is None   # [P1-PLAN-LOTE-1016] ventana de 24 h, ya no «diario»

    events = list(agent.chat_with_agent_stream("s-cap", "hola", user_id="u-1", form_data={"locale": "en-US"}))
    assert any('"type": "chunk"' in ev and "usage limit" in ev.lower() for ev in events)
    assert any('"type": "done"' in ev for ev in events)


def test_g12_g21_context_cache_invalidation_and_context_build_ms(monkeypatch):
    """Caché de contexto con TTL=30s acierta entre lecturas, invalida al mutar, y `context_build_ms` llega a métricas."""
    import agent
    import chat_phase_metrics as cpm

    monkeypatch.setenv("MEALFIT_CHAT_CTX_CACHE_TTL_S", "30")
    agent.invalidate_chat_user_ctx_cache(None)

    suppl_calls = []

    import suplementos
    monkeypatch.setattr(suplementos, "bloque_para_chat", lambda uid: suppl_calls.append(uid) or f"SUP_{len(suppl_calls)}")

    # Turn cache vacío para probar el caché entre turnos de `agent.py`
    first = agent._safe_suplementos_bloque_para_chat("u-cache")
    second = agent._safe_suplementos_bloque_para_chat("u-cache")
    assert first == "SUP_1" and second == "SUP_1" and len(suppl_calls) == 1

    # Mutación vía `execute_tools` (no read-only) invalida el caché del usuario
    class _FakeWater:
        name = "log_water_glass"
        def invoke(self, args):
            return "Listo: 1 vaso."

    monkeypatch.setattr(agent, "agent_tools", [_FakeWater()])
    monkeypatch.setenv("MEALFIT_CHAT_FAST_TOOL_CONFIRM_ENABLED", "0")
    ai_tc = AIMessage(content="", tool_calls=[{"id": "w1", "name": "log_water_glass", "args": {"user_id": "u-cache", "count_delta": 1}}])
    agent.execute_tools({"user_id": "u-cache", "session_id": "s1", "sys_prompt": "x", "messages": [HumanMessage(content="agua"), ai_tc]})

    third = agent._safe_suplementos_bloque_para_chat("u-cache")
    assert third == "SUP_2" and len(suppl_calls) == 2

    # `context_build_ms` en ChatTurnPhaseTracker y `query_chat_phase_percentiles`
    tracker = cpm.ChatTurnPhaseTracker(user_id="u1", session_id="s1", corr="corr-ctx")
    tracker.set_phase_ms("context_build_ms", 42)
    assert tracker.to_metadata()["context_build_ms"] == 42

    captured_sqls = []
    import db_core
    monkeypatch.setattr(
        db_core,
        "execute_sql_query",
        lambda sql, params=(), **k: captured_sqls.append(" ".join(sql.split())) or {"turns": 1, "samples": 0},
    )
    cpm.query_chat_phase_percentiles(24)
    assert any("context_build_p50_ms" in s and "context_build_p90_ms" in s for s in captured_sqls)
