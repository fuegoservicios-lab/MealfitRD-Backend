"""[P1-ARQ25-F1-CLOSE · 2026-09-02] Cierre de la Fase 1 del roadmap 2.5: los entregables que
faltaban tras P1-ARQ25-F1-LIFECYCLE.

  H6   `IngredientDemand` nombrada sobre `shopping_source_days` y sellada
       (`plan_data["_ingredient_demand"]`: huella de los días fuente + revisión conocida) desde el
       ÚNICO builder (`get_shopping_list_delta`), sin mover la lista.
  §13.2 Suite de inyección de fallos del worker del chunk 0, sin DB (stubs sobre
       `run_initial_chunk`): crash durante el LLM (reintento y agotamiento), resultado vacío,
       cancelación, worker desplazado en el fallo y en el commit, crash DESPUÉS del commit y
       ANTES del CAS (replay sin regenerar), zombie rescue con `attempts + 1` (parser),
       métrica `fencing_rejected` en cada CAS de 0 filas.
  Decisión registrada: `revision` sube por trigger de DB (`meal_plans_bump_revision_trg`), no
       en cada call site — cubre TODAS las escrituras de `plan_data` (incl. cron y restore).
"""
from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
_GL = (BACKEND / "generation_lifecycle.py").read_text(encoding="utf-8")
_SC = (BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
_CT = (BACKEND / "cron_tasks.py").read_text(encoding="utf-8")


def _import_or_skip(modname: str):
    try:
        return __import__(modname, fromlist=["_"])
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{modname} no importable: {type(e).__name__}: {str(e)[:80]}")


# ═══════════════════════════════════════════════════════════ H6 — IngredientDemand
def _plan():
    return {
        "days": [{"day": 1, "meals": [{"name": "Locrio", "ingredients": ["Pollo", "Arroz"]}]}],
        "_archived_days": [{"day": 0, "meals": [{"name": "Avena", "ingredients": [{"name": "Avena"}]}]}],
    }


def test_h6_demand_days_is_the_same_source_as_shopping_source_days():
    sc = _import_or_skip("shopping_calculator")
    p = _plan()
    assert sc.ingredient_demand_days(p) == sc.shopping_source_days(p)
    assert len(sc.ingredient_demand_days(p)) == 2, "ciclo completo: archivados + ventana viva (H6)"


def test_h6_stamp_is_idempotent_and_detects_source_change():
    sc = _import_or_skip("shopping_calculator")
    p = _plan()
    s1 = sc.stamp_ingredient_demand(p, revision=7, surface="test")
    assert p[sc.INGREDIENT_DEMAND_KEY] is s1
    assert s1["schema"] == sc.INGREDIENT_DEMAND_SCHEMA and s1["source_days"] == 2 and s1["revision"] == 7
    h1 = s1["source_hash"]
    s2 = sc.stamp_ingredient_demand(p, revision=8, surface="test")
    assert s2["source_hash"] == h1, "mismos días ⇒ misma huella aunque la revisión suba"
    assert sc.ingredient_demand_is_fresh(p) is True
    p["days"][0]["meals"][0]["ingredients"].append("Ajo")
    assert sc.ingredient_demand_is_fresh(p) is False, "cambió un ingrediente ⇒ demanda obsoleta"
    assert sc.ingredient_demand_is_fresh({"days": []}) is None, "sin sello (legacy) ⇒ None, no False"


def test_h6_stamp_never_breaks_and_revision_is_int_or_none():
    sc = _import_or_skip("shopping_calculator")
    assert sc.stamp_ingredient_demand(None) == {}
    p = _plan()
    assert sc.stamp_ingredient_demand(p, revision="7")["revision"] is None
    assert sc.stamp_ingredient_demand(p, revision=True)["revision"] is None
    assert sc.stamp_ingredient_demand(p, revision=3)["revision"] == 3


def test_h6_single_builder_stamps_before_aggregating():
    i = _SC.find("def get_shopping_list_delta(")
    assert i != -1
    body = _SC[i:i + 12000]
    k_days = body.find("days = shopping_source_days(plan_result)")
    k_stamp = body.find('stamp_ingredient_demand(plan_result, surface="get_shopping_list_delta")')
    assert -1 not in (k_days, k_stamp) and k_days < k_stamp < k_days + 400, \
        "el sello va justo después de resolver la fuente, dentro del ÚNICO builder"
    # nadie más escribe el sello (un segundo escritor = segunda definición de la demanda)
    assert _SC.count("stamp_ingredient_demand(") == 2, "definición + un solo call site"


def test_h6_all_persist_sites_go_through_the_single_builder():
    """Los sitios que escriben `aggregated_shopping_list` importan `get_shopping_list_delta`."""
    go = (BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    rp = (BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    assert "from shopping_calculator import get_shopping_list_delta" in go
    assert "get_shopping_list_delta" in rp and "get_shopping_list_delta" in _CT


# ═══════════════════════════════════════════════════════════ §13.2 — suite de fallos
class _Harness:
    """Stubs de `run_initial_chunk` sin DB. Registra CAS, errores de run, métricas y eventos."""

    def __init__(self, monkeypatch, *, pipeline, cas_results=None, cancel=False, filled=False):
        self.gl = _import_or_skip("generation_lifecycle")
        self.cas = []
        self.run_errors = []
        self.completed = 0
        self.cleared = 0
        self.metrics = []
        self.events = []
        self.sql = []
        self.pipeline_calls = 0
        results = list(cas_results or [True])

        def _cas(task_id, expected_attempts, new_status, expected_status="processing", extra_set_clauses=None):
            self.cas.append((task_id, expected_attempts, new_status, dict(extra_set_clauses or {})))
            return results.pop(0) if results else True

        stub = types.ModuleType("cron_tasks")
        stub._cas_update_chunk_status = _cas
        monkeypatch.setitem(sys.modules, "cron_tasks", stub)

        def _pipeline(*a, **k):
            self.pipeline_calls += 1
            return pipeline() if callable(pipeline) else pipeline

        go = types.ModuleType("graph_orchestrator")
        go.run_plan_pipeline = _pipeline
        monkeypatch.setitem(sys.modules, "graph_orchestrator", go)
        rp = types.ModuleType("routers.plans")
        rp._resolve_live_pantry = lambda uid, data: []
        rp._run_pantry_validation_for_initial_chunk = lambda **kw: kw["result"]
        rp._postprocess_pipeline_result = lambda **kw: {**kw["result"], "id": kw["existing_plan_id"]}
        routers_pkg = types.ModuleType("routers")
        routers_pkg.plans = rp
        monkeypatch.setitem(sys.modules, "routers", routers_pkg)
        monkeypatch.setitem(sys.modules, "routers.plans", rp)
        gl = self.gl
        monkeypatch.setattr(gl, "mark_run_completed", lambda rid: setattr(self, "completed", self.completed + 1))
        monkeypatch.setattr(gl, "clear_run_error", lambda rid: setattr(self, "cleared", self.cleared + 1))
        monkeypatch.setattr(gl, "mark_run_error", lambda rid, code, msg, completed=False: self.run_errors.append((code, completed)))
        monkeypatch.setattr(gl, "run_cancel_requested", lambda rid: cancel)
        monkeypatch.setattr(gl, "_placeholder_already_filled", lambda pid, uid: filled)
        monkeypatch.setattr(gl, "_emit_lifecycle_metric", lambda name, uid, meta=None: self.metrics.append((name, dict(meta or {}))))

        class _Pub:
            def __init__(s, rid):
                pass

            def __call__(s, ev):
                self.events.append(ev)

            def flush(s, ev):
                self.events.append(ev)

        monkeypatch.setattr(gl, "RunProgressPublisher", _Pub)
        monkeypatch.setattr("db_core.execute_sql_write", lambda q, *a, **k: (self.sql.append(q), True)[1])

    def run(self, *, pickup_attempts=0, use_chunking=False, max_attempts=None):
        task = {"id": "chunk-0", "user_id": "u1", "meal_plan_id": "plan-1", "attempts": pickup_attempts}
        snap = {"_run_id": "run-1", "raw_data": {"totalDays": 3}, "use_chunking": use_chunking, "totalDays": 3}
        if max_attempts is not None:
            snap["max_attempts"] = max_attempts
        self.gl.run_initial_chunk(task=task, snap=snap, form_data={"_plan_start_date": "2026-09-02T04:00:00+00:00"},
                                  pickup_attempts=pickup_attempts)
        return self


_OK = {"days": [{"meals": [{"name": "x"}]}]}


def _raise():
    raise RuntimeError("LLM murió a mitad")


def test_crash_during_llm_retries_with_backoff_and_keeps_run_alive(monkeypatch):
    h = _Harness(monkeypatch, pipeline=_raise).run(pickup_attempts=0)
    assert h.pipeline_calls == 1
    assert len(h.cas) == 1
    _, tok, status, extra = h.cas[0]
    assert (tok, status) == (0, "pending"), "vuelve a la cola con el token del pickup"
    assert extra.get("attempts") == "attempts + 1" and "execute_after" in extra
    assert h.run_errors == [("pipeline_error", False)], "el run registra el error sin cerrarse"
    assert h.completed == 0
    assert any(ev.get("event") == "error" and ev["data"]["terminal"] is False for ev in h.events)


def test_crash_during_llm_with_attempts_exhausted_dead_letters_and_fails_placeholder(monkeypatch):
    gl = _import_or_skip("generation_lifecycle")
    max_att = int(getattr(gl, "INITIAL_CHUNK_MAX_ATTEMPTS", 3) or 3)
    h = _Harness(monkeypatch, pipeline=_raise).run(pickup_attempts=max_att - 1)
    _, _, status, extra = h.cas[0]
    assert status == "failed" and "dead_lettered_at" in extra and "initial_pipeline_error" in extra.get("dead_letter_reason", "")
    assert h.run_errors == [("pipeline_error", True)], "run cerrado con error terminal"
    assert any("generation_status" in q and "failed" in q for q in h.sql), "el placeholder no se queda en 'generating'"
    assert any(ev.get("event") == "error" and ev["data"]["terminal"] is True for ev in h.events)


def test_empty_result_is_a_retry_not_a_dead_letter(monkeypatch):
    h = _Harness(monkeypatch, pipeline={"days": []}).run(pickup_attempts=0)
    assert h.cas[0][2] == "pending" and h.run_errors == [("empty_result", False)]


def test_cancel_requested_after_llm_marks_cancelled_and_persists_nothing(monkeypatch):
    h = _Harness(monkeypatch, pipeline=_OK, cancel=True).run(pickup_attempts=0)
    assert [c[2] for c in h.cas] == ["cancelled"]
    assert h.completed == 0 and h.run_errors == []
    assert not any("meal_plans" in q for q in h.sql), "no se escribe el plan tras cancelar"
    assert any(ev.get("event") == "error" and ev["data"]["code"] == "cancelled" for ev in h.events)


def test_displaced_worker_at_failure_does_not_touch_the_run(monkeypatch):
    h = _Harness(monkeypatch, pipeline=_raise, cas_results=[False]).run(pickup_attempts=0)
    assert h.run_errors == [], "un worker desplazado no puede anotar errores del run"
    assert ("fencing_rejected", {"plan_id": "plan-1", "site": "fail", "code": "pipeline_error"}) in h.metrics


def test_displaced_worker_at_commit_emits_fencing_metric(monkeypatch):
    h = _Harness(monkeypatch, pipeline=_OK, cas_results=[False]).run(pickup_attempts=1)
    assert h.completed == 0
    assert ("fencing_rejected", {"plan_id": "plan-1", "site": "commit"}) in h.metrics


def test_crash_after_commit_before_cas_replays_without_regenerating(monkeypatch):
    """El zombie rescue devolvió el chunk con attempts+1; el placeholder YA tiene días."""
    h = _Harness(monkeypatch, pipeline=_OK, filled=True).run(pickup_attempts=1)
    assert h.pipeline_calls == 0, "no se vuelve a gastar LLM"
    assert [c[:3] for c in h.cas] == [("chunk-0", 1, "completed")]
    assert h.completed == 1 and h.cleared == 1, "run cerrado (sin chunking) y error limpiado"
    assert any(ev.get("event") == "complete" and ev["data"].get("replayed") is True for ev in h.events)


def test_replay_with_chunking_keeps_run_alive(monkeypatch):
    h = _Harness(monkeypatch, pipeline=_OK, filled=True).run(pickup_attempts=1, use_chunking=True)
    assert h.pipeline_calls == 0 and h.completed == 0 and h.cleared == 1


def test_replay_displaced_emits_fencing_metric(monkeypatch):
    h = _Harness(monkeypatch, pipeline=_OK, filled=True, cas_results=[False]).run(pickup_attempts=2)
    assert h.completed == 0 and h.cleared == 0
    assert ("fencing_rejected", {"plan_id": "plan-1", "site": "post_commit_replay"}) in h.metrics


def test_happy_path_still_completes(monkeypatch):
    h = _Harness(monkeypatch, pipeline=_OK).run(pickup_attempts=0)
    assert h.pipeline_calls == 1 and [c[2] for c in h.cas] == ["completed"] and h.completed == 1
    assert h.metrics == []


# ── parser: zombie rescue y guard post-commit
def test_zombie_rescue_bumps_the_fencing_token_and_only_touches_dead_workers():
    i = _CT.find("# Rescate de zombies: chunks 'processing' cuyo worker MURIÓ")
    assert i != -1
    win = _CT[i:i + 3000]
    assert "SET attempts = COALESCE(attempts, 0) + 1" in win, "attempts+1 es lo que desplaza al worker viejo (I10)"
    assert "WHERE status = 'processing'" in win and "dead_lettered_at IS NULL" in win
    assert "heartbeat_at > NOW() - make_interval" in win, "solo sin heartbeat fresco (worker realmente muerto)"


def test_post_commit_guard_runs_before_the_pipeline_and_reads_days_with_user_filter():
    i = _GL.find("def run_initial_chunk(")
    body = _GL[i:]
    k_guard = body.find("if _placeholder_already_filled(plan_id, user_id):")
    k_pipe = body.find("result = run_plan_pipeline(")
    assert -1 not in (k_guard, k_pipe) and k_guard < k_pipe
    j = _GL.find("def _placeholder_already_filled(")
    assert "WHERE id = %s AND user_id = %s" in _GL[j:j + 900], "I2 también en la lectura"
    assert "jsonb_array_length(coalesce(plan_data->'days'" in _GL[j:j + 900]


def test_fencing_metric_goes_to_pipeline_metrics_best_effort():
    j = _GL.find("def _emit_lifecycle_metric(")
    assert j != -1
    body = _GL[j:j + 1200]
    assert "INSERT INTO pipeline_metrics" in body and '"__lifecycle__"' in body and 'f"arq25_{name}"' in body
    assert "except Exception" in body, "best-effort: jamás rompe el worker"


# ═══════════════════════════════════════════════════════════ decisión: revision por trigger
def test_revision_is_bumped_by_db_trigger_for_every_plan_data_write():
    mig = (BACKEND / "migrations" / "arq25_f1_lifecycle_expand_2026_09_02.sql").read_text(encoding="utf-8")
    assert re.search(r"CREATE (OR REPLACE )?TRIGGER meal_plans_bump_revision_trg", mig)
    assert "IS DISTINCT FROM" in mig, "solo sube cuando plan_data cambia de verdad"
    doc = (BACKEND / "docs" / "generation_lifecycle_2_5.md").read_text(encoding="utf-8")
    assert "por trigger" in doc and "P1-ARQ25-F1-CLOSE" in doc, "la decisión (trigger, no call sites) queda escrita"
