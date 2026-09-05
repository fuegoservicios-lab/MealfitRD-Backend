"""[P1-ARQ25-F5-PLAN-JOBS · 2026-09-04] Fase 5, rebanada 1: worker del outbox `plan_jobs` + consumidor
`display_i18n`. Ancla el protocolo (claim `FOR UPDATE SKIP LOCKED`, `attempts` como fencing, backoff,
dead letter, reclaim por heartbeat, `stale` por revisión) sin tocar la DB: SQL parseado + funciones puras
+ el flujo del tick con los accesos a DB simulados.
"""
import re
from pathlib import Path

import pytest

import plan_jobs as pj

_BACKEND = Path(__file__).resolve().parents[1]
_PLAN = "3957a669-c28a-40e2-9f4e-c1afffaf4e36"
_USER = "0257d89d-a294-4e00-94f3-f1b9f5599289"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("MEALFIT_PLAN_JOBS_ENABLED", "MEALFIT_PLAN_JOBS_DISPLAY_I18N", "MEALFIT_PLAN_JOBS_BATCH",
              "MEALFIT_PLAN_JOBS_MAX_ATTEMPTS", "MEALFIT_PLAN_JOBS_BACKOFF_BASE_S"):
        monkeypatch.delenv(k, raising=False)


# ----------------------------------------------------------------------------- knobs y puros
def test_a_knobs_seguros_por_defecto(monkeypatch):
    assert pj.plan_jobs_enabled() is False, "el worker nace apagado: flip explícito en el .env"
    assert pj.consumer_enabled("display_i18n") is True
    monkeypatch.setenv("MEALFIT_PLAN_JOBS_BATCH", "5000")
    assert pj.worker_batch() == 100, "clamp superior"
    monkeypatch.setenv("MEALFIT_PLAN_JOBS_MAX_ATTEMPTS", "0")
    assert pj.max_attempts() == 1, "clamp inferior"


def test_b_backoff_exponencial_con_tope(monkeypatch):
    monkeypatch.setenv("MEALFIT_PLAN_JOBS_BACKOFF_BASE_S", "60")
    assert pj.backoff_seconds(1) == 60
    assert pj.backoff_seconds(2) == 120
    assert pj.backoff_seconds(3) == 240
    assert pj.backoff_seconds(40) == 6 * 60 * 60, "tope de 6 h"


def test_c_veredicto_del_resultado_de_traduccion():
    assert pj.verdict_for_display_result({"enriched_meals": 12, "skipped": None}) == ("done", None)
    assert pj.verdict_for_display_result({"enriched_meals": 0, "skipped": "no_meals"}) == ("done", None)
    assert pj.verdict_for_display_result({"enriched_meals": 0, "skipped": "circuit_breaker_open"}) == ("failed", "circuit_breaker_open")
    assert pj.verdict_for_display_result({"enriched_meals": 6, "skipped": "partial_loss"})[0] == "failed", "el lote perdido se reintenta"
    assert pj.verdict_for_display_result("basura") == ("failed", "bad_result")


def test_d_dedup_key_por_plan_revision_locale_y_dias():
    assert pj.display_i18n_dedup_key(_PLAN, 7, "fr-FR") == f"display_i18n:{_PLAN}:7:fr-FR:all"
    assert pj.display_i18n_dedup_key(_PLAN, None, "en-US", [3, 1, 3]) == f"display_i18n:{_PLAN}:0:en-US:1,3"


# ----------------------------------------------------------------------------- SQL del protocolo
def test_e_claim_es_atomico_y_solo_de_tipos_con_consumidor():
    sql = pj.CLAIM_SQL
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert "attempts = p.attempts + 1" in sql, "attempts es el token de fencing"
    assert "j.job_type = ANY(%s)" in sql, "shopping_projection queda pending hasta tener consumidor"
    assert "j.status IN ('pending', 'failed')" in sql and "j.execute_after <= NOW()" in sql


def test_f_commit_fenced_y_reclaim_por_heartbeat():
    assert "WHERE id = %s AND claimed_by = %s AND attempts = %s AND status = 'processing'" in pj.FINISH_SQL
    assert "NOW() + (%s * INTERVAL '1 second')" in pj.FINISH_SQL, "backoff en failed"
    assert "heartbeat_at < NOW() - (%s * INTERVAL '1 second')" in pj.RECLAIM_SQL
    assert "CASE WHEN attempts >= %s THEN 'dead' ELSE 'failed' END" in pj.RECLAIM_SQL


def test_g_enqueue_idempotente_por_dedup_key_y_despierta_al_worker(monkeypatch):
    captured = {}

    def fake_write(sql, params=None, returning=False, **kw):
        captured["sql"] = sql
        captured["params"] = params
        return [{"id": "11111111-1111-1111-1111-111111111111"}]

    import db
    monkeypatch.setattr(db, "execute_sql_write", fake_write)
    woke = []
    monkeypatch.setattr(pj, "wake_plan_jobs_worker", lambda reason="": woke.append(reason) or True)
    job_id = pj.enqueue_plan_job("display_i18n", _PLAN, _USER, plan_revision=3, dedup_key="k", payload={"locale": "fr-FR"})
    assert job_id == "11111111-1111-1111-1111-111111111111"
    assert "ON CONFLICT (dedup_key) DO NOTHING RETURNING id" in captured["sql"]
    assert captured["params"][0] == "display_i18n" and captured["params"][3] == 3
    assert woke == ["enqueue:display_i18n"]
    # guests (sin UUID) jamás entran en la cola: la FK a user_profiles no los admite
    assert pj.enqueue_plan_job("display_i18n", _PLAN, "guest", plan_revision=1, dedup_key="k2") is None


def test_h_finish_convierte_failed_en_dead_al_agotar_intentos(monkeypatch):
    seen = {}

    def fake_write(sql, params=None, returning=False, **kw):
        seen["params"] = params
        return [{"id": "x"}]

    import db
    monkeypatch.setattr(db, "execute_sql_write", fake_write)
    monkeypatch.setenv("MEALFIT_PLAN_JOBS_MAX_ATTEMPTS", "3")
    job = {"id": "x", "claimed_by": "w1", "attempts": 3, "job_type": "display_i18n", "plan_id": _PLAN}
    assert pj.finish_plan_job(job, "failed", error_code="circuit_breaker_open") is True
    assert seen["params"][0] == "dead", "3 de 3 intentos → dead letter"
    job2 = dict(job, attempts=1)
    pj.finish_plan_job(job2, "failed", error_code="circuit_breaker_open")
    assert seen["params"][0] == "failed" and seen["params"][2] == pj.backoff_seconds(1)
    # fencing rechazado: 0 filas → False
    monkeypatch.setattr(db, "execute_sql_write", lambda *a, **k: [])
    assert pj.finish_plan_job(job2, "done") is False


# ----------------------------------------------------------------------------- consumidor y tick
def test_i_consumidor_stale_si_la_revision_cambio_y_reencola_para_la_vigente(monkeypatch):
    monkeypatch.setattr(pj, "current_plan_revision", lambda plan_id: 9)
    requeued = {}

    def fake_enqueue(job_type, plan_id, user_id, **kw):
        requeued.update(kw, job_type=job_type)
        return "new-id"

    monkeypatch.setattr(pj, "enqueue_plan_job", fake_enqueue)
    job = {"id": "old", "plan_id": _PLAN, "user_id": _USER, "plan_revision": 5, "payload": {"locale": "fr-FR", "day_indices": None}}
    status, code, result = pj._consume_display_i18n(job)
    assert (status, code) == ("stale", "revision_changed")
    assert requeued["plan_revision"] == 9 and requeued["dedup_key"].startswith(f"display_i18n:{_PLAN}:9:fr-FR")
    assert result["requeued_job_id"] == "new-id"


def test_j_consumidor_llama_a_enrich_con_la_revision_vigente(monkeypatch):
    monkeypatch.setattr(pj, "current_plan_revision", lambda plan_id: 5)
    calls = {}
    import plan_display_i18n as pdi

    def fake_enrich(plan_id, user_id, locale, day_indices=None):
        calls.update(plan_id=plan_id, user_id=user_id, locale=locale, day_indices=day_indices)
        return {"enriched_meals": 12, "skipped": None}

    monkeypatch.setattr(pdi, "enrich_plan_display", fake_enrich)
    job = {"id": "j", "plan_id": _PLAN, "user_id": _USER, "plan_revision": 5, "payload": {"locale": "fr-FR", "day_indices": [0, 1]}}
    status, code, result = pj._consume_display_i18n(job)
    assert (status, code) == ("done", None)
    assert calls == {"plan_id": _PLAN, "user_id": _USER, "locale": "fr-FR", "day_indices": [0, 1]}


def test_k_tick_apagado_no_toca_la_db(monkeypatch):
    import db
    monkeypatch.setattr(db, "execute_sql_write", lambda *a, **k: (_ for _ in ()).throw(AssertionError("DB tocada")))
    assert pj.process_plan_jobs()["skipped"] == "knob_off"


def test_l_tick_completo_reclaim_claim_consumo_commit(monkeypatch):
    monkeypatch.setenv("MEALFIT_PLAN_JOBS_ENABLED", "1")
    monkeypatch.setattr(pj, "reclaim_stale_processing", lambda: 1)
    job = {"id": "j1", "job_type": "display_i18n", "plan_id": _PLAN, "user_id": _USER, "plan_revision": 2,
           "payload": {"locale": "fr-FR"}, "attempts": 1, "claimed_by": "w"}
    monkeypatch.setattr(pj, "claim_plan_jobs", lambda limit, me, types: [job] if types == ["display_i18n"] else [])
    monkeypatch.setitem(pj.CONSUMERS, "display_i18n", lambda j: ("done", None, {"enriched_meals": 3}))
    finished = []
    monkeypatch.setattr(pj, "finish_plan_job", lambda j, status, **kw: finished.append((j["id"], status)) or True)
    monkeypatch.setattr(pj, "_emit_metric", lambda *a, **k: None)
    s = pj.process_plan_jobs()
    assert s["reclaimed"] == 1 and s["claimed"] == 1 and s["done"] == 1 and s["fencing_rejected"] == 0
    assert finished == [("j1", "done")]


def test_m_maybe_enqueue_respeta_knobs_guests_y_dedup(monkeypatch):
    assert pj.maybe_enqueue_display_i18n(_PLAN, _USER, "fr-FR") is False, "knob maestro apagado → hilo legacy"
    monkeypatch.setenv("MEALFIT_PLAN_JOBS_ENABLED", "1")
    assert pj.maybe_enqueue_display_i18n(_PLAN, "guest", "fr-FR") is False, "guest → hilo legacy"
    monkeypatch.setattr(pj, "current_plan_revision", lambda plan_id: 4)
    seen = {}
    monkeypatch.setattr(pj, "enqueue_plan_job", lambda *a, **kw: seen.update(kw) or "id-1")
    assert pj.maybe_enqueue_display_i18n(_PLAN, _USER, "fr-FR", [2, 0]) is True
    assert seen["dedup_key"] == f"display_i18n:{_PLAN}:4:fr-FR:0,2" and seen["payload"]["day_indices"] == [0, 2]


# ----------------------------------------------------------------------------- cableado (parser)
def test_n_el_disparador_de_traducciones_pasa_por_la_cola_antes_del_hilo():
    src = (_BACKEND / "plan_display_i18n.py").read_text(encoding="utf-8")
    i = src.index("def schedule_plan_display_enrichment(")
    body = src[i:src.index("threading.Thread(target=_run, daemon=True).start()", i)]
    assert "from plan_jobs import maybe_enqueue_display_i18n as _f5_enqueue" in body
    assert "if _f5_enqueue(plan_id, user_id, locale, day_indices):" in body
    assert body.index("_f5_enqueue(") < body.index("_prefilter_key = _inflight_key(")


def test_o_el_worker_esta_registrado_en_el_scheduler_ssot():
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    assert 'id="process_plan_jobs"' in src
    i = src.index('id="process_plan_jobs"')
    block = src[i - 600:i + 200]
    assert "_process_plan_jobs_job" in block and "max_instances=1" in block and "coalesce=True" in block
    assert re.search(r"def _process_plan_jobs_job\(\)", src)
    assert "from plan_jobs import process_plan_jobs" in src


def test_p_doc_y_claude_md():
    assert (_BACKEND / "docs" / "plan_jobs_f5.md").exists()
    claude = (_BACKEND / "CLAUDE.md").read_text(encoding="utf-8")
    assert "P1-ARQ25-F5-PLAN-JOBS" in claude and "MEALFIT_PLAN_JOBS_ENABLED" in claude
