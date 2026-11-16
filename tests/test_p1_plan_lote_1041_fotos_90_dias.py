# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1041 · A-31 · 2026-11-16] Las fotos del chat se borran pasados 90 días (decisión del dueño)."""
from pathlib import Path

import pytest

import retencion_fotos_chat as r

_BACKEND = Path(__file__).resolve().parent.parent


def test_el_plazo_es_la_promesa_publicada():
    assert r.DIAS == 90
    assert "interval '90 days'" in r._SQL
    # no toca solo las no enviadas: borra cualquier foto vieja
    assert "message_id" not in r._SQL
    assert "ORDER BY created_at" in r._SQL and "LIMIT %s" in r._SQL


def test_borra_por_lotes_hasta_vaciar(monkeypatch):
    llamadas = []
    restos = [100, 100, 37]

    def falso(limite):
        llamadas.append(limite)
        return restos.pop(0)

    monkeypatch.setattr(r, "_borrar_lote", falso)
    assert r.purgar_fotos_viejas(lote=100, max_lotes=50) == 237
    assert llamadas == [100, 100, 100]


def test_respeta_el_tope_de_lotes(monkeypatch):
    monkeypatch.setattr(r, "_borrar_lote", lambda limite: limite)
    assert r.purgar_fotos_viejas(lote=10, max_lotes=3) == 30


def test_un_fallo_de_la_base_se_relanza(monkeypatch):
    def roto(limite):
        raise RuntimeError("db caída")

    monkeypatch.setattr(r, "_borrar_lote", roto)
    with pytest.raises(RuntimeError):
        r.purgar_fotos_viejas()


def test_se_registra_cada_24_h_una_sola_vez():
    class Sched:
        def __init__(self):
            self.jobs = {}

        def get_job(self, job_id):
            return self.jobs.get(job_id)

    vistos = []

    def add_job(sched, fn, trigger, **kw):
        vistos.append((fn, trigger, kw))
        sched.jobs[kw["id"]] = object()

    s = Sched()
    r.registrar(s, add_job, 300)
    r.registrar(s, add_job, 300)
    assert len(vistos) == 1
    fn, trigger, kw = vistos[0]
    assert fn is r.purgar_fotos_viejas and trigger == "interval" and kw["hours"] == 24
    assert kw["id"] == "purge_chat_attachments_90d" and kw["max_instances"] == 1


def test_cron_tasks_lo_registra():
    src = (_BACKEND / "cron_tasks.py").read_text(encoding="utf-8")
    assert "from retencion_fotos_chat import registrar as _reg_fotos_90d" in src
    assert "_reg_fotos_90d(scheduler, _add_job_jittered, _aggregator_misfire_grace_s())" in src


def test_la_politica_de_la_app_lo_dice():
    legal = _BACKEND.parent / "frontend" / "src" / "pages" / "legal" / "LegalPages.jsx"
    if not legal.exists():
        pytest.skip("sin frontend al lado")
    txt = legal.read_text(encoding="utf-8")
    assert "y, en todo caso, pasados 90 días desde que la envió" in txt
    assert "Las fotos que usted envía al chat del coach se borran pasados 90 días desde que las envió" in txt
