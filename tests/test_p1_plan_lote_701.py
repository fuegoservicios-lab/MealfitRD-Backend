# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-701 · 2026-09-28] El coste LLM de un chunk se atribuye a SU plan.

Medido en prod el 27-sep: 115 de 121 filas `day_generator` de `llm_usage_events` sin `plan_id`, pese a que el worker
de chunks fija `set_llm_attribution(user_id, meal_plan_id)` (lote 15). Nacían durante el procesado de chunks, con
`user_id` y sin `plan_id`. Causa: el worker lanza el pipeline con `ThreadPoolExecutor().submit(run_plan_pipeline, ...)`,
y un hilo del pool NO hereda los ContextVars del que lo lanza. `user_id` sobrevivía sólo porque `arun_plan_pipeline`
lo vuelve a fijar desde `form_data`; el plan se quedaba atrás.

Arreglo en el punto por el que pasan TODOS los caminos (worker de chunks, JIT del proactive_agent, lifecycle):
`arun_plan_pipeline` fija el plan desde `_caller_target_plan_id` (clave server-side: el strip la quita a los clientes)
junto al usuario, sin pisar uno ya fijado, y lo deshace en su finally.

tooltip-anchor: P1-PLAN-LOTE-701
"""
from __future__ import annotations

import concurrent.futures as cf
import re
from pathlib import Path

import llm_attribution as la

_BACKEND = Path(__file__).resolve().parents[1]


def test_un_hilo_del_pool_no_hereda_el_plan():
    """La causa, fijada: sin `copy_context`, el hilo del pool ve el plan vacío."""
    toks = la.set_llm_attribution(None, "plan-del-worker")
    try:
        with cf.ThreadPoolExecutor(max_workers=1) as ex:
            assert ex.submit(la.plan_id_var.get).result() is None
    finally:
        la.reset_llm_attribution(toks)


def test_el_pipeline_fija_el_plan_desde_el_form_en_el_hilo_nuevo():
    def pipeline_en_otro_hilo():
        toks = la.fijar_plan_del_pipeline({"_caller_target_plan_id": "plan-7"})
        try:
            return la.plan_id_var.get()
        finally:
            la.reset_llm_attribution(toks)

    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        assert ex.submit(pipeline_en_otro_hilo).result() == "plan-7"
    assert la.plan_id_var.get() is None


def test_no_pisa_un_plan_ya_fijado():
    toks = la.set_llm_attribution(None, "plan-del-swap")
    try:
        assert la.fijar_plan_del_pipeline({"_caller_target_plan_id": "otro"}) == []
        assert la.plan_id_var.get() == "plan-del-swap"
    finally:
        la.reset_llm_attribution(toks)


def test_sin_plan_en_el_form_no_hace_nada():
    assert la.fijar_plan_del_pipeline({}) == []
    assert la.fijar_plan_del_pipeline(None) == []
    assert la.fijar_plan_del_pipeline({"_caller_target_plan_id": ""}) == []
    assert la.plan_id_var.get() is None


def test_arun_plan_pipeline_lo_fija_y_lo_suelta():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    fija = re.search(r'_p701_pid\s*=\s*__import__\("llm_attribution"\)\.fijar_plan_del_pipeline\(actual_form_data\)', src)
    suelta = re.search(r'__import__\("llm_attribution"\)\.reset_llm_attribution\(_p701_pid\)', src)
    assert fija and suelta
    assert fija.start() < suelta.start()
    # junto al usuario (misma línea que su set) y el reset en el finally de las ContextVars del pipeline
    linea_set = src[src.rfind("\n", 0, fija.start()):fija.start()]
    assert "user_id_var.set(_rate_limit_uid)" in linea_set
    # primero en su línea: si otro reset de esa línea lanza, el plan ya se soltó (el thread del pool se reutiliza)
    linea_reset = src[suelta.start():src.find("\n", suelta.start())]
    assert "request_id_var.reset(_p134_req_token)" in linea_reset
