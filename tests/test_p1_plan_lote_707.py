# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-707 · 2026-09-28] El worker de chunks espera al pipeline al menos lo que el pipeline se da a sí mismo.

En prod `CHUNK_PIPELINE_TIMEOUT_SECONDS=600` y `MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S=900`. El pipeline planifica sus
reintentos contra 900 s (aprueba un retry si le quedan ≥ ~385 s), pero el worker deja de esperarlo a los 600: el
resultado se tira, el chunk se reintenta, y el hilo huérfano sigue gastando IA hasta sus 900 s
(`shutdown(wait=False)` no mata un hilo). Medido el 28-sep: `pipeline_holistic` p90 441 s, máx 587 s (a 13 s del corte);
0 timeouts desde el 08-sep. Esperar más no cuesta nada: el hilo corre igual. Ahora la espera es
`max(tope del chunk, GLOBAL_PIPELINE_TIMEOUT_S + margen)`. Knob de rollback `MEALFIT_CHUNK_WAIT_COVERS_PIPELINE`.

tooltip-anchor: P1-PLAN-LOTE-707
"""
import re

import espera_del_chunk as ec
import graph_orchestrator as go


def test_la_espera_cubre_el_presupuesto_del_pipeline(monkeypatch):
    monkeypatch.setattr(go, "GLOBAL_PIPELINE_TIMEOUT_S", 900)
    monkeypatch.delenv("MEALFIT_CHUNK_WAIT_COVERS_PIPELINE", raising=False)
    monkeypatch.delenv("MEALFIT_CHUNK_WAIT_MARGIN_S", raising=False)
    assert ec.espera_del_pipeline(600) == 960
    assert ec.espera_del_pipeline(900) == 960


def test_nunca_acorta_una_espera_mayor(monkeypatch):
    monkeypatch.setattr(go, "GLOBAL_PIPELINE_TIMEOUT_S", 900)
    assert ec.espera_del_pipeline(1200) == 1200


def test_knob_de_rollback(monkeypatch):
    monkeypatch.setattr(go, "GLOBAL_PIPELINE_TIMEOUT_S", 900)
    monkeypatch.setenv("MEALFIT_CHUNK_WAIT_COVERS_PIPELINE", "false")
    assert ec.espera_del_pipeline(600) == 600


def test_el_worker_la_aplica_antes_de_esperar():
    src = open(__import__("cron_tasks").__file__, encoding="utf-8").read()
    i = src.index('_current_timeout = int(_current_timeout * 1.5)')
    j = src.index('result = _fut.result(timeout=_current_timeout)', i)
    assert re.search(r'_current_timeout = __import__\("espera_del_chunk"\)\.espera_del_pipeline\(_current_timeout\)',
                     src[i:j])
