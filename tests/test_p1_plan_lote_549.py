# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-549 · 2026-09-27] El fin de semana no pisa el tiempo de cocina declarado.

Auditoría del formulario: el bloque temporal (OBLIGATORIO) decía «Es FIN DE SEMANA. El usuario tiene más tiempo. Puedes
sugerir recetas un poco más elaboradas y meal prep dominical» sin mirar `cookingTime` ni `batchCooking`.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import prompts.plan_generator as pg  # noqa: E402


class _Reloj(datetime):
    fijo = datetime(2026, 9, 26, 15, 0, tzinfo=timezone.utc)      # sábado

    @classmethod
    def now(cls, tz=None):
        return cls.fijo if tz else cls.fijo.replace(tzinfo=None)


def _ctx(monkeypatch, cuando, **kw):
    _Reloj.fijo = cuando
    monkeypatch.setattr(pg, "datetime", _Reloj)
    return pg.build_time_context(country="DO", tz_offset_min=0, **kw)


def test_sabado_con_nada_de_tiempo(monkeypatch):
    txt = _ctx(monkeypatch, datetime(2026, 9, 26, 15, 0, tzinfo=timezone.utc), cooking_time="none")
    assert "FIN DE SEMANA" in txt and "más elaboradas" not in txt and "meal prep" not in txt, txt
    assert "NADA de tiempo" in txt


def test_sabado_con_tiempo_y_cocina_al_dia(monkeypatch):
    txt = _ctx(monkeypatch, datetime(2026, 9, 26, 15, 0, tzinfo=timezone.utc), cooking_time="plenty",
               batch_cooking="never")
    assert "más elaboradas" in txt and "meal prep" not in txt, txt


def test_laborable_sin_tandas(monkeypatch):
    txt = _ctx(monkeypatch, datetime(2026, 9, 28, 15, 0, tzinfo=timezone.utc), cooking_time="none")
    assert "DÍA LABORAL" in txt and "batch-cooking" not in txt and "<15 min" not in txt, txt


def test_sin_tiempo_declarado_el_texto_de_siempre(monkeypatch):
    txt = _ctx(monkeypatch, datetime(2026, 9, 26, 15, 0, tzinfo=timezone.utc))
    assert "Puedes sugerir recetas un poco más elaboradas y meal prep dominical." in txt


def test_ancla_en_el_contexto_compartido():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'cooking_time=form_data.get("cookingTime"), batch_cooking=form_data.get("batchCooking")),' in src
