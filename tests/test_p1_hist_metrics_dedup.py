# -*- coding: utf-8 -*-
"""[P1-HIST-METRICS-DEDUP · 2026-08-13] El tab Métricas del Historial repetía
chunks: `plan_chunk_metrics` guarda UNA FILA POR INTENTO (el worker inserta al
completar cada ejecución, incluidas las degradadas que luego se reintentan) y
el endpoint `/chunk-metrics` hacía `LEFT JOIN plan_chunk_metrics m ON
m.chunk_id = q.id` a secas — el JOIN multiplicaba: el plan f380821a (w2 con 3
intentos + w3 con 4) devolvía 12 filas para 7 chunks, y el modal pintaba
«Semana 2 · Días 1-4» TRES veces seguidas con duraciones distintas,
contradiciendo el contador «Métricas (2)» de la cabecera (que suma counters
embebidos, no filas).

El contrato: UNA fila por chunk de la cola, y la métrica que lo representa es
la del ÚLTIMO intento (el que de verdad completó — retries más alto,
was_degraded del desenlace real). Los intentos históricos siguen en la tabla
para forense/admin; el modal del usuario no es el lugar para esa arqueología.
"""
from __future__ import annotations

import re
from pathlib import Path

import routers.plans as plans_mod

SRC = Path(plans_mod.__file__).read_text(encoding="utf-8")


def _endpoint_body() -> str:
    i = SRC.index("def api_plan_chunk_metrics")
    return SRC[i:SRC.index("\n@router", i)]


def test_no_queda_el_join_directo_que_multiplica():
    """El JOIN plano por chunk_id es EL bug: con N intentos persistidos produce
    N filas por chunk y el modal repite semanas."""
    body = _endpoint_body()
    assert not re.search(
        r"LEFT\s+JOIN\s+plan_chunk_metrics\s+m\s+ON\s+m\.chunk_id\s*=\s*q\.id",
        body,
    ), (
        "volvió el LEFT JOIN directo a plan_chunk_metrics: la tabla guarda una "
        "fila POR INTENTO y ese join multiplica los chunks del modal"
    )


def test_las_metricas_vienen_del_ultimo_intento_via_lateral():
    """LATERAL con ORDER BY created_at DESC LIMIT 1: exactamente una fila de
    métricas por chunk, la del intento que completó."""
    body = _endpoint_body()
    m = re.search(
        r"LEFT\s+JOIN\s+LATERAL\s*\(\s*SELECT[\s\S]*?FROM\s+plan_chunk_metrics[\s\S]*?"
        r"WHERE\s+\w+\.chunk_id\s*=\s*q\.id[\s\S]*?"
        r"ORDER\s+BY\s+\w+\.created_at\s+DESC[\s\S]*?LIMIT\s+1[\s\S]*?\)\s*m\s+ON\s+TRUE",
        body,
        re.IGNORECASE,
    )
    assert m, (
        "falta el LATERAL de último-intento sobre plan_chunk_metrics "
        "(ORDER BY created_at DESC LIMIT 1) — sin él, cada intento persistido "
        "duplica el chunk en el tab Métricas"
    )


def test_ancla_del_marker_en_el_sql():
    """Tooltip-anchor (convención del repo): el SQL lleva el marker para que un
    refactor del endpoint haga fallar este test antes de cambiar producción."""
    assert "P1-HIST-METRICS-DEDUP" in _endpoint_body()
