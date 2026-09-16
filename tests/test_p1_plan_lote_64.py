# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-64 · 2026-09-15] E7 / ARQ30-P2-01 (2.ª extracción): más aire en `graph_orchestrator.py` sin cambiar conducta.

Medido primero con el método del lote 32 (AST): las funciones grandes leen 37-151 globales del módulo —moverlas sería
reescribirlas— y el bloque de telemetría de la llamada LLM con la caché de prompts leía 4 (`logger` y los dos contextvars,
que se mudan con él). Movido TAL CUAL a `llm_telemetry.py`; el grafo re-exporta. Se queda el despachador
`_submit_best_effort_metric` con su `_METRICS_EXECUTOR`: es política compartida por otros ocho sitios.

(a) lo movido: el mismo objeto, definido en un solo sitio, sin ciclo de imports y con el knob registrado donde vive;
(b) conducta: el logger conserva su nombre, el cableado del grafo sigue vivo y parchear el grafo YA NO alcanza a la caché
    (la lección del lote 32: se parchea el módulo que LEE el nombre, no el que lo re-exporta);
(c) el tope de los god-files baja con la extracción; docs y marker ≥ 64.
"""
from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_TOPE = 52_240

_MOVIDOS = {
    "llm_telemetry": [
        "_current_node_var", "user_id_var", "LLM_CACHE_TTL_S", "_emit_llm_timeout_metric", "_USAGE_EMIT_SEEN",
        "_usage_was_emitted", "_mark_usage_emitted", "_emit_llm_usage_event_best_effort", "PersistentLLMCache",
        "_LLM_CACHE", "CACHE_TTL_SECONDS",
    ],
}
#: Se quedan en el grafo: el despachador del emit y su executor (política de DÓNDE corre, compartida), y quien los usa.
_QUEDAN = ["_submit_best_effort_metric", "_METRICS_EXECUTOR", "_safe_ainvoke", "_node_label"]


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def _definidos(src: str) -> set:
    out = set()
    for n in ast.parse(src).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Assign):
            out |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            out.add(n.target.id)
    return out


def _importados_de(src: str, modulo: str) -> set:
    return {a.asname or a.name for n in ast.parse(src).body
            if isinstance(n, ast.ImportFrom) and n.module == modulo for a in n.names}


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator
    return graph_orchestrator


# ─────────────── (a) lo movido: mismo objeto, un solo sitio ───────────────

@pytest.mark.parametrize("modulo", sorted(_MOVIDOS))
def test_lo_movido_se_reexporta_y_es_el_mismo_objeto(go, modulo):
    m = importlib.import_module(modulo)
    for nombre in _MOVIDOS[modulo]:
        assert getattr(go, nombre) is getattr(m, nombre), f"{nombre}: el grafo debe re-exportar el objeto de {modulo}"


@pytest.mark.parametrize("modulo", sorted(_MOVIDOS))
def test_el_grafo_ya_no_lo_define_lo_importa(modulo):
    definidos, importados = _definidos(_src("graph_orchestrator.py")), _importados_de(_src("graph_orchestrator.py"), modulo)
    propios = _definidos(_src(f"{modulo}.py"))
    for nombre in _MOVIDOS[modulo]:
        assert nombre not in definidos, f"{nombre} volvió a definirse en el grafo: dos copias divergen"
        assert nombre in importados, f"{nombre} no se re-exporta desde {modulo}"
        assert nombre in propios, f"{nombre} no está definido en {modulo}.py"


def test_el_despachador_y_su_executor_se_quedan_en_el_grafo():
    definidos = _definidos(_src("graph_orchestrator.py"))
    for nombre in _QUEDAN:
        assert nombre in definidos, f"{nombre} debía quedarse en el grafo (política compartida por el pipeline)"


def test_la_telemetria_no_conoce_al_grafo():
    """Sin ciclo: el módulo nuevo no importa el grafo (ni perezosamente)."""
    for n in ast.walk(ast.parse(_src("llm_telemetry.py"))):
        if isinstance(n, ast.Import):
            assert all(a.name != "graph_orchestrator" for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            assert n.module != "graph_orchestrator"


def test_el_knob_sigue_registrado_y_se_lee_en_un_solo_sitio(go):
    import knobs
    assert "MEALFIT_LLM_CACHE_TTL_S" in knobs.get_knobs_registry_snapshot(), "el knob dejó de registrarse al mudarse"
    lectores = [f for f in ("graph_orchestrator.py", "llm_telemetry.py")
                if re.search(r'_env_(int|float|bool|str)\s*\(\s*["\']MEALFIT_LLM_CACHE_TTL_S["\']', _src(f))]
    assert lectores == ["llm_telemetry.py"], f"el knob se lee en {lectores}"


# ─────────────── (b) conducta: nada cambió para quien usa estos nombres ───────────────

def test_el_logger_conserva_su_nombre():
    """El formato de producción imprime `%(name)s` y los `caplog` filtran por él: mover el código no mueve sus logs."""
    import llm_telemetry
    assert llm_telemetry.logger.name == "graph_orchestrator"


def test_el_cableado_del_grafo_sigue_vivo(go):
    import llm_telemetry as t
    assert isinstance(go._LLM_CACHE, t.PersistentLLMCache) and go._LLM_CACHE.ttl == go.LLM_CACHE_TTL_S
    assert go.CACHE_TTL_SECONDS == go.LLM_CACHE_TTL_S
    token = go._current_node_var.set("lote64-probe")
    try:
        assert t._current_node_var.get() == "lote64-probe", "el contextvar es el mismo objeto en los dos módulos"
    finally:
        go._current_node_var.reset(token)
    assert "_emit_llm_usage_event_best_effort" in _src("graph_orchestrator.py"), "el grafo sigue llamándolo"


def test_parchear_el_grafo_ya_no_alcanza_a_la_cache(monkeypatch, go):
    """La regla del lote 32, ahora para la caché: se parchea el módulo que LEE el nombre, no el que lo re-exporta."""
    import llm_telemetry as t
    en_grafo, en_modulo = [], []
    monkeypatch.setattr(go, "execute_sql_query", lambda *a, **k: en_grafo.append(a) or [])
    monkeypatch.setattr(t, "execute_sql_query", lambda *a, **k: en_modulo.append(a) or [])
    monkeypatch.setattr(t, "redis_client", None)
    t.PersistentLLMCache(ttl_seconds=300).get("lote64-probe")
    assert len(en_modulo) == 1 and en_grafo == []


# ─────────────── (c) el tope baja con la extracción ───────────────

def test_el_grafo_tiene_aire_y_el_tope_baja_con_la_extraccion():
    n = _src("graph_orchestrator.py").count("\n")
    assert n <= _TOPE, f"graph_orchestrator.py {n} líneas: extraer, no subir el tope"
    caps = _src("tests/test_p3_shopping_projection_pkg.py")
    assert re.search(rf'"graph_orchestrator\.py":\s*{_TOPE // 1000}_{_TOPE % 1000:03d}', caps), \
        "el tope SSOT de los god-files baja con la extracción"


def test_docs_plan_marker_y_anclas():
    for f in ("llm_telemetry.py", "graph_orchestrator.py"):
        assert "P1-PLAN-LOTE-64" in _src(f), f
    doc = _src("docs/arq30_e5_e7_diseno_canario.md")
    assert "P1-PLAN-LOTE-64" in doc and "llm_telemetry.py" in doc
    assert "P1-PLAN-LOTE-64" in _src("docs/plan_pendientes_2026_09_11.md")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src("app.py"))
    assert m and int(m.group(1)) >= 64 and m.group(2) >= "2026-09-15"
