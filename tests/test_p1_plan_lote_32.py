# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-32 · 2026-09-13] E7 / ARQ30-P2-01: aire en `graph_orchestrator.py` sin cambiar conducta.

El god-file estaba en 53.099 líneas con el tope congelado en 53.100: el siguiente arreglo no cabía. Se movieron TAL CUAL
(mismo texto; sólo cambian los imports) dos bloques de infraestructura cuyo acoplamiento se midió por AST antes de tocar
nada — ninguno lee del grafo más que sus propios knobs:

  · `llm_concurrency.py`: los semáforos distribuidos (global y per-user), el contador de presupuesto y los dos knobs que
    sólo ellos leen. Las instancias y `acquire_user_and_global` se quedan en el grafo.
  · `llm_circuit_breaker.py`: `LLMCircuitBreaker`, el breaker local de escrituras best-effort y `LLMCircuitOpenError`.
    Los knobs `MEALFIT_CB_*`, `_circuit_breaker`, `_get_circuit_breaker` y `_record_cb_failure_unless_transient` se quedan.

El grafo re-exporta todo lo movido. Los tests que leían el FUENTE del grafo o lo parcheaban para alcanzar este código se
re-apuntaron conservando su intención; dos de esos parches (`graph_orchestrator.redis_async_client`) ya no alcanzaban nada
ANTES de mover: el breaker lee el cliente per-loop `get_redis_async()` desde P1-REDIS-ASYNC-PERLOOP-CB.
"""
from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]

_MOVIDOS = {
    "llm_concurrency": ["LLM_PER_USER_LOCAL_CACHE_MAX", "LLM_LOCAL_MAX_WAIT_S", "DistributedLLMSemaphore",
                        "DistributedPerUserSemaphore", "_LLM_BUDGET_STATS", "_LLM_BUDGET_STATS_LOCK",
                        "_inc_budget_stat", "get_llm_budget_stats_snapshot"],
    "llm_circuit_breaker": ["_BE_DB_CB_FAILURE_THRESHOLD", "_BE_DB_CB_OPEN_DURATION_S", "_BestEffortDBCircuitBreaker",
                            "_BE_DB_CB_REGISTRY", "_BE_DB_CB_REGISTRY_LOCK", "_get_be_db_cb", "_is_pool_timeout_error",
                            "LLMCircuitOpenError", "LLMCircuitBreaker"],
}
#: La POLÍTICA (qué umbral, qué fallo cuenta, qué instancia) se queda en el grafo, junto a los knobs que la definen.
_QUEDAN = ["LLM_SEMAPHORE", "PER_USER_LLM_SEMAPHORE", "acquire_user_and_global", "aacquire_user_and_global",
           "_circuit_breaker", "_CIRCUIT_BREAKERS_BY_MODEL", "_get_circuit_breaker", "_record_cb_failure_unless_transient",
           "_is_reviewer_transient_error", "CB_FAILURE_THRESHOLD", "CB_RESET_TIMEOUT_S", "CB_LOCAL_HEALTH_TTL_S",
           "LLM_MAX_CONCURRENT", "LLM_PER_USER_ENABLED", "LLM_COMBINED_MAX_WAIT_S"]


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


# ─────────────── lo movido: mismo objeto, un solo sitio ───────────────

@pytest.mark.parametrize("modulo", sorted(_MOVIDOS))
def test_lo_movido_se_reexporta_y_es_el_mismo_objeto(go, modulo):
    m = importlib.import_module(modulo)
    for nombre in _MOVIDOS[modulo]:
        assert getattr(go, nombre) is getattr(m, nombre), f"{nombre}: el grafo debe re-exportar el objeto de {modulo}"


@pytest.mark.parametrize("modulo", sorted(_MOVIDOS))
def test_el_grafo_ya_no_lo_define_lo_importa(modulo):
    src = _src("graph_orchestrator.py")
    definidos, importados = _definidos(src), _importados_de(src, modulo)
    propios = _definidos(_src(f"{modulo}.py"))
    for nombre in _MOVIDOS[modulo]:
        assert nombre not in definidos, f"{nombre} volvió a definirse en el grafo: dos copias divergen"
        assert nombre in importados, f"{nombre} no se re-exporta desde {modulo}"
        assert nombre in propios, f"{nombre} no está definido en {modulo}.py"


def test_la_politica_se_queda_en_el_grafo():
    definidos = _definidos(_src("graph_orchestrator.py"))
    for nombre in _QUEDAN:
        assert nombre in definidos, f"{nombre} debía quedarse en el grafo (lee sus knobs y los tests lo parchean allí)"


def test_el_mecanismo_no_conoce_al_grafo():
    """Sin ciclo: los módulos nuevos no importan el grafo (ni perezosamente)."""
    for modulo in _MOVIDOS:
        for n in ast.walk(ast.parse(_src(f"{modulo}.py"))):
            if isinstance(n, ast.Import):
                assert all(a.name != "graph_orchestrator" for a in n.names), modulo
            elif isinstance(n, ast.ImportFrom):
                assert n.module != "graph_orchestrator", modulo


def test_los_knobs_siguen_registrados_y_cada_uno_se_lee_en_un_solo_sitio(go):
    import knobs
    registro = knobs.get_knobs_registry_snapshot()
    for knob, dueno in (("MEALFIT_LLM_LOCAL_MAX_WAIT_S", "llm_concurrency.py"),
                        ("MEALFIT_LLM_PER_USER_LOCAL_CACHE_MAX", "llm_concurrency.py"),
                        ("MEALFIT_CB_FAILURE_THRESHOLD", "graph_orchestrator.py"),
                        ("MEALFIT_CB_RESET_TIMEOUT_S", "graph_orchestrator.py"),
                        ("MEALFIT_CB_LOCAL_HEALTH_TTL_S", "graph_orchestrator.py")):
        assert knob in registro, f"{knob} dejó de registrarse al mudarse"
        lectores = [f for f in ("graph_orchestrator.py", "llm_concurrency.py", "llm_circuit_breaker.py")
                    if re.search(rf'_env_(int|float|bool|str)\s*\(\s*["\']{knob}["\']', _src(f))]
        assert lectores == [dueno], f"{knob}: se lee en {lectores}, debía leerse sólo en {dueno}"


# ─────────────── conducta: nada cambió para quien usa estos nombres ───────────────

def test_el_logger_conserva_su_nombre():
    """El formato de producción imprime `%(name)s` y los `caplog` filtran por él: mover el código no mueve sus logs."""
    import llm_circuit_breaker
    import llm_concurrency
    assert llm_concurrency.logger.name == "graph_orchestrator"
    assert llm_circuit_breaker.logger.name == "graph_orchestrator"


def test_el_cableado_del_grafo_sigue_vivo(go):
    import llm_circuit_breaker as cbm
    import llm_concurrency as lc
    assert isinstance(go.LLM_SEMAPHORE, lc.DistributedLLMSemaphore)
    assert isinstance(go.PER_USER_LLM_SEMAPHORE, lc.DistributedPerUserSemaphore)
    assert isinstance(go._circuit_breaker, cbm.LLMCircuitBreaker)
    try:
        b = go._get_circuit_breaker("lote32-probe")
        assert isinstance(b, cbm.LLMCircuitBreaker) and go._get_circuit_breaker("lote32-probe") is b
    finally:
        go._CIRCUIT_BREAKERS_BY_MODEL.pop("lote32-probe", None)
    antes = go.get_llm_budget_stats_snapshot().get("lote32_probe", 0)
    try:
        go._inc_budget_stat("lote32_probe")
        assert lc.get_llm_budget_stats_snapshot()["lote32_probe"] == antes + 1
    finally:
        with lc._LLM_BUDGET_STATS_LOCK:
            lc._LLM_BUDGET_STATS.pop("lote32_probe", None)
    assert issubclass(go.LLMCircuitOpenError, Exception)


def test_parchear_el_grafo_ya_no_alcanza_al_breaker(monkeypatch, go):
    """La regla para el próximo test: se parchea el módulo que LEE el nombre, no el que lo re-exporta."""
    import llm_circuit_breaker as cbm
    en_grafo, en_modulo = [], []
    monkeypatch.setattr(go, "execute_sql_write", lambda *a, **k: en_grafo.append(a) or True)
    monkeypatch.setattr(cbm, "execute_sql_write", lambda *a, **k: en_modulo.append(a) or True)
    cbm.LLMCircuitBreaker(failure_threshold=3, reset_timeout=30, model_name="lote32-probe")._atomic_reset_db()
    assert len(en_modulo) == 1 and en_grafo == []


# ─────────────── el tope baja con la extracción ───────────────

def test_el_grafo_tiene_aire_y_el_tope_baja_con_la_extraccion():
    n = _src("graph_orchestrator.py").count("\n")
    assert n <= 52_240, f"graph_orchestrator.py {n} líneas: extraer, no subir el tope"   # [P1-PLAN-LOTE-64] el tope bajó con la 2.ª extracción
    caps = _src("tests/test_p3_shopping_projection_pkg.py")
    assert re.search(r'"graph_orchestrator\.py":\s*52_240', caps), "el tope SSOT de los god-files baja con la extracción"


def test_docs_plan_marker_y_anclas():
    for f in ("llm_concurrency.py", "llm_circuit_breaker.py", "graph_orchestrator.py"):
        assert "P1-PLAN-LOTE-32" in _src(f), f
    doc = _src("docs/arq30_e5_e7_diseno_canario.md")
    assert "P1-PLAN-LOTE-32" in doc and "llm_circuit_breaker.py" in doc and "llm_concurrency.py" in doc
    assert "P1-PLAN-LOTE-32" in _src("docs/plan_pendientes_2026_09_11.md")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src("app.py"))
    assert m and int(m.group(1)) >= 32
