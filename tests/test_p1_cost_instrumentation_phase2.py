"""[P1-COST-INSTRUMENTATION-PHASE2 · 2026-05-16] Populate columna
`llm_usage_events.node` via Python ContextVar + decorator en nodos.

Phase 1 (P1-COST-INSTRUMENTATION 2026-05-15) dejó la columna `node` 100%
NULL ("unknown") porque la signature de `_safe_ainvoke` no incluía el
caller — modificar 30+ callsites era invasivo. Phase 2 usa ContextVar
+ decorator `@_node_label("nombre")` aplicado a los 12 nodos LangGraph;
el contextvar propaga automáticamente a llamadas dentro del nodo + a
tasks lanzadas via `asyncio.gather`/`create_task` (Python 3.7+ copia
contexto a tasks hijas).

Tests anclan:
  1. Contextvar `_current_node_var` definida con default=None.
  2. Decorator `_node_label` aplicado a los 12 nodos pipeline.
  3. `_emit_llm_usage_event_best_effort` lee el contextvar y lo pasa
     a `log_llm_usage_event(node=...)`.
  4. Functional: el contextvar propaga a asyncio.gather children +
     se resetea tras excepción.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GO_PY = _BACKEND_ROOT / "graph_orchestrator.py"


# ---------------------------------------------------------------------------
# 1. Parser-based: artefactos en código
# ---------------------------------------------------------------------------
def test_contextvar_defined_module_level():
    """`_current_node_var` debe estar definida como ContextVar con default=None
    para que llamadas fuera del pipeline (e.g., agent tools, scripts) registren
    `node=NULL` sin levantar excepción."""
    src = _GO_PY.read_text(encoding="utf-8")
    assert "_current_node_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(" in src, (
        "ContextVar `_current_node_var` no encontrada o sin type hint correcto."
    )
    assert 'default=None' in src, (
        "ContextVar debe tener `default=None` — sin esto, llamadas fuera de "
        "scope levantan LookupError."
    )


def test_decorator_node_label_defined():
    """Decorator `_node_label(name)` debe existir, soportar funciones async,
    setear + resetear el contextvar con try/finally para defensa contra
    excepciones."""
    src = _GO_PY.read_text(encoding="utf-8")
    assert "def _node_label(name: str):" in src
    # Setea + resetea
    assert "token = _current_node_var.set(name)" in src
    assert "_current_node_var.reset(token)" in src
    # try/finally para limpieza segura
    deco_block = re.search(
        r"def _node_label\(name: str\):.*?(?=\n# |\nclass |\nasync def |\Z)",
        src,
        re.DOTALL,
    )
    assert deco_block
    body = deco_block.group(0)
    assert "try:" in body and "finally:" in body, (
        "Decorator sin try/finally — si el nodo levanta excepción, el "
        "contextvar quedaría con valor stale para la próxima invocación."
    )


def test_emit_helper_reads_contextvar():
    """`_emit_llm_usage_event_best_effort` debe leer el contextvar y pasarlo
    como `node=` a `log_llm_usage_event`."""
    src = _GO_PY.read_text(encoding="utf-8")
    fn = re.search(
        r"def _emit_llm_usage_event_best_effort\(.*?(?=\nclass |\n# |\Z)",
        src,
        re.DOTALL,
    )
    assert fn, "Función `_emit_llm_usage_event_best_effort` no encontrada."
    body = fn.group(0)
    assert "_current_node_var.get()" in body, (
        "El helper no lee el contextvar — sigue persistiendo node=NULL."
    )
    assert "node=current_node" in body, (
        "El helper no pasa `node=current_node` a `log_llm_usage_event` — "
        "el valor leído no se propaga al INSERT."
    )


def test_all_pipeline_nodes_annotated():
    """Los 12 nodos LangGraph deben tener el decorator `@_node_label(...)`
    inmediatamente arriba de su `async def`."""
    src = _GO_PY.read_text(encoding="utf-8")
    nodes_expected = {
        "context_compression_node": "compressor",
        "plan_skeleton_node": "planner",
        "generate_days_parallel_node": "day_generator",
        "adversarial_judge_node": "judge",
        "self_critique_node": "self_critique",
        "assemble_plan_node": "assembler",
        "surgical_marker_regen_node": "surgical_marker",
        "review_plan_node": "reviewer",
        "reflection_node": "meta_learning",
        "preflight_optimization_node": "preflight",
        "retry_reflection_node": "retry_reflection",
        "semantic_cache_check_node": "semantic_cache_check",
    }
    for fn_name, label in nodes_expected.items():
        pattern = rf'@_node_label\("{label}"\)\nasync def {fn_name}\b'
        assert re.search(pattern, src), (
            f"Nodo `{fn_name}` no anotado con `@_node_label({label!r})`. "
            f"Sin esto las llamadas LLM dentro persisten con node=NULL."
        )


# ---------------------------------------------------------------------------
# 2. Functional: probar el contextvar import-real
# ---------------------------------------------------------------------------
def _load_graph_orchestrator():
    """Helper para import perezoso (evita disparar inits costosos al cargar
    el módulo de tests)."""
    os.environ.setdefault("GEMINI_API_KEY", "dummy")
    os.environ.setdefault("SUPABASE_URL", "https://dummy.supabase.co")
    os.environ.setdefault("SUPABASE_KEY", "dummy")
    os.environ.setdefault("CRON_SECRET", "dummy")
    sys.path.insert(0, str(_BACKEND_ROOT))
    import graph_orchestrator
    return graph_orchestrator


def test_contextvar_propagates_to_asyncio_tasks():
    """El contextvar DEBE propagar a `asyncio.gather` children — sin esto, las
    3 day_gen paralelas verían `None` aunque el parent las haya etiquetado."""
    go = _load_graph_orchestrator()

    @go._node_label("parent_label")
    async def parent():
        captured = []

        async def child():
            captured.append(go._current_node_var.get())

        await asyncio.gather(child(), child(), child())
        return captured

    captured = asyncio.run(parent())
    assert captured == ["parent_label"] * 3, (
        f"ContextVar no propagó a children: {captured}. "
        "Esto indica que Python no está copiando el contexto a las tasks; "
        "verificar versión de Python (≥3.7 requerido)."
    )


def test_contextvar_resets_after_exception():
    """Si el nodo levanta excepción, el contextvar DEBE volver a None —
    sin esto, llamadas posteriores fuera de scope verían el valor stale."""
    go = _load_graph_orchestrator()

    @go._node_label("explosive")
    async def explode():
        raise RuntimeError("boom")

    async def main():
        try:
            await explode()
        except RuntimeError:
            pass
        return go._current_node_var.get()

    after = asyncio.run(main())
    assert after is None, (
        f"ContextVar quedó con valor stale `{after!r}` tras excepción. "
        "Esto haría que llamadas LLM POSTERIORES fuera de un nodo persistan "
        "con el label equivocado."
    )


def test_contextvar_isolated_between_concurrent_pipelines():
    """Dos pipelines corriendo concurrentemente NO deben ver el label del otro
    — cada `asyncio.run` o task tiene su propio context. Critical para multi-tenant."""
    go = _load_graph_orchestrator()

    @go._node_label("pipeline_a")
    async def pipeline_a():
        await asyncio.sleep(0.01)
        return go._current_node_var.get()

    @go._node_label("pipeline_b")
    async def pipeline_b():
        await asyncio.sleep(0.01)
        return go._current_node_var.get()

    async def main():
        # Corren en paralelo desde NIVEL TOP (sin parent decorator)
        a, b = await asyncio.gather(pipeline_a(), pipeline_b())
        return a, b, go._current_node_var.get()

    a, b, outer = asyncio.run(main())
    assert a == "pipeline_a", f"Pipeline A vio `{a}` (esperaba pipeline_a)"
    assert b == "pipeline_b", f"Pipeline B vio `{b}` (esperaba pipeline_b)"
    assert outer is None, f"Outer scope contaminado: vio `{outer}`"
