"""[P1-ASSEMBLER-NODE-LABEL · 2026-07-26] El nodo más caro del pipeline no etiquetaba su costo.

## El caso

`assemble_plan_node` —el ensamblador, ~222 s medidos, la partida más grande del pipeline— no
tenía `@_node_label("assembler")`. Sin él, `_current_node_var` sigue en `None` durante todo el
ensamblaje y **cada llamada LLM originada ahí se persiste en `llm_usage_events` con
`node = NULL`**: justo el agujero que P1-COST-INSTRUMENTATION-PHASE2 abrió para cerrar.

## Por qué costaba verlo

El archivo SÍ tenía un `@_node_label("assembler")` — pero huérfano, unas 8.000 líneas más
arriba, sobre `_swap_excess_carbs_to_protein_for_day`: un helper **síncrono** que hace **cero
llamadas LLM** (no hay nada que atribuir) y cuyo único llamador de producción ya corre dentro
del nodo. Un `grep '_node_label("assembler")'` devolvía resultado y el nodo seguía desnudo.

Es la enésima forma del mismo modo de fallo de esta sesión: **código presente, efecto ausente**.
Por eso este test no se conforma con buscar el texto en el fuente — comprueba que la función
exportada esté REALMENTE envuelta.

## Verificado like-for-like

Los 38 archivos de test que referencian `assemble_plan_node`: 9 fallos sin el fix, 8 con él,
**0 nuevos**.

tooltip-anchor: P1-ASSEMBLER-NODE-LABEL
"""
from __future__ import annotations

import asyncio
import inspect

import pytest

import graph_orchestrator as g


# ───────────── 1. el EFECTO: la función está realmente envuelta ─────────────

def test_el_nodo_esta_decorado_de_verdad():
    """`functools.wraps` deja `__wrapped__` apuntando al original. Si alguien borra el
    decorator, el atributo desaparece — y eso NO lo detecta un grep del fuente."""
    assert hasattr(g.assemble_plan_node, "__wrapped__"), (
        "assemble_plan_node no está envuelto por @_node_label → sus llamadas LLM se "
        "persistirán con node=NULL"
    )


def test_sigue_siendo_una_corrutina():
    """El wrapper async debe preservar la naturaleza del nodo: LangGraph lo await-ea."""
    assert asyncio.iscoroutinefunction(g.assemble_plan_node)


def test_conserva_su_nombre():
    """LangGraph registra por nombre; `functools.wraps` lo preserva."""
    assert g.assemble_plan_node.__name__ == "assemble_plan_node"


def test_el_contextvar_se_pone_y_se_repone():
    """Contrato del decorator, ejercitado de verdad (no leído): pone la etiqueta al entrar y la
    repone al salir, incluso si el cuerpo revienta."""
    visto = {}

    @g._node_label("assembler")
    async def _falso():
        visto["dentro"] = g._current_node_var.get()
        raise RuntimeError("boom")

    assert g._current_node_var.get() is None
    with pytest.raises(RuntimeError):
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(_falso())
    assert visto["dentro"] == "assembler"
    assert g._current_node_var.get() is None, "debe reponerse aunque el nodo levante excepción"


# ───────────── 2. el decorator huérfano no volvió ─────────────

def test_el_helper_sincrono_ya_no_lo_lleva():
    """`_swap_excess_carbs_to_protein_for_day` no hace llamadas LLM y su llamador ya corre bajo
    el scope del nodo: el decorator ahí no atribuía nada y hacía creer que el nodo sí lo tenía."""
    assert not hasattr(g._swap_excess_carbs_to_protein_for_day, "__wrapped__"), (
        "el decorator huérfano volvió al helper"
    )


def test_el_helper_no_hace_llamadas_llm():
    """La premisa de quitárselo. Si algún día el helper llama a un LLM, hay que reetiquetarlo."""
    src = inspect.getsource(g._swap_excess_carbs_to_protein_for_day)
    for token in ("_safe_ainvoke", "ainvoke(", ".invoke("):
        assert token not in src, f"el helper ahora sí llama LLM ({token}): revisa la atribución"


# ───────────── 3. ancla de la clase ─────────────

def test_todos_los_nodos_del_grafo_estan_envueltos():
    """Ancla al EFECTO para los 12 nodos, no al texto. Es el complemento de
    `test_p1_cost_instrumentation_phase2::test_all_pipeline_nodes_annotated`, que parsea el
    fuente — y que un decorator mal colocado podía satisfacer sin envolver nada."""
    nodos = [
        "context_compression_node", "plan_skeleton_node", "generate_days_parallel_node",
        "adversarial_judge_node", "self_critique_node", "assemble_plan_node",
        "surgical_marker_regen_node", "review_plan_node", "reflection_node",
        "preflight_optimization_node", "retry_reflection_node", "semantic_cache_check_node",
    ]
    sin_envolver = [n for n in nodos
                    if not hasattr(getattr(g, n, None), "__wrapped__")]
    assert not sin_envolver, f"nodos sin @_node_label aplicado: {sin_envolver}"
