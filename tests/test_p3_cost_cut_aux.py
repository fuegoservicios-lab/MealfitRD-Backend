"""[P3-COST-CUT-AUX · 2026-05-22] Tests del bundle Tier 1: migración de 3
auxiliary nodes a `gemini-3.1-flash-lite` como default sin perder calidad.

Nodes cubiertos:
  1. `tools_medical.consultar_base_datos_medica` — Q&A clínico determinístico.
     Knob existente `MEALFIT_MEDICAL_TOOL_MODEL`. Default cambió:
     `gemini-3.5-flash` → `gemini-3.1-flash-lite`.

  2. `context_compression_node` — síntesis textual sobre historial >2000 chars.
     Helper nuevo `_compressor_model_name()` + knob `MEALFIT_COMPRESSOR_MODEL`.
     Default `gemini-3.1-flash-lite`. Pre-fix usaba ruteo dinámico vía
     `_route_model(force_fast=True)` que resuelve a flash regular.

  3. `reflection_node` (meta_learning) — diagnóstico de UNA oración con
     structured output. Helper nuevo `_meta_learning_model_name()` + knob
     `MEALFIT_META_LEARNING_MODEL`. Default `gemini-3.1-flash-lite`.

Tareas todas son: structured/literal output corto, sin razonamiento creativo
ni multimodal. Safety nets existentes (CB per-modelo + fallback graceful)
cubren cualquier degradación.

Cross-link convention (P2-HIST-AUDIT-14): slug `p3_cost_cut_aux` matchea
este archivo.

Tooltip-anchor: P3-COST-CUT-AUX.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_TOOLS_MEDICAL_PY = _BACKEND_ROOT / "tools_medical.py"
_GRAPH_ORCHESTRATOR_PY = _BACKEND_ROOT / "graph_orchestrator.py"


@pytest.fixture(scope="module")
def medical_src() -> str:
    return _TOOLS_MEDICAL_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def orchestrator_src() -> str:
    return _GRAPH_ORCHESTRATOR_PY.read_text(encoding="utf-8")


# ===========================================================================
# Sección 1 — tools_medical default a lite
# ===========================================================================

def test_medical_tool_default_is_lite(medical_src: str):
    """`_medical_tool_model_name()` debe defaultear a `gemini-3.1-flash-lite`.
    Knob `MEALFIT_MEDICAL_TOOL_MODEL` sigue como rollback path."""
    fn_re = re.compile(
        r"def\s+_medical_tool_model_name\s*\(.*?(?=\ndef\s|\nclass\s|\n@)",
        re.DOTALL,
    )
    m = fn_re.search(medical_src)
    assert m is not None
    body = m.group(0)
    assert '"gemini-3.1-flash-lite"' in body or "'gemini-3.1-flash-lite'" in body, (
        "P3-COST-CUT-AUX regresión: default del tool médico no es "
        "`gemini-3.1-flash-lite`. Si volviste a flash full, documenta razón "
        "(el Q&A clínico determinístico es buen target para lite)."
    )


def test_medical_tool_knob_preserved(medical_src: str):
    """El knob `MEALFIT_MEDICAL_TOOL_MODEL` sigue siendo el path operacional
    de override sin redeploy."""
    assert "MEALFIT_MEDICAL_TOOL_MODEL" in medical_src, (
        "P3-COST-CUT-AUX regresión: knob `MEALFIT_MEDICAL_TOOL_MODEL` removido. "
        "Sin él no hay rollback operacional si lite degrada respuestas."
    )


# ===========================================================================
# Sección 2 — _compressor_model_name + callsite
# ===========================================================================

def test_compressor_helper_defined(orchestrator_src: str):
    """`_compressor_model_name()` debe existir como helper SSOT del modelo
    del compressor node."""
    assert re.search(
        r"^def\s+_compressor_model_name\s*\(\s*\)\s*->\s*str\s*:",
        orchestrator_src,
        re.MULTILINE,
    ), (
        "P3-COST-CUT-AUX regresión: helper `_compressor_model_name()` removido. "
        "Sin él, el callsite del compressor pierde el knob "
        "`MEALFIT_COMPRESSOR_MODEL` y vuelve al ruteo dinámico legacy."
    )


def test_compressor_helper_default_is_lite(orchestrator_src: str):
    """Default del helper compressor debe ser flash-lite."""
    fn_re = re.compile(
        r"def\s+_compressor_model_name\s*\(.*?(?=\ndef\s|\Z)",
        re.DOTALL,
    )
    m = fn_re.search(orchestrator_src)
    assert m is not None
    body = m.group(0)
    assert '"gemini-3.1-flash-lite"' in body or "'gemini-3.1-flash-lite'" in body, (
        "P3-COST-CUT-AUX regresión: default del helper compressor no es lite. "
        "El node solo hace síntesis textual con system prompt que fuerza "
        "preservación literal — no requiere flash full."
    )


def test_compressor_knob_referenced(orchestrator_src: str):
    """Knob `MEALFIT_COMPRESSOR_MODEL` debe leerse en el módulo."""
    assert "MEALFIT_COMPRESSOR_MODEL" in orchestrator_src, (
        "P3-COST-CUT-AUX regresión: knob `MEALFIT_COMPRESSOR_MODEL` no se "
        "lee en graph_orchestrator.py. Sin él no hay rollback sin redeploy."
    )


def test_compressor_node_uses_helper(orchestrator_src: str):
    """`context_compression_node` debe invocar `_compressor_model_name()` en
    lugar del `_route_model(force_fast=True)` directo."""
    fn_re = re.compile(
        r"async\s+def\s+context_compression_node\s*\(.*?(?=\nasync\s+def\s|\ndef\s|\n@)",
        re.DOTALL,
    )
    m = fn_re.search(orchestrator_src)
    assert m is not None
    body = m.group(0)
    assert "_compressor_model_name()" in body, (
        "P3-COST-CUT-AUX regresión: `context_compression_node` no invoca el "
        "helper. El knob `MEALFIT_COMPRESSOR_MODEL` no surte efecto en runtime."
    )


# ===========================================================================
# Sección 3 — _meta_learning_model_name + callsite
# ===========================================================================

def test_meta_learning_helper_defined(orchestrator_src: str):
    """`_meta_learning_model_name()` debe existir como helper SSOT."""
    assert re.search(
        r"^def\s+_meta_learning_model_name\s*\(\s*\)\s*->\s*str\s*:",
        orchestrator_src,
        re.MULTILINE,
    ), (
        "P3-COST-CUT-AUX regresión: helper `_meta_learning_model_name()` "
        "removido. Sin él, el reflector vuelve al ruteo dinámico legacy."
    )


def test_meta_learning_default_is_lite(orchestrator_src: str):
    """Default del helper meta_learning debe ser flash-lite."""
    fn_re = re.compile(
        r"def\s+_meta_learning_model_name\s*\(.*?(?=\ndef\s|\Z)",
        re.DOTALL,
    )
    m = fn_re.search(orchestrator_src)
    assert m is not None
    body = m.group(0)
    assert '"gemini-3.1-flash-lite"' in body or "'gemini-3.1-flash-lite'" in body, (
        "P3-COST-CUT-AUX regresión: default del helper meta_learning no es "
        "lite. La tarea es diagnóstico de UNA oración con structured output — "
        "lite cubre sin problema."
    )


def test_meta_learning_knob_referenced(orchestrator_src: str):
    """Knob `MEALFIT_META_LEARNING_MODEL` debe leerse en el módulo."""
    assert "MEALFIT_META_LEARNING_MODEL" in orchestrator_src, (
        "P3-COST-CUT-AUX regresión: knob `MEALFIT_META_LEARNING_MODEL` "
        "removido. Sin él no hay rollback sin redeploy."
    )


def test_reflection_node_uses_helper(orchestrator_src: str):
    """`reflection_node` debe invocar `_meta_learning_model_name()` en lugar
    de `_route_model(force_fast=True)` directo."""
    fn_re = re.compile(
        r"async\s+def\s+reflection_node\s*\(.*?(?=\nasync\s+def\s|\ndef\s|\n@)",
        re.DOTALL,
    )
    m = fn_re.search(orchestrator_src)
    assert m is not None
    body = m.group(0)
    assert "_meta_learning_model_name()" in body, (
        "P3-COST-CUT-AUX regresión: `reflection_node` no invoca el helper. "
        "El knob `MEALFIT_META_LEARNING_MODEL` no surte efecto en runtime."
    )
