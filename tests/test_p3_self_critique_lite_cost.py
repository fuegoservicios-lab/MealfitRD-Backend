"""[P3-SELF-CRITIQUE-LITE-COST · 2026-05-22] Tests del knob
`MEALFIT_SELF_CRITIQUE_MODEL` que permite override del modelo del evaluator
de `self_critique_node` sin afectar day_generators ni corrector.

Contexto del cambio:
  - `self_critique_node` evalúa el plan generado con un LLM que recibe un
    summary comprimido + señales determinísticas pre-calculadas (slot_issues,
    staple_repetitions). El LLM solo emite 5 scores 1-10 + bool + suggestions.
  - Las señales críticas son determinísticas. Las safety nets ya cubren los
    casos donde el LLM falla en señalar (`deterministic_days` floor en línea
    ~6928).
  - Flash-lite es candidato natural: structured output simple sobre input
    comprimido. Costo: 6× más barato en input + output que flash regular.

Knob:
  - `MEALFIT_SELF_CRITIQUE_MODEL` — string. Default `_FLASH_MODEL_NAME`
    (preserva comportamiento pre-fix). Set a `gemini-3.1-flash-lite` para
    activar el ahorro.

Precedencia (de mayor a menor):
  1. `MEALFIT_EVALUATOR_USE_PRO=1` → Pro (escape hatch, ignora el knob nuevo).
  2. `MEALFIT_SELF_CRITIQUE_MODEL` (el knob nuevo).
  3. Default `_FLASH_MODEL_NAME`.

Cross-link convention (P2-HIST-AUDIT-14): slug `p3_self_critique_lite_cost`
matchea este archivo.

Tooltip-anchor: P3-SELF-CRITIQUE-LITE-COST.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GRAPH_ORCHESTRATOR_PY = _BACKEND_ROOT / "graph_orchestrator.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _GRAPH_ORCHESTRATOR_PY.read_text(encoding="utf-8")


# ===========================================================================
# Sección 1 — helper + knob declaration
# ===========================================================================

def test_helper_self_critique_model_defined(src: str):
    """`_self_critique_model_name()` debe estar definida."""
    assert re.search(
        r"^def\s+_self_critique_model_name\s*\(\s*\)\s*->\s*str\s*:",
        src,
        re.MULTILINE,
    ), (
        "P3-SELF-CRITIQUE-LITE-COST regresión: helper "
        "`_self_critique_model_name()` removido o renombrado. Sin esta "
        "función el callsite del self_critique_node pierde el knob."
    )


def test_knob_env_var_referenced(src: str):
    """`MEALFIT_SELF_CRITIQUE_MODEL` debe ser el nombre del env var leído
    en el helper. Sin esto el operador no puede flipar el modelo sin
    redeploy."""
    assert "MEALFIT_SELF_CRITIQUE_MODEL" in src, (
        "P3-SELF-CRITIQUE-LITE-COST regresión: env var "
        "`MEALFIT_SELF_CRITIQUE_MODEL` no se referencia en "
        "graph_orchestrator.py. El knob no se auto-registra en "
        "`_KNOBS_REGISTRY` ni puede flipar el modelo sin redeploy."
    )


def test_helper_default_is_flash_regular(src: str):
    """Default debe ser `_plan_flash_model_name()` o equivalente — preserva
    el comportamiento pre-fix. Si alguien cambia el default a flash-lite
    sin documentarlo, este test lo bloquea."""
    fn_re = re.compile(
        r"def\s+_self_critique_model_name\s*\(.*?(?=\ndef\s|\Z)",
        re.DOTALL,
    )
    m = fn_re.search(src)
    assert m is not None
    body = m.group(0)
    # El default puede expresarse como `_plan_flash_model_name()` o
    # como la string literal del modelo flash GA actual.
    assert (
        "_plan_flash_model_name()" in body
        or "_FLASH_MODEL_NAME" in body
        or '"gemini-3.5-flash"' in body
    ), (
        "P3-SELF-CRITIQUE-LITE-COST regresión: default del helper NO es "
        "el flash regular. Si cambias el default a `gemini-3.1-flash-lite`, "
        "documenta el riesgo de calidad en CLAUDE.md (perdió safety net del "
        "default-preservado-pre-fix) y actualiza este test."
    )


# ===========================================================================
# Sección 2 — callsite del self_critique_node usa el helper
# ===========================================================================

def test_self_critique_node_uses_helper(src: str):
    """`self_critique_node` debe invocar `_self_critique_model_name()` en
    lugar del `_route_model(force_fast=True)` directo. Sin esto, el knob
    no afecta runtime."""
    fn_re = re.compile(
        r"async\s+def\s+self_critique_node\s*\(.*?(?=\nasync\s+def\s|\ndef\s|\Z)",
        re.DOTALL,
    )
    m = fn_re.search(src)
    assert m is not None
    body = m.group(0)
    assert "_self_critique_model_name()" in body, (
        "P3-SELF-CRITIQUE-LITE-COST regresión: `self_critique_node` no "
        "invoca el helper `_self_critique_model_name()`. El knob "
        "`MEALFIT_SELF_CRITIQUE_MODEL` no se respeta en runtime."
    )


def test_evaluator_use_pro_still_dominates(src: str):
    """[Safety net] El knob `MEALFIT_EVALUATOR_USE_PRO` (P6-EVALUATOR-USE-PRO)
    sigue teniendo precedencia. Si `EVALUATOR_USE_PRO=1`, el modelo es Pro,
    NO el del nuevo knob. Sin este escape, no hay vía operacional para
    escalar a Pro cuando la calidad degrada."""
    fn_re = re.compile(
        r"async\s+def\s+self_critique_node\s*\(.*?(?=\nasync\s+def\s|\ndef\s|\Z)",
        re.DOTALL,
    )
    m = fn_re.search(src)
    assert m is not None
    body = m.group(0)
    # Debe haber un `if EVALUATOR_USE_PRO:` que precede al uso del helper nuevo.
    pro_idx = body.find("EVALUATOR_USE_PRO")
    helper_idx = body.find("_self_critique_model_name()")
    assert pro_idx >= 0 and helper_idx >= 0, (
        "P3-SELF-CRITIQUE-LITE-COST: tanto `EVALUATOR_USE_PRO` como "
        "`_self_critique_model_name()` deben existir en self_critique_node."
    )
    assert pro_idx < helper_idx, (
        "P3-SELF-CRITIQUE-LITE-COST regresión: `EVALUATOR_USE_PRO` ya no "
        "precede al helper nuevo. Sin esa precedencia el knob Pro pierde "
        "efecto y la safety net operacional se rompe."
    )


def test_corrector_unaffected_by_new_knob(src: str):
    """[Scope] El corrector dentro de `self_critique_node` (que regenera
    días individuales) NO debe usar el nuevo knob — solo el evaluator lo
    hace. Sin esta separación, swap a flash-lite también afectaría la
    regeneración de días, que tiene constraints más duros (macros)."""
    fn_re = re.compile(
        r"async\s+def\s+self_critique_node\s*\(.*?(?=\nasync\s+def\s|\ndef\s|\Z)",
        re.DOTALL,
    )
    m = fn_re.search(src)
    assert m is not None
    body = m.group(0)
    # Buscar `_corrector_model = ...` y verificar que NO usa el helper nuevo.
    corrector_match = re.search(r"_corrector_model\s*=\s*([^\n]+)", body)
    if corrector_match is None:
        pytest.skip(
            "_corrector_model assignment no encontrado — refactor puede haber "
            "movido el corrector a otra función. Actualizar regex si aplica."
        )
    rhs = corrector_match.group(1)
    assert "_self_critique_model_name" not in rhs, (
        "P3-SELF-CRITIQUE-LITE-COST regresión: el corrector está usando el "
        "helper del evaluator. Scope incorrecto: el knob debe SOLO afectar "
        "el evaluator (que emite scores), NO el corrector (que regenera "
        "días con constraints duros de macros)."
    )
