"""[P1-GEN-EFFICIENCY · 2026-05-28] Contrato de los 3 recortes de latencia/costo
del pipeline de generación (análisis de costo prod 2026-05-28: day_generator +
self_critique = >50% del gasto; self_critique p50 ~192s = el cuello de botella).

Parser-based (robusto a venv sin langgraph). Ancla las 3 optimizaciones:

  #1 SELF-CRITIQUE-SKIP-CLEAN: cuando los DOS detectores determinísticos
     (`_count_staple_repetitions` + `_detect_slot_incoherence`) vienen vacíos,
     se salta el evaluador LLM (~30s) + sus correcciones (las llamadas más caras).
     Knob `MEALFIT_SELF_CRITIQUE_SKIP_WHEN_CLEAN` default True. El skip ocurre
     ANTES de construir el payload del evaluador.

  #3 DAYGEN-LITE-EASY: knob opt-in (default OFF) para bajar el day_generator a
     un modelo lite SOLO en perfiles fáciles (router resolvió a FLASH, no PRO) y
     SOLO en attempt 1. El override vive en `_route_model_for_day_generator`,
     NUNCA en `_route_model` (compartido por corrector/skeleton).

  #2 SKELETON-AB-TELEMETRY: la corrección de self_critique se emite con
     `node='self_critique_correction'` (distinto del evaluador) para poder
     correlacionar el modelo del skeleton (knob `MEALFIT_PLANNER_MODEL`, que YA
     existe) ↔ tasa de correcciones, sin un knob nuevo.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _REPO_ROOT / "backend"
_GRAPH = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def _func_body(src: str, signature_prefix: str) -> str:
    """Devuelve el cuerpo de la primera función cuyo `def` empieza con
    `signature_prefix`, hasta el siguiente `\\ndef ` a nivel módulo."""
    start = src.find(signature_prefix)
    assert start >= 0, f"No se encontró la función: {signature_prefix}"
    nxt = src.find("\ndef ", start + 1)
    return src[start: nxt if nxt > 0 else len(src)]


# ─────────────────────────── #1 self_critique skip-clean ───────────────────────────

def test_skip_when_clean_knob_default_true():
    assert (
        'SELF_CRITIQUE_SKIP_WHEN_CLEAN = _env_bool("MEALFIT_SELF_CRITIQUE_SKIP_WHEN_CLEAN", True)'
        in _GRAPH
    ), "Knob de skip debe existir con default True."


def test_skip_when_clean_guard_before_evaluator():
    guard = "SELF_CRITIQUE_SKIP_WHEN_CLEAN and not staple_repetitions and not slot_issues"
    assert guard in _GRAPH, "El early-exit debe condicionar por knob + ambos detectores limpios."
    body = _func_body(_GRAPH, "async def self_critique_node(")
    i_guard = body.find("SELF_CRITIQUE_SKIP_WHEN_CLEAN and not staple_repetitions")
    # La 1ª `_safe_ainvoke(` del cuerpo es la llamada del EVALUADOR (la cara, con
    # red+tokens). La instanciación de `evaluator_llm` es construcción de objeto
    # sin costo y puede estar arriba — lo que importa es saltar la LLAMADA.
    i_eval_call = body.find("_safe_ainvoke(")
    assert i_guard >= 0 and i_eval_call >= 0
    assert i_guard < i_eval_call, (
        "El skip debe ejecutarse ANTES de la llamada `_safe_ainvoke` del evaluador "
        "(si no, no ahorra la llamada cara)."
    )


# ─────────────────────────── #3 day-gen lite for easy ───────────────────────────

def test_daygen_lite_knobs_default_off():
    assert 'DAYGEN_LITE_FOR_EASY = _env_bool("MEALFIT_DAYGEN_LITE_FOR_EASY", False)' in _GRAPH, (
        "El day-gen lite debe ser opt-in (default False) por riesgo de calidad."
    )
    assert 'DAYGEN_EASY_MODEL = _env_str("MEALFIT_DAYGEN_EASY_MODEL"' in _GRAPH


def test_daygen_lite_override_scoped_to_day_generator_only():
    daygen = _func_body(_GRAPH, "def _route_model_for_day_generator(")
    # El override está en el router del day generator, gateado por attempt<=1 + FLASH.
    assert "DAYGEN_LITE_FOR_EASY" in daygen
    assert "_base == _FLASH_MODEL_NAME" in daygen
    assert "attempt <= 1 and DAYGEN_LITE_FOR_EASY" in daygen

    # Anti-regresión: el router GLOBAL `_route_model` (que comparten corrector y
    # otros) NO debe contener el override lite — su rama fácil sigue en FLASH full.
    base_router = _func_body(_GRAPH, "def _route_model(form_data")
    assert "DAYGEN_LITE_FOR_EASY" not in base_router, (
        "El override lite NO debe filtrarse a `_route_model` global (afectaría "
        "corrector/skeleton, no solo day_generator)."
    )
    assert "DAYGEN_EASY_MODEL" not in base_router
    assert "return _FLASH_MODEL_NAME" in base_router, (
        "La rama fácil de `_route_model` debe seguir devolviendo FLASH full."
    )


# ─────────────────────────── #2 skeleton A/B telemetry ───────────────────────────

def test_self_critique_correction_tagged_distinctly():
    assert '_current_node_var.set("self_critique_correction")' in _GRAPH, (
        "La corrección debe taggearse con node distinto para el A/B del skeleton."
    )
    body = _func_body(_GRAPH, "async def self_critique_node(")
    assert "_current_node_var.reset(_crit_node_token)" in body, (
        "El tag debe resetearse (try/finally) tras la llamada del corrector."
    )


def test_skeleton_model_knob_already_exists():
    # El A/B no requiere knob nuevo: `MEALFIT_PLANNER_MODEL` ya permite subir el
    # modelo del skeleton. Anclamos que sigue existiendo.
    assert 'MEALFIT_PLANNER_MODEL' in _GRAPH, (
        "El A/B del skeleton se hace con el knob existente MEALFIT_PLANNER_MODEL."
    )


# ─────────────────────────── markers ───────────────────────────

def test_markers_present():
    for marker in (
        "P1-SELF-CRITIQUE-SKIP-CLEAN",
        "P2-DAYGEN-LITE-EASY",
        "P2-SKELETON-AB-TELEMETRY",
    ):
        assert marker in _GRAPH, f"Falta el tooltip-anchor {marker} en el source."
