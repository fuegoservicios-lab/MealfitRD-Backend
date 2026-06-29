"""[P1-RECIPE-QUANTIZE-UPDATES · 2026-06-29] (re-audit objetivo · P1)

El quantize de porciones a unidades de cocina medibles (¼ taza, ½ unidad, 5 g) corría SOLO en
form-gen (`assemble_plan_node`). Las superficies de UPDATE (swap / chat-modify / regenerate-day)
corren los MISMOS closers de macros que RE-INTRODUCEN decimales no medibles ('0.66 huevos',
'4.73g de mozzarella') y los persistían crudos en receta + lista + PDF.

Fix: `_apply_portion_quantization` se añade como ÚLTIMA mutación de `finalize_single_meal_recipe_coherence`
(el hook compartido ya cableado en swap/chat-modify/recalculate; regen-day lo hereda) → cubre las 4
superficies de un golpe. Gated por ASSEMBLE_FINAL_QUANTIZE (mismo knob que form-gen).

Tests: (1) parser-based del wiring + orden (quantize tras los demás finalizadores); (2) funcional con
spy sobre `_apply_portion_quantization` (sin Neon).
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as g

_BACKEND = Path(__file__).resolve().parent.parent
_GRAPH = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def _finalizer_body() -> str:
    """Cuerpo de `finalize_single_meal_recipe_coherence` (hasta el siguiente `def `)."""
    start = _GRAPH.find("def finalize_single_meal_recipe_coherence(")
    assert start != -1, "no se encontró finalize_single_meal_recipe_coherence"
    nxt = _GRAPH.find("\ndef ", start + 1)
    return _GRAPH[start: nxt if nxt != -1 else len(_GRAPH)]


# ---------------------------------------------------------------------------
# 1. Parser/estructura
# ---------------------------------------------------------------------------
def test_quantize_wired_in_finalizer():
    body = _finalizer_body()
    assert "P1-RECIPE-QUANTIZE-UPDATES" in body
    assert '_apply_portion_quantization({"days": _wrap}' in body, \
        "el finalizer debe quantizar el meal envuelto como mini-plan"
    assert "ASSEMBLE_FINAL_QUANTIZE" in body, "el quantize debe estar gated por el knob de form-gen"


def test_quantize_is_last_mutation():
    """El quantize debe correr DESPUÉS de los finalizadores que añaden/escalan ingredientes
    (veg-fantasma, slice-grams, leaf-cap, night-rice, nonempty) — si no, redondearía y luego
    otra mutación re-introduciría decimales."""
    body = _finalizer_body()
    idx_nonempty = body.find("_ensure_nonempty_recipe")
    idx_quant = body.find("_apply_portion_quantization")
    idx_dishq = body.find("_meal_dish_quality_issue")
    assert idx_nonempty != -1 and idx_quant != -1 and idx_dishq != -1
    assert idx_nonempty < idx_quant, "quantize debe ir tras el nonempty-backstop"
    # el advisory de dish-quality es solo una LECTURA → puede ir después del quantize.
    assert idx_quant < idx_dishq, "quantize (mutación) debe preceder al advisory de dish-quality (lectura)"


# ---------------------------------------------------------------------------
# 2. Funcional: spy sobre _apply_portion_quantization
# ---------------------------------------------------------------------------
def _isolate_other_finalizers(monkeypatch):
    """Apaga los demás sub-finalizadores (self-guard por knob) para aislar el quantize."""
    for knob in (
        "RECIPE_STEP_VEG_GUARD_ENABLED", "RECIPE_SLICE_GRAMS_ENABLED",
        "LEAF_VOLUME_CAP_ENABLED", "NIGHT_RICE_AUTOFIX_ENABLED",
        "RECIPE_NONEMPTY_BACKSTOP_ENABLED",
    ):
        if hasattr(g, knob):
            monkeypatch.setattr(g, knob, False)
    monkeypatch.setattr(g, "UPDATE_RECIPE_FINALIZE_ENABLED", True)


def test_finalizer_invokes_quantize_when_enabled(monkeypatch):
    _isolate_other_finalizers(monkeypatch)
    monkeypatch.setattr(g, "ASSEMBLE_FINAL_QUANTIZE", True)

    seen = {}

    def _spy(plan, db):
        seen["plan"] = plan
        return 1  # 1 meal "quantizado"

    monkeypatch.setattr(g, "_apply_portion_quantization", _spy)

    meal = {"name": "Tortilla", "ingredients": ["0.66 huevos", "30 g de queso"], "recipe": ["paso 1", "paso 2"]}
    total = g.finalize_single_meal_recipe_coherence(meal, db=object())

    assert "plan" in seen, "el finalizer no invocó _apply_portion_quantization"
    # el meal debe envolverse como mini-plan {"days": [{"meals": [meal]}]} apuntando al MISMO objeto.
    wrapped = seen["plan"]["days"][0]["meals"][0]
    assert wrapped is meal, "el quantize debe recibir el meal real (mutación in-place)"
    assert total >= 1, "el conteo de fixes debe incluir el del quantize"


def test_finalizer_skips_quantize_when_knob_off(monkeypatch):
    _isolate_other_finalizers(monkeypatch)
    monkeypatch.setattr(g, "ASSEMBLE_FINAL_QUANTIZE", False)

    called = {"n": 0}
    monkeypatch.setattr(g, "_apply_portion_quantization", lambda plan, db: called.__setitem__("n", called["n"] + 1) or 1)

    meal = {"name": "Tortilla", "ingredients": ["0.66 huevos"], "recipe": ["paso 1", "paso 2"]}
    g.finalize_single_meal_recipe_coherence(meal, db=object())
    assert called["n"] == 0, "con ASSEMBLE_FINAL_QUANTIZE=False el quantize NO debe correr"
