"""[P1-MACRO-POSTQUANT-RECONCILE · 2026-06-15] Re-reconcile multi-macro DESPUÉS de la cuantización (gap-audit G4).

Bug original (gap-audit 2026-06-15, G4):
    El reconcile multi-macro (`_macro_aware_day_reconcile`) clava C/F al target en `assemble_plan_node`,
    PERO la cuantización (FS2, Guard 4 de `_apply_deterministic_clinical_layer`) recomputa los macros desde
    las porciones redondeadas → re-introduce deriva BIDIRECCIONAL (carbos Y grasas, sobre Y bajo target). El
    único corrector post-cuantización (Guard 4b, carb-trim) solo BAJA carbos sobre-target y está OFF por
    default. Resultado medido: carbos ~18% MAPE, all-4-en-banda ~28%, incluso con el reconcile ON.

Cierre:
    Guard 4c re-ejecuta el MISMO reconcile validado (`_macro_aware_day_reconcile`) DESPUÉS de la cuantización
    y RE-CUANTIZA una vez (el re-snap protege los enteros: huevo/wrap, no "0.73 huevos"). Vive en el clinical
    layer (SSOT → el fallback también lo hereda). Gateado por `MACRO_AWARE_RECONCILE and MACRO_POSTQUANT_
    RECONCILE` (default True → efectivo donde el reconcile ya está ON; rollback: MEALFIT_MACRO_POSTQUANT_RECONCILE=False).

La corrección del building block (`_macro_aware_day_reconcile`: preserva proteína, clava C/F, no-op en banda,
fail-safe, clamp) ya está cubierta por test_p1_macro_aware_reconcile.py. ESTE test ancla el WIRING nuevo
(Guard 4c): que corra DESPUÉS de la cuantización, gateado, y RE-CUANTICE. Parser-based (sin DB/LLM).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_SRC = (Path(__file__).resolve().parent.parent / "graph_orchestrator.py").read_text(encoding="utf-8")


def _clinical_layer_body() -> str:
    start = _SRC.find("def _apply_deterministic_clinical_layer")
    assert start != -1, "No se encontró `_apply_deterministic_clinical_layer`."
    m = re.search(r"\n(async def |def )", _SRC[start + 10:])
    end = (start + 10 + m.start()) if m else len(_SRC)
    return _SRC[start:end]


def test_knob_defined_default_true():
    assert re.search(
        r'MACRO_POSTQUANT_RECONCILE\s*=\s*_env_bool\(\s*"MEALFIT_MACRO_POSTQUANT_RECONCILE"\s*,\s*True\s*\)',
        _SRC,
    ), "Falta el knob MACRO_POSTQUANT_RECONCILE (default True) — debe ser efectivo donde el reconcile está ON."


def test_guard4c_present_and_gated():
    body = _clinical_layer_body()
    assert "P1-MACRO-POSTQUANT-RECONCILE" in body, "Falta el tooltip-anchor de Guard 4c."
    # Gateado por AMBOS: el reconcile multi-macro activo + el sub-knob de re-reconcile.
    assert re.search(r"if\s+MACRO_AWARE_RECONCILE\s+and\s+MACRO_POSTQUANT_RECONCILE\s+and\s+_db\s+is\s+not\s+None\s*:", body), (
        "Guard 4c debe gatearse por `MACRO_AWARE_RECONCILE and MACRO_POSTQUANT_RECONCILE and _db is not None`."
    )


def test_guard4c_reconciles_then_requantizes():
    """El orden DEBE ser: re-reconcile → re-cuantizar (el re-snap protege los enteros)."""
    body = _clinical_layer_body()
    anchor = body.find("P1-MACRO-POSTQUANT-RECONCILE")
    # Desde el anchor hasta el fin del body: tras Guard 4c no hay más reconcile/cuantización en el
    # clinical layer, así que el primer `_macro_aware_day_reconcile` y el primer `_apply_portion_
    # quantization` que aparecen aquí son los de Guard 4c.
    block = body[anchor:]
    idx_reconcile = block.find("_macro_aware_day_reconcile")
    idx_requant = block.find("_apply_portion_quantization")
    assert idx_reconcile != -1, "Guard 4c debe llamar `_macro_aware_day_reconcile`."
    assert idx_requant != -1, "Guard 4c debe RE-CUANTIZAR vía `_apply_portion_quantization` tras re-nivelar."
    assert idx_reconcile < idx_requant, (
        "El orden debe ser reconcile→re-cuantizar: re-nivelar primero produce porciones fraccionarias, "
        "y la re-cuantización las devuelve a cocinables (protege enteros como el huevo)."
    )


def test_guard4c_runs_after_quantization_guard4():
    """Guard 4c debe correr DESPUÉS de la cuantización original (Guard 4) — su propósito es corregir
    la deriva que ESA cuantización introduce."""
    body = _clinical_layer_body()
    idx_guard4 = body.find("Guard 4 (FS2)")          # cuantización original
    idx_guard4c = body.find("Guard 4c")
    assert idx_guard4 != -1 and idx_guard4c != -1
    assert idx_guard4 < idx_guard4c, "Guard 4c (re-reconcile) debe venir después de Guard 4 (cuantización)."
