"""[P3-NEW-5 · 2026-05-10] Lock-the-doc: comentario en
`_classify_divergence_hypothesis` (shopping_calculator.py) anchora el
threshold 0.5 del bucket `pantry_overdeduct` + trigger para subirlo a 0.75
si la evidencia lo justifica.

Bug temido (audit 2026-05-10 — no accionable sin evidencia):
  El threshold actual `act_qty < exp_qty * 0.5` puede ser conservador:
  receta 3kg + nevera 2kg promete (ratio 0.67) cae al `unknown` final.
  Subir a 0.75 capturaría ese caso PERO también inflaría falsos positivos.

Fix:
  No code change inmediato. Comentario en el código que documenta:
    - Por qué 0.5 es conservador (trade-off bucket pantry_overdeduct vs
      unknown).
    - Trigger explícito para actuar (>25% de `unknown` correlacionan
      con sobrededucción real en pipeline_metrics).
    - Plan de acción si se observa: subir + añadir knob
      `MEALFIT_PANTRY_OVERDEDUCT_RATIO_THRESHOLD`.

Cobertura:
  1. El bloque `[P3-NEW-5` sigue presente en
     `_classify_divergence_hypothesis`.
  2. Menciona el threshold `0.5` y la alternativa `0.75`.
  3. Nombra el trigger explícito (pipeline_metrics + porcentaje).
  4. Nombra el knob propuesto si se actúa.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_SHOPPING_PY = _BACKEND_ROOT / "shopping_calculator.py"


def _read_classify_divergence_block() -> str:
    """Extrae el body de `_classify_divergence_hypothesis` desde `def` hasta
    el siguiente `def `."""
    src = _SHOPPING_PY.read_text(encoding="utf-8")
    start = src.find("def _classify_divergence_hypothesis")
    assert start > -1
    next_def = src.find("\ndef ", start + 1)
    assert next_def > -1
    return src[start:next_def]


def test_p3_new_5_block_present():
    body = _read_classify_divergence_block()
    assert "[P3-NEW-5" in body, (
        "Falta anchor `[P3-NEW-5` en `_classify_divergence_hypothesis`."
    )


def test_documents_alternative_threshold():
    body = _read_classify_divergence_block()
    block = body[body.find("[P3-NEW-5"):]
    assert "0.5" in block and "0.75" in block, (
        "El bloque P3-NEW-5 perdió la mención del threshold actual `0.5` "
        "y la alternativa `0.75`. Sin ambos, el reader no entiende el "
        "trade-off documentado."
    )


def test_names_explicit_trigger_to_act():
    body = _read_classify_divergence_block()
    block = body[body.find("[P3-NEW-5"):]
    assert "pipeline_metrics" in block, (
        "El bloque P3-NEW-5 no nombra `pipeline_metrics` como métrica a "
        "vigilar — sin esto, el dev no sabe DÓNDE buscar evidencia."
    )
    # Debe haber un porcentaje explícito (25% es el threshold sugerido).
    assert "25%" in block or "25 %" in block, (
        "El bloque P3-NEW-5 perdió el threshold sugerido (>25% correlación) "
        "para decidir si actuar."
    )


def test_names_proposed_knob_if_action_taken():
    body = _read_classify_divergence_block()
    block = body[body.find("[P3-NEW-5"):]
    assert "MEALFIT_PANTRY_OVERDEDUCT_RATIO_THRESHOLD" in block, (
        "El bloque P3-NEW-5 perdió el nombre del knob propuesto. Si un dev "
        "futuro decide subir el threshold, debe encontrar el patrón de knob "
        "ya documentado para no hardcodear el nuevo valor."
    )
