"""[P1-LEGUME-GUARANTEE-GOAL-AWARE · 2026-06-16]

La `🥗 GARANTÍA NUTRICIONAL` de `ai_helpers.get_deterministic_variety_prompt`
forzaba ≥1 leguminosa como proteína PRINCIPAL del esqueleto. Para `gain_muscle`
una base leguminosa+almidón no alcanza el piso de proteína (90% de un target
alto) con porciones cocinables → el revisor médico rechaza por DÉFICIT DE
PROTEÍNA → retry-storm + entrega degradada (`plan_quality_degraded`). Observado
en vivo (corr 13117aff, 2026-06-16): la garantía impuso 'Lentejas'/'Garbanzos'
→ días 84-107g vs piso 108g, incluso cuando la directiva de retry decía
explícitamente "NO dependas solo de leguminosas".

Fix: la garantía es goal-aware — se OMITE para los goals en
`_GOALS_SKIP_LEGUME_GUARANTEE` (gain_muscle). Las leguminosas siguen apareciendo
como acompañante en la generación del día; solo no se imponen como proteína
principal del esqueleto.

Dos capas de verificación:
  1. Parser-based (ancla la regla en el source con tooltip-anchor).
  2. Conductual (caplog): gain_muscle NUNCA fuerza leguminosa; balanced nunca
     emite el log de "Omitida".
"""
import logging
import random
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_AI_SRC = (_BACKEND / "ai_helpers.py").read_text(encoding="utf-8")

try:
    import ai_helpers as _AI  # noqa: E402
except Exception:  # pragma: no cover
    _AI = None

_needs_ai = pytest.mark.skipif(_AI is None, reason="ai_helpers no importable")

# Logs que SOLO emite el force-add de la garantía (reemplazo o append).
_FORCE_ADD_MARKERS = ("Leguminosa", "reemplaza", "añadida")
# Log que SOLO emite el skip goal-aware.
_SKIP_MARKER = "Omitida para goal"


# --------------------------------------------------------------------------
# 1. Parser-based: la regla goal-aware existe en el source (anclada).
# --------------------------------------------------------------------------
def test_source_has_goal_gate_anchor():
    assert "legume_guarantee_goal_gate" in _AI_SRC, (
        "Falta el tooltip-anchor `legume_guarantee_goal_gate` — un renombre del "
        "gate goal-aware debe fallar este test ANTES de tocar producción."
    )
    assert "_GOALS_SKIP_LEGUME_GUARANTEE" in _AI_SRC, "Falta el set de goals exentos."
    assert '"gain_muscle"' in _AI_SRC.split("_GOALS_SKIP_LEGUME_GUARANTEE", 1)[1][:120], (
        "gain_muscle debe estar en _GOALS_SKIP_LEGUME_GUARANTEE."
    )
    assert "_main_goal in _GOALS_SKIP_LEGUME_GUARANTEE" in _AI_SRC, (
        "La garantía de leguminosa debe estar gateada por el goal."
    )


# --------------------------------------------------------------------------
# 2. Conductual: gain_muscle nunca fuerza leguminosa.
# --------------------------------------------------------------------------
@_needs_ai
def test_gain_muscle_never_forces_legume(caplog):
    caplog.set_level(logging.INFO, logger="ai_helpers")
    random.seed(1337)
    skip_seen = 0
    for _ in range(80):
        caplog.clear()
        _AI.get_deterministic_variety_prompt(
            "", form_data={"mainGoal": "gain_muscle"}, user_id=None
        )
        for rec in caplog.records:
            msg = rec.getMessage()
            assert not (
                "GARANTÍA NUTRICIONAL" in msg
                and any(m in msg for m in ("reemplaza", "añadida"))
            ), f"gain_muscle forzó una leguminosa como proteína principal: {msg}"
            if _SKIP_MARKER in msg:
                skip_seen += 1
    # Sanity: la rama de skip debe ejercitarse (sin legume natural ≥1 vez en 80 runs).
    assert skip_seen > 0, (
        "El skip goal-aware nunca se ejecutó — el escenario no se está ejercitando "
        "(¿catálogo sin leguminosas o todas las picks naturales fueron leguminosa?)."
    )


# --------------------------------------------------------------------------
# 3. Conductual: balanced conserva la garantía (nunca la omite).
# --------------------------------------------------------------------------
@_needs_ai
def test_balanced_keeps_legume_guarantee(caplog):
    caplog.set_level(logging.INFO, logger="ai_helpers")
    random.seed(7)
    for _ in range(40):
        caplog.clear()
        _AI.get_deterministic_variety_prompt(
            "", form_data={"mainGoal": "balanced"}, user_id=None
        )
        for rec in caplog.records:
            assert _SKIP_MARKER not in rec.getMessage(), (
                "balanced NO debe omitir la garantía de leguminosa "
                "(solo los goals de _GOALS_SKIP_LEGUME_GUARANTEE)."
            )
