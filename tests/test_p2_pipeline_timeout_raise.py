"""[P2-PIPELINE-TIMEOUT-RAISE · 2026-05-16] Regression guard: el knob
`MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S` debe ser >=900s para permitir 1 retry
del review_failed dentro del presupuesto.

Bug observado: 5 de 5 últimos planes generados (2026-05-15 / 2026-05-16)
terminaron en ~640s con review_passed=False. El log decía:

    ⏰ [ORQUESTADOR] Sin presupuesto para retry (66s restantes < 370s mínimo:
    180s generación + 80s margen post-retry + 110s cobertura hedging).
    Preservando mejor versión disponible.

Con timeout=720s y pipeline gastando ~640s en intento #1, quedan 80s — muy
por debajo del mínimo 370s requerido para retry. El plan se entregaba con
severity=minor y el reviewer médico marcaba `_review_failed_but_delivered=True`.

Fix: subir el global timeout a 900s (~+180s headroom). Con eso el pipeline
puede gastar ~640s en intento #1 + 260s para retry parcial — suficiente
para 1 día_gen + critique-fix.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _BACKEND_ROOT / ".env"


def _read_env() -> str:
    return _ENV_PATH.read_text(encoding="utf-8")


def test_global_pipeline_timeout_raised():
    """`MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S` debe ser >=900s para habilitar retry."""
    text = _read_env()
    m = re.search(
        r"^MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S\s*=\s*(\d+)",
        text,
        re.MULTILINE,
    )
    assert m, (
        "Falta knob `MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S` en .env. "
        "P2-PIPELINE-TIMEOUT-RAISE requiere override explícito (default código 720s)."
    )
    val = int(m.group(1))
    assert val >= 900, (
        f"`MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S={val}` < 900s. Sin esto, el "
        f"pipeline NO tiene presupuesto para retry post-review-fail "
        f"(observado: pipeline gasta ~640s, faltarían 370s para retry mínimo)."
    )


def test_timeout_not_too_aggressive():
    """`MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S` <=1200s. Por encima de 20min, el
    user perceptual frustration domina y el feature deep-search recovery
    no compensa."""
    text = _read_env()
    m = re.search(
        r"^MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S\s*=\s*(\d+)",
        text,
        re.MULTILINE,
    )
    assert m, "Knob no encontrado."
    val = int(m.group(1))
    assert val <= 1200, (
        f"`MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S={val}` > 1200s (20min). "
        f"Latencia perceptual demasiado alta. Si necesitas más, considerar "
        f"reducir el costo del pipeline (menos hedges, simpler prompts)."
    )
