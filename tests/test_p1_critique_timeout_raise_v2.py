"""[P1-CRITIQUE-TIMEOUT-RAISE-V2 · 2026-05-16] Subir Flash timeout en self_critique
de 150s → 200s para reducir frecuencia de Pro fallback ($0.085/event).

Decisión informada por P1-COST-INSTRUMENTATION-PHASE2 (plan_id=0872063d):
  - self_critique = 68% del costo total ($0.257 de $0.376/plan)
  - Pro fallback solo = 45% del costo (1 evento × $0.170)
  - Día 1: 123s ✓, Día 2: >150s ✗ Pro fallback, Día 3: 148s ✓ (margen 2s)

A 200s, expectativa: 0-1 Pro fallback/plan (vs 1-2 antes). Ahorro
$0.10-0.15/plan.

Trade-off:
  - Latencia P50: igual (Flash entrega cuando lista, no espera al timeout).
  - Latencia P95: +30-50s en peor caso — pero ese caso antes era Pro
    fallback de ~60s, netos similares.
  - Riesgo calidad: cero (mismo modelo, mismo prompt, solo más tiempo).

Anchor para no revertir accidentalmente: el test enforza FLOOR de 200s
en .env. Si en futuro telemetría justifica bajar (e.g., self_critique
empieza a timeoutear a 200s con frecuencia, indicando que Flash empeoró),
ajustar el floor + memoria explicando POR QUÉ.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _BACKEND_ROOT / ".env"


def test_critique_timeout_at_or_above_floor():
    """`MEALFIT_CRITIQUE_FIX_TIMEOUT_S` debe estar ≥ 200s.

    Bajar de 200s revertiría P1-CRITIQUE-TIMEOUT-RAISE-V2: el plan E2E
    (plan_id=0872063d) mostró que Flash necesita 148-150s consistentes
    para entregar; a 150s, los días con margen <5s timeoutean a Pro.
    """
    env = _ENV_PATH.read_text(encoding="utf-8")
    m = re.search(
        r"^MEALFIT_CRITIQUE_FIX_TIMEOUT_S=(\d+)\s*$",
        env,
        re.MULTILINE,
    )
    assert m, "MEALFIT_CRITIQUE_FIX_TIMEOUT_S no seteado en .env"
    val = int(m.group(1))
    assert val >= 200, (
        f"MEALFIT_CRITIQUE_FIX_TIMEOUT_S={val} bajo el floor 200s. "
        f"Plan E2E mostró Flash tarda 148-150s en self_critique; bajar el "
        f"timeout vuelve a disparar Pro fallback ($0.085/evento). Si la "
        f"baja es intencional (e.g., métricas muestran que Flash mejoró), "
        f"actualizar este test + agregar memoria explicando el ajuste."
    )


def test_critique_timeout_comment_references_v2_decision():
    """El comentario inline en .env debe mencionar V2 + el dato de
    P1-COST-INSTRUMENTATION-PHASE2 que motivó la decisión — sin esto un
    refactor cosmético podría borrar la razón."""
    env = _ENV_PATH.read_text(encoding="utf-8")
    # Anchor textual mínimo:
    assert "P1-CRITIQUE-TIMEOUT-RAISE-V2" in env, (
        "Comentario sobre V2 ausente en .env — futuro mantenedor no sabrá "
        "por qué el knob está en 200s (cree que el default 150s estaba bien)."
    )
    # Referencia al plan_id del E2E que validó la decisión.
    assert "0872063d" in env, (
        "Falta plan_id de referencia (0872063d). Sin él, no hay forma de "
        "auditar el dato que motivó el ajuste."
    )


def test_critique_timeout_clamp_safe():
    """Sanity: el valor no debe ser excesivamente alto (>600s = 10 min) — eso
    bloquearía el pipeline global timeout (900s) sin presupuesto para retry."""
    env = _ENV_PATH.read_text(encoding="utf-8")
    m = re.search(r"^MEALFIT_CRITIQUE_FIX_TIMEOUT_S=(\d+)\s*$", env, re.MULTILINE)
    assert m
    val = int(m.group(1))
    assert val <= 600, (
        f"MEALFIT_CRITIQUE_FIX_TIMEOUT_S={val} demasiado alto (>600s). "
        f"Pipeline timeout global es 900s; dar >600s al critique no deja "
        f"presupuesto para day_gen + reviewer + retry. Si necesitas más, "
        f"subir GLOBAL_PIPELINE_TIMEOUT_S también."
    )
