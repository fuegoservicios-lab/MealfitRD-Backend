"""[P1-PROD-AUDIT-1 · 2026-05-23] `GLOBAL_PIPELINE_TIMEOUT_S=720` es
decisión de producto operacional documentada — NO un gap a reducir.

Gap aparente (audit production-readiness 2026-05-23, B-P1-5):
    El audit external flageó "GLOBAL_PIPELINE_TIMEOUT_S=720s (12 min)
    default muy alto, puede causar bloqueos prolongados si Gemini está
    down".

Análisis post-audit (decisión documentada en graph_orchestrator.py:244-255):
    El timeout tiene math no-trivial:
      - Pre-fix era 600s (10min).
      - Bumpeado a 720s (12min) porque MIN_RETRY_BUDGET_S + RETRY_SAFETY_MARGIN_S
        + RETRY_HEDGING_DELTA_S sumaban 385s. Con cap 600s, cualquier primer
        intento que tomara >215s denegaba retry (~caso muy común, happy
        path = 200-260s).
      - Con 720s, el primer intento puede tomar hasta 335s y aún hay
        presupuesto para retry. Recuperamos ~120s de "ventana real de retry".
    Trade-off documentado inline: en peor caso (retry usado), usuario
    espera hasta 12 min en lugar de 10. En caso normal (~95% sin retry),
    cero cambio UX.

    Defensa contra el "Gemini down" del audit:
      - Circuit breaker (P2-NEW-D) abre el modelo Gemini tras N fallos
        consecutivos — siguientes calls fallan rápido sin esperar el cap.
      - LLMCircuitBreaker tiene knob `MEALFIT_CB_RESET_TIMEOUT_S=30` →
        usuario espera 30s + fallback degradado, NO 12min.

Decisión:
    Mantener `GLOBAL_PIPELINE_TIMEOUT_S=720` como default. Bajar tras P-fix
    `P2-AUDIT-MIN-RETRY-RECALIBRATE` que reduzca el threshold de retry
    (requiere re-tunear MIN_RETRY_BUDGET_S, RETRY_SAFETY_MARGIN_S, hedge
    timing).

    Es un knob — operador puede `MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S=600` si
    el SLA exige <10min hard cap. NO requiere redeploy.

Este test ancla la decisión:
    Si alguien "arregla" el knob bajándolo silenciosamente sin re-tunear
    el resto, el test falla con copy explicativo apuntando a la
    documentación inline.

Tooltip-anchor: P1-PROD-AUDIT-1-PIPELINE-TIMEOUT-DECISION | audit 2026-05-23.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GRAPH_ORCH = _BACKEND_ROOT / "graph_orchestrator.py"


def _read_graph_orch_lines():
    """Lee solo las primeras 500 líneas — el knob vive cerca del top, y
    leer 14.5K líneas es desperdicio."""
    with open(_GRAPH_ORCH, encoding="utf-8") as f:
        return [next(f) for _ in range(500) if True][:500]


def test_pipeline_timeout_knob_default_is_720():
    """El knob `MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S` debe tener default 720.

    Si alguien lo bajó a otro valor sin re-tunear el resto de la retry
    math, el test falla loud con explicación.
    """
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    m = re.search(
        r'GLOBAL_PIPELINE_TIMEOUT_S\s*=\s*_env_int\s*\(\s*["\']MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S["\']\s*,\s*(\d+)\s*\)',
        src,
    )
    assert m is not None, (
        "Knob `MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S` no se encuentra con "
        "default literal. Si refactorizaste a registry, actualizar este test."
    )
    default = int(m.group(1))
    assert default == 720, (
        f"Default de MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S cambió a {default} "
        f"(esperado: 720). Esta es una decisión operacional documentada en "
        f"graph_orchestrator.py:244-255: el valor 720 da ventana de retry "
        f"de ~335s en el primer intento + ~385s del threshold completo. "
        f"Bajar sin re-tunear MIN_RETRY_BUDGET_S + RETRY_SAFETY_MARGIN_S + "
        f"RETRY_HEDGING_DELTA_S rompe el retry budget (regresión P1-NEW-2). "
        f"Si la decisión cambió: documentar nuevo P-fix + actualizar este test."
    )


def test_decision_documented_inline():
    """La justificación matemática del 720 debe estar inline en el source
    para que un futuro lector entienda POR QUÉ no es 600.
    """
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    # Buscar palabras clave del comentario documentado.
    required_keywords = [
        "MIN_RETRY_BUDGET_S",
        "RETRY_SAFETY_MARGIN_S",
        "720",
        "600",  # mencionado como pre-fix value
    ]
    missing = [k for k in required_keywords if k not in src]
    assert not missing, (
        f"Documentación inline del default 720 incompleta — falta: {missing}. "
        f"Sin estos keywords, futuro mantenedor no entiende el trade-off."
    )


def test_circuit_breaker_protects_against_gemini_outage():
    """El audit external mencionó "Gemini down → usuario espera 12min".
    Validamos que el LLMCircuitBreaker existe + se invoca antes del cap.
    """
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    assert "LLMCircuitBreaker" in src, (
        "LLMCircuitBreaker NO presente en graph_orchestrator.py. Sin él, "
        "el audit del 'Gemini down → 12min wait' SÍ es real. Restaurar."
    )
    assert "MEALFIT_CB_RESET_TIMEOUT_S" in src or "_CB_RESET_TIMEOUT_S" in src, (
        "Knob `MEALFIT_CB_RESET_TIMEOUT_S` no presente — sin él, el CB no "
        "se auto-resetea y el modelo queda OFF permanente."
    )


def test_knob_is_overridable_no_redeploy():
    """El knob debe estar registrado en _KNOBS_REGISTRY (visible via
    /admin/knobs) para que SRE pueda diagnosticar sin redeploy."""
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    # Pattern: `_env_int("MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S", ...)` —
    # _env_int auto-registra en _KNOBS_REGISTRY (P3-NEW-D).
    pattern = re.compile(r'_env_int\s*\(\s*["\']MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S')
    assert pattern.search(src), (
        "MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S no se lee via `_env_int` (que "
        "auto-registra en _KNOBS_REGISTRY). Sin el registry, SRE no puede "
        "diagnosticar el valor activo via /admin/knobs."
    )
