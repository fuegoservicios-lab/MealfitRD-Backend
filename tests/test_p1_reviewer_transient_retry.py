"""[P1-REVIEWER-TRANSIENT-RETRY · 2026-06-27] (mejora arquitectónica FASE B + C — auditoría workflow + crítico)

FASE B — robustez del reviewer: el handler de excepción del reviewer colapsaba TODA excepción (timeout/5xx/pool/
parse del structured-output) en `severity="critical"` → abortaba el plan y descartaba un buen intento (visto en
vivo corr=aa61a18e: "Error en la estructura del revisor médico"). Fix: clasificar error-de-INFRAESTRUCTURA del
reviewer (`_is_reviewer_transient_error`, whitelist de TIPOS, no substring) → severity `transient_reviewer_error`:
  - should_retry lo reintenta (no es critical/high → cae al retry).
  - `_attempt_quality_rank` → 99 (NUNCA compite contra un veredicto clínico real; nunca descarta un buen intento).
  - `_severity_max` lo deja en rank 0 → un critical clínico real SIEMPRE lo enmascara (nunca al revés).

FASE C — variedad: bariátrica auto-promueve `variety_level=max` (6 comidas con 2 proteínas → repetición same-day).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import nutrition_calculator as nc

_BACKEND = Path(nc.__file__).resolve().parent


# ──────────────────────────── FASE B: clasificación de error ────────────────────────────

def test_transient_classifier_infra_errors():
    import graph_orchestrator as g
    assert g._is_reviewer_transient_error(json.JSONDecodeError("x", "", 0)) is True
    assert g._is_reviewer_transient_error(TimeoutError("slow")) is True
    assert g._is_reviewer_transient_error(asyncio.TimeoutError()) is True

    class ValidationError(Exception):  # simula pydantic.ValidationError por nombre de tipo
        pass
    assert g._is_reviewer_transient_error(ValidationError("3 validation errors")) is True


def test_transient_classifier_excludes_clinical():
    import graph_orchestrator as g
    # un error NO-infra (p.ej. un ValueError clínico) NO es transitorio → se mantiene fail-closed (critical)
    assert g._is_reviewer_transient_error(ValueError("alérgeno detectado")) is False
    assert g._is_reviewer_transient_error(KeyError("meals")) is False


# ──────────────────────────── FASE B: ranking de intentos ────────────────────────────

def test_transient_never_competes_rank99():
    import graph_orchestrator as g
    assert g._attempt_quality_rank(False, "transient_reviewer_error") == 99
    # un veredicto clínico real (incluso el peor) gana al transient
    assert g._attempt_quality_rank(False, "critical") < 99
    assert g._attempt_quality_rank(False, "high") < 99
    assert g._attempt_quality_rank(False, "minor") < 99
    assert g._attempt_quality_rank(True, None) == 0  # approved sigue siendo el mejor


def test_transient_never_masks_real_critical():
    import graph_orchestrator as g
    # un critical clínico real NUNCA queda enmascarado por un transient
    assert g._severity_max("transient_reviewer_error", "critical") == "critical"
    assert g._severity_max("critical", "transient_reviewer_error") == "critical"
    assert g._severity_max("transient_reviewer_error", "high") == "high"


# ──────────────────────────── anchors B + C ────────────────────────────

def test_anchors_fase_b_and_c():
    go = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "def _is_reviewer_transient_error" in go
    assert "transient_reviewer_error" in go
    # el catch del reviewer clasifica transient en vez de critical-siempre
    assert "_is_reviewer_transient_error(e)" in go
    ai = (_BACKEND / "ai_helpers.py").read_text(encoding="utf-8")
    # FASE C: bariátrica promueve variety_level=max
    assert "_baria_for_variety" in ai and "_GOALS_FORCE_MAX_VARIETY or _baria_for_variety" in ai
