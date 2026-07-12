"""[P1-PLANDATA-ID-HYDRATE · 2026-07-12] Todo path que hidrata `planData` desde
`/api/plans-data/latest` adjunta el `id` del row.

Caso vivo (2026-07-12 ~05:00Z): "No encontramos tu plan activo" al pulsar Actualizar Día.
`plan_data` (JSONB) NO trae el id (vive en la columna); el poller de chunks y los resumes
(day-regen/swap) tenían ramas `if (!prev) return plan.plan_data` que dejaban `planData`
SIN id cuando ganaban la carrera de montaje (el chunk nocturno se mergeó con la pestaña
abierta) → `regenerateDay`/persists client-side fallaban su guard de plan_id.

Contrato (parser sobre AssessmentContext.jsx):
1. El poller de chunks adjunta el id antes del return-fresh Y en el merge.
2. Los DOS resumes (day-regen y swap) adjuntan el id igual.
3. restoreSessionData conserva su hidratación original (`latestPlan.id = planId`).

tooltip-anchor: P1-PLANDATA-ID-HYDRATE
"""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_CTX = (_ROOT / "frontend" / "src" / "context" / "AssessmentContext.jsx").read_text(encoding="utf-8")


def test_all_fresh_return_paths_attach_id():
    assert _CTX.count("P1-PLANDATA-ID-HYDRATE") >= 3, (
        "3 sitios: poller de chunks + resume day-regen + resume swap — plan_data pelado "
        "sin id rompe los guards downstream ('No encontramos tu plan activo', vivo)"
    )
    assert _CTX.count("pdNew.id == null && plan?.id != null) pdNew.id = plan.id") == 2, (
        "los DOS resumes adjuntan el id antes del return-fresh"
    )
    assert "newPlanData.id == null && plan?.id != null) newPlanData.id = plan.id" in _CTX, (
        "el poller de chunks adjunta el id antes del return-fresh"
    )


def test_merges_repair_idless_prev():
    assert _CTX.count("id: prev.id ?? plan?.id") >= 3, (
        "los merges reparan estados previos que quedaron sin id (no solo el path fresh)"
    )


def test_restore_session_hydration_intact():
    assert "latestPlan.id = planId" in _CTX, (
        "la hidratación original de restoreSessionData sigue siendo la fuente primaria"
    )
