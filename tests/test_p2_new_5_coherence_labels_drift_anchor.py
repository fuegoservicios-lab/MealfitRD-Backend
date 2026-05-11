"""[P2-NEW-5 · 2026-05-10] Anchor — drift detection cross-language para
coherence labels YA está cubierta por 2 tests existentes.

El P2-NEW-5 reportado por el auditor pedía "test parser-based estilo
P0-form-6 para enforzar paridad entre `getCoherenceActionLabel` /
`getCoherenceHypothesisLabel` (frontend) y `_COHERENCE_ANOMALOUS_ACTIONS`
(backend)".

Verificación post-audit:
  Ya existen DOS tests con esa cobertura:

  1. `test_p1_3_coherence_labels_cross_language.py` (P1-3, 2026-05-10):
       Parsea `frontend/src/utils/coherenceLabels.js` +
       `backend/shopping_calculator.py::_classify_divergence_hypothesis` +
       `backend/graph_orchestrator.py::review_plan_node|assemble_plan_node|
       _recompute_aggregates_after_swap`. Enforza paridad:
       - Cada `hypothesis` emitido por backend está en JS.
       - Cada `action_taken` emitido por backend está en JS.
       - Inversamente: JS no expone códigos que el backend no emita.
       - Helpers `getCoherenceActionLabel/HypothesisLabel` con firma esperada.

  2. `test_p2_hist_audit_13_coherence_anomalous_ssot.py` (P2-HIST-AUDIT-13,
     2026-05-09): Parsea `backend/constants.py::COHERENCE_ANOMALOUS_ACTIONS`
     vs `frontend/src/utils/coherenceActions.js::COHERENCE_ANOMALOUS_ACTIONS`.
     Enforza paridad cross-language del set anomalous.

Sin cambios funcionales en P2-NEW-5. Este archivo solo:
  - Sirve como anchor para el marker test (slug `p2_new_5` → glob).
  - Verifica que los 2 tests-de-drift siguen presentes en disco (si alguien
    los borra/renombra sin reemplazar, este test falla).

Background completo en
`~/.claude/projects/.../memory/project_p2_new_5_coherence_labels_drift_anchor_2026_05_10.md`.
"""
from pathlib import Path


_TESTS_DIR = Path(__file__).resolve().parent
_P1_3_LABELS = _TESTS_DIR / "test_p1_3_coherence_labels_cross_language.py"
_P2_HIST_13 = _TESTS_DIR / "test_p2_hist_audit_13_coherence_anomalous_ssot.py"


def test_p1_3_coherence_labels_drift_test_present():
    assert _P1_3_LABELS.exists(), (
        f"Falta `test_p1_3_coherence_labels_cross_language.py` — el drift "
        f"detection cross-language para action_taken + hypothesis labels. "
        f"Sin él, un nuevo code backend sin entrada en `coherenceLabels.js` "
        f"renderizaría crudo al usuario sin que CI lo detecte."
    )


def test_p2_hist_audit_13_anomalous_drift_test_present():
    assert _P2_HIST_13.exists(), (
        f"Falta `test_p2_hist_audit_13_coherence_anomalous_ssot.py` — el "
        f"drift detection cross-language para COHERENCE_ANOMALOUS_ACTIONS. "
        f"Sin él, un nuevo action_taken anomalous quedaría sin contar en "
        f"el chip del Historial."
    )
