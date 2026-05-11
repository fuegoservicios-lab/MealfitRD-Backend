"""[P2-NEW-4 · 2026-05-10] Anchor en backend/tests/ para que el marker
test (test_p2_hist_audit_14_marker_test_link.py) encuentre cobertura del
P-fix vía glob `test_p2_new_4*.py`.

El fix REAL vive en `frontend/src/__tests__/History.p2_new_4_visibilitychange_all_caches.test.js`
(9 tests parser-based contra History.jsx que verifican que el listener
`visibilitychange` invalida los 6 caches dependientes de plan).

Este archivo Python solo valida que el test frontend exista en disco con
el slug correcto — es el anchor mínimo para que el cross-link funcione.

Background completo en
`~/.claude/projects/.../memory/project_p2_new_4_visibilitychange_contract_2026_05_10.md`.
"""
from pathlib import Path

_FRONTEND_TEST = (
    Path(__file__).resolve().parents[2]
    / "frontend"
    / "src"
    / "__tests__"
    / "History.p2_new_4_visibilitychange_all_caches.test.js"
)


def test_frontend_visibilitychange_test_exists():
    assert _FRONTEND_TEST.exists(), (
        f"Falta el test frontend del P-fix P2-NEW-4 en {_FRONTEND_TEST}. "
        f"Sin él, el anchor del marker queda huérfano y un futuro audit no "
        f"puede confirmar cobertura del listener visibilitychange."
    )


def test_frontend_test_marker_anchor_present():
    """El test frontend debe llevar el anchor textual `[P2-NEW-4 · ...]` —
    rotura del anchor implica que probablemente alguien renombró/borró el
    test sin renombrar este stub."""
    text = _FRONTEND_TEST.read_text(encoding="utf-8")
    assert "[P2-NEW-4" in text, (
        f"Falta el anchor `[P2-NEW-4` en el test frontend. Renombrar/borrar "
        f"sin sincronizar el anchor invalida la trazabilidad."
    )
