"""[P3-NEW-1 · 2026-05-10] Anchor backend — defense-in-depth en PDF render
contra `_shopping_coherence_block` no consumido.

El fix REAL vive en
`frontend/src/utils/shoppingHelpers.js::getActiveShoppingList`
+ `frontend/src/__tests__/utils/shoppingHelpers.p3_new_1_coherence_block_warn.test.js`.

Este archivo Python solo valida que el test frontend exista — anchor mínimo
para que el marker test cross-link funcione.

Background completo en
`~/.claude/projects/.../memory/project_p3_new_1_pdf_coherence_block_defense_2026_05_10.md`.
"""
from pathlib import Path

_FRONTEND_TEST = (
    Path(__file__).resolve().parents[2]
    / "frontend"
    / "src"
    / "__tests__"
    / "utils"
    / "shoppingHelpers.p3_new_1_coherence_block_warn.test.js"
)


def test_frontend_p3_new_1_test_exists():
    assert _FRONTEND_TEST.exists(), (
        f"Falta el test frontend del P-fix P3-NEW-1 en {_FRONTEND_TEST}. "
        f"Sin él, el anchor del marker queda huérfano."
    )


def test_frontend_test_has_marker_anchor():
    text = _FRONTEND_TEST.read_text(encoding="utf-8")
    assert "[P3-NEW-1" in text, (
        f"Falta anchor `[P3-NEW-1` en el test frontend."
    )
