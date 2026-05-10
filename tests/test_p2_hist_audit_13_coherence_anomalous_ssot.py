"""[P2-HIST-AUDIT-13 · 2026-05-09] Tests del SSOT
``constants.COHERENCE_ANOMALOUS_ACTIONS`` y drift detection
cross-archivo entre el backend (Python) y el frontend (JS).

Bug original (audit Historial 2026-05-09):
    El set `_ANOMALOUS_COHERENCE_ACTIONS` vivía en
    `routers/plans.py` (4 strings inline) y se replicaba en
    `frontend/src/pages/History.jsx` como OR encadenado de 4
    string literals. Cuando `_aggregate_coherence_block_history_metrics`
    (P3-B) añadió `post_swap_revalidation` y otros buckets, ambos
    sites tuvieron que actualizarse a mano. Cualquier adición
    futura al catálogo tenía la misma trampa.

Fix:
    SSOT en `constants.COHERENCE_ANOMALOUS_ACTIONS` (tuple, mismo
    patrón que `LESSON_COUNT_EVENT_WHITELIST` de P1-AUDIT-HIST-7).
    Backend importa con alias retrocompat. Frontend tiene mirror
    `frontend/src/utils/coherenceActions.js` con `Set` + helper
    `isAnomalousCoherenceAction`. Drift detection compara los dos.

Cobertura:
    1. Anchor del marker en constants + routers/plans + JS.
    2. Constante exportada en `constants.py` con los 4 valores.
    3. routers/plans.py importa desde constants (no inline literal).
    4. Identity check: el alias local es frozenset del SSOT tuple.
    5. Frontend JS tiene SSOT con los mismos 4 valores.
    6. Drift detection: el set Python coincide exactamente con el
       set JS (parsea ambos archivos source).
    7. History.jsx usa el helper `isAnomalousCoherenceAction`,
       NO los 4 string literals inline.
    8. NO hay redefinición inline en otros módulos del backend.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
_CONSTANTS_PATH = _BACKEND_ROOT / "constants.py"
_PLANS_PATH = _BACKEND_ROOT / "routers" / "plans.py"
_FRONTEND_HELPER_PATH = (
    _REPO_ROOT / "frontend" / "src" / "utils" / "coherenceActions.js"
)
_FRONTEND_HISTORY_PATH = (
    _REPO_ROOT / "frontend" / "src" / "pages" / "History.jsx"
)

_EXPECTED_SET = frozenset({
    "degrade",
    "reject_minor",
    "reject_high",
    "hydration_error",
})


# ---------------------------------------------------------------------------
# 1. Anchor del marker en los 3 archivos
# ---------------------------------------------------------------------------
def test_marker_in_constants_py():
    text = _CONSTANTS_PATH.read_text(encoding="utf-8")
    assert "P2-HIST-AUDIT-13" in text


def test_marker_in_plans_py():
    text = _PLANS_PATH.read_text(encoding="utf-8")
    assert "P2-HIST-AUDIT-13" in text


def test_marker_in_frontend_helper():
    text = _FRONTEND_HELPER_PATH.read_text(encoding="utf-8")
    assert "P2-HIST-AUDIT-13" in text


def test_marker_in_history_jsx():
    text = _FRONTEND_HISTORY_PATH.read_text(encoding="utf-8")
    assert "P2-HIST-AUDIT-13" in text


# ---------------------------------------------------------------------------
# 2. Backend SSOT
# ---------------------------------------------------------------------------
def test_constants_exports_coherence_anomalous_actions():
    """`constants.COHERENCE_ANOMALOUS_ACTIONS` (público, sin
    underscore) debe existir y contener exactamente los 4 actions."""
    import constants
    actions = getattr(constants, "COHERENCE_ANOMALOUS_ACTIONS", None)
    assert actions is not None
    assert isinstance(actions, tuple), (
        f"COHERENCE_ANOMALOUS_ACTIONS debe ser tuple (immutable). "
        f"Got: {type(actions)}"
    )
    assert set(actions) == _EXPECTED_SET


def test_plans_py_imports_from_constants():
    """`routers/plans.py` debe importar `COHERENCE_ANOMALOUS_ACTIONS`
    desde constants (no redefinir el set inline)."""
    text = _PLANS_PATH.read_text(encoding="utf-8")
    assert re.search(
        r"from\s+constants\s+import\s+[^#\n]*COHERENCE_ANOMALOUS_ACTIONS",
        text,
        re.IGNORECASE,
    ), (
        "routers/plans.py debe importar COHERENCE_ANOMALOUS_ACTIONS "
        "desde `constants` (SSOT)."
    )


def test_plans_py_does_not_redefine_set_inline():
    """Defensa contra regresión: el archivo NO debe contener un
    set/dict literal con los 4 strings juntos (eso indicaría que
    alguien redefinió el set en lugar de importar)."""
    text = _PLANS_PATH.read_text(encoding="utf-8")
    # Patrón: { o ( seguido de los 4 strings (en cualquier orden).
    redef = re.search(
        r"=\s*\{\s*"
        r"['\"]degrade['\"]\s*,\s*"
        r"['\"]reject_minor['\"]\s*,\s*"
        r"['\"]reject_high['\"]\s*,\s*"
        r"['\"]hydration_error['\"]",
        text,
    )
    assert not redef, (
        "routers/plans.py contiene una redefinición inline del set. "
        "El SSOT vive en constants.py — importar desde allá."
    )


# ---------------------------------------------------------------------------
# 3. Frontend SSOT
# ---------------------------------------------------------------------------
def _extract_js_set_literals(text: str) -> set[str]:
    """Extrae los strings de un Set literal `new Set([...])` en el
    helper JS. Si hay múltiples Sets, devuelve la unión."""
    out = set()
    for block in re.finditer(
        r"new\s+Set\(\s*\[\s*([^\]]+)\s*\]\s*\)", text
    ):
        for s in re.findall(r"['\"]([^'\"]+)['\"]", block.group(1)):
            out.add(s)
    return out


def test_frontend_helper_exports_set_with_canonical_actions():
    """`coherenceActions.js` debe exportar `COHERENCE_ANOMALOUS_ACTIONS`
    con los 4 actions canónicos (mirror del backend)."""
    text = _FRONTEND_HELPER_PATH.read_text(encoding="utf-8")
    assert re.search(
        r"export\s+const\s+COHERENCE_ANOMALOUS_ACTIONS\s*=\s*new\s+Set",
        text,
    ), "Helper JS debe exportar `COHERENCE_ANOMALOUS_ACTIONS` como Set."
    js_set = _extract_js_set_literals(text)
    assert js_set == _EXPECTED_SET, (
        f"Set JS diverge del expected:\n"
        f"  expected: {sorted(_EXPECTED_SET)}\n"
        f"  actual:   {sorted(js_set)}"
    )


def test_frontend_helper_exports_isAnomalous_function():
    text = _FRONTEND_HELPER_PATH.read_text(encoding="utf-8")
    assert re.search(
        r"export\s+const\s+isAnomalousCoherenceAction\s*=", text
    ), "Helper JS debe exportar `isAnomalousCoherenceAction(action)`."


# ---------------------------------------------------------------------------
# 4. Drift detection cross-archivo (Python ↔ JS)
# ---------------------------------------------------------------------------
def test_python_and_js_sets_match_exactly():
    """El set de Python (constants.py) y el set de JS
    (coherenceActions.js) deben coincidir EXACTAMENTE. Esto cierra
    el riesgo de divergencia que originó el audit."""
    import constants
    py_set = set(constants.COHERENCE_ANOMALOUS_ACTIONS)
    js_text = _FRONTEND_HELPER_PATH.read_text(encoding="utf-8")
    js_set = _extract_js_set_literals(js_text)
    assert py_set == js_set, (
        f"DRIFT cross-language detectado.\n"
        f"  Python (constants.py): {sorted(py_set)}\n"
        f"  JS (coherenceActions.js): {sorted(js_set)}\n"
        f"Si añadiste un action a uno, añádelo al OTRO en el mismo "
        f"commit."
    )


# ---------------------------------------------------------------------------
# 5. History.jsx usa el helper, no string literals inline
# ---------------------------------------------------------------------------
def test_history_jsx_uses_helper_not_inline_literals():
    """El render `getCoherenceAdjustsCount` en History.jsx debe usar
    `isAnomalousCoherenceAction(...)`. No debe quedar el OR
    encadenado original."""
    text = _FRONTEND_HISTORY_PATH.read_text(encoding="utf-8")
    # Helper presente.
    assert "isAnomalousCoherenceAction" in text
    # Patrón regresión: el OR encadenado de 4 string literals NO debe
    # estar (eso sería rollback al estado pre-fix).
    bad_pattern = re.search(
        r"action\s*===\s*['\"]degrade['\"]\s*\|\|\s*"
        r"action\s*===\s*['\"]reject_minor['\"]",
        text,
    )
    assert not bad_pattern, (
        "History.jsx contiene el OR encadenado pre-fix "
        "(`action === 'degrade' || action === 'reject_minor' ...`). "
        "El SSOT helper `isAnomalousCoherenceAction` lo reemplaza."
    )


def test_history_jsx_imports_helper_from_utils():
    text = _FRONTEND_HISTORY_PATH.read_text(encoding="utf-8")
    assert re.search(
        r"import\s*\{[^}]*isAnomalousCoherenceAction[^}]*\}\s*from\s*"
        r"['\"]\.\.\/utils\/coherenceActions['\"]",
        text,
    ), (
        "History.jsx debe importar `isAnomalousCoherenceAction` "
        "desde `../utils/coherenceActions`."
    )


# ---------------------------------------------------------------------------
# 6. NO hay redefinición inline del set en otros módulos backend
# ---------------------------------------------------------------------------
def test_no_other_backend_file_redefines_set():
    """Recorrer el árbol del backend buscando redefiniciones inline
    del set. Solo `constants.py` (SSOT) debe contenerlo;
    `routers/plans.py` lo importa.
    """
    redef_pattern = re.compile(
        r"=\s*\{\s*"
        r"['\"]degrade['\"]\s*,\s*"
        r"['\"]reject_minor['\"]\s*,\s*"
        r"['\"]reject_high['\"]\s*,\s*"
        r"['\"]hydration_error['\"]",
    )
    offenders = []
    for py_path in _BACKEND_ROOT.rglob("*.py"):
        rel = py_path.relative_to(_BACKEND_ROOT).as_posix()
        if (
            rel == "constants.py"
            or rel.startswith("venv/")
            or rel.startswith(".venv/")
            or rel.startswith("tests/")
        ):
            continue
        try:
            text = py_path.read_text(encoding="utf-8")
        except Exception:
            continue
        if redef_pattern.search(text):
            offenders.append(rel)
    assert not offenders, (
        f"Estos archivos REDEFINEN el set inline en lugar de "
        f"importar de constants: {offenders}"
    )
