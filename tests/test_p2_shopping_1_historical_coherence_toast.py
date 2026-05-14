"""[P2-SHOPPING-1 · 2026-05-14] Anchor parser-based del integration en
`Dashboard.jsx::handleDownloadShoppingList`.

El test unitario JS [`renderCoherenceWarnings.p2_shopping_1_historical_toast.test.js`](frontend/src/__tests__/utils/renderCoherenceWarnings.p2_shopping_1_historical_toast.test.js)
cubre la lógica del helper. Este test parser-based ancla el callsite que
INVOCA el helper en el flujo del PDF — si alguien borra la invocación
(pensando que `emitCoherenceToast` post-recalc basta), el bug del audit
2026-05-14 reaparece: usuarios que descargan PDF sin pasar por recalc
no ven la telemetría del `_shopping_coherence_block_history`.

Strategy:
    1. Verificar `emitHistoricalCoherenceToast` importado en Dashboard.jsx.
    2. Verificar que el handler `handleDownloadShoppingList` invoca el
       helper con `effectivePlanData?._shopping_coherence_block_history`.
    3. Verificar que la invocación está en try/catch best-effort (un fallo
       del toast no debe abortar la descarga del PDF).
    4. Verificar el anchor `[P2-SHOPPING-1 · 2026-05-14]` en comment.

Tooltip-anchor: P2-SHOPPING-1.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DASH_FP = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"
_HELPER_FP = _REPO_ROOT / "frontend" / "src" / "utils" / "renderCoherenceWarnings.js"


@pytest.fixture(scope="module")
def dash_src() -> str:
    return _DASH_FP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def helper_src() -> str:
    return _HELPER_FP.read_text(encoding="utf-8")


def _extract_handler_block(src: str) -> str:
    """Devuelve el cuerpo del handler `handleDownloadShoppingList` desde su
    declaración hasta el siguiente `const handle*` top-level del componente.
    """
    start = src.find("const handleDownloadShoppingList")
    assert start > 0, "handleDownloadShoppingList no encontrado en Dashboard.jsx"
    after = src[start + 50:]
    next_match = re.search(r"\n    const handle[A-Z]", after)
    end = (start + 50 + next_match.start()) if next_match else (start + 50 + 6000)
    return src[start:end]


# ---------------------------------------------------------------------------
# Helper exists + symbol exported
# ---------------------------------------------------------------------------

def test_helper_module_exports_emit_historical_coherence_toast(helper_src: str):
    """`emitHistoricalCoherenceToast` se exporta desde renderCoherenceWarnings.js."""
    assert re.search(
        r"export\s+const\s+emitHistoricalCoherenceToast\s*=",
        helper_src,
    ), (
        "P2-SHOPPING-1 regresión: `emitHistoricalCoherenceToast` ya no se "
        "exporta desde `renderCoherenceWarnings.js`. Sin el helper no hay "
        "telemetría histórica al usuario en el flujo PDF."
    )


def test_helper_module_exports_build_historical_coherence_toast(helper_src: str):
    """El builder también se exporta (para tests unitarios sin sonner)."""
    assert re.search(
        r"export\s+const\s+buildHistoricalCoherenceToast\s*=",
        helper_src,
    ), (
        "P2-SHOPPING-1 regresión: `buildHistoricalCoherenceToast` ya no se "
        "exporta. Sin el builder, los tests unitarios JS no pueden validar "
        "la lógica de filtrado sin mockear sonner."
    )


def test_helper_filters_blacklisted_action_types(helper_src: str):
    """El helper ignora `not_applicable` y `hydration_error` (P2-2 invariant
    placeholder + invariant violation no user-facing)."""
    assert "not_applicable" in helper_src, (
        "P2-SHOPPING-1 regresión: el blacklist ya no menciona "
        "`not_applicable`. Sin él, el warn-mode placeholder emite "
        "toasts espurios al usuario."
    )
    assert "hydration_error" in helper_src, (
        "P2-SHOPPING-1 regresión: el blacklist ya no menciona "
        "`hydration_error`. Sin él, una invariant violation interna se "
        "expone como toast al usuario final."
    )


# ---------------------------------------------------------------------------
# Dashboard.jsx integration callsite
# ---------------------------------------------------------------------------

def test_dashboard_imports_emit_historical_coherence_toast(dash_src: str):
    """Dashboard.jsx importa `emitHistoricalCoherenceToast` del helper."""
    # Acepta tanto import único como combined: `import { emitCoherenceToast, emitHistoricalCoherenceToast }`.
    assert re.search(
        r"import\s+\{[^}]*emitHistoricalCoherenceToast[^}]*\}\s+from\s+['\"][^'\"]*renderCoherenceWarnings['\"]",
        dash_src,
    ), (
        "P2-SHOPPING-1 regresión: `Dashboard.jsx` ya no importa "
        "`emitHistoricalCoherenceToast`. El handler PDF no puede emitir "
        "el toast histórico."
    )


def test_handler_invokes_historical_helper(dash_src: str):
    """`handleDownloadShoppingList` invoca el helper con history del plan."""
    body = _extract_handler_block(dash_src)
    assert "emitHistoricalCoherenceToast(" in body, (
        "P2-SHOPPING-1 regresión: el handler PDF ya NO invoca "
        "`emitHistoricalCoherenceToast`. Sin la llamada, el flujo "
        "'abrir Dashboard → descargar PDF directo' no muestra al "
        "usuario la telemetría de `_shopping_coherence_block_history` "
        "(escrita por chunk worker T2, cron diario, agent_tool, etc.)."
    )


def test_handler_reads_history_from_effective_plan_data(dash_src: str):
    """La invocación pasa `effectivePlanData?._shopping_coherence_block_history`
    (resultado del prefetch P2-NEW-14), NO `planData` stale."""
    body = _extract_handler_block(dash_src)
    # Buscar la ventana cercana a `emitHistoricalCoherenceToast(`.
    call_match = re.search(
        r"emitHistoricalCoherenceToast\s*\([^)]*\)",
        body,
        re.DOTALL,
    )
    assert call_match, (
        "P2-SHOPPING-1 regresión: invocación `emitHistoricalCoherenceToast(...)` "
        "no extraíble. Verificar paréntesis multi-línea o variable rename."
    )
    call_block = call_match.group(0)
    assert "effectivePlanData" in call_block, (
        "P2-SHOPPING-1 regresión: la invocación lee de `planData` o de "
        "otra variable — debe leer de `effectivePlanData` (el output del "
        "prefetch P2-NEW-14). Sin eso, el toast usa snapshot stale del "
        "history aunque el chunk worker haya actualizado en background."
    )
    assert "_shopping_coherence_block_history" in call_block, (
        "P2-SHOPPING-1 regresión: la invocación ya no accede a "
        "`_shopping_coherence_block_history`. ¿Renombraron la key del "
        "backend? Verificar shopping_calculator.run_shopping_coherence_guard_"
        "and_append_history."
    )


def test_handler_wraps_call_in_try_catch(dash_src: str):
    """Best-effort: un fallo del toast no debe abortar la descarga del PDF.
    Verificar try/catch alrededor de la invocación.
    """
    body = _extract_handler_block(dash_src)
    # Pattern: try { ... emitHistoricalCoherenceToast ... } catch (...)
    # Extrae 200 chars alrededor de la invocación y busca `try {` antes +
    # `catch` después.
    idx = body.find("emitHistoricalCoherenceToast(")
    assert idx > 0
    window_back = body[max(0, idx - 400):idx]
    window_fwd = body[idx:idx + 400]
    assert "try {" in window_back, (
        "P2-SHOPPING-1 regresión: la invocación a "
        "`emitHistoricalCoherenceToast` ya NO está envuelta en `try {`. "
        "Un fallo del toast abortaría la descarga del PDF."
    )
    assert re.search(r"catch\s*\(", window_fwd), (
        "P2-SHOPPING-1 regresión: el `catch(...)` después de la invocación "
        "desapareció. Restaurar para preservar best-effort."
    )


def test_anchor_comment_present(dash_src: str):
    """Comment inline `[P2-SHOPPING-1 · 2026-05-14]` ancla el rationale.
    Sin él, un refactor podría revertir el callsite sin entender la razón.
    """
    body = _extract_handler_block(dash_src)
    assert "P2-SHOPPING-1" in body, (
        "P2-SHOPPING-1 regresión: anchor `[P2-SHOPPING-1 · ...]` "
        "desapareció del handler. Restaurar el comment que documenta "
        "por qué se invoca el helper aquí (vs solo en recalc)."
    )
