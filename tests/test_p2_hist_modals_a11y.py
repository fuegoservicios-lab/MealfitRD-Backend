"""[P2-HIST-MODALS-A11Y · 2026-05-30] Audit FOCALIZADO del Historial.

Bundle 1 P2 + 2 P3 cerrados en el audit prod-readiness del Historial
2026-05-30 (la página `/history` + toda su superficie de dependencias).
Backend de lectura y mutación auditado limpio (IDOR/lost-update/fail-secure
ya endurecidos en pasadas previas); estos 3 son los gaps residuales:

  GAP-1 (P2) P2-HIST-MODALS-A11Y:
    Los 3 modales custom del Historial (`History.jsx`: detalle del plan +
    confirm "Reactivar" + confirm "Eliminar", los 2 últimos DESTRUCTIVOS)
    eran `motion.div` con click-backdrop como única vía de cierre — sin
    `role="dialog"`, `aria-modal`, ESC, focus-trap ni restore-focus. Outlier
    respecto a PaymentModal / LogoutConfirmModal / Dashboard restock / los 3
    modales de Pantry (P2-PANTRY-MODALS-A11Y), que ya usan el hook SSOT
    `useModalAccessibility` (P2-CUSTOM-MODALS-A11Y). Fix: aplicar el hook a
    los 3 + role/aria/labelledby. `onClose` memoizado con `useCallback`
    (el hook tiene onClose en sus deps de useEffect; el modal de detalle
    re-renderiza mucho y un arrow inline robaría el focus en cada render).

  GAP-2 (P3) P3-HIST-MODAL-CACHE-XUSER:
    `_clearUserScopedCaches` (AssessmentContext.jsx) limpiaba el cache del
    LISTADO del Historial (P1-XTAB-CACHE-LEAK) pero NO los 5 caches singleton
    per-plan del modal (lessons/coherence/blocked/metrics/lifetime), `Map`
    module-scope keyed por plan_id (UUID) con PII nutricional del usuario
    anterior que sobrevive al logout SPA (navigate, sin reload). Fix: nuevo
    export `clearAllModalCaches()` en historyCaches.js invocado en el logout.

  GAP-3 (P3) defense-in-depth (mismo marker):
    El re-enqueue `plan_chunk_queue` de `regenerate-simplified`
    (routers/plans.py) filtraba `WHERE id = %s` (chunk_id) sin el predicado
    redundante `AND meal_plan_id = %s` que su hermano `retry-chunk`
    (P0-HIST-IDOR-1) y el UPDATE de `meal_plans` de abajo (P1-NEW-4) sí
    llevan. No explotable hoy (ownership ya validado upstream), pero un
    refactor que rompa el check upstream re-introduciría re-enqueue
    cross-plan. Fix: añadir `AND meal_plan_id = %s`.

Test 100% parser-based (lee archivos, no importa módulos de la app ni toca
DB) → corre en venv DB-less. Cross-link guard P2-HIST-AUDIT-14: el slug
`p2_hist_modals_a11y` matchea el marker `P2-HIST-MODALS-A11Y`.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _REPO_ROOT / "backend"
_APP_PY = _BACKEND / "app.py"
_PLANS_ROUTER = _BACKEND / "routers" / "plans.py"
_FRONTEND = _REPO_ROOT / "frontend" / "src"
_HISTORY = _FRONTEND / "pages" / "History.jsx"
_HISTORY_CACHES = _FRONTEND / "utils" / "historyCaches.js"
_ASSESSMENT_CTX = _FRONTEND / "context" / "AssessmentContext.jsx"
_USE_MODAL_A11Y = _FRONTEND / "hooks" / "useModalAccessibility.js"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Sección 1 — GAP-1 P2-HIST-MODALS-A11Y (History.jsx)
# ---------------------------------------------------------------------------
def test_hook_ssot_exists():
    """El hook SSOT debe seguir existiendo (compartido con Pantry/Payment/etc.)."""
    assert _USE_MODAL_A11Y.exists(), "Falta frontend/src/hooks/useModalAccessibility.js."


def test_history_anchor_present():
    src = _read(_HISTORY)
    assert "P2-HIST-MODALS-A11Y" in src, "Falta anchor `P2-HIST-MODALS-A11Y` en History.jsx."


def test_history_imports_a11y_hook():
    src = _read(_HISTORY)
    assert re.search(
        r"import\s*\{[^}]*useModalAccessibility[^}]*\}\s*from\s*['\"]\.\./hooks/useModalAccessibility['\"]",
        src,
    ), "History.jsx debe importar useModalAccessibility desde ../hooks/useModalAccessibility."


def test_history_invokes_hook_for_three_modals():
    """3 invocaciones del hook (detalle + restore-confirm + delete-confirm)."""
    src = _read(_HISTORY)
    calls = len(re.findall(r"useModalAccessibility\s*\(", src))
    assert calls == 3, (
        f"Se esperaban 3 invocaciones de useModalAccessibility en History.jsx "
        f"(detalle + reactivar + eliminar); se encontraron {calls}."
    )


def test_history_detail_modal_trap_gated_under_confirms():
    """El trap del modal de detalle se desactiva mientras un confirm está
    apilado encima — solo UN focus-trap + UN ESC handler activo a la vez."""
    src = _read(_HISTORY)
    assert re.search(
        r"isOpen:\s*!!selectedPlan\s*&&\s*!confirmRestore\s*&&\s*!confirmDelete",
        src,
    ), (
        "El modal de detalle debe gatear `isOpen` con "
        "`!!selectedPlan && !confirmRestore && !confirmDelete` para no apilar "
        "dos focus-traps cuando el confirm de reactivar se abre desde su footer."
    )


def test_history_onclose_memoized_with_usecallback():
    """onClose memoizado: el hook tiene onClose en deps de useEffect; un arrow
    inline re-correría el effect en cada render y robaría el focus."""
    src = _read(_HISTORY)
    assert "useCallback" in src, "History.jsx debe importar/usar useCallback."
    for name in ("_closeDetailModal", "_closeRestoreConfirm", "_closeDeleteConfirm"):
        assert re.search(rf"const\s+{name}\s*=\s*useCallback\(", src), (
            f"`{name}` debe memoizarse con useCallback para estabilizar onClose."
        )


def test_history_modal_roots_have_role_and_aria():
    src = _read(_HISTORY)
    assert len(re.findall(r'role="dialog"', src)) >= 3, (
        'Los 3 modales del Historial deben tener role="dialog".'
    )
    assert len(re.findall(r'aria-modal="true"', src)) >= 3, (
        'Los 3 modales del Historial deben tener aria-modal="true".'
    )
    # Cada modal: aria-labelledby apunta a un id que existe en su heading.
    for title_id in (
        "history-detail-title",
        "history-restore-confirm-title",
        "history-delete-confirm-title",
    ):
        assert re.search(rf'aria-labelledby="{title_id}"', src), (
            f'Falta aria-labelledby="{title_id}".'
        )
        assert re.search(rf'id="{title_id}"', src), (
            f"Falta el id={title_id} en el heading correspondiente."
        )


def test_history_confirm_modals_have_describedby():
    """Los 2 confirms (destructivos) describen su cuerpo para screen readers."""
    src = _read(_HISTORY)
    for desc_id in ("history-restore-confirm-desc", "history-delete-confirm-desc"):
        assert re.search(rf'aria-describedby="{desc_id}"', src), (
            f'Falta aria-describedby="{desc_id}" en el confirm.'
        )
        assert re.search(rf'id="{desc_id}"', src), (
            f"Falta id={desc_id} en el texto del confirm."
        )


# ---------------------------------------------------------------------------
# Sección 2 — GAP-2 P3-HIST-MODAL-CACHE-XUSER (historyCaches.js + ctx)
# ---------------------------------------------------------------------------
def test_clear_all_modal_caches_exported_and_clears_five_maps():
    src = _read(_HISTORY_CACHES)
    assert "P3-HIST-MODAL-CACHE-XUSER" in src, (
        "Falta anchor `P3-HIST-MODAL-CACHE-XUSER` en historyCaches.js."
    )
    m = re.search(r"export\s+const\s+clearAllModalCaches\s*=\s*\(\s*\)\s*=>\s*\{(.*?)\}", src, re.DOTALL)
    assert m, "Falta `export const clearAllModalCaches = () => { ... }` en historyCaches.js."
    body = m.group(1)
    for cache in (
        "_lessonsDetail",
        "_coherenceHistory",
        "_blockedReasons",
        "_chunkMetrics",
        "_lifetimeLessons",
    ):
        assert f"{cache}.clear()" in body, (
            f"clearAllModalCaches debe limpiar `{cache}` (uno de los 5 caches del modal)."
        )


def test_assessment_clears_modal_caches_on_logout():
    src = _read(_ASSESSMENT_CTX)
    assert re.search(
        r"import\s*\{[^}]*clearAllModalCaches[^}]*\}\s*from\s*['\"]\.\./utils/historyCaches['\"]",
        src,
    ), "AssessmentContext debe importar clearAllModalCaches de ../utils/historyCaches."
    # Debe invocarse DENTRO de _clearUserScopedCaches.
    m = re.search(r"const\s+_clearUserScopedCaches\s*=\s*\(\s*\)\s*=>\s*\{(.*?)\n\};", src, re.DOTALL)
    assert m, "No se encontró el cuerpo de `_clearUserScopedCaches`."
    assert "clearAllModalCaches()" in m.group(1), (
        "`_clearUserScopedCaches` debe invocar `clearAllModalCaches()` (limpia los "
        "5 caches per-plan del modal en logout/user-switch — hermano de P1-XTAB-CACHE-LEAK)."
    )


# ---------------------------------------------------------------------------
# Sección 3 — GAP-3 defense-in-depth regenerate-simplified (plans.py)
# ---------------------------------------------------------------------------
def test_regenerate_simplified_reenqueue_filters_meal_plan_id():
    """El re-enqueue de `plan_chunk_queue` en regenerate-simplified debe
    filtrar `WHERE id = %s AND meal_plan_id = %s` (defense-in-depth, mismo
    patrón que retry-chunk / el UPDATE de meal_plans de abajo)."""
    src = _read(_PLANS_ROUTER)
    fn_start = src.find("def api_regenerate_dead_lettered_simplified")
    assert fn_start >= 0, "No se encontró api_regenerate_dead_lettered_simplified."
    fn_body = src[fn_start: fn_start + 6000]
    # El UPDATE de re-enqueue es único por su SET status='pending' +
    # pipeline_snapshot; su WHERE debe incluir meal_plan_id.
    assert re.search(
        r"UPDATE plan_chunk_queue.*?pipeline_snapshot\s*=\s*%s::jsonb,.*?"
        r"WHERE id = %s AND meal_plan_id = %s",
        fn_body,
        re.DOTALL,
    ), (
        "El re-enqueue de plan_chunk_queue debe filtrar "
        "`WHERE id = %s AND meal_plan_id = %s` (no solo por chunk_id)."
    )
    # Los params deben incluir plan_id como tercer bind.
    assert re.search(
        r"json\.dumps\(snap,\s*ensure_ascii=False\),\s*chunk_id,\s*plan_id\)",
        fn_body,
    ), "Los params del re-enqueue deben pasar `plan_id` como tercer bind."


# ---------------------------------------------------------------------------
# Sección 4 — Marker bumped + date-floor
# ---------------------------------------------------------------------------
def test_marker_bumped_meets_floor():
    """`_LAST_KNOWN_PFIX` con fecha >= floor del bundle (2026-05-30).
    Floor-based (no exact-match) para sobrevivir supersedes futuros — el
    exact-match vive en test_p3_1_last_known_pfix_freshness (floor global)."""
    text = _read(_APP_PY)
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    assert m, "_LAST_KNOWN_PFIX no encontrado en app.py."
    marker = m.group(1)
    date_m = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert date_m, f"Marker `{marker}` no contiene fecha ISO."
    marker_date = datetime.strptime(date_m.group(1), "%Y-%m-%d").date()
    assert marker_date >= date(2026, 5, 30), (
        f"Marker `{marker}` (fecha {marker_date}) regresó por debajo del "
        f"floor de P2-HIST-MODALS-A11Y (2026-05-30)."
    )


# ---------------------------------------------------------------------------
# Sección 5 — Cross-link guard P2-HIST-AUDIT-14
# ---------------------------------------------------------------------------
def test_anchor_present_in_test_file():
    src = _read(Path(__file__))
    assert "P2-HIST-MODALS-A11Y" in src
