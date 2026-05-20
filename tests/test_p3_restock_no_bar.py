"""[P3-RESTOCK-NO-BAR · 2026-05-20] Eliminación de la barra de progreso
del modal "Registrando compras".

Decisión de producto del user post-P3-RESTOCK-FLOW-SPEED del mismo día:
"no quiero que tenga una barra de carga ya que lo veo innecesario".

Pre-fix (post P3-RESTOCK-FLOW-SPEED, mid-state durante 2026-05-20):
  - Modal con icon spinner + título "Registrando compras" + descripción +
    BARRA DE PROGRESO con 3 capas (base, fill verde, sheen sweep) +
    indicador "EN PROCESO / 99%" + 3 PASOS animados con thresholds (20/55/90).
  - State acoplado: `restockPercent`, `restockFinishFast`,
    `RESTOCK_BAR_BASE_MS=1800`, `RESTOCK_BAR_FAST_FINISH_MS=250`.
  - 2 useEffect: rAF driver + watcher modal-close.
  - ~150 líneas de JSX + ~40 líneas de state/effect coupling.

Post-fix:
  - Modal queda con icon spinner + título + descripción. Nada más.
  - Cero state acoplado al progreso de la barra.
  - Cero useEffect dependiendo de `restockPercent` o `restockFinishFast`.
  - `setShowRestockModal(false)` directo post-response success.
  - Bundle Dashboard.jsx: 346.9kb → 338.8kb (-8.1kb).

Lo que se PRESERVA (heredado de P3-RESTOCK-FLOW-SPEED):
  - Cache singleton `pantryCache` populated post-refetch.
  - `invalidateInventoryCache()` pre-refetch defensive.
  - Refetch `.then()` no-awaited (paralelo).
  - Navigate síncrono sin setTimeout artificial.

Tooltip-anchor: P3-RESTOCK-NO-BAR.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DASHBOARD_FP = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"


@pytest.fixture(scope="module")
def src() -> str:
    return _DASHBOARD_FP.read_text(encoding="utf-8")


# ===========================================================================
# Elementos REMOVIDOS — la barra y todo lo que la alimentaba
# ===========================================================================

def test_progress_bar_motion_div_removed(src: str) -> None:
    """[P3-RESTOCK-NO-BAR] el `<motion.div>` con `animate={{ width:
    `${restockPercent}%` }}` que renderizaba la barra debe estar
    REMOVIDO. Si reaparece, regresa el bug 'barra innecesaria'."""
    bar_pattern = re.compile(
        r"animate=\{\{\s*width:\s*`\$\{restockPercent\}%`\s*\}\}",
    )
    assert not bar_pattern.search(src), (
        "[P3-RESTOCK-NO-BAR] el `<motion.div animate={{ width: `${restockPercent}%` }} />` "
        "de la barra debe estar removido del JSX del modal. Si reapareció, "
        "rompe la decisión de producto del user."
    )


def test_progress_percent_indicator_removed(src: str) -> None:
    """[P3-RESTOCK-NO-BAR] el indicador `{restockPercent}%` JSX debe
    estar removido. También el label condicional 'EN PROCESO/¡Listo!'."""
    assert "{restockPercent}%" not in src, (
        "[P3-RESTOCK-NO-BAR] el indicador `{restockPercent}%` debe estar "
        "removido (era parte del bloque de la barra)."
    )
    assert "restockPercent >= 100 ? '¡Listo!' : 'En proceso'" not in src, (
        "[P3-RESTOCK-NO-BAR] el label ternario 'En proceso/¡Listo!' debe "
        "estar removido."
    )


def test_three_steps_removed(src: str) -> None:
    """[P3-RESTOCK-NO-BAR] los 3 pasos animados (Verificando ingredientes /
    Actualizando inventario / Sincronizando Nevera) que vivían dentro del
    modal de progreso deben estar REMOVIDOS. Eran acoplados a `restockPercent`
    via thresholds 20/55/90 — sin barra ya no tienen sentido."""
    for step_label in (
        "Verificando ingredientes",
        "Actualizando inventario",
        "Sincronizando Nevera",
    ):
        assert step_label not in src, (
            f"[P3-RESTOCK-NO-BAR] el paso `'{step_label}'` debe estar "
            f"removido del JSX del modal. Si reapareció, rompe la decisión "
            f"de producto."
        )
    # Tampoco el array `.map((step) => ...)` con `threshold` debe existir.
    assert "threshold: 20" not in src
    assert "threshold: 55" not in src
    assert "threshold: 90" not in src


def test_restock_percent_state_removed(src: str) -> None:
    """[P3-RESTOCK-NO-BAR] state `restockPercent` (rAF contador) debe
    estar removido — ya no se usa sin la barra."""
    assert "const [restockPercent, setRestockPercent] = useState(" not in src, (
        "[P3-RESTOCK-NO-BAR] state `restockPercent` debe estar removido "
        "(era el contador del rAF para la barra)."
    )
    assert "setRestockPercent(" not in src, (
        "[P3-RESTOCK-NO-BAR] no debe quedar ningún `setRestockPercent(...)` "
        "callsite. Si quedó alguno, el código tiene dead code colgando."
    )


def test_restock_finish_fast_state_removed(src: str) -> None:
    """[P3-RESTOCK-NO-BAR] state `restockFinishFast` (trigger fast-finish
    de la barra) debe estar removido."""
    assert "const [restockFinishFast, setRestockFinishFast] = useState(" not in src
    assert "setRestockFinishFast(" not in src, (
        "[P3-RESTOCK-NO-BAR] no debe quedar callsite `setRestockFinishFast(true/false)`."
    )


def test_bar_constants_removed(src: str) -> None:
    """[P3-RESTOCK-NO-BAR] constantes `RESTOCK_BAR_BASE_MS` y
    `RESTOCK_BAR_FAST_FINISH_MS` deben estar removidas — ya no se usan."""
    assert "RESTOCK_BAR_BASE_MS" not in src, (
        "[P3-RESTOCK-NO-BAR] constante `RESTOCK_BAR_BASE_MS` debe estar "
        "removida (era la duración de la barra)."
    )
    assert "RESTOCK_BAR_FAST_FINISH_MS" not in src, (
        "[P3-RESTOCK-NO-BAR] constante `RESTOCK_BAR_FAST_FINISH_MS` debe "
        "estar removida (era la duración del fast-finish)."
    )


def test_raf_useeffect_removed(src: str) -> None:
    """[P3-RESTOCK-NO-BAR] el useEffect que manejaba el rAF de la barra
    debe estar removido. Patrón canónico que verificaba: rAF tick + ease-out
    cubic + setRestockPercent."""
    # `requestAnimationFrame(tick)` dentro de un useEffect que dependa de
    # `isRestocking` era el rAF driver. Si aparece, el rAF revivió.
    raf_pattern = re.compile(
        r"requestAnimationFrame\(tick\)",
    )
    assert not raf_pattern.search(src), (
        "[P3-RESTOCK-NO-BAR] `requestAnimationFrame(tick)` debe estar "
        "removido. Era el rAF driver de la barra."
    )


def test_watcher_useeffect_removed(src: str) -> None:
    """[P3-RESTOCK-NO-BAR] el watcher useEffect que cerraba el modal al
    100% debe estar removido — ahora `setShowRestockModal(false)` se llama
    directamente en el handler success."""
    watcher_pattern = re.compile(
        r"if\s*\(\s*restockFinishFast\s*&&\s*restockPercent\s*>=\s*100",
    )
    assert not watcher_pattern.search(src), (
        "[P3-RESTOCK-NO-BAR] watcher `if (restockFinishFast && restockPercent "
        ">= 100 ...)` debe estar removido. Si reapareció, depende de state "
        "ya no existente."
    )


# ===========================================================================
# Elementos PRESERVADOS — heredados de P3-RESTOCK-FLOW-SPEED
# ===========================================================================

def test_cache_singleton_populate_preserved(src: str) -> None:
    """[P3-RESTOCK-NO-BAR] el populate del cache singleton post-refetch
    (P3-RESTOCK-FLOW-SPEED) debe seguir presente — es la optimización
    real que hace Pantry montar instantáneo."""
    assert "from '../utils/pantryCache'" in src
    assert "setCachedInventory(freshInv)" in src, (
        "[P3-RESTOCK-NO-BAR] `setCachedInventory(freshInv)` dentro del "
        "`.then()` del refetch debe seguir presente. Es lo que permite a "
        "Pantry montar sin skeleton."
    )
    assert "invalidateInventoryCache()" in src, (
        "[P3-RESTOCK-NO-BAR] `invalidateInventoryCache()` pre-refetch debe "
        "seguir presente para defensive del TTL 30s del cache."
    )


def test_refetch_parallel_preserved(src: str) -> None:
    """[P3-RESTOCK-NO-BAR] el refetch sigue siendo `.then()` no-awaited
    (P3-RESTOCK-FLOW-SPEED). NO debe regresar el `await supabase.from(...)`."""
    legacy_await = re.compile(
        r"const\s*\{\s*data:\s*freshInv\s*\}\s*=\s*await\s+supabase",
        re.MULTILINE,
    )
    assert not legacy_await.search(src), (
        "[P3-RESTOCK-NO-BAR] el refetch debe seguir siendo `.then()` no-awaited "
        "para que el navigate no se bloquee."
    )
    assert ".then(({ data: freshInv })" in src


def test_no_setTimeout_navigate_artifact_preserved(src: str) -> None:
    """[P3-RESTOCK-NO-BAR] `setTimeout(navigate, 100)` debe seguir
    removido (P3-RESTOCK-FLOW-SPEED). React 18+ auto-batches."""
    legacy_pattern = re.compile(
        r"setTimeout\(\s*\(\s*\)\s*=>\s*\{\s*navigate\(\s*['\"]\/dashboard\/pantry['\"]"
    )
    assert not legacy_pattern.search(src), (
        "[P3-RESTOCK-NO-BAR] el `setTimeout(() => { navigate('/dashboard/pantry') }, 100)` "
        "legacy debe seguir REMOVIDO."
    )


def test_modal_close_called_directly_before_navigate(src: str) -> None:
    """[P3-RESTOCK-NO-BAR] `setShowRestockModal(false)` debe llamarse
    DIRECTAMENTE en el handler success (post-response) — antes del
    `navigate('/dashboard/pantry')`. Pre-fix dependía del watcher useEffect
    acoplado a `restockPercent === 100`; ahora es síncrono en el flow.

    Validación: el `setShowRestockModal(false)` aparece ANTES del
    `navigate('/dashboard/pantry')` (orden textual en el handler success).
    """
    # Localizar el `navigate('/dashboard/pantry')` y el `setShowRestockModal(false)`.
    nav_pos = src.find("navigate('/dashboard/pantry')")
    assert nav_pos >= 0, "navigate('/dashboard/pantry') no encontrado"
    # Buscar el `setShowRestockModal(false)` MÁS CERCANO antes del navigate.
    # Si existe entre el inicio del handler y el navigate, OK.
    pre_nav_window = src[max(0, nav_pos - 2000):nav_pos]
    assert "setShowRestockModal(false)" in pre_nav_window, (
        "[P3-RESTOCK-NO-BAR] `setShowRestockModal(false)` debe aparecer "
        "dentro de los ~2000 chars antes del `navigate('/dashboard/pantry')` "
        "en el handler success. Pre-fix esto lo manejaba un watcher useEffect "
        "(ya removido); ahora es directo en el flow."
    )


# ===========================================================================
# Tooltip-anchor preservado
# ===========================================================================

def test_tooltip_anchor_present(src: str) -> None:
    """[P3-RESTOCK-NO-BAR] el marker textual aparece al menos 3 veces
    (constante removida comment + comment JSX + handler comment)."""
    count = src.count("P3-RESTOCK-NO-BAR")
    assert count >= 3, (
        f"[P3-RESTOCK-NO-BAR] esperaba ≥3 menciones del marker en "
        f"Dashboard.jsx (constantes removidas + JSX comment del modal + "
        f"handler comment). Encontradas: {count}."
    )
