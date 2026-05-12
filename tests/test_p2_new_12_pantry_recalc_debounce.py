"""[P2-NEW-12 · 2026-05-11] Debounce coalescente trailing para
`_recalcShoppingListAfterPantryChange` en Pantry.jsx.

Bug original (audit 2026-05-11):
    Add/delete individual de N items genera N POST a
    `/api/plans/recalculate-shopping-list` paralelos. Backend deduplica
    por `_plan_modified_at` pero el flicker UI + el waste de cuota
    implícita son evitables. `confirmDeleteAll` ya batches en su propia
    operación (un solo DELETE masivo), así que NO necesita debounce.

Fix:
    1. Refs ref-based: `_recalcDebounceTimer`, `_recalcInFlight`,
       `_recalcPendingAfterFlight`. Constante `_RECALC_DEBOUNCE_MS = 500`.
    2. Wrapper `_scheduleRecalcShoppingList()` coalescente trailing:
       - Si call llega mientras hay timer → reset timer (debounce 500ms).
       - Si call llega mientras hay flight → marca pending y reprograma
         al terminar (preserva último estado).
    3. Los 3 callsites individuales (line 473 undo, 485 delete, 567 add)
       cambian al wrapper. `confirmDeleteAll` (line 510) sigue inline
       con `await` directo — operación mayor, toast post-éxito.
    4. Cleanup en useEffect unmount.

Estrategia del test (parser-based sobre Pantry.jsx):
    1. Refs declarados con `useRef`.
    2. Constante `_RECALC_DEBOUNCE_MS = 500`.
    3. Función `_scheduleRecalcShoppingList` definida con setTimeout
       coalescente + trailing logic.
    4. Los 3 callsites individuales invocan `_scheduleRecalcShoppingList()`,
       NO `_recalcShoppingListAfterPantryChange()` directo.
    5. `confirmDeleteAll` SIGUE llamando directo al helper original
       (await + args).
    6. Cleanup useEffect cancela el timer en unmount.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PANTRY_FP = _REPO_ROOT / "frontend" / "src" / "pages" / "Pantry.jsx"


@pytest.fixture(scope="module")
def src() -> str:
    return _PANTRY_FP.read_text(encoding="utf-8")


def test_refs_declared(src: str):
    """Las 3 refs del debounce deben declararse con `useRef`."""
    for ref in (
        "_recalcDebounceTimer",
        "_recalcInFlight",
        "_recalcPendingAfterFlight",
    ):
        pattern = re.compile(
            rf"const\s+{ref}\s*=\s*useRef\(",
        )
        assert pattern.search(src), (
            f"P2-NEW-12 regresión: ref `{ref}` ya no se declara con "
            f"`useRef`. Sin él, el debounce coalescente no puede "
            f"sobrevivir re-renders."
        )


def test_debounce_constant_500ms(src: str):
    """`_RECALC_DEBOUNCE_MS = 500`."""
    assert re.search(
        r"const\s+_RECALC_DEBOUNCE_MS\s*=\s*500",
        src,
    ), (
        "P2-NEW-12 regresión: `_RECALC_DEBOUNCE_MS = 500` ya no existe. "
        "Cambiar el valor requiere actualizar este test."
    )


def test_schedule_wrapper_exists(src: str):
    """`_scheduleRecalcShoppingList` definido como función."""
    assert re.search(
        r"const\s+_scheduleRecalcShoppingList\s*=\s*\(\s*\)\s*=>\s*\{",
        src,
    ), (
        "P2-NEW-12 regresión: el wrapper `_scheduleRecalcShoppingList` "
        "ya no existe. Sin él, los callsites debounced volverían a "
        "hammeartear al backend."
    )


def test_wrapper_uses_setTimeout_with_debounce_constant(src: str):
    """El wrapper usa `setTimeout(..., _RECALC_DEBOUNCE_MS)`."""
    wrapper_start = src.find("const _scheduleRecalcShoppingList")
    wrapper_end = src.find("// Cleanup del timer", wrapper_start)
    body = src[wrapper_start:wrapper_end]
    assert "setTimeout(" in body and "_RECALC_DEBOUNCE_MS" in body, (
        "P2-NEW-12 regresión: el wrapper ya no usa `setTimeout(..., "
        "_RECALC_DEBOUNCE_MS)`. Sin debounce el wrapper se reduce a un "
        "wrapper-de-nada."
    )
    # Trailing pending logic.
    assert "_recalcPendingAfterFlight" in body, (
        "P2-NEW-12 regresión: el wrapper ya no maneja el caso `pending "
        "after flight`. Pierde el último estado si call llega mientras "
        "el recalc anterior corre."
    )


def test_three_individual_callsites_use_wrapper(src: str):
    """Los 3 callsites individuales (undo + delete + add) usan el wrapper."""
    # Total invocaciones del wrapper ≥ 3.
    wrapper_calls = re.findall(r"_scheduleRecalcShoppingList\s*\(\s*\)", src)
    assert len(wrapper_calls) >= 3, (
        f"P2-NEW-12 regresión: solo {len(wrapper_calls)} callsites usan "
        "`_scheduleRecalcShoppingList()` (esperado ≥3: undo, delete "
        "individual, add). Algún callsite revirtió a la llamada directa."
    )


def test_confirm_delete_all_keeps_direct_await(src: str):
    """`confirmDeleteAll` sigue llamando directo al helper original
    con `await` + args. NO debe pasar por el wrapper (operación mayor,
    necesita toast post-éxito determinístico)."""
    func_start = src.find("const confirmDeleteAll")
    assert func_start > 0, "confirmDeleteAll no encontrado"
    func_end = src.find("\n    };", func_start)
    body = src[func_start:func_end]

    # DEBE seguir llamando al helper original con `await` + args.
    assert re.search(
        r"await\s+_recalcShoppingListAfterPantryChange\s*\(\s*\{",
        body,
    ), (
        "P2-NEW-12 regresión: `confirmDeleteAll` ya no llama directo al "
        "helper original con `await + args`. Si pasó al wrapper "
        "(debounced), perdería el toast determinístico post-éxito y los "
        "args `silentSuccess=false, clearRestockedFlag=true`."
    )

    # NO debe usar el wrapper debounced (rompería UX del toast).
    assert "_scheduleRecalcShoppingList" not in body, (
        "P2-NEW-12 regresión: `confirmDeleteAll` ahora usa el wrapper "
        "debounced — eso rompe la semántica de `delete all` (operación "
        "mayor que necesita toast post-éxito sincronizado)."
    )


def test_unmount_cleanup_clears_timer(src: str):
    """Un `useEffect(..., [])` con cleanup que cancela el timer."""
    # Patrón: useEffect que solo retorna cleanup function que clearTimeout
    # el ref del debounce.
    pattern = re.compile(
        r"useEffect\(\s*\(\s*\)\s*=>\s*\{\s*"
        r"return\s*\(\s*\)\s*=>\s*\{[^}]*"
        r"clearTimeout\(\s*_recalcDebounceTimer\.current\s*\)",
        re.DOTALL,
    )
    assert pattern.search(src), (
        "P2-NEW-12 regresión: el cleanup del unmount ya no clearTimeout "
        "el ref del debounce. Sin esto, un unmount mid-debounce puede "
        "disparar el recalc tras el unmount (warning React + posible "
        "state update on unmounted component)."
    )
