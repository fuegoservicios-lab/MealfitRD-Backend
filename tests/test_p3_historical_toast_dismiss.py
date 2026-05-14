"""[P3-HISTORICAL-TOAST-DISMISS · 2026-05-14] El toast histórico de
coherencia (`emitHistoricalCoherenceToast`) DEBE respetar el dismiss
persistido en localStorage.

Motivación (audit 2026-05-14):
    Pre-fix, `emitHistoricalCoherenceToast` (renderCoherenceWarnings.js)
    emitía el toast en CADA descarga de PDF si había entries en
    `_shopping_coherence_block_history` ≤48h. Si el usuario lo cerraba
    y descargaba PDF 3 veces seguidas, veía el mismo toast 3 veces —
    fricción UX innecesaria.

Fix:
    1. Antes de emitir, leer `mealfit_coherence_toast_dismissed_at` de
       localStorage. Si dismiss < `windowHours` previas (default 48h),
       omitir el toast.
    2. Al emitir, pasar `onDismiss` callback al toast de sonner que
       escribe `Date.now()` al localStorage cuando el usuario cierra
       manualmente (X o swipe).
    3. Cap defensivo: el dismiss state expira tras `windowHours` (mismo
       cap que el filtro de entries históricas). Si pasa el cap, el
       toast vuelve a aparecer (asume contexto cambió y vale re-notificar).

Drift detection (parser-based contra JS source):
    1. La key `mealfit_coherence_toast_dismissed_at` está definida.
    2. Helper `isHistoricalToastRecentlyDismissed` exportado.
    3. `emitHistoricalCoherenceToast` invoca el chequeo del dismiss
       ANTES del emit.
    4. El toast options incluye `onDismiss` callback que escribe a
       localStorage.

Whitelist:
    No prevista. Si en el futuro se decide eliminar el dismiss
    persistente (UX nueva), borrar el helper + revertir el callback
    + retirar este test.

Tooltip-anchor: P3-HISTORICAL-TOAST-DISMISS-START | gap audit 2026-05-14
"""
from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RENDER_COH = _REPO_ROOT / "frontend" / "src" / "utils" / "renderCoherenceWarnings.js"


def _read_src() -> str:
    return _RENDER_COH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. localStorage key canónico definido
# ---------------------------------------------------------------------------
def test_localStorage_key_defined():
    src = _read_src()
    pattern = re.compile(
        r"_DISMISS_STORAGE_KEY\s*=\s*['\"]mealfit_coherence_toast_dismissed_at['\"]",
    )
    assert pattern.search(src), (
        "P3-HISTORICAL-TOAST-DISMISS regresión: la key canónica "
        "`mealfit_coherence_toast_dismissed_at` no está definida como "
        "`_DISMISS_STORAGE_KEY`. Sin una key compartida entre read y "
        "write paths, el dismiss state se pierde por typo."
    )


# ---------------------------------------------------------------------------
# 2. Helper isHistoricalToastRecentlyDismissed exportado
# ---------------------------------------------------------------------------
def test_helper_exported():
    src = _read_src()
    pattern = re.compile(
        r"export\s+const\s+isHistoricalToastRecentlyDismissed\s*=",
    )
    assert pattern.search(src), (
        "P3-HISTORICAL-TOAST-DISMISS regresión: helper "
        "`isHistoricalToastRecentlyDismissed` no exportado. Sin el "
        "export, los tests vitest no pueden ejercitar la lógica de "
        "dismiss sin mock complejo del módulo entero."
    )


# ---------------------------------------------------------------------------
# 3. emitHistoricalCoherenceToast skipea si dismiss reciente
# ---------------------------------------------------------------------------
def test_emit_checks_dismiss_before_emitting():
    src = _read_src()
    # Buscar el cuerpo de la función y verificar que llama al helper
    # ANTES del emitter call.
    fn_match = re.search(
        r"export\s+const\s+emitHistoricalCoherenceToast\s*=\s*\([^)]*\)\s*=>\s*\{",
        src,
    )
    assert fn_match, (
        "Función `emitHistoricalCoherenceToast` no encontrada — "
        "refactor mayor; actualizar test."
    )
    # Capturar hasta el cierre balanceado del bloque (heurística:
    # ~2500 chars que cubren toda la función).
    body = src[fn_match.end() : fn_match.end() + 3000]

    check_call = re.search(
        r"isHistoricalToastRecentlyDismissed\s*\(",
        body,
    )
    assert check_call, (
        "P3-HISTORICAL-TOAST-DISMISS regresión: "
        "`emitHistoricalCoherenceToast` no invoca "
        "`isHistoricalToastRecentlyDismissed(...)`. Sin esta verificación, "
        "el toast se emite siempre que haya entries ≤48h aunque el "
        "usuario lo haya cerrado segundos antes."
    )

    # El return null tras el check debe estar entre el check y el emitter.
    emitter_call = re.search(r"(toast\.warning|toast\.info|emitter)\s*\(", body)
    assert emitter_call, (
        "Emitter call no encontrado dentro del body de "
        "`emitHistoricalCoherenceToast`."
    )
    assert check_call.start() < emitter_call.start(), (
        f"P3-HISTORICAL-TOAST-DISMISS regresión: el chequeo "
        f"`isHistoricalToastRecentlyDismissed` aparece DESPUÉS del "
        f"emitter call (offsets: check={check_call.start()}, "
        f"emitter={emitter_call.start()}). El skip debe ocurrir ANTES "
        f"de emitir, no después."
    )


# ---------------------------------------------------------------------------
# 4. Toast options incluye onDismiss → write to localStorage
# ---------------------------------------------------------------------------
def test_toast_passes_onDismiss_callback():
    src = _read_src()
    # `onDismiss: _writeDismissAt` debe aparecer en el call al emitter
    # dentro de `emitHistoricalCoherenceToast`.
    pattern = re.compile(
        r"onDismiss\s*:\s*_writeDismissAt",
    )
    assert pattern.search(src), (
        "P3-HISTORICAL-TOAST-DISMISS regresión: el toast options no "
        "incluye `onDismiss: _writeDismissAt`. Sin este callback, "
        "cerrar manualmente el toast NO persiste el dismiss state — "
        "el toast vuelve a aparecer en el siguiente PDF download."
    )

    # _writeDismissAt debe invocar localStorage.setItem con la key
    # canónica.
    write_pat = re.compile(
        r"localStorage\.setItem\s*\(\s*_DISMISS_STORAGE_KEY\s*,",
    )
    assert write_pat.search(src), (
        "P3-HISTORICAL-TOAST-DISMISS regresión: `_writeDismissAt` no "
        "invoca `localStorage.setItem(_DISMISS_STORAGE_KEY, ...)`. "
        "Sin esta escritura, el dismiss state nunca persiste."
    )


# ---------------------------------------------------------------------------
# 5. localStorage.getItem usa la key canónica
# ---------------------------------------------------------------------------
def test_read_uses_canonical_key():
    src = _read_src()
    # `localStorage.getItem(_DISMISS_STORAGE_KEY)` debe aparecer.
    pattern = re.compile(
        r"localStorage\.getItem\s*\(\s*_DISMISS_STORAGE_KEY\s*\)",
    )
    assert pattern.search(src), (
        "P3-HISTORICAL-TOAST-DISMISS regresión: `_readDismissAt` no "
        "invoca `localStorage.getItem(_DISMISS_STORAGE_KEY)`. Si lee "
        "con otra key (string literal), drift cero garantizado pero "
        "consumidor externo (test vitest) que escriba con la key "
        "canónica no afecta el state."
    )


# ---------------------------------------------------------------------------
# 6. Best-effort: try/catch alrededor de localStorage access
# ---------------------------------------------------------------------------
def test_localstorage_access_in_try_catch():
    src = _read_src()
    # Tanto `_readDismissAt` como `_writeDismissAt` deben envolver
    # localStorage en try/catch para resilient a iOS Private Mode /
    # storage quota exceeded.
    read_pat = re.compile(
        r"const\s+_readDismissAt\s*=\s*\(\s*\)\s*=>\s*\{[\s\S]*?try\s*\{",
    )
    write_pat = re.compile(
        r"const\s+_writeDismissAt\s*=\s*\(\s*\)\s*=>\s*\{[\s\S]*?try\s*\{",
    )
    assert read_pat.search(src), (
        "P3-HISTORICAL-TOAST-DISMISS regresión: `_readDismissAt` no "
        "envuelve el acceso a `localStorage` en try/catch. iOS Safari "
        "Private Mode lanza exception al leer localStorage; sin "
        "try/catch el toast crashea el caller."
    )
    assert write_pat.search(src), (
        "P3-HISTORICAL-TOAST-DISMISS regresión: `_writeDismissAt` no "
        "envuelve el acceso a `localStorage` en try/catch. Storage "
        "quota exceeded lanza QuotaExceededError; sin try/catch el "
        "onDismiss callback rompe el cleanup del toast."
    )


# ---------------------------------------------------------------------------
# 7. windowHours default 48 preservado
# ---------------------------------------------------------------------------
def test_window_hours_default_48():
    src = _read_src()
    # El helper debe defaultar a 48 cuando windowHours es undefined/inválido.
    pattern = re.compile(
        r"windowHours\s*=\s*48",
    )
    assert pattern.search(src), (
        "P3-HISTORICAL-TOAST-DISMISS regresión: el default de "
        "`windowHours` en `isHistoricalToastRecentlyDismissed` no es 48. "
        "El default debe matchear el filtro temporal de "
        "`buildHistoricalCoherenceToast` (también 48h) para que el "
        "dismiss state expire en el mismo momento que el entry deja "
        "de ser elegible."
    )
