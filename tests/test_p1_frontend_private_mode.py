"""[P1-FRONTEND-PRIVATE-MODE · 2026-05-23] Regression guard de los 5
sitios del frontend donde `localStorage.getItem(...)` raw rompía el
render del provider en iOS Safari Private Mode.

Motivación:
    En iOS Safari Private Mode `localStorage.getItem(key)` lanza
    `SecurityError`. Cuando el call está FUERA del callback lazy del
    `useState` (o sin try/catch externo), el throw corta la cadena de
    side-effects y deja el provider en estado inconsistente — síntoma
    visible: pantalla blanca al abrir la app, o features que fallan
    silenciosamente (regenerateSingleMeal abortado sin toast, user-switch
    detection saltada con bleed de datos cifrados del owner anterior).

    Audit production-readiness 2026-05-23 (P1-FRONTEND-PRIVATE-MODE)
    identificó 5 callsites concretos. Cada uno fue migrado a
    `safeLocalStorageGet(key, fallback)` del helper SSOT
    `frontend/src/utils/safeLocalStorage.js` (P2-AUDIT-3-SAFE-LOCALSTORAGE).

Scope intencionalmente acotado:
    El frontend tiene ~50+ callsites adicionales de `localStorage.getItem(`
    raw — la mayoría son `userId = session?.user?.id || localStorage.getItem('mealfit_user_id')`
    dentro de handlers async con try/catch externo, donde el throw no rompe
    el render. Migrarlos masivamente está fuera de scope para no inflar el
    diff. Política "boy scout" del repo aplica: cuando edites un archivo
    con esos callsites, considera migrar.

    Este test NO hace blanket scan — enforza específicamente que los 5
    sitios reportados como bug real NO regresionen al patrón raw.

Tooltip-anchor: P1-FRONTEND-PRIVATE-MODE | regression guard 2026-05-23
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND_SRC = _REPO_ROOT / "frontend" / "src"
_ASSESSMENT_CTX = _FRONTEND_SRC / "context" / "AssessmentContext.jsx"
_PANTRY = _FRONTEND_SRC / "pages" / "Pantry.jsx"


# Los 5 sitios fixed. Cada entry es (file_path, identifying_substring,
# rationale_humano). El test verifica que el substring identificador
# aparece bajo `safeLocalStorageGet(` o `safeLocalStorageRemove(` (NO
# bajo `localStorage.getItem(` raw).
_FIXED_SITES = [
    (
        _ASSESSMENT_CTX,
        "'mealfit_dislikes'",
        "savedDislikes: raw getItem fuera del callback del useState "
        "rompía render del provider en iOS Private Mode (pantalla blanca).",
    ),
    (
        _ASSESSMENT_CTX,
        "'mealfit_last_form_owner'",
        "lastOwner: user-switch detection bypassed silently — datos "
        "cifrados del owner anterior persistían en mealfit_form_secure.",
    ),
    (
        _ASSESSMENT_CTX,
        "'mealfit_user_id', 'guest_session'",
        "sessionId del swap-meal: fallback explícito reemplaza el "
        "`|| 'guest_session'` que dependía de que getItem no throw.",
    ),
    (
        _PANTRY,
        "'mealfit_pantry_dirty_at'",
        "_consumePantryDirtyFromChat: 1 getItem + 2 removeItem raw — "
        "el throw del primer removeItem dejaba la key envenenada y "
        "los siguientes mounts re-invalidaban cache perpetuamente.",
    ),
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("file_path,substr,rationale", _FIXED_SITES)
def test_fixed_site_uses_safe_helper(file_path: Path, substr: str, rationale: str):
    """Cada uno de los 5 sitios fixed DEBE aparecer bajo el helper
    `safeLocalStorageGet(...)` o `safeLocalStorageRemove(...)`. Si
    alguien revierte el fix al patrón raw `localStorage.getItem(...)`
    o `window.localStorage.removeItem(...)`, este test falla."""
    src = _read(file_path)
    # Encuentra todas las ocurrencias del substring identificador.
    matches = [m for m in re.finditer(re.escape(substr), src)]
    assert matches, (
        f"P1-FRONTEND-PRIVATE-MODE: substring identificador {substr!r} "
        f"no encontrado en {file_path.relative_to(_REPO_ROOT)}. "
        f"¿Se renombró la key sin actualizar este test? Rationale: {rationale}"
    )
    # Para cada match, la línea (o cluster cercano ±2 líneas) DEBE
    # contener `safeLocalStorageGet(` o `safeLocalStorageRemove(`. NO
    # debe contener `localStorage.getItem(` ni `window.localStorage.removeItem(`.
    src_lines = src.splitlines()
    violations: list[str] = []
    for m in matches:
        line_no = src.count("\n", 0, m.start()) + 1
        start = max(0, line_no - 3)
        end = min(len(src_lines), line_no + 2)
        window = "\n".join(src_lines[start:end])
        # OK si la ventana usa el safe helper.
        uses_safe_get = "safeLocalStorageGet(" in window
        uses_safe_remove = "safeLocalStorageRemove(" in window
        # Violation si la ventana tiene patrón raw cerca del substring.
        has_raw_get = re.search(
            r"(?:window\.)?localStorage\.getItem\s*\(",
            window,
        )
        has_raw_remove = re.search(
            r"(?:window\.)?localStorage\.removeItem\s*\(",
            window,
        )
        if (uses_safe_get or uses_safe_remove) and not (has_raw_get or has_raw_remove):
            continue
        # Si el contexto inmediato tiene raw access, es violation.
        if has_raw_get or has_raw_remove:
            violations.append(
                f"  {file_path.relative_to(_REPO_ROOT)}:{line_no} → "
                f"raw localStorage access cerca de {substr!r}. "
                f"Rationale: {rationale}"
            )

    assert not violations, (
        "P1-FRONTEND-PRIVATE-MODE regresión: uno de los 5 sitios fixed "
        "volvió al patrón raw `localStorage.getItem/removeItem(...)`. "
        "Estos sitios DEBEN usar `safeLocalStorageGet`/`safeLocalStorageRemove` "
        "del helper SSOT `frontend/src/utils/safeLocalStorage.js`. En iOS "
        "Safari Private Mode el patrón raw lanza SecurityError y rompe "
        "el render del provider o silencia features críticas.\n\n"
        "Violations:\n" + "\n".join(violations) + "\n\n"
        "Fix: reemplazar con `safeLocalStorageGet(key, fallback)` o "
        "`safeLocalStorageRemove(key)`. Ver "
        "`frontend/src/utils/safeLocalStorage.js` para la API."
    )


def test_safe_localstorage_helper_exists():
    """Sanity: el helper SSOT que los fixes invocan DEBE existir.
    Si alguien lo elimina, los call sites lanzan ReferenceError en
    runtime y este test cierra el gap antes que CI/runtime."""
    helper = _FRONTEND_SRC / "utils" / "safeLocalStorage.js"
    assert helper.exists(), (
        f"P1-FRONTEND-PRIVATE-MODE: helper SSOT {helper} fue eliminado. "
        f"Los 5 fixes dependen de `safeLocalStorageGet`/`safeLocalStorageRemove` — "
        f"sin el helper, ReferenceError en runtime al cargar la app."
    )
    src = helper.read_text(encoding="utf-8")
    assert "export function safeLocalStorageGet" in src, (
        "P1-FRONTEND-PRIVATE-MODE: `safeLocalStorageGet` ya no se exporta "
        "desde el helper SSOT. Restaurar el export o migrar los 5 call "
        "sites al nuevo nombre."
    )
    assert "export function safeLocalStorageRemove" in src, (
        "P1-FRONTEND-PRIVATE-MODE: `safeLocalStorageRemove` ya no se "
        "exporta desde el helper SSOT. Restaurar el export o migrar "
        "Pantry.jsx::_consumePantryDirtyFromChat al nuevo nombre."
    )
