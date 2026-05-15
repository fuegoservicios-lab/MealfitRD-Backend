"""[P2-AUDIT-3 · 2026-05-15] Test parser-based: los 13 callsites raw
`localStorage.setItem(...)` en `AssessmentContext.jsx` migraron al helper
SSOT `safeLocalStorageSet(key, value)` con try/catch defensivo.

Por qué este test:
    `localStorage.setItem(key, value)` puede lanzar `QuotaExceededError`
    (cuota agotada — common en iOS Private Mode donde cuota efectiva = 0) y
    `SecurityError` (storage deshabilitado por settings). 13 callsites raw
    en AssessmentContext.jsx quedaron expuestos a esos throws no atrapados,
    rompiendo el flujo del context (plan guardado en React pero NO en
    localStorage → reload pierde el plan recién generado).

    Convención del repo (P3-HISTORICAL-TOAST-DISMISS · 2026-05-14): TODA
    escritura a localStorage va via helper try/catch. AssessmentContext
    quedó como callsite legacy hasta este P-fix.

Fix esperado:
    - Helper `safeLocalStorageSet(key, value, opts?)` exportado de
      `frontend/src/utils/safeLocalStorage.js`. Atrapa throws + returns
      `true`/`false`.
    - `AssessmentContext.jsx` importa el helper y reemplaza los 13
      callsites raw.

Drift detection:
    - `utils/safeLocalStorage.js` existe y exporta `safeLocalStorageSet`.
    - `AssessmentContext.jsx` importa el helper.
    - Cero callsites `localStorage.setItem(...)` activos en
      AssessmentContext (solo en comentarios narrativos).

Cross-link convention (P2-HIST-AUDIT-14): slug `p2_audit_3`.

Tooltip-anchor: P2-AUDIT-3-START | gap audit 2026-05-15
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ASSESSMENT_CTX_JSX = _REPO_ROOT / "frontend" / "src" / "context" / "AssessmentContext.jsx"
_SAFE_LS_JS = _REPO_ROOT / "frontend" / "src" / "utils" / "safeLocalStorage.js"


@pytest.fixture(scope="module")
def assessment_ctx_src() -> str:
    return _ASSESSMENT_CTX_JSX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def safe_ls_src() -> str:
    return _SAFE_LS_JS.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. utils/safeLocalStorage.js existe y exporta los helpers
# ---------------------------------------------------------------------------
def test_safe_localstorage_module_exists():
    assert _SAFE_LS_JS.exists(), (
        "P2-AUDIT-3 regresión: `frontend/src/utils/safeLocalStorage.js` no "
        "existe. Crear el helper SSOT con `export function safeLocalStorageSet`."
    )


def test_safe_localstorage_exports_set(safe_ls_src: str):
    assert re.search(
        r"export\s+function\s+safeLocalStorageSet\s*\(",
        safe_ls_src,
    ), (
        "P2-AUDIT-3 regresión: `export function safeLocalStorageSet(...)` no "
        "encontrado en safeLocalStorage.js."
    )


def test_safe_localstorage_set_has_try_catch(safe_ls_src: str):
    """El helper debe envolver `localStorage.setItem` en try/catch para
    atrapar QuotaExceededError + SecurityError."""
    # Extraer el body de safeLocalStorageSet hasta el siguiente `export`.
    fn_match = re.search(
        r"export\s+function\s+safeLocalStorageSet\s*\([^)]*\)\s*\{",
        safe_ls_src,
    )
    assert fn_match
    body_start = fn_match.end()
    next_export = re.search(r"\nexport\s", safe_ls_src[body_start:])
    body_end = body_start + (next_export.start() if next_export else 800)
    body = safe_ls_src[body_start:body_end]
    assert re.search(r"\btry\s*\{[^}]*setItem", body, re.DOTALL), (
        "P2-AUDIT-3 regresión: `safeLocalStorageSet` no envuelve "
        "`localStorage.setItem` en `try`. Sin try, QuotaExceededError "
        "propaga al caller — el helper no aporta valor."
    )
    assert re.search(r"\bcatch\s*\(", body), (
        "P2-AUDIT-3 regresión: `safeLocalStorageSet` no tiene `catch` block."
    )


# ---------------------------------------------------------------------------
# 2. AssessmentContext.jsx importa el helper
# ---------------------------------------------------------------------------
def test_assessment_ctx_imports_helper(assessment_ctx_src: str):
    assert re.search(
        r"import\s*\{[^}]*\bsafeLocalStorageSet\b[^}]*\}\s*from\s*['\"][^'\"]*safeLocalStorage['\"]",
        assessment_ctx_src,
    ), (
        "P2-AUDIT-3 regresión: `import { safeLocalStorageSet }` no encontrado "
        "en AssessmentContext.jsx. Añadir el import."
    )


# ---------------------------------------------------------------------------
# 3. Cero callsites raw `localStorage.setItem(...)` activos
# ---------------------------------------------------------------------------
def test_no_raw_localstorage_setitem_in_assessment_ctx(assessment_ctx_src: str):
    """Strip comentarios + strings de import. Cualquier callsite raw
    `localStorage.setItem(...)` activo es regresión."""
    # Strip line comments + block comments.
    no_line_comments = re.sub(r"//[^\n]*", "", assessment_ctx_src)
    no_comments = re.sub(r"/\*.*?\*/", "", no_line_comments, flags=re.DOTALL)
    # Strip JSX comments {/* ... */}.
    no_comments = re.sub(r"\{\s*/\*.*?\*/\s*\}", "", no_comments, flags=re.DOTALL)
    pattern = re.compile(r"\blocalStorage\.setItem\s*\(")
    matches = pattern.findall(no_comments)
    assert not matches, (
        f"P2-AUDIT-3 regresión: {len(matches)} callsites raw "
        f"`localStorage.setItem(...)` activos en AssessmentContext.jsx. "
        f"Migrar a `safeLocalStorageSet(key, value)` que atrapa "
        f"QuotaExceededError + SecurityError."
    )


# ---------------------------------------------------------------------------
# 4. El helper se usa al menos N veces (proxy de migración real)
# ---------------------------------------------------------------------------
def test_helper_invoked_at_least_n_times(assessment_ctx_src: str):
    """Esperamos al menos 10 invocaciones de `safeLocalStorageSet(...)` —
    el audit identificó 13 callsites raw para migrar. Algunos pueden
    consolidarse al pasar, pero <10 sugiere que el helper se importó pero
    NO se usó (regresión silenciosa)."""
    pattern = re.compile(r"\bsafeLocalStorageSet\s*\(")
    matches = pattern.findall(assessment_ctx_src)
    # Strip import line (también matchea el regex porque importa el nombre).
    import_re = re.compile(
        r"import\s*\{[^}]*safeLocalStorageSet[^}]*\}\s*from",
    )
    import_count = len(import_re.findall(assessment_ctx_src))
    invocations = len(matches) - import_count
    assert invocations >= 10, (
        f"P2-AUDIT-3 regresión: solo {invocations} invocaciones de "
        f"`safeLocalStorageSet(...)` en AssessmentContext.jsx (esperado ≥10). "
        f"El audit identificó 13 callsites raw para migrar — un conteo bajo "
        f"sugiere que el helper se importó pero los callsites raw no se "
        f"reemplazaron, o que un cleanup posterior los consolidó "
        f"prematuramente."
    )


# ---------------------------------------------------------------------------
# 5. Anchor textual P2-AUDIT-3 presente
# ---------------------------------------------------------------------------
def test_anchor_present(assessment_ctx_src: str):
    assert "P2-AUDIT-3" in assessment_ctx_src, (
        "P2-AUDIT-3 regresión: anchor textual `P2-AUDIT-3` perdido en "
        "AssessmentContext.jsx. Restaurar para grep cross-incidente."
    )
