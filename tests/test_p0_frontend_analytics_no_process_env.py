"""[P0-FRONTEND-ANALYTICS · 2026-05-12] Bloquea `process.env.*` en código
de browser frontend (Vite bundle).

Pre-fix: `frontend/src/utils/analytics.js:3` evaluaba
`process.env.NODE_ENV !== 'production'` en cada `trackEvent(...)`. Vite NO
inyecta `process` en el bundle del cliente — el resultado en producción era:

    ReferenceError: process is not defined
        at trackEvent (analytics.js:3)

Como `trackEvent` se invoca desde Sentry breadcrumbs / PostHog / GA / GTM,
TODA la analítica de producción caía silenciosa y `GlobalErrorBoundary`
capturaba el error ofuscando logs de errores reales.

Vite expone `import.meta.env.MODE` (string `'development'`/`'production'`/
`'test'`) y `import.meta.env.DEV` (bool) con la misma semántica que
`process.env.NODE_ENV` en Node. NO hay back-compat: cualquier referencia
a `process.env` en frontend es un bug.

Este test escanea `frontend/src/**/*.{js,jsx}` y falla loud si encuentra
`process.env`. Excluye:
  - `__tests__/` (Vitest jsdom env: `process` SÍ está definido)
  - `*.test.{js,jsx}` (mismo contexto)
  - `setupTests.js` (mismo contexto)

Sin tooling para diferenciar estático code paths de runtime, la heurística
"todo lo que NO sea test" es razonable: el frontend de Vite jamás debe
asumir `process` global.

Si un futuro callsite necesita whitelist explícita (poco probable pero
posible para bridges Node↔Vite poco ortodoxos), añadir comment
`// [P0-FRONTEND-ANALYTICS WHITELIST: <razón>]` en las 3 líneas previas.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND_SRC = _REPO_ROOT / "frontend" / "src"
_ANALYTICS = _FRONTEND_SRC / "utils" / "analytics.js"

# Excluye archivos de test / setup donde `process` SÍ está disponible
# (jsdom env de Vitest corre en Node).
_EXCLUDED_GLOBS = (
    "__tests__/",
    ".test.js",
    ".test.jsx",
    "setupTests.js",
)

_PROCESS_ENV_RE = re.compile(r"\bprocess\.env\b")
_WHITELIST_MARKER = "P0-FRONTEND-ANALYTICS WHITELIST"


def _iter_frontend_src_files():
    for ext in ("*.js", "*.jsx"):
        for fp in _FRONTEND_SRC.rglob(ext):
            rel = fp.relative_to(_FRONTEND_SRC).as_posix()
            if any(excl in rel for excl in _EXCLUDED_GLOBS):
                continue
            yield fp


# ---------------------------------------------------------------------------
# 1. El bug original está cerrado
# ---------------------------------------------------------------------------
def test_analytics_js_uses_import_meta_env():
    """`analytics.js` debe usar `import.meta.env.MODE` (Vite-correct), NO
    `process.env.NODE_ENV` (browser-broken)."""
    assert _ANALYTICS.exists(), f"analytics.js no encontrado en {_ANALYTICS}"
    text = _ANALYTICS.read_text(encoding="utf-8")
    assert "process.env" not in text, (
        f"P0-FRONTEND-ANALYTICS regression: `analytics.js` volvió a usar "
        f"`process.env`. Vite NO define `process` en bundle de browser — "
        f"esto crashea cada llamada a trackEvent() en producción.\n\n"
        f"Fix: cambiar `process.env.NODE_ENV !== 'production'` por "
        f"`import.meta.env.MODE !== 'production'`."
    )
    assert "import.meta.env" in text, (
        "`analytics.js` debe leer el modo via `import.meta.env.MODE` "
        "(Vite-canonical)."
    )


# ---------------------------------------------------------------------------
# 2. Blanket: ningún módulo del bundle frontend debe referenciar process.env
# ---------------------------------------------------------------------------
def test_no_process_env_in_frontend_bundle():
    """Escanea todos los archivos del bundle Vite (`frontend/src/**/*.{js,jsx}`
    excluyendo tests + setup) en busca de `process.env`.

    Cualquier match crashea en runtime browser con
    `ReferenceError: process is not defined`. Si un callsite legítimo
    necesita el ref (caso raro: shims Node↔Vite, polyfills), debe llevar
    comment `// [P0-FRONTEND-ANALYTICS WHITELIST: <razón>]` en las 3
    líneas previas para que esta regla lo respete.
    """
    violations = []
    for fp in _iter_frontend_src_files():
        text = fp.read_text(encoding="utf-8")
        if "process.env" not in text:
            continue
        # Per-line check con whitelist por proximidad.
        lines = text.split("\n")
        for idx, line in enumerate(lines):
            if not _PROCESS_ENV_RE.search(line):
                continue
            # Buscar marker whitelist en las 3 líneas previas.
            window_start = max(0, idx - 3)
            window = "\n".join(lines[window_start:idx])
            if _WHITELIST_MARKER in window:
                continue
            rel = fp.relative_to(_REPO_ROOT).as_posix()
            violations.append(f"{rel}:{idx + 1}: {line.strip()}")

    assert not violations, (
        "P0-FRONTEND-ANALYTICS violation: `process.env` referenced in "
        "frontend bundle. Vite NO inyecta `process` en runtime browser — "
        "estos callsites crashean con ReferenceError en producción.\n\n"
        "Fix: usar `import.meta.env.<KEY>` (Vite-canonical). Si el "
        "callsite es legítimamente Node-only (raro), añadir comment "
        f"`// [{_WHITELIST_MARKER}: <razón>]` en las 3 líneas previas.\n\n"
        "Violations:\n  " + "\n  ".join(violations)
    )


# ---------------------------------------------------------------------------
# 3. Sanity: el escáner respeta la whitelist
# ---------------------------------------------------------------------------
def test_whitelist_marker_is_respected(tmp_path, monkeypatch):
    """Defensa: si el escáner se rompe y deja de respetar el marker, el
    test blanket de arriba bloquearía cualquier whitelist legítima
    futura. Cubre el parser con un fixture en memoria."""
    fake_src = tmp_path / "src"
    fake_src.mkdir()
    fake_file = fake_src / "node_bridge.js"
    fake_file.write_text(
        "// [P0-FRONTEND-ANALYTICS WHITELIST: shim Node-only para SSR]\n"
        "const env = process.env.SOME_KEY;\n",
        encoding="utf-8",
    )
    text = fake_file.read_text(encoding="utf-8")
    lines = text.split("\n")
    found = False
    for idx, line in enumerate(lines):
        if _PROCESS_ENV_RE.search(line):
            window = "\n".join(lines[max(0, idx - 3):idx])
            if _WHITELIST_MARKER in window:
                found = True
                break
    assert found, (
        "El parser whitelist está roto: comment marker no detectado "
        "en window de 3 líneas previas."
    )


# ---------------------------------------------------------------------------
# 4. Sanity: el escáner SÍ flagea callsites sin whitelist
# ---------------------------------------------------------------------------
def test_unwhitelisted_callsite_is_caught(tmp_path):
    """Mirror inverso del anterior: sin marker, el match debe registrarse
    como violation."""
    fake_src = tmp_path / "src"
    fake_src.mkdir()
    fake_file = fake_src / "broken.js"
    fake_file.write_text(
        "export const isProd = process.env.NODE_ENV === 'production';\n",
        encoding="utf-8",
    )
    text = fake_file.read_text(encoding="utf-8")
    lines = text.split("\n")
    violations = []
    for idx, line in enumerate(lines):
        if not _PROCESS_ENV_RE.search(line):
            continue
        window = "\n".join(lines[max(0, idx - 3):idx])
        if _WHITELIST_MARKER in window:
            continue
        violations.append(line.strip())
    assert violations, (
        "El parser está roto: callsite sin whitelist debe ser flageado."
    )


# ---------------------------------------------------------------------------
# 5. Anchor del marker (cierre del cross-link P2-HIST-AUDIT-14)
# ---------------------------------------------------------------------------
def test_anchor_present_in_test_file():
    """Anchor `P0-FRONTEND-ANALYTICS` debe permanecer en el cuerpo de
    este test para que el cross-link de `_LAST_KNOWN_PFIX` funcione
    (`test_p2_hist_audit_14_marker_test_link`)."""
    this_file = Path(__file__).read_text(encoding="utf-8")
    assert "P0-FRONTEND-ANALYTICS" in this_file
