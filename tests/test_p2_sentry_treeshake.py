"""[P2-SENTRY-TREESHAKE · 2026-05-23] Los 2 únicos imports de `@sentry/react`
en el frontend deben ser NAMED, no star (`import * as Sentry from ...`).

Motivación:
    Audit production-readiness 2026-05-23 detectó:
      - `frontend/src/main.jsx:3` → `import * as Sentry from "@sentry/react"`
      - `frontend/src/pages/AgentPage.jsx:30` → mismo patrón

    `import * as Sentry` bloquea tree-shaking de esbuild: el bundle
    final conserva TODOS los exports del SDK (~12 símbolos: profiling,
    feedback, captureFeedback, withScope, getCurrentScope, etc.)
    aunque solo usemos 3-5 de ellos. En un app con `console.log`/`warn`/
    `debug`/`info` dropeados por esbuild `pure:[...]` (P3-FRONTEND-1)
    y self-hosted fonts (P3-SELF-HOST-FONTS), el bundle Sentry queda
    como el wedge más grande remanente sin justificación.

    Símbolos efectivamente usados post-refactor:
      - main.jsx: init, browserTracingIntegration, replayIntegration
      - AgentPage.jsx: captureException, addBreadcrumb

    Total: 5 símbolos de 17+ exports → tree-shake elimina ~70% del SDK.

Scope del test:
    Blanket scan de `frontend/src/**/*.{js,jsx,ts,tsx}` (excluyendo
    tests) buscando `import * as <X> from "@sentry/<paquete>"`. Si
    aparece, falla con archivo+línea + la lista canónica de símbolos
    a usar. Es el primer test que enforza este patrón — si en el futuro
    se añaden imports `import * as Sentry` para otros paquetes
    (`@sentry/tracing`, `@sentry/browser`), el test los catchea.

Tooltip-anchor: P2-SENTRY-TREESHAKE | regression guard 2026-05-23
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND_SRC = _REPO_ROOT / "frontend" / "src"

# Captura `import * as <ANY_NAME> from "@sentry/<anything>"` (o single
# quotes). El binding name puede ser cualquier identificador — no
# asumimos `Sentry`.
_SENTRY_STAR_IMPORT_PATTERN = re.compile(
    r"""import\s*\*\s*as\s+\w+\s+from\s+['"]@sentry/[\w-]+['"]""",
)


def _iter_frontend_files():
    """Mismo iterator que test_p1_new_a: incluye .js/.jsx/.ts/.tsx
    bajo frontend/src, excluye carpetas __tests__ y *.test.*."""
    for f in _FRONTEND_SRC.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix not in {".js", ".jsx", ".ts", ".tsx"}:
            continue
        parts = {p.lower() for p in f.parts}
        if "__tests__" in parts:
            continue
        name_low = f.name.lower()
        if (
            name_low.endswith(".test.js")
            or name_low.endswith(".test.jsx")
            or name_low.endswith(".test.ts")
            or name_low.endswith(".test.tsx")
            or name_low.endswith(".d.ts")
        ):
            continue
        yield f


def _strip_js_comments(src: str) -> str:
    """Stripper compartido con test_p1_new_a / test_p1_settings_confirm_nativo.
    Elimina /* ... */ y // ...EOL preservando numeración de líneas."""
    no_block = re.sub(r"/\*[\s\S]*?\*/", "", src)
    no_line = re.sub(r"//[^\n]*", "", no_block)
    return no_line


def test_no_sentry_star_imports_in_frontend():
    """Ningún archivo de `frontend/src/**` debe tener
    `import * as <X> from "@sentry/..."` — bloquea tree-shaking."""
    offenders: list[str] = []
    for f in _iter_frontend_files():
        try:
            src = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        no_comments = _strip_js_comments(src)
        for m in _SENTRY_STAR_IMPORT_PATTERN.finditer(no_comments):
            line_no = no_comments.count("\n", 0, m.start()) + 1
            rel_path = f.relative_to(_REPO_ROOT)
            snippet = m.group(0).strip()
            offenders.append(f"  {rel_path}:{line_no} → {snippet}")

    assert not offenders, (
        "P2-SENTRY-TREESHAKE violation: `import * as <X> from \"@sentry/...\"` "
        "bloquea tree-shaking de esbuild. El bundle final conserva todos "
        "los exports del SDK aunque solo usemos 3-5 símbolos.\n\n"
        "Offenders:\n" + "\n".join(offenders) + "\n\n"
        "Fix: convertir a named import. Ejemplos canónicos del codebase:\n"
        "  - main.jsx (bootstrap):\n"
        "      import {\n"
        "          init as sentryInit,\n"
        "          browserTracingIntegration,\n"
        "          replayIntegration,\n"
        "      } from '@sentry/react';\n"
        "  - AgentPage.jsx (error capture + breadcrumb):\n"
        "      import { captureException, addBreadcrumb } from '@sentry/react';\n\n"
        "Si necesitas un símbolo nuevo (e.g. `withScope`, `setUser`), añadirlo "
        "al named list — el linter de esbuild dará warning si lo importas sin usar."
    )
