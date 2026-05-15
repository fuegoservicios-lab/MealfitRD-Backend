"""[P3-NEW-CONSOLE-ERROR-SENTRY · 2026-05-15] Anchor + regression guard.

`frontend/src/components/GlobalErrorBoundary.jsx` invocaba `console.error(...)`
sin gate en `componentDidCatch`. Vite/esbuild en prod build solo stripea
`log/warn/info/debug` (terserOptions.compress.pure_funcs), pero `error` se
preserva para no perder diagnósticos críticos. Resultado: en prod el usuario
ve el stack trace completo + componentStack en DevTools al abrir consola
— minor info leak + ruido visual.

Sentry React ErrorBoundary capturaría el evento, pero introducir
`@sentry/react` aquí es overkill; Sentry SDK ya está inicializado en
`main.jsx` (P1-SENTRY-PII-SCRUBBING-BACKEND) y captura unhandled errors
via window listener. El log local solo sirve para dev workflow.

Defensas que el test enforza:
  1. Anchor `P3-NEW-CONSOLE-ERROR-SENTRY` presente en GlobalErrorBoundary.
  2. `console.error` está envuelto en gate `import.meta.env.DEV`.
  3. Cero `console.error` no-gateado en el archivo (defensa contra futuros
     adds que reabran el modo de fallo).
  4. Anchor presente en este archivo (cross-link guard P2-HIST-AUDIT-14).
"""

from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BOUNDARY = _REPO_ROOT / "frontend" / "src" / "components" / "GlobalErrorBoundary.jsx"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_anchor_present():
    src = _read(_BOUNDARY)
    assert "P3-NEW-CONSOLE-ERROR-SENTRY" in src, (
        "Falta anchor `P3-NEW-CONSOLE-ERROR-SENTRY` en GlobalErrorBoundary.jsx."
    )


def test_console_error_is_dev_gated():
    """`console.error(...)` debe estar dentro de un `if (import.meta.env.DEV)`
    block. La regex acepta variantes de espaciado."""
    src = _read(_BOUNDARY)
    pat = re.compile(
        r"if\s*\(\s*import\.meta\.env\.DEV\s*\)\s*\{[^}]*?console\.error",
        re.DOTALL,
    )
    assert pat.search(src), (
        "`console.error(...)` debe estar dentro de "
        "`if (import.meta.env.DEV) { console.error(...); }`. "
        "Sin gate, prod expone stack trace + componentStack en DevTools."
    )


def test_zero_ungated_console_error():
    """Cero `console.error(...)` fuera del gate `import.meta.env.DEV`. Si
    alguien añade un nuevo `console.error` sin gate, el test falla loud."""
    src = _read(_BOUNDARY)
    # Encontrar todos los `console.error(`. Para cada uno, verificar que en
    # las 5 líneas previas hay `import.meta.env.DEV`.
    lines = src.splitlines()
    bad = []
    for idx, line in enumerate(lines):
        if "console.error(" in line:
            window = "\n".join(lines[max(0, idx - 5):idx + 1])
            if "import.meta.env.DEV" not in window:
                bad.append(f"línea {idx + 1}: {line.strip()[:120]}")
    assert not bad, (
        f"GlobalErrorBoundary.jsx tiene `console.error(...)` sin gate "
        f"`import.meta.env.DEV` en las 5 líneas previas:\n  "
        + "\n  ".join(bad)
    )


def test_anchor_present_in_test_file():
    src = _read(Path(__file__))
    assert "P3-NEW-CONSOLE-ERROR-SENTRY" in src
