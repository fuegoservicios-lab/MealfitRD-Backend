"""[P2-STRICT-MODE-ENABLE · 2026-05-12] Anchor + regression guard.

`frontend/src/main.jsx` debe envolver `<App />` en `<StrictMode>`. Pre-fix
estaba comentado por bugs antiguos (toasts duplicados, doble-fetch Plan.jsx)
que ya están guard-eados con `useRef` + sentinels P1-NEW-4. StrictMode en
dev/test detecta nuevas side-effects ANTES de prod (en prod es no-op).

Defensas que el test enforza:
  1. Anchor `P2-STRICT-MODE-ENABLE` en main.jsx.
  2. `<StrictMode>` tag activo (no comentado).
  3. `<App />` envuelto dentro de `<StrictMode>`.
"""

from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAIN = _REPO_ROOT / "frontend" / "src" / "main.jsx"


def _read() -> str:
    return _MAIN.read_text(encoding="utf-8")


def test_anchor_present():
    src = _read()
    assert "P2-STRICT-MODE-ENABLE" in src, (
        "Falta anchor `P2-STRICT-MODE-ENABLE` en frontend/src/main.jsx."
    )


def test_strict_mode_is_active_not_commented():
    """`<StrictMode>` debe aparecer activo (no comentado `// <StrictMode>`)."""
    src = _read()
    # Verificar que existe al menos una línea con `<StrictMode>` que NO sea
    # un comentario de línea (`//`) o de bloque (`/*`).
    lines = src.splitlines()
    found_active = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            continue
        if "<StrictMode>" in line:
            found_active = True
            break
    assert found_active, (
        "`<StrictMode>` está comentado o ausente. Debe envolver `<App />` "
        "activamente (no `// <StrictMode>`)."
    )


def test_strict_mode_wraps_app():
    """La cadena `<StrictMode>...<App />.../StrictMode>` debe estar presente.
    Permite GlobalErrorBoundary entre medio."""
    src = _read()
    # Stripear comentarios de línea para análisis
    no_comments = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("//")
    )
    # Match permisivo: <StrictMode>...<App ... />...</StrictMode>
    pat = re.compile(
        r"<StrictMode>.*?<App\s*/>.*?</StrictMode>",
        re.DOTALL,
    )
    assert pat.search(no_comments), (
        "`<App />` no está envuelto dentro de `<StrictMode>...</StrictMode>`. "
        "El render tree debe pasar por StrictMode para detectar side-effects."
    )


def test_anchor_present_in_test_file():
    src = Path(__file__).read_text(encoding="utf-8")
    assert "P2-STRICT-MODE-ENABLE" in src
