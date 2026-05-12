"""[P3-LAZY-MARKDOWN · 2026-05-12] Anchor + regression guard.

`react-markdown` debe estar en chunk async (lazy import via wrapper
`LazyMarkdown`), NO en static import directo en archivos que entran al
chunk AgentPage. Pre-fix:
  - AgentPage.jsx: `import ReactMarkdown from 'react-markdown'` (DEAD —
    nunca se usaba en el file, pero el static import lo pinneaba al chunk).
  - MessageBubble.jsx + ChatWidget.jsx: usaban `<ReactMarkdown>` con static
    import → react-markdown + remark/mdast deps (~60KB gzip) viajaban en el
    chunk AgentPage de 174KB.

Defensa:
  1. `frontend/src/components/common/LazyMarkdown.jsx` existe y usa
     `React.lazy(() => import('react-markdown'))`.
  2. Ningún archivo de `frontend/src/{pages,components}/**` debe tener
     `import ReactMarkdown from 'react-markdown'` (static) excepto el
     wrapper LazyMarkdown.jsx mismo.
  3. MessageBubble + ChatWidget importan `LazyMarkdown` y usan
     `<LazyMarkdown>` (no `<ReactMarkdown>`).
"""

from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND_SRC = _REPO_ROOT / "frontend" / "src"
_LAZY_WRAPPER = _FRONTEND_SRC / "components" / "common" / "LazyMarkdown.jsx"


def test_lazy_wrapper_exists():
    assert _LAZY_WRAPPER.is_file(), (
        f"Falta wrapper {_LAZY_WRAPPER.relative_to(_REPO_ROOT)}. "
        "Sin él, no hay punto único donde encapsular el lazy + Suspense."
    )


def test_lazy_wrapper_uses_react_lazy_with_dynamic_import():
    src = _LAZY_WRAPPER.read_text(encoding="utf-8")
    # `React.lazy(() => import('react-markdown'))` o `lazy(() => import(...))`
    pat = re.compile(
        r"lazy\s*\(\s*\(\s*\)\s*=>\s*import\s*\(\s*['\"]react-markdown['\"]\s*\)\s*\)",
    )
    assert pat.search(src), (
        "LazyMarkdown debe usar `lazy(() => import('react-markdown'))`. "
        "Sin dynamic import, Vite no genera chunk async separado."
    )


def test_lazy_wrapper_uses_suspense_with_fallback():
    src = _LAZY_WRAPPER.read_text(encoding="utf-8")
    # `<Suspense fallback={...}>`
    pat = re.compile(r"<Suspense\s+fallback\s*=", re.DOTALL)
    assert pat.search(src), (
        "LazyMarkdown debe envolver el render en `<Suspense fallback=...>`. "
        "Sin fallback, React lanza error durante el primer fetch del chunk."
    )


def test_anchor_present_in_wrapper():
    src = _LAZY_WRAPPER.read_text(encoding="utf-8")
    assert "P3-LAZY-MARKDOWN" in src


def test_no_static_react_markdown_import_outside_wrapper():
    """Blanket: ningún archivo JSX/JS bajo frontend/src debe tener
    `import ReactMarkdown from 'react-markdown'` static, excepto el propio
    LazyMarkdown.jsx (que LO USA via React.lazy)."""
    violations: list[str] = []
    for path in _FRONTEND_SRC.rglob("*.jsx"):
        if path == _LAZY_WRAPPER:
            continue
        if "__tests__" in path.parts:
            continue
        src = path.read_text(encoding="utf-8")
        # Static import (no `lazy(`, no `import(`)
        for ln_idx, line in enumerate(src.splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            if re.search(r"^import\s+ReactMarkdown\s+from\s+['\"]react-markdown['\"]", stripped):
                violations.append(f"{path.relative_to(_REPO_ROOT)}:{ln_idx}")

    assert not violations, (
        f"Static `import ReactMarkdown from 'react-markdown'` encontrado fuera "
        f"del wrapper LazyMarkdown.jsx en: {violations}. "
        "Reemplazar con `import LazyMarkdown from '.../LazyMarkdown'` y "
        "usar `<LazyMarkdown>` en lugar de `<ReactMarkdown>`."
    )


def test_message_bubble_uses_lazy_wrapper():
    p = _FRONTEND_SRC / "components" / "agent" / "MessageBubble.jsx"
    src = p.read_text(encoding="utf-8")
    assert "LazyMarkdown" in src, (
        "MessageBubble.jsx debe usar `LazyMarkdown` (no `ReactMarkdown` static)."
    )
    # El JSX debe usar `<LazyMarkdown>` activo (no `<ReactMarkdown>`)
    assert "<LazyMarkdown>" in src, (
        "MessageBubble.jsx falta tag JSX `<LazyMarkdown>` activo."
    )
    assert "<ReactMarkdown>" not in src, (
        "MessageBubble.jsx aún contiene tag `<ReactMarkdown>` — debe ser "
        "`<LazyMarkdown>` tras la migración."
    )


def test_chat_widget_uses_lazy_wrapper():
    p = _FRONTEND_SRC / "components" / "dashboard" / "ChatWidget.jsx"
    src = p.read_text(encoding="utf-8")
    assert "LazyMarkdown" in src and "<LazyMarkdown>" in src
    assert "<ReactMarkdown>" not in src


def test_agent_page_dead_import_removed():
    p = _FRONTEND_SRC / "pages" / "AgentPage.jsx"
    src = p.read_text(encoding="utf-8")
    # No static import de react-markdown (era dead import pre-fix)
    pat = re.compile(r"^import\s+ReactMarkdown\s+from\s+['\"]react-markdown['\"]", re.MULTILINE)
    assert not pat.search(src), (
        "AgentPage.jsx aún tiene `import ReactMarkdown from 'react-markdown'`. "
        "Era dead import (nunca usado en el archivo) y pinneaba el chunk."
    )


def test_anchor_present_in_test_file():
    src = Path(__file__).read_text(encoding="utf-8")
    assert "P3-LAZY-MARKDOWN" in src
