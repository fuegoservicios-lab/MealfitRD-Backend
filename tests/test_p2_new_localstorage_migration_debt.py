"""[P2-NEW-LOCALSTORAGE-MIGRATION-DEBT · 2026-05-15] Anchor + regression guard.

`P2-AUDIT-3 · 2026-05-15` introdujo `safeLocalStorageSet` en
`frontend/src/utils/safeLocalStorage.js` y migró 13 callsites de
`AssessmentContext.jsx`. Quedaron sin migrar:
  - `frontend/src/components/dashboard/ChatWidget.jsx` (5 callsites raw setItem)
  - `frontend/src/pages/AgentPage.jsx` (9 callsites raw setItem)

Modo de fallo del setItem raw: en iOS Private Mode la cuota efectiva es 0
→ `QuotaExceededError` uncaught corta el flow de sesión guest, dejando
state inconsistente entre React (in-memory) y storage. Ej. usuario guest
crea sesión nueva → setItem('mealfit_guest_session', newId) lanza →
`globalCancelSessionId` queda con UUID viejo → SSE cancel apunta al
session_id incorrecto.

Defensas que el test enforza:
  1. Anchor `P2-NEW-LOCALSTORAGE-MIGRATION-DEBT` presente en ambos archivos.
  2. Import de `safeLocalStorageSet` presente.
  3. Cero `localStorage.setItem(` raw en los 2 archivos migrados.
  4. Anchor presente en este archivo (cross-link guard P2-HIST-AUDIT-14).
"""

from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHAT_WIDGET = _REPO_ROOT / "frontend" / "src" / "components" / "dashboard" / "ChatWidget.jsx"
_AGENT_PAGE = _REPO_ROOT / "frontend" / "src" / "pages" / "AgentPage.jsx"
_HELPER = _REPO_ROOT / "frontend" / "src" / "utils" / "safeLocalStorage.js"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_helper_still_exists():
    """`safeLocalStorageSet` (introducido P2-AUDIT-3) debe seguir existiendo
    en `utils/safeLocalStorage.js`. Sin él, los imports de los 2 archivos
    migrados rompen al build-time."""
    assert _HELPER.exists(), f"Helper SSOT desaparecido: {_HELPER}"
    src = _read(_HELPER)
    assert "export function safeLocalStorageSet" in src, (
        "`safeLocalStorageSet` no está exportado desde safeLocalStorage.js."
    )


def test_anchor_present_in_chat_widget():
    src = _read(_CHAT_WIDGET)
    assert "P2-NEW-LOCALSTORAGE-MIGRATION-DEBT" in src, (
        "Falta anchor `P2-NEW-LOCALSTORAGE-MIGRATION-DEBT` en ChatWidget.jsx."
    )


def test_anchor_present_in_agent_page():
    src = _read(_AGENT_PAGE)
    assert "P2-NEW-LOCALSTORAGE-MIGRATION-DEBT" in src, (
        "Falta anchor `P2-NEW-LOCALSTORAGE-MIGRATION-DEBT` en AgentPage.jsx."
    )


def test_chat_widget_imports_helper():
    src = _read(_CHAT_WIDGET)
    pat = re.compile(
        r"import\s*\{\s*[^}]*\bsafeLocalStorageSet\b[^}]*\}\s*from\s*['\"][^'\"]*safeLocalStorage['\"]",
        re.DOTALL,
    )
    assert pat.search(src), (
        "ChatWidget.jsx debe importar `safeLocalStorageSet` desde "
        "`../../utils/safeLocalStorage`."
    )


def test_agent_page_imports_helper():
    src = _read(_AGENT_PAGE)
    pat = re.compile(
        r"import\s*\{\s*[^}]*\bsafeLocalStorageSet\b[^}]*\}\s*from\s*['\"][^'\"]*safeLocalStorage['\"]",
        re.DOTALL,
    )
    assert pat.search(src), (
        "AgentPage.jsx debe importar `safeLocalStorageSet` desde "
        "`../utils/safeLocalStorage`."
    )


def test_zero_raw_setitem_in_chat_widget():
    src = _read(_CHAT_WIDGET)
    bad = re.findall(r"\blocalStorage\.setItem\s*\(", src)
    assert not bad, (
        f"ChatWidget.jsx tiene {len(bad)} callsites raw `localStorage.setItem(`. "
        f"Reemplazar por `safeLocalStorageSet(` (mismo argumento, devuelve "
        f"boolean en lugar de throw)."
    )


def test_zero_raw_setitem_in_agent_page():
    src = _read(_AGENT_PAGE)
    bad = re.findall(r"\blocalStorage\.setItem\s*\(", src)
    assert not bad, (
        f"AgentPage.jsx tiene {len(bad)} callsites raw `localStorage.setItem(`. "
        f"Reemplazar por `safeLocalStorageSet(`."
    )


def test_anchor_present_in_test_file():
    """Cross-link guard P2-HIST-AUDIT-14."""
    src = _read(Path(__file__))
    assert "P2-NEW-LOCALSTORAGE-MIGRATION-DEBT" in src
