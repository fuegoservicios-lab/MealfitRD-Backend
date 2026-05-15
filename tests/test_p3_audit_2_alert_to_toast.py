"""[P3-AUDIT-2 · 2026-05-15] Test parser-based: los 2 callsites legacy
`alert()` nativo en `AgentPage.jsx` y `PaymentModal.jsx` reemplazados por
`toast.error(...)` (sonner).

Por qué este test:
    El resto de la app usa `sonner` consistentemente para feedback al
    usuario. 2 callsites legacy con `alert()` nativo:
    - Bloquean el thread (modal-blocking dialog).
    - Rompen consistencia UX (no respetan el theme dark, no animación,
      no swipe-to-dismiss).
    - En mobile son particularmente intrusivos.

Fix esperado:
    - `AgentPage.jsx::processSelectedFile`: `alert('Formato no soportado...')`
      → `toast.error('Formato no soportado...')`.
    - `PaymentModal.jsx::handleCreateSubscription`: `alert('Plan ID Anual no
      configurado.')` → `toast.error('Plan ID Anual no configurado.')`
      (+ añadir import de sonner en PaymentModal.jsx que antes no lo tenía).

Drift detection:
    - Cero `alert(...)` activos en los 2 archivos (strip comments first).
    - PaymentModal.jsx importa `toast` de sonner.
    - AgentPage.jsx ya importaba `toast` de sonner (pre-fix) — assert
      preserva el import.

Cross-link convention (P2-HIST-AUDIT-14): slug `p3_audit_2`.

Tooltip-anchor: P3-AUDIT-2-START | gap audit 2026-05-15
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENTPAGE_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "AgentPage.jsx"
_PAYMENT_MODAL_JSX = _REPO_ROOT / "frontend" / "src" / "components" / "dashboard" / "PaymentModal.jsx"


@pytest.fixture(scope="module")
def agentpage_src() -> str:
    return _AGENTPAGE_JSX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def payment_modal_src() -> str:
    return _PAYMENT_MODAL_JSX.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    """Quita line comments (//) + block comments (/* */) + JSX comments
    (`{/* */}`)."""
    s = re.sub(r"//[^\n]*", "", src)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    s = re.sub(r"\{\s*/\*.*?\*/\s*\}", "", s, flags=re.DOTALL)
    return s


# ---------------------------------------------------------------------------
# 1. AgentPage.jsx: cero `alert(...)` activos
# ---------------------------------------------------------------------------
def test_agentpage_no_native_alert(agentpage_src: str):
    no_comments = _strip_comments(agentpage_src)
    # Buscar `alert(...)` activo. Excluir window.alert (no usado en este repo).
    pattern = re.compile(r"(?<![\w.])alert\s*\(")
    matches = pattern.findall(no_comments)
    assert not matches, (
        f"P3-AUDIT-2 regresión: {len(matches)} callsites `alert(...)` "
        f"activos en AgentPage.jsx. Reemplazar por `toast.error(...)` (sonner)."
    )


def test_agentpage_imports_toast_from_sonner(agentpage_src: str):
    assert re.search(
        r"import\s*\{[^}]*\btoast\b[^}]*\}\s*from\s*['\"]sonner['\"]",
        agentpage_src,
    ), (
        "P3-AUDIT-2 regresión: `import { toast } from 'sonner'` perdido en "
        "AgentPage.jsx. Sin import, `toast.error` sería ReferenceError."
    )


def test_agentpage_uses_toast_error(agentpage_src: str):
    """El replacement del `alert` debe ser `toast.error` (no `toast.info` o
    `toast.success`) — es feedback de un input inválido."""
    no_comments = _strip_comments(agentpage_src)
    assert re.search(
        r"toast\.error\s*\(\s*['\"][^'\"]*[Ff]ormato\s+no\s+soportado",
        no_comments,
    ), (
        "P3-AUDIT-2 regresión: `toast.error('Formato no soportado...')` no "
        "encontrado en AgentPage.jsx. El replacement del `alert` original "
        "debe usar el método error (rojo, lectura assertiva)."
    )


# ---------------------------------------------------------------------------
# 2. PaymentModal.jsx: cero `alert(...)` + import sonner
# ---------------------------------------------------------------------------
def test_paymentmodal_no_native_alert(payment_modal_src: str):
    no_comments = _strip_comments(payment_modal_src)
    pattern = re.compile(r"(?<![\w.])alert\s*\(")
    matches = pattern.findall(no_comments)
    assert not matches, (
        f"P3-AUDIT-2 regresión: {len(matches)} callsites `alert(...)` "
        f"activos en PaymentModal.jsx. Reemplazar por `toast.error(...)`."
    )


def test_paymentmodal_imports_toast_from_sonner(payment_modal_src: str):
    assert re.search(
        r"import\s*\{[^}]*\btoast\b[^}]*\}\s*from\s*['\"]sonner['\"]",
        payment_modal_src,
    ), (
        "P3-AUDIT-2 regresión: `import { toast } from 'sonner'` no encontrado "
        "en PaymentModal.jsx. PRE-FIX el archivo NO importaba sonner — al "
        "reemplazar `alert(...)` por `toast.error(...)`, añadir el import "
        "evita ReferenceError en runtime."
    )


def test_paymentmodal_uses_toast_error(payment_modal_src: str):
    no_comments = _strip_comments(payment_modal_src)
    assert re.search(
        r"toast\.error\s*\(\s*['\"][^'\"]*Plan ID",
        no_comments,
    ), (
        "P3-AUDIT-2 regresión: `toast.error('Plan ID Anual no configurado.')` "
        "no encontrado en PaymentModal.jsx. El replacement debe usar el "
        "método error."
    )


# ---------------------------------------------------------------------------
# 3. Anchor textual P3-AUDIT-2 presente en ambos archivos
# ---------------------------------------------------------------------------
def test_anchor_present_agentpage(agentpage_src: str):
    assert "P3-AUDIT-2" in agentpage_src, (
        "P3-AUDIT-2 regresión: anchor textual perdido en AgentPage.jsx."
    )


def test_anchor_present_paymentmodal(payment_modal_src: str):
    assert "P3-AUDIT-2" in payment_modal_src, (
        "P3-AUDIT-2 regresión: anchor textual perdido en PaymentModal.jsx."
    )
