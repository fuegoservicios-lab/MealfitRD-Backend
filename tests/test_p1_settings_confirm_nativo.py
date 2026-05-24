"""[P1-SETTINGS-CONFIRM-NATIVO · 2026-05-23] Settings.jsx NO debe usar
`confirm()`, `alert()` ni `prompt()` nativos del browser.

Motivación:
    `Settings.jsx::handleDeleteFact` invocaba `window.confirm(...)` para
    confirmar el delete de un user_fact. El nativo:
      - Bloquea el event loop (browser unresponsive durante el prompt).
      - Rompe el dark theme (modal nativo del browser, no styled).
      - No es a11y-friendly (sin aria-live, sin focus management).
      - Imposible de testear mecánicamente.

    El resto del codebase ya usa `confirmToast` (helper Promise-based
    sobre sonner, P2-NEW-WINDOW-CONFIRM-SETTINGS · 2026-05-15) — el
    callsite legacy era el único holdout en Settings.jsx. Tests
    legacy en `AgentPage.jsx:1035` y `PaymentModal.jsx:125` ya cerraron
    `alert()` (P3-AUDIT-2 · 2026-05-15).

Scope:
    Este test enforza el contrato SOLO sobre `Settings.jsx`. Para un
    blanket scan cross-frontend, un test futuro `test_p2_no_native_dialogs.py`
    podría escanear todos los archivos — fuera de scope del bundle
    P1-FRONTEND-HARDEN.

Tooltip-anchor: P1-SETTINGS-CONFIRM-NATIVO | regression guard 2026-05-23
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SETTINGS_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Settings.jsx"


_NATIVE_DIALOG_PATTERN = re.compile(
    # Captura `confirm(`, `alert(`, `prompt(` con word boundary inicial
    # para evitar matches como `myConfirm(` o `noAlert(`.
    r"(?<![\w.])(?P<fn>confirm|alert|prompt)\s*\("
)


def _strip_js_comments(src: str) -> str:
    """Mismo stripper que test_p1_new_a — elimina /* ... */ y //...EOL
    preservando números de línea."""
    no_block = re.sub(r"/\*[\s\S]*?\*/", "", src)
    no_line = re.sub(r"//[^\n]*", "", no_block)
    return no_line


def test_settings_no_native_dialogs():
    """Settings.jsx (código ejecutable, no comentarios) NO debe llamar
    a `confirm()`, `alert()` ni `prompt()` nativos. Si necesitas
    confirmación, usa `confirmToast` del helper SSOT
    `frontend/src/utils/confirmToast.js`."""
    src = _SETTINGS_JSX.read_text(encoding="utf-8")
    no_comments = _strip_js_comments(src)
    src_lines = src.splitlines()

    offenders: list[str] = []
    for m in _NATIVE_DIALOG_PATTERN.finditer(no_comments):
        line_no = no_comments.count("\n", 0, m.start()) + 1
        fn = m.group("fn")
        # Sanity: si la línea original (con comentarios) tiene el match
        # solo dentro de un string literal, dejarlo pasar. Heurística:
        # mismo char index en línea original entre comillas balanceadas.
        line_text = src_lines[line_no - 1] if line_no - 1 < len(src_lines) else ""
        # Skip caso edge: línea como `'Press OK to confirm(...)'` — el
        # fn está dentro de string. Detección simple: si la línea
        # original NO tiene el fn como llamada (sin string wrapping),
        # skip. La heurística cuenta quotes a la izquierda del match.
        col = m.start() - no_comments.rfind("\n", 0, m.start()) - 1
        prefix = line_text[:col] if 0 <= col <= len(line_text) else ""
        # Contar comillas no escapadas en el prefix; si impar → dentro
        # de string.
        single_q = len(re.findall(r"(?<!\\)'", prefix))
        double_q = len(re.findall(r'(?<!\\)"', prefix))
        backtick = len(re.findall(r"(?<!\\)`", prefix))
        if (single_q % 2 == 1) or (double_q % 2 == 1) or (backtick % 2 == 1):
            continue
        offenders.append(
            f"  Settings.jsx:{line_no} → `{fn}(...)` en código ejecutable"
        )

    assert not offenders, (
        "P1-SETTINGS-CONFIRM-NATIVO violation: Settings.jsx llama a un "
        "dialog nativo del browser. Estos:\n"
        "  - Bloquean el event loop.\n"
        "  - Rompen el dark theme.\n"
        "  - No son a11y-friendly.\n"
        "  - Imposibles de testear mecánicamente.\n\n"
        "Offenders:\n" + "\n".join(offenders) + "\n\n"
        "Fix: para confirm() reemplazar con `confirmToast(message, opts)` "
        "del helper `frontend/src/utils/confirmToast.js` (Promise<boolean>). "
        "Para alert() usar `toast.error(...)` o `toast.success(...)` de "
        "sonner. Para prompt() construir modal custom (no hay helper "
        "Promise-based aún)."
    )


def test_confirmtoast_imported_in_settings():
    """Sanity: si Settings.jsx ya no importa `confirmToast`, no hay
    sustituto disponible para futuras confirmaciones — alerta visible
    en code review."""
    src = _SETTINGS_JSX.read_text(encoding="utf-8")
    assert "from '../utils/confirmToast'" in src or "from \"../utils/confirmToast\"" in src, (
        "P1-SETTINGS-CONFIRM-NATIVO: Settings.jsx ya no importa "
        "`confirmToast`. Si removiste todos los call sites, el import "
        "queda dead code OK — pero si esto es accidental, la próxima "
        "confirmación añadida volverá al patrón nativo. Mantener "
        "el import como anchor del patrón canónico."
    )
