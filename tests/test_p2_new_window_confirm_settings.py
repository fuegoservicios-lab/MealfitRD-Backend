"""[P2-NEW-WINDOW-CONFIRM-SETTINGS · 2026-05-15] Anchor + regression guard.

`Settings.jsx` tenía 2 `window.confirm(...)` legacy en el flujo de
Renovar/Cero (mini-confirm "¿consumir 1 regeneración?"). Modal nativo
del browser:
  - Rompe el dark theme (Settings es dark-first).
  - Bloquea el thread principal.
  - No es a11y-friendly (sin aria-live, focus management inconsistente).
  - Imposible de testear mecánicamente.

Plan.jsx ya migró su confirm a un modal propio (`P6-CANCEL-MODAL`); para
los mini-confirms inline de Settings introducimos un helper
Promise-based (`confirmToast`) que envuelve `sonner` con `action`/`cancel`.
Más liviano que un modal full-screen y reutilizable.

Defensas que el test enforza:
  1. Anchor `P2-NEW-WINDOW-CONFIRM-SETTINGS` presente en Settings.jsx.
  2. Helper `confirmToast` exportado desde `frontend/src/utils/confirmToast.js`.
  3. Settings.jsx importa `confirmToast`.
  4. Cero `window.confirm(` en Settings.jsx.
  5. ≥2 invocaciones de `confirmToast(` en Settings.jsx (los 2 callsites
     migrados: Renovar + Cero).
  6. El helper retorna una Promise (truth-path obligatorio para await).
  7. Anchor presente en este archivo (cross-link guard P2-HIST-AUDIT-14).
"""

from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS = _REPO_ROOT / "frontend" / "src" / "pages" / "Settings.jsx"
_CONFIRM_TOAST = _REPO_ROOT / "frontend" / "src" / "utils" / "confirmToast.js"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_helper_file_exists_and_exports_confirm_toast():
    assert _CONFIRM_TOAST.exists(), (
        f"Helper `confirmToast.js` no encontrado en {_CONFIRM_TOAST}."
    )
    src = _read(_CONFIRM_TOAST)
    assert "export function confirmToast" in src, (
        "`confirmToast` debe ser exportado named function desde el módulo."
    )
    # Debe retornar una Promise — patrón `return new Promise(...)`.
    assert re.search(r"return\s+new\s+Promise\s*\(", src), (
        "`confirmToast` debe retornar `new Promise(...)` para soportar "
        "`await confirmToast(msg)` en handlers async."
    )


def test_helper_uses_sonner_action_and_cancel():
    """Sonner-canonical: el toast debe pasar `action` y `cancel` props con
    onClick handlers. Sin esto, el toast es informational y el caller no
    tiene cómo afirmar/negar."""
    src = _read(_CONFIRM_TOAST)
    assert re.search(r"action\s*:\s*\{[^}]*onClick", src, re.DOTALL), (
        "Helper debe pasar `action: { label, onClick }` a sonner."
    )
    assert re.search(r"cancel\s*:\s*\{[^}]*onClick", src, re.DOTALL), (
        "Helper debe pasar `cancel: { label, onClick }` a sonner."
    )


def test_anchor_present_in_settings():
    src = _read(_SETTINGS)
    assert "P2-NEW-WINDOW-CONFIRM-SETTINGS" in src, (
        "Falta anchor `P2-NEW-WINDOW-CONFIRM-SETTINGS` en Settings.jsx."
    )


def test_settings_imports_confirm_toast():
    src = _read(_SETTINGS)
    pat = re.compile(
        r"import\s*\{\s*[^}]*\bconfirmToast\b[^}]*\}\s*from\s*['\"][^'\"]*confirmToast['\"]",
        re.DOTALL,
    )
    assert pat.search(src), (
        "Settings.jsx debe importar `confirmToast` desde `../utils/confirmToast`."
    )


def test_zero_window_confirm_in_settings():
    src = _read(_SETTINGS)
    bad = re.findall(r"\bwindow\.confirm\s*\(", src)
    assert not bad, (
        f"Settings.jsx tiene {len(bad)} callsites `window.confirm(`. "
        f"Reemplazar por `await confirmToast(...)` (handler debe ser async)."
    )


def test_settings_invokes_confirm_toast_at_least_twice():
    """Los 2 callsites pre-fix (`window.confirm` Renovar + Cero) deben estar
    migrados a `confirmToast`. Si alguien los borra sin reemplazar, el test
    falla loud — confirmando que la UX no regresó a "no preguntar"."""
    src = _read(_SETTINGS)
    calls = re.findall(r"\bconfirmToast\s*\(", src)
    assert len(calls) >= 2, (
        f"Settings.jsx debe tener ≥2 invocaciones de `confirmToast(` (Renovar + "
        f"Cero). Encontrados: {len(calls)}. Si removiste un confirm "
        f"intencionalmente, actualiza el conteo del test."
    )


def test_anchor_present_in_test_file():
    """Cross-link guard P2-HIST-AUDIT-14: el slug del marker
    `p2_new_window_confirm_settings` matchea el nombre del test file."""
    src = _read(Path(__file__))
    assert "P2-NEW-WINDOW-CONFIRM-SETTINGS" in src
