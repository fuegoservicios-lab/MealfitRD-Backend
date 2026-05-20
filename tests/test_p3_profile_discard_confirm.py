"""[P3-PROFILE-DISCARD-CONFIRM · 2026-05-20] Test anti-regresión del
modal de confirmación al click "Volver" con body metrics drafts.

Bug pre-fix:
    El click en "Volver" navegaba inmediatamente al dashboard (o al
    listado en mobile). El cleanup useEffect revertía drafts pendientes
    de peso/altura silencio. Un user que tipeó nuevos números y click
    "Volver" por error perdía los datos sin advertencia explícita —
    el banner amarillo existía pero solo era visible mientras estaban
    en Profile.

Fix:
    State `showDiscardConfirm` + Modal de confirmación. Click "Volver"
    cuando `bodyMetricsChanged && activeSection === 'profile'` → modal.
    Sino → navegación directa.

    Modal: "Tienes cambios sin guardar" / "Seguir editando" (cierra) /
    "Descartar y salir" (navega — el cleanup revierte).
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SETTINGS_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Settings.jsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_marker_present_as_tooltip_anchor():
    """[P3-PROFILE-DISCARD-CONFIRM] Marker presente como tooltip-anchor
    para que un renombre futuro falle el test antes de tocar producción."""
    src = _read(_SETTINGS_JSX)
    assert "P3-PROFILE-DISCARD-CONFIRM" in src, (
        "Marker `P3-PROFILE-DISCARD-CONFIRM` ausente. Si quieres remover "
        "el fix, primero remueve este test."
    )


def test_show_discard_confirm_state_declared():
    """[P3-PROFILE-DISCARD-CONFIRM] State `showDiscardConfirm` declarado."""
    src = _read(_SETTINGS_JSX)
    assert re.search(
        r"const\s*\[\s*showDiscardConfirm,\s*setShowDiscardConfirm\s*\]",
        src,
    ), (
        "State `[showDiscardConfirm, setShowDiscardConfirm]` no declarado. "
        "Restaurar el state del modal."
    )


def test_exit_navigation_helper_extracted():
    """[P3-PROFILE-DISCARD-CONFIRM] `_doExitNavigation` está definido
    como helper SSOT — el handler inline del onClick + el botón
    "Descartar y salir" del modal lo invocan ambos. Sin extracción,
    la lógica está duplicada (y puede driftear)."""
    src = _read(_SETTINGS_JSX)
    assert re.search(r"const\s+_doExitNavigation\s*=\s*\(", src), (
        "Helper `_doExitNavigation` no encontrado. Sin la extracción, "
        "el onClick del Volver y el del Discard modal tendrían lógica "
        "duplicada (vector de drift)."
    )


def test_volver_button_gates_on_body_metrics_changed():
    """[P3-PROFILE-DISCARD-CONFIRM] El onClick del Volver tiene gate
    `if (bodyMetricsChanged && activeSection === 'profile')` que abre
    el modal antes de navegar. Sin el gate, el modal nunca se muestra."""
    src = _read(_SETTINGS_JSX)
    # Buscar el bloque del botón Volver (className exitSettingsBtn).
    btn_idx = src.find("exitSettingsBtn")
    assert btn_idx != -1, "Botón Volver (className exitSettingsBtn) no encontrado."
    # Tomar ventana de ~4000 chars después.
    btn_window = src[btn_idx:btn_idx + 4000]
    assert re.search(
        r"bodyMetricsChanged\s*&&\s*activeSection\s*===\s*['\"]profile['\"]",
        btn_window,
    ), (
        "Gate `bodyMetricsChanged && activeSection === 'profile'` ausente "
        "del onClick de Volver. El modal de discard nunca se mostraría."
    )
    assert "setShowDiscardConfirm(true)" in btn_window, (
        "El onClick del Volver no invoca `setShowDiscardConfirm(true)` "
        "cuando hay drafts pendientes — el gate detecta pero no actúa."
    )


def _extract_button_blocks(modal_block: str):
    """Particiona el modal en bloques `<button ...>...</button>`. Cada
    bloque captura: opening tag (incluye onClick + style props), label
    de texto, closing tag."""
    return re.findall(
        r"<button\b[^>]*?onClick=\{[^}]*?(?:\{[^}]*\}[^}]*)*\}[^>]*>([\s\S]*?)</button>",
        modal_block,
    )


def _extract_modal_block(src: str) -> str:
    """Extrae el body del Modal de discard (`titleId='discard-confirm-title'`)
    hasta su cierre </Modal>. Más robusto que ventanas de N chars fijos."""
    start = src.find("discard-confirm-title")
    assert start != -1, "Modal de discard no encontrado."
    # Retroceder al `<Modal` de apertura más cercano.
    open_idx = src.rfind("<Modal", 0, start)
    assert open_idx != -1
    close_idx = src.find("</Modal>", start)
    assert close_idx != -1
    return src[open_idx:close_idx + len("</Modal>")]


def test_modal_renders_with_confirm_and_cancel_actions():
    """[P3-PROFILE-DISCARD-CONFIRM] El Modal renderiza dos botones:
    'Seguir editando' (cierra modal sin tocar) y 'Descartar y salir'
    (cierra + invoca _doExitNavigation). Sin uno de ellos, el flow
    queda atascado o el descartar no navega."""
    src = _read(_SETTINGS_JSX)
    # Heurística: localizar el Modal de discard por su titleId.
    modal_idx = src.find("discard-confirm-title")
    assert modal_idx != -1, (
        "Modal de discard (`titleId='discard-confirm-title'`) no encontrado."
    )
    # Tomar ventana de ~4000 chars hacia delante.
    modal_window = src[modal_idx:modal_idx + 4000]
    assert "Seguir editando" in modal_window, (
        "Botón 'Seguir editando' ausente del modal de discard."
    )
    assert "Descartar y salir" in modal_window, (
        "Botón 'Descartar y salir' ausente del modal de discard."
    )
    # El botón Descartar debe invocar _doExitNavigation.
    discard_block_start = modal_window.find("Descartar y salir")
    # Mirar ~1000 chars antes del label (donde está el onClick).
    pre = modal_window[max(0, discard_block_start - 1000):discard_block_start]
    assert "_doExitNavigation()" in pre, (
        "Botón 'Descartar y salir' no invoca `_doExitNavigation()`. Sin "
        "esto, el botón cierra el modal pero el user queda en Settings — "
        "el flow no termina."
    )


def test_modal_closes_on_seguir_editando():
    """[P3-PROFILE-DISCARD-CONFIRM] 'Seguir editando' cierra el modal
    SIN invocar _doExitNavigation (queda en Settings). Sin esta
    separación, el botón Seguir y Descartar serían equivalentes."""
    src = _read(_SETTINGS_JSX)
    modal_idx = src.find("discard-confirm-title")
    modal_window = src[modal_idx:modal_idx + 4000]
    seguir_idx = modal_window.find("Seguir editando")
    assert seguir_idx != -1
    pre = modal_window[max(0, seguir_idx - 800):seguir_idx]
    assert "setShowDiscardConfirm(false)" in pre, (
        "`setShowDiscardConfirm(false)` no aparece antes del label 'Seguir "
        "editando'. El botón no cierra el modal — flow atascado."
    )
    # _doExitNavigation NO debe aparecer en el handler del Seguir (heurística:
    # los 400 chars previos al label de Seguir contienen solo setShowDiscardConfirm).
    assert "_doExitNavigation" not in pre, (
        "'Seguir editando' invoca `_doExitNavigation` — eso es el handler "
        "de 'Descartar y salir'. Click en Seguir NO debe navegar."
    )


def test_last_known_pfix_bumped():
    """[P3-PROFILE-DISCARD-CONFIRM] Marker bumped en backend/app.py."""
    app_py = _REPO_ROOT / "backend" / "app.py"
    src = app_py.read_text(encoding="utf-8")
    match = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert match
    marker = match.group(1)
    assert "P3-PROFILE-DISCARD-CONFIRM" in marker or "2026-05-20" in marker, (
        f"_LAST_KNOWN_PFIX={marker!r} no refleja este P-fix."
    )
