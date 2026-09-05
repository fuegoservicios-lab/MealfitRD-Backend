"""[P2-CONFIRM-DIALOG-PLACEMENT · 2026-09-04] Las confirmaciones de `confirmToast` dejan de ser
un toast accionable arriba de la pantalla («parece una notificación y se visualiza poquito» al
borrar una comida del diario) y pasan a un diálogo real sobre el `Modal` común: centrado en
escritorio, hoja inferior en móvil, foco atrapado, Escape y clic fuera = cancelar.

Contrato que NO cambia (lo anclan `test_p2_new_window_confirm_settings` y
`test_p1_settings_confirm_nativo`): mismo módulo, mismo nombre exportado, `new Promise(...)`,
mismos `opts`. Lo nuevo: un host único montado en App.jsx, `danger` en los borrados, y el toast
solo como respaldo cuando nadie dibuja el diálogo (tests unitarios, `toastFn` inyectado).
"""
from __future__ import annotations

import re
from pathlib import Path

_FRONT = Path(__file__).resolve().parents[2] / "frontend" / "src"


def _src(rel: str) -> str:
    return (_FRONT / rel).read_text(encoding="utf-8")


def test_host_exists_and_is_mounted_once_in_app():
    host = _src("components/common/ConfirmDialogHost.jsx")
    assert "subscribeConfirmHost" in host and "isBottomSheetOnMobile={true}" in host
    assert 'role="alertdialog"' in host
    app = _src("App.jsx")
    assert app.count("<ConfirmDialogHost />") == 1
    assert "import ConfirmDialogHost from './components/common/ConfirmDialogHost';" in app


def test_confirm_toast_keeps_its_api_and_prefers_the_dialog():
    src = _src("utils/confirmToast.js")
    assert "export function confirmToast" in src
    assert "export function subscribeConfirmHost" in src and "export function hasConfirmHost" in src
    # el diálogo manda; el toast queda como respaldo explícito
    assert "if (!toastFn && hasConfirmHost())" in src
    assert "className: 'bb-confirm-toast'" in src
    assert "P2-CONFIRM-DIALOG-PLACEMENT" in src


def test_destructive_confirmations_are_marked_danger():
    tp = _src("components/dashboard/TrackingProgress.jsx")
    assert re.search(r"confirmToast\([\s\S]{0,300}danger: true", tp)
    settings = _src("pages/Settings.jsx")
    assert re.search(r"olvidar esta informaci[oó]n[\s\S]{0,200}danger: true", settings)
    # pausar planes NO es destructivo: sin danger
    i = settings.index("¿Pausar la generación de planes?")
    assert "danger: true" not in settings[i:i + 900]


def test_marker_present():
    app = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "P2-CONFIRM-DIALOG-PLACEMENT · 2026-09-04" in app
