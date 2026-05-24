"""[P1-FRONTEND-HARDEN · 2026-05-23] Test umbrella del bundle 2-en-1
que cerró los 2 gaps P1 reportados por el audit production-readiness
2026-05-23:

  - **GAP-1 (P1-FRONTEND-PRIVATE-MODE):** 5 sitios del frontend con
    `localStorage.getItem(...)` raw que rompían render del provider en
    iOS Safari Private Mode (SecurityError no atrapado). Migrados al
    helper SSOT `safeLocalStorageGet`/`safeLocalStorageRemove`. Test:
    [`test_p1_frontend_private_mode.py`](test_p1_frontend_private_mode.py).

  - **GAP-2 (P1-SETTINGS-CONFIRM-NATIVO):** `Settings.jsx::handleDeleteFact`
    invocaba `window.confirm(...)` nativo (bloqueante, rompía dark
    theme, inaccesible). Reemplazado por `confirmToast` Promise-based
    sobre sonner. Test:
    [`test_p1_settings_confirm_nativo.py`](test_p1_settings_confirm_nativo.py).

Patrón de bundle umbrella (mismo que P1-SWAP-PROD-HARDEN · 2026-05-22):
    Este archivo existe principalmente para satisfacer el contrato
    `test_p2_hist_audit_14_marker_test_link.py` — el slug derivado del
    marker `_LAST_KNOWN_PFIX` DEBE matchear al menos un archivo
    `tests/test_<slug>*.py`. Los 2 sub-tests viven en archivos
    separados (un archivo por gap, mejor diff review + fail isolation
    en CI). Aquí solo verificamos que los 2 archivos existen y que el
    marker está bumpeado correctamente.

Tooltip-anchor: P1-FRONTEND-HARDEN | bundle audit 2026-05-23
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND_ROOT / "app.py"
_TESTS_DIR = _BACKEND_ROOT / "tests"


def test_marker_bumped_to_frontend_harden_or_later():
    """`_LAST_KNOWN_PFIX` DEBE tener fecha >= 2026-05-23. Si retrocedió,
    el bundle fue revertido."""
    text = _APP_PY.read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    assert m, "_LAST_KNOWN_PFIX no encontrado en app.py"
    marker = m.group(1)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert date_match, f"Marker {marker!r} sin fecha parseable"
    assert date_match.group(1) >= "2026-05-23", (
        f"_LAST_KNOWN_PFIX={marker!r} retrocedió antes del bundle "
        f"P1-FRONTEND-HARDEN (2026-05-23). Si revertiste el bundle, "
        f"también actualizar la memoria/CLAUDE.md."
    )


@pytest.mark.parametrize(
    "expected_test_file",
    [
        "test_p1_frontend_private_mode.py",
        "test_p1_settings_confirm_nativo.py",
    ],
)
def test_subtest_files_present(expected_test_file: str):
    """Los 2 archivos del bundle DEBEN existir. Si alguno se borró,
    perdimos cobertura — bloquear merge."""
    path = _TESTS_DIR / expected_test_file
    assert path.exists(), (
        f"P1-FRONTEND-HARDEN: el sub-test `{expected_test_file}` "
        f"fue eliminado. Restaurar el archivo (parte del bundle "
        f"audit 2026-05-23) o consolidar su cobertura aquí."
    )


def test_marker_slug_matches_this_file():
    """Sanity del cross-link `test_p2_hist_audit_14_marker_test_link`:
    el slug derivado del marker DEBE matchear el filename de este
    test. Si no, el enforcer P2-HIST-AUDIT-14 fallaría primero."""
    text = _APP_PY.read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    marker = m.group(1)
    prefix = marker.split("·", 1)[0].strip()
    slug = prefix.replace("-", "_").lower()
    if slug == "p1_frontend_harden":
        assert "p1_frontend_harden" in __file__.replace("\\", "/").lower(), (
            "Filename de este test no contiene `p1_frontend_harden`."
        )
    else:
        # Marker bumpeado por otro P-fix — este test queda como historial
        # del bundle. No es un fallo.
        pytest.skip(
            f"Marker actual ({slug}) != p1_frontend_harden — bundle "
            f"posterior bumpeó el marker. Este test queda como "
            f"anchor histórico."
        )
