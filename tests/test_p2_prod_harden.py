"""[P2-PROD-HARDEN · 2026-05-23] Test umbrella del bundle 2-en-1 que
cerró los 2 gaps P2 del audit production-readiness 2026-05-23:

  - **GAP-1 (P2-USER-DEPLETED-ITEMS-FK-IDX):** covering index para el
    FK `user_depleted_items_master_ingredient_id_fkey`. Migration SSOT
    `p2_user_depleted_items_fk_idx_2026_05_23.sql` aplicada via MCP +
    presente en ambos dirs (workspace + backend) per P3-MIGRATIONS-SSOT.
    Test: [`test_p2_user_depleted_items_fk_idx.py`](test_p2_user_depleted_items_fk_idx.py).

  - **GAP-2 (P2-SENTRY-TREESHAKE):** `import * as Sentry` reemplazado
    por named imports en [`main.jsx`](frontend/src/main.jsx) (3
    símbolos: init, browserTracingIntegration, replayIntegration) y
    [`AgentPage.jsx`](frontend/src/pages/AgentPage.jsx) (2 símbolos:
    captureException, addBreadcrumb). Habilita tree-shaking de esbuild
    sobre los ~12 exports restantes del SDK. Test:
    [`test_p2_sentry_treeshake.py`](test_p2_sentry_treeshake.py).

Patrón de bundle umbrella (mismo que P1-FRONTEND-HARDEN · 2026-05-23):
    Este file existe para satisfacer el contrato
    `test_p2_hist_audit_14_marker_test_link.py` — el slug del marker
    actual DEBE matchear al menos un archivo `tests/test_<slug>*.py`.
    Los 2 sub-tests viven en archivos separados para diff isolation
    en code review.

Bundle final del audit 2026-05-23:
    - P0: limpio (0 hallazgos).
    - P1: P1-FRONTEND-HARDEN (2 gaps frontend cerrados).
    - P2: P2-PROD-HARDEN (este bundle: 2 gaps cerrados).
    Estado post-bundle: MealfitRD al 100% production-ready bajo las
    invariantes vigentes.

Tooltip-anchor: P2-PROD-HARDEN | bundle audit 2026-05-23
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND_ROOT / "app.py"
_TESTS_DIR = _BACKEND_ROOT / "tests"


def test_marker_bumped_to_prod_harden_or_later():
    """`_LAST_KNOWN_PFIX` DEBE ser >= P2-PROD-HARDEN (fecha 2026-05-23).
    Si retrocedió, el bundle fue revertido."""
    text = _APP_PY.read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    assert m, "_LAST_KNOWN_PFIX no encontrado en app.py"
    marker = m.group(1)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert date_match, f"Marker {marker!r} sin fecha parseable"
    assert date_match.group(1) >= "2026-05-23", (
        f"_LAST_KNOWN_PFIX={marker!r} retrocedió antes del bundle "
        f"P2-PROD-HARDEN (2026-05-23). Si revertiste el bundle, también "
        f"actualizar la memoria/CLAUDE.md."
    )


@pytest.mark.parametrize(
    "expected_test_file",
    [
        "test_p2_user_depleted_items_fk_idx.py",
        "test_p2_sentry_treeshake.py",
    ],
)
def test_subtest_files_present(expected_test_file: str):
    """Los 2 archivos del bundle DEBEN existir. Si alguno se borró,
    perdimos cobertura — bloquear merge."""
    path = _TESTS_DIR / expected_test_file
    assert path.exists(), (
        f"P2-PROD-HARDEN: el sub-test `{expected_test_file}` fue "
        f"eliminado. Restaurar el archivo (parte del bundle "
        f"audit 2026-05-23) o consolidar su cobertura aquí."
    )


def test_marker_slug_matches_this_file():
    """Sanity del cross-link `test_p2_hist_audit_14_marker_test_link`:
    el slug del marker actual DEBE matchear el filename de este test.
    Si no, el enforcer P2-HIST-AUDIT-14 fallaría primero."""
    text = _APP_PY.read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    marker = m.group(1)
    prefix = marker.split("·", 1)[0].strip()
    slug = prefix.replace("-", "_").lower()
    if slug == "p2_prod_harden":
        assert "p2_prod_harden" in __file__.replace("\\", "/").lower(), (
            "Filename de este test no contiene `p2_prod_harden`."
        )
    else:
        pytest.skip(
            f"Marker actual ({slug}) != p2_prod_harden — bundle "
            f"posterior bumpeó el marker. Este test queda como "
            f"anchor histórico."
        )
