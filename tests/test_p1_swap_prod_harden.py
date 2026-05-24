"""[P1-SWAP-PROD-HARDEN · 2026-05-22] Test umbrella del bundle 3-en-1
que cerró el audit production-readiness del flujo "Cambiar Plato":

  - **GAP-1 (P1-SWAP-EMPTY-PANTRY-FALLBACK):** `agent.py::swap_meal` lee
    `aggregated_shopping_list` del plan_data como fallback cuando la
    pantry virtual quedó vacía + el frontend no envió pantry list. Cumple
    el requisito explícito del owner ("si la nevera está vacía debe
    tomar en cuenta la lista de compras pdf"). Test:
    [`test_p1_swap_empty_pantry_fallback.py`](test_p1_swap_empty_pantry_fallback.py).

  - **GAP-2 (P1-SWAP-PERSIST-ATOMIC):** `/swap-meal/persist` migrado de
    `execute_sql_write` + `jsonb_set` chained (sin row lock) a
    `update_plan_data_atomic(plan_id, _mutator, user_id=...)` con
    `SELECT … FOR UPDATE`. Cierra estructuralmente la ventana
    lost-update con `_chunk_worker` concurrente. Test:
    [`test_p1_swap_persist_atomic.py`](test_p1_swap_persist_atomic.py).

  - **GAP-3 (P1-SWAP-RECIPE-COHERENCE dedicado):** test unitario+parser
    del validador `validate_meal_recipe_ingredients_coherence` y su
    wiring en los dos surfaces (swap_meal + execute_modify_single_meal).
    Pre-fix la cobertura era indirecta dentro de `test_p1_swap_macros.py`
    Section G. Test:
    [`test_p1_swap_recipe_coherence.py`](test_p1_swap_recipe_coherence.py).

Este file existe principalmente para satisfacer el contrato
`test_p2_hist_audit_14_marker_test_link.py` (slug del marker actual
DEBE matchear al menos un archivo `tests/test_<slug>*.py`). Los 3
sub-tests viven en archivos separados (un archivo por gap, mejor para
diff revisión + fail isolation en CI). Aquí solo verificamos que los
3 archivos existen y que el marker está bumpeado correctamente.

Tooltip-anchor: P1-SWAP-PROD-HARDEN | bundle audit 2026-05-22
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND_ROOT / "app.py"
_TESTS_DIR = _BACKEND_ROOT / "tests"


def test_marker_bumped_to_prod_harden_or_later():
    """`_LAST_KNOWN_PFIX` DEBE ser >= `P1-SWAP-PROD-HARDEN · 2026-05-22`.
    Si retrocedió, el bundle fue revertido."""
    text = _APP_PY.read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    assert m, "_LAST_KNOWN_PFIX no encontrado en app.py"
    marker = m.group(1)
    # Aceptamos cualquier marker con fecha >= 2026-05-22; el bump
    # posterior por otros P-fixes no debe romper este test.
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert date_match, f"Marker {marker!r} sin fecha parseable"
    assert date_match.group(1) >= "2026-05-22", (
        f"_LAST_KNOWN_PFIX={marker!r} retrocedió antes del bundle "
        f"P1-SWAP-PROD-HARDEN. Si revertiste el bundle, también "
        f"actualizar la memoria/CLAUDE.md."
    )


@pytest.mark.parametrize(
    "expected_test_file",
    [
        "test_p1_swap_empty_pantry_fallback.py",
        "test_p1_swap_persist_atomic.py",
        "test_p1_swap_recipe_coherence.py",
    ],
)
def test_subtest_files_present(expected_test_file: str):
    """Los 3 archivos del bundle DEBEN existir. Si alguno se borró,
    perdimos cobertura — bloquear merge."""
    path = _TESTS_DIR / expected_test_file
    assert path.exists(), (
        f"P1-SWAP-PROD-HARDEN: el sub-test `{expected_test_file}` "
        f"fue eliminado. Restaurar el archivo (parte del bundle "
        f"audit 2026-05-22) o consolidar su cobertura aquí."
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
    # Este file DEBE matchear el patrón `test_<slug>*.py` para satisfacer
    # P2-HIST-AUDIT-14. Permitimos que el slug actual sea posterior
    # (cuando otro P-fix bumpea el marker, este test sigue existiendo
    # para historial pero ya no es el primary anchor).
    if slug == "p1_swap_prod_harden":
        assert "p1_swap_prod_harden" in __file__.replace("\\", "/").lower(), (
            "Filename de este test no contiene `p1_swap_prod_harden`."
        )
    else:
        # Marker bumpeado por otro P-fix — este test queda como historial
        # del bundle. No es un fallo.
        pytest.skip(
            f"Marker actual ({slug}) != p1_swap_prod_harden — bundle "
            f"posterior bumpeó el marker. Este test queda como "
            f"anchor histórico."
        )
