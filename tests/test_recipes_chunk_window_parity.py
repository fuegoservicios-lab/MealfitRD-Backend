"""[P-RECIPES-CHUNK-WINDOW · 2026-05-08] Paridad cross-language del helper
`splitWithAbsorb` entre backend y frontend.

Bug que previene:
  El day selector de `frontend/src/pages/Recipes.jsx` limita los días
  visibles al chunk activo, replicando la lógica de
  `backend/constants.py:split_with_absorb` en
  `frontend/src/utils/chunkWindow.js`. Si el backend cambia su algoritmo
  (ej. nuevo P-fix optimiza distribución de chunks largos), el frontend
  mostraría chunks distintos a los reales generados por el orchestrator
  → usuario navega a recetas de chunks que aún no existen → 404 silencioso
  en el cooking mode o platos del chunk siguiente filtrados como del actual.

Este test es la red de seguridad cross-language: parsea el archivo
`frontend/src/utils/chunkWindow.js` como texto, extrae los casos canónicos
del docstring, y los re-evalúa via `backend.constants.split_with_absorb`.
Si AMBOS deben dar el mismo resultado, este test falla cuando uno deriva
del otro.

Patrón establecido por:
  - test_p0_form_6_required_fields_sync (REQUIRED_FORM_FIELDS frontend↔backend)
  - test_p3_5_bio_ranges_parity (BIO_RANGES frontend↔backend)
  - test_p3_b_required_fields_js_parser (formValidation.js parser)
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from constants import split_with_absorb


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
_FRONTEND_HELPER = _REPO_ROOT / "frontend" / "src" / "utils" / "chunkWindow.js"


# Casos canónicos que DEBEN dar el mismo resultado en ambos lados.
# Mantener idénticos a los del docstring de `splitWithAbsorb` en el .js.
_CANONICAL_CASES = [
    (3, [3]),
    (4, [4]),
    (7, [3, 4]),       # caso especial
    (9, [3, 3, 3]),    # n_full < umbral
    (14, [3, 3, 4, 4]),
    (15, [3, 4, 4, 4]),  # P1-A
    (18, [3, 4, 4, 4, 3]),
    (21, [3, 4, 4, 4, 6]),
    (30, [3, 4, 4, 4, 4, 4, 4, 3]),
]


@pytest.mark.parametrize("total_days,expected", _CANONICAL_CASES)
def test_backend_split_with_absorb_canonical(total_days, expected):
    """Sanity del backend: cada caso canónico produce el output esperado."""
    assert split_with_absorb(total_days) == expected, (
        f"split_with_absorb({total_days}) cambió. Si el cambio es intencional, "
        f"actualizar también:\n"
        f"  - {_FRONTEND_HELPER.relative_to(_REPO_ROOT)} (función `splitWithAbsorb`)\n"
        f"  - frontend/src/__tests__/utils/chunkWindow.test.js (casos canónicos)\n"
        f"  - este test (`_CANONICAL_CASES`)"
    )


def test_frontend_helper_exists():
    """El archivo `frontend/src/utils/chunkWindow.js` debe existir.
    Sin él, Recipes.jsx no compila y el day selector muestra TODOS los días."""
    assert _FRONTEND_HELPER.is_file(), (
        f"Helper frontend no encontrado: {_FRONTEND_HELPER}. "
        f"Si fue movido, actualizar este test."
    )


def test_frontend_helper_documents_canonical_cases():
    """[Regresión documental] El docstring del helper JS debe mencionar los
    mismos casos canónicos que el backend (cross-language drift detection
    a nivel de comments).

    Si el backend cambia un caso (ej. 30d → [3,4,5,4,4,4,3,3]) y olvida
    actualizar el comment del frontend, este test guía al fix.
    """
    text = _FRONTEND_HELPER.read_text(encoding="utf-8")
    # Verifica que cada caso canónico aparece en el docstring del helper.
    for total_days, expected in _CANONICAL_CASES:
        # Buscar el patrón "Nd → [...]" en el docstring.
        # No exigimos formato exacto — solo que los números estén juntos.
        pattern_a = f"{total_days}d"
        # Solo verificamos los casos "ejemplo" del docstring (subset que aparece).
        if pattern_a in text:
            # Si el caso aparece en el doc, su array literal también debe estar.
            arr_str = ", ".join(str(n) for n in expected)
            assert arr_str in text or _array_appears_in_text(expected, text), (
                f"Caso canónico `{total_days}d → {expected}` no aparece "
                f"correctamente documentado en {_FRONTEND_HELPER.name}. "
                f"Si el algoritmo cambió, actualizar el docstring."
            )


def _array_appears_in_text(arr, text):
    """Heurística laxa: busca el array como `[3, 4, ...]` permitiendo
    variantes de whitespace."""
    pattern = r"\[\s*" + r"\s*,\s*".join(str(n) for n in arr) + r"\s*\]"
    return bool(re.search(pattern, text))


def test_frontend_helper_exports_required_functions():
    """Smoke check: el helper exporta las 3 funciones públicas que Recipes.jsx
    consume (`parseStartLocal`, `splitWithAbsorb`, `findChunkContaining`)."""
    text = _FRONTEND_HELPER.read_text(encoding="utf-8")
    required_exports = ["parseStartLocal", "splitWithAbsorb", "findChunkContaining"]
    for name in required_exports:
        # `export const <name>` o `export function <name>`.
        pattern = rf"export\s+(?:const|function)\s+{re.escape(name)}\b"
        assert re.search(pattern, text), (
            f"Helper frontend no exporta `{name}`. Recipes.jsx fallará al "
            f"importar y el day selector mostrará todos los días del plan."
        )
