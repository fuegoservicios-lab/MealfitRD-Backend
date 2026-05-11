"""[P3-4 · 2026-05-10] Regression guard: Pantry.jsx muestra chip de
shelf-life con helpers `getShelfLifeBadge` + `getShelfLifeBadgeStyle`.

Bug raíz (audit 2026-05-10):
    `backend/db_inventory.py:_infer_shelf_life_days` + `_format_inventory_for_llm`
    computan urgency client-side (`Caduca en X días` / URGENTE) y lo
    embeben en el texto que envía al LLM, pero esa info NO llegaba al
    frontend. La UI de Pantry mostraba items sin diferenciar urgency,
    aunque el sistema sabía cuáles priorizar.

Fix:
    - Nuevo helper `frontend/src/utils/shelfLife.js` con
      `getShelfLifeBadge(item)` que computa days_left client-side desde
      `created_at` + `master_ingredients.shelf_life_days`.
    - Query del fetch (initial + realtime) extendido con `shelf_life_days`
      en la projection de `master_ingredients`.
    - Chip render condicional en `Pantry.jsx` después del consumption chip.
    - Solo se muestra para items con `days_left ≤ 3` (urgency real).

Cobertura de este test (parser-based, no DB ni JS runtime):
    1. Helper existe en utils/shelfLife.js.
    2. Helper retorna null cuando data falta (fallback semántico conservador).
    3. Pantry.jsx importa y usa el helper.
    4. Pantry query incluye `shelf_life_days` en master_ingredients.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHELF_LIFE_JS = _REPO_ROOT / "frontend" / "src" / "utils" / "shelfLife.js"
_PANTRY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Pantry.jsx"


def _read(path: Path) -> str:
    assert path.exists(), f"Archivo no encontrado: {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Helper utility existe + exporta las 2 funciones canónicas
# ---------------------------------------------------------------------------
def test_shelf_life_helper_exists():
    assert _SHELF_LIFE_JS.exists(), (
        f"Helper `shelfLife.js` debe existir en {_SHELF_LIFE_JS}. "
        "Si fue movido, actualizar este test."
    )


def test_helper_exports_get_shelf_life_badge():
    src = _read(_SHELF_LIFE_JS)
    assert re.search(
        r"export\s+function\s+getShelfLifeBadge\s*\(",
        src,
    ), (
        "Helper debe exportar `getShelfLifeBadge(item)` — la función "
        "principal que computa days_left + severity."
    )
    assert re.search(
        r"export\s+function\s+getShelfLifeBadgeStyle\s*\(",
        src,
    ), (
        "Helper debe exportar `getShelfLifeBadgeStyle(severity)` para "
        "que múltiples sites usen la misma palette."
    )


def test_helper_returns_null_on_missing_data():
    """El helper debe retornar `null` (no `undefined`, no objeto con
    `daysLeft: NaN`) cuando los datos faltan. Esto evita renders
    de chips con valores corruptos."""
    src = _read(_SHELF_LIFE_JS)
    # Heurística: el helper debe contener varios `return null` para los
    # paths de validación (item null, created_at falta, shelf_life_days
    # missing, daysLeft > 3).
    null_returns = len(re.findall(r"\breturn\s+null\b", src))
    assert null_returns >= 4, (
        f"`getShelfLifeBadge` debe tener ≥4 paths que retornan `null` "
        f"(validación de input), encontrados: {null_returns}. Sin estos "
        f"early returns, items sin shelf_life_days renderían chips con "
        f"`NaN días`."
    )


def test_helper_has_three_severity_buckets():
    """Las 3 severity (`expired`, `urgent`, `warn`) deben estar en el
    helper — alineado con backend `db_inventory.py:404-406`."""
    src = _read(_SHELF_LIFE_JS)
    for severity in ("expired", "urgent", "warn"):
        assert f"'{severity}'" in src or f'"{severity}"' in src, (
            f"Helper debe definir el bucket `{severity}` (alineado con "
            f"backend semántica de urgency)."
        )


# ---------------------------------------------------------------------------
# 2. Pantry.jsx usa el helper
# ---------------------------------------------------------------------------
def test_pantry_imports_helper():
    src = _read(_PANTRY_JSX)
    assert re.search(
        r"import\s+\{[^}]*getShelfLifeBadge[^}]*\}\s+from\s+['\"][^'\"]*shelfLife['\"]",
        src,
    ), (
        "Pantry.jsx debe importar `getShelfLifeBadge` desde "
        "`../utils/shelfLife`."
    )


def test_pantry_renders_chip_with_helper():
    src = _read(_PANTRY_JSX)
    # Debe haber una llamada a getShelfLifeBadge(item) y un return de span/chip.
    assert "getShelfLifeBadge(item)" in src, (
        "Pantry.jsx debe invocar `getShelfLifeBadge(item)` en el render "
        "del item card."
    )


def test_pantry_query_includes_shelf_life_days():
    """Sin `shelf_life_days` en la projection de master_ingredients, el
    helper devuelve null para todos los items (sin chip). El test guarda
    contra regresiones del query."""
    src = _read(_PANTRY_JSX)
    # Buscar la projection de master_ingredients en select(...).
    # El patrón canónico: `master_ingredients(name, category, default_unit, shelf_life_days)`.
    assert "shelf_life_days" in src, (
        "Pantry.jsx query a user_inventory debe pedir `shelf_life_days` "
        "en la projection de `master_ingredients(...)`. Sin esto, el "
        "helper recibe `undefined` y retorna null → ningún chip."
    )


def test_pantry_renders_only_when_urgent():
    """El chip solo aparece para items con urgency (days_left ≤ 3). El
    helper retorna `null` para items frescos, y el render hace early-
    return."""
    src = _read(_PANTRY_JSX)
    # Buscar el patrón `if (!badge) return null` cerca de la invocación
    # del helper.
    pattern = re.search(
        r"getShelfLifeBadge\(item\)[\s\S]{0,200}if\s*\(\s*!badge\s*\)\s*return\s+null",
        src,
    )
    assert pattern is not None, (
        "El render del chip debe hacer early-return `if (!badge) return null` "
        "antes de pintar — sin esto, un null/undefined del helper crashea "
        "el componente o pinta un chip vacío."
    )
