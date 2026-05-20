"""[P1-DASHBOARD-CACHE-INVENTORY · 2026-05-20] Test anti-regresión del
read del cache singleton de Pantry en Dashboard.

Bug observado:
    Dashboard.jsx tenía `useState(null)` para `liveInventory` y
    `useState(true)` para `isLoadingInventory` → cada navegación
    Plan/Agente → Dashboard mostraba spinner ~500-1500ms antes de que
    el fetch resolviera. El cache singleton `pantryCache.js` (P3-PANTRY-CACHE)
    YA almacenaba el inventory tras cada visita a Nevera PERO Dashboard
    solo guardaba (setCachedInventory) sin LEER al mount.

    Reportado 2026-05-20 como continuación de "el apartado de plan es el
    único que tiene eso también" — "Plan" en la sidebar es `/dashboard`.

Fix:
    Importar `getCachedInventory` y usar como initial state del useState.
    Si cache fresco (<10min tras P1-PANTRY-TTL-BUMP), arranca con datos
    → cero flash. Si no, fallback a null + fetch normal.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DASHBOARD_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_dashboard_imports_get_cached_inventory():
    """[P1-DASHBOARD-CACHE-INVENTORY] Import explícito de `getCachedInventory`
    desde `pantryCache.js`. Sin esto, el initializer no puede leer cache."""
    src = _read(_DASHBOARD_JSX)
    assert "getCachedInventory" in src, (
        "Import `getCachedInventory` ausente. Sin él, Dashboard no puede "
        "hidratar inventory desde el cache singleton. Ver "
        "P1-DASHBOARD-CACHE-INVENTORY · 2026-05-20."
    )
    # Sanity: el import debe venir del SSOT pantryCache.
    assert re.search(
        r"import\s*\{[^}]*getCachedInventory[^}]*\}\s*from\s*['\"][^'\"]*pantryCache",
        src,
    ), "getCachedInventory debe importarse desde utils/pantryCache (SSOT)."


def test_live_inventory_initializer_reads_cache():
    """[P1-DASHBOARD-CACHE-INVENTORY] `liveInventory` useState debe inicializar
    leyendo el cache antes del fallback a null. Anti-pattern bloqueado:
    `useState(null)` literal sin lookup previo."""
    src = _read(_DASHBOARD_JSX)
    # Buscar la línea del useState liveInventory.
    match = re.search(
        r"useState\((_cachedInv\s*\|\|\s*null|getCachedInventory\(\s*\)\s*\|\|\s*null)\)",
        src,
    )
    assert match, (
        "`useState(_cachedInv || null)` ausente — el initial state no usa "
        "el cache. Ver P1-DASHBOARD-CACHE-INVENTORY · 2026-05-20."
    )


def test_loading_inventory_false_if_cache_hit():
    """[P1-DASHBOARD-CACHE-INVENTORY] `isLoadingInventory` debe inicializar
    en `false` si hay cache hit, para evitar spinner cuando ya hay datos."""
    src = _read(_DASHBOARD_JSX)
    # Buscar `useState(!_cachedInv)` o equivalente lógico.
    match = re.search(
        r"useState\(!\s*_cachedInv\)",
        src,
    )
    assert match, (
        "`useState(!_cachedInv)` ausente — isLoadingInventory siempre arranca "
        "en true y el spinner aparece aunque haya cache. Ver "
        "P1-DASHBOARD-CACHE-INVENTORY · 2026-05-20."
    )
