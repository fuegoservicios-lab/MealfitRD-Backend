"""[P1-PANTRY-CACHE-LOCALSTORAGE · 2026-05-20] Anti-regresión del cache
híbrido (in-memory + localStorage) en pantryCache.js.

Bug observado:
    El cache era solo in-memory (`let _inventoryEntry = null`). Al page
    reload (F5/Ctrl+R), el módulo se re-evalúa desde cero → cache vacío
    → primer acceso a Nevera/Dashboard fetcha de Supabase ~500-1500ms
    con spinner visible. Reportado 2026-05-20: "cuando refresqué la
    página web y entré a la nevera sentí lo mismo otra vez".

Fix:
    Cache híbrido: in-memory (fast path runtime) + localStorage (sobrevive
    page reload). Write: ambos simultáneamente. Read: in-memory primero;
    fallback a localStorage si no hay in-memory. Invalidación: ambos.

    Privacy aceptable: el inventory de comida NO es PII sensible y ya
    vive en `mealfit_plan` (que contiene ingredientes). El localStorage
    está aislado por origen.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PANTRY_CACHE_JS = _REPO_ROOT / "frontend" / "src" / "utils" / "pantryCache.js"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_export(src: str, name: str) -> str:
    """Extrae el cuerpo de una `export const <name> = (...) => { ... }` con
    parens y braces balanceadas. Regex naive `\\}` corta en el primer brace
    de un object literal interno — esto evita ese bug."""
    pat = rf"export\s+const\s+{name}\s*="
    match = re.search(pat, src)
    if not match:
        return ""
    arrow = src.find("=>", match.end())
    if arrow < 0:
        return ""
    brace = src.find("{", arrow)
    if brace < 0:
        return ""
    depth = 0
    for j in range(brace, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[match.start():j + 1]
    return ""


def test_inventory_cache_persists_to_localstorage():
    """[P1-PANTRY-CACHE-LOCALSTORAGE] `setCachedInventory` debe persistir
    también en localStorage para sobrevivir page reload. Sin esto, refresh
    del browser borra el cache → spinner visible al primer acceso."""
    src = _read(_PANTRY_CACHE_JS)
    # Anchor: setItem de la cache key esperada dentro de setCachedInventory.
    body = _extract_export(src, "setCachedInventory")
    assert body, "setCachedInventory no encontrada"
    assert "localStorage.setItem" in body, (
        "`setCachedInventory` NO escribe a localStorage. Sin esto, el cache "
        "no sobrevive page reload. Ver P1-PANTRY-CACHE-LOCALSTORAGE · 2026-05-20."
    )
    assert "_INVENTORY_LS_KEY" in body or "mealfit_pantry_inventory_cache" in body, (
        "setCachedInventory no usa la cache key esperada."
    )


def test_inventory_cache_reads_localstorage_fallback():
    """[P1-PANTRY-CACHE-LOCALSTORAGE] `getCachedInventory` debe leer de
    localStorage como fallback cuando in-memory está null (post page
    reload). Sin esto, el persist no aporta nada al refresh."""
    src = _read(_PANTRY_CACHE_JS)
    body = _extract_export(src, "getCachedInventory")
    assert body, "getCachedInventory no encontrada"
    assert "localStorage.getItem" in body, (
        "`getCachedInventory` NO lee de localStorage como fallback. Sin "
        "esto, el cache localStorage queda inútil. Ver "
        "P1-PANTRY-CACHE-LOCALSTORAGE · 2026-05-20."
    )
    # Sanity: hidrata _inventoryEntry desde localStorage para siguientes
    # llamadas (fast path).
    assert re.search(r"_inventoryEntry\s*=\s*parsed", body), (
        "Tras leer de localStorage, debe hidratar `_inventoryEntry` para "
        "que próximas llamadas usen fast path. Ver "
        "P1-PANTRY-CACHE-LOCALSTORAGE."
    )


def test_invalidate_also_clears_localstorage():
    """[P1-PANTRY-CACHE-LOCALSTORAGE] `invalidateInventoryCache` debe
    limpiar AMBOS (in-memory + localStorage). Sin esto, queda cache
    stale tras delete/restock."""
    src = _read(_PANTRY_CACHE_JS)
    body = _extract_export(src, "invalidateInventoryCache")
    assert body, "invalidateInventoryCache no encontrada"
    assert re.search(r"_inventoryEntry\s*=\s*null", body), (
        "invalidateInventoryCache no resetea in-memory entry."
    )
    # Debe llamar _safeLsRemove o localStorage.removeItem.
    assert "_safeLsRemove" in body or "localStorage.removeItem" in body, (
        "invalidateInventoryCache no limpia localStorage. Cache stale tras "
        "delete/restock. Ver P1-PANTRY-CACHE-LOCALSTORAGE · 2026-05-20."
    )
