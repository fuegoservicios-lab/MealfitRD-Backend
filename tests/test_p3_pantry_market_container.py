"""[P3-PANTRY-MARKET-CONTAINER · 2026-05-19] Coherencia entre la unidad
mostrada en la Nevera (frontend) y la que aparece en el PDF/lista de
compras (backend).

**Bug original (feedback usuario 2026-05-19):**
    Item "Leche" en la Nevera mostraba `Paquete` pero el PDF mostraba
    `Cartón` para el mismo item. Mismatch de UX confunde al usuario y
    deteriora la sensación de que la nevera y la lista de compras son
    el mismo sistema coherente.

**Causa raíz:**
    El frontend leía `master_ingredients.default_unit` (valor genérico
    en la DB) mientras el backend del PDF leía `master_ingredients.market_container`
    (valor curado para mercado dominicano). Para Leche eran respectivamente
    `'paquete'` y `'cartón'`. Las dos columnas en la misma tabla con valores
    distintos generaban el mismatch.

**Fix:**
    Frontend ahora prioriza `master_ingredients.market_container` en 4 lugares:
    1. **Display** del item en `renderItemCard`: `displayUnit = market_container || item.unit`
    2. **Default** del `handleAddNewItem` al insertar: `finalUnit = market_container || default_unit || 'unidad'`
    3. **Default** del picker (3 call sites de `setPickerUnit`)
    4. **Lista de chips** del picker: incluye `market_container` como primera opción
    Además: `COMMON_PURCHASE_UNITS` añade `'cartón'` (no estaba).
    Query Supabase del JOIN `user_inventory` extiende el SELECT para incluir
    `market_container` (los 5 callsites).

**Por qué display-only override y no migración DB:**
    Items existentes en `user_inventory` tienen `unit='paquete'` persistido.
    Cambiar la DB requeriría migración cross-user y cualquier item donde el
    user genuinamente eligió "paquete" se perdería. El display layer lee
    `market_container` del master (single source of truth), así que TODOS los
    items de leche muestran "Cartón" sin tocar lo persistido. Si en el futuro
    queremos persistir el cambio, basta con una migración SQL.

**Coherencia con canonical_units.py:**
    El backend en `canonical_units.py:39` canonicaliza `'cartón' → 'paquete'`
    para shopping calculator. Eso es por SSOT de agregación (sumar quantities
    cross-recipe). El display al usuario sigue siendo `market_container`
    via `apply_smart_market_units()` (backend) o `displayUnit` (frontend).
    Este test NO toca esa canonicalización — solo el display.

Por qué parser-based:
    El bug es de display/data plumbing entre 2 capas. Anclamos las anclas
    estructurales con regex sobre el source:
    1. Marker presente
    2. SELECT del JOIN `user_inventory` incluye `market_container`
    3. `displayUnit` se computa con `master_ingredients?.market_container` fallback
    4. `'cartón'` aparece en `COMMON_PURCHASE_UNITS`
    5. `finalUnit` en `handleAddNewItem` prioriza `market_container`
    6. `setPickerUnit` calls priorizan `market_container`
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PANTRY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Pantry.jsx"
# [P1-NEON-DB-MIGRATION · 2026-06-12] El transporte del inventario migró de
# PostgREST `.select('*, master_ingredients(...)')` (frontend) a los endpoints
# backend GET /api/inventory + GET /api/catalog, cuyo SELECT vive en
# routers/user_data.py. La proyección de `market_container` que antes estaba
# en el frontend ahora está en el SQL del backend.
_USER_DATA_PY = _REPO_ROOT / "backend" / "routers" / "user_data.py"


def _read_pantry() -> str:
    assert _PANTRY_JSX.exists(), f"Pantry.jsx no encontrado en {_PANTRY_JSX}"
    return _PANTRY_JSX.read_text(encoding="utf-8")


def _read_user_data() -> str:
    assert _USER_DATA_PY.exists(), f"user_data.py no encontrado en {_USER_DATA_PY}"
    return _USER_DATA_PY.read_text(encoding="utf-8")


def test_p3_pantry_market_container_marker_present():
    """Marker textual presente — cross-link con `test_p2_hist_audit_14`."""
    src = _read_pantry()
    assert "P3-PANTRY-MARKET-CONTAINER" in src, (
        "Marker `P3-PANTRY-MARKET-CONTAINER` no encontrado en Pantry.jsx. "
        "Si reviertes la coherencia de market_container, bumpea "
        "`_LAST_KNOWN_PFIX` al P-fix de reversión y elimina este test."
    )


def test_user_inventory_select_includes_market_container():
    """El SELECT del JOIN `user_inventory` ↔ `master_ingredients` DEBE incluir
    `market_container` en la proyección. Sin esto, el endpoint no devuelve ese
    campo y el display layer (frontend) no puede leerlo (cae a `item.unit` que
    es el genérico).

    [P1-NEON-DB-MIGRATION · 2026-06-12] Pre-migración esta proyección vivía en
    los `.select('*, master_ingredients(...)')` PostgREST del frontend
    (Pantry.jsx, 5 callsites). El cutover a Neon movió el transporte a los
    endpoints backend GET /api/inventory + GET /api/catalog, cuyo SELECT vive
    en routers/user_data.py. Ahora anclamos la proyección en el backend (SSOT
    real del dato): cada `jsonb_build_object(... 'market_container', ...)` que
    embebe `master_ingredients` en el JOIN con `user_inventory` debe incluir el
    campo, más la query del catálogo.
    """
    src = _read_user_data()

    # Proyecciones que embeben master_ingredients en el JOIN (GET /api/inventory
    # + POST insert-and-return). Forma: jsonb_build_object(... mi.market_container ...).
    join_projections = re.findall(
        r"jsonb_build_object\((.*?)\)\s*END\s+AS\s+master_ingredients",
        src,
        re.DOTALL,
    )
    assert len(join_projections) >= 2, (
        f"Esperaba ≥2 proyecciones jsonb_build_object(...) AS master_ingredients "
        f"sobre el JOIN user_inventory↔master_ingredients en user_data.py, "
        f"encontré {len(join_projections)}."
    )
    for fields in join_projections:
        assert "market_container" in fields, (
            f"Proyección del JOIN master_ingredients no incluye "
            f"`market_container`. Sin él, el endpoint no devuelve el dato y el "
            f"frontend cae a `item.unit` (genérico). Block: {fields!r}"
        )

    # La query del catálogo (GET /api/catalog) también debe proyectar
    # market_container desde master_ingredients.
    assert re.search(
        r"SELECT[\s\S]*?market_container[\s\S]*?FROM\s+master_ingredients",
        src,
    ), (
        "La query del catálogo (FROM master_ingredients) no proyecta "
        "`market_container`. El picker de unidades no ofrecería el contenedor "
        "curado dominicano."
    )


def test_render_item_card_uses_market_container_as_display():
    """El render del item DEBE computar `displayUnit` priorizando
    `master_ingredients?.market_container` sobre `item.unit`.
    """
    src = _read_pantry()
    assert re.search(
        r"const\s+displayUnit\s*=\s*item\.master_ingredients\?\.market_container\s*\|\|\s*item\.unit",
        src,
    ), (
        "El cálculo `const displayUnit = item.master_ingredients?.market_container "
        "|| item.unit` no se encuentra en `renderItemCard`. Sin él, el item "
        "muestra `item.unit` (que para items viejos es `paquete` aunque el "
        "PDF diga `cartón`)."
    )

    # El render del unit-tag debe usar displayUnit, no item.unit
    unit_tag_match = re.search(
        r'<span\s+className="nevera-item-unit-tag"[^>]*>\{(\w+)\}</span>',
        src,
    )
    assert unit_tag_match, (
        "No se encontró el `<span className='nevera-item-unit-tag'>{...}</span>` "
        "que renderiza la unit del item."
    )
    assert unit_tag_match.group(1) == "displayUnit", (
        f"El unit-tag renderiza `{{{unit_tag_match.group(1)}}}` en lugar de "
        f"`{{displayUnit}}`. Reemplaza por displayUnit."
    )


def test_common_purchase_units_includes_carton():
    """`COMMON_PURCHASE_UNITS` DEBE incluir `'cartón'` (con tilde) como
    chip seleccionable en el picker. Pre-fix la lista tenía 'paquete'
    pero no 'cartón' — leche dominicana viene en cartón.
    """
    src = _read_pantry()
    match = re.search(
        r"const\s+COMMON_PURCHASE_UNITS\s*=\s*\[([^\]]+)\]",
        src,
    )
    assert match, "Constante `COMMON_PURCHASE_UNITS` no encontrada"
    block = match.group(1)
    assert "'cartón'" in block, (
        "`COMMON_PURCHASE_UNITS` no incluye `'cartón'`. Sin él, el picker "
        "no ofrece cartón como opción seleccionable. Lista actual: " + block
    )


def test_handle_add_new_item_prioritizes_market_container():
    """`handleAddNewItem` DEBE priorizar `masterItem.market_container`
    sobre `masterItem.default_unit` al insertar nuevo item.
    """
    src = _read_pantry()
    # Match relajado (paréntesis anidados en JS dificultan regex estricto):
    # buscamos que en el block de `finalUnit = ...` aparezca
    # `masterItem.market_container || masterItem.default_unit` (en ese orden).
    finalunit_match = re.search(
        r"const\s+finalUnit\s*=([^;]+);",
        src,
        re.DOTALL,
    )
    assert finalunit_match, "Declaración `const finalUnit = ...;` no encontrada."
    finalunit_block = finalunit_match.group(1)
    assert re.search(
        r"masterItem\.market_container\s*\|\|\s*masterItem\.default_unit",
        finalunit_block,
    ), (
        "`finalUnit` no prioriza `masterItem.market_container || masterItem.default_unit`. "
        f"Block actual: {finalunit_block!r}"
    )


def test_picker_default_prioritizes_market_container():
    """Los `setPickerUnit(...)` calls en el modal de añadir item DEBEN
    priorizar `market_container`. Hay al menos 2 callsites (uno en
    keyboard handler, otro en click handler del item).
    """
    src = _read_pantry()
    matches = re.findall(
        r"setPickerUnit\(([^)]+)\)",
        src,
    )
    assert len(matches) >= 2, (
        f"Esperaba ≥2 callsites de `setPickerUnit(...)`, encontré "
        f"{len(matches)}: {matches}"
    )

    # Al menos 2 de los callsites deben referenciar market_container
    market_container_count = sum(
        1 for m in matches if "market_container" in m
    )
    assert market_container_count >= 2, (
        f"Solo {market_container_count}/{len(matches)} callsites de "
        f"setPickerUnit priorizan `market_container`. Patrón esperado: "
        f"`targetItem.market_container || targetItem.default_unit || 'unidad'`. "
        f"Calls actuales: {matches}"
    )
