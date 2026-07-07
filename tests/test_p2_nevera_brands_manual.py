"""[P2-NEVERA-BRANDS-MANUAL · 2026-07-07] Elegir la marca al añadir a mano.

Pedido del owner: "quiero que se pueda elegir la marca de los alimentos".
Antes solo el restock de la lista de compras traía marca (brand_product_id →
supermarket_products → user_inventory.brand). El add MANUAL de la Nevera
(POST /api/inventory/items) no la aceptaba.

Flujo nuevo:
  Pantry add-picker abre item nuevo → consulta POST /api/supermarket/match por
  ese alimento → arma chips de marcas distintas (con precio mínimo) → el usuario
  elige una (o "Sin marca") → handleAddNewItem la manda en el body → el endpoint
  la persiste en user_inventory.brand → GET /inventory la expone → el chip de la
  Nevera (P2-NEVERA-BRANDS) la pinta.

Parser-based: el selector es UI + un endpoint que ya existe (match); correr
Playwright no aporta más señal que verificar los anclajes de contrato abajo.
"""
from __future__ import annotations

import os
import re

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FRONTEND = os.path.join(os.path.dirname(_BACKEND), "frontend")


def _read(*parts) -> str:
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


_UD = _read(_BACKEND, "routers", "user_data.py")
_PANTRY = _read(_FRONTEND, "src", "pages", "Pantry.jsx")


# ─────────────────────────── backend ───────────────────────────

def test_inventory_item_body_accepts_brand():
    # El modelo del POST /api/inventory/items debe declarar `brand: Optional[str]`.
    assert re.search(r"class InventoryItemBody\(BaseModel\)", _UD)
    body_block = _UD[_UD.index("class InventoryItemBody"):]
    body_block = body_block[: body_block.index("\n\n\n")] if "\n\n\n" in body_block else body_block[:600]
    assert re.search(r"brand\s*:\s*Optional\[str\]\s*=\s*None", body_block), (
        "InventoryItemBody debe aceptar `brand: Optional[str] = None` — sin esto "
        "el add manual no puede llevar la marca elegida en el picker."
    )


def test_insert_persists_and_returns_brand():
    i = _UD.index("INSERT INTO user_inventory")
    win = _UD[i:i + 1200]
    cols = win.split("VALUES")[0]
    assert "brand" in cols, "la columna `brand` debe estar en el INSERT del add manual"
    # RETURNING/proyección debe exponer ins.brand para que el chip se pinte al instante.
    assert "ins.brand" in _UD[i:i + 2500], (
        "el SELECT sobre `ins` debe proyectar `ins.brand` para devolver la marca "
        "recién insertada al frontend (chip inmediato sin refetch)."
    )
    # El valor de brand se trimea a NULL y se pasa como parámetro del INSERT.
    assert "_brand = (body.brand" in _UD, (
        "brand debe extraerse/trimear de body.brand (→ NULL si vacío)."
    )
    assert "_brand)" in _UD, (
        "el valor `_brand` debe pasarse como último parámetro del INSERT."
    )


# ─────────────────────────── frontend ───────────────────────────

def test_pantry_marker_present():
    assert "P2-NEVERA-BRANDS-MANUAL" in _PANTRY


def test_pantry_has_brand_picker_state():
    # State de la marca elegida en el picker.
    assert re.search(r"const\s*\[\s*pickerBrand\s*,\s*setPickerBrand\s*\]\s*=\s*useState\(", _PANTRY), (
        "State `pickerBrand` ausente — es la marca seleccionada antes de confirmar."
    )
    assert re.search(r"const\s*\[\s*brandCache\s*,\s*setBrandCache\s*\]\s*=\s*useState\(", _PANTRY), (
        "State `brandCache` ausente — cachea las variantes del súper por alimento."
    )


def test_pantry_loads_brands_from_supermarket_match():
    # El picker consulta el catálogo del súper por variantes del alimento.
    assert "_loadBrandsForItem" in _PANTRY, "helper `_loadBrandsForItem` ausente."
    assert "/api/supermarket/match" in _PANTRY, (
        "el picker debe consultar POST /api/supermarket/match para traer las "
        "marcas reales del Supermercado RD."
    )


def test_pantry_sends_brand_in_add_body():
    # handleAddNewItem debe incluir `brand: pickerBrand` en el body del POST a
    # inventory/items (patrón distintivo — búsqueda global robusta).
    assert re.search(r"brand\s*:\s*pickerBrand", _PANTRY), (
        "el body del POST /api/inventory/items debe llevar `brand: pickerBrand` "
        "(|| null) para persistir la marca elegida."
    )


def test_pantry_renders_brand_selector():
    # La sección de marca en el picker + opción explícita "Sin marca".
    assert "Sin marca" in _PANTRY, "opción por defecto 'Sin marca' ausente del selector."
    assert "brandPillStyle" in _PANTRY, "estilo de pill de marca ausente."
    # Solo se ofrece para items nuevos (no existentes) — mismo gate que las unit pills.
    assert re.search(r"brandInfo\s*=\s*\(isPickerOpen\s*&&\s*!existing\)", _PANTRY), (
        "el selector de marca debe gatearse a items nuevos (isPickerOpen && !existing)."
    )
