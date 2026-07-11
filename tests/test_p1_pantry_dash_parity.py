"""[P1-PANTRY-DASH-PARITY · 2026-07-11] La Nevera del dashboard tiene paridad con el
paso 21 del wizard: escáner por foto + marcas editables por fila.

Pedido del owner: "quiero que en la nevera del dashboard también se puedan agregar
alimentos mediante fotos como en la nevera del formulario, y también modificar las
marcas de dichos alimentos".

Contrato:
1. SSOT del escáner: el flujo completo (botón → reescala → POST photo-scan →
   checklist → adds con 409→increment) vive en `PantryScanButton.jsx` y AMBAS
   superficies lo consumen — el literal '/api/inventory/photo-scan' no existe en
   ningún otro archivo frontend (anti-drift).
2. Flag: GET /pantry-status expone `photo_scan_enabled` (la página Nevera ya lo
   consulta y no tiene los campos biométricos que exige /pantry-feasibility).
3. Marcas en la Nevera: prefetch en LOTE (un POST /supermarket/match), chip-select
   por fila en AMBOS layouts (desktop + móvil) oculto sin marcas disponibles,
   `changeItemBrand` = PATCH del item + preferencia GLOBAL (PUT /supermarket/
   preferences con productId; Genérico limpia ambas).

tooltip-anchor: P1-PANTRY-DASH-PARITY
"""
from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend" / "src"

_SCAN_SRC = (_FRONT / "components" / "pantry" / "PantryScanButton.jsx").read_text(encoding="utf-8")
_PANTRY_SRC = (_FRONT / "pages" / "Pantry.jsx").read_text(encoding="utf-8")
_QPB_SRC = (_FRONT / "components" / "assessment" / "questions" / "QPantryBuilder.jsx").read_text(encoding="utf-8")
_PLANS_SRC = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Escáner compartido (SSOT)
# ---------------------------------------------------------------------------

def test_scan_flow_is_single_source():
    assert "'/api/inventory/photo-scan'" in _SCAN_SRC
    offenders = []
    for f in _FRONT.rglob("*.jsx"):
        if f.name == "PantryScanButton.jsx":
            continue
        if "/api/inventory/photo-scan" in f.read_text(encoding="utf-8"):
            offenders.append(f.name)
    assert not offenders, (
        f"el flujo del scan debe vivir SOLO en PantryScanButton.jsx (drift): {offenders}"
    )


def test_both_surfaces_use_shared_component():
    assert "<PantryScanButton" in _QPB_SRC, "el paso 21 debe usar el componente compartido"
    assert "<PantryScanButton" in _PANTRY_SRC, "la página Nevera debe usar el componente compartido"
    assert "pantryStatus?.photo_scan_enabled" in _PANTRY_SRC, (
        "la Nevera gatea por el flag de /pantry-status (sin provider → sin botón)"
    )


def test_pantry_status_exposes_scan_flag():
    i = _PLANS_SRC.find('@router.get("/pantry-status")')
    body = _PLANS_SRC[i:i + 4000]
    assert body.count('"photo_scan_enabled"') >= 3, (
        "las 3 ramas de /pantry-status (guest, ok, error) deben exponer el flag"
    )


# ---------------------------------------------------------------------------
# 2. Marcas editables en la Nevera
# ---------------------------------------------------------------------------

def test_pantry_brand_change_patches_item_and_preference():
    i = _PANTRY_SRC.find("const changeItemBrand")
    assert i > 0, "changeItemBrand desapareció"
    body = _PANTRY_SRC[i:i + 2500]
    assert "`/api/inventory/items/${item.id}`" in body and "'PATCH'" in body.replace('"', "'")
    assert "'/api/supermarket/preferences'" in body, (
        "la elección manual persiste la preferencia GLOBAL (mismo sistema 'para "
        "siempre' que Marcas del súper / paso 21)"
    )
    assert "entry?.productId ? entry.productId : null" in body, (
        "Genérico o marca sin producto → product_id null (borra la preferencia)"
    )


def test_pantry_brand_batch_prefetch():
    assert "JSON.stringify({ names: _names })" in _PANTRY_SRC, (
        "las marcas de las filas se prefetchean en LOTE (un POST), no N requests"
    )


def test_brand_select_in_both_layouts_hidden_when_empty():
    assert _PANTRY_SRC.count("if (!_brands.length && !item.brand) return null;") >= 2, (
        "desktop Y móvil: sin marcas disponibles y sin marca puesta → sin selector "
        "(un menú con solo Genérico confunde — mismo contrato que el paso 21)"
    )
    # [P1-BRAND-SELECT-UI] el select nativo se reemplazó por el dropdown propio.
    assert _PANTRY_SRC.count("onSelect={(b) => changeItemBrand(item, b)}") >= 2


def test_marker_anchored_in_source():
    assert _SCAN_SRC.count("P1-PANTRY-DASH-PARITY") >= 1
    assert _PANTRY_SRC.count("P1-PANTRY-DASH-PARITY") >= 3
    assert _PLANS_SRC.count("P1-PANTRY-DASH-PARITY") >= 1
