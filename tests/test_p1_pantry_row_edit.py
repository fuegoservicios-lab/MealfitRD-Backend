"""[P1-PANTRY-ROW-EDIT · 2026-07-11] Cantidad directa + marcas por fila en el paso 21.

Feedback del owner: "si yo quiero escribir 200 gramos lo más rápido posible no podría
— tendría que darle al signo de '+' 200 veces" + "quiero que también se pueda
modificar las marcas de los alimentos".

Contrato:
1. PATCH /api/inventory/items/{id} — set ABSOLUTO de quantity (clamp [0, 9999]) y/o
   brand ('' limpia a NULL, ausente no toca; máx 60 chars). 422 sin campos; 404 row
   ajeno/inexistente; I2 user_id.
2. QPantryBuilder: <input type="number"> por fila con borrador local y commit en
   blur/Enter (PATCH absoluto — los +/- siguen siendo delta rápido); <select> de
   marca por fila con fetch LAZY de /api/supermarket/match (mismo contrato que la
   página Nevera: marcas reales con precio, cache por alimento, fail-soft).

tooltip-anchor: P1-PANTRY-ROW-EDIT
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_UD_SRC = (_BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8")
_QPB_SRC = (_BACKEND.parent / "frontend" / "src" / "components" / "assessment"
            / "questions" / "QPantryBuilder.jsx").read_text(encoding="utf-8")


def _patch_endpoint_body():
    i = _UD_SRC.find('@router.patch("/inventory/items/{item_id}")\n')
    assert i > 0, "PATCH genérico de item desapareció"
    return _UD_SRC[i:i + 3000]


def test_patch_sets_absolute_quantity_with_clamp():
    body = _patch_endpoint_body()
    assert "max(0.0, min(9999.0" in body, "quantity absoluta con clamp [0, 9999]"
    assert "quantity = %s::numeric" in body


def test_patch_brand_empty_clears_to_null():
    body = _patch_endpoint_body()
    assert '.strip()[:60] or None' in body, (
        "brand '' debe volverse NULL (Genérico) y truncarse a 60 chars"
    )


def test_patch_requires_some_field_and_filters_user():
    body = _patch_endpoint_body()
    assert "422" in body, "PATCH vacío → 422"
    assert "id = %s AND user_id = %s" in body, "I2"
    assert "404" in body


def test_frontend_qty_input_commits_absolute():
    assert 'type="number"' in _QPB_SRC and "qtyDrafts" in _QPB_SRC, (
        "input numérico por fila desapareció — el owner no puede escribir 200 "
        "sin darle 200 veces al +"
    )
    assert "onBlur={() => commitQty(item)}" in _QPB_SRC
    assert "JSON.stringify({ quantity: q })" in _QPB_SRC, "commit = PATCH absoluto (no delta)"


def test_frontend_brand_select_lazy_supermarket():
    assert "'/api/supermarket/match'" in _QPB_SRC, (
        "las marcas salen del Supermercado RD (mismo contrato que la página Nevera)"
    )
    assert "onFocus={() => loadBrandsFor(item)}" in _QPB_SRC, (
        "fetch LAZY al abrir el select — N fetches en el mount penalizan el paso"
    )
    assert 'Genérico (sin marca)' in _QPB_SRC


def test_marker_anchored_in_source():
    assert _UD_SRC.count("P1-PANTRY-ROW-EDIT") >= 1
    assert _QPB_SRC.count("P1-PANTRY-ROW-EDIT") >= 2
