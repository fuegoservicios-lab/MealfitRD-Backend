"""[P2-NEVERA-BRANDS · 2026-07-06] La Nevera dice la marca comprada.

Pedido del owner: "en la nevera también debería decir las marcas". Flujo:
la lista de compras lleva `brand_product_id` (P1-BRAND-DEFAULT-PRESELECTED) →
el restock lo envía → `/restock` resuelve id→brand (1 SELECT, fail-open) →
`add_or_update_inventory_item(..., brand=)` lo persiste en `user_inventory.brand`
(última compra gana, NULL no borra) → GET /inventory lo expone → Pantry pinta
el chip junto a la unidad. Ítems añadidos a mano: brand NULL, sin chip.
"""
import os
import re

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FRONTEND = os.path.join(os.path.dirname(_BACKEND), "frontend")


def _read(*parts) -> str:
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


_DBI = _read(_BACKEND, "db_inventory.py")
_PL = _read(_BACKEND, "routers", "plans.py")
_UD = _read(_BACKEND, "routers", "user_data.py")


# ───────────── migración SSOT en ambos dirs ─────────────

def test_migration_in_both_dirs():
    for base in (os.path.join(_BACKEND, "migrations"),
                 os.path.join(os.path.dirname(_BACKEND), "migrations")):
        p = os.path.join(base, "p2_nevera_brands_2026_07_06.sql")
        assert os.path.exists(p), f"migración ausente en {base} (P3-MIGRATIONS-SSOT)"
        sql = _read(p)
        assert "ADD COLUMN IF NOT EXISTS brand" in sql, "idempotente obligatorio"
        assert "RAISE EXCEPTION" in sql, "sanity DO block (P3-MIGRATION-IDEMPOTENCE-DOC)"


# ───────────── persistencia ─────────────

def test_upsert_accepts_and_persists_brand():
    assert re.search(r"def add_or_update_inventory_item\([^)]*brand:\s*str\s*=\s*None", _DBI)
    i = _DBI.index("INSERT INTO user_inventory")
    win = _DBI[i:i + 900]
    assert "brand" in win.split("VALUES")[0], "columna brand en el INSERT"
    assert "brand = COALESCE(EXCLUDED.brand, user_inventory.brand)" in win, (
        "última compra gana; NULL no borra la marca previa"
    )


def test_rpc_merge_path_refreshes_brand():
    assert "if updated and brand:" in _DBI, (
        "compra repetida cae al path RPC (merge qty) — sin el refresh la marca "
        "solo se escribiría en el primer restock de la vida del ítem"
    )


def test_restock_structured_path_passes_brand():
    i = _DBI.index("def restock_inventory(")
    win = _DBI[i:i + 5200]
    assert 'item.get("brand")' in win and "brand=_item_brand" in win


# ───────────── endpoint /restock resuelve id→brand ─────────────

def test_restock_endpoint_resolves_brand_product_id():
    i = _PL.index('@router.post("/restock")')
    win = _PL[i:i + 16000]
    assert "brand_product_id" in win, "los items estructurados traen el producto comprado"
    assert "FROM public.supermarket_products" in win, "resolución id→brand en 1 SELECT"
    assert "fail-open" in win.lower(), "cualquier error → restock sin marcas (jamás bloquea)"


# ───────────── lectura + UI ─────────────

def test_inventory_select_exposes_brand():
    assert "ui.brand," in _UD, "GET /inventory debe exponer la marca para Pantry/Dashboard"


def test_pantry_shows_brand_chip_both_layouts():
    pantry = _read(_FRONTEND, "src", "pages", "Pantry.jsx")
    assert "P2-NEVERA-BRANDS" in pantry
    # [P1-PANTRY-DASH-PARITY · 2026-07-11] el chip pasó a ser EDITABLE (select
    # disfrazado, title "Marca: X — tocar para cambiar") en ambos layouts.
    assert pantry.count("tocar para cambiar") >= 2, "chip en renderRow (desktop) Y renderMobileCard"
    assert pantry.count("if (!_brands.length && !item.brand) return null;") >= 2, \
        "sin marca Y sin marcas disponibles → sin chip"
    # [polish 2026-07-06] identidad visual propia: índigo+Tag (≠ chip de unidad
    # que usa el color de zona); "Genérico" apagado — las marcas reales resaltan.
    assert "brandChip" in pantry and "brandChipGeneric" in pantry
    for css in ("Pantry.fridge.module.css", "Pantry.mobileFridge.module.css"):
        sheet = _read(_FRONTEND, "src", "pages", css)
        assert ".brandChip" in sheet and ".brandChipGeneric" in sheet, f"clases del chip en {css}"


def test_dashboard_restock_payload_carries_product_id():
    dash = _read(_FRONTEND, "src", "pages", "Dashboard.jsx")
    # OJO: hay DOS "const sourceIngredients" en Dashboard (el del PDF primero)
    # — anclar en el builder del RESTOCK (activeShoppingList.map).
    i = dash.index("activeShoppingList.map(ing")
    win = dash[i:i + 1600]
    assert "brand_product_id" in win, "el restock envía el producto que la lista usó"
