"""[P1-BRAND-DEFAULT-GUARDS · 2026-07-06] Guards del default de marca.

Verificación read-only contra el plan vivo ff673061 (48 ítems) tras desplegar
P1-BRAND-LIST-VISIBILITY expuso 4 modos de fallo del overlay default:
1. Perejil fresco (mazo RD$44) → frasco de perejil MOLIDO Badia RD$215; cilantro
   → SEMILLAS de cilantro. Hierbas frescas jamás reciben overlay.
2. Venta-por-libra: "Guayaba 1 libra RD$48" (necesidad ¼ lb = RD$12), "Chivo 1
   libra RD$299" (¼ lb = RD$75), "Coliflor 2 libra" (antes "2 Cabezas"). Los
   productos "Lb" del mostrador fresco quedan FUERA del default (la preferencia
   manual sí los respeta — elección explícita).
3. "Yogurt" agarraba "Yogurt de cabra" (frascos 4 Oz → 8 × RD$110 = RD$880 vs
   pote 1.96 kg RD$220). La contención nombre⊆food se elimina en defaults; solo
   exacto/singular/food⊆nombre.
4. "Maní" agarraba la variante "Con Pasas". Modificadores ajenos al nombre del
   ítem quedan fuera; "Semillas de girasol" sí admite "semillas".
"""
import pytest

import shopping_calculator as sc


@pytest.fixture(autouse=True)
def _fresh_defaults_cache():
    sc._brand_defaults_cache["data"] = None
    sc._brand_defaults_cache["at"] = 0.0
    yield
    sc._brand_defaults_cache["data"] = None
    sc._brand_defaults_cache["at"] = 0.0


# Master hermético: arroz SE VENDE en envase (recibe default); perejil y queso
# blanco NO tienen envase en master (mazo / deli fraccional → jamás default).
_MASTER = [
    {"name": "Arroz blanco", "category": "Despensa", "market_container": "paquete",
     "container_weight_g": 907.0, "price_per_lb": 40.0, "default_unit": "paquete",
     "shelf_life_days": 365, "aliases": []},
    {"name": "Perejil", "category": "Vegetales", "market_container": None,
     "container_weight_g": None, "price_per_lb": 44.0, "default_unit": "mazo",
     "shelf_life_days": 5, "aliases": []},
    {"name": "Queso blanco", "category": "Lácteos", "market_container": None,
     "container_weight_g": None, "price_per_lb": 270.0, "default_unit": "lb",
     "shelf_life_days": 14, "aliases": []},
]


@pytest.fixture()
def master_stub(monkeypatch):
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: list(_MASTER))
    sc.invalidate_master_cache()
    yield
    sc.invalidate_master_cache()


# ─────────────────── per-libra fuera del default ───────────────────

def test_bare_lb_product_marked_per_lb():
    pkg = sc._pkg_from_product_row({
        "presentation": "Criolla Lb", "brand": None, "price_rd": 62.0, "size_grams": None,
    })
    assert pkg is not None and pkg.get("per_lb") is True


def test_explicit_size_not_per_lb():
    for pres in ("Funda 800 gr", "Paquete 2 Lb", "Paquete 2L", "Estelar 1.47 Lb"):
        pkg = sc._pkg_from_product_row({
            "presentation": pres, "brand": "X", "price_rd": 100.0, "size_grams": None,
        })
        assert pkg is not None and not pkg.get("per_lb"), f"{pres} tiene tamaño explícito"


def test_defaults_fetch_excludes_per_lb(monkeypatch):
    rows = [
        {"food_name": "Chivo", "brand": None, "presentation": "Fresco Lb",
         "price_rd": 299.0, "size_grams": None},
        {"food_name": "Arroz blanco", "brand": "Wala", "presentation": "Funda Selecto 1 Lb",
         "price_rd": 42.0, "size_grams": None},
    ]
    monkeypatch.setattr(sc, "execute_sql_query", lambda *a, **k: rows)
    out = sc.fetch_brand_default_packages()
    assert "chivo" not in out, "venta-por-libra (mostrador fresco) fuera del default"
    assert "arroz blanco" in out, "tamaño explícito ('1 Lb' con número) sí participa"


def test_prefs_still_accept_per_lb(monkeypatch):
    """La preferencia MANUAL respeta venta-por-libra (comportamiento pre-existente)."""
    rows = [{"food_key": "chivo", "food_name": "Chivo", "brand": None,
             "presentation": "Fresco Lb", "price_rd": 299.0, "size_grams": None}]
    monkeypatch.setattr(sc, "execute_sql_query", lambda *a, **k: rows)
    out = sc.fetch_brand_pref_packages("11111111-2222-3333-4444-555555555555")
    assert "chivo" in out, "la elección explícita del usuario no se filtra"


# ─────────────────── hierbas frescas jamás ───────────────────

def test_herbs_never_get_default():
    defaults = {"perejil": [{"grams": 141.7, "price": 215.0, "label": "Molido 5 Oz · Badia", "unit": "frasco"}]}
    assert sc._resolve_brand_default("Perejil", defaults) is None
    assert sc._resolve_brand_default("Cilantro", {"cilantro": defaults["perejil"]}) is None


# ─────────────────── dirección del matching ───────────────────

def test_item_generic_does_not_match_specific_food():
    defaults = {"yogurt de cabra natural": [
        {"grams": 113.4, "price": 110.0, "label": "4 Oz · Deliciel", "unit": "frasco"},
    ]}
    assert sc._resolve_brand_default("Yogurt", defaults) is None, (
        "nombre⊆food prohibido en defaults — 'Yogurt' no puede agarrar el de cabra (8×RD$110 vs pote RD$220)"
    )


def test_specific_item_matches_generic_food():
    defaults = {"arroz blanco": [
        {"grams": 907.0, "price": 145.0, "label": "2 Lb · Cariño", "unit": "paquete"},
    ]}
    got = sc._resolve_brand_default("Arroz Blanco Premium", defaults)
    assert got and got[0]["label"].endswith("Cariño"), "food⊆nombre sigue permitido"
    assert sc._resolve_brand_default("Arroz blanco", defaults), "exacto sigue permitido"


# ─────────────────── modificadores ajenos ───────────────────

def test_foreign_modifier_variant_dropped():
    defaults = {"mani": [
        {"grams": 55.0, "price": 48.0, "label": "Con Pasas 55 gr · Cashitas", "unit": "funda"},
        {"grams": 300.0, "price": 185.0, "label": "300 gr · Cashitas", "unit": "pote"},
    ]}
    got = sc._resolve_brand_default("Maní", defaults)
    assert got is not None and len(got) == 1 and "Pasas" not in got[0]["label"], (
        "la variante 'Con Pasas' no es maní a secas — fuera del default"
    )


def test_all_variants_foreign_means_no_overlay():
    defaults = {"mani": [
        {"grams": 55.0, "price": 48.0, "label": "Con Pasas 55 gr · Cashitas", "unit": "funda"},
    ]}
    assert sc._resolve_brand_default("Maní", defaults) is None


def test_english_salted_variant_dropped():
    """El catálogo mezcla descriptores en inglés: "Roasted Salted" = "con sal"."""
    defaults = {"mani": [
        {"grams": 150.0, "price": 159.0, "label": "Roasted Salted 150 gr · Nut Walker", "unit": "lata"},
        {"grams": 300.0, "price": 185.0, "label": "300 gr · Cashitas", "unit": "pote"},
    ]}
    got = sc._resolve_brand_default("Maní", defaults)
    assert got is not None and len(got) == 1 and "Salted" not in got[0]["label"]


def test_modifier_allowed_when_in_item_name():
    defaults = {"semillas de girasol": [
        {"grams": 400.0, "price": 145.0, "label": "Semillas 400 gr · Genérico", "unit": "paquete"},
    ]}
    got = sc._resolve_brand_default("Semillas de girasol", defaults)
    assert got, "'semillas' está en el nombre del ítem → la variante es legítima"


# ─────────────────── label polish "2 L" → "2 lb" ───────────────────

def test_solid_L_label_normalized_to_lb():
    pkg = sc._pkg_from_product_row({
        "presentation": "Paquete 2L", "brand": None, "price_rd": 165.0, "size_grams": None,
    })
    assert pkg is not None
    assert pkg["label"] == "2 lb · Genérico", (
        f"la L-libra no puede mostrarse como litros: {pkg['label']}"
    )


# ─────────────────── E2E: los guards en el aggregator ───────────────────

def test_overlay_uses_conservative_resolver_e2e(master_stub):
    defaults = {
        "perejil": [{"grams": 141.7, "price": 215.0, "label": "Molido 5 Oz · Badia", "unit": "frasco"}],
        "arroz blanco": [{"grams": 907.0, "price": 145.0, "label": "2 lb · Cariño", "unit": "paquete"}],
    }
    result = sc.aggregate_and_deduct_shopping_list(
        ["800g de arroz blanco", "20g de perejil"], [], structured=True, brand_defaults=defaults,
    )
    arroz = next((r for r in result if isinstance(r, dict) and "arroz" in str(r.get("name", "")).lower()), None)
    perejil = next((r for r in result if isinstance(r, dict) and "perejil" in str(r.get("name", "")).lower()), None)
    assert arroz is not None and "Cariño" in str(arroz.get("display_qty", ""))
    if perejil is not None:
        assert "Badia" not in str(perejil.get("display_qty", "")), (
            f"el perejil fresco jamás se costea como frasco molido: {perejil.get('display_qty')}"
        )


def test_no_container_master_never_gets_default(master_stub):
    """Deli/mostrador fraccional (master sin envase): 'Queso blanco ¼ lb RD$68'
    jamás se convierte en 'paquete 1 lb RD$270' aunque el catálogo tenga la
    variante empacada (gate master-en-envase, verificado vs plan vivo)."""
    defaults = {"queso blanco": [
        {"grams": 453.592, "price": 270.0, "label": "1 lb · Genérico", "unit": "paquete"},
    ]}
    result = sc.aggregate_and_deduct_shopping_list(
        ["100g de queso blanco"], [], structured=True, brand_defaults=defaults,
    )
    queso = next((r for r in result if isinstance(r, dict) and "queso" in str(r.get("name", "")).lower()), None)
    assert queso is not None
    disp = str(queso.get("display_qty", ""))
    assert "Genérico" not in disp and "·" not in disp, (
        f"master sin envase (venta fraccional) no recibe default: {disp}"
    )
