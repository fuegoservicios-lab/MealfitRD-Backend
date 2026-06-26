"""[P1-CATALOG-EXPANSION-BATCH1 · 2026-06-26] Inserta el LOTE 1 de 40 alimentos nuevos al catálogo
(master_ingredients, Neon): 10 proteína vegana + 12 vegetales + 6 frutas + 5 lácteos vegetales + 7 proteína
animal económica/premium. Cierra los huecos de variedad del audit (vegano + planes de 30 días).

NUTRICIÓN: 100% USDA FoodData Central, curada + verificada adversarialmente (Atwater 0 flags). Vive en
`scripts/data/new_foods_batch1_2026_06_26.json` (SSOT del dato). Convención del catálogo respetada:
leguminosas/semillas en estado SECO, vegetales/frutas/carnes CRUDOS, leches vegetales sin azúcar per 100 g.

PRECIOS: NO incluidos. TÚ los llenas en el bloque PRICES de abajo con valores reales del mercado RD
(La Sirena / Nacional / colmado). El script NO inserta un alimento sin precio (gate anti-precio-0, para no
contaminar el cálculo de la lista de compras). Llena price_per_lb O price_per_unit según el `default_unit`
de cada uno (lo imprime `--print-template`).

⚠️ WIRING PENDIENTE (verificar ANTES de usar en planes veganos): las 5 "Leche de…/Yogur de coco" tienen
category='Lácteos' por display pero son VEGETALES. El filtro de dieta vegana NO debe excluirlas por contener
"leche". Confirmar en el guard de dieta (P1-DIET-HARD-GUARD) que las trata como plant-based. Igual, Tofu lo
removiste en P3-TOFU-REMOVE — al reinsertarlo confirma que no quede en una denylist.

USO:
  python scripts/add_foods_batch1_2026_06_26.py --print-template   # imprime el esqueleto de PRICES a llenar
  python scripts/add_foods_batch1_2026_06_26.py                    # DRY-RUN (muestra priced vs skip)
  python scripts/add_foods_batch1_2026_06_26.py --commit           # inserta los priced + inexistentes
"""
import os
import sys
import json
import datetime

try:
    from dotenv import load_dotenv
    for _p in (os.path.join(os.path.dirname(__file__), "..", ".env"),
               os.path.join(os.getcwd(), ".env"), "/opt/mealfit/backend/.env"):
        if os.path.exists(_p):
            load_dotenv(_p)
            break
except Exception:
    pass

import psycopg
from psycopg.types.json import Jsonb

_NEON = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")
COMMIT = "--commit" in sys.argv
PRINT_TEMPLATE = "--print-template" in sys.argv


def _load_records():
    for p in (os.path.join(os.path.dirname(__file__), "data", "new_foods_batch1_2026_06_26.json"),
              os.path.join(os.getcwd(), "scripts", "data", "new_foods_batch1_2026_06_26.json"),
              "/tmp/new_foods_batch1_2026_06_26.json"):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    print("FATAL: no se encontró new_foods_batch1_2026_06_26.json (scp el JSON junto al script)")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# PRECIOS — LLENA AQUÍ (mercado RD). Para cada alimento, pon price_per_lb O price_per_unit según su
# default_unit (corre --print-template para ver cuál). Deja en None lo que aún no tengas → NO se inserta.
# Opcional: market_packages = [{"unit": "...", "grams": N, "label": "...", "price": N}] para el costeo por
# envase (P1-PKG-DURATION-PRICING). Si lo dejas None, el costeo usa price_per_lb/unit simple.
# ─────────────────────────────────────────────────────────────────────────────────────────────────
PRICES = {
    # --- Proteína vegana — PRECIOS RD (La Sirena/Nacional, 2026-06-26) vía "packages" (cost-óptimo) ---
    "Tofu firme":            {"packages": [{"unit": "lata", "grams": 539, "label": "19 oz", "price": 250}]},
    "Soya texturizada":      {"packages": [{"unit": "paquete", "grams": 200, "label": "200 g", "price": 100}]},
    "Edamame":               {"packages": [{"unit": "paquete", "grams": 500, "label": "500 g", "price": 195}]},
    "Guisantes secos":       {"packages": [{"unit": "lata", "grams": 425, "label": "15 oz", "price": 125}]},
    "Frijoles pintos":       {"packages": [{"unit": "paquete", "grams": 800, "label": "800 g", "price": 127}]},
    "Habas":                 {"packages": [{"unit": "paquete", "grams": 454, "label": "16 oz", "price": 205}]},
    "Merey":                 {"packages": [{"unit": "tarro", "grams": 113, "label": "4 oz", "price": 255},
                                           {"unit": "tarro", "grams": 198, "label": "7 oz", "price": 385},
                                           {"unit": "paquete", "grams": 411, "label": "14.5 oz", "price": 689}]},
    "Semillas de calabaza":  {"packages": [{"unit": "tarro", "grams": 227, "label": "8 oz", "price": 305}]},
    "Semillas de girasol":   {"packages": [{"unit": "paquete", "grams": 400, "label": "400 g", "price": 145}]},
    "Nueces mixtas":         {"packages": [{"unit": "paquete", "grams": 100, "label": "100 g", "price": 95}]},
    # --- Vegetales — PRECIOS RD (La Sirena/Nacional, 2026-06-26) ---
    "Champiñones":           {"packages": [{"unit": "paquete", "grams": 227, "label": "8 oz", "price": 205}]},
    "Remolacha":             {"price_per_lb": 45},
    "Apio":                  {"price_per_lb": 49},
    "Acelga":                {"price_per_unit": None, "price_per_lb": None, "market_packages": None},   # FLAG: el producto del "28g/149" es NORI (alga marina), NO acelga — decisión pendiente
    "Berro":                 {"packages": [{"unit": "paquete", "grams": 100, "label": "paquete (~100 g est.)", "price": 44}]},  # Inahnsa; gramos estimados (empaque no los imprime)
    "Rúcula":                {"packages": [{"unit": "paquete", "grams": 57, "label": "2 oz", "price": 43},
                                           {"unit": "paquete", "grams": 227, "label": "8 oz", "price": 130}]},
    "Calabacín":             {"price_per_lb": 49},
    "Kale":                  {"packages": [{"unit": "paquete", "grams": 170, "label": "6 oz", "price": 205}]},
    "Repollo morado":        {"price_per_lb": 59},
    "Rábano":                {"packages": [{"unit": "paquete", "grams": 200, "label": "8 unid (~200 g est.)", "price": 85}]},  # Lucas Pérez; ~8 rábanos, gramos estimados
    "Espárragos":            {"packages": [{"unit": "paquete", "grams": 450, "label": "450 g", "price": 405}]},
    "Coles de Bruselas":     {"packages": [{"unit": "paquete", "grams": 900, "label": "900 g", "price": 220}]},
    "Nori":                  {"packages": [{"unit": "paquete", "grams": 28, "label": "28 g (10 hojas)", "price": 149}]},  # alga marina (reemplaza el slot acelga)
    # --- Frutas — PRECIOS RD por libra (La Sirena/Nacional, 2026-06-26) ---
    "Guayaba":               {"price_per_lb": 48},
    "Guanábana":             {"price_per_lb": 54},
    "Níspero":               {"price_per_lb": 59},
    "Mandarina":             {"price_per_lb": 109},
    "Toronja":               {"price_per_lb": 140},
    "Uva":                   {"price_per_lb": 169},
    "Coco":                  {"packages": [{"unit": "unidad", "grams": 300, "label": "1 coco (~300g pulpa)", "price": 69}]},  # alimento nuevo (coco entero)
    # --- Lácteos vegetales — PRECIOS RD (La Sirena/Nacional, 2026-06-26) ---
    "Leche de almendras":    {"packages": [{"unit": "carton", "grams": 946, "label": "32 oz", "price": 260}]},
    "Leche de coco":         {"price_per_unit": None, "price_per_lb": None, "market_packages": None},   # FLAG: "Coco unidad@69" es COCO ENTERO (fruta), no leche de coco (bebida)
    "Leche de soya":         {"price_per_unit": None, "price_per_lb": None, "market_packages": None},   # FLAG: sin precio en la tabla
    "Leche de avena":        {"packages": [{"unit": "carton", "grams": 1000, "label": "1 L", "price": 124}]},
    "Yogur de coco":         {"packages": [{"unit": "pote", "grams": 170, "label": "6 oz regular", "price": 50},
                                           {"unit": "pote", "grams": 227, "label": "8 oz griego", "price": 95}]},
    # --- Proteína animal económica/premium — PRECIOS RD (La Sirena/Nacional, 2026-06-26) ---
    "Sardinas en lata":      {"packages": [{"unit": "lata", "grams": 125, "label": "125 g", "price": 33},
                                           {"unit": "lata", "grams": 425, "label": "15 oz", "price": 57}]},
    "Muslo de pollo":        {"price_per_lb": 68},
    "Hígado de res":         {"price_per_lb": 119},
    "Salmón":                {"packages": [{"unit": "paquete", "grams": 85, "label": "3 oz", "price": 490},
                                           {"unit": "paquete", "grams": 323, "label": "11.4 oz", "price": 1060}]},
    "Tilapia":               {"price_per_lb": 130},
    "Pavo molido":           {"packages": [{"unit": "paquete", "grams": 454, "label": "16 oz", "price": 320}]},
    "Mero":                  {"price_per_lb": 290},
}

# record-key → columna DB
_COLMAP = {
    "kcal": "kcal_per_100g", "protein_g": "protein_g_per_100g", "carbs_g": "carbs_g_per_100g",
    "fats_g": "fats_g_per_100g", "fiber_g": "fiber_g_per_100g", "sugars_g": "sugars_g_per_100g",
    "satfat_g": "saturated_fat_g_per_100g", "sodium_mg": "sodium_mg_per_100g",
    "cholesterol_mg": "cholesterol_mg_per_100g", "calcium_mg": "calcium_mg_per_100g",
    "iron_mg": "iron_mg_per_100g", "potassium_mg": "potassium_mg_per_100g",
    "magnesium_mg": "magnesium_mg_per_100g", "phosphorus_mg": "phosphorus_mg_per_100g",
    "zinc_mg": "zinc_mg_per_100g", "vit_d_mcg": "vitamin_d_mcg_per_100g",
    "b12_mcg": "vitamin_b12_mcg_per_100g", "folate_mcg_dfe": "folate_mcg_dfe_per_100g",
    "vit_a_mcg_rae": "vitamin_a_mcg_rae_per_100g", "vit_c_mg": "vitamin_c_mg_per_100g",
    "vit_e_mg": "vitamin_e_mg_per_100g", "vit_k_mcg": "vitamin_k_mcg_per_100g",
    "selenium_mcg": "selenium_mcg_per_100g", "omega3_ala_g": "omega3_ala_g_per_100g",
}


def _is_priced(p):
    if not p:
        return False
    return bool(p.get("packages")) or bool(p.get("price_per_lb")) or bool(p.get("price_per_unit"))


def _derive_price_fields(p):
    """De un PRICES entry deriva los campos DB de precio/envase.
    Con "packages" [{unit,grams,label,price}]: market_packages (cost-óptimo P1-PKG) + price_per_unit (envase
    más pequeño = default de compra) + price_per_lb (mejor $/g × 453.6, tarifa a granel) + container_weight_g/
    market_container/available_sizes_g. Garantiza price_per_unit>0 → pasa el gate de verificados (shopping
    _verified_ingredients). Sin packages: usa los precios planos."""
    pkgs = p.get("packages") or []
    if pkgs:
        pkgs_sorted = sorted(pkgs, key=lambda x: x["grams"])
        smallest = pkgs_sorted[0]
        best_per_g = min(x["price"] / x["grams"] for x in pkgs)
        return {
            "market_packages": Jsonb(pkgs),
            "available_sizes_g": Jsonb(sorted({int(round(x["grams"])) for x in pkgs})),
            "price_per_unit": smallest["price"],
            "price_per_lb": round(best_per_g * 453.592, 2),
            "container_weight_g": smallest["grams"],
            "market_container": smallest.get("unit"),
        }
    return {"price_per_lb": p.get("price_per_lb") or 0, "price_per_unit": p.get("price_per_unit") or 0}


def main():
    recs = _load_records()
    if PRINT_TEMPLATE:
        print("# Esqueleto PRICES (copia, pega en PRICES y llena el campo según default_unit):")
        for r in recs:
            du = r["default_unit"]
            campo = "price_per_lb" if du == "lb" else "price_per_unit"
            print(f'    "{r["name"]}": {{"{campo}": ___, "market_packages": None}},   # default_unit={du}')
        return
    if not _NEON:
        print("FATAL: NEON url ausente"); sys.exit(1)

    today = datetime.date.today()
    inserted = skipped_exist = skipped_price = 0
    with psycopg.connect(_NEON) as conn:
        existing = {row[0] for row in conn.execute("SELECT name FROM public.master_ingredients").fetchall()}
        for r in recs:
            nm = r["name"]
            if nm in existing:
                print(f"  ~ EXISTE, salto: {nm}")
                skipped_exist += 1
                continue
            price = PRICES.get(nm) or {}
            if not _is_priced(price):
                print(f"  ⏳ SIN PRECIO, salto: {nm} (llena PRICES['{nm}'])")
                skipped_price += 1
                continue
            cols = {
                "slug": r["slug"], "name": nm, "category": r["category"],
                "aliases": r.get("aliases") or [], "default_unit": r["default_unit"],
                "is_dominican_cultivar": bool(r.get("is_dominican_cultivar")),
                "density_g_per_cup": r.get("density_g_per_cup"),
                "density_g_per_unit": r.get("density_g_per_unit"),
                "nutrition_source": "usda", "nutrition_source_date": today,
                "fdc_id": r.get("fdc_id"),
            }
            cols.update(_derive_price_fields(price))  # price_per_lb/unit + market_packages + container + sizes
            for k, dbcol in _COLMAP.items():
                cols[dbcol] = r.get(k)
            colnames = list(cols.keys())
            placeholders = ", ".join(["%s"] * len(colnames))
            vals = [cols[c] for c in colnames]
            if COMMIT:
                conn.execute(
                    f"INSERT INTO public.master_ingredients ({', '.join(colnames)}) VALUES ({placeholders})",
                    vals,
                )
            print(f"  {'+ INSERTADO' if COMMIT else '+ (dry) insertaría'}: {nm} [{r['category']}] "
                  f"({r['kcal']}kcal/{r['protein_g']}P) unit={cols.get('price_per_unit')} lb={cols.get('price_per_lb')} "
                  f"pkgs={len(price.get('packages') or [])}")
            inserted += 1
        if COMMIT:
            conn.commit()
            print(f"\nCOMMITTED. insertados={inserted}, ya-existen={skipped_exist}, sin-precio={skipped_price}")
        else:
            print(f"\nDRY-RUN. insertaría={inserted}, ya-existen={skipped_exist}, sin-precio={skipped_price}. "
                  f"Llena PRICES y re-corre con --commit.")


if __name__ == "__main__":
    main()
