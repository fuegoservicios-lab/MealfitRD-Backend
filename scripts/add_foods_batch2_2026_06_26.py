"""[P1-CATALOG-EXPANSION-BATCH2 · 2026-06-26] Inserta el LOTE 2 de 43 alimentos nuevos al catálogo
(master_ingredients, Neon) para llevarlo de 157 → 200: 8 vegetales + 8 frutas + 5 granos + 6 especias/hierbas
+ 3 semillas/grasas + 8 proteína animal + 4 lácteos + 1 víver. Cierra los huecos de variedad restantes
(granos delgados, especias para apetecibilidad, proteína animal por presupuesto, frutas/vegetales para 30 días).
Vegetales: Puerro, Culantro, Bok choy, Lechuga romana, Nabo, Alcachofa, Palmito, Cebollín
(Berza→Culantro/recao [Ají gustoso se descartó por redundante con "Ají cubanela"]; Maíz tierno descartado =
dup de "Maíz dulce en granos" → Lechuga romana).

NUTRICIÓN: 100% USDA FoodData Central, curada + verificada adversarialmente (workflow batch2). Vive en
`scripts/data/new_foods_batch2_2026_06_26.json` (SSOT del dato). Convención del catálogo respetada:
granos/cereales SECOS, vegetales/frutas/carnes/mariscos CRUDOS, especias secas per 100 g (uso ~1-5 g),
lácteos as-sold.

PRECIOS: NO incluidos. TÚ los llenas en el bloque PRICES de abajo con valores reales del mercado RD
(La Sirena / Nacional / colmado). El script NO inserta un alimento sin precio (gate anti-precio-0, para no
contaminar el cálculo de la lista de compras). Llena price_per_lb O price_per_unit según el `default_unit`
de cada uno (lo imprime `--print-template`), o usa "packages" para costeo por envase (P1-PKG-DURATION-PRICING).

USO:
  python scripts/add_foods_batch2_2026_06_26.py --print-template   # imprime el esqueleto de PRICES a llenar
  python scripts/add_foods_batch2_2026_06_26.py                    # DRY-RUN (muestra priced vs skip)
  python scripts/add_foods_batch2_2026_06_26.py --commit           # inserta los priced + inexistentes
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

_JSON_NAME = "new_foods_batch2_2026_06_26.json"


def _load_records():
    for p in (os.path.join(os.path.dirname(__file__), "data", _JSON_NAME),
              os.path.join(os.getcwd(), "scripts", "data", _JSON_NAME),
              "/tmp/" + _JSON_NAME):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    print(f"FATAL: no se encontró {_JSON_NAME} (scp el JSON junto al script)")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# PRECIOS — LLENA AQUÍ (mercado RD). Para cada alimento, pon price_per_lb O price_per_unit según su
# default_unit (corre --print-template para ver cuál). Deja en None lo que aún no tengas → NO se inserta.
# Opcional: "packages": [{"unit": "...", "grams": N, "label": "...", "price": N}] para el costeo por envase
# (P1-PKG-DURATION-PRICING). Si lo dejas None, el costeo usa price_per_lb/unit simple.
#   Ejemplo bulk:    "Mapuey": {"price_per_lb": 45},
#   Ejemplo envase:  "Bulgur": {"packages": [{"unit": "paquete", "grams": 454, "label": "16 oz", "price": 95}]},
# ─────────────────────────────────────────────────────────────────────────────────────────────────
PRICES = {
    # --- Vegetales ---
    "Puerro":                {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Culantro":              {"price_per_unit": None, "price_per_lb": None, "market_packages": None},   # reemplaza Berza (Ají gustoso era redundante con cubanela) → Culantro/recao
    "Bok choy":              {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Lechuga romana":        {"price_per_unit": None, "price_per_lb": None, "market_packages": None},   # reemplaza Maíz tierno (dup de "Maíz dulce en granos")
    "Nabo":                  {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Alcachofa":             {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Palmito":               {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Cebollín":              {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    # --- Frutas ---
    "Pera":                  {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Kiwi":                  {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Durazno":               {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Ciruela":               {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Arándanos":             {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Tamarindo":             {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Acerola":               {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Granada":               {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    # --- Granos / carbos (Despensa) ---
    "Bulgur":                {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Cuscús":                {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Cebada":                {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Harina de avena":       {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Pan de agua":           {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    # --- Especias / hierbas (Despensa) — uso ~1-5 g; suele venderse por frasco/funda ---
    "Comino":                {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Cúrcuma":               {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Laurel":                {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Tomillo":               {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Curry en polvo":        {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Cebolla en polvo":      {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    # --- Semillas / grasas (Despensa) ---
    "Ajonjolí":              {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Pistachos":             {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Mantequilla de almendras": {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    # --- Proteína animal económica/premium ---
    "Chuleta de cerdo":      {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Costilla de cerdo":     {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Pechuga de pavo":       {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Conejo":                {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Chivo":                 {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Pulpo":                 {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Calamar":               {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Mejillones":            {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    # --- Lácteos ---
    "Queso cheddar":         {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Queso gouda":           {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Requesón":              {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Kéfir":                 {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    # --- Víveres ---
    "Mapuey":                {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
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
    """De un PRICES entry deriva los campos DB de precio/envase. Con "packages" [{unit,grams,label,price}]:
    market_packages (cost-óptimo P1-PKG) + price_per_unit (envase más pequeño) + price_per_lb (mejor $/g ×
    453.6) + container_weight_g/market_container/available_sizes_g. Garantiza price_per_unit>0 → pasa el gate
    de verificados. Sin packages: usa los precios planos."""
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
            cols.update(_derive_price_fields(price))
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
