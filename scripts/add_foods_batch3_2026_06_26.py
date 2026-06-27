"""[P1-CATALOG-EXPANSION-BATCH3 · 2026-06-26] Inserta el LOTE 3 de 5 alimentos nuevos al catálogo
(master_ingredients, Neon): Cacao en polvo, Dátiles, Pasas, Ciruela pasa, Cangrejo. Llenan huecos de aporte
(magnesio/antioxidantes, endulzantes naturales/fibra, proteína magra de marisco) que el owner pidió, todos
verificados como NO duplicados (nombre+alias) y disponibles en RD.

NUTRICIÓN: 100% USDA FoodData Central, curada + verificada adversarialmente (workflow batch3). Vive en
`scripts/data/new_foods_batch3_2026_06_26.json` (SSOT). Frutas deshidratadas (dátiles/pasas/ciruela pasa) y
cacao en estado SECO per 100 g; cangrejo CRUDO. Aliases de pasas/ciruela-pasa EXCLUYEN "uva"/"ciruela" pelados
para no colisionar con las frescas (Uva/Ciruela ya en catálogo).

PRECIOS: NO incluidos. TÚ los llenas en PRICES (mercado RD). Sin precio NO se inserta (gate anti-precio-0).

USO:
  python scripts/add_foods_batch3_2026_06_26.py --print-template
  python scripts/add_foods_batch3_2026_06_26.py            # DRY-RUN
  python scripts/add_foods_batch3_2026_06_26.py --commit   # inserta los priced + inexistentes
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
_JSON_NAME = "new_foods_batch3_2026_06_26.json"


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
# PRECIOS — LLENA AQUÍ (mercado RD). price_per_lb O price_per_unit según default_unit (ver --print-template),
# o "packages": [{"unit","grams","label","price"}]. Deja None lo que no tengas → NO se inserta.
# ─────────────────────────────────────────────────────────────────────────────────────────────────
PRICES = {
    # PRECIOS RD (La Sirena/Nacional, 2026-06-26)
    "Cacao en polvo":        {"packages": [{"unit": "paquete", "grams": 200, "label": "200 g", "price": 130}]},
    "Dátiles":               {"price_per_lb": 340},
    "Pasas":                 {"packages": [{"unit": "paquete", "grams": 250, "label": "250 g", "price": 189}]},
    "Ciruela pasa":          {"packages": [{"unit": "tarro", "grams": 454, "label": "16 oz", "price": 199}]},
    "Cangrejo":              {"price_per_lb": 479},
}

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
        print("# Esqueleto PRICES:")
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
