"""[P1-CATALOG-EXPANSION-BATCH2 · 2026-06-26] Inserta el LOTE 2 (~42 alimentos, ajustándose por disponibilidad RD)
al catálogo (master_ingredients, Neon) para llevarlo de 157 → ~199: 8 vegetales + 8 frutas + 3 granos (Cuscús/
Pan de agua descartados por no venderse) + 6 especias/hierbas + 3 semillas/grasas + 9 proteína animal (incl.
Arenque) + 4 lácteos + 1 víver. Cierra los huecos de variedad restantes
(granos delgados, especias para apetecibilidad, proteína animal por presupuesto, frutas/vegetales para 30 días).
Vegetales: Puerro, Cundeamor, Bok choy, Lechuga romana, Nabo, Alcachofa, Palmito, Cebollín
(slot de Berza→Cundeamor/melón amargo, tras descartar Ají gustoso [≈Ají cubanela] y Culantro [≈Cilantro] por
redundancia/confusión de nombre; Maíz tierno descartado = dup de "Maíz dulce en granos" → Lechuga romana).
Lácteos: Queso cheddar, Queso gouda, Leche de cabra EN POLVO (era Requesón ≈ "Queso ricotta"; POLVO porque
no se consigue líquida en RD), Kéfir. Proteína animal +Arenque (preservado/ahumado, alto sodio).
NOTA: al --commit, el script LIBERA aliases reclamados por genéricas (ver _ALIAS_RELEASES): "lechuga romana"
de "Lechuga" y "chuleta"/"chuleta de cerdo" de "Cerdo", para que las variantes específicas no dupliquen.

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
    # --- Vegetales — PRECIOS RD (La Sirena/Nacional, 2026-06-26) ---
    "Puerro":                {"packages": [{"unit": "paquete", "grams": 300, "label": "paquete (~300 g est.)", "price": 48}]},  # gramos estimados (empaque no los imprime)
    "Cundeamor":             {"packages": [{"unit": "paquete", "grams": 300, "label": "paquete (~300 g est.)", "price": 50}]},  # reemplaza Berza; gramos estimados
    "Bok choy":              {"packages": [{"unit": "paquete", "grams": 250, "label": "250 g", "price": 90}]},
    "Lechuga romana":        {"packages": [{"unit": "paquete", "grams": 907, "label": "2 lb", "price": 158}]},   # reemplaza Maíz tierno; libera alias de "Lechuga"
    "Nabo":                  {"price_per_lb": 28},
    "Alcachofa":             {"packages": [{"unit": "unidad", "grams": 128, "label": "1 alcachofa (~128 g pulpa)", "price": 345}]},  # cara (importada)
    "Palmito":               {"packages": [{"unit": "lata", "grams": 400, "label": "14.1 oz", "price": 275}]},
    "Cebollín":              {"packages": [{"unit": "paquete", "grams": 375, "label": "25 unid (~375 g)", "price": 229}]},
    # --- Frutas — PRECIOS RD (La Sirena/Nacional, 2026-06-26) ---
    "Pera":                  {"price_per_lb": 99},
    "Kiwi":                  {"packages": [{"unit": "paquete", "grams": 450, "label": "450 g", "price": 275}]},
    "Durazno en almíbar":    {"packages": [{"unit": "lata", "grams": 431, "label": "15.2 oz", "price": 180}]},  # único durazno en RD (en almíbar, azúcar añadida)
    "Ciruela":               {"packages": [{"unit": "paquete", "grams": 454, "label": "16 oz", "price": 199}]},
    "Arándanos":             {"packages": [{"unit": "paquete", "grams": 450, "label": "450 g", "price": 198}]},
    "Tamarindo":             {"price_per_lb": 245},
    "Cereza maraschino":     {"packages": [{"unit": "frasco", "grams": 170, "label": "6 oz", "price": 175},
                                           {"unit": "frasco", "grams": 283, "label": "10 oz", "price": 219},
                                           {"unit": "frasco", "grams": 454, "label": "16 oz", "price": 298}]},  # golosina (azúcar); la acerola NO se consigue en RD, solo maraschino
    "Granada":               {"packages": [{"unit": "paquete", "grams": 113, "label": "4 oz", "price": 290}]},
    # --- Granos / carbos (Despensa) ---  (Cuscús descartado: no se vende en Sirena/Nacional; Pan de agua: saltado)
    "Bulgur":                {"packages": [{"unit": "paquete", "grams": 680, "label": "24 oz", "price": 255}]},
    "Cebada":                {"packages": [{"unit": "paquete", "grams": 454, "label": "16 oz", "price": 118}]},
    "Harina de avena":       {"price_per_unit": None, "price_per_lb": None, "market_packages": None},   # PENDIENTE drop → "Harina de Negrito" (oat flour no se consigue; Negrito=crema de trigo)
    # --- Especias / hierbas (Despensa) — PRECIOS RD (La Sirena/Nacional, 2026-06-26); uso ~1-5 g ---
    "Comino":                {"packages": [{"unit": "pote", "grams": 28, "label": "1 oz", "price": 55}]},
    "Cúrcuma":               {"price_per_lb": 99},
    "Laurel":                {"packages": [{"unit": "pote", "grams": 100, "label": "100 g", "price": 150}]},
    "Tomillo":               {"packages": [{"unit": "sobre", "grams": 14, "label": "0.5 oz", "price": 55},
                                           {"unit": "frasco", "grams": 227, "label": "8 oz", "price": 325}]},
    "Curry en polvo":        {"packages": [{"unit": "frasco", "grams": 57, "label": "2 oz", "price": 100}]},
    "Cebolla en polvo":      {"packages": [{"unit": "frasco", "grams": 78, "label": "2.75 oz", "price": 105}]},
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
    "Arenque":               {"packages": [{"unit": "paquete", "grams": 454, "label": "16 oz", "price": 175},
                                           {"unit": "paquete", "grams": 227, "label": "filete 8 oz", "price": 230}]},  # alimento NUEVO (preservado, alto sodio)
    # --- Lácteos ---
    "Queso cheddar":         {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Queso gouda":           {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    "Leche de cabra en polvo": {"packages": [{"unit": "paquete", "grams": 340, "label": "12 oz", "price": 1330}]},   # reemplaza Requesón; POLVO (no había líquida en RD)
    "Kéfir":                 {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
    # --- Víveres ---
    "Mapuey":                {"price_per_unit": None, "price_per_lb": None, "market_packages": None},
}

# Aliases que entradas GENÉRICAS deben LIBERAR cuando insertamos la variante específica (la variante reclama
# el alias). El barrido de colisiones (2026-06-26) detectó que las genéricas verificadas los reclamaban →
# colisión/duplicado. Se libera SOLO si la variante específica ya está en el catálogo (no deja huérfano el
# alias). Decisión del owner: mantener "Lechuga romana" y "Chuleta de cerdo" como entradas propias.
#   {variante_específica_nueva: (entrada_genérica_existente, [aliases_a_quitar_de_la_genérica])}
_ALIAS_RELEASES = {
    "Lechuga romana":   ("Lechuga", ["lechuga romana"]),
    "Chuleta de cerdo": ("Cerdo", ["chuleta de cerdo", "chuleta"]),
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
    inserted_names = []
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
            inserted_names.append(nm)
        # Liberar aliases reclamados por entradas genéricas → la variante específica los reclama. Solo si la
        # variante está presente (insertada en esta corrida o ya existente), para no dejar el alias huérfano.
        present = set(existing) | set(inserted_names)
        for specific, (generic, to_remove) in _ALIAS_RELEASES.items():
            if specific not in present:
                continue
            for al in to_remove:
                if COMMIT:
                    conn.execute(
                        "UPDATE public.master_ingredients SET aliases = array_remove(aliases, %s) WHERE name = %s",
                        (al, generic),
                    )
            print(f"  {'~ LIBERADO' if COMMIT else '~ (dry) liberaría'} alias {to_remove} de '{generic}' → ahora los reclama '{specific}'")
        if COMMIT:
            conn.commit()
            print(f"\nCOMMITTED. insertados={inserted}, ya-existen={skipped_exist}, sin-precio={skipped_price}")
        else:
            print(f"\nDRY-RUN. insertaría={inserted}, ya-existen={skipped_exist}, sin-precio={skipped_price}. "
                  f"Llena PRICES y re-corre con --commit.")


if __name__ == "__main__":
    main()
