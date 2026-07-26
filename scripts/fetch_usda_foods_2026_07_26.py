#!/usr/bin/env python
"""[P1-CATALOG-VARIETY-GAP · 2026-07-26] Extrae de USDA FoodData Central los macros de los alimentos
que `/supermercado` vende pero el catálogo nutricional NO tiene — o sea, comprables pero imposibles
de usar en un plan.

Salieron de la auditoría de variedad del 2026-07-26: `supermarket_products` referencia 247 nombres de
alimento y `master_ingredients` sólo tiene 204; 46 nombres no casaban ni por alias ni por plural.

⚠️ Este script NO inventa nada. Cada valor viene de un `fdcId` real y se imprime la provenance para
que quede auditable, igual que los lotes de 2026-06-26. Cruza Atwater (4·prot + 4·carb + 9·grasa vs
kcal declaradas) y MARCA lo que divergo >12% en vez de aceptarlo callado.

CONVENCIÓN DEL CATÁLOGO (respetada en las queries): granos/pastas/harinas en SECO, carnes y vegetales
CRUDOS, per 100 g.

PRECIOS: no se tocan aquí. El script de inserción los exige (gate anti-precio-0) y los llena el owner
con valores reales del mercado RD.

USO (en el VPS, donde vive USDA_API_KEY):
    python scripts/fetch_usda_foods_2026_07_26.py            # busca y reporta
    python scripts/fetch_usda_foods_2026_07_26.py --json OUT # además escribe el JSON del lote

[P2-LOGGER-EXEMPT: script CLI one-shot, la salida a stdout ES el producto]
"""
import json
import os
import sys
import time

try:
    from dotenv import load_dotenv
    for _p in ("/opt/mealfit/backend/.env",
               os.path.join(os.path.dirname(__file__), "..", ".env")):
        if os.path.exists(_p):
            load_dotenv(_p)
            break
except Exception:
    pass

import requests

API = "https://api.nal.usda.gov/fdc/v1"
KEY = os.environ.get("USDA_API_KEY")

# Cada entrada: (nombre catálogo, slug, categoría, default_unit, query USDA, aliases)
# La query lleva el estado que exige la convención del catálogo (raw/dry) para no mezclar cocido.
# DESCARTADOS a propósito, con su razón — para que nadie los "complete" luego a ciegas:
#   Kale             → YA está en el catálogo (alias 'col rizada','berza'); el supermercado dice
#                      "Kale Picado" y eso se arregla con un alias, no con una fila nueva.
#   Sardina fresca   → el catálogo tiene "Sardinas en lata"; USDA sólo devolvió enlatada-en-tomate
#                      (414 mg Na, otro alimento). La fresca necesita su propia fuente cruda.
#   Yogurt de cabra  → USDA NO tiene yogurt de cabra. La búsqueda devolvía yogurt griego de VACA:
#                      usarlo sería inventar el dato con cara de fuente. Queda pendiente.
#   Chuleta costillas→ redundante con "Chuleta" + "Costilla de cerdo" ya presentes.
OBJETIVO = [
    ("Guisantes",              "guisantes",              "Vegetales",  "gramos",
     "Peas, green, raw",             ["guisantes", "guisantes frescos", "petit pois", "green peas"]),
    ("Espaguetis",             "espaguetis",             "Despensa",   "gramos",
     "Spaghetti, dry, unenriched",   ["espaguetis", "espagueti", "spaghetti", "pasta blanca"]),
    ("Huevos de codorniz",     "huevos_de_codorniz",     "Proteínas",  "unidad",
     "Egg, quail, whole, fresh, raw", ["huevos de codorniz", "huevo de codorniz", "codorniz"]),
    ("Jamón de cerdo",         "jamon_de_cerdo",         "Proteínas",  "gramos",
     "Ham, sliced, pre-packaged, deli meat",  ["jamon de cerdo", "jamón", "jamon cocido"]),
    ("Salchichas",             "salchichas",             "Proteínas",  "unidad",
     "Sausage, Vienna, canned, chicken, beef, pork",
     ["salchichas", "salchicha", "vienna sausage", "salchichón de lata"]),
    ("Pan pita integral",      "pan_pita_integral",      "Despensa",   "unidad",
     "Bread, pita, whole-wheat",     ["pan pita integral", "pita integral", "pan arabe integral"]),
    ("Pan de semillas",        "pan_de_semillas",        "Despensa",   "unidad",
     "Bread, multi-grain",
     ["pan de semillas", "pan multigrano", "pan de granos"]),
    ("Harina de garbanzo",     "harina_de_garbanzo",     "Despensa",   "gramos",
     "Chickpea flour (besan)",       ["harina de garbanzo", "besan", "chickpea flour"]),
    ("Harina de arroz integral", "harina_de_arroz_integral", "Despensa", "gramos",
     "Rice flour, brown",            ["harina de arroz integral", "brown rice flour"]),
    ("Tomate enlatado",        "tomate_enlatado",        "Despensa",   "gramos",
     "Tomatoes, red, ripe, canned, packed in tomato juice",
     ["tomate enlatado", "tomates enlatados", "tomate de lata"]),
    ("Queso de oveja",         "queso_de_oveja",         "Lácteos",    "gramos",
     "Cheese, feta",                 ["queso de oveja", "queso oveja", "feta"]),
]

# nutrient number USDA → clave del JSON del catálogo
NUM = {
    "208": "kcal", "203": "protein_g", "205": "carbs_g", "204": "fats_g",
    "291": "fiber_g", "269": "sugars_g", "606": "satfat_g", "307": "sodium_mg",
    "601": "cholesterol_mg", "301": "calcium_mg", "303": "iron_mg", "306": "potassium_mg",
    "304": "magnesium_mg", "305": "phosphorus_mg", "309": "zinc_mg", "328": "vit_d_mcg",
    "418": "b12_mcg", "435": "folate_mcg_dfe", "320": "vit_a_mcg_rae", "401": "vit_c_mg",
    "323": "vit_e_mg", "430": "vit_k_mcg", "317": "selenium_mcg", "851": "omega3_ala_g",
}
PREFERIDO = ("SR Legacy", "Foundation", "Survey (FNDDS)", "Branded")


def buscar(query):
    # Los PARÉNTESIS rompen el endpoint de búsqueda de USDA (HTTP 400 con `query=` vacío) — pasó con
    # "Chickpea flour (besan)" y "Bread, multi-grain (includes whole-grain)". Se sanean aquí y no en
    # cada literal, para que el próximo que copie una descripción de USDA tal cual no se tropiece.
    q = query.replace("(", " ").replace(")", " ")
    q = " ".join(q.split())
    r = requests.get(f"{API}/foods/search", timeout=40,
                     params={"query": q, "pageSize": 25, "api_key": KEY})
    r.raise_for_status()
    foods = r.json().get("foods") or []
    for tipo in PREFERIDO:
        for f in foods:
            if f.get("dataType") == tipo:
                return f
    return foods[0] if foods else None


def detalle(fdc_id):
    r = requests.get(f"{API}/food/{fdc_id}", timeout=40, params={"api_key": KEY})
    r.raise_for_status()
    d = r.json()
    vals = {}
    for n in d.get("foodNutrients") or []:
        num = str(((n.get("nutrient") or {}).get("number") or "")).strip()
        if num in NUM:
            v = n.get("amount")
            if v is not None:
                vals[NUM[num]] = round(float(v), 4)
    return d, vals


def main():
    if not KEY:
        print("FATAL: falta USDA_API_KEY en el entorno/.env", file=sys.stderr)
        return 2
    lote, problemas = [], []
    for nombre, slug, categoria, unidad, query, aliases in OBJETIVO:
        try:
            hit = buscar(query)
            if not hit:
                problemas.append((nombre, "sin resultado en USDA")); continue
            d, vals = detalle(hit["fdcId"])
            faltan = [k for k in ("kcal", "protein_g", "carbs_g", "fats_g") if k not in vals]
            if faltan:
                problemas.append((nombre, f"macro ausente: {faltan}")); continue
            atw = 4 * vals["protein_g"] + 4 * vals["carbs_g"] + 9 * vals["fats_g"]
            desv = (atw - vals["kcal"]) / vals["kcal"] * 100 if vals["kcal"] else 0
            marca = "  ⚠ ATWATER" if abs(desv) > 12 else ""
            print(f"{nombre:<26} fdc={hit['fdcId']:<9} {hit.get('dataType','?'):<14} "
                  f"kcal={vals['kcal']:<6} P={vals['protein_g']:<5} C={vals['carbs_g']:<6} "
                  f"G={vals['fats_g']:<5} Na={vals.get('sodium_mg','-'):<6} atwater={desv:+.1f}%{marca}")
            print(f"{'':<26} → {d.get('description','')[:88]}")
            reg = {"name": nombre, "slug": slug, "category": categoria, "aliases": aliases,
                   "default_unit": unidad, "is_dominican_cultivar": False}
            reg.update(vals)
            reg["fdc_id"] = hit["fdcId"]
            reg["provenance"] = (f"USDA {hit.get('dataType')} {hit['fdcId']} "
                                 f"({d.get('description','')}); query='{query}'; "
                                 f"Atwater {desv:+.1f}%. Extraído 2026-07-26 por "
                                 f"fetch_usda_foods_2026_07_26.py.")
            if abs(desv) > 12:
                reg["provenance"] += " ⚠ REVISAR: divergencia Atwater >12%."
            lote.append(reg)
            time.sleep(0.4)
        except Exception as e:
            problemas.append((nombre, f"{type(e).__name__}: {str(e)[:90]}"))

    print(f"\nsourced: {len(lote)}/{len(OBJETIVO)}")
    if problemas:
        print("NECESITAN REVISIÓN MANUAL (no se inventan valores):")
        for n, p in problemas:
            print(f"  {n:<26} {p}")
    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
        with open(out, "w", encoding="utf-8") as f:
            json.dump(lote, f, ensure_ascii=False, indent=2)
        print(f"\nJSON escrito en {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
