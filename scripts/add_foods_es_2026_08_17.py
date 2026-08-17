#!/usr/bin/env python
"""[P1-COUNTRY-SYSTEM-F2 · 2026-08-17] Task 5 — Catálogo España. Inserta al catálogo
(`master_ingredients`, Neon) los 32 alimentos que `scripts/country_catalog_gap.py --country ES`
clasificó DROP contra la lista curada de 80 alimentos/platos españoles (T1): 0
SUSTITUCION-SILENCIOSA, 32 DROP, 48 RESUELVE-BIEN. Estos 32 son exactamente los que la lista
curada nombra y el catálogo vivo NO resuelve por ningún tier (exacto/alias/fuzzy/semántico).

    Jamón serrano      Jamón ibérico      Chorizo español    Morcilla
    Lomo embuchado     Panceta ibérica    Gambas             Almejas
    Boquerones         Anchoas            Cordero            Requesón
    Cuajada            Nata               Judías blancas     Judías pintas
    Acelgas            Fideos             Membrillo          Higo
    Azafrán            Alioli             Turrón             Mazapán
    Sobrasada          Butifarra          Percebes           Vieira
    Chistorra          Piñones            Almendra marcona   Membrillo dulce

NUTRICIÓN: 100% USDA FoodData Central (SR Legacy). Vive en
`scripts/data/new_foods_es_2026_08_17.json` (SSOT del dato, `fdc_id` + `_usda_description` por
fila para auditabilidad). Convención del catálogo respetada: leguminosas en estado SECO,
carnes/mariscos/vegetales/frutas CRUDOS (o su estado de venta real: curado/enlatado donde
corresponde — jamón/chorizo/anchoas se CONSUMEN curados, no en canal).

⚠️ FUENTE DEL DATO: sin `USDA_API_KEY` en este entorno (ni local ni vía DEMO_KEY, que devolvió
`X-Ratelimit-Limit: 10` y no repuso en 65s — insuficiente para 32 altas) se usó el volcado bulk
oficial y gratuito de FoodData Central (`fdc.nal.usda.gov/download-datasets`, SR Legacy
2018-04 + Foundation Foods 2026-04-30, sin límite de tasa, MISMA fuente/schema que consume la
API en vivo) en vez de la API HTTP — mismo dato, mismo `fdc_id`, cero valores inventados. Igual
que en los lotes previos (`fetch_usda_foods_2026_07_26.py`), NUNCA se sustituye un `fdc_id` por
un número inventado con cara de fuente.

SUSTITUCIONES DOCUMENTADAS (USDA no distingue el producto español específico — mismo patrón que
"Queso de oveja"→feta del lote de variedad 2026-07-26, NO inventado, categoría más cercana real):
    Jamón serrano/ibérico → "Pork, cured, ham..." (USDA no tiene "prosciutto"/jamón curado
        dominicano en SR Legacy/Foundation; el corte "unheated"/"raw" es el estado en que se
        CONSUME el jamón curado — sin cocción adicional).
    Lomo embuchado        → "Canadian bacon" (lomo de cerdo curado, mismo corte/preparación).
    Cuajada                → "Cheese, cottage, creamed" (cuajo de leche fresco, análogo más
        cercano; requesón usa ricotta, que SÍ es un match directo).
    Alioli                 → "Salad dressing, mayonnaise, regular" (ajoaceite ≈ perfil graso de
        mayonesa; USDA no tiene "aioli").
    Sobrasada/Chistorra    → mismo fdc que Chorizo español (embutido de cerdo con pimentón,
        USDA no los distingue). Butifarra usa "Sausage, Italian, pork, mild" (blanca, sin
        pimentón — perfil más cercano a un embutido fresco sin adobo rojo).
    Percebes               → "Crustaceans, crab, blue, raw" (USDA no tiene percebe/goose
        barnacle; crustáceo más cercano disponible).
    Almendra marcona        → "Nuts, almonds" (USDA no distingue la variedad Marcona).
    Membrillo dulce         → "Jams and preserves" genérico (USDA no tiene pasta/dulce de
        membrillo; la categoría "confitura de fruta cocida con azúcar" es la más cercana).

Atwater >12% (marcado, NO es error — mismo patrón que "Tomate enlatado" +20,6% del lote de
variedad): Acelgas (+26,1%, 19 kcal — a esa escala la fibra no aportada a Atwater es la mayoría
de la diferencia absoluta, pocas kcal); Azafrán (+16,1%, especia deshidratada rica en fibra que
nadie come en porciones de 100 g).

PRECIOS: NINGUNO de estos 32 lleva precio RD — a propósito (contrato de la Task 5: "fila
master_ingredients SIN precio RD"). España es país BETA (`COUNTRY_PROFILES['ES']['is_beta']`,
P1-COUNTRY-SYSTEM-F1): su lista de compras corre en `pricing_mode='beta_no_prices'`
(`_strip_prices_for_beta_pricing_mode`, T7) — no hay mercado RD que cotizar. A diferencia de
`add_foods_batch1_2026_06_26.py`/`add_foods_variety_2026_07_26.py` (que EXIGEN precio, gate
anti-precio-0, para no contaminar el costeo del plan RD) este script hace lo OPUESTO a propósito:
inserta con `price_per_lb=0, price_per_unit=0` SIEMPRE. El keep sin-precio en la lista de compras
lo cubre la generalización de P1-BAKING-STAPLES (`shopping_calculator._COUNTRY_CATALOG_UNPRICED_TOKENS`,
mismo mecanismo/knob propio `MEALFIT_COUNTRY_CATALOG_UNPRICED_KEEP`) — NO el gate
`_is_verified_for_shopping` (ese sigue exigiendo precio>0, intacto, `MEALFIT_VERIFIED_INGREDIENTS_ONLY`
sin tocar).

IDEMPOTENTE: salta por `name` ya existente (mismo patrón que los 3 lotes previos) — re-correr no
duplica.

USO:
    cd backend
    python scripts/add_foods_es_2026_08_17.py              # DRY-RUN
    python scripts/add_foods_es_2026_08_17.py --commit      # inserta de verdad

[P2-LOGGER-EXEMPT: script CLI one-shot, la salida a stdout ES el producto]
"""
import datetime
import json
import os
import sys

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

_AQUI = os.path.dirname(os.path.abspath(__file__))
_NEON = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")
COMMIT = "--commit" in sys.argv

# record-key → columna DB (mismo mapeo que add_foods_batch1_2026_06_26.py / add_foods_variety_2026_07_26.py)
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


def _registros():
    for p in (os.path.join(_AQUI, "data", "new_foods_es_2026_08_17.json"),
              os.path.join(os.getcwd(), "scripts", "data", "new_foods_es_2026_08_17.json"),
              "/tmp/new_foods_es_2026_08_17.json"):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    print("FATAL: no se encontró new_foods_es_2026_08_17.json", file=sys.stderr)
    sys.exit(1)


def main():
    recs = _registros()
    if not _NEON:
        print("FATAL: NEON url ausente", file=sys.stderr)
        return 1

    hoy = datetime.date.today()
    puestos = ya = 0
    with psycopg.connect(_NEON) as conn:
        existen = {r[0] for r in conn.execute("SELECT name FROM public.master_ingredients").fetchall()}
        for r in recs:
            nm = r["name"]
            if nm in existen:
                print(f"  ~ EXISTE, salto: {nm}")
                ya += 1
                continue
            cols = {
                "slug": r["slug"], "name": nm, "category": r["category"],
                "aliases": r.get("aliases") or [], "default_unit": r["default_unit"],
                "is_dominican_cultivar": bool(r.get("is_dominican_cultivar")),
                "density_g_per_cup": r.get("density_g_per_cup"),
                "density_g_per_unit": r.get("density_g_per_unit"),
                "nutrition_source": "usda", "nutrition_source_date": hoy,
                "fdc_id": r.get("fdc_id"),
                # [T5 · SIN precio RD, a propósito — ver docstring del módulo]
                "price_per_lb": 0, "price_per_unit": 0,
            }
            for k, dbcol in _COLMAP.items():
                cols[dbcol] = r.get(k)
            nombres = list(cols.keys())
            if COMMIT:
                conn.execute(
                    f"INSERT INTO public.master_ingredients ({', '.join(nombres)}) "
                    f"VALUES ({', '.join(['%s'] * len(nombres))})",
                    [cols[c] for c in nombres])
            print(f"  {'+ INSERTADO' if COMMIT else '+ (dry) insertaría'}: {nm} [{r['category']}] "
                  f"{r['kcal']}kcal/{r['protein_g']}P/{r.get('sodium_mg','?')}mgNa "
                  f"fdc={r.get('fdc_id')} SIN-PRECIO fuente={r.get('_usda_description', '?')!r}")
            puestos += 1
        if COMMIT:
            conn.commit()
            print(f"\nCOMMITTED. insertados={puestos}, ya-existen={ya}")
        else:
            print(f"\nDRY-RUN. insertaría={puestos}, ya-existen={ya}. Re-corre con --commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
