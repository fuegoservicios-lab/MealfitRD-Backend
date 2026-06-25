"""[micro-precision corrective · 2026-06-25] Reproducible SSOT del fix de precisión de micros.

ROOT CAUSE: el backfill P1-FOOD-DB-EXTENDED-MICROS dejó ~1/3 de master_ingredients con micros
extendidos NULL porque varios `fdc_id` apuntan a entradas USDA Foundation/Survey/Branded que NO
traen el panel extendido (ej. Almendras fdc 2261420 -> USDA devuelve solo zinc/folato/selenio,
SIN vit E). `backfill_extended_micros.py` funciona bien — solo escribe lo que USDA devuelve; re-
correrlo NO arregla (determinista). Verificable con `backfill_extended_micros.py --dry-run`.

ESTE SCRIPT (idempotente, COALESCE/NULL-only, NO toca fdc_id/macros):
  1. Rellena micros extendidos NULL desde fdc_ids SR-Legacy COMPLETOS basis-matched (mapa SRC).
  2. Rellena Atún (canned-in-water) y Plátano maduro/verde con valores USDA SR (sin fdc completo).
  3. Añade density_g_per_cup a Linaza/Chía/Almendras (portioning: "1 cda" no convertía a gramos
     -> to_grams: g = ml * density_g_per_cup/240 -> el omega-3 de la linaza no se contaba).

Impacto verificado (plan real): omega-3 0.3->0.9, vit E 8.7->13.0, zinc 8.9->10.1, selenio 69->176;
vit D sin cambio (gap real, honesto). NULLs sistémicos ~mitad.

LECCIÓN: tener fdc_id != panel completo. Verificar COMPLETITUD del fdc (dry-run que imprime qué
nutrientes trae), no solo `fdc_id IS NOT NULL`.

    USDA_API_KEY + NEON_DATABASE_URL(_POOLED) en .env.  python scripts/micro_corrective_2026_06_25.py [--commit]
"""
import os, sys, time, json, urllib.request, urllib.parse, urllib.error
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass
import psycopg

USDA_KEY = os.environ.get("USDA_API_KEY", "DEMO_KEY")
NEON = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")
COMMIT = "--commit" in sys.argv

# catalog name -> COMPLETE SR-Legacy source fdc_id (basis raw/as-eaten verificado por dry-run)
SRC = {
    "Almendras fileteadas": 170567, "Yogurt": 170903, "Yogurt griego entero": 170903,
    "Yogurt griego sin azúcar": 170903, "Semillas de chía": 170554, "Pan integral familiar": 172686,
    "Pan integral personal": 172686, "Brócoli": 170379, "Coliflor": 169986, "Piña": 169124,
    "Manzana": 171688, "Pechuga de pollo": 171077, "Melón": 169092, "Vainitas": 169961,
    "Papa": 170093, "Camarones": 171971,
}
# foods cuyo fdc propio no trae panel completo y sin SR-Legacy basis-match limpio -> valores USDA SR per 100g
MANUAL = {
    "Atún en agua":    {"zinc_mg_per_100g": 0.65, "folate_mcg_dfe_per_100g": 4, "vitamin_a_mcg_rae_per_100g": 5,
                        "vitamin_c_mg_per_100g": 0, "vitamin_e_mg_per_100g": 0.85, "vitamin_k_mcg_per_100g": 0.1,
                        "selenium_mcg_per_100g": 80.4, "omega3_ala_g_per_100g": 0.001},
    "Plátano maduro":  {"vitamin_a_mcg_rae_per_100g": 56, "vitamin_e_mg_per_100g": 0.14, "vitamin_k_mcg_per_100g": 0.7,
                        "folate_mcg_dfe_per_100g": 22, "selenium_mcg_per_100g": 1.5, "omega3_ala_g_per_100g": 0.03},
    "Plátano verde":   {"vitamin_a_mcg_rae_per_100g": 30, "vitamin_e_mg_per_100g": 0.14, "vitamin_k_mcg_per_100g": 0.7,
                        "selenium_mcg_per_100g": 1.5},
}
DENSITY = {"Linaza": 112, "Semillas de chía": 190, "Almendras fileteadas": 92}  # g per 240ml cup

TARGETS = {
    "zinc_mg_per_100g": ([1095], ["zinc, zn"]),
    "folate_mcg_dfe_per_100g": ([1190, 1177], ["folate, dfe", "folate, total"]),
    "vitamin_a_mcg_rae_per_100g": ([1106], ["vitamin a, rae"]),
    "vitamin_c_mg_per_100g": ([1162], ["vitamin c, total ascorbic acid"]),
    "vitamin_e_mg_per_100g": ([1109], ["vitamin e (alpha-tocopherol)"]),
    "vitamin_k_mcg_per_100g": ([1185], ["vitamin k (phylloquinone)"]),
    "selenium_mcg_per_100g": ([1103], ["selenium, se"]),
    "omega3_ala_g_per_100g": ([1404, 1270], ["pufa 18:3 n-3 c,c,c (ala)", "18:3 n-3 c,c,c (ala)"]),
}
COLS = list(TARGETS.keys())


def fetch(fdc_id):
    url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}?api_key={urllib.parse.quote(USDA_KEY)}"
    for _ in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mealfit-corrective"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(60); continue
            if e.code == 404:
                return None, None
            raise
        by_id, by_name = {}, {}
        for n in data.get("foodNutrients", []) or []:
            nut = n.get("nutrient") or {}
            amt = n.get("amount") if n.get("amount") is not None else n.get("value")
            if amt is None:
                continue
            if nut.get("id") is not None:
                by_id[nut["id"]] = float(amt)
            nm = str(nut.get("name") or "").strip().lower()
            if nm:
                by_name[nm] = float(amt)
        out = {}
        for col, (ids, names) in TARGETS.items():
            val = next((by_id[i] for i in ids if i in by_id), None)
            if val is None:
                val = next((by_name[nm] for nm in names if nm in by_name), None)
            if val is not None:
                out[col] = round(val, 3)
        return data.get("description"), out
    return None, None


def fill(conn, name, vals):
    row = conn.execute(f"SELECT {', '.join(COLS)} FROM master_ingredients WHERE name = %s", (name,)).fetchone()
    if not row:
        print(f"  !! no row: {name}"); return
    cur = {c: row[i] for i, c in enumerate(COLS)}
    to_set = {c: vals[c] for c in COLS if cur[c] is None and c in vals}
    parts = [f"{c.split('_')[0]}={v}" for c, v in to_set.items()]
    print(f"  [{name}] fill: {', '.join(parts) or '(nada)'}")
    if COMMIT and to_set:
        sets = ", ".join(f"{c} = %s" for c in to_set)
        conn.execute(f"UPDATE master_ingredients SET {sets} WHERE name = %s", (*to_set.values(), name))


def main():
    if not NEON:
        print("FATAL: NEON url"); sys.exit(1)
    print(f"commit={COMMIT}\n-- USDA SR-Legacy map fills --")
    with psycopg.connect(NEON) as conn:
        for name, fdc in SRC.items():
            desc, vals = fetch(fdc); time.sleep(0.25)
            if vals is None:
                print(f"  !! fetch fail {name}"); continue
            print(f"  src='{desc}'", end="  "); fill(conn, name, vals)
        print("-- manual USDA SR fills --")
        for name, vals in MANUAL.items():
            fill(conn, name, vals)
        print("-- portioning densities --")
        for name, dcup in DENSITY.items():
            r = conn.execute("SELECT density_g_per_cup FROM master_ingredients WHERE name = %s", (name,)).fetchone()
            if r and r[0] is None:
                print(f"  [{name}] density_g_per_cup={dcup}")
                if COMMIT:
                    conn.execute("UPDATE master_ingredients SET density_g_per_cup = %s WHERE name = %s", (dcup, name))
            else:
                print(f"  [{name}] density ya presente ({r[0] if r else '??'}) — skip")
        if COMMIT:
            conn.commit(); print("\nCOMMITTED.")
        else:
            print("\nDRY-RUN. Re-run con --commit.")


if __name__ == "__main__":
    main()
