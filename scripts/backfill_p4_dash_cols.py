"""[P4-UNIFIED-RESOLVER · 2026-06-14] Backfill QUIRÚRGICO de las 4 columnas nuevas (Mg/P/satfat/
cholesterol) desde USDA FoodData Central, SOLO para los ingredientes con fdc_id. Usa el endpoint
detail /food/{fdc_id} (exacto, sin ranking de búsqueda) y extrae por nutrient ID (más robusto que por
nombre). UPDATE acotado a las 4 columnas nuevas (NULL hoy) — no toca macros/micros existentes.

    export USDA_API_KEY=<key>   # requerido (DEMO_KEY: 50/día, lento)
    PYTHONPATH=backend python backend/scripts/backfill_p4_dash_cols.py
"""
import os
import sys
import time

import requests

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass

import psycopg

USDA_KEY = os.environ.get("USDA_API_KEY", "DEMO_KEY")
NEON = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")
NUTRIENT_IDS = {
    1090: "magnesium_mg_per_100g",
    1091: "phosphorus_mg_per_100g",
    1258: "saturated_fat_g_per_100g",
    1253: "cholesterol_mg_per_100g",
}


def fetch(fdc_id):
    url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"
    for _ in range(3):
        r = requests.get(url, params={"api_key": USDA_KEY}, timeout=25)
        if r.status_code == 429:
            print("   429 rate-limit; espera 65s…")
            time.sleep(65)
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        out = {}
        for n in r.json().get("foodNutrients", []):
            nid = (n.get("nutrient") or {}).get("id") or n.get("nutrientId")
            if nid in NUTRIENT_IDS:
                v = n.get("amount") if n.get("amount") is not None else n.get("value")
                if v is not None:
                    out[NUTRIENT_IDS[nid]] = round(float(v), 3)
        return out
    return None


def main():
    if not NEON:
        print("FATAL: NEON_DATABASE_URL(_POOLED) no está en el entorno (.env)"); sys.exit(1)
    print(f"USDA key: {'DEMO_KEY (lento)' if USDA_KEY == 'DEMO_KEY' else 'custom (1000/hr)'}")
    with psycopg.connect(NEON) as conn:
        rows = conn.execute(
            "SELECT id, name, fdc_id FROM master_ingredients WHERE fdc_id IS NOT NULL ORDER BY name"
        ).fetchall()
        print(f"Backfill {len(rows)} ingredientes con fdc_id…")
        ok = miss = 0
        for rid, name, fdc in rows:
            try:
                d = fetch(fdc)
                time.sleep(0.35 if USDA_KEY != "DEMO_KEY" else 2.2)
            except Exception as e:
                print(f"  ERR {name}: {type(e).__name__}: {str(e)[:60]}"); miss += 1; continue
            if not d:
                print(f"  WARN {name} (fdc {fdc}): sin nutrientes nuevos"); miss += 1; continue
            sets = ", ".join(f"{c} = %s" for c in d)
            conn.execute(f"UPDATE master_ingredients SET {sets} WHERE id = %s", (*d.values(), rid))
            ok += 1
            print(f"  OK {name}: " + ", ".join(f"{k.split('_')[0]}={v}" for k, v in d.items()))
        conn.commit()
        print(f"\nDONE: {ok} actualizados, {miss} sin datos nuevos.")


if __name__ == "__main__":
    main()
