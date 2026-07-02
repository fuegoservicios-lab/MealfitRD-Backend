"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de VINAGRE BLANCO.

Vigesimoséptima familia con variantes de MARCA del Supermercado RD: 6 SKUs nuevos
transcritos del catálogo de La Sirena (capturas del owner, 2026-07-02) — Wala galón,
Heinz, Constanza, Essential Everyday (64/128 Oz) y Goya.

Notas:
  * El Wala 16 Oz RD$25 YA estaba en el catálogo (calza exacto con el genérico
    "Pote 16 Oz" RD$25) — no se duplica.
  * "Vinagre Blanco Esencial 128 Oz." de La Sirena es la misma etiqueta Essential
    Everyday del 64 Oz — se normaliza la marca (typo del listado).
  * El food usa la grafía del catálogo genérico: "Vinagre Blanco".

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_vinagre_blanco_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_vinagre_blanco_2026_07_02.py --commit   # inserta
"""
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

_NEON = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")
COMMIT = "--commit" in sys.argv

NOTES = "Precio de referencia La Sirena · 2026-07"
FOOD = "Vinagre Blanco"
CATEGORY = "Salsas y aderezos"

# (brand, presentation, price_rd, description)
ROWS = [
    ("Wala", "Galón 1 Gl", 110, "Vinagre blanco, tamaño galón"),
    ("Heinz", "Botella All Natural Destilado 16 Oz", 195, "Vinagre blanco destilado, all natural (5% acidez)"),
    ("Constanza", "Botella 17.5 Oz", 35, "Vinagre blanco"),
    ("Essential Everyday", "Botella Destilado 64 Oz", 270, "Vinagre blanco destilado (distilled white vinegar)"),
    ("Essential Everyday", "Botella Destilado 128 Oz", 485, "Vinagre blanco destilado (distilled white vinegar), tamaño galón"),
    ("Goya", "Botella Destilado 16 Oz", 75, "Vinagre blanco destilado"),
]

_INSERT = """
INSERT INTO public.supermarket_products
    (food_name, brand, presentation, portion_label, duration_label,
     price_rd, notes, category, master_food_name, description, is_verified, active)
VALUES (%s, %s, %s, NULL, NULL, %s, %s, %s, %s, %s, true, true)
ON CONFLICT (lower(food_name), lower(coalesce(brand,'')), lower(coalesce(presentation,'')))
DO NOTHING
RETURNING id
"""


def main():
    if not _NEON:
        print("FATAL: NEON_DATABASE_URL no está definido (.env)")
        sys.exit(1)

    seen = set()
    for (brand, pres, *_rest) in ROWS:
        key = ((brand or "").lower(), pres.lower())
        if key in seen:
            print(f"FATAL: fila duplicada en el seed: {brand} / {pres}")
            sys.exit(1)
        seen.add(key)

    brands = {r[0] for r in ROWS}
    print(f"Seed vinagre blanco: {len(ROWS)} SKUs · {len(brands)} marcas.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (brand, pres, price, desc) in ROWS:
                cur.execute(_INSERT, (FOOD, brand, pres, price, NOTES, CATEGORY, FOOD, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
