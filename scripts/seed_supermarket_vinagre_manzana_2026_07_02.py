"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de VINAGRE DE MANZANA.

Decimocuarta familia con variantes de MARCA del Supermercado RD: 9 SKUs transcritos
del catálogo de La Sirena (capturas del owner, 2026-07-02) — Baldom Ranchero (16/52 Oz),
Heinz, Essential Everyday, Goya, Bragg Organic (16/32 Oz), De Nigris Organic, Mayador.

El genérico del PDF calza exacto: Pote 16 Oz RD$51 = Baldom Ranchero 16 Oz.

Mismo patrón que seed_supermarket_leches_2026_07_02.py. Idempotente: ON CONFLICT
(variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_vinagre_manzana_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_vinagre_manzana_2026_07_02.py --commit   # inserta
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
FOOD = "Vinagre de manzana"
CATEGORY = "Salsas y aderezos"

# (brand, presentation, price_rd, description)
ROWS = [
    ("Baldom Ranchero", "Botella 16 Oz", 51, "Vinagre de manzana"),
    ("Baldom Ranchero", "Botella 52 Oz", 149, "Vinagre de manzana, tamaño grande"),
    ("Heinz", "Botella All Natural 16 Oz", 235, "Vinagre de sidra de manzana, all natural"),
    ("Essential Everyday", "Botella 16 Oz", 150, "Vinagre de sidra de manzana (apple cider vinegar)"),
    ("Goya", "Botella 16 Oz", 110, "Vinagre de sidra de manzana (cider vinegar)"),
    ("Bragg", "Botella Organic Raw 16 Oz", 445, "Vinagre de manzana orgánico crudo sin filtrar, con la madre"),
    ("Bragg", "Botella Organic Raw 32 Oz", 659, "Vinagre de manzana orgánico crudo sin filtrar, con la madre"),
    ("De Nigris", "Botella Organic 500 Ml", 590, "Vinagre de sidra de manzana orgánico crudo sin filtrar, con la madre (Italia)"),
    ("Mayador", "Botella 750 Ml", 190, "Vinagre de sidra de manzana (Asturias)"),
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
    print(f"Seed vinagre de manzana: {len(ROWS)} SKUs · {len(brands)} marcas.")
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
