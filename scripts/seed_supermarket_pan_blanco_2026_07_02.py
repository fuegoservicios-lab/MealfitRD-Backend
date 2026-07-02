"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de PAN BLANCO.

Decimonovena familia con variantes de MARCA del Supermercado RD: 2 SKUs nuevos
transcritos del catálogo de La Sirena (capturas del owner, 2026-07-02).

De los 4 listados de la captura, 2 YA existían en el catálogo:
  * Pan Blanco Wala Viga Med 820 G RD$105 — cargado previamente bajo
    "Pan blanco familiar" (mismo precio, calza exacto).
  * Pan Bolin Integral Blanco Viga Mediana RD$165 — cargado en el seed de
    PAN INTEGRAL bajo "Pan integral familiar" (La Sirena lo lista en ambas
    búsquedas; es grano integral con miga blanca, no pan blanco).

Los 2 nuevos se cargan con el PRECIO DE LISTA (tachado), no el de promo
transitoria (misma decisión que el dedupe Cherry Star del seed de atún):
  * Bolin Mini Club Blanco Suave: lista RD$122 (promo -6% RD$115).
  * Molino del Sol Artesano Blanco 720 gr: lista RD$155 (promo -10% RD$139).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_pan_blanco_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_pan_blanco_2026_07_02.py --commit   # inserta
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
CATEGORY = "Panadería y harinas"

# (food_name, brand, presentation, price_rd, description)
ROWS = [
    ("Pan blanco personal", "Bolin", "Viga Mini Club Blanco Suave", 122, "Pan de viga blanco suave, tamaño mini club"),
    ("Pan blanco familiar", "Molino del Sol", "Artesano 720 gr", 155, "Pan artesano blanco"),
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
    for (food, brand, pres, *_rest) in ROWS:
        key = (food.lower(), (brand or "").lower(), pres.lower())
        if key in seen:
            print(f"FATAL: fila duplicada en el seed: {food} / {brand} / {pres}")
            sys.exit(1)
        seen.add(key)

    brands = {r[1] for r in ROWS}
    print(f"Seed pan blanco: {len(ROWS)} SKUs · {len(brands)} marcas.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, brand, pres, price, desc) in ROWS:
                cur.execute(_INSERT, (food, brand, pres, price, NOTES, CATEGORY, food, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
