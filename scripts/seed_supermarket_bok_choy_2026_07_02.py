"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de BOK CHOY.

Familia 110 del Supermercado RD (capturas del owner, 2026-07-02,
**Supermercados Nacional**).

BOK CHOY (2): CALCE (con redondeo del PDF) del genérico "Paquete 250 gr"
RD$90 = Trudeau paq. 250 gr $89.95. Entra la marca + el segundo paquete
Trudeau sin gramaje $69.95 (agotado, referencia) — el TÍTULO del store
dice "Espinaca Bok Choy" pero el empaque dice BOK CHOY (empaque manda;
es bok choy baby, no espinaca).

ARENQUE: 0 filas — los 2 listings de La Sirena SON los genéricos ya
cargados ("Arenque Paquete 16 Oz" RD$175 y "Filete Arenque Paquete 8 Oz"
RD$230, calces exactos).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_bok_choy_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_bok_choy_2026_07_02.py --commit   # inserta
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

NOTES = "Precio de referencia Supermercados Nacional · 2026-07"
VEG = "Vegetales y verduras"

# (food_name, brand, presentation, price_rd, description)
ROWS = [
    ("Bok choy", "Trudeau", "Paquete 250 gr", 89.95, "Bok choy fresco"),
    ("Bok choy", "Trudeau", "Paquete Baby Bok Choy", 69.95, "Baby bok choy fresco (el store lo titula 'espinaca bok choy'; el empaque dice bok choy)"),
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
    for row in ROWS:
        if len(row) != 5:
            print(f"FATAL: fila con {len(row)} campos (esperados 5): {row[:3]}")
            sys.exit(1)
        (food, brand, pres, *_rest) = row
        key = (food.lower(), (brand or "").lower(), pres.lower())
        if key in seen:
            print(f"FATAL: fila duplicada en el seed: {food} / {brand} / {pres}")
            sys.exit(1)
        seen.add(key)

    print(f"Seed bok choy: {len(ROWS)} SKUs.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, brand, pres, price, desc) in ROWS:
                cur.execute(_INSERT, (food, brand, pres, price, NOTES, VEG, food, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
