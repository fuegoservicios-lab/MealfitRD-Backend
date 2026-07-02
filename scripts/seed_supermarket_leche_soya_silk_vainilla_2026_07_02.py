"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed complemento de
LECHE DE SOYA (Silk vainilla).

Familia 109 del Supermercado RD (captura del owner, 2026-07-02). De los
2 SKUs de la captura, la Silk Soy natural sin azúcar 32 Oz RD$230 YA
estaba cargada (precio idéntico). Entra solo la Silk vainilla 32 Oz
RD$250 ("No disponible", referencia). El genérico "Cartón 32 Oz" RD$184
no calza con ninguna de las dos.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_leche_soya_silk_vainilla_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_leche_soya_silk_vainilla_2026_07_02.py --commit   # inserta
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
LACTEOS = "Lácteos y huevos"

# (food_name, brand, presentation, price_rd, description)
ROWS = [
    ("Leche de soya", "Silk Soy", "Cartón Vainilla 32 Oz", 250, "Leche de soya sabor vainilla"),
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

    for row in ROWS:
        if len(row) != 5:
            print(f"FATAL: fila con {len(row)} campos (esperados 5): {row[:3]}")
            sys.exit(1)

    print(f"Seed leche de soya (complemento Silk vainilla): {len(ROWS)} SKU.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, brand, pres, price, desc) in ROWS:
                cur.execute(_INSERT, (food, brand, pres, price, NOTES, LACTEOS, food, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
