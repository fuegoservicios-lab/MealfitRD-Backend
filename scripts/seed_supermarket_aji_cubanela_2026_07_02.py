"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de AJÍ CUBANELA.

Familia 56 del Supermercado RD: 1 SKU transcrito del catálogo de La Sirena
(captura del owner, 2026-07-02):

  * Vaso porcionado en aros (unidad) RD$88 — "No disponible", referencia
    (patrón ají morrón porcionado).

  El listing "Ají Cubanela Verde Lb" RD$68 CALZA EXACTO con el genérico "Lb"
  ya cargado — mismo producto sin marca, NO se duplica (precedente pechuga
  de pollo importada congelada).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_aji_cubanela_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_aji_cubanela_2026_07_02.py --commit   # inserta
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
VEG = "Vegetales y verduras"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    ("Ají cubanela", None, "Vaso Porcionado (unidad)", 88, "Ají cubanela porcionado en aros, listo para cocinar", VEG),
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
        if len(row) != 6:
            print(f"FATAL: fila con {len(row)} campos (esperados 6): {row[:3]}")
            sys.exit(1)
        (food, brand, pres, *_rest) = row
        key = (food.lower(), (brand or "").lower(), pres.lower())
        if key in seen:
            print(f"FATAL: fila duplicada en el seed: {food} / {brand} / {pres}")
            sys.exit(1)
        seen.add(key)

    foods = {r[0] for r in ROWS}
    print(f"Seed ají cubanela: {len(ROWS)} SKUs · {len(foods)} foods.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, brand, pres, price, desc, category) in ROWS:
                cur.execute(_INSERT, (food, brand, pres, price, NOTES, category, food, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
