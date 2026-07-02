"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de MANTEQUILLA DE MANÍ.

Quinta familia con variantes de MARCA del Supermercado RD: 12 SKUs transcritos del
catálogo de La Sirena (capturas del owner, 2026-07-02) — Essential Everyday, Peter Pan,
Zerca, Wala, Jif, Peanut Butter & Co, Virgin Nature; cremosas y crujientes.

EXCLUIDOS a propósito: galleta Lenny & Larry's (repostería), Purina Beggin' (snack para
perros) y Chobani Flip Peanut Butter (yogurt con topping, además "No disponible").

Mismo patrón que seed_supermarket_leches_2026_07_02.py. Idempotente: ON CONFLICT
(variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_mantequilla_mani_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_mantequilla_mani_2026_07_02.py --commit   # inserta
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
FOOD = "Mantequilla de maní"
CATEGORY = "Semillas y frutos secos"

# (brand, presentation, price_rd, description)
ROWS = [
    ("Essential Everyday", "Pote Cremosa 16 Oz", 205, "Mantequilla de maní cremosa (creamy)"),
    ("Essential Everyday", "Pote Cremosa 28 Oz", 385, "Mantequilla de maní cremosa, tamaño grande"),
    ("Essential Everyday", "Pote Crujiente 16 Oz", 205, "Mantequilla de maní crujiente (crunchy)"),
    ("Peter Pan", "Pote Cremosa 16.3 Oz", 355, "Mantequilla de maní cremosa (creamy)"),
    ("Peter Pan", "Pote Crujiente 16.3 Oz", 355, "Mantequilla de maní crujiente (crunchy)"),
    ("Zerca", "Pote Cremosa 16 Oz", 117, "Mantequilla de maní cremosa"),
    ("Zerca", "Pote Crujiente 16 Oz", 117, "Mantequilla de maní crujiente"),
    ("Wala", "Pote Cremosa 12 Oz", 214, "Mantequilla de maní cremosa"),
    ("Wala", "Pote Crujiente 12 Oz", 214, "Mantequilla de maní crujiente"),
    ("Jif", "Pote Cremosa 16 Oz", 295, "Mantequilla de maní cremosa, 7g de proteína por porción"),
    ("Peanut Butter & Co", "Pote Smooth Operator 16 Oz", 445, "Mantequilla de maní natural cremosa (no-stir)"),
    ("Virgin Nature", "Pote Maní y Chocolate 8 Oz", 335, "Mantequilla de maní con chocolate, alta en proteína"),
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
    print(f"Seed mantequilla de maní: {len(ROWS)} SKUs · {len(brands)} marcas.")
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
