"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de MANZANA y ACEITE VEGETAL.

Familias 68-69 del Supermercado RD: 13 SKUs transcritos del catálogo de La Sirena
(capturas del owner, 2026-07-02):

  * "Manzana" (11): tipos por libra como variantes con nombre (criterio
    yautía) — la Roja Lb RD$78 CALZA EXACTO con el genérico "Lb"; verde,
    Fuji y amarilla. Fundas: Hess Brothers Gala 3 Lb (marca del empaque),
    Joyfully Grown Pink Lady 3 Lb y Fuji orgánica 3 Lb (marca del empaque),
    verde 3 Lb, amarilla 3 Lb, Sugar Bee 2 Lb y Rockit 2 Lb (Sugar Bee y
    Rockit son variedades trademark — van en la presentación, sin marca).
  * "Aceite vegetal" (2): Wesson 64 Oz RD$535 y 48 Oz RD$425 — la 48 Oz
    CALZA EXACTO con el genérico "Botella 48 Oz" RD$425 ya cargado (el food
    ya existía; la búsqueda pre-seed se truncó en 40 filas por las ~40 de
    aceite de oliva).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_manzana_aceite_vegetal_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_manzana_aceite_vegetal_2026_07_02.py --commit   # inserta
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
FRU = "Frutas"
GRASA = "Aceites y grasas"

MANZ = "Manzana"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Manzana · por libra (tipos) ──
    (MANZ, None, "Roja Lb", 78, "Manzana roja fresca, por libra", FRU),
    (MANZ, None, "Verde Lb", 98, "Manzana verde (granny smith) fresca, por libra", FRU),
    (MANZ, None, "Fuji Lb", 88, "Manzana fuji fresca, por libra", FRU),
    (MANZ, None, "Amarilla Lb", 96, "Manzana amarilla (golden) fresca, por libra", FRU),
    # ── Manzana · fundas ──
    (MANZ, "Hess Brothers", "Funda Gala 3 Lb", 280, "Manzanas gala en funda", FRU),
    (MANZ, None, "Funda Verde 3 Lb", 275, "Manzanas verdes en funda", FRU),
    (MANZ, None, "Funda Amarilla 3 Lb", 265, "Manzanas amarillas (golden) en funda", FRU),
    (MANZ, "Joyfully Grown", "Funda Pink Lady 3 Lb", 300, "Manzanas pink lady en funda", FRU),
    (MANZ, "Joyfully Grown", "Funda Fuji Orgánica 3 Lb", 268, "Manzanas fuji orgánicas en funda", FRU),
    (MANZ, None, "Funda Sugar Bee 2 Lb", 350, "Manzanas variedad SugarBee en funda (907 gr)", FRU),
    (MANZ, None, "Funda Rockit 2 Lb", 385, "Manzanas miniatura variedad Rockit en funda (907 gr)", FRU),
    # ── Aceite vegetal (food nuevo) ──
    ("Aceite vegetal", "Wesson", "Botella 64 Oz", 535, "Aceite vegetal (vegetable oil)", GRASA),
    ("Aceite vegetal", "Wesson", "Botella 48 Oz", 425, "Aceite vegetal (vegetable oil)", GRASA),
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
    brands = {r[1] for r in ROWS if r[1]}
    print(f"Seed manzana + aceite vegetal: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
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
