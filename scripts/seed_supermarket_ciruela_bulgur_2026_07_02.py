"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de CIRUELAS PASAS
+ BULGUR.

Familias 111-112 del Supermercado RD (capturas del owner, 2026-07-02,
La Sirena).

CIRUELA PASA (5):
  * CALCE: el clamshell "Ciruela Pasa Sin Semilla 16 Oz" RD$199 = genérico
    "Tarro 16 Oz" (es el genérico, no duplicar). FLAG al owner: el
    genérico de fruta fresca "Ciruela · Paquete 16 Oz · RD$199" tiene
    precio idéntico — probable doble-entrada del PDF desde el mismo
    listing de pasas.
  * Sunsweet ×3 (sin hueso 8 Oz, bite size 8 Oz, family size 16 Oz),
    Sun-Maid sin hueso 7 Oz, Dynasty doypack 7 Oz.

BULGUR (2):
  * CALCE: Goya Bulgur Kiepe 24 Oz RD$255 = genérico "Paquete 24 Oz".
  * + Goya Kiepe fino 24 Oz RD$260.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_ciruela_bulgur_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_ciruela_bulgur_2026_07_02.py --commit   # inserta
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
FRUTAS = "Frutas"
GRANOS = "Granos y cereales"

PASA = "Ciruela pasa"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Ciruela pasa ──
    (PASA, "Dynasty", "Funda 7 Oz", 225, "Ciruelas pasas, doypack", FRUTAS),
    (PASA, "Sunsweet", "Funda Sin Hueso 8 Oz", 280, "Ciruelas pasas sin hueso (Amaz!n Prunes pitted)", FRUTAS),
    (PASA, "Sunsweet", "Funda Bite Size 8 Oz", 280, "Ciruelas pasas bite size (Amaz!n Prunes)", FRUTAS),
    (PASA, "Sunsweet", "Funda Family Size 16 Oz", 510, "Ciruelas pasas sin hueso, tamaño familiar", FRUTAS),
    (PASA, "Sun-Maid", "Funda Sin Hueso 7 Oz", 250, "Ciruelas pasas California enteras sin hueso (whole pitted prunes)", FRUTAS),
    # ── Bulgur ──
    ("Bulgur", "Goya", "Funda Kiepe 24 Oz", 255, "Trigo bulgur grueso (bulgur wheat kiepe-blé)", GRANOS),
    ("Bulgur", "Goya", "Funda Kiepe Fino 24 Oz", 260, "Trigo bulgur fino (bulgur wheat kiepe-blé)", GRANOS),
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
    print(f"Seed ciruela pasa + bulgur: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
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
