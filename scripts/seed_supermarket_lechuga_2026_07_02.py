"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de LECHUGA.

Trigesimocuarta familia con variantes de MARCA del Supermercado RD: 9 SKUs nuevos
transcritos del catálogo de La Sirena (capturas del owner, 2026-07-02), repartidos
en los DOS foods que ya existen del PDF:

  * "Lechuga" (6): dulce criolla, iceberg cello importada, artisan Tanimura & Antle
    (4 cabezas), ensalada baby sweet orgánica, y las Boston hidropónicas de Lucas
    Pérez (funda + clamshell). Variedades de hoja bajo el mismo food con
    presentación honesta (sustitutos razonables en ensalada).
  * "Lechuga romana" (3): Andy Boy (corazones orgánicos + jumbo hearts) y Lucas
    Pérez corazón de romana.

Notas:
  * DOBLE calce exacto con genéricos del PDF: "Lechuga" Lb RD$48 = Lechuga
    Repollada Lb (no se duplica, ES el genérico); "Lechuga romana" Paquete
    RD$158 = Lucas Pérez Corazón de Romana.
  * La ensalada baby sweet orgánica no muestra marca legible — se carga sin marca.
  * 3 "No disponible" incluidos (romana Lucas Pérez, Boston funda y clamshell).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_lechuga_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_lechuga_2026_07_02.py --commit   # inserta
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
CATEGORY = "Vegetales y verduras"

LECHUGA = "Lechuga"
ROMANA = "Lechuga romana"

# (food_name, brand, presentation, price_rd, description)
ROWS = [
    # ── Lechuga ──
    (LECHUGA, None, "Dulce (unidad)", 37, "Lechuga dulce criolla, por unidad"),
    (LECHUGA, None, "Iceberg Cello Importada (unidad)", 135, "Lechuga repollada iceberg importada, envuelta en celofán"),
    (LECHUGA, "Tanimura & Antle", "Bandeja Artisan 4 unid", 395, "Lechugas artisan importadas, 4 cabezas mixtas"),
    (LECHUGA, None, "Clamshell Ensalada Baby Sweet Orgánica 5 Oz", 310, "Mix orgánico de lechugas baby sweet"),
    (LECHUGA, "Lucas Pérez", "Funda Boston Hidropónica", 85, "Lechuga Boston hidropónica"),
    (LECHUGA, "Lucas Pérez", "Clamshell Boston Hidropónica", 139, "Lechuga Boston hidropónica, envase clamshell"),
    # ── Lechuga romana ──
    (ROMANA, "Andy Boy", "Paquete Orgánico Corazones de Romana 3 unid", 350, "Corazones de lechuga romana orgánica importada"),
    (ROMANA, "Andy Boy", "Paquete Jumbo Hearts 3 unid", 377, "Corazones jumbo de lechuga romana importada"),
    (ROMANA, "Lucas Pérez", "Funda Corazón de Romana", 158, "Corazones de lechuga romana"),
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

    foods = {r[0] for r in ROWS}
    brands = {r[1] for r in ROWS if r[1]}
    print(f"Seed lechuga: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
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
