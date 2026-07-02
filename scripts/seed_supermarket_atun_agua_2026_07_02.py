"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de ATÚN EN AGUA.

Decimoséptima familia con variantes de MARCA del Supermercado RD: 14 SKUs transcritos
del catálogo de La Sirena (capturas del owner, 2026-07-02) — Wala, Brunswick, Cherry
Star, Zerca, Goya, Paco Fish (trozos/sólido/desmenuzado), Dimar, Bumble Bee y el
premium Tonnino en frasco (filetes en agua; su par EN ACEITE ya vive bajo el food
"Atún en aceite" del seed de orégano).

Notas:
  * El genérico del PDF calza exacto: Lata 170 gr RD$60 = Zerca desmenuzado.
  * Cherry Star trozos 5 Oz aparece DOS veces en La Sirena (RD$110 en promo -19% y
    RD$105 plano) — se carga UNA vez con el precio plano RD$105.
  * Incluidos 2 "No disponible" (Bumble Bee, Dimar trozos) como referencia.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_atun_agua_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_atun_agua_2026_07_02.py --commit   # inserta
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
FOOD = "Atún en agua"
CATEGORY = "Carnes, pescados y mariscos"

# (brand, presentation, price_rd, description)
ROWS = [
    ("Zerca", "Lata Desmenuzado 170 gr", 60, "Atún desmenuzado en agua, con Omega 3"),
    ("Wala", "Lata en Trozos 170 gr", 98, "Atún en trozos en agua"),
    ("Brunswick", "Lata Trozos 142 gr", 105, "Atún en trozos en agua, 26g de proteína"),
    ("Brunswick", "Lata Desmenuzado 142 gr", 69, "Atún desmenuzado en agua"),
    ("Cherry Star", "Lata Trozos 5 Oz", 105, "Atún en trozos en agua"),
    ("Cherry Star", "Lata Sólido 5 Oz", 139, "Atún sólido en agua"),
    ("Goya", "Lata Lomo en Trozos 140 gr", 185, "Lomo de atún en trozos en agua, pescado en altamar"),
    ("Paco Fish", "Lata Trozos 5 Oz", 119, "Atún en trozos en agua"),
    ("Paco Fish", "Lata Sólido 5 Oz", 125, "Atún sólido en agua"),
    ("Paco Fish", "Lata Desmenuzado 5 Oz", 69, "Atún desmenuzado (flaked) en agua"),
    ("Dimar", "Lata Desmenuzado 142 gr", 79, "Atún desmenuzado en agua"),
    ("Dimar", "Lata Trozos 142 gr", 120, "Atún en trozos en agua"),
    ("Bumble Bee", "Lata Chunk Light 5 Oz", 126, "Atún chunk light en agua, 23g de proteína"),
    ("Tonnino", "Frasco Filetes en Agua 190 gr", 469, "Filetes de atún en agua (spring water), frasco de vidrio"),
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
    print(f"Seed atún en agua: {len(ROWS)} SKUs · {len(brands)} marcas.")
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
