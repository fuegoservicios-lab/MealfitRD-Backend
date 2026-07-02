"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de FRESA.

Vigesimoctava familia con variantes de MARCA del Supermercado RD: 8 SKUs transcritos
del catálogo de La Sirena (capturas del owner, 2026-07-02) — frescas (Purama, Giugni,
Arroyo Frío premium) y congeladas (Arroyo Frío 1.5/4 Lb, Vima 1 Lb/1 Kg, Pisca 2 Lb),
todas bajo el food "Fresa" con presentación honesta (criterio habichuelas: mismo food,
formato en presentation).

TRIPLE calce exacto con los genéricos del PDF (que eran estos mismos productos
congelados; el "L" del PDF era "Lb"):
  * Paquete 1 L RD$165  = Vima Congeladas 1 Lb.
  * Paquete 1.5L RD$299 = Arroyo Frío Congeladas 1.5 Lb.
  * Paquete 4L RD$699   = Arroyo Frío Congeladas 4 Lb.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_fresa_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_fresa_2026_07_02.py --commit   # inserta
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
FOOD = "Fresa"
CATEGORY = "Frutas"

# (brand, presentation, price_rd, description)
ROWS = [
    # ── Frescas ──
    ("Purama", "Paquete Selectas 1 Lb", 235, "Fresas frescas selectas"),
    ("Giugni", "Paquete Selectas 0.75 Lb", 155, "Fresas frescas selectas"),
    ("Arroyo Frío", "Paquete Premium Frescas", 419, "Fresas frescas, premium pack"),
    # ── Congeladas ──
    ("Arroyo Frío", "Funda Congeladas 1.5 Lb", 299, "Fresas congeladas (caperucitas)"),
    ("Arroyo Frío", "Funda Congeladas 4 Lb", 699, "Fresas congeladas (caperucitas), tamaño grande"),
    ("Vima", "Funda Congeladas 1 Lb (450 gr)", 165, "Fresas congeladas 100% naturales"),
    ("Vima", "Funda Congeladas 1 Kg", 325, "Fresas congeladas"),
    ("Pisca", "Funda Congeladas 2 Lb", 280, "Fresas congeladas"),
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
    print(f"Seed fresa: {len(ROWS)} SKUs · {len(brands)} marcas.")
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
