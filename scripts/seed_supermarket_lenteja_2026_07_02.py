"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de LENTEJA.

Vigesimoquinta familia con variantes de MARCA del Supermercado RD: 8 SKUs transcritos
del catálogo de La Sirena (capturas del owner, 2026-07-02) — Giselle, Goya (funda seca
+ lata cocida), La Cochura (castellana seca + frasco cocidas), Wala y El Corte Inglés
(pardina IGP + plato con chorizo).

Mismo criterio que habichuelas: secas (funda) y cocidas (lata/frasco) bajo el MISMO
food "Lenteja", distinguidas por presentación — incluso el plato preparado con chorizo
queda en familia con descripción honesta (precedente La Famosa "con Coco").

Notas:
  * DOBLE calce exacto con los genéricos del PDF: Lata 15.5 Oz RD$95 = Goya lata;
    Paquete 800gr RD$114 = Wala funda.
  * 4 "No disponible" incluidos (Wala, La Cochura castellana, ambos El Corte Inglés).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_lenteja_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_lenteja_2026_07_02.py --commit   # inserta
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
FOOD = "Lenteja"
CATEGORY = "Legumbres y proteína vegetal"

# (brand, presentation, price_rd, description)
ROWS = [
    ("Giselle", "Funda 400 gr", 89, "Lentejas secas"),
    ("Goya", "Funda 16 Oz", 129, "Lentejas secas (masoor dal)"),
    ("Goya", "Lata 15.5 Oz", 95, "Lentejas cocidas, línea Prime Premium"),
    ("La Cochura", "Frasco Cocidas Extra 540 gr", 175, "Lentejas extra cocidas al natural, frasco de vidrio"),
    ("La Cochura", "Funda Castellana 500 gr", 109, "Lenteja castellana seca"),
    ("Wala", "Funda 800 gr", 114, "Lentejas secas sin preservativos"),
    ("El Corte Inglés", "Lata con Chorizo 400 gr", 270, "Plato preparado de lentejas con chorizo (España)"),
    ("El Corte Inglés", "Frasco Pardina Tierra de Campos IGP 570 gr", 350, "Lenteja pardina cocida, Tierra de Campos IGP (España), línea Selection"),
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
    print(f"Seed lenteja: {len(ROWS)} SKUs · {len(brands)} marcas.")
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
