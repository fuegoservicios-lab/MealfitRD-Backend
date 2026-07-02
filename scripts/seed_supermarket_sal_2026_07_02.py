"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de SAL.

Cuarta familia con variantes de MARCA del Supermercado RD: 33 SKUs transcritos del
catálogo de La Sirena (capturas del owner, 2026-07-02) — Refisal, Wala, Mosa, Morton,
Badia, Maldon, Essential Everyday. Agrupación:
  * "Sal" — refinada / marina / rosada Himalaya / kosher / parrillera / ligera (27 SKUs)
  * "Sal de ajo" / "Sal de apio" — condimentos Badia (foods propios: NO son sustituto
    de sal plana para la futura conexión con la lista de compras)
  * "Sal saborizada" — Badia gourmet (trufa negra, cítrica, sriracha)
  * BONUS: "Mantequilla" Sosúa barra con sal 113 gr (SKU legítimo que el search de
    "sal" devuelve; coincide con el genérico Barrita 113 gr RD$99 del PDF).

EXCLUIDOS a propósito (no-alimento): Sal Andrews (antiácido efervescente), sal de
Epsom Dr. Teal's / BioEva (baño/laxante), Garcita Chips y Platanitos Wala (snacks).

Mismo patrón que seed_supermarket_leches_2026_07_02.py. Idempotente: ON CONFLICT
(variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_sal_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_sal_2026_07_02.py --commit   # inserta
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

COND = "Condimentos y especias"
LACTEO = "Lácteos y huevos"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Sal refinada ──
    ("Sal", "Wala", "Funda 1 Lb", 17, "Sal refinada 100% yodada", COND),
    ("Sal", "Wala", "Funda 2 Lb", 40, "Sal refinada 100% yodada", COND),
    ("Sal", "Wala", "Frasco 432 gr", 39, "Sal refinada yodada", COND),
    ("Sal", "Wala", "Frasco 5 Lb", 185, "Sal refinada yodada, tamaño grande", COND),
    ("Sal", "Refisal", "Funda 1 Lb", 20, "Sal refinada alta pureza", COND),
    ("Sal", "Refisal", "Funda 908 gr", 35, "Sal refinada alta pureza, yodada y fluorizada", COND),
    ("Sal", "Refisal", "Salero 500 gr", 50, "Sal refinada alta pureza, grano fino", COND),
    ("Sal", "Refisal", "Tarro 5 Lb", 189, "Sal refinada alta pureza, tamaño grande", COND),
    ("Sal", "Mosa", "Funda 1 Lb", 34, "Sal refinada 100% yodada", COND),
    ("Sal", "Mosa", "Funda 2 Lb", 62, "Sal refinada de alta calidad", COND),
    ("Sal", "Mosa", "Salero 5 Oz", 28, "Sal refinada", COND),
    ("Sal", "Mosa", "Salero 454 gr", 54, "Sal refinada yodada", COND),
    ("Sal", "Morton", "Salero Yodada 26 Oz", 170, "Sal yodada (iodized)", COND),
    ("Sal", "Essential Everyday", "Salero Yodada 26 Oz", 110, "Sal yodada (iodized)", COND),
    # ── Sal marina ──
    ("Sal", "Refisal", "Salero Marina Cristal Fino 510 gr", 100, "Sal marina de origen natural, grano medio", COND),
    ("Sal", "Badia", "Frasco Marina Fina 10 Oz", 135, "Sal marina fina (fine sea salt)", COND),
    ("Sal", "Badia", "Frasco Marina Gruesa 9.5 Oz", 120, "Sal marina gruesa (coarse sea salt)", COND),
    ("Sal", "Morton", "Salero Marina Fina 17.6 Oz", 270, "Sal marina fina", COND),
    ("Sal", "Maldon", "Caja Escamas 125 gr", 305, "Escamas de sal marina (Inglaterra)", COND),
    # ── Sal rosada del Himalaya ──
    ("Sal", "Badia", "Caja Rosada Himalaya 8 Oz", 159, "Sal rosada del Himalaya", COND),
    ("Sal", "Badia", "Molinillo Rosada Himalaya 4.5 Oz", 185, "Sal rosada del Himalaya con molinillo (grinder)", COND),
    ("Sal", "Refisal", "Molinillo Rosada Himalaya 110 gr", 445, "Sal rosada del Himalaya con molino", COND),
    ("Sal", "Refisal", "Doy Pack Rosada Himalaya 400 gr", 610, "Sal rosada del Himalaya granulada, sal de origen", COND),
    # ── Especiales ──
    ("Sal", "Morton", "Caja Kosher 48 Oz", 319, "Sal kosher de grano grueso", COND),
    ("Sal", "Refisal", "Salero Parrillera 454 gr", 185, "Sal parrillera de grano grueso, para carnes y parrilla", COND),
    ("Sal", "Refisal", "Salero Vital Ligera 510 gr", 180, "Sal ligera 40% menos sodio", COND),
    ("Sal", "Refisal", "Funda Ligera en Sodio 908 gr", 159, "Sal light reducida en sodio, grano fino", COND),
    # ── Sales condimento (foods propios) ──
    ("Sal de ajo", "Badia", "Frasco 4.5 Oz", 115, "Sal de ajo (garlic salt)", COND),
    ("Sal de apio", "Badia", "Frasco 4.5 Oz", 105, "Sal de apio (celery salt)", COND),
    ("Sal saborizada", "Badia", "Frasco Trufa Negra 9 Oz", 545, "Sal marina con trufa negra (black truffle)", COND),
    ("Sal saborizada", "Badia", "Frasco Cítrica 7 Oz", 170, "Sal cítrica (citrus salt)", COND),
    ("Sal saborizada", "Badia", "Frasco Sriracha 8 Oz", 230, "Sal de sriracha", COND),
    # ── BONUS: mantequilla con sal que el search devolvió (SKU legítimo) ──
    ("Mantequilla", "Sosúa", "Barra con sal 113 gr", 99, "Mantequilla pasteurizada con sal", LACTEO),
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
    print(f"Seed sal: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
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
