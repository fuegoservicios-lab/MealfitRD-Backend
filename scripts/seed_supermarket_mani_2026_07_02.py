"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de MANÍ.

Familia 76 del Supermercado RD: 18 SKUs transcritos del catálogo de La Sirena
(capturas del owner, 2026-07-02):

  * Wala (8): los potes Con Sal y Sin Sal 300 gr a RD$185 — AMBOS CALZAN
    EXACTO con el genérico "Pote 300 gr"; limón picante, con pasas (pote y
    funda), fundas 100 gr (con sal, japonés) y la funda cruda 800 gr.
  * Essential Everyday (5, normalizado desde "Essential"/"Esencial"): dry
    roasted con sal / bajo en sal / sin sal 16 Oz, honey roasted tarro 12 Oz
    y frasco 16 Oz ("No disponible" — se carga por TÍTULO; la foto del
    listing parece reciclada del sin sal).
  * Nut Walker (3, latas): honey y BBQ 320 gr, roasted salted 150 gr.
  * Planters cocktail lightly salted 16 Oz y Cashitas con pasas 55 gr.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_mani_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_mani_2026_07_02.py --commit   # inserta
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
SEM = "Semillas y frutos secos"

MANI = "Maní"

# (food_name, brand, presentation, price_rd, description)
ROWS = [
    # ── Wala ──
    (MANI, "Wala", "Pote Con Sal 300 gr", 185, "Maní tostado con sal"),
    (MANI, "Wala", "Pote Sin Sal 300 gr", 185, "Maní tostado sin sal, 100% natural bajo en sodio"),
    (MANI, "Wala", "Pote Limón Picante 300 gr", 210, "Maní tostado sabor limón picante"),
    (MANI, "Wala", "Pote Con Pasas 300 gr", 210, "Maní tostado con pasas"),
    (MANI, "Wala", "Funda Con Sal 100 gr", 65, "Maní tostado con sal"),
    (MANI, "Wala", "Funda Con Pasas 100 gr", 75, "Maní tostado con pasas"),
    (MANI, "Wala", "Funda Japonés 100 gr", 75, "Maní japonés (cubierto crocante)"),
    (MANI, "Wala", "Funda 800 gr", 173, "Maní crudo, sin preservativos"),
    # ── Essential Everyday ──
    (MANI, "Essential Everyday", "Frasco Dry Roasted Con Sal 16 Oz", 239, "Maní tostado en seco con sal"),
    (MANI, "Essential Everyday", "Frasco Dry Roasted Bajo en Sal 16 Oz", 259, "Maní tostado en seco ligeramente salado (lightly salted)"),
    (MANI, "Essential Everyday", "Frasco Dry Roasted Sin Sal 16 Oz", 249, "Maní tostado en seco sin sal (unsalted)"),
    (MANI, "Essential Everyday", "Tarro Honey Roasted 12 Oz", 285, "Maní tostado con miel"),
    (MANI, "Essential Everyday", "Frasco Honey Roasted 16 Oz", 259, "Maní tostado con miel"),
    # ── Nut Walker ──
    (MANI, "Nut Walker", "Lata Honey 320 gr", 324, "Maní tostado con miel"),
    (MANI, "Nut Walker", "Lata BBQ 320 gr", 324, "Maní tostado sabor barbecue"),
    (MANI, "Nut Walker", "Lata Roasted Salted 150 gr", 159, "Maní tostado con sal"),
    # ── Otras marcas ──
    (MANI, "Planters", "Lata Cocktail Lightly Salted 16 Oz", 380, "Maní cocktail ligeramente salado"),
    (MANI, "Cashitas", "Funda Con Pasas 55 gr", 48, "Maní con pasas, snack"),
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
        if len(row) != 5:
            print(f"FATAL: fila con {len(row)} campos (esperados 5): {row[:3]}")
            sys.exit(1)
        (food, brand, pres, *_rest) = row
        key = (food.lower(), (brand or "").lower(), pres.lower())
        if key in seen:
            print(f"FATAL: fila duplicada en el seed: {food} / {brand} / {pres}")
            sys.exit(1)
        seen.add(key)

    foods = {r[0] for r in ROWS}
    brands = {r[1] for r in ROWS if r[1]}
    print(f"Seed maní: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, brand, pres, price, desc) in ROWS:
                cur.execute(_INSERT, (food, brand, pres, price, NOTES, SEM, food, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
