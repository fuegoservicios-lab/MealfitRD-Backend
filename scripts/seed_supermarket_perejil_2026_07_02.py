"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de PEREJIL.

Vigesimosegunda familia con variantes de MARCA del Supermercado RD: 5 SKUs nuevos
transcritos del catálogo de La Sirena (capturas del owner, 2026-07-02):

  * "Perejil" (fresco, food ya existente del PDF — Vegetales y verduras):
    Trudeau Farms perejil chino/rizado ("No disponible", referencia).
    El listado "Perejil Por Paquete" RD$44 calza EXACTO con el genérico
    ya cargado — no se duplica.
  * "Perejil seco" (food NUEVO — hierba seca ≠ hierba fresca, mismo criterio
    que Orégano/Orégano fresco): Oriente sobre 8 gr, Badia sobre 0.25 Oz,
    Badia frasco 2 Oz ("No disponible").
  * "Ajo y perejil" (food NUEVO — blend sazón, no sustituto del perejil;
    mismo criterio que Sal de ajo): Badia molido 5 Oz.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_perejil_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_perejil_2026_07_02.py --commit   # inserta
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
VEG = "Vegetales y verduras"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Perejil fresco (food existente) ──
    ("Perejil", "Trudeau Farms", "Bandeja Perejil Chino (rizado)", 55, "Perejil chino (rizado) fresco", VEG),
    # ── Perejil seco (food nuevo) ──
    ("Perejil seco", "Oriente", "Sobre 8 gr", 33, "Perejil seco en hojas, presentación de sobre", COND),
    ("Perejil seco", "Badia", "Sobre Hojas 0.25 Oz", 50, "Perejil seco en hojas (parsley flakes), sobre", COND),
    ("Perejil seco", "Badia", "Frasco Hojas 2 Oz", 210, "Perejil seco en hojas (parsley flakes)", COND),
    # ── Ajo y perejil (blend, food nuevo) ──
    ("Ajo y perejil", "Badia", "Frasco Molido 5 Oz", 215, "Sazón molido de ajo y perejil (garlic & parsley)", COND),
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
    brands = {r[1] for r in ROWS}
    print(f"Seed perejil: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
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
