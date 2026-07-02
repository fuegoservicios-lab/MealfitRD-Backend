"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de CILANTRO.

Trigesimoquinta familia con variantes de MARCA del Supermercado RD: 3 SKUs nuevos
transcritos del catálogo de La Sirena (captura del owner, 2026-07-02):

  * "Cilantro ancho" (food NUEVO): el culantro/recao fresco — planta distinta al
    cilantrico (Eryngium foetidum vs Coriandrum sativum), no sustituto directo.
  * "Cilantro seco" (food NUEVO, criterio perejil seco): Oriente sobre 13 gr
    (SEMILLAS enteras de cilantro/coriandro — la foto muestra semillas) y Badia
    sobre 0.25 Oz (HOJAS secas). Presentación honesta distingue semillas vs hojas.

El genérico del PDF calza exacto: "Cilantro" Paquete RD$48 = Cilantrico Por
Paquete (fresco) — no se duplica, ES el genérico ya cargado.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_cilantro_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_cilantro_2026_07_02.py --commit   # inserta
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
VEG = "Vegetales y verduras"
COND = "Condimentos y especias"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    ("Cilantro ancho", None, "Paquete", 48, "Cilantro ancho (culantro/recao) fresco, por paquete", VEG),
    ("Cilantro seco", "Oriente", "Sobre Semillas 13 gr", 43, "Semillas de cilantro enteras (coriandro)", COND),
    ("Cilantro seco", "Badia", "Sobre Hojas 0.25 Oz", 50, "Cilantro seco en hojas", COND),
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
    print(f"Seed cilantro: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
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
