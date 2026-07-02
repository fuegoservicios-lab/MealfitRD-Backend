"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de CANELA EN POLVO.

Tercera familia con variantes de MARCA del Supermercado RD: 4 SKUs transcritos del
catálogo de La Sirena (capturas del owner, 2026-07-02) — Badia molida (frasco 2/16 Oz,
sobre 0.5 Oz) y orgánica (2 Oz).

EXCLUIDO a propósito: "Polvo Compacto Repuesto Vogue Canela" (maquillaje que el search
de La Sirena devuelve por el color "canela" — no es alimento).

Mismo patrón que seed_supermarket_leches_2026_07_02.py. Idempotente: ON CONFLICT
(variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_canela_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_canela_2026_07_02.py --commit   # inserta
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
FOOD = "Canela en polvo"
CATEGORY = "Condimentos y especias"

# (brand, presentation, price_rd, description)
ROWS = [
    ("Badia", "Frasco 2 Oz", 105, "Canela en polvo (cinnamon powder)"),
    ("Badia", "Frasco 16 Oz", 555, "Canela en polvo, tamaño grande"),
    ("Badia", "Sobre 0.5 Oz", 55, "Canela molida, presentación de sobre, gluten free"),
    ("Badia", "Frasco Orgánica 2 Oz", 169, "Canela en polvo orgánica certificada"),
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

    print(f"Seed canela en polvo: {len(ROWS)} SKUs (Badia).")
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
