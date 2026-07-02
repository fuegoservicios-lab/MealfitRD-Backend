"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de MIEL.

Sexta familia con variantes de MARCA del Supermercado RD: 12 SKUs transcritos del
catálogo de La Sirena (capturas del owner, 2026-07-02) — Divina Miel, Queen Bee,
Delifruit, Del Campo, Miel de la Abuela, Cohen.

EXCLUIDOS a propósito: jabones de tocador "miel" (Lavador multipack/4-pack, Kinder
Naturals) y la miel bronquial Alopecil (compuesto balsámico medicinal con claim de
broncodilatador — no es alimento de cocina; si el owner la quiere, se agrega por
admin UI).

Nota de fidelidad: el "Syrup Miel Del Campo" se registra como syrup (jarabe sabor
miel), NO como miel pura — descripción honesta para el futuro selector de la lista
de compras.

Mismo patrón que seed_supermarket_leches_2026_07_02.py. Idempotente: ON CONFLICT
(variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_miel_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_miel_2026_07_02.py --commit   # inserta
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
FOOD = "Miel"
CATEGORY = "Otros"

# (brand, presentation, price_rd, description)
ROWS = [
    ("Divina Miel", "Botella 8 Oz", 114, "Miel pura de abeja — del panal a la botella"),
    ("Divina Miel", "Botella 16 Oz", 205, "Miel pura de abeja — del panal a la botella"),
    ("Queen Bee", "Botella Squeeze 16 Oz", 220, "Miel pura 100% natural, calidad superior"),
    ("Queen Bee", "Botella Squeeze 35 Oz", 430, "Miel pura 100% natural, tamaño grande"),
    ("Delifruit", "Botella Squeeze 12 Oz", 170, "Miel de abeja (honey)"),
    ("Delifruit", "Botella 175 Ml", 185, "Miel de abeja (honey)"),
    ("Del Campo", "Botella Syrup 16 Oz", 175, "Syrup de miel (jarabe sabor miel, no es miel pura)"),
    ("Del Campo", "Botella Premium 16 Oz", 260, "Miel premium"),
    ("Del Campo", "Botella Pura Multifloral 32 Oz", 465, "Miel de abeja pura multifloral"),
    ("Miel de la Abuela", "Botella 16 Oz", 235, "Miel de abeja 100% pura — el sabor de lo natural"),
    ("Miel de la Abuela", "Botella Bio-Power 8 Oz", 295, "Miel 100% natural con polen, propóleo y jalea real"),
    ("Cohen", "Botella 32 Oz", 455, "Miel de abejas pura"),
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
    print(f"Seed miel: {len(ROWS)} SKUs · {len(brands)} marcas.")
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
