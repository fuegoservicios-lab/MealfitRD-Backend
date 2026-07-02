"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de CEBOLLA y BRÓCOLI.

Familias 38-39 del Supermercado RD: 8 SKUs transcritos del catálogo de La Sirena
(capturas del owner, 2026-07-02):

  * "Cebolla" (7): tipos frescos como variantes sin marca (roja criolla, amarilla,
    colosal amarilla, blanca colosal, roja colosal — criterio tipos de tomate) +
    las cebollas perla en malla 8 Oz (roja de I Love Produce; la blanca sin marca
    legible en el empaque).
  * "Brócoli" (1): orgánico importado por paquete (sin marca legible en la banda).

DOBLE calce exacto con genéricos del PDF: "Cebolla" Lb RD$47 = roja criolla Y
amarilla; "Brócoli" Lb RD$62 = brócoli fresco Lb (no se duplica, ES el genérico).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_cebolla_brocoli_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_cebolla_brocoli_2026_07_02.py --commit   # inserta
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

CEBOLLA = "Cebolla"
BROCOLI = "Brócoli"

# (food_name, brand, presentation, price_rd, description)
ROWS = [
    # ── Cebolla ──
    (CEBOLLA, None, "Roja Criolla Lb", 47, "Cebolla roja criolla fresca, por libra"),
    (CEBOLLA, None, "Amarilla Lb", 47, "Cebolla amarilla fresca, por libra"),
    (CEBOLLA, None, "Colosal Amarilla Lb", 73, "Cebolla amarilla colosal (tamaño grande), por libra"),
    (CEBOLLA, None, "Blanca Colosal Lb", 99, "Cebolla blanca colosal, por libra"),
    (CEBOLLA, None, "Roja Colosal Lb", 116, "Cebolla roja colosal, por libra"),
    (CEBOLLA, "I Love Produce", "Malla Perla Roja 8 Oz", 250, "Cebollas perla rojas en malla"),
    (CEBOLLA, None, "Malla Perla Blanca 8 Oz", 250, "Cebollas perla blancas en malla"),
    # ── Brócoli ──
    (BROCOLI, None, "Orgánico (paquete)", 356, "Brócoli orgánico importado, por paquete"),
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
    print(f"Seed cebolla + brócoli: {len(ROWS)} SKUs · {len(foods)} foods.")
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
