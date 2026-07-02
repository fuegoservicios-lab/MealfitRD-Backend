"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de FILETE DE PESCADO BLANCO (+tilapia).

Familia 48 del Supermercado RD: 3 SKUs transcritos de capturas del owner
(2026-07-02). PRIMERA familia con fuente MIXTA: 1 SKU de La Sirena + 2 del
supermercado NACIONAL (Centro Cuesta) — la nota de precio va POR FILA.

  * "Filete de pescado blanco" (2): Panamei empanizado 32 Oz (La Sirena; mismo
    tamaño que el genérico del PDF pero producto distinto — empanizado) y el
    filete de basa congelado al vacío por libra (Nacional).
  * "Tilapia" (1, food existente): filete premium congelado al vacío por libra
    (Nacional, marca de la casa "Ramón Corripio" — se registra sin marca).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_pescado_blanco_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_pescado_blanco_2026_07_02.py --commit   # inserta
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

SIRENA = "Precio de referencia La Sirena · 2026-07"
NACIONAL = "Precio de referencia Nacional · 2026-07"
CATEGORY = "Carnes, pescados y mariscos"

BLANCO = "Filete de pescado blanco"
TILAPIA = "Tilapia"

# (food_name, brand, presentation, price_rd, description, notes)
ROWS = [
    (BLANCO, "Panamei", "Funda Empanizado 32 Oz", 550, "Filetes de pescado empanizados, listos para cocinar (breaded fish fillets)", SIRENA),
    (BLANCO, None, "Filete de Basa Congelado al Vacío Lb", 239.95, "Filete de basa congelado empacado al vacío (peso aprox. 1 libra)", NACIONAL),
    (TILAPIA, None, "Filete Premium Congelado al Vacío Lb", 226.95, "Filete de tilapia premium congelado empacado al vacío (peso aprox. 1 libra)", NACIONAL),
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
    print(f"Seed pescado blanco: {len(ROWS)} SKUs · {len(foods)} foods.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, brand, pres, price, desc, notes) in ROWS:
                cur.execute(_INSERT, (food, brand, pres, price, notes, CATEGORY, food, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
