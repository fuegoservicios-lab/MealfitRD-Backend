"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de FILETE PECHUGA DE POLLO.

Trigesimosegunda familia con variantes de MARCA del Supermercado RD: 6 SKUs nuevos
transcritos del catálogo de La Sirena (captura del owner, 2026-07-02) — Don Pollo
(congelado/fresca/granel), Pollo Cibao (fresca/congelado) y la empacada fresca
sin marca.

Notas:
  * El genérico del PDF calza exacto: Lb RD$135 = "Filete Pechuga De Pollo Importada
    Congelada Lb" — ese listado ES el genérico ya cargado, no se duplica.
  * "Filete De Pechuga Congelado Don Pollo": el título de La Sirena dice Don Pollo
    pero la FOTO muestra bandeja Unipollo — se registra por el título (en esta
    familia 4/7 fotos son stock genérico, el título es el dato de inventario).
    Si el owner confirma que es Unipollo, corregir marca via admin UI.
  * Fresca y congelada bajo el MISMO food con presentación honesta (criterio fresa).
  * 3 "No disponible" incluidos (Don Pollo fresca/granel, empacada sin marca).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_pechuga_pollo_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_pechuga_pollo_2026_07_02.py --commit   # inserta
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
FOOD = "Filete pechuga de pollo"
CATEGORY = "Carnes, pescados y mariscos"

# (brand, presentation, price_rd, description)
ROWS = [
    ("Don Pollo", "Bandeja Congelado Lb", 172, "Filete de pechuga de pollo congelado, por libra"),
    ("Don Pollo", "Bandeja Fresca Lb", 179, "Filete de pechuga de pollo fresca, por libra"),
    ("Don Pollo", "Granel Fresca Lb", 175, "Filete de pechuga de pollo fresca a granel, por libra"),
    ("Pollo Cibao", "Bandeja Lb", 179, "Filete de pechuga de pollo, por libra"),
    ("Pollo Cibao", "Bandeja Congelado Lb", 185, "Filete de pechuga de pollo congelado, por libra"),
    (None, "Empacada Fresca Lb", 179, "Filete de pechuga de pollo fresco empacado, por libra"),
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

    brands = {r[0] for r in ROWS if r[0]}
    print(f"Seed filete pechuga de pollo: {len(ROWS)} SKUs · {len(brands)} marcas.")
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
