"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de TORTILLA DE TRIGO.

Vigesimoprimera familia con variantes de MARCA del Supermercado RD: 9 SKUs transcritos
del catálogo de La Sirena (capturas del owner, 2026-07-02) — María (4: clásicas 10/1,
wraps 8/1, multigrano, burrito 5/1), Wala (3: tradicional, fajita, burrito), El Charrito
y Toufayan.

Notas:
  * El genérico del PDF calza exacto: Paquete 10 unid RD$74 = Tortillas María Trigo 10/1.
  * Wala verde: La Sirena lo titula "Burrito 8 Und" pero el empaque dice FAJITA (8 unid)
    — se registra por el empaque (mismo criterio que PB&Co en mantequilla de maní).
    El Wala Burrito real es el de 5 unid / 356 gr RD$95.
  * Toufayan "Wraps Wholesome Wheat" es trigo integral → va bajo el food "Tortilla
    integral" (mismo criterio que Bolin Integral Blanco en pan blanco: la verdad del
    producto gana sobre el bucket del search).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_tortilla_trigo_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_tortilla_trigo_2026_07_02.py --commit   # inserta
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
CATEGORY = "Panadería y harinas"

TRIGO = "Tortilla de Trigo"
INTEGRAL = "Tortilla integral"

# (food_name, brand, presentation, price_rd, description)
ROWS = [
    # ── María ──
    (TRIGO, "María", "Paquete 10 unid", 74, "Tortillas de harina de trigo (flour tortillas)"),
    (TRIGO, "María", "Paquete Para Wraps 8 unid", 103, "Tortillas de trigo para wraps"),
    (TRIGO, "María", "Paquete Multigrano 8 unid", 120, "Tortillas de trigo multigrano (multi-grain)"),
    (TRIGO, "María", "Paquete Para Burrito 5 unid", 107, "Tortillas de trigo tamaño burrito, para wraps"),
    # ── Wala ──
    (TRIGO, "Wala", "Paquete Tradicional 10 unid", 68, "Tortillas de trigo tradicionales"),
    (TRIGO, "Wala", "Paquete Fajita 8 unid", 90, "Tortillas de trigo tamaño fajita"),
    (TRIGO, "Wala", "Paquete Burrito 5 unid 356 gr", 95, "Tortillas de trigo tamaño burrito"),
    # ── El Charrito ──
    (TRIGO, "El Charrito", "Paquete Pequeñas 10 unid", 72, "Tortillas de trigo pequeñas"),
    # ── Toufayan (trigo integral → food propio ya existente) ──
    (INTEGRAL, "Toufayan", "Paquete Wraps Wholesome Wheat 6 unid 11 Oz", 150, "Wraps de trigo integral (wholesome wheat), horneados"),
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
    print(f"Seed tortilla de trigo: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
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
