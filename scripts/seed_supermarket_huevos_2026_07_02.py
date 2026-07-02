"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de HUEVOS.

Decimosexta familia con variantes de MARCA del Supermercado RD: 19 SKUs transcritos
del catálogo de La Sirena (capturas del owner, 2026-07-02) — Endy (8: blancos/marrones/
jumbo/omega-3, de 6 a 30 uds), Don Papito (8: grandes/extra/marrones/supremo/gallinas
libres, de 6 a 30 uds), Free Farm (pastoreo) y Wala (30 uds, "No disponible" pero
catálogo real).

"Huevos de codorniz" (Don Papito 24 uds) va como food propio — producto distinto,
no sustituto del huevo de gallina en la lista de compras.

Deduplicado: los Huevos Grandes Wala 30 und aparecen CUATRO veces en La Sirena
(mismo precio RD$209) — se carga una sola vez.

Los genéricos del PDF calzan exacto: Cartón 20 unid RD$200 = Endy Blancos Jumbo 20;
Cartón 30 unid RD$295 = Endy Extra Grande 30.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_huevos_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_huevos_2026_07_02.py --commit   # inserta
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
CATEGORY = "Lácteos y huevos"

# (food_name, brand, presentation, price_rd, description)
ROWS = [
    # ── Endy ──
    ("Huevos", "Endy", "Cartón Extra Grande 30 unid", 295, "Huevos blancos tamaño extra grande"),
    ("Huevos", "Endy", "Cartón Extra Grande 18 unid", 189, "Huevos blancos tamaño extra grande"),
    ("Huevos", "Endy", "Cartón Frescos 12 unid", 129, "Huevos blancos frescos"),
    ("Huevos", "Endy", "Cartón Marrones 30 unid", 329, "Huevos marrones"),
    ("Huevos", "Endy", "Cartón Jumbo Blanco 12 unid", 130, "Huevos blancos tamaño jumbo"),
    ("Huevos", "Endy", "Cartón Blancos Jumbo 20 unid", 200, "Huevos blancos tamaño jumbo"),
    ("Huevos", "Endy", "Cartón Extra Grandes Blanco 6 unid", 65, "Huevos blancos extra grandes"),
    ("Huevos", "Endy", "Cartón Omega 3 12 unid", 150, "Huevos blancos enriquecidos con Omega 3"),
    # ── Don Papito ──
    ("Huevos", "Don Papito", "Cartón Grandes 30 unid", 280, "Huevos blancos grandes"),
    ("Huevos", "Don Papito", "Cartón Marrones 30 unid", 295, "Huevos marrones"),
    ("Huevos", "Don Papito", "Cartón Extra Grande 20 unid", 205, "Huevos blancos extra grandes"),
    ("Huevos", "Don Papito", "Cartón Gallinas Libres 12 unid", 175, "Huevos de gallinas libres de jaula, grado A"),
    ("Huevos", "Don Papito", "Cartón Grandes 18 unid", 180, "Huevos blancos grandes"),
    ("Huevos", "Don Papito", "Cartón Grandes 12 unid", 125, "Huevos blancos grandes"),
    ("Huevos", "Don Papito", "Cartón Supremo 12 unid", 130, "Huevos Supremo"),
    ("Huevos", "Don Papito", "Cartón Grandes 6 unid", 70, "Huevos blancos grandes"),
    # ── Otras marcas ──
    ("Huevos", "Free Farm", "Cartón de Pastoreo 12 unid", 170, "Huevos de gallinas de pastoreo (free range)"),
    ("Huevos", "Wala", "Cartón Grandes 30 unid", 209, "Huevos blancos grandes"),
    # ── Codorniz (food propio) ──
    ("Huevos de codorniz", "Don Papito", "Bandeja 24 unid", 115, "Huevos de codorniz frescos"),
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

    brands = {r[1] for r in ROWS}
    print(f"Seed huevos: {len(ROWS)} SKUs · {len(brands)} marcas.")
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
