"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de HABICHUELAS ROJAS.

Undécima familia con variantes de MARCA del Supermercado RD: 12 SKUs transcritos del
catálogo de La Sirena (capturas del owner, 2026-07-02) — secas en funda (Wala, Giselle,
La Sanjuanera roja larga) y enlatadas/tetra (Wala, La Famosa incl. con coco, Linda,
Goya red kidney, Rica, Victorina).

Los genéricos del PDF calzan exacto: Funda 800 gr RD$129 = Wala seca; Lata 15 Oz
RD$50 = Wala enlatada.

Mismo patrón que seed_supermarket_leches_2026_07_02.py. Idempotente: ON CONFLICT
(variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_habichuelas_rojas_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_habichuelas_rojas_2026_07_02.py --commit   # inserta
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
FOOD = "Habichuelas rojas"
CATEGORY = "Legumbres y proteína vegetal"

# (brand, presentation, price_rd, description)
ROWS = [
    # ── Secas (funda) ──
    ("Wala", "Funda 800 gr", 129, "Habichuelas rojas secas"),
    ("Giselle", "Funda 800 gr", 135, "Habichuelas rojas secas"),
    ("Giselle", "Funda 400 gr", 91, "Habichuelas rojas secas"),
    ("La Sanjuanera", "Funda Roja Larga 800 gr", 165, "Habichuelas rojas largas premium, secas"),
    ("La Sanjuanera", "Funda Roja Larga 400 gr", 89, "Habichuelas rojas largas premium, secas"),
    # ── Enlatadas / tetra ──
    ("Wala", "Lata 15 Oz", 50, "Habichuelas rojas enlatadas (400 gr, 240 gr drenado)"),
    ("La Famosa", "Lata 15 Oz", 88, "Habichuelas rojas enlatadas (red beans)"),
    ("La Famosa", "Lata con Coco 15 Oz", 138, "Habichuelas rojas guisadas con coco"),
    ("Linda", "Lata 15 Oz", 60, "Habichuelas rojas enlatadas"),
    ("Goya", "Lata Red Kidney 15.5 Oz", 85, "Habichuelas coloradas (red kidney beans), prime premium"),
    ("Rica", "Cartón Tetra 400 gr", 78, "Habichuelas rojas en tetra pack — fácil de abrir y guardar"),
    ("Victorina", "Lata 15 Oz", 85, "Habichuelas rojas enlatadas (red beans)"),
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
    print(f"Seed habichuelas rojas: {len(ROWS)} SKUs · {len(brands)} marcas.")
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
