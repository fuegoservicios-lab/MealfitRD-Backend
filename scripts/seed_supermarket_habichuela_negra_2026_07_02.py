"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de HABICHUELA NEGRA.

Vigesimosexta familia con variantes de MARCA del Supermercado RD: 8 SKUs transcritos
del catálogo de La Sirena (capturas del owner, 2026-07-02) — Wala (funda + lata),
Goya, Giselle (400/800 gr), La Famosa (con coco), Linda y La Sanjuanera.

Mismo criterio que rojas/blancas/lenteja: secas (funda) y enlatadas (lata) bajo el
MISMO food "Habichuela negra" (singular, grafía del catálogo genérico).

Notas:
  * DOBLE calce exacto con los genéricos del PDF: Lata 15 Oz RD$88 = Goya;
    Paquete 800gr RD$105 = Wala funda.
  * Linda 15 Oz se carga a precio de LISTA RD$67 (promo -10% RD$60 transitoria).
  * La Famosa con coco queda en familia con descripción honesta (precedente rojas).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_habichuela_negra_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_habichuela_negra_2026_07_02.py --commit   # inserta
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
FOOD = "Habichuela negra"
CATEGORY = "Legumbres y proteína vegetal"

# (brand, presentation, price_rd, description)
ROWS = [
    ("Wala", "Funda 800 gr", 105, "Habichuelas negras secas sin preservativos"),
    ("Wala", "Lata 15 Oz", 50, "Habichuelas negras enlatadas (400 gr)"),
    ("Goya", "Lata 15 Oz", 88, "Habichuelas negras (black beans), línea Prime Premium"),
    ("Giselle", "Funda 800 gr", 126, "Habichuelas negras secas"),
    ("Giselle", "Funda 400 gr", 65, "Habichuelas negras secas"),
    ("La Famosa", "Lata con Coco 15 Oz", 138, "Habichuelas negras con coco (black beans with coconut)"),
    ("Linda", "Lata 15 Oz", 67, "Habichuelas negras enlatadas"),
    ("La Sanjuanera", "Funda 800 gr", 129, "Habichuelas negras secas, línea Premium"),
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
    print(f"Seed habichuela negra: {len(ROWS)} SKUs · {len(brands)} marcas.")
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
