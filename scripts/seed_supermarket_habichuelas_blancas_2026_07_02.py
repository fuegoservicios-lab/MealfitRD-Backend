"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de HABICHUELAS BLANCAS.

Duodécima familia con variantes de MARCA del Supermercado RD: 8 SKUs transcritos del
catálogo de La Sirena (capturas del owner, 2026-07-02) — secas en funda (Wala, Giselle,
La Sanjuanera) y enlatadas (Wala, La Famosa, Goya guisadas, Linda, Goya small white).

Los genéricos del PDF calzan exacto: Funda 800 gr RD$115 = Wala seca; Lata 15 Oz
RD$50 = Wala enlatada. Se incluyen 2 SKUs "No disponible" en La Sirena (Linda, Goya
pequeña) — catálogo real con precio de referencia.

Mismo patrón que seed_supermarket_leches_2026_07_02.py. Idempotente: ON CONFLICT
(variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_habichuelas_blancas_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_habichuelas_blancas_2026_07_02.py --commit   # inserta
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
FOOD = "Habichuelas blancas"
CATEGORY = "Legumbres y proteína vegetal"

# (brand, presentation, price_rd, description)
ROWS = [
    # ── Secas (funda) ──
    ("Wala", "Funda 800 gr", 115, "Habichuelas blancas secas"),
    ("Giselle", "Funda 800 gr", 125, "Habichuelas blancas secas"),
    ("La Sanjuanera", "Funda 800 gr", 122, "Habichuelas blancas premium, secas"),
    # ── Enlatadas ──
    ("Wala", "Lata 15 Oz", 50, "Habichuelas blancas enlatadas (400 gr, 240 gr drenado)"),
    ("La Famosa", "Lata 15 Oz", 88, "Habichuelas blancas enlatadas"),
    ("Goya", "Lata Guisadas 15 Oz", 91, "Habichuelas blancas guisadas con aceite de oliva, recao y tomate"),
    ("Goya", "Lata Small White Pequeña 15.5 Oz", 85, "Habichuelas blancas pequeñas (small white beans), prime premium"),
    ("Linda", "Lata 15 Oz", 69, "Habichuelas blancas enlatadas"),
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
    print(f"Seed habichuelas blancas: {len(ROWS)} SKUs · {len(brands)} marcas.")
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
