"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de CAMARÓN.

Trigesimotercera familia con variantes de MARCA del Supermercado RD: 17 SKUs
transcritos del catálogo de La Sirena (capturas del owner, 2026-07-02) — Wala (4:
crudo/cocido 16 Oz por talla), Panamei (9: cocido pelado / EZ peel crudo por talla,
fundas 12-16 Oz), Netuno (2: crudo 2 Lb tallas grandes), Vima (cocidos 2 Lb) y
Diamond Reef.

EXCLUIDOS a propósito (3): sopas instantáneas sabor camarón (Kinsu 85 gr, Wala
ramen 85 gr, Cantonesa 80 gr) — son ramen ultra-procesado, no camarón (mismo
criterio que galletas/chips con orégano).

Notas:
  * El genérico del PDF calza exacto: Paquete RD$299 = Wala Cola Cocido 71/90 16 Oz.
  * El "20X12 Oz" de los títulos Panamei es notación de caja del distribuidor
    (20 fundas × 12 Oz); el precio es POR FUNDA de 12 Oz — así se registra.
  * Tallas (16/20, 31/40, 100/200...) = camarones por libra; se preservan en la
    presentación para el comparador.
  * 6 "No disponible" incluidos como referencia.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_camaron_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_camaron_2026_07_02.py --commit   # inserta
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
FOOD = "Camarón"
CATEGORY = "Carnes, pescados y mariscos"

# (brand, presentation, price_rd, description)
ROWS = [
    # ── Wala (fundas 16 Oz / 454 gr) ──
    ("Wala", "Funda Cola Cocido 51/60 16 Oz", 310, "Cola de camarón cocido congelado, talla 51/60"),
    ("Wala", "Funda Cola Cocido 71/90 16 Oz", 299, "Cola de camarón cocido congelado, talla 71/90"),
    ("Wala", "Funda Crudo 16/20 16 Oz", 315, "Cola de camarón crudo congelado, talla 16/20"),
    ("Wala", "Funda Crudo 21/25 16 Oz", 305, "Cola de camarón crudo congelado, talla 21/25"),
    # ── Panamei (fundas 12 Oz salvo indicado) ──
    ("Panamei", "Funda Cocido Pelado 16/20 12 Oz", 660, "Camarón cocido pelado congelado, talla 16/20 (jumbo)"),
    ("Panamei", "Funda Cocido Pelado 21/25 12 Oz", 595, "Camarón cocido pelado congelado, talla 21/25"),
    ("Panamei", "Funda Cocido Pelado 31/40 12 Oz", 515, "Camarón cocido pelado congelado, talla 31/40"),
    ("Panamei", "Funda Cocido Pelado 41/50 12 Oz", 500, "Camarón cocido pelado congelado, talla 41/50"),
    ("Panamei", "Funda Cocido Pelado 51/60 12 Oz", 475, "Camarón cocido pelado congelado, talla 51/60"),
    ("Panamei", "Funda Cocido Pelado 100/200 12 Oz", 400, "Camarón pequeño cocido pelado congelado, talla 100/200"),
    ("Panamei", "Funda Precocido 100/200 16 Oz", 550, "Camarón pequeño precocido congelado, talla 100/200"),
    ("Panamei", "Funda Crudo EZ Peel 16/20 12 Oz", 485, "Camarón crudo congelado easy peel, talla 16/20"),
    ("Panamei", "Funda Crudo EZ Peel 26/30 12 Oz", 420, "Camarón crudo congelado easy peel, talla 26/30"),
    # ── Netuno (fundas 2 Lb / 907 gr) ──
    ("Netuno", "Funda Crudo 16/20 2 Lb", 1195, "Cola de camarón crudo congelado, talla 16/20"),
    ("Netuno", "Funda Easy Peel Crudo 8/12 2 Lb", 1465, "Cola de camarón crudo easy peel, talla 8/12 (extra jumbo)"),
    # ── Vima ──
    ("Vima", "Funda Cocidos 31/40 2 Lb", 1255, "Camarones cocidos congelados pelados y desvenados, talla 31/40"),
    # ── Diamond Reef ──
    ("Diamond Reef", "Funda Crudo EZ Peel 16/20", 560, "Cola de camarón crudo congelado easy peel, talla 16/20"),
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
    print(f"Seed camarón: {len(ROWS)} SKUs · {len(brands)} marcas.")
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
