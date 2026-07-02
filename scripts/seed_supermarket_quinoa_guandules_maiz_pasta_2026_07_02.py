"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de QUINOA, GUANDULES, MAÍZ DULCE, PASTA INTEGRAL y ESPAGUETIS.

Familias 71-75 del Supermercado RD: 51 SKUs transcritos del catálogo de La Sirena
(capturas del owner, 2026-07-02):

  * "Quinoa" (3): Goya Organics blanca 12 Oz RD$215 y roja 12 Oz RD$225 +
    Bob's Red Mill orgánica GF 13 Oz RD$1,150. El genérico "Paquete 12 Oz"
    RD$189 queda como referencia (no calza con ninguna marca).
  * "Semillas de chía" (+1): la "Quinoa Organics Chia Goya 12 Oz" RD$230 es
    CHÍA (el empaque manda sobre el bucket del search) — va a su familia.
  * "Guandules" (19): Wala lata verdes 15 Oz RD$84 CALZA EXACTO con el
    genérico "Lata 15 Oz" y Giselle funda secos 800 gr RD$199 CALZA EXACTO
    con el genérico "Paquete 800 gr" (doble calce). La Famosa ×5 (precios de
    LISTA — promos -9%/-12% ignoradas), Goya ×7 (verdes/con coco/secos),
    Linda, Victorina, Giselle 400 gr, Wala con coco y funda 800 gr.
  * "Guisantes secos" (3): los chícharos Goya van aquí — chícharos = guisantes
    secos (partidos verdes 16 Oz, enteros amarillos 16 Oz, enteros verdes
    16 Oz). Product truth: no son guandules.
  * "Guisantes" (food NUEVO, 4): guisantes verdes frescos/congelados (green
    peas) ≠ guisantes secos — Pictsweet 12/24 Oz, Mambo 32 Oz y El Corte
    Inglés al natural 280 gr.
  * "Maíz dulce" (12): Zerca 425 gr RD$55 CALZA EXACTO con el genérico "Lata
    425 gr". La Famosa (lista RD$70/RD$108), Linda, Victorina, Rica cartón,
    Del Sol y las mazorcas frescas Lucas Pérez (bandeja 3 unid).
  * "Pasta integral" (3): Pasta Zara spaghetti integral N°3 500 gr RD$209
    CALZA EXACTO con el genérico "Paquete 500 gr"; penne N°49 y Barilla
    penne rigate integrale ("No disponible" ×3, referencia).
  * "Espaguetis" (food NUEVO, 6): Barilla N°5 y GF, Milano 400 gr (lista
    RD$42) y gluten free, Bragagnolo spaghettini, Pasta Zara capellini.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_quinoa_guandules_maiz_pasta_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_quinoa_guandules_maiz_pasta_2026_07_02.py --commit   # inserta
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
GRANO = "Granos y cereales"
SEM = "Semillas y frutos secos"
LEG = "Legumbres y proteína vegetal"
VEG = "Vegetales y verduras"

GUAND = "Guandules"
MAIZ = "Maíz dulce"
ESPAG = "Espaguetis"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Quinoa ──
    ("Quinoa", "Goya", "Funda Organics Blanca 12 Oz", 215, "Quinoa blanca orgánica (white quinoa)", GRANO),
    ("Quinoa", "Goya", "Funda Organics Roja 12 Oz", 225, "Quinoa roja orgánica (red quinoa)", GRANO),
    ("Quinoa", "Bob's Red Mill", "Funda Orgánica Gluten Free 13 Oz", 1150, "Quinoa orgánica en grano entero, certificada sin gluten", GRANO),
    # ── Semillas de chía (la 'Quinoa Chia' de Goya es chía — empaque manda) ──
    ("Semillas de chía", "Goya", "Funda Organics 12 Oz", 230, "Semillas de chía orgánicas", SEM),
    # ── Guandules · Wala ──
    (GUAND, "Wala", "Lata Verdes 15 Oz", 84, "Guandules verdes en lata", LEG),
    (GUAND, "Wala", "Lata Verdes con Coco 15 Oz", 99, "Guandules verdes con coco en lata", LEG),
    (GUAND, "Wala", "Funda Secos 800 gr", 168, "Guandules secos, sin preservativos", LEG),
    # ── Guandules · La Famosa ──
    (GUAND, "La Famosa", "Lata Verdes 15 Oz", 120, "Guandules verdes en lata (green pigeon peas)", LEG),
    (GUAND, "La Famosa", "Lata Verdes 8 Oz", 78, "Guandules verdes en lata", LEG),
    (GUAND, "La Famosa", "Lata Verdes con Coco 15 Oz", 155, "Guandules verdes con coco en lata", LEG),
    (GUAND, "La Famosa", "Lata Verdes con Coco 8 Oz", 102, "Guandules verdes con coco en lata", LEG),
    (GUAND, "La Famosa", "Lata Verdes 820 gr", 205, "Guandules verdes en lata familiar", LEG),
    # ── Guandules · Goya ──
    (GUAND, "Goya", "Lata Verdes 15 Oz", 121, "Guandules verdes prime premium en lata", LEG),
    (GUAND, "Goya", "Lata Verdes 8 Oz", 82, "Guandules verdes prime premium en lata", LEG),
    (GUAND, "Goya", "Lata Verdes 29 Oz", 220, "Guandules verdes prime premium, lata familiar", LEG),
    (GUAND, "Goya", "Lata Verdes con Coco 8 Oz", 97, "Guandules verdes con coco premium en lata", LEG),
    (GUAND, "Goya", "Lata Verdes con Coco 15.5 Oz", 142, "Guandules verdes con coco premium en lata", LEG),
    (GUAND, "Goya", "Lata Secos 15 Oz", 113, "Guandules secos (dry pigeon peas) en lata", LEG),
    (GUAND, "Goya", "Lata Secos con Coco 15 Oz", 125, "Guandules secos con coco en lata", LEG),
    # ── Guandules · otras marcas ──
    (GUAND, "Linda", "Lata Verdes 15 Oz", 105, "Guandules verdes en lata", LEG),
    (GUAND, "Victorina", "Lata Verdes 15 Oz", 105, "Guandules verdes en lata, buena fuente de fibra", LEG),
    (GUAND, "Giselle", "Funda Secos 400 gr", 130, "Guandules secos en funda", LEG),
    (GUAND, "Giselle", "Funda Secos 800 gr", 199, "Guandules secos en funda", LEG),
    # ── Guisantes secos (chícharos = guisantes secos) ──
    ("Guisantes secos", "Goya", "Funda Chícharos Verdes Partidos 16 Oz", 99, "Chícharos verdes partidos (green split peas)", LEG),
    ("Guisantes secos", "Goya", "Funda Chícharos Enteros Amarillos 16 Oz", 70, "Chícharos amarillos enteros (whole yellow peas)", LEG),
    ("Guisantes secos", "Goya", "Funda Chícharos Enteros Verdes 16 Oz", 108, "Chícharos verdes enteros (arvejas)", LEG),
    # ── Guisantes (food nuevo — verdes frescos/congelados, no secos) ──
    ("Guisantes", "Pictsweet", "Funda Congelados 12 Oz", 320, "Guisantes verdes congelados (green peas)", VEG),
    ("Guisantes", "Pictsweet", "Funda Congelados 24 Oz", 379, "Guisantes verdes congelados (green peas)", VEG),
    ("Guisantes", "Mambo", "Funda Congelados Sweet Peas 32 Oz", 589, "Guisantes dulces congelados (sweet peas)", VEG),
    ("Guisantes", "El Corte Inglés", "Lata Al Natural Muy Finos 280 gr", 359, "Guisantes muy finos al natural", VEG),
    # ── Maíz dulce ──
    (MAIZ, "Zerca", "Lata 425 gr", 55, "Maíz dulce en granos", VEG),
    (MAIZ, "Zerca", "Lata 227 gr", 39, "Maíz dulce en granos", VEG),
    (MAIZ, "La Famosa", "Lata 8 Oz", 70, "Maíz dulce en lata", VEG),
    (MAIZ, "La Famosa", "Lata 15 Oz", 108, "Maíz dulce en lata", VEG),
    (MAIZ, "La Famosa", "Lata 820 gr", 179, "Maíz dulce en lata familiar", VEG),
    (MAIZ, "Linda", "Lata 8 Oz", 62, "Maíz dulce en lata", VEG),
    (MAIZ, "Linda", "Lata 15 Oz", 92, "Maíz dulce en lata", VEG),
    (MAIZ, "Victorina", "Lata 8 Oz", 66, "Maíz dulce (sweet corn) en lata", VEG),
    (MAIZ, "Victorina", "Lata 15 Oz", 85, "Maíz dulce (sweet corn) en lata", VEG),
    (MAIZ, "Rica", "Cartón 400 gr", 98, "Maíz dulce en cartón resellable", VEG),
    (MAIZ, "Del Sol", "Lata 15 Oz", 72, "Maíz dulce en lata", VEG),
    (MAIZ, "Lucas Pérez", "Bandeja Mazorcas Frescas (3 unid)", 110, "Mazorcas de maíz dulce frescas en bandeja", VEG),
    # ── Pasta integral ──
    ("Pasta integral", "Pasta Zara", "Paquete Spaghetti Integral N°3 500 gr", 209, "Espaguetis integrales (linea integrale)", GRANO),
    ("Pasta integral", "Pasta Zara", "Paquete Penne Rigate Integral N°49 500 gr", 179, "Penne rigate integrales (linea integrale)", GRANO),
    ("Pasta integral", "Barilla", "Caja Penne Rigate Integrale 500 gr", 240, "Penne rigate integrales, fuente de fibra", GRANO),
    # ── Espaguetis (food nuevo) ──
    (ESPAG, "Barilla", "Caja N°5 500 gr", 146, "Espaguetis de sémola de trigo", GRANO),
    (ESPAG, "Barilla", "Caja Gluten Free 12 Oz", 235, "Espaguetis sin gluten (maíz y arroz)", GRANO),
    (ESPAG, "Milano", "Paquete 400 gr", 42, "Espaguetis de sémola de trigo", GRANO),
    (ESPAG, "Milano", "Paquete Gluten Free (unidad)", 135, "Espaguetis libres de gluten", GRANO),
    (ESPAG, "Bragagnolo", "Paquete Spaghettini 803 500 gr", 169, "Spaghettini italianos", GRANO),
    (ESPAG, "Pasta Zara", "Paquete Capellini 500 gr", 109, "Espaguetis finos capellini", GRANO),
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
    for row in ROWS:
        if len(row) != 6:
            print(f"FATAL: fila con {len(row)} campos (esperados 6): {row[:3]}")
            sys.exit(1)
        (food, brand, pres, *_rest) = row
        key = (food.lower(), (brand or "").lower(), pres.lower())
        if key in seen:
            print(f"FATAL: fila duplicada en el seed: {food} / {brand} / {pres}")
            sys.exit(1)
        seen.add(key)

    foods = {r[0] for r in ROWS}
    brands = {r[1] for r in ROWS if r[1]}
    print(f"Seed quinoa + guandules + maíz + pastas: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, brand, pres, price, desc, category) in ROWS:
                cur.execute(_INSERT, (food, brand, pres, price, NOTES, category, food, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
