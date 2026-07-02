"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de GARBANZO, HARINAS (maíz precocida/trigo) y MOSTAZA.

Familias 77-80 del Supermercado RD: 40 SKUs transcritos del catálogo de La Sirena
(capturas del owner, 2026-07-02):

  * "Garbanzo" (10): latas La Famosa (normal/vegetales) y Goya, fundas secas
    Giselle/Wala/Goya Chana, cartón Rica, y los 3 "No disponible" (La Cochura
    cocidos + blanco lechoso 11 mm, El Corte Inglés al natural). El genérico
    "Lb" RD$70 (granel) queda de referencia.
  * "Harina de garbanzo" (food NUEVO, 1): Eva 1 Lb (chickpea flour) — harina,
    no legumbre entera.
  * "Harina de maíz precocida" (2): P.A.N. blanco 500 gr RD$62 CALZA EXACTO
    con el genérico "Paquete 500 gr" + P.A.N. amarillo 1 Kg ("No disponible" ×2).
  * "Harina de trigo" (7): DOBLE CALCE — Wala 2 Lb RD$49 = genérico "Paquete
    2 L" y Milano 5 Lb RD$199 = genérico "Paquete 5 L". Blanquita 2/5 Lb,
    Milano 2 Lb, Molini Pizzuti tipo 00 y Goya 24 Oz.
  * "Harina de trigo integral" (food NUEVO, 1): Eva integral 1.5 Lb —
    integral ≠ blanca (criterio pan integral).
  * "Mostaza" (18): Wala 8 Oz RD$75 CALZA EXACTO con el genérico "Botella
    8 Oz". Baldom ×2, Essential Everyday ×6 (amarilla/honey/spicy brown/dijon
    ×2 formatos), Heinz, Maille (dijon original + integral old style),
    Dietz & Watson (chipotle + zesty honey — DEDUPE: los listings 65312 y
    66662 son el mismo producto 11 Oz), salsa Amazon dulce con chili
    (descripción honesta) y French's ×3 ("No disponible").
  * "Mostaza en polvo" (food NUEVO, 1): Badia orgánica 2 Oz — especia seca
    (criterio jengibre molido).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_garbanzo_harinas_mostaza_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_garbanzo_harinas_mostaza_2026_07_02.py --commit   # inserta
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
LEG = "Legumbres y proteína vegetal"
PAN = "Panadería y harinas"
SALSA = "Salsas y aderezos"
COND = "Condimentos y especias"

GARB = "Garbanzo"
HTRIGO = "Harina de trigo"
MOST = "Mostaza"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Garbanzo ──
    (GARB, "La Famosa", "Lata 15 Oz", 83, "Garbanzos cocidos en lata", LEG),
    (GARB, "La Famosa", "Lata con Vegetales 15 Oz", 85, "Garbanzos con vegetales en lata", LEG),
    (GARB, "Goya", "Lata 15.5 Oz", 90, "Garbanzos prime premium en lata (chick peas)", LEG),
    (GARB, "Goya", "Funda Secos Chana 16 Oz", 165, "Garbanzos secos (chana)", LEG),
    (GARB, "Giselle", "Funda Secos 400 gr", 67, "Garbanzos secos en funda", LEG),
    (GARB, "Wala", "Funda Secos 800 gr", 123, "Garbanzos secos, sin preservativos", LEG),
    (GARB, "Rica", "Cartón 400 gr", 78, "Garbanzos cocidos en cartón resellable", LEG),
    (GARB, "La Cochura", "Frasco Cocidos 560 gr", 185, "Garbanzos cocidos en frasco de vidrio", LEG),
    (GARB, "La Cochura", "Funda Blanco Lechoso 11 mm 500 gr", 150, "Garbanzo blanco lechoso calibre 11 mm, seco", LEG),
    (GARB, "El Corte Inglés", "Frasco Al Natural 400 gr", 165, "Garbanzos extra al natural en frasco", LEG),
    # ── Harina de garbanzo (food nuevo) ──
    ("Harina de garbanzo", "Eva", "Funda 1 Lb", 169, "Harina de garbanzo (chickpea flour)", PAN),
    # ── Harina de maíz precocida ──
    ("Harina de maíz precocida", "P.A.N.", "Paquete Blanco 500 gr", 62, "Harina de maíz blanco precocida (16 arepas)", PAN),
    ("Harina de maíz precocida", "P.A.N.", "Paquete Amarillo 1 Kg", 104, "Harina de maíz amarillo precocida", PAN),
    # ── Harina de trigo ──
    (HTRIGO, "Wala", "Paquete Todo Uso 2 Lb", 49, "Harina de trigo todo uso, enriquecida y precernida", PAN),
    (HTRIGO, "Blanquita", "Paquete Todo Uso 2 Lb", 98, "Harina de trigo enriquecida todo uso, precernida", PAN),
    (HTRIGO, "Blanquita", "Paquete Todo Uso 5 Lb", 235, "Harina de trigo enriquecida todo uso (all purpose)", PAN),
    (HTRIGO, "Milano", "Paquete 2 Lb", 79, "Harina de trigo fortificada para todo uso", PAN),
    (HTRIGO, "Milano", "Paquete 5 Lb", 199, "Harina de trigo fortificada para todo uso", PAN),
    (HTRIGO, "Molini Pizzuti", "Paquete Tipo 00 35.27 Oz", 230, "Harina de trigo blando tipo 00 italiana", PAN),
    (HTRIGO, "Goya", "Funda Enriquecida 24 Oz", 125, "Harina de trigo enriquecida (wheat flour)", PAN),
    # ── Harina de trigo integral (food nuevo) ──
    ("Harina de trigo integral", "Eva", "Funda Integral 1.5 Lb", 94, "Harina de trigo integral", PAN),
    # ── Mostaza ──
    (MOST, "Wala", "Botella Amarilla 8 Oz", 75, "Mostaza amarilla", SALSA),
    (MOST, "Baldom", "Botella 14 Oz", 102, "Mostaza amarilla", SALSA),
    (MOST, "Baldom", "Botella Spray 12 Oz", 98, "Mostaza en botella squeeze", SALSA),
    (MOST, "Essential Everyday", "Botella Squeeze Amarilla 8 Oz", 145, "Mostaza amarilla (yellow mustard)", SALSA),
    (MOST, "Essential Everyday", "Botella Amarilla 14 Oz", 165, "Mostaza amarilla (yellow mustard)", SALSA),
    (MOST, "Essential Everyday", "Botella Honey 12 Oz", 219, "Mostaza con miel (honey mustard)", SALSA),
    (MOST, "Essential Everyday", "Botella Spicy Brown 12 Oz", 165, "Mostaza spicy brown", SALSA),
    (MOST, "Essential Everyday", "Botella Dijon 12 Oz", 185, "Mostaza dijon elaborada con vino blanco", SALSA),
    (MOST, "Essential Everyday", "Botella Dijon Molida Gruesa 12 Oz", 165, "Mostaza dijon molida gruesa (coarse ground)", SALSA),
    (MOST, "Heinz", "Botella Amarilla 14 Oz", 340, "Mostaza amarilla natural", SALSA),
    (MOST, "Maille", "Frasco Dijon Original 7.5 Oz", 545, "Mostaza dijon original francesa", SALSA),
    (MOST, "Maille", "Frasco Integral Old Style 7.3 Oz", 589, "Mostaza a la antigua en grano entero (whole grain)", SALSA),
    (MOST, "Dietz & Watson", "Botella Chipotle 9 Oz", 250, "Mostaza chipotle", SALSA),
    (MOST, "Dietz & Watson", "Botella Zesty Honey 11 Oz", 250, "Mostaza con miel zesty", SALSA),
    (MOST, "Amazon", "Botella Salsa Dulce con Chili 12 Oz", 250, "Salsa de mostaza dulce con chili (sweet chili mustard)", SALSA),
    (MOST, "French's", "Botella Amarilla Clásica 14 Oz", 195, "Mostaza amarilla clásica", SALSA),
    (MOST, "French's", "Botella Honey 12 Oz", 254, "Mostaza con miel", SALSA),
    (MOST, "French's", "Botella Spicy Brown 12 Oz", 169, "Mostaza spicy brown", SALSA),
    # ── Mostaza en polvo (food nuevo — especia) ──
    ("Mostaza en polvo", "Badia", "Frasco Orgánica 2 Oz", 170, "Mostaza en polvo orgánica (mustard powder)", COND),
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
    print(f"Seed garbanzo + harinas + mostaza: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
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
