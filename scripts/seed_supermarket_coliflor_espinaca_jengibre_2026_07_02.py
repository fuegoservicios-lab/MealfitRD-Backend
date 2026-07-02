"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de COLIFLOR, ESPINACA y JENGIBRE.

Familias 57-59 del Supermercado RD: 9 SKUs transcritos del catálogo de La Sirena
(capturas del owner, 2026-07-02):

  * "Coliflor" (3): importada empacada RD$256, multicolor importada (carnival)
    RD$450 y vaso porcionado RD$155 ("No disponible"). El "Fresco Lb" RD$62
    CALZA EXACTO con el genérico — no se duplica.
  * "Espinaca" (3): Vima congelada 450 gr RD$140 — DOBLE CALCE: el genérico
    "Paquete 450 gr" RD$140 ES este producto (patrón fresas congeladas);
    El Corte Inglés frasco conserva 425 gr RD$550; Lucas Pérez funda fresca
    RD$180 ("No disponible"). El "Por Paquete" RD$42 calza exacto con el
    genérico "Paquete" — no se duplica.
  * "Jengibre" (1): Kayama rosado encurtido 340 gr RD$219 (el título del
    listing dice "Okayama" pero el frasco dice KAYAMA — el empaque manda).
    El "Fresco Lb" RD$98 calza exacto con el genérico — no se duplica.
  * "Jengibre molido" (food NUEVO, 1): Badia ground ginger 1.5 Oz RD$125 —
    especia seca ≠ jengibre fresco (criterio perejil/cilantro seco).
  * "Kombucha" (food NUEVO, 1): botella de kombucha de jengibre 16 Oz RD$420 —
    bebida probiótica fermentada, no es jengibre (product truth manda).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_coliflor_espinaca_jengibre_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_coliflor_espinaca_jengibre_2026_07_02.py --commit   # inserta
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
VEG = "Vegetales y verduras"
COND = "Condimentos y especias"
BEB = "Bebidas y alternativas vegetales"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Coliflor ──
    ("Coliflor", None, "Paquete Importado (unidad)", 256, "Coliflor importada (cauliflower), cabeza empacada", VEG),
    ("Coliflor", None, "Paquete Multicolor Importado (unidad)", 450, "Coliflor multicolor importada (carnival cauliflower)", VEG),
    ("Coliflor", None, "Vaso Fresco Porcionado (unidad)", 155, "Coliflor fresca porcionada en floretes, lista para cocinar", VEG),
    # ── Espinaca ──
    ("Espinaca", "Vima", "Funda Congelada Hojas 450 gr", 140, "Hojas de espinaca congeladas (leaf spinach)", VEG),
    ("Espinaca", "El Corte Inglés", "Frasco Selection 425 gr", 550, "Espinacas en conserva, frasco de vidrio", VEG),
    ("Espinaca", "Lucas Pérez", "Funda Fresca (unidad)", 180, "Espinaca fresca en funda, gourmet", VEG),
    # ── Jengibre ──
    ("Jengibre", "Kayama", "Frasco Rosado Encurtido 340 gr", 219, "Jengibre rosado encurtido (estilo sushi)", VEG),
    # ── Jengibre molido (food nuevo — especia seca) ──
    ("Jengibre molido", "Badia", "Frasco Ground Ginger 1.5 Oz", 125, "Jengibre molido (ground ginger)", COND),
    # ── Kombucha (food nuevo — bebida probiótica) ──
    ("Kombucha", None, "Botella Jengibre 16 Oz", 420, "Kombucha artesanal de jengibre (bebida probiótica fermentada)", BEB),
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
    print(f"Seed coliflor + espinaca + jengibre: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
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
