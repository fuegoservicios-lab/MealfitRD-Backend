"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de QUESO CREMA.

Familia 54 del Supermercado RD: 25 SKUs transcritos del catálogo de La Sirena
(capturas del owner, 2026-07-02):

  * Sosua (3): la barra Lb RD$289 CALZA EXACTO con el genérico "Lb" ya cargado;
    tarros cream cheese 0.5/1 Lb.
  * Philadelphia (8): barra y tarrina original, 1/3 less fat en ambos formatos
    (la captura "original 8 oz (tarro)" muestra el EMPAQUE 1/3 less fat — el
    empaque manda), chive & onion, tarrina 12 Oz, whipped 12 Oz y jalapeño
    ("No disponible").
  * San Juan (2, barra Lb + pieza empacada), Michel (3, incl. sin lactosa y
    porción), Pauly barra 8 Oz, Yokesso Lb, Crystal Farms 8 Oz,
    Del Artesano (3 cremas de queso artesanales: original/pimientos/puerro),
    García Baquero ibérico semicurado untable, Arla Natural Light y
    Bayernland Farmi ("No disponible" ×2, referencia).

  EXCLUIDOS (no son queso crema): sopa condensada Campbell's Cheddar Cheese
  10.5 Oz (RD$175), salsa de queso Old El Paso 9 Oz (RD$435, squeeze tex-mex
  ultra-procesada) y glaseado de repostería Duncan Hines Cream Cheese 16 Oz
  (RD$265).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_queso_crema_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_queso_crema_2026_07_02.py --commit   # inserta
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
LACTEO = "Lácteos y huevos"

CREMA = "Queso crema"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Sosua ──
    (CREMA, "Sosua", "Barra Lb", 289, "Queso crema, por libra", LACTEO),
    (CREMA, "Sosua", "Tarro Cream Cheese 0.5 Lb", 220, "Queso tipo cream cheese, tarro", LACTEO),
    (CREMA, "Sosua", "Tarro Cream Cheese 1 Lb", 385, "Queso tipo cream cheese, tarro", LACTEO),
    # ── Philadelphia (Kraft) ──
    (CREMA, "Philadelphia", "Barra Original 8 Oz", 325, "Queso crema original, barra (brick)", LACTEO),
    (CREMA, "Philadelphia", "Tarrina Original Spread 8 Oz", 350, "Queso crema untable original (cream cheese spread)", LACTEO),
    (CREMA, "Philadelphia", "Tarrina 1/3 Less Fat 8 Oz", 350, "Queso crema untable con 1/3 menos grasa", LACTEO),
    (CREMA, "Philadelphia", "Barra 1/3 Less Fat 8 Oz", 350, "Queso crema con 1/3 menos grasa, barra (brick)", LACTEO),
    (CREMA, "Philadelphia", "Tarrina Chive & Onion 8 Oz", 350, "Queso crema untable con cebollino y cebolla", LACTEO),
    (CREMA, "Philadelphia", "Tarrina Original 12 Oz", 445, "Queso crema untable original, tarrina grande", LACTEO),
    (CREMA, "Philadelphia", "Tarrina Whipped 12 Oz", 445, "Queso crema batido (whipped) original", LACTEO),
    (CREMA, "Philadelphia", "Tarrina Spicy Jalapeño 8 Oz", 350, "Queso crema untable con jalapeño picante", LACTEO),
    # ── San Juan ──
    (CREMA, "San Juan", "Barra Lb", 359, "Queso crema, por libra", LACTEO),
    (CREMA, "San Juan", "Pieza Empacada (unidad)", 365, "Queso crema, pieza empacada", LACTEO),
    # ── Michel ──
    (CREMA, "Michel", "Barra Lb", 325, "Queso crema, por libra", LACTEO),
    (CREMA, "Michel", "Barra Sin Lactosa Lb", 335, "Queso crema danés sin lactosa, por libra", LACTEO),
    (CREMA, "Michel", "Porción Empacada Lb", 335, "Queso crema en porción empacada al vacío, por libra", LACTEO),
    # ── Otras marcas ──
    (CREMA, "Pauly", "Barra Original 8 Oz", 185, "Queso crema original (cream cheese), barra 226 gr", LACTEO),
    (CREMA, "Yokesso", "Barra Lb", 250, "Queso crema, por libra", LACTEO),
    (CREMA, "Crystal Farms", "Barra Original 8 Oz", 215, "Queso crema original (cream cheese), barra", LACTEO),
    (CREMA, "Del Artesano", "Tarrina Crema de Queso Original (unidad)", 199, "Crema de queso artesanal original", LACTEO),
    (CREMA, "Del Artesano", "Tarrina Crema de Queso con Pimientos (unidad)", 199, "Crema de queso artesanal con pimientos", LACTEO),
    (CREMA, "Del Artesano", "Tarrina Crema de Queso y Puerro (unidad)", 205, "Crema de queso artesanal con puerro", LACTEO),
    (CREMA, "García Baquero", "Tarrina Ibérico Semicurado 125 gr", 199, "Crema de queso ibérico semicurado untable", LACTEO),
    (CREMA, "Arla", "Tarrina Natural Light 200 gr", 205, "Queso crema fresco light (natural light fresh cheese)", LACTEO),
    (CREMA, "Bayernland", "Farmi Classic 200 gr", 149, "Queso crema clásico (Frischkäse)", LACTEO),
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
    print(f"Seed queso crema: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
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
