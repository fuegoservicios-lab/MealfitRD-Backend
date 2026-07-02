"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de GALLETA DE SODA.

Familia 85 del Supermercado RD: 20 SKUs transcritos del catálogo de La Sirena
(capturas del owner, 2026-07-02):

  * DOBLE CALCE con genéricos: Wala caja 20 unid 600 gr RD$148 = genérico
    "Caja 20 unid" y Hatuey cubo plástico 24 unid RD$265 = genérico "Caja
    24 unid". La Hatuey "Caja Regular 20 unid" RD$169 YA estaba cargada
    (bonus del seed de yogurt regular, precio de lista idéntico) — skip.
  * Hatuey (6): regular 9 unid, saltina 20/9 unid (promos -10% a precio de
    LISTA), cubo plástico 24, envase hermético 768 gr y caja 14 unid
    ("No disponible" ×2).
  * Aviva (7): regular/saltina en 20 y 9 unid, lite 9 unid (promos -8% a
    LISTA RD$92), cubos 23 y 18 unid. El listing "Aviva Soda 300 G" RD$92
    es la misma caja de 9 unid (9×32 gr ≈ 288 gr) — DEDUPE.
  * Crich (5): bio con aceite de oliva (el título decía "orégano aceite
    oliva" pero el empaque dice BIO cracker con EVOO salati — empaque
    manda), oliva y romero, salada, tomate y orégano, sin sal.
  * "Galleta de soda integral" (food NUEVO, 1): Crich 100% harina integral
    250 gr — integral ≠ regular (criterio pan integral).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_galleta_soda_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_galleta_soda_2026_07_02.py --commit   # inserta
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
PAN = "Panadería y harinas"

SODA = "Galleta de soda"

# (food_name, brand, presentation, price_rd, description)
ROWS = [
    # ── Hatuey ──
    (SODA, "Hatuey", "Caja Regular 9 unid (288 gr)", 100, "Galletas de soda regulares, 9 paquetes de 32 gr, 0% colesterol"),
    (SODA, "Hatuey", "Caja Saltina 20 unid (640 gr)", 169, "Galletas de soda saltinas, 20 paquetes de 32 gr"),
    (SODA, "Hatuey", "Caja Saltina 9 unid (10.16 Oz)", 100, "Galletas de soda saltinas, 9 paquetes de 32 gr"),
    (SODA, "Hatuey", "Cubo Plástico 24 unid", 265, "Galletas de soda en cubo plástico, 24 paquetes"),
    (SODA, "Hatuey", "Envase Hermético 768 gr", 209, "Galletas de soda en envase hermético, 24 paquetes"),
    (SODA, "Hatuey", "Caja 14 unid (15.8 Oz)", 139, "Galletas de soda, nueva presentación de 14 paquetes"),
    # ── Aviva ──
    (SODA, "Aviva", "Caja 20 unid", 154, "Galletas de soda (soda crackers)"),
    (SODA, "Aviva", "Caja Saltina 20 unid", 154, "Galletas de soda saltinas (salted soda crackers)"),
    (SODA, "Aviva", "Caja 9 unid (300 gr)", 92, "Galletas de soda (soda crackers)"),
    (SODA, "Aviva", "Caja Saltina 9 unid", 92, "Galletas de soda saltinas"),
    (SODA, "Aviva", "Caja Lite 9 unid", 92, "Galletas de soda lite"),
    (SODA, "Aviva", "Cubo 23 unid", 260, "Galletas de soda en cubo plástico"),
    (SODA, "Aviva", "Cubo 18 unid", 235, "Galletas de soda en cubo plástico"),
    # ── Wala ──
    (SODA, "Wala", "Caja 20 unid 600 gr", 148, "Galletas de soda, 20 paquetes de 30 gr"),
    # ── Crich (crackers italianos, 24 h de leudado) ──
    (SODA, "Crich", "Paquete Bio Aceite de Oliva 250 gr", 220, "Crackers orgánicos con aceite de oliva extra virgen 10%, salados"),
    (SODA, "Crich", "Paquete Oliva y Romero 250 gr", 160, "Crackers con oliva y romero"),
    (SODA, "Crich", "Paquete Salada 250 gr", 165, "Crackers salados, 24 horas de leudado"),
    (SODA, "Crich", "Paquete Tomate y Orégano 250 gr", 150, "Crackers de tomate y orégano"),
    (SODA, "Crich", "Paquete Sin Sal 250 gr", 142, "Crackers sin sal, 24 horas de leudado"),
]

# (food_name, brand, presentation, price_rd, description) — food integral separado
ROWS_INTEGRAL = [
    ("Galleta de soda integral", "Crich", "Paquete Integral 250 gr", 143, "Crackers 100% harina de trigo integral, altos en fibra"),
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

    all_rows = ROWS + ROWS_INTEGRAL
    seen = set()
    for row in all_rows:
        if len(row) != 5:
            print(f"FATAL: fila con {len(row)} campos (esperados 5): {row[:3]}")
            sys.exit(1)
        (food, brand, pres, *_rest) = row
        key = (food.lower(), (brand or "").lower(), pres.lower())
        if key in seen:
            print(f"FATAL: fila duplicada en el seed: {food} / {brand} / {pres}")
            sys.exit(1)
        seen.add(key)

    foods = {r[0] for r in all_rows}
    brands = {r[1] for r in all_rows if r[1]}
    print(f"Seed galleta de soda: {len(all_rows)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, brand, pres, price, desc) in all_rows:
                cur.execute(_INSERT, (food, brand, pres, price, NOTES, PAN, food, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
