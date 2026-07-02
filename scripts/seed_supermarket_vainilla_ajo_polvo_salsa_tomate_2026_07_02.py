"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de VAINILLA, AJO EN POLVO (Oriente) y SALSA/PASTA DE TOMATE.

Familias 81-83 del Supermercado RD: 27 SKUs transcritos del catálogo de La Sirena
(capturas del owner, 2026-07-02):

  * "Vainilla" (11): las Wala 5 Oz (negra y blanca, RD$20) CALZAN EXACTO con
    el genérico "Botella 5 Oz"; Wala 16 Oz ×2 y la línea Delifruit completa
    (blanca/negra en 4, 8 y 16 Oz + galón blanca RD$215).
  * "Ajo en polvo" (+1): Oriente frasco 85 gr RD$185 — la Badia 3 Oz YA
    estaba cargada del seed de ajo (RD$115, idéntica) — no se duplica.
  * "Salsa de tomate" (4): DOBLE CALCE — Passata Pasta Zara 680 gr RD$159 =
    genérico "Pote 680 gr" y Milano condimentada 200 gr RD$88 = genérico
    "Sobre 200 gr". + Milano crema tomate y El Corte Inglés con albahaca
    ("No disponible"). El listing de la Passata aparecía 2 veces — dedupe.
  * "Pasta de tomate" (food NUEVO, 11): concentrado ≠ salsa lista (product
    truth) — Linda ×4 (225/340/455/900 gr — la 900 a precio de LISTA RD$204,
    promo -12% ignorada), La Famosa ×4 (8/12/16 Oz + 900 gr doble
    concentrado) y Victorina ×3 (8/16 Oz + 900 gr).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_vainilla_ajo_polvo_salsa_tomate_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_vainilla_ajo_polvo_salsa_tomate_2026_07_02.py --commit   # inserta
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
COND = "Condimentos y especias"
SALSA = "Salsas y aderezos"

VAIN = "Vainilla"
PTOM = "Pasta de tomate"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Vainilla · Wala ──
    (VAIN, "Wala", "Botella Negra 5 Oz", 20, "Vainilla líquida negra", COND),
    (VAIN, "Wala", "Botella Blanca 5 Oz", 20, "Vainilla líquida blanca", COND),
    (VAIN, "Wala", "Botella Negra 16 Oz", 52, "Vainilla líquida negra", COND),
    (VAIN, "Wala", "Botella Blanca 16 Oz", 52, "Vainilla líquida blanca", COND),
    # ── Vainilla · Delifruit ──
    (VAIN, "Delifruit", "Botella Blanca 4 Oz", 23, "Vainilla sabor blanca", COND),
    (VAIN, "Delifruit", "Botella Negra 4 Oz", 28, "Vainilla sabor negra", COND),
    (VAIN, "Delifruit", "Botella Blanca 8 Oz", 42, "Vainilla sabor blanca", COND),
    (VAIN, "Delifruit", "Botella Negra 8 Oz", 42, "Vainilla sabor negra", COND),
    (VAIN, "Delifruit", "Botella Blanca 16 Oz", 58, "Vainilla sabor blanca", COND),
    (VAIN, "Delifruit", "Botella Negra 16 Oz", 58, "Vainilla sabor negra", COND),
    (VAIN, "Delifruit", "Galón Blanca", 215, "Vainilla sabor blanca, galón", COND),
    # ── Ajo en polvo (la Badia 3 Oz ya estaba cargada del seed de ajo) ──
    ("Ajo en polvo", "Oriente", "Frasco 85 gr", 185, "Ajo en polvo", COND),
    # ── Salsa de tomate ──
    ("Salsa de tomate", "Pasta Zara", "Botella Passata 680 gr", 159, "Passata de tomate (puré colado)", SALSA),
    ("Salsa de tomate", "Milano", "Sobre Condimentada 200 gr", 88, "Salsa de tomate condimentada (tomate, cebolla, ajo, perejil)", SALSA),
    ("Salsa de tomate", "Milano", "Sobre Crema Tomate 200 gr", 108, "Salsa lista sabor crema de tomate", SALSA),
    ("Salsa de tomate", "El Corte Inglés", "Frasco con Albahaca 260 gr", 207, "Salsa de tomate con aceite de oliva y albahaca", SALSA),
    # ── Pasta de tomate (food nuevo — concentrado) ──
    (PTOM, "Linda", "Lata 225 gr", 65, "Pasta de tomate concentrada", SALSA),
    (PTOM, "Linda", "Lata 340 gr", 95, "Pasta de tomate concentrada", SALSA),
    (PTOM, "Linda", "Lata 455 gr", 105, "Pasta de tomate concentrada", SALSA),
    (PTOM, "Linda", "Lata 900 gr", 204, "Pasta de tomate concentrada, lata familiar", SALSA),
    (PTOM, "La Famosa", "Lata 8 Oz", 65, "Pasta de tomate doble concentrada", SALSA),
    (PTOM, "La Famosa", "Lata 12 Oz", 97, "Pasta de tomate doble concentrada", SALSA),
    (PTOM, "La Famosa", "Lata 16 Oz", 110, "Pasta de tomate doble concentrada", SALSA),
    (PTOM, "La Famosa", "Lata 900 gr", 205, "Pasta de tomate doble concentrada, lata familiar", SALSA),
    (PTOM, "Victorina", "Lata 8 Oz", 60, "Pasta de tomate", SALSA),
    (PTOM, "Victorina", "Lata 16 Oz", 105, "Pasta de tomate", SALSA),
    (PTOM, "Victorina", "Lata 900 gr", 205, "Pasta de tomate, lata familiar", SALSA),
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
    print(f"Seed vainilla + ajo en polvo + salsa/pasta de tomate: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
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
