"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de HARINA DE
NEGRITO + PISTACHOS + MANTEQUILLA DE ALMENDRAS.

Familias 117-119 del Supermercado RD (capturas del owner, 2026-07-02).

HARINA DE NEGRITO (5, La Sirena):
  * CALCE: Quaker funda 290 gr RD$59 = genérico "Paquete 290 gr"
    (creamy wheat Nutremás).
  * El Negrito ×4 (cajas cocimiento rápido: 8 Oz RD$78, canela 8 Oz
    RD$79, 16 Oz RD$149, 28 Oz RD$234 no disp).

PISTACHOS (4, La Sirena):
  * CALCE: Eloy's tarro 8 Oz RD$459 = genérico "Tarro 8 Oz".
  * Eloy's ×3 (tarro, pouch snack 3 Oz, pelados 6 Oz) + Nut Walker
    lata asados 130 gr.

MANTEQUILLA DE ALMENDRAS (1, **Supermercados Nacional** — notes por
fila): CALCE (redondeo del PDF): Eva tarro sin gluten 200 gr $524.95 =
genérico "Tarro 200 gr" RD$525.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_negrito_pistachos_mantequilla_almendras_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_negrito_pistachos_mantequilla_almendras_2026_07_02.py --commit   # inserta
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

SIRENA = "Precio de referencia La Sirena · 2026-07"
NACIONAL = "Precio de referencia Supermercados Nacional · 2026-07"
PAN = "Panadería y harinas"
SEM = "Semillas y frutos secos"

NEGRITO = "Harina de Negrito"

# (food_name, brand, presentation, price_rd, description, category, notes)
ROWS = [
    # ── Harina de Negrito ──
    (NEGRITO, "El Negrito", "Caja 8 Oz", 78, "Harina de trigo tostada, cocimiento rápido en 5 minutos", PAN, SIRENA),
    (NEGRITO, "El Negrito", "Caja Canela 8 Oz", 79, "Harina de trigo tostada con canela, cocimiento rápido", PAN, SIRENA),
    (NEGRITO, "El Negrito", "Caja 16 Oz", 149, "Harina de trigo tostada, cocimiento rápido en 5 minutos", PAN, SIRENA),
    (NEGRITO, "El Negrito", "Caja 28 Oz", 234, "Harina de trigo tostada, cocimiento rápido", PAN, SIRENA),
    (NEGRITO, "Quaker", "Funda 290 gr", 59, "Harina de negrito estilo creamy wheat (Nutremás)", PAN, SIRENA),
    # ── Pistachos ──
    ("Pistachos", "Eloy's", "Tarro 8 Oz", 459, "Pistachos tostados con cáscara", SEM, SIRENA),
    ("Pistachos", "Eloy's", "Funda Snack 3 Oz", 265, "Pistachos, pouch snack", SEM, SIRENA),
    ("Pistachos", "Eloy's", "Tarro Pelados 6 Oz", 559, "Pistachos pelados (sin cáscara)", SEM, SIRENA),
    ("Pistachos", "Nut Walker", "Lata Asados 130 gr", 435, "Pistachos asados y salados (dry roasted salted)", SEM, SIRENA),
    # ── Mantequilla de almendras (Supermercados Nacional) ──
    ("Mantequilla de almendras", "Eva", "Tarro Sin Gluten 200 gr", 524.95, "Mantequilla de almendra sin gluten", SEM, NACIONAL),
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
        if len(row) != 7:
            print(f"FATAL: fila con {len(row)} campos (esperados 7): {row[:3]}")
            sys.exit(1)
        (food, brand, pres, *_rest) = row
        key = (food.lower(), (brand or "").lower(), pres.lower())
        if key in seen:
            print(f"FATAL: fila duplicada en el seed: {food} / {brand} / {pres}")
            sys.exit(1)
        seen.add(key)

    foods = {r[0] for r in ROWS}
    brands = {r[1] for r in ROWS if r[1]}
    print(f"Seed negrito + pistachos + mantequilla de almendras: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, brand, pres, price, desc, category, notes) in ROWS:
                cur.execute(_INSERT, (food, brand, pres, price, notes, category, food, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
