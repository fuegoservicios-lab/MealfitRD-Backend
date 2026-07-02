"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de TOMATE.

Trigesimosexta familia con variantes de MARCA del Supermercado RD: 15 SKUs
transcritos del catálogo de La Sirena (capturas del owner, 2026-07-02), MÁS los
2 SKUs diferidos de la familia del orégano ("esos van cuando hagamos la familia
del tomate"). Repartidos en 3 foods:

  * "Tomate" (7, food existente): tipos frescos como variantes sin marca (bugalú,
    ensalada, cherry, grape mixto, racimo) + bandejas Sol Tropical (bugalú/ensalada).
  * "Tomate enlatado" (7, food NUEVO): Wala (troceados/enteros pelados), Linda
    triturados, La Famosa Tomaticos (albahaca + el diferido con orégano), Granoro
    pomodori pelati 800 gr y el diferido Hunt's diced con orégano. Conserva ≠
    fresco para la lista de compras.
  * "Salsa de tomate" (1, food existente): Solís tomate frito en brik (España).

Notas:
  * El genérico del PDF calza exacto: "Tomate" Lb RD$47 = bugalú Lb Y ensalada Lb.
  * Linda triturados a precio de LISTA RD$65 (promo -9% RD$59 transitoria).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_tomate_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_tomate_2026_07_02.py --commit   # inserta
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
SALSA = "Salsas y aderezos"

TOMATE = "Tomate"
LATA = "Tomate enlatado"
SALSA_T = "Salsa de tomate"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Tomate fresco ──
    (TOMATE, None, "Bugalú Lb", 47, "Tomate bugalú fresco, por libra", VEG),
    (TOMATE, None, "De Ensalada Lb", 47, "Tomate de ensalada fresco, por libra", VEG),
    (TOMATE, None, "Cherry (paquete)", 75, "Tomates cherry frescos, paquete", VEG),
    (TOMATE, None, "Grape Mixto (paquete)", 64, "Tomates grape mixtos (rojos y amarillos), paquete", VEG),
    (TOMATE, None, "Racimo Lb", 65, "Tomate en racimo fresco, por libra", VEG),
    (TOMATE, "Sol Tropical", "Bandeja Bugalú Lb", 65, "Tomate bugalú en bandeja", VEG),
    (TOMATE, "Sol Tropical", "Bandeja De Ensalada Lb", 65, "Tomate de ensalada en bandeja", VEG),
    # ── Tomate enlatado (food nuevo) ──
    (LATA, "Wala", "Lata Pelados Troceados 400 gr", 58, "Tomates pelados troceados, 100% natural (Italia)", VEG),
    (LATA, "Wala", "Lata Pelados Enteros 400 gr", 58, "Tomates pelados enteros, 100% natural (Italia)", VEG),
    (LATA, "Linda", "Lata Triturados 15 Oz", 65, "Tomates triturados (crushed tomatoes)", VEG),
    (LATA, "La Famosa", "Lata Tomaticos Albahaca 15 Oz", 100, "Tomates pelados y picados con albahaca", VEG),
    (LATA, "La Famosa", "Lata Tomaticos con Orégano 410 gr", 110, "Tomates pelados y picados con orégano", VEG),
    (LATA, "Granoro", "Lata Pomodori Pelati 800 gr", 210, "Tomates pelados italianos (solo pomodoro italiano)", VEG),
    (LATA, "Hunt's", "Lata Diced con Orégano 14.5 Oz", 190, "Tomates en cubos (diced) sazonados con orégano", VEG),
    # ── Salsa de tomate (food existente) ──
    (SALSA_T, "Solís", "Brik Tomate Frito 350 gr", 107, "Tomate frito estilo español, 100% sabor natural (Vegas del Guadiana)", SALSA),
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
    for (food, brand, pres, *_rest) in ROWS:
        key = (food.lower(), (brand or "").lower(), pres.lower())
        if key in seen:
            print(f"FATAL: fila duplicada en el seed: {food} / {brand} / {pres}")
            sys.exit(1)
        seen.add(key)

    foods = {r[0] for r in ROWS}
    brands = {r[1] for r in ROWS if r[1]}
    print(f"Seed tomate: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
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
