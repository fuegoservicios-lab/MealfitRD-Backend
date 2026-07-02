"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de NUECES MIXTAS +
CHAMPIÑONES + REMOLACHA + APIO (+1 salsa de tomate reclasificada).

Familias 92-95 del Supermercado RD (capturas del owner, 2026-07-02).
SEMILLAS DE CALABAZA (auyama): 0 filas — los 5 SKUs de las capturas
(Multifoods bandeja 10 Oz RD$249 y funda 200 gr RD$235, BioEva sin cáscara
200 gr RD$165, Eloy's funda 4 Oz RD$160 y tarro 8 Oz RD$305 = calce con el
genérico "Tarro 8 Oz") YA estaban cargados. La Sal de Apio Badia 4.5 Oz
RD$105 también YA existe bajo "Sal de apio".

NUECES MIXTAS (8, La Sirena):
  * CALCE: Wala funda 100 gr RD$95 = genérico "Paquete 100gr".
  * Dynasty ×4 (nueces y uvas secas 6 Oz, deluxe 6 Oz, con maní 6 Oz,
    nueces y chocolates 7 Oz), Cashita's ×2 (nueces/semillas/uva pasa en
    85 y 150 gr), Pacific Nuts premium 14 Oz (no disp, referencia). Los
    mixes con fruta/chocolate quedan EN familia con descripción honesta
    (criterio cajuil-con-pasas).

CHAMPIÑONES (5 + 1 reclasificada, La Sirena):
  * CALCE: Jazma bandeja crimini 8 Oz RD$205 = genérico "Paquete 8 Oz".
  * El Corte Inglés lata entero 185 gr y lata laminado (sin gramaje en el
    listing), Roland trozos y tallos 16 Oz (no disp), Polli funghetti
    grillados en aceite 190 gr (no disp, descripción honesta).
  * "Salsa Hunts Champiñones Tetra 360 g" RD$125 → product-truth
    RECLASIFICADA a "Salsa de tomate" (es salsa para pasta con
    champiñones, no champiñones — criterio néctar/margarina).

REMOLACHA (3, La Sirena):
  * CALCE: fresca Lb RD$45 = genérico "Lb" (criterio cubanela: el único
    tipo ES el genérico, no duplicar).
  * Precocida Lb (variante con nombre), Goya lata en rodajas 8.25/15 Oz.

APIO (4, La Sirena):
  * CALCE: fresco Lb RD$49 = genérico "Lb" (no duplicar).
  * Cepa fresca Lb (raíz de apio criollo, variante con nombre), Foxy
    orgánico importado 30 Ct, Ocean Mist corazones 16 Oz, porcionado
    Und (no disp).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_nueces_champinones_remolacha_apio_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_nueces_champinones_remolacha_apio_2026_07_02.py --commit   # inserta
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
SEM = "Semillas y frutos secos"
VEG = "Vegetales y verduras"
SALSAS = "Salsas y aderezos"

NUECES = "Nueces Mixtas"
CHAMP = "Champiñones"
REMO = "Remolacha"
APIO = "Apio"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Nueces Mixtas ──
    (NUECES, "Wala", "Funda 100 gr", 95, "Nueces mixtas", SEM),
    (NUECES, "Dynasty", "Tarro Nueces y Uvas Secas 6 Oz", 235, "Mix de nueces y uvas pasas", SEM),
    (NUECES, "Dynasty", "Tarro Deluxe 6 Oz", 385, "Nueces mixtas deluxe", SEM),
    (NUECES, "Dynasty", "Funda con Maní 6 Oz", 269, "Nueces mixtas con maní (mixed nuts with peanuts)", SEM),
    (NUECES, "Dynasty", "Funda Nueces y Chocolates 7 Oz", 175, "Mix de nueces mixtas y chocolates", SEM),
    (NUECES, "Cashita's", "Funda Nueces y Semillas 85 gr", 170, "Mezcla de nueces mixtas, semillas y uva pasa", SEM),
    (NUECES, "Cashita's", "Funda Nueces y Semillas 150 gr", 275, "Mezcla de nueces mixtas, semillas y uva pasa", SEM),
    (NUECES, "Pacific Nuts", "Funda Premium 14 Oz", 230, "Nueces mixtas premium, cosecha fresca de California", SEM),
    # ── Champiñones ──
    (CHAMP, "Jazma", "Bandeja Crimini 8 Oz", 205, "Champiñones crimini frescos", VEG),
    (CHAMP, "El Corte Inglés", "Lata Entero 185 gr", 228, "Champiñones enteros primera calidad", VEG),
    (CHAMP, "El Corte Inglés", "Lata Laminado", 168, "Champiñones laminados (el listing no indica gramaje)", VEG),
    (CHAMP, "Roland", "Lata Trozos y Tallos 16 Oz", 399, "Champiñones en trozos y tallos (pieces & stems)", VEG),
    (CHAMP, "Polli", "Frasco Funghetti 190 gr", 450, "Champiñones enteros grillados en aceite (funghetti sott'olio)", VEG),
    # ── Salsa de tomate (reclasificada desde el listado de champiñones) ──
    ("Salsa de tomate", "Hunt's", "Tetra Pack con Champiñones 360 gr", 125, "Salsa de tomate para pasta con champiñones", SALSAS),
    # ── Remolacha ──
    (REMO, None, "Precocida Lb", 49, "Remolacha precocida, venta por libra", VEG),
    (REMO, "Goya", "Lata en Rodajas 8.25 Oz", 95, "Remolachas en tajadas (sliced beets)", VEG),
    (REMO, "Goya", "Lata en Rodajas 15 Oz", 128, "Remolachas en tajadas (sliced beets)", VEG),
    # ── Apio ──
    (APIO, None, "Cepa Fresca Lb", 44, "Cepa/raíz de apio criollo, venta por libra", VEG),
    (APIO, None, "Porcionado Und", 59, "Apio fresco porcionado en bastones", VEG),
    (APIO, "Foxy", "Paquete Orgánico Importado 30 Ct", 234, "Apio orgánico importado (organic celery)", VEG),
    (APIO, "Ocean Mist", "Paquete Corazones de Apio 16 Oz", 228, "Corazones de apio (celery hearts)", VEG),
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
    print(f"Seed nueces + champiñones + remolacha + apio: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
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
