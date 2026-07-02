"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de RÚCULA +
CALABACÍN + RÁBANO + ESPÁRRAGOS.

Familias 96-99 del Supermercado RD (capturas del owner, 2026-07-02).
REPOLLO MORADO: 0 filas — "Lb RD$59" y el "Radicchio Importado (unidad)
RD$355" YA estaban cargados de la tanda de repollo.

RÚCULA (food NUEVO, 3, La Sirena): Lucas Pérez funda 1/2 paq RD$130,
funda 2 Oz RD$43 y clamshell silvestre 7 Oz RD$269 (no disp, referencia).

CALABACÍN (1, La Sirena): CALCE del listing "Calabacin Lb" RD$49 con el
genérico "Lb" (criterio cubanela — no duplicar). Solo entra el VMB de
Constanza en bandeja RD$63 (calabacín clarito; ambos no disp, referencia).

RÁBANO (food NUEVO, 2, La Sirena): Sol Tierra funda RD$65 y Lucas Pérez
bandeja de rabanitos rojos RD$85.

ESPÁRRAGOS (food NUEVO, 4, La Sirena): Vima congelados verdes 450 gr
RD$405, Goya frascos en conserva (blancos largos 11.6 Oz RD$305, verdes
11.6 Oz RD$299), El Corte Inglés verdes cortos 100 gr RD$225 (no disp).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_rucula_calabacin_rabano_esparragos_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_rucula_calabacin_rabano_esparragos_2026_07_02.py --commit   # inserta
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

# (food_name, brand, presentation, price_rd, description)
ROWS = [
    # ── Rúcula (food nuevo) ──
    ("Rúcula", "Lucas Pérez", "Funda 1/2 Paq", 130, "Rúcula (arugula), ensaladas gourmet"),
    ("Rúcula", "Lucas Pérez", "Funda 2 Oz", 43, "Rúcula (arugula), ensaladas gourmet"),
    ("Rúcula", "Lucas Pérez", "Clamshell Silvestre 7 Oz", 269, "Rúcula silvestre (wild arugula)"),
    # ── Calabacín ──
    ("Calabacín", "VMB", "Paquete", 63, "Calabacín clarito de Constanza, bandeja"),
    # ── Rábano (food nuevo) ──
    ("Rábano", "Sol Tierra", "Funda", 65, "Rábano fresco"),
    ("Rábano", "Lucas Pérez", "Paquete Rabanitos Rojos", 85, "Rabanitos rojos, bandeja"),
    # ── Espárragos (food nuevo) ──
    ("Espárragos", "Vima", "Funda Congelados Verdes 450 gr", 405, "Espárragos verdes congelados (green asparagus)"),
    ("Espárragos", "Goya", "Frasco Blancos Largos 11.6 Oz", 305, "Espárragos blancos largos en conserva"),
    ("Espárragos", "Goya", "Frasco Verdes 11.6 Oz", 299, "Espárragos verdes en conserva"),
    ("Espárragos", "El Corte Inglés", "Frasco Verdes Cortos 100 gr", 225, "Espárragos verdes de tallo delgado, cortos, en conserva"),
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
        if len(row) != 5:
            print(f"FATAL: fila con {len(row)} campos (esperados 5): {row[:3]}")
            sys.exit(1)
        (food, brand, pres, *_rest) = row
        key = (food.lower(), (brand or "").lower(), pres.lower())
        if key in seen:
            print(f"FATAL: fila duplicada en el seed: {food} / {brand} / {pres}")
            sys.exit(1)
        seen.add(key)

    foods = {r[0] for r in ROWS}
    brands = {r[1] for r in ROWS if r[1]}
    print(f"Seed rúcula + calabacín + rábano + espárragos: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, brand, pres, price, desc) in ROWS:
                cur.execute(_INSERT, (food, brand, pres, price, NOTES, VEG, food, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
