"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de AVENA.

Séptima familia con variantes de MARCA del Supermercado RD: 22 SKUs transcritos del
catálogo de La Sirena (capturas del owner, 2026-07-02) — Quaker (14: instantánea/
integral/molida/proteína/cups/tarros saborizados/food service), Wala (3), Avena
Americana (4), Multifoods (1).

Nota: La Sirena lista DOS veces la Wala Integral 600 gr (RD$49) y la Wala Molida
650 gr (RD$59) — listados duplicados del mismo producto; se cargan UNA vez.

Mismo patrón que seed_supermarket_leches_2026_07_02.py. Idempotente: ON CONFLICT
(variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_avena_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_avena_2026_07_02.py --commit   # inserta
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
FOOD = "Avena"
CATEGORY = "Granos y cereales"

# (brand, presentation, price_rd, description)
ROWS = [
    # ── Quaker ──
    ("Quaker", "Funda NutreMás 618 gr", 99, "Avena instantánea NutreMás (instant oats)"),
    ("Quaker", "Funda Integral 570 gr", 102, "Avena integral en hojuelas (whole oats)"),
    ("Quaker", "Funda Molida NutreMás 525 gr", 88, "Avena molida (oat flour) NutreMás"),
    ("Quaker", "Funda Instantánea Family Size 1200 gr", 189, "Avena instantánea NutreMás, tamaño familiar"),
    ("Quaker", "Funda Proteína 300 gr", 90, "Avena entera con proteína — hojuelas de avena y harina de soya"),
    ("Quaker", "Funda Integral 285 gr", 65, "Avena integral en hojuelas (whole oats)"),
    ("Quaker", "Funda NutreMás 285 gr", 60, "Avena instantánea NutreMás"),
    ("Quaker", "Funda Instantánea Food Service 1880 gr", 298, "Avena instantánea, tamaño institucional"),
    ("Quaker", "Funda Molida para Jugo 950 gr", 179, "Avena molida para jugo, tamaño familiar"),
    ("Quaker", "Cup Original 40 gr", 72, "Avena instantánea en vaso, lista en minutos"),
    ("Quaker", "Cup Vainilla 45 gr", 72, "Avena instantánea en vaso sabor vainilla"),
    ("Quaker", "Cup Canela 45 gr", 72, "Avena instantánea en vaso sabor canela"),
    ("Quaker", "Tarro Canela 310 gr", 135, "Avena instantánea sabor canela (cinnamon)"),
    ("Quaker", "Tarro Vainilla 310 gr", 135, "Avena instantánea sabor vainilla"),
    # ── Wala ──
    ("Wala", "Funda Integral 600 gr", 49, "Avena integral en hojuelas"),
    ("Wala", "Funda Instantánea 650 gr", 47, "Avena instantánea, fácil preparación"),
    ("Wala", "Funda Molida 650 gr", 59, "Avena molida"),
    # ── Avena Americana ──
    ("Avena Americana", "Funda Instantánea 590 gr", 92, "Avena instantánea 100% natural"),
    ("Avena Americana", "Funda Old Fashioned 600 gr", 130, "Avena en hojuelas enteras (old fashioned)"),
    ("Avena Americana", "Sobre Instantánea 300 gr", 52, "Avena instantánea 100% natural"),
    ("Avena Americana", "Funda Integral 360 gr", 70, "Avena integral 100% natural"),
    # ── Multifoods ──
    ("Multifoods", "Funda Integral 12 Oz", 48, "Avena integral en hojuelas (whole oat)"),
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
    for (brand, pres, *_rest) in ROWS:
        key = ((brand or "").lower(), pres.lower())
        if key in seen:
            print(f"FATAL: fila duplicada en el seed: {brand} / {pres}")
            sys.exit(1)
        seen.add(key)

    brands = {r[0] for r in ROWS}
    print(f"Seed avena: {len(ROWS)} SKUs · {len(brands)} marcas.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (brand, pres, price, desc) in ROWS:
                cur.execute(_INSERT, (FOOD, brand, pres, price, NOTES, CATEGORY, FOOD, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
