"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de ACEITE DE OLIVA.

Decimoquinta familia con variantes de MARCA del Supermercado RD: 37 SKUs transcritos
del catálogo de La Sirena (capturas del owner, 2026-07-02) — Wala, Borges (incl.
varietales Arbequina/Balance Hojiblanca/Robusto Picual, Essences sabor ajo y blend
girasol), Figaro (incl. latas, orgánico y blend canola), Goya (incl. spray y Único
premium), Diosas de Abril (Picual/Arbequina), Carbonell, Barbera, Filippo Berio, PAM
(sprays), y el blend Wala girasol.

Los blends (girasol/canola + oliva) y sprays se cargan bajo "Aceite de oliva" con
descripción honesta de que son mezcla/spray — el selector de la lista de compras
podrá distinguirlos por descripción.

El genérico del PDF calza exacto: Pote 250 ML RD$195 = Wala Virgen Extra 250 Ml.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_aceite_oliva_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_aceite_oliva_2026_07_02.py --commit   # inserta
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
FOOD = "Aceite de oliva"
CATEGORY = "Aceites y grasas"

# (brand, presentation, price_rd, description)
ROWS = [
    # ── Wala ──
    ("Wala", "Botella Virgen Extra 250 Ml", 195, "Aceite de oliva virgen extra (España)"),
    ("Wala", "Botella Virgen Extra 500 Ml", 365, "Aceite de oliva virgen extra (España)"),
    ("Wala", "Botella Virgen Extra 750 Ml", 525, "Aceite de oliva virgen extra (España)"),
    ("Wala", "Botella Virgen Extra 2 Lt", 1320, "Aceite de oliva virgen extra, tamaño grande"),
    ("Wala", "Botella Blend Girasol 750 Ml", 199, "Aceite blend: mezcla de girasol y oliva (no es 100% oliva)"),
    # ── Borges ──
    ("Borges", "Botella Virgen Extra 125 Ml", 165, "Aceite de oliva virgen extra original"),
    ("Borges", "Botella Virgen Extra 250 Ml", 290, "Aceite de oliva virgen extra original"),
    ("Borges", "Botella Virgen Extra 500 Ml", 565, "Aceite de oliva virgen extra original"),
    ("Borges", "Botella Virgen Extra 1 Lt", 995, "Aceite de oliva virgen extra original"),
    ("Borges", "Botella Virgen Extra 2 Lt", 1920, "Aceite de oliva virgen extra original"),
    ("Borges", "Garrafa Virgen Extra 5 Lt", 4500, "Aceite de oliva virgen extra, garrafa institucional"),
    ("Borges", "Botella Arbequina 750 Ml", 775, "Aceite de oliva virgen extra varietal Arbequina"),
    ("Borges", "Botella Balance Hojiblanca 500 Ml", 520, "Aceite de oliva virgen extra varietal Hojiblanca (Balance)"),
    ("Borges", "Botella Robusto Picual 500 Ml", 520, "Aceite de oliva virgen extra varietal Picual (Robusto)"),
    ("Borges", "Botella Essences Sabor Ajo 500 Ml", 495, "Condimento de aceite de oliva virgen extra sabor ajo"),
    ("Borges", "Botella Blend Girasol 1 Lt", 349, "Aceite blend: mezcla de girasol y oliva virgen extra (no es 100% oliva)"),
    # ── Figaro ──
    ("Figaro", "Botella Virgen Extra 250 Ml", 380, "Aceite de oliva virgen extra español"),
    ("Figaro", "Botella Virgen Extra 500 Ml", 715, "Aceite de oliva virgen extra español"),
    ("Figaro", "Botella Virgen Extra 750 Ml", 1055, "Aceite de oliva virgen extra español"),
    ("Figaro", "Botella Virgen Extra 1 Lt", 1345, "Aceite de oliva virgen extra español"),
    ("Figaro", "Botella Orgánico 500 Ml", 770, "Aceite de oliva virgen extra orgánico español"),
    ("Figaro", "Lata 175 gr", 295, "Aceite de oliva español, lata"),
    ("Figaro", "Lata 375 gr", 525, "Aceite de oliva español, lata"),
    ("Figaro", "Botella Canola y Oliva 1 Lt", 455, "Aceite blend: mezcla de canola y oliva virgen extra (no es 100% oliva)"),
    # ── Goya ──
    ("Goya", "Botella Virgen Extra 250 Ml", 460, "Aceite de oliva virgen extra, extracción en frío"),
    ("Goya", "Botella Virgen Extra 500 Ml", 795, "Aceite de oliva virgen extra, extracción en frío"),
    ("Goya", "Botella Único Premium 17 Oz", 900, "Aceite de oliva virgen extra premium Único"),
    ("Goya", "Spray Extra Virgin 5 Oz", 400, "Aceite de oliva virgen extra en spray"),
    # ── Diosas de Abril ──
    ("Diosas de Abril", "Botella Picual 500 Ml", 385, "Aceite de oliva virgen extra varietal Picual (Puro Fuego)"),
    ("Diosas de Abril", "Botella Arbequina 500 Ml", 385, "Aceite de oliva virgen extra varietal Arbequina (Dulce Calma)"),
    # ── Otras marcas ──
    ("Carbonell", "Botella Virgen Extra 750 Ml", 795, "Aceite de oliva virgen extra (España)"),
    ("Barbera", "Botella Extra Virgen 250 Ml", 418, "Aceite de oliva extra virgen italiano, selezione unica"),
    ("Barbera", "Botella Extra Virgen 500 Ml", 685, "Aceite de oliva extra virgen italiano, selezione unica"),
    ("Barbera", "Botella Extra Virgen 750 Ml", 915, "Aceite de oliva extra virgen italiano, selezione unica"),
    ("Filippo Berio", "Botella Extra Virgen 500 Ml", 695, "Aceite de oliva extra virgen italiano"),
    ("PAM", "Spray Olive Oil 5 Oz", 375, "Aceite de oliva puro en spray antiadherente (no stick)"),
    ("PAM", "Spray Extra Virgin Orgánico 5 Oz", 435, "Aceite de oliva extra virgen orgánico en spray"),
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
    print(f"Seed aceite de oliva: {len(ROWS)} SKUs · {len(brands)} marcas.")
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
