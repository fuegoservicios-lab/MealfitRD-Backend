"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de AJO, AJÍ MORRÓN y BATATA.

Familias 40-42 del Supermercado RD: 16 SKUs transcritos del catálogo de La Sirena
(capturas del owner, 2026-07-02):

  * "Ajo" (1): empacado por libra (el paquete 4 unid RD$60 calza exacto con el
    genérico — no se duplica).
  * "Ajo en polvo" (1): Badia 3 Oz — calza exacto con el genérico RD$115, entra
    como variante con marca (criterio Wala vinagre).
  * "Ajo en pasta" (food NUEVO, 5): Wala (15/4.5 Oz), Constanza (15/8 Oz),
    Baldom (15.5 Oz). Condimento procesado ≠ ajo fresco (criterio Ajo y perejil).
    Constanza 15 Oz a precio de LISTA RD$225 (promo -9% RD$205).
  * "Ají morrón" (6): los 4 colores frescos por libra (RD$75) + 2 porcionados en
    vaso ("No disponible", referencia).
  * "Batata" (3): canolia y fresca a RD$32/Lb (AMBAS calzan exacto con el
    genérico) + asada RD$101/Lb.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_ajo_morron_batata_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_ajo_morron_batata_2026_07_02.py --commit   # inserta
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
VIV = "Víveres y tubérculos"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Ajo ──
    ("Ajo", None, "Empacado Lb", 119, "Ajo selecto empacado, por libra", VEG),
    # ── Ajo en polvo ──
    ("Ajo en polvo", "Badia", "Frasco Garlic Powder 3 Oz", 115, "Ajo en polvo (garlic powder)", COND),
    # ── Ajo en pasta (food nuevo) ──
    ("Ajo en pasta", "Wala", "Pote 15 Oz", 118, "Pasta de ajo", COND),
    ("Ajo en pasta", "Wala", "Pote 4.5 Oz", 59, "Pasta de ajo", COND),
    ("Ajo en pasta", "Constanza", "Pote 15 Oz", 225, "Pasta de ajo estilo criollo", COND),
    ("Ajo en pasta", "Constanza", "Pote 8 Oz", 125, "Pasta de ajo estilo criollo", COND),
    ("Ajo en pasta", "Baldom", "Pote 15.5 Oz", 199, "Pasta de ajo (garlic paste), envasado al vacío", COND),
    # ── Ají morrón ──
    ("Ají morrón", None, "Rojo Lb", 75, "Ají morrón rojo fresco, por libra", VEG),
    ("Ají morrón", None, "Amarillo Lb", 75, "Ají morrón amarillo fresco, por libra", VEG),
    ("Ají morrón", None, "Naranja Lb", 75, "Ají morrón naranja fresco, por libra", VEG),
    ("Ají morrón", None, "Verde Lb", 75, "Ají morrón verde fresco, por libra", VEG),
    ("Ají morrón", None, "Vaso Rojo Porcionado (unidad)", 108, "Ají morrón rojo porcionado en tiras, listo para cocinar", VEG),
    ("Ají morrón", None, "Vaso Verde Porcionado (unidad)", 108, "Ají morrón verde porcionado en tiras, listo para cocinar", VEG),
    # ── Batata ──
    ("Batata", None, "Canolia Lb", 32, "Batata canolia fresca, por libra", VIV),
    ("Batata", None, "Fresca Lb", 32, "Batata fresca, por libra", VIV),
    ("Batata", None, "Asada Lb", 101, "Batata asada lista para comer, por libra", VIV),
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
    print(f"Seed ajo + ají morrón + batata: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
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
