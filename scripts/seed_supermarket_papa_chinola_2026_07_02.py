"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de PAPA y CHINOLA.

Familias 66-67 del Supermercado RD: 6 SKUs transcritos del catálogo de La Sirena
(capturas del owner, 2026-07-02):

  * "Papa" (2): la Petite Malla 1 Lb RD$34 CALZA EXACTO con el genérico "Lb" —
    entra como tipo con nombre (criterio yautía); Selectas Lb RD$45 (grado
    superior).
  * "Chinola" (2): pulpas — Caña del Caribe botella 32 Oz RD$609 y Pachamama
    funda con semilla 2 Lb RD$699 (fruta procesada mínima, misma familia,
    criterio fresas congeladas). La "Lb" RD$79 calza exacto con el genérico —
    no se duplica.
  * "Néctar de chinola" (food NUEVO, 2): Barceló concentrado en galón RD$1,045
    y botella 30 Oz RD$299 — bebida/concentrado azucarado, no es la fruta
    (product truth manda; categoría Bebidas).

  EXCLUIDO: Zuko jugo en polvo "sabor a chinola" 20 gr RD$15 — mezcla en polvo
  con sabor artificial, no contiene chinola (criterio ramen/glaseado).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_papa_chinola_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_papa_chinola_2026_07_02.py --commit   # inserta
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
VIV = "Víveres y tubérculos"
FRU = "Frutas"
BEB = "Bebidas y alternativas vegetales"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Papa ──
    ("Papa", None, "Selectas Lb", 45, "Papas selectas frescas, por libra", VIV),
    ("Papa", None, "Malla Petite 1 Lb", 34, "Papas petite frescas en malla", VIV),
    # ── Chinola (pulpas) ──
    ("Chinola", "Caña del Caribe", "Botella Pulpa 32 Oz", 609, "Pulpa de chinola natural, botella", FRU),
    ("Chinola", "Pachamama", "Funda Pulpa con Semilla 2 Lb", 699, "Pulpa de chinola 100% natural con semilla", FRU),
    # ── Néctar de chinola (food nuevo — bebida concentrada) ──
    ("Néctar de chinola", "Barceló", "Galón Concentrado", 1045, "Concentrado de chinola para preparar jugo (néctar)", BEB),
    ("Néctar de chinola", "Barceló", "Botella Concentrado 30 Oz", 299, "Concentrado de chinola para preparar jugo (néctar)", BEB),
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
    print(f"Seed papa + chinola: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
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
