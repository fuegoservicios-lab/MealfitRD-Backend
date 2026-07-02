"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de PIMIENTA NEGRA.

Segunda familia con variantes de MARCA del Supermercado RD: 11 SKUs transcritos del
catálogo de La Sirena (capturas del owner, 2026-07-02) — Badia (molida/entera/orgánica/
molinillo/fina, frascos y sobres), Wala, Oriente y granel por libra (genérico).

EXCLUIDO a propósito: "Papas Fritas Hal's NY con sal marina y pimienta" (snack que el
search de La Sirena devuelve por el nombre — no es pimienta).

Mismo patrón que seed_supermarket_leches_2026_07_02.py. Idempotente: ON CONFLICT
(variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_pimienta_negra_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_pimienta_negra_2026_07_02.py --commit   # inserta
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
FOOD = "Pimienta negra"
CATEGORY = "Condimentos y especias"

# (brand | None = genérico/granel, presentation, price_rd, description)
ROWS = [
    ("Badia", "Frasco Molida 2 Oz", 139, "Pimienta negra molida"),
    ("Badia", "Sobre Molida 0.5 Oz", 59, "Pimienta negra molida, presentación de sobre"),
    ("Badia", "Frasco Molida 16 Oz", 750, "Pimienta negra molida, tamaño grande"),
    ("Badia", "Frasco Orgánica Molida 2 Oz", 180, "Pimienta negra molida orgánica certificada"),
    ("Badia", "Molinillo 2.25 Oz", 199, "Pimienta negra entera con molinillo (grinder)"),
    ("Badia", "Frasco Entera 2 Oz", 115, "Pimienta negra entera en granos"),
    ("Badia", "Sobre Entera 0.5 Oz", 48, "Pimienta negra entera en granos, presentación de sobre"),
    ("Badia", "Frasco Molida Fina 16 Oz", 780, "Pimienta negra molida fina, tamaño grande"),
    ("Wala", "Frasco Entera 70 gr", 89, "Pimienta negra entera, 100% pimienta"),
    ("Oriente", "Frasco Molida 224 gr", 395, "Pimienta negra molida"),
    (None, "Lb (granel)", 585, "Pimienta negra en polvo a granel, por libra"),
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

    brands = {r[0] for r in ROWS if r[0]}
    print(f"Seed pimienta negra: {len(ROWS)} SKUs · {len(brands)} marcas (+granel genérico).")
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
