"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de ORÉGANO.

Decimotercera familia con variantes de MARCA del Supermercado RD: 18 SKUs transcritos
del catálogo de La Sirena (capturas del owner, 2026-07-02):

  * "Orégano" seco (15): Badia (molido/entero, sobres y frascos hasta 12 Oz), Oriente,
    Wala, Constanza, granel por libra (molido/entero, genérico).
  * "Orégano fresco" (1): Lucas Pérez en funda — food propio (hierba fresca ≠ especia
    seca para la sustitución en lista de compras; mismo criterio que cilantro/perejil).
  * BONUS del search (SKUs legítimos): Tonnino filetes de atún con orégano en aceite
    de oliva (food "Atún en aceite", master→"Atún en agua" como alternativa premium)
    y queso de oveja curado La Leyenda con tomate y orégano (food "Queso de oveja").

EXCLUIDOS a propósito: galletas/chips/tostadas con orégano (Molino del Sol, Maretti,
Kikaboni, Crich ×2 — snacks) y tomates enlatados con orégano (Tomaticos La Famosa,
Hunt's Diced — familia tomate, pendiente para su propio seed).

Nota: Wala aparece con "Frasco Molido 3 Oz" (RD$92) y "Frasco Molido 90 G" (RD$99) —
listados distintos en La Sirena (85 gr vs 90 gr); se cargan ambos, el owner puede
consolidar desde la admin UI si son el mismo producto.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_oregano_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_oregano_2026_07_02.py --commit   # inserta
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
VEG = "Vegetales y verduras"
CARNE = "Carnes, pescados y mariscos"
LACTEO = "Lácteos y huevos"

MASTER_OVERRIDE = {
    "Atún en aceite": "Atún en agua",  # alternativa premium en el selector
}

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Orégano seco: Badia ──
    ("Orégano", "Badia", "Sobre Molido 0.5 Oz", 55, "Orégano molido (ground oregano), gluten free", COND),
    ("Orégano", "Badia", "Sobre Entero 0.5 Oz", 55, "Orégano entero en hojas, presentación de sobre", COND),
    ("Orégano", "Badia", "Frasco Entero 0.5 Oz", 105, "Orégano entero en hojas", COND),
    ("Orégano", "Badia", "Frasco Entero 5.5 Oz", 230, "Orégano entero en hojas, tamaño grande", COND),
    ("Orégano", "Badia", "Frasco Molido 1.5 Oz", 105, "Orégano molido (ground oregano)", COND),
    ("Orégano", "Badia", "Frasco Molido 12 Oz", 390, "Orégano molido, tamaño institucional", COND),
    # ── Oriente ──
    ("Orégano", "Oriente", "Frasco Molido 190 gr", 162, "Orégano molido (ground oregano)", COND),
    ("Orégano", "Oriente", "Frasco Molido 70 gr", 82, "Orégano molido", COND),
    ("Orégano", "Oriente", "Sobre 30 gr", 49, "Orégano en hojas, presentación de sobre", COND),
    # ── Wala ──
    ("Orégano", "Wala", "Sobre 45 gr", 39, "Orégano 100% puro, presentación de sobre", COND),
    ("Orégano", "Wala", "Frasco Molido 3 Oz", 92, "Orégano molido sin conservantes", COND),
    ("Orégano", "Wala", "Frasco Molido 90 gr", 99, "Orégano molido sin conservantes", COND),
    # ── Constanza ──
    ("Orégano", "Constanza", "Frasco Molido 3.2 Oz", 90, "Orégano molido natural (ground oregano)", COND),
    # ── Granel ──
    ("Orégano", None, "Molido Lb (granel)", 275, "Orégano molido a granel, por libra", COND),
    ("Orégano", None, "Entero Lb (granel)", 275, "Orégano entero en hojas a granel, por libra", COND),
    # ── Orégano fresco ──
    ("Orégano fresco", "Lucas Pérez", "Funda (unidad)", 59, "Orégano fresco en rama", VEG),
    # ── BONUS del search ──
    ("Atún en aceite", "Tonnino", "Frasco Filetes con Orégano 190 gr", 469, "Filetes de atún con orégano en aceite de oliva, frasco de vidrio", CARNE),
    ("Queso de oveja", "La Leyenda", "Paquete Curado Tomate y Orégano 200 gr", 400, "Queso de oveja curado saborizado con tomate y orégano", LACTEO),
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
    print(f"Seed orégano: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, brand, pres, price, desc, category) in ROWS:
                master = MASTER_OVERRIDE.get(food, food)
                cur.execute(_INSERT, (food, brand, pres, price, NOTES, category, master, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
