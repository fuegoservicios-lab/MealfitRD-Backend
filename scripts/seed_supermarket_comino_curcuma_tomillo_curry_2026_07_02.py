"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de COMINO +
CÚRCUMA MOLIDA + TOMILLO + CURRY EN POLVO.

Familias 113-116 del Supermercado RD (capturas del owner, 2026-07-02,
La Sirena). LAUREL: el owner nombró la familia pero no llegó captura —
el genérico "Pote 100 gr" RD$150 queda como está.

COMINO (6):
  * CALCE: Badia molido sobre 1 Oz RD$55 = genérico "Pote 1 Oz".
  * Badia ×4 (molido/entero en sobre 1 Oz y frasco 16 Oz), Oriente ×2
    (sobre entero 30 gr, frasco molido 56 gr).
  * EXCLUIDO el "Adobo Con Comino Goya 12 Oz" RD$129 — product-truth es
    adobo (sazón completo), no comino, y no existe food "Adobo" en el
    catálogo (criterio barras de chocolate: si el owner quiere la
    familia adobo/sazones, cargarla como familia propia).

CÚRCUMA MOLIDA (food NUEVO, 3): la especia seca ≠ raíz fresca (criterio
jengibre molido). El genérico "Cúrcuma Lb" RD$99 ES la fresca (calce con
"Curcuma Fresca Lb", no duplicar). Entran: molida a granel Lb RD$735
(sin marca), Badia polvo orgánica 2 Oz RD$200, Badia molida 16 Oz RD$365.

TOMILLO (2): DOBLE CALCE — Badia sobre 0.5 Oz RD$55 = genérico "Sobre
0.5 Oz" y Badia entero 8 Oz RD$325 = genérico "Frasco 8 Oz". Entran las
2 filas de marca.

CURRY EN POLVO (3): CALCE — Badia polvo 2 Oz RD$100 = genérico "Frasco
2 Oz". + Oriente frasco 70 gr RD$215 y Badia 16 Oz RD$405.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_comino_curcuma_tomillo_curry_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_comino_curcuma_tomillo_curry_2026_07_02.py --commit   # inserta
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

# (food_name, brand, presentation, price_rd, description)
ROWS = [
    # ── Comino ──
    ("Comino", "Badia", "Sobre Molido 1 Oz", 55, "Comino molido (ground cumin)"),
    ("Comino", "Badia", "Sobre Entero 1 Oz", 50, "Comino entero en grano (cumin seed)"),
    ("Comino", "Badia", "Frasco Molido 16 Oz", 475, "Comino molido, tamaño institucional"),
    ("Comino", "Badia", "Frasco Entero 16 Oz", 495, "Comino entero en grano, tamaño institucional"),
    ("Comino", "Oriente", "Sobre Entero 30 gr", 54, "Comino entero en grano"),
    ("Comino", "Oriente", "Frasco Molido 56 gr", 159, "Comino molido"),
    # ── Cúrcuma molida (food nuevo — la fresca queda en "Cúrcuma") ──
    ("Cúrcuma molida", None, "Molida Lb", 735, "Cúrcuma molida a granel, venta por libra"),
    ("Cúrcuma molida", "Badia", "Frasco Polvo Orgánica 2 Oz", 200, "Cúrcuma en polvo orgánica (organic ground turmeric)"),
    ("Cúrcuma molida", "Badia", "Frasco Molida 16 Oz", 365, "Cúrcuma molida (ground turmeric), tamaño institucional"),
    # ── Tomillo ──
    ("Tomillo", "Badia", "Sobre 0.5 Oz", 55, "Tomillo seco (thyme)"),
    ("Tomillo", "Badia", "Frasco Entero 8 Oz", 325, "Tomillo entero seco (thyme), tamaño institucional"),
    # ── Curry en polvo ──
    ("Curry en polvo", "Badia", "Frasco Polvo 2 Oz", 100, "Curry en polvo (curry powder)"),
    ("Curry en polvo", "Badia", "Frasco 16 Oz", 405, "Curry en polvo, tamaño institucional"),
    ("Curry en polvo", "Oriente", "Frasco 70 gr", 215, "Curry en polvo"),
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
    print(f"Seed comino + cúrcuma molida + tomillo + curry: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, brand, pres, price, desc) in ROWS:
                cur.execute(_INSERT, (food, brand, pres, price, NOTES, COND, food, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
