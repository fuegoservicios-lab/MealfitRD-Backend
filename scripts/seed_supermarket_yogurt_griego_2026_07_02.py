"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de YOGURT GRIEGO.

Trigésima familia con variantes de MARCA del Supermercado RD: 18 SKUs transcritos
del catálogo de La Sirena (capturas del owner, 2026-07-02) — Yoplait Griego (7 vasos
150 gr + mix 4-pack + 3 bebibles 220 ml), Yoka Griego (vaso vainilla + 4 bebibles
8 Oz) y Odyssey (2 vasos 5.3 Oz).

Notas:
  * El genérico del PDF calza exacto: Pote 150gr RD$100 = vasos Yoplait Griego.
  * EXCLUIDO 1 listado: Yoplait Kiw-Chia-Manz 150G aparece con precio RD$0.00 en
    La Sirena (dato inválido — no sirve como precio de referencia). Cuando tenga
    precio real, cargarlo via admin UI o próximo seed.
  * Yoplait Manzana y Canela "No disponible" incluido como referencia (precio real).
  * Este es EL caso de uso original del roadmap: "que el cliente pueda elegir
    la marca de su yogurt".

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_yogurt_griego_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_yogurt_griego_2026_07_02.py --commit   # inserta
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
FOOD = "Yogurt Griego"
CATEGORY = "Lácteos y huevos"

# (brand, presentation, price_rd, description)
ROWS = [
    # ── Yoplait Griego: vasos 150 gr ──
    ("Yoplait", "Vaso Natural 0% 150 gr", 100, "Yogurt griego natural sin grasa, 9g de proteína por envase"),
    ("Yoplait", "Vaso Fresa 150 gr", 100, "Yogurt griego con fresa"),
    ("Yoplait", "Vaso Chinola 0% 150 gr", 100, "Yogurt griego con chinola, sin grasa"),
    ("Yoplait", "Vaso Mora y Cereales 0% 150 gr", 100, "Yogurt griego con mora y cereales, sin grasa"),
    ("Yoplait", "Vaso Vainilla y Coco 0% 150 gr", 100, "Yogurt griego sabor vainilla y coco, sin grasa, 3x proteína"),
    ("Yoplait", "Vaso Piña Manzana y Nuez 150 gr", 100, "Yogurt griego con piña, manzana y nuez"),
    ("Yoplait", "Vaso Manzana y Canela 150 gr", 100, "Yogurt griego con manzana y canela"),
    ("Yoplait", "Mix 4 Vasos 150 gr", 365, "Mix de 4 yogurts griegos 150 gr (natural y frutos rojos)"),
    # ── Yoplait Griego: bebibles 220 ml ──
    ("Yoplait", "Bebible Natural 0% 220 Ml", 85, "Yogurt griego bebible natural, sin grasa"),
    ("Yoplait", "Bebible Fresa 0% 220 Ml", 85, "Yogurt griego bebible de fresa, sin grasa"),
    ("Yoplait", "Bebible Fresa y Guineo 220 Ml", 85, "Yogurt griego bebible de fresa y guineo"),
    # ── Yoka Griego ──
    ("Yoka", "Vaso Vainilla 0% 150 gr", 102, "Yogurt griego sabor vainilla, sin grasa"),
    ("Yoka", "Bebible Natural 8 Oz", 95, "Yogurt griego bebible natural"),
    ("Yoka", "Bebible Vainilla 8 Oz", 95, "Yogurt griego bebible de vainilla"),
    ("Yoka", "Bebible Fresa y Guineo 8 Oz", 95, "Yogurt griego bebible de fresa y guineo"),
    ("Yoka", "Bebible Coco 8 Oz", 95, "Yogurt griego bebible de coco"),
    # ── Odyssey ──
    ("Odyssey", "Vaso Natural 0% 5.3 Oz", 110, "Greek yogurt natural, sin grasa"),
    ("Odyssey", "Vaso Fresas 0% 5.3 Oz", 110, "Greek yogurt con fresas, sin grasa"),
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
    print(f"Seed yogurt griego: {len(ROWS)} SKUs · {len(brands)} marcas.")
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
