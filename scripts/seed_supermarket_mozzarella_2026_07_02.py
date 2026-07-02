"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de QUESO MOZZARELLA.

Vigesimonovena familia con variantes de MARCA del Supermercado RD: 17 SKUs transcritos
del catálogo de La Sirena (capturas del owner, 2026-07-02) — La Zarina (4: cereza/
trenza/corazón de leche + fresca por libra), BelGioioso (4: ball/pearls/pesto/sliced
log), Crystal Farms (3: fine shred/chunk/reduced fat), Kraft (rallado 8/16 Oz), Pauly,
Lupa (bocaditos), San Juan y Wala (barra).

Notas:
  * El genérico del PDF calza exacto: Lb RD$214 = Wala Barra Lb (el SKU "No
    disponible" con opción "Rebanado" — incluido como referencia).
  * Frescas en suero (tarros La Zarina/Lupa/BelGioioso) y baja humedad (barras,
    rallados) bajo el MISMO food con presentación/descripción honesta.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_mozzarella_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_mozzarella_2026_07_02.py --commit   # inserta
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
FOOD = "Queso mozzarella"
CATEGORY = "Lácteos y huevos"

# (brand, presentation, price_rd, description)
ROWS = [
    # ── La Zarina ──
    ("La Zarina", "Tarro Cereza de Leche 300 gr", 270, "Mozzarella fresca tipo cereza de leche, en suero"),
    ("La Zarina", "Tarro Trenza de Leche 300 gr", 270, "Mozzarella fresca tipo trenza de leche, en suero"),
    ("La Zarina", "Tarro Corazón de Leche 300 gr", 279, "Mozzarella fresca tipo corazón de leche, en suero"),
    ("La Zarina", "Fresca Lb", 290, "Mozzarella fresca, por libra"),
    # ── BelGioioso ──
    ("BelGioioso", "Bola Fresh Mozzarella 8 Oz", 235, "Mozzarella fresca en bola (fresh mozzarella)"),
    ("BelGioioso", "Tarro Pearls 8 Oz", 265, "Perlas de mozzarella fresca (pearls, little ones)"),
    ("BelGioioso", "Rollo Fresh Mozzarella con Pesto 8 Oz", 365, "Mozzarella fresca artesanal trenzada a mano con pesto"),
    ("BelGioioso", "Paquete Sliced Log 16 Oz", 460, "Mozzarella fresca rebanada (sliced log)"),
    # ── Crystal Farms ──
    ("Crystal Farms", "Funda Fine Shred 8 Oz", 285, "Queso mozzarella rallado fino (finely shredded)"),
    ("Crystal Farms", "Barra Chunk 8 Oz", 239, "Queso mozzarella en barra (chunk), baja humedad part-skim"),
    ("Crystal Farms", "Funda Shredded Reduced Fat 7 Oz", 295, "Queso mozzarella rallado reducido en grasa"),
    # ── Kraft ──
    ("Kraft", "Funda Rallado 8 Oz", 385, "Queso mozzarella rallado (shredded), baja humedad part-skim"),
    ("Kraft", "Funda Rallado 16 Oz", 595, "Queso mozzarella rallado (shredded), 1 Lb"),
    # ── Otras marcas ──
    ("Pauly", "Bandeja Rebanado Lb", 345, "Queso mozzarella rebanado, por libra"),
    ("Lupa", "Tarro Bocaditos 3/4 Lb", 305, "Bocaditos de mozzarella fresca (bocconcini de leche)"),
    ("San Juan", "Unidad (rueda)", 329, "Queso mozzarella dominicano, unidad"),
    ("Wala", "Barra Lb", 214, "Queso mozzarella en barra, por libra (rebanado disponible)"),
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
    print(f"Seed queso mozzarella: {len(ROWS)} SKUs · {len(brands)} marcas.")
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
