"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de MUSLO DE POLLO +
HÍGADO DE RES + SALMÓN + TILAPIA + PAVO MOLIDO.

Familias 104-108 del Supermercado RD (capturas del owner, 2026-07-02).

MUSLO DE POLLO (14, La Sirena — carnicería: fotos stock, el TÍTULO manda):
  * CALCE: Unipollo ancho congelado Lb RD$68 = genérico "Lb".
  * Don Pollo ×7 (entero congelado — título sin "Lb", per-libra por
    paridad de precio con sus hermanos —, entero fresco y granel, bate
    fresco y granel, deshuesado, corto), Pollo Cibao ×5 (bate "ver
    mínimo", entero fresco/congelado, ancho fresco/congelado), importado
    congelado sin marca RD$60.

HÍGADO DE RES (1, La Sirena): el fresco Lb RD$119 ES el genérico (calce,
no duplicar). Entra el importado Lb RD$91 como variante con nombre.

SALMÓN (15, La Sirena):
  * DOBLE CALCE: Gourmar ahumado 3 Oz RD$490 = genérico "Paquete 3 Oz" y
    Gourmar precocido soya y sésamo 11.4 Oz RD$1,060 = genérico
    "Paquete 11.4 Oz".
  * Gourmar ×6 (ahumado 3 Oz/1 Lb/con especias, precocido, filete 7 Oz,
    cubos 600 gr), Wala ×2 (filete 16 Oz, hamburguesas 680 gr — producto
    de salmón, queda en familia con descripción honesta), Vima ×2
    (ahumado precortado, filete coho), Netuno salar chileno, Orca Bay
    keta, Panamei ahumado 8 Oz, salar premium y rodaja chum sin marca.
  * EXCLUIDOS: 4 alimentos de mascota sabor salmón (Friskies, Felix,
    Master Cat ×2 — no es comida humana) y la PINTURA King Superlatex
    color "Salmon 95" (no es alimento). Las 2 aceitunas rellenas de
    salmón (Goya 5.25 Oz, Maestranza 300 gr) YA estaban cargadas bajo
    "Aceituna" — 0 filas.

TILAPIA (3, La Sirena): la negra importada Lb RD$130 ES el genérico
(calce, no duplicar). Entran Wala negra 2 Lb RD$272, Wala filete 1.5 Lb
RD$350 y Panamei filete 16 Oz RD$795.

PAVO MOLIDO (1, **Supermercados Nacional** — notes por fila): Butterball
molida para tacos (taco seasoned) 16 Oz $369.95. El genérico "Paquete
16 Oz" RD$320 no calza con esta variante sazonada.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_muslo_higado_salmon_tilapia_pavo_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_muslo_higado_salmon_tilapia_pavo_2026_07_02.py --commit   # inserta
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

SIRENA = "Precio de referencia La Sirena · 2026-07"
NACIONAL = "Precio de referencia Supermercados Nacional · 2026-07"
CARNES = "Carnes, pescados y mariscos"

MUSLO = "Muslo de pollo"
SALMON = "Salmón"
TILAPIA = "Tilapia"

# (food_name, brand, presentation, price_rd, description, notes)
ROWS = [
    # ── Muslo de pollo · Don Pollo ──
    (MUSLO, "Don Pollo", "Entero Congelado Lb", 75, "Muslo de pollo entero congelado (el título del listing no trae unidad; per-libra por paridad)", SIRENA),
    (MUSLO, "Don Pollo", "Entero Fresco Lb", 77, "Muslo de pollo entero fresco", SIRENA),
    (MUSLO, "Don Pollo", "Entero Fresco Granel Lb", 75, "Muslo de pollo entero fresco a granel", SIRENA),
    (MUSLO, "Don Pollo", "Bate Fresco Lb", 115, "Bate (baqueta) de pollo fresco", SIRENA),
    (MUSLO, "Don Pollo", "Bate Fresco a Granel Lb", 112, "Bate (baqueta) de pollo fresco a granel", SIRENA),
    (MUSLO, "Don Pollo", "Deshuesado Fresco Lb", 164, "Muslo de pollo deshuesado fresco", SIRENA),
    (MUSLO, "Don Pollo", "Corto Fresco Lb", 70, "Muslo corto de pollo fresco", SIRENA),
    # ── Muslo de pollo · Pollo Cibao ──
    (MUSLO, "Pollo Cibao", "Bate Lb", 115, "Bate (baqueta) de pollo (ver mínimo de compra)", SIRENA),
    (MUSLO, "Pollo Cibao", "Entero Lb", 77, "Muslo de pollo entero", SIRENA),
    (MUSLO, "Pollo Cibao", "Entero Congelado Lb", 80, "Muslo de pollo entero congelado", SIRENA),
    (MUSLO, "Pollo Cibao", "Ancho Lb", 70, "Muslo ancho de pollo", SIRENA),
    (MUSLO, "Pollo Cibao", "Ancho Congelado Lb", 71, "Muslo ancho de pollo congelado", SIRENA),
    # ── Muslo de pollo · resto ──
    (MUSLO, "Unipollo", "Ancho Congelado Lb", 68, "Muslo ancho de pollo congelado", SIRENA),
    (MUSLO, None, "Importado Congelado Lb", 60, "Muslo de pollo importado congelado", SIRENA),
    # ── Hígado de res ──
    ("Hígado de res", None, "Importado Lb", 91, "Hígado de res importado, venta por libra", SIRENA),
    # ── Salmón · Gourmar ──
    (SALMON, "Gourmar", "Paquete Ahumado 3 Oz", 490, "Salmón ahumado", SIRENA),
    (SALMON, "Gourmar", "Paquete Ahumado con Especias 3 Oz", 509, "Salmón ahumado con especias", SIRENA),
    (SALMON, "Gourmar", "Paquete Ahumado 1 Lb", 1755, "Salmón ahumado", SIRENA),
    (SALMON, "Gourmar", "Paquete Precocido Soya y Sésamo 11.4 Oz", 1060, "Salmón precocido marinado en soya y sésamo", SIRENA),
    (SALMON, "Gourmar", "Filete 7 Oz", 580, "Filete de salmón", SIRENA),
    (SALMON, "Gourmar", "Funda en Cubos 600 gr", 995, "Salmón en cubos", SIRENA),
    # ── Salmón · resto ──
    (SALMON, "Vima", "Paquete Ahumado Precortado 100 gr", 419, "Salmón ahumado precortado (jamón ahumado noruego)", SIRENA),
    (SALMON, "Vima", "Filete Chileno Coho Lb", 650, "Filete de salmón chileno coho", SIRENA),
    (SALMON, "Wala", "Filete 16 Oz", 854, "Filete de salmón atlántico", SIRENA),
    (SALMON, "Wala", "Funda Hamburguesas 680 gr", 550, "Hamburguesas de salmón (procesado)", SIRENA),
    (SALMON, "Netuno", "Filete Salar Chileno Lb", 745, "Filete de salmón salar chileno", SIRENA),
    (SALMON, "Orca Bay", "Paquete Filetes Keta 10 Oz", 715, "Filetes de salmón keta", SIRENA),
    (SALMON, "Panamei", "Paquete Ahumado 8 Oz", 945, "Salmón ahumado", SIRENA),
    (SALMON, None, "Salar Chileno Premium Lb", 715, "Salmón salar chileno premium, venta por libra", SIRENA),
    (SALMON, None, "Rodaja Chum Congelado Lb", 319, "Rodaja de salmón chum congelado, venta por libra", SIRENA),
    # ── Tilapia ──
    (TILAPIA, "Wala", "Funda Negra 2 Lb", 272, "Tilapia negra entera", SIRENA),
    (TILAPIA, "Wala", "Funda Filete 1.5 Lb", 350, "Filete de tilapia", SIRENA),
    (TILAPIA, "Panamei", "Paquete Filete 16 Oz", 795, "Filete de tilapia", SIRENA),
    # ── Pavo molido (Supermercados Nacional) ──
    ("Pavo molido", "Butterball", "Tubo para Tacos 16 Oz", 369.95, "Carne de pavo molida sazonada para tacos (taco seasoned)", NACIONAL),
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
    print(f"Seed muslo + hígado + salmón + tilapia + pavo: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, brand, pres, price, desc, notes) in ROWS:
                cur.execute(_INSERT, (food, brand, pres, price, notes, CARNES, food, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
