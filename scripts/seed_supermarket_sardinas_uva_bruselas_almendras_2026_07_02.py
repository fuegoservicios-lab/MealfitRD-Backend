"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de SARDINAS EN LATA
+ UVA + COLES DE BRUSELAS + LECHE DE ALMENDRAS (complemento).

Familias 100-103 del Supermercado RD (capturas del owner, 2026-07-02).

SARDINAS EN LATA (38, La Sirena):
  * DOBLE CALCE con genéricos: Zerca aceite 125 gr RD$33 = genérico
    "Lata 125 gr" y Zerca tomate tall 15 Oz RD$57 = genérico "Lata 15 Oz".
  * Paco Fish ×17 (aceite/agua 125 gr, tomate 200/215 gr y tall/ovalado
    15 Oz-425 gr, aceite vegetal 215 gr y 15 Oz, para locrio 215 gr y
    15 Oz, tomate y maíz, tomate maíz y coco, picantes, con vegetales),
    Cherry Star ×7 (106 gr: agua, tomate, salsa picante, mostaza, aceite,
    aceite picante, aceite de oliva), Zerca ×3, Dimar ×3 (promos -23%/-8%
    etc. a precio de LISTA), Brunswick ×4 (sin espinas agua de manantial
    — 2 listings duplicados del store DEDUPE a 1 fila, hot peppers,
    salsa tomate, aceite de soya 95 gr), Albo ×2 (oliva/tomate 120 gr),
    Cabo de Peñas ×2 (escabeche y escabeche con limón y aceite girasol).
  * "Sardina Criolla Lb" RD$78 → food NUEVO "Sardina fresca" (pescado
    fresco por libra ≠ conserva; criterio fresco≠seco/enlatado).

UVA (1, La Sirena): CALCE del genérico "Lb" RD$169 = Uva Roja Lb (Red
Globe) — el único tipo con precio real ES el genérico, no duplicar. Entra
el clamshell CMX Fruit roja 2 Lb RD$450. EXCLUIDA la "Uva Verde Sin
Semilla Lb" a RD$3,582.00 — error evidente del listing (10-20× el precio
de mercado); FLAG al owner (mismo criterio que los RD$0.00).

COLES DE BRUSELAS (3, **Supermercados Nacional** — notes por fila):
  * CALCE (con redondeo del PDF): genérico "Paquete 900 gr" RD$220 =
    Emborg congeladas 900 gr $219.95.
  * Fresco importado por paquete (5-6 unid/Lb, sin marca) $229.95 y
    Food Club petite congeladas 16 Oz $199.95.

LECHE DE ALMENDRAS (2, La Sirena): de las capturas, Silk vainilla s/a,
Blue Diamond original, chocolate s/a y vainilla s/a YA estaban cargadas
con precios idénticos. Entra la Blue Diamond ORIGINAL SIN AZÚCAR 32 Oz
RD$289 (no disp, referencia) — variante unsweetened distinta de la
original. El "Dos Pinos Leche + Proteína choco almendras 250 Ml" RD$81 →
product-truth va bajo "Leche" (bebida láctea con proteína saborizada, no
leche de almendras). EXCLUIDOS los 2 chocolates de barra (Torras leche y
almendras s/a RD$170, Chokamil blanco 90 G RD$285): son chocolate, no
leche de almendras, y no existe food "Chocolate" en el catálogo — si el
owner quiere esa familia, cargarla como familia propia.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_sardinas_uva_bruselas_almendras_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_sardinas_uva_bruselas_almendras_2026_07_02.py --commit   # inserta
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
PESC = "Carnes, pescados y mariscos"
FRUTAS = "Frutas"
VEG = "Vegetales y verduras"
LACTEOS = "Lácteos y huevos"

SAR = "Sardinas en lata"

# (food_name, brand, presentation, price_rd, description, category, notes)
ROWS = [
    # ── Sardinas en lata · Paco Fish ──
    (SAR, "Paco Fish", "Lata en Aceite 125 gr", 45, "Sardinas en aceite", PESC, SIRENA),
    (SAR, "Paco Fish", "Lata en Agua 125 gr", 45, "Sardinas en agua", PESC, SIRENA),
    (SAR, "Paco Fish", "Lata Tall en Salsa de Tomate 15 Oz", 125, "Sardinas en salsa de tomate", PESC, SIRENA),
    (SAR, "Paco Fish", "Lata Tall en Aceite Vegetal 15 Oz", 129, "Sardinas en aceite vegetal (precio de lista, promo -8%)", PESC, SIRENA),
    (SAR, "Paco Fish", "Lata Tall en Agua 15 Oz", 135, "Sardinas en agua", PESC, SIRENA),
    (SAR, "Paco Fish", "Lata Tall Para Locrio 15 Oz", 125, "Sardinas especial para locrio", PESC, SIRENA),
    (SAR, "Paco Fish", "Lata Para Locrio 215 gr", 74, "Sardinas especial para locrio", PESC, SIRENA),
    (SAR, "Paco Fish", "Lata Tall Tomate y Maíz 15 Oz", 135, "Sardinas en salsa de tomate con maíz (precio de lista, promo -7%)", PESC, SIRENA),
    (SAR, "Paco Fish", "Lata Tall Tomate Maíz y Coco 15 Oz", 135, "Sardinas en salsa de tomate con maíz y coco", PESC, SIRENA),
    (SAR, "Paco Fish", "Lata en Tomate 200 gr", 68, "Sardinas en salsa de tomate", PESC, SIRENA),
    (SAR, "Paco Fish", "Lata en Tomate 215 gr", 75, "Sardinas en salsa de tomate", PESC, SIRENA),
    (SAR, "Paco Fish", "Lata Ovalada en Tomate 425 gr", 120, "Sardinas en salsa de tomate, lata ovalada", PESC, SIRENA),
    (SAR, "Paco Fish", "Lata en Aceite Vegetal 215 gr", 75, "Sardinas en aceite vegetal", PESC, SIRENA),
    (SAR, "Paco Fish", "Lata Tomate Picante 215 gr", 74, "Sardinas en salsa de tomate picante", PESC, SIRENA),
    (SAR, "Paco Fish", "Lata en Salsa Tomate Picante 10 Oz", 125, "Sardinas en salsa de tomate picante (precio de lista, promo -8%)", PESC, SIRENA),
    (SAR, "Paco Fish", "Lata Ovalada Picante 15 Oz", 122, "Sardinas picantes, lata ovalada", PESC, SIRENA),
    (SAR, "Paco Fish", "Lata con Vegetales 142 gr", 80, "Sardinas con vegetales", PESC, SIRENA),
    # ── Sardinas en lata · Cherry Star (106 gr) ──
    (SAR, "Cherry Star", "Lata en Agua 106 gr", 58, "Sardinas en agua", PESC, SIRENA),
    (SAR, "Cherry Star", "Lata en Tomate 106 gr", 58, "Sardinas en salsa de tomate", PESC, SIRENA),
    (SAR, "Cherry Star", "Lata Salsa Picante 106 gr", 58, "Sardinas en salsa picante (precio de lista, promo -10%)", PESC, SIRENA),
    (SAR, "Cherry Star", "Lata Mostaza 106 gr", 58, "Sardinas en mostaza (precio de lista, promo -10%)", PESC, SIRENA),
    (SAR, "Cherry Star", "Lata en Aceite 106 gr", 58, "Sardinas en aceite", PESC, SIRENA),
    (SAR, "Cherry Star", "Lata Aceite Picante 106 gr", 58, "Sardinas en aceite picante", PESC, SIRENA),
    (SAR, "Cherry Star", "Lata Aceite de Oliva 106 gr", 85, "Sardinas en aceite de oliva", PESC, SIRENA),
    # ── Sardinas en lata · Zerca ──
    (SAR, "Zerca", "Lata en Aceite 125 gr", 33, "Sardinas en aceite vegetal", PESC, SIRENA),
    (SAR, "Zerca", "Lata Tall en Tomate 15 Oz", 57, "Sardinas en salsa de tomate", PESC, SIRENA),
    (SAR, "Zerca", "Lata Tall Picante 15 Oz", 57, "Sardinas picantes", PESC, SIRENA),
    # ── Sardinas en lata · Dimar ──
    (SAR, "Dimar", "Lata en Aceite 125 gr", 52, "Sardinas en aceite (precio de lista, promo -23%)", PESC, SIRENA),
    (SAR, "Dimar", "Lata en Tomate 225 gr", 65, "Sardinas en salsa de tomate", PESC, SIRENA),
    (SAR, "Dimar", "Lata en Tomate 425 gr", 120, "Sardinas en salsa de tomate", PESC, SIRENA),
    # ── Sardinas en lata · Brunswick ──
    (SAR, "Brunswick", "Lata Sin Espinas en Agua de Manantial 106 gr", 86, "Sardinas sin espinas en agua de manantial (2 listings del store, deduplicado)", PESC, SIRENA),
    (SAR, "Brunswick", "Lata Sin Espinas Hot Peppers", 86, "Sardinas sin espinas con ajíes picantes", PESC, SIRENA),
    (SAR, "Brunswick", "Lata Salsa Tomate 106 gr", 86, "Sardinas en salsa de tomate", PESC, SIRENA),
    (SAR, "Brunswick", "Lata Selectas Aceite de Soya 95 gr", 63, "Sardinas selectas en aceite de soya", PESC, SIRENA),
    # ── Sardinas en lata · Albo / Cabo de Peñas ──
    (SAR, "Albo", "Lata en Aceite de Oliva 120 gr", 189, "Sardinas en aceite de oliva", PESC, SIRENA),
    (SAR, "Albo", "Lata en Tomate 120 gr", 189, "Sardinas en tomate", PESC, SIRENA),
    (SAR, "Cabo de Peñas", "Lata en Escabeche 120 gr", 125, "Sardinas en escabeche", PESC, SIRENA),
    (SAR, "Cabo de Peñas", "Lata Escabeche con Limón y Aceite de Girasol 120 gr", 105, "Sardinas en escabeche con limón y aceite de girasol", PESC, SIRENA),
    # ── Sardina fresca (food nuevo) ──
    ("Sardina fresca", None, "Criolla Lb", 78, "Sardina criolla fresca, venta por libra", PESC, SIRENA),
    # ── Uva ──
    ("Uva", "CMX Fruit", "Clamshell Roja 2 Lb", 450, "Uva roja en clamshell", FRUTAS, SIRENA),
    # ── Coles de Bruselas (Supermercados Nacional) ──
    ("Coles de Bruselas", "Emborg", "Funda Congeladas 900 gr", 219.95, "Coles de Bruselas congeladas (brussels sprouts)", VEG, NACIONAL),
    ("Coles de Bruselas", None, "Paquete Fresco (5-6 unid por Lb)", 229.95, "Coles de Bruselas frescas importadas", VEG, NACIONAL),
    ("Coles de Bruselas", "Food Club", "Funda Petite Congeladas 16 Oz", 199.95, "Coles de Bruselas petite congeladas", VEG, NACIONAL),
    # ── Leche de almendras (complemento) ──
    ("Leche de almendras", "Blue Diamond Almond Breeze", "Cartón Original sin azúcar 32 Oz", 289, "Leche de almendras original sin azúcar (unsweetened)", LACTEOS, SIRENA),
    # ── Leche (reclasificada: bebida láctea con proteína) ──
    ("Leche", "Dos Pinos", "Brik + Proteína Choco Almendras 250 Ml", 81, "Leche con proteína sabor chocolate y almendras (listada por el store bajo leche de almendras)", LACTEOS, SIRENA),
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
        if len(row) != 7:
            print(f"FATAL: fila con {len(row)} campos (esperados 7): {row[:3]}")
            sys.exit(1)
        (food, brand, pres, *_rest) = row
        key = (food.lower(), (brand or "").lower(), pres.lower())
        if key in seen:
            print(f"FATAL: fila duplicada en el seed: {food} / {brand} / {pres}")
            sys.exit(1)
        seen.add(key)

    foods = {r[0] for r in ROWS}
    brands = {r[1] for r in ROWS if r[1]}
    print(f"Seed sardinas + uva + bruselas + almendras: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, brand, pres, price, desc, category, notes) in ROWS:
                cur.execute(_INSERT, (food, brand, pres, price, notes, category, food, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
