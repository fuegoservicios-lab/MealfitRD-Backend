"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de LECHE.

Primera familia con variantes de MARCA reales del Supermercado RD: 114 SKUs de
leche transcritos del catálogo de La Sirena (capturas del owner, 2026-07-02) —
Rica, Wala, Parmalat, Milex, Dos Pinos, Granarolo, La Granja, Kanny, Nestlé
(Carnation/Nido/Ideal/La Lechera/Alacta), Babina, Similac, Silk, Blue Diamond,
La Famosa, Goya, Nutra, Elle & Vire.

Agrupación por `food_name` (una tarjeta de producto por SKU en el catálogo):
  Leche (entera) · Leche semidescremada · Leche descremada · Leche sin lactosa ·
  Leche en polvo · Leche evaporada · Leche condensada · Leche saborizada ·
  Leche infantil y de crecimiento · Leche de almendras · Leche de coco ·
  Leche de soya · Crema de leche

EXCLUIDOS a propósito del search "leche" de La Sirena (no-alimento o no-leche):
leche de magnesia (laxante/desodorante), leche corporal/hidratante (cosmético),
barras Las 3 Rosas (dulcería), pan/galletas/chocolate "de leche".

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_leches_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_leches_2026_07_02.py --commit   # inserta
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

LACTEO = "Lácteos y huevos"
BEBVEG = "Bebidas y alternativas vegetales"

FOOD_CATEGORY = {
    "Leche": LACTEO,
    "Leche semidescremada": LACTEO,
    "Leche descremada": LACTEO,
    "Leche sin lactosa": LACTEO,
    "Leche en polvo": LACTEO,
    "Leche evaporada": LACTEO,
    "Leche condensada": LACTEO,
    "Leche saborizada": LACTEO,
    "Leche infantil y de crecimiento": LACTEO,
    "Crema de leche": LACTEO,
    "Leche de almendras": BEBVEG,
    "Leche de coco": BEBVEG,
    "Leche de soya": BEBVEG,
}

# (food_name, brand, presentation, price_rd, description)
ROWS = [
    # ── Leche entera ──
    ("Leche", "Wala", "Cartón 1 Lt", 59, "Leche entera UHT 3.1% grasa"),
    ("Leche", "Rica Listamilk", "Cartón UHT 1 Lt", 78, "Leche entera UHT grado A"),
    ("Leche", "Rica Listamilk", "Botella PET 1 Lt", 78, "Leche entera UHT larga vida"),
    ("Leche", "Rica Listamilk", "Cartón UHT 250 Ml", 30, "Leche entera UHT grado A"),
    ("Leche", "Rica La Vaquita", "Cartón 1 Lt", 69, "Leche entera"),
    ("Leche", "Parmalat", "Cartón UHT 1 Lt", 86, "Leche entera UHT"),
    ("Leche", "Granarolo", "Cartón UHT 1 Lt", 135, "Leche entera 3.6% grasa, larga vida (Italia)"),
    ("Leche", "Wala", "Cartón 200 Ml (3 pack)", 59, "Leche entera UHT, pack de 3 unidades"),
    # ── Leche semidescremada ──
    ("Leche semidescremada", "Wala", "Cartón 1 Lt", 52, "Leche semidescremada UHT 1.5% grasa"),
    ("Leche semidescremada", "Parmalat", "Cartón UHT 1 Lt", 86, "Leche semidescremada UHT"),
    ("Leche semidescremada", "Rica", "Cartón 1 Lt", 79, "Leche semidescremada 1% grasa"),
    ("Leche semidescremada", "Rica", "Botella PET 1 Lt", 79, "Leche semidescremada UHT larga vida"),
    ("Leche semidescremada", "Dos Pinos", "Cartón 1 Lt", 104, "Leche semidescremada"),
    ("Leche semidescremada", "Granarolo", "Cartón UHT 1 Lt", 120, "Leche semidescremada UHT larga vida (Italia)"),
    ("Leche semidescremada", "La Granja", "Cartón 1 Lt", 85, "Leche líquida 100% de vaca, 2% grasa"),
    # ── Leche descremada ──
    ("Leche descremada", "Wala", "Cartón 1 Lt", 49, "Leche descremada UHT 0.1% grasa"),
    ("Leche descremada", "Rica", "Cartón 1 Lt", 79, "Leche descremada 0% grasa"),
    ("Leche descremada", "Rica", "Botella PET 1 Lt", 79, "Leche descremada 0% UHT larga vida"),
    ("Leche descremada", "Milex", "Cartón 1 Lt", 89, "Leche descremada"),
    ("Leche descremada", "Dos Pinos", "Cartón 1 Lt", 104, "Leche descremada 0% grasa"),
    # ── Leche sin lactosa ──
    ("Leche sin lactosa", "Rica", "Cartón 1 Lt", 84, "Leche semidescremada sin lactosa, 2% grasa"),
    ("Leche sin lactosa", "Rica", "Botella PET 1 Lt", 86, "Leche deslactosada descremada"),
    ("Leche sin lactosa", "Dos Pinos Delactomy", "Cartón 1 Lt (semidescremada)", 104, "Leche semidescremada deslactosada 2% grasa, fácil de digerir"),
    ("Leche sin lactosa", "Dos Pinos Delactomy", "Cartón 1 Lt (descremada)", 104, "Leche descremada deslactosada 0% grasa"),
    ("Leche sin lactosa", "Milex", "Funda 800 gr (polvo)", 910, "Leche en polvo sin lactosa"),
    ("Leche sin lactosa", "Milex", "Funda 1500 gr (polvo)", 1659, "Leche en polvo sin lactosa"),
    ("Leche sin lactosa", "Milex", "Funda 360 gr (polvo)", 385, "Leche en polvo deslactosada"),
    # ── Leche en polvo ──
    ("Leche en polvo", "Wala", "Funda 900 gr", 365, "Leche entera en polvo"),
    ("Leche en polvo", "Wala", "Funda 2200 gr", 870, "Leche entera en polvo"),
    ("Leche en polvo", "Wala", "Funda 375 gr", 188, "Leche entera en polvo pasteurizada"),
    ("Leche en polvo", "Wala", "Sobre 125 gr", 65, "Leche entera en polvo"),
    ("Leche en polvo", "Rica", "Funda 1500 gr", 859, "Leche entera en polvo (whole milk powder)"),
    ("Leche en polvo", "Rica", "Funda 2200 gr", 1279, "Leche entera en polvo"),
    ("Leche en polvo", "Rica", "Funda 1000 gr", 599, "Leche entera en polvo"),
    ("Leche en polvo", "Milex", "Funda 1500 gr", 1495, "Leche entera en polvo"),
    ("Leche en polvo", "Milex", "Funda Instant 2200 gr", 1699, "Leche entera en polvo instantánea"),
    ("Leche en polvo", "Milex", "Funda Instantánea 800 gr", 660, "Leche entera en polvo instantánea"),
    ("Leche en polvo", "Milex", "Funda Clásica 1000 gr", 945, "Leche entera en polvo clásica"),
    ("Leche en polvo", "Milex", "Funda Clásica 2000 gr", 1970, "Leche entera en polvo clásica"),
    ("Leche en polvo", "Milex", "Sobre 125 gr", 110, "Leche entera en polvo"),
    ("Leche en polvo", "Milex", "Funda Mini 2200 gr", 1159, "Leche en polvo, línea económica"),
    ("Leche en polvo", "Milex Slim", "Funda 1600 gr", 1585, "Leche descremada en polvo"),
    ("Leche en polvo", "Kanny", "Funda 1500 gr", 995, "Leche entera en polvo full cream"),
    ("Leche en polvo", "Nestlé Ideal", "Bolsa 2.2 Kg", 1165, "Leche en polvo NutriForte con hierro, calcio y vitaminas"),
    ("Leche en polvo", "Nestlé Ideal", "Bolsa 800 gr", 450, "Leche en polvo NutriForte"),
    ("Leche en polvo", "Nestlé Ideal", "Bolsa 360 gr", 225, "Leche en polvo NutriForte"),
    # ── Leche evaporada ──
    ("Leche evaporada", "Nestlé Carnation", "Cartón UHT 290 Ml", 62, "Leche evaporada UHT, la más cremosa"),
    ("Leche evaporada", "Nestlé Carnation", "Lata 312 gr", 70, "Leche evaporada"),
    ("Leche evaporada", "Nestlé Carnation", "Lata 312 gr (6 pack)", 355, "Leche evaporada, pack de 6 latas"),
    ("Leche evaporada", "Nestlé Carnation", "Cartón UHT sabor queso 290 Ml", 105, "Leche evaporada parcialmente descremada sabor a queso"),
    ("Leche evaporada", "Nestlé Carnation", "Cartón UHT sabor queso 135 Ml", 59, "Leche evaporada sabor a queso"),
    ("Leche evaporada", "Wala", "Lata 377 gr", 64, "Leche evaporada enriquecida con vitaminas A y D"),
    ("Leche evaporada", "Rica", "Cartón UHT 350 Ml", 60, "Leche evaporada UHT"),
    ("Leche evaporada", "Rica", "Cartón 250 Ml", 46, "Leche evaporada enriquecida con vitaminas A y D"),
    ("Leche evaporada", "Rica", "Lata 410 gr", 70, "Leche evaporada enriquecida con vitaminas A y D"),
    ("Leche evaporada", "Rica", "Lata 170 gr", 42, "Leche evaporada"),
    ("Leche evaporada", "Rica", "Botella PET 1 Lt", 179, "Leche evaporada UHT larga vida"),
    ("Leche evaporada", "Rica", "Pack 350 gr (2 uds)", 240, "Leche evaporada, pack de 2"),
    # ── Leche condensada ──
    ("Leche condensada", "Wala", "Lata 396 gr", 109, "Leche condensada para postres y batidas"),
    ("Leche condensada", "Nestlé La Lechera", "Lata 403 gr", 130, "Leche condensada original"),
    ("Leche condensada", "Nestlé", "Lata Azucarada 395 gr", 109, "Leche condensada azucarada con grasa vegetal"),
    # ── Leche saborizada ──
    ("Leche saborizada", "Choco Rica", "Cartón 1 Lt", 114, "Leche semidescremada con chocolate"),
    ("Leche saborizada", "Choco Rica", "Cartón 200 Ml", 30, "Leche con chocolate"),
    ("Leche saborizada", "Choco Rica", "Cartón 200 Ml (6 pack)", 175, "Leche con chocolate, pack de 6"),
    ("Leche saborizada", "Parmalat", "Cartón Chocolate 200 Ml", 34, "Bebida láctea sabor chocolate"),
    ("Leche saborizada", "Parmalat", "Cartón Vainilla 200 Ml", 34, "Bebida láctea sabor vainilla"),
    ("Leche saborizada", "Parmalat", "Cartón con Avena 1 Lt", 124, "Leche UHT con avena (cereal)"),
    ("Leche saborizada", "Rica La Vaquita", "Cartón Chocolate 200 Ml", 29, "Leche semidescremada con chocolate, vitaminas A y D"),
    ("Leche saborizada", "Rica La Vaquita", "Cartón Bizcocho 200 Ml", 29, "Leche saborizada sabor bizcocho"),
    ("Leche saborizada", "Dos Pinos +Proteína", "Cartón Choco Almendras 250 Ml", 81, "Bebida láctea 50% más proteína, choco almendras, sin azúcar añadida"),
    # ── Leche infantil y de crecimiento ──
    ("Leche infantil y de crecimiento", "Nestlé Nido Crecimiento", "Bolsa 2 Kg", 1685, "Leche en polvo de crecimiento 1-5 años"),
    ("Leche infantil y de crecimiento", "Nestlé Nido Crecimiento", "Bolsa 1200 gr", 1095, "Leche en polvo de crecimiento"),
    ("Leche infantil y de crecimiento", "Nestlé Nido Crecimiento", "Bolsa 800 gr", 755, "Leche en polvo de crecimiento"),
    ("Leche infantil y de crecimiento", "Nestlé Nido Crecimiento", "Bolsa 325 gr", 315, "Leche en polvo de crecimiento"),
    ("Leche infantil y de crecimiento", "Nestlé Nido Crecimiento", "Lata 1600 gr", 1580, "Leche en polvo de crecimiento"),
    ("Leche infantil y de crecimiento", "Nestlé Nido FortiGrow", "Bolsa 360 gr", 319, "Leche en polvo fortificada edad escolar"),
    ("Leche infantil y de crecimiento", "Nestlé Nido FortiGrow", "Bolsa 800 gr", 695, "Leche en polvo fortificada edad escolar"),
    ("Leche infantil y de crecimiento", "Nestlé Nido FortiGrow", "Bolsa 1600 gr", 1330, "Leche en polvo fortificada edad escolar"),
    ("Leche infantil y de crecimiento", "Nestlé Nido FortiGrow", "Bolsa 2.2 Kg", 1655, "Leche en polvo fortificada edad escolar"),
    ("Leche infantil y de crecimiento", "Alacta Plus", "Lata 2200 gr", 2495, "Leche en polvo con DHA y hierro, desarrollo mental"),
    ("Leche infantil y de crecimiento", "Alacta Plus", "Lata 1650 gr", 1959, "Leche en polvo con DHA y hierro"),
    ("Leche infantil y de crecimiento", "Alacta Plus", "Lata 800 gr", 1169, "Leche en polvo con DHA y hierro"),
    ("Leche infantil y de crecimiento", "Alacta Plus", "Lata 375 gr", 619, "Leche en polvo con DHA y hierro"),
    ("Leche infantil y de crecimiento", "Alacta Kids", "Lata 800 gr", 1145, "Leche en polvo 5+ años"),
    ("Leche infantil y de crecimiento", "Alacta Kids", "Lata 1500 gr", 1914, "Leche en polvo 5+ años"),
    ("Leche infantil y de crecimiento", "Babina Plus", "Lata 900 gr", 1239, "Leche de crecimiento con vainilla (Suiza)"),
    ("Leche infantil y de crecimiento", "Babina Plus", "Lata 2000 gr", 2520, "Leche de crecimiento con vainilla (Suiza)"),
    ("Leche infantil y de crecimiento", "Babina Plus", "Lata 375 gr", 570, "Leche de crecimiento con vainilla (Suiza)"),
    ("Leche infantil y de crecimiento", "Milex Kinder", "Funda 2000 gr (sin azúcar)", 1835, "Leche en polvo infantil sin azúcar añadida"),
    ("Leche infantil y de crecimiento", "Milex Kinder", "Funda 800 gr", 840, "Leche en polvo infantil"),
    ("Leche infantil y de crecimiento", "Milex Kinder Gold", "Lata 2000 gr (sin azúcar)", 2405, "Leche en polvo infantil premium sin azúcar"),
    ("Leche infantil y de crecimiento", "Milex Kinder Gold", "Lata 1500 gr (sin azúcar)", 2089, "Leche en polvo infantil premium sin azúcar"),
    ("Leche infantil y de crecimiento", "Milex Kinder Gold", "Lata 800 gr (sin azúcar)", 1235, "Leche en polvo infantil premium sin azúcar"),
    ("Leche infantil y de crecimiento", "Milex Kinder Gold", "Lata 360 gr", 620, "Leche en polvo infantil premium"),
    ("Leche infantil y de crecimiento", "Milex M1", "Lata 400 gr", 410, "Fórmula infantil de inicio 0-6 meses, con prebióticos y DHA"),
    ("Leche infantil y de crecimiento", "Similac 1", "24 frascos 2 Oz (0-6 M)", 1774, "Fórmula infantil líquida lista para usar, 5 HMO"),
    ("Leche infantil y de crecimiento", "Nutra Junior", "Bolsa 400 gr", 405, "Bebida de soya en polvo para niños 2+ años"),
    # ── Leche de almendras ──
    ("Leche de almendras", "Silk Almond", "Cartón Vainilla sin azúcar 32 Oz", 260, "Bebida de almendras sabor vainilla, sin azúcar"),
    ("Leche de almendras", "Blue Diamond Almond Breeze", "Cartón Original 32 Oz", 289, "Bebida de almendras original, 60 calorías por vaso"),
    ("Leche de almendras", "Blue Diamond Almond Breeze", "Cartón Chocolate sin azúcar 32 Oz", 289, "Bebida de almendras chocolate, sin azúcar"),
    ("Leche de almendras", "Blue Diamond Almond Breeze", "Cartón Vainilla sin azúcar 32 Oz", 289, "Bebida de almendras vainilla, sin azúcar"),
    # ── Leche de coco ──
    ("Leche de coco", "La Famosa", "Lata 10.5 Oz", 99, "Leche de coco para cocinar"),
    ("Leche de coco", "La Famosa", "Lata 15 Oz", 125, "Leche de coco para cocinar"),
    ("Leche de coco", "Wala", "Lata 13.5 Oz", 78, "Leche de coco, 0% grasas trans"),
    ("Leche de coco", "Goya", "Lata 13.5 Oz", 125, "Leche de coco (coconut milk)"),
    ("Leche de coco", "Nestlé Carnation", "Cartón UHT 290 Ml", 115, "Leche evaporada sabor coco"),
    ("Leche de coco", "Blue Diamond Almond Breeze", "Cartón Coco Original sin azúcar 32 Oz", 289, "Bebida de almendras y coco, sin azúcar"),
    ("Leche de coco", "Blue Diamond Almond Breeze", "Cartón Coco Vainilla sin azúcar 32 Oz", 289, "Bebida de almendras y coco sabor vainilla, sin azúcar"),
    # ── Leche de soya ──
    ("Leche de soya", "Silk Soy", "Cartón Natural sin azúcar 32 Oz", 230, "Bebida de soya orgánica natural, 8g proteína"),
    ("Leche de soya", "Nutra", "Sobre 400 gr (polvo)", 339, "Bebida de soya en polvo, fórmula original"),
    ("Leche de soya", "Nutra", "Funda 1800 gr (polvo)", 1425, "Bebida de soya en polvo, fórmula original"),
    ("Leche de soya", "Nutra", "Funda Sin Lactosa 1800 gr (polvo)", 1620, "Bebida de soya en polvo sin lactosa"),
    # ── Crema de leche ──
    ("Crema de leche", "Wala", "Cartón UHT 200 Ml", 64, "Crema de leche UHT para postres y salsas"),
    ("Crema de leche", "Elle & Vire", "Cartón Professionnel 1 Lt", 350, "Crema de cocina profesional (Francia)"),
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
        key = (food.lower(), brand.lower(), pres.lower())
        if key in seen:
            print(f"FATAL: fila duplicada en el seed: {food} / {brand} / {pres}")
            sys.exit(1)
        seen.add(key)
        if food not in FOOD_CATEGORY:
            print(f"FATAL: food_name sin categoría: {food}")
            sys.exit(1)

    foods = {r[0] for r in ROWS}
    brands = {r[1] for r in ROWS}
    print(f"Seed leches: {len(ROWS)} SKUs · {len(foods)} tipos de leche · {len(brands)} marcas.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, brand, pres, price, desc) in ROWS:
                cur.execute(_INSERT, (food, brand, pres, price, NOTES,
                                      FOOD_CATEGORY[food], food, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
