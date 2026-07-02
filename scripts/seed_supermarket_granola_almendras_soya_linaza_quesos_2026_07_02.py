"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de GRANOLA (+barras), ALMENDRAS, SALSA DE SOYA, VINAGRE BALSÁMICO, LINAZA y QUESOS (hoja/blanco).

Familias 86-92 del Supermercado RD: 51 SKUs transcritos del catálogo de La Sirena
(capturas del owner, 2026-07-02). OCHO calces exactos genérico↔marca en la tanda:

  * "Granola" (12): Zerca funda 1 Lb RD$95 = genérico "Paquete L". Wala ×2,
    BioEva ×2, Sante Fit/Gold ×3, Vida, Multifoods ×2, Sun Mix yogurt crunch.
  * "Barra de granola" (food NUEVO, 12): barras snack ≠ granola a granel —
    Nature Valley ×7 (crunchy/chewy/wafer/singles), Wala ×2, Sante Crispy
    Bar ×3 (promos -13%/-22% a precio de LISTA). EXCLUIDO: O'bar 6/1 198g
    listado a RD$0.00 (sin precio real — flag owner, patrón Yoplait Kiwi).
  * "Almendras fileteadas" (2): Eloy's bandeja 6 Oz RD$289 = genérico
    "Paquete 6 Oz" + Dynasty rebanadas 6 Oz.
  * "Almendras" (food NUEVO, 4): enteras/tostadas ≠ fileteadas — Dynasty
    enteras y tarro tostadas/saladas, Blue Diamond lata con sal 6 Oz y funda
    lightly salted 16 Oz. Las 2 LECHES de almendras de las capturas (Silk
    vainilla RD$260, Blue Diamond original RD$289) YA estaban cargadas del
    seed de leches con precios idénticos — no se duplican.
  * "Salsa de soya" (4): Kikkoman 10 Oz RD$215 = genérico "Frasco 10 Oz";
    Kikkoman baja en sal, La Choy normal y lite.
  * "Vinagre balsámico" (4): Borges 0.25 Lt RD$179 = genérico "Pote 0.25 Lt";
    Borges 0.5 Lt, Carbonell y el glaze Pietro Coricelli (reducción,
    descripción honesta, "No disponible").
  * "Linaza" (5): Multifoods dorada 8 Oz RD$70 = genérico "Paquete 8 Oz";
    marrón, molida, BioEva dorada 200 gr y granel Lb.
  * "Queso de hoja" (3): San Juan pieza RD$249 = genérico "Paquete 1L";
    La Zarina y Águila ("No disponible" ×2).
  * "Queso blanco" (5): Sosua paquete de freír 1 Lb RD$270 = genérico
    "Paquete 1L"; Sosua bandeja rebanada, San Juan pieza y barra Lb,
    Yokesso (Induveca).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_granola_almendras_soya_linaza_quesos_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_granola_almendras_soya_linaza_quesos_2026_07_02.py --commit   # inserta
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
GRANO = "Granos y cereales"
SEM = "Semillas y frutos secos"
SALSA = "Salsas y aderezos"
LACTEO = "Lácteos y huevos"

GRAN = "Granola"
BARRA = "Barra de granola"
ALM = "Almendras"
QHOJA = "Queso de hoja"
QBLANCO = "Queso blanco"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Granola ──
    (GRAN, "Zerca", "Funda 1 Lb", 95, "Granola natural", GRANO),
    (GRAN, "Wala", "Funda Mezcla Original 400 gr", 130, "Granola mezcla original con almendras y pasas", GRANO),
    (GRAN, "Wala", "Funda Mezcla Tropical 400 gr", 148, "Granola con frutas tropicales y nueces", GRANO),
    (GRAN, "BioEva", "Funda Natural 380 gr", 180, "Granola natural", GRANO),
    (GRAN, "BioEva", "Funda Con Almendra 380 gr", 210, "Granola con almendra", GRANO),
    (GRAN, "Sante", "Funda Fit Chocolate 300 gr", 249, "Granola fit sin azúcar añadida, nueces y cacao", GRANO),
    (GRAN, "Sante", "Funda Fit Fresa y Cereza 300 gr", 249, "Granola fit sin azúcar añadida (strawberry & cherry)", GRANO),
    (GRAN, "Sante", "Funda Gold Nueces y Miel 300 gr", 224, "Granola gold con nueces y miel", GRANO),
    (GRAN, "Vida", "Funda Natural 16 Oz", 159, "Granola natural artesanal", GRANO),
    (GRAN, "Multifoods", "Tarro Artesanal Home Style 800 gr", 409, "Granola fit artesanal estilo casero", GRANO),
    (GRAN, "Multifoods", "Clamshell Clásica Estilo Casero 7 Oz", 115, "Granola clásica estilo casero", GRANO),
    (GRAN, "Sun Mix", "Sobre Yogurt Crunch 57 gr", 95, "Granola con yogurt crunch, snack", GRANO),
    # ── Barra de granola (food nuevo) ──
    (BARRA, "Nature Valley", "Caja Crunchy Oats 'n Honey 8.94 Oz", 369, "12 barras crujientes de avena y miel", GRANO),
    (BARRA, "Nature Valley", "Caja Crunchy Variety Pack 8.9 Oz", 369, "12 barras variedad: oats 'n honey, peanut butter y dark chocolate", GRANO),
    (BARRA, "Nature Valley", "Caja Sweet & Salty Almendra 7.4 Oz", 369, "6 barras chewy de almendra con cobertura de mantequilla de almendra", GRANO),
    (BARRA, "Nature Valley", "Caja Chewy Trail Mix Fruit & Nut (6 unid)", 369, "6 barras chewy trail mix con fruta real", GRANO),
    (BARRA, "Nature Valley", "Caja Wafer Peanut Butter Chocolate", 485, "Barras wafer crispy creamy de mantequilla de maní y chocolate", GRANO),
    (BARRA, "Nature Valley", "Barra Crunchy Oats 'n Dark Chocolate (unidad)", 59, "Barra crujiente de avena y chocolate oscuro", GRANO),
    (BARRA, "Nature Valley", "Barra Trail Mix Fruit & Nut (unidad)", 59, "Barra trail mix de frutos secos y fruta", GRANO),
    (BARRA, "Wala", "Caja Crujiente Avena y Miel 222 gr", 195, "12 barras de granola crujientes de avena y miel", GRANO),
    (BARRA, "Wala", "Caja Chewy Almendra 186 gr", 230, "6 barras de granola chewy de almendra", GRANO),
    (BARRA, "Sante", "Crispy Bar Avena y Chocolate 40 gr", 45, "Barra crispy de avena y chocolate", GRANO),
    (BARRA, "Sante", "Crispy Bar Avena y Miel 40 gr", 40, "Barra crispy de avena y miel", GRANO),
    (BARRA, "Sante", "Crispy Bar Avena y Cranberry 40 gr", 40, "Barra crispy de avena y cranberry", GRANO),
    # ── Almendras fileteadas ──
    ("Almendras fileteadas", "Eloy's", "Bandeja Rebanadas 6 Oz", 289, "Almendras rebanadas (fileteadas)", SEM),
    ("Almendras fileteadas", "Dynasty", "Funda Natural Rebanadas 6 Oz", 335, "Almendras naturales rebanadas (sliced)", SEM),
    # ── Almendras (food nuevo — enteras/tostadas) ──
    (ALM, "Dynasty", "Funda Enteras Naturales 7 Oz", 355, "Almendras enteras naturales", SEM),
    (ALM, "Dynasty", "Tarro Tostadas/Saladas (unidad)", 370, "Almendras tostadas y saladas", SEM),
    (ALM, "Blue Diamond", "Lata Tostadas Con Sal 6 Oz", 379, "Almendras tostadas con sal (roasted salted)", SEM),
    (ALM, "Blue Diamond", "Funda Lightly Salted 16 Oz", 785, "Almendras ligeramente saladas, bajo sodio", SEM),
    # ── Salsa de soya ──
    ("Salsa de soya", "Kikkoman", "Botella 10 Oz", 215, "Salsa de soya elaborada tradicionalmente", SALSA),
    ("Salsa de soya", "Kikkoman", "Botella Baja en Sal 10 Oz", 265, "Salsa de soya con 38% menos sodio", SALSA),
    ("Salsa de soya", "La Choy", "Botella 10 Oz", 280, "Salsa de soya original", SALSA),
    ("Salsa de soya", "La Choy", "Botella Lite 10 Oz", 280, "Salsa de soya lite, menos sodio", SALSA),
    # ── Vinagre balsámico ──
    ("Vinagre balsámico", "Borges", "Botella 0.25 Lt", 179, "Vinagre balsámico de Módena", SALSA),
    ("Vinagre balsámico", "Borges", "Botella 0.5 Lt", 265, "Vinagre balsámico de Módena", SALSA),
    ("Vinagre balsámico", "Carbonell", "Botella 0.25 Lt", 185, "Vinagre balsámico", SALSA),
    ("Vinagre balsámico", "Pietro Coricelli", "Botella Glaze 250 Ml", 259, "Crema/reducción de aceto balsámico de Módena IGP (glaze)", SALSA),
    # ── Linaza ──
    ("Linaza", "Multifoods", "Funda Dorada 8 Oz", 70, "Semillas de linaza dorada", SEM),
    ("Linaza", "Multifoods", "Funda Marrón 8 Oz", 74, "Semillas de linaza marrón", SEM),
    ("Linaza", "Multifoods", "Funda Molida Marrón 8 Oz", 100, "Linaza marrón molida", SEM),
    ("Linaza", "BioEva", "Funda Dorada 200 gr", 75, "Semillas de linaza dorada, 100% natural", SEM),
    ("Linaza", None, "Granel Lb", 145, "Semillas de linaza a granel, por libra", SEM),
    # ── Queso de hoja ──
    (QHOJA, "San Juan", "Pieza (unidad)", 249, "Queso de hoja, bola empacada", LACTEO),
    (QHOJA, "La Zarina", "Paquete Lb", 260, "Queso de hoja artesanal, por libra", LACTEO),
    (QHOJA, "Águila", "Bola Lb", 240, "Queso de hoja, por libra", LACTEO),
    # ── Queso blanco (de freír) ──
    (QBLANCO, "Sosua", "Paquete de Freír 1 Lb", 270, "Queso blanco de freír, paquete 454 gr", LACTEO),
    (QBLANCO, "Sosua", "Bandeja de Freír Rebanado Lb", 265, "Queso blanco de freír rebanado, por libra", LACTEO),
    (QBLANCO, "San Juan", "Pieza de Freír (unidad)", 359, "Queso de freír, pieza empacada", LACTEO),
    (QBLANCO, "San Juan", "Barra de Freír Lb", 345, "Queso de freír, por libra", LACTEO),
    (QBLANCO, "Yokesso", "Barra de Freír Lb", 270, "Queso blanco de freír Induveca, por libra", LACTEO),
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
    print(f"Seed granola + almendras + soya + balsámico + linaza + quesos: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
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
