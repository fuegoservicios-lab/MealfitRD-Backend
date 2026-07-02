"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de FRIJOLES PINTOS +
SEMILLAS DE CAJUIL + SEMILLAS DE GIRASOL.

Familias 89-91 del Supermercado RD (capturas del owner, 2026-07-02).
GUANDULES SECOS: 0 filas — los 2 SKUs Goya (Lata Secos 15 Oz RD$113 y Lata
Secos con Coco 15 Oz RD$125) YA estaban cargados bajo "Guandules" del seed
quinoa/guandules. FLAG al owner: el genérico "Guisantes secos · Lata 15 Oz ·
RD$125" calza exacto con el Goya Guandules Secos CON COCO — probable
mislabel del PDF (product-truth: gandules/pigeon peas ≠ guisantes/peas;
los chícharos reales Goya son fundas 16 Oz).

FRIJOLES PINTOS (5, La Sirena — los listings dicen "Habichuelas Pintas",
el food del catálogo es "Frijoles pintos"):
  * CALCE: La Sanjuanera funda 800 gr RD$127 = genérico "Paquete 800 gr".
  * Wala lata 15 Oz RD$50, Goya lata 15.5 Oz RD$88, La Famosa lata 15 Oz
    RD$88, Giselle funda "pinta americana" 800 gr RD$139.

SEMILLAS DE CAJUIL (11, La Sirena):
  * TRIPLE CALCE: Eloy's tarro 14.5 Oz RD$689 = genérico "Paquete 14.5 Oz",
    Dynasty naturales 7 Oz RD$385 = genérico "Tarro 7 Oz", Dynasty
    tostadas y sal 4 Oz RD$255 = genérico "Tarro 4 Oz".
  * Eloy's ×5 (tarro 14.5, doypack 5 Oz, con pasas 5 Oz, cajuil y
    cranberries 3 Oz, con jengibre 10 Oz — mixes quedan EN familia con
    descripción honesta, criterio granola-con-almendra), Dynasty ×3
    (naturales 7, tost y sal 4, y el tarro 4 Oz RD$285 sin variedad en el
    título — foto reciclada del store, FLAG), Wala funda 100 gr,
    Cashita's funda con sal 40 gr, Nut Walker lata lightly salted 135 gr.

SEMILLAS DE GIRASOL (5, **Supermercados Nacional** — notes por fila):
  * Imperial Nuts pote asadas 7.25 Oz $139.95, Eloy's tarro 8 Oz $169.95,
    BioEva funda 200 gr $84.95, Nut Walker funda con sal 45 gr $47.95,
    Crav'n Flavor pote tostadas con sal 7 Oz $169.95. El genérico
    "Paquete 400 gr" RD$145 no aparece en estas capturas (sin calce).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_pintos_cajuil_girasol_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_pintos_cajuil_girasol_2026_07_02.py --commit   # inserta
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
LEGUM = "Legumbres y proteína vegetal"
SEM = "Semillas y frutos secos"

PINTO = "Frijoles pintos"
CAJUIL = "Semillas de cajuil"
GIRASOL = "Semillas de girasol"

# (food_name, brand, presentation, price_rd, description, category, notes)
ROWS = [
    # ── Frijoles pintos ──
    (PINTO, "Wala", "Lata 15 Oz", 50, "Habichuelas pintas enlatadas (400 gr, 240 gr drenado)", LEGUM, SIRENA),
    (PINTO, "Giselle", "Funda 800 gr", 139, "Habichuelas pintas secas, pinta americana", LEGUM, SIRENA),
    (PINTO, "Goya", "Lata 15.5 Oz", 88, "Habichuelas pintas enlatadas (pinto beans) prime premium", LEGUM, SIRENA),
    (PINTO, "La Sanjuanera", "Funda 800 gr", 127, "Habichuelas pintas secas premium", LEGUM, SIRENA),
    (PINTO, "La Famosa", "Lata 15 Oz", 88, "Habichuelas pintas enlatadas (pinto beans)", LEGUM, SIRENA),
    # ── Semillas de cajuil · Eloy's ──
    (CAJUIL, "Eloy's", "Tarro 14.5 Oz", 689, "Semillas de cajuil tostadas", SEM, SIRENA),
    (CAJUIL, "Eloy's", "Funda 5 Oz", 159, "Semillas de cajuil, doypack", SEM, SIRENA),
    (CAJUIL, "Eloy's", "Tarro con Pasas 5 Oz", 219, "Cajuil con pasas (mix)", SEM, SIRENA),
    (CAJUIL, "Eloy's", "Funda Cajuil y Cranberries 3 Oz", 189, "Cajuil con cranberries (mix), doypack", SEM, SIRENA),
    (CAJUIL, "Eloy's", "Tarro con Jengibre 10 Oz", 459, "Cajuil con jengibre confitado", SEM, SIRENA),
    # ── Semillas de cajuil · Dynasty ──
    (CAJUIL, "Dynasty", "Tarro Naturales 7 Oz", 385, "Semillas de cajuil naturales", SEM, SIRENA),
    (CAJUIL, "Dynasty", "Tarro Tostadas y Sal 4 Oz", 255, "Semillas de cajuil tostadas con sal", SEM, SIRENA),
    (CAJUIL, "Dynasty", "Tarro 4 Oz", 285, "Semillas de cajuil (listing sin variedad en el título; foto reciclada del tarro Dynasty)", SEM, SIRENA),
    # ── Semillas de cajuil · resto ──
    (CAJUIL, "Wala", "Funda 100 gr", 155, "Semillas de cajuil", SEM, SIRENA),
    (CAJUIL, "Cashita's", "Funda con Sal 40 gr", 95, "Nuez de cajuil con sal", SEM, SIRENA),
    (CAJUIL, "Nut Walker", "Lata Lightly Salted 135 gr", 385, "Cashews ligeramente salados", SEM, SIRENA),
    # ── Semillas de girasol (Supermercados Nacional) ──
    (GIRASOL, "Imperial Nuts", "Pote Asadas 7.25 Oz", 139.95, "Semillas de girasol asadas (dry roasted kernels)", SEM, NACIONAL),
    (GIRASOL, "Eloy's", "Tarro 8 Oz", 169.95, "Semillas de girasol", SEM, NACIONAL),
    (GIRASOL, "BioEva", "Funda 200 gr", 84.95, "Semillas de girasol 100% natural (sunflower seeds)", SEM, NACIONAL),
    (GIRASOL, "Nut Walker", "Funda con Sal 45 gr", 47.95, "Semillas de girasol tostadas con sal", SEM, NACIONAL),
    (GIRASOL, "Crav'n Flavor", "Pote Tostadas con Sal 7 Oz", 169.95, "Semillas de girasol tostadas con sal (kernels)", SEM, NACIONAL),
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
    print(f"Seed pintos + cajuil + girasol: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
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
