"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de YAUTÍA, AUYAMA (+semillas) y BACALAO.

Familias 43-45 del Supermercado RD: 19 SKUs transcritos del catálogo de La Sirena
(capturas del owner, 2026-07-02):

  * "Yautía" (3): blanca/coco/amarilla por libra — las TRES calzan exacto con el
    genérico RD$78 (tipos sin marca, criterio berenjena).
  * "Auyama" (2): taiwanesa y criolla ("No disponible") — ambas calzan exacto con
    el genérico RD$29/Lb.
  * "Semillas de calabaza" (5): las semillas de auyama VAN AQUÍ (auyama =
    calabaza) — Multifoods (10 Oz/200 gr), BioEva sin cáscara, Eloy's (4 Oz +
    tarro 8 Oz que calza exacto con el genérico RD$305).
  * "Pan de semillas" (food NUEVO, 1): barra Buen Horno con semillas de calabaza
    260 gr — bonus del search, SKU legítimo sin food previo donde vivir.
  * "Bacalao" (8): TRIPLE calce exacto con genéricos (Noruego Lb RD$255, Wala
    12 Oz RD$170, Wala 16 Oz RD$225). Además: Noruego Grado A, Dimar lata,
    Cabo de Peñas a la vizcaína (lata preparada, descripción honesta), Zerca
    8 Oz y miga a granel (precio de LISTA RD$195, promo -15% RD$165).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_yautia_auyama_bacalao_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_yautia_auyama_bacalao_2026_07_02.py --commit   # inserta
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
VIV = "Víveres y tubérculos"
VEG = "Vegetales y verduras"
SEM = "Semillas y frutos secos"
PAN = "Panadería y harinas"
CARNE = "Carnes, pescados y mariscos"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Yautía ──
    ("Yautía", None, "Blanca Lb", 78, "Yautía blanca fresca, por libra", VIV),
    ("Yautía", None, "Coco Lb", 78, "Yautía coco fresca, por libra", VIV),
    ("Yautía", None, "Amarilla Lb", 78, "Yautía amarilla fresca, por libra", VIV),
    # ── Auyama ──
    ("Auyama", None, "Taiwanesa Lb", 29, "Auyama taiwanesa fresca, por libra", VEG),
    ("Auyama", None, "Criolla Lb", 29, "Auyama criolla fresca, por libra", VEG),
    # ── Semillas de calabaza (= semillas de auyama) ──
    ("Semillas de calabaza", "Multifoods", "Bandeja 10 Oz", 249, "Semillas de auyama (calabaza)", SEM),
    ("Semillas de calabaza", "Multifoods", "Funda 200 gr", 235, "Semillas de auyama (calabaza)", SEM),
    ("Semillas de calabaza", "BioEva", "Funda Sin Cáscara 200 gr", 165, "Semillas de auyama sin cáscara (pumpkin seeds), 100% natural", SEM),
    ("Semillas de calabaza", "Eloy's", "Funda 4 Oz", 160, "Semillas de auyama", SEM),
    ("Semillas de calabaza", "Eloy's", "Tarro 8 Oz", 305, "Semillas de auyama, tarro", SEM),
    # ── Pan de semillas (food nuevo, bonus del search) ──
    ("Pan de semillas", "Buen Horno", "Barra Semillas de Calabaza 260 gr", 89, "Barra de pan con semillas de calabaza (auyama)", PAN),
    # ── Bacalao ──
    ("Bacalao", None, "Noruego Lb", 255, "Bacalao noruego salado, por libra", CARNE),
    ("Bacalao", None, "Noruego Grado A Lb", 595, "Bacalao noruego grado A (lomo premium), por libra", CARNE),
    ("Bacalao", None, "Miga Lb", 195, "Miga de bacalao desmenuzada, por libra", CARNE),
    ("Bacalao", "Dimar", "Lata Listo para Cocinar 425 gr", 165, "Bacalao desmenuzado listo para cocinar", CARNE),
    ("Bacalao", "Cabo de Peñas", "Lata a la Vizcaína 111 gr", 339, "Bacalao a la vizcaína, receta tradicional con aceite de oliva (plato preparado)", CARNE),
    ("Bacalao", "Zerca", "Funda Filete 8 Oz", 113, "Filete de bacalao", CARNE),
    ("Bacalao", "Wala", "Funda Filetes 12 Oz", 170, "Filetes de bacalao", CARNE),
    ("Bacalao", "Wala", "Funda Filetes 16 Oz", 225, "Filetes de bacalao", CARNE),
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
    print(f"Seed yautía + auyama + bacalao: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
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
