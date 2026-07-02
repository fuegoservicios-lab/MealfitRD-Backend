"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de CHULETA/COSTILLAS
(complemento familia Cerdo) + CALAMAR.

Familias 120-121 del Supermercado RD (capturas del owner, 2026-07-02,
La Sirena — carnicería: fotos stock, el TÍTULO manda).

CHULETA Y COSTILLAS (9 nuevas bajo el food "Cerdo", donde ya viven las
variantes de carnicería): de las 18 capturas, 9 YA estaban cargadas
(chuleta ahumada RD$127 / fresca RD$139 / importada RD$115 / prime
francés RD$155 / T-Bone RD$150, costilla ahumada RD$175 / fresca RD$189
— CALCE del genérico "Chuleta costillas Lb" RD$189 = costilla fresca —,
Curly's baby back 680 gr RD$995 y Mister Foods chuleta ahumada premium
RD$128). Entran: chuleta ahumada premium corte de centro RD$148, chuleta
fresca premium RD$145, costilla con falda RD$195, costilla St. Louis
fresca RD$195 e importada RD$219, costilla de centro RD$199, trocitos de
costilla RD$160 y de chuleta RD$137 ahumados, y la masita de costillas
ahumadas Wala 1 Lb RD$185.

CALAMAR (9): el genérico "2 Lb" RD$460 (fresco) queda como está. Entran
conservas y congelados: Goya al ajillo 4 Oz, Cabo de Peñas ×4 (rodajas
en aceite de girasol — tacos de potón —, a la marinera, trozos en tinta
115 gr y 3-pack 80 gr), Albo trozos en tinta 112 gr, Panamei tentáculos
y aros 16 Oz, Vima tubo 16 Oz.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_chuleta_costilla_calamar_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_chuleta_costilla_calamar_2026_07_02.py --commit   # inserta
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
CARNES = "Carnes, pescados y mariscos"

CERDO = "Cerdo"
CAL = "Calamar"

# (food_name, brand, presentation, price_rd, description)
ROWS = [
    # ── Cerdo · chuletas y costillas nuevas ──
    (CERDO, None, "Chuleta Ahumada Premium Corte de Centro Lb", 148, "Chuleta de cerdo ahumada premium, corte de centro"),
    (CERDO, None, "Chuleta Fresca Premium Lb", 145, "Chuleta de cerdo fresca premium"),
    (CERDO, None, "Costilla con Falda Fresca Lb", 195, "Costilla de cerdo con falda, fresca"),
    (CERDO, None, "Costilla St. Louis Fresca Lb", 195, "Costilla de cerdo corte St. Louis, fresca"),
    (CERDO, None, "Costilla St. Louis Importada Lb", 219, "Costilla de cerdo corte St. Louis, importada"),
    (CERDO, None, "Costilla de Centro Fresca Lb", 199, "Costilla de cerdo de centro, fresca"),
    (CERDO, None, "Trocitos de Costilla Ahumada Lb", 160, "Trocitos de costilla de cerdo ahumada"),
    (CERDO, None, "Trocitos de Chuleta Ahumada Lb", 137, "Trocitos de chuleta de cerdo ahumada"),
    (CERDO, "Wala", "Funda Masita de Costillas Ahumadas 1 Lb", 185, "Masita de costillas de cerdo ahumadas"),
    # ── Calamar ──
    (CAL, "Goya", "Lata al Ajillo 4 Oz", 179, "Calamares jumbo al ajillo (jumbo squid in garlic sauce)"),
    (CAL, "Cabo de Peñas", "Lata Rodajas en Aceite de Girasol 111 gr", 220, "Tacos de potón en aceite de girasol"),
    (CAL, "Cabo de Peñas", "Lata Rodajas a la Marinera 111 gr", 270, "Rodajas de calamar en salsa marinera"),
    (CAL, "Cabo de Peñas", "Lata Trozos en Tinta 115 gr", 235, "Calamares en trozos en su tinta"),
    (CAL, "Cabo de Peñas", "Lata Trozos en Tinta 3-Pack 80 gr", 250, "Calamares en trozos en su tinta, pack de 3 latas de 80 gr"),
    (CAL, "Albo", "Lata Trozos en Tinta 112 gr", 260, "Calamares/pota en trozos en su tinta"),
    (CAL, "Panamei", "Funda Tentáculos 16 Oz", 580, "Tentáculos de calamar cocidos estilo pulpo (octopus style)"),
    (CAL, "Panamei", "Funda Aros 16 Oz", 645, "Aros de calamar (squid rings)"),
    (CAL, "Vima", "Funda Tubo 16 Oz", 375, "Tubo de calamar (squid tube)"),
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
    print(f"Seed chuleta/costillas + calamar: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, brand, pres, price, desc) in ROWS:
                cur.execute(_INSERT, (food, brand, pres, price, NOTES, CARNES, food, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
