"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de CARNE DE RES MOLIDA y CERDO.

Familias 46-47 del Supermercado RD: 56 SKUs transcritos del catálogo de La Sirena
(capturas del owner, 2026-07-02):

  * "Carne de Res molida" (5): ratios magro/grasa como variantes (80/20 calza
    exacto con el genérico RD$209), cadera y bola 96/4, 85/15, y el tubo Mister
    Foods 75/25.
  * "Cerdo" (45): TODOS los cortes del mostrador como presentaciones del food
    único del PDF (criterio berenjena/tomate) — chuletas (importada RD$115 calza
    exacto con el genérico, fresca, ahumadas, T-bone, prime francés), lomos (5),
    costillas, filetes (incl. marinados Mister Foods), paleta/pierna/bola/boliche,
    molida de cerdo, chicharrón (incl. Hugo Pork), lacón (incl. Checo), cuero,
    rabo, paticas (incl. Cuatropata precocidas), Curly's (pulled pork + baby back
    ribs), Nutricosa costillita ahumada, salchicha BBQ del mostrador.
  * "Longaniza" (+1): especial fina RD$138.
  * "Jamón de cerdo" (food NUEVO): Kretschmar Ham Off the Bone.
  * "Jamón de pavo" (+1): Kretschmar Honey — La Sirena lo titula "jamón de cerdo"
    pero el empaque dice HONEY TURKEY BREAST → se registra por el empaque.
  * "Manteca de cerdo" (food NUEVO): Don Pedro Keto 16 Oz.
  * "Dumplings de cerdo" (food NUEVO): Otasty potsticker 2 Lb.
  * "Salchichas" (food NUEVO): Dietz & Watson beef & pork franks.

EXCLUIDOS (3): 2 sobres Pedigree (alimento para PERROS) + wrap de pierna grab&go
con precio RD$0.00 (dato inválido). DEDUPES (2): Costilla de Cerdo Fresca Lb
(listada a RD$189 disponible y RD$195 "No disponible" → se carga RD$189) y
Salchicha (Ahumada) BBQ Lb (2 listados, mismo precio → 1).

Promos → precio de LISTA: Carne #7 RD$149 (-10%), Patica importada RD$70 (-14%).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_res_molida_cerdo_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_res_molida_cerdo_2026_07_02.py --commit   # inserta
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
CARNE = "Carnes, pescados y mariscos"
GRASA = "Aceites y grasas"

MOLIDA = "Carne de Res molida"
CERDO = "Cerdo"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Carne de Res molida ──
    (MOLIDA, None, "80/20 Lb", 209, "Carne de res molida 80% magra / 20% grasa, por libra", CARNE),
    (MOLIDA, None, "85/15 Lb", 235, "Carne de res molida 85% magra / 15% grasa, por libra", CARNE),
    (MOLIDA, None, "Cadera 96/4 Lb", 309, "Carne molida de cadera de res 96% magra, por libra", CARNE),
    (MOLIDA, None, "Bola 96/4 Lb", 305, "Carne molida de bola de res 96% magra, por libra", CARNE),
    (MOLIDA, "Mister Foods", "Tubo 75/25 454 gr", 149, "Carne de res molida 75/25 en tubo", CARNE),
    # ── Cerdo: chuletas ──
    (CERDO, None, "Chuleta Importada Lb", 115, "Chuleta de cerdo importada, por libra", CARNE),
    (CERDO, None, "Chuleta Fresca Lb", 139, "Chuleta de cerdo fresca, por libra", CARNE),
    (CERDO, None, "Chuleta Ahumada Lb", 127, "Chuleta de cerdo ahumada, por libra", CARNE),
    (CERDO, None, "Chuleta T-Bone Lb", 150, "Chuleta de cerdo T-bone, por libra", CARNE),
    (CERDO, None, "Chuleta Prime Estilo Francés Lb", 155, "Chuleta de cerdo prime estilo francés, por libra", CARNE),
    (CERDO, "Mister Foods", "Chuleta Ahumada Premium Lb", 128, "Chuleta de cerdo ahumada, línea premium", CARNE),
    # ── Cerdo: lomos ──
    (CERDO, None, "Lomo Premium Fresco Lb", 180, "Lomo de cerdo premium fresco, por libra", CARNE),
    (CERDO, None, "Lomo Nube Fresco Lb", 179, "Lomo nube de cerdo fresco, por libra", CARNE),
    (CERDO, None, "Lomo Ahumado Lb", 185, "Lomo de cerdo ahumado, por libra", CARNE),
    (CERDO, None, "Lomo Importado Lb", 165, "Lomo de cerdo importado, por libra", CARNE),
    (CERDO, None, "Lomo Chicago Fresco Lb", 185, "Lomo Chicago de cerdo fresco, por libra", CARNE),
    # ── Cerdo: filetes ──
    (CERDO, None, "Filete Fresco Lb", 185, "Filete de cerdo fresco, por libra", CARNE),
    (CERDO, None, "Filete Importado Lb", 165, "Filete de cerdo importado, por libra", CARNE),
    (CERDO, "Mister Foods", "Filete al Romero y Mantequilla Lb", 230, "Filete de cerdo marinado al romero y mantequilla", CARNE),
    (CERDO, "Mister Foods", "Filete Marinado BBQ Lb", 230, "Filete de cerdo marinado BBQ", CARNE),
    # ── Cerdo: costillas ──
    (CERDO, None, "Costilla Fresca Lb", 189, "Costilla de cerdo fresca, por libra", CARNE),
    (CERDO, None, "Costilla Ahumada Lb", 175, "Costilla de cerdo ahumada, por libra", CARNE),
    (CERDO, None, "Costillita Para Guisar Lb", 195, "Costillita de cerdo para guisar, por libra", CARNE),
    (CERDO, "Nutricosa", "Funda Costillita Ahumada 16 Oz", 205, "Nido de costillita de cerdo ahumada", CARNE),
    (CERDO, "Curly's", "Costillas Ahumadas Baby Back 680 gr", 995, "Costillas baby back de cerdo ahumadas con salsa BBQ", CARNE),
    # ── Cerdo: piezas y cortes ──
    (CERDO, None, "En Cuadritos Lb", 180, "Cerdo en cuadritos, por libra", CARNE),
    (CERDO, None, "Carne #7 Importada Lb", 149, "Carne #7 de cerdo importada, por libra", CARNE),
    (CERDO, None, "Paleta Importada Lb", 119, "Paleta de cerdo importada, por libra", CARNE),
    (CERDO, None, "Paleta Fresca Lb", 119, "Paleta de cerdo fresca, por libra", CARNE),
    (CERDO, None, "Pierna Fresca Importada Lb", 135, "Pierna de cerdo fresca importada, por libra", CARNE),
    (CERDO, None, "Bola Fresca Lb", 179, "Bola de cerdo fresca, por libra", CARNE),
    (CERDO, None, "Boliche Fresco Lb", 179, "Boliche de cerdo fresco, por libra", CARNE),
    (CERDO, None, "Chef Prime Fresco Lb", 179, "Corte chef prime de cerdo fresco, por libra", CARNE),
    (CERDO, None, "Sirloin Sin Hueso Lb", 145, "Sirloin de cerdo sin hueso, por libra", CARNE),
    (CERDO, None, "Churrasco Lb", 235, "Churrasco de cerdo, por libra", CARNE),
    (CERDO, None, "Milanesa Fresca Lb", 175, "Milanesa de cerdo fresca, por libra", CARNE),
    (CERDO, None, "Escalopines Frescos Lb", 169, "Escalopines de cerdo frescos, por libra", CARNE),
    (CERDO, None, "Fajitas Frescas Lb", 179, "Fajitas de cerdo frescas, por libra", CARNE),
    (CERDO, None, "Molida Fresca Lb", 169, "Carne molida de cerdo fresca, por libra", CARNE),
    (CERDO, None, "Para Guisar Lb", 125, "Carne de cerdo para guisar, por libra", CARNE),
    # ── Cerdo: especialidades ──
    (CERDO, None, "Chicharrón Lb", 199, "Chicharrón de cerdo, por libra", CARNE),
    (CERDO, "Hugo Pork", "Chicharrón Lb", 579, "Chicharrón de cerdo artesanal", CARNE),
    (CERDO, None, "Lacón Ahumado Lb", 103, "Lacón de cerdo ahumado, por libra", CARNE),
    (CERDO, "Checo", "Lacón 1 Lb", 180, "Lacón de cerdo", CARNE),
    (CERDO, None, "Cuero Ahumado Lb", 89, "Cuero de cerdo ahumado, por libra", CARNE),
    (CERDO, None, "Rabo Lb", 133, "Rabo de cerdo, por libra", CARNE),
    (CERDO, None, "Patica Importada Lb", 70, "Patica de cerdo importada, por libra (ver mínimo)", CARNE),
    (CERDO, "Cuatropata", "Funda Paticas Precocidas 1.5 Lb", 320, "Paticas de cerdo precocidas", CARNE),
    (CERDO, "Curly's", "Pote Desmenuzado Ahumado BBQ 454 gr", 465, "Cerdo desmenuzado (pulled pork) ahumado con nogal americano y salsa BBQ", CARNE),
    (CERDO, None, "Salchicha Ahumada BBQ Lb", 109, "Salchicha de cerdo ahumada BBQ del mostrador, por libra", CARNE),
    # ── Longaniza ──
    ("Longaniza", None, "Especial Fina Lb", 138, "Longaniza de cerdo especial fina, por libra", CARNE),
    # ── Jamones ──
    ("Jamón de cerdo", "Kretschmar", "Ham Off the Bone Lb", 409, "Jamón de cerdo off the bone con jugos naturales, por libra", CARNE),
    ("Jamón de pavo", "Kretschmar", "Honey Turkey Breast Off the Bone Lb", 530, "Pechuga de pavo ahumada con miel (hickory smoked), 99% libre de grasa, por libra", CARNE),
    # ── Manteca de cerdo (food nuevo) ──
    ("Manteca de cerdo", "Don Pedro", "Pote Keto Original 16 Oz", 350, "Manteca de cerdo para cocinar, línea Keto", GRASA),
    # ── Dumplings de cerdo (food nuevo) ──
    ("Dumplings de cerdo", "Otasty", "Funda Potsticker Vegetal y Cerdo 2 Lb", 950, "Dumplings (potstickers) de vegetales y cerdo, congelados", CARNE),
    # ── Salchichas (food nuevo) ──
    ("Salchichas", "Dietz & Watson", "Paquete Beef & Pork Franks 14 Oz", 595, "Salchichas de carne de res y cerdo (uncured franks)", CARNE),
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
    print(f"Seed res molida + cerdo: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
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
