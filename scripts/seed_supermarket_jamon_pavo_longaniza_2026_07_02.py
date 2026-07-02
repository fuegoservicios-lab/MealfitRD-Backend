"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de JAMÓN DE PAVO, LONGANIZA y leche descremada delactosada.

Familias 49-50 del Supermercado RD: 19 SKUs transcritos del catálogo de La Sirena
(capturas del owner, 2026-07-02):

  * "Jamón de pavo" (12): Caserío/Induveca (3), Butterball (2), Chef (2), Checo,
    Kretschmar oven roasted (su Honey ya estaba cargado del seed de cerdo),
    Sosua Balance (CALZA EXACTO con el genérico RD$255/Lb), Noel Pavosi y
    Dietz & Watson Black Forest ("No disponible" ×2).
  * "Longaniza" (6): Sosua casera, especial gruesa (CALZA EXACTO con el genérico
    RD$136 — la especial fina ya estaba del seed de cerdo), Del Artesano,
    Hugo Pork lomo ahumado, Chef tipo Villa Mella y Pollo Cibao de pollo
    (misma familia, descripción honesta).
  * "Leche descremada" (+1): Dos Pinos Delactosada 1 Lt — las otras 5 descremadas
    de las capturas YA estaban cargadas del seed de leches (precios idénticos),
    igual que la Milex Slim en polvo (RD$1,585).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_jamon_pavo_longaniza_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_jamon_pavo_longaniza_2026_07_02.py --commit   # inserta
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
LACTEO = "Lácteos y huevos"

PAVO = "Jamón de pavo"
LONGANIZA = "Longaniza"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Jamón de pavo ──
    (PAVO, "Caserío", "Pieza Pechuga de Pavo Lb (rebanado)", 415, "Jamón de pechuga de pavo, rebanado al momento, por libra", CARNE),
    (PAVO, "Caserío", "Paquete Pechuga de Pavo Rebanado 0.5 Lb", 275, "Jamón de pechuga de pavo rebanado", CARNE),
    (PAVO, "Caserío", "Pieza Jamón de Pavo Lb", 319, "Jamón de pavo Induveca Caserío, por libra", CARNE),
    (PAVO, "Butterball", "Paquete White Turkey Oven Roasted 16 Oz", 409, "Jamón de pavo blanco horneado (oven roasted, lean)", CARNE),
    (PAVO, "Butterball", "Pechuga de Pavo Horneada Lb", 585, "Pechuga de pavo horneada, por libra", CARNE),
    (PAVO, "Chef", "Pieza Jamón Cocido de Pavo Lb", 259, "Jamón cocido de pavo, por libra", CARNE),
    (PAVO, "Chef", "Paquete Jamón Cocido de Pavo Rebanado 0.75 Lb", 195, "Jamón cocido de pavo rebanado", CARNE),
    (PAVO, "Checo", "Pieza Pechuga de Pavo Lb", 409, "Jamón de pechuga de pavo, por libra", CARNE),
    (PAVO, "Kretschmar", "Oven Roasted Turkey Breast Lb", 489, "Pechuga de pavo horneada (browned in oil), 98% libre de grasa, por libra", CARNE),
    (PAVO, "Sosua", "Pieza Balance Jamón de Pavo Lb", 255, "Jamón de pavo línea Balance, por libra", CARNE),
    (PAVO, "Noel", "Pieza Pavosi Pechuga de Pavo Lb", 400, "Jamón de pechuga de pavo Pavosi, por libra", CARNE),
    (PAVO, "Dietz & Watson", "Black Forest Smoked Turkey Breast Lb", 789, "Pechuga de pavo ahumada Black Forest, sin gluten, por libra", CARNE),
    # ── Longaniza ──
    (LONGANIZA, "Sosua", "Casera 1 Lb", 179, "Longaniza casera de cerdo", CARNE),
    (LONGANIZA, None, "Especial Gruesa Lb", 136, "Longaniza de cerdo especial gruesa, por libra", CARNE),
    (LONGANIZA, "Del Artesano", "Paquete Artesanal 1 Lb", 199, "Longaniza artesanal ahumada", CARNE),
    (LONGANIZA, "Hugo Pork", "Paquete Lomo Ahumado 16 Oz", 265, "Longaniza de lomo de cerdo ahumada artesanal", CARNE),
    (LONGANIZA, "Chef", "Paquete Tipo Villa Mella 0.90 Lb", 170, "Longaniza tipo Villa Mella", CARNE),
    (LONGANIZA, "Pollo Cibao", "Paquete de Pollo 1.75 Lb", 295, "Longaniza de pollo, 100% pollo", CARNE),
    # ── Leche descremada ──
    ("Leche descremada", "Dos Pinos", "Cartón Delactosada 1 Lt", 104, "Leche descremada deslactosada, 0% grasa, fácil digestión", LACTEO),
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
        key = (food.lower(), (brand or "").lower(), pres.lower())
        if key in seen:
            print(f"FATAL: fila duplicada en el seed: {food} / {brand} / {pres}")
            sys.exit(1)
        seen.add(key)

    foods = {r[0] for r in ROWS}
    brands = {r[1] for r in ROWS if r[1]}
    print(f"Seed jamón de pavo + longaniza + leche: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
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
