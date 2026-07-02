"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de MEJILLONES +
QUESO CHEDDAR + QUESO GOUDA.

Familias 122-124 del Supermercado RD (capturas del owner, 2026-07-02,
La Sirena).

MEJILLONES (7):
  * CALCE: Vima funda 1/2 concha 32 Oz RD$519 = genérico "Paquete 32 Oz".
  * Cabo de Peñas ×2 (escabeche 4-pack 316 gr "3+1 gratis", al natural
    111 gr), Vima carne 16 Oz (promo -15% a precio de LISTA RD$265),
    carne de mejillón granel sin marca (listing sin unidad), Panamei
    entero chileno 1 Lb, Albo escabeche 115 gr (no disp).

QUESO CHEDDAR (9):
  * CALCE: Sosua cheddar Lb RD$299 = genérico "Lb".
  * Sosua premium, Yokesso importado, Michel (rebanado y sin lactosa),
    Crystal Farms ×3 (mild/sharp chunk 8 Oz, fine shredded 8 Oz —
    empaque dice Crystal FarmS), Wala paquete rebanado (no disp).

QUESO GOUDA (16):
  * CALCE: Sosua gouda Lb RD$320 = genérico "Lb".
  * Frico ×5 (holandés rueda y barra, porción 295 gr, ahumado en lonjas,
    light), Michel ×2 (regular, light), Sosua ×2 (rueda, mini 2.5 Lb),
    San Juan ×2 (barra Lb, rueda tipo holandés unidad), Muhlenhof ×3
    (ruedas pesto rojo/verde y jalapeño), Babybel malla mini 5 unid,
    Wala barra 6.6 Lb RD$245 (FLAG: precio probablemente por libra del
    bloque deli — RD$245 el bloque entero sería RD$37/lb, imposible).
  * EXCLUIDO el "Rollito Chorizo Y Queso Gouda Noel 66G" — snack
    preparado de chorizo con queso, no es queso gouda (criterio salami
    chips).

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_mejillones_cheddar_gouda_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_mejillones_cheddar_gouda_2026_07_02.py --commit   # inserta
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
PESC = "Carnes, pescados y mariscos"
LACTEOS = "Lácteos y huevos"

MEJ = "Mejillones"
CHED = "Queso cheddar"
GOUDA = "Queso gouda"

# (food_name, brand, presentation, price_rd, description, category)
ROWS = [
    # ── Mejillones ──
    (MEJ, "Vima", "Funda 1/2 Concha 32 Oz", 519, "Mejillones en media concha (half shell)", PESC),
    (MEJ, "Vima", "Funda Carne de Mejillón 16 Oz", 265, "Carne de mejillón (precio de lista, promo -15%)", PESC),
    (MEJ, "Cabo de Peñas", "Lata Escabeche 4-Pack 316 gr", 415, "Mejillones en escabeche, pack 3 latas + 1 gratis", PESC),
    (MEJ, "Cabo de Peñas", "Lata al Natural 111 gr", 200, "Mejillones al natural 13/18 piezas", PESC),
    (MEJ, "Panamei", "Funda Entero 1 Lb", 299, "Mejillón chileno entero", PESC),
    (MEJ, "Albo", "Lata en Escabeche 115 gr", 475, "Mejillones en escabeche", PESC),
    (MEJ, None, "Carne de Mejillón (granel)", 235, "Carne de mejillón cocida (el listing no indica unidad)", PESC),
    # ── Queso cheddar ──
    (CHED, "Sosua", "Lb", 299, "Queso cheddar, venta por libra", LACTEOS),
    (CHED, "Sosua", "Premium Lb", 325, "Queso cheddar premium, venta por libra", LACTEOS),
    (CHED, "Yokesso", "Importado Lb", 290, "Queso cheddar importado, venta por libra", LACTEOS),
    (CHED, "Michel", "Rebanado Lb", 340, "Queso cheddar rebanado, venta por libra", LACTEOS),
    (CHED, "Michel", "Sin Lactosa Lb", 425, "Queso cheddar sin lactosa, venta por libra", LACTEOS),
    (CHED, "Crystal Farms", "Barra Mild Chunk 8 Oz", 275, "Queso cheddar suave (mild) en barra", LACTEOS),
    (CHED, "Crystal Farms", "Barra Sharp Chunk 8 Oz", 275, "Queso cheddar añejo (sharp) en barra", LACTEOS),
    (CHED, "Crystal Farms", "Funda Fine Shredded 8 Oz", 385, "Queso cheddar rallado fino", LACTEOS),
    (CHED, "Wala", "Paquete Rebanado", 270, "Queso cheddar rebanado en paquete", LACTEOS),
    # ── Queso gouda ──
    (GOUDA, "Sosua", "Rueda Lb", 320, "Queso gouda, venta por libra", LACTEOS),
    (GOUDA, "Sosua", "Rueda Mini 2.5 Lb", 305, "Queso gouda mini, rueda de 2.5 Lb", LACTEOS),
    (GOUDA, "Frico", "Rueda Holandés Lb", 325, "Queso gouda holandés, venta por libra", LACTEOS),
    (GOUDA, "Frico", "Barra Holandés Lb", 305, "Queso gouda holandés en barra, venta por libra", LACTEOS),
    (GOUDA, "Frico", "Porción 295 gr", 245, "Queso gouda holandés en porción", LACTEOS),
    (GOUDA, "Frico", "Ahumado en Lonjas Lb", 350, "Queso gouda ahumado en lonjas", LACTEOS),
    (GOUDA, "Frico", "Rueda Light Holandés Lb", 450, "Queso gouda light holandés, venta por libra", LACTEOS),
    (GOUDA, "Michel", "Rueda Lb", 365, "Queso gouda, venta por libra", LACTEOS),
    (GOUDA, "Michel", "Rueda Light Lb", 389, "Queso gouda light, venta por libra", LACTEOS),
    (GOUDA, "San Juan", "Barra Lb", 400, "Queso gouda en barra, venta por libra", LACTEOS),
    (GOUDA, "San Juan", "Rueda Tipo Holandés Unidad", 405, "Queso gouda tipo holandés, rueda por unidad", LACTEOS),
    (GOUDA, "Muhlenhof", "Rueda Pesto Rojo Lb", 515, "Queso gouda con pesto rojo", LACTEOS),
    (GOUDA, "Muhlenhof", "Rueda Pesto Verde Lb", 515, "Queso gouda con pesto verde", LACTEOS),
    (GOUDA, "Muhlenhof", "Rueda Jalapeño Lb", 510, "Queso gouda con jalapeño", LACTEOS),
    (GOUDA, "Babybel", "Malla Mini 5 unid", 189, "Mini quesos gouda Babybel, malla de 5", LACTEOS),
    (GOUDA, "Wala", "Barra 6.6 Lb", 245, "Bloque deli de 6.6 Lb (precio probablemente por libra — verificar)", LACTEOS),
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
    print(f"Seed mejillones + cheddar + gouda: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
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
