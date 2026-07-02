"""[P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Seed de variantes de PAN INTEGRAL.

Decimoctava familia con variantes de MARCA del Supermercado RD: 26 SKUs transcritos
del catálogo de La Sirena (capturas del owner, 2026-07-02) — Lumijor, Holsum, Bolin,
Buen Horno (9 formatos: pulguitas/hogazas/moldes/baguettes/masa madre), Mi Trigo,
Bisco, La Libanesa, Toufayan, Zerca, Moulin D'Or y Wala.

Se agrupan bajo los DOS foods genéricos que ya existen del PDF:
  * "Pan integral familiar" (15): vigas/moldes/hogazas grandes (19-25 Oz / ~700-800 gr).
  * "Pan integral personal" (8): formatos individuales (viga personal, mini club,
    pulguitas, panecillo 80 gr, media baguette, francés, carioca, mini croissant).
  * "Pan pita integral" (3, food propio): Lumijor, La Libanesa, Toufayan — flatbread,
    producto distinto al pan de molde (mismo criterio que orégano fresco / codorniz).

Notas:
  * NINGÚN SKU calza exacto con los genéricos del PDF (familiar RD$155 / personal
    RD$140, "Paquete") — quedan como referencia genérica y las marcas como variantes.
  * "Integral blanca" (Holsum 24 Oz, Bolin Viga Mediana) = grano integral con miga
    blanca — descripción honesta.
  * Incluidos 4 "No disponible" (Zerca 745G, Moulin D'Or sin azúcar, Buen Horno masa
    madre, Wala 725G) como referencia de catálogo.
  * Sin exclusiones: los 26 listados eran pan.

Idempotente: ON CONFLICT (variante) DO NOTHING — no pisa ediciones de la admin UI.

USO:
  python scripts/seed_supermarket_pan_integral_2026_07_02.py            # DRY-RUN
  python scripts/seed_supermarket_pan_integral_2026_07_02.py --commit   # inserta
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
CATEGORY = "Panadería y harinas"

FAMILIAR = "Pan integral familiar"
PERSONAL = "Pan integral personal"
PITA = "Pan pita integral"

# (food_name, brand, presentation, price_rd, description)
ROWS = [
    # ── Pan integral familiar: vigas / moldes / hogazas ──
    (FAMILIAR, "Lumijor", "Viga Familiar Gourmet", 165, "Pan de viga integral tamaño familiar, línea Gourmet"),
    (FAMILIAR, "Lumijor", "Viga Macadamia 19 Oz", 269, "Pan de viga integral con macadamia"),
    (FAMILIAR, "Holsum", "Viga Integral Blanca 24 Oz", 205, "Pan de viga integral blanca (grano integral con miga blanca)"),
    (FAMILIAR, "Holsum", "Viga Pan Club Integral 24 Oz", 225, "Pan de viga integral Pan Club"),
    (FAMILIAR, "Bolin", "Viga 25 Oz", 165, "Pan de viga integral"),
    (FAMILIAR, "Bolin", "Viga Mediana Integral Blanco", 165, "Pan de viga integral blanco (grano integral con miga blanca), tamaño mediano"),
    (FAMILIAR, "Bisco", "Viga Avena y Miel", 145, "Pan de viga integral con avena y miel"),
    (FAMILIAR, "Buen Horno", "Molde con Nueces", 119, "Pan de molde integral con nueces"),
    (FAMILIAR, "Buen Horno", "Molde con Miel", 105, "Pan de molde integral con miel"),
    (FAMILIAR, "Buen Horno", "Hogaza con Cereal Premium", 130, "Hogaza integral con cereales, línea Premium"),
    (FAMILIAR, "Buen Horno", "Hogaza", 61, "Hogaza integral"),
    (FAMILIAR, "Buen Horno", "Masa Madre Multigranos", 129, "Pan de masa madre integral multigranos"),
    (FAMILIAR, "Zerca", "Viga Mediana 745 gr", 99, "Pan de viga integral, tamaño mediano"),
    (FAMILIAR, "Wala", "Viga Mediana 725 gr", 165, "Pan de viga integral, tamaño mediano"),
    (FAMILIAR, "Moulin D'Or", "Viga Sin Azúcar", 160, "Pan de viga integral sin azúcar"),
    # ── Pan integral personal: formatos individuales ──
    (PERSONAL, "Lumijor", "Viga Personal Gourmet", 147, "Pan de viga integral tamaño personal, línea Gourmet"),
    (PERSONAL, "Bolin", "Viga Mini Club", 124, "Pan de viga integral tamaño mini club"),
    (PERSONAL, "Buen Horno", "Pulguitas 10 unid", 37, "Panecillos pulguita integrales"),
    (PERSONAL, "Buen Horno", "Panecillo 80 gr (unidad)", 8, "Panecillo integral individual"),
    (PERSONAL, "Buen Horno", "Media Baguette", 22, "Media baguette integral"),
    (PERSONAL, "Buen Horno", "Francés (unidad)", 48, "Pan francés (baguette) integral"),
    (PERSONAL, "Mi Trigo", "Carioca (paquete)", 105, "Pan carioca integral (panecillos suaves)"),
    (PERSONAL, "Mi Trigo", "Mini Croissant (paquete)", 169, "Mini croissants integrales"),
    # ── Pan pita integral (food propio) ──
    (PITA, "Lumijor", "Pita 6 unid", 102, "Pan pita integral"),
    (PITA, "La Libanesa", "Pita (paquete)", 108, "Pan pita integral estilo libanés"),
    (PITA, "Toufayan", "Pita Whole Wheat 12 Oz", 119, "Pan pita integral (whole wheat)"),
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
    brands = {r[1] for r in ROWS}
    print(f"Seed pan integral: {len(ROWS)} SKUs · {len(foods)} foods · {len(brands)} marcas.")
    if not COMMIT:
        print("DRY-RUN (no escribe). Ejecuta con --commit para insertar.")
        return

    inserted = skipped = 0
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (food, brand, pres, price, desc) in ROWS:
                cur.execute(_INSERT, (food, brand, pres, price, NOTES, CATEGORY, food, desc))
                if cur.fetchone():
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    print(f"OK: {inserted} insertadas, {skipped} ya existían (no tocadas).")


if __name__ == "__main__":
    main()
