"""[P1-SUPERMARKET-DB · 2026-07-02 · curación] Cierre de los flags de datos
acumulados durante la carga familia-por-familia (autorizado por el owner:
"implementalos... para llevarlo al 100%").

8 fixes, todos idempotentes (WHERE ancla el valor actual — re-ejecutar no
re-aplica ni pisa ediciones posteriores de la admin UI):

  1. Genérico "Guisantes secos · Lata 15 Oz · RD$125" → INACTIVO. Product-truth:
     es el Goya GUANDULES Secos con Coco 15 Oz (calce exacto de precio); los
     guisantes secos reales son las fundas de chícharos Goya. La familia
     Guandules ya tiene ese SKU — el genérico era doble-entrada mal etiquetada.
  2. Genérico "Ciruela · Paquete 16 Oz · RD$199" (fruta fresca) → INACTIVO.
     Precio y tamaño idénticos al clamshell de ciruelas PASAS (genérico
     "Tarro 16 Oz") — doble-entrada del PDF desde el mismo listing.
  3-4. Las 2 filas Lifeway bajo "Kefir" (sin acento) → food_name "Kéfir"
     (unifica la familia con los genéricos y las Okava).
  5-6. Leche de coco La Famosa cargadas a precio promo → precio de LISTA
     (regla del catálogo): 10.5 Oz RD$99→108, 15 Oz RD$125→138.
  7. Leche evaporada Carnation sabor queso 135 Ml RD$59 (promo) → lista RD$65.
  8. Repollo "Unidad (mitad)" RD$56 → RD$104 (precio vigente del listing).
  +. Queso gouda Wala "Barra 6.6 Lb" → presentación "Barra 6.6 Lb (precio
     por Lb)": RD$245 el bloque entero daría RD$37/lb (imposible) — es el
     precio por libra del bloque deli.

USO:
  python scripts/fix_supermarket_data_flags_2026_07_02.py            # DRY-RUN
  python scripts/fix_supermarket_data_flags_2026_07_02.py --commit   # aplica
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

# (descripción, SQL, params) — cada WHERE ancla el valor actual (idempotente).
FIXES = [
    ("1. Guisantes secos 'Lata 15 Oz' → inactivo (es Goya guandules con coco)",
     """UPDATE public.supermarket_products
        SET active = false,
            notes = 'Desactivado 2026-07-02: el PDF tomó el listing de Goya Guandules Secos con Coco 15 Oz (calce exacto RD$125) — ver familia Guandules. Los guisantes secos reales son las fundas de chícharos Goya.'
        WHERE food_name = 'Guisantes secos' AND brand IS NULL
          AND presentation = 'Lata 15 Oz' AND active = true""", ()),
    ("2. Ciruela fresca 'Paquete 16 Oz' → inactivo (doble-entrada de las pasas)",
     """UPDATE public.supermarket_products
        SET active = false,
            notes = 'Desactivado 2026-07-02: doble-entrada del PDF — mismo listing que Ciruela pasa (Tarro 16 Oz RD$199).'
        WHERE food_name = 'Ciruela' AND brand IS NULL
          AND presentation = 'Paquete 16 Oz' AND active = true""", ()),
    ("3. Kefir→Kéfir (Lifeway Low Fat Fresa 8 Oz)",
     """UPDATE public.supermarket_products
        SET food_name = 'Kéfir', master_food_name = 'Kéfir'
        WHERE food_name = 'Kefir' AND brand = 'Lifeway'
          AND presentation = 'Botella Low Fat Fresa 8 Oz'""", ()),
    ("4. Kefir→Kéfir (Lifeway Low Fat Vainilla 32 Oz)",
     """UPDATE public.supermarket_products
        SET food_name = 'Kéfir', master_food_name = 'Kéfir'
        WHERE food_name = 'Kefir' AND brand = 'Lifeway'
          AND presentation = 'Botella Low Fat Vainilla 32 Oz'""", ()),
    ("5. Leche de coco La Famosa 10.5 Oz: promo 99 → lista 108",
     """UPDATE public.supermarket_products
        SET price_rd = 108
        WHERE food_name = 'Leche de coco' AND brand = 'La Famosa'
          AND presentation = 'Lata 10.5 Oz' AND price_rd = 99""", ()),
    ("6. Leche de coco La Famosa 15 Oz: promo 125 → lista 138",
     """UPDATE public.supermarket_products
        SET price_rd = 138
        WHERE food_name = 'Leche de coco' AND brand = 'La Famosa'
          AND presentation = 'Lata 15 Oz' AND price_rd = 125""", ()),
    ("7. Carnation sabor queso 135 Ml: promo 59 → lista 65",
     """UPDATE public.supermarket_products
        SET price_rd = 65
        WHERE food_name = 'Leche evaporada' AND brand = 'Nestlé Carnation'
          AND presentation = 'Cartón UHT sabor queso 135 Ml' AND price_rd = 59""", ()),
    ("8. Repollo 'Unidad (mitad)': 56 → 104 (precio vigente)",
     """UPDATE public.supermarket_products
        SET price_rd = 104
        WHERE food_name = 'Repollo' AND brand IS NULL
          AND presentation = 'Unidad (mitad)' AND price_rd = 56""", ()),
    ("9. Gouda Wala: presentación aclara precio por libra",
     """UPDATE public.supermarket_products
        SET presentation = 'Barra 6.6 Lb (precio por Lb)',
            description = 'Bloque deli de 6.6 Lb; el precio mostrado es por libra'
        WHERE food_name = 'Queso gouda' AND brand = 'Wala'
          AND presentation = 'Barra 6.6 Lb'""", ()),
]


def main():
    if not _NEON:
        print("FATAL: NEON_DATABASE_URL no está definido (.env)")
        sys.exit(1)

    print(f"Curación de flags: {len(FIXES)} fixes. Modo: {'COMMIT' if COMMIT else 'DRY-RUN'}")
    with psycopg.connect(_NEON) as conn:
        with conn.cursor() as cur:
            for (label, sql, params) in FIXES:
                cur.execute(sql, params)
                estado = "aplicado" if cur.rowcount else "sin efecto (ya aplicado / valor cambió)"
                print(f"  [{cur.rowcount}] {label} → {estado}")
        if COMMIT:
            conn.commit()
            print("COMMIT hecho.")
        else:
            conn.rollback()
            print("ROLLBACK (dry-run). Ejecuta con --commit para aplicar.")


if __name__ == "__main__":
    main()
