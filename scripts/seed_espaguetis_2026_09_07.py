# -*- coding: utf-8 -*-
"""Las marcas y precios de espagueti del Supermercado Nacional (capturas del dueño, 2026-09-07).

TRES DECISIONES, y las tres importan:

1. `master_food_name` SOLO en el espagueti de TRIGO normal. Los «sin gluten» quedan con
   `master_food_name` NULO a propósito: mapearlos a `Espaguetis` haría que el motor le sirviera
   trigo a un celíaco creyendo que respeta su restricción. Existir en el catálogo del súper sin
   resolver a un alimento es correcto; resolver al alimento equivocado es un peligro.
   Los INTEGRALES (Barilla Integral, Rummo Bio Integrale) sí resuelven, pero a `Pasta integral`,
   que es la fila que les corresponde.

2. El precio ancla de `Espaguetis` baja de RD$98,88/lb a RD$41,90. La primera cifra salió de
   Pasta Zara (marca italiana); las capturas muestran que el espagueti de volumen en RD es
   Líder a RD$30/400 g y Milano a RD$36,95/400 g. Se toma MILANO como ancla y no Líder porque
   Milano tiene DOS tamaños reales (400 g y 800 g) y eso es lo que hace funcionar la selección
   por duración: `_select_market_package` elige el envase que mejor cubre los gramos que hacen
   falta, así que un plan de 30 días compra el de 800 g en vez de dos de 400 g.

3. NO se tocan `Coditos` ni `Fideos`. Que a un dominicano no le salgan tiene arreglo, pero no es
   ponerles un número: hay que SACARLOS del conjunto «alta de país beta, sin precio», y eso vive
   en TRES sitios a la vez (el precio, la tupla `_COUNTRY_CATALOG_UNPRICED_TOKENS`, y las listas
   de `test_p1_country_system_f2`). Tocar solo el primero es lo que puso tres guards en rojo.

    python seed_espaguetis.py            # simula
    python seed_espaguetis.py --aplicar  # escribe
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
_B = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _B)
os.chdir(_B)
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(_B, ".env"))
import psycopg  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402
from psycopg.types.json import Jsonb  # noqa: E402

APLICAR = "--aplicar" in sys.argv
LB = 453.592
OZ = 28.3495

# (marca, presentación, gramos, precio RD$, master_food_name | None, agotado)
#   master None  -> sin gluten / especialidad: NO resuelve a un alimento (ver decisión 1)
PRODUCTOS = [
    # --- trigo normal ---------------------------------------------------------------
    ("Líder",       "Paquete 400 gr",                  400.0,  30.00, "Espaguetis", False),
    ("Princesa",    "Paquete 400 gr",                  400.0,  34.95, "Espaguetis", False),
    ("Milano",      "Paquete 400 gr",                  400.0,  36.95, "Espaguetis", False),
    ("Milano",      "Paquete 800 gr",                  800.0,  76.95, "Espaguetis", False),
    ("Reggia",      "Paquete 350 gr",                  350.0,  78.95, "Espaguetis", False),
    ("Food Club",   "Paquete 16 Oz",             16 * OZ,     138.95, "Espaguetis", False),
    ("Barilla",     "Caja Spaghetti 500 gr",           500.0, 138.95, "Espaguetis", False),
    ("Barilla",     "Caja Spaghettoni 500 gr",         500.0, 138.95, "Espaguetis", False),
    ("Barilla",     "Caja Spaghettini 500 gr",         500.0, 138.95, "Espaguetis", False),
    ("De Cecco",    "Caja 500 gr",                     500.0, 166.95, "Espaguetis", False),
    ("Rummo",       "Paquete No.3 500 gr",             500.0, 199.95, "Espaguetis", False),
    ("Rummo",       "Paquete Alla Chitarra No.104 500 gr", 500.0, 199.95, "Espaguetis", False),
    ("De Cecco",    "Caja Spaghettini No.11 500 gr",   500.0, 189.95, "Espaguetis", True),
    # --- integrales: resuelven a `Pasta integral` -----------------------------------
    ("Barilla",     "Caja Spaghetti Integral 500 gr",  500.0, 197.95, "Pasta integral", False),
    ("Rummo",       "Paquete Bio Integrale No.3 500 gr", 500.0, 234.95, "Pasta integral", False),
    ("Rummo",       "Paquete Bio No.3 500 gr",         500.0, 234.95, "Pasta integral", False),
    # --- sin gluten y especialidades: master_food_name NULO -------------------------
    ("Reggia",      "Paquete Sin Gluten 400 gr",       400.0, 149.95, None, False),
    ("Milano",      "Paquete Sin Gluten 250 gr",       250.0, 139.95, None, False),
    ("De Cecco",    "Caja Sin Gluten 400 gr",          400.0, 269.95, None, False),
    ("Rummo",       "Paquete No.3 Sin Gluten 400 gr",  400.0, 269.95, None, False),
    ("Barilla",     "Caja Sin Gluten 14.5 Oz",   14.5 * OZ,   319.95, None, False),
    ("Barilla",     "Caja Con Proteína 14.5 Oz", 14.5 * OZ,   319.95, None, False),
    ("Full Circle", "Paquete Orgánica Sin Gluten 14 Oz", 14 * OZ, 169.95, None, False),
    ("Andean",      "Paquete De Quinoa Orgánica Sin Gluten", None, 499.95, None, False),
    ("Miracle",     "Paquete Sin Gluten Noodle 7 Oz", 7 * OZ,  479.95, None, False),
    ("Explore Pasta", "Paquete Edamame Sin Gluten 8 Oz", 8 * OZ, 429.95, None, True),
    ("Explore Pasta", "Paquete De Lenteja Roja Sin Gluten", None, 399.95, None, True),
    ("Jovial",      "Paquete De Arroz Integral Orgánico 120 gr", 120.0, 429.95, None, True),
]

# Ancla nueva de `Espaguetis`: Milano, la marca de volumen, con sus DOS tamaños reales.
PRECIO_LB_NUEVO = round(36.95 / 400.0 * LB, 2)
PAQUETES_NUEVOS = [
    {"unit": "paquete", "grams": 400, "label": "400 g", "price": 36.95},
    {"unit": "paquete", "grams": 800, "label": "800 g", "price": 76.95},
]


def main() -> int:
    nuevos = actualizados = saltados = 0
    with psycopg.connect(os.environ["NEON_DATABASE_URL"], row_factory=dict_row) as c:
        for marca, pres, gramos, precio, master, agotado in PRODUCTOS:
            g = round(gramos, 1) if gramos else None
            ya = c.execute(
                "SELECT id, price_rd FROM supermarket_products "
                "WHERE lower(food_name)='espaguetis' AND lower(coalesce(brand,''))=lower(%s) "
                "AND lower(coalesce(presentation,''))=lower(%s)", (marca, pres)).fetchone()
            if ya:
                if abs(float(ya["price_rd"] or 0) - precio) < 0.005:
                    saltados += 1
                    continue
                if APLICAR:
                    c.execute("UPDATE supermarket_products SET price_rd=%s, size_grams=%s, "
                              "active=%s, updated_at=now() WHERE id=%s",
                              (precio, g, not agotado, ya["id"]))
                actualizados += 1
                continue
            if APLICAR:
                c.execute(
                    """INSERT INTO supermarket_products
                       (food_name, brand, presentation, portion_label, price_rd, size_grams,
                        category, master_food_name, is_verified, active)
                       VALUES ('Espaguetis', %s, %s, %s, %s, %s, 'Granos y cereales', %s, true, %s)""",
                    (marca, pres, (f"{g:g} g" if g else None), precio, g, master, not agotado))
            nuevos += 1

        r = c.execute("SELECT price_per_lb FROM master_ingredients WHERE name='Espaguetis'").fetchone()
        cambio_ancla = r and abs(float(r["price_per_lb"] or 0) - PRECIO_LB_NUEVO) > 0.005
        if cambio_ancla and APLICAR:
            c.execute("""UPDATE master_ingredients SET
                           price_per_lb=%s, price_per_lb_base=%s,
                           price_per_unit=36.95, price_per_unit_base=36.95,
                           container_weight_g=400, available_sizes_g=%s, market_packages=%s,
                           price_captured_at='2026-09-07'
                         WHERE name='Espaguetis'""",
                      (PRECIO_LB_NUEVO, PRECIO_LB_NUEVO, Jsonb([400, 800]), Jsonb(PAQUETES_NUEVOS)))
        if APLICAR:
            c.commit()

    print(("APLICADO" if APLICAR else "SIMULACIÓN (usa --aplicar)") + ":")
    print(f"  productos nuevos: {nuevos} · precios actualizados: {actualizados} · ya iguales: {saltados}")
    if cambio_ancla:
        print(f"  ancla de `Espaguetis`: {r['price_per_lb']} → RD${PRECIO_LB_NUEVO}/lb "
              f"(Milano 400 g @ 36,95) · envases 400 g y 800 g")
    else:
        print("  ancla de `Espaguetis`: ya estaba en el valor nuevo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
