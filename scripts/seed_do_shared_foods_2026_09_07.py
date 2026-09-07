# -*- coding: utf-8 -*-
"""[P1-DO-SHARED-FOODS · 2026-09-07] Los siete básicos criollos que estaban archivados en otro país.

EL DEFECTO. La expansión de países dio de alta 141 filas SIN precio a propósito (los países beta
no tienen mercado RD que cotizar) y las repartió por país en
`_COUNTRY_CATALOG_UNPRICED_BY_COUNTRY`. La partición asumía que cada alimento pertenece a UN país.
Siete no: son de ese país **y también** de República Dominicana.

    coditos, tocineta, salchichas  -> archivados como comida de US
    pernil                          -> PR
    fideos                          -> ES
    chicharron                      -> MX
    gallina criolla                 -> CO

Y sin precio no existen para el generador dominicano. El gate del catálogo verificado es:

    if (price_per_lb or 0) > 0 or (price_per_unit or 0) > 0: return True
    return bool(_iccui and _iccui(name, country=_vc_country))   # _iccui es None si el país es DO

Con `country == "DO"` no hay segunda rama, así que a nadie en RD le salía un espagueti con
salchichas ni un sancocho con gallina criolla — aunque las siete filas existieran, con su
nutrición y su `fdc_id` correctos desde agosto.

> Un alimento que existe en el catálogo pero no tiene precio no está incompleto: está INVISIBLE.
> Y la taxonomía por país es una PARTICIÓN, que es justo la forma equivocada para lo compartido.

## Precios: etiquetas reales de Supermercado Nacional (capturas del dueño, 2026-09-07)

| fila            | ancla                                   | RD$/lb |
|-----------------|-----------------------------------------|--------|
| Coditos         | Milano 400 g · RD$38                    |  43,09 |
| Fideos          | Milano nido 350 g · RD$43               |  55,73 |
| Tocineta        | importada premium · **por libra** RD$265| 265,00 |
| Salchichas      | Wala hot dog 8/1 · 802,8 g · RD$155     |  87,60 |
| Chicharrón      | de cerdo · **por libra** RD$199         | 199,00 |
| Gallina criolla | Unipollo congelada · **por libra** RD$89|  89,00 |
| Pernil          | pierna de cerdo fresca · RD$135         | 135,00 |

`Pernil` va con confianza **low** y el resto `high`: su producto estaba NO DISPONIBLE en la
captura y la unidad no aparece impresa (se asume libra, que es como Nacional lista la carne).

## Lo demás que cambia, y por qué NO es opcional

  · El token sale de `_COUNTRY_CATALOG_UNPRICED_BY_COUNTRY`. Con precio, el rescate SOBRA
    (`_vc_comprable` los admite por la primera rama para todos los países); dejarlo puesto los
    marcaría a la vez como priced y como «sin precio», que es un bug real en
    `canonicalize_shopping_food_name` — lo detecta `test_i2_registry_collision_sweep_...`.
  · `Pernil` gana el alias **«pierna de cerdo»**. Ya tenía `pierna de cerdo para hornear`, pero no
    el llano, que es como lo dice la gente Y como lo lista Nacional. Sin él el nombre no resuelve.

## Lo que este script NO hace

`Sofrito` y `Pan rallado` siguen sin precio a propósito: se DERIVAN de un padre ya precificado
(los vegetales del sofrito; el pan) en vez de comprarse, igual que `Clara de huevo` sale del
cartón (`price_source='derived_huevo'`). Merecen su propia pasada, no un número inventado.

    python scripts/seed_do_shared_foods_2026_09_07.py            # simula
    python scripts/seed_do_shared_foods_2026_09_07.py --aplicar  # escribe
"""
import os
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# La consola de Windows es cp1252 y este script imprime «→» y «·».
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

APLICAR = "--aplicar" in sys.argv
HOY = "2026-09-07"
PERIODO = "2026-09"
LB = 453.592
OZ = 28.3495


def _lb(precio: float, gramos: float) -> float:
    return round(precio / gramos * LB, 2)


# nombre -> (precio_lb, confianza, market_packages, productos del súper)
PROMOCIONES = {
    "Coditos": (_lb(38.0, 400.0), "high",
                [{"unit": "paquete", "grams": 400, "label": "400 g", "price": 38.0}],
                [("Milano", "Paquete 400 gr", 400.0, 38.0),
                 ("Princesa", "Paquete Grandes 400 gr", 400.0, 35.0)]),
    "Fideos": (_lb(43.0, 350.0), "high",
               [{"unit": "paquete", "grams": 350, "label": "350 g", "price": 43.0}],
               [("Milano", "Paquete Nido Fino 350 gr", 350.0, 43.0),
                ("Milano", "Paquete Nido Grueso 350 gr", 350.0, 43.0),
                ("Milano", "Paquete Cortado 350 gr", 350.0, 43.0),
                ("Princesa", "Paquete Nidos 350 gr", 350.0, 35.0),
                ("Princesa", "Paquete Cortados 350 gr", 350.0, 39.0)]),
    "Tocineta": (265.0, "high",
                 [{"unit": "libra", "grams": 453.6, "label": "1 lb", "price": 265.0},
                  {"unit": "paquete", "grams": 226.8, "label": "8 oz", "price": 190.0}],
                 [("-", "Importada Premium Lb", LB, 265.0),
                  ("Wala", "Paquete 8 Oz", 8 * OZ, 190.0),
                  ("Chef", "Paquete Rebanada 8 Oz", 8 * OZ, 219.0),
                  ("Hugo Pork", "Paquete Tiras", None, 239.0),
                  ("Smithfield", "Paquete Baja En Sodio 1 Lb", LB, 485.0),
                  ("Smithfield", "Paquete Slice 16 Oz", LB, 479.0),
                  ("Smithfield", "Paquete Thick Cut 16 Oz", LB, 485.0),
                  ("Butterball", "Paquete Baja En Sodio 12 Oz", 12 * OZ, 375.0),
                  ("Butterball", "Paquete Pavo Original 12 Oz", 12 * OZ, 375.0),
                  ("Farmland", "Paquete Thick Cut 1 Lb", LB, 469.0),
                  ("Farmland", "Paquete Original 1 Lb", LB, 469.0)]),
    "Salchichas": (_lb(155.0, 802.8), "high",
                   [{"unit": "paquete", "grams": 802.8, "label": "8/1 (28,32 oz)", "price": 155.0},
                    {"unit": "paquete", "grams": 453.6, "label": "16 oz", "price": 170.0}],
                   [("Wala", "Paquete Hot Dog 8/1 28.32 Oz", 28.32 * OZ, 155.0),
                    ("Induveca", "Paquete Gigante 8 Ud", None, 145.0),
                    ("Chef", "Paquete Hot Dog 8 Uds", None, 145.0),
                    ("Chef", "Paquete Piggy Link 1 Lb", LB, 185.0),
                    ("Induveca", "Paquete De Pollo 8/1", None, 170.0),
                    ("Induveca", "Paquete Frank's Hot Dog 8/1", None, 199.0),
                    ("Emilios", "Paquete Jumbo Hot Dog 8/1", None, 199.0),
                    ("Emilios", "Paquete Familiar 24 uds", None, 349.0),
                    ("Gwaltney", "Paquete Hot Dog Original 8/1 16 Oz", LB, 170.0),
                    ("Checo", "Paquete Hot Dog De Pavo 8/1", None, 165.0),
                    ("Jaja", "Lata Estilo Vienna 5 Oz", 5 * OZ, 39.0),
                    ("Jaja", "Lata Vienna 15 Oz", 15 * OZ, 95.0),
                    ("Libby's", "Lata Vienna 4.6 Oz", 4.6 * OZ, 62.0)]),
    "Chicharrón": (199.0, "high",
                   [{"unit": "libra", "grams": 453.6, "label": "1 lb", "price": 199.0}],
                   [("-", "De Cerdo Lb", LB, 199.0),
                    ("Hugo Pork", "De Cerdo Lb", LB, 579.0),
                    ("Wala", "Paquete De Cerdo Con Picante 100 gr", 100.0, 159.0)]),
    "Gallina criolla": (89.0, "high",
                        [{"unit": "libra", "grams": 453.6, "label": "1 lb", "price": 89.0}],
                        [("Unipollo", "Congelada por Libra", LB, 89.0)]),
    "Pernil": (135.0, "low",
               [{"unit": "libra", "grams": 453.6, "label": "1 lb", "price": 135.0}],
               [("-", "Pierna De Cerdo Fresca Importada", LB, 135.0)]),
}

CATEGORIA_SUPER = {
    "Coditos": "Granos y cereales", "Fideos": "Granos y cereales",
    "Tocineta": "Carnes y embutidos", "Salchichas": "Carnes y embutidos",
    "Chicharrón": "Carnes y embutidos", "Gallina criolla": "Carnes y embutidos",
    "Pernil": "Carnes y embutidos",
}

ALIAS_NUEVOS = {"Pernil": ["pierna de cerdo"]}


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(_BACKEND / ".env")
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb

    hechos, prod_nuevos = [], 0
    with psycopg.connect(os.environ["NEON_DATABASE_URL"], row_factory=dict_row) as c:
        for nombre, (precio_lb, confianza, paquetes, productos) in PROMOCIONES.items():
            r = c.execute("SELECT price_per_lb, aliases FROM master_ingredients WHERE name=%s",
                          (nombre,)).fetchone()
            if not r:
                hechos.append(f"⛔ {nombre}: no existe en el catálogo")
                continue
            envase = paquetes[0]
            if float(r["price_per_lb"] or 0) == precio_lb:
                hechos.append(f"{nombre}: ya estaba a RD${precio_lb}/lb")
            else:
                if APLICAR:
                    c.execute(
                        """UPDATE master_ingredients SET
                             price_per_lb=%s, price_per_lb_base=%s,
                             price_per_unit=%s, price_per_unit_base=%s,
                             price_base_period=%s, price_source='nacional_tienda',
                             price_confidence=%s, price_captured_at=%s,
                             market_packages=%s, market_container=%s,
                             container_weight_g=%s, available_sizes_g=%s
                           WHERE name=%s""",
                        (precio_lb, precio_lb, envase["price"], envase["price"], PERIODO,
                         confianza, HOY, Jsonb(paquetes), envase["unit"],
                         int(envase["grams"]), Jsonb([int(p["grams"]) for p in paquetes]), nombre))
                hechos.append(f"{nombre}: 0 → RD${precio_lb}/lb ({confianza}) · "
                              f"{len(paquetes)} envase(s)")

            faltan = [a for a in ALIAS_NUEVOS.get(nombre, [])
                      if a not in (r["aliases"] or [])]
            if faltan:
                if APLICAR:
                    c.execute("UPDATE master_ingredients SET aliases = aliases || %s::text[] "
                              "WHERE name=%s", (faltan, nombre))
                hechos.append(f"{nombre}: alias {faltan}")

            for marca, pres, gramos, precio in productos:
                g = round(gramos, 1) if gramos else None
                ya = c.execute(
                    "SELECT id FROM supermarket_products WHERE lower(food_name)=lower(%s) "
                    "AND lower(coalesce(brand,''))=lower(%s) AND lower(coalesce(presentation,''))=lower(%s)",
                    (nombre, marca, pres)).fetchone()
                if ya:
                    continue
                if APLICAR:
                    c.execute(
                        """INSERT INTO supermarket_products
                           (food_name, brand, presentation, portion_label, price_rd, size_grams,
                            category, master_food_name, is_verified, active)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, true, true)""",
                        (nombre, None if marca == "-" else marca, pres,
                         (f"{g:g} g" if g else None), precio, g,
                         CATEGORIA_SUPER[nombre], nombre))
                prod_nuevos += 1

        if APLICAR:
            c.commit()

    print(("APLICADO" if APLICAR else "SIMULACIÓN (usa --aplicar)") + ":")
    for h in hechos:
        print("  ·", h)
    print(f"  · productos nuevos en el supermercado: {prod_nuevos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
