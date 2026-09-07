# -*- coding: utf-8 -*-
"""[P1-PASTA-BLANCA-CLARAS-SKU · 2026-09-07] Clara pasteurizada + la pasta blanca que faltaba.

Dos huecos que el dueño encontró mirando la app, no un linter:

  1. `Clara de huevo` no tenía NINGÚN producto que comprar. Su precio sale del cartón de huevos
     (`price_source='derived_huevo'`), que es correcto —para tener una clara compras el huevo—
     pero deja fuera el envase que existe de verdad en el estante.
  2. El motor no conocía el espagueti BLANCO. El único «espagueti» del catálogo era un alias de
     `Pasta integral`, así que un plan no podía pedir pasta normal… mientras la tabla del
     supermercado tenía SEIS productos de espaguetis que nadie podía referenciar.

DE DÓNDE SALE CADA NÚMERO — es lo único que hay que revisar aquí:

  · Clara de huevo pasteurizada Don Papito 400 g · RD$154,95 — ETIQUETA REAL del Supermercado
    Nacional (foto, 2026-09-07). Va tal cual, `is_verified`.
  · Espaguetis · RD$98,88/lb — DERIVADO de un producto real que ya estaba en la propia tabla del
    súper: Pasta Zara 500 g a RD$109 (109/500 × 453,592). Se elige ésa y no la media de las seis
    porque dos son gluten free (otro producto) y Milano a RD$42 es el extremo barato; Pasta Zara
    es el paquete de 500 g de gama media, el mismo formato que ya usa `Pasta integral`. Queda en
    98,88 contra 189,60 de la integral: la blanca a la mitad, que es lo que se ve en el estante.
  · Nutrición de `Espaguetis` — COPIADA de `Coditos` (FDC 169736, pasta seca de trigo
    ENRIQUECIDA), misma clase de alimento. `fdc_id` queda NULO a propósito: un fdc_id es una
    AFIRMACIÓN sobre una fila concreta de la USDA y aquí no se hizo ese lookup. La procedencia va
    escrita en `nutrition_source_ref`, que para eso está.

## Lo que este script INTENTÓ hacer y NO debe hacer nunca

La primera versión también les puso precio a `Coditos` y `Fideos`, que estaban a 0. Parecía un
hueco obvio. **No lo era**: son altas T7 y T5 del catálogo de países, y estar sin precio es su
estado DELIBERADO. Ponérselo puso rojos tres guards a la vez —dos que exigen precio 0 en esas
altas y `test_i2_registry_collision_sweep_extendido_a_aliases`, que detectó que los alias
`fideos`/`fideos finos` pasaban a colisionar con `is_country_catalog_unpriced_item`, o sea un
bug real en `canonicalize_shopping_food_name`—. Se revirtió a NULL/0 exacto (las 139 filas sin
precio comparten la misma forma: `market_container`, `container_weight_g`, `available_sizes_g` y
`market_packages` todos nulos).

> Un cero en una columna de precio no es siempre un dato que falta. Antes de rellenarlo, busca
> quién lo puso a cero a propósito.

## Efecto colateral que SÍ hay que aceptar a mano

Añadir una fila al catálogo cambia el corpus de `test_c3_durable_guard_do_corpus_retarget_
baseline`. Regenerado con `gen_do_corpus_retarget_baseline_2026_08_18.py` y revisado el diff: 5
strings nuevos y 5 resoluciones cambiadas. Dos son la mejora buscada (`espagueti` y `spaghetti`
resolvían a sí mismos, sin fila; ahora a `Espaguetis`). Las otras tres son NOMBRES DE PLATO que
pasan de la proteína a la pasta («Espaguetis con atún y salsa criolla»: `Atún en agua` →
`Espaguetis`). No es un cambio de conducta: producción llama `normalize_name` sobre strings de
INGREDIENTE, nunca sobre un título de plato — los títulos están en el corpus como red de aviso.

    python scripts/seed_claras_pasta_2026_09_07.py            # dice lo que haría
    python scripts/seed_claras_pasta_2026_09_07.py --aplicar  # lo hace

Idempotente: re-correrlo no duplica nada.
"""
import os
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

APLICAR = "--aplicar" in sys.argv
HOY = "2026-09-07"
PERIODO = "2026-09"

# 109 RD$ / 500 g × 453,592 g/lb  (Pasta Zara 500 g, ya en `supermarket_products`)
PRECIO_LB_PASTA = round(109.0 / 500.0 * 453.592, 2)
PAQUETE_PASTA = [{"unit": "paquete", "grams": 500, "label": "500 g", "price": 109}]

PRODUCTO_CLARA = {
    "food_name": "Clara de huevo",
    "brand": "Don Papito",
    "presentation": "Botella Pasteurizada 400 gr",
    "portion_label": "400 g",
    "price_rd": 154.95,
    "size_grams": 400,
    "category": "Lácteos y huevos",
    "master_food_name": "Clara de huevo",
    "description": "Clara de huevo pasteurizada, lista para usar.",
    "is_verified": True,
    "active": True,
}


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(_BACKEND / ".env")
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb

    hechos = []
    with psycopg.connect(os.environ["NEON_DATABASE_URL"], row_factory=dict_row) as c:
        # ------------------------------------------------------------ 1. la clara
        ya = c.execute(
            "SELECT id FROM supermarket_products WHERE lower(food_name)=lower(%s) "
            "AND lower(coalesce(brand,''))=lower(%s) AND lower(coalesce(presentation,''))=lower(%s)",
            (PRODUCTO_CLARA["food_name"], PRODUCTO_CLARA["brand"], PRODUCTO_CLARA["presentation"]),
        ).fetchone()
        if ya:
            hechos.append("clara pasteurizada: ya estaba")
        else:
            cols = list(PRODUCTO_CLARA)
            if APLICAR:
                c.execute(
                    f"INSERT INTO supermarket_products ({', '.join(cols)}) "
                    f"VALUES ({', '.join(['%s'] * len(cols))})",
                    tuple(PRODUCTO_CLARA[k] for k in cols),
                )
            hechos.append("clara pasteurizada Don Papito 400 g · RD$154,95 → supermarket_products")

        # -------------------------------------------------- 2. Espaguetis (blanca)
        base = c.execute("SELECT * FROM master_ingredients WHERE name='Coditos'").fetchone()
        if not base:
            print("⛔ no existe `Coditos`: sin plantilla nutricional, no sigo")
            return 2
        if c.execute("SELECT id FROM master_ingredients WHERE lower(name)='espaguetis'").fetchone():
            hechos.append("Espaguetis: ya estaba")
        else:
            fila = {k: base[k] for k in base if k.endswith("_per_100g")}
            fila.update({
                "slug": "espaguetis", "name": "Espaguetis", "category": "Despensa",
                "name_en": "Spaghetti, dry",
                "aliases": ["espagueti", "spaghetti", "pasta blanca", "pasta larga",
                            "espaguetis blancos", "tallarines"],
                "default_unit": "paquete", "market_container": "paquete",
                "container_weight_g": 500, "available_sizes_g": Jsonb([500]),
                "density_g_per_cup": 140, "shelf_life_days": base["shelf_life_days"],
                "price_per_lb": PRECIO_LB_PASTA, "price_per_lb_base": PRECIO_LB_PASTA,
                "price_per_unit": 109, "price_per_unit_base": 109,
                "price_base_period": PERIODO, "price_source": "nacional_tienda",
                "price_confidence": "high", "price_captured_at": HOY,
                "market_packages": Jsonb(PAQUETE_PASTA),
                "nutrition_source": "manual", "nutrition_source_date": HOY,
                "nutrition_source_ref": ("derivado de `Coditos` (FDC 169736, pasta seca de trigo "
                                         "enriquecida): misma clase de alimento. Sin lookup propio "
                                         "en la USDA, por eso `fdc_id` va nulo."),
                "fdc_id": None, "is_dominican_cultivar": False,
                "prep_methods": ["hervir", "guisar"], "ready_to_eat": False,
            })
            cols = list(fila)
            if APLICAR:
                c.execute(
                    f"INSERT INTO master_ingredients ({', '.join(cols)}) "
                    f"VALUES ({', '.join(['%s'] * len(cols))})",
                    tuple(fila[k] for k in cols),
                )
            hechos.append(f"Espaguetis (pasta blanca seca) · RD${PRECIO_LB_PASTA}/lb → master_ingredients")

        # `Coditos` y `Fideos` NO se tocan: ver el docstring. Su 0 es deliberado.
        for nombre in ("Coditos", "Fideos"):
            r = c.execute("SELECT price_per_lb FROM master_ingredients WHERE name=%s",
                          (nombre,)).fetchone()
            if r and float(r["price_per_lb"] or 0) != 0:
                hechos.append(f"⚠ {nombre} tiene precio ({r['price_per_lb']}) y es alta de país: "
                              f"debería estar en 0 — revisa `test_p1_country_system_f2`")

        if APLICAR:
            c.commit()

    print(("APLICADO" if APLICAR else "SIMULACIÓN (usa --aplicar)") + ":")
    for h in hechos:
        print("  ·", h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
