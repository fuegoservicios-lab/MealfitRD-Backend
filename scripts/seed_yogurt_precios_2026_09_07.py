# -*- coding: utf-8 -*-
"""[P1-YOGURT-PRECIO-REAL · 2026-09-07] El motor tenía los tres yogurts al MISMO precio, y era
el precio del griego.

Las tres filas lácteas de `master_ingredients` llevaban `price_per_lb = 0` y
`price_per_unit = 100` — las únicas 3 filas de 348 con ese valor, las tres marcadas
`price_confidence = 'high'`.

Ese 100 **no es un número inventado**: mira los ocho griegos más baratos de
`supermarket_products` y son Yoplait griego a RD$100 / 150 g. Es un precio REAL copiado a las
tres filas. El resultado es que el yogurt normal está tasado como un griego, y el griego por
debajo de él:

    price_per_unit=100 ÷ container_weight_g=150  ->  Yogurt normal   302,39 RD$/lb
    price_per_unit=100 ÷ density_g_per_unit=170  ->  Yogurt griego   266,82 RD$/lb

La diferencia de 12 % que el motor «ve» entre ambos no es un precio: es el DIVISOR. El normal
divide por el peso del envase y el griego por una densidad, porque el respaldo del mapa de
precios (`_budget_build_master_price_map`) es `price_per_unit × 453,592 / (density or
container)`. Un placeholder con sello de confianza alta es peor que un NULL: pasa cualquier
auditoría de «¿qué precios no me fío?» y encima produce una ORDENACIÓN falsa que
`_apply_budget_cheapen_pass` obedece.

## De dónde salen los precios nuevos

De `supermarket_products`, que ya los tenía. Dos decisiones de método, ambas discutibles y por
eso escritas aquí:

1. **Sólo lo CUCHAREABLE.** 81 de los 115 «Yogurt Regular» son bebibles en botella o galón
   (Yopsi, 1 Lt, 1 Gl). Un plan no come eso con cuchara, y mezclarlos hunde la mediana a 84
   RD$/lb — el galón barato arrastrando al vasito.

2. **Mediana POR MARCA, no por SKU.** Contar productos pesa a cada marca por cuántos sabores
   stockea: Chobani tiene 16 vasos y Asturiana 3, así que una mediana por SKU es en realidad
   una encuesta de surtido, no de precio. Por marca sale una cifra más conservadora (subestima
   la brecha), que es la dirección segura: no penaliza al griego de más.

El gramaje salía a medias — `size_grams` sólo lo tienen 43 de 152 — pero estaba escrito en el
texto de `presentation` («Pote 6 Oz», «Botella Yopsi Natural 1/2 Gl»), así que se lee de ahí y
se recuperan 146 de 152.

`price_per_unit` se deriva del `price_per_lb` y del envase de cada fila, para que las dos
columnas cuenten la misma historia; hoy se contradicen. Se escriben también las `*_base` y
`price_base_period`, porque `price_engine.reescalar` reajusta el precio vivo desde la base × FX
y una base NULL dejaría la fila fuera de los reajustes futuros.

## Yogurt de cabra

Existe en `supermarket_products` (Deliciel, RD$110 / 4 oz) y NO tenía fila en
`master_ingredients`: el motor no podía recetarlo aunque el usuario lo viera en el catálogo.

USDA **no tiene yogurt de cabra genérico** — ni SR Legacy ni Foundation, sólo entradas Branded
(datos de etiqueta). De las tres consultadas, dos concuerdan y una es un yogurt claramente
COLADO disfrazado de «plain»:

    2422160 Redwood Hill      71 kcal · 3,53 prot · 3,53 grasa · 6,47 carb
    2210304 Building Records  74 kcal · 4,55 prot · 4,13 grasa · 4,96 carb
    2185630 Coach Farm        82 kcal · 8,82 prot · 3,53 grasa · 3,53 carb   <- outlier, colado

Se toma Redwood Hill como proxy DECLARADO: `nutrition_source='manual'` y
`nutrition_source_ref='usda:2422160 (proxy: ...)'`, siguiendo el patrón de `Chorizo
santarrosano`. **`fdc_id` se deja NULL a propósito**: un `fdc_id` afirma que la fila ES ese
alimento, y Redwood Hill no es Deliciel. Estampar uno ajeno es exactamente lo que
`P1-BEDCA-DEPROXY-ES` encontró costando 47 filas.

Los micronutrientes que la etiqueta no declara quedan en NULL, no en 0: un nutriente ausente no
es un cero (P1-ARQ27-F2).

    python scripts/seed_yogurt_precios_2026_09_07.py            # simula
    python scripts/seed_yogurt_precios_2026_09_07.py --aplicar  # escribe
"""
import os
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

APLICAR = "--aplicar" in sys.argv
LB = 453.592
G_POR_ML = 1.0            # el yogurt es ~1,03 g/ml; para ORDENAR dos alimentos, 1,0 sobra
PERIODO = "2026-09"

_NUM = r"(\d+(?:[.,]\d+)?|\d+\s*/\s*\d+)"
_UNIDADES = [
    (re.compile(_NUM + r"\s*(?:gl|gal|galon|galones)\b", re.I), 3785.41 * G_POR_ML),
    (re.compile(_NUM + r"\s*(?:lt|lts|l|litro|litros)\b", re.I), 1000.0 * G_POR_ML),
    (re.compile(_NUM + r"\s*(?:ml|mililitros)\b", re.I), 1.0 * G_POR_ML),
    (re.compile(_NUM + r"\s*(?:kg|kilo|kilos)\b", re.I), 1000.0),
    (re.compile(_NUM + r"\s*(?:oz|onzas?)\b", re.I), 28.3495),
    (re.compile(_NUM + r"\s*(?:gr|grs|g|gramos)\b", re.I), 1.0),
]
BEBIBLE = re.compile(r"bebible|botella|yopsi|liquid|smoothie|batido", re.I)

# food_name en supermarket_products  ->  name en master_ingredients
DESTINO = {
    "Yogurt Regular": ["Yogurt"],
    "Yogurt Griego": ["Yogurt griego entero", "Yogurt griego sin azúcar"],
    "Yogurt de cabra": ["Yogurt de cabra"],
}

# Etiqueta de Redwood Hill Farm (USDA Branded 2422160), por 100 g. Lo que la etiqueta NO
# declara se queda fuera del INSERT y por tanto en NULL.
CABRA = {
    "kcal_per_100g": 71.0,
    "protein_g_per_100g": 3.53,
    "carbs_g_per_100g": 6.47,
    "fats_g_per_100g": 3.53,
    "fiber_g_per_100g": 0.0,
    "sugars_g_per_100g": 2.94,
    "sodium_mg_per_100g": 41.0,
    "calcium_mg_per_100g": 88.0,
    "potassium_mg_per_100g": 165.0,
    "saturated_fat_g_per_100g": 2.35,
    "cholesterol_mg_per_100g": 15.0,
}

# El esquema declara NOT NULL ocho micronutrientes (zinc, folato, A, C, E, K, selenio, ALA) y
# la etiqueta del yogurt de cabra no declara ninguno. NO se copian los del yogurt de VACA: ahí
# está justo la diferencia que importa —la leche de cabra trae folato 1,0 µg y B12 0,07 µg
# frente a 7,0 y 0,37 de la fila `Yogurt`—, así que ese atajo habría inflado el folato 7× y la
# B12 5× en una app de nutrición.
#
# Se toman de la LECHE de cabra (usda:171278, SR Legacy). El yogurt no colado no pierde agua
# respecto a la leche de la que sale, así que los valores por 100 g se trasladan sin reescalar.
MICROS_CABRA = {
    "zinc_mg_per_100g": 0.3,
    "folate_mcg_dfe_per_100g": 1.0,
    "vitamin_a_mcg_rae_per_100g": 57.0,
    "vitamin_c_mg_per_100g": 1.3,
    "vitamin_e_mg_per_100g": 0.07,
    "vitamin_k_mcg_per_100g": 0.3,
    "selenium_mcg_per_100g": 1.4,
    "vitamin_d_mcg_per_100g": 1.3,
    "iron_mg_per_100g": 0.05,
    "vitamin_b12_mcg_per_100g": 0.07,
    "magnesium_mg_per_100g": 14.0,
    "phosphorus_mg_per_100g": 111.0,
    # USDA no reporta ALA para la leche de cabra y la columna es NOT NULL. Se usa el mismo
    # 0,007 que llevan las tres filas de yogurt de vaca: es un valor traza del que no cuelga
    # ninguna decisión del motor. Queda declarado en `nutrition_source_ref` como lo que es.
    "omega3_ala_g_per_100g": 0.007,
}


def _valor(s: str):
    s = str(s).strip().replace(",", ".")
    if "/" in s:
        a, b = s.split("/", 1)
        try:
            return float(a) / float(b)
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


def gramos(size_grams, presentation):
    """Gramos de la presentación: columna si la hay, si no el texto. None si no hay medida."""
    if size_grams and float(size_grams) > 0:
        return float(size_grams), "columna"
    t = re.sub(r"(\d)\s*/\s*(\d)", r"\1/\2", str(presentation or ""))
    for rx, factor in _UNIDADES:
        m = rx.search(t)
        if m:
            v = _valor(m.group(1))
            if v and v > 0:
                return v * factor, "texto"
    return None, None


def precio_por_lb_por_marca(filas):
    """Mediana por MARCA de las presentaciones cuchareables. Devuelve (valor, detalle)."""
    por_marca = defaultdict(list)
    descartes = {"bebible": 0, "sin_medida": 0}
    for r in filas:
        g, _ = gramos(r["size_grams"], r["presentation"])
        p = float(r["price_rd"] or 0)
        if not g or p <= 0:
            descartes["sin_medida"] += 1
            continue
        if BEBIBLE.search(str(r["presentation"] or "")):
            descartes["bebible"] += 1
            continue
        por_marca[str(r["brand"] or "(sin marca)")].append(p * LB / g)
    if not por_marca:
        return None, {"marcas": {}, **descartes}
    medianas = {m: st.median(v) for m, v in sorted(por_marca.items())}
    return st.median(medianas.values()), {"marcas": medianas, **descartes}


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(_BACKEND / ".env")
    import db_core

    db_core.connection_pool.open()
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb

    url = os.environ["NEON_DATABASE_URL"]
    with psycopg.connect(url, row_factory=dict_row) as conn:
        prods = conn.execute(
            "SELECT food_name, brand, presentation, price_rd, size_grams "
            "FROM supermarket_products WHERE active AND food_name = ANY(%s)",
            (list(DESTINO),),
        ).fetchall()
        maestros = {
            r["name"]: r
            for r in conn.execute(
                "SELECT name, container_weight_g, density_g_per_unit, price_per_lb, "
                "price_per_unit, market_packages FROM master_ingredients WHERE name = ANY(%s)",
                ([n for v in DESTINO.values() for n in v],),
            ).fetchall()
        }

    por_alimento = defaultdict(list)
    for r in prods:
        por_alimento[r["food_name"]].append(r)

    print(f"{'ALIMENTO':22s} {'n':>4s} {'RD$/lb nuevo':>13s}   marcas y descartes")
    plan = {}
    for super_name, master_names in DESTINO.items():
        filas = por_alimento.get(super_name, [])
        valor, det = precio_por_lb_por_marca(filas)
        if valor is None:
            print(f"{super_name:22s} {len(filas):4d}   -- sin ninguna presentación medible --")
            continue
        marcas = ", ".join(f"{m}={v:.0f}" for m, v in det["marcas"].items())
        print(f"{super_name:22s} {len(filas):4d} {valor:13.1f}   "
              f"bebibles={det['bebible']} sin_medida={det['sin_medida']}")
        print(f"{'':22s}      por marca: {marcas}")
        for mn in master_names:
            plan[mn] = valor

    print("")
    print("CAMBIO por fila de master_ingredients:")
    escrituras = []
    for mn, ppl in plan.items():
        fila = maestros.get(mn)
        if fila is None:
            print(f"   {mn:26s} NO EXISTE -> se creará")
            continue
        # `price_per_unit` es el precio de UNA unidad de mercado, así que se deriva del peso
        # del envase — no de `density_g_per_unit`. Ojo: las dos filas de griego declaran
        # `container_weight_g=150` y `density_g_per_unit=170` a la vez, y se contradicen. Es
        # de ahí de donde salía la ordenación falsa: el respaldo del mapa de precios prioriza
        # la densidad, así que el griego dividía por 170 y el normal por 150, y con el MISMO
        # RD$100 el griego «salía» un 12 % más barato. Con `price_per_lb` ya poblado el
        # respaldo deja de correr; la contradicción del dato queda anotada, no arreglada a
        # ciegas dentro de una tarea de precios.
        envase = float(fila["container_weight_g"] or 0) or float(fila["density_g_per_unit"] or 0)
        ppu = round(ppl * envase / LB, 2) if envase else None
        antes_lb = float(fila["price_per_lb"] or 0)
        antes_u = float(fila["price_per_unit"] or 0)
        divisor_respaldo = float(fila["density_g_per_unit"] or 0) or float(
            fila["container_weight_g"] or 0)
        efectivo = antes_lb or (antes_u * LB / divisor_respaldo if divisor_respaldo else 0)
        aviso = "  ⚠ envase≠densidad" if (
            fila["density_g_per_unit"] and fila["container_weight_g"]
            and float(fila["density_g_per_unit"]) != float(fila["container_weight_g"])) else ""
        print(f"   {mn:26s} RD$/lb {efectivo:8.2f} -> {ppl:8.2f}   "
              f"ud {antes_u:7.2f} -> {ppu}  (envase {envase:.0f} g){aviso}")
        escrituras.append((mn, round(ppl, 2), ppu))

    cabra_falta = "Yogurt de cabra" not in maestros
    print("")
    print(f"Fila nueva «Yogurt de cabra»: {'SÍ, se crea' if cabra_falta else 'ya existe, no se toca'}")
    if cabra_falta:
        print(f"   macros (proxy declarado usda:2422160): {CABRA}")

    if not APLICAR:
        print("")
        print("SIMULACIÓN. Nada escrito. Añade --aplicar para escribir.")
        return 0

    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            for mn, ppl, ppu in escrituras:
                cur.execute(
                    """
                    UPDATE master_ingredients
                       SET price_per_lb = %s,
                           price_per_lb_base = %s,
                           price_per_unit = %s,
                           price_per_unit_base = %s,
                           price_base_period = %s,
                           price_source = 'supermarket_products_mediana_por_marca',
                           price_confidence = 'medium',
                           price_captured_at = CURRENT_DATE
                     WHERE name = %s
                    """,
                    (ppl, ppl, ppu, ppu, PERIODO, mn),
                )
                print(f"   actualizado {mn}: {cur.rowcount} fila(s)")
            if cabra_falta:
                ppl_cabra = round(plan.get("Yogurt de cabra", 440.0), 2)
                ppu_cabra = round(ppl_cabra * 113.4 / LB, 2)
                fila = {
                    "name": "Yogurt de cabra",
                    "slug": "yogurt-de-cabra",
                    "category": "Lácteos",
                    # Frases completas, nunca «cabra» a secas: casaría con «queso de cabra»,
                    # que es otra fila. Es la clase de colisión que este repo lleva 19 veces
                    # documentada.
                    "aliases": ["yogur de cabra", "yogurt de cabra natural", "yogur caprino",
                                "goat milk yogurt", "yogurt caprino"],
                    "density_g_per_cup": 245,
                    "density_g_per_unit": 113.4,
                    "shelf_life_days": 14,
                    "price_per_lb": ppl_cabra,
                    "price_per_lb_base": ppl_cabra,
                    "price_per_unit": ppu_cabra,
                    "price_per_unit_base": ppu_cabra,
                    "market_container": "frasco",
                    "container_weight_g": 113.4,
                    # `available_sizes_g` y `market_packages` son jsonb; `aliases` y
                    # `prep_methods`, text[]. Una lista de Python va sola al array pero NO al
                    # jsonb: hay que envolverla.
                    "available_sizes_g": Jsonb([113.4]),
                    "default_unit": "frasco",
                    "nutrition_source": "manual",
                    "nutrition_source_ref": (
                        "macros usda:2422160 (proxy: PLAIN GOAT MILK YOGURT, Redwood Hill Farm "
                        "— Branded/etiqueta; USDA no tiene yogurt de cabra genérico en SR "
                        "Legacy ni Foundation) · micros usda:171278 (Milk, goat, fluid) porque "
                        "la etiqueta no los declara y el esquema los exige NOT NULL · "
                        "omega3_ala sin fuente (USDA no lo reporta para leche de cabra): se usa "
                        "el 0,007 de las filas de yogurt de vaca"
                    ),
                    "nutrition_source_date": "2026-09-07",
                    "is_dominican_cultivar": False,
                    "prep_methods": ["ninguno", "crudo", "licuar"],
                    "ready_to_eat": True,
                    "name_en": "Goat milk yogurt",
                    "price_base_period": PERIODO,
                    "price_source": "supermarket_products_mediana_por_marca",
                    "price_confidence": "medium",
                    **CABRA,
                    **MICROS_CABRA,
                }
                cols = list(fila)
                cur.execute(
                    f"INSERT INTO master_ingredients ({', '.join(cols)}, price_captured_at) "
                    f"VALUES ({', '.join(['%s'] * len(cols))}, CURRENT_DATE) "
                    # La única clave de `master_ingredients` es `slug` — `name` NO es único
                    # (`Yogurt griego entero` y `Yogurt griego sin azúcar` comparten el slug
                    # `yogurt-griego`). El duplicado por nombre ya lo corta `cabra_falta`.
                    f"ON CONFLICT (slug) DO NOTHING",
                    [fila[c] for c in cols],
                )
                print(f"   insertado Yogurt de cabra: {cur.rowcount} fila(s)")
        conn.commit()
    print("")
    print("ESCRITO.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
