#!/usr/bin/env python
"""[P1-CATALOG-VARIETY-GAP · 2026-07-26] Inserta al catálogo (`master_ingredients`, Neon) los 11
alimentos que `/supermercado` YA vende y ningún plan podía usar porque no tenían macros.

Salieron de la auditoría de variedad del 2026-07-26: `supermarket_products` referencia 247 nombres de
alimento contra los 204 del catálogo; 46 no casaban ni por alias ni por plural. De esos 46, éstos 11
son los relevantes para variedad (el resto son leches infantiles/en polvo/saborizadas, sales
saborizadas y condimentos secos).

    Guisantes                Espaguetis              Huevos de codorniz
    Jamón de cerdo           Salchichas              Pan pita integral
    Pan de semillas          Harina de garbanzo      Harina de arroz integral
    Tomate enlatado          Queso de oveja

NUTRICIÓN: 100% USDA FoodData Central (SR Legacy), extraída por
`scripts/fetch_usda_foods_2026_07_26.py` con cruce Atwater por alimento. Vive en
`scripts/data/new_foods_variety_2026_07_26.json` — SSOT del dato, con `fdc_id` y `provenance` por
fila para que cada valor sea auditable. NO hay un solo número inventado.

⚠️ Dos alimentos traen Atwater >12% y es NORMAL, no un error: `Tomate enlatado` (+20,6%) y el `Kale`
que se descartó. En alimentos de ~16 kcal la fibra no aporta a Atwater y la diferencia absoluta son
3 kcal. Está anotado en su provenance.

⚠️ DESCARTADOS de la lista original de 14, con su razón — para que nadie los "complete" a ciegas:
    Kale             → YA está en el catálogo (alias 'col rizada'); el supermercado decía "Kale
                       Picado" y eso lo arregla la migración de nombres, no una fila nueva.
    Sardina fresca   → el catálogo tiene "Sardinas en lata"; USDA sólo devolvió enlatada-en-tomate
                       (414 mg Na), que es otro alimento. La fresca necesita fuente cruda propia.
    Yogurt de cabra  → USDA NO tiene yogurt de cabra; la búsqueda devolvía yogurt griego de VACA.
                       Usarlo sería inventar el dato con cara de fuente.
    Chuleta costillas→ redundante con "Chuleta" + "Costilla de cerdo" ya presentes.

PRECIOS: NO incluidos, y el script NO inserta sin precio (gate anti-precio-0, para no contaminar el
cálculo de la lista de compras con un alimento a RD$0). Llena el bloque PRICES con valores reales del
mercado RD (La Sirena / Nacional / colmado). `--print-template` imprime el esqueleto con el campo que
toca a cada uno según su `default_unit`.

Las columnas y la derivación de precio se REUSAN del lote de 2026-06-26 en vez de copiarlas: si el
esquema del catálogo cambia, hay un solo sitio que arreglar.

USO:
    python scripts/add_foods_variety_2026_07_26.py --print-template
    python scripts/add_foods_variety_2026_07_26.py              # DRY-RUN
    python scripts/add_foods_variety_2026_07_26.py --commit

[P2-LOGGER-EXEMPT: script CLI one-shot, la salida a stdout ES el producto]
"""
import datetime
import importlib.util
import json
import os
import sys

try:
    from dotenv import load_dotenv
    for _p in ("/opt/mealfit/backend/.env",
               os.path.join(os.path.dirname(__file__), "..", ".env")):
        if os.path.exists(_p):
            load_dotenv(_p)
            break
except Exception:
    pass

import psycopg

_AQUI = os.path.dirname(os.path.abspath(__file__))
_NEON = os.environ.get("NEON_DATABASE_URL_POOLED") or os.environ.get("NEON_DATABASE_URL")
COMMIT = "--commit" in sys.argv
PRINT_TEMPLATE = "--print-template" in sys.argv


def _lote_previo():
    """Reusa `_COLMAP` / `_is_priced` / `_derive_price_fields` del lote de 2026-06-26."""
    ruta = os.path.join(_AQUI, "add_foods_batch1_2026_06_26.py")
    if not os.path.exists(ruta):
        print("FATAL: falta add_foods_batch1_2026_06_26.py (de ahí salen las columnas y el precio)",
              file=sys.stderr)
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("_lote1", ruta)
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [sys.argv[0]]          # que el módulo no vea --commit al importarse
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# PRECIOS — LLENA AQUÍ con valores REALES del mercado RD. Deja en None lo que no tengas: ese
# alimento NO se inserta (mejor ausente que con precio inventado, que sesga la lista de compras).
# Formato alternativo por envases: {"packages": [{"unit": "funda", "grams": 400, "price": 95.0}, …]}
PRECIOS = {
    "Guisantes":                {"price_per_lb": None},
    "Espaguetis":               {"price_per_lb": None},
    "Huevos de codorniz":       {"price_per_unit": None},
    "Jamón de cerdo":           {"price_per_lb": None},
    "Salchichas":               {"price_per_unit": None},
    "Pan pita integral":        {"price_per_unit": None},
    "Pan de semillas":          {"price_per_unit": None},
    "Harina de garbanzo":       {"price_per_lb": None},
    "Harina de arroz integral": {"price_per_lb": None},
    "Tomate enlatado":          {"price_per_lb": None},
    "Queso de oveja":           {"price_per_lb": None},
}


def _registros():
    for p in (os.path.join(_AQUI, "data", "new_foods_variety_2026_07_26.json"),
              os.path.join(os.getcwd(), "scripts", "data", "new_foods_variety_2026_07_26.json"),
              "/tmp/new_foods_variety_2026_07_26.json"):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    print("FATAL: no se encontró new_foods_variety_2026_07_26.json", file=sys.stderr)
    sys.exit(1)


def main():
    recs = _registros()
    if PRINT_TEMPLATE:
        print("# Esqueleto PRECIOS (llena el campo que toca según default_unit):")
        for r in recs:
            campo = "price_per_unit" if r["default_unit"] == "unidad" else "price_per_lb"
            print(f'    "{r["name"]}": {{"{campo}": ___}},   # default_unit={r["default_unit"]}, '
                  f'{r["kcal"]} kcal, {r["protein_g"]} g prot')
        return 0
    if not _NEON:
        print("FATAL: NEON url ausente", file=sys.stderr)
        return 1

    L = _lote_previo()
    hoy = datetime.date.today()
    puestos = ya = sin_precio = 0
    with psycopg.connect(_NEON) as conn:
        existen = {r[0] for r in conn.execute("SELECT name FROM public.master_ingredients").fetchall()}
        for r in recs:
            nm = r["name"]
            if nm in existen:
                print(f"  ~ EXISTE, salto: {nm}")
                ya += 1
                continue
            precio = PRECIOS.get(nm) or {}
            if not L._is_priced(precio):
                print(f"  ⏳ SIN PRECIO, salto: {nm}  (llena PRECIOS['{nm}'])")
                sin_precio += 1
                continue
            cols = {
                "slug": r["slug"], "name": nm, "category": r["category"],
                "aliases": r.get("aliases") or [], "default_unit": r["default_unit"],
                "is_dominican_cultivar": bool(r.get("is_dominican_cultivar")),
                "density_g_per_cup": r.get("density_g_per_cup"),
                "density_g_per_unit": r.get("density_g_per_unit"),
                "nutrition_source": "usda", "nutrition_source_date": hoy,
                "fdc_id": r.get("fdc_id"),
            }
            cols.update(L._derive_price_fields(precio))
            for k, dbcol in L._COLMAP.items():
                cols[dbcol] = r.get(k)
            nombres = list(cols.keys())
            if COMMIT:
                conn.execute(
                    f"INSERT INTO public.master_ingredients ({', '.join(nombres)}) "
                    f"VALUES ({', '.join(['%s'] * len(nombres))})",
                    [cols[c] for c in nombres])
            print(f"  {'+ INSERTADO' if COMMIT else '+ (dry) insertaría'}: {nm} [{r['category']}] "
                  f"{r['kcal']}kcal/{r['protein_g']}P/{r.get('sodium_mg','?')}mgNa "
                  f"fdc={r.get('fdc_id')}")
        if COMMIT:
            conn.commit()
            print(f"\nCOMMITTED. insertados={puestos}, ya-existen={ya}, sin-precio={sin_precio}")
        else:
            print(f"\nDRY-RUN. ya-existen={ya}, sin-precio={sin_precio}. "
                  f"Llena PRECIOS y re-corre con --commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
