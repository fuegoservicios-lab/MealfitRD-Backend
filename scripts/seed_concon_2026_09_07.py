# -*- coding: utf-8 -*-
"""[P1-CONCON-ALIAS · 2026-09-07] El concón entra al catálogo como lo que es: arroz.

El concón —la costra tostada del fondo del caldero— no existía en el catálogo. Medido antes de
tocar nada:

    normalize_name('arroz blanco cocido') = 'Arroz blanco'   <- «cocido» es stopword
    normalize_name('concón')              = 'Concón'          <- se devuelve a sí mismo: NO resuelve

Un nombre que no resuelve es un nombre invisible: no casa en la Nevera (`pantry_names_match`), no
lo ve el guard de coherencia, y el backstop de alergias no puede razonar sobre él.

## Por qué ALIAS y no fila propia

Es la regla que dio el dueño y es la correcta: **«si tienes arroz, prácticamente tienes concón»** —
la misma forma que `Clara de huevo` (`price_source='derived_huevo'`: no tiene SKU, su precio sale
del cartón porque para tener una clara compras el huevo). El concón no se compra: sale del arroz
que ya está en la lista.

Una fila propia exigiría macros de «cocido y tostado» que yo no tengo. Inventarlas sería peor que
no tenerlo: un número fabricado con formato de dato no se distingue de uno medido.

Y el alias no introduce ninguna inconsistencia NUEVA: le da exactamente el trato que «arroz blanco
cocido» ya recibe hoy, porque `cocido` está en `_NORMALIZE_STOPS` y ambos colapsan a la misma fila.

> Hay una inconsistencia PREEXISTENTE que este script NO crea ni arregla: la fila `Arroz blanco`
> son 358,6 kcal/100 g, o sea arroz CRUDO (fdc 2512381, densidad 185 g/taza = grano seco), y
> «arroz cocido» resuelve ahí igual. Una línea de receta en gramos de arroz YA cocido cuenta de
> más. Es un problema de modelado del arroz, anterior y más ancho que el concón — merece su propia
> medición, no un parche colgado de esta tarea.

## Efecto colateral que hay que aceptar a mano

Añadir un alias a una fila PRE-FASE cambia el corpus de
`test_c3_durable_guard_do_corpus_retarget_baseline`. Hay que regenerar con
`gen_do_corpus_retarget_baseline_2026_08_18.py` y revisar el diff línea a línea.

    python scripts/seed_concon_2026_09_07.py            # simula (y comprueba colisiones)
    python scripts/seed_concon_2026_09_07.py --aplicar  # escribe
"""
import os
import sys
import unicodedata
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

APLICAR = "--aplicar" in sys.argv
FILA = "Arroz blanco"

# Las formas con que una receta puede nombrarlo. Frases completas y no palabras sueltas: «raspa»
# a secas colisionaría con «raspar» en un paso, y esa clase de error (sal⊂salsa, res⊂fresco,
# pollo⊂repollo) ya tiene 17 apariciones documentadas en este proyecto.
ALIAS = ["concón", "concon", "arroz tostado", "raspa de arroz"]


def _sa(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(_BACKEND / ".env")
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(os.environ["NEON_DATABASE_URL"], row_factory=dict_row) as c:
        filas = c.execute("SELECT name, aliases FROM master_ingredients").fetchall()

        # Guard de colisión ANTES de escribir: un alias que sea subcadena del nombre o alias de
        # otra fila secuestra ese alimento en silencio.
        choques = []
        for a in ALIAS:
            na = _sa(a)
            for f in filas:
                if f["name"] == FILA:
                    continue
                for cand in [f["name"]] + list(f["aliases"] or []):
                    nc = _sa(cand)
                    if na in nc or nc in na:
                        choques.append((a, f["name"], cand))
        if choques:
            print("⛔ colisión — NO se escribe nada:")
            for a, fila, cand in choques:
                print(f"    alias {a!r} choca con {fila!r} (por {cand!r})")
            return 3

        r = c.execute("SELECT aliases FROM master_ingredients WHERE name=%s", (FILA,)).fetchone()
        if not r:
            print(f"⛔ no existe la fila {FILA!r}")
            return 2
        actuales = list(r["aliases"] or [])
        faltan = [a for a in ALIAS if _sa(a) not in {_sa(x) for x in actuales}]
        if not faltan:
            print("SIN CAMBIOS: los alias ya estaban")
            return 0
        if APLICAR:
            c.execute("UPDATE master_ingredients SET aliases = aliases || %s::text[] WHERE name=%s",
                      (faltan, FILA))
            c.commit()

    print(("APLICADO" if APLICAR else "SIMULACIÓN (usa --aplicar)") + ":")
    print(f"  · sin colisiones contra las {len(filas)} filas del catálogo")
    print(f"  · {FILA}: + {faltan}")
    print("  · recuerda regenerar el baseline del corpus y revisar su diff")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
