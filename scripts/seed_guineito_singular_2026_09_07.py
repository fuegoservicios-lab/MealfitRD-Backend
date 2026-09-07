# -*- coding: utf-8 -*-
"""[P2-GUINEITO-SINGULAR · 2026-09-07] El diminutivo estaba curado sólo en plural.

`Guineo verde` lleva tres alias y los tres son PLURAL:

    ['guineítos verdes', 'guineitos verdes', 'guineos verdes']

Falta el singular. Quien curó ese set escribió la forma que tenía delante y no la otra, y el
resultado es que «medio guineíto verde» —10 comidas vivas— no resuelve a ningún alimento del
catálogo. Un nombre que no resuelve es invisible para toda la capa culinaria: ni V3 lo reclama,
ni V7e puede comparar cantidades sobre él.

## Medido antes de tocar nada

Simulando el alias en memoria sobre las 1.194 comidas vivas, con las once capas:

    V7e   159 -> 162   (+3)
    resto     sin cambio
    disparos perdidos: 0

Los tres son la misma forma —la lista compra medio guineíto y el paso manda aplastar dos— y uno
de ellos es el mangú que el dueño marcó a mano en el juicio ciego («B utiliza 2 unidades en lugar
de ½»). O sea: el alias no inventa capturas, destapa las que ya existían detrás de un nombre que
el motor no sabía leer.

Ese +3 es lo que separa este cambio del anterior. Ese mismo día se construyó la lectura de
«medio» escrito con palabra —82 comidas, 6,9 % de la flota— y dio **+0 disparos**: se revirtió.
La prevalencia de un patrón no es la prevalencia de un defecto.

## Por qué frase completa y no «guineito» a secas

«guineito verde» son dos palabras a propósito. El diminutivo suelto colisionaría con `Guineo`
(otra fila, la del guineo maduro) en cualquier texto que hable de guineítos sin especificar, y
esa clase de error —el nombre corto que se traga al largo— lleva 19 apariciones documentadas en
este repo.

## Efecto colateral que hay que aceptar a mano

Añadir alias a una fila PRE-FASE cambia el corpus de
`test_c3_durable_guard_do_corpus_retarget_baseline`. Hay que regenerar con
`gen_do_corpus_retarget_baseline_2026_08_18.py` y revisar el diff línea a línea.

    python scripts/seed_guineito_singular_2026_09_07.py            # simula
    python scripts/seed_guineito_singular_2026_09_07.py --aplicar  # escribe
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
FILA = "Guineo verde"
NUEVOS = ["guineito verde", "guineíto verde"]


def _sa(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(_BACKEND / ".env")
    import db_core

    db_core.connection_pool.open()
    import psycopg
    from psycopg.rows import dict_row

    url = os.environ["NEON_DATABASE_URL"]
    with psycopg.connect(url, row_factory=dict_row) as conn:
        fila = conn.execute(
            "SELECT name, aliases FROM master_ingredients WHERE name = %s", (FILA,)
        ).fetchone()
        if fila is None:
            print(f"ERROR: no existe la fila {FILA!r}")
            return 2
        actuales = list(fila["aliases"] or [])
        print(f"{FILA}: alias actuales = {actuales}")

        # Un alias que ya reclama OTRO alimento no se añade: repartir una identidad ambigua es
        # peor que no tenerla (misma regla que aplica `build_culinary_index`).
        ajenos = conn.execute(
            "SELECT name, aliases FROM master_ingredients WHERE name <> %s", (FILA,)
        ).fetchall()
        ocupados = {}
        for r in ajenos:
            for a in (r["aliases"] or []):
                ocupados.setdefault(_sa(a), r["name"])
            ocupados.setdefault(_sa(r["name"]), r["name"])

        a_anadir = []
        for nuevo in NUEVOS:
            if _sa(nuevo) in {_sa(a) for a in actuales}:
                print(f"   ya presente: {nuevo!r}")
                continue
            duenio = ocupados.get(_sa(nuevo))
            if duenio:
                print(f"   COLISIÓN: {nuevo!r} ya lo reclama {duenio!r} — no se añade")
                continue
            a_anadir.append(nuevo)

        if not a_anadir:
            print("nada que añadir.")
            return 0
        print(f"se añadirían: {a_anadir}")

        if not APLICAR:
            print("")
            print("SIMULACIÓN. Nada escrito. Añade --aplicar para escribir.")
            return 0

        with conn.cursor() as cur:
            cur.execute(
                "UPDATE master_ingredients SET aliases = %s WHERE name = %s",
                (actuales + a_anadir, FILA),
            )
            print(f"   actualizado {FILA}: {cur.rowcount} fila(s)")
        conn.commit()

    print("")
    print("ESCRITO. Regenera el baseline del corpus DO y revisa el diff:")
    print("   python scripts/gen_do_corpus_retarget_baseline_2026_08_18.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
