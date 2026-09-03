"""Verifica, en modo forzosamente read-only, las dos migraciones de país.

Uso desde ``backend/``::

    python scripts/verify_country_schema.py

Retorna 0 únicamente cuando existen el CHECK y el índice de country y las 13
filas volumétricas ya tienen densidad. No aplica ni corrige nada: si retorna 1,
el dueño ejecuta primero ``p1_country_keep_density_beta_2026_08_21.sql`` y
después ``p3_country_db_check_2026_08_22.sql`` con el runner de migraciones.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass

import psycopg


COUNTRY_CODES = ("DO", "ES", "US", "MX", "PR", "CO")
COUNTRY_CONSTRAINT = "user_profiles_country_supported"
COUNTRY_INDEX = "idx_user_profiles_country"
DENSITY_NAMES = (
    "Nata",
    "Crema agria",
    "Crema mexicana",
    "Crema mitad y mitad",
    "Suero de mantequilla",
    "Suero costeño",
    "Natilla",
    "Arequipe",
    "Jarabe de arce",
    "Aceite de achiote",
    "Salsa barbacoa",
    "Salsa de salchicha",
    "Hummus",
)


def evaluate_country_schema(
    constraint_rows: Iterable[tuple],
    index_rows: Iterable[tuple],
    density_rows: Iterable[tuple],
) -> tuple[list[str], dict]:
    """Evalúa resultados SQL sin I/O para que el detector sea unit-testable."""
    constraints = {str(name): str(definition) for name, definition in constraint_rows}
    indexes = {str(row[0]) for row in index_rows}
    densities = {str(name): value for name, value in density_rows}

    failures: list[str] = []
    constraint_definition = constraints.get(COUNTRY_CONSTRAINT)
    if constraint_definition is None:
        failures.append(f"falta el CHECK {COUNTRY_CONSTRAINT}")
    else:
        missing_codes = [code for code in COUNTRY_CODES if f"'{code}'" not in constraint_definition]
        if missing_codes:
            failures.append(
                f"el CHECK {COUNTRY_CONSTRAINT} no contiene {missing_codes}"
            )

    if COUNTRY_INDEX not in indexes:
        failures.append(f"falta el índice {COUNTRY_INDEX}")

    missing_rows = [name for name in DENSITY_NAMES if name not in densities]
    null_rows = [name for name in DENSITY_NAMES if name in densities and densities[name] is None]
    if missing_rows:
        failures.append(f"faltan {len(missing_rows)} filas del lote de densidad: {missing_rows}")
    if null_rows:
        failures.append(f"quedan {len(null_rows)} densidades NULL: {null_rows}")

    report = {
        "constraint": constraint_definition is not None,
        "index": COUNTRY_INDEX in indexes,
        "density_rows_found": len(densities),
        "density_rows_expected": len(DENSITY_NAMES),
        "density_null": null_rows,
        "density_missing": missing_rows,
    }
    return failures, report


def _mask(url: str) -> str:
    return url.rsplit("@", 1)[-1].split("?", 1)[0]


def main() -> int:
    url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("NEON_DATABASE_URL_POOLED")
    if not url:
        print("[X] Falta NEON_DATABASE_URL en backend/.env")
        return 2

    # [P2-COUNTRY-MIGRACIONES-SIN-APLICAR · 2026-08-23] Defensa del
    # detector: aunque una edición futura añada SQL por error, Postgres rechaza
    # cualquier escritura en esta conexión.
    with psycopg.connect(
        url,
        options="-c default_transaction_read_only=on",
    ) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT con.conname, pg_get_constraintdef(con.oid)
              FROM pg_constraint con
              JOIN pg_namespace ns ON ns.oid = con.connamespace
             WHERE ns.nspname = 'public'
               AND con.conrelid = 'public.user_profiles'::regclass
               AND con.conname = %s
            """,
            (COUNTRY_CONSTRAINT,),
        )
        constraint_rows = cur.fetchall()

        cur.execute(
            """
            SELECT indexname
              FROM pg_indexes
             WHERE schemaname = 'public'
               AND tablename = 'user_profiles'
               AND indexname = %s
            """,
            (COUNTRY_INDEX,),
        )
        index_rows = cur.fetchall()

        cur.execute(
            """
            SELECT name, density_g_per_cup::float8
              FROM public.master_ingredients
             WHERE name = ANY(%s)
             ORDER BY name
            """,
            (list(DENSITY_NAMES),),
        )
        density_rows = cur.fetchall()

    failures, report = evaluate_country_schema(
        constraint_rows,
        index_rows,
        density_rows,
    )
    print(f"Destino read-only: {_mask(url)}")
    print(
        f"CHECK {COUNTRY_CONSTRAINT}: "
        f"{'OK' if report['constraint'] else 'FALTA'}"
    )
    print(f"Índice {COUNTRY_INDEX}: {'OK' if report['index'] else 'FALTA'}")
    print(
        "Densidades país: "
        f"{report['density_rows_found']}/{report['density_rows_expected']} filas, "
        f"{len(report['density_null'])} NULL"
    )
    if failures:
        print("RESULTADO: esquema de país incompleto")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("RESULTADO: las dos migraciones de país están aplicadas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
