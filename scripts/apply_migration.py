"""[P1-MIGRATION-RUNNER · 2026-08-07] Aplica un archivo de `migrations/` a Neon.

No había runner genérico: cada script de `scripts/` abría su propia conexión, y
las migraciones se aplicaban a mano. Pegar DDL en una consola es justo donde se
cuelan los errores que esta capa debería impedir.

    # Ver qué haría, sin tocar nada (default):
    python backend/scripts/apply_migration.py migrations/p1_consumption_ledger_2026_08_07.sql

    # Aplicar de verdad:
    python backend/scripts/apply_migration.py migrations/p1_consumption_ledger_2026_08_07.sql --apply

Tres decisiones que no son obvias:

1. **autocommit=True.** Las migraciones de este repo traen su propio `BEGIN;` …
   `COMMIT;`. psycopg3 abre una transacción implícita al ejecutar, así que sin
   autocommit el `BEGIN;` del archivo cae dentro de otra transacción y Postgres
   emite `WARNING: there is already a transaction in progress`. Con autocommit,
   el archivo manda: o entra entero, o no entra nada.

2. **URL DIRECTA, no la pooled.** El resto de scripts prefiere
   `NEON_DATABASE_URL_POOLED`, y para queries está bien. Para DDL no: PgBouncer
   en modo transaction no garantiza que todo el script caiga en la misma sesión,
   y los `DO $$ … $$` de sanity dependen de eso. Aquí la precedencia se invierte
   a propósito.

3. **Dry-run por default.** Escribe en la base de producción. Que aplicar exija
   `--apply` explícito es barato y evita el accidente de una flecha-arriba.

Idempotencia: las migraciones del repo la garantizan por convención
(P3-MIGRATION-IDEMPOTENCE-DOC). Re-correr una ya aplicada debe ser un no-op; si
revienta, el bug está en la migración, no aquí.
"""
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass

import psycopg


def _mask(url: str) -> str:
    """`postgres://user:pass@host/db` -> `host/db`. No imprimir credenciales."""
    tail = url.rsplit("@", 1)[-1]
    return tail.split("?", 1)[0]


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    apply = "--apply" in sys.argv[1:]

    if len(args) != 1:
        print(__doc__)
        return 2

    path = args[0]
    if not os.path.isfile(path):
        print(f"[X] No existe: {path}")
        return 1

    sql = open(path, encoding="utf-8").read()

    # DIRECTA primero (ver decisión 2 en el docstring).
    url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("NEON_DATABASE_URL_POOLED")
    if not url:
        print("[X] Falta NEON_DATABASE_URL en backend/.env")
        return 1

    stmts = [ln.strip() for ln in sql.splitlines()
             if ln.strip() and not ln.strip().startswith("--")]
    print(f"Migracion : {path}")
    print(f"Destino   : {_mask(url)}")
    print(f"Lineas SQL: {len(stmts)} (comentarios excluidos)")

    if not apply:
        print("\n[dry-run] Nada aplicado. Repite con --apply para ejecutar.")
        return 0

    with psycopg.connect(url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)

    print("\n[OK] Aplicada. Los DO $$ de sanity pasaron (si no, habrian lanzado).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
