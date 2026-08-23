"""[P1-MIGRATION-RUNNER · 2026-08-07] Aplica un archivo de `migrations/` a Neon.

No había runner genérico: cada script de `scripts/` abría su propia conexión, y
las migraciones se aplicaban a mano. Pegar DDL en una consola es justo donde se
cuelan los errores que esta capa debería impedir.

    # Ver qué haría, sin tocar nada (default):
    python backend/scripts/apply_migration.py migrations/p1_consumption_ledger_2026_08_07.sql

    # Aplicar de verdad (y anotarla en el libro):
    python backend/scripts/apply_migration.py migrations/p1_consumption_ledger_2026_08_07.sql --apply

    # [P2-I18N-MIGRACIONES-SIN-LIBRO · 2026-08-23] El libro:
    python backend/scripts/apply_migration.py --status            # ficheros vs schema_migrations
    python backend/scripts/apply_migration.py migrations/x.sql --record --note "aplicada a mano el 22-ago"

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

[P2-I18N-MIGRACIONES-SIN-LIBRO · 2026-08-23] El libro (`public.schema_migrations`).
Hasta hoy este runner ejecutaba y no dejaba rastro: «¿está aplicada?» era una auditoría a
mano contra `information_schema`. Medido con esa auditoría: 110 ficheros y UNO sin aplicar
en producción (`p3_country_db_check_2026_08_22.sql`) sin que nada lo dijera. Ahora:
  · `--apply` ejecuta Y anota (nombre, sha256 del fichero, quién). Si el libro no existe
    todavía, avisa y no falla: la migración que lo crea es
    `p2_i18n_migraciones_sin_libro_2026_08_23.sql`, y aplicarla con este mismo runner es
    la primera entrada del libro.
  · `--record` anota SIN ejecutar: para las 100+ que ya estaban aplicadas antes del libro
    (backfill, con `--note` diciendo cómo se verificó) y para las aplicadas a mano.
  · `--status` lista ficheros vs libro: aplicada / PENDIENTE / aplicada con OTRO contenido
    (el fichero cambió después: este repo los edita para añadir sanity checks).
"""
import hashlib
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass

import psycopg

_MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "migrations")


def _mask(url: str) -> str:
    """`postgres://user:pass@host/db` -> `host/db`. No imprimir credenciales."""
    tail = url.rsplit("@", 1)[-1]
    return tail.split("?", 1)[0]


def _checksum(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def _url() -> str | None:
    # DIRECTA primero (ver decisión 2 en el docstring).
    return os.environ.get("NEON_DATABASE_URL") or os.environ.get("NEON_DATABASE_URL_POOLED")


def _who() -> str:
    return os.environ.get("MEALFIT_MIGRATION_APPLIED_BY") or os.environ.get("USERNAME") or os.environ.get("USER") or "?"


# ---------------------------------------------------------------------------
# El libro
# ---------------------------------------------------------------------------

_LEDGER_UPSERT = """
INSERT INTO public.schema_migrations (name, checksum, applied_by, note)
VALUES (%s, %s, %s, %s)
ON CONFLICT (name) DO UPDATE
    SET checksum = EXCLUDED.checksum,
        applied_at = now(),
        applied_by = EXCLUDED.applied_by,
        note = COALESCE(EXCLUDED.note, public.schema_migrations.note)
"""


def _record(cur, name: str, checksum: str, note: str | None) -> bool:
    """Anota en el libro. `False` (y aviso) si el libro aún no existe: esa es la única
    razón aceptable para no anotar, y sólo mientras no se aplique la migración que lo crea."""
    try:
        cur.execute(_LEDGER_UPSERT, (name, checksum, _who(), note))
        return True
    except psycopg.errors.UndefinedTable:
        print(
            "[!] schema_migrations no existe todavía: NO se anotó. Aplica primero "
            "migrations/p2_i18n_migraciones_sin_libro_2026_08_23.sql con --apply."
        )
        return False


def clasificar(ficheros: dict, libro: dict) -> dict:
    """Pura, para poder probarla: `ficheros` {nombre: checksum} del disco, `libro`
    {nombre: checksum} de la tabla. Devuelve las cuatro listas ordenadas."""
    al_dia, pendientes, cambiadas, solo_en_libro = [], [], [], []
    for nombre in sorted(ficheros):
        if nombre not in libro:
            pendientes.append(nombre)
        elif libro[nombre] != ficheros[nombre]:
            cambiadas.append(nombre)
        else:
            al_dia.append(nombre)
    for nombre in sorted(libro):
        if nombre not in ficheros:
            solo_en_libro.append(nombre)
    return {
        "al_dia": al_dia, "pendientes": pendientes,
        "cambiadas": cambiadas, "solo_en_libro": solo_en_libro,
    }


def _ficheros_en_disco() -> dict:
    out = {}
    for f in sorted(os.listdir(_MIGRATIONS_DIR)):
        if f.endswith(".sql"):
            with open(os.path.join(_MIGRATIONS_DIR, f), encoding="utf-8") as fh:
                out[f] = _checksum(fh.read())
    return out


def status() -> int:
    url = _url()
    if not url:
        print("[X] Falta NEON_DATABASE_URL en backend/.env")
        return 1
    ficheros = _ficheros_en_disco()
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT name, checksum, COALESCE(note, '') FROM public.schema_migrations")
                filas = cur.fetchall()
                libro = {r[0]: r[1] for r in filas}
                # [P3-I18N-DOC-LIBRO-MEDIDO-VS-SUPUESTO · 2026-08-23] Las filas cuya nota dice
                # que el estado de la base NO se midió. El backfill del libro las anotaba como
                # «asumida aplicada», con la misma cara que las verificadas — y la primera que
                # se pudo medir (densidades beta) resultó falsa.
                sin_verificar = sorted(r[0] for r in filas if str(r[2]).startswith("SIN VERIFICAR"))
            except psycopg.errors.UndefinedTable:
                print(
                    f"[!] schema_migrations NO existe en {_mask(url)}: el libro no se ha creado. "
                    f"Aplica migrations/p2_i18n_migraciones_sin_libro_2026_08_23.sql con --apply y "
                    f"luego --record las ya aplicadas (con --note diciendo cómo se verificó)."
                )
                print(f"    {len(ficheros)} ficheros en migrations/, 0 en el libro.")
                return 3
    r = clasificar(ficheros, libro)
    print(f"Destino: {_mask(url)} · {len(ficheros)} ficheros · {len(libro)} en el libro")
    print(f"  al día                    : {len(r['al_dia'])}"
          f"   (de ellas {len(sin_verificar)} SIN VERIFICAR)")
    print(f"  PENDIENTES (sin fila)     : {len(r['pendientes'])}")
    for n in r["pendientes"]:
        print(f"      · {n}")
    print(f"  aplicadas con OTRO contenido: {len(r['cambiadas'])}")
    for n in r["cambiadas"]:
        print(f"      · {n}   (el fichero cambió tras aplicarse: revisa el diff y re-aplica o --record)")
    if r["solo_en_libro"]:
        print(f"  en el libro sin fichero   : {len(r['solo_en_libro'])}")
        for n in r["solo_en_libro"]:
            print(f"      · {n}")
    if sin_verificar:
        # No son un fallo: son honestidad. Se listan aparte para que «al día» no las tape.
        print(f"  SIN VERIFICAR (su nota lo dice): {len(sin_verificar)}")
        print("      migraciones de DATOS anotadas en el backfill sin medir el estado de la base.")
        print("      Una que no corrió NO rompe el producto: deja el dato mal en silencio.")
        for n in sin_verificar:
            print(f"      · {n}")
    # Exit ≠ 0 si hay pendientes: para que un gate pueda leerlo. Las SIN VERIFICAR no
    # cambian el exit — son deuda de evidencia, no trabajo pendiente conocido.
    return 4 if (r["pendientes"] or r["cambiadas"]) else 0


# ---------------------------------------------------------------------------
# Aplicar / anotar un fichero
# ---------------------------------------------------------------------------

def main() -> int:
    argv = sys.argv[1:]
    if "--status" in argv:
        return status()
    args = [a for a in argv if not a.startswith("-")]
    apply = "--apply" in argv
    record = "--record" in argv
    note = None
    if "--note" in argv:
        i = argv.index("--note")
        if i + 1 < len(argv):
            note = argv[i + 1]
            if note in args:
                args.remove(note)
    if len(args) != 1:
        print(__doc__)
        return 2
    path = args[0]
    if not os.path.isfile(path):
        print(f"[X] No existe: {path}")
        return 1

    sql = open(path, encoding="utf-8").read()
    name = os.path.basename(path)
    checksum = _checksum(sql)

    url = _url()
    if not url:
        print("[X] Falta NEON_DATABASE_URL en backend/.env")
        return 1

    stmts = [ln.strip() for ln in sql.splitlines()
             if ln.strip() and not ln.strip().startswith("--")]
    print(f"Migracion : {path}")
    print(f"Destino   : {_mask(url)}")
    print(f"Lineas SQL: {len(stmts)} (comentarios excluidos)")
    print(f"sha256    : {checksum[:12]}…")

    if record and not apply:
        with psycopg.connect(url, autocommit=True) as conn:
            with conn.cursor() as cur:
                ok = _record(cur, name, checksum, note or "anotada con --record (sin ejecutar)")
        print("\n[OK] Anotada en el libro SIN ejecutar." if ok else "\n[X] No se anotó.")
        return 0 if ok else 3

    if not apply:
        print("\n[dry-run] Nada aplicado. Repite con --apply para ejecutar.")
        return 0

    with psycopg.connect(url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            anotada = _record(cur, name, checksum, note)

    print("\n[OK] Aplicada. Los DO $$ de sanity pasaron (si no, habrian lanzado).")
    print("[OK] Anotada en schema_migrations." if anotada else "[!] Aplicada pero NO anotada (libro ausente).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
