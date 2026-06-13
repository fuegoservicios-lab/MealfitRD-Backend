"""[P1-NEON-DB-MIGRATION · 2026-06-12] Migración repetible Supabase → Neon.

Pipeline: dump (pg_dump) → clean (strip Supabase-only artifacts) → restore
(psql, single-transaction) → verify (row counts por tabla en ambos lados).

Arquitectura híbrida decidida 2026-06-12: los DATOS Postgres viven en Neon;
Supabase CONSERVA Auth (JWT `supabase.auth.get_user`) + Storage. Por eso el
schema `auth` NO existe en Neon y todo lo que dependa de él se elimina del
dump (las FKs a `auth.users` se reemplazan por integridad a nivel de
aplicación — el backend siempre filtra `AND user_id = %s`, invariante I2).

Qué elimina la fase clean (y por qué):
  1. `CREATE POLICY` (75) + `ENABLE/FORCE ROW LEVEL SECURITY` (64):
     RLS era la defensa para clientes PostgREST; en Neon el único cliente
     es el backend (service-role semantics). El frontend deja de hablar
     directo con la DB (tarea P1-NEON #14).
  2. `ALTER TABLE ... REFERENCES auth.users` (17 FKs): `auth.users` no
     existe en Neon. El borrado-en-cascada se reimplementa app-side si
     algún día se borra un usuario (operación manual hoy).
  3. Funciones con dependencia de `auth.uid()`/`auth.users`:
     `handle_new_user` (trigger de auth, reemplazado por ensure-profile
     backend), `increment_inventory_quantity` y `update_health_profile_merge`
     (RPCs del frontend, reemplazadas por endpoints backend en #14).
     Sus `COMMENT ON FUNCTION` también se eliminan.

Qué NO toca: el resto de funciones (qualified refs, SET search_path ''),
triggers sobre tablas public, índices HNSW, COMMENT ON INDEX (anclas de
advisors), secuencias, datos (COPY). El schema `extensions` debe existir
en Neon ANTES del restore (vector, uuid-ossp, pgcrypto) — ver
`ensure_neon_extensions()`.

Uso:
    python scripts/migrate_db_to_neon.py                # pipeline completo
    python scripts/migrate_db_to_neon.py --skip-dump    # re-usa dump previo
    python scripts/migrate_db_to_neon.py --only-clean   # dump+clean, sin tocar Neon
    python scripts/migrate_db_to_neon.py --verify-only  # solo comparar row counts

Credenciales: lee `SUPABASE_DB_URL` y `NEON_DATABASE_URL` (endpoint DIRECTO,
no -pooler, para el restore) desde el entorno o desde backend/.env.
Binarios: `PGTOOLS_BIN` apunta al dir con pg_dump/psql (default: env conda
pgtools). pg_dump DEBE ser >= major de PG del servidor Supabase (17).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_WORKDIR = Path(r"C:\tmp\neon_migration")
_DEFAULT_PGTOOLS = Path(r"C:\Users\angel\miniconda3\envs\pgtools\Library\bin")

# Funciones cuyo cuerpo depende del schema `auth` (no existe en Neon).
_AUTH_DEPENDENT_FUNCTIONS = (
    "handle_new_user",
    "increment_inventory_quantity",
    "update_health_profile_merge",
)

_DOLLAR_TAG_RE = re.compile(r"\$[A-Za-z_0-9]*\$")
_FUNC_NAMES_RE = "|".join(_AUTH_DEPENDENT_FUNCTIONS)
_KILL_HEAD_PATTERNS = (
    re.compile(r"^CREATE POLICY\s"),
    re.compile(r"^COMMENT ON POLICY\s"),
    re.compile(r"^ALTER TABLE .* (ENABLE|FORCE) ROW LEVEL SECURITY;$"),
    re.compile(
        rf"^CREATE (OR REPLACE )?FUNCTION public\.({_FUNC_NAMES_RE})\("
    ),
    re.compile(rf"^COMMENT ON FUNCTION public\.({_FUNC_NAMES_RE})[(\s]"),
    re.compile(r"^COMMENT ON SCHEMA public\b"),
    re.compile(r"^CREATE SCHEMA public;$"),  # ya existe en Neon
)


def _load_env_file(path: Path) -> dict[str, str]:
    """Parser minimalista de .env (sin dependencia python-dotenv)."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def _resolve_url(name: str, cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    if os.environ.get(name):
        return os.environ[name]
    env_file = _load_env_file(_BACKEND_ROOT / ".env")
    if env_file.get(name):
        return env_file[name]
    raise SystemExit(
        f"[FATAL] No se encontró {name} (CLI, entorno, ni backend/.env)."
    )


def _bin(pgtools: Path, exe: str) -> str:
    candidate = pgtools / f"{exe}.exe"
    return str(candidate) if candidate.exists() else exe


def run_dump(supabase_url: str, dump_path: Path, pgtools: Path) -> None:
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _bin(pgtools, "pg_dump"),
        f"--dbname={supabase_url}",
        "--schema=public",
        "--no-owner",
        "--no-privileges",
        "--no-publications",
        "--no-subscriptions",
        "--no-security-labels",
        f"--file={dump_path}",
    ]
    print(f"[dump] pg_dump -> {dump_path}")
    env = dict(os.environ, PGCONNECT_TIMEOUT="30")
    subprocess.run(cmd, check=True, env=env)
    size_mb = dump_path.stat().st_size / 1024 / 1024
    print(f"[dump] OK ({size_mb:.2f} MB)")


def _statement_is_complete(joined: str, last_line: str) -> bool:
    """Un statement cierra cuando la última línea termina en `;`, no queda
    dollar-quote abierto y la paridad de comillas simples (fuera de regiones
    dollar-quoted, donde los apóstrofes NO van doblados) es par."""
    if not last_line.rstrip().endswith(";"):
        return False
    stripped = re.sub(
        r"\$([A-Za-z_0-9]*)\$.*?\$\1\$", "", joined, flags=re.DOTALL
    )
    if _DOLLAR_TAG_RE.search(stripped):
        return False  # dollar-quote sin cerrar
    return stripped.count("'") % 2 == 0


def _iter_statements(lines: list[str]):
    """Agrupa líneas en statements SQL (respeta dollar-quoting, strings y
    bloques de datos COPY).

    Los datos de `COPY ... FROM stdin;` van en passthrough crudo hasta el
    terminador `\\.` — las filas pueden contener apóstrofes/`;`/`$$` sin
    escaping SQL y romperían cualquier heurística de paridad.
    """
    buffer: list[str] = []
    in_copy = False
    for line in lines:
        if in_copy:
            yield ("passthrough", [line])
            if line == "\\.":
                in_copy = False
            continue
        if not buffer and (not line.strip() or line.lstrip().startswith("--")):
            yield ("passthrough", [line])
            continue
        buffer.append(line)
        if _statement_is_complete("\n".join(buffer), line):
            yield ("statement", buffer)
            if buffer[0].lstrip().startswith("COPY ") and line.rstrip().endswith(
                "FROM stdin;"
            ):
                in_copy = True
            buffer = []
    if buffer:
        yield ("statement", buffer)


def clean_dump(dump_path: Path, cleaned_path: Path) -> None:
    print(f"[clean] {dump_path} -> {cleaned_path}")
    lines = dump_path.read_text(encoding="utf-8").splitlines()
    kept: list[str] = []
    dropped = {
        "policy": 0, "rls": 0, "fk_auth_users": 0, "auth_fn": 0,
        "orphan_comment": 0, "other": 0,
    }

    # Pasada 1: clasificar statements y recolectar nombres de constraints
    # FK→auth.users dropeadas (sus COMMENT ON CONSTRAINT quedarían huérfanos).
    units = list(_iter_statements(lines))
    dropped_constraints: set[str] = set()
    for kind, chunk in units:
        if kind != "statement":
            continue
        head = chunk[0].lstrip()
        body = "\n".join(chunk)
        if head.startswith("ALTER TABLE") and "REFERENCES auth.users" in body:
            m = re.search(r"ADD CONSTRAINT (\w+)", body)
            if m:
                dropped_constraints.add(m.group(1))

    # Pasada 2: emitir lo que sobrevive.
    for kind, chunk in units:
        if kind == "passthrough":
            kept.extend(chunk)
            continue
        head = chunk[0].lstrip()
        body = "\n".join(chunk)
        killed = False
        for pat in _KILL_HEAD_PATTERNS:
            if pat.search(head):
                killed = True
                if "POLICY" in pat.pattern:
                    dropped["policy"] += 1
                elif "ROW LEVEL" in pat.pattern:
                    dropped["rls"] += 1
                elif "FUNCTION" in pat.pattern:
                    dropped["auth_fn"] += 1
                else:
                    dropped["other"] += 1
                break
        if not killed and head.startswith("ALTER TABLE") and "REFERENCES auth.users" in body:
            killed = True
            dropped["fk_auth_users"] += 1
        if not killed and head.startswith("COMMENT ON CONSTRAINT"):
            m = re.match(r"COMMENT ON CONSTRAINT (\w+) ON ", head)
            if m and m.group(1) in dropped_constraints:
                killed = True
                dropped["orphan_comment"] += 1
        if not killed:
            kept.append(body)

    cleaned = "\n".join(kept) + "\n"

    # Post-condiciones: nada operativo puede seguir refiriendo al schema auth.
    residual = [
        m for m in re.finditer(r"^(?!--)(?!COMMENT ON ).*\bauth\.", cleaned, re.MULTILINE)
        if "auth.uid() IS NULL" not in m.group(0)  # solo aparecería en fn dropeada
    ]
    # COMMENT ON ... strings pueden mencionar auth.users como texto; el resto no.
    problematic = [
        m.group(0).strip()[:120]
        for m in residual
        if not m.group(0).lstrip().startswith("'")  # continuación de string literal
    ]
    if problematic:
        for sample in problematic[:10]:
            print(f"  [clean][RESIDUAL] {sample}")
        raise SystemExit(
            f"[FATAL] {len(problematic)} referencias residuales a `auth.` fuera de "
            "COMMENT — revisar patrones de limpieza antes de restaurar."
        )
    # Anclados a inicio de línea: los COMMENT ON pueden mencionar estas
    # frases como TEXTO (e.g. '[P1-HIST-AUDIT-6] FORCE ROW LEVEL SECURITY:').
    for pat, label in (
        (r"^CREATE POLICY\s", "CREATE POLICY"),
        (r"^ALTER TABLE .* (ENABLE|FORCE) ROW LEVEL SECURITY;", "RLS"),
        (r"REFERENCES auth\.users", "FK auth.users"),
        (rf"^CREATE (OR REPLACE )?FUNCTION public\.({_FUNC_NAMES_RE})\(", "fn auth-dep"),
    ):
        if re.search(pat, cleaned, re.MULTILINE):
            raise SystemExit(f"[FATAL] Post-condición fallida: queda {label} en cleaned.")

    cleaned_path.write_text(cleaned, encoding="utf-8")
    print(f"[clean] dropped: {dropped}")
    print(f"[clean] OK ({cleaned_path.stat().st_size / 1024 / 1024:.2f} MB)")


def ensure_neon_extensions(neon_url: str, pgtools: Path) -> None:
    """Idempotente: schema extensions + vector/uuid-ossp/pgcrypto (layout Supabase)."""
    statements = [
        "CREATE SCHEMA IF NOT EXISTS extensions;",
        "CREATE EXTENSION IF NOT EXISTS vector SCHEMA extensions;",
        'CREATE EXTENSION IF NOT EXISTS "uuid-ossp" SCHEMA extensions;',
        "CREATE EXTENSION IF NOT EXISTS pgcrypto SCHEMA extensions;",
    ]
    cmd = [_bin(pgtools, "psql"), neon_url, "-v", "ON_ERROR_STOP=1"]
    for st in statements:
        cmd.extend(["-c", st])
    print("[extensions] asegurando schema extensions + vector/uuid-ossp/pgcrypto")
    subprocess.run(cmd, check=True)


def reset_neon_public(neon_url: str, pgtools: Path) -> None:
    """Re-sync de cutover: tumba el schema public de NEON (jamás Supabase)
    para restaurar fresco. Pedir confirmación interactiva NO aplica: el
    operador opta-in explícito con --reset-neon."""
    print("[reset] DROP SCHEMA public CASCADE en Neon (re-sync)")
    subprocess.run(
        [
            _bin(pgtools, "psql"), neon_url, "-v", "ON_ERROR_STOP=1",
            "-c", "DROP SCHEMA IF EXISTS public CASCADE;",
            "-c", "CREATE SCHEMA public;",
        ],
        check=True,
    )


def run_restore(neon_url: str, cleaned_path: Path, pgtools: Path) -> None:
    ensure_neon_extensions(neon_url, pgtools)
    cmd = [
        _bin(pgtools, "psql"),
        neon_url,
        "-v", "ON_ERROR_STOP=1",
        "--single-transaction",
        "-q",
        "-f", str(cleaned_path),
    ]
    print(f"[restore] psql --single-transaction -f {cleaned_path}")
    env = dict(os.environ, PGCONNECT_TIMEOUT="30")
    subprocess.run(cmd, check=True, env=env)
    print("[restore] OK")


def _table_counts(url: str, pgtools: Path) -> dict[str, int]:
    list_sql = (
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='public' AND table_type='BASE TABLE' ORDER BY 1;"
    )
    out = subprocess.run(
        [_bin(pgtools, "psql"), url, "-tA", "-c", list_sql],
        check=True, capture_output=True, text=True,
    ).stdout
    tables = [t for t in out.splitlines() if t.strip()]
    if not tables:
        return {}
    union = " UNION ALL ".join(
        f"SELECT '{t}', count(*) FROM public.\"{t}\"" for t in tables
    )
    out = subprocess.run(
        [_bin(pgtools, "psql"), url, "-tA", "-F", "|", "-c", union],
        check=True, capture_output=True, text=True,
    ).stdout
    counts: dict[str, int] = {}
    for row in out.splitlines():
        if "|" in row:
            name, _, n = row.partition("|")
            counts[name] = int(n)
    return counts


def run_verify(supabase_url: str, neon_url: str, pgtools: Path) -> bool:
    print("[verify] comparando row counts public.* (Supabase vs Neon)")
    supa = _table_counts(supabase_url, pgtools)
    neon = _table_counts(neon_url, pgtools)
    all_tables = sorted(set(supa) | set(neon))
    ok = True
    print(f"{'tabla':<42} {'supabase':>10} {'neon':>10}  estado")
    for t in all_tables:
        s, n = supa.get(t, -1), neon.get(t, -1)
        status = "OK" if s == n else "** MISMATCH **"
        if s != n:
            ok = False
        print(f"{t:<42} {s:>10} {n:>10}  {status}")
    print(f"[verify] {'TODO OK' if ok else 'HAY MISMATCHES'} — {len(all_tables)} tablas")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--supabase-url", default=None)
    ap.add_argument("--neon-url", default=None, help="endpoint DIRECTO (no -pooler)")
    ap.add_argument("--workdir", type=Path, default=_DEFAULT_WORKDIR)
    ap.add_argument("--pgtools", type=Path,
                    default=Path(os.environ.get("PGTOOLS_BIN", _DEFAULT_PGTOOLS)))
    ap.add_argument("--skip-dump", action="store_true")
    ap.add_argument("--only-clean", action="store_true",
                    help="dump+clean sin tocar Neon")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--reset-neon", action="store_true",
                    help="DROP SCHEMA public CASCADE en Neon antes del "
                         "restore (re-sync de cutover)")
    args = ap.parse_args()

    supabase_url = _resolve_url("SUPABASE_DB_URL", args.supabase_url)
    dump_path = args.workdir / "supabase_full_dump.sql"
    cleaned_path = args.workdir / "supabase_dump_neon_clean.sql"

    if args.verify_only:
        neon_url = _resolve_url("NEON_DATABASE_URL", args.neon_url)
        sys.exit(0 if run_verify(supabase_url, neon_url, args.pgtools) else 1)

    if not args.skip_dump:
        run_dump(supabase_url, dump_path, args.pgtools)
    elif not dump_path.exists():
        raise SystemExit(f"[FATAL] --skip-dump pero no existe {dump_path}")

    clean_dump(dump_path, cleaned_path)

    if args.only_clean:
        print("[done] --only-clean: restore omitido.")
        return

    neon_url = _resolve_url("NEON_DATABASE_URL", args.neon_url)
    if args.reset_neon:
        reset_neon_public(neon_url, args.pgtools)
    run_restore(neon_url, cleaned_path, args.pgtools)
    ok = run_verify(supabase_url, neon_url, args.pgtools)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
