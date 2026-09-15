"""[P1-CHAT-ORPHAN-SESSIONS · 2026-09-14] Purga one-shot de checkpoints de LangGraph huérfanos.

Huérfano = hilo cuyo `thread_id` no existe en `agent_sessions` y cuyo ÚLTIMO checkpoint tiene
más de N días. Mismo predicado (`db_chat.ORPHAN_CHECKPOINT_THREADS_SQL`) y mismo borrado
(`db_chat.delete_checkpoints_for_threads`) que el barrido diario: el script NO tiene SQL propio
de selección ni de borrado, así que no puede divergir del cron.

Uso (desde backend/):
    python scripts/purge_orphan_chat_checkpoints.py                 # DRY-RUN: solo SELECT
    python scripts/purge_orphan_chat_checkpoints.py --apply         # borra, en UNA transacción

Opciones: --min-age-days N (7; mínimo 1), --limit N (1000), --env-file RUTA (.env a cargar).

Salida a stdout a propósito: es una herramienta CLI one-shot, no código de producción.
"""
from __future__ import annotations

import argparse
import os
import sys

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

# Tamaño aproximado por hilo (pg_column_size de la fila: comprimido si está en TOAST).
_SIZE_SQL = """
    SELECT thread_id, sum(sz)::bigint AS bytes, sum(n_ck)::int AS n_ck
    FROM (
        SELECT thread_id, pg_column_size(t.*) AS sz, 1 AS n_ck
        FROM public.checkpoints t WHERE thread_id = ANY(%s::text[])
        UNION ALL
        SELECT thread_id, pg_column_size(t.*), 0
        FROM public.checkpoint_blobs t WHERE thread_id = ANY(%s::text[])
        UNION ALL
        SELECT thread_id, pg_column_size(t.*), 0
        FROM public.checkpoint_writes t WHERE thread_id = ANY(%s::text[])
    ) x
    GROUP BY thread_id
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Purga checkpoints de LangGraph sin sesión (dry-run por defecto).")
    ap.add_argument("--apply", action="store_true", help="Borra de verdad (por defecto solo lista).")
    ap.add_argument("--min-age-days", type=int, default=7, help="Edad mínima del último checkpoint (días, >=1).")
    ap.add_argument("--limit", type=int, default=1000, help="Máximo de hilos a tratar.")
    ap.add_argument("--env-file", default=None, help="Ruta del .env a cargar (por defecto, búsqueda estándar).")
    args = ap.parse_args(argv)

    from dotenv import load_dotenv

    if args.env_file:
        load_dotenv(args.env_file)
    else:
        load_dotenv()
    url = os.environ.get("NEON_DATABASE_URL")
    if not url:
        print("NEON_DATABASE_URL no está definida (usa --env-file).")
        return 2

    import psycopg
    from psycopg.rows import dict_row

    # Import tardío: db_core lee las URLs de Neon al importarse (el pool nace cerrado).
    from db_chat import ORPHAN_CHECKPOINT_THREADS_SQL, delete_checkpoints_for_threads

    min_age = max(1, int(args.min_age_days))
    limit = max(1, int(args.limit))

    with psycopg.connect(url, autocommit=True, row_factory=dict_row) as conn:
        conn.execute("SET statement_timeout = '20s'")
        if not args.apply:
            # Cada sentencia en autocommit es su propia transacción: todas quedan READ ONLY.
            conn.execute("SET default_transaction_read_only = on")

        rows = conn.execute(ORPHAN_CHECKPOINT_THREADS_SQL, (min_age, limit)).fetchall()
        ids = [r["thread_id"] for r in rows]
        sizes = {}
        if ids:
            sizes = {r["thread_id"]: r for r in conn.execute(_SIZE_SQL, (ids, ids, ids)).fetchall()}

        modo = "APPLY" if args.apply else "DRY-RUN"
        print(f"[{modo}] Hilos de checkpoint sin sesión y con último checkpoint > {min_age} d: {len(ids)}")
        total = 0
        for r in rows:
            tid = r["thread_id"]
            s = sizes.get(tid) or {}
            b = int(s.get("bytes") or 0)
            total += b
            ts = r["last_ts"].strftime("%Y-%m-%d %H:%M") if r.get("last_ts") else "?"
            print(f"  {tid[:8]}  último={ts}  checkpoints={int(s.get('n_ck') or 0):>3}  {b / 1024:8.1f} KiB")
        print(f"Total aproximado: {total / (1024 * 1024):.2f} MiB en {len(ids)} hilo(s).")

        if not args.apply:
            print("DRY-RUN: no se borró nada. Repite con --apply para borrar.")
            return 0
        if not ids:
            print("Nada que borrar.")
            return 0

        with conn.transaction():
            with conn.cursor() as cur:
                # Re-selección DENTRO de la transacción: lo que se borra es lo que es huérfano AHORA.
                cur.execute(ORPHAN_CHECKPOINT_THREADS_SQL, (min_age, limit))
                ids_tx = [r["thread_id"] for r in cur.fetchall()]
                counts = delete_checkpoints_for_threads(cur, ids_tx)
        print(f"APPLY: {len(ids_tx)} hilo(s) borrados. Filas por tabla: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
