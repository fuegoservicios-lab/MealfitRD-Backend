"""[P1-CONSUMPTION-LEDGER · 2026-08-07] Verifica en la base REAL lo que la
migración dejó, y de paso contesta la pregunta abierta sobre `updated_at`.

    python backend/scripts/check_p1_ledger.py

Por qué existe: `apply_migration.py` imprime "[OK] Aplicada" cuando el DDL corrió
y los `DO $$` de sanity no lanzaron. Eso NO es lo mismo que "la Nevera ya
devuelve comida". Este script mira la forma final contra lo que el código de
producción asume, que es donde aparecen los desajustes silenciosos.

Cuatro cosas, en orden de lo que rompe primero:

1. La tabla del ledger y sus dos índices. Sin el parcial de `consumed_meal_id`,
   "Deshacer registro" hace seq-scan sobre todo el ledger.
2. Los CHECK de `outcome` y `source` con los valores que P1-PANTRY-RECONCILIATION
   añadió (`spoiled`, `reconciliation`). Si faltan, el banner revienta al
   escribir y el fallo aparece recién cuando un usuario contesta.
3. El FK de `user_id`. Debe apuntar a `public.user_profiles`, NO a `auth.users`
   (schema de Supabase, eliminado en P1-NEON-DB-MIGRATION).
4. `user_inventory.updated_at`: ¿hay trigger BEFORE UPDATE? Es la pregunta que
   quedó abierta en el PR #8 y que no se puede contestar leyendo el repo, porque
   un trigger creado a mano en la consola no deja rastro en `migrations/`.
"""
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass

import psycopg

TABLA = "inventory_consumption_events"


def main() -> int:
    url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("NEON_DATABASE_URL_POOLED")
    if not url:
        print("[X] Falta NEON_DATABASE_URL en backend/.env")
        return 1

    fallos = []
    with psycopg.connect(url) as conn, conn.cursor() as cur:

        # 1. Tabla + indices
        cur.execute(
            "SELECT indexname FROM pg_indexes "
            "WHERE schemaname='public' AND tablename=%s ORDER BY indexname", (TABLA,))
        idx = [r[0] for r in cur.fetchall()]
        if not idx:
            print(f"[X] La tabla {TABLA} no existe (o no tiene indices).")
            return 1
        print(f"Tabla {TABLA}: OK")
        for esperado in ("idx_ice_consumed_meal_pending", "idx_ice_user_created_at_desc"):
            ok = esperado in idx
            print(f"  indice {esperado:<32} {'OK' if ok else 'FALTA'}")
            if not ok:
                fallos.append(f"falta el indice {esperado}")

        # 2. CHECKs con los valores de la reconciliacion
        cur.execute(
            "SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE conrelid = %s::regclass AND contype = 'c'", (f"public.{TABLA}",))
        checks = dict(cur.fetchall())
        defs = " ".join(checks.values())
        print("\nCHECKs:")
        for valor, quien in (("spoiled", "outcome"), ("reconciliation", "source")):
            ok = valor in defs
            print(f"  {quien:<8} admite {valor:<16} {'OK' if ok else 'FALTA'}")
            if not ok:
                fallos.append(f"el CHECK de {quien} no admite '{valor}' "
                              f"(falta p1_pantry_reconciliation)")

        # 3. FK de user_id -> user_profiles, no auth.users
        cur.execute(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE conrelid = %s::regclass AND contype = 'f'", (f"public.{TABLA}",))
        fks = " ".join(r[0] for r in cur.fetchall())
        ok = "user_profiles" in fks
        print(f"\nFK user_id -> user_profiles          {'OK' if ok else 'MAL: ' + fks}")
        if not ok:
            fallos.append("el FK de user_id no apunta a public.user_profiles")

        # 4. La pregunta abierta del PR #8
        cur.execute(
            "SELECT tgname FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
            "WHERE c.relname = 'user_inventory' AND NOT t.tgisinternal")
        trg = [r[0] for r in cur.fetchall()]
        print("\nuser_inventory, triggers no-internos:", trg or "NINGUNO")
        if trg:
            print("  -> `updated_at` SI se mantiene sola. Mi lectura del PR #8 estaba mal:")
            print("     get_inventory_activity_since y el ancla del plan-freeze estan bien.")
        else:
            print("  -> CONFIRMADO lo reportado en el PR #8: nada mantiene `updated_at`.")
            print("     El RPC apply_inventory_delta no la toca, asi que las filas de")
            print("     consumo no pasan el propio WHERE de get_inventory_activity_since,")
            print("     y el ancla de gracia del plan-freeze ve inactivo a quien come.")

    print()
    if fallos:
        print("RESULTADO: hay que revisar ->")
        for f in fallos:
            print(f"  - {f}")
        return 1
    print("RESULTADO: el esquema del ledger esta como el codigo lo asume.")
    print("Falta la prueba de verdad, que es de UI: 'Me lo comi' -> la Nevera baja")
    print("-> 'Deshacer registro' -> la Nevera vuelve a subir.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
