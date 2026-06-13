# P1-NEON-DB-MIGRATION · 2026-06-12 — Migración de datos Supabase → Neon

## Arquitectura híbrida (decisión)

| Servicio | Dónde vive | Razón |
|---|---|---|
| **Postgres (datos)** | **Neon** — proyecto `Mealfitrd` (`long-recipe-82771378`), branch `production`, PG 18, us-east-1 | Connection pools sin límite de Supavisor free-tier, autoscaling, branching para staging |
| **Auth (JWT)** | **Supabase** (sin cambios) | `supabase.auth.get_user` es la única capa de auth (P0-AUDIT-1); migrarla sería otro proyecto |
| **Storage** (visual diary) | **Supabase** (sin cambios) | `supabase.storage.*` en `db_profiles._purge_visual_diary_storage` y diary uploads |
| **Realtime** | **Eliminado** | Frontend migrado a refetch on focus/visibilitychange + polling acotado (ver abajo) |

Consecuencia estructural: **el frontend ya no habla con ninguna DB** — todo dato
pasa por endpoints backend ([`routers/user_data.py`](../routers/user_data.py) +
endpoints preexistentes). PostgREST quedó prohibido en backend Y frontend
(tests blanket: [`test_p1_neon_db_migration.py`](../tests/test_p1_neon_db_migration.py)).

## Selección de backend de datos (knob)

`MEALFIT_DB_BACKEND` en [`db_core.py`](../db_core.py): `supabase` (default) | `neon`.

- `neon`: `NEON_DATABASE_URL_POOLED` (PgBouncer transaction-mode) → `connection_pool` +
  `async_connection_pool`; `NEON_DATABASE_URL` (endpoint directo, session-mode) →
  `chat_checkpoint_pool` (mismo contrato P1-CHECKPOINT-POOL-SPLIT) y
  `DB_SESSION_MODE_URL` (leader lock del scheduler — advisory locks de sesión
  requieren session mode).
- **Fail-loud**: `neon` sin URLs → RuntimeError → pools None → `/ready` 503.
  NO hay fallback silencioso a Supabase: sería split-brain (escrituras a la DB vieja).
- Rollback de cutover: flip a `supabase` + restart. Cero redeploy.

## Qué se eliminó del schema en Neon (y su reemplazo)

| Objeto Supabase | Reemplazo |
|---|---|
| 75 policies RLS + 62 `ENABLE/FORCE ROW LEVEL SECURITY` | Único cliente = backend; invariante I2 (`AND user_id = %s`) es la defensa |
| 17 FKs `REFERENCES auth.users` | Integridad app-side; delete de usuario = `delete_account_data` + `auth.admin.delete_user` |
| Trigger `on_auth_user_created` → `handle_new_user` | `db_profiles.ensure_user_profile_exists` (INSERT ON CONFLICT DO NOTHING) invocado en `auth.py::get_verified_user_id` (cache in-process) |
| RPC `increment_inventory_quantity` (auth.uid) | `POST /api/inventory/increment` (user_id explícito, I2) |
| RPC `update_health_profile_merge` (auth.uid) | `PATCH /api/profile` (merge jsonb `\|\|` server-side) |
| pg_cron `cleanup_old_meal_rejections` (semanal) | APScheduler `delete_old_meal_rejections_weekly` (Dom 03:15 UTC, cron_tasks.py SSOT) — invoca `public.delete_old_meal_rejections()` que SÍ existe en Neon |
| Realtime publication `supabase_realtime` | Frontend: refetch on visibilitychange + polling acotado en AssessmentContext mientras el plan esté incompleto |

Schema `extensions` replicado en Neon (`vector` 0.8.1, `uuid-ossp`, `pgcrypto`) —
el dump referencia tipos qualified (`extensions.vector(1536)`).

## Script repetible

[`scripts/migrate_db_to_neon.py`](../scripts/migrate_db_to_neon.py):
`dump (pg_dump 17) → clean (state machine con manejo COPY/dollar-quoting) →
restore (psql --single-transaction) → verify (row counts por tabla ambos lados)`.

Flags: `--skip-dump` / `--only-clean` / `--verify-only` / `--reset-neon`
(DROP SCHEMA public CASCADE en Neon para re-sync de cutover).
Credenciales: `SUPABASE_DB_URL` + `NEON_DATABASE_URL` (env o backend/.env).
Binarios pg_dump/psql: env conda `pgtools` (`PGTOOLS_BIN` para override).

Migración inicial ejecutada 2026-06-12: 43 tablas, 42/43 row-counts exactos
(`pipeline_metrics` drift de crons vivos, verificado al row con
`created_at <= snapshot`), 15 funciones, 3 triggers, 2 índices HNSW, vectores 1536.

## SOP de cutover

1. `python scripts/migrate_db_to_neon.py --reset-neon` (re-sync final con datos frescos; idealmente con backend detenido o en ventana de bajo tráfico).
2. Flip `MEALFIT_DB_BACKEND=neon` en el entorno (VPS/EasyPanel) + restart.
3. Verificar log de boot: `Backend de datos: NEON` + `/ready` 200 + `/health/version` drift check.
4. Golden-path manual: login → perfil → nevera → generar plan.
5. Rollback si algo falla: `MEALFIT_DB_BACKEND=supabase` + restart (los datos de Supabase quedan congelados al momento del paso 1 — minimizar la ventana).

## Trampas conocidas (para futuros call sites SQL)

- **Tipos**: psycopg devuelve `uuid.UUID`/`datetime`/`Decimal` donde PostgREST
  devolvía strings/floats JSON. Convención del rewrite: castear en el SELECT
  (`id::text`, `created_at::text` o `to_jsonb(col)#>>'{}'` para ISO-T,
  `quantity::float8`) cuando el consumer espera el shape JSON.
- **PgBouncer transaction-mode** (pooler Neon): sin prepared statements
  (`prepare_threshold=None` ya seteado), sin advisory locks de SESIÓN
  (los `pg_advisory_xact_lock` transaccionales del repo son seguros;
  el leader lock usa el endpoint directo).
- **`extensions.vector`**: casts como `%s::extensions.vector` o `%s::vector`
  funcionan (search_path de las funciones incluye `extensions`).
