# Migration rollback SOP

> [P0-PROD-AUDIT-1 · 2026-05-23] SSOT operacional para reaccionar a una
> migración Supabase rota en producción. Cierra el gap B-P0-4 del audit
> production-readiness: "Política forward-only sin runbook → SRE improvisa
> en incidente".

## Política base: forward-only

Las migraciones de Supabase en `supabase/migrations/` son **append-only**:

- Usan `CREATE TABLE IF NOT EXISTS`, `ADD COLUMN IF NOT EXISTS`, `DROP CONSTRAINT IF EXISTS`.
- NO existen archivos `*_rollback.sql` ni paths automatizados de "down migration".
- La razón: rollback automático en datos productivos riesga lost-write
  (rollback de `ADD COLUMN` con datos genuinos = pérdida silente). Mejor
  forward-fix explícito que rollback ambiguo.

Este runbook documenta el SOP para reaccionar a tres escenarios:

1. **Migración falla durante apply** (transacción rollback automático, sin daño).
2. **Migración aplica con éxito pero introduce regresión** (datos quedaron en estado nuevo, código no maneja).
3. **Migración aplica con éxito pero corrupta datos** (worst case — necesita restore selectivo).

## Escenario 1: migración falla durante apply

### Síntomas

- Supabase dashboard → Database → Migrations muestra el run en estado `failed`.
- Logs del MCP `apply_migration` muestran `ERROR: ...`.
- O `mcp__supabase__apply_migration` retorna `status: error` con SQLSTATE.

### Diagnóstico

```sql
-- Postgres rolló back la transacción automáticamente. Validar que NINGÚN
-- objeto de la migración existe (debería ser estado limpio pre-migración).
SELECT table_name, column_name FROM information_schema.columns
WHERE table_schema = 'public' AND column_name LIKE '<column_de_la_migracion>%';

-- Si la migración era CREATE TABLE:
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public' AND table_name = '<tabla_de_la_migracion>';

-- Si la migración era CREATE INDEX:
SELECT indexname FROM pg_indexes
WHERE schemaname = 'public' AND indexname = '<idx_de_la_migracion>';
```

### Recovery

1. **Investigar el SQLSTATE del error**:
   - `42P07` (`duplicate_object`): objeto ya existe — verificar si `IF NOT EXISTS` está presente.
   - `42703` (`undefined_column`): la migración asume columna que no existe — probable orden de migrations roto.
   - `23505` (`unique_violation`): violación de constraint al backfill — datos productivos no cumplen el invariant.
   - `40001` (`serialization_failure`): conflicto con sesión activa — esperar y reintentar.
   - `25001` (`active_sql_transaction`): el cliente abrió tx manual; cerrarla.

2. **Fix forward**: editar el archivo de migración (mismo nombre — Supabase rastrea por `version` que es el timestamp en el filename, NO por hash del contenido) y re-aplicar.

3. **Notificar el incidente**: insertar fila en `system_alerts`:
   ```sql
   INSERT INTO system_alerts (alert_type, alert_key, severity, message, metadata)
   VALUES ('migration', 'migration_apply_failed:<filename>', 'high',
           'Migration <filename> falló con SQLSTATE <code>. Investigado y re-aplicado: <descripción>.',
           jsonb_build_object('filename', '<filename>', 'sqlstate', '<code>', 'incident_at', NOW()));
   ```

### Tests de regresión

Cada migración nueva debe tener test parser-based en `tests/test_<slug>.py` que
valide existencia + tipo + nullability del objeto creado. Ver
[`test_p2_next_4_meal_plans_complete_requires_days.py`](../../tests/test_p2_next_4_meal_plans_complete_requires_days.py) como ejemplo canónico.

## Escenario 2: migración aplica con éxito pero introduce regresión

### Síntomas

- Migration aplicada (estado `applied` en dashboard).
- Tests fallan en CI tras el merge.
- O Sentry empieza a reportar errores nuevos (e.g. `column "old_name" does not exist`).
- O usuarios reportan funcionalidad rota.

### Diagnóstico

```bash
# Verificar el commit del deploy actual en producción.
curl -fsS https://<easypanel-domain>/health/version | jq '{last_known_pfix, git_sha}'

# Comparar con HEAD del repo.
git log --oneline -5

# Si git_sha de prod << HEAD → deploy lag. La migración aplicó vía MCP pero
# el deploy del código nuevo está rezagado (raro pero posible si el MCP run
# corrió antes del push). Forzar redeploy.
```

```sql
-- Verificar que el objeto nuevo existe pero que el código no lo está usando.
SELECT column_name, data_type FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = '<tabla_afectada>'
ORDER BY ordinal_position;
```

### Recovery

**Opción A — fix forward (preferida)**:

1. Implementar el código que maneja el estado nuevo (e.g. handler que lee la columna recién añadida).
2. Push + redeploy.
3. Validar que `system_alerts` deja de emitirse para el alert_key relacionado.

**Opción B — feature flag / knob de bypass (mitigación temporal)**:

Si el fix forward no es trivial y la regresión bloquea usuarios, usar un knob
`MEALFIT_*` que desactive la feature dependiente del schema nuevo. Ejemplo
real: `MEALFIT_SHOPPING_COHERENCE_GUARD=off` rolling back desde `block` sin
redeploy.

**Opción C — migración compensatoria forward**:

Si el problema es estructural (e.g. `NOT NULL` constraint con datos NULL
preexistentes), crear migración nueva que relaje:

```sql
-- p4_compensate_p3_too_strict_check.sql
ALTER TABLE <tabla> ALTER COLUMN <col> DROP NOT NULL;
-- COMMENT explicando el incidente + apuntar al runbook.
COMMENT ON COLUMN <tabla>.<col> IS 'Relaxed from NOT NULL on 2026-05-XX after migration <orig> broke <N> users. See docs/runbooks/migration_rollback.md';
```

### Anti-patrón: rollback "manual" con DROP

NO ejecutar `DROP COLUMN`/`DROP TABLE`/`DROP CONSTRAINT` desde el dashboard
SQL editor para "deshacer" la migración. Riesgos:

- **Datos productivos**: si la columna nueva ya tiene escrituras del worker,
  drop = pérdida silente.
- **Auditoría rota**: Supabase rastrea migrations aplicadas; un DROP
  manual deja inconsistencia entre dashboard y schema real.
- **Recovery imposible**: si después decides re-aplicar la migración, el
  estado intermedio es ambiguo.

Si DEBES dropear (caso extremo, datos comprometidos), seguir el escenario 3.

## Escenario 3: migración corruptó datos (worst case)

### Síntomas

- Migración aplicó.
- Datos productivos terminaron en estado inválido (e.g. backfill con bug,
  trigger nuevo escribió valores erróneos).
- Tests `meal_plans_audit` o `data_corruption_*` alerts emitidos.

### Recovery (SOP P3-AUDIT-6 cross-link)

Esto NO es rollback de migración — es restore de datos. La migración (DDL)
queda en su lugar; solo los DML rotos se reparan.

1. **Identificar el blast radius**:
   ```sql
   SELECT COUNT(*) FROM <tabla> WHERE <condición_de_corrupción>;
   ```

2. **Snapshot pre-fix** (defensa contra fix que rompe más):
   ```sql
   CREATE TABLE <tabla>_pre_fix_2026_05_XX AS
   SELECT * FROM <tabla> WHERE <condición_de_corrupción>;
   ```

3. **Si la tabla afectada es `meal_plans`**: usar el SOP P3-AUDIT-6
   (referencia en `~/.claude/projects/.../memory/runbook_migration_audit_6.md`).
   `meal_plans_audit` tiene snapshot `plan_data_before` de cada mutación —
   restore selectivo posible:
   ```sql
   UPDATE meal_plans SET plan_data = (
     SELECT plan_data_before FROM meal_plans_audit
     WHERE meal_plan_id = meal_plans.id
       AND created_at > '<timestamp_de_la_migración>'
     ORDER BY created_at ASC LIMIT 1
   ) WHERE id IN (<lista_afectada>);
   ```

4. **Para otras tablas sin audit**: el restore requiere backup point-in-time
   (PITR) de Supabase. Solo plan Pro+. Si no disponible: forward-fix manual
   por usuario afectado.

5. **Cerrar el incident**:
   ```sql
   UPDATE system_alerts SET resolved_at = NOW()
   WHERE alert_key LIKE 'data_corruption_%' AND resolved_at IS NULL;
   ```

## SOP general: pre-flight checklist antes de aplicar migración productiva

Antes de invocar `mcp__supabase__apply_migration` contra el proyecto de
producción:

- [ ] El archivo SQL vive en `supabase/migrations/` con nombre
      `pN_<slug>_<YYYY_MM_DD>.sql` (convención del repo).
- [ ] El archivo está duplicado en `supabase/migrations/` Y `backend/supabase/migrations/`
      (SSOT cross-repo — ver CLAUDE.md "SSOT de migrations"). Verificar:
      ```bash
      diff <(ls supabase/migrations) <(ls backend/supabase/migrations)
      # Debe retornar vacío.
      ```
- [ ] Idempotente: todas las cláusulas usan `IF NOT EXISTS` /
      `IF EXISTS` apropiados.
- [ ] Sanity check con `DO $$ RAISE EXCEPTION` al final si el invariante
      crítico no se cumple post-apply.
- [ ] Test parser-based existe en `tests/test_<slug>*.py` validando que
      el objeto creado tiene tipo/nullability esperados.
- [ ] Backfill (si aplica) testeado contra ≥3 muestras representativas.
- [ ] Si la migración hace `NOT NULL` / `CHECK` constraint en tabla con
      datos productivos: validar manualmente que TODOS los rows pre-existentes
      cumplen el constraint:
      ```sql
      SELECT COUNT(*) FROM <tabla> WHERE NOT (<constraint_expr>);
      -- Debe ser 0. Si no, fix de datos primero, constraint después.
      ```
- [ ] Estimar el tiempo de bloqueo: `EXPLAIN ANALYZE` o aplicar primero
      en un branch de Supabase (preview environment). Tablas grandes
      (~1M rows) con `ADD COLUMN ... DEFAULT <expr>` pueden bloquear
      escrituras varios minutos.

## SOP: aplicar migración en branch (preferred para cambios riesgosos)

```bash
# 1. Crear branch en Supabase via MCP.
mcp__supabase__create_branch --confirm_cost_id <id> --name "migration_test_<slug>"

# 2. Aplicar la migración en el branch.
mcp__supabase__apply_migration --project_id <branch_id> --name <slug> --query "$(cat supabase/migrations/<file>.sql)"

# 3. Smoke test: queries representativas + tests parser-based contra el branch.

# 4. Si OK → merge.
mcp__supabase__merge_branch --branch_id <branch_id>

# 5. Si fail → delete branch y fix forward.
mcp__supabase__delete_branch --branch_id <branch_id>
```

## Tests de regresión

- [`tests/test_p0_prod_audit_2_migration_rollback_runbook.py`](../../tests/test_p0_prod_audit_2_migration_rollback_runbook.py):
  Valida que este runbook existe, tiene secciones requeridas, y cubre los 3
  escenarios canónicos. Si alguien borra el runbook o le quita secciones
  críticas (e.g. el SOP de pre-flight checklist), el test falla.
