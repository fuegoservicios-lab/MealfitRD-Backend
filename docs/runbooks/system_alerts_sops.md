# Runbook: SOPs detallados de `system_alerts`

> Origen: P3-AUDIT-6 + P3-CLEANUP (2026-05-11). Movido fuera de CLAUDE.md
> por presión de tamaño. CLAUDE.md mantiene el modelo de resolution + tabla
> de `alert_keys` + link a este runbook.

Pasos paso-a-paso para resolver alerts cuyo modelo es **Manual** o
**Auto (implicit) + Manual cleanup**.

---

## SOP: resolver `plan_data_corrupted:<plan_id>:<field_name>`

Alert `Manual` — cada incidente requiere decisión humana entre rollback y
hotfix.

### 1. Extraer plan_id + field_name

Formato del `alert_key`: `plan_data_corrupted:<plan_id>:<field_name>`. El
`field_name` es una de las keys que `_check_plan_data_invariants`
(`backend/cron_tasks.py`) marca como corrupta (e.g., `days`,
`aggregated_shopping_list`, `_chunk_lessons`, `_merged_chunk_ids`).

### 2. Leer plan_data del meal_plan afectado

```sql
SELECT id, user_id, plan_data->'<field_name>' AS field_value, created_at, updated_at
FROM meal_plans
WHERE id = '<plan_id>'::uuid;
```

Si la query falla en JSON-parse del field, ya tienes la confirmación de
corrupción.

### 3. Backup defensivo ANTES de cualquier mutación

**Opción preferida — helper Python SSOT** (P2-DOC-1, `backend/db_meal_plans_audit.py`):

```python
from db_meal_plans_audit import record_meal_plan_audit_backup

audit_id = record_meal_plan_audit_backup(
    meal_plan_id="<plan_id>",
    action="corruption_repair",
    actor="sre_manual",
    note="Reparando <field_name> drift post-incidente <YYYY-MM-DD>",
)
assert audit_id, "Backup falló — abortar mutación"
```

El helper valida UUID + action contra el CHECK constraint + snapshot
automático del `plan_data` + retorna el id BIGINT del row insertado.
Si retorna None, NO mutes el plan (algo falló — investigar logs WARN).

**Fallback manual** si no tienes Python shell con backend imports:

```sql
INSERT INTO meal_plans_audit (meal_plan_id, plan_data_before, action, actor)
VALUES ('<plan_id>'::uuid,
        (SELECT plan_data FROM meal_plans WHERE id = '<plan_id>'::uuid),
        'corruption_repair', 'sre_manual');
-- Si la tabla audit no existe en tu deploy, exportar a archivo local:
--   psql ... -c "SELECT plan_data FROM meal_plans WHERE id = '<plan_id>'" > backup_<plan_id>.json
```

### 4. Decidir rollback vs hotfix

- **Rollback** (preferido cuando es posible): si el campo tiene valor obvio
  por defecto (e.g., `_merged_chunk_ids = []`), o si
  `plan_chunk_queue.learning_metrics` permite reconstruir (caso
  `_chunk_lessons` — P1-1 reconstruye desde queue).
- **Hotfix**: si el campo no es reconstruible (e.g., `days[*].meals`
  corruptos requieren regen via `regenerate-simplified`).

### 5. Aplicar fix

```sql
-- Rollback ejemplo (campo a default vacío):
UPDATE meal_plans
SET plan_data = jsonb_set(plan_data, '{<field_name>}', '[]'::jsonb),
    updated_at = NOW()
WHERE id = '<plan_id>'::uuid;

-- Hotfix ejemplo (regen completo del plan via endpoint):
--   POST /api/plans/<plan_id>/regenerate-simplified
```

### 6. Cerrar el alert

Manual porque el productor solo emite, no resuelve:

```sql
UPDATE system_alerts
SET resolved_at = NOW()
WHERE alert_key = 'plan_data_corrupted:<plan_id>:<field_name>'
  AND resolved_at IS NULL;
```

### 7. Post-mortem si los incidentes se repiten

(>3 por semana sobre el mismo `field_name`): el bug está en el mutator, no
en el dato. Investigar logs del nodo que escribe ese field (e.g.,
`_chunk_lessons` → `_record_chunk_lesson_telemetry`).

---

## SOP: resolver `deploy_lag_drift_vs_expected`

Alert `Auto (implicit)` pero el cierre requiere acción humana — el cron
solo re-emite mientras hay drift, no resuelve el origen.

### 1. Identificar el delta

Leer `metadata.live_marker` y `metadata.expected_marker` desde la fila de
`system_alerts`. El primero es el `_LAST_KNOWN_PFIX` del binario corriendo
en el VPS Oracle; el segundo es `app_kv_store.expected_last_known_pfix` que el
operador o el script `publish_pfix_marker.py` actualizó.

```sql
SELECT metadata->>'live_marker'   AS live,
       metadata->>'expected_marker' AS expected,
       triggered_at
  FROM system_alerts
 WHERE alert_key = 'deploy_lag_drift_vs_expected'
   AND resolved_at IS NULL;
```

### 2. Decidir qué lado está rezagado

- `expected > live` (caso típico): HEAD ya tiene P-fix nuevo + KV se
  actualizó, pero el binario no se redeployó. **Resolución: redeploy en
  el VPS Oracle** (pull + restart). Tras el restart, el binario nuevo reporta
  `live = expected` y el cron next-eval no re-emite.
- `live > expected` (raro): el binario tiene P-fix más nuevo que el KV.
  Caso ocurre si alguien deployó sin correr `publish_pfix_marker.py`
  post-merge. **Resolución: bumpear el KV** ejecutando
  `python backend/scripts/publish_pfix_marker.py` (lee `_LAST_KNOWN_PFIX`
  del HEAD y hace UPSERT al KV).
- `live ≠ expected` y ninguno es mayor cronológicamente (caso de
  divergencia de ramas): investigar manualmente cuál es el correcto.

### 3. Bump manual del KV (si el script falla)

```sql
UPDATE app_kv_store
   SET value = to_jsonb('Pn-X · YYYY-MM-DD'::text),
       updated_at = NOW()
 WHERE key = 'expected_last_known_pfix';
```

Formato: JSON string puro (mismo que produce `json.dumps()` del script
SSOT en `backend/scripts/publish_pfix_marker.py`).

### 4. Cerrar el alert manualmente

Si tras la corrección el cron tarda en re-evaluar (cron corre cada
`MEALFIT_DEPLOY_LAG_INTERVAL_H` horas, default 24h):

```sql
UPDATE system_alerts
   SET resolved_at = NOW()
 WHERE alert_key = 'deploy_lag_drift_vs_expected'
   AND resolved_at IS NULL;
```

Idempotente. Si el cron lo re-emite después porque la condición persiste,
vuelve a aparecer (no daño).

### 5. Verificar el cierre

`GET /health/version` debe responder con `last_known_pfix` igual a
`expected_last_known_pfix` del KV. Si difieren, el deploy no se completó.

**Alternativa P1-OBS-1**: `GET /api/system/admin/health-snapshot` retorna
`drift: false` en el body cuando los markers coinciden.

### 6. Post-mortem si este alert reaparece

(>2 veces en una semana): significa que el pipeline operacional
"merge → bump marker → publicar KV → redeploy" tiene un gap. Investigar
si el script `publish_pfix_marker.py` se está ejecutando como parte del
workflow de deploy (debería ser post-merge, antes de redeployar) o si CI
no lo invoca.

---

## Limpieza one-shot de alerts huérfanas

Alerts cuyos productores ya no existen (deploys viejos) o cuya recuperación
no quedó observable persisten en la tabla. Para purgar manualmente
(recomendado pre-deploy de cambios al listener):

```sql
UPDATE system_alerts
SET resolved_at = NOW()
WHERE resolved_at IS NULL
  AND triggered_at < NOW() - INTERVAL '7 days'
  AND alert_key NOT LIKE 'corrupted_%';  -- conserva escalaciones críticas
```
