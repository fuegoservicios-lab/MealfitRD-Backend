# DB pool load test

> [P0-PROD-AUDIT-1 · 2026-05-23] SSOT operacional para validar el comportamiento
> del pool de conexiones bajo carga. Cierra el gap B-P0-5 del audit
> production-readiness: "3 pools (sync + async + chat_checkpoint), knobs
> `MEALFIT_DB_POOL_*`, ningún test ejecutado de saturación real → primer pico
> de tráfico puede tumbar el backend".

## Por qué importa

`db_core.py` mantiene 3 pools simultáneos:

| Pool | Uso | Knob de tamaño |
|---|---|---|
| `connection_pool` (sync) | Mayoría de queries productivas (CRUD de meal_plans, user_facts, system_alerts, etc.) | `MEALFIT_DB_POOL_MAX_SIZE=60` |
| `async_connection_pool` | Crons + endpoints async (background tasks, drain queues) | mismo knob |
| `chat_checkpoint_pool` | LangGraph checkpoints — forzado a puerto :5432 (NO Supavisor pooler) | `MEALFIT_CHAT_CHECKPOINT_POOL_*` |

Con 60 conns max y queries promedio de ~50 ms, capacidad teórica máxima es
~1200 queries/s. Pero el ratio real **queries por request** varía:

- `GET /health/version` → 1 query.
- `POST /api/plans/generate` → 8-15 queries (lookup user_profile, insert
  meal_plans, insert plan_chunk_queue, telemetry, ...).
- `GET /api/plans/history-list` → 1 query (con paginación).

Sin test que ejerza estos paths bajo carga, el pico real puede saturar el
pool antes de saturar CPU/red — modo de fallo silente que solo se ve cuando
ya está rota la prod.

## Quick start

### Smoke local (sanity post-deploy de cambios DB)

```bash
# Asumiendo backend corriendo en localhost:3001.
./scripts/load_test_db_pool.py \
    --target http://localhost:3001 \
    --concurrent 15 \
    --duration 30 \
    --scenarios health,ready,version
```

Salida esperada en MVP (<100 MAU):
```
[load_test] verdict=PASS  total=4500  wall=30.0s  rps=150.0
scenario        count  errors  err%     p50     p95     p99     max
health           1502       0  0.00%   2.3ms   8.1ms  15.4ms  42.1ms
ready            1499       0  0.00%   3.1ms  10.2ms  18.7ms  61.3ms
version          1499       0  0.00%  18.5ms  45.2ms  72.1ms 110.3ms
```

### Stress en staging (validar pool ANTES de lanzamiento público)

```bash
# Requiere JWT válido del usuario de test creado para load testing.
export STAGING_JWT_FOR_LOAD_TEST_USER="..."
./scripts/load_test_db_pool.py \
    --target https://staging.mealfit.example.com \
    --concurrent 100 \
    --duration 120 \
    --bearer-token "$STAGING_JWT_FOR_LOAD_TEST_USER"
```

### Run en CI (gate post-deploy a staging)

Añadir step al workflow de deploy a staging:

```yaml
- name: Load test (smoke)
  run: |
    ./scripts/load_test_db_pool.py \
      --target https://staging.mealfit.example.com \
      --concurrent 30 \
      --duration 60 \
      --scenarios health,ready,version \
      --fail-on FAIL
```

## Lectura del output

### Verdict

El script imprime `verdict=PASS|WARN|FAIL` basado en thresholds configurables:

| Métrica | PASS | WARN | FAIL (default) |
|---|---|---|---|
| error_rate (todos los scenarios) | <= 1% | 1-5% | > 5% |
| p95 latencia (health/ready/version) | <= 200ms | 200-500ms | > 500ms |
| p95 latencia (/api/*) | <= 1500ms | 1500-3000ms | > 3000ms |

Thresholds overridables via env vars `MEALFIT_LOAD_TEST_*` (ver `--help`).

### Diagnóstico de error rate alto

| Error breakdown | Causa probable | Mitigación |
|---|---|---|
| `ConnectError` / `ConnectTimeout` | El cliente no obtuvo conexión TCP — proxy/load balancer saturado, NO el backend. | Aumentar workers del proxy upstream o el `--concurrent` del test es demasiado alto para la infra. |
| `ReadTimeout` con p95 alto | Backend recibió la request pero no respondió a tiempo. Pool DB saturado, queries lentas, o LLM call lento. | Revisar `/api/system/atomic-pool-health` durante el run — si `pool_full=true`, subir `MEALFIT_DB_POOL_MAX_SIZE`. |
| `http_429` | Rate limiter del backend activado (correcto). | El test está martillando un endpoint user-scoped — distribuir entre múltiples user_ids o reducir concurrencia. |
| `http_500` | Bug genuino en el backend. | Revisar Sentry + logs del worker durante la ventana del test. |
| `http_503` | Backend en `not_ready` state — `/ready` 503 durante el test. | Cold start de `warm_plan_graph` no completó. Esperar 30-60s post-arranque antes del test. |

### Diagnóstico de latencia degradada (p95 subiendo)

Mirar el **delta entre p50 y p95**:

- p50 ~= p95 → latencia uniforme. El backend está al límite consistente.
  Causa: pool DB cerca del límite, o LLM API lenta. Acción: subir
  `MEALFIT_DB_POOL_MAX_SIZE` o reducir `--concurrent`.
- p50 << p95 → cola de espera. Algunas requests esperan recursos
  (conn del pool, semáforo, lock). Acción: investigar contención. Mirar
  `system_alerts` con `alert_type='lock_contention'` durante la ventana.
- p95 ~= p99 (sin spike) → distribución uniforme bajo carga. OK.
- p99 >> p95 → outliers reales. Una pocas requests están tardando mucho
  (probable LLM call). Acción: validar circuit breaker activo + timeouts
  configurados (knob `MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S`).

### Diagnóstico server-side durante el run

En otra terminal, mientras corre el load test:

```bash
# Snapshot del pool DB del server (requiere CRON_SECRET).
curl -fsS https://staging.mealfit.example.com/api/system/atomic-pool-health \
    -H "Authorization: Bearer $CRON_SECRET" | jq

# Alerts emitidas en la ventana del test.
curl -fsS https://staging.mealfit.example.com/admin/cron-health | jq '.missed_last_hour'
```

`/api/system/atomic-pool-health` devuelve `{pool_full, used, idle, requests_waiting}`.
Si `requests_waiting > 0` durante el test → el pool es el cuello de botella.

## SOP: ajustar `MEALFIT_DB_POOL_MAX_SIZE`

1. **Baseline actual**: ejecutar load test con concurrent = 2x MAU esperado en pico.
2. **Si verdict=FAIL por `pool_full`**:
   - Subir `MEALFIT_DB_POOL_MAX_SIZE` en EasyPanel env vars (+20 cada iteración).
   - Recordar el límite del plan Supabase: free tier soporta ~60 conns total;
     Pro soporta ~200; Team ~500. NO subir el knob arriba del límite del plan
     o Supabase rechazará nuevas conns con `FATAL: too many connections`.
3. **Si verdict=FAIL por queries lentas**: el pool size no es el problema —
   investigar índices faltantes (revisar Supabase Database → Performance Insights).
4. **Validar con re-test**: verdict debe pasar a PASS.

## Limitaciones

- El script NO simula el SSE pattern (`/api/plans/stream-generate` mantiene
  conexión abierta minutos). Para validar SSE, usar `wscat` o un tester
  específico — out of scope de este script de pool DB.
- NO mide CPU/memoria del backend — usar `docker stats` o el dashboard
  EasyPanel en paralelo.
- NO simula el patrón de payment webhook PayPal (que viene con burst de
  retries específicos). Para eso, usar PayPal sandbox + tester nativo.

## Tests de regresión

- [`tests/test_p0_prod_audit_5_load_test_script.py`](../../tests/test_p0_prod_audit_5_load_test_script.py): valida que el script existe, es ejecutable, importable, y declara los scenarios canónicos.

## Roadmap

- **P1**: integrar verdict en `/admin/cron-health` o emitir alert `load_test_baseline_violated` si el último run pasa de PASS a WARN/FAIL.
- **P2**: scenario para SSE (`/api/plans/stream-generate` con timeout configurable).
- **P2**: dashboard Grafana con histograma de p50/p95/p99 por scenario (timestamp series para detectar drift).
