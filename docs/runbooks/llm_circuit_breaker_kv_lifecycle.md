# Runbook: Ciclo de vida del KV `llm_circuit_breaker:*`

> Origen: P3-NEW-E (audit 2026-05-11). Movido fuera de CLAUDE.md en P3-AUDIT-1
> (2026-05-12) para reducir el contexto. Las rutas `[...](backend/...)` son
> relativas a la raíz del repo — renderizar desde root para que resuelvan.

Estado persistente del `LLMCircuitBreaker` (`graph_orchestrator.py:1143`).
Decisiones: ¿cuándo abrir? ¿cuándo cerrar? ¿quién es la fuente de verdad —
Redis o `app_kv_store`?

## Keys observables en `app_kv_store`

| Pattern | Cuándo se usa | Construido por |
|---|---|---|
| `llm_circuit_breaker` | CB global / legacy (callers sin atribución de modelo) | `_circuit_breaker` singleton (`graph_orchestrator.py:1600`) |
| `llm_circuit_breaker:<model>` | CB per-modelo (P1-Q3, default desde 2026 — ej. `:gemini-3-flash-preview`, `:gemini-3.1-pro-preview`) | `_get_circuit_breaker(model)` registry (`graph_orchestrator.py:1633`). Sufijo construido en `LLMCircuitBreaker.__init__` como `f":{model_name}"` (`graph_orchestrator.py:1191`) |

## Shape del payload

```json
{"failures": <int>, "last_failure": <epoch_seconds>, "is_open": <bool>}
```

`failures = 0, last_failure = 0, is_open = false` es el **estado canónico
cero** (post-reset).

## Diagrama de transiciones

```
                            ┌─────────────────────────────┐
                            │ closed (canonical zero)     │
                            │ failures=0, is_open=false   │
                            └─────────────────────────────┘
                                  │                  ▲
                  record_failure  │                  │ record_success → _atomic_reset_db
                  (N veces)       │                  │      (UPSERT al payload cero)
                                  ▼                  │
                            ┌─────────────────────────────┐
                            │ open                        │
                            │ failures ≥ threshold        │
                            │ is_open=true                │ ← _sweep_stale_llm_circuit_breakers
                            │ last_failure=NOW().epoch    │   (P2-NEW-D · 2026-05-11)
                            └─────────────────────────────┘   horario: reset si
                                                              last_failure < NOW() - N horas
                                                              (knob MEALFIT_CB_KV_STALENESS_HOURS)
```

## Storage layers (defensa-en-profundidad)

| Layer | Keys | TTL | Path |
|---|---|---|---|
| Redis (cuando disponible) | `cb:llm:failures{,:model}` + `cb:llm:open{,:model}` | TTL = `reset_timeout` (default 30s). **Las keys EXPIRAN automáticamente** | `record_failure` / `can_proceed` fast path |
| `app_kv_store` (fallback DB) | `llm_circuit_breaker{,:model}` | **SIN TTL** — fila persiste hasta UPSERT al cero | Cuando Redis está down o no configurado |

## Las 3 vías de reset del KV en DB

1. **`record_success()` → `_atomic_reset_db()`** (`graph_orchestrator.py:1422`):
   UPSERT idempotente al payload cero. Esta es la vía "natural" — un caller
   del modelo tras success.
2. **Runtime auto-expira en `can_proceed()`** (`graph_orchestrator.py:1354`):
   `if time.time() - last_failure > reset_timeout: return True`. **NO** toca
   la fila — solo retorna proceed. La fila queda con `is_open=true` aunque
   funcionalmente esté cerrado.
3. **Sweep periódico `_sweep_stale_llm_circuit_breakers`** (`cron_tasks.py`,
   P2-NEW-D): cada `MEALFIT_CB_KV_STALENESS_SWEEP_INTERVAL_MIN` (default 60
   min) UPDATE filas cuya `(value->>'last_failure')::float < NOW() -
   MEALFIT_CB_KV_STALENESS_HOURS` (default 2 h) AND `is_open='true' OR
   failures > 0`. Reescribe al mismo payload canónico cero que
   `_atomic_reset_db`.

## Stale = `runtime cerrado + DB todavía abierto`

Caso #2 (runtime auto-expira) produce el estado stale. Sin el sweep #3,
la fila queda `is_open=true` indefinidamente si tras el outage el modelo
deja de routearse (ej. `gemini-3.1-pro-preview` sin perfil clínico activo).
Audit 2026-05-11 detectó `gemini-3.1-pro-preview` is_open=true durante 4.4
días — ese es exactamente el modo de fallo que P2-NEW-D cierra.

## Knobs operacionales

(Tabla canónica también vive en CLAUDE.md por test contract — mantener
sincronizada.)

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_CB_FAILURE_THRESHOLD` | 3 | N fallos consecutivos antes de abrir el breaker |
| `MEALFIT_CB_RESET_TIMEOUT_S` | 30 | Ventana tras `last_failure` después de la cual `can_proceed` retorna True |
| `MEALFIT_CB_LOCAL_HEALTH_TTL_S` | 1.0 | TTL del cache local in-process antes de re-consultar Redis/DB |
| `MEALFIT_CB_KV_STALENESS_HOURS` | 2 | Edad mínima de `last_failure` para que el sweep P2-NEW-D considere la fila stale |
| `MEALFIT_CB_KV_STALENESS_SWEEP_INTERVAL_MIN` | 60 | Frecuencia del cron del sweep |

## Cómo verificar el estado live

```sql
SELECT key, value->>'is_open' AS open, value->>'failures' AS fails,
  TO_TIMESTAMP((value->>'last_failure')::float) AS last_failure_iso,
  EXTRACT(EPOCH FROM (NOW() - updated_at))/3600 AS hours_since_last_update
FROM app_kv_store
WHERE key LIKE 'llm_circuit_breaker%';
```

Si una fila reporta `open=true` con `last_failure` >2 h vieja, el sweep
P2-NEW-D la cerrará en su próximo tick. Si `open=true` con `last_failure`
muy reciente, el breaker está genuinamente abierto — investigar el
proveedor LLM antes de tocarlo manualmente.

## Archivos clave

- `backend/graph_orchestrator.py:1143` — clase `LLMCircuitBreaker`.
- `backend/graph_orchestrator.py:1191` — construcción del sufijo de key per-modelo.
- `backend/graph_orchestrator.py:1422` — `_atomic_reset_db` (vía #1).
- `backend/graph_orchestrator.py:1354` — `can_proceed` auto-expiración (vía #2, runtime-only).
- `backend/cron_tasks.py` — `_sweep_stale_llm_circuit_breakers` (vía #3, P2-NEW-D).
- `backend/tests/test_p3_new_e_cb_kv_lifecycle_doc.py` — anchor parser-based cross-link docs↔código (verifica subsección + marker + knobs presentes en CLAUDE.md).

## SOPs operacionales

### SOP-1: alerta "CB stale" con fila `is_open=true` por >24h

1. Verificar si el modelo sigue routeándose:
   ```sql
   SELECT node, count(*) FROM pipeline_metrics
   WHERE created_at > NOW() - INTERVAL '24 hours'
     AND metadata->>'model_used' = '<model_name>'
   GROUP BY node;
   ```
2. Si 0 filas → el modelo está OFF (perfil clínico desactivado etc.).
   El sweep P2-NEW-D debe cerrar la fila — si no, knob
   `MEALFIT_CB_KV_STALENESS_HOURS` es muy alto, bajar temporalmente.
3. Si filas recientes → el breaker está bien abierto. Investigar el
   proveedor (`/health/version` + logs del proveedor).

### SOP-2: forzar reset manual de una fila CB

```sql
UPDATE app_kv_store
SET value = jsonb_build_object(
  'failures', 0,
  'last_failure', 0,
  'is_open', false
),
updated_at = NOW()
WHERE key = 'llm_circuit_breaker:<model>';
```

NO usar `DELETE` — el código asume la fila existe y vuelve a crearla en
el siguiente UPSERT. Reset preserva la fila con shape canónica.
