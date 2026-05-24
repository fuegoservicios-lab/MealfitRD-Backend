# Cache invalidation policy

> [P2-PROD-AUDIT-1 · 2026-05-23] SSOT operacional de la política de
> invalidación de cache (`backend/cache_manager.py`). Cierra el gap B-P2-4
> del audit production-readiness: "Cache sin política de invalidación
> documentada → stale data potencial".

## Arquitectura del cache

`backend/cache_manager.py` provee 2 capas:

```
┌─────────────────────────────────────────────────────────┐
│  centralized_cache(ttl_seconds, maxsize)                │
│  decorador para resultados de funciones puras           │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
   ┌──────────┐                ┌─────────────────┐
   │  Redis   │                │  Local OrderedDict │
   │ (shared) │                │  (per-worker)      │
   └──────────┘                └─────────────────┘
   prefer si REDIS_URL set      fallback si Redis down
```

Knob de selección: `REDIS_URL` env var.
- Si set: cache en Redis (compartido entre workers, persiste a través de
  reinicios del worker).
- Si no set: cache local in-memory (per-worker, perdido al restart).

## Modelo de invalidación: TTL-based, NO event-based

La política es **TTL-based**: cada entry expira después de `ttl_seconds`
desde su último write. NO hay invalidación event-driven (e.g. "cuando
el usuario actualiza su perfil, invalidar cache de su plan").

### Por qué TTL en lugar de event-based

- **Simplicidad**: invalidación event-driven requiere observar TODOS los
  write paths que afectan al cached value. Falla a la primera escritura
  no observada → stale data persistente hasta el next TTL expiry.
- **Convergencia**: TTL garantiza que stale data NO sobrevive más de N
  segundos. Bound predecible para SLA.
- **Trade-off conocido**: en la ventana `[write, write + TTL]`, lecturas
  pueden ver datos viejos. Aceptable para cached values que toleran
  inconsistencia bounded (e.g. extracción semántica de facts, embeddings
  de texto, USDA densidades).

### Cuándo NO usar `centralized_cache`

NO cachear (TTL-based no sirve):
- **User profile state mutable**: usar query directa + RLS.
- **plan_data en mutación activa**: usar advisory locks + read-after-write.
- **Auth tokens / session state**: managed por Supabase auth client.
- **Inventory deltas**: usar `apply_inventory_delta` RPC (atómica server-side).

SÍ cachear (TTL-based OK):
- **LLM responses puras** (mismo prompt → mismo output): TTL largo (días+).
- **USDA densidad lookups**: TTL ~permanente (datos estables).
- **Embeddings de strings**: TTL permanente (mismo input → mismo embedding).
- **Aggregaciones que toleran ventana de stale** (e.g. nudge ratios over
  7d): TTL corto (1-2h).

## Knobs operacionales

### Por función decorada (call site)

Cada `@centralized_cache(ttl_seconds=X, maxsize=Y)` declara su TTL inline.
Ver call sites canónicos:

| Función | TTL | Razón |
|---|---|---|
| `fact_extractor:_chunked_fact_extraction` | `CACHE_TTL_PERMANENT` (~años) | LLM extraction stub determinístico para input idéntico |
| `vision_agent:get_text_embedding` | `3153600000` (100 años efectivamente) | Embeddings de texto son funciones puras del input |

### Globales

- `REDIS_URL` (env var): selección Redis vs local cache.
- Sin knob para `maxsize` global — cada call site decide via decorator arg.

### Diagnóstico

`backend/cache_manager.py` no expone endpoint admin para inspeccionar
el cache. Para diagnosticar:

```bash
# Si REDIS_URL set:
redis-cli -u "$REDIS_URL" KEYS '*' | head -20
redis-cli -u "$REDIS_URL" GET '<key>'
redis-cli -u "$REDIS_URL" TTL '<key>'

# Si cache local (sin Redis):
# Restart del worker = clear total. NO hay forma de inspeccionar in-flight.
```

## Invalidación manual

Cuando se necesita forzar refresh (e.g. corregir un valor cacheado roto):

### Caso 1: una entry específica

```bash
# Si conoces la key (formato: `<func_name>:<md5_hash>`):
redis-cli -u "$REDIS_URL" DEL '<func_name>:<hash>'
```

### Caso 2: todas las entries de una función

```bash
redis-cli -u "$REDIS_URL" KEYS '<func_name>:*' | xargs redis-cli -u "$REDIS_URL" DEL
```

### Caso 3: cache local (sin Redis)

Restart del worker — única vía. EasyPanel: Stop + Start del servicio.

### Caso 4: nuke total

```bash
redis-cli -u "$REDIS_URL" FLUSHDB
```

> **PRECAUCIÓN**: `FLUSHDB` elimina TODO el DB Redis activo — no solo
> cache de Mealfit. Si Redis se comparte con otros servicios, usar
> approach por-key (caso 1-2).

## SOP: bug "el cache devuelve valor stale"

### Síntomas
- Usuario reporta "veo datos viejos" tras update.
- Logs muestran lookup hits del cache vs query directa para el mismo input.

### Diagnóstico
1. **Verificar TTL del entry**: `redis-cli TTL '<key>'`. Si retorna número
   alto (~horas), TTL aún no expiró.
2. **Validar que la función decorada es genuinamente "pura"**: si toma
   input que muta entre runs (e.g. `db_facts.get_user_facts(user_id)`
   donde facts cambian con cada extraction), TTL-based cache es INCORRECTO
   para esa función — eliminar el decorator.
3. **Validar el cache key**: hash MD5 de `{args, kwargs}` JSON serializado.
   Si el caller pasa un dict con orden inestable, dos calls "iguales"
   pueden hashear diferente y ambos cachear (memory waste, NO stale).

### Resolución
- Si TTL inadecuado para el use case: bajar a 60s-300s temporalmente,
  evaluar si el cost de query directa es prohibitivo.
- Si función NO es pura: eliminar el decorator (mover a query directa).
- Si bug en hashing: validar JSON serialización determinística.

## Tests de regresión

- [`tests/test_p2_prod_audit_3_cache_policy_documented.py`](../../tests/test_p2_prod_audit_3_cache_policy_documented.py):
  Valida que este runbook existe + cubre las secciones canónicas.

## Roadmap

- **P2**: añadir endpoint admin `GET /admin/cache-stats` que muestre
  hit ratio + count per cache key prefix.
- **P2**: emit alert `cache_redis_unavailable` cuando el fallback a local
  se active en producción (señal de incident de Redis).
- **P3**: considerar event-driven invalidation para subset específico
  (ej. user_profile changes → invalidate cached user-scoped derivations).
