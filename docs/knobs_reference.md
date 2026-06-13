# Knobs operacionales `MEALFIT_*` — referencia de discovery

[P2-KNOBS-OPERATIONAL-DOC · 2026-05-23] Este documento existe para resolver
el gap operacional del audit 2026-05-23: el backend tiene **~161 env vars
`MEALFIT_*`** referenciados en código, pero `backend/.env.example` solo
documenta ~10 overrides recomendados. Un operador nuevo no sabe dónde
están los demás, y durante un incidente puede no encontrar el knob de
mitigación.

Este doc **NO mirror** los 161 knobs (drift garantizado). En lugar de eso:

1. **Explica el mecanismo de auto-registro** (cómo descubrirlos at runtime).
2. **Enumera knobs de alto valor para producción** (los que un SRE
   probablemente quiera tunear sin esperar redeploy).
3. **Linka el endpoint público que expone los activos**.

## Mecanismo de auto-registro

Todo lector de env var con prefijo `MEALFIT_*` en el código pasa por uno
de los wrappers:

| Wrapper | Tipo | Archivo |
|---|---|---|
| `_env_int(name, default, validator=None)` | `int` | `graph_orchestrator.py` |
| `_env_float(name, default, validator=None)` | `float` | `graph_orchestrator.py` |
| `_env_bool(name, default)` | `bool` | `graph_orchestrator.py` |
| `_env_str(name, default, validator=None)` | `str` | `graph_orchestrator.py` |
| `_knob_env_float(name, default, validator=None)` | `float` | `app.py` |
| `_env_int_safe(name, default)` | `int` (no registry) | `rate_limiter.py` |

Cada llamada AÑADE una entry a `_KNOBS_REGISTRY` (módulo
`graph_orchestrator.py`) con `{name, type, default, current, validator,
caller_module}`. El `rate_limiter._env_int_safe` es la excepción
intencional (no se registra para evitar circular import upstream;
documentado vía CLAUDE.md o memoria del bundle correspondiente).

## Descubrir los knobs activos at runtime

### Endpoint público `/health/version` (sin auth)

```bash
curl -s https://<host>/health/version | jq '.knobs_registered_count, .knobs_registered_preview'
```

El response expone:
- `knobs_registered_count`: total de knobs en `_KNOBS_REGISTRY`.
- `knobs_registered_preview`: muestra de keys recientes (no el set completo
  — sería verbose en el JSON público).

### Python interactivo (dentro del proceso backend)

```python
from graph_orchestrator import get_knobs_registry_snapshot
snapshot = get_knobs_registry_snapshot()
# dict[str, dict] — `{knob_name: {type, default, current, caller_module}}`
print(sorted(snapshot.keys()))
```

### Grep cross-codebase (last resort, source-of-truth)

```bash
# Todos los lectores de env var MEALFIT_*:
grep -rn "_env_int\|_env_float\|_env_bool\|_env_str\|_knob_env_float" backend/ \
  --include="*.py" | grep -v test_

# O directo desde os.environ:
grep -rn 'os.environ.get("MEALFIT_\|os.getenv("MEALFIT_' backend/ \
  --include="*.py" | grep -v test_
```

## Knobs de alto valor para producción

Estos son los que un operador probablemente quiera tunear sin redeploy
durante un incidente. La lista no es exhaustiva — para el conjunto
completo, usar `/health/version` o `get_knobs_registry_snapshot()`.

### Coherence guard (recetas ↔ lista de compras)

| Knob | Default | Cuándo cambiar |
|---|---|---|
| `MEALFIT_SHOPPING_COHERENCE_GUARD` | `block` | Pasar a `warn` o `off` si guard está rechazando planes legítimos en masa (false-positive burst) |
| `MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT` | `0.10` | Subir a 0.15-0.20 si el LLM tiende a sub/sobre-multiplier por household >2x |
| `MEALFIT_COHERENCE_T2_BLOCK_SEVERE_ONLY` | `True` | Flip a `False` para revertir al warn-only puro si el block-severe genera retry storms |

### Sentry sampling (costo)

| Knob | Default | Cuándo cambiar |
|---|---|---|
| `MEALFIT_SENTRY_TRACES_SAMPLE_RATE` | `0.1` | Subir a `1.0` SOLO para debug intensivo de un deploy específico; lineal con costo |
| `MEALFIT_SENTRY_PROFILES_SAMPLE_RATE` | `0.1` | Igual que traces — profiling es aún más caro |

### Circuit breaker LLM

| Knob | Default | Cuándo cambiar |
|---|---|---|
| `MEALFIT_CB_FAILURE_THRESHOLD` | `3` | Subir a 5-7 si el provider LLM está flap-eando 5xx pero recovery rápido (3 es agresivo) |
| `MEALFIT_CB_RESET_TIMEOUT_S` | `30` | Subir a 60-120s si los flaps son largos (evita thundering herd post-reset) |

### Rate limiters

| Knob | Default | Cuándo cambiar |
|---|---|---|
| `MEALFIT_RATE_LIMITER_BUCKET_LIMIT_WARN` | `100000` | Bajar a 10000 si sospecha botnet (alert temprano por bucket cardinality) |

### Deploy lag detection

| Knob | Default | Cuándo cambiar |
|---|---|---|
| `MEALFIT_DEPLOY_LAG_CHECK_INTERVAL_HOURS` | `1` | Bajar a `0.25` post-deploy crítico para confirmación rápida `drift=false` |

### DB pool

| Knob | Default | Cuándo cambiar |
|---|---|---|
| `MEALFIT_DB_POOL_MIN_SIZE` | `10` | Subir si los crons + chunks + meta-learning concurrentes están timing out en `pool.checkout()` |
| `MEALFIT_DB_POOL_MAX_SIZE` | `60` | Subir si Supabase pooler está reportando connection saturation |
| `MEALFIT_DB_POOL_TIMEOUT_S` | `10` | Subir a 30 durante migrations grandes que mantienen rows lock-eadas |

### Coherence cron (knobs de frecuencia)

| Knob | Default | Cuándo cambiar |
|---|---|---|
| `MEALFIT_COHERENCE_METRICS_INTERVAL_MIN` | `60` | Bajar a 15 durante incidente para feedback rápido |
| `MEALFIT_COHERENCE_CRON_PERSIST_HISTORY` | `True` | Flip a `False` si el cron genera contención con write paths del usuario |

### LLM model selection

| Knob | Default | Cuándo cambiar |
|---|---|---|
| `MEALFIT_<FEATURE>_MODEL` | varios | Swap de modelo LLM sin redeploy. Patrón `MEALFIT_CHAT_AGENT_MODEL`, `MEALFIT_CRITIQUE_MODEL`, etc. El override per-feature gana sobre el router por tier (P0-DEEPSEEK-MIGRATION). |
| `MEALFIT_MODEL_FREE_TIER` / `MEALFIT_MODEL_PAID_TIER` | `deepseek-v4-flash` / `deepseek-v4-pro` | Router por tier de suscripción — ver `backend/docs/llm_tier_routing.md` |
| `MEALFIT_DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | Proxy/endpoint alternativo OpenAI-compatible |
| `MEALFIT_TIER_CACHE_TTL_S` | `300` | TTL del cache de `plan_tier` por usuario (clamp [10, 3600]) |

### Pantry / chunk operacional

| Knob | Default | Cuándo cambiar |
|---|---|---|
| `MEALFIT_SWEEP_ORPHAN_PLANS_AGE_DAYS` | `7` | Bajar a 2-3 si los orphans plans están saturando metrics (clamp [1, 90]) |
| `MEALFIT_SWAP_RECIPE_COHERENCE_VALIDATE` | `True` | Flip a `False` para revertir al pre-P1-SWAP-RECIPE-COHERENCE behavior si validator genera FPs |

## Cómo añadir un knob nuevo

```python
# En graph_orchestrator.py (o el módulo donde aplica):
_MEALFIT_MI_KNOB = _env_int(
    "MEALFIT_MI_KNOB",
    default=42,
    validator=lambda v: 1 <= v <= 1000,
)
```

El wrapper `_env_int` auto-registra el knob en `_KNOBS_REGISTRY`. La
entry aparecerá en `/health/version` en el próximo `import` del módulo.

**Convención** (CLAUDE.md → "Convenciones del repo"):
- Prefijo `MEALFIT_` (los demás `*_API_KEY`, `SUPABASE_*`, `PAYPAL_*` son
  secretos y NO van en el registry).
- Default SEGURO (el knob debe ser opcional — el sistema arranca sin él).
- Validator si hay clamp (e.g., porcentajes deben estar en [0, 1]).
- Si el knob va a la doc "Knobs operacionales" de CLAUDE.md, añadir
  también allí. Si es interno (no SRE-tunable), basta el registry.

## Anti-patrones

- **NO** leer env var con `os.environ.get(...)` directo en módulo
  productivo (el knob no aparece en el registry → invisible para SRE).
- **NO** dar al knob un default INSEGURO (e.g., `MEALFIT_DISABLE_AUTH=False`
  default está bien; `MEALFIT_DISABLE_AUTH=True` default abriría el
  sistema si la env var se borra accidentalmente).
- **NO** hardcodear thresholds que pueden necesitar rollback sin redeploy
  (e.g., timeouts de LLM, tolerancias de coherence guard, tier limits).

Tooltip-anchor: `P2-KNOBS-OPERATIONAL-DOC-START` | knobs discovery 2026-05-23
