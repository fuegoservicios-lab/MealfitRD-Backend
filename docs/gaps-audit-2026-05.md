# Gaps audit — MealfitRD.IA · 2026-05-20

**Auditor**: staff engineer (Claude Opus 4.7).
**Scope**: workspace-root + `backend/` (FastAPI + LangGraph + Supabase + APScheduler) + `frontend/` (React 19 + Vite 7 + JS).
**Contexto del owner**: 1 dev solo, MVP <100 usuarios, deploy EasyPanel + Nixpacks (VPS), sin incendio activo.
**Filtro de priorización**: para early-stage solo-founder priorizo (a) quick wins de seguridad/cost, (b) deuda obvia que destrabe refactors futuros, (c) observabilidad de negocio, (d) blast-radius de bugs. NO priorizo TypeScript migration, refactor masivo a paquetes, ni alternativas a PayPal — esos costos solo pagan a otra escala.

---

## Resumen Fase 1

Workspace dir con 3 repos git distintos ( `.gitignore` raíz confirma `backend/`+`frontend/` como remotes hermanos `fuegoservicios-lab/MealfitRD-Backend` y `.../MealfitRD`). El workspace-root trackea solo CLAUDE.md + supabase/migrations + scripts + .github.

**Backend**: ~50 archivos `.py` planos en raíz (sin paquetes), 770 archivos en `tests/`, 6 routers, 7 prompts, ~36 migrations en `supabase/migrations/`, 4 runbooks en `docs/runbooks/`. Archivos clave por tamaño: [`cron_tasks.py`](../cron_tasks.py) 27096 líneas, [`graph_orchestrator.py`](../graph_orchestrator.py) 14110, [`routers/plans.py`](../routers/plans.py) 9760, [`shopping_calculator.py`](../shopping_calculator.py) 7694, [`agent.py`](../agent.py) 2407.

**Frontend**: React 19 + Vite 7 + JS puro + PWA injectManifest + Sentry+Replay + PayPal SDK. ~12 pages + `components/` agrupado por feature + `utils/` con 18 helpers + 1 context (`AssessmentContext.jsx` 1935 líneas). Pages "God component": [`History.jsx`](../../frontend/src/pages/History.jsx) 5238, [`Dashboard.jsx`](../../frontend/src/pages/Dashboard.jsx) 4857, [`Pantry.jsx`](../../frontend/src/pages/Pantry.jsx) 4000, [`AgentPage.jsx`](../../frontend/src/pages/AgentPage.jsx) 3063, [`Settings.jsx`](../../frontend/src/pages/Settings.jsx) 2544.

**Flujos críticos**: generación de plan (form → orchestrator → chunks → coherence guard → PDF), vision agent, chat agent (LangGraph + 11 tools con server-side `user_id` override), pagos PayPal (`/verify` + `/cancel` + webhook firmado), crons (chunk workers, coherence, alerts, deploy lag, CB sweep), memoria/facts.

**Madurez sorprendentemente alta**: `_LAST_KNOWN_PFIX` marker, registry de knobs `MEALFIT_*`, ~80 P-fixes documentados, deploy-lag detector con SOP, Sentry PII scrubbing en ambos lados, RLS + atomic helpers + advisory locks, CHECK constraints DB-level, 32+ system_alerts canónicos, doc tables auto-paritadas con tests parser-based, marker→test cross-link enforzado. La fragilidad NO es por inmadurez, sino por **concentración de masa** en archivos enormes (graph_orchestrator, cron_tasks, routers/plans) y por **acumulación de scripts one-shot** en raíz del backend.

**Producto**: 100% es-DO hardcoded (decisión de producto P3-I18N-DEFERRED), PayPal único proveedor (no decisión documentada pero implícita), Vercel para frontend, EasyPanel+Nixpacks para backend, Supabase para DB+auth+RLS.

---

## Tabla de gaps (32 hallazgos, agrupados por categoría)

Convención: **Impacto** Alto/Medio/Bajo (qué tan visible es el daño si NO se arregla). **Esfuerzo** S (<4h) / M (1-3 días) / L (semana+). 🟢 = quick win.

### A. Arquitectura backend

| # | Hallazgo | Impacto | Esfuerzo | Recomendación | Quick win | Repo |
|---|---|---|---|---|---|---|
| A1 | Scripts one-shot quedaron en raíz: [`refactor.py`](../refactor.py) (78l, modifica graph_orchestrator), [`refactor_plans.py`](../refactor_plans.py) (61l), [`modify_cron.py`](../modify_cron.py) (11l), [`recalc_now.py`](../recalc_now.py) (52l one-shot recalc), [`test.py`](../test.py) (11l, UUID hardcoded), [`test_browser_console.js`](../test_browser_console.js) (70l, snippet de browser console). Todos cumplieron su función; ahora son ruido + riesgo de re-ejecución accidental. | Bajo | S | Mover a `backend/scratch/` (que ya existe con README de uso). Eliminar [`test.py`](../test.py) y [`test_browser_console.js`](../test_browser_console.js) (UUIDs reales hardcoded → PII leak menor en repo). [`_force_regen.json`](../_force_regen.json), [`chat_errors.json`](../chat_errors.json), [`db.sqlite3`](../db.sqlite3), [`p2_obs_err.log`](../p2_obs_err.log), [`p2_obs_out.log`](../p2_obs_out.log), [`pyright_results.json`](../pyright_results.json), [`pytest_output.txt`](../pytest_output.txt), [`push_log.txt`](../push_log.txt), [`test.mp3`](../test.mp3) ya están gitignored pero ocupan disk + confunden grep. Gitignore-clean. | 🟢 | backend |
| A2 | [`graph_orchestrator.py`](../graph_orchestrator.py) 14110 líneas. Contiene: state graph + node functions + circuit breaker + KV CB lifecycle + pipeline runner + reshape helpers + retry gate + ~30 knobs. Imposible de navegar sin grep + tooltip-anchors. CLAUDE.md ya admite esta es la zona caliente. | Alto | L | NO refactor a paquete completo (cost altísimo solo). En su lugar: extraer subzonas amantes-de-cambio en orden de fricción real: (1) `LLMCircuitBreaker` + KV lifecycle a `cb.py` (~600l, ya tiene runbook propio), (2) helpers `_env_*` ya extraídos a [`knobs.py`](../knobs.py) — extender a `_KNOBS_REGISTRY` listing, (3) reshape helpers (`expected_sum_from_recipes`, `_normalize_*`) a `plan_reshape.py`. Cada extracción cierra ~500-1500 líneas y tests parser-based actuales seguirán funcionando con `from graph_orchestrator import …` re-exports. Empezar por `cb.py` (ya hay scope claro). | No | backend |
| A3 | [`cron_tasks.py`](../cron_tasks.py) 27096 líneas. ~15 crons distintos en un archivo (chunk_worker, coherence_alert, deploy_lag, CB sweep, billing reconcile, agent_sessions TTL, etc.). | Alto | L | Mismo patrón que A2: extraer por cron. Empezar por `_chunk_worker` + helpers (~5-8k líneas, ya tiene su sub-dominio), después `_alert_*` cron suite, después `_sweep_*` suite. Cada cron a su propio módulo `crons/<name>.py`; `register_plan_chunk_scheduler` en [`cron_tasks.py`](../cron_tasks.py#L793) sigue siendo SSOT del wiring. | No | backend |
| A4 | 50+ archivos `.py` planos en raíz del backend, sin paquetes. `db_*.py` (8 archivos), `tools*.py` (3), `prompts*.py` (1 legacy + dir `prompts/`). Imports cross-module se vuelven brittle (P2-1 admite "circular imports forced extraction de helpers a `knobs.py`"). | Medio | M | Crear paquetes mínimos `db/` (mover `db_*.py` → `db/{plans,inventory,facts,chat,profiles,meal_plans_audit,core}.py` + `db/__init__.py` re-export), `tools/` (`tools/{__init__,medical,nutrition}.py`). Mantener `agent.py`, `services.py`, `graph_orchestrator.py`, `cron_tasks.py`, `app.py` en raíz por su rol de orquestación. Esto reduce confusión del flat layout + habilita refactors futuros sin tocar 80 imports a la vez. | No | backend |
| A5 | [`prompts_legacy.py`](../prompts_legacy.py) (218 líneas) coexiste con dir [`prompts/`](../prompts/) moderno. CLAUDE.md no documenta cuál es SSOT ni si hay drift. | Bajo | S | Grep cross-codebase `from prompts_legacy import` → si 0 callers prod, mover a `scratch/legacy_prompts.py` con README explicativo. Si hay callers, migrar 1 a 1 a [`prompts/`](../prompts/) y luego mover. | 🟢 | backend |
| A6 | [`db.py`](../db.py) sirve como fachada de los `db_*.py` (re-export). Pero módulos productivos importan tanto `from db import ...` como `from db_plans import ...` (services.py:30 hace ambos). Resultado: sin SSOT de cuál fachada usar. | Bajo | S | Documentar en CLAUDE.md "convención de imports DB": **siempre desde `from db import …`** (la fachada). En audit posterior, grep `from db_(plans|inventory|facts|chat|profiles)` y migrar a fachada. Reduce churn de imports cuando A4 pase a paquete. | 🟢 | backend |

### B. Arquitectura frontend

| # | Hallazgo | Impacto | Esfuerzo | Recomendación | Quick win | Repo |
|---|---|---|---|---|---|---|
| B1 | Pages "God component": [`History.jsx`](../../frontend/src/pages/History.jsx) 5238, [`Dashboard.jsx`](../../frontend/src/pages/Dashboard.jsx) 4857, [`Pantry.jsx`](../../frontend/src/pages/Pantry.jsx) 4000, [`AgentPage.jsx`](../../frontend/src/pages/AgentPage.jsx) 3063, [`Settings.jsx`](../../frontend/src/pages/Settings.jsx) 2544. Cada uno mezcla layout + state + business logic + Supabase calls + UI. Verificable: `wc -l frontend/src/pages/*.jsx`. | Alto | L | NO refactorizar todas a la vez. Política: cada vez que toques una de estas pages, extrae la sub-feature con la que estás trabajando a `components/<page>/<Feature>.jsx`. Empezar por la siguiente página que iteres orgánicamente; aplica regla "boy scout". Una página de 5k líneas no se justifica solo desde el código, pero hace iteraciones próximas más lentas — el costo se paga en cada nuevo P-fix. | No | frontend |
| B2 | Migración a TypeScript NO recomendada ahora. React 19 + `@types/react` + `@types/react-dom` ya están en devDependencies (¡!), pero `tsconfig.json` no existe y archivos son `.jsx`. Cost: ~1-2 semanas de migration + maintenance overhead. ROI: bajo con 1 dev <100 MAU. | — | — | **No actuar**. Si en el futuro agregas un dev, evaluar entonces. Documentar decisión en CLAUDE.md sección "Decisiones de producto" análoga a P3-I18N-DEFERRED. | — | frontend |
| B3 | 245 errores ESLint + 13 warnings baseline; lint job CI con `continue-on-error: true` (P2-LIVE-1). Roadmap explícitamente prevé limpieza incremental. | Medio | M | Cleanup en pasadas dedicadas: (a) primera pasada `npm run lint -- --fix` automático para auto-fixables (no-useless-escape, sort-imports). (b) Segunda pasada por archivos pequeños primero (`utils/`, `components/common/`). (c) Una vez count < 50, considerar flippear `continue-on-error: false`. Tarea diferida: NO es urgente, pero documentar en MEMORY un floor de "limpieza próxima si pasa de N errores nuevos". | No | frontend |
| B4 | `src/pages/` y `src/components/` mezclan organización por tipo y por feature. `components/agent`, `components/assessment`, `components/dashboard`, `components/home`, `components/icons`, `components/layout`, `components/common` (parcialmente feature-based ✓). Pero state global está solo en `context/AssessmentContext.jsx` (1935 líneas, también God-component). | Medio | M | `AssessmentContext` ya hace de "store global de planes + history + form". Considerar split en 3 contexts: `PlanContext` (active plan + restoration), `FormContext` (assessment wizard state), `HistoryContext` (history cache + dirty signals). Solo prioritizar si tocas el contexto y sientes la fricción; no como audit project independiente. | No | frontend |
| B5 | PWA: SW custom en `src/custom-sw.js` con `injectManifest`. `devOptions.enabled=false` (P2-PWA-DEV-MODE bien decidido). Cache strategy NO inspeccionada en este audit (no leí custom-sw.js completo) — **requiere investigación** confirmar que (a) shopping list está disponible offline, (b) la actualización del SW no atrapa al usuario en versión vieja indefinidamente, (c) `globPatterns` no incluye archivos sensibles. | Medio | M | Leer [`custom-sw.js`](../../frontend/src/custom-sw.js) y verificar: estrategia para `/api/*` (probablemente NetworkOnly), TTL de cache para HTML/CSS, lifecycle update (`registerSW({ immediate: true })` ya está en `main.jsx:9`). Si no hay strategy explícita para shopping list offline, eso es un gap UX visible para tu target RD (móvil + conectividad inestable). | No | frontend |

### C. Performance

| # | Hallazgo | Impacto | Esfuerzo | Recomendación | Quick win | Repo |
|---|---|---|---|---|---|---|
| C1 | Bundle: `vite.config.js:96-120` ya tiene `manualChunks` (vendor-react, vendor-supabase, vendor-ui) + `chunkSizeWarningLimit: 300`. Lazy routes ya configuradas en `App.jsx:18-31`. **Falta**: `html2pdf.js` se carga lazy (P2-LAZY-PDF mencionado en CLAUDE.md) pero pesa ~976KB. `framer-motion`, `react-markdown` van en bundle vendor → primer paint paga. | Medio | S | Verificar baseline corriendo `npm run build` + `npm run check:bundle-size`. Si `vendor-ui` (framer-motion + lucide-react + sonner) > 200KB gzip, mover `framer-motion` a lazy bundle por sub-feature (motion solo en Plan + History + assessment, no en home). lucide-react: prefer treeshakeable named imports si ya no lo son. | 🟢 | frontend |
| C2 | `react-virtuoso` ✓ en deps pero **requiere investigación** si se aplica en History (5k+ planes posibles para usuarios viejos), Pantry (50-200 items), AgentPage (chat history). | Medio | S | Grep `Virtuoso` y verificar uso. Si History muestra >50 cards sin virtualización, ya es perceptible en móvil RD. | 🟢 | frontend |
| C3 | Sentry frontend: `tracesSampleRate=0.1` (env-driven ✓), pero `replaysSessionSampleRate=0.1` + `replaysOnErrorSampleRate=1.0`. Replays son los productos Sentry más caros. Con <100 MAU está bien, pero subir a 1k MAU sin bajar replaysSessionSampleRate satura cuota rápido. | Bajo | — | Mantener actual; documentar en CLAUDE.md "knobs operacionales": **antes de cruzar 500 MAU, considerar `VITE_SENTRY_REPLAYS_SESSION_RATE=0.02`**. Knob ya existe. No actuar hoy. | — | frontend |
| C4 | LLM costs: el flujo de generación llama múltiples nodos (assemble, review, retry, coherence guard, swap_meal). NO hay batching ni cache cross-request del prompt. `centralized_cache` en `cache_manager.py` se usa para embeddings (TTL 100 años en `vision_agent._cached_multimodal_embedding`) pero NO para LLM responses al plan. | Alto | M | **Prompt caching de Gemini**: revisar si los system prompts grandes (CULINARY_KNOWLEDGE_BASE, build_tools_instructions) usan Gemini cache prefix. Si no, activarlo recorta input tokens ~30-50% en el chat agent. Validable con telemetría: medir tokens_in/tokens_out por request antes/después. | No | backend |
| C5 | `bg_executor.py` ya implementa pool bounded + timeout + alerts (P1-BG-THREAD-TIMEOUT). [`cpu_tasks.py`](../cpu_tasks.py) **requiere investigación**: cualquier función CPU-bound dentro de async handler bloquea event loop. | Medio | S | Grep `from cpu_tasks` en handlers async + verificar que cada caller usa `asyncio.to_thread(...)` o `bg_executor.submit_bg_task(...)`. Si alguno corre síncrono dentro de un endpoint FastAPI async, eso es event-loop blocker bajo carga. | 🟢 | backend |
| C6 | N+1 en `db_*.py`: NO leí completos los 8 archivos pero `routers/plans.py:9760l` casi seguro contiene queries por-item en loops. Patrón de la app (history-list, lessons-counts, history-status-summary) ya optimizó N+1 a single-roundtrip, pero queda iceberg. | Medio | M | **Tarea futura**: corre `EXPLAIN ANALYZE` o instrumentación SQL en hot paths (chunk worker, history listing). Implementar un middleware simple que cuente queries por request y alerta cuando > 20. Posponer a 500+ MAU. | No | backend |

### D. Seguridad

| # | Hallazgo | Impacto | Esfuerzo | Recomendación | Quick win | Repo |
|---|---|---|---|---|---|---|
| D1 | CSP en `vercel.json:37` sigue **Report-Only**. P1-VERCEL-SECURITY-HEADERS dice "promover a enforced tras 1 semana de observación". Han pasado >1 semana. | Medio | S | Verificar el endpoint Sentry CSP Report (si configurado) o tooling externo. Si 0 violations legítimas en 7 días, flip a `Content-Security-Policy` (enforced). Si hay violations, ajustar policy primero. | 🟢 | frontend |
| D2 | `auth.py::get_verified_user_id` ✓ ya fail-secure (P0-AUDIT-1). `rate_limiter.py` ✓ tiene IP fallback (P1-6). PayPal webhook ✓ firma-verified (P0-WEBHOOK-2). Webhook facts ✓ HMAC constant-time (P0-WEBHOOK-1). | — | — | **Estado bueno**. Mantener bajo monitoreo. No actuar. | — | backend |
| D3 | `vision_agent.py:31` hardcoded `gemini-3.1-pro-preview`. Viola P3-PREVIEW-MODEL-KNOB explícito. Este mismo modelo causó CB stale 4.4 días (CLAUDE.md). Si Google deprecia el preview, vision rompe sin redeploy posible. | Medio | S | Aplicar pattern existente: `_vision_model_name()` helper que lee `MEALFIT_VISION_MODEL` (default `"gemini-3.1-pro-preview"`). Auto-registra en `_KNOBS_REGISTRY`. Test parser-based análogo a `test_p3_preview_model_knob`. | 🟢 | backend |
| D4 | `verify_api_quota` paywall: limit dict hardcoded en `auth.py:130` (`{"gratis": 15, "basic": 50, "plus": 200, "ultra": 999999, "admin": 999999}`). Cambiar pricing requiere redeploy. | Bajo | S | Mover a env vars `MEALFIT_TIER_LIMIT_GRATIS=15` etc., con auto-registry. Pricing changes via env without redeploy. | 🟢 | backend |
| D5 | `_PLAN_GEN_LIMITER = RateLimiter(max_calls=3, period_seconds=60)` en `routers/plans.py:63`. Para usuarios con plan gratis cuyo límite mensual es 15, eso permite ≤3 plan gens/min — esto es razonable. Para `ultra` (999999/mes), permite picos legítimos pero pone techo. | — | — | Mantener. | — | backend |
| D6 | Frontend usa `supabase.rpc('increment_inventory_quantity', ...)` directo (P2-4 WARN aceptado). Las únicas escrituras directas frontend a tablas user-scoped son: INSERT en `Plan.jsx:398` (¿realmente solo eso?) + DELETE en `user_inventory` Pantry. **Requiere investigación**: validar test parser-based `test_p1_new_a_frontend_no_direct_meal_plans_write.py` esté pasando y la lista de whitelist sigue siendo correcta. | Medio | S | Run `pytest backend/tests/test_p1_new_a_frontend_no_direct_meal_plans_write.py -v`. Si pasa, no actuar. Si falla, hay violation nueva. | 🟢 | backend |
| D7 | PII compliance (health_profile, peso, condiciones médicas): Sentry tiene PII scrubbing en `app.py:_sentry_redact_pii` + `main.jsx:_sentryBeforeSend` ✓. ¿Logs estructurados en backend (`logger.info`/`logger.warning`) tienen alguna garantía de no loggear health_profile completo? **Requiere investigación**. | Medio | M | Grep `logger\.(info\|warning\|error).*health_profile\|.*plan_data` en backend para auditar callsites que loguen objetos completos. Sustituir por un helper `_safe_log_user_action(action, user_id, **scrubbed_kwargs)`. | No | backend |
| D8 | `path_validators.py` existe. **Requiere investigación** sobre cobertura: ¿todos los inputs de path en uploads (vision, PDF) lo usan? ¿hay rutas que aceptan filenames del usuario sin sanitización? | Medio | S | Grep `from path_validators` + verificar todos los endpoints que aceptan filename. | 🟢 | backend |
| D9 | `rehype-sanitize` ✓ en deps + P1-MARKDOWN-SANITIZE menciona uso en chat. **Requiere investigación** si TODO output de LLM renderizado al DOM pasa por `rehype-sanitize` (recipes, expanded recipe, plan name si user-influencible). | Medio | S | Grep `rehype-sanitize` en frontend src + verificar componentes que renderizan markdown LLM. | 🟢 | frontend |

### E. Base de datos y Supabase

| # | Hallazgo | Impacto | Esfuerzo | Recomendación | Quick win | Repo |
|---|---|---|---|---|---|---|
| E1 | **Drift de migraciones entre repos**: `supabase/migrations/` raíz tiene 36 archivos; `backend/supabase/migrations/` tiene 33. Diff: raíz tiene `add_water_tracker_enabled_2026_05_16.sql`, `db_p1_chat_user_id_rls_2026_05_19.sql`, `p3_profile_numeric_coerce_2026_05_20.sql`, `p3_water_tracker_2026_05_16.sql`. Backend tiene `p1_cost_instrumentation_2026_05_15.sql` que root no. Posible doble SSOT. | Alto | S | Decidir SSOT explícito: el `.gitignore` raíz indica que workspace-root es de scope cross-repo, así que su `supabase/migrations/` debería ser canónico. Sincronizar las 4 faltantes a backend repo. Documentar en CLAUDE.md "convención de migrations: workspace-root SSOT, backend/supabase/ es symlink o sync via script". O al revés. Pero NO mantener ambos directorios. | 🟢 | ambos |
| E2 | Crons usan APScheduler con SSOT en `register_plan_chunk_scheduler` ([cron_tasks.py:793](../cron_tasks.py#L793)). Listener `_scheduler_alert_listener` escala MISSED/ERROR a `system_alerts`. **Bien diseñado**. | — | — | Mantener. | — | backend |
| E3 | RLS policies: `meal_plans` ✓ + `chunk_user_locks` ✓ + RLS forced en `meal_plans_audit` (advisor INFO intencional). ¿Otras tablas user-scoped (`user_facts`, `health_profile`, `consumed_meals`, `visual_diary`, `agent_messages`, `agent_sessions`, `nightly_rotation_queue`, `failed_inventory_deductions`)? **Requiere investigación**. | Alto | M | Listar tablas via `mcp__claude_ai_Supabase__list_tables` + cross-check con `mcp__claude_ai_Supabase__get_advisors` filtrando `rls_disabled_in_public`. Si alguna tabla user-scoped sin RLS está usando service_role bypass + no tiene capa de defensa, IDOR universal. | 🟢 | backend |
| E4 | `meal_plans_audit` table: append-only backup defensivo. `db_meal_plans_audit.py` ya tiene helper SSOT (P2-DOC-1). NO se invoca automáticamente — SOP P3-AUDIT-6 dice "SRE manual". Sin retention policy: la tabla crece sin bound. | Bajo | S | Si la tabla supera 1GB (verificar con SQL `SELECT pg_size_pretty(pg_total_relation_size('meal_plans_audit'))`), considerar cron mensual que purga > 90 días excepto `action IN ('corruption_repair','manual_rollback')`. Hoy probablemente no es problema (n_live_tup=1 según CLAUDE.md). | No | backend |
| E5 | Backups Supabase / PITR: **requiere investigación**. Supabase plan gratis ≠ PITR (solo daily backups). Plan Pro tiene 7 días. ¿Cuál plan tiene este proyecto? | Medio | — | Verificar en Supabase dashboard. Si en gratis, considerar bump a Pro cuando crucen 200 MAU pagos. | — | backend |
| E6 | Índices: muchos `unused_index` documentados como WARN aceptado (cubren FK CASCADE). ¿Hay queries lentas SIN índice apropiado? Logs/stats no inspeccionados en este audit. | Bajo | M | Habilitar `pg_stat_statements` y revisar top 10 queries por `total_exec_time`. Posponer a >500 MAU. | No | backend |

### F. Calidad de código IA / agentes

| # | Hallazgo | Impacto | Esfuerzo | Recomendación | Quick win | Repo |
|---|---|---|---|---|---|---|
| F1 | [`prompts_legacy.py`](../prompts_legacy.py) — ver A5. Posible deuda de prompts no migrados. | Bajo | S | Ver A5. | 🟢 | backend |
| F2 | Manejo de fallos LLM: `LLMCircuitBreaker` per-modelo (P1-Q3) + retry con tenacity + JSON validation + timeout duro + fail-loud cuando schema inválido. **Bien diseñado**. Cron P2-NEW-D sweep stale CB rows. | — | — | Mantener. | — | backend |
| F3 | `vision_agent.py` validación de imagen: NO se ve validación de tamaño máximo, MIME type, o malware (zip bomb, exif XSS) antes de pasar a `process_image_with_vision`. **Requiere investigación** en el callsite del endpoint upload. | Medio | S | Grep el endpoint que recibe el upload (`UploadFile`) + verificar `file.size`, `file.content_type`, `await file.read()` limit. Si falta, añadir `MAX_UPLOAD_BYTES = 5 * 1024 * 1024` (5MB) + content_type whitelist `{image/jpeg, image/png, image/webp}`. | 🟢 | backend |
| F4 | `sentiment_classifier.py` — **requiere investigación** modelo usado, costo, latencia, fallback si modelo no responde. | Bajo | S | Leer archivo + agregar telemetría latencia P95 vía pipeline_metrics. | No | backend |
| F5 | Knobs operacionales `MEALFIT_*`: auto-registry ✓, ~60+ knobs, visible en `/health/version` (admin gated). **Excelente diseño**. | — | — | Mantener. Considerar generar tabla canónica `backend/docs/knobs_registry.md` parseando el registry al boot (test parser-based asegura paridad). | No | backend |
| F6 | `fact_extractor.py` + `memory_manager.py`: tests existen pero **requiere investigación** cobertura de regresión. Estos son los datos más sensibles del producto (creencias de usuario sobre comida). Un bug que confunda facts entre usuarios sería un incident grave. | Medio | M | Verificar `tests/test_*fact*.py` y `tests/test_*memory*.py`: ¿hay tests parser-based que enforce `WHERE user_id = %s` en cada query SELECT y DELETE de facts? Si no, crítico añadirlos. | No | backend |

### G. UX y frontend específico RD

| # | Hallazgo | Impacto | Esfuerzo | Recomendación | Quick win | Repo |
|---|---|---|---|---|---|---|
| G1 | i18n: P3-I18N-DEFERRED ya cierra el gap como decisión de producto. No actuar. | — | — | — | — | — |
| G2 | Default unidades: P3-DEFAULT-IMPERIAL bumpea default a lb+ft (correcto para RD). | — | — | — | — | — |
| G3 | A11y: básico no auditado en este pass (aria-labels en common/FormUI.jsx?). **Requiere investigación**. Para target móvil RD donde usuarios pueden tener disabilities cognitivas/visuales no anticipadas. | Bajo | M | Run `axe-core` o `Lighthouse a11y` en `/dashboard` + `/plan` + `/pantry`. Si score >85 ya está bien para MVP. | No | frontend |
| G4 | PWA installable + shopping list offline: ver B5. Para tu target (móvil RD con conectividad inestable), shopping list offline es valor diferencial. | Medio | M | Confirmar comportamiento offline post-instalación. Si no funciona, gap UX significativo. | No | frontend |

### H. Observabilidad

| # | Hallazgo | Impacto | Esfuerzo | Recomendación | Quick win | Repo |
|---|---|---|---|---|---|---|
| H1 | Sentry frontend ✓ + backend ✓ (P1-SENTRY-PII-SCRUBBING-* ambos). Replays maskAllText ✓. Sampling driven from env ✓. **Bien**. | — | — | Mantener. | — | ambos |
| H2 | Logging estructurado: `logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")` — texto plano, NO json. Para grep en logs EasyPanel está bien, pero correlation_id por request no existe. Difícil seguir un user_id a través de los nodos LangGraph. | Medio | S | Añadir `correlation_id` (uuid4) por request en middleware FastAPI + propagarlo via `contextvars` al logger. Cost bajo, valor alto cuando debugees un incident con 5 callsites en cascada. | 🟢 | backend |
| H3 | Métricas de negocio: `pipeline_metrics` tabla ya existe + `_chunk_heartbeat_baseline` + alerts. Pero NO existe vista agregada "planes/día por tier", "costo LLM/usuario", "conversión registro→primer-plan→pagado". Sin esto, tienes ruido operacional pero ceguera de negocio. | Alto | M | Crear dashboard mínimo (puede ser solo views SQL en Supabase): `mv_daily_business_metrics` con (date, plans_generated, plans_completed, users_active, users_paid, llm_cost_estimate). Actualizar via cron diario. Conectar a Metabase / Grafana / un Markdown statique generado por cron. | No | backend |
| H4 | Tracing: NO hay distributed tracing entre request → agent node → tool → db. Sentry tiene `browserTracingIntegration` pero esto es solo client-side. | Bajo | L | Sentry tiene `fastapi-sentry` que ya está en `sentry-sdk[fastapi]==2.59.0` (requirements.txt:14). Verificar que `@sentry_sdk.trace` decorator esté aplicado a nodos pesados. Bajo prioridad para <100 MAU. | No | backend |

### I. Testing y CI/CD

| # | Hallazgo | Impacto | Esfuerzo | Recomendación | Quick win | Repo |
|---|---|---|---|---|---|---|
| I1 | CI workflow `.github/workflows/ci.yml` ya existe (P1-LIVE-2) y se activará al primer `git push`. Repo workspace-root NO está en git todavía según README ([`/.github/README.md`](../../.github/README.md)). Backend/frontend SÍ son repos git (`.git/` existe en cada uno). | Medio | S | Workspace-root: `git init` + `git remote add` cuando tenga sentido. Si por ahora no compartes el workspace con nadie, no es urgente. PERO el lint job y los wrappers `scripts/run_ci.{sh,ps1}` ya están listos. | No | workspace |
| I2 | Tests E2E backend marcados `@pytest.mark.e2e` excluidos en CI por filtro `-m "not e2e"`. Requieren Supabase staging. | Medio | M | Crear proyecto Supabase staging (free tier; cubre los 2 GB primeros). Añadir secrets a GitHub `SUPABASE_URL_STAGING` + `SUPABASE_KEY_STAGING`. Job CI separado `backend-e2e-staging` que solo dispare en PR a main. Cierra el último gap de cobertura real. | No | backend |
| I3 | Playwright e2e ya tiene smoke del golden-path ([`frontend/e2e/golden_path.spec.js`](../../frontend/e2e/golden_path.spec.js)) + regression guards (pageerror listener + 0 requests a fonts.gstatic.com). NO cubre flujo autenticado. | Medio | M | Cuando exista Supabase staging (I2), añadir test e2e que loguee un usuario de test → genera plan → verifica plan render. | No | frontend |
| I4 | Lint frontend non-blocking — ver B3. | — | — | — | — | — |
| I5 | Dependabot: NO veo `.github/dependabot.yml`. Sin actualizaciones automáticas, security updates de deps esperan a que alguien las note. | Medio | S | Añadir `.github/dependabot.yml` con `package-ecosystem: npm` (frontend) + `pip` (backend) + `github-actions`. Frequency `weekly`. PRs auto-creados que el CI valida. | 🟢 | workspace |
| I6 | 770 tests en backend es enorme y bueno. Pero **NO veo cobertura medida**. Sin coverage, no sabes qué zonas calientes (graph_orchestrator, cron_tasks) tienen 5% vs 80%. | Bajo | S | `pip install pytest-cov` + `pytest --cov=. --cov-report=html`. Una corrida da heatmap. Posponer reporting a CI hasta tener baseline. | 🟢 | backend |

### J. Producto y negocio

| # | Hallazgo | Impacto | Esfuerzo | Recomendación | Quick win | Repo |
|---|---|---|---|---|---|---|
| J1 | PayPal único provider. Para mercado RD donde tarjeta de crédito tiene penetración baja, eso bloquea conversión. Alternativas locales: Azul, CardNet, Pagatodo. | Alto (NEGOCIO) | L | NO actuar hoy con <100 MAU. Pero monitorear conversión: si tienes >100 registros y <10 pagos, evidencia de fricción de pago. Integración Azul tiene SDK PHP/Node pero requiere acuerdo comercial. Documentar como riesgo en este reporte (sección riesgos). | No | backend |
| J2 | Costos LLM por usuario activo: no medidos hoy. Supuesto: 1 plan = ~50k tokens (system prompt + tool calls + retry). Gemini Flash-Lite ≈ $0.075/1M input + $0.30/1M output. ~$0.02 por plan. <100 MAU × 5 planes/mes = $10/mes. **Costos asumibles**. | — | — | Mantener telemetría existente (`pipeline_metrics`). Crear vista mensual de cost si crece a 1k MAU. | — | backend |
| J3 | Funnel completo (registro → onboarding → primer plan → pago) **no instrumentado** como evento separado en analytics. Sentry + GA4 capturan eventos pero no hay tabla DB con eventos de funnel. | Medio | M | Tabla simple `funnel_events(user_id, event_type, occurred_at, metadata)` con INSERT en 4 momentos. Dashboard básico via SQL view. Crítico antes de invertir tiempo en pricing/onboarding optimizations. | No | backend |

### K. DevOps

| # | Hallazgo | Impacto | Esfuerzo | Recomendación | Quick win | Repo |
|---|---|---|---|---|---|---|
| K1 | EasyPanel + Nixpacks confirmado por el owner. Deploy-lag detector ✓ (P0-PROD-1-DEPLOY). SOP via endpoint admin. | — | — | Mantener. | — | backend |
| K2 | Vercel para frontend. CSP Report-Only (D1). HSTS + X-Frame-Options + Permissions-Policy ya en `vercel.json:8+`. | — | — | Mantener + ver D1. | — | frontend |
| K3 | Secrets management: env vars en EasyPanel + Vercel. `.env.production` en frontend repo está intencionalmente trackeado (solo VITE_* públicos, documentado en `.gitignore:31`). | — | — | Bien. Mantener. | — | ambos |
| K4 | Crons orquestador: APScheduler ✓ in-process. Riesgo: si el worker se cae, los crons se caen. EasyPanel restart pero crons que estaban "missed" se pierden. | Medio | M | APScheduler tiene `MISSED` event listener — verificar que `_scheduler_alert_listener` ([app.py:102+](../app.py#L102)) los re-encola o solo alerta. Si solo alerta sin retry, considerar `RECOVERABLE` listener custom. Postponer hasta ver casos reales. | No | backend |
| K5 | Backups EasyPanel del VPS (snapshot del disk): **requiere investigación**. Si el VPS se corrompe sin backup, perderías estado local (sqlite si se usa, logs, deferrals_pending.jsonl). | Bajo | S | Verificar en EasyPanel: ¿snapshots automáticos activos? Hetzner/DigitalOcean tienen snapshots semanales. La DB (Supabase) está separada — eso protege lo valioso. | 🟢 | devops |

---

## Top 5 Quick Wins (esta semana)

🟢 **#1 — A1: Limpiar scripts one-shot de raíz backend** (S, <1h). Mover `refactor.py`, `refactor_plans.py`, `modify_cron.py`, `recalc_now.py`, `test.py`, `test_browser_console.js` a `backend/scratch/legacy_root_helpers/`. Eliminar `test.py` (UUID PII hardcoded). Validar que `test_p1_new_a_*` y demás tests parser-based no anclan a estos paths. Reduce el ruido cognitivo cada vez que listas el directorio.

🟢 **#2 — D3: Knob para vision_agent model** (S, 30 min). Añadir `_vision_model_name()` helper en `vision_agent.py:31` + test parser-based `test_p3_vision_model_knob.py`. Cierra el gap "Google deprecates preview model + 4.4 días sin vision". Pattern existente, copy-paste de `proactive_agent._proactive_model_name`.

🟢 **#3 — E1: Resolver drift de migraciones** (S, 1h). Decidir SSOT (recomiendo workspace-root) + sync los 4 faltantes en backend/. Documentar convención en CLAUDE.md. Cierra el riesgo de aplicar migration en un repo y olvidar el otro.

🟢 **#4 — D1: Promover CSP de Report-Only a enforced** (S, 30min + 1 día observación). Si en los últimos 7 días no hay reports CSP en Sentry/logs, flip a `Content-Security-Policy` header en `vercel.json:37`. XSS defense efectiva.

🟢 **#5 — I5: Dependabot config** (S, 15min). `.github/dependabot.yml` en cada repo (workspace, backend, frontend). PRs automáticos para deps con vulnerabilidades. Solo te ocupa cuando el CI te abra un PR.

---

## Roadmap sugerido — 4 semanas

### Semana 1 (~6-8h reales)
- Quick wins #1-#5 arriba.
- **A5/F1**: Auditar [`prompts_legacy.py`](../prompts_legacy.py). Si 0 callers, mover a scratch (15 min). Si hay callers, abrir issue para migrar 1 a 1.
- **C5**: Grep `cpu_tasks` callsites async + verificar wrapping `to_thread` (30 min).
- **D6**: Run `test_p1_new_a_frontend_no_direct_meal_plans_write.py` y `test_p2_chat_cleanup.py` y `test_p3_i18n_deferred.py` localmente — confirmar 100% verde (15 min).
- **D9**: Grep `rehype-sanitize` + audit superficial componentes que rinden markdown LLM (30 min).
- **I6**: Run `pytest --cov` localmente y guardar el heatmap como referencia (sin agregar a CI todavía, 30 min).

### Semana 2 (~8-12h)
- **H2: Correlation IDs en logs backend** (~2h). Middleware FastAPI + contextvars. Una corrida de un incident con 5 callsites en cascada paga el costo.
- **F3: Validación de uploads vision_agent** (~2h). Grep el endpoint + añadir limit 5MB + content_type whitelist + test.
- **D8: Auditar path_validators cobertura** (~1h).
- **D7: Audit logs por health_profile leak** (~2h). Grep + sustituir callsites con helper safe_log.
- **E3: Audit RLS por tabla** (~2h). Listar tablas + advisors + cerrar gaps.
- **D4: Tier limits a env vars** (~30min).

### Semana 3 (~10-15h, una zona caliente)
- **A2: Extraer LLMCircuitBreaker a `cb.py`** (~6-10h). Mover ~600 líneas + re-exports + tests siguen pasando. Cierre simbólico del refactor más entryable de graph_orchestrator.
- **G3: Lighthouse a11y baseline** (~1h). Score + lista de fixes obvios.
- **H3: Vista SQL de business metrics** (~3h). Una `mv_daily_business_metrics` + view + cron diario.

### Semana 4 (~10-12h)
- **I2: Supabase staging + E2E backend en CI** (~6-8h). Crea project, secrets, job CI nuevo, ajusta filtro `not e2e`.
- **B3: ESLint pasada 1** (`--fix` + archivos pequeños) (~3h). Reducir baseline de 245 a <100.
- **C1: Bundle audit baseline + tunings ligeros** (~2h).

Si algo se desliza, deja Q1 inamovible — son los más baratos con mayor ratio impacto/esfuerzo.

---

## 3 riesgos críticos a 30 días

### R1 — Doble SSOT de migraciones (E1)
**Síntoma**: cuando un día apliques una migration al root y olvides el otro repo, una zona del código va a romper con error "column does not exist" en producción tras un redeploy del backend desde su propio git. La P0-PROD-1 deploy lag detector NO captura este caso (compara `_LAST_KNOWN_PFIX`, no schema).

**Mitigación 30 días**: resolver E1 esta semana (quick win #3). Es 1h de trabajo.

### R2 — `vision_agent` hardcoded a un modelo preview (D3)
**Síntoma**: Google deprecia `gemini-3.1-pro-preview` (estos modelos preview no tienen SLA de longevidad). Vision agent rompe sin flag escape — y no es solo "imágenes no funcionan", es que `fact_extractor` recibe contexto multimodal y la cadena entera de extracción de hechos se degrada. Ya pasó con CB stale 4.4 días.

**Mitigación 30 días**: quick win #2. Es 30 min.

### R3 — Logs leak de health_profile a Sentry / logs estructurados (D7)
**Síntoma**: con Sentry PII scrubbing solo cubres lo que llega como `event.extra`/`request.data`. Si un `logger.error(f"plan_data={plan_data}")` en un except block dumpea el objeto entero, Sentry captura el log como breadcrumb sin scrubbing key-based. Salud + nutrición es PHI/GDPR-relevant; un breach pequeño y aislado destruye trust + abre litigio.

**Mitigación 30 días**: D7 en semana 2. Sustituir 5-10 callsites de log de payload completo por un helper `_safe_log_user_action`. Cost moderado, valor compliance grande.

---

## Lo que NO recomiendo atacar ahora (decisiones explícitas)

| Item | Por qué no |
|---|---|
| Migración a TypeScript | 1 dev, <100 MAU. Cost ~1-2 semanas + maintenance overhead. ROI bajo. React 19 + `@types/react` ya están listos si se decide después. |
| Refactor `graph_orchestrator.py` completo a paquete | Cost L (semana+). Cualquier extracción que no sea quirúrgica (A2 `cb.py`) introduce riesgo de breakage por el volumen de tests parser-based que parsean source. |
| Alternativas a PayPal (Azul/CardNet) | Bloqueada por escala (cost L + acuerdo comercial). Antes hay que validar que pago es el cuello de botella de conversión real. Reabrir cuando funnel (J3) muestre evidencia. |
| Distributed tracing (Sentry traces avanzado) | <100 MAU no justifica. Postponer a 1k MAU. |
| Migrar workspace-root a git | Si no compartes con nadie, es trabajo cosmético. Cuando agregues 2do dev, se vuelve importante. |
| i18n infra | P3-I18N-DEFERRED explícito. No actuar antes de 2027-01-01 review. |

---

## Sobre cosas que NO pude verificar (requiere investigación)

Listadas aquí para no inventar hallazgos:

- **B5/G4**: PWA SW cache strategy. Necesita leer [`custom-sw.js`](../../frontend/src/custom-sw.js) completo.
- **C2**: `react-virtuoso` uso en History/Pantry/AgentPage.
- **C5**: `cpu_tasks` async wrapping en todos los callsites.
- **C6**: N+1 detalle en `routers/plans.py` (9760 líneas no se auditan en un pass).
- **D8/F3**: path_validators y vision_agent upload limits — callsite específicos.
- **D9**: rehype-sanitize cobertura en todos los markdown renders.
- **E3**: RLS por tabla (algunas listadas pero no todas verificadas).
- **E5**: Plan Supabase (gratis vs Pro) y backups PITR.
- **E6**: queries lentas via `pg_stat_statements`.
- **F4**: `sentiment_classifier.py` modelo + costo + latencia.
- **F6**: tests de regresión cobertura para facts + memory.
- **G3**: a11y score Lighthouse.
- **K5**: backups EasyPanel del VPS.

Cada uno es resoluble en <1h de investigación localizada. Ninguno es bloqueante para empezar el roadmap; se atacan progresivamente.
