# Plan ejecutable para otro agente — lotes 32 a 36 · 2026-09-13

Cinco cambios del [plan de pendientes](plan_pendientes_2026_09_11.md) que no necesitan nada del dueño. Este documento
está escrito para un agente que **no tiene el contexto de las sesiones anteriores**: trae el protocolo, las cifras medidas,
los ficheros exactos y el criterio de «terminado» de cada lote. Léelo entero antes de tocar nada. Si algo aquí contradice
el código, manda el código y se anota la discrepancia en el informe.

| Lote | Ítem del plan | Qué se entrega | Tamaño |
|---|---|---|---|
| 32 | E7 / ARQ30-P2-01 | Sacar ≥ 500 líneas (ideal ≥ 1.200) de `graph_orchestrator.py` sin cambiar conducta | 1 sesión |
| 33 | F8 | Paso del gate con el perfil de knobs de producción sobre un subconjunto curado + la batería | 1 sesión |
| 34 | F6 | 9 filas del catálogo en proxy + barrido descripción-USDA contra nombre | 1-2 sesiones |
| 35 | C7 (CUL-P2-01/02/03) | Medir variedad perceptible, presupuesto de reparación y deriva sobre el corpus fijo y el bench | 1 sesión |
| 36 | B6 · E5 · D7 | Tres mediciones que dependen de tráfico real; scripts de solo lectura | ½ sesión |

Orden recomendado: 32 primero (desbloquea cualquier cambio futuro en el god-file), luego 33 → 34 → 35 → 36. Son
independientes entre sí: si uno se atasca, se cierra lo que haya con su «lo que NO se hizo» y se sigue con el siguiente.

---

## 0. Cómo se trabaja aquí (no negociable)

### 0.1 Entorno

- Workspace: `C:\Users\angel\OneDrive\Escritorio\Nodalia\MealfitRD\Software\MealfitRD.IA` (repo raíz `fuegoservicios-lab/MealfitRD-Workspace`:
  `CLAUDE.md`, `docs/`, espejo `migrations/`). `backend/` y `frontend/` son **repos propios** con su `origin/main`.
- Python: `%USERPROFILE%\miniconda3\envs\mealfit\python.exe`. Siempre con `PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONHASHSEED=0`.
- Ficheros temporales, scripts de medición de un solo uso y logs: en el **scratchpad de tu sesión**, nunca en el repo.
  Ficheros grandes se escriben con la herramienta Write (los heredocs de Bash con barras invertidas se rompen en este
  Git Bash de Windows); las modificaciones grandes a ficheros de producción, con un script Python de parcheo que
  `assert`e que su ancla aparece exactamente una vez.
- Memoria persistente del proyecto (fuera del repo):
  `C:\Users\angel\.claude\projects\C--Users-angel-OneDrive-Escritorio-Nodalia-MealfitRD-Software-MealfitRD-IA\memory\`.

### 0.2 Prohibiciones

- **Secretos**: nunca crear, leer en claro ni pegar. El `.env` local existe y tiene `USDA_API_KEY`; se usa vía `os.environ`, no se imprime.
- **VPS**: no tocar `.env` ni systemd. El deploy es el único camino a producción.
- **Base de producción (Neon)**: los scripts abren la conexión con `conn.read_only = True` y sólo hacen SELECT. Los cambios de
  datos van **exclusivamente** por migración SSOT (§0.5). Ningún bench ni medición escribe telemetría en producción
  (el bench real suprime y cuenta las escrituras: `--telemetria-prod` las dejaría pasar — no se usa).
- **Árbol del dueño**: `10k-websites/` (untracked en la raíz) es suyo, no se toca ni se commitea. Jamás `cp` de un fichero
  entero sobre el árbol; mira `git status` antes de cada commit; commits **con rutas explícitas** (`git add <fichero>…`,
  nunca `-A`).
- **Mientras corre el gate no se edita ningún fichero del backend** (pytest los está leyendo).
- **Topes de los god-files** (`tests/test_p3_shopping_projection_pkg.py:49-52`), congelados — «extraer, no subir el tope»:
  `graph_orchestrator.py` 53.100 (hoy **53.099**), `cron_tasks.py` 36.550 (35.851), `routers/plans.py` 18.100 (17.743),
  `shopping_calculator.py` 14.400 (14.174), `plan_jobs.py` 800 (714).
- **Tests**: no `sys.path.insert(0, scripts)` (carga los scripts con `importlib.util.spec_from_file_location`). No escribir
  en un fichero de test los literales `psycopg.connect(`, `connection_pool.open(` ni `load_dotenv(`: la regla `_LOCAL_ONLY` de
  `tests/conftest.py:296-320` **salta el módulo entero** cuando no hay base de datos, y el test desaparece de la CI en
  silencio. El test del lote 28 prohíbe el literal regex `\b(INSERT INTO|UPDATE meal_plans|DELETE FROM)\b` dentro de
  `scripts/bench_superficies_culinarias.py`.
- **Decisiones del dueño** (no implementar aunque parezcan obvias): dirección V7a «el paso pide MENOS» (queda en
  `gramatical`), `MEALFIT_CULINARY_JUDGE_GUARD` off→warn (C6), fase B de E5 (cohorte), encender
  `MEALFIT_GUARD_UNDERSUPPLY_SEVERE` (D7) o `MEALFIT_TRIP_WINDOWED_PERISHABLES` (D6), fusionar filas sinónimas del catálogo.

### 0.3 El gate (mismo contrato que la CI)

La CI (`backend/.github/workflows/ci.yml`, paso «Run pytest») corre dos fases: paralela sin cuarentena y cuarentena en
serie. La lista `QUARANTINE="…"` vive en ese fichero (hoy en la línea 189; léela con `grep -n 'QUARANTINE=' .github/workflows/ci.yml`).
Reprodúcelo así, desde `backend/`, en segundo plano y con la salida a ficheros de tu scratchpad:

```bash
PY="$USERPROFILE/miniconda3/envs/mealfit/python.exe"; export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONHASHSEED=0
Q=$(grep -m1 'QUARANTINE=' .github/workflows/ci.yml | sed 's/.*QUARANTINE="\([^"]*\)".*/\1/'); IGN=""; for f in $Q; do IGN="$IGN --ignore $f"; done
"$PY" -m pytest tests/ -q --tb=short -m "not e2e" -n 3 --dist loadfile --max-worker-restart=4 $IGN -p no:cacheprovider > "$SP/gate_par.txt" 2>&1; echo "EXIT_PAR=$?" >> "$SP/gate_par.txt"
"$PY" -m pytest $Q -q --tb=short -p no:cacheprovider > "$SP/gate_ser.txt" 2>&1; echo "EXIT_SER=$?" >> "$SP/gate_ser.txt"
```

Referencia del 2026-09-13 (lote 31): paralelo **24.714 passed / 91 skipped / 2 xfailed** en ~15 min; serie **107 passed**
en ~2 min. Verde = `EXIT_PAR=0` y `EXIT_SER=0`. **Desde el lote 33 hay una tercera fase**: `python scripts/prod_profile_gate.py`
(la suite con el perfil de producción menos `tests/prod_profile_excluded.txt`, y la batería) → `EXIT_PROD=0`. Un fallo nuevo se diagnostica, no se marca `xfail`. Sondea el fichero
(`until grep -q EXIT_SER …; do sleep 20; done` con `timeout`), no hagas `sleep` a ciegas.

### 0.4 Cierre de un lote (checklist — todos, en este orden)

1. Código + tests del lote: `tests/test_p1_plan_lote_<N>.py` (obligatorio: `test_p2_hist_audit_14_marker_test_link.py`
   exige un fichero `tests/test_p1_plan_lote_<N>*.py` por marker). Cada test que parsea código de producción usa un
   **ancla** en el código (`P1-PLAN-LOTE-<N>` en un comentario o docstring) para que un renombre falle el test antes de
   cambiar producción. Estructura de referencia: `tests/test_p1_plan_lote_31.py` (docstring que cuenta el lote, fixtures
   que cargan `scripts/data/*.json`, tests de conducta + tests de contrato + un test «docs/plan/marker/anclas»).
2. Marker: `backend/app.py:166` → `_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-<N> · YYYY-MM-DD"` (formato exigido por
   `tests/test_p3_1_last_known_pfix_freshness.py`).
3. Docs: la sección del lote en el doc del dominio (§ de cada lote abajo dice cuál) **y** la fila del ítem en la tabla
   «Estado» de `docs/plan_pendientes_2026_09_11.md` (`| Ítem | Estado | Dónde |`): si la fila existe se **extiende** con
   `**\`P1-PLAN-LOTE-<N>\` (dd-mmm)**: …` (así se hizo con C2 en el lote 31); si no existe, fila nueva justo debajo de
   la cabecera. Cada fila dice qué se midió antes, qué se hizo, qué se midió después y qué NO se hizo.
4. Memoria: en `…\memory\project_plan_pendientes_lotes_2026_09_11.md` inserta un párrafo
   `**Lote <N>** (\`P1-PLAN-LOTE-<N>\` · YYYY-MM-DD): …` con su lección en cursiva, **inmediatamente antes** de la línea
   ancla `**Why:** el dueño delegó el orden;` (única en el fichero; conserva la línea en blanco). En `MEMORY.md`, primera
   viñeta: sube el contador «31 lotes cerrados» y añade el literal `` `P1-PLAN-LOTE-<N>` `` dentro de la viñeta —
   `tests/test_p3_marker_memory_xlink.py` exige que el marker aparezca **literalmente** en `MEMORY.md`.
5. Gate completo en verde (§0.3). Nada se edita mientras corre.
6. Commit en `backend/` con rutas explícitas y mensaje que cuente el lote (medido antes / qué / medido después / lo que
   no se hizo), y `git push`. Si el lote tocó `CLAUDE.md`, `docs/` o `migrations/` de la raíz: commit aparte en la raíz,
   también con rutas explícitas (deja `10k-websites/` fuera).
7. Deploy desde la raíz: `pwsh -NoProfile -File ./deploy-mealfit.ps1 backend -SkipTests` (único switch). **Empaqueta el
   árbol de trabajo**, así que `git status` del backend debe estar limpio salvo lo del dueño.
8. Verificar: `curl -s https://bioboros.com/health/version` → `last_known_pfix` = tu marker y `drift: false`. Si
   `drift: true`, sigue el SOP «resolver `deploy_lag_drift_vs_expected`» de `CLAUDE.md`; no lo ignores.
9. Informe al dueño: qué se midió, qué se hizo, cifras antes/después, lo que se dejó fuera a sabiendas y por qué, y si
   algo quedó pendiente de él (marcado 👤).

Un lote que sólo mide (sin código nuevo) igualmente lleva su script en `scripts/` (solo lectura), su test y su marker: la
medición sin instrumento reproducible no cuenta.

### 0.5 Migraciones (sólo lote 34)

Toda migración vive **en dos sitios idénticos**: `backend/migrations/` y `migrations/` (raíz). Idempotente
(`IF NOT EXISTS`, `DO $$ … RAISE EXCEPTION` de sanity). Libro `public.schema_migrations`:
`python scripts/apply_migration.py --status` (exit 4 = pendientes), `python scripts/apply_migration.py <fichero>`
(dry-run), `… --apply` (ejecuta y anota), `… --record --note "…"` (anota sin ejecutar). SOP: `docs/migrations_ledger.md`.
Patrón para el catálogo: `migrations/p1_bedca_deproxy_es_2026_08_19.sql` (fuente no-USDA en `nutrition_source_ref`,
p. ej. `bedca:2264`; `fdc_id` sólo habla de USDA; `nutrition_source` con CHECK de valores).

---

## Lote 32 · Sacar aire de `graph_orchestrator.py` (E7 / ARQ30-P2-01)

### Medido (2026-09-13)

- `graph_orchestrator.py` = **53.099 líneas** con tope 53.100 en 13 tests (`grep -rn "53100\|53_100" tests/`). El próximo
  cambio en ese fichero **no cabe**. La regla de la casa: extraer, no subir el tope.
- 577 definiciones de nivel superior y 1.062 globales de módulo. Las funciones grandes (`assemble_plan_node` 2.319 líneas,
  `review_plan_node` 1.828, `arun_plan_pipeline` 1.523) están acopladas a 50-80 símbolos del módulo: **no** son el
  objetivo de este lote.
- Hay un bloque de infraestructura de resiliencia LLM (líneas 921-2599) con acoplamiento bajo y medido:

| Bloque | Símbolos (líneas hoy) | Tamaño | Tests que lo nombran | Tests que leen el FUENTE de GO buscándolo |
|---|---|---|---|---|
| **A · concurrencia** | `DistributedLLMSemaphore` (921-1136), `LLM_SEMAPHORE`, `DistributedPerUserSemaphore` (1165-1389), `PER_USER_LLM_SEMAPHORE`, `_LLM_BUDGET_STATS`, `_LLM_BUDGET_STATS_LOCK`, `_inc_budget_stat` (1416), `get_llm_budget_stats_snapshot` (1423), `acquire_user_and_global` (1430), `aacquire_user_and_global` (1468) | ≈ 570 | 0 (semáforos), 1 (`_inc_budget_stat`) | **1**: `tests/test_p0_orch_audit_impl.py::test_p2_orch_11_local_wait_bound` (líneas 256-262) |
| **B · circuit breaker** | `_BestEffortDBCircuitBreaker` (1796-1849), `_get_be_db_cb` (1856), `_is_pool_timeout_error` (1869), `LLMCircuitOpenError` (1883), `_is_reviewer_transient_error` (1894), `_record_cb_failure_unless_transient` (1981), `LLMCircuitBreaker` (2012-2521), `_circuit_breaker` (2529), `_CIRCUIT_BREAKERS_BY_MODEL`, `_CIRCUIT_BREAKERS_LOCK`, `_get_circuit_breaker` (2562-2599) | ≈ 700 | 10 (`LLMCircuitBreaker`), 7 (`_get_circuit_breaker`) | **4**: `test_p3_new_e_cb_kv_lifecycle_doc.py` (fixture `orch_source` sobre `_ORCH`), `test_p1_besteffort_db_cb.py` (`class _BestEffortDBCircuitBreaker:`, `def _get_be_db_cb(name: str)`, `def _is_pool_timeout_error(`), `test_p0_orch_audit_impl.py:85` (`if _is_pool_timeout_error(exc):`), `test_p2_new_d_cb_kv_staleness_sweep.py:125` (sólo docstring — comprobar) |

- Dependencias externas de cada bloque (medidas por AST): A usa `asyncio`, `threading`, `time`, `uuid`,
  `contextmanager`/`asynccontextmanager`, `cache_manager.get_redis_async`, `cache_manager.redis_client` y los knobs
  `LLM_LOCAL_MAX_WAIT_S` (GO:202) y `LLM_PER_USER_LOCAL_CACHE_MAX` (GO:189). B usa además `json`,
  `knobs._env_int`, `upstream_errors._is_transient_upstream_error`, `db_core.execute_sql_query/write` y
  `aexecute_sql_query/write`, y los knobs `CB_FAILURE_THRESHOLD`, `CB_RESET_TIMEOUT_S`, `CB_LOCAL_HEALTH_TTL_S` (GO:226-230).
  Los helpers `_env_int/_env_bool/…` **ya viven en `knobs.py`** (registran en `_KNOBS_REGISTRY`): no hay import circular
  si el módulo nuevo define sus knobs con `from knobs import _env_int`.
- Importadores fuera de GO y tests (siguen funcionando si GO re-exporta): `agent.py:38` y `tools.py:36`
  (`from graph_orchestrator import _get_circuit_breaker`), `dreaming.py:492/510/516` (`LLMCircuitBreaker`),
  `plan_display_i18n.py:255`. Comentarios que citan líneas de GO: `cron_tasks.py:4226`, `dreaming.py:485`.
- Tests que parchean por nombre: sólo `agent._get_circuit_breaker` y `tools._get_circuit_breaker` (7 sitios). Ningún test
  parchea `_get_be_db_cb`, `_is_pool_timeout_error` ni `_inc_budget_stat`, que son los helpers que el código movido llama
  internamente — por eso el movimiento no rompe ningún `monkeypatch`.

### Qué hacer

1. **Bloque A → `llm_concurrency.py`** (nuevo módulo, docstring con ancla `P1-PLAN-LOTE-32`). Mueve las clases, las dos
   instancias, las estadísticas de presupuesto y los cuatro helpers **tal cual** (mismo texto, mismos comentarios; sólo
   cambian los imports). Los dos knobs pueden moverse con ellos (`from knobs import _env_int`; la línea exacta
   `LLM_LOCAL_MAX_WAIT_S        = _env_int  ("MEALFIT_LLM_LOCAL_MAX_WAIT_S",        120,` se conserva byte a byte porque
   un test la busca — ver 3). En GO queda, en el mismo sitio, un bloque comentado `[P1-PLAN-LOTE-32 · movido]` y la
   re-exportación: `from llm_concurrency import (DistributedLLMSemaphore, DistributedPerUserSemaphore, LLM_SEMAPHORE,
   PER_USER_LLM_SEMAPHORE, _LLM_BUDGET_STATS, _LLM_BUDGET_STATS_LOCK, _inc_budget_stat, get_llm_budget_stats_snapshot,
   acquire_user_and_global, aacquire_user_and_global, LLM_LOCAL_MAX_WAIT_S, LLM_PER_USER_LOCAL_CACHE_MAX)`. Los
   `_LLMBackpressureCostMixin`, `ChatGLM`, `ChatOpenAIInstrumented` (GO 1492-1612) se quedan en GO y usan los nombres
   re-exportados.
2. Gate parcial (`pytest tests/test_p0_orch_audit_impl.py tests/test_p1_redis_async_perloop_cb.py -q` y los que nombren
   los semáforos) y luego **bloque B → `llm_circuit_breaker.py`** con el mismo método. `_circuit_breaker` y el registry
   per-modelo viven en el módulo nuevo; GO re-exporta `LLMCircuitBreaker, LLMCircuitOpenError, _circuit_breaker,
   _CIRCUIT_BREAKERS_BY_MODEL, _CIRCUIT_BREAKERS_LOCK, _get_circuit_breaker, _get_be_db_cb, _is_pool_timeout_error,
   _BestEffortDBCircuitBreaker, _is_reviewer_transient_error, _record_cb_failure_unless_transient, CB_FAILURE_THRESHOLD,
   CB_RESET_TIMEOUT_S, CB_LOCAL_HEALTH_TTL_S`. Cuidado con `_is_pool_timeout_error` y `_get_be_db_cb`: GO los usa también
   fuera del bloque (caché LLM, 10 y 7 usos) — siguen resolviendo por la re-exportación.
3. **Re-apuntar los tests que leen el fuente** (el ancla se muda con el código; el test conserva su intención):
   `test_p0_orch_audit_impl.py::test_p2_orch_11_local_wait_bound` lee `_G` (GO): las tres aserciones sobre
   `_inc_budget_stat("local_wait_timeout…")` y `_deadline = time.monotonic() + LLM_LOCAL_MAX_WAIT_S` (≥ 2) pasan a leer
   `llm_concurrency.py`; `test_p3_new_e_cb_kv_lifecycle_doc.py` cambia `_ORCH` al módulo nuevo y su docstring;
   `test_p1_besteffort_db_cb.py` lee `llm_circuit_breaker.py`; `test_p0_orch_audit_impl.py:85` se comprueba (si el
   `if _is_pool_timeout_error(exc):` que busca está en la caché LLM de GO, no cambia). Actualiza los comentarios de
   `cron_tasks.py:4226` y `dreaming.py:485` para que nombren el módulo nuevo.
4. **`CLAUDE.md` (repo raíz)**, sección «Ciclo de vida del KV `llm_circuit_breaker:*`»: hoy dice que el `LLMCircuitBreaker`
   vive en `graph_orchestrator.py`; pasa a `llm_circuit_breaker.py` (y la fila del test `test_p3_new_e…`). También la
   fila «P2-01 god files» de `docs/arq30_e5_e7_diseno_canario.md` (dice «53.080 hoy»): pon la cifra nueva.
5. Si el bloque B se resiste (import circular no previsto, test que no se deja re-apuntar sin perder intención), **cierra
   el lote con A solo** y escribe por qué B no: A ya deja ~570 líneas de aire.

### Criterio de terminado

- GO **≤ 52.500** líneas con A+B (≤ 52.550 con A solo), y la aserción `n <= 52_600` (o la que corresponda al aire real
  conseguido, fijada en el test del lote) para que nadie lo vuelva a llenar sin extraer.
- `tests/test_p1_plan_lote_32.py`: (a) identidad de re-exportación (`go.LLMCircuitBreaker is llm_circuit_breaker.LLMCircuitBreaker`,
  `go.LLM_SEMAPHORE is llm_concurrency.LLM_SEMAPHORE`, …); (b) `def <nombre>(`/`class <nombre>` **ausente** del fuente de GO
  para cada símbolo movido; (c) `knobs.get_knobs_registry_snapshot()` sigue conteniendo `MEALFIT_LLM_LOCAL_MAX_WAIT_S`,
  `MEALFIT_LLM_PER_USER_LOCAL_CACHE_MAX`, `MEALFIT_CB_FAILURE_THRESHOLD`, `MEALFIT_CB_RESET_TIMEOUT_S`,
  `MEALFIT_CB_LOCAL_HEALTH_TTL_S` (ningún knob se desregistró al mudarse); (d) conducta mínima
  (`LLMCircuitBreaker(failure_threshold=1)` abre tras un fallo; `_get_circuit_breaker("x") is _get_circuit_breaker("x")`);
  (e) el tope de GO; (f) docs/plan/marker/anclas.
- Gate completo verde; **cero cambios de conducta** (la extracción es textual). Fila E7 del plan extendida; sección nueva
  en `docs/arq30_e5_e7_diseno_canario.md` (E7 · P2-01) con el mapa de lo movido y lo que queda en GO por bloque.

---

## Lote 33 · F8: paso del gate con el perfil de producción

### Medido (2026-09-11, fila F8 del plan y memoria)

- `prod_profile.py` (150 líneas) declara **30 knobs** con su valor de producción (`PROD_KNOBS`, leídos del `.env` del VPS el
  2026-09-06) más **3** que allí corren en su default `True` (`PROD_DEFAULTS_ON`: `MEALFIT_SODIUM_EXCESS_GATE`,
  `MEALFIT_RECIPE_CONTRACT_GATE`, `MEALFIT_MICRO_CLOSER_PERDAY`), `perfil_aplicado()` (context manager que restaura),
  `divergencias()`. `tests/conftest.py:16-28` fuerza con `setdefault` cinco de ellos a `false`
  (`MEALFIT_VERIFIED_INGREDIENTS_ONLY`, `MEALFIT_UPDATE_DISHES_STRICT_ALL_REASONS` y los tres gates): **una variable de
  entorno real siempre gana** al `setdefault` — así se inyecta el perfil sin tocar conftest.
- La suite entera bajo el perfil: **104 fallos de 24.270**. Los peores ficheros: `test_p1_shopping_recipe_coherence` (18),
  `test_p2_protein_yield_canonical` (14), `test_p2_guard_undersupply_canonical` (12), `test_p1_trip_windowed_perishables` (7),
  `test_p1_coherence_oversupply_staples` (7), `test_p2_country_pipeline_composition` (5). Atribución por knob en los 5 peores:
  `MEALFIT_VERIFIED_INGREDIENTS_ONLY=true` explica **56/56**; los otros knobs, 0. Son harnesses con ingredientes sintéticos
  fuera del catálogo que el filtro de verificados descarta a propósito. **No se arreglan a ciegas.**
- Ya existe «la batería»: `scripts/delivery_battery.py` + `tests/test_p1_arq27_f3_bateria.py` (aplica el perfil dentro de
  `perfil_aplicado()`, matriz de cohortes, sin LLM).

### Qué hacer

1. **Medir de nuevo, con artefacto.** En un worktree desprendido (`git worktree add <scratch>/f8 HEAD`, para no bloquear
   el árbol durante los ~15 min) corre la suite con el perfil exportado y `--junitxml`; un script
   `scripts/prod_profile_gate.py --medir <junit.xml> --out scripts/data/f8_prod_profile_<fecha>.json` resume por fichero:
   `fallos`, `pasados`, `knobs_que_explican` (re-corriendo sólo los ficheros con fallos knob a knob, como se hizo el 09-11).
   Guarda el artefacto en el repo: es la evidencia de la lista de exclusión.
2. **Lista de exclusión con motivo**: `tests/prod_profile_excluded.txt`, una línea por fichero
   `tests/<fichero>.py  # <knob que lo explica> — <por qué es harness sintético>`. Sólo entran ficheros con fallos > 0 en el
   artefacto. Es la lista «negativa»: el subconjunto curado es **la suite menos esa lista menos la cuarentena**, para que un
   test nuevo entre por defecto al paso de producción y quien quiera excluirlo tenga que escribir el motivo.
3. **Runner SSOT**: `scripts/prod_profile_gate.py` (sin flags) exporta exactamente `prod_profile.perfil_completo()` al
   entorno del subproceso, imprime la cabecera con `divergencias()` (para que quede escrito qué está midiendo), invoca
   `pytest tests/ -q --tb=short -m "not e2e" -n 3 --dist loadfile --ignore <cuarentena…> --ignore <excluidos…>` y luego
   `pytest tests/test_p1_arq27_f3_bateria.py -q`, y devuelve el exit code. No modifica `conftest.py`.
4. **Engancharlo**: tercer comando en el paso «Run pytest» de `.github/workflows/ci.yml` (después del `pytest $QUARANTINE`),
   y tercera fase del gate local (§0.3, `EXIT_PROD=`). Coste: un pase más de ~15 min; si el dueño lo considera caro, la
   alternativa es el mismo runner con `--solo <patrón>` sobre los ficheros que nombran algún knob del perfil o los módulos
   de generación — pero eso se decide con la cifra de tiempo medida, no antes.
5. **Los 104 no se tocan** en este lote. En la doc queda la regla «boy scout»: al editar un fichero de la lista de exclusión,
   se migra a nombres del catálogo real y sale de la lista.

### Criterio de terminado

- `tests/test_p1_plan_lote_33.py`: la lista de exclusión existe y cada fichero listado existe, tiene motivo y **tiene
  fallos > 0 en el artefacto** (y ningún fichero con 0 fallos está excluido); ningún excluido está también en la cuarentena;
  el runner construye el entorno igual a `perfil_completo()` (importa el runner con `importlib` y compara el dict, sin
  lanzar pytest dentro de pytest); `ci.yml` contiene el tercer comando; el doc y la fila F8 existen.
- El runner ejecutado de verdad en tu máquina termina en **0** (esa es la medición «después»: subconjunto + batería en
  verde bajo el perfil de producción). Cifras al plan: n ficheros excluidos, tiempo del pase.
- Doc: sección nueva en `docs/knobs_reference.md` (o `docs/canary_entrega_e4.md`, donde ya se describe la batería) — «El paso
  de producción del gate: qué mide, qué excluye y por qué».

---

## Lote 34 · F6: las 9 filas en proxy y el barrido descripción-USDA ↔ nombre

### Medido (2026-08-19, `docs/catalog_provenance_audit.md`)

- Regla vigente (`P1-PROVENANCE-TRUTHFUL`): **un `fdc_id` es una afirmación**; sólo lo conserva la fila cuya identidad Y
  valores coinciden con la fila real de USDA. Lo demás va a `fdc_id = NULL`, `nutrition_source = 'manual'`,
  `nutrition_source_ref = 'usda:<id> (proxy: <descripción>)'`. Fuentes no-USDA en `nutrition_source_ref` (`bedca:<id>`,
  TCAC…), con CHECK de valores en `nutrition_source`. Sinónimos que son dos filas (`Requesón`/`Queso ricotta`,
  `Judías blancas`/`Habichuelas blancas`) **no se fusionan**: el catálogo se resuelve por cadena.
- **Quedan 9 filas sobre un proxy**: chiles secos mexicanos (**chipotle, guajillo, mulato**), **Xoconostle**, embutidos
  latinos (**chorizo santarrosano, chorizo verde, longaniza puertorriqueña**), **Guineo verde** y **Requesón**. Los
  mexicanos necesitan SMAE/INSP; los embutidos, una tabla que los tenga.
- **Un `fdc_id` único mal apuntado sigue siendo invisible**: `Lomo embuchado` apuntaba a lomo crudo (110 vs 321 kcal) y lo
  destapó BEDCA, no el barrido de duplicados. El barrido que lo cazaría compara la **descripción USDA** contra el **nombre**
  (y los macros) fila a fila — barato con clave propia (`USDA_API_KEY` en `backend/.env`; `DEMO_KEY` son 30 req/h).
- Herramientas: API `https://api.nal.usda.gov/fdc/v1` (patrón en `scripts/fetch_usda_foods_2026_07_26.py:42-43`;
  `GET /food/{fdc_id}` o `POST /foods` con hasta 20 ids). `master_ingredients.name_en` existe (gloss en inglés, display-only:
  `scripts/fill_catalog_name_en.py`) y sirve para la comparación de descripciones. BEDCA: precedente
  `P1-BEDCA-DEPROXY-ES` (energía en **kJ**, `<type level="3f"/>` autocerrado). El F7 ya anotó que la TCAC 2018 del ICBF es
  un PDF escaneado sin texto; la TCAC 2015 sí se pudo extraer con Atwater como guard (4P+4C+9G dentro del 5 %).

### Qué hacer

1. **Barrido primero** (solo SELECT + red): `scripts/catalog_fdc_sweep.py` — para cada fila con `fdc_id NOT NULL` pide la
   descripción y los 4 macros a USDA (lotes de 20, respetando el rate limit; cachea la respuesta en el scratchpad),
   calcula (a) similitud descripción↔`name_en`/`name` (tokens normalizados, sin acentos, con un pequeño diccionario
   es→en para lo que `name_en` no cubra) y (b) desvío de kcal/proteína/grasa/carbohidrato en % ; escribe
   `scripts/data/catalog_fdc_sweep_<fecha>.json` con `metodo`, `n_filas`, `n_consultadas`, `n_sin_respuesta` y por fila
   `{name, fdc_id, descripcion_usda, similitud, desvio_max_pct, veredicto}` con veredicto ∈ `coincide | revisar | mal_apuntado`
   (umbrales declarados en el JSON, p. ej. `revisar` si similitud < 0,4 **o** desvío > 10 %; `mal_apuntado` si además el
   desvío > 25 %). Un 404 **no** dice que la fila esté mal: se anota `sin_respuesta`. Los `revisar` se miran **a mano**, uno
   a uno, y el veredicto final se escribe en el artefacto con la razón.
2. **Las 9 en proxy**, una por una, buscando identidad real, no parecido: Requesón → BEDCA (existe como alimento español;
   kJ→kcal); Guineo verde → tablas LATINFOODS/TCAC 2015 («banano verde») o, si no hay, se documenta que **sigue proxy** con
   la descripción USDA declarada; chiles secos → USDA tiene `Peppers, ancho, dried` / `Peppers, pasilla, dried` — sólo
   sirven si la identidad cuadra (mulato ≈ poblano seco como el ancho: **sigue siendo proxy, pero mejor declarado**);
   SMAE/INSP si se encuentra tabla con texto; embutidos latinos → si ninguna tabla los tiene, quedan proxy y se dice.
   El resultado honesto puede ser «3 resueltas con fuente, 6 siguen proxy con proxy mejor declarado»: eso es un cierre
   válido si cada fila lleva su razón.
3. **Migración** `p1_plan_lote_34_catalogo_fdc_<fecha>.sql` en **ambos** directorios (§0.5): UPDATEs por `name` exacto,
   `nutrition_source`/`nutrition_source_ref` con la fuente, `fdc_id` corregido o a NULL, `DO $$` de sanity que verifique
   el después (p. ej. que ninguna fila corregida conserve el id viejo). Dry-run, `--apply`, `--status` limpio. Ningún
   UPDATE fuera de la migración.
4. **Doc**: sección nueva en `docs/catalog_provenance_audit.md` — «Barrido descripción↔nombre (`P1-PLAN-LOTE-34`)» con la
   tabla de veredictos, las 9 filas con su destino y «lo que sigue abierto». Si la muestra tiene `mal_apuntado` reales, cada
   uno con antes/después de kcal.

### Criterio de terminado

- Artefacto del barrido en `scripts/data/` con **todas** las filas con `fdc_id` (no una muestra) y cero `revisar` sin
  veredicto final humano. Migración aplicada y anotada en el libro; espejo raíz idéntico (`diff <(ls migrations) <(ls backend/migrations)` vacío).
- `tests/test_p1_plan_lote_34.py`: el script abre la base con `read_only` (compruébalo cargando el módulo con `importlib`
  e inspeccionando el fuente, sin escribir los literales prohibidos de §0.2); el artefacto tiene la estructura y los
  umbrales declarados; la migración existe en los dos directorios y es idéntica; contiene `IF NOT EXISTS`/`DO $$`; cada
  fila tocada por la migración tiene `nutrition_source_ref` no vacío; el doc tiene la sección.
- El script se puede volver a correr sin red para la parte de clasificación (lee el caché/artefacto) — así lo pide el
  propio doc en «Cómo re-ejecutarla».

---

## Lote 35 · C7 en su parte medible: CUL-P2-01, P2-02 y P2-03

### Medido y disponible

- Backlog: `docs/audits/2026-09-07-coherencia-culinaria/BACKLOG-P0-P3.md` (repo raíz), sección «P2 — calidad sostenida al
  escalar». **P2-04 (cocinado real) y los P3** son de producto/dueño: fuera de este lote, y se dice.
- Instrumentos que ya existen, todos sin LLM: corpus fijo `scripts/data/culinary_corpus_2026_09_12.json` (5 planes de
  producción, 64 comidas; claves `planes[].{plan_id, plan_data, dias, comidas, revision, huella}`), catálogo
  `catalogo_nutricion_2026_09_12.json`, bench `scripts/bench_superficies_culinarias.py` (8 superficies; `--informe`,
  `--comparar A B`, `--planes-de <artefacto real>`; artefactos `bench_superficies_real_2026_09_13.json` con 3 planes reales
  en `planes_generados[].{perfil, plan_id, plan_data}` y coste en `coste_usd_est`, y `bench_superficies_replay_2026_09_13_l31.json`),
  `scripts/culinary_baseline.py --corpus … --congelar/--verificar`, `scripts/judge_violation_rate.py --dias`, sellos del
  contrato final por comida `_recipe_contract_final` (`modo, reescritas, familias, sin_reparar` + `lista_reescrita,
  estructura, sin_lista, repeticiones, concordancia` cuando > 0), `build_variety_report` (GO:23484, con
  `_PREP_METHOD_TOKENS`, `_MAIN_PROTEIN_ALIASES`), `horizon.repetition_limits_for`, `plan_quality_index.py`,
  `scripts/measure_deterministic_day_macros.py` (14 días deterministas sin LLM).

### Qué hacer (medir, no cambiar conducta)

1. **CUL-P2-01 variedad perceptible** — `scripts/measure_variedad_perceptible.py`: define la **firma de preparación** de
   una comida = (`_template_id` si lo hay; si no: familia de proteína por `_MAIN_PROTEIN_ALIASES`, técnicas por
   `_PREP_METHOD_TOKENS`, base de carbohidrato) y compara, en ventanas de 7/15/30 días, **nombres distintos vs firmas
   distintas** por franja. Corre sobre el corpus fijo, los 3 planes reales y un run determinista de 30 días (extiende el
   script de macros con `--dias 30` o reutiliza su ensamblador). Métricas: `nombres_nuevos`, `firmas_nuevas`,
   `renombrados` (nombre nuevo, firma repetida), `compra_reusada_con_prep_distinta` (misma proteína, técnica distinta),
   cuota de recurrencia rota por ventana (la de `repetition_limits_for`). Criterio de aceptación del backlog convertido en
   aserción del test sobre un fixture: pollo-arroz-ensalada renombrado toda la semana da `firmas_nuevas = 1`.
2. **CUL-P2-02 presupuesto de reparación** — `scripts/measure_presupuesto_reparacion.py`: por comida, número de
   reescrituras del contrato (suma de los sellos), longitud de la cadena (cuántas capas tocaron la comida en el bench:
   `etapas.entrada/salida` por check y superficie), p50/p95, % de comidas con ≥ 3 reparaciones, y coste por plan válido de
   los 3 reales (`coste_usd_est` / planes sin hallazgo abierto). Propón el knob (`MEALFIT_RECIPE_REPAIR_BUDGET`) con el
   umbral **que salga de la distribución**, no un número redondo — y **no lo implementes**: va a la hoja del dueño.
3. **CUL-P2-03 deriva** — al informe del bench (`--informe`) se le añade `--desglose pais,dieta,superficie,semana` (los
   planes del corpus llevan `plan_data.form_data`/país; si algún eje no tiene dato se declara `sin_dato`, no se inventa), y
   una línea de **cobertura** por superficie (`comidas_evaluadas / comidas`, estado del scan) para que «un timeout se vea
   como menor cobertura, nunca como mayor calidad». `judge_violation_rate.py` gana `--por-pais`. Nada de esto cambia
   veredictos.
4. Todo determinista: dos corridas seguidas sobre el corpus fijo producen JSON idénticos (huella en el artefacto).
   Artefactos en `scripts/data/` con fecha; informe en `docs/culinary_coherence.md` (sección «C7 medido —
   `P1-PLAN-LOTE-35`») con las cifras y **lo que no se mide** (P2-04, P3, calibración del juez: dueño).

### Criterio de terminado

- Tres scripts de solo lectura (offline: leen JSON, no abren la base) + artefactos + test `tests/test_p1_plan_lote_35.py`
  (fixtures pequeños para las tres métricas, determinismo, estructura de artefactos, cobertura presente en el informe,
  docs/plan/marker). Fila C7 nueva en el Estado del plan («📏 medido · 👤 presupuesto y cocinado real»). Cero cambios en
  el pipeline.

---

## Lote 36 · Tres mediciones que dependen de tráfico (B6 · E5 · D7)

Las tres son de solo lectura y pueden acabar en «sin muestra suficiente»: ese también es un resultado, con la fecha en que
volver a medir. No se «arregla» nada aquí.

| Ítem | Instrumento | Última cifra | Umbral / qué buscar |
|---|---|---|---|
| **B6** ¿bajan las ventanas de repetición rotas en el canario del día determinista? | **Falta el script**: `scripts/measure_variety_windows.py` (SELECT sobre `meal_plans.plan_data.days` de planes creados **después del 2026-09-11**; ventana de 7 días, tope `horizon.repetition_limits_for` (`balanced` = 2 veces/7 días), mismo conteo que `deterministic_day._conteo_ventana`: por `_template_id` o nombre exacto contra el registry). Identifica los días deterministas por su marca en `plan_data` (busca el sello que escribe `deterministic_day.build_day_for_skeleton`, p. ej. `_template_id`/`_candidate_source`), no por la lista de usuarios canario (vive en el `.env` del VPS, fuera de tu alcance) | 09-10, plan de 30 días del dueño: **9 ventanas rotas** con puertas apagadas, 28 encendidas (`docs/deterministic_day.md` §«Memoria entre días») | Ventanas rotas por plan y por 7 días, antes (planes ≤ 09-11) vs después. Sin planes deterministas nuevos ⇒ «sin muestra» |
| **E5** sombra de la lista canónica | `scripts/measure_canonical_shadow.py --days 30` (lee `pipeline_metrics` node `canonical_shopping_shadow`; `--offline` calcula ahora sobre planes vivos; `--json`) | 09-12: 5 planes, 731 líneas, **0 parse_fail**, 24/214 divergentes (11,2 %) — topes del agregador (rábano, tomate) y densidades, no el parser | `canonical_shopping_shadow.GATE_MIN_PLANES = 30`, `parse_fail < 1 %`, divergencia `< 5 %` (`gate_verdict`). Con < 30 planes: **NO CONCLUYENTE**, anota n |
| **D7** volumen de `magnitude_undersupply` | `scripts/measure_undersupply_volume.py --days 45` (pipeline_metrics `_shopping_coherence_alert_job_tick` + history en `meal_plans`) | 09-11: **3** en 30 días, los 3 en la ráfaga del canario 09-03/06; history viva en 6 planes | Re-medir hacia el **2026-10-10** con un mes orgánico. Recomendar el flip sólo si el volumen es «unidades», no concentrado en ráfaga, y hay history en ≥ 20 planes. Antes de esa fecha: sólo re-ejecutar y anotar |

Cierre: filas B6/E5/D7 del Estado extendidas con la cifra y la fecha; párrafo en memoria; el script nuevo de B6 con su
test (`tests/test_p1_plan_lote_36.py`: aritmética de ventanas sobre un fixture, `read_only`, estructura del artefacto) y
marker. Si E5 sigue < 30 planes y D7 antes del 10-oct, el informe lo dice tal cual: «medido, sin muestra; volver el <fecha>».

---

## Informe final al dueño (plantilla, por lote)

```
LOTE <N> · <ítem> · <marker>
Medido primero: <cifras y de dónde salen>
Qué se hizo: <3-6 líneas, sin adjetivos>
Medido después: <mismas métricas, antes → después>
Lo que NO se hizo, a sabiendas: <y por qué; qué es del dueño 👤>
Gate: <passed/skipped/xfailed> · commit <sha> · deploy verificado (<marker>, drift:false)
Lección: <una frase, sólo si la hay>
```

Y al final de los cinco: qué queda abierto del plan que **sigue** sin depender del dueño (si algo), y qué le toca a él.
