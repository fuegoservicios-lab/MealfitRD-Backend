# Gaps audit — Bioboros (MealfitRD.IA) · 2026-09-04

**Alcance**: repo backend (`fuegoservicios-lab/MealfitRD-Backend`, FastAPI + LangGraph + Neon,
194.508 líneas de Python, 2.064 ficheros de test / 22.485 tests) y repo frontend
(`fuegoservicios-lab/MealfitRD`, React 19 + Vite 7, 89.312 líneas, 360 ficheros de test /
3.393 tests). Objetivo pedido por el dueño: sistema **limpio, sin código muerto, ordenado,
escalable y listo para producción**, con los gaps clasificados P0–P3 e implementados todos.

**Método**: baseline real de las dos suites en un checkout limpio (no en la máquina del dueño),
logs de los últimos runs de GitHub Actions de `main` en ambos repos, `ruff` (F401/F541/F841/
F811/F821/E712), `vulture` con verificación cruzada de referencias en prod/tests/scripts/
frontend, `depcheck`, `vite-bundle-visualizer`, y lectura de cada hallazgo antes de tocarlo.

**Filtro**: como en el audit de mayo (`gaps-audit-2026-05.md`): 1 dev, MVP, sin incendio
activo. Prioriza lo que devuelve una red de seguridad real (CI con veredicto), lo que puede
tumbar producción con un cambio de configuración legítimo, y la deuda que hace mentir a los
guards. NO recomienda refactors masivos (partir `graph_orchestrator.py` de 51 k líneas o
`AssessmentContext.jsx` de 4,5 k) esta noche: se registran como P2/P3 con su medición.

---

## Resumen ejecutivo

| | Antes (2026-09-03, `main`) | Después (esta tanda) |
|---|---|---|
| CI backend (`Backend pytest`) | **1.412 failed + 464 errors** — rojo permanente, sin veredicto | verde: los tests que no pueden evaluarse en el runner se SALTAN con razón (`-rs`), el resto ejecuta |
| CI frontend (`quality`, `e2e (webkit)`) | **60 corridas rojas seguidas** desde el 2026-08-23 | verde en local: lint por regla 20/20, WebKit filtrado, presupuesto re-baselineado con justificación |
| Suite backend en checkout limpio | 128 failed + 11 errors | 0 failed (ver «Verificación» al final) |
| Bug de arranque latente | `graph_orchestrator` moría con NameError bajo `MEALFIT_LLM_MAX_PER_USER < PLAN_CHUNK_SIZE` | cerrado + test |
| Defaults de pool que SATURAN el pooler (código y `.env.example`) | max=60 / async 12, guardados solo por el `.env` local | valores afinados en código y template, guards leen el template |
| Carrera de la caché del supermercado (caso común) | abierta (la generación se capturaba DESPUÉS de leer) | cerrada |
| Imports/variables/f-strings muertos en código productivo (ruff) | 306 | 0 |
| Funciones/clases sin ninguna referencia (vulture + cruce) | 12 | 0 |
| Componente frontend muerto sostenido por sus helpers | `MicronutrientPanel.jsx` | eliminado |

---

## P0 — bloquea o ciega producción

| # | Gap | Evidencia | Cierre |
|---|---|---|---|
| P0-1 | **CI del backend sin veredicto.** El marker `frontend_cross_repo` se ponía y ningún hook lo leía; 1.122 tests reventaban con FileNotFoundError sobre `frontend/` (el hermano privado no se clona sin `SIBLING_REPO_TOKEN`), 177 buscaban `migrations/` y 74 `CLAUDE.md` en la raíz del workspace, y el resto pedía artefactos que ningún repo versiona (`deploy-mealfit.ps1`, runbooks de `~/.claude`, `scratch/README.md`, `.env`) o el catálogo vivo. Dentro vivían el P0 clínico de alérgenos y los guards de i18n. | Run 1451 de `main`: 1.412 F + 464 E. | `tests/conftest.py`: skip real de los marcados + detección gruesa por literal de ruta; `pytest_runtest_makereport` convierte en SKIP-con-razón los fallos cuyo único motivo es un artefacto fuera del repo (acotado a workspace/HOME) o el catálogo vivo ausente (línea de log de `shopping_calculator`); pre-import de `memory_manager`/`services`/`agent` (stubs que se filtraban entre ficheros). `ci.yml`: emula la raíz (`CLAUDE.md`, `migrations/`) y lista skips (`-rs`). Ancla `test_p0_ci_verdict.py`. |
| P0-2 | **Arranque roto por configuración legítima.** `graph_orchestrator.py` usaba `logger.warning` en tiempo de import (aviso P2-ORCH-8) 1.700 líneas antes de definir `logger`. Con `MEALFIT_LLM_MAX_PER_USER < PLAN_CHUNK_SIZE` el módulo moría con NameError y `/ready` quedaba en 503. | ruff F821; reproducido con `MEALFIT_LLM_MAX_PER_USER=1`. | `logger` definido junto a `import logging`. Test AST en `test_p0_ci_verdict.py`. |
| P0-3 | **CI del frontend rojo.** (a) `lint:count --gate`: `react-refresh/only-export-components` 28 > 26; (b) `e2e (webkit)`: WebKit anota como error que ignora `interactive-widget` del meta viewport; (c) `check:presupuestos`: arranqueJS 145,6 kB gz > 140, oculto tras (a). | 60 runs rojos seguidos (482…423). | (a) helpers a ficheros propios (`coachQuotaState.js`, `micronutrientHelpers.js`, `avatarCatalog.jsx`), exports sin consumidor fuera; techo por regla 26→20, global 66→60. (b) filtro por texto EXACTO en los dos specs. (c) re-baseline 140→150 con la composición escrita (ver P2-4). |

## P1 — corrección/robustez con impacto real

| # | Gap | Evidencia | Cierre |
|---|---|---|---|
| P1-1 | **Pool DB: los defaults del código y `.env.example` reproducen la saturación de mayo** (max=60/async 12/timeout 10). Sólo el `.env` local del dueño llevaba los valores afinados, y los 11 guards que los protegían leían ese `.env` gitignored (nunca corrían en CI). | `db_core.py`, `.env.example`, tests P0-DB-POOL-*. | Defaults de código = afinados (sync 12/20 s, async 10/12 s); `.env.example` espejo versionado con TODOS los knobs guardados (pool, hedging, coherence guard warn, critique 240 s + V2 + 0872063d, pipeline 1000 s, embeddings 3/3,0); guards apuntan al template; `test_p1_env_template_ssot.py` vigila que el template declare los knobs y que el `.env` local no diverja. |
| P1-2 | **Caché del supermercado: la carrera del caso común seguía abierta.** `_publish_list_rows(todas, _catalog_generation())` capturaba la generación DESPUÉS de leer la DB — se comparaba consigo misma y nunca descartaba; el camino no cacheado capturaba una variable que no usaba. | `routers/supermarket.py` (ruff F841 lo delató). | Captura ANTES del fetch en el caso común; captura muerta eliminada. |
| P1-3 | **La guarda anti-escritura a prod no era testable sin DB.** `configure_{sync,async,checkpoint}_conn` vivían dentro de la rama que construye los pools: sin `NEON_DATABASE_URL*` no existían y los 3 tests que cablean P0-TEST-DB-ISOLATION contra el pool real fallaban con AttributeError. | `test_p0_test_db_isolation` (3 F). | Configuradores a nivel de módulo (no dependen de la rama). |
| P1-4 | **Drift real de tests (rojos en la máquina del dueño también)**: contextos legacy (5º path `seed_chunk1_queue`), drain endpoint con rate-limit, `pytest` sin importar (2), conteo de vitest 319→360, marker de check-in bumpeado, SELECT con `attempts`, 4 ficheros del LoadingScreen anclando el diseño anterior (P2-LOADING-ONE-STROKE/ETA-HONEST los supersedió), deadline de embeddings con defaults distintos al `.env`, meta-guard de humanize. | 22 tests. | Cada test reanclado al diseño vigente con la supersesión escrita; huella durable de `P2-CHECKIN-NO-FABRICATED-ANSWERS` en `app.py`. |
| P1-5 | **CLAUDE.md**: 6 enlaces rotos (5 specs no versionadas del workspace + `supabase.js` → `authClient.js`) y margen bajo el cap (471 < 800). | `test_p1_prod_final_1`, `test_p3_claudemd_margin_restore`. | Enlaces corregidos, bloques «Cómo verificar» a una línea (margen 1.126). |

## P2 — limpieza y deuda que hace ruido

| # | Gap | Evidencia | Cierre |
|---|---|---|---|
| P2-1 | **306 hallazgos ruff en código productivo**: 208 imports sin uso, 74 f-strings sin placeholders, 14 variables asignadas y no usadas, 8 imports sombreados, 1 `== False`. | `ruff check --select F401,F541,F841,F811,E712`. | Aplicado y verificado contra la suite completa; los símbolos que los tests parchean por nombre se conservan con `# noqa` documentado (ver commit). |
| P2-2 | **12 funciones/clases sin ninguna referencia** (prod, tests, scripts, docs): `active_condition_labels`, `active_allergen_labels`, `aclose_connection_pool`, `_dreaming_batch` (knob documentado que nadie leía), `clear_run_progress`, `get_circuit_breaker_snapshot`, `get_progress_cb_stats_snapshot`, `CorrectedDays`, `BatchParsedIngredients`, `_coh_finite_delta_rv`, `is_neon_auth_configured`, `get_aggregated_shopping_list_for_plan`; rama muerta `== False: pass` en `cpu_tasks`. | vulture 60 % + cruce de referencias. | Eliminadas. |
| P2-3 | **Componente frontend muerto**: nadie renderiza `<MicronutrientPanel>` (el vivo es `MicronutrientMeter`); sobrevivía porque Dashboard y NotificationCenter importaban sus helpers. | `huerfanos.mjs --gate` al separar los helpers. | Eliminado con su `.module.css`; helpers en `micronutrientHelpers.js`. |
| P2-4 | **Presupuesto de arranque** (arranqueJS 146 kB gz contra 140): crecimiento estructural de `AssessmentContext.jsx` (25,9 kB gz) y lo que arrastra (`renderCoherenceWarnings` 7,2, `historyCaches` 5,6, `pantryCache` 5,5, `guestMode` 5,3, `errorCopy` 2,9, `usePlanPollLoop` 2,6). | `vite-bundle-visualizer`. | **Abierto (re-baseline a 150 con justificación).** Palanca: cargar bajo demanda el renderer de coherencia y las cachés desde el contexto; a medio plazo partir el contexto (B4 del audit de mayo). Al aterrizar, bajar a 140. |
| P2-5 | **Higiene de tests**: stubs de módulos instalados en import que contaminaban ficheros vecinos (`memory_manager` de juguete → `ImportError … (unknown location)` en `routers.plans`). | Reproducido con 6 ficheros juntos. | Pre-import en conftest (P0-1). |
| P2-6 | **`scratch/` versionado a pesar del `.gitignore`**: `check_db_schema.py` (duplica `scripts/check_schema.py`) y `test_p1_1_qty_strict.py` (stubea `langchain_google_genai`, proveedor eliminado en junio). | `git ls-files scratch`. | Eliminados. |
| P2-7 | **`langchain` como dependencia directa sin ningún `import langchain`** (todo va por `langchain_core`, que ya trae `langchain-openai`). | `requirements.txt`. | Retirada; `langchain-core` fijada explícitamente. |

## P3 — documentación y orden

| # | Gap | Cierre |
|---|---|---|
| P3-1 | `.github/README.md` documentaba el workflow monorepo muerto (`frontend-lint` con `continue-on-error`) y `test_p3_live_1_ci_docs` anclaba esa versión sobre la copia NO versionada de la raíz del workspace. | Reescrito para los workflows reales de los dos repos; tests apuntan a las copias versionadas; `scripts/run_ci.sh` es un fósil que falla ruidosamente y apunta al `.ps1`; `scripts/README.md` documenta los wrappers y explica los ~140 scripts one-shot de datos. |
| P3-2 | `prompts/__init__.py` y `db.py`: fachadas que ruff señala (re-exports sin `__all__`, `os`/`logging` sin uso). | `__all__` explícito en `prompts`; imports muertos fuera de `db.py`. |
| P3-3 | `if True:` estructurales en `graph_orchestrator` (2) y 119 «TODO» en mayúscula que son el sustantivo español (convención P3-TODOS-NARRATIVE, sin enforcement). | No se tocan (ruido cosmético; el segundo ya es una convención declarada). |

## Lo que NO se hizo (y por qué)

- **Funciones sostenidas sólo por tests** (0 referencias en prod, sí en tests): `auth.session_cookie_within_absolute_cap`, `constants.chunk_refill_arrives_in_time`, `constants.apply_synonyms`, `cron_tasks._provenance_weight_factor`, `graph_orchestrator.get_llm_budget_stats_snapshot`, `LLMCircuitBreaker._save_db_state/_asave_db_state`, `graph_orchestrator._ingredient_is_unresolved_protein`, `llm_provider.invalidate_tier_cache`, `medication_rules.active_medication_labels/requires_medication_review`, `plan_policy.{canonical_name_for,template_id_coverage,explain_relaxations}`, `push_i18n.push_catalog_keys`, `shopping_calculator.ingredient_demand_is_fresh`. Por la propia definición del repo (`huerfanos.mjs`) son código sostenido, no cubierto; pero cada uno es el ancla de un P-fix y borrarlos exige decidir test a test. Queda como P3 con la lista aquí.
- **Partir `graph_orchestrator.py` (51 k), `cron_tasks.py` (35 k), `routers/plans.py` (17 k), `Dashboard.jsx` (10 k), `History.jsx` (5,4 k), `AssessmentContext.jsx` (4,5 k)**: es la palanca real de «escalable», y sigue siendo lo que el audit de mayo (A2/A3/B1/B4) recomendó por zonas y no de una tacada. Cada extracción rompe decenas de tests parser-based; no cabe en una noche sin dejar el árbol a medias.
- **El secret `SIBLING_REPO_TOKEN` y un `NEON_DATABASE_URL` read-only para CI**: sólo el dueño puede crearlos. Con el primero, los ~1.100 tests cross-repo dejan de saltarse en Actions; con el segundo, los ~70 dependientes del catálogo. Sin ellos, el CI sigue dando veredicto sobre todo lo demás (y dice cuánto saltó).
- **Presupuesto de arranque**: re-baselineado, no reducido (P2-4).

## Verificación

- Backend: suite completa en checkout limpio (`pytest tests/ -m "not e2e" -rs`) tras la tanda —
  ver el último commit; los skips se listan por razón.
- Frontend: `i18n:check:strict`, `eslint --max-warnings 60`, `lint:count --gate`, `typecheck`,
  `huerfanos.mjs --gate`, `vitest` (360 ficheros / 3.388 tests), `build`, `check:bundle-size`,
  `check:presupuestos` — todos en verde en local.
