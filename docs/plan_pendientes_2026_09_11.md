# Plan de lo que falta por arreglar — BioBoros, 2026-09-11

Inventario consolidado de trabajo abierto, con fuente. Nace de la verificación de la auditoría de arquitectura (`auditoria_arquitectura_verificacion_2026_09_11.md`) y del barrido de planes, specs, docs, memoria y knobs apagados. **Nada aquí está inventado: cada línea cita dónde está escrito.** Las prioridades son una propuesta de orden; las decisiones marcadas «dueño» no las cierra código.

## Cómo leerlo

- **P0** — afecta restricciones clínicas, identidad de ingredientes, cantidades o validez de lo entregado, HOY, a usuarios reales.
- **P1** — defecto entregable acotado o gate/knob que bloquea avanzar el roadmap.
- **P2** — deuda que hace caro el siguiente cambio; medición pendiente.
- **P3** — mejoras y ampliaciones.

## Fase A · Cerrar lo que esta verificación dejó abierto (1-2 semanas)

| # | Ítem | Prio | Fuente | Notas |
|---|---|---|---|---|
| A1 | Revisar, mergear y desplegar en canario la rama `claude/auditoria-arquitectura-2026-09-11` (12 correcciones, 69 tests) | P0 | esta verificación §6 | Cambia QUÉ come el canario (3/5/6 comidas, familia, mercado). Medir `_candidate_source`, `_sodium_unknown_lines` y días caídos al LLM en `pipeline_metrics` la primera semana. |
| A2 | Auditar las 14 sondas cuando llegue `BioBoros_auditoria_arquitectura.md` | P1 | esta verificación §2 | Criterio ya escrito; sin el archivo no hay nada que auditar. |
| A3 | `_line_sodium_mg` (camino LLM) coerce «sin dato» a 0 sin marca | P1 | H6a; `graph_orchestrator.py:30275` | Exige líneas en el god-file ⇒ hacer tras A6. |
| A4 | `review_plan_node` consume `_shopping_coherence_unevaluable`; fila «no evaluable» en `docs/coherence_surfaces_table.md` (referencias de línea obsoletas) | P1 | H9 | El flag ya se persiste; falta que el veredicto y el dashboard lo vean. |
| A5 | Re-encolar la proyección al cambiar precios (`routers/supermarket.py`, migraciones de `master_ingredients`) y `householdSize` | P1 | H12 | Hoy sólo la Nevera entra en la huella. |
| A6 | Refactor mínimo del god-file: extraer `_SLOT_KEY_MAP`+`_canonical_slot_fractions` a `nutrition_calculator` y `_line_sodium_mg`/`_meal_sodium_mg` a `nutrition_db` | P1 | `graph_orchestrator.py` = 53.100 = tope | Libera líneas para A3/A4 y quita los imports perezosos desde `deterministic_day`. Solapa con ARQ30-P2-01 / Fase 9. |
| A7 | `apply_library_recipe` (camino LLM): el agua de la receta congelada no escala porque no hay factor | P2 | H4 | Derivar factor implícito (constituyente ancla) o marcar `_recipe_water_unscaled`. |
| A8 | Docs obsoletos: `plan_policy_f3.md:91` (dice abierto lo que ARQ27-P1-07 cerró), `dish_registry_f6.md:48` (`prep_minutes_est` «por técnica») | P2 | esta verificación §3 | Un doc que reabre un gap cerrado cuesta una sesión. |
| A9 | Re-firmar la revisión curatorial de los 6 perfiles tras los 88 platos veganos | dueño | `ESTADO-IMPLEMENTACION.md`; `cultural_profiles_f7.md:94-96` | `_signoff_for` devuelve `None` para todos; es fail-safe pero la firma no existe. |

## Fase B · Determinismo y personalización medibles (roadmap 2.5 F3/F4 + auditoría)

| # | Ítem | Prio | Fuente |
|---|---|---|---|
| B1 | Métrica de personalización completa: códigos `anchor_portion_*`, `prep_time_over_budget`, `culture_share_*`, `equipment_unavailable`; el score no puede ser 1,0 con `n_checks = 2` | P1 | H11; `horizon.py:1467-1472` |
| B2 | Benchmark clínico dirigido con `enforce` vivo (cuesta LLM) ⇒ promover `MEALFIT_FIDELITY_GATE` `warn→block` | P1 / dueño | roadmap Fase 3; `plan_policy_f3.md:87` |
| B3 | Huella de la computación (`computation_hash`: snapshot real, catálogo extendido, prompt, modelo, temperatura, semilla) separada de `input_hash` | P1 | H10, H13 |
| B4 | Semilla determinista en seeder y `random_seed` del prompt; `PYTHONHASHSEED` en `Procfile` | P2 | H13; `ai_helpers.py:1144…`, `graph_orchestrator.py:7584` |
| B5 | «Ninguna cocina» (perfil `neutral`), `diet.exclusions` al registry, código de relajación `culture_unavailable`; cultura por FRANJA (el blueprint ya la guarda) | P1 | H15; `cultural_profiles.py:210-211`, `horizon.py:511-515` |
| B6 | Memoria entre días en `deterministic_day` (tope semanal `balanced` roto 9 ventanas incluso con las puertas apagadas) ⇒ después decidir `MEALFIT_DETERMINISTIC_DAY_SAME_DAY_VARIETY` | P1 | `deterministic_day.py:475-489`; memoria 09-10 |
| B7 | Sesgo de macros del día determinista (proteína −18 %, carbos altos): palancas reales = scorer de `elegir_plantilla` + composición de la biblioteca (plantillas bajas en carbohidrato) | P1 / dueño | `docs/deterministic_day.md:96` |
| B8 | Flip global del día determinista (`MEALFIT_RECIPE_LIBRARY_SELECT` ya en 1; `MEALFIT_DETERMINISTIC_DAY` global) tras A1 y B6 | dueño | `docs/deterministic_day.md:12-18` |
| B9 | Embudo del wizard (Fase 4): línea base de ≥2 semanas (`pipeline_metrics.wizard_funnel`) | P2 | `plan_policy_f4.md` |

## Fase C · Coherencia culinaria (backlog CUL, 17 tareas, ninguna implementada)

| # | Ítem | Prio | Fuente |
|---|---|---|---|
| C0 | Prerrequisitos: línea base con huella de corpus FIJO (la del 6-sep dejó de ser reproducible en 14 h) y las **80 etiquetas humanas** | P0 (bloquea todo C) / dueño | `2026-09-07-coherencia-culinaria/REVISION-DE-LOS-GAPS.md` |
| C1 | CUL-P0-01 identidad/versión/estado de evaluación de cada hallazgo; CUL-P0-02 referencia independiente y scorer estricto | P0 | `BACKLOG-P0-P3.md:9,:21` |
| C2 | CUL-P0-03 contrato sobre la receta final y sus reparaciones (sustituciones que dejan técnicas del alimento anterior; frecuencia sin medir) | P0 | `BACKLOG-P0-P3.md:33` |
| C3 | CUL-P0-04 huevo entero/clara/yema no intercambiables (tres reglas que se refuerzan; `MAX_EGG_WHITES_PER_MEAL` no basta) | P0 | `BACKLOG-P0-P3.md:45` |
| C4 | Asignación paso↔ingrediente en la receta congelada (`pasos[i].usa`), V6 de «puede repartir» a suma exacta | P1 | H8; `culinary_coherence.py:1158-1161` |
| C5 | CUL-P1-01…07 (porciones de clara, estructura del plato, cultura y franja como contexto, ajustar nutrición sin desmontar la receta, cantidades/estados/tiempos/equipo ejecutables, juez que admite creatividad, benchmark en todas las superficies) | P1 | `BACKLOG-P0-P3.md:59-123` |
| C6 | `MEALFIT_CULINARY_JUDGE_GUARD` nace `off` en prod; promover a `warn` cuando C1 haga interpretable el veredicto | dueño | `graph_orchestrator.py:6443`; `prod_profile.py:42` |
| C7 | CUL-P2-01…04 y CUL-P3-01…03 (variedad perceptible 7/15/30, presupuesto de reparaciones, deriva, cocinado real; gustos de preparación, biblioteca de variantes, «magia» con usuarios) | P2-P3 | `BACKLOG-P0-P3.md:135-191` |
| C8 | 4 plantillas sin receta y 2 sustituciones con nombre heredado (Chillo⇢pescado blanco, Salami de pavo⇢jamón de pavo): un plato sustituido necesita su identidad | P1 / dueño | `docs/deterministic_day.md`; `ESTADO-IMPLEMENTACION.md` |

## Fase D · Lista de compras y Nevera

| # | Ítem | Prio | Fuente |
|---|---|---|---|
| D1 | «1 sobre» de sazón entra como 40 g (peso de la CAJA, no del sobre: `density_g_per_unit`=5) ⇒ necesidad 8× y aviso falso con 2+ sobres/semana; mueve la cantidad esperada del guard | P1 | memoria `project_aviso_capado_lee_el_envase_2026_09_11.md` #1 |
| D2 | «40 g de Cebolla en polvo» resuelve a Cebolla fresca (½ lb) | P1 | ídem #2 |
| D3 | `_apply_condiment_sanity_cap` reescribe `display_qty` pero no `display_string` | P2 | ídem #3 |
| D4 | Quién escribe el literal `30` conservando la unidad (cantidades absurdas): 3-4 comidas de 1.194; decidir barrido | P2 / dueño | `hallazgo_abierto_cantidades_absurdas_en_compra.md:112` |
| D5 | Gate de Nevera: umbral mínimo de cantidad (7 g de arroz bloquea como 200 g de pollo), unidad `malla`, 4 lecturas sueltas del flag en `_chunk_worker` | P2 | `pantry_gate_fourth_guard.md:143` |
| D6 | `MEALFIT_TRIP_WINDOWED_PERISHABLES` OFF con 4 prerrequisitos y un bug latente en esa rama (`shopping_calculator.py:13892`) | P2 | `knobs_reference.md` |
| D7 | `MEALFIT_GUARD_UNDERSUPPLY_SEVERE`: medir volumen real de `magnitude_undersupply` antes de encender (encendido ⇒ dead letter por reintento idéntico) | P2 | `shopping_calculator.py:7138` |
| D8 | Fase 5 `shopping_commercial` (marcas/presentación por producto) y medición de lag p95 de `plan_jobs` | P3 | `plan_jobs_f5.md:100` |
| D9 | Supermercado: imágenes, cadencia de precios, más familias; Fase 2 de derivación de masters (dueño) | P3 / dueño | `supermarket_db.md:99` |

## Fase E · Roadmap 2.5 sin cerrar y auditoría 2.7/3.0

| # | Ítem | Prio | Fuente |
|---|---|---|---|
| E1 | Fase 0 «higiene de entrega»: tag `baseline-v1`, retirar worktrees viejos, deploy desde tag con `git status` limpio, inventario de knobs/crons/alert_keys. **Hoy el deploy empaqueta el árbol de trabajo y hay 2 repos con cambios sin commitear** (root: 8 modificados + 2 docs; frontend: ~24 modificados) | P0 (operativo) / dueño D11 | roadmap §12 Fase 0; `git status` |
| E2 | Fase 1 gate: 2 recuperaciones simuladas con kill + 7 días sin alertas ⇒ flip `MEALFIT_INITIAL_VIA_QUEUE` | P1 | roadmap Fase 1 «Estado 2026-09-02» |
| E3 | ARQ27-P2-01 trazabilidad de checkout/evidencia (`open_confirmed`) | P2 | `GAPS.md:593` |
| E4 | ARQ27-P1-06 canario de latencia/coste + swap + último chunk punta a punta | P1 | `ESTADO-IMPLEMENTACION.md` |
| E5 | ARQ30-P1-01 promover la representación canónica a compras reales (canario por cohorte) | P1 | `arq30_f4_canonical.md` |
| E6 | ARQ30-P1-03 asignación del horizonte por comidas viables; P1-04 una autoridad para porciones y reparaciones; P1-05 commit de la revisión validada en todas las superficies | P1 | `GAPS.md:284,:307,:330` |
| E7 | ARQ30-P2-01…05 (god files, aprendizaje de preferencias con evidencia útil, disponibilidad/coste por mercado con incertidumbre — sin export privado —, latencia/coste por resultado útil, adecuación nutricional separada de cobertura) | P2 | `GAPS.md:353-441` |
| E8 | Fases 8 (medios), 9 (refactor y migraciones de contrato), 10 (benchmark final y publicación) | P3 / dueño D8, D10 | roadmap §12 |
| E9 | Cruce flojo: vegetariano · día 25 sin congelador en ES deja 1 almuerzo | P2 | `ESTADO-IMPLEMENTACION.md` |

## Fase F · Países, i18n, calidad de datos

| # | Ítem | Prio | Fuente |
|---|---|---|---|
| F1 | Aplicar en Neon `p3_country_db_check_2026_08_22.sql` (única migración sin fila en el libro) | P1 / dueño | `migrations_ledger.md:46`; G41 |
| F2 | Reconciliar el estado de G01-G96 (sólo G13/G27/G29 anotados como cerrados; la memoria registra olas que no se escribieron) | P2 | `2026-08-23-paises-gaps-tareas.md` |
| F3 | i18n v3 condición de salida: corrida real por idioma con `_display[locale]` verificado por SQL (0 de 50 planes hoy), recorrido humano en la app nativa, 3 CIs verdes | P1 | `2026-08-23-i18n-produccion-gaps-v3.md` |
| F4 | `SIBLING_REPO_TOKEN` ausente en el repo backend ⇒ su CI corre 0 tests (dos rojos latentes sólo vistos a mano) | P0 (CI) / dueño | ídem «Para el dueño» |
| F5 | Auditoría de «guards inertes» nunca ejecutada; hacerla en worktree aislado (el primer intento dejó 8 mutaciones en producción) | P1 | ídem |
| F6 | 9 filas del catálogo en proxy (chiles MX, xoconostle, embutidos latinos, guineo verde, requesón); barrido descripción-USDA vs nombre para cazar el `fdc_id` mal apuntado | P2 | `catalog_provenance_audit.md:134` |
| F7 | 5 filas sin `phosphorus_mg` que sostienen 7 constituyentes (renal los exige) | P1 | `ESTADO-IMPLEMENTACION.md` |
| F8 | 6 knobs divergen entre producción y la suite (`MEALFIT_COUNTRY_SYSTEM` prod `true`/suite `False` ⇒ ningún test vio los 5 catálogos beta) | P1 | ídem ARQ27-P1-06 |
| F9 | Validación clínica independiente: ingesta de CSV revisado por nutricionista + gate por % de aprobación (bloqueado en la primera revisión humana) | P2 / dueño | `clinical_independent_validation.md:63` |
| F10 | Marker `P1-COUNTRY-CAPS-DO-LEXICON` sin línea en `MEMORY.md` (`test_p3_marker_memory_xlink` lo acusa en cada deploy) | P2 | i18n v3 «Para el dueño» |

## Fase G · Billing, frontend, infra

| # | Ítem | Prio | Fuente |
|---|---|---|---|
| G1 | Override de importe en PayPal: en `block` sólo se marca `proven_underpaid` con cupón; sin cupón nunca se bloquea y `discount_codes` está vacío ⇒ cualquier override es ilegítimo por definición. Decidir `MEALFIT_BILLING_VERIFY_AMOUNT` + bloquear sin cupón activo | P0 / dueño | `paypal_audit_2026_08_22.md:128`; `routers/billing.py:635` |
| G2 | `plan_tier` `'free'` en esquema vs `'gratis'` en código; 6 usuarios `e2e-test-*` en producción; un `plus` sin `paypal_subscription_id` | P1 | `paypal_audit_2026_08_22.md` §2-3 |
| G3 | Borrado de cuenta no purga la identidad en Neon Auth (único `TODO` real del backend; incidente de cuenta resucitada 08-sep) | P1 | `routers/system.py:1063`; memoria 09-08 |
| G4 | Ramas sin mergear: `worktree-perf-frontend-fluidez` (entry 187→147 kB; la integra el dueño), `ci/backend-layout-fix`, `paises-ola-2` (×2), `pr8`, `wip/paused-banner-no-flash` | P1 / dueño | `git branch --no-merged main` |
| G5 | `P2-CSP-ENFORCE` (Report-Only → enforce, radio total); `P2-THEME-TOKENS-BOYSCOUT` (60 overrides) | P2 / dueño | `2026-08-14-landing-produccion-design.md` |
| G6 | Benchmark nocturno de macros auto-saltado: 4 secretos de GitHub; después refrescar `macro_baseline.json` y estrechar tolerancias | P1 / dueño (5 min) | `nightly_benchmark_activation.md` |
| G7 | iOS Fase 2 (APNs, deep links, Codemagic, ficha) tras la membresía de Apple; `P2-I18N-IOS-STORE-SOLO-ES` ya es P1 (la app está en App Store Connect) | P1 / dueño | `2026-08-21-ios-native-shell-design.md`; i18n v3 |
| G8 | Landing/comercial/legal: analítica del landing (consentimiento), oferta del 15-sep, «10 créditos» vs 15 prometidos, claims de «comida dominicana» a seis mercados, GDPR con España en venta, unidades por defecto fr/it/pt | dueño | specs landing y planes de países |

## Decisiones que sólo puede tomar el dueño (resumen)

D1-D11 del roadmap (§16), más: re-firma curatorial (A9); benchmark clínico con `enforce` (B2); 80 etiquetas humanas (C0); 4 secretos del benchmark nocturno (G6); `SIBLING_REPO_TOKEN` (F4); migración G41 (F1); override de PayPal (G1); integrar `worktree-perf-frontend-fluidez` (G4); flips `MEALFIT_INITIAL_VIA_QUEUE`, `MEALFIT_FIDELITY_GATE`, `MEALFIT_DETERMINISTIC_DAY_SAME_DAY_VARIETY`, `MEALFIT_TRIP_WINDOWED_PERISHABLES`, `MEALFIT_GUARD_UNDERSUPPLY_SEVERE`, `MEALFIT_CULINARY_JUDGE_GUARD` (cada uno con su condición previa escrita).

## Orden recomendado (si sólo hay una persona)

1. **Semana 1**: A1 (canario y medición), F4 (CI que no corre tests), G1 (override de PayPal), E1 (árboles sucios: el deploy empaqueta lo que hay), F1.
2. **Semana 2**: A6 → A3/A4, A5, C0 (línea base fija + pedir las 80 etiquetas), B6.
3. **Semanas 3-4**: B1 + B2 → gate de fidelidad a `block`; B5; C1-C3; D1-D2.
4. **Después**: B3/B4 (huella de la computación y semilla), C4/C5, E2-E6, F3/F5, y las fases 8-10 con las decisiones del dueño.
