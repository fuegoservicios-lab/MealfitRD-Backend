# Coherencia culinaria determinista — `P1-CULINARY-CONTRACT`

> Doc canónica de F1 del diseño [`2026-07-31-culinary-coherence-design.md`](../../docs/superpowers/specs/2026-07-31-culinary-coherence-design.md) (secciones 4-4c). CLAUDE.md tiene el 1-line + link; el detalle vive aquí. Golden set (F0): [`P1-CULINARY-GOLDEN`](../tests/test_p1_culinary_golden.py) + fixtures [`backend/tests/fixtures/culinary_golden/`](../tests/fixtures/culinary_golden/).

Cierra la causa raíz "el conocimiento culinario vive en ~6 tuplas hardcodeadas
reactivas" (`_NO_COOK_SAFE_PROTEIN_HINT`, `_PRECOOKED_PROTEIN_HINT`,
`_LEGUME_PROTEIN_HINT`, `_COOKED_GRAIN_REF_KCAL`, `_COOKING_VERB_RE`) con
metadata SSOT en `master_ingredients` + un validador determinista puro. Esta
doc documenta lo **construido** (Tasks 1-8 de este SDD), con sus decisiones
reales — no lo planeado por el spec original, que a veces difiere del código
final (ver "Divergencias del spec" al final de cada sección).

---

## Las 3 capas y su estado

| Capa | Qué hace | Estado |
|---|---|---|
| **Capa 1 — scan determinista** | Metadata `prep_methods`/`ready_to_eat` en `master_ingredients` + `culinary_contract_scan()` (V1 verbo↔alimento, V2 estado imposible, V3 huérfanos, V4 cantidad inconsistente ingredientes↔pasos) en 3 superficies (V4 solo en superficies 1-2, ver tabla de checks abajo). | **`warn`** (F1, esta doc). Escalada a `block` es F2 (`P1-CULINARY-CONTRACT-BLOCK`), no implementada. |
| **Capa 2 — juez LLM** | `run_culinary_judge()` (GLM flash) ve recetas completas (el reviewer médico no las ve) y detecta clases abiertas (combos raros, técnica impropia, nombre no corresponde) vía `CulinaryJudgeReport`, integrado en `review_plan_node` con history propio (`_culinary_judge_history`). | **Implementada, nace `off`** (`P1-CULINARY-JUDGE`, Tasks 11-12). Calibrada 2026-08-01 (ver sección abajo) — la corrida re-medida con la rúbrica generalizada quedó **por debajo del floor de recall** (78% < 80%); **NO autoriza aún** la escalada OFF→`warn`. **Re-calibrada 2026-08-01 post-backfill** (3 corridas + mediana) — mediana 78%, **mismo veredicto**. **Iterada 2026-08-01 v3** (ver ["Iteración de rúbrica v3 (combo_absurdo) 2026-08-01"](#iteración-de-rúbrica-v3-combo_absurdo-2026-08-01)) — `combo_absurdo` de 75/50/100 a **100% en las 3 corridas**, mediana juez sube a **89%** (flash AUTORIZA). Permanece `off` en prod — pendiente re-medir `gpt-5.6-luna` (VPS) con la rúbrica v3 antes de flippear el knob (único, no por-modelo). |
| **Capa 3 — calibración medida** | Script `scripts/calibrate_culinary_judge.py`: recall/FP por clase y por capa contra el golden set (con desglose de FP por tipo) + 1 probe held-out generado en runtime (gatea el veredicto), con llamadas LLM reales (no CI). **[P2-VEG-VOLUME-TOKENS-2 · 2026-08-01]** +2 probes informativos runtime (pescado "en anillos" / doble-grano bulgur+arroz, casos reales de las capturas del owner) — sin ground-truth etiquetado en el golden set, NO gatean el veredicto, solo dejan evidencia cruda en stdout para la próxima calibración (T14). | **Hecha** (Task 13, 2026-08-01, corregida tras review; **re-calibrada 2026-08-01 post-backfill con protocolo de 3 corridas + mediana**; **rúbrica iterada a v3 2026-08-01** tras diagnóstico estructural cruzado flash+luna sobre `combo_absurdo`). Resultado post-v3 (flash): capa1 recall 100% + 0 FP en las 3 corridas; juez recall mediana 89% (89%, 100%, 89%) + FP 0%; `combo_absurdo` 100% en las 3; probe held-out out-of-sample **cazado en las 3 corridas**. |

Rollout completo (spec §7): F0 golden set (hecho) → **F1 capa 1 en warn (hecho)** → F2 capa 1 en block (pendiente) → **F3 juez calibrado con rúbrica v3, flash AUTORIZA (89% mediana), luna pendiente de re-medición en VPS** (esta doc, 2026-08-01 — ver sección v3; el knob permanece `off` hasta confirmar luna) → F4 juez warn→block medido (pendiente; requiere primero cruzar F3 con ambos modelos, luego ≥1 semana de warn limpio en prod, Task 14).

---

## Tabla de superficies (F1)

Espejo de [`coherence_surfaces_table.md`](coherence_surfaces_table.md) (mismo patrón: tabla "negativa" — qué SÍ bloquea vs qué solo mide).

| # | Superficie | Archivo | Modo | Qué garantiza |
|---|---|---|---|---|
| 1 | `review_plan_node` | [`graph_orchestrator.py:38280+`](../graph_orchestrator.py) | gate según `MEALFIT_CULINARY_CONTRACT_GUARD` (default `warn`) | Corre DESPUÉS del AUTO-PATCH de huérfanos y AUTO-PATCH-FORWARD (dueño único: reparar → medir). Escribe `plan["_culinary_contract_violations"]` + `plan["_culinary_contract_coverage"]` siempre. En `warn` solo loggea. En `block` (no es el default) cada violación se traduce a `issues.append(...)` + `_severity_max` + **`approved = False`** (fix del [Critical de la ola post-review-final](#blockers-de-f2t10-antes-de-escalar-a-block) — hasta ese fix, `block` acumulaba `issues` pero el veredicto final del nodo nunca leía ese flag, así que NO rechazaba nada; el mismo bug existía en el gate gemelo del juez, capa 2), y por diseño F1 cae a **retry completo** — "incoherencia culinaria" no está en `_SURGICAL_REJECT_SAFE_PREFIXES` ni en `_SURGICAL_REJECT_REJUDGED_PREFIXES`, así que no hay ruta quirúrgica per-día para este gate todavía (ver "Divergencias" abajo). |
| 2 | `finalize_plan_data_coherence` | [`graph_orchestrator.py:25000`](../graph_orchestrator.py) (`_fix_refill_step_verb`, def en `:28138`) | reparación | Cierra la paridad assemble↔finalize: `_fix_refill_step_verb` (repara pasos tipo "🍚 Cuece el Casabe" → "Sirve el Casabe") corría solo en `assemble_plan_node`; ahora también corre en el loop tardío `P2-MISE-COOK-SPLIT` de finalize, en el mismo orden relativo que assemble (`_align_closer_note_food_names` → `_split_cooking_from_mise` → `_fix_refill_step_verb`, al final del trío — el defecto lo introduce un renombrado posterior al productor del paso). Idempotente + fail-safe: re-correrlo donde assemble ya lo aplicó es no-op. |
| 3 | Path degradado (`_build_filtered_edge_recipe_day`, `cron_tasks.py`) | [`cron_tasks.py:24541+`](../cron_tasks.py) | scan + reparación (única capa posible ahí) | Este path NUNCA pasa por `assemble_plan_node`/`review_plan_node` — no hay LLM por construcción. Dos mejoras: (a) el verbo del paso de Desayuno se deriva de `prep_methods[0]` real del alimento en vez de un placeholder genérico fijo ("según método tradicional"); (b) el día ENSAMBLADO pasa por `culinary_contract_scan` (solo V1/V2 — V3 se tolera, ver abajo) y `_degrade_offending_steps` degrada el paso ofensor a `"Sirve el {food}."`, acotado por `meal` exacto y con el matcher canónico `find_catalog_foods` (no substring plano). Fail-open total: si el scan revienta, el día sale tal cual — el backstop de seguridad (`P0-DEGRADED-SAFETY-SCAN`) es una capa aparte y anterior. |

**V3 (huérfanos) NO corre en la superficie 3**: los Edge Recipes son 3 pasos fijos (Mise en place / El Toque de Fuego / Montaje) que nunca listan cada ingrediente paso a paso — aplicar V3 ahí degradaría pasos que ya están bien. V1/V2 sí, porque verbo y estado se pueden falsar sin necesitar que el paso enumere cada ingrediente.

**El bloque de degradación de la superficie 3 es un no-op estructural hoy** (documentado, no un bug): el verbo del Desayuno se deriva del `prep_methods[0]` del MISMO alimento que luego valida el scan, así que nunca puede violar V1 por construcción; los pasos de Almuerzo/Cena no mencionan el nombre del alimento en el texto, así que `find_catalog_foods` no encuentra a quién acusar. Es exactamente lo que el spec pide ("última palabra por si algo se coló") — un backstop de defensa-en-profundidad para ediciones FUTURAS de las plantillas, verificado con una violación inyectada a mano (ver `test_degrade_offending_steps_matcher_canonico_no_substring`), no algo que dispare hoy con las plantillas actuales.

### Checks de la capa 1 (V1-V4)

| Check | Qué compara | Detalle | Severity |
|---|---|---|---|
| **V1** — verbo↔alimento | Cada verbo de cocción del paso (`VERB_TO_METHOD`) contra `prep_methods` del alimento que resuelve como su objeto. | Fail-open sin metadata (`prep_methods IS NULL`). | `minor` |
| **V2** — estado imposible | Menciones "(ya viene cocido)"/"(ya está cocido)" contra `ready_to_eat` del alimento. | Solo dispara si `ready_to_eat = false` explícito (NULL ⇒ fail-open). | `high` |
| **V3** — huérfanos | Cada alimento de `ingredients[]` (no exento por `CONDIMENT_EXEMPT`) contra las menciones de `recipe[]` (con fallback a prefijo, `_mencionado_por_prefijo`). | `repairable=True` (AUTO-PATCH lo elimina antes de que V3 lo mida). No corre en la superficie 3 (path degradado — Edge Recipes no listan ingrediente por ingrediente). | `minor` |
| **V4** — cantidad inconsistente (`_v4_cantidad_inconsistente`, [P1-CULINARY-CONTRACT · V4 · 2026-08-01]) | El gramaje explícito ("N g") de `ingredients[]` contra el gramaje explícito del paso que lo declara (prioridad: Mise en place → primer paso que lo declare, `_v4_grams_by_food`). SOLO gramos↔gramos — nunca convierte taza/cdta/unidad → gramos (regla dura (a)); alimento resuelto vía `find_catalog_foods`, jamás substring. | Tolerancia `V4_TOLERANCIA = 0.25` (25%, generosa a propósito — redondeos de lonjas/tazas/piezas a un número "bonito" son legítimos). **[V4-FIX3 · 2026-08-01]** Gramaje precedido de `≈`/`~` (aproximación declarada, p.ej. hints de `append_gram_hint` sobre unidades vagas lonja/pedazo/porción) se **SKIP silencioso** — no es un contrato exacto, no dispara ni se compara. `repairable=False` (un desacuerdo de cantidad no tiene reparación textual obvia sin inventar un número). No corre en la superficie 3 (mismo motivo que V3). | `minor` |

**Findings conocidos de V4** (review post-implementación, 2026-08-01):
- **El golden set no ejerce V4 con datos reales.** Los 5 planes `golden_XX_bueno` (fixtures reales contra Neon) dieron 0 falsos positivos de V4 porque ninguno de sus pasos declara gramaje explícito que diverja del de `ingredients[]` — la prueba de V4 vive enteramente en los tests sintéticos de [`test_p1_culinary_contract.py`](../tests/test_p1_culinary_contract.py) (sección V4), no en `test_p1_culinary_golden.py`. Si el golden set se regenera con un caso de divergencia real, sería la primera prueba de V4 contra datos de producción.
- **V4 no corre en la superficie 3** (path degradado, `cron_tasks.py`) por la misma razón que V3: los Edge Recipes son 3 pasos fijos que nunca declaran gramaje paso a paso. En teoría podría aplicar si un Edge Recipe alguna vez declarara gramajes en texto libre, pero hoy `_degrade_offending_steps` solo consume violaciones `V1`/`V2` — extenderlo es un cambio de una línea en el filtro que `_build_filtered_edge_recipe_day` pasa a `_degrade_offending_steps`, más decidir qué "reparación" tiene sentido para V4 (no es tan simple como degradar a "Sirve el {food}." — V4 es `repairable=False` a propósito).

---

## Blockers de F2/T10 (antes de escalar a `block`)

Registro de lo que la ola de fixes post-review-final (whole-branch, 19 commits) encontró y cerró — o dejó pendiente a propósito — antes de que F2 (`P1-CULINARY-CONTRACT-BLOCK`) pueda considerarse.

1. **[CERRADO] `block` no rechazaba nada (Critical).** Ambos gates de `review_plan_node` (capa 1 contrato y capa 2 juez) apilaban violaciones en `issues` + escalaban `severity`, pero ninguno seteaba `approved = False` — el veredicto final del nodo (`if approved: ... else: ...`) nunca leía `issues`/`severity` por su cuenta, así que `MEALFIT_CULINARY_CONTRACT_GUARD=block` (y su gemelo del juez) se comportaban EXACTAMENTE como `warn`: el plan se aprobaba igual, con las violaciones descartadas en la rama aprobada. Misma clase de bug que P1-G (`_shopping_coherence_block` sin consumer). Fix: `approved = False` añadido en ambos bloques, espejando el patrón ya usado por `_shopping_coherence_block` (`graph_orchestrator.py` ~L38441). Regresión anclada por `test_p1_culinary_block_enforcement.py` (ejecuta `review_plan_node` completo con guard=`block` + catálogo/juez mockeados, confirmado rojo contra el código pre-fix antes de mergear el fix).

2. **[CERRADO] La cobertura de T10 solo medía semana 1.** El chunk worker (`cron_tasks.py`) propagaba `_quality_degraded*` de `result` → `full_plan_data` para semanas 2+ (P2-10) pero NUNCA propagaba `_culinary_contract_violations`/`_culinary_contract_coverage`/`_culinary_judge_history` — cualquier plan multi-semana perdía la telemetría culinaria de las semanas 2+ en el overlay T2, así que una métrica de cobertura/violaciones agregada (T10, futura) habría medido solo la semana 1 y reportado una cobertura optimista. Fix: las 3 keys se propagan ahora en el mismo bloque que `_quality_degraded*` + están en `P0_4_T2_INCREMENTAL_KEYS` para sobrevivir el re-read de T2. `_culinary_judge_history` específicamente se EXTIENDE (no sobrescribe) sobre el history ya persistido de semanas previas — un overwrite ciego, como el resto de las keys de esa lista, habría perdido el history acumulado de semanas 1..N-1 cada vez que una semana N+1 completaba.

3. **[PENDIENTE — decisión de producto, no gap técnico] Acoplar los knobs, o warning de arranque si `judge=block` con `contract=off`.** Hoy `MEALFIT_CULINARY_CONTRACT_GUARD` y `MEALFIT_CULINARY_JUDGE_GUARD` son independientes — nada impide (ni avisa) si un operador setea el juez (capa 2, holístico) a `block` mientras el contrato determinista (capa 1) queda en `off`. No es un estado necesariamente incorrecto (el juez es aditivo por diseño, "jamás aprueba en silencio lo que la capa 1 ya rechazó" — pero funciona igual de bien sola), pero es una combinación que probablemente nadie eligió a propósito, dado que el rollout documentado en esta doc asume capa 1 madura ANTES que capa 2. **NO implementado en esta ola** — registrado aquí como decisión pendiente para cuando F2 (block de capa 1) esté sobre la mesa: evaluar entonces si vale un `logger.warning` de arranque (o un check en `/health/version`) cuando se detecte `judge != off and contract == off`, o si se prefiere dejarlo como combinación válida y documentada.

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_CULINARY_CONTRACT_GUARD` | `warn` | `off`/`warn`/`block`. Clamp a `warn` si el valor no es uno de los 3. Registrado vía `_env_str` → `_KNOBS_REGISTRY` ([`graph_orchestrator.py:6112`](../graph_orchestrator.py)). Escalada a `block` es F2 (`P1-CULINARY-CONTRACT-BLOCK`), pendiente. |
| `MEALFIT_CULINARY_JUDGE_GUARD` | **`off`** | `off`/`warn`/`block`. Clamp fail-safe a `off` (no a `warn`) si el valor no es uno de los 3 — a diferencia del contract-guard, esta es una llamada LLM completa por plan, así que el único default seguro ante un valor raro es apagado. Calibrado 2026-08-01: recall re-medido (78%, tras generalizar la rúbrica) queda bajo el floor de 0.80 — **permanece `off`**. Re-calibrado 2026-08-01 post-backfill (3 corridas + mediana): mediana 78%, mismo resultado — **sigue `off`**. **Rúbrica iterada a v3 2026-08-01** (`combo_absurdo`, ver sección dedicada): flash re-medido AUTORIZA (mediana 89% ≥ 0.80 en las 3 corridas) — **sigue `off` en prod** hasta que el controller re-mida `gpt-5.6-luna` con la rúbrica v3 en el VPS (el knob es único, no distingue modelo). |
| `MEALFIT_CULINARY_JUDGE_MODEL` | `_FLASH_MODEL_NAME` (GLM flash) | Directiva del owner (P1-FLASH-PRIMARY) — nunca pro sin medición. Override para A/B de modelo/costo. |
| `MEALFIT_CULINARY_JUDGE_THINKING` | `False` | Activa `extra_body.thinking` (GLM-only, ignorado en modelos OpenAI). No probado en la calibración 2026-08-01 (baseline es sin thinking). |
| `MEALFIT_CULINARY_JUDGE_TIMEOUT_S` | `45` | Clamp `[10, 120]`. Timeout de la llamada al juez; `asyncio.wait_for` externo da `+5s` de margen. |

Los 4 knobs de la Capa 2 se leen a **import-time** de `graph_orchestrator.py` (constantes módulo-level, no releídas por llamada) — un script que necesite forzar `warn` para calibrar debe escribir `os.environ` ANTES del primer `import graph_orchestrator` (ver docstring de `scripts/calibrate_culinary_judge.py`, "TRAMPA Nº1").

---

## Vocabulario `prep_methods` (SSOT)

10 métodos canónicos, el mismo set en la columna Postgres, el sanity `DO $$` de la migración, y `culinary_coherence.PREP_VOCAB`:

```
hervir | plancha | freir | hornear | guisar | saltear | licuar | tostar | crudo | ninguno
```

Mapeo verbo-de-receta → método (`VERB_TO_METHOD` en [`culinary_coherence.py`](../culinary_coherence.py)), con 2 fusiones deliberadas descubiertas por el golden set real (no por los tests unitarios con catálogo sintético — el catálogo sintético no tiene alimentos con calificador recortable ni verbos ambiguos):

- **`sofr[ií]\w*`/`dora\w*` → `saltear`** (NO `freir`/`tostar`). Sofreír cebolla/ají es la base de casi toda receta dominicana y la metadata de Vegetales lleva `saltear` pero no `freir`; "dora" (sellar carne/pollo en sartén, primer paso de casi todo guiso) NO es tostar pan/casabe — `tostar` no está en `prep_methods` de proteínas frescas. Ambas fusionadas en la MISMA clave que `saltea\w*` (no en claves separadas) porque dos claves resolviendo al mismo método duplicaban la violación V1 vía `dict.fromkeys` — sin el dedup, "Sofríe y saltea el X" producía 2 violaciones idénticas.

**Política NULL = fail-open, por check**: si un alimento no tiene `prep_methods`/`ready_to_eat` (columna `NULL`, DEFAULT de la migración), el scan se salta ESE check para ESE alimento — nunca inventa, nunca asume `false`. `scan_coverage()` mide la fracción de alimentos del plan con metadata (telemetría del rollout warn→block).

**[P1-CULINARY-METADATA-BETA · 2026-08-19] Ronda 3 — el hueco que reabrió el catálogo beta.**
Las 141 filas de países beta que `P1-COUNTRY-SYSTEM-F2` insertó el 2026-08-17 nacieron con
`prep_methods`/`ready_to_eat` en NULL al 100%, devolviendo la cobertura del catálogo de 100% a
**206/347 = 59%**. Sobre un corpus de recetas beta la cobertura medida era **24%**: un plan
dominicano no notaba nada (usa filas DO), un plan español se quedaba sin capa 1 entera. El
backfill (76 filas por default de categoría + 65 de Despensa una a una) la devuelve a 100%, y el
CHECK `master_ingredients_prep_methods_not_null` impide que el próximo lote de altas lo repita —
la invariante vive ahora donde vive el dato, no en un test parser-based. Corpus:
`tests/fixtures/culinary_beta/`. Test: `test_p1_culinary_metadata_beta.py`.

**[P1-CULINARY-HASTA-DORAR · 2026-08-19] «hasta dorar» ya no es una orden de saltear.**
`dora(?!d[oa]s?)\w*` excluía los participios («dorado/dorada») pero **no el
infinitivo**, así que «Hornea las papas hasta dorar» acusaba de salteado a todo alimento
del paso sin `saltear` en `prep_methods`. Medido sobre 33 planes REALES de producción:
**12 de 63 violaciones V1 eran esto — 19% de ruido**, y contra quien menos toca el fuego
(Aceite de oliva, Miel, Vainilla, Mango, Linaza, Plátano maduro), porque V1 acusa a
cualquier alimento nombrado en un paso largo multi-cláusula.

Importaba más de lo que parecía: en `warn` era ruido de telemetría, pero la escalada a
`block` que persigue `P1-CULINARY-CONTRACT-BLOCK` convertiría ese 19% en rechazos de
planes buenos.

El fix es `(?<!hasta )dora…`, el mismo mecanismo que el `(?<!para )horno` de la ronda
anterior y por la misma razón: una palabra que describe el envase o el PUNTO de cocción
no es una instrucción. Dos alternativas se descartaron **por medición, no por intuición**:
excluir solo el infinitivo desnudo (`|r`) caza 6 de 12 y deja pasar «hasta dorarlas»;
añadir `(?<!a )` encima no cambia ni una violación sobre datos reales. El imperativo
sigue intacto («Dora la cebolla», «Dóralo por ambos lados»): romperlo reviviría la
regresión que la Task-5 del P-fix original ya pagó. Test:
[`test_p1_culinary_hasta_dorar.py`](../tests/test_p1_culinary_hasta_dorar.py).

**Cobertura actual** (migración base, antes del backfill `leche%hervir`): `prep_methods` 148/204 filas de `master_ingredients` (~72.5%), `ready_to_eat` 99/204 (~48.5%, menor porque Vegetales/Víveres solo setean `prep_methods` por diseño del backfill de categoría). El backfill `leche%hervir` solo AÑADE un método a filas ya no-NULL (no cambia el conteo de cobertura). El resto queda `NULL` — fail-open, no un gap de esta task; un audit de huecos fuera del golden set queda pendiente para una task futura.

---

## Regla "reparar → medir → juzgar" y dueños únicos

Orden fijo (spec §9, "Oscilación reparador↔gate↔juez"): los reparadores deterministas mutan el plan PRIMERO, el scan mide el residuo DESPUÉS, y el juez (F3) solo juzga, nunca muta. Dos dueños concretos hoy:

- **AUTO-PATCH** (`review_plan_node`, bloque `"huérfanos eliminados"`) **repara** — ELIMINA ingredientes huérfanos de la lista antes de que corra el scan.
- **V3** (`_v3_huerfanos` en `culinary_coherence.py`) **mide el residuo** — corre DESPUÉS del AUTO-PATCH (y de AUTO-PATCH-FORWARD), nunca antes. Nunca los dos deciden sobre el mismo caso: si el AUTO-PATCH ya borró el huérfano, V3 no lo ve.

Mismo patrón en la superficie 2: `_fix_refill_step_verb` repara ANTES de que cualquier lint/contrato posterior mida — anclado por el orden real del trío `_align_closer_note_food_names → _split_cooking_from_mise → _fix_refill_step_verb` (ambas cadenas, assemble y finalize).

---

## Migraciones

Ambas idempotentes (`IF NOT EXISTS` / `array_append` con guard `NOT (... = ANY(...))`), copiadas byte-idénticas a `migrations/` (root) y `backend/migrations/` (P3-MIGRATIONS-SSOT), aplicadas a Neon y verificadas no-op en 2ª corrida:

1. [`p1_culinary_metadata_master_ingredients_2026_07_31.sql`](../migrations/p1_culinary_metadata_master_ingredients_2026_07_31.sql) — añade `prep_methods text[]` + `ready_to_eat boolean` (ambos `DEFAULT NULL`) a `master_ingredients`. 3 backfills en cascada (defaults por `category` → tuplas hardcodeadas históricas → ~30 overrides explícitos de casos delicados como casabe/embutidos/enlatados/legumbres), sanity `DO $$` que revienta si algún `prep_methods` sale del vocabulario canónico.
2. [`p1_culinary_metadata_leche_hervir_2026_07_31.sql`](../migrations/p1_culinary_metadata_leche_hervir_2026_07_31.sql) — hueco de metadata cazado por el golden set (no por revisión manual): el backfill por categoría dejó `Lácteos` sin `hervir`, y "Hierve la leche" (avena, café con leche) es cocina dominicana básica → falso positivo V1 real contra 4/5 planes buenos de `golden_XX_bueno`. Alcance `leche%` (8 filas), no solo la fila `Leche` que el examen pisó — cierra la misma mina en el resto de la categoría.

---

## Golden set (F0, cimiento de este scan)

10 fixtures estáticos commiteados en [`backend/tests/fixtures/culinary_golden/`](../tests/fixtures/culinary_golden/): `golden_{01..05}_bueno.json` (creatividad legítima, miden falsos positivos) + `golden_{01..05}_mutado.json` (4-6 defectos inyectados y etiquetados por `golden_manifest.json`, miden falsos negativos), cubriendo los 4 slots y ≥1 vegetariano, construidos desde la DB dominicana real. Generador one-shot `scripts/build_culinary_golden_set.py` — estáticos a propósito, el ground truth no se reescribe en silencio si cambia la DB.

**Criterios de F1** (spec §6, ya en CI, deterministas — sin flakiness):
- **0 falsos positivos**: `culinary_contract_scan` sobre los 5 `golden_XX_bueno` contra el catálogo REAL de Neon → 0 violaciones (`test_capa1_cero_fp_sobre_los_buenos`).
- **100% de las clases `capa1:*`**: cada defecto inyectado en los 5 `golden_XX_mutado` con `expected_by` empezando en `capa1:` (verbo_imposible, estado_imposible, ingrediente_huerfano, tecnica_impropia) queda atrapado por el check correcto en el día correcto (`test_capa1_atrapa_100pct_de_sus_clases`). Si un test golden falla, el fix va al scan o a la metadata — **jamás relajar el fixture**.

**Cómo re-calibrar** si el catálogo cambia y el examen vuelve a rojo: correr `pytest tests/test_p1_culinary_golden.py -q` contra Neon real (`NEON_DATABASE_URL` seteada, pool abierto — los tests golden hacen `pytest.skip` limpio sin DB, así que CI de GitHub no los ejecuta hoy). Un FP nuevo casi siempre es o (a) un hueco de metadata real → migración SSOT nueva con alcance amplio (patrón `leche%hervir`, no solo la fila puntual), o (b) un verbo ambiguo mal clasificado en `VERB_TO_METHOD` → mover de grupo con el razonamiento documentado inline (patrón `dora`/`sofr`). Nunca silenciar el fixture para que pase.

3 falsos positivos reales encontrados por el examen contra Neon (no por los tests unitarios sintéticos): `Arroz blanco` mencionado por su forma genérica "el arroz" (cerrado con `_mencionado_por_prefijo`, con guard de ambigüedad contra prefijos compartidos tipo "Ají morrón"/"Ají cubanela"), `dora` mal clasificado bajo `tostar` (cerrado moviéndolo a `saltear`), y `Leche` sin `hervir` en su metadata (cerrado con la migración `leche%hervir`, no en el scan — es un hueco de datos, no de lógica).

### Calibración capa1 2026-08-01 (plan real 165dd761, primer plan en producción, fase warn)

El primer plan real que pasó por el guard en producción midió **9 violaciones V1, las 9 falsos positivos** — de solo 2 mecanismos, ambos cerrados en `culinary_coherence.py` (P1-CULINARY-CONTRACT-FP1):

- **Clase participio/montaje (6/9):** el `\w*` genérico tras la raíz del verbo capturaba la forma PARTICIPIAL/ADJETIVAL, no solo el imperativo — «el yaniqueque HORNEADO», «pollo desmechado SALTEADO con los vegetales», «las TOSTADAS», «almendras TOSTADAS» — y las 6 vivían en pasos de "Montaje:" (que por construcción ensambla, nunca cocina). Fix: negative-lookahead de participio (`d[oa]s?\b`/`t[oa]s?\b`) en cada raíz de `VERB_TO_METHOD` cuya forma participial existe en español (hornear/guisar/saltear-sofreír-dorar/licuar/tostar) + skip explícito de pasos que empiezan con "Montaje:" en V1 (V2/V3 sin cambios — V3 sigue necesitando leer montaje para las menciones).
- **Clase ventana post-verbo (3/9):** el verbo apuntaba a un alimento NO catalogado (p.ej. "almendras" en «tuesta las almendras aparte») y, sin destinatario válido en el catálogo, la salvaguarda multi-alimento de `_v1_verbo_alimento` acusaba a los acompañantes catalogados del mismo paso (avena/leche/clara) que no eran el objeto real del verbo. Fix: veto de ventana post-verbo (`_post_verb_resolves`, ~4 palabras tras el match) cuando NINGÚN alimento del paso acepta el método — si el objeto inmediato del verbo no resuelve al catálogo, no se acusa a nadie por ese verbo.

Cerrada. Los 9 FPs (re-evaluados con catálogo sintético que reproduce el caso real) dan 0 tras el fix — casos anclados como tests sintéticos PERMANENTES en [`test_p1_culinary_contract.py`](../tests/test_p1_culinary_contract.py) (sección "FP reales 2026-08-01 plan 165dd761"), no en los fixtures del golden set (el ground truth no se toca — el golden set sigue en 100%/0FP, re-confirmado tras el fix). Traza completa por fragmento de verbo: `.superpowers/culinary-fp-round1-report.md`.

---

## Juez LLM (Capa 2, F3): rúbrica y calibración

`run_culinary_judge(plan, country="DO")` ([`graph_orchestrator.py`](../graph_orchestrator.py)) hace UNA llamada batched (no por día) a `CULINARY_JUDGE_MODEL` con `with_structured_output(CulinaryJudgeReport)`. Fail-open total: knob `off`, timeout, o cualquier excepción del LLM/parseo → `None`, nunca bloquea el plan por su cuenta. Ve la receta completa por plato (`recipe`, pasos) — el único ojo LLM del pipeline que la ve; el reviewer médico solo recibe nombre+ingredientes. `country` ([P1-COUNTRY-SYSTEM-F1 · 2026-08-16, Task 3](country_system_f1.md)) selecciona la rúbrica vía `_culinary_judge_rubric_for_country`: `'DO'`/default deja `_CULINARY_JUDGE_RUBRIC` byte-idéntico (cacheado por país); país beta sustituye "Eres un juez culinario dominicano experto" por una variante que nombra la cocina de `COUNTRY_PROFILES[cc]['name_es']` + cocina internacional.

`CulinaryViolation.tipo` acepta exactamente 5 valores canónicos: `combo_absurdo`, `tecnica_impropia`, `paso_incoherente`, `slot_inapropiado`, `nombre_no_corresponde`. La rúbrica (`_CULINARY_JUDGE_RUBRIC`, construida UNA vez a import-time para cache hits de GLM sobre el prefix estable) combina: hasta 10 nombres de ejemplo por slot desde `data/dish_templates.json`, la REGLA DURA de horario (arroz/locrio/moro/pasta nunca en desayuno/cena; sopones solo en almuerzo; postre como plato principal), la GUÍA POSITIVA por horario de `constants.SLOT_POSITIVE_HINT`, y las definiciones de los 5 tipos con ejemplos.

**Iteración de rúbrica (2026-08-01, ronda 1 — spec §6):** la corrida baseline inicial con la rúbrica original de Task 11 dio juez recall 78% + FP 16.7% (**FALLA** ambos criterios). Causa raíz de los 6 FPs: la GUÍA POSITIVA de `cena` dice "evita... los guisos pesados" — el juez la trataba como regla dura y marcaba `slot_inapropiado` sobre guisos de proteína legítimos como cena (pollo/carne/pescado guisado), que SÍ son cena dominicana real y ninguno de los 5 `golden_XX_bueno` los etiqueta como defecto. Causa raíz de los 2 misses de recall: (a) el swap `golden_02_mutado` renombra "Moro de habichuelas negras" → "Moro de guandules" **sin cambiar la categoría del plato** (sigue siendo arroz+legumbre, solo cambia la legumbre) — más sutil que los otros 3 swaps de `nombre_no_corresponde` del golden set, que sí cambian de categoría de plato; (b) el defecto de `tecnica_impropia` en `golden_03_mutado` (yogurt sobre la plancha) lo detectaba pero lo etiquetaba `paso_incoherente` — ambigüedad de frontera entre "técnica mal aplicada en un paso" y "dos pasos que se contradicen".

Fix ronda 1 (`_build_culinary_judge_rubric()`, mismo archivo): (1) la GUÍA POSITIVA se rotuló explícitamente "orientativa, NO una regla dura" + un párrafo ACLARACIÓN que dice sin ambigüedad que los guisos de proteína como cena NO son `slot_inapropiado` por sí solos; (2) `slot_inapropiado` se restringió a violar la REGLA DURA únicamente, nunca "criterio propio de ligereza"; (3) `tecnica_impropia` vs `paso_incoherente` se desambiguaron: el primero es UN paso mal aplicado al alimento (incluye alimentos que deben quedar fríos recibiendo calor), el segundo es una CONTRADICCIÓN ENTRE DOS pasos de la misma receta; (4) `nombre_no_corresponde` ganó una nota sobre el caso sutil "mismo tipo de plato, ingrediente NOMBRADO ausente".

**⚠️ Corrección post-review (misma fecha):** el punto (4) de la ronda 1, tal como se escribió originalmente, usaba el ejemplo literal **"'Moro de guandules' hecho en realidad con habichuelas negras"** — que ES, palabra por palabra, el placeholder de renombrado que usan las 4 mutaciones `nombre_no_corresponde` del golden set (`golden_manifest.json` líneas 78/118/158/198, las 4 renombran a "Moro de guandules"). La rúbrica le estaba dando al LLM la respuesta del examen, no una regla generalizable — el salto de recall de esa clase (75%→100%) medía memorización del patrón exacto, no comprensión de la regla. Reescrito a una regla genérica sin nombrar ningún plato del golden set (ver el texto actual de `_CULINARY_JUDGE_RUBRIC` arriba: "el 'X' de 'plato de X'... sustituirlo por otro de la MISMA FAMILIA... sigue siendo `nombre_no_corresponde`"), y validado con un **probe held-out generado en runtime** (ver abajo) que usa una familia de ingrediente completamente distinta (mariscos↔pollo, no legumbre↔legumbre) — el probe SÍ fue cazado correctamente, lo que confirma que la regla generaliza y no es memorización.

### Trade-off: `slot_inapropiado` restringido a la REGLA DURA

El fix de la ronda 1 (punto 2 arriba) restringe `slot_inapropiado` del juez a violar SOLO la REGLA DURA (arroz/locrio/moro/pasta fuera de su horario, sopón fuera de almuerzo, postre como plato principal). Esto fue necesario para eliminar los FPs de la corrida 1 — pero tiene un costo que debe quedar explícito antes de decidir F4: el juez **renuncia** al rol de backstop semántico sobre la lista SOFT completa de `constants.SLOT_INAPPROPRIATE_FOODS` (no inyectada al prompt del juez), que incluye reglas más finas que la REGLA DURA no cubre — entre ellas: fritura pesada de proteína como plato de cena (`pollo frito`/`chicharron`/etc.), comida de desayuno servida en la cena (cereal/panqueque/waffle/avena), un plato literalmente nombrado "Desayuno..." servido de noche, guiso pesado/legumbres en el desayuno, postre standalone como plato principal del almuerzo, plato fuerte disfrazado de merienda, y vegetales crudos como vehículo de dip (merienda americana). Estas reglas SÍ están enforzadas por el validador determinista de generación/swap (`constants.py`, consumido en los paths de day-gen y chat-modify) — el juez no es la única defensa del sistema contra ellas — pero si algún día se usa `run_culinary_judge` como backstop de un path que NO pasa por ese validador (análogo a por qué existe `P0-DEGRADED-SAFETY-SCAN` para el path degradado), esta restricción significa que el juez NO las cazaría. Decisión consciente para F3 (prioriza 0 FP sobre cobertura semántica amplia); revisitar si F4 exige que el juez cubra más que la REGLA DURA.

### Calibración 2026-08-01 (corrida final, tras la corrección post-review)

Comando: `python scripts/calibrate_culinary_judge.py` (modelo `glm-5.3-flash`, sin `--thinking`, `MEALFIT_CULINARY_JUDGE_GUARD=warn` forzado por el script). El script ahora también desglosa FP por `tipo` de violación y corre 1 probe held-out generado en runtime (11 llamadas LLM reales: 5 buenos + 5 mutados + 1 probe):

```
Modelo juez: glm-5.3-flash  thinking=False  guard=warn  timeout=45s

FP capa1 sobre buenos: 0 (criterio: 0)
FP juez  sobre buenos: 0 de 36 meals = 0.0% (criterio: <5%)
  juez   combo_absurdo            recall 2/4 = 50%
  capa1  estado_imposible         recall 5/5 = 100%
  capa1  ingrediente_huerfano     recall 5/5 = 100%
  juez   nombre_no_corresponde    recall 4/4 = 100%
  capa1  tecnica_impropia         recall 1/1 = 100%
  juez   tecnica_impropia         recall 1/1 = 100%
  capa1  verbo_imposible          recall 5/5 = 100%
  TOTAL capa1  recall 16/16 = 100%
  TOTAL juez   recall 7/9 = 78%

=== Probe held-out (generado en runtime, NO es fixture del golden set) ===
Probe: 'Sancocho de mariscos' (receta real de arroz+pollo) día 1 Almuerzo → CAZADO (nombre_no_corresponde)

Veredicto: capa1=OK  juez=FALLA  probe=OK
→ NO autoriza aún la escalada OFF→warn — ver qué falló arriba.
```

(FP juez = 0 en esta corrida, así que no hay desglose por tipo que mostrar — el script lo imprime cuando `fp_juez_by_tipo` no está vacío.)

**Veredicto honesto: NO se cumple el criterio de recall del juez en esta corrida** (78% < floor 0.80) — **la escalada OFF→`warn` NO queda autorizada**. El knob permanece `off`. capa1 sigue en 100%/0FP (contrato F1 re-confirmado). El probe held-out SÍ fue cazado correctamente (`nombre_no_corresponde` sobre "Sancocho de mariscos"/receta real de pollo, familia de ingrediente distinta a cualquier fixture del golden set) — confirma que la corrección de la rúbrica generaliza y no depende de memorizar el patrón "Moro de guandules", que era la preocupación de la ronda de review.

**Diagnóstico del miss de `combo_absurdo` (2/4, no persistido — corrida ad-hoc de verificación):** los 2 misses de esta corrida fueron `golden_02_mutado` (día 2, "Pan integral con mantequilla de maní y salami frito") y `golden_05_mutado` (día 2, "Avena cremosa con salami frito"). Ninguno de los dos es un miss reproducible: el mismo defecto de `golden_02_mutado` SÍ fue cazado en la corrida de la ronda 1 (antes de tocar el ejemplo de `nombre_no_corresponde`, que no afecta a `combo_absurdo`), y el mismo patrón exacto "avena + salami" SÍ fue cazado en `golden_01_mutado`/`golden_04_mutado` de esta MISMA corrida. Es varianza de muestreo a `temperature=0.1` (no determinista), no un gap estructural de la rúbrica — pero es una medición honesta y el resultado se reporta tal cual, sin re-rodar hasta obtener un número favorable.

Caveats:
- **N pequeño**: el total juez es 9 defectos sobre 5 planes — cada miss/hit individual mueve el recall agregado ±11 puntos. Con un floor de 0.80 y N=9, el resultado puede oscilar entre corridas por pura varianza de muestreo (78% en esta corrida, 89% en la corrida previa con la rúbrica que sobreajustaba). Antes de decidir la escalada, T14 debería promediar varias corridas o ampliar el golden set — un solo run cerca del floor no es suficiente evidencia en ningún sentido.
- **Costo real acumulado de toda la sesión de calibración** (ronda 1 + corrección + verificaciones ad-hoc + corrida final + probe): **63 llamadas**, **$0.0119** vía `llm_usage_events` (`node='culinary_judge'`, `model='glm-5.3-flash'`: 221 667 input tokens, 173 184 cache hit — ~78% del prefix de la rúbrica cacheado por GLM — + 16 783 output tokens). Muy por debajo del estimado "centavos" del brief. Una corrida real de 11 llamadas (el tamaño del script con el probe incluido) cuesta una fracción de centavo.
- **`--thinking` no se probó** en esta calibración — la baseline es sin razonamiento extendido (consistente con la decisión general del owner P1-DAYGEN-TIER-MODEL de no usar thinking en nodos de red/apoyo salvo medición explícita).

### Re-calibración 2026-08-01 (post-backfill)

**Contexto:** el backfill ronda 2 de metadata culinaria (`P2-CULINARY-METADATA-ROUND2`) cerró la cobertura de `master_ingredients` a 100% (antes ~72.5%/48.5% `prep_methods`/`ready_to_eat`, ver "Vocabulario `prep_methods` (SSOT)" arriba). Ese backfill solo puede mover capa1 (el scan determinista lee esa metadata) — el juez (capa2) nunca la lee, así que a priori no debía mover su recall. El script también ganó 2 probes informativos nuevos desde la corrida anterior (`(a)` pescado "en anillos", `(b)` doble-grano bulgur+arroz, `P2-VEG-VOLUME-TOKENS-2`).

Dado el N pequeño ya documentado (9 defectos-juez, cada hit/miss mueve el recall agregado ±11pp), esta re-calibración corre el script **3 veces bajo las mismas condiciones** (`glm-5.3-flash`, sin `--thinking`, `MEALFIT_CULINARY_JUDGE_GUARD=warn` forzado por el script) y usa la **mediana** de las 3 como criterio de decisión — nunca una corrida aislada.

⚠️ **Nota operacional (Windows, no es un bug de la rúbrica):** la primera invocación crasheó con `UnicodeEncodeError` al imprimir el separador `→` bajo la consola cp1252 por defecto de PowerShell/Git Bash en Windows — el juez ya había respondido (10-11 llamadas ya facturadas) pero el script murió antes de imprimir el probe held-out y los 2 probes informativos. Se resuelve con `PYTHONIOENCODING=utf-8` antes de invocar el script (no se tocó el script — es un gotcha de terminal, no de lógica). Esa corrida parcial se **descarta** de las 3 mediciones oficiales (nunca completó los probes) pero sus llamadas sí se facturaron y están incluidas en el costo total de abajo.

Comando (×3, mismas condiciones): `PYTHONIOENCODING=utf-8 python scripts/calibrate_culinary_judge.py`

| Corrida | capa1 recall | capa1 FP | juez recall | juez FP | `combo_absurdo` | `nombre_no_corresponde` | `tecnica_impropia` | probe held-out (gating) | probe (a) anillos (informativo) | probe (b) doble-grano (informativo) | veredicto script |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16/16 = 100% | 0 | 7/9 = 78% | 0/36 = 0% | 3/4 = 75% | 3/4 = 75% | 1/1 = 100% | CAZADO (`nombre_no_corresponde`) | NO CAZADO | CAZADO (`paso_incoherente`) | capa1=OK juez=FALLA probe=OK → NO autoriza |
| 2 | 16/16 = 100% | 0 | 7/9 = 78% | 0/36 = 0% | 2/4 = 50% | 4/4 = 100% | 1/1 = 100% | CAZADO (`nombre_no_corresponde`) | NO CAZADO | CAZADO (`paso_incoherente`) | capa1=OK juez=FALLA probe=OK → NO autoriza |
| 3 | 16/16 = 100% | 0 | 8/9 = 89% | 0/36 = 0% | 4/4 = 100% | 3/4 = 75% | 1/1 = 100% | CAZADO (`nombre_no_corresponde`) | NO CAZADO | CAZADO (`paso_incoherente`) | capa1=OK juez=OK probe=OK → **AUTORIZA** |
| **Mediana** | **100%** | **0** | **78% (7/9)** | **0%** | 75% | 75% | 100% | CAZADO (3/3) | NO CAZADO (3/3) | CAZADO (3/3) | — |

**Capa1 (contrato F1): recall 1.00 + 0 FP en las 3 corridas** — re-confirmado tras el backfill ronda 2, sin cambios respecto a la corrida anterior. ✓ PASA.

**Juez (capa2): recall mediana 78% (7/9) < floor 0.80** — la MISMA cifra que la corrida original (78%), reproducida de nuevo después del backfill. La varianza ya documentada en la corrida previa sigue intacta: 2 de 3 corridas caen en 78%, 1 en 89%; ninguna clase individual falla en las 3 (ver abajo). FP juez 0% en las 3 — muy por debajo del criterio <5%. ✗ NO PASA el floor de recall en la mediana.

**Probe held-out gating (`nombre_no_corresponde`, familia mariscos↔pollo): CAZADO en las 3 corridas** — confirma que la regla generalizada de la ronda 1 de rúbrica sigue sin depender de memorizar el patrón "Moro de guandules" del golden set. ✓ PASA, consistente.

**Probes informativos nuevos (sin criterio de fallo, evidencia cruda para T14):**
- **(a) pescado "en anillos"**: NO CAZADO en las 3 corridas (consistente) — el juez no marca la técnica de corte de calamar aplicada a un filete de pescado blanco, ni como `tecnica_impropia` ni como `paso_incoherente`. Señal para T14: esta clase de error (técnica de corte incorrecta para el tipo de proteína) no está cubierta hoy por la rúbrica.
- **(b) doble-grano bulgur+arroz**: CAZADO en las 3 corridas, siempre clasificado `paso_incoherente` (nunca `combo_absurdo`) — el juez lo lee consistentemente como "el paso menciona un ingrediente ausente de la lista", no como "dos carbohidratos-base en el mismo plato". Ambos tipos están en `tipos_esperados` del script así que cuenta como CAZADO en los 3, pero la clasificación específica es sistemática, no ruido.

**¿Se tocó la rúbrica? No.** El protocolo autoriza 1 sola iteración de rúbrica SOLO si las 3 corridas fallan consistentemente en la MISMA clase-juez. Desglose por clase a través de las 3 corridas:
- `combo_absurdo`: 75%, 50%, **100%** — no falla consistentemente (corrida 3 pasa).
- `nombre_no_corresponde`: 75%, **100%**, 75% — no falla consistentemente (corrida 2 pasa).
- `tecnica_impropia`: 100%, 100%, 100% — nunca falla.

Ninguna clase individual falla en las 3 corridas — el déficit de recall agregado es varianza de muestreo distribuida entre `combo_absurdo` y `nombre_no_corresponde` (el mismo patrón que la corrida original: "78% en esta corrida, 89% en la corrida previa... por pura varianza de muestreo"), no un sesgo sistemático de una clase. Condición de re-iteración no cumplida → rúbrica sin cambios (`graph_orchestrator.py` intacto en esta ronda).

**Costo:** **50 llamadas LLM reales, $0.003894 total** (`llm_usage_events`, `node='culinary_judge'`, `model='glm-5.3-flash'`: 183 723 input tokens, 180 992 cache hit — ~98% del prefix de la rúbrica cacheado por GLM — + 10 831 output tokens), de las cuales 11 llamadas (~$0.0008) pertenecen a la corrida descartada por el crash de encoding (5 buenos + 5 mutados + 1 probe held-out, antes de morir en el print) y 39 llamadas (13×3: 5 buenos + 5 mutados + 1 probe held-out + 2 probes informativos por corrida, ~$0.0031) pertenecen a las 3 corridas oficiales de la tabla. Muy por debajo del estimado "centavos" del brief.

**Veredicto: NO AUTORIZA la escalada OFF→`warn` del knob `MEALFIT_CULINARY_JUDGE_GUARD`.** La mediana de 3 corridas post-backfill (78%, 78%, 89% → mediana 78%) reproduce EXACTAMENTE el resultado de la corrida original (78%) — el backfill de metadata (que solo afecta capa1) no movió el recall del juez (capa2, que nunca leyó `prep_methods`/`ready_to_eat`), y la rúbrica no se tocó porque ninguna clase falló consistentemente en las 3 corridas (condición de re-iteración no cumplida). El knob permanece `off`. Frente a la pregunta que motivó este protocolo de 3 corridas ("una corrida puede caer 78 u 89 por un solo punto"): la mediana confirma que 78% no es el resultado desafortunado de una sola corrida — es el centro de la distribución (2 de 3 corridas independientes cayeron ahí). capa1 sigue en 100%/0FP (contrato F1 re-confirmado, sin regresión post-backfill). El probe held-out gating sigue cazado en las 3 corridas (la regla generalizada de `nombre_no_corresponde` sigue sin sobreajuste). Los 2 probes informativos nuevos quedan como evidencia para T14: `(a)` técnica de corte incorrecta no cubierta hoy por la rúbrica, `(b)` doble-grano cazado consistentemente pero bajo `paso_incoherente` en vez de `combo_absurdo`.

Próximo paso sugerido para T14 (no ejecutado en esta re-calibración — fuera de alcance): si se quiere cerrar el gap de `combo_absurdo`/`nombre_no_corresponde` sin sobreajustar al golden set, ampliar el golden set (más fixtures por clase reduce el ±11pp por hit/miss de N=9) en vez de seguir iterando la rúbrica sobre las mismas 9 muestras.

---

## Iteración de rúbrica v3 (combo_absurdo) 2026-08-01

**Nota sobre el criterio de autorización:** la re-calibración post-backfill (sección anterior) definió el gate de re-iteración como "la clase falla en las 3 corridas de UN modelo" — bajo ese criterio estricto `combo_absurdo` NO calificaba para flash solo (corrida 3 dio 100%, ver desglose arriba: 75%, 50%, 100%). Esta ronda amplía el criterio a evidencia CRUZADA entre dos modelos independientes: el controller corrió el mismo protocolo (×3, mediana) con `gpt-5.6-luna` en el VPS (con `OPENAI_API_KEY` real de producción — el intento de reproducir ese A/B en un worktree local quedó **BLOQUEADO** por falta de esa credencial en `backend/.env` local, ver `.superpowers/ab-juez-luna-report.md`; los 3 valores de luna citados abajo vienen de esa medición del controller, no reproducidos en este documento) y obtuvo `combo_absurdo` 50%, 50%, 0% — mientras el resto de las clases-juez (`nombre_no_corresponde`, `tecnica_impropia`) y el invariante capa1 (100%/0 FP) se mantuvieron intactos en AMBOS modelos. Dos modelos independientes, mismo patrón de defecto (`combo_absurdo`), todo lo demás sano — señal estructural más fuerte que "varianza de muestreo distribuida" (la conclusión de la ronda anterior), y suficiente para autorizar **UNA** iteración generalizada de la rúbrica sin tocar ninguna otra clase.

**Diagnóstico:** las 4 mutaciones `combo_absurdo` del golden set (`golden_manifest.json`, ver `golden_01/02/04/05_mutado`) son todas variantes del mismo patrón — un dulce de desayuno (avena con canela, pan integral con mantequilla de maní) al que se le añade un embutido frito (salami) en el mismo plato. La rúbrica v2 definía `combo_absurdo` con ejemplos genéricos ("cereal con pescado crudo", "postre como fuente principal de proteína") que no cubren este patrón concreto, mientras el párrafo introductorio de la rúbrica ("la creatividad dominicana legítima... NO es una violación; en la duda, NO reportes") le da al juez una salida fácil: cada ingrediente por separado es válido (avena de desayuno legítima, salami legítimo como proteína), así que sin una regla que trace la frontera, el juez puede leer la recombinación como una "fusión creativa" más — el mismo espíritu que los ejemplos de creatividad legítima que la propia rúbrica cita (panqueques de avena, bollitos de yuca). Causa raíz: la definición de `combo_absurdo` no daba un PRINCIPIO para distinguir recombinación-dentro-de-un-patrón (tolerada) de choque-de-perfil-sin-patrón (violación) — dejaba al juez inferir la frontera caso por caso, y con el sesgo pro-creatividad del preámbulo, infería mal.

**Principio nuevo (v3):** la tolerancia a la creatividad cubre RECOMBINACIONES de ingredientes compatibles dentro de un patrón culinario real (dominicano o internacional) — fusiones, sustituciones, platos transformados. NO cubre un choque de PERFIL sin ningún patrón culinario que lo respalde, aunque cada ingrediente sea válido por separado: dulce-de-desayuno + embutido/charcutería frita en el mismo plato, postre + proteína curada, fruta + pescado frito en el mismo bowl. Si ningún patrón culinario conocido junta esos dos perfiles en un solo plato, es `combo_absurdo` — la pregunta operativa que la rúbrica le da al juez es explícita: "¿existe algún patrón culinario conocido que junte estos DOS perfiles en un solo plato?". Se refuerza además que `combo_absurdo` se evalúa **por plato** (nombre + ingredientes del MISMO `meal`, nunca cruzando meals distintos del día). Cambio quirúrgico: solo el bullet `combo_absurdo` de `_build_culinary_judge_rubric()` (`graph_orchestrator.py`) — el resto de la rúbrica (preámbulo de creatividad, los otros 4 tipos canónicos, la REGLA DURA de horario) queda intacto, para no arriesgar las clases que ya estaban en 100%/0 FP en ambos modelos.

**Verificación de no-regresión (spec, antes de medir):** los 5 `golden_XX_bueno` traen combos creativos legítimos que la v3 no puede convertir en FP — `Casabe con queso` (`golden_03_bueno`), `Batida de guineo` (`golden_02_bueno`/`golden_03_bueno`), `Tortilla de huevo con cebolla y queso` (`golden_04_bueno`). Ninguno es un choque de perfil sin patrón (casabe+queso y tortilla+queso son combinaciones dentro de patrones reales; una batida de guineo con leche y miel es un patrón de licuado dominicano estándar) — confirmado por las 3 corridas de abajo: FP juez = 0% en las 3.

### Las 3 corridas (flash, post-v3)

Mismo protocolo que la re-calibración post-backfill (`PYTHONIOENCODING=utf-8 python scripts/calibrate_culinary_judge.py`, `glm-5.3-flash`, sin `--thinking`, guard forzado a `warn` por el script):

| Corrida | capa1 recall | capa1 FP | juez recall | juez FP | `combo_absurdo` | `nombre_no_corresponde` | `tecnica_impropia` | probe held-out (gating) | probe (a) anillos | probe (b) doble-grano | veredicto script |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16/16 = 100% | 0 | 8/9 = 89% | 0/36 = 0% | **4/4 = 100%** | 3/4 = 75% | 1/1 = 100% | CAZADO (`nombre_no_corresponde`) | NO CAZADO | CAZADO (`paso_incoherente`) | capa1=OK juez=OK probe=OK → **AUTORIZA** |
| 2 | 16/16 = 100% | 0 | 9/9 = 100% | 0/36 = 0% | **4/4 = 100%** | 4/4 = 100% | 1/1 = 100% | CAZADO (`nombre_no_corresponde`) | NO CAZADO | CAZADO (`paso_incoherente`) | capa1=OK juez=OK probe=OK → **AUTORIZA** |
| 3 | 16/16 = 100% | 0 | 8/9 = 89% | 0/36 = 0% | **4/4 = 100%** | 3/4 = 75% | 1/1 = 100% | CAZADO (`nombre_no_corresponde`) | NO CAZADO | CAZADO (`paso_incoherente`) | capa1=OK juez=OK probe=OK → **AUTORIZA** |
| **Mediana** | **100%** | **0** | **89% (8/9)** | **0%** | **100% (4/4)** | 75% | 100% | CAZADO (3/3) | NO CAZADO (3/3) | CAZADO (3/3) | **AUTORIZA en 3/3** |

**`combo_absurdo`: 100% (4/4) en las 3 corridas** — sube de la mediana pre-v3 (75%: 75%, 50%, 100%) a 100% sólido, sin una sola falla en 12 evaluaciones (4 defectos × 3 corridas). Es la clase que motivó la iteración y queda cerrada con margen.

**`juez` total: mediana 89% (7/9→8/9), sube de 78% pre-v3** — el criterio de recall (≥0.80) se cumple con margen claro en las 3 corridas individuales (89%, 100%, 89%), no solo en la mediana. `nombre_no_corresponde` (75%, 100%, 75%, mediana 75%) queda intacto respecto a la corrida anterior — la v3 no tocó su definición, y su varianza reportada en la ronda previa ("no falla consistentemente") sigue vigente sin cambios; el déficit residual del recall agregado ahora se concentra ahí, no en `combo_absurdo`.

**FP juez: 0% en las 3 corridas** (criterio <5%) — confirma que la v3 no convirtió ninguno de los combos creativos legítimos del golden set (casabe+queso, batida de guineo, tortilla+queso) en falso positivo, la preocupación explícita antes de medir.

**capa1: recall 100% + 0 FP en las 3 corridas** — invariante intacto, `culinary_coherence.py` no se tocó en esta ronda.

**Probe held-out (`nombre_no_corresponde`, familia mariscos↔pollo): CAZADO en las 3** — no se tocó esa regla, se re-confirma sin cambios.

**Probes informativos (T14, sin criterio de fallo):** (a) pescado "en anillos" sigue NO CAZADO en las 3 (gap conocido, sin cambios); (b) doble-grano bulgur+arroz sigue CAZADO en las 3, siempre como `paso_incoherente` (nunca `combo_absurdo` — consistente con la ronda anterior, la v3 no le pidió al juez reclasificar ese caso).

**Costo:** 39 llamadas LLM reales (13×3: 5 buenos + 5 mutados + 1 probe held-out + 2 probes informativos por corrida), mismo orden de magnitud que la re-calibración anterior (~$0.003).

### Veredicto: AUTORIZA con flash + rúbrica v3

La mediana es clara — no hay ambigüedad ni corrida borderline: `combo_absurdo` pasa de un patrón inconsistente (75/50/100, 1 de 3 corridas por debajo del floor de clase) a **100% sólido en las 3 corridas** tras UNA iteración generalizada de su definición (principio de patrón culinario, sin nombrar ningún plato del golden set en la rúbrica — mismo estándar anti-memorización que la corrección post-review de `nombre_no_corresponde`). El recall agregado del juez sube de mediana 78% a mediana **89%**, por encima del floor 0.80 en las 3 corridas individuales, no solo en la mediana. FP juez 0% en las 3, capa1 100%/0 FP intacto en las 3, probe held-out cazado en las 3/3. **Flash queda calibrado y autorizado para la escalada OFF→`warn` bajo la rúbrica v3** — pendiente de que el controller repita el mismo protocolo con `gpt-5.6-luna` en el VPS (la credencial real está ahí, no en este worktree local) antes de decidir el flip del knob `MEALFIT_CULINARY_JUDGE_GUARD`, dado que el knob es único y no distingue modelo: si luna (el modelo de producción para tiers pagados en day-gen, no necesariamente el mismo que ejecuta el juez) sigue fallando `combo_absurdo` con la rúbrica v3, la escalada seguiría bloqueada por el modelo más débil. Esta sesión NO modificó el knob — permanece `off` en prod, sin cambios de comportamiento en producción; el cambio es únicamente el texto de la rúbrica.

---

## Enlaces

- Spec de diseño completo: [`docs/superpowers/specs/2026-07-31-culinary-coherence-design.md`](../../docs/superpowers/specs/2026-07-31-culinary-coherence-design.md) (secciones 4-4c documentan Capa 1; 5-6 documentan Capas 2-3).
- Módulo SSOT: [`backend/culinary_coherence.py`](../culinary_coherence.py) — puro, sin env vars/LLM/DB (capa 1). Juez LLM (capa 2) vive en [`backend/graph_orchestrator.py`](../graph_orchestrator.py) (`run_culinary_judge`, `_CULINARY_JUDGE_RUBRIC`).
- Tests: [`test_p1_culinary_contract.py`](../tests/test_p1_culinary_contract.py) (migración + V1/V2/V3 + 3 superficies, catálogo sintético), [`test_p1_culinary_golden.py`](../tests/test_p1_culinary_golden.py) (golden set contra Neon real), [`test_p1_culinary_judge.py`](../tests/test_p1_culinary_judge.py) (juez: schema, knobs, fail-open, integración en review, parser-based — sin llamadas LLM).
- Script de calibración: [`backend/scripts/calibrate_culinary_judge.py`](../scripts/calibrate_culinary_judge.py) — manual, hace llamadas LLM reales, no corre en CI.
- Reports de implementación (Tasks 3-13 de este SDD, decisiones + concerns detallados): `.superpowers/sdd/2026-07-31-culinary-coherence/task-{3,4,5,6,7,8,9,11,12,13}-report.md`.

---

## V5 — el paso usa algo que la lista no trae

`[P1-CULINARY-V5-GHOST-STEP · 2026-09-06]`

V3 pregunta «¿hay un ingrediente que ningún paso menciona?». **Nadie preguntaba lo contrario**, y es
la categoría más frecuente del juez culinario: de 227 comidas señaladas, **96 son
`paso_incoherente`**. El daño es directo — el usuario compra la lista y la receta le manda usar algo
que no tiene:

```
«Montaje: … coloca el cilantro por encima»    lista: orégano, ajo, cebolla… sin cilantro
«Montaje: … añade la piña»                     lista: cottage, manzana, semillas de calabaza
«Mise en place: … pela y trocea el plátano»    lista: yogurt, lechosa, fresas, leche
```

### Ocho rondas contra 1.186 comidas vivas

El detector ingenuo daba **460** acusaciones. Cada filtro nació de un falso positivo **medido**:

| | de → a | qué se descubrió |
|---|---|---|
| 1 | 460 → 364 | el índice devuelve el alias corto **y** el largo: «yogurt griego» casaba también `Yogur` |
| 2 | 364 → 287 | las notas de seguridad hablan de CLASES en abstracto («el pollo/cerdo debe cocinarse») |
| 3 | 287 → 241 | lista y paso nombran el mismo alimento con alias distintos |
| 4 | 241 → 100 | el índice no resuelve «1½ filetes de pescado», y eso **no** significa que no esté |
| 5 | 100 → 21 | «chuleta de cerdo» cuando la lista dice «chuleta»: el paso es más específico |
| 6 | 21 → 11 | un paso que USA lo nombra tras un verbo de entrada; uno que lo PRODUCE, no |

Juzgadas a mano las 11: **10 reales, 1 falso** (`ají morrón`, cuya lista dice «0.5 ají») ≈ **91 % de
precisión**.

### Las dos lecciones que costaron más

**El filtro 4.** Sin él, el detector medía el recall del **catálogo**, no la coherencia del plan: un
ceviche con «1½ filetes de pescado» en la lista salía acusado de no llevar pescado porque el índice
no resolvía esa línea. *Un detector que confunde «no lo encuentro» con «no está» acusa al plan de su
propia ceguera.*

**Los tres intentos de matar el último 30 %.** Bajar el umbral de palabra a 3 letras, ensanchar la
ventana, estrecharla a las 3 palabras previas: las tres veces el detector cayó a **CERO**,
llevándose los hallazgos reales — porque «con», «las» o el ingrediente vecino están en toda lista.

> Un filtro que descarta todo no es preciso, es ciego — y se parece muchísimo a uno que funciona si
> solo miras el número.

Lo que salvó la ronda fue tener **hallazgos ya juzgados a mano** con los que comparar. Si mira este
código y le tienta ensanchar un umbral: hágalo, pero vuelva a contar contra planes vivos antes de
darlo por bueno.

### Por qué `warn` y no `block`

`severity='minor'`, `repairable=False`. Con 91 % de precisión, bloquear castigaría un plan de cada
once sin motivo. Y calibrar el umbral contra la tasa del propio juez LLM sería el overfitting que
este repo ya pagó en agosto. **El siguiente paso es un golden set humano**, que es lo único que
convierte «el juez dice que mejoró» en «mejoró».

Test: [`test_p1_culinary_v5_ghost_step.py`](../tests/test_p1_culinary_v5_ghost_step.py).

---

## Línea base congelada y golden set

`[P0-CULINARY-GOLDEN · 2026-09-06]`

Dos capas juzgan hoy la coherencia culinaria y **de ninguna se conoce su precisión ni su recall**,
porque no hay verdad de referencia. «El juez señala el 19,1 % de las comidas» no dice que el 19,1 %
esté mal: dice que él lo cree.

### La foto congelada (`docs/culinary_baseline.json`)

96 planes · 1.186 comidas, medido el 2026-09-06 **antes** de mejorar nada — una medición posterior al
efecto no mide el efecto.

| capa | comidas | | desglose |
|---|---|---|---|
| contrato determinista (V1–V5) | 133 | 11,2 % | V1=80 V3=55 V4=34 V5=11 V2=4 |
| juez culinario (LLM) | 227 | 19,1 % | paso_incoherente=96 · nombre_no_corresponde=53 · combo_absurdo=50 · slot_inapropiado=28 · tecnica_impropia=23 |
| **coinciden** | **23** | | solo det 110 · solo juez 204 |

**Las dos capas apenas se solapan**, así que se publican separadas y **nunca sumadas** en un índice
único: fundirlas escondería que ninguna sustituye a la otra.

    python scripts/culinary_baseline.py              # la foto VIVA de hoy, con delta vs la congelada (orientativo)
    python scripts/culinary_baseline.py --congelar   # ya NO sin --corpus: ver «El corpus fijo» más abajo

### El corpus fijo (C0 · `P1-PLAN-LOTE-18` · 2026-09-12)

La foto de arriba se midió sobre la ventana VIVA (`ORDER BY created_at DESC LIMIT n` sobre `plan_data`) y **dejó de
ser reproducible en 14 h**: los mismos 96 planes daban 1.186 comidas y luego 1.182, porque el shift del cron encoge
los días de un plan ya existente. Y la purga de cuentas del 09-11 se llevó la flota entera: hoy no queda ni uno de
aquellos 96. Una re-medición sobre la ventana viva no compara «antes vs después del código»: compara dos corpus.

Por eso la línea base se congela ahora en DOS pasos, y el corpus vive en un fichero, no en la base:

    python scripts/congela_corpus_culinario.py --motivo "..."          # sólo SELECTs → scripts/data/culinary_corpus_<fecha>.json
    python scripts/culinary_baseline.py --corpus <fichero> --congelar    # → docs/culinary_baseline_<fecha>.json
    python scripts/culinary_baseline.py --corpus <fichero> --verificar   # ¿da hoy las mismas cifras?

`culinary_corpus.py` congela exactamente lo que las capas LEEN —`days` tal cual, `_culinary_judge_history`, el estado
(`revision`, `generation_status`, `updated_at`)— más el catálogo del índice culinario (`master_ingredients`: name,
aliases, category, ready_to_eat, prep_methods), porque el vocabulario del detector cambia con el catálogo. La
huella del corpus es sha256 de las huellas de los planes (ordenadas) + la del catálogo; ni el orden de lectura ni el
orden de claves entran en ella; un paso cambiado o un alias nuevo sí. Un fichero editado a mano no se mide: `cargar`
recalcula las huellas y lo rechaza. La medición publica además `computation.reglas_huella` (sha del fuente de
`culinary_coherence.py`): dos mediciones con la misma `corpus.huella` y distinta `reglas_huella` miden el efecto del
código, y sólo entonces.

`--verificar` tiene tres salidas y ninguna ambigua: **0** reproduce (o el delta es del código, y lo dice); **3**
mismas reglas y cifras distintas —el medidor no es determinista: investigar antes de leer ningún delta—; **4** sin
línea base comparable para esa huella. Y `--congelar` **sin** `--corpus` se niega (exit 2): una línea base sobre la
ventana viva es exactamente el error del 6-sep.

**Congelado el 2026-09-12** (`scripts/data/culinary_corpus_2026_09_12.json`, 310 KB, huella `087cfc31d3105f79`,
catálogo 349 filas `1f4f95b33a191dfe`, reglas `35fc77fbf3416d08`), verificado dos veces: REPRODUCIBLE. Re-congelada el mismo día con las cifras de C1 (`P1-PLAN-LOTE-22`, reglas `c0379767939450e4`): mismas 46/64 y 15/64, más `estado_evaluacion`, `hallazgos`, `juez_entregado` y `particion`; REPRODUCIBLE ×2. Y otra vez con C2 (`P1-PLAN-LOTE-23`, reglas `8f19140121f98a95`: V4 atribuye gramos por gramática): mismas cifras, REPRODUCIBLE ×2.

| | |
|---|---|
| planes · comidas en `days` | 5 · 64 (100 contando `_archived_days`, que la medición no lee) |
| contrato determinista (V1–V7) | 46 comidas · 71,9 % · V7a=47 V7e=38 V6=11 V4=4 V1=4 V3=4 V7d=1 V5=1 V7c=1 |
| juez culinario | 15 comidas · 23,4 % · paso_incoherente=9 tecnica_impropia=2 nombre_no_corresponde=2 slot_inapropiado=1 combo_absurdo=1 |
| coinciden | 10 · solo determinista 36 · solo juez 5 |
| juez sobre lo entregado | no=7 · desconocido=1 (no decidible) |

Tres cosas para leer esta tabla sin engañarse. **No es comparable con la foto del 6-sep**: otro corpus (huella
distinta) y otras reglas (aquella contaba V1–V5; el scan de hoy corre V1–V7e). **La flota es pequeña**: 5 planes tras
la purga; el instrumento vale igual y se re-congela cuando haya flota — cada corpus lleva fecha y huella, y su línea
base también. Y **V7a/V7e disparan en 47 y 38 de 64 comidas** («la lista compra N piezas y los pasos usan menos» /
«un paso pide más piezas que la lista»): el 09-07, sobre 1.194 comidas, eran 154 y 162 (13-14 %). Es un hallazgo
sobre los planes generados del 09-04 al 09-09, no sobre el medidor, y queda para el bloque C (CUL-P0-03) — aquí sólo
se deja medido y reproducible.

`docs/culinary_baseline.json` (la foto viva del 6/7-sep) se conserva tal cual: es historia y no es comparable. El
golden set (`docs/culinary_golden_set.json`) guarda el CONTENIDO de sus 80 comidas, así que las etiquetas humanas
siguen siendo posibles aunque la flota de la que salieron ya no exista. Test: `tests/test_p1_plan_lote_18.py`.

### El golden set (`docs/culinary_golden_set.json`)

80 comidas estratificadas:

| estrato | n | mide |
|---|---|---|
| solo determinista | 20 | precisión de V1–V5 |
| solo juez | 20 | precisión del juez — sus ~200 exclusivos son la incógnita |
| ambas | 15 | los casos que las dos ven |
| **sin hallazgo** | **25** | **el recall: lo que se les escapa a las dos** |

El último estrato es el que se suele omitir y el que impide engañarse: sin comidas limpias solo se
mide precisión, y **un detector que no dispara nunca sale perfecto**.

Cuatro decisiones deliberadas:

- **La muestra no la elige quien la mide**: orden por `sha256` de la clave, no azar ni fecha.
- **Se etiqueta antes de leer a la máquina** — su opinión ancla la tuya. Por eso el `.md` esconde los
  hallazgos en un `<details>`.
- **`dudoso` no cuenta en ninguna dirección.** Forzar un binario donde no lo hay contamina la medida.
- **El muestreador se niega a sobrescribir** un fichero ya etiquetado.

### El marcador

    python scripts/culinary_golden_score.py

Precisión y recall por capa, **en crudo y ponderado**: los estratos raros están sobre-representados a
propósito (15 de 23 «ambas», 25 de 919 limpias), así que sin ponderar la cifra sería inventada. Con
menos de 20 etiquetas **avisa en vez de dar un número**.

No calcula una nota de calidad de 1 a 10. Precisión y recall son propiedades del **detector**; la
calidad del plan necesitaría un criterio de gravedad que nadie ha definido, y fabricarla desde aquí
sería darle a una opinión la cara de una medición.

### Lo que falta, y no lo puede hacer una máquina

**Las etiquetas.** Si las pusiera el modelo, el marcador mediría el acuerdo del juez consigo mismo, y
cualquier «mejora» reportada después sería el sistema dándose la razón. Ese es el paso humano, y es
el que desbloquea la decisión de si V5 escala de `warn` a `block`.

**Actualización 2026-09-12 (`P1-PLAN-LOTE-22`)**: las 80 etiquetas BINARIAS existen desde el 2026-09-07
(`a7d45ffe`: el dueño etiquetó a ciegas, 68 defecto / 9 ok / 3 dudoso) y el marcador binario ya da cifras
(determinista 91,2 → 90,2 % precisión, 45,6 → 12,3 % recall; juez 97,0 → 98,9 / 47,1 → 15,2). Lo que sigue
pendiente es la anotación con RÚBRICA (clase, severidad, evidencia por defecto) y un segundo anotador — sin eso
el marcador estricto sale incompleto (exit 4). Ver «Estado explícito de evaluación» abajo.

Test: [`test_p0_culinary_golden.py`](../tests/test_p0_culinary_golden.py).

## V6 — el paso pide MÁS de lo que la lista compra

[P1-CULINARY-V6-STEP-OVERASK · 2026-09-06] V4 compara **gramos**. Las **piezas** no las miraba
nadie, y ahí vivía un patrón sistemático: la lista dice «½ diente de ajo» y el paso «pica 1 diente
de ajo». Medido sobre los mismos 96 planes y 1.186 comidas de la línea base: **37 hallazgos en 33
comidas (2,8 %)**, 13 alimentos.

El patrón es uno solo — **el modelo redondea las fracciones hacia arriba al recitar la lista en el
«Mise en place»**, que es justamente el paso que la copia:

| lista | paso | consecuencia |
|---|---|---|
| `3 rebanadas de pan integral familiar` | «mide **4 rebanadas**» | una rebanada que nadie compró |
| `½ hoja grande de repollo` | «separa **6 hojas grandes**» | el plato se llama *Canoas de repollo* |
| `1 rebanada de pan integral` | «mide **2 rebanadas**» | el doble |
| `½ cda de aceite de oliva` | «mide **1 cda**» | el doble de grasa |
| `½ pedazo mediano de yuca (≈200 g)` | «corta **¾ pedazo** (255 g)» | +55 g |
| `¼ cdta de comino` | «mide **½ cdta**» | |
| `½ tallo de cebollín` | «pica **2 tallos**» | |

Distribución: `ajo` 15, `canela en polvo` 4, `aceite de oliva` 4, y 1-2 cada uno en los otros diez.

### Las dos decisiones que lo hacen medible

**1. Solo se acusa cuando el paso pide MÁS.** Un paso que usa *menos* que el total puede estar
repartiendo el ingrediente —«calienta 1 cda» de las 2 que compra, el resto después—; acusarlo
convertiría a V6 en un impuesto sobre la receta bien escrita. Uno que pide más no tiene de dónde
sacarlo. **La dirección es lo que separa el defecto del reparto**, no el alimento: en la ronda 2
`aceite de oliva` parecía el segundo peor infractor y era casi todo reparto legítimo.

**2. La unidad es obligatoria, y eso cuesta casos reales.** «½ guineíto verde» en la lista contra
«pela y corta 2 guineítos» en el paso es un defecto, y V6 lo deja pasar a sabiendas. Admitir la
mención sin unidad sube de **37 a 153** hallazgos con ruido demostrable: sin una unidad que ancle
el número al alimento se le pega cualquier cifra vecina — «coloca el Batata como base» heredó un
`3` de otra frase de la receta, y un `huevo 4.0` salió de «2 minutos por lado». **Descartado
MEDIDO**, no por prudencia; si algún día se recupera, será con un ancla de proximidad y una cifra
al lado.

`g`/`ml`/`oz` se reconocen **para descartarlos**: si `g` no estuviera en el vocabulario, «355 g de
lechosa» caería al cubo de las piezas y se compararía contra unidades. La coherencia en gramos ya
es de V4 (`P1-STEP-GRAM-HINT-STALE`).

### Lo que V6 NO afirma

**De qué lado está el error.** En «Canoas de repollo rellenas de soya» la lista pedía ½ hoja y el
paso 6 hojas: la equivocada era **la lista**, porque con media hoja no hay canoas. V6 solo afirma
que los dos se contradicen — decidir cuál corregir necesita criterio culinario que un detector
determinista no tiene.

### Cómo se portó

El prototipo dio 36 hallazgos y el código en repo 37, con la diferencia **explicada línea a
línea**: +2 de `cebollín` porque el repo añadió `tallo` al vocabulario (casos reales), −1 de
`huevo` porque el repo quitó `claras` de las unidades (una clara es un alimento, no una unidad).
Ese contraste es lo que detecta un port incompleto: al portar V5 se me olvidó un filtro y dio 64
donde el prototipo daba 11.

`warn`, como V5. Escalarlo se decide con el golden set etiquetado, no con la tasa que se
autoinforma.

Test: [`test_p1_culinary_v6_step_overask.py`](../tests/test_p1_culinary_v6_step_overask.py).

## El sello de lo que el juez juzgó

[P1-JUDGE-REVISION-STAMP · 2026-09-06] Fui a por los 53 `nombre_no_corresponde` del juez esperando
un detector nuevo, y **la maquinaria ya estaba**: `P1-NAME-GHOST-GAPS` y `P1-NAME-SPECIFICITY`.
Lo que faltaba no era una comprobación, era **poder decidir si una queja del juez describe lo que
se entregó**.

`_culinary_judge_history` guardaba `{ts, model, violations, action_taken}` — nada que atara una
entrada a una versión del plan. Con eso, «el juez se quejó y lo arreglamos» y «se quejó y lo
entregamos» son indistinguibles.

### La medición que lo prueba (96 planes, 2026-09-06)

| comprobación | resultado |
|---|---|
| quejas juzgables «X no aparece en la lista» | 37 |
| …que nombraban algo que **SÍ está** en el plan entregado | **6** (almendras, pistachos, guineítos verdes, queso cottage) |
| líneas de queso con «lonja/pedazo» en planes vivos | 23 |
| …sobre un queso que **no viene en lonjas** (cottage, ricotta, crema…) | **0** |

Ese último es el más elocuente. El agujero existe —`_VAGUE_SLICE_FOOD_RE` es
`(queso|jamon|…)\b.*` y sí casa «queso cottage», que se vende por peso y no en lonjas— y el juez lo
citó cinco veces. Pero **la reparación lo convierte antes de entregar**: el usuario nunca lo ve.
Por eso `_VAGUE_SLICE_FOOD_RE` se queda **como está**: estrecharlo sin datos arriesga los 23 casos
legítimos para arreglar 0 reales.

### El sello: `judged_fingerprint`

SHA-256 de (día, franja, **nombre**, **ingredientes**, **pasos**) — exactamente lo que el juez lee.

**No se reutiliza `services.compute_plan_hash`** pese a declararse «fuente única de verdad para
detectar si un plan cambió»: hashea ingredientes y suplementos, y el bucket más grande del juez
(`paso_incoherente`, 96 de 250) es de PASOS. Un paso reparado dejaría ese hash quieto y la
comparación diría «es el mismo plan» justo en los casos que más importan. Una huella que no cubre
lo que se juzgó reintroduce la misma ambigüedad, sólo que más difícil de ver.

### Tres estados, y el tercero es el que importa

`judgment_covers_delivered` devuelve `True` / `False` / **`None`**. Las 96 entradas que existen hoy
no llevan sello, así que la respuesta honesta es «no se puede saber». Colapsar ese `None` hacia
cualquier lado fabricaría una cifra — que es exactamente el error que este P-fix cierra. El medidor
los cuenta por separado y los publica por separado (`juez_sobre_lo_entregado`).

### La advertencia nace del código, no del JSON

`culinary_baseline.json` lleva ahora la segunda razón con sus cifras. Pero enmendarlo a mano era una
trampa: `--congelar` reescribe el fichero entero, así que la enmienda habría desaparecido en
silencio en el siguiente congelado. La advertencia vive en `ADVERTENCIA`
(`scripts/culinary_baseline.py`) y el JSON la recibe de ahí.

Test: [`test_p1_judge_revision_stamp.py`](../tests/test_p1_judge_revision_stamp.py).

## Estado explícito de evaluación e identidad por ocurrencia (C1 · `P1-PLAN-LOTE-22` · 2026-09-12)

CUL-P0-01 y CUL-P0-02 del paquete del 09-07. Cuatro cosas que se confundían con «aprobado» y ahora tienen nombre:

| antes | ahora |
|---|---|
| `culinary_contract_scan` devolvía `[]` por «coherente», por «sin catálogo» y por «reventó» | `culinary_contract_scan_status` → `(violations, estado)` con `status ∈ {scanned, no_meals, no_catalog, error}`; el orquestador persiste `plan["_culinary_contract_scan"]` (viaja de T1 a T2 como sus hermanas) |
| la comida se identificaba por FRANJA en las dos capas | cada violación de capa 1 lleva `meal_index` (posición en su día); el juez recibe `idx` en el payload y devuelve `meal_index`; `resolve_judge_violations` ata las quejas antiguas por franja SOLO cuando es única (`declarada / unica / ambigua / sin_comida`) |
| la entrada del juez no decía con qué rúbrica, modelo ni país juzgó | `context` en cada entrada (`rubric_fingerprint`, `model`, `country`, `guard`, `schema`, `reglas_huella`) junto al sello `judged_fingerprint` |
| el medidor contaba «comidas que el juez señaló alguna vez» | `judge_evaluation_state` (`juzgado_vigente / juzgado_obsoleto / no_disponible / no_evaluado / desconocido`; sólo vigente y sin hallazgos = aprobado) y `juez_entregado` (quejas de entradas vigentes atadas a una comida que existe), separado del histórico `juez` |

**El sello por COMIDA (`meal_seal`).** Al re-congelar la línea base con estas cifras, los 5 planes del corpus salieron
«juzgado_obsoleto» y `juez_entregado` = 0 de 64: `judged_fingerprint` sella el plan ENTERO con la posición del día, y el
shift archiva y renumera días — declara obsoleto todo lo juzgado aunque la comida entregada sea byte a byte la juzgada.
Desde C1 cada violación (capa 1 y juez) lleva `meal_seal` = sha de (franja, nombre, ingredientes, pasos) de SU comida:
`resolve_judge_violations` la reencuentra por sello (`resolucion: por_sello`, `ocurrencia_actual`) aunque haya cambiado
de día, y la línea base cuenta esas quejas como vigentes (`hallazgos.juez.vigentes_por_sello`). Las entradas anteriores
no llevan sello por comida y siguen dependiendo del sello del plan: para ellas la cifra honesta sigue siendo «no se sabe»
u «obsoleto». `juez_entregado` empezará a decir algo con los planes juzgados después de este despliegue.

**Reconciliación con el denominador.** La línea base publica `particion` (ambas / solo_determinista /
solo_juez_vigente / ninguna, por ocurrencia) y comprueba que suma `comidas` (`reconcilia`); lo que no se puede atar a
una comida —franja ambigua, día archivado— se informa en `hallazgos.*.sin_comida/ambiguos`, no se reparte a nadie.
Las cifras nuevas entran en `CIFRAS`: también tienen que reproducirse sobre el corpus fijo.

**Coberturas, tres y no una.** `scan_coverage_detail`: `reconocimiento` (líneas de ingredientes en las que el índice
encontró algún alimento: el PARSER), `catalogo` (alimentos con `prep_methods`: lo que V1 necesita; es `scan_coverage`)
y `ready_to_eat` (V2), con `por_check`. Un 59 % por «no hay metadata» y un 59 % por «no reconozco la mitad de las
líneas» se reparan en sitios distintos.

**El marcador estricto (CUL-P0-02).** `culinary_golden_score.py --estricto` adjudica hallazgo a hallazgo con la
`RUBRICA` (clase humana → códigos de la máquina): un hallazgo cuenta como TP sólo si su clase corresponde a un
defecto humano de ESA comida (y menciona el `alimento` si el defecto lo nombra); si no, FP localizado y el defecto
queda como FN; los duplicados no multiplican TP; cero división → `null`. Publica acuerdo entre anotadores (kappa) y
discrepancias, usa la `adjudicacion` cuando existe, intervalos por conglomerado (plan) y `--particiones` por linaje.
**Con las etiquetas de hoy sale incompleto (exit 4)**: son binarias. `culinary_golden_sample.py --plantilla` escribe
el hueco por caso (`docs/culinary_golden_anotaciones_pendientes.json`) y `--ciego` la representación sin estrato ni
máquina para el segundo anotador (`docs/culinary_golden_set_ciego.md`). Las etiquetas pendientes NO se rellenan con el
modelo.

**Calibrador y test golden por COMIDA.** Cruzaban `(día, clase)`: en `golden_05` un V1 del Desayuno contaba como
acierto de la Cena. Ahora exigen la franja; las 16 mutaciones de capa 1 resuelven a comida (medido antes de endurecer).

**Lo que queda del dueño**: la anotación con rúbrica de los 80 casos y un segundo anotador independiente. El
software de evaluación está listo antes que las etiquetas, como el backlog admitía; la certificación de calidad
sigue pendiente hasta tenerlas.

Tests: [`test_p1_plan_lote_22.py`](../tests/test_p1_plan_lote_22.py).

## La anotación del dueño entra al instrumento (C1 cierre · `P1-PLAN-LOTE-60` · 2026-09-15)

Lote 38 del plan 38-44. Con la anotación con rúbrica del dueño (80/80, ciega) el marcador estricto daba **0 aciertos
por construcción**: exigía el `alimento` del defecto como subcadena del texto de la máquina, y el de V4 no nombra
ninguno — «V4: ingrediente declara 85 g, pasos declaran 140 g» es exactamente el defecto del caso `0108f857ae` y salía
FN + FP. Tres cambios, todos en el instrumento y ninguno en los detectores:

- **Adjudicador** (`culinary_golden_score._adjudicar_hallazgos`): el alimento sólo restringe si el hallazgo nombra uno
  —una palabra de la lista de ingredientes de esa comida, o el que declara acusar—, y entonces basta con que coincida
  cualquier palabra, sin acentos y en singular. Cada acierto dice `emparejado_por`. «defecto» sin defectos de la rúbrica
  queda `sin_rubrica` y el informe lo nombra. Sin ningún filtro, el juez ganaba dos aciertos que eran quejas de OTRO
  alimento (`098d23388f`, `0ee6d0a81c`): el filtro los descarta a propósito.
- **Refresco** ([`scripts/culinary_golden_refresh.py`](../scripts/culinary_golden_refresh.py)): las columnas `maquina_*`
  eran los hallazgos PERSISTIDOS el 09-06 (V1-V5, el juez sin `[dudosa]`). El script pasa la máquina de hoy por los 80
  casos y escribe `maquina_<capa>_<fecha>` AL LADO —nunca pisa las del 09-06— con el alimento acusado al final (`(alimento:
  X)`, el `food` que el builder tiraba: un V7e cuyo detalle cita el paso nombra OTROS alimentos). El juez es el de
  producción, una comida por llamada, con las escrituras a la base sustituidas por dobles y el coste contado en proceso
  (total $0,0387 de $0,50).
- **Línea base estricta** ([`culinary_baseline_estricto_2026-09-15.md`](culinary_baseline_estricto_2026-09-15.md) + `.json`, que
  es la salida literal de `--comparar-maquina`): antes y después del refresco con el mismo adjudicador. No se toca después.

Resultado: determinista 11/30/36 → 18/60/29 (tp/fp/fn; recall 23,4 → 38,3 %, precisión 26,8 → 23,1 %), juez 2/32/7 → 0/28/9. El determinista de hoy ve más (V7c ya caza 3 de los 11 «seco sin cocción») y acusa más cosas que el
dueño no marcó (V3, V7a); `coccion_faltante` (8) sigue sin ningún código. Test:
[`test_p1_plan_lote_60.py`](../tests/test_p1_plan_lote_60.py).

## El contrato sobre la receta final (C2 · `P1-PLAN-LOTE-23` · 2026-09-12)

CUL-P0-03 del paquete del 09-07: «sustituciones y ajustes dejan técnicas del alimento anterior, cantidades
contradictorias o ingredientes añadidos sin preparación; su frecuencia no está medida».

**Medido primero.** Corpus fijo del 09-12 (5 planes, 64 comidas): V7a 47, V7e 38, V6 11, V4 4 — **100 de 111 hallazgos de
capa 1 son cantidades** que los pasos y la lista se contradicen; de las 68 comidas que el dueño marcó con defecto, 54 lo
dicen en su nota. La causa: la lista la mutan media docena de reparadores (caps de huevo, porciones absurdas, pisos,
motor de macros, sustituciones, `_reconcile_display_raw_lines`) y los sincronizadores de pasos que ya existían
(`_sync_recipe_step_quantities`, `_egg_count_step_sync`, `_rewrite_recipe_steps_after_subs`) corren en puntos fijos:
**una reparación que corre después del sincronizador deja lista y pasos desincronizados**. Re-ejecutar el sincronizador
existente sobre el corpus sólo bajaba V7e 38 → 35 (no lee piezas desnudas ni alimentos de menos de 4 letras: «ajo»,
«pan»), pero SÍ corregía «casca 6 huevos» → 3 — prueba de que al persistir no corrió después del cap.

**El contrato (`recipe_contract.py`).** `reconcile_step_quantities(meal, index)` lee la lista con los mismos parsers del
contrato determinista (gramos V4, unidades V6, piezas V7) y reescribe en los pasos la cantidad que la contradice, familia
por familia; nunca cruza familias (eso es V7b). Reglas: la lista es la autoridad (nutrición y compra salen de ella);
una sola mención → se alinea en las dos direcciones; dos o más → sólo se recorta la que pide más de lo comprado (un
reparto no se adivina); tolerancias de las capas que miden (25 % gramos, 5 %/0,06 conteos); piezas que cruzan el
singular/plural NO se tocan (`gramatical`); un conteo de compra con gramos («1 cebolla (25 g)») no infla el paso
(`conteo_con_gramos`); rangos, «≈», notas ⚠/💡 y de procedencia quedan fuera; `ingredients_raw` no se toca. Concordancia
de unidad y artículo («tuesta las 2 rebanadas» → «tuesta la 1 rebanada»). Idempotente y fail-open.

**Dónde corre: ÚLTIMO.** `apply_final_contract(days, db)` al final de `finalize_plan_data_coherence` (el shield
pre-INSERT de `db_plans`, después de `_reconcile_display_raw_lines`) y `apply_final_contract_meal(meal, db)` al final de
`finalize_single_meal_recipe_coherence` (swap, chat-modify, regenerar día, recipe-expand). Knob
`MEALFIT_RECIPE_FINAL_CONTRACT` = `repair` (default) / `shadow` (anota sin tocar) / `off`; telemetría por plato en
`_recipe_contract_final` sólo cuando hay algo que decir. Lector: `scripts/medir_contrato_receta_final.py` (corpus fijo
y, con `--vivo N`, la ventana viva SOLO en lectura).

**V4 atribuye por gramática (`grams_owner`).** «corta 70 g de nabo, 265 g de tomate»: por cercanía los 265 g eran del
nabo (2 de los 4 V4 del corpus), y el reparador los habría reescrito sobre el alimento equivocado. Ahora el dueño de un
«N g» es el alimento que lo SIGUE («de» opcional) o el que lo PRECEDE pegado («yogur (90 g)»); sin dueño no hay
comparación. *Un medidor que atribuye por distancia no puede alimentar un reparador.*

**Resultado sobre el corpus** (copia; la base no se toca): 36 de 64 comidas tocadas, 63 cantidades reescritas
(47 piezas, 4 gramos, 12 unidades); **V7e 38 → 3, V6 11 → 0, V4 4 → 0**; V7a 47 → 44 — lo que queda es número
gramatical («2 claras» vs «la clara»), que se deja a V7a en `warn` a propósito. Segunda pasada: 0.

**Lo que este lote NO hace, dicho.** No convierte el residuo crítico en rechazo: el guard de capa 1 sigue en `warn`
(`MEALFIT_CULINARY_CONTRACT_GUARD`); pasar a `block` es la escalada F2 y necesita la precisión del marcador estricto
(C1 → etiquetas con rúbrica del dueño). Y las «técnicas del alimento anterior» tras una sustitución siguen con la
maquinaria que ya existía (`_rewrite_recipe_steps_after_subs`, `_substitute_blended_raw_egg` para el huevo licuado →
yogur): en el corpus V1 vale 4 y los cuatro son falsos positivos del aceite/queso «a la plancha», no residuos de
sustitución — sin caso medido no se construye un degradador de verbos que dependa de la precisión de V1.

Tests: [`test_p1_plan_lote_23.py`](../tests/test_p1_plan_lote_23.py).


## Las tres formas del huevo (C3 · `P1-PLAN-LOTE-24` · 2026-09-12)

**Qué decía el gap (CUL-P0-04).** Huevo entero, clara y yema no son intercambiables; «12 huevos sin yema» debe conservar
12 claras y no heredar los macros de 12 enteros; nada convierte una petición de claras en yemas; la petición sobrevive a
«enteros primero» y a los fallbacks; separar huevos y comprar claras en envase son compras distintas.

**Lo medido antes de construir.** El catálogo ya distinguía las tres formas con `fdc_id` propios (`Clara de huevo` 33 g/ud
y 0,1 g de grasa por 100 g; `Huevo` 50 g y 9,5 g; `Yema de huevo` 17 g y 26,5 g), la nutrición resolvía «12 claras de huevo»
a la clara (396 g, 0,7 g de grasa) y el techo por pieza comparaba por sustantivo cabecera. **La receta era la única capa
que las confundía.** En el corpus fijo (5 planes, 64 comidas): 14 comidas con huevo; 6 salieron del tope diario de enteros
(`_cap_daily_whole_eggs`: «6 huevos» → «3 huevos» + «3 claras de huevo» EN LA LISTA) y en las 6 los pasos seguían diciendo
«casca 6 huevos». El contrato de C2, ciego a la forma, lo reescribía a «casca 3 huevos» y las claras desaparecían de la
preparación: 5 comidas contradictorias lista↔pasos. Y una línea escrita como «12 huevos sin yema» resolvía a `Huevo`
(600 g, 57 g de grasa).

**Hallazgo de orden, y era de C2.** El contrato «ÚLTIMO en el persist boundary» no lo era: dentro de
`db_plans._finalize_plan_data_for_insert` (el SSOT del orden, por el que pasan INSERT, merges de chunk y la cadena de
calidad) detrás de `finalize_plan_data_coherence` siguen mutando la lista el band-closer, los caps de realismo, el tope
diario de huevos, el piso de proteína, los condimentos y el re-cuadre de conteos; en swap y chat-modify, el re-cuadre del
día y el motor de macros. El contrato corre ahora también en la COLA de esos tres chokepoints
(`P1-PLAN-LOTE-24-FINAL-CONTRACT-TAIL`: antes de restaurar los días pasados congelados y de los detectores; en swap y
chat-modify antes del rebuild de las listas). Los ganchos de los finalizadores se conservan: el contrato es idempotente.

**Qué hace `recipe_contract.reconcile_meal`** (el contrato completo sobre un plato, en este orden):

1. **La lista nombra la forma.** «N huevos sin yema(s)» / «N huevos (solo claras)» → «N claras de huevo»; «N huevos sin
   clara» → «N yemas de huevo». Se reescribe `ingredients` y la línea IGUAL de `ingredients_raw` (por texto, nunca por
   índice) y el llamador re-mide los macros (`_truth_up_meal_macros_from_strings`). Nunca en sentido contrario.
2. **Los pasos siguen a la forma comprada.** Con enteros y claras en la lista, toda mención «N huevos» que no coincida con
   los enteros comprados recibe el reparto real («casca 3 huevos y 2 claras de huevo»); si las claras compradas no aparecen
   en ningún paso, la primera mención de huevo lo recibe (numérica o desnuda: «Cocina huevo a la plancha» → «Cocina 3 huevos
   y 1 clara de huevo a la plancha»). Sin enteros, «N huevo(s)» y «el/los huevo(s)» pasan a la forma comprada («bate 1 clara
   de huevo», «añade las claras») y la nota de seguridad deja de exigir «yema y clara firmes». **Una lista de enteros con
   pasos que dicen «hasta que la clara cuaje» es técnica, no contradicción: no se toca.** Nada convierte claras en yemas.
3. Las cantidades de los pasos siguen a la lista (C2), que ahora ve «3 huevos» = 3 y «2 claras de huevo» = 2.

Resultado sobre el corpus (copia): **5 comidas contradictorias → 0**, 8 pasos reescritos por forma, segunda pasada 0.
Lector: `scripts/medir_formas_huevo.py [--vivo N] [--json]` (solo lectura).

**El pedido de claras sobrevive a «enteros primero» y a los fallbacks.** `plan_policy.egg_staple_forms(form)` es la única
lectura de la declaración (`stapleFoods`/`stapleAnchors`, por sustantivo cabecera, como el techo por pieza). Con la Clara de
huevo declarada básico y sin el huevo entero, `prompts.day_generator.override_egg_form_preference` sustituye la regla
«HUEVOS: ENTEROS PRIMERO» por «HUEVOS: CLARAS PRIMERO» (misma técnica que el tope de claras: una regla, sin contradicción;
prompt byte-idéntico para quien no declaró nada). Con cualquier forma del huevo declarada básico, `_diversify_egg_pools`
no le quita el huevo al planificador a partir del 3.º día. La ración pedida viaja al bloque 📐 de la política con su forma
(«Clara de huevo (… 10 unidad por comida)»). Knob `MEALFIT_EGG_STAPLE_HONORED` (default `True`).

**Ya cierto, ahora anclado por test.** La alergia a huevo se expande a clara/yema (`constants`); el vegano rechaza las tres
formas (`_DIET_EGG_TERMS`); el escáner de huevo crudo las ve (`_RAW_EGG_TERMS`); el ancla «Clara de huevo» NO se da por
cumplida con «2 huevos» (`_matches`/`anchor_in_text`), aunque «Huevo» sí acepta claras y yemas.

**Lo que NO hace, dicho.** La compra sigue colapsando claras y yemas en cartones de `Huevo` (`canonicalize_huevo`,
decisión del 2026-05-11: el usuario separa y aprovecha las yemas; 13 filas dependen de esa cadena). El supermercado YA
tiene «Clara de huevo · Don Papito · botella pasteurizada 400 g · RD$154,95» mapeada a `Clara de huevo`: comprar claras en
envase es hoy una decisión de producto sobre el agregador, no un dato que falte — del dueño. El techo del 25 % de comidas
con huevo del gate de variedad y la regla «1 comida con huevo por día» del prompt siguen; honrar «10 claras por comida»
sigue bajo el canary de la política (`MEALFIT_PLAN_POLICY_ENFORCE_USERS`, E5 fase B).


**Claras en botella a partir de 4 (`P1-PLAN-LOTE-61`, 2026-09-15): medido, no construido.** La decisión delegada del
14-sep era «botella de claras pasteurizadas cuando una comida pide 4 o más». Contadas con el mismo parser de formas del
huevo que V7 (`recipe_contract.egg_forms_in_list`): en el corpus fijo del 09-12, de 64 comidas, 14 llevan alguna forma de
huevo y las claras por comida son {1: 2, 2: 2, 3: 2, 5: 1}, así que **una** cruza el umbral (5 claras); en el golden set,
2 de 80 (4 y 5). La regla toca el agregador de compras —`_consolidate_inline_canon` funde `Clara de huevo` en `Huevo`
sobre una lista plana, sin la comida— en las 6 superficies que escriben la lista, y `shopping_calculator.py` está a unas
150 líneas de su tope. Queda para su propio lote, con estas cifras; la fila de la botella («Clara de huevo pasteurizada ·
Don Papito · 400 g») existe desde el 07-sep.

## Asignación paso↔ingrediente en la receta congelada (C4 · `P1-PLAN-LOTE-25` · 2026-09-12)

**Las tres recetas que quedaban en `revisar` (`P1-PLAN-LOTE-61`, 2026-09-15).** Redactadas por delegación del dueño (Hoja,
documento `angelo`, veredicto «corregida») y copiadas tal cual: el aceite de los yaniqueques en tres tercios, el del pollo
en mitad + cuarto + último cuarto, y la auyama de las lentejas con su paso. El parser no entendía «otro tercio» ni «el
último tercio/cuarto» (leía 1/3 y 3/4 y las dejaba `a_medias`): «otro tercio/cuarto» son fracciones fijas y «el último
tercio/cuarto» CIERRA como «el resto». Resultado: `revisar` 3 → 0 (exacta 156, estimada 37); sólo cambian esas tres. Los
dos tests del lote 25 que usaban las recetas defectuosas como ejemplo llevan ahora los pasos del 09-12 como fixture.

**Qué decía el gap (H8 de la auditoría del 09-11).** Extender la receta congelada con `pasos[i].usa: [{ingredient_id,
fraccion}]` para las recetas DO (se escribe una vez); V6 pasa de «puede repartir» a suma exacta. Coste previsto: revisión
editorial de 190 recetas y re-firma curatorial.

**Lo medido antes de construir.** La biblioteca (`recipe_library_do_v1.json`) tiene 193 recetas y 867 pasos escritos A
PROPÓSITO sin cantidades de ingrediente («valen para cualquier porción»). Materializadas como plato y pasadas por el
escáner de hoy: **6 hallazgos (V1 4, V3 1, V5 1) y ninguno de cantidad** — no porque repartan bien, sino porque V6, V7a y
V7e leen NÚMEROS del texto y aquí no hay ninguno: «la mitad del aceite» … «la otra mitad del aceite» … «la otra mitad del
aceite» (tres mitades) pasa limpio, y «la mitad del cilantro» sin la otra mitad —la cuarta clase de las notas humanas del
09-07, «usado a medias», que el comentario de V7 dejó anotada sin detector— tampoco lo ve nadie. En la flota: 0 comidas de
receta congelada en 30 días (el día determinista sigue en canario); en el corpus fijo, 0. El terreno de esta capa es la
biblioteca, no el plan vivo — todavía.

**Qué hace `recipe_usage`** (módulo nuevo, puro, sin base ni red):

1. **Asigna** a cada paso los constituyentes de SU plantilla que usa, con la FRACCIÓN de lo comprado: `usa[i]`, paralelo a
   `pasos[i]`. El vocabulario es el de la plantilla —un mundo cerrado de 3-8 alimentos—, no el catálogo: dentro de
   «Pinchos de pollo…», «el pollo» sólo puede ser la pechuga y «el morrón» el ají morrón. Un token que dos constituyentes
   comparten («aceite» con dos aceites en la misma plantilla) no decide: se declara `ambigua`.
2. **Reparte** por las pistas del texto, leídas en su cláusula (una pista no cruza un punto ni una coma, salvo la lista
   «el resto del ají, el ajo y el cilantro»): «la mitad de» → 0,5; «la otra mitad» / «el resto» / «… restante» → lo que
   queda; «parte de», «una pizca de», «un chorrito de» → una parte SIN cifra, estimada a partes iguales y marcada
   `estimada`; sin pista, toda la cantidad entra en el PRIMER paso que lo nombra y las demás menciones son referencias («el
   pollo está seguro cuando el termómetro…»). «Resérvala», «queda para untar», «corrige la sal», «sin sal» no consumen.
   Una fracción fija heredada por conjunción («la mitad de la sal **y el orégano**») no parte al de al lado. Medido en la
   biblioteca: «la mitad» en 71 pasos, «el resto» 41, «otra mitad» 34, «un poco» 45, «una parte» 19, «restante» 11; 398 de
   1017 constituyentes se nombran en dos o más pasos — sin estas reglas la asignación sería una adivinanza.
3. **Cuenta** por constituyente: Σ de fracciones = 1. Σ = 0 es «comprado y ningún paso lo usa» (V3); 0 < Σ < 1 es «usado a
   medias» (V7a); Σ > 1 o dos veces «el resto» es «reparte más de lo que compra» (V6). «Fuera de plantilla» (la familia de
   V5) se inventaría reutilizando V5 sobre la plantilla materializada: el matcher global crudo daba **50 acusaciones, casi
   todas técnica leída como alimento** («al sofrito» → Sofrito ×18, «hasta que el agua salga clara» → Clara de huevo,
   «cuajada»); V5 con sus guardas da 4. Un inventario con 46 falsos no es un inventario — descartado medido.
4. **Ata** la asignación al TEXTO: `pasos_hash` (sha256[:16] de los pasos). Vive APARTE de la biblioteca
   (`data/registry/recipe_usage_do_v1.json`), así que la receta no cambia ni un byte, su test de esquema sigue igual y **la
   firma curatorial no caduca**; si alguien edita un paso, la asignación caduca sola (`usage_for_template` → `None`) y el
   escáner vuelve a la heurística para ese plato. La «firma que caduca» de la auditoría, mecánica y sin ceremonia.

**Resultado sobre la biblioteca** (`scripts/asignar_uso_pasos.py [--write|--verificar|--revisar|--json]`): 193 recetas ·
**155 `exacta`, 35 `estimada`, 3 `revisar`** · 1017 constituyentes, **1004 con Σ = 1** (los 13 restantes: 12 veces la `Sal`
comprada que ningún paso nombra —condimento, `sin_uso_condimento`, informativo como en V3— y 1 auyama) · 95 fracciones
estimadas · idempotente y reproducible desde el texto y el catálogo congelado del corpus (`--verificar`, exit 3 si el
snapshot caducó). Las 3 que la máquina no pudo cerrar son defectos de REDACCIÓN de la receta y van al dueño con nombre:
«Yaniqueques horneados» (tres mitades del aceite), «Pollo al horno con batata y ensalada de repollo» («el resto del
aceite» y luego «lo que quede del aceite») y «Lentejas guisadas con auyama y batata» (la auyama no aparece en ningún paso).
El coste previsto —190 recetas de revisión editorial y una re-firma— queda en 3 recetas y ninguna firma.

**Dónde engancha.** En `culinary_contract_scan`, para una comida con `_recipe_source == "library"` y asignación vigente
(mismo hash, mismo número de pasos), **V3, V6, V7a y V7e dejan de adivinar por texto y leen las cuentas**
(`P1-PLAN-LOTE-25-SCAN-EXACT`): cada check se cede a sí mismo desde dentro, la cadena del escáner no cambia (los tests la
leen como texto), los hallazgos salen con `detail` «asignación exacta (estado): …» y el estado del scan cuenta las comidas
evaluadas así (`exactas`). «Usa lo mismo o menos: puede repartir» sigue siendo la regla para las recetas del LLM: sin
vocabulario cerrado ni texto congelado al que atarse, ahí no hay suma exacta posible. Knob `MEALFIT_RECIPE_USAGE_EXACT`
(default `True`); apagado ⇒ heurística de siempre, sin redeploy.

**Lo que NO hace, dicho.** No escribe cantidades en los pasos («corta 168 g de pechuga»): eso es CUL-P1-05 (C5), y la
receta congelada se escribió sin ellas a propósito. No corrige las 3 recetas: su texto es del dueño. No cambia el veredicto
sobre ninguna comida del LLM. Y no convierte el residuo en rechazo: los hallazgos exactos nacen `minor`/`warn` como sus
hermanos de texto.


## Cultura, horario, equipo, tiempo y básicos como contexto (C5 · primera parte · `P1-PLAN-LOTE-26` · 2026-09-12)

Tres gaps P1 del backlog culinario (CUL-P1-01 claras según persona, CUL-P1-03 cultura y horario como contexto, CUL-P1-05
cantidades/tiempos/equipo ejecutables) comparten la forma: una regla GENERAL escrita en un sitio (un tope de claras,
«arroz nunca de noche», «15 min») y una PERSONA que declaró algo que la contradice o la completa. `culinary_context.py`
es la única lectura de esas declaraciones; cinco superficies enganchan con dos o tres líneas y fallan abiertas. Knob
`MEALFIT_CULINARY_CONTEXT` (default `True`): apagado, cada enganche devuelve la conducta anterior.

**Lo medido antes de construir (flota, corpus fijo y biblioteca).** Claras entregadas: 1, 2, 3 y 5 por comida; ningún
plan vivo declara la clara como básico; 5 de 6 planes llevan la política EN VIGOR, así que el tope por ración (E5) ya
está activo — lo que seguía ciego era el techo de COMIDAS con huevo (`max(3, 25 %)`) y el conteo que sumaba «1 clara»
de aglutinante como otra comida de huevo. Tiempos: 5 de 64 comidas del corpus declaran 15-20 min y sus pasos hablan de
horas; 4 son conservación («consume dentro de 24 horas», «refrigera el sobrante hasta 48 horas») y 1 es real («el
arroz de la noche anterior»). Equipo: el formulario principal no lo pregunta (decisión de producto
`P2-FORM-KITCHEN-EQUIPMENT`, 2026-06-22) pero el panel de Súper Personalización sí (`kitchenEquipment`), y sólo lo leía el
prompt del plan; con «sólo estufa» declarado, 13 comidas del corpus piden horno, 5 licuadora y 2 airfryer; 53 de 193
recetas congeladas piden horno. Horario: la rúbrica del juez llevaba «arroz/pasta NUNCA en desayuno ni cena» — los países
beta la heredaban entera y la pasta de noche es legítima en RD desde 2026-06-27.

**Decisión de producto (CUL-P1-03), registrada aquí.** Lo típico, lo preferido y lo prohibido son tres cosas: la guía
positiva del juez y `SLOT_INAPPROPRIATE_FOODS` dicen lo típico; un básico declarado PARA esa franja (`stapleAnchors`
con `slots`, o `stapleFoods` sin franja) es lo preferido y no se acusa (`_slot_pref_exempt` en
`_detect_slot_appropriateness`; `_night_rice_autofix` no le quita el arroz de noche a quien lo pidió); lo prohibido son
las restricciones explícitas (alergias, dieta, clínica), que no cambian. La rúbrica del juez pasa de «REGLA DURA» a
«REGLA DE HORARIO — CONTEXTO, NO DOGMA» con tres condiciones para `slot_inapropiado`, dice que la pasta de noche es
legítima, que la avena salada, el yogur como salsa o el pescado con salsa de fruta se juzgan por técnica y composición,
y que los componentes SEPARADOS del plato (fruta al lado de unos huevos) no son una mezcla; los países beta reciben la
versión NEUTRA (la costumbre del arroz de mediodía no es universal). El juez recibe además `contexto` (básicos con su
franja, cocinas elegidas, equipo, tiempo). El detector determinista de pareo chocante respeta la separación dicha en el
nombre («con mango al lado»). Ablandar el gate de `review_plan_node` para que soft no fuerce retry sigue PARKED (decisión
aparte).

**Claras según persona (CUL-P1-01).** El techo de comidas con huevo (gate de variedad y `_egg_cap_autofix`) honra al huevo
declarado básico: al menos una comida por día, nunca menos que el de siempre (`culinary_context.egg_meal_cap`); una clara
de aglutinante (≤ 1 pieza, sin huevo en el nombre, masa/mezcla/rebozado en el texto) no cuenta como comida de huevo
(`egg_is_binder`). Con la política en modo SOMBRA, la ración de piezas por encima del tope por defecto no se honra (E5 lo
decidió: canary) pero ya no se pierde en silencio: `plan_policy._note_portions_not_enforced` la escribe en
`relaxations` con `reason_code=portion_cap_default_not_enforced`, `action=deferred` y copia legible. Sigue tal cual:
`portion_cap_for`/`build_count_caps_override` (E5), «CLARAS PRIMERO» (C3), y ningún límite clínico inventado.

**Tiempos y equipo ejecutables (CUL-P1-05).** Dos checks nuevos de capa 1, ambos `minor`, no reparables (el reparador es
CUL-P1-04): **V8a tiempo oculto** — los pasos piden horas de espera (remojo, marinado, «la noche anterior») que el
`prep_time` del plato no cubre; la conservación no cuenta; **V8b equipo no disponible** — la receta exige horno,
airfryer, licuadora, microondas, olla de presión… y la persona declaró no tenerlo; sin declaración no se evalúa y el
estado del scan lo dice (`contexto.equipo = no_declarado`). El equipo declarado llega también al prompt del día (bloque
🍳, byte-idéntico sin declaración), al selector determinista (`template_candidates(available_equipment=…)` poda las
plantillas cuya receta lo exige) y a la métrica de personalización (`equipment_unavailable` medido; antes
`form_has_no_equipment_field`, siempre). Además: una cláusula de ALMACENAJE («congela porciones de 140 g») no es consumo
para V4/V6/V7a/V7d/V7e (`_texto_de_consumo`), y V7d compara ml con ml y g con g; cruzar familias exige una densidad
respaldada (`_V7D_DENSIDAD`: leche, caldo, aceite, miel…) — «400 ml de avena» contra «120 g de avena» ya no se compara con
una conversión inventada. Corpus fijo tras el lote: V7d 1, V8a 1, V8b 0 (nadie declaró equipo).

**Lo que NO hace, dicho.** No añade la pregunta del equipo al formulario principal (decisión de producto vigente); no
convierte V8a/V8b en rechazo; no toca la regla DURA del arroz en el DESAYUNO (sigue `hard` en RD); no reparte claras por
receta ni ajusta recipiente/técnica para raciones mayores (queda para C5 segunda parte, CUL-P1-02/04); no re-calibra al
juez (CUL-P1-06 depende de la rúbrica anotada de C1).


## Estructura del plato y la cadena de reparación medida (C5 · segunda parte (a) · `P1-PLAN-LOTE-27` · 2026-09-12)

**CUL-P1-02 — un contrato LIGERO de estructura, derivado de lo que el plato ya declara.** `dish_structure.contract(meal)`
→ familia (tortilla/revoltillo, panqueque, bowl, guiso, ensalada, tostada/wrap, batido/crema, otro — por el NOMBRE; la
«tortilla de trigo» es pan, no huevo), componentes (principal, soporte, líquidos en ml, sólidos en g, vegetales de agua —
**sólo con lo que la lista declara en g/ml: una pieza sin gramos no inventa peso**), relaciones de cantidades sensibles y
una confianza (`alta/media/baja` según cuántas líneas traen cifra). Es metadata de EVALUACIÓN: no escribe el texto del
usuario (`canonical_recipe.render_line` sigue reservado a compras y macros, como fijó la revisión).

**Los umbrales salen de la biblioteca curada, no de un número redondo.** Medido sobre las 193 recetas DO: wraps y
tostadas — relleno ÷ pan 2,4 · 2,5 · 3,8 (mediana 2,5) ⇒ umbral 5,0 (el máximo curado + 30 %); batidos y cremas —
sólidos ÷ líquido **mediana 0,90**, mínimo 0,26 ⇒ una «crema» que promete espesor con menos de 0,20 g de sólidos por ml,
≥ 150 ml y sin proceso que espese (reducir, cocer hasta espesar) ni espesante en la lista (avena, chía, guineo, yogur,
aguacate…) es la crema de 10 g de legumbre y 300 ml de leche del backlog; tortilla/revoltillo con vegetales de agua —
en 12 de 14 recetas curadas se sofríen o escurren ANTES del huevo (el huevo que cuenta es el que se VIERTE o cuaja, no
el que se bate en el mise en place; un vegetal que sólo aparece en «aliña… y sírvelo al lado» no va dentro). Los tres
hallazgos —`crema_sin_espesante`, `wrap_desproporcionado`, `tortilla_vegetales_crudos`— salen del escáner como **V9**
(`minor`, no reparable, con su evidencia en el detalle). Medido: **0 falsos positivos sobre las 193 recetas curadas y 0
sobre el corpus fijo** (48 de sus 64 comidas tienen confianza `baja`: la lista viene en tazas y piezas — se dice, no se
adivina); los tres casos del backlog disparan. Familias en el corpus: otro 35, batido/crema 7, panqueque 7, tostada/wrap
6, bowl 3, ensalada 3, guiso 2, tortilla 1.

**CUL-P1-04 — ajustar nutrición sin desarmar la receta: lo que ya estaba, y lo que faltaba.** Ya estaba: los
cerradores de banda y de piso de proteína re-escalan porciones EXISTENTES (jamás añaden un alimento, así que no ponen
pollo en un batido); la sustitución de huevo reescribe sus pasos (`_rewrite_recipe_steps_after_subs`); C2 y C3
sincronizan cantidades y formas en la cola real; V2 acusa el estado imposible. Lo que faltaba: **nadie comparaba el
plato antes y después de la cadena**. `repair_stage_diff` fotografía el plan con el escáner de capa 1 en tres puntos de
`db_plans._finalize_plan_data_for_insert` — ENTRADA (antes de la coherencia), TRAS LOS CAPS de realismo (el pase que más
mueve cantidades) y SALIDA (tras el contrato final, antes de restaurar los días congelados) — y escribe en
`plan_data["_repair_stage_diff"]` los conteos por check de cada etapa, los hallazgos NUEVOS con la etapa que los
introdujo, los resueltos y el coste en ms; loguea `warning` cuando la cadena introduce algo. No muta, no bloquea, no
repara: informa, para que el reparador de CUL-P1-04 tenga qué medir. Knob `MEALFIT_REPAIR_STAGE_DIFF` (default `True`);
sin catálogo o con más de 200 comidas se dice (`estado = sin_catalogo` / no se mide). El benchmark de superficies
(siguiente lote) lo reutiliza con `medir_cadena(plan, cadena)`.

**Lo que NO hace, dicho.** No adapta el relleno del wrap ni la técnica de la tortilla (V9 acusa; reparar es otro paso).
No reabre la sustitución por LP/MILP (decisión previa vigente). La medición por etapas sobre planes REALES llega con la
telemetría de cada plan persistido; al escribir esto la base no era alcanzable (red caída) y la medición offline se hizo
con fichas fijadas en la prueba. CUL-P1-06 (juez) y CUL-P1-07 (benchmark de superficies) van en el lote siguiente.


## El juez que admite la creatividad y el benchmark de todas las superficies (C5 · segunda parte (b) · `P1-PLAN-LOTE-28` · 2026-09-12)

**CUL-P1-06 — evidencia por componente, intención del plato y estado INCIERTO.** `CulinaryViolation` gana tres campos
con defaults compatibles con los jueces viejos: `componente` (el ingrediente o la parte del plato a la que se refiere la
queja), `intencion` (`tradicional` · `fusion` · `transformado` · `dieta` · `desconocida`) y `certeza` (`segura` |
`dudosa`). La rúbrica los pide y dice qué es `dudosa`: «alguna cocina, una intención declarada o un básico del
`contexto` podría justificarlo» — y que **una dudosa se OBSERVA, no bloquea**: en `review_plan_node` sólo las `segura`
deciden `action_taken=blocked`; todas quedan en `_culinary_judge_history` (observación antes de ampliar bloqueo
semántico, como pide el backlog). El esquema de hallazgo sube a `2026-09-12.certeza` (C1 lo lleva en `judge_context`,
así que dos juicios con esquemas distintos no se comparan sin saberlo). El marcador estricto excluye las `[dudosa]` del
juez salvo `--con-dudosas` (las cuenta aparte: `dudosas_excluidas`) y `--excluir-dev ids.json` saca los casos de
DESARROLLO del holdout (el corpus fijo del 09-12 y los 5 planes que se leyeron para diseñar C2-C4 no pueden ser también la
prueba); el sampler marca `[dudosa]` al construir los casos. **Lo que NO se hizo, dicho**: no hay calibración nueva del
juez — exige llamadas LLM reales (`scripts/calibrate_culinary_judge.py`) y la rúbrica anotada con 2.º anotador (C1,
dueño); hasta entonces `MEALFIT_CULINARY_JUDGE_GUARD` sigue `off` en producción (C6).

**CUL-P1-07 — `scripts/bench_superficies_culinarias.py`: el mismo caso por TODAS las superficies, medido con las mismas
fotos.** Cada superficie es una cadena determinista `cadena(plan_data)` medida con `repair_stage_diff.medir_cadena`
(hallazgos nuevos, resueltos, contrato final estampado, ms): `insert` (`_finalize_plan_data_for_insert`), `quality` y
`chunk_t2` (`apply_plan_quality_finalize_chain`, con y sin días congelados), `swap` y `modify`
(`finalize_single_meal_recipe_coherence` × comida → `apply_update_band_parity` → `apply_final_contract`: los tres pasos
que corren `/swap-meal/persist` y el callback de `modify_single_meal`), `closers` (banda + caps), `degradado` (el postfix
del día sin LLM: `_degrade_offending_steps`) y `expand` (`/recipe/expand` persiste lo que el LLM devuelve: **sin cadena
determinista propia**, se dice). Dobles offline: el catálogo con nutrición se exporta UNA vez con la base en sólo lectura
(`--exportar-catalogo` → `scripts/data/catalogo_nutricion_<fecha>.json`); sin él el bench corre con el catálogo del corpus
(nombres) y el informe avisa de que los cerradores no tienen con qué cerrar. Artefacto JSON con git sha, huella de reglas
y catálogo; `--informe A.json` lo reconstruye offline; `--comparar A B` da el informe PAREADO por superficie; `--real`
genera planes con el LLM y **se niega sin `--perfil` y `--presupuesto-usd`** (registra intentos, estado y coste
estimado desde `llm_usage_events`). Nada escribe planes de usuarios: copias en memoria, base sólo lectura.

**Medido (corpus fijo: 5 planes, 64 comidas; catálogo con nutrición del 12-sep; git `f51de0b3`).**

| superficie | nuevos | resueltos | contrato estampado | ms |
|---|---|---|---|---|
| insert | 0 | 54 | 20 | 100.373 |
| quality | 0 | 54 | 20 | 65.435 |
| chunk_t2 (días congelados) | 0 | 9 | 4 | 65.421 |
| swap | 0 | 54 | 20 | 32.529 |
| modify | 0 | 54 | 20 | 32.404 |
| closers | 0 | 0 | 0 | 21.466 |
| degradado (postfix sin LLM) | **21** | 16 | 0 | 34.468 |
| expand | — | — | — | sin cadena |

Tres lecturas. (1) **Swap, modify, insert y quality resuelven lo mismo (54) y estampan el mismo contrato (20 comidas)**:
la receta final de la semana 3 y la cambiada por swap tienen el mismo contrato — la aceptación del backlog, medida. (2)
**Ninguna cadena de producción introduce hallazgos nuevos** sobre el corpus con el catálogo real; con el catálogo de
NOMBRES (sin nutrición) el swap «introducía» 22 — la cifra de un doble sin nutrición no es comparable, y por eso el
artefacto declara `modo_catalogo`. (3) **El postfix del día degradado introduce 21** (V3 «comprado y sin mencionar» ×15,
V7c «seco sin cocción» ×5, V7a ×1): `_degrade_offending_steps` quita el paso que ofende y deja el alimento comprado sin
paso que lo use. Es un hallazgo real del camino sin LLM, no de esta medición; queda anotado para el reparador de
CUL-P1-04 y no se toca aquí.

**Lo que NO hace, dicho.** No calibra al juez (LLM + rúbrica del dueño); no compara la interfaz/PDF con los snapshots
(superficie de frontend, fuera de este lote); `--real` está implementado y no se ejecutó (gasto). Con esto C5 queda
cerrado en sus siete ítems, con los residuos escritos fila por fila en el plan.


## La cocción que falta: lo que el dueño más marcó y la máquina no veía (C5 · CUL-P1-05 · `P1-PLAN-LOTE-62` · 2026-09-15)

Lote 39 del plan 38-44. Con la línea base del lote 38 —columna `maquina_determinista_2026-09-15`, adjudicador estricto,
anotación del dueño— `coccion_faltante` (8 defectos, todos `high`) no tenía ningún código en la RUBRICA: nada podía
acertarla. `seco_sin_coccion` (11) mapeaba a V7c, que acertaba 3; `usa_lo_que_no_esta` (9) a V5, que acertaba 1.

**Medido primero** (paso 1 del plan, hecho ejecutable): la señal que el plan describe —crudo o seco en la LISTA y ningún
verbo de cocción sobre ese alimento en los PASOS— separa `coccion_faltante` 4/8 y `seco_sin_coccion` 5/11, con 0 de los 9
`ok`. Limpia, pero la mitad. Lo que no ve, leído caso a caso: la cocción que cuece OTRO alimento de la misma frase, el
alimento que un paso usa ya cocido sin que nadie lo cueza («maja el plátano verde cocido», «desmenuza la pechuga»), el
toque de sartén de 3-4 minutos que no cuece una yuca ni unas lentejas, y «crudo» donde V7c sólo leía «seco».

**Tres cambios, todos `warn`; ninguno de severidad:**

- **V7f `coccion_faltante`** (capa 1, `minor`, no reparable; knob `MEALFIT_CULINARY_V7F`, default `True`). Víveres
  (categoría «Víveres» sin `crudo`/`ninguno` en `prep_methods`) y proteínas animales no listas para comer que la lista no
  declara cocidos y que ningún paso cuece, o que un paso usa YA cocidos antes de cocerlos. Las legumbres y los granos secos
  siguen siendo de V7c. La cocción se busca cláusula a cláusula, en orden: en la que nombra el alimento; en la siguiente
  si habla de él con un pronombre enclítico («córtalos…; hornéalas 10-12 minutos»); en una que cuece sin nombrar otro
  alimento («Hornea unos 20-25 minutos»); y, si el alimento ya se mezcló, en la que cuece la preparación («vierte la masa y
  cocina»). Un participio describe, no cuece («la yuca horneada»), salvo el de resultado («hasta que estén cuajados»);
  fuego y tiempo sin verbo sí cuecen. En víveres, dorar o saltear menos de 8 minutos no cuece. Las notas de seguridad no
  cuentan. **Un hallazgo por comida**, con los alimentos nombrados: el dueño anota «Batata y pechuga de pollo» como un
  defecto, no como dos.
- **V7c amplía la FORMA.** «crudo/cruda» marca un grano o una legumbre sin cocer igual que «seco», y el verbo genérico
  (cocinar, cocer) con una duración explícita de menos de 8 minutos no cuece lo seco («agrega los garbanzos y cocina 3
  minutos»; «por lado» cuenta doble; pasta, fideos y avena quedan fuera). Si ningún paso nombra el alimento, no es una
  cocción que falta sino un huérfano: eso es de V3.
- **V5 mira la frase del alimento.** La ventana de «el paso lo nombra con más detalle que la lista» era de ±28 caracteres
  y cruzaba a la frase vecina: «lava los arándanos; separa las almendras fileteadas» daba las almendras por declaradas
  porque los arándanos sí están en la lista, y «125 ml de leche y 1 cucharada de…» encontraba «cucharada». Ahora va de la
  última coma o conjunción antes del alimento a la primera después, sin unidades de medida. Además, un hallazgo por
  alimento y comida, y lo que la receta HACE con lo que la lista trae no falta: el sofrito si un paso sofríe (el verbo,
  no la palabra «sofrito»), el adobo si antes de usarlo una frase mezcla dos o más de sus alimentos.

**Resultado.** Estricto, anotación del dueño, 75 casos adjudicables; `culinary_golden_score.py --estricto --desde
2026-09-15 --comparar-maquina 2026-09-15-lote62`. La columna nueva la escribió `culinary_golden_refresh.py` con el
catálogo del corpus fijo: sin base y sin LLM. El juez no se re-corrió; con `--desde`, la capa sin columna nueva se compara
con la de la línea base y no con la del 09-06.

| Clase del dueño (defectos) | Lote 38 (tp/fn) | Lote 62 (tp/fn) | Recall |
|---|---|---|---|
| `coccion_faltante` (8) | 0/8 | 8/0 | 0 → 100 % |
| `seco_sin_coccion` (11) | 3/8 | 11/0 | 27,3 → 100 % |
| `usa_lo_que_no_esta` (9) | 1/8 | 2/7 | 11,1 → 22,2 % |
| `cantidad_inconsistente` (19) | 10/9 | 10/9 | igual |
| `ingrediente_huerfano` (5) | 3/2 | 3/2 | igual |
| `verbo_alimento` (1) | 1/0 | 1/0 | igual |
| `paso_incoherente` (7), `nombre_no_corresponde` (2), `estructura_del_plato` (1), `masa_sobrante` (1) | 0 | 0 | igual |
| **Determinista** (tp/fp/fn) | **18/60/37** | **35/65/20** | recall 32,7 → 63,6 %; precisión 23,1 → 35,0 % |
| Hallazgos sobre los 9 `ok` | 0 | 0 | ningún FP nuevo |

La línea base re-puntuada da 18/60/**37** y no el 18/60/29 que publicó el lote 60: entonces `coccion_faltante` no tenía
código y sus 8 defectos no entraban en el denominador. Es la misma columna medida con la RUBRICA de hoy. Los 5 FP nuevos,
leídos uno a uno: 3 son defectos reales que el dueño no anotó en ese caso (una batata que sólo se calienta 3 minutos;
cilantro y aguacate que los pasos usan y la lista no trae, en dos casos); 1 es el mismo defecto anotado con otra clase
(«aplasta 3 huevos» en un paso que el dueño marcó `paso_incoherente`), y 1 es un defecto del dueño con dos legumbres que
V7c acusa por separado.

**La biblioteca y el día determinista.** Sobre las 193 recetas curadas, V7f dispara en 2 y las dos son verdaderas:
«Arepa de maíz con queso gouda» y «Domplines con huevo y queso gouda» sirven «el huevo revuelto» y ningún paso lo revuelve.
V5 deja de acusar el sofrito en 2 (el snapshot `recipe_usage_do_v1.json` se re-derivó). Como `verifica_comida` descarta al
candidato con CUALQUIER hallazgo del escáner, con `MEALFIT_RECIPE_LIBRARY_SELECT` encendido el día determinista pasaría
de rechazar 23 a 25 de 192 platos: esos dos desayunos, hasta que su receta diga cómo se revuelve el huevo. Con el knob en
su default (apagado) no se sirve ninguna receta de la biblioteca y nada cambia. Añadir el paso toca la redacción que el
dueño juzgó a ciegas: propuesto, no aplicado.

**Corpus fijo.** Cambian las reglas (`8f19140121f98a95` → `5b2a479ef69aef85`) y no las cifras por comida (44 de 64 con
hallazgo determinista, +0); re-congelado con `culinary_baseline.py --congelar`.

**Lo que cazó el gate.** Con la ventana nueva, el fixture BUENO `golden_02` («mezcla el aceite con el ajo, la naranja…»
→ «unta la pechuga con el adobo») salía como adobo que falta en la lista. `test_p1_culinary_golden` sólo corre con el
catálogo de la base y en el worktree se saltaba. Arreglado en el escáner (el adobo que la receta mezcla), jamás en el
fixture; y los cinco buenos se miden ahora también con el catálogo del corpus, sin base.

**Lo que NO hace.** No re-corre el juez (cero gasto) ni toca su mapa de códigos (lote 40). La regla «lo que el nombre del
plato declara no es un fantasma» sigue callando unos 4 de los 9 `usa_lo_que_no_esta` del dueño: es política, medida y no
cambiada. Ningún detector pasa de `warn` a bloqueo. Test:
[`test_p1_plan_lote_62.py`](../tests/test_p1_plan_lote_62.py).


## El reparador de estructura: adaptar sin desmontar (CUL-P1-04 · `P1-PLAN-LOTE-29` · 2026-09-12)

**Corrección primero.** El lote 28 dijo que el postfix del día degradado «introducía 21 hallazgos». Era el adaptador del
benchmark, no el cron: degradaba pasos por CUALQUIER check y el cron sólo degrada por V1/V2. Corregido el adaptador y
re-medido sobre el corpus: **1 hallazgo nuevo, no 21** (un V3: el paso degradado a «Sirve el X» borraba los otros alimentos
que nombraba). *Un doble que no imita al productor mide otra cosa con la misma pinta.*

**Qué repara `recipe_repair`** — la regla que lo hace seguro para la nutrición: **no toca la lista ni una cifra de compra;
sólo el texto de los pasos**. Lo que sobra se sirve APARTE con técnica o acompañamiento deliberado (la cláusula del
backlog), no se recorta ni se estira. Los tres defectos que V9 acusa (`dish_structure`, lote 27):

- `tortilla_vegetales_crudos` → se inserta, antes del paso que VIERTE o cuaja el huevo (no del que lo bate), «Saltea
  {vegetales} 3-4 min en la sartén con un chorrito de agua hasta que suelten el agua y escúrrelos bien antes de añadir
  el huevo / las claras» — como 12 de 14 recetas curadas; sin aceite añadido, ningún macro se mueve.
- `wrap_desproporcionado` → «Montaje: rellena la {tortilla} con lo que cierra (unos {3,8 × pan} g del relleno) y sirve el
  resto (~{sobrante} g) al lado, como ensalada» — 3,8 es el máximo curado, no un redondo; misma compra, mismos macros.
- `crema_sin_espesante` → «Ajuste de textura: usa solo {2 ml por g de sólido, mín. 30} ml de {líquido} en la crema y sirve
  los {resto} ml de {líquido} restantes como bebida al lado» — las dos cifras nombran el líquido para que V7d sume el total.

`dish_structure` reconoce lo servido aparte (`_WRAP_APARTE_RE`, `_CREMA_APARTE_RE`) y deja de acusar: la reparación es
**idempotente** y V9 baja a cero sobre el plato reparado. Corre como paso (4) de `recipe_contract.reconcile_meal`, detrás
de cantidades (C2) y formas (C3), bajo el mismo knob `MEALFIT_RECIPE_FINAL_CONTRACT` (`repair` aplica, `shadow` anota sobre
copia, `off` nada) y con su propia cuenta en la telemetría (`_recipe_contract_final.estructura`, sólo si reparó). Lo que
no se puede reparar se declara en `descartado`, no se calla.

**El camino sin LLM.** `cron_tasks._degrade_offending_steps` ya no borra a los vecinos: `recipe_repair.degradar_paso`
convierte «Sofríe la cebolla, el ajo y las habichuelas con el casabe» (ofende el casabe) en «Cocina Habichuelas rojas según
su envase…» + «Sirve el Casabe con Cebolla, Ajo y Habichuelas rojas.» — los otros alimentos siguen teniendo un paso (V3) y lo
que se compra seco se cuece (V7c); sin otros alimentos el texto es EXACTAMENTE el de siempre (los tests del 07-31 siguen
intactos). Bench del corpus, superficie `degradado`: **1 nuevo → 0**, 4 resueltos.

**Lo que NO hace, dicho.** No hay casos de V9 en el corpus fijo ni en la biblioteca (0 y 0): la reparación de estructura se
midió sobre los tres casos del backlog, no sobre la flota; aparecerá en `_recipe_contract_final.estructura` cuando la flota
la necesite. No repara técnica impropia (V1) ni estado imposible (V2) fuera del camino degradado: eso sigue siendo del
generador y del gate. No reabre LP/MILP.


## El benchmark en modo real, ejecutado: 3 planes recién generados por $0,05 y un hallazgo que nace en la cadena (CUL-P1-07 · `P1-PLAN-LOTE-30` · 2026-09-13)

**Qué se corrió.** `bench_superficies_culinarias.py --real` sobre tres perfiles del wizard (baseline masculino, DM2 con
metformina, vegetariana; `totalDays` 30) con tope de $0,50. El pipeline REAL (`arun_plan_pipeline`, Luna/flash por tier de
invitado) generó el bloque síncrono de 3 días de cada uno — 12, 9 y 12 comidas; 8, 7 y 9 llamadas al LLM; 650, 475 y 411 s
(el primero pasó cuatro rechazos del revisor por proteína repetida el mismo día) — por **$0,0536 en total**, contado en
proceso con la tarifa del propio emisor. Coste por plan de 3 días: 1 a 3 centavos.

**Lo que el modo real del lote 28 no hacía, y ahora hace.** El lote 28 dejó el modo real escrito y sin ejecutar; al ejecutarlo
faltaban cuatro cosas, y las cuatro están en el bench: (1) abrir los pools como el arranque de la app (fuera de FastAPI el
catálogo sale vacío); (2) imitar a `/analyze` (`_plan_start_date`, `_days_to_generate = PLAN_CHUNK_SIZE`, rebanada del
blueprint) en vez de llamar al pipeline a pelo; (3) **no escribir en la base del dueño**: las funciones de escritura de
`db_core` se sustituyen por dobles que cuentan y no ejecutan — la corrida habría dejado **158 filas de telemetría en
producción** (`pipeline_metrics` 109, `app_kv_store` 44, `system_alerts` 5) y una decena en `llm_usage_events`; `--telemetria-prod`
las deja pasar a sabiendas; (4) contar el coste EN PROCESO (`compute_llm_cost_micros`, envolviendo a `log_llm_usage_event`),
porque el presupuesto no puede depender de leer una tabla en la que ya no escribimos. Y `--planes-de <artefacto>` re-mide
sin LLM los planes de una corrida real: el pareado real que el protocolo pedía, sin volver a pagar.

**Lo que los planes recién generados traen de fábrica** (capa 1 a la ENTRADA de la cadena, es decir, lo que el pipeline
entrega): baseline V3 1, V7a 2, V1 1, V7e 5; DM2 V7e 5, V7a 2; vegetariana 0. Ninguna cadena determinista los resuelve — no
es su trabajo — y ninguna los empeora, salvo una.

**El hallazgo: la lista pierde una línea y el paso la sigue nombrando.** En el smoothie bowl del plan vegetariano la cadena
del INSERT (`insert` y `quality`, la misma SSOT) introdujo **1 hallazgo nuevo**: `_relevel_fats_universal` bajó la granola de
15 a 10 g para cerrar la banda de grasa y `_floor_subservible_portions` (GAP-05, «una línea de 10 g sin cabida calórica no se
sirve: se retira») **borró la línea** — pero el paso siguió diciendo «corona con la granola»: **V5 nacido en la propia
cadena de persistencia**, no en el LLM. Ningún cerrador que quita una línea toca los pasos, así que la reparación va en la
cola del contrato, para todos: **paso (5) de `reconcile_meal`** — `recipe_repair.retirar_sin_lista` retira la MENCIÓN del
alimento (el ítem de la enumeración «…linaza y 15 g de granola», el complemento «corona con la granola y…», o la frase entera
si no decía otra cosa), verifica con el propio detector V5 y **se deshace si la mención sobrevive, si la receta se vacía o si
abrió otro hallazgo** (V1/V3/V5/V7a-b-c-e sobre el mismo plato). La lista no se toca jamás. Telemetría:
`_recipe_contract_final.sin_lista`; lo que no pudo retirar, `sin_reparar.sin_lista`.

**Medido antes de escribir la versión definitiva.** La primera versión quitaba sólo la CABEZA del nombre («yogur» de «yogur
de coco») y, al no casar el ítem entero, tiraba la frase completa: sobre el corpus fijo abrió **3 V3** (mango, leche y maní
se quedaron sin paso). De ahí las dos reglas: el nombre entero antes que la cabeza, y una retirada que abre un hallazgo no
es una reparación. Pareado real (`--planes-de`, mismos 3 planes): `insert`/`quality` **1 → 0** nuevos, resto 0 → 0;
sellos del contrato 5 → 9. Corpus fijo (5 planes de producción): 0 hallazgos nuevos en las 8 superficies; `insert`/`quality`/`swap`/`modify` 54 → 55 resueltos (el «yogur de coco» del batido: un V5 que escribió el LLM y que el paso (5) retira sin abrir otro) y sellos 20 → 45 (los que ya no se pierden); `chunk_t2` 9 → 7 resueltos es la FECHA, no el código — los días pasados se congelan y hoy hay uno más; el código de HEAD corrido HOY da 7 → 7 y sellos 1 → 6.

**Y un sello que se perdía.** El pareado real enseñó 8 sellos `_recipe_contract_final` del pipeline convertidos en 5 tras el
INSERT: una pasada del contrato sin nada que decir BORRABA el sello de la anterior, y la flota perdía la cuenta de lo que el
contrato sí hizo. Ahora una pasada sin nada que decir no borra lo dicho; un plato que nunca necesitó nada sigue sin sello.

**Lo que NO se hizo, dicho.** Tres planes de 3 días no son la flota: son la prueba de que el modo real corre, cuesta centavos
y encuentra cosas. El `taste_profile` del router (LLM sobre historial) no se imita: un usuario nuevo no tiene historial. Los
hallazgos «de fábrica» (V7e ×10, V7a ×4, V1, V3) son del generador y del gate, no de las cadenas; quedan medidos, no
reparados. `/recipe/expand` sigue sin cadena determinista que medir.

## Los V7e «de fábrica»: el paso pide más piezas de las que la lista compra (`P1-PLAN-LOTE-31` · 2026-09-13)

**Medido primero.** El lote 30 dejó 10 V7e en los 3 planes recién generados y los declaró «del generador y del gate, no de las
cadenas». Mirados uno a uno no eran una cosa sino tres:

1. **Siete** eran la lista bajando de 2 a ½ (o a 1) — el motor de macros o un cerrador dejaron «½ plátanos verde», «½ cebollines»,
   «½ ciruela», «1 tortilla integral» — con el paso diciendo todavía «pela los 2 plátanos verdes», «pica fino los 2 cebollines»,
   «divide las 2 ciruelas», «calienta las 2 tortillas integrales». El contrato de C2 los veía y no los tocaba: una pieza desnuda
   cuyo número cruza el 1 quedaba en `gramatical`, porque cambiar sólo el número produce «½ plátanos verdes». Los sellos lo
   decían (`sin_reparar.gramatical` 3, 1, 1, 1 en los cuatro platos).
2. **Tres** eran el detector sumando dentro del paso: «separa 3 huevos y 6 claras de huevo reservando 6 claras en un bol» son las
   mismas seis claras y V7e leía 12 contra 6; «… y 6 claras de huevo y 6 claras de huevo y 6 claras» leía 18.
3. Y esa última frase era **el LLM repitiéndose**, no la cadena: el sello del contrato registra 6 piezas recortadas (el modelo las
   había escrito con OTRO número, seis veces) y el reescritor de la forma del huevo es idempotente (probado ×3 sobre cuatro
   reconstrucciones del original). Nadie colapsaba la repetición; V7e la contaba.

**Qué se hizo.** (a) `recipe_contract.reconcile_step_quantities` reescribe CON CONCORDANCIA cuando el paso pide MÁS de lo comprado
(`_concordar_pieza`): el número, el artículo («los 2» → «el ½»), el sustantivo token a token contra el NOMBRE CANÓNICO del catálogo
—con sus tildes: «cebollines» → «cebollín»; recortando letras salía «cebollin», y un alias que ya es plural («plátanos verdes» está
en el índice tal cual) no dice cuál es su singular—, los adjetivos que le siguen («ciruelas frescas» → «ciruela fresca») y, si la
cláusula sólo habla de ese alimento, sus clíticos («córtalos en 4 trozos cada uno» → «córtalo en 4 trozos»; «resérvalas enteras» →
«resérvala entera»). Si la lista compra exactamente 1 y hay artículo, el número sobra: «divide las 2 ciruelas» → «divide la
ciruela». La dirección contraria (el paso pide MENOS: «pica 1 tomate» con 3 en la lista; 21 menciones en el corpus frente a 3 en
esta) sigue en `gramatical`: es la decisión V7a que el dueño tiene en su hoja, y se deja donde estaba a propósito. Lo que no sabe
concordar sin dejar un residuo peor («1½ tostones de casabe»: el singular pide una tilde que el texto no trae; «2 rodajas de
tomate» cuenta rodajas, no tomates) se declara y no se toca. (b) V7e compara la mención MAYOR del paso, no la suma
(`_v7_piezas(..., agregar="max")`); la lista y V7a siguen sumando, porque ahí sumar es lo correcto. (c) Paso (6) del contrato:
`recipe_repair.colapsar_repeticiones` deja una vez la misma mención numérica (alimento, familia, cantidad) repetida en cadena
—unida sólo por «y»/«e»/coma—, lo verifica con los detectores de capa 1 y se deshace si abre un hallazgo; «6 claras de huevo
reservando 6 claras» no es una cadena y no se toca. Telemetría: `_recipe_contract_final.concordancia` y `.repeticiones`.

**Medido después.** Pareado real (`--planes-de`, mismos 3 planes, `bench_superficies_replay_2026_09_13_l31.json`): V7e tras el
INSERT **10 → 0** — 7 concordados, 3 que eran la suma; el colapso deja «separa 3 huevos y 6 claras de huevo.» — con 0 hallazgos
nuevos en las 8 superficies; `insert` resueltos 0 → 6 y sellos 9 → 9; `chunk_t2` 7 → 2 (sus días pasados se congelan: la fecha, no el código); `closers` y `degradado` no corren el contrato y quedan en 7, como antes. Corpus fijo (5 planes de producción): 0 hallazgos nuevos en las 8 superficies; `insert`/`quality`/`swap`/`modify` **V7e 38 → 1** (antes 38 → 3: caen «las 2 tostadas de casabe» → «la tostada de casabe» y «2 plátanos verdes» → «½ plátano verde»; el que queda es «1½ tostones de casabe», declarado `gramatical` a propósito), resueltos 55 → 57, sellos 45 → 45 y V7a 42 → 42 — la otra dirección no se tocó, como se decidió. Cada reescritura es idempotente (la segunda
pasada no cambia nada) y ninguna toca la lista.

**Lo que NO se hizo, dicho.** La dirección V7a numérica y «la clara» sin número siguen siendo avisos: decisión del dueño. Los verbos
que conciertan con el sustantivo («hasta que estén») no se tocan. No se reescriben sustantivos de forma. La repetición del LLM se
colapsa en la cola del contrato y no se «evita» en el prompt: un prompt no garantiza nada y la cola sí. Y la lección del lote:
**diez hallazgos con el mismo código no son un defecto** — eran tres, uno de ellos del instrumento; hasta mirarlos uno a uno el
lote 30 los había dado por «de fábrica» en bloque.

## C7 medido: variedad perceptible, presupuesto de reparación y deriva (`P1-PLAN-LOTE-35` · 2026-09-13)

Tres instrumentos de solo lectura para la parte MEDIBLE de CUL-P2-01/02/03. Ninguno cambia un veredicto ni una comida.

**Variedad perceptible (CUL-P2-01)** — `scripts/measure_variedad_perceptible.py`. «Pollo con arroz y ensalada» renombrado
siete veces son siete nombres y un plato; el reporte de variedad del grafo cuenta repeticiones del MISMO día. La **firma de
preparación** separa nombre de plato: `_template_id` si viene de la biblioteca; si no, (familias de proteína, técnicas de
cocción, bases de carbohidrato), con los vocabularios que ya existen (`_MAIN_PROTEIN_ALIASES`, `VERB_TO_METHOD`). **Nada del
nombre entra**: la primera versión metía el formato sacado del nombre y un renombre cambiaba la firma — el test del
«pollo con arroz renombrado siete veces» lo destapó. Medido (artefacto `scripts/data/variedad_perceptible_2026_09_13.json`):

| Fuente | Días | 7 días | 15 días | 30 días | Ventanas de 7 d sobre el tope |
|---|---|---|---|---|---|
| día determinista (`deterministico:DO:30d`) | 30 | 18 nombres / 18 platos / 0 renombrados | 32 nombres / 32 platos / 0 renombrados | 43 nombres / 43 platos / 0 renombrados | 14 de 24 |

Las ventanas rotas las causan **dos** plantillas: Pollo guisado con bollitos de plátano; Sardinas en lata con casabe — servidas
más de 2 veces en 7 días CON la memoria entre días del lote 2, porque en su franja no hay con qué rotar (la cena sirve
10 platos distintos en 30 días, el desayuno
16). Compra reusada con otra técnica en 30 días:
huevo, pescado, pollo, yogurt; con una sola técnica:
atun, cerdo, gandules. La primera corrida salió con 30 días VACÍOS (el pool de la base
no se abría fuera de FastAPI y el catálogo llegaba vacío) y el informe decía «0 platos»: ahora una ventana sin comidas se
declara no medida.

Los planes del modelo que hay para medir (corpus fijo y bench real) tienen [3, 4] días: ninguna ventana de 7 cabe, y el
instrumento lo dice en vez de extrapolar. Medir 7/15/30 días del modelo exige planes largos generados — con coste, decisión
del dueño.

**Presupuesto de reparación (CUL-P2-02)** — `scripts/measure_presupuesto_reparacion.py`. Cuenta, por comida, las capas que
la REESCRIBIERON (32 marcas con lo que hace cada una; los diagnósticos que sólo observan no cuentan, y el solver de porciones
es composición y va aparte), más las reescrituras del contrato final. Corpus fijo: 64 comidas, capas p50
3 · p95 7 · máx 7, 59.4 % con ≥ 3 capas; las más frecuentes: piso de porciones 48, tope de realismo de porciones 37, cerrador de macros/micros 28, cerrador de proteína 22, sustitución por presupuesto 18.
Planes reales del bench: 33 comidas, p50 3 · p95 6, 69.7 % con ≥ 3.
Coste: 3 planes reales por $0.0536, 1 válidos
tras el INSERT → $0.0536 por plan válido. **Propuesta, no implementada**:
`MEALFIT_RECIPE_REPAIR_BUDGET = 8` (p95 + 1) como aviso cuando una comida necesita más capas que el
95 % de las que ya se entregan; cortar reparaciones cambia qué se entrega y es decisión del dueño.

**Deriva y cobertura (CUL-P2-03)** — `bench_superficies_culinarias.py --informe A --desglose` reparte lo medido por cohorte
(el perfil del bench real; «sin_dato» cuando el corpus no lo guarda), por semana (del día de cada hallazgo nuevo) y añade una
línea de **cobertura** por superficie (planes medidos / planes): un plan que no se pudo medir se ve como menos cobertura,
nunca como más calidad. `judge_violation_rate.py --por-pais` desglosa cada fecha por `plan_data._country`: la regresión de un
país no se esconde en la media. Hoy los 6 planes vivos son DO, así que el desglose por país tiene una sola fila — el
instrumento está listo para cuando no.

**Lo que NO se mide aquí, dicho.** CUL-P2-04 (cocinado real: panel y cocina, no se sustituye con un LLM), los P3 (gustos de
preparación, biblioteca de variantes, «magia» con usuarios: producto) y la calibración del juez con la rúbrica anotada
(dueño). El presupuesto se propone con su cifra; no se cablea.


## El orden de la receta: rotular en su sitio y medir el desorden (`P1-PLAN-LOTE-45` · 2026-09-14)

**Qué pasaba.** El reparador del contrato de pasos (`graph_orchestrator._repair_recipe_contract`, P1-RECIPE-CONTRACT-REPAIR)
nació para una receta del modelo con la cocción atrapada dentro del «Mise en place». Ante «falta 'El Toque de Fuego'»
EXTRAÍA de todos los pasos cada oración con verbo de cocción y la llevaba a un único paso delante del Montaje. En una receta
NARRATIVA —las 193 de la biblioteca, sin rótulos— eso la desordena: en el plan del dueño del 14-sep «Escúrrelos bien» y
«maja el ajo con el plátano» quedaban antes de hervir el plátano, y los cinco `paso_incoherente` del juez eran eso. Medido
sobre la biblioteca completa: 149 de 193 recetas salían con oraciones fuera de su orden (281 oraciones).

**Qué cambia.** `recipe_order.rotular_fuego_en_su_sitio` antepone «El Toque de Fuego: » al primer paso de cocción sin rótulo
(el primero que ya dice tiempo o temperatura, si lo hay) y no mueve nada; `rotular_montaje_en_su_sitio` rotula el último
paso si empieza sirviendo (73 recetas), en vez de añadir detrás un segundo «sirve» genérico. La extracción queda para lo que
nació: cocción dentro de un Mise o un Montaje sin pasos narrativos. Ahora: 0 recetas desordenadas. Knob
`MEALFIT_RECIPE_TDF_IN_PLACE`.

**Y se mide.** Los códigos V no ven un paso que usa un resultado antes del paso que lo produce. `recipe_order.fuera_de_orden`
empareja cada oración servida con la de la biblioteca (Jaccard de palabras ≥ 0,6, así que sobrevive a los retoques de
cantidad) y cuenta las que están fuera de su orden (emparejadas − la subsecuencia creciente más larga). `repair_stage_diff`
lo hace en las tres fotos y el informe lleva `orden = {"etapas", "medidas", "desordenadas", "detalle"}`; si la salida está
más desordenada que la entrada lo avisa (`[P1-PLAN-LOTE-45] … la cadena de reparación desordenó N oración(es)`).

**El Mise de plantilla ya no cocina.** La plantilla que el reparador antepone decía «…ten todo listo antes de
**cocinar**», y `cocin\w*` es señal de fuego para `_meal_is_no_cook`: al añadirla a un plato frío (batida, casabe con
aguacate) el re-lint lo leía cocinado y pedía un «El Toque de Fuego» inexistente, así que 15 de las 193 recetas de la
biblioteca salían con el badge «Receta con pasos incompletos». Ahora dice «antes de empezar»: 0 recetas con residual.

De paso, «sofrito» y no «sofrit» en `_STEWY_DISH_HINT`: «la cebolla sofrita» encima de un mangú no es olla, y el cerrador
de proteína le había escrito «añade el arenque al guiso» a un plato sin guiso. Test: `tests/test_p1_plan_lote_45.py`.

## El ingrediente que da nombre al plato y los pasos de los parches (`P1-PLAN-LOTE-46` · 2026-09-14)

**Qué pasaba.** En la segunda prueba del dueño salieron un guacamole sin aguacate, un «maní tostado» sin maní, uno «con
dátiles» sin dátiles y una avena «con leche evaporada» sin leche evaporada. El re-trim de grasas del guardado recorta
«aceite, queso, aguacate» sin mirar qué hace a ese plato ese plato, y otros ajustes de macros dejaban en cero la fuente que
sobraba. Los pasos seguían mandando tostar el maní que ya no estaba, y el contrato final retiraba el aguacate de los pasos
porque la lista ya no lo traía. Además dos parches escribían pasos sin sentido: el paso del cerrador de proteína no
reconocía «queso» a secas ni la mozzarella («Cocina queso a la plancha o hervido») y el autofix de proteína repetida
cambió el huevo del guacamole por pollo sobre pasos escritos para un huevo («la pechuga se pesa sin cáscara… pélala»).

**Qué cambia.** `identidad_plato`: en un plato de biblioteca, la IDENTIDAD son los constituyentes que el nombre nombra más
el más pesado de la plantilla. Los re-trims de grasa y carbohidrato no tocan esas líneas (`_identidad_protege`) y recortan
de las demás fuentes; si aun así falta, vuelve con el 25 % de los gramos de la plantilla por el factor del plato, en la
lista y en raw, antes del contrato final y del truth-up. Nunca vuelve lo que otro pase sustituyó a propósito ni lo que
choca con una alergia. El cerrador trata cualquier queso como lácteo que no se cocina (`_CHEESE_WORDING_HINT`), el autofix
no reescribe recetas congeladas y la harina con que se empaniza («pasa cada trozo por la harina, cubriéndolo») ya no se
toma por harina «de cumplimiento».

Simulación de solo lectura del día determinista, 7 días en dos bloques sobre el blueprint del último run del dueño (tiempo «Nada»), contra sus 3 planes más recientes: platos de esos planes 15 → 5 de 28, repeticiones dentro del bloque 4 → 2, días con proteína repetida 4 → 0, platos distintos 19 → 21, minutos medios 19,6 → 20,9. Identidad sobre una copia del plan 63eedc6b: vuelven 4 alimentos (+8 g de maní, +24 g de leche evaporada, +6 g de dátiles, +38 g de aguacate) y la grasa de los tres días queda entre el 91 % y el 107 %. La primera versión (piso del 50 % y subir también lo que quedó pequeño) tocaba 7 platos, llevaba la grasa al 116-122 % y el salami de 5 a 41 g: por eso el piso es del 25 % y sólo vuelve lo que falta. Test: `tests/test_p1_plan_lote_46.py`.

## Lo que se hace con un alimento lo decide el alimento que queda (`P1-PLAN-LOTE-47` · 2026-09-14)

**Qué pasaba.** En la tercera prueba del dueño tres pasos contradecían al alimento que nombraban. El tiempo por defecto
de «El Toque de Fuego» cayó en el paso que enfría los huevos («Pásalos a agua fría para cortar la cocción… (~10-12 min a
fuego medio)»). La quinoa de una cena pasó a arroz integral por presupuesto y el arroz a casabe por la regla del arroz de
noche: el nombre cambió dos veces y la técnica se quedó («Enjuaga 30 g de Casabe», «Cocina Casabe en agua hasta que
ablanden»). Y el autofix del tope de huevo cambió el huevo de la arepa por queso blanco y dejó su verbo («revuelve queso
blanco en una sartén»).

**Qué cambia** (`pasos_sustitucion`). Un paso que enfría sin calentar no recibe tiempo de fuego. Tras cambiar el arroz
por casabe, la frase que lo hierve pasa a tostarlo (una sola vez) y la que lo enjuaga, a tenerlo a mano. Tras cambiar un
huevo por queso, «revuelve / bate / cuaja» pasan a «dora / desmenuza» en un queso que se dora y a «incorpora / mezcla» en
uno que no; «queso blanco revuelto» pasa a «queso blanco dorado». Texto puro: no toca cantidades ni macros.

Además, la causa de fondo de que esos pasos llegaran a ese plan: el corrector de la autocrítica reescribía días enteros
y dejaba los platos de biblioteca sin su marca, así que la receta congelada ya no los protegía — ver
`docs/deterministic_day.md` («re-elegir, no reescribir»). Test: `tests/test_p1_plan_lote_47.py`.

## Lo que un cerrador añade, como la receta lo haría (`P1-PLAN-LOTE-48` · 2026-09-14)

**Qué pasaba.** En la cuarta prueba del dueño (plan 358a2cdf) 11 de 12 comidas llegaron con su receta de biblioteca,
pero los cerradores de macros les colgaron cosas que la receta no haría: el piso de calorías de ganancia muscular puso
arroz junto al mofongo y batata junto a los bollitos de plátano; el cerrador de proteína mandó a la licuadora queso
cottage dentro de un jugo de chinola; y el queso que añaden salía «queso» a secas, que la lista compra como queso
blanco: el plato se llamaba «…con queso cottage» y la compra traía otro. Además, el cambio del arenque por pescado
fresco borraba entera la frase del locrio que decía «(ya desalado y en trozos)» —y con ella el pescado y el arroz—, y
la chinola de un jugo llegó a 335 g (unas 14 frutas).

**Qué cambia** (`cierres_con_receta`, `pasos_sustitucion`). El piso de calorías escala hasta ×1,5 la base que la
receta de biblioteca ya trae, en la lista del plato y en la de compras, y si no cabe salta esa comida: nunca una segunda
base. En un jugo lo que añade el cerrador va al lado (el lácteo no se licúa: el ácido lo corta). El «queso» a secas toma el nombre del único queso que el plato
promete (en el nombre o, si no, en sus pasos), en las dos listas, antes de que el lácteo del nombre se inserte aparte y
de que el barrido de líneas muertas lo tome por sobrante; con ninguno o con dos, no se adivina. Del desalado se quita la
cláusula, no la frase. Las frutas de pulpa tienen techo propio (120 g, `MEALFIT_REALISM_PULP_CAP_G`). Test:
`tests/test_p1_plan_lote_48.py`.

## El plato, el día y la receta, hasta el final (`P1-PLAN-LOTE-49` · 2026-09-14)

**Qué pasaba.** En la quinta prueba del dueño (plan a059d7bb) los platos de biblioteca llegaron sin su ingrediente:
guacamole con 5 g de aguacate, «maní tostado» con 5 g de maní, casabe con 2,7 g de mantequilla de maní — y el día 1 al
71 % de su grasa: el recorte ni siquiera hacía falta. El lote 46 sólo devolvía lo que FALTABA, y 5 g cuentan como
presente. El cerrador de proteína puso huevo al desayuno de dos días que ya tenían huevo en otra comida y 10 g de
arenque en una merienda, y su paso se fundía en el primer paso con fuego («Añade camarones al guiso en los últimos
minutos» dentro del que hierve la yuca). El tope de huevos dejó «3 claras de huevo» en la lista y 165 g de huevo entero
en la compra. Tras cambiar las sardinas por pescado fresco, el locrio seguía con «ya escurridas» y «el líquido de la
lata», y un «Tuesta el casabe brevemente» recibió «(~10-12 min a fuego medio)».

**Qué cambia** (`identidad_plato`, `cierres_con_receta`, `pasos_sustitucion`). En la cola del guardado, lo que quedó por
debajo del piso sube al piso si el día cabe en su techo (medido; nunca baja). En desayuno y merienda el cerrador busca un
lácteo que el día no tenga antes de repetir o de pegar pescado, ningún cerrador añade un curado, y en recetas de
biblioteca su paso va al emplatado («al lado») o después del último paso con fuego. El tope de huevos reescribe la única
línea de huevo entero de la compra cuando no se empareja por alimento. Tras un cambio de lata se quitan «escurridas» y
«el líquido de la lata»; lo breve recibe un tiempo breve. Réplica de solo lectura sobre el plan entregado a059d7bb: en la cola del guardado el día 1 recupera 38 g de aguacate en el guacamole (tenía 5), 8 g de maní (5) y 7 g de mantequilla de maní (3), y pasa del 89 % al 92 % de sus kcal y del 71 % al 86 % de su grasa; los días 2 y 3 no cambian (el 2 ya está en su techo). La primera versión medía los gramos con el lector de la lista del contrato, que lee «57.2 g» como 2 g, y en la réplica «subió» la soya de la cena del día 3 de 57 a 18 g: ahora mide con el lector de la base y nunca baja.

**Hallazgo aparte, cerrado en el lote 52.** El lector de la lista del contrato (`recipe_contract._cantidades_lista`, V4)
leía mal los decimales con punto: «57.2 g» → 2 g, «2.69 g» → 69 g, «57.37 g» → 37 g (con coma, bien). Test:
`tests/test_p1_plan_lote_49.py`.

## El punto entre dos cifras no es un fin de oración (`P1-PLAN-LOTE-52` · 2026-09-15)

**Qué pasaba.** El defecto no estaba en `recipe_contract`, sino en la frontera de oración que comparten el medidor y el
reparador. `_SENTENCE_BOUNDARY_RE = [.;]` partía «0.23 g de Sal» en «0» | «23 g de sal», así que V4 y
`_cantidades_lista` leían 23 g (×100 en los condimentos) y «57.2 g» como 2 g. `dish_structure._veg_va_dentro` partía
con la misma expresión: «Sirve con 1.5 tazas de repollo rallado» metía el repollo dentro del plato. Y la plantilla «a la
plancha» del cerrador se comía la frase sólo hasta el «2» de «2.5 min».

**Qué cambia.** La frontera pasa a ser `(?<!\d)\.|\.(?!\d)|;`: el punto con una cifra a cada lado no parte, y «Añade 2.
Luego…» sigue partiendo. `dish_structure` usa `clause_bounds` (la misma frontera, no una copia) y la plantilla de la
plancha admite el decimal. Un test impide volver a escribir `r"[.;]"` pelado en un módulo de producción.

**Medido** (solo lectura, sobre los 11 planes de los últimos 21 días). 69 de 172 comidas llevan alguna mención con
decimal de punto: 21 de 1.388 líneas de `ingredients`, 100 de 1.394 de `ingredients_raw` y 3 de 712 pasos, casi todas de
condimentos («0.23 g de Sal», «1.19 g de Ajo»). Con una frontera y con la otra, el scan da las MISMAS violaciones por
código y el reparador el MISMO resultado en las 112 comidas vivas. Además, ningún paso entregado dice el número de detrás
del punto. Hoy es inerte porque los pasos casi nunca citan gramos de un condimento, pero no lo será mañana:
`formatear_cantidad` escribe los decimales con punto. El único camino donde sí mordió, la identidad del plato, ya medía
con el lector de la base desde el lote 49. Test: `tests/test_p1_plan_lote_52.py`.
