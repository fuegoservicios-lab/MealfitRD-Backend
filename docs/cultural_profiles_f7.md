# Cultura separada del mercado — Fase 7 del roadmap 2.5 (`P1-ARQ25-F7-CULTURE` · 2026-09-05)

Doc canónica de la Fase 7 («Cultura separada del mercado (capa V2.4, acotada)», roadmap §9 e invariante I16).
Cuatro subfases cerradas en el mismo día: **A** motor, **B** formulario y panel, **C** benchmark, **D** bibliotecas.

## 1. La regla (I16)

Un plan tiene DOS países, no uno:

| Puerta | Función SSOT | Decide | Superficies |
|---|---|---|---|
| **Mercado** | `constants.country_for_form_data` | precios, catálogo, moneda, unidades, validación contra la despensa | `_get_fast_filtered_catalogs`, cierre de proteína, presupuesto, lista de compras |
| **Cocina** | `constants.cultural_country_for_form_data(form_data, day_index=None)` | qué platos inspiran el plan | brief del día (`build_day_assignment_context`), prompt del sistema del day-gen, hábitos de franja (`slot_rules_for_country`), arroz nocturno, crítica, juez culinario, `dish_library` |
| **Cocina (plan ya generado)** | `constants.cultural_country_for_plan(plan_data, health_profile, day_index=None)` | la cocina SELLADA en `_plan_policy.effective.culture_weights` (sin sello: la elección viva del perfil; sin nada: el mercado del plan) | `/swap-meal` y `/regenerate-day` (el endpoint hidrata `data["_culture_weights"]` desde el plan y `agent.swap_meal` deriva `_swap_culture` para inspiración, plantilla del prompt, reglas de franja y feedbacks de retry), `/swap-meal/persist` (finalizador y backstop de franja), coach `execute_modify_single_meal` (`_modify_culture`). Catálogo, despensa y cierres de proteína siguen con el mercado (`_swap_country` / `_modify_country`). |

Gate del roadmap, cubierto por `test_f_gate_i16…`: `market_country=US` + `dominican_criolla` 0,7 ⇒ platos criollos con precios y catálogo de US.

Con el knob `MEALFIT_CULTURAL_PROFILES` apagado o sin elección de cocina, la puerta cultural devuelve el país de compra:
**legado byte-idéntico**. Y la cocina «por defecto» sale de la puerta de MERCADO (que respeta `MEALFIT_COUNTRY_SYSTEM`),
no del campo crudo: con el sistema de países apagado todo sigue siendo DO.

## 2. Motor paramétrico (`backend/cultural_profiles.py`)

Seis perfiles como **DATA** (`PROFILES`): biblioteca, mercado por defecto, básicos, familias de plato, técnicas, base de
sabor, afinidad por franja. Ninguna rama `if/elif` por cultura (§9.4; el test lo escanea).

| Perfil | Biblioteca | Mercado por defecto |
|---|---|---|
| `dominican_criolla` | `dish_templates.json` | DO |
| `puertorico_criolla` | `dish_templates_pr.json` | PR |
| `mexico_casera` | `dish_templates_mx.json` | MX |
| `colombia_casera` | `dish_templates_co.json` | CO |
| `spain_mediterranea` | `dish_templates_es.json` | ES |
| `us_everyday` | `dish_templates_us.json` | US |

Mezcla: principal + hasta **2** secundarias con intensidad (`ocasional` 0,15 · `frecuente` 0,30 · `predominante` 0,45);
`normalize_weights` garantiza principal ≥ 0,5 y suma 1. `profile_for_day(weights, day_index)` reparte los días de forma
**determinista** (0,7/0,3 ⇒ 7/3 de cada 10, siempre en el mismo orden, la principal abre). El blueprint (`horizon.build_blueprint`)
estampa `culture` por día y franja y `culture_weights`; el registry del blueprint y `registry_prompt_lines` leen la
biblioteca de la cocina DEL DÍA (`library_hashes`); el encabezado de inspiración nombra la mezcla
(«INSPIRACIÓN: COCINA DOMINICANA 70 % · COCINA ESPAÑOLA 30 %») y conserva el literal histórico para un perfil solo.

Campo del formulario: `cultureProfiles = {main, secondary: [{profile_id, intensity}]}` (en `FORM_V2_FIELDS`); la política
efectiva lleva `culture_weights` (`plan_policy._culture_weights_for`).

## 3. Formulario y panel (subfase B, frontend)

- `frontend/src/config/cultures.js`: espejo de `PROFILES` (ids, etiquetas `i18nKey`, país), intensidades, tope, normalización
  y `cultureWeightsSummary`. Knob `VITE_CULTURAL_PROFILES` (encendido salvo `0/false/off`); el paso además exige `COUNTRY_SYSTEM_UI`.
- `QCulture.jsx` («Cocinas que te representan»), justo después de `QCountry`, opcional con Siguiente interno: la cocina del
  país de compra se muestra **sugerida** y NO se siembra (`cultureProfiles` nace `null`; lección P1-COUNTRY-SYSTEM-F0).
  Sin inferencias de origen, idioma ni zona horaria: elige la persona. Copy en los 5 idiomas (28 claves, gate estricto al 100 %).
- `PlanPolicyPanel`: «Cocina: Cocina dominicana 70 % · Cocina española 30 %» y «(la de tu país de compra)» cuando es la sugerida.
- Contrato frontend↔backend anclado por `test_p1_arq25_f7_culture_front.py`.

## 4. Bibliotecas (subfase D)

Barra de cobertura por biblioteca (`test_p1_arq25_f7_culture_libraries.py`): ≥ 80 plantillas; desayuno ≥ 18, almuerzo ≥ 28,
cena ≥ 22, merienda ≥ 16; ≥ 10 familias de proteína; ≥ 12 técnicas; vocabulario unificado; 100 % de constituyentes resueltos
en el snapshot (DO 99,3 %: los 4 declarados sin resolver de siempre).

| Biblioteca | Antes | Ahora | Des/Alm/Cen/Mer |
|---|---|---|---|
| DO | 87 | **97** | 23/41/31/32 |
| ES | 55 | **83** | 21/46/36/22 |
| MX | 49 | **83** | 21/35/28/22 |
| CO | 51 | **86** | 20/34/25/22 |
| PR | 48 | **86** | 19/33/22/20 |
| US | 48 | **87** | 21/36/25/16 |

Lo que se corrigió de paso: las bibliotecas PR/US nacieron con `ninguna`/`lacteo`/`mixto`/`mariscos`/`a la plancha`/`ninguno`,
vocabulario que `dish_library._protein_matches_pool` NUNCA casaba con el pool del día — eran plantillas muertas (el 31 % de US
tenía técnica `ninguna`). Cinco snacks sin valor nutricional salieron de US (s'mores, pretzels con mostaza, miel con pecanas,
tater tots con barbacoa, hash browns con kétchup). Las plantillas DO nuevas llevan constituyentes **inline**:
`dish_registry._constituents_source` cae a ellos cuando la tabla curada (`data/dish_constituents_do.json`) no tiene la entrada.

Regla de franja heredada de F2 (`test_p1_country_system_f2`): arroz o pasta como **base** nunca en desayuno ni cena.

Cómo ampliar una biblioteca: añadir la plantilla al JSON con `constituents` de NOMBRES DEL CATÁLOGO (o alias), recompilar
(`python scripts/compile_dish_registry.py`, `--check` verifica reproducibilidad), correr el benchmark (`python cultural_benchmark.py --write`)
y el baseline del guard C3 si tocaste DO (`scripts/gen_do_corpus_retarget_baseline_2026_08_18.py`).

## 5. Benchmark cultural (subfase C, `backend/cultural_benchmark.py`)

Sin LLM, reproducible, sobre los snapshots compilados. Nueve medidas (§13.4): resolvabilidad, cobertura, contaminación
cultural (léxico EXCLUSIVO de platos de otra cocina — no de ingredientes: «huevo» no es de nadie), adecuación técnica/franja,
disponibilidad (cultivares dominicanos fuera de DO: señal para el revisor, solo rompe el gate por encima del 35 %), diversidad,
mezcla coherente (30 parejas 0,7/0,3 en 10 días con candidatos en las 4 franjas), cero bypass clínico
(`template_candidates(..., exclude_allergens=[clase])` jamás devuelve la clase) y revisión humana.

Informe committed: `data/registry/cultural_benchmark_v1.json` + `docs/cultural_benchmark_report.md`. Gate **PASA** con los seis
perfiles.

**Revisión curatorial (2026-09-05, delegada por el dueño a Claude)** — registro en `data/registry/cultural_curation_review_v1.json`,
atado al `snapshot_hash` de cada perfil: cambiar una biblioteca **caduca la firma** y deja el perfil «pendiente» (y el test en rojo)
hasta repetir la revisión. Se revisaron las 516 plantillas con un chequeo independiente por plantilla (nutrición por ración, alérgenos
esperados desde los constituyentes, técnica vs constituyentes, nombre vs constituyentes, duplicados, franja) más juicio plato a plato.
Lo que la revisión encontró y corrigió:

- **Alérgenos**: el matcher del registry usaba frontera de palabra estricta y el vocabulario está en singular — «Sardinas en lata»,
  «Fideos», «Almejas» no marcaban pescado/gluten/mariscos, y la salsa de soya (lleva trigo) no marcaba gluten. Corregido en
  `allergen_classes_for` (plural tolerante + soya como gluten); cero clases sin marcar en las seis bibliotecas.
- **Nombre ↔ constituyentes**: el «Sancocho de pollo» dominicano se componía con res (la receta del diario es de res) y dos
  ensaladas españolas «con atún» llevaban sardinas. Corregido (JSON, tabla curada y script).
- **Raciones y franjas**: meriendas de 450-675 kcal reasignadas a almuerzo/cena; cinco almuerzos puertorriqueños de 1.100-1.250 kcal
  y la bandeja paisa (1.265) reducidos a raciones reales; almuerzos < 260 kcal y cenas < 180 kcal completados; tapas españolas
  (espárragos, champiñones, boquerones) pasan a merienda.
- **Sodio**: recortes en cecina (3.179 mg → ~1.500), chicharrón, jamón de cocinar, chuletas ahumadas y embutidos de la fabada.
- **Bajas**: percebes (ES), ensalada de jícama duplicada (MX) y seis de US (duplicado exacto de yogur, huevos rellenos, queso en hebras,
  tater tots con salchichas, pepperoni con pan, salchichas con frijoles horneados); «Tacos en hoja de lechuga» → «Rollitos de lechuga».
- **Aceptado con razón** (queda en el registro): curados españoles y cultivares caribeños en PR/CO (identidad, no riesgo), snacks de
  < 60 kcal, platos ligeros reales que el generador escala a los macros del día, dos «tacos» en US (Tex-Mex cotidiano, 2 de 81).

## 6. Knobs

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_CULTURAL_PROFILES` | `True` | apagado ⇒ cocina = mercado (legado byte-idéntico) |
| `VITE_CULTURAL_PROFILES` | encendido | apagado ⇒ el paso del wizard no se monta |
| `MEALFIT_DISH_REGISTRY_PROMPT` (F6) | `True` | candidatos del registry en el prompt, por biblioteca de la cocina del día |

## 7. Tests

`test_p1_arq25_f7_culture.py` (motor, gate I16, superficies), `test_p1_arq25_f7_culture_front.py` (contrato frontend),
`test_p1_arq25_f7_culture_libraries.py` (barra de cobertura), `test_p1_arq25_f7_culture_benchmark.py` (benchmark y gate);
frontend `QCulture.p1_arq25_f7.test.jsx`.

## 8b. Subfase G — despensa duradera y congelador sincronizados (2026-09-05)

Pregunta del dueño: ¿bastan las bibliotecas para «una sola compra grande para el mes», con gente que come lo mismo
en distintas preparaciones (arroz + res + ensalada, arroz + atún con huevo + ensalada…)? No bastaban: había platos,
pero pocos de **misma despensa, distinta preparación** que aguanten 3-4 semanas, y el congelador declarado en el
formulario apenas se usaba.

- **Durabilidad SSOT** (`backend/pantry_durability.py`): `master_ingredients.shelf_life_days` es un relleno (lechuga,
  fresas, carne y repollo valen 14 por igual), así que la durabilidad es una tabla de reglas por nombre en cuatro clases:
  `pantry` (seco/enlatado/curado, 90-365 d), `cold` (huevo, quesos curados, raíces, repollo, cítricos, 21-90 d),
  `freezable` (proteína fresca: 3 d, 90 congelada), `fresh` (hojas, hierbas, frutas blandas, tomate, aguacate: ≤ 7 d).
  Tokens cortos casan por palabra exacta («sal» ≠ «salmón»), largos por prefijo («yogur» → «yogurt»).
- **Ventana de congelación** (`freeze_window_days`): sin congelador 0 · limitado 14 (semana de frescos + semana de
  congelados) · completo el ciclo. Antes «limitado» valía 7 y nadie lo consumía.
- **Registry** (schema/compiler 3): cada constituyente lleva `durability`/`days_fresh`; cada plato `logistics.days_fresh_min`,
  `days_with_freezer_min` y `pantry_only` (≥ 21 días sin congelar). `template_candidates(..., need_days, allow_frozen)`
  filtra por día del ciclo; el blueprint y `registry_prompt_lines` pasan la exigencia de cada día bajo compra única
  (`single_trip_requirements`): días 1-7 libres, después solo lo que aguanta (o lo congelable dentro de la ventana).
- **Validador** (`fresh_beyond_horizon_issues`): `fresh_beyond_horizon` para frescos que no llegan y
  `protein_beyond_freeze_window` para proteína fresca fuera de la ventana del congelador del usuario.
- **Prompt**: por modo de congelador (sin / limitado «congelada del día 8 al 14» / completo) más la consigna «misma despensa,
  distinta PREPARACIÓN».
- **Política**: sin congelador ni reposición en un ciclo > 7 días ya **no se acorta el ciclo a 7** (`cycle_shortened_no_freezer_no_topup`
  queda solo para planes antiguos): se respeta la compra única y se declara `pantry_proteins_after_first_week`.
- **PDF / lista**: el rótulo de perecederos dice lo que toca según el congelador (sin congelador: consume primero;
  limitado: congela lo de la segunda semana; completo: congela el día de la compra), en 5 idiomas.
- **Bibliotecas**: dos tandas de despensa duradera (+155 platos: DO 123 · ES 104 · MX 109 · CO 118 · PR 110 · US 103). La segunda
  tanda nació de MEDIR la cobertura real al día 30 sin congelador (MX tenía 1 desayuno, 2 cenas y 0 meriendas que aguantaran):
  se calibraron los días del refrigerado duradero (cebolla/ajo 60; papa, auyama, repollo, zanahoria y quesos curados 45; cítricos,
  tortillas, batata, ñame 30) y se añadieron platos para las franjas delgadas. Listón del test: al día 30 sin congelador,
  desayuno ≥ 7 · almuerzo ≥ 9 · cena ≥ 8 · merienda ≥ 6 candidatos en cada biblioteca. Test: `test_p1_arq25_f7_culture_pantry.py`.

## 8c. Subfase H — el pool de mercado también obedece a la cocina (2026-09-05, prueba real A)

La primera generación real «mercado US + cocina dominicana 0,7» tenía la política y el blueprint correctos (5 días DO /
2 US, candidatos del registry de las dos bibliotecas) y aun así los días dominicanos salieron «Pollo BBQ sobre frijoles
horneados», «Bagel de pollo con ranch», «calabacín al kétchup». La causa no era la capa cultural sino el **pool de mercado
del sembrador** (`COUNTRY_POOLS["US"]`): 10 carbos (bagels, frijoles horneados…, sin arroz, plátano, yuca ni aceite de
oliva) y condimentos contando como «vegetales/grasas»; ese pool es la ASIGNACIÓN OBLIGATORIA del día, y contra eso el
bloque de inspiración no puede.

- `constants.UNIVERSAL_MARKET_STAPLES` (lo que cualquier supermercado de los seis mercados vende) se suma a los pools
  beta; `MARKET_POOL_CONDIMENTS_EXCLUDED` saca ranch/barbacoa/kétchup/mostaza/sazonador de las «grasas».
- Con cocina ≠ mercado, `_market_pool_with_extras` **interseca** el pool con los constituyentes del registry de esa cocina
  (≥ 5 por categoría; si no, el pool entero): US + DO ⇒ arroz, plátano, yuca, batata, habichuelas, avena…, sin bagels.
- Opt-in por kwargs (`market_extras=True, culture_country=`) en el sembrador, el swap y el camino degradado; la firma sin
  kwargs sigue byte-idéntica (contrato de F2). Knob `MEALFIT_MARKET_POOL_UNIVERSAL`.
- Test `test_p1_arq25_f7_culture_market_pool.py`. Lección: *el blueprint puede estar perfecto y el plato salir de otra
  cocina si el sembrador de ingredientes no sabe de cultura — la prueba real con el LLM fue la que lo mostró.*

## 8. Lo que NO hace (a propósito)

- No infiere la cocina del origen, del idioma ni de la zona horaria (solo SUGIERE la del país de compra, visible).
- No traduce nombres de alimentos ni mezcla catálogos: el mercado sigue mandando en precios y despensa.
- No sustituye la revisión humana: el benchmark marca, la persona firma.


## 8d. Subfase I — el sembrador de bases conoce la cocina del día (`P1-CULTURE-STAPLE-SEED` · 2026-09-05)

Prueba real A v3 (plan `f2f7a674`, mercado US, cocina dominicana 70 % + estadounidense 30 %): política, blueprint y pool
correctos (F7-H ya traía arroz, plátano, yuca y habichuelas) y aun así el sorteo eligió *Pasta integral / Lentejas /
Garbanzos* para los 3 días; el día dominicano salió en «canastas de pasta integral». Causa: la rotación anti-repetición
penaliza lo que el usuario acaba de comer (3 planes con arroz y plátano en una hora) y el sembrador no sabía qué cocina
tocaba cada día. La autocrítica lo puntuó Cultural 6/10 y nada lo bloqueó (la fidelidad mide familia y anclas, no cocina).

Regla (`ai_helpers._culture_staple_seed`, tras `_rotate_pairs` y antes de publicar `carb_params`/`carb_pairs`): para cada
día del chunk, `profile_for_day(pesos, offset + i)` ⇒ perfil ⇒ `staples` del perfil casados contra el pool del día
(`cultural_profiles.staple_bases_for_day`: por palabra, plural tolerado, sin acentos; «papa» no casa «Papaya»). Si ninguna
de las dos bases del día es básico de esa cocina, la **segunda** se sustituye por el básico disponible menos usado
(frecuencia fatigada), alternando entre los dos menos usados para los días de la misma cocina (la lista crece ≤ 2 por
cocina). Un básico vetado por sobreuso cuenta como «ya tiene» pero jamás se inyecta. Solo actúa con mezcla o con cocina ≠
mercado: el dominicano en el mercado DO sigue byte-idéntico. Knob `MEALFIT_CULTURE_STAPLE_SEED` (True); fail-open.
Las proteínas no se tocan: las manda la rebanada del blueprint (F3). Test `tests/test_p1_culture_staple_seed.py`.
