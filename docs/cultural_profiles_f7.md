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
perfiles. La **firma humana está pendiente** por perfil (el código no la da): el informe lista las plantillas marcadas para que una
persona las mire (3 «tacos» fuera de MX, los cultivares dominicanos en PR/CO, procesados y sodio alto por biblioteca).

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

## 8. Lo que NO hace (a propósito)

- No infiere la cocina del origen, del idioma ni de la zona horaria (solo SUGIERE la del país de compra, visible).
- No traduce nombres de alimentos ni mezcla catálogos: el mercado sigue mandando en precios y despensa.
- No sustituye la revisión humana: el benchmark marca, la persona firma.
