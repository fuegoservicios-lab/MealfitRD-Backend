# Fase 4 — Formulario progresivo y UX de explicación (capa V2.3)

[P1-ARQ25-F4-FORM · 2026-09-03] Test ancla: [`tests/test_p1_arq25_f4_form.py`](../tests/test_p1_arq25_f4_form.py).
Fases anteriores: [`plan_policy_f2.md`](plan_policy_f2.md) (política), [`plan_policy_f3.md`](plan_policy_f3.md)
(blueprint y superficies). Roadmap: `docs/superpowers/plans/2026-08-29-bioboros-v22-v24-roadmap-maestro.md` §6.4, §6.7, Fase 4.

## Qué entrega la fase

| Entregable (roadmap) | Dónde vive | Cómo se verifica |
|---|---|---|
| Preguntas §6.7; solo dos obligatorias | Wizard: `QMealOrganization` (obligatoria, `mealOrganization`), ciclo principal ya obligatorio (`groceryDuration`), `QShoppingHabits` (frescos condicional a 15/30 días, congelador, tandas — opcional, sin defaults sembrados) | test parser + `test_p0_form_6_required_fields_sync` (`mealOrganization` frontend-only) |
| «Mis básicos» como editor de anclas | `QStapleFoods`: por básico, «Ajustar» → franja(s), frecuencia semanal (2-3 / 4-5 / todos los días) y misma/variada preparación → `stapleAnchors[]`; los nombres siguen en `stapleFoods` (SSOT del motor y de la Nevera) | `policy_from_form` (adapter) + tests de acotado |
| Pantalla «solicitaste / aplicamos / por qué» | `components/dashboard/PlanPolicyPanel.jsx` sobre `plan_data._plan_policy` (requested / effective / relaxations) y `_fidelity_report.mode` | copy por `reason_code` en `config/planPolicy.js` (paridad con `_REASON_COPY`) |
| Estados de generación | F1 ya expone `availability` en el snapshot del run y el Dashboard su honestidad (P1-DASH-GENERATING-HONESTY); la Fase 4 añade la relajación `waiting_user` (presupuesto bajo el piso) como aviso con CTA en el panel | test de la relajación (F2) + panel |
| Estados de proyección (`plan_jobs`) | **Fase 5** (fuera de alcance aquí, a propósito) | — |
| i18n 5 locales, accesibilidad, recuperación del wizard | claves en los 4 catálogos (`npm run i18n:check`), `radiogroup`/`fieldset`/`aria-expanded`, `mealfit_form` persiste los campos nuevos | `Assessment.plan_policy_form.test.jsx` |
| E2E guest/restore | `frontend/e2e/wizard_policy.spec.js` | Playwright |
| Contrato versionado | `POLICY_SCHEMA_VERSION` en ambos lados; enums y reason codes del frontend leídos por el test del backend | `test_el_frontend_repite_los_enums_del_compilador` |
| Línea base de conversión | `POST /api/plans/telemetry/wizard` → `pipeline_metrics.node='wizard_funnel'` (invitados incluidos, `sid` hasheado, lista blanca de claves) | tests del sink |

## Knobs

| Knob | Dónde | Default | Efecto |
|---|---|---|---|
| `VITE_PLAN_POLICY_FORM` | frontend (build) | on | `'0'`/`'false'`/`'off'` esconde las preguntas nuevas, el editor de anclas y el panel; el adapter del backend sigue tolerando ambos formularios |
| `MEALFIT_PLAN_POLICY_MODE` | backend | `shadow` en prod | El panel solo dice «tu plan sigue tu política» cuando `_fidelity_report.mode == 'enforce'`; en `shadow` dice «lo que pediste» y lo declara |

## Contrato del formulario (v2)

```json
{
  "mealOrganization": "routine | balanced | explore",
  "stapleFoods": ["Huevo", "Arroz"],
  "stapleAnchors": [{ "name": "Huevo", "slots": ["breakfast"], "min_per_7d": 5, "max_per_7d": 7, "preparation_mode": "same_preparation" }],
  "groceryDuration": "weekly | biweekly | monthly",
  "freshTopup": "yes | no",
  "freezerMode": "none | limited | full",
  "batchCooking": "never | sometimes | often"
}
```

Ausentes ⇒ los defaults de la Fase 2 (`balanced`, `limited`, reposición semanal si el ciclo > 7,
tandas según `cookingTime`) y `source.form_version = "v1"`; presentes ⇒ `"v2"`. Un cliente viejo
produce la política de ayer: ningún campo nuevo es obligatorio para el backend.

## Gate de la fase y cómo medirlo

1. CI verde en ambos repos; `npm run i18n:check` verde; E2E verde.
2. Conversión del wizard, medida (no supuesta):

```sql
SELECT date_trunc('day', created_at) AS d,
       COUNT(DISTINCT session_id) FILTER (WHERE metadata->>'event' = 'wizard_start')  AS empiezan,
       COUNT(DISTINCT session_id) FILTER (WHERE metadata->>'event' = 'wizard_submit') AS terminan
FROM pipeline_metrics WHERE node = 'wizard_funnel' AND created_at > NOW() - INTERVAL '14 days'
GROUP BY 1 ORDER BY 1;
```

La línea base es la primera semana con `policy_form=false`… que no existe: el formulario nuevo y
la telemetría nacen juntos. Se compara, por tanto, el embudo por paso (`step_view` → `step_done` por
`step_id`): si `mealOrganization` o «Mis básicos» pierden más usuarios que la mediana de los pasos
vecinos, ahí está la caída.

> **Medido el 2026-09-11 (`P1-PLAN-LOTE-10` · B9).** `pipeline_metrics.wizard_funnel` tiene **121 filas de 5 sesiones
> en 2 días con datos** (09-04 y 09-09: el dueño probando). No hay línea base posible hasta que haya usuarios; la
> condición «≥ 2 semanas» se cuenta desde el lanzamiento, no desde hoy. Y el embudo por paso tal como está escrito
> arriba era **ciego**: el frontend (`InteractiveAssessmentFlow.jsx`, `trackWizard`) emitía `wizard_start`,
> `step_view`, `wizard_submit` y `wizard_restore`, **nunca `step_done`** (0 de 121), aunque el backend ya lo aceptaba.
> Desde `P1-PLAN-LOTE-10` (frontend `840b236`) lo emite al AVANZAR de paso (volver atrás no termina un paso). Para
> las filas anteriores —y como control del nuevo evento— el paso N se da por terminado cuando la MISMA sesión ve un
> paso con índice mayor:
>
> ```sql
> WITH v AS (SELECT session_id, (metadata->>'index')::int AS idx, metadata->>'step_id' AS paso
>            FROM pipeline_metrics WHERE node = 'wizard_funnel' AND metadata->>'event' = 'step_view')
> SELECT a.paso, a.idx, COUNT(DISTINCT a.session_id) AS ven,
>        COUNT(DISTINCT a.session_id) FILTER (WHERE EXISTS (
>            SELECT 1 FROM v b WHERE b.session_id = a.session_id AND b.idx > a.idx)) AS siguen
> FROM v a GROUP BY 1, 2 ORDER BY 2;
> ```
>
> Con 5 sesiones: 5 ven `appMode`, 4 `planSource`, 3 llegan a `gender` y las 3 recorren hasta `motivation`; 3 de 5
> envían. Los pasos sin `step_id` salen como `step_<índice>` (8, 12, 13, 18, 19, 21, 22, 24, 25): al añadir el id
> en el frontend, este embudo deja de mezclarlos.

3. Antes de publicarlo a usuarios nuevos, `MEALFIT_PLAN_POLICY_MODE=enforce` global (Fase 3): el
   formulario promete franjas y bandas que el motor solo obedece en `enforce`.

## Fuera de alcance (Fase 5+)

Estados de proyección (`plan_jobs.shopping_projection`) en Dashboard/Nevera; edición de la política
desde Configuración sin volver al wizard; `culture_weights` desde el formulario (Fase 7).
