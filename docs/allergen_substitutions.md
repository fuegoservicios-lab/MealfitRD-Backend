# Sustitución determinista de alérgenos IgE — [P0-ALLERGEN-SUBS · 2026-06-14]

Cierra el gap P0 del audit clínico (2026-06-14): _"Alergias IgE no están en el registry — deberían
tener swap determinista como sodio/azúcar, no depender de que la LLM las detecte."_

## Qué cambia

Antes, una alergia declarada (`form_data['allergies']`) viajaba al prompt + la cazaba el backstop
determinista `_scan_allergen_violations` (en `review_plan_node`), que la trataba como **rechazo
crítico** → `_apply_critical_review_guardrails` → **fallback matemático** (se pierde el plan rico del
LLM). Funcionaba pero era **romo**.

Ahora la capa clínica determinista (`_apply_deterministic_clinical_layer`, Guard 2.5) hace el swap
**quirúrgico** ANTES del review: reemplaza el ingrediente ofensor por una alternativa segura que
**resuelve al catálogo** `master_ingredients`, conservando el plan. Reusa el motor compartido
`_apply_substitutions_core` (mismo que las sustituciones por condición) → preserva el prefijo de
cantidad + recalcula macros por delta (fail-safe). El backstop `_scan_allergen_violations` queda como
**red de seguridad post-swap**: cualquier residual sigue escalando a crítico → fallback. **Cero
regresión de seguridad.**

## Tabla canónica (SSOT: `condition_rules.py`)

| Alérgeno declarado | Categoría | Tokens en ingrediente (estrechos, accent-free) | Reemplazo (resuelve en catálogo) | preserve_qty |
|---|---|---|---|---|
| pescado, atún, salmón, bacalao, tilapia… | `allergen:fish` | pescado, bacalao, atun, salmon, tilapia, filete de pescado… | **Pechuga de pollo** | sí |
| mariscos, camarón, langosta… | `allergen:shellfish` | camaron, langosta, cangrejo, marisco, pulpo, calamar… | **Pechuga de pollo** | sí |
| soya, tofu, edamame | `allergen:soy` | tofu, edamame, proteina de soya | **Pechuga de pollo** | sí |
| soya (condimento) | `allergen:soy` | salsa de soya, teriyaki | **Limón con especias** | no |
| gluten, trigo, celiaquía | `allergen:gluten` | harina de trigo | **Harina de maíz precocida** | sí |
| gluten | `allergen:gluten` | pan de agua, pan integral, pan de trigo, tostada… | **Casabe** | sí |
| gluten | `allergen:gluten` | pasta integral, espagueti, macarrón, fideo, lasaña… | **Arroz blanco** | sí |
| gluten | `allergen:gluten` | galleta(s) de soda/trigo | **Galletas de arroz** | sí |
| gluten | `allergen:gluten` | avena, hojuelas/harina/salvado de avena | **Quinoa** | sí |
| gluten | `allergen:gluten` | cebada, centeno, cuscús, bulgur, tortilla de trigo… | **Arroz blanco** | sí |

> **Avena (live-fix 2026-06-14):** la avena es naturalmente sin gluten, pero el revisor médico la
> rechaza por contaminación cruzada (estándar conservador). Sin el swap, un plan con avena para un
> alérgico a gluten caía a fallback. Se sustituye por **Quinoa** (GF nativo, en catálogo, alto en
> proteína). Hallazgo de la prueba en vivo en producción.

Vetos (gluten): `sin gluten`, `libre de gluten`, `de maiz`, `de arroz`, `de yuca`, `casabe`, `pana`…
— evitan swappear un alimento que YA es libre del alérgeno.

## Decisión honesta: lo que NO se sustituye

**Lácteos, huevo, maní y frutos secos NO tienen swap determinista** y siguen por el path
crítico→fallback existente (que los excluye). Razón: el catálogo es-DO **no tiene un target libre del
alérgeno que resuelva** — no hay filas de leche/queso vegetal, ni mantequilla de semillas, ni
sustituto de huevo. Sustituir a un string que no resuelve reintroduciría el "0 silencioso" (pérdida
de proteína), el bug que el audit advirtió.

**Cuándo revisitar:** si se añaden filas de leche vegetal / queso vegano / sustituto de huevo al
catálogo `master_ingredients` (palanca de DATOS), añadir las categorías `allergen:dairy` /
`allergen:egg` a `_ALLERGEN_SUBS_BY_CAT` con esos targets. Mientras tanto, la exclusión vía fallback
es la opción **segura**.

## Disciplina de tokens

Tokens **estrechos + accent-free** (lección del bug `soya`/`pana`): nada de raíces ambiguas. `pan de
agua` (NO `pan` desnudo, que matchea `pana`=fruta de pan); `tofu`/`salsa de soya` (NO `soya` desnudo,
que borraría proteína vegetal legítima). Lo que un token estrecho no atrape (p.ej. `pan` a secas) lo
recoge el backstop `_scan_allergen_violations` (word-boundary regex) → cae a fallback. El motor de
swap usa matching por **substring**, así que la narrowness es la única defensa contra falsos positivos.

## Knob y archivos

- Knob: `MEALFIT_ALLERGEN_SUBSTITUTION` (default `True`). Flip a `False` revierte al comportamiento
  previo (detección→rechazo crítico→fallback). Auto-registrado en `_KNOBS_REGISTRY`.
- SSOT tablas + `collect_allergen_substitutions`: [`condition_rules.py`](../condition_rules.py).
- Motor compartido + guard: `_apply_substitutions_core`, `_apply_allergen_substitutions`,
  Guard 2.5 en `_apply_deterministic_clinical_layer` ([`graph_orchestrator.py`](../graph_orchestrator.py)).
- Backstop de seguridad (sin cambios): `_scan_allergen_violations` (review_plan_node), `ALLERGEN_HARD_GUARD`.
- Tests: [`test_p0_allergen_subs.py`](../tests/test_p0_allergen_subs.py).

## Limitaciones conocidas (consistentes con las sustituciones por condición)

- El **texto de la receta** puede seguir mencionando el ingrediente viejo (el swap solo toca
  `ingredients`/`ingredients_raw`, no reescribe pasos). La lista de compras y los macros sí reflejan
  el reemplazo, así que el usuario **no compra** el alérgeno. Una nota es-DO aclara el cambio.
- `_scan_allergen_violations` escanea solo `ingredients`, no la receta — gap pre-existente, fuera de
  scope de este P0.
