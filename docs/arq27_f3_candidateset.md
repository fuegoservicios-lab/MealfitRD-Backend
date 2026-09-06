# 2.7-F3 — El CandidateSet fijado al run (ARQ27-P1-04)

`[P1-ARQ27-F3-CANDIDATESET · 2026-09-06]`

## El defecto

El blueprint fijaba 3 candidatos del Dish Registry por día y franja. Pero `slice_for_chunk` **no se
los llevaba a la rebanada**, así que `registry_prompt_lines` volvía a consultar `template_candidates`
contra el registro **activo** — y de los 3 recitaba solo 2.

Consecuencia: recompilar el Dish Registry entre dos chunks cambiaba los platos que se le proponían a
un plan **ya empezado**, sin que nada lo declarara. Y como el repo recompila **en `v1`** (la práctica
vigente), el cambio no traía ni siquiera un número de versión distinto que lo delatara.

## Los tres defectos del mismo camino

| # | Qué pasaba | Qué se hizo |
|---|---|---|
| 1 | La rebanada tiraba el CandidateSet | `slice_for_chunk` copia el bloque `registry` recortado a sus días. Al ir dentro entra en `slice_hash` → `input_hash`: un cambio de catálogo se ve como revisión distinta en vez de colarse callado |
| 2 | El orden era el de **inserción en el fichero**, con `break` al llegar a `k` | Se recogen todas las compatibles y se ordenan por `sha256(template_id)` con la consulta como sal. `rotate` (el índice del día) desplaza la lista |
| 3 | Fijar el **ID** no bastaba | Se fija también el **nombre** (`candidate_names`) |

El tercero lo encontró la verificación contra el registry real, no el diseño: el primer intento
pasaba tres de los cuatro criterios y fallaba justo el que da nombre al gap. Si el registro activo
retira una plantilla, su ID deja de resolver a nombre y el conjunto del run **encoge en silencio** —
lo contrario de «fijado».

## Las tres capas de `_pinned_candidate_names`

1. **Los nombres fijados en la rebanada.** Única capa que sobrevive a que retiren la plantilla, y por
   eso va primera.
2. **Resolver los `template_id`** contra el snapshot vivo — para rebanadas fijadas antes de que se
   guardaran los nombres.
3. **Nada** ⇒ el llamador reconsulta. Es el **adaptador** que el gap pedía como rollback: los runs
   fijados antes de este cambio no traen `registry` en su rebanada y su historial no se reescribe.

## `templates_by_id` se cachea por hash de contenido

El criterio dice «resolver por hash de contenido, no solo por nombre v1». Una caché indexada por
nombre de biblioteca serviría plantillas viejas después de recompilar sin cambiar de versión — que es
exactamente lo que el repo hace. La clave es `(biblioteca, snapshot_hash)`.

## `chunk_input_hash` ya no se fía del hash que trae la rebanada

Se prefería el `slice_hash` que la propia rebanada lleva **dentro**. Una rebanada modificada en
tránsito seguía declarando su hash viejo, así que el `input_hash` decía que nada había cambiado: un
dato que se certifica a sí mismo. Ahora se recalcula del contenido. Para una rebanada bien formada
los dos valores coinciden y nada cambia.

## El swap

El criterio pide que el swap declare si usa el snapshot del plan o migra a otro. **La respuesta
medida es que no usa el registry en absoluto**: `api_swap_meal` no menciona `dish_registry`,
`template_candidates`, `registry_prompt_lines` ni `_blueprint_slice`.

Eso es la respuesta, no una omisión — y
[`test_p1_arq27_f3_candidateset.py`](../tests/test_p1_arq27_f3_candidateset.py) la ancla: si alguien
cablea el registry al swap, el test cae y tendrá que declarar cuál usa.

## Dónde vive

- [`backend/horizon.py`](../horizon.py) — `slice_for_chunk`, `registry_prompt_lines`,
  `_pinned_candidate_names`, `_registry_block_for_country`, `chunk_input_hash`.
- [`backend/dish_registry.py`](../dish_registry.py) — `template_candidates` (orden + `rotate`),
  `templates_by_id`.
- Test: [`test_p1_arq27_f3_candidateset.py`](../tests/test_p1_arq27_f3_candidateset.py).

---

# 2.7-F3 — La batería de entrega con banderas efectivas (ARQ27-P1-06)

`[P1-ARQ27-F3-BATERIA · 2026-09-06]`

## El gap, medido

`conftest.py` apaga gates a propósito para que los tests del camino OFF no dependan del entorno. El
efecto lateral no estaba medido: **seis knobs corren en un estado que producción no usa.** Leído del
`.env` del VPS el 2026-09-06, no deducido:

| knob | producción | la suite |
|---|---|---|
| `MEALFIT_COUNTRY_SYSTEM` | `true` | `False` (default; nadie lo toca) |
| `MEALFIT_VERIFIED_INGREDIENTS_ONLY` | `true` | `false` (conftest) |
| `MEALFIT_UPDATE_DISHES_STRICT_ALL_REASONS` | `true` | `false` (conftest) |
| `MEALFIT_SODIUM_EXCESS_GATE` | ausente ⇒ default `True` | `false` (conftest) |
| `MEALFIT_RECIPE_CONTRACT_GATE` | ausente ⇒ default `True` | `false` (conftest) |
| `MEALFIT_MICRO_CLOSER_PERDAY` | ausente ⇒ default `True` | `false` (conftest) |

El más caro es el primero: con `COUNTRY_SYSTEM` apagado los seis países colapsan a DO, así que
**ningún test de la suite ha visto nunca el catálogo de ES, US, MX, PR ni CO**.

Los otros tres enseñan una lección aparte: **un knob que no aparece en un `.env` no está apagado —
está en su default.** Los tres valen `True` en el código.

## Qué hay ahora

- [`backend/prod_profile.py`](../prod_profile.py) — el perfil, con procedencia y fecha. Excluye a
  sabiendas los secretos y las **listas de user_id reales** del canary (con su motivo escrito).
- [`backend/scripts/delivery_battery.py`](../scripts/delivery_battery.py) — 16 cohortes (6 cocinas ×
  dietas × alergias × condición), tasas **con denominador** por dimensión.

Medición del 2026-09-06: entrega, seguridad, nutrición, variedad y «sin relajación» al **100 %**;
completitud del catálogo **DO 140/144** (4 plantillas en `partial`: Chillo al horno, Batida de
zapote, Mangú con salami de pavo, Frutas picadas), las otras cinco cocinas al 100 %.

## Dos métricas nacieron mal y se corrigieron antes de publicar nada

Contaban «¿mordió la etapa del filtro?», así que castigaban a `renal_do` y `hta_do` por no descartar
ninguna plantilla — cuando eso significa que toda la biblioteca dominicana tiene fósforo, potasio y
sodio medidos, o sea el mejor resultado posible. **Una métrica que llama fallo al mejor caso posible
habla de la métrica, no del producto.** Ahora se miden sobre los supervivientes.

Y la cohorte «imposible» no lo era: hay 6 desayunos veganos sin gluten, soja, frutos secos ni
legumbres. Exigirle cero habría inventado un defecto. Se le exige algo más útil: que sus
supervivientes sean un **subconjunto** de los de la cohorte laxa, que es lo que detecta una
relajación silenciosa.

## Lo que la batería NO hace, dicho y no escondido

- **No mide latencia ni coste por plan entregado.** Es el canary del gap y necesita generaciones
  reales contra el proveedor. Fingirlo en una batería determinista sería peor que no tenerlo.
  **`ARQ27-P1-06` queda parcialmente abierto por esto.**
- **No ejercita el swap ni el último chunk end-to-end** (necesitan DB y LLM).

Test: [`test_p1_arq27_f3_bateria.py`](../tests/test_p1_arq27_f3_bateria.py).
