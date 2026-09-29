# Léxico de vista por país (lote 853)

[P1-PLAN-LOTE-853 · 2026-09-29] Validación beta G24 (6 planes reales, código P1-PLAN-LOTE-815).

## El problema

«guineo», «lechosa», «auyama», «ají morrón», «queso blanco», «habichuelas» y «funda» salían en nombres,
descripciones, pasos y lista de España, México, Colombia y EE. UU. No viene del prompt: son **nombres del
catálogo**, el identificador con que resuelven la Nevera, el guard de coherencia y el backstop de alergias,
y el modelo está obligado a copiarlos. El plan no se puede traducir (rompería las tres, dos en silencio).

## Qué se hace

Una capa de **vista**, por país de mercado, que solo actúa al pintar para un lector en español:

| Superficie | Dónde (frontend) | Qué cambia |
|---|---|---|
| Nombre, descripción, ingredientes y pasos del plato (pantalla, Recetas y su PDF) | `displayMeal.mealDisplay` → `lexicoDelPais.textoParaLeer` | Sustitución + la glosa del 649 en la misma pasada |
| Nombre de la lista (PDF de la compra) | `shoppingHelpers.glossShoppingItemName` → `nombreDeListaParaLeer` | «Habichuelas negras» → «Frijoles negros» (sin la glosa `gloss_es`) |
| Envase de la lista | `shoppingHelpers.glossShoppingQty` → `envaseParaLeer` | «1 funda (1 Lb)» → «1 bolsa (1 Lb)» (nunca dentro del paréntesis) |

Reglas (datos en `data/lexico_vista_pais.json`, espejo `frontend/src/data/lexicoVistaPais.json`):

- **Solo si la frase sigue siendo gramatical.** Con cambio de género (habichuelas→frijoles) se concuerdan el
  determinante de delante («las»→«los», «de la»→«del», «todas las»→«todos los») y hasta 4 adjetivos de detrás
  («rojas cocidas»→«rojos cocidos»). Si la cláusula vuelve sobre la palabra en femenino («májalas», «hasta que
  estén blandas») o un vecino no sabe concordar, **no se sustituye: se glosa** («habichuelas (frijoles)»).
- **«funda» solo como envase de la lista**: en un paso es el verbo («para que el queso funda»).
- **El plátano no se encadena**: en ES/MX «guineo»→«plátano» y el «plátano verde/maduro» dominicano →
  «plátano macho verde/maduro»; un «plátano» suelto no se toca (en España el modelo lo usa para la banana).
- **Coherente con el 649**: donde el 649 glosa el mismo nombre, dice lo mismo (test). Lo que cambia de
  género y el léxico no cubre (batata→boniato, chinola→maracuyá, tayota→chayote) se sigue glosando.
- **RD no cambia nada.** Puerto Rico solo cambia lechosa, auyama y ají morrón (guineo, habichuelas, funda y
  queso blanco son palabras de allí).

## Display-only de verdad

Nada vuelve al plan: `mealDisplay` y los helpers de la lista son funciones puras que devuelven otra cadena;
las escrituras (`/swap-meal/persist`, `/regenerate-day`, `/restore-local`, «Me lo comí») viajan desde el
estado CRUDO de `AssessmentContext` o por referencia (`plan_id`, `day_index`, `meal_index`). El contrato
del frontend (`lexicoDelPais.p1_plan_lote_853.test.js`) exige que ningún módulo que escribe importe la capa.

## Knobs

- `MEALFIT_COUNTRY_DISPLAY_LEXICON` (True): la referencia del backend (`lexico_vista_pais.py`, replay y
  cualquier superficie del backend que pinte texto del plan). Apagado ⇒ solo la glosa del 649.
- `VITE_COUNTRY_DISPLAY_LEXICON` (encendido salvo `0`/`false`/`off`): la app. Apagado ⇒ la conducta del 649.

## Lo que queda fuera

- El coach (chat y avisos) escribe con IA a partir del plan: localizar lo que lee le haría usar «plátano»
  en una herramienta y la Nevera no lo resolvería. Pide su propio lote.
- La Nevera sigue mostrando el identificador con la glosa del 649 («Guineo (plátano)»): es donde el usuario
  gestiona el alimento del catálogo.
- EE. UU. no traduce «guineo» (decisión del dueño: «banana», «plátano» o dejarlo).
