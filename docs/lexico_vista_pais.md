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
| Nombre, descripción, ingredientes y pasos del plato (pantalla, Recetas y su PDF) | `displayMeal.mealDisplay` → `lexicoDelPais.textoParaLeer` | Sustitución + la glosa del 649 en la misma pasada; los pasos, encadenados |
| Chips de platos del listado del Historial | `History*Panel.normalizePlan` → `mealDisplayName` | Lo mismo que el nombre del plato |
| Nombre de la lista (PDF de la compra) | `shoppingHelpers.glossShoppingItemName` → `nombreDeListaParaLeer` | «Habichuelas negras» → «Frijoles negros» (sin la glosa `gloss_es`) |
| Envase de la lista | `shoppingHelpers.glossShoppingQty` → `envaseParaLeer` | «1 funda (1 Lb)» → «1 bolsa (1 Lb)» (nunca dentro del paréntesis) |

Reglas (datos en `data/lexico_vista_pais.json`, espejo `frontend/src/data/lexicoVistaPais.json`):

- **Con cambio de género (habichuelas→frijoles), solo si la concordancia se sabe hacer.** Se concuerdan el
  determinante de delante («las»→«los», «de la»→«del», «todas las»→«todos los»), hasta 4 adjetivos de detrás
  («rojas cocidas»→«rojos cocidos») y los coordinados con «y/o» o coma («cocidas y escurridas»→«cocidos y
  escurridos», «rojas, cocidas,»→«rojos, cocidos,»). Luego se lee el **resto**: hasta que el texto vuelve a nombrar
  la palabra (desde ahí un pronombre es de esa mención) y, en los pasos de la receta, **siguiendo por los pasos
  siguientes** («Escurre las habichuelas.» + «Májalas…»). Si el resto vuelve sobre ella en femenino, **no se
  sustituye: se glosa** («habichuelas (frijoles)»). Cuenta como volver sobre ella:
  - pronombre pegado al verbo, con tilde o sin ella: «májalas», «hasta cubrirlas», «mezclándolas»;
  - pronombre delante del verbo tras «no/se/te/me/nos/os»: «no las revuelvas»;
  - atributo tras un copulativo, también infinitivo y coordinado: «hasta que estén blandas», «quedar suaves y
    cremosas»;
  - un adjetivo o participio femenino que no es de un nombre vecino: «, previamente remojadas», «(425 g),
    escurridas», «cocina tapadas», «, ya cocidas», «de trigo rellena». Es de un nombre si va justo detrás de él,
    saltando adverbios, y el nombre no acaba en -o/-os («zanahorias ralladas», «cebolla muy fina»), o coordinado con
    otro que lo era («uvas frescas y jugosas»); tras un número o un determinante es un nombre («2 cucharadas»).
  - con la palabra en singular, también el plural (puede ser parte de un plural coordinado: «ralla la habichuela,
    pica la cebolla y mézclalas»), salvo si es complemento de otro nombre («tortitas de habichuela apiladas»).
  Tras «adjetivo y», un femenino que no se sabe concordar y no es un alimento de la lista `sustantivos`
  («negras y blanditas») también se glosa; «negras y espinacas» se sustituye.
- **«funda» solo como envase de la lista**: en un paso es el verbo («para que el queso funda»).
- **El plátano no se encadena**: en ES/MX «guineo»→«plátano» y el «plátano verde/maduro» dominicano →
  «plátano macho verde/maduro»; un «plátano» suelto no se toca (en España el modelo lo usa para la banana).
- **Coherente con el 649**: donde el 649 glosa el mismo nombre, dice lo mismo (test). Lo que cambia de
  género y el léxico no cubre (batata→boniato, chinola→maracuyá, tayota→chayote) se sigue glosando.
- **RD no cambia nada.** Puerto Rico solo cambia lechosa, auyama y ají morrón (guineo, habichuelas, funda y
  queso blanco son palabras de allí).

## Límites conocidos

La concordancia es una **heurística**: no hay analizador sintáctico. Lo que sabemos de ella, medido (ronda 1 del
revisor, 29-sep):

- **Medido.** Los 6 textos reales de G24 con «habichuelas» (plan de RD leído en MX/CO/US): 4 se sustituyen y 2 se
  glosan, todos gramaticales. De las 5334 lecturas cruzadas de G24 (6 planes × 6 países), frente a la ronda 1
  solo cambian esas 2 frases en los 3 países (6). Prueba de estrés sin IA: 134 textos reales de G24 con un femenino de alimento («lenteja»,
  «papa», «arepa», «tortilla»…) pasado a «habichuela(s)» y leído en MX. Se leyeron las 30 diferencias con la ronda
  1: 7 frases agramaticales pasan a glosarse, 20 glosas pasan a sustituirse bien («1 frijol mediano», «frijoles
  tibios rellenos»), 1 glosa cambia de sitio y 2 glosas sobran (el femenino era de otro nombre: «…acompáñala con
  agua», «ensalada de repollo, frijol y pepino aliñada»). Se leyeron también las 106 frases sustituidas que
  resultan: la única agramatical es la catáfora de abajo («ten listas 1 frijol integral, ¾ cda de aceite…»).
- **Se escapa**: la catáfora (lo que concuerda ANTES del nombre: «Cuando estén blandas, escurre las habichuelas»,
  «ten listas 1 habichuela…»).
- **Se escapa**: el pronombre delante del verbo sin «no/se/te/me/nos/os» delante («Toma las habichuelas y las
  machacas» → «Toma los frijoles y las machacas»). «las» artículo y «las» pronombre no se distinguen sin un
  diccionario de verbos. En la receta el modelo usa el imperativo con el pronombre pegado («machácalas»), que sí se
  ve.
- **Se escapa**: un adjetivo femenino suelto que no está en `adjetivos` ni acaba en -ada(s)/-ida(s)/-osa(s)
  («Sirve las habichuelas, tiernitas» → «Sirve los frijoles, tiernitas»). Tras un copulativo sí se ve («quedan
  tiernitas»).
- **Glosa de más** (no rompe la frase, solo no sustituye): un femenino que es de otro nombre y no se le puede
  atribuir: tras coma o «y» («tortillas calientes, dobladas»), tras un nombre en -o («ensalada de repollo, frijol y
  pepino aliñada») o un pronombre de otro referente («…una comida completa; acompáñala con agua»).

Donde se escapa, el texto queda como en la ronda 1 (sustituido con la concordancia de delante y de detrás). Si
aparece en producción, el arreglo es DATA en `concordancia` (una palabra en `sustantivos`, `adjetivos`,
`copulas`…) más un caso en `casos`, que corren las dos implementaciones.

## Display-only de verdad

Nada vuelve al plan: `mealDisplay` y los helpers de la lista son funciones puras que devuelven otra cadena;
las escrituras (`/swap-meal/persist`, `/regenerate-day`, `/restore-local`, «Me lo comí») viajan desde el
estado CRUDO de `AssessmentContext` o por referencia (`plan_id`, `day_index`, `meal_index`). El contrato
del frontend (`lexicoDelPais.p1_plan_lote_853.test.js`) exige que ningún módulo que escribe importe la capa.

El país del léxico es el del **lector** (`getPaisDelUsuario`, que `AssessmentContext` fija solo con
`COUNTRY_SYSTEM_UI`), el mismo para el plato, el envase y el nombre de la lista: con el sistema de países apagado no
se localiza nada. La glosa del 649 en la lista (`gloss_es`) sigue tomando `formData.country`, como antes.

## Knobs

- `MEALFIT_COUNTRY_DISPLAY_LEXICON` (True): la referencia del backend (`lexico_vista_pais.py`, replay y
  cualquier superficie del backend que pinte texto del plan). Apagado ⇒ solo la glosa del 649.
- `VITE_COUNTRY_DISPLAY_LEXICON` (encendido salvo `0`/`false`/`off`): la app. Apagado ⇒ la conducta del 649.

## Lo que queda fuera

- El coach (chat y avisos) escribe con IA a partir del plan: localizar lo que lee le haría usar «plátano»
  en una herramienta y la Nevera no lo resolvería. Pide su propio lote.
- La Nevera sigue mostrando el identificador con la glosa del 649 («Guineo (plátano)»): es donde el usuario
  gestiona el alimento del catálogo. Las unidades del diario (`glossUnitWord`) también siguen con la glosa.
- EE. UU. no traduce «guineo» (decisión del dueño: «banana», «plátano» o dejarlo).
