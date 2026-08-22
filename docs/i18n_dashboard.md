# Idioma de la interfaz del dashboard (`P1-I18N-DASHBOARD`)

**Estado**: activo desde 2026-08-15. **Supersede** a `P3-I18N-DEFERRED` («i18n: es-DO permanente», 2026-05-13).

El dashboard se puede leer en 5 idiomas. Este documento es la doc canónica: qué se
traduce y qué no, cómo funciona el motor, y las tres decisiones cuyo *por qué* no se
deduce del código.

---

## 1. El alcance, y por qué es el que es

Se traduce **la interfaz**. No se traduce **el contenido**.

| Superficie | ¿Traducida? | Por qué |
|---|---|---|
| Chrome del dashboard (nav, botones, títulos, Configuración, toasts, validaciones, `aria-label`) | **Sí** | Es lo que hace la app usable por alguien que no lee español. |
| Plan, recetas y nombre del plan | **Sí**, desde [P1-PLAN-DISPLAY-I18N · 2026-08-19] | Capa `_display[locale]` paralela: el LLM traduce para LEER y el motor sigue operando sobre el español canónico. Detalle en [`plan_display_i18n.md`](plan_display_i18n.md). El fallback al español es conducta ESPERADA, no fallo: si la traducción falta, no cuadra por longitud o pierde una cifra o una etiqueta de sección, esa línea se pinta en español (P2-DISPLAY-VALIDADOR-SIN-CIFRAS, P1-DISPLAY-VOCAB-CERRADO). Knob `MEALFIT_PLAN_DISPLAY_I18N`, default `True`. |
| Lista de compras | **Bilingüe, y el gloss es SIEMPRE INGLÉS** | [P2-I18N-DOC-LISTA-BILINGUE-FALSA · 2026-08-22] Esta fila decía «el gloss en el idioma del usuario». No lo es: `glossShoppingItemName` compone `name_en` + el nombre español para CUALQUIER locale distinto de `es-DO`, así que un francés lee «Black beans (Habichuelas rojas)» — inglés, no francés. `name_en` es un campo ESTÁTICO del catálogo, no una traducción por idioma, y sólo existe en el PDF. Cada línea lleva el gloss Y el nombre canónico español entre paréntesis — «30 g dried red beans (Habichuelas rojas)». El paréntesis no es cortesía: es el identificador con el que resuelve el motor, y el validador descarta la línea que lo pierda. |
| Respuestas del coach (chat + notificaciones proactivas) | **Sí** | [P1-COUNTRY-SYSTEM-F2 · T3 · 2026-08-17] La PROSA del coach sigue `locale` — es el pedido en vivo del dueño (Addendum §2), no parte del sistema de países en oscuro. Frontera dura: los nombres de alimentos/platos que el coach cita, y toda tool call, SIGUEN en español canónico SIEMPRE (mismo motivo que la fila de abajo). Ver `prompts.chat_agent.build_language_directive`. |
| Nombres de alimentos y platos (`master_ingredients`, **347 filas**, todas con `name_en`) | **No, jamás** | Son el **SSOT del motor**. `pantry_names_match` (P1-PANTRY-NAME-RESOLUTION), el guard de coherencia recetas↔lista y el backstop clínico de alergias resuelven por esos nombres exactos. Traducir «Pollo» rompe las tres cosas a la vez, y dos de ellas en silencio. |
| Correo del código de acceso (OTP) | **No, y no depende de este repo** | [P3-I18N-OTP-PLANTILLA · 2026-08-21] Lo redacta y envía **Neon Auth** (Better Auth) desde una plantilla de su panel. El frontend solo hace `POST <neonAuthUrl>/email-otp/send-verification-otp` con `{email, type:'sign-in'}` — ese es el cuerpo completo: **no hay campo de idioma que mandar**, así que no es que esté sin cablear, es que la API que llamamos no ofrece el canal. Cambiarlo es editar la plantilla en el panel de Neon, y ahí sería una sola versión para todos salvo que ellos soporten variantes por idioma. Se declara porque llega en CADA login —junto con Google, es la única puerta de entrada— y una superficie que nadie declara es una superficie que nadie revisa. |
| PDF de la lista de compras y de la receta | **Sí** | [P2-I18N-PDF-* · 2026-08-22] Rótulos de sección, cantidades, leyenda, advertencia clínica, marca y nombre del fichero. Los rótulos y la nota clínica se glosan **al imprimir, nunca en el dato**: `display_category` es además clave de agrupación y la nota vive en `plan_data`. Los nombres de alimento siguen la regla de la fila de abajo. |
| Notificaciones push (43 mensajes de 6 crons) | **Sí** | [P1-I18N-PUSH-CRON-ESPANOL · 2026-08-22] Traducidas en el CUELLO DE BOTELLA (`utils_push.send_push_notification`), no en los 35 call sites: un cron nuevo queda cubierto sin wiring. Catálogo SSOT [`push_i18n.py`](../push_i18n.py), fail-open. |
| Help bot e insights | **Sí** | [P1-HELP-BOT-I18N + P1-INSIGHTS-I18N · 2026-08-20] El razonamiento del panel de insights lo genera el LLM bajo `_INSIGHTS_ADDENDUM` (espejo #12); el help bot resuelve por catálogo. |
| Autodetección del idioma en el primer arranque | **Sí** | [P1-AUTO-LOCALE] Se lee del navegador cuando el perfil no trae `locale`. Depende de que la columna admita `NULL`: mientras tuvo `NOT NULL DEFAULT 'es-DO'`, el primer login sembraba un valor y apagaba la autodetección PARA SIEMPRE (P1-I18N-PROFILE-DEFAULT-PISA-INERTE, migración aplicada 2026-08-22). |
| Páginas legales (Privacidad, Términos — 601 cadenas) | **No** | Traducir un contrato genera obligaciones en cada jurisdicción. Es una decisión legal, no de producto. |
| Landing (`bioboros.com`) | **No** | [P3-I18N-DOC-LANDING-NO-ES-ESTATICO · 2026-08-22] Esta fila decía «14 páginas estáticas fuera del build de React». **Es falso**: son 19 rutas en `PAPER_SURFACE_ROUTES` (`utils/paperSurface.js`, el SSOT), componentes React cargados con `lazy()` DENTRO del mismo build de Vite. La exclusión sigue siendo correcta y la razón real es otra: traducirlas exige URLs por idioma y `hreflang`, que es un cambio de arquitectura de rutas y de SEO, no de copy. Importa arreglarlo porque una razón falsa invita a «corregirla» metiendo el landing donde ya está. |

**La frontera, que es lo único que no se mueve**: se traduce lo que el usuario LEE; no
se traduce lo que el motor USA COMO IDENTIFICADOR. Por eso el plato entero puede salir en
francés mientras «Habichuelas rojas» sigue apareciendo, literal, dentro de la línea de
ingredientes — y por eso el prefijo «Mise en place:» viaja en español en el dato aunque la
pantalla lo pinte como «Mise en place» traducido. `pantry_names_match`, el guard de
coherencia recetas↔lista y el backstop clínico de alergias resuelven por esas cadenas
exactas, y dos de las tres fallarían en silencio.

**Lo que este párrafo decía antes**, hasta [P2-I18N-DOC-ALCANCE-MIENTE · 2026-08-21]: que
plan y recetas NO se traducían, y lo llamaba «la consecuencia honesta». Llevaba siendo
falso desde el 2026-08-19. Se anota en vez de borrarse porque el daño no fue teórico: esta
doc es la que leyó la auditoría de alcance del sistema de idiomas, y por creerle dejó fuera
de la primera pasada la superficie i18n más cara del producto. Una doc canónica equivocada
no confunde solo a las personas.

## 2. Los 5 idiomas

`es-DO` (base), `en-US`, `pt-BR`, `fr-FR`, `it-IT`.

Se eligieron 5 y no los 11 de un ChatGPT porque cada idioma multiplica ~1.800
traducciones y ninguna de ellas la revisa un hablante nativo. Cinco idiomas con
calidad defendible es un producto; once con hindi y coreano de terminología
nutricional sin revisar es una demo. **Añadir el sexto es un JSON y cuatro líneas** —
ver §6.

### Las etiquetas: sin país, y el código NO sigue a la etiqueta

`P1-I18N-LABEL-NEUTRAL · 2026-08-15`. El selector muestra **«Español», «English»,
«Português», «Français», «Italiano»** — sin paréntesis de país. La regla:

> El paréntesis existe para **desambiguar**, y no hay nada que desambiguar cuando se
> ofrece **una sola variante** por idioma.

Nació de un caso concreto: la etiqueta original decía «Español (República Dominicana)»
y a un cliente español le comunica *«esto no es para ti»* — lo contrario de lo que hace
falta si el producto se vende fuera de RD. Quitarlo solo del español dejaba a los otros
cuatro con país, lo que se lee como descuido, así que la regla se aplicó a los cinco.

El día que existan **dos** variantes de una lengua, el paréntesis vuelve **a las dos**:
«Español (España)» + «Español (Latinoamérica)», nunca a una sola — una lista donde una
variante lo lleva y su gemela no obliga al usuario a deducir cuál es cuál.
`test_p1_i18n_dashboard.py::test_a4` enforza ambas direcciones.

**⚠️ El CÓDIGO se queda en `es-DO`, y no es inercia.** `Intl` formatea:

| Locale | Ejemplo |
|---|---|
| `es-DO`, `es-419`, `es-MX` | `2,000` · `1,234.5` |
| `es`, `es-ES` | `2000` · `1234,5` |

República Dominicana usa la convención de EE.UU. «Neutralizar» el código a `es` porque
la etiqueta se neutralizó movería los separadores de miles y decimales de **toda la base
actual**, en silencio, porque ningún test de i18n mira cifras. La etiqueta es lo único
que el usuario lee; el código es un identificador interno con consecuencias de formato.
Cambiarlo sería una migración de datos deliberada (reescribir `user_profiles.locale`),
no un renombre. Anclado en `test_a5`.

## 3. El motor

`frontend/src/i18n/`. Propio, ~250 líneas, **cero dependencias**.

No se usó `react-i18next` (~30 kB gz) porque trae backends HTTP, detección de idioma,
namespaces y Suspense —nada de lo cual se usa— y este repo pasó agosto recuperando
bytes del entry (`P1-APEX-ENTRY-DIET`: 33 kB; `P2-LANDING-OLA1-DIET`: 181 iconos).

### La decisión estructural: **la clave es el texto español**

```jsx
t('Apariencia')                  // ✅  el español ES la clave
t('settings.appearance.title')   // ❌  clave simbólica
```

Tres consecuencias, y son el diseño entero:

1. **`es-DO` no tiene catálogo.** Es el fallback. El 100% de la base actual
   (dominicana) descarga **cero bytes** de i18n. Solo quien elige francés pide
   `fr-FR.json`, en su chunk.
2. **Una cadena sin traducir muestra español**, nunca `settings.save` en crudo. Una
   migración incompleta deja pantallas mitad traducidas —coherentes— en vez de rotas.
3. **El precio**: cambiar el copy español huérfana su traducción **en silencio**. Nadie
   ve un error; esa línea vuelve al español en los otros 4 idiomas y solo se detecta
   navegando en francés hasta esa pantalla.

Ese precio **no se paga con disciplina, se paga con `npm run i18n:check`** (§5). Sin
ese script este diseño es una trampa; con él, una red. Si alguien lo borra del
`package.json`, ha desarmado la única defensa del sistema.

### API

| Símbolo | Uso |
|---|---|
| `useT()` | En componentes. Devuelve `t` y **suscribe** al cambio de idioma. |
| `t(es, vars)` | Fuera de componentes (helpers, handlers). Mismo `t`, sin suscripción. |
| `tn(n, one, other, vars)` | Plural vía `Intl.PluralRules` del locale activo — **no** `n === 1`: el francés mete el 0 en singular y el portugués tiene categoría `many`. |
| `t('Plan\|nav')` | Homógrafos. Se pinta lo anterior al `\|`. |
| `formatDate` / `formatNumber` | `Intl` con el locale activo. Reemplazan los `toLocaleDateString('es-DO')` fijos. |

### La trampa: `t()` en ámbito de módulo

```js
const TABS = [{ label: t('Plan') }];        // ❌ congelado en español para siempre
const getTabs = () => [{ label: t('Plan') }]; // ✅
```

Un array de copy evaluado al importar corre **antes** de que el catálogo exista. Y en
es-DO se ve perfecto, así que pasa cualquier revisión visual. `i18n:check` lo detecta.

### Cambiar de idioma REPINTA — no remonta (y por qué se retiró el remontaje)

`P1-I18N-SWAP-SMOOTH · 2026-08-15`. La primera versión envolvía las rutas en un
`LocaleBoundary` con `key={locale}` para forzar un remontaje completo. La intención era
defensiva: había copy calculado fuera de componentes y subárboles memoizados que un
re-render no alcanzaría.

**Se retiró porque el precio se sentía en cada cambio** y el peligro resultó no existir.
El dueño lo describió como que «se siente raro»: estando en Configuración, el diálogo se
volvía a montar, repetía su animación de apertura y el scroll saltaba arriba — justo
mientras mirabas la lista de idiomas.

Lo que se midió antes de quitarlo:

| Riesgo supuesto | Realidad |
|---|---|
| Subárboles memoizados no se enteran | **`React.memo` NO bloquea la propagación de contexto.** Los 3 componentes memoizados usan `useT()`, así que se re-renderizan igual. |
| Módulos que importan `t` sin el hook | Son funciones llamadas en render (`getMacros()`, `textoNeveraBaja()`) o toasts imperativos (`confirmToast`, `renderCoherenceWarnings`), que leen el catálogo **vivo** en el momento de la llamada. |
| `useMemo` con deps vacías capturando copy | **El único hueco real**: 2 casos en `Plan.jsx` (pantalla de carga). Ahora dependen de `locale`. |

Queda lo natural: el texto cambia en el sitio, sin parpadeo ni salto de scroll.

### El clic también tiene que responder al instante

`setLocale` espera al `import()` del catálogo — 100-300 ms la primera vez que se elige
cada idioma. Sin nada más, pulsabas una fila y **no pasaba nada** durante ese rato (ni la
marca se movía), y luego cambiaba todo de golpe: se leía como que el clic no había
registrado. Por eso `Settings.jsx` lleva `pendingLocale`, que mueve la marca de selección
**ya** y se limpia en un `finally` — si la carga falla, la marca vuelve al idioma que de
verdad está activo. Un optimismo que no sabe retroceder es una mentira.

## 4. Persistencia: el idioma sigue al **usuario**

Columna `user_profiles.locale` (migración `p1_i18n_dashboard_locale_2026_08_15.sql`, en
**los dos** directorios por P3-MIGRATIONS-SSOT) + `localStorage` como caché.

- **Por qué en la DB y no solo local** (como el tema): el tema es una preferencia de
  pantalla y se acepta que cada dispositivo tenga la suya. El idioma no — elegir francés
  en el móvil y abrir el portátil en español se lee como un fallo, no como una
  preferencia local.
- **Por qué `localStorage` además**: el perfil viaja por red. El boot síncrono de
  `index.html` fija `<html lang>` **antes del primer paint**; sin eso hay parpadeo y,
  peor, VoiceOver anuncia la pantalla en el idioma equivocado durante el arranque.
- **Por qué `PATCH /api/profile` y no un endpoint propio**: `locale` es literalmente
  escribir un escalar. El docstring de `PUT /profile/plan-mode` explica por qué *aquel*
  quedó fuera del whitelist — porque es una **transacción** (cancela cola, libera locks).
  Un endpoint propio para el idioma solo añadiría un limitador, una fila más en la tabla
  de exención de cuota y una segunda puerta al mismo `UPDATE`.
- **Validación en dos capas**: `_LOCALE_VALUES` en el endpoint (400 legible) **y**
  `CHECK` en la columna (protege a cualquier escritor que no pase por ahí: scripts de
  soporte, backfills, endpoints futuros que reutilicen el whitelist). Mismo criterio que
  la invariante I8.

### El orden de las operaciones al cambiar de idioma

`Settings.jsx::handleSelectLocale` **aplica primero y persiste después**, y no es el
orden obvio:

- Si el catálogo no baja (offline, chunk 404), **no se guarda nada**. Guardar primero
  dejaría el perfil diciendo `fr-FR` con la pantalla en español, y el siguiente arranque
  intentaría un idioma que ya sabemos que no está.
- Si el catálogo baja pero el `PATCH` falla, **el idioma se queda cambiado** y se avisa
  de que no se sincronizó. Revertirle la pantalla al usuario por una escritura fallida
  que a él no le consta es peor que una sincronización pendiente.

## 5. `npm run i18n:check` — la red

| Comando | Qué hace |
|---|---|
| `npm run i18n:check` | Falla con **huérfanas**, plurales mal declarados y `t()` en ámbito de módulo. |
| `npm run i18n:check:strict` | Además exige **100% de cobertura** en los 4 idiomas. |
| `npm run i18n:template` | Rellena los catálogos con las claves faltantes en blanco, listas para traducir. |
| `npm run i18n:baseline` | **Reescribe los trinquetes.** [P3-I18N-DOC-GATE-SIN-ESCOTILLA · 2026-08-22] Esta fila faltaba, en la sección que llama al gate «la única defensa»: es la palanca que puede desactivarla, y no estaba escrita. Bajar el trinquete es lo normal —una pantalla traducida— pero SUBIRLO exige además `--allow-ratchet-up`, que no tiene alias en `package.json` a propósito: quien lo suba tiene que teclearlo entero. Desde `P2-I18N-ESCANER-RECALL` el trinquete de español sin envolver está en **0**, así que cualquier subida es una regresión, no una deuda heredada. |

Una entrada de plural declarada como cadena simple traduce **en singular siempre**, sin
avisar — por eso es un fallo duro y no un aviso.

**La escotilla `MEALFIT_CI_I18N_STRICT` existe SÓLO en el gate local, y es deliberado**
[P3-I18N-CI-ESCOTILLA-SOLO-LOCAL · 2026-08-22]. Tres superficies, tres tratos distintos, y
la asimetría es el diseño:

| Dónde | Trato | Por qué |
|---|---|---|
| `backend/scripts/run_ci.ps1` (gate local) | La respeta: `=0` baja a permisivo | El caso legítimo existe — una tanda larga a medio traducir. |
| `deploy-mealfit.ps1` | **`throw` si está puesta** | En PowerShell una variable de entorno vive TODA la sesión: quien la puso por la mañana para traducir la sigue teniendo puesta al desplegar por la tarde. `-SkipTests` es la válvula equivalente que deja rastro POR INVOCACIÓN en vez de por sesión — una la escribes cada vez, la otra se te olvida puesta. |
| GitHub Actions | **No existe** | Es el juez que no puede ser negociable. Añadirla ahí convertiría el gate en una sugerencia. |


## 6. Añadir un sexto idioma

[P2-I18N-ESPEJOS-SIN-ANCLA · 2026-08-21] Esta sección decía «**cinco** sitios» y el test
«falla si los cinco divergen». **Son doce**, y siete no estaban anclados por nada. No se
pueden colapsar: el boot corre antes de que exista ningún módulo, el CHECK debe ser SQL, y
el backend no puede importar JS.

La columna que importa es la última: **un idioma que falte en un espejo no rompe nada**.
Esa superficie sale en español y ya. Por eso el drift aquí es más callado que el de
`P1-DIET-CANON-SSOT` —donde tres tablas de dieta drifearon y a la del filtro se le olvidó
`'vegetariana'`, y el sistema servía Pollo a vegetarianas—: allí al menos alguien lo veía.

| # | Sitio | Qué se ve si falta |
|---|---|---|
| 1 | `frontend/src/i18n/locales.js` | **SSOT.** El idioma no existe: ni sale en el selector. |
| 2 | `frontend/src/i18n/locales/<code>.json` | El `import()` revienta, `loadLocale` se traga la excepción y devuelve `false`: la app entera en español, sin error en consola. |
| 3 | `frontend/src/i18n/index.js` → `LOADERS` | El selector acepta el idioma y el catálogo no se descarga nunca. Mismo síntoma que el anterior, causa distinta. |
| 4 | `frontend/index.html` → `SUPPORTED` | `<html lang>` arranca en `es-DO` hasta que React monta: parpadeo, y un lector de pantalla leyendo francés con voz española en el arranque en frío. |
| 5 | `migrations/p1_i18n_dashboard_locale_2026_08_15.sql` → `CHECK` | El `PATCH` revienta contra la constraint: el usuario elige el idioma, el navegador cambia, y al recargar vuelve al anterior. |
| 6 | `backend/migrations/…` (misma migración, P3-MIGRATIONS-SSOT) | Igual que la 5 en el entorno que despliegue esa copia. Las dos tienen que ser byte-idénticas. |
| 7 | `backend/routers/user_data.py` → `_LOCALE_VALUES` | El endpoint rechaza el valor antes de llegar a la DB. El usuario cree que la app «no guarda» su elección. |
| 8 | `backend/prompts/chat_agent.py` → `_COACH_LANGUAGE_NAMES` | La app en ese idioma y el coach contestando en español. |
| 9 | `backend/prompts/chat_agent.py` → `_TITLE_LANGUAGE_DIRECTIVES` | Los títulos de conversación del chat nacen en español dentro de una app en otro idioma. |
| 10 | `backend/plan_display_i18n.py` → `_DISPLAY_LANGUAGE_DIRECTIVES` | `_build_prompt` devuelve `None` y el enriquecimiento se salta ENTERO: plan y recetas en español, sin error. |
| 11 | `backend/plan_display_i18n.py` → `_PLAN_NAME_ADDENDUM` | Los platos salen traducidos y el nombre del plan se queda en español: media pantalla en cada idioma. |
| 12 | `backend/plan_display_i18n.py` → `_INSIGHTS_ADDENDUM` | Igual que la 11, con el razonamiento del panel de insights. |

Los espejos tienen tests **por separado** y no uno que compare los doce conjuntos, y eso es
deliberado: un único test diría «algo divergió» y te dejaría buscando; uno por espejo dice
cuál y qué se ve. Verificado por mutación — añadir `'de-DE'` al SSOT los pone rojos.

**Dónde vive el ancla de cada fila** [P2-I18N-DOC-ESPEJOS-INCOMPLETOS · 2026-08-22]: esta
sección decía «cada fila tiene su propio test» y no es exacto —
[`test_p2_i18n_espejos_sin_ancla.py`](../tests/test_p2_i18n_espejos_sin_ancla.py) cubre 10
espejos con 9 funciones (una parametrizada sobre los dos addenda del display), y las filas
**5 y 6** —las dos copias del `CHECK`— las ancla
[`test_p1_i18n_dashboard.py`](../tests/test_p1_i18n_dashboard.py), que es donde vive la
paridad de migraciones. La cifra la vigila ahora `test_p2_i18n_doc_espejos_incompletos.py`: si alguien
añade un espejo sin su test, o un test sin su fila, sale rojo.

⚠️ **La migración de la fila 5 tiene una parte SUPERSEDED.** `ADD COLUMN … NOT NULL DEFAULT
'es-DO'` fue revertido por
[`p1_i18n_profile_locale_nullable_2026_08_21.sql`](../migrations/p1_i18n_profile_locale_nullable_2026_08_21.sql)
(aplicada a Neon el 2026-08-22): la columna admite `NULL`, y ese `NULL` es lo que distingue
«no ha elegido» de «eligió español» — sin él, el primer login apagaba la autodetección para
siempre. El `CHECK`, que es lo que esta fila cuenta como espejo, sigue en la migración de
agosto-15.

## 7. Tests

| Test | Qué ancla |
|---|---|
| [`test_p1_i18n_dashboard.py`](../tests/test_p1_i18n_dashboard.py) | Paridad de los espejos históricos (boot, CHECK, backend), idempotencia de la migración, whitelist + validación de valor, `es-DO` sin catálogo, existencia del validador. |
| [`test_p2_i18n_espejos_sin_ancla.py`](../tests/test_p2_i18n_espejos_sin_ancla.py) | 10 de los 12 espejos de la lista de idiomas (9 funciones, una parametrizada), con la consecuencia de cada divergencia en el mensaje. Los dos `CHECK` los ancla `test_p1_i18n_dashboard.py`. |
| [`test_p2_i18n_doc_espejos_incompletos.py`](../tests/test_p2_i18n_doc_espejos_incompletos.py) | Que la CIFRA de esta doc siga siendo la de la realidad: filas de la tabla ↔ espejos con ancla. |
| `frontend/src/__tests__/I18n.p1_i18n_dashboard.test.js` | Contrato del motor: fallback al español, fail-closed del locale, interpolación (placeholder sin valor se queda **literal**), plural, `<html lang>`, formato por locale. |
| `test_p3_i18n_deferred.py` | **Reconvertido**: ya no guarda «es-DO permanente» sino «no añadas una librería de i18n encima del motor propio». |

### Un hallazgo que cuesta redescubrir

`es-DO` y `en-US` **formatean los números igual** (`1,234.5`): República Dominicana usa
punto decimal y coma de millares, como EE.UU. y a diferencia de España. Un test que
intente demostrar «el formato sigue al idioma» comparando ese par **falla siendo el
código correcto** (pasó al escribir la suite). Para demostrarlo hace falta `fr-FR`, que
usa coma decimal y espacio fino.
