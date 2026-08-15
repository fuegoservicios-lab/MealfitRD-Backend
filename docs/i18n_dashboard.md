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
| Plan, recetas, lista de compras, respuestas del coach | **No** | Las genera el LLM en español. Traducirlas es cambiar los prompts, y eso multiplica el coste por token y abre un frente de calidad clínica en 5 idiomas. |
| Nombres de alimentos y platos (`master_ingredients`, 206 alimentos + 60 platos criollos) | **No, jamás** | Son el **SSOT del motor**. `pantry_names_match` (P1-PANTRY-NAME-RESOLUTION), el guard de coherencia recetas↔lista y el backstop clínico de alergias resuelven por esos nombres exactos. Traducir «Pollo» rompe las tres cosas a la vez, y dos de ellas en silencio. |
| Páginas legales (Privacidad, Términos — 601 cadenas) | **No** | Traducir un contrato genera obligaciones en cada jurisdicción. Es una decisión legal, no de producto. |
| Landing (`bioboros.com`) | **No** | Son 14 páginas estáticas fuera del build de React (`project_landing_cinematico_v2`). Fuera del alcance pedido («dentro del dashboard»). |

**La consecuencia honesta**: un usuario en japonés —o en francés— ve un menú traducido
alrededor de «Pollo guisado con arroz blanco». Es deliberado y se le dice en la propia
pantalla de Configuración: *«Tu plan, tus recetas y el coach siguen en español.»* La
alternativa (traducir el contenido) es un proyecto distinto, no una fase más de este.

## 2. Los 5 idiomas

`es-DO` (base), `en-US`, `pt-BR`, `fr-FR`, `it-IT`.

Se eligieron 5 y no los 11 de un ChatGPT porque cada idioma multiplica ~1.800
traducciones y ninguna de ellas la revisa un hablante nativo. Cinco idiomas con
calidad defendible es un producto; once con hindi y coreano de terminología
nutricional sin revisar es una demo. **Añadir el sexto es un JSON y cuatro líneas** —
ver §6.

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

### Por qué se **remonta** al cambiar de idioma

`LocaleBoundary` (App.jsx) usa `key={locale}`: cambiar de idioma remonta el subárbol de
rutas. No es exceso — hay copy que se calcula fuera de componentes y subárboles
memoizados, y un re-render no alcanza a ninguno de los dos. El precio es perder estado
transitorio de la vista en un acto deliberado y único. Los providers (sesión, TanStack
Query, wizard) quedan **por encima** de la frontera; `BrowserRouter` relee
`window.location` al montar, así que la ruta se conserva.

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

Una entrada de plural declarada como cadena simple traduce **en singular siempre**, sin
avisar — por eso es un fallo duro y no un aviso.

## 6. Añadir un sexto idioma

La lista vive en **cinco** sitios (no se pueden colapsar: el boot corre antes de que
exista ningún módulo, el CHECK debe ser SQL, el backend no puede importar JS):

1. `frontend/src/i18n/locales.js` — **SSOT**
2. `frontend/index.html` — array `SUPPORTED` del boot
3. `migrations/p1_i18n_dashboard_locale_2026_08_15.sql` — el `CHECK`
4. `backend/migrations/…` — la misma migración (P3-MIGRATIONS-SSOT)
5. `backend/routers/user_data.py` — `_LOCALE_VALUES`

Más el loader en `LOADERS` (i18n/index.js) y el JSON en `i18n/locales/`.

`test_p1_i18n_dashboard.py` **falla si los cinco divergen**. Es la misma clase de drift
que cerró `P1-DIET-CANON-SSOT`: tres tablas de dieta escritas a mano drifearon y a la
del filtro se le olvidó `'vegetariana'` — el sistema servía Pollo a vegetarianas.

## 7. Tests

| Test | Qué ancla |
|---|---|
| [`test_p1_i18n_dashboard.py`](../tests/test_p1_i18n_dashboard.py) | Paridad de los 5 espejos, idempotencia de la migración, whitelist + validación de valor, `es-DO` sin catálogo, existencia del validador. |
| `frontend/src/__tests__/I18n.p1_i18n_dashboard.test.js` | Contrato del motor: fallback al español, fail-closed del locale, interpolación (placeholder sin valor se queda **literal**), plural, `<html lang>`, formato por locale. |
| `test_p3_i18n_deferred.py` | **Reconvertido**: ya no guarda «es-DO permanente» sino «no añadas una librería de i18n encima del motor propio». |

### Un hallazgo que cuesta redescubrir

`es-DO` y `en-US` **formatean los números igual** (`1,234.5`): República Dominicana usa
punto decimal y coma de millares, como EE.UU. y a diferencia de España. Un test que
intente demostrar «el formato sigue al idioma» comparando ese par **falla siendo el
código correcto** (pasó al escribir la suite). Para demostrarlo hace falta `fr-FR`, que
usa coma decimal y espacio fino.
