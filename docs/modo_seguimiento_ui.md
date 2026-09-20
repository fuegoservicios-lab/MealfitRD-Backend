# Modo seguimiento: qué se muestra y qué no (`P1-PLAN-LOTE-86` · 2026-09-17)

**Sin medidor de créditos en el contador.** Pregunta del dueño con captura («Créditos 10/10» con el generador apagado):
¿en qué se usan? En nada. El crédito mensual (`api_usage`; gratis 10, basic 50, plus 200, ultra 500) solo lo consumen
las superficies del generador, las que llevan `Depends(verify_api_quota)`: generar plan (`POST /api/plans`), `/analyze`
(+stream), `/swap-meal`, `/recipe/expand`, `/{plan_id}/regenerate…`, `/fix-sodium`, `/retry-chunk`, `/chunks/…` y
`/regen-degraded`. El coach tiene su cuota mensual aparte (`P1-COACH-QUOTA-METER`, 60 en gratis) y escanear, estimar o
anotar comidas va a `llm_usage_events`, nunca a `api_usage` (tabla de exenciones en CLAUDE.md).

Con `plan_mode = 'tracking'` ninguna de esas superficies es alcanzable (el plan está en pausa y la navegación por modo no
las ofrece), así que `DashboardTracking` deja de montar `CreditsMeter`. El dashboard de plan (`Dashboard.jsx`) lo
conserva: al reanudar el plan, el medidor vuelve con la cifra real.

Tests: `frontend/src/__tests__/DashboardTracking.no_credits_meter.test.jsx` y `backend/tests/test_p1_plan_lote_86.py`
(que además ancla que el diario y el chat NO cuelgan de `verify_api_quota`: si alguien lo hace, la premisa cae).

## El contador en el teléfono (`P1-PLAN-LOTE-87` · 2026-09-17)

Tres capturas del dueño a 392px: «en móviles todavía le falta; la señal de que quieres que la IA te arme tu plan debería
estar arriba de primero; se siente muy encogido lo del contador de macros».

- **La invitación a encender el plan va primera en el teléfono.** Era la última tarjeta de la columna lateral (debajo de
  la hidratación), que en móvil se apila bajo el contador. Ahora es hija DIRECTA de la rejilla (`.turnOnSlot`) y el orden
  lo dan las áreas: escritorio `"main side" / "main plan"` (bajo la hidratación, como antes); móvil `"plan" / "main" /
  "side"`. Ni segundo render ni `order` dentro de la columna.
- **La tarjeta de macros recupera aire en ≤480px.** `P1-MACRO-CARD-DENSITY` (2026-08-10) la apretó para que la lista de
  comidas cupiera sin scroll; el dueño la siente encogida. Se afloja donde no cuesta legibilidad: relleno 1 → 1,15 rem,
  hueco entre bloques 1 → 1,3 rem, entre macros 0,75 → 1 rem, cabecera 1/0,75 → 1,25/0,9 rem (y en ≤768px 1,15 → 1,4 y
  0,9 → 1,15). Letras y altura de barra intactas: sus dos correcciones de agosto («no las vuelvas más pequeñas») siguen.

Tests: `frontend/src/__tests__/DashboardTracking.mobile_order.test.jsx` y `backend/tests/test_p1_plan_lote_87.py`.

## El contador en el teléfono, sin «tarjeticas» (`P1-PLAN-LOTE-88` · 2026-09-17)

Pregunta del dueño con captura a 392px, tras el lote 87: «¿qué opinas si quitamos eso de las tarjeticas en móviles? hay
menos espacio y quiero que sea lo más cómodo visualmente». Medido en un arnés local a ese ancho: el relleno del shell
(13,6px), el de la página (14,4), el borde (1) y el relleno de la tarjeta (18,4) sumaban **47,4px perdidos por lado**; las
barras usaban 297 de 392px (75,8 %).

- **Las dos secciones grandes van planas en ≤480px**: progreso e hidratación pierden borde, fondo, sombra y relleno y
  usan el ancho entero (360px, canal de 16). Es opt-in por prop (`flatOnMobile` → clase `.flatMobile`): las tarjetas se
  comparten con el dashboard de plan, que NO cambia en este lote. La regla oscura va en el mismo selector —
  `html[data-theme="dark"] .card` (0,2,1) le ganaría fondo y borde a un `.card.flatMobile` (0,2,0) suelto.
- **Lo que tiene marco, se toca (o se descarta)**: la invitación al plan sigue siendo tarjeta —es un aviso— y algo más
  compacta (156 → 143px: el texto ocupa dos líneas a ese ancho y acortarlo es copy en 4 catálogos); también conservan su
  forma las filas de comidas, el vaso y los botones.
- **Una línea entre secciones**, con 1,5rem a cada lado, sobre `.sideCol:not(:empty)` (con la hidratación apagada el
  componente devuelve null y la línea colgaría sobre nada).
- **Sin marco, una sección se presenta por su título**: en plano la cabecera «Hidratación» va antes del vaso
  (`display: contents` en `.body` + `order: -1`); con tarjeta el vaso podía ir primero porque el marco ya agrupaba.

El bloque del aplanado va al FINAL de cada módulo: el primer `@media (max-width: 480px)` de `TrackingProgress.module.css`
lo anclan los tests del lote 87. Arnés (no versionado): Vite con raíz fuera del repo, `DashboardTracking` real y stubs de
`config/api`, `AssessmentContext` y `planModeResume`.

Tests: `frontend/src/__tests__/DashboardTracking.flat_mobile.test.jsx` y `backend/tests/test_p1_plan_lote_88.py`.

## Lo que se descarta no encabeza (`P1-PLAN-LOTE-91` · 2026-09-17)

Reporte del dueño con captura, tras pulsar «Ahora no» en la invitación al plan: «se puso raro lo del progreso en tiempo
real». Medido en el arnés a 392px: al descartarla, la tarjeta colapsa a un enlace (`.turnOnLink`, 27px de texto tenue) que
se quedaba en la PRIMERA posición del teléfono (`P1-PLAN-LOTE-87`). Con la tarjeta eso funcionaba —su marco la separaba—,
pero desde que las secciones van sin marco (`P1-PLAN-LOTE-88`) el enlace quedaba a 24px del título, sin nada en medio, y se
leía como una línea DEL «Progreso en Tiempo Real». El defecto no estaba en el contador: estaba encima.

- **En ≤900px, con el enlace, las áreas se reordenan a `"main" "side" "plan"`**: el contador abre la pantalla y la
  invitación la cierra. La puerta no se borra (esa fue la decisión original del «enciéndelo»), solo deja de encabezar.
  Se reordenan las ÁREAS y no la fila del bloque: en una rejilla con `gap`, una fila vacía sigue cobrando sus dos huecos.
  `:has()` mira hacia abajo desde `.page`; donde no exista, no casa la regla y el enlace se queda arriba — la conducta previa.
- **La tarjeta sin descartar no cambia**: sigue primera, que es lo que el dueño pidió en el lote 87.
- **El enlace cierra con la misma línea fina** que separa la hidratación, para que no parezca su pie.
- **`.sideCol:empty { display: none }`**: con la hidratación apagada en Preferencias el componente devuelve null y su fila
  vacía cobraba dos huecos de rejilla (48px de aire entre el contador y el enlace). Ahora sale del reparto.

Tests: `frontend/src/__tests__/DashboardTracking.dismissed_last.test.jsx` y `backend/tests/test_p1_plan_lote_91.py`.

## Descartada, no queda nada (`P1-PLAN-LOTE-98` · 2026-09-18)

El lote 91 colapsaba la invitación descartada a un enlace tenue al final del contador («¿Quieres el plan completo? Enciéndelo aquí» / «Tu plan está en pausa. Reanúdalo aquí»). El dueño, con captura: **«quítalo, ya está el interruptor en configuración»**. `TurnOnPlanCard` devuelve `null` al descartar, en las dos ofertas; la puerta de vuelta es Configuración → Capacidades. El descarte sigue persistiendo en `mealfit_turnon_card_dismissed`.

El hueco «plan» de la rejilla queda vacío: `.turnOnSlot:empty { display: none; }` lo saca del reparto y, en el teléfono, `.page:has(.turnOnSlot:empty)` deja las áreas en `"main" "side"` — una fila vacía PRIMERA seguiría cobrando su hueco (1,5rem de aire muerto sobre el título). Sin `:has()` queda ese hueco y nada roto. Las dos claves salieron de los cuatro catálogos. Test `test_p1_plan_lote_98.py`; `test_p1_plan_lote_91.py` conserva solo lo que no dependía del enlace.

## «Progreso», «Tus macros de hoy», un icono de cámara y el peso en tiempo real (`P1-PLAN-LOTE-102` · 2026-09-18)

Cuatro peticiones del dueño tras probar en el iPhone:

- **Un solo icono de cámara** en «Escanear comida»: se va el tile del título, queda el de «Usar la cámara».
- **Nombres**: la pestaña del contador es «Progreso» (antes «Hoy»: no decía de qué) y la sección «Tus macros de hoy» (antes «Progreso en Tiempo Real»). El aviso «Bórralo en «…» para desbloquear» (`todayRemaining.js`) y el prompt del coach (`prompts/chat_agent.py`) nombran la sección como se ve: mandar al usuario a un nombre que ya no existe es la trampa que cerró `todayRemaining.p1_i18n_eaten_claim`.
- **Peso/altura/edad/sexo en tiempo real en modo contador**: sin generador de planes no hay plan que regenerar ni crédito que gastar. Configuración muestra UN botón «Guardar» (`handleSaveTracking`), persiste en `health_profile` y avisa con `mealfit:targets-changed`; `DashboardTracking` vuelve a pedir `/api/nutrition/targets` (que ya leía `health_profile`) y las barras cambian sin recargar. En modo plan todo sigue igual («Actualizar Plan con Nuevos Datos» regenera).
- **«Elegir de galería» en iOS abre la hoja de tres opciones (Fototeca / Tomar foto / Seleccionar archivo)**: la decide iOS para todo `<input type="file" accept="image/*">` sin `capture`; desde la web no se puede ir directo a la fototeca. La vía directa existe en el cascarón nativo (Capacitor `Camera.getPhoto({ source: Photos })`) — pendiente de decisión del dueño.

Test `test_p1_plan_lote_102.py` (+ `lote102.test.jsx`).

## «Micros de hoy» y la pestaña «Progreso» del modo plan (`P1-PLAN-LOTE-103` · 2026-09-18)

El dueño: «un contador para micronutrientes, así tendríamos progreso en tiempo real para macros y micros; y con el generador encendido, dividir lo que es progreso (macros, micros, hidratación) en un apartado aparte». Aprobó los ocho: fibra, sodio, potasio, calcio, hierro, vitamina C, vitamina A y vitamina D.

**De dónde salen los micros.** De lo que ya guarda cada comida registrada: `consumed_meals.ingredients` (cadenas «150 g de Pechuga de pollo») — las escriben el componedor (líneas del catálogo), «me lo comí» (los ingredientes del plato del plan) y el coach. `GET /api/diary/consumed/{user_id}` las resuelve con el MISMO resolutor que el informe de micros del plan (`IngredientNutritionDB.micros_from_ingredient_string`, SSOT en `diary_micros.py`) y devuelve por comida `micros: {values, resolved, total}` o `null`, y en `totals` la suma y `micros_coverage: {con_datos, total}`. **Los ingredientes no viajan al cliente.** Una comida por foto o con macros propias no trae ingredientes → no trae micros, y la tarjeta lo dice («con datos de 1 de 2 comidas»; «las comidas por foto o con macros propias no traen micros»): mostrar 0 mg de calcio por un plato escaneado sería mentir con número. Fail-open: si el catálogo no carga, el diario responde sin micros.

**Metas.** `/api/nutrition/targets.micros` = `diary_micros.metas_micros` sobre `micronutrients.dri_targets` (DRI/IOM + OMS por sexo/edad/embarazo): el sodio es TECHO (<2000 mg; la barra avisa al pasarse), el resto SUELO. Sin sexo/edad no hay metas y la tarjeta muestra los totales sin barra.

**La división.** `dashboardNav`: en modo plan aparece `progress` («Progreso», `/dashboard/progress`) tras «Plan» — 6 pestañas abajo en el teléfono, medido a 392 px sin recortes. `ProgressPage` = `DashboardTracking modo="plan"` (metas del plan vigente, sin invitación a encenderlo; macros, micros e hidratación). `Dashboard.jsx` deja de montar `TrackingProgress` y `WaterTracker`; lo que necesita saber de hoy («Tu Menú» atenúa el plato ya comido y bloquea «Cambiar plato») le llega por `useTodaysConsumedMeals`: adopta `mealfit:today-consumed-updated` si el contador está montado y, si no, pide el diario él mismo con las mismas señales (`mealfit:refresh-inventory`, `mealfit:diary-changed` —nuevo, lo emite `TrackingProgress` al borrar—, `visibilitychange`); una época invalida los fetches en vuelo cuando llega un evento, para que el fetch del montaje no pise lo adoptado. En modo contador nada cambia de sitio: «Progreso» sigue siendo la primera pestaña y gana la sección de micros.

Tests: `test_p1_plan_lote_103.py` (módulo + anclas), `MicrosTracker.lote103.test.jsx` (tarjeta, hook y división), `Dashboard.eaten_slot_unlock.test.jsx` (reescrito: el round trip pasa por el hook).

## Macros y micros en una tarjeta, y el diario de días anteriores completo (`P1-PLAN-LOTE-105` · 2026-09-18)

El dueño: «¿y qué tal si fusionas lo de micros con lo de macro? en "ver días anteriores" quiero también ver el historial de micros y solo se ve el de macros [...] revisa si el sistema de "días anteriores" está en su 100% posible [...] y lo de la fototeca directa en la app nativa Capacitor hazlo también». Decisión mía (me lo delegó): fusionar. Razón técnica además de la de producto: «Micros de hoy» hacía un SEGUNDO fetch del mismo `GET /api/diary/consumed/{user_id}` que ya hacía «Tus macros de hoy»; dos fuentes de la misma verdad que podían divergir tras un borrado.

**La tarjeta.** «Tus macros y micros de hoy» (`TrackingProgress`): calorías y las tres barras, después la sección «Micros» (`MicrosList`, antes `MicrosTracker` con fetch propio; ahora solo pinta) con la cobertura honesta, después la lista de comidas y «Ver días anteriores». El snapshot cacheado (`_buildConsumedSnapshot`) lleva `micros`/`microsCoverage` y los recalcula desde `meals[].micros` con la aritmética de `diary_micros.resumen_micros` (`resumirMicros`), así el borrado optimista deja macros y micros coherentes sin esperar al servidor. Las metas (`microTargets`) las pasa `DashboardTracking` desde `/api/nutrition/targets.micros` en los dos modos — en modo plan el plan trae las macros, no las metas DRI de los micros. El endpoint devuelve `micros` también con `ok:false` cuando hay sexo y edad (es lo único que las DRI necesitan; el perfil de seguimiento en modo plan puede estar incompleto). El nombre cambia también en el prompt del coach y en `todayRemaining`.

**El diario de días anteriores (`DiaryHistory`), auditado.** Lo que le faltaba:
- **Micros del día**: `totals.micros` de ese día (el endpoint ya los traía para cualquier fecha) con `MicrosList compact` y sus metas (`targetMicros`).
- **Comidas «extra» invisibles** — bug real: el componedor registra `meal_type='extra'` POR DEFECTO (`ManualMealRequest`) y el cajón solo dibujaba las cuatro franjas + `snack`; la comida contaba en el total y no salía en ninguna fila. Ahora todo lo que no es franja va al grupo «Extras y snacks».
- **Borrar desde cualquier día** (solo hoy tenía papelera, en la tarjeta): mismo `DELETE /api/diary/consumed/{meal_id}` filtrado por `user_id`; tras borrar se vuelve a pedir el día y la tira, y se emite `mealfit:diary-changed` con `detail.source='diary-history'`. `TrackingProgress` lo escucha y vuelve a pedir hoy salvo que el evento sea suyo (`source='tracking-progress'`, ya actualizó el estado optimista).
- **Registrar en el día que miras**: «Registrar en este día» abre el componedor con `initialDaysAgo` (hasta 7, el tope de `days_ago` en el backend; más atrás se dice «solo hasta 7 días»); si es más atrás que «Antier», el día pedido se añade como chip con su fecha para que el usuario lo vea y pueda cambiarlo. Sin `onScan`: la foto no es retrodatable.
- **Más de 14 días**: «+14» al principio de la tira, hasta 90 (el clamp de `/consumed-range`). **La semana en una línea**: media de kcal en los días con registro de los últimos 7.
- **Refresco en caliente**: mientras está abierto escucha `mealfit:refresh-inventory` (el componedor, el escáner y el chat) y `mealfit:diary-changed` (la tarjeta) — antes había que cerrarlo y abrirlo.

**Fototeca directa en la app nativa.** En iOS el `<input type="file" accept="image/*">` abre SIEMPRE la hoja de tres opciones (Fototeca / Tomar foto / Seleccionar archivo) y no hay forma de evitarla desde la web (queja del dueño con captura, lote 102). En nativo, «Elegir de galería» del escáner usa `Camera.chooseFromGallery` (`chooseNativeGalleryImage`, mismo módulo que el chat ya usaba) con selección única; cancelar no es error y cualquier otro fallo cae al input de siempre. El plugin ya estaba en `Package.swift` y el permiso en `Info.plist`. **Requiere un build nativo nuevo** (Codemagic): la PWA no cambia.

Tests: `test_p1_plan_lote_105.py`, `lote105.test.jsx` (tarjeta fusionada, cajón con micros/extras/borrar/registrar/+14, componedor en el día pedido), `MicrosList.lote103.test.jsx` (reanclado).

## El corte del teléfono es 768, no 480 (`P1-PLAN-LOTE-92` · 2026-09-17)

Reporte del dueño con captura de su iPhone, ya con los lotes 88-91 desplegados: «se ve estrecho, mira todo el espacio que
tiene los bordes de los lados, hay vacíos».

**Lo medido antes de tocar nada.** En el log de nginx, su teléfono cargó a las 22:23 UTC el `index.html`, el
`Dashboard-la8y58kc.js` y el `Dashboard-CZ2kqYUb.css` de la release vigente (no era caché vieja del PWA); en ese JS
descargado de producción, `DashboardTracking` pasa `flatOnMobile:!0` a las dos secciones, y en ese CSS está la regla
`.card.flatMobile{padding:0;border:0;…}`. Sobre la captura, el contenido ocupaba ~75 % del ancho y las líneas divisorias
tenían margen a los lados: es el número exacto que sale con la tarjeta SIN aplanar (relleno 1,25rem + borde) y la página a
0,9rem. O sea: el CSS y el JS eran los buenos y aun así el bloque `@media (max-width: 480px)` no casaba en su pantalla.

**Por qué.** El viewport CSS de un teléfono no es fijo: el zoom de sitio de Safari por debajo del 100 % lo ENSANCHA (un
iPhone de 390 pt al 75 % reporta 520 px). Cualquier iPhone con esa preferencia se salía del diseño. Un diseño que se cae por
40 px de zoom no es el diseño: es una coincidencia.

**El cambio.** Los tres bloques del teléfono (aplanado del contador, aplanado de la hidratación, y el relleno/líneas de la
página con el enlace del lote 91) pasan de `max-width: 480px` a `max-width: 768px`, que es el MISMO corte con el que
`DashboardLayout` ya se vuelve teléfono (`.mainContent { padding: 0.65rem 0.85rem }`) y con el que la tarjeta baja a
1,25rem y su icono a 40px. Medido en el arnés: a 520px el contenido pasa de 75 % a 93,8 % del ancho; a 392px no cambia
(91,8 %); a 820px la tarjeta sigue siendo tarjeta.

El bloque de 480 no se borra: sigue llevando lo que sí es cuestión de pantalla pequeña (el aire de la tarjeta del lote 87,
que actúa en modo plan, donde no hay aplanado).

Tests: `frontend/src/__tests__/DashboardTracking.phone_breakpoint.test.jsx` y `backend/tests/test_p1_plan_lote_92.py`
(que además ata el corte al del armazón: si `DashboardLayout` mueve su frontera, cae el test y no el teléfono de nadie).
Las anclas de los lotes 88 y 91 quedaron reconvertidas al bloque de 768.

## Revisión a fondo del móvil (`P1-PLAN-LOTE-94` · 2026-09-17)

El dueño pidió una revisión completa («a ver si no hay ningún otro bug o diseño visual mal hecho»). Se barrió el armazón
real —`DashboardLayout` con Hoy, Nevera, Historial y Configuración— a 320, 392, 430, 520 y 768 px, en claro y oscuro, con
una sonda que mide desborde horizontal y detecta hijos mucho más estrechos que su padre. Dos defectos con arreglo:

- **El desplegable del tipo de comida cortaba su valor por defecto.** En el componedor, «Extra (fuera del plan)» se veía
  como «Extra (fuera del» a 392 px y «Extra (fuel» a 320. Los dos desplegables se repartían la fila a partes iguales
  (`flex: 1`) aunque, medido con la tipografía real, el primero pide 185 px y el segundo 85. Ahora la fila es una rejilla
  `minmax(0, 1fr) auto` (el día ocupa lo que mide su texto) y por debajo de 380 px se apilan. `text-overflow: ellipsis` como
  red de seguridad para traducciones más largas.
- **En tema claro el contador se leía sobre las burbujas.** Desde el lote 88 las secciones no tienen tarjeta, así que el
  texto queda sobre `DashboardLayout .container::before` (la imagen decorativa al 85 %): el subtítulo `#64748B` ronda 4,1:1
  ahí, cuando sobre la tarjeta blanca daba 4,9:1 — por debajo del 4,5:1 que pide AA para texto pequeño. La página se apoya
  ahora en el color liso (`--bg-page`) **solo en claro**; en oscuro el contraste sobra y el degradado superior es parte de
  la identidad que el dueño eligió. Mismo criterio que `SETTINGS-MOBILE-WHITE-BG`.

Lo que se revisó y salió limpio: sin desborde horizontal en ninguna página ni ancho (320-768); las cuatro pestañas y sus
estados vacíos; el componedor, el escáner y el diario de días anteriores; la campana atracada y su cajón; el contador con y
sin comidas, con la hidratación apagada y con el plan en pausa.

Queda anotado para el dueño, porque es decisión suya y no un defecto: en claro, el fondo de burbujas se ve ahora en el
resto de páginas con tarjeta (donde nunca estorbó) y ya no en el contador.

Tests: `frontend/src/__tests__/MobileReview.lote94.test.jsx` y `backend/tests/test_p1_plan_lote_94.py`.

## Configuración con el generador APAGADO: auditoría y cierre (lote 136)

[P1-PLAN-LOTE-136 · 2026-09-20] El dueño: «revisa en general el apartado de configuración cuando el generador de planes está
apagado: ¿funciona todo al 100 %? ¿No hay ninguna contradicción?». Se leyó `Settings.jsx` entero y cada endpoint que dispara.
**Lo que salió limpio:** ningún acceso a `planData` sin guarda, ninguna validación que exija campos que la rama corta no
preguntó, ningún endpoint por el paywall (cero 402), y guardar el perfil no encola generación ni gasta créditos.

**Lo que se cerró** (regla de fondo: *contador manda* en TODA la pantalla, no solo cuando no hay plan):

| # | Contradicción | Cierre |
|---|---|---|
| 1 | «Modo automático» («no pausaremos tu plan…») visible en contador; `logging_preference` solo lo lee el worker de bloques, apagado ahí | la tarjeta se pinta solo con el generador encendido |
| 2 | Con un plan EN PAUSA, «Plan & Objetivo» enseñaba las kcal congeladas del plan y vendía «Evaluar de nuevo» (1 crédito) o «sin créditos»; el contador usa `/api/nutrition/targets` | en contador las metas salen de targets con o sin plan, el panel escucha `mealfit:targets-changed`, y el botón es «Reanudar el plan» (gratis, `reanudarPlanes`) |
| 3 | El interruptor no actualizaba `userProfile.plan_mode` en memoria y sin plan no hay recarga: «Guardar», la navegación y el dashboard seguían en el modo viejo | `refreshProfileAndPlan()` tras el PUT sin plan; reanudar devuelve `appMode` a `plan` |
| 4 | El diálogo y la tarjeta prometían «tu plan queda en el Historial» a quien nunca tuvo plan | copy propio sin plan |
| 5 | Si fallaba `GET /api/profile/plan-mode`, el interruptor —la única puerta de vuelta (lote 98)— desaparecía sin aviso | un reintento y, si no, una fila con «Reintentar» |
| 6 | Con `MEALFIT_PLAN_MODE_SWITCH=false` el PUT contesta éxito sin hacer nada y la pantalla pintaba «Planes en pausa» | manda el `plan_mode` que CONTESTA el servidor |
| 7 | Móvil: «0 kcal · 0 g» sin metas (el escritorio pintaba «—») | «—» también en `PlanObjetivo` |
| 8 | «Guardar» escribía el formulario ENTERO en `health_profile` (los 12 campos nunca preguntados, `appMode`) | perfil del servidor + lo editado |
| 9 | El aviso de salir sin guardar decía «no actualizaste tu plan» | copy propio en contador |
| 10 | SERVIDOR: `_revive_paused_chunks` revivía las filas firmadas de TODOS los planes del usuario, también de uno ya sustituido | solo el plan vigente (el último); las demás se quedan canceladas hasta la purga |

Abierto a sabiendas (LOW, medido como inofensivo hoy): cambiar de idioma o pausar puede encolar traducción `display_i18n` de
un plan pausado que el contador no muestra (no gasta créditos; solo con `MEALFIT_PLAN_JOBS_ENABLED`); `mealfit_plan_mode` no
se limpia al cerrar sesión (el perfil lo corrige al cargar); la sección Suscripción enseña créditos que el contador no usa.

Tests: `frontend/src/__tests__/lote136.test.jsx`, `backend/tests/test_p1_plan_lote_136.py`.

## La app ENTERA con el generador apagado (lote 137)

[P1-PLAN-LOTE-137 · 2026-09-20] El dueño, tras el lote 136: «quiero saber si en general con el generador apagado todo está al
100 % listo para producción». Cinco auditorías de solo lectura en paralelo (rutas y efectos globales del cliente, Nevera +
Historial, Agente IA, crons/notificaciones/endpoints de plan, ciclo de vida de la cuenta); cada hallazgo se verificó a mano
antes de contarlo y la base de producción se leyó en solo lectura (2 usuarios en contador, 0 planes, 0 colas vivas).

**Lo del SERVIDOR que se cerró:**

| # | Defecto | Cierre |
|---|---|---|
| 1 | **«Encender el plan» desde el contador dejaba la generación colgada.** `POST /generation-runs` encolaba el chunk 0 sin tocar `plan_mode`; el reencendido (`ensure_plan_generation_enabled`) vivía solo en el postprocess, que en la cola corre DESPUÉS del pickup, y el pickup lleva el gate H1. Run `PAUSED` para siempre, hasta 70 min de pantalla de carga y un «Plan en preparación» vacío como plan vigente. El SSE legacy no lo sufría: lo rompió el flip a la cola | la bandera PRIMERO: se enciende antes de encolar; si encolar falla, el usuario vuelve a su contador |
| 2 | Encender por esa vía solo movía la bandera: los planes viejos quedaban sellados `paused_by_user` con el usuario ya en modo plan, y «Reactivar» uno copiaba el sello al plan activo | `_restore_paused_plan_status`: un UPDATE, dos llamadores (reanudar y encender-al-generar). La cola NO se revive ahí: se pidió un plan nuevo |
| 3 | «Reactivar este Plan» en pausa: `/restore` solo cancelaba los 5 estados vivos; las filas `cancelled` firmadas por la pausa sobrevivían y al reanudar revivían sobre el contenido restaurado | los dos cancels de `/restore` las cubren (un `OR` fuera del `IN`), las vuelven terminales y les quitan la firma |
| 4 | El coach citaba como «meta» de hoy las macros del plan EN PAUSA (y «134gg»), 20 líneas después de que las kcal ya salieran del contador | `_macro_totals_line(consumed, plan_vigente, form_data)`: sin plan que mande, `coach_day_context.metas_del_dia` |
| 5 | «Lo que el plan MANDABA» se inyectaba con el plan en pausa (el shift no corre en contador: días congelados citados como prescritos) | en pausa no hay índice ni «sin registrar» del plan; el diario multi-día sigue |
| 6 | El modo solo llegaba al prompt colgado de un plan pausado: al contador SIN plan el coach le decía «usa los botones de la página Plan» y afirmaba «su plan actual»; nada nombraba la puerta real | tercer caso `contador_sin_plan` (bullet propio: Configuración → Capacidades), `build_inventory_context(sin_plan=…)` |
| 7 | Cada tarde el prompt mandaba a una tool de mutación que el agente no tiene enlazada y daba por hecho un déficit | frase neutral, por objetivo |
| 8 | `scheduleType` ausente (la rama corta no lo pregunta) se sembraba como «Día Clásico… rigor estricto» | ausente ⇒ «no ha dicho su horario» |
| 9 | «Fui al súper» (`mark_shopping_list_purchased`) metía en la Nevera el delta del plan pausado y lo marcaba comprado; `check_shopping_list` lo daba como compra pendiente | guarda de modo en la primera; encuadre «lista del plan en pausa» en la segunda |
| 10 | `/retry-chunk`, `/regenerate-simplified` y `/regen-degraded` (pestaña vieja) revivían filas firmadas sin snapshot, pisaban `paused_by_user` y cobraban un crédito por bloques que el pickup no recoge | `_rechazar_si_generador_apagado` ⇒ 409 con la puerta real |
| 11 | El escalado de «chunk atascado» mandaba «Optimizando tu plan…» cada ~24 h a un contador con un chunk vivo detrás de la pausa | mismo `NOT EXISTS` que el gate del pickup |
| 12 | El bot de ayuda no conocía el modo contador y citaba 15 créditos gratis y «Max ilimitado» (son 10 y 500) | bloque de producto al día + cuota del Agente |

**Verificado y BIEN (no tocar):** el gate del pickup en las dos ramas y el bg-refill; `/shift-plan` con soft-skip; el sweep de
huérfanos no toca planes pausados; la purga respeta la firma dentro de la ventana; recordatorios de comidas e hidratación no
leen `meal_plans`; `/api/nutrition/targets` falla cerrado (`ok:false`), nunca 500; la Nevera no exige plan ni pasa por el paywall.

**Abierto a sabiendas:** pausar re-estampa TODOS los planes y eso sube su `revision` ⇒ el barrido de `plan_jobs` puede
traducir planes históricos de un usuario no hispano (gasto único y acotado; el sello global es load-bearing: varios crons se
apagan por lista blanca de estados, no por `plan_mode`); `/swap-meal`, `/regenerate-day` y `/recipe/expand` no miran el modo
(sin camino de UI en modo contador: la nav oculta Recetas y el cliente ahora redirige esa ruta).

La mitad del CLIENTE del lote (rutas directas, Historial, chips del Agente, campana, interruptor sin plan, espejo del modo)
vive en el repo del frontend: `frontend/src/__tests__/lote137.test.jsx`. Tests del servidor: `backend/tests/test_p1_plan_lote_137.py`.

