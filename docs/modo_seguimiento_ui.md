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
