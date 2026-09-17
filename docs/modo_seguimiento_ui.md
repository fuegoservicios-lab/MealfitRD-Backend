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
