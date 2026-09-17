# La campana de notificaciones: dónde vive (`P1-PLAN-LOTE-89` · 2026-09-17)

El centro de notificaciones (`frontend/src/components/dashboard/NotificationCenter.jsx`, P3-NOTIF-CENTER) archiva los avisos
descartables del dashboard. Su disparador tiene DOS formas según el ancho:

| Ancho | Forma | Dónde |
|---|---|---|
| > 1024px (escritorio) | tirador-pestaña 40×96 | borde derecho, `position: fixed; top: 38%` (sin cambios) |
| ≤ 1024px (teléfono y tableta) | **orbe** de 40px | **atracado en la cabecera**, a la izquierda del menú ☰ |

## Por qué atraca

Pedido del dueño con captura a 392px: el tirador del borde derecho quedaba encima de la tarjeta de la invitación —y más desde
que el contador va a todo el ancho (lote 88)—: «en vez de la derecha, ¿por qué no mejor arriba? un diseño único y abstracto,
llamativo y que no estorbe ni choque con nada».

- **No choca por construcción.** El botón vive DENTRO de la cabecera (`z-index: 40`): el velo del menú (200), los modales y el
  cajón lo tapan solos. Antes flotaba a `--z-handle: 510` y hubo que esconderlo a mano dos veces (`P3-NOTIF-HIDE-ON-MENU`, la
  clase `mealfit-hide-notif-mobile`). Atracado ya no se esconde con `hidden` (haría parpadear la cabecera al abrir el menú).
- **El hueco lo ofrece la cabecera, no lo busca la campana.** La cabecera móvil de `DashboardLayout` monta
  `<NotificationSlot />`, que registra su nodo en `utils/notifSlot.js`; `NotificationCenter` se suscribe
  (`useSyncExternalStore`) y portaliza ahí el botón. Un almacén y no un `getElementById`: la cabecera se desmonta y se
  vuelve a montar entre rutas, y solo suelta el hueco quien lo tiene. Sin hueco cae al `<body>` con la forma de siempre.
- **Una sola condición: `showNotifCenter`.** El centro vive SOLO en «Hoy» (`/dashboard`), de donde salen sus avisos; el
  hueco se monta con esa misma condición. El chat del Agente pinta su propia cabecera y no lleva campana (nunca la llevó).
- **Un solo corte: 1024px.** Es el ancho al que la cabecera móvil se hace visible (`DashboardLayout.module.css`) y el que usa
  el centro para decidir si atraca. El hueco existe también en escritorio, pero dentro de una cabecera oculta: sin el corte,
  la campana atracaría en una cabecera invisible.
- **El hueco mide 40×40 aunque esté vacío**: si la campana desaparece un momento, el botón del menú no salta de sitio.

## La forma

Un orbe de vidrio con un anillo cónico teal→índigo (la «celda de energía» del centro). En reposo el anillo está quieto y
tenue. Con algo sin leer gira despacio y un satélite lo orbita, y el contador va en la esquina: llama el MOVIMIENTO, no el
tamaño. Mide lo mismo que el botón del menú, así que la cabecera no crece. `prefers-reduced-motion` apaga las dos animaciones.
El cajón no cambió: sigue deslizándose desde la derecha y portalizado al `<body>`.

Tests: `frontend/src/__tests__/NotificationCenter.docked_header.test.jsx` (comportamiento: atraca / cae al body / ignora
`hidden` / suelta el hueco) y `backend/tests/test_p1_plan_lote_89.py` (el contrato entre los tres ficheros).
