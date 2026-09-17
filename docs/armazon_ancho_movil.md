# El ancho del dashboard en el teléfono (`P1-PLAN-LOTE-93` · 2026-09-17)

## El síntoma

El dueño, con tres capturas y dos intentos míos fallidos de por medio: «se ve estrecho, mira todo el espacio que sobra, y al
sobrar se ve chiquito el contenido, **ese problema sucede cuando se le da a "Ahora no"** de ¿quieres que la IA te arme un
plan?».

## La causa

`.mainContent` (`DashboardLayout.module.css`) lleva de base:

```css
.mainContent { padding: 2.5rem; max-width: 1200px; margin: 0 auto; }
```

Eso centra la columna de contenido en escritorio, donde `.mainContent` es un bloque normal y ocupa el ancho disponible. Pero
en `@media (max-width: 1024px)` su padre `.mainWrapper` pasa a `display: flex; flex-direction: column`, y **un ítem flex con
márgenes automáticos en el eje transversal no se estira**: se dimensiona a su contenido (`fit-content`) y se centra con el
sobrante repartido a los lados.

Consecuencia: en el teléfono, el ancho de TODO el dashboard lo decidía el texto más largo que hubiera en pantalla. La
tarjeta «¿Quieres que la IA te arme el plan?» («Te faltan 19 preguntas del formulario y usa 1 crédito de tu mes.») era justo
lo que lo mantenía ancho; al descartarla, el dashboard entero encogía.

Medido con el armazón real a 392 px: **con la tarjeta 360 px, sin ella 294 px centrados; con `width: 100%`, 360 en los dos
casos.** Afectaba a todas las páginas del dashboard, no solo al contador.

## Por qué tardé tres lotes en verlo

El arnés visual montaba `DashboardTracking` **suelto**, dentro de un `<div>` que imitaba el relleno de `.mainContent`. Esa
imitación no reprodujo el contexto flex del padre ni los márgenes automáticos, así que el bug era invisible ahí: medía 16 px
de margen cuando en el teléfono eran 49. Las dos hipótesis anteriores (caché del PWA, corte de 480 px) se descartaron con
datos —los ficheros que su teléfono pidió al servidor, y el JS y el CSS descargados de producción—, pero ninguna era la
causa. El arnés ahora monta `DashboardLayout` completo (`layout.html`).

**Lección: un arnés que imita al padre prueba el hijo, no la página.**

## Lo que NO cambia

El centrado de escritorio: `max-width: 1200px; margin: 0 auto` sigue igual. `width: 100%` no lo contradice — lo completa,
porque con un ancho definido los márgenes automáticos solo reparten el sobrante (cero en móvil, el resto en escritorio).

Tests: `frontend/src/__tests__/DashboardLayout.full_width_mobile.test.jsx` y `backend/tests/test_p1_plan_lote_93.py`.
