# Registrar comida: un botón, dos vías (`P1-PLAN-LOTE-85` · 2026-09-17)

La tarjeta **Progreso en Tiempo Real** tenía dos botones lado a lado, «+ Registrar comida» y un icono de cámara
(`P1-MANUAL-FOOD-LOG`, 2026-08-11). El dueño: «desde afuera las dos opciones se ve un poco raro estéticamente».

Ahora la tarjeta tiene **un solo botón**, y las dos vías aparecen **dentro del componedor** (`LogMealModal`) como un
selector de dos opciones bajo el título:

- **Buscar o escribir** — el propio componedor (catálogo, frecuentes, texto libre con estimación). Es la opción activa.
- **Escanear con foto** — se lo pide al padre (`onScan`), que cierra el componedor y abre el escáner de siempre
  (`ScanMealModal`). Los dos padres ya lo cableaban así desde `P1-DIARY-FREETEXT-ESTIMATE`; solo cambia dónde se elige.

Sin `onScan` (un padre que no ofrece escáner) el selector no aparece y el componedor es el de siempre.

Copy: desaparecen «Foto» y «Escanear comida con la cámara» (huérfanas en los cuatro catálogos) y nacen «Buscar o escribir»
y «Cómo registrar» (grupo accesible del selector), traducidas en en-US, pt-BR, fr-FR e it-IT.

Tests: `frontend/src/__tests__/LogMealModal.freetext_estimate.test.jsx` (las dos vías dentro, un solo botón fuera) y
`backend/tests/test_p1_plan_lote_85.py` (anclas de los ficheros y de los catálogos).

## El componedor, rehecho para el teléfono (`P1-PLAN-LOTE-99` · 2026-09-18)

El dueño, con captura del teléfono: «incómodo de interactuar y entender… quiero un cambio radical». Lo que estaba mal y qué lo sustituye:

- **Dos desplegables sin etiqueta arriba del todo** («Extra (fuera del plan)» y «Hoy»): nadie sabía qué preguntaban. Ahora son dos preguntas con **chips** —«¿Qué comida es?» (Desayuno · Almuerzo · Cena · Merienda · Extra) y «¿Cuándo?» (Hoy · Ayer · Antier)— con todas las opciones a la vista y un toque. «Extra» va a secas (la etiqueta «fuera del plan» no decía nada a quien usa la app sin plan) y explica qué es cuando se elige: «un antojo o picoteo fuera de tus comidas; cuenta igual». `initialMealType` (llegar desde «Comí otra cosa») sigue eligiendo el slot del plato.
- **Modal centrado** que se quedaba corto y sin botón a la vista: en el teléfono (≤640px) es una **hoja inferior** con cabecera y pie FIJOS y cuerpo desplazable (`94dvh`, `safe-area-inset-bottom`); en escritorio, centrado a 600px con el mismo pie fijo.
- **«Lo que más registras»** era una fila con scroll horizontal y su barra visible: ahora es una lista vertical de filas con «+». Se oculta en cuanto hay plato.
- **Campos a 0,9-0,95rem**: iOS hace zoom al enfocar cualquier campo por debajo de 16px. Todos a 1rem (`P2-MOBILE-AUDIT-LOGIN-WIZARD`).
- **Cada línea del plato en una sola fila** de 300px (nombre, cantidad, unidad, gramos, papelera): ahora dos filas, el nombre arriba (hasta dos líneas) y los controles debajo.
- **«Registrar» apagado sin decir por qué**: el pie dice «Añade al menos un alimento para registrar.» y, con plato, «659 kcal · 2 en tu plato».
- **«Cancelar» sobraba**: la X, el fondo y Escape ya cierran.
- El único `<select>` que queda (la unidad de cada línea) hereda el tratamiento de `P1-LOGMEAL-SELECT-DARK` (flecha propia + `color-scheme`); el test de ese P-fix apunta ahora a `.lineUnit`. `test_p1_plan_lote_94.py` (el select que cortaba «Extra (fuera del plan)») queda superseded en esa parte.

Tests: `test_p1_plan_lote_99.py` (ancla), `LogMealModal.lote99.test.jsx` (contrato fino), y los de siempre (`LogMealModal.p1_manual_food_log`, `LogMealModal.freetext_estimate`) sin tocar: el flujo y el contrato con el backend no cambian.

### Al cerrar, el dashboard vuelve a donde estaba (`P1-PLAN-LOTE-100` · 2026-09-18)

En el iPhone, para revelar el campo enfocado iOS desplaza el DOCUMENTO de fondo aunque el body lleve `overflow: hidden`, y al cerrar la hoja la página aparecía movida «un poco hacia abajo». El componedor recuerda `window.scrollY` al abrir y lo restaura al desmontar (y cuando el visual viewport recupera su alto). El teclado NO tapaba el buscador (el dueño lo confirmó), así que no se toca la posición de la hoja. Test `test_p1_plan_lote_100.py`.

### Deslizar hacia abajo cierra; el fondo no se mueve (`P1-PLAN-LOTE-101` · 2026-09-18)

El dueño, tras el lote 100: «todavía sigue igual dando scroll hacia abajo cuando lo cierro; y quiero cerrarlo deslizando hacia abajo, como los menús de actualizar platos». Medido en el arnés con el layout real: el cierre en sí no movía la página; lo que la movía era el TOQUE dentro de la hoja cuando su cuerpo no desborda (o está en su tope): el `overflow: hidden` del body no frena el toque en iOS y `overscroll-behavior` solo actúa en elementos que sí scrollean, así que el pan pasaba a la página y el dashboard aparecía desplazado al cerrar. Mismo remedio que `MotivoActualizarModal` (P2-SWAP-SHEET-SCROLL v5): `touchmove` NO pasivo en la hoja que cancela el pan que el cuerpo no puede consumir. Y el mismo gesto (v4), sin framer: con el cuerpo arriba y el dedo bajando, la hoja sigue al dedo; al soltar cierra por distancia (>70 px), flick (>0,35 px/ms) o distancia + inercia (>100 px), o vuelve con transición; si el scroll llega arriba con el dedo aún bajando, la hoja toma el relevo. Test `test_p1_plan_lote_101.py`.

Y la causa de fondo del «baja un poquito» (el dueño: «si estoy lo más arriba, abro y cierro, y baja»): el hook `useModalAccessibility` enfoca el panel 10 ms después de abrir y el navegador «desplaza para mostrarlo»; un diálogo fijo que entra con animación desde 24 px más abajo está en ese instante parcialmente fuera del viewport → el documento de fondo bajaba ~20 px. `focus({ preventScroll: true })` en el hook (al abrir y al devolver el foco), para todos los modales; el componedor además restaura el scroll un frame después de cerrar.

## El escáner de fotos, rehecho para el teléfono y con «¿Cuándo?» (`P1-PLAN-LOTE-106` · 2026-09-18)

El dueño, con la captura de la pantalla de revisión en el iPhone: «no sé si es el tamaño o la estructura del diseño, quiero que sea más cómodo, fácil de entender, de interactuar y también agrega algo para poder seleccionar el día, ayer y antier, ya que esa es mi cena del día de ayer que no pude agregar».

**Lo que estaba mal, medido.** Era una tarjeta centrada con `max-height: 92vh` y scroll interno: en el teléfono el botón «Registrar comida» quedaba bajo el borde (en la captura se ve cortado y en dos líneas), el formulario era una columna de campos sin jerarquía (rótulos en mayúsculas, un `<select>` para el tipo, casillas diminutas para la Nevera), el nombre —un campo de TEXTO— llevaba la flecha del desplegable porque compartía regla CSS con él (el «⌄» de la captura), y no había forma de decir que la comida fue de ayer: `POST /api/diary/consumed` aceptaba `days_ago` desde P1-DIARY-EDITABLE y el escáner nunca lo mandaba.

**La hoja.** La misma que el componedor (lote 99): en el teléfono pegada abajo con asa, cabecera y pie FIJOS, cuerpo desplazable (`.card` es un panel flex de tres zonas; ≥641px sigue centrada). El gesto —deslizar para cerrar, el toque que no pasa al fondo, el scroll del fondo que vuelve a su sitio (lotes 100/101)— se **extrae** a `useBottomSheet` (`src/hooks/useBottomSheet.js`) y lo montan las dos hojas; los tests de los lotes 100/101 se reanclan al hook. Los chips excluyentes se extraen igual (`Chips.jsx`, compartido).

**La revisión como cuatro preguntas**, con la gramática del componedor: «¿Qué es?» (nombre), «¿Cuánto comiste?» (porción ½×/1×/2× a lo ancho + las cuatro macros), «¿Qué comida es?» (chips, sin `<select>`) y «¿Cuándo?» (Hoy / Ayer / Antier → `days_ago`). La cabecera dice «Revisa y registra · La IA estimó esto por la foto. Corrige lo que no cuadre». «Descontar de tu Nevera» va al final con filas de dedo (48 px, casilla de 22 px, cantidad a 1rem). El pie fijo lleva «Volver a escanear» (discreto) y «Registrar comida» (una línea, siempre a la vista). Todo campo de texto a 1rem (iOS hace zoom por debajo de 16px). Al registrar en otro día, el aviso dice dónde verlo: «Quedó en el diario de ayer; la ves en «Ver días anteriores»» — la misma regla que el coach y que el lote 105.

Tests: `test_p1_plan_lote_106.py` (ancla), `lote106.test.jsx` (hoja, preguntas, `days_ago`, aviso), `LogMealModal.lote99.test.jsx` y `P1_logmeal_select_dark.test.js` reanclados.

### La fototeca directa de la app nativa no falla en silencio (`P1-PLAN-LOTE-107` · 2026-09-18)

El dueño, dos veces tras el lote 105/106: «sigue apareciendo esto» / «sigue igual en la app instalada», con la hoja de tres opciones de iOS (Fototeca / Tomar foto / Seleccionar archivo). **Lo que tenía instalado era la PWA** (icono añadido desde Safari): su captura mostraba la interfaz desplegada esa misma noche, y el binario nativo lleva la web EMPAQUETADA (`npm run build:native` → `cap sync` copia `dist/` a `ios/App/App/public`; `codemagic.yaml` no tiene `triggering`, el build es manual) — solo cambia con un build de Codemagic. En la web ese menú lo impone Safari a todo `<input type="file">` de imágenes: `capture` abre la cámara directa, y no existe nada equivalente para la fototeca. La fototeca directa existe SOLO en nativo (`Camera.chooseFromGallery`, lote 105).

Lo que sí estaba mal en mi código: si el plugin fallaba en nativo, el escáner caía CALLADO al input de la web, que enseña esa misma hoja — indistinguible de «no se hizo nada», y cada build nativo cuesta minutos de Mac. Ahora cancelar sigue sin ser error; cualquier otro fallo se reporta (`captureException`, tags `component=ScanMealModal`, `action=native_gallery_picker`), se dice con su código en el aviso («No pudimos abrir tus fotos… [CÓDIGO]», un pantallazo basta para diagnosticar) y DESPUÉS se abre el input para no dejar al usuario sin camino.

Tests: `test_p1_plan_lote_107.py`, `lote107.test.jsx`.

## El escáner, reconstruido: la cantidad que no se dejaba borrar y varios platos a la vez (`P1-PLAN-LOTE-223` · 2026-09-24)

Un tester de Android, con captura de «Revisa y registra»: «no me deja quitar el 0 para agregar otro número… no puedo agregar cantidades por culpa del 0». El dueño, encima: «también debería poder mandarse platos múltiples como en el agente IA chat… si quieres reconstruye esto y hazlo lo mejor y más cómodo posible para el usuario».

**El 0 pegado, medido.** La cantidad de cada ingrediente era un `<input type="number">` controlado con el número del estado, y el `onChange` convertía el vacío en 0 (`Number('')`). Al borrar para escribir otra cantidad, el 0 volvía al instante y lo tecleado quedaba detrás: la captura muestra «010». Es el defecto que `MacroInput` ya había cerrado para las macros (P3-SCAN-MACRO-INPUT-EMPTY) y que nunca llegó a la cantidad. Ahora es `QuantityStepper` (`src/components/common/`): «− [campo] +», con el TEXTO y el NÚMERO separados (el campo puede quedar vacío mientras se escribe; al salir vacío vuelve la última cantidad buena), `type="text"` + `inputMode="decimal"` (la coma de los teclados en español, francés, italiano o portugués vale, como en el componedor, P1-PLAN-LOTE-166), lo escrito se selecciona al tocar (lo siguiente lo reemplaza), y los botones dan pasos con sentido para la unidad (media taza, una pieza, 10 g; `src/utils/cantidadIngrediente.js`). Viaja con punto: «0.5 taza de habichuelas», que `_parse_quantity` ya entendía.

**Las macros siguen a los ingredientes.** Desmarcar las albóndigas no cambiaba las calorías: el análisis traía el total de la foto y una lista de componentes sin lo que aporta cada uno. El prompt YA pedía estimar cada componente por separado y sumar (P1-MEAL-SCAN-DR-DISHES); ahora cada entrada de `items` trae sus cuatro cifras, y `_coerce_meal_scan` reparte los totales YA corregidos (el ajuste 4P+4C+9G y el tope de un plato) en esas proporciones (`_repartir_macros_del_plato`), así que la suma de los componentes es exactamente lo que se precarga. Sin desglose usable (un modelo que no lo dé, un servidor anterior) los componentes van sin `macros` y el modal hace lo de siempre: el total de la foto por la porción. En plato, `_sane_item_qty(…, plato=True)` admite la media porción (½ taza, medio aguacate) y deja de convertir quince lascas en una: la regla «más de 12 es un peso impreso mal leído» es de la compra, no de un plato servido. Mismo helper con un modo, no una segunda ruta de saneado. En el modal (`scanMealDishes.js`, pura): una corrección a mano de una macro se guarda como DIFERENCIA con lo derivado, así sobrevive a desmarcar un ingrediente después; una porción (½×, 1×, **1½×**, 2×) reescala las cantidades desde lo detectado y descarta lo corregido a mano, como siempre.

**«Descontar de mi Nevera» es un interruptor propio**, el mismo del componedor. Desmarcar un ingrediente ahora dice «no lo comí» (y le quita sus calorías), así que ya no puede decir también «no salió de mi Nevera» (el arroz del restaurante). `POST /api/diary/consumed` acepta `deduct_pantry`: solo un `false` explícito apaga la resta (ausente = la conducta de siempre, para clientes anteriores); con `false` los ingredientes se GUARDAN igual y quedan marcados como sincronizados, como en `/consumed/manual`.

**Varios platos**, como en el chat: hasta 4 fotos por registro (`MAX_PLATOS`, el mismo tope que `CHAT_IMAGE_MAX_COUNT`). La galería elige varias (`multiple` en la web; en nativo `chooseNativeGalleryImages`, alias del selector del chat, con el hueco que queda), y «¿Comiste algo más?» añade otra con la cámara o la galería sin salir de la revisión. Cada foto se analiza por su cuenta y en paralelo; el visor de la cámara se cierra al disparar (antes esperaba dentro a que la IA terminara), así que se puede fotografiar el siguiente plato mientras se analiza el anterior. Una foto que falla no tumba a las demás: su tarjeta dice por qué y ofrece «Reintentar» (análisis caído, red, 429) o quitarla (compra, no es comida). Con varios platos cada uno es una tarjeta resumida (foto, nombre, kcal) que se abre para editar; «¿Qué comida es?» y «¿Cuándo?» valen para todo el registro (es UNA comida con varios platos) y cada plato va a su propia fila del diario. El registro es en serie: si uno falla, los anteriores quedan marcados «Registrado» y el reintento no los repite. Dos platos con el mismo nombre se numeran («Jugo de chinola (2)»): el anti doble toque del servidor (P2-CONSUMED-DEDUP: mismo nombre y tipo en 60 s) se habría comido el segundo.

**Lo que se vio en el camino.** A 360 px el pie no cabía en español: «Volver a escanear» y «Registrar comida» en una fila cortaban el botón verde (también antes de este lote). «Volver a escanear» se mudó a la foto (una píldora sobre ella) y el pie queda con la acción principal sola; con varios platos, encima, el total de lo que se va a registrar y cuántas fotos no se registrarán. La unidad de cada ingrediente se pinta con su número y en plural («2 tazas», «10 unidades»).

Tests: `test_p1_plan_lote_223.py` (reparto de macros, saneado en plato, `deduct_pantry`, anclas del frontend), `lote223.test.jsx` (el 0, los pasos, las macros que siguen a los ingredientes, varios platos, fallos a medias); reanclados `ScanMealModal.photo_deducts.test.jsx`, `lote107.test.jsx`, `lote110.test.jsx`, `lote162.test.jsx`, `lote165.test.jsx` y `test_p1_plan_lote_105.py`.
