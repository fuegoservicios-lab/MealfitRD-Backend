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
