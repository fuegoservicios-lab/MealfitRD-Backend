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
