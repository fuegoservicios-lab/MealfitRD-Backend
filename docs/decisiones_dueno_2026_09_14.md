# Decisiones del dueño · 2026-09-14 (delegadas al agente)

El dueño llenó la anotación con rúbrica de los 80 casos del golden set (Hoja del dueño, 2026-09-13) y el 14-sep delegó las
diez decisiones abiertas, las tres recetas y el caso sin defecto listado: *«elige tú lo que mejor consideres y hazlo»*. Este
documento es el registro de lo elegido y por qué, y lo que cada decisión dispara. La hoja (documento `angelo`) quedó
escrita con lo mismo. Fuente de las opciones: `plan_pendientes_2026_09_11.md` y la hoja.

## Las diez decisiones

| Clave | Opción | Razón y alcance |
|---|---|---|
| juez | `si` | Sí, con tope de $1. Antes hay que arreglar el adjudicador y refrescar las columnas de la máquina (lote 38); calibrar después, nunca sobre el instrumento roto. |
| e5b_cohorte | `yo` | Solo el usuario del dueño, como canario; la cohorte nace igual al canario del día determinista. |
| e5b_topes | `heredar` | La canónica hereda los topes del agregador tal cual; se revisan con 30 planes. |
| claras | `umbral` | N = 4: botella de claras pasteurizadas cuando una comida pide 4 claras o más; por debajo, cartones y el usuario separa. |
| v7a | `residuo` | Queda en warn; coherente con la decisión previa de no implementar «el paso pide MENOS». |
| b7 | `canario` | MEALFIT_DETERMINISTIC_DAY_W_CARB_SURPLUS = 2.0 y _W_FAT_DEFICIT = 1.0 solo en el canario del dueño; global sigue 1.0/1.0. |
| equipo | `dejar` | Se queda solo en Súper Personalización; el wizard no gana un paso. |
| d8 | `cerrar` | No construir shopping_commercial: precio y envase ya salen de supermarket_products. |
| g1 | `block` | MEALFIT_BILLING_VERIFY_AMOUNT = block: sin cupón activo, un importe por debajo del plan se bloquea. |
| f9 | `tengo` | Nutricionista: Juan Carlos Brito. Fecha de la revisión por confirmar. |

Lo que disparan: **lote 44** del [plan 38-43](plan_agente_lotes_38_43_2026_09_14.md) (recetas a la biblioteca, claras en
botella a partir de 4, cohorte B de la lista canónica = canario del dueño, pesos 2.0/1.0 solo en el canario, `block` en el
importe de PayPal); la calibración del juez espera al lote 38 (instrumento); D8 se cierra sin código; V7a y el equipo no
cambian nada.

## Las tres recetas (redacción propuesta por el agente y aceptada por delegación)

- **Yaniqueques** (`tpl_14c76a1c346e`): el aceite en tres tercios — masa (paso 1), bañar antes del horno (paso 2), sofrito de
  la cebolla (paso 3). La sal ya iba en dos mitades (pasos 1 y 4).
- **Pollo al horno con batata** (`tpl_a7799418aa9c`): la mitad del aceite al pollo y un cuarto a la batata (paso 2), el
  último cuarto a la ensalada (paso 4). El limón ya iba en dos mitades.
- **Lentejas con auyama y batata** (`tpl_fc758e30f5e7`): la auyama se pela y se corta con la batata (paso 1) y entra con
  ella en el paso 4, los 15 minutos finales; con las lentejas (20 min + 15) se desharía.

Texto íntegro de los pasos: en la hoja (`recetas.<tpl>.pasos`, un paso por línea). Aplicación: lote 44 (`recipe_library_do_v1.json`
con `procedencia` «dueño 2026-09-14, redacción del agente por delegación», `asignar_uso_pasos.py --write --verificar`).

## El caso `060b4fda6a`

«Tostadas al horno con huevo, habichuelas rojas y aguacate» (desayuno, día 9): la lista trae 50 g de habichuelas rojas
SECAS y los pasos las tratan como cocidas. La nota del dueño del 2026-09-07 ya lo decía; el 14-sep quedó en la rúbrica como
`seco_sin_coccion` · high · «Habichuelas rojas», transcrito por el agente. Está en `culinary_golden_anotaciones_angelo.json`.

## Qué tan lejos del 100 % (con la anotación del dueño)

El golden set es una muestra ESTRATIFICADA de 1.186 comidas de 96 planes (2026-09-06): sobre-representa lo que la máquina
marcó. Reponderando cada estrato por las comidas disponibles, **73 % de las comidas del corpus tenían al menos un
defecto según el dueño** (dudosas 8 %). Con un solo anotador, cifra orientativa.

| Estrato | Comidas en el corpus | Anotadas | Defecto | Dudoso | Ok | Defecto |
|---|---|---|---|---|---|---|
| `solo_determinista` | 110 | 20 | 17 | 1 | 2 | 85 % |
| `solo_juez` | 134 | 20 | 18 | 2 | 0 | 90 % |
| `ambas` | 23 | 15 | 14 | 0 | 1 | 93 % |
| `sin_hallazgo` | 919 | 25 | 17 | 2 | 6 | 68 % |

Lectura: el estrato «sin hallazgo» (lo que la máquina da por bueno) es el que fija la cifra, y ahí el dueño también
encuentra defectos — las clases que ninguna capa mecaniza (`coccion_faltante`) o que ven sin acertar (`seco_sin_coccion`).
Por eso el orden del plan es instrumento (38) → decisiones (44) → detectores (39) → juez (40).

## Para el dueño (sigue siendo suyo)

Las cuatro tareas de GitHub (`SIBLING_REPO_TOKEN`, los 4 secretos del benchmark nocturno, las ramas superadas, el 2.º
anotador), la fecha de la revisión de Juan Carlos Brito, y la re-lectura de estas decisiones si alguna no le convence:
revertir una es cambiar un knob, no rehacer trabajo.
