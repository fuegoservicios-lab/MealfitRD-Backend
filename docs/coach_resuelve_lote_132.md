# El coach resuelve lo que el usuario necesita (P1-PLAN-LOTE-132 · 2026-09-20)

Encargo del dueño: «al usuario le falta proteína y calorías y son las 9 de la noche, y le dice al agente "tengo proteína en
mi casa": el agente debe pedir pruebas para registrar, y si lo ve mal por la hora, decirlo… que pueda decirle "dame una
receta para comer hoy" o "dame una comida para el desayuno" y resuelva de acuerdo a lo que el usuario necesita».

## Las cuatro piezas

| Pieza | Dónde | Qué hace |
|---|---|---|
| Bloque «LO QUE LE FALTA HOY» | `coach_day_context.build_day_gap_context` → `agent._day_gap_context_for_chat` (los DOS paths) | La resta hecha (kcal y gramos que faltan o sobran), la hora, y cómo se cierra un día a ESA hora. Sin metas reales (invitado sin formulario) o sin diario legible ⇒ `""`. |
| Tool `proponer_comida` | `tools.py` + `coach_day_context.proponer_comidas` | Hasta 3 platos del Dish Registry escalados a lo que falta, con gramos, macros de la tabla de alimentos, tiempo, pasos de la biblioteca y cobertura de la Nevera. Solo lectura. |
| Reglas Q-T | `prompts/chat_agent.py::_CHAT_RESOLVE_RULES` (en las 4 constantes) | Q comida a pedido · R usar las cifras y trabajar con lo que TIENE · S prueba antes de anotar un producto de proteína de envase · T la hora sin mitos |
| Visión `etiqueta` | `vision_agent.py` + `build_vision_context` | La tabla nutricional se LEE (valores por porción), no se estima. `is_food=True`, `items=[]`: en el escáner del Dashboard cae al flujo de plato y precarga UNA porción. |

## Decisiones que no son obvias

- **«No puedes tomar proteína tan tarde» es un mito y el coach no lo repite.** De noche lo que pesa es lo pesado/frito,
  la cafeína (pre-entrenos) y el líquido justo antes de acostarse. La guía de noche solo se emite con reloj «clásico»
  (`momento_del_dia` devuelve `turno_nocturno` y se calla) y las condiciones médicas mandan sobre ella.
- **Un diario casi vacío de tarde/noche = falta REGISTRAR.** Con menos del 35 % de la meta anotado, el bloque manda
  preguntar qué comió antes de recomendar comer más. Por lo mismo, con el diario VACÍO la propuesta usa la ración normal
  de la franja (`MEAL_SLOT_SPLITS`) en vez de repartir «lo que falta».
- **La regla S es la ÚNICA excepción a «pasado = registra sin preguntar»**, y solo para proteína en polvo, ganadores,
  barras y batidas listas. Si el usuario no puede dar la prueba, se anota un genérico marcado como aproximado.
- **La receta se adjunta ANTES de verificar**: V3 acusa al ingrediente que ningún paso menciona. Sin biblioteca (fuera de
  DO) se apartan V3/V5/V6 y el backstop clínico corre igual.
- **Se verifica solo lo que se devuelve** (0,3 s por plato): de 4-13 s a ~1,5 s por llamada.
- **Prefiere, no descarta**: los ingredientes que faltan en la Nevera (`PENA_POR_INGREDIENTE_QUE_FALTA`) y los atributos
  de riesgo de la condición declarada (`_RIESGOS_POR_CONDICION`: diabetes → carga glucémica, HTA → sodio…) mandan el
  plato hacia atrás, no fuera.
- Los pasos se leen de `recipe_library._library` directamente: `recipe_for_dish_name` va detrás del knob de GENERACIÓN
  (`MEALFIT_RECIPE_LIBRARY_SELECT`) y esta superficie tiene el suyo.

## Knobs

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_CHAT_DAY_GAP_BLOCK` | `True` | Apaga el bloque «LO QUE LE FALTA HOY» |
| `MEALFIT_CHAT_MEAL_PROPOSAL_TOOL` | `True` | Retira `proponer_comida` de `agent_tools` y cambia su bullet del prompt |

## Cómo se prueba

- Sin IA (cero gasto): `scripts/coach_battery/propuesta_smoke.py` — bloque + herramienta contra los datos reales de la
  cuenta de la batería, en solo lectura. Cazó tres fallos antes del primer céntimo (ver el test).
- Con IA (dirigida, ~US$0,002 por caso con DeepSeek Flash): casos `R1`-`R11` de `battery.json` (`--only R1,R2,…`). El
  arnés ganó `diario` por caso (día a medias simulado) y el reloj simulado llega también a las tools.
- Anclas: `tests/test_p1_plan_lote_132.py`.
