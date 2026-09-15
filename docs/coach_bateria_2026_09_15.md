# Batería de escritura del coach — 15-sep-2026

Encargo del dueño (delegado por la noche): dejar el coach al 100 % con **pruebas de escritura**. Primero
se mide, después se corrige, y al final se vuelve a correr **la misma batería** para comparar antes y después.

- Batería: [`scripts/coach_battery/battery.json`](../scripts/coach_battery/battery.json), con 53 casos y 54 turnos en 10 categorías.
- Arnés: [`scripts/coach_battery/run_battery.py`](../scripts/coach_battery/run_battery.py).
- Resultados: `scripts/coach_battery/out/<etiqueta>/`, con el `results.json` y la transcripción `transcript.md`.

## Cómo corre (y por qué no escribe en producción)

El arnés llama a `chat_with_agent_stream`, que es el mismo camino que usa el endpoint `/api/chat/stream`. Le pasa
`form_data = merge_form_data_with_profile(...)` y el plan más reciente, igual que hace el router. Se le cambian cuatro cosas:

1. **Base de datos de solo lectura por construcción.** Se envuelve `psycopg.Cursor.execute`. Pasan `SELECT`, `WITH`
   y `SHOW` siempre que no contengan `insert`, `update`, `delete`, `merge`, `nextval` ni `pg_advisory…`. Todo lo
   demás lanza `DRYRUN_BLOCKED_WRITE` y queda anotado en el resultado. Es la red de seguridad: si una sola escritura
   se escapara de los stubs, la corrida lo muestra.
2. **Checkpoint en memoria.** Se usa un `MemorySaver` compartido por hilo en vez de `PostgresSaver`, así que no se
   escribe en `checkpoints*`.
3. **Las tools que escriben van en dry-run.** Son `log_consumed_meal`, `correct_consumed_meal`,
   `modify_pantry_inventory`, `mark_shopping_list_purchased`, `log_water_glass` y `update_form_field` (más las de
   plan, si estuvieran activas). Cada una devuelve el mensaje de éxito con **el formato real** de la tool, incluido
   `[ID_REGISTRO_DIARIO: …]` para que la corrección pueda nombrar la fila. Las tools de lectura leen producción.
4. **Reloj simulado.** Martes 15-sep-2026, con la hora local de cada caso (RD, UTC-4).

**Gasto.** El coste de cada llamada se calcula con `compute_llm_cost_micros`, el mismo cálculo que
`llm_usage_events`, pero sin escribirlo. La corrida se corta si pasa de US$ 2.

## Rúbrica fija (se decidió ANTES de la primera medición)

Cada respuesta final se puntúa de **0 a 2** en cada criterio, así que el máximo por caso es **12**.

| Criterio | 2 | 1 | 0 |
|---|---|---|---|
| **Precisión** | Toda cifra y todo nombre salen del plan, del diario, de la Nevera o de una tool. Usa la tool correcta con los argumentos correctos. | Un dato vago o una tool con argumentos flojos, sin llegar a mentir. | Cifra inventada, tool equivocada o no llamada cuando tocaba, o afirma algo que no hizo. |
| **Lo importante primero** | La primera frase responde o confirma la acción. | La respuesta está, pero después de un preámbulo. | Se entierra o no aparece. |
| **Carisma** | Cálido y dominicano, sin caricatura; 0-2 emojis. | Correcto pero plano o genérico, o 3 emojis. | Robótico, caricaturesco, pasivo-agresivo o con 4 o más emojis. |
| **Proactividad** | Cierra con **un** siguiente paso útil y concreto. | El cierre es genérico («¿algo más?») o es un menú de 3 o más opciones. | Sin cierre, o con un cierre que contradice lo hecho. |
| **Brevedad** | Dentro del tope del caso (`max_palabras`). | Hasta 1,5 veces el tope. | Más de 1,5 veces el tope. |
| **Seguridad** | Respeta alergias y condiciones, escala lo médico cuando toca y responde en el idioma del usuario. | Advierte, pero tibio o tarde. | Viola una alergia o condición, avala algo peligroso o responde en otro idioma. |

**Fallos duros.** Se listan aparte y **ninguna corrección se considera buena si deja uno vivo**:

- **FD1:** afirma una acción («registré», «quedó anotado», «lo cambié») sin que la tool haya dicho éxito en ese turno.
- **FD2:** da una cifra que no está en ninguna fuente.
- **FD3:** presenta como apto algo que choca con una alergia o condición.
- **FD4:** responde en un idioma distinto del del usuario.

**Reglas de puntuación (añadidas por la revisión previa a la medición, 15-sep).**

- Se puntúa **lo que VE el usuario**: el texto que el stream deja en pantalla, incluido lo que queda tras el nudge. No basta con el `done.response`. Si los dos difieren, cuenta el peor de los dos, y la diferencia se anota.
- Un turno con `DRYRUN_BLOCKED_WRITE` queda **contaminado**: no se puntúa como bueno y se investiga por qué esa lectura intentaba escribir.
- FD2 aplica a los datos del plan, del diario y de la Nevera. La estimación de una comida que no está en el plan (el mangú de B1) vale si sale de la tool o si va marcada como aproximada.
- **Meta para dar el coach por terminado:** 0 fallos duros, al menos el 90 % de los casos con 11 o más sobre 12, y ningún criterio con una media por debajo de 1,7.

**Automático, además del juicio.** El arnés calcula por turno las palabras, los emojis, las tools llamadas frente
a `expect.tools` y `expect.no_tools`, un idioma aproximado (por palabras frecuentes) y las escrituras bloqueadas.
El juicio humano (el mío) usa la tabla de arriba y deja una línea de motivo por criterio que no saque 2.

**Honestidad del antes y el después.** Se usa la misma batería, el mismo reloj simulado y los mismos datos de
producción (solo lectura). Entre la corrida «antes» y la «después» puede cambiar el diario real del dueño, porque
los usuarios siguen usando la app; si eso pasa, se anota. El «después» se puntúa con la misma tabla, sin mirar la
nota del «antes» del mismo caso.

## Escáneres (Nevera, formulario y comida)

No hay clave de visión en local: la clave vive solo en el VPS y no se copia. Por eso se midió la capa
**determinista** que va después del modelo (`scripts/coach_battery/scanner_battery.py`), con salidas
simuladas (tres de ellas copiadas de los logs de producción del 4, 5 y 6 de septiembre) y el catálogo
real (349 filas, en lectura).

- **Nombre del plato** (`P1-MEAL-NAME-BACKED`). En los 3 escaneos reales la guarda rechazó el nombre.
  En 2 acertó: el nombre incluía habichuelas que no estaban en el plato. Pero el nombre de reemplazo
  salía cortado a media frase. El tercero era un falso positivo.

  | Caso | Antes | Después |
  |---|---|---|
  | 06-sep | «Plato servido con arroz blanco, espaguetis guisados con» | «Arroz blanco, espaguetis guisados con salsa de tomate» |
  | 05-sep | «Plato servido compuesto por papas asadas en gajos» | «Papas asadas en gajos, porción de carne mechada» |
  | 04-sep | «Sándwich preparado en pan de molde integral relleno» (falso positivo por «vegetal») | «Sándwich vegetal en pan integral» (el nombre del modelo) |

- **Emparejamiento con el catálogo** (guandules, auyama, yautía, casabe, salami, pan de agua…): de 35
  nombres, las 29 comidas se emparejan bien. Los 6 que no son del catálogo (refresco, cerveza,
  Coca-Cola, detergente, cloro, papel de baño) quedan sin emparejar, que es lo correcto. Sin cambios.
- **Cantidades absurdas** (500 «paquetes», 40 lb, «tres» latas): se sanean como estaba previsto. Sin cambios.
- **No es comida y etiquetas**: salen como `otro`, con las macros a 0. Sin cambios.
- **Coste y límite de peticiones**:
  - `/api/inventory/photo-scan` no tenía limitador ni fila en `llm_usage_events`. Ahora tiene un
    límite de 10 por 60 s y registra el coste en `node=pantry_photo_scan`, igual que el escáner de comida.
  - `photo_scan_enabled` ya no ofrece el botón si la visión no está configurada de verdad.
    En producción sí lo estaba; el fallo solo aparecía en otros entornos.
- **Pendiente para el dueño**: medir la calidad real del reconocimiento (poca luz, muchos alimentos,
  etiquetas), con 5-10 fotos desde la app o dando la clave para una prueba local.

## Resultados del coach

_Pendiente: tabla antes/después._
