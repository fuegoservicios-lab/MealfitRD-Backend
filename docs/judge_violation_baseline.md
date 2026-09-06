# Línea base del juez culinario · 06-sep-2026

Capturada **antes** de desplegar los arreglos del 6 de septiembre, con
[`scripts/judge_violation_rate.py`](../scripts/judge_violation_rate.py). Comparar contra ella es
todo el punto: una medición tomada después del efecto no mide el efecto — lección que costó un
«cierra 0» que en realidad había cerrado 10.

```
python scripts/judge_violation_rate.py --dias 8
```

## Violaciones por plan, por fecha de creación

La **fecha de creación del plan es la unidad**, no la ventana. Leer un histórico de 7 días como si
fuera el estado de hoy es lo que me hizo recomendar atacar `combo_absurdo`, que ya estaba cerrado.

| fecha | planes | combo | nombre_no_corr | paso_incoh | slot_inaprop | técnica | total | /plan |
|---|---|---|---|---|---|---|---|---|
| 2026-08-31 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0,00 |
| 2026-09-02 | 30 | 3 | 13 | 23 | 10 | 5 | 54 | 1,80 |
| 2026-09-03 | 12 | 1 | 3 | 15 | 0 | 1 | 20 | 1,67 |
| 2026-09-04 | 5 | 0 | 3 | 5 | 1 | 2 | 11 | 2,20 |
| 2026-09-05 | 19 | 3 | 19 | 16 | 2 | 7 | 47 | 2,47 |
| **2026-09-06** | 2 | 0 | 2 | 6 | 0 | 0 | **8** | **4,00** |

⚠️ El 06-sep son **2 planes**: 4,00 por plan es una muestra demasiado pequeña para leerla como
tendencia. Lo que sí es sólido es que del 02 al 05 la tasa **no bajó** (1,80 → 2,47) mientras se
cerraban gaps — porque los que se cerraban no eran los que dominaban el conteo.

## Las tres predicciones

Cada arreglo del 06-sep predice que una columna tiende a cero. Si no lo hacen, la causa estaba en
otro sitio y hay que volver a buscar.

| fecha | líneas de ingrediente | hierbas > 150 g | hints en contra de su línea |
|---|---|---|---|
| 2026-09-02 | 3.413 | 0 | 1 |
| 2026-09-03 | 1.279 | 0 | 1 |
| 2026-09-04 | 756 | 0 | 1 |
| 2026-09-05 | 2.937 | 1 | 7 |
| 2026-09-06 | 235 | 0 | 2 |

- **hierbas > 150 g** mide `P1-UNKNOWN-UNIT-NOT-WHOLE`. La columna ya es casi cero porque la sonda
  cuenta la línea *individual*; el daño estaba en el **agregado** (568 g de cilantro, 415 g de
  cebollín tras sumar el plan entero) y en los 2.043 recortes diarios del cap. La señal real de
  este arreglo es el journal: `[P3-HERB-CAP]` y `[P5-VEG-CAP]` deben caer.
- **hints en contra** mide `P1-STEP-GRAM-HINT-STALE`. Cuenta solo la forma `N g de X (M g)` —un
  subconjunto de las 45 menciones medidas el 06-sep— y por eso los números son pequeños; lo que
  importa es que sea **la misma** cuenta antes y después.
- `nombre_no_corresponde` y `tecnica_impropia` se leen directamente de la tabla de arriba.

## Qué esperar mañana

Los arreglos se desplegaron el 06-sep a las ~17:00. **Todos los planes de la tabla son
anteriores**, así que la primera fila comparable es la del 07-sep. Si `paso_incoherente` no baja
—era 6 de las 8 violaciones del 06-sep— el arreglo del hint no era la causa dominante y hay que
volver al juez con casos nuevos, no con suposiciones.
