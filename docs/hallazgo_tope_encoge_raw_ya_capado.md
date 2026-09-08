# El tope de realismo vuelve a encoger un `raw` que ya estaba capado

**Fecha:** 2026-09-08 · **Estado:** VERIFICADO, sin arreglar · **Prioridad:** alta — y mi arreglo del 07-sep puede AMPLIFICARLO

## Lo que se ve en producción

Seis comidas vivas con lechosa comparten la misma firma:

```
LEE:    ½ lechosa mediana madura (405g)
COMPRA: 0.37 lechosa mediana madura (300g)
```

300 g es exactamente `REALISM_FRUIT_VOLUME_CAP_G`. O sea: **el tope llegó a la compra y no a la
receta.** El usuario lee 405 g y compra 300: se queda corto 105 g cada vez. Mismo patrón con
395→300, 385→300, 375→298.

## Lo que pasa si el tope vuelve a correr sobre ese estado

Reproducido ejecutando `_cap_unrealistic_portions` sobre esas comidas tal como están hoy:

| | LEE | COMPRA |
|---|---|---|
| antes | ½ lechosa (405 g) | 0.37 lechosa (**300 g**) |
| después | 0.37 lechosa (300 g) | 0.27 lechosa (**222 g**) |

La receta se corrige, **y la compra encoge otra vez**. El factor se calcula desde los gramos del
DISPLAY (405 → 300/405 = 0,74) y se aplica multiplicando sobre la línea de raw, que **ya estaba en
300**: 300 × 0,74 = 222.

## No es un fallo de idempotencia general

Sobre una comida SANA (display y raw idénticos, fruta sobre el techo) el tope es idempotente:
pasada 1 capa las dos a 300, pasadas 2-4 no hacen nada. El fallo es específico de **reparar sobre un
estado ya divergente** — que es justo el estado de los seis platos vivos.

## Por qué esto me toca a mí

Hasta el 07-sep el tope resolvía la línea de raw **por índice**; `P1-CAP-BIGFRUIT-BREAD-RAW-BY-FOOD`
la pasó a resolver **por alimento**. Ese cambio es correcto y cierra un fallo medido — pero tiene un
efecto que hay que decir en voz alta: antes, un índice equivocado a menudo no casaba nada y el
segundo recorte **no ocurría**; ahora la búsqueda acierta la línea de la lechosa y el recorte
compuesto **sí ocurre, de forma fiable**.

*Un arreglo que hace que la escritura llegue a su destino también hace que llegue la escritura
equivocada.* No invalida el arreglo — invalida dejar este defecto sin cerrar.

## La forma del arreglo (no aplicado)

El tope debe llevar la línea de raw a un **objetivo absoluto**, no multiplicarla por el factor del
display: si los gramos de raw ya están en o por debajo del techo, no se toca. Con eso, reparar un
estado divergente converge en vez de componer.

**No lo apliqué**: `graph_orchestrator.py` está a 2 líneas de su techo, el cambio vive dentro de una
cascada de ~15 ramas de cap con `min()` entre ellas, y tocarlo pide una pasada de gate propia con
más margen de contexto del que tenía. Prefiero dejarlo verificado y reproducible antes que a medias.

## Reproducción

```python
import graph_orchestrator as go
dia = [{"day": 1, "meals": [{
    "meal": "Merienda", "name": "Lechosa fresca",
    "ingredients":     ["½ lechosa mediana madura (405g)", "½ taza de yogurt"],
    "ingredients_raw": ["0.37 lechosa mediana madura (300g)", "½ taza de yogurt"],
}]}]
go._cap_unrealistic_portions(dia)   # raw pasa de 300 g a 222 g
```
