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

### Lo intenté y lo REVERTÍ — el arreglo obvio empeora las cosas

Implementé exactamente eso: un helper `_cap_raw_factor` que rederiva el factor contra los gramos de
raw (`min(1.0, cur_g*factor/g_raw)`) y que además distingue «no encontré el alimento» de «lo
encontré y no hay nada que recortar» —esa confusión era lo que disparaba el fallback por índice—.

En los tres casos sintéticos funcionaba perfecto: el divergente convergía a 300 en vez de caer a
222, el sano seguía idempotente, y un raw de 600 g bajaba a 300 exactos.

**Contra la flota entera, no.** Comparando con y sin el helper sobre las 1.194 comidas vivas:

| | líneas tocadas | recortes brutales (>80 %) |
|---|---|---|
| conducta previa | 48 | **1** |
| con mi arreglo | 36 | **14** |

Con ejemplos como `588 g → 12 g`, `612 g → 7 g`, `1599 g → 4 g`. El objetivo absoluto se calcula
desde UNA línea del display y se aplica a una línea de raw que puede ser otra cosa del mismo
alimento; cuando la del display ya venía recortada por otra rama de la cascada, el objetivo sale
minúsculo y arrasa la compra.

*Los tres casos sintéticos que diseñé pasaban los tres. La flota dijo que no.* Revertido — el
árbol queda como estaba.

### Una afirmación mía que RETIRO: «las ramas de conteo no sincronizan raw»

Escribí eso como «verificado en aislado» —«3 calabacín» pasaba a «1.5 calabacín» en la receta, raw
se quedaba en 3, `recortes=0`— y **es falso**. Medí con mi propio parche roto todavía activo en el
árbol: era él quien dejaba raw sin tocar, al convertir el rescale en no-op y saltarse después el
fallback que yo mismo había gateado.

Con el árbol revertido, ese mismo caso da `recortes=1` y raw pasa a «1.5 calabacín», correcto. Y
sobre la flota: el tope capa el display en **76 comidas (6,4 %)** y en **todas** cambia también la
línea de raw del mismo alimento — **0 huérfanas**.

*Medir mientras tu propio parche defectuoso está vivo es medir el parche, no el sistema.* Es la
misma familia de error que el resto de la sesión: el instrumento no estaba en el estado que yo creía.

### Lo que sí hace falta

El objetivo tiene que derivarse de **la línea de raw que se va a escribir**, no de la del display —
que es justo lo que mi intento hacía mal: tomaba los gramos de una línea del display que podía ser
otra cosa del mismo alimento. Sigue abierto, y sin la pista falsa de arriba.

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
