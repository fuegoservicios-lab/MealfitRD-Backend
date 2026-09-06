# I20 · El dato ausente no es cero, ni el default

> Entrega parcial de los contratos I20–I27 del roadmap 2.6. Este es el que se ganó el sitio con
> evidencia: **tres incidentes en un solo día**, todos de la misma forma. Los otros siete siguen
> pendientes y no se escriben aquí para no inventar contratos sin casos que los respalden.

## El contrato

Cuando un valor puede faltar, **«falta» tiene que ser distinguible de cualquier valor legítimo**
— y muy en particular de `0`, de `""`, de `[]` y del default que el código usaría si el dato
estuviera. Un `or` que colapsa las dos cosas convierte una ausencia en una afirmación.

```python
_at = int(q.get("attempts") or -1)          # ❌ attempts=0 se vuelve -1
_at_raw = q.get("attempts")                 # ✅ ausente y cero son distintos
_at = int(_at_raw) if _at_raw is not None else -1
```

La forma peligrosa no es sólo `or`: es cualquier paso en el que **el sistema deja de saber que no
sabía**. Un estado que ninguna fila tiene, una unidad que el mapa no reconoce, un contador en
cero — los tres se leen igual que un dato real y por eso nadie los ve.

## Los tres casos que lo respaldan (2026-09-05 / 06)

| Caso | La forma | Lo que costó |
|---|---|---|
| `P0-FILL-FENCED` | `int(x or -1)` sobre `attempts` | El fencing rechazaba **toda primera escritura** (`attempts=0`). Vivo 40 min en producción. Los 8 tests existentes pasaban porque todos usaban `attempts=3`. |
| `P1-QUALITY-SWEEP-STATUS` | `generation_status = 'complete'` | Un barrido que **no podía alcanzar su condición**: cero planes en ese estado en toda la base. Corría cada tick, resolvía 0 y nadie sospechaba, porque «0 resueltos» es lo que se espera de un barrido sano. |
| `P1-UNKNOWN-UNIT-NOT-WHOLE` | `canonicalize_unit(unit) or "unidad"` | Una unidad **desconocida** se volvía «una unidad entera del alimento», y para una hierba eso es el mazo de compra: «5 tallos de cebollín» = 250 g. El cap de compras recortaba 2.043 veces al día tapándolo. |

Los tres comparten la firma: **el mecanismo funcionaba, corría y no se quejaba.** Lo que estaba
mal era el dato que esperaba encontrar. Por eso ninguno lo cazó un test —todos los tests le daban
el dato que sí existía— y los tres se encontraron **midiendo producción**, no leyendo código.

## Cómo se aplica

1. **Al leer un campo que puede faltar**, separa la lectura de la conversión:
   `x if x is not None else default`, nunca `x or default` — salvo que el default y el cero sean
   intercambiables *a sabiendas*, y entonces escríbelo en un comentario.
2. **Al comparar contra un valor enumerado** (`= 'complete'`, `in ('pending',...)`), pregunta
   cuántas filas lo tienen **hoy en producción**. Un predicado que no puede casar es indistinguible
   de uno que no tiene trabajo.
3. **Al resolver contra un mapa** (unidades, alias, países), decide qué significa «no está»
   *antes* de escribir el `or`. Casi siempre es «no lo sé» y no «el primero de la lista».
4. **Al medir el efecto de un arreglo**, hazlo antes de desplegarlo. Una medición posterior al
   efecto no mide el efecto — costó un «cierra 0» que en realidad había cerrado 10.

## Anclas

- [`test_p0_fill_fenced.py`](../tests/test_p0_fill_fenced.py) · el caso `attempts=0` explícito.
- [`test_p1_quality_sweep_status.py`](../tests/test_p1_quality_sweep_status.py) · cross-link entre
  los dos sitios que deben nombrar los mismos estados terminales.
- [`test_p1_unknown_unit_not_whole.py`](../tests/test_p1_unknown_unit_not_whole.py) · «sin unidad»
  y «unidad que no conozco» probados por separado, que es la distinción entera.
- [`test_p1_invariante_dato_ausente.py`](../tests/test_p1_invariante_dato_ausente.py) · ancla este
  documento a los tres casos: si uno se revierte, falla aquí además de en su propio test.
