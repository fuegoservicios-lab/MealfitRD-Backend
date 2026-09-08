# El caso 2 del eje display↔raw, medido: 81 comidas (6,8 %)

**Fecha:** 2026-09-08 · **Estado:** medido; la definición es decisión de producto

`P2-EJE-CASO3-IDENTIDAD` dejó el eje partido en tres y midió el inequívoco en **0**. Esto mide el
ambiguo, para que la decisión no sea abstracta.

## Dos mediciones que discrepaban

| medición | `missing_in_raw` vivo |
|---|---|
| `_misalign_fingerprint` (tracer, claves de alimento EXACTAS) | **81** |
| sonda del caso 3 (tokens canónicos con contención) | **0** |

*Cuando dos mediciones honestas discrepan, la diferencia ES el hallazgo.* Reconciliadas una a una:
**81 de 81** de los «faltantes» del tracer tienen pareja por tokens en la compra. Ninguno es «el
ingrediente no se compra nunca». Los 81 son **el mismo alimento nombrado distinto**.

## De qué está hecho ese 6,8 %

| par | veces | ¿es caso 2 de verdad? |
|---|---|---|
| `batatas ~ batata` | 11 | no — plural |
| `filete de pescado ~ filete pescado blanco` | 11 | sí — especificidad |
| `papas ~ papa` | 10 | no — plural |
| `pechuga ~ pechuga pollo` | 6 | sí — especificidad |
| `queso blanco fresco ~ queso blanco` | 5 | sí — especificidad |
| `limon ~ limon` | 4 | **no — idénticos** |
| `zanahorias ~ zanahoria` | 3 | no — plural |
| `queso cheddar ~ queso cheddar` | 2 | no — orden de palabras |

Cerca de un tercio no llega ni a ambiguo: son plurales, órdenes de palabra o pares idénticos que el
resolvedor del tracer no normaliza. El resto es especificidad real.

## Consecuencia para quien quiera montar la alarma

El canal `missing_in_raw` del tracer, **consumido tal cual, sería 100 % ruido**: 81 disparos, 0
casos de ingrediente ausente. Eso explica mejor que nada por qué lleva dos meses sin consumidores
(`P2-COHERENCE-EJE-CIEGO`) — no es que nadie se acordara: la señal no es usable sin normalizar
identidad primero.

## Lo que sigue siendo del dueño

De los ~53 que sí son especificidad, ¿cuáles son defecto? Mi lectura: **«queso cheddar» leído y
«queso» genérico comprado sí lo es** (producto y precio distintos); **«pechuga» leída y «pechuga de
pollo» comprada, no**. Esa frontera es criterio de producto, no técnico — y decide el destino de
unas 53 comidas vivas.
