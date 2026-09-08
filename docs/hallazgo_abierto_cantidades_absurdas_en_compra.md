# Hallazgo abierto: la lista compra 30 cucharadas de aceite para un plato que usa ¾ de cucharadita

**Fecha:** 2026-09-08 · **Estado:** ABIERTO, productor NO identificado · **Severidad:** alta por caso, baja por volumen

## Qué se midió

Sobre las 1.194 comidas de los 95 planes vivos, buscando cantidades imposibles en unidades de
cocina (>6 cdtas, >8 cdas, >6 tazas, >12 dientes):

| campo | comidas con cantidad absurda |
|---|---|
| `ingredients` (lo que el usuario LEE) | **0** |
| `ingredients_raw` (lo que el sistema COMPRA) | **8** |

La asimetría es total: ninguna llega a la receta, todas llegan a la compra.

## Los casos, con el par display↔compra

Comprobado que **no es escalado por hogar**: `_household_multiplier` es `None` en los tres planes.

| plato | LEE | COMPRA | factor |
|---|---|---|---|
| Bollitos de Yuca (`d476023a`) | ¾ cdta de aceite de oliva | **30 cdas** de aceite de oliva | ≈1.600× |
| Mapuey Horneado (`e2bbb280`) | ½ taza de rábanos en láminas | **30 tazas** de rábanos | 60× |
| Mapuey Horneado (`e2bbb280`) | ½ cda de cebolla picada | **30 cdas** de cebolla picada | 60× |

El primero es el caro: **~440 ml de aceite de oliva comprados para una comida que usa ~4 ml.**

## El literal `30`

Aparece tres veces con TRES unidades distintas (cdas, tazas, cdas). Eso apunta a un valor por
defecto que se escribe conservando la unidad de la línea, no a un error de conversión — una
conversión equivocada daría factores distintos en cada caso, no el mismo número.

## Lo que se descartó

- **No es el humanizador.** Su rama de grasas (`humanize_ingredients.py:517-521`) sólo emite hasta
  `base_qty <= 60` y en SINGULAR (`cda`); estas líneas son plurales (`cdas`) y de magnitud mayor.
- ~~**No es residuo del cap por índice.**~~ **CORRECCIÓN (misma sesión):** esa línea decía que se
  había descartado porque re-ejecutar `_cap_unrealistic_portions` ya arreglado cura 2 de 7. **Ese
  experimento no falsa nada**: re-ejecutar un pase corregido no cura daño ya escrito, así que su
  resultado es compatible con la hipótesis y con su contraria. Además probé el TECHO cuando los
  cuatro platos llevan `_portion_floor_adjusted` — el **PISO**. Repetido con los dos: **cura 2,
  deja 5**, igual, y por la razón obvia: una línea que ya dice «30 cdas» no está por debajo del
  piso ni por encima del techo, así que ninguno la mira.
- **No es escalado por hogar** (multiplier `None`).
- **No es el cap bariátrico**, pese a que `BARIATRIC_CHEESE_CAP_G` y `BARIATRIC_AVOCADO_CAP_G` valen
  exactamente `30` y la coincidencia invitaba. Probado `_resc_cap_coherent` directamente: escala por
  factor conservando la unidad y **sólo escribe el número del tope cuando la unidad ya es gramos**
  (`450 g` → `30 g`; `3 cdas` → `0,99 cdas`). Además sólo toca `ingredients`, no `raw`.

## Un falso positivo de la sonda, para que no infle el número

«Cebada Salteada»: LEE «2¼ tazas de cebollín», COMPRA «34,75 cdas de cebollín». 2¼ tazas = 36 cdas,
así que **esa conversión es CORRECTA** y la sonda la marcó por mirar sólo el número. El recuento
honesto de casos reales es 3-4, no 8.

## Por qué el guard de coherencia no lo ve

Los dos lados del guard leen `ingredients_raw` (ver `P2-COHERENCE-EJE-CIEGO`), así que una cantidad
absurda que vive en raw está en AMBOS lados de la comparación: la lista «coincide» consigo misma.

## La hipótesis que el descarte fallido tapaba: el piso, por ÍNDICE

Con la prueba corregida, la evidencia **apunta al piso**:

- las cuatro comidas dañadas llevan `_portion_floor_adjusted`;
- `_floor_subservible_portions` calcula `factor = PORTION_SHRINK_FLOOR_G / gramos` y aplica
  `rescale_ingredient_string`, que **conserva la unidad**: con unos gramos mal resueltos (≈0,25 g) el
  factor sale de 60× a 120×, y «½ cda» pasa a «30 cdas»;
- **hasta el 07-sep ese pase escribía `ingredients_raw[idx]` por ÍNDICE**
  (`P1-FLOOR-RAW-BY-FOOD`, 9 escrituras): aplicaba el factor enorme de UNA línea a OTRA. Eso explica
  a la vez que el display esté sano y que la compra no lo esté.

No está probado —haría falta reproducir la resolución de gramos que da el factor— pero es la
hipótesis viva, y llegó por corregir un descarte mal hecho, no por una sonda nueva.

## Qué haría falta

Identificar quién escribe el literal `30` conservando la unidad. Las 5 funciones que hacen
`append` sobre raw están inventariadas (`_repair_name_phantom_dairy`,
`_repair_declared_but_unlisted_ingredients`, `_cap_daily_whole_eggs`,
`_garnish_herb_mention_backfill`, `_reconcile_display_missing_in_raw`) y ninguna emite ese formato,
así que el productor escribe por asignación, no por append — o vive fuera de `graph_orchestrator`.

Volumen bajo (3-4 comidas de 1.194, en 3 planes), pero el caso del aceite es caro y visible: quien
lo compre va a notar la botella entera.
