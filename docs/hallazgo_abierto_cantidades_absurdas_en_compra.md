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
- **No es residuo del cap por índice** (`P1-CAP-BIGFRUIT-BREAD-RAW-BY-FOOD`, cerrado el 07-sep). Se
  probó re-ejecutando `_cap_unrealistic_portions` ya arreglado sobre las 7: **cura 2, deja 5**. Las
  cinco que quedan no las cubre ese cap.
- **No es escalado por hogar** (multiplier `None`).

## Un falso positivo de la sonda, para que no infle el número

«Cebada Salteada»: LEE «2¼ tazas de cebollín», COMPRA «34,75 cdas de cebollín». 2¼ tazas = 36 cdas,
así que **esa conversión es CORRECTA** y la sonda la marcó por mirar sólo el número. El recuento
honesto de casos reales es 3-4, no 8.

## Por qué el guard de coherencia no lo ve

Los dos lados del guard leen `ingredients_raw` (ver `P2-COHERENCE-EJE-CIEGO`), así que una cantidad
absurda que vive en raw está en AMBOS lados de la comparación: la lista «coincide» consigo misma.

## Qué haría falta

Identificar quién escribe el literal `30` conservando la unidad. Las 5 funciones que hacen
`append` sobre raw están inventariadas (`_repair_name_phantom_dairy`,
`_repair_declared_but_unlisted_ingredients`, `_cap_daily_whole_eggs`,
`_garnish_herb_mention_backfill`, `_reconcile_display_missing_in_raw`) y ninguna emite ese formato,
así que el productor escribe por asignación, no por append — o vive fuera de `graph_orchestrator`.

Volumen bajo (3-4 comidas de 1.194, en 3 planes), pero el caso del aceite es caro y visible: quien
lo compre va a notar la botella entera.
