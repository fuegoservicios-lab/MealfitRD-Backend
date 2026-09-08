# -*- coding: utf-8 -*-
"""[P2-COHERENCE-EJE-CIEGO · 2026-09-08] El guard de coherencia no mira el eje donde vivió la avería.

El 07 y 08 de septiembre se cerraron **cinco** defectos de la familia `raw[idx]` con daño medido
sobre planes vivos: el humanizador (44 comidas), el tope y el piso de porción, el lácteo fantasma
(«30 huevos», 18 comidas), el pase de presupuesto (171) y su gemelo (38). En todos, la receta decía
una cosa y la lista compraba otra.

El guard de coherencia recetas↔lista corría en modo `block` durante todo ese tiempo y **no disparó ni
una vez**. No porque estuviera roto: porque **no mira ese eje**.

## Por qué es estructural, no una avería

Los DOS lados del guard leen el mismo campo:

    expected_sum_from_recipes   → `meal.get("ingredients_raw") or meal.get("ingredients")`
    el aggregator de la lista   → `meal.get("ingredients_raw") or meal.get("ingredients", [])`

La simetría es deliberada y correcta para lo que el guard defiende («la lista compra lo que las
recetas piden»): los dos lados deben hablar de la verdad de COMPRA. Pero de ahí se sigue que el guard
es ciego, por construcción, a la divergencia entre `ingredients` —lo que el usuario LEE— e
`ingredients_raw` —lo que el sistema COMPRA—. Es la misma forma que `P0-SHOPPING-CYCLE-DAYS`:
*cuando las dos referencias comparten el defecto, mutilar una MEJORA la métrica.*

## La instrumentación ya existe, y nadie la lee

`_trace_misalign` (`P1-RAW-MISALIGN-TRACE` · 2026-07-06, profundizado 24-jul) mide exactamente ese
eje en 8 puntos del pipeline y lo persiste en `plan_data._raw_misalign_stages` y en
`meal._misalign_trace`. Medido sobre la flota el 08-sep: **714 de 1.194 comidas (59,8 %) llevan
marca**, y 135 (11,3 %) siguen con huella viva en lo entregado. La etapa que más aparece es
`post_humanize` (374 de `qty`, 322 de `missing_in_raw`) — la ventana del pipeline que contiene el
humanizador Y el pase de presupuesto, o sea los DOS sitios de mayor daño medido esos días.

O sea: la traza llevaba dos meses apuntando al culpable correcto, dentro de `plan_data`, y
**tiene cero consumidores** — ni cron, ni métrica, ni alerta, ni frontend.

## Lo que NO se afirma aquí, a propósito

No se afirma que esas 135 comidas sean 135 defectos. Dos intentos de separarlas midieron mal:

  - resolviendo identidad con `_resolve_line_food_grams`, `batatas`/`batata` y
    `pechuga`/`pechuga de pollo` salían como alimentos distintos;
  - comparando gramos, el cilantro «0,4 g contra 0,2 g» superaba cualquier tolerancia relativa.

Y hay un modo que es la FUNCIÓN, no el fallo: «1¼ cdas de mantequilla de maní» en el display contra
«20 g» en la compra es la misma cantidad en dos unidades, a propósito.

La conclusión honesta no es un número: es que **este eje no tiene regla canónica de identidad**,
al contrario que el eje de la Nevera (`constants.pantry_names_match`, `P1-PANTRY-NAME-RESOLUTION`).
Por eso no tiene alarma — no se puede alarmar sobre algo que no se sabe definir. Montar el cron
antes que la definición metería ruido en `system_alerts`, que es el desenlace opuesto al que se
busca. *Un detector sin definición no mide: reparte.*

Este test ancla los DOS hechos estructurales para que el próximo que llegue no los redescubra, y
falla si alguien cambia uno sin actualizar el otro. **No ancla una demostración funcional**: se
intentó con un plan sintético y no discriminaba —el caso sano daba el mismo veredicto que el
enfermo—, así que se descartó en vez de publicarla. El argumento de la ceguera es de código.
"""
import inspect

import pytest

import shopping_calculator as sc


def test_el_lado_de_recetas_del_guard_lee_la_verdad_de_COMPRA():
    """`expected_sum_from_recipes` prefiere `ingredients_raw`. Es correcto — y es la razón de la
    ceguera. Si algún día se cambia a `ingredients`, el guard empieza a ver este eje y esta doc
    queda obsoleta: actualízala antes de tocar el orden."""
    src = inspect.getsource(sc.expected_sum_from_recipes)
    assert 'meal.get("ingredients_raw") or meal.get("ingredients")' in src


def test_el_lado_de_la_lista_lee_EL_MISMO_campo():
    """La simetría entre los dos lados es el diseño (P1-shop-coh-1). También es lo que hace que una
    divergencia display↔raw sea invisible: los dos lados la comparten."""
    src = inspect.getsource(sc)
    assert src.count('meal.get("ingredients_raw") or meal.get("ingredients"') >= 2


def test_la_ceguera_esta_argumentada_por_CODIGO_no_por_una_sonda():
    """Aquí hubo un test que decía demostrar la ceguera con un plan sintético. Era falso.

    El plan minimal (display dice pollo, `ingredients_raw` y la lista dicen camarones) daba «2
    divergencias» — pero el MISMO plan con raw y lista de acuerdo también daba 2, y con la proteína
    ausente de la lista, 2, y con la magnitud a la mitad, 2. Cuatro casos, cuatro veces lo mismo:
    *una sonda que dispara contra todo habla de la sonda*, no del sujeto. Al plan sintético le falta
    forma que el guard necesita, y montar el arnés completo es otra tarea.

    Y con el pool cerrado era peor: el catálogo sale vacío, todo degrada a `unknown`, y el assert
    original («'pollo' no aparece en las divergencias») pasaba por el nombre del alimento, no por la
    ceguera. Habría pasado igual contra un guard completamente roto.

    Lo que SÍ está establecido son los dos hechos de código de arriba —los dos lados leen
    `ingredients_raw`— y de ahí se sigue la ceguera por construcción. Ese es el argumento. Este test
    existe para que nadie vuelva a colar una demostración funcional sin un arnés que discrimine:
    antes de escribirla, comprueba que el caso SANO sale limpio.
    """
    src = inspect.getsource(sc.expected_sum_from_recipes)
    assert '"ingredients"' in src and '"ingredients_raw"' in src


def test_la_traza_del_eje_existe_y_se_persiste():
    """`_trace_misalign` mide el eje ciego y lo guarda en el plan. Que exista es la razón por la que
    la respuesta a este gap NO es instrumentar: es definir identidad y luego leer lo ya escrito."""
    import graph_orchestrator as go

    src = inspect.getsource(go)
    assert "_raw_misalign_stages" in src
    assert "def _trace_misalign" in src
    assert src.count("_trace_misalign(") >= 6, "se perdieron puntos de traza del eje"


@pytest.mark.parametrize("consumidor", ["cron_tasks", "routers.plans"])
def test_la_traza_sigue_SIN_consumidor(consumidor):
    """Cero lectores el 08-sep. Si mañana alguien la consume, este test falla y obliga a decir dónde
    — que es justo la conversación que hoy no existe."""
    import importlib

    mod = importlib.import_module(consumidor)
    src = inspect.getsource(mod)
    assert "_raw_misalign_stages" not in src and "_misalign_trace" not in src, (
        f"{consumidor} empezó a leer la traza del eje ciego: documenta el consumidor y su umbral "
        "en `P2-COHERENCE-EJE-CIEGO` antes de dejarlo entrar."
    )
