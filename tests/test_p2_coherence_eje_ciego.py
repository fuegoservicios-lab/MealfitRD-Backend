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
Sin definición no hay alarma posible — *un detector sin definición no mide: reparte.*

## Pero un tercio del eje SÍ se deja definir, y mide 0

Son tres casos y sólo uno es ambiguo (ver `P2-EJE-CASO3-IDENTIDAD` al final del fichero). El caso
«alimento DISTINTO» —la receta nombra algo que no está en la compra por ningún lado— no pide
criterio de producto, y medirlo sobre la flota da **0 de 1.172 comidas vivas**.

Eso cambia la recomendación: para la mitad inequívoca del eje **no hay alarma que montar**, porque
no hay exposición. Los defectos `raw[idx]` de esos dos días fueron reales, pero de la clase
CANTIDAD y LÍNEA EQUIVOCADA — no «lees pollo y compras camarones». Lo que sobrevive hoy en el eje
es nombre, unidad y redondeo.

Este test ancla los DOS hechos estructurales para que el próximo que llegue no los redescubra, y
falla si alguien cambia uno sin actualizar el otro. **No ancla una demostración funcional**: se
intentó con un plan sintético y no discriminaba —el caso sano daba el mismo veredicto que el
enfermo—, así que se descartó en vez de publicarla. El argumento de la ceguera es de código.
"""
import inspect

import pytest

import re as _re

import graph_orchestrator as go
import shopping_calculator as sc
from constants import canonical_pantry_key
from culinary_coherence import CONDIMENT_EXEMPT


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


# ─────────────────────────────────────────────────────────────────────────────
# [P2-EJE-CASO3-IDENTIDAD · 2026-09-08] La mitad del eje que SÍ se puede definir
#
# Arriba se dice que el eje no tiene regla canónica de identidad y que por eso no puede tener
# alarma. Cierto para el caso ambiguo — «pechuga» contra «pechuga de pollo» pide criterio de
# producto. Pero el eje tiene tres casos y sólo uno es ambiguo:
#
#     1) mismo alimento, otra unidad   («1¼ cdas» vs «20 g»)              → la FUNCIÓN
#     2) más o menos específico        («pechuga» vs «pechuga de pollo»)  → AMBIGUO
#     3) alimento DISTINTO             («pollo» vs «camarones»)           → defecto sin discusión
#
# El caso 3 se puede definir hoy, y medirlo dio **0 de 1.172 comidas vivas**. Ese 0 llegó tras
# CINCO revisiones del instrumento, y las cuatro primeras midieron de más:
#
#     37  →  singularizador ingenuo: `dientes`→`dient` mientras `diente` quedaba intacto
#     20  →  condimentos «al gusto», que a propósito NO van a la lista
#     11  →  el paréntesis de equivalencia: «(≈15 g)» contra «(≈14.81 g)», la MISMA línea
#      0  →  con las cuatro correcciones
#
# *Cada versión equivocada del instrumento acusaba a producción de su propio defecto.* Por eso las
# funciones viven aquí con sus casos de discriminación: un 0 sólo vale si el detector demuestra que
# todavía dispara.

# La lista exenta es la de la casa, no una mía — y NO se fusiona con `_ALLOWED_CONDIMENTS`, que
# responde a otra pregunta (P1-PANTRY-CONDIMENT-PARITY).
_EXENTO = frozenset(w for f in CONDIMENT_EXEMPT for w in str(f).split())

# El paréntesis de equivalencia («(≈203 g)») es una ANOTACIÓN, no identidad: dejarlo entrar hacía
# que «(≈15 g)» y «(≈14.81 g)» —la MISMA línea, redondeada distinto a cada lado— salieran como
# alimentos distintos. Tres de los últimos once positivos eran exactamente eso.
_PAREN = _re.compile(r"\([^)]*\)")

# Tokens que no identifican un alimento: preparacion, envase, cortes.
_RUIDO = {
    "cocido", "cocida", "cocidos", "cocidas", "crudo", "cruda", "crudos", "crudas",
    "picado", "picada", "rallado", "rallada", "molido", "molida", "fresco", "fresca",
    "grande", "mediano", "mediana", "pequeno", "pequena", "sin", "piel", "hueso",
    "natural", "entero", "entera", "en", "de", "la", "el", "los", "las", "y", "con",
    "cubos", "tiras", "lonjas", "mitad", "mitades", "partido", "partida", "desmenuzado",
    "desmenuzada", "lata", "escurrido", "escurrida", "hervido", "hervida", "asado", "asada",
}


def _sing(t):
    """Raiz comun de singular y plural, sin diccionario.

    En espanol no se puede deducir del plural: `diente`->`dientes` (vocal + s) y `flor`->`flores`
    (consonante + es) se ven igual por la cola. Mis dos intentos anteriores lo trataron como regla
    mecanica y partieron `dientes` en `dient` dejando `diente` intacto — dos formas del MISMO
    alimento como alimentos distintos, que era la mayoria de los positivos.

    La salida: no intentar el singular, sino una RAIZ a la que llegan los dos. Quitar la `s` final y
    luego la `e` final lleva `dientes`->`diente`->`dient` y `diente`->`dient`; `flores`->`flore`->
    `flor` y `flor`->`flor`. No es morfologia: es un punto de encuentro, que es lo unico que hace
    falta para comparar.
    """
    if len(t) <= 3:
        return t
    if t.endswith("s"):
        t = t[:-1]
    if len(t) > 3 and t.endswith("e"):
        t = t[:-1]
    return t


def _tokens(nombre):
    k = canonical_pantry_key(_PAREN.sub(" ", str(nombre) or ""))
    if not k:
        return frozenset()
    return frozenset(
        _sing(t) for t in str(k).split()
        if t and t not in _RUIDO and len(t) > 2 and not any(ch.isdigit() for ch in t))


def _mismo_alimento(a, b):
    """Mismo alimento si un conjunto de tokens contiene al otro. Nunca por subcadena."""
    if not a or not b:
        return True          # sin resolver -> no acuso
    return a <= b or b <= a


def _foods_de(lineas):
    out = []
    for s in lineas:
        try:
            f, _ = go._resolve_line_food_grams(str(s), cheap=True)
        except Exception:
            f = None
        t = _tokens(f or s)
        if t:
            out.append((t, str(s)))
    return out



_CASOS = [
    (True,  "lee pollo, compra camarones",        ["150 g de pechuga de pollo"], ["150 g de camarones"]),
    (True,  "lee granola, compra avena",          ["32 g de Granola"],           ["31 g de Avena"]),
    (False, "mismo alimento, otra unidad",        ["1¼ cdas de mantequilla de maní"], ["20 g de mantequilla de maní"]),
    (False, "singular contra plural",             ["1 diente de ajo"],           ["0.87 dientes de ajo"]),
    (False, "paréntesis redondeado distinto",     ["½ oz de queso blanco (≈15 g)"], ["0.49 oz de queso blanco (≈14.81 g)"]),
    (False, "la compra es más específica",        ["1½ filetes de pescado"],     ["220 g de filete de pescado blanco"]),
    (False, "condimento al gusto",                ["Sal al gusto"],              ["100 g de arroz"]),
]


@pytest.mark.parametrize("esperado,nombre,display,compra", _CASOS,
                         ids=[c[1].replace(" ", "_") for c in _CASOS])
def test_la_regla_del_caso_3_discrimina(esperado, nombre, display, compra):
    """Los dos primeros DEBEN disparar; los cinco siguientes son variaciones legítimas y no.

    Sin estos casos, el «0 sobre la flota» no significa nada: un detector que no dispara nunca sale
    perfecto. Cada uno de los cinco negativos es un falso positivo que este instrumento cometió de
    verdad antes de corregirse.
    """
    fd, fr = _foods_de(display), _foods_de(compra)
    huerfanos = [s for t, s in fd
                 if not (t & _EXENTO) and "al gusto" not in s.lower()
                 and not any(_mismo_alimento(t, tr) for tr, _ in fr)]
    assert bool(huerfanos) is esperado, f"{nombre}: huérfanos={huerfanos}"
