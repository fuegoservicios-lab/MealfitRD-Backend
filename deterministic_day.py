# -*- coding: utf-8 -*-
"""[P1-DETERMINISTIC-DAY · 2026-09-08] Un día de plan armado SIN llamar al modelo.

## Por qué existe

El sistema estaba en **2 de 10** de determinismo. Los guards, la lista de compras y el descuento
de la Nevera ya eran deterministas; lo que decidía **qué comes** era el LLM, distinto en cada
generación. Y toda la maquinaria para no depender de él —PlanPolicy, blueprint, Dish Registry,
CandidateSet, biblioteca de 140 recetas— estaba construida, probada, desplegada y **nunca había
corrido en producción**: 94 de 95 planes vivos sin sello de política, 1 de 1.194 comidas
coincidiendo con una plantilla.

## La medición que reordenó el trabajo

Lo primero que iba a hacer era enchufar la biblioteca de recetas. Medido contra 808 comidas de 60
planes vivos: **0 recibirían receta congelada**. Mientras el modelo invente el nombre del plato no
hay coincidencia posible. *El cuello de botella no era el texto de la receta: era quién elige el
plato.* Por eso este módulo empieza por la elección.

## Qué hace, y qué NO

Para cada franja toma los candidatos del registry —ya filtrados por alergia, dieta, nutrientes
requeridos, mercado y durabilidad en `dish_registry.template_candidates`— elige uno puntuando por
MACROS, escala los gramos al objetivo calórico, inclina proteína contra carbohidrato dentro de un
tope, y pega la receta congelada de la biblioteca.

**No reimplementa el filtro de alergia ni el de dieta**: se los PASA a
`dish_registry.template_candidates`, que es el SSOT. Escribir una segunda tabla es la lección de
`P1-DIET-CANON-SSOT` (eran 3, drifearon, y la del filtro olvidó `vegetariana` — servía pollo a
vegetarianas).

Y esa frase estuvo mal escrita durante unas horas: decía que «el CandidateSet ya filtra», cierto de
`_registry_slice` pero NO de esta función, que llama a `template_candidates` directamente y no le
pasaba nada. Lo cazó el backstop clínico rechazando un desayuno con huevo a un alérgico al huevo.
La defensa en profundidad funcionó, y por eso mismo el hueco de arriba había que cerrarlo: una
última línea de defensa que trabaja sola dejó de ser defensa en profundidad.

**Dónde se persiste.** [P1-AUDITORIA-ARQ-VERIFICADA · 2026-09-11] Este párrafo afirmaba que un día
armado aquí se persistía SIN pasar por `assemble_plan_node`, y no es cierto: el ÚNICO llamador
de producción es `_safe_gen` dentro de `generate_days_parallel_node`, y las aristas del grafo
(`generate_days_parallel → adversarial_judge → self_critique → assemble_plan → review_plan`) son
incondicionales — el día determinista pasa por el autofix de sodio, el de proteína repetida, el de
huevo y el recorte de comidas igual que uno del modelo. Las capas de este módulo (`verifica_comida`,
el techo de sodio del día) son DEFENSA EN PROFUNDIDAD, no la única defensa; la premisa falsa
justificó dos P-fixes correctos por la razón equivocada. Devolver `None` sigue siendo seguro: el
llamador cae al camino del LLM, que es el estado de siempre.

## Lo medido el 08-sep sobre 14 días × 3 perfiles clínicos

| perfil | días | calorías | proteína dentro de ±15 % |
|---|---|---|---|
| mantenimiento 2000 | 14/14 | ±0,0 % | 14/14 |
| pérdida de grasa 1700 alta proteína | 14/14 | ±0,0 % | 12/14 |
| ganancia muscular 2600 | 14/14 | −0,3 % | 14/14 |

Y lo que **no** se arregló: los carbohidratos se quedan en +23,6 % en el perfil de pérdida de
grasa. Eso no es un fallo del algoritmo — el arroz, los víveres y el plátano SON la cocina criolla.
Cerrarlo pide plantillas nuevas bajas en carbohidrato, que es una decisión de producto.

Determinismo verificado armando cada día dos veces y comparando: **14/14 idénticos byte a byte**,
0 comidas sucias de 56 en el escáner culinario.

## Los dos límites de los datos

  · **7 de 144 plantillas** tienen un constituyente sin macros en el catálogo. No se pueden escalar
    con honestidad (un nutriente ausente NO es cero, `P1-ARQ27-F1`) y se descartan como candidatas.
  · **Las meriendas del registro son grandes**: mediana 321 kcal contra un objetivo típico de 200.
    Por eso la banda de escala es POR FRANJA — media merienda se entiende sola, medio locrio no.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

# Fuera de esta banda el plato NO es candidato para ese objetivo: preferimos elegir otro plato que
# servir una porción absurda del que tocaba. Es POR FRANJA porque el sentido culinario lo es —
# medida el 08-sep: con una banda única, 2 de 14 días se quedaban sin merienda.
_BANDA_POR_FRANJA = {
    "merienda": (0.35, 1.60),
    "desayuno": (0.60, 1.60),
    "almuerzo": (0.60, 1.60),
    "cena": (0.60, 1.60),
}
_BANDA_DEFECTO = (0.60, 1.60)

# El condimento no crece con la porción: un locrio para dos no lleva el doble de orégano, y
# multiplicar la sal por 1,5 es un problema clínico, no de sabor.
_NO_ESCALAN = (
    "sal", "pimienta", "oregano", "ajo", "comino", "canela", "laurel", "vinagre", "bija",
    "achiote", "curcuma", "sazon", "perejil", "cilantro", "azafran", "nuez moscada", "clavo",
)
# [P1-AUDITORIA-ARQ-VERIFICADA · 2026-09-11] Por PALABRA, no por subcadena. `"sal" in "salami"` era
# True: Salami, Salsa de tomate, Ensalada, Salmón, Bacalao salado y Ajonjolí («ajo») quedaban sin
# escalar — 15 plantillas dominicanas con un constituyente congelado, y en «Salami guisado con yuca»
# el congelado era la proteína entera: el plato se elegía por sus 427 kcal escaladas y servía 574.
# Los dos SSOT de condimentos del repo (`constants._ALLOWED_CONDIMENTS_RES`,
# `culinary_coherence._CONDIMENT_EXEMPT_RES`) ya usan frontera de palabra; ésta era la tercera copia
# sin migrar. Se admite el plural («clavos», «ajos»).
_NO_ESCALAN_RES = tuple(
    re.compile(r"(?<![a-z])" + re.escape(k) + r"(?:s|es)?(?![a-z])") for k in _NO_ESCALAN
)

# Inclinación de constituyentes: ±35 % por ingrediente y nunca por debajo del 30 % del gramaje
# original. «Lentejas guisadas con arroz» con un 30 % menos de arroz sigue siendo ese plato; con un
# 70 % menos, no. Y bajar a cero además dejaría el ingrediente comprado y sin usar en la receta —
# el huérfano V3 que el escáner ya caza.
_TILT_TOPE = 0.35
_TILT_MIN_FRAC = 0.30

# [P1-AUDITORIA-ARQ-VERIFICADA · 2026-09-11] Aquí vivía `_REPARTO = {desayuno .25, almuerzo .35,
# cena .30, merienda .10}`: una SEGUNDA tabla frente a `nutrition_calculator.MEAL_SLOT_SPLITS` (la del
# solver del camino del LLM: .20/.35/.15/.30 para 4 comidas, con repartos propios para 2, 3, 5 y 6), y
# se leía junto a `skeleton_day["slots"]` — la forma del BLUEPRINT (`horizon.build_blueprint`) — cuando
# el único llamador de producción (`generate_days_parallel_node`) entrega un `DaySkeletonModel`, que
# trae `meal_types` y `protein_pool` y ninguna de las dos claves. Medido: `slots` era siempre `None`
# ⇒ siempre las 4 franjas de la tabla, en su orden, sin familia de proteína; un plan clínico de 3 o
# de 6 comidas recibía 4 (`_enforce_meal_count` sólo recorta, no añade). Ningún test lo vio porque
# todos pasaban `{"slots": [...]}`. Ahora las franjas salen de `_franjas_del_dia` y el reparto de
# `_fracciones_por_franja`, sobre los SSOT que ya usa el resto del sistema.
_FRANJAS_REGISTRY = ("desayuno", "almuerzo", "cena", "merienda")


def _etiqueta_a_franja(etiqueta) -> Optional[str]:
    """«Merienda AM» → `merienda`, «Desayuno» → `desayuno`, «dinner» → `cena`; `None` si no se reconoce.

    Primero el mapa del camino del LLM (`nutrition_calculator._SLOT_KEY_MAP` — vivía en el god-file
    hasta P1-PLAN-FASE-A —, el que leen
    `_canonical_slot_fractions` y `_enforce_meal_count`), después los alias del registry
    (`dish_registry.canonical_slot_es`). Los dos existen; aquí no se escribe un tercero."""
    n = _norm(etiqueta)
    if not n:
        return None
    try:
        from nutrition_calculator import _SLOT_KEY_MAP
        k = _SLOT_KEY_MAP.get(n)
        if k in _FRANJAS_REGISTRY:
            return k
    except Exception:                                                  # noqa: BLE001
        pass
    try:
        import dish_registry as dr
        k = dr.canonical_slot_es(n)
        if k in _FRANJAS_REGISTRY:
            return k
    except Exception:                                                  # noqa: BLE001
        pass
    return None


def _fracciones_por_franja(etiquetas: list) -> list:
    """Fracción de kcal por comida, del reparto fisiológico SSOT (`MEAL_SLOT_SPLITS`) y con el MISMO
    algoritmo que el solver del camino del LLM (`_canonical_slot_fractions`): las meriendas toman su
    cuota en orden AM → PM → noche, lo no mapeado reparte el remanente, el vector suma 1,0."""
    try:
        from nutrition_calculator import _canonical_slot_fractions
        return list(_canonical_slot_fractions([{"meal": e} for e in etiquetas]))
    except Exception:                                                  # noqa: BLE001
        pass
    # Respaldo si `nutrition_calculator` no importa (pruebas aisladas): los MISMOS datos, la misma regla.
    try:
        from nutrition_calculator import MEAL_SLOT_SPLITS
        split = MEAL_SLOT_SPLITS.get(len(etiquetas), MEAL_SLOT_SPLITS[4])
        meriendas = [k for k in split if k.startswith("merienda")]
        out, i = [], 0
        for e in etiquetas:
            f = _etiqueta_a_franja(e)
            if f in split:
                out.append(split[f])
            elif f == "merienda" and meriendas:
                out.append(split[meriendas[min(i, len(meriendas) - 1)]])
                i += 1
            else:
                out.append(None)
        asignado = sum(x for x in out if x is not None)
        sin = sum(1 for x in out if x is None)
        if sin:
            resto = max(0.0, 1.0 - asignado) / sin
            out = [resto if x is None else x for x in out]
        tot = sum(out) or 1.0
        return [x / tot for x in out]
    except Exception:                                                  # noqa: BLE001
        return []


def _franjas_del_dia(skeleton_day) -> Optional[list]:
    """Las franjas del día como `[(etiqueta, franja_registry, fracción_kcal)]`, o `None` si alguna
    etiqueta no se reconoce (que la haga el LLM: ese contrato sí sabe qué es un «brunch»).

    `meal_types` manda: es lo que emite el planificador (`DaySkeletonModel`) y lo que consume el
    day-generator del LLM, así que las DOS rutas arman las mismas comidas, en el mismo orden y con la
    misma etiqueta («Merienda AM» se conserva tal cual: `_enforce_meal_count` y el frontend cuentan
    por esa etiqueta). `slots` (blueprint) se acepta como forma alternativa. Sin ninguna, las 4
    comidas canónicas del producto (`meal_types_for_count(4)`)."""
    sk = skeleton_day or {}
    etiquetas = [str(x) for x in (sk.get("meal_types") or []) if x]
    if not etiquetas:
        etiquetas = [str(s).capitalize() for s in (sk.get("slots") or []) if s]
    if not etiquetas:
        try:
            from nutrition_calculator import meal_types_for_count
            etiquetas = list(meal_types_for_count(4))
        except Exception:                                              # noqa: BLE001
            etiquetas = ["Desayuno", "Almuerzo", "Merienda", "Cena"]
    franjas = [_etiqueta_a_franja(e) for e in etiquetas]
    if any(f is None for f in franjas):
        return None
    fracs = _fracciones_por_franja(etiquetas)
    if len(fracs) != len(etiquetas):
        return None
    return list(zip(etiquetas, franjas, fracs))


def _familias_del_dia(skeleton_day) -> list:
    """La familia de proteína programada para el día: `protein` (blueprint) o `protein_pool`
    (`DaySkeletonModel`: nombres de alimento como «Pechuga de pollo», que `horizon.family_matches`
    ya sabe leer). Vacía ⇒ sin restricción de familia."""
    sk = skeleton_day or {}
    if sk.get("protein"):
        return [str(sk["protein"])]
    return [str(p) for p in (sk.get("protein_pool") or []) if p]

_RE_LINEA = re.compile(r"^\s*([\d.]+)\s*g\s+de\s+(.+)$")


def deterministic_day_enabled() -> bool:
    """Knob. Por defecto APAGADO: encender esto cambia QUÉ come el usuario, y eso se decide."""
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_DETERMINISTIC_DAY", False)
    except Exception:
        return False


def deterministic_day_for_user(user_id=None) -> bool:
    """¿Este usuario recibe días deterministas?

    Tres estados, y el del medio es el que hace que esto se pueda llevar a producción:

      · knob global ON  ⇒ todos.
      · knob global OFF + usuario en `MEALFIT_DETERMINISTIC_DAY_USERS` ⇒ **sólo él**.
      · nada ⇒ nadie.

    Sin el estado del medio, encender esto cambia la dieta de TODOS a la vez y la única marcha
    atrás es otro despliegue. El repo ya tenía el patrón —`MEALFIT_PLAN_POLICY_ENFORCE_USERS`,
    «dueño → test → flip»— y no copiarlo era exactamente lo que separaba «funciona en mi medición»
    de «se puede poner delante de usuarios».
    """
    if deterministic_day_enabled():
        return True
    if not user_id:
        return False
    try:
        import os
        crudo = os.environ.get("MEALFIT_DETERMINISTIC_DAY_USERS", "") or ""
        if not crudo.strip():
            return False
        permitidos = {u.strip().lower() for u in crudo.split(",") if u.strip()}
        return str(user_id).strip().lower() in permitidos
    except Exception:
        return False

def _candidatos_k() -> int:
    """Cuántos candidatos pedir por franja. Medido: con 3 la proteína se iba a −39 % en pérdida de
    grasa; con 25 y selección por macros, 12 de 14 días entran en banda. Pedir más no cuesta
    llamadas — es una consulta al snapshot ya cargado."""
    try:
        from knobs import _env_int
        return _env_int("MEALFIT_DETERMINISTIC_DAY_CANDIDATES", 25,
                        validator=lambda v: 3 <= v <= 60)
    except Exception:
        return 25


def _norm(s) -> str:
    s = unicodedata.normalize("NFD", str(s or "").strip().lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _banda(slot) -> tuple:
    return _BANDA_POR_FRANJA.get(_norm(slot), _BANDA_DEFECTO)


def _no_escala(nombre: str) -> bool:
    """¿Es un condimento que no crece con la porción? Por palabra completa: «Sal», «Sal marina» y
    «Ajo en polvo» sí; «Salami», «Salsa de tomate», «Ensalada», «Salmón» y «Ajonjolí» no."""
    n = _norm(nombre)
    return any(rx.search(n) for rx in _NO_ESCALAN_RES)


def _clase(fila: dict) -> str:
    """Proteico / carbohidratado / otro, por DENSIDAD real y no por el nombre. Clasificar por
    nombre es cómo `"res"` acabó dentro de `"fresas"` y `"pollo"` dentro de `"repollo"`."""
    k = float(fila.get("kcal_per_100g") or 0)
    if k <= 0:
        return "otro"
    if float(fila.get("protein_g_per_100g") or 0) * 4 / k >= 0.40:
        return "proteico"
    if float(fila.get("carbs_g_per_100g") or 0) * 4 / k >= 0.55:
        return "carbo"
    return "otro"


def _macros(gramos_por_nombre, catalogo: dict) -> Optional[dict]:
    """kcal y macros de una lista de (gramos, nombre). `None` si algo no tiene macros."""
    tot = {"kcal": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fats_g": 0.0}
    for g, nombre in gramos_por_nombre:
        fila = catalogo.get(nombre)
        if not fila or fila.get("kcal_per_100g") is None:
            return None
        r = float(g) / 100.0
        tot["kcal"] += float(fila["kcal_per_100g"]) * r
        tot["protein_g"] += float(fila.get("protein_g_per_100g") or 0) * r
        tot["carbs_g"] += float(fila.get("carbs_g_per_100g") or 0) * r
        tot["fats_g"] += float(fila.get("fats_g_per_100g") or 0) * r
    return tot if tot["kcal"] > 0 else None


def _de_plantilla(t: dict) -> list:
    return [(float(c.get("grams") or 0), c.get("name"))
            for c in (t.get("constituents") or []) if c.get("name")]


def _empate_score() -> float:
    """Cuánto peor que el mejor puede ser un candidato y seguir siendo elegible.

    [P1-DIA-DETERMINISTA-VARIEDAD · 2026-09-09] Medido sobre el perfil real del dueño (2.100 kcal):
    con +0,05 hay UN elegible por franja en tres de las cuatro; con +0,50 hay entre 4 y 8. El score
    es `2·|Δp|/p + |Δc|/c + |Δf|/f`, así que 0,50 es del orden de un 15 % de desvío en proteína más
    un 10 % en los otros dos — y ese desvío lo recoge después el cerrador de banda de proteína, que
    existe justamente para eso. Comer mofongo catorce días no lo recoge nadie."""
    from knobs import _env_float
    return _env_float("MEALFIT_DETERMINISTIC_DAY_TIE_SCORE", 0.50, validator=lambda v: 0.0 <= v <= 5.0)


def _empate_max() -> int:
    from knobs import _env_int
    return _env_int("MEALFIT_DETERMINISTIC_DAY_TIE_MAX", 10, validator=lambda v: 1 <= v <= 50)


#: El MISMO piso relativo de proteína que usa el resto del sistema (`go.PROTEIN_FLOOR_HARD_PCT`,
#: `protein_floor_last_word._PISO_POR_DEFECTO`). Se escribe aquí como constante y no se importa de
#: `graph_orchestrator` para no atar este módulo al god file; el test comprueba que no divergen —
#: escribir un CUARTO número sería la lección de `P1-DIET-CANON-SSOT`.
#: tooltip-anchor: PROTEIN_FLOOR_REL (test_p1_catalogo_proteina_desayuno.py)
PROTEIN_FLOOR_REL = 0.90


def _rotacion_de(day_num, slot: str) -> int:
    """La rotación de ESTA franja en ESTE día. Determinista y estable entre corridas.

    [P1-DIA-DETERMINISTA-VARIEDAD · 2026-09-09] Rotar las cuatro franjas con el mismo `day_num` las
    mueve en formación: si dos franjas comparten elegibles —«Arepitas de maíz» sale en desayuno Y
    en merienda— avanzan juntas y el plato se repite el mismo día. El desfase por franja rompe la
    formación sin introducir azar: `hash()` de Python está salado por proceso y daría un plan
    distinto en cada arranque, así que se usa un dígito estable del sha256 del nombre.
    """
    import hashlib
    try:
        d = int(day_num or 0)
    except (TypeError, ValueError):
        d = 0            # un índice de día ilegible no puede tumbar el día entero
    h = hashlib.sha256(_norm(slot).encode("utf-8")).hexdigest()
    return d + int(h[:4], 16)


def elegir_plantillas(tids, objetivo, catalogo: dict, por_id: dict, slot: str = "",
                      rotacion: int = 0, saturados: Optional[dict] = None,
                      max_rep: Optional[int] = None) -> list:
    """Los candidatos ELEGIBLES para esta franja, del mejor al peor dentro del empate, rotados por
    el día. Lista vacía si ninguno sirve.

    [P1-DIA-DETERMINISTA-VARIEDAD · 2026-09-09] Antes esto devolvía UNO —`cands[0]`— y por eso
    catorce días daban **7 platos distintos en 56 comidas**: el mismo almuerzo los 14 días. El
    `rotate` que `template_candidates` ya aplicaba era INERTE aquí, porque reordenar una lista que
    el scorer recorre entera no cambia quién gana. *Una palanca de variedad que el consumidor de la
    lista anula no es una palanca.*

    Devolver la lista y no el ganador arregla además un segundo modo de fallo: si al elegido le
    falta la receta congelada, `construir_comida` devuelve `None` y **el día entero** se iba al
    LLM. Ahora el llamador prueba el siguiente.
    """
    lo, hi = _banda(slot)
    op = max(float(objetivo.get("protein_g") or 0), 1.0)
    oc = max(float(objetivo.get("carbs_g") or 0), 1.0)
    of = max(float(objetivo.get("fats_g") or 0), 1.0)
    ok = float(objetivo.get("kcal") or 0)
    cands = []
    for tid in (tids or []):
        t = por_id.get(tid)
        if not t:
            continue
        base = _macros(_de_plantilla(t), catalogo)
        if not base:
            continue                     # 7 de 144: constituyente sin macros
        f = ok / base["kcal"]
        if not (lo <= f <= hi):
            continue                     # servir esto sería una porción absurda
        score = (2.0 * abs(base["protein_g"] * f - op) / op
                 + abs(base["carbs_g"] * f - oc) / oc
                 + abs(base["fats_g"] * f - of) / of)
        cands.append((round(score, 6), str(tid), t, f, base["protein_g"] * f))
    if not cands:
        return []
    cands.sort(key=lambda x: (x[0], x[1]))
    # [P1-CATALOGO-PROTEINA-DESAYUNO · 2026-09-09] DOS puertas, y la segunda existe porque la
    # primera se estrecha sola: un techo `mejor + margen` se mueve con el mejor candidato, así que
    # **añadir un plato bueno EXPULSA a otros**. Medido al dar de alta 20 platos: el fondo pasó de
    # 25 a 31 supervivientes por franja y los elegibles se quedaron en 5. Un criterio relativo al
    # rival no mide al plato: mide la competencia.
    #
    # La segunda puerta es ABSOLUTA y es la que el resto del sistema ya usa: si la proteína del
    # plato, ya escalada, llega al piso clínico de la franja, el plato es servible aunque otro
    # llegue mejor. Unión, nunca sustitución — sobre merienda la puerta de proteína sola daba 3
    # donde el score daba 6, así que cambiarla habría empeorado justo la franja más pobre.
    techo = cands[0][0] + _empate_score()
    piso_p = op * PROTEIN_FLOOR_REL
    vistos, elegibles = set(), []
    for s, tid, t, f, p in cands:
        if s <= techo or (op > 0 and p >= piso_p):
            if tid not in vistos:
                vistos.add(tid)
                elegibles.append((t, f))
    # [P1-AUDITORIA-ARQ-VERIFICADA · 2026-09-11] La rotación va ANTES del corte. Cortando primero, los
    # elegibles del puesto 11 en adelante eran inalcanzables TODOS los días, con cualquier rotación:
    # la ventana giraba siempre sobre los mismos diez. Todos los elegibles ya pasaron las dos puertas
    # (empate de score o piso de proteína), así que girar sobre el conjunto entero no sirve un plato
    # peor: sirve más platos distintos. El corte sigue acotando cuántos se prueban por franja.
    # [P1-PLAN-LOTE-2 · 2026-09-11 · B6] Los que ya agotaron su cuota de repetición en la ventana de 7
    # días (`saturados` = {template_id: veces servido en los 6 días anteriores}, `max_rep` = el tope de
    # la política del usuario) van al FINAL, no fuera: quedarse sin plato por no repetir es peor que
    # repetir — la misma doctrina que `usadas_hoy`. La rotación gira sólo sobre los frescos, para que la
    # cabeza rotada no vuelva a ser un saturado.
    _llenos: list = []
    if saturados and max_rep:
        def _lleno(p):
            return int((saturados or {}).get(str((p[0] or {}).get("template_id")), 0)) >= int(max_rep)
        _llenos = [p for p in elegibles if _lleno(p)]
        elegibles = [p for p in elegibles if not _lleno(p)]
    if rotacion and len(elegibles) > 1:
        r = int(rotacion) % len(elegibles)
        elegibles = elegibles[r:] + elegibles[:r]
    return (elegibles + _llenos)[:_empate_max()]


def elegir_plantilla(tids, objetivo, catalogo: dict, por_id: dict, slot: str = ""):
    """El candidato que mejor llega al objetivo de MACROS tras escalar a sus calorías.

    `objetivo` = {"kcal","protein_g","carbs_g","fats_g"} de ESTA franja.

    La proteína pesa el doble en el score a propósito: es la que tiene consecuencia clínica y la
    que el escalado uniforme no puede arreglar (multiplicar por 1,2 sube los tres macros a la vez y
    no cambia sus proporciones). Las calorías no entran en el score porque quedan clavadas por
    construcción.

    Determinista: se ordena por score y se desempata por `template_id`. Desempatar por el orden de
    la lista dependería de cómo vino la lista; por id no depende de nada.

    [P1-DIA-DETERMINISTA-VARIEDAD · 2026-09-09] Ahora es la cabeza de `elegir_plantillas`. Se
    conserva porque es lo que dice el contrato «el MEJOR candidato» y es lo que sus tests fijan;
    quien arma el día usa la lista entera para no repetir plato.
    """
    el = elegir_plantillas(tids, objetivo, catalogo, por_id, slot)
    return el[0] if el else None


def _inclinar(lineas: list, catalogo: dict, obj_p: float) -> list:
    """Sube proteicos y baja carbohidratados devolviendo la MISMA energía. Sin esto, la proteína se
    quedaba en −16,4 % en pérdida de grasa (1 de 14 días en banda); con esto, 12 de 14."""
    prot = [l for l in lineas if l[2] == "proteico"]
    carb = [l for l in lineas if l[2] == "carbo"]
    if not prot or not carb:
        return lineas
    p_act = sum(g * float(catalogo[n].get("protein_g_per_100g") or 0) / 100 for g, n, _ in lineas)
    if p_act >= obj_p:
        return lineas
    gan_max = sum(g * _TILT_TOPE * float(catalogo[n].get("protein_g_per_100g") or 0) / 100
                  for g, n, _ in prot)
    if gan_max <= 0:
        return lineas
    frac = min(1.0, (obj_p - p_act) / gan_max)

    kcal_extra = 0.0
    for l in prot:
        d = l[0] * _TILT_TOPE * frac
        l[0] = round(l[0] + d, 1)
        kcal_extra += d * float(catalogo[l[1]].get("kcal_per_100g") or 0) / 100
    kcal_carb = sum(g * float(catalogo[n].get("kcal_per_100g") or 0) / 100 for g, n, _ in carb)
    if kcal_carb <= 0:
        return lineas
    for l in carb:
        dens = max(float(catalogo[l[1]].get("kcal_per_100g") or 1) / 100, 1e-6)
        share = (l[0] * dens) / kcal_carb
        l[0] = round(max(l[0] - (kcal_extra * share) / dens, l[0] * _TILT_MIN_FRAC), 1)
    return lineas


def construir_comida(t: dict, factor: float, catalogo: dict, slot: str, country: str,
                     objetivo: Optional[dict] = None) -> Optional[dict]:
    """La comida completa: nombre, ingredientes escalados, macros y receta congelada."""
    lineas = []
    for g, nombre in _de_plantilla(t):
        fila = catalogo.get(nombre)
        if not fila or fila.get("kcal_per_100g") is None:
            return None
        gg = g if _no_escala(nombre) else round(g * float(factor), 1)
        if gg > 0:
            lineas.append([gg, nombre, _clase(fila)])
    if not lineas:
        return None

    if objetivo and float(objetivo.get("protein_g") or 0) > 0:
        lineas = _inclinar(lineas, catalogo, float(objetivo["protein_g"]))

    tot = _macros([(l[0], l[1]) for l in lineas], catalogo)
    if not tot:
        return None
    ings = [f"{l[0]:g} g de {l[1]}" for l in lineas]

    meal = {
        "meal": str(slot).capitalize(),
        "name": t.get("name"),
        "ingredients": ings,
        "ingredients_raw": list(ings),
        "calories": int(round(tot["kcal"])),
        "protein": f"{tot['protein_g']:.0f}g",
        "carbs": f"{tot['carbs_g']:.0f}g",
        "fats": f"{tot['fats_g']:.0f}g",
        # El rastro. Sin él, un día determinista y uno del modelo se ven IGUAL en la base y nadie
        # puede medir cuántos hay — la lección del 08-sep sobre lo que es inerte sin dejar huella.
        "_meal_source": "deterministic",
        "_template_id": t.get("template_id"),
        "_scale_factor": round(float(factor), 3),
    }
    try:
        from recipe_library import recipe_for_dish_name
        pasos = recipe_for_dish_name(t.get("name"), country)
        if pasos:
            # [P1-AUDITORIA-ARQ-VERIFICADA · 2026-09-11] La receta congelada se escribió UNA vez para
            # la ración del registry y sus pasos no llevan cantidades de ingrediente — salvo el agua,
            # que SÍ se mide («tres tazas de agua» para 90 g de quinoa). Escalar los gramos y copiar
            # el paso tal cual servía 54 g de quinoa con tres tazas: el propio paso («hasta que el agua
            # se haya absorbido») dejaba de poder cumplirse. El agua medida escala con el factor.
            escalados, cambio = escalar_agua_en_pasos(list(pasos), float(factor))
            meal["recipe"] = escalados
            meal["_recipe_source"] = "library"
            if cambio:
                meal["_recipe_water_scaled"] = True
    except Exception:
        pass
    if not meal.get("recipe"):
        return None      # sin receta congelada no hay determinismo del texto: que lo haga el LLM
    # [P1-AUDITORIA-ARQ-VERIFICADA · 2026-09-11] El tiempo del plato sale de su receta
    # (`logistics.prep_minutes_est`, `prep_minutes_source='receta'`, P1-MINUTOS-DE-LA-RECETA) o de la
    # estimación por técnica — nunca del relleno «15 min» de `assemble_plan_node`, que no es un dato.
    # El número existía desde el 10-sep y ningún código de producción lo leía.
    try:
        _lg = t.get("logistics") or {}
        _src = str(_lg.get("prep_minutes_source") or "")
        _min = int(_lg.get("prep_minutes_est") or 0)
        if _src in ("receta", "tecnica") and _min > 0:
            meal["prep_time"] = f"{_min} min"
            meal["_prep_time_source"] = _src
    except (TypeError, ValueError):
        pass
    return meal


# [P1-AUDITORIA-ARQ-VERIFICADA · 2026-09-11] «N tazas/litros/ml de agua» en un paso, con número en
# cifra o en palabra («dos tazas y media», «un litro y medio», «media taza»). NO casa las frases en
# proporción («dos tazas de agua POR CADA taza de arroz»): esas ya escalan solas. Tampoco cucharadas
# ni cucharaditas de agua: no son volumen de cocción.
_NUM_PALABRA = {
    "un": 1.0, "una": 1.0, "uno": 1.0, "dos": 2.0, "tres": 3.0, "cuatro": 4.0, "cinco": 5.0,
    "seis": 6.0, "siete": 7.0, "ocho": 8.0, "media": 0.5, "medio": 0.5,
}
_RE_AGUA = re.compile(
    r"(?P<num>\d+(?:[.,]\d+)?|un|una|uno|dos|tres|cuatro|cinco|seis|siete|ocho|media|medio)\s*"
    r"(?P<unidad>tazas?|litros?|ml|mililitros?)"
    r"(?P<ymedia>\s+y\s+medi[oa])?\s+de\s+agua\b"
    r"(?!\s+por\s+cada)(?!\s+por\s+taza)",
    re.IGNORECASE,
)
_FRACCIONES = {0.25: "¼", 0.5: "½", 0.75: "¾"}


def _formato_tazas(v: float) -> str:
    v = max(0.25, round(v * 4) / 4.0)
    entero, frac = int(v), round(v - int(v), 2)
    if frac == 0:
        return f"{entero} taza" if entero == 1 else f"{entero} tazas"
    sufijo = _FRACCIONES.get(frac, f"{frac:g}")
    return (f"{sufijo} taza" if entero == 0 else f"{entero}{sufijo} tazas")


def _formato_litros(v: float) -> str:
    if v < 1.0:
        ml = max(50, int(round(v * 1000 / 50.0)) * 50)
        return f"{ml} ml"
    v = round(v * 4) / 4.0
    if v == 1.0:
        return "1 litro"
    return f"{v:g}".replace(".", ",") + " litros"


def escalar_agua_en_pasos(pasos: list, factor: float) -> tuple:
    """Los pasos con el agua MEDIDA escalada por `factor`. Devuelve `(pasos, cambió)`.

    Sólo cuando el factor se aleja de 1 más de un 5 %; sólo las menciones absolutas («tres tazas de
    agua», «3 litros de agua», «media taza de agua»); las proporciones («por cada taza de arroz») y las
    cucharadas se dejan tal cual. El agua no es un ingrediente comprado, así que ningún escáner la ve
    (V4 mide gramos, V6 exige cifra): esta es la única costura donde la cantidad del paso puede seguir
    a la cantidad servida."""
    try:
        f = float(factor)
    except (TypeError, ValueError):
        return list(pasos or []), False
    if not pasos or abs(f - 1.0) < 0.05:
        return list(pasos or []), False

    def _cantidad(m) -> Optional[float]:
        raw = m.group("num").lower()
        if raw in _NUM_PALABRA:
            v = _NUM_PALABRA[raw]
        else:
            try:
                v = float(raw.replace(",", "."))
            except ValueError:
                return None
        if m.group("ymedia"):
            v += 0.5
        return v

    def _sustituir(m):
        v = _cantidad(m)
        if v is None or v <= 0:
            return m.group(0)
        u = m.group("unidad").lower()
        if u.startswith("taza"):
            texto = _formato_tazas(v * f)
        elif u.startswith("litro"):
            texto = _formato_litros(v * f)
        else:
            ml = max(10, int(round(v * f / 10.0)) * 10)
            texto = f"{ml} ml"
        return f"{texto} de agua"

    out, cambio = [], False
    for p in pasos:
        s = str(p)
        nuevo = _RE_AGUA.sub(_sustituir, s)
        cambio = cambio or (nuevo != s)
        out.append(nuevo)
    return out, cambio


#: [P1-RETINOL-PREFORMADO · 2026-09-09] El UL de vitamina A (3.000 mcg RAE/día, IOM) es de retinol
#: PREFORMADO. El beta-caroteno de la auyama o la zanahoria **no intoxica**, y el catálogo guarda los
#: dos en la misma columna (`vitamin_a_mcg_rae_per_100g`), que es RAE total.
#:
#: Medido sobre 18 días de planes vivos: 2 pasaban el UL, y los DOS por **600 g de auyama**. Un techo
#: sobre RAE total los habría rechazado — habría roto cocina dominicana legítima para «arreglar» algo
#: que no está roto. *Un guard que no distingue la fuente no mide el riesgo: mide una columna.*
#:
#: Por eso el techo se aplica SÓLO a las fuentes animales de retinol. En las 349 filas del catálogo
#: la única que importa es `Hígado de res` (4.970 mcg/100 g); las once siguientes por RAE son todas
#: vegetales (nori, pimentón, chiles, zanahoria, batata). La lista se queda corta a propósito y el
#: test fija esa frontera.
#: tooltip-anchor: _RETINOL_ANIMAL (test_p1_retinol_preformado.py)
_RETINOL_ANIMAL = ("higado", "hígado", "viscera", "víscera", "mondongo", "molleja", "riñon",
                   "riñón", "rinon", "pate", "paté", "foie", "aceite de higado")
_UL_RETINOL_MCG = 3000.0

# [P1-SODIO-DEL-DIA-DETERMINISTA · 2026-09-10] El techo ya existe en el repo y es UNO: el de
# `graph_orchestrator`. Copiar aquí un 2000 sería la segunda tabla que `P1-DIET-CANON-SSOT` prohíbe
# — dos números que empiezan iguales y divergen a la primera edición.
_TECHO_SODIO_RESPALDO_MG = 2000.0


def _techo_sodio() -> float:
    try:
        from graph_orchestrator import SODIUM_DAY_CEILING_MG
        return float(SODIUM_DAY_CEILING_MG)
    except Exception:                                                  # noqa: BLE001
        return _TECHO_SODIO_RESPALDO_MG


def _variedad_ssot():
    """[P1-DIA-DETERMINISTA-VARIEDAD-DEL-DIA · 2026-09-10] Las dos puertas de variedad del camino del
    modelo, leídas de donde viven.

    `_SAME_DAY_PROTEIN_GATE_LABELS` (carnes, pescados y HUEVO; exime queso, legumbres y yogur, que en
    RD se repiten por cultura) y `_LIGHT_BASE_TOKENS` (avena, casabe, arepa… en desayuno Y merienda).
    Medido sobre los 30 días del dueño, 12 repetían proteína y 4 base ligera. Copiarlas aquí sería la
    segunda tabla que `P1-DIET-CANON-SSOT` prohíbe.

    [P1-AUDITORIA-ARQ-VERIFICADA · 2026-09-11] Aquí decía que el día determinista no pasaba por
    `assemble_plan_node` y por eso no las heredaba. Sí pasa (ver el docstring del módulo): los autofix
    de `assemble_plan_node` corren también sobre estos días. Aplicarlas aquí, en la ELECCIÓN, evita
    servir el plato repetido y que luego lo repare un autofix a ciegas — es preferir, no descartar.
    """
    try:
        from graph_orchestrator import _SAME_DAY_PROTEIN_GATE_LABELS
        from bases_ligeras import LIGHT_BASE_TOKENS
        return frozenset(_SAME_DAY_PROTEIN_GATE_LABELS), tuple(LIGHT_BASE_TOKENS)
    except Exception:                                                  # noqa: BLE001
        return frozenset(), ()


def _repetir_proteina_ok(form_data) -> bool:
    """El permiso de repetir proteína el mismo día sale de la política compilada del usuario.

    `horizon.repetition_limits_for` es la tabla: `routine` lo permite, `balanced` y `explore` no. Sin
    política ⇒ `balanced`, que es el defecto de `horizon` — no uno inventado aquí.
    """
    try:
        import horizon
        eff = (form_data or {}).get("_plan_policy_effective") or {}
        modo = (eff.get("recurrence") or {}).get("global_mode") or "balanced"
        return bool(horizon.repetition_limits_for(modo).get("same_day_protein_repeat_ok"))
    except Exception:                                                  # noqa: BLE001
        return False


def _variedad_del_dia_on() -> bool:
    """Knob de las dos puertas de variedad del día determinista. APAGADO por defecto, por medición.

    Encendidas, en los 30 días del dueño los días con proteína repetida bajan de 12 a 0 y los de base
    ligera de 4 a 0. Pero este módulo no tiene memoria ENTRE días, y restringir el día empuja la
    elección hacia los platos exentos: las ventanas de 7 días que rompen el tope de su política
    `balanced` (`max_exact_repeat_per_7d` = 2) pasan de 9 a 28, y el peor plato de 3 a 4 veces por
    semana. Encenderlas antes de que exista esa memoria cambia una regla del dueño por otra suya.
    `MEALFIT_DETERMINISTIC_DAY_SAME_DAY_VARIETY=true` las enciende sin redeploy.
    """
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_DETERMINISTIC_DAY_SAME_DAY_VARIETY", False)
    except Exception:                                                  # noqa: BLE001
        return False


def _bases_ligeras_de(comida, tokens) -> set:
    """Las bases ligeras que nombra una comida, con el MISMO criterio que `_detect_light_base_repeats`:
    sin acentos, en minúscula, prefijo de palabra sobre nombre + ingredientes («arepa» casa «arepitas»)."""
    try:
        from constants import strip_accents as _sa
        blob = _sa((str((comida or {}).get("name") or "") + " " +
                    " ".join(str(x) for x in ((comida or {}).get("ingredients") or []))).lower())
        from bases_ligeras import familia_base_ligera as _fam
        return {_fam(t) for t in tokens or () if re.search(r"\b" + re.escape(t), blob)}
    except Exception:                                                  # noqa: BLE001
        return set()


def _sodio_de(plantilla) -> float:
    """Los mg de sodio de UNA RACIÓN DEL REGISTRY (la porción base, sin escalar). Sin dato ⇒ 0.

    [P1-AUDITORIA-ARQ-VERIFICADA · 2026-09-11] Ya no es lo que se carga al presupuesto del día: el
    plato servido está escalado por `factor` (banda 0,60–1,60) e inclinado ±35 %, y este número no lo
    sabía — el arenque guisado a 1,6× llevaba ~1.970 mg y se anotaban 1.232. Ahora el día se carga con
    `_sodio_de_comida` (el plato armado, línea a línea contra el catálogo) y esto queda como RESPALDO
    escalado cuando ninguna línea publica sodio. El 0 por ausencia sigue existiendo, pero ya no se
    disfraza de medición: `_sodio_de_comida` devuelve cuántas líneas quedaron sin dato.
    """
    try:
        return float(((plantilla or {}).get("nutrition_per_serving") or {}).get("sodium_mg") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _sodio_de_comida(comida: dict, catalogo: dict, plantilla: Optional[dict] = None,
                     factor: float = 1.0) -> tuple:
    """mg de sodio del plato ARMADO y cuántas de sus líneas no tienen dato: `(mg, sin_dato)`.

    Suma `sodium_mg_per_100g` × gramos servidos por línea de `ingredients` («160 g de Pechuga de
    pollo»). Una línea sin dato NO se cuenta como 0 a secas: se cuenta en `sin_dato`, y si NINGUNA
    línea publica sodio se usa como respaldo la ración del registry × `factor`. Un nutriente ausente
    no es cero (ARQ27-P0-03); aquí tampoco es infinito — es un número con su incertidumbre dicha.
    """
    total, sin_dato, con_dato = 0.0, 0, 0
    for linea in ((comida or {}).get("ingredients") or []):
        m = _RE_LINEA.match(str(linea).strip())
        if not m:
            sin_dato += 1
            continue
        fila = (catalogo or {}).get(m.group(2).strip()) or {}
        v = fila.get("sodium_mg_per_100g")
        if v is None:
            sin_dato += 1
            continue
        try:
            total += float(v) * float(m.group(1)) / 100.0
            con_dato += 1
        except (TypeError, ValueError):
            sin_dato += 1
    if con_dato == 0 and plantilla is not None:
        try:
            return _sodio_de(plantilla) * max(0.0, float(factor)), sin_dato
        except (TypeError, ValueError):
            return 0.0, sin_dato
    return total, sin_dato


def _retinol_preformado_mcg(meal: dict, catalogo: dict) -> float:
    """Microgramos de retinol PREFORMADO de una comida armada. Sólo cuenta las fuentes animales."""
    total = 0.0
    for linea in (meal.get("ingredients") or []):
        m = _RE_LINEA.match(str(linea).strip())
        if not m:
            continue
        try:
            g = float(m.group(1))
        except (TypeError, ValueError):
            continue
        nombre = m.group(2).strip()
        # [P1-AUDITORIA-ARQ-VERIFICADA · 2026-09-11] El mismo vocabulario que el ancla `_RETINOL_ANIMAL`
        # (aquí había una copia más corta) y por PALABRA: «pate» como subcadena habría casado
        # «empate»/«patacón» el día que entren al catálogo.
        _n = _norm(nombre)
        if not any(re.search(r"(?<![a-z])" + re.escape(_norm(t)) + r"(?![a-z])", _n)
                   for t in _RETINOL_ANIMAL):
            continue
        fila = (catalogo or {}).get(nombre) or {}
        v = fila.get("vitamin_a_mcg_rae_per_100g")
        if v:
            try:
                total += float(v) * g / 100.0
            except (TypeError, ValueError):
                pass
    return total


def verifica_comida(meal: dict, form_data: dict, catalogo: dict) -> list:
    """Las violaciones de una comida armada sin LLM. Lista vacía = se puede servir.

    Es una verificación del PLATO ARMADO, no del candidato: que `template_candidates` ya filtre por
    alérgeno y dieta no la hace redundante (al filtro se le escapan plurales como Bulgur/Pistachos —
    `P0-DEGRADED-SAFETY-SCAN`— y lo que los caza es el backstop). El día sí pasa después por
    `assemble_plan_node` y `review_plan_node` (ver el docstring del módulo); esto es la primera línea,
    no la única. Defensa en profundidad, que es como este repo trata todo lo clínico.

    [P1-AUDITORIA-ARQ-VERIFICADA · 2026-09-11] Tres cambios de semántica, los tres medidos:
      · El backstop clínico es **fail-secure**: si no se puede evaluar (import, excepción), el plato
        se rechaza con una violación sintética — igual que `clinical_backstop_for_meal` hace por
        dentro y que el path degradado reproduce. Aquí un `except: logger.debug` convertía «no pude
        evaluar» en «sin violaciones» = servible: dos rutas, el mismo validador, semánticas opuestas.
      · El escáner culinario corre COMPLETO (`culinary_contract_scan`, las 11 comprobaciones), no una
        lista a mano de 6: el camino del LLM ya lo usaba entero. Sigue siendo `warn` por contrato
        (`CULINARY_CONTRACT_GUARD`): aquí rechaza al candidato, nunca al día.
      · «No se pudo medir» se dice en `warning`, no en `debug`.
    """
    fuera = []
    try:
        import culinary_coherence as _cc
        filas = list(catalogo.values()) if catalogo else []
        m = {"meal": meal.get("meal"), "name": meal.get("name"),
             "ingredients": meal.get("ingredients"), "recipe": meal.get("recipe")}
        mini = {"days": [{"day": 1, "meals": [m]}]}
        fuera.extend(_cc.culinary_contract_scan(mini, filas) or [])
        try:
            if _cc.scan_coverage(mini, filas) is None:
                logger.warning(f"[P1-DETERMINISTIC-DAY] escáner culinario sin cobertura medible para "
                               f"{meal.get('name')!r}: el veredicto culinario de este plato no informa")
        except Exception as _f5e:                                              # noqa: BLE001
            logger.warning(f"[P1-PLAN-LOTE-6] verifica_comida: paso tragado sin rastro ({type(_f5e).__name__}: {_f5e})")
    except Exception as e:                                             # noqa: BLE001
        logger.warning(f"[P1-DETERMINISTIC-DAY] escáner culinario NO EVALUABLE ({e!r}); el plato sigue "
                       f"a juicio del backstop clínico y del review del plan")

    # Import LAZY a propósito: `clinical_backstop_for_meal` vive en `graph_orchestrator`, que
    # importa media casa. A nivel de módulo sería un ciclo y haría este archivo imposible de probar
    # sin montar el grafo entero. Mismo patrón —y misma razón— que el import lazy que `db_plans`
    # hace de `finalize_plan_data_coherence`.
    try:
        from graph_orchestrator import clinical_backstop_for_meal as _backstop
        fd = form_data or {}
        alergias = ((fd.get("health_profile") or {}).get("allergies")
                    or fd.get("allergies") or [])
        dieta = ((fd.get("health_profile") or {}).get("dietType")
                 or fd.get("dietType") or fd.get("diet_type"))
        fuera.extend(_backstop(meal, allergies=alergias, diet_type=dieta, form_data=fd) or [])
    except Exception as e:                                             # noqa: BLE001
        logger.warning(f"[P1-DETERMINISTIC-DAY] backstop clínico NO EVALUABLE para "
                       f"{meal.get('name')!r}: {type(e).__name__}: {e} — se rechaza el plato (fail-secure)")
        fuera.append(f"backstop clínico no evaluable ({type(e).__name__}): el plato no se sirve sin "
                     f"verificación de alérgenos y dieta")

    # [P1-RETINOL-PREFORMADO · 2026-09-09] Tercera capa. El techo de vitamina A YA EXISTÍA en el
    # repo (`graph_orchestrator._MICRO_CLOSER_UL`, «vit_a_mcg»: 3000) y sólo miraba a quien SUBE:
    # impide que el cerrador de micros escale hígado, y deja pasar de largo un plato que nace por
    # encima. Lo destapó un plato que metí yo hoy —hígado encebollado, 120 g = 5.964 mcg, 2× el UL,
    # servido 8 veces en 30 días— y ni el escáner culinario ni el backstop clínico lo vieron.
    # *Un techo que sólo vigila a quien sube no es un techo.*
    try:
        _ret = _retinol_preformado_mcg(meal, catalogo)
        if _ret > _UL_RETINOL_MCG:
            fuera.append(
                f"retinol preformado {_ret:.0f} mcg en un solo plato — supera el límite superior "
                f"tolerable del DÍA ({_UL_RETINOL_MCG:.0f} mcg RAE, IOM); hepatotóxico y "
                f"teratogénico acumulado")
    except Exception as e:                                             # noqa: BLE001
        # [P1-AUDITORIA-ARQ-VERIFICADA · 2026-09-11] Un techo clínico que no se puede evaluar no
        # aprueba: el plato se rechaza (el siguiente candidato lo intenta) y se dice en warning.
        logger.warning(f"[P1-RETINOL-PREFORMADO] techo NO EVALUABLE para {meal.get('name')!r}: {e!r}")
        fuera.append(f"techo de retinol no evaluable ({type(e).__name__}): el plato no se sirve sin "
                     f"verificación de vitamina A preformada")
    return fuera

def _tier_presupuesto(form_data) -> Optional[str]:
    """[P1-CANDIDATO-CON-PRECIO · 2026-09-09] El nivel de presupuesto, de la política COMPILADA
    primero y del formulario crudo como respaldo.

    Por qué no basta `form_data["budget"]`: la Fase 2 no se limita a copiarlo —resuelve `custom`,
    y en un país sin precios relaja el modo a `advisory` con su `relaxations[]`—, así que lo
    compilado es la verdad y el campo del formulario es la materia prima. `horizon` ya lee de ahí;
    leer de otro sitio sería la segunda tabla que `P1-DIET-CANON-SSOT` prohíbe.

    Y hay una razón medida: en los 5 planes vivos del dueño `plan_data` NO persiste `form_data`, y
    lo único que prueba que el presupuesto llegó es `_plan_policy.effective.budget.tier = "low"`.
    Colgar el filtro de una clave que no se puede verificar es como se despliega algo inerte.
    """
    fd = form_data or {}
    try:
        eff = fd.get("_plan_policy_effective")     # horizon.POLICY_EFFECTIVE_KEY, sin importarlo
        t = ((eff or {}).get("budget") or {}).get("tier")
        if t:
            return str(t)
    except Exception:
        pass
    b = fd.get("budget")
    return str(b) if b else None


def _candidatos(dr, country: str, franja: str, familias: list, fijados, **kw) -> tuple:
    """[P1-AUDITORIA-ARQ-VERIFICADA · 2026-09-11] Los `template_id` candidatos de una franja y de
    dónde salen: `(ids, fuente)` con fuente ∈ {`fijado`, `vivo`, `vivo_sin_familia`}.

      1. Si el run trae CandidateSet fijado (`_blueprint_slice.registry.candidates["día:franja"]`,
         ARQ27-F3) manda: los ids fijados que siguen pasando los filtros VIVOS (alergia, dieta,
         nutrientes exigidos, mercado, durabilidad), en el orden fijado. «Cambiar el registro activo no
         altera los candidatos de un run ya iniciado» — este módulo era la única ruta que lo violaba:
         consultaba el snapshot vivo y recompilar la biblioteca a mitad de plan cambiaba los platos.
      2. Sin fijados, la consulta viva por familia de proteína, en el orden del pool y sin duplicados.
         Si el pool no deja ninguna plantilla servible se pierde la FAMILIA, no el día — y queda dicho
         en la fuente, porque una restricción que se suelta en silencio es indistinguible de una que
         se cumplió.
      3. Sin familia: la consulta viva sin restricción.
    """
    k = int(kw.get("k") or 25)
    # Alergias y dieta van EXPLÍCITAS en la llamada (no dentro de `**kw`): son los dos filtros que este
    # módulo olvidó una vez (`P1-DETERMINISTIC-DAY-BACKSTOP`) y el test parser-based los busca aquí.
    exclude_allergens = kw.pop("exclude_allergens", ())
    diet = kw.pop("diet", None)

    def _ids(fam):
        return [str(c["template_id"]) for c in
                (dr.template_candidates(country, franja, fam, exclude_allergens=exclude_allergens,
                                        diet=diet, **kw) or [])
                if c.get("template_id")]

    vivos, vistos = [], set()
    for fam in (familias or [None]):
        for tid in _ids(fam):
            if tid not in vistos:
                vistos.add(tid)
                vivos.append(tid)
    fuente = "vivo"
    if familias and not vivos:
        vivos, fuente = _ids(None), "vivo_sin_familia"
        logger.info(f"[P1-DETERMINISTIC-DAY] {franja}: ninguna plantilla sirve a la familia "
                    f"{familias!r}; se elige sin familia y se deja constancia")
    if fijados:
        base = set(vivos) or set(_ids(None))
        fij = [str(t) for t in fijados if str(t) in base]
        if fij:
            return fij, "fijado"
        logger.info(f"[P1-DETERMINISTIC-DAY] {franja}: los {len(fijados)} candidatos fijados al run no "
                    f"pasan los filtros vivos; se reconsulta el registro")
    return vivos[:max(1, k)], fuente


# ---------------------------------------------------------------------------
# [P1-PLAN-LOTE-2 · 2026-09-11 · B6] Memoria ENTRE días.
#
# Medido el 09-10 sobre los 30 días del dueño: el día determinista rompía el tope de repetición exacta de
# su política (`balanced`: 2 veces por 7 días) en 9 ventanas aun con las puertas de variedad del día
# APAGADAS, y encenderlas lo subía a 28 — porque cada día se armaba sin saber qué comieron los anteriores.
# La memoria es una lista que el llamador comparte entre los días del run (`generate_days_parallel_node`
# crea las tareas en orden y este módulo corre en la primera fase síncrona de cada una, así que el día N
# ve a los N-1 anteriores) y que este módulo LEE y ACTUALIZA. Si el bloque continúa un plan ya entregado
# (`days_offset` > 0), los días persistidos se cargan UNA vez de la base y entran al principio.


def _max_repeticion_7d(form_data) -> int:
    """El tope de repetición exacta por 7 días de la política compilada del usuario (`balanced` ⇒ 2).
    La tabla es `horizon.repetition_limits_for`; aquí no se escribe otra."""
    try:
        import horizon
        eff = (form_data or {}).get("_plan_policy_effective") or {}
        modo = (eff.get("recurrence") or {}).get("global_mode") or "balanced"
        return int(horizon.repetition_limits_for(modo).get("max_exact_repeat_per_7d") or 2)
    except Exception:                                                  # noqa: BLE001
        return 2


def _dias_previos_persistidos(form_data, user_id) -> list:
    """Los días YA ENTREGADOS del plan que este bloque continúa (archivados + vivos, en orden). `[]` si
    no hay usuario, plan o base: fail-open, la memoria se queda con los días de este run."""
    try:
        import json
        uid = user_id or (form_data or {}).get("user_id")
        if not uid:
            return []
        from db import get_latest_meal_plan_with_id
        row = get_latest_meal_plan_with_id(str(uid))
        pd = row.get("plan_data") if isinstance(row, dict) else None
        if isinstance(pd, str):
            pd = json.loads(pd)
        if not isinstance(pd, dict):
            return []
        return [d for d in list(pd.get("_archived_days") or []) + list(pd.get("days") or [])
                if isinstance(d, dict)]
    except Exception:                                                  # noqa: BLE001
        return []


def _tid_de_comida(comida, indice_nombre: dict) -> Optional[str]:
    """La plantilla de un plato servido: por `_template_id`/`_recipe_template_id` (días deterministas y
    recetas congeladas) o por nombre EXACTO normalizado contra el registry (días del modelo)."""
    if not isinstance(comida, dict):
        return None
    tid = comida.get("_template_id") or comida.get("_recipe_template_id")
    if tid:
        return str(tid)
    return indice_nombre.get(_norm(comida.get("name")))


def _conteo_ventana(memoria, offset, form_data=None, user_id=None, por_id=None, ventana: int = 6) -> dict:
    """{template_id: veces servido} en los últimos `ventana` días de la memoria. Vacío si no hay memoria."""
    if not isinstance(memoria, list):
        return {}
    try:
        # Sólo un bloque que CONTINÚA un plan (offset > 0) carga lo entregado; en el primer bloque el «último
        # plan» del usuario sería otro plan, de otra semana, y contar sus repeticiones sería mentir.
        if int(offset or 0) > 0 and not any(isinstance(d, dict) and d.get("_persistido") for d in memoria):
            previos = [dict(d, _persistido=True) for d in _dias_previos_persistidos(form_data, user_id)]
            # centinela aunque no haya nada: la base se consulta UNA vez por run, no una vez por día
            memoria[0:0] = previos or [{"_persistido": True, "meals": []}]
        indice = {}
        for tid, t in (por_id or {}).items():
            n = _norm((t or {}).get("name"))
            if n:
                indice[n] = str(tid)
        conteo: dict = {}
        for d in [x for x in memoria if isinstance(x, dict)][-max(1, int(ventana)):]:
            for m in (d.get("meals") or []):
                tid = _tid_de_comida(m, indice)
                if tid:
                    conteo[tid] = conteo.get(tid, 0) + 1
        return conteo
    except Exception:                                                  # noqa: BLE001
        return {}


# [P1-AUDITORIA-ARQ-VERIFICADA · 2026-09-11] El contrato de `skeleton_day` es el del `DaySkeletonModel`
# que emite el planificador (`meal_types`, `protein_pool`); la forma del blueprint (`slots`, `protein`)
# se acepta también. `day_num` es el día RELATIVO al bloque (1-based, como lo entrega
# `generate_days_parallel_node`); el índice absoluto del plan sale de `_blueprint_slice.days_offset` y
# es el que gobierna la rotación, la cocina del día, la durabilidad y el CandidateSet fijado. Cada
# comida deja rastro: `_candidate_source`, `_sodium_mg_est` (y `_sodium_unknown_lines` cuando el
# catálogo calla), `_prep_time_source`, `_recipe_water_scaled`.
def build_day_for_skeleton(nutrition, form_data, skeleton_day, day_num, user_id=None, memoria=None):
    """Punto de entrada desde el pipeline. Devuelve un día completo o `None`.

    `None` es la respuesta segura y la más frecuente: knob apagado, sin objetivos, sin candidatos
    escalables o sin receta congelada. El llamador cae al camino del LLM, que es el estado de
    siempre — la generación de planes no puede depender de que esto acierte.

    NO usa `state` a propósito: acoplar este módulo a `PlanState` lo haría imposible de probar sin
    montar el grafo entero, y todo lo que necesita cabe en cuatro argumentos.
    """
    _uid = user_id or (form_data or {}).get("user_id") or (form_data or {}).get("_user_id")
    if not deterministic_day_for_user(_uid):
        return None
    try:
        m = (nutrition or {}).get("macros") or {}

        def _num(v):
            try:
                return float(str(v).replace("g", "").replace("kcal", "").strip().split()[0])
            except Exception:
                return 0.0

        kcal = _num((nutrition or {}).get("target_calories"))
        if kcal <= 0:
            return None
        objetivo_dia = {"kcal": kcal, "protein_g": _num(m.get("protein")),
                        "carbs_g": _num(m.get("carbs")), "fats_g": _num(m.get("fats"))}

        _fd = form_data or {}
        # El día ABSOLUTO del plan (0-based): la rebanada del bloque sabe dónde empieza.
        _sl = _fd.get("_blueprint_slice") if isinstance(_fd.get("_blueprint_slice"), dict) else {}
        try:
            _offset = int((_sl or {}).get("days_offset") or 0)
        except (TypeError, ValueError):
            _offset = 0
        try:
            day_index = _offset + max(0, int(day_num or 1) - 1)
        except (TypeError, ValueError):
            day_index = _offset

        from constants import cultural_country_for_form_data, country_for_form_data   # cocina ≠ mercado (I16)
        # La cocina de ESTE día: con mezcla de cocinas el blueprint asigna una por día; sin `day_index`
        # todos los días caían en la principal.
        country = cultural_country_for_form_data(_fd, day_index=day_index) or "DO"
        mercado = country_for_form_data(_fd)

        import dish_registry as dr
        from shopping_calculator import get_master_ingredients
        catalogo = {str(r.get("name")): r for r in (get_master_ingredients() or [])}
        if not catalogo:
            return None
        por_id = dr.templates_by_id(country) or {}
        if not por_id:
            return None

        _hp = _fd.get("health_profile") or {}
        _alergias = [str(a) for a in (_hp.get("allergies") or _fd.get("allergies") or []) if a]
        _dieta = _hp.get("dietType") or _fd.get("dietType") or _fd.get("diet_type")
        _eff = _fd.get("_plan_policy_effective") or {}
        # Los filtros que `horizon` ya pasa al MISMO `template_candidates` y este módulo omitía: los
        # nutrientes que el perfil clínico exige conocer (renal ⇒ fósforo y potasio; HTA ⇒ sodio), el
        # MERCADO donde se compra (una cocina dominicana comprada en España no puede ofrecer lo que ese
        # mercado no vende) y la durabilidad bajo compra única. Se leen de `horizon`, no se reescriben.
        _req_nutr, _dur = (), {}
        try:
            import horizon as _hz
            _req_nutr = tuple(_hz.required_nutrients(_eff) or ())
            _dur = dict(_hz._dur_kwargs(_eff, day_index) or {})
        except Exception as _e_hz:                                     # noqa: BLE001
            logger.warning(f"[P1-DETERMINISTIC-DAY] filtros de horizon no disponibles ({_e_hz!r}); "
                           f"se consulta sin nutrientes exigidos ni durabilidad")
        # [P1-PLAN-LOTE-3 · B5] los «no me gusta» del formulario y las exclusiones compiladas, al selector
        _excluidos = [str(x) for x in (list(_hp.get("dislikes") or _fd.get("dislikes") or [])
                                        + list(((_eff.get("diet") or {}).get("exclusions")) or [])) if x]
        _kw_cands = dict(k=_candidatos_k(), rotate=int(day_index), exclude_allergens=_alergias,
                         diet=_dieta, budget_tier=_tier_presupuesto(_fd), market_country=mercado,
                         require_known_nutrients=_req_nutr, exclude_foods=_excluidos, **_dur)
        _fijados = ((_sl.get("registry") or {}).get("candidates") or {}) if _sl else {}

        franjas = _franjas_del_dia(skeleton_day)
        if not franjas:
            return None            # una franja que no sabemos repartir: que la haga el LLM
        familias = _familias_del_dia(skeleton_day)
        meals = []
        usadas_hoy = set()   # [P1-DIA-DETERMINISTA-VARIEDAD] ninguna plantilla dos veces el mismo día
        _sodio_dia = 0.0     # [P1-SODIO-DEL-DIA-DETERMINISTA] presupuesto del DÍA, no del plato
        _sodio_sin_dato = 0  # líneas servidas sin sodio en el catálogo: el presupuesto es cota INFERIOR
        # [P1-DIA-DETERMINISTA-VARIEDAD-DEL-DIA · 2026-09-10] Las dos puertas de variedad del camino del
        # modelo: proteína que fatiga repetida el mismo día, y la misma base ligera en desayuno y merienda.
        _proteinas_hoy, _bases_hoy = set(), set()
        _labels_var, _tokens_var = _variedad_ssot()
        _repite_ok = _repetir_proteina_ok(form_data)
        _variedad_on = _variedad_del_dia_on()
        # [P1-PLAN-LOTE-2 · 2026-09-11 · B6] Lo servido en los 6 días anteriores y el tope de repetición
        # exacta por 7 días de la política (`balanced` ⇒ 2). Se PREFIERE al fresco, no se descarta al repetido.
        _saturados = _conteo_ventana(memoria, _offset, _fd, _uid, por_id)
        _max_rep = _max_repeticion_7d(_fd)
        for etiqueta, slot, r in franjas:
            obj = {k: v * r for k, v in objetivo_dia.items()}
            # [P1-CANDIDATO-CON-PRECIO · 2026-09-09] Éste es el ÚNICO camino donde el candidato se
            # convierte en plato sin que el modelo pueda ignorarlo: sin el tier aquí, el filtro de
            # precio sólo aconseja. Los demás filtros viajan en `_kw_cands`, los mismos que `horizon`.
            tids, _fuente = _candidatos(dr, country, slot, familias,
                                        _fijados.get(f"{day_index}:{slot}"), **_kw_cands)
            # [P1-DIA-DETERMINISTA-VARIEDAD · 2026-09-09] La lista, no el ganador: `rotacion` mueve
            # la cabeza por día (medido: 7 platos distintos en 56 comidas cuando era siempre el
            # mejor), y si al elegido le falta la receta congelada se prueba el siguiente en vez de
            # tirar el DÍA ENTERO al LLM por un plato.
            comida = None
            _ultimo_motivo = None
            _reserva = None
            _reserva_var = None
            _elegibles = elegir_plantillas(tids, obj, catalogo, por_id, slot,
                                           rotacion=_rotacion_de(day_index, slot),
                                           saturados=_saturados, max_rep=_max_rep)
            # Primero los que no se han servido hoy; los ya usados quedan de RESPALDO al final, no
            # descartados: quedarse sin día por no repetir es peor que repetir.
            for _t, _f in sorted(_elegibles, key=lambda p: str(p[0].get("template_id")) in usadas_hoy):
                _c = construir_comida(_t, _f, catalogo, slot, country, obj)
                if not _c:
                    continue
                _c["meal"] = etiqueta          # la etiqueta del esqueleto, tal cual («Merienda AM»)
                # [P1-CATALOGO-PROTEINA-DESAYUNO · 2026-09-09] La verificación va DENTRO del bucle.
                # Estaba fuera, así que un plato que se construía pero no pasaba el escáner tiraba
                # el DÍA ENTERO en vez de ceder el turno al siguiente candidato — y sólo se notó al
                # ensanchar los elegibles: 14/14 días pasaron a 12/14 justo cuando había MÁS donde
                # elegir. Un guard que descarta el conjunto en vez del elemento castiga la abundancia.
                _viol = verifica_comida(_c, form_data or {}, catalogo)
                if _viol:
                    # Las dos capas hablan idiomas distintos y hay que respetarlo: el escáner
                    # culinario devuelve dicts con `check`, el backstop clínico devuelve STRINGS
                    # legibles. Un `.get()` a secas revienta sobre la cadena, la excepción se traga
                    # el aviso y el operador se queda sin el motivo del rechazo.
                    _ultimo_motivo = (comida_nombre := _c.get("name"), sorted({
                        (v.get("check") or v.get("detail") or "?") if isinstance(v, dict) else str(v)
                        for v in _viol})[:4])
                    logger.debug(f"[P1-DETERMINISTIC-DAY] {slot}: descartado {comida_nombre!r} "
                                 f"por {_ultimo_motivo[1]}; se prueba el siguiente candidato")
                    continue
                # [P1-SODIO-DEL-DIA-DETERMINISTA · 2026-09-10] Entre los que YA pasaron todo lo
                # demás, prefiere el que deja el día por debajo del techo. Preferir y no descartar
                # es deliberado: el sodio es un presupuesto del DÍA, no un veneno del plato, y un
                # guard que tira candidatos sanos por una cuenta acumulada castiga al último slot.
                # [P1-AUDITORIA-ARQ-VERIFICADA · 2026-09-11] El sodio es el del plato SERVIDO
                # (escalado e inclinado), línea a línea contra el catálogo — no la ración base del
                # registry, que ignoraba el factor (hasta 1,6×) y disfrazaba de 0 lo desconocido.
                _na, _na_sin = _sodio_de_comida(_c, catalogo, _t, _f)
                _c["_sodium_mg_est"] = int(round(_na))
                if _na_sin:
                    _c["_sodium_unknown_lines"] = int(_na_sin)
                if _sodio_dia + _na > _techo_sodio():
                    # [P1-DIA-DETERMINISTA-VARIEDAD-DEL-DIA] La condición era `… > techo and
                    # _reserva is None`: el PRIMER salado quedaba de reserva y el SEGUNDO se aceptaba
                    # de largo. Todo el que se pasa se salta. [P1-AUDITORIA-ARQ-VERIFICADA] Y de
                    # reserva queda el MENOS salado, no el primero: la lista viene ordenada por
                    # macros, nunca por sodio, y servir de respaldo al más salado castigaba el día.
                    if _reserva is None:
                        _reserva = (_c, _t, _na)
                    elif _na < _reserva[2]:
                        _reserva = (_c, _t, _na)
                    continue
                # [P1-DIA-DETERMINISTA-VARIEDAD-DEL-DIA · 2026-09-10] Las dos puertas de variedad del
                # modelo, con el mismo patrón: el que choca queda de reserva y se prueba el siguiente.
                _prot = str(_t.get("protein") or "")
                _bl = _bases_ligeras_de(_c, _tokens_var) if slot in ("desayuno", "merienda") else set()
                if _variedad_on and ((not _repite_ok and _prot in _labels_var and _prot in _proteinas_hoy)
                                     or (_bl & _bases_hoy)):
                    if _reserva_var is None:
                        _reserva_var = (_c, _t, _na)
                    continue
                comida, _t_srv, _na_srv = _c, _t, _na
                break
            if comida is None:
                # Entre reservas, primero la de variedad (sólo repite) y después la de sodio (el día
                # sigue pasando por el juicio final de sodio).
                _res = _reserva_var or _reserva
                if _res is not None:
                    comida, _t_srv, _na_srv = _res
                    logger.debug(f"[P1-DIA-DETERMINISTA-VARIEDAD-DEL-DIA] {slot}: ningún candidato limpio; "
                                 f"se sirve {comida.get('name')!r} de reserva y el día queda a juicio final")
            if not comida:
                logger.warning(
                    f"[P1-DETERMINISTIC-DAY] día {day_num} RECHAZADO en {slot}: ninguno de los "
                    f"{len(_elegibles)} candidatos pasó → cae al LLM. Último motivo: {_ultimo_motivo}")
                return None
            comida["_candidate_source"] = _fuente
            # Lo servido —y sólo lo servido— entra en las cuentas del día, en UN sitio.
            usadas_hoy.add(str(_t_srv.get("template_id")))
            _sodio_dia += _na_srv
            _sodio_sin_dato += int(comida.get("_sodium_unknown_lines") or 0)
            if str(_t_srv.get("protein") or "") in _labels_var:
                _proteinas_hoy.add(str(_t_srv.get("protein")))
            if slot in ("desayuno", "merienda"):
                _bases_hoy |= _bases_ligeras_de(comida, _tokens_var)
            meals.append(comida)
        if not meals:
            return None
        # [P1-SODIO-DEL-DIA-DETERMINISTA · 2026-09-10] La última palabra local. El techo de sodio del
        # repo (`SODIUM_DAY_CEILING_MG`, OMS 2.000 mg) y su autofix viven en `assemble_plan_node`, por
        # donde este día TAMBIÉN pasa; devolver `None` aquí —«que lo haga el LLM»— sigue siendo más
        # barato que servir un día que ya sabemos que rompe el techo y dejar que un autofix lo repare
        # a ciegas. Medido sobre los 30 días reales del dueño: 13 de 30 por encima del techo con la
        # sal declarada; acotarla a 0,5 g bajó eso a 1 de 30 (el arenque, salado de ORIGEN).
        if _sodio_dia > _techo_sodio():
            logger.warning(f"[P1-SODIO-DEL-DIA-DETERMINISTA] día {day_num} RECHAZADO: "
                           f"{_sodio_dia:.0f} mg de sodio > techo {_techo_sodio():.0f} mg → cae al LLM")
            return None
        logger.info(f"[P1-DETERMINISTIC-DAY] día {day_num} armado sin LLM: "
                    f"{len(meals)} comidas, {sum(m['calories'] for m in meals)} kcal, "
                    f"{_sodio_dia:.0f} mg de sodio"
                    + (f" (cota inferior: {_sodio_sin_dato} líneas sin dato)" if _sodio_sin_dato else ""))
        dia = {"day": day_num, "meals": meals, "_day_source": "deterministic",
               "_day_index": int(day_index), "_sodium_mg_est": int(round(_sodio_dia))}
        if _sodio_sin_dato:
            dia["_sodium_unknown_lines"] = int(_sodio_sin_dato)
        if isinstance(memoria, list):
            memoria.append(dia)      # quien lee la memoria la actualiza: el día siguiente ya cuenta con éste
        return dia
    except Exception as e:                                          # noqa: BLE001
        logger.debug(f"[P1-DETERMINISTIC-DAY] no-op para el día {day_num}: {e!r}")
        return None
