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

**Pero un día que sale de aquí se persiste sin pasar por `assemble_plan_node`**, igual que el path
degradado. `P0-DEGRADED-SAFETY-SCAN` enseñó qué cuesta eso: al filtro se le escapan plurales y lo
que los caza es el backstop clínico. Por eso `verifica_comida` corre las seis capas del
escáner culinario Y el backstop clínico sobre CADA plato antes de devolver el día, y devolver
`None` es seguro — el llamador cae al camino del LLM, que es el estado de siempre.

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

# Inclinación de constituyentes: ±35 % por ingrediente y nunca por debajo del 30 % del gramaje
# original. «Lentejas guisadas con arroz» con un 30 % menos de arroz sigue siendo ese plato; con un
# 70 % menos, no. Y bajar a cero además dejaría el ingrediente comprado y sin usar en la receta —
# el huérfano V3 que el escáner ya caza.
_TILT_TOPE = 0.35
_TILT_MIN_FRAC = 0.30

# Reparto tipico del dia. Vive aqui y no en el llamador porque es parte del contrato de este
# modulo: si una franja no esta en la tabla, el dia NO se arma en vez de repartirla a ojo.
_REPARTO = {"desayuno": 0.25, "almuerzo": 0.35, "cena": 0.30, "merienda": 0.10}

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
    n = _norm(nombre)
    return any(k in n for k in _NO_ESCALAN)


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


def elegir_plantilla(tids, objetivo, catalogo: dict, por_id: dict, slot: str = ""):
    """El candidato que mejor llega al objetivo de MACROS tras escalar a sus calorías.

    `objetivo` = {"kcal","protein_g","carbs_g","fats_g"} de ESTA franja.

    La proteína pesa el doble en el score a propósito: es la que tiene consecuencia clínica y la
    que el escalado uniforme no puede arreglar (multiplicar por 1,2 sube los tres macros a la vez y
    no cambia sus proporciones). Las calorías no entran en el score porque quedan clavadas por
    construcción.

    Determinista: se ordena por score y se desempata por `template_id`. Desempatar por el orden de
    la lista dependería de cómo vino la lista; por id no depende de nada.
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
        cands.append((round(score, 6), str(tid), t, f))
    if not cands:
        return None
    cands.sort(key=lambda x: (x[0], x[1]))
    _, _, t, f = cands[0]
    return t, f


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
            meal["recipe"] = list(pasos)
            meal["_recipe_source"] = "library"
    except Exception:
        pass
    if not meal.get("recipe"):
        return None      # sin receta congelada no hay determinismo del texto: que lo haga el LLM
    return meal


def verifica_comida(meal: dict, form_data: dict, catalogo: dict) -> list:
    """Las violaciones de una comida armada sin LLM. Lista vacía = se puede servir.

    Un día que sale de este módulo se persiste **sin pasar por `assemble_plan_node`**: ni reviewer
    médico, ni capa clínica determinista, ni los scans de alérgeno y dieta. Es la misma clase de
    superficie que `P0-DEGRADED-SAFETY-SCAN` cerró para el path degradado, y la lección de aquel
    P-fix es literal: al filtro de arriba se le escapan cosas (plurales como Bulgur/Pistachos) y lo
    que las caza es el backstop.

    Que `template_candidates` ya filtre por alérgeno y dieta NO hace esto redundante: eso es un
    filtro de CANDIDATOS y esto es una verificación del PLATO ARMADO. Defensa en profundidad, que
    es como este repo trata todo lo clínico.
    """
    fuera = []
    try:
        import culinary_coherence as _cc
        idx = _cc.build_culinary_index(list(catalogo.values()) if catalogo else [])
        m = {"meal": meal.get("meal"), "name": meal.get("name"),
             "ingredients": meal.get("ingredients"), "recipe": meal.get("recipe")}
        for capa in ("_v1_verbo_alimento", "_v2_estado_imposible", "_v3_huerfanos",
                     "_v4_cantidad_inconsistente", "_v5_paso_usa_lo_que_no_esta",
                     "_v6_paso_pide_mas_que_la_lista"):
            fn = getattr(_cc, capa, None)
            if fn is None:
                continue
            try:
                fuera.extend(fn({}, m, idx) or [])
            except Exception:                                          # noqa: BLE001
                pass
    except Exception as e:                                             # noqa: BLE001
        logger.debug(f"[P1-DETERMINISTIC-DAY] escáner culinario no-op: {e!r}")

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
        logger.debug(f"[P1-DETERMINISTIC-DAY] backstop clínico no-op: {e!r}")
    return fuera

def build_day_for_skeleton(nutrition, form_data, skeleton_day, day_num, user_id=None):
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

        from constants import cultural_country_for_form_data      # la COCINA, no el mercado (I16)
        country = cultural_country_for_form_data(form_data or {}) or "DO"

        import dish_registry as dr
        from shopping_calculator import get_master_ingredients
        catalogo = {str(r.get("name")): r for r in (get_master_ingredients() or [])}
        if not catalogo:
            return None
        por_id = dr.templates_by_id(country) or {}
        if not por_id:
            return None

        _fd = form_data or {}
        _hp = _fd.get("health_profile") or {}
        _alergias = [str(a) for a in (_hp.get("allergies") or _fd.get("allergies") or []) if a]
        _dieta = _hp.get("dietType") or _fd.get("dietType") or _fd.get("diet_type")

        slots = [s for s in (skeleton_day or {}).get("slots") or []] or list(_REPARTO)
        meals = []
        for slot in slots:
            r = _REPARTO.get(_norm(slot))
            if not r:
                return None            # una franja que no sabemos repartir: que la haga el LLM
            obj = {k: v * r for k, v in objetivo_dia.items()}
            # Los filtros VAN AQUI. El docstring de este modulo decia que el CandidateSet ya
            # filtraba por alergia y dieta — y es verdad de `_registry_slice`, pero esta funcion
            # llama a `template_candidates` DIRECTAMENTE y no se los pasaba. Lo cazó el backstop
            # clinico rechazando un desayuno con huevo a un alergico al huevo: la defensa en
            # profundidad funcionó, y precisamente por eso el hueco de arriba hay que cerrarlo —
            # una última linea de defensa que trabaja sola dejó de ser defensa en profundidad.
            tids = [c["template_id"] for c in
                    dr.template_candidates(country, slot, (skeleton_day or {}).get("protein"),
                                           k=_candidatos_k(), rotate=int(day_num or 0),
                                           exclude_allergens=_alergias, diet=_dieta)]
            el = elegir_plantilla(tids, obj, catalogo, por_id, slot)
            if not el:
                return None
            comida = construir_comida(el[0], el[1], catalogo, slot, country, obj)
            if not comida:
                return None
            _viol = verifica_comida(comida, form_data or {}, catalogo)
            if _viol:
                # Las dos capas hablan idiomas distintos y hay que respetarlo: el escáner culinario
                # devuelve dicts con `check`, el backstop clínico devuelve STRINGS legibles. Un
                # `.get()` a secas revienta sobre la cadena, la excepción se traga el aviso y el
                # operador se queda sin el motivo del rechazo — que es exactamente el modo de fallo
                # que `P2-ALERT-MESSAGE-REFRESH` cerró esta misma mañana.
                _motivos = sorted({
                    (v.get("check") or v.get("detail") or "?") if isinstance(v, dict) else str(v)
                    for v in _viol})[:4]
                logger.warning(
                    f"[P1-DETERMINISTIC-DAY] día {day_num} RECHAZADO en {slot} "
                    f"({comida.get('name')}) → cae al LLM. Motivos: {_motivos}")
                return None
            meals.append(comida)
        if not meals:
            return None
        logger.info(f"[P1-DETERMINISTIC-DAY] día {day_num} armado sin LLM: "
                    f"{len(meals)} comidas, {sum(m['calories'] for m in meals)} kcal")
        return {"day": day_num, "meals": meals, "_day_source": "deterministic"}
    except Exception as e:                                          # noqa: BLE001
        logger.debug(f"[P1-DETERMINISTIC-DAY] no-op para el día {day_num}: {e!r}")
        return None
