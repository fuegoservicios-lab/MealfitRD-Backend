# -*- coding: utf-8 -*-
"""[P1-RECIPE-LIBRARY-SELECT · 2026-09-08] Recetas escritas UNA vez, seleccionadas en vez de inventadas.

Hoy el motor le pide a un LLM la receta de cada plato **en el momento**: cuesta cada vez, se espera,
y el mismo plato sale escrito distinto cada vez. `recipe_library_do_v1.json` tiene las 140 recetas
del registry de RD ya escritas y revisadas; este módulo es el enganche.

## El estado, medido y sin adornos (2026-09-08)

La cadena que propone platos del catálogo —política → blueprint → rebanada → CandidateSet— **funciona
de punta a punta**: montada a mano sobre un formulario sintético propone 61 platos del registry para
7 días, y **los 61 tienen su receta escrita**. Cero huecos.

Lo que falta no es maquinaria, son tres interruptores:

  1. `MEALFIT_PLAN_POLICY_MODE` nace en `off`, así que hoy no hay blueprint ni CandidateSet y el
     modelo inventa libre. Verificado: 94 de 95 planes vivos sin sello de política.
  2. El prompt dice «elige uno de estos **o una variante equivalente**». Con esa frase el modelo
     puede irse del catálogo aunque la política esté encendida — y entonces el plato no casa con
     ninguna plantilla y no hay receta que enganchar. `library_select_enabled()` la endurece.
  3. Este módulo, que resuelve NOMBRE de plato → `template_id` → receta.

## Por qué nace apagado

Encenderlo cambia lo que el usuario lee en cada plato. El knob (`MEALFIT_RECIPE_LIBRARY_SELECT`,
default `False`) deja el código vivo y el comportamiento intacto, que es la convención de la casa:
lo que puede necesitar revertirse sin redeploy va como knob, no como hardcode.

Con el knob apagado, este módulo no se llama desde ninguna ruta de generación.
"""
from __future__ import annotations

import json
import re
import logging
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DIR = Path(__file__).resolve().parent / "data" / "registry"


def library_select_enabled() -> bool:
    """Knob maestro. Apagado ⇒ el motor sigue pidiéndole la receta al LLM, byte-idéntico a hoy."""
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_RECIPE_LIBRARY_SELECT", False)
    except Exception:
        return False


def _norm(s) -> str:
    """Normaliza un nombre de plato para compararlo: sin acentos, sin puntuación, minúsculas.

    NO usa `pantry_names_match`: aquí se compara el nombre de un PLATO contra el de una plantilla,
    no la identidad de un alimento. Confundir las dos capas es la clase de error que este repo pagó
    en `P1-PANTRY-NAME-RESOLUTION`.
    """
    t = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return " ".join("".join(c if c.isalnum() else " " for c in t).split())


@lru_cache(maxsize=8)
def _library(country: str = "DO") -> dict:
    p = _DIR / f"recipe_library_{str(country).lower()}_v1.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("por_id") or {}
    except Exception as e:
        logger.warning(f"[P1-RECIPE-LIBRARY-SELECT] biblioteca {country} ilegible: {e!r}")
        return {}


@lru_cache(maxsize=8)
def _name_index(country: str = "DO") -> dict:
    """`nombre normalizado` → `template_id`, sólo para las plantillas que TIENEN receta."""
    p = _DIR / f"dish_registry_{str(country).lower()}_v1.json"
    if not p.exists():
        return {}
    try:
        reg = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    lib = _library(country)
    idx = {}
    for t in reg.get("templates") or []:
        tid = t.get("template_id")
        if tid not in lib:
            continue
        nom = ((t.get("editorial") or {}).get("display_name") or {}).get("es") or t.get("name")
        if nom:
            idx[_norm(nom)] = tid
    return idx


def recipe_for_dish_name(name, country: str = "DO") -> Optional[list]:
    """Los pasos escritos para ese plato, o `None` si no es una plantilla con receta.

    Coincidencia EXACTA sobre el nombre normalizado, a propósito. Un parecido del 70 % serviría la
    receta de otro plato — y servir la receta equivocada es peor que no servir ninguna, que es la
    doctrina que este repo ya aplica a la lista de compras (`_remove_one_raw_line_by_food`).
    """
    if not library_select_enabled():
        return None
    tid = _name_index(country).get(_norm(name))
    if not tid:
        return None
    pasos = (_library(country).get(tid) or {}).get("pasos")
    return list(pasos) if isinstance(pasos, list) and pasos else None


def coverage(country: str = "DO") -> dict:
    """Cuántas plantillas del registry tienen receta. Para telemetría y para el test."""
    return {"con_receta": len(_name_index(country)), "recetas": len(_library(country))}


# ---------------------------------------------------------------------------
# [P1-LIBRARY-SEAM · 2026-09-08] La costura: del plato del plan a su receta congelada.
#
# `recipe_for_dish_name` existía desde el 08-sep con **cero call sites en producción** — sólo su
# definición y sus tests. La biblioteca de 140 recetas, escrita en cinco rondas de juicio a ciegas
# del dueño, con su firma curatorial y sus tests, no servía ni un paso a ningún usuario: encender
# `MEALFIT_RECIPE_LIBRARY_SELECT` sólo endurecía una frase del prompt.
#
# Y medido antes de escribir esto: contra 808 comidas de 60 planes vivos, la costura sola daría
# **0 recetas congeladas**. Mientras el modelo invente el nombre del plato no hay coincidencia
# posible. Por eso esto viaja con `deterministic_day.py`, que es quien hace que los nombres vengan
# del registro — la costura sin el selector es un cable a ninguna parte.
#
# DOS GUARDS, los dos nacidos de fallos ya cometidos en este repo:
#
#   1. **Conjunto de alimentos idéntico.** La receta habla de los alimentos de SU plantilla. Si el
#      plato servido trae otros, servirla sería pedir lo que no hay (V5) o dejar comprado lo que
#      nadie usa (V3) — las dos familias que el escáner ya caza. Se exige que el conjunto COINCIDA,
#      no que se parezca: la doctrina de `recipe_for_dish_name` («un parecido del 70 % serviría la
#      receta de otro plato») aplicada un nivel más arriba. En la medición del 08-sep hubo
#      exactamente un caso —«Huevos revueltos con cebolla y casabe»— donde el nombre casaba y el
#      conjunto no: sin este guard le habríamos servido una receta que habla de comida ausente.
#
#   2. **Idempotencia con rastro.** `_recipe_source` dice de dónde vino el texto. Sin él, un
#      segundo pase no sabría si ya sustituyó y —peor— nadie podría medir cuántas comidas reciben
#      receta congelada en producción. La lección del día: lo inerte y lo que funciona se ven igual
#      desde fuera si no dejan huella.
#
# Fail-open en todo: cualquier excepción deja la receta del modelo, que es el estado de siempre.


def _foods_de_plantilla(tid: str, country: str):
    """Los alimentos de la plantilla, normalizados. `None` si no se puede saber."""
    try:
        import dish_registry as dr
        for t in ((dr.load_registry(country) or {}).get("templates") or []):
            if t.get("template_id") == tid:
                return frozenset(
                    _norm(c.get("name") or c.get("canonical") or "")
                    for c in (t.get("constituents") or [])
                    if (c.get("name") or c.get("canonical")))
    except Exception:
        return None
    return None


def _foods_de_comida(meal: dict) -> frozenset:
    """Los alimentos del plato servido: se le quita la cantidad a la línea humanizada
    («160 g de Pechuga de pollo» → «pechuga de pollo»)."""
    fuera = set()
    for ing in (meal.get("ingredients") or []):
        s = str(ing)
        m = re.match(r"^\s*[\d.,/]+\s*[a-zA-Z\u00f1]*\s+de\s+(.+)$", s)
        fuera.add(_norm(m.group(1) if m else s))
    return frozenset(x for x in fuera if x)


def apply_library_recipe(meal, country: str = "DO") -> bool:
    """Sustituye los pasos del plato por los de la biblioteca. `True` si sustituyó.

    Los pasos NO llevan cantidades a propósito (ver `procedencia.sin_cantidades`), así que valen
    para cualquier porción: por eso esta sustitución no toca gramos ni macros, y por eso sigue
    valiendo después de que `deterministic_day` escale e incline los constituyentes.
    """
    if not library_select_enabled() or not isinstance(meal, dict):
        return False
    if meal.get("_recipe_source") == "library":
        return False
    try:
        tid = _name_index(country).get(_norm(meal.get("name")))
        if not tid:
            return False
        pasos = (_library(country).get(tid) or {}).get("pasos")
        if not pasos:
            return False
        esperados = _foods_de_plantilla(tid, country)
        if esperados is None or _foods_de_comida(meal) != esperados:
            return False
        meal["recipe"] = list(pasos)
        meal["_recipe_source"] = "library"
        meal["_recipe_template_id"] = tid
        _escalar_agua_de_biblioteca(meal, tid, country)
        return True
    except Exception:
        return False


# [P1-PLAN-FASE-A · 2026-09-11 · A7] La receta congelada está escrita para la ración BASE de la plantilla y 19
# de las 190 dicen agua MEDIDA («tres tazas de agua»); el plato del camino LLM llega ya escalado a su franja.
# `deterministic_day` escala esa agua con el factor que conoce (P1-AUDITORIA-ARQ-VERIFICADA); aquí no hay
# factor explícito, así que se DERIVA: gramos servidos frente a gramos de la plantilla, sobre los constituyentes
# que el plato trae en gramos. Si no se puede derivar y la receta trae agua medida, el plato queda marcado
# `_recipe_water_unscaled=True`: una cantidad que no se pudo ajustar se declara, no se disimula.
# Medido al escribirlo: `apply_library_recipe` no tiene llamadores en producción (0 de 60 comidas LLM de los
# últimos 14 días casan por nombre con la biblioteca); esto deja la costura correcta para cuando se cablee.
_RE_GRAMOS_LINEA = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*(?:g|gr|grs|gramos?)\s+de\s+(.+?)\s*$", re.IGNORECASE)


def _gramos_por_alimento(meal: dict) -> dict:
    """{alimento normalizado: gramos} de las líneas del plato que vienen en gramos («160 g de Pechuga de
    pollo»). Las líneas en otras unidades («2 huevos») no entran: sin gramos no hay factor."""
    out: dict = {}
    lineas = meal.get("ingredients_raw") if isinstance(meal.get("ingredients_raw"), list) else meal.get("ingredients")
    for s in (lineas or []):
        m = _RE_GRAMOS_LINEA.match(str(s))
        if not m:
            continue
        try:
            k = _norm(m.group(2))
            out[k] = out.get(k, 0.0) + float(m.group(1).replace(",", "."))
        except ValueError:
            continue
    return out


def _factor_implicito(meal: dict, tid: str, country: str) -> Optional[float]:
    """Cuánto más (o menos) plato hay que en la plantilla base: gramos servidos / gramos de la plantilla,
    sobre los constituyentes que el plato trae en gramos. `None` si ninguno se puede comparar o si la
    comparación cubre menos de la mitad de la plantilla: un factor a medias es peor que ninguno."""
    try:
        import dish_registry as dr
        t = (dr.templates_by_id(country) or {}).get(str(tid))
        if not t:
            return None
        servidos = _gramos_por_alimento(meal)
        base_total = comp_base = comp_serv = 0.0
        for c in (t.get("constituents") or []):
            g = float(c.get("grams") or 0.0)
            if g <= 0:
                continue
            base_total += g
            for k in (c.get("name"), c.get("canonical")):
                if k and _norm(k) in servidos:
                    comp_base += g
                    comp_serv += servidos[_norm(k)]
                    break
        if base_total <= 0 or comp_base <= 0 or comp_base < 0.5 * base_total:
            return None
        return comp_serv / comp_base
    except Exception:
        return None


def _escalar_agua_de_biblioteca(meal: dict, tid: str, country: str) -> None:
    """Escala el agua medida de la receta recién puesta con el factor implícito; marca lo que no pudo."""
    try:
        from deterministic_day import escalar_agua_en_pasos, _RE_AGUA
        pasos = meal.get("recipe") or []
        if not any(_RE_AGUA.search(str(p)) for p in pasos):
            return
        f = _factor_implicito(meal, tid, country)
        if f is None or f <= 0:
            meal["_recipe_water_unscaled"] = True
            return
        nuevos, cambio = escalar_agua_en_pasos(pasos, f)
        meal["_recipe_scale_factor"] = round(float(f), 3)
        if cambio:
            meal["recipe"] = nuevos
            meal["_recipe_water_scaled"] = True
    except Exception:
        return


def apply_library_recipes_to_days(days, country: str = "DO") -> int:
    """Aplica la sustitución a todas las comidas de todos los días. Devuelve cuántas cambiaron."""
    if not library_select_enabled():
        return 0
    n = 0
    for d in (days or []):
        for m in ((d.get("meals") or []) if isinstance(d, dict) else []):
            if isinstance(m, dict) and apply_library_recipe(m, country):
                n += 1
    return n


# ---------------------------------------------------------------------------
# [P1-AUDITORIA-ARQ-VERIFICADA · 2026-09-11] El tiempo del plato, del registry y no del relleno.
#
# `assemble_plan_node` completaba `prep_time` ausente con «15 min»: un dato inventado con la misma
# pinta que uno medido, para «La bandera» (105 min declarados por su receta) igual que para un jugo
# (5). El registry lleva desde P1-MINUTOS-DE-LA-RECETA el número que la receta DECLARA
# (`logistics.prep_minutes_est`, fuente `receta`) o la estimación por técnica (`tecnica`) — y ningún
# código de producción lo leía. Aquí se lee. El relleno `defecto` (30) NO se devuelve: sería otro
# número inventado con mejor disfraz.


def prep_time_for_meal(meal, country: str = "DO") -> Optional[str]:
    """«N min» si el registry conoce el plato y su tiempo viene de la receta o de la técnica; `None` si no.

    Resuelve por `_template_id`/`_recipe_template_id` (platos del camino determinista) y, si no, por
    nombre EXACTO normalizado — la doctrina de `recipe_for_dish_name`: un parecido serviría el tiempo
    de otro plato. Fail-open a `None`: sin dato no hay dato."""
    if not isinstance(meal, dict):
        return None
    try:
        import dish_registry as dr
        idx = dr.templates_by_id(country) or {}
        tid = meal.get("_template_id") or meal.get("_recipe_template_id")
        t = idx.get(str(tid)) if tid else None
        if t is None:
            tid2 = _registry_name_index(country).get(_norm(meal.get("name")))
            t = idx.get(str(tid2)) if tid2 else None
        if not t:
            return None
        lg = t.get("logistics") or {}
        if str(lg.get("prep_minutes_source") or "") not in ("receta", "tecnica"):
            return None
        n = int(lg.get("prep_minutes_est") or 0)
        return f"{n} min" if n > 0 else None
    except Exception:
        return None


def fill_prep_time(meal, form_data=None) -> None:
    """Completa `prep_time` en el plato SIN inventar: del registry si lo sabe, si no cadena vacía.

    `_prep_time_source` dice de dónde salió (`registry` | `unknown`). El frontend (`RecipesView`,
    `Dashboard`, PDF) pinta el chip sólo si `prep_time` es truthy, así que el vacío se OCULTA en vez
    de mentir. `MealModel.prep_time` es `str`: la cadena vacía valida; `None` no."""
    if not isinstance(meal, dict) or meal.get("prep_time"):
        return
    country = "DO"
    try:
        from constants import cultural_country_for_form_data
        country = cultural_country_for_form_data(form_data or {}) or "DO"
    except Exception:
        pass
    pt = prep_time_for_meal(meal, country)
    if not pt and country != "DO":
        pt = prep_time_for_meal(meal, "DO")
    meal["prep_time"] = pt or ""
    meal["_prep_time_source"] = "registry" if pt else "unknown"


# ---------------------------------------------------------------------------
# [P1-FIDELIDAD-PLATO-DEL-REGISTRY · 2026-09-09] El medidor de procedencia.
#
# Medido en producción el 09-sep, sobre un plan recién generado contra el registry de 179
# plantillas, con la política en `enforce` y `registry_in_prompt: true`: **0 de 12 platos salieron
# del catálogo**, y el informe de fidelidad de ese mismo plan puntuó **1.0 con `issues: []`**.
#
# No se contradicen: la fidelidad mide anclas, repetición y el contrato de la rebanada — nunca la
# IDENTIDAD del plato. Es la forma exacta del defecto que este repo ya nombró dos veces («un
# veredicto que no puede fallar no informa», «el gate cuenta VEREDICTOS no DESTINOS»): mientras
# nadie cuente los platos, encender o no `MEALFIT_RECIPE_LIBRARY_SELECT` es una decisión a ciegas.
#
# Tres niveles, porque la costura tiene tres condiciones y el dueño merece el número honesto:
#
#   · `del_registry` — el nombre resuelve a una plantilla. Procedencia.
#   · `con_receta`   — esa plantilla además tiene pasos escritos.
#   · `aplicables`   — nombre Y conjunto de alimentos coinciden ⇒ la costura sustituiría de verdad.
#
# `aplicables` es el único que predice el rendimiento real de encender el knob; los otros dos
# separan «no está en el catálogo» de «está pero la comida servida es otra».
#
# **No cuelga de `library_select_enabled()`, a propósito.** Un medidor gateado por el interruptor
# que existe para informar sólo sabe medir después de haber decidido — que es justo cuando ya no
# hace falta. Es la misma trampa que `P1-I18N-DEAD-VEREDICTO`: la defensa reprodujo dentro de sí
# el defecto que venía a cerrar.


@lru_cache(maxsize=8)
def _registry_name_index(country: str = "DO") -> dict:
    """`nombre normalizado` → `template_id` de TODAS las plantillas, tengan receta o no.

    Hermano de `_name_index`, que filtra a las que SÍ tienen pasos. La diferencia entre ambos es
    exactamente la brecha «plantilla sin receta», que hoy vale cero para DO y que este par vuelve
    verificable en producción y no sólo en la suite.
    """
    p = _DIR / f"dish_registry_{str(country).lower()}_v1.json"
    if not p.exists():
        return {}
    try:
        reg = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"[P1-FIDELIDAD-PLATO-DEL-REGISTRY] registry {country} ilegible: {e!r}")
        return {}
    idx = {}
    for t in reg.get("templates") or []:
        tid = t.get("template_id")
        nom = ((t.get("editorial") or {}).get("display_name") or {}).get("es") or t.get("name")
        if tid and nom:
            idx[_norm(nom)] = tid
    return idx


def dish_provenance(days, country: str = "DO") -> dict:
    """Cuántos platos servidos vienen del catálogo. Fail-open: ante cualquier fallo, ceros y `None`.

    `tasa` es `aplicables / total` —el rendimiento real de la costura—, no `del_registry / total`:
    prometer el número optimista es cómo un informe acaba afirmando algo que no ocurrió.
    """
    vacio = {"total": 0, "del_registry": 0, "con_receta": 0, "aplicables": 0, "tasa": None}
    try:
        idx_reg = _registry_name_index(country)
        idx_rec = _name_index(country)
        if not idx_reg:
            return vacio
        total = del_reg = con_rec = aplica = 0
        for d in (days or []):
            for m in ((d.get("meals") or []) if isinstance(d, dict) else []):
                if not isinstance(m, dict):
                    continue
                nombre = _norm(m.get("name"))
                if not nombre:
                    continue
                total += 1
                tid = idx_reg.get(nombre)
                if not tid:
                    continue
                del_reg += 1
                if idx_rec.get(nombre) != tid:  # la plantilla existe pero no tiene pasos escritos
                    continue
                con_rec += 1
                esperados = _foods_de_plantilla(tid, country)
                if esperados is not None and _foods_de_comida(m) == esperados:
                    aplica += 1
        return {"total": total, "del_registry": del_reg, "con_receta": con_rec,
                "aplicables": aplica, "tasa": (round(aplica / total, 3) if total else None)}
    except Exception as e:
        logger.debug(f"[P1-FIDELIDAD-PLATO-DEL-REGISTRY] procedencia no medida: {e!r}")
        return vacio
