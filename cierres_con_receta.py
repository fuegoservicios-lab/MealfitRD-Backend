# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-48 · 2026-09-14] Lo que los cerradores añaden respeta la receta.

Cuarta prueba RD del dueño (plan 358a2cdf, el primero tras el lote 47): 11 de 12 comidas llegaron con su receta de
biblioteca, pero los cerradores de macros —que suben proteína, calorías o carbohidrato cuando una comida se queda
corta— les colgaron cosas encima: arroz junto al mofongo, batata junto a los bollitos de plátano, queso cottage licuado
dentro de un jugo de chinola. Y el queso que añaden sale como «queso» a secas: la lista lo resuelve a queso blanco,
así que el plato se llamaba «…con queso cottage» y la compra traía otro queso. Tres piezas:

  · `escalar_base_propia`: el piso de calorías de ganancia muscular ya no le cuelga una segunda base a una receta de
    biblioteca que trae la suya; ESCALA la suya (hasta ×1,5) y, si no cabe, esa comida se salta.
  · `es_jugo`: en un jugo (ácido, colado) lo que añade el cerrador va AL LADO; ni a la licuadora ni al vaso.
  · `nombrar_quesos_genericos`: el «queso» genérico toma el nombre del queso que el plato promete («con queso cottage»),
    en la lista del plato y en la de compras, antes de que el lácteo del nombre se inserte aparte y de que el barrido de
    líneas muertas lo tome por sobrante.

Knobs `MEALFIT_GAINMUSCLE_FLOOR_OWN_BASE`, `MEALFIT_CLOSER_JUICE_DAIRY_ASIDE`, `MEALFIT_GENERIC_CHEESE_FROM_NAME` (True).
tooltip-anchor: P1-PLAN-LOTE-48-CIERRES-CON-RECETA
"""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

MAX_FACTOR_BASE = 1.5          # la base del plato crece hasta la mitad; más ya es otro plato
MIN_FACTOR_BASE = 1.05         # por debajo no vale la pena tocar la receta


def _knob(nombre: str) -> bool:
    try:
        from knobs import _env_bool
        return _env_bool(nombre, True)
    except Exception:                                                          # noqa: BLE001
        return True


def base_propia_on() -> bool:
    return _knob("MEALFIT_GAINMUSCLE_FLOOR_OWN_BASE")


def _sa(s) -> str:
    return unicodedata.normalize("NFD", str(s or "").lower()).encode("ascii", "ignore").decode("ascii")


# ─────────────── el jugo ───────────────
_JUGO_RE = re.compile(r"\b(?:jugos?|limonadas?|refrescos?|agua\s+de\s+\w+)\b")
_BATIDO_RE = re.compile(r"\b(?:batid[oa]s?|licuad[oa]s?|smoothies?|malteadas?)\b")


def es_jugo(meal) -> bool:
    """¿El plato es un JUGO (ácido y colado: chinola, limón, tamarindo)? Una batida no lo es: la leche ya va en el vaso y
    el lácteo se licúa con ella. En un jugo, el queso licuado se corta."""
    if not _knob("MEALFIT_CLOSER_JUICE_DAIRY_ASIDE") or not isinstance(meal, dict):
        return False
    nombre = _sa(meal.get("name"))
    return bool(_JUGO_RE.search(nombre)) and not _BATIDO_RE.search(nombre)


# ─────────────── la base propia ───────────────
def escalar_base_propia(meal, need_kcal, kcal_room, carb_room, db=None, *, max_factor: float = MAX_FACTOR_BASE) -> tuple:
    """`(kcal, carbohidratos)` añadidos escalando la base de carbohidrato que el plato YA trae (la línea más carbohidratada
    que es una base: plátano, yuca, papa, batata…), en la lista del plato y en la de compras. `(0, 0)` si no hay base,
    no resuelve o no cabe en el techo de calorías o de carbohidrato del día."""
    if not isinstance(meal, dict) or not base_propia_on():
        return 0.0, 0.0
    ings = meal.get("ingredients")
    if not isinstance(ings, list) or not ings:
        return 0.0, 0.0
    try:
        import graph_orchestrator as go
        if db is None:
            from nutrition_db import IngredientNutritionDB
            db = IngredientNutritionDB()
        mejor = None                                   # (carbos, idx, linea, kcal)
        for i, s in enumerate(ings):
            if not isinstance(s, str) or not any(t in _sa(s) for t in go._CARB_BASE_TOKENS):
                continue
            mac = db.macros_from_ingredient_string(s) or {}
            k, c = float(mac.get("kcal") or 0), float(mac.get("carbs") or 0)
            if k > 0 and c > 0 and (mejor is None or c > mejor[0]):
                mejor = (c, i, s, k)
        if mejor is None:
            return 0.0, 0.0
        c, i, s, k = mejor
        f = min(float(max_factor), 1.0 + float(need_kcal) / k, 1.0 + max(0.0, float(kcal_room)) / k,
                1.0 + max(0.0, float(carb_room)) / c)
        if f < MIN_FACTOR_BASE:
            return 0.0, 0.0
        from nutrition_db import rescale_ingredient_string
        nueva = rescale_ingredient_string(s, f)
        if not nueva or nueva == s:
            return 0.0, 0.0
        ings[i] = nueva
        raw = meal.get("ingredients_raw")
        if isinstance(raw, list) and raw:
            _raw_n, _n = go._rescale_raw_by_food(raw, [s], [f])
            if not _n:
                # sin resolver el alimento (catálogo no cargado), la línea IDÉNTICA de la compra sí se sabe cuál es
                _raw_n = [nueva if str(r).strip() == s.strip() else r for r in raw]
                _n = int(_raw_n != raw)
            if _n:
                meal["ingredients_raw"] = _raw_n
        try:
            go._truth_up_meal_macros_from_strings(meal, db)
        except Exception:                                                      # noqa: BLE001
            pass
        meal.pop("_display", None)
        meal["_base_propia_escalada"] = round(f, 3)
        logger.info(f"🍠 [P1-PLAN-LOTE-48] «{str(meal.get('name'))[:40]}»: la base del plato ×{f:.2f} ({s!r} → {nueva!r}) "
                    f"en vez de colgarle otra")
        return k * (f - 1.0), c * (f - 1.0)
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-48] escalar la base propia no-op: {type(e).__name__}: {e}")
        return 0.0, 0.0


# ─────────────── el queso con su nombre ───────────────
# Los quesos del catálogo que pueden nombrar un plato (los mismos que `_NAME_PHANTOM_DAIRY_TOKENS`), del más específico
# al menos, con el nombre con que la lista los compra.
_QUESOS = (("queso de hoja", "queso de hoja"), ("queso crema", "queso crema"), ("queso blanco", "queso blanco"),
           ("mozzarella", "queso mozzarella"), ("cheddar", "queso cheddar"), ("ricotta", "queso ricotta"),
           ("parmesano", "queso parmesano"), ("gouda", "queso gouda"), ("cottage", "queso cottage"))
# Lo que puede seguir a «queso» sin que deje de ser genérico (una preparación, no un tipo).
_PREPARACION = frozenset(("rallado", "rallada", "desmenuzado", "desmenuzada", "picado", "picada", "en", "al", "para",
                          "rebanado", "rebanada", "cortado", "cortada", "extra", "o", "y", "con"))
_QUESO_RE = re.compile(r"(?<![a-z])(quesos?)(?![a-z])", re.IGNORECASE)


def _tipo_de(texto) -> list:
    t = _sa(texto)
    tipos, visto = [], set()
    for clave, nombre in _QUESOS:
        if re.search(r"(?<![a-z])" + re.escape(clave) + r"(?![a-z])", t) and nombre not in visto:
            visto.add(nombre)
            tipos.append(nombre)
            t = t.replace(clave, " ")          # «queso blanco» no cuenta además como «queso» genérico
    return tipos


def _generico(linea) -> bool:
    """¿La línea compra «queso» a secas? Lo que sigue a la palabra decide: nada, una preparación o un paréntesis."""
    if not isinstance(linea, str):
        return False
    t = _sa(linea)
    m = re.search(r"(?<![a-z])quesos?(?![a-z])", t)
    if not m or re.search(r"requeson", t):
        return False
    sig = t[m.end():].strip(" ,.;")
    if not sig or sig.startswith("("):
        return True
    return sig.split()[0] in _PREPARACION


def _pasos_sin_mise(meal) -> str:
    return " ".join(str(p) for p in (meal.get("recipe") or [])
                    if isinstance(p, str) and not _sa(p).startswith("mise en place") and "💪" not in p)


def nombrar_quesos_genericos(days) -> int:
    """El «queso» a secas de una comida toma el nombre del queso que el plato promete —en el NOMBRE, o si no en sus
    pasos— cuando es exactamente uno. En `ingredients` y en `ingredients_raw` (la lista de compras lee raw). Devuelve
    cuántas comidas tocó. Idempotente y fail-safe."""
    if not _knob("MEALFIT_GENERIC_CHEESE_FROM_NAME") or not isinstance(days, list):
        return 0
    tocadas = 0
    for dia in days:
        for meal in ((dia.get("meals") or []) if isinstance(dia, dict) else []):
            if not isinstance(meal, dict):
                continue
            try:
                ings = meal.get("ingredients")
                if not isinstance(ings, list) or not any(_generico(x) for x in ings):
                    continue
                tipos = _tipo_de(meal.get("name")) or _tipo_de(_pasos_sin_mise(meal))
                if len(tipos) != 1:
                    continue                    # ninguno o varios: no se adivina
                nombre = tipos[0]

                def _nombrar(linea):
                    return _QUESO_RE.sub(nombre, linea, count=1) if _generico(linea) else linea

                antes = list(ings)
                meal["ingredients"] = [_nombrar(x) for x in ings]
                raw = meal.get("ingredients_raw")
                if isinstance(raw, list):
                    meal["ingredients_raw"] = [_nombrar(x) for x in raw]
                if meal["ingredients"] != antes:
                    meal.pop("_display", None)
                    meal["_queso_nombrado"] = nombre
                    tocadas += 1
                    logger.info(f"🧀 [P1-PLAN-LOTE-48] «{str(meal.get('name'))[:40]}»: el «queso» genérico es {nombre!r} "
                                f"(la lista compra lo que el plato dice)")
            except Exception as e:                                             # noqa: BLE001
                logger.debug(f"[P1-PLAN-LOTE-48] nombrar el queso no-op en {str(meal.get('name'))[:40]}: {e!r}")
    return tocadas


# ─────────────── [P1-PLAN-LOTE-49 · 2026-09-14] lo que el cerrador añade lee el día y la receta ───────────────
# Quinta prueba RD del dueño (plan a059d7bb). Knobs `MEALFIT_CLOSER_LIGHT_SLOT_CLEAN`, `MEALFIT_CLOSER_NO_SALTCURED`,
# `MEALFIT_CLOSER_STEP_PLACEMENT` y `MEALFIT_EGGCAP_RAW_WHOLE_EGG` (True). tooltip-anchor: P1-PLAN-LOTE-49-CERRADOR-LEE
def franja_ligera_limpia_on() -> bool:
    return _knob("MEALFIT_CLOSER_LIGHT_SLOT_CLEAN")


def sin_curados_on() -> bool:
    return _knob("MEALFIT_CLOSER_NO_SALTCURED")


def _es_montaje(s) -> bool:
    return _sa(s).lstrip().startswith("montaje")


def _es_nota(s) -> bool:
    return str(s).lstrip().startswith(("⚠", "💡"))


_AL_LADO_RE = re.compile(r"^\s*sirve\b.{0,80}\bal\s+lado\b", re.IGNORECASE | re.DOTALL)


def ubicar_pasos_del_cerrador(meal) -> int:
    """El 💪 del cerrador, donde la receta lo haría (receta de biblioteca). Desde el lote 45 el rótulo «El Toque de Fuego»
    va en el PRIMER paso con fuego de la receta congelada, y la fusión de siempre (`_integrate_complement_steps`) metía el
    añadido ahí: «Añade camarones al guiso en los últimos minutos» dentro del paso que hierve la yuca, «Sirve queso cottage
    al lado» dentro del que calienta la leche (plan a059d7bb, 6 comidas). Ahora «Sirve X al lado» va al emplatado y lo
    demás, como paso propio, justo después del ÚLTIMO paso con fuego. Sin paso con fuego no toca nada (queda la fusión de
    siempre). Devuelve cuántos pasos colocó. tooltip-anchor: P1-PLAN-LOTE-49-PASO-EN-SU-SITIO"""
    if not _knob("MEALFIT_CLOSER_STEP_PLACEMENT") or not isinstance(meal, dict) or meal.get("_recipe_source") != "library":
        return 0
    rec = meal.get("recipe")
    if not isinstance(rec, list):
        return 0
    bolts = [s for s in rec if isinstance(s, str) and s.lstrip().startswith("💪")]
    if not bolts:
        return 0
    import pasos_sustitucion as ps
    resto = [s for s in rec if not (isinstance(s, str) and s.lstrip().startswith("💪"))]
    for b in bolts:
        txt = b.split("💪", 1)[1].strip()
        if not txt:
            continue
        i_mont = next((j for j, s in enumerate(resto) if isinstance(s, str) and _es_montaje(s)), None)
        if i_mont is not None and _AL_LADO_RE.match(txt):
            base = str(resto[i_mont]).rstrip()
            resto[i_mont] = base + ("" if base.endswith((".", "!", "…")) else ".") + " " + txt
            continue
        tope = i_mont if i_mont is not None else len(resto)
        i_fuego = None
        for j in range(tope):
            s = resto[j]
            if (isinstance(s, str) and not _es_nota(s) and not _sa(s).lstrip().startswith("mise en place")
                    and ps.tiene_fuego(s)):
                i_fuego = j
        if i_fuego is None:
            return 0                 # sin paso con fuego: decide la fusión de siempre (no se reescribe a medias)
        resto.insert(i_fuego + 1, txt[0].upper() + txt[1:])
    meal["recipe"] = resto
    logger.info(f"🍽️ [P1-PLAN-LOTE-49] «{str(meal.get('name'))[:40]}»: {len(bolts)} paso(s) del cerrador en su sitio")
    return len(bolts)


_HUEVO_RE = re.compile(r"\bhuevos?\b")
_PARTE_HUEVO_RE = re.compile(r"\b(?:claras?|yemas?)\b")


def indice_huevo_entero_raw(raw) -> "int | None":
    """El índice de la ÚNICA línea de huevo entero de la compra, para el tope diario de huevos cuando la línea de la lista
    y la de la compra no se emparejan por alimento («3 huevos» ↔ «165 g de huevo cocido»: el cerrador escribe «huevo
    cocido» y la lista lo humaniza a unidades). Plan a059d7bb: el desayuno mostraba «3 claras de huevo» y la compra y la
    Mise seguían con 165 g de huevo entero. `None` si hay cero o más de una: no se adivina."""
    if not _knob("MEALFIT_EGGCAP_RAW_WHOLE_EGG") or not isinstance(raw, list):
        return None
    hits = [i for i, r in enumerate(raw) if isinstance(r, str) and _HUEVO_RE.search(_sa(r))
            and not _PARTE_HUEVO_RE.search(_sa(r))]
    return hits[0] if len(hits) == 1 else None
