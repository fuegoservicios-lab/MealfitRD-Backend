# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-191 · 2026-09-24] Lo que un paso MIDE tiene que estar en la lista.

Batería rd18: la cena «Pescado blanco guisado con tomate y harina de Negrito» decía «mide 20 g de harina de Negrito…
disuelve la harina en agua… incorpora poco a poco» y la lista no la traía: la compra salía sin harina. Medido sobre las
921 comidas guardadas (rd2–rd18): 11 comidas con un paso que MIDE un alimento ausente de la lista — dos harinas de
Negrito, «30 g de queso blanco fresco», «10 g de nueces», «30 g de pan rallado», «80 g de yogur» en un plato llamado
«Yogur con…». Y no es sólo la compra: la guarda de alérgenos y el revisor leen la LISTA, no los pasos; un alérgeno que
sólo aparece en un paso no lo veía nadie (0 casos reales en esas corridas, pero ninguna defensa si ocurre).

Se añade a la lista (y a `ingredients_raw`) lo que un paso mide con cantidad y unidad («N g/ml/taza/cda/cdta de X»)
cuando: X resuelve en el catálogo, ninguna palabra con contenido de X está ya en la lista, el paso no es una nota
(seguridad, clínica, sustitución), X no es agua/sal/hielo y pesa 3–300 g. Tope de 2 por comida. En la generación corre
ANTES de las guardas (la sustitución de alérgenos y la guarda ven la línea como cualquier otra); en las superficies de
actualización la guarda clínica escanea una COPIA. El catálogo sólo se carga si hay un candidato (≈1 % de las comidas:
construir el índice compila ~700 patrones). Knob `MEALFIT_LISTA_DESDE_PASOS`.
tooltip-anchor: P1-PLAN-LOTE-191-LISTA-DESDE-PASOS
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_CANT = r"(?P<q>\d+(?:[.,]\d+)?(?:\s*[½¼¾⅓⅔])?|[½¼¾⅓⅔])"
_UNID = r"(?P<u>g|gr|gramos|ml|tazas?|cdas?|cdtas?|cucharadas?|cucharaditas?)"
_CORTE = r"(?:y|o|u|en|con|para|al|a|hasta|sobre|por|del|que|sin|como|cada)"
_RX = re.compile(r"(?<![\w/.,])" + _CANT + r"\s*" + _UNID + r"\.?\s+de\s+(?P<f>[^\W\d_]+(?:\s+(?!" + _CORTE + r"\b)"
                 r"[^\W\d_]+){0,3})", re.IGNORECASE)
_NOTA = ("Seguridad alimentaria", "Nota", "nota clínica", "se reemplaz", "Sustitución", "Ajuste clínico",
         "🛡", "⚠", "💡", "🌱")
_NO = {"agua", "hielo", "sal", "pulpa", "jugo", "zumo", "caldo", "cada", "mezcla", "masa", "preparacion", "salsa",
       "liquido", "aceite"}
# Palabras que no identifican un alimento: el color, el estado, el corte. Sin ellas «queso blanco fresco» es «queso».
_VACIAS = {"blanco", "blanca", "blancos", "blancas", "fresco", "fresca", "frescos", "frescas", "rojo", "roja", "rojas",
           "rojos", "verde", "verdes", "maduro", "madura", "entero", "entera", "enteros", "enteras", "natural",
           "integral", "integrales", "molido", "molida", "picado", "picada", "picados", "picadas", "cocido", "cocida",
           "cocidos", "cocidas", "crudo", "cruda", "seco", "seca", "secos", "secas", "fino", "fina", "finos", "finas",
           "pasteurizado", "pasteurizada", "rallado", "rallada", "tostado", "tostada", "tostados", "precocida",
           "precocido", "light", "descremada", "descremado", "bajo", "baja", "grasa", "sodio", "azucar", "sin"}


def enabled() -> bool:
    try:
        from knobs import _env_bool
        return _env_bool("MEALFIT_LISTA_DESDE_PASOS", True)
    except Exception:                                                          # noqa: BLE001
        return True


def _sa(s) -> str:
    try:
        from constants import strip_accents
        return strip_accents(str(s).lower())
    except Exception:                                                          # noqa: BLE001
        return str(s).lower()


def _con_contenido(frase: str) -> list:
    return [w for w in re.findall(r"[a-zñ]{4,}", _sa(frase)) if w not in _VACIAS]


def _en_la_lista(frase: str, lista_sa: str) -> bool:
    """¿Alguna palabra con contenido de la frase ya está en la lista (singular o plural)? Conservador a propósito:
    «harina de Negrito» con «harina de maíz» en la lista cuenta como presente — añadir de más es peor que no añadir."""
    for w in _con_contenido(frase):
        variantes = {w, w[:-1] if w.endswith("s") else w, w[:-2] if w.endswith("es") else w}
        for v in variantes:
            if len(v) >= 4 and re.search(r"\b" + re.escape(v) + r"(?:s|es)?\b", lista_sa):
                return True
    return False


def _candidatos(meal) -> list:
    """(cantidad, unidad, palabras) de lo que los pasos MIDEN y la lista no trae. Sin catálogo: es el filtro barato."""
    if not isinstance(meal, dict):
        return []
    ings, rec = meal.get("ingredients"), meal.get("recipe")
    if not isinstance(ings, list) or not ings or not isinstance(rec, list):
        return []
    lista_sa = _sa(" ".join(str(x) for x in ings))
    out, vistos = [], set()
    for paso in rec:
        if not isinstance(paso, str) or any(t in paso for t in _NOTA):
            continue
        for m in _RX.finditer(paso):
            palabras = m.group("f").split()
            if not palabras or _sa(palabras[0]) in _NO:
                continue
            clave = _sa(" ".join(palabras))
            if clave in vistos or _en_la_lista(" ".join(palabras), lista_sa):
                continue
            vistos.add(clave)
            out.append((m.group("q").replace(" ", ""), m.group("u"), palabras))
    return out


def _resuelve(db, cant: str, unid: str, palabras: list):
    """La frase más larga (desde el principio) que el catálogo resuelve con cantidad: «20 g de harina de Negrito»."""
    for k in range(len(palabras), 0, -1):
        alimento = " ".join(palabras[:k])
        if _sa(alimento) in _NO:
            return None
        linea = f"{cant} {unid} de {alimento}"
        try:
            mac = db.macros_from_ingredient_string(linea)
        except Exception:                                                      # noqa: BLE001
            mac = None
        if mac and (mac.get("grams") or 0) > 0:
            return linea, mac
    return None


def _db_o_nueva(db):
    if db is not None:
        return db
    from nutrition_db import IngredientNutritionDB
    return IngredientNutritionDB()


def reconciliar_meal(meal: dict, db=None) -> list:
    """Añade a `meal` (lista y raw) lo que sus pasos miden y la lista no trae. Devuelve las líneas añadidas."""
    cands = _candidatos(meal)
    if not cands:
        return []
    db = _db_o_nueva(db)
    nuevas = []
    for cant, unid, palabras in cands:
        if len(nuevas) >= 2:
            break
        if nuevas and _en_la_lista(" ".join(palabras), _sa(" ".join(nuevas))):
            continue
        r = _resuelve(db, cant, unid, palabras)
        if r is None:
            continue
        linea, mac = r
        if not (3 <= (mac.get("grams") or 0) <= 300):          # «0.7 g de ajo» no es compra; 300 g ya no es un paso
            continue
        nuevas.append(linea)
    if nuevas:
        meal["ingredients"] = list(meal.get("ingredients") or []) + nuevas
        raw = meal.get("ingredients_raw")
        if isinstance(raw, list):
            meal["ingredients_raw"] = list(raw) + nuevas
        meal["_lista_desde_pasos"] = list(nuevas)
        meal.pop("_display", None)
    return nuevas


def reconciliar(plan, db=None) -> int:
    """Generación: muta `plan` (lista, raw y macros de la comida). Devuelve cuántas líneas añadió. Nunca lanza."""
    try:
        if not (enabled() and isinstance(plan, dict)):
            return 0
        total = 0
        for day in plan.get("days") or []:
            for meal in (day.get("meals") or []) if isinstance(day, dict) else []:
                try:
                    if not _candidatos(meal):
                        continue
                    db = _db_o_nueva(db)
                    nuevas = reconciliar_meal(meal, db)
                    if nuevas:
                        total += len(nuevas)
                        try:
                            import graph_orchestrator as go
                            go._truth_up_meal_macros_from_strings(meal, db)
                        except Exception:                                      # noqa: BLE001
                            pass
                        logger.info(f"🧾 [P1-PLAN-LOTE-191] '{str(meal.get('name'))[:44]}': el paso mide y la lista "
                                    f"no lo traía → {nuevas}")
                except Exception as e:                                         # noqa: BLE001
                    logger.debug(f"[P1-PLAN-LOTE-191] comida no-op: {type(e).__name__}: {e}")
        return total
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-191] no-op: {type(e).__name__}: {e}")
        return 0


def con_lo_que_miden_los_pasos(meal):
    """Guardas de las superficies de actualización (cambiar plato, chat, días deterministas): una COPIA de la comida
    con lo que sus pasos miden y su lista no trae, para que el escáner de alérgenos y de dieta también lo vea. No muta
    nada; sin candidatos devuelve la misma comida sin copiar ni cargar el catálogo. Si algo falla, la comida tal cual."""
    try:
        if not (enabled() and _candidatos(meal)):
            return meal
        import copy
        c = copy.deepcopy(meal)
        return c if reconciliar_meal(c) else meal
    except Exception:                                                          # noqa: BLE001
        return meal
