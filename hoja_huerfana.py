# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-543 · 2026-09-27] La hoja verde que ningún paso usa.

Batería real del 27-sep (hipotiroidismo + IMAO, con levotiroxina): «75g de espinacas» en 5 comidas por plan —un yogur con
sandía, una avena del desayuno, un casabe con lechosa— que ningún paso nombra; sólo la nota clínica («separa… espinacas…
al menos 4 horas de la levotiroxina»), con la que además chocan en el desayuno. El tope de hojas (P3-LEAF-VOLUME-CAP) las
deja en 75 g, pero no las quita. En el corpus de 4.573 comidas guardadas: 44 hojas huérfanas, 43 de ese perfil.
Aquí, tras el tope de hojas:
- en el desayuno y las meriendas (o un plato dulce) la línea sale y el plato se re-mide desde sus líneas;
- en el almuerzo y la cena se sirve: «Acompaña con las espinacas frescas.».
Una hoja que algún paso (no una nota) nombra, o que el nombre del plato promete, no se toca. tooltip-anchor: P1-PLAN-LOTE-543
"""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_NOTAS = ("⚠", "💡", "🤰", "⚕", "🧊", "🛒", "🍽", "❄", "⏱", "🌱", "🥬", "🍠")
# nombre visible por hoja (la clave es la raíz sin acentos)
_HOJAS = {"espinaca": "las espinacas", "lechuga": "la lechuga", "rucula": "la rúcula", "arugula": "la rúcula",
          "acelga": "la acelga", "berro": "los berros", "kale": "el kale"}
_HOJA_RE = re.compile(r"\b(espinacas?|lechugas?|rucula|arugula|acelgas?|berros?|kale)\b")
_DULCE_RE = re.compile(r"\b(yogur\w*|avena|batido\w*|smoothie|granola|panqueque\w*|postre|sandia|lechosa|mango|guineo|"
                       r"manzana|fresas?|pina|mandarina|chinola|pera|uvas?|melon|miel|mantequilla de mani|cereal)\b")
_SLOT_LIGERO_RE = re.compile(r"^(desayuno|merienda|snack|colaci)", re.IGNORECASE)


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def _raiz(hoja: str) -> str:
    h = _sa(hoja)
    return re.sub(r"s$", "", h) if h not in ("berros",) else "berro"


def _pasos(meal) -> str:
    return " ".join(_sa(p) for p in (meal.get("recipe") or []) if isinstance(p, str) and not p.lstrip().startswith(_NOTAS))


def _quita_raw(meal, raiz: str) -> None:
    """La pareja en `ingredients_raw`, por ALIMENTO (la primera línea con la misma hoja), nunca por índice."""
    raw = meal.get("ingredients_raw")
    if not isinstance(raw, list):
        return
    for j, r in enumerate(list(raw)):
        if isinstance(r, str) and re.search(r"\b" + raiz, _sa(r)):
            raw.pop(j)
            return


def limpiar(days, db=None) -> int:
    """Nº de líneas tratadas (quitadas o servidas); 0 ante cualquier error (fail-open)."""
    n = 0
    try:
        for day in days or []:
            for m in (day.get("meals") or []) if isinstance(day, dict) else []:
                if not isinstance(m, dict) or not isinstance(m.get("ingredients"), list):
                    continue
                pasos = _pasos(m)
                nombre = _sa(" ".join(str(m.get(k) or "") for k in ("name", "desc", "description")))
                ligero = bool(_SLOT_LIGERO_RE.match(_sa(m.get("meal") or ""))) or bool(_DULCE_RE.search(nombre))
                quitadas, servidas = [], []
                for linea in list(m["ingredients"]):
                    if not isinstance(linea, str):
                        continue
                    mm = _HOJA_RE.search(_sa(linea))
                    if not mm:
                        continue
                    raiz = _raiz(mm.group(1))
                    if re.search(r"\b" + raiz, pasos) or re.search(r"\b" + raiz, nombre):
                        continue                                       # la receta la usa o el plato la promete
                    if ligero:
                        m["ingredients"].remove(linea)
                        _quita_raw(m, raiz)
                        quitadas.append(linea)
                    else:
                        servidas.append(_HOJAS.get(raiz, "las hojas verdes"))
                if servidas:
                    frase = " Acompaña con " + " y ".join(dict.fromkeys(servidas)) + " frescas."
                    frase = frase.replace("la lechuga frescas", "la lechuga fresca").replace("la rúcula frescas", "la rúcula fresca")
                    frase = frase.replace("la acelga frescas", "la acelga fresca").replace("el kale frescas", "el kale fresco")
                    frase = frase.replace("los berros frescas", "los berros frescos")
                    rec = m.get("recipe") if isinstance(m.get("recipe"), list) else []
                    j = next((k for k, p in enumerate(rec) if isinstance(p, str) and _sa(p).lstrip().startswith("montaje")), None)
                    if j is not None:
                        rec[j] = rec[j].rstrip() + frase
                    else:
                        rec.append("Montaje:" + frase)
                    m["recipe"] = rec
                if quitadas:
                    m["_hoja_huerfana_quitada"] = quitadas
                    if db is not None:
                        try:
                            import graph_orchestrator as _go
                            _go._truth_up_meal_macros_from_strings(m, db)
                        except Exception:
                            pass
                if quitadas or servidas:
                    m.pop("_display", None)
                    n += len(quitadas) + len(servidas)
    except Exception as e:
        logger.debug(f"[P1-PLAN-LOTE-543] no-op: {type(e).__name__}: {e}")
    if n:
        logger.info(f"🥬 [P1-PLAN-LOTE-543] {n} hoja(s) verde(s) que ningún paso usaba: quitadas o servidas")
    return n


__all__ = ["limpiar"]
