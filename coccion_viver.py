# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-444 · 2026-09-26] El víver que ningún paso cuece: qué paso lo cuece y DÓNDE va.

El reparador del lote 68 (`graph_orchestrator._auto_patch_uncooked_foods`) añadía siempre, antes del Montaje,
«🍠 Añade {víver} al guiso y cocínalo 15-20 minutos… antes de servir» (con «Nada» de tiempo, «🍠 Corta {víver} en cubos
pequeños (1 cm) y hiérvelos… antes de servir»). Replay de la cola sobre 322 planes: en 10 de los 17 platos con ese paso
NO había guiso —«Bowl tropical de pollo y tostones rápidos», «Mangú de plátano verde», «Avena cremosa con batata»,
«Pescado y Batata»— y el paso llegaba DESPUÉS del que ya majaba, doraba o servía el víver («dora las rodajas de plátano
verde 2 minutos por lado y aplástalas para formar tostones» → «Añade plátano verde al guiso y cocínalo 15-20 minutos»).
Además decía «cocínalo… tierno» de la auyama. Aquí: si el plato es un guiso (`_meal_is_stewy`, el criterio del
cerrador de proteína), el víver va al guiso y con su género («Añade la auyama al guiso y cocínala… tierna»); si no, una
«💡 Cocción previa» tras el Mise en place — el texto del lote 408 (`pasos_cantidades.nota_hervor_viver`, SSOT) o, sin
tiempo, los cubos pequeños del lote 220 —, de modo que lo que el plato hace después (majar, dorar, servir) lo hace con el
víver ya cocido. tooltip-anchor: P1-PLAN-LOTE-444
"""
from __future__ import annotations

import re
import unicodedata

_VIVER_444_RE = re.compile(r"\b(platano|yuca|yautia|batata|papa|name|mapuey|auyama|guineo|guineito)s?\b")
_DURO_444 = ("yuca", "name", "yautia", "malanga", "mapuey", "platano verde", "guineo verde")
#: El corte con que el Mise en place ya deja el víver: la cocción previa hierve ESOS trozos, no «el plátano» entero.
_CORTE_444 = r"(rodajas|laminas|tiras|trozos|bastones|cubos|cubitos|dados)"
_CORTE_FEM_444 = ("rodajas", "laminas", "tiras")


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def _corte(meal: dict, n: str):
    """«corta ½ plátano verde en rodajas», «las rodajas de plátano verde» en el Mise en place → «rodajas»; o `None`."""
    nucleo = re.escape((_sa(n).split() or [""])[0])
    for p in meal.get("recipe") or []:
        t = _sa(p)
        if not t.lstrip().startswith("mise en place"):
            continue
        m = (re.search(r"\b" + nucleo + r"\w*(?:\s+[^\s.;,]+){0,4}?\s+en\s+" + _CORTE_444 + r"\b", t)
             or re.search(r"\b" + _CORTE_444 + r"\s+de\s+(?:[^\s.;,]+\s+){0,2}?" + nucleo, t))
        if m:
            return m.group(1)
    return None


def paso_viver_sin_coccion(meal: dict, food: str, sin_tiempo: bool, guiso: bool) -> tuple:
    """(paso, previa) para el víver `food` (nombre del catálogo) que ningún paso de `meal` cuece. `guiso`: lo decide
    `graph_orchestrator._meal_is_stewy` (el MISMO criterio con el que el cerrador de proteína escribe «al guiso»).
    `previa=True`: el paso va tras el Mise en place (`tras_la_mise`); `False`: antes del Montaje."""
    import pasos_cerrador as pc
    n = str(food or "").strip()
    n = n[:1].lower() + n[1:]
    art = pc._art(n)
    if guiso and not sin_tiempo:                     # con «Nada» de tiempo, jamás 15-20 minutos (lote 220)
        return (f"🍠 Añade {art} {n} al guiso y cocínal{pc._suf(n)} 15-20 minutos, hasta que {pc._este(n)} "
                f"{pc._adj(n, 'tiern')} por dentro, antes de servir.", False)
    t = "10-12" if any(x in _sa(n) for x in _DURO_444) else "8-10"
    agua = " y desecha el agua de cocción (cruda no se come)" if "yuca" in _sa(n) else ""
    c = _corte(meal, n)
    if c:
        fem = c in _CORTE_FEM_444
        return (f"💡 Cocción previa: hierve {'las' if fem else 'los'} {'láminas' if c == 'laminas' else c} de {n} en agua "
                f"{t} minutos, hasta que estén {'tiernas' if fem else 'tiernos'}; escúrrel{'as' if fem else 'os'}{agua}.",
                True)
    if sin_tiempo:
        return (f"💡 Cocción previa: corta {art} {n} en cubos pequeños (1 cm) y hiérvelos {t} minutos, hasta que estén "
                f"tiernos; escúrrelos{agua}.", True)
    import pasos_cantidades as pq
    k = _VIVER_444_RE.search(_sa(n))
    return pq.nota_hervor_viver(k.group(1) if k else None, f"{art} {n}", "10-15 min" if "maduro" in _sa(n) else ""), True


def tras_la_mise(recipe: list, paso: str) -> list:
    """`recipe` con `paso` justo después del Mise en place (como el 408); sin Mise, antes del primer Toque de Fuego;
    sin ninguno, al principio."""
    rec = list(recipe or [])
    i = next((j for j, s in enumerate(rec) if isinstance(s, str) and _sa(s).lstrip().startswith("mise en place")), None)
    if i is None:
        i = next((j - 1 for j, s in enumerate(rec) if isinstance(s, str)
                  and _sa(s).lstrip().startswith("el toque de fuego")), -1)
    return rec[:i + 1] + [paso] + rec[i + 1:]
