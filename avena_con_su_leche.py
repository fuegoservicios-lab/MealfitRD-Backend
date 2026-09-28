# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-587 · 2026-09-27] La avena se cocina con la leche que la espesa; el resto se sirve en un vaso.

El espejo del lote 311 (a la avena le faltaba líquido): el motor de macros usa la leche como fuente de proteína y la sube
mientras encoge la avena. Batería real (estudiante): «Avena cremosa de remolacha…» con 30 g de avena y 575 ml de leche
descremada, «cocina 7-9 minutos removiendo hasta que espese» — con 19 ml por gramo no espesa: es una sopa. Corpus: 12 de 410
avenas cocidas por encima de 14 ml/g (15 g con 680 ml; 15 g con 430 ml), cinco en las baterías de esta semana.

Los macros y la compra están bien (la leche se toma entera); lo imposible es la receta. Así que la leche de la lista NO se
toca: el paso que cocina la avena dice cuánta usar (8 ml por gramo, mínimo 150 ml) y el Montaje sirve el resto en un vaso
aparte. Sólo avena COCIDA con leche (las mismas reglas del 311: ni remojada, ni batido, ni masa) y sólo si el resto llega a
100 ml. tooltip-anchor: P1-PLAN-LOTE-587
"""
from __future__ import annotations

import re

RATIO_MAX = 14.0         # ml de leche por g de avena: por encima, la avena no espesa
RATIO_COCCION = 8.0      # con cuánta se cocina (cremosa)
COCCION_MIN_ML = 150.0
RESTO_MIN_ML = 100.0
_VASO_RE = re.compile(r"\bvaso\b[^.]{0,40}\bleche\b|\bleche\b[^.]{0,40}\bvaso\b")


def _montaje(rec: list):
    for i, p in enumerate(rec):
        if isinstance(p, str) and p.lstrip().lower().startswith("montaje"):
            return i
    return None


def separar(meal) -> int:
    """ml de leche que pasan al vaso (0 si no aplica o ante cualquier error). Muta sólo `recipe`."""
    try:
        import avena_liquido as al
        if not al.activo() or not isinstance(meal, dict):
            return 0
        ings, rec = meal.get("ingredients"), meal.get("recipe")
        if not isinstance(ings, list) or not isinstance(rec, list) or not rec:
            return 0
        pasos = [p for p in rec if isinstance(p, str) and not any(e in p for e in al._NOTA)]
        texto = al._sa(" ".join(pasos) + " " + str(meal.get("name") or ""))
        if not al._AVENA_RE.search(texto) or al._FRIO_RE.search(texto) or al._NOMBRE_BATIDO_RE.search(al._sa(meal.get("name"))):
            return 0
        if _VASO_RE.search(texto):
            return 0
        avena_g = leche_ml = 0.0
        for ln in ings:
            m = al._LINEA_RE.match(str(ln))
            if not m:
                continue
            v, u, food = al._num(m.group("q")), al._sa(m.group("u")), al._sa(m.group("food"))
            if v is None:
                continue
            if al._AVENA_RE.search(food) and not al._NO_HOJUELA_RE.search(food):
                hint = al._HINT_G_RE.search(str(ln))
                if u in ("g", "gr", "gramos", "gramo"):
                    avena_g += v
                elif hint:
                    avena_g += float(hint.group(1).replace(",", "."))
                elif u in ("taza", "tazas"):
                    avena_g += v * al.G_POR_TAZA_AVENA
            elif (re.search(r"\bleche\b", food) and not al._NO_LIQUIDO_RE.search(food) and u in al.ML):
                leche_ml += v * al.ML[u]
        if avena_g <= 0 or leche_ml <= RATIO_MAX * avena_g:
            return 0
        coccion = max(COCCION_MIN_ML, round(avena_g * RATIO_COCCION / 10.0) * 10.0)
        resto = leche_ml - coccion
        if resto < RESTO_MIN_ML:
            return 0
        # el paso que COCINA la avena dice con cuánta leche
        hecho = False
        for i, p in enumerate(rec):
            if not isinstance(p, str) or any(e in p for e in al._NOTA) or not al._COCCION_RE.search(al._sa(p)):
                continue
            base = al._sa(p)
            if len(base) != len(p):
                continue
            mq = re.search(r"\b\d+(?:[.,]\d+)?\s*ml\s+de\s+(?:la\s+)?leche\b", base)
            if mq:
                rec[i] = p[:mq.start()] + f"{int(coccion)} ml de la leche" + p[mq.end():]
                hecho = True
                break
            ml = al._LECHE_NOMBRADA_428_RE.search(base)
            if ml:
                rec[i] = p[:ml.start()] + f"{int(coccion)} ml de " + p[ml.start():]
                hecho = True
                break
        if not hecho:
            return 0
        iv = _montaje(rec)
        vaso = f"Sirve los {int(round(resto))} ml de leche restantes en un vaso aparte."
        if iv is None:
            rec.append(f"Montaje: {vaso}")
        else:
            s = rec[iv].rstrip()
            rec[iv] = s + ("" if s.endswith((".", "!", "?")) else ".") + " " + vaso
        meal["recipe"] = rec
        meal["_avena_leche_vaso"] = int(round(resto))
        meal.pop("_display", None)
        return int(round(resto))
    except Exception:
        return 0


# [P1-PLAN-LOTE-635 · 2026-09-28] El vaso sigue a la lista. Validación del 592 (estudiante, día 2): «Avena cremosa con
# mango, maní y queso cottage» con «275 ml de leche descremada» en la lista y, en los pasos, «mide… 360 ml», «cocina la
# avena con 150 ml de la leche» y «Sirve los 210 ml de leche restantes» (150 + 210 = 360: la lista de la PRIMERA pasada).
# El shield reescala la lista y vuelve a correr la cola, pero `separar` ya ve el vaso y no hace nada, y el 308 no toca una
# leche que tres pasos citan con tres cifras. Corpus posterior al 587: 1 de 1 vaso descuadrado. Aquí, si el vaso del 587
# está, se rehace desde la lista: vaso = lista − cocción (y la cifra vieja del total, a la de la lista); si el resto ya no
# llega a 100 ml, el vaso se va y la avena se cocina con toda la leche. tooltip-anchor: P1-PLAN-LOTE-635
_VASO_587_RE = re.compile(r"\s*Sirve los (?P<n>\d+) ml de leche restantes en un vaso aparte\.")
_COCCION_587_RE = re.compile(r"\b(?P<n>\d+) ml de la leche\b")


def _leche_de_la_lista(meal) -> float:
    import avena_liquido as al
    total = 0.0
    for ln in meal.get("ingredients") or []:
        m = al._LINEA_RE.match(str(ln))
        if not m:
            continue
        v, u, food = al._num(m.group("q")), al._sa(m.group("u")), al._sa(m.group("food"))
        if v is not None and re.search(r"\bleche\b", food) and not al._NO_LIQUIDO_RE.search(food) and u in al.ML:
            total += v * al.ML[u]
    return total


def resincronizar(meal) -> int:
    """[P1-PLAN-LOTE-635] Nº de pasos reescritos para que cocción + vaso = la leche de la lista; 0 si no aplica."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        if not isinstance(rec, list):
            return 0
        iv = next((i for i, p in enumerate(rec) if isinstance(p, str) and _VASO_587_RE.search(p)), None)
        if iv is None:
            return 0
        vaso = int(_VASO_587_RE.search(rec[iv]).group("n"))
        ic = next((i for i, p in enumerate(rec) if isinstance(p, str) and i != iv and _COCCION_587_RE.search(p)), None)
        lista = _leche_de_la_lista(meal)
        if ic is None or lista <= 0:
            return 0
        coccion = int(_COCCION_587_RE.search(rec[ic]).group("n"))
        total_viejo, total = coccion + vaso, int(round(lista))
        if abs(total - total_viejo) <= 2:
            return 0
        resto = total - coccion
        antes = list(rec)
        if resto >= RESTO_MIN_ML:
            rec[iv] = _VASO_587_RE.sub(lambda m: f" Sirve los {resto} ml de leche restantes en un vaso aparte.", rec[iv], 1)
            meal["_avena_leche_vaso"] = resto
        else:                                   # ya no sobra un vaso: toda la leche a la olla
            rec[iv] = _VASO_587_RE.sub("", rec[iv], 1).rstrip()
            rec[ic] = _COCCION_587_RE.sub(f"{total} ml de la leche", rec[ic], 1)
            meal.pop("_avena_leche_vaso", None)
        for i, p in enumerate(rec):             # la cifra vieja del total (el Mise en place), a la de la lista
            if isinstance(p, str) and i not in (iv, ic):
                rec[i] = re.sub(rf"\b{total_viejo}(\s*ml de (?:la )?leche)\b", lambda m: f"{total}{m.group(1)}", p)
        n = sum(1 for a, b in zip(antes, rec) if a != b)
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
        return n
    except Exception:
        return 0


__all__ = ["separar", "resincronizar"]
