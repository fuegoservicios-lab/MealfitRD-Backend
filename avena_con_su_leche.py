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


__all__ = ["separar"]
