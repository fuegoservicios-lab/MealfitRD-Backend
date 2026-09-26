# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-235 · 2026-09-25] Colesterol alto: las yemas del plan caben en la nota que el propio plan escribe.

La nota clínica de dislipidemia (`_CONDITION_SAFETY_CLAUSES["dyslipidemia"]`) le dice al usuario «limita las yemas a 3-4
por semana; las claras puedes usarlas libremente», y nada lo aplicaba: el generador y los cerradores de proteína usan el
huevo como proteína barata. Batería real del 25-sep (colesterol + ganar músculo): «6¼ yemas en tres días» y luego
«6½ yemas en el día 2» → dos críticos del revisor → plan matemático «Pollo y Arroz». El plan contradecía su propia nota.

Regla (determinista, idempotente): como mucho UNA yema al día y `MEALFIT_DYSLIPIDEMIA_YOLKS_PER_WEEK` (4) en cualquier
ventana de 7 días; lo que sobra pasa a claras — 2 por huevo (7,2 g de proteína contra 6,3: la proteína se mantiene, la
yema no). Los pasos que cuentan los huevos se reescriben con la misma cuenta; el raw se ajusta por ALIMENTO.
tooltip-anchor: P1-PLAN-LOTE-235-YEMAS
"""
from __future__ import annotations

import logging
import math
import os
import re
import unicodedata

logger = logging.getLogger(__name__)

_EGG_G = 50.0          # huevo sin cáscara (mismo número que `_COUNT_UNIT_WEIGHT_G["huevo"]`)
_CLARA_G = 33.0
_NUM = {"un": 1, "una": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6}
_FRAC = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3}
_LINEA_UNIDADES = re.compile(r"^\s*(\d+(?:[.,]\d+)?)?\s*([½¼¾⅓⅔])?\s*(?:huevos?)\b(?!\s+de\s+codorniz)", re.I)
_LINEA_GRAMOS = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*g(?:r|ramos)?\s+de\s+huevos?\b(?!\s+de\s+codorniz)", re.I)
_PASO_CUENTA = re.compile(r"\b(\d+|un|una|dos|tres|cuatro|cinco|seis)\s+huevos?\b", re.I)
_CLARAS_RE = re.compile(r"^\s*(\d+)\s+claras?\s+de\s+huevo\s*$", re.I)


def _knob_int(nombre, defecto):
    try:
        return int(os.environ.get(nombre, defecto))
    except Exception:
        return defecto


def activo(form_data) -> bool:
    if str(os.environ.get("MEALFIT_DYSLIPIDEMIA_YOLK_CAP", "true")).strip().lower() in ("0", "false", "off", "no"):
        return False
    try:
        from condition_rules import detect_active_rules
        return any(r.id == "dyslipidemia" for r in detect_active_rules(form_data or {}))
    except Exception:
        return False


def _sa(s):
    return unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()


def _yemas_de_linea(linea):
    """(yemas, forma) de una línea de huevo ENTERO; (0, None) si no lo es (claras, mayonesa, codorniz…)."""
    s = str(linea or "")
    if "clara" in _sa(s):
        return 0.0, None
    m = _LINEA_GRAMOS.match(s)
    if m:
        return float(m.group(1).replace(",", ".")) / _EGG_G, "g"
    m = _LINEA_UNIDADES.match(s)
    if m and (m.group(1) or m.group(2)):
        n = float(m.group(1).replace(",", ".")) if m.group(1) else 0.0
        n += _FRAC.get(m.group(2) or "", 0.0)
        return n, "u"
    return 0.0, None


def _fmt_huevos(n):
    if n <= 0:
        return None
    ent = int(round(n)) if abs(n - round(n)) < 0.05 else None
    if ent is not None:
        return f"{ent} huevo" + ("s" if ent != 1 else "")
    return f"{n:.1f}".replace(".", ",") + " huevos"


def _reescribir_pasos(meal, antes, despues_txt):
    rec = meal.get("recipe")
    if not isinstance(rec, list):
        return
    nuevo = []
    for paso in rec:
        if not isinstance(paso, str) or paso[:1] in ("⚠", "💡", "⚕", "🤰"):
            nuevo.append(paso)
            continue

        def _sub(m):
            n = _NUM.get(m.group(1).lower())
            n = n if n is not None else (int(m.group(1)) if m.group(1).isdigit() else None)
            return despues_txt if n is not None and abs(n - antes) < 0.05 else m.group(0)
        nuevo.append(_PASO_CUENTA.sub(_sub, paso))
    meal["recipe"] = nuevo


# [P1-PLAN-LOTE-325 · 2026-09-25] Los pasos siguen a la lista que este tope reescribe. Corre en la cola del escudo DESPUÉS
# del contrato final de la receta, que ya había sincronizado los pasos: batería de cierre del 25-sep (colesterol + estatina),
# «56 g de huevo» + «6 claras» → «1 huevo» + «7 claras» en la lista y el paso seguía «prepara 1 huevo y 6 claras de huevo»
# (3 de 231 comidas; las tres, de este tope). Y cuando el paso contaba los huevos, `_reescribir_pasos` dejaba dos cuentas
# de claras en la misma frase («bate 1 huevo y 2 claras y 3 claras de huevo»): se suman antes de sincronizar.
# tooltip-anchor: P1-PLAN-LOTE-325
_CLARAS_Y_CLARAS_RE = re.compile(r"\b(\d+)\s+claras?(?:\s+de\s+huevo)?\s+y\s+(\d+)\s+claras?(?:\s+de\s+huevo)?\b", re.I)


def _pasos_siguen_la_lista(meal) -> None:
    rec = meal.get("recipe")
    if not isinstance(rec, list):
        return
    antes = list(rec)

    def _suma(m):
        n = int(m.group(1)) + int(m.group(2))
        return f"{n} clara{'s' if n != 1 else ''} de huevo"
    meal["recipe"] = [_CLARAS_Y_CLARAS_RE.sub(_suma, x) if isinstance(x, str) and x[:1] not in ("⚠", "💡", "⚕", "🤰")
                      else x for x in rec]
    try:
        import pasos_cantidades as pc
        pc.sincronizar_exacto(meal)
        pc.decimales_de_cocina(meal)
        pc.frases_repetidas(meal)
    except Exception:
        pass
    if meal.get("recipe") != antes:
        meal.pop("_display", None)      # DELETE-on-write: `_display[locale]` espeja los pasos


def topar_yemas(plan: dict, form_data, db=None) -> int:
    """Aplica el tope. Devuelve cuántas líneas de huevo se convirtieron. Muta `plan`. Nunca lanza."""
    try:
        if not isinstance(plan, dict) or not activo(form_data):
            return 0
        semana = max(1, _knob_int("MEALFIT_DYSLIPIDEMIA_YOLKS_PER_WEEK", 4))
        por_dia = 1.0
        historial = []   # yemas de los días ya procesados (ventana de 7)
        cambios = 0
        for day in plan.get("days") or []:
            if not isinstance(day, dict):
                continue
            usadas_hoy = 0.0
            for meal in day.get("meals") or []:
                if not isinstance(meal, dict) or not isinstance(meal.get("ingredients"), list):
                    continue
                ings = meal["ingredients"]
                nuevos = []
                tocado = False
                for linea in ings:
                    yemas, forma = _yemas_de_linea(linea)
                    if not yemas:
                        nuevos.append(linea)
                        continue
                    ventana = sum(historial[-6:])
                    cabe = max(0.0, min(por_dia - usadas_hoy, semana - ventana - usadas_hoy))
                    if yemas <= cabe + 1e-6:
                        usadas_hoy += yemas
                        nuevos.append(linea)
                        continue
                    quedan = math.floor(cabe + 1e-6)
                    exceso = yemas - quedan
                    claras = max(1, int(round(exceso * 2)))
                    txt_h = _fmt_huevos(quedan)
                    if txt_h:
                        nuevos.append(txt_h)
                    nuevos.append(f"{claras} claras de huevo")
                    usadas_hoy += quedan
                    tocado = True
                    cambios += 1
                    if forma == "u":
                        _reescribir_pasos(meal, yemas,
                                          (f"{txt_h} y {claras} claras" if txt_h else f"{claras} claras de huevo"))
                if tocado:
                    # [P1-PLAN-LOTE-242] una sola línea de claras por comida y con su número gramatical («1 clara»):
                    # la batería real dio «1 claras de huevo» + «3 claras de huevo» en el mismo plato.
                    _total_cl = 0
                    _sin_cl = []
                    for _l in nuevos:
                        _mc = _CLARAS_RE.match(str(_l))
                        if _mc:
                            _total_cl += int(_mc.group(1))
                        else:
                            _sin_cl.append(_l)
                    if _total_cl:
                        _sin_cl.append(f"{_total_cl} clara{'s' if _total_cl != 1 else ''} de huevo")
                    nuevos = _sin_cl
                    meal["ingredients"] = nuevos
                    raw = meal.get("ingredients_raw")
                    if isinstance(raw, list):
                        raw2 = []
                        for r in raw:
                            y, _f = _yemas_de_linea(r)
                            # [P1-PLAN-LOTE-242] las claras que ya traía el raw también salen: el total de abajo las
                            # incluye (contarlas dos veces compraría el doble)
                            if not y and "clara" not in _sa(r):
                                raw2.append(r)
                        yemas_meal = sum(_yemas_de_linea(x)[0] for x in nuevos)
                        claras_meal = sum(int(m.group(1)) for x in nuevos for m in [_CLARAS_RE.match(str(x))] if m)
                        if yemas_meal:
                            raw2.append(f"{int(round(yemas_meal * _EGG_G))}g de huevo")
                        if claras_meal:
                            raw2.append(f"{int(round(claras_meal * _CLARA_G))}g de clara de huevo")
                        meal["ingredients_raw"] = raw2
                    _pasos_siguen_la_lista(meal)  # [P1-PLAN-LOTE-325] «prepara 1 huevo y 6 claras» con 7 en la lista
                    if db is not None:
                        try:
                            import graph_orchestrator as go
                            go._truth_up_meal_macros_from_strings(meal, db)
                        except Exception:
                            pass
            historial.append(usadas_hoy)
        if cambios:
            plan["_yemas_topadas"] = int(plan.get("_yemas_topadas") or 0) + cambios
            logger.warning(f"🥚 [P1-PLAN-LOTE-235] colesterol alto: {cambios} línea(s) de huevo pasaron yemas a claras "
                           f"(≤1 yema/día y ≤{semana}/semana, como dice la nota del plan).")
        return cambios
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-235] tope de yemas no-op: {type(e).__name__}: {e}")
        return 0
