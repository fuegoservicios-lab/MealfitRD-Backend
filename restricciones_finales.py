# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-233 · 2026-09-25] La última palabra de las restricciones del formulario: alergia, dieta y rechazo.

Auditoría de solo lectura del 25-sep (escritores deterministas): el último guard de alérgenos/dieta es el escaneo del
revisor, y DESPUÉS corren cinco superficies del escudo que añaden o cambian comida —rellenos fantasma (pan integral, maní,
avena, fresas, aguacate cuando la palabra sale en el nombre o en un paso, incluso «Galletas sin maní»), el relleno de
ganar músculo, la restauración del nombre del plato, el sustituto de la compra única, el tope de pescado del embarazo—
con un contexto que hasta el lote 231 no traía lo tecleado a mano. Arreglar cada pasada deja abierta la siguiente que
alguien escriba; esto corre al FINAL del escudo (`db_plans._finalize_plan_data_for_insert`, detrás de las etiquetas) y
retira la LÍNEA que viola una restricción declarada cuando no es la identidad del plato (el añadido de un relleno).

Lo que NO hace, a propósito:
  · no toca el plato cuya identidad viola la restricción («Pescado a la plancha» a un alérgico): eso lo rechaza el revisor
    (crítico) y reintenta; quitar la línea dejaría un nombre que miente. Se registra en `_restricciones_en_el_plato`.
  · sin contexto clínico no hace nada: `{}` significa «no sé», no «sin restricciones» (doctrina de
    `build_clinical_form_from_profile`).
  · nunca deja una comida sin ingredientes.
El raw se limpia POR ALIMENTO, nunca por índice (la familia `raw[idx]`). tooltip-anchor: P1-PLAN-LOTE-233-ULTIMA-PALABRA
"""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_SENT = frozenset({"", "ninguna", "ninguno", "none", "no", "nada", "n/a"})


def _sa(s) -> str:
    return unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()


def _en_el_nombre(term, nombre) -> bool:
    """El término nombra el plato (no negado con «sin»)."""
    n, t = _sa(nombre), _sa(term).strip()
    if not t or not n:
        return False
    for m in re.finditer(r"\b" + re.escape(t) + r"(?:es|s)?\b", n):
        if n[max(0, m.start() - 4):m.start()] == "sin ":
            continue
        return True
    return False


def _nucleo_en_el_nombre(linea, nombre) -> bool:
    """Para la dieta (el escáner devuelve la CATEGORÍA, no el término): alguna palabra ≥4 letras del alimento de la línea
    nombra el plato."""
    txt = _sa(re.sub(r"^\s*[\d½¼¾⅓⅔.,/\s]*(?:g|gr|ml|kg|lb|oz|tazas?|cdas?|cdtas?|unidades?|rebanadas?|lonjas?|"
                     r"filetes?|pechugas?|porciones?)?\s*(?:de\s+)?", "", str(linea or "")))
    palabras = [w for w in re.findall(r"[a-z]+", txt)[:3] if len(w) >= 4]
    return any(_en_el_nombre(w, nombre) for w in palabras)


def _motivo(linea, nombre, alergias, dieta, rechazos, go, ctx_farmacos=None):
    mini = {"days": [{"meals": [{"name": nombre, "ingredients": [linea]}]}]}
    if alergias:
        v = go._scan_allergen_violations(mini, alergias)
        if v:
            return "alergia", str(v[0][2])
    if dieta and getattr(go, "DIET_HARD_GUARD", True):
        v = go._scan_diet_violations(mini, dieta)
        if v:
            return "dieta", str(v[0][2])
    if rechazos and getattr(go, "DISLIKE_HARD_GUARD", True):
        v = go._scan_allergen_violations(mini, [], terminos=rechazos)   # [P1-PLAN-LOTE-258] términos, no declaraciones
        if v:
            return "rechazo", str(v[0][2])
    if ctx_farmacos is not None:   # [P1-PLAN-LOTE-246] IMAO + tiramina
        import medication_rules as _mr
        if _mr.tyramine_violations(mini, ctx_farmacos):
            m = _mr._TYRAMINE_RX.search(_sa(linea)) if _mr._TYRAMINE_RX else None
            return "farmaco", (m.group(0) if m else "tiramina")
        if _mr.grapefruit_violations(mini, ctx_farmacos):   # [P1-PLAN-LOTE-251] toronja con estatina / presión
            m = _mr._GRAPEFRUIT_RX.search(_sa(linea))
            return "farmaco", (m.group(0) if m else "toronja")
    return None


def retirar_prohibidos(plan: dict, ctx: dict, db=None, surface: str = "") -> dict:
    """Retira de cada comida las líneas añadidas que violan alergia/dieta/rechazo declarados. Muta `plan`. Nunca lanza."""
    out = {"retiradas": [], "en_el_plato": []}
    try:
        if not isinstance(plan, dict) or not isinstance(ctx, dict) or not ctx:
            return out
        import graph_orchestrator as go
        alergias = [str(a).strip() for a in (ctx.get("allergies") or []) if _sa(a).strip() not in _SENT]
        dieta = ctx.get("dietType") or ctx.get("diet_type")
        rechazos = __import__("rechazos").terminos_de_rechazo(ctx)   # [P1-PLAN-LOTE-258] clase solo si la nombra
        if not (alergias or rechazos or dieta or ctx.get("medications") or ctx.get("otherMedications")):
            return out
        for day in plan.get("days") or []:
            for meal in (day.get("meals") or []) if isinstance(day, dict) else []:
                if not isinstance(meal, dict):
                    continue
                ings = meal.get("ingredients")
                if not isinstance(ings, list) or not ings:
                    continue
                nombre = meal.get("name") or ""
                keep, quitar = [], []
                for linea in ings:
                    mot = _motivo(str(linea), nombre, alergias, dieta, rechazos, go, ctx_farmacos=ctx)
                    if mot is None:
                        keep.append(linea)
                        continue
                    kind, term = mot
                    identidad = (_nucleo_en_el_nombre(linea, nombre) if kind == "dieta"
                                 else _en_el_nombre(term, nombre))
                    if identidad:
                        keep.append(linea)
                        out["en_el_plato"].append({"day": day.get("day"), "meal": meal.get("meal"), "name": nombre,
                                                   "line": str(linea), "kind": kind, "term": term})
                    else:
                        quitar.append((linea, kind, term))
                if not quitar or not keep:
                    continue
                meal["ingredients"] = keep
                raw = meal.get("ingredients_raw")
                if isinstance(raw, list) and raw:
                    meal["ingredients_raw"] = [
                        r for r in raw
                        if _motivo(str(r), nombre, alergias, dieta, rechazos, go, ctx_farmacos=ctx) is None
                        or _en_el_nombre(_motivo(str(r), nombre, alergias, dieta, rechazos, go, ctx_farmacos=ctx)[1], nombre)]
                # La nota determinista que acompañaba al añadido («🫓 Acompaña con el casabe…», «🍚 Cuece el arroz…»)
                # se va con él: un paso que nombra un alimento que ya no está es otra contradicción.
                rec = meal.get("recipe")
                if isinstance(rec, list) and rec:
                    _terms = [_sa(t) for _l, _k, t in quitar]
                    meal["recipe"] = [p for p in rec
                                      if not (isinstance(p, str) and p.strip()[:1] and not p.strip()[0].isalpha()
                                              and not p.strip().startswith(("⚠", "⚕", "🤰"))
                                              and any(re.search(r"\b" + re.escape(t) + r"(?:es|s)?\b", _sa(p))
                                                      for t in _terms))]
                # [P1-PLAN-LOTE-248 · 2026-09-25] Si lo retirado era un ALÉRGENO (o tiramina con IMAO) y un paso de la
                # receta aún lo nombra, se dice en una nota: el relleno lo había añadido PORQUE el paso lo nombraba, y la
                # instrucción que el usuario sigue es el paso. (Escanear los pasos en el revisor dio 5 de 5 falsos
                # positivos en la batería: «semillas tostadas», «claras cuajadas», «hasta lograr una crema».)
                # tooltip-anchor: P1-PLAN-LOTE-248-NOTA-DE-OMISION
                _rec248 = meal.get("recipe")
                if isinstance(_rec248, list):
                    _pasos248 = " ".join(_sa(p) for p in _rec248 if isinstance(p, str) and p.strip()[:1].isalpha())
                    for _l248, _k248, _t248 in quitar:
                        if _k248 not in ("alergia", "farmaco"):
                            continue
                        if not re.search(r"\b" + re.escape(_sa(_t248)) + r"(?:es|s)?\b", _pasos248):
                            continue
                        _nota248 = (f"⚠️ {'Alergia declarada' if _k248 == 'alergia' else 'Interacción con tu medicamento'}: "
                                    f"esta receta nombra «{_t248}» — omítelo, no lo añadas.")
                        if _nota248 not in _rec248:
                            _rec248.append(_nota248)
                if db is not None:
                    try:
                        go._truth_up_meal_macros_from_strings(meal, db)
                    except Exception:
                        pass
                for linea, kind, term in quitar:
                    out["retiradas"].append({"day": day.get("day"), "meal": meal.get("meal"), "name": nombre,
                                             "line": str(linea), "kind": kind, "term": term})
        if out["retiradas"]:
            plan["_restricciones_retiradas"] = out["retiradas"][:40]
            logger.warning(f"🧯 [P1-PLAN-LOTE-233] {surface}: {len(out['retiradas'])} línea(s) añadida(s) que violaban "
                           f"una restricción declarada, retiradas: "
                           + "; ".join(f"{r['kind']} «{r['term']}» → '{r['line'][:40]}' en {r['name'][:40]}"
                                       for r in out["retiradas"][:6]))
        if out["en_el_plato"]:
            plan["_restricciones_en_el_plato"] = out["en_el_plato"][:40]
            logger.error(f"🚨 [P1-PLAN-LOTE-233] {surface}: {len(out['en_el_plato'])} plato(s) cuya IDENTIDAD viola una "
                         f"restricción declarada (no se toca: lo rechaza el revisor): "
                         + "; ".join(f"{r['kind']} «{r['term']}» en {r['name'][:40]}" for r in out["en_el_plato"][:6]))
    except Exception as e:  # noqa: BLE001 — corre en el camino del INSERT: jamás puede tumbarlo
        logger.warning(f"[P1-PLAN-LOTE-233] última palabra de restricciones no-op: {type(e).__name__}: {e}")
    return out
