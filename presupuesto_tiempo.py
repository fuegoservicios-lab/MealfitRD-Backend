# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-611 · 2026-09-27] El grano que pone el presupuesto cabe en el tiempo del formulario.

Corpus de baterías (421 planes): las 27 sustituciones «quinoa → Arroz integral» del ajuste de presupuesto cayeron TODAS
en usuarios de «30 min» (y una de «Nada»). El prompt de 30 min les dice lo contrario («el arroz integral, 40-45 min:
usa arroz blanco, quinoa o cuscús», lote 580); el modelo obedece con quinoa y el pase de presupuesto la cambia por el
grano que no cabe. El paso seguía con el hervor de la quinoa («cocina arroz integral en agua hirviendo durante 12-15
minutos»: son 35-45) y una merienda dulce («Yogurt natural con fresas y quinoa tostada») salía «…y arroz integral
tostado».

Los dos pases que abaratan (el estático y el de los ítems caros de la lista) consultan `candidato` por comida:
  · el tope del formulario admite el arroz integral (1 hora, sin límite) ⇒ Arroz integral, como antes;
  · 30 min ⇒ Arroz blanco (15-20 min); con diabetes, prediabetes o SOP, ninguno: el arroz blanco es el almidón de IG
    alto que el motor no escala en ese perfil, y la quinoa (IG bajo, 12-15 min) se queda;
  · «Nada» (10 min) ⇒ ninguno: ni el arroz blanco cabe;
  · un plato dulce ⇒ ninguno: el arroz no sustituye a la quinoa de un yogur;
  · harina, quinoa inflada o en hojuelas no se hierven: la regla del tiempo no aplica.
Hecho el cambio, `tiempo_en_pasos` pasa el hervor del paso al tiempo del grano nuevo (`pasos_cantidades._HERVOR_394`).
tooltip-anchor: P1-PLAN-LOTE-611
"""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_INTEGRAL = "arroz integral"
_BLANCO = "Arroz blanco"
_MIN_INTEGRAL = 45                  # el arroz integral: 35-45 min de hervor
_MIN_BLANCO = 20                    # el arroz blanco: 15-20
_NO_HIERVE = re.compile(r"harina|inflad|hojuela|pop\b|crocant|galleta")
_COCCION = re.compile(r"\ben agua\b|\bhierv\w*|\bherv\w*|\bcuec\w*|\bcuece\b|\bcocin\w*|\bsancoch\w*|\ba fuego\b")
_MINUTOS = re.compile(r"(?P<n>\d+(?:\s*[-–]\s*\d+)?)(?P<u>\s*(?:minutos|min)\b)")


def _sa(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s or "")) if unicodedata.category(c) != "Mn").lower()


def _tope(form_data):
    """Minutos por comida del formulario (`horizon._COOKING_TIME_BUDGET_MIN`); None = sin tope o desconocido."""
    try:
        from horizon import _COOKING_TIME_BUDGET_MIN
        return _COOKING_TIME_BUDGET_MIN.get(str((form_data or {}).get("cookingTime") or "").strip().lower())
    except Exception:                                                          # noqa: BLE001
        return None


def _glucemico(form_data) -> bool:
    try:
        import graph_orchestrator as go
        from constants import PCOS_CONDITION_TERMS
        return bool(go._is_diabetes_condition(form_data)
                    or any(any(t in c for t in PCOS_CONDITION_TERMS) for c in go._condition_strings(form_data)))
    except Exception:                                                          # noqa: BLE001
        return True                  # fail-secure: sin saber, no se pone el almidón de IG alto


def _dulce(meal) -> bool:
    try:
        import graph_orchestrator as go
        return bool(isinstance(meal, dict) and go._is_sweet_meal(meal, getattr(go, "strip_accents", None) or _sa))
    except Exception:                                                          # noqa: BLE001
        return False


def _rechazado(nombre, form_data) -> bool:
    """El pase por ítems caros comprobó alergias y rechazos contra «Arroz integral», no contra el arroz blanco."""
    try:
        from constants import alergias_y_rechazos
        n = _sa(nombre)
        return any(t and re.search(r"\b" + re.escape(t) + r"\b", n)
                   for t in (_sa(x).strip() for x in alergias_y_rechazos(form_data)))
    except Exception:                                                          # noqa: BLE001
        return True


def candidato(linea, candidato, form_data, meal=None):
    """El sustituto económico que cabe; None = no sustituir esta línea. Sólo decide sobre «Arroz integral»."""
    try:
        if _INTEGRAL not in _sa(candidato) or _NO_HIERVE.search(_sa(linea)):
            return candidato
        if _dulce(meal):
            return None
        tope = _tope(form_data)
        if tope is None or tope >= _MIN_INTEGRAL:
            return candidato
        if tope < _MIN_BLANCO or _glucemico(form_data) or _rechazado(_BLANCO, form_data):
            return None
        return _BLANCO
    except Exception:                                                          # noqa: BLE001
        return candidato


def tiempo_en_pasos(meal, candidato) -> int:
    """En la frase que hierve el grano nuevo, el primer tiempo tras el grano pasa al suyo. Nº de frases; 0 ante error."""
    try:
        rec = meal.get("recipe") if isinstance(meal, dict) else None
        cand = _sa(candidato)
        if not isinstance(rec, list) or not cand.startswith("arroz"):
            return 0
        import pasos_cantidades as pc
        rango = next((t for k, t in pc._HERVOR_394 if cand.startswith(k)), None)
        if not rango:
            return 0
        rango = rango.replace(" min", "")
        rx = re.compile(r"\b" + re.escape(cand).replace(r"\ ", r"\s+") + r"\b", re.IGNORECASE)
        n = 0
        for i, p in enumerate(rec):
            if not isinstance(p, str) or pc._es_nota(p) or pc.nota_plantilla(p):
                continue
            trozos = re.split(r"((?<=[.;])\s+)", p)
            cambio = False
            for k in range(0, len(trozos), 2):
                c = trozos[k]
                g = rx.search(c)
                if not g or "microondas" in _sa(c) or not _COCCION.search(_sa(c)):
                    continue
                m = _MINUTOS.search(c, g.end())
                if not m or re.match(r"\s*(?:por\s+lado|m[aá]s)\b", c[m.end():]) or m.group("n").replace("–", "-") == rango:
                    continue
                trozos[k] = c[:m.start("n")] + rango + c[m.end("n"):]
                cambio = True
                n += 1
            if cambio:
                rec[i] = "".join(trozos)
        if n:
            meal["recipe"] = rec
            meal.pop("_display", None)
            logger.info(f"⏱️ [P1-PLAN-LOTE-611] hervor de «{candidato}» a {rango} min en {n} frase(s)")
        return n
    except Exception:                                                          # noqa: BLE001
        return 0


__all__ = ["candidato", "tiempo_en_pasos"]
