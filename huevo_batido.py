# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-189 · 2026-09-23] El huevo crudo de un batido se cambia por algo que ESTE usuario pueda comer.

`_substitute_blended_raw_egg` (P2-RAW-EGG-SUBSTITUTE) cambiaba el huevo crudo de un batido por «Yogurt griego sin
azúcar» sin mirar las alergias. Batería real (alergia a lácteos y mariscos): 2 de las 3 violaciones de lácteos de
rd11–rd16 («¼ taza de yogurt griego sin azúcar» en un batido) las metió ESTA pasada, no el modelo; la guarda las cazaba
y el intento se quemaba. En rd17, con la sustitución de lácteos del 188 ya viva, el yogur acababa en yogur de coco pero
la nota seguía diciendo «se reemplazó el huevo crudo del batido por yogur griego» y el revisor la rechazó como crítica.
Peor en las superficies de actualización (cambiar plato, chat): allí la guarda de alérgenos corre ANTES que esta
pasada, así que el yogur quedaba guardado en el plan de un alérgico.

Se elige el primer candidato que no choque con alergias, rechazos ni dieta: yogur griego (la conducta de siempre),
yogur de coco, tofu firme. Si ninguno sirve, no se sustituye y la nota pide huevo pasteurizado sin nombrar ningún
alimento. Lo que la maquinaria del swap escribe como «yogur griego» (pasos, nombre, descripción) pasa a decir el
sustituto real. Sin datos del usuario, la conducta de siempre. tooltip-anchor: P1-PLAN-LOTE-189-HUEVO-BATIDO
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_CANDIDATOS = ("Yogurt griego sin azúcar", "Yogur de coco", "Tofu firme")
_NOTA_SIN_SUSTITUTO = ("⚠️ Seguridad alimentaria: NO uses huevo crudo en un batido (riesgo de Salmonella). "
                       "Usa huevo PASTEURIZADO o prepáralo sin huevo.")
_YOGUR_GRIEGO = re.compile(r"\byogu?rt?\s+griego\b", re.IGNORECASE)
_MINUSCULAS = ("de", "del", "la", "el", "y", "con")


def _como(muestra: str, texto: str) -> str:
    """El sustituto con la mayúscula de lo que reemplaza: «Yogurt Griego» → «Yogur de Coco», «yogur griego» → «yogur de
    coco», «Yogur griego» → «Yogur de coco»."""
    t = texto.lower()
    palabras = muestra.split()
    if len(palabras) > 1 and palabras[0][:1].isupper() and palabras[1][:1].isupper():
        return " ".join(w if w in _MINUSCULAS else w[:1].upper() + w[1:] for w in t.split())
    return t[:1].upper() + t[1:] if muestra[:1].isupper() else t


def _renombrar(meal: dict, repl: str) -> bool:
    """Pasos, nombre y descripción: «yogur griego» → el sustituto. Los pasos se mutan EN SITIO: el caller
    (`_apply_food_safety_fixes`) guarda su referencia a la lista antes del swap y anexa la nota sobre ella."""
    cambio = False

    def _sub(s: str) -> str:
        return _YOGUR_GRIEGO.sub(lambda m: _como(m.group(0), repl), s)

    rec = meal.get("recipe")
    if isinstance(rec, list):
        nuevos = [_sub(s) if isinstance(s, str) else s for s in rec]
        if nuevos != rec:
            rec[:] = nuevos
            cambio = True
    for campo in ("name", "desc"):
        v = meal.get(campo)
        if isinstance(v, str) and _YOGUR_GRIEGO.search(v):
            meal[campo] = _sub(v)
            cambio = True
    if cambio:
        meal.pop("_display", None)
    return cambio


class Eleccion:
    """Qué hacer con el huevo crudo de un batido para ESTE usuario: `repl` (None = no sustituir), la nota tras el swap
    (`nota_sustituido`) y la nota sin swap (`nota_batido`)."""

    def __init__(self, form_data=None, allergies=None):
        import graph_orchestrator as go
        self.repl = go._BLEND_EGG_REPLACEMENT
        self.nota_batido = go._FOOD_SAFETY_NOTE_BLENDED
        self._nota_sust = go._FOOD_SAFETY_NOTE_BLENDED_SUBBED
        try:
            from constants import alergias_y_rechazos, canonicalize_diet_type
            fd = form_data if isinstance(form_data, dict) else {}
            vetos = list(dict.fromkeys(alergias_y_rechazos(fd) + alergias_y_rechazos({"allergies": list(allergies or [])})))
            dieta = canonicalize_diet_type(fd.get("dietType")) if fd.get("dietType") else None
        except Exception as e:                                                  # noqa: BLE001
            logger.debug(f"[P1-PLAN-LOTE-189] sin perfil legible, conducta de siempre: {type(e).__name__}: {e}")
            return
        if not vetos and dieta in (None, "balanced"):
            return
        try:
            libres = [c for c in _CANDIDATOS
                      if not (vetos and go._allergen_pool_item_banned(c, vetos))
                      and not (dieta and go._diet_pool_item_banned(c, dieta))]
            huevo_vetado = bool(vetos) and go._allergen_pool_item_banned("Huevo", vetos)
        except Exception as e:                                                  # noqa: BLE001
            # con alergias declaradas, la duda NO sustituye por algo que no sabemos si puede comer
            logger.warning(f"[P1-PLAN-LOTE-189] no pude comprobar los sustitutos ({type(e).__name__}): sin sustituto")
            libres, huevo_vetado = [], False
        self.repl = libres[0] if libres else None
        if self.repl == go._BLEND_EGG_REPLACEMENT and not huevo_vetado:
            return
        if self.repl != go._BLEND_EGG_REPLACEMENT:
            self.nota_batido = _NOTA_SIN_SUSTITUTO
        if self.repl:
            nombre = "yogur griego" if self.repl == go._BLEND_EGG_REPLACEMENT else self.repl.lower()
            self._nota_sust = (f"⚠️ Seguridad alimentaria: se reemplazó el huevo crudo del batido por {nombre} "
                               "(sin riesgo de Salmonella)."
                               + ("" if huevo_vetado else " Si prefieres huevo, úsalo PASTEURIZADO."))

    def nota_sustituido(self, meal) -> str:
        """Tras el swap: si el sustituto no es el yogur griego, los textos pasan a decir el sustituto real."""
        try:
            import graph_orchestrator as go
            if self.repl and self.repl != go._BLEND_EGG_REPLACEMENT and isinstance(meal, dict):
                if _renombrar(meal, self.repl):
                    logger.info(f"🥚 [P1-PLAN-LOTE-189] huevo crudo del batido → {self.repl} (alergia/dieta): "
                                f"{str(meal.get('name'))[:48]}")
        except Exception as e:                                                  # noqa: BLE001
            logger.debug(f"[P1-PLAN-LOTE-189] renombrado no-op: {type(e).__name__}: {e}")
        return self._nota_sust
